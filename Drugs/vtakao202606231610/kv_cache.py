"""KV-cached incremental forward for the ADT model (Phase 2 rollout speedup).

The ADT model is nn.TransformerEncoder(nn.TransformerEncoderLayer(norm_first=True,
activation='gelu', batch_first=True)) x n_layers + ln_final + heads. The serial /
batched rollouts re-run the FULL sequence at every generated token (O(L^2) over a
generation), which dominates rollout wall-time. This module reimplements the layer
forward with a per-layer K/V cache so each new token is O(L): process one token,
attend to the cached past, append its K/V.

It REUSES the trained layer parameters (in_proj/out_proj/norms/linears) -- no new
weights -- so it is numerically identical to the full forward (verified in __main__
to ~1e-4). Use kv_init(prompt) then kv_step(token) per generated token.
"""
import os, sys, math
sys.path.insert(0, os.path.expanduser("~/ADT/common"))
sys.path.insert(0, ".")

import torch
import torch.nn.functional as F


def _sa_cached(layer, x, k_cache, v_cache, H, hd):
    """Self-attention block (norm_first) with K/V cache. x: (B,S,d). Returns
    (x_after_residual, k_all, v_all) where k_all/v_all = (B,H,Lc+S,hd) include the
    new positions (to be stored back as the cache)."""
    B, S, d = x.shape
    xn = layer.norm1(x)
    qkv = F.linear(xn, layer.self_attn.in_proj_weight, layer.self_attn.in_proj_bias)
    q, k, v = qkv.chunk(3, dim=-1)                       # (B,S,d) each
    q = q.view(B, S, H, hd).transpose(1, 2)              # (B,H,S,hd)
    k = k.view(B, S, H, hd).transpose(1, 2)
    v = v.view(B, S, H, hd).transpose(1, 2)
    if k_cache is not None:
        k = torch.cat([k_cache, k], dim=2)              # (B,H,Lc+S,hd)
        v = torch.cat([v_cache, v], dim=2)
    # Use the SAME fused kernel as nn.MultiheadAttention (F.scaled_dot_product_attention,
    # which applies the 1/sqrt(hd) scaling) so the numerics match bit-for-bit. Prompt
    # path (S>1, no cache => Lk==S): is_causal=True = standard causal. Step path (S==1,
    # Lk>1): is_causal=False = the single new query attends to ALL cached keys (= causal
    # for the last position).
    out = F.scaled_dot_product_attention(q, k, v, is_causal=(S > 1))  # (B,H,S,hd)
    out = out.transpose(1, 2).reshape(B, S, d)
    out = layer.self_attn.out_proj(out)
    return x + out, k, v


def _ff_cached(layer, x):
    """Feed-forward block (norm_first, gelu, eval => no dropout)."""
    xn = layer.norm2(x)
    return x + layer.linear2(F.gelu(layer.linear1(xn)))


def _layer_cached(layer, x, k_cache, v_cache, H, hd):
    x, k, v = _sa_cached(layer, x, k_cache, v_cache, H, hd)
    x = _ff_cached(layer, x)
    return x, k, v


def kv_init(model, emb):
    """Process the prompt embeddings (B,Lp,d) building a fresh cache. Returns
    (caches, h) where caches=[[k,v],...] per layer and h=ln_final output (B,Lp,d)."""
    H = model.n_heads
    hd = model.d_model // H
    x = emb
    caches = []
    for layer in model.transformer.layers:
        x, k, v = _layer_cached(layer, x, None, None, H, hd)
        caches.append([k, v])
    return caches, model.ln_final(x)


def kv_step(model, emb_new, caches):
    """Process one new token's embedding (B,1,d), updating caches in place. Returns
    (h, caches) with h = ln_final output (B,1,d) for that position."""
    H = model.n_heads
    hd = model.d_model // H
    x = emb_new
    for i, layer in enumerate(model.transformer.layers):
        x, k, v = _layer_cached(layer, x, caches[i][0], caches[i][1], H, hd)
        caches[i][0] = k
        caches[i][1] = v
    return model.ln_final(x), caches


# ----------------------------------------------------------------------------- #
#  numerical equivalence self-test: cached forward vs the model's full forward
# ----------------------------------------------------------------------------- #
def _selftest():
    import argparse
    from adt_model import build_model
    from adt_dataset import FrameSampler
    from rollout_batched import rollout_batch_kv
    from train import N_SLOTS
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--frame_cache", required=True)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(args.ckpt, map_location=device)
    model = build_model(ckpt["config"]).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    fs = FrameSampler.load(args.frame_cache)

    # a real token sequence
    torch.manual_seed(0)
    tokens, n_frame, *_ = rollout_batch_kv(model, fs, device, 1, max_steps=40)[0]
    L = len(tokens)
    vals = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)
    slots = (torch.arange(L, device=device) % N_SLOTS).unsqueeze(0)
    grp = (torch.arange(L, device=device) // N_SLOTS) * N_SLOTS
    acts = vals[0, grp].unsqueeze(0)

    with torch.no_grad():
        emb = model.embed_input(vals, slots, acts)                 # (1,L,d)
        # full forward
        causal = torch.nn.Transformer.generate_square_subsequent_mask(L, device=device)
        h_full = model.ln_final(model.transformer(emb, mask=causal))  # (1,L,d)
        logits_full, _ = model(vals, slots, acts)

        # cached: prompt = first n_frame tokens, then step the rest
        caches, h_pr = kv_init(model, emb[:, :n_frame])            # (1,n_frame,d)
        h_list = [h_pr]
        for p in range(n_frame, L):
            hp, caches = kv_step(model, emb[:, p:p + 1], caches)
            h_list.append(hp)
        h_cached = torch.cat(h_list, dim=1)                        # (1,L,d)

    dh = (h_cached - h_full).abs()
    print(f"L={L} n_frame={n_frame}")
    print(f"hidden max|diff|={dh.max().item():.2e}  mean|diff|={dh.mean().item():.2e}  "
          f"(|h|~{h_full.abs().mean().item():.2f})")
    # check the action logits at every position match (heads are linear in h)
    la_full = logits_full[0]                                       # (1,L,vocab)
    la_cached = model.head_action(h_cached)
    dla = (la_cached - la_full).abs()
    print(f"action-logit max|diff|={dla.max().item():.2e}")
    ok = dh.max().item() < 1e-3 and dla.max().item() < 1e-3
    print(f"NUMERICALLY EQUIVALENT: {'YES' if ok else 'NO -- INVESTIGATE'}")


if __name__ == "__main__":
    _selftest()
