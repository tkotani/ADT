# vtakao202606231610 — emb_offset (counter-less) ADT architecture

A **self-contained, single-source** version directory, deliberately isolated from the per-machine code
drift / monkey-patching that exists elsewhere in the tree. Created 2026-06-23.

## What this version is

The emb_offset architecture: the parent pointer is stored and embedded as a **relative offset**
(`emb_offset(offset-1)`, `Embedding(max_offset, d)`) instead of the legacy **absolute** index
(`emb_pointer(from_id)`, capped by `max_pointer`). Translation-invariant → **unbounded atom count**.
The output heads were already offset in the v26 model, so only the input pathway + the loss target read
changed (loss reads the stored offset directly — no offset-of-offset double conversion).

`arch_version = vtakao202606231610`. The legacy absolute architecture (E142, `~/ADT/common`,
`geom`) is **unstamped legacy** — not this version.

## Anti-scatter rules (do NOT break these)

1. **Single edit source = t14.** Edit ONLY here. Never hand-patch the copy on another machine.
2. **Deploy by one-way rsync** to the run host (kt1):
   `rsync -a ~/ADT/Drugs/vtakao202606231610/ kt1:~/ADT/Drugs/vtakao202606231610/`
3. **This dir does not modify `~/ADT/common` or `geom`.** `adt_model.py` imports only
   `math/os/torch` — it is fully self-contained and cannot be affected by per-machine patches to the
   legacy model. The only shared deps are `common/relative_pointer.py` (offset conversion) and
   `common/adt_tokenizer.py` (token constants) — verified md5-identical on t14 and kt1 (2026-06-23).
4. **Validate on the RUN host (kt1), not t14** (t14 lacks `healpy`): `python3 smoke_test.py <E142.pt>`.
5. **Import order:** put THIS dir first on `sys.path`, then `~/ADT/common`, so the version-local
   `adt_model.py` shadows the legacy one. (See smoke_test.py header — getting this backwards silently
   loads the legacy emb_pointer model.)

## Files

- `adt_model.py` — dedicated emb_offset model (85.7M params at E142 config). Module names identical to
  the legacy model EXCEPT `emb_pointer`→`emb_offset`, so a legacy ckpt loads with `strict=False`
  (warm-start/FT): every weight transfers, only `emb_offset` is fresh-initialised.
- `smoke_test.py` — 4 checks: offset roundtrip / build / forward+loss / E142 FT-load transfer. All PASS
  on kt1 (2026-06-23).
- (next) the supervised trainer copy + launch recipe for the E142-equivalent run.

## E142-equivalent config (read from the E142 ckpt, not guessed)

`d_model=768, n_heads=8, n_layers=12, d_ff=3072, dropout=0.2, max_offset=50, n_r_bins=200,
n_atom_types=119, output_pointer_mode=offset, input_pointer_mode=offset`.

## Planned runs (kt1, 2 GPUs)

- **FT**: init from `/mnt/data1/ckpts/E142.pt` (strict=False) → emb_offset fresh → fine-tune.
- **scratch**: random init → train from scratch.
Same arch, init is the only difference — an A/B on whether the absolute→offset change is cheaply
recoverable (FT) or needs full retraining (scratch).
