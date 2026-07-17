"""torsion_ikt.py — IKT as a TORSION corrector.

Why torsions and not coordinates (measured, 2026-07-12/13):
  * the relaxation displacement is 0.47 A but bond lengths move only 0.013 A and 1-3 distances 0.027 A --
    it is almost pure torsion. Regressing coordinates therefore chases an unpredictable, XTP-irrelevant
    quantity, and indeed a coordinate-output IKT could not even fit its training set.
  * 97% of the "declared but not realized" bonds are RING CLOSURES (LINK): the ring simply never closed
    (median 1.64 A past the bonding threshold). Closing a declared ring by rotating torsions IS inverse
    kinematics -- the thing the IKT was named for.
  * ADT's own parametrisation already exposes exactly this degree of freedom: an atom's local frame has
    x = (parent -> self), so a rotation about that x-axis is the torsion of the whole subtree below it,
    and it CANNOT change bond lengths or bond angles. One scalar per atom, native to the token stream.

Model: bidirectional transformer over the ADT token stream (LINK tokens included as context) -> one
scalar dtheta per atom, near-identity (zero) init. Forward kinematics is differentiable, so the loss is
written on the resulting geometry:

    L = sum_(i,j) in LINK   (d_ij - d0_ij)^2        close the declared rings / keep the closed ones closed
      + sum_nonbonded       relu(d_min - d_ij)^2    un-stab the tail
      + lam * sum_k dtheta_k^2                      stay near the ADT conformer

No xTB in the loss, no teacher coordinates -> it is defined for EVERY molecule, including the failures
(the fatal gap of the previous design, whose teacher existed only for successes).
"""

import numpy as np
import torch
import torch.nn as nn

from adt_model import ADTv2Model
from ikt_model import IKTModel                      # reuse build_inputs / atom_token_index

COV = {1: 0.31, 6: 0.76, 7: 0.71, 8: 0.66, 9: 0.57, 15: 1.07, 16: 1.05, 17: 1.02, 35: 1.20, 53: 1.39}
BOND_FAC = 1.3          # the XTP bonding rule: d < 1.3 * (r_i + r_j)
D_NB_MIN = 2.50         # measured: p1 of the non-bonded minimum distance in xTB-relaxed molecules (2.49 A)

# FLAT-BOTTOM bond ranges, measured on xTB-relaxed generated molecules (p0.5, p99.5). A single target
# (e.g. the covalent-radius sum) would DRAG A CLOSED AROMATIC RING (1.39 A) OUT to 1.52 A -- i.e. break
# what ADT already got right. The corrector must be silent inside the physical range and only pull back
# what is outside it.
BOND_RANGE = {(6, 6): (1.328, 1.540), (6, 8): (1.194, 1.436), (6, 7): (1.264, 1.462),
              (6, 16): (1.677, 1.895), (7, 7): (1.240, 1.439), (6, 17): (1.705, 1.865),
              (6, 9): (1.326, 1.399), (6, 35): (1.868, 2.181), (7, 8): (1.208, 1.420),
              (8, 16): (1.417, 1.756), (7, 16): (1.546, 2.215), (6, 53): (2.05, 2.20)}
DEFAULT_RANGE = (1.20, 2.00)


def bond_bounds(z1, z2):
    return BOND_RANGE.get(tuple(sorted((int(z1), int(z2)))), DEFAULT_RANGE)


class TorsionIKT(nn.Module):
    """ADT body (bidirectional) -> torsion correction per atom.

    CATEGORICAL head (n_bins over [-pi, pi)), not a scalar regression: the torsion correction is
    MULTIMODAL (gauche+/gauche-/anti). An L2 regression averages the modes, which lands on ~0 -- exactly
    what we measured: the regression head could not beat "predict zero" on held-out molecules. ADT itself
    discretises r and the HEALPix direction for the same reason.
    """

    def __init__(self, config=None, n_bins=60, max_deg=60.0, bend_bins=0, bend_deg=0.0):
        """bins span [-max_deg, +max_deg] (default +-60 deg, 2 deg per bin).

        A corrector, not a re-generator: the clamp->unclamp torsion change is small (measured median 3.2
        deg, p90 25.5 deg), so a full +-180 deg output space would waste capacity and make the search
        needlessly hard. Targets outside the window are clipped onto the edge bins."""
        super().__init__()
        self.body = ADTv2Model(dict(config or {}))
        d = self.body.d_model
        self.n_bins = n_bins
        self.max_rad = float(np.deg2rad(max_deg))
        self.head = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, n_bins))
        nn.init.zeros_(self.head[-1].weight)        # uniform logits at init
        nn.init.zeros_(self.head[-1].bias)
        self.full_circle = bool(max_deg >= 179.9)             # +-180 deg == the whole circle
        centers = -self.max_rad + (torch.arange(n_bins) + 0.5) / n_bins * 2 * self.max_rad
        self.register_buffer("bin_centers", centers)          # [-max, +max]

        # ---- optional BEND head: the bond angle at the parent, in a SMALL window ----
        # sp-hybridisation nearly fixes the bond angle, so this is a fine adjustment, not a free
        # coordinate: the window is small (+-10 deg) and the head is separate from the torsion head.
        self.bend = bool(bend_bins > 0 and bend_deg > 0)
        if self.bend:
            self.bend_bins = int(bend_bins)
            self.bend_max = float(np.deg2rad(bend_deg))
            self.bend_head = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, self.bend_bins))
            nn.init.zeros_(self.bend_head[-1].weight)
            nn.init.zeros_(self.bend_head[-1].bias)
            bc = -self.bend_max + (torch.arange(self.bend_bins) + 0.5) / self.bend_bins * 2 * self.bend_max
            self.register_buffer("bend_centers", bc)

    def bend_soft_labels(self, ang, sigma_bins=2.0):
        """Gaussian soft label over the (small, NON-circular) bend window; outside targets clip."""
        width = 2 * self.bend_max / self.bend_bins
        a = ang.clamp(-self.bend_max, self.bend_max).unsqueeze(-1)
        d = self.bend_centers.view(*([1] * (a.dim() - 1)), -1) - a
        w = torch.exp(-(d / width) ** 2 / (2 * sigma_bins ** 2))
        return w / w.sum(-1, keepdim=True).clamp(min=1e-9)

    def predict_bend(self, blogits, mode="argmax", temperature=1.0):
        if blogits is None:
            return None
        if mode == "sample":
            p = torch.softmax(blogits / temperature, dim=-1)
            idx = torch.multinomial(p.reshape(-1, p.shape[-1]), 1).reshape(p.shape[:-1])
        else:
            idx = blogits.argmax(-1)
        return self.bend_centers.to(blogits.device)[idx]

    def soft_labels(self, ang, sigma_bins=2.0):
        """Gaussian soft label over the window (ADT uses the same trick for r / HEALPix).

        With a FULL-CIRCLE window (+-180 deg) the two edge bins are NEIGHBOURS on the circle, not the
        farthest pair: -180 deg IS +180 deg. The distance must wrap, otherwise a target sitting near the
        edge splits its label mass across the two ends and the model is taught that they are opposites.
        Below full circle, targets outside the window fall onto the edge bin (they are clipped)."""
        width = 2 * self.max_rad / self.n_bins
        if self.full_circle:
            a = torch.atan2(torch.sin(ang), torch.cos(ang)).unsqueeze(-1)
            d = self.bin_centers.view(*([1] * (a.dim() - 1)), -1) - a
            d = torch.atan2(torch.sin(d), torch.cos(d))            # circular distance
        else:
            a = ang.clamp(-self.max_rad, self.max_rad).unsqueeze(-1)
            d = self.bin_centers.view(*([1] * (a.dim() - 1)), -1) - a
        w = torch.exp(-(d / width) ** 2 / (2 * sigma_bins ** 2))
        return w / w.sum(-1, keepdim=True).clamp(min=1e-9)

    def warm_start_from_adt(self, sd):
        sd2 = {f"body.{k}": v for k, v in sd.items()}
        missing, unexpected = self.load_state_dict(sd2, strict=False)
        return len(sd2)

    def forward(self, tokens_list, atom_idx_list, device, want_bend=False):
        """returns logits (B,K,n_bins) and the atom mask (B,K); want_bend -> (logits, bend_logits, mask)"""
        vals, slots, acts, pad = IKTModel.build_inputs(tokens_list, device)
        x = self.body.embed_input(vals, slots, acts)
        h = self.body.ln_final(self.body.transformer(x, mask=None, src_key_padding_mask=pad))
        B = len(tokens_list)
        K = max(max((len(ix) for ix in atom_idx_list), default=1), 1)
        gi = torch.zeros(B, K, dtype=torch.long, device=device)
        m = torch.zeros(B, K, dtype=torch.bool, device=device)
        for b, ix in enumerate(atom_idx_list):
            if ix:
                gi[b, :len(ix)] = torch.tensor(ix, dtype=torch.long, device=device)
                m[b, :len(ix)] = True
        ha = h.gather(1, gi.unsqueeze(-1).expand(-1, -1, h.shape[-1]))
        if want_bend:
            return self.head(ha), (self.bend_head(ha) if self.bend else None), m
        return self.head(ha), m                      # (B,K,n_bins)

    def predict(self, logits, mode="argmax", temperature=1.0):
        """logits -> dtheta (rad). mode: argmax (the dominant mode) or sample"""
        if mode == "sample":
            p = torch.softmax(logits / temperature, dim=-1)
            idx = torch.multinomial(p.reshape(-1, p.shape[-1]), 1).reshape(p.shape[:-1])
        else:
            idx = logits.argmax(-1)
        return self.bin_centers.to(logits.device)[idx]


def rotmat(axis, ang):
    """Rodrigues, batched: axis (...,3) unit, ang (...) -> (...,3,3)"""
    x, y, z = axis[..., 0], axis[..., 1], axis[..., 2]
    c, s = torch.cos(ang), torch.sin(ang)
    C = 1 - c
    O = torch.zeros_like(x)
    R = torch.stack([
        torch.stack([c + x * x * C, x * y * C - z * s, x * z * C + y * s], -1),
        torch.stack([y * x * C + z * s, c + y * y * C, y * z * C - x * s], -1),
        torch.stack([z * x * C - y * s, z * y * C + x * s, c + z * z * C], -1),
    ], -2)
    return R + O.unsqueeze(-1).unsqueeze(-1)


def forward_kinematics(pos, parent, order, dtheta, dbend=None):
    """Rotate each atom's subtree about its own (parent -> self) axis by dtheta (torsion), and
    optionally BEND the bond at the parent by dbend (rotation about the normal of the
    grandparent-parent-self plane).

    pos    : (K,3)  ADT coordinates (constant)
    parent : (K,)   parent index, -1 for the root(s)
    order  : list of atom indices in generation order (parents before children)
    dtheta : (K,)   with grad
    returns new_pos (K,3), differentiable in dtheta.

    Bond lengths and bond angles are untouched BY CONSTRUCTION: a rotation about the (parent->self) axis
    moves only the subtree below the atom, rigidly.
    """
    K = pos.shape[0]
    dev = pos.device
    newp = [None] * K
    M = [None] * K                                   # accumulated rotation applied to an atom's subtree
    eye = torch.eye(3, device=dev, dtype=pos.dtype)
    for k in order:
        p = int(parent[k])
        if p < 0:
            newp[k] = pos[k]
            axis = torch.tensor([1.0, 0.0, 0.0], device=dev, dtype=pos.dtype)
            M[k] = rotmat(axis, dtheta[k]) @ eye
            continue
        Mp = M[p]
        v = Mp @ (pos[k] - pos[p])                   # the bond, carried by the parent's rotation
        B = eye
        if dbend is not None:
            g = int(parent[p])
            if g >= 0:
                u = newp[p] - newp[g]
                nrm = torch.cross(u, v, dim=-1)
                s = nrm.norm()
                if float(s) > 1e-6:                  # collinear g-p-k has no defined bend plane
                    B = rotmat(nrm / s.clamp(min=1e-8), dbend[k])
                    v = B @ v                        # |v| unchanged -> the BOND LENGTH is exact
        newp[k] = newp[p] + v
        axis = v / v.norm().clamp(min=1e-8)
        # the subtree below k inherits B (rigid -> every angle inside it survives) and then the torsion
        M[k] = rotmat(axis, dtheta[k]) @ B @ Mp
    return torch.stack(newp, 0)


def geometry_loss(new_pos, anums, bonds_tree, bonds_link, nb_pairs, dtheta,
                  lam_dth=0.02, w_link=1.0, w_nb=1.0):
    """Self-supervised: close the declared rings, un-stab the contacts, stay near the ADT conformer.
    Needs no xTB and no teacher coordinates -> defined for failures too."""
    dev = new_pos.device
    L = torch.zeros((), device=dev)
    if bonds_link:
        i = torch.tensor([a for a, _ in bonds_link], device=dev)
        j = torch.tensor([b for _, b in bonds_link], device=dev)
        d = (new_pos[i] - new_pos[j]).norm(dim=-1)
        lo = torch.tensor([bond_bounds(anums[a], anums[b])[0] for a, b in bonds_link],
                          device=dev, dtype=new_pos.dtype)
        hi = torch.tensor([bond_bounds(anums[a], anums[b])[1] for a, b in bonds_link],
                          device=dev, dtype=new_pos.dtype)
        # FLAT BOTTOM: zero gradient while the declared ring closure is a physically plausible bond;
        # pull only what is outside the range (a ring that never closed sits 1.6 A past it, median).
        L = L + w_link * (torch.relu(d - hi) ** 2 + torch.relu(lo - d) ** 2).sum()
    if nb_pairs:
        i = torch.tensor([a for a, _ in nb_pairs], device=dev)
        j = torch.tensor([b for _, b in nb_pairs], device=dev)
        d = (new_pos[i] - new_pos[j]).norm(dim=-1)
        L = L + w_nb * torch.relu(D_NB_MIN - d).pow(2).sum()
    L = L + lam_dth * (dtheta ** 2).sum()
    return L


def selfmis(anums, pos, bonds0, na):
    """|declared graph  XOR  distance-perceived graph| -- the xTB-free measure of the damage."""
    ref = set()
    P = pos.detach().cpu().numpy() if torch.is_tensor(pos) else np.asarray(pos)
    for i in range(na):
        for j in range(i + 1, na):
            if np.linalg.norm(P[i] - P[j]) < BOND_FAC * (COV.get(int(anums[i]), .75) + COV.get(int(anums[j]), .75)):
                ref.add((i, j))
    b = set(bonds0)
    return len(b - ref), len(ref - b)                 # (miss = ring not closed, spur = stabbing)
