"""rl_train_klrl.py -- KL-RLVR: data-free RLVR for 3D molecular generation.

Implements KLRLmethod.md (2026-06-22) exactly. One loss:

    L = L_RL(XVR)                              # REINFORCE on verifiable validity   (§2)
        + beta * squared-KL(pi || pi_ref)      # trust-region anchor + ratchet      (§3)
        + lam_s * KL(P_tgt || P_hat_theta)     # size, censored virtual dist        (§4/§7)
        + lam_c * KL(f_tgt || f_bar_theta)     # composition, element marginal      (§5)

size target = Gaussian N(mu, sigma) (§7); lam_s/lam_c = PID thermostats (§6);
diversity = scaffold_floor PID gate, OUTSIDE the loss (§8). Control is 3-layer:
the distribution-KL (autodiff) fixes the SHAPE; lam_s/lam_c/scaffold PIDs hold
the KL setpoints / diversity; and the §15 adaptive target-moment staircase
(--size_moment_ctl) steps the Gaussian target (mu,sigma) at ratchet timing to
land the EQUILIBRIUM moment on the goal, compensating the RL validity-pull
offset. (The old in-loss mean/sigma thermostats are gone -- moment control now
lives on the TARGET, not as a loss term.) No dm_log / morph / flatten paths:
this file is the canonical method only. History lives in rl_train_kr1.py.

The differentiable size/composition losses come from log_p.compute_log_p_batch
(one forward pass yields log_p AND L_size, L_spec). L_size uses the censored
at-risk-hazard survival distribution P_hat_theta (log_p.censored_size_kl).
"""
import os, sys, time, argparse, glob, re, gc
sys.path.insert(0, os.path.expanduser("~/ADT/common"))
# v2 emb_offset: THIS dir FIRST so adt_model/rollout_batched/log_p/kv_cache/relative_pointer/klrl_control
# resolve to the OFFSET-aware copies here (reward_xtb/frame/util now come from common; rl_v1+freeorder removed).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("AROMATIZE_RINGS", "1")   # aromatic-ring topology for XVR (§2); MUST be 1

import numpy as np
import torch
from rdkit import Chem, RDLogger
RDLogger.DisableLog("rdApp.*")
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem import AllChem, DataStructs, Crippen
from rdkit.Chem.QED import qed as rdkit_qed

from adt_model import build_model
from adt_dataset import FrameSampler
from ikt_model import IKTModel
from torsion_ikt import TorsionIKT, forward_kinematics
from train_torsion_online import tree_from_atoms, geom_score
from util_validation import validate_3D
from train import ALLOWED_ATOMS
from rollout_batched import rollout_batch_kv
from log_p import build_batch, compute_log_p_batch

# Full GEOM-Drugs heavy-atom composition (v21 nolimit, N~270k): C73.1 N11.3 O11.66 S2.07 F0.89 Cl0.77 Br0.19 %
GEOM_COMP = {6: 0.731, 7: 0.113, 8: 0.117, 16: 0.0207, 9: 0.0089, 17: 0.0077, 35: 0.0019}
ELEMS = [6, 7, 8, 16, 9, 17, 35]   # C N O S F Cl Br -- column order for logging


# ---------- reward: XVR = verifiable validity (§2) ----------
def reward_xvr_proxy(atoms, na):
    """Proxy XVR = RDKit mol_stable (0/1) + canonical SMILES. Single connected
    component, allowed elements, RDKit-sanitizable. (xTB-XVR path = reward_xtb.)"""
    if na is None or na < 3:
        return 0.0, ""
    try:
        anums = [atoms[k].atomic_num for k in range(na)]
        if any(a not in ALLOWED_ATOMS for a in anums):
            return 0.0, ""
        coords = [list(atoms[k].pos) for k in range(na)]
        mol, smi, info = validate_3D(anums, coords)
        if not info.get("valid", False):
            return 0.0, ""
        from rdkit.Chem import rdmolops
        if len(rdmolops.GetMolFrags(mol)) != 1:
            return 0.0, ""
        if not info.get("mol_stable", False):
            return 0.0, smi
        return 1.0, smi
    except Exception:
        return 0.0, ""


def internal_div(smis):
    """1 - mean pairwise Morgan-Tanimoto over valid SMILES (MONITOR only, never reward)."""
    fps = []
    for s in smis:
        if not s:
            continue
        m = Chem.MolFromSmiles(s)
        if m is not None:
            fps.append(AllChem.GetMorganFingerprintAsBitVect(m, 2, 2048))
    if len(fps) < 2:
        return float("nan")
    sims = [DataStructs.TanimotoSimilarity(fps[a], fps[b])
            for a in range(len(fps)) for b in range(a + 1, len(fps))]
    return 1.0 - sum(sims) / len(sims)


def qed_of(smi):
    if not smi:
        return float("nan")
    m = Chem.MolFromSmiles(smi)
    return float(rdkit_qed(m)) if m is not None else float("nan")


# ---------- deterministic Gaussian size target (§7/§13) ----------
def make_size_target(mu, sigma, nmax, device):
    """Absolute-atom-count-indexed discrete N(mu,sigma) pmf over sizes [0, nmax], sums to 1.
    Bin-integral via erfc (KLRLmethod §13/§14): p[n] = Phi(n+0.5) - Phi(n-0.5). Passed to
    compute_log_p_batch as p_size_target for the censored forward-KL (§4/§7)."""
    import math
    s2 = max(sigma, 1e-6) * (2.0 ** 0.5)
    phi = lambda x: 0.5 * (1.0 + math.erf((x - mu) / s2))     # N(mu,sigma) CDF
    p = np.array([phi(n + 0.5) - phi(n - 0.5) for n in range(nmax + 1)], dtype=np.float64)
    p[:3] = 0.0                       # sizes < 3 are impossible
    p = p / p.sum()
    return torch.tensor(p, dtype=torch.float, device=device)


# ---------- lam thermostat (one-sided leaky-integral PID, §6 eq 9) ----------
def pid_lambda(measured_kl, kappa, base, integ, kp, ki, int_max, decay, max_mult):
    """Hold the measured KL at setpoint kappa by ramping lam up when KL > kappa, relaxing
    to base when satisfied. Returns (lam, new_integ). lam in [base, base*max_mult]."""
    err = max(measured_kl - kappa, 0.0)
    integ = float(np.clip(integ * decay + err, 0.0, int_max))
    lam = base * float(np.clip(1.0 + kp * err + ki * integ, 1.0, max_mult))
    return lam, integ


def main():
    p = argparse.ArgumentParser(description="KL-RLVR canonical (KLRLmethod.md)")
    # --- core ---
    p.add_argument("--ckpt", required=True)
    p.add_argument("--allow_arch_mismatch", action="store_true", help="resume/finetune across architecture versions even when the ckpt's stamped arch_version != this code's CODE_VERSION (default: refuse). Use ONLY when you know the weights are compatible (e.g. a code edit that did not touch the model).")
    p.add_argument("--frame_cache", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--batch", type=int, default=48)
    p.add_argument("--n_train_steps", type=int, default=2000)
    p.add_argument("--max_steps_per_mol", type=int, default=150)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save_every", type=int, default=100)
    p.add_argument("--max_keep_ckpts", type=int, default=10)
    # --- reward (validity only; size/comp are LOSS terms) ---
    p.add_argument("--reward", choices=["proxy", "xtb"], default="xtb")
    p.add_argument("--xtb_bin", default=os.path.expanduser("~/xtb/bin/xtb"))
    p.add_argument("--xtb_workers", type=int, default=16)
    p.add_argument("--proxy_warmup", type=int, default=0, help="use proxy reward for the first N steps, then xTB")
    p.add_argument("--ema_baseline", type=float, default=0.9)
    p.add_argument("--size_baseline", action="store_true",
                   help="ROOT-CAUSE offset fix: per-size REINFORCE baseline (advantage = r_i - base[size_i]) so validity "
                        "reward is decorrelated from size -> removes the 'small=more valid' downward force that pulls the "
                        "size equilibrium below the L_size target. Validity is still learned WITHIN each size.")
    # --- anchor: squared-KL + ratchet (§3) ---
    p.add_argument("--beta_kl", type=float, default=0.1)
    p.add_argument("--ratchet_every", type=int, default=100, help="re-anchor pi_ref -> policy every N steps (0=off)")
    p.add_argument("--lam_rl", type=float, default=1.0, help="scale on the REINFORCE(XVR) loss; --lam_rl 0 = NO RL (size driven ONLY by L_size -> tests whether L_size alone reaches the target, isolating the validity-pull)")
    # --- size target Gaussian (§7) ---
    p.add_argument("--size_mu", type=float, default=24.8, help="Gaussian size-target mean (GEOM 24.8); raise to climb")
    p.add_argument("--size_sigma", type=float, default=5.45, help="Gaussian size-target std (GEOM 5.45)")
    p.add_argument("--size_nmax", type=int, default=56, help="upper size of target pmf & rollout ceiling; raise for climb")
    p.add_argument("--eps_size", type=float, default=1e-3, help="KL floor for L_size (§7/§9)")
    # --- size curriculum (slow climb): ramp the target mean gradually so each size is reached with good XVR
    #     before the target moves on -> the per-size baseline warms up at each level (no T2 frontier cold-start),
    #     and the up-climb proceeds in small validity-mastered steps instead of one jump. ---
    p.add_argument("--size_curriculum", action="store_true", help="ramp cur_mu from size_mu_start to size_mu gradually")
    p.add_argument("--size_mu_start", type=float, default=-1.0, help="curriculum start mean (default<0 = no ramp, hold size_mu)")
    p.add_argument("--size_mu_step", type=float, default=1.0, help="target-mean increment per advanced level")
    p.add_argument("--size_mu_gate_xvr", type=float, default=0.80, help="advance a level when smoothed XVR-frac > this AND size reached the level")
    p.add_argument("--size_mu_patience", type=int, default=40, help="advance a level after this many steps regardless (timeout)")
    # --- §15 adaptive target-moment control (equilibrium/ratchet-gated staircase): the RL validity-pull
    #     shifts the size EQUILIBRIUM below the L_size target (offset, NOT a bug -- KLRLmethod §13). Compensate
    #     by stepping the Gaussian target (mu_tgt,sigma_tgt) at ratchet timing by the deficit (gain~1; plant~1:1
    #     so ~1 cycle converges) so the equilibrium lands on (moment_mu_star, moment_sigma_star). Initial target
    #     = star + offset (empirical RL offset). Gated on settle (skip the step if size still moving). Replaces
    #     the old fixed +5 climb schedule. Mutually exclusive with --size_curriculum (both move cur_mu). ---
    p.add_argument("--size_moment_ctl", action="store_true",
                   help="§15: step the size target (mu,sigma) at ratchet timing so the EQUILIBRIUM moment lands on (moment_mu_star,moment_sigma_star), compensating the RL validity-pull offset")
    p.add_argument("--moment_mu_star", type=float, default=25.0, help="§15 GOAL equilibrium mean (G25=25; round values fine, no need to match GEOM 24.8)")
    p.add_argument("--moment_sigma_star", type=float, default=5.0, help="§15 GOAL equilibrium sigma (G25=5)")
    p.add_argument("--moment_step0", type=float, default=1.0, help="§15 INITIAL target-step size for the coarse-to-fine ladder (mu & sigma)")
    p.add_argument("--moment_step_floor", type=float, default=0.06, help="§15 ladder floor: step size never shrinks below this")
    p.add_argument("--moment_mu_off0", type=float, default=0.0, help="§15 initial target offset: mu_tgt = mu_star + off. DEFAULT 0 = start a hold AT the goal (no spurious shrink-in). For a climb, set --moment_mu_init to the native size instead.")
    p.add_argument("--moment_sigma_off0", type=float, default=0.0, help="§15 initial target sigma offset: sigma_tgt = sigma_star + off. DEFAULT 0 = start at the goal sigma. WARNING: a POSITIVE offset makes the target WIDER than natural, and forward-KL(P_tgt||P_hat) then forces the policy to widen into the size tails -> less-stable molecules -> VR drop (diagnosed 2026-06-23). Keep 0 (or negative) unless you deliberately want to widen.")
    p.add_argument("--moment_mu_init", type=float, default=-1.0, help="§15 INITIAL mu_tgt = native ckpt size (avoids a jump; climb then RAMPS it up to the goal, XVR-gated). -1 = moment_mu_star+off0 (hold default). Also the floor for the XVR step-back.")
    p.add_argument("--moment_sigma_init", type=float, default=-1.0, help="§15 INITIAL sigma_tgt (native). -1 = moment_sigma_star+off0.")
    p.add_argument("--moment_settle", type=float, default=0.3, help="§15 settle gate: step only if |Δmean_ema| and |Δsigma_ema| since the last ratchet < this (else size still moving -> hold)")
    p.add_argument("--moment_off_cap", type=float, default=8.0, help="§15 bound the target to goal±this (runaway guard: if a low-plant-gain policy reads as 'settled' below the goal, the target would otherwise climb without bound)")
    # --- lam_s PID: size-KL thermostat (§6) ---
    p.add_argument("--ikt_ckpt", default="",
                   help="RL AGAINST 'ADT + IKT' INSTEAD OF ADT ALONE. A molecule xTB rejects is handed to "
                        "the FROZEN torsion corrector, which proposes rotamers; if any of them survives the "
                        "XTP protocol, the molecule counts as a PASS and the generator is rewarded for it. "
                        "The point: the generator no longer has to get the global conformer right -- the "
                        "corrector does that -- so it is free to declare bigger topologies. The corrector is "
                        "frozen here; only the ADT learns.")
    p.add_argument("--ikt_n_cand", type=int, default=48, help="rotamer proposals per rejected molecule")
    p.add_argument("--ikt_xtb_cand", type=int, default=4, help="of those, how many are verified with xTB")
    p.add_argument("--ikt_temp", type=float, default=1.2)
    p.add_argument("--ikt_min_na", type=int, default=30,
                   help="do not spend the corrector's xTB budget below this size: small molecules pass on "
                        "their own")
    p.add_argument("--end_bias", type=float, default=0.0,
                   help="EXPLORATION device for a size CLIMB. The END logit is lowered by this amount during "
                        "rollout, so the sampler reaches sizes the current policy would essentially never "
                        "produce. That matters because L_size shapes the model's OWN end-hazards, and a "
                        "hazard at n=50 can only be shaped if a rollout ever gets to n=50. The bias does NOT "
                        "corrupt L_size: P_hat_theta is the censored at-risk-hazard survival distribution "
                        "computed from the model's own probabilities, not from the sampled size histogram. "
                        "Fade it out with --end_bias_fade and the policy has to hold the size on its own.")
    p.add_argument("--end_bias_fade", type=int, default=0,
                   help="linearly fade --end_bias to 0 over this many steps (0 = keep it constant). The "
                        "point of the climb is that the model ends up generating large molecules NATIVELY, "
                        "so the bias must go away.")
    p.add_argument("--lam_size", type=float, default=5.0, help="base weight on L_size")
    p.add_argument("--lam_size_pid", action="store_true", help="PID auto-tune lam_size to hold L_size at kappa_size")
    p.add_argument("--kappa_size", type=float, default=0.05, help="L_size setpoint for the PID")
    p.add_argument("--lam_size_kp", type=float, default=10.0)
    p.add_argument("--lam_size_ki", type=float, default=5.0)
    p.add_argument("--lam_size_int_max", type=float, default=0.3)
    p.add_argument("--lam_size_decay", type=float, default=0.95)
    p.add_argument("--lam_size_max_mult", type=float, default=4.0)
    # --- lam_c PID: composition-KL thermostat (§6) ---
    p.add_argument("--lam_comp", type=float, default=2.0, help="base weight on L_spec")
    p.add_argument("--lam_comp_pid", action=argparse.BooleanOptionalAction, default=True, help="PID auto-tune lam_comp to hold L_spec at kappa_comp (DEFAULT ON; --no-lam_comp_pid to disable)")
    p.add_argument("--kappa_comp", type=float, default=0.01, help="L_spec setpoint for the PID")
    p.add_argument("--lam_comp_kp", type=float, default=50.0)
    p.add_argument("--lam_comp_ki", type=float, default=30.0)
    p.add_argument("--lam_comp_int_max", type=float, default=0.3)
    p.add_argument("--lam_comp_decay", type=float, default=0.95)
    p.add_argument("--lam_comp_max_mult", type=float, default=4.0)
    p.add_argument("--eta_comp", type=float, default=0.1, help="relative floor eps_e=eta*f_tgt for L_spec (§5 eq 8)")
    p.add_argument("--comp_target", type=str, default="", help="DESIRED composition per element 'Z:frac,...' (e.g. '6:0.721,16:0.0307' = C 72.1%/S 3.07%); default GEOM. The composition DIAL (§5 eq 8); renormalized over the 7 elements.")
    p.add_argument("--comp_thermostat", action=argparse.BooleanOptionalAction, default=True, help="thermostat on f_tgt: at each ratchet, nudge the L_spec target ±step so the REALIZED composition lands within ±band of the desired (comp_target/GEOM). Composition analogue of §15 size moment control. DEFAULT ON (aggressive); --no-comp_thermostat to disable. 2026-06-24: aggressive tuning adopted as canonical -- reaches GEOM minors (S/F/Br).")
    p.add_argument("--comp_therm_band", type=float, default=0.05, help="comp thermostat tolerance (fraction of desired; 0.05 = ±5%): inside this band f_tgt is left alone")
    p.add_argument("--comp_therm_step", type=float, default=0.08, help="comp thermostat multiplicative step per ratchet (0.08 = ±8%, aggressive: pushes minors to GEOM faster)")
    p.add_argument("--comp_therm_cap", type=float, default=2.5, help="cap f_tgt at desired*this (2.5 lets minors like S be pushed up to 2.5x GEOM before clipping; prevents extreme overshoot)")
    p.add_argument("--xvr_floor", type=float, default=0.0, help="XVR-gate (size + comp): push the size target toward the goal AND the comp f_tgt toward GEOM ONLY while XVR_ema >= this; below it, step BOTH targets one step BACK toward their inits (native size / desired comp) so validity recovers. Protects the feedback (equilibrium is measured from valid molecules; low XVR -> unreliable -> the loop must not advance). 0=off; 0.9 keeps XVR~90%. Does NOT scale loss weights (no vf size-drift) -- only gates the target steps.")
    p.add_argument("--comp_ema", type=float, default=0.9, help="EMA decay for the logged element-fraction estimate")
    # --- scaffold_floor PID: diversity gate, OUTSIDE the loss (§8) ---
    p.add_argument("--scaffold_floor", action="store_true")
    p.add_argument("--target_scaffdiv", type=float, default=0.96, help="floor on within-window unique-Murcko fraction")
    p.add_argument("--scaff_kp", type=float, default=3.0)
    p.add_argument("--scaff_ki", type=float, default=0.3)
    p.add_argument("--scaff_kd", type=float, default=0.0)
    p.add_argument("--scaff_ema", type=float, default=0.8)
    p.add_argument("--lam_scaff_max", type=float, default=1.0)
    p.add_argument("--scaff_int_max", type=float, default=10.0)
    p.add_argument("--scaff_window", type=int, default=300)
    # --- logP ceiling PID: hold population-mean Crippen logP at/below target (mirror of scaffold_floor, reward-space) ---
    p.add_argument("--logp_ceiling", action="store_true", help="pull window-mean logP back to --target_logp when it drifts up, via a PID-tuned mean-centered reward reshaping (like scaffold_floor). Default off = no effect.")
    p.add_argument("--target_logp", type=float, default=2.8, help="ceiling/target for population-mean Crippen logP (GEOM-Drugs ~2.8)")
    p.add_argument("--logp_kp", type=float, default=0.5)
    p.add_argument("--logp_ki", type=float, default=0.05)
    p.add_argument("--logp_kd", type=float, default=0.0)
    p.add_argument("--logp_ema", type=float, default=0.8)
    p.add_argument("--lam_logp_max", type=float, default=0.5)
    p.add_argument("--logp_int_max", type=float, default=10.0)
    p.add_argument("--logp_scale", type=float, default=2.0, help="normalizer for per-mol logP deviation in the reshaping (~logP spread)")
    # --- Tanimoto ceiling PID: hold population Morgan-Tanimoto median at/below target (fingerprint diversity; L_ctrl^k, mirror of scaffold_floor) ---
    p.add_argument("--tani_ceiling", action="store_true", help="lower population Tanimoto median toward --target_tani by rewarding molecules dissimilar to a rolling FP window (PID-gated, mean-centered, base-scaled). Default off = no effect.")
    p.add_argument("--target_tani", type=float, default=0.13, help="target/ceiling for population Morgan-Tanimoto median (GEOM-Drugs ~0.122)")
    p.add_argument("--tani_kp", type=float, default=15.0)
    p.add_argument("--tani_ki", type=float, default=3.0)
    p.add_argument("--tani_kd", type=float, default=0.0)
    p.add_argument("--tani_ema", type=float, default=0.8)
    p.add_argument("--lam_tani_max", type=float, default=1.0)
    p.add_argument("--tani_int_max", type=float, default=0.3)
    p.add_argument("--tani_window", type=int, default=500)
    p.add_argument("--tani_scale", type=float, default=0.1, help="normalizer for the (small) per-mol Tanimoto deviation in the reshaping")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    _logf = open(os.path.join(args.out_dir, "train.log"), "a")
    def log(m):
        print(m, flush=True)
        _logf.write(m + "\n")
        _logf.flush()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log("=" * 80)
    log(f"KL-RLVR (KLRLmethod.md)  device={device}  time={time.strftime('%Y-%m-%d %H:%M')}")
    log(f"L = L_RL(XVR) + {args.beta_kl}*sqKL(ratchet {args.ratchet_every}) "
        f"+ lam_s*KL(N({args.size_mu},{args.size_sigma})||P_hat) + lam_c*KL(GEOM||f_hat)")
    log(f"  lam_s={args.lam_size} {'PID(kappa=%g)' % args.kappa_size if args.lam_size_pid else 'FIXED'}   "
        f"lam_c={args.lam_comp} {'PID(kappa=%g)' % args.kappa_comp if args.lam_comp_pid else 'FIXED'}   "
        f"diversity={'scaffold_floor(tgt %g)' % args.target_scaffdiv if args.scaffold_floor else 'OFF'}")
    if args.size_moment_ctl:
        log(f"  §15 moment-ctl: equilibrium goal=({args.moment_mu_star},{args.moment_sigma_star})  "
            f"init target=({(args.moment_mu_init if args.moment_mu_init >= 0 else args.moment_mu_star + args.moment_mu_off0):.1f},"
            f"{(args.moment_sigma_init if args.moment_sigma_init >= 0 else args.moment_sigma_star + args.moment_sigma_off0):.1f})"
            f"{' xvr_floor=%.2f' % args.xvr_floor if args.xvr_floor > 0 else ''}  "
            f"ladder step0={args.moment_step0}->floor{args.moment_step_floor} (binary-search) settle={args.moment_settle} off_cap={args.moment_off_cap} @ ratchet K={args.ratchet_every}")
    log(f"args = {vars(args)}")
    log("=" * 80)

    # --- model + frozen reference (KL anchor) ---
    ckpt = torch.load(args.ckpt, weights_only=False, map_location=device)
    # §17 ckpt arch-version consistency check (resume/finetune): refuse a ckpt produced by a different
    # architecture version unless --allow_arch_mismatch. CODE_VERSION reused below to stamp new saves.
    from klrl_control import check_ckpt_compat, CODE_VERSION
    _ckpt_ver = check_ckpt_compat(ckpt, allow_mismatch=args.allow_arch_mismatch)
    log(f"arch_version: ckpt={_ckpt_ver or 'UNSTAMPED (legacy -> assumed v1)'}  code={CODE_VERSION}"
        + ("  [MISMATCH ALLOWED]" if (_ckpt_ver and _ckpt_ver != CODE_VERSION) else ""))
    cfg = ckpt["config"]
    # gen-param consistency (2026-07-06): a ckpt saved with a "gen" block records the max_steps_per_mol /
    # size_nmax / max_offset it was trained with. Log it and flag if launch argv disagrees. A resume may
    # legitimately change these (e.g. a size climb) so this WARNS, never refuses; pretrain ckpts have no
    # gen block (no-op). Generation code hard-checks via klrl_control.check_gen_compat.
    if ckpt.get("gen"):
        _g = ckpt["gen"]
        _mm = [f"{k}:argv={a}!=ckpt={_g.get(k)}" for k, a in
               [("max_steps_per_mol", args.max_steps_per_mol), ("size_nmax", args.size_nmax),
                ("max_offset", cfg.get("max_offset"))] if _g.get(k) is not None and a != _g.get(k)]
        log(f"gen-params: ckpt.gen={_g}" + (f"  [WARN argv MISMATCH: {'; '.join(_mm)}]" if _mm else "  [argv matches]"))
    policy = build_model(cfg).to(device); policy.load_state_dict(ckpt["model"])
    ref = build_model(cfg).to(device); ref.load_state_dict(ckpt["model"])
    for pr in ref.parameters():
        pr.requires_grad_(False)
    ref.eval()
    opt = torch.optim.AdamW(policy.parameters(), lr=args.lr, weight_decay=0.0)
    if ckpt.get("opt") is not None:                              # 2026-07-10: restore AdamW state on resume (avoid cold-start XVR dip)
        try:
            opt.load_state_dict(ckpt["opt"]); log("[resume] AdamW optimizer state restored")
        except Exception as _e:
            log(f"[resume] opt restore failed ({str(_e)[:50]}); cold-start")
    best_xema = float(ckpt.get("best_xema", 0.0) or 0.0)         # running-max Xema -> best.pt (rotate-proof)
    frame_sampler = FrameSampler.load(args.frame_cache)
    log(f"ckpt step/epoch={ckpt.get('step', ckpt.get('epoch', '?'))}  "
        f"frames={len(frame_sampler.frames)}  cfg={cfg}")

    _ikt = None
    if args.ikt_ckpt:
        _ic = torch.load(args.ikt_ckpt, weights_only=False, map_location=device)
        _ia = _ic["args"]
        _ikt = TorsionIKT(cfg, n_bins=_ia["n_bins"], max_deg=_ia["max_deg"],
                          bend_bins=_ia.get("bend_bins", 0), bend_deg=_ia.get("bend_deg", 10.0)).to(device)
        _ikt.load_state_dict(_ic["ikt"]); _ikt.eval()
        for _q in _ikt.parameters():
            _q.requires_grad_(False)
        log(f"[IKT] FROZEN corrector in the REWARD loop: {args.ikt_ckpt} (updates={_ic.get('updates')})  "
            f"{args.ikt_n_cand} proposals -> top {args.ikt_xtb_cand} verified with xTB. The generator is "
            f"now rewarded for topologies that ADT+IKT can realise, not ones ADT alone can.")

    # --- deterministic targets (size target rebuilt when the curriculum ramps cur_mu, or §15 moment-ctl steps it) ---
    from klrl_control import initial_target
    cur_mu = args.size_mu_start if (args.size_curriculum and args.size_mu_start > 0) else args.size_mu
    cur_sigma = args.size_sigma
    if args.size_moment_ctl:   # §15: start at goal (hold) or native size (climb, via --moment_mu_init); ramp to goal
        cur_mu, cur_sigma = initial_target(args.moment_mu_star, args.moment_sigma_star,
                                           args.moment_mu_init, args.moment_sigma_init,
                                           args.moment_mu_off0, args.moment_sigma_off0)
    mu_init, sig_init = cur_mu, cur_sigma   # §15 floor for the XVR step-back (= native start)
    p_size_target = make_size_target(cur_mu, cur_sigma, args.size_nmax, device)
    comp_des = dict(GEOM_COMP)         # DESIRED composition (the goal); --comp_target overrides per element
    if args.comp_target:
        for kv in args.comp_target.split(","):
            z, fr = kv.split(":"); comp_des[int(z)] = float(fr)
    comp_tgt = dict(comp_des)          # L_spec target (f_tgt DIAL) = desired, unless --comp_thermostat moves it
    def build_f_target():              # (re)build normalized f_target tensor from comp_tgt; called again at ratchet
        ft = torch.tensor([comp_tgt[z] for z in ELEMS], dtype=torch.float, device=device)
        return ft / ft.sum()
    f_target = build_f_target()
    log("  comp desired(%): " + " ".join(f"{z}={100 * comp_des[z]:.2f}" for z in ELEMS)
        + (f"  [comp-thermostat ON ±{100 * args.comp_therm_band:.0f}%]" if args.comp_thermostat else ""))
    elem_idx = torch.tensor(ELEMS, dtype=torch.long, device=device)

    # --- state ---
    baseline = float(ckpt.get("baseline", 0.0) or 0.0)
    base_by_size = {}   # --size_baseline: per-size EMA reward baseline (decorrelate validity from size)
    history = []
    cur_lam_size, size_int = args.lam_size, 0.0
    cur_lam_comp, comp_int = args.lam_comp, 0.0
    comp_ema = {}
    size_ema = None   # EMA of done-size mean (de-noises the per-batch SEM ~0.8 so the real trend is readable)
    sigma_ema = None  # EMA of done-size std (§15 moment-ctl equilibrium-sigma estimate)
    moment_mean_last, moment_sig_last = None, None   # §15: EMA snapshot at the last ratchet (settle gate)
    mu_step_sz = args.moment_step0   # §15 coarse-to-fine ladder step for mu (monotone down: 1->0.5->0.25..floor)
    sig_step_sz = args.moment_step0  # §15 coarse-to-fine ladder step for sigma (monotone down, never coarsens back)
    curr_level_steps, xvr_ema = 0, None   # size curriculum: steps at current level + smoothed XVR for the advance gate
    scaff_win, scaff_ema, scaff_int, scaff_prev = [], None, 0.0, 0.0
    logp_ema, logp_int, logp_prev = None, 0.0, 0.0
    fp_win, tani_ema, tani_int, tani_prev = [], None, 0.0, 0.0
    cur_lam_scaff, cur_scaffdiv, cur_coll = 0.0, 1.0, 0.0
    # --- §16: restore §15 target-moment + ladder + comp_tgt from the ckpt if present, so a RESUME CONTINUES
    #     the control state instead of resetting to the init formula (saved by the checkpoint writer below).
    #     The GOAL (moment_*_star) stays from the CLI; only the current target/ladder/comp dial are restored. ---
    _ks = ckpt.get("klrl_state")
    if _ks and args.size_moment_ctl:
        cur_mu = float(_ks.get("cur_mu", cur_mu)); cur_sigma = float(_ks.get("cur_sigma", cur_sigma))
        mu_init = float(_ks.get("mu_init", mu_init)); sig_init = float(_ks.get("sig_init", sig_init))
        # NOTE: do NOT restore the 2-bisection ladder step (mu_step_sz/sig_step_sz). It is monotone-down
        # (never re-coarsens), so a ckpt whose ladder had CONVERGED (~floor 0.06, e.g. a settled hold)
        # would cripple a NEW climb to a farther goal -- the target would crawl ~0.06/ratchet and never
        # catch up. A fresh process must start the ladder COARSE (moment_step0) per §15 "climb resets the
        # ladder"; only the target POSITION (cur_mu/cur_sigma) + comp dial are continued.
        p_size_target = make_size_target(cur_mu, cur_sigma, args.size_nmax, device)
        if _ks.get("comp_tgt"):
            comp_tgt = {int(z): float(v) for z, v in _ks["comp_tgt"].items()}; f_target = build_f_target()
        log(f"  [§16 resume] restored from ckpt: target moment ({cur_mu:.2f},{cur_sigma:.2f}) + comp_tgt; "
            f"ladder RESET to coarse (step0={args.moment_step0}); goal=({args.moment_mu_star:.1f},{args.moment_sigma_star:.1f})")
    if args.size_moment_ctl and args.moment_mu_init >= 0:          # 2026-07-10 JUMP-START: override the ckpt-restored cur_mu with the CLI init
        cur_mu = mu_init = float(args.moment_mu_init)              # (skip the slow XVR-gated ratchet from the ckpt's native ~24.8). mu_init=init => no retreat below the cliff.
        p_size_target = make_size_target(cur_mu, cur_sigma, args.size_nmax, device)
        log(f"  [§16 jump-start] moment_mu_init={cur_mu:.1f} overrides ckpt cur_mu -> size target HELD at {cur_mu:.1f} (no slow ramp)")
    use_xtb = (args.reward == "xtb")
    # ALWAYS make xtb available so external control can switch XVR<->MVR live (the import is cheap; xtb only
    # runs when step_use_xtb is True). xtb_workdir is created either way.
    from reward_xtb import xvr_reward_batch
    xtb_workdir = os.path.join(args.out_dir, "xtb_work")
    os.makedirs(xtb_workdir, exist_ok=True)
    _ikt_dir = os.environ.get("IKT_SAVE_DIR")                 # IKT data collection: gen(hadd=Hprerelax)->realizable/relaxed pairs (needs BANK_STRUCT=1). off unless set.
    _ikt_buf = []; _ikt_flush = int(os.environ.get("IKT_FLUSH", "2000"))
    if _ikt_dir:
        os.makedirs(_ikt_dir, exist_ok=True)
        log(f"[IKT-collect] gen->realizable pairs -> {_ikt_dir} (flush every {_ikt_flush})")

    t0 = time.time()

    for step in range(args.n_train_steps):
        # === (1) rollout (no grad) ===
        policy.eval()
        tokens_list, n_frame_list, mols, dones, sizes = [], [], [], [], []
        _eb = args.end_bias
        if args.end_bias_fade > 0:
            _eb = args.end_bias * max(0.0, 1.0 - step / float(args.end_bias_fade))
        for tokens, n_frame, atoms, bonds, na, done in rollout_batch_kv(
                policy, frame_sampler, device, args.batch,
                max_steps=args.max_steps_per_mol, temperature=args.temperature,
                end_bias=_eb, end_bias_arr=None, size_ceiling=args.size_nmax):
            tokens_list.append(tokens); n_frame_list.append(n_frame)
            mols.append((atoms, bonds, na)); dones.append(done); sizes.append(na)

        # === (2) reward = XVR (validity ONLY; runaway -> 0) ===
        step_use_xtb = use_xtb and step >= args.proxy_warmup
        if step_use_xtb:
            rinfo = xvr_reward_batch(mols, args.xtb_bin, xtb_workdir, max_workers=args.xtb_workers)
            base = [ri["reward"] for ri in rinfo]
            smis = [ri["smi"] for ri in rinfo]
            xvr_n = sum(1 for ri in rinfo if ri["same_topo"])
            xtb_ok_n = sum(1 for ri in rinfo if ri["xtb_ok"])
            ikt_rescued = 0
            if _ikt is not None:
                import copy as _copy
                _fail = [i for i in range(len(mols))
                         if not rinfo[i].get("same_topo") and mols[i][2] >= args.ikt_min_na]
                _use, _meta = [], {}
                for i in _fail:
                    _atoms, _bonds, _na = mols[i]
                    _ix = IKTModel.atom_token_index(tokens_list[i])
                    if len(_ix) < _na:
                        continue
                    _par, _ch, _b0 = tree_from_atoms(_atoms, _bonds, _na)
                    _meta[i] = {"na": _na, "idx": _ix[:_na],
                                "pos": np.array([_atoms[k].pos for k in range(_na)], float),
                                "anum": np.array([_atoms[k].atomic_num for k in range(_na)], int),
                                "parent": _par, "b0": _b0, "order": list(range(_na))}
                    _use.append(i)
                if _use:
                    with torch.no_grad():
                        _lg, _bl, _ = _ikt([tokens_list[i] for i in _use],
                                           [_meta[i]["idx"] for i in _use], device, want_bend=True)
                        _lg = _lg.cpu(); _bl = _bl.cpu() if _bl is not None else None
                    _jobs = []
                    for _j, i in enumerate(_use):
                        m = _meta[i]; _na = m["na"]
                        _pos = torch.tensor(m["pos"], dtype=torch.float32)
                        _cands = []
                        with torch.no_grad():
                            for _c in range(args.ikt_n_cand):
                                _mode = "argmax" if _c == 0 else "sample"
                                _dth = _ikt.predict(_lg[_j:_j+1], mode=_mode, temperature=args.ikt_temp)[0]
                                _dbn = (_ikt.predict_bend(_bl[_j:_j+1], mode=_mode, temperature=args.ikt_temp)[0]
                                        if _bl is not None else None)
                                _new = forward_kinematics(_pos, m["parent"], m["order"], _dth[:_na],
                                                          dbend=(_dbn[:_na] if _dbn is not None else None)).numpy()
                                _cands.append((geom_score(m["anum"], _new, m["b0"], _na), _new))
                        _cands.sort(key=lambda t: t[0])
                        for _c, (_, _new) in enumerate(_cands[:args.ikt_xtb_cand]):
                            _at = _copy.deepcopy(mols[i][0])
                            for _k in range(_na):
                                _at[_k].pos = _new[_k].astype(np.float64)
                            _jobs.append((i, _at))
                    _r2 = xvr_reward_batch([(_at, mols[i][1], mols[i][2]) for i, _at in _jobs],
                                           args.xtb_bin, xtb_workdir, max_workers=args.xtb_workers)
                    _won = {}
                    for (i, _), _rr in zip(_jobs, _r2):
                        if _rr.get("same_topo") and (i not in _won or _rr["reward"] > _won[i]):
                            _won[i] = _rr["reward"]
                    for i, _rw in _won.items():
                        base[i] = _rw               # the generator is credited for a topology IKT realised
                        ikt_rescued += 1
                    xvr_n += ikt_rescued            # XVR now means "ADT + IKT"
            if _ikt_dir is not None:                          # IKT training pairs (BANK_STRUCT=1): gen/Hprerelax(hadd)->full-relax(relaxed)/clamp-realizable(realizable_heavy)
                for ri in rinfo:
                    if ri.get("hadd_coords") is None:
                        continue
                    _ikt_buf.append({
                        "hadd_anums": np.asarray(ri.get("hadd_anums"), np.int16),
                        "hadd_coords": np.asarray(ri["hadd_coords"], np.float32),
                        "relaxed_anums": (np.asarray(ri["relaxed_anums"], np.int16) if ri.get("relaxed_anums") is not None else None),
                        "relaxed_coords": (np.asarray(ri["relaxed_coords"], np.float32) if ri.get("relaxed_coords") is not None else None),
                        "realizable_heavy": (np.asarray(ri["realizable_heavy"], np.float32) if ri.get("realizable_heavy") is not None else None),
                        "same_topo": bool(ri.get("same_topo")), "rescued": bool(ri.get("rescued")),
                        "xtb_ok": bool(ri.get("xtb_ok")), "strain_pa": ri.get("strain_pa"), "step": step,
                    })
                if len(_ikt_buf) >= _ikt_flush:
                    _p = os.path.join(_ikt_dir, f"ikt_s{step:06d}_n{len(_ikt_buf)}.pt")
                    torch.save(_ikt_buf, _p); log(f"  [IKT-collect] flushed {len(_ikt_buf)} -> {_p}"); _ikt_buf = []
            if _ikt is not None and step % 10 == 0:
                log(f"  [IKT] rescued {ikt_rescued} of the {len(mols)-xtb_ok_n if False else '?'} rejected "
                    f"-> XVR(ADT+IKT) = {xvr_n}/{len(mols)}")
        elif os.environ.get("XVR_PFREE") == "1":
            # perception-free MVR proxy = completer n_H -> SanitizeMol (HADD mol_stable), NOT RDKit validate_3D
            from reward_pfree import pfree_mvr_batch
            rinfo = pfree_mvr_batch(mols)
            base = [ri["reward"] for ri in rinfo]
            smis = [ri["smi"] for ri in rinfo]
            xvr_n = int(sum(base)); xtb_ok_n = -1
        else:
            base, smis = [], []
            for (a, bo, n) in mols:
                r01, smi = reward_xvr_proxy(a, n); base.append(r01); smis.append(smi)
            xvr_n = int(sum(base)); xtb_ok_n = -1
        rewards = [b if d else 0.0 for b, d in zip(base, dones)]
        n_runaway = sum(1 for d in dones if not d)
        # --- validity-aware lam (§6): EMA of XVR fraction; scales size/comp KL down when XVR drops ---
        xvr_frac = xvr_n / max(args.batch, 1)
        xvr_ema = xvr_frac if xvr_ema is None else 0.9 * xvr_ema + 0.1 * xvr_frac

        # --- size metrics (done molecules only) ---
        sizes_done = [n for n, d in zip(sizes, dones) if d and n and n > 0]
        mean_size = float(np.mean(sizes_done)) if sizes_done else 0.0
        std_size = float(np.std(sizes_done)) if len(sizes_done) > 1 else 0.0
        mean_size_all = float(np.mean([n for n in sizes if n])) if any(sizes) else 0.0
        if sizes_done:
            size_ema = mean_size if size_ema is None else 0.9 * size_ema + 0.1 * mean_size
            sigma_ema = std_size if sigma_ema is None else 0.9 * sigma_ema + 0.1 * std_size
        elif args.size_moment_ctl:
            # §15 audit fix: a step with ZERO valid/done molecules leaves size_ema/sigma_ema FROZEN. A
            # frozen EMA reads as "settled" (Δ≈0) at the next ratchet -> a spurious target-moment step on
            # NO data. Invalidate the settle snapshot so the gate can only re-snapshot (never step) until
            # valid measurements resume.
            moment_mean_last = moment_sig_last = None
        # --- size curriculum: advance cur_mu when the size reached this level with good XVR (or timeout) ---
        if args.size_curriculum and not args.size_moment_ctl and cur_mu < args.size_mu:
            xf = xvr_n / max(args.batch, 1)
            xvr_ema = xf if xvr_ema is None else 0.9 * xvr_ema + 0.1 * xf
            curr_level_steps += 1
            reached = size_ema is not None and size_ema >= cur_mu - 1.0
            if (reached and xvr_ema >= args.size_mu_gate_xvr) or curr_level_steps >= args.size_mu_patience:
                cur_mu = min(cur_mu + args.size_mu_step, args.size_mu)
                p_size_target = make_size_target(cur_mu, cur_sigma, args.size_nmax, device)
                curr_level_steps = 0
                log(f"  [curriculum] size target mu -> {cur_mu:.1f} @ step {step} (xvr_ema={xvr_ema:.2f})")

        # --- composition metrics (EMA element fractions, logging only) ---
        _bc, _bt = {}, 0
        for (a, bo, n), d in zip(mols, dones):
            if d and n and n > 0:
                for k in range(n):
                    z = a[k].atomic_num
                    _bc[z] = _bc.get(z, 0) + 1; _bt += 1
        if _bt > 0:
            for z in ELEMS:
                f = _bc.get(z, 0) / _bt
                comp_ema[z] = f if comp_ema.get(z) is None else args.comp_ema * comp_ema[z] + (1.0 - args.comp_ema) * f

        # --- diversity metrics (Murcko scaffold over a rolling window) ---
        _scaf = []; _logp = []; _fp = []
        for s, d in zip(smis, dones):
            sk = ""; lp = None; fp = None
            if d and s:
                try:
                    m = Chem.MolFromSmiles(s)
                    if m is not None:
                        sk = MurckoScaffold.MurckoScaffoldSmiles(mol=m) or ""
                        lp = float(Crippen.MolLogP(m))
                        fp = AllChem.GetMorganFingerprintAsBitVect(m, 2, 2048)
                except Exception:
                    sk = ""; lp = None; fp = None
            _scaf.append(sk); _logp.append(lp); _fp.append(fp)
        _vs = [sk for sk in _scaf if sk]
        _wcnt = {}
        if _vs:
            scaff_win.extend(_vs)
            if len(scaff_win) > args.scaff_window:
                del scaff_win[:len(scaff_win) - args.scaff_window]
            for sk in scaff_win:
                _wcnt[sk] = _wcnt.get(sk, 0) + 1
            _N = len(scaff_win)
            cur_coll = sum(c * (c - 1) for c in _wcnt.values()) / (_N * (_N - 1)) if _N > 1 else 0.0
            cur_scaffdiv = len(_wcnt) / _N if _N > 0 else 1.0

        # --- scaffold_floor PID: reward rare scaffolds when div < target (§8, base-scaled, mean-centered) ---
        if args.scaffold_floor:
            scaff_ema = cur_scaffdiv if scaff_ema is None else \
                args.scaff_ema * scaff_ema + (1.0 - args.scaff_ema) * cur_scaffdiv
            _serr = args.target_scaffdiv - scaff_ema                       # >0 = below floor
            scaff_int = float(np.clip(scaff_int + _serr, 0.0, args.scaff_int_max))
            _sder = _serr - scaff_prev; scaff_prev = _serr
            cur_lam_scaff = float(np.clip(args.scaff_kp * _serr + args.scaff_ki * scaff_int + args.scaff_kd * _sder,
                                          0.0, args.lam_scaff_max))
            if cur_lam_scaff > 0.0 and _wcnt:
                _rar = [(1.0 / _wcnt[sk]) if sk and sk in _wcnt else 0.0 for sk in _scaf]
                _present = [rr for rr, sk in zip(_rar, _scaf) if sk]
                _mr = float(np.mean(_present)) if _present else 0.0
                rewards = [(r + b * cur_lam_scaff * (rr - _mr)) if sk else r
                           for r, sk, b, rr in zip(rewards, _scaf, base, _rar)]

        # --- logP ceiling PID: pull window-mean logP back to target when it drifts up (mirror of scaffold_floor) ---
        _plp = [lp for lp, sk in zip(_logp, _scaf) if sk and lp is not None]
        cur_logp = float(np.mean(_plp)) if _plp else float("nan")
        cur_lam_logp = 0.0
        if args.logp_ceiling and _plp:
            logp_ema = cur_logp if logp_ema is None else args.logp_ema * logp_ema + (1.0 - args.logp_ema) * cur_logp
            _lerr = logp_ema - args.target_logp                              # >0 = above ceiling
            logp_int = float(np.clip(logp_int + _lerr, 0.0, args.logp_int_max))
            _lder = _lerr - logp_prev; logp_prev = _lerr
            cur_lam_logp = float(np.clip(args.logp_kp * _lerr + args.logp_ki * logp_int + args.logp_kd * _lder,
                                         0.0, args.lam_logp_max))
            if cur_lam_logp > 0.0:
                rewards = [(r + b * cur_lam_logp * (cur_logp - lp) / args.logp_scale) if (sk and lp is not None) else r
                           for r, sk, b, lp in zip(rewards, _scaf, base, _logp)]

        # --- Tanimoto ceiling PID: pull population Tanimoto median down toward target (fingerprint diversity; mirror of scaffold_floor) ---
        cur_tani = float("nan"); cur_lam_tani = 0.0
        _pf = [f for f, sk in zip(_fp, _scaf) if sk and f is not None]
        if args.tani_ceiling and _pf:
            _sim = [float(np.mean(DataStructs.BulkTanimotoSimilarity(f, fp_win)))
                    if (f is not None and fp_win) else None for f in _fp]
            fp_win.extend(_pf)
            if len(fp_win) > args.tani_window:
                del fp_win[:len(fp_win) - args.tani_window]
            _pair = []
            for i in range(len(_pf) - 1):
                _pair.extend(DataStructs.BulkTanimotoSimilarity(_pf[i], _pf[i + 1:]))
            cur_tani = float(np.median(_pair)) if _pair else float("nan")
            if cur_tani == cur_tani:
                tani_ema = cur_tani if tani_ema is None else args.tani_ema * tani_ema + (1.0 - args.tani_ema) * cur_tani
                _terr = tani_ema - args.target_tani                          # >0 = too similar (above target)
                tani_int = float(np.clip(tani_int + _terr, 0.0, args.tani_int_max))
                _tder = _terr - tani_prev; tani_prev = _terr
                cur_lam_tani = float(np.clip(args.tani_kp * _terr + args.tani_ki * tani_int + args.tani_kd * _tder,
                                             0.0, args.lam_tani_max))
                if cur_lam_tani > 0.0:
                    _ps = [x for x in _sim if x is not None]
                    _ms = float(np.mean(_ps)) if _ps else 0.0
                    rewards = [(r + b * cur_lam_tani * (_ms - si) / args.tani_scale) if (sk and si is not None) else r
                               for r, sk, b, si in zip(rewards, _scaf, base, _sim)]

        # === (3) grad pass + (4) loss + backward + step ===
        # OOM-resilient: a rare runaway molecule pads the batch to a huge length and can OOM the
        # grad pass. Catch it, free the cache, and skip this batch (don't kill the whole run).
        try:
            policy.eval()
            tokens_padded, lengths, frame_lens, grad_mask = build_batch(tokens_list, n_frame_list, device)
            log_p_policy, L_size_t, L_spec_t = compute_log_p_batch(
                policy, tokens_padded, grad_mask, p_size_target, f_target, elem_idx,
                size_eps=args.eps_size, comp_eta=args.eta_comp)
            with torch.no_grad():
                log_p_ref = compute_log_p_batch(ref, tokens_padded, grad_mask)
            _Lsz, _Lsp = float(L_size_t.item()), float(L_spec_t.item())
            # loss = L_RL + anchor + lam_s*L_size + lam_c*L_spec  (eq 1)
            rewards_t = torch.tensor(rewards, dtype=torch.float, device=device)
            mean_r = rewards_t.mean().item()
            if args.size_baseline:   # per-size baseline -> size-neutral advantage (root-cause offset fix)
                bsz = torch.tensor([base_by_size.get(int(n), baseline) if n else baseline for n in sizes],
                                   dtype=torch.float, device=device)
                advantage = rewards_t - bsz
            else:
                advantage = rewards_t - baseline
            loss_rl = -(log_p_policy * advantage.detach()).mean()              # (3)
            dlp = log_p_policy - log_p_ref.detach()
            loss_kl = args.beta_kl * (dlp * dlp).mean()                        # (4) squared-KL anchor
            loss = args.lam_rl * loss_rl + loss_kl + cur_lam_size * L_size_t + cur_lam_comp * L_spec_t  # (1)
            opt.zero_grad()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), 5.0).item()
            opt.step()
        except torch.cuda.OutOfMemoryError:
            # CRITICAL: drop the autograd graph before empty_cache. After an OOM (typically in the grad
            # forward or backward), loss / log_p_policy / L_size_t ... still REFERENCE the partially-built
            # graph and its activations; empty_cache() cannot reclaim referenced memory, so without nulling
            # these the FIRST transient OOM (a runaway molecule padding the batch) spirals into a PERMANENT
            # one (next step's forward + the stale graph -> OOM again). None them, gc, empty, and log the
            # post-free allocation so we can SEE memory was actually released (24GB fits batch48 normally).
            opt.zero_grad(set_to_none=True)
            loss = loss_rl = loss_kl = dlp = None
            log_p_policy = L_size_t = L_spec_t = log_p_ref = None
            advantage = rewards_t = tokens_padded = grad_mask = None
            try:
                del tokens_list, n_frame_list
            except NameError:
                pass
            gc.collect(); torch.cuda.empty_cache()
            log(f"step {step:4d}  [OOM] batch skipped + graph freed; "
                f"gpu_alloc={torch.cuda.memory_allocated()/1e9:.2f}GB resv={torch.cuda.memory_reserved()/1e9:.2f}GB")
            continue

        # === (5) baseline + ratchet ===
        baseline = args.ema_baseline * baseline + (1.0 - args.ema_baseline) * mean_r
        if args.size_baseline:                       # per-size baseline EMA update (done molecules only)
            _bysz = {}
            for r, n, d in zip(rewards, sizes, dones):
                if d and n:
                    _bysz.setdefault(int(n), []).append(r)
            for n, rs in _bysz.items():
                m = float(np.mean(rs))
                base_by_size[n] = m if n not in base_by_size else \
                    args.ema_baseline * base_by_size[n] + (1.0 - args.ema_baseline) * m
        if args.ratchet_every > 0 and (step + 1) % args.ratchet_every == 0:
            ref.load_state_dict(policy.state_dict()); ref.eval()
            log(f"  [ratchet] pi_ref <- policy @ step {step + 1}")
            # §15: at ratchet timing, if the size has SETTLED (EMA barely moved since the last ratchet), step
            #      the target by the deficit (gain~1, plant~1:1) so the equilibrium -> (mu_star, sigma_star).
            #      The first ratchet only snapshots (let the policy first equilibrate to the initial target).
            if args.size_moment_ctl and size_ema is not None and sigma_ema is not None:
                # XVR-gate (size): the §15 ladder TRUSTS the equilibrium (size_ema), reliable only when many
                # molecules are valid. Advance toward the goal ONLY while XVR_ema >= floor; below it, do NOT
                # advance -- step mu_tgt/sigma_tgt ONE step BACK toward the native init (mu_init/sig_init) so the
                # size eases and XVR recovers (floor = init). Re-coarsen the ladder for the next climb attempt.
                size_push_ok = (args.xvr_floor <= 0.0) or (xvr_ema is None) or (xvr_ema >= args.xvr_floor)
                if size_push_ok:
                    # forward: RAMP mu_tgt/sigma_tgt toward the goal EVERY ratchet (no settle-gate, so a climb
                    # actually ramps); the binary-search step halves as the gap shrinks (fine near the goal);
                    # bounded [mu_init, goal+off_cap]. Converges mu_tgt -> goal+RL_offset (equilibrium -> goal).
                    gap_mu = args.moment_mu_star - size_ema
                    gap_sg = args.moment_sigma_star - sigma_ema
                    if abs(gap_mu) < mu_step_sz and mu_step_sz > args.moment_step_floor:
                        mu_step_sz = max(mu_step_sz * 0.5, args.moment_step_floor)
                    if abs(gap_sg) < sig_step_sz and sig_step_sz > args.moment_step_floor:
                        sig_step_sz = max(sig_step_sz * 0.5, args.moment_step_floor)
                    mv_mu = (1.0 if gap_mu >= 0 else -1.0) * min(mu_step_sz, abs(gap_mu))
                    mv_sg = (1.0 if gap_sg >= 0 else -1.0) * min(sig_step_sz, abs(gap_sg))
                    new_mu = float(np.clip(cur_mu + mv_mu, mu_init, args.moment_mu_star + args.moment_off_cap))
                    new_mu = float(np.clip(new_mu, 3.0, args.size_nmax - 1))
                    new_sg = float(np.clip(cur_sigma + mv_sg, max(1.0, args.moment_sigma_star - args.moment_off_cap),
                                           args.moment_sigma_star + args.moment_off_cap))
                    new_sg = float(np.clip(new_sg, 1.0, args.size_nmax / 2.0))
                    if abs(new_mu - cur_mu) > 1e-6 or abs(new_sg - cur_sigma) > 1e-6:
                        log(f"  [moment §15] ramp (eq mean={size_ema:.2f} sig={sigma_ema:.2f}) -> "
                            f"mu {cur_mu:.2f}->{new_mu:.2f}(step{mu_step_sz:.3g}) "
                            f"sig {cur_sigma:.2f}->{new_sg:.2f}(step{sig_step_sz:.3g})")
                        cur_mu, cur_sigma = new_mu, new_sg
                        p_size_target = make_size_target(cur_mu, cur_sigma, args.size_nmax, device)
                else:
                    # XVR < floor: feedback unreliable -> step mu_tgt/sigma_tgt ONE step BACK toward native init.
                    new_mu = max(cur_mu - args.moment_step0, mu_init)
                    sg_dir = 0.0 if abs(cur_sigma - sig_init) < 1e-6 else (-1.0 if cur_sigma > sig_init else 1.0)
                    new_sg = cur_sigma + sg_dir * min(args.moment_step0, abs(cur_sigma - sig_init))
                    if abs(new_mu - cur_mu) > 1e-6 or abs(new_sg - cur_sigma) > 1e-6:
                        log(f"  [size XVRgate] XVR {xvr_ema:.2f}<{args.xvr_floor:.2f} -> step back "
                            f"mu {cur_mu:.2f}->{new_mu:.2f} sig {cur_sigma:.2f}->{new_sg:.2f} (toward native init)")
                        cur_mu, cur_sigma = new_mu, new_sg
                        mu_step_sz = sig_step_sz = args.moment_step0   # re-coarsen ladder for the next climb
                        p_size_target = make_size_target(cur_mu, cur_sigma, args.size_nmax, device)
            # comp thermostat: at ratchet, nudge the L_spec target (f_tgt) ±step so the REALIZED element
            # fractions land within ±band of the desired composition (analogue of §15; oscillation OK by design).
            if args.comp_thermostat:
                # XVR-gate: only PUSH composition toward GEOM while validity has budget (XVR_ema >= floor);
                # below the floor, EASE f_tgt overshoot back toward desired so XVR recovers (composition only).
                push_ok = (args.xvr_floor <= 0.0) or (xvr_ema is None) or (xvr_ema >= args.xvr_floor)
                ch = []
                for z in ELEMS:
                    fr = comp_ema.get(z)
                    if fr is None:
                        continue
                    fd = comp_des[z]
                    if fr < fd * (1.0 - args.comp_therm_band) and push_ok:
                        comp_tgt[z] = min(comp_tgt[z] * (1.0 + args.comp_therm_step), fd * args.comp_therm_cap)
                        ch.append(f"{z}:{100 * comp_tgt[z]:.2f}↑")
                    elif fr > fd * (1.0 + args.comp_therm_band):
                        comp_tgt[z] = max(comp_tgt[z] * (1.0 - args.comp_therm_step), fd * 0.1)
                        ch.append(f"{z}:{100 * comp_tgt[z]:.2f}↓")
                    elif (not push_ok) and comp_tgt[z] > fd:   # XVR below floor: relax overshoot toward desired
                        comp_tgt[z] = max(comp_tgt[z] * (1.0 - args.comp_therm_step), fd)
                        ch.append(f"{z}:{100 * comp_tgt[z]:.2f}~")
                if ch:
                    f_target = build_f_target()
                    log(f"  [comp-therm{'' if push_ok else ' XVRgate'}] f_tgt% -> " + " ".join(ch))

        # === (6) PID update lam_s, lam_c for the NEXT step (§6 eq 9) ===
        if args.lam_size_pid:
            cur_lam_size, size_int = pid_lambda(
                _Lsz, args.kappa_size, args.lam_size, size_int,
                args.lam_size_kp, args.lam_size_ki, args.lam_size_int_max, args.lam_size_decay, args.lam_size_max_mult)
        if args.lam_comp_pid:
            cur_lam_comp, comp_int = pid_lambda(
                _Lsp, args.kappa_comp, args.lam_comp, comp_int,
                args.lam_comp_kp, args.lam_comp_ki, args.lam_comp_int_max, args.lam_comp_decay, args.lam_comp_max_mult)

        # === (7) logging (fixed full-set table) + checkpoint ===
        valid_smis = [s for s in smis if s]
        uniq_frac = len(set(valid_smis)) / max(len(valid_smis), 1)
        idiv = internal_div(valid_smis)
        _qeds = [qed_of(s) for s in valid_smis]; _qeds = [q for q in _qeds if q == q]
        mean_qed = float(np.mean(_qeds)) if _qeds else 0.0
        elapsed = time.time() - t0
        idiv_s = f"{idiv:.3f}" if idiv == idiv else "nan"
        ce = lambda z, d=1: 100 * comp_ema.get(z, 0.0)
        xtb_str = f" xtbok={xtb_ok_n}/{args.batch}" if step_use_xtb else ""
        _szema = size_ema if size_ema is not None else mean_size
        _sgema = sigma_ema if sigma_ema is not None else std_size   # realized size sigma, 10-step EMA
        log(f"step {step:4d}  R={mean_r:.3f} base={baseline:.3f}{xtb_str} "
            f"sz={mean_size:.1f}±{std_size:.1f}(10MA {_szema:.1f}±{_sgema:.1f} / P̂tgt {cur_mu:.1f}±{cur_sigma:.1f}) {'XVR' if step_use_xtb else 'MVR'}={xvr_n}/{args.batch} run={n_runaway} "
            f"C%={ce(6):.1f} N%={ce(7):.1f} O%={ce(8):.1f} S%={ce(16):.2f} F%={ce(9):.2f} Cl%={ce(17):.2f} Br%={ce(35):.2f} "
            f"KLsz={_Lsz:.3f} KLsp={_Lsp:.4f} scf={cur_scaffdiv * 100:.0f}%(tgt{args.target_scaffdiv * 100:.0f}) coll={cur_coll * 100:.2f}% "
            f"idiv={idiv_s} uniq={uniq_frac:.2f} qed={mean_qed:.2f} "
            f"lamS={cur_lam_size:.1f} lamC={cur_lam_comp:.1f} Xema={xvr_ema:.2f} lamScf={cur_lam_scaff:.2f} logP={cur_logp:.2f} lamLP={cur_lam_logp:.2f} tani={cur_tani:.3f} lamTani={cur_lam_tani:.2f} "
            f"loss={loss.item():.3f}(rl={loss_rl.item():.2f} kl={loss_kl.item():.3f} szKL={_Lsz:.3f} spKL={_Lsp:.4f}) "
            f"|g|={grad_norm:.1f} lp={log_p_policy.mean().item():.0f} t={elapsed:.0f}s")

        history.append({
            "step": step, "R": mean_r, "baseline": baseline, "loss": loss.item(),
            "loss_rl": loss_rl.item(), "loss_kl": loss_kl.item(),
            "L_size": _Lsz, "L_spec": _Lsp, "lam_size": cur_lam_size, "lam_comp": cur_lam_comp,
            "grad_norm": grad_norm, "log_p_policy": log_p_policy.mean().item(),
            "xvr": xvr_n, "mean_size": mean_size, "std_size": std_size, "mean_size_all": mean_size_all,
            "uniq_frac": uniq_frac, "idiv": (idiv if idiv == idiv else None), "mean_qed": mean_qed,
            "scaffdiv": cur_scaffdiv, "coll": cur_coll, "lam_scaff": cur_lam_scaff,
            "comp": {int(z): comp_ema.get(z, 0.0) for z in ELEMS},
            "n_runaway": n_runaway, "elapsed": elapsed,
        })

        if (step + 1) % args.save_every == 0 or step == args.n_train_steps - 1:
            sp = os.path.join(args.out_dir, f"ckpt_step{step}.pt")
            torch.save({"step": step, "config": cfg, "model": policy.state_dict(),
                        # §17: stamp this code's architecture version so a future resume/generation can
                        # consistency-check it (check_ckpt_compat); see klrl_control.CODE_VERSION.
                        "arch_version": CODE_VERSION,
                        # 2026-07-06: embed the generation-time settings this ckpt was trained with, so a
                        # ckpt is self-describing and a generator can consistency-check its argv against
                        # them (check_gen_compat). max_offset is architectural (also in config) but is
                        # mirrored here so all three generation bounds live in one place.
                        "gen": {"max_steps_per_mol": args.max_steps_per_mol, "size_nmax": args.size_nmax,
                                "max_offset": cfg.get("max_offset")},
                        "baseline": baseline, "history": history,
                        "opt": opt.state_dict(), "best_xema": best_xema,   # 2026-07-10: opt-state + best tracker (resume-safe)
                        # §16: persist the §15 target-moment + 2-bisection ladder + comp dial so a resume
                        # CONTINUES them (read back at init above) instead of resetting to the off0 formula
                        "klrl_state": {"cur_mu": cur_mu, "cur_sigma": cur_sigma, "mu_init": mu_init,
                                       "sig_init": sig_init, "mu_step_sz": mu_step_sz,
                                       "sig_step_sz": sig_step_sz, "comp_tgt": comp_tgt}}, sp)
            log(f"  saved {sp}")
            if args.max_keep_ckpts > 0:
                cks = []
                for f in glob.glob(os.path.join(args.out_dir, "ckpt_step*.pt")):
                    m = re.match(r"ckpt_step(\d+)\.pt$", os.path.basename(f))
                    if m:
                        cks.append((int(m.group(1)), f))
                cks.sort()
                for _, f in cks[:-args.max_keep_ckpts]:
                    try:
                        os.remove(f)
                    except Exception:
                        pass
            if xvr_ema is not None and xvr_ema > best_xema:      # 2026-07-10: preserve best (rotate-proof best.pt; max_keep only touches ckpt_step*.pt)
                best_xema = xvr_ema
                bp = os.path.join(args.out_dir, "best.pt")
                torch.save({"step": step, "config": cfg, "model": policy.state_dict(),
                            "arch_version": CODE_VERSION, "opt": opt.state_dict(),
                            "gen": {"max_steps_per_mol": args.max_steps_per_mol, "size_nmax": args.size_nmax,
                                    "max_offset": cfg.get("max_offset")},
                            "baseline": baseline, "best_xema": best_xema, "best_step": step,
                            "klrl_state": {"cur_mu": cur_mu, "cur_sigma": cur_sigma, "mu_init": mu_init,
                                           "sig_init": sig_init, "mu_step_sz": mu_step_sz,
                                           "sig_step_sz": sig_step_sz, "comp_tgt": comp_tgt}}, bp)
                log(f"  [best] Xema={best_xema:.3f} step={step} -> {bp}")

    log(f"done. {args.n_train_steps} steps in {time.time() - t0:.0f}s")
    _logf.close()


if __name__ == "__main__":
    main()
