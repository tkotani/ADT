"""Perception-free XVR reward (RDKit valence/kekulize/AddHs 排除).

Drop-in for reward_xtb.xvr_reward_batch when XVR_PFREE=1:
  generate heavy (atoms,bonds,na)
    -> screen ② disconnection (ADT bond graph 連結成分==1)
    -> screen ① clash (check_collisions)
    -> completer n_H (COMPLETER_CKPT, MAIN-thread pre-pass=GPU)
    -> VSEPR H placement -> all-atom molblock (RDKit=graph container only)
    -> xtb_relax (再利用: e_gain/opt_heavy 実績あり) -> 距離topology保存
    -> reward: R_FAIL / R_XTB / estrain-shaped R_XVR (strain_pa=|e_gain|/n_heavy, 現行と同一)

Completer runs in the MAIN thread (GPU); xTB in the thread pool (subprocess).
"""
import os, sys, math, subprocess, shutil, re
sys.path.insert(0, os.path.expanduser("~/ADT/common"))
sys.path.insert(0, os.path.expanduser("~/ADT/Hcompleter"))
from concurrent.futures import ThreadPoolExecutor
from collections import deque
import numpy as np
import torch
from rdkit import Chem, RDLogger
RDLogger.DisableLog("rdApp.*")
from train_completer import HCompleter, N_SLOTS
import adt_tokenizer as tk
from relative_pointer import absolute_to_relative
from collision_check import check_collisions
from gen_eval_lib import xtb_relax, xtb_hrelax

ALLOWED_ATOMS = {1, 5, 6, 7, 8, 9, 14, 15, 16, 17, 33, 35, 53}
R_FAIL, R_CLASH, R_XTB, R_XVR = 0.0, 0.3, 0.6, 1.0     # clashVR: clash-pass(構造valid)に0.3 credit
CLASHVR = os.environ.get("CLASHVR", "1") == "1"        # clashVR tier on/off (default on)
H_PLACER = os.environ.get("H_PLACER", "rdkit")         # rdkit(completer n_H駆動 AddHs) / vsepr
CLASHVR_SWITCH = float(os.environ.get("CLASHVR_SWITCH", "0.90"))  # clash-pass EMA閾値→pure XVRへ切替
_clash_ema = None                                      # rolling clash-pass rate
_switched = False                                      # True after auto-switch to pure XVR
XVR_ESTRAIN_TAU = float(os.environ.get("XVR_ESTRAIN_TAU", "0") or 0)
H_PRERELAX = os.environ.get("H_PRERELAX") == "1"       # insert H-only xTB prerelax (freeze heavy) before full relax
BL = {6: 1.09, 7: 1.01, 8: 0.96, 16: 1.34, 15: 1.42, 9: 0.92, 17: 1.27, 35: 1.41, 53: 1.61}
COV = {1: 0.31, 6: 0.76, 7: 0.71, 8: 0.66, 9: 0.57, 15: 1.07, 16: 1.05, 17: 1.02, 35: 1.20, 53: 1.39}
_PT = Chem.GetPeriodicTable()
MLNH_PARITY = os.environ.get("MLNH_PARITY", "1") == "1"   # 修正MLnH: fix completer n_H parity so neutral molecule is closed-shell (even electrons). default ON
STD_VAL = {5: 3, 6: 4, 7: 3, 8: 2, 9: 1, 14: 4, 15: 3, 16: 2, 17: 1, 33: 3, 35: 1, 53: 1}
_PARITY_STATS = {"n": 0, "odd": 0, "remove": 0, "add": 0, "fail": 0}
H_INTEGRITY = os.environ.get("H_INTEGRITY", "1") == "1"   # handle a molecule whose H-prerelaxed structure has a detached/stray H (fragmentation). default ON
H_INTEGRITY_MODE = os.environ.get("H_INTEGRITY_MODE", "correct")  # "correct" = strip the detached(excess) H + re-prerelax, recover if intact; "reject" = drop
_H_STRAY_A = float(os.environ.get("H_STRAY_A", "1.6"))    # an H farther than this from every heavy atom = detached
# ストッパー: H prerelax へ戻れる回数の上限（初回 + 最大 MAX_RETRY 回のやり直し）。上限に達したら F1b として記録・reject。
H_INTEGRITY_MAX_RETRY = int(os.environ.get("H_INTEGRITY_MAX_RETRY", "3"))
_INTEGRITY_STATS = {"n": 0, "correct": 0, "reject": 0, "n_full": 0, "correct_full": 0, "reject_full": 0,
                    "F1a_odd_parity": 0, "F1b_retry_exhausted": 0}   # F1a: 奇数個除去が必要 / F1b: retry 上限超過
# MLnH(+MLHplacer) 性能: 何個(偶数)・何回 弾いてから成功したか の分布。strip も parity 補正も無し = H数を一発で当てた。
_MLNH_PERF = {"ok_first": 0, "parity_only": 0, "strip_hist": {}, "attempt_hist": {}, "at_prerelax": 0, "at_full": 0,
              "odd_extra_removed": 0}   # xTB が奇数個(1H)弾いた -> もう1個除去して偶数に揃えた回数
# _prep で screen 落ちした理由の内訳（崖では screened が最大の失敗群。どの screen が効いているかを知る）
_SCREEN_STATS = {"atom": 0, "disconnect": 0, "clash": 0, "completer": 0, "placer": 0}
# --- realizable-XTP via clamp->unclamp (2026-07-10): when free relax FLIPS heavy topology, try to RESCUE it
# by constraining the generated heavy bonds (clamp) -> relax -> release (unclamp) -> relax; realizable if the
# unclamp result still preserves the generated topology (= a stable topo-preserving minimum EXISTS = honest XTP).
# --- 積算誤差を「生成側」で減らすための報酬項（2026-07-10, IKT とは直交） ---
# λ1: 自己整合ペナルティ = |bonds0 △ _heavy_conn(生成座標)| / |bonds0|
#     「ADT が宣言したグラフを、ADT 自身の3D座標が実現できているか」。xtb 不要なので
#     **xtb 非収束(=積算誤差 最大)の分子にも勾配が届く**のが肝（現行は reward 0 で情報ゼロ）。
XVR_SELFMIS_LAM = float(os.environ.get("XVR_SELFMIS_LAM", "0") or 0)
# λ2: 緩和変位 RMSD(gen_heavy -> 緩和後 heavy) のペナルティ = 積算誤差の直接量（実測 median ~0.9Å）
XVR_RMSD_LAM = float(os.environ.get("XVR_RMSD_LAM", "0") or 0)
# λ_c: **clash の段階罰**。崖では screened の 100% が clash で、全分子の ~30% を占める最大の失敗群。
#      現在は screen して reward 0（勾配ゼロ）。めり込み深さ Σ(thr - d)/na に比例した負の報酬を与える。
#      xtb 不要。selfmis の「偽接触」と同じ現象の重篤版なので、積算誤差を最も直接的に押し下げる。
XVR_CLASH_LAM = float(os.environ.get("XVR_CLASH_LAM", "0") or 0)
XVR_CLAMP = os.environ.get("XVR_CLAMP") == "1"          # env-gate; default OFF (proven free-relax). ON for the step2b kt1 run.
XVR_CLAMP_FC = os.environ.get("XVR_CLAMP_FC", "0.5")    # $constrain force constant
# `distance: i,j,auto` は「その時点の距離」を固定する。積算誤差で bonds0 の結合が結合閾値を超えて
# 伸びていると、auto は**壊れた結合を壊れたまま固定**し、解放しても形成されない -> F3_unclamp_flip と誤判定。
# XVR_CLAMP_IDEAL=1 なら、伸びた結合(d > 1.3*(cov_i+cov_j)) だけ **結合距離 cov_i+cov_j を明示して引き寄せる**。
XVR_CLAMP_IDEAL = os.environ.get("XVR_CLAMP_IDEAL") == "1"
# --- 2026-07-10: XTP の底上げ 3点セット（いずれも既定 off = 従来挙動）--------------------
# ① CLAMP_ONLY: free relax を廃し、全分子を H-prerelax -> clamp(bonds0) -> unclamp で判定する。
#    従来は free relax が収束した分子しか clamp を試されず、最も clamp が効くはずの
#    relax_fail (full relax 非収束 = 積算誤差最大) に clamp が一度も届いていなかった。
XVR_CLAMP_ONLY = os.environ.get("XVR_CLAMP_ONLY") == "1"
# ② STRAIN_HPRE: strain_pa の基準点を全分子 H-prerelax 構造に統一する。
#    従来 clamp 救済分子は「clamp 後の極小からの利得」で測られ、歪みが過小評価 -> 報酬が過大だった。
XVR_STRAIN_HPRE = os.environ.get("XVR_STRAIN_HPRE") == "1"
#    ※ 和集合(free で通れば合格 / 落ちたら clamp)は棄却した: 分子ごとに手順が変わる採点は
#      測定としても報酬としても不健全。全分子を必ず同一手順に通す。
# ①'' CLAMP_FADE: 拘束を一気に切らず、力の定数を段階的に 0 へ落とす連続変形 (homotopy)。
#    目標距離 d0 は最初に一度だけ決めて固定し、k のみ下げる: E_k = E_xtb + k*Σ(d-d0)^2, k -> 0。
#    最終段は k=0 = ただの free relax なので、合格証明は「拘束なし xTB 極小が bonds0 を保つ」のまま。
#    狙い: 急な解放で起きる F3_unclamp_flip (B の最大失敗, 25-33/192) の削減。
#    最終段が free relax なので A(free) が通す分子も原理的に通り、和集合 C が不要 = 全分子 単一手順。
XVR_CLAMP_FADE = os.environ.get("XVR_CLAMP_FADE", "")   # 例 "1.0,0.3,0.1,0.03"（空なら従来の一段 clamp）
# ①''' CLAMP_LOOSE: clamp(拘束)段を --opt loose で粗く収束させ速度を稼ぐ。最終 unclamp は full --opt のまま
#    ＝合格判定(bonds0を保つ拘束なし極小)の厳密さは不変。clamp 段の役割は「bonds0 のベイスンに入れる」
#    だけで最終精密化は unclamp が担うので、粗くても品質はほぼ落ちない想定。速度検証用（default off）。
XVR_CLAMP_LOOSE = os.environ.get("XVR_CLAMP_LOOSE") == "1"
# ③ FAIL_CREDIT: 「xtb は収束したが bonds0 を実現できない」分子への credit（旧 R_XTB=0.6 の役割①）。
#    成功報酬は R_XTB + (R_XVR-R_XTB)exp(-strain/tau) で strain 大なら 0.6 に漸近するため、
#    0.6 の credit があると崖(strain 大)で XTP の勾配が消える。0 にすると常に 0.6 のギャップが立つ。
#    失敗側の密な勾配は selfmis/clash 減点が担う（xtb 不要・screened にも届く）。
XVR_FAIL_CREDIT = float(os.environ.get("XVR_FAIL_CREDIT", "0.6"))
# ④ RMSD_RHO: 幾何の積算誤差 (ADT の生成重原子座標 -> 最終緩和後) を報酬に入れる。
#    エネルギーは軟モード(ねじれ)に盲目 = 大きく動いてもほぼ無コスト。RMSD はそこを直接見る。
#    ★減算ではなく shaping の因子にする: FAIL_CREDIT=0 で作った「成功と失敗の 0.6 のギャップ」を
#      成功側からの減算で潰さないため。reward は必ず [R_XTB, R_XVR] に留まる。
#      success = R_XTB + (R_XVR-R_XTB)*exp(-strain/tau)*exp(-rmsd/rho)
XVR_RMSD_RHO = float(os.environ.get("XVR_RMSD_RHO", "0") or 0)   # 0 = off。推奨 0.3-0.5 (A)
PFREE_DUMP = os.environ.get("PFREE_DUMP")                # set -> 生成分子 (Z, coords, bonds, na) を append pickle
HARTREE2KCAL = 627.5094740631


def _kabsch_rmsd(P, Q):
    """並進・回転を除いた heavy RMSD。xtb は重心/慣性主軸を動かすので生の座標差は使えない。"""
    P = np.asarray(P, float); Q = np.asarray(Q, float)
    if P.shape != Q.shape or len(P) == 0:
        return None
    P = P - P.mean(0); Q = Q - Q.mean(0)
    V, S, Wt = np.linalg.svd(P.T @ Q)
    d = np.sign(np.linalg.det(V @ Wt))
    D = np.diag([1.0, 1.0, d])
    P = P @ (V @ D @ Wt)
    return float(np.sqrt(((P - Q) ** 2).sum(1).mean()))


def _cu_energy_kcal(path):
    """xtbopt.xyz のコメント行から絶対全エネルギー(Hartree) -> kcal/mol。拘束バイアスを含まない生の GFN2 値。"""
    try:
        with open(path) as f:
            comment = f.readlines()[1]
        m = re.search(r"energy:\s*(-?[\d.]+)", comment)
        return float(m.group(1)) * HARTREE2KCAL if m else None
    except Exception:
        return None
_CLAMP_STATS = {"tried": 0, "rescued": 0}


def _bank_struct():
    """Dynamic gate (read per call, no import-order dependency): freeze full HADD + relaxed
    all-atom structures into the reward dict so a molecule bank can store 生/HADD/緩和.

    DEFAULT ON (2026-07-13): a record without the relaxed structure cannot yield a SMILES, and without a
    SMILES there is no diversity, no novelty and no rdkit_valid -- i.e. the record is unusable for the
    paper. It was opt-in before, and one forgotten env var cost a 80,000-molecule run. Set BANK_STRUCT=0
    to switch it off deliberately (RL speed)."""
    return os.environ.get("BANK_STRUCT", "1") != "0"


def _parse_xyz_all(path_or_text):
    """Full all-atom parse of an xyz FILE PATH *or* of the xyz TEXT itself -> (anums, coords) or (None, None).

    _clamp_unclamp returns the xyz CONTENT in "opt_xyz" (not a path); passing that to a path-only parser
    failed silently, which nulled struct_relaxed in every bank record under CLAMP_ONLY (= the current XTP
    definition) and disabled the post-relax H-integrity check. Accept both."""
    try:
        if isinstance(path_or_text, str) and "\n" in path_or_text:
            lines = path_or_text.splitlines()                     # already the file content
        else:
            with open(path_or_text) as f:
                lines = f.read().splitlines()
        n = int(lines[0].split()[0])
        zs, xs = [], []
        for ln in lines[2:2 + n]:
            p = ln.split()
            zs.append(_PT.GetAtomicNumber(p[0]))
            xs.append([float(p[1]), float(p[2]), float(p[3])])
        return zs, np.asarray(xs, np.float32)
    except Exception:
        return None, None

_dev = "cuda" if torch.cuda.is_available() else "cpu"
_cck = torch.load(os.environ["COMPLETER_CKPT"], weights_only=False)
_comp = HCompleter(_cck["cfg"]["d_model"], _cck["cfg"]["n_layers"]).to(_dev).eval()
_comp.load_state_dict(_cck["model"])
print("[reward_pfree] completer loaded: %s (d=%d L=%d) dev=%s tau=%s h_prerelax=%s"
      % (os.environ["COMPLETER_CKPT"], _cck["cfg"]["d_model"], _cck["cfg"]["n_layers"], _dev, XVR_ESTRAIN_TAU, H_PRERELAX), flush=True)
print("[reward_pfree] XTP mode: clamp_only=%s fade=%s loose=%s strain_hpre=%s fail_credit=%.2f rho=%.2f clamp=%s ideal=%s | lam selfmis=%.2f clash=%.2f rmsd=%.2f"
      % (XVR_CLAMP_ONLY, XVR_CLAMP_FADE or "off", XVR_CLAMP_LOOSE, XVR_STRAIN_HPRE, XVR_FAIL_CREDIT, XVR_RMSD_RHO, XVR_CLAMP, XVR_CLAMP_IDEAL,
         XVR_SELFMIS_LAM, XVR_CLASH_LAM, XVR_RMSD_LAM), flush=True)


def _bonded(z1, z2, d):
    return d < 1.3 * (COV.get(z1, 0.75) + COV.get(z2, 0.75))


def _ncomp(n, bonds):
    adj = [[] for _ in range(n)]
    for a, b in bonds:
        adj[a].append(b); adj[b].append(a)
    seen = [False] * n; c = 0
    for s in range(n):
        if seen[s]:
            continue
        c += 1; dq = deque([s]); seen[s] = True
        while dq:
            u = dq.popleft()
            for v in adj[u]:
                if not seen[v]:
                    seen[v] = True; dq.append(v)
    return c


def _hdirs(nbr, nH, seed):
    rng = np.random.RandomState(seed)
    fixed = (np.array([np.asarray(d, float) / (np.linalg.norm(d) + 1e-9) for d in nbr])
             if nbr else np.zeros((0, 3)))
    H = rng.randn(nH, 3); H /= (np.linalg.norm(H, axis=1, keepdims=True) + 1e-9)
    nf = len(fixed)
    for _ in range(400):
        pts = np.vstack([fixed, H]) if nf else H
        g = np.zeros((nH, 3))
        for i in range(nH):
            for k in range(len(pts)):
                if k == nf + i:
                    continue
                diff = H[i] - pts[k]; d2 = (diff * diff).sum() + 1e-6
                g[i] += diff / d2 ** 1.5
        H = H + 0.03 * g; H /= (np.linalg.norm(H, axis=1, keepdims=True) + 1e-9)
    return H


def _heavy_conn(anums, coords, nh):
    s = set()
    for i in range(nh):
        for j in range(i + 1, nh):
            if _bonded(anums[i], anums[j], float(np.linalg.norm(coords[i] - coords[j]))):
                s.add((i, j))
    return s


def _h_intact(mb):
    """True iff every H in the molblock is within _H_STRAY_A of some heavy atom
    (no detached/stray H). H detachment = the completer over-counted H and xTB ejected
    the excess -> fragmentation. Permissive (True) if the block can't be parsed."""
    try:
        m = Chem.MolFromMolBlock(mb, sanitize=False, removeHs=False)
        if m is None or m.GetNumConformers() == 0:
            return True
        P = m.GetConformer().GetPositions()
        Z = np.array([a.GetAtomicNum() for a in m.GetAtoms()])
        heavy = np.where(Z > 1)[0]
        if len(heavy) == 0:
            return True
        hp = P[heavy]
        for i in np.where(Z == 1)[0]:
            if np.linalg.norm(hp - P[i], axis=1).min() > _H_STRAY_A:
                return False
        return True
    except Exception:
        return True


def _strip_detached_h(mb):
    """離脱した(xTB に弾かれた余剰) H を molblock から除去 -> (corrected_molblock, n_removed)。
    弾かれた数が **奇数** の場合は電子パリティが崩れてラジカルになるため、**最も緩く付いている H
    (最近接 heavy 原子から最も遠い H) を 1 個追加で除去して偶数に揃える**（2026-07-10, ユーザ指示）。
    追加除去できる H が残っていない場合のみ (None,0) を返し、caller は F1a として reject する。
    heavy 原子は不変なので `na` は変わらない（= bonds0 は保たれる）。"""
    try:
        m = Chem.MolFromMolBlock(mb, sanitize=False, removeHs=False)
        if m is None or m.GetNumConformers() == 0:
            return None, 0
        P = m.GetConformer().GetPositions()
        Z = np.array([a.GetAtomicNum() for a in m.GetAtoms()])
        heavy = np.where(Z > 1)[0]
        if len(heavy) == 0:
            return None, 0
        hp = P[heavy]
        rem = [int(i) for i in np.where(Z == 1)[0]
               if np.linalg.norm(hp - P[i], axis=1).min() > _H_STRAY_A]
        if not rem:
            return mb, 0
        if len(rem) % 2 != 0:            # 奇数個の離脱 -> もう1個除去して偶数に揃える(閉殻を保つ)
            _rs = set(rem)
            cand = [(float(np.linalg.norm(hp - P[i], axis=1).min()), int(i))
                    for i in np.where(Z == 1)[0] if int(i) not in _rs]
            if not cand:
                return None, 0           # 追加除去できる H が無い -> F1a reject
            cand.sort(reverse=True)      # 最近接 heavy から最も遠い H = 最も緩く付いている = 余剰の可能性が高い
            rem.append(cand[0][1])
            _MLNH_PERF["odd_extra_removed"] += 1
        ed = Chem.RWMol(m)
        for i in sorted(rem, reverse=True):
            ed.RemoveAtom(i)
        return Chem.MolToMolBlock(ed.GetMol()), len(rem)
    except Exception:
        return None, 0


def _intact_zc(Z, P):
    """(is_intact, detached_H_indices): an H is 'detached' if farther than _H_STRAY_A from every heavy atom.
    Operates directly on atomic numbers + coords (for the full-relax structure, parsed from xyz)."""
    Z = np.asarray(Z); P = np.asarray(P, dtype=np.float64)
    heavy = np.where(Z > 1)[0]
    if len(heavy) == 0:
        return True, []
    hp = P[heavy]
    det = [int(i) for i in np.where(Z == 1)[0]
           if np.linalg.norm(hp - P[i], axis=1).min() > _H_STRAY_A]
    return (len(det) == 0), det


def _zc_to_mb(Z, P):
    """Bondless molblock from atomic numbers + coords (xTB uses coords/elements only; bonds irrelevant)."""
    rw = Chem.RWMol()
    for z in Z:
        rw.AddAtom(Chem.Atom(int(z)))
    conf = Chem.Conformer(len(Z))
    for j in range(len(Z)):
        conf.SetAtomPosition(j, [float(P[j][0]), float(P[j][1]), float(P[j][2])])
    rw.AddConformer(conf, assignId=True)
    return Chem.MolToMolBlock(rw.GetMol())


def _strip_detached_zc(Z, P):
    """離脱(余剰) H を Z/coords 構造から除去 -> (bondless molblock, n_removed)。
    奇数個なら **最も緩く付いている H を 1 個追加除去して偶数に揃える**（閉殻を保つ）。
    除去対象が無い / 追加除去できる H が無い場合のみ (None,0)。"""
    Z = np.asarray(Z); P = np.asarray(P, dtype=np.float64)
    _, det = _intact_zc(Z, P)
    if not det:
        return None, 0
    if len(det) % 2 != 0:                # 奇数個の離脱 -> もう1個除去して偶数に
        heavy = np.where(Z > 1)[0]
        if len(heavy) == 0:
            return None, 0
        hp = P[heavy]; _ds = set(det)
        cand = [(float(np.linalg.norm(hp - P[i], axis=1).min()), int(i))
                for i in np.where(Z == 1)[0] if int(i) not in _ds]
        if not cand:
            return None, 0               # 追加除去できる H が無い -> F1a reject
        cand.sort(reverse=True)
        det = list(det) + [cand[0][1]]
        _MLNH_PERF["odd_extra_removed"] += 1
    keep = [i for i in range(len(Z)) if i not in set(det)]
    return _zc_to_mb(Z[keep], P[keep]), len(det)


def _completer_nH(anums, coords, bonds):
    rw = Chem.RWMol()
    for z in anums:
        rw.AddAtom(Chem.Atom(int(z)))
    for a, b in bonds:
        rw.AddBond(int(a), int(b), Chem.BondType.SINGLE)
    conf = Chem.Conformer(len(anums))
    for j in range(len(anums)):
        conf.SetAtomPosition(j, [float(x) for x in coords[j]])
    rw.AddConformer(conf, assignId=True); mol = rw.GetMol()
    try:
        Chem.FastFindRings(mol)
    except Exception:
        pass
    pos = np.asarray(coords, dtype=np.float64)
    for _ in range(4):
        try:
            tokd = tk.tokenize_molecule(mol, pos)
        except Exception:
            tokd = None
        if tokd is None:
            continue
        arr = tk.tokens_to_array(tokd.tokens).astype(np.int64); off = absolute_to_relative(arr); L = len(off)
        v = torch.tensor(off, device=_dev).unsqueeze(0)
        sl = (torch.arange(L, device=_dev) % N_SLOTS).unsqueeze(0)
        ss = (torch.arange(L, device=_dev) // N_SLOTS) * N_SLOTS
        ac = v.gather(1, ss.unsqueeze(0)).clamp(min=0)
        pm = torch.zeros(1, L, dtype=torch.bool, device=_dev)
        with torch.no_grad():
            pr = _comp(v, sl, ac, pm).argmax(-1)[0]
        steps = sorted(tokd.atom_table.keys()); out = [-1] * len(anums); j = 0
        for t in range(L // N_SLOTS):
            if int(arr[t * N_SLOTS]) <= 3:
                out[tokd.atom_table[steps[j]].original_idx] = int(pr[t * N_SLOTS + 2]); j += 1
        if all(x >= 0 for x in out):
            return _parity_correct_nH(anums, bonds, out) if MLNH_PARITY else out
    return None


def _parity_correct_nH(anums, bonds, nH):
    """修正MLnH parity fix. A neutral closed-shell molecule needs an even electron count
    (sum Z_all = sum Z_heavy + sum nH). If odd, the completer mis-counted H parity ->
    forced radical (RDKit rejects it; xTB still relaxes it). Adjust ONE atom's H by +-1
    to restore even parity, preferring REMOVAL (the completer over-counts in ~98% of errors:
    the excess H is ejected as stray H/H2 in relaxation)."""
    _PARITY_STATS["n"] += 1
    nH = list(nH)
    if (sum(int(z) for z in anums) + sum(int(h) for h in nH)) % 2 == 0:
        return nH                                              # already even -> unchanged
    _PARITY_STATS["odd"] += 1
    na = len(anums)
    deg = [0] * na
    for a, b in bonds:
        deg[int(a)] += 1; deg[int(b)] += 1
    deficit = [STD_VAL.get(int(anums[i]), 4) - deg[i] - nH[i] for i in range(na)]
    adj = [[] for _ in range(na)]
    for a, b in bonds:
        adj[int(a)].append(int(b)); adj[int(b)].append(int(a))
    # REMOVE candidate (nH_i -= 1): over-valent(deficit<0) first, then an H-bearing atom next
    # to an unsatisfied (deficit>=1) neighbor (removing frees a double bond), then any H-bearer.
    best = None; bestkey = None
    for i in range(na):
        if nH[i] < 1:
            continue
        if deficit[i] < 0:
            key = (0, deficit[i])
        elif any(deficit[j] >= 1 for j in adj[i]):
            key = (1, deficit[i])
        else:
            key = (2, deficit[i])
        if bestkey is None or key < bestkey:
            bestkey = key; best = i
    if best is not None:
        nH[best] -= 1; _PARITY_STATS["remove"] += 1
        return nH
    add = [(deficit[i], i) for i in range(na) if deficit[i] >= 1]   # else ADD to most-deficient
    if add:
        add.sort(reverse=True)
        nH[add[0][1]] += 1; _PARITY_STATS["add"] += 1
        return nH
    _PARITY_STATS["fail"] += 1
    return nH


def _prep(atoms, bonds, na):
    """MAIN-thread: screen + completer + placeH -> all-atom molblock. None if screened/failed."""
    anums = [atoms[k].atomic_num for k in range(na)]
    if any(a not in ALLOWED_ATOMS for a in anums):
        _SCREEN_STATS["atom"] += 1
        return None
    coords = np.array([list(atoms[k].pos) for k in range(na)], dtype=np.float64)
    bonds0 = set()
    for e1, e2 in bonds:
        a, b = int(e1) - 1, int(e2) - 1
        if a != b and 0 <= a < na and 0 <= b < na:
            bonds0.add((min(a, b), max(a, b)))
    bonds0 = list(bonds0)
    # === 案B (2026-07-10): トポロジーは ADT が宣言した結合グラフ `bonds0` を尊重する ===
    # 旧 `ref = _heavy_conn(生成座標)` は距離ベースなので、積算誤差で接近した 1-3(geminal) 原子等を
    # 「結合」と誤認する(偽結合)。その結果 (a) 緩和が偽接触を解消すると「flip=失敗」と誤判定され XTP を
    # 過小評価し、(b) clamp がその偽結合を距離拘束して あり得ない幾何 を強制するため rescue が働かなかった。
    # bonds0 基準なら H付与→H緩和→全体緩和 は全て幾何操作でトポロジーを変えない(一気通貫)ので、
    # 「トポロジーが変わった」失敗は消え、「bonds0 を保ったまま安定に緩和できなかった」失敗だけが残る。
    ref = set(bonds0)                                              # XTP 判定・clamp 拘束の基準 = ADT の結合グラフ
    ref_dist = _heavy_conn(anums, coords, na)                      # 生成座標の距離トポロジー（旧定義の ref でもある）
    # 自己整合ミス: ADT の宣言グラフ(bonds0) と 生成幾何の距離トポロジー の対称差。
    # >0 なら「宣言したグラフを自分の3D座標が実現できていない」= 積算誤差の直接の症状
    # （伸びて切れた結合 = bonds0\ref_dist / 圧縮角などによる偽接触 = ref_dist\bonds0）。xtb 不要。
    selfmis_miss = len(ref - ref_dist)    # bonds0 にあるが生成幾何では結合していない = 伸びて切れた結合
    selfmis_spur = len(ref_dist - ref)    # 生成幾何で結合と誤認される近接 = 偽接触(1-3 圧縮角など)
    selfmis = selfmis_miss + selfmis_spur
    if _ncomp(na, list(ref)) != 1:                                 # ② disconnection: ADT 結合グラフが1分子か
        _SCREEN_STATS["disconnect"] += 1
        return None
    _det = []
    if os.environ.get("XVR_NOCLASH") == "1":
        hc = False                                                 # XVR_NOCLASH: skip clash screen (ablation; eval leaves it unset -> honest)
    else:
        try:
            hc, _det = check_collisions([list(c) for c in coords], anums, set(bonds0))
        except Exception:
            hc = False; _det = []
    if hc:                                                          # ① clash（崖の最大失敗群）
        _SCREEN_STATS["clash"] += 1
        # None を返さず「screened dict」を返す -> reward 0 のままだが、めり込み深さと selfmis を持ち帰れる。
        # batch 側で λ_c·Σ(thr-d)/na の段階罰を与えれば、今まで勾配ゼロだった 3割に密な信号が入る。
        _depth = float(sum(max(0.0, t - d) for (_a, _b, d, t) in _det))
        return dict(screened="clash", clash_n=len(_det), clash_depth=_depth, clash_max=float(max([t - d for (_a, _b, d, t) in _det], default=0.0)),
                    selfmis=selfmis, selfmis_miss=selfmis_miss, selfmis_spur=selfmis_spur, nbond0=len(ref), na=na)
    nH = _completer_nH(anums, coords, bonds0)
    if nH is None:
        _SCREEN_STATS["completer"] += 1
        return None
    # --- H placement (M): MLHadd learned H positions (perception-free H GEOMETRY; RDKit only best-effort
    # for the diversity SMILES side-channel, not for the XVR judgment). Falls through to VSEPR if it fails.
    if H_PLACER == "mlhadd":
        try:
            import mlhadd
            mb = mlhadd.place_h(anums, coords, bonds0, [int(x) for x in nH])
        except Exception:
            mb = None
        if mb is not None:
            smi = ""
            try:                                                    # real SMILES (diversity only), crude fallback
                rw = Chem.RWMol()
                for i, z in enumerate(anums):
                    a = Chem.Atom(int(z)); a.SetNumExplicitHs(int(nH[i])); a.SetNoImplicit(True); rw.AddAtom(a)
                for a, b in bonds0:
                    rw.AddBond(int(a), int(b), Chem.BondType.SINGLE)
                m = rw.GetMol(); Chem.SanitizeMol(m); smi = Chem.MolToSmiles(m)
            except Exception:
                try:
                    m2 = Chem.RWMol()
                    for z in anums:
                        m2.AddAtom(Chem.Atom(int(z)))
                    for a, b in bonds0:
                        m2.AddBond(int(a), int(b), Chem.BondType.SINGLE)
                    smi = Chem.MolToSmiles(Chem.RemoveHs(m2.GetMol(), sanitize=False))
                except Exception:
                    smi = ""
            return dict(mb=mb, init_heavy=[list(coords[i]) for i in range(na)], anums=anums, na=na,
                        ref=ref, ref_dist=ref_dist, selfmis=selfmis, nbond0=len(ref),
                        selfmis_miss=selfmis_miss, selfmis_spur=selfmis_spur, placer="mlhadd", smi=smi, nH=[int(x) for x in nH], bonds0=bonds0)
        # mlhadd failed -> fall through to VSEPR
    # --- H placement (A): RDKit AddHs, driven by completer n_H (forces valence -> kekulize -> good geom)
    if H_PLACER == "rdkit":
        try:
            rw = Chem.RWMol()
            for i, z in enumerate(anums):
                a = Chem.Atom(int(z)); a.SetNumExplicitHs(int(nH[i])); a.SetNoImplicit(True)
                rw.AddAtom(a)
            for a, b in bonds0:
                rw.AddBond(int(a), int(b), Chem.BondType.SINGLE)
            conf = Chem.Conformer(na)
            for j in range(na):
                conf.SetAtomPosition(j, [float(x) for x in coords[j]])
            rw.AddConformer(conf, assignId=True)
            mol = rw.GetMol()
            Chem.SanitizeMol(mol)                                  # kekulize/hybridization from forced n_H
            try:
                smi = Chem.MolToSmiles(mol)                        # for diversity (tani/scaffold/qed/logP)
            except Exception:
                smi = ""
            molH = Chem.AddHs(mol, addCoords=True)                 # RDKit's good H placement
            c = molH.GetConformer()
            init_heavy = [[c.GetAtomPosition(i).x, c.GetAtomPosition(i).y, c.GetAtomPosition(i).z]
                          for i in range(na)]
            d = dict(mb=Chem.MolToMolBlock(molH), init_heavy=init_heavy, anums=anums, na=na,
                     ref=ref, ref_dist=ref_dist, selfmis=selfmis, nbond0=len(ref),
                        selfmis_miss=selfmis_miss, selfmis_spur=selfmis_spur, placer="rdkit", smi=smi, nH=[int(x) for x in nH], bonds0=bonds0)
            if _bank_struct():                                     # HADD (all-atom, pre-relax)
                d["hadd_anums"] = [a.GetAtomicNum() for a in molH.GetAtoms()]
                d["hadd_coords"] = np.array(
                    [[c.GetAtomPosition(i).x, c.GetAtomPosition(i).y, c.GetAtomPosition(i).z]
                     for i in range(molH.GetNumAtoms())], np.float32)
            return d
        except Exception:
            pass                                                   # -> VSEPR fallback
    # --- H placement (B): VSEPR repulsion fallback ---
    adj = [[] for _ in range(na)]
    for a, b in bonds0:
        adj[a].append(b); adj[b].append(a)
    rw = Chem.RWMol()
    for z in anums:
        rw.AddAtom(Chem.Atom(int(z)))
    for a, b in bonds0:
        rw.AddBond(int(a), int(b), Chem.BondType.SINGLE)
    cpos = [list(coords[i]) for i in range(na)]                     # heavy first, then H
    for i in range(na):
        k = int(nH[i])
        if k <= 0:
            continue
        nbr = [coords[j] - coords[i] for j in adj[i]]
        for hd in _hdirs(nbr, k, i):
            hi = rw.AddAtom(Chem.Atom(1)); rw.AddBond(hi, i, Chem.BondType.SINGLE)
            cpos.append(list(coords[i] + BL.get(int(anums[i]), 1.09) * hd))
    conf = Chem.Conformer(rw.GetNumAtoms())
    for j, p in enumerate(cpos):
        conf.SetAtomPosition(j, [float(x) for x in p])
    rw.AddConformer(conf, assignId=True)
    vmol = rw.GetMol()
    try:
        mb = Chem.MolToMolBlock(vmol)
    except Exception:
        _SCREEN_STATS["placer"] += 1                                  # mlhadd も VSEPR も molblock 化できず
        return None
    try:
        smi = Chem.MolToSmiles(Chem.RemoveHs(vmol, sanitize=False))   # crude (single-bond) smiles
    except Exception:
        smi = ""
    d = dict(mb=mb, init_heavy=cpos[:na], anums=anums, na=na, ref=ref, ref_dist=ref_dist, selfmis=selfmis, nbond0=len(ref),
                        selfmis_miss=selfmis_miss, selfmis_spur=selfmis_spur, placer="vsepr", smi=smi,
             nH=[int(x) for x in nH], bonds0=bonds0)
    if _bank_struct():                                                # HADD (all-atom, pre-relax)
        d["hadd_anums"] = [a.GetAtomicNum() for a in vmol.GetAtoms()]
        d["hadd_coords"] = np.asarray(cpos, np.float32)
    return d


def _cu_write_xyz(path, z, c):
    with open(path, "w") as f:
        f.write("%d\n\n" % len(z))
        for zz, xyz in zip(z, c):
            f.write("%s %.6f %.6f %.6f\n" % (_PT.GetElementSymbol(int(zz)), xyz[0], xyz[1], xyz[2]))


_CU_FLIP_CACHE = {}     # FLIP_KEEP=1: idx -> (anums, coords) of the post-unclamp geometry
_CU_CLAMP_CACHE = {}    # CLAMP_KEEP=1: idx -> (anums, coords) of the CLAMPED geometry, i.e. the one
                        # that DOES realise bonds0 (the rings are closed) but is metastable: releasing
                        # the restraints lets it fall back. It is the natural starting point for an
                        # alternating search -- physics closes the rings, the corrector then moves the
                        # global conformer, physics closes them again, and so on.


def _cu_read_xyz(path):
    with open(path) as f:
        L = f.read().split("\n")
    n = int(L[0].split()[0]); z = []; c = []
    for ln in L[2:2 + n]:
        pp = ln.split(); z.append(_PT.GetAtomicNumber(pp[0])); c.append([float(pp[1]), float(pp[2]), float(pp[3])])
    return np.array(z), np.array(c, float)


def _cu_xtbopt(wd, xtb_bin, extra, inp=None, timeout=180):
    args = [xtb_bin, "m.xyz", "--chrg", "0", "--namespace", "m", "--parallel", "1"] + extra
    if inp:
        args += ["--input", inp]
    env = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
    try:
        r = subprocess.run(args, cwd=wd, capture_output=True, text=True, timeout=timeout, env=env)
    except Exception:
        return None
    conv = "GEOMETRY OPTIMIZATION CONVERGED" in (r.stdout or "")
    m = re.search(r"total energy gain.*?(-?[\d.]+)\s+kcal/mol", r.stdout or "")
    eg = float(m.group(1)) if m else None
    optf = os.path.join(wd, "m.xtbopt.xyz")
    if not os.path.exists(optf):
        return None
    try:
        z, c = _cu_read_xyz(optf)
    except Exception:
        return None
    return conv, z, c, eg


def _shape_reward(strain_pa, rmsd_heavy):
    """XTP 成功分子の報酬: R_XTB(=0.6, 成功の下限) から R_XVR(=1.0) の間を strain と RMSD で変調。
    掛け算なので下限 0.6 を割らない -> FAIL_CREDIT=0 が作る「成功/失敗の 0.6 ギャップ」を保つ。"""
    shape = 1.0
    if XVR_ESTRAIN_TAU > 0:
        if strain_pa is None:
            return R_XVR
        shape *= math.exp(-float(strain_pa) / XVR_ESTRAIN_TAU)
    if XVR_RMSD_RHO > 0 and rmsd_heavy:
        shape *= math.exp(-float(rmsd_heavy) / XVR_RMSD_RHO)
    return R_XTB + (R_XVR - R_XTB) * shape


def _clamp_unclamp(mb_hprerelax, ref, anums, na, idx, workdir, xtb_bin, E_start=None, init_heavy=None):
    """「bonds0 を保ったまま安定に緩和できるか」を試す（案B の realizable 判定）:
    ref(=bonds0 の実結合)を距離拘束して relax (clamp) → 拘束を外して再 relax (unclamp) →
    最終幾何が bonds0 を実現していれば realizable。
      成功 -> xtb_relax 互換の dict（ok/opt_xyz/opt_heavy_coords/e_full/e_gain/rmsd_heavy/strain_pa/opt_heavy）
      失敗 -> 理由文字列（F2_* = clamp 段階 / F3_* = unclamp 段階）
    E_start (kcal/mol, H-prerelax 構造のエネルギー) を渡すと strain_pa を
    (E_start - E_final)/na で返す = 全分子共通の基準点。渡さなければ従来どおり unclamp の利得のみ。"""
    m = Chem.MolFromMolBlock(mb_hprerelax, sanitize=False, removeHs=False)
    if m is None or m.GetNumConformers() == 0:
        return "F2_clamp_parse"
    cf = m.GetConformer()
    z = np.array([a.GetAtomicNum() for a in m.GetAtoms()])
    c = np.array([[cf.GetAtomPosition(i).x, cf.GetAtomPosition(i).y, cf.GetAtomPosition(i).z]
                  for i in range(m.GetNumAtoms())], float)
    wd = os.path.join(workdir, "clamp_%d" % idx)
    shutil.rmtree(wd, ignore_errors=True); os.makedirs(wd, exist_ok=True)
    try:
        _cu_write_xyz(os.path.join(wd, "m.xyz"), z, c)
        hc = c[:na]                                                      # heavy 座標 (heavy-first)
        # 目標距離 d0 は最初の幾何から一度だけ決めて固定する（fade 中は k のみ下げる）
        _npull = 0
        targets = []
        for (i, j) in sorted(ref):
            d = float(np.linalg.norm(hc[i] - hc[j]))
            if XVR_CLAMP_IDEAL:
                csum = COV.get(int(anums[i]), 0.75) + COV.get(int(anums[j]), 0.75)
                if d > 1.3 * csum:                                       # 伸びて結合していない -> 結合距離へ引き寄せる
                    targets.append((i, j, csum)); _npull += 1
                    continue
            targets.append((i, j, d))                                    # heavy-first: xyz idx = heavy idx + 1
        if _npull:
            _CLAMP_STATS["ideal_pulled"] = _CLAMP_STATS.get("ideal_pulled", 0) + _npull

        def _write_inp(fc):
            with open(os.path.join(wd, "c.inp"), "w") as f:
                f.write("$constrain\n force constant=%s\n" % fc)
                for (i, j, d0) in targets:
                    f.write(" distance: %d,%d,%.4f\n" % (i + 1, j + 1, d0))
                f.write("$end\n")

        fcs = [s for s in XVR_CLAMP_FADE.split(",") if s.strip()] or [XVR_CLAMP_FC]
        cz = cc = None
        for _si, fc in enumerate(fcs):                                   # 拘束段: 力の定数を段階的に下げる
            _write_inp(fc.strip())
            _last = (_si == len(fcs) - 1)
            _lvl = ["--opt"] if (_last and not XVR_CLAMP_LOOSE) else ["--opt", "loose"]   # 途中段は粗く; LOOSE 時は最終拘束段も粗く（速度）。unclamp は下で full
            rc = _cu_xtbopt(wd, xtb_bin, ["--gfn", "2"] + _lvl, inp="c.inp")
            if rc is None:
                _CLAMP_STATS["fade_fail_stage_%d" % _si] = _CLAMP_STATS.get("fade_fail_stage_%d" % _si, 0) + 1
                return "F2_clamp_fail"                                    # bonds0 を保つ幾何に到達不能
            _cv, cz, cc, _ = rc
            if not _cv and _last:
                _CLAMP_STATS["clamp_nonconv"] = _CLAMP_STATS.get("clamp_nonconv", 0) + 1  # 情報として記録（続行）
            _cu_write_xyz(os.path.join(wd, "m.xyz"), cz, cc)
        if os.environ.get("CLAMP_KEEP") == "1" and cz is not None:
            _CU_CLAMP_CACHE[idx] = (cz.tolist(), cc.tolist())               # bonds0 IS realised here
        ru = _cu_xtbopt(wd, xtb_bin, ["--gfn", "2", "--opt"])                # k=0: 拘束なし free relax（合格証明はここ）
        if ru is None:
            return "F3_unclamp_fail"
        uconv, uz, uc, ueg = ru
        if not uconv:
            return "F3_unclamp_nonconv"
        heavy = uc[uz != 1]
        if len(heavy) != na:
            return "F3_atom_mismatch"
        if _heavy_conn(anums, heavy, na) != ref:
            if os.environ.get("FLIP_KEEP") == "1":                          # diagnostics: keep the FLIPPED geometry
                _CU_FLIP_CACHE[idx] = (uz.tolist(), uc.tolist())            # (all atoms incl. H) so the mechanism
            return "F3_unclamp_flip"                                        # 解放で bonds0 が壊れる = 準安定・実現不可
        optf = os.path.join(wd, "m.xtbopt.xyz")
        E_final = _cu_energy_kcal(optf)                                     # unclamp 後の生 GFN2 全エネルギー (拘束バイアス無し)
        try:
            with open(optf) as f:
                opt_xyz_txt = f.read()                                      # H整合チェック #2 用 (H が離脱していないか)
        except Exception:
            opt_xyz_txt = ""
        if E_start is not None and E_final is not None:
            strain = abs(E_start - E_final) / na                            # 統一基準: H-prerelax 構造 -> 最終極小
        else:
            strain = abs(ueg) / na if ueg is not None else None             # 旧: unclamp 区間の利得のみ (過小評価)
        rmsd_h = None
        if init_heavy is not None:
            try:
                rmsd_h = _kabsch_rmsd(init_heavy, heavy)   # 生成重原子 -> 最終極小 の積算誤差
            except Exception:
                rmsd_h = None
        return {"ok": True, "realizable": True, "strain_pa": strain, "opt_heavy": heavy.tolist(),
                "opt_heavy_coords": heavy.tolist(), "opt_xyz": opt_xyz_txt,
                "e_full": E_final, "e_gain": (None if (E_start is None or E_final is None) else (E_final - E_start)),
                "rmsd_heavy": rmsd_h, "clamped": True}
    finally:
        shutil.rmtree(wd, ignore_errors=True)


def _xtb_reward(p, xtb_bin, workdir, idx, collect_relax, use_clashvr):
    out = {"reward": R_FAIL, "rdkit_ok": False, "xtb_ok": False, "same_topo": False,
           "smi": "", "topo_post": "", "strain_pa": None, "clash_pass": False}
    if p is None:
        return out                                                 # disconnected/completer/placer -> 0.0（情報なし）
    if p.get("screened"):                                          # clash で screen（reward は 0 のまま。段階罰は batch 側）
        for _k in ("screened", "clash_n", "clash_depth", "clash_max", "selfmis", "selfmis_miss", "selfmis_spur", "nbond0", "na"):
            out[_k] = p.get(_k)
        return out
    out["rdkit_ok"] = True; out["clash_pass"] = True               # clash-pass + completer (clashVR)
    out["placer"] = p.get("placer"); out["smi"] = p.get("smi", "")
    # 自己整合ミス（xtb 不要）は screen を通った全分子に付ける -> xtb 非収束でもペナルティ勾配が届く
    out["selfmis"] = p.get("selfmis"); out["nbond0"] = p.get("nbond0"); out["na"] = p["na"]
    out["selfmis_miss"] = p.get("selfmis_miss"); out["selfmis_spur"] = p.get("selfmis_spur")
    if use_clashvr:
        out["reward"] = R_CLASH                                     # 0.3 clashVR credit
    hprerelax_ok = None; E_hpre = None; res = None; n_corr = 0
    mb_start = p["mb"]                                              # placement struct; re-stripped + restarted on fragmentation
    mb_use = mb_start
    _where = None                                                  # 離脱Hが検出された段階 ("prerelax" / "full")
    for _att in range(1 + H_INTEGRITY_MAX_RETRY):                  # ストッパー: H prerelax へ戻れるのは最大 H_INTEGRITY_MAX_RETRY(既定3) 回
        mb_use = mb_start
        if H_PRERELAX:                                             # H-only prerelax (freeze heavy) -> isolate heavy strain
            try:
                mb_h, E_hpre = xtb_hrelax(idx, mb_start, p["na"], workdir, xtb_bin, charge=0)
            except Exception:
                mb_h, E_hpre = None, None
            hprerelax_ok = mb_h is not None
            out["hprerelax_ok"] = hprerelax_ok; out["E_hprerelax"] = E_hpre
            if mb_h is None:
                return out                                         # H-prerelax REQUIRED (no fallback): drop here
            if H_INTEGRITY and not _h_intact(mb_h):                # CHECK #1 (end of H-prerelax): detached/stray H = excess H ejected by xTB
                _INTEGRITY_STATS["n"] += 1
                mb_s, nrem = _strip_detached_h(mb_h) if H_INTEGRITY_MODE == "correct" else (None, 0)
                if mb_s is not None:
                    mb_start = mb_s; n_corr += nrem; _INTEGRITY_STATS["correct"] += 1
                    _where = "prerelax"; _MLNH_PERF["at_prerelax"] += 1
                    continue                                       # 余剰H(偶数個)を除去 -> H-prerelax からやり直し
                out["h_intact"] = False; _INTEGRITY_STATS["reject"] += 1
                _INTEGRITY_STATS["F1a_odd_parity"] += 1; out["F"] = "F1a_odd_parity"   # 奇数個除去が必要=ラジカル化 -> reject
                return out
            mb_use = mb_h                                          # heavy frozen; H at optimum -> full relax start point
        if XVR_CLAMP_ONLY:                                          # free relax を廃し、最初から bonds0 拘束 -> 解放
            _rz = _clamp_unclamp(mb_use, p["ref"], p["anums"], p["na"], idx, workdir, xtb_bin,
                                 E_start=(E_hpre if XVR_STRAIN_HPRE else None), init_heavy=p["init_heavy"])
            _CLAMP_STATS["tried"] += 1
            if not isinstance(_rz, dict):                           # F2_* (clamp 段階) / F3_* (unclamp 段階)
                _CLAMP_STATS[_rz] = _CLAMP_STATS.get(_rz, 0) + 1
                out["F"] = _rz
                out["xtb_ok"] = _rz.startswith("F3")                # F3 = clamp 幾何は得られた -> xtb は緩和できている
                out["reward"] = XVR_FAIL_CREDIT if out["xtb_ok"] else R_FAIL
                if os.environ.get("CLAMP_KEEP") == "1" and idx in _CU_CLAMP_CACHE:
                    _cz, _cc = _CU_CLAMP_CACHE.pop(idx)
                    _cz = np.asarray(_cz); _cc = np.asarray(_cc)
                    out["clamp_heavy"] = _cc[_cz != 1][:p["na"]].tolist()   # the geometry that DOES close
                    out["clamp_all_z"] = _cz.tolist()                       # the rings (metastable)
                    out["clamp_all_xyz"] = _cc.tolist()
                if os.environ.get("FLIP_KEEP") == "1" and idx in _CU_FLIP_CACHE:
                    _fz, _fc = _CU_FLIP_CACHE.pop(idx)              # diagnostics: the structure xTB actually
                    out["flip_anums"] = _fz; out["flip_coords"] = _fc   # relaxed this FAILED molecule to
                return out
            res = _rz
        else:
            try:
                res = xtb_relax(idx, mb_use, p["init_heavy"], workdir, xtb_bin, charge=0)
            except Exception:
                res = {"ok": False}
        if not res.get("ok"):
            return out                                             # clashVR: 0.3 / pure XVR: 0.0
        if H_INTEGRITY:                                            # CHECK #2 (end of full relax, REQUIRED): full relax moves everything -> H can still detach
            _ra, _rc = _parse_xyz_all(res.get("opt_xyz", ""))
            if _ra is not None and not _intact_zc(_ra, _rc)[0]:
                _INTEGRITY_STATS["n_full"] += 1
                mb_s, nrem = _strip_detached_zc(_ra, _rc) if H_INTEGRITY_MODE == "correct" else (None, 0)
                if mb_s is not None:
                    mb_start = mb_s; n_corr += nrem; _INTEGRITY_STATS["correct_full"] += 1
                    _where = "full"; _MLNH_PERF["at_full"] += 1
                    continue                                       # 余剰H(偶数個)を除去 -> H-prerelax からやり直し
                out["h_intact"] = False; _INTEGRITY_STATS["reject_full"] += 1
                _INTEGRITY_STATS["F1a_odd_parity"] += 1; out["F"] = "F1a_odd_parity"
                return out
        break                                                      # intact at BOTH H-prerelax and full relax
    else:
        _INTEGRITY_STATS["F1b_retry_exhausted"] += 1               # ストッパー作動: 上限まで戻しても intact にできず
        out["F"] = "F1b_retry_exhausted"; out["n_strip"] = n_corr; out["n_attempt"] = H_INTEGRITY_MAX_RETRY
        return out
    out["h_intact"] = True
    out["n_strip"] = n_corr                                        # 弾かれた H の総数(必ず偶数)
    out["n_attempt"] = _att                                        # H-prerelax へ戻った回数 (0 = 一発で intact)
    out["where"] = _where                                          # 離脱が検出された段階 (prerelax / full / None)
    _MLNH_PERF["strip_hist"][n_corr] = _MLNH_PERF["strip_hist"].get(n_corr, 0) + 1
    _MLNH_PERF["attempt_hist"][_att] = _MLNH_PERF["attempt_hist"].get(_att, 0) + 1
    if n_corr == 0:
        _MLNH_PERF["ok_first"] += 1                                # strip 無しで成功 = MLnH の H 数がそのまま通った
    if n_corr:
        out["h_corrected"] = n_corr
    out["xtb_ok"] = True; out["reward"] = XVR_FAIL_CREDIT           # 「緩和はできたが bonds0 未実現」の暫定値。成功なら下で上書き
    eg = res.get("e_gain"); na = p["na"]
    if XVR_STRAIN_HPRE and E_hpre is not None and res.get("e_full") is not None:
        strain_pa = abs(E_hpre - res["e_full"]) / na                # 統一基準: H-prerelax 構造 -> 最終極小 (free/clamp 共通)
    elif eg is not None:
        strain_pa = abs(eg) / na
    else:
        strain_pa = res.get("strain_pa")                            # clamp-only かつ基準統一 off: unclamp 区間の利得（過小評価）
    out["strain_pa"] = strain_pa
    out["clamped"] = bool(res.get("clamped"))
    out["rmsd_heavy"] = res.get("rmsd_heavy")                      # 積算誤差の直接量: 緩和で heavy がどれだけ動いたか
    if _bank_struct():                                             # freeze HADD + 緩和 (all-atom) + funnel record
        out["e_gain"] = eg                                         # (hprerelax_ok/E_hprerelax set above, pre-gate)
        out["E_full"] = res.get("e_full")
        out["nH"] = p.get("nH"); out["bonds0"] = p.get("bonds0")
        out["rmsd_heavy"] = res.get("rmsd_heavy")
        _him = Chem.MolFromMolBlock(mb_use, sanitize=False)        # mb_use = H-prerelaxed struct = E_hprerelax の構造 = full relax の起点
        if _him is not None:
            _hc = _him.GetConformer()
            out["hadd_anums"] = [a.GetAtomicNum() for a in _him.GetAtoms()]
            out["hadd_coords"] = np.array([[_hc.GetAtomPosition(i).x, _hc.GetAtomPosition(i).y, _hc.GetAtomPosition(i).z]
                                           for i in range(_him.GetNumAtoms())], np.float32)
        ra, rc = _parse_xyz_all(res.get("opt_xyz", ""))            # full relaxed geometry (heavy+H)
        out["relaxed_anums"] = ra; out["relaxed_coords"] = rc
    if collect_relax:
        out["e_gain"] = eg; out["rmsd_all"] = res.get("rmsd"); out["rmsd_heavy"] = res.get("rmsd_heavy")
    oc = res.get("opt_heavy_coords")
    if oc and len(oc) == na:
        post = _heavy_conn(p["anums"], np.asarray(oc, float), na)   # 緩和後の距離トポロジー
        out["same_topo_old"] = (post == p.get("ref_dist"))          # 旧定義(距離 ref)の XTP: 比較・併記用のみ
        out["bonds0_subset"] = set(p["ref"]) <= post                # 緩め基準(ii): bonds0 の結合が全て在るか(余分な接触は許容)
        if post == p["ref"]:                                        # 厳密基準(i): 緩和後の距離トポロジー == bonds0
            out["same_topo"] = True
            if res.get("clamped"):                                  # clamp 経由で到達した bonds0 極小（clamp-only / fallback 共通）
                _CLAMP_STATS["rescued"] += 1
                out["rescued"] = True; out["realizable_heavy"] = oc   # IKT 教師 (gen_heavy -> realizable_heavy)
            out["reward"] = _shape_reward(strain_pa, out.get("rmsd_heavy"))
        elif XVR_CLAMP:                                            # bonds0 を実現できず -> bonds0 を拘束して clamp->unclamp で救済
            _CLAMP_STATS["tried"] += 1
            rz = _clamp_unclamp(mb_use, p["ref"], p["anums"], na, idx, workdir, xtb_bin,
                                E_start=(E_hpre if XVR_STRAIN_HPRE else None), init_heavy=p["init_heavy"])
            if isinstance(rz, dict):                              # bonds0 を保つ安定極小が存在 = realizable
                _CLAMP_STATS["rescued"] += 1
                out["same_topo"] = True; out["rescued"] = True
                out["bonds0_subset"] = True                       # 最終採用幾何(unclamp)で bonds0 を実現しているので (ii) も成立
                out["realizable_heavy"] = rz["opt_heavy"]         # IKT 教師 (gen_heavy -> realizable_heavy), on-distribution
                out["strain_pa"] = rz.get("strain_pa"); out["rmsd_heavy"] = rz.get("rmsd_heavy")
                out["reward"] = _shape_reward(rz.get("strain_pa"), rz.get("rmsd_heavy"))
            else:
                out["F"] = rz or "F3_not_realizable"              # F2_clamp_* / F3_unclamp_* （_clamp_unclamp が理由を返す）
                if os.environ.get("FLIP_KEEP") == "1" and idx in _CU_FLIP_CACHE:
                    _fz, _fc = _CU_FLIP_CACHE.pop(idx)                # the geometry xTB actually relaxed to
                    out["flip_anums"] = _fz; out["flip_coords"] = _fc  # -> lets us diff the graph and see WHAT changed
                _CLAMP_STATS[out["F"]] = _CLAMP_STATS.get(out["F"], 0) + 1
        else:
            out["F"] = "F3_not_realizable"                        # clamp 無効時: bonds0 を実現できなかった
    return out


def pfree_reward_batch(mols, xtb_bin, workdir, max_workers=16, collect_relax=False):
    global _clash_ema, _switched
    if PFREE_DUMP:                                                  # 採点方式の A/B 比較用に生成分子を素の形で吐く
        import pickle
        with open(PFREE_DUMP, "ab") as _f:
            for (atoms, bonds, na) in mols:
                if not na or na < 3:
                    continue
                pickle.dump(([int(atoms[k].atomic_num) for k in range(na)],
                             [list(map(float, atoms[k].pos)) for k in range(na)],
                             [(int(a), int(b)) for (a, b) in bonds], int(na)), _f)
    os.makedirs(workdir, exist_ok=True)
    results = [None] * len(mols)
    prep = [None] * len(mols)
    n_proc = 0
    for i, (atoms, bonds, na) in enumerate(mols):                   # MAIN-thread completer pre-pass
        if na and na >= 3:
            n_proc += 1
            try:
                prep[i] = _prep(atoms, bonds, na)
            except Exception:
                prep[i] = None
    # clashVR auto-switch: clash-pass EMA >= threshold -> drop 0.3 tier -> pure XVR
    n_valid = sum(1 for p in prep if p is not None)
    rate = n_valid / max(n_proc, 1)
    _clash_ema = rate if _clash_ema is None else 0.9 * _clash_ema + 0.1 * rate
    if CLASHVR and not _switched and _clash_ema >= CLASHVR_SWITCH:
        _switched = True
        print("[CLASHVR] clash-pass EMA %.3f >= %.2f -> SWITCH to pure XVR (drop 0.3 tier)"
              % (_clash_ema, CLASHVR_SWITCH), flush=True)
    use_clashvr = CLASHVR and not _switched

    def _task(i):
        return i, _xtb_reward(prep[i], xtb_bin, workdir, i, collect_relax, use_clashvr)

    with ThreadPoolExecutor(max_workers=max_workers) as ex:         # parallel xTB
        for i, r in ex.map(_task, range(len(mols))):
            results[i] = r
    if os.environ.get("XVR_ESTRAIN_MEASURE") == "1":
        try:
            _sp = [r["strain_pa"] for r in results if r and r.get("strain_pa") is not None]
            if _sp:
                _a = np.asarray(_sp, float)
                print("[ESTRAIN] n=%d/%d strain_pa median=%.3f mean=%.3f p90=%.3f max=%.3f"
                      % (len(_a), len(results), np.median(_a), _a.mean(),
                         np.percentile(_a, 90), _a.max()), flush=True)
        except Exception:
            pass
    if MLNH_PARITY and _PARITY_STATS["odd"]:
        s = _PARITY_STATS
        print("[MLNH_PARITY] odd-parity corrected %d/%d calls (remove %d / add %d / unfixed %d)"
              % (s["odd"], s["n"], s["remove"], s["add"], s["fail"]), flush=True)
    if H_INTEGRITY and (_INTEGRITY_STATS["n"] or _INTEGRITY_STATS["n_full"]):
        s = _INTEGRITY_STATS
        print("[H_INTEGRITY] detached-H at H-prerelax: %d (corrected %d / rejected %d) | at full-relax: %d (corrected %d / rejected %d)"
              % (s["n"], s["correct"], s["reject"], s["n_full"], s["correct_full"], s["reject_full"]), flush=True)
    if XVR_CLAMP and _CLAMP_STATS["tried"]:
        s = _CLAMP_STATS
        print("[XVR_CLAMP] bonds0-clamp rescued (clamp->unclamp realizable): %d/%d tried | fail: %s"
              % (s["rescued"], s["tried"],
                 {k: v for k, v in s.items() if k.startswith(("F2", "F3", "clamp_"))}), flush=True)
    # --- 積算誤差ペナルティ: 自己整合ミス(xtb不要・全分子) + 緩和変位RMSD(xtb収束分子のみ) ---
    if XVR_SELFMIS_LAM > 0 or XVR_RMSD_LAM > 0 or XVR_CLASH_LAM > 0:
        for r in results:
            if not r or r.get("selfmis") is None:
                continue                                            # 情報の無い screen 落ち(p=None)はそのまま 0
            if r.get("same_topo"):
                continue                     # 成功側は shaping(strain,rmsd) で格付け済み。減算は 0.6 のギャップを潰す
            pen = 0.0
            if XVR_SELFMIS_LAM > 0:
                pen += XVR_SELFMIS_LAM * (r["selfmis"] / max(int(r.get("nbond0") or 1), 1))
            if XVR_RMSD_LAM > 0 and r.get("rmsd_heavy"):
                pen += XVR_RMSD_LAM * float(r["rmsd_heavy"])
            if XVR_CLASH_LAM > 0 and r.get("clash_depth"):          # clash 段階罰（崖の最大失敗群に密な勾配）
                pen += XVR_CLASH_LAM * (float(r["clash_depth"]) / max(int(r.get("na") or 1), 1))
            r["reward"] = r["reward"] - pen                          # 負値も許容(baseline が吸収; 最悪分子に最強の押し)
    # --- clash の重篤度分布（λ_c のスケール決定用。xtb 不要） ---
    _cd = [r for r in results if r and r.get("clash_depth") is not None]
    if _cd:
        _dep = np.asarray([float(r["clash_depth"]) for r in _cd])
        _dpa = np.asarray([float(r["clash_depth"]) / max(int(r.get("na") or 1), 1) for r in _cd])
        _cn = np.asarray([int(r["clash_n"]) for r in _cd])
        print("[CLASH] clash分子 %d/%d | 衝突対 mean=%.1f max=%d | Σめり込み深さ(Å) mean=%.2f p90=%.2f max=%.2f | 深さ/原子 mean=%.4f p90=%.4f"
              % (len(_cd), len(results), float(_cn.mean()), int(_cn.max()), float(_dep.mean()),
                 float(np.percentile(_dep, 90)), float(_dep.max()), float(_dpa.mean()), float(np.percentile(_dpa, 90))), flush=True)
    # --- 自己整合の統計（積算誤差の直接指標, xtb 不要）: 生成幾何が bonds0 を実現できているか ---
    _sm = [int(r["selfmis"]) for r in results if r and r.get("selfmis") is not None]
    if _sm:
        _a = np.asarray(_sm)
        _ms = np.asarray([int(r["selfmis_miss"]) for r in results if r and r.get("selfmis") is not None])
        _sp = np.asarray([int(r["selfmis_spur"]) for r in results if r and r.get("selfmis") is not None])
        _nb = np.asarray([max(int(r.get("nbond0") or 1), 1) for r in results if r and r.get("selfmis") is not None])
        print("[SELFCONSIST] 生成幾何が bonds0 を実現: %d/%d (%.1f%%) | |Δ| mean=%.2f p90=%.0f max=%.0f | "
              "内訳: 伸びて切れた結合 mean=%.2f (>0 の分子 %.1f%%) / 偽接触 mean=%.2f (>0 の分子 %.1f%%) | |Δ|/|bonds0| mean=%.4f p90=%.4f"
              % (int((_a == 0).sum()), len(_a), 100 * float((_a == 0).mean()), float(_a.mean()),
                 float(np.percentile(_a, 90)), int(_a.max()),
                 float(_ms.mean()), 100 * float((_ms > 0).mean()), float(_sp.mean()), 100 * float((_sp > 0).mean()),
                 float((_a / _nb).mean()), float(np.percentile(_a / _nb, 90))), flush=True)
    # --- 失敗が「どの段階」で起きたか（崖の主因を特定する）---
    # screened      : _prep で落ちた (連結性 / clash / completer nH / mlhadd の H 付与失敗)
    # hprerelax_fail: H は付いたが、heavy 凍結で H だけ緩和しても収束しない (骨格が酷すぎて H の置き場が無い)
    # relax_fail    : H 付与も H 緩和も通過。**全部動かす full relax が非収束** = 重原子骨格の積算誤差が主因
    # not_realizable: full relax は収束したが bonds0 を実現できず (F3)
    _stage = {"ok": 0, "not_realizable": 0, "relax_fail": 0, "hprerelax_fail": 0, "F1a": 0, "F1b": 0, "screened": 0}
    _stage_sm = {k: [] for k in _stage}                              # 段階別の自己整合ミス |Δ|
    for r in results:
        if not r:
            continue
        if r.get("same_topo"):
            _b = "ok"
        elif not r.get("rdkit_ok"):
            _b = "screened"
        elif r.get("F") == "F1a_odd_parity":
            _b = "F1a"
        elif r.get("F") == "F1b_retry_exhausted":
            _b = "F1b"
        elif r.get("hprerelax_ok") is False:
            _b = "hprerelax_fail"
        elif not r.get("xtb_ok"):
            _b = "relax_fail"
        else:
            _b = "not_realizable"
        _stage[_b] += 1
        if r.get("selfmis") is not None:
            _stage_sm[_b].append(int(r["selfmis"]))
    print("[FAILSTAGE] %s   ※relax_fail = H付与もH緩和も成功したが full relax が非収束 = 重原子幾何の積算誤差" % _stage, flush=True)
    # 崖では screened が最大の失敗群。その内訳（累積）: どの screen が効いているか
    if sum(_SCREEN_STATS.values()):
        print("[SCREEN] 内訳(累積): %s   ※これらは _prep で落ちるため selfmis も付かず reward 0（勾配ゼロ）" % _SCREEN_STATS, flush=True)
    # ★λ1 が relax_fail を狙えるかの決定的診断: 各段階の |Δ| 平均と「|Δ|>0 の割合」
    print("[FAILSTAGE|Δ] 段階別 自己整合ミス |Δ| mean (n, |Δ|>0の割合): %s" %
          {k: (round(float(np.mean(v)), 2), len(v), "%.0f%%" % (100 * float(np.mean(np.asarray(v) > 0))))
           for k, v in _stage_sm.items() if v}, flush=True)
    # --- 案B: 新旧 XTP の併記 + 失敗内訳 (F1a/F1b/F2/F3) ---
    _n = len(results)
    _new = sum(1 for r in results if r and r.get("same_topo"))
    _old = sum(1 for r in results if r and r.get("same_topo_old"))
    _loose = sum(1 for r in results if r and r.get("bonds0_subset"))
    _F = {}
    for r in results:
        if r and r.get("F"):
            _F[r["F"]] = _F.get(r["F"], 0) + 1
    print("[XTP] bonds0-厳密(i)=%d/%d (%.1f%%) | bonds0-緩め(ii)=%d (%.1f%%) | 旧定義(距離ref)=%d (%.1f%%) | 失敗内訳 %s"
          % (_new, _n, 100 * _new / max(_n, 1), _loose, 100 * _loose / max(_n, 1),
             _old, 100 * _old / max(_n, 1), _F), flush=True)
    # --- サイズ別 XTP: 母集団平均は 40+ テールに引かれるので、サイズ帯ごとに割る ---
    _BUCKETS = [(0, 24), (25, 29), (30, 34), (35, 39), (40, 999)]
    _sz_tot = {b: 0 for b in _BUCKETS}
    _sz_ok = {b: 0 for b in _BUCKETS}
    for r in results:
        if not r or r.get("na") is None:
            continue
        _na = int(r["na"])
        for (lo, hi) in _BUCKETS:
            if lo <= _na <= hi:
                _sz_tot[(lo, hi)] += 1
                if r.get("same_topo"):
                    _sz_ok[(lo, hi)] += 1
                break
    _parts = []
    for (lo, hi) in _BUCKETS:
        t = _sz_tot[(lo, hi)]
        if t == 0:
            continue
        _lab = "%d-%d" % (lo, hi) if hi < 999 else "%d+" % lo
        _parts.append("%s=%d/%d(%.0f%%)" % (_lab, _sz_ok[(lo, hi)], t, 100 * _sz_ok[(lo, hi)] / t))
    print("[XTP|size] " + " | ".join(_parts), flush=True)
    _ct={}; _co={}
    for r in results:
        if not r or r.get("na") is None:
            continue
        _n=int(r["na"]); _ct[_n]=_ct.get(_n,0)+1
        if r.get("same_topo"): _co[_n]=_co.get(_n,0)+1
    print("[XTP|count] "+" ".join("%d:%d/%d"%(k,_co.get(k,0),_ct[k]) for k in sorted(_ct)), flush=True)
    _sp = sorted(float(r["strain_pa"]) for r in results if r and r.get("same_topo") and r.get("strain_pa") is not None)
    if _sp:
        def _q(f):
            return _sp[min(len(_sp) - 1, int(f * len(_sp)))]
        _rw = [float(r["reward"]) for r in results if r and r.get("same_topo")]
        _rm = sorted(float(r["rmsd_heavy"]) for r in results
                     if r and r.get("same_topo") and r.get("rmsd_heavy"))
        _rs = ("| 緩和変位RMSD(Kabsch,heavy) mean=%.3f p50=%.3f p90=%.3f max=%.3f Å"
               % (sum(_rm) / len(_rm), _rm[len(_rm) // 2], _rm[min(len(_rm) - 1, int(.9 * len(_rm)))], _rm[-1])) if _rm else ""
        print("[STRAIN] XTP成功分子の strain/heavy (kcal/mol/atom, 基準=%s): mean=%.2f p50=%.2f p90=%.2f max=%.2f | 報酬 mean=%.3f min=%.3f %s"
              % ("H-prerelax" if XVR_STRAIN_HPRE else "緩和区間の利得",
                 sum(_sp) / len(_sp), _q(0.5), _q(0.9), _sp[-1],
                 sum(_rw) / max(len(_rw), 1), min(_rw) if _rw else 0.0, _rs), flush=True)
    # --- MLnH(+MLHplacer) 性能: 何個(偶数)・何回 弾いてから成功したか ---
    m = _MLNH_PERF
    if m["ok_first"] or m["strip_hist"]:
        _tot = sum(m["strip_hist"].values()) or 1
        print("[MLNH_PERF] strip無しで成功=%d/%d (%.1f%%) | strip分布(H数:件)=%s | retry分布(回:件)=%s | 検出段階 prerelax=%d full=%d | 奇数→追加除去=%d | F1a_odd_reject=%d F1b_exhausted=%d"
              % (m["ok_first"], _tot, 100 * m["ok_first"] / _tot, dict(sorted(m["strip_hist"].items())),
                 dict(sorted(m["attempt_hist"].items())), m["at_prerelax"], m["at_full"], m["odd_extra_removed"],
                 _INTEGRITY_STATS["F1a_odd_parity"], _INTEGRITY_STATS["F1b_retry_exhausted"]), flush=True)
    return results


def _completer_mol_stable(anums, coords, bonds0):
    """Perception-free mol_stable (HADD MVR): force completer n_H -> RDKit SanitizeMol succeeds.
    completer decides n_H (learned, not RDKit valence-from-geometry); RDKit only checks consistency +
    kekulizes. Recovers aromatics that RDKit-own kekulize drops. Returns (ok: bool, smi: str)."""
    nH = _completer_nH(anums, coords, bonds0)
    if nH is None:
        return False, ""
    try:
        rw = Chem.RWMol()
        for i, z in enumerate(anums):
            a = Chem.Atom(int(z)); a.SetNumExplicitHs(int(nH[i])); a.SetNoImplicit(True); rw.AddAtom(a)
        for a, b in bonds0:
            rw.AddBond(int(a), int(b), Chem.BondType.SINGLE)
        m = rw.GetMol()
        Chem.SanitizeMol(m)
        return True, Chem.MolToSmiles(m)
    except Exception:
        return False, ""


def _rdkit_mol_stable(anums, coords):
    """RDKit's OWN mol_stable (validate_3D = RDKit adds H / assigns valence from geometry). For the
    RDKit-H MVR variant (MVR_HMODE=rdkit): same clash+connected screen as the completer MVR, but
    RDKit (not the completer) decides H/valence -> isolates completer-vs-RDKit at fixed screen.
    Returns (ok, smi)."""
    try:
        from util_validation import validate_3D
        mol, smi, info = validate_3D(list(anums), [list(c) for c in coords])
        if mol is not None and info.get("mol_stable", False):
            return True, (smi or "")
    except Exception:
        pass
    return False, ""


def pfree_mvr_batch(mols, xtb_bin=None, workdir=None, max_workers=None, collect_relax=False):
    """Perception-free MVR proxy (cheap dense Stage-1 signal, NO xTB): reward 1.0 iff
    clash-free + connected(distance) + mol_stable; else 0.0. mol_stable via completer n_H ->
    SanitizeMol (default, "new HADD mol_stable") OR via RDKit validate_3D (MVR_HMODE=rdkit, ablation:
    same screen, RDKit decides H). Interface matches pfree_reward_batch."""
    hmode = os.environ.get("MVR_HMODE", "completer")
    noclash = os.environ.get("MVR_NOCLASH") == "1"     # ablation: drop the clash requirement from the MVR reward only
    results = []
    for atoms, bonds, na in mols:
        out = {"reward": 0.0, "smi": "", "same_topo": False, "xtb_ok": False,
               "clash_pass": False, "rdkit_ok": False, "strain_pa": None}
        if na and na >= 3:
            anums = [atoms[k].atomic_num for k in range(na)]
            if all(a in ALLOWED_ATOMS for a in anums):
                coords = np.array([list(atoms[k].pos) for k in range(na)], dtype=np.float64)
                bonds0 = set()
                for e1, e2 in bonds:
                    a, b = int(e1) - 1, int(e2) - 1
                    if a != b and 0 <= a < na and 0 <= b < na:
                        bonds0.add((min(a, b), max(a, b)))
                bonds0 = list(bonds0)
                if _ncomp(na, list(_heavy_conn(anums, coords, na))) == 1:      # connected (distance)
                    if noclash:
                        hc = False                                             # MVR_NOCLASH: skip clash check
                    else:
                        try:
                            hc, _ = check_collisions([list(c) for c in coords], anums, set(bonds0))
                        except Exception:
                            hc = False
                    if not hc:                                                 # clash-free (or skipped)
                        out["clash_pass"] = True
                        if hmode == "rdkit":
                            ok, smi = _rdkit_mol_stable(anums, coords)         # RDKit-H ablation
                        else:
                            ok, smi = _completer_mol_stable(anums, coords, bonds0)
                        if ok:
                            out.update(reward=1.0, smi=smi, same_topo=True, rdkit_ok=True)
        results.append(out)
    return results
