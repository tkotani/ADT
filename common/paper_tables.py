"""Every generation number of the paper from the molrecord banks, with one set of definitions.

Reads <bank_dir>/<scaffold>/pfree_bank_<scaffold>.pt (written by run_gen_records.sh) for the seven
ring scaffolds and the unconditional bootstrap3 ("Btriple") and prints:

  Table 2 (tab:drugs30)        per scaffold + the average over the seven ring scaffolds + Btriple
  Table 3 (tab:divers30)       model side: N^gen, N_eff^scaf, #distinct Murcko, IntDiv_1, MW, logP, QED
  Table 4 (tab:drugs30honest)  Btriple: N^gen/N, strain/heavy percentiles, RMSD, bond/angle shift,
                               novelty, size, heavy-atom composition
  Fig. 6                       Btriple bond-length / bond-angle shift histograms (pgfplots coordinates)
  charge separation            share of N^gen molecules with an S-mediated / a non-S formal charge
  body text                    the aggregate numbers quoted in the Results section
and writes all of it to --json.

Definitions:
  N_Hprerx / N_fullrx  records whose H-prerelax / full relaxation converged
  N_XTP                XTP-accepted records (the relaxed geometry keeps the declared topology bonds0)
  N_XTP^smiles         XTP-accepted molecules that RDKit reads back as the DECLARED molecule: its with-H
                       perception of the relaxed structure gives a neutral Lewis structure AND the
                       heavy-atom bond set it perceives equals bonds0
  N^gen                distinct non-isomeric canonical SMILES among them
  N^Novel              N^gen molecules whose normalized SMILES (largest fragment -> neutralized ->
                       non-isomeric) is absent from the equally normalized GEOM-Drugs set (--geom_smi)
  RMSD                 heavy-atom Kabsch RMSD, generated (H-placed) -> relaxed
  dE                   per-molecule strain E_Hprerx - E_fullrx (kcal/mol)
  size, RMSD, dE, strain, bond/angle shifts and composition are over the XTP-accepted set;
  Table 3 properties and diversity are over the N^gen molecules.

usage: python3 common/paper_tables.py <bank_dir> --geom_smi geom_drugs.smi [--json out.json] [--procs 16]
"""
import os, sys, glob, json, random, math, argparse
from collections import Counter
from multiprocessing import Pool
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import Descriptors, Crippen, QED, AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem.MolStandardize import rdMolStandardize
RDLogger.DisableLog("rdApp.*")
from util_validation import validate_3D

RING = ["benzene_real", "pyrimidine_real", "furan_real", "pyridine_real",
        "thiophene_real", "pyrazine_real", "cyclohexane_real"]         # row order of Table 2
UNCOND = "bootstrap3"
ELEM = {6: "C", 7: "N", 8: "O", 16: "S", 9: "F", 17: "Cl", 35: "Br"}
R_CENT = np.round(np.arange(-0.155, 0.1451, 0.01), 3)                   # Fig. 6 bins
R_EDGES = np.round(np.arange(-0.16, 0.1501, 0.01), 3)
T_CENT = np.arange(-15.5, 14.51, 1.0)
T_EDGES = np.arange(-16.0, 15.01, 1.0)
KCAL_TO_MEV = 43.364

_LFC = rdMolStandardize.LargestFragmentChooser()
_UNC = rdMolStandardize.Uncharger()


def norm_smiles(m):
    """largest fragment -> neutralize -> non-isomeric canonical SMILES (both sides of novelty)."""
    try:
        return Chem.MolToSmiles(_UNC.uncharge(_LFC.choose(m)), isomericSmiles=False)
    except Exception:
        return None


def norm_ref(s):
    m = Chem.MolFromSmiles(s)
    return norm_smiles(m) if m is not None else None


def kabsch(P, Q):
    Pc, Qc = P - P.mean(0), Q - Q.mean(0)
    V, _, Wt = np.linalg.svd(Pc.T @ Qc)
    d = np.sign(np.linalg.det(Wt.T @ V.T))
    R = Wt.T @ np.diag([1, 1, d]) @ V.T
    return float(np.sqrt(np.mean(np.sum((Pc @ R.T - Qc) ** 2, axis=1))))


def per_record(r):
    """Everything the tables need from one XTP-accepted record (None if it lacks structures)."""
    si, sr = r.get("struct_init"), r.get("struct_relaxed")
    if not (isinstance(si, dict) and isinstance(sr, dict)):
        return None
    ai, ci = np.asarray(si["anums"]), np.asarray(si["coords"], float)
    ar, cr = np.asarray(sr["anums"]), np.asarray(sr["coords"], float)
    Hi, Hr, he = ci[ai > 1], cr[ar > 1], ai[ai > 1]
    if not (len(Hi) == len(Hr) == len(he) >= 3):
        return None
    out = {"na": int(r["n_heavy"]), "strain_pa": r.get("strain_pa"), "strain_dE": r.get("strain_dE"),
           "rmsd": kabsch(Hi, Hr), "elems": [ELEM.get(int(z), "other") for z in he],
           "smi": None, "nsm": None, "chg": False, "s_inv": False}
    bonds = [(int(a), int(b)) for a, b in (r.get("bonds0") or [])]
    dr, dth, adj = [], [], {}
    for i, j in bonds:
        if i < len(Hi) and j < len(Hi):
            dr.append(float(np.linalg.norm(Hr[i] - Hr[j]) - np.linalg.norm(Hi[i] - Hi[j])))
            adj.setdefault(i, []).append(j); adj.setdefault(j, []).append(i)
    for j, nb in adj.items():
        for a in range(len(nb)):
            for b in range(a + 1, len(nb)):
                p, q = nb[a], nb[b]

                def ang(X):
                    u, v = X[p] - X[j], X[q] - X[j]
                    return float(np.degrees(np.arccos(np.clip(
                        u @ v / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-9), -1, 1))))
                dth.append(ang(Hr) - ang(Hi))
    out["dr"], out["dth"] = dr, dth
    try:
        mol, _, _ = validate_3D(list(ar), cr, charge=0)
        if mol is not None:
            na = len(Hi)
            per = set()
            for bd in mol.GetBonds():
                i2, j2 = bd.GetBeginAtomIdx(), bd.GetEndAtomIdx()
                if i2 < na and j2 < na:
                    per.add((min(i2, j2), max(i2, j2)))
            if per == set((min(a, b), max(a, b)) for a, b in bonds):
                m = Chem.RemoveAllHs(mol)
                chg = [a for a in m.GetAtoms() if a.GetFormalCharge() != 0]
                out.update(smi=Chem.MolToSmiles(m, isomericSmiles=False), nsm=norm_smiles(m),
                           chg=bool(chg), s_inv=any(a.GetSymbol() == "S" for a in chg))
    except Exception:
        pass
    return out


def neff(mols):
    c = Counter(MurckoScaffold.MurckoScaffoldSmiles(mol=m) for m in mols)
    tot = sum(c.values())
    return math.exp(-sum((n / tot) * math.log(n / tot) for n in c.values())), len(c)


def intdiv(mols):
    """IntDiv_1 = 1 - mean pairwise Tanimoto, and the median pairwise Tanimoto, on a 1500-molecule sample."""
    s = random.Random(0).sample(mols, min(1500, len(mols)))
    fps = [AllChem.GetMorganFingerprintAsBitVect(m, 2, 2048) for m in s]
    allsims = []
    for i in range(len(fps)):
        allsims.extend(DataStructs.BulkTanimotoSimilarity(fps[i], fps[i + 1:]))
    if not allsims:
        return 0.0, 0.0
    return 1 - float(np.mean(allsims)), float(np.median(allsims))


def hist(centres, edges, v):
    h, _ = np.histogram(v, bins=edges)
    pct = 100.0 * h / max(h.sum(), 1)
    return " ".join("(%.3g,%.2f)" % (c, p) for c, p in zip(centres, pct))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bank_dir")
    ap.add_argument("--geom_smi", required=True, help="GEOM-Drugs SMILES, one per line (geom_smiles.py)")
    ap.add_argument("--json", default="", help="write all numbers here")
    ap.add_argument("--procs", type=int, default=16)
    args = ap.parse_args()

    banks = {}
    for f in glob.glob(os.path.join(args.bank_dir, "*", "pfree_bank_*.pt")):
        d = torch.load(f, weights_only=False)
        banks[d["scaffold"]] = d["mols"]
    have = [s for s in RING + [UNCOND] if s in banks]
    if not have:
        sys.exit("no pfree_bank_*.pt under %s" % args.bank_dir)

    ref = [l.split()[0] for l in open(args.geom_smi) if l.strip()]
    with Pool(args.procs) as p:
        geom = set(x for x in p.map(norm_ref, ref, chunksize=500) if x)
    print("GEOM reference: %d lines -> %d normalized SMILES" % (len(ref), len(geom)), flush=True)

    rows, per = {}, {}
    for s in have:
        B = banks[s]
        acc = [r for r in B if r.get("xvr")]
        with Pool(args.procs) as p:
            res = [x for x in p.map(per_record, acc, chunksize=100) if x]
        per[s] = res
        smis = [x["smi"] for x in res if x["smi"]]
        uniq = {}
        for x in res:
            if x["smi"]:
                uniq.setdefault(x["smi"], x)
        umols = [m for m in (Chem.MolFromSmiles(k) for k in uniq) if m is not None]
        sz = np.array([r["n_heavy"] for r in acc])
        ne, nmk = neff(umols)
        idv, tmed = intdiv(umols)
        props = lambda f: [f(m) for m in umols]
        MW, LP, QD = props(Descriptors.MolWt), props(Crippen.MolLogP), props(QED.qed)
        rows[s] = dict(
            N=len(B), hp=sum(1 for r in B if r.get("hprerelax_ok")), fu=sum(1 for r in B if r.get("full_ok")),
            xt=len(acc), smi=len(smis), gen=len(uniq),
            novel=sum(1 for x in uniq.values() if x["nsm"] and x["nsm"] not in geom),
            size=[float(sz.mean()), float(sz.std()), float(np.median(sz)), int(sz.min()), int(sz.max())],
            rmsd=float(np.median([x["rmsd"] for x in res])),
            de=float(np.median([r["strain_dE"] for r in acc if r.get("strain_dE") is not None])),
            neff=ne, murcko=nmk, idiv=idv, tanimoto_median=tmed,
            MW=[float(np.mean(MW)), float(np.std(MW))], logP=[float(np.mean(LP)), float(np.std(LP))],
            QED=[float(np.mean(QD)), float(np.std(QD))],
            chg_S=sum(1 for x in uniq.values() if x["chg"] and x["s_inv"]),
            chg_nonS=sum(1 for x in uniq.values() if x["chg"] and not x["s_inv"]))
        print("done", s, flush=True)

    ring = [s for s in RING if s in rows]
    A = {k: float(np.mean([rows[s][k] for s in ring])) for k in ("hp", "fu", "xt", "smi", "gen", "novel", "rmsd", "de")}
    A["size"] = [float(v) for v in np.mean([rows[s]["size"] for s in ring], axis=0)]
    A["N"] = float(np.mean([rows[s]["N"] for s in ring]))
    name = lambda s: "Btriple (uncond.)" if s == UNCOND else s.replace("_real", "")

    print("\n=== Table 2 (tab:drugs30): N_Hprerx N_XTP N_XTP^smiles N^gen(rate) N^Novel N_fullrx size RMSD dE ===")
    def t2(nm, r, bold=False):
        m_, s_, md, mn, mx = r["size"]
        return "%-18s %6.0f %6.0f %6.0f %6.0f (%.1f%%) %6.0f %6.0f  %.1f/%.1f/%.0f/%.0f/%.0f  %.2f  %.1f" % (
            nm, r["hp"], r["xt"], r["smi"], r["gen"], 100 * r["gen"] / r["N"], r["novel"], r["fu"],
            m_, s_, md, mn, mx, r["rmsd"], r["de"])
    for s in ring:
        print(t2(name(s), rows[s]))
    print(t2("average", A))
    if UNCOND in rows:
        print(t2(name(UNCOND), rows[UNCOND]))

    print("\n=== Table 3 (tab:divers30), model side: N^gen N_eff #Murcko IntDiv MW logP QED dE ===")
    for s in ring + ([UNCOND] if UNCOND in rows else []):
        r = rows[s]
        print("%-18s %6d %6.0f %6d %.3f  %.1f+-%.1f  %+.2f+-%.2f  %.3f+-%.3f  %.1f" % (
            name(s), r["gen"], r["neff"], r["murcko"], r["idiv"], *r["MW"], *r["logP"], *r["QED"], r["de"]))

    out = {"rows": rows, "average": A}
    if UNCOND in rows:
        res = per[UNCOND]
        spa = np.array([x["strain_pa"] for x in res if x["strain_pa"] is not None])
        dr = np.array([v for x in res for v in x["dr"]]); dth = np.array([v for x in res for v in x["dth"]])
        ec = Counter(e for x in res for e in x["elems"]); tot = sum(ec.values())
        sz = np.array([x["na"] for x in res]); r = rows[UNCOND]
        honest = dict(gen_rate=100 * r["gen"] / r["N"], strain_pct=[float(v) for v in np.percentile(spa, [25, 50, 75, 90])],
                      rmsd=float(np.median([x["rmsd"] for x in res])), dr=float(np.median(np.abs(dr))),
                      dth=float(np.median(np.abs(dth))), novel=100 * r["novel"] / r["gen"],
                      size=[float(sz.mean()), float(sz.std()), int(sz.max())],
                      comp={k: 100 * ec.get(k, 0) / tot for k in ("C", "N", "O", "S", "F", "Cl", "Br")},
                      fig6_dr=hist(R_CENT, R_EDGES, dr), fig6_dth=hist(T_CENT, T_EDGES, dth))
        out["honest"] = honest
        print("\n=== Table 4 (tab:drugs30honest), Btriple ===")
        print("N^gen/N = %.1f%%" % honest["gen_rate"])
        print("strain/heavy p25/p50/p75/p90 = %.2f / %.2f / %.2f / %.2f kcal/mol" % tuple(honest["strain_pct"]))
        print("Kabsch RMSD median = %.2f A;  median |dr| = %.4f A, |dtheta| = %.2f deg" % (honest["rmsd"], honest["dr"], honest["dth"]))
        print("novel = %.1f%%;  size = %.1f +- %.1f (max %d)" % (honest["novel"], *honest["size"]))
        print("C/N/O = %.2f / %.2f / %.2f;  S/F/Cl/Br = %.2f / %.2f / %.2f / %.2f (%%)" % tuple(honest["comp"][k] for k in ("C", "N", "O", "S", "F", "Cl", "Br")))
        print("\n=== Fig. 6 (Btriple), RLVR/model curve ===")
        print("dr:", honest["fig6_dr"]); print("dtheta:", honest["fig6_dth"])

    allS = ring + ([UNCOND] if UNCOND in rows else [])
    NT = sum(rows[s]["gen"] for s in allS)
    allspa = [x["strain_pa"] for s in allS for x in per[s] if x["strain_pa"] is not None]
    ringspa = [x["strain_pa"] for s in ring for x in per[s] if x["strain_pa"] is not None]
    body = dict(xtp_avg=A["xt"], xtp_rate=100 * A["xt"] / A["N"], gen_rate_avg=100 * A["gen"] / A["N"],
                gen_rate_uncond=(100 * rows[UNCOND]["gen"] / rows[UNCOND]["N"]) if UNCOND in rows else None,
                rmsd_avg=A["rmsd"], de_avg=A["de"], smi_over_xtp=100 * A["smi"] / A["xt"],
                novelty_min_ring=min(100 * rows[s]["novel"] / rows[s]["gen"] for s in ring),
                strain_pa_median_all=float(np.median(allspa)), strain_pa_median_ring=float(np.median(ringspa)),
                chg_S=100 * sum(rows[s]["chg_S"] for s in allS) / NT, chg_nonS=100 * sum(rows[s]["chg_nonS"] for s in allS) / NT,
                idiv_range=[min(rows[s]["idiv"] for s in ring), max(rows[s]["idiv"] for s in ring)],
                uniqueness=[min(100 * rows[s]["gen"] / rows[s]["smi"] for s in ring), max(100 * rows[s]["gen"] / rows[s]["smi"] for s in ring)],
                murcko_uniqueness=[min(100 * rows[s]["murcko"] / rows[s]["gen"] for s in ring), max(100 * rows[s]["murcko"] / rows[s]["gen"] for s in ring)],
                tanimoto_median=[min(rows[s]["tanimoto_median"] for s in ring), max(rows[s]["tanimoto_median"] for s in ring)])
    out["body"] = body
    print("\n=== body text ===")
    print("N_XTP average %.0f (%.1f%%);  N^gen/N average %.1f%%;  Btriple %.1f%%" % (body["xtp_avg"], body["xtp_rate"], body["gen_rate_avg"], body["gen_rate_uncond"] or float("nan")))
    print("RMSD average %.2f A;  dE average %.1f kcal/mol;  N_XTP^smiles/N_XTP %.1f%%;  novelty >= %.1f%% (ring)" % (body["rmsd_avg"], body["de_avg"], body["smi_over_xtp"], body["novelty_min_ring"]))
    print("strain/heavy median: all banks %.3f kcal/mol (%.0f meV), ring scaffolds %.3f" % (body["strain_pa_median_all"], body["strain_pa_median_all"] * KCAL_TO_MEV, body["strain_pa_median_ring"]))
    print("charge separation over %d N^gen molecules: S-mediated %.1f%%, non-S %.1f%%" % (NT, body["chg_S"], body["chg_nonS"]))
    print("IntDiv range over ring scaffolds %.3f-%.3f" % tuple(body["idiv_range"]))
    print("uniqueness N^gen/N_XTP^smiles %.1f-%.1f%%;  Murcko uniqueness #Murcko/N^gen %.0f-%.0f%%;  median pair Tanimoto %.2f-%.2f"
          % (*body["uniqueness"], *body["murcko_uniqueness"], *body["tanimoto_median"]))
    if args.json:
        json.dump(out, open(args.json, "w"), indent=1)
        print("wrote", args.json)


if __name__ == "__main__":
    main()
