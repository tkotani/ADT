"""Extract the XVR-positive SMILES shown in the paper example tables.

Table tab:gen50 (50-atom) is reproduced EXACTLY by a deterministic rule:
the LARGEST geometrically-clean XVR-positive molecule per scaffold
(heavy-atom RMSD on relaxation < 0.5 A), illustrating the larger-molecule
regime. For Table tab:gen30 (30-atom) this script lists the first 2 unique
XVR-positive SMILES per scaffold as an illustrative sample; the paper curates
its 2 examples from a larger pool for readability, so those rows are not
reproduced verbatim by the simple first-unique rule below.

dE is the GFN2-xTB energy gain on relaxation; native unit is kcal/mol
(parsed from the xTB "total energy gain ... kcal/mol" line), converted to
eV (1 eV = 23.0605 kcal/mol) for the paper, which reports energies in eV.
"""
import os as _os_adt
_HERE_adt = _os_adt.path.dirname(_os_adt.path.abspath(__file__))
ADT_ROOT = _os_adt.environ.get("ADT_ROOT", _os_adt.path.abspath(_os_adt.path.join(_HERE_adt, "..", "..")))
DRUGS_DATA = _os_adt.path.join(ADT_ROOT, "Drugs", "data", "freeorder_v26")

import json
from rdkit import Chem
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")

EV = 23.0605  # kcal/mol per eV
SCAFFOLDS = ["benzene", "pyridine", "pyrimidine", "pyrazine", "furan", "thiophene", "cyclohexane"]


def _xvr_positive(base, s):
    d = json.load(open(f"{base}/{s}/xtb_results.json"))
    return [r for r in d if r.get("sdf_tag") == "kekulized" and r.get("ok") and r.get("same_topo")]


def _fmt(smi, eg, rh, n=None):
    sz = f"{n} atoms, " if n is not None else ""
    return f"    {smi}   ({sz}dE={eg/EV:.2f} eV [={eg:.1f} kcal/mol], rmsd_heavy={rh:.3f} A)"


def first_unique(base, label, n=2):
    """Table gen30: first n unique XVR-positive SMILES per scaffold."""
    print(f"\n=== {label}: first {n} XVR-positive per scaffold (gen30 sampler, illustrative) ===")
    for s in SCAFFOLDS:
        try:
            ok = _xvr_positive(base, s)
        except FileNotFoundError:
            continue
        seen, picks = set(), []
        for r in ok:
            smi = r.get("smi")
            if smi and smi not in seen:
                seen.add(smi)
                picks.append((smi, r.get("e_gain"), r.get("rmsd_heavy")))
            if len(picks) >= n:
                break
        print(f"\n  {s}:")
        for smi, eg, rh in picks:
            print(_fmt(smi, eg, rh))


def largest_clean(base, label, rmsd_cap=0.5):
    """Table gen50: largest XVR-positive molecule per scaffold with
    heavy-atom RMSD < rmsd_cap (tie-break: smaller RMSD)."""
    print(f"\n=== {label}: largest clean (rmsd<{rmsd_cap} A) XVR-positive per scaffold (Table gen50) ===")
    for s in SCAFFOLDS:
        try:
            ok = _xvr_positive(base, s)
        except FileNotFoundError:
            continue
        seen, best = set(), None
        for r in ok:
            smi = r.get("smi"); rh = r.get("rmsd_heavy")
            if not smi or rh is None or rh >= rmsd_cap or smi in seen:
                continue
            seen.add(smi)
            m = Chem.MolFromSmiles(smi)
            if m is None:
                continue
            nha = m.GetNumHeavyAtoms()
            key = (nha, -rh)
            if best is None or key > best[0]:
                best = (key, smi, r.get("e_gain"), rh, nha)
        if best is None:
            continue
        _, smi, eg, rh, nha = best
        print(f"\n  {s}:")
        print(_fmt(smi, eg, rh, n=nha))


first_unique(DRUGS_DATA + "/v26s_scaffolds_n10k", "30-atom", n=2)
largest_clean(DRUGS_DATA + "/v26g_scaffolds_n10k", "50-atom", rmsd_cap=0.5)
