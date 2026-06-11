"""Compute novelty of generated molecules against PubChem.

Reads the aggregated JSONL files, canonicalizes each preserved mol,
computes InChIKey skeleton (first 14 chars), and checks set intersection
against the PubChem InChIKey skeleton set extracted from CID-InChI-Key.gz.

Usage:
    python pubchem_novelty.py \
        --pubchem ~/ADT/data/pubchem/CID-InChI-Key.gz \
        --agg ~/ADT/data/Drugs/freeorder_v21/browse_aggregated
"""
import argparse, json, gzip, time, os, sys
from rdkit import Chem, RDLogger
RDLogger.logger().setLevel(RDLogger.ERROR)


def canon_inchikey_skel(smi):
    try:
        m = Chem.MolFromSmiles(smi)
        if m is None:
            return None
        return Chem.MolToInchiKey(m).split("-")[0]
    except Exception:
        return None


def load_generated(agg_dir, files=("kr2_xtb.jsonl", "kr3_xtb.jsonl", "kr2_v3_xtb.jsonl")):
    all_preserved = []
    all_stable = []
    for fn in files:
        with open(os.path.join(agg_dir, fn)) as f:
            for line in f:
                d = json.loads(line)
                smi = d.get("topo_pre") or d.get("smi")
                if not smi:
                    continue
                all_stable.append(smi)
                if d.get("xtb_ok") and (d.get("same_topo") or d.get("same_inchi")):
                    all_preserved.append(smi)
    return all_stable, all_preserved


def canonical_skel_set(smiles_list):
    s = set()
    for smi in smiles_list:
        k = canon_inchikey_skel(smi)
        if k: s.add(k)
    return s


def load_pubchem_skels(gzpath):
    """Stream CID-InChI-Key.gz and return set of InChIKey skeletons (first
    14 chars of the InChIKey)."""
    skel_set = set()
    t0 = time.time()
    n = 0
    with gzip.open(gzpath, "rt") as f:
        for line in f:
            # Format: CID\tInChI\tInChIKey
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            ik = parts[2]
            if len(ik) < 14:
                continue
            skel_set.add(ik[:14])
            n += 1
            if n % 10_000_000 == 0:
                print(f"  {n//1_000_000}M rows, {len(skel_set)} unique skels, {time.time()-t0:.0f}s", flush=True)
    print(f"  PubChem total rows: {n}, unique InChIKey skeletons: {len(skel_set)} ({time.time()-t0:.0f}s)", flush=True)
    return skel_set


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pubchem", required=True)
    ap.add_argument("--agg", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    print("=== Loading generated data ===", flush=True)
    t0 = time.time()
    stable, preserved = load_generated(args.agg)
    stable_skels = canonical_skel_set(stable)
    preserved_skels = canonical_skel_set(preserved)
    print(f"  mol_stable SMILES: {len(stable)} → {len(stable_skels)} unique skeletons", flush=True)
    print(f"  preserved SMILES : {len(preserved)} → {len(preserved_skels)} unique skeletons", flush=True)
    print(f"  ({time.time()-t0:.0f}s)", flush=True)

    print("\n=== Loading PubChem InChIKey skeletons ===", flush=True)
    pubchem_skels = load_pubchem_skels(args.pubchem)

    print("\n=== Novelty analysis ===")
    in_pubchem_stable = stable_skels & pubchem_skels
    in_pubchem_pres = preserved_skels & pubchem_skels
    novel_stable = stable_skels - pubchem_skels
    novel_pres = preserved_skels - pubchem_skels
    print(f"All mol_stable unique skels ({len(stable_skels)}):")
    print(f"  in PubChem: {len(in_pubchem_stable)} ({len(in_pubchem_stable)/max(len(stable_skels),1)*100:.2f}%)")
    print(f"  novel     : {len(novel_stable)} ({len(novel_stable)/max(len(stable_skels),1)*100:.2f}%)")
    print(f"Preserved unique skels ({len(preserved_skels)}):")
    print(f"  in PubChem: {len(in_pubchem_pres)} ({len(in_pubchem_pres)/max(len(preserved_skels),1)*100:.2f}%)")
    print(f"  novel     : {len(novel_pres)} ({len(novel_pres)/max(len(preserved_skels),1)*100:.2f}%)")

    out = {
        "n_stable_unique_skels": len(stable_skels),
        "n_preserved_unique_skels": len(preserved_skels),
        "pubchem_n_skels": len(pubchem_skels),
        "stable_in_pubchem": len(in_pubchem_stable),
        "stable_novel_vs_pubchem": len(novel_stable),
        "preserved_in_pubchem": len(in_pubchem_pres),
        "preserved_novel_vs_pubchem": len(novel_pres),
    }
    out_path = args.out or os.path.join(args.agg, "pubchem_novelty.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
