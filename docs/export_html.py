#!/usr/bin/env python3
"""
export_html.py — Export filtered molecules as a self-contained HTML 3D browser.

Usage:
    python export_html.py kr1kr2.pt --stable -o browse.html
    python export_html.py kr1kr2.pt --stable --novel --shuffle -o browse.html
    # Then just: open browse.html (or xdg-open browse.html)
"""

import argparse, json, numpy as np
from rdkit import Chem
from rdkit.Chem import rdDetermineBonds
from rdkit import RDLogger
RDLogger.logger().setLevel(RDLogger.ERROR)

STANDARD_VALENCE = {'H': 1, 'C': 4, 'N': 3, 'O': 2, 'F': 1}

import os

def load_qm9_smiles():
    cache_path = 'qm9_canonical_cache.pt'
    if os.path.exists(cache_path):
        import torch
        cached = torch.load(cache_path, weights_only=False)
        if isinstance(cached, dict):
            cached = cached.get('canonical', cached)
        return set(cached) if not isinstance(cached, set) else cached
    return set()


def validate_mol(z_list, positions):
    n = len(z_list)
    try:
        rw = Chem.RWMol()
        for z in z_list:
            rw.AddAtom(Chem.Atom(int(z)))
        conf = Chem.Conformer(n)
        for j in range(n):
            conf.SetAtomPosition(j, [float(x) for x in positions[j]])
        rw.AddConformer(conf, assignId=True)
        rdDetermineBonds.DetermineConnectivity(rw)
        rdDetermineBonds.DetermineBondOrders(rw, charge=0)
        mol = rw.GetMol()
        Chem.SanitizeMol(mol)
        smiles = Chem.MolToSmiles(Chem.RemoveAllHs(mol))
        n_stable = 0
        for atom in mol.GetAtoms():
            sym = atom.GetSymbol()
            if sym in STANDARD_VALENCE:
                bond_sum = sum(b.GetBondTypeAsDouble() for b in atom.GetBonds())
                if abs(bond_sum - STANDARD_VALENCE[sym]) < 0.1:
                    n_stable += 1
        is_stable = (n_stable == n)
        return mol, smiles, is_stable
    except Exception:
        return None, None, False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('inputs', nargs='+')
    parser.add_argument('-n', type=int, default=500)
    parser.add_argument('--stable', action='store_true')
    parser.add_argument('--novel', action='store_true')
    parser.add_argument('--known', action='store_true')
    parser.add_argument('--shuffle', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--scaffold-size', type=int, default=6,
                        help='Number of leading scaffold atoms shown in the first frame (default: 6 for benzene)')
    parser.add_argument('--frame-ms', type=int, default=250,
                        help='Milliseconds per growth frame')
    parser.add_argument('-o', default='browse.html')
    args = parser.parse_args()

    qm9 = load_qm9_smiles()
    print(f"QM9 reference: {len(qm9)} SMILES")

    all_z, all_pos, all_mols = [], [], []
    for p in args.inputs:
        print(f"Loading {p}...")
        if p.endswith('.sdf') or p.endswith('.sdf.gz'):
            suppl = Chem.SDMolSupplier(p, removeHs=False, sanitize=True)
            for mol in suppl:
                if mol is None:
                    continue
                all_mols.append(mol)
                all_z.append(None)
                all_pos.append(None)
        else:
            import torch
            d = torch.load(p, weights_only=False)
            for i in range(min(len(d['coords']), len(d['atoms']))):
                if d['coords'][i] is not None and d['atoms'][i] is not None:
                    all_mols.append(None)
                    all_z.append(d['atoms'][i])
                    all_pos.append(np.asarray(d['coords'][i], dtype=np.float64))

    if args.shuffle:
        rng = np.random.RandomState(args.seed)
        idx = rng.permutation(len(all_z))
        all_z = [all_z[i] for i in idx]
        all_pos = [all_pos[i] for i in idx]
        all_mols = [all_mols[i] for i in idx]

    def make_growth_frames(mol, scaffold_size=6):
        n = mol.GetNumAtoms()
        conf = mol.GetConformer()
        frames = []
        start = min(scaffold_size, n)
        for k in range(start, n + 1):
            rw = Chem.RWMol()
            for i in range(k):
                rw.AddAtom(Chem.Atom(mol.GetAtomWithIdx(i).GetAtomicNum()))
            for b in mol.GetBonds():
                a, bi = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
                if a < k and bi < k:
                    rw.AddBond(a, bi, b.GetBondType())
            new_conf = Chem.Conformer(k)
            for i in range(k):
                p = conf.GetAtomPosition(i)
                new_conf.SetAtomPosition(i, (p.x, p.y, p.z))
            rw.AddConformer(new_conf, assignId=True)
            frames.append(Chem.MolToMolBlock(rw, kekulize=False))
        return frames

    mol_data = []
    for pre_mol, z, pos in zip(all_mols, all_z, all_pos):
        if len(mol_data) >= args.n:
            break
        if pre_mol is not None:
            mol = pre_mol
            try:
                smi = Chem.MolToSmiles(Chem.RemoveAllHs(mol))
            except Exception:
                continue
            stable = True
        else:
            mol, smi, stable = validate_mol(z, pos)
            if mol is None:
                continue
        novel = smi not in qm9
        if args.stable and not stable:
            continue
        if args.novel and not novel:
            continue
        if args.known and novel:
            continue
        mb = Chem.MolToMolBlock(mol)
        frames = make_growth_frames(mol, scaffold_size=args.scaffold_size)
        mol_data.append({
            'sdf': mb,
            'frames': frames,
            'smiles': smi,
            'n_atoms': mol.GetNumAtoms(),
            'stable': stable,
            'novel': novel,
        })

    print(f"Filtered: {len(mol_data)} molecules")

    # Serialize SDF blocks and metadata
    meta = [{'smiles': m['smiles'], 'n_atoms': m['n_atoms'],
             'stable': m['stable'], 'novel': m['novel']} for m in mol_data]
    sdfs = [m['sdf'] for m in mol_data]
    frames_per_mol = [m['frames'] for m in mol_data]

    # Inline 3Dmol.js for offline use if available alongside this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    local_3dmol = os.path.join(script_dir, '3Dmol-min.js')
    if os.path.exists(local_3dmol):
        with open(local_3dmol) as f:
            mol3d_lib = f.read()
        mol3d_tag = f'<script>{mol3d_lib}</script>'
    else:
        mol3d_tag = '<script src="https://3Dmol.org/build/3Dmol-min.js"></script>'

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>ADT Molecule Browser ({len(mol_data)} molecules)</title>
{mol3d_tag}
<style>
* {{ margin:0; padding:0; box-sizing:border-box; }}
html, body {{ height:100%; overflow:hidden; }}
body {{ background:#0f0f23; color:#e0e0e0; font-family:'Courier New',monospace;
        display:flex; flex-direction:column; }}
#header {{ padding:10px 20px; background:#1a1a2e; border-bottom:1px solid #333;
           display:flex; align-items:center; gap:20px; flex-wrap:wrap; flex-shrink:0; }}
#header h1 {{ font-size:16px; color:#88aaff; white-space:nowrap; }}
#nav {{ display:flex; align-items:center; gap:8px; }}
#nav button {{ background:#2a2a4e; color:#e0e0e0; border:1px solid #444;
              padding:6px 14px; cursor:pointer; font-size:14px; border-radius:3px; }}
#nav button:hover {{ background:#3a3a6e; }}
#counter {{ font-size:14px; color:#aaa; min-width:80px; text-align:center; }}
#info {{ font-size:13px; display:flex; gap:16px; align-items:center; flex-wrap:wrap; }}
#smiles {{ color:#ccc; font-size:13px; max-width:500px; word-break:break-all; }}
.tag {{ padding:2px 8px; border-radius:3px; font-size:12px; font-weight:bold; }}
.tag-stable {{ background:#003322; color:#00ff88; }}
.tag-unstable {{ background:#330011; color:#ff4444; }}
.tag-novel {{ background:#002244; color:#44aaff; }}
.tag-qm9 {{ background:#332200; color:#ffaa00; }}
#slider-row {{ padding:4px 20px; background:#12122a; flex-shrink:0;
               display:flex; align-items:center; gap:12px; }}
#slider-row label {{ font-size:11px; color:#888; white-space:nowrap; min-width:70px; }}
#slider, #frame-slider {{ flex:1; }}
#frame-slider {{ accent-color:#88aaff; }}
#viewer {{ flex:1; min-height:0; position:relative; }}
</style>
</head>
<body>
<div id="header">
  <h1>ADT Molecule Browser</h1>
  <div id="nav">
    <button id="btn-prev" title="← or A">◀ Prev</button>
    <span id="counter">1 / {len(mol_data)}</span>
    <button id="btn-next" title="→ or D">Next ▶</button>
  </div>
  <div id="info">
    <span id="smiles"></span>
    <span id="atoms"></span>
    <span id="tag-stable" class="tag"></span>
    <span id="tag-novel" class="tag"></span>
    <span id="spin-indicator" style="color:#88ff88;font-size:12px;"></span>
  </div>
</div>
<div id="slider-row">
  <label for="slider">mol</label>
  <input type="range" id="slider" min="0" max="{len(mol_data)-1}" value="0">
</div>
<div id="slider-row">
  <label for="frame-slider"><span id="frame-label">atoms 6/?</span></label>
  <input type="range" id="frame-slider" min="0" max="0" value="0">
  <button id="btn-play" title="Play/Pause (space)">▶</button>
  <button id="btn-replay" title="Replay (r)">↻</button>
</div>
<div id="viewer"></div>

<script>
var SDFS = {json.dumps(sdfs)};
var FRAMES = {json.dumps(frames_per_mol)};
var META = {json.dumps(meta)};
var FRAME_MS = {args.frame_ms};
var idx = 0;
var viewer = null;
var animTimer = null;
var savedView = null;
var curFrame = 0;
var playing = true;
var SCAFFOLD_SIZE = {args.scaffold_size};

var STYLE = {{
    stick: {{radius: 0.12, colorscheme: "Jmol"}},
    sphere: {{scale: 0.25, colorscheme: "Jmol"}}
}};

function webglOK() {{
    try {{
        var c = document.createElement("canvas");
        return !!(window.WebGLRenderingContext &&
            (c.getContext("webgl") || c.getContext("experimental-webgl")));
    }} catch (e) {{ return false; }}
}}

function showNoWebGL() {{
    document.getElementById("viewer").innerHTML =
        '<div style="max-width:640px;margin:50px auto;padding:24px;line-height:1.7;' +
        'background:#1a1a2e;border:1px solid #555;border-radius:8px;color:#e0e0e0;font-size:14px;">' +
        '<div style="color:#ff8866;font-size:18px;font-weight:bold;margin-bottom:10px;">' +
        '3D view unavailable &mdash; WebGL is off</div>' +
        'This page renders molecules with WebGL, but your browser could not create a WebGL ' +
        'context. This is common in <b>Chrome on a machine without a GPU</b>, where software ' +
        'WebGL is disabled by default. To fix it:' +
        '<ul style="margin:10px 0 10px 22px;">' +
        '<li>Open this page in <b>Firefox</b> (software WebGL works out of the box), or</li>' +
        '<li>In Chrome, set <code style="background:#0f0f23;padding:1px 5px;border-radius:3px;">' +
        'chrome://flags/#enable-unsafe-swiftshader</code> to <b>Enabled</b> and relaunch.</li>' +
        '</ul>The molecule data is fully loaded; only the rendering needs WebGL.</div>';
}}

function initViewer() {{
    if (!webglOK()) {{ showNoWebGL(); return; }}
    var el = document.getElementById("viewer");
    try {{
        viewer = $3Dmol.createViewer(el, {{ backgroundColor: "#1a1a2e" }});
    }} catch (e) {{ showNoWebGL(); return; }}
    if (!viewer) {{ showNoWebGL(); return; }}
    requestAnimationFrame(function() {{
        viewer.resize();
        showMol(0);
    }});
    window.addEventListener("resize", function() {{
        viewer.resize();
        viewer.render();
    }});
}}

function computeView(i) {{
    // Render the full molecule once to derive the fixed camera that fits it.
    viewer.removeAllModels();
    viewer.addModel(SDFS[i], "sdf");
    viewer.setStyle({{}}, STYLE);
    viewer.resize();
    viewer.zoomTo();
    viewer.rotate(60, "x");
    viewer.rotate(30, "y");
    viewer.zoom(1.15);
    return viewer.getView();
}}

function drawFrame(frameSdf) {{
    viewer.removeAllModels();
    viewer.addModel(frameSdf, "sdf");
    viewer.setStyle({{}}, STYLE);
    viewer.setView(savedView);
    viewer.render();
}}

function stopAnim() {{
    if (animTimer) {{ clearTimeout(animTimer); animTimer = null; }}
}}

function setFrame(f) {{
    var frames = FRAMES[idx];
    curFrame = Math.max(0, Math.min(f, frames.length - 1));
    drawFrame(frames[curFrame]);
    var slider = document.getElementById("frame-slider");
    slider.value = curFrame;
    var nAtoms = SCAFFOLD_SIZE + curFrame;
    document.getElementById("frame-label").textContent =
        "atoms " + nAtoms + "/" + META[idx].n_atoms;
}}

function playAnim() {{
    stopAnim();
    playing = true;
    document.getElementById("btn-play").textContent = "⏸";
    var frames = FRAMES[idx];
    function tick() {{
        setFrame(curFrame);
        if (curFrame < frames.length - 1) {{
            curFrame++;
            animTimer = setTimeout(tick, FRAME_MS);
        }} else {{
            animTimer = null;
            playing = false;
            document.getElementById("btn-play").textContent = "▶";
        }}
    }}
    tick();
}}

function togglePlay() {{
    if (animTimer) {{
        stopAnim();
        playing = false;
        document.getElementById("btn-play").textContent = "▶";
    }} else {{
        if (curFrame >= FRAMES[idx].length - 1) curFrame = 0;
        playAnim();
    }}
}}

function showMol(i) {{
    idx = Math.max(0, Math.min(i, SDFS.length - 1));
    stopAnim();
    savedView = computeView(idx);

    var frames = FRAMES[idx];
    var slider = document.getElementById("frame-slider");
    slider.min = 0;
    slider.max = frames.length - 1;
    curFrame = 0;

    var m = META[idx];
    document.getElementById("counter").textContent = (idx+1) + " / " + SDFS.length;
    document.getElementById("smiles").textContent = m.smiles;
    document.getElementById("atoms").textContent = "atoms=" + m.n_atoms;

    var ts = document.getElementById("tag-stable");
    ts.textContent = m.stable ? "STABLE" : "UNSTABLE";
    ts.className = "tag " + (m.stable ? "tag-stable" : "tag-unstable");

    var tn = document.getElementById("tag-novel");
    tn.textContent = m.novel ? "NOVEL" : "QM9";
    tn.className = "tag " + (m.novel ? "tag-novel" : "tag-qm9");

    document.getElementById("slider").value = idx;

    playAnim();
}}

function replayAnim() {{ curFrame = 0; playAnim(); }}

document.getElementById("btn-prev").onclick = function() {{ showMol(idx - 1); }};
document.getElementById("btn-next").onclick = function() {{ showMol(idx + 1); }};
document.getElementById("slider").oninput = function() {{ showMol(parseInt(this.value)); }};
document.getElementById("frame-slider").oninput = function() {{
    stopAnim();
    document.getElementById("btn-play").textContent = "▶";
    setFrame(parseInt(this.value));
}};
document.getElementById("btn-play").onclick = togglePlay;
document.getElementById("btn-replay").onclick = replayAnim;

document.addEventListener("keydown", function(e) {{
    if (e.target.tagName === "INPUT") return;
    if (e.key === "ArrowLeft" || e.key === "a") showMol(idx - 1);
    if (e.key === "ArrowRight" || e.key === "d") showMol(idx + 1);
    if (e.key === "Home") showMol(0);
    if (e.key === "End") showMol(SDFS.length - 1);
    if (e.key === "r") replayAnim();
    if (e.key === " ") {{ e.preventDefault(); togglePlay(); }}
}});

document.addEventListener("DOMContentLoaded", initViewer);
</script>
</body>
</html>"""

    with open(args.o, 'w') as f:
        f.write(html)
    print(f"Wrote {args.o} ({os.path.getsize(args.o) / 1024:.0f} KB)")
    print(f"Open: xdg-open {args.o}")


if __name__ == '__main__':
    main()
