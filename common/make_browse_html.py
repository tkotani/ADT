"""Build single-file HTML browser for gen_browse_data.jsonl.

Usage:
  python make_browse_html.py <jsonl> <output.html> [--title "..."] [--filter preserved|changed|all]
"""
import argparse, json, sys, os

_ATOM_COLOR = {
    1: "#FFFFFF", 6: "#808080", 7: "#3050F8", 8: "#FF0D0D", 9: "#90E050",
    15: "#FF8000", 16: "#FFFF30", 17: "#1FF01F", 35: "#A62929", 53: "#940094",
}

HTML_TEMPLATE = r"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8" />
<title>__TITLE__</title>
<script src="https://3Dmol.org/build/3Dmol-min.js"></script>
<style>
  body { margin:0; background:#111; color:#ddd; font-family: monospace, Consolas; }
  #topbar { padding:8px 12px; background:#1a1a2a; border-bottom:1px solid #333;
            display:flex; align-items:center; gap:12px; flex-wrap:wrap; }
  #topbar h2 { margin:0; font-size:18px; color:#9cc0ff; }
  #topbar button { background:#2a4a8a; color:#eee; border:none; padding:4px 10px;
                   cursor:pointer; border-radius:3px; }
  #topbar button:hover { background:#3a6acb; }
  #topbar button:disabled { background:#444; cursor:not-allowed; }
  #counter { min-width:80px; text-align:center; }
  #info { font-size:12px; padding:4px 12px; background:#15152a; border-bottom:1px solid #333; }
  #info .tag { display:inline-block; padding:1px 6px; margin-right:6px; border-radius:3px; font-size:10px; }
  .tag.ok { background:#163; color:#dfd; }
  .tag.bad { background:#833; color:#fdd; }
  #sliderbar { padding:6px 12px; background:#1a1a2a; border-bottom:1px solid #333;
               display:flex; align-items:center; gap:10px; }
  #anim_slider { flex:1; }
  #viewer { width:100vw; height: calc(100vh - 160px); position:relative; }
  select { background:#2a4a8a; color:#eee; border:none; padding:3px 6px; border-radius:3px; }
  label { font-size:12px; }
</style>
</head>
<body>
<div id="topbar">
  <h2>GFN2-xTB ADT Browser</h2>
  <button id="prev_btn">◀ Prev</button>
  <span id="counter">1 / N</span>
  <button id="next_btn">Next ▶</button>
  <label>Filter:
    <select id="filter_sel">
      <option value="all">all</option>
      <option value="preserved">preserved</option>
      <option value="changed">changed only</option>
    </select>
  </label>
  <button id="xtb_btn">xTB relax: OFF</button>
  <button id="play_btn">▶ Play placement</button>
  <label>speed:
    <select id="speed_sel">
      <option value="500">slow</option>
      <option value="180" selected>normal</option>
      <option value="80">fast</option>
      <option value="30">very fast</option>
    </select>
  </label>
</div>
<div id="info"><span id="mol_info"></span></div>
<div id="sliderbar">
  <label>step:</label>
  <input type="range" id="anim_slider" min="0" max="1" step="1" value="1">
  <span id="anim_label">final</span>
</div>
<div id="viewer"></div>

<script>
const ATOM_COLOR = __ATOM_COLOR__;
const ATOM_SYMBOL = {1:"H",6:"C",7:"N",8:"O",9:"F",15:"P",16:"S",17:"Cl",35:"Br",53:"I"};
const MOLS = __MOLS__;

const DEFAULT_FRAME = 6;   // fallback if n_frame_atoms missing
let FRAME_ATOMS = DEFAULT_FRAME;  // per-mol, set in loadCurrent
let filteredIdx = MOLS.map((_,i) => i);
let currentPos = 0;
let currentMol = null;
let animStep = null;
let showXtb = false;
let playTimer = null;
let playSpeed = 180;
let viewer = null;
let lockedView = null;   // non-null while animating or xTB overlay active → fix camera
let needFit = true;      // run zoomTo on next render (new molecule loaded)

function applyFilter(mode){
  filteredIdx = MOLS.map((_,i)=>i).filter(i => {
    const m = MOLS[i];
    const pres = m.same_topo || m.same_inchi;
    if (mode==="preserved") return pres;
    if (mode==="changed") return !pres;
    return true;
  });
  if (filteredIdx.length === 0) filteredIdx = [0];
  currentPos = Math.min(currentPos, filteredIdx.length-1);
}

function buildBlock(atoms, bonds, anums){
  // build MOL block (V2000)
  const natom = atoms.length;
  const nbond = bonds.length;
  let s = "\n  made\n\n";
  s += natom.toString().padStart(3) + nbond.toString().padStart(3) + "  0  0  0  0  0  0  0  0999 V2000\n";
  for (let k=0;k<natom;k++){
    const [x,y,z] = atoms[k];
    const sym = ATOM_SYMBOL[anums[k]] || "C";
    s += x.toFixed(4).padStart(10) + y.toFixed(4).padStart(10) + z.toFixed(4).padStart(10) + " " + sym.padEnd(2) + "  0  0  0  0  0  0  0  0  0  0  0  0\n";
  }
  for (const b of bonds){
    const ord = Math.max(1, Math.round(b[2]));
    s += (b[0]+1).toString().padStart(3) + (b[1]+1).toString().padStart(3) + ord.toString().padStart(3) + "  0  0  0  0\n";
  }
  s += "M  END\n";
  return s;
}

function render(){
  viewer.clear();
  const m = currentMol;
  const na = m.n_atoms;
  const k = Math.min(Math.max(animStep, FRAME_ATOMS), na);
  const atFinal = k === na;

  // pre (always shown, thin when xTB overlayed, thicker when alone)
  const atomsSub = m.coords_pre.slice(0, k);
  const anumsSub = m.anums.slice(0, k);
  const bondsSub = m.bonds.filter(b => b[0] < k && b[1] < k);
  const blk = buildBlock(atomsSub, bondsSub, anumsSub);
  viewer.addModel(blk, "mol");
  if (showXtb && atFinal){
    // pre in full ball-and-stick with reduced opacity (ghost / afterimage)
    viewer.setStyle({model:-1}, {stick:{radius:0.14, opacity:0.45}, sphere:{scale:0.26, opacity:0.45}});
  } else {
    viewer.setStyle({model:-1}, {stick:{radius:0.14}, sphere:{scale:0.26}});
  }

  // post + arrows only when xTB toggle ON and at final step
  if (showXtb && atFinal){
    const blkPost = buildBlock(m.coords_post, m.bonds, m.anums);
    viewer.addModel(blkPost, "mol");
    viewer.setStyle({model:-1}, {stick:{radius:0.14}, sphere:{scale:0.26}});

    for (let i=0;i<na;i++){
      const a = m.coords_pre[i], b = m.coords_post[i];
      const dx=b[0]-a[0], dy=b[1]-a[1], dz=b[2]-a[2];
      const d = Math.sqrt(dx*dx+dy*dy+dz*dz);
      if (d < 0.05) continue;
      // arrow spans pre→post; tip sits at midpoint (arrow start→mid), shaft continues mid→post
      const mx=(a[0]+b[0])/2, my=(a[1]+b[1])/2, mz=(a[2]+b[2])/2;
      viewer.addArrow({
        start: {x:a[0], y:a[1], z:a[2]},
        end:   {x:mx,   y:my,   z:mz},
        radius: 0.03, color: "red", radiusRatio: 2.5,
      });
      viewer.addCylinder({
        start: {x:mx,   y:my,   z:mz},
        end:   {x:b[0], y:b[1], z:b[2]},
        radius: 0.03, color: "red",
      });
    }
  }

  if (lockedView !== null){
    viewer.setView(lockedView);
  } else if (needFit){
    viewer.zoomTo();
    needFit = false;
  }
  viewer.render();
}

function updateInfo(){
  const m = currentMol;
  const tag_topo = m.same_topo ? '<span class="tag ok">topo ✓</span>' : '<span class="tag bad">topo ✗</span>';
  const tag_inchi = m.same_inchi ? '<span class="tag ok">InChI ✓</span>' : '<span class="tag bad">InChI ✗</span>';
  const rmsd = m.rmsd_heavy !== null ? m.rmsd_heavy.toFixed(3) : "n/a";
  document.getElementById("mol_info").innerHTML =
    `${tag_topo}${tag_inchi} <b>SMI:</b> ${m.smi} &nbsp; ` +
    `<b>HA-RMSD</b>=${rmsd}Å &nbsp; <b>dE</b>=${m.e_gain.toFixed(1)}kcal/mol &nbsp; ` +
    `<b>n_atoms</b>=${m.n_atoms}` +
    (m.same_topo ? '' : `<br><b>→post:</b> ${m.topo_post || '(none)'}`);
  document.getElementById("counter").innerText = `${currentPos+1} / ${filteredIdx.length}`;
}

function loadCurrent(){
  currentMol = MOLS[filteredIdx[currentPos]];
  const na = currentMol.n_atoms;
  FRAME_ATOMS = (currentMol.n_frame_atoms !== undefined) ? currentMol.n_frame_atoms : DEFAULT_FRAME;
  const slider = document.getElementById("anim_slider");
  slider.min = FRAME_ATOMS;
  slider.max = na;
  slider.value = na;
  animStep = na;
  document.getElementById("anim_label").innerText = "final";
  lockedView = null;
  needFit = true;        // fit camera once for this new molecule
  showXtb = false;
  document.getElementById("xtb_btn").innerText = "xTB relax: OFF";
  updateInfo();
  render();
}

function setStep(s){
  animStep = s;
  const na = currentMol.n_atoms;
  const label = (s === FRAME_ATOMS) ? `frame (${FRAME_ATOMS})` :
                (s >= na)              ? `final (${na})` :
                                          `atom ${s}/${na}`;
  document.getElementById("anim_label").innerText = label;
  render();
}

document.getElementById("prev_btn").onclick = () => { if (currentPos>0) { currentPos--; loadCurrent(); } };
document.getElementById("next_btn").onclick = () => { if (currentPos<filteredIdx.length-1) { currentPos++; loadCurrent(); } };
document.getElementById("filter_sel").onchange = e => { applyFilter(e.target.value); currentPos = 0; loadCurrent(); };
document.getElementById("anim_slider").oninput = e => setStep(parseInt(e.target.value));
document.getElementById("xtb_btn").onclick = () => {
  showXtb = !showXtb;
  document.getElementById("xtb_btn").innerText = "xTB relaxed: " + (showXtb ? "ON" : "OFF");
  if (showXtb){
    const na = currentMol.n_atoms;
    document.getElementById("anim_slider").value = na;
    // lock camera to current view while overlay shown
    lockedView = viewer.getView();
    setStep(na);
  } else {
    lockedView = null;
    render();
  }
};
document.getElementById("speed_sel").onchange = e => {
  playSpeed = parseInt(e.target.value);
  if (playTimer){
    clearInterval(playTimer);
    startPlay();
  }
};

function startPlay(){
  const na = currentMol.n_atoms;
  playTimer = setInterval(() => {
    if (animStep >= na) {
      clearInterval(playTimer); playTimer=null;
      document.getElementById("play_btn").innerText="▶ Play placement";
      lockedView = null;   // release camera after play ends
      return;
    }
    const s = animStep + 1;
    document.getElementById("anim_slider").value = s;
    setStep(s);
  }, playSpeed);
}

document.getElementById("play_btn").onclick = () => {
  if (playTimer) {
    clearInterval(playTimer); playTimer=null;
    document.getElementById("play_btn").innerText="▶ Play placement";
    lockedView = null;   // release camera for user rotation
    return;
  }
  // snapshot current camera, lock during play
  lockedView = viewer.getView();
  document.getElementById("play_btn").innerText="⏸ Pause";
  document.getElementById("anim_slider").value = FRAME_ATOMS;
  setStep(FRAME_ATOMS);
  startPlay();
};

document.addEventListener("keydown", e => {
  if (e.key === "ArrowRight") document.getElementById("next_btn").click();
  else if (e.key === "ArrowLeft") document.getElementById("prev_btn").click();
});

window.addEventListener("load", () => {
  viewer = $3Dmol.createViewer(document.getElementById("viewer"), {backgroundColor: "#111"});
  applyFilter("all");
  loadCurrent();
});
</script>
</body>
</html>
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("jsonl")
    ap.add_argument("output")
    ap.add_argument("--title", default="GFN2-xTB ADT Browser")
    args = ap.parse_args()

    mols = []
    with open(args.jsonl) as f:
        for line in f:
            line = line.strip()
            if line:
                mols.append(json.loads(line))
    print(f"Loaded {len(mols)} molecules from {args.jsonl}")

    html = HTML_TEMPLATE.replace("__TITLE__", args.title)
    html = html.replace("__ATOM_COLOR__", json.dumps(_ATOM_COLOR))
    html = html.replace("__MOLS__", json.dumps(mols))
    with open(args.output, "w") as f:
        f.write(html)
    print(f"Wrote {args.output} ({os.path.getsize(args.output)/1024:.1f} KB)")


if __name__ == "__main__":
    main()
