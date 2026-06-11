# ADT — Autoregressive Discrete Tokens for 3D Molecular Generation

`adt.tex` paper のコード・データ再現セット。`QM9/` (Tab 6/7、Fig 6) と `Drugs/` (Tab 4/5、Fig 4/5) を網羅。

## 構成

```
~/ADT/                    5.6 MB code (git で管理)
├── README.md             ← このファイル
├── .gitignore            (PTBANK/ 除外)
├── common/               共有 lib (model / tokenizer / dataset / util)
├── tools/
│   ├── data_manifest.tsv 必要 data 一覧 (md5 + ptbank_name 対応)
│   └── setup_data.sh     PTBANK → data/ symlink 自動生成
├── ADTpaper1/            paper Fig/Table 集計 scripts (13 .py + log symlinks)
├── docs/                 interactive 3D browser (GitHub Pages site, 7 scaffold × 50 mol)
├── QM9/                  paper QM9 (Tab 6/7、Fig 6 + App F)
│   ├── README.md
│   ├── src/{freeorder, freeorder_inferior}/
│   └── data/{freeorder, freeorder_inferior}/
├── Drugs/                paper Drugs (Tab 4/5、Fig 4/5、App B-E)
│   ├── README.md
│   ├── src/{freeorder_v26, freeorder_v26_inferior}/
│   └── data/{freeorder_v26, freeorder_v26_inferior}/
└── PTBANK/               ★ 5.1 GB data hub (git 除外、Zenodo 候補)
    ├── ckpts × 4 (QM9 E562 + Drugs 30/50)
    ├── frame_caches (qm9 + drugs + scaffold × 7)
    ├── training data caches (drugs_mols + qm9_mols)
    └── training logs × 3 (Fig 6 / App F source)
```

## セットアップ (clone → 再現可能状態)

```bash
git clone <repo-url> ~/ADT
cd ~/ADT
export ADT_ROOT=~/ADT

# PTBANK populate (将来 Zenodo から download)
mkdir -p $ADT_ROOT/PTBANK
# ... place files (or wget zenodo tarball)

# data/ 内に symlink 生成
tools/setup_data.sh                # → "Summary: 20 linked, 0 missing"
tools/setup_data.sh --verify       # md5 確認付き
```

---

## 1. Training (paper ckpt 再生成)

paper の ckpt が手元にない場合、または別 GPU で再現する場合。

| paper task | runner | GPU 構成 | 時間 |
|---|---|---|---|
| QM9 E562 (Tab 6/7) | `QM9/src/freeorder/run_paper_qm9_train.sh` | RTX 4090 ×1 | ~16.5 h (600 epoch) |
| Drugs 30-atom E142 (Tab 4) | `MAX=30 Drugs/src/freeorder_v26/run_paper_drugs_train.sh` | RTX 4090 ×2 DDP | ~14 h (300 epoch、early stop) |
| Drugs 50-atom E157 (Tab 5) | `MAX=50 Drugs/src/freeorder_v26/run_paper_drugs_train.sh` | RTX 4090 ×2 DDP | ~3.5 h (30 epoch ft from 30-atom) |

```bash
# QM9 (E562 を data/freeorder/checkpoints_v26_repro/best.pt に生成)
QM9/src/freeorder/run_paper_qm9_train.sh

# Drugs 30-atom (E142 を data/freeorder_v26/checkpoints_v26_scratch_repro/ に)
MAX=30 Drugs/src/freeorder_v26/run_paper_drugs_train.sh

# Drugs 50-atom (E157 を data/freeorder_v26/checkpoints_v26_max50_ft_v2_repro/ に)
MAX=50 Drugs/src/freeorder_v26/run_paper_drugs_train.sh
```

env で上書き可: `EPOCHS=`, `SAVE_DIR=`, `PRETRAIN=` (Drugs 50 のみ、ft 元 ckpt)

---

## 2. Generation (分子サンプリング)

学習済 ckpt から分子を生成する (eval pipeline の Phase 1 部分)。

| paper task | scope | コマンド |
|---|---|---|
| QM9 unconditional | 全分子をフラットに sampling | `GEN_N=10000 QM9/src/freeorder/run_paper_qm9_eval.sh` |
| Drugs scaffold-conditional | 7 scaffold (benzene/pyridine/...) × N | `MAX=30 GEN_N=10000 Drugs/src/freeorder_v26/run_paper_drugs_eval.sh` |

### QM9 generation
```bash
GEN_N=10000 QM9/src/freeorder/run_paper_qm9_eval.sh
# 出力: data/freeorder/eval_v26_E562_N10000/
#   mol_stable_kekulized.sdf       生成 valid 分子の SDF
#   mol_stable_smiles.txt           SMILES list
#   gen_stats.json                  生成統計 (mol_stable, collision, valid_3d)
```
※ runner は Phase 2 (xTB) も自動実行されるが、Phase 1 だけ欲しい場合は `GEN_N=` で N=100 等にして停止。

### Drugs scaffold-conditional generation
```bash
MAX=30 GEN_N=10000 Drugs/src/freeorder_v26/run_paper_drugs_eval.sh
# 出力: data/freeorder_v26/v26s_scaffolds_n10000_repro/<scaffold>/
#   benzene/, pyridine/, pyrimidine/, pyrazine/, furan/, thiophene/, cyclohexane/
#   各 scaffold dir に gen_stats.json + SDF + xtb_results.json
```

---

## 3. Evaluation (paper 数値を出す)

生成 SDF / xtb 結果から paper Tab/Fig 数値を集計。

### QM9 evaluation (Tab 6 / Tab 7)

`run_paper_qm9_eval.sh` は **Generation + xTB Evaluation を1ジョブで完結**:
- Phase 1: N=10000 分子を sample → mol_stable 判定
- Phase 2: 全 mol_stable 分子に xTB 適用 → topology preserved 等の paper Tab 7 metric

```bash
GEN_N=10000 QM9/src/freeorder/run_paper_qm9_eval.sh
# stats.json に paper 数値:
#   mol_stable / N      → Tab 6 (paper 88.0%)
#   topology preserved  → Tab 7 xVR (paper 86.3%)
#   xtb_ok / mol_stable → Tab 7 (paper 100%)
```

検証済 (N=1000、本機 4090): mol_stable **88.8%**, topology **87.2%**, novel/unique **42.0%** (paper の 88.0% / 86.3% / 42.7% を統計揺らぎ範囲内で再現)。

### Drugs evaluation (Tab 4 / Tab 5、Fig 4/5、App B-E)

scaffold 別 N=10000 生成後、集計 scripts で各 paper Fig/Table を再構成:

```bash
cd ADTpaper1/scripts

python3 per_scaffold_stats.py      # → App B (per-scaffold kekulized/unique/atom dist、Table 6/7)
python3 per_scaffold_novelty.py    # → App B (novel / novel% 列)
python3 per_scaffold_diversity.py  # → App C (BM scaffold / Tanimoto / MW / logP、Table 8/9、seed=42 で厳密再現)
python3 success_examples.py        # → App D (XVR-positive 生成例、Table 10/11)
python3 failure_breakdown.py       # → App E (Type-A 失敗内訳、Table 12/13)
python3 failure_examples.py        # → App E (Type-A1 失敗 SMILES、Table 14)
python3 bond_angle_errors.py       # → Fig 5 左 (bond/angle error 分布)
python3 bond_angle_hist.py         # → Fig 5 右
python3 geom_drugs_v26_dist.py     # → Fig 4 (heavy-atom dist)
python3 v21_nolimit_hist.py        # → Fig 4 (元 GEOM 分布)
python3 extract_curves.py          # → Fig 6 / App F (training curves)
```

各 script は `tools/data_manifest.tsv` で symlink 済の data dir を参照、`ADT_ROOT` 環境変数で repo path を解決。

---

## 4. Interactive 3D browser (docs/)

scaffold 別生成分子を 3Dmol.js で表示し、ADT が原子を1つずつ自己回帰生成する様子を
アニメ表示する自己完結 HTML。論文 §Reproducibility からリンク。

```
docs/
├── index.html              7 scaffold へのランディング
├── <scaffold>.html         自己完結ビューア (50分子/各、3Dmol.js inline、CDN不要)
├── data/<scaffold>_50.sdf   生成元 SDF (先頭50 XVR-positive)
├── export_html.py          SDF → 自己完結HTML 生成スクリプト
├── 3Dmol-min.js            bundled 3Dmol.js (生成時に inline される)
└── README.md               使い方・再生成・GitHub Pages 有効化手順
```

- **閲覧**: `xdg-open docs/index.html`、または GitHub Pages 有効化後 <https://tkotani.github.io/ADT/>
- **GitHub Pages**: Settings → Pages → Source: main / folder `/docs`
- **再生成**: `docs/README.md` 参照 (rdkit + numpy のみ、GPU不要)

---

## 検証済み数値 (本機 RTX 4090)

| 項目 | paper | 我々の再現 | 一致 |
|---|---|---|---|
| QM9 mol_stable | 88.0% (N=10000) | 88.8% (N=1000) | ✓ +0.8pt |
| QM9 topology preserved | 86.3% | 87.2% | ✓ +0.9pt |
| QM9 novel/unique | 42.7% | 42.0% | ✓ −0.7pt |
| Drugs benzene mol_stable (Tab 4) | 39.07% | 39.07% (既存 N=10000) | ✓ 一致 |
| Drugs benzene novelty rate | 99.97% (3216/3217) | 99.97% | ✓ |

## paper 採用 config (要点)

| | QM9 | Drugs 30 (Tab 4) | Drugs 50 (Tab 5) |
|---|---|---|---|
| epoch | 562 (val=2.7496) | 142 (val=1.980) | 157 (ft from 30) |
| max_offset | 32 | 50 | 50 |
| max_atoms / max_pointer | 9 / 30 | 30 / 64 | 50 / 64 |
| AROMATIZE_RINGS (eval) | 0 | 1 | 1 |
| pointer mode | offset (v26 hybrid) | offset | offset |
| r mesh | log 200 bins | log 200 bins | log 200 bins |
| ckpt md5 | 17ec74f1...858a2 | 451ffc3c...c488e6 | 0a569723...d52a3c39 |

## 関連プロジェクト

- 続編 (Pocket-ADT): kr2:`~/ADT/Pocket/v27〜v41/` (paper には登場しない、別軸)
- 共有 lib: `common/` は QM9 / Drugs で共通

## ライセンス

コードは **MIT License**(`LICENSE`)。自由に使用・改変・再配布可(企業含む)、attribution(著作権表記の保持)が条件。

ただし本手法は**特許出願中**(特願2026-16495 / 2026-65995)。MIT は著作権の許諾のみで特許権は付与しない。**特許化された手法の商業利用**(売上を生む製品・サービス・プロセスでの実施)には別途特許ライセンスが必要(`PATENTS.md`)。非商用(研究・評価・教育・個人利用)は自由。

同梱の `docs/3Dmol-min.js` は 3Dmol.js(BSD-3-Clause)で別ライセンス(`THIRD-PARTY-NOTICES.md`)。

## Citation

論文(arXiv)公開後に BibTeX を記載予定。利用時は本リポジトリと論文 "Atomic Design Transformer: xTB-Validated 3D Molecule Generation from Scaffolds" (T. Kotani) を引用すること。
