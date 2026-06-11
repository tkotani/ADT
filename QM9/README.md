# QM9 paper 再現 — `adt.tex` Table 2 / Table 3 / Fig 6

QM9 unconditional generation の paper 数値を再現する。

## ★ Table 2 / Table 3 の正準データ (canonical, N=10000, E562)

paper の Table 2/3 は **この N=10000 E562 eval の出力そのもの**。raw データは
`data/freeorder/eval_v26_E562_N10000/`（PTBANK archive `qm9_eval_E562_N10000.tgz`）に保存。

| Table 3 行 | count | % | stats.json フィールド |
|---|---|---|---|
| mol_stable / N | 8746/10000 | **87.5%** | `gen_stats.mol_stable` |
| unique / mol_stable | 8508/8746 | 97.3% | `n_unique` |
| novel / unique | 3635/8508 | 42.7% | run.log（in QM9 4873/8508） |
| kekulizable | 8746/8746 | 100% | `n_kekulizable` |
| xTB converged | 8743/8746 | 100.0% | `xtb_ok` |
| topology preserved / N（= XVR） | 8566/10000 | **85.7%** | `topo_same` |
| RMSD heavy median | 0.184 A | | `xtb_results.json: rmsd_heavy` |
| Energy gain median | -1.31 eV | | `xtb_results.json: e_gain`（kcal->eV） |

**サンプリング変動**: 生成は T=1.0 確率的サンプリングなので、再実行すると +-0.3pt
（+-sqrt(N*p*(1-p))≈+-33 分子）程度ぶれる。`mol_stable` が **87+-1%**、`topology` が
**86+-1%** の範囲なら再現成功とみなす。seed 固定でも生成の乱数で完全一致はしない。

> **注**: 初期 draft では別 N=10000 run の値（8800/10000=88.0%, 8633/10000=86.3%）を
> 記載していたが、その raw 出力が保存されていなかったため、再現可能な本 run
> （8746/8566）に paper を合わせた。結論（~88%、SOTA と 2-3pt 差）は不変。

### quick verification（N=1000, ~5 min）
`data/freeorder/eval_v26_E562_N1000_verification/stats.json`:
mol_stable 888/1000 = 88.8%、topology 872/1000 = 87.2%（正準 N=10000 と統計的に整合）。


> **Note**: 以下のコマンド内の `$ADT_ROOT` は repo の root (この README のあるディレクトリの 1 つ上、つまり `QM9/` の親 = clone 先) を指す。runner shell 内では自動解決、Python 例コマンドは `export ADT_ROOT=/path/to/clone` してから実行 (デフォルト `$HOME/ADT` 想定)。

## 0. 再現に必要な前提

- **GPU**: RTX 4090 ×1 (~99 s/epoch、N=10000 eval ~50 min、N=1000 ~5 min)
- **Python env**: `~/.venv/bin/python3` (torch 2.x + rdkit 2026 + xtb 6 + meeko 等)
- **xtb**: `~/miniconda3/bin/xtb` 等が PATH または明示

## 1. paper ckpt (E562) の確認

```bash
ls -la ~/ADT/QM9/data/freeorder/checkpoints_v26_scratch/best.pt
# 期待値 1028581605 bytes
md5sum ~/ADT/QM9/data/freeorder/checkpoints_v26_scratch/best.pt
# 期待値 17ec74f188b0dd4c216ed05dab9c85a2

~/.venv/bin/python -c '
import torch
ck = torch.load("$ADT_ROOT/QM9/data/freeorder/checkpoints_v26_scratch/best.pt", map_location="cpu", weights_only=False)
print("epoch:", ck["epoch"])         # → 562
print("val_loss:", ck["val_loss"])   # → 2.7496
print("config:", ck["config"])       # → max_offset=32, output_pointer_mode=offset, n_r_bins=200
'
```

## 2. Generation + Evaluation (Table 2 / Table 3)

```bash
GEN_N=10000 ~/ADT/QM9/src/freeorder/run_paper_qm9_eval.sh
```

実行内容:
- `gen_eval_v26.py` を `CUDA_VISIBLE_DEVICES=0` で起動
- Phase 1 (生成): N=10000 分子を sampling、collision 弾き、RDKit で `mol_stable` 判定
- Phase 2 (xTB): mol_stable 全分子に対し xTB optimization 実行、`topo_same` (re-perceive 後 SMILES 一致) 判定

出力: `~/ADT/QM9/data/freeorder/eval_v26_E562_N10000/`
- `stats.json` ← Table 2/3 数値 (`mol_stable`, `topo_same`, `n_unique`, `xtb_ok`, `inchi_same`, …)
- `gen_stats.json` ← Phase 1 だけの抜粋
- `mol_stable_kekulized.sdf` ← 全 valid 分子 (xTB 前)
- `xtb_results.json` / `xtb_results_incremental.jsonl` ← 各分子の xTB 結果 (energy, RMSD, post-relax SMILES)
- `xtb_work/` ← 各分子の xTB 実行 dir (mol_*.xyz, xtbopt.xyz, charges, wbo, …)
- `run.log` ← stdout

確認用 N=1000 (~5 min):
```bash
GEN_N=1000 ~/ADT/QM9/src/freeorder/run_paper_qm9_eval.sh
# 既に走らせ済 → ~/ADT/QM9/data/freeorder/eval_v26_E562_N1000_verification/stats.json で 88.8% 確認
```

### `run_paper_qm9_eval.sh` の中身 (env 一覧)

```bash
CUDA_VISIBLE_DEVICES=0
GEN_N=${GEN_N:-1000}        # 生成数
ADT_R_BINS=200              # log mesh 200 段 (E562 学習時と一致)
AROMATIZE_RINGS=0           # paper QM9: 芳香環ほぼ無いので 0
CKPT       =  data/freeorder/checkpoints_v26_scratch/best.pt
FRAME_CACHE=  data/freeorder/frame_cache_200bin.pt
QM9_CACHE  =  data/freeorder/qm9_mols_cache_v3b_noh.pkl
```

## 3. Training (paper E562 を再生成、from scratch)

paper ckpt が消失した場合、または別 GPU で再現する場合:

```bash
# 既定 600 epoch、~16.5h on RTX 4090
~/ADT/QM9/src/freeorder/run_paper_qm9_train.sh

# epoch 短縮 / 別 save_dir を指定する場合
EPOCHS=200 SAVE_DIR=/tmp/qm9_test ~/ADT/QM9/src/freeorder/run_paper_qm9_train.sh
```

詳細コマンド (env 内訳):

```bash
cd ~/ADT/QM9/src/freeorder
~/.venv/bin/python3 fo_train_v26.py \
    --epochs 600 --batch_size 128 --lr 2e-4 --warmup_epochs 5 \
    --d_model 768 --n_layers 12 --n_heads 8 --d_ff 3072 \
    --dropout 0.2 --amp bf16 --nohydrogen --dynamic \
    --max_offset 32 --num_workers 4 \
    --save_every 10 --eval_after 9999 --eval_every 9999 \
    --save_dir ~/ADT/QM9/data/freeorder/checkpoints_v26_repro \
    --frame_cache ~/ADT/QM9/data/freeorder/frame_cache_200bin.pt \
    > ~/ADT/QM9/data/freeorder/train_v26_repro.log 2>&1
```

時間: 600 epoch × 99 s ≈ **16.5 h on RTX 4090**

期待最終: `best.pt` が `epoch≈562, val≈2.75` 付近で保存。
seed が同じでなければ **完全一致は不可**、val_loss が ±0.005 以内、`mol_stable` が ±2pt 以内ならOK。

学習 dataset (QM9 134k SMILES + 3D coords) は `qm9_mols_cache_v3b_noh.pkl` 内にキャッシュ済 (1モル = (RDKit Mol, positions, SMILES) のリスト)。

## 4. Fig 6 (training curves、App F)

paper Fig 6 の QM9 panel (epoch 1〜600 の train/val cross-entropy 曲線) は **既存の training log から数値抽出** して TikZ に貼り付け。

ソース log:
- 元 paper 用 (kr3 で 4/19-4/20 学習): `~/ADTbackup/ADT/data/QM9/freeorder_backup/.../nohup_v26_qm9.out` (1 GB の trash 内、または kr3:`~/ADT/trash/QM9_paper_era_unused/nohup_v26_qm9.out`)
- フォーマット: `EXXX [HH:MM:SS] | train=Y.YYY val=Z.ZZZ | ...`

抽出スクリプト: `~/ADTbackup/ADT/ADTpaper1/scripts/extract_curves.py` (kr2 + t14 backup あり)。

新しく学習し直した場合は、`train_v26_repro.log` から同様に抽出 (paper の epoch 軸点は 1, 16, 36, 76, 126, 176, 236, 296, 356, 416, 476, 536, 600 の 13 点)。

## 5. 検証 (paper 数値再現の確認)

paper Table 2 (QM9 比較表):

| Method | mol_stable |
|---|---|
| EDM (diffusion) | 82.0% |
| MiDi (diffusion) | 84.0% |
| GeoLDM (latent diffusion) | 89.4% |
| Quetzal (transformer+diffusion) | 90.4% |
| InertialAR (transformer+diffusion) | 94.7% |
| **ADT (ours, E562)** | **87.5%** ← Table 2 |

paper Table 3 (ADT 詳細 N=10000):

| 指標 | count | % |
|---|---|---|
| mol_stable / N | 8746/10000 | 87.5% |
| unique / mol_stable | 8508/8746 | 97.3% |
| novel / unique | 3635/8508 | 42.7% |
| kekulizable | 8746/8746 | 100% |
| xTB converged | 8743/8746 | 100.0% |
| **topology preserved / N** | **8566/10000** | **85.7%** ← xVR |
| RMSD heavy median | 0.184 Å | |
| Energy gain median | −1.31 eV | |

GEN_N=10000 で再現すれば全ての値が ±2pt (mol_stable / topology / novel) 以内に収まる。

## 6. 構成

```
~/ADT/QM9/
├── README.md                            ← このファイル
├── src/
│   ├── freeorder/                       paper 必須 code (5 .py + runner)
│   │   ├── fo_train_v26.py              paper trainer
│   │   ├── gen_eval_v26.py              paper evaluator (Phase1+Phase2)
│   │   ├── gen_eval_parallel.py         N=10000 並列評価 (任意)
│   │   ├── fo_tokenizer.py
│   │   └── run_paper_qm9_eval.sh        runner
│   └── freeorder_inferior/              paper 非依存 code (旧 logmesh era)
└── data/
    ├── freeorder/                       paper 必須 data (1.1 GB)
    │   ├── checkpoints_v26_scratch/
    │   │   └── best.pt                  paper E562 (1.0 GB、md5 17ec74f1...)
    │   ├── frame_cache_200bin.pt        76 MB (log-spaced 200 bins、log mesh)
    │   ├── qm9_mols_cache_v3b_noh.pkl   56 MB (QM9 dataset cache)
    │   └── eval_v26_E562_N1000/         検証済み eval 結果
    └── freeorder_inferior/              paper 非依存 data (logmesh 試行)

~/ADT/common/                            共有 lib (master HEAD)
├── adt_model.py                         ★ offset pointer 対応 (paper E562 model)
├── adt_tokenizer.py                     ★ log mesh 200 bins 既定
├── adt_dataset.py
├── util_validation.py
├── collision_check.py
├── relative_pointer.py
├── generate.py / gen_eval_lib.py
├── xtb_post_process.py
├── make_browse_html.py
└── pubchem_novelty.py
```

## 7. paper 採用 config (要点)

| key | value |
|---|---|
| epoch | **562** (val=2.7496) |
| trainer | `fo_train_v26.py --max_offset 32` |
| pointer | `output_pointer_mode=offset`, `max_pointer=30` |
| r mesh | **log-spaced, R_BINS=200** (frame_cache 名 `200bin` だが log mesh) |
| HP bins | 12 × 16 × 16 |
| AROMATIZE_RINGS | **0** |
| 学習機 | kr3 RTX 4090 ×1 |
| 学習日時 | 2026-04-19 13:19 〜 2026-04-20 06:00 (~16.5 h) |
| md5 (best.pt) | `17ec74f188b0dd4c216ed05dab9c85a2` |

## 8. inferior が paper でない理由

`freeorder_inferior/` 内の ckpt:
- `eval_logmesh_*` の元になった `checkpoints_fo_wide_logmesh/best.pt` (E570、val=2.785、kr1 4/7-4/8 学習)
  - 構成: pre-v26、**absolute pointer** (offset なし)、R_BINS=100
  - paper Fig 6 の loss 曲線とは log が完全一致するが、**Table 2 の 88% は出ない (78.5% 止まり)**
- `checkpoints_fo_wide_200bin_repro/best.pt` (E298、val=2.779、kr1 4/11-4/12 学習)
  - 構成: R_BINS=200 だが offset pointer なし、300 epoch 止まり
  - Table 2 では使用されず (eval_N10000 経由で 86.7% 到達)

**paper 88% を出せるのは v26 era ckpt (`max_offset=32` + `output_pointer_mode=offset` + R_BINS=200) のみ**。これは `kr3:checkpoints_v26_scratch/best.pt` (E562) で確定。

## 9. 関連

- Drugs paper (Tab 4 / Tab 5): `~/ADT/Drugs/{src,data}/freeorder_v26/` 配下に同様の構造で整備予定 (kr2 にデータあり、kr1 へ統合は未着手)
- 共有 library: `~/ADT/common/` (Drugs と共通)
