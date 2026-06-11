# Drugs paper 再現 — `adt.tex` Tab 4 / Tab 5 (scaffold-conditional generation)

GEOM-Drugs scaffold-conditional generation の paper 数値を再現する。
配置: `~/ADT/Drugs/{src,data}/{freeorder_v26,freeorder_v26_inferior}/`
共有 lib: `~/ADT/common/`

## paper 数値 (既存 N=10000 結果から再生成済)

### Tab 4 (30-atom、E142、scratch)

| Scaffold | mol_stable / N | unique / kek | xVR (topo) / N |
|---|---|---|---|
| benzene | 3907/10000 (39.07%) | 3216/3217 (100.0%) | 2966/10000 (29.66%) |
| pyridine | 1602+ | 1600/1602 (99.9%) | — |
| pyrimidine | 1625+ | 1599/1625 (98.4%) | — |
| pyrazine | 1255+ | 1236/1255 (98.5%) | — |
| furan | 2106+ | 2092/2106 (99.3%) | — |
| thiophene | 2308+ | 2293/2308 (99.4%) | — |
| cyclohexane | 1637+ | 1628/1637 (99.5%) | — |

(per_scaffold_stats.py で v26s_scaffolds_n10k から再集計可)

### Tab 5 (50-atom、E157、ft from 30-atom)

| Scaffold | unique / kek | mean atoms |
|---|---|---|
| benzene | 1941/1942 (99.9%) | 28.5 |
| pyridine | 764/770 (99.2%) | 27.5 |
| pyrimidine | 851/866 (98.3%) | 26.5 |
| ... (per_scaffold_stats.py で v26g_scaffolds_n10k から取得) |


> **Note**: 以下のコマンド内の `$ADT_ROOT` は repo の root (この README のあるディレクトリ = `Drugs/` の親 = clone 先) を指す。runner shell 内では自動解決、Python 例コマンドは `export ADT_ROOT=/path/to/clone` してから実行 (デフォルト `$HOME/ADT` 想定)。

## 0. 前提

- **GPU**: RTX 4090 ×2 推奨 (paper 学習 365 s/epoch DDP)。生成は 4090 ×1 OK
- **Python env**: `~/.venv/bin/python3` (torch + rdkit + xtb + meeko 等)
- **共有 lib**: `~/ADT/common/` (master HEAD、offset pointer 対応)

## 1. paper ckpts の確認

```bash
ls -la ~/ADT/Drugs/data/freeorder_v26/checkpoints_v26_scratch/{best.pt,E142.pt}
ls -la ~/ADT/Drugs/data/freeorder_v26/checkpoints_v26_max50_ft_v2/best.pt
md5sum ~/ADT/Drugs/data/freeorder_v26/checkpoints_v26_scratch/E142.pt
# 期待値: 451ffc3c2f2b6c06cfefe16e52c488e6 (paper Tab 4 採用)

~/.venv/bin/python -c '
import torch
ck = torch.load("$ADT_ROOT/Drugs/data/freeorder_v26/checkpoints_v26_scratch/E142.pt", map_location="cpu", weights_only=False)
print("epoch:", ck["epoch"], "val:", ck["val_loss"])
# → epoch=142, val=1.980 (paper採用、val=1.949 の best.pt = E156 ではない)
'
```

| ckpt | epoch | val | 用途 |
|---|---|---|---|
| `checkpoints_v26_scratch/E142.pt` | **142** (val=1.9797) | paper Tab 4 採用 (30-atom scratch) |
| `checkpoints_v26_scratch/best.pt` | 156 (val=1.949) | paper未採用 (best.pt は E156) |
| `checkpoints_v26_max50_ft_v2/best.pt` | **157** (val ~1.81) | paper Tab 5 採用 (50-atom ft from 30-atom E142) |

## 2. Generation + Evaluation (Tab 4 / Tab 5)

```bash
# 30-atom (Tab 4) — 7 scaffold × N=10000、~6 h
MAX=30 GEN_N=10000 ~/ADT/Drugs/src/freeorder_v26/run_paper_drugs_eval.sh

# 50-atom (Tab 5) — 7 scaffold × N=10000、~10 h (大分子で gen 遅い)
MAX=50 GEN_N=10000 ~/ADT/Drugs/src/freeorder_v26/run_paper_drugs_eval.sh

# 確認用 N=1000 (各 scaffold ~6 min)
MAX=30 GEN_N=1000 ~/ADT/Drugs/src/freeorder_v26/run_paper_drugs_eval.sh
```

実行内容: 7 scaffold (`benzene, pyridine, pyrimidine, pyrazine, furan, thiophene, cyclohexane`) を順次、`gen_eval_v26.py` で生成 + xtb。

出力: `~/ADT/Drugs/data/freeorder_v26/v26{s,g}_scaffolds_n${GEN_N}_repro/<scaffold>/`
- `gen_stats.json` / `stats.json`
- `mol_stable_kekulized.sdf`
- `xtb_results.json`
- 等

集計:
```bash
cd ~/ADT/ADTpaper1/scripts
~/.venv/bin/python per_scaffold_stats.py    # Tab 4 / Tab 5 / App B
~/.venv/bin/python success_examples.py       # App C
~/.venv/bin/python failure_breakdown.py      # App D
~/.venv/bin/python failure_examples.py       # App E
~/.venv/bin/python bond_angle_errors.py      # Fig 5
~/.venv/bin/python geom_drugs_v26_dist.py    # Fig 4
```

## 3. Training (paper E142 / E157 を再生成、from scratch)

### Runner (簡易):

```bash
# 30-atom scratch (paper Tab 4 用、E142、~14h on 4090×2 DDP)
MAX=30 ~/ADT/Drugs/src/freeorder_v26/run_paper_drugs_train.sh

# 50-atom ft from 30-atom (paper Tab 5 用、E157、~5h on 4090×2 DDP)
MAX=50 ~/ADT/Drugs/src/freeorder_v26/run_paper_drugs_train.sh
```

### 詳細コマンド (env 内訳):

### Drugs 30-atom (E142, scratch)

```bash
cd ~/ADT/Drugs/src/freeorder_v26
torchrun --nproc_per_node=2 fo_train_v26.py \
    --max_atoms 30 --max_pointer 64 --max_offset 50 --require_benzene 0 \
    --data_cache ~/ADT/Drugs/data/freeorder_v26/drugs_mols_v26_max30.pkl \
    --frame_cache ~/ADT/Drugs/data/freeorder_v26/frame_cache_v21_benzene.pt \
    --epochs 300 --batch_size 64 --lr 2e-4 --warmup_epochs 5 \
    --d_model 768 --n_layers 12 --n_heads 8 --d_ff 3072 \
    --dropout 0.2 --amp bf16 --nohydrogen --dynamic \
    --save_dir ~/ADT/Drugs/data/freeorder_v26/checkpoints_v26_scratch_repro \
    --save_every 5 --num_workers 4 \
    --eval_after 9999 --eval_every 9999
# → ~14 h on 4090×2 DDP (365s/epoch × 142 epoch)、E142 で best.pt 切り替わる前に確保
```

paper の E142 は val=1.980 で記録、後に E156 (val=1.949) で best.pt 上書き。**E142 を逃さないため `--save_every 5` で epoch_140.pt / epoch_145.pt を保存**して E142 付近を後で抽出。

### Drugs 50-atom (E157, ft from 30-atom)

```bash
cd ~/ADT/Drugs/src/freeorder_v26
torchrun --nproc_per_node=2 fo_train_v26.py \
    --max_atoms 50 --max_pointer 64 --max_offset 50 --require_benzene 0 \
    --data_cache ~/ADT/Drugs/data/freeorder_v26/drugs_mols_v26_max50.pkl \
    --pretrain ~/ADT/Drugs/data/freeorder_v26/checkpoints_v26_scratch/best.pt \
    --frame_cache ~/ADT/Drugs/data/freeorder_v26/frame_cache_v21_benzene.pt \
    --epochs 30 --batch_size 48 --lr 5e-5 --warmup_epochs 0 \
    --d_model 768 --n_layers 12 --n_heads 8 --d_ff 3072 \
    --dropout 0.2 --amp bf16 --nohydrogen --dynamic \
    --save_dir ~/ADT/Drugs/data/freeorder_v26/checkpoints_v26_max50_ft_v2_repro \
    --save_every 1 --num_workers 4 \
    --eval_after 9999 --eval_every 9999
# → ~5 h on 4090×2 DDP、~5 epoch で plateau、E157 で best.pt
```

## 4. Fig 4 / Fig 5 集計 (heavy-atom dist / bond-angle errors)

Fig 4 (heavy-atom dist):
```bash
cd ~/ADT/ADTpaper1/scripts
~/.venv/bin/python geom_drugs_v26_dist.py    # ~/ADT/Drugs/data/freeorder_v26/drugs_mols_v26_max50.pkl 読む
~/.venv/bin/python v21_nolimit_hist.py       # drugs_mols_v21_nolimit.pkl 読む
```

Fig 5 (bond/angle errors):
```bash
cd ~/ADT/ADTpaper1/scripts
~/.venv/bin/python bond_angle_errors.py      # v26[sg]_scaffolds_n10k/<scaffold>/xtb_results.json 読む
~/.venv/bin/python bond_angle_hist.py
```

App F Fig (training curves) は `extract_curves.py` で paper 学習 log (`~/ADT/Drugs/data/freeorder_v26/logs/fo_train_20260420_155621.log` + `nohup_v26_max50_ft_v2.out`) から抽出。

## 5. 構成

```
~/ADT/Drugs/
├── README.md                                ← このファイル
├── src/
│   ├── freeorder_v26/                       ★ paper 必須 code
│   │   ├── fo_train_v26.py                  paper trainer
│   │   ├── gen_eval_v26.py                  paper evaluator (Phase1+Phase2)
│   │   ├── build_virtual_seeds.py           virtual seed sampler
│   │   └── run_paper_drugs_eval.sh          paper 再現 runner (per scaffold)
│   └── freeorder_v26_inferior/              旧 code (legacy_freeorder, v26benzentest, train.py 等)
└── data/
    ├── freeorder_v26/                       ★ paper 必須 data (~3.5 GB)
    │   ├── checkpoints_v26_scratch/
    │   │   ├── best.pt                      E156 (paper未採用)
    │   │   └── E142.pt                      ★ paper Tab 4 採用 (md5 451ffc3c...)
    │   ├── checkpoints_v26_max50_ft_v2/
    │   │   └── best.pt                      ★ paper Tab 5 採用 = E157 (md5 0a569723...)
    │   ├── v26s_scaffolds_n10k/             ★ Tab 4 N=10000 (7 scaffold)
    │   ├── v26g_scaffolds_n10k/             ★ Tab 5 N=10000 (7 scaffold)
    │   ├── scaffolds/                       7 scaffold frame_cache
    │   ├── frame_cache_v21_benzene.pt       generic frame_cache
    │   ├── drugs_mols_v26_max30.pkl         training data (30-atom)
    │   ├── drugs_mols_v26_max50.pkl         training data (50-atom)
    │   ├── drugs_mols_v21_nolimit.pkl       Fig 4 heavy-atom dist 用
    │   ├── browse_v21_displacement.html     paper QR 用 demo
    │   ├── nohup_v26_max50_ft_v2.out        50-atom ft 学習 nohup
    │   └── logs/                            paper 学習 log
    │       ├── fo_train_20260420_155621.log     paper 30-atom (E142)
    │       └── fo_train_20260421_141429.log     paper 50-atom ft (E157)
    └── freeorder_v26_inferior/              paper 非依存 data
        ├── logs/                            v22 / v23 era log (4/15-4/16, 4/20 11:48)
        └── legacy_logs/                     更に古い (3/30-3/31)

~/ADT/ADTpaper1/scripts/                     paper Fig/Table 集計 scripts
├── per_scaffold_stats.py                    Tab 4 / Tab 5 / App B
├── success_examples.py                      App C
├── failure_breakdown.py / failure_examples.py    App D / E
├── bond_angle_errors.py / bond_angle_hist.py     Fig 5
├── geom_drugs_v26_dist.py / v21_nolimit_hist.py  Fig 4
├── heavy_atom_stats.py / heavy_atom_stats_v26g.py
└── extract_curves.py                        App F training curves
```

## 6. paper 採用 config (要点)

| key | 30-atom (Tab 4) | 50-atom (Tab 5) |
|---|---|---|
| epoch | **142** (E142.pt) | **157** (best.pt of `_v2`) |
| val_loss | 1.9797 | ~1.81 |
| max_offset | 50 | 50 |
| max_atoms | 30 | 50 |
| max_pointer | 64 | 64 |
| pointer | offset (v26 hybrid) | offset (v26 hybrid) |
| r mesh | log 200 bins | log 200 bins |
| AROMATIZE_RINGS (eval) | **1** (Drugs default、QM9 と異なる) | 1 |
| 学習 GPU | 4090 ×2 DDP, 365 s/epoch | 4090 ×2 DDP, 563 s/epoch |
| 学習日時 | 2026-04-20 15:56 〜 4/21 03:00 (~14h) | 2026-04-21 14:14 〜 18:00 (~3.5h, ~5 epoch plateau) |

## 7. なぜ inferior なのか

`freeorder_v26_inferior/`:
- `legacy_freeorder/` — paper 以前の Drugs FO トレーナー (v21/v22 era)
- `v26benzentest/` — benzene-only テスト学習 (paper未採用)
- `legacy_logs/` — 3/30-3/31 (v21 era 学習)
- `logs/` — 4/15-4/16 (v22) + 4/20 11:48 (v26 attempt with --pretrain v23 init、paper の 4/20 15:56 scratch とは別)

`v26_scratch/best.pt` (E156) も paper非採用だが ckpt として保持 (E142 export 元の training run の最終 best)。

## 8. 関連

- QM9 paper (Tab 6/7、Fig 6): `~/ADT/QM9/` (kr1 にある同様の構造)
- Pocket-ADT 続編 (v27〜v32): `~/ADT/Pocket/` (kr2、別軸プロジェクト、paper には登場しない)
- 共有 lib: `~/ADT/common/` (Drugs と QM9 で共通、master HEAD snapshot)
