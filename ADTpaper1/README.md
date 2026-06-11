# ADT paper 1 (adt.tex) - data & scripts集約

すべて kr2 上で v26 era の ADT paper (`Documents/paper/ADT/adt.tex`) 用。

## 構成

- `scripts/` - Fig/Table 集計スクリプト 11 本 (kr2 /tmp から退避)
- `logs/` - 学習ログ (実体は元位置、symlink)

## eval データの正式置き場 (2026-05-07 移動)

scaffold 生成結果は `/tmp` でも `ADTpaper1/` でもなく **`~/ADT/data/Drugs/freeorder_v26/` 直下** に常駐:

- `~/ADT/data/Drugs/freeorder_v26/v26s_scaffolds_n10k/` (Drugs 30-atom × 7 scaffold, E142, 99M)
- `~/ADT/data/Drugs/freeorder_v26/v26g_scaffolds_n10k/` (Drugs 50-atom × 7 scaffold, E157, 75M)

scripts は上記絶対パスを直接読む。

## ckpt 場所 (移動せず)

- QM9 E562: ~/ADT/data/QM9/freeorder/checkpoints_fo_wide_200bin_repro/
- Drugs 30 E142: ~/ADT/data/Drugs/freeorder_v23/checkpoints_v26_scratch/
- Drugs 50 E157: ~/ADT/data/Drugs/freeorder_v26/checkpoints_v26_max50_ft_v2/

## scripts → 論文要素 対応

- bond_angle_errors.py / bond_angle_hist.py → Fig 5 (bond/angle error 分布)
- geom_drugs_v26_dist.py / v21_nolimit_hist.py → Fig 4 (重原子分布)
- extract_curves.py → App F (training curves)
- heavy_atom_stats.py / heavy_atom_stats_v26g.py → 重原子分布 pooled
- per_scaffold_stats.py → Tbl 4/5, App B
- success_examples.py → App C (SMILES + ΔE_xTB + RMSD_h)
- failure_breakdown.py → App D
- failure_examples.py → App E

## 欠落

- hist_v26.py / aggregate_stats.py: kr1/kr2 /tmp とも消失。再構築は同種スクリプトから容易。
