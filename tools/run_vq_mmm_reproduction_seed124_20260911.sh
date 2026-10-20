#!/usr/bin/env bash
set -euo pipefail
cd /mnt/8T-HDD/wenning/HHC

# Independent seed replication; unchanged model/loss/schedule. Not January RNG recovery.
# Provenance and predeclared acceptance: docs/VQ_MMM_REPRODUCTION_AUDIT.md.
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
/home/wenning/miniconda3/envs/New_MMM/bin/python tools/check_vq_mmm_fp32_parity.py
exec /home/wenning/miniconda3/envs/New_MMM/bin/python -u train/train_vq_interx.py \
  --device cuda:0 \
  --exp-name hhc_original_mmm_vq_reproduction_seed124_300k_fp32_20260911_v1 \
  --checkpoint-dir checkpoints/hhc_original_mmm_vq_reproduction_seed124_300k_fp32_20260911_v1 \
  --train-split-file dataset/interx/splits/train.txt \
  --train-motion-file dataset/interx/motions/train.h5 \
  --validation-split-file dataset/interx/splits/val.txt \
  --validation-motion-file dataset/interx/motions/val.h5 \
  --stats-dir dataset/interx --stats-prefix Interx \
  --nb-code 12288 --code-dim 256 --output-emb-width 256 \
  --width 512 --depth 3 --down-t 2 --stride-t 2 --dilation-growth-rate 3 \
  --batch-size 256 --window-size 64 --seed 124 --precision fp32 \
  --total-iter 300000 --warm-up-iter 1000 \
  --schedule-mode historical_separate_warmup_cosine \
  --lr 0.0002 --weight-decay 0.00002 --mu 0.99 --commit 0.02 \
  --recons-loss l1_smooth --loss_explicit 3 --loss_vel 100 \
  --loss_root 0.1 --loss_geo 0.01 --loss_fc 10 --loss_distance_map 0.1 \
  --loss_inter_oriental 0 --shape-mask-prob 0.1 \
  --eval-iter 5000 --print-iter 200 --wandb-mode online
