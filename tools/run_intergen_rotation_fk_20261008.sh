#!/usr/bin/env bash
set -euo pipefail
cd /mnt/8T-HDD/wenning/HHC
mkdir -p output/intergen_rotation_fk_20261008
test ! -e output/intergen_rotation_fk_20261008/run.log
exec >output/intergen_rotation_fk_20261008/run.log 2>&1
trap 'rotation_exit=$?; echo "$rotation_exit" >output/intergen_rotation_fk_20261008/exit_code' EXIT
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
/home/wenning/miniconda3/envs/New_MMM/bin/python -u tools/train_intergen_rotation.py \
  --output /home/wenning/HHC_interhuman_runs/intergen_rotation_fk_seed20261008 \
  --steps 10000 --micro-batch 4 --accumulation 8 --lr 1e-5 --eval-every 2000 --device cuda:0
