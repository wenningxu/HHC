#!/usr/bin/env bash
set -euo pipefail

cd /mnt/8T-HDD/wenning/HHC

if [[ $# -ne 1 || ! "$1" =~ ^[0-9]+$ ]]; then
  echo "usage: $0 GPU_INDEX" >&2
  exit 2
fi
gpu="$1"
used_mib="$(nvidia-smi --id="${gpu}" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')"
if [[ ! "${used_mib}" =~ ^[0-9]+$ ]] || (( used_mib > 1024 )); then
  echo "GPU ${gpu} is not free (memory.used=${used_mib:-unknown} MiB)" >&2
  exit 2
fi

run_dir="output/t2i_trans/2026-09-07-11-49-54_hhc_historical_paper_mask_no_mgt_shape_pilot50k_seed114514_v4"
checkpoint="${run_dir}/model-000300000-0.3973.ckpt"

/home/wenning/miniconda3/bin/conda run --no-capture-output -n New_MMM \
  python -u tools/audit_mgt_mask_exposure_gap.py \
  --training-manifest "${run_dir}/training_manifest.json" \
  --historical-checkpoint "${checkpoint}" \
  --output output/hhc_no_shape_full300k_logit_sharpness_validation.json \
  --device "cuda:${gpu}" --samples 256 --batch-size 16 --seed 90123 \
  --split-role validation --sharpness-temperature 0.5
