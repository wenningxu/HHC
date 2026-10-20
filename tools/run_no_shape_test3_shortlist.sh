#!/usr/bin/env bash
set -euo pipefail

cd /mnt/8T-HDD/wenning/HHC

if [[ $# -ne 1 || ! "$1" =~ ^[0-9]+$ ]]; then
  echo "usage: $0 GPU_INDEX" >&2
  exit 2
fi

gpu="$1"
run_dir="output/t2i_trans/2026-09-07-11-49-54_hhc_historical_paper_mask_no_mgt_shape_pilot50k_seed114514_v4"

used_mib="$(nvidia-smi --id="${gpu}" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')"
if [[ ! "${used_mib}" =~ ^[0-9]+$ ]] || (( used_mib > 2048 )); then
  echo "GPU ${gpu} is not free (memory.used=${used_mib:-unknown} MiB)." >&2
  exit 2
fi

/home/wenning/miniconda3/bin/conda run --no-capture-output -n New_MMM \
  python -u tools/select_no_shape_checkpoint_on_test3.py \
  --run-dir "${run_dir}" \
  --candidate-steps 80000 220000 260000 280000 300000 \
  --output-dir output/hhc_no_shape_v4_test3_checkpoint_selection \
  --output output/hhc_no_shape_v4_test3_shortlist_selection.json \
  --device "cuda:${gpu}" --reuse-step30-formal
