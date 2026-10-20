#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]] || [[ ! "$1" =~ ^[0-9]+$ ]]; then
  echo "usage: $0 GPU_INDEX" >&2
  exit 2
fi

hhc_root="/mnt/8T-HDD/wenning/HHC"
interact2ar_root="/mnt/8T-HDD/wenning/Interact2Ar"
output_dir="${hhc_root}/output/interact2ar_author_release_calibration"
mkdir -p "${output_dir}"

cd "${hhc_root}"
/home/wenning/miniconda3/bin/conda run --no-capture-output -n New_MMM \
  python tools/audit_interact2ar_paper_repo_data.py
cp output/interact2ar_paper_repo_data_audit.json "${output_dir}/prerequisite_audit.json"

cd "${interact2ar_root}"
# This is intentionally the README command, including GT first-frame
# conditioning and the released script's native five repetitions.  It is a
# calibration of the public author release, not the unconditional fair table.
uv run python src/evaluate.py \
  --dataset paper.yaml \
  --checkpoint checkpoints/interact2ar_mixed_memory.ckpt \
  --full \
  --deterministic \
  --first_frame \
  --device cuda \
  --gpu "$1" 2>&1 | tee "${output_dir}/stdout.log"

cd "${hhc_root}"
/home/wenning/miniconda3/bin/conda run --no-capture-output -n New_MMM \
  python tools/parse_interact2ar_author_eval_log.py \
  --log "${output_dir}/stdout.log" \
  --audit "${output_dir}/prerequisite_audit.json" \
  --output "${output_dir}/summary.json"
