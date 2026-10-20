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
  echo "GPU ${gpu} is not free (memory.used=${used_mib:-unknown} MiB)." >&2
  exit 2
fi

bash tools/run_volumetric_smpl_frame_batching_gate.sh "${gpu}"
query_gate="output/volumetric_smpl_current_query_chunk_assessment.json"
frame_gate="output/volumetric_smpl_current_frame_batch_assessment.json"
frame_size="$(python -c "import json; print(json.load(open('${frame_gate}'))['selected_candidate']['frame_batch_size'])")"
query_size="$(python -c "import json; print(json.load(open('${query_gate}'))['query_chunk_authorized'])")"
if [[ ! "${frame_size}" =~ ^[0-9]+$ ]] || (( frame_size < 2 )); then
  echo "No current frame batching candidate was authorized." >&2
  exit 2
fi

output="output/interx_shape_aug_volumetric_f32_exact_pilot25_seed20260828"
/home/wenning/miniconda3/bin/conda run --no-capture-output -n New_MMM \
  python -u tools/run_interx_shape_augmentation.py \
  --output-dir "${output}" --mode optimized --device "cuda:${gpu}" \
  --selection-seed 20260828 --limit 25 --positive-timing-only \
  --num-epochs 200 --learning-rate 0.01 --fine-stage-start 100 \
  --signed-distance-backend volumetric_smpl --query-chunk "${query_size}" \
  --volumetric-smpl-frame-batch-size "${frame_size}" \
  --foot-contact-mask-policy released_dynamic \
  --contact-loss-reduction-policy original_mmm \
  --optimization-variable-policy released_all_per_frame

/home/wenning/miniconda3/bin/conda run --no-capture-output -n New_MMM \
  python -u tools/assess_shape_pilot_raw.py \
  --random-dir output/interx_shape_aug_random_seed123_v2 \
  --optimized-dir "${output}" \
  --contact-dir dataset/interx/contact_pairs_codex_luna_strict_v3_train \
  --output "${output}_assessment.json" --device "cuda:${gpu}"

python tools/assess_volumetric_shape_pilot.py \
  --nearest-completion output/interx_shape_aug_original_mmm_pilot25_seed20260828/_completion_manifest.json \
  --volumetric-completion "${output}/_completion_manifest.json" \
  --assessment "${output}_assessment.json" \
  --query-gate "${query_gate}" --frame-gate "${frame_gate}" \
  --output "${output}_gate.json"

echo "VolumetricSMPL pilot25 passed mismatch, skating, and one-day runtime gates."
