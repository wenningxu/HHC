#!/usr/bin/env bash
set -euo pipefail

cd /mnt/8T-HDD/wenning/HHC

if [[ $# -ne 2 || ! "$1" =~ ^[0-9]+$ || ! "$2" =~ ^[01]$ ]]; then
  echo "usage: $0 GPU_INDEX SHARD_INDEX_0_OR_1" >&2
  exit 2
fi
gpu="$1"
shard="$2"
used_mib="$(nvidia-smi --id="${gpu}" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')"
if [[ ! "${used_mib}" =~ ^[0-9]+$ ]] || (( used_mib > 1024 )); then
  echo "GPU ${gpu} is not free (memory.used=${used_mib:-unknown} MiB)." >&2
  exit 2
fi

gate="output/interx_shape_aug_volumetric_fast_selective_pilot25_gate.json"
python -c "import json; p=json.load(open('${gate}')); assert p['status']=='pass' and all(p['checks'].values())"

limit=2934
offset=$(( shard * limit ))
output="output/interx_shape_aug_volumetric_fast_steps120_full_shard${shard}"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
/home/wenning/miniconda3/bin/conda run --no-capture-output -n New_MMM \
  python -u tools/run_interx_shape_augmentation.py \
  --plan configs/interx_shape_augmentation/timing_covered_seed123_v2/plan_manifest.json \
  --contact-dir dataset/interx/contact_pairs_codex_luna_strict_v3_train \
  --output-dir "${output}" --mode optimized --device "cuda:${gpu}" \
  --selection-seed 20260828 --offset "${offset}" --limit "${limit}" \
  --positive-timing-only --num-epochs 120 --learning-rate 0.01 \
  --fine-stage-start 100 --signed-distance-backend volumetric_smpl \
  --query-chunk 2048 --volumetric-smpl-frame-batch-size 4 \
  --volumetric-smpl-penetration-vertex-count 256 \
  --volumetric-smpl-penetration-frame-count 16 \
  --volumetric-smpl-body-encoding-samples 64 \
  --foot-contact-mask-policy released_dynamic \
  --contact-loss-reduction-policy original_mmm \
  --optimization-variable-policy released_all_per_frame
