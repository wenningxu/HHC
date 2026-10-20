#!/usr/bin/env bash
set -euo pipefail

cd /mnt/8T-HDD/wenning/HHC

echo "Archived invalid full-shard launcher: it encodes the superseded 100-iteration/freeze@50 custom foot-mask, loss-reduction, and variable policies. Keep only as historical backup; use run_original_mmm_shape_pilot25.sh and create a new full launcher only after its gate passes." >&2
exit 2

if (( $# != 2 )); then
  echo "usage: $0 GPU_ID SHARD_INDEX" >&2
  exit 2
fi
gpu_id=$1
shard_index=$2
if (( shard_index < 0 || shard_index > 7 )); then
  echo "SHARD_INDEX must be in [0, 7]" >&2
  exit 2
fi

python -c "import json; d=json.load(open('output/interx_shape_aug_full_single_process_topology_audit.json')); assert d.get('status') == 'pass' and d['decision']['formal_topology'] == 'one_process_per_gpu' and d['decision']['frame_batch_size'] == 8 and d['decision']['query_chunk'] == 10475"
python -c "import json; d=json.load(open('output/interx_shape_aug_groundfix_groundedchain_fixedroot_pilot25_body55floor_assessment_seed20260828.json')); assert d.get('status') == 'pass' and d.get('full_augmentation_authorized') is True"
python -c "import json; d=json.load(open('output/interx_shape_augmentation_disk_capacity_audit_20260902.json')); assert d.get('status') == 'pass'"

offset=$((shard_index * 780))
limit=780
if (( shard_index == 7 )); then
  limit=778
fi

env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  conda run --no-capture-output -n New_MMM python -u \
  tools/run_interx_shape_augmentation.py \
  --plan configs/interx_shape_augmentation/timing_covered_seed123_v2/plan_manifest.json \
  --contact-dir dataset/interx/contact_pairs_codex_luna_strict_v3_train \
  --mode optimized --device "cuda:${gpu_id}" --query-chunk 10475 \
  --volumetric-smpl-frame-batch-size 8 --seed 1234 \
  --selection-seed 20260828 \
  --num-epochs 100 --fine-stage-start 50 \
  --foot-contact-mask-policy initial_or_current_grounded_horizontal_mean_speed \
  --contact-loss-reduction-policy active_contact_mean \
  --optimization-variable-policy grounded_chain_fixed_root \
  --offset "$offset" --limit "$limit" \
  --output-dir "output/interx_shape_aug_full_batch8_query10475_seed20260828_shard${shard_index}"
