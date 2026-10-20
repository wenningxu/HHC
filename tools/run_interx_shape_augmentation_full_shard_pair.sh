#!/usr/bin/env bash
set -euo pipefail

cd /mnt/8T-HDD/wenning/HHC

echo "Archived invalid paired-shard launcher: it encodes the superseded 100-iteration/freeze@50 custom foot-mask, loss-reduction, and variable policies. Keep only as historical backup; use run_original_mmm_shape_pilot25.sh and create a new full launcher only after its gate passes." >&2
exit 2

if (( $# != 2 )); then
  echo "usage: $0 GPU_ID PAIR_INDEX" >&2
  exit 2
fi
gpu_id=$1
pair_index=$2
if (( pair_index < 0 || pair_index > 3 )); then
  echo "PAIR_INDEX must be in [0, 3]" >&2
  exit 2
fi

python -c "import json; d=json.load(open('output/interx_shape_aug_groundfix_groundedchain_fixedroot_pilot25_body55floor_assessment_seed20260828.json')); assert d.get('status') == 'pass' and d.get('full_augmentation_authorized') is True and d.get('authorized_action') == 'full_augmentation'"
python -c "import json; d=json.load(open('output/interx_shape_aug_shard_seed_concurrency2_batch8_audit.json')); assert d.get('status') == 'pass' and d.get('accepted_topology') == 'two_processes_one_gpu_contiguous_shards' and d.get('formal_frame_batch_size') == 8 and d.get('formal_query_chunk') == 10475"
python -c "import json; d=json.load(open('output/interx_shape_augmentation_disk_capacity_audit_20260902.json')); assert d.get('status') == 'pass'"
python -c "import json; d=json.load(open('dataset/interx/contact_pairs_codex_luna_strict_v3_train/_extraction_manifest.json')); assert d.get('status') == 'complete' and d.get('split_count') == d.get('completed_count') == 9110"

common=(
  tools/run_interx_shape_augmentation.py
  --plan configs/interx_shape_augmentation/timing_covered_seed123_v2/plan_manifest.json
  --contact-dir dataset/interx/contact_pairs_codex_luna_strict_v3_train
  --mode optimized --device "cuda:${gpu_id}" --query-chunk 10475
  --volumetric-smpl-frame-batch-size 8 --seed 1234
  --selection-seed 20260828
  --num-epochs 100 --fine-stage-start 50
  --foot-contact-mask-policy initial_or_current_grounded_horizontal_mean_speed
  --contact-loss-reduction-policy active_contact_mean
  --optimization-variable-policy grounded_chain_fixed_root
)

first_shard=$((pair_index * 2))
second_shard=$((first_shard + 1))
first_offset=$((first_shard * 780))
second_offset=$((second_shard * 780))
first_limit=780
second_limit=780
if (( second_shard == 7 )); then
  second_limit=778
fi

env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  conda run --no-capture-output -n New_MMM python -u "${common[@]}" \
  --offset "$first_offset" --limit "$first_limit" \
  --output-dir "output/interx_shape_aug_full_batch8_query10475_seed20260828_shard${first_shard}" &
pid0=$!

env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  conda run --no-capture-output -n New_MMM python -u "${common[@]}" \
  --offset "$second_offset" --limit "$second_limit" \
  --output-dir "output/interx_shape_aug_full_batch8_query10475_seed20260828_shard${second_shard}" &
pid1=$!

status0=0
status1=0
wait "$pid0" || status0=$?
wait "$pid1" || status1=$?
if (( status0 != 0 || status1 != 0 )); then
  echo "full shard pair failed: shard${first_shard}=${status0}, shard${second_shard}=${status1}" >&2
  exit 1
fi
