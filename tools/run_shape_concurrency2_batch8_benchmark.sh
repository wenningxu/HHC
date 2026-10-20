#!/usr/bin/env bash
set -euo pipefail

cd /mnt/8T-HDD/wenning/HHC

common=(
  tools/run_interx_shape_augmentation.py
  --plan configs/interx_shape_augmentation/timing_covered_seed123_v2/plan_manifest.json
  --contact-dir dataset/interx/contact_pairs_codex_luna_strict_v3_train
  --mode optimized --device cuda:1 --query-chunk 10475
  --volumetric-smpl-frame-batch-size 8 --seed 1234
  --selection-seed 20260828 --positive-timing-only --limit 1
  --num-epochs 100 --fine-stage-start 50
  --foot-contact-mask-policy initial_or_current_grounded_horizontal_mean_speed
  --contact-loss-reduction-policy active_contact_mean
  --optimization-variable-policy grounded_chain_fixed_root
)

env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  conda run --no-capture-output -n New_MMM python -u "${common[@]}" \
  --offset 0 \
  --output-dir output/interx_shape_aug_concurrency2_framebatch8_query10475_seedfix_shard0 &
pid0=$!

env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  conda run --no-capture-output -n New_MMM python -u "${common[@]}" \
  --offset 1 \
  --output-dir output/interx_shape_aug_concurrency2_framebatch8_query10475_seedfix_shard1 &
pid1=$!

status0=0
status1=0
wait "$pid0" || status0=$?
wait "$pid1" || status1=$?
if (( status0 != 0 || status1 != 0 )); then
  echo "concurrency benchmark failed: shard0=${status0}, shard1=${status1}" >&2
  exit 1
fi
