#!/usr/bin/env bash
set -euo pipefail

cd /mnt/8T-HDD/wenning/HHC

pilot5_assessment="output/interx_shape_aug_groundfix_groundedchain_fixedroot_pilot5_assessment_seed20260828.json"

python -c "import json; a=json.load(open('${pilot5_assessment}')); assert a.get('status') == 'pass' and a.get('authorized_action') == 'pilot25' and a.get('pilot25_authorized') is True and a.get('full_augmentation_authorized') is False"

env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  conda run --no-capture-output -n New_MMM python -u \
  tools/run_interx_shape_augmentation.py \
  --plan configs/interx_shape_augmentation/timing_covered_seed123_v2/plan_manifest.json \
  --contact-dir dataset/interx/contact_pairs_codex_luna_strict_v3_train \
  --output-dir output/interx_shape_aug_groundfix_groundedchain_fixedroot_pilot25_metricfoot_contactmean_batch8_query10475_seed20260828 \
  --mode optimized --device cuda:1 --query-chunk 10475 \
  --volumetric-smpl-frame-batch-size 8 --seed 1234 \
  --selection-seed 20260828 --positive-timing-only --limit 25 \
  --num-epochs 100 --fine-stage-start 50 \
  --foot-contact-mask-policy initial_or_current_grounded_horizontal_mean_speed \
  --contact-loss-reduction-policy active_contact_mean \
  --optimization-variable-policy grounded_chain_fixed_root
