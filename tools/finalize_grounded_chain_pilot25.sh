#!/usr/bin/env bash
set -euo pipefail

cd /mnt/8T-HDD/wenning/HHC

candidate="output/interx_shape_aug_groundfix_groundedchain_fixedroot_pilot25_metricfoot_contactmean_batch8_query10475_seed20260828"
candidate_physical="output/interx_shape_aug_groundfix_groundedchain_fixedroot_pilot25_body55floor_physical_seed20260828"
random_physical="output/interx_shape_aug_pilot_random_body55floor_physical_seed20260828"
comparison="output/interx_shape_aug_groundfix_groundedchain_fixedroot_pilot25_body55floor_physical_comparison_seed20260828.json"
assessment="output/interx_shape_aug_groundfix_groundedchain_fixedroot_pilot25_body55floor_assessment_seed20260828.json"

conda run --no-capture-output -n New_MMM python -u \
  tools/audit_grounded_chain_run_invariants.py \
  --run-dir "${candidate}" --expected-samples 25 \
  --output output/interx_grounded_chain_fixedroot_pilot25_invariants_audit.json

env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  conda run --no-capture-output -n New_MMM python -u \
  tools/score_volumetric_smpl_npz.py \
  --folder "${candidate}" --output-dir "${candidate_physical}" \
  --device cuda:1 --query-chunk 10475 --seed 1234

conda run --no-capture-output -n New_MMM python -u \
  tools/compare_paired_physical_metrics.py \
  --before "${random_physical}" --after "${candidate_physical}" \
  --output "${comparison}" --seed 20260828 --bootstrap-samples 10000

conda run --no-capture-output -n New_MMM python -u \
  tools/assess_shape_augmentation_pilot.py \
  --comparison "${comparison}" \
  --gate configs/interx_shape_augmentation/pilot_physical_gate_v1.json \
  --output "${assessment}"
