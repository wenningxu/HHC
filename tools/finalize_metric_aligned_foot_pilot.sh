#!/usr/bin/env bash
set -euo pipefail

cd /mnt/8T-HDD/wenning/HHC

random="output/interx_shape_aug_pilot_random_seed20260828"
reference="output/interx_shape_aug_pilot_optimized_flathand_frozenfoot_contactmean_query10475_seed20260828"
candidate="output/interx_shape_aug_pilot_optimized_flathand_groundedhorizontal_contactmean_framebatch8_query10475_seed20260828"
random_physical="output/interx_shape_aug_pilot_random_flathand_physical_seed20260828"
reference_physical="output/interx_shape_aug_pilot_optimized_flathand_frozenfoot_contactmean_query10475_physical_seed20260828"
candidate_physical="output/interx_shape_aug_pilot_optimized_flathand_groundedhorizontal_contactmean_framebatch8_query10475_physical_seed20260828"
activation="output/interx_shape_aug_pilot_flathand_frozenfoot_contactmean_query10475_assessment_seed20260828.json"
random_comparison="output/interx_shape_aug_pilot_flathand_groundedhorizontal_contactmean_framebatch8_query10475_paired_physical_seed20260828.json"
reference_comparison="output/interx_shape_aug_pilot_contactmean_frozenfoot_vs_groundedhorizontal_framebatch8_query10475_paired_physical_seed20260828.json"
assessment="output/interx_shape_aug_pilot_flathand_groundedhorizontal_contactmean_framebatch8_query10475_assessment_seed20260828.json"

conda run --no-capture-output -n New_MMM python -u \
  tools/audit_metric_aligned_foot_preregistration.py

python -c "import json; from pathlib import Path; c=json.loads(Path('${candidate}/_completion_manifest.json').read_text()); a=json.loads(Path('${activation}').read_text()); f=a['metric_decisions']['mean_foot_sliding_m_per_frame']; assert c.get('complete') is True and c.get('completed_samples') == c.get('requested_samples') == 25; assert a.get('full_augmentation_authorized') is False and f.get('pass') is False and f['observed']['paired_delta_after_minus_before_mean'] > 0"

conda run --no-capture-output -n New_MMM python -u \
  tools/audit_metric_aligned_foot_single_factor.py \
  --reference "${reference}/_completion_manifest.json" \
  --candidate "${candidate}/_completion_manifest.json" \
  --activation-assessment "${activation}" \
  --output output/interx_shape_aug_pilot_metric_aligned_foot_single_factor_audit.json

conda run --no-capture-output -n New_MMM python -u \
  tools/analyze_shape_augmentation_runtime.py \
  --run-dir "${candidate}" \
  --plan configs/interx_shape_augmentation/timing_covered_seed123_v2/plan_manifest.json \
  --output "${candidate}/runtime_analysis.json" \
  --bootstrap-replicates 10000 --seed 20260901

conda run --no-capture-output -n New_MMM python -u \
  tools/audit_shape_augmentation_loss_balance.py \
  --run-dir "${candidate}" --require-complete \
  --output output/interx_shape_aug_pilot_metric_aligned_foot_loss_balance_audit.json

env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  conda run --no-capture-output -n New_MMM python -u \
  tools/score_volumetric_smpl_npz.py \
  --folder "${candidate}" --output-dir "${candidate_physical}" \
  --device cuda:1 --query-chunk 10475 --seed 1234

conda run --no-capture-output -n New_MMM python -u \
  tools/compare_paired_physical_metrics.py \
  --before "${random_physical}" --after "${candidate_physical}" \
  --output "${random_comparison}" --seed 1234 --bootstrap-samples 10000

conda run --no-capture-output -n New_MMM python -u \
  tools/compare_paired_physical_metrics.py \
  --before "${reference_physical}" --after "${candidate_physical}" \
  --output "${reference_comparison}" --seed 1234 --bootstrap-samples 10000

assessment_status=0
conda run --no-capture-output -n New_MMM python -u \
  tools/assess_shape_augmentation_pilot.py \
  --comparison "${random_comparison}" \
  --gate configs/interx_shape_augmentation/pilot_physical_gate_v1.json \
  --output "${assessment}" || assessment_status=$?

python -c "import json; from pathlib import Path; paths={'single_factor':Path('output/interx_shape_aug_pilot_metric_aligned_foot_single_factor_audit.json'),'loss_balance':Path('output/interx_shape_aug_pilot_metric_aligned_foot_loss_balance_audit.json'),'authorization':Path('${assessment}'),'random_to_candidate':Path('${random_comparison}'),'frozenfoot_to_candidate':Path('${reference_comparison}')}; print(json.dumps({k:json.loads(v.read_text()) for k,v in paths.items()},indent=2))"

exit "${assessment_status}"
