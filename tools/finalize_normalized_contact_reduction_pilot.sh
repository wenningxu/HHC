#!/usr/bin/env bash
set -euo pipefail

cd /mnt/8T-HDD/wenning/HHC

random="output/interx_shape_aug_pilot_random_seed20260828"
reference="output/interx_shape_aug_pilot_optimized_flathand_frozenfoot_query10475_seed20260828"
candidate="output/interx_shape_aug_pilot_optimized_flathand_frozenfoot_contactmean_query10475_seed20260828"
random_physical="output/interx_shape_aug_pilot_random_flathand_physical_seed20260828"
reference_physical="output/interx_shape_aug_pilot_optimized_flathand_frozenfoot_query10475_physical_seed20260828"
candidate_physical="output/interx_shape_aug_pilot_optimized_flathand_frozenfoot_contactmean_query10475_physical_seed20260828"
random_comparison="output/interx_shape_aug_pilot_flathand_frozenfoot_contactmean_query10475_paired_physical_seed20260828.json"
reference_comparison="output/interx_shape_aug_pilot_frozenfoot_rawsum_vs_contactmean_query10475_paired_physical_seed20260828.json"
activation="output/interx_shape_aug_pilot_flathand_frozenfoot_query10475_assessment_seed20260828.json"
assessment="output/interx_shape_aug_pilot_flathand_frozenfoot_contactmean_query10475_assessment_seed20260828.json"

python - <<'PY'
import json
from pathlib import Path

candidate = json.loads(Path(
    'output/interx_shape_aug_pilot_optimized_flathand_frozenfoot_'
    'contactmean_query10475_seed20260828/_completion_manifest.json'
).read_text(encoding='utf-8'))
activation = json.loads(Path(
    'output/interx_shape_aug_pilot_flathand_frozenfoot_'
    'query10475_assessment_seed20260828.json'
).read_text(encoding='utf-8'))
if not (
    candidate.get('complete') is True
    and candidate.get('completed_samples')
    == candidate.get('requested_samples') == 25
):
    raise SystemExit('normalized-contact 25-pair pilot is not complete')
if activation.get('full_augmentation_authorized') is not False:
    raise SystemExit('conditional normalized-contact pilot was not activated')
PY

conda run --no-capture-output -n New_MMM python -u \
  tools/audit_contact_reduction_single_factor.py \
  --reference "${reference}/_completion_manifest.json" \
  --candidate "${candidate}/_completion_manifest.json" \
  --activation-assessment "${activation}" \
  --output output/interx_shape_aug_pilot_contactmean_single_factor_audit.json

conda run --no-capture-output -n New_MMM python -u \
  tools/analyze_shape_augmentation_runtime.py \
  --run-dir "${candidate}" \
  --plan configs/interx_shape_augmentation/timing_covered_seed123_v2/plan_manifest.json \
  --output "${candidate}/runtime_analysis.json" \
  --bootstrap-replicates 10000 --seed 20260901

conda run --no-capture-output -n New_MMM python -u \
  tools/audit_shape_augmentation_loss_balance.py \
  --run-dir "${candidate}" --require-complete \
  --output output/interx_shape_aug_pilot_contactmean_loss_balance_audit.json

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

python - <<'PY'
import json
from pathlib import Path

paths = {
    'single_factor': Path('output/interx_shape_aug_pilot_contactmean_single_factor_audit.json'),
    'loss_balance': Path('output/interx_shape_aug_pilot_contactmean_loss_balance_audit.json'),
    'authorization': Path('output/interx_shape_aug_pilot_flathand_frozenfoot_contactmean_query10475_assessment_seed20260828.json'),
    'random_to_candidate': Path('output/interx_shape_aug_pilot_flathand_frozenfoot_contactmean_query10475_paired_physical_seed20260828.json'),
    'rawsum_to_candidate': Path('output/interx_shape_aug_pilot_frozenfoot_rawsum_vs_contactmean_query10475_paired_physical_seed20260828.json'),
}
print(json.dumps({
    key: json.loads(path.read_text(encoding='utf-8'))
    for key, path in paths.items()
}, indent=2))
PY

exit "${assessment_status}"
