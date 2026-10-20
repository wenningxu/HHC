#!/usr/bin/env bash
set -euo pipefail

cd /mnt/8T-HDD/wenning/HHC

released="output/interx_shape_aug_pilot_optimized_flathand_query10475_seed20260828"
random="output/interx_shape_aug_pilot_random_seed20260828"
frozen="output/interx_shape_aug_pilot_optimized_flathand_frozenfoot_query10475_seed20260828"
random_physical="output/interx_shape_aug_pilot_random_flathand_physical_seed20260828"
released_physical="output/interx_shape_aug_pilot_optimized_flathand_query10475_physical_seed20260828"
frozen_physical="output/interx_shape_aug_pilot_optimized_flathand_frozenfoot_query10475_physical_seed20260828"
random_comparison="output/interx_shape_aug_pilot_flathand_frozenfoot_query10475_paired_physical_seed20260828.json"
released_comparison="output/interx_shape_aug_pilot_released_vs_frozenfoot_query10475_paired_physical_seed20260828.json"
assessment="output/interx_shape_aug_pilot_flathand_frozenfoot_query10475_assessment_seed20260828.json"

python - <<'PY'
import json
from pathlib import Path

path = Path(
    'output/interx_shape_aug_pilot_optimized_flathand_'
    'frozenfoot_query10475_seed20260828/_completion_manifest.json'
)
payload = json.loads(path.read_text(encoding='utf-8'))
if not (
    payload.get('complete') is True
    and payload.get('completed_samples') == payload.get('requested_samples') == 25
):
    raise SystemExit('frozen-foot 25-pair pilot is not complete')
PY

conda run --no-capture-output -n New_MMM python -u \
  tools/audit_frozen_foot_single_factor.py \
  --released "${released}/_completion_manifest.json" \
  --frozen "${frozen}/_completion_manifest.json" \
  --output output/interx_shape_aug_pilot_frozenfoot_single_factor_audit.json

conda run --no-capture-output -n New_MMM python -u \
  tools/analyze_shape_augmentation_runtime.py \
  --run-dir "${frozen}" \
  --plan configs/interx_shape_augmentation/timing_covered_seed123_v2/plan_manifest.json \
  --output "${frozen}/runtime_analysis.json" \
  --bootstrap-replicates 10000 --seed 20260901

conda run --no-capture-output -n New_MMM python -u \
  tools/audit_shape_augmentation_loss_balance.py \
  --run-dir "${frozen}" --require-complete \
  --output output/interx_shape_aug_pilot_frozenfoot_loss_balance_audit.json

env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  conda run --no-capture-output -n New_MMM python -u \
  tools/score_volumetric_smpl_npz.py \
  --folder "${frozen}" --output-dir "${frozen_physical}" \
  --device cuda:1 --query-chunk 10475 --seed 1234

conda run --no-capture-output -n New_MMM python -u \
  tools/compare_paired_physical_metrics.py \
  --before "${random_physical}" --after "${frozen_physical}" \
  --output "${random_comparison}" --seed 1234 --bootstrap-samples 10000

conda run --no-capture-output -n New_MMM python -u \
  tools/compare_paired_physical_metrics.py \
  --before "${released_physical}" --after "${frozen_physical}" \
  --output "${released_comparison}" --seed 1234 --bootstrap-samples 10000

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
    'single_factor': Path('output/interx_shape_aug_pilot_frozenfoot_single_factor_audit.json'),
    'loss_balance': Path('output/interx_shape_aug_pilot_frozenfoot_loss_balance_audit.json'),
    'authorization': Path('output/interx_shape_aug_pilot_flathand_frozenfoot_query10475_assessment_seed20260828.json'),
    'random_to_frozen': Path('output/interx_shape_aug_pilot_flathand_frozenfoot_query10475_paired_physical_seed20260828.json'),
    'released_to_frozen': Path('output/interx_shape_aug_pilot_released_vs_frozenfoot_query10475_paired_physical_seed20260828.json'),
}
print(json.dumps({key: json.loads(path.read_text(encoding='utf-8')) for key, path in paths.items()}, indent=2))
PY

exit "${assessment_status}"
