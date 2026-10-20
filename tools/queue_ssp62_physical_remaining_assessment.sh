#!/usr/bin/env bash
# One-shot continuation of the frozen 2026-09-12 production run; no training.
set -euo pipefail
cd /mnt/8T-HDD/wenning/HHC
generator_pid=${1:?Pass the existing generator PID; never starts another generator}
[[ "$generator_pid" =~ ^[0-9]+$ ]]
exec 9>output/ssp62_physical_remaining_assessment_20260912.lock
flock -n 9
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
py=/home/wenning/miniconda3/envs/New_MMM/bin/python
production=output/ssp62_production_physical_remaining6138_20260912
floor=output/ssp62_production_physical_remaining6138_floor_20260912.json
report=output/ssp62_production_physical_remaining6138_assessment_20260912.json
gate=output/ssp62_production_physical_remaining6138_gate_20260912.json
echo "Waiting for existing generator PID $generator_pid; completion is checked independently."
while kill -0 "$generator_pid" 2>/dev/null; do sleep 60; done
"$py" - "$production" <<'PY'
import json, sys
from pathlib import Path
m = json.loads((Path(sys.argv[1]) / '_completion_manifest.json').read_text())
assert m['complete'] and m['requested_samples'] == m['completed_samples'] == 6138
assert len(set(m['completed_names'])) == 6138
assert m['selection']['full_selection_positions'] == list(range(100, 6238))
assert m['optimizer']['ground_alignment_output_coordinates'] == 'physical_pair_common_p1_origin'
print('Complete remaining 6138 verified; starting independent assessment.', flush=True)
PY
"$py" tools/diagnose_shape_physical_ground.py \
  --candidate-dir "$production" --limit 6138 --output "$floor"
"$py" tools/diagnose_shape_support_transfer.py \
  --candidate-dir "$production" --floor-report "$floor" --ground-dir "$production" \
  --device cuda:1 --penetration --volumetric \
  --contact-dir dataset/interx/contact_pairs_codex_luna_strict_v3_train --output "$report"
"$py" tools/build_physical_shape_quality_gate.py \
  --report "$report" --completion "$production/_completion_manifest.json" \
  --candidate-dir "$production" --floor-report "$floor" --output "$gate"
echo 'Remaining-shard gate complete; full 6238 aggregation and acceptance still required. No training started.'
