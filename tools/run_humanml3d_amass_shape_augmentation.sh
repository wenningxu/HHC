#!/usr/bin/env bash
set -euo pipefail

cd /mnt/8T-HDD/wenning/HHC

if (( $# < 1 || $# > 2 )) || [[ ! "$1" =~ ^[0-9]+$ ]] \
    || { (( $# == 2 )) && [[ ! "$2" =~ ^[0-9]+$ ]]; }; then
  echo "usage: $0 GPU_INDEX [SECOND_GPU_INDEX]" >&2
  exit 2
fi
gpu0="$1"
gpu1="${2:-}"
if [[ -n "${gpu1}" && "${gpu0}" == "${gpu1}" ]]; then
  echo "GPU indices must be distinct." >&2
  exit 2
fi
for gpu in "${gpu0}" ${gpu1:+"${gpu1}"}; do
  used_mib="$(nvidia-smi --id="${gpu}" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')"
  if [[ ! "${used_mib}" =~ ^[0-9]+$ ]] || (( used_mib > 1024 )); then
    echo "GPU ${gpu} is not free (memory.used=${used_mib:-unknown} MiB)." >&2
    exit 2
  fi
done

conda_run=(/home/wenning/miniconda3/bin/conda run --no-capture-output -n New_MMM)
python_env=/home/wenning/miniconda3/envs/New_MMM/bin/python
plan="configs/interx_shape_augmentation/humanml3d_train_amass_seed123_v1/plan_manifest.json"
contacts="dataset/interx/contact_pairs_codex_luna_strict_v3_train"
selection_seed=20260828
positive_count=5868
zero_count=370

# Licensed assets are never downloaded here. This stage either resumes the
# official transfer pipeline or fails before any optimization work.
"${conda_run[@]}" python -u tools/run_humanml3d_amass_shape_donor_pipeline.py --resume
"${python_env}" - "${plan}" <<'PY'
import json
import sys
p = json.load(open(sys.argv[1]))
assert p.get('status') == 'complete'
assert p.get('donor_pool_kind') == 'humanml3d_train_amass_transferred_smplx'
assert p.get('eligible_samples') == 6238
assert p.get('positive_timing_samples') == 5868
assert p['inputs']['external_donor_manifest'] is not None
PY

query_gate="output/volumetric_smpl_current_query_chunk_assessment.json"
frame_gate="output/volumetric_smpl_current_frame_batch_assessment.json"
test -s "${query_gate}"
test -s "${frame_gate}"
query_size="$("${python_env}" -c "import json; d=json.load(open('${query_gate}')); assert d['status']=='pass'; print(d['query_chunk_authorized'])")"
frame_size="$("${python_env}" -c "import json; d=json.load(open('${frame_gate}')); assert d['status']=='pass' and d['production_edit_authorized'] is True; print(d['selected_candidate']['frame_batch_size'])")"

common=(
  --plan "${plan}" --contact-dir "${contacts}"
  --selection-seed "${selection_seed}" --seed 1234
  --num-epochs 120 --learning-rate 0.01 --fine-stage-start 100
  --foot-contact-mask-policy released_dynamic
  --contact-loss-reduction-policy original_mmm
  --optimization-variable-policy released_all_per_frame
)
volume=(
  --signed-distance-backend volumetric_smpl
  --query-chunk "${query_size}"
  --volumetric-smpl-frame-batch-size "${frame_size}"
  --volumetric-smpl-penetration-vertex-count 256
  --volumetric-smpl-penetration-frame-count 16
  --volumetric-smpl-body-encoding-samples 64
)

materialize_overlay() {
  local expected_mode="$1"
  local output_root="$2"
  shift 2
  local manifest="${output_root}/materialization_manifest.json"
  if [[ -s "${manifest}" ]]; then
    "${python_env}" - "${output_root}" "${manifest}" "${expected_mode}" <<'PY'
import sys
from utils.interx_shape_augmentation import (
    validate_materialized_interx_shape_augmentation,
)
d = validate_materialized_interx_shape_augmentation(
    sys.argv[1], sys.argv[2], verify_augmented_motion_content=False,
)
assert d.get('augmentation_mode') == sys.argv[3]
assert d['counts']['augmented'] == 6238
assert d['counts']['official_train'] == 9110
assert d['counts']['combined_train'] == 15348
assert d['normalization']['copy_is_byte_identical'] is True
PY
    return
  fi
  local command=(python -u tools/materialize_interx_shape_augmentation.py)
  local optimization_dir
  for optimization_dir in "$@"; do
    command+=(--optimization-dir "${optimization_dir}")
  done
  command+=(--plan "${plan}" --output-root "${output_root}")
  "${conda_run[@]}" "${command[@]}"
  test -s "${manifest}"
  "${python_env}" - "${manifest}" "${expected_mode}" <<'PY'
import json
import sys
d = json.load(open(sys.argv[1]))
assert d.get('status') == 'complete'
assert d.get('augmentation_mode') == sys.argv[2]
assert d['counts']['augmented'] == 6238
assert d['counts']['official_train'] == 9110
assert d['counts']['combined_train'] == 15348
assert d['normalization']['copy_is_byte_identical'] is True
PY
}

validated_gate_status() {
  local gate="$1"
  [[ -s "${gate}" ]] || return 1
  "${python_env}" - "${gate}" <<'PY'
import hashlib
import json
import pathlib
import sys

def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()

d = json.load(open(sys.argv[1]))
assert d.get('schema') == 'hhc_fast_selective_shape_pilot_gate_v2'
assert d.get('status') in {'pass', 'fail'}
artifacts = d.get('artifacts', {})
assert 'gate_implementation' in artifacts
for row in artifacts.values():
    if row is None:
        continue
    path = pathlib.Path(row['path']).resolve(strict=True)
    assert path.stat().st_size == row['bytes']
    assert digest(path) == row['sha256']
implementation = pathlib.Path('tools/assess_fast_selective_shape_pilot.py').resolve()
assert pathlib.Path(artifacts['gate_implementation']['path']).resolve() == implementation
checks = d.get('checks', {})
assert checks
assert (d['status'] == 'pass') == all(checks.values())
print(d['status'])
PY
}

# Validate all plan, caption, motion, manual timing and LLM-region bindings.
"${conda_run[@]}" python -u tools/run_interx_shape_augmentation.py \
  "${common[@]}" "${volume[@]}" --device "cuda:${gpu0}" \
  --output-dir output/interx_shape_aug_amass_preflight --mode optimized \
  --preflight-only

# Same-donor/same-sample pilot. VolumetricSMPL is promoted only if its
# conservative projected selection is physically safe, stays within one day,
# and is strictly faster than nearest-vertex on these exact 25 samples.
pilot_random="output/interx_shape_aug_amass_pilot25_positive_random"
pilot_nearest="output/interx_shape_aug_amass_pilot25_nearest"
pilot_nearest_projected="output/interx_shape_aug_amass_pilot25_nearest_projected"
pilot_nearest_assessment="output/interx_shape_aug_amass_pilot25_nearest_assessment.json"
pilot_nearest_gate="output/interx_shape_aug_amass_pilot25_nearest_gate.json"
pilot_volume="output/interx_shape_aug_amass_pilot25_volumetric"
pilot_projected="output/interx_shape_aug_amass_pilot25_volumetric_projected"
pilot_assessment="output/interx_shape_aug_amass_pilot25_volumetric_assessment.json"
pilot_gate="output/interx_shape_aug_amass_pilot25_volumetric_gate.json"

selected_backend="volumetric_smpl"
selected_backend_args=("${volume[@]}")
pilot_status="$(validated_gate_status "${pilot_gate}" || true)"
if [[ "${pilot_status}" != "pass" && "${pilot_status}" != "fail" ]]; then
  "${conda_run[@]}" python -u tools/run_interx_shape_augmentation.py \
    "${common[@]}" --device "cuda:${gpu0}" \
    --output-dir "${pilot_random}" --mode random_shape_only \
    --positive-timing-only --limit 25
  "${conda_run[@]}" python -u tools/run_interx_shape_augmentation.py \
    "${common[@]}" --device "cuda:${gpu0}" \
    --output-dir "${pilot_nearest}" --mode optimized \
    --signed-distance-backend nearest_vertex --positive-timing-only --limit 25
  "${conda_run[@]}" python -u tools/run_interx_shape_augmentation.py \
    "${common[@]}" "${volume[@]}" --device "cuda:${gpu0}" \
    --output-dir "${pilot_volume}" --mode optimized \
    --positive-timing-only --limit 25
  "${conda_run[@]}" python -u tools/assess_original_mmm_shape_pilot.py \
    --random-dir "${pilot_random}" --optimized-dir "${pilot_volume}" \
    --contact-dir "${contacts}" --projected-dir "${pilot_projected}" \
    --output "${pilot_assessment}" --device "cuda:${gpu0}"
  if "${conda_run[@]}" python -u tools/assess_fast_selective_shape_pilot.py \
      --assessment "${pilot_assessment}" \
      --completion "${pilot_volume}/_completion_manifest.json" \
      --nearest-completion "${pilot_nearest}/_completion_manifest.json" \
      --output "${pilot_gate}" --maximum-two-gpu-hours 24; then
    pilot_status=pass
  else
    pilot_status="$(validated_gate_status "${pilot_gate}")"
  fi
fi
if [[ "${pilot_status}" == "fail" ]]; then
  # A numerical or speed gate failure falls back to the current original-MMM
  # nearest-vertex path. A hash-valid existing fallback gate is reused.
  nearest_status="$(validated_gate_status "${pilot_nearest_gate}" || true)"
  if [[ "${nearest_status}" != "pass" ]]; then
    "${conda_run[@]}" python -u tools/run_interx_shape_augmentation.py \
      "${common[@]}" --device "cuda:${gpu0}" \
      --output-dir "${pilot_random}" --mode random_shape_only \
      --positive-timing-only --limit 25
    "${conda_run[@]}" python -u tools/run_interx_shape_augmentation.py \
      "${common[@]}" --device "cuda:${gpu0}" \
      --output-dir "${pilot_nearest}" --mode optimized \
      --signed-distance-backend nearest_vertex --positive-timing-only --limit 25
    "${conda_run[@]}" python -u tools/assess_original_mmm_shape_pilot.py \
      --random-dir "${pilot_random}" --optimized-dir "${pilot_nearest}" \
      --contact-dir "${contacts}" --projected-dir "${pilot_nearest_projected}" \
      --output "${pilot_nearest_assessment}" --device "cuda:${gpu0}"
    "${conda_run[@]}" python -u tools/assess_fast_selective_shape_pilot.py \
      --assessment "${pilot_nearest_assessment}" \
      --completion "${pilot_nearest}/_completion_manifest.json" \
      --output "${pilot_nearest_gate}" --maximum-two-gpu-hours 24
    nearest_status="$(validated_gate_status "${pilot_nearest_gate}")"
  fi
  [[ "${nearest_status}" == "pass" ]]
  selected_backend="nearest_vertex"
  selected_backend_args=(--signed-distance-backend nearest_vertex)
fi

# Full random sources are immutable references for conservative selection.
# Zero-timing samples never enter the optimizer and occupy global positions
# 5868..6237 so all three materialization components form 0..6237 exactly.
full_random_positive="output/interx_shape_aug_amass_full_positive_random"
full_random_zero="output/interx_shape_aug_amass_full_zero_random"
"${conda_run[@]}" python -u tools/run_interx_shape_augmentation.py \
  "${common[@]}" --device "cuda:${gpu0}" \
  --output-dir "${full_random_positive}" --mode random_shape_only \
  --positive-timing-only
"${conda_run[@]}" python -u tools/run_interx_shape_augmentation.py \
  "${common[@]}" --device "cuda:${gpu0}" \
  --output-dir "${full_random_zero}" --mode random_shape_only \
  --zero-timing-only --selection-position-base "${positive_count}"

# Materialize the strict random-shape-only training control from the exact
# same donor assignments, crops and 0..6237 selection positions. This arm is
# required to separate beta-distribution diversity from optimization effects.
random_overlay="dataset/interx_shape_aug_amass_random_only_seed123_v1"
materialize_overlay random_shape_only "${random_overlay}" \
  "${full_random_positive}" "${full_random_zero}"

run_optimizer_shard() {
  local gpu="$1"
  local shard="$2"
  local offset=$((shard * 2934))
  "${conda_run[@]}" python -u tools/run_interx_shape_augmentation.py \
    "${common[@]}" "${selected_backend_args[@]}" --device "cuda:${gpu}" \
    --output-dir "output/interx_shape_aug_amass_full_${selected_backend}_shard${shard}" \
    --mode optimized --positive-timing-only --offset "${offset}" --limit 2934
}

if [[ -n "${gpu1}" ]]; then
  run_optimizer_shard "${gpu0}" 0 & pid0=$!
  run_optimizer_shard "${gpu1}" 1 & pid1=$!
  status0=0; status1=0
  wait "${pid0}" || status0=$?
  wait "${pid1}" || status1=$?
  if (( status0 != 0 || status1 != 0 )); then
    echo "AMASS ${selected_backend} shards failed: shard0=${status0}, shard1=${status1}" >&2
    exit 1
  fi
else
  run_optimizer_shard "${gpu0}" 0
  run_optimizer_shard "${gpu0}" 1
fi

# Per-sample selection retains the projected optimization only when mismatch
# improves, penetration does not increase and absolute skate delta <=2e-6.
# Otherwise the output is byte-for-byte copied from its random input.
selected_dirs=()
for shard in 0 1; do
  selected="output/interx_shape_aug_amass_full_selected_shard${shard}"
  "${conda_run[@]}" python -u tools/select_interx_shape_optimization.py \
    --optimized-dir "output/interx_shape_aug_amass_full_${selected_backend}_shard${shard}" \
    --random-dir "${full_random_positive}" --contact-dir "${contacts}" \
    --output-dir "${selected}" --device "cuda:${gpu0}" \
    --skate-tolerance 0.000002
  selected_dirs+=("${selected}")
done

overlay="dataset/interx_shape_aug_amass_optimized_selective_seed123_v1"
materialize_overlay optimized_positive_plus_random_zero "${overlay}" \
  "${selected_dirs[0]}" "${selected_dirs[1]}" "${full_random_zero}"

# These are normal sequential experiment stages, not waiters. Both arms use
# the same code, seed and telemetry configuration; only their immutable
# augmentation overlay differs.
bash tools/run_hhc_no_shape_amass_aug_pilot.sh "${gpu0}" optimized_selective
bash tools/run_hhc_no_shape_amass_aug_pilot.sh "${gpu0}" random_only
