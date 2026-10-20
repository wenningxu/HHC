#!/usr/bin/env bash
set -euo pipefail

cd /mnt/8T-HDD/wenning/HHC

if [[ -n "${PYTORCH_CUDA_ALLOC_CONF:-}" ]] \
    && [[ "${PYTORCH_CUDA_ALLOC_CONF}" != "expandable_segments:True" ]]; then
  echo "Unexpected PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF}"
  exit 2
fi
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

conda() {
  /home/wenning/miniconda3/bin/conda "$@"
}

plan="configs/interx_shape_augmentation/timing_covered_seed123_v2/plan_manifest.json"
contact_dir="dataset/interx/contact_pairs_codex_luna_strict_v3_train"
random_dir="output/interx_shape_aug_pilot_random_seed20260828"
optimized_dir="output/interx_shape_aug_pilot_optimized_flathand_query10475_seed20260828"
random_physical="output/interx_shape_aug_pilot_random_flathand_physical_seed20260828"
optimized_physical="output/interx_shape_aug_pilot_optimized_flathand_query10475_physical_seed20260828"
comparison="output/interx_shape_aug_pilot_flathand_query10475_paired_physical_seed20260828.json"
assessment="output/interx_shape_aug_pilot_flathand_query10475_assessment_seed20260828.json"

conda run --no-capture-output -n New_MMM python -u \
  tools/require_interx_volumetric_smpl_contract.py \
  --query-chunk 10475 \
  --output output/interx_volumetric_smpl_contract_gate.json

# Random-only output is independent of optimizer code. Its dedicated audit
# reconstructs every output from the frozen crop and donor shape, so later
# optimizer-only source changes do not force a fake recomputation.
conda run --no-capture-output -n New_MMM python -u \
  tools/audit_random_shape_augmentation_completion.py \
  --run-dir "${random_dir}" --plan "${plan}" --expected-samples 25 \
  --output "${random_dir}/completion_content_audit.json"
# The optimized arm still requires a strict current-source contract replay.
conda run --no-capture-output -n New_MMM python -u \
  tools/run_interx_shape_augmentation.py \
  --plan "${plan}" --contact-dir "${contact_dir}" \
  --output-dir "${optimized_dir}" --mode optimized \
  --device cuda:1 --query-chunk 10475 --seed 1234 \
  --selection-seed 20260828 --positive-timing-only --limit 25

conda run --no-capture-output -n New_MMM python -u \
  tools/analyze_shape_augmentation_runtime.py \
  --run-dir "${optimized_dir}" --plan "${plan}" \
  --output "${optimized_dir}/runtime_analysis.json" \
  --bootstrap-replicates 10000 --seed 20260901

conda run --no-capture-output -n New_MMM python -u \
  tools/score_volumetric_smpl_npz.py \
  --folder "${random_dir}" --output-dir "${random_physical}" \
  --device cuda:1 --query-chunk 10475 --seed 1234
conda run --no-capture-output -n New_MMM python -u \
  tools/score_volumetric_smpl_npz.py \
  --folder "${optimized_dir}" --output-dir "${optimized_physical}" \
  --device cuda:1 --query-chunk 10475 --seed 1234

conda run --no-capture-output -n New_MMM python -u \
  tools/compare_paired_physical_metrics.py \
  --before "${random_physical}" --after "${optimized_physical}" \
  --output "${comparison}" --seed 1234 --bootstrap-samples 10000

conda run --no-capture-output -n New_MMM python -u \
  tools/assess_shape_augmentation_pilot.py \
  --comparison "${comparison}" \
  --gate configs/interx_shape_augmentation/pilot_physical_gate_v1.json \
  --output "${assessment}"
