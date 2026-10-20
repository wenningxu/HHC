#!/usr/bin/env bash
set -euo pipefail

cd /mnt/8T-HDD/wenning/HHC

# This coordinator is intentionally non-waiting.  Both independent method
# finalizers call it once; the one that observes all six completed summaries
# builds the paired table, then uses its just-released GPU for the review
# experiments that must precede shape augmentation.  A non-blocking lock makes
# simultaneous completion safe without adding another background waiter.
gpu="${1:-}"
if [[ -n "${gpu}" ]] && [[ ! "${gpu}" =~ ^[0-9]+$ ]]; then
  echo "usage: $0 [GPU_INDEX]" >&2
  exit 2
fi
lock_file="output/.current_interx_paired_finalize.lock"
exec 9>"${lock_file}"
if ! flock --nonblock 9; then
  echo "Current Inter-X paired finalization is already running."
  exit 0
fi

main_complete=false
if [[ -s output/current_interx_main_table.json ]] && jq -e '
    .schema == "hhc_current_interx_main_table_v1"
    and .status == "complete"
    and .protocol.replications == 20
    and .protocol.samples_per_replication == 1696
  ' output/current_interx_main_table.json >/dev/null; then
  main_complete=true
fi

if [[ "${main_complete}" != true ]]; then
  required=(
    output/fair_eval/hhc_historical_plus_paper_mask_ratio_no_contact_common50/summary.json
    output/fair_eval/hhc_historical_plus_paper_mask_ratio_contact_common50/summary.json
    output/fair_eval/intermask_official_exact_common50/summary.json
    output/fair_eval/interact2ar_official_fair_no_gt_first_frame/summary.json
    output/fair_eval/coshmdm_interx_retrained_2daycap_seed123/summary.json
    output/fair_eval/timotion_transformer_lpa_interx_retrained_fair/summary.json
  )
  missing=()
  for path in "${required[@]}"; do
    if [[ ! -s "${path}" ]]; then
      missing+=("${path}")
    fi
  done
  if (( ${#missing[@]} != 0 )); then
    printf 'Current Inter-X paired finalization is not ready; missing: %s\n' \
      "${missing[*]}"
    exit 0
  fi

  bash tools/finalize_current_interx_paired_results.sh
fi

if [[ -z "${gpu}" ]]; then
  echo "Current Inter-X paired main table is complete; no review GPU was requested."
  exit 0
fi

pre_shape_outputs=(
  output/current_interx_mesh_physical/manifest.json
  output/current_interx_mesh_physical/paired_vs_hhc_no_timing.json
  output/current_interx_qualitative/manifest.json
  output/interact2ar_joint_evaluator_suite/pre_shape_summary.json
  output/fair_eval/hhc_historical_paper_mask_contact_timing_ablation_rep0/summary.json
  output/fair_eval/hhc_historical_paper_mask_sampling_pareto_no_timing_rep0/summary.json
)
pre_shape_complete=true
for path in "${pre_shape_outputs[@]}"; do
  if [[ ! -s "${path}" ]]; then
    pre_shape_complete=false
  fi
done
if [[ "${pre_shape_complete}" == true ]]; then
  echo "Current Inter-X pre-shape review experiments are already materialized."
  exit 0
fi

exec bash tools/run_current_interx_review_experiments_before_shape.sh "${gpu}"
