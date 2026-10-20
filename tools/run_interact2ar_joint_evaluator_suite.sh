#!/usr/bin/env bash
set -euo pipefail

cd /mnt/8T-HDD/wenning/HHC

if (( $# < 1 || $# > 2 )) || [[ ! "$1" =~ ^[0-9]+$ ]]; then
  echo "usage: $0 GPU_INDEX [pre_shape|post_shape]" >&2
  exit 2
fi
gpu="$1"
stage="${2:-pre_shape}"
if [[ "${stage}" != pre_shape && "${stage}" != post_shape ]]; then
  echo "stage must be pre_shape or post_shape" >&2
  exit 2
fi
device="cuda:${gpu}"

conda() {
  /home/wenning/miniconda3/bin/conda "$@"
}

# This suite is intentionally separate from the legacy Inter-X main table.
# It first calibrates the untouched author README protocol (which uses a GT
# first frame), then scores the already frozen unconditional generations with
# the three released joint evaluators for exactly twenty paired replications.
required=(
  output/interact2ar_paper_repo_data_audit.json
  output/external_baseline_reproduction_audit.json
  output/fair_eval/hhc_historical_plus_paper_mask_ratio_no_contact_common50/summary.json
  output/fair_eval/intermask_official_exact_common50/summary.json
  output/fair_eval/interact2ar_official_fair_no_gt_first_frame/summary.json
  output/fair_eval/coshmdm_interx_retrained_2daycap_seed123/summary.json
  output/fair_eval/timotion_transformer_lpa_interx_retrained_fair/summary.json
)
for path in "${required[@]}"; do
  if [[ ! -s "${path}" ]]; then
    echo "Interact2AR evaluator prerequisite is incomplete: ${path}" >&2
    exit 2
  fi
done
if [[ "${stage}" == post_shape ]]; then
  for path in \
    output/fair_eval/hhc_historical_paper_mask_shapeaug_random_no_contact_common50/summary.json \
    output/fair_eval/hhc_historical_paper_mask_shapeaug_optimized_no_contact_common50/summary.json
  do
    if [[ ! -s "${path}" ]]; then
      echo "Post-shape Interact2AR evaluator prerequisite is incomplete: ${path}" >&2
      exit 2
    fi
  done
fi

used_mib="$(nvidia-smi --id="${gpu}" --query-gpu=memory.used \
  --format=csv,noheader,nounits | tr -d ' ')"
if [[ ! "${used_mib}" =~ ^[0-9]+$ ]] || (( used_mib > 1024 )); then
  echo "GPU ${gpu} is not free (memory.used=${used_mib:-unknown} MiB)." >&2
  exit 2
fi

conda run --no-capture-output -n New_MMM python -u \
  tools/audit_existing_intermask_native20.py
conda run --no-capture-output -n New_MMM python -u \
  tools/audit_intermask_common50_protocol_drift.py
conda run --no-capture-output -n New_MMM python -u \
  tools/audit_coshmdm_public_release_scope.py
conda run --no-capture-output -n New_MMM python -u \
  tools/audit_interact2ar_paper_repo_data.py
author_summary="output/interact2ar_author_release_calibration/summary.json"
author_audit="output/interact2ar_author_release_calibration/prerequisite_audit.json"
if [[ ! -s "${author_summary}" || ! -s "${author_audit}" ]] \
    || ! cmp -s output/interact2ar_paper_repo_data_audit.json "${author_audit}"; then
  bash tools/run_interact2ar_author_release_calibration.sh "${gpu}"
fi
conda run --no-capture-output -n New_MMM python -u \
  tools/audit_interact2ar_author_calibration.py
conda run --no-capture-output -n New_MMM python -u \
  tools/build_external_baseline_reproduction_audit.py

conda run --no-capture-output -n New_MMM python -u \
  tools/audit_interact2ar_joint_scorer_parity.py \
  --device "${device}" --batch-size 32

OMP_NUM_THREADS=8 conda run --no-capture-output -n New_MMM python -u \
  tools/replay_fair_interx_raw_ground_truth.py \
  --replications 20 --device cpu

suite_root="output/interact2ar_joint_evaluator_suite"
if [[ "${stage}" == pre_shape ]]; then
  conda run --no-capture-output -n New_MMM python -u \
    tools/score_interact2ar_joint_evaluators.py \
    --replications 20 --batch-size 32 --device "${device}" \
    --method hhc=output/fair_eval/hhc_historical_plus_paper_mask_ratio_no_contact_common50 \
    --method intermask=output/fair_eval/intermask_official_exact_common50 \
    --method interact2ar=output/fair_eval/interact2ar_official_fair_no_gt_first_frame \
    --method coshmdm=output/fair_eval/coshmdm_interx_retrained_2daycap_seed123 \
    --method timotion=output/fair_eval/timotion_transformer_lpa_interx_retrained_fair \
    --output-root "${suite_root}"

  conda run --no-capture-output -n New_MMM python -u \
    tools/aggregate_interact2ar_joint_evaluators.py \
    --input-root "${suite_root}" \
    --expected-methods hhc intermask interact2ar coshmdm timotion \
    --output "${suite_root}/pre_shape_summary.json"
else
  conda run --no-capture-output -n New_MMM python -u \
    tools/score_interact2ar_joint_evaluators.py \
    --replications 20 --batch-size 32 --device "${device}" \
    --method hhc_shape_random=output/fair_eval/hhc_historical_paper_mask_shapeaug_random_no_contact_common50 \
    --method hhc_shape_optimized=output/fair_eval/hhc_historical_paper_mask_shapeaug_optimized_no_contact_common50 \
    --output-root "${suite_root}"

  conda run --no-capture-output -n New_MMM python -u \
    tools/aggregate_interact2ar_joint_evaluators.py \
    --input-root "${suite_root}" \
    --expected-methods hhc intermask interact2ar coshmdm timotion \
      hhc_shape_random hhc_shape_optimized \
    --output "${suite_root}/summary.json"
fi

echo "Interact2AR ${stage} author-calibrated 20-rep joint-evaluator supplement complete."
