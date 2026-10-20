#!/usr/bin/env bash
set -euo pipefail

cd /mnt/8T-HDD/wenning/HHC

if [[ $# -ne 1 ]] || [[ ! "$1" =~ ^[0-9]+$ ]]; then
  echo "usage: $0 GPU_INDEX" >&2
  exit 2
fi
gpu="$1"
device="cuda:${gpu}"

conda() {
  /home/wenning/miniconda3/bin/conda "$@"
}

# Direct, fail-fast post-main entrypoint.  It never polls a trainer and must
# finish before the last-stage shape augmentation entrypoint is permitted.
required=(
  output/current_interx_main_table.json
  output/fair_eval/hhc_historical_plus_paper_mask_ratio_no_contact_common50/summary.json
  output/fair_eval/hhc_historical_plus_paper_mask_ratio_contact_common50/summary.json
  output/fair_eval/intermask_official_exact_common50/summary.json
  output/fair_eval/interact2ar_official_fair_no_gt_first_frame/summary.json
  output/fair_eval/coshmdm_interx_retrained_2daycap_seed123/summary.json
  output/fair_eval/timotion_transformer_lpa_interx_retrained_fair/summary.json
)
for path in "${required[@]}"; do
  if [[ ! -s "${path}" ]]; then
    echo "Current pre-shape review experiments are not ready: ${path}" >&2
    exit 2
  fi
done

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
  tools/build_external_baseline_reproduction_audit.py

requests="output/fair_interx_requests/replication_00.npz"

# Do not optimize HHC/GT motions or report an "optimized FID" here.  Contact
# optimization is a train-only random-shape augmentation study; its target is
# the manually timed shape-induced contact mismatch.  The former HHC/GT formal
# optimizer mixed that mechanism study with ordinary generation metrics and is
# explicitly invalidated.

conda run --no-capture-output -n New_MMM python -u \
  tools/run_final_interx_mesh_physical.py \
  --requests "${requests}" \
  --output-root output/current_interx_mesh_physical \
  --device "${device}" --query-chunk 10475 --seed 1234

conda run --no-capture-output -n New_MMM python -u \
  tools/compare_current_interx_mesh_methods_paired.py

qualitative_manifest="output/current_interx_qualitative/manifest.json"
if [[ ! -s "${qualitative_manifest}" ]]; then
  conda run --no-capture-output -n New_MMM python -u \
    tools/prepare_qualitative_comparison_selection.py \
    --requests "${requests}" \
    --output configs/qualitative_comparison_selection.json

  # The nine-case FK/render is inexpensive on CPU.  Pinning CPU prevents a
  # later run on another GPU from changing figure bytes through FK roundoff.
  conda run --no-capture-output -n New_MMM python -u \
    tools/render_fair_interx_qualitative.py \
    --selection configs/qualitative_comparison_selection.json \
    --fair-root output/fair_eval \
    --output-dir output/current_interx_qualitative \
    --device cpu
fi

timing_summary="output/fair_eval/hhc_historical_paper_mask_contact_timing_ablation_rep0/summary.json"
if [[ ! -s "${timing_summary}" ]]; then
  conda run --no-capture-output -n New_MMM python -u \
    tools/run_hhc_contact_timing_ablation.py --device "${device}"
  conda run --no-capture-output -n New_MMM python -u \
    tools/summarize_hhc_contact_timing_ablation.py
fi

pareto_root="output/fair_eval/hhc_historical_paper_mask_sampling_pareto_no_timing_rep0"
if [[ ! -s "${pareto_root}/summary.json" ]]; then
  conda run --no-capture-output -n New_MMM python -u \
    tools/run_hhc_sampling_pareto.py \
    --selection-manifest output/hhc_historical_plus_paper_mask_ratio_selection.json \
    --output-root "${pareto_root}" --device "${device}"
  conda run --no-capture-output -n New_MMM python -u \
    tools/summarize_hhc_sampling_pareto.py \
    --manifest "${pareto_root}/grid_manifest.json" \
    --output "${pareto_root}/summary.json"
fi

bash tools/run_interact2ar_joint_evaluator_suite.sh "${gpu}" pre_shape

echo "Current Inter-X pre-shape review experiments complete."
