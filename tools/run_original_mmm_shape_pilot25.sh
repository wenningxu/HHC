#!/usr/bin/env bash
set -euo pipefail

cd /mnt/8T-HDD/wenning/HHC

if [[ $# -ne 1 ]] || [[ ! "$1" =~ ^[0-9]+$ ]]; then
  echo "usage: $0 GPU_INDEX" >&2
  exit 2
fi
gpu="$1"
device="cuda:${gpu}"

# Shape augmentation is deliberately the last Inter-X stage.  Fail fast on
# incomplete comparison evidence instead of turning this launcher into a
# polling queue or competing with an evaluator that still owns the GPU.
required=(
  output/current_interx_main_table.json
  output/current_interx_mesh_physical/manifest.json
  output/current_interx_mesh_physical/paired_vs_hhc_no_timing.json
  output/current_interx_qualitative/manifest.json
  output/interact2ar_joint_evaluator_suite/pre_shape_summary.json
  output/fair_eval/hhc_historical_paper_mask_contact_timing_ablation_rep0/summary.json
  output/fair_eval/hhc_historical_paper_mask_sampling_pareto_no_timing_rep0/summary.json
)
for path in "${required[@]}"; do
  if [[ ! -s "${path}" ]]; then
    echo "Original-MMM shape pilot25 prerequisite is incomplete: ${path}" >&2
    exit 2
  fi
done

used_mib="$(nvidia-smi --id="${gpu}" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')"
if [[ ! "${used_mib}" =~ ^[0-9]+$ ]] || (( used_mib > 1024 )); then
  echo "GPU ${gpu} is not free (memory.used=${used_mib:-unknown} MiB)." >&2
  exit 2
fi

conda_run=(/home/wenning/miniconda3/bin/conda run --no-capture-output -n New_MMM)
optimized="output/interx_shape_aug_original_mmm_pilot25_seed20260828"
random="output/interx_shape_aug_random_seed123_v2"
projected="output/interx_shape_aug_original_mmm_pilot25_seed20260828_constant_projection"
aligned="output/interx_shape_aug_original_mmm_pilot25_seed20260828_ground_diagnostic"
contacts="dataset/interx/contact_pairs_codex_luna_strict_v3_train"
records="configs/interx_shape_augmentation/timing_covered_seed123_v2/augmentation_records.jsonl"

"${conda_run[@]}" python -u tools/run_interx_shape_augmentation.py \
  --output-dir "${optimized}" --mode optimized --device "${device}" \
  --selection-seed 20260828 --limit 25 --positive-timing-only \
  --num-epochs 200 --learning-rate 0.01 --fine-stage-start 100 \
  --signed-distance-backend nearest_vertex \
  --foot-contact-mask-policy released_dynamic \
  --contact-loss-reduction-policy original_mmm \
  --optimization-variable-policy released_all_per_frame

"${conda_run[@]}" python -u tools/assess_original_mmm_shape_pilot.py \
  --random-dir "${random}" --optimized-dir "${optimized}" \
  --contact-dir "${contacts}" --projected-dir "${projected}" \
  --output "${optimized}_assessment.json" --device "${device}"

"${conda_run[@]}" python -u tools/diagnose_shape_ground_restore_mismatch.py \
  --random-dir "${random}" --optimized-dir "${optimized}" \
  --projected-dir "${projected}" --contact-dir "${contacts}" \
  --plan-records "${records}" --aligned-dir "${aligned}" \
  --output "${optimized}_ground_diagnostic.json" --device "${device}"

"${conda_run[@]}" python -u tools/diagnose_original_mmm_role_assignment.py \
  --aligned-dir "${aligned}" --optimized-dir "${optimized}" \
  --contact-dir "${contacts}" --plan-records "${records}" \
  --output "${optimized}_role_diagnostic.json" --device "${device}"

python tools/build_original_mmm_shape_pilot_gate.py \
  --assessment "${optimized}_assessment.json" \
  --alignment-diagnostic "${optimized}_ground_diagnostic.json" \
  --role-diagnostic "${optimized}_role_diagnostic.json" \
  --optimizer-completion "${optimized}/_completion_manifest.json" \
  --output "${optimized}_gate.json"

echo "Original-MMM shape pilot25 diagnostics complete; no full run was launched."
