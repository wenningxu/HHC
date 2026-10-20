#!/usr/bin/env bash
set -euo pipefail

cd /mnt/8T-HDD/wenning/HHC

conda() {
  /home/wenning/miniconda3/bin/conda "$@"
}

# Deliberately a direct, fail-fast entrypoint: it never polls trainers and
# never starts a background terminal.  Run it once all six main rows exist.
required=(
  output/fair_eval/hhc_historical_plus_paper_mask_ratio_no_contact_common50/summary.json
  output/fair_eval/hhc_historical_plus_paper_mask_ratio_contact_common50/summary.json
  output/fair_eval/intermask_official_exact_common50/summary.json
  output/fair_eval/interact2ar_official_fair_no_gt_first_frame/summary.json
  output/fair_eval/coshmdm_interx_retrained_2daycap_seed123/summary.json
  output/fair_eval/timotion_transformer_lpa_interx_retrained_fair/summary.json
)
for path in "${required[@]}"; do
  if [[ ! -s "${path}" ]]; then
    echo "Current Inter-X paired comparison is not ready: ${path}" >&2
    exit 2
  fi
  if ! jq -e '
      .replications == 20
      and .samples_per_replication == 1696
      and ([
        .metrics.fid, .metrics.matching,
        .metrics.top1, .metrics.top2, .metrics.top3,
        .metrics.diversity, .metrics.multimodality
      ] | all(.values | length == 20))
    ' "${path}" >/dev/null; then
    echo "Incomplete 20-replication summary: ${path}" >&2
    exit 2
  fi
done

# Freeze every current main-table row under one evaluator runtime before any
# paired statistics are computed.  Generation can run on either GPU, but the
# semantic embeddings and aggregate metrics must all be recomputed on the same
# CPU official-evaluator implementation.
current_labels=(
  hhc_historical_plus_paper_mask_ratio_no_contact_common50
  hhc_historical_plus_paper_mask_ratio_contact_common50
  intermask_official_exact_common50
  interact2ar_official_fair_no_gt_first_frame
  coshmdm_interx_retrained_2daycap_seed123
  timotion_transformer_lpa_interx_retrained_fair
)
rescore_args=()
for label in "${current_labels[@]}"; do
  rescore_args+=(--rescore-label "${label}" --audit-label "${label}")
done
conda run --no-capture-output -n New_MMM python -u \
  tools/rescore_final_interx_ordinary.py \
  --device cpu --batch-size 32 "${rescore_args[@]}" \
  --output output/fair_eval/current_main_ordinary_only_metric_audit.json

conda run --no-capture-output -n New_MMM python -u \
  tools/compare_fair_interx_methods_paired.py \
  --reference \
    hhc_no_timing=output/fair_eval/hhc_historical_plus_paper_mask_ratio_no_contact_common50 \
  --comparators \
    intermask=output/fair_eval/intermask_official_exact_common50 \
    interact2ar=output/fair_eval/interact2ar_official_fair_no_gt_first_frame \
    coshmdm=output/fair_eval/coshmdm_interx_retrained_2daycap_seed123 \
    timotion=output/fair_eval/timotion_transformer_lpa_interx_retrained_fair \
  --output \
    output/fair_eval/paired_fair_method_comparison_historical_paper_mask.json

conda run --no-capture-output -n New_MMM python -u \
  tools/compare_fair_interx_methods_paired.py \
  --reference \
    hhc_oracle=output/fair_eval/hhc_historical_plus_paper_mask_ratio_contact_common50 \
  --comparators \
    hhc_no_timing=output/fair_eval/hhc_historical_plus_paper_mask_ratio_no_contact_common50 \
  --output \
    output/fair_eval/paired_oracle_timing_effect_historical_paper_mask.json

conda run --no-capture-output -n New_MMM python -u \
  tools/build_current_interx_main_table.py \
  --paired-main \
    output/fair_eval/paired_fair_method_comparison_historical_paper_mask.json \
  --paired-timing \
    output/fair_eval/paired_oracle_timing_effect_historical_paper_mask.json \
  --ordinary-only-audit \
    output/fair_eval/current_main_ordinary_only_metric_audit.json \
  --output-json output/current_interx_main_table.json \
  --output-markdown output/current_interx_main_table.md

echo 'Current Inter-X paired comparisons and main table complete.'
