#!/usr/bin/env bash
set -euo pipefail

# The original report watcher may already have read the pre-patch shell file.
# Wait for it, then make the paired analysis/report refresh idempotent.
while ps -p 642995 >/dev/null 2>&1; do
  sleep 30
done

if [ ! -s output/fair_eval/paired_fair_method_comparison.json ]; then
  conda run --no-capture-output -n New_MMM python -u \
    tools/compare_fair_interx_methods_paired.py \
    --reference hhc_no_timing=output/fair_eval/hhc_formal_raw12288_repaired_mgt_common50_no_contact \
    --comparators \
      intermask=output/fair_eval/intermask_official_exact_common50 \
      interact2ar=output/fair_eval/interact2ar_official_fair_no_gt_first_frame \
      coshmdm=output/fair_eval/coshmdm_interx_retrained_2daycap_seed123 \
      timotion=output/fair_eval/timotion_transformer_lpa_interx_retrained_fair \
    --output output/fair_eval/paired_fair_method_comparison.json
fi

if [ ! -s output/fair_eval/paired_oracle_timing_effect.json ]; then
  conda run --no-capture-output -n New_MMM python -u \
    tools/compare_fair_interx_methods_paired.py \
    --reference hhc_oracle=output/fair_eval/hhc_formal_raw12288_repaired_mgt_common50_contact \
    --comparators \
      hhc_no_timing=output/fair_eval/hhc_formal_raw12288_repaired_mgt_common50_no_contact \
    --output output/fair_eval/paired_oracle_timing_effect.json
fi

conda run --no-capture-output -n New_MMM python -u \
  tools/build_review_results_report.py
