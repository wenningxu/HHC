#!/usr/bin/env bash
set -euo pipefail

cd /mnt/8T-HDD/wenning/HHC

conda() {
  /home/wenning/miniconda3/bin/conda "$@"
}

required=(
  output/fair_eval/hhc_historical_plus_paper_mask_ratio_no_contact_common50/summary.json
  output/fair_eval/hhc_historical_plus_paper_mask_ratio_contact_common50/summary.json
  output/fair_eval/hhc_historical_paper_mask_shapeaug_random_no_contact_common50/summary.json
  output/fair_eval/hhc_historical_paper_mask_shapeaug_random_contact_common50/summary.json
  output/fair_eval/hhc_historical_paper_mask_shapeaug_optimized_no_contact_common50/summary.json
  output/fair_eval/hhc_historical_paper_mask_shapeaug_optimized_contact_common50/summary.json
  output/interact2ar_paper_repo_data_audit.json
  output/interact2ar_author_release_calibration/audit.json
  output/interact2ar_joint_scorer_parity_audit.json
  output/interact2ar_joint_evaluator_suite/pre_shape_summary.json
  output/interact2ar_joint_evaluator_suite/summary.json
)
for path in "${required[@]}"; do
  if [[ ! -s "${path}" ]]; then
    echo "Current shape comparison is incomplete: ${path}" >&2
    exit 2
  fi
done

conda run --no-capture-output -n New_MMM python -u \
  tools/compare_fair_interx_methods_paired.py \
  --reference baseline=output/fair_eval/hhc_historical_plus_paper_mask_ratio_no_contact_common50 \
  --comparators \
    random_shape_only=output/fair_eval/hhc_historical_paper_mask_shapeaug_random_no_contact_common50 \
    optimized_shape=output/fair_eval/hhc_historical_paper_mask_shapeaug_optimized_no_contact_common50 \
  --output output/fair_eval/paired_current_shape_augmentation_no_contact.json

conda run --no-capture-output -n New_MMM python -u \
  tools/compare_fair_interx_methods_paired.py \
  --reference baseline=output/fair_eval/hhc_historical_plus_paper_mask_ratio_contact_common50 \
  --comparators \
    random_shape_only=output/fair_eval/hhc_historical_paper_mask_shapeaug_random_contact_common50 \
    optimized_shape=output/fair_eval/hhc_historical_paper_mask_shapeaug_optimized_contact_common50 \
  --output output/fair_eval/paired_current_shape_augmentation_contact.json

conda run --no-capture-output -n New_MMM python -u \
  tools/build_current_shape_augmentation_results.py \
  --paired-no-contact output/fair_eval/paired_current_shape_augmentation_no_contact.json \
  --paired-contact output/fair_eval/paired_current_shape_augmentation_contact.json \
  --output output/current_interx_shape_augmentation_results.json \
  --markdown output/current_interx_shape_augmentation_results.md

conda run --no-capture-output -n New_MMM python -u \
  tools/audit_coshmdm_public_release_scope.py
conda run --no-capture-output -n New_MMM python -u \
  tools/build_external_baseline_reproduction_audit.py
conda run --no-capture-output -n New_MMM python -u \
  tools/audit_review_requirement_coverage.py
conda run --no-capture-output -n New_MMM python -u \
  tools/build_current_interx_review_bundle.py

echo "Current Inter-X shape augmentation comparison complete."
