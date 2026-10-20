#!/usr/bin/env bash
set -euo pipefail

cd /mnt/8T-HDD/wenning/HHC

if [[ $# -ne 3 || ! "$1" =~ ^[0-9]+$ ]]; then
  echo "usage: $0 GPU_INDEX OPTIMIZED_50K_RUN_DIR RANDOM_ONLY_50K_RUN_DIR" >&2
  exit 2
fi

gpu="$1"
optimized_run="$2"
random_run="$3"

# The frozen unaugmented result is reused by both arms. Each new arm performs
# its own held-out-val checkpoint selection and the same 20-rep common50 test.
bash tools/finalize_no_shape_amass_aug_joint.sh \
  "${gpu}" "${optimized_run}" optimized_selective
bash tools/finalize_no_shape_amass_aug_joint.sh \
  "${gpu}" "${random_run}" random_only

optimized_label="hhc_no_mgt_shape_amass_optimizedaug_pilot50k_bestval_common50"
random_label="hhc_no_mgt_shape_amass_randomaug_pilot50k_bestval_common50"
optimized_scores="output/interact2ar_joint_evaluator_candidates/${optimized_label}/scores/${optimized_label}"
random_scores="output/interact2ar_joint_evaluator_candidates/${random_label}/scores/${random_label}"
matrix_root="output/interact2ar_joint_evaluator_candidates/hhc_no_mgt_shape_amass_aug_matrix"

mkdir -p "${matrix_root}"
/home/wenning/miniconda3/envs/New_MMM/bin/python \
  tools/compare_interact2ar_joint_augmentation_paired.py \
  --baseline "${random_scores}" \
  --augmented "${optimized_scores}" \
  --comparison-kind optimization_effect_with_fixed_amass_donors \
  --claim-scope matched_training \
  --replications 20 \
  --output "${matrix_root}/random_vs_optimized_paired.json"

/home/wenning/miniconda3/envs/New_MMM/bin/python \
  tools/build_amass_shape_augmentation_result_card.py \
  --optimized-run "${optimized_run}" \
  --random-run "${random_run}" \
  --output output/hhc_amass_shape_augmentation_result_card.json

echo "AMASS no-augmentation/random-only/optimized-selective matrix complete."
