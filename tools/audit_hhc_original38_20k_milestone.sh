#!/usr/bin/env bash
set -euo pipefail

cd /mnt/8T-HDD/wenning/HHC

conda() {
  /home/wenning/miniconda3/bin/conda "$@"
}

experiment="hhc_original_mmm38_window_control_raw12288_300k_seed123_v1"
run_dir="output/t2i_trans/2026-09-01-06-18-28_${experiment}"
candidate_wandb="wandb/run-20260901_061834-rx72y8g9/run-rx72y8g9.wandb"
reference_wandb="wandb/offline-run-20260830_160819-k72vurdd/run-k72vurdd.wandb"
before="${run_dir}/prevalidation-000020000.ckpt"
after="${run_dir}/postvalidation-000020000.ckpt"

for required in \
  "${candidate_wandb}" "${reference_wandb}" \
  "${run_dir}/latest-000020000.ckpt" "${before}" "${after}"
do
  if [[ ! -s "${required}" ]]; then
    echo "Original-38 20k milestone is incomplete: ${required}"
    exit 2
  fi
done

conda run --no-capture-output -n New_MMM python -u \
  tools/compare_mgt_wandb_validation.py \
  --reference "${reference_wandb}" --candidate "${candidate_wandb}" \
  --steps 9999 19999 --checkpoint-step-offset 1 \
  --skip-candidate-sha256 --summary-only \
  --output output/hhc_original38_vs_formal50_validation_10k_20k.json

conda run --no-capture-output -n New_MMM python -u \
  tools/compare_mgt_wandb_trajectory.py \
  --reference "${reference_wandb}" --candidate "${candidate_wandb}" \
  --steps 999 4999 9999 14999 19999 \
  --skip-sha256 --summary-only \
  --output output/hhc_original38_vs_formal50_training_trajectory_through20k.json

conda run --no-capture-output -n New_MMM python -u \
  tools/select_best_mgt_checkpoint.py \
  --run-glob "${run_dir}" --expected-final-step 20000 \
  --output output/hhc_original38_token_window_control_20k_selection.json

conda run --no-capture-output -n New_MMM python -u \
  tools/audit_lightning_checkpoint_pair.py \
  --before "${before}" --after "${after}" --expected-step 20000 \
  --require-fixed-pair-marker \
  --output output/hhc_original38_token_window_control_20k_pre_post_validation_pair_audit.json

conda run --no-capture-output -n New_MMM python -u \
  tools/resolve_mgt_training_state.py \
  --run-glob "output/t2i_trans/*_${experiment}" \
  --experiment "${experiment}" --expected-final-step 50000 \
  --acknowledge-failed-no-checkpoint-run-id ans4ij26 \
  --acknowledge-failed-no-checkpoint-run-id qgcs8cnb \
  --output output/hhc_original38_token_window_control_training_state.json
