#!/usr/bin/env bash
set -euo pipefail

cd /mnt/8T-HDD/wenning/HHC

run_dir="output/t2i_trans/2026-08-29-16-44-04_mgt_known_good_historical_contract_v7_trainmode_runtime_loss_sampler_seed114514"
reference_wandb="/home/wenning/projects/MMM/wandb/run-20260213_015522-vq_mask_gen_4_ablation_adalnshape_1_cfg/run-vq_mask_gen_4_ablation_adalnshape_1_cfg.wandb"
candidate_wandb="wandb/offline-run-20260829_164412-1axfxrw1/run-1axfxrw1.wandb"
before_checkpoint="${run_dir}/latest-000020000.ckpt"

if [[ ! -s "${before_checkpoint}" ]]; then
  echo 'V7 step-20k rolling checkpoint is missing; refusing an incomplete audit.'
  exit 2
fi

after_checkpoint="$(find "${run_dir}" -maxdepth 1 -type f -name 'model-000020000-*.ckpt' -print -quit)"
if [[ -z "${after_checkpoint}" ]]; then
  echo 'V7 step-20k post-validation checkpoint is missing; validation is not complete.'
  exit 2
fi

conda run --no-capture-output -n New_MMM python -u \
  tools/compare_mgt_wandb_trajectory.py \
  --reference "${reference_wandb}" --candidate "${candidate_wandb}" \
  --all-common-steps --max-abs-loss-diff 0 --skip-sha256 --summary-only \
  --output output/historical_mgt_v7_training_trajectory_20k_audit.json

conda run --no-capture-output -n New_MMM python -c \
  "import json; p='output/historical_mgt_v7_training_trajectory_20k_audit.json'; d=json.load(open(p)); assert d['status']=='pass' and d['latest_available_steps']['candidate']>=19999, d"

conda run --no-capture-output -n New_MMM python -u \
  tools/compare_mgt_wandb_validation.py \
  --reference "${reference_wandb}" --candidate "${candidate_wandb}" \
  --steps 9999 19999 --max-abs-diff 0 --skip-candidate-sha256 \
  --summary-only \
  --output output/historical_mgt_v7_validation_20k_audit.json

conda run --no-capture-output -n New_MMM python -u \
  tools/audit_lightning_checkpoint.py \
  --checkpoint "${before_checkpoint}" \
  --expected-step 20000 --expected-lr 0.0002 --expected-weight-decay 0.000001 \
  --output output/historical_mgt_v7_20k_prevalidation_checkpoint_audit.json

conda run --no-capture-output -n New_MMM python -u \
  tools/audit_lightning_checkpoint_pair.py \
  --before "${before_checkpoint}" --after "${after_checkpoint}" \
  --expected-step 20000 \
  --output output/historical_mgt_v7_20k_pre_post_validation_pair_audit.json

echo 'V7 step-20k trajectory, validation, checkpoint, and side-effect audits passed.'
