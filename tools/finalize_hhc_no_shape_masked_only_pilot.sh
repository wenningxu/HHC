#!/usr/bin/env bash
set -euo pipefail

cd /mnt/8T-HDD/wenning/HHC

run_dir="output/t2i_trans/2026-09-09-17-39-10_hhc_historical_paper_mask_no_shape_masked_only_pilot50k_seed114514_v1"
manifest="${run_dir}/training_manifest.json"
selection="output/hhc_no_shape_masked_only_pilot50k_validation_selection.json"
selection_dir="output/hhc_no_shape_masked_only_pilot50k_validation_candidates"
wandb_audit="output/hhc_no_shape_masked_only_pilot50k_wandb_history_audit.json"

/home/wenning/miniconda3/bin/conda run --no-capture-output -n New_MMM \
  python -u tools/audit_lightning_checkpoint.py \
  --checkpoint "${run_dir}/latest-000050000.ckpt" --expected-step 50000 \
  --expected-lr 0.0002 --expected-weight-decay 0.000001 \
  --output output/hhc_no_shape_masked_only_pilot50k_checkpoint_audit.json

/home/wenning/miniconda3/bin/conda run --no-capture-output -n New_MMM \
  python -u tools/audit_wandb_online_metric_history.py \
  --manifest "${manifest}" --metric-profile masked_token_classification \
  --expected-final-step 50000 --output "${wandb_audit}"

/home/wenning/miniconda3/bin/conda run --no-capture-output -n New_MMM \
  python -u tools/select_no_shape_checkpoint_on_official_interx_val.py \
  --run-dir "${run_dir}" \
  --recovery-checkpoint output/hhc_no_shape_masked_only_recovery/pre_validation_000010000.ckpt \
  --stats-dir dataset/interx \
  --cache-root checkpoints/vq_precleanup_raw12288_historical_mmm_cache \
  --raw-ground-truth-manifest output/interx_validation_raw_ground_truth_manifest.json \
  --output-dir "${selection_dir}" --output "${selection}" \
  --device cuda:0 --batch-size 32 --final-step 50000 --interval 10000 \
  --expected-training-loss-contract paper_masked_only

/home/wenning/miniconda3/bin/conda run --no-capture-output -n New_MMM \
  python -u tools/compare_hhc_no_shape_masked_only_pilot.py \
  --control-manifest output/t2i_trans/2026-09-07-11-49-54_hhc_historical_paper_mask_no_mgt_shape_pilot50k_seed114514_v4/training_manifest_at_50k.json \
  --control-wandb-audit output/hhc_no_shape_control_step_aligned_wandb_audit.json \
  --control-selection output/hhc_no_shape_v4_pilot50k_validation_selection.json \
  --pilot-manifest "${manifest}" --pilot-wandb-audit "${wandb_audit}" \
  --pilot-selection "${selection}" \
  --source-equivalence-audit output/hhc_no_shape_masked_only_source_equivalence_audit.json \
  --output output/hhc_no_shape_masked_only_pilot50k_comparison.json
