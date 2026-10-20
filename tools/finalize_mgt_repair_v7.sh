#!/usr/bin/env bash
set -euo pipefail

cd /mnt/8T-HDD/wenning/HHC

conda_bin="/home/wenning/miniconda3/bin/conda"
if [[ ! -x "${conda_bin}" ]]; then
  echo "Required Conda executable is missing: ${conda_bin}"
  exit 2
fi

run_dir="output/t2i_trans/2026-08-29-16-44-04_mgt_known_good_historical_contract_v7_trainmode_runtime_loss_sampler_seed114514"
reference_wandb="/home/wenning/projects/MMM/wandb/run-20260213_015522-vq_mask_gen_4_ablation_adalnshape_1_cfg/run-vq_mask_gen_4_ablation_adalnshape_1_cfg.wandb"
candidate_wandb="wandb/offline-run-20260829_164412-1axfxrw1/run-1axfxrw1.wandb"
reference_checkpoint="/home/wenning/projects/MMM/output/t2i_trans/2026-02-13-01-55-15_vq_mask_gen_4_ablation_adalnshape_1_cfg/model-step=280000-val/FID=0.1605.ckpt"
reference_metrics="output/fair_eval/hhc_precleanup_git_default_mgt_common50_contact_rep00/replication_00_ordinary_rescore.json"
candidate_label="hhc_mgt_repair_v7_step280k_common50_paper_text_cfg_contact_rep00"
candidate_metrics="output/fair_eval/${candidate_label}/replication_00_ordinary_rescore.json"
reference_generation_root="output/fair_eval/hhc_precleanup_git_default_mgt_common50_contact_rep00"

if pgrep -f 'python -u train/train_t2i_trans.py.*mgt_known_good_historical_contract_v7' >/dev/null 2>&1; then
  echo 'V7 is still training; refusing premature finalization.'
  exit 2
fi
if [[ ! -s "${run_dir}/latest-000300000.ckpt" ]]; then
  echo 'V7 final 300k checkpoint is missing.'
  exit 2
fi

candidate_checkpoint="$(find "${run_dir}" -maxdepth 1 -type f -name 'model-000280000-*.ckpt' -print -quit)"
if [[ -z "${candidate_checkpoint}" ]]; then
  echo 'V7 step-280k validation checkpoint is missing.'
  exit 2
fi

"${conda_bin}" run --no-capture-output -n New_MMM python -u \
  tools/compare_mgt_wandb_trajectory.py \
  --reference "${reference_wandb}" --candidate "${candidate_wandb}" \
  --all-common-steps --max-abs-loss-diff 0 \
  --summary-only \
  --output output/historical_mgt_v7_training_trajectory_300k_audit.json

validation_steps=()
for ((step=9999; step<=299999; step+=10000)); do
  validation_steps+=("${step}")
done
"${conda_bin}" run --no-capture-output -n New_MMM python -u \
  tools/compare_mgt_wandb_validation.py \
  --reference "${reference_wandb}" --candidate "${candidate_wandb}" \
  --steps "${validation_steps[@]}" --max-abs-diff 0 \
  --summary-only \
  --output output/historical_mgt_v7_validation_300k_audit.json

"${conda_bin}" run --no-capture-output -n New_MMM python -u \
  tools/audit_mgt_historical_checkpoint_pair.py \
  --reference "${reference_checkpoint}" --candidate "${candidate_checkpoint}" \
  --expected-step 280000 \
  --output output/historical_mgt_v7_step280k_checkpoint_pair_audit.json

"${conda_bin}" run --no-capture-output -n New_MMM python -u \
  tools/run_fair_interx_method.py --method hhc \
  --label "${candidate_label}" --gpu 1 \
  --checkpoint "${candidate_checkpoint}" \
  --vq-checkpoint checkpoints/vq_precleanup_raw12288/net_175000.pth \
  --hhc-vq-contract precleanup_raw12288 --hhc-mgt-contract historical_mmm \
  --hhc-inference-conditioning-contract paper_text_cfg \
  --hhc-contact-map checkpoints/vq_precleanup_raw12288/code_contact_probs.npy \
  --hhc-contact-map-contract paper_formula_train_only \
  --sampling-steps 50 --temperature 0.5 --cfg-scale 3.5 \
  --contact-guidance-scale 1.0 --hhc-contact-timing-source request_oracle \
  --contact-guidance --start-replication 0 --end-replication 1

# Re-score both frozen HDF5 pairs with the same current ordinary-only scorer.
# The reference generation predates explicit evaluator provenance fields, so
# reusing an older JSON here would not prove evaluator identity.
"${conda_bin}" run --no-capture-output -n New_MMM python -u \
  tools/score_fair_interx_generations.py \
  --requests output/fair_interx_requests/replication_00.npz \
  --main-generations "${reference_generation_root}/replication_00_main.h5" \
  --mm-generations "${reference_generation_root}/replication_00_mm.h5" \
  --output "${reference_metrics}" --device cpu --batch-size 32

"${conda_bin}" run --no-capture-output -n New_MMM python -u \
  tools/score_fair_interx_generations.py \
  --requests output/fair_interx_requests/replication_00.npz \
  --main-generations "output/fair_eval/${candidate_label}/replication_00_main.h5" \
  --mm-generations "output/fair_eval/${candidate_label}/replication_00_mm.h5" \
  --output "${candidate_metrics}" --device cpu --batch-size 32

"${conda_bin}" run --no-capture-output -n New_MMM python -u \
  tools/compare_mgt_common50_bridge.py \
  --reference "${reference_metrics}" --candidate "${candidate_metrics}" \
  --expected-step 280000 --max-abs-fid-diff 0.02 \
  --output output/historical_mgt_v7_step280k_common50_paper_text_cfg_bridge.json

# Rebind the frozen VQ reconstruction to the current main-only scorer.  The
# scorer now honors an HDF5 effective-length field for baselines whose output
# length differs from the request; the VQ reconstruction is exact-length, so
# every old/new metric must remain bitwise identical before refreshing the gate.
"${conda_bin}" run --no-capture-output -n New_MMM python -u \
  tools/score_fair_interx_main.py \
  --requests output/fair_interx_requests/replication_00.npz \
  --main-generations output/vq_historical_eval/net_175000_reconstructions.h5 \
  --output output/vq_historical_eval/net_175000_reconstruction_ordinary_fid_current_scorer.json \
  --device cpu --batch-size 32

"${conda_bin}" run --no-capture-output -n New_MMM python -u \
  tools/audit_vq_reconstruction_scorer_migration.py \
  --before output/vq_historical_eval/net_175000_reconstruction_ordinary_fid.json \
  --after output/vq_historical_eval/net_175000_reconstruction_ordinary_fid_current_scorer.json \
  --output output/vq_historical_eval/net_175000_reconstruction_scorer_migration_audit.json

"${conda_bin}" run --no-capture-output -n New_MMM python -u \
  tools/build_mgt_repair_gate.py \
  --vq-reconstruction output/vq_historical_eval/net_175000_reconstruction_ordinary_fid_current_scorer.json \
  --trajectory-audit output/historical_mgt_v7_training_trajectory_300k_audit.json \
  --validation-audit output/historical_mgt_v7_validation_300k_audit.json \
  --checkpoint-pair-audit output/historical_mgt_v7_step280k_checkpoint_pair_audit.json \
  --common50-bridge output/historical_mgt_v7_step280k_common50_paper_text_cfg_bridge.json \
  --launch-source-recovery output/historical_mgt_v7_launch_source_recovery.json \
  --scorer-migration-audit output/vq_historical_eval/net_175000_reconstruction_scorer_migration_audit.json \
  --output output/mgt_repair_gate.json

"${conda_bin}" run --no-capture-output -n New_MMM python -u \
  tools/require_mgt_repair_gate.py --gate output/mgt_repair_gate.json
