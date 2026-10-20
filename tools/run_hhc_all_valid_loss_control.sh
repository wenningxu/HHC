#!/usr/bin/env bash
set -euo pipefail

cd /mnt/8T-HDD/wenning/HHC

conda() {
  /home/wenning/miniconda3/bin/conda "$@"
}

experiment="hhc_all_valid_loss_control_raw12288_300k_seed123_v1"
formal_selection="output/hhc_formal_raw12288_repaired_mgt_selection.json"
selection="output/hhc_all_valid_loss_control_50k_selection.json"
assessment="output/hhc_all_valid_loss_control_assessment.json"
training_state="output/hhc_all_valid_loss_control_training_state.json"
cache_root="checkpoints/vq_precleanup_raw12288_formal_mgt_v1"
vq_checkpoint="checkpoints/vq_precleanup_raw12288/net_175000.pth"

for required in \
  "${formal_selection}" \
  output/post_formal_online_wandb_metrics_application.json
do
  if [[ ! -s "${required}" ]]; then
    echo "All-valid control prerequisite is missing: ${required}"
    exit 2
  fi
done

conda run --no-capture-output -n New_MMM python -u \
  tools/assess_hhc_all_valid_loss_control.py \
  --formal-selection "${formal_selection}" --output "${assessment}"

conda run --no-capture-output -n New_MMM python -u \
  tools/audit_mgt_historical_loss_contract.py \
  --output output/historical_mgt_loss_reduction_audit.json

conda run --no-capture-output -n New_MMM python -u \
  tools/audit_online_wandb_training_contract.py \
  --launcher tools/run_hhc_all_valid_loss_control.sh \
  --training-source train/train_t2i_trans.py \
  --output output/hhc_all_valid_loss_control_online_wandb_audit.json

conda run --no-capture-output -n New_MMM python -u \
  tools/resolve_mgt_training_state.py \
  --run-glob "output/t2i_trans/*_${experiment}" \
  --experiment "${experiment}" --expected-final-step 50000 \
  --output "${training_state}"
training_status="$(/home/wenning/miniconda3/envs/New_MMM/bin/python -c \
  "import json; print(json.load(open('${training_state}'))['status'])")"

if [[ "${training_status}" != complete ]]; then
  resume_args=()
  if [[ "${training_status}" == resume ]]; then
    resume_checkpoint="$(/home/wenning/miniconda3/envs/New_MMM/bin/python -c \
      "import json; print(json.load(open('${training_state}'))['resume_checkpoint'])")"
    resume_step="$(/home/wenning/miniconda3/envs/New_MMM/bin/python -c \
      "import json; print(json.load(open('${training_state}'))['active_run']['latest_step'])")"
    conda run --no-capture-output -n New_MMM python -u \
      tools/audit_lightning_checkpoint.py \
      --checkpoint "${resume_checkpoint}" --expected-step "${resume_step}" \
      --expected-weight-decay 0.000001 \
      --output output/hhc_all_valid_loss_control_resume_checkpoint_audit.json
    resume_args=(--resume-training-ckpt "${resume_checkpoint}" --resume-in-place)
  fi
  conda run --no-capture-output -n New_MMM python -u \
    train/train_t2i_trans.py \
    --device cuda:1 --exp-name "${experiment}" \
    --vq-checkpoint "${vq_checkpoint}" \
    --cache-root "${cache_root}" --codebook-dir "${cache_root}/codebook" \
    --token-cache-contract historical_mmm_exact \
    --historical-codebook-source /home/wenning/projects/MMM/checkpoints/vq/codebook \
    --historical-contact-source /home/wenning/projects/MMM/checkpoints/vq/contact_token_label \
    --nb-code 12288 --code-dim 256 --shape-condition-dim 8 --width 512 \
    --embed-dim-gpt 512 --n-head-gpt 8 --num-layers 9 --num-local-layer 9 \
    --batch-size 96 --total-iter 300000 --diagnostic-stop-iter 50000 \
    --warm-up-iter 1000 --lr 0.0002 --weight-decay 0.000001 \
    --contact --no-contact-conditioning --shape --paper-architecture \
    --training-mask-contract paper_kinematic \
    --training-loss-contract historical_all_valid \
    --optimizer-contract paper_warmup_cosine \
    --conditioning-contract paper_text_cfg \
    --training-token-length-contract paper_50 \
    --training-validation-contract paper_isolated_common \
    --sampler sample_intra_cas --temperature 0.5 --sampling-steps 50 \
    --cfg-scale 3.5 --contact-guidance-scale 1.0 \
    --wandb-mode online --seed 123 "${resume_args[@]}"
fi

conda run --no-capture-output -n New_MMM python -u \
  tools/select_best_mgt_checkpoint.py \
  --run-glob "output/t2i_trans/*_${experiment}" \
  --expected-final-step 50000 --output "${selection}"
training_manifest="$(/home/wenning/miniconda3/envs/New_MMM/bin/python -c \
  "import json; print(json.load(open('${selection}'))['training_manifest'])")"
run_dir="$(dirname "${training_manifest}")"
wandb_run_id="$(/home/wenning/miniconda3/envs/New_MMM/bin/python -c \
  "import json; print(json.load(open('${training_manifest}'))['observability']['wandb_run_id'])")"
candidate_wandb="$(find wandb -maxdepth 2 -type f \
  -path "*/run-${wandb_run_id}.wandb" -print -quit)"
if [[ ! -f "${candidate_wandb}" ]]; then
  echo "Local W&B transaction log is missing for run ${wandb_run_id}"
  exit 2
fi

conda run --no-capture-output -n New_MMM python -u \
  tools/compare_mgt_wandb_validation.py \
  --reference wandb/offline-run-20260830_160819-k72vurdd/run-k72vurdd.wandb \
  --candidate "${candidate_wandb}" \
  --steps 9999 19999 29999 39999 49999 \
  --checkpoint-step-offset 1 --summary-only \
  --output output/hhc_all_valid_vs_formal50_validation_10k_50k.json

conda run --no-capture-output -n New_MMM python -u \
  tools/freeze_mgt_training_sources.py \
  --training-manifest "${training_manifest}" \
  --output-dir "${run_dir}/launch_sources" \
  --output-manifest "${run_dir}/launch_source_snapshot_manifest.json"

conda run --no-capture-output -n New_MMM python -u \
  tools/audit_mgt_wandb_warmup.py \
  --manifest "${training_manifest}" \
  --output output/hhc_all_valid_loss_control_mgt_wandb_warmup_audit.json

conda run --no-capture-output -n New_MMM python -u \
  tools/audit_wandb_warmup_stability.py \
  --manifest "${training_manifest}" \
  --max-weight-relative-span 0.15 \
  --output output/hhc_all_valid_loss_control_wandb_audit.json

conda run --no-capture-output -n New_MMM python -u \
  tools/audit_wandb_online_metric_history.py \
  --manifest "${training_manifest}" \
  --metric-profile masked_token_classification --expected-final-step 50000 \
  --output output/hhc_all_valid_loss_control_wandb_history_audit.json

conda run --no-capture-output -n New_MMM python -u \
  tools/assess_hhc_all_valid_loss_control.py \
  --formal-selection "${formal_selection}" \
  --control-selection "${selection}" --output "${assessment}"
