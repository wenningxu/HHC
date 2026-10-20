#!/usr/bin/env bash
set -euo pipefail

cd /mnt/8T-HDD/wenning/HHC

conda() {
  /home/wenning/miniconda3/bin/conda "$@"
}

formal_selection="output/hhc_formal_raw12288_repaired_mgt_selection.json"
assessment="output/hhc_global_masked_loss_control_assessment.json"
dispatch="output/hhc_post_50k_training_dispatch.json"
shape_pair="output/hhc_formal_mgt_shape_dependence_pair.json"
control_selection="output/hhc_global_masked_loss_control_50k_selection.json"
experiment="hhc_global_masked_loss_control_raw12288_300k_seed123_v1"
training_state="output/hhc_global_masked_loss_control_training_state.json"
cache_root="checkpoints/vq_precleanup_raw12288_formal_mgt_v1"
vq_checkpoint="checkpoints/vq_precleanup_raw12288/net_175000.pth"

dispatch_completed_assessment() {
  local completed_decision
  completed_decision="$(/home/wenning/miniconda3/envs/New_MMM/bin/python -c \
    "import json; print(json.load(open('${assessment}'))['decision'])")"
  local dispatch_args=(--assessment "${assessment}" --output "${dispatch}")
  if [[ "${completed_decision}" == global_loss_not_sufficient_investigate_mask_conditioning_next ]]; then
    if [[ ! -s "${shape_pair}" ]]; then
      echo "Negative loss control requires completed shape evidence: ${shape_pair}"
      return 2
    fi
    dispatch_args+=(--shape-pair "${shape_pair}")
  fi
  conda run --no-capture-output -n New_MMM python -u \
    tools/dispatch_hhc_post_50k_training.py "${dispatch_args[@]}"
  local dispatch_action
  dispatch_action="$(/home/wenning/miniconda3/envs/New_MMM/bin/python -c \
    "import json; print(json.load(open('${dispatch}'))['action'])")"
  if [[ "${dispatch_action}" == continue_global_control_to_300k_then_shape ]]; then
    bash tools/continue_hhc_global_masked_loss_control_300k.sh
  fi
}

control_artifacts=(
  "${control_selection}"
  output/hhc_formal_mgt_mask_exposure_diagnostic.json
  output/hhc_formal_mgt_mask_exposure_training_diagnostic.json
  output/hhc_global_masked_loss_control_validation_mask_exposure.json
  output/hhc_global_masked_loss_control_training_mask_exposure.json
)
control_complete=true
for control_artifact in "${control_artifacts[@]}"; do
  if [[ ! -s "${control_artifact}" ]]; then
    control_complete=false
  fi
done

if [[ "${control_complete}" == true ]]; then
  conda run --no-capture-output -n New_MMM python -u \
    tools/assess_hhc_global_masked_loss_control.py \
    --formal-selection "${formal_selection}" \
    --control-selection "${control_selection}" \
    --baseline-validation-mask-exposure \
      output/hhc_formal_mgt_mask_exposure_diagnostic.json \
    --baseline-training-mask-exposure \
      output/hhc_formal_mgt_mask_exposure_training_diagnostic.json \
    --control-validation-mask-exposure \
      output/hhc_global_masked_loss_control_validation_mask_exposure.json \
    --control-training-mask-exposure \
      output/hhc_global_masked_loss_control_training_mask_exposure.json \
    --output "${assessment}"
  dispatch_completed_assessment
  exit 0
fi

conda run --no-capture-output -n New_MMM python -u \
  tools/assess_hhc_global_masked_loss_control.py \
  --formal-selection "${formal_selection}" --output "${assessment}"
decision="$(/home/wenning/miniconda3/envs/New_MMM/bin/python -c \
  "import json; print(json.load(open('${assessment}'))['decision'])")"
if [[ "${decision}" == retain_formal_contract_and_continue_shape_training ]]; then
  dispatch_completed_assessment
  exit 0
fi

# Keep this entrypoint independently restartable.  The normal GPU-1 queue
# applies these two guarded post-formal changes immediately before calling
# this script, but an operator may need to resume here after a host restart.
# Both appliers are idempotent and fail closed unless the completed formal
# common-50 artifacts and frozen source hashes still match.
bash tools/apply_post_formal_shape_root_tokenizer_wiring.sh
conda run --no-capture-output -n New_MMM python -u \
  tools/apply_post_formal_online_wandb_metrics.py --apply
bash tools/apply_post_formal_global_masked_loss_diagnostic.sh

conda run --no-capture-output -n New_MMM python -u \
  tools/audit_wandb_online_runtime.py \
  --output output/hhc_global_masked_loss_control_wandb_runtime_audit.json
conda run --no-capture-output -n New_MMM python -u \
  tools/audit_online_wandb_training_contract.py \
  --launcher tools/run_hhc_global_masked_loss_control.sh \
  --training-source train/train_t2i_trans.py \
  --output output/hhc_global_masked_loss_control_online_wandb_audit.json

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
      --output output/hhc_global_masked_loss_control_resume_checkpoint_audit.json
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
    --training-loss-contract paper_masked_global_token_mean \
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
  --expected-final-step 50000 --output "${control_selection}"
control_training_manifest="$(/home/wenning/miniconda3/envs/New_MMM/bin/python -c \
  "import json; print(json.load(open('${control_selection}'))['training_manifest'])")"
control_checkpoint="$(/home/wenning/miniconda3/envs/New_MMM/bin/python -c \
  "import json; print(json.load(open('${control_selection}'))['selected_checkpoint'])")"
conda run --no-capture-output -n New_MMM python -u \
  tools/audit_wandb_online_metric_history.py \
  --manifest "${control_training_manifest}" \
  --metric-profile masked_token_classification \
  --expected-final-step 50000 \
  --output output/hhc_global_masked_loss_control_wandb_history_audit.json

conda run --no-capture-output -n New_MMM python -u \
  tools/audit_mgt_mask_exposure_gap.py \
  --training-manifest "${control_training_manifest}" \
  --checkpoint "${control_checkpoint}" --device cuda:1 \
  --samples 256 --batch-size 16 --seed 90123 \
  --output output/hhc_global_masked_loss_control_validation_mask_exposure.json
conda run --no-capture-output -n New_MMM python -u \
  tools/audit_mgt_mask_exposure_gap.py \
  --training-manifest "${control_training_manifest}" \
  --checkpoint "${control_checkpoint}" --device cuda:1 \
  --samples 256 --batch-size 16 --seed 90123 --split-role training \
  --output output/hhc_global_masked_loss_control_training_mask_exposure.json

conda run --no-capture-output -n New_MMM python -u \
  tools/assess_hhc_global_masked_loss_control.py \
  --formal-selection "${formal_selection}" \
  --control-selection "${control_selection}" \
  --baseline-validation-mask-exposure \
    output/hhc_formal_mgt_mask_exposure_diagnostic.json \
  --baseline-training-mask-exposure \
    output/hhc_formal_mgt_mask_exposure_training_diagnostic.json \
  --control-validation-mask-exposure \
    output/hhc_global_masked_loss_control_validation_mask_exposure.json \
  --control-training-mask-exposure \
    output/hhc_global_masked_loss_control_training_mask_exposure.json \
  --output "${assessment}"

dispatch_completed_assessment
