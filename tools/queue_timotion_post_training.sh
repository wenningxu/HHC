#!/usr/bin/env bash
set -euo pipefail

cd /mnt/8T-HDD/wenning/HHC

conda() {
  /home/wenning/miniconda3/bin/conda "$@"
}

experiment_root="/mnt/8T-HDD/wenning/TIMotion/checkpoints/TIMotion-Transformer-LPA-InterX-fair-seed123-bs32x8"
manifest="${experiment_root}/meta/training_manifest.json"
history="${experiment_root}/meta/official_validation_history.json"
final_checkpoint="${experiment_root}/model/last.ckpt"
selection="output/timotion_interx_val_selection.json"
standard_validation_root="${experiment_root}/official_validation_standard"
standard_history="${experiment_root}/meta/official_validation_history_standard.json"
resume_from="${TIMOTION_FINALIZE_RESUME_FROM:-start}"

# This is a post-training recovery entrypoint, not a waiter.  It exits rather
# than competing with a live trainer, and every later command is resumable or
# content-addressed.
if pgrep -f 'python -u tools/train_interx_fair.py.*TIMotion-Transformer-LPA-InterX-fair-seed123-bs32x8' >/dev/null 2>&1; then
  echo 'TiMotion formal trainer is still active; post-training evaluation is not ready.'
  exit 2
fi
if [[ ! -s "${manifest}" ]] || ! jq -e \
    '.status == "complete" and .final_global_step == 106500' \
    "${manifest}" >/dev/null; then
  echo 'TiMotion formal 106500-step training manifest is incomplete.'
  exit 2
fi

if [[ "${resume_from}" == "start" ]]; then
conda run --no-capture-output -n New_MMM python -u \
  tools/audit_timotion_post_training_contract.py

conda run --no-capture-output -n New_MMM python -u \
  tools/audit_wandb_warmup_stability.py \
  --manifest "${manifest}" --warmup-epochs 10 \
  --metric-profile diffusion_regression_20260831_launch \
  --output output/timotion_warmup_stability_audit.json

conda run --no-capture-output -n New_MMM python -u \
  tools/audit_timotion_transient_velocity_spike.py \
  --manifest "${manifest}" \
  --output output/timotion_transient_velocity_spike_audit.json

conda run --no-capture-output -n New_MMM python -u \
  tools/audit_timotion_scheduler_boundary.py \
  --manifest "${manifest}" \
  --output output/timotion_scheduler_boundary_audit.json

conda run --no-capture-output -n New_MMM python -u \
  tools/audit_timotion_training_contract.py \
  --manifest "${manifest}" \
  --output output/timotion_interx_training_contract_audit.json \
  --rehash-trees

conda run --no-capture-output -n New_MMM python -u \
  tools/audit_lightning_checkpoint.py \
  --checkpoint "${final_checkpoint}" \
  --output output/timotion_interx_final_checkpoint_audit.json \
  --expected-step 106500

conda run --no-capture-output -n New_MMM python -u \
  tools/audit_wandb_online_metric_history.py \
  --manifest "${manifest}" \
  --metric-profile diffusion_regression_20260831_launch \
  --expected-final-step 106500 \
  --output output/timotion_interx_formal_wandb_history_audit.json

conda run --no-capture-output -n New_MMM python -u \
  tools/finalize_timotion_validation_history.py \
  --manifest "${manifest}" --history "${history}"

# Every standard-validation candidate is the lightweight snapshot written by
# the callback immediately before the corresponding full Lightning
# checkpoint.  Prove all 15 state_dicts are bitwise identical before replaying
# or selecting them; this prevents a stale weights-only file from silently
# becoming a validation candidate.
steps_per_epoch="$(/home/wenning/miniconda3/envs/New_MMM/bin/python -c \
  "import json; print(json.load(open('${manifest}'))['batch_emulation']['optimizer_updates_per_epoch'])")"
final_epoch="$(/home/wenning/miniconda3/envs/New_MMM/bin/python -c \
  "import json; print(json.load(open('${manifest}'))['paper_protocol']['epochs'])")"
for epoch in $(seq 100 100 "${final_epoch}"); do
  step=$((epoch * steps_per_epoch))
  full_checkpoint="${experiment_root}/model/epoch=$((epoch - 1))-step=${step}.ckpt"
  weights_checkpoint="${experiment_root}/official_validation/step_$(printf '%09d' "${step}")_weights.ckpt"
  metrics="${experiment_root}/official_validation/step_$(printf '%09d' "${step}").json"
  conda run --no-capture-output -n New_MMM python -u \
    tools/audit_timotion_validation_checkpoint_pair.py \
    --full-checkpoint "${full_checkpoint}" \
    --weights-checkpoint "${weights_checkpoint}" --metrics "${metrics}" \
    --expected-step "${step}" \
    --output "output/timotion_step$(printf '%09d' "${step}")_validation_checkpoint_pair_audit.json"
done

# The callback-time FIDs above retain the historical repository decoder and
# are diagnostic only.  Re-run every held-out candidate with the formal
# standard row-concatenated 6D adapter before selecting any checkpoint.
conda run --no-capture-output -n New_MMM python -u \
  tools/audit_timotion_cosh_adapter_equivalence.py \
  --timotion-adapter tools/export_timotion_official_interx_standard.py \
  --timotion-rotation-contract standard \
  --output output/timotion_cosh_adapter_equivalence_audit_standard_formal.json

conda run --no-capture-output -n New_MMM python -u \
  tools/audit_timotion_generation_contract.py \
  --output output/timotion_generation_contract_audit_standard_formal.json
elif [[ "${resume_from}" == "replay" || "${resume_from}" == "fair_test" ]]; then
  required_audits=(
    output/timotion_post_training_contract_audit.json
    output/timotion_warmup_stability_audit.json
    output/timotion_transient_velocity_spike_audit.json
    output/timotion_scheduler_boundary_audit.json
    output/timotion_interx_training_contract_audit.json
    output/timotion_interx_final_checkpoint_audit.json
    output/timotion_interx_formal_wandb_history_audit.json
    output/timotion_cosh_adapter_equivalence_audit_standard_formal.json
    output/timotion_generation_contract_audit_standard_formal.json
  )
  for audit in "${required_audits[@]}"; do
    if [[ ! -s "${audit}" ]] || ! jq -e '.status == "pass"' "${audit}" >/dev/null; then
      echo "TiMotion replay resume prerequisite failed: ${audit}"
      exit 2
    fi
  done
  for epoch in $(seq 100 100 1500); do
    step=$((epoch * 71))
    pair_audit="output/timotion_step$(printf '%09d' "${step}")_validation_checkpoint_pair_audit.json"
    if [[ ! -s "${pair_audit}" ]] || ! jq -e '.status == "pass"' "${pair_audit}" >/dev/null; then
      echo "TiMotion replay checkpoint-pair prerequisite failed: ${pair_audit}"
      exit 2
    fi
  done
  if [[ ! -s "${history}" ]] || ! jq -e \
      '.status == "complete" and .final_global_step == 106500 and (.records | length) == 15' \
      "${history}" >/dev/null; then
    echo 'TiMotion replay validation inventory is incomplete.'
    exit 2
  fi
  echo 'TiMotion replay resume prerequisites passed; skipping completed audits.'
else
  echo "Unknown TIMOTION_FINALIZE_RESUME_FROM=${resume_from}"
  exit 2
fi

if [[ "${resume_from}" != "fair_test" ]]; then
  conda run --no-capture-output -n New_MMM python -u \
    tools/replay_timotion_standard_validation.py \
    --manifest "${manifest}" --history "${history}" \
    --output-root "${standard_validation_root}" \
    --output-history "${standard_history}" --gpu 0 --batch-size 32

  conda run --no-capture-output -n New_MMM python -u \
    tools/select_timotion_checkpoint_on_interx_val.py \
    --manifest "${manifest}" --history "${standard_history}" \
    --output "${selection}"
else
  if [[ ! -s "${standard_history}" ]] || ! jq -e \
      '.status == "complete" and .final_global_step == 106500 and (.records | length) == 15' \
      "${standard_history}" >/dev/null; then
    echo 'TiMotion fair-test resume standard validation history is incomplete.'
    exit 2
  fi
  if [[ ! -s "${selection}" ]] || ! jq -e \
      '.status == "complete" and .test_metrics_used_for_selection == false' \
      "${selection}" >/dev/null; then
    echo 'TiMotion fair-test resume selection is incomplete.'
    exit 2
  fi
  echo 'TiMotion fair-test resume prerequisites passed; skipping validation replay.'
fi

selected_checkpoint="$(/home/wenning/miniconda3/envs/New_MMM/bin/python -c \
  "import json; print(json.load(open('${selection}'))['selected_checkpoint'])")"

conda run --no-capture-output -n New_MMM python -u \
  tools/run_fair_interx_method.py \
  --method timotion --label timotion_transformer_lpa_interx_retrained_fair \
  --checkpoint "${selected_checkpoint}" --selection-manifest "${selection}" \
  --gpu 0 --batch-size 32

bash tools/finalize_current_interx_if_ready.sh 0
