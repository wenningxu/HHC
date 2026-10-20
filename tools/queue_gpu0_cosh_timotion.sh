#!/usr/bin/env bash
set -euo pipefail

cd /mnt/8T-HDD/wenning/HHC

while pgrep -f 'python -u tools/train.py --dataset interx' >/dev/null 2>&1; do
  sleep 30
done

# Re-hash the frozen author source/data contract before either accepting or
# resuming a checkpoint.  This branch is normally reached when the existing
# trainer finishes; the resume path also makes a fresh invocation safe after
# an OS restart without ever starting a second trainer alongside a live one.
/home/wenning/miniconda3/bin/conda run --no-capture-output -n New_MMM python -u \
  tools/audit_coshmdm_training_contract.py \
  --output output/coshmdm_training_contract_audit.json

cosh_experiment="CoShMDM-InterX-fair-statsfix-seed123-bs32x2"
cosh_checkpoint="/mnt/8T-HDD/wenning/coshmdm-code/checkpoints/${cosh_experiment}/model/last.ckpt"
cosh_budget_cap="/mnt/8T-HDD/wenning/HHC/output/coshmdm_training_budget_cap.json"
cosh_log="/mnt/8T-HDD/wenning/HHC/output/queue_logs/gpu0_cosh_statsfix_20260830.log"
cosh_state="output/coshmdm_training_state.json"
cosh_status="new"
if [[ -s "${cosh_checkpoint}" ]]; then
  /home/wenning/miniconda3/bin/conda run --no-capture-output -n New_MMM python -u \
    tools/resolve_lightning_checkpoint_state.py \
    --checkpoint "${cosh_checkpoint}" --expected-final-step 568000 \
    --output "${cosh_state}"
  cosh_status="$(/home/wenning/miniconda3/envs/New_MMM/bin/python -c \
    "import json; print(json.load(open('${cosh_state}'))['status'])")"
fi
if [[ -s "${cosh_budget_cap}" ]]; then
  # The user capped this baseline to two additional days.  Never silently
  # resume toward the original 568k author budget after the cap monitor stops
  # it.  Prefer the clean 200k milestone and disclose the reduced budget in
  # all downstream labels/audits.
  preferred_capped_checkpoint="/mnt/8T-HDD/wenning/coshmdm-code/checkpoints/${cosh_experiment}/model/milestone-000200000.ckpt"
  if [[ -s "${preferred_capped_checkpoint}" ]]; then
    cosh_checkpoint="${preferred_capped_checkpoint}"
  fi
  cosh_status="user_capped"
fi
if [[ "${cosh_status}" == new || "${cosh_status}" == resume ]]; then
  resume_args=()
  if [[ "${cosh_status}" == resume ]]; then
    resume_args=(--resume-checkpoint "${cosh_checkpoint}")
  fi
  cd /mnt/8T-HDD/wenning/coshmdm-code
  /home/wenning/miniconda3/bin/conda run --no-capture-output -n New_MMM python -u tools/train.py \
    --dataset interx --gpu 0 --seed 123 --batch-size 32 \
    --accumulate-grad-batches 2 --no-progress-bar \
    --stats-prefix interx_global --exp-name "${cosh_experiment}" \
    "${resume_args[@]}" >> "${cosh_log}" 2>&1
  cd /mnt/8T-HDD/wenning/HHC
  /home/wenning/miniconda3/bin/conda run --no-capture-output -n New_MMM python -u \
    tools/resolve_lightning_checkpoint_state.py \
    --checkpoint "${cosh_checkpoint}" --expected-final-step 568000 \
    --output "${cosh_state}"
fi

/home/wenning/miniconda3/bin/conda run --no-capture-output -n New_MMM python -u \
  tools/audit_coshmdm_representation_adapter.py \
  --output output/coshmdm_representation_adapter_audit.json

if [[ "${cosh_status}" == user_capped ]]; then
  val_request_dir="output/coshmdm_interx_val_requests"
  val_request="${val_request_dir}/validation.npz"
  val_request_manifest="${val_request_dir}/manifest.json"
  if [[ ! -s "${val_request}" || ! -s "${val_request_manifest}" ]]; then
    /home/wenning/miniconda3/bin/conda run --no-capture-output -n New_MMM python -u \
      tools/prepare_interx_validation_requests.py \
      --output-dir "${val_request_dir}" --device cuda:0 \
      --seed 500123 --expected-samples 544
  fi
  /home/wenning/miniconda3/bin/conda run --no-capture-output -n New_MMM python -u \
    tools/select_coshmdm_checkpoint_on_interx_val.py \
    --checkpoint-dir "/mnt/8T-HDD/wenning/coshmdm-code/checkpoints/${cosh_experiment}/model" \
    --requests "${val_request}" \
    --request-manifest "${val_request_manifest}" \
    --budget-cap-manifest "${cosh_budget_cap}" \
    --maximum-step 200000 --gpu 0 --batch-size 32 \
    --output-dir output/coshmdm_interx_val_selection \
    --output output/coshmdm_interx_val_selection.json
  cosh_checkpoint="$(/home/wenning/miniconda3/envs/New_MMM/bin/python -c \
    "import json; print(json.load(open('output/coshmdm_interx_val_selection.json'))['selected_checkpoint'])")"
  cosh_selected_step="$(/home/wenning/miniconda3/envs/New_MMM/bin/python -c \
    "import json; print(json.load(open('output/coshmdm_interx_val_selection.json'))['selected_step'])")"
fi

audit_step_args=(--expected-step 568000)
if [[ "${cosh_status}" == user_capped ]]; then
  # Bind the full-state audit to the validation-selected milestone, rather
  # than merely checking that the selected path is a readable checkpoint.
  audit_step_args=(--expected-step "${cosh_selected_step}")
fi
/home/wenning/miniconda3/bin/conda run --no-capture-output -n New_MMM python -u \
  tools/audit_lightning_checkpoint.py \
  --checkpoint "${cosh_checkpoint}" \
  --output output/coshmdm_final_checkpoint_audit.json \
  "${audit_step_args[@]}"

if [[ "${cosh_status}" == user_capped ]]; then
  /home/wenning/miniconda3/bin/conda run --no-capture-output -n New_MMM python -u \
    tools/audit_coshmdm_resume_provenance.py \
    --output output/coshmdm_resume_provenance.json
  cosh_label="coshmdm_interx_retrained_2daycap_seed123"
else
  /home/wenning/miniconda3/bin/conda run --no-capture-output -n New_MMM python -u \
    tools/audit_coshmdm_resume_provenance.py --require-complete \
    --output output/coshmdm_resume_provenance.json
  cosh_label="coshmdm_interx_retrained_author2000_seed123"
fi

/home/wenning/miniconda3/bin/conda run --no-capture-output -n New_MMM python -u \
  tools/run_fair_interx_method.py --method coshmdm \
  --label "${cosh_label}" --gpu 0 \
  --checkpoint "${cosh_checkpoint}"

# Continue with TiMotion in the same GPU-0 terminal.
bash tools/queue_timotion_fair_after_cosh.sh none
