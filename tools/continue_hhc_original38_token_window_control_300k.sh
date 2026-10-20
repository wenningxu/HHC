#!/usr/bin/env bash
set -euo pipefail

cd /mnt/8T-HDD/wenning/HHC

conda() {
  /home/wenning/miniconda3/bin/conda "$@"
}

assessment="output/hhc_original38_token_window_control_assessment.json"
decision="$(/home/wenning/miniconda3/envs/New_MMM/bin/python -c \
  "import json; print(json.load(open('${assessment}'))['decision'])")"
if [[ "${decision}" != promote_original_mmm_38_window ]]; then
  echo "Original-38 window was not promoted: ${decision}"
  exit 2
fi

experiment="hhc_original_mmm38_window_control_raw12288_300k_seed123_v1"
training_state="output/hhc_original38_token_window_control_300k_training_state.json"
selection="output/hhc_original38_token_window_control_300k_selection.json"
vq_checkpoint="checkpoints/vq_precleanup_raw12288/net_175000.pth"
cache_root="checkpoints/vq_precleanup_raw12288_formal_mgt_v1"

resolve_state() {
  conda run --no-capture-output -n New_MMM python -u \
    tools/resolve_mgt_training_state.py \
    --run-glob "output/t2i_trans/*_${experiment}" \
    --experiment "${experiment}" --expected-final-step 300000 \
    --output "${training_state}"
}

resolve_state
training_status="$(/home/wenning/miniconda3/envs/New_MMM/bin/python -c \
  "import json; print(json.load(open('${training_state}'))['status'])")"
if [[ "${training_status}" == new ]]; then
  echo 'The promoted original-38 run has no 50k lineage to resume.'
  exit 2
fi
if [[ "${training_status}" == resume ]]; then
  resume_checkpoint="$(/home/wenning/miniconda3/envs/New_MMM/bin/python -c \
    "import json; print(json.load(open('${training_state}'))['resume_checkpoint'])")"
  resume_step="$(/home/wenning/miniconda3/envs/New_MMM/bin/python -c \
    "import json; print(json.load(open('${training_state}'))['active_run']['latest_step'])")"
  conda run --no-capture-output -n New_MMM python -u \
    tools/audit_lightning_checkpoint.py \
    --checkpoint "${resume_checkpoint}" --expected-step "${resume_step}" \
    --expected-weight-decay 0.000001 \
    --output output/hhc_original38_token_window_control_resume_checkpoint_audit.json
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
    --batch-size 96 --total-iter 300000 \
    --warm-up-iter 1000 --lr 0.0002 --weight-decay 0.000001 \
    --contact --no-contact-conditioning --shape --paper-architecture \
    --training-mask-contract paper_kinematic \
    --training-loss-contract paper_masked_only \
    --optimizer-contract paper_warmup_cosine \
    --conditioning-contract paper_text_cfg \
    --training-token-length-contract original_mmm_38 \
    --training-validation-contract paper_isolated_common \
    --sampler sample_intra_cas --temperature 0.5 --sampling-steps 50 \
    --cfg-scale 3.5 --contact-guidance-scale 1.0 \
    --wandb-mode online --seed 123 \
    --resume-training-ckpt "${resume_checkpoint}" --resume-in-place
  resolve_state
fi

final_status="$(/home/wenning/miniconda3/envs/New_MMM/bin/python -c \
  "import json; print(json.load(open('${training_state}'))['status'])")"
if [[ "${final_status}" != complete ]]; then
  echo "Original-38 continuation did not reach 300k: ${final_status}"
  exit 2
fi

conda run --no-capture-output -n New_MMM python -u \
  tools/select_best_mgt_checkpoint.py \
  --run-glob "output/t2i_trans/*_${experiment}" \
  --expected-final-step 300000 --output "${selection}"
training_manifest="$(/home/wenning/miniconda3/envs/New_MMM/bin/python -c \
  "import json; print(json.load(open('${selection}'))['training_manifest'])")"
selected_checkpoint="$(/home/wenning/miniconda3/envs/New_MMM/bin/python -c \
  "import json; print(json.load(open('${selection}'))['selected_checkpoint'])")"

conda run --no-capture-output -n New_MMM python -u \
  tools/audit_hhc_paper_training_contract.py \
  --training-manifest "${training_manifest}" \
  --vq-manifest checkpoints/vq_precleanup_raw12288/training_manifest.json \
  --output output/hhc_original38_token_window_control_training_contract_audit.json
conda run --no-capture-output -n New_MMM python -u \
  tools/audit_wandb_online_metric_history.py \
  --manifest "${training_manifest}" \
  --metric-profile masked_token_classification --expected-final-step 300000 \
  --output output/hhc_original38_token_window_control_300k_wandb_history_audit.json

for guidance in no_contact contact; do
  generation_args=(--no-contact-guidance --hhc-contact-timing-source none)
  if [[ "${guidance}" == contact ]]; then
    generation_args=(--contact-guidance --hhc-contact-timing-source request_oracle)
  fi
  conda run --no-capture-output -n New_MMM python -u \
    tools/run_fair_interx_method.py --method hhc \
    --label "hhc_original38_token_window_control_300k_common50_${guidance}" \
    --gpu 1 --checkpoint "${selected_checkpoint}" \
    --vq-checkpoint "${vq_checkpoint}" \
    --hhc-vq-contract precleanup_raw12288 --hhc-mgt-contract current \
    --hhc-stats-dir dataset/interx --hhc-stats-prefix Interx \
    --hhc-contact-map "${cache_root}/code_contact_probs.npy" \
    --hhc-contact-map-contract paper_formula_train_only \
    --selection-manifest "${selection}" \
    --sampling-steps 50 --temperature 0.5 --cfg-scale 3.5 \
    --contact-guidance-scale 1.0 "${generation_args[@]}"
done
