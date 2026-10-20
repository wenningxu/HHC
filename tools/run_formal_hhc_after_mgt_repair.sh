#!/usr/bin/env bash
set -euo pipefail

cd /mnt/8T-HDD/wenning/HHC

conda_bin="/home/wenning/miniconda3/bin/conda"
if [[ ! -x "${conda_bin}" ]]; then
  echo "Required Conda executable is missing: ${conda_bin}"
  exit 2
fi
# User services do not inherit the interactive shell's Conda PATH.
conda() {
  "${conda_bin}" "$@"
}

conda run --no-capture-output -n New_MMM python -u \
  tools/require_mgt_repair_gate.py --gate output/mgt_repair_gate.json

conda run --no-capture-output -n New_MMM python -u \
  tools/audit_historical_mgt_tokenizer_runtime.py \
  --output output/historical_mgt_tokenizer_runtime_audit.json

conda run --no-capture-output -n New_MMM python -u \
  tools/audit_hhc_formal_launch_contract.py

conda run --no-capture-output -n New_MMM python -u \
  tools/audit_hhc_model_structure_runtime.py \
  --output output/hhc_model_structure_runtime_audit.json

conda run --no-capture-output -n New_MMM python -u \
  tools/audit_hhc_training_mask_runtime.py \
  --output output/hhc_training_mask_runtime_audit.json

conda run --no-capture-output -n New_MMM python -u \
  tools/audit_hhc_sampling_runtime.py \
  --output output/hhc_sampling_runtime_audit.json

vq_checkpoint="checkpoints/vq_precleanup_raw12288/net_175000.pth"
cache_root="checkpoints/vq_precleanup_raw12288_formal_mgt_v1"
codebook_dir="${cache_root}/codebook"
historical_codebook_source="/home/wenning/projects/MMM/checkpoints/vq/codebook"
historical_contact_source="/home/wenning/projects/MMM/checkpoints/vq/contact_token_label"
experiment="hhc_formal_raw12288_repaired_mgt_300k_seed123_v1"
training_state="output/hhc_formal_raw12288_repaired_mgt_training_state.json"

resolve_training_state() {
  conda run --no-capture-output -n New_MMM python -u \
    tools/resolve_mgt_training_state.py \
    --run-glob "output/t2i_trans/*_${experiment}" \
    --experiment "${experiment}" --expected-final-step 300000 \
    --output "${training_state}"
}

resolve_training_state
training_status="$(/home/wenning/miniconda3/envs/New_MMM/bin/python -c \
  "import json; print(json.load(open('${training_state}'))['status'])")"
resume_args=()
if [[ "${training_status}" != complete ]]; then
  if [[ "${training_status}" == new ]]; then
    conda run --no-capture-output -n New_MMM python -u \
      train/train_t2i_trans.py --cache-only \
      --device cuda:1 --exp-name "${experiment}_cache" \
      --vq-checkpoint "${vq_checkpoint}" \
      --cache-root "${cache_root}" --codebook-dir "${codebook_dir}" \
      --token-cache-contract historical_mmm_exact \
      --historical-codebook-source "${historical_codebook_source}" \
      --historical-contact-source "${historical_contact_source}" \
      --nb-code 12288 --code-dim 256 --shape-condition-dim 8 --width 512 \
      --seed 123 --wandb-mode disabled
  else
    resume_checkpoint="$(/home/wenning/miniconda3/envs/New_MMM/bin/python -c \
      "import json; print(json.load(open('${training_state}'))['resume_checkpoint'])")"
    resume_args=(--resume-training-ckpt "${resume_checkpoint}" --resume-in-place)
  fi

  conda run --no-capture-output -n New_MMM python -u \
    train/train_t2i_trans.py \
    --device cuda:1 --exp-name "${experiment}" \
    --vq-checkpoint "${vq_checkpoint}" \
    --cache-root "${cache_root}" --codebook-dir "${codebook_dir}" \
    --token-cache-contract historical_mmm_exact \
    --historical-codebook-source "${historical_codebook_source}" \
    --historical-contact-source "${historical_contact_source}" \
    --nb-code 12288 --code-dim 256 --shape-condition-dim 8 --width 512 \
    --embed-dim-gpt 512 --n-head-gpt 8 --num-layers 9 --num-local-layer 9 \
    --batch-size 96 --total-iter 300000 --warm-up-iter 1000 \
    --lr 0.0002 --weight-decay 0.000001 \
    --contact --no-contact-conditioning --shape --paper-architecture \
    --training-mask-contract paper_kinematic \
    --training-loss-contract paper_masked_only \
    --optimizer-contract paper_warmup_cosine \
    --conditioning-contract paper_text_cfg \
    --training-token-length-contract paper_50 \
    --training-validation-contract paper_isolated_common \
    --sampler sample_intra_cas --temperature 0.5 --sampling-steps 50 \
    --cfg-scale 3.5 --contact-guidance-scale 1.0 \
    --wandb-mode online --seed 123 "${resume_args[@]}"

  resolve_training_state
fi

selection="output/hhc_formal_raw12288_repaired_mgt_selection.json"
conda run --no-capture-output -n New_MMM python -u \
  tools/select_best_mgt_checkpoint.py \
  --run-glob "output/t2i_trans/*_${experiment}" \
  --expected-final-step 300000 --output "${selection}"

selected_checkpoint="$(/home/wenning/miniconda3/envs/New_MMM/bin/python -c \
  "import json; print(json.load(open('${selection}'))['selected_checkpoint'])")"
selected_training_manifest="$(/home/wenning/miniconda3/envs/New_MMM/bin/python -c \
  "import json; print(json.load(open('${selection}'))['training_manifest'])")"
run_dir="$(dirname "${selected_training_manifest}")"
launch_source_snapshot_manifest="${run_dir}/launch_source_snapshot_manifest.json"

conda run --no-capture-output -n New_MMM python -u \
  tools/freeze_mgt_training_sources.py \
  --training-manifest "${selected_training_manifest}" \
  --output-dir "${run_dir}/launch_sources" \
  --output-manifest "${launch_source_snapshot_manifest}"

conda run --no-capture-output -n New_MMM python -u \
  tools/audit_hhc_formal_launch_contract.py \
  --source-snapshot-manifest "${launch_source_snapshot_manifest}"

validation_before="${run_dir}/prevalidation-000020000.ckpt"
validation_after="${run_dir}/postvalidation-000020000.ckpt"
if [[ ! -s "${validation_before}" || ! -s "${validation_after}" ]]; then
  echo 'Formal HHC 20k pre/post-validation checkpoint pair is incomplete.'
  exit 2
fi
conda run --no-capture-output -n New_MMM python -u \
  tools/audit_lightning_checkpoint_pair.py \
  --before "${validation_before}" --after "${validation_after}" \
  --expected-step 20000 --require-fixed-pair-marker \
  --output output/hhc_formal_raw12288_repaired_mgt_20k_pre_post_validation_pair_audit.json

conda run --no-capture-output -n New_MMM python -u \
  tools/audit_hhc_paper_training_contract.py \
  --training-manifest "${selected_training_manifest}" \
  --source-snapshot-manifest "${launch_source_snapshot_manifest}" \
  --vq-manifest checkpoints/vq_precleanup_raw12288/training_manifest.json \
  --output output/hhc_formal_raw12288_repaired_mgt_training_contract_audit.json

for guidance in no_contact contact; do
  if [[ "${guidance}" == contact ]]; then
    timing_args=(--hhc-contact-timing-source request_oracle --contact-guidance)
  else
    timing_args=(--hhc-contact-timing-source none --no-contact-guidance)
  fi
  conda run --no-capture-output -n New_MMM python -u \
    tools/run_fair_interx_method.py --method hhc \
    --label "hhc_formal_raw12288_repaired_mgt_common50_${guidance}" --gpu 1 \
    --checkpoint "${selected_checkpoint}" --vq-checkpoint "${vq_checkpoint}" \
    --hhc-vq-contract precleanup_raw12288 --hhc-mgt-contract current \
    --hhc-stats-dir dataset/interx --hhc-stats-prefix Interx \
    --hhc-contact-map "${cache_root}/code_contact_probs.npy" \
    --hhc-contact-map-contract paper_formula_train_only \
    --selection-manifest "${selection}" \
    --sampling-steps 50 --temperature 0.5 --cfg-scale 3.5 \
    --contact-guidance-scale 1.0 "${timing_args[@]}"
done
