#!/usr/bin/env bash
set -euo pipefail

cd /mnt/8T-HDD/wenning/HHC

if (( $# < 1 || $# > 2 )) || [[ ! "$1" =~ ^[0-9]+$ ]]; then
  echo "usage: $0 GPU_INDEX [random|optimized]" >&2
  exit 2
fi
gpu="$1"
arm="${2:-random}"
if [[ "${arm}" != random && "${arm}" != optimized ]]; then
  echo "arm must be random or optimized" >&2
  exit 2
fi
device="cuda:${gpu}"

if [[ -n "${PYTORCH_CUDA_ALLOC_CONF:-}" ]] \
    && [[ "${PYTORCH_CUDA_ALLOC_CONF}" != "expandable_segments:True" ]]; then
  echo "Unexpected PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF}" >&2
  exit 2
fi
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

conda() {
  /home/wenning/miniconda3/bin/conda "$@"
}

# This launcher deliberately has no waiter loop.  Invoke it only after the
# selected GPU is free; a busy GPU is a hard failure rather than another
# hidden background terminal.
used_mib="$(nvidia-smi --id="${gpu}" \
  --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')"
if [[ ! "${used_mib}" =~ ^[0-9]+$ ]] || (( used_mib > 1024 )); then
  echo "GPU ${gpu} is not free (memory.used=${used_mib:-unknown} MiB)." >&2
  exit 2
fi

launcher="tools/run_shape_random_mgt_when_gpu_free.sh"
plan="configs/interx_shape_augmentation/timing_covered_seed123_v2/plan_manifest.json"
vq_checkpoint="checkpoints/vq_precleanup_raw12288/net_175000.pth"
cache_root="checkpoints/vq_precleanup_raw12288_shapeaug_${arm}_cache"
experiment="hhc_formal_raw12288_repaired_mgt_shapeaug_${arm}_300k_seed123_v1"
training_state="output/hhc_formal_raw12288_repaired_mgt_shapeaug_${arm}_training_state.json"
selection="output/hhc_formal_raw12288_repaired_mgt_shapeaug_${arm}_selection.json"
if [[ "${arm}" == random ]]; then
  overlay="dataset/interx_shape_aug_random_seed123_v2"
  augmentation="output/interx_shape_aug_random_seed123_v2"
else
  overlay="dataset/interx_shape_aug_optimized_flathand_seed123_v2"
  aggregate_dir="output/interx_shape_aug_full_batch8_query10475_seed20260828_aggregate"
  aggregate_manifest="${aggregate_dir}/_completion_manifest.json"
  shard_args=()
  for shard in {0..7}; do
    shard_args+=(
      --shard-dir
      "output/interx_shape_aug_full_batch8_query10475_seed20260828_shard${shard}"
    )
  done
  if [[ ! -s "${aggregate_manifest}" ]]; then
    conda run --no-capture-output -n New_MMM python -u \
      tools/aggregate_interx_shape_augmentation_shards.py \
      "${shard_args[@]}" --expected-samples 6238 --expected-shards 8 \
      --output "${aggregate_manifest}"
  fi
  if [[ ! -s "${overlay}/materialization_manifest.json" ]]; then
    materialize_args=()
    for shard in {0..7}; do
      materialize_args+=(
        --optimization-dir
        "output/interx_shape_aug_full_batch8_query10475_seed20260828_shard${shard}"
      )
    done
    conda run --no-capture-output -n New_MMM python -u \
      tools/materialize_interx_shape_augmentation.py \
      "${materialize_args[@]}" --plan "${plan}" --output-root "${overlay}"
  fi
fi

conda run --no-capture-output -n New_MMM python -u \
  tools/require_mgt_repair_gate.py --gate output/mgt_repair_gate.json
conda run --no-capture-output -n New_MMM python -u \
  tools/require_hhc_formal_mgt_quality_gate.py \
  --gate output/hhc_formal_mgt_quality_gate.json
conda run --no-capture-output -n New_MMM python -u \
  tools/audit_wandb_online_runtime.py \
  --output "output/interx_shape_${arm}_mgt_wandb_runtime_audit.json"
conda run --no-capture-output -n New_MMM python -u \
  tools/audit_online_wandb_training_contract.py \
  --launcher "${launcher}" --training-source train/train_t2i_trans.py \
  --metric-profile masked_token_classification \
  --output "output/interx_shape_${arm}_mgt_online_wandb_audit.json"
if [[ "${arm}" == random ]]; then
  conda run --no-capture-output -n New_MMM python -u \
    tools/audit_random_shape_augmentation_completion.py \
    --run-dir "${augmentation}" --plan "${plan}" --expected-samples 6238 \
    --output "${augmentation}/completion_content_audit.json"
fi

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
resume_args=()

if [[ "${training_status}" != complete ]]; then
  if [[ "${training_status}" == new ]]; then
    conda run --no-capture-output -n New_MMM python -u \
      train/train_t2i_trans.py --cache-only \
      --device "${device}" --exp-name "${experiment}_cache" \
      --vq-checkpoint "${vq_checkpoint}" \
      --cache-root "${cache_root}" --codebook-dir "${cache_root}/codebook" \
      --token-cache-contract deterministic_historical_root_v1 \
      --nb-code 12288 --code-dim 256 --shape-condition-dim 8 --width 512 \
      --embed-dim-gpt 512 --n-head-gpt 8 --num-layers 9 --num-local-layer 9 \
      --train-asset-root "${overlay}" \
      --augmentation-manifest "${overlay}/materialization_manifest.json" \
      --train-split-file "${overlay}/splits/train.txt" \
      --val-split-file dataset/interx/splits/val.txt \
      --all-split-file "${overlay}/splits/all.txt" \
      --train-motion-file "${overlay}/motions/train.h5" \
      --val-motion-file dataset/interx/motions/val.h5 \
      --all-motion-file "${overlay}/motions/all.h5" \
      --stats-dir "${overlay}" --stats-prefix Interx \
      --seed 123 --wandb-mode disabled
  else
    resume_checkpoint="$(/home/wenning/miniconda3/envs/New_MMM/bin/python -c \
      "import json; print(json.load(open('${training_state}'))['resume_checkpoint'])")"
    resume_args=(--resume-training-ckpt "${resume_checkpoint}" --resume-in-place)
  fi

  # Every model-training event is online W&B.  The cache-only command above
  # performs deterministic VQ inference and is not a training run.
  conda run --no-capture-output -n New_MMM python -u \
    train/train_t2i_trans.py \
    --device "${device}" --exp-name "${experiment}" \
    --vq-checkpoint "${vq_checkpoint}" \
    --cache-root "${cache_root}" --codebook-dir "${cache_root}/codebook" \
    --token-cache-contract deterministic_historical_root_v1 \
    --nb-code 12288 --code-dim 256 --shape-condition-dim 8 --width 512 \
    --embed-dim-gpt 512 --n-head-gpt 8 --num-layers 9 --num-local-layer 9 \
    --train-asset-root "${overlay}" \
    --augmentation-manifest "${overlay}/materialization_manifest.json" \
    --train-split-file "${overlay}/splits/train.txt" \
    --val-split-file dataset/interx/splits/val.txt \
    --all-split-file "${overlay}/splits/all.txt" \
    --train-motion-file "${overlay}/motions/train.h5" \
    --val-motion-file dataset/interx/motions/val.h5 \
    --all-motion-file "${overlay}/motions/all.h5" \
    --stats-dir "${overlay}" --stats-prefix Interx \
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
fi

resolve_state
training_status="$(/home/wenning/miniconda3/envs/New_MMM/bin/python -c \
  "import json; print(json.load(open('${training_state}'))['status'])")"
if [[ "${training_status}" != complete ]]; then
  echo "Random-shape MGT returned without reaching optimizer step 300000." >&2
  exit 2
fi

conda run --no-capture-output -n New_MMM python -u \
  tools/select_best_mgt_checkpoint.py \
  --run-glob "output/t2i_trans/*_${experiment}" \
  --expected-final-step 300000 --output "${selection}"

training_manifest="$(/home/wenning/miniconda3/envs/New_MMM/bin/python -c \
  "import json; print(json.load(open('${selection}'))['training_manifest'])")"
conda run --no-capture-output -n New_MMM python -u \
  tools/audit_hhc_paper_training_contract.py \
  --training-manifest "${training_manifest}" \
  --vq-manifest checkpoints/vq_precleanup_raw12288/training_manifest.json \
  --output "output/hhc_formal_raw12288_repaired_mgt_shapeaug_${arm}_training_contract_audit.json"
conda run --no-capture-output -n New_MMM python -u \
  tools/audit_wandb_online_metric_history.py \
  --manifest "${training_manifest}" \
  --metric-profile masked_token_classification --expected-final-step 300000 \
  --output "output/hhc_formal_raw12288_repaired_mgt_shapeaug_${arm}_wandb_history_audit.json"
