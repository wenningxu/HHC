#!/usr/bin/env bash
set -euo pipefail

cd /mnt/8T-HDD/wenning/HHC

if [[ $# -ne 1 || ! "$1" =~ ^[0-9]+$ ]]; then
  echo "usage: $0 GPU_INDEX" >&2
  exit 2
fi
gpu="$1"
used_mib="$(nvidia-smi --id="${gpu}" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')"
if [[ ! "${used_mib}" =~ ^[0-9]+$ ]] || (( used_mib > 1024 )); then
  echo "GPU ${gpu} is not free (memory.used=${used_mib:-unknown} MiB)." >&2
  exit 2
fi

overlay="dataset/interx_shape_aug_optimized_selective_seed123_v3"
plan="configs/interx_shape_augmentation/timing_covered_seed123_v2/plan_manifest.json"
augmentation_manifest="${overlay}/materialization_manifest.json"
experiment="hhc_historical_paper_mask_no_mgt_shape_optimizedaug_pilot50k_seed114514_v1"
cache_root="checkpoints/vq_precleanup_raw12288_no_shape_optimizedaug_cache"
vq_checkpoint="checkpoints/vq_precleanup_raw12288/net_175000.pth"
selected0="output/interx_shape_aug_volumetric_fast_selective_full_shard0"
selected1="output/interx_shape_aug_volumetric_fast_selective_full_shard1"

for selected in "${selected0}" "${selected1}"; do
  test -s "${selected}/_completion_manifest.json"
  /home/wenning/miniconda3/envs/New_MMM/bin/python -c \
    "import json; p=json.load(open('${selected}/_completion_manifest.json')); assert p['complete'] is True; assert p['optimizer']['postprocessing']=='selective_constant_delta_projection_with_random_shape_fallback'"
done

if [[ ! -s "${augmentation_manifest}" ]]; then
  /home/wenning/miniconda3/bin/conda run --no-capture-output -n New_MMM \
    python -u tools/materialize_interx_shape_augmentation.py \
    --optimization-dir "${selected0}" --optimization-dir "${selected1}" \
    --plan "${plan}" --output-root "${overlay}"
fi

# Cache creation is deterministic VQ inference. Original samples are imported
# byte-identically from the historical MMM cache; only optimized augmented
# motions are encoded with the recovered historical root/velocity statistics.
/home/wenning/miniconda3/bin/conda run --no-capture-output -n New_MMM \
  python -u train/train_t2i_trans.py --cache-only \
  --device "cuda:${gpu}" --exp-name "${experiment}_cache" \
  --vq-checkpoint "${vq_checkpoint}" \
  --cache-root "${cache_root}" --codebook-dir "${cache_root}/codebook" \
  --token-cache-contract hybrid_historical_original_plus_deterministic_augmented_v1 \
  --historical-codebook-source /home/wenning/projects/MMM/checkpoints/vq/codebook \
  --historical-contact-source /home/wenning/projects/MMM/checkpoints/vq/contact_token_label \
  --nb-code 12288 --code-dim 256 --shape-condition-dim 8 --width 512 \
  --embed-dim-gpt 1024 --n-head-gpt 16 --num-layers 9 --num-local-layer 2 \
  --train-asset-root "${overlay}" --augmentation-manifest "${augmentation_manifest}" \
  --train-split-file "${overlay}/splits/train.txt" \
  --val-split-file dataset/interx/splits/val.txt \
  --all-split-file "${overlay}/splits/all.txt" \
  --train-motion-file "${overlay}/motions/train.h5" \
  --val-motion-file dataset/interx/motions/val.h5 \
  --all-motion-file "${overlay}/motions/all.h5" \
  --stats-dir "${overlay}" --stats-prefix Interx \
  --validation-mm-repeats 11 \
  --seed 114514 --wandb-mode disabled

# Paired with the no-augmentation v4 run. The only model-training difference is
# the optimized augmentation overlay/cache; MGT has no beta/gender condition.
/home/wenning/miniconda3/bin/conda run --no-capture-output -n New_MMM \
  python -u train/train_t2i_trans.py \
  --device "cuda:${gpu}" --exp-name "${experiment}" \
  --vq-checkpoint "${vq_checkpoint}" \
  --cache-root "${cache_root}" --codebook-dir "${cache_root}/codebook" \
  --token-cache-contract hybrid_historical_original_plus_deterministic_augmented_v1 \
  --historical-codebook-source /home/wenning/projects/MMM/checkpoints/vq/codebook \
  --historical-contact-source /home/wenning/projects/MMM/checkpoints/vq/contact_token_label \
  --nb-code 12288 --code-dim 256 --shape-condition-dim 8 --width 512 \
  --embed-dim-gpt 1024 --n-head-gpt 16 --num-layers 9 --num-local-layer 2 \
  --train-asset-root "${overlay}" --augmentation-manifest "${augmentation_manifest}" \
  --train-split-file "${overlay}/splits/train.txt" \
  --val-split-file dataset/interx/splits/val.txt \
  --all-split-file "${overlay}/splits/all.txt" \
  --train-motion-file "${overlay}/motions/train.h5" \
  --val-motion-file dataset/interx/motions/val.h5 \
  --all-motion-file "${overlay}/motions/all.h5" \
  --stats-dir "${overlay}" --stats-prefix Interx \
  --batch-size 64 --total-iter 300000 --diagnostic-stop-iter 50000 \
  --warm-up-iter 1000 --lr 0.0002 --weight-decay 0.000001 \
  --lr-scheduler 150000 --gamma 0.05 \
  --no-contact --no-contact-conditioning --no-shape --no-paper-architecture \
  --training-mask-contract historical_pipeline_paper_kinematic_ratio \
  --training-loss-contract historical_all_valid \
  --optimizer-contract historical_multistep \
  --conditioning-contract historical_mmm_shape_dropout \
  --training-token-length-contract historical_38 \
  --training-validation-contract paper_isolated_common \
  --save-all-validation-checkpoints --pkeep 0.5 \
  --sampler sample_intra_cas --temperature 0.5 --sampling-steps 50 \
  --cfg-scale 3.5 --contact-guidance-scale 1.0 \
  --mm-num-samples 100 --validation-mm-repeats 11 --mm-num-repeats 30 \
  --wandb-mode online --seed 114514
