#!/usr/bin/env bash
set -euo pipefail

cd /mnt/8T-HDD/wenning/HHC

experiment="hhc_historical_plus_paper_mask_ratio_raw12288_300k_seed114514_v2"
cache_root="checkpoints/vq_precleanup_raw12288_historical_mmm_cache"
vq_checkpoint="checkpoints/vq_precleanup_raw12288/net_175000.pth"

# This is the proven historical 0.1357-FID training contract.  The sole
# numerical training change is training-mask-contract.  Its in-training
# historical test hook is retained to preserve the original RNG trajectory;
# model selection and final reporting are performed later on official val/test
# with sample_intra_cas, 50 outer steps, temperature 0.5, CFG 3.5, contact 1.
/home/wenning/miniconda3/bin/conda run --no-capture-output -n New_MMM \
  python -u train/train_t2i_trans.py \
  --device cuda:1 --exp-name "${experiment}" \
  --vq-checkpoint "${vq_checkpoint}" \
  --cache-root "${cache_root}" --codebook-dir "${cache_root}/codebook" \
  --token-cache-contract historical_mmm_exact \
  --historical-codebook-source /home/wenning/projects/MMM/checkpoints/vq/codebook \
  --historical-contact-source /home/wenning/projects/MMM/checkpoints/vq/contact_token_label \
  --nb-code 12288 --code-dim 256 --shape-condition-dim 8 --width 512 \
  --embed-dim-gpt 1024 --n-head-gpt 16 --num-layers 9 --num-local-layer 2 \
  --batch-size 64 --total-iter 300000 --warm-up-iter 1000 \
  --lr 0.0002 --weight-decay 0.000001 --lr-scheduler 150000 --gamma 0.05 \
  --no-contact --no-contact-conditioning --shape --no-paper-architecture \
  --training-mask-contract historical_pipeline_paper_kinematic_ratio \
  --training-loss-contract historical_all_valid \
  --optimizer-contract historical_multistep \
  --conditioning-contract historical_mmm_shape_dropout \
  --training-token-length-contract historical_38 \
  --training-validation-contract historical_test_side_effect \
  --save-all-validation-checkpoints \
  --pkeep 0.5 \
  --sampler sample_intra_cas --temperature 0.0 --sampling-steps 10 \
  --cfg-scale 3.5 --contact-guidance-scale 1.0 \
  --wandb-mode online --seed 114514
