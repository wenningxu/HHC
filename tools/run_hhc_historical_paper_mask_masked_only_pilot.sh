#!/usr/bin/env bash
set -euo pipefail

cd /mnt/8T-HDD/wenning/HHC

experiment="hhc_historical_paper_mask_masked_only_pilot50k_seed114514_v1"
cache_root="checkpoints/vq_precleanup_raw12288_historical_mmm_cache"
vq_checkpoint="checkpoints/vq_precleanup_raw12288/net_175000.pth"

# Training-side MModality diagnosis.  Relative to the current strong HHC
# family, model/VQ/data/mask/optimizer/conditioning/token-window parameters are
# retained and the supervised objective is changed to the paper's masked-token
# NLL.  The validation hook follows the user's current 50-step requirement and
# is observational, so it cannot perturb subsequent training RNG state.
/home/wenning/miniconda3/bin/conda run --no-capture-output -n New_MMM \
  python -u train/train_t2i_trans.py \
  --device cuda:0 --exp-name "${experiment}" \
  --vq-checkpoint "${vq_checkpoint}" \
  --cache-root "${cache_root}" --codebook-dir "${cache_root}/codebook" \
  --token-cache-contract historical_mmm_exact \
  --historical-codebook-source /home/wenning/projects/MMM/checkpoints/vq/codebook \
  --historical-contact-source /home/wenning/projects/MMM/checkpoints/vq/contact_token_label \
  --nb-code 12288 --code-dim 256 --shape-condition-dim 8 --width 512 \
  --embed-dim-gpt 1024 --n-head-gpt 16 --num-layers 9 --num-local-layer 2 \
  --batch-size 64 --total-iter 300000 --diagnostic-stop-iter 50000 \
  --warm-up-iter 1000 --lr 0.0002 --weight-decay 0.000001 \
  --lr-scheduler 150000 --gamma 0.05 \
  --no-contact --no-contact-conditioning --shape --no-paper-architecture \
  --training-mask-contract historical_pipeline_paper_kinematic_ratio \
  --training-loss-contract paper_masked_only \
  --optimizer-contract historical_multistep \
  --conditioning-contract historical_mmm_shape_dropout \
  --training-token-length-contract historical_38 \
  --training-validation-contract paper_isolated_common \
  --save-all-validation-checkpoints --pkeep 0.5 \
  --sampler sample_intra_cas --temperature 0.5 --sampling-steps 50 \
  --cfg-scale 3.5 --contact-guidance-scale 1.0 \
  --mm-num-samples 100 --validation-mm-repeats 3 --mm-num-repeats 30 \
  --wandb-mode online --seed 114514
