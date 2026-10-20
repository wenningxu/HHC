#!/usr/bin/env bash
set -euo pipefail

cd /mnt/8T-HDD/wenning/HHC

# Controlled MGT rerun with the original "8192" VQ line.  The serialized
# checkpoint contains 8096 learnable entries per part (24+24+16 latent dims),
# so all model/cache arguments use the checkpoint's exact tensor contract.
vq_checkpoint="checkpoints/vq_original_8192/net_best_fid.pth"
codebook_dir="checkpoints/vq_original_8192/codebook"
experiment="paper_mgt_original_8192vq_300k_seed123"

conda run --no-capture-output -n New_MMM python -u \
  train/train_t2i_trans.py \
  --device cuda:1 --exp-name "${experiment}" \
  --vq-checkpoint "${vq_checkpoint}" \
  --codebook-dir "${codebook_dir}" \
  --nb-code 8096 --code-dim 64 --shape-condition-dim 8 --width 512 \
  --batch-size 96 --total-iter 300000 --warm-up-iter 1000 \
  --lr 0.0002 --weight-decay 0.000001 \
  --contact --no-contact-conditioning --shape --paper-architecture \
  --sampler sample_intra_cas --temperature 0.5 --sampling-steps 50 \
  --cfg-scale 3.5 --contact-guidance-scale 1.0 \
  --wandb-mode online --seed 123

conda run --no-capture-output -n New_MMM python -u \
  tools/select_best_mgt_checkpoint.py \
  --run-glob "output/t2i_trans/*_${experiment}" \
  --expected-final-step 300000 \
  --output output/hhc_mgt_original_8192vq_selection.json

selected_checkpoint="$(/home/wenning/miniconda3/envs/New_MMM/bin/python -c \
  "import json; print(json.load(open('output/hhc_mgt_original_8192vq_selection.json'))['selected_checkpoint'])")"

conda run --no-capture-output -n New_MMM python -u \
  tools/run_fair_interx_method.py --method hhc \
  --label hhc_original_8192vq_common50_contact_guidance --gpu 1 \
  --checkpoint "${selected_checkpoint}" \
  --vq-checkpoint "${vq_checkpoint}" \
  --hhc-vq-contract original_8192 \
  --selection-manifest output/hhc_mgt_original_8192vq_selection.json \
  --sampling-steps 50 --contact-guidance

conda run --no-capture-output -n New_MMM python -u \
  tools/run_fair_interx_method.py --method hhc \
  --label hhc_original_8192vq_common50_no_contact_guidance --gpu 1 \
  --checkpoint "${selected_checkpoint}" \
  --vq-checkpoint "${vq_checkpoint}" \
  --hhc-vq-contract original_8192 \
  --selection-manifest output/hhc_mgt_original_8192vq_selection.json \
  --sampling-steps 50 --hhc-contact-timing-source none --no-contact-guidance
