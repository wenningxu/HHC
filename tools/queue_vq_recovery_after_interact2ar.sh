#!/usr/bin/env bash
set -euo pipefail

cd /mnt/8T-HDD/wenning/HHC

vq_dir="checkpoints/vq_recovery_8096_w64_shapemask01"
vq_best="${vq_dir}/net_best_validation_fid.pth"
mgt_name="mgt_recovery_vq8096_w64_shapemask01_300k_seed123"

# Recovery run: retain the paper architecture/codebook, but restore the
# result-producing training settings identified from the historical run.
conda run --no-capture-output -n New_MMM python -u \
  train/train_vq_interx.py \
  --device cuda:1 \
  --exp-name vq_recovery_8096_w64_shapemask01_300k_seed123 \
  --checkpoint-dir "${vq_dir}" \
  --batch-size 256 --window-size 64 \
  --nb-code 8096 --code-dim 256 \
  --total-iter 300000 --warm-up-iter 1000 \
  --schedule-mode historical_separate_warmup_cosine \
  --lr 0.0002 --weight-decay 0.00002 \
  --shape-mask-prob 0.1 \
  --eval-iter 5000 --print-iter 200 \
  --precision fp32 --wandb-mode online --seed 123

test -s "${vq_best}"

# Frozen test reconstruction is reporting-only.  Checkpoint selection above is
# based solely on the held-out validation split.
conda run --no-capture-output -n New_MMM python -u \
  tools/eval_vq_interx_reconstruction.py \
  --checkpoint "${vq_best}" \
  --output output/vq_recovery_8096_eval/best_validation_reconstruction.json \
  --reconstruction-h5 output/vq_recovery_8096_eval/best_validation_reconstructions.h5 \
  --device cuda:1 --batch-size 16

conda run --no-capture-output -n New_MMM python -u \
  tools/score_fair_interx_main.py \
  --requests output/fair_interx_requests/replication_00.npz \
  --main-generations output/vq_recovery_8096_eval/best_validation_reconstructions.h5 \
  --output output/vq_recovery_8096_eval/best_validation_reconstruction_fid.json \
  --device cuda:1 --batch-size 32

conda run --no-capture-output -n New_MMM python -u \
  train/train_t2i_trans.py \
  --device cuda:1 --exp-name "${mgt_name}" \
  --vq-checkpoint "${vq_best}" \
  --codebook-dir "${vq_dir}/codebook" \
  --batch-size 96 --total-iter 300000 --warm-up-iter 1000 \
  --lr 0.0002 --weight-decay 0.000001 \
  --contact --no-contact-conditioning --shape --paper-architecture \
  --sampler sample_intra_cas --temperature 0.5 --sampling-steps 50 \
  --cfg-scale 3.5 --contact-guidance-scale 1.0 \
  --wandb-mode online --seed 123

conda run --no-capture-output -n New_MMM python -u \
  tools/select_best_mgt_checkpoint.py \
  --run-glob "output/t2i_trans/*_${mgt_name}" \
  --expected-final-step 300000 \
  --output output/hhc_mgt_recovery_selection.json

selected_checkpoint="$(/home/wenning/miniconda3/envs/New_MMM/bin/python -c \
  "import json; print(json.load(open('output/hhc_mgt_recovery_selection.json'))['selected_checkpoint'])")"

conda run --no-capture-output -n New_MMM python -u \
  tools/run_fair_interx_method.py --method hhc \
  --label hhc_recovery8096_common50_contact_guidance --gpu 1 \
  --checkpoint "${selected_checkpoint}" \
  --vq-checkpoint "${vq_best}" \
  --selection-manifest output/hhc_mgt_recovery_selection.json \
  --sampling-steps 50 --contact-guidance

conda run --no-capture-output -n New_MMM python -u \
  tools/run_fair_interx_method.py --method hhc \
  --label hhc_recovery8096_common50_no_contact_guidance --gpu 1 \
  --checkpoint "${selected_checkpoint}" \
  --vq-checkpoint "${vq_best}" \
  --selection-manifest output/hhc_mgt_recovery_selection.json \
  --sampling-steps 50 --hhc-contact-timing-source none --no-contact-guidance
