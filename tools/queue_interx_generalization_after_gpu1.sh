#!/usr/bin/env bash
set -euo pipefail

wait_pid="${1:-623106}"
if [[ "${wait_pid}" != none ]]; then
  while ps -p "${wait_pid}" >/dev/null 2>&1; do
    sleep 30
  done
fi

run_generalization() {
  local experiment="$1"
  local stats_dir="configs/interx_generalization/${experiment}/stats"
  local stats_prefix="generalization_${experiment}"
  local vq_dir="checkpoints/vq_generalization_${experiment}"
  local vq_checkpoint="${vq_dir}/net_150000.pth"
  local mgt_name="paper_mgt_generalization_${experiment}_300k_seed123"
  local selection="output/hhc_mgt_generalization_${experiment}_selection.json"

  conda run --no-capture-output -n New_MMM python -u \
    train/train_vq_interx.py \
    --device cuda:1 --batch-size 256 --total-iter 150000 \
    --warm-up-iter 1000 --lr 0.0002 --weight-decay 0.000001 \
    --nb-code 8096 --code-dim 256 --down-t 2 --stride-t 2 \
    --train-split-file "configs/interx_generalization/${experiment}/train_seen.txt" \
    --train-motion-file dataset/interx/motions/train.h5 \
    --stats-dir "${stats_dir}" --stats-prefix "${stats_prefix}" \
    --exp-name "paper_vq_generalization_${experiment}_150k_seed123" \
    --checkpoint-dir "${vq_dir}" --wandb-mode online --precision fp32 \
    --print-iter 200 --eval-iter 5000 --seed 123

  conda run --no-capture-output -n New_MMM python -u \
    tools/audit_vq_checkpoint.py --checkpoint "${vq_checkpoint}" \
    --output "output/vq_generalization_${experiment}_state_audit.json" \
    --expected-iteration 150000

  conda run --no-capture-output -n New_MMM python -u \
    train/train_t2i_trans.py \
    --device cuda:1 --exp-name "${mgt_name}" \
    --vq-checkpoint "${vq_checkpoint}" --codebook-dir "${vq_dir}/codebook" \
    --train-split-file "configs/interx_generalization/${experiment}/train_seen.txt" \
    --val-split-file "configs/interx_generalization/${experiment}/val_seen.txt" \
    --train-motion-file dataset/interx/motions/train.h5 \
    --val-motion-file dataset/interx/motions/val.h5 \
    --stats-dir "${stats_dir}" --stats-prefix "${stats_prefix}" \
    --batch-size 96 --total-iter 300000 --warm-up-iter 1000 \
    --lr 0.0002 --weight-decay 0.000001 \
    --contact --no-contact-conditioning --shape --paper-architecture \
    --sampler sample_intra_cas --temperature 0.5 --sampling-steps 50 \
    --cfg-scale 3.5 --contact-guidance-scale 1.0 \
    --wandb-mode online --seed 123

  conda run --no-capture-output -n New_MMM python -u \
    tools/select_best_mgt_checkpoint.py \
    --run-glob "output/t2i_trans/*_${mgt_name}" \
    --expected-final-step 300000 --output "${selection}"

  local selected_checkpoint
  selected_checkpoint="$(/home/wenning/miniconda3/envs/New_MMM/bin/python -c \
    "import json; print(json.load(open('${selection}'))['selected_checkpoint'])")"

  for partition in test_unseen test_seen; do
    conda run --no-capture-output -n New_MMM python -u \
      tools/run_fair_interx_method.py --method hhc \
      --label "hhc_generalization_${experiment}_${partition}" \
      --output-root output/fair_eval_generalization \
      --requests-dir "output/fair_interx_generalization_requests/${experiment}/${partition}" \
      --requests-manifest output/fair_interx_generalization_requests/manifest.json \
      --checkpoint "${selected_checkpoint}" --vq-checkpoint "${vq_checkpoint}" \
      --selection-manifest "${selection}" \
      --hhc-stats-dir "${stats_dir}" --hhc-stats-prefix "${stats_prefix}" \
      --gpu 1 --batch-size 32 --sampling-steps 50 --contact-guidance
  done
}

run_generalization shape
run_generalization action
