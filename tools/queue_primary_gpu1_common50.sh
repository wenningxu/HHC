#!/usr/bin/env bash
set -euo pipefail

cd /mnt/8T-HDD/wenning/HHC
gpu0_branch_pid="${1:-}"

# The active paper VQ run owns GPU 1.  Wait for both its process and atomic
# final checkpoint before auditing and building the final token cache.
while pgrep -f 'python -u train/train_vq_interx.py.*paper_vq_8096_150k_seed123_exact_rawstats_ema_safe' >/dev/null 2>&1; do
  sleep 30
done
while [[ ! -s checkpoints/vq_paper/net_150000.pth ]]; do
  sleep 30
done

conda run --no-capture-output -n New_MMM python -u \
  tools/audit_vq_checkpoint.py \
  --checkpoint checkpoints/vq_paper/net_iter_000125000.pth \
  --output output/vq_paper_eval/net_iter_000125000_state_audit.json \
  --expected-iteration 125000

conda run --no-capture-output -n New_MMM python -u \
  tools/audit_vq_checkpoint.py \
  --checkpoint checkpoints/vq_paper/net_150000.pth \
  --output output/vq_paper_eval/net_150000_state_audit.json \
  --expected-iteration 150000

conda run --no-capture-output -n New_MMM python -u \
  tools/eval_vq_interx_reconstruction.py \
  --checkpoint checkpoints/vq_paper/net_150000.pth \
  --output output/vq_paper_eval/net_150000_reconstruction.json \
  --device cuda:1 --batch-size 32

conda run --no-capture-output -n New_MMM python -u \
  train/train_t2i_trans.py \
  --device cuda:1 --exp-name paper_mgt_8096_300k_seed123 \
  --vq-checkpoint checkpoints/vq_paper/net_150000.pth \
  --codebook-dir checkpoints/vq_paper/codebook \
  --batch-size 96 --total-iter 300000 --warm-up-iter 1000 \
  --lr 0.0002 --weight-decay 0.000001 \
  --contact --no-contact-conditioning --shape --paper-architecture \
  --sampler sample_intra_cas --temperature 0.5 --sampling-steps 50 \
  --cfg-scale 3.5 --contact-guidance-scale 1.0 \
  --wandb-mode online --seed 123

conda run --no-capture-output -n New_MMM python -u \
  tools/select_best_mgt_checkpoint.py \
  --run-glob 'output/t2i_trans/*_paper_mgt_8096_300k_seed123' \
  --expected-final-step 300000 \
  --output output/hhc_mgt_final_selection.json

selected_checkpoint="$(/home/wenning/miniconda3/envs/New_MMM/bin/python -c \
  "import json; print(json.load(open('output/hhc_mgt_final_selection.json'))['selected_checkpoint'])")"

conda run --no-capture-output -n New_MMM python -u \
  tools/run_fair_interx_method.py --method hhc \
  --label hhc_common50_contact_guidance --gpu 1 \
  --checkpoint "${selected_checkpoint}" \
  --vq-checkpoint checkpoints/vq_paper/net_150000.pth \
  --selection-manifest output/hhc_mgt_final_selection.json \
  --sampling-steps 50 --contact-guidance

conda run --no-capture-output -n New_MMM python -u \
  tools/run_fair_interx_method.py --method hhc \
  --label hhc_common50_no_contact_guidance --gpu 1 \
  --checkpoint "${selected_checkpoint}" \
  --vq-checkpoint checkpoints/vq_paper/net_150000.pth \
  --selection-manifest output/hhc_mgt_final_selection.json \
  --sampling-steps 50 --hhc-contact-timing-source none --no-contact-guidance

conda run --no-capture-output -n New_MMM python -u \
  tools/run_fair_interx_method.py --method intermask \
  --label intermask_official_exact_common50 --gpu 1 \
  --sampling-steps 50

# Keep the whole GPU-1 dependency chain in this one background terminal.
bash tools/queue_interact2ar_after_primary.sh none
# VQ recovery and the older 8096 rerun are retained only as investigation
# backups.  Reproduce the actual pre-cleanup raw-VQ/MGT line first.
bash tools/queue_precleanup_raw_vq_mgt_after_interact2ar.sh
# Reuse this dependency chain for all remaining contact annotations.  The
# launcher is idempotent, so an earlier manual completion only revalidates the
# frozen caches; this avoids creating another long-lived waiter terminal.
bash tools/run_remaining_interx_codex_annotations.sh
bash tools/queue_post_hhc_gpu1_experiments.sh none
bash tools/queue_hhc_timing_after_post_gpu1.sh none
# The earlier held-out shape/action split proposal was rejected.  Keep its
# launcher only as an investigation backup; the active review experiment uses
# training-time shape augmentation on the unchanged official Inter-X split.

# The report needs both independent GPU branches, but no extra waiter terminal.
if [[ -n "${gpu0_branch_pid}" ]]; then
  while ps -p "${gpu0_branch_pid}" >/dev/null 2>&1; do
    sleep 30
  done
fi
bash tools/queue_review_report_after_all_training.sh none none
