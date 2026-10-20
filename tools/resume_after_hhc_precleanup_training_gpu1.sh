#!/usr/bin/env bash
set -euo pipefail

echo 'Retired diagnostic launcher: its selected MGT has ordinary FID 0.645 and is excluded from all active training/evaluation queues.'
echo 'Use tools/run_formal_hhc_after_mgt_repair.sh after output/mgt_repair_gate.json passes.'
exit 2

# Resume the single GPU-1 dependency chain after the completed raw-12288 HHC
# MGT run.  This entrypoint never launches MGT training; it first revalidates
# the completed terminal checkpoint and fixed-validation selection, then runs
# only the downstream evaluations and review experiments.

cd /mnt/8T-HDD/wenning/HHC

gpu0_branch_pid="${1:-763028}"
experiment="paper_mgt_precleanup_raw12288_300k_seed123"
vq_checkpoint="checkpoints/vq_precleanup_raw12288/net_175000.pth"

conda run --no-capture-output -n New_MMM python -u \
  tools/select_best_mgt_checkpoint.py \
  --run-glob "output/t2i_trans/*_${experiment}" \
  --expected-final-step 300000 \
  --output output/hhc_mgt_precleanup_raw12288_selection.json

selected_checkpoint="$(/home/wenning/miniconda3/envs/New_MMM/bin/python -c \
  "import json; print(json.load(open('output/hhc_mgt_precleanup_raw12288_selection.json'))['selected_checkpoint'])")"

conda run --no-capture-output -n New_MMM python -u \
  tools/run_fair_interx_method.py --method hhc \
  --label hhc_precleanup_raw12288_current_mgt_common50_contact --gpu 1 \
  --checkpoint "${selected_checkpoint}" --vq-checkpoint "${vq_checkpoint}" \
  --hhc-vq-contract precleanup_raw12288 --hhc-mgt-contract current \
  --selection-manifest output/hhc_mgt_precleanup_raw12288_selection.json \
  --sampling-steps 50 --temperature 0.5 --cfg-scale 3.5 \
  --contact-guidance-scale 1.0 --hhc-contact-timing-source request_oracle \
  --contact-guidance

conda run --no-capture-output -n New_MMM python -u \
  tools/run_fair_interx_method.py --method hhc \
  --label hhc_precleanup_raw12288_current_mgt_common50_no_contact --gpu 1 \
  --checkpoint "${selected_checkpoint}" --vq-checkpoint "${vq_checkpoint}" \
  --hhc-vq-contract precleanup_raw12288 --hhc-mgt-contract current \
  --selection-manifest output/hhc_mgt_precleanup_raw12288_selection.json \
  --sampling-steps 50 --temperature 0.5 --cfg-scale 3.5 \
  --contact-guidance-scale 1.0 --hhc-contact-timing-source none \
  --no-contact-guidance

conda run --no-capture-output -n New_MMM python -u \
  tools/audit_hhc_contact_guidance_maps.py

conda run --no-capture-output -n New_MMM python -u \
  tools/run_fair_interx_method.py --method hhc \
  --label hhc_precleanup_raw12288_current_mgt_common50_contact_legacy_full_axis_bug_rep00 \
  --gpu 1 \
  --checkpoint "${selected_checkpoint}" --vq-checkpoint "${vq_checkpoint}" \
  --hhc-vq-contract precleanup_raw12288 --hhc-mgt-contract current \
  --selection-manifest output/hhc_mgt_precleanup_raw12288_selection.json \
  --sampling-steps 50 --temperature 0.5 --cfg-scale 3.5 \
  --contact-guidance-scale 1.0 --hhc-contact-timing-source request_oracle \
  --hhc-contact-map \
    checkpoints/vq_precleanup_raw12288/code_contact_probs_mmm_legacy_full_axis_bug.npy \
  --hhc-contact-map-contract mmm_legacy_full_axis_bug \
  --contact-guidance --start-replication 0 --end-replication 1

# Keep every remaining GPU-1 dependency in this same shell.  The annotation
# launcher is idempotent and only revalidates the already-complete Luna caches.
bash tools/run_remaining_interx_codex_annotations.sh
bash tools/queue_post_hhc_gpu1_experiments.sh none
bash tools/queue_hhc_timing_after_post_gpu1.sh none

# The final report consumes both independent GPU branches.  Waiting here avoids
# creating another long-lived report waiter.
if [[ "${gpu0_branch_pid}" != none ]]; then
  while ps -p "${gpu0_branch_pid}" >/dev/null 2>&1; do
    sleep 30
  done
fi
bash tools/queue_review_report_after_all_training.sh none none
