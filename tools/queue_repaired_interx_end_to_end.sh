#!/usr/bin/env bash
set -euo pipefail

# Continue both already-running GPU branches without allocating another tmux
# pane. Every child launcher is resumable or fail-closed, so rerunning this
# orchestrator cannot silently mix experiment states. CoShMDM has a full-state
# OS-restart resume path; the exact historical MGT replay remains fail-closed
# after an interruption because silently changing its RNG/W&B trajectory would
# invalidate the bit-exact repair proof.
cd /mnt/8T-HDD/wenning/HHC

run_gpu0_branch() {
  # Wait for/reuse the active CoShMDM run, audit its exact terminal state and
  # evaluate it on the frozen Inter-X requests. TiMotion follows only because
  # no released Inter-X checkpoint exists for that review baseline.
  bash tools/queue_gpu0_cosh_timotion.sh
}

run_gpu1_branch() {
  while pgrep -f \
      'python -u train/train_t2i_trans.py.*mgt_known_good_historical_contract_v7' \
      >/dev/null 2>&1
  do
    sleep 30
  done

  bash tools/finalize_mgt_repair_v7.sh
  bash tools/run_formal_hhc_after_mgt_repair.sh
  bash tools/queue_post_hhc_gpu1_experiments.sh none
  bash tools/queue_hhc_timing_after_post_gpu1.sh none
}

run_gpu0_branch &
gpu0_branch_pid=$!

gpu1_status=0
run_gpu1_branch || gpu1_status=$?
gpu0_status=0
wait "${gpu0_branch_pid}" || gpu0_status=$?

if [[ "${gpu1_status}" -ne 0 || "${gpu0_status}" -ne 0 ]]; then
  echo "Inter-X branch failed: gpu0=${gpu0_status}, gpu1=${gpu1_status}."
  exit 2
fi

# Both GPU branches and all shape/optimization experiments are now complete.
# Build the report in this same process; no report-waiter terminal is needed.
bash tools/queue_review_report_after_all_training.sh none
