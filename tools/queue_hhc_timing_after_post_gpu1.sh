#!/usr/bin/env bash
set -euo pipefail

cd /mnt/8T-HDD/wenning/HHC

# Historical dependency chain retained for provenance only.  Its final edge
# targets the archived polling shape queue and is not part of the current
# manually gated Inter-X workflow.
echo "Archived dependency chain: shape augmentation is launched only through the explicit last-stage entrypoint." >&2
exit 2

conda() {
  /home/wenning/miniconda3/bin/conda "$@"
}

wait_pid="${1:-602372}"
if [[ "${wait_pid}" != none ]]; then
  while ps -p "${wait_pid}" >/dev/null 2>&1; do
    sleep 30
  done
fi

# Reuse this same GPU-1 chain for the MGT control and two matched
# shape-augmentation arms; no additional background waiter terminal is
# created.  The historical V7
# replay and the following formal-HHC run bind the current training entrypoint
# bytes, so root-stat tokenizer wiring is deliberately applied only after both
# have ended.  The guarded applier requires both formal common-50 summaries,
# exact pre-wiring source hashes, and no live MGT process.  Keep the durable
# service alive at the explicit audit gate if any later check still fails.
bash tools/apply_post_formal_shape_root_tokenizer_wiring.sh

conda run --no-capture-output -n New_MMM python -u \
  tools/apply_post_formal_online_wandb_metrics.py --apply

# The original MMM development loader used a 38-position token tensor, while
# the completed formal baseline used 50.  The frozen fair evaluator observes
# at most 150 frames, and the cache audit proves that this changes real motion
# payload rather than padding alone.  Enable the explicit source-derived
# contract only after preserving the 50-window baseline, then test it before
# changing the loss reduction.  Audit failures stop the foreground queue
# immediately; they must never leave a polling shell occupying a terminal.
bash tools/apply_post_formal_global_masked_loss_diagnostic.sh
conda run --no-capture-output -n New_MMM python -u \
  tools/apply_post_formal_original38_token_window.py --apply

conda run --no-capture-output -n New_MMM python -u \
  tools/audit_shape_root_tokenizer_wiring.py

bash tools/run_hhc_original38_token_window_control.sh

window_decision="$(/home/wenning/miniconda3/envs/New_MMM/bin/python -c \
  "import json; print(json.load(open('output/hhc_original38_token_window_control_assessment.json'))['decision'])")"
if [[ "${window_decision}" == promote_original_mmm_38_window ]]; then
  bash tools/continue_hhc_original38_token_window_control_300k.sh
  selected_38_fid="$(/home/wenning/miniconda3/envs/New_MMM/bin/python -c \
    "import json; print(json.load(open('output/hhc_original38_token_window_control_300k_selection.json'))['selected_validation_fid'])")"
  if ! /home/wenning/miniconda3/envs/New_MMM/bin/python -c \
    "raise SystemExit(0 if float('${selected_38_fid}') <= 0.4 else 1)"
  then
    echo "Promoted 38-window run remains above the preregistered 0.4 validation-FID threshold (${selected_38_fid}); loss diagnosis is required before shape training."
    exit 2
  fi
  conda run --no-capture-output -n New_MMM python -u \
    tools/apply_promoted_original38_shape_baseline.py --apply
elif [[ "${window_decision}" == retain_50_window_for_global_loss_diagnostic ]]; then
  bash tools/run_hhc_global_masked_loss_control.sh
else
  echo "Unknown token-window decision: ${window_decision}"
  exit 2
fi

bash tools/queue_interx_shape_augmentation_gpu1.sh

# Reviewer-only optimizer, sampler-Pareto and timing studies must not delay the
# weak-FID control/retraining decision above.
exec bash tools/queue_post_shape_review_experiments_gpu1.sh
