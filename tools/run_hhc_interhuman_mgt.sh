#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 || ! "$1" =~ ^[0-9]+$ || ! "$2" =~ ^(pilot|resume-pilot|continue|finalize)$ ]]; then
  echo "usage: $0 GPU_INDEX {pilot|resume-pilot|continue|finalize}" >&2
  exit 2
fi

gpu="$1"
mode="$2"
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

# Latest user ordering: accepted sampling exploration, then completed shape
# augmentation, then InterHuman. External-method comparisons are not a gate.
shape_root="dataset/interx_shape_ssp62_physical_full6238_reviewfiltered_20260930"
shape_manifest="$shape_root/materialization_manifest.json"
shape_status="$shape_root/_materialization_status.json"
test "$(jq -r '.status' "$shape_status")" = "complete"
test "$(sha256sum "$shape_manifest" | cut -d ' ' -f 1)" = \
  "$(jq -r '.manifest_sha256' "$shape_status")"
jq -e '.status == "complete" and .quality_selection.requested_count == 6238
  and .quality_selection.pass_fraction > 0.6' "$shape_manifest" >/dev/null

output_root="${HHC_INTERHUMAN_OUTPUT_ROOT:-$repo_root/output}"
vq_run="$output_root/vq_interhuman/hhc_interhuman_vq_bodyroot_ema_v3"
vq_manifest="$vq_run/training_manifest.json"
vq_wandb_audit="$vq_run/wandb_history_audit.json"
test -s "$vq_manifest"
test "$(jq -r '.status' "$vq_manifest")" = "complete"
test "$(jq -r '.final_optimizer_step' "$vq_manifest")" = "30000"
vq_checkpoint="$(jq -r '.best_validation_checkpoint' "$vq_manifest")"
test -s "$vq_checkpoint"
test -s "$vq_wandb_audit"

case "${HHC_INTERHUMAN_MGT_VARIANT:-pilot_v1}" in
  pilot_v1)
    name="hhc_interhuman_mgt_pilot_v1"
    eval_name="hhc_interhuman_mgt_common50"
    contract="$repo_root/configs/interhuman_mgt_pilot_v1.json"
    ;;
  dropout03_v2)
    name="hhc_interhuman_mgt_dropout03_v2"
    eval_name="${name}_common50"
    contract="$repo_root/configs/interhuman_mgt_dropout03_v2.json"
    ;;
  *) echo "unsupported HHC_INTERHUMAN_MGT_VARIANT" >&2; exit 2 ;;
esac
run_dir="$output_root/mgt_interhuman/$name"
manifest="$run_dir/training_manifest.json"
resume_args=()
run_training=true
if [[ "$mode" == "pilot" ]]; then
  if [[ -e "$run_dir" ]]; then
    echo "refusing to overwrite existing InterHuman MGT run: $run_dir" >&2
    exit 2
  fi
  run_until=50000
elif [[ "$mode" == "resume-pilot" ]]; then
  test -s "$manifest"
  test "$(jq -r '.status' "$manifest")" = "running"
  resume_checkpoint="$run_dir/checkpoints/latest.pth"
  test -s "$resume_checkpoint"
  resume_args=(--resume "$resume_checkpoint")
  run_until=50000
elif [[ "$mode" == "continue" ]]; then
  test -s "$manifest"
  status="$(jq -r '.status' "$manifest")"
  if [[ "$status" != "pilot_decision_ready" ]]; then
    echo "continue requires pilot_decision_ready, got $status" >&2
    exit 2
  fi
  resume_checkpoint="$(jq -r '.resume_checkpoint' "$manifest")"
  test -s "$resume_checkpoint"
  resume_args=(--resume "$resume_checkpoint")
  run_until=300000
else
  test -s "$manifest"
  test "$(jq -r '.status' "$manifest")" = "complete"
  run_training=false
fi

used_mib="$(nvidia-smi --id="$gpu" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')"
if [[ ! "$used_mib" =~ ^[0-9]+$ ]] || (( used_mib > 1024 )); then
  echo "GPU $gpu is not free (memory.used=${used_mib:-unknown} MiB)." >&2
  exit 2
fi

if [[ "$run_training" == "true" ]]; then
  conda run --no-capture-output -n New_MMM python -u train/train_mgt_interhuman.py \
    --confirm-after-interx --name "$name" --contract "$contract" --device "cuda:$gpu" \
    --output-root "$output_root/mgt_interhuman" \
    --evaluator-device "cuda:$gpu" \
    --vq-checkpoint "$vq_checkpoint" \
    --vq-training-manifest "$vq_manifest" \
    --vq-wandb-audit "$vq_wandb_audit" \
    --total-steps 300000 --run-until-step "$run_until" \
    --micro-batch-size 64 --gradient-accumulation 1 \
    --num-workers 4 --wandb-mode online \
    "${resume_args[@]}"
fi

if [[ "$mode" == "pilot" || "$mode" == "resume-pilot" ]]; then
  test "$(jq -r '.status' "$manifest")" = "pilot_decision_ready"
  test "$(jq -r '.current_optimizer_step' "$manifest")" = "50000"
  exit 0
fi

test "$(jq -r '.status' "$manifest")" = "complete"
selected="$(jq -r '.best_validation_checkpoint' "$manifest")"
test -s "$selected"
tools/finalize_hhc_interhuman_common50.sh \
  "$gpu" "$selected" "$manifest" \
  "$output_root/interhuman_fair_eval/$eval_name"
