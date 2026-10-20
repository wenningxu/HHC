#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 || ! "$1" =~ ^[0-9]+$ || ! "$2" =~ ^(pilot|resume-pilot|continue)$ ]]; then
  echo "usage: $0 GPU_INDEX {pilot|resume-pilot|continue}" >&2
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

name="hhc_interhuman_vq_bodyroot_ema_v3"
output_root="${HHC_INTERHUMAN_OUTPUT_ROOT:-$repo_root/output}"
run_dir="$output_root/vq_interhuman/$name"
manifest="$run_dir/training_manifest.json"
resume_args=()
if [[ "$mode" == "pilot" ]]; then
  if [[ -e "$run_dir" ]]; then
    echo "refusing to overwrite existing InterHuman VQ run: $run_dir" >&2
    exit 2
  fi
  run_until=10000
elif [[ "$mode" == "resume-pilot" ]]; then
  test -s "$manifest"
  test "$(jq -r '.status' "$manifest")" = "running"
  resume_checkpoint="$run_dir/checkpoints/latest.pth"
  test -s "$resume_checkpoint"
  resume_args=(--resume "$resume_checkpoint")
  run_until=10000
else
  test -s "$manifest"
  status="$(jq -r '.status' "$manifest")"
  step="$(jq -r '.current_optimizer_step' "$manifest")"
  if [[ "$status" != "pilot_decision_ready" || "$step" != "10000" ]]; then
    echo "continue requires the unique completed 10k pilot, got $status/$step" >&2
    exit 2
  fi
  resume_checkpoint="$(jq -r '.resume_checkpoint' "$manifest")"
  test -s "$resume_checkpoint"
  resume_args=(--resume "$resume_checkpoint")
  run_until=30000
fi

used_mib="$(nvidia-smi --id="$gpu" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')"
if [[ ! "$used_mib" =~ ^[0-9]+$ ]] || (( used_mib > 1024 )); then
  echo "GPU $gpu is not free (memory.used=${used_mib:-unknown} MiB)." >&2
  exit 2
fi

conda run --no-capture-output -n New_MMM python -u train/train_vq_interhuman.py \
  --confirm-after-interx --device "cuda:$gpu" --name "$name" \
  --output-root "$output_root/vq_interhuman" \
  --total-steps 30000 --run-until-step "$run_until" \
  --interhuman-quantizer fp32_window_reset_v1 \
  --batch-size 64 --num-workers 4 --wandb-mode online \
  "${resume_args[@]}"

if [[ "$mode" == "pilot" || "$mode" == "resume-pilot" ]]; then
  test "$(jq -r '.status' "$manifest")" = "pilot_decision_ready"
  test "$(jq -r '.current_optimizer_step' "$manifest")" = "10000"
  exit 0
fi

test "$(jq -r '.status' "$manifest")" = "complete"
test "$(jq -r '.final_optimizer_step' "$manifest")" = "30000"
selected="$(jq -r '.best_validation_checkpoint' "$manifest")"
test -s "$selected"
result_dir="$output_root/interhuman_vq_validation/$name"
mkdir -p "$result_dir"

conda run -n New_MMM python tools/audit_wandb_online_metric_history.py \
  --manifest "$manifest" --metric-profile interhuman_vq \
  --expected-final-step 30000 --output "$run_dir/wandb_history_audit.json"

conda run -n New_MMM python tools/evaluate_hhc_interhuman_vq_validation.py \
  --checkpoint "$selected" --training-manifest "$manifest" \
  --device "cuda:$gpu" --output "$result_dir/hhc_selected.json"

official_h5="$output_root/interhuman_vq_validation/official_intermask_reconstruction.h5"
if [[ ! -s "$official_h5" ]]; then
  conda run --no-capture-output -n New_MMM python -u \
    tools/export_official_intermask_interhuman_vq_reconstruction.py \
    --validation-manifest output/interhuman_validation_requests/manifest.json \
    --output "$official_h5" --device "cuda:$gpu"
fi
conda run -n New_MMM python tools/score_interhuman_vq_validation.py \
  --generations "$official_h5" --device "cuda:$gpu" \
  --output "$result_dir/official_intermask.json"
conda run -n New_MMM python tools/compare_interhuman_vq_validation.py \
  --hhc "$result_dir/hhc_selected.json" \
  --official-intermask "$result_dir/official_intermask.json" \
  --output "$result_dir/hhc_vs_official_intermask.json"
