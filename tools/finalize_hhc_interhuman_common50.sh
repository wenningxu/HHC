#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 4 ]]; then
  echo "usage: $0 GPU_ID MGT_CHECKPOINT MGT_TRAINING_MANIFEST OUTPUT_DIR" >&2
  exit 2
fi

gpu_id="$1"
mgt_checkpoint="$2"
mgt_manifest="$3"
output_dir="$4"
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

main_requests="$repo_root/output/interhuman_fair_requests_v3"
mm_requests="$repo_root/output/interhuman_fair_mm_requests"
main_generations="$output_dir/main_generations"
mm_generations="$output_dir/mm_generations"
main_scores="$output_dir/main_scores"
mm_scores="$output_dir/mm_scores"
mkdir -p "$main_generations" "$mm_generations" "$main_scores" "$mm_scores"

conda run -n New_MMM python tools/audit_wandb_online_metric_history.py \
  --manifest "$mgt_manifest" \
  --metric-profile interhuman_mgt \
  --expected-final-step 300000 \
  --output "$output_dir/wandb_history_audit.json"

conda run -n New_MMM python tools/verify_selected_interhuman_mgt.py \
  --checkpoint "$mgt_checkpoint" \
  --training-manifest "$mgt_manifest" \
  --output "$output_dir/selected_checkpoint_verification.json"

for replication in $(seq -w 0 19); do
  conda run -n New_MMM python tools/export_hhc_interhuman.py \
    --mgt-checkpoint "$mgt_checkpoint" \
    --mgt-training-manifest "$mgt_manifest" \
    --request-file "$main_requests/replication_${replication}.json" \
    --output "$main_generations/replication_${replication}.h5" \
    --mode main --device "cuda:${gpu_id}" --batch-size 8 \
    --sampling-steps 50 --temperature 0.5 --cfg-scale 3.5

  conda run -n New_MMM python tools/export_hhc_interhuman.py \
    --mgt-checkpoint "$mgt_checkpoint" \
    --mgt-training-manifest "$mgt_manifest" \
    --request-file "$mm_requests/replication_${replication}.json" \
    --output "$mm_generations/replication_${replication}.h5" \
    --mode mm --device "cuda:${gpu_id}" --batch-size 8 \
    --sampling-steps 50 --temperature 0.5 --cfg-scale 3.5

  conda run -n New_MMM python tools/score_fair_interhuman_main.py \
    --manifest "$main_requests/manifest.json" \
    --replication "$((10#$replication))" \
    --main-generations "$main_generations/replication_${replication}.h5" \
    --output "$main_scores/replication_${replication}.json" \
    --device "cuda:${gpu_id}" --batch-size 96 --require-hhc-common50

  conda run -n New_MMM python tools/score_fair_interhuman_mm.py \
    --manifest "$mm_requests/manifest.json" \
    --replication "$((10#$replication))" \
    --mm-generations "$mm_generations/replication_${replication}.h5" \
    --output "$mm_scores/replication_${replication}.json" \
    --device "cuda:${gpu_id}" --require-hhc-common50
done

conda run -n New_MMM python tools/aggregate_fair_interhuman_metrics.py \
  --main-dir "$main_scores" \
  --mm-dir "$mm_scores" \
  --main-manifest "$main_requests/manifest.json" \
  --mm-manifest "$mm_requests/manifest.json" \
  --output "$output_dir/summary_20rep.json"
