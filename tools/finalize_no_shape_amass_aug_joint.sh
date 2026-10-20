#!/usr/bin/env bash
set -euo pipefail

cd /mnt/8T-HDD/wenning/HHC

if (( $# < 2 || $# > 3 )) || [[ ! "$1" =~ ^[0-9]+$ ]]; then
  echo "usage: $0 GPU_INDEX AMASS_AUGMENTED_50K_RUN_DIR [optimized_selective|random_only]" >&2
  exit 2
fi

gpu="$1"
augmented_run="$2"
arm="${3:-optimized_selective}"
case "${arm}" in
  optimized_selective)
    arm_label="optimizedaug"
    comparison_kind="optimized_shape_augmentation"
    ;;
  random_only)
    arm_label="randomaug"
    comparison_kind="random_shape_augmentation"
    ;;
  *)
    echo "unsupported AMASS augmentation arm: ${arm}" >&2
    exit 2
    ;;
esac
baseline_label="hhc_no_mgt_shape_v4_pilot50k_bestval_common50"
augmented_label="hhc_no_mgt_shape_amass_${arm_label}_pilot50k_bestval_common50"
baseline_joint_root="output/interact2ar_joint_evaluator_candidates/${baseline_label}"
augmented_joint_root="output/interact2ar_joint_evaluator_candidates/${augmented_label}"
baseline_scores="${baseline_joint_root}/scores/${baseline_label}"
augmented_scores="${augmented_joint_root}/scores/${augmented_label}"
augmented_selection="output/hhc_no_shape_amass_${arm_label}_pilot50k_validation_selection.json"
raw_val_gt_manifest="output/interx_validation_raw_ground_truth_manifest.json"
base_stats="dataset/interx"
base_cache="checkpoints/vq_precleanup_raw12288_historical_mmm_cache"

# The unaugmented arm is the current frozen 20-rep common-50 job. Reusing it is
# both statistically preferable and avoids a redundant ~20-hour generation.
test -s "${baseline_joint_root}/summary.json"
test -s "${baseline_joint_root}/paper_acceptance.json"
for replication in $(seq -w 0 19); do
  test -s "${baseline_scores}/replication_${replication}.json"
done

test -s "${augmented_run}/training_manifest.json"
test -s "${augmented_run}/latest-000050000.ckpt"
/home/wenning/miniconda3/envs/New_MMM/bin/python \
  tools/audit_lightning_checkpoint.py \
  --checkpoint "${augmented_run}/latest-000050000.ckpt" \
  --expected-step 50000 --expected-lr 0.0002 \
  --expected-weight-decay 0.000001 \
  --output "output/hhc_no_shape_amass_${arm_label}_pilot50k_final_checkpoint_audit.json"
/home/wenning/miniconda3/envs/New_MMM/bin/python \
  tools/audit_wandb_online_metric_history.py \
  --manifest "${augmented_run}/training_manifest.json" \
  --metric-profile masked_token_classification \
  --expected-final-step 50000 \
  --output "output/hhc_no_shape_amass_${arm_label}_pilot50k_wandb_history_audit.json"

/home/wenning/miniconda3/envs/New_MMM/bin/python \
  tools/promote_interx_validation_request_contract.py
test -s "${raw_val_gt_manifest}"

used_mib="$(nvidia-smi --id="${gpu}" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')"
if [[ ! "${used_mib}" =~ ^[0-9]+$ ]] || (( used_mib > 1024 )); then
  echo "GPU ${gpu} is not free (memory.used=${used_mib:-unknown} MiB)." >&2
  exit 2
fi

# Select the augmented checkpoint on the same frozen held-out validation
# request and with the original Inter-X inference statistics. Test metrics are
# never used for checkpoint selection.
/home/wenning/miniconda3/envs/New_MMM/bin/python \
  tools/select_no_shape_checkpoint_on_official_interx_val.py \
  --run-dir "${augmented_run}" --stats-dir "${base_stats}" \
  --cache-root "${base_cache}" \
  --raw-ground-truth-manifest "${raw_val_gt_manifest}" \
  --output-dir output/hhc_no_shape_amass_optimizedaug_pilot50k_official_val \
  --output "${augmented_selection}" --device "cuda:${gpu}"

augmented_checkpoint="$(jq -r '.selected_checkpoint' "${augmented_selection}")"
test -s "${augmented_checkpoint}"

# Generate only the new arm. The called evaluator freezes the same 20 request
# seeds, 1696 samples/replication and 100x30 MModality contract as baseline.
tools/run_hhc_no_shape_joint_evaluation.sh \
  "${gpu}" "${augmented_checkpoint}" "${augmented_label}" \
  "${base_stats}" "${base_cache}"

/home/wenning/miniconda3/envs/New_MMM/bin/python \
  tools/compare_interact2ar_joint_augmentation_paired.py \
  --baseline "${baseline_scores}" \
  --augmented "${augmented_scores}" \
  --comparison-kind "${comparison_kind}" \
  --claim-scope matched_training \
  --replications 20 \
  --output "${augmented_joint_root}/baseline_vs_amass_augmented_paired.json"
