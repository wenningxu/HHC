#!/usr/bin/env bash
set -euo pipefail

cd /mnt/8T-HDD/wenning/HHC

if [[ $# -ne 4 || ! "$1" =~ ^[0-9]+$ || ! "$2" =~ ^[0-9]+$ ]]; then
  echo "usage: $0 BASELINE_GPU AUGMENTED_GPU BASELINE_RUN AUGMENTED_RUN" >&2
  exit 2
fi
baseline_gpu="$1"
augmented_gpu="$2"
baseline_run="$3"
augmented_run="$4"
if [[ "${baseline_gpu}" == "${augmented_gpu}" ]]; then
  echo "checkpoint selection and final evaluation require two distinct GPUs" >&2
  exit 2
fi

test -s "${baseline_run}/training_manifest.json"
test -s "${augmented_run}/training_manifest.json"

for arm in baseline augmented; do
  if [[ "${arm}" == baseline ]]; then
    arm_run="${baseline_run}"
  else
    arm_run="${augmented_run}"
  fi
  test -s "${arm_run}/latest-000050000.ckpt"
  /home/wenning/miniconda3/envs/New_MMM/bin/python \
    tools/audit_lightning_checkpoint.py \
    --checkpoint "${arm_run}/latest-000050000.ckpt" \
    --expected-step 50000 --expected-lr 0.0002 \
    --expected-weight-decay 0.000001 \
    --output "output/hhc_no_shape_${arm}_pilot50k_final_checkpoint_audit.json"
  /home/wenning/miniconda3/envs/New_MMM/bin/python \
    tools/audit_wandb_online_metric_history.py \
    --manifest "${arm_run}/training_manifest.json" \
    --metric-profile masked_token_classification \
    --expected-final-step 50000 \
    --output "output/hhc_no_shape_${arm}_pilot50k_wandb_history_audit.json"
done

raw_val_gt_manifest="output/interx_validation_raw_ground_truth_manifest.json"
/home/wenning/miniconda3/envs/New_MMM/bin/python \
  tools/promote_interx_validation_request_contract.py
if [[ ! -s "${raw_val_gt_manifest}" ]]; then
  /home/wenning/miniconda3/envs/New_MMM/bin/python \
    tools/replay_interx_validation_raw_ground_truth.py \
    --manifest "${raw_val_gt_manifest}"
fi

base_stats="dataset/interx"
base_cache="checkpoints/vq_precleanup_raw12288_historical_mmm_cache"
baseline_selection="output/hhc_no_shape_v4_pilot50k_validation_selection.json"
augmented_selection="output/hhc_no_shape_optimizedaug_pilot50k_validation_selection.json"
baseline_label="hhc_no_mgt_shape_v4_pilot50k_bestval"
augmented_label="hhc_no_mgt_shape_optimizedaug_pilot50k_bestval"

# The paired final suite contains two sets of 20 x (1696 main + 100 x 30 MM)
# generations. Fail before work starts if there is not enough room.
available_kib="$(df --output=avail -k . | tail -n 1 | tr -d ' ')"
required_kib="$((80 * 1024 * 1024))"
if [[ ! "${available_kib}" =~ ^[0-9]+$ ]] || (( available_kib < required_kib )); then
  echo "insufficient disk for paired joint evaluation: ${available_kib:-unknown} KiB" >&2
  exit 2
fi

# Both arms are selected on the identical frozen official Inter-X validation
# request. The augmented overlay copies the original mean/std byte-identically;
# use the original paths explicitly here so inference normalization is identical.
/home/wenning/miniconda3/envs/New_MMM/bin/python \
  tools/select_no_shape_checkpoint_on_official_interx_val.py \
  --run-dir "${baseline_run}" --stats-dir "${base_stats}" \
  --cache-root "${base_cache}" \
  --raw-ground-truth-manifest "${raw_val_gt_manifest}" \
  --output-dir output/hhc_no_shape_v4_pilot50k_official_val \
  --output "${baseline_selection}" --device "cuda:${baseline_gpu}" &
baseline_selection_pid="$!"
/home/wenning/miniconda3/envs/New_MMM/bin/python \
  tools/select_no_shape_checkpoint_on_official_interx_val.py \
  --run-dir "${augmented_run}" --stats-dir "${base_stats}" \
  --cache-root "${base_cache}" \
  --raw-ground-truth-manifest "${raw_val_gt_manifest}" \
  --output-dir output/hhc_no_shape_optimizedaug_pilot50k_official_val \
  --output "${augmented_selection}" --device "cuda:${augmented_gpu}" &
augmented_selection_pid="$!"
wait "${baseline_selection_pid}"
wait "${augmented_selection_pid}"

baseline_checkpoint="$(jq -r '.selected_checkpoint' "${baseline_selection}")"
augmented_checkpoint="$(jq -r '.selected_checkpoint' "${augmented_selection}")"

tools/run_hhc_no_shape_joint_evaluation.sh \
  "${baseline_gpu}" "${baseline_checkpoint}" "${baseline_label}" \
  "${base_stats}" "${base_cache}" &
baseline_pid="$!"
tools/run_hhc_no_shape_joint_evaluation.sh \
  "${augmented_gpu}" "${augmented_checkpoint}" "${augmented_label}" \
  "${base_stats}" "${base_cache}" &
augmented_pid="$!"
wait "${baseline_pid}"
wait "${augmented_pid}"

/home/wenning/miniconda3/envs/New_MMM/bin/python \
  tools/compare_interact2ar_joint_augmentation_paired.py \
  --baseline "output/interact2ar_joint_evaluator_candidates/${baseline_label}/scores/${baseline_label}" \
  --augmented "output/interact2ar_joint_evaluator_candidates/${augmented_label}/scores/${augmented_label}" \
  --replications 20 \
  --output output/interact2ar_joint_evaluator_candidates/no_shape_optimizedaug_paired.json
