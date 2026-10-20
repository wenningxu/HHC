#!/usr/bin/env bash
set -euo pipefail

cd /mnt/8T-HDD/wenning/HHC

if [[ $# -ne 2 || ! "$1" =~ ^[0-9]+$ ]]; then
  echo "usage: $0 GPU_INDEX NO_SHAPE_50K_RUN_DIR" >&2
  exit 2
fi
gpu="$1"
run_dir="$2"
stats_dir="dataset/interx"
cache_root="checkpoints/vq_precleanup_raw12288_historical_mmm_cache"
selection="output/hhc_no_shape_v4_pilot50k_validation_selection.json"
selection_dir="output/hhc_no_shape_v4_pilot50k_official_val"
label="hhc_no_mgt_shape_v4_pilot50k_bestval_common50"
joint_root="output/interact2ar_joint_evaluator_candidates/${label}"
shape_reference="output/interact2ar_joint_evaluator_suite/hhc"
raw_val_gt_manifest="output/interx_validation_raw_ground_truth_manifest.json"
final_checkpoint="${run_dir}/latest-000050000.ckpt"
checkpoint_audit="output/hhc_no_shape_v4_pilot50k_final_checkpoint_audit.json"
wandb_audit="output/hhc_no_shape_v4_pilot50k_wandb_history_audit.json"

test -s "${run_dir}/training_manifest.json"
test -s output/interact2ar_joint_evaluator_suite/pre_shape_summary.json
test -s output/interact2ar_joint_evaluator_suite/hhc/replication_00.json
test -s output/interact2ar_joint_evaluator_suite/hhc/replication_19.json
test -s "${final_checkpoint}"

/home/wenning/miniconda3/envs/New_MMM/bin/python \
  tools/promote_interx_validation_request_contract.py

# Training completion is not inferred from the process exit alone. Require the
# final optimizer state and the completed online W&B history, including train,
# validation, gradient and weight telemetry requested by the user.
/home/wenning/miniconda3/envs/New_MMM/bin/python \
  tools/audit_lightning_checkpoint.py \
  --checkpoint "${final_checkpoint}" --expected-step 50000 \
  --expected-lr 0.0002 --expected-weight-decay 0.000001 \
  --output "${checkpoint_audit}"
/home/wenning/miniconda3/envs/New_MMM/bin/python \
  tools/audit_wandb_online_metric_history.py \
  --manifest "${run_dir}/training_manifest.json" \
  --metric-profile masked_token_classification --expected-final-step 50000 \
  --output "${wandb_audit}"

# Checkpoint selection uses the same released Interact2AR joint representation
# as the final acceptance table. Replay the frozen held-out validation loader
# only once; the selector validates all request/raw-GT hashes before use.
if [[ ! -s "${raw_val_gt_manifest}" ]]; then
  /home/wenning/miniconda3/envs/New_MMM/bin/python \
    tools/replay_interx_validation_raw_ground_truth.py \
    --manifest "${raw_val_gt_manifest}"
fi

used_mib="$(nvidia-smi --id="${gpu}" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')"
if [[ ! "${used_mib}" =~ ^[0-9]+$ ]] || (( used_mib > 1024 )); then
  echo "GPU ${gpu} is not free (memory.used=${used_mib:-unknown} MiB)." >&2
  exit 2
fi

# Select only on the frozen official validation split, using exactly the same
# common-50 sampling contract as the final generation run.
/home/wenning/miniconda3/envs/New_MMM/bin/python \
  tools/select_no_shape_checkpoint_on_official_interx_val.py \
  --run-dir "${run_dir}" --stats-dir "${stats_dir}" \
  --cache-root "${cache_root}" --output-dir "${selection_dir}" \
  --raw-ground-truth-manifest "${raw_val_gt_manifest}" \
  --output "${selection}" --device "cuda:${gpu}"

checkpoint="$(jq -r '.selected_checkpoint' "${selection}")"
test -s "${checkpoint}"

# Generate 20 paired main/MM replications, score all three released Interact2AR
# joint evaluators, and assess the paper thresholds.
tools/run_hhc_no_shape_joint_evaluation.sh \
  "${gpu}" "${checkpoint}" "${label}" "${stats_dir}" "${cache_root}"

# The retained shape-conditioned HHC reference was generated with the same
# request seeds and scored by the byte-identical joint evaluator source.  The
# comparator fails closed on any request, raw-GT, or evaluator drift.
/home/wenning/miniconda3/envs/New_MMM/bin/python \
  tools/compare_interact2ar_joint_augmentation_paired.py \
  --baseline "${shape_reference}" \
  --augmented "${joint_root}/scores/${label}" \
  --comparison-kind shape_condition_ablation \
  --claim-scope diagnostic_unmatched_training_budget \
  --replications 20 \
  --output "${joint_root}/shape_conditioned_vs_no_shape_paired.json"

# Freeze one fail-closed result card only after every generation, score,
# acceptance check, paired comparison and MM-collapse diagnostic is complete.
/home/wenning/miniconda3/envs/New_MMM/bin/python \
  tools/build_hhc_no_shape_diagnostic_result_card.py
