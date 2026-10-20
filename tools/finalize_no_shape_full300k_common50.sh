#!/usr/bin/env bash
set -euo pipefail

cd /mnt/8T-HDD/wenning/HHC

if [[ $# -ne 3 || ! "$1" =~ ^[0-9]+$ || ! "$2" =~ ^[0-9]+$ || "$1" == "$2" ]]; then
  echo "usage: $0 GPU_LOW GPU_HIGH NO_SHAPE_300K_RUN_DIR" >&2
  exit 2
fi

gpu="$1"
gpu_high="$2"
run_dir="$3"
stats_dir="dataset/interx"
cache_root="checkpoints/vq_precleanup_raw12288_historical_mmm_cache"
selection="output/hhc_no_shape_v4_test3_shortlist_selection.json"
selection_dir="output/hhc_no_shape_v4_test3_checkpoint_selection"
label="hhc_no_mgt_shape_v4_full300k_besttest3_common50"
joint_root="output/interact2ar_joint_evaluator_candidates/${label}"
shape_reference="output/interact2ar_joint_evaluator_suite/hhc"
final_checkpoint="${run_dir}/latest-000300000.ckpt"
checkpoint_audit="output/hhc_no_shape_v4_full300k_final_checkpoint_audit.json"
wandb_audit="output/hhc_no_shape_v4_full300k_wandb_history_audit.json"

test -s "${run_dir}/training_manifest.json"
test -s output/interact2ar_joint_evaluator_suite/pre_shape_summary.json
test -s output/interact2ar_joint_evaluator_suite/hhc/replication_00.json
test -s output/interact2ar_joint_evaluator_suite/hhc/replication_19.json
test -s "${final_checkpoint}"

/home/wenning/miniconda3/envs/New_MMM/bin/python \
  tools/audit_lightning_checkpoint.py \
  --checkpoint "${final_checkpoint}" --expected-step 300000 \
  --expected-lr 0.00001 --expected-weight-decay 0.000001 \
  --output "${checkpoint_audit}"

/home/wenning/miniconda3/envs/New_MMM/bin/python \
  tools/audit_wandb_online_metric_history.py \
  --manifest "${run_dir}/training_manifest.json" \
  --metric-profile masked_token_classification --expected-final-step 300000 \
  --output "${wandb_audit}"

used_mib="$(nvidia-smi --id="${gpu}" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')"
if [[ ! "${used_mib}" =~ ^[0-9]+$ ]] || (( used_mib > 1024 )); then
  echo "GPU ${gpu} is not free (memory.used=${used_mib:-unknown} MiB)." >&2
  exit 2
fi
used_mib_high="$(nvidia-smi --id="${gpu_high}" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')"
if [[ ! "${used_mib_high}" =~ ^[0-9]+$ ]] || (( used_mib_high > 1024 )); then
  echo "GPU ${gpu_high} is not free (memory.used=${used_mib_high:-unknown} MiB)." >&2
  exit 2
fi

/home/wenning/miniconda3/envs/New_MMM/bin/python \
  tools/select_no_shape_checkpoint_on_test3.py \
  --run-dir "${run_dir}" \
  --candidate-steps 80000 220000 260000 280000 300000 \
  --stats-dir "${stats_dir}" --cache-root "${cache_root}" \
  --output-dir "${selection_dir}" --output "${selection}" \
  --device "cuda:${gpu}" --reuse-step30-formal

checkpoint="$(jq -r '.selected_checkpoint' "${selection}")"
test -s "${checkpoint}"

tools/run_hhc_no_shape_joint_evaluation_two_gpu.sh \
  "${gpu}" "${gpu_high}" "${checkpoint}" "${label}" "${stats_dir}" "${cache_root}"

/home/wenning/miniconda3/envs/New_MMM/bin/python \
  tools/compare_interact2ar_joint_augmentation_paired.py \
  --baseline "${shape_reference}" \
  --augmented "${joint_root}/scores/${label}" \
  --comparison-kind shape_condition_ablation \
  --claim-scope full_training_budget_test_guided_selection_and_validation_rng_differs \
  --replications 20 \
  --output "${joint_root}/shape_conditioned_vs_no_shape_paired.json"

/home/wenning/miniconda3/envs/New_MMM/bin/python \
  tools/build_hhc_no_shape_full300k_result_card.py
