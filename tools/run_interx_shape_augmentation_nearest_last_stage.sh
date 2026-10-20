#!/usr/bin/env bash
set -euo pipefail

cd /mnt/8T-HDD/wenning/HHC

echo "Archived invalid shape launcher: it encodes the superseded 100/0.02/freeze@50 and custom loss/variable policies. Use run_original_mmm_shape_pilot25.sh; a full launcher will be created only after that gate passes." >&2
exit 2

if (( $# < 1 || $# > 2 )) || [[ ! "$1" =~ ^[0-9]+$ ]] \
    || { (( $# == 2 )) && [[ ! "$2" =~ ^[0-9]+$ ]]; }; then
  echo "usage: $0 GPU_INDEX [SECOND_GPU_INDEX]" >&2
  exit 2
fi
gpu0="$1"
gpu1="${2:-}"
if [[ -n "${gpu1}" && "${gpu0}" == "${gpu1}" ]]; then
  echo "GPU indices must be distinct." >&2
  exit 2
fi
device0="cuda:${gpu0}"

conda() {
  /home/wenning/miniconda3/bin/conda "$@"
}

# This is deliberately a last-stage entrypoint, never a waiter.  It refuses
# to start until all main Inter-X generation comparisons are already complete.
required_json=(
  output/hhc_historical_plus_paper_mask_ratio_selection.json
  output/external_baseline_reproduction_audit.json
  output/current_interx_main_table.json
  output/current_interx_optimization_backend_contract.json
  output/optimization_historical_paper_mask_rep0/formal_manifest.json
  output/current_interx_mesh_physical/manifest.json
  output/current_interx_qualitative/manifest.json
  output/fair_eval/hhc_historical_paper_mask_contact_timing_ablation_rep0/summary.json
  output/fair_eval/hhc_historical_paper_mask_sampling_pareto_no_timing_rep0/summary.json
  output/fair_eval/hhc_historical_plus_paper_mask_ratio_contact_common50/summary.json
  output/fair_eval/hhc_historical_plus_paper_mask_ratio_no_contact_common50/summary.json
  output/timotion_interx_val_selection.json
  output/fair_eval/timotion_transformer_lpa_interx_retrained_fair/summary.json
  output/fair_eval/coshmdm_interx_retrained_2daycap_seed123/summary.json
  output/fair_eval/intermask_official_exact_common50/summary.json
  output/fair_eval/interact2ar_official_fair_no_gt_first_frame/summary.json
  output/interact2ar_author_release_calibration/summary.json
  output/interact2ar_author_release_calibration/audit.json
  output/interact2ar_joint_scorer_parity_audit.json
  output/interact2ar_joint_evaluator_suite/pre_shape_summary.json
)
for required in "${required_json[@]}"; do
  if [[ ! -s "${required}" ]]; then
    echo "Shape augmentation is last; prerequisite is incomplete: ${required}" >&2
    exit 2
  fi
done

gpus=("${gpu0}")
if [[ -n "${gpu1}" ]]; then
  gpus+=("${gpu1}")
fi
for gpu in "${gpus[@]}"; do
  used_mib="$(nvidia-smi --id="${gpu}" --query-gpu=memory.used \
    --format=csv,noheader,nounits | tr -d ' ')"
  if [[ ! "${used_mib}" =~ ^[0-9]+$ ]] || (( used_mib > 1024 )); then
    echo "GPU ${gpu} is not free (memory.used=${used_mib:-unknown} MiB)." >&2
    exit 2
  fi
done

conda run --no-capture-output -n New_MMM python -u \
  tools/audit_coshmdm_public_release_scope.py
conda run --no-capture-output -n New_MMM python -u \
  tools/build_external_baseline_reproduction_audit.py
conda run --no-capture-output -n New_MMM python -u \
  tools/audit_hhc_shape_aug_training_launch_contract.py
conda run --no-capture-output -n New_MMM python -u \
  tools/audit_shape_augmentation_last_stage_contract.py

# Project the complete two-arm queue from the actually selected primary HHC
# run and its 20-rep output.  Each arm retains 30 validation candidates plus
# latest/last recovery checkpoints.  This gate only refuses launch; it never
# deletes or compacts existing artifacts.
primary_selection="output/hhc_historical_plus_paper_mask_ratio_selection.json"
primary_training_manifest="$(/home/wenning/miniconda3/envs/New_MMM/bin/python -c \
  "import json; print(json.load(open('${primary_selection}'))['artifacts']['training_manifest']['path'])")"
weights_only_reference="output/shape_capacity/model-000300000-0.0000.ckpt"
weights_only_reference_manifest="output/shape_capacity/lightning_weights_only_reference.json"
conda run --no-capture-output -n New_MMM python -u \
  tools/export_lightning_weights_only_reference.py \
  --checkpoint "$(dirname "${primary_training_manifest}")/latest-000300000.ckpt" \
  --output "${weights_only_reference}" \
  --manifest "${weights_only_reference_manifest}"
conda run --no-capture-output -n New_MMM python -u \
  tools/audit_shape_augmentation_disk_capacity.py \
  --filesystem-root . \
  --reference-label output/fair_eval/hhc_historical_plus_paper_mask_ratio_no_contact_common50 \
  --reference-training-run "$(dirname "${primary_training_manifest}")" \
  --arms 2 --labels-per-arm 2 \
  --validation-checkpoints-per-arm 30 \
  --rolling-full-checkpoints-per-arm 2 \
  --reference-validation-checkpoint "${weights_only_reference}" \
  --reference-validation-manifest "${weights_only_reference_manifest}" \
  --free-space-after-queue-gib 40 \
  --output output/interx_shape_augmentation_disk_capacity_audit_v3.json

plan="configs/interx_shape_augmentation/timing_covered_seed123_v2/plan_manifest.json"
contacts="dataset/interx/contact_pairs_codex_luna_strict_v3_train"
selection_seed=20260828
common=(
  --plan "${plan}" --contact-dir "${contacts}"
  --signed-distance-backend nearest_vertex --seed 1234
  --selection-seed "${selection_seed}" --num-epochs 100 --fine-stage-start 50
  --foot-contact-mask-policy initial_or_current_grounded_horizontal_mean_speed
  --contact-loss-reduction-policy active_contact_mean
  --optimization-variable-policy grounded_chain_fixed_root
)

conda run --no-capture-output -n New_MMM python -u \
  tools/audit_interx_codex_contact_cache.py \
  --pairs-dir "${contacts}" --split dataset/interx/splits/train.txt \
  --output output/interx_luna_train_cache_audit.json
conda run --no-capture-output -n New_MMM python -u \
  tools/audit_interx_contact_timing_consistency.py \
  --pairs-dir "${contacts}" \
  --output output/interx_luna_train_timing_consistency_audit.json
conda run --no-capture-output -n New_MMM python -u \
  tools/run_interx_shape_augmentation.py "${common[@]}" \
  --device "${device0}" \
  --output-dir output/interx_shape_aug_nearest_vertex_preflight \
  --mode optimized --preflight-only

# Freeze a matched 25-sample pilot before the full 6238-sample run.  Random
# shape-only and optimized arms share plan rows, donors, crops and seeds.
pilot_random="output/interx_shape_aug_nearest_vertex_pilot25_random_seed20260828"
pilot_optimized="output/interx_shape_aug_nearest_vertex_pilot25_optimized_seed20260828"
pilot_before_physical="output/interx_shape_aug_nearest_vertex_pilot25_random_physical_seed20260828"
pilot_after_physical="output/interx_shape_aug_nearest_vertex_pilot25_optimized_physical_seed20260828"
pilot_comparison="output/interx_shape_aug_nearest_vertex_pilot25_paired_physical_seed20260828.json"
pilot_assessment="output/interx_shape_aug_nearest_vertex_pilot25_assessment_seed20260828.json"
runtime_analysis="${pilot_optimized}/runtime_analysis.json"
runtime_gate="output/interx_shape_aug_nearest_vertex_runtime_gate_seed20260828.json"

conda run --no-capture-output -n New_MMM python -u \
  tools/run_interx_shape_augmentation.py "${common[@]}" \
  --device "${device0}" \
  --output-dir "${pilot_random}" --mode random_shape_only \
  --positive-timing-only --limit 25
conda run --no-capture-output -n New_MMM python -u \
  tools/run_interx_shape_augmentation.py "${common[@]}" \
  --device "${device0}" \
  --output-dir "${pilot_optimized}" --mode optimized \
  --positive-timing-only --limit 25
conda run --no-capture-output -n New_MMM python -u \
  tools/analyze_shape_augmentation_runtime.py \
  --run-dir "${pilot_optimized}" --plan "${plan}" \
  --output "${runtime_analysis}" --bootstrap-replicates 10000 --seed 20260902
conda run --no-capture-output -n New_MMM python -u \
  tools/require_shape_augmentation_runtime_gate.py \
  --analysis "${runtime_analysis}" --output "${runtime_gate}" \
  --minimum-pilot-samples 25 --target-seconds-per-sample 10 \
  --maximum-projected-hours 24

# VolumetricSMPL is used here only as a common read-only physical metric; it
# is not used inside optimization and therefore cannot reintroduce the slow
# per-iteration neural-SDF path.
conda run --no-capture-output -n New_MMM python -u \
  tools/score_volumetric_smpl_npz.py --folder "${pilot_random}" \
  --output-dir "${pilot_before_physical}" --device "${device0}" \
  --query-chunk 10475 --seed 1234
conda run --no-capture-output -n New_MMM python -u \
  tools/score_volumetric_smpl_npz.py --folder "${pilot_optimized}" \
  --output-dir "${pilot_after_physical}" --device "${device0}" \
  --query-chunk 10475 --seed 1234
conda run --no-capture-output -n New_MMM python -u \
  tools/compare_paired_physical_metrics.py \
  --before "${pilot_before_physical}" --after "${pilot_after_physical}" \
  --output "${pilot_comparison}" --seed 1234 --bootstrap-samples 10000
conda run --no-capture-output -n New_MMM python -u \
  tools/assess_shape_augmentation_pilot.py \
  --comparison "${pilot_comparison}" \
  --gate configs/interx_shape_augmentation/pilot_physical_gate_v1.json \
  --output "${pilot_assessment}"

# Generate/resume the complete random-only control on the same full plan.
# This arm has no optimizer backend, so it is not part of the deprecated
# VolumetricSMPL optimization output namespace.
full_random="output/interx_shape_aug_full_random_seed20260828"
conda run --no-capture-output -n New_MMM python -u \
  tools/run_interx_shape_augmentation.py "${common[@]}" \
  --device "${device0}" \
  --output-dir "${full_random}" --mode random_shape_only
random_overlay="dataset/interx_shape_aug_random_seed123_v3"
conda run --no-capture-output -n New_MMM python -u \
  tools/materialize_interx_shape_augmentation.py \
  --optimization-dir "${full_random}" --plan "${plan}" \
  --output-root "${random_overlay}"

full="output/interx_shape_aug_full_nearest_vertex_seed20260828"
optimization_dirs=()
if [[ -n "${gpu1}" ]]; then
  sample_count="$(/home/wenning/miniconda3/envs/New_MMM/bin/python -c \
    "import json; print(json.load(open('${plan}'))['eligible_samples'])")"
  first_count=$((sample_count / 2))
  second_count=$((sample_count - first_count))
  shard0="${full}_shard0"
  shard1="${full}_shard1"
  conda run --no-capture-output -n New_MMM python -u \
    tools/run_interx_shape_augmentation.py "${common[@]}" \
    --device "${device0}" --offset 0 --limit "${first_count}" \
    --output-dir "${shard0}" --mode optimized &
  pid0=$!
  conda run --no-capture-output -n New_MMM python -u \
    tools/run_interx_shape_augmentation.py "${common[@]}" \
    --device "cuda:${gpu1}" --offset "${first_count}" --limit "${second_count}" \
    --output-dir "${shard1}" --mode optimized &
  pid1=$!
  status0=0
  status1=0
  wait "${pid0}" || status0=$?
  wait "${pid1}" || status1=$?
  if (( status0 != 0 || status1 != 0 )); then
    echo "nearest-vertex shards failed: shard0=${status0}, shard1=${status1}" >&2
    exit 1
  fi
  conda run --no-capture-output -n New_MMM python -u \
    tools/aggregate_interx_shape_augmentation_shards.py \
    --shard-dir "${shard0}" --shard-dir "${shard1}" \
    --expected-samples "${sample_count}" --expected-shards 2 \
    --output "${full}_aggregate_manifest.json"
  optimization_dirs=("${shard0}" "${shard1}")
else
  conda run --no-capture-output -n New_MMM python -u \
    tools/run_interx_shape_augmentation.py "${common[@]}" \
    --device "${device0}" --output-dir "${full}" --mode optimized
  conda run --no-capture-output -n New_MMM python -u \
    tools/analyze_shape_augmentation_runtime.py \
    --run-dir "${full}" --plan "${plan}" \
    --output "${full}/runtime_analysis.json" \
    --bootstrap-replicates 10000 --seed 20260902
  optimization_dirs=("${full}")
fi

overlay="dataset/interx_shape_aug_optimized_nearest_vertex_seed123_v3"
materialize_args=()
for directory in "${optimization_dirs[@]}"; do
  materialize_args+=(--optimization-dir "${directory}")
done
conda run --no-capture-output -n New_MMM python -u \
  tools/materialize_interx_shape_augmentation.py \
  "${materialize_args[@]}" --plan "${plan}" --output-root "${overlay}"

if [[ -n "${gpu1}" ]]; then
  bash tools/run_hhc_shape_aug_historical_paper_mask_arm.sh "${gpu0}" random &
  train_pid0=$!
  bash tools/run_hhc_shape_aug_historical_paper_mask_arm.sh "${gpu1}" optimized &
  train_pid1=$!
  train_status0=0
  train_status1=0
  wait "${train_pid0}" || train_status0=$?
  wait "${train_pid1}" || train_status1=$?
  if (( train_status0 != 0 || train_status1 != 0 )); then
    echo "shape training arms failed: random=${train_status0}, optimized=${train_status1}" >&2
    exit 1
  fi
else
  bash tools/run_hhc_shape_aug_historical_paper_mask_arm.sh "${gpu0}" random
  bash tools/run_hhc_shape_aug_historical_paper_mask_arm.sh "${gpu0}" optimized
fi

bash tools/run_interact2ar_joint_evaluator_suite.sh "${gpu0}" post_shape
bash tools/finalize_current_interx_shape_augmentation_results.sh

echo "Matched random-only and optimized shape augmentation/training/evaluation complete: ${random_overlay}, ${overlay}"
