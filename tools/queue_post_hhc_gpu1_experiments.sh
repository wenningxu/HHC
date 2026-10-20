#!/usr/bin/env bash
set -euo pipefail

# Interact2AR starts after the final HHC/InterMask chain and owns GPU 1 until
# its strict evaluation is complete.  Only then run additional review studies.
wait_pid="${1:-504612}"
if [[ "${wait_pid}" != none ]]; then
  while ps -p "${wait_pid}" >/dev/null 2>&1; do
    sleep 30
  done
fi

cd /mnt/8T-HDD/wenning/HHC

conda() {
  /home/wenning/miniconda3/bin/conda "$@"
}

# Diagnose a weak FID trajectory before any further HHC retraining.  This uses
# held-out validation tokens only and tests the exact train mask, fully masked
# initial state, and Root->Body->Hands conditional contexts at early versus
# selected checkpoints.  The artifact is idempotent because the active formal
# launcher may already have produced it.
selection="output/hhc_formal_raw12288_repaired_mgt_selection.json"
selected_checkpoint="$(/home/wenning/miniconda3/envs/New_MMM/bin/python -c \
  "import json; print(json.load(open('${selection}'))['selected_checkpoint'])")"
selected_training_manifest="$(/home/wenning/miniconda3/envs/New_MMM/bin/python -c \
  "import json; print(json.load(open('${selection}'))['training_manifest'])")"
run_dir="$(dirname "${selected_training_manifest}")"

# Marginal code coverage alone cannot characterize the Root -> Body -> Hands
# cascade. Freeze the support of the exact conditional tuples and within-part
# transitions on train versus validation before interpreting exact-token
# accuracy. This is data-only, uses no test records, and is cheap to rerun.
conda run --no-capture-output -n New_MMM python -u \
  tools/audit_mgt_train_val_conditional_support.py \
  --training-manifest "${selected_training_manifest}" \
  --output output/hhc_formal_mgt_train_val_conditional_support_audit.json

mask_exposure_args=(--checkpoint "${selected_checkpoint}")
for diagnostic_step in 10000 20000 30000 40000 50000; do
  diagnostic_prefix="$(printf '%09d' "${diagnostic_step}")"
  diagnostic_matches=("${run_dir}/model-${diagnostic_prefix}-"*.ckpt)
  if [[ -s "${diagnostic_matches[0]}" \
      && "${diagnostic_matches[0]}" != "${selected_checkpoint}" ]]; then
    mask_exposure_args+=(--checkpoint "${diagnostic_matches[0]}")
  fi
done
mask_exposure_output="output/hhc_formal_mgt_mask_exposure_diagnostic.json"
if [[ ! -s "${mask_exposure_output}" ]]; then
  conda run --no-capture-output -n New_MMM python -u \
    tools/audit_mgt_mask_exposure_gap.py \
    --training-manifest "${selected_training_manifest}" \
    "${mask_exposure_args[@]}" \
    --device cuda:1 --samples 256 --batch-size 16 --seed 90123 \
    --output "${mask_exposure_output}"
fi

mask_exposure_training_output="output/hhc_formal_mgt_mask_exposure_training_diagnostic.json"
if [[ ! -s "${mask_exposure_training_output}" ]]; then
  # Reuse the exact same checkpoints and masks on training tokens.  The paired
  # artifact distinguishes train/validation memorization from sampler or
  # checkpoint corruption without touching the test set.
  conda run --no-capture-output -n New_MMM python -u \
    tools/audit_mgt_mask_exposure_gap.py \
    --training-manifest "${selected_training_manifest}" \
    "${mask_exposure_args[@]}" \
    --device cuda:1 --samples 256 --batch-size 16 --seed 90123 \
    --split-role training \
    --output "${mask_exposure_training_output}"
fi

# The original Inter-X corpus binds one continuous body shape to each motion.
# Compare requested versus deterministically permuted shapes under identical
# cascade contexts on train and validation tokens.  A train-only advantage is
# evidence for shape-as-sample-ID memorization and directly motivates the
# random-shape/optimization augmentation requested for the formal retrain.
for split_role in validation training; do
  shape_dependence_output="output/hhc_formal_mgt_shape_dependence_${split_role}.json"
  if [[ ! -s "${shape_dependence_output}" ]]; then
    conda run --no-capture-output -n New_MMM python -u \
      tools/audit_mgt_shape_condition_dependence.py \
      --training-manifest "${selected_training_manifest}" \
      --checkpoint "${selected_checkpoint}" \
      --device cuda:1 --samples 256 --batch-size 16 --seed 90123 \
      --split-role "${split_role}" --output "${shape_dependence_output}"
  fi
done
conda run --no-capture-output -n New_MMM python -u \
  tools/summarize_mgt_shape_condition_dependence.py \
  --training output/hhc_formal_mgt_shape_dependence_training.json \
  --validation output/hhc_formal_mgt_shape_dependence_validation.json \
  --output output/hhc_formal_mgt_shape_dependence_pair.json

# Before expensive optimization or test-set sampling sweeps, cheaply rule in
# or out inference calibration on the exact 544-record validation set.  Every
# row retains the user-specified 50 outer iterations and is diagnostic-only.
conda run --no-capture-output -n New_MMM python -u \
  tools/prepare_hhc_validation_bridge_requests.py
conda run --no-capture-output -n New_MMM python -u \
  tools/run_hhc_validation_sampler_calibration.py \
  --device cuda:1 --batch-size 32

echo 'Core HHC mask/shape/sampler diagnostics complete; releasing the MGT control stage.'
