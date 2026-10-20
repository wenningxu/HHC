#!/usr/bin/env bash
set -euo pipefail

cd /mnt/8T-HDD/wenning/HHC

if (( $# < 1 || $# > 2 )) || [[ ! "$1" =~ ^[0-9]+$ ]]; then
  echo "usage: $0 GPU_INDEX [optimized_selective|random_only]" >&2
  exit 2
fi
gpu="$1"
arm="${2:-optimized_selective}"
python_env=/home/wenning/miniconda3/envs/New_MMM/bin/python
used_mib="$(nvidia-smi --id="${gpu}" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')"
if [[ ! "${used_mib}" =~ ^[0-9]+$ ]] || (( used_mib > 1024 )); then
  echo "GPU ${gpu} is not free (memory.used=${used_mib:-unknown} MiB)." >&2
  exit 2
fi

plan="configs/interx_shape_augmentation/humanml3d_train_amass_seed123_v1/plan_manifest.json"
pipeline="output/humanml3d_amass_shape_pipeline_v1/pipeline_manifest.json"
case "${arm}" in
  optimized_selective)
    overlay="dataset/interx_shape_aug_amass_optimized_selective_seed123_v1"
    expected_mode="optimized_positive_plus_random_zero"
    arm_label="optimizedaug"
    ;;
  random_only)
    overlay="dataset/interx_shape_aug_amass_random_only_seed123_v1"
    expected_mode="random_shape_only"
    arm_label="randomaug"
    ;;
  *)
    echo "unsupported AMASS augmentation arm: ${arm}" >&2
    exit 2
    ;;
esac
augmentation_manifest="${overlay}/materialization_manifest.json"
cache_root="checkpoints/vq_precleanup_raw12288_no_shape_amass_${arm_label}_cache"
vq_checkpoint="checkpoints/vq_precleanup_raw12288/net_175000.pth"
experiment="hhc_historical_paper_mask_no_mgt_shape_amass_${arm_label}_pilot50k_seed114514_v1"

test -s "${pipeline}"
test -s "${plan}"
test -s "${augmentation_manifest}"
/home/wenning/miniconda3/envs/New_MMM/bin/python - "${pipeline}" "${plan}" "${augmentation_manifest}" "${expected_mode}" <<'PY'
import json
import sys

pipeline, plan, materialization = [json.load(open(path)) for path in sys.argv[1:]]
assert pipeline.get('status') == 'complete'
assert pipeline['invariants']['humanml3d_train_only'] is True
assert pipeline['invariants']['official_smplh_to_smplx_transfer'] is True
assert pipeline['invariants']['direct_beta_copy'] is False
assert plan.get('status') == 'complete'
assert plan.get('donor_pool_kind') == 'humanml3d_train_amass_transferred_smplx'
assert plan.get('eligible_samples') == 6238
assert plan.get('positive_timing_samples') == 5868
assert materialization.get('status') == 'complete'
assert materialization.get('augmentation_mode') == sys.argv[4]
assert materialization['counts']['augmented'] == 6238
assert materialization['counts']['official_train'] == 9110
assert materialization['counts']['combined_train'] == 15348
assert materialization['normalization']['copy_is_byte_identical'] is True
PY

shared_args=(
  --vq-checkpoint "${vq_checkpoint}"
  --cache-root "${cache_root}" --codebook-dir "${cache_root}/codebook"
  --token-cache-contract hybrid_historical_original_plus_deterministic_augmented_v1
  --historical-codebook-source /home/wenning/projects/MMM/checkpoints/vq/codebook
  --historical-contact-source /home/wenning/projects/MMM/checkpoints/vq/contact_token_label
  --nb-code 12288 --code-dim 256 --shape-condition-dim 8 --width 512
  --embed-dim-gpt 1024 --n-head-gpt 16 --num-layers 9 --num-local-layer 2
  --train-asset-root "${overlay}" --augmentation-manifest "${augmentation_manifest}"
  --train-split-file "${overlay}/splits/train.txt"
  --val-split-file dataset/interx/splits/val.txt
  --all-split-file "${overlay}/splits/all.txt"
  --train-motion-file "${overlay}/motions/train.h5"
  --val-motion-file dataset/interx/motions/val.h5
  --all-motion-file "${overlay}/motions/all.h5"
  --stats-dir "${overlay}" --stats-prefix Interx
  --validation-mm-repeats 11 --seed 114514
)

# Cache generation is deterministic preprocessing, not a training experiment.
# A completed immutable cache is validated again by the training entrypoint and
# is not needlessly re-exported after a machine restart.
cache_manifest="${cache_root}/codebook/_cache_manifest.json"
cache_complete="$(${python_env} - "${cache_manifest}" <<'PY'
import json
import pathlib
import sys
p = pathlib.Path(sys.argv[1])
try:
    print('yes' if json.loads(p.read_text()).get('status') == 'complete' else 'no')
except (FileNotFoundError, json.JSONDecodeError):
    print('no')
PY
)"
if [[ "${cache_complete}" != "yes" ]]; then
  /home/wenning/miniconda3/bin/conda run --no-capture-output -n New_MMM \
    python -u train/train_t2i_trans.py --cache-only \
    --device "cuda:${gpu}" --exp-name "${experiment}_cache" \
    "${shared_args[@]}" --wandb-mode disabled
fi

# Matched to the unaugmented no-shape v4 pilot and the other AMASS arm. The
# augmentation overlay/token cache is the only training factor; all model,
# optimizer, RNG-isolation, sampler and telemetry arguments are identical.
train_args=(
  --device "cuda:${gpu}" --exp-name "${experiment}"
  "${shared_args[@]}"
  --batch-size 64 --total-iter 300000 --diagnostic-stop-iter 50000 \
  --warm-up-iter 1000 --lr 0.0002 --weight-decay 0.000001 \
  --lr-scheduler 150000 --gamma 0.05 \
  --no-contact --no-contact-conditioning --no-shape --no-paper-architecture \
  --training-mask-contract historical_pipeline_paper_kinematic_ratio \
  --training-loss-contract historical_all_valid \
  --optimizer-contract historical_multistep \
  --conditioning-contract historical_mmm_shape_dropout \
  --training-token-length-contract historical_38 \
  --training-validation-contract paper_isolated_common \
  --save-all-validation-checkpoints --pkeep 0.5 \
  --sampler sample_intra_cas --temperature 0.5 --sampling-steps 50 \
  --cfg-scale 3.5 --contact-guidance-scale 1.0 \
  --mm-num-samples 100 --validation-mm-repeats 11 --mm-num-repeats 30 \
  --wandb-mode online
)

resume_json="$(${python_env} tools/resolve_mgt_diagnostic_resume.py \
  --experiment "${experiment}" --stop-step 50000)"
resume_action="$(jq -r '.action' <<<"${resume_json}")"
if [[ "${resume_action}" == "complete" ]]; then
  echo "MGT diagnostic already complete: $(jq -r '.run_directory' <<<"${resume_json}")"
  exit 0
fi
resume_args=()
if [[ "${resume_action}" == "resume" ]]; then
  resume_checkpoint="$(jq -r '.resume_checkpoint' <<<"${resume_json}")"
  resume_args=(--resume-in-place --resume-training-ckpt "${resume_checkpoint}")
elif [[ "${resume_action}" != "new" ]]; then
  echo "unexpected MGT resume action: ${resume_action}" >&2
  exit 2
fi

/home/wenning/miniconda3/bin/conda run --no-capture-output -n New_MMM \
  python -u train/train_t2i_trans.py "${train_args[@]}" "${resume_args[@]}"
