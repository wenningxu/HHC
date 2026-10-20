#!/usr/bin/env bash
set -euo pipefail

repo=/mnt/8T-HDD/wenning/HHC
run_name=2026-09-07-11-49-54_hhc_historical_paper_mask_no_mgt_shape_pilot50k_seed114514_v4
run_dir="${repo}/output/t2i_trans/${run_name}"
source_root="${run_dir}/resume_source_tree"
recovery="${repo}/output/hhc_no_shape_v4_launch_source_recovery.json"
source_tree_audit="${repo}/output/hhc_no_shape_v4_resume_source_tree.json"
checkpoint="${source_root}/output/t2i_trans/${run_name}/latest-000050000.ckpt"

if [[ $# -ne 1 || ! "$1" =~ ^[0-9]+$ ]]; then
  echo "usage: $0 GPU_INDEX" >&2
  exit 2
fi
gpu="$1"
used_mib="$(nvidia-smi --id="${gpu}" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')"
if [[ ! "${used_mib}" =~ ^[0-9]+$ ]] || (( used_mib > 1024 )); then
  echo "GPU ${gpu} is not free (memory.used=${used_mib:-unknown} MiB)" >&2
  exit 2
fi

/home/wenning/miniconda3/envs/New_MMM/bin/python - "${recovery}" "${source_tree_audit}" <<'PY'
import json, sys
for path in sys.argv[1:]:
    payload = json.load(open(path))
    assert payload['status'] == 'pass', path
assert json.load(open(sys.argv[1]))['all_snapshot_hashes_match_training_manifest'] is True
assert json.load(open(sys.argv[2]))['all_bound_hashes_match_training_manifest'] is True
PY

mkdir -p "${source_root}/dataset"
for mapping in \
  "${source_root}/output:${repo}/output" \
  "${source_root}/checkpoints:${repo}/checkpoints" \
  "${source_root}/body_models:${repo}/body_models" \
  "${source_root}/dataset/interx:${repo}/dataset/interx"; do
  link="${mapping%%:*}"
  target="${mapping#*:}"
  if [[ -L "${link}" ]]; then
    [[ "$(readlink "${link}")" == "${target}" ]] || {
      echo "resume source-tree link drifted: ${link}" >&2
      exit 2
    }
  elif [[ -e "${link}" ]]; then
    echo "resume source-tree path exists but is not the expected link: ${link}" >&2
    exit 2
  else
    ln -s "${target}" "${link}"
  fi
done

if [[ ! -s "${run_dir}/training_manifest_at_50k.json" ]]; then
  cp -p "${run_dir}/training_manifest.json" "${run_dir}/training_manifest_at_50k.json"
fi
test -s "${checkpoint}"

cd "${source_root}"
exec /home/wenning/miniconda3/bin/conda run --no-capture-output -n New_MMM \
  python -u train/train_t2i_trans.py \
  --device "cuda:${gpu}" \
  --exp-name hhc_historical_paper_mask_no_mgt_shape_pilot50k_seed114514_v4 \
  --vq-checkpoint checkpoints/vq_precleanup_raw12288/net_175000.pth \
  --cache-root "${repo}/checkpoints/vq_precleanup_raw12288_historical_mmm_cache" \
  --codebook-dir checkpoints/vq_precleanup_raw12288_historical_mmm_cache/codebook \
  --token-cache-contract historical_mmm_exact \
  --historical-codebook-source /home/wenning/projects/MMM/checkpoints/vq/codebook \
  --historical-contact-source /home/wenning/projects/MMM/checkpoints/vq/contact_token_label \
  --nb-code 12288 --code-dim 256 --shape-condition-dim 8 --width 512 \
  --embed-dim-gpt 1024 --n-head-gpt 16 --num-layers 9 --num-local-layer 2 \
  --batch-size 64 --total-iter 300000 \
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
  --wandb-mode online --seed 114514 \
  --resume-in-place --resume-training-ckpt "${checkpoint}"
