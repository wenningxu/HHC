#!/usr/bin/env bash
set -euo pipefail

repo=/mnt/8T-HDD/wenning/HHC
run_name=2026-09-09-17-39-10_hhc_historical_paper_mask_no_shape_masked_only_pilot50k_seed114514_v1
run_dir="${repo}/output/t2i_trans/${run_name}"
checkpoint="${run_dir}/latest-000050000.ckpt"
comparison="${repo}/output/hhc_no_shape_masked_only_pilot50k_comparison.json"

if [[ $# -ne 1 || ! "$1" =~ ^[0-9]+$ ]]; then
  echo "usage: $0 GPU_INDEX" >&2
  exit 2
fi
gpu="$1"

# Never turn a diagnostic into a full-budget run merely because it exited.
# The matched validation comparison is the sole authorization gate.
if [[ ! -s "${comparison}" ]]; then
  echo "masked-only continuation gate is absent: ${comparison}" >&2
  exit 2
fi
/home/wenning/miniconda3/envs/New_MMM/bin/python - \
  "${comparison}" "${checkpoint}" <<'PY'
import json
import pathlib
import sys

comparison = json.load(open(sys.argv[1], encoding="utf-8"))
if comparison.get("schema") != "hhc_no_shape_masked_only_pilot_comparison_v1":
    raise SystemExit("wrong masked-only comparison schema")
if comparison.get("status") != "complete":
    raise SystemExit("masked-only comparison is incomplete")
if comparison.get("decision") != "continue_same_masked_only_run_to_300k":
    raise SystemExit(f"masked-only continuation rejected: {comparison.get('decision')}")
if not all(comparison.get("promotion_screen", {}).values()):
    raise SystemExit("masked-only continuation lacks a fully passing promotion screen")
if not pathlib.Path(sys.argv[2]).is_file():
    raise SystemExit("50k atomic recovery checkpoint is missing")
PY

used_mib="$(nvidia-smi --id="${gpu}" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')"
if [[ ! "${used_mib}" =~ ^[0-9]+$ ]] || (( used_mib > 1024 )); then
  echo "GPU ${gpu} is not free (memory.used=${used_mib:-unknown} MiB)" >&2
  exit 2
fi

if [[ ! -s "${run_dir}/training_manifest_at_50k.json" ]]; then
  cp -p "${run_dir}/training_manifest.json" "${run_dir}/training_manifest_at_50k.json"
fi

cd "${repo}"
exec /home/wenning/miniconda3/bin/conda run --no-capture-output -n New_MMM \
  python -u train/train_t2i_trans.py \
  --device "cuda:${gpu}" \
  --exp-name hhc_historical_paper_mask_no_shape_masked_only_pilot50k_seed114514_v1 \
  --vq-checkpoint checkpoints/vq_precleanup_raw12288/net_175000.pth \
  --cache-root checkpoints/vq_precleanup_raw12288_historical_mmm_cache \
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
  --training-loss-contract paper_masked_only \
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
