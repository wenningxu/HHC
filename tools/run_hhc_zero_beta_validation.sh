#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

python_bin="/home/wenning/miniconda3/envs/New_MMM/bin/python"
requests="output/interx_official_val_requests/validation.npz"
checkpoint="output/t2i_trans/2026-09-07-11-49-54_hhc_historical_paper_mask_no_mgt_shape_pilot50k_seed114514_v4/model-000300000-0.3973.ckpt"
vq_checkpoint="checkpoints/vq_precleanup_raw12288/net_175000.pth"
ground_truth="$("${python_bin}" -c 'import json; print(json.load(open("output/interx_validation_raw_ground_truth_manifest.json"))["raw_ground_truth"]["path"])')"
output_root="output/hhc_zero_beta_validation_v1"
device="${1:-cpu}"

[[ -s "${requests}" ]] || { echo "missing requests: ${requests}" >&2; exit 1; }
[[ -s "${ground_truth}" ]] || { echo "missing raw ground truth: ${ground_truth}" >&2; exit 1; }
mkdir -p "${output_root}"

for beta_policy in requested zero; do
  generation="${output_root}/${beta_policy}.h5"
  metrics="${output_root}/${beta_policy}.json"

  "${python_bin}" -u tools/export_hhc_official_interx.py \
    --requests "${requests}" --output "${generation}" \
    --checkpoint "${checkpoint}" --vq-checkpoint "${vq_checkpoint}" \
    --vq-contract precleanup_raw12288 --mgt-contract historical_mmm \
    --no-mgt-shape --inference-conditioning-contract paper_text_cfg \
    --stats-dir dataset/interx \
    --beta-policy "${beta_policy}" --no-contact-guidance \
    --contact-timing-source none --device "${device}" --batch-size 32 \
    --repeats 1 --seed 500123 --sampling-steps 50 --temperature 0.5 \
    --cfg-scale 3.5 --resume

  "${python_bin}" -u tools/score_interact2ar_joint_validation.py \
    --requests "${requests}" --raw-ground-truth "${ground_truth}" \
    --generations "${generation}" --output "${metrics}" \
    --device cpu --batch-size 32
done
