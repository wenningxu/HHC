#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
python_bin=/home/wenning/miniconda3/envs/New_MMM/bin/python
requests=output/interx_official_val_requests/validation.npz
checkpoint=output/t2i_trans/2026-09-07-11-49-54_hhc_historical_paper_mask_no_mgt_shape_pilot50k_seed114514_v4/model-000300000-0.3973.ckpt
vq_checkpoint=checkpoints/vq_precleanup_raw12288/net_175000.pth
root=output/hhc_zero_beta_validation_v1
device=${1:-cpu}
mkdir -p "$root"
for beta in requested zero; do
  generation="$root/${beta}_mm.h5"
  "$python_bin" -u tools/export_hhc_official_interx.py --requests "$requests" --output "$generation" --checkpoint "$checkpoint" --vq-checkpoint "$vq_checkpoint" --vq-contract precleanup_raw12288 --mgt-contract historical_mmm --no-mgt-shape --inference-conditioning-contract paper_text_cfg --stats-dir dataset/interx --beta-policy "$beta" --no-contact-guidance --contact-timing-source none --indices-key mm_indices --device "$device" --batch-size 32 --repeats 11 --seed 600123 --sampling-steps 50 --temperature 0.5 --cfg-scale 3.5 --resume
  "$python_bin" -u tools/score_hhc_validation_mm.py --requests "$requests" --generations "$generation" --output "$root/${beta}_mm.json" --repeats 11 --batch-size 32 --device cpu
done
