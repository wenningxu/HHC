#!/usr/bin/env bash
# Frozen shape-conditioned checkpoint: original sampling versus order noise.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
python_bin=/home/wenning/miniconda3/envs/New_MMM/bin/python
checkpoint=output/t2i_trans/2026-09-02-11-49-08_hhc_historical_plus_paper_mask_ratio_raw12288_300k_seed114514_v2/model-000260000-0.1246.ckpt
requests=output/interx_official_val_requests/validation.npz
gt=output/interx_validation_raw_ground_truth.h5
root=output/hhc_shape_confidence_validation_20260911
device=${1:-cuda:1}
mkdir -p "$root"
common=(--checkpoint "$checkpoint" --vq-checkpoint checkpoints/vq_precleanup_raw12288/net_175000.pth
  --requests "$requests" --vq-contract precleanup_raw12288 --mgt-contract historical_mmm
  --mgt-shape --no-contact-guidance --contact-timing-source none
  --mgt-beta-policy requested --vq-beta-policy requested
  --device "$device" --batch-size 32 --resume --save-token-ids)
for arm in control confidence_all; do
  noise=0
  if [[ "$arm" == confidence_all ]]; then noise=1; fi
  settings=(--temperature 0.5 --cfg-scale 3.5 --sampling-steps 50
    --confidence-noise-scale "$noise" --confidence-noise-scope all
    --inference-conditioning-contract paper_text_cfg)
  "$python_bin" -u tools/export_hhc_official_interx.py "${common[@]}" "${settings[@]}" \
    --output "$root/${arm}_main.h5" --repeats 1 --seed 500123
  "$python_bin" -u tools/score_interact2ar_joint_validation.py --requests "$requests" \
    --raw-ground-truth "$gt" --generations "$root/${arm}_main.h5" \
    --output "$root/${arm}_main_cpu.json" --device cpu --batch-size 32
  "$python_bin" -u tools/score_fair_interx_physical.py --requests "$requests" \
    --generations "$root/${arm}_main.h5" --output "$root/${arm}_physical_per_sample.json" \
    --device cpu --batch-size 32 --floor-policy valid_frames --save-per-sample
  "$python_bin" -u tools/export_hhc_official_interx.py "${common[@]}" "${settings[@]}" \
    --output "$root/${arm}_mm.h5" --indices-key mm_indices --repeats 11 --seed 600123
  "$python_bin" -u tools/score_hhc_validation_mm.py --requests "$requests" \
    --generations "$root/${arm}_mm.h5" --output "$root/${arm}_mm_cpu.json" \
    --device cpu --batch-size 32 --repeats 11 --mgt-shape \
    --mgt-beta-policy requested --vq-beta-policy requested "${settings[@]}"
  echo "completed arm=$arm"
done
