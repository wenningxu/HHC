#!/usr/bin/env bash
# September 10 user-authorized sampling experiments; old common-50 is control.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
python_bin=/home/wenning/miniconda3/envs/New_MMM/bin/python
checkpoint=output/t2i_trans/2026-09-07-11-49-54_hhc_historical_paper_mask_no_mgt_shape_pilot50k_seed114514_v4/model-000300000-0.3973.ckpt
requests=output/interx_official_val_requests/validation.npz
gt="$($python_bin -c 'import json; print(json.load(open("output/interx_validation_raw_ground_truth_manifest.json"))["raw_ground_truth"]["path"])')"
device=${1:-cuda:0}
case "${2:-calibration}" in
  calibration)
    root=output/hhc_no_shape_sampling_validation_20260910
    arms=(control cfg2p5 anneal0p8to0p5 temperature0p8 matched_text_dropout) ;;
  confidence)
    root=output/hhc_no_shape_confidence_validation_20260910
    arms=(control confidence_all confidence_root zero_beta) ;;
  budget)
    # Reuse calibration's same-device control; do not compare to another model.
    root=output/hhc_no_shape_sampling_validation_20260910
    arms=(steps20 steps10) ;;
  noise_strength)
    root=output/hhc_no_shape_sampling_validation_20260910
    arms=(noise0p5 noise1_temp0 noise0p25) ;;
  root_strength)
    root=output/hhc_no_shape_confidence_validation_20260910
    arms=(root_noise0p5 root_noise0p25) ;;
  log_odds)
    root=output/hhc_no_shape_confidence_validation_20260910
    arms=(log_odds log_odds_noise1) ;;
  partwise)
    root=output/hhc_no_shape_sampling_validation_20260910
    arms=(partwise partwise_noise1) ;;
  *) echo 'unknown sampling grid' >&2; exit 2 ;;
esac
mkdir -p "$root"
common=(--checkpoint "$checkpoint" --vq-checkpoint checkpoints/vq_precleanup_raw12288/net_175000.pth
  --requests "$requests" --vq-contract precleanup_raw12288 --mgt-contract historical_mmm
  --no-mgt-shape --no-contact-guidance --contact-timing-source none
  --device "$device" --batch-size 32 --resume)
case "${2:-calibration}" in
  noise_strength|root_strength|log_odds|partwise) common+=(--save-token-ids) ;;
esac
for arm in "${arms[@]}"; do
  temperature=0.5 cfg=3.5 steps=50 contract=paper_text_cfg
  noise=0 scope=all beta=requested
  ranking=probability schedule=interleaved
  annealing=()
  case "$arm" in
    cfg2p5) cfg=2.5 ;;
    anneal0p8to0p5) temperature=0.8; annealing=(--temperature-end 0.5) ;;
    temperature0p8) temperature=0.8 ;;
    matched_text_dropout) contract=historical_mmm_shape_dropout ;;
    confidence_all) noise=1 ;;
    confidence_root) noise=1; scope=root ;;
    zero_beta) beta=zero ;;
    steps20) steps=20 ;;
    steps10) steps=10 ;;
    noise0p5) noise=0.5 ;;
    noise0p25) noise=0.25 ;;
    noise1_temp0) noise=1; temperature=0 ;;
    root_noise0p5) noise=0.5; scope=root ;;
    root_noise0p25) noise=0.25; scope=root ;;
    log_odds) ranking=log_odds ;;
    log_odds_noise1) ranking=log_odds; noise=1 ;;
    partwise) schedule=partwise ;;
    partwise_noise1) schedule=partwise; noise=1 ;;
  esac
  settings=(--temperature "$temperature" --cfg-scale "$cfg" --sampling-steps "$steps"
    --confidence-noise-scale "$noise" --confidence-noise-scope "$scope"
    --confidence-ranking "$ranking" --channel-schedule "$schedule"
    --inference-conditioning-contract "$contract" "${annealing[@]}")
  start=$SECONDS
  "$python_bin" -u tools/export_hhc_official_interx.py "${common[@]}" "${settings[@]}" \
    --output "$root/${arm}_main.h5" --repeats 1 --seed 500123 --beta-policy "$beta"
  "$python_bin" -u tools/score_interact2ar_joint_validation.py --requests "$requests" \
    --raw-ground-truth "$gt" --generations "$root/${arm}_main.h5" \
    --output "$root/${arm}_main.json" --device "$device" --batch-size 32
  "$python_bin" -u tools/export_hhc_official_interx.py "${common[@]}" "${settings[@]}" \
    --output "$root/${arm}_mm.h5" --indices-key mm_indices --repeats 11 --seed 600123 --beta-policy "$beta"
  "$python_bin" -u tools/score_hhc_validation_mm.py --requests "$requests" \
    --generations "$root/${arm}_mm.h5" --output "$root/${arm}_mm.json" \
    --device "$device" --repeats 11 "${settings[@]}"
  echo "completed arm=$arm wall_seconds=$((SECONDS-start)) (generation+scoring; resume excludes prior work)"
done
