#!/usr/bin/env bash
set -euo pipefail

output_dir="output/hhc_repredict_validation_diagnostic"
generation="${output_dir}/repredict_50step_temp0p5_cfg3p5_contact.h5"
metrics="${output_dir}/repredict_50step_temp0p5_cfg3p5_contact.json"
baseline="${output_dir}/baseline_50step_temp0p5_cfg3p5_contact_cpu.json"

mkdir -p "${output_dir}"

if [[ ! -s "${baseline}" ]]; then
  conda run --no-capture-output -n New_MMM python -u \
    tools/score_fair_interx_main.py \
    --requests output/hhc_mgt_validation_bridge_requests/validation.npz \
    --main-generations output/hhc_validation_sampler_calibration/exact_order_temp0p5_cfg3p5.h5 \
    --output "${baseline}" --device cpu --batch-size 32
fi

if [[ ! -s "${generation}" ]]; then
  conda run --no-capture-output -n New_MMM python -u \
    tools/export_hhc_repredict_diagnostic.py \
    --requests output/hhc_mgt_validation_bridge_requests/validation.npz \
    --output "${generation}" \
    --checkpoint output/t2i_trans/2026-08-30-16-08-07_hhc_formal_raw12288_repaired_mgt_300k_seed123_v1/model-000060000-0.6364.ckpt \
    --vq-checkpoint checkpoints/vq_precleanup_raw12288/net_175000.pth \
    --vq-contract precleanup_raw12288 --mgt-contract current \
    --inference-conditioning-contract paper_text_cfg \
    --contact-map checkpoints/vq_precleanup_raw12288_formal_mgt_v1/code_contact_probs.npy \
    --contact-map-contract paper_formula_train_only \
    --device cuda:1 --batch-size 32 --repeats 1 --seed 500123 \
    --contact-guidance --contact-timing-source request_oracle \
    --sampling-steps 50 --temperature 0.5 --cfg-scale 3.5 \
    --contact-guidance-scale 1.0
fi

conda run --no-capture-output -n New_MMM python -u \
  tools/score_fair_interx_main.py \
  --requests output/hhc_mgt_validation_bridge_requests/validation.npz \
  --main-generations "${generation}" --output "${metrics}" \
  --device cpu --batch-size 32

conda run --no-capture-output -n New_MMM python -u \
  tools/compare_hhc_repredict_validation.py \
  --baseline "${baseline}" --repredict "${metrics}" \
  --output "${output_dir}/comparison.json"
