#!/usr/bin/env bash
set -euo pipefail

cd /mnt/8T-HDD/wenning/HHC

run_dir="output/t2i_trans/2026-09-02-11-49-08_hhc_historical_plus_paper_mask_ratio_raw12288_300k_seed114514_v2"
wandb_file="wandb/run-20260902_114917-zzpqa3ag/run-zzpqa3ag.wandb"
selection_dir="output/hhc_historical_plus_paper_mask_ratio_val_candidates"
selection="output/hhc_historical_plus_paper_mask_ratio_selection.json"
vq="checkpoints/vq_precleanup_raw12288/net_175000.pth"
contact_map="checkpoints/vq_precleanup_raw12288_historical_mmm_cache/code_contact_probs.npy"
final_checkpoint="${run_dir}/latest-000300000.ckpt"
model_structure_audit="output/hhc_historical_plus_paper_mask_ratio_model_structure_audit.json"
resume_from="${HHC_FINALIZE_RESUME_FROM:-start}"
if [[ "${resume_from}" != start && "${resume_from}" != selection ]]; then
  echo "HHC_FINALIZE_RESUME_FROM must be start or selection" >&2
  exit 2
fi

if [[ "${resume_from}" == start ]]; then
/home/wenning/miniconda3/bin/conda run --no-capture-output -n New_MMM \
  python -u tools/audit_hhc_sampling_runtime.py \
  --output output/hhc_sampling_runtime_audit.json

/home/wenning/miniconda3/bin/conda run --no-capture-output -n New_MMM \
  python -u tools/audit_lightning_checkpoint.py \
  --checkpoint "${final_checkpoint}" \
  --output output/hhc_historical_plus_paper_mask_ratio_final_checkpoint_audit.json \
  --expected-step 300000 --expected-lr 0.00001 \
  --expected-weight-decay 0.000001

/home/wenning/miniconda3/bin/conda run --no-capture-output -n New_MMM \
  python -u tools/audit_hhc_historical_model_checkpoint_structure.py \
  --checkpoint "${final_checkpoint}" \
  --training-manifest "${run_dir}/training_manifest.json" \
  --sampling-runtime-audit output/hhc_sampling_runtime_audit.json \
  --expected-step 300000 --output "${model_structure_audit}"

/home/wenning/miniconda3/bin/conda run --no-capture-output -n New_MMM \
  python -u tools/audit_wandb_online_metric_history.py \
  --manifest "${run_dir}/training_manifest.json" \
  --metric-profile masked_token_classification --expected-final-step 300000 \
  --output output/hhc_historical_plus_paper_mask_ratio_wandb_history_audit.json

/home/wenning/miniconda3/bin/conda run --no-capture-output -n New_MMM \
  python -u tools/audit_hhc_mask_ratio_runtime_history.py \
  --wandb-file "${wandb_file}" \
  --manifest "${run_dir}/training_manifest.json" \
  --through-step 300000 --telemetry-interval 50 --mean-tolerance 0.01 \
  --output output/hhc_historical_plus_paper_mask_ratio_runtime_history_final_audit.json

/home/wenning/miniconda3/bin/conda run --no-capture-output -n New_MMM \
  python -u tools/promote_interx_validation_request_contract.py
else
  echo "Resuming HHC finalization from official validation selection; prior final-state audits will be validated by the selector, not rerun."
fi

/home/wenning/miniconda3/bin/conda run --no-capture-output -n New_MMM \
  python -u tools/select_hhc_checkpoint_on_official_interx_val.py \
  --run-dir "${run_dir}" --output-dir "${selection_dir}" \
  --model-structure-audit "${model_structure_audit}" \
  --output "${selection}" --device cuda:1 --batch-size 32

checkpoint="$(/home/wenning/miniconda3/envs/New_MMM/bin/python -c \
  "import json; print(json.load(open('${selection}'))['selected_checkpoint'])")"

common=(
  --method hhc --checkpoint "${checkpoint}" --vq-checkpoint "${vq}"
  --hhc-vq-contract precleanup_raw12288 --hhc-mgt-contract historical_mmm
  --hhc-inference-conditioning-contract paper_text_cfg
  --hhc-contact-map "${contact_map}"
  --hhc-contact-map-contract paper_formula_train_only
  --selection-manifest "${selection}"
  --requests-dir output/fair_interx_requests
  --requests-manifest output/fair_interx_requests/manifest.json
  --output-root output/fair_eval --gpu 1 --batch-size 32
  --sampling-steps 50 --temperature 0.5 --cfg-scale 3.5
  --contact-guidance-scale 1.0 --start-replication 0 --end-replication 20
)

/home/wenning/miniconda3/bin/conda run --no-capture-output -n New_MMM \
  python -u tools/run_fair_interx_method.py "${common[@]}" \
  --label hhc_historical_plus_paper_mask_ratio_contact_common50 \
  --contact-guidance --hhc-contact-timing-source request_oracle

/home/wenning/miniconda3/bin/conda run --no-capture-output -n New_MMM \
  python -u tools/run_fair_interx_method.py "${common[@]}" \
  --label hhc_historical_plus_paper_mask_ratio_no_contact_common50 \
  --no-contact-guidance --hhc-contact-timing-source none

bash tools/finalize_current_interx_if_ready.sh 1
