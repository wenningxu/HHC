#!/usr/bin/env bash
set -euo pipefail

cd /mnt/8T-HDD/wenning/HHC

echo "Archived shape-training launcher: its overlays and prerequisites belong to the invalid custom optimizer. Use the replacement launcher produced after the original-MMM pilot25 gate." >&2
exit 2

if [[ $# -ne 2 ]] || [[ ! "$1" =~ ^[0-9]+$ ]]; then
  echo "usage: $0 GPU_INDEX random|optimized" >&2
  exit 2
fi
gpu="$1"
arm="$2"
if [[ "${arm}" != random && "${arm}" != optimized ]]; then
  echo "arm must be random or optimized" >&2
  exit 2
fi
device="cuda:${gpu}"

conda() {
  /home/wenning/miniconda3/bin/conda "$@"
}

# Shape generalization is the last Inter-X stage.  Do not create a waiter or
# start either training arm before every main baseline has finished.
required_json=(
  output/hhc_historical_plus_paper_mask_ratio_selection.json
  output/external_baseline_reproduction_audit.json
  output/current_interx_main_table.json
  output/current_interx_optimization_backend_contract.json
  output/optimization_historical_paper_mask_rep0/formal_manifest.json
  output/current_interx_mesh_physical/manifest.json
  output/current_interx_qualitative/manifest.json
  output/fair_eval/hhc_historical_paper_mask_contact_timing_ablation_rep0/summary.json
  output/fair_eval/hhc_historical_paper_mask_sampling_pareto_no_timing_rep0/summary.json
  output/fair_eval/hhc_historical_plus_paper_mask_ratio_contact_common50/summary.json
  output/fair_eval/hhc_historical_plus_paper_mask_ratio_no_contact_common50/summary.json
  output/timotion_interx_val_selection.json
  output/fair_eval/timotion_transformer_lpa_interx_retrained_fair/summary.json
  output/fair_eval/coshmdm_interx_retrained_2daycap_seed123/summary.json
  output/fair_eval/intermask_official_exact_common50/summary.json
  output/fair_eval/interact2ar_official_fair_no_gt_first_frame/summary.json
  output/interact2ar_author_release_calibration/summary.json
  output/interact2ar_author_release_calibration/audit.json
  output/interact2ar_joint_scorer_parity_audit.json
  output/interact2ar_joint_evaluator_suite/pre_shape_summary.json
)
for required in "${required_json[@]}"; do
  if [[ ! -s "${required}" ]]; then
    echo "Shape training is last; prerequisite is incomplete: ${required}" >&2
    exit 2
  fi
done

used_mib="$(nvidia-smi --id="${gpu}" --query-gpu=memory.used \
  --format=csv,noheader,nounits | tr -d ' ')"
if [[ ! "${used_mib}" =~ ^[0-9]+$ ]] || (( used_mib > 1024 )); then
  echo "GPU ${gpu} is not free (memory.used=${used_mib:-unknown} MiB)." >&2
  exit 2
fi

if [[ "${arm}" == random ]]; then
  overlay="dataset/interx_shape_aug_random_seed123_v3"
else
  overlay="dataset/interx_shape_aug_optimized_nearest_vertex_seed123_v3"
fi
augmentation_manifest="${overlay}/materialization_manifest.json"
if [[ ! -s "${augmentation_manifest}" ]]; then
  echo "Missing completed ${arm} augmentation overlay: ${augmentation_manifest}" >&2
  exit 2
fi

primary_selection="output/hhc_historical_plus_paper_mask_ratio_selection.json"
primary_training_manifest="$(/home/wenning/miniconda3/envs/New_MMM/bin/python -c \
  "import json; print(json.load(open('${primary_selection}'))['artifacts']['training_manifest']['path'])")"
weights_only_reference="output/shape_capacity/model-000300000-0.0000.ckpt"
weights_only_reference_manifest="output/shape_capacity/lightning_weights_only_reference.json"
if [[ ! -s "${weights_only_reference}" || ! -s "${weights_only_reference_manifest}" ]]; then
  echo "Missing measured weights-only capacity reference or manifest." >&2
  exit 2
fi
conda run --no-capture-output -n New_MMM python -u \
  tools/audit_shape_augmentation_disk_capacity.py \
  --filesystem-root . \
  --reference-label output/fair_eval/hhc_historical_plus_paper_mask_ratio_no_contact_common50 \
  --reference-training-run "$(dirname "${primary_training_manifest}")" \
  --arms 1 --labels-per-arm 2 \
  --validation-checkpoints-per-arm 30 \
  --rolling-full-checkpoints-per-arm 2 \
  --reference-validation-checkpoint "${weights_only_reference}" \
  --reference-validation-manifest "${weights_only_reference_manifest}" \
  --free-space-after-queue-gib 40 \
  --output "output/interx_shape_augmentation_${arm}_disk_capacity_audit_v3.json"

vq="checkpoints/vq_precleanup_raw12288/net_175000.pth"
cache_root="checkpoints/vq_precleanup_raw12288_shapeaug_${arm}_nearest_v3_cache"
experiment="hhc_historical_plus_paper_mask_ratio_shapeaug_${arm}_raw12288_300k_seed114514_v1"
historical_codebook="/home/wenning/projects/MMM/checkpoints/vq/codebook"
historical_contact="/home/wenning/projects/MMM/checkpoints/vq/contact_token_label"
hybrid_preflight="output/${experiment}_hybrid_cache_preflight.json"
state="output/${experiment}_training_state.json"
selection_dir="output/${experiment}_val_candidates"
selection="output/${experiment}_selection.json"

conda run --no-capture-output -n New_MMM python -u \
  tools/audit_hybrid_shape_cache_preflight.py \
  --overlay-root "${overlay}" \
  --historical-code-dir "${historical_codebook}" \
  --historical-contact-dir "${historical_contact}" \
  --output "${hybrid_preflight}"

resolve_state() {
  conda run --no-capture-output -n New_MMM python -u \
    tools/resolve_mgt_training_state.py \
    --run-glob "output/t2i_trans/*_${experiment}" \
    --experiment "${experiment}" --expected-final-step 300000 \
    --output "${state}"
}

resolve_state
training_status="$(/home/wenning/miniconda3/envs/New_MMM/bin/python -c \
  "import json; print(json.load(open('${state}'))['status'])")"
resume_args=()
if [[ "${training_status}" == resume ]]; then
  resume_checkpoint="$(/home/wenning/miniconda3/envs/New_MMM/bin/python -c \
    "import json; print(json.load(open('${state}'))['resume_checkpoint'])")"
  resume_args=(--resume-training-ckpt "${resume_checkpoint}" --resume-in-place)
fi

if [[ "${training_status}" == new ]]; then
  conda run --no-capture-output -n New_MMM python -u \
    train/train_t2i_trans.py --cache-only \
    --device "${device}" --exp-name "${experiment}_cache" \
    --vq-checkpoint "${vq}" --cache-root "${cache_root}" \
    --codebook-dir "${cache_root}/codebook" \
    --token-cache-contract hybrid_historical_original_plus_deterministic_augmented_v1 \
    --historical-codebook-source /home/wenning/projects/MMM/checkpoints/vq/codebook \
    --historical-contact-source /home/wenning/projects/MMM/checkpoints/vq/contact_token_label \
    --nb-code 12288 --code-dim 256 --shape-condition-dim 8 --width 512 \
    --embed-dim-gpt 1024 --n-head-gpt 16 --num-layers 9 --num-local-layer 2 \
    --train-asset-root "${overlay}" \
    --augmentation-manifest "${augmentation_manifest}" \
    --train-split-file "${overlay}/splits/train.txt" \
    --val-split-file dataset/interx/splits/val.txt \
    --all-split-file "${overlay}/splits/all.txt" \
    --train-motion-file "${overlay}/motions/train.h5" \
    --val-motion-file dataset/interx/motions/val.h5 \
    --all-motion-file "${overlay}/motions/all.h5" \
    --stats-dir "${overlay}" --stats-prefix Interx \
    --seed 114514 --wandb-mode disabled
fi

if [[ "${training_status}" != complete ]]; then
  # Exact strong historical numerical pipeline, with only the paper's
  # asymmetric kinematic mask-ratio rule changed, as in the primary v2 run.
  conda run --no-capture-output -n New_MMM python -u \
    train/train_t2i_trans.py \
    --device "${device}" --exp-name "${experiment}" \
    --vq-checkpoint "${vq}" --cache-root "${cache_root}" \
    --codebook-dir "${cache_root}/codebook" \
    --token-cache-contract hybrid_historical_original_plus_deterministic_augmented_v1 \
    --historical-codebook-source /home/wenning/projects/MMM/checkpoints/vq/codebook \
    --historical-contact-source /home/wenning/projects/MMM/checkpoints/vq/contact_token_label \
    --nb-code 12288 --code-dim 256 --shape-condition-dim 8 --width 512 \
    --embed-dim-gpt 1024 --n-head-gpt 16 --num-layers 9 --num-local-layer 2 \
    --train-asset-root "${overlay}" \
    --augmentation-manifest "${augmentation_manifest}" \
    --train-split-file "${overlay}/splits/train.txt" \
    --val-split-file dataset/interx/splits/val.txt \
    --all-split-file "${overlay}/splits/all.txt" \
    --train-motion-file "${overlay}/motions/train.h5" \
    --val-motion-file dataset/interx/motions/val.h5 \
    --all-motion-file "${overlay}/motions/all.h5" \
    --stats-dir "${overlay}" --stats-prefix Interx \
    --batch-size 64 --total-iter 300000 --warm-up-iter 1000 \
    --lr 0.0002 --weight-decay 0.000001 --lr-scheduler 150000 --gamma 0.05 \
    --no-contact --no-contact-conditioning --shape --no-paper-architecture \
    --training-mask-contract historical_pipeline_paper_kinematic_ratio \
    --training-loss-contract historical_all_valid \
    --optimizer-contract historical_multistep \
    --conditioning-contract historical_mmm_shape_dropout \
    --training-token-length-contract historical_38 \
    --training-validation-contract historical_test_side_effect \
    --save-all-validation-checkpoints --pkeep 0.5 \
    --sampler sample_intra_cas --temperature 0.0 --sampling-steps 10 \
    --cfg-scale 3.5 --contact-guidance-scale 1.0 \
    --wandb-mode online --seed 114514 "${resume_args[@]}"
fi

resolve_state
run_dir="$(/home/wenning/miniconda3/envs/New_MMM/bin/python -c \
  "import json; print(json.load(open('${state}'))['active_run']['directory'])")"
contact_map="${cache_root}/code_contact_probs.npy"
final_checkpoint="${run_dir}/latest-000300000.ckpt"
final_checkpoint_audit="output/${experiment}_final_checkpoint_audit.json"
wandb_history_audit="output/${experiment}_wandb_history_audit.json"
model_structure_audit="output/${experiment}_model_structure_audit.json"
conda run --no-capture-output -n New_MMM python -u \
  tools/audit_hhc_sampling_runtime.py \
  --output output/hhc_sampling_runtime_audit.json
conda run --no-capture-output -n New_MMM python -u \
  tools/audit_lightning_checkpoint.py \
  --checkpoint "${final_checkpoint}" --output "${final_checkpoint_audit}" \
  --expected-step 300000 --expected-lr 0.00001 \
  --expected-weight-decay 0.000001
conda run --no-capture-output -n New_MMM python -u \
  tools/audit_hhc_historical_model_checkpoint_structure.py \
  --checkpoint "${final_checkpoint}" \
  --training-manifest "${run_dir}/training_manifest.json" \
  --sampling-runtime-audit output/hhc_sampling_runtime_audit.json \
  --expected-step 300000 --output "${model_structure_audit}"
conda run --no-capture-output -n New_MMM python -u \
  tools/audit_wandb_online_metric_history.py \
  --manifest "${run_dir}/training_manifest.json" \
  --metric-profile masked_token_classification --expected-final-step 300000 \
  --output "${wandb_history_audit}"
conda run --no-capture-output -n New_MMM python -u \
  tools/promote_interx_validation_request_contract.py
conda run --no-capture-output -n New_MMM python -u \
  tools/select_hhc_checkpoint_on_official_interx_val.py \
  --run-dir "${run_dir}" --contact-map "${contact_map}" \
  --final-checkpoint-audit "${final_checkpoint_audit}" \
  --wandb-history-audit "${wandb_history_audit}" \
  --model-structure-audit "${model_structure_audit}" \
  --output-dir "${selection_dir}" --output "${selection}" \
  --device "${device}" --batch-size 32

checkpoint="$(/home/wenning/miniconda3/envs/New_MMM/bin/python -c \
  "import json; print(json.load(open('${selection}'))['selected_checkpoint'])")"
common_eval=(
  --method hhc --checkpoint "${checkpoint}" --vq-checkpoint "${vq}"
  --hhc-vq-contract precleanup_raw12288 --hhc-mgt-contract historical_mmm
  --hhc-inference-conditioning-contract paper_text_cfg
  --hhc-stats-dir "${overlay}" --hhc-stats-prefix Interx
  --hhc-contact-map "${contact_map}"
  --hhc-contact-map-contract paper_formula_train_only
  --selection-manifest "${selection}" --gpu "${gpu}" --batch-size 32
  --sampling-steps 50 --temperature 0.5 --cfg-scale 3.5
  --contact-guidance-scale 1.0 --start-replication 0 --end-replication 20
)
conda run --no-capture-output -n New_MMM python -u \
  tools/run_fair_interx_method.py "${common_eval[@]}" \
  --label "hhc_historical_paper_mask_shapeaug_${arm}_no_contact_common50" \
  --no-contact-guidance --hhc-contact-timing-source none
conda run --no-capture-output -n New_MMM python -u \
  tools/run_fair_interx_method.py "${common_eval[@]}" \
  --label "hhc_historical_paper_mask_shapeaug_${arm}_contact_common50" \
  --contact-guidance --hhc-contact-timing-source request_oracle
