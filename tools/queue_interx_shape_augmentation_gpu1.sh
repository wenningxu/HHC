#!/usr/bin/env bash
set -euo pipefail

cd /mnt/8T-HDD/wenning/HHC

# Historical queue retained for provenance only.  It contains the superseded
# VolumetricSMPL workflow and a polling loop, so executing it would violate the
# current last-stage/nearest-vertex contract.
echo "Archived shape queue: use tools/run_interx_shape_augmentation_nearest_last_stage.sh after all Inter-X prerequisites complete." >&2
exit 2

if [[ -n "${PYTORCH_CUDA_ALLOC_CONF:-}" ]] \
    && [[ "${PYTORCH_CUDA_ALLOC_CONF}" != "expandable_segments:True" ]]; then
  echo "Unexpected PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF}"
  exit 2
fi
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

conda() {
  /home/wenning/miniconda3/bin/conda "$@"
}

# Asset generation is safe independently, but this queue also launches two
# formal MGT arms.  Keep the whole chain behind the repaired-MGT hard gate.
conda run --no-capture-output -n New_MMM python -u \
  tools/require_mgt_repair_gate.py \
  --gate output/mgt_repair_gate.json

conda run --no-capture-output -n New_MMM python -u \
  tools/audit_shape_augmentation_launch_contract.py

conda run --no-capture-output -n New_MMM python -u \
  tools/audit_shape_root_tokenizer_wiring.py

conda run --no-capture-output -n New_MMM python -u \
  tools/audit_wandb_online_runtime.py \
  --output output/interx_shape_augmentation_wandb_runtime_audit.json

conda run --no-capture-output -n New_MMM python -u \
  tools/audit_online_wandb_training_contract.py \
  --launcher tools/queue_interx_shape_augmentation_gpu1.sh \
  --training-source train/train_t2i_trans.py \
  --output output/interx_shape_augmentation_online_wandb_audit.json

formal_selection="output/hhc_formal_raw12288_repaired_mgt_selection.json"
formal_no_contact="output/fair_eval/hhc_formal_raw12288_repaired_mgt_common50_no_contact"
formal_contact="output/fair_eval/hhc_formal_raw12288_repaired_mgt_common50_contact"
for required in \
  "${formal_selection}" \
  "${formal_no_contact}/summary.json" \
  "${formal_contact}/summary.json"
do
  if [[ ! -s "${required}" ]]; then
    echo "Formal repaired-HHC baseline is incomplete: ${required}"
    exit 2
  fi
done

contact_dir="dataset/interx/contact_pairs_codex_luna_strict_v3_train"
contact_manifest="${contact_dir}/_extraction_manifest.json"
plan="configs/interx_shape_augmentation/timing_covered_seed123_v2/plan_manifest.json"
vq_checkpoint="checkpoints/vq_precleanup_raw12288/net_175000.pth"

# Stay in the existing GPU-1 dependency chain.  Do not create a second waiter
# terminal while the ChatGPT-authenticated full-train annotation is running.
while true; do
  annotation_status="missing"
  if [[ -s "${contact_manifest}" ]]; then
    annotation_status="$(/home/wenning/miniconda3/envs/New_MMM/bin/python -c \
      "import json; print(json.load(open('${contact_manifest}'))['status'])")"
  fi
  if [[ "${annotation_status}" == complete ]]; then
    break
  fi
  if [[ "${annotation_status}" == incomplete ]] \
      && ! pgrep -f '(extract_interx_contacts_codex|resume_interx_contacts_codex_preserve_failures).py.*contact_pairs_codex_luna_strict_v3_train' >/dev/null 2>&1; then
    echo 'Luna train annotation stopped incomplete; refusing shape augmentation.'
    exit 2
  fi
  sleep 60
done

conda run --no-capture-output -n New_MMM python -u \
  tools/audit_interx_codex_contact_cache.py \
  --pairs-dir "${contact_dir}" \
  --split dataset/interx/splits/train.txt \
  --output output/interx_luna_train_cache_audit.json

conda run --no-capture-output -n New_MMM python -u \
  tools/audit_interx_contact_timing_consistency.py \
  --pairs-dir "${contact_dir}" \
  --output output/interx_luna_train_timing_consistency_audit.json

conda run --no-capture-output -n New_MMM python -u \
  tools/audit_shared_volumetric_backend_equivalence.py

# The original development default added the MANO mean hand pose in the SDF
# branch.  Bind the corrected flat-hand geometry and the only promoted serial
# query chunk before creating or scoring any optimized asset.
conda run --no-capture-output -n New_MMM python -u \
  tools/require_interx_volumetric_smpl_contract.py \
  --query-chunk 10475 \
  --output output/interx_volumetric_smpl_contract_gate.json

conda run --no-capture-output -n New_MMM python -u \
  tools/run_interx_shape_augmentation.py \
  --plan "${plan}" --contact-dir "${contact_dir}" \
  --output-dir output/interx_shape_aug_preflight_seed123_v2 \
  --mode optimized --preflight-only --device cuda:1

# Matched, preregistered 25-sample pilot: the same timing-positive plan rows,
# donor shapes and seeds differ only by optimization on/off.
pilot_random="output/interx_shape_aug_pilot_random_seed20260828"
pilot_optimized="output/interx_shape_aug_pilot_optimized_flathand_query10475_seed20260828"
if [[ ! -s "${pilot_random}/_completion_manifest.json" ]]; then
  conda run --no-capture-output -n New_MMM python -u \
    tools/run_interx_shape_augmentation.py \
    --plan "${plan}" --contact-dir "${contact_dir}" \
    --output-dir "${pilot_random}" --mode random_shape_only \
    --device cuda:1 --query-chunk 512 --seed 1234 \
    --selection-seed 20260828 --positive-timing-only --limit 25
fi
conda run --no-capture-output -n New_MMM python -u \
  tools/audit_random_shape_augmentation_completion.py \
  --run-dir "${pilot_random}" --plan "${plan}" --expected-samples 25 \
  --output "${pilot_random}/completion_content_audit.json"
conda run --no-capture-output -n New_MMM python -u \
  tools/run_interx_shape_augmentation.py \
  --plan "${plan}" --contact-dir "${contact_dir}" \
  --output-dir "${pilot_optimized}" --mode optimized \
  --device cuda:1 --query-chunk 10475 --seed 1234 \
  --selection-seed 20260828 --positive-timing-only --limit 25

conda run --no-capture-output -n New_MMM python -u \
  tools/analyze_shape_augmentation_runtime.py \
  --run-dir "${pilot_optimized}" --plan "${plan}" \
  --output "${pilot_optimized}/runtime_analysis.json" \
  --bootstrap-replicates 10000 --seed 20260901

pilot_random_physical="output/interx_shape_aug_pilot_random_flathand_physical_seed20260828"
pilot_optimized_physical="output/interx_shape_aug_pilot_optimized_flathand_query10475_physical_seed20260828"
conda run --no-capture-output -n New_MMM python -u \
  tools/score_volumetric_smpl_npz.py \
  --folder "${pilot_random}" --output-dir "${pilot_random_physical}" \
  --device cuda:1 --query-chunk 10475 --seed 1234
conda run --no-capture-output -n New_MMM python -u \
  tools/score_volumetric_smpl_npz.py \
  --folder "${pilot_optimized}" --output-dir "${pilot_optimized_physical}" \
  --device cuda:1 --query-chunk 10475 --seed 1234

conda run --no-capture-output -n New_MMM python -u \
  tools/compare_paired_physical_metrics.py \
  --before "${pilot_random_physical}" \
  --after "${pilot_optimized_physical}" \
  --output output/interx_shape_aug_pilot_flathand_query10475_paired_physical_seed20260828.json \
  --seed 1234 --bootstrap-samples 10000

conda run --no-capture-output -n New_MMM python -u \
  tools/assess_shape_augmentation_pilot.py \
  --comparison output/interx_shape_aug_pilot_flathand_query10475_paired_physical_seed20260828.json \
  --gate configs/interx_shape_augmentation/pilot_physical_gate_v1.json \
  --output output/interx_shape_aug_pilot_flathand_query10475_assessment_seed20260828.json

# The historical-replay gate above proves that old MMM behavior is recoverable.
# This separate content-bound gate proves that the current paper-contract MGT
# itself reaches a minimally credible ordinary FID before two multi-day arms.
conda run --no-capture-output -n New_MMM python -u \
  tools/require_hhc_formal_mgt_quality_gate.py \
  --gate output/hhc_formal_mgt_quality_gate.json

# Retain every historical result.  Project the two training runs, four
# common-50 labels and optimizer/physical intermediates from already measured
# formal artifacts; fail closed instead of filling the disk or deleting data.
formal_training_manifest="$(/home/wenning/miniconda3/envs/New_MMM/bin/python -c \
  "import json; print(json.load(open('${formal_selection}'))['training_manifest'])")"
conda run --no-capture-output -n New_MMM python -u \
  tools/audit_shape_augmentation_disk_capacity.py \
  --filesystem-root . \
  --reference-label "${formal_no_contact}" \
  --reference-training-run "$(dirname "${formal_training_manifest}")" \
  --output output/interx_shape_augmentation_disk_capacity_audit.json

# Full matched augmentation assets.  The random-only arm controls for added
# shape diversity and sample count; the optimized arm adds only the paper
# contact/physical refinement.
full_random="output/interx_shape_aug_random_seed123_v2"
full_optimized_aggregate="output/interx_shape_aug_full_batch8_query10475_seed20260828_aggregate"
if [[ ! -s "${full_random}/_completion_manifest.json" ]]; then
  conda run --no-capture-output -n New_MMM python -u \
    tools/run_interx_shape_augmentation.py \
    --plan "${plan}" --contact-dir "${contact_dir}" \
    --output-dir "${full_random}" --mode random_shape_only \
    --device cuda:1 --query-chunk 512 --seed 1234
fi
conda run --no-capture-output -n New_MMM python -u \
  tools/audit_random_shape_augmentation_completion.py \
  --run-dir "${full_random}" --plan "${plan}" --expected-samples 6238 \
  --output "${full_random}/completion_content_audit.json"

optimized_shard_args=()
for shard in {0..7}; do
  bash tools/run_interx_shape_augmentation_full_single_shard.sh 1 "${shard}"
  optimized_shard_args+=(
    --shard-dir
    "output/interx_shape_aug_full_batch8_query10475_seed20260828_shard${shard}"
  )
done
conda run --no-capture-output -n New_MMM python -u \
  tools/aggregate_interx_shape_augmentation_shards.py \
  "${optimized_shard_args[@]}" --expected-samples 6238 --expected-shards 8 \
  --output "${full_optimized_aggregate}/_completion_manifest.json"

random_overlay="dataset/interx_shape_aug_random_seed123_v2"
optimized_overlay="dataset/interx_shape_aug_optimized_flathand_seed123_v2"
if [[ ! -s "${random_overlay}/materialization_manifest.json" ]]; then
  conda run --no-capture-output -n New_MMM python -u \
    tools/materialize_interx_shape_augmentation.py \
    --optimization-dir "${full_random}" --plan "${plan}" \
    --output-root "${random_overlay}"
fi
if [[ ! -s "${optimized_overlay}/materialization_manifest.json" ]]; then
  optimized_materialize_args=()
  for shard in {0..7}; do
    optimized_materialize_args+=(
      --optimization-dir
      "output/interx_shape_aug_full_batch8_query10475_seed20260828_shard${shard}"
    )
  done
  conda run --no-capture-output -n New_MMM python -u \
    tools/materialize_interx_shape_augmentation.py \
    "${optimized_materialize_args[@]}" --plan "${plan}" \
    --output-root "${optimized_overlay}"
fi

run_arm() {
  local arm="$1"
  local overlay="$2"
  local cache_root="checkpoints/vq_precleanup_raw12288_shapeaug_${arm}_cache"
  local experiment="hhc_formal_raw12288_repaired_mgt_shapeaug_${arm}_300k_seed123_v1"
  local selection="output/hhc_formal_raw12288_repaired_mgt_shapeaug_${arm}_selection.json"
  local label_prefix="hhc_formal_raw12288_repaired_mgt_shapeaug_${arm}_common50"
  local training_state="output/hhc_formal_raw12288_repaired_mgt_shapeaug_${arm}_training_state.json"

  resolve_arm_training_state() {
    conda run --no-capture-output -n New_MMM python -u \
      tools/resolve_mgt_training_state.py \
      --run-glob "output/t2i_trans/*_${experiment}" \
      --experiment "${experiment}" --expected-final-step 300000 \
      --output "${training_state}"
  }

  resolve_arm_training_state
  local training_status
  training_status="$(/home/wenning/miniconda3/envs/New_MMM/bin/python -c \
    "import json; print(json.load(open('${training_state}'))['status'])")"
  local resume_args=()

  if [[ "${training_status}" != complete ]]; then
    if [[ "${training_status}" == new ]]; then
      conda run --no-capture-output -n New_MMM python -u \
        train/train_t2i_trans.py --cache-only \
        --device cuda:1 --exp-name "${experiment}_cache" \
        --vq-checkpoint "${vq_checkpoint}" \
        --cache-root "${cache_root}" --codebook-dir "${cache_root}/codebook" \
        --token-cache-contract deterministic_historical_root_v1 \
        --nb-code 12288 --code-dim 256 --shape-condition-dim 8 --width 512 \
        --embed-dim-gpt 512 --n-head-gpt 8 --num-layers 9 --num-local-layer 9 \
        --train-asset-root "${overlay}" \
        --augmentation-manifest "${overlay}/materialization_manifest.json" \
        --train-split-file "${overlay}/splits/train.txt" \
        --val-split-file dataset/interx/splits/val.txt \
        --all-split-file "${overlay}/splits/all.txt" \
        --train-motion-file "${overlay}/motions/train.h5" \
        --val-motion-file dataset/interx/motions/val.h5 \
        --all-motion-file "${overlay}/motions/all.h5" \
        --stats-dir "${overlay}" --stats-prefix Interx \
        --seed 123 --wandb-mode disabled
    else
      local resume_checkpoint
      resume_checkpoint="$(/home/wenning/miniconda3/envs/New_MMM/bin/python -c \
        "import json; print(json.load(open('${training_state}'))['resume_checkpoint'])")"
      resume_args=(--resume-training-ckpt "${resume_checkpoint}" --resume-in-place)
    fi

    conda run --no-capture-output -n New_MMM python -u \
      train/train_t2i_trans.py \
      --device cuda:1 --exp-name "${experiment}" \
      --vq-checkpoint "${vq_checkpoint}" \
      --cache-root "${cache_root}" --codebook-dir "${cache_root}/codebook" \
      --token-cache-contract deterministic_historical_root_v1 \
      --nb-code 12288 --code-dim 256 --shape-condition-dim 8 --width 512 \
      --embed-dim-gpt 512 --n-head-gpt 8 --num-layers 9 --num-local-layer 9 \
      --train-asset-root "${overlay}" \
      --augmentation-manifest "${overlay}/materialization_manifest.json" \
      --train-split-file "${overlay}/splits/train.txt" \
      --val-split-file dataset/interx/splits/val.txt \
      --all-split-file "${overlay}/splits/all.txt" \
      --train-motion-file "${overlay}/motions/train.h5" \
      --val-motion-file dataset/interx/motions/val.h5 \
      --all-motion-file "${overlay}/motions/all.h5" \
      --stats-dir "${overlay}" --stats-prefix Interx \
      --batch-size 96 --total-iter 300000 --warm-up-iter 1000 \
      --lr 0.0002 --weight-decay 0.000001 \
      --contact --no-contact-conditioning --shape --paper-architecture \
      --training-mask-contract paper_kinematic \
      --training-loss-contract paper_masked_only \
      --optimizer-contract paper_warmup_cosine \
      --conditioning-contract paper_text_cfg \
      --training-token-length-contract paper_50 \
      --training-validation-contract paper_isolated_common \
      --sampler sample_intra_cas --temperature 0.5 --sampling-steps 50 \
      --cfg-scale 3.5 --contact-guidance-scale 1.0 \
      --wandb-mode online --seed 123 "${resume_args[@]}"

    resolve_arm_training_state
  fi

  conda run --no-capture-output -n New_MMM python -u \
    tools/select_best_mgt_checkpoint.py \
    --run-glob "output/t2i_trans/*_${experiment}" \
    --expected-final-step 300000 --output "${selection}"

  local selected_checkpoint
  selected_checkpoint="$(/home/wenning/miniconda3/envs/New_MMM/bin/python -c \
    "import json; print(json.load(open('${selection}'))['selected_checkpoint'])")"
  local selected_training_manifest
  selected_training_manifest="$(/home/wenning/miniconda3/envs/New_MMM/bin/python -c \
    "import json; print(json.load(open('${selection}'))['training_manifest'])")"
  local selected_run_dir
  selected_run_dir="$(dirname "${selected_training_manifest}")"

  local validation_before="${selected_run_dir}/prevalidation-000020000.ckpt"
  local validation_after="${selected_run_dir}/postvalidation-000020000.ckpt"
  if [[ ! -s "${validation_before}" || ! -s "${validation_after}" ]]; then
    echo "${arm}: formal 20k pre/post-validation checkpoint pair is incomplete."
    exit 2
  fi
  conda run --no-capture-output -n New_MMM python -u \
    tools/audit_lightning_checkpoint_pair.py \
    --before "${validation_before}" --after "${validation_after}" \
    --expected-step 20000 --require-fixed-pair-marker \
    --output "output/hhc_formal_raw12288_repaired_mgt_shapeaug_${arm}_20k_pre_post_validation_pair_audit.json"

  conda run --no-capture-output -n New_MMM python -u \
    tools/audit_hhc_paper_training_contract.py \
    --training-manifest "${selected_training_manifest}" \
    --vq-manifest checkpoints/vq_precleanup_raw12288/training_manifest.json \
    --output "output/hhc_formal_raw12288_repaired_mgt_shapeaug_${arm}_training_contract_audit.json"

  conda run --no-capture-output -n New_MMM python -u \
    tools/audit_wandb_online_metric_history.py \
    --manifest "${selected_training_manifest}" \
    --metric-profile masked_token_classification \
    --expected-final-step 300000 \
    --output "output/hhc_formal_raw12288_repaired_mgt_shapeaug_${arm}_wandb_history_audit.json"

  conda run --no-capture-output -n New_MMM python -u \
    tools/run_fair_interx_method.py --method hhc \
    --label "${label_prefix}_no_contact" --gpu 1 \
    --checkpoint "${selected_checkpoint}" --vq-checkpoint "${vq_checkpoint}" \
    --hhc-vq-contract precleanup_raw12288 --hhc-mgt-contract current \
    --hhc-stats-dir "${overlay}" --hhc-stats-prefix Interx \
    --hhc-contact-map "${cache_root}/code_contact_probs.npy" \
    --selection-manifest "${selection}" --sampling-steps 50 \
    --temperature 0.5 --cfg-scale 3.5 --contact-guidance-scale 1.0 \
    --hhc-contact-timing-source none \
    --no-contact-guidance

  conda run --no-capture-output -n New_MMM python -u \
    tools/run_fair_interx_method.py --method hhc \
    --label "${label_prefix}_contact" --gpu 1 \
    --checkpoint "${selected_checkpoint}" --vq-checkpoint "${vq_checkpoint}" \
    --hhc-vq-contract precleanup_raw12288 --hhc-mgt-contract current \
    --hhc-stats-dir "${overlay}" --hhc-stats-prefix Interx \
    --hhc-contact-map "${cache_root}/code_contact_probs.npy" \
    --selection-manifest "${selection}" --sampling-steps 50 \
    --temperature 0.5 --cfg-scale 3.5 --contact-guidance-scale 1.0 \
    --hhc-contact-timing-source request_oracle \
    --contact-guidance
}

run_arm random "${random_overlay}"
run_arm optimized "${optimized_overlay}"

conda run --no-capture-output -n New_MMM python -u \
  tools/compare_fair_interx_methods_paired.py \
  --reference baseline="${formal_no_contact}" \
  --comparators \
    random_shape_only=output/fair_eval/hhc_formal_raw12288_repaired_mgt_shapeaug_random_common50_no_contact \
    optimized_shape=output/fair_eval/hhc_formal_raw12288_repaired_mgt_shapeaug_optimized_common50_no_contact \
  --output output/fair_eval/paired_shape_augmentation_comparison.json

conda run --no-capture-output -n New_MMM python -u \
  tools/compare_fair_interx_methods_paired.py \
  --reference baseline="${formal_contact}" \
  --comparators \
    random_shape_only=output/fair_eval/hhc_formal_raw12288_repaired_mgt_shapeaug_random_common50_contact \
    optimized_shape=output/fair_eval/hhc_formal_raw12288_repaired_mgt_shapeaug_optimized_common50_contact \
  --output output/fair_eval/paired_shape_augmentation_contact_guidance_comparison.json

conda run --no-capture-output -n New_MMM python -u \
  tools/build_shape_augmentation_results.py
