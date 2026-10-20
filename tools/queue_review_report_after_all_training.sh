#!/usr/bin/env bash
set -euo pipefail

cd /mnt/8T-HDD/wenning/HHC

conda() {
  /home/wenning/miniconda3/bin/conda "$@"
}

timotion_pid="${1:-580502}"
if [[ "${timotion_pid}" != none ]]; then
  while ps -p "${timotion_pid}" >/dev/null 2>&1; do
    sleep 30
  done
fi

formal_training_manifest="$(/home/wenning/miniconda3/envs/New_MMM/bin/python -c \
  "import json; print(json.load(open('output/hhc_formal_raw12288_repaired_mgt_selection.json'))['training_manifest'])")"
formal_run_dir="$(dirname "${formal_training_manifest}")"

conda run --no-capture-output -n New_MMM python -u \
  tools/audit_historical_mmm_wandb.py \
  --selection-manifest output/hhc_formal_raw12288_repaired_mgt_selection.json

conda run --no-capture-output -n New_MMM python -u \
  tools/audit_mgt_common50_mm_replay.py \
  --reference output/fair_eval/hhc_precleanup_git_default_mgt_common50_contact_rep00/replication_00_mm.h5 \
  --candidate output/fair_eval/hhc_mgt_repair_v7_step280k_common50_paper_text_cfg_contact_rep00/replication_00_mm.h5 \
  --reference-manifest output/fair_eval/hhc_precleanup_git_default_mgt_common50_contact_rep00/run_manifest.json \
  --expected-repeats 30 \
  --output output/historical_mgt_v7_step280k_common50_mm_full30_audit.json

conda run --no-capture-output -n New_MMM python -u \
  tools/audit_mgt_validation_selection.py \
  --output output/mgt_validation_selection_manifest.json

conda run --no-capture-output -n New_MMM python -u \
  tools/recover_fair_request_provenance.py

request_replay_dir="output/fair_interx_request_replay_cpu_seed1234"
conda run --no-capture-output -n New_MMM python -u \
  tools/prepare_fair_eval_requests.py \
  --output-dir "${request_replay_dir}" --device cpu \
  --replications 1 --seed 1234

conda run --no-capture-output -n New_MMM python -u \
  tools/audit_fair_interx_request_replay.py \
  --replay "${request_replay_dir}/replication_00.npz" \
  --embedding-atol 0.000001 \
  --output output/fair_interx_request_replay_audit.json

conda run --no-capture-output -n New_MMM python -u \
  tools/verify_fair_eval_requests.py \
  --request-dir output/fair_interx_requests \
  --replications 20 --base-seed 1234 \
  --output output/fair_interx_requests/manifest.json

conda run --no-capture-output -n New_MMM python -u \
  tools/audit_historical_mmm_local_history.py

conda run --no-capture-output -n New_MMM python -u \
  tools/verify_interx_protocol.py \
  --output output/interx_protocol_audit.json

conda run --no-capture-output -n New_MMM python -u \
  tools/audit_historical_mgt_tokenizer_runtime.py \
  --output output/historical_mgt_tokenizer_runtime_audit.json

conda run --no-capture-output -n New_MMM python -u \
  tools/audit_hhc_formal_launch_contract.py \
  --source-snapshot-manifest "${formal_run_dir}/launch_source_snapshot_manifest.json"

conda run --no-capture-output -n New_MMM python -u \
  tools/audit_hhc_sampling_runtime.py \
  --output output/hhc_sampling_runtime_audit.json

conda run --no-capture-output -n New_MMM python -u \
  tools/audit_review_external_prerequisites.py

conda run --no-capture-output -n New_MMM python -u \
  tools/audit_training_wallclock.py

conda run --no-capture-output -n New_MMM python -u \
  tools/audit_hhc_paper_training_contract.py \
  --training-manifest "${formal_training_manifest}" \
  --source-snapshot-manifest "${formal_run_dir}/launch_source_snapshot_manifest.json" \
  --output output/hhc_formal_raw12288_repaired_mgt_training_contract_audit.json

pair_before="${formal_run_dir}/prevalidation-000020000.ckpt"
pair_after="${formal_run_dir}/postvalidation-000020000.ckpt"
if [[ ! -s "${pair_before}" || ! -s "${pair_after}" ]]; then
  echo 'Formal repaired-MGT 20k validation checkpoint pair is missing.'
  exit 2
fi
conda run --no-capture-output -n New_MMM python -u \
  tools/audit_lightning_checkpoint_pair.py \
  --before "${pair_before}" --after "${pair_after}" \
  --expected-step 20000 --require-fixed-pair-marker \
  --output output/hhc_formal_raw12288_repaired_mgt_20k_pre_post_validation_pair_audit.json

conda run --no-capture-output -n New_MMM python -u \
  tools/audit_hhc_contact_guidance_maps.py

conda run --no-capture-output -n New_MMM python -u \
  tools/audit_coshmdm_training_contract.py \
  --output output/coshmdm_training_contract_audit.json

conda run --no-capture-output -n New_MMM python -u \
  tools/audit_coshmdm_statsfix_startup.py \
  --output output/coshmdm_statsfix_startup_audit.json

conda run --no-capture-output -n New_MMM python -u \
  tools/audit_coshmdm_resume_provenance.py \
  --output output/coshmdm_resume_provenance.json

conda run --no-capture-output -n New_MMM python -u \
  tools/audit_coshmdm_representation_adapter.py \
  --output output/coshmdm_representation_adapter_audit.json

conda run --no-capture-output -n New_MMM python -u \
  tools/audit_coshmdm_generation_contract.py \
  --output output/coshmdm_generation_contract_audit.json

conda run --no-capture-output -n New_MMM python -u \
  tools/audit_timotion_generation_contract.py \
  --output output/timotion_generation_contract_audit.json

conda run --no-capture-output -n New_MMM python -u \
  tools/audit_review_requirement_coverage.py

cosh_selected_checkpoint="$(/home/wenning/miniconda3/envs/New_MMM/bin/python -c \
  "import json; print(json.load(open('output/coshmdm_interx_val_selection.json'))['selected_checkpoint'])")"
conda run --no-capture-output -n New_MMM python -u \
  tools/audit_model_capacity.py \
  --hhc-vq-checkpoint checkpoints/vq_precleanup_raw12288/net_175000.pth \
  --hhc-vq-contract precleanup_raw12288 \
  --hhc-selection-manifest output/hhc_formal_raw12288_repaired_mgt_selection.json \
  --coshmdm-checkpoint "${cosh_selected_checkpoint}" \
  --output output/model_capacity_audit.json

# Reuse all generated HDF5 files and rescore every final method on the same
# CPU runtime with the current official full-672D evaluator, matching the
# recovered command that created the frozen GT/text request embeddings.  This
# removes the old body-controlled fields and prevents CPU/CUDA:0/CUDA:1
# embedding roundoff from being mixed across methods.  Audit all six methods
# before any paired comparison or report is accepted.
conda run --no-capture-output -n New_MMM python -u \
  tools/rescore_final_interx_ordinary.py --device cpu --batch-size 32

# Reviewer-requested contact/penetration main results use one common mesh
# protocol for every final method.  Each method is converted with the same
# frozen replication-0 request beta/gender and scored by the released
# male/female VolumetricSMPL weights.  Conversion and scoring are strict and
# resumable per sample, so a service/machine restart does not lose completed
# work or silently accept stale inputs.
conda run --no-capture-output -n New_MMM python -u \
  tools/require_interx_volumetric_smpl_contract.py \
  --query-chunk 10475 \
  --output output/interx_volumetric_smpl_contract_gate.json

conda run --no-capture-output -n New_MMM python -u \
  tools/run_final_interx_mesh_physical.py \
  --device cuda:0 --query-chunk 10475 --seed 1234

# Self-heal the disclosed one-replication legacy-map diagnostic in case the
# already-running HHC queue had parsed its pre-edit script body before this
# command was added.  Completed/resumable HDF5 files are never regenerated.
legacy_map_result="output/fair_eval/hhc_formal_raw12288_repaired_mgt_common50_contact_legacy_full_axis_bug_rep00/replication_00.json"
if [[ ! -f "${legacy_map_result}" ]]; then
  selected_checkpoint="$(/home/wenning/miniconda3/envs/New_MMM/bin/python -c \
    "import json; print(json.load(open('output/hhc_formal_raw12288_repaired_mgt_selection.json'))['selected_checkpoint'])")"
  conda run --no-capture-output -n New_MMM python -u \
    tools/run_fair_interx_method.py --method hhc \
    --label hhc_formal_raw12288_repaired_mgt_common50_contact_legacy_full_axis_bug_rep00 \
    --gpu 1 \
    --checkpoint "${selected_checkpoint}" \
    --vq-checkpoint checkpoints/vq_precleanup_raw12288/net_175000.pth \
    --hhc-vq-contract precleanup_raw12288 --hhc-mgt-contract current \
    --selection-manifest output/hhc_formal_raw12288_repaired_mgt_selection.json \
    --sampling-steps 50 --temperature 0.5 --cfg-scale 3.5 \
    --contact-guidance-scale 1.0 --hhc-contact-timing-source request_oracle \
    --hhc-contact-map \
      checkpoints/vq_precleanup_raw12288/code_contact_probs_mmm_legacy_full_axis_bug.npy \
    --hhc-contact-map-contract mmm_legacy_full_axis_bug \
    --contact-guidance --start-replication 0 --end-replication 1
fi

conda run --no-capture-output -n New_MMM python -u \
  tools/compare_fair_interx_methods_paired.py \
  --reference hhc_no_timing=output/fair_eval/hhc_formal_raw12288_repaired_mgt_common50_no_contact \
  --comparators \
    intermask=output/fair_eval/intermask_official_exact_common50 \
    interact2ar=output/fair_eval/interact2ar_official_fair_no_gt_first_frame \
    coshmdm=output/fair_eval/coshmdm_interx_retrained_2daycap_seed123 \
    timotion=output/fair_eval/timotion_transformer_lpa_interx_retrained_fair \
  --output output/fair_eval/paired_fair_method_comparison.json

conda run --no-capture-output -n New_MMM python -u \
  tools/compare_fair_interx_methods_paired.py \
  --reference hhc_oracle=output/fair_eval/hhc_formal_raw12288_repaired_mgt_common50_contact \
  --comparators \
    hhc_no_timing=output/fair_eval/hhc_formal_raw12288_repaired_mgt_common50_no_contact \
  --output output/fair_eval/paired_oracle_timing_effect.json

# One-replication implementation sensitivity: the original MMM map leaks the
# test split and exactly replays an axis bug.  It is never promoted to METHODS.
conda run --no-capture-output -n New_MMM python -u \
  tools/compare_fair_interx_methods_paired.py \
  --reference \
    fair_map=output/fair_eval/hhc_formal_raw12288_repaired_mgt_common50_contact \
  --comparators \
    legacy_bug_map=output/fair_eval/hhc_formal_raw12288_repaired_mgt_common50_contact_legacy_full_axis_bug_rep00 \
  --expected-replications 1 --replication-ids 0 \
  --output output/fair_eval/paired_contact_map_implementation_sensitivity_rep0.json

# Same-request MGT-state diagnostic.  These are replication-0 effect sizes,
# not main-table rows or inferential evidence.
conda run --no-capture-output -n New_MMM python -u \
  tools/compare_fair_interx_methods_paired.py \
  --reference \
    corrected_paper=output/fair_eval/hhc_formal_raw12288_repaired_mgt_common50_contact \
  --comparators \
    precleanup_latest_complete_50k=output/fair_eval/hhc_precleanup_latest_complete_mgt_common50_contact_rep00 \
    precleanup_git_default_280k=output/fair_eval/hhc_precleanup_git_default_mgt_common50_contact_rep00 \
  --expected-replications 1 --replication-ids 0 \
  --output output/fair_eval/paired_hhc_mgt_training_state_sensitivity_rep0.json

conda run --no-capture-output -n New_MMM python -u \
  tools/prepare_qualitative_comparison_selection.py

conda run --no-capture-output -n New_MMM python -u \
  tools/render_fair_interx_qualitative.py \
  --selection configs/qualitative_comparison_selection.json \
  --fair-root output/fair_eval \
  --output-dir output/qualitative_comparison --device cpu

# Re-exercise all protocol gates, including the cache-ready formal optimizer
# branch, immediately before accepting the final report.
conda run --no-capture-output -n New_MMM python \
  -m tests.test_paper_consistency

conda run --no-capture-output -n New_MMM python \
  -m tests.test_online_wandb_policy

conda run --no-capture-output -n New_MMM python \
  -m tests.test_post_formal_patch_sequence

conda run --no-capture-output -n New_MMM python \
  -m tests.test_post_50k_dispatch

conda run --no-capture-output -n New_MMM python \
  -m tests.test_post_hhc_driver_binding

conda run --no-capture-output -n New_MMM python \
  -m tests.test_codex_contact_cache_audit

conda run --no-capture-output -n New_MMM python \
  -m tests.test_shape_augmentation_pipeline

conda run --no-capture-output -n New_MMM python -u \
  tools/build_review_results_report.py
