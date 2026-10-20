#!/usr/bin/env bash
set -euo pipefail

echo 'Retired diagnostic launcher: it targets the pre-repair MGT and must not generate active results.'
echo 'Use tools/run_formal_hhc_after_mgt_repair.sh after output/mgt_repair_gate.json passes.'
exit 2

cd /mnt/8T-HDD/wenning/HHC

# Formal MGT/HHC work is forbidden until the repaired historical replay has
# completed and the same checkpoint has passed the official common-50 bridge.
conda run --no-capture-output -n New_MMM python -u \
  tools/require_mgt_repair_gate.py \
  --gate output/mgt_repair_gate.json

vq_checkpoint="checkpoints/vq_precleanup_raw12288/net_175000.pth"
codebook_dir="checkpoints/vq_precleanup_raw12288/codebook"
git_default_mgt="/home/wenning/projects/MMM/output/t2i_trans/2026-02-13-01-55-15_vq_mask_gen_4_ablation_adalnshape_1_cfg/model-step=280000-val/FID=0.1605.ckpt"
latest_complete_mgt="/home/wenning/projects/MMM/output/t2i_trans/2026-02-26-13-43-25_vq_mask_gen_4_ablation_adalnshape_1_test_cfg_text_hbr/model-step=50000-val/FID=0.1125.ckpt"
experiment="paper_mgt_precleanup_raw12288_300k_seed123"

# Build a fresh, train+validation-scoped cache and train-only contact map for
# the byte-identical pre-cleanup raw VQ.  Never reuse the mutable MMM cache.
conda run --no-capture-output -n New_MMM python -u \
  train/train_t2i_trans.py --cache-only \
  --device cuda:1 --exp-name precleanup_raw12288_cache \
  --vq-checkpoint "${vq_checkpoint}" --codebook-dir "${codebook_dir}" \
  --nb-code 12288 --code-dim 256 --shape-condition-dim 8 --width 512 \
  --seed 123 --wandb-mode disabled

# First obtain two diagnostic replication-00 results under the user's locked
# 50-step sampler: the checkpoint hard-coded by the pre-cleanup Git tree and
# the validation-best checkpoint from the last complete 300k MGT run.
conda run --no-capture-output -n New_MMM python -u \
  tools/run_fair_interx_method.py --method hhc \
  --label hhc_precleanup_git_default_mgt_common50_contact_rep00 --gpu 1 \
  --checkpoint "${git_default_mgt}" --vq-checkpoint "${vq_checkpoint}" \
  --hhc-vq-contract precleanup_raw12288 \
  --hhc-mgt-contract historical_mmm \
  --sampling-steps 50 --temperature 0.5 --cfg-scale 3.5 \
  --contact-guidance-scale 1.0 --hhc-contact-timing-source request_oracle \
  --contact-guidance \
  --start-replication 0 --end-replication 1

conda run --no-capture-output -n New_MMM python -u \
  tools/run_fair_interx_method.py --method hhc \
  --label hhc_precleanup_latest_complete_mgt_common50_contact_rep00 --gpu 1 \
  --checkpoint "${latest_complete_mgt}" --vq-checkpoint "${vq_checkpoint}" \
  --hhc-vq-contract precleanup_raw12288 \
  --hhc-mgt-contract historical_mmm \
  --sampling-steps 50 --temperature 0.5 --cfg-scale 3.5 \
  --contact-guidance-scale 1.0 --hhc-contact-timing-source request_oracle \
  --contact-guidance \
  --start-replication 0 --end-replication 1

# Isolate MGT changes by retraining the current corrected architecture against
# exactly the same pre-cleanup raw tokenizer and freshly bound token cache.
conda run --no-capture-output -n New_MMM python -u \
  train/train_t2i_trans.py \
  --device cuda:1 --exp-name "${experiment}" \
  --vq-checkpoint "${vq_checkpoint}" --codebook-dir "${codebook_dir}" \
  --nb-code 12288 --code-dim 256 --shape-condition-dim 8 --width 512 \
  --batch-size 96 --total-iter 300000 --warm-up-iter 1000 \
  --lr 0.0002 --weight-decay 0.000001 \
  --contact --no-contact-conditioning --shape --paper-architecture \
  --sampler sample_intra_cas --temperature 0.5 --sampling-steps 50 \
  --cfg-scale 3.5 --contact-guidance-scale 1.0 \
  --wandb-mode online --seed 123

conda run --no-capture-output -n New_MMM python -u \
  tools/select_best_mgt_checkpoint.py \
  --run-glob "output/t2i_trans/*_${experiment}" \
  --expected-final-step 300000 \
  --output output/hhc_mgt_precleanup_raw12288_selection.json

selected_checkpoint="$(/home/wenning/miniconda3/envs/New_MMM/bin/python -c \
  "import json; print(json.load(open('output/hhc_mgt_precleanup_raw12288_selection.json'))['selected_checkpoint'])")"

conda run --no-capture-output -n New_MMM python -u \
  tools/run_fair_interx_method.py --method hhc \
  --label hhc_precleanup_raw12288_current_mgt_common50_contact --gpu 1 \
  --checkpoint "${selected_checkpoint}" --vq-checkpoint "${vq_checkpoint}" \
  --hhc-vq-contract precleanup_raw12288 --hhc-mgt-contract current \
  --selection-manifest output/hhc_mgt_precleanup_raw12288_selection.json \
  --sampling-steps 50 --temperature 0.5 --cfg-scale 3.5 \
  --contact-guidance-scale 1.0 --hhc-contact-timing-source request_oracle \
  --contact-guidance

conda run --no-capture-output -n New_MMM python -u \
  tools/run_fair_interx_method.py --method hhc \
  --label hhc_precleanup_raw12288_current_mgt_common50_no_contact --gpu 1 \
  --checkpoint "${selected_checkpoint}" --vq-checkpoint "${vq_checkpoint}" \
  --hhc-vq-contract precleanup_raw12288 --hhc-mgt-contract current \
  --selection-manifest output/hhc_mgt_precleanup_raw12288_selection.json \
  --sampling-steps 50 --temperature 0.5 --cfg-scale 3.5 \
  --contact-guidance-scale 1.0 --hhc-contact-timing-source none \
  --no-contact-guidance

# Reproduce the original MMM contact-map implementation for one diagnostic
# replication only.  This map used all splits and an axis error, so it is
# explicitly excluded from the fair main table.
conda run --no-capture-output -n New_MMM python -u \
  tools/audit_hhc_contact_guidance_maps.py

conda run --no-capture-output -n New_MMM python -u \
  tools/run_fair_interx_method.py --method hhc \
  --label hhc_precleanup_raw12288_current_mgt_common50_contact_legacy_full_axis_bug_rep00 \
  --gpu 1 \
  --checkpoint "${selected_checkpoint}" --vq-checkpoint "${vq_checkpoint}" \
  --hhc-vq-contract precleanup_raw12288 --hhc-mgt-contract current \
  --selection-manifest output/hhc_mgt_precleanup_raw12288_selection.json \
  --sampling-steps 50 --temperature 0.5 --cfg-scale 3.5 \
  --contact-guidance-scale 1.0 --hhc-contact-timing-source request_oracle \
  --hhc-contact-map \
    checkpoints/vq_precleanup_raw12288/code_contact_probs_mmm_legacy_full_axis_bug.npy \
  --hhc-contact-map-contract mmm_legacy_full_axis_bug \
  --contact-guidance --start-replication 0 --end-replication 1
