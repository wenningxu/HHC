#!/usr/bin/env bash
set -euo pipefail

cd /mnt/8T-HDD/wenning/HHC

train_dir="dataset/interx/contact_pairs_codex_luna_strict_v3_train"
train_manifest="${train_dir}/_extraction_manifest.json"
train_status="$(/home/wenning/miniconda3/envs/New_MMM/bin/python -c \
  "import json; print(json.load(open('${train_manifest}'))['status'])")"
if [[ "${train_status}" != complete ]]; then
  echo "Train contact cache is ${train_status}; finish it before starting the remaining splits."
  exit 2
fi

conda run --no-capture-output -n New_MMM python -u \
  tools/audit_interx_codex_contact_cache.py \
  --pairs-dir "${train_dir}" --split dataset/interx/splits/train.txt \
  --output output/interx_luna_train_cache_audit.json

conda run --no-capture-output -n New_MMM python -u \
  tools/audit_interx_contact_timing_consistency.py \
  --pairs-dir "${train_dir}" --split dataset/interx/splits/train.txt \
  --motion-file dataset/interx/motions/train.h5 \
  --output output/interx_luna_train_timing_consistency_audit.json

# Highest priority: exact replication-00 captions for the 1,696 samples used
# by the official fair-evaluation request set and the formal optimizer.
conda run --no-capture-output -n New_MMM python -u \
  eval/extract_interx_contacts_codex_frozen_requests.py \
  --split-file dataset/interx/splits/test.txt \
  --requests output/fair_interx_requests/replication_00.npz \
  --out-dir dataset/interx/contact_pairs_codex_luna_strict_v3_test \
  --model gpt-5.6-luna --reasoning-effort low \
  --batch-size 32 --max-retries 2

conda run --no-capture-output -n New_MMM python -u \
  tools/audit_interx_codex_contact_cache.py \
  --pairs-dir dataset/interx/contact_pairs_codex_luna_strict_v3_test \
  --split dataset/interx/splits/test.txt \
  --requests output/fair_interx_requests/replication_00.npz \
  --output output/interx_luna_test_frozen_cache_audit.json

# Pre-registered 128-caption robustness subset: 32 timing-negative, 32
# laterality-positive, 32 multi-contact-cue-positive, and 32 role-reference
# positive examples.  Terra is a stronger-model consistency reference, while
# the deterministic rule parser is a non-LLM baseline.  Manual Inter-X timing
# provides only binary contact presence, never fake 75-region ground truth.
conda run --no-capture-output -n New_MMM python -u \
  tools/prepare_interx_llm_robustness_requests.py

conda run --no-capture-output -n New_MMM python -u \
  eval/extract_interx_contacts_codex_frozen_requests.py \
  --split-file dataset/interx/splits/test.txt \
  --requests output/interx_llm_robustness/requests.npz \
  --out-dir dataset/interx/contact_pairs_codex_terra_robustness_test \
  --model gpt-5.6-terra --reasoning-effort low \
  --batch-size 32 --max-retries 2

conda run --no-capture-output -n New_MMM python -u \
  tools/audit_interx_codex_contact_cache.py \
  --pairs-dir dataset/interx/contact_pairs_codex_terra_robustness_test \
  --split dataset/interx/splits/test.txt \
  --requests output/interx_llm_robustness/requests.npz \
  --output output/interx_terra_robustness_cache_audit.json

conda run --no-capture-output -n New_MMM python -u \
  tools/predict_interx_contact_regions_rule.py

conda run --no-capture-output -n New_MMM python -u \
  tools/evaluate_interx_llm_robustness.py

# Complete first-caption annotations for every official val/test sequence as a
# dataset-wide auxiliary cache.  These are not substituted for frozen test
# captions in the formal fair comparison.
for split_name in val test; do
  out_dir="dataset/interx/contact_pairs_codex_luna_strict_v3_${split_name}_full"
  conda run --no-capture-output -n New_MMM python -u \
    eval/resume_interx_contacts_codex_preserve_failures.py \
    --split-file "dataset/interx/splits/${split_name}.txt" \
    --out-dir "${out_dir}" \
    --model gpt-5.6-luna --reasoning-effort low \
    --batch-size 32 --max-retries 2

  conda run --no-capture-output -n New_MMM python -u \
    tools/audit_interx_codex_contact_cache.py \
    --pairs-dir "${out_dir}" \
    --split "dataset/interx/splits/${split_name}.txt" \
    --output "output/interx_luna_${split_name}_full_cache_audit.json"

  conda run --no-capture-output -n New_MMM python -u \
    tools/audit_interx_contact_timing_consistency.py \
    --pairs-dir "${out_dir}" \
    --split "dataset/interx/splits/${split_name}.txt" \
    --motion-file "dataset/interx/motions/${split_name}.h5" \
    --output "output/interx_luna_${split_name}_full_timing_audit.json"
done
