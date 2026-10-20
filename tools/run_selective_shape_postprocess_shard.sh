#!/usr/bin/env bash
set -euo pipefail

cd /mnt/8T-HDD/wenning/HHC

if [[ $# -ne 2 || ! "$1" =~ ^[0-9]+$ || ! "$2" =~ ^[01]$ ]]; then
  echo "usage: $0 GPU_INDEX SHARD_INDEX_0_OR_1" >&2
  exit 2
fi
gpu="$1"
shard="$2"
input="output/interx_shape_aug_volumetric_fast_steps120_full_shard${shard}"
output="output/interx_shape_aug_volumetric_fast_selective_full_shard${shard}"

test -s "${input}/_completion_manifest.json"
/home/wenning/miniconda3/envs/New_MMM/bin/python -c \
  "import json; assert json.load(open('${input}/_completion_manifest.json'))['complete'] is True"

# This is a normal experiment stage and has no waiter loop. It is launched as
# soon as its optimizer shard completes, on the same explicitly selected GPU.
/home/wenning/miniconda3/bin/conda run --no-capture-output -n New_MMM \
  python -u tools/select_interx_shape_optimization.py \
  --optimized-dir "${input}" \
  --random-dir output/interx_shape_aug_random_seed123_v2 \
  --contact-dir dataset/interx/contact_pairs_codex_luna_strict_v3_train \
  --output-dir "${output}" --device "cuda:${gpu}" \
  --skate-tolerance 0.000002
