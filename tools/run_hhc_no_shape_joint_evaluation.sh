#!/usr/bin/env bash
set -euo pipefail

cd /mnt/8T-HDD/wenning/HHC

if [[ $# -ne 5 || ! "$1" =~ ^[0-9]+$ ]]; then
  echo "usage: $0 GPU_INDEX CHECKPOINT LABEL STATS_DIR CACHE_ROOT" >&2
  exit 2
fi
gpu="$1"
checkpoint="$2"
label="$3"
stats_dir="$4"
cache_root="$5"
vq="checkpoints/vq_precleanup_raw12288/net_175000.pth"
fair_dir="output/fair_eval/${label}"
joint_root="output/interact2ar_joint_evaluator_candidates/${label}"

test -s "${checkpoint}"
test -s "${vq}"
test -s "${cache_root}/code_contact_probs.npy"
test -s output/interact2ar_joint_scorer_parity_audit.json

used_mib="$(nvidia-smi --id="${gpu}" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')"
if [[ ! "${used_mib}" =~ ^[0-9]+$ ]] || (( used_mib > 1024 )); then
  echo "GPU ${gpu} is not free (memory.used=${used_mib:-unknown} MiB)." >&2
  exit 2
fi

# Generate and score only HHC. Other methods are intentionally absent from
# this run while their reproduction discrepancy is paused.
/home/wenning/miniconda3/bin/conda run --no-capture-output -n New_MMM \
  python -u tools/run_fair_interx_method.py \
  --method hhc --label "${label}" --checkpoint "${checkpoint}" \
  --vq-checkpoint "${vq}" --hhc-vq-contract precleanup_raw12288 \
  --hhc-mgt-contract historical_mmm --no-hhc-mgt-shape \
  --hhc-inference-conditioning-contract paper_text_cfg \
  --hhc-stats-dir "${stats_dir}" --hhc-stats-prefix Interx \
  --hhc-contact-map "${cache_root}/code_contact_probs.npy" \
  --hhc-contact-map-contract paper_formula_train_only \
  --no-contact-guidance --hhc-contact-timing-source none \
  --sampling-steps 50 --temperature 0.5 --cfg-scale 3.5 \
  --start-replication 0 --end-replication 20 \
  --gpu "${gpu}" --batch-size 32

/home/wenning/miniconda3/bin/conda run --no-capture-output -n New_MMM \
  python -u tools/score_interact2ar_joint_evaluators.py \
  --replications 20 --batch-size 32 --device "cuda:${gpu}" \
  --method "${label}=${fair_dir}" --output-root "${joint_root}/scores"

/home/wenning/miniconda3/bin/conda run --no-capture-output -n New_MMM \
  python -u tools/aggregate_interact2ar_joint_evaluators.py \
  --input-root "${joint_root}/scores" --expected-methods "${label}" \
  --output "${joint_root}/summary.json"

/home/wenning/miniconda3/bin/conda run --no-capture-output -n New_MMM \
  python -u tools/assess_hhc_against_interact2ar_joint_paper.py \
  --summary "${joint_root}/summary.json" --method "${label}" \
  --output "${joint_root}/paper_acceptance.json"

# Keep MModality interpretable: report exact discrete-output collapse in the
# first frozen replication in addition to the evaluator-space mean.
/home/wenning/miniconda3/bin/conda run --no-capture-output -n New_MMM \
  python -u tools/diagnose_mm_repeat_collapse.py \
  --input "${label}=${fair_dir}/replication_00_mm.h5" \
  --output "${joint_root}/mm_repeat_collapse.json" --pairs-per-prompt 50
