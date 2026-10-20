#!/usr/bin/env bash
set -euo pipefail

cd /mnt/8T-HDD/wenning/HHC

if [[ $# -ne 6 || ! "$1" =~ ^[0-9]+$ || ! "$2" =~ ^[0-9]+$ || "$1" == "$2" ]]; then
  echo "usage: $0 GPU_LOW GPU_HIGH CHECKPOINT LABEL STATS_DIR CACHE_ROOT" >&2
  exit 2
fi
gpu_low="$1"
gpu_high="$2"
checkpoint="$3"
label="$4"
stats_dir="$5"
cache_root="$6"
vq="checkpoints/vq_precleanup_raw12288/net_175000.pth"
fair_dir="output/fair_eval/${label}"
joint_root="output/interact2ar_joint_evaluator_candidates/${label}"

test -s "${checkpoint}"
test -s "${vq}"
test -s "${cache_root}/code_contact_probs.npy"
test -s output/interact2ar_joint_scorer_parity_audit.json

for gpu in "${gpu_low}" "${gpu_high}"; do
  used_mib="$(nvidia-smi --id="${gpu}" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')"
  if [[ ! "${used_mib}" =~ ^[0-9]+$ ]] || (( used_mib > 1024 )); then
    echo "GPU ${gpu} is not free (memory.used=${used_mib:-unknown} MiB)." >&2
    exit 2
  fi
done

common=(
  python -u tools/run_fair_interx_method.py
  --method hhc --label "${label}" --checkpoint "${checkpoint}"
  --vq-checkpoint "${vq}" --hhc-vq-contract precleanup_raw12288
  --hhc-mgt-contract historical_mmm --no-hhc-mgt-shape
  --hhc-inference-conditioning-contract paper_text_cfg
  --hhc-stats-dir "${stats_dir}" --hhc-stats-prefix Interx
  --hhc-contact-map "${cache_root}/code_contact_probs.npy"
  --hhc-contact-map-contract paper_formula_train_only
  --no-contact-guidance --hhc-contact-timing-source none
  --sampling-steps 50 --temperature 0.5 --cfg-scale 3.5
  --batch-size 32 --shard-manifest
)

/home/wenning/miniconda3/bin/conda run --no-capture-output -n New_MMM \
  "${common[@]}" --start-replication 0 --end-replication 10 --gpu "${gpu_low}" &
pid_low=$!
/home/wenning/miniconda3/bin/conda run --no-capture-output -n New_MMM \
  "${common[@]}" --start-replication 10 --end-replication 20 --gpu "${gpu_high}" &
pid_high=$!

cleanup_children() {
  kill "${pid_low}" "${pid_high}" 2>/dev/null || true
}
trap cleanup_children INT TERM
status=0
wait "${pid_low}" || status=$?
wait "${pid_high}" || status=$?
trap - INT TERM
if (( status != 0 )); then
  exit "${status}"
fi

/home/wenning/miniconda3/envs/New_MMM/bin/python \
  tools/merge_fair_interx_shard_manifests.py \
  --method-dir "${fair_dir}" --ranges 0:10 10:20 \
  --output "${fair_dir}/run_manifest.json"
/home/wenning/miniconda3/envs/New_MMM/bin/python \
  tools/aggregate_fair_interx_metrics.py \
  --inputs "${fair_dir}/replication_[0-9][0-9].json" \
  --output "${fair_dir}/summary.json" --expected-samples 1696
/home/wenning/miniconda3/envs/New_MMM/bin/python \
  tools/aggregate_fair_interx_generation_timings.py \
  --inputs "${fair_dir}/replication_[0-9][0-9]_generation_timing.json" \
  --output "${fair_dir}/generation_timing_summary.json"
/home/wenning/miniconda3/envs/New_MMM/bin/python \
  tools/aggregate_fair_interx_multimodality_audits.py \
  --metrics "${fair_dir}/replication_[0-9][0-9].json" \
  --audits "${fair_dir}/replication_[0-9][0-9]_multimodality_audit.json" \
  --output "${fair_dir}/multimodality_audit_summary.json"

/home/wenning/miniconda3/bin/conda run --no-capture-output -n New_MMM \
  python -u tools/score_interact2ar_joint_evaluators.py \
  --replications 20 --batch-size 32 --device "cuda:${gpu_low}" \
  --method "${label}=${fair_dir}" --output-root "${joint_root}/scores"
/home/wenning/miniconda3/bin/conda run --no-capture-output -n New_MMM \
  python -u tools/aggregate_interact2ar_joint_evaluators.py \
  --input-root "${joint_root}/scores" --expected-methods "${label}" \
  --output "${joint_root}/summary.json"
/home/wenning/miniconda3/bin/conda run --no-capture-output -n New_MMM \
  python -u tools/assess_hhc_against_interact2ar_joint_paper.py \
  --summary "${joint_root}/summary.json" --method "${label}" \
  --output "${joint_root}/paper_acceptance.json"
/home/wenning/miniconda3/bin/conda run --no-capture-output -n New_MMM \
  python -u tools/diagnose_mm_repeat_collapse.py \
  --input "${label}=${fair_dir}/replication_00_mm.h5" \
  --output "${joint_root}/mm_repeat_collapse.json" --pairs-per-prompt 50
