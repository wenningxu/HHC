#!/usr/bin/env bash
set -euo pipefail

conda_bin="/home/wenning/miniconda3/bin/conda"
if [[ ! -x "${conda_bin}" ]]; then
  echo "Required Conda executable is missing: ${conda_bin}"
  exit 2
fi
conda() {
  "${conda_bin}" "$@"
}

# The CoShMDM final watcher owns GPU 0 through its strict 20-repeat official
# full-672D ordinary-only evaluation. Wait for that entire chain, not only its
# trainer, so the two jobs can never overlap.
wait_pid="${1:-474427}"
if [[ "${wait_pid}" != none ]]; then
  while ps -p "${wait_pid}" >/dev/null 2>&1; do
    sleep 30
  done
fi

cd /mnt/8T-HDD/wenning/TIMotion
export TIMOTION_STATS_PREFIX=interx_global

# Bind the complete public-source/data contract before consuming GPU time.
# This preflight is deliberately a separate experiment so it cannot overwrite
# the provenance of a resumable formal run.
conda run --no-capture-output -n New_MMM python -u tools/train_interx_fair.py \
  --exp-name TIMotion-Transformer-LPA-InterX-contract-preflight \
  --gpu 0 --seed 123 --batch-size 32 --accumulate-grad-batches 8 \
  --microbatches-per-epoch 568 --epoch 1500 --lpa --manifest-only \
  --wandb-mode disabled --no-progress-bar

cd /mnt/8T-HDD/wenning/HHC

# All training launched after 2026-08-31 is blocked until the implementation
# provides the mandatory online W&B training/validation telemetry contract.
conda run --no-capture-output -n New_MMM python -u \
  tools/audit_wandb_online_runtime.py \
  --output output/timotion_wandb_runtime_audit.json
conda run --no-capture-output -n New_MMM python -u \
  tools/audit_online_wandb_training_contract.py \
  --launcher tools/queue_timotion_fair_after_cosh.sh \
  --training-source /mnt/8T-HDD/wenning/TIMotion/tools/train_interx_fair.py \
  --telemetry-source /mnt/8T-HDD/wenning/TIMotion/tools/train_interx_fair.py \
  --metric-profile diffusion_regression \
  --output output/timotion_online_wandb_training_audit.json
conda run --no-capture-output -n New_MMM python -u \
  tools/audit_timotion_training_contract.py \
  --manifest /mnt/8T-HDD/wenning/TIMotion/checkpoints/TIMotion-Transformer-LPA-InterX-contract-preflight/meta/training_manifest.json \
  --output output/timotion_interx_training_contract_preflight_audit.json

conda run --no-capture-output -n New_MMM python -u \
  tools/audit_timotion_cosh_adapter_equivalence.py \
  --output output/timotion_cosh_adapter_equivalence_audit.json

conda run --no-capture-output -n New_MMM python -u \
  tools/audit_timotion_generation_contract.py \
  --output output/timotion_generation_contract_audit.json

# Runtime gate: one complete global-batch update, including the exact released
# loss, LPA block, backward pass, clip, AdamW and scheduler/checkpoint state.
cd /mnt/8T-HDD/wenning/TIMotion
conda run --no-capture-output -n New_MMM python -u tools/train_interx_fair.py \
  --exp-name TIMotion-Transformer-LPA-InterX-smoke-bs32x8-s50-v2 \
  --gpu 0 --seed 123 --batch-size 32 --accumulate-grad-batches 8 \
  --microbatches-per-epoch 400 --epoch 1 --save-every-n-epochs 1 \
  --lpa --wandb-mode online --semantic-val-every-n-epochs 1 \
  --no-progress-bar

cd /mnt/8T-HDD/wenning/HHC
conda run --no-capture-output -n New_MMM python -u \
  tools/audit_lightning_checkpoint.py \
  --checkpoint /mnt/8T-HDD/wenning/TIMotion/checkpoints/TIMotion-Transformer-LPA-InterX-smoke-bs32x8-s50-v2/model/last.ckpt \
  --output output/timotion_interx_smoke_checkpoint_audit.json \
  --expected-step 50

conda run --no-capture-output -n New_MMM python -u \
  tools/audit_wandb_online_metric_history.py \
  --manifest /mnt/8T-HDD/wenning/TIMotion/checkpoints/TIMotion-Transformer-LPA-InterX-smoke-bs32x8-s50-v2/meta/training_manifest.json \
  --metric-profile diffusion_regression --expected-final-step 50 \
  --output output/timotion_interx_smoke_wandb_history_audit.json

cd /mnt/8T-HDD/wenning/TIMotion
formal_manifest=checkpoints/TIMotion-Transformer-LPA-InterX-fair-seed123-bs32x8/meta/training_manifest.json
formal_checkpoint=checkpoints/TIMotion-Transformer-LPA-InterX-fair-seed123-bs32x8/model/last.ckpt
if [[ -s "${formal_checkpoint}" ]] && [[ -s "${formal_manifest}" ]] \
    && jq -e '.status == "complete" and .final_global_step == 106500' \
      "${formal_manifest}" >/dev/null; then
  echo "TiMotion formal 106500-step run is already complete; reusing it."
else
  conda run --no-capture-output -n New_MMM python -u tools/train_interx_fair.py \
    --exp-name TIMotion-Transformer-LPA-InterX-fair-seed123-bs32x8 \
    --gpu 0 --seed 123 --batch-size 32 --accumulate-grad-batches 8 \
    --microbatches-per-epoch 568 --epoch 1500 --resume auto \
    --lpa --wandb-mode online --semantic-val-every-n-epochs 100 \
    --no-progress-bar
fi

cd /mnt/8T-HDD/wenning/HHC
exec bash tools/queue_timotion_post_training.sh
