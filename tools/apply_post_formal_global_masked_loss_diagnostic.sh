#!/usr/bin/env bash
set -euo pipefail

cd /mnt/8T-HDD/wenning/HHC

if pgrep -f 'python -u train/train_t2i_trans.py' >/dev/null 2>&1; then
  echo 'An MGT process is active; refusing diagnostic source patch.'
  exit 2
fi

for required in \
  output/hhc_formal_raw12288_repaired_mgt_selection.json \
  output/post_formal_online_wandb_metrics_application.json \
  output/shape_root_tokenizer_wiring_audit.json
do
  if [[ ! -s "${required}" ]]; then
    echo "Missing post-formal prerequisite: ${required}"
    exit 2
  fi
done

PYTHON=/home/wenning/miniconda3/envs/New_MMM/bin/python

"${PYTHON}" -m py_compile \
  tools/apply_post_formal_online_wandb_metrics.py \
  utils/training_telemetry.py

if rg -q 'paper_masked_global_token_mean' options/option_transformer.py; then
  echo 'Global masked-token loss diagnostic patch already applied.'
else
  git apply --check \
    tools/provenance_snapshots/post_formal_global_masked_loss_diagnostic.patch
  git apply --whitespace=error-all \
    tools/provenance_snapshots/post_formal_global_masked_loss_diagnostic.patch
fi

"${PYTHON}" -m py_compile \
  options/option_transformer.py train/train_t2i_trans.py utils/motion_masks.py
"${PYTHON}" tools/audit_hhc_masked_loss_reduction.py
