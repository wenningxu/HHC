#!/usr/bin/env bash
set -euo pipefail

cd /mnt/8T-HDD/wenning/HHC

post_service="${1:-hhc-post-formal-v7-v3.service}"

if [[ "${post_service}" != none ]]; then
  while systemctl --user is-active --quiet "${post_service}"; do
    sleep 30
  done

  post_result="$(systemctl --user show "${post_service}" -p Result --value)"
  if [[ "${post_result}" != success ]]; then
    echo "Post-formal service result is ${post_result}; final report remains blocked."
    exit 2
  fi
fi

while pgrep -f 'bash tools/queue_gpu0_cosh_timotion.sh' >/dev/null 2>&1; do
  sleep 30
done

for required in \
  output/mgt_repair_gate.json \
  output/hhc_formal_raw12288_repaired_mgt_selection.json \
  output/hhc_formal_mgt_shape_dependence_pair.json \
  output/hhc_global_masked_loss_control_assessment.json \
  output/hhc_post_50k_training_dispatch.json \
  output/interx_shape_augmentation_results.json \
  output/fair_eval/coshmdm_interx_retrained_2daycap_seed123/summary.json \
  output/timotion_interx_val_selection.json \
  output/fair_eval/timotion_transformer_lpa_interx_retrained_fair/summary.json
do
  if [[ ! -s "${required}" ]]; then
    echo "Final report dependency is missing: ${required}"
    exit 2
  fi
done

dispatch_action="$(/home/wenning/miniconda3/envs/New_MMM/bin/python -c \
  "import json; print(json.load(open('output/hhc_post_50k_training_dispatch.json'))['action'])")"
if [[ "${dispatch_action}" == continue_global_control_to_300k_then_shape \
    && ! -s output/hhc_global_masked_loss_control_300k_results.json ]]; then
  echo 'Final report dependency is missing: output/hhc_global_masked_loss_control_300k_results.json'
  exit 2
fi

exec bash tools/queue_review_report_after_all_training.sh none
