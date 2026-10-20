#!/usr/bin/env bash
set -euo pipefail

cd /mnt/8T-HDD/wenning/HHC

formal_service="${1:-hhc-formal-gate-waiter-v7-v3.service}"

if [[ "${formal_service}" != none ]]; then
  while systemctl --user is-active --quiet "${formal_service}"; do
    sleep 30
  done

  formal_result="$(systemctl --user show "${formal_service}" -p Result --value)"
  if [[ "${formal_result}" != success ]]; then
    echo "Formal HHC service result is ${formal_result}; post-formal experiments remain blocked."
    exit 2
  fi
fi

for required in \
  output/hhc_formal_raw12288_repaired_mgt_selection.json \
  output/fair_eval/hhc_formal_raw12288_repaired_mgt_common50_no_contact/summary.json \
  output/fair_eval/hhc_formal_raw12288_repaired_mgt_common50_contact/summary.json
do
  if [[ ! -s "${required}" ]]; then
    echo "Formal HHC service succeeded without required artifact: ${required}"
    exit 2
  fi
done

# The formal HHC process requires its launch-time driver bytes to remain live
# through the final aggregation/source-stability marker.  Reapply the later
# TiMotion selection-manifest binding only after that marker is present.
bash tools/apply_post_hhc_timotion_selection_binding.sh

bash tools/queue_post_hhc_gpu1_experiments.sh none
exec bash tools/queue_hhc_timing_after_post_gpu1.sh none
