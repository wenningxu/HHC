#!/usr/bin/env bash
set -euo pipefail

cap_service="${1:-coshmdm-two-day-cap-v2.service}"
cap_status="/mnt/8T-HDD/wenning/HHC/output/coshmdm_training_budget_cap.json"

while systemctl --user is-active --quiet "${cap_service}"; do
  sleep 30
done

if [[ ! -s "${cap_status}" ]]; then
  echo "CoShMDM cap service exited without its status manifest."
  exit 2
fi
status="$(/usr/bin/python3 -c \
  "import json; print(json.load(open('${cap_status}'))['status'])")"
case "${status}" in
  stopped_gracefully|sigterm_sent_after_180s_grace|trainer_exited_before_monitor_signal)
    ;;
  *)
    echo "CoShMDM cap failed closed with status ${status}; refusing post-processing."
    exit 2
    ;;
esac

while pgrep -f 'python -u tools/train.py --dataset interx' >/dev/null 2>&1; do
  sleep 5
done

cd /mnt/8T-HDD/wenning/HHC
exec bash tools/queue_gpu0_cosh_timotion.sh
