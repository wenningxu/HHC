#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 GPU_INDEX" >&2
  exit 2
fi

exec /mnt/8T-HDD/wenning/HHC/tools/run_shape_random_mgt_when_gpu_free.sh \
  "$1" optimized
