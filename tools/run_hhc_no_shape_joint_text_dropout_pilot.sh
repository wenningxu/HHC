#!/usr/bin/env bash
# Conditioning-contract arm: joint local/global dropout 0.2 instead of
# global-only dropout 0.2. All-valid loss matches the original control.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
bash tools/run_hhc_no_shape_masked_only_pilot.sh \
  --exp-name hhc_no_shape_joint_text_dropout_p0p2_pilot50k_seed114514_v1 \
  --training-loss-contract historical_all_valid \
  --conditioning-contract paper_text_cfg --text-drop-prob 0.2 "$@"
