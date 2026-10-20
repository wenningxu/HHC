#!/usr/bin/env bash
set -euo pipefail
cd /mnt/8T-HDD/wenning/HHC
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4

manifest="checkpoints/hhc_original_mmm_vq_reproduction_seed124_300k_fp32_20260911_v1/training_manifest.json"
run_dir="$(dirname "$manifest")"
out_dir="$(realpath output/hhc_original_vq_reproduction_20260910)"
prefix="$out_dir/candidate_seed124_step"
lock=/tmp/hhc_score_vq_seed124_milestones_20260911.lock
pid=3703449
contract=d84e96686d035cccba451caa1ef7a34d6a02ca42d43117dcfe31e7a71ce65dcf
artifacts_sha256=509ef8c4e9c428895476ea60397e1f31c6e86e2528991710f59b862e67886ca0
check_only=0

for arg in "$@"; do
  case "$arg" in
    --check-only) check_only=1 ;;
    '') ;;
    *) echo "unknown argument: $arg" >&2; exit 2 ;;
  esac
done
exec 9>"$lock"
flock -n 9 || { echo "another seed124 scorer is running" >&2; exit 75; }

validate_manifest() {
  /home/wenning/miniconda3/envs/New_MMM/bin/python - "$manifest" "$contract" "$artifacts_sha256" <<'PY'
import hashlib, json, sys
from pathlib import Path
p, contract, expected = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
m=json.loads(p.read_text())
if m.get('schema') != 'hhc_vq_training_manifest_v2': raise SystemExit('manifest schema mismatch')
if m.get('contract_sha256') != contract: raise SystemExit('manifest contract mismatch')
actual=hashlib.sha256(json.dumps(m.get('artifacts',{}),sort_keys=True,separators=(',',':')).encode()).hexdigest()
if actual != expected: raise SystemExit('manifest artifacts fingerprint mismatch')
PY
}

score_point() {
  local step="$1" ckpt="$(realpath "$run_dir/net_iter_$(printf '%09d' "$1").pth")" p="$prefix$(printf '%09d' "$1")"
  for target in "${p}_reconstruction.json" "${p}_reconstruction.h5" "${p}_joint_cpu.json" "${p}_physical_per_sample.json"; do
    [[ ! -e "$target" ]] || { echo "refusing existing target: $target" >&2; exit 1; }
  done
  /home/wenning/miniconda3/envs/New_MMM/bin/python - "$ckpt" "$step" <<'PY'
import sys, torch
c=torch.load(sys.argv[1],map_location='cpu',weights_only=False)
if c.get('iter') != int(sys.argv[2]) or c.get('args',{}).get('seed') != 124: raise SystemExit('checkpoint iter/seed mismatch')
PY
  if (( check_only )); then
    echo "check-only seed124 step$(printf '%09d' "$step") OK"
    return 0
  fi
  echo "scoring seed124 step$(printf '%09d' "$step")"
  /home/wenning/miniconda3/envs/New_MMM/bin/python tools/eval_vq_interx_reconstruction.py --checkpoint "$ckpt" --requests output/interx_official_val_requests/validation.npz --ground-truth output/interx_validation_raw_ground_truth.h5 --input-motion-rep raw --output-motion-rep raw --device cpu --batch-size 16 --output "${p}_reconstruction.json" --reconstruction-h5 "${p}_reconstruction.h5"
  /home/wenning/miniconda3/envs/New_MMM/bin/python tools/score_interact2ar_joint_validation.py --requests output/interx_official_val_requests/validation.npz --raw-ground-truth output/interx_validation_raw_ground_truth.h5 --generations "${p}_reconstruction.h5" --output "${p}_joint_cpu.json" --device cpu --batch-size 32
  /home/wenning/miniconda3/envs/New_MMM/bin/python tools/score_fair_interx_physical.py --requests output/interx_official_val_requests/validation.npz --generations "${p}_reconstruction.h5" --output "${p}_physical_per_sample.json" --floor-policy valid_frames --save-per-sample --device cpu --batch-size 16
  echo "completed seed124 step$(printf '%09d' "$step")"
}

validate_manifest
steps=(5000 50000 175000 300000)
for step in "${steps[@]}"; do
  ckpt="$run_dir/net_iter_$(printf '%09d' "$step").pth"
  while :; do
    validate_manifest
    if [[ -f "$ckpt" ]] && (( $(date +%s) - $(stat -c %Y "$ckpt") >= 60 )); then
      echo "ready seed124 step$(printf '%09d' "$step")"
      break
    fi
    if (( check_only )); then
      echo "seed124 step000005000 is not ready" >&2
      exit 1
    fi
    if ! kill -0 "$pid" 2>/dev/null; then
      echo "training exited before seed124 step$(printf '%09d' "$step") was ready" >&2
      exit 1
    fi
    sleep 600
  done
  score_point "$step"
  (( check_only )) && exit 0
done
