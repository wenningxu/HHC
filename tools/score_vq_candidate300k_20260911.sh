#!/usr/bin/env bash
set -euo pipefail
cd /mnt/8T-HDD/wenning/HHC
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4

manifest="checkpoints/hhc_original_mmm_vq_reproduction_seed123_300k_fp32_20260910_v1/training_manifest.json"
run_dir="$(dirname "$manifest")"
checkpoint="$(realpath "$run_dir/net_300000.pth")"
out_dir="$(realpath output/hhc_original_vq_reproduction_20260910)"
prefix="$out_dir/candidate_step000300000"
lock="/tmp/hhc_score_vq_candidate300k_20260911.lock"
training_pid=3599262
check_only=0
wait_mode=0

for arg in "$@"; do
  case "$arg" in
    --check-only) check_only=1 ;;
    --wait) wait_mode=1 ;;
    '') ;;
    *) echo "unknown argument: $arg" >&2; exit 2 ;;
  esac
done

exec 9>"$lock"
flock -n 9 || { echo "another candidate scorer is running" >&2; exit 75; }

while :; do
  status="$(jq -r '.status // empty' "$manifest" 2>/dev/null || true)"
  if [[ "$status" == complete ]]; then break; fi
  if [[ "$wait_mode" == 0 ]]; then
    echo "training manifest is not complete (status=${status:-missing})" >&2
    exit 1
  fi
  if ! kill -0 "$training_pid" 2>/dev/null; then
    echo "training stopped before manifest became complete (pid=$training_pid, status=${status:-missing})" >&2
    exit 1
  fi
  sleep 600
done

for target in \
  "$prefix"_reconstruction.json \
  "$prefix"_reconstruction.h5 \
  "$prefix"_joint_cpu.json \
  "$prefix"_physical_per_sample.json; do
  [[ ! -e "$target" ]] || { echo "refusing to overwrite existing target: $target" >&2; exit 1; }
done

/home/wenning/miniconda3/envs/New_MMM/bin/python - "$manifest" "$checkpoint" <<'PY'
import hashlib, json, sys
from pathlib import Path
import torch

manifest_path, checkpoint = map(Path, sys.argv[1:])
expected_contract = 'c0386308489c7e3d5c2e8726c72f7509340c339a95582c7872f426b8ba2ce7bb'
expected_checkpoint = checkpoint.resolve()
m = json.loads(manifest_path.read_text())
if m.get('schema') != 'hhc_vq_training_manifest_v2': raise SystemExit('manifest schema mismatch')
if m.get('status') != 'complete': raise SystemExit('manifest status is not complete')
if m.get('final_optimizer_step') != 300999: raise SystemExit('final_optimizer_step mismatch')
if m.get('contract_sha256') != expected_contract: raise SystemExit('contract_sha256 mismatch')
if Path(m.get('final_checkpoint', '')).resolve() != expected_checkpoint:
    raise SystemExit('final_checkpoint mismatch')
record = m.get('final_checkpoint_sha256')
if not record: raise SystemExit('manifest has no final checkpoint sha256')
h = hashlib.sha256()
with expected_checkpoint.open('rb') as f:
    for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''): h.update(chunk)
if h.hexdigest() != record: raise SystemExit('final checkpoint sha256 mismatch')
ckpt = torch.load(expected_checkpoint, map_location='cpu', weights_only=False)
if ckpt.get('iter') != 300000: raise SystemExit('checkpoint iter mismatch')
print('candidate readiness OK')
PY

if (( check_only )); then exit 0; fi

mkdir -p "$out_dir"
/home/wenning/miniconda3/envs/New_MMM/bin/python tools/eval_vq_interx_reconstruction.py \
  --checkpoint "$checkpoint" \
  --ground-truth output/interx_validation_raw_ground_truth.h5 \
  --requests output/interx_official_val_requests/validation.npz \
  --output "$prefix"_reconstruction.json \
  --reconstruction-h5 "$prefix"_reconstruction.h5 \
  --input-motion-rep raw --output-motion-rep raw \
  --device cpu --batch-size 16
/home/wenning/miniconda3/envs/New_MMM/bin/python tools/score_interact2ar_joint_validation.py \
  --requests output/interx_official_val_requests/validation.npz \
  --raw-ground-truth output/interx_validation_raw_ground_truth.h5 \
  --generations "$prefix"_reconstruction.h5 \
  --output "$prefix"_joint_cpu.json \
  --device cpu --batch-size 32
/home/wenning/miniconda3/envs/New_MMM/bin/python tools/score_fair_interx_physical.py \
  --requests output/interx_official_val_requests/validation.npz \
  --generations "$prefix"_reconstruction.h5 \
  --output "$prefix"_physical_per_sample.json \
  --device cpu --batch-size 16 --floor-policy valid_frames --save-per-sample
