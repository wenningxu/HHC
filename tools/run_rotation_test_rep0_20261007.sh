#!/usr/bin/env bash
set -euo pipefail
cd /mnt/8T-HDD/wenning/HHC
model=${1:?intermask or intergen}
case "$model" in
  intermask)
    rotation_device=cuda:0
    rotation_checkpoint=output/rotation_methods_20261007/rotation_only_cpu/best.pth
    ;;
  intergen)
    rotation_device=cuda:1
    rotation_checkpoint=output/rotation_methods_20261007/geometric_final/best.pth
    ;;
  *) exit 2 ;;
esac
rotation_python=/home/wenning/miniconda3/envs/New_MMM/bin/python
rotation_out=output/rotation_test_rep0_20261007/$model
rotation_cache=/dev/shm/rotation_test_rep0_20261007_${model}.h5
mkdir -p "$rotation_out"
test ! -e "$rotation_out/run.log"
exec >"$rotation_out/run.log" 2>&1
trap 'rotation_exit=$?; echo "$rotation_exit" >"$rotation_out/exit_code"' EXIT
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
if [ "$model" = intermask ]; then
  "$rotation_python" -u tools/ablate_intermask_generated_rotation.py --cache "$rotation_cache" --replication 0 --limit 1056 --frames 0 --seed 1234 --device "$rotation_device"
else
  "$rotation_python" -u tools/ablate_intergen_generated_rotation.py --source /dev/shm/hhc_intergen_20261006.CtYnWA/source --checkpoint /dev/shm/hhc_intergen_20261006.CtYnWA/intergen.ckpt --cache "$rotation_cache" --replication 0 --limit 1056 --frames 0 --seed 1234 --device "$rotation_device"
fi
"$rotation_python" -u tools/evaluate_frozen_rotation_corrector.py --cache "$rotation_cache" --checkpoint "$rotation_checkpoint" --output "$rotation_out/paired_embeddings.npz" --replication 0 --device "$rotation_device"
"$rotation_python" -u tools/score_rotation_method_metrics.py --embeddings "$rotation_out/paired_embeddings.npz" --replication 0 --random-seed 1234 --text-cache "$rotation_out/text_embeddings.npz" --device cpu
