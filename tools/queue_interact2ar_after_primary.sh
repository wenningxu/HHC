#!/usr/bin/env bash
set -euo pipefail

wait_pid="${1:?primary GPU-1 queue PID or 'none' is required}"
if [[ "${wait_pid}" != none ]]; then
  while ps -p "${wait_pid}" >/dev/null 2>&1; do
    sleep 30
  done
fi

cd /mnt/8T-HDD/wenning/HHC

/mnt/8T-HDD/wenning/Interact2Ar/.venv/bin/python -u \
  tools/export_interact2ar_official_interx.py \
  --checkpoint /mnt/8T-HDD/wenning/Interact2Ar/checkpoints/interact2ar_mixed_memory.ckpt \
  --repository-root /mnt/8T-HDD/wenning/Interact2Ar \
  --requests output/fair_interx_requests/replication_00.npz \
  --output output/fair_eval/interact2ar_official_batch32_probe/replication_00_main_limit64.h5 \
  --gpu 1 --batch-size 32 --repeats 1 --seed 1234 --limit 64 --resume

conda run -n New_MMM python -c \
  "import h5py,numpy as np; p='output/fair_eval/interact2ar_official_batch32_probe/replication_00_main_limit64.h5'; f=h5py.File(p); x=f['motions'][:]; lens=f['lengths'][:]; assert x.shape==(64,1,150,672),x.shape; assert np.isfinite(x).all(); assert f['completed_repeats'][:].astype(bool).all(); assert np.all([np.count_nonzero(x[i,:,int(n):])==0 for i,n in enumerate(lens)]); assert f.attrs['batch_size']==32; print({'probe':p,'shape':x.shape,'finite':True,'completed':True,'max_same_length_batch':32})"

conda run --no-capture-output -n New_MMM python -u \
  tools/run_fair_interx_method.py --method interact2ar \
  --label interact2ar_official_fair_no_gt_first_frame \
  --checkpoint /mnt/8T-HDD/wenning/Interact2Ar/checkpoints/interact2ar_mixed_memory.ckpt \
  --gpu 1 --batch-size 32
