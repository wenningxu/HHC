#!/usr/bin/env bash
set -euo pipefail
cd /workspace/HHC
if ! test -x /opt/hhc-env/bin/python; then
    python -m venv --system-site-packages /opt/hhc-env
    /opt/hhc-env/bin/pip install setuptools==80.9.0 wheel numpy==1.26.4 scipy==1.14.1 h5py==3.12.1 smplx==0.1.28 yacs==0.1.8 einops==0.8.1 matplotlib==3.9.4 contourpy==1.3.1 tqdm ftfy regex requests 'git+https://github.com/openai/CLIP.git@d05afc436d78f1c48dc0dbf8e5980a9d471f35f6'
    /opt/hhc-env/bin/pip install --no-build-isolation chumpy==0.70
fi
if ! test -e /workspace/hhc-env; then ln -sfn /opt/hhc-env /workspace/hhc-env; fi
mkdir -p /mnt/8T-HDD/wenning
ln -sfn /workspace/HHC /mnt/8T-HDD/wenning/HHC
ln -sfn /workspace/HHC/third_party/InterMask /mnt/8T-HDD/wenning/InterMask
if ! test -d /dev/shm/hhc_inputs/dataset/interhuman_official/motions_processed || ! test -f /root/.cache/clip/ViT-L-14-336px.pt; then
    /opt/hhc-env/bin/python -u tools/runpod_stage_assets.py
fi
exec /opt/hhc-env/bin/python -u tools/runpod_start_verified.py
