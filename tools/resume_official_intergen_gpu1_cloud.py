"""Resume the unchanged official baseline from a separately validated receipt on physical GPU1."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys

from resume_official_intergen_full_state_cloud import digest, install_full_resume, WRAPPER_SHA

ROOT=Path('/workspace/HHC')

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint-receipt',type=Path,required=True)
    p.add_argument('--event-dir',type=Path,required=True)
    p.add_argument('--runtime-wrapper',type=Path,required=True)
    a=p.parse_args()
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='1'
    assert Path(__file__).resolve().parents[1]==ROOT
    assert a.runtime_wrapper==ROOT/'tools/run_official_intergen_checkpointed_cloud.py'
    assert digest(a.runtime_wrapper)==WRAPPER_SHA
    assert digest(ROOT/'tools/resume_official_intergen_full_state_cloud.py')=='0093c5346d4cedaa76c77d4175b2ad2a7cc21a447a31faf594b2a2fa3072ba0b'
    r=json.loads(a.checkpoint_receipt.read_text())
    assert r['status']=='verified_full_resume_candidate'
    assert r['step']>=255200 and r['step']%352==0 and r['step']<704000
    assert r['optimizer_states']==196 and (r['epoch']+1)*352==r['step']
    checkpoint=Path(r['snapshot'])
    assert checkpoint.stat().st_size==r['bytes'] and digest(checkpoint)==r['sha256']
    assert ROOT/'output' in a.event_dir.resolve().parents
    a.event_dir.mkdir()
    lock=(a.event_dir/'training.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    apps=subprocess.check_output(['/usr/bin/nvidia-smi','-i','1','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True)
    assert not apps.strip()
    import lightning.pytorch as pl
    assert pl.__version__=='1.9.1'
    install_full_resume(pl,r,a.event_dir)
    # Keep the original activation implementation byte-identical; invoke its
    # functions directly because its CLI guard is pinned to physical GPU0.
    report=json.loads((ROOT/'output/intergen_official_baseline_20261017/ACTIVATION_CHECKPOINT_PROBE.json').read_text())
    assert report['status']=='passed' and report['runtime_wrapper_sha256']==WRAPPER_SHA
    module=runpy.run_path(str(a.runtime_wrapper),run_name='gpu1_activation_runtime')
    module['install_checkpointing']()
    source=ROOT/'third_party/intergen_official_baseline_20261017/tools/train.py'
    sys.argv=[str(source)]
    runpy.run_path(str(source),run_name='__main__')

if __name__=='__main__':
    main()
