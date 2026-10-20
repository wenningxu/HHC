"""Detach the attributed cloud VQ driver without leaving an SSH job hanging."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT)!='/workspace/HHC':raise RuntimeError('Cloud only')
parser=argparse.ArgumentParser();parser.add_argument('--retry-failed-initialization',action='store_true');args=parser.parse_args()
control=ROOT/'output/hhc_vq_rebuild_control_20261016';control.mkdir(exist_ok=True)
if (control/'STATUS.json').exists():
    old=json.loads((control/'STATUS.json').read_text())
    assert args.retry_failed_initialization and old['status']=='failed' and old['phase']=='initial4'
    assert not Path(old['run']).exists(),'Never reinitialize an existing run'
    for pid in (old['pid'],old['child_pid']):
        try:os.kill(pid,0)
        except ProcessLookupError:pass
        else:raise RuntimeError('The previous process is still alive')
    (control/f'failed_initialization_{int(old["started"])}.json').write_text(json.dumps(old,indent=2)+'\n')
with (control/'launch.log').open('a') as log:
    process=subprocess.Popen([sys.executable,'-u','tools/run_hhc_vq_rebuild_cloud.py','--run',
        str(ROOT/'output/vq_hhc_native_rebuild_seed123_20261016')],cwd=ROOT,
        env=dict(os.environ,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2'),
        stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,close_fds=True)
print(process.pid)
