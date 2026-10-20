"""Run on Edmond: resumable cloud-to-cloud copy; never deletes destination files."""
from pathlib import Path
import json
import subprocess
import time

BASE = Path('/mnt/8T-HDD/wenning/hhc_migration_20261020')
SSH = f'ssh -i {BASE}/transfer_key -p 33923 -o BatchMode=yes -o StrictHostKeyChecking=yes -o UserKnownHostsFile={BASE}/known_hosts'
SOURCES = [
    ('/workspace/HHC/', 'workspace/HHC'),
    ('/workspace/intergen-official-env-py38/', 'workspace/intergen-official-env-py38'),
    ('/root/.local/share/uv/python/cpython-3.8.20-linux-x86_64-gnu/', 'python38'),
    ('/usr/local/lib/python3.12/', 'python312-local'),
    ('/opt/hhc-env/', 'hhc-env'),
    ('/root/.cache/clip/', 'cache/clip'),
    ('/dev/shm/hhc_inputs/', 'hhc_inputs'),
]

def main():
    records=[]
    for source, relative in SOURCES:
        target=BASE/relative
        target.mkdir(parents=True, exist_ok=True)
        cmd=['rsync','-aH','--no-owner','--no-group','--partial','--info=progress2',
             '--exclude=__pycache__/', '--exclude=*.pyc']
        if source == '/workspace/HHC/':
            cmd += ['--exclude=/output/intergen_official_baseline_20261017/official_intergen_scratch_20261017/model/']
        cmd += ['-e',SSH,'root@157.157.221.29:'+source,str(target)+'/']
        start=time.time()
        print(json.dumps({'copying':source,'destination':str(target)}),flush=True)
        result=subprocess.run(cmd)
        records.append({'source':source,'destination':str(target),'exit_code':result.returncode,
                        'started':start,'finished':time.time()})
        (BASE/'copy_status.json').write_text(json.dumps(records,indent=2)+'\n')
        if result.returncode not in (0,24):
            raise SystemExit(result.returncode)
    (BASE/'copy_complete.json').write_text(json.dumps({'time':time.time(),'records':records},indent=2)+'\n')

if __name__=='__main__':
    main()
