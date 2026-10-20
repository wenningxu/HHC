"""Install a verified code-only bundle, archiving every overwritten server file."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile
import time

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--bundle-dir',type=Path,required=True)
    p.add_argument('--git',action='store_true')
    a=p.parse_args()
    manifest=json.loads((a.bundle_dir/'code_manifest.json').read_text())
    stamp=str(time.time_ns())
    backup=a.root/'output/edmond_migration_20261020/source_backups'/stamp
    backup.mkdir(parents=True)
    different=[]
    with tarfile.open(backup/'before.tar.gz','w:gz') as saved:
        for name,expected in manifest.items():
            target=a.root/name
            assert a.root.resolve() in target.resolve().parents
            if target.exists() and sha(target)!=expected['sha256']:
                saved.add(target,arcname=name,recursive=False)
                different.append(name)
    (backup/'changed_files.json').write_text(json.dumps(different,indent=2)+'\n')
    with tarfile.open(a.bundle_dir/'code_sync.tar.gz') as archive:
        for member in archive.getmembers():
            assert member.isfile() and member.name in manifest
            target=a.root/member.name
            data=archive.extractfile(member).read()
            assert hashlib.sha256(data).hexdigest()==manifest[member.name]['sha256']
            if target.exists() and sha(target)==manifest[member.name]['sha256']:
                continue
            target.parent.mkdir(parents=True,exist_ok=True)
            target.write_bytes(data)
            target.chmod(member.mode)
    assert all(sha(a.root/n)==v['sha256'] for n,v in manifest.items())
    if a.git:
        def git(*args):
            return subprocess.check_output(['git','-C',str(a.root),*args],text=True).strip()
        if not (a.root/'.git').exists():
            git('init','-b','real_clean')
            git('remote','add','origin','https://github.com/wenningxu/HHC.git')
        else:
            assert git('branch','--show-current')=='real_clean'
            git('diff','--binary')  # Working changes were archived above; unlisted files remain.
        git('fetch',str(a.bundle_dir/'source.bundle'),'real_clean')
        commit=git('rev-parse','FETCH_HEAD')
        current=subprocess.run(['git','-C',str(a.root),'rev-parse','--verify','HEAD'],capture_output=True,text=True)
        if current.returncode==0:
            subprocess.run(['git','-C',str(a.root),'merge-base','--is-ancestor',
                            current.stdout.strip(),commit],check=True)
        git('reset','--mixed',commit)
        # Windows source commits do not encode the existing Linux executable bits.
        # Preserve server permissions while checking identical file contents.
        git('config','core.filemode','false')
        assert not git('diff','--name-only')
    receipt={'status':'source_manifest_verified','files':len(manifest),
             'overwritten_files_archived':different,'backup':str(backup)}
    (backup/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt))

if __name__=='__main__':
    main()
