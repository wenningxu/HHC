"""Build an explicit source-only sync bundle; never include runtime data or secrets."""
from pathlib import Path
import hashlib
import json
import re
import subprocess
import tarfile

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT/'output/edmond_migration_20261020'
DIRECTORIES = {'dataset','eval','models','optimization','optimize','options',
               'train','utils','tools','tests','configs','docs','generate','gen'}
EXTENSIONS = {'.py','.sh','.ps1','.md','.json','.yaml','.yml','.toml','.txt','.cfg','.ini'}
SECRET = re.compile(rb'ghp_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|hf_[A-Za-z0-9]{20,}|sk-[A-Za-z0-9]{25,}|BEGIN (?:OPENSSH|RSA|EC) PRIVATE KEY')

def main():
    tracked=subprocess.check_output(['git','ls-files','-z'],cwd=ROOT).decode().split('\0')
    paths={p for p in tracked if p and (ROOT/p).is_file()}
    for directory in DIRECTORIES:
        for p in (ROOT/directory).rglob('*'):
            if (p.is_file() and not p.is_symlink() and p.suffix in EXTENSIONS
                    and not any(x in p.parts for x in ('__pycache__','.git','node_modules'))
                    and p.stat().st_size < 2_000_000):
                paths.add(p.relative_to(ROOT).as_posix())
    # Only first-party source/configuration; datasets are represented by adapters.
    paths={p for p in paths if not (p.startswith('dataset/') and len(Path(p).parts)>2
                                    and not p.endswith('.py'))}
    records={}
    for name in sorted(paths):
        data=(ROOT/name).read_bytes()
        assert len(data)<2_000_000, ('oversized source',name)
        assert not SECRET.search(data), ('credential pattern',name)
        records[name]={'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}
    DEST.mkdir(parents=True,exist_ok=True)
    (DEST/'code_manifest.json').write_text(json.dumps(records,indent=2)+'\n')
    with tarfile.open(DEST/'code_sync.tar.gz','w:gz') as tar:
        for name in records:tar.add(ROOT/name,arcname=name,recursive=False)
    print(json.dumps({'files':len(records),'bytes':sum(x['bytes'] for x in records.values()),
                      'archive_bytes':(DEST/'code_sync.tar.gz').stat().st_size}))

if __name__=='__main__':
    main()
