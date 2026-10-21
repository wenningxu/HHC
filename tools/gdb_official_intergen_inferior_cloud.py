"""Register the actual Python trainer identity when its parent is GDB."""
import argparse
import json
import os
from pathlib import Path
import runpy
import sys


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--identity-file', type=Path, required=True)
    p.add_argument('--driver', type=Path, required=True)
    a, remaining = p.parse_known_args()
    root = Path('/workspace/HHC')
    assert root/'output' in a.identity_file.resolve().parents
    assert a.driver.parent == root/'tools'
    assert a.driver.name in ('resume_official_intergen_gpu1_cloud.py',
                              'resume_official_intergen_single_loader_cloud.py')
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '1'
    fields = Path('/proc/self/stat').read_text().rsplit(')', 1)[1].split()
    temporary = a.identity_file.with_suffix('.tmp')
    with temporary.open('x') as stream:
        json.dump({'pid': os.getpid(), 'birth': fields[19]}, stream)
        stream.flush()
        os.fsync(stream.fileno())
    assert not a.identity_file.exists()
    os.replace(temporary, a.identity_file)
    sys.argv = [str(a.driver)] + remaining
    runpy.run_path(str(a.driver), run_name='__main__')


if __name__ == '__main__':
    main()
