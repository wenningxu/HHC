"""Bounded CPU-only reproduction of unchanged official dataset preprocessing."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import runpy
import sys
import time
from types import SimpleNamespace


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--max-samples', type=int, default=600000)
    p.add_argument('--max-seconds', type=int, default=600)
    a = p.parse_args()
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    root = Path('/workspace/HHC')
    source = root/'third_party/intergen_official_baseline_20261017'
    assert root/'output' in a.output.resolve().parents
    assert not a.output.exists()
    os.chdir(source)
    sys.path.insert(0, str(source))
    import numpy as np
    import torch
    import yaml
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    random.seed(20261021)
    np.random.seed(20261021)
    hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in
              [source/'datasets/interhuman.py', source/'utils/utils.py']}
    cfg = SimpleNamespace(**yaml.safe_load((source/'configs/datasets.yaml').read_text())['interhuman'])
    cls = runpy.run_path(str(source/'datasets/interhuman.py'))['InterHumanDataset']
    dataset = cls(cfg)
    assert len(dataset) == 11274
    started = time.monotonic()
    progress = a.output.with_suffix('.progress.json')
    count = 0
    while count < a.max_samples and time.monotonic()-started < a.max_seconds:
        row = dataset[count % len(dataset)]
        assert len(row) == 5
        for motion in row[2:4]:
            assert motion.shape == (300, 262) and np.isfinite(motion).all()
        assert 14 <= int(row[4]) <= 299
        count += 1
        if count % 1000 == 0:
            data = dict(status='running', samples=count, seconds=time.monotonic()-started,
                        pid=os.getpid(), cuda_initialized=torch.cuda.is_initialized())
            temp = progress.with_suffix('.tmp')
            temp.write_text(json.dumps(data))
            os.replace(temp, progress)
            print(json.dumps(data), flush=True)
    assert not torch.cuda.is_initialized()
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest() == h for p, h in hashes.items())
    result = dict(status='bounded_cpu_preprocessing_passed', samples=count,
                  seconds=time.monotonic()-started, source_sha256=hashes,
                  numpy=np.__version__, torch=torch.__version__, cuda_initialized=False,
                  limitation='No training/native-model workload; passing does not establish absence of heap corruption')
    with a.output.open('x') as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
