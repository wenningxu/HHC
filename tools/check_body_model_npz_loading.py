"""Check single-decompression loading against the repository's original BodyModel."""
import subprocess
import sys
import time
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from optimization.human_body_prior.body_model.body_model import BodyModel


def main():
    source = subprocess.check_output(
        ['git', 'show', 'HEAD:optimization/human_body_prior/body_model/body_model.py'],
        cwd=ROOT, text=True,
    )
    namespace = {'__name__': 'original_body_model',
                 '__package__': 'optimization.human_body_prior.body_model'}
    exec(compile(source, '<original_body_model>', 'exec'), namespace)
    for gender in ('NEUTRAL', 'MALE', 'FEMALE'):
        path = str(ROOT / f'body_models/smplx/SMPLX_{gender}.npz')
        start = time.perf_counter()
        original = namespace['BodyModel'](path)
        old_seconds = time.perf_counter() - start
        reads = []
        getitem = np.lib.npyio.NpzFile.__getitem__

        def tracked(archive, key):
            reads.append(key)
            return getitem(archive, key)

        start = time.perf_counter()
        with patch.object(np.lib.npyio.NpzFile, '__getitem__', tracked):
            current = BodyModel(path)
        new_seconds = time.perf_counter() - start
        assert reads.count('shapedirs') == reads.count('posedirs') == 1, reads
        for key, value in original.named_buffers():
            assert torch.equal(value, dict(current.named_buffers())[key]), key
        results = []
        for model in (original, current):
            betas = torch.linspace(-0.5, 0.5, 10)[None].requires_grad_()
            result = model(betas=betas)
            result.v.square().mean().backward()
            results.append((result.v.detach(), result.Jtr.detach(), betas.grad))
        for before, after in zip(*results):
            assert torch.equal(before, after)
        print(gender, 'buffers/vertices/joints/beta gradients exact;',
              f'load {old_seconds:.3f}s -> {new_seconds:.3f}s', flush=True)


if __name__ == '__main__':
    main()
