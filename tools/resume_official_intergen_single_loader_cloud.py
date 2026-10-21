"""Isolate the observed DataLoader IPC crash; do not alter official source bytes."""
import inspect
import functools
import json
from pathlib import Path
import runpy
import sys

from resume_official_intergen_full_state_cloud import digest

ROOT = Path('/workspace/HHC')
BASE_SHA = 'd68c7ec051b51763b0f84c03e198ae94933a6e193af31e8a0d659eebcef02fa0'


def install_single_loader(torch, event_dir):
    original = torch.utils.data.DataLoader.__init__
    signature = inspect.signature(original)

    @functools.wraps(original)
    def initialize(self, *args, **kwargs):
        bound = signature.bind(self, *args, **kwargs)
        dataset = bound.arguments['dataset']
        if type(dataset).__module__ == 'datasets.interhuman' and type(dataset).__name__ == 'InterHumanDataset':
            before = bound.arguments.get('num_workers', 0)
            assert before > 0
            assert not bound.arguments.get('persistent_workers', False)
            assert bound.arguments.get('multiprocessing_context') is None
            bound.arguments['num_workers'] = 0
            original(*bound.args, **bound.kwargs)
            evidence = dict(original_num_workers=before, actual_num_workers=self.num_workers,
                            batch_size=self.batch_size, drop_last=self.drop_last,
                            pin_memory=self.pin_memory, sampler=type(self.sampler).__name__,
                            dataset_rows=len(dataset), batches=len(self),
                            rng_sequence_equivalence_claimed=False)
            with (event_dir/'single_loader.jsonl').open('a') as stream:
                stream.write(json.dumps(evidence)+'\n')
            return
        return original(self, *args, **kwargs)

    torch.utils.data.DataLoader.__init__ = initialize


def main():
    base = ROOT/'tools/resume_official_intergen_gpu1_cloud.py'
    assert digest(base) == BASE_SHA
    event_dir = Path(sys.argv[sys.argv.index('--event-dir')+1])
    import torch
    install_single_loader(torch, event_dir)
    runpy.run_path(str(base), run_name='__main__')


if __name__ == '__main__':
    main()
