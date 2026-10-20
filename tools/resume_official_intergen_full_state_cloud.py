"""Resume the pinned official entry through Lightning's complete checkpoint API.

The original source/config and activation wrapper stay unchanged. Epoch-end
checkpoints precede the official module's manual scheduler hook in Lightning
1.9.1, so replay that pending hook exactly once before the next epoch.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import runpy
import sys
import time
from types import MethodType

ROOT = Path('/workspace/HHC')
WRAPPER_SHA = '54a4b2759c638fd38b0a662cdef3d12b71ac14809d3a00d4fc2b769d0a2e48f3'


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def event(path, value):
    value = dict(value, time_unix=time.time())
    with path.open('x') as handle:
        json.dump(value, handle, indent=2)
        handle.write('\n')
        handle.flush()
        os.fsync(handle.fileno())
    print(json.dumps({'resume_event': str(path), **value}), flush=True)


def install_full_resume(pl, receipt, directory, max_epochs=2000):
    original_fit = pl.Trainer.fit

    class ResumeProof(pl.Callback):
        first_batch = True

        def on_train_start(self, trainer, module):
            assert trainer.global_step == receipt['step']
            optimizer = trainer.optimizers[0]
            steps = [int(value['step']) for value in optimizer.state.values() if 'step' in value]
            assert len(steps) == receipt['optimizer_states']
            assert min(steps) == max(steps) == receipt['step']
            scheduler = module.lr_schedulers()
            saved = receipt['scheduler']
            before = scheduler.state_dict()
            assert before == saved, (before, saved)
            assert [g['lr'] for g in optimizer.param_groups] == saved['_last_lr']
            progress = receipt['loops']['fit_loop']['epoch_progress']['current']
            assert progress['processed'] == receipt['epoch'] + 1
            assert progress['completed'] == receipt['epoch']
            assert scheduler.last_epoch == receipt['epoch']
            # The callback saves before this exact official hook is executed.
            # FitLoop.on_run_start skips to processed epochs after reload.
            module.on_train_epoch_end()
            assert scheduler.last_epoch == receipt['epoch'] + 1
            event(directory/'full_state_restored.json', {
                'status': 'full_state_restored_pending_epoch_hook_replayed',
                'global_step': trainer.global_step, 'current_epoch': trainer.current_epoch,
                'optimizer_states': len(steps), 'optimizer_step_min': min(steps),
                'optimizer_step_max': max(steps), 'scheduler_before': before,
                'scheduler_after': scheduler.state_dict(),
                'checkpoint_sha256': receipt['sha256'],
                'note': 'No checkpoint RNG/dataloader cursor guarantee; not bitwise InterGen replay.',
            })

        def on_train_batch_start(self, trainer, module, batch, batch_idx):
            if not self.first_batch:
                return
            self.first_batch = False
            assert trainer.global_step == receipt['step']
            assert trainer.current_epoch == receipt['epoch'] + 1
            assert module.it == receipt['step']
            assert batch_idx == 0
            event(directory/'first_resumed_batch.json', {
                'status': 'first_resumed_batch_verified', 'global_step': trainer.global_step,
                'epoch': trainer.current_epoch, 'logging_counter': module.it,
                'scheduler_last_epoch': module.lr_schedulers().last_epoch,
                'optimizer_lr': [g['lr'] for g in trainer.optimizers[0].param_groups],
            })

    def fit(self, model=None, *args, **kwargs):
        assert self.max_epochs == max_epochs
        assert model is not None and not kwargs.get('ckpt_path')
        original_start = model.on_train_start

        def restore_logging_counter(module):
            original_start()
            module.it = receipt['step']
            module.epoch = receipt['epoch']

        model.on_train_start = MethodType(restore_logging_counter, model)
        self.callbacks.insert(0, ResumeProof())
        kwargs['ckpt_path'] = receipt['snapshot']
        return original_fit(self, model, *args, **kwargs)

    pl.Trainer.fit = fit
    return original_fit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime-wrapper', type=Path, required=True)
    parser.add_argument('--checkpoint-receipt', type=Path, required=True)
    parser.add_argument('--event-dir', type=Path, required=True)
    args = parser.parse_args()
    assert Path(__file__).resolve().parents[1] == ROOT
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '0'
    assert args.runtime_wrapper.resolve() == ROOT/'tools/run_official_intergen_checkpointed_cloud.py'
    assert digest(args.runtime_wrapper) == WRAPPER_SHA
    receipt = json.loads(args.checkpoint_receipt.read_text())
    assert receipt['status'] == 'verified_full_resume_candidate'
    assert receipt['step'] == 255200 and receipt['epoch'] == 724
    snapshot = Path(receipt['snapshot'])
    assert snapshot.stat().st_size == receipt['bytes'] and digest(snapshot) == receipt['sha256']
    assert ROOT/'output' in args.event_dir.resolve().parents
    args.event_dir.mkdir(exist_ok=True)
    assert not (args.event_dir/'full_state_restored.json').exists()
    import fcntl
    import subprocess
    lock = (args.event_dir/'training.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid',
                                   '--format=csv,noheader,nounits'], text=True)
    assert not apps.strip(), 'GPU already has a compute process; no concurrent launch'
    import lightning.pytorch as pl
    assert pl.__version__ == '1.9.1'
    install_full_resume(pl, receipt, args.event_dir)
    sys.argv = [str(args.runtime_wrapper)]
    runpy.run_path(str(args.runtime_wrapper), run_name='__main__')


if __name__ == '__main__':
    main()
