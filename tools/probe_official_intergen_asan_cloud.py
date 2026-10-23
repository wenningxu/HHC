"""Bounded GPU1 diagnostic; never produces a production training checkpoint."""
import argparse
import json
import os
from pathlib import Path
import runpy
import sys


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--receipt', type=Path, required=True)
    p.add_argument('--updates', type=int, default=10000)
    a = p.parse_args()
    root = Path('/workspace/HHC')
    assert root / 'output' in a.output.resolve().parents
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '1'
    assert 'libasan.so' in os.environ.get('LD_PRELOAD', '')
    assert 1 <= a.updates <= 10000
    a.output.mkdir(parents=True, exist_ok=True)
    identity = {'pid': os.getpid(), 'birth': Path('/proc/self/stat').read_text().rsplit(')', 1)[1].split()[19]}
    with (a.output / 'inferior_identity.json').open('x') as f:
        json.dump(identity, f)
    accepted = json.loads(a.receipt.read_text())
    start = accepted['step']
    stop = start + a.updates
    sys.path.insert(0, str(root / 'tools'))
    import lightning.pytorch as pl

    class Bounded(pl.Callback):
        def on_train_batch_start(self, trainer, module, batch, batch_idx):
            if trainer.global_step == start:
                assert trainer.max_steps == -1
                assert module.lr_schedulers().last_epoch == accepted['epoch'] + 1

        def on_train_batch_end(self, trainer, module, outputs, batch, batch_idx):
            if trainer.global_step % 100 == 0:
                print(json.dumps({'diagnostic_step': trainer.global_step,
                                  'epoch': trainer.current_epoch}), flush=True)
            if trainer.global_step >= stop:
                assert trainer.global_step == stop
                trainer.should_stop = True
                (a.output / 'actual_stop.json').write_text(json.dumps({
                    'global_step': trainer.global_step, 'epoch': trainer.current_epoch,
                    'scheduler_last_epoch': module.lr_schedulers().last_epoch}))

    original = pl.Trainer.__init__

    def bounded(self, *args, **kw):
        kw.update(strategy=pl.strategies.SingleDeviceStrategy(device='cuda:0'),
                  devices=1, callbacks=[Bounded()], enable_checkpointing=False,
                  logger=False, enable_progress_bar=False)
        return original(self, *args, **kw)

    pl.Trainer.__init__ = bounded
    wrapper = root / 'tools/run_official_intergen_checkpointed_cloud.py'
    runpy.run_path(str(wrapper), run_name='diagnostic_paths')
    import configs
    get = configs.get_config

    def isolated(path, *args, **kw):
        cfg = get(path, *args, **kw)
        if path == 'configs/train.yaml':
            cfg = cfg.clone()
            cfg.defrost()
            cfg.GENERAL.CHECKPOINT = str(a.output / 'isolated_logs')
            cfg.freeze()
        return cfg

    configs.get_config = isolated
    driver = root / 'tools/resume_official_intergen_single_loader_cloud.py'
    sys.argv = [str(driver), '--runtime-wrapper', str(wrapper),
                '--checkpoint-receipt', str(a.receipt), '--event-dir', str(a.output / 'events')]
    runpy.run_path(str(driver), run_name='__main__')
    assert (a.output / 'actual_stop.json').exists()
    (a.output / 'completed.json').write_text(json.dumps({
        'status': 'bounded_training_returned', 'expected_stop_step': stop,
        'production_checkpoint_writes': False,
        'scope': 'ASAN preload interceptors only; uninstrumented native direct writes not fully covered'}))


if __name__ == '__main__':
    main()
