"""Pinned Lightning CPU regression for the full-state recovery wrapper."""
import ast
import copy
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace

assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
import numpy as np
import torch
import lightning.pytorch as pl
from yacs.config import CfgNode

ROOT = Path('/workspace/HHC')
OUT = ROOT/'output/intergen_official_baseline_20261017/recovery_20261020/cpu_probe_v2'


def main():
    OUT.mkdir(exist_ok=False)
    torch.set_num_threads(1)
    tree = ast.parse((ROOT/'third_party/intergen_official_baseline_20261017/models/utils.py').read_text())
    cls = next(n for n in tree.body if getattr(n, 'name', None) == 'CosineWarmupScheduler')
    ns = {'torch': torch, 'np': np}
    exec(compile(ast.Module(body=[cls], type_ignores=[]), '<official scheduler AST>', 'exec'), ns)
    scheduler_class = ns['CosineWarmupScheduler']
    torch.manual_seed(731)
    data = torch.utils.data.TensorDataset(torch.arange(16).float().reshape(8, 2)/16)
    loader = torch.utils.data.DataLoader(data, batch_size=2)

    class Mini(pl.LightningModule):
        def __init__(self):
            super().__init__()
            self.net = torch.nn.Linear(2, 1)
            self.automatic_optimization = False
            self.cfg = CfgNode({'TRAIN': {'LAST_ITER': 0, 'LAST_EPOCH': 0}})
            self.cfg.freeze()

        def configure_optimizers(self):
            opt = torch.optim.AdamW(self.parameters(), lr=.001, weight_decay=2e-5)
            return [opt], [scheduler_class(opt, 10, 2000)]

        def on_train_start(self):
            self.it = self.cfg.TRAIN.LAST_ITER

        def training_step(self, batch, batch_idx):
            x = batch[0]
            loss = (self.net(x)-x.sum(-1, keepdim=True)).square().mean()
            opt = self.optimizers()
            opt.zero_grad()
            self.manual_backward(loss)
            torch.nn.utils.clip_grad_norm_(self.parameters(), .5)
            opt.step()
            return loss

        def on_train_batch_end(self, outputs, batch, batch_idx):
            self.it += 1

        def on_train_epoch_end(self):
            self.lr_schedulers().step()

    callback = pl.callbacks.ModelCheckpoint(dirpath=OUT/'reference', every_n_epochs=1, save_top_k=-1)
    reference = Mini()
    trainer = pl.Trainer(max_epochs=2, accelerator='cpu', devices=1, logger=False,
        enable_progress_bar=False, enable_model_summary=False, callbacks=[callback])
    trainer.fit(reference, loader)
    snapshot = OUT/'reference/epoch=0-step=4.ckpt'
    saved = torch.load(snapshot, map_location='cpu')
    assert saved['epoch'] == 0 and saved['global_step'] == 4
    receipt = {'snapshot': str(snapshot), 'epoch': 0, 'step': 4,
        'optimizer_states': 2, 'scheduler': saved['lr_schedulers'][0],
        'loops': saved['loops'], 'sha256': 'synthetic_CPU_checkpoint_only'}
    assert receipt['scheduler']['last_epoch'] == 0
    spec = importlib.util.spec_from_file_location('resume_wrapper', ROOT/'tools/resume_official_intergen_full_state_cloud.py')
    wrapper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(wrapper)
    events = OUT/'resumed_events'
    events.mkdir()
    original = wrapper.install_full_resume(pl, receipt, events, max_epochs=2)
    restored = Mini()
    resumed_trainer = pl.Trainer(max_epochs=2, accelerator='cpu', devices=1, logger=False,
        enable_checkpointing=False, enable_progress_bar=False, enable_model_summary=False)
    resumed_trainer.fit(restored, loader)
    pl.Trainer.fit = original
    assert trainer.global_step == resumed_trainer.global_step == 8
    assert reference.it == restored.it == 8
    assert restored.cfg.is_frozen() and restored.cfg.TRAIN.LAST_ITER == 0
    assert all(torch.equal(v, restored.state_dict()[k]) for k, v in reference.state_dict().items())
    opt_a, opt_b = trainer.optimizers[0].state_dict(), resumed_trainer.optimizers[0].state_dict()
    assert opt_a['param_groups'] == opt_b['param_groups']
    for key, state in opt_a['state'].items():
        for field, value in state.items():
            assert torch.equal(value, opt_b['state'][key][field]) if isinstance(value, torch.Tensor) else value == opt_b['state'][key][field]
    assert reference.lr_schedulers().state_dict() == restored.lr_schedulers().state_dict()
    assert not torch.cuda.is_initialized()
    result = {'status': 'passed', 'cpu_cuda_initialized': False, 'lightning': pl.__version__,
        'restored_step': 4, 'final_step': 8, 'final_logging_counter': restored.it,
        'weights_optimizer_scheduler_equal_to_continuous_CPU_reference': True,
        'scope': 'Tiny deterministic CPU manual-optimization model using literal official scheduler; not bitwise InterGen/RNG/dataloader equivalence.',
        'events': [json.loads(p.read_text()) for p in sorted(events.glob('*.json'))]}
    with (OUT/'result.json').open('x') as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
