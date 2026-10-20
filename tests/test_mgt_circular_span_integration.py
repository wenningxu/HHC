"""Small real Lightning loop: train-only scope and checkpoint provenance."""
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, TensorDataset
import utils.motion_masks as masks
from train.train_t2i_trans import atomic_write_json
from tools import train_mgt_circular_span as entry


def test_lightning_scope():
    helper = masks.sample_kinematic_cascade_mask
    original_fit = pl.Trainer.fit
    with tempfile.TemporaryDirectory(prefix='hhc_span_integration_') as directory:
        cfg = SimpleNamespace(
            out_dir=directory, seed=114514, pkeep=.5, shape=False, contact=False,
            resume_training_ckpt=None, resume_trans=None,
            training_mask_contract='historical_pipeline_paper_kinematic_ratio',
            training_loss_contract='historical_all_valid',
            conditioning_contract='historical_mmm_shape_dropout')
        atomic_write_json(str(Path(directory) / 'training_manifest.json'), {'test_fixture': True})

        class Probe(pl.LightningModule):
            def __init__(self, args):
                super().__init__()
                self.save_hyperparameters()
                self.args = args
                self.weight = torch.nn.Parameter(torch.tensor(1.))
                self.train_calls = self.val_calls = 0

            def configure_optimizers(self):
                return torch.optim.SGD(self.parameters(), lr=.1)

            def training_step(self, batch, batch_idx):
                assert masks.sample_kinematic_cascade_mask is not helper
                target = batch[0]
                valid = torch.ones_like(target, dtype=torch.bool)
                _, selected, _, _ = masks.sample_historical_mmm_corruption_with_kinematic_ratios(target, valid, 16)
                transitions = (selected != selected.roll(1, dims=1)).sum(1)
                assert (transitions <= 2).all()
                self.train_calls += 1
                return self.weight.square()

            def validation_step(self, batch, batch_idx):
                assert masks.sample_kinematic_cascade_mask is helper
                self.val_calls += 1

        model = Probe(cfg)
        bound_step = model.training_step
        loader = DataLoader(TensorDataset(torch.zeros(4, 12, 3, dtype=torch.long)), batch_size=2)
        trainer = pl.Trainer(accelerator='cpu', devices=1, max_steps=2,
                             logger=False, enable_checkpointing=False,
                             enable_progress_bar=False, enable_model_summary=False,
                             num_sanity_val_steps=1, val_check_interval=1)

        def run_training(*args, **kwargs):
            trainer.fit(model, train_dataloaders=loader, val_dataloaders=loader)

        with patch.object(entry.runpy, 'run_path', run_training), patch('sys.argv', ['train_mgt_circular_span.py']):
            entry.main()
        assert model.train_calls == 2 and model.val_calls >= 3
        assert model.weight.item() < 1 and model.training_step == bound_step
        assert masks.sample_kinematic_cascade_mask is helper and pl.Trainer.fit is original_fit
        manifest = json.loads((Path(directory) / 'training_manifest.json').read_text())
        assert manifest['runtime_training_mask_intervention']['validation'].startswith('unchanged')
        checkpoint = str(Path(directory) / 'probe.ckpt')
        trainer.save_checkpoint(checkpoint)
        saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
        assert saved['hyper_parameters']['args'].runtime_training_mask_intervention == manifest['runtime_training_mask_intervention']


if __name__ == '__main__':
    test_lightning_scope()
    print('real CPU Lightning train/validation/restoration/checkpoint integration PASS')
