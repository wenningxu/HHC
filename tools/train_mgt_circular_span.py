#!/usr/bin/env python
"""Isolated training-only span-mask experiment; historical validation is untouched."""
import functools
import json
import runpy
import shutil
import sys
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
import pytorch_lightning as pl
import utils.motion_masks as masks
from tools.audit_mgt_span_copy_shortcut import circular_mask
from tools.eval_vq_interx_reconstruction import sha256_file


@contextmanager
def training_mask_scope(seed):
    original = masks.sample_kinematic_cascade_mask
    generator = torch.Generator().manual_seed(seed)

    def sample(valid, generator=None):
        mask, ratios, taus = original(valid, generator=generator)
        # ponytail: tiny masks use the verified CPU helper; port only if profiling
        # shows the CPU/GPU synchronization materially affects training throughput.
        span = circular_mask(valid.cpu(), mask.sum(1).cpu(), span_generator).to(valid.device)
        return span, ratios, taus

    span_generator = generator
    with patch.object(masks, 'sample_kinematic_cascade_mask', sample):
        yield


def self_check():
    target = torch.arange(42).reshape(2, 7, 3) % 16
    valid = (torch.arange(7)[None, :] < torch.tensor([7, 4])[:, None]).unsqueeze(-1).expand_as(target)
    target = target.masked_fill(~valid, 17)
    helper = masks.sample_kinematic_cascade_mask
    torch.manual_seed(123)
    ref, mask, ratios, taus = masks.sample_historical_mmm_corruption_with_kinematic_ratios(target, valid, 16)
    expected_rng = torch.get_rng_state()
    torch.manual_seed(123)
    with training_mask_scope(456):
        got, span, sr, st = masks.sample_historical_mmm_corruption_with_kinematic_ratios(target, valid, 16)
    assert torch.equal(expected_rng, torch.get_rng_state())
    assert masks.sample_kinematic_cascade_mask is helper
    assert torch.equal(mask.sum(1), span.sum(1)) and torch.equal(ratios, sr) and torch.equal(taus, st)
    common = valid & ~mask & ~span
    assert torch.equal(ref[common], got[common]) and torch.equal(got[~valid], target[~valid])
    try:
        with training_mask_scope(456):
            raise RuntimeError('check restoration on error')
    except RuntimeError:
        pass
    assert masks.sample_kinematic_cascade_mask is helper
    torch.manual_seed(123)
    again = masks.sample_historical_mmm_corruption_with_kinematic_ratios(target, valid, 16)
    assert all(torch.equal(a, b) for a, b in zip((ref, mask, ratios, taus), again))
    print('training scope counts/ratios/RNG/tail/validation replay/exception restoration PASS')


def main():
    if sys.argv[1:] == ['--self-check']:
        self_check(); return
    original_fit = pl.Trainer.fit

    @functools.wraps(original_fit)
    def fit(trainer, model, *args, **kwargs):
        cfg = model.args
        if (cfg.training_mask_contract != 'historical_pipeline_paper_kinematic_ratio'
                or cfg.training_loss_contract != 'historical_all_valid'
                or cfg.conditioning_contract != 'historical_mmm_shape_dropout'
                or cfg.pkeep != .5 or cfg.shape or cfg.contact
                or cfg.resume_training_ckpt or cfg.resume_trans):
            raise ValueError('span pilot requires fresh matched historical no-shape/no-contact all-valid training')
        manifest_path = Path(cfg.out_dir) / 'training_manifest.json'
        manifest = json.loads(manifest_path.read_text())
        intervention = {
            'contract': 'training_only_circular_span_equal_counts_v1',
            'span_seed': 'args.seed + 1000000 + global_step',
            'validation': 'unchanged historical corruption and generation',
            'random_replacement_and_kinematic_counts': 'unchanged original helper',
            'launcher_sha256': sha256_file(__file__),
            'span_helper_sha256': sha256_file(Path(__file__).with_name('audit_mgt_span_copy_shortcut.py')),
        }
        snapshots = Path(cfg.out_dir) / 'source' / 'span_runtime'
        snapshots.mkdir(parents=True, exist_ok=True)
        for source in (Path(__file__), Path(__file__).with_name('audit_mgt_span_copy_shortcut.py')):
            target = snapshots / source.name
            if target.exists():
                raise FileExistsError(target)
            shutil.copy2(source, target)
            if sha256_file(target) != sha256_file(source):
                raise RuntimeError('runtime source snapshot mismatch')
        intervention['source_snapshot_directory'] = str(snapshots.resolve())
        manifest['runtime_training_mask_intervention'] = intervention
        model.training_step.__func__.__globals__['atomic_write_json'](str(manifest_path), manifest)
        cfg.runtime_training_mask_intervention = intervention
        model.hparams['args'].runtime_training_mask_intervention = intervention
        original_step = model.training_step

        @functools.wraps(original_step)
        def step(*step_args, **step_kwargs):
            with training_mask_scope(int(cfg.seed) + 1000000 + int(model.global_step)):
                return original_step(*step_args, **step_kwargs)

        model.training_step = step
        try:
            return original_fit(trainer, model, *args, **kwargs)
        finally:
            model.training_step = original_step

    with patch.object(pl.Trainer, 'fit', fit):
        runpy.run_path(str(Path(__file__).resolve().parents[1] / 'train/train_t2i_trans.py'), run_name='__main__')


if __name__ == '__main__':
    main()
