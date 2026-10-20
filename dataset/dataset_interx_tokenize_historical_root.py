"""Deterministic Inter-X tokenizer matching MMM's historical MGT cache input.

The historical MGT cache was encoded after 6D conversion with the
``Interx_*_root.npy`` statistics.  This module deliberately lives beside the
raw-stat tokenizer so shape-augmentation cache builds can select that recovered
contract without changing an active historical-replay training process.
"""

import torch

from dataset.dataset_interx_tokenize import (
    CACHE_CROP_POLICY,
    Text2MotionDatasetV2HHI as _RawText2MotionDatasetV2HHI,
    collate_fn,
    cycle,
    deterministic_cache_crop,
)
from utils.motion_process import MotionNormalizerTorchInterX


TOKENIZER_CONTRACT = 'deterministic_historical_root_v1'


class Text2MotionDatasetV2HHI(_RawText2MotionDatasetV2HHI):
    """Source-compatible tokenizer dataset using recovered root statistics."""

    def __init__(self, split_file, w_vectorizer, motion_file, unit_length=4,
                 fill=False, cache_seed=None, include_names=None,
                 beta_dir='dataset/interx/betas',
                 contact_dir='dataset/interx/contact',
                 fallback_beta_dir=None, fallback_contact_dir=None,
                 strict_assets=False, stats_dir=None, stats_prefix=None):
        super().__init__(
            split_file, w_vectorizer, motion_file, unit_length,
            fill=fill, cache_seed=cache_seed, include_names=include_names,
            beta_dir=beta_dir, contact_dir=contact_dir,
            fallback_beta_dir=fallback_beta_dir,
            fallback_contact_dir=fallback_contact_dir,
            strict_assets=strict_assets,
        )
        self.normalizer = MotionNormalizerTorchInterX(
            motion_rep='vel', stats_dir=stats_dir, stats_prefix=stats_prefix
        )
        self.tokenizer_contract = TOKENIZER_CONTRACT


def DATALoader(dataset_name,
               batch_size,
               split_file,
               w_vectorizer,
               motion_file,
               num_workers=8,
               unit_length=8,
               shuffle=True,
               cache_seed=None,
               include_names=None,
               beta_dir='dataset/interx/betas',
               contact_dir='dataset/interx/contact',
               fallback_beta_dir=None,
               fallback_contact_dir=None,
               strict_assets=False,
               stats_dir=None,
               stats_prefix=None):
    """Build a loader whose cache crops and normalization are fully explicit."""
    del dataset_name  # Retained for source-compatible call sites.
    train_set = Text2MotionDatasetV2HHI(
        split_file, w_vectorizer, motion_file, unit_length,
        cache_seed=cache_seed, include_names=include_names,
        beta_dir=beta_dir, contact_dir=contact_dir,
        fallback_beta_dir=fallback_beta_dir,
        fallback_contact_dir=fallback_contact_dir,
        strict_assets=strict_assets,
        stats_dir=stats_dir, stats_prefix=stats_prefix,
    )
    return torch.utils.data.DataLoader(
        train_set,
        batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=True,
    )


__all__ = [
    'CACHE_CROP_POLICY',
    'TOKENIZER_CONTRACT',
    'Text2MotionDatasetV2HHI',
    'DATALoader',
    'collate_fn',
    'cycle',
    'deterministic_cache_crop',
]
