"""Deterministic training requests for HHC MGT on official InterHuman."""

from __future__ import annotations

import hashlib
import random

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from dataset.interhuman_hhc_adapter import OfficialInterHumanAdapter


def draw_seed(base_seed: int, epoch: int, position: int, identifier: str) -> int:
    payload = f'{int(base_seed)}:{int(epoch)}:{int(position)}:{identifier}'.encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], 'big')


class InterHumanMGTTrainDataset(Dataset):
    """Reproduce the released text loader's stochastic support explicitly.

    There are two base records per eligible train interaction: original and
    left/right swapped.  A supplied draw seed independently fixes caption,
    crop, coordinate anchor and final person order.  Motion is processed and
    padded before VQ encoding, matching released InterMask's online-tokenizer
    path rather than cropping an already encoded full-sequence token cache.
    """

    def __init__(self, base_seed=123, adapter=None):
        self.base_seed = int(base_seed)
        self.adapter = adapter or OfficialInterHumanAdapter()
        identifiers = self.adapter.split_ids('train', require_text=True)
        if len(identifiers) != 5637:
            raise ValueError(f'expected 5637 official-loader-eligible train entries, got {len(identifiers)}')
        self.records = [
            (identifier, left_right_swap)
            for identifier in identifiers
            for left_right_swap in (False, True)
        ]

    def __len__(self):
        return len(self.records)

    def materialize(self, item: int, seed: int):
        identifier, left_right_swap = self.records[int(item)]
        python_rng = random.Random(int(seed))
        numpy_rng = np.random.default_rng(int(seed))
        captions = self.adapter.texts(identifier)
        caption_index = python_rng.randrange(len(captions))
        raw_length = self.adapter._pair_raw_length(identifier)
        crop_start = (
            python_rng.randrange(0, raw_length - 300)
            if raw_length > 300 else 0
        )
        anchor_person = 2 if numpy_rng.random() > 0.5 else 1
        output_person_swap = bool(numpy_rng.random() > 0.5)
        replay = self.adapter.frozen_text_pair_request(
            identifier,
            caption_index=caption_index,
            crop_start=crop_start,
            max_raw_frames=300,
            left_right_swap=left_right_swap,
            anchor_person=anchor_person,
            output_person_swap=output_person_swap,
            normalize=True,
            pad_to=300,
        )
        return {
            'name': identifier + ('_swap' if left_right_swap else ''),
            'identifier': identifier,
            'caption': replay['caption'],
            'caption_index': caption_index,
            'crop_start': crop_start,
            'left_right_swap': left_right_swap,
            'anchor_person': anchor_person,
            'output_person_swap': output_person_swap,
            'processed_length': replay['processed_length'],
            'motion1': replay['motion1'],
            'motion2': replay['motion2'],
            'draw_seed': int(seed),
        }

    def __getitem__(self, item):
        if isinstance(item, tuple):
            index, seed = item
        else:
            index = int(item)
            identifier = self.records[index][0]
            seed = draw_seed(self.base_seed, 0, index, identifier)
        value = self.materialize(int(index), int(seed))
        return (
            value['name'],
            value['caption'],
            torch.from_numpy(value['motion1'].copy()).float(),
            torch.from_numpy(value['motion2'].copy()).float(),
            value['processed_length'],
        )


class InterHumanMGTDrawBatchSampler(Sampler):
    """Shuffle without replacement and bind every stochastic draw to its step."""

    def __init__(
        self, dataset: InterHumanMGTTrainDataset, batch_size: int,
        total_steps: int, base_seed: int, start_step: int = 0,
    ):
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.total_steps = int(total_steps)
        self.base_seed = int(base_seed)
        self.start_step = int(start_step)
        self.steps_per_epoch = len(dataset) // self.batch_size
        if self.steps_per_epoch <= 0:
            raise ValueError('InterHuman MGT dataset is smaller than one batch')
        if not 0 <= self.start_step <= self.total_steps:
            raise ValueError('start_step must lie within total_steps')

    def __len__(self):
        return self.total_steps - self.start_step

    def __iter__(self):
        step = self.start_step
        while step < self.total_steps:
            epoch = step // self.steps_per_epoch
            batch_in_epoch = step % self.steps_per_epoch
            permutation = np.random.default_rng(self.base_seed + epoch).permutation(
                len(self.dataset)
            )
            while batch_in_epoch < self.steps_per_epoch and step < self.total_steps:
                start = batch_in_epoch * self.batch_size
                batch_indices = permutation[start:start + self.batch_size].tolist()
                batch = []
                for position, index in enumerate(batch_indices, start=start):
                    identifier = self.dataset.records[index][0]
                    seed = draw_seed(self.base_seed, epoch, position, identifier)
                    batch.append((index, seed))
                yield batch
                batch_in_epoch += 1
                step += 1
