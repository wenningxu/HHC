"""Fail-closed adapter around the released InterMask InterHuman preprocessing.

The numerical motion conversion is called directly from the original
InterMask source tree.  HHC owns only path resolution, source-identity checks,
normalization with explicit files, and deterministic split enumeration.
"""

from __future__ import annotations

import hashlib
import importlib
import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


EXPECTED_SOURCE_HASHES = {
    'data/interhuman.py': 'abf921b687e62f73f101eda789f2c13ca8cd615a3866700e562fe2d4104d9fa1',
    'data/utils.py': '16cce02600004466a6b0aed45f383b248f2b8fa61577faee3caba3281864abf0',
    'data/quaternion.py': '7cefe9caa94daf15da55a746ef1f74f96eeac636ebbccbed6da95de9c7747722',
}
EXPECTED_MEAN_HASH = '2f97b88889c773551cd597c32e88bfa070c2fb1721a992a80bfb7cf4c22e03b6'
EXPECTED_STD_HASH = 'dd5988b979e9803786ab497f5945316c2d52b97c90b593d0427220fd49b03e4b'


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _require_hash(path: Path, expected: str) -> None:
    actual = _sha256(path)
    if actual != expected:
        raise ValueError(f'official artifact changed: {path}: {actual} != {expected}')


def _load_official_modules(intermask_root: Path):
    """Import released preprocessing without silently accepting module shadowing."""
    intermask_root = intermask_root.resolve()
    for relative, expected in EXPECTED_SOURCE_HASHES.items():
        _require_hash(intermask_root / relative, expected)

    existing = sys.modules.get('data')
    if existing is not None:
        expected_data_root = intermask_root / 'data'
        package_paths = [
            Path(value).resolve() for value in getattr(existing, '__path__', ())
        ]
        source_value = getattr(existing, '__file__', None)
        source = Path(source_value).resolve() if source_value else None
        valid_package = expected_data_root in package_paths
        valid_module = source is not None and expected_data_root in source.parents
        if not (valid_package or valid_module):
            raise RuntimeError(
                f'cannot load official InterMask preprocessing: data is already '
                f'imported from {source or package_paths}'
            )

    root_string = str(intermask_root)
    inserted = root_string not in sys.path
    if inserted:
        sys.path.insert(0, root_string)
    try:
        interhuman = importlib.import_module('data.interhuman')
        data_utils = importlib.import_module('data.utils')
        quaternion = importlib.import_module('data.quaternion')
    finally:
        if inserted:
            sys.path.remove(root_string)

    for module in (interhuman, data_utils, quaternion):
        source = Path(module.__file__).resolve()
        if intermask_root not in source.parents:
            raise RuntimeError(f'official InterMask module resolved incorrectly: {source}')
    return interhuman, data_utils, quaternion


class OfficialInterHumanAdapter:
    """Read and preprocess official InterHuman pairs with frozen provenance."""

    def __init__(
        self,
        data_root='dataset/interhuman_official',
        intermask_root='/mnt/8T-HDD/wenning/InterMask',
        mean_path='/mnt/8T-HDD/wenning/InterMask/data/stats/global_mean.npy',
        std_path='/mnt/8T-HDD/wenning/InterMask/data/stats/global_std.npy',
    ):
        self.data_root = Path(data_root).resolve()
        self.intermask_root = Path(intermask_root).resolve()
        self.mean_path = Path(mean_path).resolve()
        self.std_path = Path(std_path).resolve()
        _require_hash(self.mean_path, EXPECTED_MEAN_HASH)
        _require_hash(self.std_path, EXPECTED_STD_HASH)
        self.mean = np.load(self.mean_path).astype(np.float32)
        self.std = np.load(self.std_path).astype(np.float32)
        if self.mean.shape != (262,) or self.std.shape != (262,):
            raise ValueError(
                f'official InterHuman stats must be 262D: {self.mean.shape}, {self.std.shape}'
            )
        if not np.all(np.isfinite(self.std)) or np.any(self.std <= 0):
            raise ValueError('official InterHuman std is non-finite or non-positive')
        self.interhuman, self.data_utils, self.quaternion = _load_official_modules(
            self.intermask_root
        )

    def split_ids(
        self, split: str, require_motion=True, require_text=False,
        min_raw_length=None,
    ) -> list[str]:
        if split not in {'train', 'val', 'test'}:
            raise ValueError(f'unknown InterHuman split: {split!r}')
        path = self.data_root / 'split' / f'{split}.txt'
        identifiers = [line.strip() for line in path.read_text().splitlines() if line.strip()]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError(f'duplicate identifiers in official split {path}')
        if require_text and not require_motion:
            raise ValueError('InterHuman text eligibility requires a motion pair')
        if require_motion:
            identifiers = [identifier for identifier in identifiers if self._has_pair(identifier)]
        if require_text and min_raw_length is None:
            min_raw_length = 15
        if min_raw_length is not None:
            identifiers = [
                identifier for identifier in identifiers
                if self._pair_raw_length(identifier) >= int(min_raw_length)
            ]
        if require_text:
            identifiers = [
                identifier for identifier in identifiers
                if self._text_status(identifier) == 'ok'
            ]
        return identifiers

    def _person_path(self, identifier: str, person: int) -> Path:
        if person not in (1, 2):
            raise ValueError(f'person must be 1 or 2, got {person}')
        return self.data_root / 'motions_processed' / f'person{person}' / f'{identifier}.npy'

    def _has_pair(self, identifier: str) -> bool:
        return self._person_path(identifier, 1).is_file() and self._person_path(identifier, 2).is_file()

    def _pair_raw_length(self, identifier: str) -> int:
        lengths = [
            int(np.load(self._person_path(identifier, person), mmap_mode='r').shape[0])
            for person in (1, 2)
        ]
        if lengths[0] != lengths[1]:
            raise ValueError(f'InterHuman pair lengths differ for {identifier}: {lengths}')
        return lengths[0]

    def texts(self, identifier: str) -> list[str]:
        path = self.data_root / 'annots' / f'{identifier}.txt'
        # Released code retains blank lines as empty-caption choices.
        with path.open('r') as handle:
            texts = [line.replace('\n', '') for line in handle.readlines()]
        if not texts:
            raise ValueError(f'zero-line InterHuman annotation in {path}')
        return texts

    def _text_status(self, identifier: str) -> str:
        path = self.data_root / 'annots' / f'{identifier}.txt'
        if not path.is_file():
            return 'missing'
        try:
            with path.open('r') as handle:
                texts = [line.replace('\n', '') for line in handle.readlines()]
        except UnicodeDecodeError:
            return 'non_utf8'
        return 'ok' if texts else 'zero_line_file'

    def split_text_exclusions(self, split: str) -> dict[str, list[str]]:
        motion_ids = self.split_ids(split, require_motion=True, require_text=False)
        result = {
            'missing': [], 'non_utf8': [], 'zero_line_file': [],
            'shorter_than_15_raw_frames': [],
        }
        for identifier in motion_ids:
            status = self._text_status(identifier)
            if status != 'ok':
                result[status].append(identifier)
            elif self._pair_raw_length(identifier) < 15:
                result['shorter_than_15_raw_frames'].append(identifier)
        return result

    def _load_raw_pair(self, identifier: str, left_right_swap: bool, min_length: int):
        if not self._has_pair(identifier):
            raise FileNotFoundError(f'missing official InterHuman motion pair {identifier}')
        loaded = []
        for person in (1, 2):
            motion, swapped = self.data_utils.load_motion(
                str(self._person_path(identifier, person)), min_length, swap=left_right_swap
            )
            if motion is None:
                raise ValueError(
                    f'InterHuman motion {identifier}/person{person} is shorter than {min_length}'
                )
            loaded.append(swapped if left_right_swap else motion)
        return loaded

    def process_pair(
        self,
        identifier: str,
        *,
        left_right_swap=False,
        anchor_person=1,
        output_person_swap=False,
        normalize=True,
        min_length=15,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Apply the exact released pair alignment and optional normalization."""
        motion1, motion2 = self._load_raw_pair(identifier, left_right_swap, min_length)
        return self.process_loaded_pair(
            motion1,
            motion2,
            anchor_person=anchor_person,
            output_person_swap=output_person_swap,
            normalize=normalize,
        )

    def process_loaded_pair(
        self,
        motion1: np.ndarray,
        motion2: np.ndarray,
        *,
        anchor_person=1,
        output_person_swap=False,
        normalize=True,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Process already selected raw frames using the released pair logic."""
        if motion1.shape != motion2.shape or motion1.ndim != 2 or motion1.shape[1] != 192:
            raise ValueError(
                f'loaded InterHuman motions must match [T,192], got '
                f'{motion1.shape}, {motion2.shape}'
            )
        if anchor_person not in (1, 2):
            raise ValueError(f'anchor_person must be 1 or 2, got {anchor_person}')
        if anchor_person == 2:
            motion1, motion2 = motion2, motion1
        motion1, root_quat1, root_pos1 = self.interhuman.process_motion_np(
            motion1, 0.001, 0, n_joints=22
        )
        motion2, root_quat2, root_pos2 = self.interhuman.process_motion_np(
            motion2, 0.001, 0, n_joints=22
        )
        relative_rotation = self.quaternion.qmul_np(
            root_quat2, self.quaternion.qinv_np(root_quat1)
        )
        angle = np.arctan2(
            relative_rotation[:, 2:3], relative_rotation[:, 0:1]
        )
        xz = self.quaternion.qrot_np(root_quat1, root_pos2 - root_pos1)[:, [0, 2]]
        relative = np.concatenate([angle, xz], axis=-1)[0]
        motion2 = self.interhuman.rigid_transform(relative, motion2)

        motion1 = np.asarray(motion1, dtype=np.float32)
        motion2 = np.asarray(motion2, dtype=np.float32)
        if motion1.shape != motion2.shape or motion1.ndim != 2 or motion1.shape[1] != 262:
            raise ValueError(
                f'official InterHuman preprocessing returned {motion1.shape}, {motion2.shape}'
            )
        if normalize:
            motion1 = (motion1 - self.mean) / self.std
            motion2 = (motion2 - self.mean) / self.std
        if output_person_swap:
            motion1, motion2 = motion2, motion1
        return motion1.astype(np.float32), motion2.astype(np.float32)

    def frozen_text_pair_request(
        self,
        identifier: str,
        *,
        caption_index=0,
        crop_start=0,
        max_raw_frames=300,
        left_right_swap=False,
        anchor_person=1,
        output_person_swap=False,
        normalize=True,
        pad_to=300,
    ) -> dict:
        """Materialize every random choice made by the released text dataset."""
        captions = self.texts(identifier)
        if caption_index < 0 or caption_index >= len(captions):
            raise IndexError(f'caption_index {caption_index} for {identifier}/{len(captions)}')
        caption = captions[caption_index]
        if left_right_swap:
            caption = (
                caption.replace('left', 'tmp').replace('right', 'left')
                .replace('tmp', 'right').replace('clockwise', 'tmp')
                .replace('counterclockwise', 'clockwise')
                .replace('tmp', 'counterclockwise')
            )
        caption = caption.strip()
        motion1, motion2 = self._load_raw_pair(identifier, left_right_swap, min_length=15)
        raw_length = motion1.shape[0]
        max_raw_frames = int(max_raw_frames)
        if raw_length > max_raw_frames:
            # Preserve released random.choice(range(0, length-max_gt_length)):
            # its upper endpoint is deliberately exclusive.
            maximum_start = raw_length - max_raw_frames - 1
            if crop_start < 0 or crop_start > maximum_start:
                raise ValueError(
                    f'crop_start {crop_start} is outside released range '
                    f'[0,{maximum_start}] for raw length {raw_length}'
                )
            raw_frames = max_raw_frames
        else:
            if crop_start != 0:
                raise ValueError(
                    f'released loader requires crop_start=0 when raw length '
                    f'{raw_length} <= {max_raw_frames}'
                )
            raw_frames = raw_length
        if raw_frames < 15:
            raise ValueError(f'crop leaves only {raw_frames} raw frames for {identifier}')
        end = crop_start + raw_frames
        processed = self.process_loaded_pair(
            motion1[crop_start:end], motion2[crop_start:end],
            anchor_person=anchor_person,
            output_person_swap=output_person_swap,
            normalize=normalize,
        )
        processed_length = processed[0].shape[0]
        if pad_to is not None:
            if processed_length > pad_to:
                raise ValueError(
                    f'processed length {processed_length} exceeds pad_to={pad_to}'
                )
            padded = []
            for motion in processed:
                padding = np.zeros((pad_to - processed_length, 262), dtype=np.float32)
                padded.append(np.concatenate([motion, padding], axis=0))
            processed = tuple(padded)
        return {
            'identifier': identifier,
            'caption': caption,
            'caption_index': int(caption_index),
            'crop_start': int(crop_start),
            'raw_frames': int(raw_frames),
            'processed_length': int(processed_length),
            'left_right_swap': bool(left_right_swap),
            'anchor_person': int(anchor_person),
            'output_person_swap': bool(output_person_swap),
            'motion1': processed[0],
            'motion2': processed[1],
        }

    def denormalize(self, motion: np.ndarray) -> np.ndarray:
        if motion.shape[-1] != 262:
            raise ValueError(f'InterHuman motion must end in 262D, got {motion.shape}')
        return (motion * self.std + self.mean).astype(np.float32)


class InterHumanVQWindowDataset(Dataset):
    """Lazy, deterministic equivalent of InterMask's VQ window corpus.

    Each official interaction contributes the Cartesian product of the
    released loader's left/right augmentation, coordinate-anchor choice and
    final output-person order: eight paired variants. Every returned window is
    ``[person1_262, person2_262]`` = 524D so one HHC token triple jointly
    represents the dyad.
    Unlike the released ``os.walk`` construction, ordering is explicitly bound
    to the official split file.  Numerical preprocessing is still delegated to
    the hash-verified released implementation.
    """

    variants = (
        ('anchor1', False, 1, False),
        ('anchor1_lr_swap', True, 1, False),
        ('anchor1_output_swap', False, 1, True),
        ('anchor1_lr_and_output_swap', True, 1, True),
        ('anchor2', False, 2, False),
        ('anchor2_lr_swap', True, 2, False),
        ('anchor2_output_swap', False, 2, True),
        ('anchor2_lr_and_output_swap', True, 2, True),
    )

    def __init__(
        self,
        split='train',
        window_size=64,
        window_stride=10,
        identifiers=None,
        cache_pairs=16,
        **adapter_kwargs,
    ):
        self.adapter = OfficialInterHumanAdapter(**adapter_kwargs)
        self.split = split
        self.window_size = int(window_size)
        self.window_stride = int(window_stride)
        self.cache_pairs = int(cache_pairs)
        if self.window_size <= 0 or self.window_stride <= 0:
            raise ValueError('window_size and window_stride must be positive')
        if self.cache_pairs < 0:
            raise ValueError('cache_pairs must be non-negative')

        split_ids = self.adapter.split_ids(split)
        if identifiers is not None:
            requested = [str(value) for value in identifiers]
            unavailable = sorted(set(requested) - set(split_ids))
            if unavailable:
                raise ValueError(
                    f'InterHuman identifiers are absent from effective {split}: {unavailable}'
                )
            requested_set = set(requested)
            split_ids = [value for value in split_ids if value in requested_set]

        self.records = []
        cumulative = [0]
        for identifier in split_ids:
            shapes = []
            for person in (1, 2):
                path = self.adapter._person_path(identifier, person)
                motion = np.load(path, mmap_mode='r')
                if motion.ndim != 2 or motion.shape[1] != 492:
                    raise ValueError(f'official raw InterHuman shape changed: {path}: {motion.shape}')
                shapes.append(tuple(motion.shape))
            if shapes[0] != shapes[1]:
                raise ValueError(f'InterHuman pair lengths differ for {identifier}: {shapes}')
            processed_length = shapes[0][0] - 1
            if processed_length < self.window_size:
                continue
            starts = (processed_length - self.window_size) // self.window_stride + 1
            count = starts * len(self.variants)
            self.records.append({
                'identifier': identifier,
                'raw_length': shapes[0][0],
                'processed_length': processed_length,
                'starts': starts,
            })
            cumulative.append(cumulative[-1] + count)
        self.cumulative = np.asarray(cumulative, dtype=np.int64)
        self._pair_cache = OrderedDict()

    def __len__(self):
        return int(self.cumulative[-1])

    def _cached_pair(self, identifier: str, swapped: bool, anchor_person: int):
        key = (identifier, bool(swapped), int(anchor_person))
        if key in self._pair_cache:
            self._pair_cache.move_to_end(key)
            return self._pair_cache[key]
        pair = self.adapter.process_pair(
            identifier,
            left_right_swap=swapped,
            anchor_person=anchor_person,
            normalize=True,
            min_length=self.window_size + 1,
        )
        if self.cache_pairs:
            self._pair_cache[key] = pair
            self._pair_cache.move_to_end(key)
            while len(self._pair_cache) > self.cache_pairs:
                self._pair_cache.popitem(last=False)
        return pair

    def index_metadata(self, item: int) -> dict:
        if item < 0:
            item += len(self)
        if item < 0 or item >= len(self):
            raise IndexError(item)
        record_index = int(np.searchsorted(self.cumulative, item, side='right') - 1)
        record = self.records[record_index]
        local = int(item - self.cumulative[record_index])
        variant = local // record['starts']
        start_number = local % record['starts']
        name, left_right_swap, anchor_person, output_person_swap = self.variants[variant]
        return {
            'identifier': record['identifier'],
            'variant_index': variant,
            'variant': name,
            'left_right_swap': left_right_swap,
            'anchor_person': anchor_person,
            'output_person_swap': output_person_swap,
            'start': start_number * self.window_stride,
        }

    def __getitem__(self, item):
        metadata = self.index_metadata(item)
        pair = self._cached_pair(
            metadata['identifier'], metadata['left_right_swap'],
            metadata['anchor_person'],
        )
        if metadata['output_person_swap']:
            pair = (pair[1], pair[0])
        motion = np.concatenate(pair, axis=-1)
        start = metadata['start']
        window = motion[start:start + self.window_size]
        if window.shape != (self.window_size, 524):
            raise RuntimeError(
                f'InterHuman VQ window shape mismatch at {metadata}: {window.shape}'
            )
        return torch.from_numpy(window.copy()).float()
