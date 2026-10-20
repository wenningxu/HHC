"""Hash-bound wrapper for the released InterMask InterCLIP evaluator."""

from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import yaml

from dataset.interhuman_hhc_adapter import EXPECTED_MEAN_HASH, EXPECTED_STD_HASH


INTERMASK_ROOT = Path('/mnt/8T-HDD/wenning/InterMask')
EVALUATOR_SOURCE = INTERMASK_ROOT / 'models/evaluator/evaluator_models.py'
EVALUATOR_SOURCE_SHA256 = '5bd9b1f6964d9ec5fe4d31f64f924dbe17b315008d996bc92080991ff7a10b34'
METRICS_SOURCE = INTERMASK_ROOT / 'utils/metrics.py'
METRICS_SOURCE_SHA256 = '9b6461032246313cbbf99c50d358812bd88b8b4333c858105039028a4a7b20d2'
CONFIG_PATH = INTERMASK_ROOT / 'checkpoints/eval_model/eval_model.yaml'
CONFIG_SHA256 = 'df6643ccdc84d0b27ce220603d62f302191d7138030378a834f7b538db2bb61f'
CHECKPOINT_PATH = INTERMASK_ROOT / 'checkpoints/eval_model/interclip.ckpt'
CHECKPOINT_BYTES = 1513626777
CHECKPOINT_SHA256 = '4e14b8ee5c2d965778fe003fb37d98ed8e5c7e0940668515fc7dcfa2831c17f8'
MEAN_PATH = INTERMASK_ROOT / 'data/stats/global_mean.npy'
STD_PATH = INTERMASK_ROOT / 'data/stats/global_std.npy'


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _require_hash(path: Path, expected: str) -> None:
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(f'official InterHuman evaluator artifact changed: {path}: {actual}')


def _load_source_module(name: str, path: Path, expected_hash: str):
    _require_hash(path, expected_hash)
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f'cannot create import spec for {path}')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_official_metrics_module():
    """Load the released metric functions without copying their equations."""
    return _load_source_module(
        '_hhc_official_interhuman_metrics', METRICS_SOURCE, METRICS_SOURCE_SHA256
    )


def load_official_interclip(device='cpu'):
    """Instantiate the exact released evaluator and load its official weights."""
    _require_hash(CONFIG_PATH, CONFIG_SHA256)
    if CHECKPOINT_PATH.stat().st_size != CHECKPOINT_BYTES:
        raise ValueError(
            f'official InterCLIP checkpoint size changed: '
            f'{CHECKPOINT_PATH.stat().st_size} != {CHECKPOINT_BYTES}'
        )
    # The released Lightning checkpoint includes NumPy scalars. PyTorch >=2.6
    # defaults to weights_only=True, which rejects that metadata. Only unpickle
    # after verifying the exact previously audited official artifact.
    _require_hash(CHECKPOINT_PATH, CHECKPOINT_SHA256)
    config_values = yaml.safe_load(CONFIG_PATH.read_text())
    config = SimpleNamespace(**config_values)
    if config.INPUT_DIM != 258:
        raise ValueError(f'official InterCLIP INPUT_DIM changed: {config.INPUT_DIM}')
    module = _load_source_module(
        '_hhc_official_interclip_model', EVALUATOR_SOURCE, EVALUATOR_SOURCE_SHA256
    )
    model = module.InterCLIP(config)
    checkpoint = torch.load(CHECKPOINT_PATH, map_location='cpu', weights_only=False)
    if 'state_dict' not in checkpoint:
        raise ValueError('official InterCLIP checkpoint has no state_dict')
    state = dict(checkpoint['state_dict'])
    for key in list(state):
        if 'model' in key:
            state[key.replace('model.', '')] = state.pop(key)
    model.load_state_dict(state, strict=True)
    return model.to(device).eval(), config


class InterHumanEvaluatorWrapper:
    """Convert HHC normalized motions to released evaluator coordinates.

    InterMask's EvaluationDataset denormalizes generated motion, and eval.py
    switches the GT dataset to normalize=False before scoring. HHC stores
    normalized GT and generated arrays, so both paths need this conversion.
    """

    def __init__(self, model, device='cpu'):
        self.device = torch.device(device)
        self.model = model.to(self.device).eval()
        _require_hash(MEAN_PATH, EXPECTED_MEAN_HASH)
        _require_hash(STD_PATH, EXPECTED_STD_HASH)
        self.mean = torch.from_numpy(np.load(MEAN_PATH).astype(np.float32))
        self.std = torch.from_numpy(np.load(STD_PATH).astype(np.float32))

    @classmethod
    def from_official_checkpoint(cls, device='cpu'):
        model, _ = load_official_interclip(device)
        return cls(model, device)

    def _prepare(self, batch_data):
        name, text, motion1, motion2, motion_lens = batch_data
        del name
        if motion1.shape != motion2.shape or motion1.ndim != 3 or motion1.shape[-1] != 262:
            raise ValueError(
                f'InterHuman evaluator expects paired [B,T,262], got '
                f'{motion1.shape}, {motion2.shape}'
            )
        motions = torch.cat([motion1.detach().float(), motion2.detach().float()], dim=-1)
        batch_size, time = motions.shape[:2]
        motions = (
            motions.reshape(batch_size, time, 2, 262) * self.std.to(motions)
            + self.mean.to(motions)
        ).reshape(batch_size, time, 524)
        lengths = motion_lens.detach().long()
        align_numpy = np.argsort(lengths.cpu().numpy())[::-1].copy()
        align = torch.from_numpy(align_numpy).long()
        motions = motions[align].to(self.device)
        lengths = lengths[align]
        batch, time = motions.shape[:2]
        padded_length = max(min(time, int(value)) for value in lengths)
        prepared = {
            # Official wrapper encodes text in original order, then applies
            # align_idx to the returned text embeddings.
            'text': list(text),
            'motions': motions.reshape(batch, time, -1)[:, :padded_length],
            'motion_lens': lengths,
        }
        return prepared, align.to(self.device)

    @torch.no_grad()
    def get_co_embeddings(self, batch_data):
        batch, align = self._prepare(batch_data)
        motion_embedding = self.model.encode_motion(batch)['motion_emb']
        text_embedding = self.model.encode_text(batch)['text_emb'][align]
        return text_embedding, motion_embedding

    @torch.no_grad()
    def get_motion_embeddings(self, batch_data):
        batch, _ = self._prepare(batch_data)
        return self.model.encode_motion(batch)['motion_emb']


def evaluator_provenance(include_checkpoint_sha256=False):
    result = {
        'contract': 'released_intermask_interclip_interhuman',
        'input_contract': 'normalized_262D_per_person_denormalized_once_before_interclip',
        'wrapper': {'path': str(Path(__file__).resolve()),
                    'sha256': sha256_file(Path(__file__).resolve())},
        'normalization_mean': {'path': str(MEAN_PATH), 'sha256': EXPECTED_MEAN_HASH},
        'normalization_std': {'path': str(STD_PATH), 'sha256': EXPECTED_STD_HASH},
        'evaluator_source': {
            'path': str(EVALUATOR_SOURCE), 'sha256': EVALUATOR_SOURCE_SHA256,
        },
        'metrics_source': {
            'path': str(METRICS_SOURCE), 'sha256': METRICS_SOURCE_SHA256,
        },
        'config': {'path': str(CONFIG_PATH), 'sha256': CONFIG_SHA256},
        'checkpoint': {
            'path': str(CHECKPOINT_PATH), 'bytes': CHECKPOINT_BYTES,
        },
    }
    if include_checkpoint_sha256:
        result['checkpoint']['sha256'] = sha256_file(CHECKPOINT_PATH)
    return result
