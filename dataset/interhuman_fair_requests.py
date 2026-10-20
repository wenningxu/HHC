"""Replay frozen InterHuman fair-evaluation requests exactly."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from dataset.interhuman_hhc_adapter import OfficialInterHumanAdapter


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / 'output/interhuman_fair_requests_v3/manifest.json'
DEFAULT_VALIDATION_MANIFEST = ROOT / 'output/interhuman_validation_requests/manifest.json'


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(',', ':'), ensure_ascii=False,
    ).encode('utf-8')).hexdigest()


class InterHumanFrozenRequestDataset(Dataset):
    """Materialize one hash-bound official GT request at a time."""

    def __init__(
        self,
        replication: int,
        manifest_path=DEFAULT_MANIFEST,
        adapter=None,
    ):
        self.manifest_path = Path(manifest_path).resolve()
        self.manifest = json.loads(self.manifest_path.read_text())
        if self.manifest.get('schema') != 'interhuman_fair_requests_manifest_v3':
            raise ValueError(f'unsupported request manifest {self.manifest_path}')
        if self.manifest.get('status') != 'complete':
            raise ValueError(f'incomplete request manifest {self.manifest_path}')
        entries = {
            int(entry['replication']): entry
            for entry in self.manifest['replications']
        }
        if replication not in entries:
            raise ValueError(f'replication {replication} absent from request manifest')
        self.replication = int(replication)
        self.entry = entries[self.replication]
        self.request_path = Path(self.entry['path']).resolve()
        if self.request_path.stat().st_size != int(self.entry['bytes']):
            raise ValueError(f'frozen request size changed: {self.request_path}')
        if sha256_file(self.request_path) != self.entry['sha256']:
            raise ValueError(f'frozen request file changed: {self.request_path}')
        self.payload = json.loads(self.request_path.read_text())
        self.requests = self.payload['requests']
        if canonical_sha256(self.requests) != self.payload['request_sha256']:
            raise ValueError(f'frozen request content digest changed: {self.request_path}')
        if self.payload['request_sha256'] != self.entry['request_sha256']:
            raise ValueError('request file and manifest content digests disagree')
        protocol = self.manifest['protocol']
        if (
            self.payload['evaluated_entries'] != 1056
            or protocol['evaluated_entries_per_replication'] != 1056
            or self.payload['batch_size'] != 96
            or not self.payload['drop_last']
        ):
            raise ValueError('frozen InterHuman main protocol is not 1056/96/drop_last')

        adapter_record = self.manifest['artifacts']['adapter']
        adapter_path = Path(adapter_record['path']).resolve()
        if sha256_file(adapter_path) != adapter_record['sha256']:
            raise ValueError(
                'official InterHuman adapter changed after requests were frozen; '
                'create a versioned request set rather than silently replaying it'
            )
        self.adapter = adapter or OfficialInterHumanAdapter()

    def __len__(self):
        return len(self.requests)

    def __getitem__(self, item):
        request = self.requests[item]
        if int(request['position']) != int(item):
            raise ValueError(f'non-canonical request position at item {item}')
        replay = self.adapter.frozen_text_pair_request(
            request['identifier'],
            caption_index=int(request['caption_index']),
            crop_start=int(request['crop_start']),
            max_raw_frames=300,
            left_right_swap=bool(request['left_right_swap']),
            anchor_person=int(request['anchor_person']),
            output_person_swap=bool(request['output_person_swap']),
            normalize=True,
            pad_to=300,
        )
        for key in ('caption', 'raw_frames', 'processed_length'):
            if replay[key] != request[key]:
                raise ValueError(
                    f'frozen request replay mismatch {request["identifier"]}/{key}: '
                    f'{replay[key]!r} != {request[key]!r}'
                )
        return (
            request['identifier'],
            request['caption'],
            torch.from_numpy(replay['motion1'].copy()).float(),
            torch.from_numpy(replay['motion2'].copy()).float(),
            int(replay['processed_length']),
        )


def frozen_request_loader(
    replication: int,
    manifest_path=DEFAULT_MANIFEST,
    batch_size=96,
    num_workers=0,
):
    if batch_size != 96:
        raise ValueError('formal InterHuman evaluator requires batch_size=96')
    dataset = InterHumanFrozenRequestDataset(replication, manifest_path)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=True,
        num_workers=num_workers,
    )


class InterHumanValidationRequestDataset(Dataset):
    """Replay the one fixed 480-sample official validation request set."""

    def __init__(self, manifest_path=DEFAULT_VALIDATION_MANIFEST, adapter=None):
        self.manifest_path = Path(manifest_path).resolve()
        self.manifest = json.loads(self.manifest_path.read_text())
        if self.manifest.get('schema') != 'interhuman_validation_requests_manifest_v1':
            raise ValueError(f'unsupported validation manifest {self.manifest_path}')
        if self.manifest.get('status') != 'complete':
            raise ValueError(f'incomplete validation manifest {self.manifest_path}')
        entry = self.manifest['request']
        self.request_path = Path(entry['path']).resolve()
        if self.request_path.stat().st_size != int(entry['bytes']):
            raise ValueError(f'validation request size changed: {self.request_path}')
        if sha256_file(self.request_path) != entry['sha256']:
            raise ValueError(f'validation request file changed: {self.request_path}')
        self.payload = json.loads(self.request_path.read_text())
        self.requests = self.payload['requests']
        if canonical_sha256(self.requests) != self.payload['request_sha256']:
            raise ValueError('validation request content digest changed')
        if self.payload['request_sha256'] != entry['request_sha256']:
            raise ValueError('validation request and manifest content digests disagree')
        protocol = self.manifest['protocol']
        if (
            self.payload.get('split') != 'val'
            or self.payload.get('evaluated_entries') != 480
            or protocol.get('official_loader_effective_val_entries') != 547
            or protocol.get('evaluated_entries') != 480
            or self.payload.get('batch_size') != 96
            or not self.payload.get('drop_last')
        ):
            raise ValueError('frozen InterHuman validation protocol is not val/480/96/drop_last')
        adapter_record = self.manifest['artifacts']['adapter']
        if sha256_file(Path(adapter_record['path']).resolve()) != adapter_record['sha256']:
            raise ValueError('InterHuman adapter changed after validation requests were frozen')
        self.adapter = adapter or OfficialInterHumanAdapter()

    def __len__(self):
        return len(self.requests)

    def __getitem__(self, item):
        request = self.requests[item]
        if int(request['position']) != int(item):
            raise ValueError(f'non-canonical validation request position at item {item}')
        replay = self.adapter.frozen_text_pair_request(
            request['identifier'],
            caption_index=int(request['caption_index']),
            crop_start=int(request['crop_start']),
            max_raw_frames=300,
            left_right_swap=False,
            anchor_person=int(request['anchor_person']),
            output_person_swap=bool(request['output_person_swap']),
            normalize=True,
            pad_to=300,
        )
        for key in ('caption', 'raw_frames', 'processed_length'):
            if replay[key] != request[key]:
                raise ValueError(
                    f'validation replay mismatch {request["identifier"]}/{key}: '
                    f'{replay[key]!r} != {request[key]!r}'
                )
        return (
            request['identifier'],
            request['caption'],
            torch.from_numpy(replay['motion1'].copy()).float(),
            torch.from_numpy(replay['motion2'].copy()).float(),
            int(replay['processed_length']),
        )


def validation_request_loader(
    manifest_path=DEFAULT_VALIDATION_MANIFEST,
    batch_size=96,
    num_workers=0,
):
    if batch_size != 96:
        raise ValueError('official InterHuman validation requires batch_size=96')
    return DataLoader(
        InterHumanValidationRequestDataset(manifest_path),
        batch_size=batch_size,
        shuffle=False,
        drop_last=True,
        num_workers=num_workers,
    )
