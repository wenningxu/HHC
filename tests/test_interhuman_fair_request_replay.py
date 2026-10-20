import hashlib
import json
from pathlib import Path

import torch

from dataset.interhuman_fair_requests import (
    InterHumanFrozenRequestDataset,
    canonical_sha256,
    frozen_request_loader,
)


def test_frozen_request_dataset_replays_exact_metadata_and_shapes():
    dataset = InterHumanFrozenRequestDataset(0)
    assert len(dataset) == 1056
    name, text, motion1, motion2, length = dataset[0]
    request = dataset.requests[0]
    assert name == request['identifier']
    assert text == request['caption']
    assert length == request['processed_length']
    assert motion1.shape == motion2.shape == (300, 262)
    assert motion1.dtype == motion2.dtype == torch.float32
    assert torch.count_nonzero(motion1[length:]) == 0
    assert torch.count_nonzero(motion2[length:]) == 0


def test_frozen_loader_preserves_request_order_and_official_batch_contract():
    dataset = InterHumanFrozenRequestDataset(1)
    loader = frozen_request_loader(1)
    batch = next(iter(loader))
    names, texts, motion1, motion2, lengths = batch
    assert list(names) == [row['identifier'] for row in dataset.requests[:96]]
    assert list(texts) == [row['caption'] for row in dataset.requests[:96]]
    assert motion1.shape == motion2.shape == (96, 300, 262)
    assert lengths.shape == (96,)
    assert len(loader) == 11


def test_manifest_binds_every_replication_file_and_content_digest():
    manifest = json.loads(Path('output/interhuman_fair_requests_v3/manifest.json').read_text())
    assert len(manifest['replications']) == 20
    for entry in manifest['replications']:
        path = Path(entry['path'])
        raw = path.read_bytes()
        assert len(raw) == entry['bytes']
        assert hashlib.sha256(raw).hexdigest() == entry['sha256']
        payload = json.loads(raw)
        assert canonical_sha256(payload['requests']) == entry['request_sha256']
