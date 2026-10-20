import json

from dataset.interhuman_fair_requests import InterHumanValidationRequestDataset
from dataset.interhuman_hhc_adapter import OfficialInterHumanAdapter
from tools.build_interhuman_validation_requests import build_validation_request


def test_validation_builder_is_fixed_val_only_and_deterministic():
    adapter = OfficialInterHumanAdapter()
    first = build_validation_request(adapter)
    second = build_validation_request(adapter)
    assert first == second
    assert first['split'] == 'val'
    assert first['official_loader_effective_entries'] == 547
    assert first['evaluated_entries'] == 480
    train = set(adapter.split_ids('train', require_text=True))
    test = set(adapter.split_ids('test', require_text=True))
    selected = {row['identifier'] for row in first['requests']}
    assert not selected & train
    assert not selected & test


def test_frozen_validation_manifest_replays_exactly():
    dataset = InterHumanValidationRequestDataset()
    assert len(dataset) == 480
    for index in (0, 95, 479):
        name, text, motion1, motion2, length = dataset[index]
        request = dataset.requests[index]
        assert (name, text, length) == (
            request['identifier'], request['caption'], request['processed_length']
        )
        assert motion1.shape == motion2.shape == (300, 262)
        assert json.loads(dataset.request_path.read_text())['split'] == 'val'
