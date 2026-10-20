import json

from tools.build_interhuman_validation_mm_requests import (
    DESTINATION,
    SOURCE,
    build_payload,
    sha256_file,
)


def test_validation_mmodality_requests_are_frozen_val_only_and_replay_exactly():
    payload = json.loads(DESTINATION.read_text())
    source = json.loads(SOURCE.read_text())
    assert payload == build_payload()
    assert payload['split'] == 'val'
    assert payload['prompts'] == 100 and payload['repeats'] == 11
    assert payload['pairs'] == 10
    assert payload['source_validation_request_sha256'] == sha256_file(SOURCE)
    replay = []
    for prompt_index, row in enumerate(payload['requests']):
        assert row['prompt_index'] == prompt_index
        replay_row = dict(row)
        replay_row.pop('prompt_index')
        replay.append(replay_row)
    assert replay == source['requests'][:100]
    assert len({row['identifier'] for row in payload['requests']}) == 100
