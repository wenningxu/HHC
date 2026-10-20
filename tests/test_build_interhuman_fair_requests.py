from dataset.interhuman_hhc_adapter import OfficialInterHumanAdapter
from tools.build_interhuman_fair_requests import build_replication, sample_seed


def test_formal_replication_is_deterministic_and_drop_last_exact():
    adapter = OfficialInterHumanAdapter()
    identifiers = adapter.split_ids('test', require_text=True)
    first = build_replication(adapter, identifiers, 0, 1234, 96)
    second = build_replication(adapter, identifiers, 0, 1234, 96)
    assert first == second
    assert first['evaluated_entries'] == 1056
    assert first['effective_test_entries'] == 1098
    assert first['dropped_entries'] == 42
    assert len({row['identifier'] for row in first['requests']}) == 1056
    assert all(row['left_right_swap'] is False for row in first['requests'])
    assert all(row['caption'] == row['caption'].strip() for row in first['requests'])


def test_replications_and_positions_have_independent_sample_seeds():
    values = {
        sample_seed(1234, replication, position, '42')
        for replication in range(2) for position in range(3)
    }
    assert len(values) == 6


def test_request_choices_replay_through_official_adapter():
    adapter = OfficialInterHumanAdapter()
    payload = build_replication(
        adapter, adapter.split_ids('test', require_text=True)[:96], 3, 1234, 96
    )
    request = payload['requests'][0]
    replay = adapter.frozen_text_pair_request(
        request['identifier'], caption_index=request['caption_index'],
        crop_start=request['crop_start'], left_right_swap=False,
        anchor_person=request['anchor_person'],
        output_person_swap=request['output_person_swap'],
    )
    assert replay['caption'] == request['caption']
    assert replay['processed_length'] == request['processed_length']
    assert replay['motion1'].shape == replay['motion2'].shape == (300, 262)
