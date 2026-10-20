from dataset.interhuman_hhc_adapter import OfficialInterHumanAdapter
from tools.build_interhuman_mm_requests import build_mm_replication


def test_mm_requests_are_deterministic_unique_and_use_full_effective_set():
    adapter = OfficialInterHumanAdapter()
    first = build_mm_replication(adapter, 0)
    second = build_mm_replication(adapter, 0)
    assert first == second
    assert first['full_effective_entries'] == 1098
    assert len(first['requests']) == 100
    assert len({row['source_position_in_full_1098_order'] for row in first['requests']}) == 100
    assert len({row['identifier'] for row in first['requests']}) == 100
    assert all(row['caption'] == row['caption'].strip() for row in first['requests'])


def test_mm_request_replays_exact_official_gt_choice():
    adapter = OfficialInterHumanAdapter()
    request = build_mm_replication(adapter, 2)['requests'][0]
    replay = adapter.frozen_text_pair_request(
        request['identifier'], caption_index=request['caption_index'],
        crop_start=request['crop_start'], anchor_person=request['anchor_person'],
        output_person_swap=request['output_person_swap'],
    )
    assert replay['caption'] == request['caption']
    assert replay['processed_length'] == request['processed_length']

