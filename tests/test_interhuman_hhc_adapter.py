import numpy as np

from dataset.interhuman_hhc_adapter import (
    InterHumanVQWindowDataset,
    OfficialInterHumanAdapter,
)


def _adapter():
    return OfficialInterHumanAdapter()


def test_official_effective_split_counts():
    adapter = _adapter()
    assert len(adapter.split_ids('train', require_motion=False)) == 6022
    assert len(adapter.split_ids('val', require_motion=False)) == 580
    assert len(adapter.split_ids('test', require_motion=False)) == 1177
    assert len(adapter.split_ids('train')) == 6021
    assert len(adapter.split_ids('val')) == 580
    assert len(adapter.split_ids('test')) == 1176
    assert len(adapter.split_ids('train', require_text=True)) == 5637
    assert len(adapter.split_ids('val', require_text=True)) == 547
    assert len(adapter.split_ids('test', require_text=True)) == 1098
    train_exclusions = adapter.split_text_exclusions('train')
    assert train_exclusions['non_utf8'] == ['5834']
    assert len(train_exclusions['shorter_than_15_raw_frames']) == 383
    assert train_exclusions['zero_line_file'] == []
    test_exclusions = adapter.split_text_exclusions('test')
    assert test_exclusions['non_utf8'] == ['4642']
    assert len(test_exclusions['shorter_than_15_raw_frames']) == 77


def test_official_pair_processing_and_normalization_roundtrip():
    adapter = _adapter()
    raw1, raw2 = adapter.process_pair('3292', normalize=False)
    normalized1, normalized2 = adapter.process_pair('3292', normalize=True)
    assert raw1.shape == raw2.shape == (80, 262)
    assert raw1.dtype == raw2.dtype == np.float32
    assert np.isfinite(raw1).all() and np.isfinite(raw2).all()
    np.testing.assert_allclose(adapter.denormalize(normalized1), raw1, rtol=2e-5, atol=2e-5)
    np.testing.assert_allclose(adapter.denormalize(normalized2), raw2, rtol=2e-5, atol=2e-5)


def test_anchor_choice_and_output_order_are_distinct_official_operations():
    adapter = _adapter()
    anchor1 = adapter.process_pair('3292', normalize=False)
    anchor2 = adapter.process_pair('3292', anchor_person=2, normalize=False)
    output_swapped = adapter.process_pair(
        '3292', output_person_swap=True, normalize=False
    )
    np.testing.assert_array_equal(output_swapped[0], anchor1[1])
    np.testing.assert_array_equal(output_swapped[1], anchor1[0])
    assert not np.array_equal(anchor2[0], output_swapped[0])
    assert not np.array_equal(anchor2[1], output_swapped[1])


def test_left_right_swap_is_an_explicit_distinct_augmentation():
    adapter = _adapter()
    original = adapter.process_pair('3292', normalize=False)
    swapped = adapter.process_pair('3292', left_right_swap=True, normalize=False)
    assert swapped[0].shape == original[0].shape
    assert not np.array_equal(swapped[0], original[0])
    assert not np.array_equal(swapped[1], original[1])


def test_caption_source_is_nonempty():
    texts = _adapter().texts('3292')
    assert texts
    assert all(text == text.strip() and text for text in texts)


def test_released_blank_line_annotations_are_empty_caption_choices():
    assert _adapter().texts('7561') == ['', '', '']


def test_frozen_text_request_materializes_released_crop_and_padding():
    adapter = _adapter()
    short = adapter.frozen_text_pair_request('3292', caption_index=1)
    assert short['raw_frames'] == 81
    assert short['processed_length'] == 80
    assert short['motion1'].shape == short['motion2'].shape == (300, 262)
    assert np.count_nonzero(short['motion1'][80:]) == 0
    assert np.count_nonzero(short['motion2'][80:]) == 0

    long = adapter.frozen_text_pair_request(
        '2576', crop_start=100, anchor_person=2, output_person_swap=True,
    )
    assert long['raw_frames'] == 300
    assert long['processed_length'] == 299
    assert np.count_nonzero(long['motion1'][299:]) == 0
    try:
        adapter.frozen_text_pair_request('2576', crop_start=1537 - 300)
    except ValueError as exc:
        assert 'released range' in str(exc)
    else:
        raise AssertionError('released exclusive crop endpoint was accepted')


def test_frozen_text_request_uses_released_caption_swap_sequence():
    adapter = _adapter()
    original = adapter.frozen_text_pair_request('6876', caption_index=0)
    swapped = adapter.frozen_text_pair_request(
        '6876', caption_index=0, left_right_swap=True
    )
    assert 'left arm' in original['caption'] and 'right arm' in original['caption']
    assert 'right arm' in swapped['caption'] and 'left arm' in swapped['caption']
    assert original['caption'] != swapped['caption']


def test_vq_window_dataset_matches_official_processed_variants():
    dataset = InterHumanVQWindowDataset(
        split='train', identifiers=['3292'], window_size=64,
        window_stride=1, cache_pairs=2,
    )
    # Raw length 81 -> processed length 80 -> 17 starts for each of 8 variants.
    assert len(dataset) == 136
    assert dataset.index_metadata(0) == {
        'identifier': '3292', 'variant_index': 0, 'variant': 'anchor1',
        'left_right_swap': False, 'anchor_person': 1,
        'output_person_swap': False, 'start': 0,
    }
    assert dataset.index_metadata(17)['variant'] == 'anchor1_lr_swap'
    assert dataset.index_metadata(34)['variant'] == 'anchor1_output_swap'
    assert dataset.index_metadata(68)['variant'] == 'anchor2'
    assert dataset.index_metadata(119)['variant'] == 'anchor2_lr_and_output_swap'
    assert dataset.index_metadata(-1)['start'] == 16

    original = dataset.adapter.process_pair('3292', normalize=True)
    swapped = dataset.adapter.process_pair(
        '3292', left_right_swap=True, normalize=True
    )
    anchor2 = dataset.adapter.process_pair(
        '3292', anchor_person=2, normalize=True
    )
    anchor2_swapped = dataset.adapter.process_pair(
        '3292', left_right_swap=True, anchor_person=2, normalize=True
    )
    np.testing.assert_array_equal(
        dataset[0].numpy(), np.concatenate(original, axis=-1)[:64]
    )
    np.testing.assert_array_equal(
        dataset[34].numpy(), np.concatenate(original[::-1], axis=-1)[:64]
    )
    np.testing.assert_array_equal(
        dataset[17].numpy(), np.concatenate(swapped, axis=-1)[:64]
    )
    np.testing.assert_array_equal(
        dataset[68].numpy(), np.concatenate(anchor2, axis=-1)[:64]
    )
    np.testing.assert_array_equal(
        dataset[119].numpy(), np.concatenate(anchor2_swapped[::-1], axis=-1)[:64]
    )


def test_vq_window_stride_and_indexing_are_deterministic():
    first = InterHumanVQWindowDataset(
        split='train', identifiers=['3292'], window_size=64,
        window_stride=4, cache_pairs=0,
    )
    second = InterHumanVQWindowDataset(
        split='train', identifiers=['3292'], window_size=64,
        window_stride=4, cache_pairs=0,
    )
    assert len(first) == len(second) == 40
    assert [first.index_metadata(i) for i in range(len(first))] == [
        second.index_metadata(i) for i in range(len(second))
    ]
