import numpy as np

from dataset.interhuman_mgt import (
    InterHumanMGTDrawBatchSampler,
    InterHumanMGTTrainDataset,
    draw_seed,
)


def test_interhuman_mgt_train_records_match_official_support():
    dataset = InterHumanMGTTrainDataset()
    assert len(dataset) == 5637 * 2
    assert dataset.records[0][0] == dataset.records[1][0]
    assert dataset.records[0][1] is False
    assert dataset.records[1][1] is True


def test_seeded_mgt_draw_is_exact_and_worker_rng_independent():
    dataset = InterHumanMGTTrainDataset()
    seed = draw_seed(123, 4, 0, dataset.records[0][0])
    first = dataset.materialize(0, seed)
    np.random.seed(999)
    second = dataset.materialize(0, seed)
    for key in (
        'name', 'caption', 'caption_index', 'crop_start', 'left_right_swap',
        'anchor_person', 'output_person_swap', 'processed_length', 'draw_seed',
    ):
        assert first[key] == second[key]
    np.testing.assert_array_equal(first['motion1'], second['motion1'])
    np.testing.assert_array_equal(first['motion2'], second['motion2'])
    assert first['motion1'].shape == first['motion2'].shape == (300, 262)


def test_left_right_record_changes_caption_and_motion_under_same_choices():
    dataset = InterHumanMGTTrainDataset()
    # Locate an interaction whose first caption includes both laterality words.
    base = next(
        index for index in range(0, len(dataset), 2)
        if 'left' in dataset.adapter.texts(dataset.records[index][0])[0]
        and 'right' in dataset.adapter.texts(dataset.records[index][0])[0]
    )
    chosen = None
    for seed in range(10000):
        if dataset.materialize(base, seed)['caption_index'] == 0:
            chosen = seed
            break
    assert chosen is not None
    original = dataset.materialize(base, chosen)
    swapped = dataset.materialize(base + 1, chosen)
    assert original['caption'] != swapped['caption']
    assert not np.array_equal(original['motion1'], swapped['motion1'])
    assert original['processed_length'] == swapped['processed_length']


def test_mgt_draw_sampler_resume_is_exact_and_epoch_draws_change():
    dataset = InterHumanMGTTrainDataset()
    # Large batch makes a short test cross an epoch boundary.
    batch_size = len(dataset) // 3
    full = list(InterHumanMGTDrawBatchSampler(
        dataset, batch_size=batch_size, total_steps=7, base_seed=123,
    ))
    resumed = list(InterHumanMGTDrawBatchSampler(
        dataset, batch_size=batch_size, total_steps=7,
        base_seed=123, start_step=4,
    ))
    assert resumed == full[4:]
    assert all(len(batch) == batch_size for batch in full)
    first_epoch = {index for batch in full[:3] for index, _ in batch}
    assert len(first_epoch) == batch_size * 3
    # The same dataset record can be revisited in a later epoch, but its draw
    # seed is epoch/position bound and therefore changes reproducibly.
    first_seeds = {index: seed for batch in full[:3] for index, seed in batch}
    later = [(index, seed) for batch in full[3:] for index, seed in batch if index in first_seeds]
    assert later and all(seed != first_seeds[index] for index, seed in later)
