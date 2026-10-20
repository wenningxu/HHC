import numpy as np

from utils.interhuman_representation import (
    BODY_MASK,
    CASCADE_CHANNEL_ORDER,
    CHANNEL_MASKS,
    FEATURE_DIM,
    FOOT_CONTACT_SLICE,
    PAIR_CHANNEL_MASKS,
    PAIR_FEATURE_DIM,
    ROOT_MASK,
    validate_partition,
)


def test_interhuman_two_channels_are_disjoint_and_exhaustive():
    stacked = np.stack(CHANNEL_MASKS)
    assert stacked.shape == (2, FEATURE_DIM)
    assert np.all(stacked.sum(axis=0) == 1)
    assert tuple(mask.sum() for mask in CHANNEL_MASKS) == (256, 6)
    assert CASCADE_CHANNEL_ORDER == (1, 0)
    assert validate_partition()['channel_order'] == (
        'body', 'root'
    )


def test_root_contains_only_pelvis_position_and_velocity():
    expected = np.zeros(FEATURE_DIM, dtype=bool)
    expected[0:3] = True
    expected[66:69] = True
    assert np.array_equal(ROOT_MASK, expected)


def test_foot_contact_is_body_only():
    assert BODY_MASK[FOOT_CONTACT_SLICE].all()
    assert not ROOT_MASK[FOOT_CONTACT_SLICE].any()


def test_split_and_merge_by_masks_is_exact():
    motion = np.arange(4 * FEATURE_DIM, dtype=np.float32).reshape(4, FEATURE_DIM)
    rebuilt = np.zeros_like(motion)
    for mask in CHANNEL_MASKS:
        rebuilt[:, mask] = motion[:, mask]
    assert np.array_equal(rebuilt, motion)


def test_paired_masks_jointly_encode_both_people():
    stacked = np.stack(PAIR_CHANNEL_MASKS)
    assert stacked.shape == (2, PAIR_FEATURE_DIM) == (2, 524)
    assert np.all(stacked.sum(axis=0) == 1)
    assert tuple(mask.sum() for mask in PAIR_CHANNEL_MASKS) == (512, 12)
    for pair_mask, person_mask in zip(PAIR_CHANNEL_MASKS, CHANNEL_MASKS):
        assert np.array_equal(pair_mask[:FEATURE_DIM], person_mask)
        assert np.array_equal(pair_mask[FEATURE_DIM:], person_mask)


def test_arms_remain_in_body_with_positions_velocities_and_rotations():
    for joint in (13, 14, 16, 17, 18, 19, 20, 21):
        assert BODY_MASK[joint * 3:(joint + 1) * 3].all()
        assert BODY_MASK[66 + joint * 3:66 + (joint + 1) * 3].all()
        assert BODY_MASK[132 + (joint - 1) * 6:132 + joint * 6].all()
