import torch

from utils.interhuman_vq_evaluation import (
    reconstruct_padded_interhuman_pair,
    reconstruction_errors,
)


class IdentityVQ(torch.nn.Module):
    temporal_downsample = 4

    def forward(self, motion, type='full'):
        assert type == 'full'
        return motion.clone(), motion.new_zeros(()), motion.new_ones((3,))


def test_reconstruct_pair_groups_true_lengths_and_keeps_padding_zero():
    model = IdentityVQ()
    motion1 = torch.randn(3, 16, 262)
    motion2 = torch.randn(3, 16, 262)
    lengths = torch.tensor([9, 12, 9])
    prediction1, prediction2 = reconstruct_padded_interhuman_pair(
        model, motion1, motion2, lengths
    )
    for row, length in enumerate(lengths.tolist()):
        torch.testing.assert_close(prediction1[row, :length], motion1[row, :length])
        torch.testing.assert_close(prediction2[row, :length], motion2[row, :length])
        assert torch.count_nonzero(prediction1[row, length:]) == 0
        assert torch.count_nonzero(prediction2[row, length:]) == 0


def test_reconstruction_errors_are_length_masked_and_in_metres():
    motion1 = torch.zeros(1, 4, 262)
    motion2 = torch.zeros(1, 4, 262)
    prediction1 = motion1.clone()
    prediction2 = motion2.clone()
    # One-metre x offset for every joint/person on two valid frames only.
    prediction1[:, :2, :66] = 1.0
    prediction2[:, :2, :66] = 1.0
    prediction1[:, 2:, :66] = 1000.0
    prediction2[:, 2:, :66] = 1000.0
    mean = torch.zeros(262)
    std = torch.ones(262)
    result = reconstruction_errors(
        motion1, motion2, prediction1, prediction2, torch.tensor([2]), mean, std
    )
    # Each xyz component differs by 1, hence Euclidean joint error sqrt(3).
    assert abs(result['mpjpe_m'] - 3 ** 0.5) < 1e-6
    assert result['valid_person_frames'] == 4
    assert abs(result['normalized_feature_mae'] - 66 / 258) < 1e-7


def test_reconstruct_pair_rejects_impossible_length():
    motion = torch.zeros(1, 10, 262)
    try:
        reconstruct_padded_interhuman_pair(IdentityVQ(), motion, motion, [10])
    except ValueError as exc:
        assert 'needs 12 frames' in str(exc)
    else:
        raise AssertionError('unrepresentable padded length was accepted')
