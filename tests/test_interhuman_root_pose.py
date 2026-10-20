import torch

from tools.diagnose_interhuman_root_pose import decompose, finalize


def test_translation_pose_and_padding_decomposition():
    p = torch.zeros(1, 2, 262)
    lengths, mean, std = torch.tensor([1]), torch.zeros(262), torch.ones(262)
    q = p.clone()
    q[0, 0, :66].reshape(22, 3)[:, 0] = 2
    q[0, 1] = float('nan')  # Padding must never contribute.
    common = finalize(decompose(p, p, q, q, lengths, mean, std))
    assert common['raw_mpjpe_m'] == common['root_position_error_m'] == common['pair_root_center_error_m'] == 2
    assert common['root_relative_body_mpjpe_m'] == common['pair_root_vector_error_m'] == common['pair_center_aligned_mpjpe_m'] == 0
    opposite = finalize(decompose(p, p, q, -q, lengths, mean, std))
    assert opposite['pair_root_vector_error_m'] == 4 and opposite['pair_root_center_error_m'] == 0
    assert opposite['root_relative_body_mpjpe_m'] == 0 and opposite['pair_center_aligned_mpjpe_m'] == 2
    pose = p.clone()
    pose[0, 0, 4] = 3
    local = finalize(decompose(p, p, pose, p, lengths, mean, std))
    assert local['raw_mpjpe_m'] == 3 / 44 and local['root_relative_body_mpjpe_m'] == 3 / 42
    assert local['root_position_error_m'] == local['pair_root_vector_error_m'] == 0
    pose[0, 0, 4] = float('nan')
    try:
        decompose(p, p, pose, p, lengths, mean, std)
    except ValueError:
        pass
    else:
        raise AssertionError('nonfinite valid frame accepted')
