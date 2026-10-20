import math

import torch

from tools.score_interhuman_vq_geometry import error_sums, finalize, physical_sums, physical_metrics


def test_coordinate_errors_mask_scale_groups_and_pooling():
    target = torch.zeros(2, 3, 262)
    prediction = target.clone()
    prediction[..., 0] = 1
    prediction[..., 3] = 2
    prediction[0, 1:] = float('nan')  # padding must not count
    mean, std = torch.ones(262) * 10, torch.ones(262) * 2
    lengths = torch.tensor([1, 3])
    sums = error_sums(target, prediction, lengths, mean, std)
    result = finalize(sums)
    assert result['valid_person_frames'] == 4
    assert math.isclose(result['joint_coordinate_rmse_m'], math.sqrt(20 / 66))
    assert math.isclose(result['normalized_feature_rmse_0_258'], math.sqrt(5 / 258))
    assert math.isclose(result['full_mpjpe_m'], 6 / 22)
    assert math.isclose(result['body_nonroot_mpjpe_m'], 4 / 21)
    assert result['root_position_error_m'] == 2
    split = [error_sums(target[i:i+1], prediction[i:i+1], lengths[i:i+1], mean, std) for i in range(2)]
    assert sums == {k: sum(row[k] for row in split) for k in sums}
    zero = finalize(error_sums(target, target, lengths, mean, std))
    assert all(v == 0 for k, v in zero.items() if k != 'valid_person_frames')
    for bad_lengths in (torch.tensor([0, 3]), torch.tensor([1., 3.]), torch.tensor([1, 4])):
        try:
            error_sums(target, target, bad_lengths, mean, std)
        except ValueError:
            pass
        else:
            raise AssertionError('invalid lengths accepted')
    prediction[0, 0, 0] = float('nan')
    try:
        error_sums(target, prediction, lengths, mean, std)
    except ValueError:
        pass
    else:
        raise AssertionError('nonfinite valid frame accepted')


def test_physical_proxy_contact_counts_and_gt_foot_denominator():
    p1 = torch.zeros(1, 3, 262)
    p2 = p1.clone()
    q1, q2 = p1.clone(), p2.clone()
    q2[0, 1, :66] = torch.tensor([1., 1., 1.]).repeat(22)
    q2[0, 2] = float('nan')  # ignored padding
    mean, std = torch.zeros(262), torch.ones(262)
    sums = physical_sums(p1, p2, q1, q2, torch.tensor([2]), mean, std, .1)
    metrics = physical_metrics(sums)
    assert sums['pair_frames'] == 2
    assert (sums['contact_tp'], sums['contact_fp'], sums['contact_fn']) == (1, 0, 1)
    assert sums['gt_contact_foot_transitions'] == 8
    assert math.isclose(metrics['geometry_contact_f1'], 2/3)
    assert math.isclose(metrics['gt_contact_foot_sliding_mm_per_frame'], 1000 * math.sqrt(2) / 2)
    assert metrics['gt_reference_foot_sliding_mm_per_frame'] == 0
    # One frame has no transitions; never turn an empty denominator into zero.
    one = physical_metrics(physical_sums(p1, p2, p1, p2, torch.tensor([1]), mean, std, .1))
    assert one['gt_contact_foot_sliding_mm_per_frame'] is None
    assert one['geometry_contact_f1'] == 1
