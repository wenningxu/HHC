import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.diagnose_shape_foot_contact_drift import foot_contact_drift


def test_fixed_reference_mask_exposes_dynamic_mask_escape():
    before = torch.zeros((3, 12, 3), dtype=torch.float32)
    after = before.clone()
    # All four feet are initially stationary and below their thresholds.  The
    # optimized result translates them horizontally enough to fail the
    # released dynamic speed predicate, while remaining grounded.
    after[1:, (7, 10, 8, 11), 0] = 0.04
    after[2:, (7, 10, 8, 11), 0] = 0.08
    result = foot_contact_drift(before, after)
    assert result['reference_contact_observations'] == 8
    assert result['reference_contacts_still_below_height_threshold'] == 8
    assert result['reference_contacts_still_below_speed_threshold'] == 0
    assert result['reference_contacts_retained_by_dynamic_mask'] == 0
    assert result['after_reference_horizontal_speed_sum_m_per_frame'] > 0
    assert result['final_dynamic_contact_observations'] == 0


def test_static_result_retains_all_reference_contacts():
    before = torch.zeros((3, 12, 3), dtype=torch.float32)
    result = foot_contact_drift(before, before.clone())
    assert result['reference_contact_observations'] == 8
    assert result['reference_contacts_retained_by_dynamic_mask'] == 8
    assert result['after_reference_horizontal_speed_sum_m_per_frame'] == 0.0
