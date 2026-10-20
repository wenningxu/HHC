#!/usr/bin/env python3

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.audit_timotion_transient_velocity_spike import (
    component_snapshot_alignment,
)


def records(*steps):
    return [{"wandb_step": step, "value": float(step)} for step in steps]


def test_running_snapshot_allows_only_aligned_trailing_skew():
    result = component_snapshot_alignment(
        {
            "loss": records(1, 2, 3),
            "simple": records(1, 2, 3, 4, 5),
            "velocity": records(1, 2, 3, 4),
        },
        remote_state="running",
        maximum_running_suffix_skew=2,
    )
    assert result["pass"] is True
    assert result["common_prefix_events"] == 3
    assert result["suffix_skew_events"] == 2
    assert result["exact_counts"] is False


def test_running_snapshot_rejects_missing_middle_event():
    result = component_snapshot_alignment(
        {
            "loss": records(1, 2, 3, 4),
            "simple": records(1, 3, 4, 5),
        },
        remote_state="running",
        maximum_running_suffix_skew=2,
    )
    assert result["pass"] is False
    assert result["common_prefix_aligned"] is False


def test_finished_snapshot_requires_exact_counts():
    result = component_snapshot_alignment(
        {
            "loss": records(1, 2, 3),
            "simple": records(1, 2, 3, 4),
        },
        remote_state="finished",
        maximum_running_suffix_skew=2,
    )
    assert result["pass"] is False
    assert result["common_prefix_aligned"] is True
    assert result["suffix_skew_allowed"] is False


if __name__ == "__main__":
    test_running_snapshot_allows_only_aligned_trailing_skew()
    test_running_snapshot_rejects_missing_middle_event()
    test_finished_snapshot_requires_exact_counts()
