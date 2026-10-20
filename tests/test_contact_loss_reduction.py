from pathlib import Path
import inspect
import sys

import torch
import torch.nn as nn


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from optimization.contact_loss_reduction import reduce_contact_pair_sums
from optimization.losses import DyadicInteractionLoss
from optimize.optimize_full_body import optimize_full_body


def test_raw_sum_value_and_gradient_are_literal_pair_sums():
    value = torch.tensor(2.0, requires_grad=True)
    result = reduce_contact_pair_sums(
        [3.0 * value, 4.0 * value], 2.0, "paper_literal_raw_sum"
    )
    assert float(result) == 14.0
    result.backward()
    assert float(value.grad) == 7.0


def test_active_contact_mean_normalizes_frames_and_pairs():
    value = torch.tensor(2.0, requires_grad=True)
    result = reduce_contact_pair_sums(
        [3.0 * value, 4.0 * value], 2.0, "active_contact_mean"
    )
    assert float(result) == 3.5
    result.backward()
    assert float(value.grad) == 1.75


def test_active_anchor_mean_also_normalizes_locked_vertex_pairs():
    value = torch.tensor(2.0, requires_grad=True)
    result = reduce_contact_pair_sums(
        [3.0 * value, 4.0 * value], 2.0, "active_contact_mean",
        elements_per_pair=[2, 4],
    )
    assert float(result) == 1.25
    result.backward()
    assert float(value.grad) == 0.625


def test_invalid_policy_and_counts_fail_closed():
    value = torch.tensor(1.0)
    for arguments in (
        ([value], 1.0, "unknown", None),
        ([], 1.0, "active_contact_mean", None),
        ([value], 0.0, "active_contact_mean", None),
        ([value], 1.0, "active_contact_mean", [0]),
        ([value], 1.0, "active_contact_mean", [1, 1]),
    ):
        try:
            reduce_contact_pair_sums(*arguments)
        except ValueError:
            pass
        else:
            raise AssertionError(f"invalid reduction was accepted: {arguments}")


def test_optimizer_entry_points_default_to_paper_literal_raw_sum():
    loss_default = inspect.signature(DyadicInteractionLoss.__init__).parameters[
        "contact_loss_reduction_policy"
    ].default
    optimizer_default = inspect.signature(optimize_full_body).parameters[
        "contact_loss_reduction_policy"
    ].default
    assert loss_default == optimizer_default == "paper_literal_raw_sum"


def test_shape_augmentation_binds_reduction_to_contract_and_optimizer():
    source = (ROOT / "tools/run_interx_shape_augmentation.py").read_text(
        encoding="utf-8"
    )
    assert "choices=CONTACT_LOSS_REDUCTION_POLICIES" in source
    assert "'contact_loss_reduction_policy': (" in source
    assert "contact_loss_reduction_policy=(" in source
    assert "ROOT / 'optimization/contact_loss_reduction.py'" in source


def _anchored_loss(policy):
    loss = DyadicInteractionLoss.__new__(DyadicInteractionLoss)
    nn.Module.__init__(loss)
    loss.device = "cpu"
    loss.is_anchored = True
    loss.contact_loss_reduction_policy = policy
    loss.fixed_pairs = {
        0: {
            "idx_A": torch.tensor([0]),
            "idx_B": torch.tensor([0]),
        },
        1: {
            "idx_A": torch.tensor([1, 2]),
            "idx_B": torch.tensor([1, 2]),
        },
    }
    return loss


def test_fine_anchor_path_applies_active_frame_vertex_and_pair_mean():
    loss = _anchored_loss("active_contact_mean")
    verts_a = torch.zeros(2, 3, 3)
    verts_b = torch.ones(2, 3, 3)
    normals = torch.zeros_like(verts_a)
    attraction, _ = loss._calc_fine_region_contact_loss(
        verts_a, verts_b, normals, normals,
        [{"p1_rid": 0, "p2_rid": 0}, {"p1_rid": 1, "p2_rid": 1}],
        torch.ones(2),
    )
    # Every locked vertex has squared distance 3.  The result is first a
    # frame/vertex mean per semantic pair, then a mean over the two pairs.
    assert torch.isclose(attraction, torch.tensor(3.0))


def test_semantic_anchor_raw_sum_remains_paper_literal():
    loss = _anchored_loss("paper_literal_raw_sum")
    verts_a = torch.zeros(2, 3, 3)
    verts_b = torch.ones(2, 3, 3)
    normals = torch.zeros_like(verts_a)
    attraction, alignment = loss._calc_semantic_contact_loss(
        verts_a, verts_b, normals, normals,
        [{"align_type": "none"}, {"align_type": "none"}],
        torch.ones(2),
    )
    assert torch.isclose(attraction, torch.tensor(18.0))
    assert float(alignment) == 0.0


if __name__ == "__main__":
    test_raw_sum_value_and_gradient_are_literal_pair_sums()
    test_active_contact_mean_normalizes_frames_and_pairs()
    test_active_anchor_mean_also_normalizes_locked_vertex_pairs()
    test_invalid_policy_and_counts_fail_closed()
    test_optimizer_entry_points_default_to_paper_literal_raw_sum()
    test_shape_augmentation_binds_reduction_to_contract_and_optimizer()
    test_fine_anchor_path_applies_active_frame_vertex_and_pair_mean()
    test_semantic_anchor_raw_sum_remains_paper_literal()
