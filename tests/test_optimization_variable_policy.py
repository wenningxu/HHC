import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from optimization.optimization_variable_policy import (
    GROUNDED_CHAIN_POSE_BODY_INDICES,
    GROUNDED_CHAIN_FIXED_ROOT_POLICY,
    GROUNDED_CHAIN_VARIABLE_POLICY,
    configure_optimization_variables,
    optimization_variable_diagnostics,
    project_optimization_variables,
)


def make_vars(frames=5):
    return {
        "root_orient": torch.zeros(frames, 3, requires_grad=True),
        "trans": torch.zeros(frames, 3, requires_grad=True),
        "pose_body": torch.zeros(frames, 21, 3, requires_grad=True),
    }


def main():
    variables = make_vars()
    initial = {key: value.detach().clone() for key, value in variables.items()}
    handles = configure_optimization_variables(
        variables, GROUNDED_CHAIN_VARIABLE_POLICY
    )
    assert handles
    assert not variables["root_orient"].requires_grad

    variables["pose_body"].sum().backward()
    preserved = variables["pose_body"].grad[
        :, GROUNDED_CHAIN_POSE_BODY_INDICES, :
    ]
    assert torch.count_nonzero(preserved) == 0
    assert torch.count_nonzero(variables["pose_body"].grad) > 0

    with torch.no_grad():
        variables["trans"].copy_(torch.arange(15.0).reshape(5, 3))
    project_optimization_variables(
        variables, initial, GROUNDED_CHAIN_VARIABLE_POLICY
    )
    expected = torch.arange(15.0).reshape(5, 3).mean(dim=0)
    assert torch.equal(variables["trans"], expected.expand(5, 3))

    diagnostics = optimization_variable_diagnostics(
        variables, initial, GROUNDED_CHAIN_VARIABLE_POLICY
    )
    assert diagnostics["root_orientation_max_abs_delta"] == 0.0
    assert diagnostics["grounded_chain_pose_max_abs_delta"] == 0.0
    assert diagnostics[
        "root_translation_delta_temporal_max_abs_deviation"
    ] == 0.0

    fixed_root = make_vars()
    fixed_initial = {
        key: value.detach().clone() for key, value in fixed_root.items()
    }
    configure_optimization_variables(
        fixed_root, GROUNDED_CHAIN_FIXED_ROOT_POLICY
    )
    assert not fixed_root["root_orient"].requires_grad
    assert not fixed_root["trans"].requires_grad
    project_optimization_variables(
        fixed_root, fixed_initial, GROUNDED_CHAIN_FIXED_ROOT_POLICY
    )
    fixed_diagnostics = optimization_variable_diagnostics(
        fixed_root, fixed_initial, GROUNDED_CHAIN_FIXED_ROOT_POLICY
    )
    assert fixed_diagnostics["root_translation_constant_delta"] == [0.0] * 3
    print("optimization variable policy test: PASS")


if __name__ == "__main__":
    main()
