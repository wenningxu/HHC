"""Optimization-variable policies for auditable local contact refinement."""

from __future__ import annotations

import torch


RELEASED_VARIABLE_POLICY = "released_all_per_frame"
GROUNDED_CHAIN_VARIABLE_POLICY = "grounded_chain_constant_root"
GROUNDED_CHAIN_FIXED_ROOT_POLICY = "grounded_chain_fixed_root"
OPTIMIZATION_VARIABLE_POLICIES = (
    RELEASED_VARIABLE_POLICY,
    GROUNDED_CHAIN_VARIABLE_POLICY,
    GROUNDED_CHAIN_FIXED_ROOT_POLICY,
)

# ``pose_body`` contains SMPL-X joints 1..21. Preserve both complete
# pelvis-to-foot chains: hips, knees, ankles and feet. Root orientation is a
# separate tensor and is preserved by the grounded-chain policy as well.
GROUNDED_CHAIN_POSE_BODY_INDICES = (0, 1, 3, 4, 6, 7, 9, 10)


def configure_optimization_variables(vars_dict, policy):
    """Apply the policy before optimizer parameter collection."""
    if policy not in OPTIMIZATION_VARIABLE_POLICIES:
        raise ValueError(f"unknown optimization variable policy: {policy}")
    if policy == RELEASED_VARIABLE_POLICY:
        return []

    vars_dict["root_orient"].requires_grad_(False)
    if policy == GROUNDED_CHAIN_FIXED_ROOT_POLICY:
        vars_dict["trans"].requires_grad_(False)
    pose_body = vars_dict["pose_body"]
    if pose_body.ndim != 3 or pose_body.shape[1:] != (21, 3):
        raise ValueError(
            "grounded-chain policy requires pose_body shaped [T,21,3], got "
            f"{tuple(pose_body.shape)}"
        )
    gradient_mask = torch.ones_like(pose_body)
    gradient_mask[:, GROUNDED_CHAIN_POSE_BODY_INDICES, :] = 0
    return [pose_body.register_hook(lambda gradient: gradient * gradient_mask)]


@torch.no_grad()
def project_optimization_variables(vars_dict, init_dict, policy):
    """Project each update onto the policy's feasible parameter set."""
    if policy not in OPTIMIZATION_VARIABLE_POLICIES:
        raise ValueError(f"unknown optimization variable policy: {policy}")
    if policy == RELEASED_VARIABLE_POLICY:
        return
    if policy == GROUNDED_CHAIN_FIXED_ROOT_POLICY:
        return

    # One rigid translation delta per person permits relative placement while
    # preserving every frame-to-frame root velocity exactly.
    delta = vars_dict["trans"] - init_dict["trans"]
    constant_delta = delta.mean(dim=0, keepdim=True)
    vars_dict["trans"].copy_(init_dict["trans"] + constant_delta)


@torch.no_grad()
def optimization_variable_diagnostics(vars_dict, init_dict, policy):
    """Return explicit invariants for sidecars and fail-closed audits."""
    trans_delta = vars_dict["trans"] - init_dict["trans"]
    centered_trans_delta = trans_delta - trans_delta.mean(dim=0, keepdim=True)
    pose_delta = (
        vars_dict["pose_body"][:, GROUNDED_CHAIN_POSE_BODY_INDICES, :]
        - init_dict["pose_body"][:, GROUNDED_CHAIN_POSE_BODY_INDICES, :]
    )
    root_delta = vars_dict["root_orient"] - init_dict["root_orient"]
    return {
        "policy": policy,
        "preserved_smplx_joint_indices": [0, 1, 2, 4, 5, 7, 8, 10, 11],
        "root_orientation_max_abs_delta": float(root_delta.abs().max()),
        "grounded_chain_pose_max_abs_delta": float(pose_delta.abs().max()),
        "root_translation_delta_temporal_max_abs_deviation": float(
            centered_trans_delta.abs().max()
        ),
        "root_translation_constant_delta": [
            float(value) for value in trans_delta.mean(dim=0)
        ],
    }
