#!/usr/bin/env python3
"""Prove that the VolumetricSMPL and contact meshes share Inter-X geometry."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import smplx
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.calc_pene_from_npz import npz_to_amass_dict
from optimization.human_body_prior.body_model.body_model import BodyModel
from optimization.losses import vertex_normals
from optimization.volumetric_smpl_backend import VolumetricSMPLBackend


PARAMETER_KEYS = (
    "root_orient",
    "pose_body",
    "pose_lhand",
    "pose_rhand",
    "trans",
    "betas",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact(path: Path) -> dict:
    path = path.resolve()
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def atomic_json(path: Path, payload: dict) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def normalize_gender(value) -> str:
    if isinstance(value, np.ndarray):
        value = value.item() if value.ndim == 0 else value.reshape(-1)[0]
    if isinstance(value, (bytes, np.bytes_)):
        value = value.decode("utf-8")
    value = str(value)
    if value not in ("male", "female"):
        raise ValueError(f"unsupported gender {value!r}")
    return value


def load_frame(path: Path, frame_index: int, device: torch.device) -> tuple[dict, str]:
    with np.load(path, allow_pickle=False) as archive:
        payload = npz_to_amass_dict(archive)
    gender = normalize_gender(payload["gender"])
    frame_count = int(payload["trans"].shape[0])
    selected = frame_index if frame_index >= 0 else frame_count + frame_index
    if selected < 0 or selected >= frame_count:
        raise IndexError(f"{path}: frame {frame_index} outside {frame_count} frames")
    result = {}
    for key in PARAMETER_KEYS:
        value = np.asarray(payload[key], dtype=np.float32)
        if key != "betas":
            value = value[selected:selected + 1]
        else:
            value = value.reshape(1, 10)
        result[key] = torch.as_tensor(value, device=device).clone()
    result["gender"] = torch.tensor(
        0 if gender == "male" else 1,
        dtype=torch.long,
        device=device,
    )
    return result, gender


def trainable_clone(params: dict) -> dict:
    result = {}
    for key, value in params.items():
        result[key] = value.detach().clone()
        if key in PARAMETER_KEYS:
            result[key].requires_grad_(True)
    return result


def body_model_geometry(model: BodyModel, params: dict) -> tuple[torch.Tensor, torch.Tensor]:
    hand_pose = torch.cat(
        [params["pose_lhand"], params["pose_rhand"]], dim=1
    ).reshape(1, 90)
    output = model(
        pose_body=params["pose_body"].reshape(1, 63),
        pose_hand=hand_pose,
        root_orient=params["root_orient"].reshape(1, 3),
        betas=params["betas"].reshape(1, 10),
        trans=params["trans"].reshape(1, 3),
    )
    return output.v, output.Jtr[:, :55]


def projection(vertices: torch.Tensor, joints: torch.Tensor) -> torch.Tensor:
    vertex_weights = torch.linspace(
        -0.5, 0.5, vertices.numel(), device=vertices.device,
        dtype=vertices.dtype,
    ).reshape_as(vertices)
    joint_weights = torch.linspace(
        0.25, -0.25, joints.numel(), device=joints.device,
        dtype=joints.dtype,
    ).reshape_as(joints)
    return (
        (vertices * vertex_weights).sum() / vertices.numel()
        + (joints * joint_weights).sum() / joints.numel()
    )


def gradient_record(reference: torch.Tensor, candidate: torch.Tensor) -> dict:
    difference = (reference - candidate).abs()
    denominator = max(float(reference.abs().max()), 1e-12)
    return {
        "reference_max_abs": float(reference.abs().max()),
        "candidate_max_abs": float(candidate.abs().max()),
        "maximum_abs_difference": float(difference.max()),
        "maximum_relative_to_reference": float(difference.max()) / denominator,
        "all_finite": bool(
            torch.isfinite(reference).all() and torch.isfinite(candidate).all()
        ),
    }


def default_control_geometry(
    model_root: Path,
    gender: str,
    params: dict,
) -> tuple[torch.Tensor, torch.Tensor]:
    model = smplx.create(
        str(model_root),
        model_type="smplx",
        gender=gender,
        use_pca=False,
        flat_hand_mean=False,
        num_betas=10,
        batch_size=1,
    ).to(params["trans"].device).eval()
    zeros3 = torch.zeros_like(params["trans"])
    expression = torch.zeros(
        1,
        model.num_expression_coeffs,
        device=zeros3.device,
        dtype=zeros3.dtype,
    )
    with torch.no_grad():
        output = model(
            body_pose=params["pose_body"].reshape(1, 63),
            left_hand_pose=params["pose_lhand"].reshape(1, 45),
            right_hand_pose=params["pose_rhand"].reshape(1, 45),
            global_orient=params["root_orient"].reshape(1, 3),
            transl=params["trans"].reshape(1, 3),
            betas=params["betas"].reshape(1, 10),
            jaw_pose=zeros3,
            leye_pose=zeros3,
            reye_pose=zeros3,
            expression=expression,
            return_full_pose=True,
        )
    return output.vertices, output.joints[:, :55]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--motion", type=Path, action="append", required=True)
    parser.add_argument("--frame", type=int, default=0)
    parser.add_argument("--model-root", type=Path, default=Path("body_models"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--maximum-absolute-error", type=float, default=1e-7)
    parser.add_argument("--maximum-gradient-error", type=float, default=1e-7)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    device = torch.device(args.device)
    model_root = args.model_root.resolve()
    backend = VolumetricSMPLBackend(
        model_root=str(model_root),
        device=device,
        genders=("male", "female"),
        max_query_points_per_call=4096,
    )
    body_models = {
        gender: BodyModel(
            bm_fname=str(model_root / "smplx" / f"SMPLX_{gender.upper()}.npz"),
            num_betas=10,
        ).to(device).eval()
        for gender in ("male", "female")
    }

    records = []
    for motion_path in args.motion:
        motion_path = motion_path.resolve()
        raw_params, gender = load_frame(motion_path, args.frame, device)

        reference_params = trainable_clone(raw_params)
        candidate_params = trainable_clone(raw_params)
        reference_vertices, reference_joints = body_model_geometry(
            body_models[gender], reference_params
        )
        candidate_output, candidate_gender = backend.body(candidate_params)
        candidate_vertices = candidate_output.vertices
        candidate_joints = candidate_output.joints[:, :55]
        if candidate_gender != gender:
            raise RuntimeError("backend gender normalization changed the sample")

        reference_projection = projection(reference_vertices, reference_joints)
        candidate_projection = projection(candidate_vertices, candidate_joints)
        reference_projection.backward()
        candidate_projection.backward()
        gradient_comparisons = {
            key: gradient_record(
                reference_params[key].grad,
                candidate_params[key].grad,
            )
            for key in PARAMETER_KEYS
        }

        reference_normals = vertex_normals(
            reference_vertices.detach(), body_models[gender].f
        )
        candidate_normals = vertex_normals(
            candidate_vertices.detach(), body_models[gender].f
        )
        default_vertices, default_joints = default_control_geometry(
            model_root, gender, raw_params
        )
        vertex_difference = (reference_vertices - candidate_vertices).abs()
        joint_difference = (reference_joints - candidate_joints).abs()
        normal_difference = (reference_normals - candidate_normals).abs()
        default_vertex_difference = (
            reference_vertices.detach() - default_vertices
        ).abs()
        default_joint_difference = (
            reference_joints.detach() - default_joints
        ).abs()
        maximum_gradient_error = max(
            row["maximum_abs_difference"]
            for row in gradient_comparisons.values()
        )
        passed = (
            float(vertex_difference.max()) <= args.maximum_absolute_error
            and float(joint_difference.max()) <= args.maximum_absolute_error
            and float(normal_difference.max()) <= args.maximum_absolute_error
            and maximum_gradient_error <= args.maximum_gradient_error
            and all(row["all_finite"] for row in gradient_comparisons.values())
        )
        records.append({
            "motion": artifact(motion_path),
            "gender": gender,
            "frame": args.frame,
            "flat_hand_production": {
                "vertex_max_abs_difference": float(vertex_difference.max()),
                "vertex_mean_abs_difference": float(vertex_difference.mean()),
                "joint_max_abs_difference": float(joint_difference.max()),
                "joint_mean_abs_difference": float(joint_difference.mean()),
                "normal_max_abs_difference": float(normal_difference.max()),
                "gradient_comparisons": gradient_comparisons,
                "maximum_gradient_abs_difference": maximum_gradient_error,
            },
            "smplx_default_mano_mean_control": {
                "flat_hand_mean": False,
                "vertex_max_abs_difference": float(
                    default_vertex_difference.max()
                ),
                "vertex_mean_abs_difference": float(
                    default_vertex_difference.mean()
                ),
                "joint_max_abs_difference": float(default_joint_difference.max()),
                "joint_mean_abs_difference": float(
                    default_joint_difference.mean()
                ),
            },
            "pass": passed,
        })

    genders = sorted({record["gender"] for record in records})
    checks = {
        "production_declares_flat_hand_mean": (
            backend.FLAT_HAND_MEAN is True
            and backend.representation_contract.get("smplx_flat_hand_mean") is True
            and all(bool(model.flat_hand_mean) for model in backend.models.values())
        ),
        "male_and_female_real_motions_covered": genders == ["female", "male"],
        "all_geometry_and_gradient_checks_pass": all(
            record["pass"] for record in records
        ),
        "default_mano_mean_control_demonstrates_mismatch": all(
            record["smplx_default_mano_mean_control"][
                "vertex_max_abs_difference"
            ] > args.maximum_absolute_error
            for record in records
        ),
    }
    if not all(math.isfinite(value) for record in records for value in (
        record["flat_hand_production"]["vertex_max_abs_difference"],
        record["flat_hand_production"]["joint_max_abs_difference"],
        record["flat_hand_production"]["normal_max_abs_difference"],
        record["flat_hand_production"]["maximum_gradient_abs_difference"],
    )):
        checks["all_reported_errors_finite"] = False
    else:
        checks["all_reported_errors_finite"] = True

    payload = {
        "schema": "hhc_volumetric_smpl_interx_flat_hand_geometry_audit_v1",
        "status": "pass" if all(checks.values()) else "fail",
        "checks": checks,
        "thresholds": {
            "maximum_absolute_error": args.maximum_absolute_error,
            "maximum_gradient_error": args.maximum_gradient_error,
        },
        "genders": genders,
        "representation_contract": backend.representation_contract,
        "records": records,
        "body_models": {
            gender: artifact(
                model_root / "smplx" / f"SMPLX_{gender.upper()}.npz"
            )
            for gender in ("male", "female")
        },
        "volumetric_checkpoints": backend.checkpoint_records,
        "backend_implementation": artifact(
            ROOT / "optimization/volumetric_smpl_backend.py"
        ),
        "implementation": artifact(Path(__file__)),
    }
    atomic_json(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    if payload["status"] != "pass":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
