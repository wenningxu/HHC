#!/usr/bin/env python3
"""Compare HBP and production-SDF joint-based foot metrics on paired assets."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eval.calc_pene_from_npz import npz_to_amass_dict, to_torch_params
from optimization.human_body_prior.body_model.body_model import BodyModel
from optimization.volumetric_smpl_backend import VolumetricSMPLBackend
from tools.score_volumetric_smpl_npz import discover_npz_pairs, foot_sliding


def atomic_json(path: Path, payload: dict) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def load_params(path: Path, device: torch.device) -> dict:
    data = npz_to_amass_dict(dict(np.load(path, allow_pickle=True)))
    params = to_torch_params(data, device)
    frames = min(int(params["trans"].shape[0]), 150)
    for key in ("root_orient", "pose_body", "pose_lhand", "pose_rhand", "trans"):
        params[key] = params[key][:frames]
    return params


@torch.no_grad()
def backend_record(
    hbp_models: dict[str, BodyModel],
    volumetric: VolumetricSMPLBackend,
    path: Path,
    device: torch.device,
) -> dict:
    params = load_params(path, device)
    frames = int(params["trans"].shape[0])
    gender = volumetric.normalize_gender(params["gender"])
    betas = params["betas"].reshape(1, 10).expand(frames, -1)
    hbp = hbp_models[gender](
        pose_body=params["pose_body"].reshape(frames, 63),
        pose_hand=torch.cat(
            (params["pose_lhand"], params["pose_rhand"]), dim=1
        ).reshape(frames, 90),
        root_orient=params["root_orient"].reshape(frames, 3),
        betas=betas,
        trans=params["trans"].reshape(frames, 3),
    )
    volumetric_output, _ = volumetric.body(params)
    hbp_joints = hbp.Jtr
    volumetric_full_joints = volumetric_output.joints
    volumetric_joints = volumetric_full_joints[:, : hbp_joints.shape[1]]
    hbp_metric = foot_sliding(hbp_joints)
    volumetric_metric = foot_sliding(volumetric_joints)
    volumetric_full_metric = foot_sliding(volumetric_full_joints)
    difference = (hbp_joints - volumetric_joints).abs()
    return {
        "path": str(path.resolve()),
        "gender": gender,
        "frames": frames,
        "joint_max_abs_difference_m": float(difference.max()),
        "joint_mean_abs_difference_m": float(difference.mean()),
        "hbp": hbp_metric,
        "volumetric": volumetric_metric,
        "volumetric_full": volumetric_full_metric,
        "floor_y_m": {
            "hbp_55": float(hbp_joints[..., 1].min()),
            "volumetric_55": float(volumetric_joints[..., 1].min()),
            "volumetric_full": float(volumetric_full_joints[..., 1].min()),
        },
    }


def pair_metric(rows: list[dict], backend: str) -> float:
    observations = sum(row[backend]["contact_observations"] for row in rows)
    if not observations:
        return 0.0
    return sum(
        row[backend]["contact_observations"]
        * row[backend]["mean_horizontal_speed_m_per_frame"]
        for row in rows
    ) / observations


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", type=Path, required=True)
    parser.add_argument("--after", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    device = torch.device(args.device)
    hbp_models = {
        gender: BodyModel(
            bm_fname=f"body_models/smplx/SMPLX_{gender.upper()}.npz",
            num_betas=10,
        ).to(device).eval()
        for gender in ("male", "female")
    }
    volumetric = VolumetricSMPLBackend(
        model_root="body_models",
        device=device,
        genders=("male", "female"),
        max_query_points_per_call=10475,
    ).eval()
    before = {row[0]: row for row in discover_npz_pairs(args.before)}
    after = {row[0]: row for row in discover_npz_pairs(args.after)}
    if before.keys() != after.keys() or not before:
        raise ValueError("before/after sample keys must be identical and nonempty")

    records = []
    for key in sorted(before):
        before_rows = [
            backend_record(hbp_models, volumetric, path, device)
            for path in before[key][3:5]
        ]
        after_rows = [
            backend_record(hbp_models, volumetric, path, device)
            for path in after[key][3:5]
        ]
        hbp_before = pair_metric(before_rows, "hbp")
        hbp_after = pair_metric(after_rows, "hbp")
        volumetric_before = pair_metric(before_rows, "volumetric")
        volumetric_after = pair_metric(after_rows, "volumetric")
        volumetric_full_before = pair_metric(before_rows, "volumetric_full")
        volumetric_full_after = pair_metric(after_rows, "volumetric_full")
        records.append({
            "sample_key": key,
            "persons_before": before_rows,
            "persons_after": after_rows,
            "hbp_delta_after_minus_before": hbp_after - hbp_before,
            "volumetric_delta_after_minus_before": (
                volumetric_after - volumetric_before
            ),
            "volumetric_full_delta_after_minus_before": (
                volumetric_full_after - volumetric_full_before
            ),
        })

    payload = {
        "schema": "hhc_shape_foot_backend_mismatch_diagnostic_v1",
        "status": "complete",
        "sample_count": len(records),
        "aggregate": {
            "hbp_mean_delta_after_minus_before": float(np.mean([
                row["hbp_delta_after_minus_before"] for row in records
            ])),
            "volumetric_mean_delta_after_minus_before": float(np.mean([
                row["volumetric_delta_after_minus_before"] for row in records
            ])),
            "volumetric_full_mean_delta_after_minus_before": float(np.mean([
                row["volumetric_full_delta_after_minus_before"]
                for row in records
            ])),
            "maximum_joint_abs_difference_m": max(
                person["joint_max_abs_difference_m"]
                for row in records
                for side in ("persons_before", "persons_after")
                for person in row[side]
            ),
        },
        "records": records,
    }
    atomic_json(args.output, payload)
    print(json.dumps(payload["aggregate"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
