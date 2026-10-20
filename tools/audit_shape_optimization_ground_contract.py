#!/usr/bin/env python3
"""Audit whether raw Inter-X SMPL-X parameters satisfy optimizer y=0 ground."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import h5py
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eval.calc_pene_from_npz import npz_to_amass_dict, to_torch_params
from optimization.human_body_prior.body_model.body_model import BodyModel
from tools.score_volumetric_smpl_npz import discover_npz_pairs


def body_geometry(models, path, device):
    with np.load(path, allow_pickle=True) as source:
        payload = {key: source[key] for key in source.files}
    params = to_torch_params(npz_to_amass_dict(payload), device)
    frames = int(params["trans"].shape[0])
    gender = str(params["gender"])
    output = models[gender](
        pose_body=params["pose_body"].reshape(frames, 63),
        pose_hand=torch.cat((
            params["pose_lhand"], params["pose_rhand"]
        ), dim=1).reshape(frames, 90),
        root_orient=params["root_orient"].reshape(frames, 3),
        betas=params["betas"].reshape(1, 10).expand(frames, -1),
        trans=params["trans"].reshape(frames, 3),
    )
    return output.v, output.Jtr, params["trans"]


def summary(values):
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "minimum": float(array.min()),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "maximum": float(array.max()),
    }


def atomic_json(path, payload):
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--random-dir", required=True, type=Path)
    parser.add_argument("--source-h5", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    device = torch.device(args.device)
    models = {
        gender: BodyModel(
            bm_fname=str(
                ROOT / f"body_models/smplx/SMPLX_{gender.upper()}.npz"
            ),
            num_betas=10,
        ).to(device).eval()
        for gender in ("male", "female")
    }
    rows = []
    joint_floors = []
    raw_ground_losses = []
    aligned_ground_losses = []
    root_y = []
    source_root_y = []
    with h5py.File(args.source_h5.resolve(), "r") as source:
        with torch.no_grad():
            for key, _sample_name, _repeat, p1, p2, _naming in (
                discover_npz_pairs(args.random_dir.resolve())
            ):
                sidecar = json.loads(
                    (args.random_dir.resolve() / f"{key}.json").read_text(
                        encoding="utf-8"
                    )
                )
                source_name = sidecar["source_name"]
                person_rows = []
                for person, path in enumerate((p1, p2)):
                    vertices, joints, trans = body_geometry(
                        models, path, device
                    )
                    floor = float(joints[..., 1].min())
                    raw_ground = float(torch.relu(-vertices[..., 1]).mean())
                    aligned_ground = float(torch.relu(
                        -(vertices[..., 1] - floor)
                    ).mean())
                    root_stats = {
                        "minimum": float(trans[:, 1].min()),
                        "mean": float(trans[:, 1].mean()),
                        "maximum": float(trans[:, 1].max()),
                    }
                    joint_floors.append(floor)
                    raw_ground_losses.append(raw_ground)
                    aligned_ground_losses.append(aligned_ground)
                    root_y.append(root_stats["mean"])
                    raw = source[source_name][:]
                    channel = 1 if person == 0 else 4
                    source_root_y.append(float(raw[:, 55, channel].mean()))
                    person_rows.append({
                        "person": person + 1,
                        "joint_floor_y_m": floor,
                        "raw_ground_loss": raw_ground,
                        "ground_loss_after_mmm_joint_floor_alignment": (
                            aligned_ground
                        ),
                        "random_shape_root_translation_y_m": root_stats,
                    })
                rows.append({
                    "sample_key": key,
                    "source_name": source_name,
                    "people": person_rows,
                })
    mmm_optimize = Path("/home/wenning/projects/MMM/optimize_full_body.py")
    mmm_source = mmm_optimize.read_text(encoding="utf-8")
    exact_alignment = (
        "root_transl[:, 1] = root_transl[:, 1] - np.min(joints[:, :, 1])"
    )
    checks = {
        "raw_random_shape_inputs_are_not_ground_aligned": (
            float(np.mean(raw_ground_losses)) > 0.05
        ),
        "mmm_entrypoint_aligns_each_person_before_optimization": (
            mmm_source.count(exact_alignment) >= 2
        ),
        "joint_floor_alignment_materially_reduces_ground_loss": (
            float(np.mean(aligned_ground_losses))
            < 0.1 * float(np.mean(raw_ground_losses))
        ),
        "source_h5_uses_pelvis_centered_root_y_near_zero": (
            abs(float(np.median(source_root_y))) < 0.25
        ),
    }
    payload = {
        "schema": "hhc_shape_optimization_ground_contract_audit_v1",
        "status": "fail_raw_direct_optimization" if all(checks.values()) else "inconclusive",
        "checks": checks,
        "sample_count": len(rows),
        "aggregate": {
            "joint_floor_y_m": summary(joint_floors),
            "raw_ground_loss": summary(raw_ground_losses),
            "ground_loss_after_mmm_joint_floor_alignment": summary(
                aligned_ground_losses
            ),
            "random_shape_root_translation_y_mean_m": summary(root_y),
            "source_h5_root_translation_y_mean_m": summary(source_root_y),
        },
        "interpretation": (
            "The optimizer's ground plane is y=0, but raw Inter-X stores a "
            "pelvis-centred translation. MMM explicitly floor-aligns before "
            "calling optimize_full_body. Direct raw optimization is invalid."
        ),
        "production_constraint": (
            "Ground alignment must be explicit in the optimization contract. "
            "Because augmented motions are inserted beside raw Inter-X and "
            "normalized with its original mean/std, any temporary alignment "
            "must be restored before materialization."
        ),
        "records": rows,
    }
    atomic_json(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    if payload["status"] != "fail_raw_direct_optimization":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
