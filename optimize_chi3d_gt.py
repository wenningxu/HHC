import argparse
import json
import os
import re
from pathlib import Path

import h5py
import numpy as np
import torch

from optimize_full_body import create_trapezoidal_mask, optimize_full_body
from test_data import _compute_geometry
from utils.paramUtil import CANDIDATES


def parse_args():
    parser = argparse.ArgumentParser(description="Optimize Chi3D GT motions using annotated contact pairs.")
    parser.add_argument("--sample-name", nargs="+", required=True, help="Chi3D sample names, e.g. G2_Grab_1")
    parser.add_argument("--motion-file", default="dataset/chi3d/motions/chi3d-shape.h5")
    parser.add_argument("--betas-dir", default="dataset/chi3d/betas")
    parser.add_argument("--labels-dir", default="dataset/chi3d/interaction_labels")
    parser.add_argument("--out-dir", default="results/chi3d_gt_optimized")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument(
        "--time-mask-mode",
        default="label",
        choices=["label", "manual", "auto"],
        help="label: use contact label start/end with trapezoid ramps; manual: use explicit times; auto: let optimize_full_body detect.",
    )
    parser.add_argument("--ramp-frames", type=int, default=3, help="Ramp size for label-based time masks.")
    parser.add_argument("--t-start", type=int, default=None)
    parser.add_argument("--t-peak-start", type=int, default=None)
    parser.add_argument("--t-peak-end", type=int, default=None)
    parser.add_argument("--t-end", type=int, default=None)
    parser.add_argument("--fps", type=int, default=30)
    return parser.parse_args()


def parse_sample_name(sample_name):
    match = re.fullmatch(r"G(\d+)_([^_]+)_(\d+)", sample_name)
    if match is None:
        raise ValueError(f"Invalid Chi3D sample name: {sample_name}")
    group_id = int(match.group(1))
    action = match.group(2)
    take_idx = int(match.group(3))
    return group_id, action, take_idx


def get_label_entry(sample_name, labels_dir):
    group_id, action, take_idx = parse_sample_name(sample_name)
    label_path = Path(labels_dir) / f"s{group_id:02d}_interaction_contact_signature.json"
    if not label_path.exists():
        raise FileNotFoundError(f"Missing label file for group {group_id}: {label_path}")

    with open(label_path, "r", encoding="utf-8") as f:
        labels = json.load(f)

    key = f"{action} {take_idx}"
    if key not in labels:
        raise KeyError(f"Missing label entry '{key}' in {label_path}")
    return labels[key]


def load_chi3d_gt_motion(sample_name, motion_file):
    with h5py.File(motion_file, "r") as f:
        if sample_name not in f:
            raise KeyError(f"Missing motion '{sample_name}' in {motion_file}")
        return f[sample_name][:].astype(np.float32)


def load_chi3d_betas(group_id, betas_dir):
    beta_path = Path(betas_dir) / f"{group_id}.npy"
    betas = np.load(beta_path).astype(np.float32)
    if betas.shape != (2, 10):
        raise ValueError(f"Expected betas shape (2, 10), got {betas.shape} from {beta_path}")
    return betas


def chi3d_motion_to_amass_pair(motion, betas):
    def build_person(person_idx):
        offset = person_idx * 3
        return {
            "root_orient": motion[:, 0, offset:offset + 3].copy(),
            "pose_body": motion[:, 1:22, offset:offset + 3].copy(),
            "pose_lhand": motion[:, 25:40, offset:offset + 3].copy(),
            "pose_rhand": motion[:, 40:55, offset:offset + 3].copy(),
            "trans": motion[:, -1, offset:offset + 3].copy(),
            "betas": betas[person_idx].copy(),
            "gender": "neutral",
        }

    return build_person(0), build_person(1)


def align_ground(amass_dict):
    _, joints = _compute_geometry(amass_dict)
    root_trans = amass_dict["trans"].copy()
    root_trans[:, 1] = root_trans[:, 1] - np.min(joints[:, :, 1])
    amass_dict["trans"] = root_trans
    return amass_dict


def build_rid_to_semantic_parts():
    rid_to_parts = {}
    for part_name, region_ids in CANDIDATES.items():
        for rid in region_ids:
            rid_to_parts.setdefault(int(rid), []).append(part_name)
    return rid_to_parts


def select_most_specific_parts(parts):
    if not parts:
        return []
    min_size = min(len(CANDIDATES[part]) for part in parts)
    return sorted([part for part in parts if len(CANDIDATES[part]) == min_size])


def build_interaction_pairs_from_labels(label_entry, rid_to_parts):
    region_pairs = label_entry["smplx_signature"]["region_id"]
    interaction_pairs = []
    seen = set()
    for pair in region_pairs:
        if len(pair) != 2:
            continue
        rid_a = int(pair[0])
        rid_b = int(pair[1])
        parts_a = select_most_specific_parts(rid_to_parts.get(rid_a, []))
        parts_b = select_most_specific_parts(rid_to_parts.get(rid_b, []))
        for part_a in parts_a:
            for part_b in parts_b:
                key = (part_a, part_b)
                if key in seen:
                    continue
                seen.add(key)
                interaction_pairs.append(
                    {
                        "p1_part": part_a,
                        "p1_ids": list(CANDIDATES[part_a]),
                        "p2_part": part_b,
                        "p2_ids": list(CANDIDATES[part_b]),
                        "contact_type": "flat",
                        "align_type": "strict",
                        "w_p1": 0.5,
                        "w_p2": 0.5,
                    }
                )
    return interaction_pairs


def build_temporal_mask(label_entry, total_frames, mode, device, ramp_frames, manual_times):
    if mode == "auto":
        return None

    if mode == "manual":
        if any(v is None for v in manual_times):
            raise ValueError("Manual mode requires --t-start --t-peak-start --t-peak-end --t-end")
        t_start, t_peak_start, t_peak_end, t_end = manual_times
    else:
        start_fr = int(label_entry["start_fr"])
        end_fr = int(label_entry["end_fr"])
        ramp = max(0, int(ramp_frames))
        t_start = max(0, start_fr - ramp)
        t_peak_start = start_fr
        t_peak_end = end_fr
        t_end = min(total_frames - 1, end_fr + ramp)

    return create_trapezoidal_mask(total_frames, t_start, t_peak_start, t_peak_end, t_end, device=device)


def amass_pair_to_npz_dict(person_dict):
    poses = np.zeros((len(person_dict["trans"]), 55, 3), dtype=np.float32)
    poses[:, 0, :] = person_dict["root_orient"]
    poses[:, 1:22, :] = person_dict["pose_body"]
    poses[:, 25:40, :] = person_dict["pose_lhand"]
    poses[:, 40:55, :] = person_dict["pose_rhand"]
    gender_value = person_dict["gender"]
    if isinstance(gender_value, np.ndarray):
        gender_value = gender_value.item()
    if isinstance(gender_value, (np.integer, int)):
        gender_value = {0: "male", 1: "female", 2: "neutral"}.get(int(gender_value), "neutral")
    else:
        gender_value = str(gender_value)
    return {
        "poses": poses,
        "trans": person_dict["trans"],
        "betas": person_dict["betas"],
        "gender": gender_value,
    }


def save_pair(sample_name, refined_a, refined_b, out_dir, fps):
    sample_dir = Path(out_dir)
    sample_dir.mkdir(parents=True, exist_ok=True)
    for suffix, data in (("p1", refined_a), ("p2", refined_b)):
        npz_dict = amass_pair_to_npz_dict(data)
        np.savez(
            sample_dir / f"{sample_name}_optimized_{suffix}.npz",
            poses=npz_dict["poses"],
            trans=npz_dict["trans"],
            betas=npz_dict["betas"],
            gender=npz_dict["gender"],
            mocap_framerate=fps,
        )


def main():
    args = parse_args()
    device = args.device if torch.cuda.is_available() else "cpu"
    manual_times = (args.t_start, args.t_peak_start, args.t_peak_end, args.t_end)
    rid_to_parts = build_rid_to_semantic_parts()

    for sample_name in args.sample_name:
        group_id, _, _ = parse_sample_name(sample_name)
        motion = load_chi3d_gt_motion(sample_name, args.motion_file)
        betas = load_chi3d_betas(group_id, args.betas_dir)
        label_entry = get_label_entry(sample_name, args.labels_dir)

        data_a, data_b = chi3d_motion_to_amass_pair(motion, betas)
        data_a = align_ground(data_a)
        data_b = align_ground(data_b)

        interaction_pairs = build_interaction_pairs_from_labels(label_entry, rid_to_parts)
        temporal_mask = build_temporal_mask(
            label_entry,
            total_frames=len(data_a["trans"]),
            mode=args.time_mask_mode,
            device=device,
            ramp_frames=args.ramp_frames,
            manual_times=manual_times,
        )

        refined_a, refined_b = optimize_full_body(
            initial_data_A=data_a,
            initial_data_B=data_b,
            contacts_A=None,
            contacts_B=None,
            temporal_mask=temporal_mask,
            interaction_pairs=interaction_pairs,
            fine_region_pairs=None,
            num_epochs=args.num_epochs,
            lr=args.lr,
            device=device,
        )

        save_pair(sample_name, refined_a, refined_b, args.out_dir, args.fps)
        print(f"Saved optimized Chi3D GT for {sample_name} to {args.out_dir}")


if __name__ == "__main__":
    main()
