#!/usr/bin/env python
"""Export the public TIMotion Transformer+LPA on frozen Inter-X requests."""

import argparse
import hashlib
import os
import random
import sys

import h5py
import numpy as np
import torch


TIMOTION_ROOT = "/mnt/8T-HDD/wenning/TIMotion"
MODEL_CONFIG_PATH = os.path.join(TIMOTION_ROOT, "configs/model.yaml")
MODEL_SOURCE_PATH = os.path.join(TIMOTION_ROOT, "models/timotion.py")
DIFFUSION_SOURCE_PATH = os.path.join(TIMOTION_ROOT, "models/nets.py")
INTERACTION_SOURCE_PATH = os.path.join(TIMOTION_ROOT, "models/blocks.py")
GAUSSIAN_DIFFUSION_SOURCE_PATH = os.path.join(
    TIMOTION_ROOT, "models/gaussian_diffusion.py"
)
sys.path.insert(0, TIMOTION_ROOT)
os.environ.setdefault("TIMOTION_STATS_PREFIX", "interx_global")

from configs import get_config  # noqa: E402
from models import TIMotion  # noqa: E402
from utils.rotation_conversions import (  # noqa: E402
    matrix_to_rotation_6d, rotation_6d_to_matrix,
)


BODY_PARENTS = (
    -1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14,
    16, 17, 18, 19,
)


def sha256_file(path, chunk_size=8 * 1024 * 1024):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_resume_attributes(source, expected):
    mismatches = {
        key: (source.attrs.get(key), value)
        for key, value in expected.items()
        if key not in source.attrs or source.attrs.get(key) != value
    }
    if mismatches:
        raise ValueError(f"resume provenance mismatch: {mismatches}")


def build_model(checkpoint_path):
    cfg = get_config(MODEL_CONFIG_PATH)
    # The released training/evaluation CLI overrides these YAML values.
    cfg.NUM_HEADS = 16
    cfg.NUM_LAYERS = 5
    cfg.LATENT_DIM = 512
    cfg.DROPOUT = 0.1
    cfg.LPA = True
    cfg.conv_layers = 1
    cfg.dilation_rate = 1
    cfg.norm = "AdaLN"
    model = TIMotion(cfg)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state = checkpoint.get("state_dict", checkpoint.get("model", checkpoint))
    cleaned = {}
    for key, value in state.items():
        if key.startswith("model."):
            key = key[len("model."):]
        cleaned[key] = value
    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"checkpoint mismatch: missing={missing}, unexpected={unexpected}"
        )
    return model, cfg


def _root_rotation_from_positions(positions):
    r_hip, l_hip, r_shoulder, l_shoulder = (2, 1, 17, 16)
    across = (
        positions[..., r_hip, :] - positions[..., l_hip, :]
        + positions[..., r_shoulder, :] - positions[..., l_shoulder, :]
    )
    valid = across.norm(dim=-1, keepdim=True) > 1e-8
    fallback = torch.zeros_like(across)
    fallback[..., 0] = -1.0
    across = torch.where(valid, across, fallback)
    across = torch.nn.functional.normalize(across, dim=-1)
    up = torch.zeros_like(across)
    up[..., 1] = 1.0
    forward = torch.nn.functional.normalize(torch.cross(up, across, dim=-1), dim=-1)
    right = torch.nn.functional.normalize(torch.cross(up, forward, dim=-1), dim=-1)
    return torch.stack([right, up, forward], dim=-1)


def _fit_shape_specific_root(positions, body_rotations, skeleton):
    skeleton = skeleton[..., :22, :].to(
        device=positions.device, dtype=positions.dtype
    )
    while skeleton.ndim < positions.ndim:
        skeleton = skeleton.unsqueeze(-3)
    skeleton = skeleton.expand(positions.shape)
    parents = torch.as_tensor(
        BODY_PARENTS, dtype=torch.long, device=positions.device
    )
    offsets = skeleton.clone()
    offsets[..., 1:, :] -= skeleton[..., parents[1:], :]
    local = rotation_6d_to_matrix(body_rotations)
    identity = torch.eye(
        3, device=positions.device, dtype=positions.dtype
    ).expand(positions.shape[:-2] + (3, 3))
    posed = torch.zeros_like(positions)
    posed[..., 0, :] = offsets[..., 0, :]
    global_rotations = [identity]
    for joint in range(1, 22):
        parent = BODY_PARENTS[joint]
        parent_rotation = global_rotations[parent]
        posed[..., joint, :] = (
            posed[..., parent, :]
            + torch.matmul(
                parent_rotation, offsets[..., joint, :, None]
            ).squeeze(-1)
        )
        global_rotations.append(
            torch.matmul(parent_rotation, local[..., joint - 1, :, :])
        )
    source_to_model = torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]],
        dtype=positions.dtype, device=positions.device,
    )
    observed_source = torch.matmul(positions, source_to_model)
    posed_centered = posed - posed[..., :1, :]
    observed_centered = observed_source - observed_source[..., :1, :]
    covariance = torch.matmul(
        posed_centered.transpose(-1, -2), observed_centered
    )
    left, _, right_t = torch.linalg.svd(covariance)
    row_rotation = torch.matmul(left, right_t)
    reflection = torch.linalg.det(row_rotation) < 0
    correction = torch.eye(
        3, device=positions.device, dtype=positions.dtype
    ).expand(row_rotation.shape).clone()
    correction[..., -1, -1] = torch.where(
        reflection,
        torch.full_like(reflection, -1, dtype=positions.dtype),
        torch.ones_like(reflection, dtype=positions.dtype),
    )
    row_rotation = torch.matmul(
        torch.matmul(left, correction), right_t
    )
    return (
        row_rotation.transpose(-1, -2),
        observed_source[..., 0, :] - posed[..., 0, :],
    )


def timotion_to_official_interx(features, skeleton=None):
    """Map released 22-joint/262-D features to official raw 672-D input."""
    if features.shape[-1] == 524:
        features = features.reshape(features.shape[:-1] + (2, 262))
    if features.shape[-2:] != (2, 262):
        raise ValueError(f"expected [..., 2, 262], got {features.shape}")
    output = torch.zeros(
        features.shape[:-2] + (56, 12),
        dtype=features.dtype,
        device=features.device,
    )
    identity = torch.tensor(
        [1, 0, 0, 0, 1, 0], dtype=features.dtype, device=features.device
    )
    output[..., :55, :6] = identity
    output[..., :55, 6:12] = identity
    source_to_model = torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]],
        dtype=features.dtype,
        device=features.device,
    )
    model_to_source = source_to_model.transpose(0, 1)
    for person, start in enumerate((0, 6)):
        person_features = features[..., person, :]
        positions = person_features[..., :66].reshape(
            person_features.shape[:-1] + (22, 3)
        )
        rotations = person_features[..., 132:258].reshape(
            person_features.shape[:-1] + (21, 6)
        )
        output[..., 1:22, start:start + 6] = rotations
        if skeleton is None:
            root_model = _root_rotation_from_positions(positions)
            root_source = model_to_source @ root_model @ source_to_model
            root_translation = (
                positions[..., 0, :]
                @ model_to_source.transpose(0, 1)
            )
        else:
            skeleton_tensor = torch.as_tensor(
                skeleton, dtype=features.dtype, device=features.device
            )
            if (
                skeleton_tensor.shape[-3] != 2
                or skeleton_tensor.shape[-2] < 22
                or skeleton_tensor.shape[-1] != 3
            ):
                raise ValueError(
                    "skeleton must end in [2, at least 22, 3]"
                )
            root_source, root_translation = _fit_shape_specific_root(
                positions, rotations,
                skeleton_tensor[..., person, :, :],
            )
        output[..., 0, start:start + 6] = matrix_to_rotation_6d(root_source)
        output[..., 55, start:start + 3] = root_translation
    return output.reshape(output.shape[:-2] + (56 * 12,))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--requests", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--indices-key", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--skeleton-root",
        default="/mnt/8T-HDD/wenning/HHC/dataset/interx/skeleton",
    )
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(f"cuda:{args.gpu}")
    request = np.load(args.requests, allow_pickle=False)
    captions = request["captions"].astype(str)
    lengths = request["lengths"].astype(np.int64)
    names = request["names"].astype(str)
    if args.indices_key:
        indices = request[args.indices_key].astype(np.int64)
        captions, lengths, names = captions[indices], lengths[indices], names[indices]
    if args.limit is not None:
        captions, lengths, names = (
            captions[:args.limit], lengths[:args.limit], names[:args.limit]
        )
    if not (len(captions) == len(lengths) == len(names)):
        raise ValueError("request arrays have inconsistent lengths")
    used_groups = sorted({int(name[1:4]) for name in names})
    skeleton_paths = {
        group: os.path.join(args.skeleton_root, f"{group}.npy")
        for group in used_groups
    }
    skeleton_bank_paths = {
        group: os.path.join(args.skeleton_root, f"{group}.npy")
        for group in range(1, 60)
    }
    skeleton_digest = hashlib.sha256()
    for group, path in skeleton_bank_paths.items():
        skeleton_digest.update(f"{group}:".encode("ascii"))
        skeleton_digest.update(sha256_file(path).encode("ascii"))

    mean_path = os.path.join(TIMOTION_ROOT, "data/interx_global_mean.npy")
    std_path = os.path.join(TIMOTION_ROOT, "data/interx_global_std.npy")
    runtime_cfg = get_config(MODEL_CONFIG_PATH)
    if str(runtime_cfg.STRATEGY) != "ddim50":
        raise ValueError(
            f"fair TiMotion export requires ddim50, got {runtime_cfg.STRATEGY!r}"
        )
    if int(runtime_cfg.DIFFUSION_STEPS) != 1000:
        raise ValueError(
            "fair TiMotion export requires the author 1000-step training "
            f"diffusion, got {runtime_cfg.DIFFUSION_STEPS!r}"
        )
    if float(runtime_cfg.CFG_WEIGHT) != 3.5:
        raise ValueError(
            f"fair TiMotion export requires CFG 3.5, got {runtime_cfg.CFG_WEIGHT!r}"
        )
    stats_prefix = os.environ.get("TIMOTION_STATS_PREFIX", "global")
    if stats_prefix != "interx_global":
        raise ValueError(
            "fair Inter-X export requires TIMOTION_STATS_PREFIX=interx_global, "
            f"got {stats_prefix!r}"
        )
    expected = {
        "checkpoint": os.path.abspath(args.checkpoint),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "requests": os.path.abspath(args.requests),
        "requests_sha256": sha256_file(args.requests),
        "seed": args.seed,
        "seed_strategy": "base_seed + repeat_index",
        "method": "public_TIMotion_Transformer_LPA",
        "architecture": "layers5_heads16_latent512_LPA_AdaLN_conv3x1",
        "adapter": (
            "22-body rotations + shape-specific all-joint rigid root fit v2; "
            "identity hands"
        ),
        "skeleton_root": os.path.abspath(args.skeleton_root),
        "skeleton_files_sha256": skeleton_digest.hexdigest(),
        "generation_length_policy": "exact_request_length_grouped",
        "first_frame_conditioning": "none",
        "sampling_process": "DDIM50_eta0_CFG3.5",
        "sampling_steps": 50,
        "ddim_eta": 0.0,
        "training_diffusion_steps": int(runtime_cfg.DIFFUSION_STEPS),
        "cfg_weight": 3.5,
        "batch_size": args.batch_size,
        "model_config": os.path.abspath(MODEL_CONFIG_PATH),
        "model_config_sha256": sha256_file(MODEL_CONFIG_PATH),
        "normalization_prefix": stats_prefix,
        "normalization_mean": os.path.abspath(mean_path),
        "normalization_mean_sha256": sha256_file(mean_path),
        "normalization_std": os.path.abspath(std_path),
        "normalization_std_sha256": sha256_file(std_path),
        "exporter_sha256": sha256_file(__file__),
        "model_source_sha256": sha256_file(MODEL_SOURCE_PATH),
        "diffusion_source_sha256": sha256_file(DIFFUSION_SOURCE_PATH),
        "interaction_source_sha256": sha256_file(INTERACTION_SOURCE_PATH),
        "gaussian_diffusion_source_sha256": sha256_file(
            GAUSSIAN_DIFFUSION_SOURCE_PATH
        ),
        "adapter_source_sha256": sha256_file(__file__),
        "request_indices_key": args.indices_key or "",
        "request_limit": -1 if args.limit is None else int(args.limit),
    }
    shape = (len(captions), args.repeats, 150, 56 * 12)
    if args.resume and os.path.isfile(args.output):
        with h5py.File(args.output, "r") as existing:
            validate_resume_attributes(existing, expected)
            complete = (
                existing["motions"].shape == shape
                and np.array_equal(existing["lengths"][:], lengths)
                and np.array_equal(existing["names"][:].astype(str), names)
                and "completed_repeats" in existing
                and existing["completed_repeats"][:].astype(bool).all()
            )
        if complete:
            print({"output": args.output, "status": "all repeats already complete"})
            return

    model, _ = build_model(args.checkpoint)
    model.to(device).eval()
    mean = torch.from_numpy(np.load(mean_path)).float().to(device)
    std = torch.from_numpy(np.load(std_path)).float().to(device)
    skeleton_cache = {
        group: np.load(path).astype(np.float32)
        for group, path in skeleton_paths.items()
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    resume_existing = args.resume and os.path.isfile(args.output)
    target = h5py.File(args.output, "r+" if resume_existing else "w")
    try:
        if resume_existing:
            validate_resume_attributes(target, expected)
            if target["motions"].shape != shape:
                raise ValueError("resume shape mismatch")
            if not np.array_equal(target["lengths"][:], lengths):
                raise ValueError("resume lengths mismatch")
            if not np.array_equal(target["names"][:].astype(str), names):
                raise ValueError("resume names mismatch")
            motions = target["motions"]
            completed = target["completed_repeats"]
        else:
            motions = target.create_dataset(
                "motions",
                shape=shape,
                dtype=np.float32,
                chunks=(1, 1, 150, 56 * 12),
                compression="lzf",
            )
            target.create_dataset("lengths", data=lengths)
            target.create_dataset("names", data=np.asarray(names, dtype="S"))
            completed = target.create_dataset(
                "completed_repeats", data=np.zeros(args.repeats, dtype=np.bool_)
            )
            for key, value in expected.items():
                target.attrs[key] = value

        for repeat in range(args.repeats):
            if bool(completed[repeat]):
                print({"repeat": repeat, "status": "already complete"}, flush=True)
                continue
            repeat_seed = args.seed + repeat
            random.seed(repeat_seed)
            np.random.seed(repeat_seed)
            torch.manual_seed(repeat_seed)
            torch.cuda.manual_seed_all(repeat_seed)
            generated_count = 0
            for valid_length in sorted(np.unique(lengths).tolist()):
                selected = np.flatnonzero(lengths == valid_length)
                for start in range(0, len(selected), args.batch_size):
                    batch_indices = selected[start:start + args.batch_size]
                    batch = {
                        "text": captions[batch_indices].tolist(),
                        "motion_lens": torch.full(
                            (len(batch_indices),), int(valid_length), dtype=torch.long
                        ),
                    }
                    with torch.inference_mode():
                        generated = model.forward_test(batch)["output"].reshape(
                            len(batch_indices), int(valid_length), 2, 262
                        )
                        if not torch.isfinite(generated).all():
                            bad = int((~torch.isfinite(generated)).sum().item())
                            raise FloatingPointError(
                                "TiMotion sampler produced non-finite normalized "
                                f"features: count={bad}, step_length={valid_length}"
                            )
                        generated = generated * std + mean
                        if not torch.isfinite(generated).all():
                            bad = int((~torch.isfinite(generated)).sum().item())
                            raise FloatingPointError(
                                "TiMotion denormalization produced non-finite "
                                f"features: count={bad}, step_length={valid_length}"
                            )
                        batch_skeletons = torch.from_numpy(np.stack([
                            skeleton_cache[int(names[index][1:4])]
                            for index in batch_indices
                        ])).to(device)
                        official_valid = timotion_to_official_interx(
                            generated, batch_skeletons
                        ).cpu().numpy()
                    if not np.isfinite(official_valid).all():
                        bad = int((~np.isfinite(official_valid)).sum())
                        raise FloatingPointError(
                            "TiMotion official Inter-X adapter produced non-finite "
                            f"features: count={bad}, step_length={valid_length}"
                        )
                    official = np.zeros(
                        (len(batch_indices), 150, 56 * 12), dtype=np.float32
                    )
                    official[:, :int(valid_length)] = official_valid
                    motions[batch_indices, repeat] = official
                    generated_count += len(batch_indices)
                    print(
                        {
                            "repeat": repeat,
                            "generated": generated_count,
                            "total": len(captions),
                            "request_length": int(valid_length),
                        },
                        flush=True,
                    )
            target.flush()
            completed[repeat] = True
            target.flush()
    finally:
        target.close()


if __name__ == "__main__":
    main()
