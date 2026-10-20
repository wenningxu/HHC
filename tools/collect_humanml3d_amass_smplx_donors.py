#!/usr/bin/env python3
"""Validate official SMPL+H->SMPL-X fits and build the final donor pool."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
from collections import Counter
from pathlib import Path

import numpy as np
import torch


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact(path: Path) -> dict:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return {"path": str(path), "bytes": path.stat().st_size,
            "sha256": sha256_file(path)}


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)


def canonical_shape_id(gender: str, betas: np.ndarray) -> str:
    payload = {
        "gender": gender,
        "betas": [float(value) for value in np.asarray(betas, dtype=np.float32)],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def numpy_value(value: object) -> np.ndarray:
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def obj_vertices(path: Path) -> np.ndarray:
    vertices = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("v "):
                fields = line.split()
                if len(fields) != 4:
                    raise ValueError(f"invalid vertex row in {path}: {line!r}")
                vertices.append([float(value) for value in fields[1:]])
    result = np.asarray(vertices, dtype=np.float64)
    if result.ndim != 2 or result.shape[1:] != (3,) or not np.isfinite(result).all():
        raise ValueError(f"invalid OBJ vertices in {path}: {result.shape}")
    return result


def deformation_matrix(path: Path) -> np.ndarray:
    with path.open("rb") as handle:
        payload = pickle.load(handle, encoding="latin1")
    if "mtx" in payload:
        matrix = payload["mtx"]
        if hasattr(matrix, "todense"):
            matrix = matrix.todense()
        matrix = np.asarray(matrix, dtype=np.float64)
        matrix = matrix[:, :matrix.shape[1] // 2]
    elif "matrix" in payload:
        matrix = payload["matrix"]
        if hasattr(matrix, "todense"):
            matrix = matrix.todense()
        matrix = np.asarray(matrix, dtype=np.float64)
    else:
        raise KeyError("official correspondence contains neither mtx nor matrix")
    if matrix.ndim != 2 or not np.isfinite(matrix).all():
        raise ValueError("invalid deformation transfer matrix")
    return matrix


def shape_statistics(betas: np.ndarray) -> dict:
    betas = np.asarray(betas, dtype=np.float64)
    if betas.ndim != 2 or not len(betas) or not np.isfinite(betas).all():
        raise ValueError("shape statistics require finite non-empty 2D betas")
    covariance = np.cov(betas, rowvar=False, ddof=0)
    covariance = np.atleast_2d(covariance)
    eigenvalues = np.maximum(np.linalg.eigvalsh(covariance), 0.0)[::-1]
    total = float(eigenvalues.sum())
    if total > 0:
        probabilities = eigenvalues[eigenvalues > 0] / total
        effective_rank = float(np.exp(-(probabilities * np.log(probabilities)).sum()))
    else:
        effective_rank = 0.0
    return {
        "count": len(betas), "dimension": int(betas.shape[1]),
        "per_dimension_std": betas.std(axis=0).tolist(),
        "covariance_eigenvalues_descending": eigenvalues.tolist(),
        "effective_rank": effective_rank,
    }


def nearest_distances(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    # Pools are expected to contain hundreds, not millions, of identities.
    distances = ((source[:, None, :] - target[None, :, :]) ** 2).sum(axis=-1)
    return np.sqrt(distances.min(axis=1))


def gender_matched_nearest_distances(
    source: np.ndarray,
    source_genders: list[str],
    target: np.ndarray,
    target_genders: list[str],
) -> np.ndarray:
    """Return beta-space distances without conflating male/female models."""
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    source_genders = np.asarray(source_genders).astype(str)
    target_genders = np.asarray(target_genders).astype(str)
    if (source.ndim != 2 or target.ndim != 2
            or source.shape[1] != target.shape[1]
            or source.shape[0] != len(source_genders)
            or target.shape[0] != len(target_genders)):
        raise ValueError("shape/gender arrays are inconsistent")
    result = np.empty(len(source), dtype=np.float64)
    for gender in sorted(set(source_genders.tolist())):
        source_indices = np.flatnonzero(source_genders == gender)
        target_indices = np.flatnonzero(target_genders == gender)
        if not len(target_indices):
            raise ValueError(f"target pool has no {gender} shapes")
        result[source_indices] = nearest_distances(
            source[source_indices], target[target_indices]
        )
    return result


def load_interx_train_shapes(split: Path, beta_dir: Path) -> tuple[np.ndarray, list[str]]:
    rows = {}
    for name in [line.strip() for line in split.read_text().splitlines() if line.strip()]:
        with np.load(beta_dir / f"{name}.npz", allow_pickle=False) as archive:
            betas = np.asarray(archive["betas"], dtype=np.float64).reshape(2, 10)
            genders = np.asarray(archive["gender"]).astype(str).tolist()
        for gender, beta in zip(genders, betas):
            key = gender + ":" + hashlib.sha256(np.asarray(beta, dtype="<f8").tobytes()).hexdigest()
            rows.setdefault(key, (beta, gender))
    ordered = [rows[key] for key in sorted(rows)]
    return np.stack([row[0] for row in ordered]), [row[1] for row in ordered]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transfer-input-manifest", type=Path, required=True)
    parser.add_argument("--interx-train-split", type=Path,
                        default=Path("dataset/interx/splits/train.txt"))
    parser.add_argument("--interx-beta-dir", type=Path,
                        default=Path("dataset/interx/betas"))
    parser.add_argument("--maximum-rmse-m", type=float, default=0.01)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if not 0 < args.maximum_rmse_m <= 0.05:
        raise ValueError("maximum RMSE must be in (0, 0.05]")

    input_manifest_path = args.transfer_input_manifest.resolve()
    input_manifest = json.loads(input_manifest_path.read_text(encoding="utf-8"))
    if (input_manifest.get("schema")
            != "hhc_humanml3d_amass_smplh_to_smplx_transfer_input_v1"
            or input_manifest.get("status") != "ready_to_run_official_transfer_model"
            or input_manifest.get("transfer_batch_size") != 1):
        raise ValueError("invalid or unsafe transfer input manifest")
    correspondence_record = input_manifest["required_assets"]["correspondence"]
    correspondence_path = Path(correspondence_record["path"])
    if artifact(correspondence_path) != correspondence_record:
        raise ValueError("official correspondence artifact is stale")
    matrix = deformation_matrix(correspondence_path)

    accepted = []
    rejected = []
    for record in input_manifest["mesh_records"]:
        identity = record["identity_id"]
        gender = record["gender"]
        source_mesh = Path(record["mesh"]["path"])
        if artifact(source_mesh) != record["mesh"]:
            raise ValueError(f"source mesh artifact is stale: {identity}")
        fitted_path = (input_manifest_path.parent / "fitted" / gender
                       / f"{identity}.pkl")
        if not fitted_path.is_file():
            raise FileNotFoundError(fitted_path)
        with fitted_path.open("rb") as handle:
            fit = pickle.load(handle)
        betas = numpy_value(fit["betas"]).reshape(-1, 10)
        vertices = numpy_value(fit["vertices"])
        if betas.shape != (1, 10) or vertices.ndim != 3 or vertices.shape[0] != 1:
            raise ValueError(f"unsafe batched or malformed transfer output: {fitted_path}")
        source_vertices = obj_vertices(source_mesh)
        if matrix.shape[1] != len(source_vertices):
            raise ValueError(
                f"correspondence/source topology mismatch {matrix.shape} vs "
                f"{source_vertices.shape} for {identity}"
            )
        target = matrix @ source_vertices
        estimate = np.asarray(vertices[0], dtype=np.float64)
        if target.shape != estimate.shape:
            raise ValueError(f"target/estimate topology mismatch for {identity}")
        errors = np.linalg.norm(estimate - target, axis=1)
        rmse = float(np.sqrt(np.mean(errors ** 2)))
        result = {
            "source_identity_id": identity, "gender": gender,
            "betas": betas[0].astype(np.float32), "transfer_vertex_rmse_m": rmse,
            "transfer_vertex_mean_error_m": float(errors.mean()),
            "transfer_vertex_max_error_m": float(errors.max()),
            "fit": artifact(fitted_path), "source_mesh": record["mesh"],
        }
        if (not np.isfinite(betas).all() or not np.isfinite(errors).all()
                or rmse > args.maximum_rmse_m):
            rejected.append({key: value for key, value in result.items() if key != "betas"})
        else:
            accepted.append(result)

    if rejected:
        raise ValueError(
            f"{len(rejected)} transferred identities failed the {args.maximum_rmse_m} m "
            f"RMSE/finite gate; first: {rejected[0]}"
        )
    if len(accepted) != input_manifest["identity_count"]:
        raise ValueError("accepted donor count does not match frozen input inventory")
    target_shapes = {}
    for row in accepted:
        shape_id = canonical_shape_id(row["gender"], row["betas"])
        if shape_id not in target_shapes:
            target_shapes[shape_id] = {
                **row, "shape_id": shape_id,
                "source_identity_ids": [row["source_identity_id"]],
            }
        else:
            target_shapes[shape_id]["source_identity_ids"].append(
                row["source_identity_id"]
            )
            if (row["transfer_vertex_rmse_m"]
                    < target_shapes[shape_id]["transfer_vertex_rmse_m"]):
                source_ids = target_shapes[shape_id]["source_identity_ids"]
                target_shapes[shape_id] = {
                    **row, "shape_id": shape_id,
                    "source_identity_ids": source_ids,
                }
    accepted = [target_shapes[key] for key in sorted(target_shapes)]
    donor_betas = np.stack([row["betas"] for row in accepted])
    donor_genders = [row["gender"] for row in accepted]
    donor_ids = [row["shape_id"] for row in accepted]

    interx_betas, interx_genders = load_interx_train_shapes(
        args.interx_train_split.resolve(), args.interx_beta_dir.resolve())
    donor_to_interx = gender_matched_nearest_distances(
        donor_betas, donor_genders, interx_betas, interx_genders
    )
    interx_to_donor = gender_matched_nearest_distances(
        interx_betas, interx_genders, donor_betas, donor_genders
    )
    output_dir = args.output_dir.resolve()
    donor_path = output_dir / "humanml3d_amass_smplx_donors.npz"
    atomic_npz(
        donor_path, shape_id=np.asarray(donor_ids, dtype="U64"),
        gender=np.asarray(donor_genders, dtype="U6"), betas=donor_betas,
        transfer_vertex_rmse_m=np.asarray(
            [row["transfer_vertex_rmse_m"] for row in accepted], dtype=np.float32),
    )
    for row in accepted:
        row.pop("betas")
    payload = {
        "schema": "hhc_humanml3d_amass_smplx_shape_donor_pool_v1",
        "status": "complete",
        "transfer_input_manifest": artifact(input_manifest_path),
        "donor_pool": artifact(donor_path),
        "donor_count": len(donor_ids),
        "transferred_source_identity_count": input_manifest["identity_count"],
        "target_shape_duplicates_removed": input_manifest["identity_count"] - len(donor_ids),
        "gender_counts": dict(sorted(Counter(donor_genders).items())),
        "maximum_allowed_transfer_vertex_rmse_m": args.maximum_rmse_m,
        "maximum_observed_transfer_vertex_rmse_m": max(
            row["transfer_vertex_rmse_m"] for row in accepted),
        "shape_statistics": {
            "amass_smplx": shape_statistics(donor_betas),
            "interx_train_118_pool": shape_statistics(interx_betas),
            "interx_gender_counts": dict(sorted(Counter(interx_genders).items())),
            "amass_to_nearest_same_gender_interx_l2": {
                "mean": float(donor_to_interx.mean()),
                "median": float(np.median(donor_to_interx)),
                "maximum": float(donor_to_interx.max()),
            },
            "interx_to_nearest_same_gender_amass_l2": {
                "mean": float(interx_to_donor.mean()),
                "median": float(np.median(interx_to_donor)),
                "maximum": float(interx_to_donor.max()),
            },
        },
        "records": accepted,
        "invariants": input_manifest["invariants"],
    }
    manifest_path = output_dir / "humanml3d_amass_smplx_donor_manifest.json"
    atomic_json(manifest_path, payload)
    print(json.dumps({
        "status": "complete", "donor_count": len(donor_ids),
        "transferred_source_identity_count": input_manifest["identity_count"],
        "gender_counts": payload["gender_counts"],
        "maximum_rmse_m": payload["maximum_observed_transfer_vertex_rmse_m"],
        "manifest": str(manifest_path),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
