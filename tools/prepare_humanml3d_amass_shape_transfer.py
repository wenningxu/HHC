#!/usr/bin/env python3
"""Render neutral SMPL+H donor meshes and freeze official SMPL-X transfer configs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch


SMPLX_TRANSFER_REVISION = "1265df7ba545e8b00f72e7c557c766e15c71632f"
TRUST_NCG_REVISION = "7510ae709bc0722955343e3bd06144406c72c7f0"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def atomic_json(path: Path, payload: dict) -> None:
    atomic_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def artifact(path: Path) -> dict:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return {"path": str(path), "bytes": path.stat().st_size,
            "sha256": sha256_file(path)}


def repository_revision(path: Path) -> str:
    return subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"], check=True,
        text=True, capture_output=True,
    ).stdout.strip()


def repository_diff(path: Path) -> bytes:
    return subprocess.run(
        ["git", "-C", str(path), "diff", "--binary"], check=True,
        capture_output=True,
    ).stdout


def write_obj(path: Path, vertices: np.ndarray, faces: np.ndarray) -> None:
    lines = [f"v {x:.9g} {y:.9g} {z:.9g}\n" for x, y, z in vertices]
    lines.extend(f"f {a + 1} {b + 1} {c + 1}\n" for a, b, c in faces)
    atomic_text(path, "".join(lines))


def transfer_config(
    mesh_dir: Path,
    output_dir: Path,
    correspondence: Path,
    smplx_model_root: Path,
    gender: str,
    batch_size: int,
) -> dict:
    return {
        "use_cuda": True,
        "interactive": False,
        "output_folder": str(output_dir.resolve()),
        "batch_size": batch_size,
        "summary_steps": 100,
        "datasets": {
            "num_workers": 0,
            "name": "mesh-folder",
            "mesh_folder": {"data_folder": str(mesh_dir.resolve())},
        },
        "deformation_transfer_path": str(correspondence.resolve()),
        "mask_ids_fname": "",
        "edge_fitting": {"per_part": False, "reduction": "mean"},
        "optim": {"type": "trust-ncg", "maxiters": 100, "gtol": 1e-6},
        "body_model": {
            "model_type": "smplx",
            "gender": gender,
            "folder": str(smplx_model_root.resolve()),
            "ext": "npz",
            "use_compressed": False,
            "use_face_contour": True,
            "smplx": {"betas": {"num": 10}, "expression": {"num": 10}},
        },
    }


def validate_inventory(manifest_path: Path) -> tuple[dict, np.lib.npyio.NpzFile]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (manifest.get("schema") != "hhc_humanml3d_amass_smplh_shape_inventory_v1"
            or manifest.get("status")
            != "ready_for_official_smplh_to_smplx_transfer"):
        raise ValueError("invalid SMPL+H shape inventory manifest")
    record = manifest["inventory"]["npz"]
    inventory_path = Path(record["path"]).resolve()
    if artifact(inventory_path) != record:
        raise ValueError("SMPL+H shape inventory artifact is stale")
    inventory = np.load(inventory_path, allow_pickle=False)
    required = {"identity_id", "gender", "betas", "source_sequence_count"}
    if set(inventory.files) != required:
        raise ValueError(f"inventory fields {set(inventory.files)} != {required}")
    count = len(inventory["identity_id"])
    if (inventory["betas"].shape != (count, 16)
            or inventory["gender"].shape != (count,)
            or set(inventory["gender"].tolist()) - {"male", "female"}
            or len(set(inventory["identity_id"].tolist())) != count):
        raise ValueError("invalid identity/gender/beta shapes in inventory")
    return manifest, inventory


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory-manifest", type=Path, required=True)
    parser.add_argument("--humanml3d-root", type=Path,
                        default=Path("/mnt/8T-HDD/HumanML3D"))
    parser.add_argument("--human-body-prior-root", type=Path,
                        default=Path("/mnt/8T-HDD/HumanML3D"))
    parser.add_argument("--smplx-model-root", type=Path,
                        default=Path("/home/wenning/projects/MMM/body_models"))
    parser.add_argument("--transfer-root", type=Path,
                        default=Path("/mnt/8T-HDD/wenning/smplx-transfer"))
    parser.add_argument("--trust-ncg-root", type=Path,
                        default=Path("/mnt/8T-HDD/wenning/torch-trust-ncg"))
    parser.add_argument("--compatibility-patch", type=Path,
                        default=Path("patches/smplx_transfer_python312_io_compat.patch"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--render-batch-size", type=int, default=32)
    parser.add_argument("--transfer-batch-size", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.render_batch_size < 1 or args.transfer_batch_size < 1:
        raise ValueError("batch sizes must be positive")
    if args.transfer_batch_size != 1:
        raise ValueError(
            "official transfer_model writes unsliced batch var_dict into every pkl; "
            "transfer batch size must remain 1 to preserve identity-to-beta pairing"
        )

    inventory_manifest, inventory = validate_inventory(
        args.inventory_manifest.resolve())
    output_dir = args.output_dir.resolve()
    correspondence = (args.transfer_root.resolve()
                      / "transfer_data/smplh2smplx_deftrafo_setup.pkl")
    transfer_root = args.transfer_root.resolve()
    trust_ncg_root = args.trust_ncg_root.resolve()
    compatibility_patch = args.compatibility_patch.resolve()
    if repository_revision(transfer_root) != SMPLX_TRANSFER_REVISION:
        raise ValueError("SMPL-X transfer repository revision changed")
    if repository_revision(trust_ncg_root) != TRUST_NCG_REVISION:
        raise ValueError("torch-trust-ncg repository revision changed")
    patch_bytes = compatibility_patch.read_bytes()
    active_diff = repository_diff(transfer_root)
    if active_diff != patch_bytes:
        raise ValueError(
            "SMPL-X transfer worktree differs from the frozen Python3.12/I-O patch"
        )
    required = {
        "smplh_male": args.humanml3d_root.resolve()
            / "body_models/smplh/male/model.npz",
        "smplh_female": args.humanml3d_root.resolve()
            / "body_models/smplh/female/model.npz",
        "smplx_male": args.smplx_model_root.resolve()
            / "smplx/SMPLX_MALE.npz",
        "smplx_female": args.smplx_model_root.resolve()
            / "smplx/SMPLX_FEMALE.npz",
        "correspondence": correspondence,
    }
    required_artifacts = {key: artifact(path) for key, path in required.items()}

    human_body_prior_root = args.human_body_prior_root.resolve()
    sys.path.insert(0, str(human_body_prior_root))
    from human_body_prior.body_model.body_model import BodyModel

    ids = inventory["identity_id"].tolist()
    genders = inventory["gender"].tolist()
    betas = np.asarray(inventory["betas"], dtype=np.float32)
    mesh_records = []
    for gender in ("female", "male"):
        indices = [index for index, value in enumerate(genders) if value == gender]
        model = BodyModel(
            bm_fname=str(required[f"smplh_{gender}"]), num_betas=16,
            persistant_buffer=False,
        ).eval()
        faces = model.f.detach().cpu().numpy().astype(np.int64)
        mesh_dir = output_dir / "meshes" / gender
        mesh_dir.mkdir(parents=True, exist_ok=True)
        with torch.no_grad():
            for start in range(0, len(indices), args.render_batch_size):
                batch_indices = indices[start:start + args.render_batch_size]
                batch_betas = torch.from_numpy(betas[batch_indices])
                vertices = model(betas=batch_betas).v.detach().cpu().numpy()
                for local_index, inventory_index in enumerate(batch_indices):
                    path = mesh_dir / f"{ids[inventory_index]}.obj"
                    if not (args.resume and path.is_file()):
                        write_obj(path, vertices[local_index], faces)
                    mesh_records.append({
                        "identity_id": ids[inventory_index], "gender": gender,
                        "source_beta_index": inventory_index, "mesh": artifact(path),
                    })

    configs = {}
    commands = {}
    python_path = os.pathsep.join((str(transfer_root), str(trust_ncg_root)))
    for gender in ("female", "male"):
        mesh_dir = output_dir / "meshes" / gender
        fitted_dir = output_dir / "fitted" / gender
        fitted_dir.mkdir(parents=True, exist_ok=True)
        config = transfer_config(
            mesh_dir, fitted_dir, correspondence, args.smplx_model_root,
            gender, args.transfer_batch_size,
        )
        config_path = output_dir / "configs" / f"smplh2smplx_{gender}.yaml"
        # JSON is valid YAML and avoids string-escaping ambiguity in generated paths.
        atomic_json(config_path, config)
        configs[gender] = artifact(config_path)
        commands[gender] = [
            "env", f"PYTHONPATH={python_path}", sys.executable,
            "-m", "transfer_model", "--exp-cfg",
            str(config_path.resolve()),
        ]

    payload = {
        "schema": "hhc_humanml3d_amass_smplh_to_smplx_transfer_input_v1",
        "status": "ready_to_run_official_transfer_model",
        "inventory_manifest": artifact(args.inventory_manifest.resolve()),
        "inventory_npz": inventory_manifest["inventory"]["npz"],
        "required_assets": required_artifacts,
        "runtime": {
            "python": sys.executable,
            "working_directory": str(transfer_root),
            "pythonpath": python_path,
            "smplx_transfer_revision": SMPLX_TRANSFER_REVISION,
            "torch_trust_ncg_revision": TRUST_NCG_REVISION,
            "compatibility_patch": artifact(compatibility_patch),
            "compatibility_patch_scope": (
                "Python 3.12 dataclass default_factory and output mesh I/O only; "
                "deformation transfer, fitting losses and optimizer are unchanged"
            ),
        },
        "identity_count": len(ids),
        "gender_counts": {gender: genders.count(gender)
                          for gender in ("female", "male")},
        "neutral_pose": True,
        "source_shape_dimension": 16,
        "target_shape_dimension": 10,
        "transfer_batch_size": 1,
        "transfer_batch_size_reason": (
            "released transfer_model pickles the unsliced batch var_dict for each path; "
            "batch_size=1 prevents cross-identity parameter ambiguity"
        ),
        "mesh_records": sorted(mesh_records, key=lambda row: row["identity_id"]),
        "configs": configs,
        "commands": commands,
        "invariants": {
            "official_deformation_correspondence": True,
            "target_gender_matches_source_gender": True,
            "direct_beta_copy": False,
        },
    }
    manifest_path = output_dir / "transfer_input_manifest.json"
    atomic_json(manifest_path, payload)
    print(json.dumps({
        "status": payload["status"], "identity_count": len(ids),
        "gender_counts": payload["gender_counts"],
        "manifest": str(manifest_path), "commands": commands,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
