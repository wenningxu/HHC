import hashlib
import json
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools/prepare_interx_shape_augmentation_plan.py"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def shape_id(gender: str, beta: np.ndarray) -> str:
    payload = {"gender": gender, "betas": [float(value) for value in beta]}
    return hashlib.sha256(json.dumps(
        payload, sort_keys=True, separators=(",", ":")
    ).encode()).hexdigest()


def write_fixture(root: Path) -> tuple[Path, Path]:
    data = root / "interx"
    for name in ("splits", "betas", "contact", "texts_processed", "motions"):
        (data / name).mkdir(parents=True)
    sample = "G001T000A000R000"
    (data / "splits/train.txt").write_text(sample + "\n")
    with h5py.File(data / "motions/train.h5", "w") as handle:
        handle.create_dataset(sample, data=np.zeros((40, 56, 6), dtype=np.float32))
    np.savez_compressed(
        data / "betas" / f"{sample}.npz",
        betas=np.stack([np.zeros(10), np.ones(10)]).astype(np.float32),
        gender=np.asarray(["male", "female"]),
    )
    timing = np.zeros(40, dtype=np.uint8)
    timing[10:20] = 1
    np.save(data / "contact" / f"{sample}.npy", timing)
    (data / "texts_processed" / f"{sample}.txt").write_text(
        "two people hug#two/OTHER people/OTHER hug/OTHER#0#0\n"
    )

    donor_dir = root / "donors"
    donor_dir.mkdir()
    genders = np.asarray(["male", "female", "male", "female"])
    betas = np.stack([
        np.full(10, value, dtype=np.float32) for value in (0, 1, 2, 3)
    ])
    ids = np.asarray([shape_id(gender, beta)
                      for gender, beta in zip(genders, betas)])
    donor_npz = donor_dir / "donors.npz"
    np.savez_compressed(
        donor_npz, shape_id=ids, gender=genders, betas=betas,
        transfer_vertex_rmse_m=np.full(4, 0.001, dtype=np.float32),
    )
    manifest = donor_dir / "manifest.json"
    manifest.write_text(json.dumps({
        "schema": "hhc_humanml3d_amass_smplx_shape_donor_pool_v1",
        "status": "complete",
        "donor_count": 4,
        "donor_pool": {
            "path": str(donor_npz.resolve()), "bytes": donor_npz.stat().st_size,
            "sha256": sha256(donor_npz),
        },
        "records": [
            {
                "shape_id": identity, "gender": gender,
                "source_identity_ids": [f"source-{index}"],
                "transfer_vertex_rmse_m": 0.001,
            }
            for index, (identity, gender) in enumerate(zip(ids, genders))
        ],
    }))
    return data, manifest


def test_external_amass_donor_plan_is_single_factor_and_strictly_validated(tmp_path):
    data, donor_manifest = write_fixture(tmp_path)
    output = tmp_path / "plan"
    subprocess.run([
        sys.executable, str(SCRIPT),
        "--split", str(data / "splits/train.txt"),
        "--motion-file", str(data / "motions/train.h5"),
        "--beta-dir", str(data / "betas"),
        "--contact-dir", str(data / "contact"),
        "--text-dir", str(data / "texts_processed"),
        "--external-donor-manifest", str(donor_manifest),
        "--output-dir", str(output), "--seed", "123",
    ], check=True)
    from utils.interx_shape_augmentation_plan import (
        validate_interx_shape_augmentation_plan,
    )
    manifest, _, records, _, _ = validate_interx_shape_augmentation_plan(
        output / "plan_manifest.json"
    )
    assert manifest["donor_pool_kind"] == (
        "humanml3d_train_amass_transferred_smplx"
    )
    assert manifest["unique_donor_shapes"] == 4
    assert len(records) == 1
    assert len({row["shape_id"] for row in records[0]["shape_donors"]}) == 2
    assert all(row["donor_pool_kind"] == manifest["donor_pool_kind"]
               for row in records[0]["shape_donors"])
