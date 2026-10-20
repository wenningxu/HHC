import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools/build_humanml3d_amass_shape_inventory.py"


def write_fixture(root: Path) -> None:
    (root / "HumanML3D").mkdir(parents=True)
    (root / "HumanML3D/train.txt").write_text("000001\nM000001\n000002\n000003\n")
    rows = [
        ("000001.npy", "./pose_data/CMU/S1/a.npy"),
        ("000002.npy", "./pose_data/CMU/S1/b.npy"),
        ("000003.npy", "./pose_data/humanact12/human1.npy"),
    ]
    with (root / "index.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["new_name", "source_path"])
        writer.writeheader()
        for new_name, source_path in rows:
            writer.writerow({"new_name": new_name, "source_path": source_path})
    for clip, gender in [("a", "male"), ("b", b"male")]:
        path = root / "amass_data/CMU/S1" / f"{clip}.npz"
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, betas=np.arange(16, dtype=np.float64), gender=gender)


def test_inventory_is_train_only_mirror_deduplicated_and_identity_deduplicated(tmp_path):
    source = tmp_path / "HumanML3D"
    output = tmp_path / "out"
    write_fixture(source)
    subprocess.run([
        sys.executable, str(SCRIPT), "--humanml3d-root", str(source),
        "--amass-root", str(source / "amass_data"), "--output-dir", str(output),
    ], check=True)
    manifest = json.loads((output / "smplh_shape_inventory_manifest.json").read_text())
    assert manifest["source"]["train_entries_including_mirrors"] == 4
    assert manifest["source"]["canonical_train_entries"] == 3
    assert manifest["source"]["humanact12_entries_excluded"] == 1
    assert manifest["inventory"]["source_sequence_count"] == 2
    assert manifest["inventory"]["identity_count"] == 1
    assert manifest["inventory"]["gender_counts"] == {"male": 1}
    inventory_artifact = manifest["inventory"]["npz"]
    assert inventory_artifact["path"] == str(output / "smplh_shape_inventory.npz")
    assert len(inventory_artifact["sha256"]) == 64
    inventory = np.load(inventory_artifact["path"])
    assert inventory["betas"].shape == (1, 16)
    assert inventory["source_sequence_count"].tolist() == [2]


def test_inventory_fails_closed_when_a_required_train_source_is_missing(tmp_path):
    source = tmp_path / "HumanML3D"
    write_fixture(source)
    (source / "amass_data/CMU/S1/b.npz").unlink()
    result = subprocess.run([
        sys.executable, str(SCRIPT), "--humanml3d-root", str(source),
        "--amass-root", str(source / "amass_data"),
        "--output-dir", str(tmp_path / "out"),
    ], text=True, capture_output=True)
    assert result.returncode != 0
    assert "1/2 HumanML3D-train AMASS files are missing" in result.stderr
