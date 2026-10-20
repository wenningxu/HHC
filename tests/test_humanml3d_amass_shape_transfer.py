import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "prepare_transfer", ROOT / "tools/prepare_humanml3d_amass_shape_transfer.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_transfer_config_preserves_gender_and_official_correspondence(tmp_path):
    config = MODULE.transfer_config(
        tmp_path / "mesh", tmp_path / "fitted", tmp_path / "official.pkl",
        tmp_path / "body_models", "female", 1,
    )
    assert config["body_model"]["model_type"] == "smplx"
    assert config["body_model"]["gender"] == "female"
    assert config["body_model"]["smplx"]["betas"]["num"] == 10
    assert config["deformation_transfer_path"].endswith("official.pkl")
    assert config["datasets"]["mesh_folder"]["data_folder"].endswith("mesh")
    assert config["edge_fitting"]["per_part"] is False
    assert config["batch_size"] == 1


def test_inventory_validation_fails_on_stale_hash(tmp_path):
    inventory = tmp_path / "inventory.npz"
    np.savez_compressed(
        inventory,
        identity_id=np.asarray(["a" * 64]), gender=np.asarray(["female"]),
        betas=np.zeros((1, 16), dtype=np.float32),
        source_sequence_count=np.asarray([1], dtype=np.int32),
    )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({
        "schema": "hhc_humanml3d_amass_smplh_shape_inventory_v1",
        "status": "ready_for_official_smplh_to_smplx_transfer",
        "inventory": {"npz": {
            "path": str(inventory), "bytes": inventory.stat().st_size,
            "sha256": "0" * 64,
        }},
    }))
    try:
        MODULE.validate_inventory(manifest)
    except ValueError as error:
        assert "stale" in str(error)
    else:
        raise AssertionError("stale shape inventory was accepted")


def test_inventory_validation_accepts_exact_contract(tmp_path):
    inventory = tmp_path / "inventory.npz"
    np.savez_compressed(
        inventory,
        identity_id=np.asarray(["a" * 64, "b" * 64]),
        gender=np.asarray(["female", "male"]),
        betas=np.zeros((2, 16), dtype=np.float32),
        source_sequence_count=np.asarray([2, 3], dtype=np.int32),
    )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({
        "schema": "hhc_humanml3d_amass_smplh_shape_inventory_v1",
        "status": "ready_for_official_smplh_to_smplx_transfer",
        "inventory": {"npz": {
            "path": str(inventory), "bytes": inventory.stat().st_size,
            "sha256": sha256(inventory),
        }},
    }))
    _, loaded = MODULE.validate_inventory(manifest)
    try:
        assert loaded["betas"].shape == (2, 16)
    finally:
        loaded.close()


def test_transfer_runtime_patch_exactly_matches_active_official_clone():
    transfer_root = Path("/mnt/8T-HDD/wenning/smplx-transfer")
    trust_root = Path("/mnt/8T-HDD/wenning/torch-trust-ncg")
    patch = ROOT / "patches/smplx_transfer_python312_io_compat.patch"
    assert MODULE.repository_revision(transfer_root) == MODULE.SMPLX_TRANSFER_REVISION
    assert MODULE.repository_revision(trust_root) == MODULE.TRUST_NCG_REVISION
    assert MODULE.repository_diff(transfer_root) == patch.read_bytes()


def test_generated_config_parses_with_pinned_official_runtime(tmp_path):
    transfer_root = Path("/mnt/8T-HDD/wenning/smplx-transfer")
    trust_root = Path("/mnt/8T-HDD/wenning/torch-trust-ncg")
    sys.path[:0] = [str(transfer_root), str(trust_root)]
    from transfer_model.config.cmd_parser import parse_args
    config_path = tmp_path / "config.yaml"
    config_path.write_text(json.dumps(MODULE.transfer_config(
        tmp_path / "mesh", tmp_path / "out", tmp_path / "correspondence.pkl",
        tmp_path / "models", "male", 1,
    )))
    previous = sys.argv
    try:
        sys.argv = ["transfer_model", "--exp-cfg", str(config_path)]
        config = parse_args()
    finally:
        sys.argv = previous
    assert config.body_model.gender == "male"
    assert config.body_model.smplx.betas.num == 10
    assert config.optim.type == "trust-ncg"
    assert config.edge_fitting.per_part is False
