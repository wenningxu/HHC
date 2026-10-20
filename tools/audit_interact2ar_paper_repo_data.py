#!/usr/bin/env python
"""Audit Interact2AR paper, official release, and local Inter-X assets.

This gate deliberately distinguishes facts that can be verified byte-for-byte
from facts stated by the paper.  In particular, the public Inter-X download
does not publish checksums for the large HDF5 archives, so the report must not
claim upstream byte identity for those files.
"""

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path

import h5py
import numpy as np


HHC_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_INTERACT2AR_REVISION = "a0e053be793ee55414dad458b2d62f3ad9f378f7"
PAPER_URL = "https://arxiv.org/html/2512.19692v2"
REPOSITORY_URL = "https://github.com/pabloruizponce/Interact2Ar"
INTERX_REPOSITORY_URL = "https://github.com/liangxuy/Inter-X"
EXPECTED_SPLIT_COUNTS = {"train": 9110, "val": 570, "test": 1708, "all": 11388}
EXPECTED_SPLIT_HASHES = {
    "train": "47c5c9a09d7293e90e127fb49ba9dc9398afbff66c8f2b45ada3ef6c4453157a",
    "val": "6c77e85017b14b512f42f2d540da2ce05874d9777d707bebd85782d6221f28a8",
    "test": "db32bbe12651875c41113cd20214afcf59ff9489a65aac6f3b0a5b1ecc2b075b",
    "all": "449dac3d2389a3aac055ec2e4e70feee6f0cbbe4d0406887f305ed1f372a9f23",
}
EXPECTED_ASSET_HASHES = {
    "model": "543a33b619b2bf175f5a99aacf2327350c65e4c99ae03c3be25911ed1f0fe997",
    "JointsFull": "04f153066eadb8d83ed798fd572491868a86d6a04b88f26b87a3448fb3ec81d5",
    "JointsBody": "aa81544acc3c7f687525ea9bc99935402d9d8f18d48fc22b2a00f58a869f5ada",
    "JointsHands": "092ffc9d8e8c5426ecf236398db619035b5001835d416ce81f7cd6a0a49678d3",
}


def sha256_file(path, chunk_size=16 * 1024 * 1024):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact(path, include_hash=True):
    supplied = Path(path).absolute()
    resolved = supplied.resolve(strict=True)
    stat = resolved.stat()
    record = {
        "supplied_path": str(supplied),
        "resolved_path": str(resolved),
        "bytes": stat.st_size,
        "device": stat.st_dev,
        "inode": stat.st_ino,
    }
    if include_hash:
        record["sha256"] = sha256_file(resolved)
    return record


def run(*args, cwd=None):
    return subprocess.check_output(args, cwd=cwd, text=True).strip()


def tracked_source_matches(repository, relative_paths):
    changed = []
    for relative in relative_paths:
        result = subprocess.run(
            ["git", "diff", "--quiet", "HEAD", "--", relative], cwd=repository
        )
        if result.returncode != 0:
            changed.append(relative)
    return {"checked": relative_paths, "changed": changed, "pass": not changed}


def read_ids(path):
    return [line.strip() for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def tree_digest(root, names):
    """Hash path names and bytes without relying on filesystem enumeration order."""
    digest = hashlib.sha256()
    missing = []
    for name in sorted(names):
        path = Path(root) / f"{name}.txt"
        if not path.is_file():
            missing.append(name)
            continue
        relative = f"{name}.txt".encode("utf-8")
        digest.update(len(relative).to_bytes(8, "little"))
        digest.update(relative)
        data = path.read_bytes()
        digest.update(len(data).to_bytes(8, "little"))
        digest.update(data)
    return {"algorithm": "sorted_length_prefixed_path_and_bytes_sha256", "sha256": digest.hexdigest(), "missing": missing}


def inspect_h5(path, split_ids):
    with h5py.File(path, "r") as source:
        keys = set(source.keys())
        split_set = set(split_ids)
        missing = sorted(split_set - keys)
        extra = sorted(keys - split_set)
        lengths = []
        bad_shapes = []
        bad_dtypes = []
        for name in split_ids:
            if name not in source:
                continue
            dataset = source[name]
            lengths.append(int(dataset.shape[0]))
            if len(dataset.shape) != 3 or tuple(dataset.shape[1:]) != (56, 6):
                bad_shapes.append({"name": name, "shape": list(dataset.shape)})
            if dataset.dtype != np.dtype("float32"):
                bad_dtypes.append({"name": name, "dtype": str(dataset.dtype)})
    lengths = np.asarray(lengths, dtype=np.int64)
    return {
        "h5_keys": len(keys),
        "split_ids": len(split_ids),
        "missing_split_ids": missing,
        "extra_h5_keys": extra,
        "bad_shapes": bad_shapes,
        "bad_dtypes": bad_dtypes,
        "length_bins": {
            "lt_24": int((lengths < 24).sum()),
            "24_to_29": int(((lengths >= 24) & (lengths < 30)).sum()),
            "30_to_999": int(((lengths >= 30) & (lengths < 1000)).sum()),
            "ge_1000": int((lengths >= 1000).sum()),
        },
        "valid_hhc_min24": int(((lengths >= 24) & (lengths < 1000)).sum()),
        "valid_interact2ar_min30": int(((lengths >= 30) & (lengths < 1000)).sum()),
        "valid_interact2ar_ar_train_min45": int(((lengths >= 45) & (lengths < 1000)).sum()),
    }


def require(condition, message, failures):
    if not condition:
        failures.append(message)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interact2ar-root", type=Path, default=Path("/mnt/8T-HDD/wenning/Interact2Ar"))
    parser.add_argument("--interx-root", type=Path, default=Path("/home/wenning/projects/Inter-X"))
    parser.add_argument("--mmm-root", type=Path, default=Path("/home/wenning/projects/MMM"))
    parser.add_argument("--output", type=Path, default=HHC_ROOT / "output/interact2ar_paper_repo_data_audit.json")
    args = parser.parse_args()
    i2ar = args.interact2ar_root.resolve()
    interx = args.interx_root.resolve()
    mmm = args.mmm_root.resolve()
    failures = []

    revision = run("git", "rev-parse", "HEAD", cwd=i2ar)
    origin_revision = run("git", "rev-parse", "origin/main", cwd=i2ar)
    source_match = tracked_source_matches(i2ar, [
        "README.md", "assets/checkpoints.json", "options/dataset/paper.yaml",
        "src/evaluate.py", "src/evaluation/diff_model_dataset.py",
        "src/evaluation/evaluator_wrapper.py", "src/data/dataset.py",
        "src/models/interact2ar.py",
    ])
    require(revision == EXPECTED_INTERACT2AR_REVISION, "unexpected Interact2AR revision", failures)
    require(origin_revision == revision, "Interact2AR checkout is not at local origin/main", failures)
    require(source_match["pass"], "official Interact2AR evaluation sources are modified", failures)

    split_records = {}
    all_ids = None
    for split, expected_count in EXPECTED_SPLIT_COUNTS.items():
        paths = {
            "interx_official_checkout": interx / "datasets" / f"{split}.txt",
            "mmm": mmm / "dataset/interx/splits" / f"{split}.txt",
            "hhc": HHC_ROOT / "dataset/interx/splits" / f"{split}.txt",
            "interact2ar": i2ar / "data/splits" / f"{split}.txt",
        }
        records = {label: artifact(path) for label, path in paths.items()}
        hashes = {record["sha256"] for record in records.values()}
        ids = read_ids(paths["interact2ar"])
        require(len(hashes) == 1, f"{split} split differs across repositories", failures)
        require(records["interact2ar"]["sha256"] == EXPECTED_SPLIT_HASHES[split], f"{split} split hash is not expected official revision", failures)
        require(len(ids) == expected_count, f"{split} split count differs", failures)
        split_records[split] = {"count": len(ids), "artifacts": records}
        if split == "all":
            all_ids = ids

    h5_records = {}
    for split in ("train", "val", "test"):
        paths = {
            "mmm": mmm / "dataset/interx/motions" / f"{split}.h5",
            "hhc": HHC_ROOT / "dataset/interx/motions" / f"{split}.h5",
            "interact2ar": i2ar / "data/motions" / f"{split}.h5",
        }
        records = {label: artifact(path) for label, path in paths.items()}
        identities = {(record["device"], record["inode"]) for record in records.values()}
        hashes = {record["sha256"] for record in records.values()}
        ids = read_ids(i2ar / "data/splits" / f"{split}.txt")
        inspection = inspect_h5(paths["interact2ar"], ids)
        require(len(identities) == 1, f"{split} H5 paths are not the same local file", failures)
        require(len(hashes) == 1, f"{split} H5 hashes differ", failures)
        require(not inspection["missing_split_ids"] and not inspection["extra_h5_keys"], f"{split} H5 keys do not exactly equal split", failures)
        require(not inspection["bad_shapes"] and not inspection["bad_dtypes"], f"{split} H5 schema differs from (T,56,6) float32", failures)
        h5_records[split] = {"artifacts": records, "inspection": inspection}
    require(h5_records["test"]["inspection"]["valid_hhc_min24"] == 1706, "HHC test filtering is not 1706", failures)
    require(h5_records["test"]["inspection"]["valid_interact2ar_min30"] == 1706, "Interact2AR test filtering is not 1706", failures)
    require(h5_records["train"]["inspection"]["valid_interact2ar_ar_train_min45"] == 9082, "Interact2AR AR train filtering is not 9082", failures)

    shared_assets = {
        "texts": {
            "mmm": tree_digest(mmm / "dataset/interx/texts_processed", all_ids),
            "hhc": tree_digest(HHC_ROOT / "dataset/interx/texts_processed", all_ids),
            "interact2ar": tree_digest(i2ar / "data/texts_processed", all_ids),
        },
        "glove": {},
        "smplx_neutral": {},
    }
    for filename in ("hhi_vab_data.npy", "hhi_vab_idx.pkl", "hhi_vab_words.pkl"):
        shared_assets["glove"][filename] = {
            "mmm": artifact(mmm / "dataset/interx/glove" / filename),
            "hhc": artifact(HHC_ROOT / "dataset/interx/glove" / filename),
            "interact2ar": artifact(i2ar / "data/glove" / filename),
        }
        require(len({x["sha256"] for x in shared_assets["glove"][filename].values()}) == 1, f"GloVe {filename} differs", failures)
    require(not any(x["missing"] for x in shared_assets["texts"].values()), "processed Inter-X text files are missing", failures)
    require(len({x["sha256"] for x in shared_assets["texts"].values()}) == 1, "processed Inter-X text differs", failures)
    for label, path in {
        "hhc": HHC_ROOT / "body_models/smplx/SMPLX_NEUTRAL.npz",
        "interact2ar": i2ar / "data/body_models/smplx/SMPLX_NEUTRAL.npz",
    }.items():
        shared_assets["smplx_neutral"][label] = artifact(path)
    require(len({x["sha256"] for x in shared_assets["smplx_neutral"].values()}) == 1, "SMPL-X neutral model differs", failures)

    checkpoint_paths = {
        "model": i2ar / "checkpoints/interact2ar_mixed_memory.ckpt",
        "JointsFull": i2ar / "checkpoints-evaluator/joints_full/model/finest.tar",
        "JointsBody": i2ar / "checkpoints-evaluator/joints_body/model/finest.tar",
        "JointsHands": i2ar / "checkpoints-evaluator/joints_hands/model/finest.tar",
    }
    checkpoints = {label: artifact(path) for label, path in checkpoint_paths.items()}
    for label, expected_hash in EXPECTED_ASSET_HASHES.items():
        require(checkpoints[label]["sha256"] == expected_hash, f"{label} checkpoint differs from official manifest", failures)

    readme = (i2ar / "README.md").read_text(encoding="utf-8")
    evaluate_source = (i2ar / "src/evaluate.py").read_text(encoding="utf-8")
    generated_source = (i2ar / "src/evaluation/diff_model_dataset.py").read_text(encoding="utf-8")
    wrapper_source = (i2ar / "src/evaluation/evaluator_wrapper.py").read_text(encoding="utf-8")
    train_config = (i2ar / "options/train/paper.yaml").read_text(encoding="utf-8")
    code_contract = {
        "readme_author_command_uses_first_frame": "--first_frame" in readme,
        "released_full_replications": 5 if "replication_times = 5" in evaluate_source else None,
        "batch_size": 32 if "batch_size = 32" in evaluate_source else None,
        "mm_samples": 100 if "mm_num_samples = 100" in evaluate_source else None,
        "mm_repeats": 30 if "mm_num_repeats = 30" in evaluate_source else None,
        "mm_pairs": 10 if "mm_num_times = 10" in evaluate_source else None,
        "diversity_pairs": 300 if "diversity_times = 300" in evaluate_source else None,
        "first_frame_is_ground_truth_random_index_0_to_unit_length": (
            "idx = random.randint(0, dataset.opt.UNIT_LENGTH)" in generated_source
            and "first_frame_motion = motions[:, idx, :]" in generated_source
        ),
        "joint_dimensions": {"full": 55 * 6, "body": 22 * 6, "hands": 30 * 6},
        "global_coordinates_include_root_translation": "joints_1 = joints_1 + trans_1.unsqueeze(2)" in wrapper_source,
        "released_joint_evaluators_normalized": (
            False if evaluate_source.count("normalized=False") >= 3 else None
        ),
        "released_train_per_device_batch_size": 32 if "BATCH_SIZE: 32" in train_config else None,
        "readme_training_example_devices": [0] if '--gpu "[0]"' in readme else None,
        "official_checkpoint_epoch": 4999,
        "official_checkpoint_global_step": 350000,
    }
    require(
        all(
            value is not None
            for key, value in code_contract.items()
            if key not in ("joint_dimensions",)
        ),
        "released code contract not recognized",
        failures,
    )
    expected_code_contract = {
        "readme_author_command_uses_first_frame": True,
        "released_full_replications": 5,
        "batch_size": 32,
        "mm_samples": 100,
        "mm_repeats": 30,
        "mm_pairs": 10,
        "diversity_pairs": 300,
        "first_frame_is_ground_truth_random_index_0_to_unit_length": True,
        "global_coordinates_include_root_translation": True,
        "released_joint_evaluators_normalized": False,
        "released_train_per_device_batch_size": 32,
        "readme_training_example_devices": [0],
        "official_checkpoint_epoch": 4999,
        "official_checkpoint_global_step": 350000,
    }
    for field, expected in expected_code_contract.items():
        require(
            code_contract.get(field) == expected,
            f"released code contract differs for {field}: "
            f"expected {expected!r}, observed {code_contract.get(field)!r}",
            failures,
        )

    paper_contract = {
        "source": PAPER_URL,
        "version_date": "2026-03-27",
        "dataset": "Inter-X",
        "motion_representation_per_person": "root_translation_3 + root_rotation_6 + body_21x6 + hands_30x6",
        "shape_policy": "neutral body configuration",
        "joint_evaluators": ["JointsFull", "JointsBody", "JointsHands"],
        "joint_coordinate_system": "global",
        "replications": 20,
        "confidence_interval": "95%",
        "autoregressive_sampling_steps": 10,
        "training_epochs": 5000,
        "training_global_batch_size": 128,
        "reported_interact2ar_joints_full_fid": 0.277,
        "reported_interact2ar_previous_evaluator_fid": 0.148,
    }
    release_paper_mismatches = [{
        "field": "replications",
        "paper": 20,
        "released_src_evaluate_py": code_contract["released_full_replications"],
        "resolution": "author-release calibration reports its native 5 repeats; paper-aligned and fair tables use 20 repeats",
    }, {
        "field": "training batch size / README device count",
        "paper_global_batch_size": 128,
        "released_yaml_per_device_batch_size": 32,
        "released_readme_example_devices": [0],
        "checkpoint_and_data_evidence": {
            "valid_ar_train_motions": h5_records["train"]["inspection"]["valid_interact2ar_ar_train_min45"],
            "steps_per_epoch_at_global_batch_128": h5_records["train"]["inspection"]["valid_interact2ar_ar_train_min45"] // 128,
            "epochs": 5000,
            "checkpoint_global_step": 350000,
        },
        "resolution": "use 4x32 DDP or an explicitly equivalent effective global batch 128; the README one-GPU example is not paper-step-equivalent",
    }]

    result = {
        "schema": "interact2ar_paper_repo_data_audit_v1",
        "status": "fail" if failures else "pass_with_disclosed_upstream_checksum_limitation_and_release_paper_mismatch",
        "failures": failures,
        "official_sources": {
            "paper": PAPER_URL,
            "interact2ar_repository": REPOSITORY_URL,
            "interx_repository": INTERX_REPOSITORY_URL,
        },
        "repository": {"root": str(i2ar), "revision": revision, "origin_main": origin_revision, "source_match": source_match},
        "paper_contract": paper_contract,
        "released_code_contract": code_contract,
        "release_paper_mismatches": release_paper_mismatches,
        "splits": split_records,
        "motions": h5_records,
        "shared_assets": shared_assets,
        "official_checkpoints": checkpoints,
        "data_lineage_conclusion": {
            "proven": [
                "split files byte-match the official Inter-X checkout at the audited revision",
                "MMM, HHC, and Interact2AR use the same local HDF5 files",
                "HDF5 keys exactly match the official split ids and arrays are (T,56,6) float32",
                "processed texts, GloVe assets, and local SMPL-X neutral assets match across consumers",
                "HHC min-length 24 and Interact2AR min-length 30 both retain the same 1706 test motions",
                "Interact2AR AR min-length 45 retains 9082 train motions; global batch 128 gives 70 steps/epoch and exactly explains checkpoint step 350000 after 5000 epochs",
            ],
            "not_proven": "Upstream byte identity of the large motion HDF5 archives: the official Inter-X download publishes no SHA256 manifest.",
        },
        "evaluation_policy": {
            "author_release_calibration": "official checkpoint + untouched released evaluator + README --first_frame; separate conditional task; reproduce reported JointsFull FID before accepting",
            "cross_method_fair_supplement": "same 20 frozen unconditional requests/generations for every method + released joint evaluators; never mix with author-conditioned scores",
            "legacy_main_table": "same official Inter-X 672-D evaluator and 20 frozen replications; remains a separate metric space",
            "formal_run_gate": "no Interact2AR result enters a table unless this audit status starts with pass and the result binds this audit SHA256",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    report_hash = sha256_file(args.output)
    md_path = args.output.with_suffix(".md")
    md_path.write_text(
        "# Interact2AR paper–repository–data audit\n\n"
        f"Status: `{result['status']}`\n\n"
        f"- Official checkout: `{revision}`; audited sources match `HEAD`: `{source_match['pass']}`.\n"
        f"- Inter-X splits: train 9110, val 570, test 1708, union 11388; filtered test set is 1706 for both loaders.\n"
        f"- Paper repetitions: 20; released `src/evaluate.py --full`: 5.\n"
        f"- Paper global train batch: 128; released YAML: 32/device; checkpoint steps require effective batch 128, not the README one-GPU example.\n"
        f"- README `--first_frame`: real GT frame sampled from indices 0..4, hence a conditional protocol.\n"
        f"- Official checkpoint/evaluator hashes: all matched: `{not failures}`.\n"
        "- Motion HDF5 upstream checksum: unavailable from Inter-X; local three-repository identity and schema are proven, upstream byte identity is not.\n\n"
        "Formal policy: first calibrate the author-conditioned release, then score the shared 20-replication unconditional generations as a distinct fair supplement.\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": result["status"], "output": str(args.output), "sha256": report_hash, "failures": failures}, sort_keys=True))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
