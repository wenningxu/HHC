#!/usr/bin/env python
"""Audit the public TiMotion+Transformer+LPA fair Inter-X training contract."""

import argparse
import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np


EXPECTED_REVISION = "172580fd7f2551a7cc10fde1f15ff73315742e65"
EXPECTED_UPSTREAM_README_SHA256 = (
    "48377e5b2031acd18ba8bd30b97a6cbcc692bc3514b960ad18e2bee2421a22eb"
)
EXPECTED_UPSTREAM_DATASET_CONFIG_SHA256 = (
    "ff441a5d6c577196f1a5c0975c9dd32e1acd00fb2cac1be8fc575c21db56a4e7"
)
EXPECTED_UPSTREAM_EVALUATOR_DOWNLOAD_SHA256 = (
    "fe36d172784a55f37a418ea6896515d12694a51339439ef360f92367201a1df0"
)
EXPECTED_SPLITS = {
    "train": (9110, "47c5c9a09d7293e90e127fb49ba9dc9398afbff66c8f2b45ada3ef6c4453157a"),
    "val": (570, "6c77e85017b14b512f42f2d540da2ce05874d9777d707bebd85782d6221f28a8"),
    "test": (1708, "db32bbe12651875c41113cd20214afcf59ff9489a65aac6f3b0a5b1ecc2b075b"),
}
EXPECTED_MEAN = "aa281b878b8a71c6d10c001059b29374921c18545c082102bf69a404982f568d"
EXPECTED_STD = "30a83e92965a465301a3642eb55b2ba705d3dceaba05e45846aa433d3900b104"
REQUIRED_SOURCE_ARTIFACTS = {
    "model_config", "dataset_config", "train_config", "config_loader",
    "runner", "dataset_loader", "dataset_implementation",
    "dataset_preprocess", "model", "model_exports", "denoiser",
    "interaction_block", "interaction_layers", "lpa_resnet",
    "classifier_free_sampler", "scheduler_and_model_utils", "losses",
    "diffusion", "normalizer_source", "quaternion_source",
    "rotation_conversion_source", "dataset_conversion_source",
    "dataset_conversion_skeleton_source", "official_interx_all_motion",
    "processed_person1", "processed_person2", "processed_annotations",
    "normalization_mean", "normalization_std",
    "normalization_recompute_manifest", "clip_vit_l_14_336",
    "train_split", "val_split", "test_split",
}


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(payload):
    return hashlib.sha256(payload).hexdigest()


def git_blob(repo_root, revision, path):
    return subprocess.check_output(
        ["git", "show", f"{revision}:{path}"], cwd=repo_root
    )


def audit_released_checkpoint_compatibility(repo_root):
    readme = git_blob(repo_root, EXPECTED_REVISION, "README.md")
    dataset_config = git_blob(
        repo_root, EXPECTED_REVISION, "configs/datasets.yaml"
    )
    evaluator_download = git_blob(
        repo_root, EXPECTED_REVISION,
        "prepare/download_evaluation_model.sh",
    )
    require(
        sha256_bytes(readme) == EXPECTED_UPSTREAM_README_SHA256,
        "public TiMotion README drifted",
    )
    require(
        sha256_bytes(dataset_config)
        == EXPECTED_UPSTREAM_DATASET_CONFIG_SHA256,
        "public TiMotion dataset config drifted",
    )
    require(
        sha256_bytes(evaluator_download)
        == EXPECTED_UPSTREAM_EVALUATOR_DOWNLOAD_SHA256,
        "public TiMotion evaluator download script drifted",
    )
    readme_text = readme.decode("utf-8")
    dataset_text = dataset_config.decode("utf-8")
    evaluator_text = evaluator_download.decode("utf-8")
    require(
        "https://tr3e.github.io/intergen-page/" in readme_text,
        "public TiMotion README no longer points to InterHuman data",
    )
    require(
        readme_text.count("https://drive.google.com/file/d/") == 2,
        "public TiMotion pretrained-link inventory changed",
    )
    require(
        "./data/InterHuman" in dataset_text
        and "interx" not in dataset_text.lower(),
        "public TiMotion dataset config is not InterHuman-only",
    )
    require(
        "# InterHuman" in evaluator_text,
        "public evaluator download is no longer documented as InterHuman",
    )
    tracked_paths = subprocess.check_output(
        ["git", "ls-tree", "-r", "--name-only", EXPECTED_REVISION],
        cwd=repo_root, text=True,
    ).splitlines()
    interx_paths = [
        path for path in tracked_paths
        if "interx" in path.lower() or "inter-x" in path.lower()
    ]
    checkpoint_paths = [
        path for path in tracked_paths
        if Path(path).suffix.lower() in {".ckpt", ".pth", ".pt", ".tar"}
    ]
    require(not interx_paths, "public TiMotion revision contains Inter-X assets")
    require(not checkpoint_paths, "public TiMotion Git tree embeds checkpoints")
    return {
        "public_revision": EXPECTED_REVISION,
        "upstream_readme_sha256": sha256_bytes(readme),
        "upstream_dataset_config_sha256": sha256_bytes(dataset_config),
        "upstream_evaluator_download_sha256": sha256_bytes(
            evaluator_download
        ),
        "readme_data_source": "InterHuman via InterGen project page",
        "upstream_dataset_config_names": [
            "interhuman", "interhuman_val", "interhuman_test"
        ],
        "upstream_evaluator_download_label": "InterHuman",
        "documented_pretrained_model_links": 2,
        "tracked_interx_paths": interx_paths,
        "tracked_checkpoint_paths": checkpoint_paths,
        "documented_interx_checkpoint_available": False,
        "fairness_decision": (
            "public checkpoints are not documented for Inter-X; train the "
            "public Transformer+LPA architecture on the frozen official "
            "Inter-X train split rather than relabel an InterHuman checkpoint"
        ),
    }


def tree_digest(path, pattern):
    path = Path(path)
    files = sorted(item for item in path.rglob(pattern) if item.is_file())
    digest = hashlib.sha256()
    total_bytes = 0
    for item in files:
        relative = item.relative_to(path).as_posix()
        size = item.stat().st_size
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(size).encode("ascii"))
        digest.update(b"\0")
        digest.update(sha256_file(item).encode("ascii"))
        digest.update(b"\n")
        total_bytes += size
    return {
        "files": len(files),
        "bytes": total_bytes,
        "content_inventory_sha256": digest.hexdigest(),
    }


def require(condition, message):
    if not condition:
        raise ValueError(message)


def atomic_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def read_split(path):
    names = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    require(all(names), f"blank ID in {path}")
    require(len(names) == len(set(names)), f"duplicate ID in {path}")
    return names


def validate_artifacts(manifest, rehash_trees):
    artifacts = manifest.get("artifacts", {})
    require(REQUIRED_SOURCE_ARTIFACTS <= set(artifacts),
            "TiMotion source/data artifact closure is incomplete")
    checked = {}
    for label, record in sorted(artifacts.items()):
        path = Path(record["path"])
        require(path.exists(), f"missing TiMotion artifact {label}: {path}")
        if "sha256" in record:
            observed_size = path.stat().st_size
            observed_sha = sha256_file(path)
            require(observed_size == record.get("bytes"),
                    f"TiMotion artifact size drift: {label}")
            require(observed_sha == record["sha256"],
                    f"TiMotion artifact hash drift: {label}")
            checked[label] = {
                "path": str(path), "bytes": observed_size,
                "sha256": observed_sha, "content_rehashed": True,
            }
        else:
            require(path.is_dir(), f"TiMotion tree artifact is not a directory: {label}")
            observed = None
            if rehash_trees:
                observed = tree_digest(path, record["file_pattern"])
                for key in ("files", "bytes", "content_inventory_sha256"):
                    require(observed[key] == record.get(key),
                            f"TiMotion tree content drift: {label}/{key}")
            checked[label] = {
                "path": str(path),
                "files": record.get("files"),
                "bytes": record.get("bytes"),
                "content_inventory_sha256": record.get(
                    "content_inventory_sha256"
                ),
                "content_rehashed": rehash_trees,
            }
    return checked


def validate_dataset(manifest):
    artifacts = manifest["artifacts"]
    splits = {}
    split_sets = {}
    for split, (expected_count, expected_sha) in EXPECTED_SPLITS.items():
        path = Path(artifacts[f"{split}_split"]["path"])
        names = read_split(path)
        require(len(names) == expected_count, f"wrong TiMotion {split} split count")
        require(sha256_file(path) == expected_sha, f"wrong TiMotion {split} split hash")
        splits[split] = {
            "path": str(path), "count": len(names), "sha256": expected_sha,
        }
        split_sets[split] = set(names)
    require(not (split_sets["train"] & split_sets["val"]), "train/val overlap")
    require(not (split_sets["train"] & split_sets["test"]), "train/test overlap")
    require(not (split_sets["val"] & split_sets["test"]), "val/test overlap")
    all_names = set().union(*split_sets.values())
    require(len(all_names) == 11388, "official Inter-X split union is not 11,388")

    data_root = Path(artifacts["processed_person1"]["path"]).parents[1]
    lengths = []
    short_by_split = {split: [] for split in EXPECTED_SPLITS}
    split_for_name = {
        name: split for split, names in split_sets.items() for name in names
    }
    for index, name in enumerate(sorted(all_names)):
        person1 = data_root / "motions_processed/person1" / f"{name}.npy"
        person2 = data_root / "motions_processed/person2" / f"{name}.npy"
        annotation = data_root / "annots" / f"{name}.txt"
        require(person1.is_file() and person2.is_file() and annotation.is_file(),
                f"incomplete converted TiMotion sample: {name}")
        motion1 = np.load(person1, mmap_mode="r")
        motion2 = np.load(person2, mmap_mode="r")
        require(motion1.shape == motion2.shape, f"person shape mismatch: {name}")
        require(len(motion1.shape) == 2 and motion1.shape[1] == 312,
                f"wrong converted motion representation: {name}/{motion1.shape}")
        if motion1.shape[0] < 15:
            short_by_split[split_for_name[name]].append(name)
        require(annotation.read_text(encoding="utf-8").strip(),
                f"empty TiMotion annotation: {name}")
        lengths.append(int(motion1.shape[0]))

    all_h5_path = Path(artifacts["official_interx_all_motion"]["path"])
    with h5py.File(all_h5_path, "r") as source:
        source_names = set(source.keys())
    require(source_names == all_names,
            "converted TiMotion IDs differ from official MMM all.h5 IDs")
    require(not short_by_split["train"],
            "official TiMotion train split contains motions below loader minimum")

    mean_path = Path(artifacts["normalization_mean"]["path"])
    std_path = Path(artifacts["normalization_std"]["path"])
    require(sha256_file(mean_path) == EXPECTED_MEAN, "wrong TiMotion train mean")
    require(sha256_file(std_path) == EXPECTED_STD, "wrong TiMotion train std")
    mean, std = np.load(mean_path), np.load(std_path)
    require(mean.shape == (262,) and std.shape == (262,),
            "wrong TiMotion normalization shape")
    require(np.isfinite(mean).all() and np.isfinite(std).all() and (std > 0).all(),
            "non-finite/non-positive TiMotion normalization")
    stats_manifest = json.loads(Path(
        artifacts["normalization_recompute_manifest"]["path"]
    ).read_text(encoding="utf-8"))
    require(
        stats_manifest.get("processed_samples") == 9110
        and stats_manifest.get("requested_samples") == 9110
        and stats_manifest.get("configurations_per_sample") == 4
        and stats_manifest.get("statistics_scope")
        == "official_interx_train_with_mirror_and_person_order"
        and stats_manifest.get("mean_sha256") == EXPECTED_MEAN
        and stats_manifest.get("std_sha256") == EXPECTED_STD,
        "TiMotion normalization recompute proof is invalid",
    )
    return {
        "splits": splits,
        "union_count": len(all_names),
        "all_h5_exact_id_match": True,
        "converted_pair_count": len(lengths),
        "minimum_frames": min(lengths),
        "maximum_frames": max(lengths),
        "below_15_frames_by_split": {
            split: {"count": len(names), "names": names}
            for split, names in short_by_split.items()
        },
        "all_train_ids_pass_loader_minimum": True,
        "motion_file_dim": 312,
        "model_feature_dim": 262,
        "normalization_mean_sha256": EXPECTED_MEAN,
        "normalization_std_sha256": EXPECTED_STD,
        "normalization_train_only_recompute": True,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", type=Path,
        default=Path(
            "/mnt/8T-HDD/wenning/TIMotion/checkpoints/"
            "TIMotion-Transformer-LPA-InterX-contract-preflight/"
            "meta/training_manifest.json"
        ),
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("output/timotion_interx_training_contract_audit.json"),
    )
    parser.add_argument("--rehash-trees", action="store_true")
    args = parser.parse_args()

    manifest_path = args.manifest.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    require(manifest.get("repository", {}).get("revision") == EXPECTED_REVISION,
            "wrong public TiMotion revision")
    require(manifest.get("status") in {"declared", "running", "complete", "resuming"},
            "invalid TiMotion manifest status")
    require(manifest.get("architecture") == {
        "layers": 5, "heads": 16, "latent_dim": 512, "dropout": 0.1,
        "lpa": True, "conv_layers": 1, "dilation_rate": 1,
        "normalization": "AdaLN",
    }, "TiMotion architecture differs from public Transformer+LPA")
    paper = manifest.get("paper_protocol", {})
    require(
        paper.get("epochs") == 1500
        and paper.get("optimizer") == "AdamW"
        and paper.get("betas") == [0.9, 0.999]
        and paper.get("learning_rate") == 1e-4
        and paper.get("weight_decay") == 2e-5
        and paper.get("warmup_epochs") == 10
        and paper.get("scheduler_horizon_epochs") == 2000
        and paper.get("diffusion_steps") == 1000
        and paper.get("classifier_free_dropout") == 0.1
        and paper.get("classifier_free_guidance") == 3.5
        and paper.get("sampling") == "DDIM50 eta=0"
        and "1500 epochs" in paper.get("paper_training_budget_source", ""),
        "TiMotion paper/released training protocol drift",
    )
    batch = manifest.get("batch_emulation", {})
    require(
        batch.get("paper_global_batch") == 256
        and batch.get("local_microbatch") == 32
        and batch.get("gradient_accumulation") == 8
        and batch.get("microbatches_per_epoch") == 568
        and batch.get("optimizer_updates_per_epoch") == 71
        and batch.get("expected_final_optimizer_updates") == 106500
        and batch.get("examples_per_epoch") == 18176
        and batch.get("gradient_clip_norm") == 0.5,
        "TiMotion global-batch emulation contract drift",
    )
    require(manifest.get("official_split_counts") == {
        "train": 9110, "val": 570, "test": 1708,
    }, "TiMotion official split counts drift")
    require(manifest.get("processed_data_counts") == {
        "person1": 11388, "person2": 11388, "annotations": 11388,
    }, "TiMotion processed data counts drift")
    require(manifest.get("loader_train_entries_after_mirroring") == 18220,
            "TiMotion mirrored train entry count drift")
    require(manifest.get("ignore_list_present") is False,
            "TiMotion loader unexpectedly excludes an ignore list")

    checked_artifacts = validate_artifacts(manifest, args.rehash_trees)
    dataset = validate_dataset(manifest)
    runner_text = Path(manifest["artifacts"]["runner"]["path"]).read_text(
        encoding="utf-8"
    )
    for token in (
        "self.manual_backward(loss / self.accumulate)",
        "torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)",
        "batch_idx >= self.microbatches_per_epoch",
        "self.micro_step = int(self.global_step) * self.accumulate",
        "expected_final_optimizer_updates",
        "refusing TiMotion resume because code/data/training contract drifted",
    ):
        require(token in runner_text, f"missing TiMotion runner invariant: {token}")
    backward_index = runner_text.index(
        "self.manual_backward(loss / self.accumulate)"
    )
    telemetry_index = runner_text.index(
        "telemetry = parameter_and_gradient_telemetry(self.model)"
    )
    clip_index = runner_text.index(
        "torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)"
    )
    step_index = runner_text.index("optimizer.step()", clip_index)
    require(
        backward_index < telemetry_index < clip_index < step_index,
        "TiMotion gradient telemetry/clip/optimizer ordering drifted",
    )

    repo_root = Path(manifest["repository"]["path"])
    current_revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo_root, text=True
    ).strip()
    require(current_revision == EXPECTED_REVISION, "TiMotion repository moved")
    released_checkpoint_compatibility = audit_released_checkpoint_compatibility(
        repo_root
    )
    payload = {
        "schema": "hhc_timotion_interx_training_contract_v1",
        "status": "pass",
        "audited_at_utc": datetime.now(timezone.utc).isoformat(),
        "manifest": {
            "path": str(manifest_path),
            "bytes": manifest_path.stat().st_size,
            "sha256": sha256_file(manifest_path),
            "status": manifest["status"],
        },
        "public_repository_revision": EXPECTED_REVISION,
        "reported_method": "public TiMotion Transformer+LPA (not unreleased RWKV/Mamba)",
        "paper_protocol": {
            "source": "https://arxiv.org/html/2408.17135v5",
            "version": "v5",
            "revised": "2026-03-10",
            "interx_table": 7,
            "evaluation_replications": 20,
            "confidence_interval": "95%",
            "diffusion_training_steps": 1000,
            "sampler": "DDIM",
            "sampling_steps": 50,
            "ddim_eta": 0.0,
            "classifier_free_guidance": 3.5,
            "interx_fid": {
                "Transformer": 0.385,
                "Mamba": 0.348,
                "RWKV": 0.261,
            },
            "release_scope_consequence": (
                "the frozen public revision exposes the runnable Transformer "
                "interaction block but not runnable Mamba/RWKV blocks or an "
                "Inter-X checkpoint; the local fair result is therefore the "
                "public Transformer+LPA Inter-X port, not the paper-best RWKV row"
            ),
        },
        "released_checkpoint_compatibility": released_checkpoint_compatibility,
        "paper_budget": {
            "epochs": 1500,
            "global_batch": 256,
            "optimizer_updates": 106500,
            "sample_exposures": 27264000,
            "source": "arXiv:2408.17135v5 section 4.1.1",
            "public_readme_default_1900_overridden_by_paper": True,
        },
        "dataset": dataset,
        "artifacts": checked_artifacts,
        "tree_contents_independently_rehashed": args.rehash_trees,
        "resume_contract": {
            "checkpoint_must_be_epoch_boundary": True,
            "micro_step_derived_from_global_step": True,
            "code_data_contract_drift_rejected": True,
        },
        "gradient_update_contract": {
            "accumulated_loss_divisor": 8,
            "telemetry_is_pre_clip": True,
            "gradient_clip_global_norm": 0.5,
            "clip_precedes_optimizer_step": True,
            "matches_released_tools_train_py": True,
        },
    }
    atomic_json(args.output.resolve(), payload)
    print(json.dumps({
        "output": str(args.output.resolve()),
        "status": "pass",
        "optimizer_updates": 106500,
        "converted_pairs": dataset["converted_pair_count"],
        "rehash_trees": args.rehash_trees,
    }, indent=2))


if __name__ == "__main__":
    main()
