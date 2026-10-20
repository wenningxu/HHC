#!/usr/bin/env python
"""Score frozen Inter-X generations with Interact2AR's released joint evaluators.

This is a supplementary evaluator space.  It does not replace the official
Inter-X full-672D evaluator used by the main comparison table.
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import torch


LAUNCH_CWD = Path.cwd().resolve()
HHC_ROOT = Path(__file__).resolve().parents[1]
INTERACT2AR_ROOT = Path(
    os.environ.get("INTERACT2AR_ROOT", "/mnt/8T-HDD/wenning/Interact2Ar")
).resolve()
sys.path.insert(0, str(INTERACT2AR_ROOT / "src"))

from data.dataset import _pad_or_trim_tokens, _vectorize_tokens  # noqa: E402
from evaluation.evaluator_wrapper import EvaluatorModelWrapper  # noqa: E402
from evaluation.metrics import (  # noqa: E402
    calculate_activation_statistics,
    calculate_diversity,
    calculate_frechet_distance,
    calculate_multimodality,
    calculate_top_k,
    euclidean_distance_matrix,
)
from utils.word_vectorizer import WordVectorizer  # noqa: E402


WRAPPERS = {
    "JointsFull": ("full", "checkpoints-evaluator/joints_full/model/finest.tar"),
    "JointsBody": ("body", "checkpoints-evaluator/joints_body/model/finest.tar"),
    "JointsHands": ("hands", "checkpoints-evaluator/joints_hands/model/finest.tar"),
}


def resolve_cli_path(value):
    path = Path(value)
    return path.resolve() if path.is_absolute() else (LAUNCH_CWD / path).resolve()


def sha256_file(path, chunk_size=8 * 1024 * 1024):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact(path):
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def repository_revision():
    return subprocess.check_output(
        ["git", "-C", str(INTERACT2AR_ROOT), "rev-parse", "HEAD"], text=True
    ).strip()


def serializable_attributes(source):
    result = {}
    for key, value in source.attrs.items():
        if isinstance(value, bytes):
            value = value.decode("utf-8")
        elif isinstance(value, np.generic):
            value = value.item()
        result[key] = value
    return result


def require_finite(dataset, label, chunk_samples=8):
    for start in range(0, dataset.shape[0], chunk_samples):
        values = np.asarray(dataset[start:start + chunk_samples])
        if not np.isfinite(values).all():
            raise FloatingPointError(f"{label} contains NaN or Inf near sample {start}")


def validate_h5(source, names, lengths, repeats, label):
    if source["motions"].shape != (len(names), repeats, 150, 672):
        raise ValueError(f"{label} motion shape is {source['motions'].shape}")
    actual_names = source["names"][:].astype(str)
    if not np.array_equal(actual_names, names):
        raise ValueError(f"{label} names/order differ from frozen request")
    if not np.array_equal(source["lengths"][:], lengths):
        raise ValueError(f"{label} lengths differ from frozen request")
    if "completed_repeats" in source:
        completed = source["completed_repeats"][:].astype(bool)
        if completed.shape != (repeats,) or not completed.all():
            raise ValueError(f"{label} is incomplete: {completed.tolist()}")
    require_finite(source["motions"], label)


def effective_lengths(source, requested):
    if "generated_lengths" not in source:
        return requested
    generated = source["generated_lengths"][:].astype(np.int64)
    if generated.shape != requested.shape:
        raise ValueError("generated_lengths shape mismatch")
    if np.any(generated <= 0) or np.any(generated > requested):
        raise ValueError("generated_lengths must be in (0, requested_length]")
    return generated


def validate_prior_score(method_dir, replication, request_path, main_path, mm_path):
    score_path = method_dir / f"replication_{replication:02d}.json"
    score = json.loads(score_path.read_text(encoding="utf-8"))
    checks = {
        "request": (score.get("requests_sha256"), sha256_file(request_path)),
        "main": (score.get("main_generations_sha256"), sha256_file(main_path)),
        "mm": (score.get("mm_generations_sha256"), sha256_file(mm_path)),
    }
    failures = [name for name, pair in checks.items() if pair[0] != pair[1]]
    if score.get("status") != "complete" or failures:
        raise ValueError(
            f"prior official Inter-X score validation failed for {method_dir.name}/"
            f"replication {replication}: {failures}"
        )
    return artifact(score_path)


def caption_tokens(text_dir, name, caption):
    matches = []
    with open(text_dir / f"{name}.txt", "r", encoding="utf-8") as handle:
        for line in handle:
            fields = line.rstrip("\n").split("#")
            if len(fields) != 4:
                raise ValueError(f"malformed processed text row for {name}")
            if fields[0] == caption:
                matches.append(tuple(fields[1].split(" ")))
    unique = set(matches)
    if len(unique) != 1:
        raise ValueError(
            f"caption/token lookup for {name} has {len(matches)} matches and "
            f"{len(unique)} unique token sequences"
        )
    return list(next(iter(unique)))


def vectorize_captions(names, captions, text_dir, vectorizer):
    words, pos, sent_lens = [], [], []
    for name, caption in zip(names, captions):
        tokens = caption_tokens(text_dir, name, caption)
        tokens, sent_len = _pad_or_trim_tokens(tokens, 35)
        word, pos_one_hot = _vectorize_tokens(tokens, vectorizer)
        words.append(word)
        pos.append(pos_one_hot)
        sent_lens.append(sent_len)
    return (
        np.asarray(words, dtype=np.float32),
        np.asarray(pos, dtype=np.float32),
        np.asarray(sent_lens, dtype=np.int64),
    )


@torch.inference_mode()
def embed_text(wrapper, words, pos, sent_lens, batch_size):
    result = []
    for start in range(0, len(sent_lens), batch_size):
        end = min(start + batch_size, len(sent_lens))
        lengths = sent_lens[start:end]
        order = np.argsort(lengths)[::-1].copy()
        inverse = np.empty_like(order)
        inverse[order] = np.arange(len(order))
        embedding = wrapper.text_encoder(
            torch.from_numpy(words[start:end][order]).to(wrapper.device),
            torch.from_numpy(pos[start:end][order]).to(wrapper.device),
            torch.from_numpy(lengths[order]).long(),
        )
        result.append(embedding[inverse].cpu().numpy())
    return np.concatenate(result)


@torch.inference_mode()
def embed_motion(wrapper, motion_dataset, lengths, batch_size, repeat=0):
    result = []
    for start in range(0, len(lengths), batch_size):
        end = min(start + batch_size, len(lengths))
        batch_lengths = np.asarray(lengths[start:end], dtype=np.int64)
        motion = np.asarray(motion_dataset[start:end, repeat], dtype=np.float32)
        embedding = wrapper.get_motion_embeddings(
            torch.from_numpy(motion), torch.from_numpy(batch_lengths).long()
        )
        order = np.argsort(batch_lengths)[::-1].copy()
        inverse = np.empty_like(order)
        inverse[order] = np.arange(len(order))
        result.append(embedding[inverse].cpu().numpy())
    output = np.concatenate(result)
    if not np.isfinite(output).all():
        raise FloatingPointError("joint evaluator produced non-finite motion embeddings")
    return output


@torch.inference_mode()
def embed_motion_all_wrappers(wrappers, motion_dataset, lengths, batch_size,
                              repeats=1):
    """Recover neutral global joints once, then run every released encoder."""
    if repeats <= 0:
        raise ValueError("repeats must be positive")
    if not wrappers:
        raise ValueError("at least one joint evaluator is required")
    reference = next(iter(wrappers.values()))
    for name, wrapper in wrappers.items():
        if (wrapper.representation != "joints" or wrapper.normalized
                or wrapper.device != reference.device):
            raise ValueError(f"{name}: incompatible shared-joint evaluator")
    collected = {
        name: [[] for _ in range(repeats)] for name in wrappers
    }
    for start in range(0, len(lengths), batch_size):
        end = min(start + batch_size, len(lengths))
        batch_lengths = np.asarray(lengths[start:end], dtype=np.int64)
        order = np.argsort(batch_lengths)[::-1].copy()
        inverse = np.empty_like(order)
        inverse[order] = np.arange(len(order))
        sorted_lengths = torch.from_numpy(batch_lengths[order]).long()
        for repeat in range(repeats):
            raw = np.asarray(
                motion_dataset[start:end, repeat], dtype=np.float32
            )
            joints = reference.get_joints_from_motion(
                torch.from_numpy(raw).to(reference.device)
            )
            for name, wrapper in wrappers.items():
                if wrapper.part == "full":
                    features = joints.reshape(len(joints), joints.shape[1], -1)
                elif wrapper.part == "body":
                    features = joints[:, :, :22, :].reshape(
                        len(joints), joints.shape[1], -1
                    )
                elif wrapper.part == "hands":
                    features = joints[:, :, -30:, :].reshape(
                        len(joints), joints.shape[1], -1
                    )
                else:
                    raise ValueError(f"{name}: unsupported evaluator part")
                movements = wrapper.movement_encoder(features[order]).detach()
                embeddings = wrapper.motion_encoder(
                    movements, sorted_lengths // wrapper.opt.unit_length
                )
                collected[name][repeat].append(
                    embeddings[inverse].cpu().numpy()
                )
    result = {}
    for name, by_repeat in collected.items():
        arrays = [np.concatenate(chunks) for chunks in by_repeat]
        value = arrays[0] if repeats == 1 else np.stack(arrays, axis=1)
        if not np.isfinite(value).all():
            raise FloatingPointError(
                f"{name}: shared-joint evaluator produced NaN or Inf"
            )
        result[name] = value
    return result


def batched_retrieval(text, motion, batch_size):
    if len(text) % batch_size:
        raise ValueError("Interact2AR full protocol requires complete batches")
    matching = 0.0
    top_k = np.zeros(3, dtype=np.float64)
    for start in range(0, len(text), batch_size):
        distance = euclidean_distance_matrix(
            text[start:start + batch_size], motion[start:start + batch_size]
        )
        matching += float(np.trace(distance))
        indices = np.argsort(distance, axis=1)
        top_k += calculate_top_k(indices, 3).sum(axis=0)
    return matching / len(text), top_k / len(text)


def score_embeddings(text, ground_truth, generated, multimodality, batch_size,
                     metric_seed):
    gt_mu, gt_cov = calculate_activation_statistics(ground_truth)
    pred_mu, pred_cov = calculate_activation_statistics(generated)
    matching, r_precision = batched_retrieval(text, generated, batch_size)
    # Reset per method so every method receives the same sampled pair indices.
    np.random.seed(metric_seed)
    diversity_gt = float(calculate_diversity(ground_truth, 300))
    diversity = float(calculate_diversity(generated, 300))
    mm = float(calculate_multimodality(multimodality, 10))
    return {
        "fid": float(calculate_frechet_distance(gt_mu, gt_cov, pred_mu, pred_cov)),
        "matching": float(matching),
        "top1": float(r_precision[0]),
        "top2": float(r_precision[1]),
        "top3": float(r_precision[2]),
        "diversity_gt": diversity_gt,
        "diversity": diversity,
        "multimodality": mm,
    }


def artifact_matches(record, path):
    """Return whether a recorded artifact still identifies the exact file."""
    path = Path(path).resolve()
    return (
        isinstance(record, dict)
        and record.get("path") == str(path)
        and path.is_file()
        and record.get("bytes") == path.stat().st_size
        and record.get("sha256") == sha256_file(path)
    )


def parse_method(value):
    if "=" not in value:
        raise argparse.ArgumentTypeError("method must be LABEL=METHOD_DIRECTORY")
    label, path = value.split("=", 1)
    if not label or not path:
        raise argparse.ArgumentTypeError("method must be LABEL=METHOD_DIRECTORY")
    return label, resolve_cli_path(path)


def load_prerequisite_audit(path):
    path = resolve_cli_path(path)
    record = json.loads(path.read_text(encoding="utf-8"))
    if (
        record.get("schema") != "interact2ar_paper_repo_data_audit_v1"
        or not str(record.get("status", "")).startswith("pass")
        or record.get("paper_contract", {}).get("replications") != 20
        or record.get("released_code_contract", {}).get(
            "first_frame_is_ground_truth_random_index_0_to_unit_length"
        ) is not True
    ):
        raise ValueError("Interact2AR paper/repository/data prerequisite audit did not pass")
    return artifact(path)


def load_author_calibration_audit(path, paper_repository_data_artifact):
    """Require the untouched author protocol to pass before fair scoring."""
    path = resolve_cli_path(path)
    record = json.loads(path.read_text(encoding="utf-8"))
    if (
        record.get("schema")
        != "interact2ar_author_release_calibration_audit_v1"
        or record.get("status") != "pass"
        or not all(record.get("checks", {}).values())
        or record.get("current_paper_repository_data_audit")
        != paper_repository_data_artifact
    ):
        raise ValueError(
            "Interact2AR author-release calibration prerequisite did not pass "
            "against the current paper/repository/data audit"
        )
    return artifact(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", action="append", required=True, type=parse_method)
    parser.add_argument("--requests-dir", default=str(HHC_ROOT / "output/fair_interx_requests"))
    parser.add_argument(
        "--ground-truth-dir",
        default=str(HHC_ROOT / "output/interact2ar_joint_evaluator_ground_truth"),
    )
    parser.add_argument(
        "--output-root", default=str(HHC_ROOT / "output/interact2ar_joint_evaluator_suite")
    )
    parser.add_argument("--replications", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--prerequisite-audit",
        default=str(HHC_ROOT / "output/interact2ar_paper_repo_data_audit.json"),
    )
    parser.add_argument(
        "--author-calibration-audit",
        default=str(
            HHC_ROOT / "output/interact2ar_author_release_calibration/audit.json"
        ),
    )
    args = parser.parse_args()
    requests_dir = resolve_cli_path(args.requests_dir)
    ground_truth_dir = resolve_cli_path(args.ground_truth_dir)
    output_root = resolve_cli_path(args.output_root)
    if args.replications != 20:
        raise ValueError("paper-aligned fair protocol requires exactly 20 replications")
    if args.batch_size != 32:
        raise ValueError("formal Interact2AR full protocol requires batch size 32")

    prerequisite_audit = load_prerequisite_audit(args.prerequisite_audit)
    author_calibration_audit = load_author_calibration_audit(
        args.author_calibration_audit, prerequisite_audit
    )
    revision = repository_revision()
    text_dir = INTERACT2AR_ROOT / "data/texts_processed"
    vectorizer = WordVectorizer(str(INTERACT2AR_ROOT / "data/glove"), "hhi_vab")
    os.chdir(INTERACT2AR_ROOT)
    wrappers = {}
    for wrapper_name, (part, checkpoint_rel) in WRAPPERS.items():
        wrappers[wrapper_name] = EvaluatorModelWrapper(
            SimpleNamespace(), args.device, checkpoint_rel,
            representation="joints", part=part, normalized=False,
        )

    provenance = {
        "contract": "interact2ar_released_joint_evaluator_suite_v1",
        "repository": str(INTERACT2AR_ROOT),
        "repository_revision": revision,
        "representation": "joints",
        "normalized": False,
        "parts": list(WRAPPERS),
        "wrapper_source": artifact(INTERACT2AR_ROOT / "src/evaluation/evaluator_wrapper.py"),
        "metrics_source": artifact(INTERACT2AR_ROOT / "src/evaluation/metrics.py"),
        "dataset_source": artifact(INTERACT2AR_ROOT / "src/data/dataset.py"),
        "word_vectorizer_source": artifact(INTERACT2AR_ROOT / "src/utils/word_vectorizer.py"),
        "scorer_source": artifact(__file__),
        "body_model": artifact(INTERACT2AR_ROOT / "data/body_models/smplx/SMPLX_NEUTRAL.npz"),
        "glove_data": artifact(INTERACT2AR_ROOT / "data/glove/hhi_vab_data.npy"),
        "checkpoints": {
            name: artifact(INTERACT2AR_ROOT / path) for name, (_, path) in WRAPPERS.items()
        },
        "paper_aligned_fair_replications": 20,
        "released_author_script_replications": 5,
        "samples_per_replication": 1696,
        "batch_size": 32,
        "diversity_pairs": 300,
        "mm_samples": 100,
        "mm_repeats": 30,
        "mm_pairs": 10,
        "generation_conditioning": "unchanged frozen unconditional main-table generations",
        "neutral_smplx_joint_recovery": (
            "computed once per raw-motion batch and shared across JointsFull/Body/Hands; "
            "released evaluator encoders and part slices remain independent"
        ),
        "metric_rng": "paired across methods; seed=request_seed+1000*wrapper_index",
        "separate_from_main_table": True,
        "prerequisite_paper_repository_data_audit": prerequisite_audit,
        "prerequisite_author_release_calibration_audit": author_calibration_audit,
    }

    for replication in range(args.replications):
        request_path = requests_dir / f"replication_{replication:02d}.npz"
        gt_path = ground_truth_dir / f"replication_{replication:02d}.h5"
        request = np.load(request_path, allow_pickle=False)
        names = request["names"].astype(str)
        captions = request["captions"].astype(str)
        lengths = request["lengths"].astype(np.int64)
        mm_indices = request["mm_indices"].astype(np.int64)
        pending_methods = []
        for method_label, method_dir in args.method:
            main_path = method_dir / f"replication_{replication:02d}_main.h5"
            mm_path = method_dir / f"replication_{replication:02d}_mm.h5"
            prior_score = validate_prior_score(
                method_dir, replication, request_path, main_path, mm_path
            )
            target_dir = output_root / method_label
            target = target_dir / f"replication_{replication:02d}.json"
            if target.is_file():
                existing = json.loads(target.read_text(encoding="utf-8"))
                expected_artifacts = {
                    "requests": request_path,
                    "raw_ground_truth": gt_path,
                    "main_generations": main_path,
                    "mm_generations": mm_path,
                    "prior_official_interx_score": Path(prior_score["path"]),
                }
                if (
                    existing.get("schema")
                    == "interact2ar_joint_evaluator_replication_v1"
                    and existing.get("status") == "complete"
                    and existing.get("method") == method_label
                    and existing.get("replication") == replication
                    and existing.get("evaluator_provenance") == provenance
                    and all(
                        artifact_matches(existing.get(label), path)
                        for label, path in expected_artifacts.items()
                    )
                ):
                    print(json.dumps({
                        "status": "resume_skip",
                        "replication": replication,
                        "method": method_label,
                        "output": str(target),
                    }, sort_keys=True), flush=True)
                    continue
                raise RuntimeError(
                    f"existing joint-evaluator output has another contract: {target}"
                )
            pending_methods.append((
                method_label, main_path, mm_path, prior_score,
                target_dir, target,
            ))
        if not pending_methods:
            continue
        words, pos, sent_lens = vectorize_captions(names, captions, text_dir, vectorizer)

        with h5py.File(gt_path, "r") as gt_source:
            validate_h5(gt_source, names, lengths, 1, "raw ground truth")
            if gt_source.attrs.get("requests_sha256") != sha256_file(request_path):
                raise ValueError("raw GT replay is not bound to this frozen request")
            gt_embeddings = embed_motion_all_wrappers(
                wrappers, gt_source["motions"], lengths, args.batch_size
            )

        text_embeddings = {
            wrapper_name: embed_text(wrapper, words, pos, sent_lens, args.batch_size)
            for wrapper_name, wrapper in wrappers.items()
        }

        for (method_label, main_path, mm_path, prior_score,
             target_dir, target) in pending_methods:
            metrics = {}
            with h5py.File(main_path, "r") as main_source, h5py.File(mm_path, "r") as mm_source:
                validate_h5(main_source, names, lengths, 1, f"{method_label} main")
                mm_names = names[mm_indices]
                mm_lengths = lengths[mm_indices]
                validate_h5(mm_source, mm_names, mm_lengths, 30, f"{method_label} MM")
                main_lengths = effective_lengths(main_source, lengths)
                effective_mm_lengths = effective_lengths(mm_source, mm_lengths)
                if not np.array_equal(effective_mm_lengths, main_lengths[mm_indices]):
                    raise ValueError("main/MM effective length policies differ")
                generated_embeddings = embed_motion_all_wrappers(
                    wrappers, main_source["motions"], main_lengths,
                    args.batch_size,
                )
                repeated_embeddings = embed_motion_all_wrappers(
                    wrappers, mm_source["motions"], effective_mm_lengths,
                    args.batch_size, repeats=30,
                )
                for wrapper_index, (wrapper_name, wrapper) in enumerate(wrappers.items()):
                    metrics[wrapper_name] = score_embeddings(
                        text_embeddings[wrapper_name], gt_embeddings[wrapper_name],
                        generated_embeddings[wrapper_name],
                        repeated_embeddings[wrapper_name], args.batch_size,
                        int(request["seed"]) + 1000 * wrapper_index,
                    )
                    print(json.dumps({
                        "replication": replication,
                        "method": method_label,
                        "wrapper": wrapper_name,
                        **metrics[wrapper_name],
                    }, sort_keys=True), flush=True)

                result = {
                    "schema": "interact2ar_joint_evaluator_replication_v1",
                    "status": "complete",
                    "method": method_label,
                    "replication": replication,
                    "seed": int(request["seed"]),
                    "samples": len(names),
                    "mm_samples": len(mm_indices),
                    "mm_repeats": 30,
                    "requests": artifact(request_path),
                    "raw_ground_truth": artifact(gt_path),
                    "main_generations": artifact(main_path),
                    "mm_generations": artifact(mm_path),
                    "prior_official_interx_score": prior_score,
                    "main_generation_metadata": serializable_attributes(main_source),
                    "mm_generation_metadata": serializable_attributes(mm_source),
                    "effective_length_policy": (
                        "generated_lengths" if not np.array_equal(main_lengths, lengths)
                        else "requested_lengths"
                    ),
                    "metrics": metrics,
                    "evaluator_provenance": provenance,
                }
            target_dir.mkdir(parents=True, exist_ok=True)
            temporary = target.with_suffix(target.suffix + f".tmp.{os.getpid()}")
            with open(temporary, "w", encoding="utf-8") as handle:
                json.dump(result, handle, indent=2, sort_keys=True)
            os.replace(temporary, target)


if __name__ == "__main__":
    main()
