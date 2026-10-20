#!/usr/bin/env python3
"""Measure GT-to-VQ distance retention; action-group proxy, NOT text MM."""
import argparse
import json
import os
from pathlib import Path
import re
import sys
from types import SimpleNamespace

import h5py
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.score_interact2ar_joint_validation import atomic_json, joint


def selected_pairs(names, lengths, gap):
    actions = []
    for name in names:
        match = re.fullmatch(r"G\d+T\d+(A\d+)R\d+", str(name))
        if match is None:
            raise ValueError(f"unrecognized Inter-X name: {name}")
        actions.append(match.group(1))
    return np.asarray([
        (i, j) for i in range(len(names)) for j in range(i + 1, len(names))
        if actions[i] == actions[j] and abs(int(lengths[i]) - int(lengths[j])) <= gap
    ], dtype=np.int64).reshape(-1, 2)


def compare_distances(gt, reconstructed, pairs):
    if gt.shape != reconstructed.shape or gt.ndim != 2 or not len(pairs):
        raise ValueError("incompatible embeddings or empty pair selection")
    if not np.isfinite(gt).all() or not np.isfinite(reconstructed).all():
        raise ValueError("nonfinite embeddings")
    i, j = pairs.T
    before = np.linalg.norm(gt[i] - gt[j], axis=1)
    after = np.linalg.norm(reconstructed[i] - reconstructed[j], axis=1)
    if before.mean() <= 0:
        raise ValueError("GT has zero mean pair distance")
    correlation = (float(np.corrcoef(before, after)[0, 1])
                   if before.std() > 0 and after.std() > 0 else None)
    return {
        "gt_mean_distance": float(before.mean()),
        "reconstruction_mean_distance": float(after.mean()),
        "ratio_of_mean_distances": float(after.mean() / before.mean()),
        "distance_pearson_correlation": correlation,
        "normalized_distance_rmse": float(np.sqrt(np.mean((after - before) ** 2)) / before.mean()),
        "gt_pair_distances": before.tolist(),
        "reconstruction_pair_distances": after.tolist(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reconstruction-report", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    if args.self_check:
        names = ["G001T001A001R001", "G002T001A001R001", "G003T001A002R001", "G004T001A001R001"]
        pairs = selected_pairs(names, [100, 108, 100, 130], 8)
        assert pairs.tolist() == [[0, 1]]
        x = np.array([[0., 0.], [3., 0.], [0., 4.]])
        pairs = np.array([[0, 1], [0, 2], [1, 2]])
        assert compare_distances(x, x, pairs)["normalized_distance_rmse"] == 0
        assert compare_distances(x, x / 2, pairs)["ratio_of_mean_distances"] == .5
        assert compare_distances(x, np.zeros_like(x), pairs)["ratio_of_mean_distances"] == 0
        print("self-check: PASS", flush=True)
        if args.reconstruction_report is None:
            return
    if args.reconstruction_report is None or args.output is None:
        parser.error("--reconstruction-report and --output are required")
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    report_path = args.reconstruction_report.resolve()
    report = json.loads(report_path.read_text())
    if report["checkpoint_sha256"] != "fdf5afaa5d3513f933254fafc831f6046f7c38b5e649db08fb6be7e0ce1006bb":
        raise ValueError("this diagnostic requires the original VQ175k")
    artifacts = {key: joint.artifact(report[key]) for key in
                 ("requests", "ground_truth", "reconstruction_h5", "checkpoint")}
    for key, artifact in artifacts.items():
        if artifact["sha256"] != report[key + "_sha256"]:
            raise ValueError(f"changed source: {key}")
    requests = np.load(artifacts["requests"]["path"], allow_pickle=False)
    names, lengths = requests["names"].astype(str), requests["lengths"]
    if len(names) != 544 or len(set(names)) != 544:
        raise ValueError("expected 544 distinct frozen validation samples")
    prerequisite = joint.load_prerequisite_audit(Path("output/interact2ar_paper_repo_data_audit.json"))
    calibration = joint.load_author_calibration_audit(
        Path("output/interact2ar_author_release_calibration/audit.json"), prerequisite)
    os.chdir(joint.INTERACT2AR_ROOT)
    wrappers = {name: joint.EvaluatorModelWrapper(
        SimpleNamespace(), args.device, checkpoint, representation="joints", part=part, normalized=False)
        for name, (part, checkpoint) in joint.WRAPPERS.items()}
    embeddings = {}
    with torch.inference_mode():
        for key in ("ground_truth", "reconstruction_h5"):
            with h5py.File(artifacts[key]["path"]) as source:
                joint.validate_h5(source, names, lengths, 1, key)
                if source.attrs.get("requests_sha256") != artifacts["requests"]["sha256"]:
                    raise ValueError(f"request binding mismatch: {key}")
                embeddings[key] = joint.embed_motion_all_wrappers(wrappers, source["motions"], lengths, 32)
            print(f"embedded {key}", flush=True)
    results = {}
    for gap in (8, 150):
        pairs = selected_pairs(names, lengths, gap)
        results[str(gap)] = {
            "maximum_length_gap_frames": gap, "pairs": pairs.tolist(),
            "pair_count": len(pairs),
            "metrics": {part: compare_distances(embeddings["ground_truth"][part],
                        embeddings["reconstruction_h5"][part], pairs) for part in wrappers},
        }
    payload = {
        "schema": "vq_action_group_distance_retention_v1", "status": "complete",
        "diagnostic_only": True,
        "limitations": "Same action ID is not identical text; pair distances are dependent. This is not MM, a quality pass, or proof of semantic preservation.",
        "samples": len(names), "names": names.tolist(),
        "unique_normalized_captions": len({str(c).strip().lower() for c in requests["captions"]}),
        "artifacts": artifacts, "reconstruction_report": joint.artifact(report_path),
        "evaluator_checkpoints": {name: joint.artifact(joint.INTERACT2AR_ROOT / path)
                                  for name, (_, path) in joint.WRAPPERS.items()},
        "prerequisite": prerequisite, "calibration": calibration,
        "runtime": {"device": args.device, "batch_size": 32, "torch_version": torch.__version__},
        "script": joint.artifact(Path(__file__)), "results": results,
    }
    atomic_json(output, payload)
    print(f"saved {output}", flush=True)


if __name__ == "__main__":
    main()
