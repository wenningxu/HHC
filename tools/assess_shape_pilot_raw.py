#!/usr/bin/env python
"""Measure random-shape versus optimized raw outputs without projection."""

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.calc_pene_from_npz import PeneCalculator


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--random-dir", required=True, type=Path)
    parser.add_argument("--optimized-dir", required=True, type=Path)
    parser.add_argument("--contact-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    calculator = PeneCalculator(device=args.device)
    rows = []
    sidecars = sorted(
        path for path in args.optimized_dir.glob("*.json")
        if not path.name.startswith("_")
    )
    for sidecar in sidecars:
        metadata = json.loads(sidecar.read_text(encoding="utf-8"))
        name = metadata["augmented_name"]
        source_name = metadata["source_name"]
        with open(args.contact_dir / f"{source_name}.pkl", "rb") as handle:
            contacts = pickle.load(handle)
        paths_by_label = {
            "random_shape_mismatch": [
                args.random_dir / f"{name}_p{person}.npz" for person in (1, 2)
            ],
            "original_mmm_raw_output": [
                args.optimized_dir / f"{name}_p{person}.npz" for person in (1, 2)
            ],
        }
        conditions = {}
        for label, paths in paths_by_label.items():
            conditions[label] = {
                metric: float(calculator.compute_from_npz(
                    paths[0], paths[1], metric=metric,
                    interaction_pairs=(contacts if metric == "mismatch" else None),
                ))
                for metric in ("mismatch", "skate", "pene")
            }
        rows.append({
            "source_name": source_name,
            "augmented_name": name,
            "conditions": conditions,
        })
    if not rows:
        raise ValueError("optimized directory contains no completed samples")

    aggregate = {}
    baseline = "random_shape_mismatch"
    for label in (baseline, "original_mmm_raw_output"):
        aggregate[label] = {}
        for metric in ("mismatch", "skate", "pene"):
            values = np.asarray([
                row["conditions"][label][metric] for row in rows
            ], dtype=np.float64)
            base = np.asarray([
                row["conditions"][baseline][metric] for row in rows
            ], dtype=np.float64)
            delta = values - base
            aggregate[label][metric] = {
                "mean": float(values.mean()),
                "mean_delta_vs_random": float(delta.mean()),
                "improved_samples": int(np.sum(delta < -1e-8)),
                "degraded_samples": int(np.sum(delta > 1e-8)),
                "max_abs_delta_vs_random": float(np.max(np.abs(delta))),
            }
    payload = {
        "schema": "hhc_shape_pilot_raw_assessment_v1",
        "status": "complete",
        "samples": len(rows),
        "aggregate": aggregate,
        "rows": rows,
        "projection_or_postprocessing": "none",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output), "samples": len(rows)}))


if __name__ == "__main__":
    main()
