#!/usr/bin/env python
"""Audit unchanged dependencies used by the recovered historical MGT path."""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MMM = Path("/home/wenning/projects/MMM")
RUN_START = datetime.fromisoformat("2026-02-13T01:55:22.272525+00:00")
RECOVERED_PARAMUTIL = ROOT / (
    "output/historical_mmm_local_history_paramutil/"
    "1773176137050_set-468943_change-468942_ContentChange.py"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_record(path: Path) -> dict:
    modified = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "modified_utc": modified.isoformat(),
        "modified_before_run_start": modified < RUN_START,
    }


def top_level_node(path: Path, name: str, node_type) -> ast.AST:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return next(
        node for node in tree.body
        if isinstance(node, node_type) and node.name == name
    )


def class_method(path: Path, class_name: str, method_name: str) -> ast.AST:
    class_node = top_level_node(path, class_name, ast.ClassDef)
    return next(
        node for node in class_node.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == method_name
    )


def ast_equal(left: ast.AST, right: ast.AST) -> bool:
    return ast.dump(left, include_attributes=False) == ast.dump(
        right, include_attributes=False
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "output/historical_mgt_dependency_contract_audit.json",
    )
    parser.add_argument(
        "--local-history-audit", type=Path,
        default=ROOT / "output/historical_mmm_sampler_local_history_audit.json",
    )
    args = parser.parse_args()

    local_history_audit = json.loads(
        args.local_history_audit.read_text(encoding="utf-8")
    )
    store_evidence = local_history_audit.get("jetbrains", {}).get(
        "evidence_files", []
    )
    store_evidence_matches = bool(store_evidence) and all(
        Path(record["path"]).is_file()
        and Path(record["path"]).stat().st_size == record["bytes"]
        and sha256_file(Path(record["path"])) == record["sha256"]
        for record in store_evidence
    )

    byte_identical_pairs = {
        relative: (MMM / relative, ROOT / relative)
        for relative in (
            "models/encdec.py",
            "models/pos_encoding.py",
            "models/trans_encdec.py",
            "models/quantize_cnn.py",
            "utils/fk.py",
        )
    }
    missing = [
        str(path) for pair in byte_identical_pairs.values() for path in pair
        if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError(f"missing dependency sources: {missing}")

    motion_historical = MMM / "models/motion_transformer.py"
    motion_current = ROOT / "models/motion_transformer.py"
    motion_classes = (
        "Attention", "Block", "AdaBlock", "CrossAttention",
        "Block_crossatt", "AdaLN",
    )
    motion_checks = {
        name: ast_equal(
            top_level_node(motion_historical, name, ast.ClassDef),
            top_level_node(motion_current, name, ast.ClassDef),
        )
        for name in motion_classes
    }

    runtime_historical = MMM / "exit/utils.py"
    runtime_current = ROOT / "utils/runtime.py"
    runtime_functions = (
        "generate_src_mask", "uniform", "cosine_schedule", "log",
        "gumbel_noise", "gumbel_sample", "top_k", "top_p",
    )
    runtime_checks = {
        name: ast_equal(
            top_level_node(runtime_historical, name, ast.FunctionDef),
            top_level_node(runtime_current, name, ast.FunctionDef),
        )
        for name in runtime_functions
    }

    dataset_historical = MMM / "dataset/dataset_interx_TM.py"
    dataset_current = ROOT / "dataset/dataset_interx_TM.py"
    dataset_getitem_equal = ast_equal(
        class_method(
            dataset_historical, "Text2MotionDatasetV2HHI_Token", "__getitem__"
        ),
        class_method(
            dataset_current, "Text2MotionDatasetV2HHI_Token", "__getitem__"
        ),
    )

    paramutil_current = ROOT / "utils/paramUtil.py"
    paramutil_manifest = RECOVERED_PARAMUTIL.parent / "recovery_manifest.tsv"
    with paramutil_manifest.open(encoding="utf-8", newline="") as handle:
        manifest_rows = list(csv.DictReader(handle, delimiter="\t"))
    paramutil_row = next(
        row for row in manifest_rows
        if row["output"] == RECOVERED_PARAMUTIL.name
    )
    recovered_paramutil_text = RECOVERED_PARAMUTIL.read_text(encoding="utf-8")
    current_paramutil_text = paramutil_current.read_text(encoding="utf-8")
    used_tail_marker = "SMPL_MODEL_PATH ="
    recovered_used_tail = recovered_paramutil_text[
        recovered_paramutil_text.index(used_tail_marker):
    ]
    current_used_tail = current_paramutil_text[
        current_paramutil_text.index(used_tail_marker):
    ]
    first_paramutil_change = min(
        int(row["timestamp_ms"]) for row in manifest_rows
    )

    byte_checks = {
        relative: sha256_file(historical) == sha256_file(current)
        for relative, (historical, current) in byte_identical_pairs.items()
    }
    historical_sources = {
        str(path.relative_to(MMM)): source_record(path)
        for path in {
            *(pair[0] for pair in byte_identical_pairs.values()),
            motion_historical,
            runtime_historical,
            dataset_historical,
        }
    }
    checks = {
        "all_byte_identical_dependencies": all(byte_checks.values()),
        "all_used_motion_transformer_classes_ast_equal": all(
            motion_checks.values()
        ),
        "all_used_runtime_functions_ast_equal": all(runtime_checks.values()),
        "token_dataset_getitem_ast_equal": dataset_getitem_equal,
        "recovered_paramutil_used_skeleton_tail_equal": (
            recovered_used_tail == current_used_tail
        ),
        "recovered_paramutil_vfs_binding_matches": (
            int(paramutil_row["bytes"]) == RECOVERED_PARAMUTIL.stat().st_size
            and paramutil_row["sha256"] == sha256_file(RECOVERED_PARAMUTIL)
            and int(paramutil_row["content_id"]) > 0
        ),
        "first_recorded_paramutil_change_is_after_run": (
            datetime.fromtimestamp(
                first_paramutil_change / 1000, tz=timezone.utc
            ) > RUN_START
        ),
        "frozen_local_history_store_audit_passes": (
            local_history_audit.get("status") == "pass"
            and store_evidence_matches
        ),
        "all_historical_dependencies_predate_run": all(
            record["modified_before_run_start"]
            for record in historical_sources.values()
        ),
    }
    payload = {
        "schema": "hhc_historical_mgt_dependency_contract_audit_v1",
        "status": "pass" if all(checks.values()) else "mismatch",
        "run_start_utc": RUN_START.isoformat(),
        "checks": checks,
        "byte_identity_checks": byte_checks,
        "motion_transformer_class_ast_checks": motion_checks,
        "runtime_function_ast_checks": runtime_checks,
        "token_dataset_getitem_ast_equal": dataset_getitem_equal,
        "recovered_paramutil": {
            **source_record(RECOVERED_PARAMUTIL),
            "recovery_manifest": str(paramutil_manifest),
            "vfs_content_id": int(paramutil_row["content_id"]),
            "first_recorded_change_utc": datetime.fromtimestamp(
                first_paramutil_change / 1000, tz=timezone.utc
            ).isoformat(),
            "used_skeleton_tail_sha256": hashlib.sha256(
                recovered_used_tail.encode("utf-8")
            ).hexdigest(),
        },
        "frozen_local_history_store_audit": {
            "path": str(args.local_history_audit),
            "sha256": sha256_file(args.local_history_audit),
            "status": local_history_audit.get("status"),
            "evidence_file_count": len(store_evidence),
            "evidence_files_match": store_evidence_matches,
        },
        "historical_sources": historical_sources,
        "current_sources": {
            str(path.relative_to(ROOT)): source_record(path)
            for path in {
                *(pair[1] for pair in byte_identical_pairs.values()),
                motion_current,
                runtime_current,
                dataset_current,
                paramutil_current,
            }
        },
        "body_controlled_fid_used": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "output": str(args.output),
        "status": payload["status"],
        "failed_checks": [name for name, passed in checks.items() if not passed],
    }, indent=2))
    if payload["status"] != "pass":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
