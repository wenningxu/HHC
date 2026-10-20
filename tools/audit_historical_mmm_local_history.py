#!/usr/bin/env python
"""Audit the uncommitted MMM sampler recovered from JetBrains Local History.

The recovered ``ContentChange`` files contain the file state immediately before
the timestamp encoded in their filename.  For a W&B run, the authoritative
state is therefore the first such snapshot after the run start, provided the
preceding content-change timestamp is before the run start.  This script binds
those time brackets, source files, W&B metadata/logs, and the original
LocalHistory/VFS stores by SHA-256, then checks the sampler and test-hook
semantics with Python's AST without importing the historical project.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


RUNS = {
    "2026-02-17_033057": {
        "wandb_dir": "run-20260217_033057-vq_mask_gen_4_ablation_adalnshape_1_cfg",
        "model": "historical_mmm_local_history/1771299984324_set-427147_change-427146_ContentChange.py",
        "eval": "historical_mmm_local_history_eval/1772239498876_set-430119_change-430118_ContentChange.py",
        "train": "historical_mmm_local_history_train/1771299688408_set-427052_change-427051_ContentChange.py",
        "expected": {
            "max_steps": 10,
            "temperature": 1.0,
            "rand_pos": True,
            "cfg": -1,
            "contact": False,
            "num_repeat": 1,
            "outer_repeat_time": 10,
            "completed_metric_repeats_before_interrupt": 3,
        },
    },
    "2026-03-05_101848": {
        "wandb_dir": "run-20260305_101848-vq_mask_gen_4_ablation_adalnshape_1_cfg_hbr_wocas",
        "model": "historical_mmm_local_history/1772712417174_set-461757_change-461756_ContentChange.py",
        "eval": "historical_mmm_local_history_eval/1772706694902_set-460986_change-460985_ContentChange.py",
        "train": "historical_mmm_local_history_train/1772706002840_set-460786_change-460785_ContentChange.py",
        "expected": {
            "max_steps": 10,
            "temperature": 0.5,
            "rand_pos": False,
            "cfg": 3.5,
            "contact": False,
            "num_repeat": 30,
            "outer_repeat_time": 1,
            "completed_metric_repeats_before_interrupt": 0,
        },
    },
}


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def iso_from_epoch_ms(value: int) -> str:
    return datetime.fromtimestamp(value / 1000, tz=timezone.utc).isoformat()


def parse_iso(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def snapshot_timestamp(path: Path) -> int:
    match = re.match(r"(\d+)_set-", path.name)
    if not match:
        raise ValueError(f"snapshot timestamp missing from {path}")
    return int(match.group(1))


def snapshot_interval(path: Path) -> dict[str, Any]:
    candidates = sorted(
        (snapshot_timestamp(candidate), candidate)
        for candidate in path.parent.glob("*_ContentChange.py")
    )
    current = snapshot_timestamp(path)
    positions = [index for index, item in enumerate(candidates) if item[1] == path]
    if len(positions) != 1 or positions[0] == 0:
        raise ValueError(f"cannot establish preceding content change for {path}")
    index = positions[0]
    previous_ms, previous_path = candidates[index - 1]
    return {
        "state_valid_from_previous_change_epoch_ms": previous_ms,
        "state_valid_from_previous_change_utc": iso_from_epoch_ms(previous_ms),
        "state_valid_until_snapshot_change_epoch_ms": current,
        "state_valid_until_snapshot_change_utc": iso_from_epoch_ms(current),
        "preceding_snapshot": str(previous_path.relative_to(ROOT)),
    }


def recovery_manifest_record(path: Path) -> dict[str, Any]:
    manifest_path = path.parent / "recovery_manifest.tsv"
    with manifest_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    matches = [row for row in rows if row["output"] == path.name]
    if len(matches) != 1:
        raise ValueError(
            f"expected one recovery-manifest row for {path}, found {len(matches)}"
        )
    row = matches[0]
    verified = (
        int(row["timestamp_ms"]) == snapshot_timestamp(path)
        and int(row["bytes"]) == path.stat().st_size
        and row["sha256"] == sha256_file(path)
        and row["change_type"] == "ContentChange"
        and int(row["content_id"]) > 0
    )
    return {
        "manifest": str(manifest_path.relative_to(ROOT)),
        "manifest_sha256": sha256_file(manifest_path),
        "manifest_rows": len(rows),
        "timestamp_ms": int(row["timestamp_ms"]),
        "set_id": int(row["set_id"]),
        "change_id": int(row["change_id"]),
        "change_type": row["change_type"],
        "vfs_content_id": int(row["content_id"]),
        "bytes": int(row["bytes"]),
        "sha256": row["sha256"],
        "output": row["output"],
        "verified": verified,
    }


def literal(node: ast.AST | None) -> Any:
    if node is None:
        return None
    try:
        return ast.literal_eval(node)
    except (ValueError, TypeError):
        return None


def find_class_method(tree: ast.Module, class_name: str, method_name: str) -> ast.FunctionDef:
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for child in node.body:
                if isinstance(child, ast.FunctionDef) and child.name == method_name:
                    return child
    raise ValueError(f"{class_name}.{method_name} not found")


def function_defaults(node: ast.FunctionDef) -> dict[str, Any]:
    names = [argument.arg for argument in node.args.args]
    values = [literal(value) for value in node.args.defaults]
    return dict(zip(names[-len(values):], values)) if values else {}


def named_assignments(node: ast.AST) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for child in ast.walk(node):
        if isinstance(child, ast.Assign) and len(child.targets) == 1:
            target = child.targets[0]
            if isinstance(target, ast.Name):
                value = literal(child.value)
                if value is not None:
                    result[target.id] = value
    return result


def sampler_record(path: Path, rand_pos: bool, cfg: float) -> dict[str, Any]:
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    method = find_class_method(tree, "Text2Motion_Transformer", "sample_intra_cas")
    defaults = function_defaults(method)
    assignments = named_assignments(method)

    cascade_order = None
    for child in ast.walk(method):
        if not isinstance(child, ast.Assign) or len(child.targets) != 1:
            continue
        if not isinstance(child.targets[0], ast.Name) or child.targets[0].id != "cascade_order":
            continue
        if isinstance(child.value, (ast.List, ast.Tuple)):
            cascade_order = [
                assignments.get(element.id) if isinstance(element, ast.Name) else literal(element)
                for element in child.value.elts
            ]

    outer_loop = any(
        isinstance(child, ast.For)
        and isinstance(child.iter, ast.Call)
        and isinstance(child.iter.func, ast.Name)
        and child.iter.func.id == "range"
        and len(child.iter.args) == 1
        and isinstance(child.iter.args[0], ast.Name)
        and child.iter.args[0].id == "max_steps"
        for child in ast.walk(method)
    )
    cascade_loop = any(
        isinstance(child, ast.For)
        and isinstance(child.iter, ast.Name)
        and child.iter.id == "cascade_order"
        for child in ast.walk(method)
    )

    temperature_by_rand_pos: dict[str, Any] = {}
    for child in ast.walk(method):
        if not isinstance(child, ast.If) or not isinstance(child.test, ast.Name):
            continue
        if child.test.id != "rand_pos":
            continue
        for label, body in (("true", child.body), ("false", child.orelse)):
            for statement in body:
                if not isinstance(statement, ast.Assign):
                    continue
                if any(isinstance(target, ast.Name) and target.id == "temperature" for target in statement.targets):
                    temperature_by_rand_pos[label] = literal(statement.value)

    forward_calls = [
        child for child in ast.walk(method)
        if isinstance(child, ast.Call)
        and isinstance(child.func, ast.Attribute)
        and child.func.attr == "forward_function"
    ]
    max_steps = defaults.get("max_steps")
    channel_updates = max_steps * len(cascade_order or [])
    model_forwards_per_outer = len(cascade_order or []) * (2 if cfg > 0 else 1)
    selected_temperature = temperature_by_rand_pos["true" if rand_pos else "false"]
    return {
        "path": str(path.relative_to(ROOT)),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
        "defaults": defaults,
        "cascade_order_channel_indices": cascade_order,
        "outer_range_max_steps": outer_loop,
        "inner_cascade_loop": cascade_loop,
        "temperature_by_rand_pos": temperature_by_rand_pos,
        "selected_temperature": selected_temperature,
        "channel_updates": channel_updates,
        "model_forward_calls_per_outer_step_for_run_cfg": model_forwards_per_outer,
        "model_forward_calls_for_generation": max_steps * model_forwards_per_outer,
        "forward_function_call_sites_in_method": len(forward_calls),
        "interval": snapshot_interval(path),
        "recovery_manifest": recovery_manifest_record(path),
    }


def call_name(node: ast.Call) -> str | None:
    if isinstance(node.func, ast.Name):
        return node.func.id
    if isinstance(node.func, ast.Attribute):
        return node.func.attr
    return None


def keyword_literals(call: ast.Call) -> dict[str, Any]:
    return {keyword.arg: literal(keyword.value) for keyword in call.keywords if keyword.arg}


def train_record(path: Path) -> dict[str, Any]:
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    hook = find_class_method(tree, "MaskGITLightning", "on_test_epoch_start")
    calls = [
        child for child in ast.walk(hook)
        if isinstance(child, ast.Call) and call_name(child) == "evaluation_transformer"
    ]
    if len(calls) != 1:
        raise ValueError(f"expected one test evaluation call in {path}, found {len(calls)}")
    assignments = named_assignments(hook)
    keywords = keyword_literals(calls[0])
    eval_defaults = {
        "num_repeat": 1,
        "rand_pos": False,
        "CFG": -1,
        "contact": False,
        "shape_change": None,
    }
    effective = {
        name: keywords.get(name, value)
        for name, value in eval_defaults.items()
    }
    trainer_tests = [
        child for child in ast.walk(tree)
        if isinstance(child, ast.Call) and call_name(child) == "test"
    ]
    return {
        "path": str(path.relative_to(ROOT)),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
        "test_hook_keywords": keywords,
        "test_hook_effective_arguments": effective,
        "outer_repeat_time": assignments.get("repeat_time"),
        "trainer_test_call_count": len(trainer_tests),
        "interval": snapshot_interval(path),
        "recovery_manifest": recovery_manifest_record(path),
    }


def validation_train_record(path: Path) -> dict[str, Any]:
    """Recover the exact custom validation hook used by a training run."""
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    hook = find_class_method(
        tree, "MaskGITLightning", "on_validation_epoch_start"
    )
    calls = [
        child for child in ast.walk(hook)
        if isinstance(child, ast.Call)
        and call_name(child) == "evaluation_transformer"
    ]
    if len(calls) != 1:
        raise ValueError(
            f"expected one validation evaluation call in {path}, "
            f"found {len(calls)}"
        )
    call = calls[0]
    literals = keyword_literals(call)
    expressions = {
        keyword.arg: ast.get_source_segment(source, keyword.value)
        for keyword in call.keywords if keyword.arg
    }
    return {
        "path": str(path.relative_to(ROOT)),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
        "validation_hook": "on_validation_epoch_start",
        "evaluation_call_count": len(calls),
        "keyword_literals": literals,
        "keyword_expressions": expressions,
        "rand_pos_omitted": "rand_pos" not in expressions,
        "max_steps_omitted": "max_steps" not in expressions,
        "interval": snapshot_interval(path),
        "recovery_manifest": recovery_manifest_record(path),
    }


def eval_record(path: Path) -> dict[str, Any]:
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    function = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "evaluation_transformer"
    )
    calls = []
    for child in ast.walk(function):
        if not isinstance(child, ast.Call):
            continue
        keywords = keyword_literals(child)
        if keywords.get("type") == "sample_intra_cas":
            calls.append({
                "keywords": [keyword.arg for keyword in child.keywords if keyword.arg],
                "literal_keywords": keywords,
                "explicit_max_steps": any(keyword.arg == "max_steps" for keyword in child.keywords),
            })
    return {
        "path": str(path.relative_to(ROOT)),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
        "function_defaults": function_defaults(function),
        "sample_intra_cas_call_count": len(calls),
        "sample_intra_cas_calls": calls,
        "all_calls_omit_max_steps": bool(calls) and all(not item["explicit_max_steps"] for item in calls),
        "raw_normalizer": "MotionNormalizerTorchInterX('raw')" in source,
        "interval": snapshot_interval(path),
        "recovery_manifest": recovery_manifest_record(path),
    }


def count_completed_fids(log_text: str) -> int:
    return len(re.findall(r"^\s*FID\.\s+[0-9]", log_text, flags=re.MULTILINE))


def store_record(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mmm-repo", type=Path, default=Path("/home/wenning/projects/MMM"))
    parser.add_argument(
        "--ide-system-root", type=Path,
        default=ROOT / "output/historical_mmm_local_history_store_232",
    )
    parser.add_argument("--recovered-root", type=Path, default=ROOT / "output")
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "output/historical_mmm_sampler_local_history_audit.json",
    )
    args = parser.parse_args()

    required_store_files = [
        args.ide_system_root / ".appinfo",
        args.ide_system_root / "LocalHistory/changes.storageData",
        args.ide_system_root / "LocalHistory/changes.storageRecordIndex",
        args.ide_system_root / "caches/records.dat",
        args.ide_system_root / "caches/names.dat",
        args.ide_system_root / "caches/content.dat.storageData",
        args.ide_system_root / "caches/content.dat.storageRecordIndex",
    ]
    missing = [str(path) for path in required_store_files if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing LocalHistory/VFS evidence: {missing}")

    appinfo = (args.ide_system_root / ".appinfo").read_text(encoding="utf-8")
    runs: dict[str, Any] = {}
    checks: list[bool] = []
    for run_name, specification in RUNS.items():
        wandb_files = args.mmm_repo / "wandb" / specification["wandb_dir"] / "files"
        metadata_path = wandb_files / "wandb-metadata.json"
        log_path = wandb_files / "output.log"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        log_text = log_path.read_text(encoding="utf-8", errors="replace")
        start = parse_iso(metadata["startedAt"])

        model = sampler_record(
            args.recovered_root / specification["model"],
            specification["expected"]["rand_pos"],
            specification["expected"]["cfg"],
        )
        evaluation = eval_record(args.recovered_root / specification["eval"])
        train = train_record(args.recovered_root / specification["train"])
        expected = specification["expected"]

        interval_checks = {}
        for name, record in (("model", model), ("eval", evaluation), ("train", train)):
            lower = datetime.fromtimestamp(
                record["interval"]["state_valid_from_previous_change_epoch_ms"] / 1000,
                tz=timezone.utc,
            )
            upper = datetime.fromtimestamp(
                record["interval"]["state_valid_until_snapshot_change_epoch_ms"] / 1000,
                tz=timezone.utc,
            )
            interval_checks[name] = lower <= start < upper

        effective = train["test_hook_effective_arguments"]
        run_checks = {
            "wandb_start_inside_all_source_intervals": all(interval_checks.values()),
            "all_sources_bound_to_vfs_content_ids": all(
                record["recovery_manifest"]["verified"]
                for record in (model, evaluation, train)
            ),
            "upstream_git_base_matches": metadata.get("git", {}).get("commit")
            == "8c3e7ad107ed717b878779724958cf6eff00938e",
            "runtime_traceback_calls_sample_intra_cas": 'type="sample_intra_cas"' in log_text,
            "model_default_outer_steps_match": model["defaults"].get("max_steps") == expected["max_steps"],
            "cascade_order_root_body_hand": model["cascade_order_channel_indices"] == [2, 0, 1],
            "nested_outer_and_channel_loops": model["outer_range_max_steps"] and model["inner_cascade_loop"],
            "thirty_channel_updates": model["channel_updates"] == 30,
            "eval_omits_explicit_max_steps": evaluation["all_calls_omit_max_steps"],
            "eval_uses_raw_normalizer": evaluation["raw_normalizer"],
            "test_rand_pos_matches": effective["rand_pos"] == expected["rand_pos"],
            "test_cfg_matches": effective["CFG"] == expected["cfg"],
            "test_contact_matches": effective["contact"] == expected["contact"],
            "test_num_repeat_matches": effective["num_repeat"] == expected["num_repeat"],
            "outer_repeat_time_matches": train["outer_repeat_time"] == expected["outer_repeat_time"],
            "selected_temperature_matches": model["selected_temperature"] == expected["temperature"],
            "completed_metric_repeats_match": count_completed_fids(log_text)
            == expected["completed_metric_repeats_before_interrupt"],
            "run_was_interrupted": "KeyboardInterrupt" in log_text,
        }
        checks.extend(run_checks.values())
        runs[run_name] = {
            "status": "pass" if all(run_checks.values()) else "fail",
            "wandb": {
                "directory": str(wandb_files.parent),
                "started_at": metadata["startedAt"],
                "metadata_sha256": sha256_file(metadata_path),
                "output_log_sha256": sha256_file(log_path),
                "git": metadata.get("git"),
                "program": metadata.get("program"),
                "args": metadata.get("args"),
                "completed_metric_repeats_before_interrupt": count_completed_fids(log_text),
            },
            "source_interval_checks": interval_checks,
            "model": model,
            "evaluation": evaluation,
            "train": train,
            "effective_runtime": {
                "sampler": "sample_intra_cas",
                "outer_steps": model["defaults"]["max_steps"],
                "channel_updates": model["channel_updates"],
                "cascade_order": "root_body_hand",
                "rand_pos": effective["rand_pos"],
                "temperature": model["selected_temperature"],
                "cfg_scale": effective["CFG"],
                "contact": effective["contact"],
                "num_repeat_inside_eval": effective["num_repeat"],
                "outer_repeat_time": train["outer_repeat_time"],
                "model_forward_calls_per_generation": model["model_forward_calls_for_generation"],
            },
            "checks": run_checks,
        }

    payload = {
        "schema": "historical_mmm_sampler_local_history_audit_v1",
        "status": "pass" if all(checks) else "fail",
        "scope": "read_only_recovery_of_uncommitted_runtime_source_not_git_history",
        "jetbrains": {
            "appinfo": appinfo,
            "expected_build": "PY-232.10335.17",
            "build_matches": "app.build=PY-232.10335.17" in appinfo,
            "evidence_files": [store_record(path) for path in required_store_files],
        },
        "interpretation": {
            "exact_historical_outer_steps": 10,
            "exact_historical_channel_updates": 30,
            "later_pre_cleanup_git_snapshot_outer_steps": 3,
            "later_snapshot_is_not_substituted_for_runtime": True,
            "current_formal_protocol_outer_steps": 50,
            "current_formal_protocol_source": "user_fixed_common50_not_historical_default",
        },
        "runs": runs,
    }
    payload["jetbrains"]["build_matches"] = bool(payload["jetbrains"]["build_matches"])
    if not payload["jetbrains"]["build_matches"]:
        payload["status"] = "fail"

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": payload["status"],
        "output": str(args.output),
        "runs": {name: record["effective_runtime"] for name, record in runs.items()},
    }, indent=2, sort_keys=True))
    if payload["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
