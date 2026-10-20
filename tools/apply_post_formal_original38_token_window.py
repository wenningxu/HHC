#!/usr/bin/env python3
"""Guardedly restore original MMM's 38-token Inter-X training window.

The completed 50-token baseline remains immutable.  This transformation is
allowed only after its common-50 evaluation and source snapshot are complete.
It changes future diagnostic/shape runs, not any historical manifest.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "output/post_formal_original38_token_window_application.json"
ALIGNMENT = ROOT / "output/hhc_formal_mgt_token_window_alignment_audit.json"
FILES = {
    "options": ROOT / "options/option_transformer.py",
    "train": ROOT / "train/train_t2i_trans.py",
    "paper_audit": ROOT / "tools/audit_hhc_paper_training_contract.py",
}


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def replace_once(source: str, old: str, new: str, label: str) -> str:
    count = source.count(old)
    if count != 1:
        raise ValueError(f"{label}: expected one preimage, found {count}")
    return source.replace(old, new, 1)


def replace_exact_count(
    source: str, old: str, new: str, expected: int, label: str
) -> str:
    count = source.count(old)
    if count != expected:
        raise ValueError(f"{label}: expected {expected} preimages, found {count}")
    return source.replace(old, new)


def transform(sources: dict[str, str]) -> dict[str, str]:
    transformed = dict(sources)
    transformed["options"] = replace_once(
        transformed["options"],
        'choices=["paper_50", "historical_38"],\n'
        '        default="paper_50",\n'
        '        help=(\n'
        '            "MGT diagnostic bridge. The recovered 2026-02-13 token dataset "\n'
        '            "padded/truncated every sequence to 38 tokens despite block_size=51; "\n'
        '            "the current paper/common protocol uses 50."\n'
        '        ),',
        'choices=["original_mmm_38", "paper_50", "historical_38"],\n'
        '        default="original_mmm_38",\n'
        '        help=(\n'
        '            "Inter-X training-window provenance. original_mmm_38 restores "\n'
        '            "the released MMM loader default: 38 tensor positions reserve "\n'
        '            "one END slot and therefore retain 37 motion tokens/148 frames, "\n'
        '            "versus the fair-evaluation cap of 150 frames; paper_50 retains "\n'
        '            "the completed "\n'
        '            "50-window baseline; historical_38 is the legacy replay alias."\n'
        '        ),',
        "token-window CLI contract",
    )
    transformed["train"] = replace_once(
        transformed["train"],
        "and args.training_token_length_contract == 'paper_50'",
        "and args.training_token_length_contract in {\n"
        "            'original_mmm_38', 'paper_50'\n"
        "        }",
        "formal contract accepts disclosed baseline and original window",
    )
    transformed["train"] = replace_exact_count(
        transformed["train"],
        "38 if args.training_token_length_contract == 'historical_38'",
        "38 if args.training_token_length_contract in {\n"
        "                    'original_mmm_38', 'historical_38'\n"
        "                }",
        2,
        "38-token loader branches",
    )

    transformed["paper_audit"] = replace_once(
        transformed["paper_audit"],
        "        'token_length_contract': 'paper_50',\n",
        "",
        "remove obsolete single allowed token window",
    )
    transformed["paper_audit"] = replace_once(
        transformed["paper_audit"],
        "    formal_mismatches = {\n"
        "        key: {'expected': expected, 'active': configured.get(key)}\n"
        "        for key, expected in formal_expected.items()\n"
        "        if configured.get(key) != expected\n"
        "    }\n",
        "    formal_mismatches = {\n"
        "        key: {'expected': expected, 'active': configured.get(key)}\n"
        "        for key, expected in formal_expected.items()\n"
        "        if configured.get(key) != expected\n"
        "    }\n"
        "    allowed_token_windows = {'original_mmm_38', 'paper_50'}\n"
        "    if configured.get('token_length_contract') not in allowed_token_windows:\n"
        "        formal_mismatches['token_length_contract'] = {\n"
        "            'expected_one_of': sorted(allowed_token_windows),\n"
        "            'active': configured.get('token_length_contract'),\n"
        "        }\n",
        "audit disclosed formal token windows",
    )

    return transformed


def atomic_write(path: Path, source: str) -> None:
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(source, encoding="utf-8")
    os.replace(temporary, path)


def artifact(path: Path) -> dict:
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()

    if any(
        Path(path).is_file() is False
        for path in (
            ROOT / "output/fair_eval/hhc_formal_raw12288_repaired_mgt_common50_no_contact/summary.json",
            ROOT / "output/fair_eval/hhc_formal_raw12288_repaired_mgt_common50_contact/summary.json",
            ALIGNMENT,
            ROOT / "output/shape_root_tokenizer_wiring_audit.json",
            ROOT / "output/post_formal_online_wandb_metrics_application.json",
        )
    ):
        raise FileNotFoundError("post-formal original38 prerequisites are incomplete")
    alignment = json.loads(ALIGNMENT.read_text(encoding="utf-8"))
    if alignment.get("status") != "pass":
        raise ValueError("token-window alignment audit failed")
    if subprocess.run(
        ["pgrep", "-f", "python -u train/train_t2i_trans.py"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    ).returncode == 0:
        raise RuntimeError("an MGT process is active")

    before = {key: path.read_text(encoding="utf-8") for key, path in FILES.items()}
    already = (
        'choices=["original_mmm_38", "paper_50", "historical_38"]'
        in before["options"]
        and "'original_mmm_38', 'historical_38'" in before["train"]
    )
    after = before if already else transform(before)
    for key in ("options", "train", "paper_audit"):
        ast.parse(after[key], filename=str(FILES[key]))

    if args.apply and not already:
        for key, path in FILES.items():
            atomic_write(path, after[key])

    payload = {
        "schema": "hhc_post_formal_original38_token_window_application_v1",
        "status": (
            "already_applied" if already else "applied" if args.apply else "check_pass"
        ),
        "write_requested": bool(args.apply),
        "alignment_audit": artifact(ALIGNMENT),
        "files": {
            key: {
                "path": str(FILES[key]),
                "before_sha256": sha256_bytes(before[key].encode()),
                "after_sha256": sha256_bytes(after[key].encode()),
                "changed": before[key] != after[key],
            }
            for key in FILES
        },
        "implementation": artifact(Path(__file__)),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, args.output)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
