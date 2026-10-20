#!/usr/bin/env python
"""Recover exact MGT launch sources by reversing later recorded patches.

The training manifest is authoritative.  Live files that still match it are
copied directly; changed files are first looked up in earlier content-addressed
launch snapshots, then reconstructed in a temporary directory by reversing
post-launch ``apply_patch`` records from the local Codex session.  No live
source file is modified.
"""

import argparse
import datetime as dt
import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "hhc_mgt_launch_source_recovery_since_manifest_v1"
OPERATION = re.compile(r"^\*\*\* (Update|Add|Delete) File: (.+)$")
# These three recorded patches add per-sample sampler RNG, refine it, then
# remove it again.  Their only net byte change is the formatting of the
# original gumbel_sample call.  Treating them as independent inverse patches
# fails because the removal patch does not reproduce the add patch's original
# context byte-for-byte.  The exact hashes make this exception fail-closed.
NET_ZERO_SAMPLER_PATCHES = {
    "51c77be4132336fede763ed31e1a3be32887f221f05b8150ce4392a3bbc6075d",
    "30108a2876e69d9d5064c697ba9b9c8ce07bba0d62b4808adec285cd7ef86804",
    "e1decd4f869afef3f3372eb84dc26c24a4061e39754259120c97861f7c16194d",
}
GUMBEL_MULTILINE = (
    "                predictions = gumbel_sample(\n"
    "                    channel_logits, temperature=sample_temperature, dim=-1\n"
    "                )"
)
GUMBEL_LAUNCH_BYTES = (
    "                predictions = gumbel_sample(channel_logits, "
    "temperature=sample_temperature, dim=-1)"
)


def sha256_bytes(value):
    return hashlib.sha256(value).hexdigest()


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact(path):
    path = Path(path).resolve()
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    os.replace(temporary, path)


def timestamp(value):
    return dt.datetime.fromisoformat(value.replace("Z", "+00:00"))


def extract_patch(source):
    marker = "const patch"
    start = source.find(marker)
    if start < 0:
        return None
    equals = source.find("=", start + len(marker))
    if equals < 0:
        return None
    tail = source[equals + 1:].lstrip()
    try:
        value, _ = json.JSONDecoder().raw_decode(tail)
    except json.JSONDecodeError:
        return None
    if not isinstance(value, str) or not value.startswith("*** Begin Patch\n"):
        return None
    return value


def normalized_target(value):
    path = Path(value)
    if not path.is_absolute():
        path = ROOT / path
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return None


def update_sections(patch, wanted):
    lines = patch.splitlines()
    starts = [index for index, line in enumerate(lines) if OPERATION.match(line)]
    selected = []
    for ordinal, start in enumerate(starts):
        match = OPERATION.match(lines[start])
        end = starts[ordinal + 1] if ordinal + 1 < len(starts) else len(lines) - 1
        if match.group(1) != "Update":
            continue
        relative = normalized_target(match.group(2))
        if relative in wanted:
            selected.append((relative, lines[start:end]))
    return selected


def reverse_section(relative, lines, temporary_root):
    rewritten = [f"*** Update File: {(temporary_root / relative).resolve()}"]
    for line in lines[1:]:
        if line.startswith("+") and not line.startswith("+++"):
            line = "-" + line[1:]
        elif line.startswith("-") and not line.startswith("---"):
            line = "+" + line[1:]
        rewritten.append(line)
    return rewritten


def session_patches(session, launched_at, wanted):
    records = []
    with Path(session).open(encoding="utf-8", errors="strict") as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                # The actively written final line may be incomplete.
                continue
            row_time = row.get("timestamp")
            if not row_time or timestamp(row_time) <= launched_at:
                continue
            payload = row.get("payload", {})
            if payload.get("type") != "custom_tool_call" or payload.get("name") != "exec":
                continue
            patch = extract_patch(payload.get("input", ""))
            if patch is None:
                continue
            sections = update_sections(patch, wanted)
            if sections:
                records.append({
                    "line": line_number,
                    "timestamp": row_time,
                    "patch_sha256": sha256_bytes(patch.encode()),
                    "sections": sections,
                })
    return records


def find_existing_snapshot(relative, expected):
    candidates = sorted((ROOT / "output/t2i_trans").glob(
        f"*/launch_sources/{relative}"
    ))
    for candidate in candidates:
        if candidate.is_file() and sha256_file(candidate) == expected:
            return candidate
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-manifest", type=Path, required=True)
    parser.add_argument("--session", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    training_manifest = args.training_manifest.resolve()
    manifest = json.loads(training_manifest.read_text())
    expected = {
        name: digest for name, digest in manifest.get("sha256", {}).items()
        if name.endswith(".py") and "/" in name
    }
    if not expected:
        raise ValueError("training manifest has no bound Python sources")
    launched_at = timestamp(manifest["launch_time"])
    changed = {
        relative for relative, digest in expected.items()
        if not (ROOT / relative).is_file()
        or sha256_file(ROOT / relative) != digest
    }
    records = session_patches(args.session.resolve(), launched_at, changed)
    apply_patch = shutil.which("apply_patch")
    if changed and apply_patch is None:
        raise FileNotFoundError("apply_patch executable is unavailable")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    provenance = {}
    used_records = []
    with tempfile.TemporaryDirectory(prefix="hhc_mgt_source_recovery_") as temp:
        temporary_root = Path(temp)
        for relative in expected:
            source = ROOT / relative
            destination = temporary_root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)

        unresolved = set(changed)
        for relative in sorted(changed):
            existing = find_existing_snapshot(relative, expected[relative])
            if existing is not None:
                shutil.copy2(existing, temporary_root / relative)
                unresolved.remove(relative)
                provenance[relative] = {
                    "method": "validated_existing_launch_snapshot",
                    "source": artifact(existing),
                }

        sampler_records = [
            record for record in records
            if record["patch_sha256"] in NET_ZERO_SAMPLER_PATCHES
        ]
        sampler_hashes = {record["patch_sha256"] for record in sampler_records}
        if (
            "models/two_person_transformer.py" in unresolved
            and sampler_hashes == NET_ZERO_SAMPLER_PATCHES
        ):
            sampler_path = temporary_root / "models/two_person_transformer.py"
            sampler_text = sampler_path.read_text()
            if sampler_text.count(GUMBEL_MULTILINE) != 1:
                raise ValueError(
                    "post-launch sampler cancellation residue is not unique"
                )
            sampler_path.write_text(
                sampler_text.replace(GUMBEL_MULTILINE, GUMBEL_LAUNCH_BYTES)
            )
            provenance["models/two_person_transformer.py"] = {
                "method": "reversed_net_zero_sampler_formatting_residue",
                "patch_hashes": sorted(sampler_hashes),
            }

        for record in reversed(records):
            if record["patch_sha256"] in NET_ZERO_SAMPLER_PATCHES:
                continue
            reverse_lines = ["*** Begin Patch"]
            touched = []
            for relative, section in record["sections"]:
                if relative not in unresolved:
                    continue
                reverse_lines.extend(reverse_section(relative, section, temporary_root))
                touched.append(relative)
            if not touched:
                continue
            reverse_lines.append("*** End Patch")
            completed = subprocess.run(
                [apply_patch], input="\n".join(reverse_lines) + "\n",
                text=True, cwd=ROOT, capture_output=True,
            )
            if completed.returncode != 0:
                raise RuntimeError(
                    f"reverse patch failed at session line {record['line']}: "
                    f"{completed.stdout}\n{completed.stderr}"
                )
            used_records.append({
                "line": record["line"],
                "timestamp": record["timestamp"],
                "patch_sha256": record["patch_sha256"],
                "touched_sources": touched,
            })

        for relative in sorted(unresolved):
            actual = sha256_file(temporary_root / relative)
            if actual != expected[relative]:
                raise ValueError(
                    f"recovered source hash mismatch for {relative}: "
                    f"{actual} != {expected[relative]}"
                )
            provenance[relative] = {
                "method": "reversed_post_launch_codex_patches",
                "session": str(args.session.resolve()),
            }

        sources = {}
        for relative, digest in sorted(expected.items()):
            recovered = temporary_root / relative
            if sha256_file(recovered) != digest:
                raise ValueError(f"source hash mismatch after recovery: {relative}")
            destination = output_dir / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_suffix(destination.suffix + f".tmp.{os.getpid()}")
            shutil.copy2(recovered, temporary)
            os.replace(temporary, destination)
            sources[relative] = artifact(destination)
            provenance.setdefault(relative, {"method": "live_source_unchanged"})

    payload = {
        "schema": SCHEMA,
        "status": "pass",
        "training_manifest": artifact(training_manifest),
        "launch_time": manifest["launch_time"],
        "session": artifact(args.session.resolve()),
        "source_count": len(sources),
        "changed_live_sources": sorted(changed),
        "reverse_patch_records": sorted(used_records, key=lambda row: row["line"]),
        "provenance": provenance,
        "sources": sources,
        "all_snapshot_hashes_match_training_manifest": True,
        "implementation": artifact(Path(__file__)),
    }
    atomic_json(args.output.resolve(), payload)
    print(json.dumps({
        "status": "pass",
        "sources": len(sources),
        "changed": sorted(changed),
        "reversed_patches": len(used_records),
        "output_dir": str(output_dir),
    }, indent=2))


if __name__ == "__main__":
    main()
