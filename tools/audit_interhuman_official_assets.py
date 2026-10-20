#!/usr/bin/env python3
"""Audit the locally installed InterHuman assets and their public provenance.

The official Google Drive folder contains more entries than its anonymous folder
views currently return.  This audit therefore treats the official Drive files as
the authority, uses a public mirror only as a transport for the complete annotation
archive, and compares a deterministic spread of mirror annotations byte-for-byte
against the files that are directly visible from the official Drive folder.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

import requests
from bs4 import BeautifulSoup


OFFICIAL_ROOT_ID = "1oyozJ4E7Sqgsr7Q747Na35tWo5CjNYk3"
OFFICIAL_ANNOTS_ID = "1cYffmLMC3-eturODlM1231YFbgZLDj0W"
OFFICIAL_MOTIONS_ZIP_ID = "1qpFij33tVZhUkk9-kDaDy1FuQb0LJSEf"
OFFICIAL_SPLIT_IDS = {
    "test.txt": "1xRe3m1Pl4UEpgqNgl304yYFXygpTRyHW",
    "train.txt": "1b0Y6uwNK5cyU4Itzavr3FeHiy8VrF-Vj",
    "val.txt": "1I1i1V2fRmYArIM51kXM2OC8dCznvY_1H",
}
MIRROR_REPO = "Hang19/Interhuman"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact(path: Path) -> dict[str, object]:
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def numeric_stems(directory: Path, suffix: str) -> set[int]:
    pattern = re.compile(rf"^(\d+){re.escape(suffix)}$")
    result = set()
    for path in directory.iterdir():
        match = pattern.fullmatch(path.name)
        if path.is_file() and match:
            result.add(int(match.group(1)))
    return result


def read_split(path: Path) -> set[int]:
    return {int(line.strip()) for line in path.read_text().splitlines() if line.strip()}


def official_visible_annotations() -> list[tuple[str, str]]:
    url = f"https://drive.google.com/embeddedfolderview?id={OFFICIAL_ANNOTS_ID}"
    response = requests.get(url, timeout=120)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, features="html.parser")
    result = []
    pattern = re.compile(r"https://drive\.google\.com/file/d/([-\w]{25,})/view")
    for tag in soup.find_all("a"):
        match = pattern.match(str(tag.get("href", "")))
        if match:
            result.append((match.group(1), tag.get_text(strip=True)))
    if not result:
        raise RuntimeError("official embedded folder view returned no annotations")
    return result


def deterministic_spread(items: list[tuple[str, str]], count: int) -> list[tuple[str, str]]:
    if count <= 0:
        return []
    if count >= len(items):
        return items
    indices = {(i * (len(items) - 1)) // (count - 1) for i in range(count)}
    return [items[index] for index in sorted(indices)]


def compare_official_annotation(
    item: tuple[str, str], annotation_dir: Path
) -> dict[str, object]:
    file_id, name = item
    url = (
        "https://drive.usercontent.google.com/download"
        f"?id={file_id}&export=download&confirm=t"
    )
    response = requests.get(url, timeout=120)
    response.raise_for_status()
    official_bytes = response.content
    local_path = annotation_dir / name
    local_bytes = local_path.read_bytes()
    official_sha = hashlib.sha256(official_bytes).hexdigest()
    local_sha = hashlib.sha256(local_bytes).hexdigest()
    return {
        "name": name,
        "official_drive_file_id": file_id,
        "bytes": len(official_bytes),
        "official_sha256": official_sha,
        "local_sha256": local_sha,
        "byte_identical": official_bytes == local_bytes,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root", type=Path, default=Path("dataset/interhuman_official")
    )
    parser.add_argument(
        "--annotation-dir", type=Path,
        default=Path("dataset/interhuman_official/mirror_staging/annots"),
    )
    parser.add_argument("--sample-count", type=int, default=64)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--output", type=Path,
        default=Path("output/interhuman_official_assets_audit.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.data_root.resolve()
    annotation_dir = args.annotation_dir.resolve()
    motion1_dir = root / "motions_processed" / "person1"
    motion2_dir = root / "motions_processed" / "person2"
    split_dir = root / "split"

    motion1_ids = numeric_stems(motion1_dir, ".npy")
    motion2_ids = numeric_stems(motion2_dir, ".npy")
    annotation_ids = numeric_stems(annotation_dir, ".txt")
    split_ids = {name: read_split(split_dir / name) for name in OFFICIAL_SPLIT_IDS}
    split_union = set().union(*split_ids.values())

    visible = official_visible_annotations()
    name_counts = Counter(name for _, name in visible)
    unambiguous = [
        (file_id, name)
        for file_id, name in visible
        if re.fullmatch(r"\d+\.txt", name)
        and name_counts[name] == 1
        and (annotation_dir / name).is_file()
    ]
    unambiguous.sort(key=lambda item: int(Path(item[1]).stem))
    selected = deterministic_spread(unambiguous, args.sample_count)
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        comparisons = list(
            pool.map(lambda item: compare_official_annotation(item, annotation_dir), selected)
        )

    duplicate_annotation_files = sorted(
        path.name for path in annotation_dir.iterdir()
        if path.is_file() and not re.fullmatch(r"\d+\.txt", path.name)
    )
    non_utf8_annotation_ids = []
    for annotation_id in sorted(annotation_ids):
        try:
            (annotation_dir / f"{annotation_id}.txt").read_text(encoding="utf-8")
        except UnicodeDecodeError:
            non_utf8_annotation_ids.append(annotation_id)
    stats_paths = [
        Path("/mnt/8T-HDD/wenning/InterMask/data/stats/global_mean.npy"),
        Path("/mnt/8T-HDD/wenning/InterMask/data/stats/global_std.npy"),
        Path("/mnt/8T-HDD/wenning/coshmdm-code/data/global_mean.npy"),
        Path("/mnt/8T-HDD/wenning/coshmdm-code/data/global_std.npy"),
    ]

    checks = {
        "person_motion_id_sets_equal": motion1_ids == motion2_ids,
        "all_motion_ids_have_annotations": motion1_ids <= annotation_ids,
        "all_motion_ids_are_in_official_split_union": motion1_ids <= split_union,
        "split_partitions_disjoint": not (
            (split_ids["train.txt"] & split_ids["val.txt"])
            or (split_ids["train.txt"] & split_ids["test.txt"])
            or (split_ids["val.txt"] & split_ids["test.txt"])
        ),
        "sampled_official_annotations_byte_identical": bool(comparisons)
        and all(item["byte_identical"] for item in comparisons),
        "intermask_cosh_stats_byte_identical": (
            sha256_file(stats_paths[0]) == sha256_file(stats_paths[2])
            and sha256_file(stats_paths[1]) == sha256_file(stats_paths[3])
        ),
    }

    report = {
        "schema_version": 1,
        "status": "pass" if all(checks.values()) else "fail",
        "authority": {
            "official_repository": "https://github.com/tr3e/InterGen",
            "official_drive_root_id": OFFICIAL_ROOT_ID,
            "official_drive_annotations_folder_id": OFFICIAL_ANNOTS_ID,
            "official_drive_motions_processed_zip_id": OFFICIAL_MOTIONS_ZIP_ID,
            "official_drive_split_file_ids": OFFICIAL_SPLIT_IDS,
            "annotation_transport_mirror": (
                f"https://huggingface.co/datasets/{MIRROR_REPO}"
            ),
            "transport_policy": (
                "mirror accepted only after exact official split comparison and "
                "deterministic byte-level annotation sampling"
            ),
        },
        "artifacts": {
            "license": artifact(root / "LICENSE.md"),
            "motions_processed_zip": artifact(root / "motions_processed.zip"),
            "annotation_transport_zip": artifact(root / "annots_mirror.zip"),
            "split_transport_zip": artifact(root / "split_mirror.zip"),
            "official_split_files": {
                name: artifact(split_dir / name) for name in OFFICIAL_SPLIT_IDS
            },
            "normalization_stats": [artifact(path) for path in stats_paths],
        },
        "inventory": {
            "person1_motion_count": len(motion1_ids),
            "person2_motion_count": len(motion2_ids),
            "canonical_annotation_count": len(annotation_ids),
            "duplicate_annotation_copy_count": len(duplicate_annotation_files),
            "duplicate_annotation_copy_names": duplicate_annotation_files,
            "split_counts": {name: len(ids) for name, ids in split_ids.items()},
            "split_union_count": len(split_union),
            "split_ids_without_processed_motion": sorted(split_union - motion1_ids),
            "annotation_ids_without_processed_motion": sorted(annotation_ids - motion1_ids),
            "canonical_annotation_ids_not_utf8": non_utf8_annotation_ids,
            "loader_policy_for_non_utf8": (
                "preserve official bytes; the unmodified official loader catches the "
                "decode error and excludes that sequence"
            ),
            "official_anonymous_visible_entry_count": len(visible),
            "official_anonymous_unambiguous_entry_count": len(unambiguous),
        },
        "official_annotation_sample": {
            "selection": "numeric-name-sorted deterministic even spread",
            "requested_count": args.sample_count,
            "actual_count": len(comparisons),
            "comparisons": comparisons,
        },
        "checks": checks,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({
        "status": report["status"],
        "output": str(args.output.resolve()),
        "inventory": report["inventory"],
        "checks": checks,
    }, indent=2))
    if report["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
