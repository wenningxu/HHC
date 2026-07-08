import argparse
import csv
import math
from pathlib import Path

from calc_pene_from_npz import PeneCalculator


SHAPE_DIRS = {
    "shape_0": ("final_demo/shape_0", None),
    "shape_1": ("final_demo/shape_1", None),
    "shape_4": ("final_demo/shape_4", "_G4"),
    "shape_6": ("final_demo/shape_6", "_G6"),
}


def parse_args():
    parser = argparse.ArgumentParser(description="Rank final_demo sample ids by cumulative pene and mismatch.")
    parser.add_argument("--device", default="cuda:0", help="Torch device")
    parser.add_argument("--contact-dir", type=Path, default=Path("dataset/interx/contact_pairs"))
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--out-csv", type=Path, default=Path("final_demo/final_demo_metric_ranking.csv"))
    return parser.parse_args()


def sample_id_from_filename(path: Path) -> str:
    name = path.name
    if not name.endswith("_pred_p1.npz"):
        raise ValueError(f"Unexpected file name: {name}")
    stem = name[: -len("_pred_p1.npz")]
    if stem.endswith("_G4") or stem.endswith("_G5") or stem.endswith("_G6"):
        stem = stem[:-3]
    return stem


def build_shape_maps():
    shape_maps = {}
    for shape_name, (folder_str, _) in SHAPE_DIRS.items():
        folder = Path(folder_str)
        current = {}
        for p1_path in sorted(folder.glob("*_pred_p1.npz")):
            sample_id = sample_id_from_filename(p1_path)
            p2_path = Path(str(p1_path).replace("_pred_p1.npz", "_pred_p2.npz"))
            if p2_path.exists():
                current[sample_id] = (p1_path, p2_path)
        shape_maps[shape_name] = current
    return shape_maps


def min_max_normalize(values):
    min_val = min(values)
    max_val = max(values)
    if math.isclose(min_val, max_val):
        return [0.0 for _ in values]
    scale = max_val - min_val
    return [(v - min_val) / scale for v in values]


def main():
    args = parse_args()
    shape_maps = build_shape_maps()
    sample_sets = [set(m.keys()) for m in shape_maps.values()]
    common_ids = sorted(set.intersection(*sample_sets))
    if not common_ids:
        raise RuntimeError("No common sample ids found across shape_0/1/4/6")

    calculator = PeneCalculator(device=args.device)
    rows = []
    no_contact_ids = []

    for idx, sample_id in enumerate(common_ids, start=1):
        interaction_pairs = calculator.load_interaction_pairs(args.contact_dir / f"{sample_id}.pkl")
        if not interaction_pairs:
            no_contact_ids.append(sample_id)
            continue
        row = {"sample_id": sample_id}

        for shape_name, mapping in shape_maps.items():
            p1_path, p2_path = mapping[sample_id]
            pene = calculator.compute_from_npz(p1_path, p2_path, metric="pene").item()
            mismatch = calculator.compute_from_npz(
                p1_path,
                p2_path,
                metric="mismatch",
                interaction_pairs=interaction_pairs,
            ).item()
            row[f"pene_{shape_name}"] = pene
            row[f"mismatch_{shape_name}"] = mismatch

        row["cum_pene"] = sum(row[f"pene_{shape_name}"] for shape_name in SHAPE_DIRS)
        row["cum_mismatch"] = sum(row[f"mismatch_{shape_name}"] for shape_name in SHAPE_DIRS)
        rows.append(row)

        if idx % 25 == 0 or idx == len(common_ids):
            print(f"progress {idx}/{len(common_ids)}", flush=True)

    pene_values = [row["cum_pene"] for row in rows]
    mismatch_values = [row["cum_mismatch"] for row in rows]
    norm_pene = min_max_normalize(pene_values)
    norm_mismatch = min_max_normalize(mismatch_values)

    for row, p_score, m_score in zip(rows, norm_pene, norm_mismatch):
        row["norm_pene"] = p_score
        row["norm_mismatch"] = m_score
        row["final_score"] = p_score + m_score

    rows.sort(key=lambda x: (x["final_score"], x["cum_pene"], x["cum_mismatch"], x["sample_id"]))

    fieldnames = [
        "rank",
        "sample_id",
        "cum_pene",
        "cum_mismatch",
        "norm_pene",
        "norm_mismatch",
        "final_score",
        "pene_shape_0",
        "pene_shape_1",
        "pene_shape_4",
        "pene_shape_6",
        "mismatch_shape_0",
        "mismatch_shape_1",
        "mismatch_shape_4",
        "mismatch_shape_6",
    ]

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for rank, row in enumerate(rows, start=1):
            out_row = {
                "rank": rank,
                "sample_id": row["sample_id"],
                "cum_pene": row["cum_pene"],
                "cum_mismatch": row["cum_mismatch"],
                "norm_pene": row["norm_pene"],
                "norm_mismatch": row["norm_mismatch"],
                "final_score": row["final_score"],
                "pene_shape_0": row["pene_shape_0"],
                "pene_shape_1": row["pene_shape_1"],
                "pene_shape_4": row["pene_shape_4"],
                "pene_shape_6": row["pene_shape_6"],
                "mismatch_shape_0": row["mismatch_shape_0"],
                "mismatch_shape_1": row["mismatch_shape_1"],
                "mismatch_shape_4": row["mismatch_shape_4"],
                "mismatch_shape_6": row["mismatch_shape_6"],
            }
            writer.writerow(out_row)

    print("SUMMARY")
    print(f"num_common_ids: {len(common_ids)}")
    print(f"num_no_contact_ids: {len(no_contact_ids)}")
    print(f"num_ranked_ids: {len(rows)}")
    print(f"output_csv: {args.out_csv}")
    if no_contact_ids:
        print(f"no_contact_preview: {no_contact_ids[:10]}")
    print("TOP_K")
    for rank, row in enumerate(rows[: args.top_k], start=1):
        print(
            f"{rank}\t{row['sample_id']}\t"
            f"cum_pene={row['cum_pene']:.8f}\t"
            f"cum_mismatch={row['cum_mismatch']:.8f}\t"
            f"norm_pene={row['norm_pene']:.8f}\t"
            f"norm_mismatch={row['norm_mismatch']:.8f}\t"
            f"final_score={row['final_score']:.8f}"
        )


if __name__ == "__main__":
    main()
