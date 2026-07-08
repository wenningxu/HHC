import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from calc_pene_from_npz import PeneCalculator
from models.evaluator_wrapper import EvaluatorModelWrapper
from options.get_eval_option import get_opt
from utils.eval_trans import calculate_R_precision
from utils.word_vectorizer import WordVectorizer


def parse_args():
    parser = argparse.ArgumentParser(
        description="Filter noisy motions by pene/mismatch, then recompute multimodality on remaining embeddings."
    )
    parser.add_argument("run_dir", type=Path, help="Run directory containing manifest.jsonl, motions/, embeddings/")
    parser.add_argument("--device", default="cuda:0", help="Torch device, e.g. cuda:0 or cpu")
    parser.add_argument("--pene-thresh", type=float, default=0.5, help="Drop repeats with mean_pene_per_frame > threshold")
    parser.add_argument("--mismatch-thresh", type=float, default=None, help="Drop repeats with mean_mismatch_per_frame > threshold")
    parser.add_argument("--contact-dir", type=Path, default=Path("dataset/interx/contact_pairs"), help="Directory containing sample_name.pkl contact files")
    parser.add_argument("--text-dir", type=Path, default=Path("dataset/interx/texts_processed"), help="Directory containing processed text files")
    parser.add_argument("--eval-opt-path", type=Path, default=Path("checkpoints/hhi/Comp_v6_KLD01/opt.txt"), help="Evaluator opt path")
    parser.add_argument("--glove-dir", type=Path, default=Path("dataset/interx/glove"), help="GloVe directory for text vectorizer")
    parser.add_argument("--vocab-prefix", default="hhi_vab", help="Word vectorizer vocab prefix")
    parser.add_argument("--top-k", type=int, default=3, help="Top-k for R-precision")
    parser.add_argument("--multimodality-times", type=int, default=10, help="Number of random pairs per sample")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducible multimodality sampling")
    return parser.parse_args()


def sample_multimodality(embeddings, multimodality_times, rng):
    num_repeats = embeddings.shape[0]
    if num_repeats <= multimodality_times:
        return None

    first_indices = rng.choice(num_repeats, multimodality_times, replace=False)
    second_indices = rng.choice(num_repeats, multimodality_times, replace=False)
    dist = np.linalg.norm(embeddings[first_indices] - embeddings[second_indices], axis=1)
    return float(dist.mean())


def load_records_from_manifest(run_dir):
    manifest_path = run_dir / "manifest.jsonl"
    if not manifest_path.exists():
        return None

    records = []
    with manifest_path.open("r") as f:
        for line in f:
            records.append(json.loads(line))
    return records


def load_records_from_flat_dir(run_dir):
    pattern = re.compile(r"^(?P<sample>.+)_repeat(?P<repeat>\d+)_(?P<kind>p1|p2|emb)\.(?P<ext>npz|npy)$")
    grouped = defaultdict(dict)

    for path in run_dir.iterdir():
        if not path.is_file():
            continue
        match = pattern.match(path.name)
        if match is None:
            continue

        sample_name = match.group("sample")
        repeat_id = int(match.group("repeat"))
        kind = match.group("kind")
        grouped[(sample_name, repeat_id)][kind] = path.name

    records = []
    for (sample_name, repeat_id), items in sorted(grouped.items()):
        if {"p1", "p2", "emb"}.issubset(items.keys()):
            records.append(
                {
                    "sample_name": sample_name,
                    "repeat_id": repeat_id,
                    "motion_p1_path": items["p1"],
                    "motion_p2_path": items["p2"],
                    "embedding_path": items["emb"],
                }
            )
    return records


def load_text_entries(text_dir, sample_name):
    text_path = text_dir / f"{sample_name}.txt"
    if not text_path.exists():
        return []

    entries = []
    with text_path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("#")
            if len(parts) < 2:
                continue
            entries.append({"caption": parts[0], "tokens": parts[1].split(" ")})
    return entries


def resolve_tokens_for_record(record, text_dir, entry_cache):
    sample_name = record["sample_name"]
    if sample_name not in entry_cache:
        entry_cache[sample_name] = load_text_entries(text_dir, sample_name)

    entries = entry_cache[sample_name]
    if not entries:
        return None

    record_text = record.get("text")
    if record_text:
        for entry in entries:
            if entry["caption"] == record_text:
                return entry["tokens"]

    return entries[0]["tokens"]


def build_text_features(tokens, w_vectorizer, max_text_len=35):
    if len(tokens) < max_text_len:
        tokens = ["sos/OTHER"] + tokens + ["eos/OTHER"]
        sent_len = len(tokens)
        tokens = tokens + ["unk/OTHER"] * (max_text_len + 2 - sent_len)
    else:
        tokens = tokens[:max_text_len]
        tokens = ["sos/OTHER"] + tokens + ["eos/OTHER"]
        sent_len = len(tokens)

    pos_one_hots = []
    word_embeddings = []
    for token in tokens:
        try:
            word_emb, pos_oh = w_vectorizer[token]
        except Exception:
            word_emb, pos_oh = w_vectorizer["unk/OTHER"]
        pos_one_hots.append(pos_oh[None, :])
        word_embeddings.append(word_emb[None, :])

    pos_one_hots = np.concatenate(pos_one_hots, axis=0)
    word_embeddings = np.concatenate(word_embeddings, axis=0)
    return word_embeddings, pos_one_hots, sent_len


def encode_text_embeddings(records, text_dir, eval_wrapper, w_vectorizer):
    entry_cache = {}
    missing_text_samples = []
    word_embeddings = []
    pos_one_hots = []
    sent_lens = []

    for record in records:
        tokens = resolve_tokens_for_record(record, text_dir, entry_cache)
        if tokens is None:
            missing_text_samples.append(record["sample_name"])
            continue
        word_emb, pos_oh, sent_len = build_text_features(tokens, w_vectorizer)
        word_embeddings.append(word_emb)
        pos_one_hots.append(pos_oh)
        sent_lens.append(sent_len)

    if missing_text_samples:
        raise RuntimeError(
            "Missing processed text entries for samples: "
            f"{sorted(set(missing_text_samples))[:10]}"
        )

    word_embeddings = torch.from_numpy(np.stack(word_embeddings, axis=0)).float().to(eval_wrapper.device)
    pos_one_hots = torch.from_numpy(np.stack(pos_one_hots, axis=0)).float().to(eval_wrapper.device)
    sent_lens = torch.tensor(sent_lens, dtype=torch.long, device=eval_wrapper.device)
    sorted_lens, sort_idx = torch.sort(sent_lens, descending=True)
    word_embeddings = word_embeddings[sort_idx]
    pos_one_hots = pos_one_hots[sort_idx]

    with torch.no_grad():
        text_embeddings = eval_wrapper.text_encoder(word_embeddings, pos_one_hots, sorted_lens)

    inv_sort_idx = torch.empty_like(sort_idx)
    inv_sort_idx[sort_idx] = torch.arange(sort_idx.shape[0], device=sort_idx.device)
    text_embeddings = text_embeddings[inv_sort_idx]
    return text_embeddings.detach().cpu().numpy()


def evaluate_subset(name, records, args, eval_wrapper, w_vectorizer):
    if len(records) < args.top_k:
        return {
            "scenario": name,
            "num_records": len(records),
            "error": f"Only {len(records)} repeats remain, which is smaller than top_k={args.top_k}.",
        }

    kept_embeddings = defaultdict(list)
    for record in records:
        kept_embeddings[record["sample_name"]].append(record["embedding"])

    rng = np.random.default_rng(args.seed)
    sample_mmodality = {}
    insufficient_samples = {}
    for sample_name, embeddings in kept_embeddings.items():
        embeddings = np.stack(embeddings, axis=0)
        mm = sample_multimodality(embeddings, args.multimodality_times, rng)
        if mm is None:
            insufficient_samples[sample_name] = embeddings.shape[0]
            continue
        sample_mmodality[sample_name] = mm

    if not sample_mmodality:
        return {
            "scenario": name,
            "num_records": len(records),
            "error": "No samples left with enough repeats to compute multimodality.",
            "insufficient_samples": insufficient_samples,
        }

    motion_embeddings = np.stack([record["embedding"] for record in records], axis=0)
    text_embeddings = encode_text_embeddings(records, args.text_dir, eval_wrapper, w_vectorizer)
    r_precision_counts, matching_score = calculate_R_precision(
        text_embeddings,
        motion_embeddings,
        top_k=args.top_k,
        sum_all=True,
    )

    return {
        "scenario": name,
        "num_records": len(records),
        "num_unique_samples": len(kept_embeddings),
        "num_samples_used_for_mmodality": len(sample_mmodality),
        "num_samples_insufficient_repeats": len(insufficient_samples),
        "sample_mmodality": sample_mmodality,
        "insufficient_samples": insufficient_samples,
        "r_precision": r_precision_counts.astype(np.float64) / motion_embeddings.shape[0],
        "mm_dist": float(matching_score / motion_embeddings.shape[0]),
        "mean_pene": float(np.mean([record["pene"] for record in records])),
        "mean_mismatch": float(np.mean([record["mismatch"] for record in records])),
    }


def print_scenario_summary(result, top_k):
    print(f"SCENARIO: {result['scenario']}")
    print(f"num_records: {result['num_records']}")
    if "error" in result:
        print(f"error: {result['error']}")
        if result.get("insufficient_samples"):
            preview = list(result["insufficient_samples"].items())[:10]
            print(f"insufficient_samples_preview: {preview}")
        return

    mm_values = np.array(list(result["sample_mmodality"].values()), dtype=np.float64)
    print(f"num_unique_samples: {result['num_unique_samples']}")
    print(f"num_samples_used_for_mmodality: {result['num_samples_used_for_mmodality']}")
    print(f"num_samples_insufficient_repeats: {result['num_samples_insufficient_repeats']}")
    print(f"mean_pene: {result['mean_pene']:.8f}")
    print(f"mean_mismatch: {result['mean_mismatch']:.8f}")
    print(f"mmodality: {mm_values.mean():.8f}")
    print(f"r_precision_top1: {result['r_precision'][0]:.8f}")
    if top_k >= 2:
        print(f"r_precision_top2: {result['r_precision'][1]:.8f}")
    if top_k >= 3:
        print(f"r_precision_top3: {result['r_precision'][2]:.8f}")
    print(f"mm_dist: {result['mm_dist']:.8f}")


def main():
    args = parse_args()
    records = load_records_from_manifest(args.run_dir)
    input_mode = "manifest"
    if records is None:
        records = load_records_from_flat_dir(args.run_dir)
        input_mode = "flat"
    if not records:
        raise RuntimeError(f"No valid records found in {args.run_dir}")

    calculator = PeneCalculator(device=args.device)
    w_vectorizer = WordVectorizer(str(args.glove_dir), args.vocab_prefix)
    eval_wrapper = EvaluatorModelWrapper(get_opt(str(args.eval_opt_path), args.device))
    all_records = []
    contact_records = []
    threshold_records = []
    sample_stats = defaultdict(lambda: {"total": 0, "no_contact": 0, "threshold_drop": 0})
    dropped_records = []
    no_contact_records = []

    for record in tqdm(records, desc="Filtering by quality", unit="repeat"):
        sample_name = record["sample_name"]
        sample_stats[sample_name]["total"] += 1

        p1_path = args.run_dir / record["motion_p1_path"]
        p2_path = args.run_dir / record["motion_p2_path"]
        emb_path = args.run_dir / record["embedding_path"]
        embedding = np.load(emb_path).reshape(-1)
        pene = calculator.compute_from_npz(p1_path, p2_path, metric="pene").item()
        record_with_embedding = dict(record)
        record_with_embedding["embedding"] = embedding
        record_with_embedding["pene"] = pene
        record_with_embedding["mismatch"] = 0.0
        all_records.append(record_with_embedding)

        contact_path = args.contact_dir / f"{sample_name}.pkl"
        interaction_pairs = calculator.load_interaction_pairs(contact_path)

        if not interaction_pairs:
            sample_stats[sample_name]["no_contact"] += 1
            no_contact_records.append(
                {
                    "sample_name": sample_name,
                    "repeat_id": record["repeat_id"],
                    "drop_reason": "no_contact_pairs",
                    "embedding_path": str(record["embedding_path"]),
                }
            )
            continue

        contact_records.append(record_with_embedding)
        mismatch = calculator.compute_from_npz(
            p1_path,
            p2_path,
            metric="mismatch",
            interaction_pairs=interaction_pairs,
        ).item()
        record_with_embedding["pene"] = pene
        record_with_embedding["mismatch"] = mismatch

        drop_for_pene = pene > args.pene_thresh
        drop_for_mismatch = args.mismatch_thresh is not None and mismatch > args.mismatch_thresh
        if drop_for_pene or drop_for_mismatch:
            sample_stats[sample_name]["threshold_drop"] += 1
            dropped_records.append(
                {
                    "sample_name": sample_name,
                    "repeat_id": record["repeat_id"],
                    "mean_pene_per_frame": pene,
                    "mean_mismatch_per_frame": mismatch,
                    "drop_reason": "pene" if drop_for_pene and not drop_for_mismatch else "mismatch" if drop_for_mismatch and not drop_for_pene else "pene+mismatch",
                    "embedding_path": str(record["embedding_path"]),
                }
            )
            continue

        threshold_records.append(record_with_embedding)

    all_result = evaluate_subset("1_all_records", all_records, args, eval_wrapper, w_vectorizer)
    contact_result = evaluate_subset("2_contact_only", contact_records, args, eval_wrapper, w_vectorizer)
    threshold_result = evaluate_subset("3_contact_and_threshold", threshold_records, args, eval_wrapper, w_vectorizer)

    print("SUMMARY")
    print(f"run_dir: {args.run_dir}")
    print(f"input_mode: {input_mode}")
    print(f"num_manifest_records: {len(records)}")
    print(f"pene_threshold: {args.pene_thresh}")
    print(f"mismatch_threshold: {args.mismatch_thresh}")
    print(f"num_unique_samples: {len(sample_stats)}")
    print(f"num_all_repeats: {len(all_records)}")
    print(f"num_contact_repeats: {len(contact_records)}")
    print(f"num_threshold_kept_repeats: {len(threshold_records)}")
    print(f"num_no_contact_repeats: {len(no_contact_records)}")
    print(f"num_threshold_dropped_repeats: {len(dropped_records)}")
    print_scenario_summary(all_result, args.top_k)
    print_scenario_summary(contact_result, args.top_k)
    print_scenario_summary(threshold_result, args.top_k)


if __name__ == "__main__":
    main()
