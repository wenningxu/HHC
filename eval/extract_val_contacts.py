import argparse
import codecs as cs
import os
import pickle as pkl
from pathlib import Path

from tqdm import tqdm

from optimization.text_2_regions import ContactExtractor
from utils.paramUtil import CANDIDATES


def parse_args():
    parser = argparse.ArgumentParser(description="Extract and cache contact pairs for InterX validation samples.")
    parser.add_argument("--val-file", type=Path, default=Path("dataset/interx/splits/val.txt"))
    parser.add_argument("--text-dir", type=Path, default=Path("dataset/interx/texts_processed"))
    parser.add_argument("--out-dir", type=Path, default=Path("dataset/interx/contact_pairs"))
    parser.add_argument("--api-key", default=os.environ.get("GEMINI_API_KEY", ""), help="Gemini API key")
    parser.add_argument("--model-name", default="gemini-3-flash-preview")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_first_caption(text_path):
    with cs.open(text_path, "r") as f:
        for line in f:
            return line.strip().split("#")[0]
    return ""


def main():
    args = parse_args()
    if not args.api_key:
        raise ValueError("Missing Gemini API key. Pass --api-key or set GEMINI_API_KEY.")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    extractor = ContactExtractor(api_key=args.api_key, model_name=args.model_name)

    sample_names = []
    with cs.open(args.val_file, "r") as f:
        for line in f:
            sample_names.append(line.strip())

    for sample_name in tqdm(sample_names, desc="Extracting contacts", unit="sample"):
        out_path = args.out_dir / f"{sample_name}.pkl"
        if out_path.exists() and not args.overwrite:
            continue

        text_path = args.text_dir / f"{sample_name}.txt"
        caption = read_first_caption(text_path)
        contacts = extractor.get_contacts(caption, CANDIDATES)
        with open(out_path, "wb") as f:
            pkl.dump(contacts, f)


if __name__ == "__main__":
    main()
