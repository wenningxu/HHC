#!/usr/bin/env python
import argparse
import json
import os
import sys

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from utils.contact_map import build_code_contact_map


def main():
    parser = argparse.ArgumentParser(description='Build paper contact-guidance lookup table')
    parser.add_argument('--indices-dir', default='checkpoints/vq_paper/codebook')
    parser.add_argument('--contact-dir', default='checkpoints/vq_paper/contact_token_label')
    parser.add_argument('--codebook-size', type=int, default=8096)
    parser.add_argument('--alpha', type=float, default=1.0)
    parser.add_argument('--output', default='checkpoints/vq_paper/code_contact_probs.npy')
    args = parser.parse_args()

    probabilities, counts, used_files = build_code_contact_map(
        args.indices_dir, args.contact_dir, args.codebook_size, args.alpha
    )
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    np.save(args.output, probabilities)
    report = {
        'output': args.output,
        'shape': list(probabilities.shape),
        'used_files': used_files,
        'dead_codes': (counts == 0).sum(axis=1).astype(int).tolist(),
        'min_probability': float(probabilities.min()),
        'max_probability': float(probabilities.max()),
        'alpha': args.alpha,
    }
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
