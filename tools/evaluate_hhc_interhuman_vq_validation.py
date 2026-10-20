#!/usr/bin/env python3
"""Independently rescore the selected HHC InterHuman VQ on frozen val."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataset.interhuman_fair_requests import validation_request_loader
from models.interhuman_evaluator_wrapper import (
    InterHumanEvaluatorWrapper,
    evaluator_provenance,
    load_official_metrics_module,
)
from utils.interhuman_artifacts import load_hhc_interhuman_vq, sha256_file
from utils.interhuman_vq_evaluation import evaluate_interhuman_vq


def atomic_json(path, value):
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.tmp.{os.getpid()}')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    os.replace(temporary, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--training-manifest', required=True)
    parser.add_argument(
        '--validation-manifest',
        default='output/interhuman_validation_requests/manifest.json',
    )
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()

    validation_manifest = Path(args.validation_manifest).resolve()
    validation = json.loads(validation_manifest.read_text())
    loader = validation_request_loader(validation_manifest, num_workers=0)
    model, provenance = load_hhc_interhuman_vq(
        args.checkpoint, args.training_manifest, device=args.device,
    )
    evaluator = InterHumanEvaluatorWrapper.from_official_checkpoint(args.device)
    metrics = evaluate_interhuman_vq(
        model, loader, evaluator, load_official_metrics_module(),
        device=args.device, mean=loader.dataset.adapter.mean,
        std=loader.dataset.adapter.std, random_seed=20260130,
    )
    result = {
        'schema': 'hhc_interhuman_vq_official_validation_score_v1',
        'status': 'complete',
        'split': 'val',
        'samples': 480,
        **metrics,
        'model': provenance,
        'validation_manifest': str(validation_manifest),
        'validation_manifest_sha256': sha256_file(validation_manifest),
        'request_file_sha256': validation['request']['sha256'],
        'request_content_sha256': validation['request']['request_sha256'],
        'evaluator_provenance': evaluator_provenance(include_checkpoint_sha256=True),
        'implementation_sha256': sha256_file(Path(__file__).resolve()),
    }
    atomic_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
