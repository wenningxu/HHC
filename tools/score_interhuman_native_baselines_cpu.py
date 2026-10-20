"""Frozen main/MM scores for method-native baseline lengths; CPU InterCLIP only."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from dataset.interhuman_fair_requests import InterHumanFrozenRequestDataset
from dataset.interhuman_native_baseline_h5 import METHODS, NativeBaselineH5
from dataset.interhuman_generated_h5 import sha256_file
from models.interhuman_evaluator_wrapper import (
    InterHumanEvaluatorWrapper, evaluator_provenance, load_official_interclip, load_official_metrics_module,
)
from tools.score_fair_interhuman_main import main_metrics_from_embeddings
from tools.score_fair_interhuman_mm import multimodality_from_embeddings
from tools.hhc_text_only_formal_contract import validate_manifests


def score_main(data, generated, model):
    arrays = {key: [] for key in ('GT', 'raw', 'text')}
    with torch.inference_mode():
        for i, row in enumerate(data.requests):
            name, caption, a, b, n = data[i]
            pn, pt, p, q, ln = generated[i]
            if name != pn or caption != pt or row['processed_length'] != n:
                raise ValueError('Request identities changed')
            for first, second, length, key in ((a, b, n, 'GT'), (p, q, ln, 'raw')):
                raw = torch.stack((first, second), dim=1)[:length].numpy() * data.adapter.std + data.adapter.mean
                z = model.encode_motion({'motions': torch.from_numpy(raw.astype(np.float32)).flatten(1)[None],
                                         'motion_lens': torch.tensor([length])})['motion_emb']
                if z.shape != (1, 512) or not torch.isfinite(z).all():
                    raise ValueError('Invalid motion embedding')
                arrays[key].append(z.numpy()[0])
            arrays['text'].append(model.encode_text({'text': [caption]})['text_emb'].numpy()[0])
            if (i + 1) % 96 == 0:
                print(json.dumps({'encoded': i + 1}), flush=True)
    return {key: np.stack(value) for key, value in arrays.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--generations', type=Path, required=True)
    parser.add_argument('--main-manifest', type=Path, required=True)
    parser.add_argument('--mm-manifest', type=Path, required=True)
    parser.add_argument('--method', choices=METHODS, required=True)
    parser.add_argument('--mode', choices=['main', 'mm'], required=True)
    parser.add_argument('--replication', type=int, choices=range(20), required=True)
    parser.add_argument('--reference-embeddings', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise RuntimeError('This evaluator is CPU only; hide GPUs')
    embeddings = args.output.with_suffix('.embeddings.npz')
    if args.output.exists() or embeddings.exists():
        raise FileExistsError('Existing scoring output; never overwrite')
    validate_manifests(args.main_manifest, args.mm_manifest)
    torch.set_num_threads(2); torch.set_num_interop_threads(2)
    if os.name != 'nt': os.nice(10)
    path = args.main_manifest if args.mode == 'main' else args.mm_manifest
    manifest = json.loads(path.read_text())
    entry = next(row for row in manifest['replications'] if row['replication'] == args.replication)
    generated = NativeBaselineH5(args.generations, entry['path'], entry, args.method, args.mode)
    started = time.time()
    comparison = None
    if args.mode == 'main':
        data = InterHumanFrozenRequestDataset(args.replication, args.main_manifest)
        model, _ = load_official_interclip('cpu'); model.eval().requires_grad_(False)
        arrays = score_main(data, generated, model)
        if args.reference_embeddings:
            with np.load(args.reference_embeddings) as reference:
                comparison = {key: {'equal': bool(np.array_equal(arrays[key], reference[key])),
                    'max_abs': float(np.abs(arrays[key] - reference[key]).max())} for key in ('GT', 'text')}
            if not all(row['equal'] for row in comparison.values()):
                raise ValueError('GT/text differs from the reference method')
        metrics = main_metrics_from_embeddings(arrays['text'], arrays['raw'], arrays['GT'],
                                               load_official_metrics_module(), batch_size=96, random_seed=1234)
    else:
        if args.reference_embeddings:
            parser.error('MM has no GT/text reference embeddings argument')
        wrapper = InterHumanEvaluatorWrapper.from_official_checkpoint('cpu')
        arrays = {'motion': np.stack([wrapper.get_motion_embeddings(generated[i]).cpu().numpy()
                                     for i in range(len(generated))])}
        metrics = {'multimodality': multimodality_from_embeddings(arrays['motion'],
            load_official_metrics_module(), pairs=10, random_seed=1234 + args.replication)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with embeddings.open('xb') as stream: np.savez_compressed(stream, **arrays)
    report = {'schema': 'interhuman_native_baseline_score_v1', 'status': 'complete', 'split': 'test',
        'method': args.method, 'mode': args.mode, 'replication': args.replication,
        'samples': 1056 if args.mode == 'main' else 100, 'mm_repeats': None if args.mode == 'main' else 30,
        'metrics': metrics, 'generation_metadata': generated.metadata,
        'generation': str(args.generations.resolve()), 'generation_sha256': sha256_file(args.generations),
        'request_file_sha256': entry['sha256'], 'request_content_sha256': entry['request_sha256'],
        'request_manifest_sha256': sha256_file(path), 'evaluator': evaluator_provenance(include_checkpoint_sha256=True),
        'embeddings': str(embeddings.resolve()), 'embeddings_sha256': sha256_file(embeddings),
        'reference_gt_text': comparison,
        'reference_embeddings_sha256': sha256_file(args.reference_embeddings) if args.reference_embeddings else None,
        'protocol': 'native output length; normalized262 denormalized once; CPU InterCLIP; main per-request encoding / batch96 retrieval; MM released batch30',
        'script_sha256': sha256_file(Path(__file__)), 'elapsed_s': time.time() - started}
    generated.close()
    temporary = args.output.with_suffix('.tmp'); temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
    os.replace(temporary, args.output)
    print(json.dumps({'status': 'complete', 'metrics': metrics}), flush=True)


if __name__ == '__main__':
    main()
