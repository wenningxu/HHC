#!/usr/bin/env python
"""Score generated 672-D motions with the official Inter-X evaluator."""

import argparse
import hashlib
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import h5py
import numpy as np
import torch

from models.evaluator_wrapper import EvaluatorModelWrapper
from options.get_eval_option import get_opt
from utils.eval_trans import (
    calculate_R_precision,
    calculate_activation_statistics,
    calculate_diversity,
    calculate_frechet_distance,
    calculate_multimodality,
)


def sha256_file(path, chunk_size=8 * 1024 * 1024):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b''):
            digest.update(chunk)
    return digest.hexdigest()


@torch.inference_mode()
def embed_motions(evaluator, motions, lengths, batch_size, neutralize_hands=False):
    embeddings = []
    for start in range(0, len(lengths), batch_size):
        end = min(start + batch_size, len(lengths))
        batch_motion = torch.from_numpy(np.asarray(motions[start:end])).float()
        batch_lengths = torch.from_numpy(lengths[start:end]).long()
        if neutralize_hands:
            structured = batch_motion.reshape(
                batch_motion.shape[0], batch_motion.shape[1], 56, 12
            )
            identity = torch.tensor([1, 0, 0, 0, 1, 0], dtype=structured.dtype)
            for sample_index, valid_length in enumerate(batch_lengths.tolist()):
                structured[sample_index, :valid_length, 22:55, :6] = identity
                structured[sample_index, :valid_length, 22:55, 6:12] = identity
        sorted_embedding = evaluator.get_motion_embeddings(batch_motion, batch_lengths)
        order = np.argsort(batch_lengths.numpy().tolist())[::-1].copy()
        inverse = np.empty_like(order)
        inverse[order] = np.arange(len(order))
        embeddings.append(sorted_embedding[inverse].cpu().numpy())
    return np.concatenate(embeddings)


def calculate_batched_retrieval(text_embeddings, motion_embeddings, batch_size):
    if len(text_embeddings) % batch_size:
        raise ValueError(
            f'official retrieval requires complete batches, got {len(text_embeddings)} '
            f'samples for batch size {batch_size}'
        )
    r_precision = np.zeros(3, dtype=np.float64)
    matching = 0.0
    for start in range(0, len(text_embeddings), batch_size):
        end = start + batch_size
        batch_r, batch_matching = calculate_R_precision(
            text_embeddings[start:end], motion_embeddings[start:end],
            top_k=3, sum_all=True,
        )
        r_precision += batch_r
        matching += batch_matching
    return r_precision / len(text_embeddings), matching / len(text_embeddings)


def validate_generation_file(source, expected_names, expected_lengths, repeats, label):
    expected_shape = (len(expected_lengths), repeats)
    if source['motions'].shape[:2] != expected_shape:
        raise ValueError(
            f'{label} generation shape mismatch: expected {expected_shape}, '
            f'got {source["motions"].shape[:2]}'
        )
    if 'lengths' not in source or not np.array_equal(source['lengths'][:], expected_lengths):
        raise ValueError(f'{label} lengths do not exactly match the frozen requests')
    if 'names' not in source:
        raise ValueError(f'{label} is missing sample names')
    actual_names = source['names'][:].astype(str)
    if not np.array_equal(actual_names, expected_names):
        mismatch = np.flatnonzero(actual_names != expected_names)
        first = int(mismatch[0]) if len(mismatch) else -1
        raise ValueError(f'{label} sample order/name mismatch at index {first}')
    if 'completed_repeats' in source:
        completed = source['completed_repeats'][:].astype(bool)
        if completed.shape != (repeats,) or not completed.all():
            raise ValueError(
                f'{label} generation is incomplete: completed repeats {completed.tolist()}'
            )


def embedding_lengths(source, requested_lengths):
    if 'generated_lengths' not in source:
        return requested_lengths
    generated = source['generated_lengths'][:].astype(np.int64)
    if generated.shape != requested_lengths.shape:
        raise ValueError('generated_lengths shape mismatch')
    if np.any(generated <= 0) or np.any(generated > requested_lengths):
        raise ValueError('generated_lengths must be in (0, requested_lengths]')
    return generated


def serializable_attributes(source):
    result = {}
    for key, value in source.attrs.items():
        if isinstance(value, bytes):
            value = value.decode('utf-8')
        elif isinstance(value, np.generic):
            value = value.item()
        result[key] = value
    return result


def validate_generation_metadata(main, multimodality, request_seed):
    """Reject main/MM files produced by different model or sampler contracts."""
    identity_keys = {
        'checkpoint', 'checkpoint_sha256', 'vq_checkpoint',
        'vq_checkpoint_sha256', 'normalization_mean',
        'normalization_mean_sha256', 'normalization_std',
        'normalization_std_sha256', 'contact_map', 'contact_map_sha256',
        'requests', 'requests_sha256', 'seed_strategy', 'sampling_steps', 'temperature',
        'sampler', 'sampler_semantics', 'cascade_order', 'sampling_steps_unit',
        'channel_updates_per_outer_step',
        'model_forward_calls_per_outer_step_with_cfg',
        'cfg_scale', 'contact_guidance', 'contact_guidance_scale',
        'contact_timing_source', 'contact_timing_file',
        'contact_timing_file_sha256',
        'generation_length_policy', 'time_steps', 'cond_scale',
        'topk_filter_thres', 'length_policy', 'adapter', 'batch_size',
        'strategy', 'cfg_weight', 'projection', 'sampling_process',
        'first_frame_conditioning',
    }
    mismatches = {}
    for key in sorted(identity_keys):
        main_has, mm_has = key in main, key in multimodality
        if main_has != mm_has or (main_has and main[key] != multimodality[key]):
            mismatches[key] = (main.get(key), multimodality.get(key))
    if mismatches:
        raise ValueError(f'main/MM generation provenance mismatch: {mismatches}')
    if 'seed' in main and int(main['seed']) != int(request_seed):
        raise ValueError(
            f'main generation seed {main["seed"]} != request seed {request_seed}'
        )
    if 'seed' in multimodality and int(multimodality['seed']) != int(request_seed) + 100_000:
        raise ValueError(
            'MM generation seed does not equal request seed + 100000: '
            f'{multimodality["seed"]} vs {int(request_seed) + 100_000}'
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--requests', required=True)
    parser.add_argument('--main-generations', required=True)
    parser.add_argument('--mm-generations', required=True)
    parser.add_argument(
        '--gt-generations', default=None,
        help=(
            'Optional one-repeat GT HDF5 replacing frozen GT embeddings. '
            'Used for explicitly labelled representation-controlled audits.'
        ),
    )
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--batch-size', type=int, default=32)
    args = parser.parse_args()

    request = np.load(args.requests, allow_pickle=False)
    lengths = request['lengths'].astype(np.int64)
    names = request['names'].astype(str)
    text_embeddings = request['text_embeddings']
    gt_embeddings = request['gt_embeddings']
    gt_body_embeddings = request['gt_body_embeddings']
    mm_indices = request['mm_indices'].astype(np.int64)
    evaluator = EvaluatorModelWrapper(
        get_opt('checkpoints/hhi/Comp_v6_KLD01/opt.txt', args.device)
    )

    if args.gt_generations:
        with h5py.File(args.gt_generations, 'r') as source:
            validate_generation_file(source, names, lengths, 1, 'GT override')
            gt_embeddings = embed_motions(
                evaluator, source['motions'][:, 0], lengths, args.batch_size
            )
            gt_body_embeddings = embed_motions(
                evaluator, source['motions'][:, 0], lengths, args.batch_size,
                neutralize_hands=True,
            )

    with h5py.File(args.main_generations, 'r') as source:
        main_metadata = serializable_attributes(source)
        validate_generation_file(source, names, lengths, 1, 'main')
        main_embedding_lengths = embedding_lengths(source, lengths)
        generated_embeddings = embed_motions(
            evaluator, source['motions'][:, 0], main_embedding_lengths, args.batch_size
        )
        generated_body_embeddings = embed_motions(
            evaluator, source['motions'][:, 0], main_embedding_lengths, args.batch_size,
            neutralize_hands=True,
        )

    with h5py.File(args.mm_generations, 'r') as source:
        mm_metadata = serializable_attributes(source)
        mm_lengths = lengths[mm_indices]
        mm_names = names[mm_indices]
        validate_generation_file(source, mm_names, mm_lengths, 30, 'MM')
        mm_embedding_lengths = embedding_lengths(source, mm_lengths)
        expected_mm_embedding_lengths = main_embedding_lengths[mm_indices]
        if not np.array_equal(
            mm_embedding_lengths, expected_mm_embedding_lengths
        ):
            raise ValueError(
                'main and MM generation files use different effective length policies'
            )
        per_repeat = []
        for repeat in range(30):
            per_repeat.append(embed_motions(
                evaluator, source['motions'][:, repeat], mm_embedding_lengths,
                args.batch_size
            ))
        mm_embeddings = np.stack(per_repeat, axis=1)

    validate_generation_metadata(main_metadata, mm_metadata, int(request['seed']))
    request_sha256 = sha256_file(args.requests)
    for label, metadata in (('main', main_metadata), ('MM', mm_metadata)):
        recorded = metadata.get('requests_sha256')
        if recorded is not None and recorded != request_sha256:
            raise ValueError(
                f'{label} generation request hash {recorded} != {request_sha256}'
            )

    np.random.seed(int(request['seed']))
    gt_mu, gt_cov = calculate_activation_statistics(gt_embeddings)
    gt_body_mu, gt_body_cov = calculate_activation_statistics(gt_body_embeddings)
    pred_mu, pred_cov = calculate_activation_statistics(generated_embeddings)
    pred_body_mu, pred_body_cov = calculate_activation_statistics(generated_body_embeddings)
    r_precision, matching = calculate_batched_retrieval(
        text_embeddings, generated_embeddings, args.batch_size,
    )
    result = {
        'fid': float(calculate_frechet_distance(gt_mu, gt_cov, pred_mu, pred_cov)),
        'fid_body_controlled': float(calculate_frechet_distance(
            gt_body_mu, gt_body_cov, pred_body_mu, pred_body_cov
        )),
        'diversity_gt': float(calculate_diversity(gt_embeddings, 300)),
        'diversity_gt_body_controlled': float(calculate_diversity(gt_body_embeddings, 300)),
        'diversity': float(calculate_diversity(generated_embeddings, 300)),
        'diversity_body_controlled': float(calculate_diversity(
            generated_body_embeddings, 300
        )),
        'top1': float(r_precision[0]),
        'top2': float(r_precision[1]),
        'top3': float(r_precision[2]),
        'matching': float(matching),
        'multimodality': float(calculate_multimodality(mm_embeddings, 10)),
        'samples': int(len(lengths)),
        'mm_samples': int(len(mm_indices)),
        'mm_repeats': 30,
        'requests': os.path.abspath(args.requests),
        'requests_sha256': request_sha256,
        'generation_request_hash_verified': (
            'requests_sha256' in main_metadata and 'requests_sha256' in mm_metadata
        ),
        'main_generations': os.path.abspath(args.main_generations),
        'main_generations_sha256': sha256_file(args.main_generations),
        'mm_generations': os.path.abspath(args.mm_generations),
        'mm_generations_sha256': sha256_file(args.mm_generations),
        'gt_generations': (
            os.path.abspath(args.gt_generations) if args.gt_generations else None
        ),
        'gt_generations_sha256': (
            sha256_file(args.gt_generations) if args.gt_generations else None
        ),
        'representation_controlled': bool(args.gt_generations),
        'generation_length_policy': (
            'generated_lengths' if not np.array_equal(main_embedding_lengths, lengths)
            else 'requested_lengths'
        ),
        'main_generation_metadata': main_metadata,
        'mm_generation_metadata': mm_metadata,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
