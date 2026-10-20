#!/usr/bin/env python
"""Freeze official Inter-X test crops, captions and GT evaluator embeddings."""

import argparse
import hashlib
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import h5py
import numpy as np
import torch

from dataset import dataset_interx_TM
from models.evaluator_wrapper import EvaluatorModelWrapper
from options.get_eval_option import get_opt
from utils.motion_process import MotionNormalizerTorchInterX
from utils.word_vectorizer import WordVectorizer


def sha256_file(path, chunk_size=8 * 1024 * 1024):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b''):
            digest.update(chunk)
    return digest.hexdigest()


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', default='output/fair_interx_requests')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--replications', type=int, default=20)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--mm-num-samples', type=int, default=100)
    parser.add_argument(
        '--ground-truth-h5', default=None,
        help=(
            'Optional HDF5 path for the exact denormalized GT crops from one '
            'replication. This uses the same generation-file contract as the methods.'
        ),
    )
    parser.add_argument('--ground-truth-replication', type=int, default=0)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    vectorizer = WordVectorizer('./dataset/interx/glove', 'hhi_vab')
    loader = dataset_interx_TM.DATALoader(
        'hhi', 32,
        'dataset/interx/splits/test.txt', vectorizer,
        'dataset/interx/motions/test.h5',
        unit_length=4, shuffle=True,
    )
    evaluator = EvaluatorModelWrapper(
        get_opt('checkpoints/hhi/Comp_v6_KLD01/opt.txt', args.device)
    )
    normalizer = MotionNormalizerTorchInterX('raw')

    for replication in range(args.replications):
        replication_seed = args.seed + replication
        seed_everything(replication_seed)
        gt_target = None
        gt_partial_path = None
        gt_cursor = 0
        if args.ground_truth_h5 and replication == args.ground_truth_replication:
            gt_path = os.path.abspath(args.ground_truth_h5)
            os.makedirs(os.path.dirname(gt_path), exist_ok=True)
            gt_partial_path = gt_path + '.partial'
            gt_target = h5py.File(gt_partial_path, 'w')
            gt_target.create_dataset(
                'motions', shape=(1696, 1, 150, 56 * 12), dtype=np.float32,
                chunks=(1, 1, 150, 56 * 12), compression='lzf',
            )
        all_names, all_captions, all_lengths = [], [], []
        all_text_embeddings, all_gt_embeddings, all_gt_body_embeddings = [], [], []
        all_betas, all_gender, all_contact = [], [], []
        for batch in loader:
            (word_embeddings, pos_one_hots, captions, sent_len, pose, lengths,
             _, names, betas, gender, contact) = batch
            raw_pose = normalizer.backward(pose.float())
            text_embedding, gt_embedding = evaluator.get_co_embeddings(
                word_embeddings, pos_one_hots, sent_len, raw_pose, lengths
            )
            body_pose = raw_pose.reshape(raw_pose.shape[0], raw_pose.shape[1], 56, 12).clone()
            identity = torch.tensor([1, 0, 0, 0, 1, 0], dtype=body_pose.dtype)
            for sample_index, valid_length in enumerate(lengths.tolist()):
                body_pose[sample_index, :valid_length, 22:55, :6] = identity
                body_pose[sample_index, :valid_length, 22:55, 6:12] = identity
            _, gt_body_embedding = evaluator.get_co_embeddings(
                word_embeddings, pos_one_hots, sent_len,
                body_pose.reshape_as(raw_pose), lengths,
            )
            order = np.argsort(lengths.numpy().tolist())[::-1].copy()
            if gt_target is not None:
                batch_gt = raw_pose.cpu().numpy()[order]
                gt_target['motions'][gt_cursor:gt_cursor + len(batch_gt), 0] = batch_gt
                gt_cursor += len(batch_gt)
            all_names.extend(np.asarray(names)[order].tolist())
            all_captions.extend(np.asarray(captions)[order].tolist())
            all_lengths.extend(lengths.numpy()[order].tolist())
            all_text_embeddings.append(text_embedding.cpu().numpy())
            all_gt_embeddings.append(gt_embedding.cpu().numpy())
            all_gt_body_embeddings.append(gt_body_embedding.cpu().numpy())
            all_betas.append(betas.numpy()[order])
            all_gender.append(gender.numpy()[order])
            all_contact.append(contact.numpy()[order])

        names = np.asarray(all_names)
        captions = np.asarray(all_captions)
        lengths = np.asarray(all_lengths, dtype=np.int64)
        text_embeddings = np.concatenate(all_text_embeddings)
        gt_embeddings = np.concatenate(all_gt_embeddings)
        gt_body_embeddings = np.concatenate(all_gt_body_embeddings)
        betas = np.concatenate(all_betas).astype(np.float32)
        gender = np.concatenate(all_gender).astype(np.int64)
        contact = np.concatenate(all_contact).astype(np.int8)
        if len(names) != 1696:
            raise RuntimeError(f'official batch/drop_last contract expected 1696 samples, got {len(names)}')
        generator = np.random.default_rng(args.seed + 10_000 + replication)
        mm_indices = generator.permutation(len(names))[:args.mm_num_samples]
        path = os.path.join(args.output_dir, f'replication_{replication:02d}.npz')
        np.savez_compressed(
            path,
            names=names,
            captions=captions,
            lengths=lengths,
            text_embeddings=text_embeddings,
            gt_embeddings=gt_embeddings,
            gt_body_embeddings=gt_body_embeddings,
            betas=betas,
            gender=gender,
            contact=contact,
            mm_indices=mm_indices,
            seed=np.asarray(replication_seed),
        )
        if gt_target is not None:
            if gt_cursor != len(names):
                raise RuntimeError(
                    f'GT HDF5 cursor mismatch: wrote {gt_cursor}, expected {len(names)}'
                )
            gt_target.create_dataset('lengths', data=lengths)
            gt_target.create_dataset('names', data=np.asarray(names, dtype='S'))
            gt_target.attrs['requests'] = os.path.abspath(path)
            gt_target.attrs['requests_sha256'] = sha256_file(path)
            gt_target.attrs['seed'] = replication_seed
            gt_target.attrs['source'] = 'official Inter-X frozen ground truth'
            gt_target.close()
            gt_target = None
            os.replace(gt_partial_path, os.path.abspath(args.ground_truth_h5))
        print({'replication': replication, 'samples': len(names), 'request': path})


if __name__ == '__main__':
    main()
