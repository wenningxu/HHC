#!/usr/bin/env python
"""Evaluate an HHC VQ checkpoint on the frozen official Inter-X GT crops."""

import argparse
import hashlib
import json
import math
import os
import re
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import h5py
import numpy as np
import torch

from models.model_tools import build_model
from utils.fk import FK_Pair_Layer_Batch
from utils.motion_process import MotionNormalizerTorchInterX
from utils.paramUtil import INTERX_SKELETONS, SMPLX_TREE
from utils.rotation_conversions import rotation_6d_to_matrix


GROUP_PATTERN = re.compile(r'^G(\d{3})')


def sha256_file(path, chunk_size=1024 * 1024):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b''):
            digest.update(chunk)
    return digest.hexdigest()


def group_ids(names, device):
    groups = []
    for name in names:
        match = GROUP_PATTERN.match(str(name))
        if match is None:
            raise ValueError(f'cannot parse Inter-X group from {name!r}')
        group = int(match.group(1)) - 1
        groups.append((0, group, 1, group))
    return torch.tensor(groups, dtype=torch.long, device=device)


def entropy_perplexity(counts):
    probabilities = counts[counts > 0].double()
    probabilities /= probabilities.sum()
    return float(torch.exp(-(probabilities * probabilities.log()).sum()))


def model_from_checkpoint(checkpoint, device, nb_code=None, code_dim=None):
    saved_args = dict(checkpoint.get('args', {}))
    state_dict = checkpoint['net']
    # Legacy VQ checkpoints predate argument snapshots.  Recover the two
    # architecture fields that are unambiguously encoded in their tensors
    # instead of silently constructing the current defaults.
    codebook = state_dict['quantizer_upper.codebook']
    inferred_nb_code = int(codebook.shape[0])
    inferred_code_dim = int(codebook.shape[1] * 8 // 3)
    inferred_shape_condition_dim = int(
        state_dict['condition_mlp.2.weight'].shape[0]
    )
    saved_args.update({
        'device': str(device), 'dataname': 'hhi', 'motion_rep': 'raw',
        'nb_joints': 55,
    })
    required_defaults = {
        'nb_code': 8096, 'code_dim': 256, 'output_emb_width': 512,
        'down_t': 2, 'stride_t': 2, 'width': 512, 'depth': 3,
        'dilation_growth_rate': 3, 'vq_act': 'relu', 'vq_norm': None,
        'mu': 0.99,
    }
    for key, value in required_defaults.items():
        saved_args.setdefault(key, value)
    saved_args['nb_code'] = inferred_nb_code if nb_code is None else int(nb_code)
    saved_args['code_dim'] = inferred_code_dim if code_dim is None else int(code_dim)
    saved_args['shape_condition_dim'] = inferred_shape_condition_dim
    model = build_model(SimpleNamespace(**saved_args), 'interx_vq')
    incompatible = model.load_state_dict(state_dict, strict=False)
    # The only tolerated schema migration is the FK skeleton lookup: early
    # checkpoints stored one lookup per person, while the current FK layer
    # stores a unified table.  It is not used by encode/decode reconstruction.
    non_fk_missing = [key for key in incompatible.missing_keys if not key.startswith('fk_layers.')]
    non_fk_unexpected = [key for key in incompatible.unexpected_keys if not key.startswith('fk_layers.')]
    if non_fk_missing or non_fk_unexpected:
        raise RuntimeError(
            f'checkpoint mismatch: missing={non_fk_missing}, '
            f'unexpected={non_fk_unexpected}'
        )
    return model.to(device).eval(), saved_args


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument(
        '--ground-truth', default='output/fair_interx_ground_truth/replication_00.h5'
    )
    parser.add_argument(
        '--requests', default='output/fair_interx_requests/replication_00.npz'
    )
    parser.add_argument('--output', required=True)
    parser.add_argument(
        '--reconstruction-h5', default=None,
        help=(
            'Optional fair-evaluation HDF5 containing the exact VQ '
            'encode/decode reconstructions in frozen-request order.'
        ),
    )
    parser.add_argument('--device', default='cuda:1')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument(
        '--input-motion-rep', choices=('raw', 'vel'), default='raw',
        help='Statistics used to normalize the frozen raw motion before encoding.',
    )
    parser.add_argument(
        '--output-motion-rep', choices=('raw', 'vel'), default='raw',
        help='Statistics used to de-normalize the decoder output.',
    )
    parser.add_argument(
        '--nb-code', type=int, default=None,
        help='Override for legacy checkpoints that did not save their arguments.',
    )
    parser.add_argument(
        '--code-dim', type=int, default=None,
        help='Override for legacy checkpoints; otherwise inferred from the codebook.',
    )
    parser.add_argument('--limit', type=int, default=None)
    args = parser.parse_args()

    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    model, saved_args = model_from_checkpoint(
        checkpoint, device, args.nb_code, args.code_dim
    )
    input_normalizer = MotionNormalizerTorchInterX(args.input_motion_rep)
    output_normalizer = MotionNormalizerTorchInterX(args.output_motion_rep)
    request = np.load(args.requests, allow_pickle=False)
    names = request['names'].astype(str)
    lengths = request['lengths'].astype(np.int64)
    betas = request['betas'].astype(np.float32)
    gender = request['gender'].astype(np.int64)
    if args.limit is not None:
        names, lengths, betas, gender = (
            names[:args.limit], lengths[:args.limit], betas[:args.limit], gender[:args.limit]
        )

    skeletons = torch.from_numpy(np.asarray(INTERX_SKELETONS).copy()).float()
    fk = FK_Pair_Layer_Batch(
        SMPLX_TREE, skeletons[:, 0], skeletons[:, 1], device=device
    ).to(device).eval()
    code_counts = torch.zeros(3, model.num_code, dtype=torch.long)
    sums = {
        'absolute_error': 0.0,
        'squared_error': 0.0,
        'feature_values': 0,
        'joint_error_m': 0.0,
        'body_joint_error_m': 0.0,
        'hand_joint_error_m': 0.0,
        'joint_values': 0,
        'body_joint_values': 0,
        'hand_joint_values': 0,
        'root_translation_error_m': 0.0,
        'root_translation_values': 0,
        'rotation_error_rad': 0.0,
        'rotation_values': 0,
        'tokens': 0,
    }

    reconstruction_target = None
    reconstruction_partial = None
    if args.reconstruction_h5 is not None:
        reconstruction_path = os.path.abspath(args.reconstruction_h5)
        os.makedirs(os.path.dirname(reconstruction_path), exist_ok=True)
        reconstruction_partial = reconstruction_path + f'.partial.{os.getpid()}'
        reconstruction_target = h5py.File(reconstruction_partial, 'w')
        reconstruction_target.create_dataset(
            'motions', shape=(len(names), 1, 150, 56 * 12), dtype=np.float32,
            chunks=(1, 1, 150, 56 * 12), compression='lzf',
        )

    with h5py.File(args.ground_truth, 'r') as source:
        if not np.array_equal(source['names'][:len(names)].astype(str), names):
            raise ValueError('ground-truth names do not match the frozen request')
        if not np.array_equal(source['lengths'][:len(lengths)], lengths):
            raise ValueError('ground-truth lengths do not match the frozen request')
        for start in range(0, len(names), args.batch_size):
            end = min(start + args.batch_size, len(names))
            raw = torch.from_numpy(source['motions'][start:end, 0]).float().to(device)
            batch_betas = torch.from_numpy(betas[start:end]).float().to(device)
            batch_gender = torch.from_numpy(gender[start:end]).long().to(device)
            normalized = input_normalizer.forward(raw)
            batch_lengths = torch.from_numpy(lengths[start:end]).long().to(device)
            token_lengths = torch.ceil(batch_lengths.float() / model.temporal_stride).long()
            reconstructed_raw = torch.zeros_like(raw)
            # Convolutional encoding floors lengths. Pad in normalized space
            # to the actual VQ stride (4 for the reference, 8 for the new arm).
            for token_length in token_lengths.unique(sorted=True).tolist():
                selection = token_lengths == token_length
                padded_frames = int(token_length) * model.temporal_stride
                normalized_group = torch.zeros(
                    int(selection.sum()), padded_frames, raw.shape[-1],
                    dtype=normalized.dtype, device=device,
                )
                group_normalized = normalized[selection, :padded_frames]
                normalized_group[:, :group_normalized.shape[1]] = group_normalized
                group_lengths = batch_lengths[selection]
                group_frame_index = torch.arange(padded_frames, device=device).view(1, -1)
                normalized_group = normalized_group.masked_fill(
                    (group_frame_index >= group_lengths.view(-1, 1)).unsqueeze(-1), 0.0
                )
                group_tokens = model(
                    normalized_group, batch_betas[selection], batch_gender[selection],
                    type='encode',
                )
                group_reconstructed = model(
                    group_tokens, batch_betas[selection], batch_gender[selection],
                    type='decode',
                )
                group_raw = output_normalizer.backward(group_reconstructed)
                copy_frames = min(150, group_raw.shape[1])
                reconstructed_raw[selection, :copy_frames] = group_raw[:, :copy_frames]
                group_tokens = group_tokens.cpu()
                for channel in range(3):
                    code_counts[channel] += torch.bincount(
                        group_tokens[:, :, channel].reshape(-1), minlength=model.num_code
                    )
                sums['tokens'] += int(group_tokens.shape[0] * group_tokens.shape[1])

            frame_index = torch.arange(150, device=device).view(1, -1)
            valid = frame_index < torch.from_numpy(lengths[start:end]).to(device).view(-1, 1)
            valid_features = valid.unsqueeze(-1).expand_as(raw)
            difference = reconstructed_raw - raw
            sums['absolute_error'] += float(difference[valid_features].abs().double().sum())
            sums['squared_error'] += float(difference[valid_features].square().double().sum())
            sums['feature_values'] += int(valid_features.sum())

            raw_structured = raw.reshape(end - start, 150, 56, 12)
            rec_structured = reconstructed_raw.reshape(end - start, 150, 56, 12)
            groups = group_ids(names[start:end], device)
            gt_joints1, gt_joints2 = fk(raw_structured, groups)
            rec_joints1, rec_joints2 = fk(rec_structured, groups)
            for gt_joints, rec_joints in (
                (gt_joints1, rec_joints1), (gt_joints2, rec_joints2)
            ):
                joint_error = torch.linalg.vector_norm(rec_joints - gt_joints, dim=-1)
                sums['joint_error_m'] += float(joint_error[valid].double().sum())
                sums['joint_values'] += int(valid.sum()) * 55
                sums['body_joint_error_m'] += float(joint_error[:, :, :22][valid].double().sum())
                sums['body_joint_values'] += int(valid.sum()) * 22
                sums['hand_joint_error_m'] += float(joint_error[:, :, 25:55][valid].double().sum())
                sums['hand_joint_values'] += int(valid.sum()) * 30

            translation_error = torch.linalg.vector_norm(
                rec_structured[:, :, 55, (0, 1, 2, 6, 7, 8)].reshape(end-start, 150, 2, 3)
                - raw_structured[:, :, 55, (0, 1, 2, 6, 7, 8)].reshape(end-start, 150, 2, 3),
                dim=-1,
            )
            sums['root_translation_error_m'] += float(translation_error[valid].double().sum())
            sums['root_translation_values'] += int(valid.sum()) * 2

            for offset in (0, 6):
                gt_rot = rotation_6d_to_matrix(raw_structured[:, :, :55, offset:offset + 6])
                rec_rot = rotation_6d_to_matrix(rec_structured[:, :, :55, offset:offset + 6])
                relative = gt_rot.transpose(-1, -2) @ rec_rot
                cosine = ((relative.diagonal(dim1=-2, dim2=-1).sum(-1) - 1) / 2).clamp(-1, 1)
                angle = torch.acos(cosine)
                sums['rotation_error_rad'] += float(angle[valid].double().sum())
                sums['rotation_values'] += int(valid.sum()) * 55

            if reconstruction_target is not None:
                reconstruction_target['motions'][start:end, 0] = (
                    reconstructed_raw.cpu().numpy()
                )

            print({'evaluated': end, 'total': len(names)}, flush=True)

    if reconstruction_target is not None:
        reconstruction_target.create_dataset('lengths', data=lengths)
        reconstruction_target.create_dataset('names', data=np.asarray(names, dtype='S'))
        reconstruction_target.create_dataset(
            'completed_repeats', data=np.ones(1, dtype=np.bool_)
        )
        reconstruction_target.attrs['requests'] = os.path.abspath(args.requests)
        reconstruction_target.attrs['requests_sha256'] = sha256_file(args.requests)
        reconstruction_target.attrs['checkpoint'] = os.path.abspath(args.checkpoint)
        reconstruction_target.attrs['checkpoint_sha256'] = sha256_file(args.checkpoint)
        reconstruction_target.attrs['source'] = 'HHC VQ encode-decode reconstruction'
        reconstruction_target.attrs['generation_length_policy'] = 'exact_request_length'
        reconstruction_target.attrs['input_motion_rep'] = args.input_motion_rep
        reconstruction_target.attrs['output_motion_rep'] = args.output_motion_rep
        reconstruction_target.close()
        reconstruction_target = None
        os.replace(reconstruction_partial, os.path.abspath(args.reconstruction_h5))

    usage = []
    for channel, label in enumerate(('body', 'hand', 'root')):
        counts = code_counts[channel]
        usage.append({
            'part': label,
            'active_codes': int((counts > 0).sum()),
            'active_fraction': float((counts > 0).sum() / model.num_code),
            'perplexity': entropy_perplexity(counts),
            'tokens': int(counts.sum()),
        })
    result = {
        'checkpoint': os.path.abspath(args.checkpoint),
        'checkpoint_sha256': sha256_file(args.checkpoint),
        'checkpoint_iteration': int(checkpoint.get('iter', -1)),
        'temporal_stride': model.temporal_stride,
        'fk_skeleton_contract': 'absolute_subject_joints_nonmutating_v2',
        'input_motion_rep': args.input_motion_rep,
        'output_motion_rep': args.output_motion_rep,
        'samples': int(len(names)),
        'frames': int(sums['feature_values'] // 672),
        'mae_raw_features': sums['absolute_error'] / sums['feature_values'],
        'rmse_raw_features': math.sqrt(sums['squared_error'] / sums['feature_values']),
        'mpjpe_m': sums['joint_error_m'] / sums['joint_values'],
        'body_mpjpe_m': sums['body_joint_error_m'] / sums['body_joint_values'],
        'hand_mpjpe_m': sums['hand_joint_error_m'] / sums['hand_joint_values'],
        'root_translation_error_m': (
            sums['root_translation_error_m'] / sums['root_translation_values']
        ),
        'mean_rotation_error_rad': sums['rotation_error_rad'] / sums['rotation_values'],
        'codebook_size_per_part': int(model.num_code),
        'codebook_usage': usage,
        'requests': os.path.abspath(args.requests),
        'requests_sha256': sha256_file(args.requests),
        'ground_truth': os.path.abspath(args.ground_truth),
        'ground_truth_sha256': sha256_file(args.ground_truth),
        'checkpoint_args': {
            key: saved_args[key] for key in (
                'nb_code', 'code_dim', 'down_t', 'stride_t', 'width', 'depth'
            )
        },
        'reconstruction_h5': (
            os.path.abspath(args.reconstruction_h5)
            if args.reconstruction_h5 is not None else None
        ),
        'reconstruction_h5_sha256': (
            sha256_file(args.reconstruction_h5)
            if args.reconstruction_h5 is not None else None
        ),
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
