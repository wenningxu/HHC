#!/usr/bin/env python
"""Freeze deterministic Inter-X cases for cross-method qualitative figures."""

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.score_fair_interx_generations import sha256_file


ACTION_KEYWORDS = {
    'action_hug': ('hug', 'embrac', 'cuddl'),
    'action_high_five': ('high five', 'high-five', 'highfive'),
    'action_handshake': ('handshake', 'shake hands', 'shaking hands'),
    'action_kick': ('kick',),
}
GENDER_LABELS = {0: 'male', 1: 'female'}
METHODS = [
    'hhc_historical_plus_paper_mask_ratio_no_contact_common50',
    'hhc_historical_plus_paper_mask_ratio_contact_common50',
    'intermask_official_exact_common50',
    'interact2ar_official_fair_no_gt_first_frame',
    'coshmdm_interx_retrained_2daycap_seed123',
    'timotion_transformer_lpa_interx_retrained_fair',
]


def ranked_choice(indices, category, names, captions, seed, excluded):
    candidates = [int(index) for index in indices if int(index) not in excluded]
    if not candidates:
        raise ValueError(f'no unused candidates for qualitative category {category}')
    return min(candidates, key=lambda index: hashlib.sha256(
        f'{seed}|{category}|{names[index]}|{captions[index]}'.encode('utf-8')
    ).hexdigest())


def atomic_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f'.tmp.{os.getpid()}')
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + '\n', encoding='utf-8'
    )
    os.replace(temporary, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--requests', type=Path,
        default=Path('output/fair_interx_requests/replication_00.npz'),
    )
    parser.add_argument(
        '--output', type=Path,
        default=Path('configs/qualitative_comparison_selection.json'),
    )
    parser.add_argument('--seed', type=int, default=20260826)
    args = parser.parse_args()

    request_path = args.requests.resolve()
    with np.load(request_path, allow_pickle=False) as request:
        names = request['names'].astype(str)
        captions = request['captions'].astype(str)
        lengths = request['lengths'].astype(np.int64)
        gender = request['gender'].astype(np.int64)
        betas = request['betas'].astype(np.float64).reshape(-1, 2, 10)

    selected = []
    used = set()
    for category, keywords in ACTION_KEYWORDS.items():
        candidates = [
            index for index, caption in enumerate(captions)
            if any(keyword in caption.lower() for keyword in keywords)
        ]
        index = ranked_choice(
            candidates, category, names, captions, args.seed, used
        )
        selected.append((category, index, {
            'rule': 'minimum_sha256_over_keyword_matches',
            'keywords': list(keywords),
            'candidate_count': len(candidates),
        }))
        used.add(index)

    gender_categories = {
        'gender_male_male': (0, 0),
        'gender_female_female': (1, 1),
        'gender_mixed': None,
    }
    for category, target in gender_categories.items():
        if target is None:
            candidates = np.flatnonzero(gender[:, 0] != gender[:, 1])
        else:
            candidates = np.flatnonzero(
                (gender[:, 0] == target[0]) & (gender[:, 1] == target[1])
            )
        index = ranked_choice(
            candidates, category, names, captions, args.seed, used
        )
        selected.append((category, index, {
            'rule': 'minimum_sha256_over_gender_matches',
            'gender_target': (
                'mixed' if target is None
                else [GENDER_LABELS[target[0]], GENDER_LABELS[target[1]]]
            ),
            'candidate_count': int(len(candidates)),
        }))
        used.add(index)

    beta_distance = np.linalg.norm(betas[:, 0] - betas[:, 1], axis=1)
    beta_norm = np.linalg.norm(betas, axis=2).max(axis=1)
    for category, score, rule in (
        (
            'shape_max_pair_beta_distance', beta_distance,
            'maximum_l2_distance_between_pair_beta_vectors',
        ),
        (
            'shape_max_subject_beta_norm', beta_norm,
            'maximum_subject_beta_l2_norm',
        ),
    ):
        candidates = [
            int(index) for index in np.argsort(-score, kind='stable')
            if int(index) not in used
        ]
        if not candidates:
            raise ValueError(f'no unused candidates for {category}')
        index = candidates[0]
        selected.append((category, index, {
            'rule': rule,
            'score': float(score[index]),
        }))
        used.add(index)

    records = []
    for category, index, selection in selected:
        records.append({
            'category': category,
            'request_index': int(index),
            'name': str(names[index]),
            'caption': str(captions[index]),
            'length': int(lengths[index]),
            'gender': [GENDER_LABELS[int(value)] for value in gender[index]],
            'betas': betas[index].tolist(),
            'beta_norms': np.linalg.norm(betas[index], axis=1).tolist(),
            'pair_beta_distance': float(beta_distance[index]),
            'selection': selection,
        })

    payload = {
        'schema': 'hhc_qualitative_comparison_selection_v1',
        'status': 'pass',
        'requests': str(request_path),
        'requests_sha256': sha256_file(request_path),
        'replication': 0,
        'repeat': 0,
        'selection_seed': args.seed,
        'selection_is_metric_independent': True,
        'methods': METHODS,
        'view_contract': {
            'same_request_and_repeat_across_methods': True,
            'same_camera_and_axis_limits_within_case': True,
            'frame_positions': [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        },
        'records': records,
    }
    atomic_json(args.output.resolve(), payload)
    print(json.dumps({
        'output': str(args.output.resolve()),
        'records': len(records),
        'names': [record['name'] for record in records],
    }, indent=2))


if __name__ == '__main__':
    main()
