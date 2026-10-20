#!/usr/bin/env python
"""Render deterministic same-request qualitative grids for all fair methods."""

import argparse
import hashlib
import json
import os
import sys
import textwrap
from pathlib import Path

import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.score_fair_interx_generations import (
    sha256_file, validate_generation_file,
)
from tools.score_fair_interx_physical import groups_from_names
from utils.fk import FK_Pair_Layer_Batch
from utils.paramUtil import INTERX_SKELETONS, SMPLX_TREE


COLORS = ('#2468B4', '#D94B40')
DISPLAY_NAMES = {
    'hhc_historical_plus_paper_mask_ratio_no_contact_common50': (
        'HHC-M (no timing)'
    ),
    'hhc_historical_plus_paper_mask_ratio_contact_common50': (
        'HHC-M (oracle timing)'
    ),
    # Retained for rendering archived figures only.
    'hhc_formal_raw12288_repaired_mgt_common50_no_contact': (
        'HHC-M raw12288 (no timing)'
    ),
    'hhc_formal_raw12288_repaired_mgt_common50_contact': (
        'HHC-M raw12288 (oracle timing)'
    ),
    'intermask_official_exact_common50': 'InterMask (50 steps)',
    'interact2ar_official_fair_no_gt_first_frame': (
        'Interact2AR (fair, no GT frame)'
    ),
    'coshmdm_interx_retrained_2daycap_seed123': 'CoShMDM (2-day cap)',
    'timotion_transformer_lpa_interx_retrained_fair': 'TiMotion',
}


def artifact(path):
    path = path.resolve()
    return {
        'path': str(path), 'bytes': path.stat().st_size,
        'sha256': sha256_file(path),
    }


def atomic_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f'.tmp.{os.getpid()}')
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + '\n', encoding='utf-8'
    )
    os.replace(temporary, path)


def draw_person(ax, joints, color):
    ax.scatter(
        joints[:, 0], joints[:, 2], joints[:, 1],
        s=2.0, c=color, depthshade=False,
    )
    for child, parent in enumerate(SMPLX_TREE):
        if parent < 0:
            continue
        segment = joints[[int(parent), child]]
        ax.plot(
            segment[:, 0], segment[:, 2], segment[:, 1],
            color=color, linewidth=0.65,
        )


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--selection', type=Path,
        default=Path('configs/qualitative_comparison_selection.json'),
    )
    parser.add_argument('--fair-root', type=Path, default=Path('output/fair_eval'))
    parser.add_argument(
        '--output-dir', type=Path,
        default=Path('output/qualitative_comparison'),
    )
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--limit', type=int)
    parser.add_argument(
        '--allow-missing-request-hash-smoke', action='store_true',
        help='Permit a legacy HDF5 without requests_sha256 for renderer smoke only.',
    )
    args = parser.parse_args()

    selection_path = args.selection.resolve()
    selection = json.loads(selection_path.read_text(encoding='utf-8'))
    if (selection.get('schema')
            != 'hhc_qualitative_comparison_selection_v1'
            or selection.get('status') != 'pass'):
        raise ValueError('qualitative selection manifest is invalid')
    request_path = Path(selection['requests']).resolve()
    if sha256_file(request_path) != selection.get('requests_sha256'):
        raise ValueError('qualitative selection request hash is stale')

    with np.load(request_path, allow_pickle=False) as request:
        names = request['names'].astype(str)
        lengths = request['lengths'].astype(np.int64)
    records = selection['records']
    if args.limit is not None:
        records = records[:args.limit]
    indices = np.asarray([record['request_index'] for record in records])
    for record, index in zip(records, indices):
        if (record['name'] != names[index]
                or int(record['length']) != int(lengths[index])):
            raise ValueError(f'stale selection record: {record["category"]}')

    device = torch.device(args.device)
    skeletons = torch.from_numpy(np.asarray(INTERX_SKELETONS).copy()).float()
    fk = FK_Pair_Layer_Batch(
        SMPLX_TREE, skeletons[:, 0], skeletons[:, 1], device=device
    ).to(device).eval()
    selected_names = names[indices]
    selected_lengths = lengths[indices]
    groups = groups_from_names(selected_names, device)

    method_joints = {}
    source_artifacts = {}
    for method in selection['methods']:
        generation_path = (
            args.fair_root / method / 'replication_00_main.h5'
        ).resolve()
        with h5py.File(generation_path, 'r') as source:
            validate_generation_file(source, names, lengths, 1, method)
            request_hash = source.attrs.get('requests_sha256')
            if (request_hash is None
                    and args.allow_missing_request_hash_smoke):
                pass
            elif request_hash != selection['requests_sha256']:
                raise ValueError(f'{method} generation request hash mismatch')
            motion = torch.from_numpy(
                np.stack([
                    source['motions'][int(index), 0] for index in indices
                ])
            ).float().to(device).reshape(len(indices), 150, 56, 12)
        if not torch.isfinite(motion).all():
            raise ValueError(f'{method} selected motions contain non-finite values')
        joints1, joints2 = fk(motion, groups)
        method_joints[method] = (
            joints1.cpu().numpy(), joints2.cpu().numpy()
        )
        source_artifacts[method] = artifact(generation_path)

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_records = []
    fractions = selection['view_contract']['frame_positions']
    for case_index, record in enumerate(records):
        length = int(selected_lengths[case_index])
        frames = np.rint(
            np.asarray(fractions, dtype=np.float64) * (length - 1)
        ).astype(np.int64)
        points = np.concatenate([
            np.concatenate((joints1[case_index, frames],
                            joints2[case_index, frames]), axis=1)
            for joints1, joints2 in method_joints.values()
        ], axis=0)
        horizontal = points[..., (0, 2)]
        center = horizontal.reshape(-1, 2).mean(axis=0)
        radius = max(
            float(np.abs(horizontal - center).max()) * 1.08, 0.75
        )
        vertical_min = float(points[..., 1].min())
        vertical_max = float(points[..., 1].max())
        vertical_center = 0.5 * (vertical_min + vertical_max)
        vertical_radius = max(0.5 * (vertical_max - vertical_min) * 1.08, 0.75)

        method_count = len(selection['methods'])
        fig = plt.figure(figsize=(20, 17), constrained_layout=False)
        figure_left = 0.12
        figure_right = 0.995
        figure_bottom = 0.015
        figure_top = 0.93
        fig.subplots_adjust(
            left=figure_left, right=figure_right,
            bottom=figure_bottom, top=figure_top,
            wspace=0.01, hspace=0.025,
        )
        for row, method in enumerate(selection['methods']):
            row_y = figure_top - (
                (row + 0.5) * (figure_top - figure_bottom) / method_count
            )
            fig.text(
                0.008, row_y, DISPLAY_NAMES.get(method, method),
                va='center', ha='left', fontsize=9,
            )
            joints1, joints2 = method_joints[method]
            for column, frame in enumerate(frames):
                ax = fig.add_subplot(
                    len(selection['methods']), len(frames),
                    row * len(frames) + column + 1,
                    projection='3d',
                )
                draw_person(ax, joints1[case_index, frame], COLORS[0])
                draw_person(ax, joints2[case_index, frame], COLORS[1])
                ax.set_xlim(center[0] - radius, center[0] + radius)
                ax.set_ylim(center[1] - radius, center[1] + radius)
                ax.set_zlim(
                    vertical_center - vertical_radius,
                    vertical_center + vertical_radius,
                )
                ax.set_box_aspect((1, 1, 1))
                ax.view_init(elev=18, azim=-70)
                ax.set_axis_off()
                if row == 0:
                    ax.set_title(f'frame {int(frame)}', fontsize=8)
        title = (
            f'{record["category"]} | {record["name"]} | '
            f'{record["gender"][0]}/{record["gender"][1]} | '
            f'{record["length"]} frames\n'
            + '\n'.join(textwrap.wrap(record['caption'], width=130))
        )
        fig.suptitle(title, fontsize=10)
        figure_path = output_dir / f'{case_index:02d}_{record["category"]}_{record["name"]}.png'
        fig.savefig(figure_path, dpi=120, facecolor='white')
        plt.close(fig)
        figure_records.append({
            'category': record['category'], 'name': record['name'],
            'request_index': int(indices[case_index]),
            'frames': frames.tolist(), 'figure': artifact(figure_path),
        })
        print({'rendered': str(figure_path)}, flush=True)

    manifest = {
        'schema': 'hhc_fair_interx_qualitative_render_v1',
        'status': 'complete',
        'selection': artifact(selection_path),
        'requests': artifact(request_path),
        'implementation': artifact(Path(__file__)),
        'methods': selection['methods'],
        'source_generations': source_artifacts,
        'same_request_repeat_camera_contract_verified': True,
        'strict_request_hashes_verified': not args.allow_missing_request_hash_smoke,
        'render_runtime': {
            'device': str(device),
            'matplotlib_backend': matplotlib.get_backend(),
            'matplotlib_version': matplotlib.__version__,
            'numpy_version': np.__version__,
            'torch_version': torch.__version__,
        },
        'figures': figure_records,
    }
    atomic_json(output_dir / 'manifest.json', manifest)
    print(json.dumps({
        'manifest': str(output_dir / 'manifest.json'),
        'figures': len(figure_records), 'status': 'complete',
    }))


if __name__ == '__main__':
    main()
