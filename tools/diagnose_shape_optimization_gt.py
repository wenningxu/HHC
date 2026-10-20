"""Paired original-GT shape diagnostic; nearest-vertex penetration is a proxy."""
import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval.calc_pene_from_npz import PeneCalculator, npz_to_amass_dict, to_torch_params
from tools.run_interx_shape_augmentation import source_person_payload, sha256_file, atomic_json
from tools.score_volumetric_smpl_npz import foot_sliding


@torch.no_grad()
def measure(calculator, people, frames, gt_masks=None):
    geometries = []
    for person in people:
        assert len(person['trans']) == len(person['pose_body']) == frames
        geometries.append(calculator._compute_geometry(to_torch_params(person, calculator.device)))
    (va, ja, na), (vb, jb, nb) = geometries
    pene = float(calculator._calc_penetration_metric_per_frame(va, vb, na, nb).mean())
    sliding = [foot_sliding(joints) for joints in (ja, jb)]
    count = sum(item['contact_observations'] for item in sliding)
    value = sum(item['mean_horizontal_speed_m_per_frame'] * item['contact_observations']
                for item in sliding) / max(count, 1)
    feet = [7, 10, 8, 11]
    if gt_masks is None:
        gt_masks = [joints[:-1, feet, 1] - joints[:, :55, 1].min()
                    < joints.new_tensor([.12, .05, .12, .05]) for joints in (ja, jb)]
    speeds = []
    for joints, mask in zip((ja, jb), gt_masks):
        speed = torch.linalg.vector_norm((joints[1:, feet] - joints[:-1, feet])[..., [0, 2]], dim=-1)
        speeds.append(speed[mask])
    fixed = torch.cat(speeds)
    assert fixed.numel() > 0, 'No original-GT grounded feet; cannot assess skating'
    return {'nearest_vertex_squared_sum_m2_per_frame': pene,
            'foot_sliding_m_per_frame': value, 'grounded_observations': count,
            'gt_mask_horizontal_speed_m_per_frame': float(fixed.mean())}, gt_masks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--condition', action='append', required=True, help='label=output_directory')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    folders = dict(item.split('=', 1) for item in args.condition)
    assert len(folders) == len(args.condition) and not {'original_gt', 'random_shape'} & folders.keys()
    folders = {key: Path(value) for key, value in folders.items()}
    contracts = {key: json.loads((folder / '_completion_manifest.json').read_text())
                 for key, folder in folders.items()}
    completion_hashes = {key: sha256_file(folder / '_completion_manifest.json')
                         for key, folder in folders.items()}
    contract = next(iter(contracts.values()))
    for other in contracts.values():
        assert other['complete'] and other['source_motion_sha256'] == contract['source_motion_sha256']
        assert other['completed_names'] == contract['completed_names']
    assert sha256_file(contract['source_motion']) == contract['source_motion_sha256']
    calculator = PeneCalculator(args.device)
    rows = []
    with h5py.File(contract['source_motion'], 'r') as source:
        for name in contract['completed_names']:
            metadata = json.loads((next(iter(folders.values())) / f'{name}.json').read_text())
            raw = source[metadata['source_name']][metadata['crop_start']:metadata['crop_stop']]
            frames = metadata['crop_stop'] - metadata['crop_start']
            assert raw.shape == (frames, 56, 6) and frames > 1
            shapes = metadata['source_shapes']
            original = [source_person_payload(raw, i, {'betas': shapes['betas'][i],
                        'gender': shapes['genders'][i]}) for i in (0, 1)]
            donors = sorted(metadata['augmented_shapes'], key=lambda item: item['target_person'])
            random = [source_person_payload(raw, i, donors[i]) for i in (0, 1)]
            assert any(not np.array_equal(original[i]['betas'], random[i]['betas']) for i in (0, 1))
            gt, masks = measure(calculator, original, frames)
            conditions = {'original_gt': gt, 'random_shape': measure(calculator, random, frames, masks)[0]}
            input_bindings = {}
            for label, folder in folders.items():
                other = json.loads((folder / f'{name}.json').read_text())
                input_bindings[label] = {'sidecar_sha256': sha256_file(folder / f'{name}.json'),
                                        'p1_sha256': other['output_p1_sha256'],
                                        'p2_sha256': other['output_p2_sha256']}
                for key in ('source_name', 'crop_start', 'crop_stop', 'source_shapes', 'augmented_shapes'):
                    assert other[key] == metadata[key], (name, key)
                people = []
                for i in (1, 2):
                    path = folder / f'{name}_p{i}.npz'
                    assert sha256_file(path) == other[f'output_p{i}_sha256']
                    with np.load(path, allow_pickle=False) as payload:
                        person = npz_to_amass_dict(payload)
                    assert np.array_equal(person['betas'], random[i-1]['betas'])
                    assert person['gender'] == random[i-1]['gender']
                    people.append(person)
                conditions[label] = measure(calculator, people, frames, masks)[0]
            passes = {label: all(values[key] <= gt[key] for key in (
                'foot_sliding_m_per_frame', 'nearest_vertex_squared_sum_m2_per_frame'))
                for label, values in conditions.items() if label != 'original_gt'}
            rows.append({'name': name, 'frames': frames, 'conditions': conditions,
                         'input_bindings': input_bindings,
                         'proxy_pass_vs_original_gt': passes})
            print(name, conditions, passes, flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output, {'status': 'diagnostic_only_nearest_vertex_proxy',
                'frames': 'exact metadata crop, no implicit truncation',
                'foot_metric': 'raw horizontal speed, per-person floor from 55 skeleton joints',
                'gt_mask_metric': 'original-GT foot contacts frozen across conditions',
                'source_motion_sha256': contract['source_motion_sha256'],
                'condition_completion_sha256': completion_hashes,
                'implementation_sha256': {str(path.resolve()): sha256_file(path) for path in (
                    Path(__file__), Path('eval/calc_pene_from_npz.py'),
                    Path('tools/score_volumetric_smpl_npz.py'),
                    Path('optimization/human_body_prior/body_model/body_model.py'))},
                'condition_directories': {key: str(value.resolve()) for key, value in folders.items()},
                'rows': rows})


if __name__ == '__main__':
    main()
