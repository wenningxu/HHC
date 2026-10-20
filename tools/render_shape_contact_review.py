"""Bound three-condition skeleton storyboards; visual aid, not a semantic gate."""
import argparse
import json
import sys
import textwrap
from pathlib import Path

import h5py
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.render_fair_interx_qualitative import draw_person, COLORS, plt
from tools.run_interx_shape_augmentation import sha256_file, source_person_payload, atomic_json
from eval.calc_pene_from_npz import PeneCalculator, to_torch_params, npz_to_amass_dict
from utils.interx_physical_ground import place_pair_on_mesh_ground, decode_physical_pair


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--candidate-dir', type=Path, required=True)
    p.add_argument('--floor-report', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--name', action='append', required=True)
    a = p.parse_args()
    assert not a.output_dir.exists() and len(set(a.name)) == len(a.name)
    mp = a.candidate_dir / '_completion_manifest.json'
    m = json.loads(mp.read_text()); f = json.loads(a.floor_report.read_text())
    assert m['complete'] and f['completion_sha256'] == sha256_file(mp)
    assert sha256_file(m['source_motion']) == m['source_motion_sha256']
    floors = {(r['name'], r['condition']): r for r in f['rows']}
    calc = PeneCalculator('cpu'); results = []
    a.output_dir.mkdir(parents=True)
    with h5py.File(m['source_motion'], 'r') as source:
        for name in a.name:
            assert name in m['completed_names']
            sp = a.candidate_dir / f'{name}.json'; s = json.loads(sp.read_text())
            raw = source[s['source_name']][s['crop_start']:s['crop_stop']]
            frames = np.unique(np.linspace(0, len(raw) - 1, min(6, len(raw)), dtype=int))
            pairs = {}
            for label in ('original_shape', 'new_shape'):
                floor = floors[name, label]
                assert floor['sidecar_sha256'] == sha256_file(sp)
                donors = ([{'betas': s['source_shapes']['betas'][i], 'gender': s['source_shapes']['genders'][i]}
                           for i in range(2)] if label == 'original_shape' else s['augmented_shapes'])
                pairs[label] = place_pair_on_mesh_ground(
                    [source_person_payload(raw, i, donors[i]) for i in range(2)],
                    [r['mesh_floor_raw_m'] for r in floor['people']])
            loaded = []
            for i in (1, 2):
                path = a.candidate_dir / f'{name}_p{i}.npz'
                assert sha256_file(path) == s[f'output_p{i}_sha256']
                with np.load(path, allow_pickle=False) as z:
                    loaded.append(npz_to_amass_dict(dict(z)))
            pairs['optimized'] = decode_physical_pair(loaded, s['diagnostics']['ground_alignment']['common_export_origin'])
            joints = {label: [calc._compute_geometry(to_torch_params(person, 'cpu'), compute_normals=False)[1][frames, :55].numpy()
                              for person in pair] for label, pair in pairs.items()}
            cloud = np.concatenate([j.reshape(-1, 3) for pair in joints.values() for j in pair])
            assert np.isfinite(cloud).all()
            center = (cloud.min(0) + cloud.max(0)) / 2
            radius = max(float(np.ptp(cloud, axis=0).max()) / 2, .1) * 1.1
            for azimuth in (-60, 30):
                fig = plt.figure(figsize=(18, 10))
                for row, (label, pair) in enumerate(joints.items()):
                    for col, frame in enumerate(frames):
                        ax = fig.add_subplot(3, len(frames), row * len(frames) + col + 1, projection='3d')
                        for i in range(2):
                            draw_person(ax, pair[i][col], COLORS[i])
                        ax.set(xlim=(center[0]-radius, center[0]+radius), ylim=(center[2]-radius, center[2]+radius),
                               zlim=(center[1]-radius, center[1]+radius), title=f'{label} | source frame {s["crop_start"] + int(frame)}')
                        ax.set_box_aspect((1, 1, 1)); ax.view_init(elev=15, azim=azimuth)
                        ax.set_axis_off()
                # ponytail: six-frame skeletons cannot establish touch or transient actions; inspect dense mesh video for ambiguous cases.
                fig.suptitle(name + '\n' + textwrap.fill(s['caption'], 130) + '\nBlue=P1, Red=P2; common physical frame; skeleton proximity is not surface contact', fontsize=11)
                fig.tight_layout(rect=(0, 0, 1, .9))
                fig.savefig(a.output_dir / f'{name}_az{azimuth}.png', dpi=120); plt.close(fig)
            results.append({'name': name, 'sidecar_sha256': sha256_file(sp), 'source_frames': (frames + s['crop_start']).tolist(), 'caption': s['caption']})
    atomic_json(a.output_dir / 'manifest.json', {'status': 'visual_aid_not_semantic_acceptance', 'completion_sha256': sha256_file(mp),
                'floor_report_sha256': sha256_file(a.floor_report), 'implementation_sha256': sha256_file(Path(__file__)), 'rows': results})


if __name__ == '__main__':
    main()
