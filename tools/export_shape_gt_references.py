"""Export exact cropped original-GT and random-shape NPZs for paired scoring."""
import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.run_interx_shape_augmentation import atomic_json, atomic_npz, sha256_file, source_person_payload
from eval.calc_pene_from_npz import npz_to_amass_dict


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    manifest = args.candidate_dir / '_completion_manifest.json'
    run = json.loads(manifest.read_text())
    assert run['complete'] and len(run['completed_names']) == run['completed_samples']
    assert sha256_file(run['source_motion']) == run['source_motion_sha256']
    # Refuse overwrite: a different candidate/crop must never reuse stale references.
    args.output_dir.mkdir(parents=True, exist_ok=False)
    for label in ('original_gt', 'random_shape'):
        (args.output_dir / label).mkdir()
    records = []
    with h5py.File(run['source_motion'], 'r') as source:
        for name in run['completed_names']:
            sidecar = args.candidate_dir / f'{name}.json'
            meta = json.loads(sidecar.read_text())
            raw = source[meta['source_name']][meta['crop_start']:meta['crop_stop']]
            frames = meta['crop_stop'] - meta['crop_start']
            assert raw.shape == (frames, 56, 6) and 1 < frames <= 150
            donors = sorted(meta['augmented_shapes'], key=lambda x: x['target_person'])
            assert [x['target_person'] for x in donors] == [0, 1]
            originals = [{'betas': meta['source_shapes']['betas'][i],
                          'gender': meta['source_shapes']['genders'][i]} for i in (0, 1)]
            assert any(not np.array_equal(originals[i]['betas'], donors[i]['betas']) for i in (0, 1))
            outputs = {}
            for i in (0, 1):
                assert sha256_file(args.candidate_dir / f'{name}_p{i+1}.npz') == meta[f'output_p{i+1}_sha256']
                channel = slice(i * 3, i * 3 + 3)
                for label, shapes in (('original_gt', originals), ('random_shape', donors)):
                    payload = {'poses': raw[:, :55, channel].astype(np.float32),
                               'trans': raw[:, 55, channel].astype(np.float32),
                               'betas': np.asarray(shapes[i]['betas'], dtype=np.float32),
                               'gender': np.asarray(shapes[i]['gender']),
                               'mocap_framerate': np.asarray(30)}
                    path = args.output_dir / label / f'{name}_p{i+1}.npz'
                    atomic_npz(path, payload)
                    with np.load(path, allow_pickle=False) as saved:
                        actual = npz_to_amass_dict(saved)
                    expected = source_person_payload(raw, i, shapes[i])
                    assert all(np.array_equal(actual[k], v) for k, v in expected.items())
                    outputs[f'{label}_p{i+1}'] = {'path': str(path.resolve()), 'sha256': sha256_file(path)}
            records.append({'name': name, 'frames': frames, 'sidecar_sha256': sha256_file(sidecar), 'outputs': outputs})
    atomic_json(args.output_dir / 'references.json', {
        'candidate_manifest': str(manifest.resolve()), 'candidate_manifest_sha256': sha256_file(manifest),
        'source_motion_sha256': run['source_motion_sha256'], 'records': records})
    print({'exported_pairs_per_condition': len(records), 'roundtrip_check': 'passed'})


if __name__ == '__main__':
    main()
