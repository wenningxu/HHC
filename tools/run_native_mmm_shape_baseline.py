"""Prepare matched physical inputs, then run upstream MMM in an isolated process."""
import argparse
import hashlib
import inspect
import json
import os
import pickle
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MMM = Path('/home/wenning/projects/MMM')


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input-dir', type=Path, required=True)
    p.add_argument('--output-dir', type=Path)
    p.add_argument('--candidate-dir', type=Path)
    p.add_argument('--floor-report', type=Path)
    p.add_argument('--contact-dir', type=Path)
    p.add_argument('--limit', type=int, default=25)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--worker', action='store_true')
    a = p.parse_args()
    import numpy as np
    if not a.worker:
        sys.path.insert(0, str(ROOT))
        import h5py
        from tools.run_interx_shape_augmentation import source_person_payload, atomic_npz, atomic_json, canonical_sha256
        from utils.interx_physical_ground import place_pair_on_mesh_ground
        assert a.limit > 0 and not a.input_dir.exists()
        mp = a.candidate_dir / '_completion_manifest.json'
        m = json.loads(mp.read_text()); f = json.loads(a.floor_report.read_text())
        assert m['complete'] and f['completion_sha256'] == sha(mp)
        assert sha(m['source_motion']) == m['source_motion_sha256']
        assert sha(m['plan_records']) == m['plan_records_sha256']
        records = {r['augmented_name']: r for r in map(json.loads, Path(m['plan_records']).read_text().splitlines())}
        names = m['completed_names'][:a.limit]; assert len(names) == a.limit
        floors = {r['name']: r for r in f['rows'] if r['condition'] == 'new_shape'}
        a.input_dir.mkdir(parents=True); rows = []
        with h5py.File(m['source_motion'], 'r') as source:
            for name in names:
                sp = a.candidate_dir / f'{name}.json'; s = json.loads(sp.read_text()); record = records[name]
                assert floors[name]['sidecar_sha256'] == sha(sp) and canonical_sha256(record) == s['plan_record_sha256']
                raw = source[s['source_name']][s['crop_start']:s['crop_stop']]
                people = place_pair_on_mesh_ground([source_person_payload(raw, i, s['augmented_shapes'][i]) for i in range(2)],
                                                  [r['mesh_floor_raw_m'] for r in floors[name]['people']])
                cp = (a.contact_dir / f'{s["source_name"]}.pkl').resolve()
                assert sha(cp) == s['contact_pairs_sha256'] and sha(record['timing_file']) == record['timing_sha256']
                payload = {f'p{i}_{k}': v for i, person in enumerate(people) for k, v in person.items()}
                payload['timing'] = np.load(record['timing_file'])[s['crop_start']:s['crop_stop']]
                assert len(payload['timing']) == len(raw)
                dest = a.input_dir / f'{name}.npz'; atomic_npz(dest, payload)
                with np.load(dest, allow_pickle=False) as z:
                    assert all(np.array_equal(z[k], v) for k, v in payload.items())
                rows.append({'name': name, 'input_sha256': sha(dest), 'contact_path': str(cp),
                             'contact_sha256': sha(cp), 'input_sidecar_sha256': sha(sp)})
        atomic_json(a.input_dir / 'manifest.json', {'rows': rows, 'floor_report_sha256': sha(a.floor_report),
                    'completion_sha256': sha(mp), 'scope': 'Identical physical new-shape inputs; no original example joint-floor preprocessing.'})
        print(f'Prepared and read-back verified {len(rows)} inputs', flush=True)
        return

    # No HHC imports in this branch: all optimizer dependencies resolve upstream.
    sys.path.insert(0, str(MMM)); os.chdir(MMM)
    import torch
    import optimize_full_body as native
    from optimization.human_body_prior.body_model.body_model import BodyModel
    dependencies = [Path(inspect.getfile(x)).resolve() for x in (native.optimize_full_body, native.DyadicInteractionLoss, BodyModel)]
    assert all(path.is_relative_to(MMM) for path in dependencies), dependencies
    assert a.output_dir is not None and not a.output_dir.exists()
    a.output_dir.mkdir(parents=True)
    manifest = json.loads((a.input_dir / 'manifest.json').read_text()); rows = []
    for item in manifest['rows']:
        path = a.input_dir / f'{item["name"]}.npz'
        assert sha(path) == item['input_sha256'] and sha(item['contact_path']) == item['contact_sha256']
        with np.load(path, allow_pickle=False) as z:
            people = [{k[3:]: (str(z[k].item()) if k == f'p{i}_gender' else z[k]) for k in z.files if k.startswith(f'p{i}_')} for i in range(2)]
            timing = torch.as_tensor(z['timing'], dtype=torch.float32, device=a.device)
        with open(item['contact_path'], 'rb') as f:
            contacts = pickle.load(f)
        torch.manual_seed(1234); torch.cuda.synchronize(a.device); started = time.perf_counter()
        refined = native.optimize_full_body(*people, temporal_mask=timing, interaction_pairs=contacts, num_epochs=200, lr=.01, device=a.device)
        torch.cuda.synchronize(a.device); elapsed = time.perf_counter() - started
        hashes = []
        for i, person in enumerate(refined):
            assert np.array_equal(person['betas'], people[i]['betas'])
            assert all(np.isfinite(v).all() for k, v in person.items() if k != 'gender')
            person['gender'] = people[i]['gender']
            dest = a.output_dir / f'{item["name"]}_p{i+1}.npz'
            np.savez_compressed(dest, **person); hashes.append(sha(dest))
        rows.append({**item, 'common_export_origin': [0., 0., 0.], 'output_npz_sha256': hashes, 'optimizer_call_seconds': elapsed})
        print(f'Native MMM completed {len(rows)}/{len(manifest["rows"])}', flush=True)
    report = {'status': 'ground_diagnostic_complete_not_acceptance', 'scope': manifest['scope'] + ' Native MMM function/loss/BodyModel; output remains physical world coordinates, zero decode offset.',
              'floor_report_sha256': manifest['floor_report_sha256'], 'input_manifest_sha256': sha(a.input_dir / 'manifest.json'),
              'optimizer': {'iterations': 200, 'learning_rate': .01, 'seed': 1234, 'ground_policy': 'original_relu_vertices_weight20'},
              'implementation_sha256': {str(path): sha(path) for path in dependencies + [Path(__file__)]}, 'rows': rows}
    # ponytail: small diagnostic is not resumable; interrupted outputs remain unaccepted and are preserved.
    (a.output_dir / 'report.json').write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
