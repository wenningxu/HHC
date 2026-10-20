"""Materialize native SMPL data in an Inter-X-style paired layout and audit it."""
import argparse
import hashlib
import json
from pathlib import Path
import pickle
import shutil
import sys

import h5py
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dataset.interhuman_smpl_pair import SCHEMA, pack_raw, unpack_raw, model_view, body192_from_motion
from dataset.interhuman_hhc_adapter import OfficialInterHumanAdapter
from tools.reconstruct_interhuman_raw_body import load_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--limit', type=int, default=0, help='0=all; nonzero makes an explicitly partial pilot')
    args = parser.parse_args()
    source = ROOT/'dataset/interhuman_official'
    download_path = ROOT/'output/interhuman_raw_download_20261003.json'
    download = json.loads(download_path.read_text())
    assert download['status'] == 'complete' and download['verified_count'] == 7779
    if shutil.disk_usage(args.output.parent).free < (3*1024**3 if not args.limit else 256*1024**2):
        raise RuntimeError('Need 3 GiB workspace for full lossless dataset build')
    args.output.mkdir(exist_ok=False)
    (args.output/'split').mkdir()
    for split in ('train', 'val', 'test'):
        shutil.copyfile(source/'split'/f'{split}.txt', args.output/'split'/f'{split}.txt')
    (args.output/'texts_processed').symlink_to(source/'annots', target_is_directory=True)
    train_ids = set((source/'split/train.txt').read_text().splitlines())
    model, adapter = load_model(), OfficialInterHumanAdapter()
    files = download['files'][:args.limit] if args.limit else download['files']
    sums = np.zeros(276, np.float64)
    squares = np.zeros(276, np.float64)
    count, restored_arrays = 0, 0
    records = []
    pending = args.output/'motions.h5.partial'
    with h5py.File(pending, 'x') as handle:
        handle.attrs.update(schema=SCHEMA, status='building', model_type='smpl', body_model_gender='female',
                            model_fps=30.0, rotation_6d='first_two_rows_HHC', velocity='backward_metres_per_frame_first_zero',
                            coordinates='native_raw_no_centering', source=str(source))
        for index, entry in enumerate(files, 1):
            payload = (source/'motions'/entry['name']).read_bytes()
            assert len(payload) == int(entry['size']) and hashlib.md5(payload).hexdigest() == entry['md5Checksum']
            raw = pickle.loads(payload)
            name = Path(entry['name']).stem
            axis, beta = pack_raw(raw)
            group = handle.create_group(name)
            group.attrs.update(source_md5=entry['md5Checksum'], source_fps=raw['mocap_framerate'],
                               source_frames=raw['frames'], source_gender=json.dumps([raw[p]['gender'] for p in ('person1','person2')]))
            group.create_dataset('axis_angle_full', data=axis, compression='lzf')
            group.create_dataset('betas', data=beta)
            restored = unpack_raw(group['axis_angle_full'][:], group['betas'][:])
            for person in ('person1', 'person2'):
                for key, array in restored[person].items():
                    expected = raw[person][key]
                    assert array.dtype == expected.dtype and array.shape == expected.shape
                    assert array.tobytes() == expected.tobytes(), (name, person, key)
                    restored_arrays += 1
            if len(axis) == 0:
                group.attrs['status'] = 'empty_source'
                records.append({'id': name, 'status': 'empty_source'})
                continue
            motion, indices = model_view(axis, raw['mocap_framerate'])
            with torch.no_grad():
                body = body192_from_motion(motion, torch.from_numpy(beta), model).numpy()
            joints = np.concatenate([p[:, :66].reshape(len(motion), 22, 3) for p in body], axis=-1)
            velocities = np.zeros_like(joints)
            velocities[1:] = joints[1:]-joints[:-1]
            group.create_dataset('frame_indices30', data=indices)
            group.create_dataset('motion30', data=motion.numpy(), compression='lzf')
            group.create_dataset('joints30', data=joints, compression='lzf')
            group.create_dataset('joint_velocities30', data=velocities, compression='lzf')
            assert group['motion30'][:].tobytes() == motion.numpy().tobytes()
            official = []
            for person in ('person1', 'person2'):
                target = np.load(source/'motions_processed'/person/f'{name}.npy')
                official.append(np.concatenate([target[:, :66], target[:, 186:312]], axis=-1))
            official = np.stack(official)
            assert body.shape == official.shape
            error = np.linalg.norm((body[:, :, :66]-official[:, :, :66]).reshape(2, -1, 22, 3), axis=-1)
            generated262 = np.stack(adapter.process_loaded_pair(*body, normalize=False))
            official262 = np.stack(adapter.process_loaded_pair(*official, normalize=False))
            row = {'id': name, 'status': 'complete', 'frames30': len(motion),
                   'body_mpjpe_mm': float(error.mean()*1000), 'max_joint_error_mm': float(error.max()*1000),
                   'rotation_max': float(abs(body[:, :, 66:]-official[:, :, 66:]).max()),
                   'online_position_max_m': float(abs(generated262[:, :, :66]-official262[:, :, :66]).max()),
                   'contact_differences': int(np.count_nonzero(generated262[:, :, 258:] != official262[:, :, 258:])),
                   'contact_values': int(generated262[:, :, 258:].size)}
            records.append(row)
            if name in train_ids:
                flat = motion.numpy().reshape(len(motion), 276).astype(np.float64)
                sums += flat.sum(0)
                squares += np.square(flat).sum(0)
                count += len(motion)
            group.attrs['status'] = 'complete'
            if index % 250 == 0:
                handle.flush()
                print(f'packed/roundtrip/eval {index}/{len(files)}', flush=True)
        assert count > 0
        mean = sums/count
        std = np.sqrt(np.maximum(squares/count-mean*mean, 0))
        std[std < 1e-6] = 1.0
        np.savez(args.output/'normalization.npz', mean=mean.astype(np.float32), std=std.astype(np.float32),
                 count=count, source_split='train_only')
        handle.attrs['status'] = 'complete' if not args.limit else 'pilot_complete'
    pending.rename(args.output/'motions.h5')
    good = [r for r in records if r['status'] == 'complete']
    summary = {'status': 'complete' if not args.limit else 'pilot_complete', 'files': len(records),
               'valid_pairs': len(good), 'empty_ids': [r['id'] for r in records if r['status']=='empty_source'],
               'raw_arrays_bitwise_restored': restored_arrays, 'train_stat_frames': count,
               'body_mpjpe_mm_mean': float(np.mean([r['body_mpjpe_mm'] for r in good])),
               'max_joint_error_mm': max(r['max_joint_error_mm'] for r in good),
               'rotation_max': max(r['rotation_max'] for r in good),
               'online_position_max_m': max(r['online_position_max_m'] for r in good),
               'contact_differences': sum(r['contact_differences'] for r in good),
               'contact_values': sum(r['contact_values'] for r in good)}
    bound = [Path(__file__), ROOT/'dataset/interhuman_smpl_pair.py', ROOT/'tools/reconstruct_interhuman_raw_body.py',
             ROOT/'body_models/smpl/SMPL_FEMALE.pkl', download_path]
    manifest = {'schema': SCHEMA, 'summary': summary, 'records': records,
                'sources': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in bound},
                'notes': ['Raw full-rate arrays preserved bitwise; 30FPS is a view, not a claim of lossless downsampling.',
                          'Shape applied by native female SMPL; source gender metadata unchanged.',
                          'No GT residual/contact labels used in generated evaluation conversion.',
                          'Old Inter-X normalizer, SMPL-X FK and 524D InterHuman checkpoints are incompatible.']}
    (args.output/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
