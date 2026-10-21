"""Cloud-only main/MM export, gated by a real frozen five-candidate val selection.

One invocation exports one replication/mode. No test score or GT shape is read.
MM holds the train-only shape draw fixed per prompt while varying motion noise.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.hhc_text_only_formal_contract import (
    SAMPLING, file_sha, read_json, require, validate_selection,
)


def load_requests(receipt, mode, replication):
    require(mode in ('main', 'mm') and replication in range(20), 'Invalid mode/replication')
    manifest = read_json(receipt['main_manifest' if mode == 'main' else 'mm_manifest']['path'])
    entry = next(row for row in manifest['replications'] if row['replication'] == replication)
    path = Path(entry['path'])
    require(path.stat().st_size == entry['bytes'] and file_sha(path) == entry['sha256'],
            'Frozen request file changed')
    payload = read_json(path)
    rows = payload['requests']
    canonical = json.dumps(rows, sort_keys=True, separators=(',', ':'), ensure_ascii=False)
    require(hashlib.sha256(canonical.encode()).hexdigest()
            == payload['request_sha256'] == entry['request_sha256'], 'Request content changed')
    require(payload['replication'] == replication and len(rows) == (1056 if mode == 'main' else 100),
            'Wrong formal replication/request count')
    require(len({row['identifier'] for row in rows}) == len(rows), 'Duplicate prompt identifiers')
    require(all(14 <= row['processed_length'] <= 299 for row in rows), 'Invalid motion lengths')
    if mode == 'mm':
        require(payload['mm_repeats'] == 30 and payload['mm_pairs'] == 10, 'Wrong MM protocol')
    return rows, path, entry


def assign_shapes(rows, policy, prior=None):
    import numpy as np
    if policy == 'zero':
        return {row['identifier']: np.zeros((2, 10), np.float32) for row in rows}, {}
    require(prior is not None, 'Train prior absent')
    # Matches frozen validation. Replication randomness is already in sample_seed.
    # MM repeats deliberately reuse the same draw; only motion noise changes.
    return prior.request_betas(rows, policy, replication=SAMPLING['shape_seed_replication'])


def write_generations(destination, rows, request_path, entry, scorer, betas, metadata, mode):
    """Stream chunks without materializing repeated identifiers in the beta lookup."""
    import h5py
    import numpy as np
    import torch
    from dataset.interhuman_generated_h5 import (
        MEAN_SHA256, STD_SHA256, InterHumanGeneratedH5Dataset,
        InterHumanMMGeneratedH5Dataset, mm_repeat_seed,
    )
    destination = Path(destination)
    partial = destination.with_name(destination.stem + '.partial.h5')
    require(not destination.exists() and not partial.exists(), 'Refusing to replay/overwrite export')
    count, repeats = len(rows), 1 if mode == 'main' else 30
    beta_array = np.stack([betas[row['identifier']] for row in rows]).astype(np.float32)
    require(beta_array.shape == (count, 2, 10) and np.isfinite(beta_array).all(), 'Invalid beta assignment')
    metadata = dict(metadata, shape_assignment_sha256=hashlib.sha256(beta_array.tobytes()).hexdigest())
    seeds = np.asarray([[mm_repeat_seed(row['sample_seed'], repeat) for repeat in range(repeats)]
                        for row in rows], dtype=np.uint64)
    started = time.time()
    with h5py.File(partial, 'x') as out:
        out.attrs.update(schema='interhuman_fair_generation_v1' if mode == 'main'
                         else 'interhuman_fair_mm_generation_v1', status='running', split='test',
                         request_file_sha256=entry['sha256'], request_content_sha256=entry['request_sha256'],
                         normalization_mean_sha256=MEAN_SHA256, normalization_std_sha256=STD_SHA256,
                         sampler=SAMPLING['sampler'], sampling_steps=50, cfg_scale=3.5, temperature=.5,
                         gt_shape_used=False, shape_policy=metadata['shape_policy'],
                         metadata=json.dumps(metadata, sort_keys=True))
        strings = h5py.string_dtype('utf-8')
        out.create_dataset('names', data=[row['identifier'] for row in rows], dtype=strings)
        out.create_dataset('texts', data=[row['caption'] for row in rows], dtype=strings)
        out.create_dataset('lengths', data=[row['processed_length'] for row in rows], dtype=np.int64)
        out.create_dataset('shape_betas', data=beta_array)
        if mode == 'mm':
            out.create_dataset('repeat_seeds', data=seeds)
        shape = (count, 300, 262) if mode == 'main' else (count, 30, 300, 262)
        chunks = (1, 300, 262) if mode == 'main' else (1, 1, 300, 262)
        for key in ('motion1', 'motion2'):
            out.create_dataset(key, shape, dtype=np.float32, chunks=chunks, compression='lzf')
        with torch.inference_mode():
            for repeat in range(repeats):
                for begin in range(0, count, 4):
                    requests = rows[begin:begin + 4]
                    predicted = scorer.generate(requests, seeds[begin:begin + len(requests), repeat].tolist(), steps=50)
                    require(tuple(predicted.shape) == (len(requests), 300, 524)
                            and predicted.dtype == torch.float32 and torch.isfinite(predicted).all().item(),
                            'Invalid generated motion tensor')
                    predicted = predicted.detach().cpu().numpy()
                    for index, request in enumerate(requests):
                        require(not np.any(predicted[index, request['processed_length']:]), 'Nonzero padding')
                    for part, key in enumerate(('motion1', 'motion2')):
                        values = predicted[..., part * 262:(part + 1) * 262]
                        if mode == 'main':
                            out[key][begin:begin + len(requests)] = values
                        else:
                            out[key][begin:begin + len(requests), repeat] = values
                print(json.dumps({'mode': mode, 'repeat_complete': repeat + 1,
                                  'samples_per_repeat': count, 'elapsed_s': time.time() - started}), flush=True)
        out.attrs['status'] = 'complete'
    # Validate the exact evaluator input before publishing the final filename.
    cls = InterHumanGeneratedH5Dataset if mode == 'main' else InterHumanMMGeneratedH5Dataset
    dataset = cls(partial, request_path, request_entry=entry, formal=True)
    dataset.close()
    os.replace(partial, destination)
    return metadata


def load_cloud_model(receipt, rows):
    # This function is reached only after the full CPU receipt/source gate.
    import clip
    import torch
    from models.mgt_interhuman_smpl import SMPLMGT
    from models.mgt_interhuman_smpl_cfg import configure_conditioning
    from tools.export_hhc_mgt_train_prior_validation_cloud import TrainPriorScorer
    from tools.reconstruct_interhuman_raw_body import load_model
    from dataset.interhuman_hhc_adapter import OfficialInterHumanAdapter
    from utils.hhc_frozen_vq import load_frozen_vq
    from utils.hhc_train_shape_prior import TrainShapePrior
    import utils.interhuman_smpl_mgt as assets
    require(str(ROOT) == '/workspace/HHC' and os.environ.get('CUDA_VISIBLE_DEVICES') == '1',
            'GPU work is restricted to authorized Edmond cloud GPU1')
    import subprocess
    free = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.free',
        '--format=csv,noheader,nounits'], text=True).strip().splitlines()
    require(len(free) == 1 and int(free[0]) >= 13 * 1024, 'Single cloud GPU needs >=13 GiB free')
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    os.nice(10)
    torch.cuda.set_per_process_memory_fraction(.35, 0)
    contract = read_json(receipt['training_manifest']['path'])['contract']
    saved = torch.load(receipt['mgt_checkpoint']['path'], map_location='cpu', weights_only=False)
    require(saved['step'] == 40000 and saved['contract_sha256'] == receipt['training_contract_sha256'],
            'MGT checkpoint internal contract differs')
    native = ROOT / 'dataset/interhuman_smpl_interx_layout_v1_20261004'
    assets.DATA = native
    vq, mean, std, spec = load_frozen_vq(receipt['vq_spec']['path'], 'cuda:0', native / 'manifest.json')
    net = SMPLMGT(vq).cuda()
    loaded = net.load_state_dict(saved['model'], strict=False)
    del saved
    require(not loaded.unexpected_keys and all(key.startswith('trans_base.vqvae.') for key in loaded.missing_keys),
            'MGT state keys differ')
    net.eval().requires_grad_(False)
    configure_conditioning(net, contract['conditioning_arm'], sampling=True)
    clip_model, _ = clip.load('ViT-B/32', device='cpu', jit=False)
    clip_model.eval().requires_grad_(False)
    prior = None
    if receipt['shape_policy'] != 'zero':
        prior = TrainShapePrior(receipt['shape_prior']['path'], contract['data_sha256'],
                                contract['sources']['utils/interhuman_smpl_mgt.py'])
    betas, indices = assign_shapes(rows, receipt['shape_policy'], prior)
    scorer = TrainPriorScorer(net, vq, mean, std, clip_model, None, None,
        OfficialInterHumanAdapter(), load_model(), 'cuda:0', 'cpu', betas)
    return scorer, betas, indices, spec


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--selection-receipt', type=Path, required=True)
    parser.add_argument('--mode', choices=['main', 'mm'], required=True)
    parser.add_argument('--replication', type=int, choices=range(20), required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    require(str(ROOT) == '/workspace/HHC', 'This export is cloud only')
    receipt_sha = file_sha(args.selection_receipt)
    receipt = validate_selection(args.selection_receipt)
    rows, request_path, entry = load_requests(receipt, args.mode, args.replication)
    output = args.output.resolve()
    require(output.is_relative_to(ROOT / 'output'), 'Output must stay in cloud output/')
    output.mkdir(exist_ok=False)
    status = output / 'STATUS.json'
    status.write_text(json.dumps({'status': 'loading_after_real_val_selection',
        'receipt_sha256': receipt_sha, 'mode': args.mode, 'replication': args.replication}) + '\n')
    scorer, betas, indices, vq = load_cloud_model(receipt, rows)
    metadata = {'method': 'hhc_native_smpl_mgt', 'is_formal_test': True, 'gt_shape_used': False,
        'shape_policy': receipt['shape_policy'], 'shape_draw_indices': indices,
        'shape_seed_replication': 0, 'mm_shape_rule': SAMPLING['mm_shape_rule'],
        'shape_scope': 'Same betas for both MGT conditions, VQ decoder and native FK',
        'selection_receipt_sha256': receipt_sha, 'winner': receipt['winner'],
        'mgt_checkpoint_sha256': receipt['mgt_checkpoint']['sha256'], 'mgt_step': 40000,
        'vq_checkpoint_sha256': vq['sha256'], 'vq_spec_sha256': receipt['vq_spec']['sha256'],
        'prior_spec_sha256': receipt['shape_prior']['sha256'] if receipt['shape_prior'] else None,
        'sampling': SAMPLING, 'replication': args.replication, 'mode': args.mode,
        'exporter_sha256': file_sha(Path(__file__))}
    destination = output / ('generations.h5' if args.mode == 'main' else 'mm_generations.h5')
    metadata = write_generations(destination, rows, request_path, entry, scorer, betas, metadata, args.mode)
    require(file_sha(args.selection_receipt) == receipt_sha, 'Receipt changed during generation')
    validate_selection(args.selection_receipt)
    import torch
    result = {'status': 'complete', 'mode': args.mode, 'replication': args.replication,
              'generations': len(rows) * (1 if args.mode == 'main' else 30),
              'h5': str(destination), 'h5_sha256': file_sha(destination), 'metadata': metadata,
              'gpu_peak_gib': torch.cuda.max_memory_allocated() / 1024**3,
              'scope': 'Generation only; no formal score or SOTA claim'}
    temporary = status.with_suffix('.tmp')
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    os.replace(temporary, status)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
