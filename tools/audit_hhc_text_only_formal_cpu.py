"""CPU synthetic gate tests + actual frozen-request H5 protocol tests; no model quality."""
import copy
import json
import os
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import h5py
import numpy as np
import torch
from tools import hhc_text_only_formal_contract as contract
from tools import export_hhc_text_only_formal_cloud as exporter
from utils.hhc_train_shape_prior import TrainShapePrior
from dataset.interhuman_generated_h5 import mm_repeat_seed


def main():
    assert str(ROOT) == '/workspace/HHC' and os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    torch.set_num_threads(2)
    passed = []
    main_path = ROOT / 'output/interhuman_fair_requests_v3/manifest.json'
    mm_path = ROOT / 'output/interhuman_fair_mm_requests/manifest.json'
    contract.validate_manifests(main_path, mm_path)
    passed.append('real_main20x1056_mm20x100x30_manifest_binding')
    bound = json.loads((ROOT / 'migration/hhc-train-shape-validation.json').read_text())['sources']
    assert all(contract.file_sha(ROOT / name) == digest for name, digest in bound.items())

    def rejects(name, action):
        try:
            action()
        except (ValueError, FileNotFoundError, FileExistsError, KeyError):
            passed.append(name)
        else:
            raise AssertionError('Unsafe formal condition accepted: ' + name)

    with tempfile.TemporaryDirectory(prefix='formal_cpu_fixture_', dir=ROOT / 'output') as directory:
        folder = Path(directory)
        # Entirely synthetic files, confined to this deleted temporary directory.
        # These are not a real training result or a persistent selection receipt.
        for name in contract.EXTRA_SOURCES:
            path = folder / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('CPU synthetic source fixture\n')
        run = folder / 'run'
        run.mkdir()
        checkpoint = run / 'latest.pth'
        checkpoint.write_bytes(b'CPU fixture, not a model checkpoint')
        vq_path = folder / 'vq.json'
        vq_path.write_text(json.dumps({'selection_scope': 'completed_training_val_best', 'sha256': 'vq'}))
        training = {'status': 'pilot_complete_validation_only', 'step': 40000,
                    'contract_sha256': 'mock_contract', 'contract': {'steps': 40000,
                    'vq_spec_sha256': contract.file_sha(vq_path), 'conditioning_arm': 'historical',
                    'token_window_policy': 'historical', 'sources': {}}}
        (run / 'training_manifest.json').write_text(json.dumps(training))
        paths = []
        for index, name in enumerate(contract.CANDIDATES):
            destination = folder / name
            destination.mkdir()
            generation = destination / 'generations.h5'
            generation.write_bytes(b'CPU synthetic generation provenance fixture')
            policy = ('train_mean' if name.endswith('train_mean') else
                      'empirical_train_pair' if name.endswith('empirical_train_pair') else 'zero')
            metadata = {'conditioning_arm': 'matched' if index == 1 else 'historical',
                'token_window': 'uniform' if index == 2 else 'historical', 'shape_policy': policy,
                'gt_shape_used': False, 'is_formal_test': False, 'mgt_step': 40000,
                'selection': 'final', 'sampling_steps': 50, 'cfg': 3.5, 'temperature': .5,
                'sampling_seed': 'mm_repeat_seed(request.sample_seed,0)',
                'mgt_checkpoint_sha256': contract.file_sha(checkpoint),
                'vq_spec_sha256': contract.file_sha(vq_path), 'vq_checkpoint_sha256': 'vq'}
            score = {'status': 'complete', 'split': 'val', 'samples': 480, 'replication': None,
                'request_sha256': contract.VAL_REQUEST_SHA,
                'script_sha256': contract.file_sha(folder / 'tools/score_frozen_interhuman_h5_cpu.py'),
                'reference_gt_text': {key: {'equal': True} for key in ('GT', 'text')},
                'generation_metadata': {'metadata': json.dumps(metadata)},
                'generation_sha256': contract.file_sha(generation), 'metrics': {'fid': 10.0 + index},
                'evaluator': {'scope': 'synthetic_CPU_fixture'}, 'reference_embeddings_sha256': 'mock_reference'}
            path = destination / 'metrics.json'
            path.write_text(json.dumps(score))
            paths.append(path)
        with patch.object(contract, 'ROOT', folder):
            freeze = lambda scores=paths: contract.freeze_selection(scores, run, vq_path, None, main_path, mm_path)
            receipt = freeze()
            assert receipt['winner'] == 'A_historical_zero'
            receipt_path = folder / 'synthetic_selection.json'
            receipt_path.write_text(json.dumps(receipt))
            assert contract.validate_selection(receipt_path) == receipt
            passed.append('synthetic_five_candidate_min_val_fid_freeze_and_rebuild')
            rejects('missing_candidate_rejected', lambda: freeze(paths[:-1]))
            rejects('duplicate_candidate_rejected', lambda: freeze(paths[:-1] + paths[:1]))
            original = json.loads(paths[0].read_text())
            for name, mutate in (
                ('test_score_rejected', lambda row: row.update(split='test')),
                ('GT_shape_rejected', lambda row: row['generation_metadata'].update(metadata=json.dumps(
                    dict(json.loads(row['generation_metadata']['metadata']), gt_shape_used=True)))),
                ('missing_baseline_parity_rejected', lambda row: row.update(reference_gt_text=None)),
                ('nonfinite_metric_rejected', lambda row: row.update(metrics={'fid': float('nan')})),
                ('different_sampling_rejected', lambda row: row['generation_metadata'].update(metadata=json.dumps(
                    dict(json.loads(row['generation_metadata']['metadata']), temperature=1)))),
                ('different_motion_seed_rejected', lambda row: row['generation_metadata'].update(metadata=json.dumps(
                    dict(json.loads(row['generation_metadata']['metadata']), sampling_seed='other')))),
            ):
                changed = copy.deepcopy(original)
                mutate(changed)
                paths[0].write_text(json.dumps(changed))
                rejects(name, freeze)
                paths[0].write_text(json.dumps(original))
            changed = dict(receipt, winner='B_matched_text_zero')
            receipt_path.write_text(json.dumps(changed))
            rejects('receipt_winner_edit_rejected', lambda: contract.validate_selection(receipt_path))
            receipt_path.write_text(json.dumps(receipt))
            source = folder / contract.EXTRA_SOURCES[0]
            source.write_text('changed')
            rejects('runtime_source_drift_rejected', lambda: contract.validate_selection(receipt_path))
        del receipt

        # Real request metadata only; no GT motion or beta datasets are opened.
        minimal_receipt = {'main_manifest': {'path': str(main_path)}, 'mm_manifest': {'path': str(mm_path)}}
        prior = TrainShapePrior(ROOT / 'output/hhc_train_shape_prior_20261016/prior_spec.json')
        rng_before = np.random.get_state()
        audit_files = {}
        for mode in ('main', 'mm'):
            rows, request_path, entry = exporter.load_requests(minimal_receipt, mode, 0)
            betas, indices = exporter.assign_shapes(rows, 'empirical_train_pair', prior)
            reversed_betas, _ = exporter.assign_shapes(list(reversed(rows)), 'empirical_train_pair', prior)
            assert all(np.array_equal(betas[name], reversed_betas[name]) for name in betas)
            calls = []
            class FakeGenerator:
                def generate(self, requests, seeds, steps):
                    assert steps == 50 and len({row['identifier'] for row in requests}) == len(requests)
                    calls.append((requests, seeds))
                    output = torch.zeros(len(requests), 300, 524)
                    for i, (request, seed) in enumerate(zip(requests, seeds)):
                        output[i, :request['processed_length']] = (seed % 997) / 997
                    return output
            destination = folder / (mode + '.h5')
            exporter.write_generations(destination, rows, request_path, entry, FakeGenerator(), betas,
                {'shape_policy': 'empirical_train_pair', 'scope': 'CPU synthetic protocol fixture'}, mode)
            repeats = 1 if mode == 'main' else 30
            expected = [(row['identifier'], mm_repeat_seed(row['sample_seed'], repeat))
                        for repeat in range(repeats) for row in rows]
            actual = [(row['identifier'], seed) for requests, seeds in calls for row, seed in zip(requests, seeds)]
            assert actual == expected
            with h5py.File(destination, 'r') as handle:
                assert np.array_equal(handle['shape_betas'][:], np.stack([betas[row['identifier']] for row in rows]))
                assert handle.attrs['status'] == 'complete' and not handle.attrs['gt_shape_used']
                if mode == 'mm':
                    assert not np.array_equal(handle['motion1'][0, 0], handle['motion1'][0, 1])
                    assert handle['repeat_seeds'][0, 29] == mm_repeat_seed(rows[0]['sample_seed'], 29)
            audit_files[mode] = {'generated': len(expected), 'bytes': destination.stat().st_size,
                                 'sha256': contract.file_sha(destination)}
            passed.append('real_' + mode + '_requests_synthetic_H5_schema_order_seed_shape_padding')
            rejects(mode + '_export_replay_rejected', lambda: exporter.write_generations(
                destination, rows, request_path, entry, FakeGenerator(), betas, {}, mode))
        rng_after = np.random.get_state()
        assert rng_before[0] == rng_after[0] and np.array_equal(rng_before[1], rng_after[1]) and rng_before[2:] == rng_after[2:]
        passed.append('shape_draws_order_independent_and_global_numpy_rng_unchanged')
    assert all(contract.file_sha(ROOT / name) == digest for name, digest in bound.items())
    passed.append('all_28_active_sources_unchanged')
    result = {'status': 'passed_CPU_synthetic_gate_and_real_request_protocol_audit', 'checks': passed,
              'temporary_synthetic_files_deleted': True, 'synthetic_h5_audit': audit_files,
              'GPU_work': False, 'real_MGT_generation': False, 'real_selection_receipt_created': False,
              'scope': 'Not quality evidence. Real final40000 val selection and real main/MM GPU E2E remain required.',
              'sources': {name: contract.file_sha(ROOT / name) for name in (
                  'tools/hhc_text_only_formal_contract.py', 'tools/export_hhc_text_only_formal_cloud.py',
                  'tools/audit_hhc_text_only_formal_cpu.py')}}
    output = ROOT / 'output/hhc_sota_20261016/text_only_formal_cpu_audit.json'
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
