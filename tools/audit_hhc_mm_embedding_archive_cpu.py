"""Synthetic CPU H5 -> MM embeddings -> score -> verified consumption checks.

Uses real frozen request formatting and released metric arithmetic, but mocks
the validation selection and InterCLIP encoder. Never an actual test result.
"""
import copy
import json
import os
from pathlib import Path
import tempfile
from unittest.mock import patch

import h5py
import numpy as np
import torch
from tools import score_hhc_text_only_formal_mm_cpu as scorer
from tools.consume_interhuman_formal_generation_cloud import consume
from tools.interhuman_formal_score_records import read_record, verify_artifacts
from dataset.interhuman_generated_h5 import MEAN_SHA256, STD_SHA256, mm_repeat_seed
from tools.hhc_text_only_formal_contract import SAMPLING, file_sha
from tools.score_fair_interhuman_mm import multimodality_from_embeddings
from models.interhuman_evaluator_wrapper import load_official_metrics_module

ROOT = Path(__file__).resolve().parents[1]


def run_archive_checks():
    assert str(ROOT) == '/workspace/HHC' and os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    passed = []
    def rejects(name, action, message):
        try:
            action()
        except ValueError as error:
            assert message in str(error), (name, str(error))
            passed.append(name)
        else:
            raise AssertionError('Invalid evidence accepted: ' + name)
    metrics = load_official_metrics_module()
    with tempfile.TemporaryDirectory(prefix='synthetic_hhc_mm_archive_', dir=ROOT / 'output') as directory:
        folder = Path(directory)
        generation = folder / 'output/mm_generations.h5'
        generation.parent.mkdir()
        manifest_path = ROOT / 'output/interhuman_fair_mm_requests/manifest.json'
        entry = json.loads(manifest_path.read_text())['replications'][0]
        requests = json.loads(Path(entry['path']).read_text())['requests']
        assert len(requests) == 100
        selection = folder / 'mocked_selection.json'
        selection.write_text('{"scope":"synthetic CPU mock only; not a real selection"}')
        vq_spec = folder / 'mocked_vq_spec.json'
        vq_spec.write_text('{"sha256":"synthetic_vq"}')
        selected = {'winner': 'A_historical_zero', 'shape_policy': 'zero', 'shape_prior': None,
                    'mgt_checkpoint': {'sha256': 'synthetic_mgt'},
                    'vq_spec': {'path': str(vq_spec), 'sha256': file_sha(vq_spec)},
                    'mm_manifest': {'path': str(manifest_path), 'sha256': file_sha(manifest_path)}}
        metadata = {'method': 'hhc_native_smpl_mgt', 'is_formal_test': True, 'gt_shape_used': False,
                    'shape_policy': 'zero', 'mgt_step': 40000, 'winner': 'A_historical_zero',
                    'mgt_checkpoint_sha256': 'synthetic_mgt', 'vq_checkpoint_sha256': 'synthetic_vq',
                    'vq_spec_sha256': file_sha(vq_spec), 'prior_spec_sha256': None,
                    'selection_receipt_sha256': file_sha(selection), 'sampling': SAMPLING,
                    'mm_shape_rule': SAMPLING['mm_shape_rule'], 'mode': 'mm', 'replication': 0,
                    'exporter_sha256': 'synthetic_exporter'}
        marker = np.arange(3000, dtype=np.float32).reshape(100, 30)
        with h5py.File(generation, 'x') as out:
            out.attrs.update(schema='interhuman_fair_mm_generation_v1', status='complete', split='test',
                request_file_sha256=entry['sha256'], request_content_sha256=entry['request_sha256'],
                normalization_mean_sha256=MEAN_SHA256, normalization_std_sha256=STD_SHA256,
                sampler=SAMPLING['sampler'], sampling_steps=50, cfg_scale=3.5, temperature=.5,
                gt_shape_used=False, shape_policy='zero', metadata=json.dumps(metadata))
            strings = h5py.string_dtype('utf-8')
            out.create_dataset('names', data=[row['identifier'] for row in requests], dtype=strings)
            out.create_dataset('texts', data=[row['caption'] for row in requests], dtype=strings)
            out.create_dataset('lengths', data=[row['processed_length'] for row in requests])
            out.create_dataset('repeat_seeds', data=np.asarray([[mm_repeat_seed(row['sample_seed'], repeat)
                for repeat in range(30)] for row in requests], dtype=np.uint64))
            for key in ('motion1', 'motion2'):
                out.create_dataset(key, (100, 30, 300, 262), dtype=np.float32,
                    chunks=(1, 1, 300, 262), compression='lzf', fillvalue=0)
            out['motion1'][:, :, 0, 0] = marker / 1000
            out['motion1'][:, :, 0, 1] = np.sin(marker / 13)
            out['motion2'][:, :, 0, 0] = np.cos(marker / 17)
        class Encoder:
            calls = 0
            def get_motion_embeddings(self, batch):
                type(self).calls += 1
                return torch.stack([batch[2][:, 0, 0], batch[2][:, 0, 1], batch[3][:, 0, 0]], dim=1)
        expected = np.stack([marker / 1000, np.sin(marker / 13), np.cos(marker / 17)], axis=-1)
        output = generation.parent / 'score.json'
        with patch.object(scorer, 'validate_selection', return_value=selected), \
             patch.object(scorer.InterHumanEvaluatorWrapper, 'from_official_checkpoint', return_value=Encoder()), \
             patch.object(scorer, 'evaluator_provenance', return_value={'scope': 'mocked CPU encoder, not real test score'}):
            rng_before = np.random.get_state()
            score = scorer.score(selection, 0, generation, output)
            rng_after = np.random.get_state()
            assert rng_before[0] == rng_after[0] and np.array_equal(rng_before[1], rng_after[1]) and rng_before[2:] == rng_after[2:]
            assert Encoder.calls == 100
            passed.append('real_MM_H5_request_dataset_and100_prompt_encoder_calls')
            with np.load(score['embeddings'], allow_pickle=False) as archive:
                assert np.array_equal(archive['motion'], expected)
            assert score['multimodality'] == multimodality_from_embeddings(expected, metrics, pairs=10, random_seed=1234)
            passed.append('retained_embeddings_exact_prompt_repeat_order_and_original_metric_parity')
            passed.append('numpy_RNG_unchanged_by_original_MM_pair_sampling')
            record = read_record(output, 'hhc', 'mm')
            assert verify_artifacts(record) == 'present_and_rehashed'
            passed.append('v2_score_and_archive_hashes_accepted_by_comparison_adapter')
            rejects('score_and_archive_never_overwritten', lambda: scorer.score(selection, 0, generation, output), 'Never replay')
            for label, key, wrong in [
                ('different_winner_rejected_before_encoder', 'winner', 'B_matched_text_zero'),
                ('GT_shape_rejected_before_encoder', 'gt_shape_used', True),
                ('changing_shape_across_repeats_rejected_before_encoder', 'mm_shape_rule', 'redraw_per_repeat'),
                ('different_MGT_checkpoint_rejected_before_encoder', 'mgt_checkpoint_sha256', 'other'),
                ('different_VQ_checkpoint_rejected_before_encoder', 'vq_checkpoint_sha256', 'other'),
            ]:
                changed = copy.deepcopy(metadata); changed[key] = wrong
                with h5py.File(generation, 'r+') as out: out.attrs['metadata'] = json.dumps(changed)
                before = Encoder.calls
                rejects(label, lambda: scorer.score(selection, 0, generation, generation.parent / (label + '.json')),
                        'pure text-only' if key == 'gt_shape_used' else 'identity mismatch')
                assert Encoder.calls == before
                with h5py.File(generation, 'r+') as out: out.attrs['metadata'] = json.dumps(metadata)
            # Restoring an H5 attribute may change file serialization: bind the
            # final verified fixture bytes again, without altering the score.
            score['mm_generations_sha256'] = file_sha(generation)
            output.write_text(json.dumps(score))
            rejects('invalid_replication_rejected', lambda: scorer.score(selection, 20, generation, generation.parent / 'bad_rep.json'), 'replication')
            old = copy.deepcopy(score); old['schema'] = 'interhuman_fair_mm_score_v1'
            for key in ('embeddings', 'embeddings_sha256', 'embedding_contract'): old.pop(key)
            old_path = output.with_name('v1_score.json'); old_path.write_text(json.dumps(old))
            rejects('v1_HHC_MM_without_archive_keeps_H5', lambda: consume(old_path, 'hhc', 'mm', root=folder), 'Keep H5')
            assert generation.exists()
            bad = copy.deepcopy(score); bad['embeddings_sha256'] = '0' * 64
            bad_path = output.with_name('wrong_hash.json'); bad_path.write_text(json.dumps(bad))
            rejects('corrupted_archive_hash_prevents_consumption', lambda: consume(bad_path, 'hhc', 'mm', root=folder), 'Embedding archive')
            assert generation.exists()
            bad = copy.deepcopy(score); bad['embedding_contract']['shape'] = [100, 29, 3]
            bad_path.write_text(json.dumps(bad))
            rejects('wrong_MM_archive_shape_rejected', lambda: read_record(bad_path, 'hhc', 'mm'), 'embedding contract')
            bad = copy.deepcopy(score); bad['embeddings'] = str(generation)
            bad_path.write_text(json.dumps(bad))
            rejects('H5_cannot_alias_retained_archive', lambda: read_record(bad_path, 'hhc', 'mm'), 'separate NPZ')
            consumed = consume(output, 'hhc', 'mm', root=folder)
            assert consumed['status'] == 'hash_verified_then_generation_deleted'
            assert output.exists() and Path(score['embeddings']).exists() and not generation.exists()
            assert verify_artifacts(read_record(output, 'hhc', 'mm')) == 'hash_verified_at_consumption_then_deleted'
            with np.load(score['embeddings'], allow_pickle=False) as archive:
                assert multimodality_from_embeddings(archive['motion'], metrics, pairs=10, random_seed=1234) == score['multimodality']
            passed.append('v2_consumption_keeps_exact_MM_recomputable_from_retained_embeddings')
    return passed
