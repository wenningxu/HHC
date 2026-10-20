"""Synthetic CPU integration of full160-score comparison and safe H5 consumption."""
import ast
import copy
import json
import os
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from tools import aggregate_interhuman_text_only_comparison_cpu as aggregate
from tools.consume_interhuman_formal_generation_cloud import consume
from tools.interhuman_formal_score_records import read_record, verify_artifacts
from tools.hhc_text_only_formal_contract import file_sha


def main():
    assert str(ROOT) == '/workspace/HHC' and os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    passed = []
    main_manifest = ROOT / 'output/interhuman_fair_requests_v3/manifest.json'
    mm_manifest = ROOT / 'output/interhuman_fair_mm_requests/manifest.json'
    entries = {'main': json.loads(main_manifest.read_text())['replications'],
               'mm': json.loads(mm_manifest.read_text())['replications']}
    def rejects(name, action, message):
        try: action()
        except ValueError as error:
            assert message in str(error), (name, str(error))
            passed.append(name)
        else: raise AssertionError('Invalid evidence accepted: ' + name)
    with tempfile.TemporaryDirectory(prefix='comparison_cpu_fixture_', dir=ROOT / 'output') as directory:
        folder = Path(directory)
        # This is a deleted CPU fixture, never an actual selection receipt.
        selected_path = folder / 'synthetic_selection_fixture.json'
        selected_path.write_text('{"scope":"mocked CPU test; not a valid real selection receipt"}')
        selection_sha = file_sha(selected_path)
        selected = {'winner': 'A_historical_zero', 'mgt_checkpoint': {'sha256': 'synthetic_mgt'}, 'shape_policy': 'zero'}
        config = {'main_manifest': str(main_manifest), 'mm_manifest': str(mm_manifest),
                  'selection_receipt': str(selected_path), 'methods': {}}
        reference_hashes = {}
        method_order = ('intergen', 'hhc', 'intermask', 'timotion')
        originals = {}
        for method in method_order:
            config['methods'][method] = {'main_scores': [], 'mm_scores': []}
            for mode in ('main', 'mm'):
                for rep in range(20):
                    work = folder / 'output' / method / mode / f'{rep:02d}'; work.mkdir(parents=True)
                    path = work / 'score.json'
                    generation = work / ('generations.h5' if mode == 'main' else 'mm_generations.h5')
                    generation.write_bytes(b'SYNTHETIC CPU H5 provenance fixture; not real motion')
                    embedding = path.with_suffix('.embeddings.npz')
                    if not (method == 'hhc' and mode == 'mm'):
                        with embedding.open('wb') as stream: np.savez(stream, fixture=np.asarray([rep], np.float32))
                    embedding_sha = file_sha(embedding) if embedding.exists() else None
                    if method == 'intergen' and mode == 'main': reference_hashes[rep] = embedding_sha
                    offset = aggregate.METHODS.index(method) + 1
                    metrics = {key: offset + rep * .01 for key in aggregate.MAIN}
                    metrics['diversity_gt'] = 7.5 + rep * .001
                    metrics.update(top1=.25 + offset * .01 + rep * .001,
                                   top2=.45 + offset * .01 + rep * .001,
                                   top3=.65 + offset * .01 + rep * .001)
                    evaluator = {'scope': 'SYNTHETIC CPU fixture, not real evaluator output'}
                    parity = {key: {'equal': True} for key in ('GT', 'text')}
                    if method == 'hhc':
                        metadata = {'method': 'hhc_native_smpl_mgt', 'winner': 'A_historical_zero',
                            'mgt_checkpoint_sha256': 'synthetic_mgt', 'vq_checkpoint_sha256': 'synthetic_vq',
                            'vq_spec_sha256': 'synthetic_vq_spec', 'shape_policy': 'zero', 'prior_spec_sha256': None,
                            'selection_receipt_sha256': selection_sha, 'sampling': {'fixture': True},
                            'exporter_sha256': 'synthetic_exporter', 'mm_shape_rule': 'fixed',
                            'is_formal_test': True, 'gt_shape_used': False}
                        outer = {'metadata': json.dumps(metadata)}
                        if mode == 'main':
                            score = {'status': 'complete', 'split': 'test', 'samples': 1056, 'replication': rep,
                                'generation_metadata': outer, 'metrics': metrics,
                                'request_sha256': entries[mode][rep]['request_sha256'], 'evaluator': evaluator,
                                'generation_sha256': file_sha(generation), 'embeddings_sha256': embedding_sha,
                                'reference_embeddings_sha256': reference_hashes[rep], 'reference_gt_text': parity}
                        else:
                            score = {'schema': 'interhuman_fair_mm_score_v1', 'status': 'complete', 'replication': rep,
                                'generation_metadata': outer, 'multimodality': offset + rep * .01,
                                'request_content_sha256': entries[mode][rep]['request_sha256'],
                                'evaluator_provenance': evaluator, 'mm_generations': str(generation),
                                'mm_generations_sha256': file_sha(generation), 'metric_contract': {
                                    'prompts': 100, 'repeats': 30, 'pairs': 10, 'embedding_scale': 6,
                                    'distance_divisor': 2, 'pair_sampling_seed': 1234 + rep}}
                    else:
                        metadata = {'method': method, 'model': 'SYNTHETIC_' + method, 'checkpoint_sha256': method,
                            'source_commit': 'fixture', 'source_files_sha256': {'fixture': 'only'}, 'sampler': {'steps': 50},
                            'native_length_policy': 'fixture', 'exporter_sha256': 'fixture',
                            'baseline_cpu_receipt_sha256': 'fixture_' + method, 'selection_receipt_sha256': selection_sha,
                            'gt_motion_or_shape_used': False, 'vq_checkpoint_sha256': 'fixture_vq'}
                        score = {'schema': 'interhuman_native_baseline_score_v1', 'status': 'complete',
                            'method': method, 'mode': mode, 'split': 'test', 'replication': rep,
                            'samples': 1056 if mode == 'main' else 100, 'mm_repeats': None if mode == 'main' else 30,
                            'generation_metadata': metadata, 'metrics': metrics if mode == 'main' else {'multimodality': offset + rep * .01},
                            'request_content_sha256': entries[mode][rep]['request_sha256'], 'evaluator': evaluator,
                            'generation': str(generation), 'generation_sha256': file_sha(generation),
                            'embeddings': str(embedding), 'embeddings_sha256': embedding_sha,
                            'reference_embeddings_sha256': reference_hashes[rep] if method != 'intergen' else None,
                            'reference_gt_text': parity if method != 'intergen' else None}
                    path.write_text(json.dumps(score)); originals[str(path)] = copy.deepcopy(score)
                    config['methods'][method][mode + '_scores'].append(str(path))
        with patch.object(aggregate, 'validate_selection', return_value=selected):
            result = aggregate.summarize(config)
            assert len(result['score_files_sha256']) == 160
            assert abs(result['metrics']['hhc']['fid']['mean'] - 1.095) < 1e-12
            assert abs(result['paired_differences']['intergen']['fid']['hhc_minus_baseline'] + 1) < 1e-12
            assert np.allclose(result['paired_differences']['intergen']['fid']['paired_replication_bootstrap95'], [-1, -1])
            passed.append('full160_synthetic_score_join_mean_CI_and_paired_bootstrap')
            wrong = copy.deepcopy(config); wrong['methods']['timotion']['mm_scores'] = wrong['methods']['timotion']['mm_scores'][:-1]
            rejects('nineteen_MM_scores_rejected', lambda: aggregate.summarize(wrong), 'twenty distinct')
            wrong = copy.deepcopy(config); wrong['methods']['coshmdm'] = wrong['methods']['intergen']
            rejects('InterGen_alias_as_CoSh_rejected', lambda: aggregate.summarize(wrong), 'all three reproducible')
            path = Path(config['methods']['intermask']['main_scores'][1])
            for label, mutate, message in (
                ('duplicate_replication_rejected', lambda row: row.update(replication=0), 'Replications'),
                ('different_evaluator_rejected', lambda row: row.update(evaluator={'different': True}), 'Evaluator provenance'),
                ('missing_GT_text_parity_rejected', lambda row: row.update(reference_gt_text={'GT': {'equal': False}, 'text': {'equal': True}}), 'GT/text parity'),
                ('different_GT_diversity_seed_rejected', lambda row: row['metrics'].update(diversity_gt=999), 'GT diversity'),
                ('nonfinite_metric_rejected', lambda row: row['metrics'].update(fid=float('nan')), 'nonfinite'),
            ):
                changed = copy.deepcopy(originals[str(path)]); mutate(changed); path.write_text(json.dumps(changed))
                rejects(label, lambda: aggregate.summarize(config), message)
                path.write_text(json.dumps(originals[str(path)]))
            score_path = Path(config['methods']['intergen']['main_scores'][0])
            report = consume(score_path, 'intergen', 'main', root=folder)
            assert report['status'] == 'hash_verified_then_generation_deleted'
            assert not Path(report['generation']).exists() and Path(report['embedding']).exists()
            result = aggregate.summarize(config)
            assert result['artifact_verification_states'][str(score_path)] == 'hash_verified_at_consumption_then_deleted'
            passed.append('hash_verified_consumption_keeps_score_embedding_and_comparison_valid')
            rejects('consumption_not_replayed', lambda: consume(score_path, 'intergen', 'main', root=folder), 'Only this cloud')
            changed = copy.deepcopy(originals[str(score_path)]); changed['metrics']['fid'] += 1
            score_path.write_text(json.dumps(changed))
            rejects('changed_consumed_score_rejected', lambda: verify_artifacts(read_record(score_path, 'intergen', 'main')), 'Invalid consumption')
            score_path.write_text(json.dumps(originals[str(score_path)]))
            protected = Path(config['methods']['hhc']['mm_scores'][0])
            h5 = read_record(protected, 'hhc', 'mm')['generation']
            rejects('HHC_MM_without_embedding_archive_keeps_H5', lambda: consume(protected, 'hhc', 'mm', root=folder), 'Keep H5')
            assert h5.exists()
            score = Path(config['methods']['timotion']['main_scores'][0])
            record = read_record(score, 'timotion', 'main'); raw = record['generation'].read_bytes(); record['generation'].unlink()
            rejects('missing_H5_without_receipt_rejected', lambda: aggregate.summarize(config), 'without verified consumption')
            record['generation'].write_bytes(raw)
            changed = copy.deepcopy(originals[str(score)]); changed['split'] = 'val'; score.write_text(json.dumps(changed))
            rejects('validation_score_cannot_consume', lambda: consume(score, 'timotion', 'main', root=folder), 'Not a matching')
            score.write_text(json.dumps(originals[str(score)]))
    # Loader-only CPU compatibility changes did not alter the tested I/O/RNG functions.
    old = ROOT / 'tools/provenance_snapshots/813b8211280aa388c7e29c7a5047be36e8db991f219a193d3ba906237a39c3de__export_interhuman_native_baselines_formal_cloud.py'
    new = ROOT / 'tools/export_interhuman_native_baselines_formal_cloud.py'
    def function(path, name):
        return ast.dump(next(node for node in ast.parse(path.read_text()).body if isinstance(node, ast.FunctionDef) and node.name == name), include_attributes=False)
    assert all(function(old, name) == function(new, name) for name in ('request_rng', 'write_native_h5'))
    passed.append('33_check_protocol_tested_IO_RNG_AST_unchanged_after_loader_fix')
    config = json.loads((ROOT / 'migration/hhc-train-shape-validation-v2.json').read_text())
    assert all(file_sha(ROOT / name) == digest for name, digest in config['sources'].items())
    passed.append('29_active_sources_unchanged')
    from tools.audit_hhc_mm_embedding_archive_cpu import run_archive_checks
    passed.extend(run_archive_checks())
    output = ROOT / 'output/hhc_sota_20261016/comparison_consumption_v2_cpu_audit.json'
    result = {'status': 'passed_CPU_synthetic_comparison_and_consumption', 'checks': passed,
        'all_synthetic_score_and_generation_fixtures_deleted': True, 'real_generation_files_deleted': 0,
        'GPU_work': False, 'real_formal_comparison_created': False,
        'scope': 'Mocked selection validator plus synthetic score fixtures only; no real result or SOTA evidence',
        'sources': {str(p.relative_to(ROOT)): file_sha(p) for p in (Path(__file__),
            ROOT / 'tools/interhuman_formal_score_records.py', ROOT / 'tools/consume_interhuman_formal_generation_cloud.py',
            ROOT / 'tools/aggregate_interhuman_text_only_comparison_cpu.py', ROOT / 'tools/audit_hhc_mm_embedding_archive_cpu.py',
            ROOT / 'tools/score_hhc_text_only_formal_mm_cpu.py')}}
    with output.open('x') as stream: json.dump(result, stream, indent=2, sort_keys=True)
    print(json.dumps(result), flush=True)


if __name__ == '__main__': main()
