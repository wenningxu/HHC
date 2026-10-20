"""Freeze a real validation winner before any text-only formal generation.

This is deliberately independent of the active 28-source training/val contract.
No receipt can be produced until all five planned text-only val scores exist.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VAL_REQUEST_SHA = '751efc9ced91e2a6ad4eec4a7b9316946761fe13f86d09af7d5c04a9226b7f93'
SAMPLING = {'sampler': 'sample_intra_cas', 'sampling_steps': 50,
            'cfg_scale': 3.5, 'temperature': .5,
            'motion_seed': 'mm_repeat_seed(request.sample_seed, repeat_index)',
            'shape_seed_replication': 0,
            'mm_shape_rule': 'fixed_per_prompt_across_30_motion_repeats'}
CANDIDATES = ('A_historical_zero', 'B_matched_text_zero', 'C_uniform_window_zero',
              'A_historical_train_mean', 'A_historical_empirical_train_pair')
EXTRA_SOURCES = (
    'tools/hhc_text_only_formal_contract.py',
    'tools/export_hhc_text_only_formal_cloud.py',
    'tools/export_hhc_mgt_train_prior_validation_cloud.py',
    'utils/hhc_train_shape_prior.py',
    'dataset/interhuman_generated_h5.py',
    'tools/reconstruct_interhuman_raw_body.py',
    'tools/score_frozen_interhuman_h5_cpu.py',
    'tools/score_fair_interhuman_main.py',
    'tools/score_fair_interhuman_mm.py',
    'models/interhuman_evaluator_wrapper.py',
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def record(path):
    path = Path(path).resolve()
    return {'path': str(path), 'bytes': path.stat().st_size, 'sha256': file_sha(path)}


def verify(record_):
    path = Path(record_['path'])
    require(path.stat().st_size == record_['bytes'] and file_sha(path) == record_['sha256'],
            'Frozen artifact changed: ' + str(path))
    return path


def read_json(path):
    return json.loads(Path(path).read_text())


def generation_metadata(score):
    outer = score['generation_metadata']
    value = outer.get('metadata', outer)
    return json.loads(value) if isinstance(value, str) else value


def candidate_identity(metadata):
    arm = metadata['conditioning_arm']
    window = metadata['token_window']
    policy = metadata['shape_policy']
    if arm == 'historical' and window == 'historical':
        label = 'A_historical'
    elif arm == 'matched' and window == 'historical':
        label = 'B_matched_text'
    elif arm == 'historical' and window == 'uniform':
        label = 'C_uniform_window'
    else:
        raise ValueError('Unplanned conditioning/window candidate')
    return label + '_' + policy


def validate_score(score, expected_script_sha):
    require(score['status'] == 'complete' and score['split'] == 'val'
            and score['samples'] == 480 and score['replication'] is None,
            'Selection requires completed fixed480 validation, never test')
    require(score['request_sha256'] == VAL_REQUEST_SHA, 'Different validation requests')
    require(score['script_sha256'] == expected_script_sha, 'Different evaluator script')
    require(score['reference_gt_text'] is not None
            and all(score['reference_gt_text'][key]['equal'] is True
                    for key in ('GT', 'text')), 'Baseline GT/text parity missing')
    metadata = generation_metadata(score)
    require(metadata['gt_shape_used'] is False and metadata['is_formal_test'] is False,
            'Selection requires actual text-only validation')
    require(metadata['shape_policy'] in ('zero', 'train_mean', 'empirical_train_pair'),
            'Unsupported shape policy')
    require(metadata['mgt_step'] == 40000 and metadata['selection'] == 'final',
            'Only predeclared final40000 candidates are eligible')
    require(metadata['sampling_steps'] == 50 and metadata['cfg'] == 3.5
            and metadata['temperature'] == .5, 'Validation sampling differs')
    require(metadata['sampling_seed'] == 'mm_repeat_seed(request.sample_seed,0)',
            'Validation motion seed differs')
    require(all(math.isfinite(float(v)) for v in score['metrics'].values()),
            'Nonfinite validation metric')
    return candidate_identity(metadata), metadata


def validate_manifests(main_path, mm_path):
    main, mm = read_json(main_path), read_json(mm_path)
    require(main['schema'] == 'interhuman_fair_requests_manifest_v3'
            and mm['schema'] == 'interhuman_fair_mm_requests_manifest_v1'
            and main['status'] == mm['status'] == 'complete', 'Incomplete formal manifests')
    for item in (main, mm):
        require(sorted(row['replication'] for row in item['replications']) == list(range(20)),
                'Formal evaluation requires replications 0..19')
        canonical = json.dumps(item['replications'], sort_keys=True, separators=(',', ':'), ensure_ascii=False)
        require(hashlib.sha256(canonical.encode()).hexdigest() == item['replication_set_sha256'],
                'Formal replication-set digest mismatch')
    require(main['protocol']['evaluated_entries_per_replication'] == 1056,
            'Main protocol must be 20 x 1056')
    require(mm['protocol'] == {'replications': 20, 'effective_entries': 1098,
            'prompts': 100, 'repeats': 30, 'pairs': 10, 'embedding_scale': 6,
            'distance_divisor': 2}, 'MM protocol must be 20 x 100 x 30')
    require(mm['main_request_manifest']['sha256'] == file_sha(main_path),
            'MM is not bound to the selected main manifest')
    return main, mm


def freeze_selection(score_paths, mgt_run, vq_spec, shape_prior, main_manifest, mm_manifest):
    script_sha = file_sha(ROOT / 'tools/score_frozen_interhuman_h5_cpu.py')
    candidates = {}
    for path in score_paths:
        score = read_json(path)
        name, metadata = validate_score(score, script_sha)
        require(name in CANDIDATES and name not in candidates, 'Duplicate/unplanned candidate')
        generation = Path(path).parent / 'generations.h5'
        require(file_sha(generation) == score['generation_sha256'], 'Validation generation changed')
        candidates[name] = {'score': record(path), 'generation': record(generation), 'fid': score['metrics']['fid'],
                            'metadata': metadata}
    require(set(candidates) == set(CANDIDATES), 'All five planned text-only val candidates required')
    scores = [read_json(row['score']['path']) for row in candidates.values()]
    require(all(score['evaluator'] == scores[0]['evaluator']
                and score['reference_embeddings_sha256'] == scores[0]['reference_embeddings_sha256']
                and score['reference_embeddings_sha256'] is not None for score in scores),
            'Candidates use different evaluator/reference embeddings')
    require(len({row['metadata']['vq_spec_sha256'] for row in candidates.values()}) == 1,
            'Candidates must share the same frozen VQ')
    # Tie-break by fixed candidate name, independent of any test result.
    winner = min(candidates, key=lambda name: (candidates[name]['fid'], name))
    selected = candidates[winner]['metadata']
    training_path = Path(mgt_run) / 'training_manifest.json'
    training = read_json(training_path)
    contract = training['contract']
    require(training['status'] == 'pilot_complete_validation_only'
            and training['step'] == contract['steps'] == 40000, 'MGT training incomplete')
    checkpoint = Path(mgt_run) / 'latest.pth'
    require(file_sha(checkpoint) == selected['mgt_checkpoint_sha256'], 'Winner MGT checkpoint mismatch')
    vq = read_json(vq_spec)
    require(vq['selection_scope'] == 'completed_training_val_best', 'VQ training incomplete')
    require(file_sha(vq_spec) == selected['vq_spec_sha256'] == contract['vq_spec_sha256']
            and vq['sha256'] == selected['vq_checkpoint_sha256'], 'Winner VQ mismatch')
    require(contract['conditioning_arm'] == selected['conditioning_arm']
            and contract['token_window_policy'] == selected['token_window'], 'Winner training arm differs')
    prior_record = None
    if selected['shape_policy'] != 'zero':
        require(shape_prior is not None, 'Selected train prior required')
        prior_record = record(shape_prior)
        prior = read_json(shape_prior)
        require(prior_record['sha256'] == selected['shape_prior_spec_sha256']
                and prior['array_sha256'] == selected['shape_array_sha256'], 'Winner prior differs')
    validate_manifests(main_manifest, mm_manifest)
    sources = dict(contract['sources'])
    for path, digest in sources.items():
        require(file_sha(ROOT / path) == digest, 'Training source changed: ' + path)
    sources.update({path: file_sha(ROOT / path) for path in EXTRA_SOURCES})
    return {'schema': 'hhc_text_only_formal_selection_v1', 'status': 'frozen_validation_selection',
            'selection_rule': 'min_fixed480_text_only_val_fid_among_predeclared_five; name_tiebreak',
            'winner': winner, 'candidates': candidates, 'shape_policy': selected['shape_policy'],
            'gt_shape_used': False, 'mgt_checkpoint': record(checkpoint),
            'training_manifest': record(training_path), 'training_contract_sha256': training['contract_sha256'],
            'vq_spec': record(vq_spec), 'shape_prior': prior_record,
            'main_manifest': record(main_manifest), 'mm_manifest': record(mm_manifest),
            'sources': sources, 'sampling': SAMPLING,
            'formal_results_used_for_selection': False,
            'scope': 'Frozen generation plan only; actual model generation/20-rep scoring still required'}


def validate_selection(path):
    receipt = read_json(path)
    require(receipt['schema'] == 'hhc_text_only_formal_selection_v1'
            and receipt['status'] == 'frozen_validation_selection'
            and receipt['gt_shape_used'] is False
            and receipt['formal_results_used_for_selection'] is False,
            'Missing real text-only validation selection receipt')
    require(receipt['sampling'] == SAMPLING, 'Frozen sampling contract changed')
    require(set(EXTRA_SOURCES).issubset(receipt['sources']), 'Missing runtime source hashes')
    for source, digest in receipt['sources'].items():
        require(file_sha(ROOT / source) == digest, 'Frozen source changed: ' + source)
    for candidate in receipt['candidates'].values():
        verify(candidate['score'])
        verify(candidate['generation'])
    for key in ('mgt_checkpoint', 'training_manifest', 'vq_spec', 'main_manifest', 'mm_manifest'):
        verify(receipt[key])
    if receipt['shape_prior'] is not None:
        verify(receipt['shape_prior'])
    rebuilt = freeze_selection([row['score']['path'] for row in receipt['candidates'].values()],
        Path(receipt['training_manifest']['path']).parent, receipt['vq_spec']['path'],
        receipt['shape_prior']['path'] if receipt['shape_prior'] else None,
        receipt['main_manifest']['path'], receipt['mm_manifest']['path'])
    require(rebuilt == receipt, 'Selection receipt is inconsistent with real validation evidence')
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate-scores', nargs=5, type=Path, required=True)
    parser.add_argument('--mgt-run', type=Path, required=True)
    parser.add_argument('--vq-spec', type=Path, required=True)
    parser.add_argument('--shape-prior', type=Path)
    parser.add_argument('--main-manifest', type=Path, required=True)
    parser.add_argument('--mm-manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    receipt = freeze_selection(args.candidate_scores, args.mgt_run, args.vq_spec,
                              args.shape_prior, args.main_manifest, args.mm_manifest)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        stream.write(json.dumps(receipt, indent=2, sort_keys=True) + '\n')
    validate_selection(args.output)
    print(json.dumps({'status': receipt['status'], 'winner': receipt['winner'],
                      'receipt_sha256': file_sha(args.output)}))


if __name__ == '__main__':
    main()
