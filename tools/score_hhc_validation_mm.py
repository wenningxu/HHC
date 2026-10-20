#!/usr/bin/env python3
"""Score multimodality for the frozen Inter-X validation MM subset."""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from collections import Counter
import h5py, numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

def duplicate_motion_summary(motion_dataset, lengths):
    motions = motion_dataset
    lengths = np.asarray(lengths)
    if len(motions.shape) < 3 or motions.shape[0] != len(lengths) or not len(lengths) or motions.shape[1] < 2:
        raise ValueError('motions must be [N,R,...], with at least 2 repeats')
    if lengths.ndim != 1 or not np.issubdtype(lengths.dtype, np.integer) or np.any(lengths <= 0) or np.any(lengths > motions.shape[2]):
        raise ValueError('lengths must be positive and within motion time dimension')
    counts = []
    for index, length in enumerate(lengths):
        sample = np.asarray(motions[index])
        hashes = [hashlib.sha256(np.ascontiguousarray(sample[r, :length], dtype=np.float32).tobytes()).hexdigest() for r in range(sample.shape[0])]
        counts.append(Counter(hashes))
    pairs = sum(n * (n - 1) // 2 for c in counts for n in c.values())
    total = len(counts) * motions.shape[1] * (motions.shape[1] - 1) // 2
    return {'duplicate_pairs': int(pairs), 'total_pairs': int(total),
            'exact_duplicate_pair_rate': pairs / total,
            'mean_unique_motions_per_prompt': float(np.mean([len(c) for c in counts])),
            'definition': 'byte-identical valid-frame float32 motions; no distance threshold'}

def all_pair_mm_summary(embedding):
    embedding = np.asarray(embedding)
    if (embedding.ndim != 3 or not embedding.shape[0] or embedding.shape[1] < 2
            or not embedding.shape[2] or not np.isfinite(embedding).all()):
        raise ValueError('finite nonempty [prompts, repeats>=2, features] embeddings required')
    first, second = np.triu_indices(embedding.shape[1], k=1)
    distances = np.linalg.norm(embedding[:, first] - embedding[:, second], axis=-1)
    prompt_means = distances.mean(axis=1)
    total = float(prompt_means.sum())
    top_count = max(1, int(np.ceil(len(prompt_means) * 0.05)))
    return {'pairs_per_prompt': len(first), 'mean': float(prompt_means.mean()),
            'per_prompt_mean': prompt_means.tolist(),
            'per_prompt_median': np.median(distances, axis=1).tolist(),
            'prompt_mean_percentiles': {str(p): float(np.percentile(prompt_means, p))
                                        for p in (0, 5, 50, 95, 100)},
            'top_5pct_prompts_fraction_of_total_distance':
                float(np.sort(prompt_means)[-top_count:].sum()) / total if total else 0.0}

def validate_generation_metadata(metadata, *, sampling_steps=50, temperature=0.5,
                                cfg_scale=3.5, temperature_end=None,
                                inference_conditioning_contract='paper_text_cfg',
                                confidence_noise_scale=0.0, confidence_noise_until_step=0,
                                confidence_noise_scope='all', generation_seed=600123,
                                confidence_ranking='probability', channel_schedule='interleaved',
                                mgt_shape_conditioning=False, mgt_beta_policy=None,
                                vq_beta_policy=None, root_replay_control=False):
    if confidence_ranking not in ('probability', 'log_odds', 'max_probability') or channel_schedule not in ('interleaved', 'partwise'):
        raise ValueError('unsupported confidence ranking or channel schedule')
    if type(confidence_noise_until_step) is not int or not 0 <= confidence_noise_until_step <= sampling_steps:
        raise ValueError('invalid confidence noise cutoff')
    expected = {
        'sampling_steps': sampling_steps, 'temperature': temperature, 'cfg_scale': cfg_scale,
        'sampler': 'sample_intra_cas', 'mgt_shape_conditioning': bool(mgt_shape_conditioning),
        'contact_guidance': False, 'contact_timing_source': 'none',
        'vq_contract': 'precleanup_raw12288', 'mgt_contract': 'historical_mmm',
        'inference_conditioning_contract': inference_conditioning_contract, 'seed': generation_seed,
        'temperature_end': 'constant' if temperature_end is None else temperature_end,
        'temperature_schedule': ('constant' if temperature_end is None else
                                 'linear_by_channel_step' if channel_schedule == 'partwise'
                                 else 'linear_by_outer_step'),
    }
    mismatches = {k: (metadata.get(k), v) for k, v in expected.items()
                  if metadata.get(k) != v}
    if bool(metadata.get('root_replay_control', False)) != bool(root_replay_control):
        mismatches['root_replay_control'] = (metadata.get('root_replay_control', False),
                                             bool(root_replay_control))
    if root_replay_control:
        for key, value in {
            'root_replay_control_ranking': 'probability',
            'root_replay_control_noise_scale': 0.0,
            'root_replay_rng_policy': 'preserve_control_batch_chain',
            'root_replay_passes_per_batch': 2,
        }.items():
            if metadata.get(key) != value:
                mismatches[key] = (metadata.get(key), value)
    for key, requested in (('mgt_beta_policy', mgt_beta_policy), ('vq_beta_policy', vq_beta_policy)):
        if requested is not None and requested not in ('requested', 'zero'):
            raise ValueError(f'unsupported {key}')
        if requested is not None and metadata.get(key, metadata.get('beta_policy', 'requested')) != requested:
            mismatches[key] = (metadata.get(key), requested)
    for key in ('mgt_beta_policy', 'vq_beta_policy'):
        if key in metadata and metadata[key] not in ('requested', 'zero'):
            mismatches[key] = metadata[key]
    for key, value, default in (('confidence_ranking', confidence_ranking, 'probability'),
                                ('confidence_noise_until_step', confidence_noise_until_step, 0),
                                ('channel_schedule', channel_schedule, 'interleaved')):
        if metadata.get(key, default) != value:
            mismatches[key] = (metadata.get(key), value)
    if channel_schedule == 'partwise':
        for key, value in {'sampling_steps_unit': 'maskgit_iterations_per_channel',
                           'maximum_active_channel_stages_per_outer_iteration': 1,
                           'maximum_model_forward_calls_per_outer_iteration_with_cfg': 2}.items():
            if metadata.get(key) != value: mismatches[key] = (metadata.get(key), value)
    if confidence_noise_scale == 0.0:
        if metadata.get('confidence_noise_scale', 0.0) != 0.0:
            mismatches['confidence_noise_scale'] = metadata.get('confidence_noise_scale')
    else:
        for key, value in {'confidence_noise_scale': confidence_noise_scale,
                           'confidence_noise_scope': confidence_noise_scope,
                           'confidence_noise_schedule': ('linear_to_zero_with_hard_cutoff'
                                                         if confidence_noise_until_step else 'linear_to_zero')}.items():
            if metadata.get(key) != value: mismatches[key] = metadata.get(key)
    if metadata.get('beta_policy', 'requested') not in ('requested', 'zero'):
        mismatches['beta_policy'] = metadata.get('beta_policy')
    if mismatches:
        raise ValueError(f'unsupported generation contract: {mismatches}')

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--requests'); p.add_argument('--generations')
    p.add_argument('--output'); p.add_argument('--repeats', type=int, default=11)
    p.add_argument('--batch-size', type=int, default=32); p.add_argument('--device', default='cpu')
    p.add_argument('--self-check', action='store_true')
    p.add_argument('--sampling-steps', type=int, default=50)
    p.add_argument('--temperature', type=float, default=0.5)
    p.add_argument('--cfg-scale', type=float, default=3.5)
    p.add_argument('--temperature-end', type=float, default=None)
    p.add_argument('--inference-conditioning-contract', choices=('paper_text_cfg', 'historical_mmm_shape_dropout'), default='paper_text_cfg')
    p.add_argument('--confidence-noise-scale', type=float, default=0.0)
    p.add_argument('--confidence-noise-until-step', type=int, default=0)
    p.add_argument('--confidence-noise-scope', choices=('all', 'root'), default='all')
    p.add_argument('--confidence-ranking', choices=('probability', 'log_odds', 'max_probability'), default='probability')
    p.add_argument('--channel-schedule', choices=('interleaved', 'partwise'), default='interleaved')
    p.add_argument('--mgt-shape', action=argparse.BooleanOptionalAction, default=False)
    p.add_argument('--root-replay-control', action=argparse.BooleanOptionalAction, default=False)
    p.add_argument('--mgt-beta-policy', choices=('requested', 'zero'), default=None)
    p.add_argument('--vq-beta-policy', choices=('requested', 'zero'), default=None)
    p.add_argument('--generation-seed', type=int, default=600123,
                   help='Expected export seed; MM pair-selection seed stays bound to the frozen request.')
    p.add_argument('--prerequisite-audit', type=Path, default=Path('output/interact2ar_paper_repo_data_audit.json'))
    p.add_argument('--author-calibration-audit', type=Path, default=Path('output/interact2ar_author_release_calibration/audit.json'))
    a = p.parse_args()
    if a.self_check:
        valid = {'sampling_steps': 50, 'temperature': 0.5, 'cfg_scale': 3.5,
                 'sampler': 'sample_intra_cas', 'mgt_shape_conditioning': False,
                 'contact_guidance': False, 'contact_timing_source': 'none',
                 'vq_contract': 'precleanup_raw12288', 'mgt_contract': 'historical_mmm',
                 'inference_conditioning_contract': 'paper_text_cfg', 'seed': 600123,
                 'temperature_end': 'constant', 'temperature_schedule': 'constant'}
        validate_generation_metadata(valid)
        cutoff = dict(valid, confidence_noise_scale=1., confidence_noise_scope='all',
                      confidence_noise_until_step=10,
                      confidence_noise_schedule='linear_to_zero_with_hard_cutoff')
        validate_generation_metadata(cutoff, confidence_noise_scale=1., confidence_noise_until_step=10)
        for metadata, options in ((cutoff, {'confidence_noise_scale': 1.}),
                                  (valid, {'confidence_noise_until_step': 10})):
            try: validate_generation_metadata(metadata, **options)
            except ValueError: pass
            else: raise AssertionError('undeclared cutoff accepted')
        validate_generation_metadata(dict(valid, confidence_ranking='log_odds'), confidence_ranking='log_odds')
        validate_generation_metadata(dict(valid, confidence_ranking='max_probability'), confidence_ranking='max_probability')
        partwise = dict(valid, channel_schedule='partwise', sampling_steps_unit='maskgit_iterations_per_channel',
                        maximum_active_channel_stages_per_outer_iteration=1,
                        maximum_model_forward_calls_per_outer_iteration_with_cfg=2)
        validate_generation_metadata(partwise, channel_schedule='partwise')
        for key, value in (('confidence_ranking', 'log_odds'), ('confidence_ranking', 'max_probability'), ('channel_schedule', 'partwise')):
            for metadata, options in ((dict(valid, **{key: value}), {}), (valid, {key: value})):
                try: validate_generation_metadata(metadata, **options)
                except ValueError: pass
                else: raise AssertionError(f'undeclared {key} accepted')
        validate_generation_metadata(dict(valid, seed=600124), generation_seed=600124)
        validate_generation_metadata(dict(valid, beta_policy='zero'))
        validate_generation_metadata(dict(valid, mgt_shape_conditioning=True), mgt_shape_conditioning=True)
        validate_generation_metadata(dict(valid, mgt_beta_policy='zero', vq_beta_policy='requested'),
                                     mgt_beta_policy='zero', vq_beta_policy='requested')
        validate_generation_metadata(dict(valid, sampling_steps=20, temperature=0.3,
                                          temperature_end=0.1,
                                          temperature_schedule='linear_by_outer_step'),
                                     sampling_steps=20, temperature=0.3, temperature_end=0.1)
        validate_generation_metadata(dict(valid, confidence_noise_scale=1.0,
                                          confidence_noise_scope='root',
                                          confidence_noise_schedule='linear_to_zero'),
                                     confidence_noise_scale=1.0, confidence_noise_scope='root')
        try: validate_generation_metadata(dict(valid, confidence_noise_scale=1.0))
        except ValueError: pass
        else: raise AssertionError('undeclared confidence noise accepted')
        for change in ({'sampler': 'wrong'}, {'temperature': 0.8}, {'seed': 600124},
                       {'cfg_scale': 2.5}, {'sampling_steps': 20},
                       {'temperature_schedule': 'linear_by_outer_step'},
                       {'inference_conditioning_contract': 'historical_mmm_shape_dropout'}):
            try:
                validate_generation_metadata(dict(valid, **change))
            except ValueError:
                continue
            raise AssertionError(f'undeclared sampling change accepted: {change}')
        print('self-check: ok')
        return
    if not a.requests or not a.generations or not a.output:
        p.error('--requests, --generations, and --output are required')
    if a.repeats <= 10: raise ValueError('--repeats must be > 10')
    # External evaluator has its own top-level utils package. Keep pure
    # metadata/duplicate helpers importable without loading that namespace.
    import tools.score_interact2ar_joint_evaluators as joint
    req_path = joint.resolve_cli_path(a.requests)
    gen_path = joint.resolve_cli_path(a.generations)
    out_path = joint.resolve_cli_path(a.output)
    request = np.load(req_path, allow_pickle=False); indices = request['mm_indices'].astype(np.int64)
    names, lengths = request['names'][indices].astype(str), request['lengths'][indices].astype(np.int64)
    prerequisite = joint.load_prerequisite_audit(a.prerequisite_audit)
    calibration = joint.load_author_calibration_audit(a.author_calibration_audit, prerequisite)
    with h5py.File(gen_path, 'r') as source:
        joint.validate_h5(source, names, lengths, a.repeats, 'validation MM generation')
        if source.attrs.get('requests_sha256') != joint.sha256_file(req_path): raise ValueError('generation is not bound to requests')
        metadata = joint.serializable_attributes(source)
        repeat_collapse = duplicate_motion_summary(source['motions'], lengths)
        validate_generation_metadata(metadata, sampling_steps=a.sampling_steps,
                                     temperature=a.temperature, cfg_scale=a.cfg_scale,
                                     temperature_end=a.temperature_end,
                                     inference_conditioning_contract=a.inference_conditioning_contract,
                                     confidence_noise_scale=a.confidence_noise_scale,
                                     confidence_noise_until_step=a.confidence_noise_until_step,
                                     confidence_noise_scope=a.confidence_noise_scope,
                                     confidence_ranking=a.confidence_ranking,
                                     channel_schedule=a.channel_schedule,
                                     mgt_shape_conditioning=a.mgt_shape,
                                     root_replay_control=a.root_replay_control,
                                     mgt_beta_policy=a.mgt_beta_policy,
                                     vq_beta_policy=a.vq_beta_policy,
                                     generation_seed=a.generation_seed)
        os.chdir(joint.INTERACT2AR_ROOT)
        wrappers = {n: joint.EvaluatorModelWrapper(SimpleNamespace(), a.device, c, representation='joints', part=part, normalized=False) for n, (part, c) in joint.WRAPPERS.items()}
        embeddings = joint.embed_motion_all_wrappers(wrappers, source['motions'], lengths, a.batch_size, repeats=a.repeats)
    metrics = {}
    for index, (name, embedding) in enumerate(embeddings.items()):
        np.random.seed(int(request['seed']) + 1000 * index)
        metrics[name] = {'multimodality': float(joint.calculate_multimodality(embedding, 10))}
    payload = {'schema':'hhc_interx_validation_mm_v1','status':'complete','diagnostic_only':True,'split':'validation','samples':len(names),'repeats':a.repeats,'metrics':metrics,'repeat_collapse':repeat_collapse,'sampling_expectation':{'sampling_steps':a.sampling_steps,'temperature':a.temperature,'cfg_scale':a.cfg_scale,'temperature_end':a.temperature_end,'inference_conditioning_contract':a.inference_conditioning_contract,'confidence_noise_scale':a.confidence_noise_scale,'confidence_noise_scope':a.confidence_noise_scope,'mgt_shape_conditioning':a.mgt_shape,'root_replay_control':a.root_replay_control,'mgt_beta_policy':a.mgt_beta_policy,'vq_beta_policy':a.vq_beta_policy},'requests':joint.artifact(req_path),'generations':joint.artifact(gen_path),'generation_metadata':metadata,'evaluator_provenance':{'repository':str(joint.INTERACT2AR_ROOT),'repository_revision':joint.repository_revision(),'prerequisite_paper_repository_data_audit':prerequisite,'prerequisite_author_release_calibration_audit':calibration,'checkpoints':{n:joint.artifact(joint.INTERACT2AR_ROOT/c) for n,(_,c) in joint.WRAPPERS.items()}}}
    payload['sampling_expectation']['generation_seed'] = a.generation_seed
    payload['sampling_expectation']['confidence_noise_until_step'] = a.confidence_noise_until_step
    payload['sampling_expectation'].update(confidence_ranking=a.confidence_ranking,
                                           channel_schedule=a.channel_schedule)
    payload['all_pairs_diagnostic'] = {name: all_pair_mm_summary(e) for name, e in embeddings.items()}
    payload['all_pairs_prompt_order'] = names.tolist()
    payload['evaluator_runtime'] = {
        'device': a.device, 'batch_size': a.batch_size,
        'torch_version': str(joint.torch.__version__),
        'device_name': (joint.torch.cuda.get_device_name(a.device)
                        if joint.torch.device(a.device).type == 'cuda' else 'cpu'),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n')

if __name__ == '__main__': main()
