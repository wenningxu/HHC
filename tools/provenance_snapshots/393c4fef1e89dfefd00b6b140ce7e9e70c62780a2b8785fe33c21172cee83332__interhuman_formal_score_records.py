"""Normalize current formal score schemas and verify retained/consumed artifacts."""
import json
from pathlib import Path
from tools.hhc_text_only_formal_contract import file_sha, require


def read_record(path, method, mode):
    path = Path(path).resolve()
    value = json.loads(path.read_text())
    require(value['status'] == 'complete', 'Incomplete formal score')
    if method == 'hhc' and mode == 'main':
        require(value['split'] == 'test' and value['samples'] == 1056, 'Not HHC formal main')
        metadata = value['generation_metadata']
        metadata = json.loads(metadata['metadata']) if 'metadata' in metadata else metadata
        metrics, request_sha, evaluator = value['metrics'], value['request_sha256'], value['evaluator']
        generation, digest = path.parent / 'generations.h5', value['generation_sha256']
        embedding = path.with_suffix('.embeddings.npz')
    elif method == 'hhc' and mode == 'mm':
        require(value['schema'] == 'interhuman_fair_mm_score_v1', 'Not HHC formal MM')
        require(value['metric_contract'] == {'prompts': 100, 'repeats': 30, 'pairs': 10,
            'embedding_scale': 6, 'distance_divisor': 2, 'pair_sampling_seed': 1234 + value['replication']},
            'Wrong HHC MM protocol')
        metadata = value['generation_metadata']
        metadata = json.loads(metadata['metadata']) if 'metadata' in metadata else metadata
        metrics, request_sha = {'multimodality': value['multimodality']}, value['request_content_sha256']
        evaluator = value['evaluator_provenance']
        generation, digest = Path(value['mm_generations']), value['mm_generations_sha256']
        # The existing HHC MM scorer saves the score, not a standalone embedding archive.
        embedding = None
    else:
        require(method in ('intergen', 'intermask', 'timotion')
            and value['schema'] == 'interhuman_native_baseline_score_v1'
            and value['method'] == method and value['mode'] == mode and value['split'] == 'test',
            'Not a matching native-baseline formal score')
        require(value['samples'] == (1056 if mode == 'main' else 100)
            and value['mm_repeats'] == (None if mode == 'main' else 30), 'Wrong baseline sample/repeat count')
        metadata, metrics = value['generation_metadata'], value['metrics']
        request_sha, evaluator = value['request_content_sha256'], value['evaluator']
        generation, digest = Path(value['generation']), value['generation_sha256']
        embedding = Path(value['embeddings'])
    if method == 'hhc':
        require(metadata['is_formal_test'] is True and metadata['gt_shape_used'] is False,
                'HHC score is not pure text-only formal inference')
        require(metadata['shape_policy'] in ('zero', 'train_mean', 'empirical_train_pair'), 'Unsupported HHC shape policy')
        identity = {key: metadata[key] for key in ('method', 'winner', 'mgt_checkpoint_sha256',
            'vq_checkpoint_sha256', 'vq_spec_sha256', 'shape_policy', 'prior_spec_sha256',
            'selection_receipt_sha256', 'sampling', 'exporter_sha256', 'mm_shape_rule')}
    else:
        require(metadata['gt_motion_or_shape_used'] is False, 'Baseline used GT motion/shape')
        identity = {key: metadata[key] for key in ('method', 'model', 'checkpoint_sha256',
            'source_commit', 'source_files_sha256', 'sampler', 'native_length_policy',
            'exporter_sha256', 'baseline_cpu_receipt_sha256', 'selection_receipt_sha256')}
        if method == 'intermask': identity['vq_checkpoint_sha256'] = metadata['vq_checkpoint_sha256']
    return {'path': path, 'value': value, 'method': method, 'mode': mode,
        'replication': int(value['replication']), 'metrics': metrics, 'identity': identity,
        'request_sha256': request_sha, 'evaluator': evaluator,
        'generation': generation.resolve(), 'generation_sha256': digest,
        'embedding': embedding.resolve() if embedding else None,
        'embedding_sha256': value.get('embeddings_sha256')}


def verify_artifacts(record):
    embedding = record['embedding']
    if embedding is not None:
        require(embedding.is_file() and file_sha(embedding) == record['embedding_sha256'], 'Embedding archive missing/changed')
    generation = record['generation']
    if generation.is_file():
        require(file_sha(generation) == record['generation_sha256'], 'Generation H5 changed')
        return 'present_and_rehashed'
    receipt_path = record['path'].with_suffix('.consumed.json')
    require(receipt_path.is_file(), 'Generation missing without verified consumption receipt')
    receipt = json.loads(receipt_path.read_text())
    require(receipt['schema'] == 'interhuman_formal_generation_consumption_v1'
        and receipt['status'] == 'hash_verified_then_generation_deleted'
        and receipt['score_sha256'] == file_sha(record['path'])
        and receipt['generation_sha256'] == record['generation_sha256']
        and receipt['generation'] == str(generation)
        and receipt['embedding_sha256'] == record['embedding_sha256'], 'Invalid consumption receipt')
    return 'hash_verified_at_consumption_then_deleted'
