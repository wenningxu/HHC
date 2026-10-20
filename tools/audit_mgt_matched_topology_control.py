#!/usr/bin/env python
"""Verify that an MGT control changes topology and no training setting."""

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path


SCHEMA = 'hhc_mgt_matched_topology_control_audit_v1'
ALLOWED_ARGUMENT_DIFFERENCES = {
    'exp_name', 'out_dir', 'paper_architecture', 'num_local_layer',
}
MATCHED_PAPER_FIELDS = (
    'total_iterations', 'batch_size', 'learning_rate', 'warmup_iterations',
    'schedule', 'optimizer', 'weight_decay', 'codebook_size_per_part',
    'blocks', 'attention_heads', 'latent_dimension', 'mask_schedule',
    'objective', 'sampling_order', 'sampling_steps', 'temperature',
    'cfg_scale', 'contact_guidance_scale', 'contact_as_training_input',
    'shape_conditioning',
)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def ast_class_sha256(path, class_name):
    tree = ast.parse(Path(path).read_text(encoding='utf-8'))
    matches = [
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == class_name
    ]
    if len(matches) != 1:
        raise ValueError(f'{path}: expected one class {class_name}, got {len(matches)}')
    canonical = ast.dump(matches[0], annotate_fields=True, include_attributes=False)
    return hashlib.sha256(canonical.encode('utf-8')).hexdigest()


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def atomic_json(path, payload):
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f'.tmp.{os.getpid()}')
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + '\n', encoding='utf-8'
    )
    os.replace(temporary, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--paper-manifest', type=Path, required=True)
    parser.add_argument('--control-manifest', type=Path, required=True)
    parser.add_argument('--paper-launch-source', type=Path, required=True)
    parser.add_argument('--control-launch-source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()

    paper = load(args.paper_manifest)
    control = load(args.control_manifest)
    paper_args = paper['resolved_arguments']
    control_args = control['resolved_arguments']
    argument_differences = {
        key: {'paper': paper_args.get(key), 'control': control_args.get(key)}
        for key in sorted(set(paper_args) | set(control_args))
        if paper_args.get(key) != control_args.get(key)
    }
    if set(argument_differences) != ALLOWED_ARGUMENT_DIFFERENCES:
        raise ValueError(
            'topology control has unexpected argument differences: '
            f'{argument_differences}'
        )
    expected_topology = {
        'paper_architecture': {'paper': True, 'control': False},
        'num_local_layer': {'paper': 9, 'control': 2},
    }
    for key, expected in expected_topology.items():
        if argument_differences[key] != expected:
            raise ValueError(f'unexpected topology switch for {key}')

    paper_configuration = paper['paper_configuration']
    control_configuration = control['paper_configuration']
    configuration_mismatches = {
        key: {
            'paper': paper_configuration.get(key),
            'control': control_configuration.get(key),
        }
        for key in MATCHED_PAPER_FIELDS
        if paper_configuration.get(key) != control_configuration.get(key)
    }
    if configuration_mismatches:
        raise ValueError(
            f'training/sampling configuration is not matched: {configuration_mismatches}'
        )
    if control_configuration.get('architecture_contract') != (
            'legacy_mmm_topology_control_not_for_final_reporting'):
        raise ValueError('control manifest is not explicitly diagnostic-only')

    if paper['cache_manifest'] != control['cache_manifest']:
        raise ValueError('paper and topology-control runs do not bind the same cache')
    source_hash_differences = {
        key: {'paper': paper['sha256'].get(key), 'control': control['sha256'].get(key)}
        for key in sorted(set(paper['sha256']) | set(control['sha256']))
        if paper['sha256'].get(key) != control['sha256'].get(key)
    }
    if set(source_hash_differences) != {'train/train_t2i_trans.py'}:
        raise ValueError(
            'unexpected launch artifact/source differences: '
            f'{source_hash_differences}'
        )

    paper_source_hash = sha256_file(args.paper_launch_source)
    control_source_hash = sha256_file(args.control_launch_source)
    if paper_source_hash != paper['sha256']['train/train_t2i_trans.py']:
        raise ValueError('paper launch-source snapshot does not match its manifest')
    if control_source_hash != control['sha256']['train/train_t2i_trans.py']:
        raise ValueError('control launch source does not match its manifest')
    paper_lightning_hash = ast_class_sha256(
        args.paper_launch_source, 'MaskGITLightning'
    )
    control_lightning_hash = ast_class_sha256(
        args.control_launch_source, 'MaskGITLightning'
    )
    if paper_lightning_hash != control_lightning_hash:
        raise ValueError('optimizer/training/validation class changed between controls')

    result = {
        'schema': SCHEMA,
        'status': 'pass',
        'interpretation': (
            'diagnostic topology control only; never a final paper-model result'
        ),
        'paper_manifest': {
            'path': str(args.paper_manifest.resolve()),
            'sha256': sha256_file(args.paper_manifest),
        },
        'control_manifest': {
            'path': str(args.control_manifest.resolve()),
            'sha256': sha256_file(args.control_manifest),
        },
        'argument_differences': argument_differences,
        'matched_paper_configuration': {
            key: paper_configuration[key] for key in MATCHED_PAPER_FIELDS
        },
        'identical_cache_manifest_sha256': hashlib.sha256(
            json.dumps(paper['cache_manifest'], sort_keys=True).encode('utf-8')
        ).hexdigest(),
        'launch_sources': {
            'paper': {
                'path': str(args.paper_launch_source.resolve()),
                'sha256': paper_source_hash,
            },
            'control': {
                'path': str(args.control_launch_source.resolve()),
                'sha256': control_source_hash,
            },
            'only_byte_difference': 'train/train_t2i_trans.py',
            'identical_maskgit_lightning_ast_sha256': paper_lightning_hash,
            'other_manifest_source_and_artifact_hashes_identical': True,
        },
    }
    atomic_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
