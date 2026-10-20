"""CPU formal HHC MM scoring with retained, hash-bound InterCLIP embeddings.

Requires the real validation selection used by the exporter. This independent
entrypoint leaves the historical scorer and all active training sources intact.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from dataset.interhuman_generated_h5 import InterHumanMMGeneratedH5Dataset, sha256_file
from models.interhuman_evaluator_wrapper import (
    InterHumanEvaluatorWrapper, evaluator_provenance, load_official_metrics_module,
)
from tools.hhc_text_only_formal_contract import SAMPLING, require, validate_selection
from tools.score_fair_interhuman_mm import atomic_json, multimodality_from_embeddings


def score(selection_path, replication, generations, output):
    selection_path, generations, output = [Path(p).resolve() for p in (selection_path, generations, output)]
    require(str(ROOT) == '/workspace/HHC' and os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'Cloud CPU only')
    require(replication in range(20), 'Formal replication must be0..19')
    require(output.is_relative_to(ROOT / 'output'), 'Score must stay in this cloud output/')
    archive = output.with_suffix('.embeddings.npz')
    partial = archive.with_suffix('.npz.partial')
    require(not any(p.exists() for p in (output, archive, partial)), 'Never replay/overwrite a score or embedding archive')
    selected = validate_selection(selection_path)
    selection_sha = sha256_file(selection_path)
    manifest_path = Path(selected['mm_manifest']['path'])
    require(sha256_file(manifest_path) == selected['mm_manifest']['sha256'], 'Frozen MM manifest changed')
    manifest = json.loads(manifest_path.read_text())
    entry = next(row for row in manifest['replications'] if row['replication'] == replication)
    request_path = Path(entry['path']).resolve()
    generation_sha = sha256_file(generations)
    generated = InterHumanMMGeneratedH5Dataset(generations, request_path, request_entry=entry, formal=True)
    try:
        outer = generated.metadata
        metadata = json.loads(outer['metadata']) if 'metadata' in outer else outer
        require(outer.get('status') == 'complete' and outer.get('split') == 'test', 'Not a complete formal H5')
        require(metadata.get('is_formal_test') is True and metadata.get('gt_shape_used') is False,
                'HHC MM must be pure text-only formal inference')
        required = {'method': 'hhc_native_smpl_mgt', 'is_formal_test': True, 'gt_shape_used': False,
                    'mode': 'mm', 'replication': replication, 'mgt_step': 40000,
                    'winner': selected['winner'], 'shape_policy': selected['shape_policy'],
                    'mgt_checkpoint_sha256': selected['mgt_checkpoint']['sha256'],
                    'vq_spec_sha256': selected['vq_spec']['sha256'],
                    'prior_spec_sha256': selected['shape_prior']['sha256'] if selected['shape_prior'] else None,
                    'selection_receipt_sha256': selection_sha, 'sampling': SAMPLING,
                    'mm_shape_rule': SAMPLING['mm_shape_rule']}
        for key, expected in required.items():
            require(metadata.get(key) == expected, 'Frozen HHC MM identity mismatch: ' + key)
        for key in ('sampler', 'sampling_steps', 'temperature', 'cfg_scale'):
            require(outer.get(key) == SAMPLING[key], 'Outer HHC sampler mismatch: ' + key)
        for key in ('shape_policy', 'gt_shape_used'):
            require(outer.get(key) == metadata[key], 'Outer/nested metadata conflict: ' + key)
        vq = json.loads(Path(selected['vq_spec']['path']).read_text())
        require(metadata['vq_checkpoint_sha256'] == vq['sha256'], 'HHC MM VQ identity mismatch')
        wrapper = InterHumanEvaluatorWrapper.from_official_checkpoint('cpu')
        with torch.inference_mode():
            embeddings = np.stack([wrapper.get_motion_embeddings(generated[prompt]).cpu().numpy()
                                   for prompt in range(len(generated))], axis=0)
        require(embeddings.shape[:2] == (100, 30) and embeddings.ndim == 3
                and embeddings.shape[2] > 0 and embeddings.dtype == np.float32
                and np.isfinite(embeddings).all(), 'Invalid retained MM embeddings')
        metrics = load_official_metrics_module()
        value = multimodality_from_embeddings(embeddings, metrics, pairs=10, random_seed=1234 + replication)
        require(np.isfinite(value), 'Nonfinite MM score')
        require(sha256_file(generations) == generation_sha, 'H5 changed during scoring')
        output.parent.mkdir(parents=True, exist_ok=True)
        with partial.open('xb') as stream:
            np.savez_compressed(stream, motion=embeddings)
            stream.flush()
            os.fsync(stream.fileno())
        with np.load(partial, allow_pickle=False) as retained:
            require(set(retained.files) == {'motion'} and np.array_equal(retained['motion'], embeddings), 'MM archive readback differs')
        os.replace(partial, archive)
        require(sha256_file(selection_path) == selection_sha, 'Selection changed during scoring')
        validate_selection(selection_path)
        result = {
            'schema': 'interhuman_fair_mm_score_v2', 'status': 'complete', 'replication': replication,
            'multimodality': value, 'request_manifest': str(manifest_path),
            'request_manifest_sha256': sha256_file(manifest_path),
            'request_file': str(request_path), 'request_file_sha256': entry['sha256'],
            'request_content_sha256': entry['request_sha256'], 'mm_generations': str(generations),
            'mm_generations_sha256': generation_sha, 'generation_metadata': outer,
            'evaluator_provenance': evaluator_provenance(include_checkpoint_sha256=True),
            'metric_contract': {'prompts': 100, 'repeats': 30, 'pairs': 10, 'embedding_scale': 6,
                                'distance_divisor': 2, 'pair_sampling_seed': 1234 + replication},
            'embeddings': str(archive), 'embeddings_sha256': sha256_file(archive),
            'embedding_contract': {'key': 'motion', 'shape': list(embeddings.shape), 'dtype': 'float32'},
            'scorer_sha256': sha256_file(Path(__file__)),
            'scope': 'Formal text-only HHC MM; exact historical metric calculation, retained motion embeddings, fixed per-prompt train-only shape.',
        }
        atomic_json(output, result)
        return result
    finally:
        generated.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--selection-receipt', type=Path, required=True)
    parser.add_argument('--replication', type=int, required=True)
    parser.add_argument('--mm-generations', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    os.nice(10)
    result = score(args.selection_receipt, args.replication, args.mm_generations, args.output)
    print(json.dumps({k: result[k] for k in ('schema', 'status', 'replication', 'multimodality', 'embeddings_sha256')}), flush=True)


if __name__ == '__main__':
    main()
