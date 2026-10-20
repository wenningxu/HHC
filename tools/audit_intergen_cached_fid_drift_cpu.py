"""Read-only InterHuman validation-cache FID decomposition; no model inference."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
METRICS = Path('/mnt/8T-HDD/wenning/InterMask/utils/metrics.py')
METRICS_SHA = '9b6461032246313cbbf99c50d358812bd88b8b4333c858105039028a4a7b20d2'
REQUEST_SHA = '751efc9ced91e2a6ad4eec4a7b9316946761fe13f86d09af7d5c04a9226b7f93'


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def record(path):
    return {'path': str(path), 'bytes': path.stat().st_size, 'sha256': sha(path)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--steps', nargs='+', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert str(ROOT) == '/workspace/HHC' and os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    assert not args.output.exists() and len(set(args.steps)) == len(args.steps)
    assert args.run.resolve().is_relative_to(ROOT / 'output')
    assert args.output.resolve().is_relative_to(ROOT / 'output')
    os.nice(15)
    assert sha(METRICS) == METRICS_SHA
    spec = importlib.util.spec_from_file_location('fid_drift_official_metrics', METRICS)
    metrics = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(metrics)
    reference_gt = None
    reference_provenance = None
    rows = []
    for step in args.steps:
        directory = args.run / f'validation_{step:06d}'
        report_path = directory / 'validation_metrics.json'
        embedding_path = directory / 'validation_embeddings.npz'
        report = json.loads(report_path.read_text())
        assert report['status'] == 'complete' and report['split'] == 'validation'
        assert report['samples'] == 480 and report['batch_size'] == 96
        assert report['random_seed'] == 20261006 and report['replication'] is None
        assert report['request_sha256'] == REQUEST_SHA
        assert Path(report['source_embeddings']).resolve() == embedding_path.resolve()
        assert sha(embedding_path) == report['source_embeddings_sha256']
        if reference_provenance is None:
            reference_provenance = report['evaluator_provenance']
        assert report['evaluator_provenance'] == reference_provenance
        with np.load(embedding_path, allow_pickle=False) as source:
            assert set(source.files) == {'GT', 'raw'}
            gt, generated = source['GT'].copy(), source['raw'].copy()
        assert gt.shape == generated.shape == (480, 512)
        assert np.isfinite(gt).all() and np.isfinite(generated).all()
        if reference_gt is None:
            reference_gt = gt.copy()
        assert np.array_equal(gt, reference_gt), 'GT embeddings changed across checkpoints'
        mu_gt, cov_gt = metrics.calculate_activation_statistics(gt, 6)
        mu_generated, cov_generated = metrics.calculate_activation_statistics(generated, 6)
        total = float(metrics.calculate_frechet_distance(mu_gt, cov_gt, mu_generated, cov_generated))
        assert abs(total - report['arms']['raw']['FID']) < 1e-4
        delta = mu_gt - mu_generated
        mean_term = float(delta.dot(delta))
        covariance_term = total - mean_term
        assert covariance_term >= -1e-4
        rows.append({'step': step, 'fid': total, 'mean_term': mean_term,
                     'covariance_term': covariance_term,
                     'mean_term_fraction': mean_term / total if total else None,
                     'gt_covariance_trace': float(np.trace(cov_gt)),
                     'generated_covariance_trace': float(np.trace(cov_generated)),
                     'generated_to_gt_trace_ratio': float(np.trace(cov_generated) / np.trace(cov_gt)),
                     'gt_exactly_equal_to_first_step': True,
                     'reported_metrics': report['arms']['raw'],
                     'report': record(report_path), 'embeddings': record(embedding_path)})
        print(json.dumps({'step': step, 'fid': total, 'mean_term': mean_term,
                          'covariance_term': covariance_term}), flush=True)
    assert 'torch' not in sys.modules, 'This diagnostic must not import a model runtime'
    result = {'schema': 'intergen_cached_validation_fid_drift_cpu_v1', 'status': 'complete',
              'cpu_only': True, 'model_inference': False, 'arrays_modified': False,
              'request_sha256': REQUEST_SHA, 'embedding_scale': 6, 'rows': rows,
              'metrics_source': record(METRICS), 'source': record(Path(__file__)),
              'evaluator_provenance': reference_provenance, 'unix': time.time(),
              'scope': 'Descriptive mean/covariance decomposition of the same reported FID. '
                       'No feature correction, causal claim, new generation score or SOTA claim. '
                       'Covariance trace alone cannot establish absence of mode collapse.'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(json.dumps({'status': 'complete', 'output': record(args.output)}), flush=True)


if __name__ == '__main__':
    main()
