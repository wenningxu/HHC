"""CPU counterexamples for official versus archived custom temporal reductions.

Extract the real reducer ASTs, without loading models or changing either source.
This diagnoses loss semantics only; it is not an objective replacement.
"""
import ast
import hashlib
import json
import os
from pathlib import Path
import time

assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
import torch

ROOT = Path(__file__).resolve().parents[1]
SOURCES = {
    'official': ROOT / 'third_party/intergen_base_20261008/models/losses.py',
    'archived_grouped': ROOT / 'tools/intergen_rotation_grouped_objective.py',
    'sparse_v2': ROOT / 'tools/intergen_rotation_objective_sparse_v2.py',
}


def record(path):
    data = path.read_bytes()
    return {'path': str(path), 'bytes': len(data),
            'sha256': hashlib.sha256(data).hexdigest()}


def extract(path, ancestors, name, namespace):
    node = ast.parse(path.read_text())
    for parent in ancestors:
        node = next(item for item in node.body if getattr(item, 'name', None) == parent)
    function = next(item for item in node.body if getattr(item, 'name', None) == name)
    module = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))
    exec(compile(module, str(path), 'exec'), namespace)
    return namespace[name]


def compare(name, lengths, active, amplitudes):
    lengths = torch.tensor(lengths)
    active = torch.tensor(active, dtype=torch.bool)
    batch, time_steps = len(lengths), int(lengths.max())
    valid = torch.arange(time_steps)[None] < lengths[:, None]
    geometric = valid & active[:, None]
    prediction = torch.tensor(amplitudes, dtype=torch.float64)[:, None, None, None]
    prediction = prediction.expand(batch, time_steps, 1, 1).clone().requires_grad_()
    target = torch.zeros_like(prediction)
    manager = type('ReducerOnly', (), {'Loss': torch.nn.MSELoss(reduction='none')})()
    official = extract(SOURCES['official'], ['InterLoss'], 'mix_masked_mse', {})
    reference = official(manager, prediction, target, valid[:, :, None, None], active)
    values, gradients = {'official': float(reference.detach())}, {}
    gradients['official'] = torch.autograd.grad(reference, prediction)[0]
    for label in ('archived_grouped', 'sparse_v2'):
        reducer = extract(SOURCES[label], ['rotation_objective'], 'avg',
                          {'valid': valid, 'reference_micro_batch': 4})
        loss = reducer((prediction - target).square(), geometric)
        values[label] = float(loss.detach())
        gradients[label] = torch.autograd.grad(loss, prediction)[0]
    assert values['archived_grouped'] == values['sparse_v2']
    assert torch.equal(gradients['archived_grouped'], gradients['sparse_v2'])
    for gradient in gradients.values():
        assert torch.isfinite(gradient).all()
        assert gradient[~geometric].count_nonzero() == 0
    return {'case': name, 'lengths': lengths.tolist(), 'active': active.tolist(),
            'per_frame_squared_error': [float(value * value) for value in amplitudes],
            'loss': values,
            'gradient_sum_per_sequence': {k: v.sum((1, 2, 3)).tolist()
                                          for k, v in gradients.items()}}


def main():
    torch.set_num_threads(1)
    assert ROOT == Path('/workspace/HHC') and not torch.cuda.is_initialized()
    os.nice(15)
    rows = [
        compare('equal_lengths_all_active_control', [6] * 4, [True] * 4, [1, 3, 1, 3]),
        compare('unequal_lengths_all_active', [6, 2, 6, 2], [True] * 4, [1, 3, 1, 3]),
        compare('one_active_group_of_eight', [6] * 32, [True] * 4 + [False] * 28, [1] * 32),
        compare('unequal_active_counts_across_groups', [6] * 8, [True] + [False] * 3 + [True] * 4,
                [3] * 4 + [1] * 4),
        compare('all_inactive_finite_zero_control', [6, 2, 6, 2], [False] * 4, [1, 3, 1, 3]),
    ]
    expected = [(5, 5), (5, 3), (1, .125), (2.6, 5), (0, 0)]
    for row, (official, custom) in zip(rows, expected):
        assert abs(row['loss']['official'] - official) < 1e-6
        assert abs(row['loss']['archived_grouped'] - custom) < 1e-12
    # Geometry uses a different sequence reduction from official; simple MSE
    # is deliberately not covered by these counterexamples (official simple
    # itself uses a global valid-frame mean).
    result = {
        'schema': 'intergen_temporal_reduction_cpu_v1', 'status': 'passed',
        'cpu_only': True, 'cuda_initialized': torch.cuda.is_initialized(),
        'source': record(Path(__file__)),
        'source_files': {k: record(p) for k, p in SOURCES.items()}, 'rows': rows,
        'scope': 'Exact reducer ASTs, synthetic errors, no model/checkpoint/data loaded. '
                 'Official geometric terms average valid frames within each sequence, '
                 'then average active sequences. Custom pools active frames within '
                 'each micro4 group then averages groups including empty groups. '
                 'Sparse-v2 deliberately changes spatial denominators only, so this '
                 'separate temporal weighting mismatch remains. '
                 'No measured FID effect or claim that it explains the full gap. '
                 'Official simple MSE is a separate global frame reduction.',
        'production_sources_modified': False, 'time_unix': time.time(),
    }
    out = ROOT / 'output/hhc_sota_20261016/intergen_temporal_reduction_cpu_20261017.json'
    with out.open('x') as handle:
        json.dump(result, handle, indent=2)
        handle.write('\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
