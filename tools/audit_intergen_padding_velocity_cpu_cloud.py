"""Test padding-edge leakage in the pinned official geometry loss, CPU only.

This is an independent diagnostic. The candidate AST changes only the adjacent
frame validity mask; it is never installed into the official training source.
"""
import argparse
import ast
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import sys
import time
from types import MethodType, SimpleNamespace
from unittest.mock import patch

assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT/'third_party/intergen_official_baseline_20261017'
LOSS = SOURCE/'models/losses.py'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def methods(candidate=False):
    tree = ast.parse(LOSS.read_text())
    cls = next(n for n in tree.body if getattr(n, 'name', None) == 'GeometricLoss')
    selected = [copy.deepcopy(n) for n in cls.body if getattr(n, 'name', None) in
                {'mix_masked_mse', 'forward_vel', 'forward_contact', 'foot_detect'}]
    assert len(selected) == 4
    changes = []

    class PairMask(ast.NodeTransformer):
        def visit_Subscript(self, node):
            if ast.unparse(node) == 'self.mask[:, :-1]':
                changes.append(node.lineno)
                return ast.copy_location(ast.parse('self.mask[:, :-1] * self.mask[:, 1:]',
                                                   mode='eval').body, node)
            return self.generic_visit(node)

    if candidate:
        selected = [PairMask().visit(n) for n in selected]
        assert len(changes) == 2
    module = ast.fix_missing_locations(ast.Module(body=selected, type_ignores=[]))
    namespace = {'torch': torch}
    exec(compile(module, str(LOSS), 'exec'), namespace)
    return {n.name: namespace[n.name] for n in selected}, changes


def run_case(name, lengths, changes, active=None, target=None):
    batch, frames = len(lengths), 6 if target is None else target.shape[1]
    if target is None:
        target = torch.zeros(batch, frames, 22, 3, dtype=torch.float64)
    prediction = target.detach().clone()
    for b, frame, amplitude in changes:
        prediction[b, frame] += amplitude
    prediction.requires_grad_()
    lengths = torch.tensor(lengths)
    mask = (torch.arange(frames)[None] < lengths[:, None])[:, :, None].double()
    active = torch.ones(batch, dtype=torch.float64) if active is None else torch.tensor(active).double()
    result = {'case': name, 'lengths': lengths.tolist(), 'perturbations': changes,
              'old_cross_boundary_edges': int(((mask[:, :-1] != 0) & (mask[:, 1:] == 0)).sum()),
              'implementations': {}}
    for label, funcs in [('official', methods()[0]), ('pair_mask_candidate', methods(True)[0])]:
        manager = SimpleNamespace(Loss=torch.nn.MSELoss(reduction='none'),
            pred_g_joints=prediction, tgt_g_joints=target, mask=mask,
            timestep_mask=active, weights={'VEL': 30, 'FC': 30},
            fids=[7, 10, 8, 11], name='A', losses={})
        for key, function in funcs.items():
            setattr(manager, key, MethodType(function, manager))
        manager.forward_vel()
        manager.forward_contact()
        row = {}
        for key, value in manager.losses.items():
            gradient = torch.autograd.grad(value, prediction, retain_graph=True)[0]
            assert torch.isfinite(value) and torch.isfinite(gradient).all()
            invalid = ~mask[:, :, 0].bool()
            row[key] = {'loss': float(value.detach()),
                        'padding_gradient_abs_sum': float(gradient[invalid].abs().sum())}
        result['implementations'][label] = row
    return result


def real_dataset_case():
    sys.path.insert(0, str(ROOT))
    from dataset.interhuman_hhc_adapter import OfficialInterHumanAdapter
    adapter = OfficialInterHumanAdapter(data_root=ROOT/'dataset/interhuman_official')
    identifier = adapter.split_ids('train', require_text=True, min_raw_length=15)[0]
    assert not any(k == 'utils' or k.startswith('utils.') for k in sys.modules)
    sys.path.insert(0, str(SOURCE))
    spec = importlib.util.spec_from_file_location('padding_audit_official_data', SOURCE/'datasets/interhuman.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    paths = [adapter._person_path(identifier, person) for person in (1, 2)]
    motions = [module.load_motion(str(path), 15, swap=False)[0] for path in paths]
    item = module.InterHumanDataset.__new__(module.InterHumanDataset)
    item.cache = True
    item.max_length = item.max_gt_length = 300
    item.data_list = [{'name': identifier, 'motion_id': 0, 'swap': False, 'texts': ['audit']}]
    item.motion_dict = {0: motions}
    with patch.object(random, 'choice', side_effect=lambda x: 'audit' if isinstance(x[0], str) else 0), \
            patch.object(np.random, 'rand', side_effect=[.25, .25]):
        _, _, left, right, length = item[0]
    assert 0 < length < 300 and not np.any(left[length:]) and not np.any(right[length:])
    target = torch.from_numpy(left[:, :66].copy()).reshape(1, 300, 22, 3).double()
    row = run_case('real_official_train_item_padding_only_1m', [length], [(0, length, 1.)], target=target)
    row.update(identifier=identifier, raw_sha256=[sha(path) for path in paths],
               data_source_sha256=sha(SOURCE/'datasets/interhuman.py'),
               official_padding_is_raw_zero=True, valid_prediction_exactly_equals_gt=True,
               candidate_scope='Velocity only is asserted here; contact uses predicted feet/height gating.')
    assert abs(row['implementations']['official']['VEL_A']['loss'] - 30/length) < 1e-6
    assert row['implementations']['pair_mask_candidate']['VEL_A']['loss'] == 0
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert ROOT == Path('/workspace/HHC') and not args.output.exists()
    assert not torch.cuda.is_initialized()
    torch.set_num_threads(1)
    os.nice(15)
    before = sha(LOSS)
    assert before == sha(ROOT/'third_party/intergen_base_20261008/models/losses.py')
    rows = [
        run_case('zero_control', [3], []),
        run_case('first_padding_1m', [3], [(0, 3, 1.)]),
        run_case('first_padding_1cm_contact_active', [3], [(0, 3, .01)]),
        run_case('later_padding_only_control', [3], [(0, 5, 1.)]),
        run_case('all_frames_valid_control', [6], [(0, 3, .01)]),
        run_case('inactive_timestep_control', [3], [(0, 3, .01)], [0]),
        run_case('one_valid_frame_boundary', [1], [(0, 1, .01)]),
        run_case('variable_length_batch', [3, 5], [(0, 3, 1.), (1, 5, 1.)]),
    ]
    expected_vel = [0, 10, .001, 0, .0012, 0, .003, 8]
    for row, expected in zip(rows, expected_vel):
        old = row['implementations']['official']
        new = row['implementations']['pair_mask_candidate']
        # The unchanged official reducers add 1e-7 to two denominators.
        assert abs(old['VEL_A']['loss'] - expected) < 1e-5, row
        assert new['VEL_A']['padding_gradient_abs_sum'] == 0
        assert new['FC_A']['padding_gradient_abs_sum'] == 0
        if row['case'] == 'all_frames_valid_control':
            assert old == new
        else:
            assert new['VEL_A']['loss'] == new['FC_A']['loss'] == 0
    for metric in ('VEL_A', 'FC_A'):
        assert abs(rows[2]['implementations']['official'][metric]['loss'] - .001) < 1e-8
        assert rows[2]['implementations']['official'][metric]['padding_gradient_abs_sum'] > 0
    rows.append(real_dataset_case())
    assert sha(LOSS) == before and not torch.cuda.is_initialized()
    report = {'schema': 'official_intergen_padding_velocity_cpu_v1', 'status': 'passed',
        'cpu_only': True, 'cuda_initialized': False, 'cases_count': len(rows),
        'source_sha256': sha(Path(__file__)), 'official_loss_sha256': before,
        'candidate_changed_lines': methods(True)[1], 'rows': rows,
        'scope': 'Literal official loss methods; independent mask-only AST candidate; '
                 'eight synthetic controls and one actual official train item. '
                 'No model/checkpoint or generation. Official reproduction source unchanged. '
                 'Archived custom rotation loss already uses validity of BOTH adjacent frames; '
                 'this official-side bug cannot be cited as the cause of custom FID degradation. '
                 'No measured effect on FID or SOTA.',
        'production_sources_modified': False, 'time_unix': time.time()}
    with args.output.open('x') as handle:
        json.dump(report, handle, indent=2)
        handle.write('\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
