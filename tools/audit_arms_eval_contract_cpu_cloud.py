"""Trace frozen ARMS evaluation ASTs with tiny CPU stubs; no model weights.

This verifies argument routing, initial history, and the optional MM batch path.
It does not evaluate ARMS quality or modify its public source.
"""
import ast
import hashlib
import json
import os
from pathlib import Path
import time
from types import SimpleNamespace
from typing import Optional

assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
import numpy as np
import torch
from einops import rearrange, repeat
from torch.utils.data import Dataset, DataLoader

ROOT = Path('/workspace/HHC')
SOURCE = ROOT / 'output/hhc_sota_20261016/arms_frozen_eval_source_and_weight_metadata_20261017.json'
EXPECTED_SOURCE_SHA = 'b07324951445ad5ae96ccd71d7884fd6df8dbbef5a29178eca06117cbc92a527'
COMMIT = 'e3819179209d9f3d4b717673af4f839f5b2bdf44'


def sha(data):
    return hashlib.sha256(data).hexdigest()


def compile_nodes(nodes, filename, namespace):
    module = ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[]))
    exec(compile(module, filename, 'exec'), namespace)


class GT(Dataset):
    max_length = 8

    def __init__(self, value):
        self.value = value

    def __len__(self):
        return 2

    def __getitem__(self, index):
        motion = torch.full((8, 266), float(self.value + index))
        return 'caption_' + str(index), motion, motion.clone(), 8, 8


class TraceDit:
    def __init__(self):
        self.calls = []

    def generate(self, *args, **kwargs):
        # Record every argument reaching the model. No GT motion is accepted.
        text, lengths, scale = args
        self.calls.append({'text': list(text), 'lengths': lengths.tolist(),
                           'scale': scale, 'keywords': dict(kwargs)})
        batch, length = len(text), int(lengths.max())
        tags = torch.arange(batch).float() + 10 * (len(self.calls) - 1)
        latent = tags[:, None, None].expand(batch, length, 1).clone()
        return latent, latent.clone()


class TraceVae:
    def decode(self, latent, semantic_idx):
        return latent.repeat_interleave(4, dim=1)


def wrapper_case(evaluation_class, gt_value, mm_samples=0, mm_repeats=3):
    torch.manual_seed(234)
    np.random.seed(234)
    dit = TraceDit()
    output = evaluation_class(TraceVae(), dit, GT(gt_value), batch_size=2,
                              mm_num_samples=mm_samples, mm_num_repeats=mm_repeats,
                              cond_scale=2.5, feature_type='stub_only', history_length=3,
                              sampling_timesteps=2, uncertainty_scale=1.0,
                              scheduling_matrix_type='full_sequence', infer_from_velocity=False)
    return dit.calls, output


def main():
    assert Path(__file__).resolve().parents[1] == ROOT
    os.nice(15)
    torch.set_num_threads(1)
    assert not torch.cuda.is_initialized()
    raw = SOURCE.read_bytes()
    assert sha(raw) == EXPECTED_SOURCE_SHA
    evidence = json.loads(raw)
    assert evidence['commit'] == COMMIT
    sources = {r['path']: r for r in evidence['responses'] if 'path' in r}
    for path, row in sources.items():
        data = row['text'].encode('utf-8')
        assert sha(data) == row['sha256']
        assert hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest() == row['git_blob_sha1']

    evaluator_ast = ast.parse(sources['model/evaluator.py']['text'])
    cls = next(n for n in evaluator_ast.body if isinstance(n, ast.ClassDef) and n.name == 'EvaluationDataset')
    init = next(n for n in cls.body if getattr(n, 'name', None) == '__init__')
    calls = [n for n in ast.walk(init) if isinstance(n, ast.Call)
             and isinstance(n.func, ast.Attribute) and n.func.attr == 'generate']
    assert len(calls) == 1
    assert [ast.unparse(n) for n in calls[0].args] == ['text', 'ids_length', 'cond_scale']
    assert not {'context1', 'context2'} & {n.arg for n in calls[0].keywords}
    namespace = {'torch': torch, 'np': np, 'Dataset': Dataset, 'DataLoader': DataLoader,
                 'VAE': object, 'DiT': object, 'repeat': repeat,
                 'tqdm': lambda iterable, **kwargs: iterable,
                 'features_to_interhuman': lambda a, b, **kwargs: (a, b),
                 'reset_coordinate': lambda a, b: (a, b)}
    compile_nodes([cls], 'ARMS/model/evaluator.py', namespace)
    evaluation_class = namespace['EvaluationDataset']
    before_calls, before = wrapper_case(evaluation_class, 0)
    after_calls, after = wrapper_case(evaluation_class, 999)
    assert before_calls == after_calls
    assert len(before) == len(after) == 2
    for a, b in zip(before.generated_motions, after.generated_motions):
        assert a['text'] == b['text'] and a['motion_lens'] == b['motion_lens']
        assert np.array_equal(a['motion1'], b['motion1']) and np.array_equal(a['motion2'], b['motion2'])
    mm_error = None
    try:
        wrapper_case(evaluation_class, 0, mm_samples=1, mm_repeats=3)
    except IndexError as error:
        mm_error = str(error)
    assert mm_error and 'out of bounds' in mm_error

    # Execute the unchanged DiT.generate method and all of its mask/schedule
    # helpers. Only encode_cond and denoise are CPU stubs; no network is built.
    dit_ast = ast.parse(sources['model/dit.py']['text'])
    dit_cls = next(n for n in dit_ast.body if isinstance(n, ast.ClassDef) and n.name == 'DiT')
    generate = next(n for n in dit_cls.body if getattr(n, 'name', None) == 'generate')
    defaults = dict(zip([a.arg for a in generate.args.args][-len(generate.args.defaults):], generate.args.defaults))
    assert ast.literal_eval(defaults['context1']) is None and ast.literal_eval(defaults['context2']) is None
    helper_names = {'generate_non_pad_mask', 'build_causal_mask', 'generate_attention_mask',
                    'generate_pyramid_scheduling_matrix', 'generate_full_sequence_scheduling_matrix',
                    'generate_autoregressive_scheduling_matrix', 'generate_scheduling_matrix', 'lengths_to_mask'}
    helpers = [n for n in dit_ast.body if isinstance(n, ast.FunctionDef) and n.name in helper_names]
    assert len(helpers) == len(helper_names)
    ns = {'th': torch, 'np': np, 'time': time, 'Optional': Optional,
          'rearrange': rearrange, 'repeat': repeat, 'logger': SimpleNamespace(debug=lambda *a, **k: None)}
    compile_nodes(helpers + [generate], 'ARMS/model/dit.py', ns)
    trace = []

    def denoise(x, **kwargs):
        trace.append({'context1': None if kwargs['context1'] is None else kwargs['context1'].clone(),
                      'context2': None if kwargs['context2'] is None else kwargs['context2'].clone(),
                      'input': x.clone()})
        return x.clone()

    fake = SimpleNamespace(parameters=lambda: iter([torch.zeros(1)]), input_dim=2,
                           max_length=3, num_frame_per_block=0, clip_noise=20,
                           encode_cond=lambda c: (torch.zeros(len(c), 1, 2), torch.ones(len(c), 1)),
                           denoise=denoise)
    torch.manual_seed(235)
    first, second = ns['generate'](fake, ['a', 'b'], torch.tensor([[4, 4], [4, 4]]),
                                   cond_scale=1.0, sampling_timesteps=2, history_length=1)
    assert len(trace) == 4 and first.shape == second.shape == (2, 4, 2)
    assert trace[0]['context1'] is None and trace[0]['context2'] is None
    assert trace[0]['input'].count_nonzero() > 0 and torch.isfinite(trace[0]['input']).all()
    previous_output = trace[1]['input'].reshape(2, 2, 3, 2)
    assert torch.equal(trace[2]['context1'], previous_output[:, 0, 2:3])
    assert torch.equal(trace[2]['context2'], previous_output[:, 1, 2:3])
    eval_ast = ast.parse(sources['eval/eval.py']['text'])
    evaluation = next(n for n in eval_ast.body if getattr(n, 'name', None) == 'evaluation')
    mm_called = any(isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                    and n.func.id == 'evaluate_multimodality' for n in ast.walk(evaluation))
    assert mm_called is False and 'mm_num_samples: 0' in sources['config/eval_interhuman.yaml']['text']
    assert not torch.cuda.is_initialized()
    result = {'schema': 'arms_eval_contract_cpu_trace_v1', 'status': 'passed',
              'source_commit': COMMIT, 'source_metadata_sha256': EXPECTED_SOURCE_SHA,
              'source_sha256': sha(Path(__file__).read_bytes()), 'cpu_only': True,
              'cuda_initialized': torch.cuda.is_initialized(), 'trained_models_loaded': False,
              'GT_perturbation_wrapper_generation_arguments_identical': True,
              'GT_perturbation_stub_outputs_identical': True,
              'model_call_positional_arguments': [ast.unparse(n) for n in calls[0].args],
              'default_history_contexts': [None, None], 'actual_generate_first_denoise_has_no_history': True,
              'actual_generate_next_window_uses_generated_history': True,
              'actual_generate_denoise_calls': len(trace),
              'default_MM_samples': 0, 'default_MM_scoring_call_present': mm_called,
              'enabled_MM_batch2_repeats3_exception': mm_error,
              'source_files': {k: {'sha256': v['sha256'], 'git_blob_sha1': v['git_blob_sha1']}
                               for k, v in sources.items()},
              'scope': 'Literal source AST routing/sampler control flow, synthetic CPU stubs. '
                       'Only text/length/sampling knobs reach generation; no GT prefix in inspected entry. '
                       'GT bodies/coordinate conversion/VAE/denoiser not executed. '
                       'MM opt-in exposes a public wrapper batch indexing error, not evidence that '
                       'the paper MM score was computed with this source. No generation-quality/FID claims.',
              'production_changes': False, 'time_unix': time.time()}
    output = ROOT / 'output/hhc_sota_20261016/arms_eval_contract_cpu_trace_20261017.json'
    with output.open('x') as f:
        json.dump(result, f, indent=2)
        f.write('\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
