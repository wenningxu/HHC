"""Pinned public InterGen/InterMask/TIMotion main or MM, retaining native lengths.

Formal inference requires real HHC val selection first. --inspect-only is CPU
strict-load acceptance and cannot produce generations or a benchmark score.
"""
import argparse
from contextlib import contextmanager, nullcontext
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import h5py
import numpy as np
import torch
import yaml
from dataset.interhuman_native_baseline_h5 import METHODS, NativeBaselineH5, native_length
from dataset.interhuman_generated_h5 import MEAN_SHA256, STD_SHA256, mm_repeat_seed
from tools.hhc_text_only_formal_contract import file_sha, read_json, require, validate_selection
from tools.export_hhc_text_only_formal_cloud import load_requests

PINS = {
    'intergen': ('third_party/intergen_base_20261008', 'b1750df4a859443b8a69329f759ce3a7503ae179',
        'checkpoints/intergen.ckpt', 2701617525, 'bab341123c27af9d8c3ee7ebc43c2d341bfd0f4103425d06f8af6c717a086f8d'),
    'timotion': ('third_party/TIMotion', '172580fd7f2551a7cc10fde1f15ff73315742e65',
        'checkpoints/timotion.ckpt', 1430359445, 'f364605f7e67cc272dc31810f384ff0dedb564331442257bf776f8d7285ffcee'),
}


@contextmanager
def request_rng(seed, device):
    """Local request seeds preserve caller RNG, independent of request order."""
    py, numpy = random.getstate(), np.random.get_state()
    devices = [torch.device(device).index or 0] if torch.device(device).type == 'cuda' else []
    with torch.random.fork_rng(devices=devices):
        try:
            random.seed(seed)
            np.random.seed(seed % (2**32))
            torch.manual_seed(seed)
            yield
        finally:
            random.setstate(py)
            np.random.set_state(numpy)


def load_public_model(method, device):
    """One method per process, before any HHC models namespace import."""
    require('models' not in sys.modules, 'Official models namespace already shadowed')
    from dataset.interhuman_hhc_adapter import OfficialInterHumanAdapter
    adapter = OfficialInterHumanAdapter()
    if method == 'intermask':
        sys.path.insert(0, str(ROOT / 'tools'))
        from export_official_intermask_interhuman_vq_reconstruction import build_official_vq, require_official_sources, INTERMASK_ROOT
        require_official_sources()
        from models.mask_transformer.transformer import MaskTransformer
        source = INTERMASK_ROOT
        from unittest.mock import patch
        cpu_constructor = torch.device(device).type == 'cpu'
        # Released reset_codebook hardcodes .cuda(). For CPU inspection only,
        # redirect that allocation; strict weights/architecture stay unchanged.
        with patch.object(torch.Tensor, 'cuda', lambda tensor, *a, **k: tensor) if cpu_constructor else nullcontext():
            vq, _ = build_official_vq(device)
        for relative, digest in {
            'models/mask_transformer/transformer.py': '652f5639e22414816ac86008a22e8211b04f1275afa01032fadedc34a2389379',
            'checkpoints/interhuman/trans_default/opt.txt': 'eea88116566997576f28c31dc0bf5e1407645f9d9a56c45d638a4da22b739535',
        }.items():
            require(file_sha(source / relative) == digest, 'Pinned InterMask transformer/options changed')
        model = MaskTransformer(code_dim=512, cond_mode='text', latent_dim=384,
            ff_size=1024, num_layers=6, num_heads=6, dropout=.2, clip_dim=768,
            cond_drop_prob=.1, clip_version='ViT-L/14@336px',
            opt=SimpleNamespace(num_tokens=1024, dataset_name='interhuman')).eval()
        checkpoint = source / 'checkpoints/interhuman/trans_default/model/best_fid.tar'
        expected = '392ca8c9e42c2c23aeb97ba7b810e8fa4bdcc5f4839a5a909be2f56bf1698665'
        require(checkpoint.stat().st_size == 651546446 and file_sha(checkpoint) == expected, 'InterMask MGT changed')
        saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
        loaded = model.load_state_dict(saved.get('t2m_transformer', saved.get('trans')), strict=False)
        require(not loaded.unexpected_keys and all(key.startswith('clip_') for key in loaded.missing_keys), 'Non-CLIP InterMask key missing')
        del saved
        model = model.to(device)
        def generate(row):
            ids = model.generate([row['caption']], torch.tensor([row['processed_length'] // 4], device=device),
                20, 2, topk_filter_thres=.9, temperature=1)
            return torch.stack([vq.forward_decoder(part.unsqueeze(-1))[0] for part in ids.chunk(2, dim=1)], dim=1)
        sampler = {'steps': 20, 'cfg': 2, 'temperature': 1, 'topk_filter_thres': .9, 'gsample': False, 'batch_size': 1}
        extra = {'vq_checkpoint_sha256': 'af57bbdde94ed4d4a01f3ba5f9fe24f0012c2a37b3e38c2fa57a2a91ebcc1272',
            'CPU_inspection_only_VQ_constructor_cuda_allocation_redirect': cpu_constructor}
        model_label = 'InterMask public InterHuman'
    else:
        relative, revision, checkpoint_name, size, expected = PINS[method]
        source = ROOT / relative
        require(subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=source, text=True).strip() == revision, 'Baseline revision changed')
        os.chdir(source)
        sys.path.insert(0, str(source))
        cfg_values = yaml.safe_load((source / 'configs/model.yaml').read_text())
        if method == 'intergen':
            from models.intergen import InterGen
            cfg = SimpleNamespace(**cfg_values)
            model = InterGen(cfg).eval()
            model_label = 'InterGen public'
        else:
            from yacs.config import CfgNode
            from models import TIMotion
            cfg = CfgNode(cfg_values)
            cfg.NUM_HEADS = 16; cfg.NUM_LAYERS = 5; cfg.LATENT_DIM = 512
            cfg.LPA = True; cfg.conv_layers = 1; cfg.dilation_rate = 1; cfg.norm = 'AdaLN'
            model = TIMotion(cfg).eval()
            model_label = 'TIMotion public Transformer+LPA'
        require(cfg.STRATEGY == 'ddim50' and cfg.CFG_WEIGHT == 3.5, 'Released sampler differs')
        for name, value in (('global_mean.npy', adapter.mean), ('global_std.npy', adapter.std)):
            require(np.array_equal(np.load(source / 'data' / name), value), 'Official normalization changed')
        checkpoint = source / checkpoint_name
        require(checkpoint.stat().st_size == size and file_sha(checkpoint) == expected, 'Public checkpoint changed')
        saved = torch.load(checkpoint, map_location='cpu', weights_only=False, mmap=True)
        loaded = model.load_state_dict({key.removeprefix('model.'): value for key, value in saved['state_dict'].items()}, strict=True)
        del saved
        model = model.to(device)
        def generate(row):
            n = row['processed_length']
            return model.forward_test({'text': [row['caption']], 'motion_lens': torch.tensor([n], device=device)})['output'][0].reshape(n, 2, 262)
        sampler = {'strategy': 'ddim50', 'steps': 50, 'cfg': 3.5, 'batch_size': 1}
        extra = {'config': dict(vars(cfg)) if method == 'intergen' else yaml.safe_load(cfg.dump())}
    require(Path(sys.modules[type(model).__module__].__file__).resolve().is_relative_to(source.resolve()), 'Wrong official model namespace')
    # InterMask is the transferred official source snapshot without .git.
    # Its critical files are pinned above/by require_official_sources; bind the
    # entire Python catalogue for comparison with CPU acceptance before GPU use.
    if method == 'intermask':
        revision = None
        tracked = [str(p.relative_to(source)) for directory in ('models', 'configs', 'utils', 'options')
                   for p in (source / directory).rglob('*.py')]
    else:
        require(not subprocess.check_output(['git', 'diff', '--name-only', 'HEAD', '--', 'models', 'configs', 'utils'], cwd=source, text=True).strip(), 'Official model/config/utils edited')
        revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=source, text=True).strip()
        tracked = subprocess.check_output(['git', 'ls-files', 'models', 'configs', 'utils'], cwd=source, text=True).splitlines()
    hashes = {name: file_sha(source / name) for name in tracked if (source / name).is_file()}
    model.requires_grad_(False)
    metadata = {'method': method, 'model': model_label, 'source_commit': revision,
        'source_files_sha256': hashes, 'checkpoint_sha256': expected, 'checkpoint_bytes': checkpoint.stat().st_size,
        'sampler': sampler, 'gt_motion_or_shape_used': False, 'input': 'caption and frozen requested length only',
        'native_length_policy': 'floor(L/4)*4' if method == 'intermask' else 'L',
        'missing_keys': loaded.missing_keys, 'unexpected_keys': loaded.unexpected_keys,
        'parameters': sum(p.numel() for p in model.parameters()),
        'dataset_sha256': file_sha(ROOT / 'dataset/interhuman_native_baseline_h5.py'),
        'official_VQ_bridge_sha256': file_sha(ROOT / 'tools/export_official_intermask_interhuman_vq_reconstruction.py') if method == 'intermask' else None,
        'exporter_sha256': file_sha(Path(__file__)), **extra}
    def seeded_generate(row, seed):
        with request_rng(seed, device), torch.inference_mode():
            return generate(row).detach().float().cpu()
    return seeded_generate, metadata, adapter


def write_native_h5(path, rows, request_path, entry, method, mode, generate, metadata):
    path = Path(path)
    partial = path.with_name(path.stem + '.partial.h5')
    require(not path.exists() and not partial.exists(), 'Export already exists; never replay')
    count, repeats = len(rows), 1 if mode == 'main' else 30
    lengths = [native_length(method, row['processed_length']) for row in rows]
    seeds = np.asarray([[mm_repeat_seed(row['sample_seed'], repeat) for repeat in range(repeats)] for row in rows], dtype=np.uint64)
    with h5py.File(partial, 'x') as handle:
        handle.attrs.update(schema='interhuman_native_baseline_generation_v1', status='running', split='test',
            method=method, mode=mode, request_file_sha256=entry['sha256'], request_content_sha256=entry['request_sha256'],
            normalization_mean_sha256=MEAN_SHA256, normalization_std_sha256=STD_SHA256, metadata=json.dumps(metadata, sort_keys=True))
        strings = h5py.string_dtype('utf-8')
        handle.create_dataset('names', data=[row['identifier'] for row in rows], dtype=strings)
        handle.create_dataset('texts', data=[row['caption'] for row in rows], dtype=strings)
        handle.create_dataset('requested_lengths', data=[row['processed_length'] for row in rows], dtype=np.int64)
        handle.create_dataset('lengths', data=lengths, dtype=np.int64)
        handle.create_dataset('repeat_seeds', data=seeds)
        shape = (count, 300, 262) if mode == 'main' else (count, 30, 300, 262)
        chunk = (1, 300, 262) if mode == 'main' else (1, 1, 300, 262)
        for key in ('motion1', 'motion2'):
            handle.create_dataset(key, shape, dtype=np.float32, chunks=chunk, compression='lzf', fillvalue=0.)
        for repeat in range(repeats):
            for i, row in enumerate(rows):
                prediction = generate(row, int(seeds[i, repeat]))
                require(tuple(prediction.shape) == (lengths[i], 2, 262)
                    and prediction.dtype == torch.float32 and torch.isfinite(prediction).all().item(), 'Invalid native generation')
                values = prediction.numpy()
                for person, key in enumerate(('motion1', 'motion2')):
                    if mode == 'main': handle[key][i, :lengths[i]] = values[:, person]
                    else: handle[key][i, repeat, :lengths[i]] = values[:, person]
            print(json.dumps({'repeat_complete': repeat + 1, 'prompts': count, 'method': method}), flush=True)
        handle.attrs['status'] = 'complete'
    checked = NativeBaselineH5(partial, request_path, entry, method, mode); checked.close()
    os.replace(partial, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--method', choices=METHODS, required=True)
    parser.add_argument('--inspect-only', action='store_true')
    parser.add_argument('--selection-receipt', type=Path)
    parser.add_argument('--baseline-cpu-receipt', type=Path)
    parser.add_argument('--mode', choices=['main', 'mm'])
    parser.add_argument('--replication', type=int, choices=range(20))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    require(str(ROOT) == '/workspace/HHC', 'Cloud only; no local weights or GPU work')
    output = args.output.resolve()
    require(output.is_relative_to(ROOT / 'output') and not output.exists(), 'New cloud output required')
    torch.set_num_threads(2); torch.set_num_interop_threads(2); os.nice(10)
    if args.inspect_only:
        require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'Strict load inspection must hide GPUs')
        _, metadata, _ = load_public_model(args.method, 'cpu')
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open('x') as stream:
            json.dump({'status': 'public_baseline_strict_CPU_load_passed', 'real_generation': False,
                       'metadata': metadata}, stream, indent=2)
        print(json.dumps({'status': 'public_baseline_strict_CPU_load_passed', 'method': args.method}), flush=True)
        return
    require(args.selection_receipt is not None and args.baseline_cpu_receipt is not None
            and args.mode is not None and args.replication is not None,
            'Formal inference requires actual val selection, baseline CPU acceptance, mode and replication')
    selection_sha = file_sha(args.selection_receipt)
    baseline_sha = file_sha(args.baseline_cpu_receipt)
    receipt = validate_selection(args.selection_receipt)
    baseline = read_json(args.baseline_cpu_receipt)
    require(baseline['status'] == 'public_baseline_strict_CPU_load_passed'
            and baseline['real_generation'] is False and baseline['metadata']['method'] == args.method,
            'Missing actual baseline CPU acceptance')
    rows, request_path, entry = load_requests(receipt, args.mode, args.replication)
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '0', 'Only existing cloud GPU0')
    free = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.free', '--format=csv,noheader,nounits'], text=True).splitlines()
    require(len(free) == 1 and int(free[0]) >= 13 * 1024, 'Insufficient single-GPU headroom')
    torch.cuda.set_per_process_memory_fraction(.5, 0)
    generate, metadata, _ = load_public_model(args.method, 'cuda:0')
    for key in ('method', 'model', 'checkpoint_sha256', 'source_files_sha256', 'sampler', 'source_commit',
                'exporter_sha256', 'dataset_sha256', 'official_VQ_bridge_sha256'):
        require(metadata[key] == baseline['metadata'][key], 'Baseline changed after CPU acceptance: ' + key)
    metadata.update(selection_receipt_sha256=selection_sha, replication=args.replication,
        baseline_cpu_receipt_sha256=baseline_sha,
        mode=args.mode, seed_contract='mm_repeat_seed per request/repeat; torch64/NumPy seed mod 2^32/Python64',
        dataset_sha256=file_sha(ROOT / 'dataset/interhuman_native_baseline_h5.py'))
    output.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    write_native_h5(output, rows, request_path, entry, args.method, args.mode, generate, metadata)
    require(file_sha(args.selection_receipt) == selection_sha
            and file_sha(args.baseline_cpu_receipt) == baseline_sha, 'Frozen acceptance changed during generation')
    print(json.dumps({'status': 'complete_generation_only', 'elapsed_s': time.time() - started,
                      'h5_sha256': file_sha(output), 'metadata': metadata}), flush=True)


if __name__ == '__main__':
    main()
