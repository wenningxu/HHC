#!/usr/bin/env python
"""Current MMM/HHC numerical parity, NOT certification of January source/training.

Run from the HHC root with the New_MMM Python. Uses four real raw clips on CPU.
Extracts only the reference HHI class to avoid importing unrelated Chi3D code;
shared encoder/quantizer/masks are hash-checked before reuse.
"""
import ast
import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
import models.vqvae_sep3_shape as current
from dataset.dataset_interx import MotionDatasetV2HHI
from utils.losses import ReConsLossInter
from utils.motion_process import MotionNormalizerTorchInterX
from utils.quantizer_state import quantizer_training_state


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def equal(a, b, label):
    if isinstance(a, torch.Tensor):
        assert a.dtype == b.dtype and torch.equal(a, b), label
        assert torch.isfinite(a).all(), label + ': nonfinite'
    elif isinstance(a, dict):
        assert a.keys() == b.keys(), label
        for key in a:
            equal(a[key], b[key], label + '/' + str(key))
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b), label
        for i, (x, y) in enumerate(zip(a, b)):
            equal(x, y, label + '/' + str(i))
    else:
        assert a == b, label


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--steps', type=int, default=1)
    parser.add_argument('--training-manifest', type=Path)
    options = parser.parse_args()
    if options.steps < 1:
        parser.error('--steps must be positive')
    reference = Path('/home/wenning/projects/MMM')
    hashes = {}
    for filename in ('models/encdec.py', 'models/resnet.py',
                     'models/quantize_cnn.py', 'utils/paramUtil.py',
                     'utils/rotation_conversions.py'):
        local, old = (ROOT / filename).read_bytes(), (reference / filename).read_bytes()
        assert local == old, 'Shared dependency differs: ' + filename
        hashes[filename] = hashlib.sha256(local).hexdigest()
    model_path = reference / 'models/vqvae_sep3_shape.py'
    tree = ast.parse(model_path.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef)
               and n.name == 'VQVAE_SEP3_shape')
    namespace = vars(current).copy()
    old_masks = load(reference / 'utils/motion_masks.py', 'mmm_masks')
    for name in ('HHI_BODY_MASK', 'HHI_HAND_MASK', 'HHI_ROOT_MASK', 'HHI_ROOT_MASK_OLD'):
        equal(torch.as_tensor(getattr(current, name)),
              torch.as_tensor(getattr(old_masks, name)), name)
        namespace[name] = getattr(old_masks, name)
    namespace['INTERX_SKELETONS'] = current.INTERX_SKELETONS.copy()
    namespace['FK_Pair_Layer_Batch'] = load(reference / 'utils/fk.py', 'mmm_fk').FK_Pair_Layer_Batch
    exec(compile(ast.Module(body=[cls], type_ignores=[]), str(model_path), 'exec'), namespace)
    old_loss = load(reference / 'utils/losses.py', 'mmm_loss').ReConsLossInter
    args = SimpleNamespace(dataname='hhi', motion_rep='raw', device='cpu', mu=.99,
                           loss_explicit=3., loss_vel=100., loss_root=.1,
                           loss_geo=.01, loss_fc=10., loss_distance_map=.1)
    manifest_hash = None
    if options.training_manifest:
        manifest = json.loads(options.training_manifest.read_text())
        args = SimpleNamespace(**manifest['resolved_arguments'])
        args.device = 'cpu'
        assert args.precision == 'fp32' and args.motion_rep == 'raw'
        manifest_hash = manifest['contract_sha256']
    torch.set_num_threads(4)
    models = []
    for model_class, emb in ((namespace['VQVAE_SEP3_shape'], 256),
                             (current.VQVAE_SEP3_shape, 512)):
        torch.manual_seed(123)
        if options.training_manifest:
            if model_class is current.VQVAE_SEP3_shape:
                from models.model_tools import build_model
                models.append(build_model(args, 'interx_vq'))
            else:
                models.append(model_class(
                    args, args.nb_code, args.code_dim, args.output_emb_width,
                    args.down_t, args.stride_t, args.width, args.depth,
                    args.dilation_growth_rate, args.vq_act, args.vq_norm))
        else:
            models.append(model_class(args, nb_code=12288, code_dim=256,
                                      output_emb_width=emb, down_t=2, stride_t=2,
                                      width=512, depth=3))
    equal(models[0].state_dict(), models[1].state_dict(), 'initialization')
    dataset = MotionDatasetV2HHI('dataset/interx/splits/val.txt',
                                 'dataset/interx/motions/val.h5', 64, 'raw')
    batch = [torch.stack(items) for items in zip(*(dataset[i] for i in (0, 1000, 10000, 20000)))]
    motion, beta, gender, group = batch
    motion = motion.reshape(4, 64, 672)
    optimizers, loss_functions = [], []
    for model, loss_class in zip(models, (old_loss, ReConsLossInter)):
        optimizers.append(torch.optim.AdamW(model.parameters(), lr=.0002,
                                            betas=(.9, .99), weight_decay=2e-5))
        loss_functions.append(loss_class(
            args, 'l1_smooth', 55, model.fk_layers,
            MotionNormalizerTorchInterX(motion_rep='raw')))
    torch.manual_seed(90210)
    reset_steps = []
    for step in range(1, options.steps + 1):
        records = []
        rng = torch.get_rng_state()
        for model, optimizer, loss_fn in zip(models, optimizers, loss_functions):
            torch.set_rng_state(rng)
            optimizer.zero_grad()
            output = model(motion, beta, gender)
            losses = loss_fn(output[0], motion, beta, gender, group)
            (sum(losses.values()) + .02 * output[1]).backward()
            gradients = {n: None if p.grad is None else p.grad.detach().clone()
                         for n, p in model.named_parameters()}
            optimizer.step()
            with torch.no_grad():
                tokens = model(motion, beta, gender, type='encode')
            records.append(dict(output=[x.detach() for x in output],
                                losses={k: v.detach() for k, v in losses.items()},
                                gradients=gradients, state=model.state_dict(),
                                ema=quantizer_training_state(model), tokens=tokens,
                                optimizer=optimizer.state_dict(), rng=torch.get_rng_state()))
        equal(records[0], records[1], f'full_fp32_training_step_{step}')
        if all(q['reset_count'] == 0 for q in records[0]['ema'].values()):
            reset_steps.append(step)
        print(json.dumps({'step': step, 'status': 'PASS'}), flush=True)
    if options.steps >= 21:
        assert 21 in reset_steps, 'EMA reset branch was not exercised'
    print(json.dumps({'status': 'PASS', 'scope': 'current reference, not January provenance or training reproduction',
                      'device': 'cpu', 'batch': 4, 'frames': 64,
                      'steps': options.steps, 'reset_steps': reset_steps,
                      'training_manifest_contract_sha256': manifest_hash,
                      'checks': list(records[0]), 'shared_sha256': hashes,
                      'reference_model_sha256': hashlib.sha256(model_path.read_bytes()).hexdigest()}))


if __name__ == '__main__':
    main()
