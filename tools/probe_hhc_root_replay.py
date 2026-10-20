#!/usr/bin/env python3
"""Diagnostic root-token trajectory replay; production sampling is unchanged."""
import argparse
import inspect
import json
import sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from models.two_person_transformer import Text2Motion_Transformer


def trace_root_replay(model, reference_rows, *args, **kwargs):
    sampler = Text2Motion_Transformer.sample_intra_cas
    lines, start = inspect.getsourcelines(sampler)
    # ponytail: source-pinned debugger intervention, diagnostic only; fail closed on drift.
    stops = [start+i for i, line in enumerate(lines) if line.strip() == 'ids[:, :, channel] = updated']
    if len(stops) != 1 or sys.gettrace() is not None:
        raise RuntimeError('unique commitment statement and an unused debugger required')
    rows = [r for r in reference_rows if int(r['channel']) == 2]
    steps = [int(r['step']) for r in rows]
    if not steps or steps != sorted(set(steps)):
        raise ValueError('nonempty strictly increasing root schedule required')
    by_step, consumed = dict(zip(steps, rows)), []

    def observe(frame, event, arg):
        if frame.f_code is not sampler.__code__:
            return None
        if event == 'line' and frame.f_lineno == stops[0]:
            v = frame.f_locals
            if v['channel'] == 2:
                step = int(v['step'])
                if step not in by_step:
                    raise ValueError('missing reference root stage')
                row = by_step[step]
                before = torch.as_tensor(row['before_ids'], device=v['ids'].device)
                after = torch.as_tensor(row['after_ids'], device=v['ids'].device)
                if (before.shape != v['ids'].shape or after.shape != before.shape
                        or before.dtype != torch.long or after.dtype != torch.long):
                    raise ValueError('reference shape/dtype mismatch')
                current, replacement = v['current'], after[..., 2]
                if not torch.equal(current, before[..., 2]):
                    raise ValueError('reference root history mismatch')
                valid, mask = v['valid_mask'], v['mask_id']
                allowed = ((replacement >= 0) & (replacement < model.num_vq)) | (replacement == mask)
                if not allowed[valid].all():
                    raise ValueError('invalid reference root vocabulary')
                if not torch.equal(replacement[current != mask], current[current != mask]):
                    raise ValueError('reference changed fixed root/PAD/END')
                if not torch.equal(((replacement == mask) & valid).sum(1),
                                   ((v['updated'] == mask) & valid).sum(1)):
                    raise ValueError('reference mask-count schedule mismatch')
                v['updated'].copy_(replacement)
                consumed.append(step)
        return observe

    sys.settrace(observe)
    try:
        result = sampler(model, *args, **kwargs)
        if consumed != steps:
            raise ValueError('unused or repeated reference root stages')
        return result
    finally:
        sys.settrace(None)


def sample_with_control_root_replay(model, *args, **kwargs):
    """Two passes; advance outer RNG exactly as the ordinary control batch."""
    from tools.trace_hhc_sampling import trace_cascade
    if kwargs.pop('type', 'sample_intra_cas') != 'sample_intra_cas':
        raise ValueError('root replay requires the cascade sampler')
    if kwargs.get('channel_schedule', 'interleaved') != 'interleaved':
        raise ValueError('root replay requires interleaved control')
    device = args[0].device
    cpu_before = torch.get_rng_state().clone()
    gpu_before = torch.cuda.get_rng_state(device).clone() if device.type == 'cuda' else None
    control_options = dict(kwargs, confidence_noise_scale=0., confidence_noise_scope='all',
                           confidence_ranking='probability')
    control, records = trace_cascade(model, *args, **control_options)
    devices = [device.index if device.index is not None else torch.cuda.current_device()] if gpu_before is not None else []
    # Keep the baseline batch chain independent of the candidate's extra random draws.
    with torch.random.fork_rng(devices=devices):
        torch.set_rng_state(cpu_before)
        if gpu_before is not None:
            torch.cuda.set_rng_state(gpu_before, device)
        candidate = trace_root_replay(model, records, *args, **kwargs)
    if not torch.equal(candidate[..., 2], control[..., 2]):
        raise AssertionError('root replay changed reference root tokens')
    return candidate, control


def main():
    import clip
    from tools.export_hhc_official_interx import load_models, seed_everything, sha256_file
    from tools.score_hhc_validation_mm import duplicate_motion_summary
    from utils.motion_process import MotionNormalizerTorchInterX
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    a = parser.parse_args()
    manifest_path = a.reference_dir / 'summary.json'
    manifest = json.loads(manifest_path.read_text())
    if manifest['status'] != 'complete' or a.output_dir.exists():
        raise ValueError('complete reference and a new output directory required')
    for artifact in manifest['provenance'].values():
        if sha256_file(artifact['path']) != artifact['sha256']:
            raise ValueError(f'reference provenance changed: {artifact["path"]}')
    request = np.load(manifest['provenance']['requests']['path'], allow_pickle=False)
    indices = np.asarray(manifest['request_indices'])
    names, lengths = request['names'][indices].astype(str), request['lengths'][indices]
    np.testing.assert_array_equal(names, manifest['names'])
    device = torch.device(a.device)
    options = SimpleNamespace(checkpoint=manifest['provenance']['checkpoint']['path'],
        vq_checkpoint=manifest['provenance']['vq_checkpoint']['path'], vq_contract='precleanup_raw12288',
        mgt_contract='historical_mmm', mgt_shape=False, inference_conditioning_contract='paper_text_cfg')
    tokenizer, model, text_encoder = load_models(options, device)
    normalizer = MotionNormalizerTorchInterX('raw', stats_dir='dataset/interx', stats_prefix='Interx')
    betas = torch.as_tensor(request['betas'][indices], device=device).float()
    genders = torch.as_tensor(request['gender'][indices], device=device).long()
    sampling = dict(m_length=torch.as_tensor(lengths, device=device), betas=betas, gender=genders,
                    CFG=manifest['cfg_scale'], max_steps=manifest['sampling_steps'], temperature=manifest['temperature'])
    a.output_dir.mkdir(parents=True, exist_ok=False)
    motions, tokens, root_rms, artifacts = [], [], [], {}
    with torch.inference_mode():
        text = clip.tokenize(request['captions'][indices].astype(str).tolist(), truncate=True).to(device)
        global_text, words = text_encoder(text)
        for repeat in range(manifest['repeats']):
            filename = f'control_repeat{repeat:02d}.npz'
            path = a.reference_dir / filename
            if sha256_file(path) != manifest['artifacts'][filename]:
                raise ValueError('reference trace artifact changed')
            with np.load(path, allow_pickle=False) as archive:
                np.testing.assert_array_equal(archive['names'], names)
                np.testing.assert_array_equal(archive['lengths'], lengths)
                rows = [{k: archive[k][i] for k in ('step', 'channel', 'before_ids', 'after_ids')}
                        for i in range(len(archive['step']))]
                seed = manifest['seed'] + repeat
                seed_everything(seed)
                plain = model.sample_intra_cas(global_text, words, **sampling)
                cpu_rng = torch.get_rng_state().clone()
                gpu_rng = torch.cuda.get_rng_state(device).clone() if device.type == 'cuda' else None
                np.testing.assert_array_equal(plain.cpu().numpy(), archive['final_ids'])
                seed_everything(seed)
                replay = trace_root_replay(model, rows, global_text, words, **sampling)
                assert torch.equal(plain, replay) and torch.equal(cpu_rng, torch.get_rng_state())
                if gpu_rng is not None:
                    assert torch.equal(gpu_rng, torch.cuda.get_rng_state(device))
                seed_everything(seed)
                ids = trace_root_replay(model, rows, global_text, words, **sampling,
                                        confidence_ranking='log_odds', confidence_noise_scale=1.)
                assert torch.equal(ids[..., 2], plain[..., 2])
                decoded = torch.zeros(len(names), 150, 672, device=device)
                control_decoded = torch.zeros_like(decoded)
                for i, length in enumerate(lengths.tolist()):
                    for code, target_motion in ((ids, decoded), (plain, control_decoded)):
                        raw = tokenizer(code[i:i+1, :(length+3)//4], betas[i:i+1], genders[i:i+1], type='decode')
                        target_motion[i, :length] = normalizer.backward(raw).squeeze(0)[:length]
                if not torch.isfinite(decoded).all() or not torch.isfinite(control_decoded).all():
                    raise ValueError('nonfinite replay generation')
                motion = decoded.cpu().numpy()
                control_motion = control_decoded.cpu().numpy()
                valid = np.arange(150)[None, :] < lengths[:, None]
                root_mask = torch.as_tensor(tokenizer.root_mask).cpu().numpy()
                delta = (motion - control_motion)[..., root_mask][valid]
                root_rms.append(float(np.sqrt(np.mean(delta**2))))
            target = a.output_dir / f'replay_repeat{repeat:02d}.npz'
            np.savez_compressed(target, motions=motion, control_motions=control_motion,
                                final_ids=ids.cpu().numpy(), names=names, lengths=lengths)
            artifacts[target.name] = sha256_file(target)
            motions.append(motion); tokens.append(ids.cpu().numpy())
            print({'repeat': repeat, 'control_token_and_rng_parity': True, 'root_ids_equal': True}, flush=True)
    tokens = np.stack(tokens, axis=1)
    pair_counts = dict.fromkeys(('body', 'hands', 'root'), 0)
    for i, length in enumerate(lengths):
        for first in range(manifest['repeats']):
            for second in range(first):
                for channel, name in enumerate(pair_counts):
                    pair_counts[name] += int(np.array_equal(tokens[i, first, :(int(length)+3)//4, channel],
                                                           tokens[i, second, :(int(length)+3)//4, channel]))
    payload = dict(status='complete', diagnostic_only=True, device=a.device,
        reference={'path': str(manifest_path.resolve()), 'sha256': sha256_file(manifest_path)},
        implementation_sha256=sha256_file(__file__), artifacts=artifacts, samples=len(names),
        normalization={str(path): sha256_file(path) for path in (normalizer.mean_path, normalizer.std_path)},
        repeats=manifest['repeats'], names=names.tolist(),
        control_parity='every repeat: archived tokens; fresh self-replay tokens and CPU/device RNG',
        duplicate_motions=duplicate_motion_summary(np.stack(motions, axis=1), lengths),
        identical_token_pairs=pair_counts, root_feature_rms_vs_control=root_rms,
        caveat='Root token history replayed from each paired generated control, not GT. Joint VQ decoding may change root geometry. Not official MM/FID or a quality claim.')
    (a.output_dir / 'summary.json').write_text(json.dumps(payload, indent=2) + '\n')


if __name__ == '__main__':
    main()
