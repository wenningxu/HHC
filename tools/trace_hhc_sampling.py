"""Observe actual cascade decisions in a dedicated diagnostic process."""
import inspect
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from models.two_person_transformer import Text2Motion_Transformer


def select_indices(lengths, samples, policy):
    lengths = np.asarray(lengths)
    if lengths.ndim != 1 or not 1 <= samples <= len(lengths) or np.any(lengths <= 0):
        raise ValueError('positive lengths and sample count within request count required')
    if policy == 'prefix':
        return np.arange(samples)
    if policy == 'length_quantiles':
        return np.argsort(lengths, kind='stable')[np.linspace(0, len(lengths)-1, samples, dtype=int)]
    raise ValueError('unknown selection policy')


def trace_cascade(model, *args, **kwargs):
    """Read sampler locals immediately before commitment; never replace sampling."""
    sampler = Text2Motion_Transformer.sample_intra_cas
    lines, start = inspect.getsourcelines(sampler)
    # ponytail: one source-pinned debugger stop; fail closed if the statement moves/changes.
    stops = [start + i for i, line in enumerate(lines)
             if line.strip() == 'ids[:, :, channel] = updated']
    if len(stops) != 1 or sys.gettrace() is not None:
        raise RuntimeError('unique commitment statement and an unused debugger required')
    records = []

    def observe(frame, event, arg):
        if frame.f_code is not sampler.__code__:
            return None
        if event == 'line' and frame.f_lineno == stops[0]:
            v = frame.f_locals
            with torch.no_grad():
                logits = v['channel_logits'].detach().float()
                logp = (logits / max(v['sample_temperature'], 1e-10)).log_softmax(-1)
                best_two = logits.topk(2, dim=-1)
                after = v['ids'].detach().clone()
                after[:, :, v['channel']] = v['updated']
                row = dict(step=v['step'], channel=v['channel'], temperature=v['sample_temperature'])
                values = dict(before_ids=v['ids'], after_ids=after,
                              proposals=v['predictions'], argmax=best_two.indices[..., 0],
                              was_masked=v['was_masked'], newly_fixed=v['newly_fixed'],
                              confidence=v['confidence'],
                              margin=best_two.values[..., 0] - best_two.values[..., 1],
                              proposal_entropy=-(logp.exp() * logp).sum(-1))
                row.update({key: value.detach().cpu().numpy().copy() for key, value in values.items()})
                records.append(row)
        return observe

    sys.settrace(observe)
    try:
        result = sampler(model, *args, **kwargs)
    finally:
        sys.settrace(None)
    return result, records


def summarize_records(records):
    result = {}
    for channel, name in enumerate(('body', 'hands', 'root')):
        rows = [r for r in records if r['channel'] == channel]
        proposed = sum(int(r['was_masked'].sum()) for r in rows)
        accepted = sum(int(r['newly_fixed'].sum()) for r in rows)
        alternative_proposed = sum(int(((r['proposals'] != r['argmax']) & r['was_masked']).sum()) for r in rows)
        alternative_accepted = sum(int(((r['proposals'] != r['argmax']) & r['newly_fixed']).sum()) for r in rows)
        result[name] = dict(proposals=proposed, accepted=accepted,
                            non_argmax_proposals=alternative_proposed,
                            non_argmax_accepted=alternative_accepted)
    return result


def main():
    import argparse
    import json
    import clip
    from tools.export_hhc_official_interx import load_models, seed_everything, sha256_file
    from utils.motion_process import MotionNormalizerTorchInterX

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--vq-checkpoint', default='checkpoints/vq_precleanup_raw12288/net_175000.pth')
    parser.add_argument('--requests', default='output/interx_official_val_requests/validation.npz')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--samples', type=int, default=8)
    parser.add_argument('--repeats', type=int, default=4)
    parser.add_argument('--seed', type=int, default=700123)
    parser.add_argument('--confidence-noise-until-step', type=int, default=0)
    parser.add_argument('--selection', choices=('prefix', 'length_quantiles'), default='prefix')
    arm_options = {'control': ('probability', 0.),
                   'log_probability_noise1': ('probability', 1.),
                   'log_odds_noise1': ('log_odds', 1.),
                   'max_probability': ('max_probability', 0.)}
    parser.add_argument('--arms', nargs='+', choices=tuple(arm_options),
                        default=['control', 'log_probability_noise1', 'log_odds_noise1'])
    args = parser.parse_args()
    if not 0 <= args.confidence_noise_until_step <= 50:
        parser.error('confidence noise cutoff must be in [0, 50]')
    if not 1 <= args.samples <= 32 or args.repeats < 2:
        parser.error('require 1..32 samples and at least2 repeats')
    if len(set(args.arms)) != len(args.arms):
        parser.error('duplicate arms would overwrite artifacts')
    request = np.load(args.requests, allow_pickle=False)
    if args.samples > len(request['names']):
        parser.error('not enough requests')
    indices = select_indices(request['lengths'], args.samples, args.selection)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    args.vq_contract, args.mgt_contract = 'precleanup_raw12288', 'historical_mmm'
    args.mgt_shape, args.inference_conditioning_contract = False, 'paper_text_cfg'
    device = torch.device(args.device)
    tokenizer, model, text_encoder = load_models(args, device)
    normalizer = MotionNormalizerTorchInterX('raw', stats_dir='dataset/interx', stats_prefix='Interx')
    lengths = torch.as_tensor(request['lengths'][indices], device=device).long()
    betas = torch.as_tensor(request['betas'][indices], device=device).float()
    gender = torch.as_tensor(request['gender'][indices], device=device).long()
    names = request['names'][indices].astype(str)
    with torch.inference_mode():
        texts = clip.tokenize(request['captions'][indices].astype(str).tolist(), truncate=True).to(device)
        global_text, words = text_encoder(texts)
        options = dict(m_length=lengths, betas=betas, gender=gender, CFG=3.5,
                       max_steps=50, temperature=0.5)
        summaries, artifacts = {}, {}
        for arm in args.arms:
            ranking, noise = arm_options[arm]
            options.update(confidence_ranking=ranking, confidence_noise_scale=noise,
                           confidence_noise_until_step=args.confidence_noise_until_step if noise else 0)
            summaries[arm] = []
            for repeat in range(args.repeats):
                if repeat == 0:
                    seed_everything(args.seed)
                    plain = model.sample_intra_cas(global_text, words, **options)
                    plain_cpu_rng = torch.get_rng_state().clone()
                    plain_gpu_rng = torch.cuda.get_rng_state(device).clone() if device.type == 'cuda' else None
                seed_everything(args.seed + repeat)
                ids, records = trace_cascade(model, global_text, words, **options)
                if repeat == 0:
                    assert torch.equal(plain, ids), 'observer changed tokens'
                    assert torch.equal(plain_cpu_rng, torch.get_rng_state()), 'observer changed CPU RNG'
                    if plain_gpu_rng is not None:
                        assert torch.equal(plain_gpu_rng, torch.cuda.get_rng_state(device)), 'observer changed GPU RNG'
                motions = torch.zeros(args.samples, 150, 672, device=device)
                for i, length in enumerate(lengths.tolist()):
                    decoded = tokenizer(ids[i:i+1, :(length+3)//4], betas[i:i+1], gender[i:i+1], type='decode')
                    motions[i, :length] = normalizer.backward(decoded).squeeze(0)[:length]
                if not torch.isfinite(motions).all():
                    raise FloatingPointError('nonfinite decoded motion')
                path = args.output_dir / f'{arm}_repeat{repeat:02d}.npz'
                arrays = {key: np.stack([r[key] for r in records]) for key in records[0]}
                np.savez_compressed(path, **arrays, final_ids=ids.cpu().numpy(),
                                    motions=motions.cpu().numpy(), names=names,
                                    lengths=lengths.cpu().numpy())
                summaries[arm].append(summarize_records(records))
                artifacts[path.name] = sha256_file(path)
                print({'arm': arm, 'repeat': repeat, 'stages': len(records)}, flush=True)
    provenance = {key: {'path': str(Path(value).resolve()), 'sha256': sha256_file(value)}
                  for key, value in dict(checkpoint=args.checkpoint, vq_checkpoint=args.vq_checkpoint,
                                        requests=args.requests, implementation=__file__,
                                        sampler=inspect.getsourcefile(Text2Motion_Transformer)).items()}
    payload = dict(status='complete', diagnostic_only=True, device=args.device,
                   selection=args.selection, request_indices=indices.tolist(), selection_uses_metrics=False,
                   names=names.tolist(), samples=args.samples, repeats=args.repeats, seed=args.seed,
                   batch_size=args.samples, sampling_steps=50, temperature=.5, cfg_scale=3.5,
                   arm_options={arm: arm_options[arm] for arm in args.arms},
                   confidence_noise_until_step=args.confidence_noise_until_step,
                   baseline_parity='each arm repeat0: identical tokens and CPU/device RNG',
                   provenance=provenance, artifacts=artifacts, counts=summaries,
                   caveat='Different batch size/seed from headline validation; not headline metrics or sampling timing.')
    (args.output_dir / 'summary.json').write_text(json.dumps(payload, indent=2) + '\n')


if __name__ == '__main__':
    main()
