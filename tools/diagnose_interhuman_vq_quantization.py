"""Diagnostic-only quantizer bypass on immutable snapshots, not model selection."""
import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from dataset.interhuman_fair_requests import validation_request_loader
from models.model_tools import build_model
from tools.score_interhuman_vq_geometry import error_sums, finalize
from tools.score_interhuman_vq_validation import atomic_json
from utils.interhuman_artifacts import sha256_file
from utils.interhuman_vq_evaluation import reconstruct_padded_interhuman_pair


class BypassProbe(torch.nn.Module):
    def __init__(self, model, bypass):
        super().__init__()
        self.model = model
        self.bypass = bypass
        self.temporal_downsample = model.temporal_downsample
        self.counts = [torch.zeros(model.num_code, dtype=torch.long, device=next(model.parameters()).device) for _ in range(2)]

    def forward(self, motion, type='full'):
        assert type == 'full' and not self.model.training
        values = []
        for channel, (latent, quantizer) in enumerate(zip(self.model._encode_latents(motion), self.model.quantizers)):
            quantized, _, _ = quantizer(latent)
            if not self.bypass:
                indices = quantizer.quantize(latent.permute(0, 2, 1).reshape(-1, latent.shape[1]))
                self.counts[channel] += torch.bincount(indices, minlength=self.model.num_code)
            values.append(latent if channel in self.bypass else quantized)
        output = self.model._postprocess(self.model.decoder(torch.cat(values, dim=1)))
        return output, motion.new_zeros(()), motion.new_zeros(2)


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--training-manifest', required=True)
    parser.add_argument('--device', default='cuda:1')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    manifest = json.loads(Path(args.training_manifest).read_text())
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    model = build_model(SimpleNamespace(**manifest['resolved_arguments']), 'interhuman_vq')
    model.load_state_dict(checkpoint['net'], strict=True)
    model.to(args.device).eval()
    before = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    loader = validation_request_loader(num_workers=0)
    results = {}
    for name, bypass in [('quantized', ()), ('body_continuous', (0,)), ('root_continuous', (1,)), ('both_continuous', (0, 1))]:
        probe, totals = BypassProbe(model, bypass), {}
        for _, _, p1, p2, lengths in loader:
            q1, q2 = reconstruct_padded_interhuman_pair(probe, p1.to(args.device), p2.to(args.device), lengths)
            for target, prediction in ((p1, q1.cpu()), (p2, q2.cpu())):
                for key, value in error_sums(target, prediction, lengths, loader.dataset.adapter.mean, loader.dataset.adapter.std).items():
                    totals[key] = totals.get(key, 0) + value
        results[name] = finalize(totals)
        if not bypass:
            results[name]['code_usage'] = []
            for count in probe.counts:
                probability = count.double() / count.sum()
                positive = probability[probability > 0]
                results[name]['code_usage'].append({'active': int((count > 0).sum()), 'tokens': int(count.sum()), 'perplexity': float((-positive * positive.log()).sum().exp()), 'histogram': count.cpu().tolist()})
        print(name, results[name]['full_mpjpe_m'], flush=True)
    assert all(torch.equal(before[k], v.cpu()) for k, v in model.state_dict().items()), 'probe changed VQ state'
    atomic_json(args.output, {
        'status': 'complete', 'purpose': 'diagnostic_only_not_checkpoint_selection',
        'caveat': 'Continuous encoder latents can be off the decoder training distribution; bypass is not an achievable discrete-model score or proof of causal codebook collapse.',
        'token_count_policy': 'ceil(length/4), includes last partially padded block; no all-padding blocks',
        'checkpoint': str(Path(args.checkpoint).resolve()), 'checkpoint_sha256': sha256_file(Path(args.checkpoint)),
        'step': checkpoint['optimizer_step'], 'model_state_unchanged': True,
        'validation_manifest_sha256': sha256_file(ROOT / 'output/interhuman_validation_requests/manifest.json'),
        'implementation_sha256': sha256_file(Path(__file__)), 'results': results,
    })


if __name__ == '__main__':
    main()
