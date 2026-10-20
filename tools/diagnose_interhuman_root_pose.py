"""Diagnostic GT-based translation removal; not deployable reconstruction scores."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from torch.utils.data import DataLoader

from dataset.interhuman_fair_requests import validation_request_loader
from dataset.interhuman_generated_h5 import InterHumanGeneratedH5Dataset
from tools.score_interhuman_vq_geometry import error_sums
from tools.score_interhuman_vq_validation import atomic_json
from utils.interhuman_artifacts import load_hhc_interhuman_vq, sha256_file
from utils.interhuman_vq_evaluation import reconstruct_padded_interhuman_pair


def decompose(p1, p2, q1, q2, lengths, mean, std):
    """Return additive distances in metres, excluding padding; no rotations fitted."""
    for p, q in ((p1, q1), (p2, q2)):
        error_sums(p, q, lengths, mean, std)  # Reuse shape/length/finite validation.
    if p1.shape != p2.shape:
        raise ValueError('paired tensors must have matching shapes')
    valid = torch.arange(p1.shape[1])[None] < lengths[:, None]
    scale = torch.as_tensor(std, dtype=torch.float64)
    d1, d2 = [((q[valid].double() - p[valid].double()) * scale)[:, :66].reshape(-1, 22, 3)
              for p, q in ((p1, q1), (p2, q2))]
    center = (d1[:, :1] + d2[:, :1]) / 2
    normsum = lambda x: float(torch.linalg.vector_norm(x, dim=-1).sum())
    return {
        'pair_frames': int(valid.sum()),
        'raw_joint_sum': normsum(d1) + normsum(d2),
        'root_sum': normsum(d1[:, 0]) + normsum(d2[:, 0]),
        'pair_root_vector_sum': normsum(d2[:, 0] - d1[:, 0]),
        'pair_center_sum': normsum(center),
        'root_relative_body_sum': normsum(d1[:, 1:] - d1[:, :1]) + normsum(d2[:, 1:] - d2[:, :1]),
        'pair_center_aligned_joint_sum': normsum(d1 - center) + normsum(d2 - center),
    }


def finalize(s):
    n = s['pair_frames']
    if n <= 0:
        raise ValueError('no valid pair frames')
    return {'pair_frames': n, 'raw_mpjpe_m': s['raw_joint_sum'] / (n * 44),
            'root_position_error_m': s['root_sum'] / (n * 2),
            'pair_root_vector_error_m': s['pair_root_vector_sum'] / n,
            'pair_root_center_error_m': s['pair_center_sum'] / n,
            'root_relative_body_mpjpe_m': s['root_relative_body_sum'] / (n * 42),
            'pair_center_aligned_mpjpe_m': s['pair_center_aligned_joint_sum'] / (n * 44)}


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--training-manifest', required=True)
    parser.add_argument('--official-generations', required=True)
    parser.add_argument('--device', default='cuda:1')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    loader = validation_request_loader(num_workers=0)
    manifest_path = ROOT / 'output/interhuman_validation_requests/manifest.json'
    manifest = json.loads(manifest_path.read_text())
    model, provenance = load_hhc_interhuman_vq(args.checkpoint, args.training_manifest, device=args.device)
    before = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    entry = manifest['request']
    generated = InterHumanGeneratedH5Dataset(args.official_generations, Path(entry['path']).resolve(), request_entry=entry, formal=False)
    if len(generated) != 480 or generated.metadata.get('split') != 'val' or generated.metadata.get('status') != 'complete':
        raise ValueError('expected complete frozen 480-val official reconstructions')
    totals = {'hhc': {}, 'official': {}}
    predictions = iter(DataLoader(generated, batch_size=96, shuffle=False))
    try:
        for names, texts, p1, p2, lengths in loader:
            pn, pt, o1, o2, pl = next(predictions)
            if list(names) != list(pn) or list(texts) != list(pt) or not torch.equal(lengths, pl):
                raise ValueError('official requests do not match GT')
            q1, q2 = reconstruct_padded_interhuman_pair(model, p1.to(args.device), p2.to(args.device), lengths)
            for key, a, b in (('hhc', q1.cpu(), q2.cpu()), ('official', o1, o2)):
                sums = decompose(p1, p2, a, b, lengths, loader.dataset.adapter.mean, loader.dataset.adapter.std)
                for k, v in sums.items():
                    totals[key][k] = totals[key].get(k, 0) + v
    finally:
        generated.close()
    assert all(torch.equal(before[k], v.cpu()) for k, v in model.state_dict().items()), 'diagnostic changed model'
    result = {'status': 'complete', 'schema': 'interhuman_root_pose_diagnostic_v1',
              'purpose': 'diagnostic_only_not_model_selection', 'samples': 480,
              'caveat': 'GT-based per-frame translations are oracle diagnostics, not deployable scores. Norm errors are not additive contributions. No rotation/scale alignment.',
              'metrics': {k: finalize(v) for k, v in totals.items()}, 'model_state_unchanged': True,
              'model': provenance, 'official_generations_sha256': sha256_file(args.official_generations),
              'validation_manifest_sha256': sha256_file(manifest_path),
              'request_file_sha256': entry['sha256'], 'implementation_sha256': sha256_file(__file__)}
    atomic_json(args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
