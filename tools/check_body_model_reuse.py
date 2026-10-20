"""Check static BodyModel reuse across different samples, including input gradients."""
import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from optimization.human_body_prior.body_model.body_model import BodyModel
from tools.run_interx_shape_augmentation import atomic_json, sha256_file


def evaluate(model, values):
    inputs = {k: v.detach().clone().requires_grad_() for k, v in values.items()}
    out = model(**inputs)
    gradients = torch.autograd.grad(out.v.square().mean() + out.Jtr.square().mean(), tuple(inputs.values()))
    return [out.v.detach(), out.Jtr.detach(), *[g.detach() for g in gradients]]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    torch.manual_seed(1234)
    rows = []
    for gender in ('NEUTRAL', 'MALE', 'FEMALE'):
        path = Path(f'body_models/smplx/SMPLX_{gender}.npz')
        model = BodyModel(str(path), num_betas=10).to(args.device)
        assert not list(model.parameters())
        buffers = {k: v.clone() for k, v in model.named_buffers()}
        samples = [{k: torch.randn(n, d, device=args.device) * .1
                    for k, d in [('pose_body', 63), ('pose_hand', 90),
                                 ('root_orient', 3), ('trans', 3), ('betas', 10)]}
                   for n in (4, 7)]
        first = evaluate(model, samples[0])
        evaluate(model, samples[1])
        repeated = evaluate(model, samples[0])
        fresh = BodyModel(str(path), num_betas=10).to(args.device)
        independent = evaluate(fresh, samples[0])
        errors = [float((a - b).abs().max()) for a, b in zip(first, repeated)]
        fresh_errors = [float((a - b).abs().max()) for a, b in zip(first, independent)]
        assert all(torch.isfinite(x).all() for x in first + repeated + independent)
        assert all(torch.equal(v, buffers[k]) for k, v in model.named_buffers())
        assert all(x == 0 for x in errors + fresh_errors), (gender, errors, fresh_errors)
        rows.append({'gender': gender, 'asset_sha256': sha256_file(path),
                     'repeat_max_abs_errors': errors, 'fresh_max_abs_errors': fresh_errors,
                     'buffers_unchanged': True})
        del model, fresh, buffers, first, repeated, independent
    atomic_json(args.output, {'device': args.device, 'rows': rows,
        'implementation_sha256': sha256_file(Path(__file__)),
        'scope': 'FP32 A4/B7/A4 and fresh model; vertices, joints, five input gradients. Not full optimizer equivalence or speed acceptance.'})
    print(f'Passed three-gender reuse: {args.output}', flush=True)


if __name__ == '__main__':
    main()
