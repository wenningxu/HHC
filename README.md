# HHC: Two-Person Motion Generation

This repository is being cleaned and reorganized from the original MMM codebase into a two-person motion generation project.

## Current Scope

The active code paths are focused on:

- InterX two-person text-conditioned motion generation.
- Chi3D two-person VQ and action-conditioned generation experiments.
- Contact-aware post-processing and penetration/multimodality evaluation utilities.

Legacy single-person MMM/HumanML/KIT entry scripts and the bundled Text2Motion training/evaluation package have been removed from the active tree.

## Model Layout

Selected active model implementations:

```text
models/vqvae_sep3_shape.py
models/two_person_transformer.py
```

Compatibility export modules are kept for older imports:

```text
models/vqvae_sep.py
models/t2i_trans.py
models/vqvae.py
models/vqvae_hi.py
models/vqtrans.py
models/t2m_trans_uplow.py
```

Ablation and experimental variants are archived separately:

```text
models/vq_ablation/
models/transformer_ablation/
```

Legacy single-person KIT/HumanML VQ models and older transformer/VAE experiments are no longer active top-level implementations; their code lives under the ablation folders above. Shared low-level modules such as encoders, decoders, quantizers, ResNet blocks, cross-attention blocks, and positional encodings remain in the top-level `models/` package.

## Main Entrypoints

```bash
python train_t2i_trans.py
python train_vq_chi3d.py
python train_mgt_chi3d.py
python generate.py
python generate_mm_samples.py
python optimize_chi3d_gt.py
python optimize_full_body.py
```

Diffusion/MLD experiments are archived under:

```text
archive/diffusion_mld/
```

## Local Data Layout

Large datasets, checkpoints, body models, generated samples, and experiment outputs are intentionally ignored by Git. Keep them local under the expected paths:

```text
dataset/interx/
dataset/chi3d/
body_models/
checkpoints/
output/
result/
results/
final_demo/
wandb/
```

Small normalization statistics required by the code are versioned:

```text
dataset/interx/*Mean*.npy
dataset/interx/*Std*.npy
dataset/chi3d/stats/*.npy
```

## Cleanup Notes

Reusable motion masks are centralized in `utils/motion_masks.py`. Shared transformer blocks live in `models/motion_transformer.py`. Some model classes still retain legacy class names for checkpoint compatibility and should be simplified only after checkpoint loading paths are confirmed.

## License

This repository is derived from the MMM codebase and retains the original license file. External datasets, SMPL/SMPL-X assets, CLIP, PyTorch3D, and evaluator assets have their own licenses and usage restrictions.
