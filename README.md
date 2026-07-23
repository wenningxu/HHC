# HHC: Two-Person Motion Generation

InterX two-person VQ-VAE + MaskGIT text-to-motion transformer. This tree is the
minimal active path: InterX VQ training, t2i transformer training, generation,
evaluation, visualization, and the SMPL-X body-model optimization/refinement
step. Chi3D, archived diffusion/MLD, and penetration-volume (`VolumetricSMPL`)
code paths have been removed.

## Model Layout

```text
models/vqvae_sep3_shape.py      # VQVAE_SEP3_shape — InterX two-person VQ-VAE (sep3: body/hand/root)
models/two_person_transformer.py # Text2Motion_Transformer — MaskGIT-style t2i transformer
models/encdec.py resnet.py quantize_cnn.py   # encoder/decoder/EMA quantizer
models/motion_transformer.py trans_encdec.py pos_encoding.py  # transformer blocks
models/evaluator_wrapper.py modules.py    # T2M-style evaluation nets
models/model_tools.py           # build_model registry (interx_vq, interx_transformer)
```

Shared low-level modules in `models/` and `utils/` (FK, losses, motion
processing, rotation conversions, runtime helpers, plotly visualization).

## Main Entrypoints

Run from the repository root (scripts resolve `./dataset/`, `./body_models/`,
`./checkpoints/`, `./output/` relative to CWD):

```bash
python train/train_vq_interx.py   # InterX two-person VQ-VAE — produces checkpoints/vq/net_175000.pth
python train/train_t2i_trans.py   # InterX t2i transformer — loads the VQ ckpt + builds the codebook cache
python gen/generate.py            # generate from text + betas
python gen/generate_mm_samples.py # multimodal generation from dataset selection
python optimize/optimize_full_body.py  # SMPL-X body-model contact-aware refinement (core)
```

Evaluation entrypoints under `eval/`: `calc_pene_from_npz.py`,
`filter_pene_and_compute_mmodality.py`, `extract_val_contacts.py`,
`rank_final_demo_metrics.py` (FID/diversity/matching/multimodality +
penetration). Geometry/SMPL-X helpers live in `tools/test_data.py` (also
generates `checkpoints/vq/code_contact_probs.npy` from the codebook cache).
Shared runtime helpers (masking, save-folder setup, sampling schedules,
plotly visualization) are in `utils/runtime.py`.

## Local Data Layout

Large datasets, checkpoints, body models, generated samples, and experiment
outputs are ignored by Git. Keep them local under the expected paths:

```text
dataset/interx/                 # inputs: motions/{train,val,test,all}.h5, glove/, texts_processed/, betas/, skeleton/ (1..59.npy), {train,val,test,all}.txt, annots/
body_models/                    # smplx/{SMPLX_NEUTRAL,SMPLX_MALE,SMPLX_FEMALE}.npz, smpl/SMPL_NEUTRAL.pkl  (symlink to a local SMPL-X body-model dir)
checkpoints/                    # vq/net_175000.pth + codebook/, contact_token_label/, code_contact_probs.npy; hhi/ text2motion eval ckpts
output/                         # training runs + tensorboard + samples
```

Versioned normalization statistics under `dataset/interx/*Mean*.npy`,
`dataset/interx/*Std*.npy`.

### Data / asset preparation (not in Git)
- `dataset/interx/skeleton/{1..59}.npy`: dual-person SMPLX rest-pose joints per
  InterX group, generated with `prepare_skeleton.py` (uses `body_models/smplx/`
  + per-group betas from the raw `motions/` P1/P2.npz).
- `dataset/interx/betas/<name>.npz`: per-sample `{betas:(2,10), gender:[str,str]}`,
  built with `prepare_betas.py` from the raw `motions/` P1/P2.npz.
- `dataset/interx/motions/all.h5`: concatenation of train/val/test h5 (needed by
  the tokenizer/codebook-gen step).

## Cleanup Notes

Reusable motion masks are in `utils/motion_masks.py`. Shared transformer blocks
are in `models/motion_transformer.py`. `optimation/text_2_regions.py` imports
`google-genai` lazily (only the `ContactExtractor` live-LLM path needs it; the
rest of the module imports without the dependency or network).

## License

Derived from the MMM codebase; retains the original license file. External
datasets, SMPL/SMPL-X assets, CLIP, and evaluator assets have their own licenses.
