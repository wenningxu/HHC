# HumanML3D-train AMASS shape assets

Status: blocked on licensed source assets; no substitute donor distribution is
allowed for the final Inter-X shape-augmentation experiment.

## Local inputs already verified

- HumanML3D root: `/mnt/8T-HDD/HumanML3D`
- HumanML3D index: `/mnt/8T-HDD/HumanML3D/index.csv`
- HumanML3D train split: `/mnt/8T-HDD/HumanML3D/HumanML3D/train.txt`
- MMM SMPL-X female/male models:
  `/home/wenning/projects/MMM/body_models/smplx/`
- Official transfer repository:
  `/mnt/8T-HDD/wenning/smplx-transfer`
- Torch trust-region optimizer:
  `/mnt/8T-HDD/wenning/torch-trust-ncg`

The frozen split resolves to 8,805 unique AMASS source sequences after
excluding HumanAct12 and deduplicating HumanML3D mirror entries. The current
asset audit finds 0/8,805 sequences.

## Licensed inputs still required

1. Download the AMASS SMPL+H G archives corresponding to the datasets named
   in `output/humanml3d_amass_shape_asset_audit.json`, after accepting the
   AMASS dataset licenses. Extract them without renaming into:
   `/mnt/8T-HDD/HumanML3D/amass_data/`.
2. Download the AMASS-compatible extended SMPL+H female and male models with
   16 shape coefficients, after accepting their model license. Place them at:
   - `/mnt/8T-HDD/HumanML3D/body_models/smplh/female/model.npz`
   - `/mnt/8T-HDD/HumanML3D/body_models/smplh/male/model.npz`
3. From the SMPL-X downloads page, download **Model correspondences** and
   place `smplh2smplx_deftrafo_setup.pkl` at:
   `/mnt/8T-HDD/wenning/smplx-transfer/transfer_data/`.

These files require account/license acceptance and therefore are not fetched
automatically. SMPL+H beta values must never be copied directly into SMPL-X:
the official project states that the shape spaces are incompatible. The
pipeline renders neutral SMPL+H meshes and fits gender-matched SMPL-X shapes
with the official deformation correspondence.

Official references:

- AMASS: <https://amass.is.tue.mpg.de/>
- SMPL-X model transfer instructions:
  <https://github.com/vchoutas/smplx/tree/main/transfer_model>
- SMPL+H model preparation notes:
  <https://github.com/vchoutas/smplx/tree/main/tools>

## Fail-closed verification and launch

Run the asset-only preflight first:

```bash
conda run -n New_MMM python tools/run_humanml3d_amass_shape_donor_pipeline.py \
  --stop-after audit
```

It must report all 8,805 required AMASS sequences and all five model/transfer
assets as present. Then run the resumable pipeline:

```bash
conda run -n New_MMM python tools/run_humanml3d_amass_shape_donor_pipeline.py \
  --resume
```

The resulting donor manifest is accepted only when every fitted shape is
finite, is gender matched, and reconstructs the source neutral mesh with at
most 1 cm vertex RMSE. The deterministic seed-123 augmentation plan then
assigns converted donors to 6,238 Inter-X training samples while leaving the
official train/test split, motions, captions and contact timing unchanged.

