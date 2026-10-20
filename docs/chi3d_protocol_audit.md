# CHI3D protocol and local-asset audit

## Scope

The locally available processed data under
`/home/wenning/projects/MMM/dataset/chi3d` contain the three official training
subject pairs only.  The frozen `train.txt`, both motion HDF5 files, and all
three contact-signature files contain the same 373 sequence identities after
the explicit `s02→G2`, `s03→G3`, and `s04→G4` mapping.  There is no local
`splits/test.txt`.

A second local search found ProsePose files named
`datasets/processed/CHI3D/test_optimization.pkl` and
`datasets/original/CHI3D/test/test_gpt4v_prompt2_custom_losses.json`.  The name
`test` here denotes ProsePose's internal experiment split, not the official
CHI3D held-out test package: all 504 contact-record keys are from `s03`, one of
the three official training subject pairs, and the processed records point to
the local `CHI3D/train/s03` tree.  The prerequisite audit records hashes for
both files and fails closed if that subject classification changes.  Neither
file makes an official-test claim permissible.

This is consistent with the official site, which describes three subject pairs
in the training set and two in the test set.  The official challenge evaluates
test submissions using metrics including MPJPE, Procrustes-aligned error,
pelvis translation, and mesh error.  See the official
[dataset page](https://ci3d.imar.ro/chi3d),
[challenge protocol](https://ci3d.imar.ro/challenges), and
[download page](https://ci3d.imar.ro/download).  The public processed assets in
MMM therefore do not authorize an “official test” claim.

## Allowed claims

- Results on all 373 local sequences are **official-train, in-sample
  diagnostics** only.  This is suitable for checking GT optimization,
  preprocessing, contact constraints, and reconstruction, but not generation
  generalization.
- Local generalization experiments use a predeclared three-fold
  leave-one-subject-pair-out protocol: train on two of G2/G3/G4 and test on the
  third.  These results must be called **custom CHI3D CV**, not official test.
- An official-test result requires the authorized test package and an official
  challenge-server submission (or equivalent official evaluation access).

For custom CV, each fold must derive normalization, learned codebooks,
checkpoint selection, hyperparameters, and any contact lookup from its two
training groups only.  The held-out group cannot be used for early stopping or
model selection.  Report every fold as well as the macro mean; with only three
groups, do not imply a conventional large-sample confidence interval.

There is an additional model-level guardrail.  MMM's current experimental
`VQVAE_SEP3_Chi3DShape` and `Chi3DActionTransformer` both condition on an
`nn.Embedding(3)` indexed by G2/G3/G4.  In a leave-one-group-out run, the held-out
row would remain at random initialization, so merely replacing `train.txt` in
those scripts is **not** a valid shape-generalization experiment.  Before custom
CV is run, this categorical condition must be replaced by an explicit continuous
shape/skeleton condition (or a predeclared shape-agnostic condition), and a test
must prove that held-out inference contains no untrained category-specific
parameters.  The existing all-373 action-conditioned checkpoints remain
in-sample qualitative diagnostics only.

A separate, currently actionable claim is to evaluate an HHC model trained only
on Inter-X against the 373 CHI3D training sequences.  Because CHI3D is unseen by
that model, this is an **external-data generalization diagnostic**, but it is
still not an official CHI3D-test result.  Generation can use frozen action-to-text
templates and report unpaired physical/contact/diversity metrics.  Pointwise
MPJPE is meaningful only for reconstruction or GT optimization, not for
stochastic action/text-conditioned generation.

## Reproduction

Run:

```bash
conda run -n New_MMM python tools/audit_chi3d_assets.py \
  --output output/chi3d_asset_audit.json
```

The JSON records group/action counts, exact file hashes, HDF5/split/contact-ID
set equality, array finiteness, and the protocol guardrail above.  It fails
closed if required files are missing or identity sets diverge.

Create the frozen custom-CV folds with:

```bash
conda run -n New_MMM python tools/prepare_chi3d_cv_splits.py
```

`configs/chi3d_cv/manifest.json` partitions every source identity exactly once
per fold.  The held-out G group is the test set; a deterministic 10% sample from
each remaining group/action stratum is reserved for inner validation.  The
manifest records every split hash and the ranking rule, so regenerated folds
must be byte-identical before training starts.

Then derive normalization from each fold's **training identities only**:

```bash
conda run -n New_MMM python tools/prepare_chi3d_cv_stats.py
```

Before writing any fold stats, this command recomputes all-data features and
requires them to reproduce MMM's existing mean/std within `1e-6`.  The resulting
`stats_manifest.json` binds each mean/std file to the frozen training split and
motion-HDF5 hashes.  Training and held-out evaluation must both use that fold's
stats; MMM's global CHI3D stats are forbidden for custom CV.
