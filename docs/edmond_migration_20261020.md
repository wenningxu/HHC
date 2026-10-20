# Edmond migration and three-way source synchronization

User authorized moving the active official InterGen experiment back to the free
Edmond server and synchronizing code on Edmond, Runpod, and GitHub.

## Verified destination and transport

- SSH alias: `newstation`; hostname: `edmond-Z790-Taichi`.
- `/mnt/8T-HDD` had 693 GiB available at inspection; the root disk had only 3.4 GiB.
- Two RTX 4090 devices were idle at initial inspection.
- Existing project `/mnt/8T-HDD/wenning/HHC` is preserved.
- Migration runtime: `/mnt/8T-HDD/wenning/hhc_migration_20261020`.
- Direct Edmond-to-Runpod rsync uses a temporary restricted read-only key.
  No model weights or generated arrays are downloaded locally.
- User namespaces preserve `/workspace/HHC` and runtime paths without sudo.

## Source synchronization policy

The local working tree is the recorded source authority. An explicit first-party
source/configuration/docs/tests manifest is used instead of adding all files.
Weights, datasets, Python environments, third-party checkouts, runtime control
receipts, and private keys are excluded from Git. Existing differing server files
must be archived before replacement. Fixed official upstream source and active
training/loss implementations retain their existing content hashes.

## Handoff acceptance

Static assets are copied first. A stable full checkpoint must be validated and
copied with matching SHA-256, then restored with optimizer, scheduler, and step
counters. A new completed checkpoint after resumed updates is required before
stopping the paid pod. Default official checkpoints do not guarantee RNG or
dataloader-cursor continuity; bitwise training equivalence is not claimed.

HHC remains paused at step 4000. Official training to 704000, original 20-repeat
evaluation/MM, and independent review remain the release gates.

## Accepted handoff

- Source suspended after epoch 744, step **262240**.
  Full checkpoint SHA-256:
  `a6bd69af16a6d645bedde61750f6901b4ec1150e761616e105dd9b993e6ab618`.
- Destination independently verified all **26836** static assets,
  **44155474050** logical bytes. Manifest SHA-256:
  `8aabededa9ea751eac092ec0a144398ac36394a146f850ce463f37914a07b7ce`.
- Original Dataset: **11274** rows, **352** updates per epoch; all **45**
  official source catalogue entries matched.
- Actual restored optimizer: **196** states at step 262240.
  Pending original scheduler hook replayed once: epoch 744 → 745,
  LR `6.956868334186013e-5` → `6.949638438940942e-5`.
- New completed destination checkpoint: epoch 745, step **262592**,
  **2701620469** bytes, 196 matching optimizer states, finite model/optimizer.
  SHA-256:
  `b67e368c8cfb744cd042b3679d2dd894946eed2add90bb7b047571f13cbaf224`.
- Actual training was independently observed beyond **262670**.
  No new FID or SOTA result is claimed.

Current identities at handoff: trainer **138991 / 35423489**;
HHC/eval waiters **139371 / 35441518**, **139469 / 35450774**;
text-only/shape waiters **139470 / 35450785**, **139471 / 35450788**;
observer **139818 / 35465491**. Read current LAUNCH/STATUS and `/proc` rather
than relying on these historical PID values.

Both existing automations remain PAUSED. Runpod is retired from active compute;
the separate control-plane stop result is recorded in
`output/edmond_migration_20261020/runpod_stop_receipt.json` and
`migration/control/compute-owner.json`. Stopping retains its disk backup.
Do not automatically restart the paid pod.

## Runtime adaptations and preserved failures

The original Python 3.8.20 / PyTorch 1.13.1+cu117 / Lightning 1.9.1 environment
was copied. The auxiliary Python 3.12.3 / PyTorch 2.8.0+cu128 environment was
relocated on the HDD, preserving its installed packages and supplying its
original global package path through a `.pth` file. No system Python was replaced.

The namespace selects GPU 0; a namespaced `nvidia-smi` wrapper scopes queue
headroom checks to that GPU. Training remains single-GPU, FP32, batch 32.
The original model, loss, data transformations, and training schedule are unchanged.

The first destination preflight failed because Matplotlib auto-selected Tk.
The outer environment now sets `MPLBACKEND=Agg`; the original source is untouched.
The first observer exited on permission-denied access to another user namespace's
`/proc/PID/cwd`. Its read-only process reader now records `cwd=null` in that case
and still validates PID, birth time, command, liveness, and logged progress.
Both failed logs are retained; neither failure restarted or altered training.

## Code synchronization and future use

`real_clean` is the shared branch. Source-only bundles are verified file-by-file;
different pre-existing files are retained in each checkout's
`output/edmond_migration_20261020/source_backups/`. No server-only first-party
source files were found in the Edmond project after comparison.

The local HTTPS push credential failed. Pushing through Edmond's existing GitHub
SSH identity succeeded; no credentials were copied between hosts. The local
working tree, Edmond project, Edmond execution checkout, Runpod checkout, and
GitHub are aligned to the final handoff commit. Linux executable permissions are
preserved while `core.filemode=false` permits content parity with Windows.

Future changes must be committed and pushed, then applied to both active Edmond
checkouts with the same source manifest. The stopped Runpod copy preserves this
handoff version; if its use is later authorized, synchronize it from GitHub before
running anything. Do not start paid compute just to refresh an inactive checkout.
