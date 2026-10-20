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

Status at creation: transfer in progress; no completed handoff or new FID claimed.
