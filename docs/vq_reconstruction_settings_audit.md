# Inter-X VQ reconstruction settings audit

Updated: 2026-08-28

## Fixed-protocol results

All numbers below use the same frozen Inter-X request set and evaluator unless
explicitly marked as historical W&B validation.

The `0.294417` raw-12288 result is not accepted from a metric value alone.
`output/vq_historical_eval/net_175000_reconstruction_ordinary_fid.json`
records and re-hashes the official full-672D evaluator options, checkpoint,
wrapper, option loader, and the main-only scorer.  The MGT repair gate and the
final review report reject this reconstruction result if any of those records
is missing or changes.  It also freezes the reconstruction scorer runtime as
CPU with the recorded PyTorch/CUDA build; changing that runtime invalidates the
repair gate even if the rounded FID is unchanged.

| Checkpoint | VQ size / latent | MPJPE | Reconstruction FID | Notes |
|---|---:|---:|---:|---|
| Current failed paper-text run, 150k | 8096 / 256 (96/96/64) | 0.143921 m | 0.592939 | 96-frame crops, shape mask 0, 150k integrated warm-up/cosine |
| Historical MMM, 2026-01-30, 175k | 12288 / 256 (96/96/64) | 0.136261 m | 0.294417 | Ordinary-only official full-672D evaluator; 64-frame crops, 300k cosine horizon; historical W&B best FID was 0.261876 at 175k |
| Historical MMM, 2025-11-09, best FID | 8096 / 64 (24/24/16) | 0.143384 m | 0.276392 | 64-frame crops, shape mask 0.1, 300k main phase; historical W&B best FID was 0.216716 |

The old and current 8096 checkpoints differ by only 0.000537 m (0.54 mm) in
MPJPE, while their reconstruction FIDs differ by 0.316547. Therefore codebook
size and mean joint error do not explain the failed reconstruction
distribution.

## Provenance-backed setting differences

The exact source snapshot for the failed run is preserved under
`output/vq-interx-sep/2026-08-26-00-09-40_paper_vq_8096_150k_seed123_exact_rawstats_ema_safe/source`.
Compared with the pre-cleanup developer implementation:

- `utils/losses.py` has no mathematical difference.
- The VQ class differs materially only in `shape_mask_prob`, changed from 0.1
  to 0. The tensor-allocation change in `merge_body_hand` is equivalent.
- The failed run forced 96-frame crops; historical successful runs logged
  64-frame crops.
- The failed run used 150k total updates with warm-up included in that horizon.
  The historical training source/logs use 999 separate warm-up updates followed
  by a 300k cosine main phase.
- The failed run used AdamW weight decay 1e-6. The January successful run used
  2e-5; the November 8096 run used 0.
- The failed run reported the terminal 150k checkpoint. Historical runs saved
  the checkpoint selected by validation reconstruction FID.

The train split aliases `dataset/interx/train.txt` and
`dataset/interx/splits/train.txt` are byte-identical (SHA-256
`47c5c9a09d7293e90e127fb49ba9dc9398afbff66c8f2b45ada3ef6c4453157a`).
The raw mean/std files used by HHC and MMM are also byte-identical.

## Ruled-out normalization hypothesis

The failed run's source snapshot and manifest prove that the training dataset
already passed `motion_rep=raw` to `MotionNormalizerTorchInterX`; it was not
trained with the later root/velocity statistics. A 128-sample diagnostic that
forces root statistics gives MPJPE 0.726007 m, far worse than raw-stat
reconstruction. The January checkpoint also fails badly when root statistics
are forced. Normalization representation is therefore not the recovery lever.

## Recovery experiment

The queued recovery run deliberately keeps the paper architecture fixed at
8096 codes and 256 total latent dimensions. It restores only the historical
result-producing settings:

- crop length 64;
- shape-mask probability 0.1;
- 999 separate warm-up updates plus a 300k cosine main phase;
- AdamW weight decay 2e-5;
- deterministic held-out validation reconstruction FID every 5k updates;
- selection of `net_best_validation_fid.pth`, with the frozen test set used
  only once for reporting.

The launcher is `tools/queue_vq_recovery_after_interact2ar.sh`. After VQ
validation, the downstream MGT and HHC evaluation retain the user-fixed
sampler: `sample_intra_cas`, 50 outer steps, temperature 0.5, CFG 3.5, contact
guidance 1.0, Root -> Body -> Hands.

## Machine-readable evidence

- Current reconstruction:
  `output/vq_paper_eval/net_150000_reconstruction.json` and
  `output/vq_paper_eval/net_150000_reconstruction_fid.json`
- Historical January reconstruction:
  `output/vq_historical_eval/net_175000_reconstruction.json` and
  `output/vq_historical_eval/net_175000_reconstruction_ordinary_fid.json`
- Historical 8096 reconstruction:
  `output/vq_historical_8096_eval/net_best_fid_reconstruction.json` and
  `output/vq_historical_8096_eval/net_best_fid_reconstruction_fid.json`
- Root-stat falsification diagnostic:
  `output/vq_paper_reconstruction/root_stats_limit128.json`
