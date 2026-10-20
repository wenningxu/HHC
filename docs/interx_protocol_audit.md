# Inter-X / InterMask protocol audit

Audit date: 2026-08-29.

- Official Inter-X source: `/mnt/8T-HDD/wenning/Inter-X`.
- InterMask source: `/mnt/8T-HDD/wenning/InterMask`.
- Shared motion/text/shape assets under `dataset/interx`, plus
  `checkpoints/hhi`, `checkpoints/vq`, and `body_models`, are linked or copied
  byte-identically from `/home/wenning/projects/MMM`.
- The active motion files are MMM's official-preprocessed Inter-X HDF5 files:
  `motions/train.h5` has 9,110 sequences, `val.h5` 570 and `test.h5` 1,708.
  Each HDF5 frame is the official `[56, 6]` axis-angle layout (two persons
  concatenated in the last dimension); the loader converts it to raw 672-D
  `[55 x rot6d + padded root translation] x 2` before tokenization.
- All three active split files are byte-identical to the files in the official
  Inter-X repository.  The HHC training normalization files,
  `Interx_Mean.npy` and `Interx_Std.npy`, are instead MMM/HHC-derived 672-D
  pair-level raw-representation statistics; they are not files published by
  the original Inter-X repository.  Their SHA-256 values are
  `a54dcfd...438446` and `23a2cf52...bc376`.  They are used only to normalize
  HHC VQ/MGT inputs and are inverted before export.  The official evaluator
  receives raw 672-D motion and does not use these training statistics.
  MMM also contains `Interx_Mean_root/Std_root` (672-D float64) and
  `Mean/Std` (336-D float32); neither pair is used by this HHC run.  The
  complete source-link, shape, dtype and hash contract is machine checked in
  `output/interx_protocol_audit.json`.
- Official test split: 1708 IDs; 1706 pass the official length/existence
  filter; batch 32 with `drop_last=True` evaluates 1696 samples per replication.
- Motion protocol: 150 frames, 30 FPS, VQ unit length 4.
- Statistics: official Inter-X test GT and the HHC loader were compared on
  short, medium and clipped-long samples. Names and lengths matched exactly;
  after HHC de-normalization the maximum absolute motion difference was
  `1.1920929e-07` (allclose at `1e-5`).
- Evaluator: the same official `text_mot_match/model/finest.tar` checkpoint was
  executed through both wrappers on a 32-sample batch. Text and motion
  embeddings were bit-identical (maximum absolute difference `0.0`).
- `body_models/smplx` and `checkpoints/hhi` resolve directly to MMM.  The audit
  also pins both gender-specific SMPL-X NPZ files and the male/female
  VolumetricSMPL checkpoints by SHA-256, so physical evaluation cannot silently
  switch body or SDF weights.
- The official evaluator consumes denormalized/raw 672-D motion.  Frozen GT
  embeddings were regenerated from those raw motions and match the official
  wrapper; method exporters therefore must not apply Inter-X normalization a
  second time before scoring.
- Official repeated protocol: 20 replications, Diversity 300 pairs, and
  MModality on 100 prompts × 30 generations with 10 embedding pairs.
- All 20 request files pass exact schema/order checks and are frozen in
  `output/fair_interx_requests/manifest.json`; replication 0 has SHA-256
  `01d894c2e95183b2aaa510470819d64cd6c5862beb47b114320e91270d0100d7`.
- R-Precision follows the official evaluator: retrieval is performed within
  each complete 32-sample batch and counts are summed over 1,696 samples.  It
  is not a 1,696-way global retrieval problem.
- The paper table uses only the official full-motion embedding metrics. No
  hand-neutralized auxiliary FID is computed or used as a gate.
- The requested common fair-table budget is 50 iterative sampling steps for
  both HHC and InterMask.  Separate audits retain the HHC Fig.-2 / historical
  parameter context, the recovered original-development runtime's 10 outer
  steps (30 Root/Body/Hands channel updates), the later pre-cleanup Git
  snapshot's `sample_intra_cas` default of 3 outer steps (9 channel updates),
  and the released InterMask default of 20 steps; these protocols are never pooled.
  HHC temperature 0.5, text CFG 3.5 and contact logit weight 1.0 come from
  Sec. 6.1.  The snapshot's cascade default temperature 1.0 is recorded only
  in its source-parameter audit, because it conflicts with the paper's explicit 0.5.
  The 3-step sensitivity keeps the paper's fixed-high-confidence-token semantics: MMM's
  development implementation overwrites previously fixed tokens and decays
  contact guidance, so its 3-step row is explicitly a snapshot-parameter
  sensitivity rather than an exact historical-runtime claim.
- Historical MMM behavior is reconstructed from the repository *before* the
  large cleanup.  The frozen baseline is commit
  `75814e57e0702789176da6713056a15ed4a192e2`; the cleanup starts at
  `0f58ffb`, whose verified direct parent is that baseline.  The audit stores
  the baseline/cleanup tree IDs plus Git-blob and SHA-256 identities for
  `models/t2i_trans.py`, `utils/eval_trans.py`, `train_t2i_trans.py`,
  `generate_mm_samples.py` and `options/option_transformer.py`, so later
  cleaned files cannot silently replace the historical call chain.  Because
  the February--March training runs used additional
  uncommitted development code, contemporaneous W&B runtime tracebacks take
  precedence over that later Git snapshot.  Those logs explicitly record
  `type="sample_intra_cas"` (for example runs dated 2026-02-06, 2026-02-08,
  2026-02-13, 2026-02-17 and 2026-03-05).  The cleaned `main` branch's
  remaining `type="sample"` validation call must therefore never be used to
  infer the historical production sampler.  The machine-readable audit
  freezes two representative traceback files by SHA-256 and records the exact
  call sites at lines 107 and 60.  It enforces the evidence order
  `historical_runtime_logs -> local_history_time_bracketed_source ->
  pre_cleanup_git_tree -> cleaned_main`; every
  pre-cleanup Git file is read with `git show`, never reconstructed from the
  cleaned working tree.  The paired W&B metadata files are also frozen: both
  report upstream MMM commit `8c3e7ad` while executing the local
  `train_t2i_trans.py`, confirming that the two-person runtime code was
  uncommitted at the time.  The matching JetBrains PyCharm 2023.2.8 Local
  History/VFS store was therefore recovered read-only from a copy and each
  source state was bracketed around the W&B `startedAt` timestamp.  AST and
  hash audit now proves that both representative runs loaded a
  `sample_intra_cas(max_steps=10)` implementation with nested
  Root→Body→Hands updates, while `utils/eval_trans.py` omitted an explicit
  `max_steps`; the exact historical budget was thus 10 outer / 30 channel
  updates.  The 2026-02-17 run used temperature 1.0 with CFG disabled; the
  2026-03-05 run used temperature 0.5 with CFG 3.5.  Evidence is frozen in
  `output/historical_mmm_sampler_local_history_audit.json`; each recovered
  source is additionally bound to its original VFS content ID, byte count and
  SHA-256 through `recovery_manifest.tsv`.  The 134 MB LocalHistory/VFS input
  snapshot is preserved under `output/historical_mmm_local_history_store_232`
  so future IDE activity cannot rewrite the audit source.  It is explicitly labelled Local
  History recovery, not Git history.  All new formal runs still
  use the user-specified 50-step protocol above.
- The common fair table uses each requested frame length exactly.  InterMask's
  released generator instead floors frame length to a multiple of four; a
  separately labelled author-native audit records the resulting effective
  length (for example 150→148) in `generated_lengths`.  The scorer uses that
  explicit length and rejects main/MM files with different policies.  Native
  and exact-length numbers are never pooled into one aggregate.
- The full 20-replication **author-sampler / exact-length** InterMask audit is
  complete.  Its HDF5 attributes bind `time_steps=20`, `cond_scale=2.0`,
  `temperature=1.0` and `length_policy=exact`; it is therefore a native
  20-step reference, **not** the requested common-50 result. The latter is a
  fresh, separately labelled `intermask_official_exact_common50` run and must
  never reuse these values. The native-20 exact-length audit has FID
  `0.39169 ± 0.01430`, Diversity
  `9.01711 ± 0.05896`, Matching `3.72855 ± 0.01924`, official 10-pair
  MModality `1.80842 ± 0.06971`, and R-Precision Top-1/2/3
  `0.40032/0.58824/0.69959`.  Its exhaustive 435-pair MModality is
  `1.88245 ± 0.04404`.
  The 40 actual main/MM HDF5 headers, shapes, completion flags and exact
  generated lengths are frozen in
  `output/fair_eval/intermask_official_exact/native20_hdf5_protocol_audit.json`.
- The paired author-native replication-0 audit changes exactly the 836
  capped-150 samples to 148 frames and leaves the other 860 unchanged.
  Native/exact FID is `0.42365/0.44002`, official MModality is
  `1.92074/1.89830`, and exhaustive
  MModality is `1.83786/1.89887`.
- Raw GT root translations are finite and in metre scale: xyz ranges are
  `[-3.42, 3.60]`, `[-0.84, 0.76]`, and `[-3.46, 4.03]`; translation-norm
  P50/P95/P99/max is `0.63/1.88/2.52/4.84 m`.  This check catches the earlier
  accidental use of velocity/root-feature statistics for raw translations.
- CoShMDM fair adapter: its native 262-D/person representation retains 21
  non-root body rotations exactly. Root orientation and translation are
  recovered with the frozen shape-specific all-joint rigid fit, and the Y-up
  frame is converted back to Inter-X coordinates. The complete 1,696-request,
  59-skeleton projection-idempotence audit has all-coordinate RMSE mean
  `1.28e-7` and maximum `1.38e-5`. Because CoShMDM does
  not generate fingers, all 33 hand rotations are identity on valid frames and
  padding remains zero. The official full-space score therefore exposes that
  limitation. Projected-GT, representation-controlled and hand-neutralized
  scoring are outside the experiment scope and are not part of this protocol.
- Fair CoShMDM training uses the author's FP32, AdamW, DDIM50, CFG 3.5 and 2000
  epoch configuration.  A 4090 cannot fit physical batch 64, so batch 32 with
  two explicit manual-optimization accumulation steps preserves effective
  batch 64.  The unmatched 569th micro-batch is discarded at each epoch
  boundary, exactly matching physical batch 64 with `drop_last=True`: 284
  updates/epoch and 568,000 updates over 2,000 epochs.  Inter-X statistics cover
  both person orderings and the author's
  left/right mirror augmentation.  A numerical guard assigns identity facing
  only to all-zero padded skeletons before the original relative-rotation loss;
  valid-frame loss values are unchanged.
  Because worker assignment occurs at micro-batch granularity, the stochastic
  caption/crop/person-order stream is distributionally rather than bitwise
  identical to a physical batch-64 run; the optimizer grouping and exposure
  counts are exact.  Runtime verification at the first replacement-run boundary
  observed exactly 284 epoch-0 updates and the dedicated dropped-half-batch
  scalar equal to 1.
- CoShMDM Lightning checkpoints retain model, AdamW, scheduler, counters,
  loops and callbacks.  The released two-worker dataset's caption/crop/person
  RNG streams are not serialized, so mid-epoch recovery is explicitly labelled
  distributionally rather than bitwise equivalent.  The active final run is
  therefore kept uninterrupted.
- The superseded CoShMDM run's milestone at global step 20,000 passed a complete
  finite-state audit for model and AdamW tensors.  Its one-request DDIM50 export
  also passed exact name/length/completion and embedded checkpoint/request-hash
  checks.  It remains only an exporter diagnostic: that run was later excluded
  because it carried an incomplete half-batch across epochs, producing 569,000
  rather than 568,000 updates.  It is neither a reported quality result nor a
  checkpoint selected on test data; the replacement restarted from update 0.
- The epoch-boundary-correct predecessor independently reached global step
  20,000 at epoch 70 and passed the same state audit: all 346 model and 588
  AdamW floating tensors are finite, with 196 optimizer parameter-state entries
  and scheduler `last_epoch=70`.  Its checkpoint SHA-256 is
  `bfb4a20f9986a678811289424ff41bc8b779cf679b66c2871cc0d57cefb28e44`;
  `output/coshmdm_corrected_20k_checkpoint_audit.json` binds that predecessor's
  complete file. A later resume of the same run silently switched from
  `interx_global` to the released InterHuman statistics, so the entire run is
  now a superseded diagnostic and this 20k checkpoint is not evidence for the
  active `CoShMDM-InterX-fair-statsfix-seed123-bs32x2` run. The active run must
  produce and pass its own 20k and terminal 568k audits before reporting.
- The CoShMDM normalization was independently recomputed from all 9,110 names
  in the official train split (12,938,776 augmented frames; mirror and both
  person orders).  Recomputed mean and std are bitwise equal to the files used
  by the active process: SHA-256 `aa281b878b8a71c6d10c001059b29374921c18545c082102bf69a404982f568d`
  and `30a83e92965a465301a3642eb55b2ba705d3dceaba05e45846aa433d3900b104`.
  The machine-readable recomputation manifest is included as an artifact in
  every later CoShMDM fair-evaluation run manifest.
- The stats-fixed runtime's epoch-boundary behavior is verified from the actual
  TensorBoard scalar series rather than copied from its static launch manifest.
  `tools/audit_coshmdm_statsfix_startup.py` accepts one or more event files,
  requires contiguous epoch indices from zero, and requires every recorded
  `dropped_incomplete_accumulation_micro_batches` value to equal one. The final
  report reruns this audit after training and requires exactly 2,000 completed
  epoch records (`0..1999`), binding the current runtime manifest, complete
  training log and all event-file hashes.

Run the fail-fast source/data audit with:

```bash
conda run -n New_MMM python tools/verify_interx_protocol.py
```

The expected output ends with `"status": "pass"` and
`"evaluated_entries": 1696`.

Freeze evaluator requests (including GT/text embeddings and the shared MM
subset) before evaluating any method.  The same files also freeze the aligned
shape, gender and cropped contact schedule required by HHC sampling:

```bash
conda run -n New_MMM python tools/prepare_fair_eval_requests.py \
  --output-dir output/fair_interx_requests --device cuda:0 \
  --replications 20 --seed 1234
```

Validate every field and freeze SHA-256 checksums with:

```bash
conda run -n New_MMM python tools/verify_fair_eval_requests.py
```

The version-2 request manifest also hashes the exact test split/HDF5, raw
normalization statistics, request builder, dataset loader, evaluator wrapper,
`Comp_v6_KLD01/opt.txt`, and the evaluator that is actually loaded by that
wrapper: `text_mot_match/model/finest.tar`.  This distinguishes the evaluator
from the unrelated conditional-generation checkpoint under `Comp_v6_KLD01/model`.

For a paired, LLM-free physical reference, the exact denormalized GT crop for
replication 0 is frozen in
`output/fair_interx_ground_truth/replication_00.h5`.  A deterministic rerun
reproduced the request SHA-256 byte-for-byte.  Score GT or any full method HDF5
with the same subject skeletons and contact schedule using:

```bash
conda run -n New_MMM python tools/score_fair_interx_physical.py \
  --requests output/fair_interx_requests/replication_00.npz \
  --generations METHOD_MAIN.h5 --output METHOD_PHYSICAL.json \
  --device cuda:0
```

The report includes a fixed 5/10/15 cm threshold sweep, contact precision,
recall, F1 and temporal IoU, false-contact rate, contact/non-contact distance,
and grounded-foot horizontal speed.  This is a joint-level contact protocol;
mesh penetration and CHI3D surface contact remain separately reported metrics.

For each replication, CoShMDM exports one main generation and 30 generations
for the frozen `mm_indices`; `tools/score_fair_interx_generations.py` then runs
the same official evaluator.  The request file, checkpoint and seed are stored
in every output manifest/HDF5 attribute.  The run manifest additionally records
all three repository revisions, dirty status and tracked-diff hashes, plus
SHA-256 records for the actual exporter, scorers, aggregator, protocol,
evaluator, frozen-request manifest and model artifacts.
Export resume is provenance-safe: it rejects a different checkpoint, request,
seed, batch/length policy or sampling configuration even when array shapes
match.  New HHC/InterMask/CoShMDM HDF5 files bind the frozen request SHA-256;
all three also bind their model checkpoint SHA-256 (and HHC/InterMask bind the
separate VQ checkpoint).  The common semantic and physical
scorers verify the recorded request hash against the supplied NPZ and reject
main/MM files whose model or sampler metadata differ.  The already-completed
InterMask 20-replication artifacts predate the request-hash attribute and are
explicitly reported as legacy-unverified at the HDF level; their immutable
request arrays remain covered by the v1/v2 checksum manifests.  Stale metric
JSON is recomputed whenever an input HDF5 is newer.
The final aggregator additionally rejects missing/duplicate replication IDs,
non-unique request contents, request-hash drift, generation files without an
embedded request hash, and any checkpoint/sampler identity change across the
20 rows.  It also re-hashes each complete main/MM HDF5 against the digest
written by the scorer, so post-score replacement or corruption is rejected.
Physical diagnostics and legacy non-metric artifacts cannot match its exact
`replication_NN.json` selector and therefore cannot be pooled into the official
confidence interval accidentally.
