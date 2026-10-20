# Paper reproduction and fair-comparison runbook

All commands use the `New_MMM` environment and are run from the indicated
repository root.

The claim-by-claim architecture, mask, training and sampler mapping is recorded
in `docs/paper_implementation_audit.md`; review-driven experiments and their
priority are tracked separately in `docs/review_experiment_plan.md`.

## Mandatory paper–repository–data gate for external baselines

No external baseline may be trained, generated, scored, or admitted to a table
from repository defaults alone.  Before a formal run, freeze and cross-check:

1. the latest paper and supplementary protocol (task conditioning, data split,
   representation, filtering, sampling, evaluator, repetitions and CI);
2. the official repository revision, released configuration, model/evaluator
   weights and the exact runtime call chain; and
3. the local split, motion archive, text features, normalization statistics and
   body-model assets consumed by that call chain.

Paper/release disagreements create two explicitly named results rather than an
implicit compromise: an untouched **author-release calibration**, and a
**cross-method fair protocol** using the same frozen requests and evaluator as
every method.  Conditional inputs such as GT poses may never enter the latter.
If an upstream dataset does not publish file checksums, report exactly which
lineage properties are proven locally and retain the missing upstream checksum
as a disclosed limitation.

This is a fail-closed rule, not a recommendation.  Its machine-readable source
is `configs/external_baseline_reproduction_policy.json`; the aggregate gate is
rebuilt by `tools/build_external_baseline_reproduction_audit.py`.  A missing or
failed item blocks training, generation, evaluation and table admission.  In
particular, reading an official repository is not a substitute for reading the
paper/supplement, and matching split filenames is not a substitute for checking
split hashes, archive keys/shapes/dtypes, filtering, representation, train-only
normalization statistics, text assets and body-model provenance.

Interact2AR is currently gated by
`output/interact2ar_paper_repo_data_audit.json`, reproducible with:

```bash
conda run -n New_MMM python tools/audit_interact2ar_paper_repo_data.py
```

Its paper specifies 20 evaluation repetitions, while the released `--full`
script uses five.  The README evaluation command also supplies `--first_frame`;
the released loader implements this as a random GT motion frame from indices
0--4.  Therefore the native five-repeat conditional calibration, the
paper-aligned 20-repeat result, and the 20-repeat unconditional fair supplement
are separate protocols and must not be pooled.

## Mandatory online training observability

Every newly launched training experiment must use online W&B from its first
optimizer update.  A stable run ID is written into the training manifest and
must be reused by an in-place resume.  At least every 50 optimizer updates the
run records the global FP32 L2 norm and maximum absolute value of all trainable
weights and of their **unclipped** gradients, measured after backward (and AMP
unscale, when applicable) but before clipping and the optimizer update.  It
also records non-finite gradient/weight element counts, learning rate,
throughput and peak GPU memory.  Any non-finite parameter or gradient fails the
training closed.  When gradient clipping is enabled, the same telemetry event
also records the post-clip global L2 norm, configured clip threshold and a
clip-trigger flag; the completed-run audit reports the resulting clip-hit
rate.  Every new run also records the same norm/max/non-finite diagnostics,
gradient-to-weight L2 ratio and tensor-value counts for each trainable
top-level child under `train/modules/<name>/`; the all-parameter global values
remain the pass/fail authority.  Runs already in progress when this conditional rule was introduced are
not restarted or retroactively relabelled.

The same run records total/component losses and task-meaningful accuracy or
mask-ratio metrics during training.  Official validation records validation
loss, FID, R-Precision top 1/2/3, Matching and Diversity; classification
accuracy is required only for discrete-token models and is never fabricated
for diffusion regression.  A formal checkpoint cannot enter an evaluation
table until the remote W&B history audit confirms that every metric required
by its profile is present, finite, and continues through the declared final
optimizer step; in particular, a complete loss stream cannot mask truncated
gradient/weight telemetry.  Both non-finite counters must remain zero over the
entire run.  Training coverage is measured only on an explicit optimizer axis
(`train/optimizer_step` or Lightning's `trainer/global_step`).  Because some
frameworks attach a microbatch counter to epoch-level validation, validation
coverage instead requires every mandatory validation stream to have a finite
event at or after the final training event.  W&B's internal `_step` is used
only for that event ordering and is never treated as optimizer progress.
The machine-readable rule is
`configs/online_wandb_training_policy.json`; historical runs are not relabelled
as compliant retroactively.

After the current historical-configuration-plus-paper-mask HHC and TiMotion
rows both complete, run `tools/finalize_current_interx_paired_results.sh`
directly.  It contains no service, polling loop, sleep, or background waiter;
it refuses incomplete summaries and constructs the full paired Holm family
against InterMask, Interact2AR, CoShMDM and TiMotion, plus the separate oracle
timing ablation.  Its output names explicitly contain
`historical_paper_mask`, so they cannot be confused with the archived
all-paper-configuration HHC comparison.

The validation-only HHC sampler calibration uses a separate bridge request at
`output/hhc_mgt_validation_bridge_requests/validation.npz`.  The shared
cross-method validation request stores samples in evaluator motion-length
order, whereas HHC's in-training hook consumes CUDA randomness in the
DataLoader's collated order.  `tools/prepare_hhc_validation_bridge_requests.py`
restores that exact batch/within-batch sampling order without changing the 544
sample set, captions, crops, shapes, contacts, or evaluator embeddings.  In the
audited request only 26/544 positions were already fixed.  The corrected
formal replay is `0.63639148` versus the checkpoint record `0.63642776`
(absolute difference `3.63e-5`); the earlier evaluator-sorted replay was
`0.64752969`.  Because the checkpoint selection is recorded to four decimals
and the standalone path re-embeds HDF5 motions, the declared numerical bridge
tolerance is `1e-4`; no bitwise-equivalence claim is made.

The original shared 544-sample request was first frozen under the CoShMDM-only
purpose at `output/coshmdm_interx_val_requests/`.  Completed CoShMDM evidence
continues to bind that immutable manifest.  HHC and later method selection use
the byte-identical method-neutral view at
`output/interx_official_val_requests/`; its manifest validates and records the
original generator, split, motions, evaluator assets and source manifest, and
explicitly forbids use as a reported test row.  The HHC finalizer regenerates
this view with `tools/promote_interx_validation_request_contract.py` before
selection, so a stale source or changed byte stream fails closed.

Before training or scoring, bind the active MMM data/evaluator/body assets and
the common/native sampler budgets to their expected shapes and SHA-256 values:

```bash
python tools/audit_historical_mmm_local_history.py

conda run --no-capture-output -n New_MMM python -u \
  tools/verify_interx_protocol.py \
  --output output/interx_protocol_audit.json
```

The final report requires this artifact to have `status=pass`; it verifies the
9,110/570/1,708 `[T,56,6]` motion HDF5 files, all six potentially-confusable
normalization arrays, official evaluator options/weights, SMPL-X models,
VolumetricSMPL weights, and five deliberately separate sampler records: the
user-fixed 50-step common protocol; the recovered original-development runtime
at 10 outer steps / 30 Root→Body→Hands channel updates; the later pre-cleanup
Git snapshot's 3-step `sample_intra_cas` function default; the paper Fig.-2
10-step parameter audit; and InterMask's native 20-step reference.  None of
the historical or sensitivity records overrides the current 50-step fair protocol.

The optimization core used for shape augmentation is the unmodified original
MMM implementation, not the earlier reconstructed "paper optimizer".  Its
actual defaults are Adam, 200 iterations and LR 0.01; at iteration 100 it tries
to freeze one-to-one anchors that pass the original 5 cm and opposing-normal
filters.  It retains the original 0.2/0.5 m penetration thresholds and CUDA
Chamfer nearest-vertex signed distance.  `optimization/losses.py` is byte-equal
to `/home/wenning/projects/MMM/optimization/losses.py` (SHA-256
`29f67e9830c0a0c9dbfffc8361ce4ae91dc76fef691cca17cff39613d7151f7c`),
and the `optimize_full_body()` AST is equal to the original development file.
Run `python tests/test_original_mmm_optimizer_restoration.py` to verify this.

The former 100/0.02/freeze-at-50, VolumetricSMPL-backed and altered-reduction
implementation is retained only in files ending in `experimental_invalid` for
root-cause evidence.  No production or training entry point may import it, and
all outputs made by that path are excluded.  VolumetricSMPL is now used only as
the common read-only mesh/physical evaluator across methods.

The Inter-X random-shape wrapper is necessarily additional to MMM because the
original repository did not ship that dataset transformation.  It replaces
betas, supplies the frozen LLM regions and manual timing, temporarily applies
MMM's per-person joint-min ground alignment, and records/reverses coordinate
transforms.  These wrapper operations must be reported separately from the
unchanged numerical optimizer and cannot be described as part of original MMM.

The legacy `dataset/interx/contact_pairs` directory is an audit input only.  It
contains all 570 validation IDs but no raw/model/prompt/parse provenance and
fails the strict schema for 345 files.  Never overwrite it or use it for final
optimization.  A fresh, versioned extraction for the formal official-test
comparison uses the ChatGPT-authenticated Codex CLI and must consume the exact frozen
request captions; using the first text-file caption is not equivalent because
Inter-X selects a caption stochastically.  Use:

```bash
conda run --no-capture-output -n New_MMM python -u \
  eval/extract_interx_contacts_codex_frozen_requests.py \
  --split-file dataset/interx/splits/test.txt \
  --requests output/fair_interx_requests/replication_00.npz \
  --out-dir dataset/interx/contact_pairs_codex_luna_strict_v3_test \
  --model gpt-5.6-luna --reasoning-effort low \
  --batch-size 32 --max-retries 2
```

No API key is used or recorded; the call consumes the existing ChatGPT/Codex
subscription allowance.  Each successful pickle has a JSON sidecar bound to
the exact frozen caption, request index/hash, requested/returned model,
reasoning effort, prompt-template hash, content-addressed binding writer, raw
batch response, session, retries, token metadata and wall time for every CLI
attempt.  Invalid schema fails closed and does not write an
optimization cache.  The directory manifest freezes the split, complete
request NPZ, selected names/captions and implementation hashes; resume
validates the full contract before skipping any record.  There are 1,696
frozen requests (not all 1,708 test IDs).

After the 9,110-item train job completes, the single sequential launcher
`tools/run_remaining_interx_codex_annotations.sh` produces the formal frozen
1,696-caption test cache first.  It then freezes a disjoint 128-caption
robustness subset (32 each for timing-negative, laterality, multiple-contact
cues and role references), runs `gpt-5.6-terra`/low as a stronger-model
consistency reference plus the deterministic rule parser, and finally creates
auxiliary first-caption caches for all 570 validation and all 1,708 test
sequences.  Manual timing evaluates only binary contact presence; the runbook
does not treat Terra as ground truth or claim human 75-region F1.  This gives
full split coverage while keeping the formal evaluator captions separate and
unambiguous.

Validate both the directory-level request contract and every per-sample
caption/index/hash before optimization:

```bash
conda run --no-capture-output -n New_MMM python -u \
  tools/audit_interx_codex_contact_cache.py \
  --pairs-dir dataset/interx/contact_pairs_codex_luna_strict_v3_test \
  --split dataset/interx/splits/test.txt \
  --requests output/fair_interx_requests/replication_00.npz \
  --output output/interx_luna_test_frozen_cache_audit.json
```

`tools/verify_fair_eval_requests.py` is non-mutating when the manifest already
exists.  It verifies all 20 NPZ hashes against the original 7,836-byte manifest
(`d4007b...`) and resolves later source drift only through exact
content-addressed snapshots.  The evaluator loader was subsequently extended
for train-only augmentation assets, so its original 19,551-byte request-time
source (`c3993a...`) and the original verifier (`6b357b...`) are preserved in
`tools/provenance_snapshots/`.  Later sampling-history and ordinary-only metric
edits also changed the live protocol YAML; its request-time 3,721-byte version
(`1db8e...`) was recovered by reversing the recorded rollout patches and is
preserved in the same content-addressed directory.  Its byte count and SHA-256
exactly match the frozen manifest.  The reconstruction is hash-gated and audited by
`output/fair_interx_requests/provenance_recovery.json`; it modifies zero request
NPZ files.  Refreshing provenance requires the explicit
`--refresh-provenance` flag and is not part of the formal queue.

The same recovery now binds the exact successful request-creation execution
record.  The command was
`OMP_NUM_THREADS=8 conda run --no-capture-output -n New_MMM python -u
tools/prepare_fair_eval_requests.py --output-dir output/fair_interx_requests
--device cpu --replications 20 --seed 1234`.  An earlier identical attempt was
interrupted after replication 16 and exited 1; it is not used as provenance.
The selected second launch and its `CommandExecution` completion report all 20
replications with 1,696 samples and exit 0.  Their exact two JSONL records are
preserved as the 16,063-byte `e2ae556f...` content-addressed snapshot.  Thus the
creation device is recovered as CPU from immutable external execution evidence,
not inferred from the later replay tolerance.

The final queue also replays replication 0 from the official loader and
evaluator on CPU with seed 1234.  Names, captions, lengths, shape, gender,
contact timing, MM indices and seed must be bitwise equal to the frozen NPZ.
Although the request NPZ schema itself does not embed its evaluator device, the
replay audit now directly binds the recovered successful CPU creation record.
The three FP32 embedding arrays are compared with a strict `1e-6` absolute tolerance;
the observed maximum is `9.5367431640625e-7`.  The frozen and replayed files,
every per-field comparison and the audit implementation are bound in
`output/fair_interx_request_replay_audit.json`.

For the train-only optimized random-shape augmentation, the active contact
region cache uses the ChatGPT-authenticated Codex CLI instead of the invalid
hard-coded MMM Gemini key.  Luna is run with low reasoning and a closed JSON
Schema whose body-part fields enumerate the paper region dictionary:

```bash
conda run --no-capture-output -n New_MMM python -u \
  eval/extract_interx_contacts_codex.py \
  --split-file dataset/interx/splits/train.txt \
  --out-dir dataset/interx/contact_pairs_codex_luna_strict_v3_train \
  --model gpt-5.6-luna --reasoning-effort low \
  --batch-size 32 --max-retries 2
```

This path does not read an API key.  It stores raw batch responses, Codex
session IDs, token counts, every failed attempt, per-sample strict pickles and
metadata, plus split/caption/prompt/schema/implementation hashes.  The formal
shape optimizer refuses an incomplete or range-limited train cache.
The 9,110-row train extraction was already active before the timed frozen-
request wrapper was introduced: all 294 successful/failed attempts have token
metadata, but none has a recorded wall-time boundary.  Its audit therefore
reports wall time as unavailable rather than reconstructing it.  The formal
1,696-caption test extraction and the Luna/Terra robustness calls use the timed
wrapper and include every successful and failed attempt in mean/median/P95
wall-time and token accounting.

The first-caption auxiliary extraction is complete for all official splits:
9,110 train, 570 validation and 1,708 test sequences (11,388 total).  Their
cache audits contain 8,318, 471 and 1,543 structured contact rows respectively,
with no missing or invalid sample.  The 1,708-row test cache is intentionally
separate from the 1,696-caption frozen-request cache used by the formal
optimizer.  Final report construction requires all three full-split cache and
timing audits, the frozen-test audit and the Terra-128 audit; a count or hash
drift is fatal.

The completed 6,238-row augmentation preflight additionally binds the frozen
plan, Luna manifest and MMM train-motion hashes. Exactly one source caption
has an otherwise identical trailing space before the processed-text `#`
delimiter. The augmentation loader permits only this class of leading/trailing
whitespace normalization, records both UTF-8 hashes and the match mode, and
still rejects any internal/content difference. The audited preflight artifact
is `output/interx_shape_aug_preflight_seed123_v2/_preflight_manifest.json`.

Audit the completed prefix during extraction, and rerun without
`--allow-incomplete` after all 9,110 samples finish:

```bash
conda run --no-capture-output -n New_MMM python -u \
  tools/audit_interx_contact_timing_consistency.py \
  --allow-incomplete \
  --output output/interx_luna_train_timing_consistency_partial.json

conda run --no-capture-output -n New_MMM python -u \
  tools/audit_interx_codex_contact_cache.py \
  --pairs-dir dataset/interx/contact_pairs_codex_luna_strict_v3_train \
  --split dataset/interx/splits/train.txt \
  --output output/interx_luna_train_cache_audit.json
```

Do not treat an action class with no timing files as a no-contact class.  The
missing classes include explicit grasp/push/pat interactions.  The immutable
v2 plan therefore uses only train samples with an actual manual binary timing
file, while Luna still annotates the full train split:

The full-cache audit also records the extractor manifest's end-to-end wall
clock span.  This is an elapsed annotation-run total including retries,
resume/pause and orchestration overhead, not a fabricated per-request latency;
per-attempt latency remains N/A for old sidecars that predate explicit timers.
Every cache audit additionally binds and re-hashes its own audit implementation,
not only the extractor/parser sources recorded by the cache manifest.  The
three full-split manual-timing/LLM-region fusion audits bind their own audit
implementation in the same way.
The final report exposes both audited disagreement directions.  Positive
manual timing with no caption-derived region uses physical losses only; a
caption-derived region under explicit zero timing is inactive.  Neither case
is silently overwritten to manufacture modal agreement.

```bash
conda run --no-capture-output -n New_MMM python -u \
  tools/prepare_interx_shape_augmentation_plan.py

conda run --no-capture-output -n New_MMM python -u \
  tools/audit_interx_shape_augmentation_plan.py \
  --output output/interx_shape_augmentation_plan_v2_audit.json
```

The frozen v2 plan contains 6,238 eligible timing-covered samples: 5,868 have
positive timing and 370 are explicitly annotated all-zero.  It excludes 2,857
otherwise eligible samples whose timing is unannotated and 15 samples outside
the 30--999-frame tokenizer range.  The old `seed123` v1 plan is investigation
backup only and must not be optimized or used for training.
The audit re-hashes the official train split/motion HDF5, every caption,
manual timing file, source/donor beta file and the frozen 118-shape pool; it
also verifies crop bounds, timing counts, donor identity/distance and the exact
6,238/2,857/15 partition before either optimization arm can start.

Once the full Luna cache passes its formal audit, run a matched pilot on a free
GPU.  Both arms use the same v2 rows and shape assignments:

The launcher fixes `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`. This is
an allocator-only setting: it does not change the model, stochastic draws or
Eq. 6 values. It is required because a high-collision pilot sample retained
about 5 GiB as non-contiguous CUDA cache and otherwise failed a 4.05-GiB
backward allocation even though the same `query_chunk=10475` numerical
contract remained valid.

```bash
conda run --no-capture-output -n New_MMM python -u \
  tools/run_interx_shape_augmentation.py \
  --plan configs/interx_shape_augmentation/timing_covered_seed123_v2/plan_manifest.json \
  --contact-dir dataset/interx/contact_pairs_codex_luna_strict_v3_train \
  --output-dir output/interx_shape_aug_pilot_random_seed20260828 \
  --mode random_shape_only --selection-seed 20260828 \
  --positive-timing-only --limit 25 \
  --device cuda:1 --query-chunk 512 --seed 1234

conda run --no-capture-output -n New_MMM python -u \
  tools/run_interx_shape_augmentation.py \
  --plan configs/interx_shape_augmentation/timing_covered_seed123_v2/plan_manifest.json \
  --contact-dir dataset/interx/contact_pairs_codex_luna_strict_v3_train \
  --output-dir output/interx_shape_aug_pilot_optimized_flathand_query10475_seed20260828 \
  --mode optimized --selection-seed 20260828 \
  --positive-timing-only --limit 25 \
  --device cuda:1 --query-chunk 10475 --seed 1234
```

After the resumable optimized runner exits, finish runtime analysis, both
flat-hand physical scoring arms, the paired 10,000-bootstrap comparison and
the preregistered authorization gate with one foreground command (it creates
no waiter terminal):

```bash
bash tools/finalize_interx_shape_pilot_flathand.sh
```

The random-shape-only pilot is complete for all 25 requested samples in
`output/interx_shape_aug_pilot_random_seed20260828`.  Its completion manifest
binds the frozen v2 plan/contact-cache/source-motion hashes, the exact selected
names and donor assignments, records `iterations=0`, and confirms that the
VolumetricSMPL backend was not instantiated. Later flat-hand fixes changed
optimizer-source hashes which that early generic manifest recorded but never
consumed. Random-only resumption therefore uses
`tools/audit_random_shape_augmentation_completion.py` instead of pretending
the historical generic contract is current. The audit reconstructs every
output from the frozen HDF5 crop and donor beta/gender and requires bitwise
equality, sidecar/output hashes, `iterations=0`, and no loaded VolumetricSMPL
checkpoint; its artifact is
`output/interx_shape_aug_pilot_random_seed20260828/completion_content_audit.json`.
The corrected optimized pilot reused this exact selected-name set and is now
complete at 25/25; neither full 6,238-row arm may be materialized until a
paired physical gate passes.

Before the optimized pilot completed, the full-run authorization rule was
frozen in `configs/interx_shape_augmentation/pilot_physical_gate_v1.json`.
All 25 paired keys must be present; the mean deltas for penetrating-vertex
fraction and foot sliding must be non-positive, and the paper Eq.-6 squared
penetration mean plus the upper bound of its 10,000-bootstrap 95% interval must
be non-positive. `tools/assess_shape_augmentation_pilot.py` exits nonzero on
any failure, and the launcher places that assessment before its disk gate and
before either full asset command. No generator/test metric participates in
this decision.

The completed released optimizer pilot passes both penetration requirements
but fails the unchanged foot-sliding requirement. Relative to random shape,
paper Eq.-6 squared sum changes by `-0.39236324 m^2/frame` with bootstrap 95%
CI `[-0.73780645,-0.14126754]`, penetrating-vertex fraction changes by
`-0.01228992`, but foot sliding changes by `+0.01250786 m/frame` with 95% CI
`[+0.01018552,+0.01502604]`. Therefore
`output/interx_shape_aug_pilot_flathand_query10475_assessment_seed20260828.json`
has `full_augmentation_authorized=false` and the released result is retained
only as a failed, hash-bound reference.

That result was later superseded for augmentation authorization by a proven
coordinate-setting error, not by a hyperparameter change. The original MMM
development entrypoint and pre-cleanup commit `75814e5` floor-align each person
by its sequence-minimum joint y before optimizing. Earlier shape pilots fed
pelvis-centred raw Inter-X translations directly to a `y=0` ground loss. Their
outputs remain archived as diagnostics but cannot authorize training assets.
The active runner now optimizes in temporary per-person floor-aligned
coordinates and restores each offset before writing raw-coordinate NPZ/HDF5;
see `output/interx_shape_optimization_ground_contract_audit.json`.

After correcting that setting, a matched smoke still attributed foot
regression to changing root orientation and the pelvis-to-foot chains. The
hash-preregistered `grounded_chain_preservation_pilot_v1` is explicitly a
local-refinement stabilization variant, not the released all-variable
optimizer. Its strict fixed-root fallback preserves root translation,
orientation and both leg chains bitwise while optimizing the upper-body
contact variables. On the first frozen sample, foot sliding is exactly
unchanged (`0.0028626729249322147 m/frame`) and Eq.-6 penetration falls from
`0.0017800781080344072` to `0.00003767293237615377 m^2/frame`. It must pass
the frozen 5-pair and then unchanged 25-pair gates before full augmentation.
The assessment represents those authorization scopes explicitly: a passing
five-pair result sets `pilot25_authorized=true` while keeping
`full_augmentation_authorized=false`; only the unchanged 25-pair gate may set
the latter true. This semantic fix did not edit the already preregistered
five-pair gate: its SHA-256 remains
`0574349e48244d75d60edef2251291365a9aef0c1f7aa49a2102f814ea4f2e06`.

The frozen five-pair gate has now completed and passed.  Against the same
random-shape inputs, mean paper Eq.-6 penetration changes from
`0.05619538655` to `0.01181453976 m^2/frame`, penetrating-vertex fraction
from `0.00655662224` to `0.00132424550`, and foot sliding from
`0.00376482564` to `0.00375996063 m/frame`.  Every one of the five samples
improves both penetration measures and no sample degrades foot sliding.  The
machine-readable decision is
`output/interx_shape_aug_groundfix_groundedchain_fixedroot_pilot5_assessment_seed20260828.json`:
it authorizes the preregistered 25-pair pilot only and still records
`full_augmentation_authorized=false`.  The unchanged 25-pair run subsequently
completed 25/25 and passed every source/input/invariant check.  Its first
physical assessment was deliberately retained as a failure: Eq. 6 and
penetrating-vertex fraction improved strongly, but the reported foot-sliding
mean increased by `2.7329700189e-6 m/frame`.  The diagnosis in
`output/interx_shape_aug_groundedchain_pilot25_foot_backend_mismatch_diagnostic.json`
proved that root and the complete lower chain were unchanged and that both HBP
and the first 55 VolumetricSMPL skeleton joints give an exact zero paired
delta.  The old scorer estimated the floor over all SMPL-X output joints,
including appended face/hand landmarks moved by contact optimization; this
changed the grounded mask without moving a foot.  The preserved invalid
assessment is
`output/interx_shape_aug_groundfix_groundedchain_fixedroot_pilot25_assessment_seed20260828.json`.

The corrected scorer now freezes the floor definition to the shared first 55
SMPL-X skeleton joints and records that policy in its scoring contract.  A
regression test moves every appended landmark while holding the skeleton fixed
and requires the foot metric to remain identical.  Fresh, separate output
directories reproduce an exact `0.0` foot-sliding delta for all 25 pairs while
retaining Eq. 6 delta `-0.3760066` (bootstrap 95% CI
`[-0.7233238,-0.1267883]`) and penetrating-vertex-fraction delta
`-0.01242975`.  The original gate JSON/hash was not modified; its corrected
assessment is
`output/interx_shape_aug_groundfix_groundedchain_fixedroot_pilot25_body55floor_assessment_seed20260828.json`
and authorizes full augmentation.  The completed runtime audit projects
`23.90` serial GPU-days for 6,238 assets (bootstrap mean 95% interval
`18.46--29.54`).  The subsequent sharding audit found that the runner derived
the VolumetricSMPL seed from a local post-`offset` enumeration, so the same
sample changed trajectory when moved to another process.  The fixed policy is
`base_seed + 100 * full_selection_position`, where the full deterministic
position is frozen before `offset/limit`; both the position and actual seed are
now hash-bound in every sidecar and run contract.  Two real samples launched
as separate concurrent shards reproduce the old serial pilot NPZ SHA-256 and
all 100 total-loss values exactly.  That short benchmark is retained at
`output/interx_shape_aug_shard_seed_concurrency2_batch8_audit.json`, but it did
not authorize the final topology: on the real 6,238-row run, two concurrent
processes reached a 150-frame sample and raised `torch.OutOfMemoryError` after
three complete atomic samples.  The full-corpus observation supersedes the
short benchmark.  Formal production therefore uses one process per GPU while
retaining `frame_batch=8`, `query_chunk=10475`, and exactly the same per-sample
seed.  The extreme completed sample still improved Eq. 6 from `1.02533` to
`0.40748`, penetrating-vertex fraction from `0.04197` to `0.02038`, and depth
sum from `23.12` to `10.26`, with foot sliding bitwise unchanged.  The
superseding topology/physical audit is
`output/interx_shape_aug_full_single_process_topology_audit.json`.
The formal corpus is partitioned into eight immutable contiguous position
shards (780 rows for shards 0--6 and 778 for shard 7).  Completion is not
inferred from file counts: `tools/aggregate_interx_shape_augmentation_shards.py`
re-hashes both person NPZs for every row, checks the sidecar position, requires
identical plan/contact/optimizer/SMPL/implementation contracts across shards,
and accepts only the exact gap-free position set `0..6237`.  The optimized
overlay records all eight original completion-manifest paths and hashes; the
final result builder compares that list to the aggregate manifest exactly.
`tools/run_shape_optimized_mgt_when_gpu_free.sh` performs this aggregate and
multi-shard materialization before cache construction/training, and fails
immediately instead of creating a GPU waiter when the requested GPU is busy.

An explicitly incomplete eight-pair diagnostic exposed the failure mechanism
before that result was known, without changing the frozen gate or the running
implementation. On the optimizer's exact initial
stationary-and-grounded predicate, 7,096 foot observations have mean horizontal
displacement `0.00348254 m/frame` before optimization and `0.01640763 m/frame`
after it. Height retention is `99.9718%`, but only `78.4667%` remain below the
dynamic velocity threshold. Thus the affected feet generally do not leave the
ground; they move fast enough to escape a mask that the released loss
recomputes from the current motion every iteration. The hash-bound diagnostic
is `output/interx_shape_aug_pilot_partial_foot_contact_drift_diagnostic.json`.
It is not a completed-pilot result. After the complete gate confirmed the
trend, a separately labelled single-factor variant froze the initial
contact mask; its loss weight, velocity definition, optimizer, samples and
physical gate must remain unchanged. This conditional variant was itself
hash-preregistered while the released pilot was only `11/25` complete; the
evidence is `output/interx_frozen_foot_pilot_preregistration_audit.json`.
Its active output is
`output/interx_shape_aug_pilot_optimized_flathand_frozenfoot_query10475_seed20260828`;
the per-sample atomic completion manifest is authoritative during execution.
An incomplete four-pair grounded-foot preview is diagnostic only: random,
released and frozen means are respectively `0.00699`, `0.01803` and
`0.01546 m/frame`. Frozen improves the released behavior but has not yet shown
that it meets the unchanged random-shape gate. Do not tune its weight or infer
authorization from this preview. Once all 25 pairs finish, run the source/input
single-factor audit, runtime analysis, physical scoring, 10,000-bootstrap
comparisons and original gate with:

```bash
bash tools/finalize_frozen_foot_pilot.sh
```

Before the frozen-mask arm completed, a second conditional single-factor
experiment was preregistered in
`configs/interx_shape_augmentation/normalized_contact_reduction_pilot_v1.json`.
It activates only if the complete frozen-mask arm still fails the unchanged
physical gate.  At preregistration, 10/25 frozen samples were complete; among
the eight contact-positive samples, the raw contact sum was a median `108x`
the corresponding active-frame/contact-pair mean and the first fine-stage
contact contribution was a median `747.56x` the weighted skate contribution.
This diagnosis is hash-frozen in
`output/interx_shape_aug_pilot_frozenfoot_partial10_loss_balance_audit.json`.
The candidate changes only the contact reduction to an active-contact mean;
it retains frozen initial foot masks and every sample, shape donor, timing,
weight, optimizer, stage boundary and geometry setting.  It is explicitly a
stabilization ablation, not a claim that the mean form is the literal printed
Eq. 4/Eq. 6.  `tools/audit_contact_reduction_single_factor.py` verifies that
contract, and its final results are preserved even if the original gate fails
by:

```bash
bash tools/finalize_normalized_contact_reduction_pilot.sh
```

The optimized arm constructs the frozen male/female VolumetricSMPL backend
once per serial process and reuses those exact read-only weights across
samples. Every sample still receives a fresh `DyadicInteractionLoss`, Adam
optimizer, sample-derived seed, role assignment and coarse/fine anchor state;
the backend clears its mutable body-code cache at every iteration. Device and
query-chunk mismatches fail closed, and each sidecar records that backend reuse
was active. The arm-level completion manifest additionally binds the male and
female checkpoint paths and SHA-256 values to the official protocol audit;
the random-only control must record that no backend was instantiated. This
changes initialization overhead only, not the 100-step paper optimization
contract or its outputs. `tools/build_shape_augmentation_results.py` and the
final review report both fail closed if the equivalence artifact, weights or any
of its five implementation-source hashes are stale.

A real Inter-X geometry audit then exposed a representation mismatch hidden by
the original development defaults: `smplx.create(flat_hand_mean=False)` adds
the MANO mean hand pose, while the data/contact `human_body_prior.BodyModel`
uses a zero flat-hand pose.  The maximum vertex discrepancies were 5.33 cm for
the female sample and 6.42 cm for the male sample.  The production backend now
pins `flat_hand_mean=True`; vertices, the first 55 joints, normals, and
gradients with respect to root/body/left-hand/right-hand pose, translation and
betas match BodyModel exactly in
`output/volumetric_smpl_interx_flat_hand_geometry_audit.json`.  The old
default-hand pilot stopped after 6/25 and is diagnostic-only.  Because this
source fix invalidated the earlier chunk evidence, query chunking was
re-audited on the corrected backend: chunk 10,475 has zero loss difference,
maximum relative gradient error 1.53e-6, and at least 7.09x speedup on two real
motions.  Its gate is
`output/volumetric_smpl_flat_hand_query_chunk_assessment.json`; the new clean
25-pair pilot uses this query chunk and freezes `frame_batch=8`.  A full
100-step same-seed check showed that changing the frame batch to 4 changes the
optimization trajectory (final total loss 1.59938 versus 1.51023 on the
diagnostic sample), even though a one-step micro-audit had looked equivalent.
Consequently frame batch is a formal method setting, not a free throughput
knob; full assets must retain 8.

The fusion rule is an intersection, not a forced equivalence: semantic contact
attraction is active only where manual timing is positive and Luna returned an
inter-human region pair.  Positive timing with no textual pair keeps the global
physical losses but does not hallucinate a body-part contact; regions on zero
timing are inactive.  The full 6,238-row preflight resolves these cases
explicitly: 5,666 rows have active timed semantic contact, 202 have positive
manual timing but no semantic pair and therefore use physical losses only,
and 137 have a returned pair that is inactive because the manual timing is
all-zero.  It also finds exactly one boundary-whitespace-only caption match;
all other selected captions match byte-for-byte.  Score the two matching NPZ directories with
`tools/score_volumetric_smpl_npz.py`, then compare their `metrics.json` files
with `tools/compare_paired_physical_metrics.py` before choosing the full-run
augmentation ratio.

Materialize each completed arm into a different immutable overlay.  The
materializer preserves the original raw Inter-X Mean/Std byte-for-byte; its
manifest records both original source paths/hashes and copied hashes, and the
materializer refuses a non-identical copy.  The training entrypoint re-hashes
the immutable copied statistics and also validates every combined split,
HDF5 link/dataset,
beta/contact/text file, optimization completion and hash before cache export:

```bash
conda run --no-capture-output -n New_MMM python -u \
  tools/materialize_interx_shape_augmentation.py \
  --optimization-dir output/interx_shape_aug_pilot_optimized_flathand_query10475_seed20260828 \
  --output-root dataset/interx_shape_aug_pilot_optimized_flathand_query10475_seed20260828

aug_root=dataset/interx_shape_aug_pilot_optimized_flathand_query10475_seed20260828
cache_root=checkpoints/vq_precleanup_raw12288_shapeaug_pilot_optimized_cache
conda run --no-capture-output -n New_MMM python -u \
  train/train_t2i_trans.py --cache-only --device cuda:1 \
  --vq-checkpoint checkpoints/vq_precleanup_raw12288/net_175000.pth \
  --nb-code 12288 --code-dim 256 --shape-condition-dim 8 --width 512 \
  --train-asset-root "$aug_root" \
  --augmentation-manifest "$aug_root/materialization_manifest.json" \
  --train-split-file "$aug_root/splits/train.txt" \
  --all-split-file "$aug_root/splits/all.txt" \
  --train-motion-file "$aug_root/motions/train.h5" \
  --all-motion-file "$aug_root/motions/all.h5" \
  --stats-dir "$aug_root" --stats-prefix Interx \
  --codebook-dir "$cache_root/codebook" --cache-root "$cache_root" \
  --seed 123 --wandb-mode disabled
```

VQ is fixed to the audited pre-cleanup raw-12288 checkpoint; it is not
retrained for augmentation.  Each MGT arm is retrained for the same 300k update
budget and selected/evaluated on the unchanged official validation/test
protocol.  `--cache-root` and `--codebook-dir` must be unique per arm so the
baseline, random-shape-only and optimized caches cannot overwrite one another.

The former repeat-0 HHC/GT "paper optimization" workflow is invalid historical
diagnostic material and must not be run or reported.  It mixed a train-only
shape-augmentation mechanism with test-generation FID and used the rejected
100/0.02/freeze-50/custom-loss contract.  Its driver now requires an explicit
invalid-diagnostic acknowledgement.  The active experiment optimizes only
random-shape train motions through `tools/run_interx_shape_augmentation.py`,
with the exact original MMM 200/0.01/freeze-100 core.  VolumetricSMPL remains a
read-only common mesh scorer; it is never substituted into the optimizer.

The scorer separates metres (`depth`) from square metres (paper Eq. 6), rather
than reproducing the legacy script's dimensionally incorrect `Penetration (m)`
label for a squared nearest-vertex sum.

The strict frozen-request test contact cache remains available only for contact
annotation audits.  No current P0 driver optimizes HHC or GT test motions.
Generated HHC and GT are scored in their unchanged coordinates; optimization
contribution is measured by the matched baseline/random-shape/optimized-shape
training arms instead.

For the final six-method semantic table, the frozen HDF5 generations are all
rescored on CPU after both training branches finish, matching the recovered
CPU command that created the frozen GT/text request embeddings. Each record
binds the requested/resolved device, GPU name, PyTorch/CUDA runtime, official
evaluator artifacts, request file, and generation hashes.  This prevents the
small CPU/CUDA or CUDA-device embedding roundoff from being mixed across
methods while leaving every method's generated motions unchanged.

## HHC paper model

The active formal run freezes the byte-identical pre-cleanup raw-12288 VQ at
175k and retrains only the corrected paper MGT.  This tokenizer was selected
after its ordinary-only official-request reconstruction audit (FID
`0.2944172273`, MPJPE
`0.1362613624 m`).  It preserves the paper's 96/96/64 latent dimensions and
shape-aware raw representation, but its 12,288-entry codebooks and 175k
checkpoint are an explicitly disclosed historical-mainline exception to the
paper's stated 8,096 entries and 150k VQ budget.  It must not be described as
an exact paper VQ retraining.  The FID artifact also binds and re-hashes the
official full-672D evaluator options/checkpoint/wrapper/option loader and the
main-only scorer; both the MGT repair gate and final report fail closed on any
provenance drift.

The main-only evaluator used for VQ reconstruction and the two formal
optimization rows records the same runtime fields.  The VQ repair evidence and
both HHC+optimization/GT+optimization rows are frozen to CPU; a mixed-device
optimization table is rejected.

The repaired-MGT common-50 bridge is held to the same rule.  Its historical
reference and repaired candidate must both report the exact CPU evaluator
runtime contract, those two runtime dictionaries must be identical, and the
bridge runtime must also be byte-for-byte equal to the VQ reconstruction
runtime stored by the repair gate.  A numerically plausible FID produced on a
different device or PyTorch runtime therefore cannot release formal HHC.

The earlier multi-change paper diagnostic used
`tools/run_formal_hhc_after_mgt_repair.sh`; it is retained only for historical
root-cause evidence and must not be launched as the current model. The current
training entrypoint is:

```bash
tools/run_hhc_historical_plus_paper_mask_ratio.sh
```

It preserves the proven historical 1024-d/16-head structure: a 9-block base
made of 7 global self-attention and 2 local cross-attention blocks, followed
by 9 output self-attention blocks. It also preserves the all-valid loss,
38-token window and MultiStepLR, changing only the training mask-ratio
distribution to the paper's kinematic rule. Final selection and
reported generation separately bind the user-fixed
`sample_intra_cas/50/0.5/3.5/1.0` contract. The old paper-architecture,
masked-only-loss and warm-up/cosine run remains a disclosed failed diagnostic,
not the formal HHC row.

The imported tokenizer contract and reconstruction reports are frozen in
`checkpoints/vq_precleanup_raw12288/training_manifest.json` and
`output/vq_historical_eval/`.  The active run selects the minimum official-val
FID checkpoint through `output/hhc_formal_raw12288_repaired_mgt_selection.json` and
then produces the two formal labels
`hhc_formal_raw12288_repaired_mgt_common50_{no_contact,contact}`.

If that formal MGT remains above the preregistered core-FID threshold, the
post-formal diagnostics retain the full 300k scheduler horizon.  The matched
38-position control returned validation FID `1.1610538960` at 10k versus
`0.9789292216` for the paper-50 reference at the same fixed common-50 event;
it therefore showed no initial token-window improvement.  The completed 50k
endpoint is `0.8790842295` versus `0.8862313628`, which fails both promotion
margins, even though the control's non-monotone validation-best at 30k is
`0.6275210977`.  The decision is consequently diagnostic-only and keeps
paper-50 for subsequent formal arms.  A newly added exploratory
`historical_all_valid` arm changes only the loss support, because the native
MMM trajectory at 10k had masked/visible accuracies `67.70%/81.67%`, whereas
the current masked-only arm had `73.38%/67.27%`.  This arm diagnoses whether
visible-token supervision explains the historical FID gap and is explicitly
forbidden from replacing the paper masked-only result.  The existing 50k
`paper_masked_global_token_mean` arm separately tests masked-loss reduction.
Its first fixed common-50 validation at update 10,000 is now complete: FID is
`0.75840902`, Matching is `4.02768421`, R-Precision Top-1/2/3 is
`0.363971/0.544118/0.654412`, and Diversity is `8.96459`. Against the
synchronous paper masked-only value `0.97892922`, this is an absolute/relative
FID improvement of `0.22052020`/`22.53%`; it is promising reduction evidence,
not a promotion decision before the preregistered 50k endpoint. The matched
semantic metrics are not uniformly better: Matching changes by `+0.16001511`
(worse), Top-1/2/3 by `-0.03492647/-0.03676474/-0.03860289`, and Diversity by
`+0.03717899`. The exact-step paired record is
`output/hhc_global_masked_loss_vs_formal50_validation_10k_live.json`. The 10k
checkpoint has 947 finite model tensors, 1,041 finite optimizer tensors and
scheduler `last_epoch=10000`. Online W&B `edmond_gla/NewInter/g6sqo3fd`
contains all nine required validation streams plus continuous 50-step
gradient/weight telemetry. Evidence is
`output/hhc_global_masked_loss_control_10k_checkpoint_audit.json` and
`output/hhc_global_masked_loss_control_mgt_wandb_warmup_audit.json`.
At exact trainer step 9,999, masked accuracy is bitwise equal to the
paper-masked-only reference (`73.75552368%`); visible/all-token accuracy is
only `+1.07527161/+0.40747833` percentage points and batch loss is actually
`+0.01165444` higher. Thus the FID change is consistent with the intended
within-batch masked-token weighting change rather than broadly better masked
classification. The trajectory evidence is
`output/hhc_global_masked_loss_vs_formal50_training_trajectory_10k_live.json`.
The synchronous 20k validation is non-monotone: FID regresses to
`1.96005368` while Matching improves to `3.81014061`, Top-1/2/3 to
`0.404412/0.608456/0.713235`, and Diversity to `8.944830`. The paper
masked-only reference at exactly the same trainer step has FID `1.21681869`,
so the global-mean candidate is worse by `0.74323499`; its Matching is also
worse by `0.08655691`, Top-1/2 are lower by `0.022059/0.027574`, and Top-3 is
equal. This is not evaluator mutation: the 20k pre/post-validation model,
optimizer, scheduler and hyperparameters are bitwise identical. Moreover,
the candidate's exact-step masked accuracy is `0.700043` percentage points
higher than the reference even though its batch loss is `0.04577023` higher.
Evidence is
`output/hhc_global_masked_loss_vs_formal50_validation_10k_20k_live.json`,
`output/hhc_global_masked_loss_vs_formal50_training_trajectory_10k_20k_live.json`
and `output/hhc_global_masked_loss_control_20k_pre_post_validation_pair_audit.json`.
The preregistered decision therefore remains open until the 50k endpoint;
the 10k checkpoint remains the validation-best candidate so far.
The fixed common-50 validation at 30k then recovers to FID `1.01232600`,
essentially matching the synchronous paper masked-only `1.01760387`
(candidate minus reference `-0.00527787`). This is not a uniform semantic
recovery: Matching is `3.80849147` versus `3.62438869`, R-Precision
Top-1/2/3 is `0.406250/0.608456/0.704044` versus
`0.463235/0.636029/0.733456`, and Diversity is `9.028303`. At the same exact
training step, candidate loss differs by only `-0.00209561`, masked accuracy
by `-0.066811` percentage points, and visible accuracy by `-1.889969` points.
Evidence is
`output/hhc_global_masked_loss_vs_formal50_validation_10k_30k_live.json` and
`output/hhc_global_masked_loss_vs_formal50_training_trajectory_10k_30k_live.json`.
The refreshed remote W&B audit passes every required training stream through
optimizer step 30,200 and all nine validation streams through 30k. At the
last audited telemetry event, gradient L2/max-abs is
`1.434748/0.060303`, weight L2/max-abs is `370.351410/2.740283`, and both
nonfinite counts remain zero. The weight increase is not independently
diagnostic: direct trainable-tensor checkpoint inspection at 20k gives L2
`340.015875` for the formal reference and `339.856317` for this control,
while the healthy formal reference itself reaches `418.678923@50k` and
`552.304406@300k`. Consequently no clipping or optimizer change is introduced
mid-run; the registered 40k/50k fixed validations decide the control. The
online evidence is
`output/hhc_global_masked_loss_control_mgt_wandb_warmup_audit.json`.
At 40k the same fixed protocol gives FID `1.08753538`, Matching
`3.69119167`, R-Precision Top-1/2/3
`0.443015/0.625000/0.715074`, and Diversity `9.186666`.  Relative to the
synchronous paper masked-only reference, FID and Matching are worse by
`0.11227012` and `0.10865664`, while Top-1/2/3 are lower by
`0.020221/0.011029/0.022059`.  The candidate's exact-step training loss is
nevertheless `0.02333662` lower, again showing that the batch training
objective alone does not predict fixed-sample generation quality.  The
remote audit passes 804 telemetry events through optimizer step 40,200:
gradient L2/max-abs is `0.943263/0.043701`, weight L2/max-abs is
`395.714417/2.756534`, and both nonfinite counts are zero.  The fourth
validation event has loss `10.621068` and masked accuracy `2.933152%`; those
loss-side values remain supporting evidence because the validation corruption
seed changes by milestone, whereas the generation seed and common-50 contract
remain fixed.  The 40k checkpoint contains 947 finite model tensors and 1,041
finite optimizer tensors with scheduler `last_epoch=40000`.  Evidence is
`output/hhc_global_masked_loss_vs_formal50_validation_10k_40k_live.json`,
`output/hhc_global_masked_loss_vs_formal50_training_trajectory_10k_40k_live.json`,
`output/hhc_global_masked_loss_control_40k_checkpoint_audit.json`, and the
refreshed W&B audit above.  The 10k checkpoint remains this run's fixed-FID
best and the preregistered decision still waits for 50k.  The final rolling
checkpoint before that endpoint independently passes at 45k: all 947 model
tensors and 1,041 optimizer tensors are finite, the optimizer still has
weight decay `1e-6`, and the scheduler has `last_epoch=45000`.  Its frozen
state and SHA-256 are recorded in
`output/hhc_global_masked_loss_control_45k_checkpoint_audit.json`.
At the preregistered 50k endpoint, the current fixed-sample FID is
`1.05582905`, versus `0.88623136` for the synchronous paper masked-only
reference; Matching is `3.63405228` versus `3.55796647`, and R-Precision
Top-1/2/3 is `0.446691/0.628676/0.707721` versus
`0.463235/0.641544/0.737132`.  This regression occurs even though the exact
step-49,999 training loss is lower by `0.04541746` and masked accuracy is
higher by `1.328926` percentage points.  Selection over the frozen 10k-spaced
candidate set therefore retains the 10k checkpoint at FID `0.75840902`.
Its absolute improvement over the formal early best is `0.12782234`, but the
relative improvement is only `14.4231%`, below the registered `15%` gate.
The global-loss run is not promoted to 300k; the evidence-bound dispatcher
selects the paper masked-only shape-augmentation branch.  Evidence is
`output/hhc_global_masked_loss_vs_formal50_validation_10k_50k.json`,
`output/hhc_global_masked_loss_vs_formal50_training_trajectory_10k_50k.json`,
`output/hhc_global_masked_loss_control_50k_checkpoint_audit.json`,
`output/hhc_global_masked_loss_control_assessment.json`, and
`output/hhc_post_50k_training_dispatch.json`.
The machine-readable dispatcher is
`tools/dispatch_hhc_post_50k_training.py`: an absolute FID gain of at least
0.1 and a relative gain of at least 15% promotes the same run to an audited
in-place 300k continuation; a negative loss control releases the matched
shape-augmentation experiment after binding the independently preregistered
train-only shape-identity diagnostic.  If neither diagnostic hypothesis
passes, the dispatcher records mask conditioning as unresolved and still runs
the two user-required matched shape-augmentation arms; a negative diagnostic
cannot silently cancel that requested experiment.  Interrupted
50k/300k runs resume only after a finite Lightning checkpoint audit and reuse
their online W&B run identity.  This replaces the former unconditional
post-control exit code that could stop the queue regardless of evidence.
The all-valid control and its non-promotion assessor are
`tools/run_hhc_all_valid_loss_control.sh` and
`tools/assess_hhc_all_valid_loss_control.py`.
Its fixed common-50 validation FID is `0.9772065282` at 10k and
`1.3704825640` at 20k, versus `0.9789292216` and `1.2168186903` for the
paper masked-only reference.  Thus the initially negligible 10k difference
becomes a `0.1536638737` regression at 20k.  The 20k pre/post-validation
checkpoints are bitwise identical for model, optimizer and scheduler state;
the result is not a validation-state side effect.  The online W&B audit currently covers
454 exact 50-step telemetry events through optimizer step 22,700, including
pre-clip FP32 global gradient/weight L2 and max-absolute values and zero
non-finite counts.  Its live validation section separately proves that W&B
received finite `val/loss`, both validation accuracies, FID, all three
R-Precision values, Matching and Diversity at both 10k and 20k; merely finding
metric names in source is not accepted as runtime evidence.  The corresponding evidence is
`output/hhc_all_valid_vs_formal50_validation_10k_20k.json`,
`output/hhc_all_valid_vs_formal50_training_trajectory_10k_20k.json`,
`output/hhc_all_valid_loss_control_20k_pre_post_validation_pair_audit.json`,
and `output/hhc_all_valid_loss_control_mgt_wandb_warmup_live_audit.json`.
The 25k rolling checkpoint independently passes the full finite-state audit:
global step 25,000, epoch 280, 947 model tensors, 1,041 optimizer tensors and
scheduler `last_epoch=25000`; see
`output/hhc_all_valid_loss_control_25k_checkpoint_audit.json`.
If the global masked-token-mean control also fails both preregistered promotion
margins, `tools/run_hhc_historical_mask_control.sh` is the separately gated
50k corruption-only diagnostic.  It changes only `training_mask_contract` from
`paper_kinematic` to `historical_flat_random_replace`; paper architecture,
paper-50, masked-only loss, optimizer, conditioning, seed and common-50
validation remain exact matches.  The launcher refuses to run before the
global-loss assessment emits the negative release decision, and its online
W&B contract is pre-audited in
`output/hhc_historical_mask_control_online_wandb_audit.json`.  This diagnostic
can explain a legacy-performance difference but cannot replace the paper mask
or delay the required shape-augmentation experiment.

Run `tools/audit_hhc_paper_training_contract.py --training-manifest <selected
formal manifest> --source-snapshot-manifest <selected formal run>/launch_source_snapshot_manifest.json`
before final aggregation. Both arguments are required after an authorized
post-training source transition: the selected manifest prevents auditing an
obsolete pre-repair run, while the immutable launch snapshot proves the code
that actually produced that run without treating later diagnostic/shape
changes as historical training drift. Its
machine-readable report,
`output/hhc_formal_raw12288_repaired_mgt_training_contract_audit.json`,
separates every numeric/qualitative setting explicitly stated by the paper from
the raw-12288/175k tokenizer and common-50 user exceptions.  It also binds the
paper PDF, VQ/training manifests and all 14 launched Python sources by SHA-256.
The paper does not specify numerical precision; active BF16 is therefore
reported as an implementation choice.  The relevant visible pre-cleanup
two-person entrypoint is `train_t2i_trans.py`, and it also uses BF16 mixed
precision.  BF16 is therefore not an observed legacy-vs-paper difference and
is not used to explain the current training curve.  (An earlier audit
mistakenly inspected the inherited single-person `train_t2m_trans.py`; the
machine-readable audit now rejects that file.)

### Superseded 8,096-code VQ recovery investigation (backup only)

The commands and observations below are retained to audit the earlier
investigation.  Their VQ/MGT checkpoints and method labels are not inputs to
the active formal table.

VQ-VAE (HHC repository, GPU 1):

```bash
conda run --no-capture-output -n New_MMM python -u train/train_vq_interx.py \
  --device cuda:1 --batch-size 256 --total-iter 150000 \
  --warm-up-iter 1000 --lr 0.0002 --weight-decay 0.000001 \
  --nb-code 8096 --code-dim 256 --down-t 2 --stride-t 2 \
  --exp-name paper_vq_8096_150k_seed123_exact_rawstats_ema_safe \
  --checkpoint-dir checkpoints/vq_paper --wandb-mode offline \
  --precision fp32 --print-iter 200 --eval-iter 5000 --seed 123
```

The final checkpoint is `checkpoints/vq_paper/net_150000.pth`.  The cache
export in the next command is per-sample, atomic and restartable.
The exact launch command, relevant source/data SHA-256 records, published vs.
source-derived settings, and complete-EMA checkpoint policy are frozen in
`checkpoints/vq_paper/training_manifest.json`.

Later shape/action-generalization work changed four source files while this
already-loaded VQ process was still running.  The formal launch bytes—not the
later working-tree versions—are archived under
`output/vq-interx-sep/2026-08-26-00-09-40_paper_vq_8096_150k_seed123_exact_rawstats_ema_safe/source/`.
Its `complete_source_snapshot_manifest.json` verifies all nine training-code
SHA-256 values against
the frozen training manifest and records the exact post-launch task-journal
diffs that were reversed.  Verify the materialized archive by independently
reconstructing the launch bytes, without changing the active training files:

```bash
conda run -n New_MMM python tools/recover_vq_launch_sources.py \
  --verify-archive
```

The architecture and optimizer schedule above follow the paper.  The paper
does not publish the individual geometric reconstruction-loss coefficients;
those are therefore marked as artifact-derived rather than paper-derived.  We
recovered them from the matching MMM W&B run configuration as
`explicit=3`, `velocity=100`, `root=0.1`, `geometry=0.01`,
`foot-contact=10`, and `distance-map=0.1` (with the outer commitment factor
`0.02`).  This distinction is retained in the run metadata.

The VQ training crop is 96 frames because that is the window forced by the
released HHC training source.  The paper itself does not state a crop length;
historical MMM runs include 64-frame windows, so the 96-frame choice is marked
source-derived (not paper-derived) in the audit.
The paper also does not specify codebook-update internals.  The released path's
EMA-reset quantizer (decay 0.99, unused-code reset check every 21 updates) is
retained and labelled source-derived; its otherwise non-buffer accumulators are
included explicitly in every new checkpoint.

VQ checkpoints now include the EMA quantizers' `code_sum`, `code_count`,
`usage`, initialization flag, and reset counter in addition to the registered
codebook tensor.  This prevents a new-format resume from resetting the learned
EMA history.  It does not claim bitwise-identical data order after a restart,
because the source VQ loader does not checkpoint its in-epoch shuffle cursor.
Legacy checkpoints—including the preserved
`normalizer_mismatch_aborted` run—lack these fields, so their compatibility
fallback preserves the loaded codebooks and rebuilds the missing accumulators
with a warning; it cannot reconstruct prior EMA history.  The corrected
`paper_vq_8096_150k_seed123_exact_rawstats_ema_safe` replacement was launched
after both the EMA-checkpoint and observational-evaluation fixes, so every
checkpoint it emits includes the complete EMA training state and no evaluation
export is allowed to update that state.

MGT (start after VQ completion):

```bash
conda run --no-capture-output -n New_MMM python -u train/train_t2i_trans.py \
  --device cuda:1 --exp-name paper_mgt_8096_300k_seed123 \
  --vq-checkpoint checkpoints/vq_paper/net_150000.pth \
  --codebook-dir checkpoints/vq_paper/codebook \
  --batch-size 96 --total-iter 300000 --warm-up-iter 1000 \
  --lr 0.0002 --weight-decay 0.000001 --contact \
  --no-contact-conditioning --shape --paper-architecture \
  --sampler sample_intra_cas --temperature 0.5 --sampling-steps 50 \
  --cfg-scale 3.5 --contact-guidance-scale 1.0 \
  --wandb-mode offline --seed 123
```

This trains without contact inputs, with independent sorted Root/Body/Hand
cosine masks and masked-token-only NLL.  Contact enters only as the paper's
sampling-time code-logit bias.  `last.ckpt` and a rolling latest checkpoint are
written every 5,000 updates; resume with `--resume-training-ckpt PATH`.
The first `latest-000005000.ckpt` state audit passes: global step/scheduler are
5,000, all 947 model and 1,041 optimizer floating tensors are finite, and the
optimizer has 347 parameter-state entries.  Lightning preserves loop,
optimizer and scheduler state but this checkpoint contains no process/DataLoader
RNG snapshot, so a crash resume is numerically valid but not bitwise identical;
the formal run is therefore kept uninterrupted unless recovery is necessary.
The `latest-000010000.ckpt` audit passes the same invariants at update 10,000
(epoch 112, scheduler 10,000).  Its first fixed-protocol validation produced
FID 1.2309, Diversity 8.8516, Top-1/2/3 0.4265/0.5974/0.7096 and Matching
3.7650, retaining `model-000010000-1.2309.ckpt`.  W&B records this validation
at zero-based `trainer/global_step=9999`; the audit/report converts that field
to `completed_updates=10000` explicitly.
The same frozen validation at 20,000 updates improved FID to `1.126821` and
retained `model-000020000-1.1268.ckpt` (SHA-256
`868ddc84bd9db6a0c7241830534696af1aec26de10c7513062e46b07399ba5ce`).
At 30,000 updates FID regressed to `1.422628`, so 20k remained the selected
milestone.  Both the rolling 30k state and
`model-000030000-1.4226.ckpt` pass full model/optimizer finiteness and exact
global-step/scheduler audits; the latter's SHA-256 is
`c15249eeb1fc571b95f36ec03fe50d08d03ebc31ad208bc376317aa4b3ee26f2`.
The superseded 8,096-code investigation subsequently completed all 300,000
updates.  Its minimum milestone FID remained `1.126821` at 20k and its 300k
milestone was `2.258645`; all 30 milestones are frozen in
`output/historical_mmm_wandb_comparison.json`.  The active raw-12,288 run uses
the same paper architecture/training source hashes and already improves the
20k milestone to `0.780112`, but also shows the same later-regression pattern
through 60k.  This repetition is diagnostic evidence about the corrected MGT
trajectory, not a fair VQ comparison and not a reason to substitute the
superseded 8,096 tokenizer.
The validation hook's Diversity, Top-1/2/3 and Matching fields are running-best
state, not fresh per-milestone estimates; only FID is used for this checkpoint
trajectory.  Independent common-protocol evaluation will recompute every
reported metric for the final selected checkpoint.
Checkpoint selection uses the 570-entry validation split copied byte-for-byte
from MMM.  One 1,039-frame sequence (`G035T005A029R021`) is excluded by the
official `<1000` rule, leaving 569 eligible and 544 evaluated examples in 17
complete batches of 32.  The active 24-frame minimum and 150-frame cap match
InterMask and the official Inter-X evaluator.  MMM development code instead
contains a legacy 30-frame/152-frame loader contract; it yields the same number
of eligible validation examples here, but 299 clips are truncated at a
different boundary.  This is recorded as another reason its historical W&B
metric is not directly comparable.  The final test table remains the separate
1,696-sample official test protocol, not this checkpoint-selection split.
The training manifest also lists a batch-96, five-batch validation loader:
that is Lightning's token-loss monitor (480 entries after `drop_last`) and is
never used for FID or checkpoint selection.  The contract audit records both
loader roles explicitly so those counts cannot be conflated.
`tools/audit_mgt_validation_selection.py` runs the exact shuffled loader twice
with seed 500123 and freezes all 544 ordered sample/caption/length records plus
a streaming hash of the word/POS tensors, normalized GT crop, contact crop and
length tensors.  The final report refuses checkpoint selection unless both
runs are byte-deterministic and the manifest contains all 17 complete batches.
Its direct-script entrypoint explicitly prepends the repository root before
importing `dataset`/`utils`; this matters because the final queue invokes the
file as `python tools/audit_mgt_validation_selection.py`.  A direct `--help`
regression test and the real two-pass 569-eligible/544-evaluated audit both
pass, preventing an end-of-queue import failure.  The selection manifest now
also stores the audit entrypoint's path, size and SHA-256, and final report
validation resolves that exact file or a content-addressed snapshot; the
deterministic record/input hashes are therefore bound to the implementation
that produced them.
The same direct-entrypoint audit found and fixed the identical repository-root
import issue in `tools/render_fair_interx_qualitative.py`.  All 19 Python
entrypoints invoked by the final report queue now complete a real `--help`
import smoke test, and both formerly failing entrypoints are covered by the
paper-consistency regression suite.
Checkpoint-selection validation uses the same requested common 50-step,
temperature-0.5 sampler as the primary fair table.  A separate 10-step audit
retains the paper-native Fig.-2 budget.
Here, an HHC "step" is one outer MaskGIT cascade iteration.  Each iteration
attempts Root, Body and Hands sequentially while unresolved masks remain.  A
full-length sequence therefore performs at most 150 active channel stages and,
with conditional/unconditional CFG evaluation, at most 300 model forward
calls.  Short sequences can resolve before the final outer iteration and skip
those later forwards, but the cosine schedule still executes all 50 outer
iterations.  New HHC generation HDF5 files record these quantities explicitly
as maxima.  `output/hhc_sampling_runtime_audit.json` demonstrates both the
full-length 150/300 case and a one-token 3/6 case at runtime; the final report
rejects stale or failed audit evidence.  It also includes measured wall-clock
throughput and does not present equal step counts as equal compute.
`output/hhc_formal_launch_contract_audit.json` separately binds the training
validation and fair-export source hashes and verifies that both paths forward
the same sampler, steps, temperature, CFG, contact-guidance scale, raw Inter-X
normalization and train-only contact map.  Validation uses a fixed preserved
RNG stream; the final table exposes both request-oracle contact timing and the
no-contact-guidance ablation explicitly.
Every milestone uses the same fixed
`training_seed + 500000` validation seed inside a
Python/NumPy/Torch/CUDA RNG-preservation context, so stochastic validation does
not alter later masks, dropout, caption/crop augmentation, or optimizer updates.
This is also verified on the real 90k run: the pre-validation rolling
checkpoint and post-validation FID checkpoint have bitwise-identical 947 model
tensors, optimizer state, scheduler state, and hyperparameters.  Only expected
callback/loop validation metadata differs.  The bound result is
`output/hhc_precleanup_raw12288_corrected_90k_pre_post_validation_pair_audit.json`.
The two source checkpoints are intentionally ephemeral under Lightning's
rolling `save_top_k=1` policy and have since expired.  The final report queue
therefore reruns the pair audit when both files still exist; otherwise it
accepts only this frozen audit after checking its schema/status, expected
step, whole-file SHA-256 (`d481e46c...`), all four exact-state flags, empty
differing-model-key list, both source SHA-256 values, and the
before/after/best FID values.  This fallback branch was
exercised successfully after the source checkpoints expired, so a normal
rolling-checkpoint lifecycle cannot break final report generation or silently
weaken the pair claim.
The same check independently passes again at 100k in
`output/hhc_precleanup_raw12288_corrected_100k_pre_post_validation_pair_audit.json`
(current milestone FID `2.432243585586548`; validation-best remains the 20k
checkpoint at `0.78011155128479`).
The 105k rolling checkpoint also passes the full finite-state audit (SHA-256
`f40676a95cb095638763b0748d43c06c82d3dddd31dbd9fc2f17a778b025c2f8`).
At 110k the current milestone FID improves to `2.0190837383270264`; the
pre/post-validation pair again has bitwise-identical model, optimizer,
scheduler and hyperparameter state.  The bound pair audit is
`output/hhc_precleanup_raw12288_corrected_110k_pre_post_validation_pair_audit.json`;
the rolling and validation checkpoint SHA-256 values are respectively
`c70ee81bbd9e3795a5dbdfaaca7a870413d2727dbdfaecada850d39c58118eda` and
`6c1483fc0520ad7b2301a7dcb82b442ef9f6d32e5bb2fbc03889c9578c3d6987`.
The 115k rolling checkpoint independently passes the same finite model,
optimizer and scheduler-state audit (SHA-256
`e0d90d9c716571db9247d225b63c45a9b497a8441c9be84b11a95734526a3b8f`).
At 120k the current FID is `2.2572407722473145`; the pre/post-validation model,
optimizer, scheduler and hyperparameter states remain bitwise identical.  The
pair audit is
`output/hhc_precleanup_raw12288_corrected_120k_pre_post_validation_pair_audit.json`;
rolling/validation checkpoint SHA-256 values are
`e359d76f635d81cb66ce90756d0041ff366954305706f57d8976d6634ac0cdf6` and
`ad9c61b1a2af262402cbefd54e4710a377374e88217836be46b150287934bd07`.
The 125k rolling checkpoint also passes the finite state/step/scheduler audit
(SHA-256
`5d902d50022f98d4867bdfcd8a75db3e3fde995d6c33dde7431e64e2ef358ed9`).
At 130k the current FID is `1.720682144165039`; the pre/post-validation model,
optimizer, scheduler and hyperparameter states are again bitwise identical.
The pair audit is
`output/hhc_precleanup_raw12288_corrected_130k_checkpoint_pair_audit.json`;
rolling/validation checkpoint SHA-256 values are
`625240ee3a921977cdd326b7cac0d0fbc48c0e337ba38e52e0a0c067ce879653` and
`23a9587a5cbfa6fc28b3aea8dd8a80e4ca4712735c87fb423215be5b77f8a6d3`.
The 135k rolling checkpoint also passes the finite state/step/scheduler audit
(SHA-256
`9e5eab0b197b7d11f4553df577c6f699dbd50c85960e76f0e0032a5e27819646`).
At 140k the current FID is `1.8247472047805786`; the pre/post-validation
model, optimizer, scheduler and hyperparameter states remain bitwise
identical.  The pair audit is
`output/hhc_precleanup_raw12288_corrected_140k_checkpoint_pair_audit.json`;
rolling/validation checkpoint SHA-256 values are
`5e25dd68416f09d74b131999d92be3406d78c9bad6a0ea5effd56c472c371561`
and `a478dcf34f7f59619275fda64c7f932e644110ea46bb8aa37ebef2f065c45e8c`.
The 145k rolling checkpoint also passes the finite state/step/scheduler audit
(SHA-256
`9b30b05be2e6db13cee8b20ddc01d49f17242fffacdb11a688b459a3ac686637`).
At 150k the current FID is `2.2717506885528564`.  This value ranks outside
the callback's `save_top_k=10`, so Lightning correctly removed the validation
checkpoint instead of retaining an eleventh model.  The 150k rolling
checkpoint was observed complete before the next `save_top_k=1` rolling save
replaced it, but it was not independently content-audited; no pairwise-state
claim is made for this milestone.  The subsequent 155k rolling checkpoint
passes the finite state/step/scheduler audit (SHA-256
`4fb0793d27ad859d180616d35457e8b8939c266ecf95f4b6a5ae03287dfd19df`).
The 160k rolling checkpoint was audited immediately on atomic completion and
also passes (SHA-256
`48aa3422b53b557ee7dad8dbcf8d4ef00ac95abc924b63d5b01de14390d09a61`);
the full 544-sample validation FID is `2.3308000564575195`.  Like 150k, this
does not enter `save_top_k=10`, so no validation checkpoint is retained and no
rolling/validation pairwise-state claim is made for this milestone.
The 165k rolling checkpoint also passes the finite state/step/scheduler audit
(SHA-256
`941e3486644ac2a0ab94db600ae64e5dfdb842251a007765299e9fb0e56b54b6`).
The 170k rolling checkpoint was likewise audited immediately on atomic
completion and passes (SHA-256
`03a4d5c7098bec0396dddb243e885843692f88bfb397e6ab363876af6da7187d`);
the full validation FID is `1.8352792263031006`.  The retained post-validation
checkpoint has SHA-256
`7a0ed442e8afd0c4ce9a207df11afbe9ee549fa0a427b8aa104d2da35173027f`;
model, optimizer, scheduler and hyperparameters are bitwise identical to the
rolling checkpoint.  The pair audit is
`output/hhc_precleanup_raw12288_corrected_170k_checkpoint_pair_audit.json`.
Its pre-validation callback `current_score` is correctly the most recent
retained 140k score (`1.8247472047805786`), not the independently logged
150k/160k W&B scores that fell outside `save_top_k=10`.
The 175k rolling checkpoint also passes the finite state/step/scheduler audit
(SHA-256
`f930f0788e18ec8346790b726f6b7ac10980fcf29f733e1027c719106d2ad56a`).
The 180k rolling checkpoint was audited immediately on atomic completion and
passes (SHA-256
`7a5a2121bdafba6c7be0da42fd74535988d672102a2f9eee70a5097c7c09610e`);
the full validation FID is `2.021514892578125`.  The retained post-validation
checkpoint has SHA-256
`393a2bcff7c4d6f98772f516c4372ad9da8826ae5b23d8bdd7a62bbde7ff8ea5`;
model, optimizer, scheduler and hyperparameters are bitwise identical to the
rolling checkpoint.  The pair audit is
`output/hhc_precleanup_raw12288_corrected_180k_checkpoint_pair_audit.json`.
The 185k rolling checkpoint also passes the finite state/step/scheduler audit
(SHA-256
`0d4cc68d9cefdbc53c31489682ebb026ae96b501ad17ab26411885af2b469bbd`).
The 190k rolling checkpoint was audited immediately on atomic completion and
passes (SHA-256
`91d81014a31be0fd703d4f14778ec824349aa0517231d3af05f6986b92aebce9`);
the full validation FID is `2.21270751953125`.  After the 170k and 180k
checkpoints entered the retained set, the worst `save_top_k=10` score tightened
to `2.021514892578125`; therefore 190k is correctly not retained and no
rolling/validation pairwise-state claim is made for this milestone.
The 195k rolling checkpoint also passes the finite state/step/scheduler audit
(SHA-256
`9ad3648da7189555b6042fc39eacf07900d83ff47e5dd882fa98d51f972aac21`).
The 200k rolling checkpoint was audited immediately on atomic completion and
passes (SHA-256
`99046a70eb70dfa2369950197f836f93be044d8a7a447c0ac6987461e2755280`);
the full validation FID is `1.6259876489639282`.  The retained post-validation
checkpoint has SHA-256
`ea12750b23e1fda361191ce8e33b31db9e1b2ac3411d0800402cce3a8c7c2d1b`;
model, optimizer, scheduler and hyperparameters are bitwise identical to the
rolling checkpoint.  The pair audit is
`output/hhc_precleanup_raw12288_corrected_200k_checkpoint_pair_audit.json`.
The 205k rolling checkpoint was likewise audited immediately after atomic
completion and passes the step, scheduler and finite-state checks: all 947
model floating tensors and all 1,041 optimizer floating tensors are finite.
Its SHA-256 is
`9fc82d299d778916a679a6f1233a3e8b09bd1e4f0cafc028fb9875abf4ccdb2c`, and the
audit is `output/hhc_precleanup_raw12288_corrected_205k_checkpoint_audit.json`.
The 210k rolling checkpoint passes the same audit (SHA-256
`6093dfbbf4e48bc88c1efae4a84715a9a930b39d2197a9a03f22b1acf9199067`).
Its full validation FID is `1.788170337677002`; the retained post-validation
checkpoint has SHA-256
`007a6877e4946556841e38915b4360557804e7ee7a6fbedd7e0d45bd50f6d6a9`.
The two checkpoints are bitwise identical for model, optimizer, scheduler and
hyperparameters, as recorded in
`output/hhc_precleanup_raw12288_corrected_210k_checkpoint_pair_audit.json`.
The best validation checkpoint remains 20k at `0.78011155128479`.  The actual
retained file used by final selection, `model-000020000-0.7801.ckpt`, also
passes an independent step/scheduler/finite-state audit; its SHA-256 is
`7fcc995cffdd6017e0655df52ad8476f9dfddae1c40e7dfd12b58293deff2fcd` and the
audit is
`output/hhc_precleanup_raw12288_corrected_20k_validation_checkpoint_audit.json`.
The 215k rolling checkpoint also passes the step/scheduler/finite-state audit;
its SHA-256 is
`fbc5984133f9a310041c57085c74ef5c3f9995955dc01ed547a187446c957f39`, with
the machine-readable result in
`output/hhc_precleanup_raw12288_corrected_215k_checkpoint_audit.json`.
The 220k rolling checkpoint also passes (SHA-256
`5add5c2d52809dc010ad45adca0f673aba04783450eb8858b5ea28fe9a533da9`).
Its full validation FID is `2.696781873703003`, above the current worst retained
top-10 FID `1.9620590209960938`; no validation checkpoint is therefore retained and no
rolling/post-validation pairwise-state claim is made at 220k.  The rolling
audit is `output/hhc_precleanup_raw12288_corrected_220k_checkpoint_audit.json`.
The 225k rolling checkpoint also passes the step/scheduler/finite-state audit;
its SHA-256 is
`9c0b9590e31fe7c779c8b7d015b76e3e22e9825ba23c1cf6c8deeb4a06d4691c`, and
the audit is `output/hhc_precleanup_raw12288_corrected_225k_checkpoint_audit.json`.
The 230k rolling checkpoint also passes (SHA-256
`8cbcc797fca3a3785bdf8b81ef618fbb12f9101ee0bc94230a760ff321dc9b7c`).
Its full validation FID is `2.5630898475646973`, above the current worst
retained top-10 FID `1.9620590209960938`; no validation checkpoint or pairwise
claim is therefore made.  The rolling audit is
`output/hhc_precleanup_raw12288_corrected_230k_checkpoint_audit.json`.
The 235k rolling checkpoint also passes the step/scheduler/finite-state audit;
its SHA-256 is
`405e8f94a6515260c29d2f2981aad0b620d957954f07cbbfefafa32280ffb57b`, and
the audit is `output/hhc_precleanup_raw12288_corrected_235k_checkpoint_audit.json`.
The 240k rolling checkpoint also passes (SHA-256
`242b6d9f21c9d63068e86306ff9dcf12686a4234df881165a58ebe73c68948e8`).
Its full validation FID is `2.377925157546997`, above the current worst
retained top-10 FID `1.9620590209960938`; no validation checkpoint or pairwise
claim is therefore made.  The rolling audit is
`output/hhc_precleanup_raw12288_corrected_240k_checkpoint_audit.json`.
The 245k rolling checkpoint also passes the step/scheduler/finite-state audit;
its SHA-256 is
`828a99cd85c41b770249eb69557cfc0e03b29e26d4531ca73449586f61f059c4`, and
the audit is `output/hhc_precleanup_raw12288_corrected_245k_checkpoint_audit.json`.
The 250k rolling checkpoint also passes (SHA-256
`2c537265f34476a2e02f5284531ec3b991207f81ac4a791e4d49cce75d7ceb78`).
Its full validation FID is `2.464520215988159`, above the current worst
retained top-10 FID `1.9620590209960938`; no validation checkpoint or pairwise
claim is therefore made.  The rolling audit is
`output/hhc_precleanup_raw12288_corrected_250k_checkpoint_audit.json`.
The 255k rolling checkpoint also passes the step/scheduler/finite-state audit;
its SHA-256 is
`1f85c02722035e1e36972111aa8d0808d3f17d45bf06dc0fc1356db9cec0287c`, and
the audit is `output/hhc_precleanup_raw12288_corrected_255k_checkpoint_audit.json`.
The 260k rolling checkpoint also passes (SHA-256
`36244bf03e5769b7a6864a0056276c36d3a2b6101720014ccf3ea8e047d9065b`).
Its full validation FID is `2.4598724842071533`, above the current worst
retained top-10 FID `1.9620590209960938`; no validation checkpoint or pairwise
claim is therefore made.  The rolling audit is
`output/hhc_precleanup_raw12288_corrected_260k_checkpoint_audit.json`.
The 265k rolling checkpoint also passes the step/scheduler/finite-state audit;
its SHA-256 is
`7764a71995192cd72a05164045bf8d736a0ee2defb90d4d5e8d9839261b44882`, and
the audit is `output/hhc_precleanup_raw12288_corrected_265k_checkpoint_audit.json`.
The 270k rolling checkpoint also passes (SHA-256
`1fdb614ab29a7445b7bfebf1d8baf618a1edc83215b32b7259cd7ea899ebbe0e`).
Its full validation FID is `2.8896591663360596`, above the current worst
retained top-10 FID `1.9620590209960938`; no validation checkpoint or pairwise
claim is therefore made.  The rolling audit is
`output/hhc_precleanup_raw12288_corrected_270k_checkpoint_audit.json`.
The 275k rolling checkpoint also passes the same global-step, scheduler and
finite model/optimizer-state audit; its SHA-256 is
`ec978f51d3662e1ec0f96633589328677c49f98d201c3ca633e3d8fbfe0c639a`, with
evidence in `output/hhc_precleanup_raw12288_corrected_275k_checkpoint_audit.json`.
This is a health checkpoint only; the next validation occurs at 280k.
The 280k rolling checkpoint also passes (SHA-256
`936edb88e3a9e6cd09ea85933c880cc7de9bffed1e007fa3bf127125ce8f1547`).
Its full validation FID is `2.4878127574920654`, again above the retained
top-10 cutoff `1.9620590209960938`; Lightning therefore correctly keeps no
`model-280k` checkpoint and no rolling/validation pair claim is made.  The
rolling audit is `output/hhc_precleanup_raw12288_corrected_280k_checkpoint_audit.json`.
The 285k rolling checkpoint also passes the full step/scheduler and finite
model/optimizer-state audit.  Its SHA-256 is
`04da61a5559ae39fbc4ab66633190668e6fb94b5055abe92337099df23b50165`;
the bound audit is
`output/hhc_precleanup_raw12288_corrected_285k_checkpoint_audit.json`.
The 290k rolling checkpoint likewise passes: `global_step=290000`, scheduler
`last_epoch=290000`/`_step_count=290001`, and all 947 model plus 1,041
optimizer floating tensors are finite.  Its SHA-256 is
`2722397200ef012091ede182365525fa28f63379f9f7eee2f538c0575475c728`.
The corresponding 544-sample fixed validation FID is
`2.781623601913452`, above the retained top-10 cutoff
`1.9620590209960938`; consequently no `model-290k` checkpoint was retained
and no rolling/validation pair claim is made.  The state evidence is
`output/hhc_precleanup_raw12288_corrected_290k_checkpoint_audit.json`.
The 295k rolling health checkpoint also passes with `global_step=295000`,
scheduler `last_epoch=295000`/`_step_count=295001`, and all 947 model plus
1,041 optimizer floating tensors finite.  Its SHA-256 is
`a847abcfe4c1cc498a160f4fa8366b7714fb42ee2217b1db14b4ad6c89c55747`,
with evidence in
`output/hhc_precleanup_raw12288_corrected_295k_checkpoint_audit.json`.
The run completed all 300,000 updates and all 30 fixed validations.  The
terminal checkpoint passes the same full audit at epoch 3,370 with scheduler
`last_epoch=300000`/`_step_count=300001`; all 947 model and 1,041 optimizer
floating tensors are finite, and its SHA-256 is
`4d25e9ffbc68810d46d1ba6ecef044e9548d91036cfa8c9d921495e201ba1a8a`.
Its current-milestone validation FID is `2.5655829906463623`.  Fixed-validation
selection therefore remains the 20k checkpoint at exact FID
`0.78011155128479`, SHA-256
`7fcc995cffdd6017e0655df52ad8476f9dfddae1c40e7dfd12b58293deff2fcd`;
test metrics were not used.  The independent state and selection records are
`output/hhc_precleanup_raw12288_corrected_300k_checkpoint_audit.json` and
`output/hhc_mgt_precleanup_raw12288_selection.json`.

The original long-lived GPU-1 parent shell disappeared after the successful
trainer shutdown but before writing the selection manifest; the terminal
checkpoint, W&B stream and all 30 validation records were already complete.
The selection command was replayed successfully, and downstream work resumed
without retraining through
`tools/resume_after_hhc_precleanup_training_gpu1.sh`. Subsequent MGT diagnosis
showed that this refactored run is invalid for the final table; the launcher is
now retained only as a fail-fast historical record. No active optimizer,
augmentation, paired comparison, qualitative render or report consumes its
selection or generations.
The historical diagnostic used the checkpoint with the lowest FID under its
fixed validation protocol; test metrics did not select it. Replication 0
contains 1,696 main samples plus 100 fixed
multimodality prompts with 30 generations each, and records FID `0.6453559`,
matching `3.8436885`, Top-1/2/3
`0.400943/0.584906/0.694575`, diversity `8.9324675`, and official
multimodality `0.4061893`.  The 10-cm physical audit records contact F1
`0.5545339`, temporal IoU `0.3836367`, false-contact rate `0.1531794`, and
foot sliding `0.352998 m/s`.  These are replication-0 diagnostics only; do not
report them as the final 20-replication mean or confidence interval.  The
same-request effect-size comparison against the pre-cleanup 50k and Git-default
280k MGT checkpoints is stored in
`output/fair_eval/paired_hhc_mgt_training_state_sensitivity_rep0.json` and is a
required final-report artifact.
The released HHC training path uses BF16 mixed precision for MGT; the paper
does not state numerical precision, so this is recorded as source-derived
rather than a paper-specified hyperparameter.
Likewise, the paper reports inference CFG 3.5 but not its training-time
unconditional probability.  The released model drops both global and word text
embeddings with probability 0.1 while retaining shape/gender; this is preserved
and recorded as source-derived in the training manifest.
The nine blocks self-attend only the interleaved motion tokens; global CLIP and
shape/gender features enter through AdaLN, while CLIP word features are the
cross-attention keys and values.  Variable sequence length is handled by the
attention mask; the paper path does not expose a legacy `[END]` token to the
motion stream.
Unreported architectural details are retained from the released path and
labelled source-derived: FF expansion 4, dropout 0.1, CLIP ViT-B/32 and a
50-motion-token maximum.  Likewise, the VQ convolutional internals are width
512, depth 3, dilation growth 3, ReLU and no normalization.

The token/contact cache writes `_cache_manifest.json` before the first sample
and binds all partial/final entries to the VQ checkpoint SHA-256, codebook size,
latent width, downsampling factor, cache seed, train/validation split hashes and
expected sample count.  It contains only the 9,664 loader-eligible
train+validation
samples (9,095 train and 569 validation; present in HDF5 and 30–999 frames).
The released MMM builder's undocumented `single/single/double` crop
distribution is retained, but its two random choices are keyed by
`SHA256(cache_seed, sample_name)`.  Consequently, shuffling, worker scheduling,
and skipping already-complete samples during resume cannot change a sample's
cached tokens.  The versioned crop-policy identifier is part of the manifest,
so caches produced by a different policy are rejected.  Although the source
`all.txt` also lists test samples, the cache loader filters to the currently
missing eligible train+validation names before loading or tokenization; test
motions are never passed through the cache-building VQ.
The code-contact probability map is estimated from those 9,095 training
samples only; validation/test contact labels cannot leak into sampling
guidance.  For every observed code the map is the paper's exact empirical
`contact_count / total_count`; an unseen code is assigned neutral probability
0.5 because its empirical value is undefined and the sampler takes `log(p)`.
The estimator identity is recorded in the manifest.  An interrupted export
resumes only when that identity matches; a
cache from another VQ or split is rejected instead of silently reused.  On
resume from any pre-`complete` state, the small train-only contact map is
deterministically rebuilt from the verified token/contact files even if a map
file is already present; contact-guided generation refuses a partial manifest.
On completion the manifest records aggregate content hashes for every code-token
and contact-token file plus the derived contact-guidance map SHA-256.  A later
training resume recomputes and verifies all three; the MGT dataset must expose
exactly the expected split names instead of silently swallowing missing text or
shape assets.  The loaded VQ is explicitly frozen and excluded from the MGT
optimizer.
During generation, Transformer checkpoint loading also excludes its duplicated
VQ submodule keys, so the independently named and hashed `--vq-checkpoint`
cannot be silently replaced by a tokenizer copy embedded in the MGT file.
Before the first optimizer step, `training_manifest.json` is written beside the
MGT checkpoints.  It binds the exact command and dirty-code snapshot to the
final VQ, token/cache manifest, contact map, official split/statistics,
evaluator and CLIP weights, aggregate shape/text inputs, active loader counts,
and all paper/source-derived hyperparameters.  A resumed run additionally
records the full Lightning checkpoint hash.

The released MGT token dataset initializes its training bucket at 20 latent
tokens, a threshold not stated in the paper.  A read-only preflight with the
complete historical cache resolves 9,095 train and 569 validation entries and
exposes 8,617/534 at this threshold.  A superseded rebuilt-cache diagnostic
resolved the same 9,095/569 IDs but exposed 8,606/534 because 11 training
sequences crossed the boundary under a newly frozen per-sample crop; that cache
is not used by the selected raw-12,288 formal MGT.  Its active manifest instead
binds the source-compatible historical cache and records 8,617/534, yielding 89
and 5 drop-last batches at batch 96.  Split membership did not change.  The train-only contact map is
still estimated from all 9,095 eligible train entries rather than only the
long-motion bucket.

## Contact-map provenance and legacy implementation sensitivity

The fair HHC path uses
`checkpoints/vq_precleanup_raw12288/code_contact_probs.npy` under the strict
`paper_formula_train_only` contract.  It is recomputed from 9,095 eligible
official-training samples, treats cached indices as `[T,3]` Body/Hand/Root
tokens, and assigns unseen codes the neutral probability 0.5.  The exporter
rejects a map whose VQ, normalization statistics, map hash, scope, or estimator
does not match its cache manifest.

The original MMM file is preserved byte-identically as
`code_contact_probs_mmm_legacy_full_axis_bug.npy`.  Auditing shows that its
cache covers train/val/test (`9,095/569/1,706`) and that `test_data.py` indexes
cached `[1,T,3]` arrays with `indices[:, part_idx]`.  This selects temporal rows
0/1/2 and all three channels, not one part over time.  Exact replay reconstructs
the saved map byte for byte.  It may therefore be selected only with the
explicit `mmm_legacy_full_axis_bug` contract, which is restricted to
replication 0 and excluded from the fair main table.  Reproduce and compare it
with:

```bash
conda run -n New_MMM python tools/audit_hhc_contact_guidance_maps.py
conda run -n New_MMM python tools/compare_fair_interx_methods_paired.py \
  --reference fair_map=output/fair_eval/hhc_formal_raw12288_repaired_mgt_common50_contact \
  --comparators legacy_bug_map=output/fair_eval/hhc_formal_raw12288_repaired_mgt_common50_contact_legacy_full_axis_bug_rep00 \
  --expected-replications 1 --replication-ids 0 \
  --output output/fair_eval/paired_contact_map_implementation_sensitivity_rep0.json
```

This one-replication result reports implementation-sensitivity effect sizes;
it is not an inferential comparison and cannot replace the 20-replication fair
result.

For both training-time shape-augmentation arms, the materialized overlays copy
the original raw Inter-X `Interx_Mean.npy` and `Interx_Std.npy` byte for byte;
statistics are not recomputed on synthetic shapes.  The result builder checks
the official source hashes against the overlay, token-cache manifest, MGT
training arguments, and generation provenance.  It also requires the augmented
train-only map contract and distinguishes the no-timing and frozen-request
oracle-timing rows explicitly.  Both the GPU queue and
`tools/build_shape_augmentation_results.py` are bound to the repaired formal
MGT selectors and `hhc_formal_raw12288_repaired_mgt_shapeaug_*` generation
directories; outputs from the retired `hhc_mgt_shapeaug_*` runs are rejected.
The queue first regenerates and verifies
`output/interx_shape_augmentation_launch_contract_audit.json`, then each arm
must pass its paper-training audit and exact 20k pre/post-validation state audit
before any fair-test generation starts.

## CoShMDM on the same Inter-X split

The author's `prepare/download_pretrain_model.sh` labels its only released
`coshmdm.ckpt` as `# InterHuman`; neither the release nor the local checkout
contains an author-trained Inter-X checkpoint.  Reusing that checkpoint would
therefore violate the same-training-dataset comparison.  The run below is an
explicitly labelled Inter-X retrained port, not a replacement for a compatible
author checkpoint.

Run from `/mnt/8T-HDD/wenning/coshmdm-code` on GPU 0:

```bash
conda run --no-capture-output -n New_MMM python -u tools/train.py \
  --dataset interx --gpu 0 --seed 123 \
  --batch-size 32 --accumulate-grad-batches 2 --no-progress-bar \
  --stats-prefix interx_global \
  --exp-name CoShMDM-InterX-fair-statsfix-seed123-bs32x2
```

This is the author's 2,000-epoch FP32 setup with effective batch 64.  The
released source also calls `torch.set_float32_matmul_precision('medium')`, so
RTX 4090 matrix multiplications are TF32-eligible; this numerical mode is
retained and recorded separately from Lightning's `precision=32` setting.  The
physical split is required because batch 64 uses more than a 24 GB 4090.  Run
`tools/check_training_stability.py` before a fresh launch after changing data
statistics or loss code.  Checkpoints are written every 20,000 effective
optimizer updates.  Resume the Lightning optimization state (including AdamW,
scheduler, epoch/update counters, loops and callbacks) with:

```bash
conda run --no-capture-output -n New_MMM python -u tools/train.py \
  --dataset interx --gpu 0 --seed 123 --batch-size 32 \
  --accumulate-grad-batches 2 --no-progress-bar \
  --stats-prefix interx_global \
  --exp-name CoShMDM-InterX-fair-statsfix-seed123-bs32x2 \
  --resume-checkpoint checkpoints/CoShMDM-InterX-fair-statsfix-seed123-bs32x2/model/last.ckpt
```

The released dataset chooses captions, long-motion crops and person order in
two worker processes, and Lightning does not serialize those worker RNG
streams.  A mid-epoch resume therefore preserves the optimization state but is
distributionally—not bitwise—equivalent to uninterrupted training.  An earlier
300k continuation omitted the normalization setting and silently fell back to
the released InterHuman `global_mean/std`.  Those arrays differ from the
official-train `interx_global_mean/std`; its 300k/320k/340k artifacts are kept
only as diagnostics and are explicitly excluded.  Because the exact 300k file
had already been rotated by the author's `save_top_k=1` callback, the corrected
arm restarts from update 0 under a new experiment name.  The training entrypoint
now requires `--stats-prefix interx_global`, embeds both statistics hashes in
every checkpoint, and refuses a resume whose embedded runtime contract differs.
`tools/audit_coshmdm_resume_provenance.py` binds the corrected static manifest,
live runtime manifest, excluded predecessor and all corrected logs.  The author
target remains 568k for disclosure, but the user imposed a later two-day cap:
stop after a stable 200k milestone, or at the hard deadline
`2026-09-01T17:51:54+01:00`, whichever comes first.  The cap is enforced by
`coshmdm-two-day-cap-v2.service`; it terminates the superseded pre-cap waiter
before interrupting the trainer, so the old queue cannot silently resume toward
568k.  Its machine record is `output/coshmdm_training_budget_cap.json`.
After the cap finalizes, every completed 20k milestone at or below 200k is
scored only on the frozen 544-record validation request.  Resumable validation
HDF5 filenames include the first 16 hex characters of the checkpoint SHA-256,
and the selector re-hashes the checkpoint after export.  The final full-state
audit is explicitly given the validation-selected global step.  The report
builder independently recomputes the minimum `(validation FID, step)` row and
rejects stale same-step caches, non-finite candidate scores, or a selected row
that does not match the recorded checkpoint hash.
`coshmdm-milestone-preserver-v1.service` preserves each stable 20k callback
checkpoint through step 200k as a same-inode hard link.  This is required
because the released callback keeps only one top-k file; without preservation,
earlier validation candidates and the cap monitor's 200k target would vanish.
The 60k link was created successfully with inode `126307594`; its audited
checkpoint SHA-256 is
`b3710d511b28ac985e2f7f23f9d5703d310f2cd4121a457eeea2937abcb869e5`.
Future queue restarts call `tools/resolve_lightning_checkpoint_state.py` and
resume only a checkpoint containing model, optimizer, scheduler and Lightning
loop state.  The legacy YAML `TRAIN.RESUME` remains a model-only warm start and
must not be used as if it were a stateful resume.

The 9,110 training IDs are mirrored to 18,220 dataset entries.  The author's
physical batch 64 with `drop_last=True` yields `floor(18,220 / 64) = 284`
updates per epoch and drops 44 entries.  Micro-batch 32 yields 569 micro-batches;
the unmatched final half-batch is explicitly discarded at every epoch boundary,
so accumulation exposes the same 18,176 entries and the 2,000 epochs correspond
to exactly 568,000 optimizer updates (below the 1,000,000 safety cap).  Carrying
that half-batch into the next epoch would be a subtly different 569,000-update
budget and is rejected for the final run.  The two-worker loader assigns
micro-batches to workers differently from a physical batch-64 loader, so its
caption/crop/person-order augmentation stream is distributionally equivalent,
not bitwise identical; the sample exposure, optimizer batch grouping, epoch
schedule and update count are identical.  The replacement run's first boundary
provided a runtime check: epoch 0 completed exactly 284 updates and logged
`dropped_incomplete_accumulation_micro_batches=1` before epoch 1 began.
This is not accepted from the static manifest alone. The final report reruns
`tools/audit_coshmdm_statsfix_startup.py`, hashes the complete log and every
TensorBoard event file, and requires completed-epoch scalar steps to be
contiguous from zero through the actual capped terminal epoch, with every
recorded discard value equal to one.  It records that the original 2,000-epoch
budget was not completed instead of treating that as a failure of the explicitly
labelled capped arm.
The active stats-fixed run's first 20k checkpoint passes the independent state
audit at global step 20,000/epoch 70: scheduler `last_epoch=70`, all 346 model
and 588 optimizer floating tensors finite, and all 196 optimizer parameter-state
entries present. The step checkpoint SHA-256 is
`140a2496e941615c3377cb186f187c2c283455e2fb033632ba5694d5387a338d`.
It and `last.ckpt` have bitwise-identical model, optimizer, scheduler, loop,
hyperparameter and embedded runtime-contract state; only Lightning callback
last-file metadata differs. The embedded `interx_global` mean/std hashes are
`aa281b878...f568d` and `30a83e929...0b104`, so this artifact cannot be
confused with the superseded `bfb4...` predecessor checkpoint. Evidence is
`output/coshmdm_statsfix_20k_checkpoint_audit.json`,
`output/coshmdm_statsfix_20k_last_checkpoint_audit.json` and
`output/coshmdm_statsfix_20k_checkpoint_pair_audit.json`.
Because the author's `save_top_k=1` callback will unlink the step-labelled file
at 40k, the exact same inode is retained without another 2.7 GB copy as
`model/milestone-000020000.ckpt`. Its SHA and complete state are independently
rechecked in `output/coshmdm_statsfix_20k_preserved_checkpoint_audit.json`.
Both live and terminal resume provenance bind the audit and checkpoint itself;
the final report re-hashes the 2.7 GB checkpoint and enforces its fixed SHA.
Training continued through logged update 20,100 with no full-state resume,
segmentation fault, or trainer-finished marker. Its 70 completed epoch
boundaries (`0..69`) were contiguous and duplicate-free, and every recorded
discard value was exactly one. The rolling evidence is
`output/coshmdm_resume_provenance_live.json` and
`output/coshmdm_statsfix_startup_audit.json`; these counts are explicitly
nonterminal and must be replaced by the capped terminal audit.

After the trainer stops, `coshmdm-post-cap-v1.service` freezes one separate
544-sample Inter-X validation request (seed 500123, batch 32), evaluates every
completed 20k milestone at or below 200k with DDIM50/eta0/CFG3.5, and selects
minimum official-evaluator validation FID.  The test split is never used for
checkpoint selection.  Only the selected checkpoint proceeds to the frozen
20-replication common test protocol, under label
`coshmdm_interx_retrained_2daycap_seed123`.  Selection provenance is written to
`output/coshmdm_interx_val_selection.json`; the frozen request and hashes live in
`output/coshmdm_interx_val_requests/`.

The user-capped run ended after 105,860 logged optimizer updates; the last
stable milestone is 100k and validation selected 40k.  The five official
validation FIDs at 20k/40k/60k/80k/100k are respectively
`46.100842/44.948572/90.454256/61.526379/104.913370`.  Frozen TensorBoard
windows and all five checkpoints were audited in
`output/coshmdm_training_trajectory_diagnosis.json`: trailing-1k training loss
continues to decrease (`0.093849` to `0.065959`), all 305,442,310 milestone
weight values are finite, and maximum absolute weight remains `8.359298`.
This contradicts a NaN/exploding-loss explanation and instead indicates that
the released training objective is becoming less aligned with held-out
official-evaluator quality after 40k.  The historical run did not log gradient
norms, so they cannot be reconstructed retrospectively.

The selected 40k CoShMDM HDF5 and the formal HHC HDF5 have now both been
rescored for all 20 frozen replications with the same current scorer on CPU.
CoShMDM obtains FID `41.444770 +/- 0.479421`, Matching
`9.050552 +/- 0.037148`, Top-1/2/3
`0.120106/0.201975/0.263296`, Diversity `8.138122 +/- 0.074804`, and
Multimodality `6.122696 +/- 0.254564`; formal HHC contact obtains FID
`0.545132 +/- 0.024679`.  In the paired comparison HHC wins all 20
replications for FID, Matching, and each R-Precision rank, while CoShMDM wins
all 20 for Multimodality.  Every one of these method-by-metric differences
remains significant after one Holm-Bonferroni family correction.  The
machine-readable evidence is
`output/fair_eval/{hhc_formal_raw12288_repaired_mgt_common50_contact,coshmdm_interx_retrained_2daycap_seed123}_cpu_metric_audit.json` and
`output/hhc_vs_coshmdm_cpu_evaluator_paired.json`.  This is a complete
same-request/same-evaluator comparison of the already generated files, but it
remains a provisional pair until the final HHC no-timing reference and all
other methods enter the single global correction family.

The final fair reference, HHC no-timing, has also been rescored on CPU over
all 20 replications.  Its FID/Matching/Top-1/2/3 are
`0.555712/3.712078/0.422022/0.602860/0.710407`.  Against the same checkpoint
with frozen request-oracle timing, FID improves by only `0.010580` and is not
significant (bootstrap interval crosses zero; Holm p=`0.6925`).  Matching
improves by `0.019208` (Holm p=`0.03834`) and Top-2 by `0.008343` (Holm
p=`0.002779`); the other ordinary and multimodality metrics are not
significant.  The no-timing integrity audit and paired evidence are
`output/fair_eval/hhc_formal_raw12288_repaired_mgt_common50_no_contact_cpu_metric_audit.json`
and `output/hhc_contact_timing_cpu_evaluator_paired.json`.  This comparison is
an oracle-input ablation and is not merged into the baseline method family.

For diagnosis only, frozen replication 0 is also projected through CoShMDM's
observable 262-D representation and restored to the same official 672-D
evaluator.  GT-to-model FID is `38.996055`, GT-to-representation-oracle FID is
`15.863267`, and representation-oracle-to-model FID is `11.833752`.  These
FIDs are not additive and the oracle is neither a body-controlled result nor a
reported comparison row.  It demonstrates that both irreversible
representation loss and learned-generator mismatch contribute to the poor
formal result.  Evidence is frozen in
`output/coshmdm_embedding_gap_diagnosis.json` and
`output/coshmdm_representation_oracle/ordinary_fid.json`.
TiMotion's separately implemented 262-to-672 exporter is runtime-tested against
this same CoShMDM adapter in eight float32/float64, pair/flat and
shape-free/shape-specific cases. Every output is bitwise identical with zero
maximum error, so the two public 262-D baselines do not receive different
geometry adapters. The cross-implementation evidence is
`output/timotion_cosh_adapter_equivalence_audit.json`.

The active `interx_global` mean/std were independently regenerated from the
complete official 9,110-sample train split with the same mirror and person-order
augmentation.  Both arrays are bitwise identical to the training files; the
regeneration manifest is
`/mnt/8T-HDD/wenning/coshmdm-code/output/interx_stats_recompute_audit/interx_global_stats_manifest.json`
and is hashed into the later fair-evaluation provenance.

From the HHC repository, independently recheck the frozen training source,
three split files, train-only normalization statistics, Git identity and the
2,000-epoch/568,000-update arithmetic with:

```bash
conda run --no-capture-output -n New_MMM python -u \
  tools/audit_coshmdm_training_contract.py \
  --output output/coshmdm_training_contract_audit.json

conda run --no-capture-output -n New_MMM python -u \
  tools/audit_coshmdm_representation_adapter.py \
  --output output/coshmdm_representation_adapter_audit.json
```

The GPU-0 completion queue reruns this audit before it accepts the final
checkpoint.  The final report requires both the 13 independent training-source
hashes and the 1,696-sample, 59-skeleton adapter idempotence audit.  The latter
also proves that every non-root body rotation is preserved exactly across all
requested frames; it separately records the intentionally unmodelled initial
global root transform and 33 hand-joint rotations.

## TIMotion public Transformer+LPA on the same Inter-X split

The fixed public revision exposes only the Transformer interaction block; its
Mamba/RWKV imports are disabled and no executable Inter-X loader or checkpoint
is released.  The fair port therefore reports **public TIMotion+Transformer+LPA**
and never labels it as the paper's best RWKV variant.  `data/InterX` links to the
same official-HDF5-derived 262-D data used by the CoShMDM port, while
`TIMOTION_STATS_PREFIX=interx_global` selects the same audited train-only mean
and standard deviation.  A read-only audit finds 9,110 valid training IDs,
18,220 mirrored entries, and a 44-entry epoch-boundary remainder.

This does not ignore a compatible author checkpoint.  The upstream README does
link two pretrained models, but the same frozen revision documents only the
InterHuman/InterGen data source, its original dataset config contains only
`interhuman{,_val,_test}`, and the tracked tree contains neither an Inter-X
path nor an embedded checkpoint.  The evaluator downloader is likewise
explicitly labelled InterHuman.  The contract audit hashes the three upstream
Git blobs and records `documented_interx_checkpoint_available=false`; therefore
the released links cannot be relabelled as an official-split Inter-X model, and
the public Transformer+LPA architecture must be trained on the frozen Inter-X
train split for this comparison.

The formal run was launched on 2026-08-31 from
`/mnt/8T-HDD/wenning/TIMotion`. The active replacement is online W&B run
`kdz5s3ra`:

```bash
TIMOTION_STATS_PREFIX=interx_global \
conda run --no-capture-output -n New_MMM python -u \
  tools/train_interx_fair.py --gpu 0 --seed 123 \
  --batch-size 32 --accumulate-grad-batches 8 \
  --microbatches-per-epoch 568 --epoch 1500 --lpa --no-progress-bar
```

This preserves the paper/released hyperparameters: 5 layers, 16 heads,
512-dimensional motion embeddings, AdamW (`1e-4`, weight decay `2e-5`, betas
`0.9/0.999`), ten warm-up epochs, cosine schedule, 1,000 diffusion steps,
10% condition dropout, DDIM50 with eta zero, and CFG 3.5.  The released scheduler
hard-codes a 2,000-epoch cosine horizon even though the paper trains for 1,500
epochs; this source-derived discrepancy is retained and recorded.  Eight local
micro-batches form one global batch of 256.  Exactly 568 micro-batches are used
per epoch, giving 71 updates/18,176 examples per epoch and 106,500 total updates,
which matches the complete batches of the paper's eight-GPU exposure.  The
manifest hashes the code, split, stats and all settings before model creation.
The public `train.py` default of 1,900 epochs is not used: paper Sec. 4.1.1
explicitly specifies 1,500 epochs for all diffusion denoisers, while the README
does not override the default on its `--LPA` example.  The discrepancy is frozen
in the manifest rather than silently choosing the longer budget.  Before the GPU
smoke, `tools/queue_timotion_fair_after_cosh.sh` creates a separate contract
preflight and runs `tools/audit_timotion_training_contract.py`; after the formal
run it repeats the audit with full content rehashing.  `--resume auto` restores
only complete epoch-boundary checkpoints and rejects any code/data contract
drift.  The conversion contains one sub-15-frame official test ID but no such
training IDs, so the 9,110-ID/18,220-entry training exposure is unchanged.

Before the formal launch, the first one-update smoke exposed non-finite
generations. The cause was isolated to the released relative-orientation (RO)
loss: it normalized hip-to-hip vectors on every padded frame before applying
the validity mask. Both padded people have coincident hips, so a padding-only
zero vector produced `0/0`, and the later `NaN*0` still contaminated the loss.
The fair port substitutes a fixed x-axis only outside the validity mask before
normalization. An exhaustive audit proves that valid-frame predictions,
targets and RO values are bitwise unchanged, while the bad smoke reproduces
18,324,708 non-finite exported motion values and the stabilized smoke is fully
finite. The machine-readable evidence is
`output/timotion_padding_ro_stabilization_audit.json`; this must be disclosed
as a numerical padding stabilization, not an objective-function modification.

The replacement 50-update smoke is online W&B run `oi4wa5qn`. Its remote
history audit passed and recorded, at logical optimizer step 50, gradient L2
`5145.385742`, gradient max-abs `496.141113`, weight L2 `156.836288`, weight
max-abs `1.00004017`, zero non-finite gradients/weights, `96.6893` samples/s,
and `10.5072 GiB` peak GPU allocation. Its small validation smoke produced
loss `3.68893647` and official FID `141.9958536`; these are launch diagnostics,
not a comparison result. The formal run uses the same audited telemetry and
fails closed if its online W&B contract or finite-value checks fail.

The first formal W&B run `jqh616am` was manually interrupted during a warm-up
source audit before any checkpoint was written. A subsequent source-order
audit confirmed that the executed runner had already matched the released
global gradient clip norm `0.5`: telemetry is intentionally collected before
the clip, and the clip precedes `optimizer.step`. The interruption was thus an
operator audit error rather than an executed-contract defect. The remote run
is marked failed and `result_eligible=false`; the exclusion record is
`output/timotion_formal_jqh616am_exclusion.json`. Run `kdz5s3ra` restarts the
same audited contract from optimizer step 0 and is the only formal candidate.
The separate runtime arithmetic audit
`output/timotion_gradient_accumulation_audit.json` verifies that eight
mean-reduced 32-example microbatches divided by eight reproduce a 256-example
global-batch gradient: pre/post-clip gradient differences are at most
`1.49e-17`, and the subsequent AdamW weights are bitwise identical.
The formal replacement then passed its complete ten-epoch warm-up audit in
`output/timotion_warmup_stability_audit.json`: all 14 expected 50-update
telemetry events are present, gradient/weight non-finite counts remain zero,
and global weight L2/max-abs span only `0.0462%`/`0.3251%` relative to their
first observations. Validation loss at the tenth completed epoch is
`5.22971392`. The large individual-batch velocity-loss range is retained in
W&B as diagnostic evidence, while the bounded weights and mandatory released
global-norm clip justify continuing the formal run.  A refresh of this audit
keeps `metrics` restricted to those first 14 warm-up events and writes later
telemetry separately as `full_run_metrics`; normal long-run weight drift can
therefore no longer be mislabeled as warm-up instability.
The 2026-09-01 live refresh reaches sparse-telemetry optimizer step 21,400,
dense loss/LR step 21,412, and completed epoch 301.
It reads the dense loss/LR event group and sparse 50-update telemetry group
separately, and all ten required diffusion-training metrics are present and
finite.  The latest step-21,400 pre-clip gradient L2/max-abs observations are
`5.771327/0.454889`, followed by the released global-norm `0.5` clip; the
current weight L2/max-abs are `190.798294/1.206591`, and both non-finite
streams remain zero. Dense step 21,412 loss is a finite `0.089208`. The
earlier finite step-17,500 velocity excursion remains frozen as a separate
diagnostic (`loss=0.213035`, `VEL_A+VEL_B=0.119416`, next update `0.128459`)
and did not persist. Its runtime validation section also proves that W&B received 301 finite
`val/loss` events and all three finite official semantic events for FID, all three
R-Precision values, Matching and Diversity; the values exactly match the
independently hashed step-7,100/14,200/21,300 validation JSON files.
The 710-update warm-up window contains exactly the 14 telemetry events at
steps 50 through 700; the auditor no longer rounds up and incorrectly includes
the post-warm-up step-750 event.

A later transient excursion is frozen separately in
`output/timotion_transient_velocity_spike_audit.json`.  At completed epoch 66,
mean train loss rose to `0.743690` and validation loss to `1.248939`; the
increase is entirely explained by the two released velocity terms
(`VEL_A+VEL_B` explains `100.16%` of the excess because the remaining terms
slightly decreased), while the simple diffusion loss remained `0.107253`.
There were no non-finite gradients or weights, global weight norms remained
smooth, and by epoch 69 train/validation loss had returned to
`0.173800`/`0.208269`.  The refreshed diagnosis reaches complete epoch 304,
where train/validation loss is `0.080676`/`0.093935` and the two velocity
terms are `0.001942/0.001935`; the excursion therefore did not recur around
epoch 301. Because W&B has no transaction spanning several per-metric history
queries, the live auditor requires identical `_step` values over the full
common prefix and permits only a bounded trailing suffix while the remote
state is `running`; the current 21,623-event prefix is exact and the
query-time suffix skew is six events. A finished run still requires exact
counts. This is explicitly a post-hoc operational diagnosis,
not a checkpoint-selection rule or authorization to alter the released loss;
the first official semantic checkpoint comparison remains epoch 100.
The apparent `1.0999321479e-4` learning-rate value at zero-based epoch 10 is
also source-expected rather than a port-side scheduler error. The released
`CosineWarmupScheduler` uses the inclusive condition `epoch <= warmup`, so it
applies an `11/10` multiplier once and then returns to the cosine branch at
`9.9992536297e-5` in epoch 11. The formal W&B transitions match the released
formula exactly at every audited boundary; the source/hash/runtime proof is
`output/timotion_scheduler_boundary_audit.json`. The fair port deliberately
preserves this released behavior instead of silently correcting it.
The first three validation-only semantic checkpoints are now complete.  At
optimizer step 7,100 (epoch 100), FID/Matching/Top-1 are
`78.6949127/10.8952579/0.0588235`; at step 14,200 (epoch 200), they are
`68.6293408/10.4777746/0.0533088`. At step 21,300 (epoch 300), the same
metrics are `76.2587084/10.7737961/0.0735294`: Top-1 improves but FID
regresses, so the validation-only selector continues to retain the
step-14,200 minimum-FID checkpoint. The absolute quality remains far behind
HHC/InterMask.
Epoch 400 later improved FID to `55.234978`, while epoch 500 regressed to
`90.508970` despite finite, stable train/validation losses and gradient/weight
telemetry.  A representation audit then confirmed a protocol bug in the
TiMotion-only shape-specific root fit: prepared Inter-X rotations use standard
row-concatenated 6D, but the exporter imported TiMotion's interleaved-index
decoder.  Even identity 6D `[1,0,0,0,1,0]` becomes a rank-deficient matrix
(max identity and orthogonality errors both `1.0`). Evidence is frozen in
`output/timotion_rotation6d_adapter_contract_audit.json`.  The earlier
TiMotion/CoSh equivalence audit was a false pass because both repositories use
the top-level package name `utils`; loading them in one interpreter made the
second adapter reuse the first adapter's cached rotation module.  Isolated
dependency loading now exposes shape-specific RMSE `0.09436--0.09734` and
max-absolute differences `1.87--1.94` in
`output/timotion_cosh_adapter_equivalence_audit_isolated.json`.  Therefore all
current TiMotion semantic FIDs are diagnostic only.  The active frozen run is
not modified; after it finishes, epoch 400/500 are regenerated with matched
seeds through
`tools/export_timotion_official_interx_rotationfix_diagnostic.py`, rescored,
and only then may a corrected adapter be preregistered for final selection.
The proposed one-line convention repair itself is already isolated and
bitwise-equal to CoSh for all eight dtype/layout/root-path cases in
`output/timotion_cosh_adapter_equivalence_audit_rotationfix.json`; this proves
the code-level repair but not yet its checkpoint-level FID effect.
The completed CoShMDM run does not require regeneration: its frozen repository
revision kept `utils/rotation_conversions.py` clean, the committed/current
SHA-256 is the standard implementation hash `ba21bc55...`, identity roundtrip
error is zero, and the adapter snapshot is content-addressed. The supplemental
proof is `output/coshmdm_rotation_dependency_provenance_audit.json`.
These are early validation diagnostics only; the public 1,500-epoch training
budget and validation-only selector remain unchanged.
After update 106,500, the result remains blocked until
`output/timotion_interx_formal_wandb_history_audit.json` verifies the finished
remote run, every required train/validation metric, zero non-finite
gradients/weights, and semantic validation at the final step.
This run started on 2026-08-31 and is audited with the run-ID-bound
`diffusion_regression_20260831_launch` profile: it requires the global
gradient/weight L2 and max-absolute streams, finite counters, throughput,
learning rate, losses, and all semantic validation metrics that were
instrumented at launch.  The module-level, gradient/weight-ratio, and element-
count expansion became effective on 2026-09-02 and is mandatory for every new
run after that date; it is not retroactively used as a reason to retrain this
otherwise fair third-party baseline.

After training, run the same frozen 20-replication protocol:

```bash
bash tools/queue_timotion_post_training.sh
```

The post-training entrypoint first proves that the full 106,500-update state is
complete and validates the online W&B history. It then runs
`tools/finalize_timotion_validation_history.py`, which requires all 15 expected
validation epochs, exact `epoch*71` optimizer steps, 544 held-out samples per
epoch, and byte-valid checkpoint/generation/metric artifacts before changing
the history status from `running` to `complete`. The selector accepts only this
explicitly finalized history bound to the completed training manifest, then
selects among all 15 100-epoch official-validation checkpoints by minimum
held-out FID.  The final
full-state `last.ckpt` remains separately bound for optimizer/training-state
integrity; test generation uses only
`output/timotion_interx_val_selection.json`'s selected weights.  Thus TiMotion
uses the same validation-only checkpoint-selection principle as HHC-M and
CoShMDM, without consulting any Inter-X test metric.

The exporter groups requests by exact length and retains the author DDIM50/CFG
sampler without GT first-frame conditioning. Its 262-D output must be mapped to
the official 672-D evaluator input with the same standard-6D body/root
conversion as CoShMDM; fingers are neutral because the public representation
cannot generate them. The frozen active exporter currently violates this root
conversion contract, so its outputs are ineligible for the final table until
the isolated matched-seed repair above passes. After repair, report only the
frozen official full-space Inter-X metrics; no projected-GT score is added.

## Frozen official evaluation

First create all 20 request files with the command in
`docs/interx_protocol_audit.md`.  For each replication, export CoShMDM main and
MM generations from its repository:

```bash
COSHMDM_STATS_PREFIX=interx_global conda run -n New_MMM python \
  tools/export_official_interx.py --checkpoint CHECKPOINT \
  --requests REQUEST.npz --output MAIN.h5 --gpu 0 --repeats 1 --seed SEED

COSHMDM_STATS_PREFIX=interx_global conda run -n New_MMM python \
  tools/export_official_interx.py --checkpoint CHECKPOINT \
  --requests REQUEST.npz --indices-key mm_indices \
  --output MM.h5 --gpu 0 --repeats 30 --seed MM_SEED
```

Then score from HHC with the common evaluator:

```bash
conda run -n New_MMM python tools/score_fair_interx_generations.py \
  --requests REQUEST.npz --main-generations MAIN.h5 \
  --mm-generations MM.h5 --output METRICS.json --device cuda:0
```

HHC uses the same HDF5 contract via `tools/export_hhc_official_interx.py`.
Run it once with `--contact-guidance --hhc-contact-timing-source request_oracle`
and once with
`--no-contact-guidance --hhc-contact-timing-source none` for the paired
HHC-M/contact-guidance rows. The no-guidance path refuses an oracle source, so
its request and HDF5 provenance both prove that test timing was not consumed.
After the 300k run is complete, `tools/select_best_mgt_checkpoint.py` requires
the final latest-step artifact and selects only the minimum fixed-validation
FID among Lightning's retained `model-*` checkpoints.  It records every
candidate, the selected checkpoint SHA-256 and training-manifest SHA-256; no
test metric participates in selection.  Selection also loads the terminal
300k checkpoint, verifies its actual global step, loop/scheduler/optimizer
state and every floating model/optimizer tensor, and embeds that finite-state
audit plus the terminal checkpoint hash in the selection manifest.  Thus a
file merely named `latest-300000.ckpt` is not accepted as proof of completion.

InterMask uses the same request and scoring files via its
`tools_export_official_interx.py`; method-specific defaults are 20 sampling
steps, CFG 2, top-k threshold 0.9 and temperature 1.0.  Its default
`--length-policy exact` uses enough 4-frame tokens to cover the frozen request
and crops back to that exact length (notably 150 rather than the native 148).
Use `--length-policy author-floor` only for the separate native-code audit.

CoShMDM's released sampler accepts only one temporal size per batch.  Its fair
exporter therefore groups frozen requests by identical length and generates
each group directly at that declared length; it does not generate every sample
inside a 150-frame context and crop afterward.  The HDF5 attribute
`generation_length_policy=exact_request_length_grouped` makes the policy
machine-checkable and prevents an older long-context export from being resumed
or scored as the fair run.

Interact2AR uses its official repository, locked `uv` environment and released
mixed-memory checkpoint, but is scored by this same frozen Inter-X evaluator.
The author's README evaluation supplies a GT first frame; the fair main table
must not do that because HHC, InterMask and CoShMDM do not receive one.  The
adapter therefore records `first_frame_conditioning=none`, retains author CFG
2.5 and autoregressive ancestral 10-step sampling, and groups requests by exact
length.  Run all 20 replications after a batch-memory probe:

```bash
conda run --no-capture-output -n New_MMM python -u \
  tools/run_fair_interx_method.py \
  --method interact2ar \
  --label interact2ar_official_fair_no_gt_first_frame \
  --checkpoint /mnt/8T-HDD/wenning/Interact2Ar/checkpoints/interact2ar_mixed_memory.ckpt \
  --gpu 1 --batch-size 32
```

The driver launches generation with
`/mnt/8T-HDD/wenning/Interact2Ar/.venv/bin/python` and scoring with `New_MMM`.
Its run manifest hashes the author checkpoint manifest, split, `uv.lock`, model
source, common requests/evaluator and relevant repository launch state.  Every
implementation file is copied at launch to a content-addressed immutable
snapshot; repository revision/diff records remain historical context, while
the snapshot hashes are the executable-byte integrity gate.  Data/SMPL-X
symlinks from MMM are expected to make the author repository appear dirty; its
tracked launch diff is empty.  A one-sample no-first-frame execution smoke is recorded
under `output/fair_eval/interact2ar_official_fair_smoke_v2/`; it is not a metric.
See `docs/baseline_availability_audit.md` for exact hashes and the distinction
between the executable public TiMotion Transformer port and the unreleased
Mamba/RWKV Inter-X variants.  The strict TiMotion row remains pending until its
from-scratch Inter-X retraining finishes.

After all 20 JSON files for one method are present:

```bash
conda run -n New_MMM python tools/aggregate_fair_interx_metrics.py \
  --inputs 'output/fair_eval/METHOD/replication_*.json' \
  --output output/fair_eval/METHOD/summary.json
```

The aggregator accepts that broad glob but selects only exact
`replication_NN.json` metric files, excluding physical and non-metric adapter
audits.  It verifies IDs 00--19, independently hashes 20 distinct frozen
request files, and requires one checkpoint/sampler identity across all rows.
The sampler-identity projection includes HHC's clarified maximum active-stage,
CFG-forward and resolved-channel-skip fields; omitting those fields previously
made the final report disagree with metadata that was already present in every
HDF5 and replication JSON.  Re-aggregating the two HHC rows changed only this
provenance projection: their ordinary metric values remain unchanged.  For
CoShMDM/TiMotion, the skeleton bank path is accepted only when the recorded
directory and configured directory resolve to the same existing directory;
the independently required bank-content SHA-256 remains the identity key.
This handles the repository's HHC-to-MMM skeleton symlink without accepting a
different bank.  The regression tests reject a different canonical directory.
Old pre-provenance JSON files are accepted only with the explicitly labelled
`--allow-legacy-provenance` audit flag; they are not valid inputs to a final
strict result table.

The common driver also writes
`replication_NN_generation_timing.json` and, after 20 replications,
`generation_timing_summary.json`.  The timing boundary is the complete exporter
subprocess (model load, sampling and HDF5 write), excluding evaluator/scorers;
mean, 95% CI, median and P95 are reported for both wall time and amortized
seconds/sample.  This common boundary is used for every baseline.  A resumed
artifact whose original timing sidecar is unavailable must be labelled as a
resume-only timing audit rather than silently compared with fresh runs.

Do not compute a projected-GT, representation-controlled, body-controlled, or
hand-neutralized FID.  The required adapter audit above already records which
fields the public 262-D representation preserves or omits.  Every quantitative
method comparison uses the same frozen official full-space Inter-X evaluator.

Run the LLM-free physical protocol on replication 0 main generations:

```bash
conda run -n New_MMM python tools/score_fair_interx_physical.py \
  --requests output/fair_interx_requests/replication_00.npz \
  --generations MAIN.h5 --output PHYSICAL.json --device cuda:0
```

It uses the frozen Inter-X contact schedule and subject-specific skeletons and
reports a predeclared 5/10/15 cm threshold sweep.  The paired GT reference is
`output/fair_interx_ground_truth/replication_00.h5`.  The main-table joint
contact row is fixed at 10 cm and grounded-foot scoring at a 5 cm height
threshold.  The final report rejects any method whose physical artifact does
not declare this same `interx_official_v1_physical_joint_v1` LLM-free contract;
mesh penetration remains a separately reported metric.  The fair runner also
checks these fields before reusing a cached replication-0 physical report, so
a stale report with another threshold is rescored even when its request and
generation hashes still match.

The review timing ablation separates oracle annotations from deployable inputs.
First freeze the train/validation-only text predictor and score it once on the
held-out request (the score must not feed back into hyperparameter selection):

```bash
conda run --no-capture-output -n New_MMM python -u \
  tools/predict_interx_contact_timing_from_text.py \
  --requests output/fair_interx_requests/replication_00.npz \
  --output output/contact_timing/text_knn_replication_00.npz
conda run --no-capture-output -n New_MMM python -u \
  tools/score_contact_timing_predictions.py \
  --requests output/fair_interx_requests/replication_00.npz \
  --predictions output/contact_timing/text_knn_replication_00.npz \
  --output output/contact_timing/text_knn_replication_00_score.json
```

The selected validation setting is k=10/threshold=0.4; frozen-test frame-level
F1 is 0.60021.  Missing Inter-X contact files follow the dataset's original
zero-mask fallback and are counted in the manifest.  Once the final checkpoint
and the oracle/no-guidance strict runs exist, generate the other two conditions
and collect all four rows with:

```bash
conda run --no-capture-output -n New_MMM python -u \
  tools/run_hhc_contact_timing_ablation.py --device cuda:1
conda run --no-capture-output -n New_MMM python -u \
  tools/summarize_hhc_contact_timing_ablation.py
```

`request_oracle` reads test annotations, `text_predicted` reads only the
request-bound predictor, `full_valid_weak` uses an all-valid-frame mask with
lambda=0.25, and `none` has no contact bias.  This is a replication-0
sensitivity analysis; it does not replace the 20-replication oracle/none rows.

The earlier held-out shape/action split proposal is retained only as an
investigation backup and is not launched or reported.  The active shape
experiment keeps the official Inter-X split unchanged.  It samples new body
shapes for train-split motions, uses train-split human contact timing and the
paper optimizer to restore physically consistent contacts after the shape
change, and appends the optimized samples as training augmentation.  The
required comparison is no augmentation versus random-shape-only versus
optimized random-shape augmentation under the same training and official-test
protocol.  Action generalization is not part of the active experiment plan.

Before that final shape stage, run
`tools/run_interact2ar_joint_evaluator_suite.sh GPU_INDEX`. It first executes
the untouched Interact2AR README evaluation with its released GT-first-frame
conditioning and refuses continuation unless JointsFull FID is within the
predeclared diagnostic tolerance of the paper value. It then replays the
twenty frozen raw-GT crops and scores the unchanged unconditional HHC,
InterMask, Interact2AR, CoShMDM and TiMotion generations with JointsFull,
JointsBody and JointsHands. The resulting pre-shape gate is
`output/interact2ar_joint_evaluator_suite/pre_shape_summary.json`; it is
supplementary and must never be merged numerically into the legacy Inter-X
evaluator columns.
The raw-GT replay is fixed to the request-creation runtime:
`OMP_NUM_THREADS=8` and the CPU official evaluator. Both settings are
fail-closed. Omitting the thread pin produced a `1.1920929e-6` text-embedding
drift at replication 9; restoring it gives exact zero difference for all three
stored embedding arrays over all twenty replications, while every request field
also matches exactly. Do not replace this runtime match by relaxing the
`1e-6` tolerance.
The author result is re-authorized by
`audit_interact2ar_author_calibration.py`, which re-hashes its stdout and
paper/repository/data audit. Before full scoring, a numerical batch-level
parity test also verifies that caption sorting, motion-length sorting and the
inverse permutation in the frozen-request adapter reproduce the released
loader/wrapper embedding order.

Once every pre-shape artifact exists and a GPU is actually free, run only
`tools/run_original_mmm_shape_pilot25.sh GPU_INDEX`.  This explicit foreground
entry performs 25 original-MMM optimizations, records per-sample wall time,
builds the mismatch/skating gate, and then stops.  It cannot launch full
augmentation or training.  The gate must improve timing-weighted mismatch,
preserve skating within `1e-6`, and project the 5,868 timing-positive records
to at most 24 single-GPU hours.  Only after that evidence passes may a new full
launcher be frozen.  The old
`tools/run_interx_shape_augmentation_nearest_last_stage.sh` is archived and
exits immediately.

The complete random-only materialization
`output/interx_shape_aug_random_seed123_v2` is reused because its 6,238×2 arrays,
donor beta/gender, crops and hashes pass
`output/interx_shape_aug_random_seed123_v2_completion_audit.json`; it never
called an optimizer.  The full optimized arm processes 5,868 timing-positive
records and deterministically falls back to that random-only arm for the 370
zero-timing records.  Both subsequent 300k training arms must use online W&B
and byte-exactly reuse the historical MMM token/contact cache for original
train/validation samples.  After their frozen 20-replication generation, run
the Interact2AR `post_shape` stage and aggregate five baselines plus two shape
models into `output/interact2ar_joint_evaluator_suite/summary.json`; only then
may the final review bundle be built.

After both GPU branches and the shape-augmentation runs finish,
`tools/build_review_results_report.py` verifies and collects the six strict
20-replication method summaries, 435-pair MM audits, replication-0 physical
metrics, common-boundary timing summaries and shape-augmentation results into
`output/review_results_report.json` and `docs/review_results_report.md`.  It
also requires the frozen request manifest, capacity audit, final VQ state and
reconstruction, selected completed HHC-M run, and final CoShMDM/TiMotion state
audits.  Paired method/oracle-timing tests, the timing and sampling
sensitivities, train/val-only contact-timing predictor score, and the
CoShMDM/TiMotion geometric adapter audit are required rather than optional
report decorations. No secondary projected or body-controlled score is
required. It fails closed when any required artifact is
absent; the queued final report does not manufacture placeholders for
unfinished methods.
The current pre-final audit has passed manifest, implementation-snapshot,
sampler, MM and physical/timing validation for the five completed method rows;
it stops at the missing TiMotion final run manifest, as intended while that
training is still active.  No provisional TiMotion placeholder is created.

The capacity table is rebuilt by `tools/audit_model_capacity.py`, not copied
from a hand-written JSON.  It instantiates HHC, InterMask and CoShMDM in three
isolated processes, loads the exact evaluation checkpoints, counts unique
`Parameter` objects (including a shared-VQ identity check for HHC), and hashes
every constructor/config source.  The formal invocation binds HHC to
`output/hhc_formal_raw12288_repaired_mgt_selection.json` and CoShMDM to its audited 568k
`last.ckpt`; the final report re-hashes every recorded source and checkpoint.

Before building that report, run the paired method analysis on the same 20
request replications.  The fair reference is HHC without GT contact timing;
the oracle-timing effect is a separate input-information ablation:

```bash
conda run --no-capture-output -n New_MMM python -u \
  tools/compare_fair_interx_methods_paired.py \
  --reference hhc_no_timing=output/fair_eval/hhc_formal_raw12288_repaired_mgt_common50_no_contact \
  --comparators \
    intermask=output/fair_eval/intermask_official_exact_common50 \
    interact2ar=output/fair_eval/interact2ar_official_fair_no_gt_first_frame \
    coshmdm=output/fair_eval/coshmdm_interx_retrained_2daycap_seed123 \
    timotion=output/fair_eval/timotion_transformer_lpa_interx_retrained_fair \
  --output output/fair_eval/paired_fair_method_comparison.json
```

The tool byte-verifies each paired request, uses paired bootstrap intervals and
two-sided Wilcoxon tests, and applies one Holm-Bonferroni correction over the
method×metric family.  Diversity is compared as absolute error from the same
replication's GT diversity, rather than incorrectly treating larger raw
Diversity as always better.

The fair-HDF5-to-SMPL-X converter remains part of the read-only common mesh
evaluation path.  It must not feed the archived test-time contact optimizer.
Its conversion manifest still binds request, HDF5, caption, length, beta,
gender and repeat so the six-method mesh table is paired and reproducible.

For the review's multimodality variance audit, compute all 435 pairs among the
30 generations for each prompt, then aggregate them against the official
10-pair estimator:

```bash
conda run -n New_MMM python tools/analyze_fair_interx_multimodality.py \
  --requests REQUEST.npz --mm-generations MM.h5 \
  --output MM_AUDIT.json --device cpu

conda run -n New_MMM python \
  tools/aggregate_fair_interx_multimodality_audits.py \
  --metrics 'output/fair_eval/METHOD/replication_[0-9][0-9].json' \
  --audits 'output/fair_eval/METHOD/replication_[0-9][0-9]_multimodality_audit.json' \
  --output output/fair_eval/METHOD/multimodality_audit_summary.json
```

Historical audit JSONs that predate MM-HDF5 digests can be re-aggregated only
with `--allow-legacy-provenance`; final driver runs omit that flag and require
matching hashes in the official metric and 435-pair audit reports.

The restartable end-to-end driver creates main/MM HDF5 files, rejects partial
repeat sets during scoring, runs the common evaluator, computes the review's
all-435-pair multimodality audit by default, and aggregates both reports over
20 replications.  For example:

The final report independently closes the checkpoint chain rather than trusting
the launch command alone.  Both HHC timing variants must hash to the exact
validation-FID-selected checkpoint; CoShMDM and TiMotion must hash to their
568k/106.5k state-audited final checkpoints.  It also pins InterMask's released
Inter-X Transformer/VQ hashes and Interact2AR's 350k mixed-memory checkpoint.
For retrained HHC and CoShMDM, the Git revision and tracked-diff hash captured
at generation must equal the training-launch manifest, so later source changes
cannot be silently evaluated as the original training run.

```bash
# Author-native 20-step sampler with exact-length adaptation.  This is a
# historical/reference audit and is not an input to the common-50 table.
conda run -n New_MMM python tools/run_fair_interx_method.py \
  --method intermask --label intermask_official_exact --gpu 1 \
  --sampling-steps 20

# Requested fair-table sampler budget.  Keep this label and its artifacts
# separate from the native-20 audit above.
conda run -n New_MMM python tools/run_fair_interx_method.py \
  --method intermask --label intermask_official_exact_common50 --gpu 1 \
  --sampling-steps 50

conda run -n New_MMM python tools/run_fair_interx_method.py \
  --method hhc \
  --label hhc_formal_raw12288_repaired_mgt_common50_contact --gpu 1 \
  --checkpoint HHC_MGT.ckpt \
  --vq-checkpoint checkpoints/vq_precleanup_raw12288/net_175000.pth \
  --hhc-vq-contract precleanup_raw12288 --hhc-mgt-contract current \
  --selection-manifest output/hhc_formal_raw12288_repaired_mgt_selection.json \
  --sampling-steps 50 --temperature 0.5 --cfg-scale 3.5 \
  --contact-guidance-scale 1.0 --hhc-contact-timing-source request_oracle \
  --contact-guidance

conda run -n New_MMM python tools/run_fair_interx_method.py \
  --method coshmdm --label coshmdm_interx --gpu 0 \
  --checkpoint COSHMDM.ckpt
```

Never refresh an old run manifest with a newer driver: that would falsely bind
old samples to code that did not generate them.  If a hash-bound implementation
file is still byte-identical, freeze it without rerunning samples using:

```bash
conda run -n New_MMM python \
  tools/freeze_fair_manifest_implementation_sources.py \
  output/fair_eval/METHOD/run_manifest.json
```

The freezer refuses any source whose current SHA-256 differs from the recorded
generation hash.  For the completed common-50 InterMask and Interact2AR runs,
the only drifted file was the orchestration driver; its exact 30,087-byte
version was recovered from the append-only execution record and verified
against the manifest SHA-256 `0880c47b...46fcef` before freezing.  All model,
exporter, scorer, evaluator and request hashes were independently unchanged, so
no baseline generation was rerun or relabelled.

Every exporter seeds repeat `r` with `base_seed + r` and records that strategy
in the HDF5.  `--resume` skips only fully flushed repeats; the scorer refuses a
file with any incomplete repeat.  Resume additionally checks the checkpoint,
request file, seed, length policy and every method-specific sampling parameter;
a same-shaped output from a different experiment is rejected instead of being
silently reused.  CoShMDM also binds batch size because its exact-length grouping
changes the random-number assignment across batches.  New HHC and CoShMDM
generation files additionally bind checkpoint SHA-256, so overwriting a model
at the same path cannot make stale generations look current.  Existing
InterMask files predate this field but are bound to both official checkpoint
hashes by their frozen run manifest.
Every newly scored replication also records SHA-256 for the complete main and
MM HDF5 files; strict aggregation re-hashes both artifacts and rejects any
post-score replacement or silent corruption.  Pre-hash reports require the
explicit legacy-provenance flag and are not final-table inputs.
The restart driver performs the same digest check before reusing semantic,
physical, or exhaustive-MM reports; a changed artifact is rescored rather than
silently accepted by timestamp alone.

## Current preserved artifacts

- The two VQ files copied from MMM, `checkpoints/vq/net_175000.pth` and
  `checkpoints/vq/net_best_fid.pth`, were inspected tensor-by-tensor.  Both use
  three 12,288-entry codebooks with latent widths 96/96/64.  Their matching
  historical run used 300k updates and weight decay `2e-5`; they are not the
  paper's stated 8,096-code, 150k, `1e-6` configuration.  Per the user's final
  tokenizer decision, the byte-identical 175k file is nevertheless the active
  formal VQ under the explicitly disclosed raw-12288 exception; it is not
  presented as an exact paper-tokenizer reproduction.  The old training
  trajectory also remains evidence for the unpublished reconstruction-loss weights.
  The matching historical MGT W&B run is frozen in
  `output/historical_mmm_wandb_comparison.json`: its final validation FID was
  `0.180908`.  It shares the raw-12288 tokenizer with the active run, but used
  a 1024-D/16-head legacy architecture, different mask/objective, batch 64 and
  seed 114514; the active MGT is 512-D/8-head, batch 96 and seed 123.  The
  report therefore enforces `same_configuration=false` and treats the old
  metric as historical context rather than a fair baseline.
  Historical tracebacks bind the actual sampler to `sample_intra_cas`; this is
  not a parallel-vs-cascade difference.  The authoritative low-FID training is
  the original W&B run started 2026-02-13 01:55:22 UTC and completed
  2026-02-13 22:43:45 UTC, wholly inside one recovered Local History source
  interval.  Its 10k/20k/50k/200k/300k validation FIDs were respectively
  `0.697935/0.642393/0.244270/0.135719/0.180908`.  The 2026-02-17 entries are
  resumed/copied evaluation-history context and must not be cited as the source
  interval of that training.  The original run used the raw-12288 cache,
  1024-D/16-head legacy topology, flattened random-replacement corruption,
  all-valid-token loss, AdamW `(0.5,0.9)`, 150k MultiStepLR, seed 114514, and
  the historical CLIP/shape conditioning dropout.  The time-bracketed training
  source also fixes the token window at 38 (not the paper path's 50) and runs a
  full 53-batch test-set sanity validation before the first optimizer step.
  That historical validation neither isolates nor restores RNG, so its random
  draws affect the subsequent training stream; the exact-replay diagnostic
  reproduces this side effect deliberately.  These are historical diagnosis
  settings only: formal inference remains the user-fixed common-50 contract,
  and test metrics are never used to select the final paper model.  Current comparisons keep
  the user-fixed `sample_intra_cas/50/0.5/3.5/1.0` evaluator protocol constant;
  therefore the substantially worse corrected-MGT FID is treated as a training
  pipeline failure to repair, not as a sampler-parameter explanation.
  An independent initialization audit confirms exact equality of the recovered
  VQ/transformer construction RNG states and initial transformer weights, so
  initialization is no longer an unresolved explanation.  The repair gate is:
  reproduce the historical initialization FID, inspect a finite 5k checkpoint,
  approach `0.697935@10k` and `0.642393@20k`, then re-evaluate the same accepted
  checkpoint with the frozen common-50 evaluator.  Until those gates pass, no
  formal HHC, ablation, or shape-augmentation MGT training may start.  No stage
  computes or reports any secondary body-space FID.

The repaired historical MGT run is finalized only with
`tools/finalize_mgt_repair_v7.sh`. The script refuses to run while V7 is
active or without the atomic 300k checkpoint, compares every logged training
row and all 30 ordinary validation milestones to the frozen W&B reference,
checks the same-step 280k checkpoint state, regenerates replication 0 with the
common-50 protocol, and builds `output/mgt_repair_gate.json`. Formal HHC and
shape-augmentation queues remain fail-fast until that gate verifies.
If `score_fair_interx_main.py` changes, the finalizer now re-scores the same
frozen VQ reconstruction on CPU and runs
`tools/audit_vq_reconstruction_scorer_migration.py` before rebuilding the
gate.  The effective-length scorer refresh was checked this way: FID remains
exactly `0.29441722725218256`, and FID, Diversity, all three R-Precision
values, and Matching have zero old/new difference because this reconstruction
is exact-request-length.  The refreshed gate includes the migration audit as
a seventh content-bound evidence record; source drift is never waived.
The candidate side deliberately uses the validation checkpoint rather than the
single retained rolling checkpoint: historical 280k FID
`0.16047221422195435` ranks ninth among all 30 milestones, so the
`save_top_k=10` validation callback retains it through 300k.  Because V7 must
match every historical validation value exactly, this retention argument is
deterministic; the finalizer still fails closed if the actual 280k file is not
present.
After the historical replay and formal HHC have both released the training
entrypoint, the persistent GPU-1 chain reaches a separate root-tokenizer wiring
gate. `tools/apply_post_formal_shape_root_tokenizer_wiring.sh` first requires
both formal common-50 summaries, no live MGT process, and exact pre-wiring
source hashes, then applies the frozen, preflighted patch. The preimage hashes
are bound directly to the active formal training manifest:
`options/option_transformer.py=253537...60b` and
`train/train_t2i_trans.py=792355...eb0d`. An earlier queued value accidentally
used the post-patch options hash `d0f894...aa`; the gate was corrected before
execution and now has a regression test against both manifest entries. The chain reruns
`tools/audit_shape_root_tokenizer_wiring.py` every five minutes if the wiring
still does not pass and does not invoke the inner shape queue until it does.
This prevents changes to source bytes owned by either active MGT run; the inner
shape queue independently repeats the same fail-closed audit.
The exact 14-file V7 launch source set is independently frozen under the run's
`source/` directory by content hash. Two post-launch files are reconstructed
from seven timestamped local Codex `apply_patch` records; every reconstructed
byte must equal the immutable `training_manifest.json` hash. The selected seven
JSONL records are frozen separately, so the evidence does not depend on the
subsequently growing session log. `tools/recover_mgt_launch_sources_from_codex_history.py`
is idempotent: after materialization it validates the content-addressed set and
does not require the evolving worktree to remain equal to launch. The repair
gate binds the recovery JSON, selected patch records, training manifest, and
all 14 source snapshots and rehashes them before allowing formal training.
After the formal 300k run, its launcher first writes the strict selector and
derives the run directory from the selected training manifest. This avoids
binding audits to an older same-name interrupted directory after a restart;
the 20k state-pair and paper-contract audits still run before fair generation.
Before training, `tools/resolve_mgt_training_state.py` distinguishes a new,
uniquely resumable, or complete run. A resumable run uses
`--resume-training-ckpt ... --resume-in-place`; the trainer verifies the full
architecture/data/sampling contract and all 14 launch-source hashes before
loading optimizer/scheduler state. Ambiguous multiple runs or any drift fail
closed. A complete 300k run is never trained a second time.
At step 20k, training permanently preserves both
`prevalidation-000020000.ckpt` and `postvalidation-000020000.ckpt`
independently of the rolling and validation-ranked callbacks. The pre file uses
a same-filesystem hard link when possible. This is necessary because
`save_top_k=1` deletes `latest-000020000.ckpt` at the next rolling save and a
later top-k update may delete `model-000020000-*.ckpt`. The final report and
both shape-augmentation result arms require the permanent basenames and the
callback's pre/post completion markers before accepting the state-pair audit.
The detached continuation entrypoint is
`tools/queue_repaired_interx_end_to_end.sh`: it waits inside one non-tmux
orchestrator for the current historical-MGT and CoShMDM processes, then runs
the repair gate, formal HHC, review/shape experiments, remaining fair baseline
branch and final report. It does not allocate additional background terminals.
Before paired comparison, the final report queue reuses the completed HDF5
generations and re-scores only the older InterMask/Interact2AR metric JSONs
with the current official evaluator. `final_ordinary_only_metric_audit.json`
then binds all 120 final-method replication records and six summaries by hash
and requires every record to embed the same official evaluator options,
checkpoint, wrapper, option loader and scorer hashes. It rejects every
body-controlled FID/diversity field; no baseline model is retrained or
re-sampled for this cleanup.
The completed individual CPU audits are
`output/fair_eval/intermask_official_exact_common50_cpu_metric_audit.json` and
`output/fair_eval/interact2ar_official_fair_no_gt_first_frame_cpu_metric_audit.json`.
Their current-scorer FIDs are respectively `0.3742799127` and `2.4442555382`;
the corresponding Matching scores are `3.7000037789` and `4.4733528137`.
Together with no-timing HHC and capped CoShMDM, the provisional three-comparator
Holm family is frozen in
`output/fair_eval/paired_fair_method_comparison_without_timotion.json`.  It is
explicitly superseded, not appended, when TiMotion produces all 20 final
replications.
  The first 38-token/RNG-side-effect replay still returned initial FID
  `16.04839` instead of `25.7493` and was interrupted below 300 updates.  The
  remaining mismatch was an implementation contract, not a sampler argument:
  it was still using the repaired sampler, which only lets currently masked
  tokens compete, while the time-bracketed historical implementation
  re-predicts every valid channel position on every round.  The latter now has
  the explicit diagnostic-only dispatch name `sample_intra_cas_historical`;
  training-validation replay selects it automatically, whereas every formal
  common-50 run continues to dispatch `sample_intra_cas`.  Replaying with that
  isolated implementation returns initial FID `25.74927`, exactly matching the
  historical log at its printed precision (`25.7493`); the diagnostic may
  therefore proceed to the 5k/10k/20k training gates.  The independent
  `output/historical_mgt_sampler_behavior_audit.json` additionally requires
  bitwise-identical recovered/current historical logits, sampled tokens and
  post-sampling RNG state while asserting that the formal sampler remains a
  different dispatch.  Its 5k rolling checkpoint passed the full state
  audit: global step 5000, finite model/optimizer tensors, AdamW betas
  `(0.5,0.9)`, LR `2e-4`, weight decay `1e-6`, and MultiStepLR last epoch 5000.
  The frozen evidence is `output/historical_mgt_v4_5k_checkpoint_audit.json`.
  A subsequent expression-level audit found that V4's historical all-valid
  loss changed the recovered floating-point reduction order. Unequal-length
  batches produced a `4.77e-7` loss difference and non-bitwise-identical
  gradients. V4 was therefore stopped around 5.9k and marked non-reusable by
  its `aborted_diagnostic.json`. V5 restores the exact recovered weighted-token
  then batch-wide-sum reduction and restarted from initialization after the full
  consistency suite passed. Its initial FID was again exactly `25.74927`.
  The enhanced original-200k audit then exposed another structural mismatch:
  the recovered AdamW group contains 482 parameter tensors, including 138
  detached VQ parameters, while V5 filtered the group down to the 344 tensors
  that acquire Adam state. V5 stopped at update 742 and is marked non-reusable.
  V6 restored the 482-parameter membership but exposed a separate Lightning
  child-module mode error: the refactor explicitly called
  `trans_encoder.eval()` before the sanity loop. Lightning captures and
  restores every submodule's pre-sanity mode verbatim, and its fit loop does
  not then call `model.train()` to repair an eval child. Transformer dropout
  was consequently disabled throughout V6. The local W&B transaction log
  records `Train/all=7.631497@999` and `5.958329@1999`, versus historical
  `7.760624` and `6.512545`; V6 was stopped at logged step 4049 and marked
  non-reusable. The frozen comparison is
  `output/historical_mgt_v6_training_trajectory_comparison.json`, and the
  run's `aborted_diagnostic.json` states the exclusion explicitly. V7 removes
  the premature eval call and fails fast in `on_train_start` unless the
  transformer is train and the loaded VQ is eval. Its startup summary confirms
  `trans_encoder=train`, `VQ=eval`, and the original 322M/151M/473M
  trainable/frozen/total counts. Its initial FID is again exactly `25.74927`,
  and all four locally recorded W&B metrics are bitwise identical to the
  February 13 transaction log at steps 49, 499, 849, 999, 1499 and 1999
  (including `Train/all=7.760623931884766@999` and
  `6.512544631958008@1999`). The zero-tolerance 24-value gate is frozen in
  `output/historical_mgt_v7_training_trajectory_2k_audit.json` with status
  `pass`; later milestones continue in a separate live comparison. The independent
  `output/historical_mgt_module_mode_contract_audit.json` binds the recovered
  and current AST plus Lightning 2.6 restore semantics and passes. V7 now owns
  the 5k/10k/20k gates. All three pass. At 20k,
  `tools/audit_mgt_repair_v7_20k.sh` compares all 403 candidate training rows
  through step 20149 and both 10k-spaced validation rows to the frozen W&B run;
  every value has absolute difference zero. The 20k ordinary FID is exactly
  `0.6423928737640381`. The pre/post-validation checkpoints have bitwise-equal
  model, optimizer, scheduler and hyperparameters; only callback/loop metadata
  records validation. The four 20k artifacts are prefixed
  `output/historical_mgt_v7_20k_` or
  `output/historical_mgt_v7_{training_trajectory,validation}_20k_`.
  The bitwise loss
  and gradient evidence is `output/historical_mgt_loss_reduction_audit.json`
  and is a required input to the aggregate historical training-contract audit.
  That aggregate audit also verifies the active environment against the frozen
  W&B requirements: torch 2.5.1, Lightning and PyTorch-Lightning 2.6.0, NumPy
  2.2.6, and W&B 0.23.1 all match exactly.

  The live replay has now passed the 60k milestone as well: all 1,203 training
  rows available through candidate step 60,149 have zero loss difference, and
  every one of the six validation metrics at 10k through 60k is exactly equal.
  The 60k FID is `0.3720308244228363`. Its atomic rolling checkpoint contains
  931 finite model tensors, 482 optimizer parameter tensors, 344 Adam state
  entries and scheduler `last_epoch=60000`. Evidence is frozen in
  `output/historical_mgt_v7_training_trajectory_all_rows_live.json`,
  `output/historical_mgt_v7_validation_60k_live.json` and
  `output/historical_mgt_v7_60k_checkpoint_audit.json`.
  The same zero-tolerance replay now passes 120k: all 2,410 training rows
  available through candidate step 120,499 and all 72 validation values from
  10k through 120k are exactly equal to the historical W&B transaction log.
  The 120k ordinary FID is `0.20432139933109283`.  Its rolling checkpoint has
  931 finite model tensors, 482 optimizer parameter tensors, 344 Adam state
  entries and scheduler `last_epoch=120000`; the post-validation copy is
  bitwise equal for model, optimizer, scheduler and hyperparameters.  Evidence
  is `output/historical_mgt_v7_training_trajectory_all_rows_live.json`,
  `output/historical_mgt_v7_validation_120k_live.json`,
  `output/historical_mgt_v7_120k_prevalidation_checkpoint_audit.json` and
  `output/historical_mgt_v7_120k_pre_post_validation_pair_audit.json`.
  At 130k, all 78 validation values from 10k through 130k remain exactly equal;
  the ordinary FID is `0.21991795301437378`.  The rolling checkpoint is finite
  and the same-step post-validation copy remains bitwise equal for model,
  optimizer, scheduler and hyperparameters.  The three new artifacts are
  prefixed `output/historical_mgt_v7_130k_` or named
  `output/historical_mgt_v7_validation_130k_live.json`.
  The 140k milestone also passes at zero tolerance: all 84 ordinary validation
  values through 140k equal the historical log and FID is
  `0.3215497136116028`.  Its pre-validation checkpoint has 931 finite model
  tensors, 482 optimizer parameters, 344 Adam states and scheduler step 140k;
  model, optimizer, scheduler and hyperparameters are bitwise unchanged by
  validation.  Evidence is `output/historical_mgt_v7_validation_140k_live.json`,
  `output/historical_mgt_v7_140k_prevalidation_checkpoint_audit.json` and
  `output/historical_mgt_v7_140k_pre_post_validation_pair_audit.json`.
  The 150k and 160k milestones continue the exact replay. Their ordinary FIDs
  are `0.2316998988389969` and `0.14545394480228424`, respectively; all 96
  validation values across the sixteen 10k-spaced milestones through 160k are
  exactly equal to the historical W&B log. All 3,205 candidate training rows
  through step 160,249 likewise have zero loss difference. The 160k rolling
  checkpoint SHA-256 is
  `91a53ad1c0f70bc939cb47cc9f92af974898028478a3ddc0ecb6a408ea27abb5`;
  the historical 150k scheduler milestone has reduced LR to `1e-5`, and all
  931 model tensors, 482 optimizer parameters and 344 Adam states pass the
  finite-state audit. Model, optimizer, scheduler and hyperparameters remain
  bitwise equal across validation. Evidence is
  `output/historical_mgt_v7_training_trajectory_all_rows_live.json`,
  `output/historical_mgt_v7_validation_160k_live.json`,
  `output/historical_mgt_v7_160k_prevalidation_checkpoint_audit.json` and
  `output/historical_mgt_v7_160k_pre_post_validation_pair_audit.json`.
  At 170k the ordinary FID is `0.1580042988061905`; all 102 validation values
  through that milestone are exactly equal to the historical transaction log,
  and all 3,404 candidate training rows through step 170,199 retain zero loss
  difference. The pre-validation checkpoint SHA-256 is
  `028229df9747a0249550f84658bf55823a9068caa1fe6ad08206c491c7fd3a6d`;
  its complete model/optimizer state is finite and its scheduler/LR are exact.
  Model, optimizer, scheduler and hyperparameters remain bitwise unchanged in
  the post-validation checkpoint, while the best checkpoint correctly remains
  the 160k FID `0.14545394480228424`. Evidence is
  `output/historical_mgt_v7_validation_170k_live.json`,
  `output/historical_mgt_v7_170k_prevalidation_checkpoint_audit.json` and
  `output/historical_mgt_v7_170k_pre_post_validation_pair_audit.json`.
  The 180k milestone establishes a new exactly reproduced historical best:
  ordinary FID is `0.138837531208992`; all 108 validation values through 180k
  and all 3,602 training rows through candidate step 180,099 have zero
  difference. The pre-validation checkpoint SHA-256 is
  `6f334565414bf1aae92c8ed8d4f65e56d08195c9e6e48ef998e34f74c0ebeaa2`;
  its full model/optimizer state is finite and LR remains `1e-5`. Validation
  correctly advances the best path from 160k to 180k while model, optimizer,
  scheduler and hyperparameters remain bitwise unchanged. Evidence is
  `output/historical_mgt_v7_validation_180k_live.json`,
  `output/historical_mgt_v7_180k_prevalidation_checkpoint_audit.json` and
  `output/historical_mgt_v7_180k_pre_post_validation_pair_audit.json`.
  The atomic 185k rolling checkpoint also passes the complete state audit at
  global step 185,000/epoch 1,380: scheduler `last_epoch=185000` and
  `_step_count=185001`, all 931 model and 1,032 optimizer tensors finite, and
  LR/weight decay exactly `1e-5/1e-6`. Its SHA-256 is
  `85aa27bb6bc12a3dad980e921a126a3afcb4164e3f754b1455f77a53e63f6b89`.
  All 3,699 candidate training rows through step 184,949 still have zero loss
  difference. Evidence is `output/historical_mgt_v7_185k_checkpoint_audit.json`.
  This rolling milestone does not run validation and therefore makes no FID
  claim.
  The 190k validation remains an exact replay: current ordinary FID is
  `0.15922923386096954`, all 114 validation values through 190k have zero
  difference, and all 3,802 training rows through candidate step 190,099 retain
  zero loss difference. The pre/post checkpoint SHA-256 values are
  `0c8a41f2777efe0aaede5550ae3d3dbed05577f86d68a0f0c8dc6c19a9343263` and
  `d9711889e58b8a43b0ff28a70cbd8a6e780b0a2879320bbdb5eb89e8383ecc9b`.
  Model, optimizer, scheduler and hyperparameters are bitwise identical across
  validation; the callback updates current FID to the 190k value while the
  historical best correctly remains 180k FID `0.138837531208992`. Evidence is
  `output/historical_mgt_v7_validation_190k_live.json`,
  `output/historical_mgt_v7_190k_prevalidation_checkpoint_audit.json` and
  `output/historical_mgt_v7_190k_pre_post_validation_pair_audit.json`.
  The 195k rolling checkpoint passes the complete state audit at global step
  195,000/epoch 1,455: scheduler `last_epoch=195000` and
  `_step_count=195001`, all 931 model and 1,032 optimizer tensors finite, and
  LR/weight decay still `1e-5/1e-6`. Its SHA-256 is
  `9df5cfe0e7d99fc548c6c8b3d0b0f7e5f2bd47607f6a8f3b00646f23abb0da8d`.
  After the logger flush, all 3,901 training rows through candidate step
  195,049 retain zero loss difference. Evidence is
  `output/historical_mgt_v7_195k_checkpoint_audit.json`; this rolling milestone
  does not execute validation and has no FID row.
  The 200k validation establishes another exact historical best: ordinary FID
  is `0.13571932911872864`, all 120 validation values through 200k have zero
  difference, and all 4,001 training rows through candidate step 200,049 retain
  zero loss difference. The pre/post SHA-256 values are
  `5a6231992124da4e5c8d3581d91ef6dcc144499f1d3f157818cca91806400f4e` and
  `6198934de1df6fd7be41da03d6f2df2ff67606b483b5836bc2165f6c7b2721c6`.
  Model, optimizer, scheduler and hyperparameters remain bitwise identical,
  while callback best correctly advances from 180k FID
  `0.138837531208992` to 200k. Evidence is
  `output/historical_mgt_v7_validation_200k_live.json`,
  `output/historical_mgt_v7_200k_prevalidation_checkpoint_audit.json` and
  `output/historical_mgt_v7_200k_pre_post_validation_pair_audit.json`.
  The 205k rolling checkpoint also passes the complete state audit at global
  step 205,000/epoch 1,529: scheduler `last_epoch=205000` and
  `_step_count=205001`, all 931 model and 1,032 optimizer tensors finite, and
  LR/weight decay `1e-5/1e-6`. Its SHA-256 is
  `48282e7bd31acf367b995ec57705fc7f58fa8029dca4b7e56a31d3e2119fa304`.
  All 4,099 training rows through candidate step 204,949 retain zero loss
  difference. Evidence is `output/historical_mgt_v7_205k_checkpoint_audit.json`;
  this rolling milestone does not execute validation.
  The 210k validation remains exact: current ordinary FID is
  `0.16243889927864075`, all 126 validation values through 210k have zero
  difference, and all 4,202 training rows through candidate step 210,099 retain
  zero loss difference. The pre/post SHA-256 values are
  `c1ee8aeffff4d4d3ef146aa921942c1d21e7df3cdacbe4f10816f56f3ca27564` and
  `e06f06efbae42af79b76d7e9ccabc8e4fcb8e89b04c82223be369c3c3d5ee38c`.
  Model, optimizer, scheduler and hyperparameters are bitwise identical, and
  callback best correctly remains the 200k FID `0.13571932911872864`.
  Evidence is `output/historical_mgt_v7_validation_210k_live.json`,
  `output/historical_mgt_v7_210k_prevalidation_checkpoint_audit.json` and
  `output/historical_mgt_v7_210k_pre_post_validation_pair_audit.json`.
  The 215k rolling checkpoint also passes the complete state audit at global
  step 215,000/epoch 1,604: scheduler `last_epoch=215000` and
  `_step_count=215001`, all 931 model and 1,032 optimizer tensors finite, and
  LR/weight decay `1e-5/1e-6`. Its SHA-256 is
  `20a6a9c5ad8d051c3f71d9c387660f3bb7db11885e29b54aac9c0721a60535df`.
  All 4,302 training rows through candidate step 215,099 retain zero loss
  difference. Evidence is `output/historical_mgt_v7_215k_checkpoint_audit.json`;
  this rolling milestone does not execute validation.
  The 220k validation remains exact: current ordinary FID is
  `0.1590881049633026`, all 132 validation values through 220k have zero
  difference, and all 4,407 training rows through candidate step 220,349 retain
  zero loss difference. The pre/post SHA-256 values are
  `9cb46d66ca547bf474cc9fd8c2b9694a9fb0be93bd22c17bffb84b985571449d` and
  `37d23995901418b33f548feba441624d63457d7c2c4aafffbcb90b50e069e7b0`.
  Model, optimizer, scheduler and hyperparameters are bitwise identical, and
  callback best correctly remains the 200k FID `0.13571932911872864`.
  Evidence is `output/historical_mgt_v7_validation_220k_live.json`,
  `output/historical_mgt_v7_220k_prevalidation_checkpoint_audit.json` and
  `output/historical_mgt_v7_220k_pre_post_validation_pair_audit.json`.
  The 225k rolling checkpoint also passes the complete state audit at global
  step 225,000/epoch 1,679: scheduler `last_epoch=225000` and
  `_step_count=225001`, all 931 model and 1,032 optimizer tensors finite, and
  LR/weight decay `1e-5/1e-6`. Its SHA-256 is
  `c981c44162246feb6ce8c8cfc88b312eabb93513653769d4bf6609b58b936721`.
  All 4,503 training rows through candidate step 225,149 retain zero loss
  difference. Evidence is `output/historical_mgt_v7_225k_checkpoint_audit.json`;
  this rolling milestone does not execute validation.
  The 230k validation remains exact: current ordinary FID is
  `0.15852901339530945`, all 138 validation values through 230k have zero
  difference, and all 4,606 training rows through candidate step 230,299 retain
  zero loss difference. The pre/post SHA-256 values are
  `44663f6b0d743d9318a171a6a6292d7b6211a6a4efaa34be7c2333cf8d454d09` and
  `1d918979e5b60242e7aaac175e6d5161959b559526ccfec7eb4061ff18d9921a`.
  Model, optimizer, scheduler and hyperparameters are bitwise identical, and
  callback best correctly remains the 200k FID `0.13571932911872864`.
  Evidence is `output/historical_mgt_v7_validation_230k_live.json`,
  `output/historical_mgt_v7_230k_prevalidation_checkpoint_audit.json` and
  `output/historical_mgt_v7_230k_pre_post_validation_pair_audit.json`.
  The 235k rolling checkpoint also passes the complete state audit at global
  step 235,000/epoch 1,753: scheduler `last_epoch=235000` and
  `_step_count=235001`, all 931 model and 1,032 optimizer tensors finite, and
  LR/weight decay `1e-5/1e-6`. Its SHA-256 is
  `39db17295056ccd6745c4bdebdd563120c2b8292a7d1f54e35c1c5854d723930`.
  All 4,698 training rows through candidate step 234,899 retain zero loss
  difference. Evidence is `output/historical_mgt_v7_235k_checkpoint_audit.json`;
  this rolling milestone does not execute validation.
  The 240k validation remains exact: current ordinary FID is
  `0.14474517107009888`, all 144 validation values through 240k have zero
  difference, and all 4,801 training rows through candidate step 240,049 retain
  zero loss difference. The pre/post SHA-256 values are
  `3d9e0def8efb7206b816618f2a7a66189a1754145e75d822c653136a928dd1c7` and
  `8bee23c01b6aed841369a07f86b44683f1c72569db0f191f028f0f1e5a91bec2`.
  Model, optimizer, scheduler and hyperparameters are bitwise identical, and
  callback best correctly remains the 200k FID `0.13571932911872864`.
  Evidence is `output/historical_mgt_v7_validation_240k_live.json`,
  `output/historical_mgt_v7_240k_prevalidation_checkpoint_audit.json` and
  `output/historical_mgt_v7_240k_pre_post_validation_pair_audit.json`.
  The 245k rolling checkpoint also passes the complete state audit at global
  step 245,000/epoch 1,828: scheduler `last_epoch=245000` and
  `_step_count=245001`, all 931 model and 1,032 optimizer tensors finite, and
  LR/weight decay `1e-5/1e-6`. Its SHA-256 is
  `53425216ecc403526a5cd40ad339e63c7c2051c2a800e701a39245ddfbe65ed9`.
  All 4,902 training rows through candidate step 245,099 retain zero loss
  difference. Evidence is `output/historical_mgt_v7_245k_checkpoint_audit.json`;
  this rolling milestone does not execute validation.
  The 250k validation remains exact: current ordinary FID is
  `0.16886506974697113`, all 150 validation values through 250k have zero
  difference, and all 5,005 training rows through candidate step 250,249 retain
  zero loss difference. The pre/post SHA-256 values are
  `a7576771014df24219280978cedc1cfb657630ac93bf8f7e85d019c3dd0bab24` and
  `fae4dc63b186618246d5c1c48762f276a596acfe1169285a217dda274fbc3e87`.
  Model, optimizer, scheduler and hyperparameters are bitwise identical, and
  callback best correctly remains the 200k FID `0.13571932911872864`.
  Evidence is `output/historical_mgt_v7_validation_250k_live.json`,
  `output/historical_mgt_v7_250k_prevalidation_checkpoint_audit.json` and
  `output/historical_mgt_v7_250k_pre_post_validation_pair_audit.json`.
  The 255k rolling checkpoint also passes the complete state audit at global
  step 255,000/epoch 1,902: scheduler `last_epoch=255000` and
  `_step_count=255001`, all 931 model and 1,032 optimizer tensors finite, and
  LR/weight decay `1e-5/1e-6`. Its SHA-256 is
  `7191d81f6b6a3ac719b939d25d748462a71495a03d466cf9a1915caba6cb5b42`.
  All 5,099 training rows through candidate step 254,949 retain zero loss
  difference. Evidence is `output/historical_mgt_v7_255k_checkpoint_audit.json`;
  this rolling milestone does not execute validation.
  The 260k validation remains exact: current ordinary FID is
  `0.1608208417892456`, all 156 validation values through 260k have zero
  difference, and all 5,207 training rows through candidate step 260,349 retain
  zero loss difference. The pre/post SHA-256 values are
  `a1503c3592f0c8810f7043e6321bac9f01879d569d253c4d5aefee5078efef01` and
  `de2dbdb5213b7aac9ac9425f2c8480b5417eb879934a84ee1dab227d50429e4b`.
  Model, optimizer, scheduler and hyperparameters are bitwise identical, and
  callback best correctly remains the 200k FID `0.13571932911872864`.
  Evidence is `output/historical_mgt_v7_validation_260k_live.json`,
  `output/historical_mgt_v7_260k_prevalidation_checkpoint_audit.json` and
  `output/historical_mgt_v7_260k_pre_post_validation_pair_audit.json`.
  The 265k rolling checkpoint also passes the complete state audit at global
  step 265,000/epoch 1,977: scheduler `last_epoch=265000` and
  `_step_count=265001`, all 931 model and 1,032 optimizer tensors finite, and
  LR/weight decay `1e-5/1e-6`. Its SHA-256 is
  `c05d8fe50e8aa244ebb43eeaf7a315672dde8066e4b4b21c063447b7903de1df`.
  All 5,304 training rows through candidate step 265,199 retain zero loss
  difference. Evidence is `output/historical_mgt_v7_265k_checkpoint_audit.json`;
  this rolling milestone does not execute validation.
  The 270k validation remains exact: current ordinary FID is
  `0.17415158450603485`, all 162 validation values through 270k have zero
  difference, and all 5,419 training rows through candidate step 270,949 retain
  zero loss difference. The pre-validation checkpoint SHA-256 is
  `1b01373df9eb623d46502e6a69ce679476d52fb807827eafd8bcaa20a7461cac`.
  This FID ranks twelfth through 270k and therefore correctly falls outside
  `save_top_k=10`; Lightning does not create a `model-000270000-*.ckpt`.
  The ten actually retained steps exactly equal the exact-FID top ten, so no
  pre/post pair is claimed for a nonexistent file. Evidence is
  `output/historical_mgt_v7_validation_270k_live.json`,
  `output/historical_mgt_v7_270k_prevalidation_checkpoint_audit.json` and
  `output/historical_mgt_v7_270k_topk_retention_audit.json`.
  The 275k rolling checkpoint also passes the complete state audit at global
  step 275,000/epoch 2,052: scheduler `last_epoch=275000` and
  `_step_count=275001`, all 931 model and 1,032 optimizer tensors finite, and
  LR/weight decay `1e-5/1e-6`. Its SHA-256 is
  `2da4b5930fdb6eeb57f900bd7119d17807800eadd29e0e7b3f532368a6cd2ad6`.
  All 5,499 training rows through candidate step 274,949 retain zero loss
  difference. Evidence is `output/historical_mgt_v7_275k_checkpoint_audit.json`;
  this rolling milestone does not execute validation.
  The 280k validation is also exact: ordinary FID is
  `0.16047221422195435`, and Diversity/TOP1/TOP2/TOP3/Matching together with
  every prior milestone make 168 validation values with zero maximum
  difference. All 5,602 training rows through candidate step 280,099 also
  retain zero loss difference. As expected, this FID enters the top ten and
  retains `model-000280000-0.1605.ckpt`, whose SHA-256 is
  `07865f93733d432da1bd01e2eff967640b9e1fd054881c5c17fb70fc6e8bb8ba`.
  Its 931 model tensors, optimizer and scheduler are bitwise identical to the
  original pre-cleanup 280k checkpoint. It is likewise training-state
  identical to the same-step pre-validation rolling checkpoint; only
  callback/loop metadata records validation, and best remains the 200k FID
  `0.13571932911872864`. Evidence is
  `output/historical_mgt_v7_validation_280k_live.json`,
  `output/historical_mgt_v7_280k_checkpoint_audit.json`,
  `output/historical_mgt_v7_step280k_checkpoint_pair_audit_live.json` and
  `output/historical_mgt_v7_280k_pre_post_validation_pair_audit.json`.
  The 285k rolling checkpoint continues to pass the complete state audit at
  global step 285,000/epoch 2,126: scheduler `last_epoch=285000` and
  `_step_count=285001`, all 931 model and 1,032 optimizer tensors finite, and
  LR/weight decay `1e-5/1e-6`. Its SHA-256 is
  `219d9fcbdd6a9832322d7a11316f587275d8d1dc6e6d344990e6978fdeb0b7b0`.
  All 5,699 training rows through candidate step 284,949 retain zero loss
  difference. Evidence is `output/historical_mgt_v7_285k_checkpoint_audit.json`;
  this rolling milestone does not execute validation.
  The 290k validation remains exactly equal: ordinary FID is
  `0.19474996626377106`, all 174 validation values through this milestone have
  zero maximum difference, and all 5,803 training rows through candidate step
  290,149 retain zero loss difference. The pre-validation checkpoint SHA-256
  is `681c918a73f7bc3c9e08e2e2152ac886536eedb9ab8706358ccb437a46a5ffa2`.
  This FID ranks fourteenth and correctly does not create a
  `model-000290000-*.ckpt`; the ten actually retained checkpoints exactly equal
  the exact-FID top ten. Evidence is
  `output/historical_mgt_v7_validation_290k_live.json`,
  `output/historical_mgt_v7_290k_prevalidation_checkpoint_audit.json` and
  `output/historical_mgt_v7_290k_topk_retention_audit.json`.
  The 295k rolling checkpoint passes the final pre-terminal complete state
  audit at global step 295,000/epoch 2,201: scheduler `last_epoch=295000` and
  `_step_count=295001`, all 931 model and 1,032 optimizer tensors finite, and
  LR/weight decay `1e-5/1e-6`. Its SHA-256 is
  `0701d1726a5dc1f2cdfe7e4d6576b606747811cf57ac648b0243f267f96d1fae`.
  All 5,899 training rows through candidate step 294,949 retain zero loss
  difference. Evidence is `output/historical_mgt_v7_295k_checkpoint_audit.json`;
  this rolling milestone does not execute validation.
  `output/historical_mgt_dependency_contract_audit.json` additionally binds
  byte-identical encoder/VQ/FK dependencies and AST-identical legacy attention,
  runtime mask/Gumbel, and token-dataset RNG logic. The only dependency with a
  post-run mtime, `paramUtil.py`, was recovered from the frozen Local History/VFS
  store: later edits only split contact-region palm/back labels, while the
  `SMPLX_TREE` and `INTERX_SKELETONS` tail consumed by MGT is byte-identical.
  The complete authoritative run binding is
  `output/historical_mgt_original_training_contract_audit.json`.  It binds the
  February 13 metadata/config/summary/transaction log, the initial plus thirty
  10k-spaced FID milestones, all four time-bracketed Local History sources,
  the dataset and optimizer sources that predate the launch, the finite 200k
  checkpoint audit, and both initialization/sampler behavior audits.  Its
  status must be `pass` before historical trajectory comparisons are used.
  The two separately audited development **test** runs remain useful sampler
  history: both used the function's default 10 outer steps; the 2026-02-17
  test used temperature 1.0 with CFG disabled, while the 2026-03-05 test used
  temperature 0.5 with CFG 3.5.  The 3-step value in
  commit `75814e57e0702789176da6713056a15ed4a192e2`, immediately before the
  large cleanup beginning at `0f58ffbb9db95a5d9ddc7f2e8a40b8602c17a325`,
  is retained only as a function-default snapshot and is not substituted for
  the recovered original-development runtime.  The AST/time/hash evidence is
  `output/historical_mmm_sampler_local_history_audit.json`; it is Local History
  recovery, not a claim that the uncommitted files existed in Git.  The three
  recovery manifests bind every selected source to the JetBrains VFS content
  ID, byte count and SHA-256, while the audit separately hashes the original
  LocalHistory and VFS storage files.  A byte-identical 134 MB durable snapshot
  is preserved at `output/historical_mmm_local_history_store_232`; the audit
  reads this fixed copy by default rather than the mutable live IDE cache.
  The completed-run validation binding and the two parser implementation
  hashes are stored in `output/historical_mmm_wandb_comparison.json`; the final
  report rejects a missing source interval, VFS binding, or exact runtime
  field instead of falling back to “unknown historical parameters.”
  The final audit reads the current offline W&B transaction log directly,
  binds its SHA-256 to the report, requires a 300,000-step terminal record and
  prints every fixed-protocol validation milestone beside the historical row.
  In that diagnostic table, `val/FID` is the current-milestone value used for
  checkpoint selection; the inherited validation hook writes Diversity,
  Top-1/2/3 and Matching as running-best state, so the report labels them as
  such instead of presenting them as per-checkpoint measurements.  Final fair
  tables are always produced by the independent common50 evaluator.
  Training losses are retained for diagnostics but are not treated as directly
  comparable because the mask/objective and architecture changed.
  The completed online W&B history is separately frozen in
  `output/historical_mmm_wandb_remote_history.json`.  It contains all 30
  validation milestones, binds the server history-artifact digest, verifies
  every server summary field against the local summary, and records that the
  run has no uploaded `code/` or `diff.patch`.  Refresh it read-only with:

  ```bash
  conda run -n New_MMM python \
    tools/freeze_historical_mmm_wandb_remote_history.py
  ```

  The final report pairs old/current validation FID only at identical update
  counts and labels the entire table non-comparable; the frozen trajectory is
  diagnostic evidence, never a baseline result.

### Historical MGT terminal replay and common-50 conditioning repair

- V7 completed 300,000 updates. All 6,000 logged training rows and all 180
  validation values at 30 milestones are byte/numerically exact against the
  pre-cleanup W&B history. At step 280,000 the 931 model tensors, optimizer,
  and scheduler are bitwise exact against the original checkpoint. The final
  audits are `output/historical_mgt_v7_training_trajectory_300k_audit.json`,
  `output/historical_mgt_v7_validation_300k_audit.json`, and
  `output/historical_mgt_v7_step280k_checkpoint_pair_audit.json`.
- The first common-50 bridge was invalid because it compared two different CFG
  contracts. Historical training requires
  `historical_mmm_shape_dropout` (word tokens visible; global CLIP and shape
  dropped), whereas the frozen Aug-28 pre-cleanup exporter used
  `paper_text_cfg` at inference (word/global text dropped; requested shape
  retained). The failed FID `0.2267311` versus reference `0.1869647` is kept in
  `output/historical_mgt_v7_step280k_common50_bridge.json` as a diagnostic and
  must not be used as a model result.
- Export now records separate `training_conditioning_contract` and
  `inference_conditioning_contract` attributes and switches inference behavior
  only after an exact checkpoint load. With V7 step 280k and
  `paper_text_cfg`, the first 32 common-50 samples reproduce the frozen
  reference motion array byte-for-byte (`max_abs=0`, zero differing elements).
  Evidence is
  `output/historical_mgt_v7_step280k_common50_conditioning_first32_audit.json`.
  After the fresh main HDF5 completed, the same comparison was extended to all
  1,696 official samples: names and lengths match, every motion byte matches,
  `max_abs=0`, and the two motion-array SHA-256 values are both
  `bc39eb6e...a5c8`.  The full evidence is
  `output/historical_mgt_v7_step280k_common50_conditioning_full1696_audit.json`.
  An independent official CPU main-only score of that completed candidate gives
  FID `0.18696466786173005`, exactly equal to the frozen reference re-score;
  the diagnostic score is
  `output/mgt_diagnostics/v7_step280k_common50_paper_text_cfg_full1696_main_only_score.json`.
  MM is checked separately by the streaming
  `tools/audit_mgt_common50_mm_replay.py`: it requires all 30 completion
  markers and compares every one of the 3,000 generated motions repeat by
  repeat without loading both multi-GB HDF5 files at once.  The completed
  audit passes for all 30 repeats: names and lengths match, every motion byte
  matches, `max_abs=0`, the number of differing elements is zero, and both
  combined motion-array SHA-256 values are
  `e27cc46374ac1742ad41917587574ca04de1db265650edb49932a07d82ebcfea`.
  The final report reruns and binds
  `output/historical_mgt_v7_step280k_common50_mm_full30_audit.json`.
  The full 1,696+MM bridge completed under label
  `hhc_mgt_repair_v7_step280k_common50_paper_text_cfg_contact_rep00`.  A fresh
  official-CPU re-score gives identical reference/candidate values for every
  ordinary metric (FID `0.18696466786173005`; absolute difference zero), so it
  created the passing `output/mgt_repair_gate.json`.  The gate then released
  the formal run
  `output/t2i_trans/2026-08-30-16-08-07_hhc_formal_raw12288_repaired_mgt_300k_seed123_v1`;
  its launch manifest declares the complete paper training contract.
- The original durable v3 service chain had three fail-closed stages: the
  common-50 finalizer, `hhc-formal-gate-waiter-v7-v3.service`, and
  `hhc-post-formal-v7-v3.service`.  Those user services are now failed/obsolete
  after the host restart and are not treated as live orchestration state.  The
  post-formal stage is implemented by
  `tools/queue_post_formal_after_v3.sh`; it requires the formal service to exit
  successfully when a service name is supplied, or accepts the explicit
  `none` sentinel for manual recovery, and in both cases rechecks the selection
  plus both formal common-50 summaries
  before running the mask-exposure/shape-dependence diagnostics.  Both token
  diagnostics draw 256 records uniformly without replacement from the active
  split at fixed seed `90123`; they do not take the lexicographically first
  actor groups.  The shape-dependence donor mapping hashes the exact float32
  beta plus integer gender condition and requires all 256 donor pairs to have
  different signatures.  A mere motion-name derangement is rejected because
  adjacent Inter-X motions commonly share the same `Gxxx` actor shape.  The
  conditional-support audit separately mirrors the formal token loader's
  `>=20`-token bucket and 49-token payload cap: its evidence is computed on the
  actual `8617/534` train/validation token records, not all `9095/569` cached
  records.  After this correction, validation marginal occurrences remain
  `99.56%/99.97%/99.88%` covered for Body/Hand/Root, whereas Root-Body pairs
  and Root-Body-Hand triplets cover only `7.83%/0.268%`; the conditional
  composition gap therefore remains and is no longer based on out-of-loader
  rows.  Weak-FID
  diagnosis is prioritized: root-tokenizer/online-W&B wiring, the 50k
  masked-loss control and matched shape-augmentation arms run next.  The full
  optimization study, sampling Pareto grid and contact-timing ablation now run
  before shape through the direct, non-polling
  `tools/run_current_interx_review_experiments_before_shape.sh` entrypoint.
  The former `queue_post_shape_review_experiments_gpu1.sh` is archived and
  exits immediately so it cannot launch duplicate legacy experiments. Every post-formal launcher
  binds Conda by absolute path so the systemd environment does not depend on an
  interactive-shell `PATH`.  The older end-to-end GPU-1 branch already failed
  closed on the superseded gate and now owns only the independent GPU-0
  CoShMDM/TiMotion branch; it cannot launch a duplicate HHC or post-HHC chain.
  Before either new MGT arm starts, the source-derived token-window audit
  `output/hhc_formal_mgt_token_window_alignment_audit.json` separates a real
  training-payload difference from sampler settings.  Clean MMM revision
  `a01d5ac...54a7` defaults its Inter-X token tensor to 38 positions and
  reserves one for END, so it retains at most 37 motion tokens/148 frames.  The
  completed formal run used 50 positions and retains 49 motion tokens/196
  frames.  The imported 9,664-sequence cache reaches 247 motion tokens: these
  contracts truncate 5,006 versus 2,792 samples and discard 106,792 versus
  60,087 tokens.  Frozen fair requests reach 150 frames/38 motion tokens, so
  the settings do not differ only in padding, while the original 38-position
  setting is also not falsely described as an exact 150-frame match.  A
  matched online-W&B 50k control changes only
  `training_token_length_contract` from `paper_50` to `original_mmm_38`; it is
  promoted only if both validation-set checkpoint selection and the matched
  50k endpoint improve by at least 0.1 absolutely and 15% relatively.  At 10k this control was worse (`1.1610538960` versus
  `0.9789292216`), while at 20k it was better (`1.0573199987` versus
  `1.2168186903`).  Because the sign changes across milestones, neither point
  is treated as a causal diagnosis; the pre-registered decision waits for the
  complete 30k--50k trajectory.  At 30k the control improves to
  `0.6275210977`, versus paper-50 `1.0176038742` (absolute FID improvement
  `0.3900827765`).  At 40k it regresses to `0.9207732677`, only
  `0.0544919968` below paper-50 `0.9752652645`; this no longer passes the
  absolute promotion margin at that synchronized step.  At the terminal 50k
  synchronized point it is `0.8790842295` versus paper-50 `0.8862313628`, an
  improvement of only `0.0071471334`, so the pointwise promotion threshold is
  not met.  Selection over the complete 10k--50k validation set instead picks
  the 38-window 30k checkpoint (`0.6275210977`), which is better than the
  paper-50 early best (`0.8862313628`).  Because that conclusion is driven by
  a single non-monotone minimum rather than the matched 50k endpoint, it is
  retained as a checkpoint-selection result and must not be described as a
  50k causal effect.  The fail-closed assessor therefore records
  `checkpoint_selection_supports_38_but_matched_endpoint_does_not`; it does
  not authorize an original-38 full retrain or replace the paper-50 shape
  baseline.  Its masked accuracy is only `1.8574` percentage points
  below the pre-cleanup native MMM at the same step, but visible accuracy is
  still `14.0944` points lower; token length therefore explains an important
  part of the validation gap without eliminating the separate loss-support
  hypothesis.  The 20k model/optimizer/scheduler state is
  finite, and its pre/post-validation states are bitwise identical apart from
  validation loop/callback metadata.  Evidence is frozen in
  `output/hhc_original38_vs_formal50_validation_10k_30k.json`,
  `output/hhc_original38_vs_native_mmm_training_trajectory_through30k.json`,
  `output/hhc_original38_token_window_control_20k_selection.json`, and
  `output/hhc_original38_token_window_control_20k_pre_post_validation_pair_audit.json`.
  The completed run is online W&B `edmond_gla/NewInter/rx72y8g9`; its remote
  history audit passes with 1,001 telemetry events from optimizer step 50
  through 50,000.  Pre-clip gradient L2/max-abs ranges are
  `0.731383--5.554327`/`0.016357--0.882812`, weight L2/max-abs ranges are
  `220.186096--416.797943`/`1.000194--2.679024`, and both non-finite counters
  remain zero.  The next 50k diagnostic retains `paper_50` and changes only
  the loss support to historical all-valid tokens; it is online W&B run
  `edmond_gla/NewInter/fx7a9y1i`.  The paper masked-only run remains the formal
  result regardless of this diagnostic.  Its first 1,000-update W&B audit
  passes: gradient L2/max-abs is `0.752957--2.416588`/
  `0.042236--0.291016`, weight L2 grows from `220.185989` to `226.857285`, and
  both non-finite counts are zero.  At matched step 999, masked/visible
  accuracy is `6.6114%/27.0270%`, versus `2.1629%/1.6700%` for paper
  masked-only.  The 5k checkpoint then passes a full finite model/optimizer/
  scheduler audit.  At step 4,999 its masked accuracy is nearly unchanged
  (`50.9486%` versus `51.5811%`), whereas visible accuracy is `99.0747%`
  versus `55.9467%`; the support change is therefore isolated to the intended
  visible-token path rather than better early masked-token fitting.  This is
  strong learning-dynamics evidence but not yet an FID.  The live audit now
  keeps its pass/fail window fixed to the first 1,000 warm-up updates while
  reporting every later gradient/weight event separately under
  `full_run_metrics`; the completed remote run reaches update 50,000 (1,001
  events for the continuously synced health streams),
  zero non-finite gradient/weight elements, gradient L2/max-abs ranges
  `0.441381--3.958138`/`0.010803--0.593750`, and weight L2/max-abs ranges
  `220.185989--412.299438`/`1.000198--2.790677`.  At the terminal event the
  corresponding values are `0.546756/0.021118` and
  `412.299438/2.788017`; remote state is `finished`. Evidence is
  `output/hhc_all_valid_loss_control_wandb_audit.json`.
  HHC invokes this generic warm-up auditor with its predeclared
  `--max-weight-relative-span 0.15`: the first-1,000-step weight max-abs span
  is `10.6434%` while weight L2 spans `3.0298%`. Omitting that explicit HHC
  threshold would incorrectly apply the tool's TiMotion-oriented 5% default;
  it is an audit-invocation error, not changed training state.
  This remains operational monitoring and does not add a model-selection
  criterion.  Its first fixed-protocol validation at 10k gives FID
  `0.97720653`, Matching `3.81753993`, R-Precision Top-1/2/3
  `0.400735/0.577206/0.694853`, and Diversity `9.169179`.  Against the
  paper masked-only run at the same checkpoint, FID improves by only
  `0.00172269` (`0.97892922 -> 0.97720653`), while Matching improves by
  `0.05012918`; the all-valid objective has therefore not explained the FID
  gap at this milestone.  The 10k checkpoint and all optimizer state are
  finite.  Evidence is
  `output/hhc_all_valid_vs_formal50_validation_10k_live.json` and
  `output/hhc_all_valid_loss_control_10k_checkpoint_audit.json`; the run
  continues to the predeclared 50k diagnostic endpoint.  Its 15k checkpoint
  also passes full model/optimizer finite-state validation.  At training step
  15,999, masked accuracy is `78.5551%` versus `81.0539%` for masked-only,
  while visible accuracy is `99.7633%` versus `74.9236%`; all-valid therefore
  continues to specialize the visible-token path without improving masked
  fitting.  The 20k fixed validation subsequently regresses to FID
  `1.37048256` versus paper masked-only `1.21681869`.  At 30k it recovers to
  FID `1.22663069`, but remains worse than the matched paper masked-only
  `1.01760387` by `0.20902681`; Matching is `3.72356272`, R-Precision is
  `0.437500/0.623162/0.716912`, and Diversity is `9.308086`.  Its atomic
  `latest-000030000.ckpt` has finite model and optimizer state at global step
  30,000 with scheduler `last_epoch=30000`; the fixed pre/post-validation pair
  was intentionally registered only at 20k.  Evidence is
  `output/hhc_all_valid_vs_formal50_validation_10k_30k.json`,
  `output/hhc_all_valid_vs_formal50_training_trajectory_10k_30k.json`, and
  `output/hhc_all_valid_loss_control_30k_checkpoint_audit.json`. At 40k the
  common-50 FID reaches `0.71260744`, versus synchronous paper masked-only
  `0.97526526`, an improvement of `0.26265782` (`26.93%`). Matching improves
  by `0.02049661`, while R-Precision Top-1/Top-2 fall by
  `0.04411763/0.01470590` and Top-3 rises by `0.01286763`. Synchronous masked
  accuracy remains `1.1936` percentage points lower and visible accuracy is
  `21.0779` points higher, so this establishes loss support as a material FID
  factor without showing uniform semantic improvement or changing the formal
  paper loss. The 40k rolling checkpoint has 947 finite model tensors, 1,041
  finite optimizer tensors, and scheduler `last_epoch=40000`/
  `_step_count=40001`. Evidence is
  `output/hhc_all_valid_vs_formal50_validation_10k_40k.json`,
  `output/hhc_all_valid_vs_formal50_training_trajectory_10k_40k.json`, and
  `output/hhc_all_valid_loss_control_40k_checkpoint_audit.json`; the final
  decision still waits for the predeclared 50k endpoint and matched candidate
  set. The subsequent 45k rolling checkpoint also passes the full audit at
  epoch 505: all 947 model and 1,041 optimizer tensors are finite and the
  scheduler is exactly `last_epoch=45000`/`_step_count=45001`; evidence is
  `output/hhc_all_valid_loss_control_45k_checkpoint_audit.json`. A later control changes only the
  masked-NLL reduction to one global masked-token mean.
  The predeclared 50k endpoint has now completed: FID is `0.76316077`, versus
  synchronous paper masked-only `0.88623136` (absolute improvement
  `0.12307060`, relative `13.89%`). It improves absolutely but misses the
  predeclared 15% endpoint threshold. Selection over the identical 10k-spaced
  candidate set picks the all-valid 40k checkpoint (`0.71260744`) versus the
  formal early best at 50k (`0.88623136`), an absolute/relative improvement of
  `0.17362392`/`19.59%`; the assessor therefore says all-valid supervision
  explains a substantial part of the early FID gap, while remaining an
  exploratory diagnostic that cannot replace the paper masked-only loss.
  Terminal R-Precision Top-1/2/3 is
  `0.417279/0.613971/0.742647`, Matching `3.652760`, and Diversity `9.1891`.
  The 50k checkpoint, scheduler, all 947 model tensors and all 1,041 optimizer
  tensors pass the finite-state audit. The full remote W&B history audit also
  passes every training and validation stream through step 50,000. Evidence is
  `output/hhc_all_valid_loss_control_50k_selection.json`,
  `output/hhc_all_valid_loss_control_assessment.json`,
  `output/hhc_all_valid_loss_control_50k_checkpoint_audit.json`, and
  `output/hhc_all_valid_loss_control_wandb_history_audit.json`.
  Both assessors load the training manifests and fail closed on any other
  model/data/mask/optimizer/seed/cache/sampler drift.  The legacy
  `historical_38` spelling remains a replay alias and is not accepted as the
  new control's manifest identity.
  The former fourth single-process waiter,
  `hhc-final-report-v7-v3.service`, uses
  `tools/queue_final_report_after_v3.sh` to require the post-formal service to
  succeed, wait for the independent CoShMDM/TiMotion GPU-0 queue to exit, and
  verify every terminal result artifact before executing the final audit,
  CPU rescore, paired-comparison, physical-metric and report pipeline.  The
  same launcher now handles `none` before any `systemctl` call, so after a host
  restart the operator invokes `bash tools/queue_final_report_after_v3.sh none`;
  missing result dependencies still fail immediately and cannot be bypassed.
  No additional background waiter terminal is required.  The
  report launcher also binds Conda by absolute path and runs the local-history
  audit inside `New_MMM`, so it is safe under systemd's minimal `PATH`.
  Before the post-formal root-tokenizer transition, the formal launcher now
  runs `tools/freeze_mgt_training_sources.py` after selecting the completed
  run.  It materializes the exact 14 Python files whose hashes are recorded in
  that run's training manifest under `launch_sources/` and writes
  `launch_source_snapshot_manifest.json`.  The formal launcher-parity audit,
  paper-contract audit and final report bind those immutable files; the two later shape-augmentation
  arms continue to bind the modified live tree.  This prevents the intentional
  augmentation-only wiring from being misreported as formal-HHC source drift.
  The freezer is idempotent after the live transition and repairs a damaged
  snapshot from the live tree only while the live hashes still equal the
  original training manifest.
- The first HHC VQ run was stopped at 9,000 because the legacy tokenizer used
  an unreported 10% shape dropout.  It is preserved under
  `checkpoints/vq_paper_shape_drop_aborted_9000` and is not used for results.
- A second VQ attempt was stopped at 2,305 because its cosine scheduler was
  internally advanced during linear warm-up.  Its logs are preserved with the
  suffix `aborted_scheduler_2305`; the final run uses an explicit 1,000-step
  linear warm-up followed by cosine decay over the remaining steps.
- The next nominally exact VQ attempt was stopped immediately after its 10k
  checkpoint when the exported GT translation reached `117.6 m`.  The VQ
  dataset accepted `motion_rep=raw` but silently instantiated the legacy
  velocity/root-feature normalizer, while the geometric loss and exporter used
  raw-translation statistics.  Its checkpoints, logs, samples and 5k full-test
  reconstruction score are preserved under names containing
  `normalizer_mismatch_aborted`; none are valid results.  The dataset and MGT
  tokenizer now select raw statistics explicitly, a startup equality check
  guards dataset/loss statistics, and a regression test checks raw and velocity
  branches against their respective files.  The replacement run starts from
  seed 123 at update zero.
- The first final-VQ MGT launch reached its first BF16 forward but performed no
  optimizer update: projected token features were BF16 while the shared
  special-token embedding used by masked inputs remained FP32, and PyTorch
  rejected the indexed assignment.  The assignment now casts the special
  embedding to the projection dtype.  A CPU BF16 regression and a full-size
  GPU BF16 forward/backward smoke pass (`loss=9.06029`, logits BF16, peak
  allocation 606 MB).  The zero-update timestamp directory is retained for
  provenance; final checkpoint selection ignores incomplete same-name runs and
  requires exactly one directory containing `latest-300000.ckpt`.
- The pre-fix CoShMDM NaN smoke checkpoint is explicitly named
  `smoke_nan_before_padding_fix.ckpt` and is not a valid evaluation checkpoint.
- The first stable CoShMDM run reached logged step 21,620 before a budget audit
  found that its unmatched 569th micro-batch was carried across epoch
  boundaries.  That would create 569,000 rather than the physical-batch-64
  contract's 568,000 updates.  The complete run directory is preserved as
  `CoShMDM-InterX-fair-seed123-bs32x2_aborted_cross_epoch_accum_step21620` and
  excluded; the final run restarted from seed 123/update 0 with explicit
  epoch-boundary discard.
- The superseded epoch-boundary-correct predecessor's first checkpoint at
  optimizer step 20,000 passes the independent Lightning-state audit: all 346
  model and 588 optimizer floating tensors are finite, the checkpoint records
  epoch 70 and scheduler epoch 70, and its SHA-256 is
  `bfb4a20f...fb28e44`. The complete audit is
  `output/coshmdm_corrected_20k_checkpoint_audit.json`. It does not belong to
  the active `...fair-statsfix...` experiment and is retained only as a health
  diagnostic for the predecessor that was later excluded for mixed statistics.
- The same superseded predecessor passes the exact 300,000-step audit at epoch 1,056: all 346
  model and 588 optimizer floating tensors are finite, all 196 optimizer
  parameter-state entries are present, and the checkpoint SHA-256 is
  `912c4bae51a13284d06d029ca66b4f6bdb27de21238dafaac0836fa7dc8ce5af`.
  The 2,701,618,150-byte checkpoint contains one optimizer, one scheduler
  (`last_epoch=1,056`) and Lightning loop state.  The complete audit is
  `output/coshmdm_corrected_300k_checkpoint_audit.json`.
  This remains a superseded health milestone; the active stats-fixed run must
  pass its own 20,000-step audit, and only its audited 568,000-step terminal
  state can enter the fair table.
- The normalization-correct run that first reached 5k was subsequently found
  to have one extra single-sample EMA update: its visualization forward pass
  used `no_grad()` but had not switched the quantizers out of training mode.
  It is preserved with suffix `ema_export_contaminated_aborted_7276`; its
  `0.15955 m` full-test MPJPE is useful only as a scale diagnostic, not as a
  valid model result.  Visualization export now uses a mode-restoring
  `evaluating(...)` context, and a regression test verifies that training-mode
  side effects are disabled and the original mode is restored.
- The EMA-safe replacement's 5k checkpoint has now been evaluated on all 1,696
  frozen official-test requests.  Its MPJPE is `0.159550 m` (body
  `0.130510 m`, hand `0.183607 m`, root translation `0.107625 m`); active
  Body/Hand/Root codes are `3,089/2,287/1,528`, with perplexities
  `1,846.99/1,671.92/910.62`.  This is an early-training reconstruction
  diagnostic, not the final 150k result.  A paired checkpoint audit against
  the excluded run found bitwise-identical optimizer, scheduler, and all
  non-quantizer tensors, while only the three codebooks and their explicit EMA
  state differed.  This localizes the old discrepancy to the visualization
  side effect rather than nondeterministic retraining.  Machine-readable
  results are in `output/vq_paper_eval/net_iter_000005000_reconstruction.json`
  and `net_iter_000005000_export_side_effect_audit.json`.
- At 10k, the same 1,696-sample audit improves MPJPE to `0.144245 m`
  (`-9.59%` relative to 5k), with body `0.117370 m`, hand `0.166559 m`, and
  root translation `0.095090 m`.  Active Body/Hand/Root codes rise to
  `3,667/2,480/1,785` and perplexities to `2,142.53/1,818.72/1,063.12`.
  This confirms continued reconstruction and code-usage improvement without an
  early collapse; it remains an intermediate diagnostic rather than the final
  150k result.  The checkpoint and all frozen inputs are SHA-256-bound in
  `output/vq_paper_eval/net_iter_000010000_reconstruction.json`.
# Online W&B rule for all future training

Effective 2026-08-31, every newly launched training experiment must use W&B
`online`. The fail-closed policy is frozen in
`configs/online_wandb_training_policy.json`. It requires training loss and
task-appropriate accuracy/perplexity, learning rate, gradient/weight norms and finite-value counts,
throughput and peak GPU memory, plus validation loss, validation accuracy,
FID, R-Precision Top-1/2/3, Matching and Diversity. Multimodality is logged
when the validation protocol produces it. Telemetry is sampled every 50
optimizer steps to avoid a full parameter traversal on every update. The
gradient and weight telemetry includes a global L2 norm and global maximum
absolute value over every trainable parameter. Effective 2026-09-02, the same
L2/max-absolute/non-finite/value-count and gradient-to-weight ratio diagnostics
are also emitted for every trainable top-level child module under
`train/modules/<name>/`; the global values remain authoritative for root-owned
parameters. Gradients are measured after
backward/AMP unscale and before clipping or the optimizer update; weights are
measured at the same pre-update event. Every online training manifest records
a stable W&B run ID; strict checkpoint resume must reuse that ID. A completed
model is result-eligible only after its remote history reaches the declared
final optimizer step with every profile-required train/validation metric
present and finite. This applies to both 300k shape-augmentation MGT arms.
Warm-up audits keep their stability window immutable and expose subsequent
gradient/weight/loss/accuracy events in a separate `full_run_metrics` section,
so later training drift cannot be hidden or incorrectly reclassified as a
warm-up failure.
When gradient clipping is enabled, clipping is applied on every optimizer
update and the sparse telemetry events additionally record post-clip global
L2, configured threshold, event trigger flag and cumulative clip-hit rate.
Any non-finite gradient or weight count aborts the run. The VQ entrypoint uses
the dedicated `vq_autoencoder` profile: reconstruction/commitment losses and
three codebook perplexities during training, reconstruction L1/FID on the
official validation split, and the same full gradient/weight telemetry. It
does not fabricate token accuracy or downstream text-retrieval scores.

Runs launched before this rule retain their original provenance: the formal
HHC run is a W&B offline run and the capped CoShMDM run has TensorBoard logs.
They may be uploaded/imported as historical records, but must never be
relabeled as natively online runs.

The first completed post-rule HHC control (`rx72y8g9`) provides a runtime
acceptance test of this rule: all required training metrics reach optimizer
step 50,000, the final validation event follows the final training telemetry,
the remote run state is `finished`, and every finite-value check passes.  Its
machine-readable audit is
`output/hhc_original38_token_window_control_wandb_history_audit.json`.
