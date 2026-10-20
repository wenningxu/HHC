# HHC MModality and MGT shape conditioning

## 09-11｜confidence noise0.5 paired bootstrap（中文记录）

- 已保存 `output/hhc_no_shape_confidence_validation_20260910/log_odds_noise0p5_paired_prompt_bootstrap.json`：544 对齐 prompts、seed90210、10000 次同索引 bootstrap。Slide delta（noise0.5-control）`-0.00238322 m/s`，95% CI `[-0.00613865,+0.00086511]`；contact F1 delta `-0.00446482`，95% CI `[-0.01661966,+0.00682458]`。
- 两个 CI 都包含 0；这是 prompt bootstrap 不确定性，不是训练 seed 方差，也不是 non-inferiority 证据。GPU0 candidate 20k 四臂导出仍待完成。

## 2026-09-11 paired CPU confirmation: log-odds noise 0.5

The frozen 544-request `log_odds_noise0p5_main.h5` was scored on CPU with the same joint evaluator protocol as the control. Control/candidate Full/Body/Hands FID were `0.240967/0.234382/0.210853` and `0.214715/0.229177/0.219536`; matching distances were `3.028678/3.081421/3.084399` and `3.040173/3.108250/3.093625`. Candidate physical sliding/contact-F1 were `0.301087 m/s` and `0.553115`, versus control `0.303470 m/s` and `0.557580`. MM was not measured; these paired CPU results do not promote the arm or establish quality preservation.

## 2026-09-11 true-shape beta factorial diagnostic (val16)

The four requested/zero beta-routing arms completed serially (sessions `17937`, `39158`, `45021`, `75862`; 16 `mm_indices` × 4 repeats). Exact byte-duplicate pair counts were `68/96` requested/requested, `73/96` zero/requested, `68/96` requested/zero, and `73/96` zero/zero. MGT-zero changed `4650/7296` valid token IDs (`0.6373355263`), while requested-MGT/VQ-zero token IDs were identical (`0/7296` changed). This is a small diagnostic only; no official MM/FID evaluation or promotion follows.

## 2026-09-11 matched 20k candidate CPU quality pairing

The full 544-request candidate export (`30914`, exit 0) and CPU scorers (`79740` joint, `87334` physical; both exit 0) completed. Candidate/control Full/Body/Hands FID were `0.422868/0.384182/0.729064` vs `0.558887/0.573045/0.617900`; physical sliding/contact-F1 were `0.333730/0.527806` vs `0.325428/0.539508`. No MM score was run here; this does not by itself establish MModality preservation.

## 09-11｜part thinning 固定 decoder val32 诊断

- `output/hhc_vq_part_thinning_val32_20260911.json` 的六臂 retained token ratio 均为 `0.8342541436464088`。相对 baseline Full MSE：body even/odd `+1.624%/+1.714%`，hands `+4.368%/+5.593%`，root `+0.587%/+0.575%`；被 thinning 部位 MSE 分别为 body `+2.462%/+3.408%`、hands `+8.573%/+11.165%`、root `+20.812%/+86.455%`。
- 仅为固定 decoder token 干预；没有 joint FID、MPJPE、物理或 MM 实测，不能宣称质量通过或解码速度提高。`VQVAE_SEP3_shape` 将三流 concat 后联合解码，单流变化可影响其它部位；body-only 可作为后续训练候选，root-only 虽 Full MSE 较低但 root 自身退化大，不应因 feature 权重低选它。

## 最新用户override（09-11）

- 用户接受当前 VQ 质量“基本可以复现”，解除必须等待严格10项逐项不退化的探索阻断；MModality质量保持重新进入当前 scope，采样、条件训练、token压缩、零shape可探索。
- 原数值/流程来源限制、严格门槛未过证据及下文历史结论保留；不声称 bitwise 或十项全优。MGT PID3537043已恢复，采样原session3616恢复后已544/544导出并exit0，正在接续质量评分；两条复现训练与EMA训练保留暂停状态。

## 09-11验收前历史状态与结论边界（状态已被上方用户override覆盖）

- 用户前置门槛仍是实际复现原MMM VQ。seed123的175k质量验收失败，seed124同配置独立复现已启动；自己的EMA VQ、文本dropout MGT及log_odds_noise0p5导出仍暂停。下文“live/launched/next”均是当时记录，不代表当前授权恢复。
- 本轮仅重读已完成JSON与条件分支，未生成新样本、未读取暂停的半成品H5。待VQ复现通过后才恢复改进实验。
- **全零shape结论不能泛化到有shape输入的MGT。** 完成的zero_beta_main.json明确mgt_shape_conditioning=false、beta_policy=zero、contact_guidance=false；two_person_transformer仅在self.shape为真时使用betas。该对照检验的是同一无shape-MGT下改变解码shape的效果，不检验shape对MGT token采样的作用。
- 该已完成对照：Full MM 0.043913→0.042119，Full/Hands FID 0.240975/0.210871→0.295673/0.278802，未获质量保持的MM改善。保留用户想法：真正有shape输入的MGT上，之后应区分“只将采样条件置零”“只将解码条件置零”“两者置零”，并固定gender、文本、长度与随机种子；这只是后续实验区分，不在复现前执行。
- 采样已证明影响重复率，但并未解决质量约束：强置信度扰动提高MM却损害Hands/物理指标；log_odds_noise1的有利单次结果未通过额外3seed质量复查。因此恢复后不直接扩大强噪声，而先完成已暂停的较低强度对照，再据同设备配对结果决定是否保留该方向。

## Trained temporal compression pilot (September10, live)

GPU0 session80349 sequentially trains fresh stride4 and stride8 VQs,10000
updates each, raw12288/code_dim256/shape8, batch256/window64, same losses,
bf16, seed114514 and integrated1000-step warmup/cosine to10k. Checkpoint roots
`checkpoints/hhc_vq_stride{4,8}_paired10k_seed114514_20260910_rotationfp32_v2`.
This is a short matched learnability/rate-distortion pilot, NOT a continuation
of net175000 or an end-to-end MGT result. Native validation crops depend on
stride; final comparison MUST reconstruct the same frozen544 requests with
exact requested lengths and a common scorer, not compare native valFID.

An actual training failure was found before comparison: initial bf16 stride4
run h91qllqo failed at50-step telemetry (session65467 exit1), so stride8 never
started in that loop. Main reproduced nonfinite gradients in the actual
rotation loss on identical valid rotations: bf16 matrix multiplication makes
cosine/clamping bf16 and rounds1-1e-7 to1, exposing acos's singular derivative.
FP32 counterpart has finite gradients. Minimal precision protection in
`utils/losses.py` covers pair/single/relative rotation matrix products; no
geometry loss was removed. CPU/CUDA regression plus stride tests4/4 pass.
Fresh v2 stride4 run iak9i2o5 now passes real training telemetry through300:
6 finite loss records, all nonfinite-gradient/weight counts zero. Preserve
failed artifacts and do not treat the failed run's manifest as live status.

## Paired replication update (September 10, after 13:40 BST)

Length-stratified actual sampling trace completed with observer parity checks
(`output/hhc_sampling_trajectory_length_quantiles_val8_20260910`, exit0).
Eight metric-independent length-quantile requests span40..150 frames,4 repeats.
Of48 pairs, final identical token sequences: control42, noisy logprob12,
noisy logodds20. First shared committed-value differences occur at stages
140..142,1..147,24..129 respectively (zero-based150 cascade stages).
Nonargmax accepted totals are8/3000,21/3000,8/3000 respectively. Main
independently recomputed using valid token lengths and ordinary token IDs.
This extends the order/context diagnostic beyond the first all150-frame
sample; it does not establish semantic quality or a population effect size.
Per-prompt follow-up: G051T006A031R008 (96 frames, standing/seated thumbs-up)
has all6/6 final pairs identical in ALL three policies. Its caption is present
and nonempty. G039T003A012R018 (mutual pointing) and G018T001A008R016
(pulling another person forward) remain6/6 identical under logodds noise,
while logprob noise reduces each to3/6. Thus order randomization alone does
not universally remove collapse; do not attribute remaining repetition to
missing text or short duration on this evidence.

The original log_odds_noise1 quality improvement is NOT robustly reproduced.
Same-device paired main validation, 544 requests each, frozen settings:

| Seed | Arm | Full FID | Body FID | Hands FID | Body Matching | Full R3 |
|---|---|---:|---:|---:|---:|---:|
| 500124 | control | .242118 | .239069 | .213049 | 3.075038 | .766544 |
| 500124 | log_odds_noise1 | .256086 | .240489 | .232299 | 3.113477 | .748162 |
| 500126 | control | .245737 | .234425 | .212885 | 3.070654 | .766544 |
| 500126 | log_odds_noise1 | .229256 | .246098 | .222834 | 3.103758 | .744485 |
| 500125 | control | .244233 | .234419 | .215346 | 3.072581 | .766544 |
| 500125 | log_odds_noise1 | .250695 | .241346 | .240520 | 3.072428 | .755515 |

Source: `output/hhc_log_odds_paired_main_validation_20260910/*seed*.json`.
Do not promote or expand formal MM based on the original favorable FID alone.
All three additional seeds are complete (session28490/30441 exit0); all
three worsen Body/Hands FID and Full R3. These are screening replications, not a formal
noninferiority test or an estimate of training-seed variation.

Next training intervention launched13:55 BST: joint word/global text dropout
probability0.2, matched to historical global-only probability0.2, no shape,
same historical all-valid loss/VQ/cache/model/masking/optimizer/seed. Fresh
50k diagnostic, total300k schedule, W&B781f0cgp, session78308 onGPU1.
Initial batches and actual manifest verified, no quality result yet. Training
configuration/manifest/resume support added without modifying shared sampler;
default probability and evaluation behavior remain unchanged. Four dedicated
tests include actual masking and strict resume checks. Compare against
September7 no-MGT-shape v4 corresponding checkpoints, not the unrelated
September1 all-valid control.

Physical scorer now optionally saves primary per-prompt sufficient counts
with `--save-per-sample`; assertions verify their sums reproduce pooled
headline counts. Original seed500123 control/candidate files are
`output/hhc_no_shape_confidence_validation_20260910/{control,log_odds_noise1}_physical_per_sample.json`.
Paired bootstrap of544 prompts,10000 draws, NumPy default_rng seed90210,
recomputing pooled F1 and sliding ratios in each draw gives candidate-minus-
control percentile95% CIs: contactF1[-.0183749442,+.0048240541],
sliding[-.0045294800,+.0023567145]m/s. Both include zero; this is NOT evidence
of noninferiority. Main independently recomputed headline equality and CIs.

Failure-selected skeleton figures (NOT representative quality examples) are
in `output/hhc_log_odds_contact_failures_20260910/figures`, renderer exit0.
G048T001A006R005 waving gains147 false-contact frames; its actors appear
closer/partly overlapping in the candidate. Massage G048T005A026R010 loses58
true-contact frames; assisted-walking G058T003A019R027 loses53. Sparse views
suggest relative-placement/coordination problems, not proven mesh penetration
or an established causal mechanism. Selection explicitly records diagnostic,
metric-dependent selection. Next distinguish placement from temporal contact
coordination; do not relabel joint contact as mesh penetration.

## User override: sampling is in scope (2026-09-10)

Sampling improvements are explicitly part of task 1. Common-50/T0.5/CFG3.5
remains the reference, but sampling algorithms, remasking, confidence ranking,
temperature/CFG schedules and step budgets may change in declared experiments.
Assess quality, MM, duplicate rates and runtime jointly. This supersedes the
older sampling restrictions below; the currently launched paired controls
remain unchanged.

September10 completed sampling evidence (single frozen validation seed, not
formal20-replication test): confidence-noise1/all changes Full/Body/Hands MM
from0.043913/0.047964/0.083007 to1.160336/1.228393/1.814799; duplicate-pair
rate falls0.678545→0.019818. Full FID improves0.240975→0.203193, but Hands
FID worsens0.210871→0.265895, sliding0.303470→0.313003m/s, contactF1
0.557580→0.541148. This isolates a substantial sampling contribution to
collapse, but is NOT yet a quality-preserving solution. Root-only noise1 is
complete (Full MM0.962233, Full FID0.204547, but Body/Hands FID and sliding
worsen). Lower-strength sweeps are complete: all-part noise0.25 gives
Full/Body/Hands MM1.028576/1.094035/1.617258 but HandsFID0.270148 and
sliding0.312578; root-only noise0.25 gives MM0.958262/1.108113/1.502831,
FullFID0.200182 but HandsFID0.233905 and sliding0.309054. No promotion.
Effectively argmax proposals (T0, noise1) retain almost all the MM gain,
supporting commitment-order randomness as a major mechanism.
Source directory: `output/hhc_no_shape_confidence_validation_20260910`.

At12:39 BST new independent ranking/schedule grids are live: GPU0 session31601
tests stable selected log-odds with noise0/1; GPU1 session24378 tests strict
root-complete→body-complete→hands-complete with noise0/1. Preserve50 steps per
part and all other control settings. Defaults pass golden output/RNG and
runtime checks; changed contracts are explicit. Do not change live sources.
Twenty-step deterministic sampling completed but did not preserve all quality:
FullFID0.247113, Matching3.051890, MM0.051785; repeated-motion rate0.672364.
Full/Body/Hands and physical metrics remain the joint screening criteria.

Completed-token check (100 prompts x11 repeats,55 pairs each; valid positions
only) confirms root-only noise propagates into the other token streams, not
just decoded root translation. For root_noise0p25, body/hands/root have
1620/1590/851 identical stream pairs out of5500; mean token disagreement
is0.430589/0.445184/0.465002. All-three-stream identical pairs=580, exactly
the generated-motion duplicate count. Main-agent recomputation checks valid
IDs are0..12287 and excludes END/PAD. This measures categorical variability,
not geometry distance, semantic diversity or a quality pass.

The released joint evaluator's `correct_representation_motion` selects the
last30 FK joints for Hands without wrist-centering. Thus HandsFID changes
can reflect global root/body placement as well as finger articulation; retain
the official score unchanged and avoid calling it a finger-only error metric.

A translation-independent local-finger diagnostic uses joints25:55, both
actors separately, valid adjacent frames and30fps. Convert each6D rotation
with existing rotation_6d_to_matrix, then angle=acos(clamp((trace(R_next
R_prev^T)-1)/2,-1,1)). Each arm contains4,228,920 joint/actor transitions.
Mean/p95 angular speed (rad/s): rawGT0.611751/2.207702;
control0.469966/1.584520; noise1-all0.496035/1.675720;
root_noise0p25=0.493955/1.679367; log_odds0.470043/1.582147;
partwise0.467887/1.584113. This does not prove realism, absence of jitter or
semantic diversity; speed is not a monotonic quality objective. An initial
child-agent axis error reported half the transitions and near-zero speeds;
those numbers are rejected. Main independently recomputed GT/control, then
the child reran all arms with explicit [T,30,2,6] actor-preserving reshape.

The paired CPU all55-pair audit confirms the gain is broad: Full distances
increase for100/100 prompts, median per-prompt mean0.007598→1.225320,
and the highest5% prompts account for15.22% rather than44.02% of total
distance. Do not replace the headline metric with this larger all-pair value.
The released `src/evaluation/metrics.py:77` independently draws two arrays
of10 indices, so self-pairs are possible. With this frozen11-repeat request's
metric seeds, Full/Body/Hands include3/2/0 self-pairs respectively. This partly
explains the headline/all-pair gap and limits cross-wrapper interpretation.
Keep the released metric unchanged for comparability; promising settings need
the original30-repeat protocol and paired replications, not a renamed metric.

September 10 physical screening: use the existing joint scorer with explicit
`--floor-policy valid_frames` (protocol `interx_official_v1_physical_joint_v2`).
The historical v1 floor estimate included padding frames; its default remains
available for audited historical replay, not for new sampling comparisons.
The regression test changes only padded foot heights and verifies the corrected
floor is invariant (`tests.test_physical_valid_floor`). Joint sliding/contact
are supplementary diagnostics, not a substitute for mesh penetration or
shape-aware foot geometry. All-zero beta needs the latter before a physical
quality claim, because the joint scorer uses the original subject skeleton.

Primary-source follow-up: MaskGIT's official
[parallel decoder](https://github.com/google-research/maskgit/blob/main/maskgit/libml/parallel_decode.py)
already uses log-probability plus annealed Gumbel noise for commitment ranking.
Our noise arm adapts this established idea to HHC's three-part cascade; it is
not claimed as a new general sampling algorithm or byte-faithful MaskGIT
reproduction (HHC keeps its own count schedule and proposal temperature).

[MaskBit](https://arxiv.org/abs/2409.16211) motivates a separate, longer-term
representation arm: structured quantized tokens rather than arbitrary
categorical IDs. Its reported results are for images, not evidence of gains
on HHC. HHC already uses detached VQ codebook vectors as its ordinary-token
input embeddings (`CrossCondTransBase`); adding those is not a new arm.
Instead consider geometry-aware output targets or factorized quantization
only after reconstruction and geometry checks; do not merely
reinterpret existing arbitrary code IDs as meaningful bits.

The first geometry probe reuses `tools/probe_vq_temporal_compression.py
--intervention code_neighbors`: nearest-other-code replacement at all tokens,
nearest replacement at10% of positions, and random-other-code replacement at
the identical10% positions. It measures fixed-decoder reconstruction/delta MSE
on the same32 validation crops, not semantic multimodality. No MGT, training
cache or production sampling policy is changed by this diagnostic.

Completed result: `output/hhc_vq_code_neighbors_val32.json`. Baseline normalized
MSE0.317181; replacing9.208% of IDs with nearest other codes gives0.322882,
versus0.349758 with random other codes at identical positions. Decoder delta
MSE is0.004818 versus0.050824. Replacing every ID with its nearest neighbor
gives0.368975, so neighbors are meaningfully more local than random codes but
not interchangeable at no quality cost. This supports investigating weak
geometry-aware training targets, not post-generation random token substitution
as a claimed solution. It does not establish increased semantic MModality.

Training-text support audit (train only, actual historical38 loader filtering):
9110 split IDs →9095 cached →8617 loaded (15 missing cache,478 shorter than20
tokens;4718 cropped to37 payload tokens). Their25861 within-motion-deduplicated
captions contain24773 unique exact strings;937 (3.7823%) map to multiple
motions, covering1514 motions (17.5699%). Sources: no-shape v4 training manifest,
its codebook_dir, `dataset/interx/splits/train.txt`, `texts_processed/*.txt`.
This is descriptive conditional-support evidence, not a causal explanation
of collapse or permission to merge semantically different captions.

## New user-prioritized experiments (2026-09-09)

### Completed sampling screen (September10, single validation seed)

GPU joint scores; Full/Body/Hands FID and Matching are lower-is-better.
MM is the unchanged released11-repeat screening metric, not a formal test
result. Exact duplicate rate uses all55 valid-motion pairs per prompt.
Controls from both devices were previously checked identical; subsequent
paired confirmation must retain each arm's same-device control. This table
does not supersede the physical/visual screening described above.

| Arm | Full FID | Body FID | Hands FID | Full Matching | Full MM | Duplicate rate |
|---|---:|---:|---:|---:|---:|---:|
| control | .240975 | .234513 | .210871 | 3.028667 | .043913 | .678545 |
| cfg2p5 | .239811 | .239587 | .224183 | 3.046784 | .053000 | .629091 |
| anneal0p8to0p5 | .242031 | .233383 | .210923 | 3.028160 | .046374 | .675091 |
| temperature0p8 | .242969 | .233684 | .215267 | 3.025871 | .066102 | .561455 |
| matched_text_dropout | .196494 | .233342 | .248330 | 3.060419 | .096480 | .530182 |
| zero_beta | .295673 | .225125 | .278802 | 3.038075 | .042119 | .678545 |
| steps20 | .247113 | .224632 | .216192 | 3.051890 | .051785 | .672364 |
| confidence_all | .203193 | .209604 | .265895 | 3.039249 | 1.160336 | .019818 |
| noise0p5 | .215512 | .202943 | .264532 | 3.036350 | 1.068956 | .027273 |
| noise0p25 | .218422 | .209054 | .270148 | 3.003055 | 1.028576 | .035636 |
| noise1_temp0 | .214321 | .184310 | .279680 | 3.027661 | 1.161310 | .021455 |
| confidence_root | .204547 | .253948 | .261269 | 2.986860 | .962233 | .076545 |
| root_noise0p5 | .206120 | .252473 | .263400 | 2.978846 | .974518 | .084000 |
| root_noise0p25 | .200182 | .245033 | .233905 | 3.003983 | .958262 | .105455 |
| log_odds | .239769 | .230698 | .211173 | 3.054476 | .044627 | .658545 |
| partwise | .258888 | .227015 | .249594 | 3.092749 | .059207 | .613455 |
| log_odds_noise1 | .221012 | .232894 | .205684 | 2.999605 | .394574 | .321273 |
| partwise_noise1 | .252602 | .232150 | .268754 | 3.042183 | 1.030374 | .076909 |

Complete MModality investigation before shape augmentation, InterHuman, or
external baselines. VQ-VAE + MGT remains fixed as the framework; token
compression, quantization, model structure and training/data interventions are
explicitly open for exploration.

User proposal: generate with all-zero shape. Compare requested beta versus
zero beta, preserving gender, text, length, checkpoint and repeat seeds.
`tools/export_hhc_official_interx.py --beta-policy zero` now supports this
explicitly labelled diagnostic and rejects cross-policy resume. First test
the current no-MGT-shape 300k model: its tokens are independent of beta, so
any change isolates the VQ decoder. A shape-conditioned reference additionally
tests MGT sensitivity. Zero beta is the gender-specific model mean, not removal
of the shape branch. Do not vary beta between repeats to inflate MModality.
The paired GPU0 validation comparison completed September10 at11:42 BST in
`output/hhc_no_shape_confidence_validation_20260910`. Zero-beta Full/Body/Hands
MM0.042119/0.058930/0.083426 versus control0.043913/0.047964/0.083007;
duplicate3732/5500 and mean unique2.59 are unchanged. Full/Hands FID worsen
from0.240975/0.210871 to0.295673/0.278802. This setting is rejected for the
current no-MGT-shape checkpoint; it does not test a shape-conditioned MGT.
The redundant CPU jobs were interrupted with SIGINT after GPU completion;
their partial H5 files remain readable under `output/hhc_zero_beta_validation_v1`.
Neither partial CPU output is a completed paired comparison.

Status: the no-shape 300k/20-replication result and repeat-collapse audit are
complete. The masked-only50k diagnostic finished and failed quality selection;
do not continue it to300k. Lower confidence-noise sampling is now running.

## Current observation

The selected no-shape 300k checkpoint was evaluated for 20 complete test
replications under the frozen common-50 sampler. Released Interact2AR joint
metrics are:

| Evaluator | FID | Matching | R@1/2/3 | Diversity | MModality |
|---|---:|---:|---:|---:|---:|
| JointsFull | 0.09811 | 3.12018 | 0.44891/0.65678/0.76804 | 9.22477 | 0.05701 |
| JointsBody | 0.10262 | 3.17488 | 0.46365/0.66825/0.77656 | 9.04775 | 0.06069 |
| JointsHands | 0.12196 | 3.10812 | 0.41636/0.62529/0.74357 | 8.48953 | 0.06146 |

FID and GT-distance-to-Diversity beat the Interact2AR paper row for all three
wrappers. Hands Matching also passes, but Full/Body Matching and all retrieval
metrics miss narrowly. MModality misses by about 1.36--1.38, so the required
comprehensive gate fails.

This is not an evaluator-scale artifact. The untouched Interact2AR public
README evaluation with GT-first-frame conditioning locally reproduces
MModality 1.3600/1.4168/1.4002 for Full/Body/Hands. The local joint scorer and
released wrapper produce exactly equal text/motion embeddings and retrieval
metrics in the scorer-parity audit. The author-conditioned table and our
unconditional fair table still must not be mixed as a fairness claim.

The no-shape repeat audit finds exact duplicate generation pairs at rate
0.6837. Across 30 generations per prompt there are only 3.57 unique motions on
average (median 2), and median raw pairwise RMS is zero. Distinct repeat seeds
are present and `sample_intra_cas` performs genuine Gumbel sampling, ruling out
seed reuse and an accidental argmax implementation.

The completed 300k W&B trajectory provides independent temporal evidence.
Diagnostic validation MModality falls from a mean of 0.1970 over steps
10k--50k to 0.0515 over steps 260k--300k. Across all 30 aligned validation
events its Pearson correlation is -0.865 with trainable-weight L2 norm,
-0.780 with masked-token accuracy, -0.740 with visible-token accuracy, and
+0.786 with training loss. Thus lower loss, higher accuracy, and larger weight
norm coincide with lower conditional diversity. The same trajectory shows a
quality/diversity tradeoff (correlation +0.498 with FID and -0.564 with R@1).
These are single-run correlations and are used to prioritize the masked-only
intervention, not as causal proof.

## What the implementation actually does

The retained strong model uses the historical topology and
`historical_mmm_shape_dropout` during training:

- the global CLIP vector is independently zeroed with probability 0.2;
- the two-person beta vector is independently zeroed with probability 0.2;
- gender embeddings remain present when beta is zeroed;
- local CLIP word tokens are not dropped by this historical contract.

The official common-50 exporter intentionally switches inference to
`paper_text_cfg`. Its conditional branch receives text and requested shape.
Its CFG reference branch removes both global and word-level text but retains
the requested beta and gender. Thus shape is controlled in both branches and
CFG amplifies text relative to the same body shape. This is not a missing-shape
or accidentally unconditional evaluation.

Code anchors:

- `models/two_person_transformer.py`: `mask_text`, `mask_shape`,
  `CrossCondTransBase.forward`, and `sample_intra_cas`;
- `tools/export_hhc_official_interx.py`: explicit
  `paper_text_cfg`, `mgt_shape_conditioning`, and frozen common-50 metadata;
- each generation HDF5/score record freezes the resolved checkpoint, shape,
  sampler, CFG, normalization and request hashes.

## Working explanation

At sufficient budget, removing the requested shape improves JointsFull FID
and most retrieval metrics relative to the retained shape-conditioned run,
while worsening JointsHands FID. It does not prevent late conditional
collapse, so the evidence rejects shape conditioning as the sole cause.
Native re-prediction of already selected positions was also tested
with all other sampling parameters fixed and worsened validation FID
0.63626 -> 0.67120 together with Matching and R-Precision.

The remaining training-side hypothesis is that historical all-valid-token NLL
rewards copying already visible tokens and drives excessive confidence as
training progresses. The matched diagnostic changes only this objective to
masked-token NLL. It does not alter the user-fixed sampler, temperature,
sampling steps, CFG, VQ, data, architecture, or mask-ratio distribution.
This arm averages masked NLL within each sample and then across the batch.
HHC Eq. 3 does not specify the variable-length batch reduction; released
InterMask instead uses one global mean across all masked labels. Therefore the
arm diagnoses masked-only supervision but is not claimed to reproduce
InterMask's reduction exactly. If it passes, global masked-token mean remains
a separate controlled confirmation before any loss-parity claim.

## Ordered experiments

1. Finish the active no-shape masked-only 50k single-factor diagnostic. Select
   among 10k--50k only on the frozen official validation JointsFull FID, and
   compare its synchronized FID, Matching, R@1/2/3, Diversity, MModality and
   W&B optimization telemetry with the all-valid control. Continue to 300k
   only if the preregistered quality screen passes and MModality is not lower.
2. Complete the user's requested/zero-beta paired diagnostic, including
   quality and MM, without changing beta across repeats.
3. Test synchronized local/global text dropout matching inference. The
   existing `paper_text_cfg` implementation also changes dropout probability
   from global-only 0.2 to joint 0.1; report this as a conditioning-contract
   intervention, not a probability-matched ablation. Keep the active
   masked-only run unchanged.
4. Explore temporal token compression and span masking inside VQ-VAE+MGT.
   First quantify reconstruction and duration/alignment costs; token
   redundancy alone does not establish a generative benefit. Duration-aware
   run compression and stronger temporal downsampling are distinct arms.
   Span masking can test reliance on repeated visible neighbors before
   committing to a new tokenizer. Compare full quality metrics and exact
   duplicate rates as well as MM; do not select on MM alone.
5. Only after the MModality phase, proceed to public shape donors and faster
   optimization/augmentation, then two-part InterHuman, then external
   baselines, following the user's four-point order. Donor acquisition is
   our task when legally/publicly accessible, not contingent on the user
   supplying a dataset.

## New measured evidence (2026-09-09 evening)

September 10 sampling follow-up: `output/hhc_no_shape_sampling_selection_val8.json`
uses 8 fixed validation samples and 128 categorical draws per sample on fully
masked guided logits (T0.5/CFG3.5). Highest raw-confidence commitment accepts
zero non-argmax candidates out of 1024 decisions for each part; noisy
log-confidence ranking accepts some alternatives. Body proposals include
1642/35200 non-argmax tokens, but all are rejected by highest-confidence
one-position commitment (noisy ranking retains 18/1024, uniform 56/1024).
This isolates a selection bias and motivates stochastic confidence ranking
with annealed noise, not just raising proposal temperature. Only the root
fully-masked context matches the actual first cascade stage; body/hands are
counterfactual contexts, and none of these counts measures end-to-end MM or
quality. Tests: `python -m unittest tests.test_sampling_selection`.

September 10 compression probe: `output/hhc_vq_temporal_thinning_val32.json`
encodes 32 uniformly selected frozen validation crops once, retains one token
per adjacent pair (even and odd phases separately), then expands back to the
original token length before the unchanged shape-aware decoder. It preserves
the final singleton for odd lengths and pads in normalized space. Retained
token ratio is 0.50276; both phases change 31.80% of token IDs. Baseline full
normalized reconstruction MSE is 0.31718; odd-phase thinning increases it to
0.34236. This rejects naive thinning as a lossless shortcut but does not reject
a newly trained stride-8 tokenizer. It reports feature-weighted normalized
errors, not geometric quality or MModality. Test:
`python -m unittest tests.test_vq_temporal_compression`.

On the same 32 validation samples, the first fully masked forward pass at
300k has conditional effective vocabulary about 2.0 per part, reduced to
1.15--1.18 by paper-text CFG at scale 3.5 and temperature 0.5. Its text-free
reference branch has effective vocabulary 74--163. With the historical
global-only dropout contract at inference instead, the reference branch and
guided branch both stay near 2.0. This supports investigating the unseen
local-text-dropout condition; it does not prove higher generation quality.
Artifacts: `output/hhc_no_shape_10k_300k_cfg_entropy_val32.json` and
`output/hhc_no_shape_300k_matched_dropout_entropy_val32.json`.

In the historical38 training cache, exact adjacent repeats are 61.3% body,
46.5% hands and 48.9% root. Ideal run encoding would reduce token counts by
59.5%, 45.1% and 47.5%, respectively, before charging for durations. This is
a lossless storage bound, not a measured generation speedup or MM gain.
Artifact: `output/hhc_no_shape_token_compression_statistics.json`.

The interrupted masked-only 10k checkpoint is preserved separately at
`output/hhc_no_shape_masked_only_recovery/pre_validation_000010000.ckpt`.
Its original MM validation and W&B validation event did not finish. Before
50k promotion, recover this candidate explicitly and align validation by
optimizer step; do not substitute the first available 20k event for 10k.

Temperature and step-count changes are now allowed by the September 10 user
override. A higher MM alone is insufficient: use quality/runtime comparisons
and compare algorithmic improvements against simple temperature/CFG baselines.

## 2026-09-10: paired 10k text-dropout and VQ precision diagnostics

- Joint text dropout p=0.2 candidate 10k export/scoring session 19669
  completed (exit 0), frozen 544 requests, seed 500123, GPU0 scorer.
  Versus historical matched 10k control, Full/Body/Hands FID changed
  0.468289/1.229597/0.818116 to 0.566041/0.381083/0.987637.
  Physical CPU evaluation: sliding 0.291672 to 0.350876 m/s,
  contact F1 0.522846 to 0.539211. This checkpoint does not pass quality;
  no extra MM export or promotion on these results. Training remains active.
  Artifacts: `output/hhc_joint_text_dropout_p0p2_matched_validation_20260910/`.
- Stride4 VQ 5k to 10k degradation also occurs on fixed 64-frame training
  snippets, not just long held-out motions. On 32 fixed snippets per split,
  normalized SmoothL1 train 0.111213 to 0.128474, val 0.174221 to 0.198337.
  This does not establish a pure overfitting explanation.
- CPU bf16 nearest-neighbor diagnostic uses identical encoded vectors and
  codebook for bf16/FP32/double distance comparisons. At 10k, body/hands/root
  bf16 ID disagreement with double is 42.97%/31.45%/15.04%; FP32 is 0%.
  Encoder latent RMS grows markedly from 5k to 10k. End-to-end bf16 val
  SmoothL1 still worsens from 0.173789 to 0.195948: numerical mismatch alone
  does NOT explain away the regression. GPU confirmation is a separate probe.
  Stored diagnostic JSONs: `output/hhc_vq_stride_compression_paired10k_20260910/`
  `fixed64_train_val_fp32_diagnostic.json` and `fixed64_cpu_precision_probe.json`.
  No live shared quantizer/training source was changed. Stride8 training
  session 80349 remains live; logged losses at step 1800 are finite.
- GPU confirmation session 42160 completed (exit 0), RTX4090 CUDA:0,
  TF32 disabled, same 32 val snippets. At 10k bf16 distance ID disagreement
  versus double is body/hands/root 41.80%/27.73%/14.65%, FP32 0% throughout.
  End-to-end bf16 SmoothL1 still worsens 0.173743 to 0.194642. Artifact:
  `fixed64_gpu_precision_probe.json` in the same compression output directory.
  Next causal test should isolate FP32 quantization during training in a
  separately versioned matched pilot after the current paired run, rather
  than modifying live shared modules or claiming an inference-only fix.
- Follow-up quantizer-only inference counterfactual completed, session 29801
  exit 0. Reproducible entry: `tools/probe_vq_quantizer_precision.py`; it uses
  the saved 32-snippet selection/hash, patches only the isolated model's
  quantize method, and asserts finite output and unchanged codebooks.
  At 10k, GPU SmoothL1 FP32 / bf16 / bf16-with-FP32-distance is
  0.198341 / 0.194642 / 0.198360; root L2 is
  0.171469 / 0.178146 / 0.172814 m. This inference-only intervention does
  not repair reconstruction. Saved as
  `stride4_step10000_quantizer_only_counterfactual.json` in the compression
  output directory. Training-time quantization remains an untested hypothesis.
- Corrected training-trend interpretation after direct log verification:
  stride4 v2 training Recons is 0.11413 at 5k and 0.13013 at 10k;
  total loss 0.19007 to 0.37965, raw commitment 2.44741 to 10.52260
  (weighted by 0.02). Recons temporarily improves through 6k/8k before
  worsening. Thus this is not a continued training-loss improvement versus
  evaluation-only decline. Geometric log values are already weighted by
  their configured coefficients; they are not unweighted component losses.
  Source: `output/vq-interx-sep/2026-09-10-14-05-25_hhc_vq_stride4_paired10k_seed114514_20260910_rotationfp32_v2/run.log`.
- Stationary-input EMA isolation completed, session 10526 exit 0;
  `tools/probe_vq_ema_precision.py` invokes the actual QuantizeEMAReset
  initialization/update methods. Synthetic fixed bf16-representable centroids,
  32 codes, 96 dimensions, four identical observations per code, fixed IDs,
  mu=0.99, seed=90210, CUDA:0. Thus exact updates should leave centroids fixed.
  Relative codebook L2 error at 1000 updates:
  native bf16 accumulator 0.2653306;
  FP32 accumulator with bf16 aggregation 0.00165319;
  full FP32 update 0.00000266303. FP32 stationarity and all-finite assertions
  pass. Artifact `stationary_ema_precision_probe.json` in compression output.
  This proves a numerical drift mechanism in the controlled update, not the
  magnitude of its effect on real generation quality.
- Actual stride4 v2 10k checkpoint was loaded independently: all three
  saved `quantizer_training_state.*.code_sum` tensors ARE bf16, while
  counts/usage are FP32. Therefore the tested accumulator path is present in
  the real run, not just hypothetical. Full EMA state is saved/restored by
  existing `utils/quantizer_state.py`; do not assume it is absent merely
  because code_sum is not a registered buffer.
- Next training experiment priority is revised by this evidence: isolate
  FP32 EMA accumulation/update first, retaining existing distance selection,
  then test FP32 nearest-neighbor distance separately if needed. Do not lump
  both interventions into an allegedly single-factor experiment. Use fresh
  matched-budget runs for causal comparison; retain current stride4/8 paired
  run unchanged until terminal. Neither numerical probe is an MM success.
- Implemented isolated accumulator-only training entry
  `tools/train_vq_fp32_ema_accumulator.py`. It reuses the existing trainer and
  original quantizer init/update; casts only persistent code_sum to FP32.
  Native aggregation, nearest-neighbor selection and initialization RNG remain
  unchanged. Intervention identity and launcher/quantizer SHA256 enter resolved
  arguments, hence W&B/checkpoints/manifest contract. Fresh-only guard rejects
  resume. Two tests passed (stationarity/init/RNG/native selection; provenance
  and resume rejection), exit 0. No live shared source modified.
- Queued session 84406: waits for confirmed live baseline outer PID 3547884,
  then requires stride8 manifest status `complete` before launching stride4
  accumulator-only fresh 10k. Run name/checkpoint directory:
  `hhc_vq_stride4_fp32emaaccum_paired10k_seed114514_20260910_v1`.
  Same seed 114514, batch256/window64, code12288/dim256, down2/stride2,
  width512/depth3, bf16, shape-mask0.1, commit0.02, LR2e-4/wd1e-6,
  integrated warmup1000/cosine10000, 5k frozen-evaluation milestones.
  It is QUEUED, not yet trained or evaluated. Preserve/poll session84406;
  never restart merely because the wait emits no output.
- Accumulator override integration check session 65645 completed (exit 0):
  actual stride4 5k three-part model + restored EMA state, CUDA bf16,
  synthetic batch2 x64 frames, forward/backward through reconstruction and
  commitment loss. Output/all parameter gradients finite; all three EMA
  accumulators serialized as FP32 after update. Loss 0.49831924 is a smoke
  check, not a quality metric. Shared training/model code remains unchanged.
- Baseline stride8 session80349 passed step5216 and produced its 5k snapshot;
  frozen544 CPU reconstruction/joint/physical evaluation delegated to the
  existing low-cost metrics agent. MGT session78308 completed native 10k
  MM diagnostic (legacy-wrapper MM0.6344, not official joint protocol) and
  resumed training. Do not mix this number with frozen official MM metrics.
- Stride8 5k frozen544 reconstruction and physical outputs are now available.
  Main agent independently verified H5 completion, all finite values, and
  exact request names/lengths (71026 valid frames); matched request/GT hashes
  with stride4 5k. Stride4 -> stride8: MPJPE 0.165687 -> 0.167933 m,
  raw RMSE 0.0941573 -> 0.0971948, root L2 0.0968187 -> 0.1005080 m;
  sliding 0.327472 -> 0.315722 m/s, contact F1 0.681648 -> 0.672917.
  Mixed early rate-distortion evidence: slightly worse reconstruction/contact,
  less sliding. No MGT/MM or speed claim follows from this VQ-only pilot.
  Joint-evaluator score was still pending at this entry; wait for that and
  matched10k results before deciding compression direction.
- Stride8 5k evaluation terminal handles: reconstruction26636, joint18789,
  physical13359, all exit0. Main verified joint-evaluator provenance EXACTLY
  equal between stride4/8 and both scores status complete.
  Full/Body/Hands FID stride4 -> stride8:
  0.308566/1.337971/0.156177 -> 0.361346/1.488118/0.149126.
  Matching: 3.489136/3.845617/3.451415 -> 3.573166/3.862283/3.452508.
  All three matching distances worsen. Tokens per part 17895 -> 9019
  (49.60045% reduction), aggregate codebook perplexities increase, but those
  are NOT conditional multimodality. Do not promote to costly stride8 MGT
  on this early mixed VQ evidence. Await matched10k and numerical-stability
  pilot. Current baseline session80349 verified live through6353 updates;
  queued EMA session84406 verified waiting, not yet training.

## 09-11｜真正 shape-conditioned MGT 零 shape 四臂筛查登记（尚未运行）

- 候选为既有选定 MGT `output/t2i_trans/2026-09-02-11-49-08_hhc_historical_plus_paper_mask_ratio_raw12288_300k_seed114514_v2/model-000260000-0.1246.ckpt`（SHA256 `58b976b8cb579e4a687b9614789bbf088b73e4496eb6b6c06ef7190e80f0d072`），配原 VQ 175k；manifest 明确 `shape=true`。四臂固定为 MGT/VQ 的 requested/requested、zero/requested、requested/zero、zero/zero。
- 固定 gender、text、length、seed600123、50 steps、temperature 0.5、CFG 3.5、`paper_text_cfg`，关闭 contact guidance；第一阶段冻结 `mm_indices` 前16条、4重复并保存 token_ids，只诊断 token 重复与解码差异，不称官方 MM/FID，也不在小样本宣布质量不降。筛查有价值后再做完整同协议评测；当前正在拆分两端 beta policy 并测试，尚未生成、禁止新运行。

## 09-11｜interleaved cascade 分阶段 replay 诊断计划（尚未运行）

- 实现检查确认 interleaved cascade 中，关闭 root noise 仍不能阻止 body/hands 对后续 root logits 的影响；VQ 是联合 decoder，held root tokens 也不等于固定世界坐标 root 几何。
- 计划对 paired-repeat control（非 GT）做 per-stage replay：固定相同8个长度分位×4 repeats 的 control root token trajectory，仅在剩余 streams 应用 noise1/log-odds，保存 archived traces；使用隔离脚本，不改共享模型或训练。
- 当前仍在实现审查，未 GPU launch；不宣称 MM 增益或生产方法。20k candidate 评分另行进行中。

## 09-11｜root replay 完成（token/解码诊断，非质量评测）

- session `50120` exit0；四个 repeat 均通过 archived control token equality、fresh self-replay token 与 CPU/device RNG 校验，paired root IDs 全部匹配 control。结果见 `output/hhc_root_replay_length_quantiles_val8_20260911/summary.json`。
- identical token pairs body/hands/root=`28/37/43`；duplicate motion pairs `28/48`（旧 control `42/48`），unclamped log-odds `20/48`（交叉计数仍核对中）。fresh decoded root feature RMS=`0.002751/0.001564/0.002359/0.001980`，单位待按 root translation mask 复核；仅为 root token/解码诊断，不宣称 FID、MM 或质量提升。

## 09-11｜root replay full-quality prereg（尚未运行）

- 新评分开关 `--root-replay-control` 默认关闭；预注册每 batch 两次 pass：probability/noise=0 control 后接 log-odds/noise=1 candidate，candidate 按 stage replay control root。恢复 control batch RNG chain，candidate 不消耗下一 control batch RNG，control/candidate token IDs 全部归档。
- 完整配置固定为544、seed500123、batch32、50 steps、T=.5、CFG3.5、no-shape 300k×4、原 VQ175k、无 contact、requested betas；后续 CPU joint32/physical16 对比既有 CPU-confirmed control。首轮32条 self-replay parity gate 由 overnight 先跑；尚无 MM/质量结果。
- 两 pass 与 debugger overhead 需先测成本，不能据此宣称生产方法或质量改善；共享训练/模型未修改。

## 09-11｜root replay 32 条 parity gate 完成

- session `5488` exit0；root 独立核对旧 control 前32条两组 motions 与 candidate/control token IDs 均 `np.array_equal`，`maxabs=0.0`。`output/hhc_root_replay_validation_20260911/source/` 四个 source 文件 SHA 全部匹配归档值。
- frozen544 的 noise1/log-odds full replay 当前由 PID `3762168` 在 GPU0 运行（`/usr/bin/time` PID `3762167`）；原始 agent session 仍待回报，不伪造其句柄，也没有质量指标。frozen 配置未变，10 个 unittest、self-check 与 5 个 plain checks 已通过。

## 09-11｜text-dropout native validation 10k/20k 日志核对

- 原生训练验证日志记录 Full native FID 10k=`.6700`（best）、20k=`.7886`；20k native MM diagnostic 为 `.4979`，10k 为 `.6344`（100-subset×11）。20k 同段 subset FID=`1.1259`，不是 Full FID；首个完整评估的 MM=`0.0000` 是 placeholder。
- native training metrics 与 frozen joint metrics 协议不同；不以此覆盖或重命名已有 20k frozen joint 质量结果。

## 09-11｜root replay MM 监控交接（评分未完成）

- root 的 functions cell `1298` 独占监控原始 export session `35016`，每60秒检查且不唤醒模型；不得由其他调用重复轮询，使用 `functions.wait` 获取新输出。
- export 完成后同一 cell 将核验 100×11 finite/name/length/root IDs，再运行 CPU root-replay MM scorer（repeats11、seed600123、batch32、log-odds/noise1），输出 `root_replay_log_odds_noise1_mm_cpu.json`。当前无评分完成结论。

## 09-11｜root replay full544 MM/质量评分完成

- root MM cell `1298` 已完成：export session `35016` exit0（1674.82s），CPU scorer exit0；候选 H5 SHA=`4e6d4373d9d463b19610ded5c1a9d3ae8f2bc6efe447ae2ed88ab5ee12da32b4`，requests/order/runtime/evaluator/checkpoints/sampling 经独立核验一致。
- MM：候选 Full/Body/Hands=`.079413/.114994/.146442`，control=`.043909/.047873/.083011`；exact duplicate rate 候选 `.482`（2651/5500），control `.678545`（3732/5500）。这是 MM/重复率改善，不等于整体生成质量通过。

| 指标 | control | root-replay candidate |
|---|---:|---:|
| Full FID / matching / R1-3 / diversity | .240967 / 3.028678 / .433824,.643382,.766544 / 9.646897 | .230924 / 3.038019 / .431985,.643382,.759191 / 9.645536 |
| Body FID / matching / R1-3 / diversity | .234382 / 3.081421 / .454044,.647059,.781250 / 9.208961 | .236343 / 3.077199 / .459559,.645221,.781250 / 9.199762 |
| Hands FID / matching / R1-3 / diversity | .210853 / 3.084399 / .422794,.613971,.735294 / 8.839807 | .211511 / 3.081597 / .424632,.610294,.735294 / 8.852642 |
| 滑步 m/s / contact F1 | .303470 / .557580 | .304015 / .558232 |

- 因此严格逐项质量门槛未通过（Full/Body/Hands FID、滑步等存在退化）；后续 seed500124–126 仅待 root 排队，未启动，不宣称已改善或完成多 seed 结论。

## 09-11｜root replay 三 seed 队列已创建（尚未运行）

- native queue session `64522` 等待 30k pipeline PID `3784161` 退出；随后每600秒检查，串行执行 seed500124/125/126 的 544×1 root-replay、CPU joint32 与 valid_frames physical16。任一步 set-e 失败即停，已有 H5 拒绝覆盖。
- 当前仅排队，尚无新 seed 结果；control 三 seed 由 overnight_metrics 负责。

## 09-11｜root replay seed500124 完整评分完成（未晋级）

- seed500124 candidate H5/CPU joint/physical 已完成；H5 SHA=`4c75ba606e4b8e420732ca2e146f6d764af985b1e47fa9f88d642fbc8f2cb058`，544 finite、同 eval/request-GT，candidate root-replay token IDs 与旧 control 全相等且 root IDs 相等。首 seed set-e 链路通过；session `64522` 尚未结束，seed500125 已生成文件。

| 指标 | control | seed500124 candidate |
|---|---:|---:|
| Full FID / matching / R1-3 / diversity | .242136 / 3.040499 / .439338,.647059,.766544 / 9.637486 | .237796 / 3.035654 / .431985,.637868,.761029 / 9.640163 |
| Body FID / matching / R1-3 / diversity | .238888 / 3.074780 / .457721,.641544,.777574 / 9.197988 | .235066 / 3.084247 / .459559,.639706,.781250 / 9.197407 |
| Hands FID / matching / R1-3 / diversity | .213007 / 3.079741 / .422794,.612132,.735294 / 8.842059 | .213476 / 3.080662 / .426471,.613971,.731618 / 8.854974 |
| 滑步 m/s / contact F1 | .303542 / .559557 | .303857 / .559169 |

- 结果为混合变化，未晋级；本 seed 没有新增 MM 评分。native 40k Full FID `.5762`、matching `3.6598021984`、R `.41911765/.62683824/.71875`、diversity `8.9921`，30k `.5415` 仍为 native best；native MM 仍运行中。

## 09-11｜root replay seed500125 完整对照已核验（未晋级）

- root 独立核验 544 complete/finite、provenance/eval/GT/requests、H5 SHA=`1925de65cfb019795663b973bbc5e2537a07466833e87c2666bef4022bd02e01`、candidate root-replay token IDs 与旧 control 全相等及 root IDs 相等，均 PASS。

| 指标 | control | seed500125 candidate |
|---|---:|---:|
| Full FID / matching / R1-3 / diversity | .244283 / 3.036593 / .435662,.645221,.766544 / 9.647035 | .248630 / 3.040329 / .437500,.636029,.761029 / 9.646081 |
| Body FID / matching / R1-3 / diversity | .234251 / 3.072344 / .452206,.647059,.783088 / 9.197659 | .224001 / 3.072072 / .454044,.647059,.783088 / 9.173633 |
| Hands FID / matching / R1-3 / diversity | .215322 / 3.077598 / .422794,.617647,.740809 / 8.840570 | .219286 / 3.073315 / .426471,.615809,.735294 / 8.848769 |
| 滑步 m/s / contact F1 | .303833 / .556776 | .303991 / .559202 |

- 整体仍未晋级；本 seed 无新增 MM 评估。seed500126 正在原队列导出。另一训练线 native 40k 的 MM `.3970`、subset FID `1.9291` 不得与本 frozen root-replay 结果混用。

## 09-11｜root replay 三 seed（500124–126）汇总

- cell `1339` 已完成，session `64522` exit0；三 seed 均通过 544 complete/finite、source/eval/GT/requests、control token/root IDs exact 及 H5/joint/physical 一致性核验。以下为每 seed candidate/control 的 FID、滑步、contact F1：

| seed | Full FID | Body FID | Hands FID | slide m/s | contact F1 |
|---|---:|---:|---:|---:|---:|
| 500124 | .237796/.242136 | .235066/.238888 | .213476/.213007 | .303857/.303542 | .559169/.559557 |
| 500125 | .248630/.244283 | .224001/.234251 | .219286/.215322 | .303991/.303833 | .559202/.556776 |
| 500126 | .245938/.245763 | .229943/.234291 | .219828/.212860 | .303906/.303564 | .559627/.556742 |
| 算术均值 | .244121/.244061 | .229670/.235810 | .217530/.213729 | .303918/.303646 | .559333/.557692 |

- 算术均值中 Body FID 三 seed 均改善；Hands FID 与滑步三 seed 均小幅退化；Full 均值几乎不变。整体不达质量不降，未晋级；MM 提升仍仅是既有 100×11 实验，不扩展为三 seed MM 结论。当前无 cell `1339` 运行。

## 09-11｜posthoc token restoration 解码归因启动

- `tools/decode_hhc_token_restoration.py` 经 root 审核修正并 self-check PASS。smoke8 session `65847` exit0；fresh candidate/control 各8/8 与 seed500124 archive bitwise exact、`maxabs=0`，hands/body 两臂 mixed token IDs、finite/completion/zero-padding 均通过。
- full544 四臂 B1 解码 session `2841` exit0；fresh candidate/control 均544/544、与原 archive bitwise exact（`maxabs=0`），两恢复臂 mixed IDs、names/lengths、finite/completion/raw-zero padding 均 PASS。复用旧 candidate/control 评分，不重评一模一样 motions；这仍是 posthoc 联合解码归因，不是新采样/MM。
- 两恢复臂 CPU joint(batch32)+physical(batch16 valid_frames per-sample) 串行 pipeline session `47722` 已启动，输出原目录 `*_cpu.json` / `*_physical_per_sample.json`，尚无结果。
- 40k session `18879` 已完成 export `544/544`，当前 CPU 评分阶段，未终态；control 验收 session `6435` exit0，候选尚未验收，暂不报告配对质量结论。

## 09-11｜text-dropout 40k candidate 冻结评分完成（未晋级）

- session `18879` 与 root 验收 `94621` 均 exit0；候选 544 complete/finite、names/lengths、同 eval/GT/requests、实际 H5/checkpoint SHA 均通过。候选 H5 SHA=`4af4ea779e21c4a3e14bd6901a6871178cef25c34cbd89bb161a3d3ee057336b`。

| 指标 | control | candidate |
|---|---:|---:|
| Full FID / matching / R1-3 / diversity | .423530 / 3.236657 / .417279,.628676,.727941 / 9.597493 | .379335 / 3.138947 / .443015,.608456,.726103 / 9.480058 |
| Body FID / matching / R1-3 / diversity | .337148 / 3.233866 / .420956,.634191,.751838 / 9.014789 | .284926 / 3.219336 / .452206,.648897,.764706 / 9.031921 |
| Hands FID / matching / R1-3 / diversity | .486570 / 3.219543 / .386029,.590074,.716912 / 8.474455 | .476173 / 3.109669 / .395221,.591912,.705882 / 8.522694 |
| 滑步 m/s / contact F1 | .327072 / .523550 | .342347 / .536920 |

- 三项 FID 与 matching 改善，但滑步恶化，Full R2/R3 与 Hands R3 下降；整体未达质量不降，未晋级。frozen MM 未测；token restoration session `47722` 仍无结果。

## 09-11｜full544 decoder coupling RMS 诊断

- root 读取 full544 四臂 H5，以 valid frames 和 `utils.motion_masks` 的 `HHI_BODY_MASK/HHI_HAND_MASK/HHI_ROOT_MASK_OLD` 聚合，与 fresh control 比较：fresh candidate body rotation6d RMS=`.0113282942553`、hands=`.0192964787273`、root translation=`.00333990590911 m`；restore_hands（body candidate/hand control）=`.0104026786934/.00619219776973/.00279537854854 m`；restore_body（body control/hand candidate）=`.00461624549266/.0182269864797/.00190648275772 m`。
- 该结果支持 decoder stream coupling：固定手部 token 仍有身体 token 引起的手部输出差异，固定 root token 仍有 root 平移变化。仅为耦合诊断，不是 MPJPE/FID 质量改善证据。session `47722` 仍活且无结果；45k checkpoint 已存在、50k 尚无为 overnight 只读核查结果。

## 09-11｜token restoration 两臂消融评分完成

- 原 session `47722` exit2：physical 命令误用 `--floor-height-policy`，被 argparse 拒绝；未修改源码、未重复 joint。restore_hands H5 SHA=`a8eab12d09df18d3018feed2c516e365d91ee14414537086191a63ea461b4b59`，eval/requests/GT 一致性已独立核验。

| 指标 | control | root-replay candidate | restore_hands |
|---|---:|---:|---:|
| Full FID / matching / R1-3 / diversity | .242136 / 3.040499 / .439338,.647059,.766544 / 9.637486 | .237796 / 3.035654 / .431985,.637868,.761029 / 9.640163 | .238176 / 3.039632 / .435662,.637868,.759191 / 9.639022 |
| Body FID / matching / R1-3 / diversity | .238888 / 3.074780 / .457721,.641544,.777574 / 9.197988 | .235066 / 3.084247 / .459559,.639706,.781250 / 9.197407 | .233578 / 3.087422 / .455882,.637868,.781250 / 9.207550 |
| Hands FID / matching / R1-3 / diversity | .213007 / 3.079741 / .422794,.612132,.735294 / 8.842059 | .213476 / 3.080662 / .426471,.613971,.731618 / 8.854974 | .212808 / 3.083155 / .430147,.610294,.731618 / 8.851320 |
| 滑步 m/s / contact F1 | .303542 / .559557 | .303857 / .559169 | .303709 / .559102 |

- restore_hands joint 仅显示部分 FID变化，matching/R仍有退化；session `68433` 已 exit0，restore_hands physical 与 restore_body joint/physical 全部完成。restore_body H5 SHA=`4f46268a43118f712d5598beee4a29ae82c1895af70263edee4b0497bdd69fb0`。

| 指标 | control | root-replay candidate | restore_hands | restore_body |
|---|---:|---:|---:|---:|
| Full FID | .242136 | .237796 | .238176 | .242599 |
| Body FID | .238888 | .235066 | .233578 | .239316 |
| Hands FID | .213007 | .213476 | .212808 | .213513 |
| 滑步 m/s | .303542 | .303857 | .303709 | .303623 |
| contact F1 | .559557 | .559169 | .559102 | .559400 |

- 两恢复臂仍未恢复 control 的滑步/F1，消融未消除质量取舍；四臂评分已结束，不宣称质量全过。

## 09-11｜恢复动作可行性筛查（非质量评测）

- seed500124 在满足 `speed_sum<=control` 且 `grounded_transitions>=control` 的320条中，仅36条发生真实 token 变化，284条不变；未输出混合 H5。36条混合滑步 `.3031197943` vs control `.3035416065`，仅为可行性信号，无新 MM/质量结论；root 尚未启动少量质量守门采样。

## 09-11｜三 seed generated-motion sliding gate 固定规则与只读统计

- 固定规则：仅按生成动作 `foot speed sum<=对应生成 control` 且 `grounded transitions>=control` 接受 candidate，否则完整回退 control，所有样本保留；禁止用 GT contact/FID/embedding 选样，labels 仅事后评估。
- 原 per-sample counts 只读统计（非重新完整评分）：500124 accepted `320`、changed `36/260`，slide `.303541606539→.303119794334`，事后 F1 `.559557202965→.559617167339`；500125 accepted `321`、changed `37/260`，`.303833447997→.303403546143`，`.556776373420→.557052694851`；500126 accepted `334`、changed `47/257`，`.303563661217→.302942036184`，`.556742390051→.557851446913`。
- 尚未输出混合 H5，MM 未知；`apply_hhc_generated_sliding_gate.py` 待 root 审核，未启动评测。该方法只属任务1可行性探索，不能宣称质量改善。

## 09-11｜generated sliding gate seed500124 混合 H5 已生成，评分待定

- `tools/apply_hhc_generated_sliding_gate.py` 已由 root 修正原 SyntaxError，并补齐 physical/H5 names-request binding；self-check/py_compile exit0。session `84839` exit0，输出 `output/hhc_generated_sliding_gate_validation_20260911/gated_seed500124.h5`，SHA=`92795b495ffc1ee492d400c095567d3fd19433d8794a2047926fb53a91cdeadd`。
- root 独立核验逐条 mask、motions/tokens exact、544 complete/finite、names/lengths；accepted `320` 中真正变化仅 `36`，全544样本保留，不能把320称为真正多样化样本。筛选只用生成滑步门槛，不用 GT contact/FID。
- CPU joint32+physical16 valid_frames 串行 pipeline session `12176` 已启动，输出 `gated_seed500124_cpu.json` / `_physical_per_sample.json`，尚无评分或 MM 结果。

## 09-11｜generated sliding gate seed500124 评分完成（未晋级）

- monitor cell `1407` 已完成，session `12176` exit0；gated H5 实际 SHA、joint/physical 同源及 eval/requests/GT 一致性均通过 root 独立核验。

| 指标 | control | gated seed500124 |
|---|---:|---:|
| Full FID / matching / R1-3 / diversity | .242136 / 3.040499 / .439338,.647059,.766544 / 9.637486 | .240200 / 3.037585 / .439338,.645221,.766544 / 9.640872 |
| Body FID / matching / R1-3 / diversity | .238888 / 3.074780 / .457721,.641544,.777574 / 9.197988 | .240367 / 3.074243 / .457721,.641544,.777574 / 9.200251 |
| Hands FID / matching / R1-3 / diversity | .213007 / 3.079741 / .422794,.612132,.735294 / 8.842059 | .213873 / 3.078153 / .420956,.610294,.737132 / 8.845661 |
| 滑步 m/s / contact F1 | .303542 / .559557 | .303120 / .559617 |

- 三部 matching 改善，但 Full R2、Hands R1/R2 下降；未通过整体质量门槛，MM 未测。原 gate 规则不调参，seed500125/126 由 queue `55670` 继续。

## 09-11｜text-dropout 50k candidate pipeline 已启动

- checkpoint 保留 session `28520` exit0；`candidate_step_000050000.ckpt` 与 `latest-50000` same inode，CPU `global_step=50000`、epoch=`373`、bytes=`3401001523`，SHA=`1333fc1ed71c0381ff963bbc31215d378891b45cd818480bcd66e9c4da73449a`。
- candidate 50k pipeline session `8789` 已启动：GPU0 export544 后串行 CPU joint32/physical16 valid_frames，固定原 VQ175k、no-MGT-shape、paper_text_cfg、requested betas、seed500123、无 contact/root-replay并保存 tokens；尚无结果。native 50k `.5737` 不等于 frozen 指标；gate queue `55670` 由 cell `1410` 独占监控。

## 09-11｜generated sliding gate seed500125 评分完成（未晋级）

- root 独立验收 fixed gate mask、逐条 motions/token exact、H5 SHA=`2e7c517623c01fa8e9ec2d6a3d6c027c9d9a33268a7a12ec3ef7831e4f6b350c` 及 scores/eval/requests/GT 一致。

| 指标 | control | gated seed500125 |
|---|---:|---:|
| Full FID / matching / R1-3 / diversity | .244283 / 3.036593 / .435662,.645221,.766544 / 9.647035 | .242395 / 3.039182 / .433824,.641544,.766544 / 9.644775 |
| Body FID / matching / R1-3 / diversity | .234251 / 3.072344 / .452206,.647059,.783088 / 9.197659 | .234982 / 3.073526 / .452206,.647059,.783088 / 9.196669 |
| Hands FID / matching / R1-3 / diversity | .215322 / 3.077598 / .422794,.617647,.740809 / 8.840570 | .216213 / 3.077310 / .420956,.615809,.740809 / 8.841092 |
| 滑步 m/s / contact F1 | .303833 / .556776 | .303404 / .557053 |

- Full FID、slide、F1 改善，但 Body/Hands FID 略差；未晋级。seed500126 已进入评分，50k candidate `8789` 仍活且无 stdout。

## 09-11｜native 监控 ownership 收敛

- monitor cell `1339` 独占回收 session `32506`（30k）后再回收 `64522`（三 seed）；每次仅 `write_stdin` 60 秒、不唤醒模型，终态才通知，失败即停止。后续 main 只 `functions.wait` cell `1339`，不直接轮询或重复启动。
- 30k 仍按600秒调度等待；control seed500124 CPU 评分当前 PID `3793565` 运行。

## 09-11｜text-dropout 30k candidate 冻结 CPU/physical 评分完成

- session `32506` exit0；候选 H5 SHA=`fc24955602b43e69841db492ebdb1416b6edbbcfceaae9fc886138c580965f05`，544×1 complete/finite、names/lengths、checkpoint SHA、seed500123、batch32、no-MGT-shape/no-root-replay、CPU evaluator/GT provenance 均由 root 独立核验通过。

| 指标 | control | candidate |
|---|---:|---:|
| Full FID / matching / R1-3 / diversity | .339169 / 3.377028 / .373162,.577206,.705882 / 9.264928 | .399425 / 3.268703 / .402574,.606618,.722426 / 9.258535 |
| Body FID / matching / R1-3 / diversity | .437286 / 3.567663 / .400735,.595588,.696691 / 8.845899 | .455314 / 3.456135 / .428309,.619485,.737132 / 8.983942 |
| Hands FID / matching / R1-3 / diversity | .477634 / 3.231369 / .380515,.586397,.718750 / 8.184380 | .333738 / 3.341600 / .340074,.551471,.681985 / 8.457946 |
| 滑步 m/s / contact F1 | .335857 / .551843 | .311264 / .534756 |

- 结果不是整体质量通过：Full/Body FID、Hands matching/R、contact F1 退化，只有 Hands FID 与滑步改善。冻结 MM 尚未测，不得用 native `.3661` 代替；训练继续原50k计划，root replay 三 seed 仍由 cell `1339` 回收。

## 09-11｜root replay MM all-pairs paired-bootstrap 辅助诊断

- root 独立从 `all_pairs_diagnostic.per_prompt_mean` 重算，100 prompts 同序、10000 draws、seed90210；Full/Body/Hands 共用同一 `10000×100` 重采样索引矩阵（不是按部位推进 RNG）。candidate-control delta 与95% CI：Full `.05814540039165877` `[.03536927795349038,.08400447676400653]`；Body `.07718344953143969` `[.0391304038731032,.12403616830229289]`；Hands `.06446841141907499` `[.03896032110316446,.09330019382032333]`。
- prompt 层增/降/不变分别为 Full `59/12/29`、Body `58/13/29`、Hands `57/14/29`。此前按部位推进 RNG 的区间略有差异但数值并非错误；本项仅为 all-pairs 55-pair 辅助诊断，不是官方随机配对 MM CI，更不是质量非劣性证据。

## 09-11｜MGT 30k checkpoint 保留 watcher 已交接

- native watcher session `6457`（`root_mgt30k_watch_session`）每600秒检查 `prevalidation-000030000.ckpt` 与训练 PID `3537043`；当前仍等待，未重复启动 watcher。
- 就绪后 CPU 核对 `global_step=30000`，以 no-overwrite hardlink 保留至 `output/hhc_joint_text_dropout_p0p2_matched_validation_20260910/candidate_step_000030000.ckpt` 并记录 SHA；尚未声称保留完成。MM cell `1298` 独立。

## 09-11｜root replay full-quality CPU 结果（不晋级）

- Full replay H5 已完成 544/544 且 finite；首32条 archived root replay token IDs exact match。原始 exporter 进程已结束，但 exec exit code/walltime 未回收，保守记录为未回收。
- CPU joint session `50374` exit0，physical session `2327` exit0。候选 FID=`0.2309235324/0.2363433037/0.2115106628`，Matching=`3.0380189699/3.0771992768/3.0815968373`；对照 FID=`0.2409670257/0.2343818610/0.2108525635`，Matching=`3.0286780806/3.0814206881/3.0843989008`。Full FID改善，Body/Hands FID与Full Matching略变差。
- Physical 候选 sliding=`0.30401523258115926` m/s、contact F1=`0.5582317226152843`；对照=`0.30347044983617727`、`0.557580010526052`。滑步略差、F1略好；无 MM，不晋级。
- 配对 prompt bootstrap（10000 draws，seed90210，`root_replay_paired_prompt_bootstrap.json`）的 candidate-control sliding 差=`+0.0005447827` m/s，95% CI `[-0.0001088478,+0.0012573575]`；contact F1 差=`+0.0006517121`，95% CI `[-0.0021792958,+0.0036187368]`。两者均跨0，不构成非劣性证明。
## 09-11｜generated sliding gate 三 seed 收尾（未晋级）

- cell `1410` completed，queue `55670` exit0；三 seed 均通过 fixed gate mask、整条 motions/tokens exact、544 complete/finite、names/lengths、H5/joint/physical 哈希及 eval/GT/requests 一致性核验。

| 部位 | control 算术均值（FID/matching/R1/R2/R3/diversity） | gate 算术均值（FID/matching/R1/R2/R3/diversity） |
|---|---|---|
| Full | .244061 / 3.039665 / .436275 / .645833 / .765931 / 9.645012 | .243178 / 3.040629 / .435049 / .642770 / .765931 / 9.644324 |
| Body | .235810 / 3.072519 / .455882 / .644608 / .781863 / 9.197671 | .236956 / 3.074170 / .455882 / .645221 / .781250 / 9.198364 |
| Hands | .213729 / 3.079655 / .422794 / .613971 / .739583 / 8.842378 | .214882 / 3.079599 / .420956 / .612132 / .738971 / 8.843904 |
| 滑步 m/s / contact F1 | .303646 / .557692 | .303155 / .558174 |

- Body/Hands FID 三 seed 均变差；虽 Full FID、滑步、F1 改善，整体质量保持未通过、未晋级。不扩大或调 gate 阈值，不做 gate MM；第一项继续 50k text-dropout。
## 09-11｜text-dropout 50k candidate 冻结评分完成（未晋级）

- session `8789` 与 root 验收 `82459` 均 exit0；544 complete/finite、names/lengths、same evaluator/request/GT/source checkpoint 与 H5/joint/physical 一致均通过。candidate H5 SHA=`90286f5e5a465fe18583a6ad9e97c3201cd50416f3be59a36435fa6d97f4817b`。

| 部位 | control（FID/matching/R1/R2/R3/diversity） | candidate（FID/matching/R1/R2/R3/diversity） |
|---|---|---|
| Full | .366109 / 3.325272 / .391544 / .586397 / .709559 / 9.286224 | .429040 / 3.269688 / .411765 / .650735 / .746324 / 9.382929 |
| Body | .370827 / 3.466001 / .404412 / .610294 / .716912 / 8.918449 | .267876 / 3.269369 / .455882 / .652574 / .750000 / 8.794077 |
| Hands | .438952 / 3.275609 / .362132 / .566176 / .711397 / 8.243631 | .588571 / 3.240389 / .378676 / .591912 / .711397 / 8.514886 |
| 滑步 m/s / contact F1 | .326886 / .519287 | .308151 / .527835 |

- Body FID、matching/多数 R、滑步与 F1 改善，但 Full/Hands FID 明显恶化；整体未通过质量门槛。50k frozen MM 由 cell `1430` 独占运行，不能覆盖该 main 质量结论。

## 09-11｜50k frozen MM 配对完成（多样性改善，质量未晋级）

- 原 session `2537`/monitor `1430` exit0、cell completed；两组100×11 completed/finite，names/lengths 对应 request `mm_indices`，实际 H5/JSON/checkpoint SHA 及 requests/evaluator/runtime/sampling expectation 全部一致，generation metadata 仅 checkpoint/path SHA 差异。candidate H5 SHA=`0f4ea3c928a3066e4491c5d7174a15d25b4d3c17a696ba8494566d73f27b03cc`，control=`9af9e9b27b162bfe09a61f1c050c593d4cb81cb7209be3d2d0a7f2bc7a516f56`。
- MM Full/Body/Hands：control `.202199488878/.242317542434/.255214601755` → candidate `.271381229162/.348421573639/.418696016073`；duplicate `819/5500→370/5500`，平均 unique `7.65→9.38`。这是多样性信号，不等于质量通过。
- root paired all-pairs bootstrap（100 prompts、10000 draws、seed90210）delta/CI：Full `.153284990825` `[.016216490864,.303089862190]`；Body `.106132353237` `[-.035509846019,.247936985031]`；Hands `.167460946674` `[.055415586240,.292561440418]`。仅为 all-pairs 辅助统计，不是正式随机配对 MM CI。

| 指标 | control | 50k candidate |
|---|---:|---:|
| Full FID / matching / R1-3 / diversity | .366109 / 3.325272 / .391544,.586397,.709559 / 9.286224 | .429040 / 3.269688 / .411765,.650735,.746324 / 9.382929 |
| Body FID / matching / R1-3 / diversity | .370827 / 3.466001 / .404412,.610294,.716912 / 8.918449 | .267876 / 3.269369 / .455882,.652574,.750000 / 8.794077 |
| Hands FID / matching / R1-3 / diversity | .438952 / 3.275609 / .362132,.566176,.711397 / 8.243631 | .588571 / 3.240389 / .378676,.591912,.711397 / 8.514886 |
| 滑步 m/s / contact F1 | .326886 / .519287 | .308151 / .527835 |

- Full/Hands FID 退化，整体未质量晋级；不延长该 candidate，继续 span 单因素 pilot。

## 09-11｜50k CFG4.5 单臂采样校准诊断完成（未恢复质量）

- 针对 joint-dropout MM 提升但 Full/Hands FID 退化，仅预定 CFG=`4.5`（原 `3.5`）一臂；固定 candidate50k/VQ175k、544 冻结请求、seed500123、50 steps/T=.5、batch32、paper_text_cfg、requested MGT/VQ beta、no-shape/no-contact、无 noise/root-replay。
- session `83664` 经 monitor `1499` 回收 exit0；root 核验 544 names/lengths/finite、REQ/evaluator/GT、H5/joint/physical 绑定一致，H5 SHA=`5b6fb2202360185be29e2de0de62e01dcd222be676f41c70cd68d72c6915a01e`。

| 指标 | CFG4.5 |
|---|---:|
| Full/Body/Hands FID | .621480 / .304352 / .809347 |
| Full matching / R1-3 | 3.340484 / .400735,.619485,.724265 |
| Body matching / R1-3 | 3.373029 / .428309,.632353,.731618 |
| Hands matching / R1-3 | 3.328315 / .363971,.568015,.689338 |
| Full/Body/Hands diversity | 9.178613 / 8.621680 / 8.416808 |
| 滑步 m/s / contact F1 | .329123 / .524506 |

- 相对 CFG3.5 三部位 FID 与滑步全更差；停止该校准臂，不做 MM/扩网格。span v2 仍运行，overnight 监控 cell `911`。
## 09-11｜mask span-copy 暴露诊断完成（非模型评测）

- 原手写 tau 统计因 body/hand 错位且无脚本/artifact，不作为验收数据；root 改用 `tools/audit_mgt_span_copy_shortcut.py`，复用真实 `sample_historical_mmm_corruption_with_kinematic_ratios` 与 `sample_kinematic_cascade_mask`，局部只改 mask 布局，并断言随机替换、每 part count、hand≥body≥root、END/PAD、共享 visible 输入。self-check `70612`、val32×64 draws session `59206` exit0；sample hash=`adf4b1c443ffb549cc3f2a15498905d6576c4023539c56538880d34c501d9221`，masked B/H/R=`53474/71372/28391`，两臂一致。

| mask 布局 | ±1 body / hand / root | ±2 body / hand / root |
|---|---|---|
| original | 11.4635% / 2.9661% / 19.1892% | 16.2471% / 4.3729% / 26.6317% |
| span | 1.6756% / 0.7118% / 3.0890% | 3.1698% / 1.2904% / 5.7131% |

- 这是缓存暴露诊断，不含模型、训练、MM 或质量结论；仅支持继续做 mask-layout 单因素验证。原50k 配对 MM 仍由 cell `1430`/session `2537` 运行。
## 09-11｜circular-span 训练入口实现并通过 self-check（未启动训练）

- 新入口 `tools/train_mgt_circular_span.py` 在 `Trainer.fit` 内按 training step 临时替换 `utils.motion_masks.sample_kinematic_cascade_mask`，并在 `finally` 恢复；validation/生成不变。保留原 Bernoulli 替换、ratio/count/global RNG 消耗，span 使用 CPU generator seed=`seed+1000000+global_step`，复用已验证 circular-mask CPU helper。
- manifest/checkpoint args 明确 `runtime_intervention`，不支持 resume/预载 MGT。self-check `19979`、py_compile exit0，覆盖 count/ratio/tau/global RNG/tail/共同 visible 输入、异常恢复及恢复后原 corruption 精确重放；尚无 Lightning/GPU/长训练结果，不能写作已启动或质量改善。

## 09-11｜circular-span CPU integration 通过，GPU smoke 已启动

- `tests/test_mgt_circular_span_integration.py` session `74675` exit0，真实 CPU Lightning 两 training steps + sanity/validation 均通过：mask scope、validation helper identity、weight update、`Trainer.fit` 恢复及 checkpoint/manifest runtime intervention 标识一致。
- 原 MGT PID `3537043` 已退出，session `78308` 已回收 exit0。旧 smoke `72348` 因 main 规则拒绝 `wandb disabled` exit1；改 `wandb online` 后 session `62679` exit0，两步 loss `9.63→9.61`。checkpoint 验收 `58744` exit0：global_step2、浮点/optimizer finite、runtime intervention args/manifest 一致、非 FK 原 VQ tensors 与 net175000 逐值一致，SHA=`30e1c7031d9ddade75ec05dd0e87c70e2bc665497e8bb29b4754c47a7e23cee6`。长训练尚未启动，先绑定独立入口 source snapshot；50k MM 仍由 cell `1430` 运行。

## 09-11｜circular-span matched pilot（已异常终止，历史记录）

- `tools/train_mgt_circular_span.py` 与 `audit_mgt_span_copy_shortcut.py` 已保存到 run `source/span_runtime` 并 hash 写入/复核 manifest/checkpoint args；CPU integration `66342` exit0。
- GPU1 pilot session `96186`（`hhc_no_shape_circular_span_pilot50k_seed114514_v1`）曾按固定 recipe 启动，随后 exit132；最终约4379 optimizer steps（epoch32 batch91），无10k validation/无5000 checkpoint。无质量/MM 结论。
- root 只读复核 manifest、`source/span_runtime` 与实际源码逐字节一致：launcher SHA=`de95069d5d41d287eb12fb56e5b3ee0821f32fed557b55d6d09328bb23cc20bd`，helper SHA=`f0222696d12ebda76c2ab45d52f5d148622aeb99c75dc9527bca3029a240e0e1`。旧监控已终态；旧 joint-dropout 50k 配对 MM 已完成。

## 09-11｜circular-span pilot v2 受控重试已训练（非终态）

- v2 session `28378`、PID `3868579` 当前约8057 optimizer steps（epoch60 batch17），Train/all=`1.200`、约4.47 it/s；已验收2000步历史 checkpoint，现有 `latest-000008000.ckpt`/`last.ckpt`，尚无10k checkpoint/validation。已超过 v1 约4379步崩溃位置，但不能称底层故障已修复；cell `1525` 每600秒继续等待10k，225/2000 仅历史节点。
- 仅 checkpoint interval=`1000`、加入 `-X faulthandler`，其余 50k/300k 日程与学习设置不变；v1 exit132 历史证据保留。

## 09-11｜span v2 1000-step checkpoint 完整性验收

- session `73930` exit0；`latest-000001000.ckpt` CPU 完整读取，global_step=`1000`、epoch=`7`，1823 model/optimizer tensors finite，optimizer_states/lr_schedulers 存在，runtime intervention 与 manifest 一致；bytes=`3401000630`，SHA=`79eb1e06601fe5304e9a51b08966a4fbbf5943e38727055235aa837325f79d65`。两 VQ namespace 各140个非FK tensors 与原175k 逐值一致。
- launcher 仍明确拒绝 resume；这是 checkpoint 保存状态验收，不是断点续训/全轨迹恢复证据。恢复前须实现带 span intervention 的验证入口；无质量验证，未改 live 源码。

## 09-11｜span v2 原生 10k validation 与冻结评测已启动

- v2 已进入原生 `[Validation] Starting evaluation at iter10000`，PID `3868579` 仍运行。`latest10000` 已固定至 `output/hhc_circular_span_matched_validation_20260911/step_000010000.ckpt`；audit `7891` exit0，global_step=`10000`、epoch=`74`、runtime span contract 存在，SHA=`1a079b2114f9c031942fd24682454aaf2b766286b76cdd691a6db24b4ca0e046`。
- GPU0 冻结544 pipeline `10473` 已启动，monitor `1532` 独占回收；固定50/T=.5/CFG3.5、seed500123、batch32、requested beta、no-shape/no-contact/noise/no-replay，CPU joint+valid_frames physical；尚无新质量指标，不先跑新 MM。cell `1525` 继续等待 native 10k model checkpoint，训练未终态。

## 09-11｜span v2 10k candidate 冻结评分完成，control 同 CPU 重算中

- pipeline `10473` 经 `1532` 回收 exit0；candidate 544 complete/finite、REQ names/lengths 通过，H5 SHA=`fbed29668ccfcbb7fd982f882113311c8fd0511275795c8ab1610be28d0748d7`，physical 协议/实际 H5 hash/REQ/floor valid_frames 一致。

| 部位 | candidate FID / matching / R1-3 / diversity |
|---|---:|
| Full | .479971 / 3.576650 / .382353,.560662,.674632 / 9.383414 |
| Body | .818115 / 3.881188 / .358456,.540441,.659926 / 8.838158 |
| Hands | .493728 / 3.394106 / .369485,.545956,.672794 / 8.337760 |
| 滑步 m/s / contact F1 | .327429 / .528829 |

- 旧 control 10k JSON 的 GT diversity 与当前 runtime 略有差异，不能直接混用；同 CPU batch32 control 重算 session `82838` 正在运行。control physical 可复用 slide `.291672 m/s`、F1 `.522846`。待 control 确认再作配对报告，不以部分 FID 宣布成功、不启动 MM。

补充 native 10k：W&B 首次验证 FID=`.59884`、diversity=`8.90847`、R1/2/3=`.41544118/.59926471/.70036765`、matching=`3.85747`；native MM=`0.0000` 仍是 placeholder，实际 MM 子循环仅到1/4，PID `3868579` 仍运行。native 指标不可与冻结 joint 混用，验证尚未收尾。

补充 native 10k 收尾：monitor `1525` exit0，`model-000010000-0.5988.ckpt` 存在，native Full FID=`.59884` 仍为10k best；native 100-subset MM=`.2407`、subset FID=`1.19564`、diversity=`9.3171`、R=`.39/.60/.72`、matching=`3.74691224`，不可混作 frozen joint MM。PID `3868579` 仍 live（epoch76 batch97）；rolling 10k 与 named checkpoint 的896 state tensors 逐值一致、global_step均10000，CPU对比 exit0。monitor `1543` 每600秒等待20k latest/model checkpoint。

## 09-11｜span v2 10k 配对冻结评分完成（未晋级）

- control CPU 复算 session `82838` exit0；control/candidate JSON 的 GT diversity、REQ/GT/evaluator/seed/effective_length/samples 逐值一致，实际 H5 SHA 已核验。

| 部位 | control（FID / matching / R1-3 / diversity） | span v2（FID / matching / R1-3 / diversity） |
|---|---|---|
| Full | .468141 / 3.626958 / .363971,.553309,.676471 / 9.151043 | .479971 / 3.576650 / .382353,.560662,.674632 / 9.383414 |
| Body | 1.228750 / 3.875731 / .362132,.553309,.667279 / 8.601533 | .818115 / 3.881188 / .358456,.540441,.659926 / 8.838158 |
| Hands | .817906 / 3.563200 / .330882,.536765,.674632 / 8.104405 | .493728 / 3.394106 / .369485,.545956,.672794 / 8.337760 |
| 滑步 m/s / contact F1 | .291672 / .522846 | .327429 / .528829 |

- Body/Hands FID 早期改善，但 Full FID 与滑步退化；pooled-rate bootstrap（10000、seed90210）slide delta `+.035756471798` CI `[+.024448431166,+.047084300307]`，F1 delta `+.005982403233` CI `[-.027236613742,+.039861801113]`。未质量保持、未晋级，不做额外 frozen MM；继续50k pilot，20k 再同 step 验证。native MM 仍独立运行，不用 placeholder 代替。

## 09-11｜span v2 20k 冻结配对评分完成（未晋级）

- pipeline `44555` 经 monitor `1563` exit0；candidate H5 SHA=`4989db19eb7b6a8aef33e85906713a914911c0018b3a97fee714bf5ba48539cd`。

| 部位 | control（FID / matching / R1-3 / diversity） | span v2（FID / matching / R1-3 / diversity） |
|---|---|---|
| Full | .558887 / 3.542590 / .380515,.571691,.691176 / 9.203939 | .352108 / 3.447433 / .384191,.556985,.698529 / 9.256051 |
| Body | .573045 / 3.721830 / .402574,.588235,.696691 / 8.727955 | .508324 / 3.721463 / .378676,.545956,.658088 / 8.716905 |
| Hands | .617900 / 3.373708 / .369485,.547794,.691176 / 8.212367 | .401563 / 3.411173 / .354779,.547794,.667279 / 8.236732 |
| 滑步 m/s / contact F1 | .325428 / .539508 | .388338 / .531824 |

- 三部位 FID 改善，但滑步与 contact F1 退化，未质量保持、未晋级、不额外 MM。root audit `12808` exit0：两 H5 544 complete/finite、names/lengths 与 REQ 一致，H5 SHA 与 joint/physical、checkpoint attrs、evaluator/GT/REQ/seed/effective_length_policy、GT diversity、physical floor/protocol 全部一致；首次错误使用 `complete` 键 exit1，改用实际 `completed_repeats` 后通过。
- pooled-rate bootstrap（10000、seed90210）slide delta `+.0629104062` CI `[+.05173478,+.07429599]`，F1 delta `-.0076837595` CI `[-.04467526,+.02829080]`。native 20k FID=`.50341` 仍为 best20k；MM 真实完成4/4=`.1818`，100-subset FID=`1.32655`、div=`9.14739`、R=`.49/.65/.74`、matching=`3.4359824657`。此前 `0` 是中途 placeholder；subset 指标不可混作 frozen MM。20k质量保持仍失败，继续50k pilot，monitor `1564` 等30k，原 session `28378` 未终态。

## 09-11｜span v2 step2000 导出集成 smoke 通过

- 初次 session `78867` exit1 是 rolling `latest-1000` 被2000替换导致的 FileNotFound，未生成 H5；root 固定 step2000 hardlink 为 `output/hhc_span_v2_step2000_export_smoke2_20260911.ckpt`，避免滚动竞态并保留 inode。
- GPU0 两样本官方 export session `12277` exit0，audit `30008` exit0：global_step=`2000`、runtime span contract 存在，H5 motions=`(2,1,150,672)`、tokens=`(2,1,50,3)`，两条 complete/finite；checkpoint SHA=`cf751d02b5bafbdb90dc4a112742bd5e88403467102ab9a8effefd67add0dc21` 与 H5 attrs 一致，VQ SHA 与原175k一致。
- 仅为评测入口兼容性 smoke，不是质量指标；不跑两条 FID、不比较 batch2 与正式 batch32。训练尚未到10k validation，正式评测前须固定同 step checkpoint。

补充 v1 exit132 的只读 ELF 证据：kernel 给出 offset `0x29f9ef`，`readelf/objdump/addr2line` 对应当前 Python3.12 binary 的 `decode_current_locale+0x4f`，磁盘字节为普通 x86-64 `cmp rax,-1`（`48 83 f8 ff`），不是 `UD2` 或可归因于 AVX 专用指令。无 core、运行时内存字节或故障栈，不能据此确诊硬件、Python 或本实验算法；环境未修改。
## 09-11｜50k control frozen MM 评分完成

- `output/hhc_joint_text_dropout_p0p2_matched_validation_20260910/control_step_000050000_mm_cpu.json` 已完成；H5 SHA=`9af9e9b27b162bfe09a61f1c050c593d4cb81cb7209be3d2d0a7f2bc7a516f56`，100×11 repeats 全部完成且 motions finite。
- Full/Body/Hands MM=`.202199488878/.242317542434/.255214601755`；duplicate=`819/5500=.1489090909`，平均 `7.65` unique motions。candidate 尚待完成，因此不下配对结论、不声称质量过关；新版 `sampling_expectation` 增加字段，与旧300k JSON 不作跨 step 同协议比较。cell `1430` 仍 live。
## 09-11｜v4→span 基线差异与 sampler 局部回归核查

- v4 历史 manifest 与 span manifest 的 cache/dataset aggregates/loader contract/source-derived configuration/paper configuration 相等；resolved 参数只有 run/path/resume/W&B 与 50k diagnostic stop 差异。源码差异包括 InterHuman 分支、VQ `temporal_stride`、可选 `text_drop_prob`（本轮默认与原一致）、shapeaug hybrid 路径及 latent_dims fallback/采样策略；Inter-X 无 latent_dims 时仍为原比例。
- 历史 launch sampler 与当前 `sample_intra_cas` 在 CPU 合成 logits 下，2 seeds×CFG(-1,3.5)、50 steps、T=.5 的 token IDs/RNG 逐值一致，exec exit0。该证据仅是局部回归，不是全训练轨迹 bitwise 一致或质量结论；无需新评测框架，复用现有 export/scorer 同 step 评测。
## 09-11｜span v2 30k checkpoint 固定，冻结评测已启动

## 09-11｜span v2 30k 冻结评分完成，MM 方向诊断已启动

- pipeline `25229`/monitor `1577` exit0；root audit `87790` exit0，candidate H5 SHA=`8fed9107fe1f34bcea8a4687b2eab57ace9ab22762d4add07613e747148cab16`，两 H5 544 complete/finite、names/lengths、REQ/GT/evaluator/seed/length policy、SHA/attrs/floor 协议一致。

| 部位 | control（FID / matching / R1-3 / diversity） | span v2（FID / matching / R1-3 / diversity） |
|---|---|---|
| Full | .339169 / 3.377028 / .373162,.577206,.705882 / 9.264927 | .272544 / 3.499738 / .393382,.575368,.681985 / 9.392608 |
| Body | .437286 / 3.567663 / .400735,.595588,.696691 / 8.845899 | .290904 / 3.628796 / .376838,.584559,.704044 / 8.816486 |
| Hands | .477634 / 3.231369 / .380515,.586397,.718750 / 8.184380 | .356063 / 3.407224 / .375000,.582721,.707721 / 8.468070 |
| 滑步 m/s / contact F1 | .335857 / .551843 | .322701 / .546104 |

- 三部位 FID/滑步改善但 matching 退化、F1 略低，未晋级。bootstrap slide delta `-.01315563` CI `[-.02511621,-.00146350]`、F1 delta `-.00573927` CI `[-.04012491,.02762252]`；100×11 MM pipeline `22914`/monitor `1585` 已启动，仅判断多样性方向，不表示质量门槛通过。

补充 native 30k：Full FID=`.41091`，新 best30k（20k `.50341`）；div=`9.1539`，R1/2/3=`.43933824/.63051471/.72610294`，matching=`3.7564303875`。native 100-subset MM 真实4/4=`.1765`，subset FID=`.83023`、div=`9.20940`、R=`.37/.66/.76`、matching=`3.6190307140`；此前 `0` 是中途 placeholder，不可替代 frozen MM。PID `3868579` epoch236 batch94 仍存活；冻结配对 MM `22914`/monitor `1585` 仍运行，monitor `1578` 等40k。

补充 native 40k 收尾：native MM 真实完成=`.1852`，100-subset FID=`1.15428`、div=`8.94210`、R=`.43/.69/.78`、matching=`3.4038772583`；此前 `0` 是中途 placeholder。Full native 40k FID=`.6075`，30k `.41091` 仍 best；subset 指标不可混作 frozen MM。PID `3868579` 已到 epoch337 batch25，monitor `1613` 继续等50k，不能称50k完成。

- monitor `1564` exit0；固定 hardlink=`output/hhc_circular_span_matched_validation_20260911/step_000030000.ckpt`。audit `11396` exit0：global_step=`30000`、epoch=`223`，runtime span 配置/源 SHA 保留，文件 SHA=`6b4cb02f3a5f6e248d6cd65f1df44e8456c2882c0eb7297d949b65e038bfde22`。
- pipeline `25229` 已启动，monitor `1577` 每600秒独占回收；尚无30k质量/MM结果，不先追加 MM。control30k 文件待候选核验后复用；原训练 session `28378` 未终态，monitor `1578` 每600秒等待40k或 PID 消失。
## 09-11｜span v2 30k paired MM 完成（无一致增益）

- root audit `75050` exit0；初次 `54270` exit1 是把布尔 `completed_repeats` 误作计数11，纠正后两 H5 100×11 completed/finite、REQ names/lengths、文件/JSON/checkpoint SHA、H5 attrs、协议/REQ/evaluator/runtime 全部一致，仅 checkpoint metadata 不同。control/candidate H5 SHA=`5f11a009045143feae7f70906cd485ccd413e3cc4a656235190ce208f7a1d11e` / `f1ded42872a162b4a1190894e623ea32ab0a0908f6f8d398697c40b28b81cc37`。

| 指标 | control | span v2 candidate |
|---|---:|---:|
| Full/Body/Hands MM | .166925 / .223767 / .261704 | .192432 / .198570 / .272696 |
| exact duplicate pairs | 711/5500 | 724/5500 |
| mean unique motions/prompt | 7.49 | 7.62 |

- root 独立重算与 JSON 一致。all-pairs prompt bootstrap（10000、seed90210）delta/CI：Full `.05430294` `[-.04666430,.16907067]`，Body `-.02962807` `[-.11851366,.06222363]`，Hands `.00106874` `[-.07138280,.07170228]`，均跨0；没有一致多样性改善或重复坍缩缓解。
- 30k FID+slide 虽改善但 matching/F1 退化，仍未晋级；不追加30k grid，继续50k pilot观察40/50k。pipeline worker 已退出且 JSON status complete，原 session `22914` 已由 monitor `1585` 正常回收，exit0。
## 09-11｜span v2 40k checkpoint 固定，冻结评测已启动

- monitor `1578` exit0 返回 `STEP_40000_CHECKPOINT_READY`；固定 hardlink=`output/hhc_circular_span_matched_validation_20260911/step_000040000.ckpt`。root session `19658` exit0：global_step=`40000`、epoch=`298`，SHA=`a223e1f23eeb6bab280b0a4b899d2065fdd898003dcecc184bd065b6f85d2f81`；audit `30149` exit0，hyper_parameters.args 相对30k无变化，runtime span contract 与 launcher/helper SHA 一致。
- pipeline `70984` 已启动，monitor `1612` 独占回收；544 export→CPU joint32→CPU physical(valid_frames/per-sample)，固定原 VQ175k、seed500123、50/T=.5/CFG3.5、requested betas、no-shape/no-contact/noise/replay/save-token-ids。尚无40k评分，不追加 MM；原 session `28378`/PID `3868579` 继续，monitor `1613` 每600秒等待50k或 PID 消失。
## 09-11｜span v2 40k 冻结评分完成（未晋级）

- pipeline `70984`/monitor `1612` exit0；audit `17834` exit0。candidate H5 SHA=`0f31b1ccfcf12846bccad93df15c74d83c9c8df357464267e9d55edc573f6904`，control H5 SHA=`48ee89f9ebf5ce3d1f8acf6e2a5611c152a7821692c97fe74d58e93d15aa0915`，checkpoint SHA=`124cd42604c2d8d2d526530e058a0b09e120faf698864621e536f07d7be3b804`；544 complete/finite、REQ/eval/GT/physical 一致。除 checkpoint 外，old control 缺 beta_policy/save_token_ids 诊断字段，不能宣称所有 attrs 逐字相同。

| 部位 | control（FID / matching / R1-3 / diversity） | span v2（FID / matching / R1-3 / diversity） |
|---|---|---|
| Full | .423530 / 3.236657 / .417279,.628676,.727941 / 9.597493 | .394411 / 3.216262 / .404412,.593750,.735294 / 9.357484 |
| Body | .337148 / 3.233866 / .420956,.634191,.751838 / 9.014789 | .250502 / 3.338116 / .446691,.626838,.713235 / 9.007552 |
| Hands | .486570 / 3.219543 / .386029,.590074,.716912 / 8.474455 | .478037 / 3.152926 / .369485,.579044,.711397 / 8.543981 |
| 滑步 m/s / contact F1 | .327072 / .523550 | .348083 / .562119 |

- 三部位 FID 改善，但滑步退化，matching/部分 R 下降，未质量晋级、不额外 MM。bootstrap slide delta `+.021011110660` CI `[+.008868982088,+.033294283129]`，F1 delta `+.038569690658` CI `[+.003916744111,+.073491913612]`。native 40k Full FID=`.6075`（30k `.41091` best），MM `0` 仍 placeholder、真实子循环1/4，不与 frozen MM 混用；继续50k pilot，monitor `1613` 等50k。
## 09-11｜VQ active-code 与冻结请求结构补证

- `tools/measure_hhc_token_compression.py --self-check --training-manifest` 重算 exit0（chunk `82b1e1`）：8617 active train samples，各通道287161 historical38 payload；used codes Body/Hand/Root=`9749/7121/6884`（每码本12288），全局有效词汇=`4950.7649/4436.8763/4441.1641`，与旧统计一致。该证据不支持“VQ 只激活1–2码”，也不能把全局缓存熵与 MGT 逐位置条件预测熵直接等同，或证明语义差异完全保留。
- 544 冻结请求 normalized captions exact，重复文本组数0、每文本仅1样本；没有 GT→VQ 同条件 MM 的直接同文本重复实证。计划中的动作组内距离必须标记为 action-group proxy，不作为正式同文本 MM；token_compression_statistics 正在只读定位原 VQ175k 重构/ token/eval artifacts。
## 09-11｜VQ pair-distance retention 诊断已启动

- 原 VQ175k 175k reconstruction report 已定位：544 samples，checkpoint SHA=`fdf5af...`，冻结评测 codebook usage Body/Hand/Root=`2741/3217/4379`（各17895 tokens）；与 historical38、8617 train-cache 统计范围不同，不混用。重构 H5 SHA=`d5ee5d9d9cf98c54df9b6603fc68abc22ce0168acba9529951040eda87df2534`，本轮未重算距离。
- `tools/diagnose_vq_pair_distance_retention.py` self-check chunk `449c29` exit0；CPU4-thread pipeline `83819` 已启动，monitor `1637` 每600秒独占回收。输入 reconstruction report，核验 GT/recon/REQ/checkpoint SHA 后 embed544 GT/重构，输出 `output/hhc_original_vq175k_pair_distance_retention_20260911.json`，尚无结果。
- 以相同 action ID 配对，主 gap≤8、敏感性≤150，输出三部位距离均值比、Pearson、normalized-distance RMSE 与逐 pair 距离。该诊断不是同文本 MM，pair 不独立，不证明语义完整或质量通过；monitor `1613` 仍等50k。
## 09-11｜原 VQ175k action-group distance retention 诊断完成

- pipeline `83819`/monitor `1637` exit0；audit `78992` exit0，实际 GT/recon/REQ/checkpoint/script/evaluator/report 哈希、544 names、两组 gap pair selection、finite distance arrays 与 ratio/Pearson/normRMSE 重算均一致。

| 配对范围 | pairs / samples | GT距离均值 Full/Body/Hands | 重构距离均值 Full/Body/Hands | ratio Full/Body/Hands | Pearson Full/Body/Hands | normRMSE Full/Body/Hands |
|---|---:|---|---|---|---|---|
| gap≤8 | 1529 / 492 | 4.974273 / 4.758752 / 4.899147 | 5.228084 / 5.148247 / 5.089142 | 1.051025 / 1.081848 / 1.038781 | .902140 / .835363 / .882855 | .267482 / .343116 / .269800 |
| gap≤150 | 3668 / 544 | — | — | 1.055022 / 1.082391 / 1.044518 | .906222 / .845240 / .893207 | .265346 / .334292 / .259850 |

- 结果不支持原 VQ 将 action-group 内差异普遍压成同一动作；平均距离稍扩张也不是额外有效多样性。544 captions 唯一，因此这不是同文本 MM；pair 不独立，不提供伪 CI，也不证明语义完整保留。结合 active-code 与固定权重采样/MGT step 对照，更支持先定位 MGT 预测尖锐与采样，而非盲重训 VQ。50k monitor `1613` 仍 live。
## 09-11｜span v2 50k checkpoint 固定，冻结评测已启动

- monitor `1613` exit0 返回 `STEP_50000_CHECKPOINT_READY`；root session `7502` exit0，固定 hardlink=`output/hhc_circular_span_matched_validation_20260911/step_000050000.ckpt`，global_step=`50000`、epoch=`373`，hyper_parameters.args 相对40k逐值无差异（含 runtime span SHA），checkpoint SHA=`398326d8f914381e1acfac6eb28a0ca6621f0e1620961611397936a45d21323f`。
- pipeline `59684` 已启动，monitor `1646` 独占回收；544 export→CPU joint32→physical valid_frames/per-sample，固定原 VQ175k、seed500123、50/T=.5/CFG3.5、requested betas、no-shape/no-contact/noise/replay/token IDs。尚无50k结果，不追加 MM；native 50k eval 当时仅5/17。monitor `1647` 等 PID 消失，训练未宣称正常结束。
## 09-11｜span v2 50k 冻结评分完成（未晋级）

- pipeline `59684`/monitor `1646` exit0；audit `76606` exit0。candidate H5 SHA=`d2faba837792778f7ca6b299c17304701fa674e444b1c665f45c434100ac7dc0`，control H5 SHA=`2616217ccea4d4ae97d284492cf5ac4697bfb5f16269ef1101fdee2c5e3fd0cd`，control checkpoint SHA=`0e2308c493124ad87f69528606175c30dcd1cc0324031db8ec5c0c778724e055`；544 complete/finite、REQ/eval/GT/physical 一致。old control 缺 beta_policy/save_token_ids 诊断 attrs，其余采样 attrs 一致，不能称全部 attrs 相等。

| 部位 | control（FID / matching / R1-3 / diversity） | span v2（FID / matching / R1-3 / diversity） |
|---|---|---|
| Full | .366109 / 3.325272 / .391544,.586397,.709559 / 9.286224 | .364234 / 3.285307 / .415441,.630515,.740809 / 9.347459 |
| Body | .370827 / 3.466001 / .404412,.610294,.716912 / 8.918449 | .219427 / 3.257190 / .433824,.612132,.751838 / 8.891309 |
| Hands | .438952 / 3.275609 / .362132,.566176,.711397 / 8.243631 | .450787 / 3.209277 / .409926,.593750,.704044 / 8.400132 |
| 滑步 m/s / contact F1 | .326886 / .519287 | .352584 / .533215 |

- FID 与部分 R 改善，但滑步明显退化、Hands FID 变差；bootstrap slide delta `+.025697625943` CI `[+.013334034210,+.037692006971]`，F1 delta `+.013927762141` CI `[-.018062935041,+.045862711888]`。未质量晋级、不追加 MM、不延长 span 训练。
## 09-11｜max-probability 排序假设登记（尚未实施）

- 读取 `sample_intra_cas` 全链确认当前 `confidence=softmax(channel_logits).gather(predictions)`，因此非 argmax proposal 会被排序额外惩罚；noise 仍线性退火到0，尚未测试 candidate-independent confidence。
- 预定单因素 max-probability 排序：位置置信度改取该位置 max probability，不依赖本次抽样 token；保留候选 Gumbel 抽样、温度/CFG/steps、RNG draws 与 mask counts 不变。先做 baseline parity 与 synthetic selection self-check，再小规模生成和完整质量决定是否 MM；尚未实施或验证提升。待 monitor `1647` 终态后才改共享 sampler/exporter，50k失败结论保留。
## 09-11｜span v2 训练 clean stop，native 50k 验证收尾

- session `28378` 已在 step50000 validation 后 clean stop，exit0。native 50k Full FID=`.5167`、div=`9.0198`、R=`.42830882/.62316176/.71875`、matching=`3.769199609756`；最终100-subset MM=`.1848`、subset FID=`.86011`、div=`9.23729`、R=`.48/.63/.74`、matching=`3.484842538834`。native best 仍30k Full FID=`.41091`；冻结50k失败结论不变，不延长训练。
- max-probability 单因素排序正在实施，尚未通过测试或生成结果。
## 09-11｜max_probability trace 诊断已启动（尚无结果）

- 12项重跑 session `72977` exit0，包含 max_probability observer token/RNG parity；MM scorer `--self-check` exit0。
- trace session `56635` 已在 GPU1 启动，输出 `output/hhc_max_probability_trace_val8x4_20260911`；monitor cell `1664` 每600秒独占回收。模型 exact load、requested VQ preserved 已确认，control repeat0/1 各150 stages 完成；全流程及评分尚未完成，不得称任务完成。
## 09-11｜max_probability 8×4 trace 诊断完成（无改善信号）

- trace session `56635` exit0；audit `63a13b` exit0：REQ/checkpoint/VQ/script/sampler SHA、8 artifact SHA、names/lengths/finite、轨迹衔接/final IDs/stored counts 全部通过。8×4 两臂 motion duplicate 均42/48；Body/Hands/Root token duplicate control=`43/42/43`→maxprob=`45/42/42`。
- 每 part 1000 accepted；nonargmax accepted control=`2/1/5`→candidate=`1/4/6`，nonargmax proposals control=`798/939/822`→candidate=`796/940/823`，各31508 proposals。未见多样性改善，不追加544质量/MM，不外推为全域无效。
- 这说明解除 drawn-token 的概率惩罚不足，高置信位置本身仍尖锐主导。下一假设尚未实施：仅在早期确定位置引入排序随机性，之后回归原排序，以区分全程线性退火噪声；第1项仍未达质量不降目标。
## 09-11｜early ranking-noise cutoff trace 已启动（尚无结果）

- `confidence_noise_until_step` 默认0保持原 full schedule；本实验 cutoff=`10`，仅 step≤10 使用原50步线性幅度、不重缩放，其他采样保持。14项 unittest session `9011` exit0，覆盖 golden token/RNG、full endpoint、noise=0、2-step 恰好6次 noise draw、前2步轨迹及 invalid/resume 防混；MM scorer self-check exit0。
- trace session `77111` 已在 GPU1 启动，输出 `output/hhc_early_noise10_trace_val8x4_20260911`；原 v4 300k+VQ175k、8 length quantiles×4 repeats、seed700123、50/T=.5/CFG3.5，两臂 control/log_probability_noise1、cutoff10（仅 control 无噪声）。尚无结果；默认行为不变，生成停止后才考虑共享 source 变更。
## 09-11｜early-noise10 trace 完成，full544 质量评测已启动

- trace session `77111`/cell `1675` exit0，audit `89750e` exit0；所有 provenance/artifact SHA、names/lengths/finite、counts、轨迹衔接/final IDs 通过，control NPZ 与上轮 maxprob control bitwise 一致。motion duplicate `42→12/48`；Body/Hands/Root token duplicate `43/42/43→16/19/16`；各1000 accepted、31508 proposals，nonargmax accepted `2/1/5→6/2/4`，proposals `798/939/822→888/938/810`。
- 该结果支持 commitment-order 机制的多样性信号，但不是正式 MM，也不证明质量。full544 pipeline `17976` 已在 GPU1 启动，monitor cell `1680` 每600秒独占回收；启动 exact model load/requested VQ preserved，已生成 `32/544`。固定 noise1 cutoff10、50/T=.5/CFG3.5、seed500123、no-shape/no-contact/requested betas/save IDs；尚未评分。
- baseline control H5 SHA=`3827dacb07d81c37e9a67cb7269bca0671b083ade66459dce40535e6ad504e0a` 与 CPU/physical 文件绑定、GT diversity及采样/ckpt/VQ 参数核验通过；baseline JSON 缺 `evaluator_runtime`，不能声称该字段完全一致。
## 09-11｜early-noise10 full544 评分完成（未晋级）

- pipeline `17976` exit0；评分文件齐全后仅终止 timer `1680` 并回收 pipeline，未终止训练任务。audit `31302` exit0（chunk `614e3c`）：candidate H5 SHA=`cb867e7986bea7106afa188b0e8333c9b458f8381a3ded9aeadd11f3c35f2202`，544 finite/complete、names/lengths、checkpoint/VQ/sampler/REQ/GT hashes、noise1 cutoff10 metadata、eval/GT/length policy/seed/physical protocol 均通过；双方 JSON 缺 runtime 字段，不声称其一致。

| 部位 | baseline（FID / matching / R1-3 / diversity） | early-noise10（FID / matching / R1-3 / diversity） |
|---|---|---|
| Full | .240967 / 3.028678 / .433824,.643382,.766544 / 9.646897 | .270059 / 3.018091 / .455882,.652574,.773897 / 9.729600 |
| Body | .234382 / 3.081420 / .454044,.647059,.781250 / 9.208961 | .208631 / 3.081273 / .459559,.656250,.762868 / 9.170169 |
| Hands | .210853 / 3.084399 / .422794,.613971,.735294 / 8.839807 | .291210 / 3.060107 / .409926,.641544,.751838 / 8.800935 |
| 滑步 m/s / contact F1 | .303470 / .557580 | .306257 / .567495 |

- Full/Hands FID 分别约+12.1%/+38.1%，Body FID 改善但 matching/部分 R 退化；slide delta=`+.002786625901`，95% CI `[-.00335632,+.00897634]`；F1 delta=`+.009914885655`，CI `[-.00979474,+.02981778]`，均跨0。未晋级，无正式 MM，不追加 noise 网格，返回训练侧诊断。
## 09-11｜MGT mask-exposure calibration 诊断已启动（尚无结果）

- 已核对 masked-only comparison 与 train loss 分支：50k quality failure、10k 恢复缺 RNG 证据，不能作完全 clean 因果。复用 `tools/audit_mgt_mask_exposure_gap.py`，pipeline `54586`（GPU1）比较 v4 MGT 30k (`model-000030000-0.6630.ckpt`) 与300k；validation256 后 training256，seed90123、batch16、paper_text_cfg、CFG3.5、sharpness_temperature=1，NLL temperature grid=`.5/1/1.5/2`。
- validation256 JSON 已完成，两个 checkpoint exact load 且 VQ preserved；training256 仍运行，由 cell `1695` 每600秒独占回收，尚无完整配对诊断结论。
- 初读 validation 全 mask、未 CFG NLL：Body/Hand/Root 30k→300k=`10.6222/11.5429/11.0630→12.7670/13.9538/13.8460`，置信变高但 NLL 变差；需等 training 对照，暂不归因过拟合，也不把绝对 token 低 accuracy 等同动作差（多模态标签与近似 codes）。
- 限制：`paper_scheduled_mask` 为纯 mask 而非历史 random replacement；`body_after_gt_root/hand_after_gt_root_body` 为 GT-context probes 而非生成行为；30k/300k 主比较同 split/samples/mask，train/val 不逐样本配对；NLL 最优温度不等于生成质量最优。
## 09-11｜MGT mask-exposure calibration 完成

- pipeline `54586` exit0；两 JSON 齐全且进程退出，root 仅回收 timer `1695`。audit `7f32b8` exit0，split/manifest/checkpoint actual SHA 已核验。
- fully masked/noCFG/T1 training 30k→300k：NLL B/H/R=`3.6174/3.9641/4.6721→1.3662/1.4396/1.3536`，accuracy=`40.3213/36.1717/27.6050%→84.2875/83.3299/83.6594%`；validation NLL=`10.6222/11.5429/11.0630→12.7670/13.9538/13.8460`，accuracy=`1.6517/.3874/.8666→2.0086/.7239/1.4682%`，mean maxprob=`.3525/.3073/.2894→.5574/.5202/.5448`。
- teacher-GT cascade temperature grid（train 30k→300k：T=.5/1/1.5/2=`3.9028/2.5388/2.7107/3.5075→2.6770/1.4471/1.1770/1.3856`；val=`19.6355/11.0961/9.4068/9.0904→25.0479/13.2973/10.2554/9.3975`）同样显示 train/val gap。结论支持晚期置信增长伴随泛化差距，但不等于纯低熵=坏；code support、文本、subject 混杂尚未排除。
- 下一步只读排查全 train code frequency、val unseen-code ratio、首次文本 train match，并复核 val256 sample hash；不启动新训练。
## 09-11｜calibration code-support 缺口核查

- root CPU audit `89328` exit0（chunk `71238a`）：按历史训练前37 payload 计数，train8617 各287161/part、active codes Body/Hand/Root=`9749/7121/6884`；calibration val256 实际9808 tokens，train 未见=`59/8/23`（`.602/.082/.235%`），historical38 whole val534 未见=`113/14/29`。该缺口不能解释 validation <2% accuracy。
- train/val names 无交集；val256 首次 caption 精确匹配 train 任意 caption `22/256`。training samplehash=`c79d5cc3e1405d9d4aa7e1cc0df9faec3657971fa5bd353cab3c4346ce4ebc2c`、validation=`c2c0f24ca2db312ff9c0bedeb87d4fe484900d344c2120488fa5a300047a0fcd` 与正确 calibration JSON 一致；子 agent 对旧 val32 JSON 的不一致说法作废。
- 现有 audit `make_batch` 按 model block_size-1 截断49 payload，而训练 historical38 为37 payload；结果有效但不是完全匹配训练窗口。下一步先增加显式38-window probe，不启动新训练。
## 09-11｜historical38 token-window probe 已启动

- `audit_mgt_mask_exposure_gap.py` 新增显式 `--token-window`，默认保持 model block_size-1；输出保存实际 window/max_payload/script SHA，`make_batch` 增加校验。`test_mgt_logit_sharpness` 新增38/50 prefix/end/padding/非法窗口测试，共5项 session `77719` exit0。
- GPU1 pipeline `79729` 已启动：validation256→training256、v4 30k/300k、seed90123、batch16、Tsharp=1、CFG3.5、paper_text_cfg、NLL grid `.5/1/1.5/2`，唯一因子 `--token-window38`；输出 `output/hhc_mgt30k300k_calibration_validation256_window38_20260911.json` 及 training 同名。仍是纯 mask/GT-context probe，不是完整训练 corruption；无训练改动或质量结论。
## 09-11｜historical38 window probe 完成，weight-decay pilot 已启动

- pipeline `79729`/cell `1708` exit0；audit `3edb77` exit0，checkpoint/manifest/split/script SHA、旧 sample hash、window38 payload37 均通过核验。fully masked train 30k→300k NLL B/H/R=`2.71286/2.98286/3.86607→.17197/.18340/.20020`，accuracy=`45.672/41.89461/31.20189→95.65423/95.28715/95.02664%`；validation NLL=`10.66111/11.66873/11.06442→13.30775/14.67705/14.36496`，accuracy=`1.77903/.38624/.88951→2.04822/.74906/1.49813%`，maxprob=`.38313/.33585/.31138→.61079/.57392/.59061`。window 不能解释泛化差距，非单一根因证据。
- weight-decay 单因素 50k pilot session `79369` 已在 GPU1 启动，目标 wd=`0.01`、control=`1e-6`；其余 VQ/cache/结构/mask/dropout/seed114514/300k schedule、50k stop、10k eval保持。manifest/optimizer 尚待实际核验，不另开 monitor。
## 09-11｜weight-decay pilot 启动核验通过

- run=`output/t2i_trans/2026-09-11-16-20-27_hhc_no_shape_weight_decay0p01_pilot50k_seed114514_v1`，W&B `8v983xtw`，session `79369` 已到 epoch0 batch45（约4.8 it/s）。root audit `4b08ef` exit0：相对 control training_manifest_at_50k，resolved args 仅 wd `1e-6→.01`、GPU0→GPU1、run/output/W&B 不同；cache_manifest/dataset_aggregates/loader_contract 逐值一致，source-derived bf16mixed/dropout.1/betas.5,.9/historical38/observational validation 保持。
- monitor cell `1717` 每600秒独占 session `79369`，等待 latest checkpoint 做 optimizer state 验收或训练终态；启动回收 cell `1713` 已完成但训练仍 live。optimizer 尚未由 checkpoint 核验。
## 09-11｜weight-decay pilot 5k checkpoint 验收

- root audit `92776` exit0（chunk `7b2329`）：`latest-000005000.ckpt` SHA=`2f1365eeed5bdddc94545e44276aa23cc70955568c0f1a17873f59db3c0b4446`，global_step=`5000`、epoch=`37`；AdamW 单 group wd=`.01`、lr=`.0002`、betas=`(.5,.9)`，447 params/309 state，optimizer step 全为5000，模型/optimizer finite，args loss=`historical_all_valid`、stop=`50k`一致。
- PID `4025534` 仍运行至约5718 steps（epoch42 batch90），非停滞；checkpoint 间隔为5000。旧 timer `1717` 不结束训练，monitor `1724` 每600秒等待 10k checkpoint 或 PID 消失；尚无10k质量结果。
## 09-11｜weight-decay pilot 10k checkpoint 固定，冻结评测已启动

- PID `4025534` 仍运行（native 10k eval 当时7/17）；10k ready 后仅终止 timer `1724`，session `79369` 继续。固定 hardlink=`output/hhc_weight_decay0p01_matched_validation_20260911/step_000010000.ckpt`；audit `25237` exit0（chunk `f3e63c`）：global_step=`10000`、epoch=`74`、optimizer step=`10000`、wd=`.01`、model floats finite，SHA=`1dc7671d6bc7e52d774fc9268a8c04960326d81c4834bd35ac2d83a598873d0e`。
- GPU0 544 pipeline `48301` 已启动，monitor `1729` 每600秒独占回收；启动 exact model load/requested VQ preserved，已生成 `32/544`，输出 `step_000010000_main.h5/_cpu.json/_physical_per_sample.json`；固定原 VQ175k、requested beta、no-shape/no-contact/no-noise、seed500123、50/T=.5/CFG3.5、save IDs。尚无评分，不追加 MM。monitor `1730` 每600秒只读等待 training latest20k 或 PID `4025534` 消失，session `79369` 保持。
