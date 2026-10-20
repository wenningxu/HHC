# HHC / Inter-X experiment handoff (2026-09-09 18:19 BST)

## Latest user override (2026-09-11)

用户已收束第①项并明确“可以开始下一个任务”，当前执行第②项公开shape augmentation。以`docs/PROGRESS.md`顶部为当前状态，不把下方旧运行列表当作仍活跃任务。62个SSP公开shape已转换；200例双估计器交集134/200=67%，134例训练overlay完整验证通过；全6238例任务session86163运行中，尚未全量验收。第③④项仍后置。

## Earlier user override (2026-09-11, VQ acceptance)

用户接受当前 VQ 质量“基本可以复现”，解除必须等待严格10项逐项不退化的探索阻断；任务回到 MModality 质量保持，采样/条件训练/token压缩/零shape在 scope，其他任务后置。严格门槛未过、原数值/流程来源限制与历史证据保留，不声称 bitwise 或十项全优；暂停进程是否恢复待 root 操作确认。

## Previous user override (2026-09-10)

Task 1 explicitly includes SAMPLING IMPROVEMENTS. The old common-50 sampler,
temperature 0.5 and CFG 3.5 are a reference control, not restrictions on new
sampling arms. Explore sampling algorithms, remasking/confidence selection,
temperature/CFG schedules and compute budgets with quality/MM/runtime measured
together. Label changed contracts explicitly; never mix them into the baseline.
The four-task order remains unchanged. The persistent thread goal is active.

September 10 morning progress:
- At14:34 BST stride4 VQ v2 FINISHED10000 updates, complete manifest and
  net_10000/net_iter_000010000/net_last checkpoints present. The SAME serial
  loop80349 has advanced to stride8; active W&Btfk63rcm, new manifest running,
  initial updates already advance past80. Main verifies paired manifests:
  differing resolved args ONLY down_t/exp_name/checkpoint_dir; dataset/stats/
  source artifact hashes are identical. Stride4 W&B completion is recorded,
  but the outer loop is still live; do not label outer session80349 exited.
  Child overnight_metrics assigned complete10k stride4 CPU reconstruction,
  then CPU joint/physical scores, outputs `stride4_step000010000_*` under
  `output/hhc_vq_stride_compression_paired10k_20260910`. No duplicate jobs.
  MGT training78308 now in native10k validation (its seed614514 differs from
  frozen paired exportseed500123; do NOT reuse native generated samples).
  Paired export19669 advances64/544 onGPU0, alongside stride8 training.
  Both memory use and observed slowdown are expected; no speed comparison
  between compression arms from this shared-GPU wall time.
- At14:32 BST new MGT10k paired main evaluation LIVE session19669 onGPU0.
  Atomic latest-000010000 checkpoint preserved by hard link as
  `output/hhc_joint_text_dropout_p0p2_matched_validation_20260910/candidate_step_000010000.ckpt`.
  Main torch.load verifies global_step10000, paper_text_cfg, text_drop_prob.2,
  seed114514; session54585 exit0. Exporter strict architecture load succeeds
  and preserves requested OLD net175000 VQ (not new VQ compression pilot).
  Generation544x1, seed500123,50steps/T.5/CFG3.5,batch32,no shape/contact,
  paper_text_cfg inference; then SAME GPU0 joint scorer as re-scored control.
  Outputs in same directory: candidate_step_000010000.{h5,json}.
  GPU0 shares VQ training (~8.6GB combined), GPU1 MGT continues (~18.1GB);
  this is quality evaluation, NOT a runtime comparison. No OOM; RAM39GB
  available. Original MGT78308/VQ80349 remain live. Do not duplicate export.
  On complete H5, request CPU physical paired scoring and inspect all three
  joint metrics before expanding to MM. Baseline control10k JSON already ready.
  New rolling5k checkpoint existed before10k; no unmatched5k generation was
  evaluated against10k control. This preserves stage-budget comparability.
- ~14:23 BST stride4 v2 checkpoint5k CPU joint/physical COMPLETE, child
  sessions41699/13963 exit0, files already assigned below. Main verified
  joint status/checkpoint hashes against corrected legacy VQ CPU evaluation.
  Full/Body/Hands FID=.308566405/1.337971119/.156177255;
  sliding=.327472366m/s, contactF1=.681648496. Body reconstruction distribution
  is still poor; do not hide it behind aggregate MPJPE. Legacy175k corrected
  CPU FID=.110864/.311243/.134449 is a reference with UNEQUAL training budget,
  not a causal stride4-vs8 comparison. Await matching10k/stride8 endpoints.
  W&B audits now show MGT7850 and VQ6350 optimizer steps, all recorded loss
  finite, VQ nonfinite-gradient/weight maxima zero. Original handles78308 and
  80349 remain live. Child overnight_metrics is idle after completed scores.
- At14:20 BST FIRST trained VQ frozen reconstruction COMPLETE: stride4 v2
  checkpoint5000, child CPU eval session63526 exit0; original monitor99476
  found a fully loadable checkpoint before evaluating. Output
  `output/hhc_vq_stride_compression_paired10k_20260910/stride4_step000005000_reconstruction.{json,h5}`.
  Main independently verified544 names/lengths/request hashes,71026 valid
  frames, checkpoint_iteration5000/stride4, completed marker and finite H5.
  RawRMSE=.094157318625, MPJPE=.165687442387m, handMPJPE=.1922480986m,
  root translation error=.096818728725m, rotation error=.1369291816rad.
  Body/hands/root active codes2992/3749/2419, perplexities1848.41/2461.36/
  1281.68 over17895 tokens each. Code usage is not conditional MModality.
  This is only stride4's5k reference, not compression or generation success.
  VQ loop80349 remains live beyond5400 steps. Child `overnight_metrics` now
  assigned CPU joint + physical-v2 scoring of this COMPLETE H5, outputs
  `stride4_step000005000_joint_cpu.json` and
  `stride4_step000005000_physical_per_sample.json` alongside it. Do not
  duplicate its jobs. Stride8 still queued after stride4 reaches10k.
- ~14:10 BST both original handles78308(MGT)/80349(VQ) confirmed live.
  Main read frozen VQ evaluator end-to-end: it restores checkpoint down_t,
  pads normalized input to each actual stride, masks padding, scores only
  requested valid frames, and writes request/checkpoint hashes. No source edit
  needed. Low-cost child `overnight_metrics` is now MONITORING stride4 v2
  first checkpoint `net_iter_000005000.pth`; after complete torch.load/iter5000
  validation it will run CPU batch8/OMP4 reconstruction on SAME544 requests.
  Planned outputs `output/hhc_vq_stride_compression_paired10k_20260910/stride4_step000005000_reconstruction.{json,h5}`.
  No checkpoint or reconstruction result exists yet. Preserve child handles,
  do not duplicate its future evaluation; missing5k while training is normal
  waiting, not a blocker. No joint scoring assigned to this child yet.
- Recovery verified~14:07 BST: VQ loop80349 remains LIVE; stride4 v2 W&B
  iak9i2o5 (`wandb/run-20260910_140525-iak9i2o5`). Actual telemetry through
  optimizer300:6/6 finite loss records, latest.4679961, zero nonfinite
  gradients AND weights at every recorded point. Main inspected values, not
  merely audit exit status. Audit artifact
  `output/hhc_vq_stride4_rotationfp32_v2_live_progress_20260910.json`.
  First audit had no history yet because buffering; same live handle retained,
  no duplicate restart. Stride8 remains queued. Need eventual 5k/10k frozen
  reconstruction assessment; passing startup does not prove compression quality.
- IMPORTANT14:06 BST correction: FIRST VQ loop65467 EXITED1 at optimizer50
  telemetry (loss/weights/gradients nonfinite); stride8 NEVER launched.
  Keep failed h91qllqo directory and manifest as failure evidence, not healthy
  training. Main reproduced actual ReConsLossInter.forward_rotation endpoint:
  FP32 finite gradients, bf16 loss0 with1320 nonfinite gradient entries on
  identical valid rotations. bf16 bmm makes cosine bf16; clamp1-1e-7 rounds
  to1, so acos backward is singular. This is a verified numerical defect,
  not evidence that temporal compression is infeasible.
  Fixed only `utils/losses.py` rotation matrix products/cosine precision with
  autocast-disabled FP32, including single-person and relative-orientation
  siblings. Did NOT disable geometry losses or edit shared VQ/MGT model code.
  MGT live source-contract list does not include this VQ-only loss module.
  Regression `tests/test_rotation_loss_autocast.py` covers CPU/CUDA bf16,
  endpoint/nearby values, all3 methods and exact FP32 value/gradient parity.
  Together with stride tests4 tests PASS, session94027 exit0.
  New distinct VQ loop80349 LIVE, same paired10k config but checkpoint/exp
  suffix `_rotationfp32_v2`. First stride4 then stride8, no reuse of corrupted
  weights or overwrite of failed artifacts. Must verify real-data telemetry
  beyond50 before calling recovery successful. GPU1 MGT session78308 remains
  healthy; local audit2250 steps,45 finite records, latest loss5.716153.
- At14:02 BST new ACTUAL VQ compression pilot is LIVE onGPU0 session65467,
  direct immutable shell loop: stride4/down_t2 first, then stride8/down_t3.
  Both fresh10000 updates, integrated1000-step warmup/cosine to10k (NOT a
  continuation of175k), seed114514, batch256/window64, raw normalization,
  codebooks12288, code_dim256 => shape width8, width512/depth3, stride_t2,
  bf16, shape mask0.1, LR2e-4/weight decay1e-6, commit.02, same default
  physical/reconstruction losses. Explicit official train/val split paths.
  Checkpoint roots `checkpoints/hhc_vq_stride{4,8}_paired10k_seed114514_20260910`.
  New directories prechecked absent; no old VQ/checkpoint/cache overwritten.
  Stride4 W&Bh91qllqo online, local `wandb/run-20260910_140152-h91qllqo`;
  actual manifest and initial training updates verified, GPU0~5GB, RAM33GB
  available. Stride8 is QUEUED, not running yet. Expected stride4~22min before
  validation overhead, not a promised completion time. Freeze shared sources.
  THREE stride tests passed session41210 exit0, including actual forward/
  backward for both strides. Child incorrectly said no stride8 smoke; main
  independently found and reran `tests/test_vq_temporal_stride.py`.
  IMPORTANT: native VQ validation loader uses unit_length=temporal_stride,
  changing crop lengths across arms. Do NOT directly compare native valFID.
  After both checkpoints exist, use the SAME frozen544 raw requests/lengths
  through `tools/eval_vq_interx_reconstruction.py`, then identical joint/physical
  scorers. This pilot measures compression/reconstruction learnability only;
  no MModality success claim, no reuse of old stride4 MGT/token cache with8.
  A promising tokenizer still needs corresponding MGT training and quality
  comparison against the current300k baseline, not merely these10k pilots.
- Matched MGT control10k/20k re-scoring COMPLETE onGPU0, child sessions
  73350/26963 exit0 and main JSON verified. Outputs
  `output/hhc_joint_text_dropout_p0p2_matched_validation_20260910/control_step_0000{1,2}0000.json`.
  Actual filenames `control_step_000010000.json`, `control_step_000020000.json`.
  Full/Body/Hands FID10k=.468288678/1.229596516/.818115690;
  20k=.558744721/.573019360/.618051646. Existing H5 requests/settings match;
  old JSON omitted device, so rescore rather than assume GPU/CPU parity.
  Source generations `output/hhc_no_shape_v4_pilot50k_official_val/step_*.h5`.
  MGT session78308 remains LIVE onGPU1; local W&B audit at~13:58 verified
  optimizer700,14/14 finite loss records, latest8.049718. Artifact
  `output/hhc_joint_text_dropout_p0p2_live_progress_20260910.json`.
- At13:55 BST probability-matched TRAINING IS LIVE on GPU1, session78308:
  `bash tools/run_hhc_no_shape_joint_text_dropout_pilot.sh --device cuda:1`,
  OMP/MKL4. New run
  `output/t2i_trans/2026-09-10-13-55-10_hhc_no_shape_joint_text_dropout_p0p2_pilot50k_seed114514_v1`,
  W&B781f0cgp online; initial optimizer batches advance with finite loss.
  Freeze launcher/options/training/model sources while this run is live.
  Training-only --text-drop-prob is now IMPLEMENTED: defaultNone resolves
  to unchanged0.1; explicit override only paper_text_cfg and finite[0,1].
  Assignment is in MaskGITLightning, shared model/sampler unchanged. Effective
  probability recorded in existing manifest, raw value in strict resume keys;
  altered probability is not mislabelled complete paper training contract.
  Four new tests pass(session47269 exit0), including actual joint masks,
  eval/forced-mask invariance, old manifest missing-value resume and rejection
  of changing old-default resume to0.2. Trace/floor combined7 tests also pass.
  Main compared actual new/control manifests: seed/cache/VQ/loss/mask/model/
  LR/batch/total-iteration/validation settings match; new run fresh, stopping
  at50k against historical all-valid checkpoint `model-000050000-0.4075.ckpt`
  in September7 no-MGT-shape v4 directory. Do NOT use September1 seed123
  all-valid control (wrong model/cache/conditioning); child lookup corrected.
- Paired sampling session28490 FINISHED exit0 at~13:54 BST. Seed500125
  candidate FID=.250695395/.241345829/.240520427; FullMatching3.090760918,
  FullR3=.755514706. All THREE new seeds worsen Body/HandsFID and FullR3
  relative to their paired controls. No quality-preserving promotion; retain
  MM gain as mechanism evidence, not final solution. Last CPU physical scores
  session66114/69481 exit0 independently checked: control/candidate125
  sliding=.303833448/.300669145, F1=.556776373/.552493966;544 rows each.
  All paired-main/physical jobs finished. GPU0 is free. Next collect new
  training10k/20k checkpoints and compare matched control checkpoints, while
  designing coherent sampling/token representation interventions from diagnosis.
- Follow-up13:50 BST: session28490 confirmed live on LAST candidate
  seed500125, generated160/544; paired control125 completed with FID
  .244232705/.234418918/.215346434, FullMatching3.036601053.
  GPU1 is free; no new training launched. Low-cost agent audited the next
  training-only probability override: add optional --text-drop-prob default
  None; reject overrides outside paper_text_cfg or outside[0,1] (including
  NaN); set trans_encoder.trans_base.text_drop_prob in MaskGITLightning,
  record effective probability in existing manifest and raw value in resume
  identity. Missing old manifest field maps toNone; existing source-hash resume
  protections must remain. Keep model default0.1 unchanged; next pilot uses
  explicit0.2 and new experiment identity. Current exporter IMPORTS training
  module, so defer those edits until paired session28490 terminates.
  User's order explanation was clarified: current noise intervention changes
  temporal POSITION commitment priorities within a part; it does not freely
  exchange root/body/hands cascade order (partwise is a separate rejected arm).
- By13:47 BST: GPU1 paired session30441 exited0; seed500126 candidate
  Full/Body/Hands FID=.229256/.246098/.222834 versus paired control
  .245737/.234425/.212885. GPU0 session28490 remains LIVE, now seed500125;
  completed candidate500124 FID=.256086/.240489/.232299 versus control
  .242118/.239069/.213049. Both new candidates worsen Body Matching and
  FullR3. Original-seed FID gains are NOT robust; defer formal30-repeat MM.
  CPU physical per-sample scoring (low-cost child, all exit0 sessions
  12786/68897/47575/98043) independently checked by main:
  seed124 control/candidate sliding=.303541607/.301465875,
  F1=.559557203/.554279381; seed126 sliding=.303563661/.304593028,
  F1=.556742390/.557312946. Files alongside main results, suffix
  `_physical_per_sample.json`;544 rows, matched request hashes, valid_frames.
- Length-quantile trajectory follow-up COMPLETE session81856 exit0:
  `output/hhc_sampling_trajectory_length_quantiles_val8_20260910`.
  Request indices95,443,535,463,76,224,358,527; lengths40,96,116,140,
  150,150,150,150; metric-independent selection,4 repeats,3 policies.
  Trace parity tokens AND CPU/GPU RNG passes for every arm repeat0.
  Main recomputed48 pairs: final identical control42, noisy logprob12,
  noisy logodds20. First shared committed-value differences: control
  stages140..142, logprob1..147, logodds24..129. Nonargmax accepted across
  all repeats/parts:8/3000,21/3000,8/3000 respectively. Confirms order/context
  mechanism beyond all150-frame prefix, still a small diagnostic sample.
  Trace/floor unittest suite4 tests exit0(session26193).
- Original-seed physical per-prompt counts and paired bootstrap now complete;
  full protocol/CIs and failure-selected case interpretation are recorded at
  top of `docs/hhc_mmodality_shape_condition_diagnosis.md`. ContactF1 and
  sliding CIs cross zero, NOT a noninferiority pass. No active training.
  Next: finish seed125 same-device pair, collect physical counts, then decide
  next intervention. Probability-matched joint text dropout still requires
  training-only config/manifest/resume support; not implemented or launched.
- By13:26 BST previous sessions31601/24378 both exited0. Final MM complete:
  log_odds_noise1 Full/Body/Hands0.394574463/0.425801575/0.624503970,
  duplicates1767/5500=0.321272727, unique5.82. Main/physical results above
  make this a promising, not yet quality-preserving, candidate. partwise_noise1
  MM1.030373812/1.154445767/1.674919248, duplicates423/5500=0.076909091,
  unique9.28; quality still worse. Both complete H5 markers independently checked.
  EXTRA paired main validation now LIVE: GPU0 session28490 sequentially runs
  control/candidate seed500124 then control/candidate seed500125; GPU1
  session30441 runs control/candidate seed500126. Directory
  `output/hhc_log_odds_paired_main_validation_20260910`, files
  `{control,log_odds_noise1}_seed{500124,500125,500126}.{h5,json}`.
  Direct shell loops over existing exporter/scorer;544 main/request,50steps,
  T0.5/CFG3.5, same checkpoint, save-token-ids. Compare each pair on SAME GPU.
  No new training launched. Complete quality replication before formal MM.
- Trajectory diagnosis IMPLEMENTED and COMPLETE, GPU1 session4701 exit0:
  `tools/trace_hhc_sampling.py`, outputs
  `output/hhc_sampling_trajectory_val8_20260910` (summary +12 NPZ files).
  Uses native Python tracing at actual commitment statement, not a copied
  sampler; no production source edit. Saves proposals, argmax, probability,
  proposal entropy, logit margins, accepted masks, before/after IDs and final
  decoded motion for first8 frozen validation requests,4 repeats,3 policies.
  Dedicated batch8/seed700123 differs from headline batch32/seed600123;
  NOT a headline metric/timing result. Every arm's first repeat verified exact
  unobserved/observed tokens AND CPU/GPU RNG state on actual checkpoint.
  Ten trace/cascade/selection regression tests pass; trace removed on failure.
  Main compared6 repeat pairs x8 prompts =48 trajectories perpolicy:
  control34/48 never diverge; remaining14 first committed-value differences
  only at stages147..149 (the FINAL root/body/hands iteration).
  Noisy log-probability48/48 diverge in selected-position state at stage0;
  first shared-position code differences can appear much earlier than control.
  Noisy log-odds48/48 have some position-state divergence,19 atstage0,
  but13 reconverge to identical final IDs. First committed-value divergence
  for the other35 spansstage38..149. This directly localizes collapse to
  early/middle commitment trajectories in these8 prompts, not merely final
  decoding or the evaluator. It does NOT identify the cause of contact loss.
  IMPORTANT mechanism refinement from actual full trajectories: total accepted
  non-argmax tokens across all3 parts/4 repeats are only6/3648 control,
  7/3648 noisy log-probability,6/3648 noisy log-odds (main recomputed from
  NPZ masks/proposals/argmax). Thus this sample does NOT support attributing
  the MM gain to accepting substantially more low-probability alternatives.
  Different commitment order changes later contexts and their argmax choices.
  Earlier first-step selection audits are a separate limited diagnostic and
  must not be extrapolated into an end-to-end non-argmax acceptance claim.
  Scope caveat: all first8 requests have150 frames, so this is a long-motion
  diagnostic, not a length-stratified validation sample.
- At13:15 BST main inspected all4 fixed-case control/log_odds_noise1 images
  in `output/hhc_log_odds_qualitative_validation_20260910/figures` (hug,
  high-five, handshake, kick). Renderer session44134 exited0;9 selected
  names/indices match prior metric-independent selection. Gross pose sequences
  look similar; sparse skeleton snapshots do not prove fine contact, sliding,
  penetration or broad semantic quality. Do not claim a visual quality pass.
  User discussed the research plan: pursue coherent early-to-late sampling,
  training/inference text consistency, then partwise/hierarchical temporal
  token representations and conditional data support. These are hypotheses,
  not implemented improvements. For the next text-dropout training arm,
  prefer probability0.2 matched to historical global dropout to isolate
  which text channels are masked; current launcher still uses joint0.1.
  A configurable training probability is NOT yet added. Preserve all live
  sampling sources until sessions31601/24378 finish, then decide from MM.
- By13:10 BST both noise1 counterparts have COMPLETE main quality scores.
  log_odds_noise1 is a promising BUT UNPROVEN candidate: Full/Body/Hands
  FID0.221011912/0.232894442/0.205683841 (all lower than control), Matching
  2.999605277/3.087571270/3.090443373 (Full better, Body/Hands slightly worse).
  FullR1/2/3=0.452205882/0.650735294/0.761029412; R3 slightly worse.
  CPU physical session4062 exited0: sliding0.302461540 improves versus
  control0.303470450, but contactF1=0.551139772 worsens versus0.557580010.
  Main independently checked checkpoint/VQ/request/seeds/steps/T/CFG/contracts
  match control and H5 completion is true. MM is LIVE in session31601.
  partwise_noise1 FID0.252601714/0.232149699/0.268754087; Matching
  3.042182782/3.101772266/3.114898808. Physical session2723 exit0:
  sliding0.310734272/F1=0.551983067. Its MM is LIVE in session24378.
  If log_odds_noise1 MM gains meaningfully, prioritize extra paired main
  generation seeds (same-device control/candidate, frozen validation) before
  expensive formal30-repeat MM; do not equate single-seed all-FID wins with
  quality preservation. Keep text-dropout training as a parallel next option
  after current samplers finish. No production source or live launcher edits.
- By13:01 BST log_odds and partwise deterministic main/MM are COMPLETE;
  both MM JSON statuses and all11 H5 completion markers verified.
  log_odds Full/Body/Hands MM0.044626508/0.048965801/0.085006565,
  duplicates3622/5500=0.658545455, unique2.57. No substantial collapse fix.
  partwise MM0.059206855/0.061540172/0.083022900,
  duplicates3374/5500=0.613454545, unique3.0. Modest MM gain does not
  compensate for the measured Full/Hands FID, Matching and sliding losses.
  Completed-arm walltimes1331s/1310s include loading+generation+scoring.
  Original live session31601 now runs log_odds_noise1 main onGPU0;
  session24378 now runs partwise_noise1 main onGPU1. Both loaded checkpoint
  successfully. These are the final arms in each launcher; do not restart
  or edit their sources. Next: main quality + CPU physical, then MM. If no
  quality-preserving candidate, prioritize the preflighted joint-text-dropout
  training diagnosis over more dense scalar noise sweeps.
- Training-side fallback preflight (NOT launched): existing
  `tools/run_hhc_no_shape_joint_text_dropout_pilot.sh` resolves to fresh
  `hhc_no_shape_joint_text_dropout_pilot50k_seed114514_v1`, historical_all_valid
  loss + paper_text_cfg training, no shape/contact, same VQ/cache/mask/seed,
  total300k with diagnostic stop50k. Late CLI `--device cuda:1` overrides
  base cuda:0. Both Bash syntax checks pass; paths exist and no prior matching
  run directory was found. Child verified statically; main inspected source.
  Main CPU behavioral check exit0: hooks before word/global projections show
  historical contract retains word features when global is masked, whereas
  paper_text_cfg masks both (tested forced inference and train drop probability1).
  Default joint probability0.1 yields synchronized masks for4096 synthetic
  items, observed fraction0.103027 with seed114514. This verifies plumbing,
  not a learned quality/MM benefit. Actual historical probability0.2 versus
  joint0.1 remains a declared conditioning-contract intervention, not a
  probability-matched ablation. Await current noise1 sampler arms before
  allocating a GPU to training; do not restart the rejected masked-only arm.
- By12:47 BST both new deterministic arms have COMPLETE main H5/JSON;
  their11-repeat MM is running on the original sessions31601/24378.
  log_odds Full/Body/Hands FID0.239769/0.230698/0.211173,
  Matching3.054476/3.090738/3.087238. FullR1/2/3
  0.441176/0.647059/0.761029. FID near control, Matching worse.
  partwise FID0.258888/0.227015/0.249594,
  Matching3.092749/3.134013/3.087526. FullR1/2/3
  0.441176/0.628676/0.742647. Strict part completion alone does NOT solve
  the quality issue. Main H5 completion, token archive shape544x1x50x3,
  ranking/schedule/step-unit metadata and model source hash verified.
  CPU physical scores complete: log_odds sliding0.303752623/F1=0.554989075
  (session51166 exit0); partwise sliding0.307425054/F1=0.570937446
  (session22807 exit0). Main rechecked544 samples and physical v2 protocol.
  Neither deterministic arm improves all quality measures. Noise1 counterparts
  remain queued in the existing two-arm launchers. No source changes.
  Completed14-arm screening table added to
  `docs/hhc_mmodality_shape_condition_diagnosis.md`; not formal results.
- By12:39 BST two NEW sampling grids launched after both old jobs exited0:
  GPU0 session31601 `log_odds` grid (log_odds, log_odds_noise1), in
  `output/hhc_no_shape_confidence_validation_20260910`; GPU1 session24378
  `partwise` grid (partwise, partwise_noise1), in
  `output/hhc_no_shape_sampling_validation_20260910`. Each has544 main and
  100x11 MM, same checkpoint/request/seed/T0.5/CFG3.5/50steps and same-device
  control. No training change. Archive sampled IDs. Freeze model/exporter/
  scorer/launcher until these grids finish; do not edit live shell scripts.
  `sample_intra_cas` now has default-preserving confidence_ranking
  (probability/log_odds) and channel_schedule (interleaved/partwise).
  Log-odds helper moved from audit into model; audit imports same implementation.
  Partwise completes root, then body, then hands, each with50 steps (NOT50/3),
  max300 CFG forwards total. New metadata records per-channel step units,
  ranking/schedule/source hash; exporter resume and MM scorer reject undeclared
  policy changes in either direction. Defaults remain unchanged.
  Thirteen regression tests PASS, including golden default IDs AND RNG,
  strict part ordering/CFG budget, END/PAD, reproducibility, stable log-odds,
  resume protection and logit-sharpness. Scorer self-check and Bash syntax PASS.
  `output/hhc_sampling_runtime_after_order_policies_20260910.json` passes all
  default runtime checks (150 active stages/300 CFG forwards at38 tokens).
- By12:39 BST steps20 session40025 finished exit0, main/MM H5 and JSON
  completion verified. MM Full/Body/Hands0.051785/0.073034/0.072629,
  duplicates3698/5500=0.672364, unique2.81; no quality-preserving improvement.
  Whole-arm loading+generation+scoring walltime550s, not sampling-only timing.
  Noise_strength session26261 also finished exit0. noise0p25 MM
  1.028576/1.094035/1.617258, duplicates196/5500=0.035636, unique10.12;
  all completion markers verified. Lower all-part noise retains large MM
  but does not solve Hands/physical quality loss. Do not promote either arm.
- At12:35 BST steps20 main is COMPLETE, MM still running in session40025.
  Full/Body/Hands FID0.247113/0.224632/0.216192; Matching
  3.051890/3.067592/3.093721. FullR1/2/3=0.4375/0.632353/0.755515.
  CPU physical v2 session19796 exited0: sliding0.303527385,
  contactF1=0.555151576. This is not a quality improvement over50steps.
  GPU1 noise0p25 is in its last MM repeat. No production sampler/source edit
  until both current jobs finish. New-policy regression tests are staged.
- At12:27 BST root_strength launcher session48529 finished exit0. Both
  root_noise0p5/root_noise0p25 main/MM JSON and H5 completion markers verified.
  root_noise0p25 Full/Body/Hands MM0.958262/1.108113/1.502831,
  duplicates580/5500=0.105455, unique8.86. Full quality/MM are promising,
  but Body/Hands FID and sliding tradeoffs remain; no quality-preserving claim.
  GPU0 now runs a ONE-arm steps20 comparison in session40025, output
  `output/hhc_no_shape_confidence_validation_20260910/steps20_{main,mm}.{h5,json}`.
  Same GPU0 control/request/checkpoint/seeds, T0.5/CFG3.5/no confidence noise,
  token archival enabled. Direct four-command export/score sequence, not the
  budget shell grid (which would also run steps10). Model/launcher source is
  unchanged. Final shell wall_seconds will include loading+generation+scoring,
  not isolated sampling latency; existing control H5 does not record timings.
  GPU1 session26261 still runs final noise0p25 MM; main is complete with
  FID0.218422/0.209054/0.270148, FullMatching3.003055, sliding0.312578,
  contactF1=0.552738. Further dense all-parts noise-strength sweeps are lower
  priority than stable ranking and partwise scheduling given weak quality trend.
- At12:15 BST noise1_temp0 main/MM are COMPLETE: Full/Body/Hands MM
  1.161310/1.230506/1.814111, duplicates118/5500=0.021455, unique10.41.
  Nearly the same MM as noise1/T0.5 despite effectively argmax token proposals:
  commitment-order randomness is a major contributor. Hands FID0.279680,
  sliding0.314164 and contactF1=0.536439 still fail quality screening.
  GPU1 session26261 now runs final noise0p25 arm.
  root_noise0p25 main FID0.200182/0.245033/0.233905, Matching
  3.003983/3.085111/3.046208; FullR1/2/3=0.472426/0.672794/0.773897.
  Sliding0.309054 worsens; contactF1=0.569776 improves. MM is running onGPU0
  session48529. Hands FID gap narrows but quality preservation is unproven.
- VQ part-coupling probe is COMPLETE (`output/hhc_vq_part_neighbors_val32.json`).
  Existing probe now supports `--intervention part_neighbors`; four tests pass.
  Only one token stream is replaced with nearest-other codes at a time, using
  the same32 frozen GT crops, VQ and shape. Normalized output delta MSE matrix
  (rows changed body/hands/root; columns decoded body/hands/root):
  [0.076158,0.022948,0.006676], [0.014189,0.043696,0.000901],
  [0.008958,0.004031,0.016850]. Baseline full reconstruction MSE0.3171808665
  exactly matches the earlier probe. This confirms cross-part decoder effects,
  not a bug or a quality/MM improvement. Current VQ has three encoders but ONE
  shared decoder; the old sep_decoder branch is incomplete and must not simply
  be enabled. Fixing root IDs does not guarantee fixed decoded root geometry.
- Next sampling hypothesis after both live grids finish: compare strict
  partwise root-complete→body-complete→hands-complete ordering against the
  current interleaved cascade, preserving per-part step budgets/CFG/T/noise
  and END/PAD rules. Later root commitments currently change conditions after
  body/hands tokens have already been fixed; shared decoding makes this worth
  testing. Existing legacy `sample_cascade` is NOT a valid drop-in control:
  it lacks the current CFG/shape arguments and differs in remasking/temperature.
  Reuse the current sampler with explicit loop-order policy instead of copying
  the old implementation. Isolate scheduling from log-odds ranking; neither
  change is installed or launched yet. Root-prefix idea remains unimplemented
  and is lower priority given the measured shared-decoder coupling.
- At12:07 BST root_noise0p5 main/MM are COMPLETE: Full/Body/Hands MM
  0.974518/1.187322/1.576289; duplicate462/5500=0.084, unique9.18.
  It still fails Body/Hands FID and sliding (see main metrics below).
  GPU0 session48529 now runs root_noise0p25 main. GPU1 session26261 runs
  noise1_temp0 MM; its main FIDs0.214321/0.184310/0.279680 show Hands
  degradation persists even with effectively argmax token proposals.
- Relative-root trajectory audit (100 prompts/all55 pairs, complete files):
  control/noise1-all/noise1-root/noise0.5-all mean pair RMS meters
  0.014938/0.479574/0.487199/0.483688; time-centered RMS
  0.012963/0.225811/0.218519/0.227938. Correct GLOBAL dynamic-energy ratios
  sum(centered_delta^2)/sum(delta^2) are0.766998/0.235656/0.212079/0.235549.
  Exact zero-difference pairs contribute to neither energy sum; do not call
  the control ratio near zero because it has many exact duplicate pairs.
  Rotation-invariant distance ||root2-root1|| gives mean pair RMS meters
  0.003962/0.172588/0.177209/0.174923, centered0.003327/0.110233/0.112156/0.110868,
  dynamic-energy ratios0.681386/0.380040/0.377514/0.375987. These are relative
  placement diagnostics, not proof of semantic MM or mesh quality.
- Minimal initial-horizontal-distance diagnostic is implemented in
  `tools/diagnose_generated_initial_distance.py`, test passes. It shifts only
  person2 root x/z by a constant to match the paired GENERATED control's
  initial horizontal separation, never GT. Output is separate, exclusive-create,
  source/reference hashes and extra reference-generation cost are recorded.
  `output/hhc_generated_distance_anchor_validation_20260910/confidence_all_anchored_main.h5`
  is complete, mean shift0.131713m, zero degenerate directions. Full544 check:
  nontranslation channels/padding bitwise unchanged, max velocity change
  7.15e-6m/s (float32), max distance error2.38e-7m. Physical-v2 exit0:
  sliding0.313003455m/s unchanged; contactF1 drops0.541148→0.516802.
  Do NOT expand this into an MM/postprocessing solution on present evidence.
  Anchored joint CPU scorer session19618 completed exit0 at12:08: Full/Body/Hands
  FID0.226624/0.222349/0.215937, Matching3.220214/3.312205/3.197467.
  Unaligned same-CPU noise1-all FID0.203157/0.209598/0.265925,
  Matching3.039146/3.062837/3.110037. Hands FID recovers substantially but
  retrieval and contact worsen; reject this post-shift intervention.
  This suggests investigating geometry-consistent early root decisions during
  sampling (e.g. a generated, non-GT root prefix) rather than shifting poses
  afterward. Not implemented; finish live grids before model source changes.
  CPU control and unaligned noise1-all main JSONs are complete in confidence
  root (`control_main_cpu.json`, `confidence_all_main_cpu.json`); original
  subagent deferred handle was lost, so exact shell exit_code is unverified,
  but both scoring files are complete and its shell/scorer PIDs are absent.
- At11:55 BST noise0p5 main/MM are COMPLETE: Full/Body/Hands MM
  1.068956/1.219152/1.700170, duplicates150/5500=0.027273, unique10.26.
  It retains most of noise1's diversity but still fails Hands/physical quality.
  GPU1 session26261 has moved to noise1_temp0 main; noise0p25 follows.
  root_noise0p5 main is complete: Full/Body/Hands FID0.206120/0.252473/0.263400,
  FullMatching2.978846, FullR1/2/3=0.474265/0.670956/0.781250.
  Sliding0.310300m/s worsens vs0.303470; contactF1=0.570801 improves.
  GPU0 session48529 is generating root_noise0p5 MM; root_noise0p25 follows.
- Four metric-independent qualitative cases are rendered and visually inspected
  under `output/hhc_sampling_qualitative_validation_20260910/figures`:
  hug/high-five/handshake/kick, four rows control/all-noise1/root-noise1/all-noise0.5.
  Gross action structure is largely retained in these snapshots; they cannot
  establish hand-detail, temporal smoothness, penetration or broad noninferiority.
  Existing selector/renderer reused with symlinks, no generation files copied.
- New CPU diagnostic completed (`hhc_no_shape_confidence_saturation_val8.json`):
  untempered first-step CFG softmax top1 equals exactly1 for20.36% body,
  17.82% hands,13.09% root tokens (all0% before CFG). This supports investigating
  saturation of probability ranking, not claiming a complete collapse cause.
  `tools/audit_mgt_mask_exposure_gap.py` now reports this separately from
  proposal-temperature entropy. Only root's fully masked context is an actual
  first cascade stage; body/hands remain counterfactual.
- A stable selected-token log-odds ranking is prototyped ONLY in
  `tools/audit_sampling_selection.py`, not in the production sampler/exporter.
  It explicitly excludes the max logit when computing the remaining mass.
  CPU128-draw val8 probe `hhc_no_shape_log_odds_selection_val8.json`:
  root mean distinct first-committed positions per prompt is1 for deterministic
  confidence,32 for noisy log-probability,3.125 for noisy log-odds (same noise1).
  Log-odds accepted0/1024 non-argmax root proposals vs19/1024 log-probability.
  This offers a gentler-order hypothesis, not demonstrated end-to-end MM/quality.
  Wait for both live grids to finish before changing model source (source hashes
  bind export resume). Next tests should isolate deterministic log-odds versus
  noisy log-odds and original control, not conflate scoring and noise changes.
- At11:42 BST GPU0 zero-beta main/MM are COMPLETE, and all four confidence
  arms passed JSON/H5 completion audit. Zero-beta Full/Body/Hands MM:
  0.0421190634/0.0589297116/0.0834255069 vs control
  0.0439132303/0.0479640439/0.0830070376. Duplicate3732/5500 and mean
  unique2.59 are unchanged. Full/Hands FID degrade (see below); reject this
  zero-beta setting for the current no-MGT-shape checkpoint.
  Old session84164 exited2 after successful final scoring with the known
  live-script-edit trailing parse error; no results were rerun.
  GPU0 now runs root_strength (root_noise0p5/root_noise0p25), session48529.
  GPU1 noise_strength remains session26261, currently noise0p5 MM.
- Redundant CPU zero-beta jobs were deliberately interrupted AFTER the GPU
  paired experiment completed: exporter PIDs3406450/3489684 received SIGINT;
  original sessions30798/81289 both exited130. All files are retained, not
  deleted. Interrupted H5 files reopen normally with incomplete-repeat markers;
  do not score them as complete or restart the extra17-hour CPU MM sequence.
- At11:38 BST noise0p5 main is complete: Full/Body/Hands FID
  0.215512/0.202943/0.264532, Full Matching3.036350, FullR1=0.422794.
  Physical-v2 CPU scoring exit0: sliding0.3131554m/s, contactF1=0.5448400.
  Hands FID and both physical proxies still worsen; MM is running.
  First real token archive passed shape/int32/valid-code/END/PAD checks:
  (544,1,50,3), not38 storage slots. Exporter model block_size=51;
  actual request payload lengths use at most38 tokens, rest are END/PAD.
- VQ new-run telemetry now uses profile `vq_autoencoder_body_root_hands`.
  Canonical perplexity names correctly map full-forward tuple[0,1,2] to
  body/root/hands; token IDs still use body/hands/root. Historical right/left
  aliases and old profile are retained, old run files untouched. Contract
  auditor accepts the new profile and the direct VQ policy regression passes.
  This changes training source/manifest hashes: do not silently resume an old
  strict-contract run with the changed training source. No VQ training launched.
- Calibration completion audit: all five arms have complete main/MM JSON and
  all-true H5 completion markers. Matched-text-dropout Full/Body/Hands MM is
  0.096480/0.114976/0.146190, duplicate2916/5500=0.530182, unique mean4.24.
  Full FID0.196494 improves, but Matching3.060419 and Hands FID0.248330
  worsen versus control. CPU physical-v2 score completed exit0:
  sliding0.2940446m/s improves, contactF1_10cm0.5519905 worsens.
  This arm is not a quality-preserving success. No new training run launched.
- At11:30 BST, GPU1 calibration finished all five arms including matched-text
  dropout generation/scoring. Session88355 then exited2 with a trailing shell
  parse error: editing the live launcher had changed its unread file contents.
  Do not report the whole launcher as exit0 or rerun completed arms. The current
  script passes `bash -n`; freeze it while either GPU launcher is live.
  GPU1 now runs noise_strength in session26261 (noise0p5, noise1_temp0,
  noise0p25). GPU0 session84164 is still generating zero-beta MM.
- Zero-beta main Full/Body/Hands FID is0.295673/0.225125/0.278802 versus
  control0.240975/0.234513/0.210871: zero shape is not a quality-preserving
  improvement in this screen. MM remains pending at11:30 BST.
- Future noise_strength/root_strength exports now opt into raw token archival
  (`--save-token-ids`, int32 H5 sample/repeat/time/body-hand-root layout).
  Default generation is unchanged; incompatible archive resume is rejected.
  Five archive/sampler tests passed and launcher syntax passed. This is token
  storage only, not an implemented decoder replay or a measured quality gain.
- Masked-only five-candidate selection and comparison are COMPLETE. Decision
  `do_not_continue_masked_only_arm` in
  `output/hhc_no_shape_masked_only_pilot50k_comparison.json`.
  Joint validation FIDs at 10/20/30/40/50k are
  0.60879/0.99897/1.18982/1.13180/1.11956. Selected10k is worse than
  all-valid selected30k FID0.33935. Its missing MM stays null; failing quality
  alone suffices to reject continuation, but missing MM can never permit
  promotion. All checkpoints preserved; do NOT launch the masked-only300k script.
- New confidence-noise inference code is now implemented in the model and
  exporter, AFTER masked-only selection finished. It keeps proposal T/CFG
  fixed and adds linearly annealed Gumbel noise to log-confidence ranking,
  either all parts or root only. Scale0 preserves original outputs and RNG
  in the pre-change golden test (`tests.test_cascade_sampling_reference`).
  Changed policies reject cross-policy resume and are explicitly labelled.
  Training-bound model source has consequently changed; do not assume old
  strict training resume source hashes still match.
- GPU0 now runs `bash tools/run_hhc_no_shape_sampling_validation.sh cuda:0 confidence`
  (session84164): matched GPU0 control, noise1/all, noise1/root, zero-beta.
  Outputs `output/hhc_no_shape_confidence_validation_20260910`.
  This repeats control on the same device before new comparisons and should
  finish the zero-beta idea sooner than CPU alone. GPU1 session88355 continues
  its calibration grid; CFG2.5 main and MM are COMPLETE, temperature annealing
  is running. At 10:26 BST both GPU jobs were checked healthy.
- CFG2.5 screening versus control: Full MM0.043913→0.053000,
  exact duplicate-pair rate0.678545→0.629091, Full FID0.240975→0.239811.
  However Full Matching worsens3.028667→3.046784 and Hands FID worsens
  0.210871→0.224183. This single-seed result is NOT a quality-preserving win.
  Promising later arms require paired replications and physical-quality checks.
- Temperature0.8→0.5 annealing main quality completed at10:28 BST:
  Full/Body/Hands FID0.242031/0.233383/0.210923, Full Matching3.028160,
  Full R@1/2/3=0.435662/0.645221/0.768382. These are close to control,
  MM completed at10:44 BST: Full0.046374 vs control0.043913,
  duplicate-pair rate0.675091 vs0.678545; mean unique2.62 vs2.59.
  This small single-seed change is not a demonstrated quality-preserving gain.
  GPU1 has moved on to constant temperature0.8.
- Next training arm is PREPARED, NOT launched:
  `tools/run_hhc_no_shape_joint_text_dropout_pilot.sh`. It reuses all-valid
  baseline settings and changes joint/global text dropout contract (0.1 joint
  vs0.2 global-only); do not silently combine this with masked-only loss.
- Token thinning diagnostic on32 frozen validation crops is complete:
  `output/hhc_vq_temporal_thinning_val32.json`. Retains50.3%tokens but changes
  31.8%IDs; normalized reconstruction MSE grows0.31718→0.33852/0.34236.
  This is not a trained stride8 tokenizer or a MM/quality success.
- Additional completed diagnostics (September10,10:40 BST):
  `output/hhc_vq_code_neighbors_val32.json` shows replacing9.208% of IDs by
  nearest codebook neighbors gives normalized MSE0.322882 vs baseline0.317181
  and matched-position random replacements0.349758. Geometry is useful but
  replacement is not free of reconstruction cost. Input embeddings already
  use codebook geometry; a future output-target intervention would be distinct.
  The existing temporal probe now accepts `--intervention code_neighbors`;
  default temporal thinning is preserved and helper tests pass.
- New joint physical screens explicitly use `--floor-policy valid_frames`,
  protocol `interx_official_v1_physical_joint_v2`. This excludes padding from
  floor estimation; the script retains legacy default for historical replay.
  In calibration root, `{control,cfg2p5,anneal0p8to0p5}_physical_v2.json` are
  complete. Sliding m/s:0.303470/0.301686/0.303573; contactF1:
  0.557580/0.562008/0.557360. These are not mesh penetration measurements.
  Zero-beta physical quality also needs shape-aware geometry, not this
  original-subject-skeleton proxy. Regression test passes.
- GPU0 control main and MM are now COMPLETE and match GPU1 metrics exactly.
  Confidence_all main is complete: Full FID0.203193 improves vs0.240975,
  but Hands FID0.265895 worsens vs0.210871; Full Matching3.039249 is worse
  than3.028667. Corrected joint sliding worsens to0.313003m/s and contactF1
  to0.541148. Do not promote this all-parts arm on FID alone.
  At10:57 BST, confidence_all MM is COMPLETE: Full1.160336,
  Body1.228393, Hands1.814799 (control0.043913/0.047964/0.083007).
  Exact duplicate pairs109/5500=0.019818, mean unique10.46/11.
  This is a major sampling-driven reduction in collapse, but still NOT a
  quality-preserving success because Hands FID and joint physical metrics
  degraded. GPU0 has started confidence_root. Prioritize part/noise-strength
  calibration and paired replication after that result, not a wholesale
  framework change or premature promotion of the all-parts setting.
  Next GPU1 priority after its live calibration grid finishes is now
  `bash tools/run_hhc_no_shape_sampling_validation.sh cuda:1 noise_strength`
  (noise0.5/all, noise1/all with proposal temperature0, then noise0.25/all;
  PREPARED, NOT launched), ahead of the older
  step-budget and joint-text-dropout training plans. This prioritization follows
  the measured26x Full-MM gain from confidence ranking, not a completed quality
  gate. Keep GPU0's root-only and zero-beta arms running in their existing order.
  Sampling step-budget follow-up is PREPARED, NOT launched:
  `bash tools/run_hhc_no_shape_sampling_validation.sh cuda:1 budget` tests
  20/10 steps with other settings fixed, reusing GPU1 calibration control.
  `bash -n` passes; actual experiment execution remains required.
  MM scorer now accepts explicit `--generation-seed` for paired follow-up
  replications (default600123 unchanged; self-check rejects undeclared seed
  changes). Pair-selection RNG remains fixed by request, independently of
  generation seed. No extra replication has been launched yet.
- Stride8 VQ prerequisites implemented and CPU-smoked, not trained:
  VQ records its actual `temporal_stride`; VQ training validation, reconstruction
  evaluator and diagnostic probe now pad to that stride. Trainer rejects
  incompatible windows/decoder strides before starting W&B. Tiny real VQ
  encode/decode/backward tests pass at stride4 and8.
  `utils/fk.py` now clones absolute skeleton inputs before offset conversion;
  previously repeated FK construction could mutate the shared global table.
  Test proves repeated construction preserves both inputs and identical FK
  offsets. This fixes a root cause, not just one caller's defensive copy.
  New corrected baseline: `output/hhc_vq_raw12288_corrected_fk_val544.{json,h5}`,
  544 validation samples, raw feature RMSE0.089197529, MPJPE0.157910m.
  Feature RMSE matched exactly when regenerated with the H5 saved. Old
  reconstruction FK joint metrics must not be mixed with this corrected
  baseline; reconstructed features and released joint scorers are distinct.
  CPU released-joint scoring is COMPLETE:
  `output/hhc_vq_raw12288_corrected_fk_val544_joint_cpu.json`,
  Full/Body/Hands FID0.110864/0.311243/0.134448. Compare future reconstructed
  VQ candidates using the same CPU scorer, not against GPU-generation numbers.
  The cheap agent mistakenly restarted this silent CPU scoring after an
  observation timeout; root verified both PIDs alive and terminated ONLY the
  duplicate3507438. Original3507238 completed, report status complete, both
  PIDs absent afterward. No artifacts were deleted. Original session_id was
  lost because the agent did not print exec_command's return value; no exact
  exit_code is claimed for that scoring process. AGENTS.md now records the
  handle/terminal-state discipline explicitly.
  IMPORTANT: MGT/cache/sampler/exporter still contain stride4/38-token
  contracts. End-to-end stride8 MGT is NOT supported yet; do not merely point
  the existing exporter at a new VQ checkpoint. No stride8 training launched.
  Before any new VQ launch, also fix/declare legacy telemetry naming:
  `perplexity_right_hand` currently records root and `perplexity_left_hand`
  records the joint hand part. The existing W&B profile expects those legacy
  keys, so a correctly named new profile is needed without rewriting old runs.

September10,11:09 BST follow-up evidence:
- Confidence_root main COMPLETE: Full FID0.204547, Matching2.986860,
  R@1/2/3=0.470588/0.669118/0.773897 improve against control. However
  Body/Hands FID0.253948/0.261269 are worse; valid-frame joint sliding
  0.312139m/s is worse, contactF1=0.569743 improves.
  MM COMPLETE at11:20 BST: Full0.962233, Body1.171071, Hands1.526755;
  duplicate-pair rate0.076545, mean unique9.28/11. Do not collapse the quality
  gate to just the improved Full wrapper.
  GPU0 session84164 has now STARTED zero_beta. After this existing grid exits,
  next GPU0 priority is `bash tools/run_hhc_no_shape_sampling_validation.sh
  cuda:0 root_strength` (root-noise0.5 and0.25; PREPARED, NOT launched).
  Keep GPU1's next noise_strength grid unchanged; both use their same-device
  archived controls. This is an evidence-driven sampling-quality calibration,
  not a new training run or completion of task1.
- Constant temperature0.8 completed: Full MM0.066102, Body0.082991,
  Hands0.110457, duplicate-pair rate0.561455, mean unique3.47/11.
  Full FID0.242969 and Matching3.025871 are close to control; Body/Hands
  FID0.233684/0.215267. Sliding0.303759m/s, contactF1=0.557475.
  GPU1 session88355 is now running its last arm, matched_text_dropout.
- CPU re-scoring with all55 off-diagonal repeat pairs COMPLETE (sessions
  43876/26926 exited0): `{control,confidence_all}_mm_cpu_detail.json` in
  confidence root. Full all-pair mean0.058625→1.568891; per-prompt-mean
  median0.007598→1.225320. Full increases100/100 prompts, Body99/100,
  Hands100/100. Highest5% prompt share of Full distance drops44.02%→15.22%.
  These extra all-pair statistics do NOT replace official10-pair MM; CPU
  and GPU results remain separate. MM scorer now records device/runtime and
  per-prompt diagnostics alongside unchanged original metric calculation.
- Raw repeat diagnostic complete:
  `control_vs_confidence_all_raw_diversity.json` in confidence root.
  Median unique repeats2→11; minimum1→4. Median raw pair RMS0→0.057514
  (mixed raw-feature units, not a perceptual-quality measure). The existing
  diagnostic now accepts explicit `--repeats11`; default30 is unchanged.
  Root corrected the cheap-agent patch to validate completion-array shape,
  ignore padding in finite-value checks, and stream prompt arrays; tests pass.
- T=0/noisy-order follow-up leaves the model and noise formula unchanged;
  the existing Gumbel sampler clamps proposal temperature to1e-10, making it
  effectively argmax away from ties. This separates commitment-order noise
  from low-probability token proposals. A golden-fixture check verifies valid
  argmax IDs, protected END/PAD and distinct noise RNG consumption; no real
  T=0/noisy-order generation has run yet.

Earlier September 10 startup details (superseded by completion statuses above):
- Masked-only run finished cleanly at 50k (W&B remote state finished).
  Training-hook 50k MM is 0.26748 vs all-valid 0.21608, but FID is worse:
  0.68964 vs 0.40753. This is NOT a quality-preserving win.
  The 50k structural checkpoint audit passes, but reports no RNG state.
  Recovery preserves optimizer/step state, not a proven bitwise-matched
  minibatch/corruption sequence. Keep this caveat in any causal interpretation.
- Five-candidate held-out joint validation is running on GPU 0, session 85261.
  Selector now explicitly accepts the preserved 10k recovery checkpoint and
  validates its internal step/run/training arguments instead of fabricating
  a validation-FID filename. Test: `python -m unittest tests.test_selector_recovery`.
- W&B audits now attach the nearest preceding explicit training optimizer step
  to validation events, labelled inferred. The comparator uses these actual
  aligned steps; missing 10k stays null, and a selected candidate missing MM
  fails promotion until replayed. Pilot audit is complete/pass; control audit
  session 10007 writes `output/hhc_no_shape_control_step_aligned_wandb_audit.json`.
- GPU 1 was rechecked free of compute processes after the other user's job
  ended. Sampling grid started there (session 88355) using
  `tools/run_hhc_no_shape_sampling_validation.sh cuda:1`: common-50 control,
  CFG2.5, temperature0.8-to0.5 anneal, constant0.8, matched historical dropout.
  Each arm uses fixed 544 main samples plus 100 prompts x11 repeats, joint
  quality and MM scoring, no MGT shape/contact, fixed300k checkpoint. Outputs:
  `output/hhc_no_shape_sampling_validation_20260910`.
  Control has now completed: validation JointsFull FID 0.24097,
  Matching 3.02867, MM 0.04391; exact repeat-pair rate 0.67855,
  mean unique motions 2.59/11. CFG2.5 arm is running next.
- CPU requested/zero-beta jobs continue unchanged; do not mix CPU and GPU
  samples within a paired comparison. Later sampling algorithm changes remain
  in scope; the parameter/conditioning grid is not the whole sampling task.

## User update (2026-09-09, supersedes the older ordering below)

1. First investigate better HHC MModality without degrading generation quality.
   Preserve the VQ-VAE + MGT framework, but architecture, token compression,
   quantizer design, training objectives and data are open to controlled
   experiments. The existing raw12288 model is the reference, not a restriction
   on new tokenization arms. Keep common-50 as a reference control; the September
   10 override explicitly permits new sampling protocols in separate arms.
2. Only then obtain public SMPL-family shape donors and perform augmentation.
   Aim for <5 seconds/sample and >60% acceptance. A sample passes only when BOTH
   foot skating and penetration after shape replacement/optimization do not
   increase relative to the ORIGINAL-SHAPE source motion. Old gates comparing
   against random-shape pre-optimization input do not establish this criterion.
3. Then run HHC InterHuman experiments with TWO VQ parts (body/root); there are
   no articulated hands. The current unlaunched three-part adapter is obsolete.
4. External comparisons, especially CoShMDM/TIMotion Inter-X reproduction, last.

On resumption PID 3355920 was absent. The same W&B run `5ll47jok` was restored
from `latest-000010000.ckpt` with `--diagnostic-stop-iter 50000`, preserving
optimizer/scheduler state. The launcher now forwards explicit resume arguments.
Its interrupted 10k validation did not finish the MM pass; do not manufacture
that event or silently relabel subsequent validation steps.

Current additional MModality diagnostic: CPU comparison of the 10k and 300k
all-valid checkpoints on 32 validation prompts, measuring conditional,
unconditional and CFG=3.5 first-step entropy under `paper_text_cfg`.
Output: `output/hhc_no_shape_10k_300k_cfg_entropy_val32.json` (complete).
At 300k, conditional first-step effective vocabulary is body/hand/root
2.058/2.077/1.996; text-free is 73.881/162.641/119.236; CFG3.5 reduces it to
1.146/1.164/1.178 with mean top-1 probability 0.946/0.942/0.934. These are
teacher-forced/input-mask diagnostics, not measured generation improvements.
The matched historical-dropout reference is complete at
`output/hhc_no_shape_300k_matched_dropout_entropy_val32.json`: both reference
and guided effective vocabulary remain about 2.0. Next conditioning-contract
training should test synchronized word/global dropout, without changing the
currently running loss-only experiment.

User also proposed all-zero shape during generation. Added and tested
`--beta-policy zero` to `tools/export_hhc_official_interx.py`, fixing genders,
labelling outputs diagnostic-only, and rejecting cross-policy resume. No
zero-beta MModality/FID result exists yet. Paired CPU jobs are running:
`tools/run_hhc_zero_beta_validation.sh cpu` (544 main samples; session 81289)
and `tools/run_hhc_zero_beta_validation_mm.sh cpu` (100 prompts x 11 repeats;
session 30798), each with 4 CPU threads. Both requested and zero policies are
scheduled sequentially in each job. Outputs: `output/hhc_zero_beta_validation_v1`.
Training remains on GPU 0 (resumed PID 3389321, session 97718); do not duplicate
any of these jobs. Resume launch time was approximately 18:32 BST.

Token statistics are complete in
`output/hhc_no_shape_token_compression_statistics.json`; train adjacent exact
repeats are body 61.3%, hands 46.5%, root 48.9%. RLE reductions 59.5/45.1/47.5%
are storage bounds before duration costs, not a measured generative gain.
Updated experiment order/evidence: `docs/hhc_mmodality_shape_condition_diagnosis.md`.

The interrupted 10k recovery checkpoint is hardlinked outside rolling cleanup
at `output/hhc_no_shape_masked_only_recovery/pre_validation_000010000.ckpt`.
Before using the 50k finalizer, explicitly support/re-evaluate this candidate:
the selector currently expects a validation-named checkpoint for all five
steps, and the comparator assumes five positional W&B events. The missing
10k event must not cause 20k to be called 10k. This recovery is still pending.

User persistent preference is now in global and project `AGENTS.md`: delegate
simple independent tasks to low-cost subagents (currently `gpt-5.6-luna`, low
reasoning); retain research design/complex diagnosis/integration in the primary.

Lower-priority work paused on the user's ordering clarification: SSP-3D was
downloaded from the author's public repository to `dataset/ssp3d/ssp_3d.zip`
(311 rows, 62 unique gender/shape pairs); labels were extracted. Local ProsePose
has `essentials/body_model_utils/smpl_to_smplx.pkl`. No accepted transferred
donor pool or augmentation has been created. Initial full-mesh linear fits
exceeded 1 cm RMSE, so these must not be reported as validated donors.

## Bootstrap instruction for a new Codex session

Read this document completely, then continue the active goal autonomously from
`/mnt/8T-HDD/wenning/HHC`. Preserve the user's dirty worktree and all running
jobs. Do not delete anything unless disk space is actually exhausted. First
check whether PID 3355920 or the masked-only run below is still active. Do not
restart it if it is active. Training jobs are checked hourly; experiments and
their scheduled validation should otherwise run normally. Report measured
results, not repeated provenance audits.

## Active objective

1. Make HHC's Inter-X training mask, model structure, training, and sampling
   consistent with the paper/historical high-performing implementation.
2. Use the official MMM Inter-X data/assets and the unified InterMask/Inter-X
   evaluator protocol.
3. Diagnose and fix the remaining MGT multimodality collapse while preserving
   strong FID and retrieval quality.
4. Acceptance threshold: comprehensively exceed the official Interact2AR
   paper's joint-only Inter-X metrics.
5. Shape augmentation is deferred until the user supplies the donor shape
   dataset. After Inter-X is accepted, adapt/train/evaluate InterHuman.

## User-mandated contracts

- Sampling: `sample_intra_cas`, 50 outer MaskGIT steps, temperature 0.5,
  CFG 3.5. Do not change these as a shortcut to improve MModality.
- Use raw12288 VQ checkpoint:
  `checkpoints/vq_precleanup_raw12288/net_175000.pth`.
- Historical transformer: 1024 embedding width, 16 heads, 9 layers total,
  2 local layers; historical base/output layout.
- Inter-X training token length contract: historical 38.
- W&B must be online for every training experiment and record train loss,
  masked/visible accuracy, validation loss/FID/Matching/R-Precision/Diversity/
  MModality, gradient norms, weight norms, and gradient/weight ratio.
- InterHuman baseline is already running on another server; do not reproduce it.
- Other external baselines are paused. CoShMDM has been quarantined and should
  not consume more time unless explicitly requested.
- No body-controlled FID is requested.
- No action-generalization split. Shape generalization means training-data
  augmentation using random shape plus annotated contact timing and optimization,
  not a held-out dataset split.
- For GT contact optimization, optimize mismatch only; it must not damage GT FID
  or foot sliding. VolumetricSMPL changes only distance calculation; nearest and
  other logic stay unchanged. Validate accuracy and speed on a small pilot first.
- Do not delete files/checkpoints while the user is away; record candidates only.

## Completed no-shape 300k HHC baseline

Run directory:

`output/t2i_trans/2026-09-07-11-49-54_hhc_historical_paper_mask_no_mgt_shape_pilot50k_seed114514_v4`

W&B run: `ywzcskfr`.

Training used paper kinematic mask ratios but historical all-valid-token loss,
no MGT shape/contact/contact-conditioning, no paper architecture flag, AdamW
2e-4, weight decay 1e-6, milestone 150k, gamma 0.05, batch 64, seed 114514.
The 300k checkpoint was selected using three fixed test replications from five
saved candidates at the user's direction. This test-guided selection must be
disclosed.

Formal unified joint-only test, 20 replications, common-50 sampler:

| Evaluation | FID | Matching | R@1 | R@2 | R@3 | Diversity | MModality |
|---|---:|---:|---:|---:|---:|---:|---:|
| JointsFull | 0.098106 | 3.120184 | 0.448909 | 0.656781 | 0.768042 | 9.224770 | 0.057011 |
| JointsBody | 0.102617 | 3.174884 | 0.463650 | 0.668249 | 0.776563 | 9.047750 | 0.060692 |
| JointsHands | 0.121956 | 3.108118 | 0.416362 | 0.625295 | 0.743573 | 8.489530 | 0.061464 |

Result card:
`output/hhc_no_mgt_shape_full300k_result_card.json`.

Interpretation: FID and Diversity are strong, but comprehensive Interact2AR
acceptance fails because retrieval is not consistently better and MModality has
collapsed. In MM rep00, 68.3724% of generated pairs are exact duplicates; 30
samples per prompt contain only 3.57 unique motions on average (median 2).
Distinct random seeds and real Gumbel sampling were verified. Repredicting tokens
worsened FID (0.6363 to 0.6712 in the diagnostic), so it is not the next change.

The official Interact2AR scorer was independently calibrated by feeding GT with
the GT first frame: Full/Body/Hands MModality = 1.3600/1.4168/1.4002, matching
the paper-scale behavior. Therefore HHC's approximately 0.06 MM is real and not
an evaluator bug.

W&B trajectory analysis:
`output/hhc_no_shape_full300k_training_collapse_trajectory.json`.
MM fell from a first-five-validation mean of 0.19703 to a last-five mean of
0.051539. Correlation with weight norm was -0.8649, masked accuracy -0.7797,
visible accuracy -0.7397, and train loss +0.7858. These are correlations, not
proof of causality, but point to growing confidence/weight norm and mode collapse.

## Currently running single-factor diagnostic

Run directory:

`output/t2i_trans/2026-09-09-17-39-10_hhc_historical_paper_mask_no_shape_masked_only_pilot50k_seed114514_v1`

- Process at handoff: PID 3355920, started 17:39 BST on GPU 0.
- W&B: ID `5ll47jok`,
  `https://wandb.ai/edmond_gla/NewInter/runs/5ll47jok`.
- At 18:15 it saved `latest-000010000.ckpt` and began the 10k automatic
  validation (17 regular validation batches followed by the MM diagnostic).
- It stops after the 50k validation. Validation/checkpoints occur every 10k;
  atomic recovery checkpoints also exist every 5k.
- Only intended behavioral change from the completed baseline:
  `training_loss_contract: historical_all_valid -> paper_masked_only`.
- Here `paper_masked_only` is a per-sample masked-token mean followed by batch
  mean. InterMask's released code uses a global masked-token mean; that remains a
  separate confirmation arm only if this diagnostic is positive.

Launch script: `tools/run_hhc_no_shape_masked_only_pilot.sh`.

After the process stops at 50k, run:

`bash tools/finalize_hhc_no_shape_masked_only_pilot.sh`

This audits the final checkpoint/W&B history, evaluates the 10k/20k/30k/40k/50k
checkpoints on the fixed official validation protocol, and compares them to the
matched historical-all-valid baseline. It must not use formal test metrics for
the promotion decision.

If and only if
`output/hhc_no_shape_masked_only_pilot50k_comparison.json` decides
`continue_same_masked_only_run_to_300k`, run:

`bash tools/continue_hhc_no_shape_masked_only_to_300k.sh`

That script resumes optimizer/scheduler/W&B state in place and fails closed if
the screen/provenance/GPU checks fail.

Source-equivalence audit:
`output/hhc_no_shape_masked_only_source_equivalence_audit.json`.
Three source hashes changed between baseline and pilot, but not behavior here:
`model_tools.py` only added InterHuman branches; `two_person_transformer.py`
added optional explicit VQ latent widths, while the Inter-X VQ has no
`latent_dims` and therefore still resolves to 96/96/64; the sampler diff is only
formatting; the augmentation change is unreachable because `train_asset_root`
is unset. The final comparison now requires this audit to pass. Unit tests:
`python -m unittest tests.test_compare_hhc_no_shape_masked_only_pilot` (3 pass).

If the current training process died during conversation transfer, do not launch
a fresh run. Resume the latest atomic checkpoint in place using the same complete
argument contract plus `--resume-in-place --resume-training-ckpt <latest-ckpt>`.
Use `tools/continue_hhc_no_shape_masked_only_to_300k.sh` as the strict-resume
reference, but retain `--diagnostic-stop-iter 50000` until the pilot gate passes.

## Other completed or paused work

- CoShMDM adapted attempt (not an official Inter-X reproduction):
  Full FID 52.1084, Matching 8.6456, R@1/2/3 0.1192/0.2041/0.2689,
  Diversity 5.3712, MM 4.1506. Body FID 56.3343; Hands FID 34.1557.
  Upstream has no complete runnable Inter-X path. Diagnosis:
  `output/coshmdm_interx_reproduction_diagnosis.md`. Do not retrain now.
- MGT sharpness diagnostic script:
  `tools/run_hhc_no_shape_logit_sharpness.sh`. It was not run because GPU 1 was
  occupied by another user. Never take GPU 1 without confirming it is free.
- MModality/root-cause notes:
  `docs/hhc_mmodality_shape_condition_diagnosis.md`.
- Execution plan:
  `configs/current_interx_to_interhuman_execution_plan.json`.

## Ordered next work

1. Let the current 50k masked-only pilot run and consume its scheduled 10k-step
   validation results. Check hourly, not continuously.
2. Finalize at 50k and apply the fixed validation-only promotion gate.
3. If promoted, resume the exact same run to 300k; otherwise diagnose from the
   five-step trajectory and do not spend 300k compute.
4. If masked-only is promising, test exact global masked-token reduction as a
   separate one-factor confirmation before treating it as InterMask parity.
5. Preserve the common-50 sampler and investigate model-distribution entropy/
   confidence rather than merely raising sampling temperature.
6. Wait for the user's donor shape data, then perform random-shape augmentation
   with full-data LLM contact-region labels, annotated contact timing, and the
   verified optimization pipeline. Run a small correctness/speed pilot before
   materializing roughly 6000 samples.
7. Once Inter-X passes the official Interact2AR joint-only gate, proceed to
   InterHuman adaptation/training/evaluation using official data and evaluator.

## Workspace safety

The worktree is intentionally very dirty and contains extensive user and agent
changes. Do not reset, checkout, or clean it. Use `apply_patch` for edits. Do not
delete old diffusion/HHC checkpoints unless disk is actually exhausted and the
user approves the exact targets. Last known free space was approximately 167 GB.
