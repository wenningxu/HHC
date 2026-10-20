# HHC review 补充实验与公平比较计划

## 当前生效指令（覆盖下文历史阶段描述）

本文档保留了排查过程和早期实验计划以便追溯；若下文旧段落与本节冲突，以本节及
`configs/current_interx_to_interhuman_execution_plan.json` 为准。

- 当前只完成 Inter-X。InterHuman 在 Inter-X 验收完成后适配；用户已在另一台服务器
  运行的 InterHuman baseline 不在本机重复训练。
- HHC 主模型使用历史强实现（1024 hidden、16 heads、7 个 global base + 2 个 local
  cross-attention + 9 个 output blocks）和 raw-12288 VQ；相对历史强训练唯一改动是论文
  kinematic mask-ratio。当前 no-MGT-shape run 是 MModality 目标因素诊断，不是论文结构
  改动；它与保留的 shape run 还存在训练预算和 validation-RNG 合同差异，因此不能直接
  当作严格的 shape-only 因果消融。
- 正式采样固定 `sample_intra_cas`、50 个 outer iterations、Root→Body→Hands、
  temperature 0.5、CFG 3.5；不以升温或减少步数作为模型改进。
- 当前验收采用 Interact2AR Table 2 的 JointsFull/JointsBody/JointsHands 指标，并独立于
  旧 672D InterMask/Inter-X evaluator 报告；两种 FID 不混列。其他 baseline 的重新复现
  暂停，已有结果保留，待数据/表示问题后续重新核实。
- action 泛化实验取消且不主张 action generalization。shape generalization 指训练数据增强，
  不是重切 test split。
- shape augmentation 每臂最终增加 6,238 条 train-only 样本：5,868 条 positive manual
  contact timing 使用 optimization，370 条 explicit-zero timing 使用 random-shape 回退。
  donor 必须来自 HumanML3D train 对应的 AMASS shape，经官方 SMPL+H→SMPL-X 转换；
  Inter-X 的 118 个身份只作可恢复旧对照，不进入最终增强实验。
- shape augmentation 的当前首轮是三个严格配对的 50k diagnostic arms：冻结的
  no-augmentation/no-MGT-shape 基线、同一 6,238 条 donor assignment 的
  random-shape-only、以及 5,868 条 optimized-selective + 370 条 random-zero。
  `tools/finalize_amass_shape_augmentation_matrix.sh` 对两条新臂分别做相同 held-out-val
  选点和 joint common50 20-rep 评测，再报告 baseline-vs-random、baseline-vs-optimized
  与 random-vs-optimized。总结果卡同时强制纳入三臂 100×30 生成的 exact-pair
  collapse、每 prompt 唯一输出数和 raw pair RMS，并绑定 replication-0 MM H5 与诊断
  实现 hash，避免把少数重复模板导致的 evaluator-space 数值误称为真实多样性改善；
  在 diagnostic 门禁通过前不把旧段落中的 300k 写成已授权。
  长流程允许机器重启：已完成 overlay 先验证 manifest/hash 后跳过，不覆盖；训练仅在
  唯一同名 run 下从最高 `<50k` 的 atomic latest checkpoint 原目录续跑并复用同一
  W&B run id，同时存在多个 run 会 fail closed。只有 50k rolling checkpoint 和 50k
  validation checkpoint 同时存在才视为该臂训练完成。optimizer 与 selective
  projection 原本已按 sample-level 原子三件套续跑；外层现在也仅在验证 gate schema、
  全部输入 artifact SHA-256、当前 gate 实现 hash 以及 status/check 一致后跳过已完成的
  25-sample 评测，避免重启后重复做同一物理审计。
- optimization 只修复换 shape 后引入的 contact mismatch；原 GT 不做无关优化。
  VolumetricSMPL 只允许替换距离/几何查询后端，nearest 与 loss 逻辑不变，并且只有数值
  验证和相同 optimizer work contract 的速度验证同时通过才可采用；120-step 对
  200-step 的旧 `1.221x` 已判无效。
- 后续所有训练必须使用 online W&B，完整记录训练/验证指标、梯度与权重范数。
- 用户不在时不删除、不移动数据、checkpoint 或结果；候选项仅记录在
  `configs/deferred_cleanup_candidates.json`。

## Review 原文到实验的对应

| 来源 | 未解决的核心问题 | 对应实验 |
|---|---|---|
| Meta-review | 优化阶段贡献归因不足；CHI3D 泛化有限；MModality 解释不足；强 LLM 鲁棒性不足 | P0-3/5，P1-6/9 |
| Reviewer L4xb | MModality 下降；LLM-contact 指标对 baseline 不公平；CHI3D 定量不足；优化耗时；缺 TiMotion 定性 | P0-3/4/5，P1-6，P2-12/14 |
| Reviewer ZyzM | 单数据集；优化仅 `FID 0.137→0.132`；低 MModality；LLM parsing/sensitivity；脚滑；无精确 timing | P0-3，P1-6/8/9/10/11 |
| Reviewer NbvC | split 与 unseen shape/action 不清；形体/方法定性不足；contact/penetration 和 user study 数值未进主文；LLM 版本与 hallucination 校验不清 | P1-7，P2-12/13/14/15 |

下列优先级以最终 review（不是只看初始评分）为准；所有小幅提升采用 paired
sample/seed 分析，避免把 evaluator 或抽样噪声解释为模块贡献。
所有新启动的训练实验必须从第一个 optimizer update 起使用稳定 run ID 的 online
W&B；每 50 个 optimizer update 至少记录未裁剪全局 FP32 gradient L2/max-abs、
weight L2/max-abs、gradient/weight L2 比值、参数/梯度元素数、两者 non-finite
元素数、learning rate、吞吐和 GPU 峰值；同时为每个可训练顶层子模块记录同组
范数与有限性诊断，并按
任务记录适用的 loss/accuracy/perplexity/mask ratio 及 official-val 指标；启用
gradient clipping 时还必须记录裁剪后 global L2、clip 阈值、是否触发并在完整运行
审计中汇总 clip 命中率。MGT/扩散生成器记录 loss、FID、R-Precision、Matching
和 Diversity，VQ 则记录重建
loss/FID，不伪造不适用的分类或文本检索指标。出现非有限梯度或权重立即失败；远端 history 不完整的 checkpoint
不得进入结果表。完成审计要求每一条必需训练流都覆盖到 final optimizer step，
并要求所有验证流发生在最终训练事件之后；W&B 内部 `_step` 不得冒充 optimizer
step。具体门禁以 `configs/online_wandb_training_policy.json` 为准。
逐条 concern（包括重复出现在多位 reviewer 中的 concern）、当前状态和证据路径
冻结在 `configs/review_requirement_matrix.json`，并由
`tools/audit_review_requirement_coverage.py` 对原始 `HHC_review.pdf` 的哈希、下列
1--16 项实验和实际产物逐项门禁。`coverage_complete` 只表示没有漏列 review
问题，不表示实验已经完成；`current_interx_pending`、后置数据集、取消且不再主张的
action 泛化、以及尚未实施的真实用户研究必须分别披露，不能合并写成 resolved。
严格主表完成后，`tools/compare_fair_interx_methods_paired.py` 会先逐 replication
核对两方法使用的冻结 request 文件字节哈希，再对 FID、retrieval、Matching、
MModality 和相对 GT 的 Diversity 绝对误差作配对 Wilcoxon 检验与
10,000 次 paired bootstrap；所有方法×指标的 p 值统一作 Holm-Bonferroni
family-wise 校正，避免多重比较造成假阳性。
跨方法检验以不读取 GT contact timing 的 HHC-M 为 reference；oracle timing 只与
同一 HHC-M 作独立配对消融，不把测试时额外信息带来的差异写成 baseline 优势。

## 统一协议

所有 Inter-X 主结果使用官方 `splits/test.txt`、150 帧上限、batch 32、
`drop_last=True` 和 `checkpoints/hhi` 官方 evaluator。测试 split 有 1708 个 ID，
其中 1706 个有效，故每次 replication 严格评测 1696 个样本。报告 20 次结果的
均值和 95% 置信区间。MModality 每次从测试集固定抽取 100 个文本，每个生成
30 次，并使用 10 对距离。随机种子与抽样 ID 对所有方法共享。完整参数见
`configs/fair_interx_protocol.yaml`。HHC 的 code-contact 概率映射只由 train
split 统计；val/test contact labels 不参与该采样引导，避免测试标签泄漏。

当前主表不再由历史 `tools/build_review_results_report.py` 直接生成，因为该脚本保留
了旧 HHC 实验标签供历史审计。六个当前正式结果全部完成后，只运行无后台轮询的
`tools/finalize_current_interx_paired_results.sh`：它要求每个方法均具备完整
20×1696 summary，先在逐 replication 相同且 request SHA-256 一致的条件下执行
paired Wilcoxon + Holm–Bonferroni，再由
`tools/build_current_interx_main_table.py` 生成
`output/current_interx_main_table.{json,md}`。缺少本次
`hhc_historical_plus_paper_mask_ratio_*_common50` 或 TiMotion 结果时会直接失败，
不会回退或混入旧 HHC 行。
主表完成后、shape augmentation 启动前，使用无轮询入口
`tools/run_current_interx_review_experiments_before_shape.sh` 依次完成：六个当前方法的
统一 mesh-physical 指标、同 request/相机的定性图，以及在本次 historical MGT、
paper-text CFG、train-only contact map 下的四组 contact-timing ablation，最后运行
以当前 no-timing HHC、50-step/0.5/3.5 为中心的 MModality 采样敏感性网格。网格中的
3/10/20-step 行只用于 reviewer 调查，正式主结果始终固定为用户指定的 50 steps。
旧的 HHC/GT paper-optimization 步骤已从该入口删除；shape augmentation 只走经过
源码等价检查的原 MMM optimizer，GT 物理基准不做无关优化。
shape 数据生成和两条增强
训练入口均把这些输出设为硬前置条件，因而“shape 最后”不仅是文档约定，也由
机器门禁执行。
在 shape 入口之前新增 Interact2AR 双协议门禁：先用官方 README 的 GT-first-frame
条件流程校准论文 JointsFull 结果，再对五种方法已冻结的无 GT 生成运行 20 次
JointsFull/Body/Hands 补充评测；后者单独成表，不与原 Inter-X evaluator 的 FID 混列。
官方 README 五次校准现已完成并通过哈希审计：JointsFull FID
`0.2665 ± 0.0194`，与论文 20 次结果 `0.277` 的绝对差为 `0.0105`；JointsBody/
JointsHands FID 分别为 `0.3305 ± 0.0232` 和 `0.2482 ± 0.0184`。该结果只证明发布
checkpoint、官方数据和三套 evaluator 接线正确；因为使用 GT first frame，不参加
无 GT 公平主表。后续五方法 joint-evaluator supplement 仍按冻结生成运行 20 次。
raw-GT replay 必须恢复冻结 request 的原始 `CPU + OMP_NUM_THREADS=8` evaluator
runtime。仅固定 CPU 而遗漏线程数时，replication 9 的 text embedding 曾漂移
`1.1920929e-6`；恢复 8 线程后，20 次 request 的 names/captions/lengths、beta/
gender/contact/MM indices 均逐元素一致，text/GT/body-GT 三类 embedding 的最大差
全部为 `0.0`。当前 replay 工具会拒绝 CUDA 或非 8 线程运行，而不是放宽容差。
正式 shape 阶段先在单张空闲 GPU 上显式运行 25 条原 MMM pilot 并停止。门禁通过后，
5,868 条 timing-positive optimization 记录才可按冻结顺序划分为可恢复 shard；370 条
timing-zero 记录确定性回退到已经逐数组审计通过的 random-only 对照。完整 6,238 条
random-only materialization 直接复用，不重复生成。聚合覆盖/hybrid-cache 审计通过后，
再并行训练 random-only 与 optimized 两个 300k 实验臂。
五方法的新 evaluator 结果先冻结为 `pre_shape_summary.json`；两个 shape 模型完成后
再追加同一 20 次协议的评分，并将七方法最终结果冻结为 `summary.json`，防止最终增强
模型遗漏用户新增的评测标准。
两条增强训练和 20-rep 配对评测完成后，
`tools/finalize_current_interx_shape_augmentation_results.sh` 会调用
`tools/build_current_interx_review_bundle.py`，把当前六方法主表、held-out-val
checkpoint 选择、nearest-vertex optimization、mesh physical、contact timing、
MModality、定性图和 matched shape augmentation 汇总成
`output/current_interx_review_bundle.{json,md}`。最终 bundle 会递归重新计算所有嵌套
artifact 的 SHA-256，而不接受仅仅“过去曾经 pass”的陈旧审计 JSON。
它还逐方法载入 `multimodality_audit_summary.json`，强制 20 次 replication、每个
prompt 的 `30 choose 2 = 435` 个 exhaustive pair、禁用 legacy provenance，并要求
其中 official 10-pair 向量与当前 CPU 主表逐值完全相同；因此当前 bundle 同时保存
headline MModality 和 pair-subsampling 稳定性，而不是借用旧 HHC 报告。
冻结 request manifest 保持最初的 7,836 字节与 `d4007b...` SHA，不允许 verifier
随工作树变化重写。后续 train-only augmentation 对 dataset loader 的扩展通过
`c3993a...` 原始源文件和 `6b357b...` 原 verifier 的 content-addressed snapshot
解析；`provenance_recovery.json` 明确记录 request NPZ 修改数为 0。最终报告必须
同时验证原 manifest、20 个 NPZ 和三个历史源 snapshot，任一缺失即 fail closed。
冻结后协议 YAML 又加入历史采样审计并移除了 body-controlled 指标；request 时刻的
3,721-byte/`1db8e...` 原文件已通过本线程 rollout 中的逐补丁逆重放精确恢复并作为
第三个 content-addressed snapshot 保存。fresh verifier 现在同时解析这三个快照，
而不会把当前 YAML 的变化悄悄写回冻结 manifest。
此外已用官方 loader/evaluator、seed 1234 在 CPU 重放 replication 0：1696 个
name/caption/length/beta/gender/contact/MM-index 全部逐元素一致；三类 FP32 embedding
的最大绝对差均不超过 `9.5367431640625e-7`。request NPZ 字段本身未记录 evaluator
device，所以这里只对 embedding 声明严格 `1e-6` 容差一致，不虚报 bitwise；replay
audit 现直接绑定 provenance recovery，后者从原始 Codex 执行记录恢复出正式创建命令的
`--device cpu --replications 20 --seed 1234`，
并证明第二次执行完整报告 20 个 replication、exit 0。对应的两条 JSONL 记录冻结为
16,063-byte/`e2ae556f...` content-addressed snapshot，并由
`provenance_recovery.json` 和最终报告共同校验。第一次相同命令仅运行到 replication
16 后中断、exit 1，不作为正式 provenance。最终报告同时强制读取
`output/fair_interx_request_replay_audit.json`。

全 train Luna 标注另从 extractor manifest 的 timezone-aware
`created_at_utc/completed_at_utc` 报告端到端 wall-clock；该值明确包含 retry、恢复、
暂停与 orchestration 开销，不冒充单次模型 latency。旧 batch sidecar 未逐 attempt
记录 timer，因此逐样本 latency 继续标为 N/A，而不是由 token 数反推。
train/val/test/frozen/Terra cache reports 也都记录并由最终报告重新哈希 cache-audit
implementation 本身，避免审计器逻辑变化后继续接受未绑定的旧 JSON。
train/val/test 三份 manual-timing/LLM-region fusion reports 同样绑定 timing-audit
implementation，最终报告对五份 cache 与三份 timing artifact 都 fail closed。
optimization readiness 也绑定自身 implementation，并在 pending 阶段就重新哈希
frozen requests、官方 split、strict contact manifest 与 contact-cache audit；只有
正式 HHC generation/conversion artifacts 可以保持显式缺失。
最终 full-split 表同时公开 `positive timing / empty regions` 与
`zero timing / nonempty regions`；前者只启用 physical losses，后者停用语义接触，
不通过强行补标签或覆盖人工 timing 制造一致率。

## Checkpoint 与重训原则

- HHC 是本工作的目标模型：保留已经通过 reconstruction 审计、且与大清理前
  主线字节一致的 raw-12288 VQ checkpoint；保留已经验证有效的历史 MGT 结构和
  数值训练流程，只将训练 mask-ratio 改为论文 kinematic 分布后重新训练 300k；
  正式生成另固定用户指定的 common-50 合同。历史 HHC checkpoint 只做同协议诊断，不替代
  修正模型的重训结果。
- 对比方法默认直接使用作者发布 checkpoint，只增加冻结请求、精确长度、表示
  转换和统一 evaluator 适配，不因代码移植方便而重训。只有作者 checkpoint
  无法加载，或其训练数据与目标主表不同、因而不能作同数据公平比较时，才允许
  重训，并在表中明确标成 retrained port。
- 当前 Inter-X 主表中，InterMask 与 Interact2AR 使用作者权重；CoShMDM 与
  TiMotion 没有可用于该 Inter-X 协议的作者权重，因此是上述数据不一致例外。
  CoShMDM 仓库的官方下载脚本明确只发布 `# InterHuman` 的
  `coshmdm.ckpt`，本机也不存在其他作者 Inter-X checkpoint；因此当前 568k
  Inter-X run 是公平比较所必需的 retrained port，而不是无理由重复训练 baseline。
  第二数据集 InterHuman 若已有作者权重，则同样优先直接评测，不重复训练
  baseline。
- 当前执行阶段只完成 Inter-X。InterHuman baseline 已在另一台服务器运行，本机
  不重新训练也不重新评测；待 Inter-X 全部完成后再按 checkpoint、split、请求数
  和 evaluator provenance 合并外部结果。CHI3D 等其他数据集同样后置。

## P0：主结论必须补充

1. **论文实现复现。** 使用大清理前主线的 raw-12288、code-dim-256 shape-aware
   VQ-VAE checkpoint（已单独复核 reconstruction FID/MPJPE），并以大清理前已经
   验证有效的 1024-d/16-head/7-global+2-local MGT 为唯一结构基线重新训练 300k；
   相对该历史训练合同只引入论文给出的 kinematic mask-ratio 分布，不再同时改变
   hidden size、head 数、loss support、scheduler、token window 或 sampler。报告 VQ reconstruction、codebook
   perplexity，以及 HHC-M 的 FID、Diversity、R-Precision、Matching、MModality。
   当前第一版严格论文配置已经完成，但统一 scorer 下 test FID 为 `0.46451`，未
   通过相对历史低 FID 模型预注册的 `<=0.30` 质量门禁，因此不得直接进入正式
   shape-augmentation 重训。冻结 544 条 val、只改变 token 更新支持的诊断表明，
   MMM 原生“整通道重预测”反而把 FID 从 `0.63626` 恶化到 `0.67120`，所以正式
   sampler 继续采用用户指定的 50-step fixed-token 语义。大清理前 Local History、
   原 W&B 和 checkpoint 的时间链则确认：与论文表中 `0.137` 几乎相同的真实旧
   轨迹在 200k 达到 `0.1357`，但其配置是 1024-d/16-head、batch 64、legacy
   corruption/all-valid loss 和 MultiStepLR，而不是附录的 512-d/8-head、batch 96、
   论文 kinematic mask/masked-only loss 和 warm-up+cosine。这个数值/时间关联是
   强证据推断，不冒充论文表 checkpoint 的直接 provenance，也不把 legacy 模型
   错标为论文结构。当前已排除 evaluator 迁移、VQ 重建和重预测 sampler 为唯一
   原因；剩余因果集合与全部哈希冻结在
   `output/hhc_mgt_root_cause_status.json`。当前主模型及后续两个增强模型均固定
   “历史强基线 + 仅论文 mask-ratio”合同并全程 online W&B，不再把多项同时变化
   的附录配置称为当前正式模型。
   当前单变量 run 在 40k 的训练内 FID 为 `0.19412325`，同期历史 run 为
   `0.37795949`；两者旧式训练内 sampler/RNG 合同相同，且 FID 都是当前
   checkpoint 值。历史源码当时却把 Diversity、三项 R-Precision 和 Matching
   记录成 best-so-far 包络，而当前源码记录当前 checkpoint 值，因此这些字段不能
   从两份 W&B history 做逐 checkpoint 差值；最终只能以冻结 official-val 重新
   生成选模和 20-repetition official-test 结果比较。源码级证明与输入哈希记录在
   `output/hhc_historical_vs_paper_mask_ratio_v2_validation_10k_40k_semantics_audit.json`，
   防止把日志口径差异误判成 mask-ratio 的语义能力退化。
   作为训练期诊断，原 MMM `output.log` 与 V7 复现时保存在 Codex JSONL 的原始
   40k stdout 又恢复出了非包络值，并逐字段相等：历史当前值的 R1/R2/R3 为
   `0.39504717/0.56544811/0.69221698`，当前 mask-ratio run 为
   `0.36615565/0.53655660/0.65448111`；Matching 为 `3.78005457` 对
   `3.89298820`，而 FID 为约 `0.3780` 对 `0.19412325`。因此下降方向在恢复后仍
   存在，但它仍只是单次旧 sampler 的训练期诊断，不能替代最终 fixed-seed
   official-val/test。恢复脚本同时绑定 686 MB session、原日志、V7 零差复现审计和
   当前 W&B comparison 的 SHA，证据为
   `output/hhc_historical_vs_paper_mask_ratio_v2_current_validation_40k_recovered.json`。
   50k 里程碑也已完成同样的非包络恢复与 checkpoint/结构审计。历史当前值对
   paper-mask 当前值分别为：FID `0.2443` 对 `0.21726498`，R1/R2/R3
   `0.40566038/0.58962264/0.69103774` 对
   `0.38443395/0.55719340/0.66568398`，Matching `3.71646118` 对
   `3.89091754`，Diversity `9.0552` 对 `8.89093399`。因此 mask-ratio 在该点
   改善 FID `0.02703502`，但语义匹配指标仍退化；当前 run 的训练内 best FID
   仍是 40k 的 `0.19412325`。这不提前停止 300k 训练，也不替代最终 common-50
   official-val 选模。证据为
   `output/hhc_historical_vs_paper_mask_ratio_v2_current_validation_50k_recovered.json`、
   `output/hhc_historical_plus_paper_mask_ratio_v2_50k_checkpoint_audit.json` 与
   `output/hhc_historical_plus_paper_mask_ratio_v2_50k_model_structure_audit.json`。
   60k 里程碑也已完成并继续支持同一结论。历史当前值对 paper-mask 当前值分别为：
   FID `0.3720` 对 `0.26331753`，R1/R2/R3
   `0.39917453/0.56721698/0.67747642` 对
   `0.36615565/0.55011791/0.65919811`，Matching `3.80878878` 对
   `3.84111333`，Diversity `8.9347` 对 `8.90347290`。即 paper mask 在该点将
   FID 改善 `0.10868247`，但三项 R-Precision 均较低；当前 run 的训练内最佳
   仍为 40k 的 `0.19412325`。60k checkpoint 的 global step 为 `60,000`、epoch
   `447`，931 个模型浮点 tensor 与 1,032 个 optimizer 浮点 tensor 全部 finite，
   1024-d/16-head/7-global+2-local/raw-12288 结构合同未变。对应证据为
   `output/hhc_historical_vs_paper_mask_ratio_v2_current_validation_60k_recovered.json`、
   `output/hhc_historical_plus_paper_mask_ratio_v2_60k_checkpoint_audit.json` 与
   `output/hhc_historical_plus_paper_mask_ratio_v2_60k_model_structure_audit.json`。
   70k 的方向发生反转：历史当前值对 paper-mask 当前值的 FID 为 `0.2612` 对
   `0.30590948`，R1/R2/R3 为
   `0.41568396/0.60023585/0.71049528` 对
   `0.38797170/0.56014150/0.67216980`，Matching `3.65282464` 对
   `3.83892560`，Diversity `8.7967` 对 `9.02943611`。该点 paper mask 的 FID
   变差 `0.04470948`，语义指标也更差，进一步说明训练内单次随机诊断不应替代
   全 30 候选 common-50 official-val 选择。70k full checkpoint 的 model/optimizer
   浮点 tensor `931/1,032` 均 finite，历史拓扑与 launch hash 保持一致；证据为
   `output/hhc_historical_vs_paper_mask_ratio_v2_current_validation_70k_recovered.json`、
   `output/hhc_historical_plus_paper_mask_ratio_v2_70k_checkpoint_audit.json` 与
   `output/hhc_historical_plus_paper_mask_ratio_v2_70k_model_structure_audit.json`。
   80k 的差距缩小但没有反转为优势：历史当前值对 paper-mask 当前值的 FID 为
   `0.2612` 对 `0.27149162`（paper mask 差 `+0.01029162`），R1/R2/R3 为
   `0.39091981/0.57959906/0.69634434` 对
   `0.38443395/0.56073111/0.66981131`，Matching 为 `3.67867088` 对
   `3.83362651`，Diversity 为 `8.8897` 对 `8.79576015`。因此该训练期随机诊断中
   paper mask 的三项 retrieval 与 Matching 仍较差，本 run 的最低 FID 仍是 40k
   的 `0.19412325`；是否最终优于历史实现仍只能由全 30 个 checkpoint 的 frozen
   official-val/common-50 重生成决定。80k checkpoint 位于 epoch 597，931 个模型
   浮点 tensor 与 1,032 个 optimizer 浮点 tensor 全部 finite，1024-D/16-head/
   7-global+2-local/9-output/raw-12288 结构未漂移。证据为
   `output/hhc_historical_vs_paper_mask_ratio_v2_current_validation_80k_recovered.json`、
   `output/hhc_historical_plus_paper_mask_ratio_v2_80k_checkpoint_audit.json` 与
   `output/hhc_historical_plus_paper_mask_ratio_v2_80k_model_structure_audit.json`。
   除源码/参数审计外，80k 时又直接扫描了在线 W&B 本地 transaction 中全部 1,600
   个 50-step telemetry event。root/body/hand mask-ratio 的实际均值为
   `0.362140/0.664776/0.883739`，而三个独立均匀变量经排序并代入
   `cos(pi*tau/2)` 的理论期望为 `0.361786/0.664439/0.883635`；全部 event 均满足
   `root <= body <= hand`，无缺步、越界或非有限值。这证明当前进程运行时确实走了
   论文分层 ratio 分支，而不只是 manifest 参数名正确。证据为
   `output/hhc_historical_plus_paper_mask_ratio_v2_runtime_history_80k_audit.json`；最终
   300k 后 finalizer 会对完整 6,000 个 telemetry event 重跑同一门禁。
   训练继续运行期间又把门禁扩展到 85k：1,700/1,700 个预期 telemetry event
   完整存在且全部 finite、全部满足 `root <= body <= hand`；实测均值
   `0.362097/0.664589/0.883762` 仍落在上述理论期望的 0.01 容差内。85k 完整
   checkpoint 的 931 个模型浮点 tensor 和 1,032 个 optimizer 浮点 tensor 也全部
   finite，step/LR/MultiStepLR/历史模型结构均通过。证据为
   `output/hhc_historical_plus_paper_mask_ratio_v2_runtime_history_85k_audit.json`、
   `output/hhc_historical_plus_paper_mask_ratio_v2_85k_checkpoint_audit.json` 与
   `output/hhc_historical_plus_paper_mask_ratio_v2_85k_model_structure_audit.json`。
   90k 的点对点当前值已从冻结的 V7 replay stdout 恢复，并由原 MMM 日志、V7
   W&B 零误差复现审计和可选 Codex stdout 捕获交叉验证。历史当前值对 paper-mask
   当前值的 FID 为 `0.2705` 对 `0.21097779`（paper mask 改善
   `0.05952221`）；R1/R2/R3 为
   `0.39445755/0.60023585/0.71462264` 对
   `0.37676886/0.56132078/0.67099059`，Matching 为 `3.59609938` 对
   `3.80546880`，Diversity 为 `9.1700` 对 `9.22036648`。因此 90k 再次出现
   FID 改善但 retrieval/Matching 退化；当前训练内最低 FID 仍为 40k 的
   `0.19412325`，不能据此提前选择 checkpoint。90k checkpoint 的 global step
   为 `90,000`、epoch `671`，931 个模型与 1,032 个 optimizer 浮点 tensor
   全部 finite，历史结构未漂移；累计 1,800/1,800 个 mask telemetry event 全部
   finite 且满足 `root <= body <= hand`，实测 root/body/hand 均值为
   `0.362115/0.664558/0.883677`。证据为
   `output/hhc_historical_vs_paper_mask_ratio_v2_current_validation_90k_recovered.json`、
   `output/hhc_historical_plus_paper_mask_ratio_v2_90k_checkpoint_audit.json`、
   `output/hhc_historical_plus_paper_mask_ratio_v2_90k_model_structure_audit.json` 与
   `output/hhc_historical_plus_paper_mask_ratio_v2_runtime_history_90k_audit.json`。
   95k rolling checkpoint 也已通过完整状态与结构审计：global step `95,000`、
   epoch `708`、LR `2e-4`、weight decay `1e-6`，931 个模型和 1,032 个
   optimizer 浮点 tensor 均 finite；模型仍为上述 1024-D 历史强结构。W&B 在
   checkpoint 落盘后短暂只 flush 了 1,898/1,900 条事件，随后日志事务完成后严格
   得到 1,900/1,900，全部满足层级和理论均值门禁；这次瞬时竞态没有被误写为训练
   失败。最终通过证据为
   `output/hhc_historical_plus_paper_mask_ratio_v2_95k_checkpoint_audit.json`、
   `output/hhc_historical_plus_paper_mask_ratio_v2_95k_model_structure_audit.json` 与
   `output/hhc_historical_plus_paper_mask_ratio_v2_runtime_history_95k_audit.json`。
   100k validation 也已完成：paper-mask run 的当前 FID 为 `0.22626428`，历史
   V7 同 step 的当前 FID 为 `0.33693248`，即该点改善 `0.11066820`；当前 run
   自身的 R1/R2/R3、Matching、Diversity 分别为
   `0.40801886/0.57016510/0.68691039`、`3.76482272`、`9.04622364`。由于历史
   W&B 对 retrieval/Matching/Diversity 仍保存 best-so-far 包络，本段不把这些
   字段与历史 run 做伪造的逐点比较。训练内最低 FID 仍为 40k 的
   `0.19412325`，最终选点仍须等待全部 30 个 checkpoint 的 frozen official-val
   common-50 重生成。100k rolling checkpoint 的 global step/epoch 为
   `100,000/746`，931 个模型与 1,032 个 optimizer 浮点 tensor 全部 finite，
   LR `2e-4`、weight decay `1e-6` 且历史 1024-D 结构未漂移。checkpoint 先于
   W&B 最后一条异步事务落盘时，首轮 runtime audit 如实得到 1,999/2,000；日志
   flush 后重跑严格得到 2,000/2,000，全部满足 `root <= body <= hand`，实测
   root/body/hand 均值为 `0.362235/0.664854/0.883932`。最终通过证据为
   `output/hhc_historical_plus_paper_mask_ratio_v2_100k_checkpoint_audit.json`、
   `output/hhc_historical_plus_paper_mask_ratio_v2_100k_model_structure_audit.json`、
   `output/hhc_historical_plus_paper_mask_ratio_v2_runtime_history_100k_audit.json` 与
   `output/hhc_historical_vs_paper_mask_ratio_v2_validation_10k_100k.json`。
   清理旧 checkpoint 后的在线存活复核显示，当前 run 已继续到 optimizer step
   `102,300/300,000`（epoch 763），并非被清理或重启打断。该步的 train loss、
   all/masked/visible accuracy、gradient L2、weight L2 分别为
   `0.256406`、`91.5163%/90.0147%/94.2652%`、`0.832374`、`1111.130859`；
   2,046 条已落盘训练 telemetry 中所有被要求的 loss/accuracy/mask ratio/LR、
   gradient/weight norm、吞吐和显存流均存在且无 NaN/Inf。这个 live scan 只证明
   在线训练健康，不能替代 300k 后的远端 W&B finished-state 审计；证据为
   `output/hhc_historical_plus_paper_mask_ratio_wandb_progress_live.json`。
   105k rolling checkpoint 继续通过完整状态和结构审计：global step/epoch 为
   `105,000/783`，LR `2e-4`、weight decay `1e-6`，931 个模型与 1,032 个
   optimizer 浮点 tensor 全部 finite；结构仍为 1024-D、16-head、
   7-global+2-local/9-output、raw-12288 和历史 end-token 布局。checkpoint
   SHA-256 为 `23363001...d93fad0`。首次同步扫描只看到 2,098/2,100 条 mask
   telemetry；W&B transaction 后台 flush 后严格重扫得到 2,100/2,100，全部满足
   `root <= body <= hand`，root/body/hand 均值为
   `0.362321/0.665166/0.884004`，并落在论文 order-statistic 理论期望容差内。
   最终通过证据为
   `output/hhc_historical_plus_paper_mask_ratio_v2_105k_checkpoint_audit.json`、
   `output/hhc_historical_plus_paper_mask_ratio_v2_105k_model_structure_audit.json` 与
   `output/hhc_historical_plus_paper_mask_ratio_v2_runtime_history_105k_audit.json`。
2. **统一 baseline。** 在同一 Inter-X train/val/test split 上比较 InterMask、
   Interact2AR、TiMotion、CoShMDM 与 HHC-M；不混用 InterHuman checkpoint 或
   baseline 自带 evaluator。CoShMDM 按作者 262-D body 表示在 Inter-X 重训，
   DDIM50、CFG 3.5；转换到 SMPL-X 时将未建模的手部设为 neutral，并在官方
   full-motion 指标中如实反映其表示能力限制。
   公开资产可运行性已逐项审计于 `docs/baseline_availability_audit.md`：Interact2AR
   官方 split 与冻结 split 哈希完全一致，官方 full-body 权重的无 GT 首帧推理
   已通过单样本 smoke，完整 20 次统一评测排队。TiMotion 公开 revision 只提供
   可运行的 Transformer 路径（Mamba/RWKV 未启用）；Inter-X loader、train-only
   normalization、单卡 global-batch-256 等价训练和精确长度 exporter 现已完成，
   public Transformer+LPA 的 1,500-epoch online-W&B 重训已启动（run
   `kdz5s3ra`；无 checkpoint 的中断 run `jqh616am` 已标为结果不可用）。只有该最终 checkpoint
   才进入严格主表，不能用作者的 InterHuman checkpoint 或论文 RWKV 数值冒充。
   清理后的 live transaction 再确认该 run 已到 optimizer step
   `60,124/106,500`（epoch 846）：train loss `0.0635324`、gradient L2
   `0.459306`、weight L2 `228.353333`，所有已记录训练流无 NaN/Inf。callback
   在 56,800 步写出的 `FID=60.3543` 仍由已确认错误的 interleaved-6D decoder
   产生，只用于证明 callback 正常执行，禁止进入论文表；训练结束后必须对全部
   15 个候选使用 standard row-concatenated 6D exporter 重放再选点。live 证据为
   `output/timotion_interx_formal_wandb_progress_live.json`。
   当前六方法主表还新增 ordinary-only 同设备门禁：所有生成 HDF5 均保留不变，但在
   paired comparison 之前统一使用 CPU 上的官方 full-672D Inter-X evaluator 重算
   20 次 FID、Diversity、R-Precision、Matching 与 MModality。重评脚本现显式接受
   `hhc_historical_plus_paper_mask_ratio_{no_contact,contact}_common50` 两个当前标签；
   `tools/finalize_current_interx_paired_results.sh` 生成并绑定
   `output/fair_eval/current_main_ordinary_only_metric_audit.json`。主表构建器逐一复核
   六个 summary、120 个 replication metric JSON、scorer/evaluator 权重与源码哈希，
   若混入 GPU0/GPU1 evaluator 结果或审计过期则 fail closed。
   HHC 与 TiMotion 两条独立 post 链现均在末尾调用同一个无轮询 coordinator
   `tools/finalize_current_interx_if_ready.sh`：先完成的一条在依赖不齐时立即正常返回，
   后完成的一条在六个 20-rep summary 齐全后触发上述统一 CPU 重评与配对主表。
   coordinator 使用 non-blocking `flock` 防止两条链同时完成时重复执行；主表完成
   后，由最后完成的方法把自己刚释放的 GPU 显式传给
   `tools/run_current_interx_review_experiments_before_shape.sh`，顺序执行 optimization、
   mesh physical、timing ablation、sampling Pareto 与定性图。这里不会启动 shape
   augmentation；它仍由最后阶段门禁单独控制。整个链路不创建新 background
   terminal，也不会等待训练进程。直接断言测试为
   `tests/test_current_interx_paired_finalize_contract.py`。
3. **优化模块的边际贡献（当前唯一有效方案）。** 不再优化 test GT 或生成的 HHC
   动作，也不报告所谓 optimized-GT/optimized-HHC FID。optimization 仅用于用户指定的
   train-only shape augmentation：固定同一 6,238 条 random-shape plan、donor、crop、
   timing 和 LLM region，以 random-shape-only 为严格对照，量化原 MMM optimization
   是否修复随机 shape 引入的人工 contact-timing mismatch。核心必须保持原 MMM
   `losses.py` 字节一致、`optimize_full_body()` AST 一致，以及 Adam/200 iterations/
   lr 0.01/freeze_epoch 100/原逐帧变量/动态 foot mask/nearest-vertex numerics；禁止
   loss override、VolumetricSMPL optimizer 或自定义 reduction。

   GPU 空闲且所有 pre-shape 比较完成后，先运行
   `tools/run_original_mmm_shape_pilot25.sh GPU_INDEX`。25 条门禁同时要求：原角色的
   timing-weighted mismatch 严格改善、脚滑 aggregate delta 的绝对值不超过 `1e-6`、
   至少有候选被接受，以及按实测 optimizer wall time 外推 5,868 条不超过单 GPU
   24 小时；脚本完成后必须停止。门禁通过才允许冻结全量候选接受/回退协议。
   完整 optimized arm 只优化 5,868 条 timing-positive 记录，370 条 timing-zero 记录
   回退到已逐数组审计通过的 random-only 数据。随后 baseline、random-only、optimized
   三臂用同一 VQ、历史 MMM original cache、raw mean/std、MGT 架构、300k updates、
   validation 选点和 50-step test 协议重训/评测；新增训练全部 online W&B，记录训练/
   validation 指标、梯度与权重范数。最终在 no-timing/oracle 两种条件下各跑 20 次，
   用 paired bootstrap/Wilcoxon/Holm 区分“随机扩大 shape 分布”和“contact optimization”
   的贡献。该 matched training comparison 是 optimization contribution 的正式证据。

### 已撤销的 optimization 历史调查记录（禁止执行或引用为结果）

以下文字仅保留错误定位轨迹；其中 HHC-M/GT test-time optimization、100/0.02/
freeze-50、VolumetricSMPL optimizer、loss ablation 和旧 launcher 均已撤销。

**旧记录：** 主表至少包含 HHC-M、HHC-M + contact guidance、
   HHC-M + optimization，以及 GT + optimization。除语义指标外同时报告接触
   Chamfer、contact success、penetration depth/volume、foot sliding 和运行时间。
   对 reviewer 指出的 `FID 0.137→0.132` 小增益，不只比较总开关：再逐项移除
   contact attraction、collision/penetration、temporal smoothness/pose prior，扫描
   optimization steps/constraint weight，并在同一 sample/seed 上报告 paired delta、
   bootstrap 95% CI 与显著性，判断收益究竟来自哪一项以及是否超过随机波动。
   论文复核发现旧优化入口为 `LR=0.01/freeze@100`，100 次示例从未进入后半段
   fine stage，穿透阈值也误写为 `0.2/0.5 m`；当前默认已修正为论文的 Adam、
   `LR=0.02`、100 次、`50/50` coarse/fine 和 `0.05/10 m`。所有旧优化输出均不
   进入补表。后续源码复核更正了此前“论文指定 VolumetricSMPL”的误读：Eq. 4/
   Eq. 6 只要求 signed distance，未指定神经 SDF 实现；原始 MMM 开发仓库实际使用
   CUDA Chamfer nearest vertex 加目标表面法向符号。VolumetricSMPL 仅保留为统一
   的只读 mesh 物理指标和慢速诊断，不再进入逐迭代 optimization。
   `tools/run_paper_contact_optimization.py` 已实现 strict-cache/caption/hash
   门禁、逐迭代 loss trace，以及 CUDA 同步后的逐样本 optimizer wall-clock；
   completion manifest 汇总总计、均值、中位数和 P95，恢复运行时复用已有样本
   的实测时间而不是重新估算。`tools/score_volumetric_smpl_npz.py` 固定报告双向
   penetration depth sum、Eq. 6 平方和、穿透顶点比例/深度/帧比例和脚滑，且明确
   单位；`tools/compare_paired_physical_metrics.py` 对同名样本输出 paired delta、
   bootstrap 95% CI 和 Wilcoxon p 值。三者的 CPU execution smoke 已通过。
   25 对 grounded-chain/fixed-lower-chain pilot 已完整结束。旧 scorer 错把
   smplx 输出中附加的面部/手部 landmark 也用于 `floor_y`，导致上身接触优化在
   脚完全未动时仍改变 grounded mask，并伪造 `+2.733e-6 m/frame` 脚滑 delta；
   HBP 与前 55 个 SMPL-X 骨架关节的逐样本 delta 均严格为 0。修正后的 scorer
   将 floor scope 固定并写入 scoring contract，同一个未修改 gate 现通过：脚滑
   25/25 不变，Eq. 6 delta 为 `-0.37601`（bootstrap 95% CI 全负），穿透顶点比例
   delta 为 `-0.01243`。旧 fail 与新 pass 均保留，禁止覆盖或把 metric bug 隐去。
   全量 6,238 个资产的旧 pilot 串行投影为 23.90 GPU-days。完整 100-step 对照发现
   `frame-batch=4` 会改变优化轨迹，不能作为透明加速替换；正式资产固定沿用 pilot
   的 `frame-batch=8/query-chunk=10475`。进一步发现旧 runner 在 `offset` 切片后用
   局部 index 派生 seed，使同一样本随 shard 边界改变；现已改为切片前完整确定性
   selection position，sidecar 同时绑定 position 与实际 seed。两个真实样本的双进程
   shard 与旧串行 pilot 的 NPZ SHA-256、100 步 total loss 全部严格一致；但该短测
   没覆盖全量显存峰值。正式 6,238-row 运行进入真实 150 帧样本后，在保留 3 个完整
   原子样本时触发 `torch.OutOfMemoryError`，故短测的双进程拓扑已被全量证据撤销。
   正式设置改为每张 GPU 单进程，`frame-batch=8/query-chunk=10475/seed` 均不改变。
   极端样本仍将 Eq.6 从 `1.02533` 降到 `0.40748`、穿透比例从 `4.197%` 降到
   `2.038%`，脚滑严格不变；证据为
   `output/interx_shape_aug_full_single_process_topology_audit.json`。
   正式 strict cache 可用后，`tools/run_paper_optimization_formal.py` 会在相同
   1,696 个冻结请求上同时运行 HHC-M + optimization 与 GT + optimization；
   其中 HHC-M 必须来自 common50 的 no-timing/no-contact-guidance 主结果，不能从
   oracle-contact 行启动。optimization 的时间掩码由初始生成动作的跨人几何邻近
   自动推导，不读取 test contact timing；oracle-contact 始终作为单独消融报告。
   `tools/export_optimizer_npz_to_fair_h5.py` 把优化后的 SMPL-X 严格转回官方
   672-D 表示，再由 `tools/score_fair_interx_main.py` 使用同一 evaluator 报告
   FID/R-Precision/Matching/Diversity。单 repeat 优化没有 30-repeat
   MModality，报告固定写 N/A，禁止复制优化前数值。GT 恒等回环的 rotation-6D
   最大误差为 `1.79e-7`、translation 误差为 0；完整 GT main-only evaluator
   smoke 的 FID 为 `3.79e-10`（数值零）。
   严格 HDF5→optimizer-NPZ 桥接也已补齐：
   `tools/export_fair_h5_to_optimizer_npz.py` 从冻结 request 恢复逐人的 beta、gender、
   caption、精确长度和 repeat，并把 672-D rotation-6D 转为 SMPL-X axis-angle；
   转换现在支持严格逐文件恢复：已有 NPZ 必须逐 key、shape、dtype 和数值完全等于
   当前 request/HDF5 的确定性转换，缺失文件原子补写，任何陈旧文件均 fail-closed。
   GT identity smoke 的 translation/beta 误差为 0，双人的 rotation-matrix 最大
   回环误差分别为 `2.35e-7/1.81e-7`。最终 HHC replication 0 完成后自动转换，
   后续优化 runner 会再次校验每个输入/contact/output 和 run-contract 哈希。
4. **独立于 LLM 的接触度量。** 已冻结 replication 0 的精确 GT crop，并实现
   `tools/score_fair_interx_physical.py`：用 Inter-X 逐帧二值 contact annotation
   和 subject-specific skeleton 直接报告最小关节距离、contact P/R/F1/temporal
   IoU、false-contact rate 与脚滑，且固定报告 5/10/15 cm 阈值扫描，避免选择性
   调阈值；主表固定使用 10 cm contact 与 5 cm grounded-foot 高度阈值，最终报告
   会拒绝任何未声明同一 `interx_official_v1_physical_joint_v1` 合同的方法。
   六方法还必须在同一 1,696 条 replication-0 HDF5、同一冻结 request beta/gender
   上，经 `tools/run_final_interx_mesh_physical.py` 转为 SMPL-X，并用相同的 released
   male/female VolumetricSMPL 权重报告双向 penetrating-vertex fraction、collision
   frame rate、penetration depth sum 与论文 Eq. 6。转换 ledger、逐样本评分 ledger、
   HDF5/request/权重/实现哈希必须全部一致才允许渲染主表。mesh 指标是完整 rep0 的
   确定性描述统计，不伪装为 20-replication CI 或显著性结果；LLM 只产生优化约束，
   不参与任何指标定义。CHI3D mesh-level 指标留到 Inter-X 完成后的独立数据集阶段。
   aggregate 表完成后，`tools/compare_current_interx_mesh_methods_paired.py` 以无 GT
   timing 的 HHC 为 reference，对 oracle-HHC 和四个外部方法的 7 项指标执行 1,696 条
   逐样本 delta、10,000 次 paired bootstrap 与 Wilcoxon，并对全部 35 个检验统一作
   Holm–Bonferroni 校正。该统计仍明确标为 replication-0 physical sensitivity，不能
   冒充 20-rep semantic confidence interval；oracle-HHC 也单独披露为测试信息消融。
5. **CHI3D 泛化。** 本地 MMM 资产仅含 CHI3D 官方 train 的三个主体对
   s02/s03/s04（处理后为 G2/G3/G4，共 373 条），不含可用的官方 test split/GT；
   因而不得把在这 373 条上训练后再评测的结果称为 test 或泛化。当前可复现的
   定量方案预先固定为三折 leave-one-subject-pair-out：每折用两个 G 组训练、
   剩余一个 G 组测试，汇总 macro mean 与逐折结果，并明确标成 custom CV。
   全 373 条只允许用于 GT optimization/reconstruction 的 in-sample diagnostic。
   官方 test 表须等待有权限的 test package，并经官方 challenge server 评测；
   再报告 HHC-M 优化前后与 ContactGen/可运行 baseline 的 contact、penetration、
   MPJPE/mesh error、多样性和失败案例。官方数据页说明 train 有三个主体对、
   test 有两个主体对；challenge 页列出的官方 test 指标包括 MPJPE、PA、pelvis
   translation 和 mesh error（[CHI3D](https://ci3d.imar.ro/chi3d)，
   [challenge](https://ci3d.imar.ro/challenges)，
   [download](https://ci3d.imar.ro/download)）。本地资产及哈希由
   `tools/audit_chi3d_assets.py` 固化，避免后续误标 split。
   目前 MMM 的 CHI3D VQ/MGT 都用 G2/G3/G4 的离散可学习 embedding；直接留出
   一组会使该组 embedding 从未训练，不能作为有效 shape generalization。custom
   CV 启动前必须改成连续 beta/skeleton conditioning，或预先声明为 shape-agnostic。
   另一条无需 CHI3D 重训的可执行路径，是用仅在 Inter-X 训练的最终 HHC 对这
   373 条作 external-data diagnostic；生成只报非配对 contact/physical/diversity，
   MPJPE 仅用于 reconstruction/GT optimization，不对随机生成做逐帧配对误差。

## P1：回应主要质疑

6. **低 MModality 分析。** 使用相同 100 个文本和 30 个噪声种子，报告数值与
   per-action 分布；为 hug、high-five、handshake、kick 等至少展示 4 个同文本
   多样样本。增加 temperature、CFG、sampling steps 的小型网格，展示质量–
   多样性 Pareto 曲线，而不是只挑单个最好结果。除严格保留官方共享 10 对的
   主指标外，补充报告每文本 30 次生成全部 435 对的均值和分位数，以量化该
   Monte Carlo 估计本身的方差。InterMask 完整 20 次的全部 435 对均值为
   `1.88245 ± 0.04404`，官方 10 对估计为 `1.80842 ± 0.06971`；配对偏差均值
   为 `-0.07403 ± 0.05651`，RMSE 为 `0.14868`，单次最大绝对偏差为
   replication 6 的 `0.40115`。
   这证明异常既包含随机 100 文本子集差异，也会被稀疏配对进一步放大。
   最终 HHC checkpoint 的 replication-0 one-factor grid 已预注册并排到 GPU 1
   严格主评测之后：以 `temp=0.5/CFG=3.5/steps=50` 为中心，分别测试 temperature
   `0.3/0.8`、CFG `2.5/4.5`、steps `10/20`，并额外加入大清理前 Git 快照中
   `sample_intra_cas` 函数的默认参数 `temp=1.0/steps=3`，共 8 行。该 3-step
   行只把快照函数的默认步数/温度迁移到论文语义 sampler，不声称复现开发代码中
   与论文“高置信 token 固定”相冲突的覆盖更新或 guidance 衰减行为。3、10 和 20
   分别覆盖清理前快照的 cascade 函数参数（每轮 root/body/hand，无 CFG 时共 9
   次前向）、论文图示参数，以及 InterMask 发布默认预算；每行都生成相同 1,696 个
   main 样本和固定 100×30 MM 样本，同时报告官方 10-pair、全 435-pair、FID、
   retrieval、contact/脚滑，并由 `tools/summarize_hhc_sampling_pareto.py` 标记
   FID–MModality 非支配解；历史 W&B traceback 已确认运行时 sampler 是
   `sample_intra_cas`。网格中心的 50-step/0.5/3.5 no-timing 行严格复用当前正式主表
   replication-0 的 main/MM HDF5，并重新评分；复用前会核对 checkpoint、request、
   seed、模型/推理合同和 completed repeats，避免重复生成约三小时的相同样本。
   Local History 与 W&B 启动时间交叉审计进一步恢复出真实默认
   为 10 outer / 30 channel updates：2026-02-17 为 temp 1.0/CFG off，2026-03-05
   为 temp 0.5/CFG 3.5。该历史备份不新增重跑项，也不改变上述以论文语义 sampler
   执行的单 replication sensitivity，更不替代主参数的 20 次结果。
   另对产生旧 W&B 30 点轨迹的 2026-02-17 14:04 完整训练 run 本身做了独立时间夹取：
   其 validation 精确为 `sample_intra_cas`、10 outer/30 channel、temp 0、CFG off、
   contact off；当前 validation 为 50/150、temp 0.5、CFG 3.5、validation-oracle
   contact。旧 final FID `0.180908` 与当前 selected/final
   `0.7801115513/2.5655829906` 因而只作非公平诊断，不再以“历史参数未知”解释。
   当前 replication-0 网格已完成。以相同冻结 request/seed 计算，FID–全 435-pair
   MModality 的非支配行包括 `temp=0.5/CFG=3.5/steps=20`
   （FID `0.138734`，MModality `0.072643`）、`temp=0.8/CFG=3.5/steps=50`
   （`0.145859/0.085723`）和历史函数默认参数诊断
   `temp=1.0/CFG=3.5/steps=3`（`0.179083/0.216743`）。它说明减少步数未必单调损害
   FID，而更高 temperature 明显提高同文本多样性；仍只作单 replication
   sensitivity，不替代用户指定的 50-step 正式 20-rep 结果。八行均通过汇总器新增的
   evaluator-runtime 一致性门禁，统一在 CUDA evaluator 上评分。
7. **基于 optimization 的 shape 数据增强。** 不改变官方 Inter-X
   train/val/test split，也不构造 held-out shape/action split；action 泛化实验取消。
   **当前最终执行顺序与原实现约束（2026-09-06；覆盖本节后文所有旧的
   VolumetricSMPL、自定义 loss、100-step/0.02/freeze-50 和 nearest-last-stage
   描述）：** shape augmentation 是 Inter-X 的最后阶段。核心优化器必须逐字节使用
   原 MMM `optimization/losses.py`，且 `optimize_full_body()` 函数 AST 必须与原开发
   仓库相同；固定 Adam、200 iterations、lr=0.01、freeze_epoch=100、原始逐帧变量、
   动态 foot mask、原始 reduction，以及 CUDA Chamfer nearest-vertex + target-normal
   sign。`tests/test_original_mmm_optimizer_restoration.py` 同时锁定源文件、函数 AST、
   wrapper 调用参数和禁止的自定义实现不得进入生产 import。

   在两块 GPU 上的正式 evaluator/mesh 任务完成前不抢占资源。GPU 空闲后，由显式
   前台入口 `tools/run_original_mmm_shape_pilot25.sh` 对冻结 plan 中 25 个 timing-positive
   样本运行原 MMM 优化；它只生成诊断和门禁，不会自动启动全量或训练。门禁要求以
   原角色 contact timing 计算的 mismatch 改善、脚滑不恶化，并报告 penetration 与
   运行时间；任何 upper-body constant projection/fallback 均须标为 wrapper 诊断，
   不得冒充原 MMM optimizer。只有 pilot 门禁通过后才能冻结全量候选接受/回退策略。

   完整 random-shape-only 对照 `output/interx_shape_aug_random_seed123_v2` 已对 6,238×2
   个数组逐项验证：每条 motion crop 与冻结 train H5 一致，beta/gender 与 donor plan
   一致，sidecar、plan 和输入哈希全部通过；审计证据为
   `output/interx_shape_aug_random_seed123_v2_completion_audit.json`。该臂没有调用任何
   optimizer，因此可以安全复用，不因旧 manifest 中记录过历史 optimizer source hash
   而重复生成。全量 optimized 臂只处理 5,868 个 timing-positive 样本，其余 370 个
   timing-zero 样本确定性回退到同一 random-only 数据。旧
   `tools/run_interx_shape_augmentation_nearest_last_stage.sh` 和旧 shape-training launcher
   已强制退出，禁止作为正式入口。最终 baseline、random-only、optimized 三个训练臂
   使用相同 raw mean/std、VQ、MGT 配置、300k updates、checkpoint 选择和冻结测试请求，
   并分别执行相同的 20-rep no-timing/oracle 评测与配对统计。
   磁盘门禁已于 2026-09-02 修正：旧门禁从一个仅约 18 GB 的已裁剪训练目录外推，
   会漏算 `--save-all-validation-checkpoints` 的 30 个候选。新门禁使用当前实测
   `3,943,080,038` bytes/checkpoint，按每个 arm 30 个候选加 latest/last 两个恢复点
   计算，并在 full random/optimized 资产生成前执行。当前 451.06 GiB 可用空间下，
   旧的全量 checkpoint 策略需要 523.80 GiB（含 100 GiB 完成后余量），所以正确
   状态为 `insufficient_capacity`，证据为
   `output/interx_shape_augmentation_disk_capacity_live_projection.json`。门禁绝不删除
   或压缩现有结果；待当前 HHC 训练及其源码哈希审计冻结后，再实现仅 validation
   candidate 使用 weights-only、latest/last 保留完整 optimizer state 的策略并重新
   实测门禁，不能在活动训练中修改 `train/train_t2i_trans.py`。
   门禁读取当前 selector 的正式 `artifacts.training_manifest.path` schema；last-stage
   与每个可直接调用的 random/optimized arm 均在 cache/training 前独立复核容量，
   防止绕过总入口。
   旧 `queue_interx_shape_augmentation_gpu1.sh` 及其上游
   `queue_hhc_timing_after_post_gpu1.sh` 保留作历史 provenance，但入口已 fail-closed，
   防止其 `while/sleep` waiter 或旧 VolumetricSMPL 流程被误启动；正式入口只有上述
   nearest-vertex last-stage 脚本。
   机器审计为 `output/interx_shape_augmentation_last_stage_contract.json`。
   shape 泛化通过训练数据增强实现：只从官方 train split 读取原动作、caption、
   beta/gender 和人工逐帧 contact timing；用 GPT-5.6 Luna 对 train 9,110 条
   caption 全量标注双方接触部位，再为双人确定性采样新的随机 body shape，以原
   动作作为初始化，将 LLM 部位与人工逐帧 timing 一起送入论文 optimization，
   修正换形后产生的接触错位、穿透和脚滑，再将优化后的
   `motion + new beta/gender` 作为新增训练样本。val/test annotation 绝不用于
   生成增强训练数据。数据核查推翻了“整类缺 timing 即无接触”的旧假设：缺文件
   的 A006/A013/A017/A020/A024/A029/A037/A038 等类别中存在明确的抓、推、拍肩
   描述，不能补成零标签。v2 冻结计划因此只选择既满足 30--999 帧训练条件又确有
   人工 timing 文件的 6,238 条；2,857 条长度合格但 timing 未覆盖的样本和 15 条
   长度不合格样本均显式排除。人工 timing 为全零的 370 条仍作为已标注负样本保留。
   Luna 仍标注全部 9,110 条，便于单独报告覆盖和将来补 timing，但未标 timing 的
   结果不得进入本轮 optimization augmentation。
   清理 checkpoint 后的独立全量重哈希审计再次通过：计划仍精确包含 6,238 条
   timing-covered 样本，其中 5,868 条正 timing、370 条显式零 timing；排除
   2,857 条缺人工 timing 与 15 条训练长度不合格样本。两个人共 12,476 个 donor
   assignment 只来自冻结的 118 个 Inter-X train shape，证据为
   `output/interx_shape_augmentation_plan_v2_audit_live.json`。
   timing 与 LLM region 的含义也不强制伪一致：仅在 timing 正帧且 region 非空时
   启用语义 attraction；timing 正但文本无双人部位时只运行 penetration、pose/
   temporal、ground 与 foot-skating 物理项；timing 为零时 region 不激活。所有
   分歧由独立审计器逐样本记录，禁止为了通过门禁而虚构接触部位。全量 preflight
   的精确分解为：5,666 条启用 timing-gated semantic contact，202 条 timing 为正
   但 region 为空、仅运行物理项，137 条虽有 region 但人工 timing 为全零所以不
   激活语义 attraction。
   全量 6,238 条输入 preflight 已通过。只有
   `G044T005A028R021` 的 Luna 原始 caption 在 processed-text 的 `#` 分隔符前
   多一个行尾空格；正文完全一致，其余 6,237 条字节精确一致。门禁只允许这种
   首尾空白归一化，并逐样本同时保存 plan/raw caption hash 与匹配模式；任何正文
   或内部空白差异仍 fail-closed。
   **2026-09-06 superseding correction.** 本段后面保留的 `100/0.02/50`、
   VolumetricSMPL optimizer、frozen-foot/grounded-chain 及其旧 pilot 数字仅是已否决
   实验的历史记录，不再定义生产协议，也不得被 launcher、gate 或论文表格引用。
   当前 production `optimization/losses.py` 与原 MMM 逐字节一致（SHA-256
   `29f67e...51f7c`），`optimize_full_body()` 函数 AST 也一致；实际设置为
   `200/0.01/freeze@100`、原始 5 cm/法线 anchor 和 nearest-vertex signed distance。
   五样本重新诊断见
   `output/interx_shape_aug_original_mmm_pilot5_assessment.json`、
   `output/interx_shape_aug_original_mmm_pilot5_ground_diagnostic.json` 与
   `output/interx_shape_aug_original_mmm_pilot5_gate.json`：未经约束的原 MMM 最终输出
   会增加 foot skating；人工 timing 内、按原 MMM 初始化角色计算的 mismatch 则
   明确下降。当前唯一候选是把原 MMM 输出显式投影回输入 root/lower-body 运动，
   只保留常量 translation delta 与 upper-body 修正，并逐样本要求 timing mismatch
   严格下降且 skating 不变，否则回退 random-shape 输入。该候选在五样本中接受
   4 条，timing mismatch 均值 `0.186506 -> 0.092928`，skating 均值保持
   `3.233327`，但仍是 `diagnostic_not_production`；更大 paired pilot 通过前不得全量。

   以下文字为历史审计日志，不是当前执行规范。

   增强数据必须逐条保存 source sample、原/新 beta、gender、shape sampling seed、
   caption、contact timing hash、optimization contract、before/after motion 和物理
   指标；optimization 使用论文的 100 iterations、LR 0.02、50/50 coarse/fine，并
   显式记录 `nearest_vertex` backend、原 MMM `optimization/losses.py` 的 SHA-256、
   sample seed、角色匹配和 coarse/fine anchor。optimized completion manifest 必须
   逐样本绑定该合同；random-only arm 明确记录不执行任何 optimization。
   进一步的真实 Inter-X 几何审计发现，optimization 使用的 `smplx.create` 若保留
   默认 `flat_hand_mean=False`，会向 MANO 手部姿态加入非零均值，而数据/接触分支
   的 `human_body_prior.BodyModel` 使用 flat-hand 零姿态；female/male 顶点最大
   偏差分别达到 `5.33/6.42 cm`。production backend 已显式冻结
   `flat_hand_mean=True`，male/female 的顶点、前 55 joints、normals 以及对 root、
   body、双手、translation、betas 的梯度与 BodyModel 全部误差为 0；门禁证据为
   `output/volumetric_smpl_interx_flat_hand_geometry_audit.json`。旧 default-hand
   optimized pilot 虽完成 `6/25`，但表示合同错误，保留为诊断且不得进入实验结果。
   同一 `selection_seed=20260828` 的 25 条匹配 pilot 中，random-only 已完成
   `25/25`，逐条记录 `iterations=0`、未向 optimizer 提供 timing/region，且 completion
   manifest 已绑定 v2 plan、Luna cache、train motion 和确定性 donor assignment；
   新的 flat-hand optimized pilot 正在 GPU1 上从空目录对完全相同的 25 条样本运行，
   完成并通过物理门禁前不得开始全量 6,238 条 materialization。

   **2026-09-01 ground-coordinate correction.** 对 MMM 原始开发仓库及清理前
   commit `75814e5` 的逐行核对确认，其正式入口在优化每个人之前都执行
   `root_transl[:,1] -= min(joints[:,:,1])`。早期 augmentation runner 错把 pelvis-
   centred 的 raw Inter-X translation 直接送入以 `y=0` 为地面的 loss；25 个随机
   shape/50 个人的 joint floor 平均为 `-1.239096 m`，raw ground loss 平均为
   `0.230503/person`，按 MMM 对齐后仅 `1.41705e-4/person`。因此此前所有未做该
   对齐的 optimized shape pilot 均保留为坐标设置错误的诊断，不进入正式
   augmentation 结论。生产流程现在逐人临时落地优化，写回 NPZ/HDF5 前恢复各自
   floor offset，因而仍使用原版 raw mean/std；证据为
   `output/interx_shape_optimization_ground_contract_audit.json`。

   修正坐标后的同一样本 smoke 将 Eq.-6 penetration 从 `1.780078e-3` 降到
   `1.603608e-5 m^2/frame`，但 all-variable optimizer 仍把 foot sliding 从
   `0.002862673` 提高到 `0.008513596 m/frame`。变量归因表明，主要回归来自 root
   orientation 和两条 pelvis-to-foot 链，而不是评测参数。首次候选运行前已预注册
   `grounded_chain_preservation_pilot_v1.json`，明确标注为 local-refinement
   stabilization，不冒充 released optimizer。为保持原门禁严格为 delta `<=0`
   而不接受常量平移带来的 `7e-10` 浮点舍入，fixed-root fallback 同样在首次运行前
   冻结。首样本中 root translation/orientation 与两条腿链均逐 bit 不变，foot
   sliding 精确不变，Eq.-6 降至 `3.767293e-5 m^2/frame`；五样本门禁正在运行，
   全量 25 对与 6,238 条资产仍未获授权。
   授权语义已显式拆开：5 对通过只产生 `pilot25_authorized=true`，同时强制
   `full_augmentation_authorized=false`；只有未改动的 25 对门禁通过后才能授权
   全量资产。该修正没有改动已经冻结的 5 对 gate JSON，其 SHA-256 仍为
   `0574349e...2e06`。
   冻结的 5 对门禁现已完成并通过：相对相同 random-shape 输入，论文 Eq.-6
   penetration 均值从 `0.05619538655` 降到 `0.01181453976 m^2/frame`，
   penetrating-vertex fraction 从 `0.00655662224` 降到 `0.00132424550`，
   foot sliding 从 `0.00376482564` 微降到 `0.00375996063 m/frame`；五个样本的
   两项穿插指标全部改善且没有样本脚滑退化。机器决策
   `output/interx_shape_aug_groundfix_groundedchain_fixedroot_pilot5_assessment_seed20260828.json`
   只授权 25 对 pilot，仍明确保持 `full_augmentation_authorized=false`。25 对已按
   未改动的 optimizer 和物理门禁启动。
   在 optimized pilot 完成前已冻结机器可读的 full-run 门禁：25 个 paired key
   必须完整；penetrating-vertex fraction 与 foot sliding 的平均 paired delta 必须
   `<=0`；论文 Eq.-6 squared penetration 的平均 delta 及其 10,000 次 bootstrap
   95% CI 上界都必须 `<=0`。`tools/assess_shape_augmentation_pilot.py` 位于全量资产
   命令之前并 fail-closed，门禁不读取任何 test/generator 指标。
   当前仅完成 8 对时的只读诊断已发现 released foot-skating loss 的动态 mask
   可能发生“自我逃逸”：按优化前同一静止且落地 mask 统计的 7,096 个脚接触观测，
   水平位移均值从 `0.00348254` 增至 `0.01640763 m/frame`；优化后 `99.9718%`
   仍低于高度阈值，但只有 `78.4667%` 仍低于动态速度阈值。该结果只作为 incomplete
   pilot 诊断，不提前替代 25 对门禁，也不允许放宽 foot-sliding 阈值。若完整门禁
   复现此失败，将保留原论文/released optimization 结果，并另做唯一改动为“冻结
   初始接触 mask”的 paired 单因素修正版；loss 权重、三维速度定义、样本、optimizer
   和物理门禁均保持不变。该条件方案已在 released pilot 仅完成 `11/25` 时完成
   哈希预注册，证据为
   `output/interx_frozen_foot_pilot_preregistration_audit.json`，因此不是根据完整结果
   事后选择阈值或改动因素。
   共享后端已通过 3 帧 CPU 确定性端到端审计：fresh、首次 shared 和连续第二次
   shared 的全部优化输出均为有限值且逐 bit 一致，最大绝对差为 0。最终 shape
   结果与 review 报告必须同时绑定该审计、两份权重及五个实现文件的当前 SHA-256。
   由于 timing-positive pilot 表明优化预计需要数十个 GPU-days，曾预注册 exact
   frame-batching 晋级门禁；但后续完整 100-step same-seed 真实样本已否决把 8 改为
   4：虽然一迭代 micro-audit 可过，长轨迹最终 total loss 分别为 1.51023 和
   1.59938。故不再以速度阈值晋级 frame-batch=4，正式设置固定为 8。样本级分片
   仍用于跨 GPU 分配与原子恢复，但同一 GPU 只运行一个进程；该分片显式要求
   `query_chunk=10475`，旧开发值 `512` 不能混入正式资产。
   flat-hand 修复使先前的 query-chunk 证据失效，因此已在修复后的同一 production
   backend 上重新做两个真实 motion 的串行 query-chunk 审计：`query_chunk=10475`
   与基准 loss 完全一致，最大梯度相对误差 `1.53e-6`，两条 motion 的最小加速
   `7.09x`；晋级证据为
   `output/volumetric_smpl_flat_hand_query_chunk_assessment.json`，当前 pilot 只使用
   这一通过门禁的查询分块，不启用尚未通过的 frame batching。
   后续加速门禁又发现一项比较口径错误：旧记录把 VolumetricSMPL 的 120-step
   pilot（`24.0561 s/sample`）与 nearest 的 200-step pilot（`29.3729
   s/sample`）相除，因此原先报告的 `1.221x` 不是公平加速结论。现在
   `tools/assess_fast_selective_shape_pilot.py` 会强制两侧的 ordered samples、迭代数、
   学习率、fine-stage、seed、timing、foot-mask、loss reduction、优化变量及输入
   hash 全部相同；机器证据
   `output/interx_shape_aug_volumetric_fast_selective_pilot25_gate_v3_fair_speed_audit.json`
   已按新门禁正确拒绝旧比较。HumanML3D/AMASS 正式 pilot 的 nearest 与 volume
   均从同一个 120-step 参数数组启动；若数值、物理或同预算速度任一项失败，就回退
   原版 nearest，不能通过减少步数宣称加速。
   训练对照为同一官方 train split 和相同预算下的 `no augmentation`
   与 `original + optimized random-shape augmentation`；二者在相同官方 test 冻结
   请求和统一 evaluator 上比较 FID/R-Precision/Matching/Diversity/MModality，
   并报告 contact、temporal IoU、penetration、foot sliding、按身高/BMI/beta 距离
   分桶的变化。随机换 shape 后不重新 optimization 的版本作为必要消融，用来证明
   收益来自物理一致的增强而不是单纯扩大 beta 分布。
8. **无 contact timing。** 比较 GT timing、文本预测 timing、全片弱约束和完全
   无 timing 四种设置，报告语义、接触、穿透和耗时。该实验直接回答实际使用时
   是否必须提供精确接触时刻。四种设置现已在 exporter 中显式标记，避免把 test
   annotation 偷称为普通输入。文本预测器只用 train caption/contact 建 kNN 先验，
   在 val 上固定 `k=10/threshold=0.4`（frame-level F1 `0.65158`），不读取 test
   contact；冻结 test 的独立 timing 评测为 precision/recall/F1
   `0.52438/0.70170/0.60021`，onset/offset MAE 为 `24.03/24.35` 帧。全片弱约束
   预注册为论文 oracle logit bias 的四分之一（`lambda=0.25`），不依据 test 调参。
   最终 checkpoint 的 replication-0 四路语义/MM/物理评测已排在 GPU1 主链和
   sampling Pareto grid 之后。
9. **LLM 鲁棒性。** 在冻结的 128-caption 子集上比较全量标注所用
   `gpt-5.6-luna`/low、较强的 `gpt-5.6-terra`/low 一致性参照和确定性规则
   parser；四个互斥 strata 各 32 条，分别覆盖左右、多接触、人物指代和人工
   timing 阴性描述。人工 Inter-X timing 只能监督二值接触存在性，因此报告
   presence precision/recall/F1；Inter-X 没有人工 75-region 标签，部位指标只能
   报 Luna/规则相对 Terra 的 ordered 与人物交换容忍 agreement，不能称为
   oracle part F1，Terra 也不能称为 GT。另报 invalid final JSON、不可执行约束、
   retry/failure、每样本摊销 token 和平均/P95 调用时间。接触优化后的几何指标由
   独立的 optimizer ablation 负责，不把 LLM 自洽性冒充几何正确性。
   旧 MMM val cache 的离线审计已完成：570/570 ID 齐全，共 683 条结构化 pair，
   但 152 个样本为空；559 条记录的 alignment 值越界，7 条记录的 Contact Area
   是未允许的组合值，合计 345/570 个文件不能通过当前 schema。更重要的是旧
   pickle 未保存 raw response、模型版本、prompt hash、parse status、重试、耗时
   或 token usage，因此不能从空列表反推“确实无接触”还是 API/解析失败。报告见
   `output/interx_contact_pair_audit.json`。新 `codex_strict_v3` extractor 删除了
   论文 prompt 中不存在的 Normal Alignment 列，未知字段 fail-closed，并逐样本
   保存完整 provenance；必须用新目录重建，旧缓存不进入正式 optimization。
   正式 test cache 进一步绑定 `replication_00.npz` 的 1,696 个精确 caption、
   request index 与 SHA-256；不能读取每个文本文件第一行，因为冻结 evaluator 会
   在三条 caption 中随机选择（首个请求即选择了第二行）。optimizer 同时核对
   generation record 的 request/caption hash。原 MMM 硬编码 Gemini key 已被
   Google 标记为 leaked 并返回 403，不能使用。shape augmentation 的 train 部位
   标注和正式 test cache 均改用 ChatGPT 登录态下的 `gpt-5.6-luna`/low，无 API key；严格 v3 JSON
   schema 将身体部位直接限制为论文 75-region 字典，并保存 batch raw response、
   session、token、重试和逐样本 hash。9,110 条 train 标注在计时 wrapper 加入前
   已启动，因此其 294 次成功/失败调用都有 token、但没有可靠 wall-time；禁止用
   进程总时长回填逐调用耗时。正式 1,696-caption test 与 Luna/Terra 鲁棒性调用均
   使用计时 wrapper，并把成功及失败尝试计入平均/中位数/P95 时间和 token。
   Gemini 只作为论文 LLM 的可选
   复现来源，不再阻塞 train augmentation，也不进入当前可复现的正式鲁棒性表。
10. **脚滑。** 报告足底接触帧上的平均/95 分位水平速度、滑动距离和有脚滑样本
    比例；同时比较优化前后，确认 contact 优化未把手部收益转化为脚部伪影。
11. **第二生成数据集（Inter-X 完成后处理）。** 在 InterHuman 官方 split 上比较 HHC-M、InterMask 与
    CoShMDM；HHC-M 在确需该数据集模型时重训，baseline 优先使用作者发布的
    InterHuman checkpoint，只有权重不可用或协议确实不兼容时才重训。所有方法
    使用相同 evaluator、样本数和 repeated-eval 协议，报告
    FID/R-Precision/Matching/Diversity/MModality。由于 InterHuman
    不提供与 Inter-X 等价的 SMPL-X shape/hand 标注，该表只验证跨数据集生成，
   不用于证明 shape-aware 或精细接触；后两项由 Inter-X optimized random-shape
   augmentation 对照与 CHI3D mesh/contact 实验验证。
   当前机器只有已审计的 InterMask InterHuman 作者 VQ/Transformer 权重，官方
   InterHuman train/val/test split 尚未在本机完整落地；按用户安排，baseline 已在
   另一台服务器运行，本机不重复执行，待 Inter-X 完成后再导入并核验 checkpoint、
   split、请求数与 evaluator provenance。Inter-X 的 1,696 条 frozen strict Luna
   test-contact cache 已完整通过审计，正式 optimization 目前只等待最终 HHC
   replication-00 生成及其 SMPL-X 转换；CHI3D official test 仍不可用。所有外部
   资产以最新审计为准，不拿其他数据版本补空缺。额外检查到
   ProsePose 中虽有文件名为
   `CHI3D/test_optimization.pkl` 的缓存，但配套 504 条 contact key 全部属于官方
   train 主体 `s03`，processed record 也指向 `CHI3D/train/s03`；它只是该仓库的
   内部 experiment split，不得冒充官方 CHI3D test package。

## P2：完整性和定性证据

12. 增加 TiMotion 定性结果，以及不同身高、体型、性别组合下各 baseline 的
    统一视角视频/帧图；选择规则和随机种子写入 manifest。
13. 用户研究报告参与者数量、问题、随机化/盲测方式、偏好比例、置信区间和
    显著性检验，而非只给定性陈述。真实参与者研究尚未进行，机器审计会保持
    `blinded_user_study_not_conducted`，绝不生成合成参与者或虚构偏好结果。
14. 报告 LLM 名称/版本、prompt、输入输出校验、重试策略和每样本 token；当前
    标注通过 ChatGPT 登录态的 Codex CLI，不使用 API key，因此 API 账单费用记为
    N/A，不拿公开 API 单价伪造实际成本。另报生成、LLM、优化三个阶段的平均与
    P95 时间。
15. 发布所有评测生成文件的 sample ID、caption、shape、seed、checkpoint 和
    protocol manifest，避免 GT 与生成样本数量或裁剪方式再次不一致。
16. 修正文稿中 Fig. 2 caption 的 `Maksed` 拼写，并把 contact/penetration、用户
    研究数值、所用 LLM 版本和防 hallucination 校验从补充材料移入主文。当前
    工作区只有投稿 PDF、没有 LaTeX 或 Fig. 2 可编辑源文件，因此该项保持
    `manuscript_pending`；不得直接篡改 PDF 或仅在实验报告中把它标成已修复。

## 执行顺序与依赖

- VQ 150k 完成后生成全 split token cache 和 contact-code lookup；随后启动 MGT
  300k。两阶段 checkpoint 均可断点续训。
- CoShMDM 的 Inter-X 数据转换与专用 normalization stats 完成后，先跑单 batch
  前后向，再按作者训练预算启动重训。
- TiMotion public Transformer+LPA 使用同一 262-D Inter-X 转换和 train-only
  stats；现有主训练释放 GPU 后先做单 batch 前后向，再以 micro-batch 32×8、
  每 epoch 568 micro-batch、1,500 epochs 启动，最终为 106,500 optimizer updates。
  这里重训不是忽略作者 checkpoint：固定 public revision 的 README 虽提供两条
  预训练权重链接，但其唯一数据入口指向 InterHuman/InterGen，原始
  `configs/datasets.yaml` 也只有 `interhuman{,_val,_test}`，Git tree 中没有任何
  Inter-X 路径或 checkpoint。因此没有可直接用于官方 Inter-X split 的发布权重；
  合同审计逐 blob 哈希绑定上述事实，禁止把 InterHuman 权重改名后进入公平表。
  1,500 来自论文 Sec. 4.1.1；公开 runner 的 1,900 默认值与 README 的 LPA 命令
  存在歧义，已作为披露项而不是训练预算采用。启动前 contract audit 已验证全部
  9,110 个 train ID 均通过 15-frame loader 下限，并绑定 11,388 对转换数据、
  官方 `all.h5`、转换脚本、stats 与模型传递依赖；正式结束后还会全量重哈希。
  预训练 smoke 发现并定位了发布代码的 padding-only RO NaN：发布实现先对所有帧的
  hip-to-hip orientation 做 normalize，再乘有效帧 mask；padding 中零向量的 `0/0`
  因而污染总 loss。端口仅在 mask 外以固定有限向量代入后再归一化；有效帧的
  prediction、target 和 RO 值 bitwise 不变。证据冻结在
  `output/timotion_padding_ro_stabilization_audit.json`。修复后的 50-update smoke
  checkpoint、生成和 official scorer 都为 finite，online W&B run `oi4wa5qn` 的
  远端 history audit 通过；正式 replacement run `kdz5s3ra` 已从 step 0 在 GPU 0
  启动。中断且无 checkpoint 的 `jqh616am` 仅保留 provenance，禁止进入结果。
  `output/timotion_gradient_accumulation_audit.json` 另以运行时数值验证了
  `32×8` 对 global batch 256 的 mean-loss 梯度、0.5 global-norm clip 和 AdamW
  update 等价：clip 前后最大梯度差不超过 `1.49e-17`，更新后权重 bitwise 相同。
  正式 replacement 已完成全部 10 个 warm-up epoch；冻结的远端 W&B 审计
  `output/timotion_warmup_stability_audit.json` 验证 14/14 次预期遥测齐全、
  gradient/weight nonfinite 均为 0，global weight L2/max-abs 相对跨度分别只有
  `0.0462%`/`0.3251%`。第 10 个 epoch 的 val loss 为 `5.229714`；训练继续，
  单 batch 的 VEL 峰值只保留为诊断，不作为中止依据。
  最新在线刷新已覆盖至 sparse telemetry optimizer step 17,650、dense loss/LR
  step 17,688 / 完成 epoch 249；审计现按实际共写
  事件分别读取逐 update 的 loss/LR 和每 50 update 的 telemetry，完整 diffusion
  profile 十项训练指标全部存在且 finite。最新 step 17,650 的 pre-clip gradient
  L2/max-abs 为 `4.086609/0.226875`，随后仍执行发布配置的 global-norm `0.5`
  clip；weight L2/max-abs 为 `185.255936/1.173651`，两类 non-finite count 仍为 0，
  dense step 17,688 loss 为 `0.085512`。此前 step-17,500 的有限尖峰
  (`loss=0.213035`、`VEL_A+VEL_B=0.119416`、下一步 `0.128459`) 继续只保留为诊断，
  没有持续。W&B runtime history
  还实际包含 249 个有限 `val/loss` 事件，以及 step 7,100/14,200 两组 FID、三项
  R-Precision、Matching、Diversity；数值与独立哈希的正式 validation JSON 完全
  一致，不只依赖静态字段检查。warm-up 遥测窗口也修正为不越过边界：710 update
  中只包含 step 50--700 的 14 个事件，不再错误向上取整并纳入 step 750。
  进一步核查确认日志中 zero-based epoch 10 的 `1.099932e-4` 不是端口错误：
  发布 `CosineWarmupScheduler` 明确使用 `epoch <= warmup`，因此在边界产生一次
  `11/10` overshoot，并在 epoch 11 回到 `9.999254e-5` cosine 分支。远端逐 epoch
  学习率与发布公式误差为 0，证据为
  `output/timotion_scheduler_boundary_audit.json`；公平端口保留此发布行为。
  两个预注册 100-epoch semantic-validation 节点现均完成：step 7,100 的
  FID/Matching/Top-1 为 `78.694913/10.895258/0.058824`，step 14,200 改善到
  `68.629341/10.477775/0.053309`。FID 与 Matching 在下降，但绝对质量仍远弱于
  InterMask/HHC，且 Top-1 略降；训练按发布 1,500-epoch 预算继续，不因这两个
  early validation 点改动目标或选择规则。
  后续 epoch 400/500 FID 分别为 `55.234978/90.508970`，但隔离依赖后的适配审计
  已确认当前 TiMotion shape-specific root fit 错把标准 row-concatenated 6D 当作
  interleaved 6D 解码；identity roundtrip 的矩阵与正交误差均为 `1.0`。旧的
  TiMotion/CoSh “bitwise equal” 审计还受到同名 `utils` 模块缓存污染，属于假通过。
  证据为 `output/timotion_rotation6d_adapter_contract_audit.json` 和
  `output/timotion_cosh_adapter_equivalence_audit_isolated.json`。当前训练不中断，
  但现有 semantic FID 只作诊断；训练完成后必须先对 epoch 400/500 做修正导出器的
  matched-seed 重放与统一 evaluator 复评，之后才能正式选择 checkpoint。修正约定
  的代码级等价审计已在八组测试上 bitwise equal，证据为
  `output/timotion_cosh_adapter_equivalence_audit_rotationfix.json`；它不替代 FID 重放。
  epoch 800/step 56,800 候选现已完整生成：旧 adapter 训练内诊断 FID 为
  `60.354285`，Matching `10.043025`，Top-1/2/3 为
  `0.075368/0.128676/0.167279`；该数值继续只作诊断，不能参加最终选点。
  对应 1.43GB full checkpoint 的 294 个 state_dict tensor 与 812MB
  weights-only validation checkpoint 逐 tensor bitwise identical，205,483,014 个
  tensor value 全部 finite；metric JSON 也逐哈希绑定该 weights checkpoint 和
  544 条 validation。证据为 `output/timotion_step56800_checkpoint_audit.json` 与
  `output/timotion_step56800_validation_checkpoint_pair_audit.json`。训练结束后的
  post-processing 现要求对全部 15 个候选完成同一 bitwise pair audit，之后才运行
  standard row-concatenated 6D 重放与 selection。
  CoShMDM 的正式 20-rep 输出无需重跑：其冻结 revision 中 rotation dependency
  保持 clean 且 identity roundtrip 误差为零，补充证据为
  `output/coshmdm_rotation_dependency_provenance_audit.json`。
- 两个模型完成后先跑 1 replication 的协议 smoke test，核对样本数为 1696，
  再跑完整 20 replications + MModality。
- P0 物理指标和 CHI3D 不复用语义 evaluator；它们从同一生成 manifest 读取
  SMPL-X 参数，保证优化前后与 baseline 的样本逐条配对。
- timing ablation 的 `request_oracle` 明确使用冻结 test annotation；`none` 不施加
  logit bias；`text_predicted` 读取 request-hash-bound 的 train/val-only 预测；
  `full_valid_weak` 只知道请求长度。它是一个 replication-0 sensitivity，不能以
  单次结果替代 oracle/none 的 20 次主结果。
  当前四行已统一用 CPU evaluator 重评分，汇总器会拒绝 CPU/CUDA 混用。FID 分别为
  none `0.147044`、request-oracle `0.148354`、text-predicted `0.145504`、
  full-valid-weak `0.149320`；对应 contact F1 为 `0.566989/0.574636/0.569453/0.566315`。
  因此 oracle timing 在该单次种子上提高 contact F1，但没有提高 FID；text-predicted
  获得最低 FID 和最高 Top-1 (`0.441627`)，这些差值必须保持“单次敏感性”表述。
- Inter-X shape augmentation 不改变官方 validation/test；增强样本、LLM 部位、
  timing、随机 shape pool 和 optimization 全部只来自官方 train。无增强、仅随机
  换 shape、optimization 后随机 shape 三组使用相同 VQ、原版 raw mean/std、MGT
  配置、300k update 数、checkpoint 选择与官方冻结 test requests；action 泛化
  取消。两个新增训练臂必须使用稳定 W&B run ID 在线记录，并在进入
  对比表前通过完整 300k history audit（train/val 指标齐全、梯度/权重 finite）；
  resume 必须复用同一 run ID，不能把断点续训拆成无法审计的多条曲线。
- CHI3D custom CV 的三折严格按 G2/G3/G4 划分；所有 preprocessing stats、
  codebook、模型选择和超参数选择只能从每折的两个训练组（可再做确定性 inner
  validation）获得。不得使用 held-out 组选择 checkpoint 或 normalization。
  若只运行全 373 条 GT optimization，则标题和 manifest 必须包含
  `official_train_in_sample_diagnostic`，不得并入生成模型泛化主表。
  已生成 `configs/chi3d_cv/manifest.json` 及 train-only
  `configs/chi3d_cv/stats_manifest.json`；三折 train/inner-val/test 数量分别为
  `223/30/120`、`217/30/126`、`216/30/127`，每折均精确分割 373 条且无 ID
  交集。stats 特征转换在全数据上以 mean/std 最大绝对误差
  `5.96e-8/2.98e-8` 复现 MMM 参考实现，之后才写出逐折统计。
- replication 0 精确 GT 物理基准目前为：10 cm contact F1 `0.77315`、temporal
  IoU `0.63019`、false-contact rate `0.10363`、contact-frame 最小关节距离
  `0.06694 m`、grounded-foot horizontal speed `0.12166 m/s`。这些数值是
  annotation/skeleton 协议的 reference，不作为训练目标。
- Google 官方当前把论文所称 Gemini 3 Flash 的 API code 列为
  [`gemini-3-flash-preview`](https://ai.google.dev/gemini-api/docs/models/gemini-3-flash-preview)；
  它是 preview alias，可能弃用或变化。每条新 response 必须记录 requested model
  与返回的 model version，不能仅在论文中写一个家族名。
- 定性比较预注册在 `configs/qualitative_comparison_selection.json`：从冻结的
  replication 0 独立于指标地选择 hug/high-five/handshake/kick、male-male、
  female-female、mixed-gender 和两个 beta 极端，共 9 个案例。最终六方法逐案例
  共享 request、repeat、六个相对帧位置、相机和坐标范围，输出与源 HDF5 哈希
  写入 `output/qualitative_comparison/manifest.json`。

## 已完成的公平协议结果

- 容量审计使用实例化后的参数逐项计数：HHC 的 shape-aware VQ 为
  `36,606,748`；正式 raw-12288 MGT（排除其共享引用的冻结 VQ）为
  `73,986,640`，两阶段唯一 motion 参数合计 `110,593,388`；推理时另加载固定的
  CLIP ViT-B/32 `151,277,313` 参数，总驻留参数为 `261,870,701`。较早的
  `71,840,336/108,447,084/259,724,397` 是 8,096-code 调查分支口径，不进入
  raw-12288 正式表。CoShMDM 总参数为
  `305,442,310`，其中可训练参数 `182,440,966`、冻结文本侧参数
  `123,001,344`。InterMask 的 VQ 为 `45,418,502`，Transformer 总参数
  `177,333,376`（其中可训练 `54,332,032`），故唯一 motion 参数
  `99,750,534`、包含固定文本 encoder 的总驻留参数 `222,751,878`。最终效率表
  将分别报告 motion/trainable 参数与包含固定文本
  encoder 的总驻留参数，避免用两种口径制造不公平的“模型大小”结论。
  训练预算也分开披露：正式 HHC 使用导入的大清理前 raw-12288 VQ 175k
  checkpoint，其历史配置为 `175k×256 = 44.8M` 已完成 windows；MGT 为
  `300k×96 = 28.8M` cached sequences。`150k×256 = 38.4M` 只属于废弃的
  8,096-code 调查分支。CoShMDM 为 `568k×64 = 36.352M`
  augmented sequences。主表的“公平”指同一原始 train/val/test ID、冻结请求、
  生成次数和 evaluator，而不是假装不同范式的作者训练预算相同；另补 wall-clock、
  GPU 与 sampling latency 供 compute-aware 解读。最终队列通过
  `tools/audit_training_wallclock.py` 将 HHC VQ/MGT、CoShMDM 和 TiMotion 的
  launch、完成端点、最终 checkpoint/状态审计哈希、optimizer updates、effective
  batch 与 sample exposure 绑定到同一报告；TiMotion runner 会直接记录精确
  post-fit 时间，已启动的 HHC/CoSh/VQ 则使用冻结 launch time 到 terminal
  checkpoint mtime 的统一 wall-clock 边界。

- HHC MGT checkpoint 按固定 validation FID 选择时使用 validation split 自带的
  oracle contact timing（不读取 test），采样协议为 `sample_intra_cas`、50 个外层
  step、temperature 0.5、CFG 3.5、contact guidance 1.0；Transformer 训练输入本身
  不含 contact conditioning。最终 no-timing 与 oracle-timing 主表行共享该同一
  checkpoint，selection manifest 会显式记录上述条件，避免把 oracle-val 选择
  含糊写成无 timing 选择。正式 no-timing 命令必须显式传
  `--hhc-contact-timing-source none`；runner 会拒绝
  `--no-contact-guidance` 与任何 oracle/predicted timing source 的组合，且最终
  manifest 同时门禁 requested/effective source 均为 `none`。
- 已用同一 CPU/current-scorer 重评 no-timing 与 oracle-timing 的全部 20 次冻结
  HDF5。no-timing 主表 reference 的 FID/Matching/Top-1/2/3 为
  `0.555712/3.712078/0.422022/0.602860/0.710407`；oracle timing 的 FID 改善仅
  `0.010580`，paired bootstrap 95% CI 跨 0，Holm p=`0.6925`，不显著。
  Matching 改善 `0.019208`（Holm p=`0.03834`）与 Top-2 改善 `0.008343`
  （Holm p=`0.002779`）显著，其余 ordinary/MM 指标不显著。证据为
  `output/fair_eval/hhc_formal_raw12288_repaired_mgt_common50_no_contact_cpu_metric_audit.json`
  和 `output/hhc_contact_timing_cpu_evaluator_paired.json`；oracle timing 仍只作为
  独立输入信息消融，不作为公平 baseline reference。
- InterMask common-50 的同一 CPU/current-scorer 重评已完成并通过全部 HDF5、请求、
  evaluator 源码/权重 provenance 哈希门禁：FID
  `0.374280 ± 0.014655`、Matching `3.700004 ± 0.020057`、Top-1/2/3
  `0.405366/0.598850/0.709906`、Diversity `8.991844 ± 0.088159`、
  MModality `1.675817 ± 0.070547`。相对 HHC no-timing，InterMask 的 FID
  优势为 `0.181432`，20/20 replication 获胜且 Holm 校正后显著；HHC 的 Top-1
  优势为 `0.016657`，18/20 获胜且显著。Top-2/3、Matching 和 diversity error
  不显著。证据为
  `output/fair_eval/intermask_official_exact_common50_cpu_metric_audit.json` 与
  `output/hhc_vs_intermask_cpu_evaluator_paired.json`。这证明当前 HHC 的 FID 缺口
  不是旧/新 evaluator 差异所致，MGT 训练设置诊断仍必须继续。
- 截至 corrected raw-12288 MGT 的 200k 节点，固定 validation FID 依次为
  `1.18237 / 0.78011 / 1.58449 / 1.26432 / 2.39061 / 3.06874 / 1.96206 / 2.81110 / 2.26448 / 2.43224 / 2.01908 / 2.25724 / 1.72068 / 1.82475 / 2.27175 / 2.33080 / 1.83528 / 2.02151 / 2.21271 / 1.62599`
  （10k 到 200k），因此当前 validation-best 仍是 20k；90k rolling 与 validation checkpoint 的 global step、scheduler、
  947 个模型与 1,041 个 optimizer 浮点 tensor 已全部通过有限值审计。这一早期
  曲线不会替代 300k 完整训练后的选择结果。无 validation 的 95k rolling
  checkpoint 也已通过完整状态审计（global step `95,000`、947 个模型 tensor、
  1,041 个 optimizer tensor 均有限），文件 SHA-256 为
  `ee6d0c0832ff8bd7716af5c04329e11646068cf56a24e65f275c388b3d6b2ce8`。
  100k rolling 与 validation checkpoint 也分别通过完整状态审计；两者的 947 个
  model tensor、optimizer、scheduler 与 hyperparameters bitwise identical，只有
  预期的 callback/loop validation 元数据变化。100k rolling/validation SHA-256
  分别为 `c0755d9371675ee01040054b33d4aab6f5453bf9a41a5c6a102f0049833b509c`
  与 `13c069a398714a3e4aa33d26fe3f37b97ca402b7ac0b6bb39798feee90c9f0c3`。
  105k rolling checkpoint 也通过相同有限值/step/scheduler 审计，SHA-256 为
  `f40676a95cb095638763b0748d43c06c82d3dddd31dbd9fc2f17a778b025c2f8`。
  110k 当前 FID 为 `2.0190837383270264`；rolling 与 validation checkpoint
  的模型、optimizer、scheduler、hyperparameters 再次 bitwise identical，审计见
  `output/hhc_precleanup_raw12288_corrected_110k_pre_post_validation_pair_audit.json`，
  两个 checkpoint 的 SHA-256 分别为
  `c70ee81bbd9e3795a5dbdfaaca7a870413d2727dbdfaecada850d39c58118eda`
  与 `6c1483fc0520ad7b2301a7dcb82b442ef9f6d32e5bb2fbc03889c9578c3d6987`。
  115k rolling checkpoint 也通过相同的 step/scheduler/有限值审计，SHA-256 为
  `e0d90d9c716571db9247d225b63c45a9b497a8441c9be84b11a95734526a3b8f`。
  120k 当前 FID 为 `2.2572407722473145`；rolling/validation 状态再次 bitwise
  identical，审计见
  `output/hhc_precleanup_raw12288_corrected_120k_pre_post_validation_pair_audit.json`，
  两个 checkpoint 的 SHA-256 分别为
  `e359d76f635d81cb66ce90756d0041ff366954305706f57d8976d6634ac0cdf6`
  与 `ad9c61b1a2af262402cbefd54e4710a377374e88217836be46b150287934bd07`。
  125k rolling checkpoint 也通过 step/scheduler/有限值审计，SHA-256 为
  `5d902d50022f98d4867bdfcd8a75db3e3fde995d6c33dde7431e64e2ef358ed9`。
  130k 当前 FID 为 `1.720682144165039`；rolling/validation checkpoint 的
  947 个模型 tensor、optimizer、scheduler 与 hyperparameters 再次 bitwise
  identical，审计见
  `output/hhc_precleanup_raw12288_corrected_130k_checkpoint_pair_audit.json`。
  rolling/validation checkpoint 的 SHA-256 分别为
  `625240ee3a921977cdd326b7cac0d0fbc48c0e337ba38e52e0a0c067ce879653`
  与 `23a9587a5cbfa6fc28b3aea8dd8a80e4ca4712735c87fb423215be5b77f8a6d3`。
  135k rolling checkpoint 也通过同样的 step/scheduler/有限值审计，
  SHA-256 为
  `9e5eab0b197b7d11f4553df577c6f699dbd50c85960e76f0e0032a5e27819646`。
  140k 当前 FID 为 `1.8247472047805786`；rolling/validation checkpoint 的
  模型、optimizer、scheduler 与 hyperparameters 再次 bitwise identical，
  审计见
  `output/hhc_precleanup_raw12288_corrected_140k_checkpoint_pair_audit.json`。
  rolling/validation checkpoint 的 SHA-256 分别为
  `5e25dd68416f09d74b131999d92be3406d78c9bad6a0ea5effd56c472c371561`
  与 `a478dcf34f7f59619275fda64c7f932e644110ea46bb8aa37ebef2f065c45e8c`。
  145k rolling checkpoint 也通过同样的 step/scheduler/有限值审计，
  SHA-256 为
  `9b30b05be2e6db13cee8b20ddc01d49f17242fffacdb11a688b459a3ac686637`。
  150k 当前 FID 为 `2.2717506885528564`，未进入 callback 的
  `save_top_k=10`，因此 Lightning 按配置淘汰了该 validation checkpoint；150k
  rolling 曾被监视器确认为完整文件，但在独立内容审计前已被下一次
  `save_top_k=1` rolling 保存替换，故不对该节点声称 checkpoint pair bitwise
  一致。随后 155k rolling checkpoint 已通过 step/scheduler/有限值审计，
  SHA-256 为
  `4fb0793d27ad859d180616d35457e8b8939c266ecf95f4b6a5ae03287dfd19df`。
  160k rolling 在原子落盘后立即完成 step/scheduler/有限值审计，SHA-256 为
  `48aa3422b53b557ee7dad8dbcf8d4ef00ac95abc924b63d5b01de14390d09a61`；
  对应的 544-sample validation FID 为 `2.3308000564575195`，同样未进入
  `save_top_k=10`，因此没有保留 validation checkpoint，也不对该节点声称
  rolling/validation pair bitwise 一致。
  165k rolling checkpoint 也通过 step/scheduler/有限值审计，SHA-256 为
  `941e3486644ac2a0ab94db600ae64e5dfdb842251a007765299e9fb0e56b54b6`。
  170k rolling 在原子落盘后立即通过 step/scheduler/有限值审计，SHA-256 为
  `03a4d5c7098bec0396dddb243e885843692f88bfb397e6ab363876af6da7187d`；
  对应 validation FID 为 `1.8352792263031006`，post-validation checkpoint
  SHA-256 为
  `7a0ed442e8afd0c4ce9a207df11afbe9ee549fa0a427b8aa104d2da35173027f`。
  两份 checkpoint 的模型、optimizer、scheduler、hyperparameters bitwise
  identical，审计见
  `output/hhc_precleanup_raw12288_corrected_170k_checkpoint_pair_audit.json`。
  rolling 中 callback `current_score` 为最近一次实际保留的 140k FID
  `1.8247472047805786`；150k/160k 被 top-10 淘汰的 W&B FID 不会写入该
  callback 字段，二者语义已分开记录。
  175k rolling checkpoint 也通过 step/scheduler/有限值审计，SHA-256 为
  `f930f0788e18ec8346790b726f6b7ac10980fcf29f733e1027c719106d2ad56a`。
  180k rolling 在原子落盘后立即通过 step/scheduler/有限值审计，SHA-256 为
  `7a5a2121bdafba6c7be0da42fd74535988d672102a2f9eee70a5097c7c09610e`；
  对应 validation FID 为 `2.021514892578125`，post-validation checkpoint
  SHA-256 为
  `393a2bcff7c4d6f98772f516c4372ad9da8826ae5b23d8bdd7a62bbde7ff8ea5`。
  两份 checkpoint 的模型、optimizer、scheduler、hyperparameters bitwise
  identical，审计见
  `output/hhc_precleanup_raw12288_corrected_180k_checkpoint_pair_audit.json`。
  185k rolling checkpoint 也通过 step/scheduler/有限值审计，SHA-256 为
  `0d4cc68d9cefdbc53c31489682ebb026ae96b501ad17ab26411885af2b469bbd`。
  190k rolling 在原子落盘后立即通过 step/scheduler/有限值审计，SHA-256 为
  `91d81014a31be0fd703d4f14778ec824349aa0517231d3af05f6986b92aebce9`；
  对应 validation FID 为 `2.21270751953125`。170k/180k 进入保留集合后，
  `save_top_k=10` 的最差保留阈值已收紧到 `2.021514892578125`，因此 190k
  validation checkpoint 按配置被淘汰，不对该节点声称 rolling/validation
  pair bitwise 一致。
  195k rolling checkpoint 也通过 step/scheduler/有限值审计，SHA-256 为
  `9ad3648da7189555b6042fc39eacf07900d83ff47e5dd882fa98d51f972aac21`。
  200k rolling 在原子落盘后立即通过 step/scheduler/有限值审计，SHA-256 为
  `99046a70eb70dfa2369950197f836f93be044d8a7a447c0ac6987461e2755280`；
  对应 validation FID 为 `1.6259876489639282`，post-validation checkpoint
  SHA-256 为
  `ea12750b23e1fda361191ce8e33b31db9e1b2ac3411d0800402cce3a8c7c2d1b`。
  两份 checkpoint 的模型、optimizer、scheduler、hyperparameters bitwise
  identical，审计见
  `output/hhc_precleanup_raw12288_corrected_200k_checkpoint_pair_audit.json`。
  205k rolling checkpoint 在原子落盘后也通过 step/scheduler/有限值审计：
  global step 为 `205000`，947 个模型浮点张量与 1041 个 optimizer 浮点张量
  全部有限，SHA-256 为
  `9fc82d299d778916a679a6f1233a3e8b09bd1e4f0cafc028fb9875abf4ccdb2c`；
  审计见 `output/hhc_precleanup_raw12288_corrected_205k_checkpoint_audit.json`。
  210k rolling 同样通过状态审计，SHA-256 为
  `6093dfbbf4e48bc88c1efae4a84715a9a930b39d2197a9a03f22b1acf9199067`；
  对应完整 validation FID 为 `1.788170337677002`，进入 top-10 后的 checkpoint
  SHA-256 为
  `007a6877e4946556841e38915b4360557804e7ee7a6fbedd7e0d45bd50f6d6a9`。
  两份 checkpoint 的模型、optimizer、scheduler、hyperparameters bitwise
  identical，审计见
  `output/hhc_precleanup_raw12288_corrected_210k_checkpoint_pair_audit.json`；
  当前最优仍为 20k 的 `0.78011155128479`。最终 selection 实际读取的保留文件
  `model-000020000-0.7801.ckpt` 已单独通过 step/scheduler/有限值审计，SHA-256 为
  `7fcc995cffdd6017e0655df52ad8476f9dfddae1c40e7dfd12b58293deff2fcd`，见
  `output/hhc_precleanup_raw12288_corrected_20k_validation_checkpoint_audit.json`。
  215k rolling checkpoint 也通过 step/scheduler/有限值审计，SHA-256 为
  `fbc5984133f9a310041c57085c74ef5c3f9995955dc01ed547a187446c957f39`，见
  `output/hhc_precleanup_raw12288_corrected_215k_checkpoint_audit.json`。
  220k rolling checkpoint 同样通过审计，SHA-256 为
  `5add5c2d52809dc010ad45adca0f673aba04783450eb8858b5ea28fe9a533da9`；
  对应完整 validation FID 为 `2.696781873703003`，高于当前 top-10 最差保留
  FID `1.9620590209960938`，因此 validation checkpoint 按配置被淘汰，不对该节点声称
  rolling/validation pair 一致。rolling 审计见
  `output/hhc_precleanup_raw12288_corrected_220k_checkpoint_audit.json`。
  225k rolling checkpoint 也通过 step/scheduler/有限值审计，SHA-256 为
  `9c0b9590e31fe7c779c8b7d015b76e3e22e9825ba23c1cf6c8deeb4a06d4691c`，见
  `output/hhc_precleanup_raw12288_corrected_225k_checkpoint_audit.json`。
  230k rolling checkpoint 同样通过审计，SHA-256 为
  `8cbcc797fca3a3785bdf8b81ef618fbb12f9101ee0bc94230a760ff321dc9b7c`；
  对应完整 validation FID 为 `2.5630898475646973`，高于 top-10 最差保留
  FID `1.9620590209960938`，因此没有 validation checkpoint 或 pair 声明。rolling
  审计见 `output/hhc_precleanup_raw12288_corrected_230k_checkpoint_audit.json`。
  235k rolling checkpoint 也通过 step/scheduler/有限值审计，SHA-256 为
  `405e8f94a6515260c29d2f2981aad0b620d957954f07cbbfefafa32280ffb57b`，见
  `output/hhc_precleanup_raw12288_corrected_235k_checkpoint_audit.json`。
  240k rolling checkpoint 同样通过审计，SHA-256 为
  `242b6d9f21c9d63068e86306ff9dcf12686a4234df881165a58ebe73c68948e8`；
  对应完整 validation FID 为 `2.377925157546997`，高于 top-10 最差保留
  FID `1.9620590209960938`，因此没有 validation checkpoint 或 pair 声明。rolling
  审计见 `output/hhc_precleanup_raw12288_corrected_240k_checkpoint_audit.json`。
  245k rolling checkpoint 也通过 step/scheduler/有限值审计，SHA-256 为
  `828a99cd85c41b770249eb69557cfc0e03b29e26d4531ca73449586f61f059c4`，见
  `output/hhc_precleanup_raw12288_corrected_245k_checkpoint_audit.json`。
  250k rolling checkpoint 同样通过审计，SHA-256 为
  `2c537265f34476a2e02f5284531ec3b991207f81ac4a791e4d49cce75d7ceb78`；
  对应完整 validation FID 为 `2.464520215988159`，高于 top-10 最差保留
  FID `1.9620590209960938`，因此没有 validation checkpoint 或 pair 声明。rolling
  审计见 `output/hhc_precleanup_raw12288_corrected_250k_checkpoint_audit.json`。
  255k rolling checkpoint 也通过 step/scheduler/有限值审计，SHA-256 为
  `1f85c02722035e1e36972111aa8d0808d3f17d45bf06dc0fc1356db9cec0287c`，见
  `output/hhc_precleanup_raw12288_corrected_255k_checkpoint_audit.json`。
  260k rolling checkpoint 同样通过审计，SHA-256 为
  `36244bf03e5769b7a6864a0056276c36d3a2b6101720014ccf3ea8e047d9065b`；
  对应完整 validation FID 为 `2.4598724842071533`，高于 top-10 最差保留
  FID `1.9620590209960938`，因此没有 validation checkpoint 或 pair 声明。rolling
  审计见 `output/hhc_precleanup_raw12288_corrected_260k_checkpoint_audit.json`。
  265k rolling checkpoint 也通过 step/scheduler/有限值审计，SHA-256 为
  `7764a71995192cd72a05164045bf8d736a0ee2defb90d4d5e8d9839261b44882`，见
  `output/hhc_precleanup_raw12288_corrected_265k_checkpoint_audit.json`。
  270k rolling checkpoint 同样通过审计，SHA-256 为
  `1fdb614ab29a7445b7bfebf1d8baf618a1edc83215b32b7259cd7ea899ebbe0e`；
  对应完整 validation FID 为 `2.8896591663360596`，高于 top-10 最差保留
  FID `1.9620590209960938`，因此没有 validation checkpoint 或 pair 声明。rolling
  审计见 `output/hhc_precleanup_raw12288_corrected_270k_checkpoint_audit.json`。
  275k rolling checkpoint 也通过 global-step、scheduler 与模型/optimizer 全量
  finite 审计，SHA-256 为
  `ec978f51d3662e1ec0f96633589328677c49f98d201c3ca633e3d8fbfe0c639a`；该点仅作
  训练健康证据，下一次完整 validation 在 280k。审计见
  `output/hhc_precleanup_raw12288_corrected_275k_checkpoint_audit.json`。
  280k rolling checkpoint 继续通过相同审计，SHA-256 为
  `936edb88e3a9e6cd09ea85933c880cc7de9bffed1e007fa3bf127125ce8f1547`；完整
  validation FID 为 `2.4878127574920654`，高于 top-10 截止值
  `1.9620590209960938`，因此没有保留 `model-280k` 或 pair 声明。rolling 审计见
  `output/hhc_precleanup_raw12288_corrected_280k_checkpoint_audit.json`。
  285k rolling checkpoint 也通过 global-step、scheduler 以及模型/optimizer
  全量有限值审计，SHA-256 为
  `04da61a5559ae39fbc4ab66633190668e6fb94b5055abe92337099df23b50165`，见
  `output/hhc_precleanup_raw12288_corrected_285k_checkpoint_audit.json`。290k
  rolling checkpoint 继续通过：`global_step=290000`，scheduler 为
  `last_epoch=290000`/`_step_count=290001`，947 个 model 与 1,041 个 optimizer
  浮点 tensor 全部有限；SHA-256 为
  `2722397200ef012091ede182365525fa28f63379f9f7eee2f538c0575475c728`。对应 544 条
  固定 validation FID 为 `2.781623601913452`，高于 top-10 截止值
  `1.9620590209960938`，因此未保留 `model-290k`，也不作 rolling/validation pair
  声明。状态审计见
  `output/hhc_precleanup_raw12288_corrected_290k_checkpoint_audit.json`。295k rolling
  health checkpoint 也通过：`global_step=295000`，scheduler 为
  `last_epoch=295000`/`_step_count=295001`，947 个 model 与 1,041 个 optimizer
  浮点 tensor 全部有限；SHA-256 为
  `a847abcfe4c1cc498a160f4fa8366b7714fb42ee2217b1db14b4ad6c89c55747`，审计见
  `output/hhc_precleanup_raw12288_corrected_295k_checkpoint_audit.json`。正式训练现已
  完成 300,000 updates 和 30 次固定 validation；终点 epoch 3,370、scheduler
  `last_epoch=300000`/`_step_count=300001`，947 个 model 与 1,041 个 optimizer
  浮点 tensor 全部有限，SHA-256 为
  `4d25e9ffbc68810d46d1ba6ecef044e9548d91036cfa8c9d921495e201ba1a8a`。300k 当前
  validation FID 为 `2.5655829906463623`；固定 val 选择仍为 20k 的精确 FID
  `0.78011155128479`，选中 checkpoint SHA-256 为
  `7fcc995cffdd6017e0655df52ad8476f9dfddae1c40e7dfd12b58293deff2fcd`，未使用 test
  metric。终点与选择证据分别为
  `output/hhc_precleanup_raw12288_corrected_300k_checkpoint_audit.json` 和
  `output/hhc_mgt_precleanup_raw12288_selection.json`。原 GPU1 长链 shell 在 trainer
  正常关闭后、selection 前消失；已用
  `tools/resume_after_hhc_precleanup_training_gpu1.sh` 从完成状态幂等恢复，没有重训，
  并曾启动 raw-12288 HHC 公平生成。MGT 排错后该 launcher 已设为 fail-fast，
  selection 与生成结果全部降级为历史诊断，不再进入任何活动队列或最终报告。
  作为设置隔离检查，同一 raw-12288
  VQ、同一 frozen replication-00、同一官方 evaluator 和显式
  `sample_intra_cas/50/0.5/3.5/1.0` 下，大清理前 Git 默认 280k MGT 得到 FID
  `0.18686`，另一最新完整 50k MGT 得到 `0.29113`。这证明 VQ、50-step exporter 和 evaluator 链路
  本身能复现低 FID；当前差异定位在 corrected paper-architecture MGT 的训练轨迹，
  而不是用 sampler 参数解释。两条旧 MGT 仅为单 replication 诊断，不能并入最终
  20-replication 主表。错误重构 MGT 已完成的 replication 0 诊断为：FID
  `0.64536`、Top-1/2/3
  `0.40094/0.58491/0.69458`、Matching `3.84369`、Diversity
  `8.93247`、官方 MModality `0.40619`。相对上述最新完整 50k 旧 MGT 的同请求、
  同 seed 诊断，FID 增加 `0.35422`；这进一步确认差异出现在 MGT 训练结果而非
  导出协议。10 cm 物理诊断为 contact F1 `0.55453`、precision `0.57244`、
  recall `0.53772`、temporal IoU `0.38364`、false-contact rate `0.15318`；旧
  50k MGT 对应为 `0.56438/0.52564/0.60927/0.39312/0.20970`，即新模型减少
  false contact 并提高 precision，但以 recall 为代价。上述仍只是 replication-0
  配对诊断，不能替代正在运行的 20-replication 均值与置信区间。逐字节 request
  配对和全部 effect size 已冻结在
  `output/fair_eval/paired_hhc_mgt_training_state_sensitivity_rep0.json`，最终结果构建器
  将其作为必需诊断证据读取，但不会把单次 Wilcoxon 数值解释为显著性结论。
  这里 W&B 的 `val/FID` 是每个 milestone 的当前值；`val/Diversity`、
  `val/TOP1/2/3` 与 `val/Matching` 是历史 best-so-far，因此后几次记录相同并不
  表示生成结果被复用。正式 checkpoint selection 只读取当前 milestone FID。
  进一步回查清理前 MMM 后发现，失败的重训并未使用原 MGT token cache：当前缓存
  显式以 `Interx_Mean/Std` 重建，而清理前 tokenizer 的默认 normalizer 实际绑定
  `Interx_Mean_root/Std_root`，且随机 crop 已冻结在原 11,370 条缓存中。同名 token
  文件内容并不一致。修复流程因此先从原 MMM byte-exact 导入官方 train+val 的
  9,664 个 token/contact 对，并只用 9,095 个 train 项重建 contact map；源目录中
  的 test token 不复制、不加载、不参与训练。新 MGT 在 10k/20k 固定 validation
  FID 通过门槛前，所有正式 HHC 下游实验保持暂停。该 exact-cache 修复在
  `10k` 得到精确 decoded FID `1.3668916225`，checkpoint 的模型、优化器与
  scheduler 数值审计均通过，但结果仍明显差于历史 `0.697935@10k`，因此 cache
  差异是真实问题但不是唯一原因；该长训已在 10k 安全停止。下一项只切换
  reconstructed paper topology 为清理前 MMM legacy topology，保持
  `512-D/8-head/batch-96`、exact cache、论文 mask/loss、优化器和共同 50-step
  validation 不变，以隔离结构设置。该 matched control 在 10k 得到 FID
  `1.265656`，checkpoint 数值审计通过，但仍远差于历史 `0.697935@10k`；因此
  legacy topology 单项也不能恢复，已在 10k 停止。下一步是完整复现历史
  `1024-D/16-head/batch-64` 的 known-good 训练合同：旧 random replacement/flat
  mask、all-valid-token loss、AdamW `betas=(0.5,0.9)`、150k MultiStepLR、
  seed `114514`，以及清理前真实 conditioning（word token 不做 text dropout，
  global CLIP 与 shape 各自独立按 `0.2` dropout）。Local History 又确认了两个
  此前漏掉的训练设置：原训练 token 窗口是 38 而不是当前论文路径的 50；训练前
  会先在完整 test loader 上运行一次 53-batch sanity FID，且不恢复该评测消耗的
  RNG。训练期 validation 继续使用同一 historical test-side-effect 合同。V2
  控制仍使用 50-token/隔离 validation RNG，约 3,138 updates 后作为不完整诊断安全
  停止，不保留可复用 checkpoint。V3 加入 38-token、test sanity 和 RNG 副作用，
  但 0-step FID 为 `16.04839`，未复现历史 `25.7493`，因此在不足 300 updates 时
  按门禁停止。继续逐函数比较后确认 V3 仍误用了重构后的正式 sampler：历史实现
  每一轮会重预测当前通道的全部有效 token，而正式实现只让仍 masked 的 token
  竞争，两者不是超参数差异。V4 将 time-bracketed 旧实现隔离为
  `sample_intra_cas_historical`，并与正式 sampler 分开测试和记录；它首先只对齐
  0-step FID。V4 实测 `25.74927`，与历史日志四舍五入后的 `25.7493` 精确一致，
  因此确认剩余差异确实来自 sampler 实现合同，而非 VQ、初始化、test loader 或
  evaluator。独立行为审计还在同一小模型状态、输入和 RNG 下证明旧源码与隔离
  实现的 forward logits、最终 sampled tokens 和采样后 RNG 均逐位一致，并证明
  正式 sampler 仍是独立 dispatch；证据为
  `output/historical_mgt_sampler_behavior_audit.json`。V4 的 5k rolling
  checkpoint 曾落盘并通过全状态审计：`global_step=5000`，模型与 AdamW
  optimizer 的全部浮点状态均有限，实际参数组为 `betas=(0.5,0.9)`、
  `lr=2e-4`、`weight_decay=1e-6`，MultiStepLR 的 `last_epoch=5000`；证据为
  `output/historical_mgt_v4_5k_checkpoint_audit.json`。但进一步逐式核对发现，V4
  的 historical all-valid loss 将旧源码的“token 先乘各样本权重、再作一次
  batch-wide sum”改写成了数学等价的“各样本先求和、再 mean”；不等长 batch
  实测 loss 相差 `4.77e-7` 且梯度不逐位一致。V4 因而在约 5.9k 安全停止，保留
  checkpoint 作为诊断 provenance，但由 `aborted_diagnostic.json` 明确禁止恢复或
  报告。V5 恢复旧归约顺序，完整一致性测试通过后从头启动；下一硬门槛仍为
  0-step、5k 状态及 10k/20k 历史 FID。V5 的 0-step FID 已再次得到
  `25.74927`，与旧日志精确一致。增强后的旧 200k checkpoint 审计随后又暴露出
  optimizer membership 差异：旧 AdamW 参数组为 482 个 parameter tensors，
  V5 因显式冻结并过滤 VQ 只有 344 个；旧/新真正获得 Adam state 的都是 344，
  说明其余 138 个虽因 detach 不更新，仍属于历史 checkpoint 合同。V5 因此在
  742 updates 停止并标记不可恢复。V6 恢复了 482 参数 membership，但进一步的
  本地 W&B 逐 step 对比暴露出另一个明确运行态错误：重构代码在 sanity
  validation 前显式调用了 `trans_encoder.eval()`；Lightning 会逐子模块恢复
  sanity 前的 mode，而 fit loop 随后不会再把该 child 切回 train，因此 V6 训练期
  Transformer dropout 始终关闭。V6 的 `Train/all` 为 `7.631497@999`、
  `5.958329@1999`，旧轨迹为 `7.760624/6.512545`；它在 logged step 4049
  停止并标记不可恢复，证据为
  `output/historical_mgt_v6_training_trajectory_comparison.json` 及该 run 的
  `aborted_diagnostic.json`。V7 已移除提前 eval，并在 `on_train_start` 强制检查
  `trans_encoder=train、VQ=eval`；启动摘要确认该 mode 与旧版
  `322M trainable/151M non-trainable/473M total`。V7 的 0-step FID 再次精确为
  `25.74927`，step 49/499/849/999/1499/1999 的
  `Train/all/every_token/masked/no_masked` 四项指标也与 2 月 13 日 W&B
  transaction log 逐位一致（包括 `Train/all=7.760623931884766@999`、
  `6.512544631958008@1999`）。零容差的 24 数值门禁已冻结为
  `output/historical_mgt_v7_training_trajectory_2k_audit.json`，状态为 `pass`；
  后续 milestone 另写 live comparison。独立 Lightning mode/AST 审计
  `output/historical_mgt_module_mode_contract_audit.json` 已通过。V7 现重新承担
  5k/10k/20k 门禁。V7 的 4999-step 四项训练量仍与旧 W&B 逐位一致；
  `latest-000005000.ckpt` 的全状态审计亦通过：`global_step=5000`、482 个
  optimizer parameter tensors、344 个 Adam state entries、
  `betas=(0.5,0.9)`、`lr=2e-4`、`weight_decay=1e-6`、scheduler
  `last_epoch=5000`，模型及 optimizer 全部浮点张量有限。证据分别为
  `output/historical_mgt_v7_training_trajectory_5k_audit.json` 和
  `output/historical_mgt_v7_5k_checkpoint_audit.json`。精确
  V7 的 10k 门槛也已通过：step 9999 的四项训练量与旧 W&B 逐位一致，
  historical ordinary validation 的 FID `0.6979351043701172`、Diversity
  `8.949728012084961`、Top-1/2/3 `0.383844/0.557193/0.65625`、Matching
  `3.970064878463745` 六项均为零误差。validation 前 rolling checkpoint 与
  validation 后 FID checkpoint 的 931 个 model tensors、optimizer、scheduler
  和 hyperparameters 也逐位一致；仅 callback/loop 元数据按预期记录了评测。
  证据为 `output/historical_mgt_v7_training_trajectory_10k_audit.json`、
  `output/historical_mgt_v7_validation_10k_audit.json`、
  `output/historical_mgt_v7_10k_prevalidation_checkpoint_audit.json` 和
  `output/historical_mgt_v7_10k_pre_post_validation_pair_audit.json`。
  20k 门槛现也通过：截至 candidate step 20149 的 403 个 W&B 训练记录点、
  以及 10k/20k 两个 milestone 的全部 12 个 ordinary validation 数值均与旧 run
  严格零误差。20k 的 FID/Diversity/Top-1/2/3/Matching 为
  `0.6423928737640381/8.981422424316406/0.3838443458080292/`
  `0.557193398475647/0.65625/3.9059505462646484`。20k rolling checkpoint
  仍为 931 个 model tensors、482 个 optimizer parameters、344 个 Adam state
  entries，scheduler `last_epoch=20000`；validation 前后 model、optimizer、
  scheduler 与 hyperparameters 逐位一致。可重跑入口为
  `tools/audit_mgt_repair_v7_20k.sh`，证据为
  `output/historical_mgt_v7_{training_trajectory,validation}_20k_audit.json`、
  `output/historical_mgt_v7_20k_prevalidation_checkpoint_audit.json` 和
  `output/historical_mgt_v7_20k_pre_post_validation_pair_audit.json`。
  这证明提前 `trans_encoder.eval()` 是已实证的训练轨迹/FID 错误，而非参数猜测；
  60k milestone 核对已通过：截至 candidate step 60149 的 1,203 个训练记录点
  最大 loss 误差仍为 0，10k/20k/30k/40k/50k/60k 全部六项 validation 指标也保持
  逐值零误差；30k/40k/50k ordinary FID 分别为
  `0.47360625863075256`、`0.3779594898223877` 和
  `0.24427056312561035`，60k FID/Diversity/Top-1/2/3/Matching 为
  `0.3720308244228363/8.934673309326172/0.40566039085388184/`
  `0.5896226167678833/0.6922169923782349/3.716461181640625`。50k validation 前后的 931 个 model tensors、optimizer、
  scheduler 与 hyperparameters 逐位一致，只有 callback/loop 元数据按预期记录了
  新的 best FID；60k rolling checkpoint 另通过全状态有限性审计，scheduler
  `last_epoch=60000`、482 个 optimizer parameter tensors 和 344 个 Adam state
  entries 均正确。在线证据为
  `output/historical_mgt_v7_training_trajectory_all_rows_live.json` 和
  `output/historical_mgt_v7_validation_60k_live.json`、
  `output/historical_mgt_v7_60k_checkpoint_audit.json` 以及
  `output/historical_mgt_v7_50k_pre_post_validation_pair_audit.json`；V7 继续承担
  完整 300k 门禁。最新 100k 门槛继续通过：当前 W&B 中 2,002 个可比训练记录点
  的 loss 最大绝对差为 0，10k 至 100k 十个 validation milestone 的
  FID/Diversity/Top-1/2/3/Matching 共 60 个值也全部逐值相等。100k 指标为
  `FID=0.3369324803352356`、`Diversity=8.902233123779297`、
  `Top-1/2/3=0.41568395495414734/0.604363203048706/0.7146226167678833`、
  `Matching=3.596099376678467`。进入候选集的 post-validation checkpoint
  全状态有限，validation 前后 931 个模型张量、optimizer 和 scheduler 逐位相等。
  证据为 `output/historical_mgt_v7_training_trajectory_100k_live.json`、
  `output/historical_mgt_v7_validation_100k_live.json`、
  `output/historical_mgt_v7_100k_postvalidation_checkpoint_audit.json` 和
  `output/historical_mgt_v7_100k_pre_post_validation_pair_audit.json`。
  最新 120k 门槛也通过：当前可比的 2,410 个训练记录点截至 candidate step
  120499 的最大 loss 误差仍为 0；10k 至 120k 的 72 个 validation 数值全部与
  旧 W&B 逐值相等。120k ordinary FID 为 `0.20432139933109283`。120k rolling
  checkpoint 的 931 个模型张量和 1,032 个 optimizer state 张量均有限，optimizer
  参数组仍为 482 个 tensor、344 个 Adam state entries，scheduler
  `last_epoch=120000`；validation 前后的模型、optimizer、scheduler 和
  hyperparameters 逐位相等。证据为
  `output/historical_mgt_v7_training_trajectory_all_rows_live.json`、
  `output/historical_mgt_v7_validation_120k_live.json`、
  `output/historical_mgt_v7_120k_prevalidation_checkpoint_audit.json` 和
  `output/historical_mgt_v7_120k_pre_post_validation_pair_audit.json`。
  130k milestone 随后继续以零容差通过：10k 至 130k 的 78 个 validation 数值
  全部相等，130k ordinary FID 为 `0.21991795301437378`；rolling checkpoint
  全状态有限，validation 前后模型、optimizer、scheduler 与 hyperparameters
  逐位相等。对应证据为 `output/historical_mgt_v7_validation_130k_live.json`、
  `output/historical_mgt_v7_130k_prevalidation_checkpoint_audit.json` 和
  `output/historical_mgt_v7_130k_pre_post_validation_pair_audit.json`。
  140k milestone 同样零容差通过：10k 至 140k 的 84 个 validation 数值全部
  相等，ordinary FID 为 `0.3215497136116028`；140k rolling checkpoint 仍为
  931 个有限模型张量、482 个 optimizer parameters、344 个 Adam state，且
  validation 前后的模型、optimizer、scheduler 与 hyperparameters 逐位相等。
  证据为 `output/historical_mgt_v7_validation_140k_live.json`、
  `output/historical_mgt_v7_140k_prevalidation_checkpoint_audit.json` 和
  `output/historical_mgt_v7_140k_pre_post_validation_pair_audit.json`。
  150k 与 160k 继续通过同一零容差门槛：150k ordinary FID 为
  `0.2316998988389969`，160k ordinary FID 为 `0.14545394480228424`；10k 至
  160k 的 16 个 milestone、每个 6 项指标共 96 个数值全部与大清理前 W&B
  transaction log 逐值相等。截至 candidate step 160249 的 3,205 条训练记录
  loss 最大绝对误差仍为 0。160k rolling checkpoint 的 SHA-256 为
  `91a53ad1c0f70bc939cb47cc9f92af974898028478a3ddc0ecb6a408ea27abb5`，
  scheduler 已按历史合同越过 150k milestone 并处于 `lr=1e-5`；931 个模型
  tensor、482 个 optimizer parameter tensor 和 344 个 Adam state entry 均通过
  有限值审计。validation 前后模型、optimizer、scheduler 与 hyperparameters
  逐位相等。证据为
  `output/historical_mgt_v7_training_trajectory_all_rows_live.json`、
  `output/historical_mgt_v7_validation_160k_live.json`、
  `output/historical_mgt_v7_160k_prevalidation_checkpoint_audit.json` 和
  `output/historical_mgt_v7_160k_pre_post_validation_pair_audit.json`。
  170k ordinary validation 也以零容差通过，FID 为
  `0.1580042988061905`；10k 至 170k 的 102 个 validation 数值全部与历史
  transaction log 相等，截至 candidate step 170199 的 3,404 条训练记录最大
  loss 误差仍为 0。170k pre-validation checkpoint SHA-256 为
  `028229df9747a0249550f84658bf55823a9068caa1fe6ad08206c491c7fd3a6d`，
  全部模型/optimizer 状态有限且 scheduler/LR 正确；post-validation 文件中的
  模型、optimizer、scheduler 与 hyperparameters 逐位不变，best checkpoint
  仍为 160k 的 `0.14545394480228424`。证据为
  `output/historical_mgt_v7_validation_170k_live.json`、
  `output/historical_mgt_v7_170k_prevalidation_checkpoint_audit.json` 和
  `output/historical_mgt_v7_170k_pre_post_validation_pair_audit.json`。
  180k milestone 继续精确复现并产生新的历史 best：ordinary FID 为
  `0.138837531208992`，10k 至 180k 共 108 个 validation 数值全部零误差；截至
  candidate step 180099 的 3,602 条训练记录最大 loss 误差同样为 0。180k
  pre-validation checkpoint SHA-256 为
  `6f334565414bf1aae92c8ed8d4f65e56d08195c9e6e48ef998e34f74c0ebeaa2`，
  模型与 optimizer 全状态有限、LR 仍为 `1e-5`。validation 后 best path 正确
  从 160k 更新到 180k，但模型、optimizer、scheduler 与 hyperparameters
  逐位不变。证据为 `output/historical_mgt_v7_validation_180k_live.json`、
  `output/historical_mgt_v7_180k_prevalidation_checkpoint_audit.json` 和
  `output/historical_mgt_v7_180k_pre_post_validation_pair_audit.json`。
  185k rolling checkpoint 也在原子落盘后通过完整状态审计：global step
  `185000`、epoch `1380`，scheduler 为 `last_epoch=185000`/
  `_step_count=185001`，931 个模型 tensor 与 1,032 个 optimizer tensor 全部
  finite，LR/weight decay 为 `1e-5/1e-6`；SHA-256 为
  `85aa27bb6bc12a3dad980e921a126a3afcb4164e3f754b1455f77a53e63f6b89`。
  截至 candidate step 184949 的 3,699 条训练记录继续保持最大 loss 误差 0。
  rolling 证据为 `output/historical_mgt_v7_185k_checkpoint_audit.json`；该节点
  不执行 validation，因此不报告或推断新的 FID。
  190k validation 继续零容差复现：当前 ordinary FID 精确为
  `0.15922923386096954`，10k 至 190k 共 114 个 validation 数值最大差异 0；截至
  candidate step 190099 的 3,802 条训练记录最大 loss 误差也为 0。pre-validation
  checkpoint SHA-256 为
  `0c8a41f2777efe0aaede5550ae3d3dbed05577f86d68a0f0c8dc6c19a9343263`，
  post-validation SHA-256 为
  `d9711889e58b8a43b0ff28a70cbd8a6e780b0a2879320bbdb5eb89e8383ecc9b`；
  模型、optimizer、scheduler 与 hyperparameters 逐位相同，callback 的当前 FID
  更新为 190k 数值而 best 正确保留 180k 的 `0.138837531208992`。证据为
  `output/historical_mgt_v7_validation_190k_live.json`、
  `output/historical_mgt_v7_190k_prevalidation_checkpoint_audit.json` 和
  `output/historical_mgt_v7_190k_pre_post_validation_pair_audit.json`。
  195k rolling checkpoint 也通过完整状态审计：global step `195000`、epoch
  `1455`，scheduler 为 `last_epoch=195000`/`_step_count=195001`，931 个模型
  tensor 与 1,032 个 optimizer tensor 全部 finite，LR/weight decay 仍为
  `1e-5/1e-6`；SHA-256 为
  `9df5cfe0e7d99fc548c6c8b3d0b0f7e5f2bd47607f6a8f3b00646f23abb0da8d`。
  logger flush 后，截至 candidate step 195049 的 3,901 条训练记录继续保持最大
  loss 误差 0。证据为 `output/historical_mgt_v7_195k_checkpoint_audit.json`；
  rolling 节点不执行 validation，因此不产生 FID 行。
  200k validation 再次产生精确历史新 best：ordinary FID 为
  `0.13571932911872864`，10k 至 200k 共 120 个 validation 数值最大差异 0；截至
  candidate step 200049 的 4,001 条训练记录最大 loss 误差也为 0。pre-validation
  checkpoint SHA-256 为
  `5a6231992124da4e5c8d3581d91ef6dcc144499f1d3f157818cca91806400f4e`，
  post-validation SHA-256 为
  `6198934de1df6fd7be41da03d6f2df2ff67606b483b5836bc2165f6c7b2721c6`；
  模型、optimizer、scheduler 与 hyperparameters 逐位相同，callback best 从 180k
  的 `0.138837531208992` 正确更新为 200k。证据为
  `output/historical_mgt_v7_validation_200k_live.json`、
  `output/historical_mgt_v7_200k_prevalidation_checkpoint_audit.json` 和
  `output/historical_mgt_v7_200k_pre_post_validation_pair_audit.json`。
  205k rolling checkpoint 继续通过完整状态审计：global step `205000`、epoch
  `1529`，scheduler 为 `last_epoch=205000`/`_step_count=205001`，931 个模型
  tensor 与 1,032 个 optimizer tensor 全部 finite，LR/weight decay 为
  `1e-5/1e-6`；SHA-256 为
  `48282e7bd31acf367b995ec57705fc7f58fa8029dca4b7e56a31d3e2119fa304`。
  截至 candidate step 204949 的 4,099 条训练记录最大 loss 误差仍为 0。证据为
  `output/historical_mgt_v7_205k_checkpoint_audit.json`；该 rolling 节点不执行
  validation。
  210k validation 继续精确回放：当前 ordinary FID 为
  `0.16243889927864075`，10k 至 210k 共 126 个 validation 数值最大差异 0；截至
  candidate step 210099 的 4,202 条训练记录最大 loss 误差仍为 0。pre-validation
  checkpoint SHA-256 为
  `c1ee8aeffff4d4d3ef146aa921942c1d21e7df3cdacbe4f10816f56f3ca27564`，
  post-validation SHA-256 为
  `e06f06efbae42af79b76d7e9ccabc8e4fcb8e89b04c82223be369c3c3d5ee38c`；
  模型、optimizer、scheduler 与 hyperparameters 逐位相同，best 正确保留 200k
  的 `0.13571932911872864`。证据为
  `output/historical_mgt_v7_validation_210k_live.json`、
  `output/historical_mgt_v7_210k_prevalidation_checkpoint_audit.json` 和
  `output/historical_mgt_v7_210k_pre_post_validation_pair_audit.json`。
  215k rolling checkpoint 继续通过完整状态审计：global step `215000`、epoch
  `1604`，scheduler 为 `last_epoch=215000`/`_step_count=215001`，931 个模型
  tensor 与 1,032 个 optimizer tensor 全部 finite，LR/weight decay 为
  `1e-5/1e-6`；SHA-256 为
  `20a6a9c5ad8d051c3f71d9c387660f3bb7db11885e29b54aac9c0721a60535df`。
  截至 candidate step 215099 的 4,302 条训练记录最大 loss 误差仍为 0。证据为
  `output/historical_mgt_v7_215k_checkpoint_audit.json`；该 rolling 节点不执行
  validation。
  220k validation 继续精确回放：当前 ordinary FID 为
  `0.1590881049633026`，10k 至 220k 共 132 个 validation 数值最大差异 0；截至
  candidate step 220349 的 4,407 条训练记录最大 loss 误差仍为 0。pre-validation
  checkpoint SHA-256 为
  `9cb46d66ca547bf474cc9fd8c2b9694a9fb0be93bd22c17bffb84b985571449d`，
  post-validation SHA-256 为
  `37d23995901418b33f548feba441624d63457d7c2c4aafffbcb90b50e069e7b0`；
  模型、optimizer、scheduler 与 hyperparameters 逐位相同，best 正确保留 200k
  的 `0.13571932911872864`。证据为
  `output/historical_mgt_v7_validation_220k_live.json`、
  `output/historical_mgt_v7_220k_prevalidation_checkpoint_audit.json` 和
  `output/historical_mgt_v7_220k_pre_post_validation_pair_audit.json`。
  225k rolling checkpoint 继续通过完整状态审计：global step `225000`、epoch
  `1679`，scheduler 为 `last_epoch=225000`/`_step_count=225001`，931 个模型
  tensor 与 1,032 个 optimizer tensor 全部 finite，LR/weight decay 为
  `1e-5/1e-6`；SHA-256 为
  `c981c44162246feb6ce8c8cfc88b312eabb93513653769d4bf6609b58b936721`。
  截至 candidate step 225149 的 4,503 条训练记录最大 loss 误差仍为 0。证据为
  `output/historical_mgt_v7_225k_checkpoint_audit.json`；该 rolling 节点不执行
  validation。
  230k validation 继续精确回放：当前 ordinary FID 为
  `0.15852901339530945`，10k 至 230k 共 138 个 validation 数值最大差异 0；截至
  candidate step 230299 的 4,606 条训练记录最大 loss 误差仍为 0。pre-validation
  checkpoint SHA-256 为
  `44663f6b0d743d9318a171a6a6292d7b6211a6a4efaa34be7c2333cf8d454d09`，
  post-validation SHA-256 为
  `1d918979e5b60242e7aaac175e6d5161959b559526ccfec7eb4061ff18d9921a`；
  模型、optimizer、scheduler 与 hyperparameters 逐位相同，best 正确保留 200k
  的 `0.13571932911872864`。证据为
  `output/historical_mgt_v7_validation_230k_live.json`、
  `output/historical_mgt_v7_230k_prevalidation_checkpoint_audit.json` 和
  `output/historical_mgt_v7_230k_pre_post_validation_pair_audit.json`。
  235k rolling checkpoint 继续通过完整状态审计：global step `235000`、epoch
  `1753`，scheduler 为 `last_epoch=235000`/`_step_count=235001`，931 个模型
  tensor 与 1,032 个 optimizer tensor 全部 finite，LR/weight decay 为
  `1e-5/1e-6`；SHA-256 为
  `39db17295056ccd6745c4bdebdd563120c2b8292a7d1f54e35c1c5854d723930`。
  截至 candidate step 234899 的 4,698 条训练记录最大 loss 误差仍为 0。证据为
  `output/historical_mgt_v7_235k_checkpoint_audit.json`；该 rolling 节点不执行
  validation。
  240k validation 继续精确回放：当前 ordinary FID 为
  `0.14474517107009888`，10k 至 240k 共 144 个 validation 数值最大差异 0；截至
  candidate step 240049 的 4,801 条训练记录最大 loss 误差仍为 0。pre-validation
  checkpoint SHA-256 为
  `3d9e0def8efb7206b816618f2a7a66189a1754145e75d822c653136a928dd1c7`，
  post-validation SHA-256 为
  `8bee23c01b6aed841369a07f86b44683f1c72569db0f191f028f0f1e5a91bec2`；
  模型、optimizer、scheduler 与 hyperparameters 逐位相同，best 正确保留 200k
  的 `0.13571932911872864`。证据为
  `output/historical_mgt_v7_validation_240k_live.json`、
  `output/historical_mgt_v7_240k_prevalidation_checkpoint_audit.json` 和
  `output/historical_mgt_v7_240k_pre_post_validation_pair_audit.json`。
  245k rolling checkpoint 继续通过完整状态审计：global step `245000`、epoch
  `1828`，scheduler 为 `last_epoch=245000`/`_step_count=245001`，931 个模型
  tensor 与 1,032 个 optimizer tensor 全部 finite，LR/weight decay 为
  `1e-5/1e-6`；SHA-256 为
  `53425216ecc403526a5cd40ad339e63c7c2051c2a800e701a39245ddfbe65ed9`。
  截至 candidate step 245099 的 4,902 条训练记录最大 loss 误差仍为 0。证据为
  `output/historical_mgt_v7_245k_checkpoint_audit.json`；该 rolling 节点不执行
  validation。
  250k validation 继续精确回放：当前 ordinary FID 为
  `0.16886506974697113`，10k 至 250k 共 150 个 validation 数值最大差异 0；截至
  candidate step 250249 的 5,005 条训练记录最大 loss 误差仍为 0。pre-validation
  checkpoint SHA-256 为
  `a7576771014df24219280978cedc1cfb657630ac93bf8f7e85d019c3dd0bab24`，
  post-validation SHA-256 为
  `fae4dc63b186618246d5c1c48762f276a596acfe1169285a217dda274fbc3e87`；
  模型、optimizer、scheduler 与 hyperparameters 逐位相同，best 正确保留 200k
  的 `0.13571932911872864`。证据为
  `output/historical_mgt_v7_validation_250k_live.json`、
  `output/historical_mgt_v7_250k_prevalidation_checkpoint_audit.json` 和
  `output/historical_mgt_v7_250k_pre_post_validation_pair_audit.json`。
  255k rolling checkpoint 继续通过完整状态审计：global step `255000`、epoch
  `1902`，scheduler 为 `last_epoch=255000`/`_step_count=255001`，931 个模型
  tensor 与 1,032 个 optimizer tensor 全部 finite，LR/weight decay 为
  `1e-5/1e-6`；SHA-256 为
  `7191d81f6b6a3ac719b939d25d748462a71495a03d466cf9a1915caba6cb5b42`。
  截至 candidate step 254949 的 5,099 条训练记录最大 loss 误差仍为 0。证据为
  `output/historical_mgt_v7_255k_checkpoint_audit.json`；该 rolling 节点不执行
  validation。
  260k validation 继续精确回放：当前 ordinary FID 为
  `0.1608208417892456`，10k 至 260k 共 156 个 validation 数值最大差异 0；截至
  candidate step 260349 的 5,207 条训练记录最大 loss 误差仍为 0。pre-validation
  checkpoint SHA-256 为
  `a1503c3592f0c8810f7043e6321bac9f01879d569d253c4d5aefee5078efef01`，
  post-validation SHA-256 为
  `de2dbdb5213b7aac9ac9425f2c8480b5417eb879934a84ee1dab227d50429e4b`；
  模型、optimizer、scheduler 与 hyperparameters 逐位相同，best 正确保留 200k
  的 `0.13571932911872864`。证据为
  `output/historical_mgt_v7_validation_260k_live.json`、
  `output/historical_mgt_v7_260k_prevalidation_checkpoint_audit.json` 和
  `output/historical_mgt_v7_260k_pre_post_validation_pair_audit.json`。
  265k rolling checkpoint 继续通过完整状态审计：global step `265000`、epoch
  `1977`，scheduler 为 `last_epoch=265000`/`_step_count=265001`，931 个模型
  tensor 与 1,032 个 optimizer tensor 全部 finite，LR/weight decay 为
  `1e-5/1e-6`；SHA-256 为
  `c05d8fe50e8aa244ebb43eeaf7a315672dde8066e4b4b21c063447b7903de1df`。
  截至 candidate step 265199 的 5,304 条训练记录最大 loss 误差仍为 0。证据为
  `output/historical_mgt_v7_265k_checkpoint_audit.json`；该 rolling 节点不执行
  validation。
  270k validation 继续精确回放：当前 ordinary FID 为
  `0.17415158450603485`，10k 至 270k 共 162 个 validation 数值最大差异 0；截至
  candidate step 270949 的 5,419 条训练记录最大 loss 误差仍为 0。pre-validation
  checkpoint SHA-256 为
  `1b01373df9eb623d46502e6a69ce679476d52fb807827eafd8bcaa20a7461cac`。
  本次 FID 在截至 270k 的精确排名为第 12，未进入 `save_top_k=10`，所以 Lightning
  正确地不生成 `model-000270000-*.ckpt`；实际保留的 10 个 step 与精确 FID 前十
  完全一致。故本节点不虚构 pre/post checkpoint pair，证据为
  `output/historical_mgt_v7_validation_270k_live.json`、
  `output/historical_mgt_v7_270k_prevalidation_checkpoint_audit.json` 和
  `output/historical_mgt_v7_270k_topk_retention_audit.json`。
  275k rolling checkpoint 也通过完整状态审计：global step `275000`、epoch
  `2052`，scheduler 为 `last_epoch=275000`/`_step_count=275001`，931 个模型
  tensor 与 1,032 个 optimizer tensor 全部 finite，LR/weight decay 为
  `1e-5/1e-6`；SHA-256 为
  `2da4b5930fdb6eeb57f900bd7119d17807800eadd29e0e7b3f532368a6cd2ad6`。
  截至 candidate step 274949 的 5,499 条训练记录与旧 W&B 的最大 loss 误差仍为
  0。证据为 `output/historical_mgt_v7_275k_checkpoint_audit.json`；该 rolling
  节点不执行 validation。
  280k validation 同样精确回放：ordinary FID 为
  `0.16047221422195435`，Diversity/TOP1/TOP2/TOP3/Matching 连同此前全部验证点
  共 168 个数值的最大差异为 0；截至 candidate step 280099 的 5,602 条训练记录
  最大 loss 误差也为 0。该 FID 按预期进入 top-10 并保留
  `model-000280000-0.1605.ckpt`，SHA-256 为
  `07865f93733d432da1bd01e2eff967640b9e1fd054881c5c17fb70fc6e8bb8ba`。
  它与大清理前原始 280k checkpoint 的 931 个模型 tensor、optimizer 和 scheduler
  逐位相同；与同一步 pre-validation rolling checkpoint 相比，模型、optimizer、
  scheduler 与 hyperparameters 也逐位相同，仅 callback/loop 元数据记录本次验证，
  best 仍为 200k 的 `0.13571932911872864`。证据为
  `output/historical_mgt_v7_validation_280k_live.json`、
  `output/historical_mgt_v7_280k_checkpoint_audit.json`、
  `output/historical_mgt_v7_step280k_checkpoint_pair_audit_live.json` 和
  `output/historical_mgt_v7_280k_pre_post_validation_pair_audit.json`。
  285k rolling checkpoint 继续通过完整状态审计：global step `285000`、epoch
  `2126`，scheduler 为 `last_epoch=285000`/`_step_count=285001`，931 个模型
  tensor 与 1,032 个 optimizer tensor 全部 finite，LR/weight decay 为
  `1e-5/1e-6`；SHA-256 为
  `219d9fcbdd6a9832322d7a11316f587275d8d1dc6e6d344990e6978fdeb0b7b0`。
  截至 candidate step 284949 的 5,699 条训练记录最大 loss 误差仍为 0。证据为
  `output/historical_mgt_v7_285k_checkpoint_audit.json`；该 rolling 节点不执行
  validation。
  290k validation 继续逐值相等：ordinary FID 为
  `0.19474996626377106`，截至该点 174 个 validation 数值最大差异为 0；截至
  candidate step 290149 的 5,803 条训练记录最大 loss 误差也为 0。pre-validation
  checkpoint SHA-256 为
  `681c918a73f7bc3c9e08e2e2152ac886536eedb9ab8706358ccb437a46a5ffa2`。
  该 FID 精确排名为第 14，正确地未生成 `model-000290000-*.ckpt`；实际保留的十个
  checkpoint 与精确 FID 前十完全一致。证据为
  `output/historical_mgt_v7_validation_290k_live.json`、
  `output/historical_mgt_v7_290k_prevalidation_checkpoint_audit.json` 和
  `output/historical_mgt_v7_290k_topk_retention_audit.json`。
  295k rolling checkpoint 通过最后一个预终点完整状态审计：global step
  `295000`、epoch `2201`，scheduler 为
  `last_epoch=295000`/`_step_count=295001`，931 个模型 tensor 与 1,032 个
  optimizer tensor 全部 finite，LR/weight decay 为 `1e-5/1e-6`；SHA-256 为
  `0701d1726a5dc1f2cdfe7e4d6576b606747811cf57ac648b0243f267f96d1fae`。
  截至 candidate step 294949 的 5,899 条训练记录最大 loss 误差仍为 0。证据为
  `output/historical_mgt_v7_295k_checkpoint_audit.json`；该 rolling 节点不执行
  validation。
  V7 已正常结束于 300k：最终 `latest-000300000.ckpt` SHA-256 为
  `8d834f66...9cf44a`，6,000/6,000 条训练记录 loss 最大误差为 0，30 次
  validation 的 180 个数值最大误差也为 0；300k diagnostic FID 与旧 W&B
  同为 `0.18090814352035522`。终态证据为
  `output/historical_mgt_v7_training_trajectory_300k_audit.json` 和
  `output/historical_mgt_v7_validation_300k_audit.json`。
  loss/gradient 证据冻结为
  `output/historical_mgt_loss_reduction_audit.json`，并已纳入总历史合同审计。
  最终 common-50 bridge 另强制绑定 evaluator runtime：历史参考与 V7 候选都必须
  使用同一个 CPU official evaluator runtime，且该 runtime 必须与 repair gate 中
  VQ reconstruction 的 runtime 字典完全一致；任一 CUDA 结果、runtime 缺失或版本
  漂移都会 fail closed，不能放行正式 HHC。历史参考 rep00 的 CPU 预检重评分 FID
  为 `0.1869647079683432`，这里只作为 bridge 输入检查，不冒充最终 20 次汇总。
  正式 HHC 与 shape-augmentation MGT 队列现均先执行
  `tools/require_mgt_repair_gate.py`。在
  `output/mgt_repair_gate.json` 尚未由完整 300k 历史重放、普通 FID milestone
  复现和同 checkpoint 的官方 common-50（1696 条）桥接共同生成前，队列会
  fail-fast，不能因重启或误调用越过 MGT 修复阶段。该门槛固定使用
  raw12288/175k tokenizer 和 full-space ordinary evaluator；另要求 280k
  旧/新 checkpoint 的 model/optimizer/scheduler 逐位一致，并将 common-50
  两侧 HDF5 用同一当前 scorer 重新评分。第一次 bridge 的候选 FID
  `0.2267311238` 与参考 `0.1869646679` 不一致，已经定位为比较了不同的 CFG
  conditioning：历史训练必须使用 `historical_mmm_shape_dropout`，而清理前
  common-50 导出实际使用 `paper_text_cfg`。这不是 checkpoint 误差。exporter
  现将 training/inference conditioning 分开记录，并只在 checkpoint 精确加载后
  切换推理语义；V7 的前 32 条已与冻结参考 motion 逐元素、逐字节完全一致
  (`max_abs=0`)。失败结果保留在
  `output/historical_mgt_v7_step280k_common50_bridge.json`，修复证据为
  `output/historical_mgt_v7_step280k_common50_conditioning_first32_audit.json`。
  fresh main 完成后又将同一审计扩展到官方全部 1,696 条：name/length 全等、motion
  逐字节全等、最大绝对误差 0、非零差异元素 0；完整证据为
  `output/historical_mgt_v7_step280k_common50_conditioning_full1696_audit.json`。
  该候选随后由同一官方 CPU main-only evaluator 独立评分为
  FID `0.18696466786173005`，与冻结参考重评分完全相同；诊断文件为
  `output/mgt_diagnostics/v7_step280k_common50_paper_text_cfg_full1696_main_only_score.json`。
  MM 另由 `tools/audit_mgt_common50_mm_replay.py` 流式审计全部 30 个完成标记和
  3,000 条生成，不用同时加载两个多 GB HDF5。完整审计已通过：30 个
  repeat 的 name/length 全等，motion 逐字节全等，最大误差为 0，非零差异
  元素为 0，两侧整体 motion SHA-256 同为 `e27cc463...bcfea`。最终报告
  会重跑并绑定
  `output/historical_mgt_v7_step280k_common50_mm_full30_audit.json`。
  完整桥接使用新标签
  `hhc_mgt_repair_v7_step280k_common50_paper_text_cfg_contact_rep00`，并显式要求
  training=`historical_mmm_shape_dropout`、inference=`paper_text_cfg`；旧错误
  common-50 结果不能放行 gate。新 bridge 已完成，两侧用同一 CPU official
  evaluator 重评后 FID 均为 `0.18696466786173005`，其他 ordinary metrics 也
  全部相同；`output/mgt_repair_gate.json` 已通过并于 2026-08-30 16:08
  放行正式 run
  `output/t2i_trans/2026-08-30-16-08-07_hhc_formal_raw12288_repaired_mgt_300k_seed123_v1`。
  历史 280k FID `0.16047221422195435`
  在 30 个 milestone 中排名第 9，故 `save_top_k=10` 会将对应 validation
  checkpoint 保留到 300k；V7 的逐值相等门禁使该排名同样成立，finalizer 仍会对
  实际文件 fail closed。bridge 逐项绑定
  `sample_intra_cas/50/0.5/3.5`、contact guidance `1.0`、request-oracle timing、
  train-only contact-map/VQ/mean/std 哈希，以及官方 evaluator options、checkpoint
  和实现源码哈希；任何 body-controlled 字段都会被拒绝。它不再仅凭 FID 数值和
  部分采样参数放行。
  普通 FID 与同一步旧 checkpoint 的绝对差限制在 0.02 内。gate 只能由
  `tools/build_mgt_repair_gate.py` 从六类带 SHA-256 的 pass 证据生成，再由
  `tools/require_mgt_repair_gate.py` 复核。新增的第六类证据是 V7 启动时 14 份
  Python 源码的内容寻址快照：其中两份启动后修改的文件由本地 Codex session 中
  7 条带时间戳的 `apply_patch` 记录逆向恢复，恢复字节必须逐一等于不可变 training
  manifest 的 SHA-256；选中的 7 行 session 记录也单独冻结，避免依赖持续增长的
  原始 session 文件。gate 会重新散列 recovery JSON、training manifest、patch
  records 和全部 14 份源码快照，之后工作树继续接入 shape tokenizer 也不会污染
  历史重放的源码证据。
  2026-09-01 的 shape 启动 preflight 进一步发现 main-only scorer 在 gate 冻结后
  加入了 generated-length 读取，因此旧 gate 正确地因源码 byte/hash 漂移而拒绝。
  对同一个冻结 VQ reconstruction HDF5 用当前 scorer 做 CPU 重评分后，FID
  `0.29441722725218256`、Diversity、三项 R-Precision 与 Matching 全部逐位零差；
  该 reconstruction 的 length policy 仍为 exact requested length。刷新后的 gate
  新增第七类 `vq_reconstruction_scorer_migration` 证据，同时绑定旧/新结果、当前
  scorer 源码以及 migration auditor SHA-256，而不是跳过 provenance 检查。证据为
  `output/vq_historical_eval/net_175000_reconstruction_scorer_migration_audit.json`。
  gate 通过后的唯一正式入口为
  `tools/run_formal_hhc_after_mgt_repair.sh`；其冻结启动审计
  `output/hhc_formal_launch_contract_audit.json` 已通过。入口使用独立实验名
  `hhc_formal_raw12288_repaired_mgt_300k_seed123_v1`，selector 因而不能混入
  8 月 28 日错误重构的完整 300k run，六个论文训练合同均在命令行显式给出。
  正式入口在训练结束后先由 selector 解析唯一完整 300k run，再从 selection
  反解 training manifest/run directory 后执行 20k pair 与论文合同审计；不再用
  `find -print -quit`，因此同名中断目录不会在重启后被误审计或误评测。
  selector 后、任何 augmentation-only 源码接线前，入口还会用
  `tools/freeze_mgt_training_sources.py` 将 training manifest 绑定的 14 个启动源码
  冻结到 run 内；formal launch/paper audit 与最终报告验证不可变快照，shape 两臂则验证接线后的
  live tree，避免把合法的后续接线误报为正式 HHC source drift。
  正式 paper-50 run 的 validation 最优 FID 为 `0.6364277601`（60k），同阶段
  50k 为 `0.8862313628`；这仍明显弱于大清理前已逐值复现的 50k
  `0.2442705631`，因此不能直接进入 shape 增强归因。对 clean MMM HEAD
  `a01d5ac...54a7` 的源码/缓存审计发现，原 loader 默认 tensor window 是 38，且
  END 占一位，实际只保留 37 个 motion token/148 帧；正式 run 的 window 50
  保留 49 token/196 帧。9,664 条缓存序列最大 247 token，两者分别截断
  5,006/2,792 条并丢弃 106,792/60,087 个 token，所以不是 padding 参数差异。
  frozen fair request 上限为 150 帧/38 motion token；报告必须同时披露原 38-window
  的 148-frame payload，不能称为精确覆盖 150 帧。GPU-1 下一项先做仅改变
  `paper_50 -> original_mmm_38` 的 online-W&B 50k validation 控制，晋级阈值固定为
  FID 绝对改善至少 0.1 且相对改善至少 15%。该控制 10k 的 common-50 FID 为
  `1.1610538960`，而 paper-50 同步点为 `0.9789292216`，尚无早期改善。原生 MMM
  20k 控制现已完成并通过 checkpoint/评测隔离审计：38-window FID 为
  `1.0573199987`，paper-50 同步点为 `1.2168186903`，单点反而改善
  `0.1594986916`；Matching 为 `3.7030270100` 对 `3.7235836983`。由于 10k 与
  20k 的方向相反，当前不得把 token window 判为根因或提前晋级，必须继续读取
  30k/40k/50k 的同协议曲线后按预注册阈值决策。30k 已进一步降至
  `0.6275210977`，而 paper-50 同步点为 `1.0176038742`，FID 改善扩大到
  `0.3900827765`。40k 又回升至 `0.9207732677`，相对 paper-50
  `0.9752652645` 仅改善 `0.0544919968`，未达 0.1 绝对门槛；这一波动支持 token
  window 是重要设置差异，但仍不替代 50k 预注册决策。与大清理前原生 MMM 的
  30k 训练量比较，masked accuracy 仅低
  `1.8574` 个百分点，visible accuracy 仍低 `14.0944` 个百分点，因此 loss
  support 仍是必须单独检验的剩余因素。50k 终点 FID 为
  `0.8790842295`，相对 paper-50 同步点 `0.8862313628` 只改善
  `0.0071471334`，不通过 pointwise 门槛；但在相同 10k--50k 候选集合中，
  38-window 的 validation-best 是 30k `0.6275210977`，优于 paper-50 的 early
  best。该非单调 minimum-selection 结果与 50k 同步效应必须分别披露，不能把
  checkpoint selection 简化成“50k 提升”。assessor 现明确输出
  `checkpoint_selection_supports_38_but_matched_endpoint_does_not`，因此该臂只保留为
  调查备份，不会替换后续 paper-50 正式/shape 配置。20k 的 pre/post-validation
  checkpoint 在 model、optimizer、scheduler 和 hyperparameters 上位级一致，且
  二者所有浮点状态均 finite；证据见
  `output/hhc_original38_vs_formal50_validation_10k_30k.json`、
  `output/hhc_original38_vs_native_mmm_training_trajectory_through30k.json`、
  `output/hhc_original38_token_window_control_20k_selection.json` 与
  `output/hhc_original38_token_window_control_20k_pre_post_validation_pair_audit.json`。
  完整控制的 online W&B run 是 `edmond_gla/NewInter/rx72y8g9`；50--50,000
  共 1,001 个遥测事件覆盖全部 loss/accuracy/mask/lr/throughput/GPU/梯度/权重
  指标，pre-clip gradient L2/max-abs 为
  `0.731383--5.554327`/`0.016357--0.882812`，weight L2/max-abs 为
  `220.186096--416.797943`/`1.000194--2.679024`，两类 non-finite count 始终为
  0，远端终态为 `finished`。原生 MMM
  10k 的 masked/visible accuracy 为 `67.70%/81.67%`，当前论文 masked-only 为
  `73.38%/67.27%`，因此另加一个 paper-50 下仅把 loss support 改为
  `historical_all_valid` 的 50k 解释性控制；该臂现已以 online W&B run
  `edmond_gla/NewInter/fx7a9y1i` 在 GPU1 启动。其前 1,000 update W&B 门禁已通过：
  gradient L2/max-abs 为 `0.752957--2.416588`/`0.042236--0.291016`，weight
  L2 从 `220.185989` 到 `226.857285`，non-finite count 为 0。与 paper masked-only
  同 step 比较，masked accuracy 从 `2.1629%` 升至 `6.6114%`，visible accuracy
  从 `1.6700%` 升至 `27.0270%`。5k checkpoint 也通过 model/optimizer/scheduler
  全 finite 状态审计；step 4,999 的 masked accuracy 已基本相同
  (`50.9486%` 对 `51.5811%`)，但 visible accuracy 是 `99.0747%` 对
  `55.9467%`。这支持 loss support 确实改变了学习动力学，并把差异精确定位到
  visible-token 学习，而不是 masked-token early fit；但
  是否改善 ordinary FID 仍必须等 10k--50k 固定验证。其在线审计已扩展为在不改变
  1,000-step warm-up 门禁的前提下另列 `full_run_metrics`；40k 验证后刷新时健康流已
  覆盖到最终 step 50,000（1,001 个事件），gradient L2/max-abs 为
  `0.441381--3.958138`/`0.010803--0.593750`，weight L2/max-abs 为
  `220.185989--412.299438`/`1.000198--2.790677`，两类 non-finite count 始终为
  0。审计还逐一确认 W&B 已实际收到 `val/loss`、两项 val accuracy、FID、三项
  R-Precision、Matching 和 Diversity，并分别列出 10k/20k 的两个有限事件，而不
  只是检查训练源码中是否出现这些字段。10k 固定协议验证已经完成：
  FID `0.97720653`、Matching `3.81753993`、R-Precision Top-1/2/3
  `0.400735/0.577206/0.694853`、Diversity `9.169179`。相同 checkpoint 下论文
  masked-only FID 是 `0.97892922`，只差 `-0.00172269`，说明 all-valid loss
  至少在该里程碑并没有解释 FID 缺口；run 仍按预注册计划继续到 50k。10k checkpoint
  的全部模型与 optimizer 浮点状态均 finite，15k checkpoint 也通过同一完整审计。
  step 15,999 的 masked accuracy 已低于 masked-only `2.4988` 个百分点，而 visible
  accuracy 仍高 `24.8397` 个百分点，说明该目标继续主要学习 visible token。20k
  固定验证进一步回退到 FID `1.37048256`、Matching `3.87215877`、R-Precision
  Top-1/2/3 `0.397059/0.591912/0.693015`、Diversity `9.002078`；同 step 的
  paper masked-only FID 是 `1.21681869`，all-valid 反而差 `0.15366387`。其
  step 19,999 masked accuracy 也低 `0.8966` 个百分点，而 visible accuracy 仍高
  `25.6534` 个百分点。20k pre/post-validation checkpoint 的 model、optimizer、
  scheduler 和 hyperparameters 已逐项验证一致，故该回退不是 validation 改写训练
  状态导致。证据为
  `output/hhc_all_valid_vs_formal50_validation_10k_20k.json`、
  `output/hhc_all_valid_vs_formal50_training_trajectory_10k_20k.json`、
  `output/hhc_all_valid_loss_control_20k_pre_post_validation_pair_audit.json` 与
  `output/hhc_all_valid_loss_control_mgt_wandb_warmup_live_audit.json`。
  25k rolling checkpoint 也已通过完整状态审计：global step `25,000`、epoch
  `280`，947 个模型 tensor 与 1,041 个 optimizer tensor 全部 finite，scheduler
  为 `last_epoch=25000`/`_step_count=25001`；证据为
  `output/hhc_all_valid_loss_control_25k_checkpoint_audit.json`。
  30k 固定验证 FID 为 `1.22663069`、Matching `3.72356272`、R-Precision
  Top-1/2/3 `0.437500/0.623162/0.716912`、Diversity `9.308086`；虽然从 20k
  的 `1.37048256` 回升，但同 step paper masked-only FID 为 `1.01760387`，
  all-valid 仍差 `0.20902681`。30k 的 masked accuracy 仍低 `0.8670` 个百分点，
  visible accuracy 则高 `21.1090` 个百分点。原子 rolling checkpoint
  `latest-000030000.ckpt` 通过完整状态审计：global step `30,000`、epoch `337`，
  947 个模型 tensor 与 1,041 个 optimizer tensor 全 finite，scheduler 为
  `last_epoch=30000`/`_step_count=30001`。20k 是预注册的唯一固定 pre/post-validation
  checkpoint pair；30k 不虚构不存在的 pair。对应证据为
  `output/hhc_all_valid_vs_formal50_validation_10k_30k.json`、
  `output/hhc_all_valid_vs_formal50_training_trajectory_10k_30k.json`、
  `output/hhc_all_valid_loss_control_30k_checkpoint_audit.json` 与
  `output/hhc_all_valid_loss_control_wandb_audit.json`。
  通用 warm-up auditor 对该 HHC run 必须显式使用预声明的
  `--max-weight-relative-span 0.15`：前 1,000 step 的 weight max-abs/L2 相对跨度为
  `10.6434%/3.0298%`；若遗漏参数会误用工具面向 TiMotion 的 5% 默认值并产生假
  fail，不代表训练状态变化。40k common-50 FID 进一步降至 `0.71260744`，相对
  同步 paper masked-only 的 `0.97526526` 改善 `0.26265782`（`26.93%`）；Matching
  改善 `0.02049661`，但 Top-1/Top-2 分别下降 `0.04411763/0.01470590`，Top-3
  上升 `0.01286763`。同 step masked accuracy 仍低 `1.1936` 个百分点而 visible
  accuracy 高 `21.0779` 个百分点，所以该点支持 loss support 是重要 FID 因素，
  但不支持把 all-valid 改成正式论文配置。40k checkpoint 的 model/optimizer 全部
  finite、scheduler step 精确；最终结论继续等待预注册 50k endpoint 与相同候选集
  selection。45k rolling checkpoint 也通过完整状态审计：epoch 505、947 个模型
  tensor 和 1,041 个 optimizer tensor 全 finite，scheduler 为
  `last_epoch=45000`/`_step_count=45001`；证据为
  `output/hhc_all_valid_loss_control_45k_checkpoint_audit.json`。
  50k 固定验证现已完成：FID `0.76316077`，同步 paper masked-only 为
  `0.88623136`，绝对改善 `0.12307060`、相对改善 `13.89%`，因此匹配 endpoint
  没达到预注册的 15% 相对门槛；但相同 10k 间隔候选集的 selection 选择 40k
  `0.71260744`，相对 formal early best 改善 `0.17362392`/`19.59%`。assessor
  因此将其判为“解释了相当部分 early FID gap”，但仍严格标为单因素诊断，不能替换
  论文 masked-only 正式配置。50k 的 Top-1/2/3 为
  `0.417279/0.613971/0.742647`、Matching `3.652760`、Diversity `9.1891`；
  checkpoint 与完整 remote W&B 历史审计均 PASS。
  无论结果如何都不得晋级为论文正式配置。
  global masked-token-mean 50k 控制继续单独检验 reduction。该控制的 10k
  common-50 FID 已为 `0.75840902`，相对同步 paper masked-only `0.97892922`
  改善 `0.22052020`（`22.53%`）；Matching 为 `4.02768421`，R-Precision
  Top-1/2/3 为 `0.363971/0.544118/0.654412`，Diversity 为 `8.96459`。10k
  同步配对中 Matching 变差 `0.16001511`，Top-1/2/3 分别下降
  `0.03492647/0.03676474/0.03860289`，因此不能表述成语义指标全面改善。10k
  checkpoint 与线上 W&B 九个 validation streams 均已审计通过；该点只作为早期
  reduction 证据，最终晋级仍等待预注册 50k endpoint。
  同一 step 9999 的 masked accuracy 与 paper masked-only 恰好相同，均为
  `73.75552368%`；visible/all-token accuracy 仅高 `1.07527161/0.40747833` 个
  百分点，batch loss 反而高 `0.01165444`。因此当前 FID 改善符合 batch 内 masked
  token 归约权重这一目标设置差异，而不是整体 masked classification 更强。
  同步 20k 验证随后出现非单调回退：FID 为 `1.96005368`，而同 step 论文
  masked-only 为 `1.21681869`，候选差 `0.74323499`；Matching 也差
  `0.08655691`，Top-1/2 低 `0.022059/0.027574`，Top-3 相同。候选的同步 masked
  accuracy 反而高 `0.700043` 个百分点，只是 batch loss 高 `0.04577023`，所以该
  FID 回退不能归因成简单分类准确率不足。20k pre/post-validation 的 model、
  optimizer、scheduler、hyperparameters 全部逐 bit 一致，排除了 evaluator 改写
  训练状态；证据为
  `output/hhc_global_masked_loss_vs_formal50_validation_10k_20k_live.json`、
  `output/hhc_global_masked_loss_vs_formal50_training_trajectory_10k_20k_live.json` 与
  `output/hhc_global_masked_loss_control_20k_pre_post_validation_pair_audit.json`。
  30k common-50 验证恢复到 FID `1.01232600`，与同步 paper masked-only
  `1.01760387` 基本相同（候选减参考 `-0.00527787`）；但语义指标并未同步恢复：
  Matching 为 `3.80849147` 对 `3.62438869`，Top-1/2/3 为
  `0.406250/0.608456/0.704044` 对 `0.463235/0.636029/0.733456`，Diversity
  为 `9.028303`。同一步训练 loss 仅差 `-0.00209561`，masked/visible accuracy
  分别低 `0.066811/1.889969` 个百分点。30k 后远程 W&B 审计覆盖到 optimizer
  step 30,200：最新 gradient L2/max-abs 为 `1.434748/0.060303`，weight
  L2/max-abs 为 `370.351410/2.740283`，gradient/weight 非有限计数均为零，九个
  validation streams 全部存在。进一步按 checkpoint 的真实可训练 tensor 比较，
  20k formal/current 的 weight L2 为 `340.015875/339.856317`，而健康 formal
  参考本身会增到 `418.678923@50k`、`552.304406@300k`，所以权重增长不是该差异的
  独立根因，不能据此在 run 中途改 optimizer 或添加 clipping。证据为
  `output/hhc_global_masked_loss_vs_formal50_validation_10k_30k_live.json`、
  `output/hhc_global_masked_loss_vs_formal50_training_trajectory_10k_30k_live.json` 与
  `output/hhc_global_masked_loss_control_mgt_wandb_warmup_audit.json`。
  40k 的同一固定协议得到 FID `1.08753538`、Matching `3.69119167`、
  Top-1/2/3 `0.443015/0.625000/0.715074`、Diversity `9.186666`。相对同步
  paper masked-only，FID/Matching 分别差 `0.11227012/0.10865664`，Top-1/2/3
  分别低 `0.020221/0.011029/0.022059`；候选同一步 train loss 反而低
  `0.02333662`，再次说明训练 batch loss 不能单独解释固定采样质量。远程审计已覆盖
  804 个 telemetry event 至 step 40,200：gradient L2/max-abs
  `0.943263/0.043701`、weight L2/max-abs `395.714417/2.756534`，非有限计数为
  0。第四次 val loss/masked accuracy 为 `10.621068/2.933152%`；由于 loss 侧
  corruption seed 随 milestone 变化，这两项只作支持证据，固定生成 seed 和
  common-50 才用于严格同步比较。40k checkpoint 的 947 个模型 tensor、1,041 个
  optimizer tensor 和 scheduler `last_epoch=40000` 全部通过审计。证据为
  `output/hhc_global_masked_loss_vs_formal50_validation_10k_40k_live.json`、
  `output/hhc_global_masked_loss_vs_formal50_training_trajectory_10k_40k_live.json`、
  `output/hhc_global_masked_loss_control_40k_checkpoint_audit.json` 与上述 W&B 审计。
  当前 run 的 best-FID 仍为 10k，最终决定等待预注册 50k。
  进入终点前的 45k rolling checkpoint 也已独立通过有限状态审计：947 个模型
  tensor、1,041 个 optimizer tensor 均有限，weight decay 仍为 `1e-6`，scheduler
  `last_epoch=45000`；证据为
  `output/hhc_global_masked_loss_control_45k_checkpoint_audit.json`。
  50k 终点现已完成：候选/同步 paper masked-only 的 FID 为
  `1.05582905/0.88623136`，Matching 为 `3.63405228/3.55796647`，候选
  Top-1/2/3 分别低 `0.016544/0.012868/0.029412`。与此同时候选同一步训练
  loss 低 `0.04541746`、masked accuracy 高 `1.328926` 个百分点，说明 global
  loss 的训练拟合改善没有转成生成改善。固定 10k 间隔选择仍保留 10k FID
  `0.75840902`；相对 formal early-best 的绝对改善为 `0.12782234`，但相对改善
  仅 `14.4231%`，未过预注册 `15%`，因此不续到 300k。最终 assessment/dispatch
  已绑定结论并转向 paper masked-only shape augmentation。证据为
  `output/hhc_global_masked_loss_vs_formal50_validation_10k_50k.json`、
  `output/hhc_global_masked_loss_vs_formal50_training_trajectory_10k_50k.json`、
  `output/hhc_global_masked_loss_control_50k_checkpoint_audit.json`、
  `output/hhc_global_masked_loss_control_assessment.json` 与
  `output/hhc_post_50k_training_dispatch.json`。
  各 assessor 均逐字段比较 training manifest，除目标因子外任何结构、mask、
  optimizer、seed、cache、sampler
  漂移都 fail closed。若 global reduction 同样不能达到绝对 `0.1` 且相对 `15%`
  的预注册 FID 改善，下一项诊断只把 `paper_kinematic` corruption 换成已逐值恢复的
  `historical_flat_random_replace`，其余 paper-50 结构、masked-only loss、optimizer、
  conditioning、seed 和 common-50 validation 全部不变。该 50k 单因素臂只用于解释
  旧 FID 差异，不能替换正式论文 mask 配置，也不能延迟用户要求的 shape 增强主实验。
  该入口现已实现为 `tools/run_hhc_historical_mask_control.sh`，assessor 为
  `tools/assess_hhc_historical_mask_control.py`；它会先读取 global-loss 最终 decision，
  当前 decision 尚为 `run_global_masked_token_mean_50k_control` 时已实测 fail-fast，
  不会提前启动。其 online W&B 静态合同审计已通过，证据为
  `output/hhc_historical_mask_control_online_wandb_audit.json`；预注册 assessor 状态
  保存在 `output/hhc_historical_mask_control_assessment.json`。
  主 HHC 与两个 shape-augmentation 训练臂还共用严格 restart resolver：唯一
  incomplete run 才允许在原目录恢复 optimizer/scheduler，且恢复前逐字段核对
  数据、结构、mask/loss、50-step sampling 与 14 个启动源码哈希；多个 incomplete
  run、源码漂移或合同漂移全部 fail-closed，完整 300k run 则直接跳过重训。
  训练在 20k validation 前后分别永久保留
  `prevalidation-000020000.ckpt` 与 `postvalidation-000020000.ckpt`；不能依赖
  `latest-000020000.ckpt` 或 `model-000020000-*`，因为 rolling `save_top_k=1`
  会在 25k 删除前者，后续 validation top-k 也可能删除后者。最终主 HHC 与两个
  shape 臂不仅绑定这两个永久文件，还强制核对 callback 的 pre/post 完成 marker。
  当前两个训练进程之后的全部依赖由
  `tools/queue_repaired_interx_end_to_end.sh` 在一个非 tmux orchestrator 中接续；
  它不再新建 watcher terminal，并且任一 GPU 分支失败时禁止生成最终报告。
  最终 paired comparison 前只复用现有 HDF5 对较早完成的 InterMask/Interact2AR
  做官方 full-672D ordinary evaluator 重评分，不重新采样；随后对六种最终方法的
  120 个 replication JSON 逐文件哈希审计，并要求每条记录内嵌且匹配同一
  evaluator options/checkpoint/wrapper/loader/scorer provenance；任何
  body-controlled 指标字段都会被拒绝。最终六种方法还会在两条训练分支结束后
  统一用 CPU 重评现有 HDF5，以匹配冻结 GT/text embedding 的原始 CPU 创建命令；
  每条 metric JSON 记录 PyTorch/CUDA/device runtime，避免把 CPU、CUDA 0 与
  CUDA 1 的 embedding 舍入差异混入主表。
  最终报告、paired tests、contact-timing ablation、optimization、qualitative、
  sampling Pareto 与 capacity audit 现全部绑定
  `hhc_formal_raw12288_repaired_mgt_{selection,common50_*}`，并逐路径验证 paired
  evidence 的 source directory；shape-augmentation 队列与结果 builder 也只接受
  `hhc_formal_raw12288_repaired_mgt_shapeaug_*` selector/generation，不再接受旧 FID
  `0.645` selection。两个旧恢复
  launcher 保留作调查记录但入口已 fail-fast。正式 launcher 还会生成自身 20k
  validation 前后 checkpoint pair 审计，最终报告不再借用旧错误 run 的 90k
  side-effect 证据。VQ reconstruction FID 门槛也已切换为无任何 secondary/body
  字段的 ordinary-FID `0.29441722725218256`。
  全量 optimized 资产现按冻结 selection position 切为 8 个连续 shard；最终
  aggregate 会逐条复验 sidecar position 与双人 NPZ hash，并要求 `0..6237`
  恰好覆盖、无重名无缺口、所有 shard 的 plan/contact/optimizer/SMPL/implementation
  合同一致。materialization manifest 必须逐一绑定 8 份原始 completion hash，
  result builder 不接受手工合成的单目录完成标记。两个 shape MGT 臂的 GPU-free
  launcher 启动前还会审计 online W&B 合同；每 50 optimizer step 记录全局及顶层
  trainable module 的梯度/权重 L2、max-abs、grad/weight ratio、元素数和 nonfinite
  计数，启用 clipping 时另记 post-clip norm、阈值、触发事件与累计命中率。
  另一个配对的 64-sample validation-token probe 排除了“旧 MGT 只是单步 token
  分类更准”这一简单解释：在完全相同的 paper-scheduled mask 上，formal 10k 的
  加权 token accuracy/CE 是 `7.4004%/8.4078`，历史 280k 是
  `1.5417%/12.9520`；全 mask 下二者为 `0.8444%/10.2295` 和
  `1.0074%/13.2973`。也就是说，历史 checkpoint 的 common-50 decoded-motion
  FID 明显更好，不能由更好的 one-shot masked-token CE/accuracy 解释。由于两者
  checkpoint 步数与训练合同不同，这只是把后续诊断收窄到迭代采样所匹配的训练
  corruption/reduction/conditioning 分布，不能单独作因果结论；证据为
  `output/hhc_historical_mgt_mask_exposure_280k_early_comparator.json`。
  当前冻结请求 replication 0 的诊断量级为：清理前 Git 默认 280k MGT 的
  common-50 生成用当前 main/MM scorer 在 CPU 重评后，普通 FID 为
  `0.186965`（早期 scorer/GPU JSON 为 `0.186856`，仅作历史记录）；错误重构
  训练所选 20k MGT 为 `0.645356`；
  raw12288/175k VQ 对同一 1696 条请求的 main-only 普通重建 FID 为
  `0.294417`。前两个 checkpoint step 不同，故只能证明重构 MGT 表现异常，
  不能作为最终因果配对；最终 gate 使用 280k 同步 checkpoint 及同一请求重评。
  旧 W&B requirements 与当前 `New_MMM` 的 PyTorch `2.5.1`、Lightning/
  PyTorch-Lightning `2.6.0`、NumPy `2.2.6`、W&B `0.23.1` 也逐项一致，并已作为
  总合同的 fail-fast 检查，排除框架版本造成的 AMP/DataLoader/optimizer 时序差异。
  依赖源码审计进一步确认 encdec/position/VQ/FK 文件逐字节一致，旧/current
  attention blocks、runtime mask/gumbel 及 token-dataset `__getitem__` 的 AST
  一致。唯一修改时间晚于原训练的 `paramUtil.py` 已从冻结 Local History/VFS
  恢复；3 月修改只拆分了 contact-region 的手掌/手背标签，MGT 实际读取的
  `SMPLX_TREE/INTERX_SKELETONS` 尾部逐字节一致。证据为
  `output/historical_mgt_dependency_contract_audit.json`，并已纳入总合同。
  V4/V5 的旧
  `sample_intra_cas/10 outer steps/temperature=0/CFG=3.5` 只复现训练时 validation
  副作用，不替代用户指定的正式 common-50 评测。

  单独的初始化审计已经证明：恢复的 VQ 与 transformer 构造顺序、构造前后 RNG
  状态、初始 transformer 权重和参数量与 2 月 13 日源码逐项一致，因此初始化不再
  是待猜测变量。known-good 控制必须至少恢复历史 `0.697935@10k`、
  `0.642393@20k` 的同量级轨迹；通过后还必须将同一 checkpoint 用冻结 common-50
  官方 evaluator 重评，建立 historical-validation 到正式协议的桥接。之后才逐项
  迁移 conditioning、mask、loss、optimizer/scheduler 与论文结构，每次只改一组
  设置并在 10k/20k 回归。未通过该门槛时，所有正式 HHC 重训、消融与
  shape-augmentation MGT 都暂停；最终主表只接受全部论文合同，diagnostic legacy
  分支不得进入主表。body-controlled FID 不参与任何上述诊断或门禁。
  原训练的 W&B metadata/config/summary/transaction log、31 个初始与 10k--300k
  milestone、四份 time-bracketed Local History 源码、run 前未修改的数据集/优化器
  工厂、200k checkpoint、初始化与 sampler 行为审计已统一冻结在
  `output/historical_mgt_original_training_contract_audit.json`，状态为 `pass`。
  checkpoint FID 使用官方 batch-32、17 个完整 batch、544 条 val；训练 manifest
  中另一个 batch-96、5-batch、480 条的 validation loader 只监控 token loss，
  不参与 FID 或 checkpoint selection。审计报告已分别冻结两者的角色与计数。
  `output/hhc_formal_raw12288_repaired_mgt_training_contract_audit.json`
  另行证明论文明确给出的
  9 blocks/8 heads/512 hidden、batch 96、300k、1k warm-up、AdamW/cosine、
  `2e-4/1e-6`、独立排序 cosine masks、Root→Body→Hands、shape conditioning
  以及 `0.5/3.5/1.0` 均与本次启动配置一致，并逐个校验 14 个启动源码哈希。
  raw-12288/175k 与共同 50 步仍作为用户指定的公开例外。BF16 是论文未说明的
  实现选择，但相关的大清理前双人入口 `train_t2i_trans.py` 同样使用
  `bf16-mixed`，所以它不是旧低 FID 与当前轨迹之间的观测差异。真正尚未隔离的
  差异包括旧 `AdamW(0.5,0.9)+150k MultiStepLR`、旧 flattened mask/all-token loss、
  1024/16-head/额外 output-head blocks、batch 与 seed；不能把其中任一项未经 matched
  sensitivity 就写成单一原因。
  90k 同一步的 validation 前 rolling checkpoint 与 validation 后 FID checkpoint
  已进一步逐 tensor 配对：947 个 model tensor、optimizer、scheduler 和
  hyperparameters 均 bitwise identical；只有预期的 callback/loop validation
  metadata 不同。这给出了真实 544-sample validation 无训练状态副作用的运行证据，
  不只依赖单元测试。由于 `save_top_k=1` 的 rolling checkpoint 会被后续节点正常
  替换，90k 的两个源 checkpoint 现已过期；最终报告队列在源文件存在时重跑审计，
  否则只接受上述冻结报告，并逐项核对整份报告 SHA-256（`d481e46c...`）、
  schema/status、step、四类 exact-state、空 differing-key 列表、两个源 SHA-256
  以及 before/after/best FID。该回退分支已在
  源文件过期的真实状态下通过，不会把缺失源 checkpoint 误当成新的实验失败。

- 不计算也不报告 body-controlled FID。Inter-X 主结果只使用冻结官方 full-space
  evaluator 的 FID；body/hand MPJPE 仅属于 VQ reconstruction 诊断，不得改名或
  汇入生成 FID。可执行 scorer、聚合器、报告门槛与活动配置均不得再依赖
  body-controlled FID。

- InterMask 官方权重、作者默认采样参数、精确 150-frame policy 的 20 个
  replications 已端到端完成（每次 1696 main + 固定 100 文本 × 30 MM）。
  该产物 HDF5 明确记录 `time_steps=20`、`cond_scale=2.0`、
  `temperature=1.0`；因此下列数值是 **native-20 / exact-length 备份对照**，
  不是共同 50 步主表结果。共同 50 步的
  `intermask_official_exact_common50` 现已独立完成 20 次：FID
  `0.374280 ± 0.014655`、
  Top-1/2/3 `0.405366/0.598850/0.709906`、Matching `3.700004 ± 0.020057`、
  Diversity `8.991844 ± 0.088159`、官方 MModality `1.675817 ± 0.070547`。
  这些是最终统一 CPU/current-scorer 数值；早期 GPU/旧 scorer 汇总仅保留在
  provenance 中，不再进入主表。
  native-20 备份对照的均值和 95% CI 为 FID `0.39169 ± 0.01430`、
  Top-1/2/3 `0.40032/0.58824/0.69959`、Matching
  `3.72855 ± 0.01924`、Diversity `9.01711 ± 0.05896`、官方 10-pair
  MModality `1.80842 ± 0.06971`；全 435-pair MModality 为
  `1.88245 ± 0.04404`。Replication 0 为 FID `0.44002`、Top-1/2/3
  `0.40625/0.60554/0.70873`、Matching `3.65569`、Diversity `9.04641`、
  MModality `1.89830`。10 cm 独立物理指标为 contact F1 `0.53666`、temporal
  IoU `0.36674`、false-contact rate `0.20644`、foot sliding `0.21441 m/s`。
  公平主表一律按请求精确生成/裁剪到 150 帧；另跑一次作者原生 floor-to-4
  （150→148 帧）审计，用显式 `generated_lengths` 送入 evaluator，以判断与论文
  MModality 差异中有多少来自长度策略，但该审计不与公平主表混合。该审计已
  用 replication 0 的相同请求和 seed 完成：1696 个样本中 836 个 capped-150
  样本变为 148 帧，其余 860 个长度不变。原生/精确长度的 FID 分别为
  `0.42365/0.44002`，官方 MM 为 `1.92074/1.89830`，全 435-pair MM 为
  `1.83786/1.89887`。两种长度策略不能混入同一主表。
- CoShMDM 采样器要求一个 batch 共享时间长度；公平导出按请求长度分桶，并直接
  生成各自的目标长度。禁止把所有动作先置于 150 帧扩散上下文再裁剪短动作；
  HDF5 以 `generation_length_policy=exact_request_length_grouped` 固化这一约束。
  2026-09-05 按“论文/项目页 + 官方仓库”联合复核后，正式 claim scope 进一步
  收紧：论文 DOI 为 `10.1109/TVCG.2026.3675725`，但全文 closed access、项目页
  Paper/arXiv 按钮禁用；公开摘要所述 VQ-VAE、body-shape-conditioned diffusion
  和 RL-SDF policy 不存在于官方仓库唯一的 `main` branch（无 tag）。公开可运行
  主链是每人 262-D 的直接 diffusion，只有 InterHuman checkpoint。因此下述实验
  只称 **public-repository diffusion Inter-X port**；它满足用户指定仓库的公平
  运行要求，但不是论文架构或摘要所报 Inter-X FID `0.013` 的复现。机器审计为
  `output/coshmdm_public_release_scope_audit.json`。
  作者配置的完整目标是 **2,000 个训练 epoch / 568,000 updates**，不是
  2,000 个推理采样步；它保留在 provenance 中作为原预算披露。根据用户
  2026-08-30 的资源上限，当前结果标签改为
  `coshmdm_interx_retrained_2daycap_seed123`：优先在稳定 200k milestone 停止，
  否则最迟在 `2026-09-01T17:51:54+01:00` 中断；不得再静默恢复到 568k。
  训练 manifest 与最终报告门禁仍强制推理为
  `sampling_strategy=ddim50`、`CFG=3.5`，训练 diffusion process 为 1,000 steps。
  实际训练按用户上限停止于 105,860 update，并由独立 validation 选择 40k。
  已用同一当前 scorer、官方 full-672D evaluator 和 CPU 对 CoShMDM 与正式 HHC
  contact 的冻结 HDF5 各重评 20 次：CoShMDM FID 为
  `41.444770 ± 0.479421`，HHC 为 `0.545132 ± 0.024679`；HHC 在 FID、Matching
  和 Top-1/2/3 上均为 20/20 胜，CoShMDM 在 MModality 上为 20/20 胜，所有差异
  在当前两方法×指标 Holm family 中均显著。证据为
  `output/fair_eval/{hhc_formal_raw12288_repaired_mgt_common50_contact,coshmdm_interx_retrained_2daycap_seed123}_cpu_metric_audit.json` 与
  `output/hhc_vs_coshmdm_cpu_evaluator_paired.json`；全方法主表完成前该两方法检验
  仍标记 provisional，最终必须放入一个更大的 Holm family 重算。
  CPU ordinary 重评使 replication 3 和 10 的官方 10-pair MModality 各发生一个
  float32 ULP（`4.77e-7`）变化，旧 435-pair 汇总因此被最终报告的 exact-value
  门禁拒绝；现已从当前 20 个 per-repeat JSON 重新聚合，二者向量逐值完全相同。
  当前官方 10-pair MModality 为 `6.122696 ± 0.254564`，全 435-pair 审计为
  `6.491935 ± 0.042577`，说明 10-pair estimator 的方差明显更大。统一 CPU
  rescore 入口现会在每个方法重评分后强制同步重建 exhaustive-MM summary，避免
  最终六方法报告再次出现同类陈旧绑定。
- Interact2AR 官方 mixed-memory 权重（global step 350k，SHA-256
  `543a33...e997`）已接入同一冻结请求。作者 README 的主评测默认提供 GT
  `first_frame`，而其他方法不具备该信息；严格主表禁用首帧条件，保留作者 CFG
  2.5、自回归 ancestral 10-step sampler，并按精确请求长度分组。无 GT 首帧的
  完整 20-replication 已用最终统一 CPU/current scorer 重评：FID
  `2.444256 ± 0.055243`、Top-1/2/3
  `0.291775/0.464033/0.582252`、Matching
  `4.473353 ± 0.014890`、Diversity `9.244821 ± 0.084768`、MModality
  `3.058983 ± 0.111831`。单个 150-frame 与全量 HDF5 均为 finite，且冻结 request、
  精确长度、实现快照和 evaluator provenance 已通过最终门禁，证据为
  `output/fair_eval/interact2ar_official_fair_no_gt_first_frame_cpu_metric_audit.json`。
  在尚缺 TiMotion 的 4-method 临时 Holm family（HHC reference 加 InterMask、
  Interact2AR、CoShMDM）中，HHC 对 Interact2AR 的 FID/Matching/Top-1/2/3 均为
  20/20 胜且校正后显著，Interact2AR 的 MModality 20/20 胜且显著；diversity
  absolute error 不显著。InterMask 仍在 FID 上 20/20 胜 HHC，而 HHC Top-1
  显著更高。完整配对证据为
  `output/fair_eval/paired_fair_method_comparison_without_timotion.json`；TiMotion
  完成后必须用四个 comparator 的更大 family 重算，不能直接沿用这里的 p 值。
  同一 replication-0、LLM-free joint 物理协议下，no-timing HHC/InterMask/
  Interact2AR/CoShMDM 的 contact F1 分别为
  `0.51590/0.53629/0.54013/0.48093`，foot sliding 为
  `0.36483/0.20694/0.20638/0.11742 m/s`。oracle timing HHC 的 contact F1
  提升到 `0.59718`，但 foot sliding 为 `0.36816 m/s`，说明 timing guidance
  改善接触命中却没有解决脚滑，优化/foot-skating 实验仍是必要项。这些都是单个
  冻结 replication 的 effect-size 描述，不能替代最终 TiMotion 齐全后的主表。
- LLM robustness 的预注册 128 条（timing-negative/laterality/multi-contact/role
  各 32）已完成。相对人工二值 timing，Luna 的 contact-presence accuracy/F1 为
  `0.90625/0.94059`，Terra reference 为 `0.89844/0.93596`，规则 baseline 为
  `0.75781/0.83770`；Luna 与 Terra 的 75-region pair agreement 为 ordered F1
  `0.65052`、unordered F1 `0.76056`，sample exact match 分别为 `0.65625/0.75`。
  这些 region 数值只表示跨模型一致性，不能称为人工 part oracle。Luna 正式 test
  全 1,696 条缓存审计通过（一次失败尝试后成功重试）：成功 batch token
  `347,489`，含失败尝试合计 `353,134` token；摊销 wall-clock 均值
  `1.3924 s/sample`。Terra-128 为 `73,574` token、`1.7026 s/sample`。
  全数据集 first-caption 辅助标注也已完成并独立复审：train `9,110/9,110`
  （8,318 个结构化 contact rows）、val `570/570`（471 rows）、test
  `1,708/1,708`（1,543 rows），合计 11,388 条、零缺失/非法样本。test-full 与
  frozen-test-1,696 是不同 caption binding，正式 optimizer 只使用 request-hash
  绑定的 frozen-test 缓存。最终结果构建器已把三组全量 cache+timing audit、
  frozen-test 和 Terra-128 共八个 artifact 设为强制门禁。
- 已废弃 CoShMDM run 的 20k checkpoint 通过了状态审计：global step `20,000`、epoch `70`、
  scheduler `last_epoch=70`、196 个 optimizer parameter states，模型的 346 个
  浮点 tensor 与 optimizer 的 588 个浮点 tensor 全部 finite。带步数文件与
  `last.ckpt` 的 model/optimizer/scheduler/loop 状态 bitwise identical，仅
  callback 的 last-file 元数据不同。单样本 DDIM50 exporter smoke 生成精确
  `(1,1,150,672)`、全 finite 的运动，并验证 sample name、150 帧长度、完成标志、
  checkpoint SHA 与 request SHA；目录明确标记 `20k_early_smoke_not_final`，不进入
  论文结果或 checkpoint 选择。随后发现该 batch-32×2 实现把第 569 个半批梯度
  带入下一 epoch，导致 569k 而非作者 physical-batch-64 的 568k 更新预算；整个
  run 已保存在 `CoShMDM-InterX-fair-seed123-bs32x2_aborted_cross_epoch_accum_step21620`
  且排除。随后那条 epoch-boundary-correct predecessor 在每个 epoch 边界丢弃未完成
  accumulation，并从 seed 123/update 0 重训。该 predecessor 的首个 20k checkpoint 已通过独立状态
  审计：`global_step=20,000`/epoch `70`，scheduler `last_epoch=70`，196 个
  optimizer parameter states；模型 346 个和 optimizer 588 个浮点 tensor
  全部 finite。checkpoint 为 `2,701,618,022` bytes，SHA-256
  `bfb4a20f...fb28e44`，机器可读记录为
  `output/coshmdm_corrected_20k_checkpoint_audit.json`。该里程碑只证明训练状态
  可继续，不进入最终主表，也不属于当前 `...fair-statsfix...` run。后续审计发现该
  predecessor 的 300k 后恢复命令没有
  传入 `COSHMDM_STATS_PREFIX`；实际进程因而回退到作者 InterHuman 的
  `global_mean/std`，而不是从官方 Inter-X train split 重算的
  `interx_global_mean/std`。两组数组逐字节不同，所以 300k 后的 checkpoint 混合了
  两套 normalization 合同，不能用于公平比较；而精确 300k 文件已被作者的
  `save_top_k=1` callback 轮换掉，无法安全回滚。该目录及 300k/320k/340k 审计全部
  保留为诊断证据，但由
  `output/coshmdm_mixed_stats_aborted_diagnostic.json` 明确标记为不可复用。
  正式 CoShMDM arm 因数据设置不一致这一公平性原因，使用新实验名从 seed 123/update 0
  重训。训练入口现强制 Inter-X 必须显式给出
  `--stats-prefix interx_global`，把 mean/std SHA 写入每个 checkpoint，并在任何恢复时
  逐字段核对 runtime contract；缺失或不一致会在更新前失败。在线合同记录为新目录
  `meta/runtime_contract.json`。最终用户限时结果必须由
  `output/coshmdm_resume_provenance.json` 同时绑定 corrected manifest、runtime
  manifest、被排除旧 run、训练日志、两天预算 manifest、validation-only
  checkpoint selection 和最终 checkpoint 审计后才允许进入报告；同时必须明确
  `original_author_budget_completed=false`，不得伪装成作者完整预算结果。
  validation exporter 虽支持断点续跑，但缓存文件名现在同时绑定 milestone step
  与 checkpoint SHA 前 16 位，并在每个候选导出后重新哈希 checkpoint；最终状态
  审计显式要求 selection 中的 `selected_step`。报告构建器还会重算全部候选中的
  `(最低 validation FID, 最早 step)`，拒绝同 step 异权重缓存、非有限 FID 或伪造
  selected row，不能只凭 selection JSON 的 `status=complete` 进入主表。
  epoch-boundary 更新预算不再只信任静态 manifest：
  `tools/audit_coshmdm_statsfix_startup.py` 会合并同一 run 的全部 TensorBoard event
  文件，强制 discard scalar 的 epoch step 从 0 连续且每个值都为 1；限时停止后
  要求从 epoch 0 到实际完成 epoch 无缺口，并同时绑定完整日志、event 文件和当前
  runtime manifest。原始 2,000-epoch 数量仅作未完成作者预算的披露，不再作为用户
  限时 arm 的通过条件。
  当前 `...fair-statsfix...` 正式 run 的首个 20k checkpoint 已通过独立审计：
  `global_step=20,000`/epoch `70`，scheduler `last_epoch=70`，196 个 optimizer
  parameter states，模型 346 个和 optimizer 588 个浮点 tensor 全部 finite。
  带步数文件 SHA-256 为
  `140a2496e941615c3377cb186f187c2c283455e2fb033632ba5694d5387a338d`；它与
  `last.ckpt` 的 model、optimizer、scheduler、loop、hyperparameters 和嵌入式
  runtime contract 逐位一致，只有 Lightning callback 的 last-file 元数据不同。
  checkpoint 内 mean/std SHA 分别为 `aa281b878...f568d`/`30a83e929...0b104`，明确
  属于 `interx_global`，不能与上述 SHA `bfb4...` 的旧 predecessor 混淆。审计为
  `output/coshmdm_statsfix_20k_checkpoint_audit.json`、
  `output/coshmdm_statsfix_20k_last_checkpoint_audit.json` 和
  `output/coshmdm_statsfix_20k_checkpoint_pair_audit.json`。为防作者
  `save_top_k=1` 在 40k 轮换掉精确 20k 文件，已创建同 inode、零额外数据占用的
  `model/milestone-000020000.ckpt` 硬链接；其 SHA 和完整状态由
  `output/coshmdm_statsfix_20k_preserved_checkpoint_audit.json` 再次验证。在线和最终
  `coshmdm_resume_provenance` 均绑定该 audit 与 checkpoint 本体，最终报告会重新哈希
  2.7GB 文件并强制固定 SHA。20k 后训练继续，在线状态保持无 full-state resume、
  segfault 或完成标志；已完成边界的 discard scalar 连续、无重复且全部为 1。
  当前机器证据为 `output/coshmdm_resume_provenance_live.json` 和
  `output/coshmdm_statsfix_startup_audit.json`；这些在线计数会继续增长，最终只接受
  两天预算守护正常停止、validation-only 选点和用户限时终态审计三者同时通过。
  该正式 run 的 40k checkpoint 也已到达并通过完整状态审计：global step
  `40,000`、epoch `140`，346 个模型 tensor 与 588 个 optimizer tensor 全部
  finite，196 个 optimizer parameter-state entries 完整；checkpoint 内仍绑定
  `interx_global` mean/std SHA。文件 SHA-256 为
  `72dcb5cf...dbd051`，审计位于
  `output/coshmdm_fair_statsfix_step40000_checkpoint_audit.json`。另建同 inode
  的 `milestone-000040000.ckpt` 硬链接以防 callback 轮换；训练最多继续到稳定
  `200,000` 更新或上述硬截止时间。40k 仍不直接进入最终结果，除非限时结束后它在
  冻结 544 条 Inter-X validation 上取得所有候选 milestone 的最低官方 FID。
  60k checkpoint 现也已通过完整审计：global step `60,000`、epoch `211`，346 个
  模型 tensor、588 个 optimizer tensor 与 196 个 parameter-state entries 全部
  finite；嵌入的 `interx_global` mean/std SHA 未变，文件 SHA-256 为
  `b3710d51...b869e5`，证据为
  `output/coshmdm_fair_statsfix_step60000_checkpoint_audit.json`。新增的轻量
  systemd milestone preserver 在源 checkpoint 稳定 45 秒后只创建同 inode 硬链接，
  不复制 2.7GB 文件；60k 源与 `milestone-000060000.ckpt` 的 inode 均为
  `126307594`。它持续保护至用户允许的 200k，并由
  `output/coshmdm_milestone_preservation.json` 记录，解决作者 `save_top_k=1`
  轮换掉 validation 候选、以及 200k 限时器看不到终点的问题。
- 正式 CoShMDM 训练合同现另由
  `tools/audit_coshmdm_training_contract.py` 独立复核，而不是只信任 launch
  manifest：13/13 个源码、split、mean/std 与统计 manifest 哈希一致，官方
  train/val/test 为 `9,110/570/1,708` 且零交叉，262-D mean/std 全 finite 且
  std 全正；Git revision/tracked-diff、epoch-boundary discard 和作者原始
  `2,000×284=568,000` 预算算术均通过。该算术证明原合同，但当前不会执行到
  568k；GPU0 完成队列与最终报告另外强制验证用户两天 cap 和 validation-only
  checkpoint selection。
- 接触概率图新增独立设置审计：公平主表使用官方 train-only 的 9,095 条有效样本，
  正确按 `[T,3]` 的 Body/Hand/Root 轴统计，未见 code 取中性概率 0.5。MMM 原始
  `code_contact_probs.npy` 的缓存实际覆盖 train/val/test `9,095/569/1,706`，且
  `test_data.py` 对 `[1,T,3]` 使用 `indices[:, part_idx]`，把时间行 0/1/2 误作
  部位轴；审计已逐字节复现该旧图。正式 exporter 默认只接受
  `paper_formula_train_only`；旧图必须显式声明
  `mmm_legacy_full_axis_bug`，限 replication 0，仅报告 effect size 并排除出公平
  主表。机器证据为 `output/hhc_contact_guidance_map_audit.json`，配对结果将写入
  `output/fair_eval/paired_contact_map_implementation_sensitivity_rep0.json`。
- shape 增强的两条正式 arm 不重算 normalization。为了让相对未增强 baseline 的
  差异只来自新增 shape 数据，两臂中的原始 9,664 个 train+val token/contact 对必须
  从 MMM cache 逐文件 byte-exact 导入，禁止重新 crop 或重新编码；仅新增随机-shape/
  optimized-shape 名称使用从 2026-02-01 cache 创建运行恢复出的
  tokenizer 合同：先 `to_rot_6d`，再使用
  `Interx_{Mean,Std}_root.npy`，不调用 `prepare_interx`，并以 sample-keyed RNG
  复现原 `single/single/double` crop 分布。该运行证据为
  `output/historical_mgt_tokenizer_runtime_audit.json`。原 shape launcher 曾把原始
  样本也按 deterministic contract 重建 cache，因而混入 crop/cache 设置变化；正式
  shape 训练前必须替换为
  `hybrid_historical_original_plus_deterministic_augmented_v1`，并分别审计 original
  子集与 MMM cache 逐字节一致、augmented 子集只包含 materialization manifest 中的
  新名称。当前 shape launch 审计因此主动 fail-closed，而不是把该混杂隐藏为已完成；
  训练入口只在原 MMM 25-sample 门禁通过后接线。当前不再使用持久 GPU waiter 或
  轮询队列：旧 queue、`run_interx_shape_augmentation_nearest_last_stage.sh` 和旧
  shape-training 入口均已归档并在开头 fail closed。当前唯一可执行的下一阶段入口
  是 `tools/run_original_mmm_shape_pilot25.sh GPU_INDEX`；它要求 GPU 已空闲，完成后
  停止，绝不串联全量生成或训练。由此 shape augmentation 被强制为 Inter-X 最后
  阶段，不会提前占卡；在门禁冻结前不得启动 shape 训练。materialized
  overlay 同时逐字节保存 root 与 raw
  两套统计，避免新增样本的 tokenizer 输入和生成/评测反归一化混用。
  generation/evaluator 仍使用原版 raw `Interx_{Mean,Std}.npy` 反归一化与官方
  evaluator 输入，两者职责不混用。结果构建器同时强制 augmented
  train-only contact map、`sample_intra_cas/50/0.5/3.5/1.0`，并分别验证
  no-timing arm 未读取 timing、contact arm 只读取冻结 request oracle timing。
  两条 shape arm 还必须等待当前 HHC mask-only v2、TiMotion、CoShMDM、InterMask
  和 Interact2AR 的正式 Inter-X summary 完成。旧
  `tools/run_hhc_shape_aug_historical_paper_mask_arm.sh` 已归档，新的正式训练入口只能
  在 pilot25 门禁通过、全量 optimized materialization 与 hybrid-cache 审计完成后
  生成。届时 random-only 与 optimized arm 均固定 1024-D/16-head/9-layer/2-local、batch 64、
  seed 114514、historical all-valid loss/MultiStep/shape-dropout/38-token 训练合同，唯一
  论文对齐变化仍为分层 mask ratio；训练一律 online W&B 并保存全部 10k checkpoint，
  训练内随机验证副作用也与主 HHC 完全相同，固定为历史诊断的 10-step/temperature
  0.0，避免 shape 消融改变后续训练 RNG；它不作为论文结果。结束后另用官方 544 条
  validation、50-step/temperature 0.5 逐 checkpoint 选点，再按用户固定的
  `sample_intra_cas/50/0.5/3.5/1.0` 跑 test。两臂使用上述 hybrid cache：original
  子集逐字节复用 MMM，只有 augmented 子集使用 deterministic historical-root
  tokenizer；生成反归一化仍使用原版 raw mean/std。机器门禁为
  `output/hhc_shape_aug_training_launch_contract.json`，hybrid 分支与逐子集 provenance
  审计完成前不得转为 pass。新的只读 preflight 已在旧 random-v2 materialization
  上验证真实规模分区：15,902 个 eligible cache name 恰好分成 9,664 个 original 和
  6,238 个 augmented；original 的历史 code/contact 缺失数均为 0，聚合命名内容哈希
  分别为 `e660c236...` 与 `41c20c66...`。该结果只证明 hybrid 方案和 MMM 输入齐全，
  当时尚未证明旧 v2 的完整性；现在
  `output/interx_shape_aug_random_seed123_v2_completion_audit.json` 已逐数组证明它是
  完整、确定性的 random-only 正式对照，可以复用。optimized overlay 生成后仍必须
  重跑同一 `tools/audit_hybrid_shape_cache_preflight.py`。
  最新容量复核确认：若两条 shape arm 各把 30 个验证候选都保存成含 Adam moments
  的全量 Lightning checkpoint，当前仅约 348 GiB 空闲时需要 538.36 GiB，无法启动。
  正式入口现将全部 10k 验证候选改为 Lightning 原生 weights-only；官方 validation
  仍对完整 30 个点选优，`latest/last` 与固定审计点继续保存完整 optimizer 状态用于
  恢复。由当前同架构 checkpoint 实测，weights-only 文件为 `1,656,317,434` bytes，
  完整文件为 `3,943,090,318` bytes；key 集、state-dict key 集和无 optimizer-state
  属性由 `output/shape_capacity/lightning_weights_only_reference.json` 哈希绑定。
  新投影同时计入四套最终 HDF5、两臂 checkpoint、40 GiB 资产/optimizer 预留、
  30 GiB 物理指标预留和额外 40 GiB 尾部余量，当前门槛已通过，预计完成后仍剩约
  50.4 GiB。没有为此删除任何历史 checkpoint 或实验结果。
- CoShMDM 的 GT→262-D preprocessing→672-D adapter 表示审计已实现。精确冻结
  crop 下，通过 terminal-velocity padding 保留完整请求长度，非 root 的 21 个
  身体关节旋转在全部 1,696 个样本、全部有效帧上逐值完全一致。shape-specific
  59-skeleton 全关节刚体 root 拟合的“投影一次→投影两次”总体 RMSE 均值为
  `1.28e-7`、最大为 `1.38e-5`；旧的 hip/shoulder-only root 推断造成的约 20 cm
  二次漂移已被排除。作者预处理仍不可逆地丢弃 sequence-initial absolute root
  transform，且 262-D 输出不含 33 个手部关节旋转。该审计只报告投影幂等性和
  各字段几何 RMSE，用于说明公开表示的可表达范围；不计算 projected-GT、
  representation-controlled 或手部中和 FID。正式主表、消融、门禁和最终报告
  只接受冻结官方 full-space Inter-X evaluator 的普通指标。TiMotion 独立复制的
  262→672 exporter 现与 CoShMDM adapter 做了跨实现运行时审计：float32/float64、
  `[... ,2,262]`/`[...,524]`、shape-free/shape-specific 共八组输出全部 bitwise
  identical，最大误差 0；证据为
  `output/timotion_cosh_adapter_equivalence_audit.json`。
- normalization 修正后的首个 5k checkpoint 在全 1696 样本上得到 MPJPE
  `0.15955 m`，证明米制尺度已恢复；但随后审计发现 5k 可视化导出在
  training mode 下造成了一次额外的单样本 EMA 更新。该 run 因而整体标记为
  `ema_export_contaminated_aborted_7276`，不能作为最终结果。导出路径现已强制
  临时 `eval()` 并恢复原训练模式，replacement 从 update 0 重训。无副作用的
  新 5k checkpoint 已完成同一套 1696 样本审计：MPJPE `0.159550 m`（body
  `0.130510 m`、hand `0.183607 m`、root translation `0.107625 m`），Body/Hand/
  Root 的 active code 为 `3,089/2,287/1,528`，perplexity 为
  `1,846.99/1,671.92/910.62`。与旧 checkpoint 配对比较时，optimizer、scheduler
  和所有非 quantizer tensor bitwise identical；只有三个 codebook 及显式 EMA
  状态不同，因而可将旧差异严格定位到 visualization side effect。该数值仅作
  5k early-training diagnostic。10k 的同协议 MPJPE 已进一步降至
  `0.144245 m`（相对 5k 改善 `9.59%`），body/hand/root translation 为
  `0.117370/0.166559/0.095090 m`；active code 增至 `3,667/2,480/1,785`，
  perplexity 增至 `2,142.53/1,818.72/1,063.12`，未见 early codebook collapse。
  15k 状态 checkpoint 也通过完整性检查：iteration/scheduler 均为 15,000，
  AdamW 有 137 个 state entries，三套 8,096-code EMA 累积量全部 finite，
  reset count 均为 6；SHA-256 为 `c7304e...b255b19c`。该里程碑不重复跑
  全集重建。20k checkpoint 的 iteration/scheduler 也均为 20,000，AdamW
  仍有 137 个 state entries，三套 EMA 状态 finite、reset count 均为 8；
  SHA-256 为 `f34528a4...15176bc`。25k 的全 1,696 样本重建进一步得到
  MPJPE `0.139833 m`（body `0.113115 m`、hand `0.162047 m`、root translation
  `0.090176 m`），比 10k 分别改善 `3.06%/3.63%/2.71%/5.17%`。Body/Hand/Root
  active code 为 `3,552/2,496/1,794`，perplexity 为
  `2,058.17/1,825.57/1,072.08`；三套 EMA finite、reset count 均为 10。
  checkpoint SHA-256 为 `8d2335f1...a4dff4d2`。
  30k checkpoint 继续通过状态审计：iteration/scheduler 均为 30,000，
  optimizer 137 个 state entries 与三套 EMA 状态全部 finite，reset count 均为
  12；SHA-256 为 `23f223ba...38a7118`。该点不重复运行全测试集重建。
  60k checkpoint 的中程状态审计也已通过：iteration/scheduler 均为 60,000，
  model 141、optimizer 411 个浮点 tensor 和三套 8,096-code EMA 状态全部 finite，
  optimizer 仍有 137 个 parameter-state entries；当前 reset-cycle counter 均为 3，
  checkpoint SHA-256 为 `a7198111...0194bc`。训练 loss 约 `0.0755`，未见数值漂移；
  75k 与 100k 审计也均通过。100k checkpoint 的 iteration/scheduler
  均为 100,000，model 141、optimizer 411 个浮点 tensor 全部 finite，
  三路 EMA reset counter 均为 19，SHA-256 为
  `2ebc1af8...a722bb5`。100k 的 loss 为 `0.06797`，Body/Hand/Root
  batch perplexity 为 `776.90/669.72/1021.09`。全集重建仍只保留预先设定的
  稀疏里程碑和最终 150k，避免评测干扰训练。
  最终 150k checkpoint 现已完成：iteration/scheduler 均为 150,000，model、
  optimizer 与三套 EMA 状态全部 finite，SHA-256 为
  `8415d2e9...55fa6695`。固定 1,696 样本重建的 MPJPE 为 `0.143936 m`
  （body `0.114610 m`、hand `0.168328 m`、root translation
  `0.089608 m`），Body/Hand/Root active code 为 `3,381/2,498/1,837`，
  perplexity 为 `2,002.78/1,807.20/1,129.76`。相对 25k，最终 root translation
  改善约 `0.63%`，但总 MPJPE 高约 `2.93%`；该差异如实保留，不把训练末点
  描述成 reconstruction-selected checkpoint。该 8,096-code/150k 分支只保留为
  调查备份，不进入正式 MGT 或最终主表；最终状态/重建证据分别在
  `output/vq_paper_eval/net_150000_state_audit.json` 和
  `output/vq_paper_eval/net_150000_reconstruction.json`。

  当前正式 tokenizer 是用户指定的大清理前 raw-12288 `net_175000.pth`，其
  SHA-256 为 `fdf5afaa...1006bb`，与旧 MMM 主线别名逐字节一致。冻结 1,696
  样本 ordinary-only FID 为 `0.2944172273`、MPJPE 为 `0.1362613624 m`；证据位于
  `checkpoints/vq_precleanup_raw12288/training_manifest.json` 与
  `output/vq_historical_eval/net_175000_reconstruction.json`、
  `output/vq_historical_eval/net_175000_reconstruction_ordinary_fid.json`。论文附录写明
  8,096 code/150k，因此 12,288 code/175k 必须作为已披露的 tokenizer 例外，
  不能误写成逐项一致；论文 MGT 结构、mask、300k 训练和 50-step 正式采样仍按
  当前冻结合同重训。
