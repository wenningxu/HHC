# HHC 论文实现一致性审计

本表将论文正文/附录中的可核验声明与当前重训练路径逐项对应。判定中的
“论文一致”只用于论文明确给出的内容；未公开的选择单独标为“发布代码派生”或
“历史配置派生”，不把二者混写成论文超参数。

| 项目 | 论文声明 | 当前实现与证据 | 判定 |
|---|---|---|---|
| 两人 tokenizer | 联合编码完整双人 interaction，拆分 Root/Body/Hands | `VQVAE_SEP3_shape` 使用三套双人 encoder、quantizer 和 decoder 分支 | 论文一致 |
| latent / codebook | Body、Hand 为 96 维，Root 为 64 维；每部分 8,096 codes | 正式冻结 VQ 保留 `96/96/64`，但使用大清理前主线的 12,288 codes；checkpoint SHA-256 `fdf5afaa...1006bb` | latent 维度一致；codebook 是用户指定、明确披露的历史主线例外 |
| 时间压缩 | 下采样到原长度的 1/4 | `down_t=2, stride_t=2` | 论文一致 |
| VQ 卷积细节 | 论文未报告 width/depth/dilation/activation/norm | 发布实现为 width `512`、depth `3`、dilation growth `3`、ReLU、无 normalization | 发布代码派生，已披露 |
| VQ 训练 | 150k、batch 256、AdamW、LR `2e-4`、WD `1e-6`、1k linear warm-up + cosine | 正式 run 不重训 VQ，导入历史 raw-12288 的 175k checkpoint；其原 W&B 配置为 batch 256、LR `2e-4`、WD `2e-5`、计划 300k | 非逐项论文复现；固定 tokenizer 例外，重建质量单独审计 |
| AdamW betas | 论文未报告 | 当前论文路径使用 `(0.9, 0.99)`；大清理前双人 MGT 的 `utils/utils_model.py` 使用 `(0.5, 0.9)`，两者分开冻结 | 发布代码派生且属于旧/新真实差异，已披露 |
| VQ crop | 论文未报告 | 导入 checkpoint 的历史 W&B/source provenance 单独冻结；旧 96-frame 复现分支仅作备份调查 | 历史配置派生，已披露 |
| VQ precision / seed | 论文未报告 | 导入 VQ 的历史 seed 为 `123`；本次不产生新的 VQ precision/训练轨迹 | 历史配置，已披露 |
| VQ codebook 更新 | 论文未报告 EMA/reset 细节 | 发布实现使用 `ema_reset`、decay `0.99`，每 21 updates 检查并重置持续未使用 code；完整非-buffer 状态进入 checkpoint | 发布代码派生，已披露 |
| VQ loss 权重 | 论文未报告分项系数 | 从匹配 MMM run config 恢复 `3/100/0.1/0.01/10/0.1` 和 commitment 外权重 `0.02` | 历史配置派生，已披露 |
| MGT 主干 | 9 blocks、8 heads、512 dim | `--paper-architecture` 构造 9 个 `AdaCrossSelfBlock(512, 8)` | 论文一致 |
| MGT 未报告细节 | 论文未报告 FF expansion、dropout、CLIP variant 和最大 token 数 | 发布实现保留 FF×4、dropout `0.1`、CLIP ViT-B/32、最多 50 motion tokens；均写入 manifest | 发布代码派生，已披露 |
| block 顺序 | 每个 block 先 word-level CLIP cross-attention，再 motion self-attention | `AdaCrossSelfBlock.forward` 严格执行 cross → self → MLP；motion token 为 query，word embedding 为 key/value | 论文一致 |
| 全局条件 | global CLIP 与两人体型/性别通过 AdaLN 调制两个 attention layer | `paper_condition_mlp([clip, shape, gender])` 驱动 cross/self 两套 AdaLN；CFG 仅丢弃 text、保留请求 shape | 论文一致 |
| token stream | 每帧 Root、Body、Hands 交错 | 模型把 tokenizer 的 `[Body, Hand, Root]` channel 显式重排为 `[Root, Body, Hand]` interleaved stream，并加入 part/time embedding | 论文一致 |
| token classifier | 论文未说明三个 codebook 是否使用参数独立的输出 projection | 发布/历史 HHC 实现将同一 `Linear(512, K)` 作用于 Root、Body、Hand 三类位置，得到三个逻辑 `[B,T,3,K]` 输出；它不是三套参数独立 head | 发布代码派生；若正式 FID 持续异常，列为结构消融而非冒充论文明示配置 |
| 训练 mask | 三个独立 `tau` 排序为 `tau_root > tau_body > tau_hand`，`gamma(tau)=cos(pi*tau/2)` | `sample_kinematic_cascade_mask` 按 channel 映射排序并分别随机选位；Root mask ratio 最小、Hand 最大 | 论文一致 |
| mask 数量离散化 | 正文未说明 fractional token count 如何取整 | `round(valid_part_length * gamma)`，非空 part 至少 mask 1 个；与原 HHC/MMM 和 InterMask 源码约定一致，并有精确 count 回归测试 | 源码约定，非论文显式规格 |
| 训练目标 | 只最小化 masked token NLL | `masked_token_nll` 只聚合 mask 内 token；mask 外 END/PAD 会先替换为安全 target，不能污染 loss 或触发越界 | 论文一致 |
| masked NLL reduction | 论文 Eq. 3 未说明变长 batch 中先按 sample 还是按全体 masked token 归一化 | 当前实现先对每个 sample 的 masked token 求均值，再对 batch 求均值；InterMask 发布实现则先用 `where(mask, target, ignore_id)` 仅保留 masked label，再由默认 `cross_entropy` 对 batch 内全部非 ignore token 作一次全局均值。审计冻结官方 InterMask revision `5100c555...c5c87f5`、`transformer.py` SHA-256 `652f5639...89379` 与 `tools.py` SHA-256 `fd6bb95b...9f36`，并逐条验证上述调用链；数值差异记录于 `output/hhc_masked_loss_reduction_audit.json` | 未报告实现细节；正式 FID 持续高时作严格单变量消融 |
| contact 输入 | 训练不把 contact signal 作为网络条件 | 主训练命令为 `--contact --no-contact-conditioning`；contact 仅用于构建 train-only code map | 论文一致 |
| MGT 训练 | 300k、batch 96、同一 LR/schedule/WD、AdamW | 训练命令与附录相同；固定 VQ 与 CLIP 不进入 optimizer | 论文一致 |
| MGT 最短 token bucket | 论文未报告 | 发布 dataset 从 20 个 latent token 起训；当前冻结 cache 实际对应 8,606 train / 534 val active entries。MMM 历史 cache 为 8,617/534，11 条 train 差异来自重建 cache 的确定性 crop/token 长度，而非 split 变化 | 发布代码派生，已披露 |
| MGT cache crop | 论文未报告 | 保留发布 MMM 的 `single/single/double` 长度分布；以 `(seed, sample_name)` 独立确定裁剪，策略版本写入 manifest，断点续建不改变 token | 发布代码派生并做确定性加固 |
| MGT precision | 论文未报告 | 当前路径与相关的大清理前双人 `train_t2i_trans.py` 均使用 BF16 mixed precision；它不是旧/新轨迹差异 | 发布代码派生，已披露 |
| CFG 训练 dropout | 论文只报告 inference scale `3.5`，未报告 unconditional dropout | 发布 HHC 模型以 `0.1` 概率同时置零 global/word text embedding，shape/gender 保留；写入 training manifest | 发布代码派生，已披露 |
| 初始状态与迭代数 | 全 masked；Fig. 2 画到 Step 10，但正文/附录未显式声明 `S` | 用户指定的共同公平主表使用 `S=50`。JetBrains Local History 与两次 W&B 时间区间交叉绑定后，已恢复大清理前真实运行源码：`sample_intra_cas` 默认 10 个 outer steps，每步 Root→Body→Hands，共 30 次 channel updates。稍后的 Git 快照默认 3 outer/9 updates 只保留为快照参数敏感性；Fig.-2 另作 10-step 参数审计 | 论文、历史运行、稍后快照与共同预算分开披露 |
| 级联顺序 | 每一步 Root → Body → Hands；当前部位更新立即用于下一部位 | 每个 step 内依次重新 forward 三次，并原位写回 Root、Body、Hand | 论文一致；有回归测试 |
| remask | `floor(cos(pi*s/2S)*L)` 个最低置信 token 保持 masked，高置信 token 固定 | sampler 对当前 masked 位置按置信度确定性排序；已固定 token 不会被后续步骤覆盖。恢复出的 MMM 开发实现会重新覆盖此前 fixed token，因此其真实 10-outer runtime 与稍后 3-outer Git 默认都只作历史/参数审计，不作为论文主路径 | 论文一致；开发差异显式披露 |
| 随机采样 | temperature `0.5`、CFG `3.5` | exporter、validation 与 test hook 均显式使用 temperature 0.5、CFG 3.5；共同公平主表 50 steps。恢复出的 2026-02-17 test 为 `rand_pos=True`/temperature 1.0/CFG off，2026-03-05 test 为 `rand_pos=False`/temperature 0.5/CFG 3.5；二者均调用默认 10-outer `sample_intra_cas`。这些历史设置不覆盖当前论文/公平协议 | 论文参数与逐次历史运行证据分开报告 |
| contact guidance | `logits + lambda*log(p+epsilon)`，`lambda=1.0`；map 是 empirical contact probability | map 仅由 9,095 个 train 样本统计；观测 code 用精确频率，未观测 code 置中性 `0.5`；采样权重恒为 1.0；论文未给数值的 `epsilon` 保留发布值 `1e-6` | 公式/权重论文一致；未见 code 与 epsilon 的数值约定已披露 |
| checkpoint 隔离 | 论文未讨论 | VQ checkpoint 保存完整 EMA 非 buffer 状态；可视化强制 eval；MGT checkpoint 不能覆盖独立指定的 VQ | 复现完整性加固 |
| optimization schedule | 论文 optimization 描述不能覆盖原开发仓库的实际实现 | 当前生产核心已恢复为原 MMM：Adam、LR `0.01`、默认 200 iterations、iteration 100 尝试冻结满足 5 cm 与 opposing-normal 条件的 one-to-one anchors。`optimize_full_body()` AST 与 `/home/wenning/projects/MMM/optimize_full_body.py` 完全相同；`optimization/losses.py` 逐字节 SHA-256 为 `29f67e...51f7c`。此前自行改成 `100/0.02/50` 的实现和产物均标为 invalid，不能进入实验 | 原 MMM 开发实现；论文差异只作披露，不再擅改核心 |
| penetration threshold | 论文公式不充分确定实现细节 | 当前生产路径保留原 MMM 的非交互/交互阈值 `0.2/0.5 m` 及按 penetrating vertex 数归一化的优化 loss；此前 `0.05/10 m`、未归一化等改写已隔离到 `experimental_invalid` 文件 | 原 MMM 开发实现；不宣称论文唯一实现 |
| signed-distance backend | 论文写 signed distance，但未要求具体库 | 当前优化严格使用原 MMM CUDA Chamfer nearest-vertex 加目标顶点法向符号；VolumetricSMPL 只用于所有方法共同的只读 mesh/physical evaluator，不再参与 augmentation optimizer | 原 MMM 开发实现；统一物理评测另行报告 |
| LLM model / prompt | Gemini-3-flash；四列输出为两方 body part、Contact Area、Role Ratio | active 全量标注使用 ChatGPT 登录态 `gpt-5.6-luna`/low；prompt 删除论文没有的 Normal Alignment，closed JSON schema 拒绝未知部位、组合 contact type 与未允许 ratio。Gemini 仅保留为可选论文模型/鲁棒性对照 | prompt/schema 对齐；模型替换明确披露，不冒充论文 Gemini |
| LLM cache provenance | 论文未报告缓存格式 | 旧 570-file pickle cache 没有 raw response/model/prompt hash/parse status/retry/tokens；345 文件不符合当前 schema，不能用于正式表。Codex v3 extractor 保存 batch raw response、session/token/retry 与逐样本 caption/request/implementation hashes，schema error fail-closed，无 API key | 复现加固；旧缓存无效 |

## 自动检查

运行：

```bash
PYTHONPATH=. conda run -n New_MMM python tests/test_paper_consistency.py
```

The active full nine-block/8-head/512-dim paper MGT produces three logical
12,288-way channel outputs through the released shared classifier projection,
bound to the fixed raw tokenizer.  Earlier 8,096-way CUDA smoke results are
retained only as implementation diagnostics, not as evidence for the active
checkpoint.

当前测试套件覆盖 mask 层级、masked-only NLL、9-block 模型、级联
即时更新、150-frame contact pooling、EMA 状态恢复、观测式导出、train-only
contact provenance、MGT checkpoint 保留指定 VQ，以及 main/MM 生成 provenance
必须来自同一模型和采样合同；optimization 的有效回归改为锁定原 MMM loss 文件
哈希与 `optimize_full_body()` AST，并拒绝任何生产入口引用
`losses_experimental_invalid.py` 或 `optimize_full_body_experimental_invalid.py`；同时检查
LLM schema fail-closed、冻结 caption、逐 replication 的
contact-timing source/path/hash、Holm 多重比较和各指标的 paired direction，
以及最终报告对 failed/wrong-step 训练审计的拒绝。VolumetricSMPL 的真实权重执行
只约束统一 mesh/physical evaluator，不是优化器一致性证据。
正式 MGT launcher 还会在建 cache 和训练前执行
`tools/audit_hhc_training_mask_runtime.py`：以固定 seed 实际采样 4,096 组可变长度
mask，检查 `tau_root>=tau_body>=tau_hand`、Root/Body/Hand mask-ratio 顺序、
`round(length*ratio)` 离散计数和 masked-only loss 对 visible logits 的不变性；
证据写入 `output/hhc_training_mask_runtime_audit.json`，任一检查失败即停止训练。
同一启动门禁还实际构造论文 MGT 并执行一次 forward，验证 9 个 block 都严格按
word cross-attention→motion self-attention 顺序运行、8 heads/512 hidden、没有
额外 legacy output blocks、Root/Body/Hand 交错输入以及 `[B,T,3,V]` 有限输出；
证据为 `output/hhc_model_structure_runtime_audit.json`。

## 当前训练状态

raw-12288 VQ 已按字节级 checkpoint、官方 1,696-request reconstruction 和官方
evaluator 闭环审计，不再重训。绑定该 VQ 的 train+val token cache 与 train-only
contact map 已完成（9,664 cache samples；9,095 train-only map samples）。修正后的
9-block/8-head/512-dim MGT 于 2026-08-30 16:08 BST 启动 300k 重训；正式结果只接受
validation-FID selection manifest 绑定的 checkpoint。旧 8,096/150k VQ 分支仍作为
调查备份保留，但不进入正式表。

50k 固定 validation FID 为 `0.886226`，从 40k 的 `0.9752653` 继续改善并成为
当前最低值（10k `0.9789292`、20k `1.2168187`、30k `1.0176039`）；该节点
Top-1/2/3、Matching、Diversity 分别为
`0.45037/0.64154/0.72794/3.55797/9.0854`。作为诊断参照，大清理前低-FID
legacy topology 在 40k/50k 的 FID 分别为 `0.3779595/0.2442706`，但其 40k
Top-1/2/3、Matching 仅为 `0.39505/0.57252/0.69222/3.78005`。因此当前模型
已经学习到更强的文本检索语义，异常更集中在生成分布、采样校准、训练目标或拓扑，
而不是 evaluator、VQ 重建或文本条件完全失效。50k checkpoint SHA-256 为
`7e41b0...37f8`；global step `50,000`、epoch `561`、LR
`1.8703700675e-4`，947 个模型 tensor 与 1,041 个 optimizer tensor 全部 finite。
完整状态审计为 `output/hhc_formal_raw12288_repaired_mgt_50k_checkpoint_audit.json`；
10k/20k/30k/40k/50k checkpoint 均步数精确，训练源码 diff hash 仍为启动时的
`6369cd...7bf`。固定 64 条 train/validation token 的 mask-exposure 诊断显示
明显 exact-sequence 泛化差距；但历史低-FID MGT 在同一诊断上也呈现约 90%+
training accuracy、约 1% validation accuracy，因此该 token 命中率不能解释或
替代生成 FID。另一个独立审计按真实 `>=20` token bucket 与 49-token payload
重新确认，validation 中训练未见 token occurrence 占比仅为 Body/Hand/Root
`0.438%/0.0292%/0.117%`，边际 code 支持集不匹配仍不足以解释异常。条件组合
支持审计进一步显示，各分支 validation token 的训练集边际覆盖为
`99.56%/99.97%/99.88%`，同帧 `(Root, Body)` occurrence 覆盖仅 `7.833%`，完整
`(Root, Body, Hand)` occurrence 覆盖仅 `0.268%`；Body/Hand/Root 的相邻时序
转移覆盖分别为 `69.65%/58.77%/73.87%`。这说明近零 validation exact-token
accuracy 可由高基数 code 的新条件组合解释，并支持“训练组合记忆/组合泛化差距”
这一描述；但历史低-FID MGT 使用同一 token 数据且也呈现该 exact-token gap，
因此该数据属性不能单独解释当前与历史 FID 的差异，也不被当作新训练方案的因果
证据。结果冻结在
`output/hhc_formal_mgt_train_val_conditional_support_audit.json`。当前继续以
50k 及后续官方 generation FID 为门槛；若轨迹持续异常，按顺序执行同 checkpoint
采样校准、global-masked-token loss reduction 与 legacy-topology matched control，
不在活跃正式 run 上原位改源码。采样校准只使用与训练 selection 完全相同的
544 条 validation 记录，所有行保持 50 outer steps；正式 `temp=0.5/CFG=3.5`
行必须在 `1e-5` 内复现 selected checkpoint 的训练期 validation FID，否则先判定
离线 exporter/evaluator bridge 失配并停止解释参数网格，绝不使用 test 指标调参。
由于原始 Inter-X 训练记录把每条 motion 固定绑定到一个连续 body shape，后续链路
还会对同一 checkpoint、文本、token 与 Root→Body→Hands teacher-forced context 做
train/validation 成对 shape 置换：requested shape、beta+gender 签名严格不同的
donor shape、zero-beta 三种条件只改变 shape 输入；每个 split 从 active loader
集合固定 seed 均匀无放回抽 256 条，避免按 `Gxxx` 排序造成 actor-pair 偏置。
预注册门槛要求 requested 相对
置换 shape 在训练集至少提升 5 个 accuracy percentage points，且该优势相对验证集
再高至少 3 points，才记为 `train_only_shape_identity_shortcut_supported`。这项诊断
只用于判断随机-shape augmentation 的机制，不代替 decoded-motion FID；实现与汇总
分别为 `tools/audit_mgt_shape_condition_dependence.py` 和
`tools/summarize_mgt_shape_condition_dependence.py`。

50k loss-control 的后续动作不是人工观察日志后临时决定。评估器预注册绝对 FID
改善至少 `0.1` 且相对改善至少 `15%`；
`tools/dispatch_hhc_post_50k_training.py` 将 assessment 与上述 shape 诊断按 SHA-256
绑定，并只允许三种结果：(1) 达到 loss 门槛时由
`tools/continue_hhc_global_masked_loss_control_300k.sh` 在同一 checkpoint/W&B lineage
原地续训至 300k 并运行 common-50；(2) loss 不足但 train-only shape shortcut 成立时
保留 `paper_masked_only` 并继续 matched shape augmentation；(3) 两项均不成立时在
mask-conditioning 调查门失败关闭。50k/300k 任一阶段被重启打断时，恢复前必须审计
checkpoint step、模型/优化器有限性和 weight decay，并复用原 W&B run id。
