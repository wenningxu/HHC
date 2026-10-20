# Inter-X baseline 可运行性与公平协议审计

初次审计日期为 2026-08-26，结果状态更新至 2026-09-05。这里把“论文中报告过 Inter-X”与“公开仓库可以在冻结
Inter-X 请求上直接运行”分开，避免为补表而把不可复现结果当成公平结果。

## InterMask

- 论文主来源固定为 2025-03-02 修订的
  [arXiv v3](https://arxiv.org/html/2410.10010v3)；其 Inter-X 报告 FID 为
  `0.399`。
- 官方仓库为 [gohar-malik/intermask](https://github.com/gohar-malik/intermask)，
  本地固定 revision `5100c555de9839b325d0f3d6904669698c5c87f5`。Inter-X
  motion、text、GloVe、split、evaluator 和 SMPL-X 均直接链接到 MMM 资产；没有重新
  划分数据。正式 test 仍是过滤后的 1,706 条、batch 32 `drop_last` 后 1,696 条。
- 先独立运行作者原生协议：20 个采样步、CFG 2.0、temperature 1.0、top-k threshold
  0.9、20 次 replication。得到 FID `0.3917 ± 0.0143`，接近论文 `0.399`，这同时
  对官方 checkpoint、数据和 evaluator 链路提供了数值校准。需要保留的 provenance
  限制是：这批较早生成的 native HDF5 没有嵌入 checkpoint/full-file SHA-256，因而
  不能把“当时使用当前官方权重”表述成密码学证明；后续 common-50 run 已严格绑定
  当前官方 VQ/Transformer 权重及所有执行源码快照。
- 论文规定 FID/Matching/R-Precision/Diversity 运行 20 次，但 MModality 是唯一
  例外，只运行 5 次。本地 native 备份为降低方差，对 MModality 也保留了全部
  20 次生成；因此其主指标是论文重复数，MModality 是 20 次精度扩展，不能称为
  论文原样的 5 次 MModality 估计。
- 公平主表不是把该数字直接抄入，而是在同一冻结请求上按用户固定的 common-50
  重新生成 20 次，得到 FID `0.3743`。作者原生 20-step 和 common-50 结果分别保存、
  分别命名，不能互相冒充。机器证据为
  `output/fair_eval/intermask_official_exact/native20_hdf5_protocol_audit.json`。

## CoShMDM

- 官方仓库为 [AliManjotho/coshmdm-code](https://github.com/AliManjotho/coshmdm-code)，
  本地固定 revision `c5116ab3bead55051ffec2ec9a161a38d2ab5116`。公开项目页能
  访问，README 现给出正式论文 DOI `10.1109/TVCG.2026.3675725`；但项目页的
  Paper/arXiv 按钮仍被禁用，当前环境也无法取得 IEEE 全文。因此已有论文身份和
  项目摘要，尚没有可逐项读取的完整方法/补充材料，不能声称 paper-exact 复现。
- 项目摘要描述了 contact/shape-aware latent representation、motion/contact 双编码器
  VQ-VAE、body-shape-conditioned diffusion 和 RL+SDF penetration policy。对官方
  revision 的 85 个 tracked 文件及可运行调用链审计表明，release 主链是每人
  262-D motion 的直接 diffusion；没有 VQ/codebook、shape 输入或训练/推理用
  RL-SDF policy。仓库仍含 geometric/contact loss，但它不是摘要所述完整论文架构。
  因此本行已改名为 **public-repository diffusion port**，而不是“官方论文架构”。
  机器证据为 `output/coshmdm_public_release_scope_audit.json`。
- 论文公开摘要列出的 InterHuman/Inter-X headline FID 为 `4.801/0.013`。由于全文和
  supplementary 不可取得，无法核实 `0.013` 的数据版本、evaluator、条件输入和重复
  次数；公开仓库也没有产生该结果的 VQ/shape/RL-SDF 实现或 Inter-X checkpoint。
  因此 `0.013` 只能作为未本地复现的论文报告值，不得与当前官方 Inter-X evaluator
  主表做数值对照，也不得用当前 port 的 `41.4448` 表示“复现失败的论文模型”。
- 公开 release 的 checkpoint 标记为 InterHuman；没有公开原生 Inter-X checkpoint
  或可直接执行的官方 Inter-X 评测命令。本实验只能使用公开 CoShMDM 架构，在与其他
  方法相同的官方 Inter-X train/val/test split 上移植并从头训练。该范围在表中固定
  命名为 **public-repository diffusion port**。
- 已复核官方 `InterXDataset.__getitem__` 的两次人物交换。第二次交换发生在坐标
  对齐之后，但 normalization 对两个人逐人复用同一组 262-D mean/std，而统计重算
  将两人的帧共同累积；因此该交换只重排相同的两块特征，不改变逐维和或平方和。
  当前统计枚举“左右镜像 × 对齐方向”四种配置是完整的，不能把第二次人物交换
  误判成缺失的第八种统计配置，也不需要因此重训。
- 移植保留公开模型的 262-D 双人 body 表示、8 层/8 头/1024-D denoiser、1,000 个
  diffusion steps、DDIM50 eta=0 和 CFG 3.5；训练 normalization 由官方 Inter-X
  train-only 数据重算。训练受用户两日上限约束，在 validation-only 选择 step 40k，
  严格 20 次冻结 test FID 为 `41.4448`。
- 262-D→672-D 适配器只为接入同一官方 evaluator；未建模手部保持 identity。正文
  只报告 GT→模型的 full-672D FID，representation-oracle 分解仅用于诊断为何结果差，
  明确不是 body-controlled FID 或额外对比行。训练、采样和表示范围分别由
  `output/coshmdm_training_contract_audit_live.json`、
  `output/coshmdm_generation_contract_audit.json` 和
  `output/coshmdm_representation_adapter_audit_live.json` 约束。

## Interact2AR

- 官方仓库：[pabloruizponce/Interact2Ar](https://github.com/pabloruizponce/Interact2Ar)，
  本地固定 revision `a0e053be793ee55414dad458b2d62f3ad9f378f7`。仓库公开了
  Inter-X 训练、推理、评测代码及主模型/三套 evaluator 权重；项目说明见
  [Interact2AR project](https://pabloruizponce.com/papers/Interact2Ar)。
- 官方 `train/val/test/all` split 与本项目冻结的 Inter-X split 逐字节一致，
  SHA-256 分别为 `47c5c9...157a`、`6c77e8...28a8`、
  `db32bb...75b`、`449dac...f23`。因此不需要重新划分或映射 sample ID。
- 主模型是作者发布的 mixed-memory checkpoint（epoch 4999、global step
  350,000、autoregressive），大小 `3,522,226,119` bytes，SHA-256
  `543a33b619b2bf175f5a99aacf2327350c65e4c99ae03c3be25911ed1f0fe997`。
  `assets/checkpoints.json` 的哈希与实际文件严格匹配。
- 用 `uv sync --frozen` 建立了作者锁定环境；实际版本为 PyTorch
  `2.9.1+cu128`、Lightning `2.5.2`、NumPy `1.26.4`、h5py `3.14.0`。
  Inter-X motion/text、GloVe 与 SMPL-X 模型来自本地 MMM 资产，通过符号链接
  接入，未复制或改写数据。仓库的 tracked diff 为空；`git status` 的 dirty
  标志只来自这三个数据链接，并由 run manifest 明确记录。
- 官方 README 的主评测命令带 `--first_frame`，这会给模型一帧 GT motion；
  HHC、InterMask 和 CoShMDM 的公平协议都没有此条件。公平主表因此明确禁用
  GT 首帧，采用作者 CFG `2.5`、自回归 ancestral 10-step sampler，并按冻结请求
  的精确长度分组生成。作者原始 `--first_frame` 结果只能另列为条件生成审计，
  不能混入无条件首帧的主表。
- 论文正文明确要求全部评测重复 20 次并报告 95% CI，但当前官方
  `src/evaluate.py --full` 固定为 5 次。因此先原封不动运行 5 次作者发布协议校准，
  再运行单独命名的 20 次论文/公平协议，禁止将二者的均值或置信区间混合。
- 三方机器审计已固化为 `output/interact2ar_paper_repo_data_audit.json`：官方
  revision、核心源码、模型和三个 evaluator 权重哈希均通过；train/val/test/all
  分别为 9,110/570/1,708/11,388，HHC 与 Interact2AR 的长度过滤均留下 1,706 条，
  `drop_last=True` 后每次实际评测 1,696 条。Inter-X 下载未发布大 H5 的 SHA256，
  所以审计只声明本地三仓同 inode/同哈希、key/shape/dtype 与官方 split 一致，
  不声称已密码学证明远端下载文件字节身份。
- 训练协议也存在发布示例陷阱：论文为 global batch 128，发布 YAML 为每设备 32，
  README 却示例单卡 `[0]`。作者 AR 过滤后 train 为 9,082 条，global batch 128
  每 epoch 正好 70 step，5,000 epoch 对应官方 checkpoint 的 350,000 global step；
  单卡 batch 32 会是 283 step/epoch，不能复现该 checkpoint。若权重不可用而需重训，
  只能用 4×32 DDP 或明确等价的有效 batch 128。
- `tools/export_interact2ar_official_interx.py` 只做 I/O 和协议适配，模型、权重、
  sampler 与 motion representation 均来自作者仓库；输出保留完整 672-D、双人、
  含手部的 SMPL-X 表示。`tools/run_fair_interx_method.py --method interact2ar`
  将其接入与其他方法相同的 20 replication、固定 100×30 MModality 请求、官方
  Inter-X evaluator 和独立物理指标。
- 单样本执行验证已通过：`G039T006A021R010` 输出形状
  `(1, 1, 150, 672)`，所有值 finite、completion flag 为真、请求长度/名称一致，
  metadata 为 `first_frame_conditioning=none` 和
  `author_autoregressive_ancestral_10_steps`。该 smoke HDF5 的 SHA-256 为
  `107ad3b5bd07fc4e512b09a699cb902bd7c3f54106e9e7d3b13ecb421de7046f`；它只证明
  执行路径有效，不作为质量结果。
- Interact2AR 不接收冻结请求中的 subject beta/shape。公平物理指标会像处理其他
  不具备相同 shape 接口的 baseline 一样，用请求 beta 对生成 rotations 做统一
  retarget；主表必须注明这一点，不能将其描述成 shape-conditioned 模型。

## TiMotion

- 论文主来源固定为 2026-03-10 修订的
  [arXiv v5](https://arxiv.org/html/2408.17135v5)，而不是旧审计中暂记的 v4。
  v5 Table 7 仍明确采用 Inter-X test、20 次运行和 95% CI；论文统一采样设置为
  DDIM-50、`eta=0`、CFG `3.5`。
- 官方仓库：[AIGC-Explorer/TIMotion](https://github.com/AIGC-Explorer/TIMotion)，
  本地固定 revision `172580fd7f2551a7cc10fde1f15ff73315742e65`；项目页为
  [TIMotion](https://aigc-explorer.github.io/TIMotion-page/)。
- 当前公开 revision 只有 `configs/datasets.yaml` 中的 `interhuman` loader，
  `tools/train.py` 和 `tools/eval.py` 也固定选择该配置。公开输入是 InterHuman 的
  22-joint、262-D body 表示；仓库没有 Inter-X split/loader、SMPL-X 56-joint
  转换、Inter-X normalization stats、Inter-X checkpoint 或与本项目 evaluator
  对接的 exporter。
- 公开 `models/blocks.py` 实际启用的是 Transformer；Mamba/RWKV import 被注释，
  仓库也没有相应可运行 block。补充材料中 Inter-X 的 Transformer/Mamba/RWKV
  FID 分别为 `0.385/0.348/0.261`，因此本地能够公平重训的项目只能明确命名为
  **public TIMotion+Transformer+LPA**，不能把它冒充论文最佳 RWKV 行。
- Inter-X 移植现已完成：`data/InterX` 只读链接到由 MMM 官方 HDF5 转换出的
  9,110-ID train/570-ID val/1,708-ID test 数据；split 哈希与其他方法一致；
  normalization 复用并哈希审计过的 `interx_global` 262-D train-only stats。
  `tools/train_interx_fair.py` 保留论文/发布代码的 5-layer、16-head、512-dim、
  LPA、AdamW `1e-4/2e-5`、1,000 diffusion steps、10-epoch warm-up、1,500 epochs、
  DDIM50/CFG3.5。单卡 micro-batch 32×8 模拟论文 global batch 256；每 epoch
  只用前 568 个 micro-batch，即 71 个 update/18,176 个样本，与八卡完整 global
  batch 的边界丢弃相同。训练现已完成，并由最终 checkpoint、W&B history、
  official-validation selector 和 20 次冻结评测共同审计。
- 论文 Sec. 4.1.1 明确写的是所有 denoiser 训练 1,500 epochs；公开
  `train.py` 的默认 1,900 与 README 中 `--LPA` 命令没有显式 epoch 的组合与论文
  冲突，因此公平重训以论文 1,500 为准并披露该差异。启动前强审计已通过：
  `output/timotion_interx_training_contract_preflight_audit.json` 逐个检查了 11,388
  个转换 pair、官方 `all.h5` ID、train-only stats 和完整传递源码。唯一短于作者
  loader 15-frame 下限的样本在 test（`G021T003A016R007`），train 为 0 条，故
  18,220 个 mirrored train entries 和 106,500 updates 不受影响。正式 runner 还
  要求断点 checkpoint 位于完整 epoch 边界，并由 `global_step` 恢复 micro-step；
  代码或数据哈希漂移时拒绝续训。50-update 的 online-W&B smoke 已完成且远端
  metric-history audit 通过（run `oi4wa5qn`）；正式 1,500-epoch run 已于
  2026-08-31 启动。首个 formal run `jqh616am` 在 warm-up 检查时被人工中断且
  没有保留 checkpoint，远端已标为 failed/结果不可用；替代 run `kdz5s3ra`
  从 optimizer step 0、相同已审计合同启动并完成；远端 W&B、最终 checkpoint
  与 validation-only step 42,600 选择记录均已冻结。
- `tools/export_timotion_official_interx.py` 已接入冻结 request：按精确长度分组、
  无 GT 首帧、DDIM50 η=0/CFG3.5，将作者 262-D 输出通过与 CoShMDM 相同的
  root/body adapter 转为 672-D，未建模手部固定 identity。随机输入上的两份
  adapter 输出逐元素 bitwise identical（最大绝对差 0）。
- 原发布 loss 在所有时间帧上先归一化 hip-to-hip orientation、之后才乘 padding
  mask；Inter-X padding 的两个人 hip 重合，导致只出现在无效帧的 `0/0` NaN 仍以
  `NaN*0` 污染 RO loss。公平端口只对 mask 外向量代入固定 x 轴再归一化；有效帧
  prediction/target 与 loss 逐元素 bitwise 不变。修复、坏 smoke 中的非有限输出和
  修复后 finite checkpoint 均冻结在
  `output/timotion_padding_ro_stabilization_audit.json`。这是一项 padding 数值稳定化，
  不改变有效 support 上的作者目标函数。
- 从头重训和验证选点现已完成：仅用官方 Inter-X validation 选择 step 42,600
  （validation FID `36.9030`），冻结 20 次 test 得到 FID `33.3584`。该结果只命名为
  public Transformer+LPA Inter-X port；262-D 表示限制由非指标的几何适配器审计说明，
  不另算 projected-GT 或 body-controlled FID。作者公开论文数值仍只能作为非统一
  协议参考，不能填入严格结果列冒充本地复现。

## 可报告状态

| 方法 | 同一官方 Inter-X split | 同一冻结请求/evaluator | 当前可执行状态 | 主表资格 |
|---|---:|---:|---|---|
| InterMask | 是 | 是 | 官方 checkpoint、native-20 与共同 50 步×20 次均完成；common50 FID `0.3743` | 是 |
| CoShMDM | 是 | 是 | 用户两日上限内训练已结束；val 选择 40k，common50 FID `41.4448` | 是，但仅声明公共代码 Inter-X 移植 |
| Interact2AR | 是 | 是 | 官方权重无 GT 首帧 20 次已完成，FID `2.4443`；GT-first-frame 与新 joint evaluator 校准待运行 | 是，须注明无 shape 输入和条件协议差异 |
| TiMotion | 是（本地审计移植） | 是 | public Transformer+LPA 从头重训、val 选点及 20 次评测均完成，FID `33.3584` | 是，但不能冒充论文 RWKV 行 |
