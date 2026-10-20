# InterHuman evaluator 特征敏感性与 baseline 目标核查

2026-10-06。范围：released InterMask InterCLIP、InterMask 与官方 InterGen；不修改当前训练，不使用 GPU。按 ponytail 复用官方 motion encoder 和已有数据适配器，仅新增一个 CPU 探针及结果文件。

## 结论与证据强度

- **已由代码与实测确认**：evaluator 使用双人的位置、逐帧位置差分速度、21 个关节的 6D 旋转；每人最后 4 个脚接触特征被直接删除。它不是位置专用 evaluator，也不是 SMPL mesh/接触质量 evaluator。
- **已由训练代码确认**：InterGen 与 InterMask 都重建完整连续表示，并施加多项位置派生约束；不能概括为“只优化位置”。InterMask 的 MGT 阶段主要优化离散 token CE，而不是直接优化位置或 InterCLIP。
- **本次局部探针支持**：在 4 条 validation 前缀、等标准化总扰动能量下，旋转噪声的 embedding 响应约为位置噪声的 1.3 倍，速度噪声约为位置的 1/20。因此“旋转和速度共同主导 evaluator”目前没有得到支持；速度部分的初步证据方向相反。
- **尚未证实**：全分布特征重要性、对任一生成方法的系统性不公平、某个 FID 差距由旋转误差造成。随机特征噪声会破坏运动一致性，局部敏感度不是自然生成误差贡献，更不是方法排名的因果解释。

## 1. evaluator 实际读取什么

每个人 262D，双人按人拼接。切片均为 Python 左闭右开。

| 切片 | 维数 | 含义 | 进入 InterCLIP |
|---|---:|---|---|
| `[0:66]` | 22×3 | 共同参考系的全局关节位置，含 pelvis/root 位置 | 是 |
| `[66:132]` | 22×3 | `position[t+1]-position[t]`，单位为每帧位移，未乘 FPS | 是 |
| `[132:258]` | 21×6 | 原数据中的非 root 关节 6D 旋转 | 是 |
| `[258:262]` | 4 | 左/右脚跟脚尖接触标记 | **否，显式删除** |

证据：[官方数据预处理](/mnt/8T-HDD/wenning/InterMask/data/utils.py:124)，其中 185–198 行拼接位置/速度/旋转/接触；[原数据切片](/mnt/8T-HDD/wenning/InterMask/data/utils.py:53) 提取 22 joints 与 21 rotations；[evaluator 删除接触](/mnt/8T-HDD/wenning/InterMask/models/evaluator/evaluator_models.py:159)。root 平移已包含在位置及速度中，身体朝向也可由骨架位置推断；不能将“无独立 root rotation 特征”解释为“看不到朝向”。手指、shape 参数、表面穿插与 mesh 没有直接通道。

预处理先变坐标轴、落地、以初始 pelvis 的 XZ 平移及朝向确定参考系，再将第二人对齐到第一人；局部旋转数组在此过程中沿用输入，不能自行对所有 6D 通道施加世界坐标旋转。[共同对齐](/mnt/8T-HDD/wenning/HHC/dataset/interhuman_hhc_adapter.py:245)、[rigid transform](/mnt/8T-HDD/wenning/InterMask/data/utils.py:200)。这也要求后续比较固定 anchor、人序、裁剪与地面协议。

**归一化边界**：VQ 的输入为 `(x-mean)/std`；生成结果在送 evaluator 前恢复 `x*std+mean`。HHC 对 GT 和生成统一做一次反归一化：[wrapper](/mnt/8T-HDD/wenning/HHC/models/interhuman_evaluator_wrapper.py:109)。官方 InterMask 生成路径同样反归一化：[evaluator.py](/mnt/8T-HDD/wenning/InterMask/models/evaluator/evaluator.py:91)，GT 在正式评分改为 `normalize=False`：[eval.py](/mnt/8T-HDD/wenning/InterMask/eval.py:271)。InterCLIP motion encoder 内部没有逐特征 z-score，只有网络层内的 LayerNorm。故不能在进入 evaluator 时再 z-score，也不能将所有模态加相同原始幅度的噪声视为公平比较。

编码器为 `Linear(516,1024)`、query token、时间位置编码、8 层/8 头 Transformer，取 query 输出后投影为 512D，L2 normalize 后乘学习到的 `latent_scale`。[架构](/mnt/8T-HDD/wenning/InterMask/models/evaluator/evaluator_models.py:130)、[配置](/mnt/8T-HDD/wenning/InterMask/checkpoints/eval_model/eval_model.yaml:1)、[输出归一化](/mnt/8T-HDD/wenning/InterMask/models/evaluator/evaluator_models.py:103)。训练目标为同 batch 文本/运动的双向对比 CE：[loss](/mnt/8T-HDD/wenning/InterMask/models/evaluator/evaluator_models.py:62)。该目标没有保证对三类特征同等敏感，也没有直接对齐 MPJPE 或物理接触质量。

官方 InterGen 的同构 motion encoder 也明确删除末 4 维：[固定 commit 的 nets.py L19–69](https://github.com/tr3e/InterGen/blob/b1750df4a859443b8a69329f759ce3a7503ae179/models/nets.py#L19)。结构相同不等于本次证明两仓库 checkpoint 字节相同：本次实际测的是本地 InterMask released checkpoint。

## 2. baseline 究竟优化什么

### InterMask：本地 release 核实

本地仓库 origin=`https://github.com/gohar-malik/intermask.git`，HEAD=`5100c555de9839b325d0f3d6904669698c5c87f5`。参数以随 checkpoint 的 opt.txt 为准，而非仅看 parser 默认值。

VQ 对各人使用 `22×12` 的 position/velocity/rotation 网格，root rotation 填零；接触不进网格，解码返回的 4 contact 通道也填零。[model.py](/mnt/8T-HDD/wenning/InterMask/models/vq/model.py:63)。这是 **encoder/decoder 没有建模接触标签**，但仍可由输出位置计算物理接触误差。

| VQ 项 | 权重 | 实际计算 |
|---|---:|---|
| reconstruction | 1 | 标准化后的 258D SmoothL1，排除 contact |
| commitment | 0.02 | VQ commitment |
| explicit | 1 | 标准化位置 SmoothL1 |
| velocity | 100 | **位置块的时间差分** SmoothL1，非单独速度块损失 |
| bone length | 5 | 由输入位置块算骨长 |
| geodesic | 0.01 | 旋转块转矩阵后计算角距离 |
| foot contact | 500 | 反归一化后、GT 判定接触处的预测脚速度 |

证据：[released opt.txt](/mnt/8T-HDD/wenning/InterMask/checkpoints/interhuman/vq_default/opt.txt:18)、[加权总式](/mnt/8T-HDD/wenning/InterMask/models/vq/vq_trainer.py:46)、[各项切片](/mnt/8T-HDD/wenning/InterMask/models/losses.py:87)。`loss_geo` 实现直接接收标准化旋转块，`bone` 也直接接收标准化位置，只有 foot-contact 显式反归一化；应按实现解释，不能把各权重当成同单位的物理误差权重，或据 500/0.01 推断梯度贡献比。

MGT 先以固定 VQ 得 token，再预测 masked token；含 step-unroll CE（released `step_unroll=1`）。[trainer](/mnt/8T-HDD/wenning/InterMask/models/mask_transformer/transformer_trainer.py:49)、[CE 总式](/mnt/8T-HDD/wenning/InterMask/models/mask_transformer/transformer.py:362)、[MGT opt](/mnt/8T-HDD/wenning/InterMask/checkpoints/interhuman/trans_default/opt.txt:26)。训练调用没有把 InterCLIP embedding loss 直接回传给生成器。

[InterMask 论文 v2 §3.1.1](https://arxiv.org/html/2410.10010v2#S3.SS1.SSS1) 写的是 L1 reconstruction/commitment 加 velocity、bone、foot-contact；本地 release 则是 SmoothL1，另有 explicit 与 geodesic。论文公式和发布实现不可混写；这里的权重结论来自本地 release。

### InterGen：官方固定 commit 核实

在 `/mnt/8T-HDD/wenning` 与 `/home/wenning/projects` 的文件清单中未找到独立 InterGen checkout；TIMotion/coshmdm 的同名 loss 属于其他项目，未冒充原 baseline。使用官方 `tr3e/InterGen` 的 `b1750df4a859443b8a69329f759ce3a7503ae179`。以下是发布代码审阅，不是本地 InterGen 复现实验。

输入每人262D，x0 prediction；训练先标准化。完整262D的 masked L2 是基础项。几何附加项从反归一化位置计算：每人 VEL=30、BL=10、FC=30；交互 DM=3、JA=3、RO=0.01，其中 RO 由髋位置推导相对朝向，并非 6D 旋转块。辅助几何项受 `t<=700` 门控；完整表示基础损失不受该门控。配置中虽有 POSE=1/TR=100 常量，但 forward 未调用对应方法，不能算成启用的 loss。[losses.py](https://github.com/tr3e/InterGen/blob/b1750df4a859443b8a69329f759ce3a7503ae179/models/losses.py#L44)、[调用与门控](https://github.com/tr3e/InterGen/blob/b1750df4a859443b8a69329f759ce3a7503ae179/models/gaussian_diffusion.py#L1381)、[配置](https://github.com/tr3e/InterGen/blob/b1750df4a859443b8a69329f759ce3a7503ae179/configs/model.yaml#L1)、[x0 设置](https://github.com/tr3e/InterGen/blob/b1750df4a859443b8a69329f759ce3a7503ae179/models/nets.py#L181)。

生成评测也先反归一化，再送 InterCLIP：[datasets/evaluator.py L36–37](https://github.com/tr3e/InterGen/blob/b1750df4a859443b8a69329f759ce3a7503ae179/datasets/evaluator.py#L36)。因此两 baseline 都额外约束位置派生几何，但主损失仍监督旋转/速度；“位置偏好”可描述 loss 结构，尚不可描述实际优化后的误差或 evaluator 排名。

## 3. 已完成的低开销 CPU 检查

脚本：[probe_interhuman_evaluator_features.py](/mnt/8T-HDD/wenning/HHC/tools/probe_interhuman_evaluator_features.py)。逐样本原始结果：[JSON](/mnt/8T-HDD/wenning/HHC/output/interhuman_evaluator_feature_probe_20261006.json)。运行句柄 **63855，exit_code=0**。仅 CPU 2 线程，`CUDA_VISIBLE_DEVICES=''`；没有改训练源码、参数或占用 GPU。

使用本地 official checkpoint，strict 加载仅 motion encoder；源码/config hash 通过现有 hash-bound wrapper 常量核验。省略文本分支与公共 `latent_scale` 后对输出单位化，因而表内不是 official FID 或 MM Distance。checkpoint 学到的 scale=4.317841053，若需要 official embedding L2，乘这个常量即可。

样本是 val split 顺序最先满足64帧的4条：6396、4078、581、3598。取经过官方配对预处理的前64帧，固定 anchor=1、无左右/人序交换、无 padding。该选样仅供低成本机制检查，不代表完整 val 分布。

对位置、速度、旋转分别加 Gaussian 方向，逐样本归一为相同 `||δ/std||₂ = ε√(64×2×66)`；各组都限制相同**总标准化能量**，避免126维旋转仅因维数多而得到更多能量。ε=0.02/0.1，方向 seeds=20261006/20261007。表为4样本×2方向均值的单位 embedding L2。

| ε | 位置 | 速度 | 旋转6D |
|---|---:|---:|---:|
| 0.02 | 0.00094028 | 0.000048115 | 0.00125091 |
| 0.10 | 0.00474122 | 0.000240687 | 0.00616749 |

三组 std 中位数分别约0.56559、0.013804、0.141671。这个尺度差异也是必须明确扰动单位的原因。本次比较的是相对数据分布尺度的局部响应，不能与“各组相同原始 L2”比较混为一谈。

两项自检也通过：

1. contact 全部加100后，单位 embedding 最大绝对差 **0.0**，符合代码中的硬删除；这仅证明标签通道不可见，不等于脚位置/脚运动变化不可见。
2. 原运动的 `Δposition=velocity` 最大误差为 **3.32e-7**。另只给第二人施加平滑正弦 X 平移，并同步修改速度，保持该恒等式（assert通过）。1cm振幅下 embedding L2 范围0.000431–0.001582；5cm为0.002131–0.007802。该检查说明相对轨迹改变可被编码，不把位置与速度的独立噪声误当成物理可行运动。

限制：独立 position/velocity 扰动会破坏它们的恒等式；6D Gaussian 可能离开合法旋转流形；平滑双人相对位移也可能改变真实接触，不保证整段物理可行。本次没有用零填充/跨样本替换来宣称重要性，未估算 FID，未编码文本，未比较生成结果，未给统计显著性结论。只在这组局部随机方向下，rotation 略敏感、velocity 明显较弱。

## 4. 下一步最小可识别消融

不建议据此次探针直接改训练权重或换 evaluator。最小下一步是固定32条独立 validation请求、完整有效长度、4个扰动方向，冻结 evaluator、文本、人序与预处理，仍使用现有缓存，先回答两个不同问题：

1. **特征依赖**：重复小幅、等标准化总能量的 P/V/R 局部噪声，报告每样本 embedding L2、cosine、对原 caption 的距离变化及配对区间。P单改、V单改、P+同步ΔP三臂必须并列；旋转用合法 SO(3) 小扰动后重新编码6D，并另报角度。总标准化能量和每维RMS两个口径分开列，避免维数决定结论。零填充或跨样本替换仅作为明确标注 OOD 的压力测试。
2. **模型误差贡献**：先在同一GT的 VQ reconstruction 上比较 HHC/InterMask，再做 GT↔reconstruction 的 P/V/R替换及 P+V一致性修复。混合输入只是诊断，不能当正式得分；必须和独立 MPJPE、旋转角误差、速度误差、`V-ΔP`残差并报。若研究 text-to-motion，不能把任意同caption GT当逐帧真值替换，因为生成与GT没有确定时间对应；要使用相同请求的自然生成缓存并分别报告评测特征误差分布和几何质量。

32条不适合稳定估计512D FID：先用配对 embedding 与文本距离回答机制问题；只有局部证据可复现且自然误差方向也一致，才在冻结完整请求上测试排名与 FID。若确需比较 position-only 与完整 evaluator，应对前者重训并验证文本检索能力；在预训练 encoder 上删列或置零造成 OOD，不能证明另一种公平协议。

“系统性偏差”的可检验判据应预先定义：在相同几何质量/感知质量下，某一表示或方法仅因可操控冗余特征得到稳定得分收益。当前证据不足以满足此判据；contact通道盲点是确定事实，而旋转相对位置的重要性仍只是有限局部证据。
