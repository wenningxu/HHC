# 当前 InterGen 旋转/FK 实验与官方训练的差异

2026-10-17。用户质疑当前结果远离约 6 的复现值后，核查了本机源码及云端实际配置。结论：**当前实验不是“仅替换动作表示”的官方 InterGen 复现，不能用于判断官方方法是否可复现，也不能将指标差异归因于动作表示。** 此前对实验性质的说明不充分。

## 来源与实际变更

云端官方目录 `/workspace/HHC/third_party/intergen_base_20261008` 的提交为 `b1750df4a859443b8a69329f759ce3a7503ae179`。读取了网络、扩散、官方 loss、调度器、模型/训练配置、官方训练入口及数据集共九个文件；其完整字节均与该提交一致。工作树仅删除三项 README 媒体。核对通过 SSH 只读完成，exit 0，无模型构建或 GPU 工作。

| 项目 | 官方实现 | 当前云端运行 |
|---|---|---|
| 网络、文本模块、扩散 | 官方 InterGen/InterDenoiser/CLIP 与扩散代码 | 从上述官方目录导入；输入维度改为 268，旋转输出 bias 改为单位旋转 |
| 表示 | 262 维 | 保留原 262 维，追加 root6，共 268 维，并非纯旋转格式 |
| 训练入口与数据 | 官方 Lightning 训练入口、InterHuman 数据集 | 项目自写 `train_intergen_rotation_fast_cloud.py`、`interhuman_root_rotation.py`；数据适配器借用 InterMask 的预处理；未完成与官方数据管线逐样本一致性验证 |
| 训练损失 | 原始归一化表示 MSE + 官方 InterLoss / GeometricLoss；含 VEL、BL、FC、DM、JA、RO | 绕过官方 `compute_loss`；自写旋转、矩阵、FK 位置/速度/接触/距离等损失；原位置/速度/接触仅权重 0.1 的辅助项，接触标签、距离 mask 与 reduction 也不同 |
| 学习率 | 按 epoch 更新，warmup 10 epoch，2000 epoch cosine 至 0 | 自写逐 step 调度，704000 步终点保留基础 LR 的 0.1；warmup 由数据长度换算 |
| 数值与 batch | FP32；配置 batch32 | BF16 网络、FP32 FK/loss；当前 micro16×accum2，有效 batch32；此前 micro4×accum8 |
| 训练连续性 | 标准完整状态恢复 | 10000 步迁移只恢复权重，optimizer/RNG 未恢复；84000 步性能迁移则恢复了完整 optimizer/RNG |
| 验证 | 论文/正式完整测试协议 | 当前只做固定 480 条训练内 validation，不能作为完整 test 成绩 |

AdamW 基础 LR 1e-4、weight decay 2e-5、梯度裁剪 0.5、扩散 1000/cosine、条件 dropout 0.1、生成 DDIM50/CFG3.5 与对应官方设置相同。相同的部分不足以消除上表中的混杂因素。

## 额外发现与解释界限

当前生成出口反归一化后直接裁掉追加 root6，保留原 262 维，**没有从预测旋转通过 FK 重建送入 evaluator 的位置/速度**。因此训练中较强的 FK 监督与评测读取的位置通道存在不一致风险；原位置通道仍有 0.1 辅助监督。这是代码事实与待验证假设，尚不能认定为 FID 恶化的原因。

官方 GeometricLoss 中 POSE/TR 虽有定义，但没有在其 forward 调用；不能将它们列为当前遗漏的官方有效损失。

100k 的 FID 35.2390586118、历史最佳 20k/24.3141428796 均属于此自定义旋转/FK 实验的固定 480 条 validation。不能用“步数还不够”解释与用户约 6 的官方复现值的差距；也不能只靠更换评测集合说明该差距。当前尚未用完全官方的 262 维训练入口、数据和损失，从零建立同平台基线。

可靠的下一项方法验证应先恢复官方 262 维基线、确认可复现水平，再逐项改变表示及必要适配，独立记录 loss/数据/调度变更。当前未启动这项 GPU 实验。此次只做来源审计与记录，没有停止、重启、改写正在运行的训练，也没有更改 HHC 等待 InterGen 完整 704000 步的既有门控。

## 证据与主要源码

- `output/hhc_sota_20261016/intergen_official_fidelity_audit_evidence_20261017.json`：云端实际 config、官方 commit/status、九项逐字节校验与 SHA256。config SHA256 为 `afb7e4fdf07deef45be076af3bc755f4c19f94f6566bf9c3c7b0e381fa8753f8`。
- `tools/train_intergen_rotation_fast_cloud.py:173`：官方模型导入及维度修改；318 起为实际训练/调度，80 起为验证出口。
- `tools/intergen_rotation_objective.py:22`：自写损失及权重；实际 fast 路径使用分组版本以保留旧 micro4 reduction。
- `third_party/intergen_base_20261008/tools/train.py:39`、`models/utils.py:6`：官方优化器、调度与 FP32 训练。
- `third_party/intergen_base_20261008/models/gaussian_diffusion.py:1381`、`models/losses.py`：官方有效损失路径。

目标仍未达到 SOTA；此审计纠正实验标签，不构成新的性能结果。
