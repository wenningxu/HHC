# InterGen 几何 loss 时间与样本归一化复核

2026-10-17 22:29 SGT。此次补查针对旧自定义旋转/FK实验，不改当前官方基线或已绑定HHC队列。稀疏空间分母修正版仍不是整体官方objective；本次为其中另一个明确的reduction差异提供数值反例。

## 源码事实

官方 InterLoss.mix_masked_mse / GeometricLoss.mix_masked_mse 的几何项先按每条序列有效帧数平均，再按 timestep mask 选中的序列数平均。旧custom的avg将一个micro4组的所有有效帧合并平均，最终对所有组取平均，包括没有有效几何帧、贡献0的组。因此长度和组内timestep分布会改变权重；有效batch32并不保证与官方batch32等价。

这与距离/亲和项除以全部484对的空间分母错误是两个层面。tools/intergen_rotation_objective_sparse_v2.py有意只修空间分母和接触按人求和，时间权重保留；本次实测也确认其时间结果与archived_grouped完全相同。不能将v2称为官方loss复现。官方simple MSE本身采用全局有效帧均值，不把这里的几何项规则错误推广到simple。公式差异并非单一固定缩放，不能靠统一乘一个常数消除。

## 独立CPU验证

工具tools/audit_intergen_temporal_reduction_cpu.py直接抽取固定源文件中实际函数的AST，不重写被对照的公式；不构建模型、不读checkpoint/动作数组、不做推理。云端CPU单线程、nice15，CUDA_VISIBLE_DEVICES为空，CUDA未初始化。正式训练与所有已绑定源码不变。

| 合成输入 | 官方几何reducer | 旧custom与sparse_v2 |
|---|---:|---:|
| 等长6，全有效，逐帧平方误差1/9/1/9 | 5 | 5 |
| 长度6/2/6/2，全有效，相同平方误差 | 5 | 3 |
| 8个micro4组，仅1组有效，平方误差全1 | 1 | 0.125 |
| 两组分别1/4个有效样本，平方误差分别9/1 | 2.6 | 5 |
| 全部timestep无效 | 0 | 0 |

同时验证梯度有限、padding和timestep无效位置梯度全0。变长例中，官方每序列梯度总和约.5/1.5/.5/1.5，custom为.75/.75/.75/.75，证明差异实际改变梯度分配，不只是日志显示。上述8倍例为刻意构造的极端情况，不是实际训练测得的典型倍率。没有对真实训练分布估计平均影响，也没有FID因果证据。

## 验收与边界

- 原exec session71376完整收取exit_code=0，不把等待超时当退出。
- 工具5604B/SHA f742aea819a5ae59c406ea94decf1d84aef326479f3c384ec290638f0d6d46f4。
- 官方loss13855B/SHA082e5ebb7ff31146dcd6948339c6647b9751e7a9541a0ed4334f2ac02e72a3c9；grouped2870B/SHA90075686d4131ea8d9a0e19f3af3f1618ba53f10ccbce386a6d69eb0b50cab9e；v2的SHA仍f7565755db1ae68298dd5ab5ea12da36bdf7077311d8e0fee898a62dedb81856。
- 小报告output/hhc_sota_20261016/intergen_temporal_reduction_cpu_20261017.json，8531B/SHA1f20caa369228a1d83b7e88d8b9596373d2fad66198ee071cb60e17891ea7f42；独立SSH第二次逐文件bytes/SHA读回与本机一致。
- 额外收据intergen_temporal_reduction_upload_20261017.json、intergen_temporal_reduction_acceptance_20261017.json。本机只接收小JSON，没有checkpoint、生成或token数组下载。

当前官方训练使用未改官方loss，故本轮只补强旧实验的诊断与未来单变量实验合同。没有部署新的自定义objective、恢复旧103k实验或额外占GPU；不声称修复后能达到约6或SOTA。

最近实际官方观察14:23:16 UTC：trainer22675/birth467827326，epoch17/6310，LR1e-4/total.19有限；GPU仅22675/7880MiB。HHC仍saved4000，两项automation PAUSED，下一普通观察不早于14:33:16 UTC。该loss不是FID；正式官方20rep/MM和HHC生成对比尚待完成。
