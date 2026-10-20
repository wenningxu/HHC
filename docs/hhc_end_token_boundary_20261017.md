# HHC 最长 token 序列的 END 上下文诊断

2026-10-17，新加坡时间。

**2026-10-17 22:10 SGT：HHC END边界CPU诊断完成，不晋升改动。** 训练TokenData输出76位置且最长75个有效token后含END；现有sample_intra_cas只分配75位置，n75时缺END。实际生产采样方法的独立假logits输入追踪确认这一差异（假logits只验布局，不作为模型效果）。训练真实cache4567条，未删首/尾时215条达到75、删除1token后210条；固定val480中35条（7.29%，均L299）达到此边界。root重新计算验证请求长度，云CPU诊断独立重算cache计数；subagent原直方图标签误写已纠正为validation_processed_length_hist，数值没有据此当训练token直方图引用。

对实际保存A4000权重/冻结VQ170k，选择3个真实训练交互4789/6952/2111，2种mask输入、n74控制/n75边界，共12例，云CPU FP32/2线程/nice15前向。n74仅多一个被屏蔽PAD的6个控制，conditional及CFG3.5 logits逐位相同；n75的6个对照有变化，conditional最大logit差.19413、CFG最大.66248、CFG最小top1一致率98%。补END的CFG masked teacher-forced NLL有1例好/5例差，等权平均delta+0.004037857 nats；conditional为2好/4差、均值-.000385553。没有一致收益，不改已绑定A/B/C/五候选队列、不新增GPU采样消融。诊断仅3个train交互/早期A4000/CPU FP32，不是独立6样本、最终val或GPU BF16结果；小logit变化仍可能经迭代放大，不能据此断言完整生成FID一定不受影响。

原exec85432、SSH和诊断均exit0，CUDA未初始化；没有GPU模型推理、训练源码/权重改写或本机数组/权重下载。报告17214B/SHAabd92645a4014f46252a01825780b70637604e598d4ad835c333fb29acb0e990，源码8547B/SHAb9c1288b301cb8e2a2a8cbec081b9ac45a6754367e781308b1eb744277ee6625，云源码与小报告均按bytes/SHA读回通过。详见docs/hhc_end_token_boundary_20261017.md及hhc_end_boundary_cpu_summary_20261017.json。

14:10:44 UTC实际官方trainer22675/birth467827326推进epoch15/step5350、LR1e-4、total.2222，GPU仅22675/7880MiB。无FID；HHC仍saved4000，官方原始eval20等待26133、HHC等待25145不变，两项automation保持PAUSED。下一普通训练观察不早于14:20:44 UTC；优先等正式HHC matched-CFG/window和text-only验证，不能从本次局部诊断宣称SOTA。目标active。

## 对照设计与结论边界

保存的A4000完整checkpoint来自云端不可变快照saved_A_step4000.pth，3429290971B/SHAfb1b5a2c606ffaa68fcff878f1038d122713d652b1767de35c64acd0f8614963；冻结VQ170k SHA2f676f63466135385ded7016d60da9dd3b6e86e3086b85e14a201732d1a55002。真实token cache7071413B/SHA9c31d30b17354a20f675b0123846fadf413aa2830814197b3bfb4a0b0a6bfdcd。诊断使用生产TokenData的真实训练target/shape/caption，模型eval模式；两种输入为全MASK和原历史corruption。只改变forward上下文的序列宽度75/76：n74额外位置是被attention屏蔽的PAD，n75额外位置是可见END；对同样的有效位置、target与mask支持比较logits/teacher-forced NLL。训练与采样源文件均保持原绑定SHA，未调用GPU采样。

| 对照 | conditional最大logit差 | CFG3.5最大logit差 | 补END的CFG NLL变化 |
|---|---:|---:|---:|
| n74，额外masked PAD，6例 | 0 | 0 | 全部0 |
| n75，额外可见END，6例 | 0.194132 | 0.662477 | 平均+0.004038；1好/5差 |

6例来自3条交互×2种mask状态，不能视为6个独立交互或做未经设计的置信区间。NLL只在所选mask位置上计算，不是当前all-valid训练目标的完整评估，更不是生成FID。CFG分布与conditional分布不同，表中差异不能用来证明CFG本身有害。未做验证集选模、50步生成或test评分；不能把训练内teacher-forced响应直接当作最终生成收益。

目前保留此布局差异作为后续独立候选，不晋升到已注册的五候选或新GPU任务。若未来按val证据开展END消融，应保持输出token数量、每条请求随机种子及Gumbel采样噪声形状：直接把整个采样buffer从75改76会改变随机数消耗，混入额外因素。只在forward上下文补END、原75位置上采样是可检查的候选设计，尚未实现/验证，也未接入实验。

当前证据不足以说明END差异解释HHC与公开基线之间的差距。继续已注册的matched text-CFG、窗口与无GT shape候选；在真实val生成结果之前，不改动主训练或按少数NLL条目挑成功。

## 已验收的小证据

- hhc_end_boundary_cpu_report_20261017.json：17214B/SHAabd92645a4014f46252a01825780b70637604e598d4ad835c333fb29acb0e990。
- hhc_end_boundary_cpu_summary_20261017.json：root逐例检查PAD控制全0、source与云SHA一致、12例/3IDs/6边界例，并按实际原始值重新计算汇总。
- hhc_end_boundary_cpu_execution_20261017.json：实际SSH exit0、完整stdout/stderr，源代码SHA与上传回执一致。
- token_length_coverage_metadata_20261017.json：元数据计数；validation直方图标签已纠正，root在本机重新计算全部480请求，在云CPU模型诊断中重新核对训练cache计数与真实train token直方图。

所有权重、cache和张量均留在云端，只有源码与小JSON到本机。没有新FID、没有HHC生成改进成绩、没有SOTA结论。

备份收据：本轮HHC END CPU诊断源码与小证据包hhc_end_boundary_cpu_evidence_20261017.zip，430160B/SHA058c626811076d0c8e7236c1d172e0951e0e19e5f13545729e3d0dcee6162e45、20成员。7项源码与云端诊断报告中的实际源码SHA匹配，云端整包/成员SHA均验收；Drive文件1_snRoI0qCdCfTF00mocaC01svvRtY-O4，大小/父目录读回一致，未返回checksum，不声称Drive SHA核验。只含代码/文档/小JSON，无checkpoint、token cache、生成数组，不是权重异地备份。收据output/hhc_sota_20261016/hhc_end_boundary_cpu_backup_receipt_20261017.json；归档docs快照早于本条。CPU诊断85432已exit0，不重跑；不晋升END改动，继续已注册CFG/window/无GT shape验证路线。官方最近5350/暂无FID，目标active，SOTA未验证。

