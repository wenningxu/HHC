# Shape augmentation 全量验收（2026-09-30）

第②项交付：公开 SSP-3D SMPL shape 经既有转换流程得到 SMPL-X donor；62组不同gender/shape，非AMASS。冻结weight1、50步、lr=.001、共同物理地面方案。原始数据、旧实验和检查点保留；尚未启动增强训练。

## 速度与物理质量

分母为全部6238个生产请求，两个actor均换shape。原shape GT与优化后使用同一完整crop、独立VolumetricSMPL、逐name固定seed1234；Eq.6为神经SDF估计，不是精确网格交体积。比较无epsilon容忍，不按好结果缩小分母。

| 指标 | 原shape GT | 换shape并优化后 | 逐例不高于GT |
|---|---:|---:|---:|
| Foot sliding，mm/frame，逐样本均值 | 6.049045 | 5.547326 | 4823/6238 |
| Eq.6 squared sum，m²/frame，逐样本均值 | 0.146984298 | 0.027224764 | 5330/6238 |
| 两项同时满足 | — | — | 4154/6238＝66.591856% |

正式gate要求原/优化pair的脚部观测总数均为正；不是要求每个人均有正观测。若额外要求原/优化双方各自观测均为正，联合通过4150/6238，作为敏感性分析，不改变正式口径。

- Optimizer调用均值4.192379s，中位4.516103s，330/6238超过5s。
- 样本处理均值4.451407s，中位4.766455s，1230/6238超过5s。
- 处理时间不含整次初始化、独立评分以及部分sidecar/completion日志写入；仅均值达到5s目标，不能称每例都小于5s。

## 接触诊断

7278个三条件共同有效样本-关系观测，覆盖5666例；572例没有有效关系，不将其算成零距离。每个关系先按历史时段权重聚合，再对关系等权平均。

| 指标 | 原shape GT | 换shape未优化 | 优化后 |
|---|---:|---:|---:|
| Weighted min vertex distance，mm | 103.776348 | 107.015564 | 96.204579 |
| Weighted symmetric nearest 5%，mm | 113.937677 | 118.106221 | 108.147807 |
| 2cm contact frame fraction | 47.496220% | 43.089069% | 49.809549% |

相对换shape未优化，4425/7278个关系的min distance严格下降，2853上升；2985/7278个关系的2cm帧比例增加。整体吸引有效，但不能声称所有接触改善。部位标签、角色与自动距离划分的旧时段仍存在语义局限；近距离不是文本正确性证明。已知7例风险全部隔离，详见`shape_semantic_exclusions_20260930.json`和既有配对诊断文档。

## 交付与证据

物理通过4154，过滤已知风险后4148/6238＝66.495672%。过滤没有缩小请求分母；七个风险中六个原本物理通过。

- 数据：`dataset/interx_shape_ssp62_physical_full6238_reviewfiltered_20260930`
- 训练名单：9110原样本＋4148增强＝13258；原验证集和统计量保持不变。
- 原始motion以H5 external links引用，交付是本地overlay，不是脱离源资产可搬运的数据包。
- 保留4148例自身GT/优化foot均值7.318486/6.592864mm/frame，Eq.6均值0.155268343/0.015472503m²/frame；这些是筛选后统计，不能替代上面全6238统计。
- `output/ssp62_production_physical_first100_assessment_20260912.json`
- `output/ssp62_production_physical_remaining6138_assessment_recovered_20260930.json`
- 两分片`...first100_gate_20260912.json`、`...remaining6138_gate_20260912.json`
- `output/ssp62_full6238_reviewfiltered_selection_20260930.json`
- `output/ssp62_full6238_reviewfiltered_materialization_20260930.log`

恢复评分76361 exit0；标准remaining gate重建33297 exit0；root覆盖/前缀/选择核验15878 exit0；打包与完整内容读回28765 exit0。全部6238名称唯一、位置0..6237；旧4879评分记录逐值保留。恢复前最后一例重新评分一致（仅重放比较rtol1e-6/atol1e-10，正式质量gate仍无容忍）。旧partial缺少scorer自身哈希，采用冻结首100报告、现有依赖绑定和断点重放交叉印证；最终报告recovery字段明确记录此历史证据限制。原段错误涉及nvidiactl，根因未定，不能声称已修复驱动。

第②项的平均速度、严格物理通过率与训练数据交付已通过上述验收；没有认证全部caption正确，也没有证明使用增强数据必然改善下游生成。下一步按用户顺序进入第③项InterHuman两部分body/root VQ实验。
