# 旋转后处理：免训练与学习型方法的评估收益

2026-10-07。最新用户目标是评估指标提升；位置—旋转一致性只是手段，不作为方法成功的必要或充分条件。

## 第一批完整480验证结果

同一生成cache/seed20261006、冻结480val、原版CPU InterCLIP。下表各模型原始FID均精确复现；位置、速度、接触通道保持不变。

| 方法 | 是否新增训练 | InterMask FID | InterGen FID |
|---|---|---:|---:|
| 原始 | 否 | 8.780644 | 9.681665 |
| 冻结VQ旋转重建，混合.25 | 否 | 8.747113 | 9.675755 |
| 冻结VQ旋转重建，混合.5 | 否 | 8.717303 | 9.702095 |
| 冻结VQ旋转重建，全替换 | 否 | 8.671397 | 9.813278 |
| 恢复release固定腕identity | 否 | 8.780609 | 9.681665 |
| P+R时序残差校正，全修正 | 是 | 8.722070 | 9.550627 |
| R-only时序去噪，全修正 | 是 | 8.700338 | 9.507802 |
| 几何监督+噪声增强，固定2000步全修正 | 是 | 8.721918 | 9.473583 |
| 前轮生成旋转初始化IK | 否 | 11.520416 | 9.163593 |

VQ只改善InterMask较明显，InterGen会随强度增加变差。两小型学习校正器都有小幅FID改善；加入位置输入未优于R-only。这并不证明位置信息没有用，只说明当前训练数据/模型/目标未体现优势。R-only的FK残差变化很小，但依然是一个评估改善候选，不因一致性未明显改善而淘汰。

## R-only完整非MM指标

| 模型/状态 | Matching↓ | R1↑ | R2↑ | R3↑ | Div |
|---|---:|---:|---:|---:|---:|
| InterMask原始 | 3.793051 | .433333 | .606250 | .693750 | 8.066027 |
| InterMask R-only | 3.792646 | .433333 | .608333 | .697917 | 8.061790 |
| InterGen原始 | 3.806231 | .429167 | .575000 | .666667 | 7.953854 |
| InterGen R-only | 3.805548 | .431250 | .579167 | .670833 | 7.946630 |

GTDiv=7.741262。MM待评：当前缓存每请求只有一次生成，不能编造MM。变化较小、单seed、重复使用val进行探索，不能声称显著或正式test改善。VQ alpha1虽InterMask FID更低，但R1/R2/R3=.425/.589583/.689583，较原模型下降，不能只报FID。

## 固定2000步候选补评（10:43 BST完成）

内部holdout角误差4.859576→4.883283°略差，但两模型FID均改善。因此辅助角误差不能替代最终指标筛选。

| 模型 | Matching↓ | R1↑ | R2↑ | R3↑ | Div |
|---|---:|---:|---:|---:|---:|
| InterMask几何终态 | 3.792391 | .431250 | .606250 | .697917 | 8.068943 |
| InterGen几何终态 | 3.805532 | .435417 | .579167 | .670833 | 7.954598 |

InterMask R1比raw低.002083；InterGen Matching和R检索均小幅改善，Div略远离GT。MM与独立test仍待评，不能以这轮480val探索宣称稳定提升。

## 学习方法与分割

### 独立test rep0结果（10-07 17:43 BST收取）

主70655/55717均exit0；1056条冻结test、生成seed1234，CPU评测，GTDiv7.655049。使用下述预冻结候选，不按test结果调参。

| 模型/状态 | FID↓ | Matching↓ | R1↑ | R2↑ | R3↑ | Div |
|---|---:|---:|---:|---:|---:|---:|
| InterMask raw | 5.195238 | 3.792476 | .442235 | .592803 | .688447 | 7.951096 |
| InterMask R-only | 5.142410 | 3.792310 | .439394 | .593750 | .686553 | 7.948615 |
| InterGen raw | 6.403511 | 3.798156 | .434659 | .618371 | .690341 | 7.797199 |
| InterGen geometric_final | 6.150488 | 3.797202 | .435606 | .616477 | .693182 | 7.791027 |

FID分别下降1.02%/3.95%，Matching略改善，检索有涨有跌，不能称全面提升。只有1个rep，尚无显著性判断；MM没有重复生成，待评。完整输出`output/rotation_test_rep0_20261007/{intermask,intergen}/paired_metrics.json`，冻结权重hash和请求来源见同目录`paired_embeddings.provenance.json`。不与480val FID跨协议比较。

### 10-07 独立test预冻结

在读取test指标前固定：InterMask采用R-only step600、full强度；InterGen采用geometric_final step2000、full强度。选择依据为480val的FID及Matching/R检索权衡，不再优化一致性代理指标。InterMask冻结权重SHA256 `43141173eabae51d021aec31245cebdba01648d41964c7c90c86544ff5177012`；InterGen `5f8cd7727ecf7950a0c67fd0f525b32912efe1c421659fcc5eecbe9dc1c02c0f`。

第一轮使用现有official test冻结request replication0全部1056条、生成seed1234、各自官方采样设置，原始/校正共用同一次生成；test不用于调参。生成改用GPU但两臂共用完全相同缓存，评分仍CPU；不能与之前CPU生成的480val作跨协议绝对数值比较。先做2条全长GPU接口检查，再完整生成并评分。MM需要单独重复生成，尚未完成。该轮只有1个rep，不能声称20rep基准结果或统计显著。

- 1024个official train动作，各取64帧，冻结官方InterMask VQ重建形成真实有损输入；仅896训练，128个不同动作ID作为内部holdout。assert所有ID仅train且无val/test交集。
- 3层小型时序卷积残差头，以原旋转为起点，最后层零初始化、.25*tanh修正上限。P+R输入使用去全局刚体坐标的每人位置及原旋转；R-only将位置输入置零。
- 2000步，AdamW3e-4，batch64，25% clean identity/75%真实VQ重建；normalized6D MSE+.1时间差分MSE。前两候选原先按内部holdout旋转角选步：P+R200、R-only600；该选择发生在用户最新明确指标优先之前，不能把它当最终最优指标选模方案。
- 原GPU评分相对CPU约有.006偏差，已重载检查点重做CPU，表内不混用GPU结果。
- 独立新增几何SO3监督+合法旋转噪声训练对照，旧角误差准则选择step0，因此那次输出是identity，不能冒称真实训练效果。按最新要求重新评估固定2000步实际训练终态，不以几何角误差作为挡住评估的门槛。

## 句柄与产物

- 免训练重建49174/评分11962 exit0；其它指标26199 exit0。
- P+R和R-only初训38741/49840 exit0；CPU复核60966/60507 exit0。
- 几何旧选择27867 exit0但selected0。实际终态候选`geometric_final`主4064 exit0；完整非MM补评9208 exit0，summary.json及各模型metrics均在该目录。
- `tools/train_rotation_residual.py`、`tools/probe_vq_rotation_projection.py`、`tools/score_rotation_method_metrics.py`。
- 学习结果以`output/rotation_methods_20261007/position_rotation_cpu_retry/`和`rotation_only_cpu/`为准；每模型`*_metrics.json`包含完整指标和来源hash。
- 评分器原输出仅用basename导致不同方法同名metrics覆盖，已改为保存在各自embedding父目录并重新评分；顶层`intermask_metrics.json`是早期被覆盖副本，不作归档结果。

训练对标签曾U15截断，按split_index896的训练划分不受影响。agent重建U16修复时原位写造成一次CPU复核初始化BadZipFile退出，文件稳定后重试成功；未把失败结果当有效。前后数值未进行保留副本的逐元素比较，不宣称bitwise一致；当前数据hash见training_pairs_manifest.json。
