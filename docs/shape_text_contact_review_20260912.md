# SSP62 生产前100接触文本审查（2026-09-12）

## 主agent确认的文本—结构标注矛盾

### R001逐帧几何核查（主agent，session49636 exit0）

补充session62781 exit0：原shape帧124–131，P1右脚最近的P2顶点依次落在region [5]、[5,72]、[17,72]、[17]、[17]、[17]、[17]、[3]；对应左臂/左手掌背，距离41.63、9.73、1.90、2.04、3.42、2.39、1.52、15.37mm。这是几何最近部位，可能为阻挡，但不是证明文本意图为“踢左手”；不能将stomach替换成另一个未经语义验证的攻击目标。可在独立实验中作为原几何接触保持关系，明确与文本目标分开。

原shape、原crop0:150、独立full-sequence mesh floor、CPU SMPL-X完整顶点。用scipy cKDTree逐帧精确最近顶点距离；手掌区域→对方head，脚区域→对方whole body；5cm仅接近诊断，不是精确表面接触真值。

- 实际P2 right hand→P1 head：最小0.111mm（帧62），5cm内帧26–103，timing内均距2.119mm；P2 left hand→P1 head最小616.988mm。支持文本的右手动作，排除缓存左手。
- 实际P1 right foot→P2 whole body：最小1.524mm（帧130），5cm内帧124–131，8帧全部在原timing外；P1 left foot最小189.854mm。支持另一人右脚的后续接近事件，说明现有timing没有覆盖这段几何接近。
- 实际P1双手→P2 head最小分别601.379/644.467mm；P2双脚→P1 whole body最小243.458/189.854mm，无5cm接近。整组角色固定为文本攻击者=P2后，第二关系必须反向，不能两条均由同一角色施动。
- 脚的目标使用whole body，尚未判定kick具体命中stomach；不因文本未指定部位而替其补造腹部真值。原timing26–103恰覆盖手—头接近，但不覆盖后续脚部事件；修订时应逐关系时段，而非将两种接触同时扩展到所有帧。

- `G026T001A009R022`：caption为一人坐在另一人腿上、另一人将手放在其上臂；实际pickle第一条P1 butt→P2 legs，后续却P1 hands→P2 arms。第一条固定P1为坐腿者，按该文本第二条应P1 arms→P2 hands；整组交换仍不能同时满足。代词存在语境依赖，但主agent查看-60度同帧图，蓝人坐向红人腿部、红人双手靠近蓝人上臂，与上述解读一致。骨架图不证明表面接触，仍需密集mesh核验。与R001一样，不应通过加大学习率强行满足错误组合。
- `G013T007A030R001`：实际读取生产sidecar与pickle。caption明确right hand扇对方left cheek，缓存第一条却是P1 left hand (palm)→P2 head；caption第二动作是被打者抬right leg踢回，缓存第二条仍P1 right leg→P2 stomach。两条攻击者角色相同，不符合文本中的角色反转；一次全局P1/P2交换无法同时修复。生产sidecar的caption match为exact、双方caption SHA一致，证明“文本哈希一致”不足以验证LLM抽取语义。实际动作与文本谁正确仍须视觉检查，不擅自修正缓存。
- 此例接触距离下降不能算文本语义成功；所有当前产物仅候选，正式增强晋级前需处理错误标注并重新验收受影响样本。运行中缓存保持不变。
- 溯源：`_metadata/G013T007A030R001.json`指向batch `1bee17d10477b66d`；batch保存的模型`parsed_response`已含相同left hand与P1 right leg行，错误在抽取回复中即存在，并非后续region ID映射产生。`map_contact_rows`直接映射p1_part/p2_part；role_ratio变成方向loss权重，不会交换身体部位或建立逐事件角色。源码`_determine_role_assignment`只做整组两人交换。仍需视觉区分原文本本身与原动作是否一致。

来源：`output/ssp62_production_physical_first100_assessment_20260912.json`，按优化后加权最小距离较远/退化取8例。距离为 nearest-vertex proxy；`orig/new/opt` 分别是 original_shape/new_shape/optimized，单位 m；末组三项是 nearest 穿透平方和，单位 m²/frame。不能据此做视觉验收。

| name | caption（原文） | crop；timing非零相对帧 | contact cache 标注（p1→p2） | min dist orig/new/opt；pene orig/new/opt |
|---|---|---|---|---|
| G013T007A030R001__shapeaug_s00000123 | Two people stand facing each other. One extends his/her right hand and forcefully slaps the other person's left cheek. In anger, the second person raises his/her right leg to kick the first person. | 0:150；26–103 | left hand (palm)→head；right leg→stomach | .755/.738/.704；.3998/.2903/.0008 |
| G044T000A004R030__shapeaug_s00000123 | One stands still while another stretches out a left hand to push the first person's stomach, causing a fall backward. | 0:36；17–30 | left hand (palm)→stomach | .316/.345/.363；.0158/1.0524/.0036 |
| G026T001A009R022__shapeaug_s00000123 | A person sits on the other person's leg, while the other person places his/her hands on his/her upper arms. | 10:160；22–141 | butt→left leg,right leg；left/right hand (palm)→left/right arm | .267/.334/.341；.4989/.3941/.0042 |
| G008T001A005R001__shapeaug_s00000123 | One person lifts his/her right leg and kicks the second person's left knee, causing the second person to spin to the left. | 0:106；34–50 | left leg→right foot | .346/.337/.335；.0008/.0179/.0010 |
| G029T002A011R010__shapeaug_s00000123 | One person steps forward, turns around, and pats the other person's back with his/her right hand. The other person steps back, pats his/her leg, and turns left. | 2:152；55–118 | back→right hand (palm) | .248/.244/.299；.0729/1.0122/.0081 |
| G021T000A004R023__shapeaug_s00000123 | One person approaches and lightly taps the other person's right shoulder. | 0:120；44–63 | right hand (palm)→right shoulder | .289/.271/.265；.0193/.0214/.0002 |
| G004T009A023R007__shapeaug_s00000123 | The first person puts his/her right hand on the waist of the second person, while the second person puts his/her left arm around the first person's left shoulder. Then, they walk forward together in a line. | 73:223；0–149 | right hand (palm)→waist；left shoulder→left arm | .309/.252/.261；.1982/1.6035/.0901 |
| G023T004A023R017__shapeaug_s00000123 | The two people place their hands on each other's shoulders and walk in a circle together. | 101:251；0–149 | left/right hand (palm)→left/right shoulder；left/right shoulder→left/right hand (palm) | .263/.203/.251；.9711/1.8056/.0476 |

文本绑定检查：8例均由同一 `augmentation_records.jsonl` 的 `source_name/crop_start/crop_stop/caption/timing_file` 读取，timing 数组长度覆盖 source motion；因此只能确认 caption、crop、timing 文件的记录绑定，不能从文本确认动作在实际 motion 中是否发生或接触时刻是否准确。原始 production manifest 固定 `iterations=50, lr=.001, seed=1234`，使用 exact cropped manual binary timing。

固定顺序前4例（报告 rows 顺序）对照：

| name | min dist orig/new/opt (m) | pene orig/new/opt (m²/frame) |
|---|---:|---:|
| G033T002A014R027__shapeaug_s00000123 | .149/.112/.119 | .0112/.0158/.0001 |
| G006T004A015R005__shapeaug_s00000123 | .248/.245/.233 | .0015/.0006/.0008 |
| G055T006A030R000__shapeaug_s00000123 | .010/.009/.002 | .0063/.0085/.0002 |
| G019T003A015R003__shapeaug_s00000123 | .003/.003/.003 | .0278/.164/.0193 |

本清单的8例按接触距离排序，不等同于固定顺序；固定顺序前4例未在本次文本审查中重新推断 caption/动作事实。

## 原始 parsed_response 与 caption 审查

输入：`dataset/interx/contact_pairs_codex_luna_strict_v3_train/_batches/1bee17d10477b66d.json`。分母为该 batch 的 `sample_names`，共 **32 条**；逐条对照同一 `augmentation_records.jsonl` 的 caption。此处只审查文字角色/左右关系，不检查动作视频。

### 前版计数撤回

前版“明确角色或左右矛盾 18/32、歧义 14/32”的计数是**未复核初筛，不作为结论**，现撤回。原因是原 loss 允许每个样本统一进行一次全局 P1/P2 置换；单条 `head→hand` 对 `hand→head` 不能单独判错，且 R003 不应同时落入两个类别。

当前不提供新的错误率或分母结论。正确审核必须逐样本比较两种统一映射：原始 `p1→p2` 与整组交换后的 `p2→p1`；允许整组置换，但不允许逐条独立交换；左右手仍按解剖侧保留。

R001 等条目仍需在上述双映射规则下逐条复核；不能仅凭“caption 施动者方向”和单条 contact 的箭头下结论。`role_ratio` 只记录权重，也不能替代统一 P1/P2 映射审计。

### 待复核项（不计数）

`G013T006A028R016`、`R018`、`R019`（腕力/双方手部，左右手与主导者存在语义歧义）；
`G013T006A029R022`、`R023`、`R024`（caption 对应空接触，不能据空列表判定左右）；
`G013T007A031R005`、`R006`、`R007`、`R008`、`R009`（需 caption 的具体接触时序/角色，文本不足以确认）；
`G013T007A032R014`、`G013T007A033R017`、`R019`（自触摸/模仿/指向等，parsed 空或部位关系不能由接触文字唯一确定）。

特别是 R003：caption 描述双方互相拍双方脸，回复重复两条 `hands→head`，并用 `9:1` 与 `1:9`；必须先检查统一原映射和统一交换映射，不能把两条 ratio 当作逐条角色交换证据。完整 caption 与 parsed contacts 原始内容均保留在上述 batch 文件中；本节不再强行将任何条目归为“明确矛盾”或“仅歧义”。
