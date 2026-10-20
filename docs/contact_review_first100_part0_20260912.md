# First100 前50 contact 语义初筛（2026-09-12）

只读审查范围为生产 manifest `completed_names[0:50]`，caption 来自 `configs/interx_shape_augmentation/ssp3d_all62_fulltrain_20260911/augmentation_records.jsonl`，contacts 来自 `dataset/interx/contact_pairs_codex_luna_strict_v3_train/<source_name>.pkl`。判断规则是：每个样本允许统一 P1/P2 重命名一次；不同关系不得分别交换；左右按解剖侧保持。

本轮不把 `p1_part→p2_part` 与 `p2_part→p1_part` 单条差异直接判错，也不把空接触且 caption 无明确接触判错。逐条复核后，**明确矛盾/重要遗漏：1/50**，**未见明显文本冲突：49/50**。完整原文 caption 与 contacts 保留在上述两个绑定源文件，以下是已核对的完整 manifest 顺序与分类：

## positions 0–49

- 0 `G033T002A014R027__shapeaug_s00000123`
- 1 `G006T004A015R005__shapeaug_s00000123`
- 2 `G055T006A030R000__shapeaug_s00000123`
- 3 `G019T003A015R003__shapeaug_s00000123`
- 4 `G046T005A026R010__shapeaug_s00000123`
- 5 `G035T002A010R001__shapeaug_s00000123`
- 6 `G044T002A014R032__shapeaug_s00000123`
- 7 `G014T004A021R006__shapeaug_s00000123`
- 8 `G008T001A005R001__shapeaug_s00000123`
- 9 `G029T000A003R021__shapeaug_s00000123`
- 10 `G031T008A032R012__shapeaug_s00000123`
- 11 `G055T000A000R002__shapeaug_s00000123`
- 12 `G057T002A011R008__shapeaug_s00000123`
- 13 `G052T000A004R027__shapeaug_s00000123`
- 14 `G021T001A005R003__shapeaug_s00000123`
- 15 `G015T001A009R027__shapeaug_s00000123`
- 16 `G038T006A034R020__shapeaug_s00000123`
- 17 `G034T006A030R001__shapeaug_s00000123`
- 18 `G031T008A032R013__shapeaug_s00000123`
- 19 `G055T004A021R005__shapeaug_s00000123`
- 20 `G005T005A027R009__shapeaug_s00000123`
- 21 `G022T000A001R007__shapeaug_s00000123`
- 22 `G032T002A011R007__shapeaug_s00000123`
- 23 `G055T002A010R002__shapeaug_s00000123`
- 24 `G046T005A025R002__shapeaug_s00000123`
- 25 `G036T005A026R009__shapeaug_s00000123`
- 26 `G050T001A009R023__shapeaug_s00000123`
- 27 `G038T002A014R022__shapeaug_s00000123`
- 28 `G047T000A000R004__shapeaug_s00000123`
- 29 `G021T004A022R015__shapeaug_s00000123`
- 30 `G056T000A003R019__shapeaug_s00000123`
- 31 `G026T001A009R022__shapeaug_s00000123` — 明确角色/部位组合冲突：caption 是一人坐在另一人的腿上，另一人用手放在前者上臂；原 contacts 的 `butt→legs` 可对应坐腿，但 `hands→arms` 与该统一人物关系不一致；建议为 `P1 butt→P2 legs` 与 `P1 arms→P2 hands`，仍需主 agent 验收全局命名。
- 32 `G058T000A003R019__shapeaug_s00000123`
- 33 `G045T001A008R022__shapeaug_s00000123`
- 34 `G004T009A023R007__shapeaug_s00000123`
- 35 `G046T007A036R008__shapeaug_s00000123`
- 36 `G046T004A021R007__shapeaug_s00000123`
- 37 `G002T004A021R004__shapeaug_s00000123`
- 38 `G029T006A026R006__shapeaug_s00000123`
- 39 `G013T002A009R025__shapeaug_s00000123`
- 40 `G016T001A005R003__shapeaug_s00000123`
- 41 `G044T002A014R028__shapeaug_s00000123`
- 42 `G024T005A027R013__shapeaug_s00000123`
- 43 `G044T006A034R023__shapeaug_s00000123`
- 44 `G029T002A011R010__shapeaug_s00000123`
- 45 `G058T006A021R004__shapeaug_s00000123`
- 46 `G031T006A026R006__shapeaug_s00000123`
- 47 `G015T001A008R018__shapeaug_s00000123`
- 48 `G052T000A003R019__shapeaug_s00000123`
- 49 `G026T004A022R014__shapeaug_s00000123`

这不是视觉检查、不是 contact 正确率，也未修改缓存或 timing。其余49例为 `no_obvious_text_conflict`，不等于标签正确；主 agent 仍应逐例测试原映射/统一交换映射。
