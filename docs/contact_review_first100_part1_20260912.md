# Contact 语义初筛：first100 positions 50–99（2026-09-12）

范围：`_completion_manifest.json` 的 `completed_names[50:100]`。逐例读取原始 `dataset/interx/texts_processed/{base}.txt` 首行 caption 与 `contact_pairs_codex_luna_strict_v3_train/{base}.pkl`。判定允许整组 P1/P2 全局交换一次；左右解剖侧不随角色交换。方向相反的单关系不作为错误；caption 无接触文字时空配对不算错。这里只做语义初筛，不替代主 agent 的物理复核。

## 50 例名单与结论

|pos|sample|结论|
|---:|---|---|
|50|G019T000A000R005|no_obvious_text_conflict|
|51|G041T006A015R004|no_obvious_text_conflict（两关系在统一角色映射下有效；箭头不代表动作方向）|
|52|G021T003A019R023|通过|
|53|G038T007A036R007|no_obvious_text_conflict|
|54|G047T003A015R001|no_obvious_text_conflict|
|55|G045T007A032R016|no_obvious_text_conflict|
|56|G014T006A032R013|no_obvious_text_conflict|
|57|G045T002A010R003|待角色/视觉复核：双方轮流用各自 right hand 打脸，pickle 两条均 `right hand -> head`；文本不构成确定错误，但需确认两条是否覆盖互相动作|
|58|G058T000A004R025|no_obvious_text_conflict|
|59|G012T002A011R009|no_obvious_text_conflict|
|60|G021T000A004R023|no_obvious_text_conflict|
|61|G026T000A001R006|no_obvious_text_conflict|
|62|G055T002A011R009|no_obvious_text_conflict|
|63|G054T006A026R033|通过|
|64|G035T005A027R011|通过|
|65|G020T005A027R012|通过|
|66|G050T007A035R002|通过|
|67|G023T002A011R011|明确遗漏：caption 明说拍背，pickle 为空；两种全局映射都没有 back↔hand|
|68|G023T004A023R017|通过|
|69|G019T000A004R027|通过|
|70|G006T004A015R002|通过（caption 明说 miss，空配对合理）|
|71|G059T005A034R010|通过|
|72|G031T005A021R008|通过|
|73|G035T003A015R004|歧义：caption 只说踩 left foot；记录把 source 扩为 left+right foot，不能仅凭文字判错|
|74|G032T003A016R009|通过|
|75|G013T000A001R010|通过|
|76|G022T003A016R006|通过|
|77|G003T001A008R016|通过|
|78|G014T001A008R023|通过|
|79|G059T005A033R007|通过（仅模仿抬手，无接触）|
|80|G009T005A028R001|通过|
|81|G037T005A025R005|通过|
|82|G044T000A004R030|通过|
|83|G020T001A008R018|通过|
|84|G011T006A035R011|通过|
|85|G024T001A008R019|通过|
|86|G012T003A019R026|通过|
|87|G025T000A000R002|通过（hug 的 broad arms↔chest）|
|88|G040T003A016R009|通过|
|89|G006T006A027R011|通过|
|90|G041T009A033R027|通过（仅模仿蹲起，无接触）|
|91|G001T003A019R024|待角色/视觉复核：双方互抓对方手臂，pickle 为 `left arm -> both hands` 与 `right arm -> both hands`；存在角色/左右对应疑点，但箭头方向不能单独判错|
|92|G017T006A033R018|通过（踢腿动作但 caption 未称接触）|
|93|G047T003A015R003|通过|
|94|G053T002A011R007|待角色/视觉复核：双方先后互拍背，pickle 为 `right hand -> back` 与 `left hand -> back`；需确认统一 P1/P2 映射下两次互拍角色归属|
|95|G013T007A030R001|明确矛盾/待视觉复核：除第一人 right hand 被记为 left hand 外，两关系还有同一腿施动者/角色归属疑点；全局交换不能同时消除，主 agent 已有几何证据|
|96|G016T003A016R006|通过|
|97|G005T000A000R006|通过|
|98|G016T001A005R001|通过|
|99|G019T006A000R023|待角色/视觉复核：hug+拍背的两关系可能存在代词指向/统一角色映射不一致，待视觉核实|

## 明确问题（保留完整 caption 与 contacts）

### pos 51 — G041T006A015R004

Caption：`One person extends his/her right foot and steps on both feet of the other person, while the other person takes a step back and then steps on both feet of the first person.`

Contacts：`right foot -> left foot, right foot; left foot, right foot -> right foot`。

两条分别为 `right foot -> both feet` 与 `both feet -> right foot`；在统一角色映射下可对应 caption 的两次互踩。不能把箭头当作 caption 中的动作方向，因此不构成遗漏。

### pos 67 — G023T002A011R011

Caption：`The person standing extends his/her hand to pat the back of the person sitting, but the person being patted turns around.`

Contacts：空列表。

caption 明确描述 hand↔back 接触；空配对在任意全局 P1/P2 映射下都无法满足。后半句转身不取消前述拍背动作。

### pos 95 — G013T007A030R001

Caption：`Two people stand facing each other. One extends his/her right hand and forcefully slaps the other person's left cheek. In anger, the second person raises his/her right leg to kick the first person.`

Contacts：`left hand (palm) -> head; right leg -> stomach`。

第一条把明确的第一人 right hand 记成 left hand；目标 `head` 是可接受的 cheek 的粗粒度归类，但左右 hand 侧不符。交换 P1/P2 后第一条变成 head↔left hand，仍不能对应 caption 的 right-hand slap；因此不是单纯关系方向问题。第二条只说 kick，未指定目标部位，`right leg -> stomach` 属于可能过窄但不能仅凭 caption 证明错误，列为重要不确定性而非额外明确错误。

## 歧义（不计明确错误）

- pos 73 `G035T003A015R004`：caption 指出踩对方 left foot；pickle 为 `left foot,right foot -> left foot`，source 多包含 right foot，但不能仅凭自然语言确定动作执行侧/检测区域，暂不判错。
- pos 95 第二条：caption 未指定被踢部位，pickle 的 stomach 不能由文字证实，也不能由文字证伪。
