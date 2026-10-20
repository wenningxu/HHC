# InterGen bug 排查与官方基线切换（2026-10-17）

**2026-10-18 22:52 SGT：新增确认官方速度/接触 loss 的 padding 差分边界泄漏，云 CPU 验收通过。** `GeometricLoss.forward_vel/forward_contact` 使用 `self.mask[:, :-1]`，因此最后有效帧到第一 padding 帧仍计入损失；应使用两端都有效的 pair mask。独立脚本抽取当前官方实际 AST，只在内存候选中替换这两处 mask，8 个合成控制加 1 个真实官方训练 item 共 9 例通过；正式复现源码与排队绑定源未改。

真实 train ID3292 有效80帧、官方补到300帧；令有效位置预测严格等于 GT，只将第一 padding 帧全部坐标增加1米，官方单人 VEL=0.374999962、padding 梯度绝对和0.749999924，pair-mask 候选二者均为0。这是人工 padding 扰动的有效性反例，不是实际模型输出误差。合成3有效帧+padding且边界扰动1厘米时，VEL/FC各0.000999999867且padding梯度非零；候选二者均0。全有效帧对照完全相同，后续padding扰动/无效timestep等控制通过。FC接触门控由预测速度/高度决定；真实样本 FC 不要求归零，也不能用此反例说它总是受边界惩罚。

此问题来自官方实现，旧custom旋转loss已使用相邻两帧有效 mask，不能解释custom比官方差。原稀疏分母和变长reduction是另一组已证实的custom差异。此次无checkpoint载入、无GPU工作、新FID或SOTA收益，未修改当前官方训练以保持基线身份。

工具 `tools/audit_intergen_padding_velocity_cpu_cloud.py` 8845B/SHA d41c009cd5881673d3cf84969aaa4d77a26cf1de5851107716f5a629c0af19e3；报告 `output/hhc_sota_20261016/intergen_padding_velocity_cpu_20261018.json` 8416B/SHA 3abbfefbaa85eaa2d2fbbef4422a9acb007c073f300a2ec165243a305403ff7c。原session32827及SSH明确exit0，云端源码/报告完整字节与SHA独立读回匹配，CUDA未初始化。第一次诊断因断言未留足官方两处分母epsilon误差而退出1，修正仅诊断精度并保留attempt1日志，不归为模型故障。未下载任何权重/生成或token数组；HHC保存4000等待的策略不变。

本轮证据已归档 `intergen_padding_velocity_evidence_20261018.zip`（20082B/SHA38002e6d6977e473ed9c1d247a6e43c7c71fb8eacb8748d6a16d1cbb2bde67eb，8成员），云端整包及逐成员SHA核验通过；Drive文件1zfsZM98yEMEagXjqsh6gVGCOXCeOa4zG的id/name/size/parent读回一致，API无checksum，不声称Drive SHA。仅代码/小JSON/文档，非模型备份；包内文档快照早于本条备份说明。回执intergen_padding_velocity_backup_receipt_20261018.json。

**2026-10-17 23:49 SGT：按用户“检查bug”补查真实文本路径，新增确认镜像方向替换错误。** `counterclockwise` 在官方、自定义旋转数据集和 HHC adapter 中都会变成 `countercounterclockwise`；云 CPU 对全部 5637 交互/16911 描述扫描，影响 60 交互/150 描述，按当前抽样规则约占训练抽样 0.4435%。六个合成例和全训练描述的三处真实 AST 表达式对照通过，原会话97960 exit0，CUDA未初始化；源码/7741B结果 SHA 独立读回通过。此 bug 由官方继承，不能解释两者 FID 差距，未改任何绑定源或接入候选。独立复核当前官方29项Python与固定提交完全相同，包装 SHA 保持；22675/25145/26133实际身份匹配，两个队列仍等待。详见 `docs/intergen_mirror_text_bug_20261017.md`。原稀疏/接触分母和时间 reduction 错误仍是已确认的自定义 loss 问题，强FK训练/弱位置头评测仍需独立消融；没有新 FID 或修复收益结果。

**2026-10-17 22:29 SGT：InterGen时间/样本reduction独立CPU反例已验收。** 除已确认稀疏空间分母错误，旧custom按micro4池化有效帧后平均组，官方几何项则先每序列平均、再平均t≤700的有效序列。抽取两侧真实reducer AST，5个合成算例exit0：长度6/2/6/2、逐帧平方误差1/9/1/9时官方5、custom3；跨组有效样本数不同可导致官方2.6/custom5，梯度权重同步改变；仅一个micro4组有效的极端构造为1/.125，不作为实际训练典型8倍缩小。等长全有效/全无效控制通过，padding与无效梯度0。sparse_v2只修空间分母，明确仍保留该时间权重差异，不能称整个objective等价官方。没有新修正版接入、重训或FID证据。详见docs/intergen_temporal_reduction_20261017.md；原exec71376终态exit0、独立SSH读回源和报告SHA一致，CUDA未初始化。官方最近14:23:16 UTC/epoch17/step6310/total.19，GPU仅22675/7880MiB，HHC saved4000及两CPU队列保持；下一普通观察≥14:33:16 UTC，继续官方基线优先，goal active/SOTA未验证。

**2026-10-17 21:11 SGT：官方基线训练与完整checkpoint已验收。** 云端CPU验收固定工具原exec64670明确exit0；取得当前可用epoch1/global_step704 checkpoint，2701618229B/SHA fbbba5ee3cbfcc50abcd1189eeb75172af83b2ee0af3fa9c40e721d239cbe5aa，云端accepted_checkpoints硬链接快照保留；输入/输出头262D、196项optimizer状态全部step704、权重与optimizer均有限、scheduler warmup10/max_iters2000通过，官方源码与运行包装SHA均匹配。仅云端存权重，不下载本机。默认官方Lightning checkpoint未含顶层RNG，不能声称bitwise恢复；此处验收权重/optimizer/scheduler/counters完整性。

13:11:32 UTC实际trainer22675活跃，epoch2/step880，LR3e-5、官方total .6334，GPU仅22675/7880MiB；首次epoch checkpoint与后续更新都已发生，不能再把此run标为仅launch。CPU观察器21986的/proc也已独立验明存活，每600秒观察不恢复。下一普通复核≥10min，仅验收/异常提前；早先root定时exec64670已收终态不重启。旧50420/cell1053只读旧run观察仍过时，不能作为官方job句柄或续训依据。

真实稀疏loss CPU复核v2已由root审源码、检查归一化/请求重放/零mask项为零/其余loss逐位不变，并对云端源码与报告SHA独立读回一致：有非零affinity的3请求，修正项/旧项335.27–415.85倍；contact2.00–4.41倍。v1曾混淆物理量/归一化且GT构造错误，已明确INVALID，不引用其数值。v2报告13359B/SHA057fab6a...d159b9，结果只量化同样本损失缩放，不是新FID，也不能断言解释所有退化。独立loss修正版未接入任何训练，官方基线始终使用官方loss。

SOTA/正式HHC生成比较未完成；HHC仍saved4000且未启动，官方baseline最终完成后的新HHC门控待安装，不能启用旧自定义704k门控。后续先维护官方baseline稳定运行/里程碑评分与新queue，再做单变量表示实验。完整证据见docs/intergen_bug_audit_20261017.md、output/hhc_sota_20261016/official_intergen_startup_checkpoint_acceptance_20261017.json及official_intergen_observer_status_20261017.json。

用户已授权：保存并暂停自定义旋转/FK实验，先复现官方 InterGen，并要求检查 bug。旧实验在103000步完成权重/optimizer/RNG校验后有意退出；HHC仍保存4000排队。所有GPU诊断均在既有云端4090执行，无新增本机权重或生成数组下载。

## 已确认的权重迁移错误

自写 `intergen_rotation_objective.py` 及其 grouped 版本中，`fk_distance`、`fk_affinity` 在乘稀疏距离mask后，调用对全部22×22=484项求均值的 `avg`。官方先除以每帧选中项数。沿用系数3并不能保持几何项尺度：当前贡献等于“选中项均值 × K/484”。GT/pred距离mask本身的差异与这个分母错误是两个问题。

接触项也把两人全部8个脚点作为分母；官方分别按每人的有效接触数归一化，再将两人loss相加。新修正版明确保留每人归一化后求和，避免把合并两人均值误称为官方总loss。

云端CPU数值回归已通过（SSH exit0，CUDA未初始化）：

| 固定合成算例 | 旧实现 | 独立修正版 | 官方同一mask/输入参考 |
|---|---:|---:|---:|
| 484对中仅1对有效，误差平方1，系数3 | 0.00619835 | 3.0 | 2.99999905 |
| 两人各1个有效脚点，xyz误差平方各1，系数30 | 7.5 | 60.0 | 59.99997711 |

这是构造反例的实测数值；不能声称所有真实样本都缩小484倍，实际系数取决于稀疏率。测试另外检查了稠密/空mask、稀疏梯度、其余loss项完全不变、完整loss梯度有限、padding不变性和padding梯度为0。

修复文件 `tools/intergen_rotation_objective_sparse_v2.py` 为独立版本，3608B/SHA f7565755db1ae68298dd5ab5ea12da36bdf7077311d8e0fee898a62dedb81856。未改旧训练源码、检查点或结果，也未将其接入官方基线。它仅修正分母与接触按人求和；GT距离mask、grouped valid-frame reduction、其余自定义项仍保留，不能称整个objective等价官方。

回归工具 `tools/check_intergen_sparse_reduction_cpu.py` 4931B/SHA81562107df0fdbbb293b29751c8834968d85738129f0112943941050d41a0e9f。完整源码上传云端并按字节/SHA读回一致。结果 `output/hhc_sota_20261016/intergen_sparse_reduction_cpu_regression_20261017.json` 1208B/SHA991abe1aea5299686febb72cd911d30a5a2927b0eb28ce7871e1a944ff4f3919。

## 生成输出与去噪检查

在已暂停的20k最佳和103k最后权重上，抽取8个固定validation请求，使用相同噪声种子比较。生成仅消费文本和长度；生成FK使用中性零beta SMPL骨架，不使用GT shape。GT shape只用于另行标记的teacher-forced去噪诊断。没有计算小样本FID或声称改善正式成绩。

| 八个请求等权平均 | 20k | 103k |
|---|---:|---:|
| 生成直接位置与中性骨架FK非root关节差异/mm | 74.715 | 47.984 |
| t10去噪直接位置MPJPE/mm | 62.495 | 41.617 |
| t10去噪FK位置MPJPE/mm | 45.115 | 24.388 |
| t500去噪直接/FK MPJPE/mm | 81.645 / 70.000 | 59.362 / 49.673 |
| t900去噪直接/FK MPJPE/mm | 236.938 / 234.434 | 194.736 / 188.230 |

生成FK差异含中性骨架与直接位置隐含个体体型的差异，因此不能全部归为旋转预测错误。训练以较强FK loss监督、评测却直接取0.1辅助位置通道的口径不一致已被数值诊断支持，但差异随训练变小、去噪误差也下降，仍不足以解释FID从约24到35的全部变化。没有找到重复归一化或明显通道错位的证据；这不等于证明所有代码无bug。

GPU诊断21104已完成且/proc进程不存在；云端raw268数组保留，未下载。报告20635B/SHAcc60fac8dce4a6414634e3504c1bee72c1125cf2d63944f2fd5874d18d6f2516，对应本机小JSON `intergen_heads_bug_diagnostic_report_20261017.json`。诊断源码6294B/SHA191acfb8bb1fd222c17890b73d354c28c94a3e903ee32b3c1972103223c8838f。

## 已排查范围与官方基线

云端CPU实际对照8个训练ID、64种镜像/锚点/输出人物顺序，前262维与未改官方Dataset.__getitem__逐位相同，mean/std相同。采样未覆盖长裁剪分支，不能推广为完整随机训练分布已证明一致。

官方基线源自固定提交 b1750df4a859443b8a69329f759ce3a7503ae179 的完整git archive；Python训练/网络/损失源码不改，直接运行 `python -u tools/train.py`。只有训练配置的输出路径/名称和checkpoint保存间隔（100epoch改为每epoch）变化。保留262D、官方loss、batch32、FP32、2000epoch和官方按epoch学习率调度，RESUME为空。原入口没有固定seed，未额外加入seed包装。

独立Python3.8环境采用官方torch1.13.1+cu117、Lightning1.9.1、numpy1.23.5、scipy1.4.1、matplotlib3.2.0。为修复旧框架当前依赖解析的导入问题，固定FastAPI0.88.0、Pydantic1.10.13、Starlette0.22.0、httpx0.23.3、lightning-cloud0.5.20，并设置MPLBACKEND=Agg。CPU导入预检已通过，未修改官方Python代码。

首次启动在任何更新前因os.walk不跟随嵌套数据目录符号链接而数据量为0；修正为指向实际含motions_processed目录的顶层视图。第二次启动在更新前被原始训练注释非UTF8编码阻断；正在用官方支持的ignore_list排除损坏注释，并与现有适配器的完整训练ID集合核对。不得将这些初始化失败当作训练进度或官方不能复现的证据。后续实际训练验收以LAUNCH.json、OBSERVER_STATUS.json、/proc及训练日志为准。

整体HHC/SOTA目标仍进行中；上述bug修复尚无重训后的评测收益证据。

## 后续启动验收与显存处理

官方Dataset的CPU预检最终确认5637个训练ID/11274条含镜像样本，与当前adapter完整训练ID集合相同。损坏注释仅训练ID5834（SHA579b429ff71d56e984c44bc9c2b46188c50ef76c7261981a9e4d3e0d1b9f8fc9），通过官方支持的ignore_list排除；没有根据模型指标筛选数据。

第三次启动在首个batch前向中因batch32/FP32显存不足退出，未完成任何optimizer更新。为保留batch和数值精度，新增外部执行包装 `tools/run_official_intergen_checkpointed_cloud.py`；它只给官方TransformerBlock加入保留RNG的非重入激活重计算，再经runpy执行未改的官方tools/train.py。官方训练循环、DataModule、loss、optimizer及scheduler仍来自原始源码。包装不是新的自写训练循环，也不能隐瞒为毫无运行时改动。

显存重计算GPU对照已通过：完整8层、FP32、dropout0.1、双人、含padding，并将测试用输出头置为非零以确保内部层均有梯度。输出、170个参数梯度、输入/条件梯度最大差均0；反向后CPU与CUDA RNG完全相同。probe进程及SSH exit0。结论限于该对照，不声称多步训练bitwise相同。包装4582B/SHA54a4b2759c638fd38b0a662cdef3d12b71ac14809d3a00d4fc2b769d0a2e48f3；正式入口不包含测试用非零输出头或测试随机种子，这些仅在--probe独立进程执行。

当前attempt4已启动，PID22675/birth467827326；仍待训练日志实际更新验收。三个失败attempt均保留日志与启动记录，不作为模型效果结果。新官方训练不会使用修正后的自定义loss；其稀疏分母来自原始官方loss。

## 真实生成缓存的稀疏分母复核

已对同一103k缓存的8个固定请求做云端CPU复核：读取物理量generated268，先转换为normalized prediction；按各请求的实际裁剪/锚点/输出人物顺序重放真实268D GT和offsets，并断言前262D与冻结请求解归一化一致。当前8个validation请求未做镜像。K统计使用GT直接位置，与objective对应；root另验了零mask对应零loss、其他loss项完全不变与结果源码SHA一致。

实际修正版/旧版loss比值：distance在非零项中约1.17–2.32倍，另一个稀疏请求479为97.46倍；三个有非零亲和mask的请求0/136/68，affinity分别为398.79/335.27/415.85倍；contact为2.00–4.41倍。这是同一生成结果、同一GT上只换分母的损失数值对比，不是重训后的FID改善。

可信结果为 `intergen_sparse_masks_actual_cpu_v2_20261017.json`，13359B/SHA057fab6a538f30242c92268f20275360ec97ca4bc610ff10cdf0a0d0d5d159b9；当前CPU脚本SHA55198e1fc70632f9caeb355e6c1e7fa729b68b5a9cebd8df2b36bcb49db5ad0e。云端执行exit0，CUDA未初始化。最初v1审计脚本混淆物理量/归一化并错误构造GT，root验收拒绝，已有v1_INVALID sidecar，v1不属于有效实验结果，不入对比表。

## 官方基线首批更新验收

2026-10-17 13:00:32 UTC：trainer22675/birth467827326实际活跃，官方epoch0/step40，LR1e-5；total5.2207、simple1.3785、DM2.765、JA.1465、RO.005，均有限。GPU仅22675，占7880MiB。显存方案已通过真实训练首批更新；尚无正式评测或完整epoch checkpoint验收，不称复现达到约6或SOTA。

## 2026-10-17 21:39 SGT：恢复控制与正式复现证据门控

**2026-10-17 21:39 SGT：恢复控制补强已在云端验收，训练侧既有bug结论不变。** 新CPU等待器25145/birth468066055（tools/resume_hhc_after_official_intergen_cloud.py，13950B/SHA8e61c066cd2061955d707a463970e917e24d585ff34619d4665ce3d7d40d64ab；配置migration/control/hhc-after-official-intergen-resume.json，SHA cc58d8299bef03b751fd7c22cf98e9aa0ed3ed14d9c210bf9969e2f653050c2f）实际/proc与状态均为waiting_for_official_training、无phase/exitcodes/launch-intent。它仅在官方完整704000 checkpoint与训练终态通过验收、真实原始20rep+MM评测经源码/权重/逐rep日志核对并获得审查收据后，才从A4000续训并顺序B/C；此收据由实际实验审查产生，不伪造结果，也不要求人重复批准已授权排队。原始官方eval20执行队列与真实结果仍待完成，不能把门控安装当作复现成功。

控制bug修复：在Popen前创建并fsync不可覆盖的launch intent，再写phase，之后记录child PID+birth；即使STATUS仍停留waiting、或丢失，也因intent而拒绝自动重放。纯CPU合同测试1个评测正例/32个拒绝例及4个等待恢复正例/19个拒绝例，在本机与云端均exit0。首次实际预检因无关容器服务/proc/600/cwd不可读而失败；新源允许cwd读取被拒为None，不更改训练源，第二次预检exit0、release_gate=false。4000完整3.43GB snapshot再次CPU核对bytes/SHA/contract/optimizer/RNG，并在云端保存不可变硬链接；无本机checkpoint下载。原来的saved_hhc_snapshot已有幂等判断，不采用子agent误报的FileExistsError结论。

最近真实官方质量观察为13:34:15 UTC/step2590、epoch7、LR8e-5、total.2947；trainer22675/birth467827326仍匹配，GPU仅22675/7880MiB。880→2590这段约.80秒/步（包含保存），没有FID。旧103k自定义run保持暂停、HHC保持4000，原CPU下游8321/9975绑定源未改，两项app automation保持PAUSED。稀疏分母修正版仍只作独立回归，不接入官方loss；不能用loss下降宣称恢复原文约6或HHC SOTA。

证据：output/hhc_sota_20261016/hhc_official_resume_gate_{upload_receipt,preflight_v2,launch,acceptance}_20261017.json，official_intergen_release_gate_cpu_tests_{local,cloud}_20261017.json，official_intergen_step2590_observation_20261017.json。新源完整上传后逐文件大小/SHA读回匹配；没有原地修改运行中的训练或旧绑定29项HHC源码。


## 长动作裁剪边界补查

**2026-10-17 21:41 SGT：长动作裁剪边界补查通过。** 训练集中357个动作超过300原始帧，按固定训练ID顺序取首/中/末3个长动作2576、7596、2111，分别验证起始/中间/最后可选裁剪，再组合镜像、锚定人物和输出人物顺序，共72例。真实未改官方Dataset.__getitem__与当前前262维，FP32逐位一致且原始最大误差0；mean/std同样逐位一致，CUDA未初始化。原exec87857与SSH均exit0。不是全357动作穷举或完整随机分布等价证明，也不涉及root追加6维与生成FID。

独立新工具tools/audit_intergen_rotation_longcrop_cpu.py SHAe467a9dff87c3b059fa98ae77e4fca0dad7ed3991770a0aa54705c417a78d37b；云端完整源码bytes/SHA读回通过。报告41482B/SHAae3abcb541071868c50c9ad83415aafece0aab4ed5de591901d053aa64006d3b，本机小JSON intergen_longcrop_parity_cpu_20261017.json，root再次断言3IDs×3starts×8variants、299有效帧、所有通道误差0并与云端SHA核对。只下载此小结果，没有权重或生成数组。已查范围内不支持“长裁剪/归一化错位导致退化”；有明确数值证据的训练bug仍为稀疏分母错误，修复收益待单变量重训。官方基线继续独占，新CPU等待25145已验收、HHC无GPU子任务。

