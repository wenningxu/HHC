# HHC / InterHuman：指标改进与基线比较

2026-10-16，新加坡时间。目标仍在进行；**尚未达到、也未验证达到 SOTA**。

备份与首个10k观察：official_intergen_monitor_transition_evidence_20261017.zip，434666B/SHA5e102a27e49d15717ed28dbb10eaad44078516da102394da6c2e8124cde94c3f，13文件+MANIFEST共14成员，云端整包/逐成员SHA已验；Drive文件1T5thB23dP1ZtVGvkSp1OoFSUbXjuAl2D，大小/父目录读回一致，未返回checksum，不声称Drive SHA。仅本机监控配置快照/小JSON/文档，无模型/数组，不是权重异地备份；收据official_monitor_transition_backup_receipt_20261017.json。ZIP文档快照及9780观察早于本条，不把后来的10560记录说成已在该固定包内。

15:19:55 UTC固定observer已实测官方epoch29/step10560、LR1e-4、total.167有限，GPU仅22675/7880MiB，滚动checkpoint文件epoch29-step10560存在（仅文件观察，未重新载入验收）。9780→10560的780更新/632.2815秒约.81062秒/更新，包含日志/保存，不是与旧custom等条件性能对照。官方尚无FID，HHC4000及25145/26133等待/automation PAUSED不变。本机小JSON official_intergen_first10k_observation_20261017.json单独记录新快照，云端原始observations.jsonl仍保留；未下载checkpoint或数组。下一普通观察≥15:29:55 UTC。goal active，SOTA未验证。

**2026-10-17 23:17 SGT：修正了两项实际过时的监控/归属记录。** 既有hhc与hhc-runpod automation仍PAUSED，但原prompt还指向旧custom fast恢复器与旧HHC门控；已通过应用automation_update原id更新为官方262D/FP32/batch32/704k、现行25145/26133及真实原始20rep/MM独立review门控。逐字段TOML读回证明只有prompt与updated_at改变，原name/rrule/target_thread_id/created_at及PAUSED保持，未执行调度或启用任务。新完整提示保存migration/control/official-intergen-automation-updates.json，SHA df5d6ad8dc086d1029cb6ad32e15e588e52058c0579df8f34a3d37d699b29664。

本机compute-owner已指向官方run，却仍把旧custom100k FID35.239及best20k/24.314挂在顶层，且current_hhc_waiter_pid空、control指旧fast。已将这些FID原值归入previous_custom_validation_metrics，官方last/best validation及baseline_fid统一null待评，填写真实25145/26133与21986进程身份和现行control。本机runpod-idle旧2353/旧ABC记录也改为当前观察，idle_since=null、action keep_running、没有stop/start动作；这些是本机状态清单，远端migration/下原本没有副本，未凭空创建远端控制配置。旧hhc-compute-priority及全部已绑定配置仍保持原SHA。

实际验收：6项本机/云端核心source/config SHA完全保持；/proc22675/467827326、25145/468066055、26133/468171108、21986/467800611全部S，两个等待器均waiting_for_official_training且无phase/child/intent；8321/9975下游纯等待也实际核验。Windows未发现旧local_runpod_control(_fast)或local_compute_control Python进程。15:09:23 UTC固定observer官方epoch27/9780、LR1e-4/total.185有限，GPU仅22675/7880MiB；该run无FID。下一普通观察≥15:19:23 UTC。没有改训练/排队绑定源、下载本机权重/数组、启用automation或新建资源。

上轮progress为ARMS源码/CPU条件反例；本轮progress为实际stale automation和跨实验FID归属修复，并有live trainer/queue verified wait。SOTA与完整HHC生成结果仍未完成、goal active。详见docs/official_intergen_monitor_transition_20261017.md；原始before/after/acceptance小JSON在output/hhc_sota_20261016/official_monitor_*_20261017.json。

备份验收：interhuman_frontier_source_cpu_evidence_20261017.zip，539809B/SHAe9789010051bf62675c6dcfc7b482618cd5fa34c93ef538af8578bc8ae08117b，34文件+MANIFEST共35成员，云端整包/逐成员bytes与SHA均读回匹配。Drive文件1HSPypYD9sZAznxW3cOJ60B9NoYopHV3m，大小/父目录读回一致，API无checksum，不声称Drive SHA核验。只含源码/小JSON/文档（含失败probe及CRLF副本记录），没有权重/生成或token数组，不是模型异地备份。收据interhuman_frontier_source_cpu_backup_receipt_20261017.json；归档docs快照早于本条。CPU probe8586与最终验收均exit0，无运行诊断句柄待收。ARMS公共MM需未来独立出口、真实权重仍未验；HINT初始化继续未知，源/访问条件未变时不重复同一403/401查询。主线仍优先完成官方InterGen训练及20rep/MM，然后HHC真实选模和生成对照，不增加未验证变体；goal active，SOTA未验证。

**2026-10-17 23:01 SGT：SOTA前沿输入条件有新的源码证据。** HINT定位到作者OpenReview note CGALrQfkYi及代码补充ZIP字段，但文档指定的附件端点403、正文/forum API403，未下载附件、未独立读其初始化实现；继续保留online/initial-prefix未知，不能只凭InterCMDM的GT-first-frame说法删去3.100。ARMS官方commit e3819179209d9f3d4b717673af4f839f5b2bdf44的12份源码通过固定git tree/blob/SHA核验，实际InterHuman评测只传text/ids_length/采样参数、默认context=None，首段高斯噪声；此入口允许作为text-only比较候选，不能把所有AR模型都视为GT前缀条件。

云CPU字面AST跟踪8586最终exit0：替换GT motion为999后，生成调用参数/替身输出不变；实际DiT.generate及mask/schedule helper首段context空、下窗历史来自自身生成。没有载入真实网络/权重，VAE/denoiser/坐标转换均替身，采样控制流probe用CFG1，不能当CFG2.5/GPU或FID复现。另确认ARMS默认MM samples=0且评分调用注释；仅启用MM时batch2×repeat3触发index2越界，repeat-major拼接与原B条length/text索引不匹配。未来共同MM必须独立prompt/repeat出口，不可直接打开原分支；不据此否定论文FID。HF kksix/arms未认证元数据401，checkpoint尚未取得或验身份；云env未安装HF hub库，本轮没有查询/使用认证token，也不能把401说成仓库不存在。

诊断首轮90937因本工具替身返回numpy而失败，非ARMS模型错误；已保留源码/失败sidecar，修正替身Tensor后正式probe通过。最初验收29778因本机重序列化CRLF而SHA不同，已保留副本、确认JSON内容与仅换行差异，收取云端原始字节后源/报告SHA全部独立匹配。有效报告3701B/SHA099658693ede1f8e8255974d116895b93b8dda61886271c9da5b919a3f97b93a，工具10051B/SHA707223e3ca35fd353645aae1db90160991b9f0367caaa90df41c8c86141ec36f。新frontier清单5658B/SHA49cc285a...c9468c及docs/interhuman_frontier_source_audit_20261017.md记录边界。上一轮progress（公共身份来源）；本轮progress（ARMS无GT前缀入口与MM真实控制流反例），同时官方训练verified wait；不标SOTA完成。

14:58:06 UTC固定observer实际trainer22675/birth467827326、epoch25/step8920、LR1e-4/total.1623有限，GPU仅22675/7880MiB。HHC仍saved4000、25145/26133排队、automations PAUSED与已绑定训练/评分源不改。本轮新增GPU任务0、本机checkpoint/生成或token数组下载0；下一普通观察≥15:08:06 UTC，goal active，尚无新HHC生成FID或完整test/MM。

备份验收：public_interhuman_baseline_identity_evidence_20261017.zip，435154B/SHA6a0393f9fd4e65e65ec8f83e2258aff29b99feb312da7d8c1e5931217859f197，14文件+MANIFEST共15成员；云端整包与逐成员bytes/SHA读回通过。Drive文件1ZhnmgLxgQFtt3h5PBdwRdX_QO148EqOq，大小/父目录读回一致、无API checksum，不声称Drive SHA核验。仅公开来源元数据/文档/小JSON，没有权重或数组，不是模型异地备份。收据public_baseline_identity_backup_receipt_20261017.json；归档文档快照早于本条。原归档exec42825已exit0；正式训练/队列未改。本轮关闭已记录公共资产路径与版本归属的不确定项，下一独立CPU事项可核验HINT官方初始化代码，避免仅引用其他论文的GT-first-frame判断；不在源未变时重复本轮搜索。官方最近7300、无FID；SOTA仍未验证，目标active。

**2026-10-17 22:40 SGT：公开基线身份清单与原始来源核验完成。** 本轮独立HTTP读取24个已记录端点，全部200：TIMotion/CoSh各1branch、0tag/0release；TIMotion全部7个issue及7个comments页保留真实body、作者、OWNER身份、时间、响应SHA。此前agent JSON中的body/replies是改写摘要，已加明确warning，不能当作者原话；以新public_baseline_asset_routes_raw_verified_20261017.json（133187B/SHA41070731...16c7cd7）为准。新增核对issue7：作者询问Transformer backbone，讨论对象为公开Transformer+LPA；不能用它声称RWKV4.702复现。HF TIMotion论文关联模型0，TIMotion/CoSh名称搜索及作者AliManjotho模型列表均HTTP200空；有限范围内没有新匹配权重，不推断所有渠道不存在。CoSh项目Paper/arXiv/Video href为#；DOI的Crossref出版元数据为32(7):5911–5924，纠正项目bib的32(4):1–14，但不把占位链接视为论文本身不存在。

新增机器可读public_baseline_comparison_identity_20261017.json（11334B/SHAc99facc610bd640519045480b70212b8f90d7fb50f7259467414c2cc7874a48b）重验三条既有480val报告SHA/metrics与comparability收据一致，独立标注public Transformer、paper RWKV、CoSh text+shape条件/独立权重未验。没有重载embedding/重跑生成，本轮未新增FID。三条val FID仍InterGen8.762189/InterMask8.931384/TIMotion public8.721812；不和论文test值混排。HHC无GT-shape真实选模、20rep main/MM及SOTA尚未完成。上一轮分类progress（loss反例）；本轮progress（原始来源/遗漏issue7及HF作者路径闭合、身份目录）并有verified wait。

14:36:27 UTC固定observer实测官方trainer22675/birth467827326，epoch20/step7300、LR1e-4、total.1653有限，GPU仅22675/7880MiB。随后CPU-only /proc核对25145/birth468066055及26133/birth468171108均S/对应源码命令/STATUS waiting_for_official_training，无GPU子任务。两项automation仍PAUSED、HHC saved4000及绑定源不改；下一普通训练观察≥14:46:27 UTC。仅源码/小JSON读取与记录，没有本机checkpoint/生成数组下载、额外GPU或新资源，目标active。

备份验收：本轮InterGen时间/样本reduction诊断源码、小报告及文档包intergen_temporal_reduction_cpu_evidence_20261017.zip，410248B/SHA17af3a0df6d78ad20f00bc2ef74a634728f14e29322ffc6643976368fd7e2da6，14文件+MANIFEST共15成员。首次SCP仅归档传输120秒超时（原93974最终exit1，孤立本机SFTP辅助进程按完整身份核对后停止，未操作训练）；后续SSH stdin完整上传并逐成员读回验收，partial不作成功。Drive文件1aTPLlI8pOzuRPfJXST45fIQIop_87_rK，大小/父目录读回一致，API无checksum，不声称Drive SHA。仅源码/文档/小JSON，没有权重/生成或token数组，不是模型异地备份。收据intergen_temporal_reduction_cpu_backup_receipt_20261017.json；归档docs快照早于此备份收据段。主诊断71376 exit0且独立源/报告SHA复核已通过，未加入自定义训练新版本。goal active、官方基线优先、SOTA未验证。

**2026-10-17 22:29 SGT：InterGen时间/样本reduction独立CPU反例已验收。** 除已确认稀疏空间分母错误，旧custom按micro4池化有效帧后平均组，官方几何项则先每序列平均、再平均t≤700的有效序列。抽取两侧真实reducer AST，5个合成算例exit0：长度6/2/6/2、逐帧平方误差1/9/1/9时官方5、custom3；跨组有效样本数不同可导致官方2.6/custom5，梯度权重同步改变；仅一个micro4组有效的极端构造为1/.125，不作为实际训练典型8倍缩小。等长全有效/全无效控制通过，padding与无效梯度0。sparse_v2只修空间分母，明确仍保留该时间权重差异，不能称整个objective等价官方。没有新修正版接入、重训或FID证据。详见docs/intergen_temporal_reduction_20261017.md；原exec71376终态exit0、独立SSH读回源和报告SHA一致，CUDA未初始化。官方最近14:23:16 UTC/epoch17/step6310/total.19，GPU仅22675/7880MiB，HHC saved4000及两CPU队列保持；下一普通观察≥14:33:16 UTC，继续官方基线优先，goal active/SOTA未验证。

备份收据：本轮HHC END CPU诊断源码与小证据包hhc_end_boundary_cpu_evidence_20261017.zip，430160B/SHA058c626811076d0c8e7236c1d172e0951e0e19e5f13545729e3d0dcee6162e45、20成员。7项源码与云端诊断报告中的实际源码SHA匹配，云端整包/成员SHA均验收；Drive文件1_snRoI0qCdCfTF00mocaC01svvRtY-O4，大小/父目录读回一致，未返回checksum，不声称Drive SHA核验。只含代码/文档/小JSON，无checkpoint、token cache、生成数组，不是权重异地备份。收据output/hhc_sota_20261016/hhc_end_boundary_cpu_backup_receipt_20261017.json；归档docs快照早于本条。CPU诊断85432已exit0，不重跑；不晋升END改动，继续已注册CFG/window/无GT shape验证路线。官方最近5350/暂无FID，目标active，SOTA未验证。

**2026-10-17 22:10 SGT：HHC END边界CPU诊断完成，不晋升改动。** 训练TokenData输出76位置且最长75个有效token后含END；现有sample_intra_cas只分配75位置，n75时缺END。实际生产采样方法的独立假logits输入追踪确认这一差异（假logits只验布局，不作为模型效果）。训练真实cache4567条，未删首/尾时215条达到75、删除1token后210条；固定val480中35条（7.29%，均L299）达到此边界。root重新计算验证请求长度，云CPU诊断独立重算cache计数；subagent原直方图标签误写已纠正为validation_processed_length_hist，数值没有据此当训练token直方图引用。

对实际保存A4000权重/冻结VQ170k，选择3个真实训练交互4789/6952/2111，2种mask输入、n74控制/n75边界，共12例，云CPU FP32/2线程/nice15前向。n74仅多一个被屏蔽PAD的6个控制，conditional及CFG3.5 logits逐位相同；n75的6个对照有变化，conditional最大logit差.19413、CFG最大.66248、CFG最小top1一致率98%。补END的CFG masked teacher-forced NLL有1例好/5例差，等权平均delta+0.004037857 nats；conditional为2好/4差、均值-.000385553。没有一致收益，不改已绑定A/B/C/五候选队列、不新增GPU采样消融。诊断仅3个train交互/早期A4000/CPU FP32，不是独立6样本、最终val或GPU BF16结果；小logit变化仍可能经迭代放大，不能据此断言完整生成FID一定不受影响。

原exec85432、SSH和诊断均exit0，CUDA未初始化；没有GPU模型推理、训练源码/权重改写或本机数组/权重下载。报告17214B/SHAabd92645a4014f46252a01825780b70637604e598d4ad835c333fb29acb0e990，源码8547B/SHAb9c1288b301cb8e2a2a8cbec081b9ac45a6754367e781308b1eb744277ee6625，云源码与小报告均按bytes/SHA读回通过。详见docs/hhc_end_token_boundary_20261017.md及hhc_end_boundary_cpu_summary_20261017.json。

14:10:44 UTC实际官方trainer22675/birth467827326推进epoch15/step5350、LR1e-4、total.2222，GPU仅22675/7880MiB。无FID；HHC仍saved4000，官方原始eval20等待26133、HHC等待25145不变，两项automation保持PAUSED。下一普通训练观察不早于14:20:44 UTC；优先等正式HHC matched-CFG/window和text-only验证，不能从本次局部诊断宣称SOTA。目标active。

备份验收：官方eval20队列源码/小证据包official_intergen_eval20_queue_source_evidence_20261017.zip，475954B/SHA0f4298fa6a16e26cc844edbf00d25b2ed8e73729c357fcd0ec2a273d460652f2、62成员，其中37个官方文本源码与云端绑定评测源码逐字节一致。云端整包/逐成员SHA通过；Drive文件1uTKq3klnaNW2vIOKJpMFolNOheZ0gwZY，大小/父目录读回一致，未返回checksum。只含源码、配置、文档、小JSON，无权重/生成数组，不是模型异地备份。收据output/hhc_sota_20261016/official_intergen_eval20_queue_backup_receipt_20261017.json；归档docs快照早于本条回执。现有21986后台观察600秒继续，主线程下一普通观察不早于14:07:43 UTC（验收/异常可提前），没有新FID不重复报成绩。当前train22675、HHC CPU waiter25145、eval CPU waiter26133均已实际验明；不重跑setup/队列，不恢复旧custom103k。goal active，SOTA未验证。

**2026-10-17 21:57 SGT：官方原始eval20队列已安装并验收。** CPU等待器26133/birth468171108，tools/run_official_intergen_reproduction_cloud.py（10388B/SHAf4794c59946074660a29ef04d4cf25d8ad9805d7814227d472cb11ab81767ba0），配置migration/control/official-intergen-eval20-after-final.json（2502B/SHAe3b524eaccdc0168b577c8c35be436426fe49d144b0325b9c3cdf0b551317158）。实际/proc/STATUS为waiting_for_official_training、dispatch_started=false、无评测child或launch intent；HHC等待25145/birth468066055也真实存活且无phase。此前“eval执行队列尚待准备”已被本条覆盖。

新官方评测源码独立固定commit b1750df，所有Python原字节不改，只改model.yaml的checkpoint路径；不触碰正在训练的源码/数据。CPU实际原始Dataset得到1098条、batch96/drop_last=1056每rep，首batch形状96×300×262/有效长度14–299；生成模型346键及官方InterCLIP207键strict load成功，CUDA未初始化。只用官方ignore_list排除非UTF8测试注释4642，不按指标筛样本。配置参数原样20rep、MM100×30、MMpairs10、Div300、DDIM50/CFG3.5。CPU探针加载的是既有704-step架构检查快照，不是正式评测分数。setup原exec81809 exit0、执行队列v2测试3个恢复正例/24个拒绝例和实际预检exit0。

唯一已有HHC CPU等待器25145负责最终704k权重验收；26133只消费该收据，避免并发创建硬链接。随后在训练终态且云GPU空闲时运行未改tools/eval.py，生成原始20轮日志和解析报告；不自动生成review收据。root需审实际结果/日志/权重身份，才能让HHC从4000继续。此评测是独立官方复现审计；HHC统一fair-test仍须真实val选模后另行执行，不能拿它直接代替公平比较表。

最近实际观察13:57:43 UTC：trainer22675/birth467827326，epoch12/step4370，LR1e-4、total.2420，GPU仅22675/7880MiB。3360→4370约.791秒/步含保存；没有FID或HHC改进成绩。两项automation保持PAUSED，无新增本机权重/数组，无HHC或评测GPU子任务。详见docs/intergen_official_reproduction_20261017.md和小证据official_intergen_eval_queue_acceptance_20261017.json、official_intergen_eval_queue_training_observation_20261017.json。目标active，SOTA未完成。

备份收据：本轮long-crop/新HHC门控源码与小证据归档intergen_bug_longcrop_and_queue_control_20261017.zip，424351B/SHA3c5098a2cf40129249e3b75d0947db87eb88e08fa9d54f3721104807f57edf21、27成员；云端归档及全部成员SHA验收exit0。Drive文件11pQY2_GyFlOhhHw7c-agvucASQZY5IOG，大小与父目录已读回一致，未返回checksum，不声称Drive SHA核验。没有模型/生成数组，不是checkpoint异地备份。小收据output/hhc_sota_20261016/intergen_longcrop_queue_backup_receipt_20261017.json。归档docs快照早于本条回执。CPU新队列25145已验收等待、longcrop原87857 exit0，官方原22675继续独占；原始官方eval20执行队列/真实结果仍待后续完成，goal active、SOTA未验证。

**2026-10-17 21:41 SGT：长动作裁剪边界补查通过。** 训练集中357个动作超过300原始帧，按固定训练ID顺序取首/中/末3个长动作2576、7596、2111，分别验证起始/中间/最后可选裁剪，再组合镜像、锚定人物和输出人物顺序，共72例。真实未改官方Dataset.__getitem__与当前前262维，FP32逐位一致且原始最大误差0；mean/std同样逐位一致，CUDA未初始化。原exec87857与SSH均exit0。不是全357动作穷举或完整随机分布等价证明，也不涉及root追加6维与生成FID。

独立新工具tools/audit_intergen_rotation_longcrop_cpu.py SHAe467a9dff87c3b059fa98ae77e4fca0dad7ed3991770a0aa54705c417a78d37b；云端完整源码bytes/SHA读回通过。报告41482B/SHAae3abcb541071868c50c9ad83415aafece0aab4ed5de591901d053aa64006d3b，本机小JSON intergen_longcrop_parity_cpu_20261017.json，root再次断言3IDs×3starts×8variants、299有效帧、所有通道误差0并与云端SHA核对。只下载此小结果，没有权重或生成数组。已查范围内不支持“长裁剪/归一化错位导致退化”；有明确数值证据的训练bug仍为稀疏分母错误，修复收益待单变量重训。官方基线继续独占，新CPU等待25145已验收、HHC无GPU子任务。

**2026-10-17 21:39 SGT：恢复控制补强已在云端验收，训练侧既有bug结论不变。** 新CPU等待器25145/birth468066055（tools/resume_hhc_after_official_intergen_cloud.py，13950B/SHA8e61c066cd2061955d707a463970e917e24d585ff34619d4665ce3d7d40d64ab；配置migration/control/hhc-after-official-intergen-resume.json，SHA cc58d8299bef03b751fd7c22cf98e9aa0ed3ed14d9c210bf9969e2f653050c2f）实际/proc与状态均为waiting_for_official_training、无phase/exitcodes/launch-intent。它仅在官方完整704000 checkpoint与训练终态通过验收、真实原始20rep+MM评测经源码/权重/逐rep日志核对并获得审查收据后，才从A4000续训并顺序B/C；此收据由实际实验审查产生，不伪造结果，也不要求人重复批准已授权排队。原始官方eval20执行队列与真实结果仍待完成，不能把门控安装当作复现成功。

控制bug修复：在Popen前创建并fsync不可覆盖的launch intent，再写phase，之后记录child PID+birth；即使STATUS仍停留waiting、或丢失，也因intent而拒绝自动重放。纯CPU合同测试1个评测正例/32个拒绝例及4个等待恢复正例/19个拒绝例，在本机与云端均exit0。首次实际预检因无关容器服务/proc/600/cwd不可读而失败；新源允许cwd读取被拒为None，不更改训练源，第二次预检exit0、release_gate=false。4000完整3.43GB snapshot再次CPU核对bytes/SHA/contract/optimizer/RNG，并在云端保存不可变硬链接；无本机checkpoint下载。原来的saved_hhc_snapshot已有幂等判断，不采用子agent误报的FileExistsError结论。

最近真实官方质量观察为13:34:15 UTC/step2590、epoch7、LR8e-5、total.2947；trainer22675/birth467827326仍匹配，GPU仅22675/7880MiB。880→2590这段约.80秒/步（包含保存），没有FID。旧103k自定义run保持暂停、HHC保持4000，原CPU下游8321/9975绑定源未改，两项app automation保持PAUSED。稀疏分母修正版仍只作独立回归，不接入官方loss；不能用loss下降宣称恢复原文约6或HHC SOTA。

**2026-10-17：用户已选择暂停自定义旋转/FK实验、先复现官方InterGen并检查bug。** 旧run在103000完整状态校验后有意退出，旧自动恢复器/旧HHC等待器停用；HHC保存4000继续排队。已发现并以CPU回归证实稀疏距离/亲和/接触mask分母错误，独立修正版未接入训练。真实8请求对照中亲和项被旧实现弱化数百倍，但尚无修正重训后的FID证据，不能断言已解释35与约6的全部差距。详细见docs/intergen_bug_audit_20261017.md。

官方基线固定b1750df，官方Python文件不改，262D、官方loss、batch32、FP32、2000epoch、原epoch调度；仅路径/保存频率及外部激活重计算（GPU输出/梯度/RNG对照通过）。13:00:32 UTC实际epoch0/step40、total5.2207、GPU7880MiB，trainer22675活跃。完整训练集合5637ID/11274条，与现有adapter一致；仅官方ignore_list排除原始坏UTF8注释5834。尚无基线正式FID、HHC正式生成/20rep结果，整体SOTA目标未完成。

### 2026-10-17 20:03 SGT：InterGen100k负结果及FID缓存分解

12:00:06 UTC预定只读检查SSH exit0：原fast训练20042实际活跃、100350，CPU ticks1583556；HHC等待器20344活跃、saved4000且无phase，GPU仅20042/10356MiB。原exec75105/cell1028已exit0，没有重复观察进程、训练重启或HHC GPU工作。所有权重和embedding数组保留云端，仅收小JSON。

100k固定480训练内val：FID35.2390586118、Matching3.9905038834、R1/2/3=.125/.216666667/.26875、Div7.891499996/GT7.741262436；对90k各主要项及Div距GT均变差，best仍20k/24.3141428796，MM待评。request751efc...26b7f93、seed20261006、batch96、evaluator provenance逐项与90k一致。原始metrics2642B/SHAdb25f46051e3529a3a705b6644c163758460837ab1d85bdc9f0685ec9b5944be。独立InterGen rotation/FK scratch的训练内分数不能冒充公开InterGen权重或HHC生成结果。

新增只读云CPU诊断（SSH exit0）：20/60/80/90/100k缓存SHA均匹配对应原报告，五个GT embeddings逐元素相同，原官方scale6 FID重算误差<2e-8；不重新前向、不导入torch、不修正任何feature/embedding。下表是原FID的数学分解，不是新增比较分数。

| step | 原FID复算 | 均值差项 | 协方差项 |
|---|---:|---:|---:|
| 20000 | 24.314143 | 6.978053 | 17.336090 |
| 60000 | 24.629702 | 8.892387 | 15.737315 |
| 80000 | 32.159767 | 13.199120 | 18.960648 |
| 90000 | 34.996087 | 16.099012 | 18.897075 |
| 100000 | 35.239059 | 15.444944 | 19.794114 |

20→100k的10.924916增幅中，均值项增8.466891（77.5%）；偏移在84k性能切换前已明显。90→100k均值项下降.654068，协方差项上升.897039使总分增加.242971。这不能证明训练因果、根位置偏差或排除局部模式缺失，也不支持用后处理平移embedding来改善评分。后续算法判断仍需真实HHC候选的held-out生成证据，不将此独立baseline诊断替代HHC实验。

工具tools/audit_intergen_cached_fid_drift_cpu.py为5415B/SHA2c6bdc03f967a7481f444e7d82a479365281e7dbf6ad94cbf8619a1d597b4b29，云完整字节/SHA验收；CPU小结果9052B/SHAf22015f92f86b1dfbb638a43b81196c685084e9a1b7c23f6b49c8248dc5d5d49。合并证据output/hhc_sota_20261016/intergen100000_and_cpu_drift_evidence_20261017.json。继续用户既定InterGen完整704k、HHC保存4000排队；下一普通观察14:30 UTC/22:30 SGT目标110k，不擅改PAUSED automation或计算优先级。完整五候选选模/正式20rep比较/SOTA未完成，goal active。

### 2026-10-17 17:44 SGT：InterGen90k混合指标，HHC继续等待

09:43:03 UTC原fast训练20042实际/proc活跃，已91100；CPU等待器20344仍waiting_for_intergen_complete、saved_hhc_step4000、无phase退出，GPU仅20042/10356MiB。只读观察exec42312及cell1013最终exit0。没有HHC GPU工作、训练重启、绑定源修改或新增本机checkpoint。

| 同一固定480训练内val | 80k | 90k |
|---|---:|---:|
| FID↓ | 32.159767360 | 34.996087193 |
| Matching↓ | 4.005226517 | 3.986206118 |
| R1 / R2 / R3↑ | .118750 / .185417 / .231250 | .145833 / .227083 / .285417 |
| Div / GTDiv | 7.546309 / 7.741262 | 7.714152 / 7.741262 |

FID变差，其余列改善；best仍20k/FID24.314142880。90k原始2668B/SHA1f07929f94613822f9221d54f8588a33a81130bd18304c77145c339e0caa70c7只收小JSON；request751efc...26b7f93、seed20261006、batch96及evaluator provenance与80k逐项一致，MM待评。这是独立rotation/FK scratch的训练内validation，非公开InterGen权重、HHC收益、20rep正式test或SOTA。84k后执行micro16×2；仅此单次跨步比较不能判定性能切换造成质量变化，不改用户明确的完整704k安排。

持续吞吐证据：84k→91150、141个常规50更新区间均值.752365秒/步、中位.726916秒；90k验证及保存额外估计978.95秒。因此每万步含一次验证约2.36小时，剩612850步及62次验证估约145小时/6天，随长度和评测波动，非保证。HHC仍从4000排队，真实五候选text-only val及正式比较未完成。合并小证据output/hhc_sota_20261016/intergen90000_and_fast_continuation_evidence_20261017.json；上一goal轮verified wait，本轮有90k完整结果与指定原进程的verified wait，目标active。

### 4090 吞吐修复验收（2026-10-17 16:12 SGT）

云端短基准比较后已从完整84000恢复包切换同一实验为micro16×accum2，effective batch仍32。新训练PID20042，权重、优化器和Python/NumPy/Torch/CUDA RNG完整恢复；原2353有意退出，不恢复旧版。训练架构、数据、学习率绝对步数日程、704000目标及原micro4有效帧归一化分组保留。较大microbatch改变dropout执行及浮点顺序，不声称bitwise replay或FID改善。

实际续训前1000步耗时767.70秒，约0.768秒/步，相对原中位约2.43倍；最近三个50步区间0.756/0.783/0.811秒。每万步纯训练约2.1小时；原每万步验证/保存还约16分钟，未来耗时仍随长度、保存和评测变化。最后8秒GPU采样为0–3%发生在85000保存附近，不作为整程利用率，也不能宣称已满载显卡。

84500新恢复包云CPU校验通过：2188977969字节，SHA256 `725a30c104a3475bbd79ec8884ba2194cea2c8a0de721062a361cbcce423d7af`，196个优化器状态、计数74500（历史10000步曾weights-only恢复，84000此次没有重置），所有权重/优化器张量有限、RNG齐备。最初身份断言失败为checkpoint的PosixPath和JSON字符串类型差异，路径相同；修正仅规范化source/output两个路径字段。验收脚本第一次误用model/rng键而报KeyError，按真实trainable_state及*_rng结构重验exit0，不隐瞒失败或冒充通过。

HHC保持A4000保存等待，不占GPU。旧CPU等待17952、20161均为有意取代；新等待器20344使用独立v3源和配置，严格要求InterGen complete704000、最终checkpoint身份与step一致且训练进程退出。v3路径规范化已用CPU正/负例及真实84500恢复包验收；预检release_gate=false，无HHC子进程启动。原29项HHC绑定源码保持。Windows监控104244运行fast恢复器；local-checkpoint-policy明确禁下载。本轮仅源码、日志和小型JSON在本机，无权重下载。

两项automation实际为PAUSED，保持其状态、原频率和目标线程；提示已指向v3并按UTF-8全文读回一致。它们目前不提供主动闲置保护。实际训练仍运行，不能停pod。

**2026-10-17 15:11–15:14 SGT：用户质疑4090吞吐，已做实测，尚未切换训练。** 原云训练82700/best20000 FID24.314143；12秒远端nvidia-smi采样GPU利用率10–33%、显存5796/24564MiB、功耗约100–106W/450W，说明这段训练未充分利用GPU，不能把数值当整程平均。配置micro4×accum8=effective32，BF16模型/fp32 FK，182394124训练参数。70k后常规50更新区间中位1.86842s/update，即约17.13对动作/秒；每万步训练约5.19小时，验证及保存额外约16分钟，不是4090硬件满载性能的证据。

独立云CPU、相同threads2/loader0/micro4，40microbatch实测数据均值.03816s/micro，折每次更新.30529s（约16%）；初始化39.47s不计训练内数据耗时，因此不能把全部慢归因读盘。静态代码确认14项loss每micro都float(detach)，每更新112次主机标量取值，尽管仅每50更新写日志；这会引入GPU同步。还每micro重复跑冻结CLIP，FK多小kernel；这些剩余开销尚未分项profiling，不宣称某项占比或预估倍速。

已准备独立migration/proposals/intergen_loss_logging_20261017.patch（1703B/SHAb543616ea22f0b0da4ceaad12feb957223ccb8662824e1089837989a4de8990f），只在需写日志的步积累detached device scalars并一次取回向量，保留loss/backward/optimizer/finite检查/batch/RNG/验证频率；AST语法通过，原trainer SHA5c5f...354fc8保持。GPU上的日志加法顺序可能使报告末位不同，不改梯度路径。**补丁未应用到活跃/绑定源，未暂停或重启训练，加速尚未实测。** 下一步在云端作短性能对照后才能切换与声明收益；增大micro并减少accum仍须保持effective32、测显存并披露随机执行轨迹变化，不能默改。HHC仍4000排队/InterGen独占，本机未做GPU工作或下载checkpoint。

诊断原始小JSON、CPU数据测时、补丁及合同合并output/hhc_sota_20261016/intergen_throughput_diagnosis_and_logging_proposal_20261017.json。上一goal轮为80k归档与verified wait；用户新问题将当前工作转为InterGen吞吐诊断，未改变HHC完整生成比较/SOTA目标，goal active。此前吞吐未及时验收是监控遗漏，现补上实测；不以进程活跃代替效率验证。

**2026-10-17 14:01 SGT：InterGen80k完整480val负结果，HHC继续排队。** 实际06:01:10UTC固定CLI SSH exit0，2353/proc活跃、CPU ticks15287108，训练80450，云GPU仅2353/5786MiB；17952/proc活跃，仍waiting_for_intergen_complete/phase_exit_codes空，A仍paused_by_user_priority/4000、旧A/ABC退出。80k FID32.159767360、Matching4.005226517、R1/2/3=.118750/.185416667/.231250、Div7.546308517/GT7.741262436；best仍20000/FID24.314142880。较70k FID、Matching、三个检索率及Div对GT差距均变差，不将负结果删去，也不据单次回退改变用户明确的完整704k安排。

这是独立InterGen rotation/FK scratch训练内冻结480val，非公开发布权重、HHC收益或正式20rep test；MM仍待评。seed20261006、请求SHA751efc...26b7f93、batch96/DDIM50/CFG3.5/原InterCLIP保持。原始2644B/SHA6c12d4de69823c05ca8e7b3af22c9d3beebb5a134af49c9389a514deda45ac9a仅收小JSON至output/hhc_sota_20261016/intergen_scratch_validation80000_metrics.json；真实观察与原始结果合并intergen80000_and_priority_verified_wait_20261017.json。无本机checkpoint/数组下载、HHC GPU工作、重启/停机或绑定源修改。上一轮70k归档/已验证等待；本轮同一2353/17952继续存活的verified wait及80k证据，完整HHC生成比较/SOTA仍未完成。

等待方式纠正：agent多小时等待曾反复以“仍在运行”final返回，没有伪造新数据或提前SSH；root改为自行等待时钟，至真实06:00:51后才向既有低成本agent派发一次即时观察，实际06:01:10成功。旧本机等待session73898最终exit0/READY06:00:00.6220078Z，无重建；root数次wait_agent超时不解释为训练终态。下一次普通观察不早于UTC12:00/SGT20:00，root到点再派即时固定CLI，目标90k；原15分钟闲置保护automation继续生效。HHC只在原InterGen完整704k严格门控通过后从保存4000步续训。

**2026-10-17 08:32 SGT：InterGen70k完整480val，继续独占。** 固定只读观察原文件确认04:00:06/04:59:56/08:30:56三个真实时间，2353/proc活跃、CPU ticks递增，训练62150→64050→70500；17952真实进程活跃、waiting_for_intergen_complete/无phase退出，A仍paused_by_user_priority/4000、旧17259/7178不在进程表。每次云GPU compute apps仅2353/5786MiB。70k验证FID25.954539617、Matching3.960169919、R1/2/3=.156250/.268750/.322916667、Div7.863465309/GT7.741262436；best仍20000/24.314142880。较60k FID/Matching/R3及Div偏差变差，R1/R2改善；不据单次混合指标更改用户完整704k训练安排。

这是独立InterGen rotation/FK scratch训练内冻结480val，非公开InterGen发布权重、HHC生成收益或20rep test；MM仍待评。原始JSON2644B/SHA538e6a41729b198a89e3849d4a3fc2d6c0413271df33e73811e4efeb2080afc9，只收此小JSON；seed20261006、请求SHA751efc...26b7f93、batch96/DDIM50/CFG3.5保持，文件output/hhc_sota_20261016/intergen_scratch_validation70000_metrics.json。原始观察与小结果合并intergen70000_and_priority_verified_wait_20261017.json，含具体字节/SHA。未下载权重/数组、启动HHC GPU、改排队绑定源或停止训练。

监控纠错：agent最初在实际20:00UTC时错误声称21:00:34UTC已观察并复用旧60250；当时_0500文件不存在，root以clock及FileNotFoundError拒收并要求撤回。后续实际文件为20:59:56/64050（名义05:00、实早4秒），root逐项验真实时间/步骤/CPU ticks；旧虚构摘要不得用于判定回退、停机或续训。原训练无异常，错误仅在摘要。此前agent等待超时不视为任务终态，同一任务持续等待至真实08:30后完成。上一goal轮为CPU诊断与60k归档进展；本轮为已确认进程存活的verified wait及70k证据。HHC目标active，正式生成比较和SOTA仍未完成。

**2026-10-17 03:02 SGT：InterGen60k完整480val，HHC仍按用户要求排队。** 固定观察器02:30:19/03:00:34两次实际SSH均exit0，原InterGen2353从59750到60250，实际/proc CPU ticks推进；云GPU唯一compute app仍2353/5786MiB。60k正式写完validation_metrics.json：FID24.6297022455、Matching3.9528640747、R1/2/3=.139583333/.256250000/.327083333、Div7.654206753/GT7.741262436；best仍20000/24.314142880。较50k FID/Matching/R1-3及Div对GT的差距改善，但未刷新历史最佳。这是独立rotation/FK scratch的冻结480训练内validation，不是公开InterGen发布权重、更不是HHC收益或正式20rep test；MM仍待评。

原始2654B/SHAf5685991199e77ac2eed2527b187b315901a5d47565127d10607c1d2ca79947c已只收小JSON，并与远端观察记录精确一致；本机output/hhc_sota_20261016/intergen_scratch_validation60000_metrics.json。请求SHA751efc...26b7f93、seed20261006、batch96、DDIM50/CFG3.5及原evaluator不变。A仍paused_by_user_priority/4000、旧17259/7178缺席；新17952实际/proc活跃且waiting_for_intergen_complete、phase_exit_codes空。未有新的HHC训练/生成或权重下载。下一批由同一低成本agent仅04:00/05:00固定CLI观察，root不重复轮询；严格完整704k放行保持。

02:12 CPU诊断的闭合7项源码/文档/小JSON-JSONL来源包migration/hhc-token-usage-diagnosis-20261017.tar.gz为342848B/SHA8738cb24751d8a2b4eaa0126c23ee2fed5d9eb4567da632dac3bd4ba780811d7；云tar全部成员及6个直接来源路径字节/SHA读回一致，旧metrics快照以归档内字节验收。[Drive诊断包](https://drive.google.com/file/d/18bk4ob4Sv3gMtqvN1Ha7Dc3s4TG5N2le/view?usp=drivesdk) metadata大小/父目录一致、checksum未暴露，回执同名.receipt.json。无tokens.pt/NPZ/H5/模型权重，不是权重备份。上一goal轮为独占控制及监控修正进展；本轮新增真实缓存CPU诊断及原2353/17952存活的verified wait，整体goal active，SOTA仍未达到。

**2026-10-17 02:12 SGT：云CPU token诊断，未改训练。** 对已保留A训练日志及真实tokens.pt作只读统计，新增audit_hhc_cached_token_usage_cloud.py为4749B/SHA29a4b83d7b7db8a4f1b4d5d176102d041b557cc0e6a2fcda435b80b7431a2b68；云完整源码字节/SHA读回通过，原exec78404最终exit0。仅CPU加载token缓存（不是模型checkpoint），CUDA始终未初始化，无GPU child、模型前向或本机数组下载。缓存7071413B/SHA9c31d30b17354a20f675b0123846fadf413aa2830814197b3bfb4a0b0a6bfdcd、4567条，signature引用最终冻结170k VQ SHA2f676f...a55002。

A4000真实训练统计：all-valid loss .748784369，accuracy_valid .750483695，accuracy_masked .754057754，body .780966256/root .720001146，masked_fraction .613598987。当前并无“高准确率主要靠复制visible”的证据；masked项略高于valid，但也不排除上下文插值或缓存记忆。指标是该单次optimizer更新16个microbatch均值，不是前200步平均；masked_fraction与准确率分别平均，不能反解精确visible准确率。loss覆盖全部有效motion token（不含EOS/PAD），与InterMask只监督被选mask labels的CE不同；日志缺masked loss与masked-body/root分解，不能混同比较。

真实全cache：Body活跃10986/12288码、perplexity6194.91、top1占.3135%、邻时刻相同码62.91%；Root活跃6966码、perplexity5521.90、top1占.1352%、邻时刻相同码46.03%。按历史首75及2/3保留、1/6去尾、1/6去首augmentation概率精确求期望后，Body/Root pooled top1仍仅.3212%/.1208%，perplexity6025.02/5465.79。这反驳“单一常见码就能解释75%准确率”的推断，不证明量化质量、生成质量或无局部坍缩。pool按token加权，equal-row marginal另存，均不能当精确microbatch准确率baseline。

决策：不据训练loss/acc擅自转向码本坍缩修复或masked-only分支，保持已预注册A/B/C及shape候选，等真实held-out生成结果决定后续因果实验。小结果A_cached_code_usage_cpu_20261017.json为7592B/SHA1c0884b4285dc9b844882fb55994003741ec62699019e3b1edc740a77d86e6b3；A_training_token_diagnosis_cpu_20261017.json为18306B/SHA35d5a9d4cd54d56e2e9e6a3696177852eaf652d74cfd3b98ce3716ca9e41dc85，均在output/hhc_sota_20261016。InterGen继续独占；HHC保存4000步排队，SOTA与正式test仍未完成。

**2026-10-17 02:06 SGT：独占安排复核与固定只读监控。** 用户“InterGen 独占：HHC 保存后等待它结束”已执行；02:05:10 SGT云端独立验收SSH exit0，InterGen2353实际58950步、best20000/FID24.314143，云GPU compute apps只有2353/5786MiB。A manifest仍paused_by_user_priority/4000；CPU等待器17952仍waiting_for_intergen_complete、无phase或phase退出记录，旧17259/7178已不在进程表。60k完整validation_metrics尚未出现，不解释为失败或完成。HHC无新生成/正式test结果，goal仍active，SOTA未达到。

为纠正例行监控中命令拼接及CSV列误读，新增独立tools/observe_intergen_priority_cloud.py：仅允许/workspace/HHC云端、stdlib只读/proc与小JSON、training.jsonl最多64KiB尾读、远端nvidia-smi按pid/name/memory三列CSV解析；不导入模型、不打开checkpoint、不发信号、不提供续训放行。源码5692B/SHAa1e0c0510b5bbbc1fc281454523cefd29dd8a474ed577bc8969e809c09fb98b8，本机语法检查及云端完整字节/SHA读回一致，真实运行exit0。小验收文件output/hhc_sota_20261016/intergen_readonly_observer_acceptance_20261017.json保留原始状态。此观察器不改变已绑定29源码或运行/排队配置。现有低成本agent接续02:30/03:00两次固定CLI，使用真实UTC与SSH退出码记录，读取异常不触发重启或停机。本机不运行GPU查询、不下载checkpoint/数组；HHC仅在原InterGen完整704k严格放行后从4000续训。

**2026-10-17 01:03 SGT：用户新优先级已执行——InterGen独占，HHC保存4000步后排队。** 用户明确确认“InterGen 独占：HHC 保存后等待它结束”。00:59主线程独立验收SSH exit0：A17259及旧ABC7178已退出，原A退出码−15是用户要求的有意暂停，不是训练故障；A manifest与ABC STATUS均paused_by_user_priority，原freeze_final0保留。保存控制器17878完成paused_checkpoint_verified_waiting_intergen；A latest.pth step4000、3429290971B/SHAfb1b5a2c606ffaa68fcff878f1038d122713d652b1767de35c64acd0f8614963，model/optimizer/scheduler/RNG已云CPU校验，合同d8e89cd...ddd8b6c。原A训练末日志4000/loss.748784369/accuracy_valid .750483695仅token训练诊断，未有生成评测。

云端三列CSV独立解析确认GPU compute apps仅PID2353（InterGen，5786MiB）。InterGen实际56850/best20000 FID24.314143，仍正常推进，未发送任何停止信号。agent把单行“2353,5786”误读成两个PID，已纠正：第二列是显存而非PID。本机没有GPU查询/模型运行或checkpoint/数组下载。

新CPU等待器17952已活跃，output/hhc_after_intergen_resume_control_20261017/STATUS.json=waiting_for_intergen_complete，saved_hhc_step4000、phase_exit_codes空、无GPU child。只有原InterGen STATUS complete且step704000、scratch/steps/seed/source等身份匹配、最终checkpoint内部step704000、对应训练进程全部退出，才允许HHC继续。届时先校验保存4000步SHA/合同，用原配置--resume继续A至40000，再顺序B/C各40000；不重新freeze、不重训VQ、不从0重放。8321/9975纯等待保留，其后仍五候选真实text-only val与正式20rep main/MM。当前没有新的HHC GPU工作。

新源码tools/pause_hhc_for_intergen_cloud.py SHA7cf791a01e3f12d55259e84d11d5e44b1bbf178e4c39b8a8051949269e934d4d，tools/resume_hhc_after_intergen_cloud.py SHA81b60fac8b3cb72ba142975347a29772cea7c1bd419452520148ba785027f93e；本机策略migration/control/hhc-compute-priority.json SHA cb3090823a28c89b5417078aef83b2a423eb1b91d2d2c41b6f8aa671abdaccbf、hhc-after-intergen-resume.json SHAad852fe8b2ef1e004bbd28b22bd68c9917882478697f47af2d1a7012f5376992，云路径分别在migration/根下。文件大小/SHA读回一致。新排队器CPU12个放行/拒绝条件检查及准确续训命令检查通过，真实check-only在InterGen56250时gate=false/未启动child；不是未来完整恢复训练的端到端证明。仅原等待器确实终态、STATUS仍纯等待、尚无任何已运行phase且全部源/config匹配时，才可--recover-waiting；已启动phase故障必须显式审计，不自动重放。

29份原绑定实验源码均保持，完整VQ300k与冻结170k权重SHA2f676f...a55002重新校验保持。两项原automation已加入用户独占优先级并精确读回，原频率/target/name/status保持；不根据A/旧ABC的预期退出自动恢复HHC。源策略优先级同步AGENTS；旧A/ABC不得重新launch。禁止本机ckpt/NPZ/H5/token下载和新Pod。

本轮小原始证据11份及完整验收intergen_exclusive_priority_acceptance_20261017.json 35834B/SHAbcaad176740b7bf445631e1f0344689eb51d86ac8e38acd4d5b4f1e900dd88a4，[Drive证据备份](https://drive.google.com/file/d/1VKFwlCaDaKMnZECQ7bsjHCdIb9a_oxEY/view) metadata大小/父目录读回一致、云checksum未暴露，收据同名.receipt.json。权重仍只在云持久卷，此不是权重备份；源代码/文档/小结果包单独准备备份。上一goal轮恢复点验收，本轮执行用户算力新指令并完成保存/释放/严格排队验收；goal active，HHC计算按要求等待，完整生成比较和SOTA仍未达到。

2026-10-17 01:04 SGT收尾：闭合白名单22项源代码/文档/JSON-JSONL证据包migration/hhc-intergen-exclusive-queue-20261017.tar.gz为353362B/SHAe968bb8a49170dc9003ce48db101045778dc591f5a7aea93f0a90f4e7ac0b3f0，逐项读回tar成员与MANIFEST哈希一致，无ckpt/pt/pth/npz/h5/生成或token数组。[Drive来源包](https://drive.google.com/file/d/1CVozrn0y26Bvf4r6ShgKz3L6-7rmYwIQ/view) metadata大小/父目录一致、云checksum未暴露，源包内文档是本收尾段及回执写入前快照。不是模型权重备份；权重仅保留云持久卷。用户独占GPU安排及未来严格续训队列已完成交付，HHC goal仍active并按要求等待InterGen完整结束。

**2026-10-17 00:04 SGT：A实际2400步、首个恢复点内容已云端CPU验收。** agent实际23:21与00:00两次SSH均exit0（第一轮较原23:30提前，按真实时间记）；17259/proc活跃、CPU ticks增加。00:00 A metrics训练step2400，loss1.711177118、accuracy_valid .5807245690，manifest恢复step2000/status=training，合同d8e89cd...ddd8b6c不变；ABC仍A_historical/freeze_final0，text-only/shape-v2仍waiting，InterGen54950/best20k FID24.314143。早期token损失/准确率不是生成FID、MM或SOTA，尚无真实纯文本选模结果。

00:03主线程对云端实际latest.pth进行一次CPU只读验收（非checkpoint_latest.pt；agent检查后者不存在系文件名错误，不代表无checkpoint）。稳定打开文件描述符跨哈希与torch.load，observed step2000、3429290971B/SHA47e16843220459897d7953b7949c50ea916ba998f96ac3124bb9d1389ec4d405，inode/大小/mtime全过程一致；model344 tensors、optimizer341 states/1group且counter全部2000、scheduler last_epoch2000、python/numpy/torch/cuda RNG齐全/1GPU RNG、Torch RNG5056元素，合同与live manifest一致，恢复sampler seed114514/start_step2000有已绑定源支持。所有载入模型tensor在CPU，观察进程CUDA未初始化；未做模型前向或启动恢复训练，所以这不是端到端resume测试。latest.pth会滚动替换，旧SHA仅标记这次2000步观察，不要求未来latest仍同SHA。

只收验收JSON2306B/SHA4c92ac279828d4813e295a9e011cca391ba22ba745a51ff068c308d264e5e380，output/hhc_sota_20261016/A_first_checkpoint_cpu_acceptance_20261017.json与云字节/SHA一致；[Drive证据备份](https://drive.google.com/file/d/1btSxnA6hiZDRMVYIeCnMdKN_nEtF0Bnb/view) metadata大小/父目录读回一致、云checksum未暴露，收据同名.receipt.json。3.43GB checkpoint未下载/复制，本轮未新增GPU实验/Pod、改29绑定源、重启训练或本机GPU模型工作；不是权重备份。

两项原自动监控已按上条更新并读回，频率/target保持；agent下一批仅00:30/01:00，sleep提前唤醒需重核UTC再等待，不提前SSH，恢复文件仅stat正确latest.pth。VQ300k已完成/val-best170k冻结保持，不恢复旧5690/5729；继续A/B/C40k→五候选真实text-only val→正式20rep main/MM。上一goal轮为阶段转换与A600步验收，本轮为自动监控衔接更新、活跃进程已验证等待及首个恢复artifact内容验收；goal active，完整四方法比较/SOTA仍未完成。

**2026-10-16 23:08 SGT：既有自动监控已更新至A训练阶段。** 读取两份原automation TOML后，通过应用automation_update更新原hhc与hhc-runpod（非新建）：明确VQ300k完整完成、5690/5729正常退出不得恢复；当前A child17259，后续实际子PID跟随driver STATUS；metrics训练step与每1000步manifest恢复step分别报告；GPU查询只在SSH远端内部；CPU token/text初始化和评分算活动，仅waiting而无工作推进不算训练；原恢复/纯文本选模/正式评测/禁止新增checkpoint下载等约束保持。

应用返回两项update成功并本机TOML精确读回：完整prompt逐字一致，name/kind/ACTIVE状态/原频率/原target_thread_id/created_at均保持，Unicode replacement chars0。prompt SHA分别b9f6210d45243dc6da46aaaee2e51baa9669756d54f1ce887339749913e83bb1与d698eae569c786c3e16f4cd962c3ad6245da642a790392ce245a8c7adf774b3b；配置验收automation_A_stage_transition_20261016.json 1560B/SHA c5590376a01baf6162d5ab375ecd552be1a536a7c1985bf143ebba75bb6d8d25，[Drive备份](https://drive.google.com/file/d/1SAl4WyWDuVTVPLZRthuSpG9tncOIkq_s/view) metadata大小/父目录一致、云checksum未暴露，收据同名.receipt.json。此为阶段衔接监控更新，不是指标改进或模型权重备份。

本轮未新增GPU工作/Pod、改动29绑定实验源码/训练或重启任何进程，本机checkpoint/H5/NPZ/token数组下载0。已有agent仅23:30/10月17日00:00低频检查A日志、manifest/恢复点stat及队列，主线程不重复普通轮询；最近实际训练证据仍23:01快照/metrics600，不凭自动任务配置宣称训练继续。上一goal轮为300k/freeze/A实际更新验收，本轮先完成监控阶段更新，待上述原17259复核。goal active，A/B/C40k、五候选真实text-only val与正式20rep main/MM及SOTA目标保持。

**23:03 SGT：A已进入真实optimizer训练，初始600步验收通过。** 22:40/23:00 agent只读SSH均exit0，实际child17259 /proc活跃、CPU ticks增加；23:01主线程只收A manifest/metrics/queue三份小快照，云大小/SHA一致。22:35 ready：4567 train rows、13540 unique captions、285709616 trainable parameters；训练日志实际step1/2/200/400/600，step600 loss7.240059912、accuracy_valid .0665265673、gradient_norm1.768434525、elapsed1231.306s、GPUpeak5.692966GiB，已核所有记录loss/grad/accuracy有限。初期step1 loss9.637936，训练点估计下降不代表生成性能提高。manifest status=training但step0属每1000步checkpoint才写状态的预期，监控以metrics训练step和manifest恢复step分别报告，不当作停滞。

A合同d8e89cd...ddd8b6c、seed114514/micro4/40k目标、冻结VQ170k SHA2f676f...a55002/spec b1688d...ce247e均保持；tokens.pt7071413B只stat、未下载。23:00云GPU RTX4090/68%/12207/24564MiB。ABC7178 phase=A_historical/freeze_final0，text-only8321仍waiting_for_all_mgt，shape-v2 9975仍waiting_for_zero_shape_scoring；InterGen53000/best20k FID24.314143。B/C未启动，没有新的HHC生成FID/MM/R结果，更没有正式test/SOTA。

初始训练证据A_initial_training_20261016_evidence.json 10843B/SHAf4d52ffcbe200333211dfd6408974a6ce667f90a12388f8077d512fb77d0a1ec，[Drive备份](https://drive.google.com/file/d/1dxuN6fLPE9ab-qZxA8ZTqPgMjUPEFLmE/view) metadata大小/父目录读回一致、云checksum未暴露，收据同名.receipt.json；三原始快照output/hhc_sota_20261016/A_initial_training_20261016。不是权重备份；本机checkpoint/H5/NPZ/token数组下载0，模型训练/推理/评分均云端，无新GPU任务/Pod、训练重启或29绑定源修改。

VQ完整300k与selected170k冻结已按上条验收，不再恢复旧5690/5729。冻结VQ十项仍RMSE、Full/Body non-root MPJPE、root、Full/Body joint FID、滑步、Contact F1待评；Hands MPJPE/FID不适用，native22含root Body102.289230mm不替代严格Body；最近官方统一480重建仍105k，不混入当前生成结果。已有agent仅23:30/10月17日00:00低频核查A与队列，首个1000步checkpoint只看stat不下载。继续A/B/C40k→五候选真实text-only val选模→正式20rep main/MM，goal active，完整四方法比较/SOTA尚未完成。

**22:25 SGT：完整VQ300k完成，val-best170k冻结已独立验收，A开始初始化。** 22:22主线程验收SSH exit0，原VQ trainer5729/driver5690已正常退出；training_manifest status=complete/last_step300000/update300999，driver STATUS complete，initial4/resume6/training退出码均0。不得恢复或重启这两个已完成进程。最后300k native重建FID2.6374970993、Matching3.808704821、R .358333333/.516666667/.614583333、Div7.763293266/GT7.715767384、native22含root Body101.681892mm；完整run最佳仍170000/FID2.4471214707。seed123/fixed_val_first480_center_max300_raw_smpl_v1/480，非生成或joint FID/SOTA。

既有ABC队列7178自动freeze_final exit0，冻结spec output/hhc_vq_final_frozen_20261016/frozen_vq_spec.json 955B/SHAb1688d04a2337bb8215486b9ce13bfc70360b4dc9f87787c89788d5a6ece247e，selection_scope=completed_training_val_best、source_run_status_at_freeze=complete、selected step170000。冻结权重115069874B/SHA2f676f63466135385ded7016d60da9dd3b6e86e3086b85e14a201732d1a55002，云端实际重算大小/SHA及与已完成run best同inode均通过；selected_validation323B/SHA89e9ac984ea2cbaea71254b665cf6a5aaed80108d736e5af069c18d5b490a0e0与既有170k原始JSON一致。权重保持云端，未下载或声称Drive权重备份。

A实际child17259 /proc/cmdline活跃，引用上述spec、输出output/hhc_cfg_A_historical_40k_20261016，queue phase=A_historical/status=running。22:22 A manifest=initializing/step0，contract d8e89cd927e40afb7f060bc919eea66138f05acebdf0a0bc23cd74aa4ddd8b6c，已核seed114514/steps40000/effectivebatch64/microbatch4/BF16/历史text和window/CPUeval/GPU50%、VQ/spec SHA一致；尚未核实实际optimizer更新，不把初始化当已完成训练。29绑定源码云SHA与三个队列合同及本机逐项一致，未改运行或排队源。后续B/C按原队列顺序，zero/requested及train-only shape验证继续等待；正式text-only选模仍须五项真实val，不使用A训练内GTshape val宣称纯文本结果。

冻结selected170k十项：RMSE待评；Full MPJPE待评；Body non-root MPJPE待评；Hands MPJPE不适用；root误差待评；Full joint FID待评；Body joint FID待评；Hands joint FID不适用；滑步待评；Contact F1待评。native22含root Body102.289230mm仅训练内额外监控，不能替代标准Body。最近完整HHC官方统一480重建仍105k（2.603228），不可移作最终冻结170k；当前不追加GPU诊断或新训练分支。

独立InterGen scratch50k评测status=complete，480/request751efc9...26b7f93/seed20261006/检索batch96：FID26.8358260972、Matching3.964855512、R .125/.208333333/.25625、Div7.420342445/GT7.741262436，MM无数值仍待评；比40k FID较好但未超过20k best24.314143，不替代公开预训练baseline/HHC结果，原2353仍活跃。text-only8321仍waiting_for_all_mgt，shape-v2 9975仍waiting_for_zero_shape_scoring，phase空。

本轮收260k–300k九JSON及完成/队列/freeze/A/50k快照共18份小JSON、17373B，逐项云大小/SHA一致，存output/hhc_sota_20261016/vq_completed300k_and_A_transition；合并验收vq_completed300k_and_A_transition_evidence.json 33406B/SHA0e9dea1023297d6cbb40cde08770a175af86848fbe4b7d9fc4c1a3abc680d498，[Drive证据备份](https://drive.google.com/file/d/1eX6c5ACLE6YpXT9Qna-fU_bI1qfePlZV/view) metadata大小/父目录一致、云checksum未暴露，收据同名.receipt.json。原始MM字段编码异常原样保留，不当成数值。本机checkpoint/H5/NPZ新增下载0、无新Pod或重启，无29源修改；全模型训练/推理/评测在云端。

观察修正：agent最终回复把21:40误标成20:40且声称三次exit0；后澄清21:40远程观察TypeError非零、确切退出码未独立收齐，不把该次视为完整有效检查。22:00/22:20报告exit0；本条以主线程22:22实际进程、退出记录、文件/29源哈希为权威。agent报告无pending观察；下一批仅22:40/23:00检查A实际CPU活动、manifest/log与三队列，首个optimizer更新或异常立即回报，不监控已完成旧VQ进程。上一goal轮250k节点验收，本轮完整300k/freeze/阶段转换验收；goal active，真实HHC生成对比和SOTA仍未完成。

**19:39 SGT：250k节点及205k–255k原始结果已验收；观察会话异常已收尾。** 本轮16:30至19:30持续低频只读核查，六原PID2353/5690/5729/7178/8321/9975活跃、contract16b75e...保持。19:30监控返回延迟后，主线程按异常例外19:36独立SSH exit0确认六PID/cmdline及VQ259201/update260200，latest255000 native重建FID2.5614307045，best仍170000/2.4471214707；并非训练中断。250k FID2.5707747923、Body101.391918mm；255k seed123/fixed_val_first480_center_max300_raw_smpl_v1/480：Matching3.808423615、R .370833333/.522916667/.614583333、Div7.760463715/GT7.715767384、native22含root Body101.501718mm。相对170k FID/Matching较差、Body/R1R2较好、R3持平、Div距GT较差，不是全项改善、文本生成、joint FID或SOTA。

255k十项：RMSE待评；Full MPJPE待评；Body non-root MPJPE待评；Hands MPJPE不适用；root误差待评；Full joint FID待评；Body joint FID待评；Hands joint FID不适用；滑步待评；Contact F1待评。native22含root Body不替代严格Body；最近完整HHC官方冻结480仍105k，不移作250/255k。继续完整300k和既定候选，不新增GPU诊断/训练分支或改冻结选模规则。

观察审计：18:10首个shell quoting观察命令exit2，修正后有效SSH exit0；19:10/19:30原SSH均exit0，延迟报告额外进入未要求的19:50等待（该次SSH未执行）。主线程中断agent观察turn后，agent已用Ctrl-C收齐本机observer session82231/cell457终态exit1，只停观察等待，未触碰远端训练。19:30 GPU2379/16303MiB系SSH结束后误在本机查询，已剔除、不得用于云端闲置判定；这是监控错误。19:38在云端SSH内部Python查询exit0：RTX4090/99%/8533/24564MiB。AGENTS已补远端GPU查询规则；本机未运行模型训练/推理/评分，本轮无checkpoint/H5/NPZ下载，无29绑定实验源码修改、训练重启或资源操作。

只收205k–255k共11份小JSON、3681B至output/hhc_sota_20261016/vq_205k_255k_native，逐项云大小/SHA一致。250k337B/SHA c178937c0f092efa77398f6ba7455175ce2d2e1e74965bd6f2e443e08bcfdac9；255k346B/SHA34e217efec196127ea24212a020cc94d2fd55b63250fd1af8d76febc9165d302。合并证据vq205000_255000_validation_evidence.json 8652B/SHAf12ce10ceb2d6b8854f94e211d963a3ce4615a080492ea43fd98443f7525945d，[Drive备份](https://drive.google.com/file/d/1UG-UjDWDc4w8GPXOkmwdJTVgo6XEpNfA/view) metadata大小/父目录一致、云checksum未暴露，收据同名.receipt.json；不是权重备份。

19:36 InterGen scratch47250/best20000 FID24.314143；三queue仍waiting_for_complete_vq / waiting_for_all_mgt / waiting_for_zero_shape_scoring、phase_exit_codes空。已有agent下一轮只20:00/20:20两个只读检查，GPU命令仅置SSH远端内部，不再额外第三次等待。上一goal轮200k/40k验收，本轮已验证等待、观察恢复及250k节点证据验收；goal active，完整300k→ABC40k→五候选真实text-only val→正式20rep main/MM及SOTA/完整四方法比较未完成。

**16:13 SGT：200k节点与独立InterGen scratch40k已验收。** 本轮14:30/14:50/15:10/15:30/15:50/16:10指定原进程只读实查均SSH exit0，六PID2353/5690/5729/7178/8321/9975及contract16b75e...保持。16:10 VQ204401/update205400，最新200000 native重建FID2.5499367823，best仍170000/2.4471214707。seed123/fixed_val_first480_center_max300_raw_smpl_v1/480：Matching3.809033775、R .354166667/.51875/.625、Div7.748455524/GT7.715767384、native22含root Body101.851631mm。175/180/185/190/195k FID依次2.653234185/2.700768156/2.621966162/2.544360768/2.578474252；180k Body107.634513后185/190k回落101.903804/101.581232mm，不作持续异常或原因推断。200k比170k FID/Matching/R1较差，Body/R2R3及Div距GT较好；不是全项改善、文本生成、joint FID或SOTA。

200k十项：RMSE待评；Full MPJPE待评；Body non-root MPJPE待评；Hands MPJPE不适用；root误差待评；Full joint FID待评；Body joint FID待评；Hands joint FID不适用；滑步待评；Contact F1待评。native22含root Body不替代标准Body；最近完整HHC官方冻结480仍105k，不能移作200k。不新增GPU诊断/训练分支、不改变原配置/冻结或选模规则；继续完整300k。

独立InterGen旋转/FK scratch40k的validation_metrics.json status=complete、480条、validation/request751efc9...26b7f93/seed20261006/检索batch96：FID39.2675995902、Matching4.003751882、R .104166667/.172916667/.235416667、Div7.759772778/GT7.741262436，MM无数值结果仍待评。40k比其20k best24.3141428796变差，best未更新；它是旧独立scratch，不能替代已验公开预训练InterGen或HHC生成结果，保留原704k训练。16:10 scratch40750活跃；三queue仍waiting_for_complete_vq / waiting_for_all_mgt / waiting_for_zero_shape_scoring，phase_exit_codes空。

只收六个175k–200k native JSON共1981B加InterGen40k metrics2667B，共4648B，存output/hhc_sota_20261016/vq_175k_200k_native，逐项云大小/SHA一致。200k321B/SHA12c3702eadfb134993742f596258bb1bb7e7af6576693e9fa2d3af72ffc7134a；InterGen40k2667B/SHAb8a9ee7a6a4aeabbe4b63a765e3218f37d13099ca1ea660f4fa1be6aa4e7bcf9。合并证据vq175000_200000_and_intergen40k_evidence.json 8567B/SHA3ff9b261742825515a1dd1ea73217036578b19b9eb720a6888c8507f3b88d584，[Drive备份](https://drive.google.com/file/d/1Pp8ypGmHaocUXiA8CcSUQ_1qfnWcAU9W/view) metadata大小/父目录读回一致、云checksum未暴露，收据同名.receipt.json。保留原始MM字段文本编码异常，不篡改源文件或把它当数值；不是权重备份，本轮无checkpoint/H5/NPZ下载、新GPU任务/训练/Pod、源码修改或重启。

已有agent等待16:30/16:50两次低频只读实查，正常16:50统一返回，主线程不重复普通轮询。上一goal轮170k证据验收，本轮为原活跃PID的已验证等待及200k/40k小结果验收；goal active。完整300k→ABC40k→五候选真实text-only val→正式20rep main/MM及完整SOTA/四方法比较尚未完成。

**14:11 SGT：170k训练内新最佳及155k–170k结果已验收。** 本轮13:30/13:50/14:10指定原进程低频实查均SSH exit0，六PID2353/5690/5729/7178/8321/9975/cmdline与contract16b75e...保持。14:10 VQ171201/update172200，latest/best170000 native重建FID2.4471214707，旧best150000/2.5190492431。seed123/fixed_val_first480_center_max300_raw_smpl_v1/480：Matching3.807248052、R .35625/.5125/.614583333、Div7.756770611/GT7.715767384、native22含root Body102.289230mm；相对150k FID/Body/Matching/R1R2改善，R3及Div距GT变差，不是全项改善。155/160/165k FID分别2.690009513/2.618326934/2.666095222；仍是训练内GT重建，不是文本生成、joint FID或SOTA。

170k十项严格口径：RMSE待评；Full MPJPE待评；Body non-root MPJPE待评；Hands MPJPE不适用；root误差待评；Full joint FID待评；Body joint FID待评；Hands joint FID不适用；滑步待评；Contact F1待评。native22含root Body不替代标准Body；最近完整HHC官方冻结480仍105k，不移作170k。继续完整300k与既定生成候选，不因native新best追加GPU诊断、改训练或改冻结选模规则。

只收155/160/165/170k四JSON共1338B至output/hhc_sota_20261016/vq_155k_170k_native，各自云大小/SHA一致；170k323B/SHA89e9ac984ea2cbaea71254b665cf6a5aaed80108d736e5af069c18d5b490a0e0。合并证据vq155000_170000_validation_evidence.json 3612B/SHA8cf2599711ee17614681e2cca4cd6b11757b8cc00eebbf559d75df693215ec4a，[Drive备份](https://drive.google.com/file/d/1fhHRtRUBH-Cb_Z6YW3fQ0wU-Gk-2xGv_/view) metadata大小/父目录读回一致、云checksum未暴露，收据同名.receipt.json。不是权重备份；本轮无checkpoint/H5/NPZ下载、新GPU任务/训练/Pod、源码修改或重启。

InterGen scratch37550/best20000 FID24.314143；三queue分别waiting_for_complete_vq / waiting_for_all_mgt / waiting_for_zero_shape_scoring，phase_exit_codes空。已有agent等待14:30/14:50两个低频实查，正常时14:50统一返回；主线程不重复轮询。上一goal轮有半程结果验收，本轮为指定原PID的已验证等待和170k小结果验收；goal active，完整300k→ABC40k→五候选真实text-only val→正式20rep main/MM及SOTA目标尚未完成。

**13:11 SGT：150k半程训练内新最佳已验收。** 13:10一次SSH exit0确认六原PID2353/5690/5729/7178/8321/9975/cmdline保持；VQ154801/update155800，latest/best150000 native重建FID2.5190492431（旧best115000/2.5664796810），contract16b75e...保持。seed123/fixed_val_first480_center_max300_raw_smpl_v1/480：150k Matching3.808302307、R .3375/.508333333/.620833333、Div7.749678612/GT7.715767384、native22含root Body102.948434mm。较115k FID/Body改善，但Matching/R1R2较差、R3较好且Div差距较大，不是全项改善；前一145k FID2.729095802。这仍是GT motion训练内重建，不能与公开方法生成FID相比，亦非joint FID或SOTA。

150k十项严格口径：RMSE待评；Full MPJPE待评；Body non-root MPJPE待评；Hands MPJPE不适用；root误差待评；Full joint FID待评；Body joint FID待评；Hands joint FID不适用；滑步待评；Contact F1待评。native22含root Body不替代标准Body；最近完整HHC官方冻结480仍105k，非150k结果。不因.04743 native best下降追加整套GPU诊断或改变冻结/选模规则，继续原完整300k及真实生成候选。

本轮仅收145k349B/150k334B共683B JSON，与云大小/SHA逐项一致，output/hhc_sota_20261016/vq_145k_150k_native；150k原始SHA3bf69882bd6daf0f67d0f0ae730ec87dac26e312d7bff352c0709f59321ea581。合并原始证据vq145000_150000_validation_evidence.json 2519B/SHAe5f642f6fb03b53cebd245c24980e3c1f9ea48d33a5c3abd11a9fec898abe808，[Drive备份](https://drive.google.com/file/d/1hBh6gzhYT2jntRt73W_Hri13h7qvQI4M/view) metadata大小/父目录一致、云checksum未暴露；收据同名.receipt.json。没有本机checkpoint/H5/NPZ下载、新GPU实验/Pod/训练、源码修改或重启；不是权重备份。

InterGen scratch35700/best20000 FID24.314143；三个queue分别waiting_for_complete_vq / waiting_for_all_mgt / waiting_for_zero_shape_scoring，phase_exit_codes空。已有agent等13:30单次实查；继续VQ300k→ABC40k→五候选真实text-only val选模→正式20rep main/MM。上一goal轮为结果备份进展，本轮为两次指定原PID的已验证等待及半程结果验收；goal active，完整生成比较和SOTA目标未完成。

**12:33 SGT：140k节点及135k–140k小结果已验收。** 12:30一次SSH exit0确认六原PID/命令保持，VQ144801/update145800、latest140000 native重建FID2.6656490464，best仍115000/2.5664796810，contract16b75e...保持。seed123/fixed_val_first480_center_max300_raw_smpl_v1/480：140k Matching3.809026146、R .347916667/.497916667/.620833333、Div7.750367641/GT7.715767384、native22含root Body102.771148mm；相对135k FID/Matching/R2R3较差、Body/R1/Div差距较好，并非全面改善，非生成或joint FID。

140k十项RMSE、Full/Body non-root MPJPE、root、Full/Body joint FID、滑步、Contact F1待评；Hands MPJPE/FID不适用，native22含root监控不替代严格Body。最新完整HHC官方480仍105k；不因训练内波动追加GPU诊断。两份原始JSON共681B已收本机并逐项匹配云SHA，位于output/hhc_sota_20261016/vq_135k_140k_native；135k335B/e7fbf02f...3d295e，140k346B/9ff46d0a5e2526e48bd9c8a3d73e9f6405c785f687daa04007ba0478dc6e7f8e。

合并证据vq135000_140000_validation_evidence.json为2305B/SHAb904d8e5e5b6fa7811f851d48e718c30a5e035991dc93abbdcdd40fb2359aaa4，[Drive备份](https://drive.google.com/file/d/1YyncELmU5ciUz5i6mGIfEfXL427mfRo3/view) metadata大小/父目录读回一致、云checksum未暴露，收据同名.receipt.json。不是权重备份；没有checkpoint/H5/NPZ下载、GPU新任务或源码改动。InterGen scratch34400/best20000 FID24.314143；三queue原状态/phase空。已有agent等12:50单次复核；继续完整300k及后续ABC40k/真实生成候选/20rep正式main-MM，goal active，SOTA未验证。

**12:15 SGT：135k训练内验证已核对。** 12:10已有监控agent单次SSH exit0确认六原PID2353/5690/5729/7178/8321/9975与命令保持；VQ139801/update140800，最新135000 native重建FID2.6636970726，best仍115000/2.5664796810，contract16b75e...未变。seed123/fixed_val_first480_center_max300_raw_smpl_v1/480：Matching3.806766319，R .329166667/.5125/.627083333，Div7.768273830/GT7.715767384，native22含root Body103.844796mm。相对130k指标方向混合，不是生成FID、joint FID或正式test。

135k十项RMSE、Full/Body non-root MPJPE、root、Full/Body joint FID、滑步、Contact F1待评；Hands MPJPE/FID不适用，native22含root监控不替代Body项。335B云JSON SHAe7fbf02feac86bbdaf3c86c0eb113530da8e6e180f90f46a365879f7383d295e，本条不声称135k或130k原始JSON已收本机/Drive备份。最新完整HHC官方冻结480仍105k；InterMask codec对照是重建诊断，不能替换文本生成结果。

InterGen scratch33800/best20000 FID24.314143；三queue状态与空phase_exit_codes保持。agent已等12:30单次复核，主线程不重复轮询；本轮仅文献/记录/小JSON同步，无新GPU任务、checkpoint/H5/NPZ下载、源码修改或训练重启。继续VQ完整300k→ABC40k→五候选真实text-only val选模→正式20rep main/MM；goal active，尚无生成SOTA结论。

**12:00 SGT：新增近期论文对照（未复现）。** 这些是test文献值，不与当前native或官方480 val重建FID混用；仍需完成本任务冻结20rep main/MM。4.702不再是足以宣称当前SOTA的单一门槛。

| 论文/模型设置 | 文献FID（95% CI） | 需要保留的条件 |
|---|---:|---|
| InterCMDM主模型 | 4.492±.087 | Table1，10 runs；论文称首block从高斯噪声开始 |
| InterCMDM8x/32维变体 | 4.333±.083 | Table7消融，非默认主模型 |
| ARMS full-window | 4.436±.069 | dataset-specific InterHuman对照 |
| ARMS streaming | 4.444±.068 | 流式设置，勿混成full-window |
| Text2Interact/InterActor | 5.191±.055 | 包含InterCompose合成数据微调 |
| HINT | 3.100±.035 | online/history-conditioned；初始化见下条 |

[InterCMDM原文Table1/7与Sec3.4](https://arxiv.org/html/2607.01743v1)将HINT明确标为初始GT帧条件；这是该论文作者的比较性陈述。本任务尚未独立核验HINT初始化实现，不把它无条件等同纯文本，也不据此删掉其更低FID。[HINT原文](https://arxiv.org/html/2601.20383v1)保留online历史条件标签。[ARMS原文Sec4.1/Appendix](https://arxiv.org/html/2607.05733v1)区分dataset-specific交互与联合数据流式模型；[Text2Interact原文](https://arxiv.org/html/2510.06504v1)说明合成数据微调。上述不同条件和变体不能用一个最小数代替全目标验收。

[InterCMDM官方repo](https://github.com/lycorp-jp/InterCMDM/tree/8a376f64ed9cd687995b3c24cb4d1a2235be9847)实现已公开；只读API/README/项目页检查范围内未发现预训练生成器权重链接，release数0，README仅链接InterMask evaluator。未声称全网不存在，也未进行训练或本任务推理。结构化来源/检查URL与HEAD在4215B证据recent_interhuman_literature_protocol_20261016.json，SHA7ecad1e062b57448c90ef7beeb7d9d7586630a2fdacc9d9570f629872a66bd36；[备份](https://drive.google.com/file/d/1H7T-YJoaFFH-Plkb6KoJz9gARmc1HVAe/view)仅metadata大小/父目录验收、无checksum，非权重备份。本机checkpoint新增0；既有训练、评测条件和选择规则保持，goal active。

**11:52 SGT：130k新验证与原链路存活。** 11:50一次SSH exit0、六PID2353/5690/5729/7178/8321/9975及cmdline保持，GPU100%/8533MiB；VQ134801/update135800，latest130000 native重建FID2.5867905778，best仍115000/2.5664796810，contract16b75e...保持。130k seed123/fixed_val_first480_center_max300_raw_smpl_v1/480：Matching3.808081500、R .35625/.49375/.627083333、Div7.744163513/GT7.715767384、native22含root Body105.531544mm；较125k FID/Matching及Div差距较好、Body与R1R2较差/R3持平，非全项提升，非生成或joint FID。

130k十项RMSE、Full/Body non-root MPJPE、root、Full/Body joint FID、滑步、Contact F1待评；Hands MPJPE/FID不适用。此节点325B云JSON/SHA201bad873b460d3bd5dfea1ddc10527690246cea12c6f94417405c711d873942，本条不声称已下载/Drive备份。最近HHC完整官方480仍105k；本轮新InterMask公开VQ完整480重建对照见下方，不能替换任一生成baseline数字。InterGen scratch33200/best20000 FID24.314143；原三个queue状态仍waiting_for_complete_vq / waiting_for_all_mgt / waiting_for_zero_shape_scoring、phase_exit_codes空。

本轮新增工作是明确的同请求codec重建诊断，所有成功GPU/CPU原句柄已exit0；首启导入失败1及修正已完整记录，没有训练重启或绑定源码改动。已有agent等12:10只查一次；本机checkpoint/H5/NPZ下载0，GPU仅云端。继续完整VQ300k→ABC40k→五候选真实text-only val选模→正式20rep main/MM；整体SOTA及InterGen/InterMask/TIMotion/CoShMDM全目标仍未完成，不以重建对照替代生成比较。

**11:44 SGT：InterMask原生VQ统一480重建对照完成。** 复用已有官方源码/公开VQ ep46/iteration20608与原导出器，InterMask重建shared InterCLIP FID.985963612；当前HHC105k中间快照2.603228230，相差1.617264618。二者同冻结480/GT-text embeddings/evaluator/逐请求编码/96检索batch/metric seed20261006/91450 person-frames，GT/text maxdiff0。InterMask Matching3.793553988、R .427083333/.5875/.689583333均优于HHC105k点估计；Div7.592753887相对GT7.741262436差距大于HHC7.711113453。这里是GT motion重建，不能替换此前InterMask文本生成FID8.931384，亦不表示HHC text-to-motion输赢或SOTA。

InterMask重建十项：坐标RMSE78.177mm；Full22-joint MPJPE109.967mm；Body21-nonroot110.963mm；Hands MPJPE不适用；root89.063mm；Full joint FID待评；Body joint FID待评；Hands joint FID不适用；滑步3.212800mm/frame；关节距离接触代理F1 .586442786（semantic/mesh contact仍待评）。对照HHC105k分别为RMSE285.025/Full269.304/Body270.111/root252.367/滑步12.403673/代理F1 .670726166；HHC代理F1更高。GT参考滑步4.258247mm/frame；较低滑步不自动证明整体物理质量更好，未估跨codec差异CI。

此比较覆盖表征、量化及坐标转换整条路径，不能孤立归因某一模块。模型结构、输入及训练合同不同：Mask用normalized262D、现有导出器zero-pad到ceil(L/4)*4再trim；HHC用raw SMPL + requested GT betas、repeat-last raw pad及官方转换。released Mask与HHC中间105k不是等预算或等条件对照，iteration数字也不代表相同训练成本；重建FID差不是理论生成性能界。原VQ300k和A/B/C/shape/text-only队列保持，真实生成候选出来后再据证据决定改表征/量化/转换，不直接换baseline码本冒充HHC改进。

GPU成功原exec86749 exit0/peak1.165411GiB；shared CPU62731与geometry33974均exit0。首启误用依赖路径缺torch_geometric在模型导入前exit1，确认输出目录为空后改用已有/opt/hhc-baseline-deps:/opt/hhc-vq-deps完成；源码/模型方程未改，失败及环境路径已记入export_receipt。现有公开VQ权重af57bb...与四份官方源SHA逐项通过、strict state加载。H5 99539788B/SHA379e0fab6f2e5194e495beff428ee1584ca8452d34660e1689b52023f961d580；权重/H5/embeddings均留云。29绑定源码前后保持，未新增训练分支/Pod或重启原训练。

只收3个JSON共9988B，与云大小/SHA逐项一致，本机output/hhc_sota_20261016/intermask_vq_official480_codec；acceptance明确限制。合并原始证据intermask_hhc105k_official480_codec_evidence.json 28363B/SHA12174bfc2e9f9ead638b462948b4f1985b8837afc26d9cc9cb59334ee34b9a1d，[Drive证据备份](https://drive.google.com/file/d/1zP6l5LZ1E_Z4NKPJ-zRqQpGfl3FBCbHs/view) metadata大小/父目录读回一致、云checksum未暴露；收据同名.receipt.json。无本机checkpoint/H5/NPZ下载，不是权重备份。原agent仍等11:50单次普通核查，goal active，正式生成比较与SOTA仍未完成。

| 同一冻结480 GT重建指标 | released InterMask VQ | HHC native105k |
|---|---:|---:|
| InterCLIP FID（非joint FID） | .985964 | 2.603228 |
| 坐标RMSE（mm） | 78.177 | 285.025 |
| Full22-joint MPJPE（mm） | 109.967 | 269.304 |
| Body21-nonroot MPJPE（mm） | 110.963 | 270.111 |
| Hands MPJPE | 不适用 | 不适用 |
| root误差（mm） | 89.063 | 252.367 |
| Full joint FID | 待评 | 待评 |
| Body joint FID | 待评 | 待评 |
| Hands joint FID | 不适用 | 不适用 |
| 滑步（mm/frame） | 3.212800 | 12.403673 |
| 关节距离接触代理F1 | .586443 | .670726 |

上述数值不是文本生成性能；semantic/mesh contact F1均待评，输入/架构/训练/补帧条件不同，限定为同输出坐标/请求下的重建诊断。

**11:32 SGT：125k节点及110k–125k证据已保存。** 11:30低频SSH exit0、六原PID2353/5690/5729/7178/8321/9975 live且cmdline保持，GPU96%/8533MiB；VQ129001/update130000，latest125000 native重建FID2.8569413646，best仍115000/2.5664796810；120k为2.6634757373。seed123/fixed_val_first480_center_max300_raw_smpl_v1/480：125k Matching3.808280118、R .358333333/.525/.627083333、Div7.782913208/GT7.715767384、native22含root Body104.126064mm。相对120k FID/Matching/Body/Div差距较差，R1R2较好而R3较差；不归因数值故障、不改原配置或据波动选取有利指标。

125k十项RMSE、Full/Body non-root MPJPE、root、Full/Body joint FID、滑步、Contact F1均待评；Hands MPJPE/FID不适用，native22含root Body不能替代标准Body。最新完整官方冻结480仍为105k/shared FID2.603228及其十项，非text-only生成或SOTA。已确认两种480仅418共同ID，其中97条共同裁剪起点/长度同时相同；后续比较继续既有统一冻结请求，不把native与official数值互换。

本轮只收110k347B/120k335B/125k334B共1016B小JSON，均逐项匹配云SHA；加既有115k原始值构成合并证据vq110000_125000_validation_evidence.json 3654B/SHA279586420773e3e95646096045d90eda7fc1214981511752f7647b4d850635c9，[Drive证据备份](https://drive.google.com/file/d/18StIUTSiXVd3XT8bgELWwTv4FinDcYGx/view) metadata大小/父目录一致、云checksum未暴露，收据同名.receipt.json。没有新checkpoint/H5/NPZ下载、GPU评测、训练、源码修改或重启；不是权重备份。

InterGen scratch32550/best20000 FID24.314143；三个queue分别waiting_for_complete_vq / waiting_for_all_mgt / waiting_for_zero_shape_scoring，phase_exit_codes为空。已有agent等11:50只做一次普通复核。继续原300k→ABC40k→五候选真实val→20rep正式main/MM，goal active，生成SOTA及四方法正式比较未完成。

**10:52 SGT：115k小幅新最佳。** 10:50一次SSH exit0确认六PID2353/5690/5729/7178/8321/9975及原命令保持，GPU91%/8533MiB，VQ118001/update119000、latest/best115000 FID2.5664796810（旧best105000/2.5744459359）。seed123/native fixed_val_first480_center_max300_raw_smpl_v1/480：Matching3.807768250、R .34375/.5125/.616666667、Div7.735663891/GT7.715767384、native22含root Body104.184688mm；相较110k FID/Body改善，但Matching与三项R回退。不是全面提升、生成FID、joint FID或SOTA，不因.008的native best变化重做整套官方480重建；最近完整官方480仍105k/shared FID2.603228。

115k十项RMSE、Full/Body non-root MPJPE、root、Full/Body joint FID、滑步、Contact F1待评；Hands MPJPE/FID不适用，native22含root Body不替代标准Body项。323B JSON SHAf12c8efd9041284246ee20ee36b07ec298f97352ae991ebcee670862275007b9已与云匹配，[小型结果备份](https://drive.google.com/file/d/1N3VSoMxNyvnRM9eUU81yFi8i5fQd5ojj/view) metadata大小/父目录一致、checksum未暴露；收据vq_rebuild_validation_115000.receipt.json。本机checkpoint新增0，无新GPU评测/训练/源码改动。

InterGen scratch31300/best20000 FID24.314143；原三个queue仍waiting_for_complete_vq / waiting_for_all_mgt / waiting_for_zero_shape_scoring，phase_exit_codes空。已有agent只等11:10下一次普通复核。继续完整300k→ABC40k→五候选真实text-only val→正式20rep比较；goal active，生成SOTA尚未证明。

**10:31 SGT：110k新验证与请求审计补充已落实。** 10:30一次低频SSH exit0、六PID2353/5690/5729/7178/8321/9975 live，GPU94%/8533MiB；VQ112601/update113600，latest110000 native重建FID2.6711099246、best105000/2.5744459359，contract16b75e...保持。110k Matching3.807584572，R .358333333/.514583333/.629166667，Div7.775691032/GT7.715767384，native22含root Body104.632332mm；seed123/fixed_val_first480_center_max300_raw_smpl_v1/480。较105k FID/Matching较差、Body略好、检索混合，非全面改善；原始347B JSON云SHAc88da3c5e079de14a60a9ae5c308dc1637ee647b6f5b2c499cec9919148eb0aa，本条监控不声称该110k JSON已收本机/备份。

110k十项RMSE、Full/Body non-root MPJPE、root、Full/Body joint FID、滑步、Contact F1待评；Hands MPJPE/FID不适用，native22含root Body不替代标准Body。最新完整官方冻结480验收仍为105k/shared FID2.603228及下方十项与60k配对区间；不移作110k结果。InterGen scratch30700/best20000 FID24.314143；三queue仍waiting_for_complete_vq / waiting_for_all_mgt / waiting_for_zero_shape_scoring、phase空。没有启动新GPU/训练分支或重启；本机checkpoint新增0，goal active，继续原完整300k及后续真实生成选模/正式比较。

**10:21 SGT：105k官方冻结480重建已完整验收。** native seed123最新/best105000 FID2.5744459359，旧best65000/2.6810599747；官方固定480/metric seed20261006 shared InterCLIP FID2.6032282300，较60k2.5775033109点估计略差，未估FID差异CI。Matching3.801341502、R .397916667/.579166667/.670833333、Div7.711113453/GT7.741262436均较60k方向更好。是GT motion+requested GT betas重建、较60k额外45000更新，不是等预算算法或text-only生成；两种请求crop协议不混用，native新最佳不等于统一FID新最佳。

10:25 SGT只读CPU请求审计补充：训练内480与官方冻结480各有480个唯一ID，交集418、相同位置仅3；418共同请求中裁剪起点相同188、raw长度相同128、两者同时相同97、官方caption_index=0仅137。官方anchor1/2为261/219、输出swap false/true为246/234；native默认anchor1/no output swap/first caption；diversity seed20260130与官方shared20261006不同。源码确认native Body为SMPL世界基底下22joint含root的FK误差，官方Body是统一processed坐标下21个non-root；不是同一指标定义。审计量化协议差异，不构成差异原因的因果证明，也不是改分动作，所有协议/29源不改。新1259B JSON SHA7b61dfd00668b9c6a343c8ec6b7a973a8e23bd720400e661369887e9db5cf7e6与云一致，[小型审计备份](https://drive.google.com/file/d/1UZqIlg3Hw1nzWjrgd13_f0NURN2NN5Hr/view) metadata大小/父目录一致、checksum未暴露；无模型/生成数组读取、GPU任务或checkpoint下载。

105k完整十项（官方坐标，无刚体/root对齐）：坐标RMSE285.025mm（60k256.300，变差）；Full22-joint MPJPE269.304mm；Body21-nonroot MPJPE270.111mm；Hands MPJPE不适用；root252.367mm；Full joint FID待评；Body joint FID待评；Hands joint FID不适用；滑步12.403673mm/frame；接触距离代理F1 .670726166（semantic/mesh contact F1仍待评）。60k→105k配对val标识cluster/10000bootstrap：Full delta−14.054mm CI[−41.002,14.031]、root−15.268[−42.668,13.094]、滑步−.11048[−.31629,.09772]均跨0；接触代理+.033556 CI[.009617,.057157]不跨0，为逐指标未作多重校正区间、固定两模型val不确定性，不是训练seed或生成SOTA置信区间。

本轮freeze/GPU export9695 child13710/sharedCPU13425/geometry26115/paired30870均exit0，原句柄已全部收齐，不重跑。恒等坐标max.004701mm、接触365800值不匹配0、冻结weights未变/GPUpeak.127GiB，GT/text embeddings与三公开基线参考逐项完全一致；云及本地29绑定源SHA均保持。spec output/hhc_vq_intermediate_frozen_best105k_20261016/frozen_vq_spec.json SHA7baf4ae7a0e62dfd915d1548daf004f7344f181c4eb1ad48c497f4dbfa14aa38；ckpt115069874B/SHA81314b813fa95564ca705bba6f5322aacd705610810fbe85eeaa3a752536e672；官方H5 95109004B/SHAa71cd3e0a809946327055920abedd172abf15fe73ff513754bee157ac46317b1，均留云。scope仍intermediate_val_selected_pilot，不替代完整300k final冻结。

本机6个小JSON共12717B逐项对云大小/SHA一致，acceptance.json在output/hhc_sota_20261016/vq_official480_best105k。合并原始证据vq105000_official480_evidence.json 20673B/SHA328197ed7100c56c7ef4ad59599d728a46719c1b1cf61f14286c1f991bf17bc3；Drive [证据备份](https://drive.google.com/file/d/168PpXasza2BdD-_iMlBDUA10FECrv5TW/view) metadata大小/父目录读回一致，云checksum未暴露；收据同名.receipt.json。无ckpt/H5/embedding/prior数组下载，不是模型权重备份。

最近普通10:10六原PID live/三queue仍waiting，主线程未追加普通PID轮询；已有agent等待10:30只查一次。继续原300k→ABC40k→五项真实text-only val选模→正式20rep main/MM，不因单项proxy变好改选模协议，不新增算法/外围代码；goal active，尚无生成SOTA证据。

| 同一官方480重建指标 | 60k | 105k |
|---|---:|---:|
| InterCLIP FID（非joint FID） | 2.577503 | 2.603228 |
| 坐标RMSE（mm） | 256.300 | 285.025 |
| Full22-joint MPJPE（mm） | 283.358 | 269.304 |
| Body21-nonroot MPJPE（mm） | 284.107 | 270.111 |
| Hands MPJPE | 不适用 | 不适用 |
| root误差（mm） | 267.635 | 252.367 |
| Full joint FID | 待评 | 待评 |
| Body joint FID | 待评 | 待评 |
| Hands joint FID | 不适用 | 不适用 |
| 滑步（mm/frame） | 12.514150 | 12.403673 |
| 关节距离接触代理F1 | .637170 | .670726 |

GT参考滑步4.258247mm/frame；接触代理是每帧双人22×22关节最短距离≤.1m，不是semantic/mesh contact。normalized feature RMSE另为.607187（60k.599847），单位无量纲，不替代上表坐标RMSE。

**09:52 SGT：100k里程碑及90k–100k原始结果验收。** fresh VQ seed123/fixed_val_first480_center_max300_raw_smpl_v1/480：最新100000重建FID2.8204264645，best仍65000/2.6810599747；90k=2.9255896749、95k=2.7455787029。100k Matching3.808834394，R1/2/3=.345833333/.53125/.633333333，Div7.783712387/GT7.715767384，native22含root Body105.544894mm。较95k FID/Body略差、Matching/R2/R3较好、R1较差、Div差距略收窄；未证明整体提升，也不能从100k尚未刷新最佳推断后续必然无效。它是训练内重建验证，非text-only生成、joint FID或SOTA。

100k十项：RMSE待评；Full MPJPE待评；Body non-root MPJPE待评（native22含root监控不能替代）；Hands MPJPE不适用；root待评；Full joint FID待评；Body joint FID待评；Hands joint FID不适用；滑步待评；Contact F1待评。最近官方冻结480完整评分仍为60k，禁止把其数值移作100k。主训来源/contract保持；没有因训练内波动新增GPU实验或改参。

09:30与09:50两次低频SSH均exit0、六PID2353/5690/5729/7178/8321/9975均live；最近09:50 GPU99%/8533MiB，VQ101401/update102400，InterGen scratch30000/best20000 FID24.314143。三个queue仍waiting_for_complete_vq / waiting_for_all_mgt / waiting_for_zero_shape_scoring，phase_exit_codes均空；09:51只取小文件时VQ manifest继续到101801/update102800。数字验证已严格筛选，90k–100k文件及validation日志互证，trainerSHA b6fb51074b3a981b5d7401925022af30cc2267eda252e04be5be5ea7b86ed8bc实际匹配。

三原始JSON共1003B与云逐项大小/SHA一致，本机output/hhc_sota_20261016/vq_90k_100k_native。合并证据vq90000_100000_validation_evidence.json为4688B/SHA06efa58834526bf07075d9fe7364ea59cf607cfbefc4b44ebdee1c7fb6f4eebf，Drive [小型证据备份](https://drive.google.com/file/d/1Xo66yXXW3xcBP0-6BHvSwMTrqBL_6pav/view) metadata大小/父目录读回一致、云checksum未暴露；收据同名.receipt.json。无checkpoint/H5/NPZ下载，不是权重备份。

继续原VQ300k→ABC40k→五项真实text-only val选模→20rep正式main/MM，尚未产生formal test/SOTA证据，goal active。下一次普通监控不早于10:00 SGT；保护规则及本地新增checkpoint=0保持。

**09:11 SGT：90k验证与原队列存活确认。** 09:10低频一次云端实查SSH exit0，六PID2353/5690/5729/7178/8321/9975及命令保持、GPU100%/8533MiB。VQ manifest running/last90601/update91600，最新90000完整480 native重建FID2.9255896749，best仍65000/2.6810599747；Matching3.809105555，R1/2/3=.360416667/.508333333/.60625，Div7.739639282/GT7.715767384，native22含root Body104.771998mm，seed123/fixed_val_first480_center_max300_raw_smpl_v1。90k文件mtime及validation事件01:08:34UTC互证；train90601/01:10:17UTC loss.074296318。不是生成或joint FID，未有新的官方冻结480完整评分。

90k十项：RMSE、Full MPJPE、Body non-root MPJPE、root误差、Full/Body joint FID、滑步、接触F1均待评；Hands MPJPE和Hands joint FID不适用，native22含root Body不能替代标准Body项。InterGen scratch28800/best20000 FID24.314143，是独立训练。三个queue分别waiting_for_complete_vq / waiting_for_all_mgt / waiting_for_zero_shape_scoring，phase_exit_codes均空。新训练/GPU评测启动0、重启0、本机checkpoint下载0；继续原300k→ABC40k→五候选真实val→20rep main/MM，SOTA未证明、goal active。

本次监控仍误将validation_panel.json纳入“最后两份”枚举，未按要求全匹配数字文件；90k的指定文件和对应日志直接读取有效，不据错误枚举推断其他节点缺失，已要求后续先re.fullmatch数字文件再排序。本条是已存活句柄的verified wait；没有因等待启动新分支。下一次普通监控不早于09:20 SGT。

**09:02 SGT：80k/85k完整验证与监控误报更正。** fresh VQ seed123、fixed_val_first480_center_max300_raw_smpl_v1/480：80k FID2.7475328198，85k FID3.0052041532，历史best仍65000/2.6810599747。85k Matching3.811837387，R1/2/3=.35625/.50000/.61250，Div7.761117935/GT7.715767384，native22含root Body104.925321mm；较80k Body略好，但FID/Matching及Div差距变差，检索指标混合，不能称全面改进。这是训练内重建监控，不是text-only生成、joint FID或SOTA。

85k十项：RMSE待评；Full MPJPE待评；Body non-root MPJPE待评（native22含root监控不能替代）；Hands MPJPE不适用；root待评；Full joint FID待评；Body joint FID待评；Hands joint FID不适用；滑步待评；Contact F1待评。最近完整官方冻结480仍为60k及其对应几何结果，不挪作85k数值。

08:59–09:00异常复核确认85k文件真实存在308B/SHAa971ef2069f50aa103d823cb2745d21676d73cce6873c5cdccf6a5ac5ea12048，原始metrics.jsonl记录00:49:55UTC validation85000。先前08:55“85k不存在”结论错误。监控agent追溯原始查询确认：当次只读metrics最后3条（恰为train），没有文件存在性/目录/85k读取检查；把未显示事件推成文件缺失没有证据，不能归因训练故障。actual trainer SHA b6fb51074b3a981b5d7401925022af30cc2267eda252e04be5be5ea7b86ed8bc与绑定一致，manifest running/last88601/update89600/best65000；无重启、改参或源码修改。无输出的只读观察session77654已人工Ctrl-C收回exit1，该终态仅属观察命令，不属训练退出。

80k319B/SHA831fcd1f1f0e5ef48e7dcd087b269cc616986596d7b60cb44b1873586e3673ec与85k小JSON均逐项对云大小/SHA一致。合并证据vq80000_85000_validation_evidence.json为3017B/SHA859891b37ead607088ad04922e4ee6fb2d9a21311fa0ec5b185882247ada0bd2；Drive [小型证据备份](https://drive.google.com/file/d/12gVJCQvs0HTcMo7tEEexJkRSth6Eh6Dz/view) metadata大小/父目录读回一致，云checksum未暴露；收据同名.receipt.json。仅JSON，无checkpoint、H5或NPZ下载；不是权重备份。

最近六PID存活普通复核仍为08:55，不把本次单VQ文件核验冒充六进程实查；三个队列尚无已完成phase证据。继续原VQ300k→ABC40k→五项真实text-only val→20rep正式main/MM。保护规则/本地禁新增checkpoint保持；goal active，尚未证实生成SOTA。

**08:17 SGT 新完整70k/75k及日志核查：** VQ最新75000 native重建FID3.1021847726，best保持65000/2.6810599747；70000为2.6933088819。75k Matching3.811388/Body native22含root112.060877mm较70k3.809607/104.785631变差，R1 .350000改善、R2/3 .506250/.610417回退，Div7.746270/GT7.715767差距略收窄。70k R .333333/.516667/.639583、Div7.752460/GT同。seed123/fixed_val_first480_center_max300_raw_smpl_v1/480；非生成、joint FID或SOTA，不能声称全项回退/改进。

05:50 SGT本轮源码/小结果17项已逐payload和归档读回，263941B/SHA984c395b6965167c45ef720846e0ca8598c3b963f42e0d42c8a73f7acb797a89；Drive [增量归档](https://drive.google.com/file/d/1SJO69YpKYFNzyud0Px95ulzfp6upwZLZ/view) metadata大小/父目录一致，未提供云checksum。无checkpoint、motion H5、prior NPZ。归档包含备份完成前的文档快照，这条收据说明另存本机；不是模型权重备份。

06:25 SGT本轮源码/小结果33项逐payload和归档读回通过，297075B/SHA8e37784f43aab7a1db526afbd126444b4b5cb738ec74f0bb89fd04ab64360125；Drive [比较路径增量归档](https://drive.google.com/file/d/1J3JeNmjQkZWCZGMiPKSl1t-kSUQ-i57z/view) metadata大小/父目录一致，未暴露云checksum。无checkpoint/motion H5/prior数组，非权重备份；归档保存备份完成前的文档快照，此收据说明另存本机。模型加载验收细节：Gen/TI全state严格，Mask VQ严格，Mask transformer依官方接口只允许缺冻结预训练clip_键（由实际CLIP加载），所有其他缺键/意外键均拒绝；不是CoSh那种跳过可训练text encoder键。

**06:40 SGT 50k完整验证与备份覆盖：** VQ最新完整50000重建FID2.9257481069，best仍35000/2.8272352923；native22含root Body107.314958mm改善，Matching3.809512/R .358333/.529167/.625000/Div7.731749 GT7.715767，非全指标改进。seed123/native fixed_val_first480_center_max300_raw_smpl_v1；其十项RMSE/Full/Body nonroot/root/Full-Body joint FID/滑步/contact待统一评，Hands MPJPE/FID不适用。不能混用35k官方480重建的完整几何指标，不能当生成/SOTA。334B小JSON SHA1d763b79bb1dc8398f7bf598e002a91502883c79c67e520bd5ba954a7bdd1f73与云一致。

本轮35k官方480源码/小结果13payload包migration/hhc-vq35k-official480-20261016.tar.gz，263587B/SHA401ab03910629a81578617fd41120b26e0ffb5f7bbf9637b490e36c0edf1eb3e，逐payload与tar读回通过，Drive1sKH5cp84IUcitutBohevAglkHLrWOSLJ metadata大小/父目录一致、未暴露云checksum；收据同名-backup-receipt.json。无checkpoint/motion H5/embedding/prior数组；不是权重备份，归档含完成前文档快照，50k JSON/本条完成说明在包后新增。原29源/训练/三个队列保持，普通监控仍>=06:46，不需因这次artifact验收补查PID。

06:47 SGT长度/时间诊断及50k小结果7payload归档migration/hhc-vq-length-time-diagnostic-20261016.tar.gz，260474B/SHAdbf63f010b05e60c3975aba7de5763d87fc1fa12e54605ce6944aeef59dff00b，逐payload/tar读回通过；Drive1AxIkZYTtCvK1kQT3OsFl1V_zz081_LHr metadata大小/父目录一致、未暴露云checksum。无weights/motion H5/embedding/prior数组；非权重备份，包内是完成前文档快照，本条完成说明与收据另存。06:46普通实查六PID live，VQ52401/latest50000 FID2.925748/best35000 2.827235，InterGen24250/best20000 24.314143，GPU99%/8533MiB；原三个队列phase空/仍waiting，v2为9975。下一次普通检查>=06:56，不改29源/不新增本机checkpoint。

07:27 SGT 60k官方480源码/小结果12payload归档migration/hhc-vq60k-official480-20261016.tar.gz，273921B/SHAd581165dd927a9b682c84b7f0677ad37cc38db63f3f571a566fec52eba1efd31，逐payload和tar读回通过；Drive1kjj4IdKdmj0KfabKGQX7aRGSEYuLN1QB metadata大小/父目录一致、未暴露云checksum，收据同名-backup-receipt.json。无checkpoint/motion H5/NPZ embedding/prior数组，不是权重备份；包中是完成前文档快照，本条完成说明另存。所有本轮CPU/GPU原句柄终态exit0、不要重跑。正式导出/主-MM评分/汇总/存储路径已备齐；除具体故障修复外，未有真实生成候选前不再追加外围代码，优先等待原300k训练与后续生成消融。07:24原六PID live/三queue等待，下一次普通检查>=07:34；本机ckpt新增0，29源不改，goal仍active。

## 08:17 SGT：70k/75k训练内验证与回退日志核查

**08:17 SGT 新完整70k/75k及日志核查：** VQ最新75000 native重建FID3.1021847726，best保持65000/2.6810599747；70000为2.6933088819。75k Matching3.811388/Body native22含root112.060877mm较70k3.809607/104.785631变差，R1 .350000改善、R2/3 .506250/.610417回退，Div7.746270/GT7.715767差距略收窄。70k R .333333/.516667/.639583、Div7.752460/GT同。seed123/fixed_val_first480_center_max300_raw_smpl_v1/480；非生成、joint FID或SOTA，不能声称全项回退/改进。

75k十项：RMSE待评；Full MPJPE待评；Body non-root MPJPE待评（原生22含root监控不替代）；Hands MPJPE不适用；root待评；Full joint FID待评；Body joint FID待评；Hands joint FID不适用；滑步待评；Contact F1待评。最近完整官方冻结480仍为60k/shared FID2.577503和对应十项，不移作75k数值。

因75k验证回退，只读核查metrics完整222816B快照：391个train/15个validation/3个ready/2个resumed，非有限值0、malformed0；55–75k最近train原始行的loss从.086821到.080875，lr/perplexity/components均有限。未发现已记录的数值故障，不等于全面参数检查、泛化或原因判定；单批loss不可作训练趋势统计。原有ready/resumed为历史记录，本轮重启0。此前口头提取把20个非train事件都概括为validation不准确，以完整快照分类为准。manifest running/last75601/update76600/best65000，contract16b75e...保持；不因单次FID反弹重启、改源码或新增分支。

70k346B/SHA23e85552d81f38e9c95dac03a0d55b1bb1f55510f870ae9c562e923436eb62df；75k320B/SHAcc500b6653416f4b56a045739a2829d33684609091b820575d19d99a50f130dc；日志审计4652B/SHA5d42a9133e60bf8debe9662874656df934348326053d09a04f6fbc857a1d62ae，均与云大小/SHA匹配。合并小证据vq70000_75000_validation_and_log_evidence.json在output/hhc_sota_20261016。只收JSON，无GPU新增工作/模型或数组读取/ckpt下载，75k原观察exec71862 exit0。

08:10:49真实/proc复核六PID2353/5690/5729/7178/8321/9975 live、VQ74801/InterGen26950（best20000 24.314143）、GPU12%/8533MiB、三个queue waiting/phase空；随后75k完整节点和日志manifest继续推进。下一次普通>=08:21。继续原300k→ABC40k→五项真实text-only val→20rep正式main/MM，goal active，尚无生成SOTA证据。

## 07:39 SGT：65k训练内完整验证

**07:39 SGT 新完整65k训练内验证：** seed123/native fixed_val_first480_center_max300_raw_smpl_v1/480，latest/best65000 FID2.6810599747（旧best60000/2.6950839104）；Matching3.807880656改善，但Body native22含root108.376098mm较60k107.325754变差，R1/2/3 .345833/.506250/.610417均回退，Div7.754404/GT7.715767差距扩大。不称全项改进、生成或SOTA。334B JSON SHAb4b82bee1b8359c1942bcd118a96204d9651f2825c63907db9d3bf3963bcbe1e已与云只读SHA核对；本轮仅取小JSON，无GPU新增工作/ckpt下载/训练源码修改。

65k十项：RMSE待评；Full MPJPE待评；Body non-root MPJPE待评（native22含root监控不替代）；Hands MPJPE不适用；root待评；Full joint FID待评；Body joint FID待评；Hands joint FID不适用；滑步待评；Contact F1待评。最近一次完整官方冻结480是60k/shared FID2.577503及对应十项，不能移作65k结果。

07:38实际/proc和两次SSH exit0确认六PID2353/5690/5729/7178/8321/9975 live；VQ66001/update67000，InterGen25900/best20000 FID24.314143，GPU100%/8533MiB，三个queue保持waiting/phase空。下一次普通检查>=07:48；继续原300k→ABC40k→五候选真实val→20rep正式main/MM，无新增分支/外围代码，goal active。

## 07:25 SGT：60k统一官方480重建完整结果

VQ训练seed123、共享评测seed20261006，same frozen480 requests（content SHA751efc9ced91e2a6ad4eec4a7b9316946761fe13f86d09af7d5c04a9226b7f93）。60k和35k的source_run/training_contract/normalization相同，增加25000主训练更新；两者为intermediate_val_selected_pilot，不代表300k最终冻结或等预算新方法。输入GT motion和requested GT betas作重建，不能同公开text-only生成表直接比较胜负。

| 共享官方480重建 | 35k | 60k |
|---|---:|---:|
| InterCLIP FID | 2.6485284692 | 2.5775033109 |
| Matching | 3.8040469488 | 3.8047686895 |
| R@1 | .393750 | .395833 |
| R@2 | .558333 | .562500 |
| R@3 | .645833 | .652083 |
| Diversity，GT7.741262436 | 7.687822342 | 7.674005508 |

FID观察值下降.0710251583，未估计FID差异CI。Matching均值变差，Div到GT的差距扩大。native center480是另一协议，60k FID2.6950839104优于35k2.8272352923；不可混成同列或把shared FID当joint FID。shared GT/text embeddings对公开三基线逐项相等/maxdiff0。

| VQ十项，官方480，无root/rigid对齐，valid-person-frame pooled | 35k | 60k |
|---|---:|---:|
| Joint coordinate RMSE/mm | 253.412691 | 256.299515 |
| Full MPJPE/mm，22joints含root | 268.367411 | 283.358483 |
| Body non-root MPJPE/mm，21joints | 269.183969 | 284.107205 |
| Hands MPJPE | 不适用 | 不适用 |
| Root position error/mm | 251.219696 | 267.635334 |
| Full joint FID | 待评 | 待评 |
| Body joint FID | 待评 | 待评 |
| Hands joint FID | 不适用 | 不适用 |
| 滑步/mm每帧，GT-contact transitions | 12.348529 | 12.514150 |
| Contact F1 | 几何代理.632373 | 几何代理.637170 |

双方91450 person-frames；GT滑步4.258246824mm/frame。接触代理为22×22跨人关节最小距<=.1m，不是semantic/mesh contact F1，后者待评。60k normalized feature RMSE0:258=.5998470716；关节坐标RMSE不能与该无量纲值混淆。

| candidate60k-control35k，480 identifier clusters，bootstrap10000次/seed20261016 | 差值 | 95% CI |
|---|---:|---|
| Full MPJPE/mm | +14.991072 | [-4.174809,35.108712] |
| Root/mm | +16.415638 | [-3.170328,37.061989] |
| Sliding/mm每帧 | +.165621 | [-.008764,.341296] |
| Geometry contact F1 | +.004796 | [-.023391,.033022] |

四项CI均跨0；不是训练seed或生成/SOTA不确定性。位置/滑步均值变差，未证明稳定恶化，也没有几何改善证据。**保留多指标结论，不以FID下降晋升为整体成功或提前替代最终VQ。** 主300k及原ABC/shape生成消融继续，真实HHC生成/正式比较仍待。

云冻结checkpoint115069874B/SHA3df73e015053f1c7a625a1eba8ae3a39f0edfc053c5875035cd74874719c214c，spec SHA4e62e6f5be2f02a6def52b15225e4143b141c26c74f0d7029be470e36f5d1126。GPU export54186/child12096 exit0，H5 95130020B/SHAb6b709f2125663ffe8e45ea6c9bba60e6c7710ee9840b771b88ca250e0ce0e49；恒等转换最大.004701mm/接触365800值0不符、padding先去除、权重未变、GPUpeak.127GiB。CPU32269/48271/43843全部exit0；不得重跑同目录。新配对工具验证两spec/H5 SHA和共同训练contract，其metrics和10000 bootstrap AST与原实现相同；ledger额外核对paired与geometry差<1e-7mm、29绑定源/云到本机6小JSON大小SHA通过。小JSON12703B，无ckpt/motion/embedding下载；本机ledger=migration/record_hhc_vq60000_official_20261016.py。

07:24普通/proc监控六PID活跃，VQ62601/latest-best60000/2.695084，InterGen25450/best20000/24.314143；三个queue phase空且waiting。下一次普通>=07:34。当前仅中间重建测量，正式final40k五候选选模receipt仍不存在。

## 07:20 SGT：60k训练内新最佳与统一重建验收启动

**07:20 SGT 60k新最佳及官方480验收进行中：** native seed123/fixed_val_first480_center_max300_raw_smpl_v1/480的step60000重建FID2.6950839104，best刷新为60000（旧35000/2.8272352923）；native22含root Body107.325754mm、Matching3.810742760、R .358333/.510417/.622917、Div7.746598/GT7.715767。R2/R3及Matching较55k变差，不能称全项提升；非生成/SOTA。新348B JSON SHA26cc99f51ce9c01c4033201949298f45e35a338046c932b0a569ed9495cd4031。

云端中间spec output/hhc_vq_intermediate_frozen_0717_20261016/frozen_vq_spec.json，checkpoint115069874B/SHA3df73e015053f1c7a625a1eba8ae3a39f0edfc053c5875035cd74874719c214c，scope仍intermediate_val_selected_pilot，不替代300k final freeze。原GPU export exec54186/child12096 exit0，output/hhc_native_vq_official480_best60k_20261016/reconstructions.h5 95130020B/SHAb6b709f2125663ffe8e45ea6c9bba60e6c7710ee9840b771b88ca250e0ce0e49；恒等坐标max .004701mm/接触365800值0不匹配，冻结权重保持/GPU peak.127GiB，全在云端。主训/29绑定源不改，实测15551MiB GPU free后才导出。

**当前三个云CPU原句柄：** shared InterCLIP exec32269、geometry exec48271、35k→60k paired exec43843，均待收终态，不重复启动。新compare_hhc_vq_training_checkpoints_cpu.py SHA0f6cb32e3cad4b46e8186730c645f72866d307379eee6938b5685fb55cfb9fc2，校验两spec/H5 SHA、同source_run/training contract/normalization，计算metrics/10000次bootstrap AST与原工具相同；额外25000训练步，不是等预算新方法。60k十项RMSE/Full/Body nonroot/root/Full-Body joint FID/滑步/contact待统一评，Hands MPJPE/FID不适用。共享FID不能替代joint FID；原始GT motion+requested GT betas重建不可作为text-only生成。

07:13:48主线程异常接管直接/proc确认六PID2353/5690/5729/7178/8321/9975 live，VQ59801/update60800，InterGen实际STATUS.json=25150/best20000/FID24.314143，三queue waiting/phase空；下一次普通>=07:24。此前多节点监控的中间报告未送达主线程，已仅中止观察agent等待，没有停止任何云训练。InterGen无training_manifest.json，应读STATUS.json/training.jsonl；VQ validation glob需仅匹配validation_六位数字.json，排除validation_panel.json。初次接管读取路径失败已补齐，绝不能误判训练失败。后续例行检查一次一返final，不再依赖中间send_message。

## 06:58 SGT：55k验证及MM归档备份验收

**06:58 SGT 最新完整55k覆盖：** native seed123/fixed_val_first480_center_max300_raw_smpl_v1的VQ55000重建FID2.8667573289，best仍35000/2.8272352923；Body native22含root108.769865mm较50k107.314958变差，Matching3.810460与R1/2 .352083/.525也变差，R3 .627083略改善，Div7.719928/GT7.715767，不能称全项进步。55k十项RMSE/Full/Body nonroot/root/Full-Body joint FID/滑步/contact待统一评、Hands MPJPE/FID不适用；不要挪用35k官方480几何数值。新334B JSON SHAd0004d4abc73e3a724b936a214b3254bde33334130e616d9fd9c8636f8c84d1b与云一致。节点观察/proc确认原5729 live且exec立即exit0，未重复启动。

06:56普通实查六PID2353/5690/5729/7178/8321/9975 live、VQ54801/update55800、InterGen24600/best20000 FID24.314143、GPU26%/8533MiB，原三个queue仍waiting/phase空；下一次普通监控>=07:06，55k节点验收不要求补查训练。完整300k→ABC40k→五候选真实val→正式20rep main/MM仍未完成，goal active。

06:58 MM归档源码/小验收11payload包migration/hhc-formal-mm-archive-20261016.tar.gz，271243B/SHA11309072e15c098485bec35540f1623ab792787d9602e9ad263231d764452814逐payload及tar读回通过；Drive1YN0cuon7PbG_Y0E6p3fmlx6hwxd1K1qv metadata大小/父目录一致、未暴露云checksum，收据同名-backup-receipt.json。无checkpoint/motion H5/NPZ embedding/prior数组，不是权重备份。归档含完成前文档快照，55k JSON及本条完成说明在包后新增；本机新增ckpt0，29绑定源不改，不追加研究分支。

## 06:55 SGT：正式MM embedding归档与复核

**06:55 SGT 正式MM归档路径验收：** 新tools/score_hhc_text_only_formal_mm_cpu.py只在真实五候选val选模receipt核验后做云CPU formal MM，要求相同MGT/VQ/shape/sampling及每prompt固定体型。它保留原score_fair_interhuman_mm.multimodality_from_embeddings的同样100×30×D官方encoder输出、seed1234+rep/pairs10/scale6/divisor2，并保存float32 NPZ及SHA。归档保持encoder返回顺序；没有为了对齐seed改变旧wrapper排序。新score schema interhuman_fair_mm_score_v2可由既有四方法汇总接入，显式消费先验score/H5/独立NPZ哈希；旧v1没有archive仍保留H5。未执行真实文件回收/正式test。

云CPU exec57685 exit0、32项合成集成检查：原160score汇总/CI/保护，真实MM H5请求布局与mock encoder100次，保留矩阵逐位相同、原官方MM算式完全一致、Numpy RNG保持，错误winner/GTshape/跨repeat改体型/模型SHA拒绝，归档损坏/错维度/H5别名拒绝，v2仅删自建fixture H5后仍能从NPZ复算MM；所有fixture删除，真实生成删除0。**InterCLIP encoder和真实选模receipt在此测试中mock，不能当实际HHC test或质量验收。** 已通过测试不重跑；2683B audit SHA0a724af3a2d26f85ff27fc7124e8726a702650dc9bf976237dcf3e32504fa548与云一致。新scorer SHA575bf2d149fa17c6579c6ebedd420b5ddf430784f3198dabcec6ad6a54e08651，future score adapter SHAa3a5fb44691090961b811bc8f8458570fec293fd0f518f8a49bb3351db6ad0c9，旧393c4f...源码快照保留；29活跃绑定源不改。hourly hhc已app更新并TOML读回ACTIVE、原target/rrule/禁本机ckpt下载保持。小验收mm_embedding_archive_acceptance.json。

## 06:44 SGT：误差来源的补充CPU诊断

**06:44 SGT root/长度诊断完成：** 云CPU exec46787 exit0，仅读已有5k/35k官方480重建，无模型加载/GPU。35k Full268.367mm、root251.220mm与已验汇总相差<1e-7mm；每帧剔除自身root误差后22joint残差75.332mm（GT oracle诊断，不能修正输出或作主成绩）。初帧root196.530mm、末帧277.063mm；归一时间四分位220.277/243.959/257.663/283.980mm，说明初始空间误差和时间累积均存在。按长度14–63/64–127/128–191/192–255/256–299分组的root为207.661/285.083/238.992/313.805/231.918mm，组间非单调且动作不同，不能宣称只扩训练窗口即可解决。新10,075B JSON与云SHA9f1dbd3f85bf0722a6664b418cfc018c11a62f8a2e954bbc322ca2f8198df706一致，29源不变，本机ckpt/H5/embedding新增0。保留原ABC/形状生成消融，暂不扩分支。

## 06:40 SGT：35k VQ统一官方480重建、配对区间与训练覆盖

VQ seed123，fresh rebuild主训练step35000，从仍运行主训best原子替换文件创建云端不可变hardlink，scope为intermediate_val_selected_pilot。checkpoint115069874B/SHA694908189b37f6582a6e7496a6679b27f0708d1a2b195a52e90bf05f17471f46；不作为300k终态冻结VQ。请求content SHA751efc9ced91e2a6ad4eec4a7b9316946761fe13f86d09af7d5c04a9226b7f93，与公开三基线相同；GT/text embeddings逐项完全相等。使用GT motion与requested GT betas作重建，原始帧repeat-last补4倍数、重建后先去padding再官方处理，不能当pure text-only生成成绩。

共享InterCLIP FID **2.6485284692**、Matching3.8040469488、R@1/2/3=.39375/.5583333333/.6458333333、Div7.6878223419/GT7.7412624359。它与训练内center480的best FID2.827235是不同请求/crop协议，也不是Full/Body joint FID。相较5k共享FID4.5758377859改善，但额外30000步训练并非等预算新算法收益；不能拿重建FID胜过生成基线来宣布SOTA。

| VQ十项，step35000，官方480重建 | 数值/状态 |
|---|---:|
| Joint coordinate RMSE | 253.412691mm |
| Full MPJPE，22 joints含root | 268.367411mm |
| Body non-root MPJPE，21 joints | 269.183969mm |
| Hands MPJPE | 不适用，无手指 |
| Root position error | 251.219696mm |
| Full joint FID | 待评，当前InterCLIP不可代替 |
| Body joint FID | 待评，当前InterCLIP不可代替 |
| Hands joint FID | 不适用 |
| 滑步，GT-contact transitions | 12.348529mm/frame，GT4.258247 |
| Contact F1 | 几何代理 .632373；semantic/mesh F1待评 |

坐标误差按有效person-frame汇总、无root/rigid对齐；91450 person-frames。接触代理为两人22×22 joints最小距<=.1m的逐帧F1，不能称语义/mesh接触。共享scorer76738、GPU export8291/child10962、geometry42272、paired29367均exit0，不重跑；GPU峰值.127GiB且全部云端。恒等位置最大差.004701mm、feature66:258最大差1.669e-6、接触365800值无不匹配；冻结权重保持。

| 对5k的candidate-control，480 identifier cluster bootstrap10000次/seed20261016 | 变化 | 95% CI |
|---|---:|---|
| Full MPJPE mm | -103.742207 | [-148.171559,-67.934919] |
| Root mm | -101.631587 | [-146.981814,-65.263782] |
| Sliding mm/frame | -1.347584 | [-1.614712,-1.091691] |
| Geometry contact F1 | +.007863 | [-.032725,.053766] |

位置与滑步改善有固定模型验证集配对证据，contact区间跨0。区间不包含训练seed不确定性、文本生成质量或SOTA结论。新增compare_hhc_vq5000_35000_cpu.py只在独立版本放开旧decoder同base约束，并严格断言fresh5000/35000与中间scope；数学计算沿用已核对paired聚合。5份JSON12151B大小/SHA逐一与云一致；acceptance.json为CPU ledger验收，29绑定源不变。H5、embeddings、weights全留云，无本机新增下载。

训练覆盖另经CPU只读motion30.shape与caption行数核对：6022 train ID，缺失1；rawN<15为384，15–29为930，30–63为1444，64–299为2904，300–999为219，>=1000为140。当前build_tokens只保留30<=N<1000的4567，排除1454条长度及1条缺失；caption非空18057行/空白6行，无缺失空文件。TokenData还会最终tokens[:75]，长序列token尾部覆盖需单独核对。官方InterMask min15和max300是处理后motion/window单位，不能与原始N直接同列。该发现支持后续独立数据覆盖消融，当前不改29绑定源/缓存或追加训练分支，不宣称能提升FID。

06:36低频监控VQ49801、latest完整45000/FID2.940790、best35000/2.827235；InterGen23950/best20000/24.314143；原六PID活跃，ABC/text-only/v2prior仍原等待态、phase空。继续完整VQ300k→原ABC40k→真实zero/train-only-shape验证选模→20rep正式test/MM，目标仍active，尚无HHC生成SOTA证据。

## 06:23 SGT：fresh VQ45000完整训练内验证

seed123/native fixed_val_first480_center_max300_raw_smpl_v1/480：latest45000 FID2.9407902419，best35000/2.8272352923；Matching3.8094565074、R .3583333333/.5333333333/.6166666667、Div7.7354021072/GT7.7157673836。Body native22含root108.947519296mm，较40k108.307变差，FID也回升；检索R3回退。实际06:22训练45801/update46800；原InterGen23500/best20000 FID24.314143、六PID live/GPU90%/8533MiB，三个queue继续等待。不是生成test、joint FID或SOTA。348B JSON SHA84c08671bcc403334ec6ba5beb5a0986f6747e1bae1299f1f0b51c642b6b4a4e与云一致。

| VQ十项，step45000 | 状态/单位 |
|---|---|
| Joint coordinate RMSE | 待统一评测 |
| Full MPJPE | 待统一评测 |
| Body non-root MPJPE | 待统一评测；native22含root监控108.947519mm |
| Hands MPJPE | 不适用，无手指关节 |
| Root误差 | 待统一评测 |
| Full joint FID | 待评 |
| Body joint FID | 待评 |
| Hands joint FID | 不适用 |
| 滑步 | 待评 |
| Contact F1 | 待评 |

## 06:20 SGT：三基线正式入口与真实云GPU采样验收

三公开基线使用独立native-length H5/scorer，Mask严格保留floor(L/4)*4，Gen/TI保留L；不补帧改变公开方法。新正式export必须同时有真实HHC验证选模receipt和本方法实际CPU_final验收，模型/采样/源码catalogue/bridge一致。三CPU_final实际加载总exec83084和各child exit0。Mask原构造器硬编码.cuda，首CPU-hidden尝试96723 exit1；CPU检查临时重定向该分配后严格加载通过，官方源未改，真实GPU未patch。Mask迁移源码无.git，按已验transformer/options及VQ四源SHA与27个Python catalogue固定，不虚构git commit；Gen/TI官方git固定。最终exporter SHA0d88373666edf68692ad688a378bf7585d6f19fad343baab0025a49d8699d6ce。

原GPU exec45990整体及三child exit0，在原云4090顺序只生成验证集first/min/max，9case每seed两次bitwise相同、finite，外层CPU/CUDA RNG保持；只读caption/length，无GT motion/shape/root/prefix，不保存motion/ckpt，无test请求和质量评分。

| 方法 | 采样 | 请求14/19/299对应输出长度 | 单进程GPU峰值GiB |
|---|---|---|---:|
| InterGen | DDIM50/CFG3.5 | 14/19/299 | 1.2001 |
| TIMotion Transformer+LPA | DDIM50/CFG3.5 | 14/19/299 | .8206 |
| InterMask | 20步/CFG2/topk .9/T1 | 12/16/296 | .9136 |

CPU协议exec54194 exit0、33项：40份实际冻结请求SHA/count，三方法main1056/MM100×30合成H5的长度/order/seed/padding，错长/错seed/GT条件/重放拒绝。loader后续兼容修复未动被测I/O/RNG函数AST，原813b...源快照保留。新scorer对既有真实Mask480 cache回放exec80877 exit0：GT/text embedding maxdiff0，gen maxdiff2.98e-7，FID delta -2.5984e-7，其余Matching/R/Div一致；只是float32 raw-normalized往返误差，不是生成收益，既有FID8.931384253不改。云真实参考文件名为validation480_metrics_v2.json。

新`aggregate_interhuman_text_only_comparison_cpu.py`适配HHC新shared-main/fair-MM及三native score：四方法各20main+20MM、实际val选模receipt、冻结请求/模型/采样/evaluator一致，非Gen方法GT/text对Gen严格相同、GT Div种子一致；输出1.96×population std/√20和10,000次paired replication bootstrap。这些是重复生成不确定性，不等于多训练seed或新数据泛化；不自动宣布全球SOTA。CoSh论文4.801仍单列待匹配资产，禁止借用InterGen补行。

正式输出H5量大时，新增云端显式消费工具：完整正式score/embedding/H5 SHA验收、持久回执后，仅unlink本任务output单个H5，保留score/embedding与模型/源码/seed。聚合区分“当前H5重算SHA”与“消费前验SHA、原H5已删除”。HHC现有MM scorer无embedding archive，则拒绝回收其H5。exec4788 exit0、16项合成集成检查包含完整160份score join/mean/CI/paired差、缺项/改身份/错评测器/错reference/CoSh别名/缺H5回执/消费后改score/validation拒绝及HHC MM保护；所有fixture已删除，真实生成文件删除0。本轮未执行真实回收。

9份CPU/GPU/parity/审计小JSON大小/SHA对云一致、29绑定源保持，acceptance在`output/hhc_sota_20261016/native_baseline_formal_acceptance.json`。finalVQ spec仍不存在，尚无真实formal选模receipt/正式生成/比较成绩；所有GPU工作只在云，本机只收小JSON/源码，无checkpoint/motion H5/prior NPZ下载。

## 06:11 SGT：VQ40000完整验证，FID回升

seed123/native fixed_val_first480_center_max300_raw_smpl_v1/480：最新40000 FID2.9247688863、Matching3.8106952667、R .3416666667/.5166666667/.625、Div7.7540802956/GT7.7157673836、native22含root Body108.3070989527mm。best仍35000/FID2.827235；Body改善但FID/Matching/R1变差。实际43601/update44600；原InterGen23150/best20000 FID24.314143，六PID live/GPU89%/8533MiB、队列等待。331B JSON SHA35e87b53a4650b0e13052c63ed6d4f747334b7b2e0bee6f8df80676fb2f1473e与云一致；不是生成test或SOTA。

| VQ十项，step40000 | 状态/单位 |
|---|---|
| Joint coordinate RMSE | 待统一评测 |
| Full MPJPE | 待统一评测 |
| Body non-root MPJPE | 待统一评测；native22含root监控108.307099mm |
| Hands MPJPE | 不适用，无手指关节 |
| Root误差 | 待统一评测 |
| Full joint FID | 待评 |
| Body joint FID | 待评 |
| Hands joint FID | 不适用 |
| 滑步 | 待评 |
| Contact F1 | 待评 |

## 05:48 SGT：fresh VQ35000完整训练内验证

step35000、seed123、480/native center/max300协议：FID2.827235292340447、Matching3.8104145685831705、R .3479166667/.5020833333/.6229166667、Div7.7094631195/GT7.7157673836，native22含root Body109.0428247114mm。比30k FID2.870298/Body110.311改善，R1由.370833降至.347917；该变化不能解释为整体生成获益。manifest实际main37001/update38000、running、best35000，完成artifact与云SHA0769af45c850c07d0a318b656b0e1e06afe0a771cde885d6e123397ba8ec7f74一致。只是训练内重建监控，不能和三方法生成FID或joint FID混排。29绑定源保持，零新增本机权重。

| VQ十项，step35000 | 状态/单位 |
|---|---|
| Joint coordinate RMSE | 待统一评测 |
| Full MPJPE | 待统一评测 |
| Body non-root MPJPE | 待统一评测；native22含root监控109.042825mm |
| Hands MPJPE | 不适用，无手指关节 |
| Root误差 | 待统一评测 |
| Full joint FID | 待评 |
| Body joint FID | 待评 |
| Hands joint FID | 不适用 |
| 滑步 | 待评 |
| Contact F1 | 待评 |

## 05:46 SGT：正式text-only入口与后评测队列集成修复

正式主/MM已有共享评分器，独立新入口补齐生成与验证选模门限；旧训练`--formal-test`仍带requested GT shape，不能替代它。新contract要求A/B/C zero及A train_mean/empirical_train_pair这五个预登记final40000候选的真实480val全部完成，核对实际H5/模型/VQ文件SHA、相同VQ、官方evaluator、与基线GT/text embedding逐项相同，再按最小val FID、固定名称破同分冻结。该选择规则不代表所有指标获益或SOTA；尚无真实receipt，未提前生成test或以test调参。

新exporter只接受冻结receipt，先CPU验源/文件/分数/请求绑定，再检查唯一云GPU及13GiB余量后加载模型。纯text-only构造器直接传zero或训练先验betas，不读GT H5；MGT条件、VQ、FK一致。主20×1056；MM20×100×30。MM每prompt保持与主路径相同的训练体型draw（shape replication0），30次仅motion noise变化；这避免把改变体型条件混入动作多样性。样本/H5固定顺序、length/padding、repeat seed及新增shape assignment digest都有校验；partial/已存在目标不重放。

云CPU `audit_hhc_text_only_formal_cpu.py`原exec1613终态exit0，18项通过。真实请求rep0主1056/MM3000合成生成tensor写H5，经原正式validators接受；shape draw不改全局NumPy RNG、请求重排不改结果，GT-shape/test分数/缺候选/源码漂移/receipt换winner/重复导出均拒绝。合成小临时文件已删除；**没有真实MGT模型加载/生成、没有真实选模receipt，更没有test/MM质量结果**。后续必须进行真实云模型端到端验收。验收JSON1816B/SHA2469de19cc498d7386b8dd0126eeecc8b5e678388ce5e7f717a1204e17ed3401。

集成时发现此前train-prior队列9549直接索引`generation_metadata['shape_policy']`和MGT/VQ SHA，而两个MGT val exporter把这些写入H5 `attrs.metadata` JSON；共享CPU scorer保留H5 attrs结构。原9项mock没有走真实nested布局的完整成功链，因此会在真实val后KeyError。独立v2修复flat/nested解码、冲突拒绝，不修改原28绑定源；新14项云CPU mock增加全部4phase成功与checkpoint不符停后续，exit0。此修复没有改变生成算法或评测数值。

05:44:47守卫实查旧9549命令、waiting状态、空phase、无child/输出后，SIGTERM并确认/proc消失；旧STATUS标记superseded_before_any_child。旧实际退出码未获wait状态，不声称143。新driver9975确认live/waiting，control `output/hhc_train_shape_val_v2_control_20261016`，配置`migration/control/hhc-train-shape-validation-v2.json` SHA a7494e3ef9d4135b8c78c23064470df79906125683b57804560c072d3d8b3857，29源=原28+v2。2353/5690/5729/7178/8321命令前后逐项相同；原源码/训练均保留。handoff JSON1615B/SHA6cd56eb981daa1306851b6b2a99491843eab0e58fba2f80210cc124e0eb6bd90，v2 mock JSON1042B/SHAb96bbda24effceba54b43f35ba38e9806f631ba622bcdd6aa8aa81b015df1dfe。不能重启旧9549。hourly automation已app更新/读回新句柄与29源，原schedule/target保持。

05:42普通监控：VQ34801、最新完整/best30000重建FID2.870298；InterGen22250、最新完整/best20000 FID24.314143；当时全部旧PID仍live，GPU29%/8533MiB。无新完整35k结果。所有GPU工作继续仅在云；本机只收上述小JSON和源码，不下载模型/H5/先验NPZ。

## 05:17 SGT：体型条件分布差异与train-only推理候选

原生MGT训练按各clip的GT betas提供shape条件；公平text-only主评测固定zero betas同时供MGT/VQ/FK。zero不是训练时隐藏shape embedding的同一操作，且beta=0可能偏离实际训练体型中心，因此先统计训练支持、再登记不读取验证GT shape的推理对照，保持网络/码本/采样不变。

云CPU builder exec84936 exit0，完全复用`build_tokens`的train split顺序、motion30存在、30<=L<1000与非空文本门限：4567行、1556种不同配对betas，排除长度1454/缺motion1；没有读取val/test beta或motion数组。按逐维train std缩放的到训练均值欧氏距离：zero9.04293，训练median3.87615/p95=6.57825，zero距值的经验分位98.423%。该距离忽略协方差，不等同密度、严格OOD判定或生成质量结论。

先验保持原始两人配对及person顺序，interaction-uniform取样，避免分别抽人破坏配对统计。云端NPZ145147B/SHA4c5d547d9602abc3fd5b361c4689a3d5ba10207a1022994515e9037b802cfd64，spec SHAaa2a9d7678216b345295629c36cbd8e06cf438f33b2569de764b683eb4c73a98；本机只收小JSON，没有下载NPZ/ckpt。统计及数组在看到新MGT生成结果前冻结。

| A_historical final40000验证因素 | 体型来源 | 验证集GT beta | 控制 |
|---|---|---|---|
| zero（原主参考） | 两人各10维全0 | 不用 | 已在8321排队 |
| train_mean | 上述训练4567对的固定均值 | 不用 | 相同MGT/VQ/采样随机种子 |
| empirical_train_pair | 上述训练4567对均匀抽一对 | 不用 | 独立shape RNG、不改token RNG |

经验抽样seed为`sha256(hhc-trainshape-v1:sample_seed:replication)`前8字节小端，再用NumPy PCG64均匀取行；单次val固定replication0，不用caption、root或GT motion/beta。480请求中抽到457个不同row index（不等于457种不同betas），顺序/批次分块不影响结果。未来MM应先明确固定或重抽latent shape的条件协议，本次未做MM，也未用test选形状策略。

独立exporter继承原`StreamingScorer.generate`原函数，构造阶段直接传训练先验beta字典，禁止H5读取仍通过真实CPU FK；MGT条件、VQ decode、FK/archive/body conversion用同一组betas。生成主循环AST与原zero/requested exporter逐项相同，原23锁定源码未改。CPU策略审计exit0，覆盖480请求只取id/seed、train/val ID不交、全局RNG不变、重排序/分块不变、data/cache来源不符拒绝、重复identifier拒绝；不是完整MGT质量测试。

新driver9549真实/proc和STATUS已验收为`waiting_for_zero_shape_scoring`，配置`migration/control/hhc-train-shape-validation.json`绑定28源。等待原8321的12个GPU生成/CPU评分phase全部exit0，且有13GiB显存余量，才顺序运行A的两种train-only shape导出及共享CPU评分；严格核对与原A_zero的MGT/VQ SHA及三基线GT/text embeddings相同。9项CPU mock验收包括上游活跃等待、死亡/失败/配置不符、完整phase、显存/多卡、child exit9停后续、已有queue拒绝重放，均未启动真实GPU child。候选无收益断言、不晋升B/C、不提前formal test；真实FID/R/Div及后续物理质量决定下一步。源码/原始小结果见`output/hhc_sota_20261016/train_shape_*`。

## 05:25 SGT：fresh VQ主步30000训练内验证

seed123、固定480/center/max300/native raw协议，最新完整及best30000：InterCLIP重建FID2.870298、Matching3.812147、R1/2/3 .370833/.500000/.600000、Div7.735841/GT7.715767。前一完整25k是FID3.143828、Body115.773666mm、Matching3.811189、R .358333/.497917/.610417、Div7.730003；当前Body110.311079mm改善，但R3与Matching未改善，不声称全指标胜出。原观察exec10152确认5729真实live后等30000完整artifact，终态exit0。不同请求/crop协议，不能与上面的官方冻结480重建或生成基线混排。

| VQ十项，step30000 | 状态/单位 |
|---|---|
| Joint coordinate RMSE | 待统一评测 |
| Full MPJPE | 待统一评测 |
| Body non-root MPJPE | 待统一评测；现有原生22joint含root监控110.311079mm |
| Hands MPJPE | 不适用，无手指关节 |
| Root误差 | 待统一评测 |
| Full joint FID | 待评 |
| Body joint FID | 待评 |
| Hands joint FID | 不适用 |
| 滑步 | 待评 |
| Contact F1 | 待评 |

这里的InterCLIP FID不是joint FID。05:21普通只读复核原InterGen scratch21600、最新完整/best20000 FID24.314143；2353/5690/5729/7178/8321均live；05:29单独复核9549真实live/waiting。未重启/暂停。仍缺真实新HHC生成及独立test/MM/CI，SOTA未达成。

### 正式test/MM与存储验收边界

当前main manifest协议为20rep×1056/rep、batch96/drop_last，共21120次生成/模型；MM为20×100prompt×30repeat，共60000次生成/模型。MM manifest明确从每rep全1098有效prompt中选100，非从drop_last后的1056中选；它绑定的main manifest SHA与当前本机副本一致。仅核对JSON元数据，没有计算test分数、读取GT beta用于策略或以test调参。

现有`train_hhc_smpl_cfg_ablation.py --formal-test`仍以requested GT betas作shape条件；当前独立zero/prior exporter仅支持val。因此正式pure text-only入口、选定模型/shape/sampling协议与MM latent-shape重复规则仍待真实val选模后冻结实现，尤其要处理同identifier的多次MM生成，不能直接调用当前prior请求字典并绕过重复检查。不能把旧GT-shape CLI成绩、val或重建FID当作公平正式SOTA。所有28活跃绑定源保持不变，未来正式入口应独立版本化。

Runpod live read确认pod RUNNING，persistent mount `/workspace`配置80GB；`du -sx --block-size=1 /workspace`实测22831495680B，HHC目录约19GiB。共享文件系统df的487TB不是本pod配额；未扩大资源或删除数据。源/模型恢复包都留云，不把持久卷当成已验收的外部checkpoint备份。证据`output/hhc_sota_20261016/vq30000_storage_and_formal_readiness.json`记录以上范围和未完成项。

**04:50 SGT 当前结论：** 三个公开权重基线已在同一480val冻结请求下重跑；新的HHC原生VQ已训练至20601，最新训练内重建FID3.244720/best20000。decoder-only root geodesic与canonical-pair两项短pilot均未达到综合晋升条件；后者位置精度改善但FID、滑步、接触恶化，详细数据见下文新增验收。最终300k VQ后的A/B/C40k生成对照及无GT shape的统一后评测均已排队，尚无新HHC正式生成test/MM结果。下面早期“尚未开始”段落是历史阶段记录，以本段及最新验收为准。

## 04:50 SGT：canonical-pair位置损失有取舍，不晋升

可微loss按官方实际代码使用hips[2,1]朝向、独立floor、锚点XZ和共同yaw，双anchor等权；不改evaluator，不在推理时读取GT。全480 GT/pred×两anchor×两swap的3840组合与官方处理最大位置差.02199mm，方向导数误差5.98e-12。由此只证明实现一致；真实质量由下述独立导出和原版InterCLIP衡量。

等预算对照复用上一轮已完成control：base VQ5000冻结encoder/codebooks/shape condition，decoder9,331,982参数、1000更新、batch256/window64/FP32、AdamW原参数、seed20261016，所有训练批次SHA均为16f0a33ffe6f8bff57de0469c88932373653ef28cbcd334f5310d54d067f823d。唯一新增3×双anchor canonical joint SmoothL1；保留原全部损失。训练exec20986、导出38366、共享评分28897、geometry85711、paired bootstrap76336全部exit0；GT/text embedding同三个公开基线逐项一致。选模范围仍是早期VQ的重建实验，非text-only生成。

| 官方冻结480val重建 | 等预算decoder control | canonical joint loss +3 |
|---|---:|---:|
| InterCLIP FID↓ | 4.201122 | 4.312691 |
| Matching↓ | 3.811854 | 3.809505 |
| R1/R2/R3↑ | .36250/.53750/.63125 | .36250/.53958/.62917 |
| Div / GT Div | 7.667987/7.741262 | 7.608159/7.741262 |
| Joint coordinate RMSE mm↓ | 333.324 | 270.973 |
| Full MPJPE mm↓，22含root | 358.891 | 306.399 |
| Body non-root MPJPE mm↓ | 359.858 | 307.411 |
| Hands MPJPE | 不适用 | 不适用 |
| Root error mm↓ | 338.587 | 285.135 |
| Full joint FID | 待评 | 待评 |
| Body joint FID | 待评 | 待评 |
| Hands joint FID | 不适用 | 不适用 |
| 滑步 mm/frame↓，GT4.258 | 13.771 | 14.916 |
| geometry Contact F1↑ | .600663 | .564069 |

semantic/mesh contact F1待评，表内geometry为阈值.1m的22×22跨人最近关节伪标签；InterCLIP FID不替代Full/Body joint FID。滑步分母固定为257461个GT-contact脚转换，不能靠抬脚缩小分母。

480 identifier聚类、10000次配对bootstrap/seed20261016，candidate-control差值：Full MPJPE −52.492mm，95%CI[−83.601,−23.107]；root −53.452mm，[−84.959,−23.408]；contact F1 −.036594，[−.074991,−.002902]；滑步 +1.144870mm/frame，[.985732,1.328863]。区间只反映这些固定模型的val抽样，不含训练seed、生成20rep或test不确定性。

**决策：不晋升主VQ/decoder或MGT，不扩跑这版损失。** 位置误差下降并没有转化成FID与物理质量共同改善。保留canonical可微坐标及因果诊断证据，重点推进已登记的文本CFG/window生成对照；目前不凭少数指标继续尝试随意增大损失权重。所有23项排队源码绑定保持不变。云端新decoder SHA7a8b23e427c7dd6cfa9d94a9dfc9387a39ae9ebdcb3298b1367d2bbaea72b798，重建H5 SHA7c638cf7c56e9e44399ad948c9b94f024263e1ee4b681d6fb149e32ef11512ca；仅小JSON取回，本机未下载权重/H5。机器验收见`output/hhc_sota_20261016/decoder_canonical_pilot/acceptance.json`。

04:49低频状态：fresh VQ20601，最新完整及best20000/FID3.244720、Matching3.813437、R .372917/.493750/.614583、Div7.738170/GT7.715767，原生监控Body22含root112.941mm。协议`fixed_val_first480_center_max300_raw_smpl_v1`，与上表官方冻结480不同；Full/Body non-root MPJPE、RMSE、root、Full/Body joint FID、滑步/接触均待统一评测，Hands两项不适用，不能借训练内FID排生成排名。原InterGen scratch20600/最新完整及best20000 FID24.314143；两个原训练和7178/8321队列均活跃，无重启/新增pod。

## 本轮已完成

1. 审阅既有 evaluator、旋转校正与 HHC 正式结果记录；原服务器 SSH 本轮连接超时。旧诊断的 JSON/embedding 大多尚未迁入本机，不能把文档记录说成本轮重新评分。
2. 本机 CPU 完成 32 条冻结验证请求 × 4 个方向 × 2 个能量级 × 5 类扰动。原版 InterCLIP motion encoder 严格加载，源码/配置/完整 checkpoint SHA256 校验通过。**未使用本地 GPU**。
3. 确认 HHC 的文本 null 条件存在训练/采样差异；独立 CFG 单因素入口与 CPU 路由/梯度检查完成。没有真实训练新 HHC，尚无新 HHC FID。
4. 按用户新要求关闭所有自动 checkpoint 本地下载；GPU 工作仅云端。旧下载器已停止，Runpod controller 仅保留云训练状态/恢复监控，已加载新政策。已有历史文件未删除。

## 新 evaluator 实测：不能直接沿用“旋转最敏感”的解释

结果：`output/hhc_sota_20261016/legal_sensitivity32_gaussian_control/report.json`；汇总：`sensitivity_summary.json`。

从既有 480val 中预先等间距选择 32 个请求，回放 caption/crop/anchor/person order，使用每条全部有效帧（14–271帧），不读取 test。各组每条固定 `||delta/std||₂ = epsilon sqrt(T×2×66)`；P/V 用平滑随机方向，SO(3) 用合法轴角左乘局部旋转，另设非流形 6D Gaussian 对照。P+V 组同步重算位置变化的差分；其能量在 P/V 联合空间归一，不能当作与 P-only 同位置振幅的对照。

| epsilon | P | V | P+一致V | 合法SO(3)旋转 | 6D Gaussian旋转 |
|---|---:|---:|---:|---:|---:|
| .02 | .00167574 | .00008127 | .00024742 | .00077756 | .00122548 |
| .10 | .00837700 | .00040644 | .00123773 | .00388947 | .00614997 |

单位为 **unit-normalized motion embedding L2**，不是 FID 或 MM Distance。epsilon=.02 时合法旋转/P 响应比 .4640，按动作聚类的 10000 次配对 bootstrap 95% 区间 [.4192,.5121]；Gaussian旋转/合法旋转为1.5760，[1.4882,1.6660]。epsilon=.10 方向相同。4个扰动方向先在同动作内平均，没有把128次观测当128个独立动作。contact+100 的 embedding 差仍精确为0。

旧4条/64帧Gaussian探针给出旋转响应略大于位置；本轮同时改变样本、长度和扰动族，不能把两轮差异全部归因于 SO(3)。本轮同样本 Gaussian 对照说明旋转响应确实依赖扰动方式。P-only会破坏P/V一致性，合法旋转仍未保证与P或mesh一致。因此这些结果**不能**直接给出最优loss权重，也不能解释真实模型FID差距。

CPU进程62505：四类初版，exit0；37363：加入Gaussian对照的完整五类，exit0。完整五类才作为本轮主探针。源码、请求、模型hash与逐样本扰动能量均已保存。

## 已有 HHC 与四类基线

以下是 InterHuman **text-to-two-person generation**。论文报告值与HHC历史记录分开标注；不是本轮同请求统一重跑的排名。

| 方法/版本 | FID↓ | R@1↑ | R@2↑ | R@3↑ | MM Dist↓ | Div | MModality↑ | 证据 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| InterGen | 5.918±.079 | .371±.010 | .515±.012 | .624±.010 | 5.108±.014 | 7.387±.029 | 2.141±.063 | 论文 |
| InterMask | 5.154±.061 | .449±.004 | .599±.005 | .683±.004 | 3.790±.002 | 7.944±.033 | 1.737±.020 | 论文 |
| TIMotion Transformer | 5.433±.080 | .491±.005 | .648±.004 | .724±.004 | 3.775±.001 | 8.032±.030 | .952±.032 | 作者v3 |
| TIMotion RWKV | 4.702±.069 | .501±.005 | .656±.006 | .734±.006 | 3.769±.001 | 7.943±.034 | 1.005±.020 | 作者v3 |
| CoShMDM | 4.801±.031 | .451±.001 | .598±.002 | .691±.003 | 3.763±.004 | 7.946±.002 | 3.051±.019 | 作者项目表 |
| DHVAE | 5.015±.085 | .496±.004 | .647±.006 | .720±.005 | 3.772±.002 | 7.952±.045 | .804±.030 | 2026作者v1 |
| Text2Interact / InterActor | 5.191±.055 | .483±.005 | .638±.005 | .717±.005 | 3.778±.001 | 7.900±.030 | 1.051±.031 | ICLR2026，额外合成数据微调 |
| HHC SMPL+shape，VQ185k/MGT40k | 8.701897±.108161 | .320218±.005517 | .447585±.005209 | .523153±.004880 | 3.853481±.001380 | 7.831259±.025797 | .849837±.038988 | 10-07记录，20rep |

来源：[InterMask Table1](https://arxiv.org/html/2410.10010v2#S4)、[TIMotion v3 Table1](https://arxiv.org/html/2408.17135v3)、[CoShMDM原表](https://alimanjotho.github.io/coshmdm/images-paper/fig-quant-eval-interhuman.png)。CoShMDM表已另存259143B PNG并视觉核对，物理MMPE/CSim单位尚未逐项核实，因此不混入本地物理指标排名。

HHC来源为`docs/PROGRESS.md`的10-07完成记录和`docs/interhuman_smpl_mgt_20261005.md`的协议。HHC用请求真实betas作条件；InterGen/InterMask/TIMotion不具有相同shape输入，CoShMDM也需核对其shape/contact条件。HHC GTDiv记录7.788421，论文GTDiv7.948，抽样/请求不完全相同。正式比较需在同一冻结request/evaluator/批次分组下重跑。论文InterMask主指标20rep、MM5rep；本地HHC使用MM20rep，必须披露。

四类指定方法中，已核实的最佳FID为TIMotion RWKV 4.702；最佳MM Dist为CoShMDM 3.763。该结论**不等于2026年全部方法的全球SOTA**。[DHVAE原表](https://arxiv.org/html/2603.00144v1#S4.SS1)本轮已核查，该文重跑公开TIMotion得到5.600±.106，不能替代原论文RWKV分项。[Text2Interact Table1/2](https://proceedings.iclr.cc/paper_files/paper/2026/file/a99f732df9b668284b449da0214a3286-Paper-Conference.pdf)的5.191来自额外合成数据微调，条件需披露。

[HINT Table1](https://arxiv.org/html/2601.20383v1#S4)报告online AR FID3.100±.035、R3 .672±.004、Matching3.796±.001、Div7.898±.023。方法明确H=4/K=16及历史窗口条件，但文本未明确推理首窗口是否由真实起始帧初始化；不能猜测GT条件，未核验实现前不纳入同条件排名。4.702不是充分的全球SOTA验收线，需继续核实online/offline及额外条件。不能只击败旧InterGen就声称SOTA，也不能把reaction-generation或GT行FID .273误作本任务生成基线。

## 首个改进候选：匹配文本CFG的训练null条件

现有`SMPLTransBase.forward`在historical训练中只对global CLIP调用`mask_shape(.2)`，逐词cross-attention始终有真实文本；生成则切换paper_text_cfg，同时去掉word/global文本。无条件CFG路径缺少对应训练输入，是明确的结构差异；是否造成指标差距仍是假设。

独立代码：`models/mgt_interhuman_smpl_cfg.py`、`tools/train_hhc_smpl_cfg_ablation.py`。未修改原HHC入口、原模型或正在云端运行的InterGen训练源码。

| 对照因素 | historical | matched |
|---|---|---|
| 训练文本dropout | .2，仅global | .2，word/global同一mask |
| 独立shape dropout | .2 | .2 |
| 采样文本null | word/global均去掉 | 相同 |
| 采样shape | 保留 | 保留 |
| VQ/网络/CE/masking/采样 | 原设置 | 相同 |

先做同seed114514、相同初始化/静态token cache/有效batch64/AdamW/原all-valid CE的40k双臂pilot；每10k完整480val和固定MM。VQ185k冻结，仍Root→Body/50步/T.5/CFG3.5。只按val选模，pilot默认不读取test；训练完成后需明确`--formal-test`才执行20rep。后续seed114515/114516只改变训练抽样/初始化，tokenizer cache种子固定，不能声称独立VQ训练。

CPU检查使用真实HHC拓扑的tiny随机模型和假VQ码表，仅验证路由与梯度：旧训练forced-drop仍有word文本，matched两条文本路径均为0；shape独立drop保留；采样null与文本无关且shape仍有效；4096例joint-drop实测.212890625；state_dict不变，exit0。**这不是训练收益测试，也不是加载真实VQ后的端到端smoke**。

## 第二候选：长序列随机token窗口

云CPU只读native H5得到6021个有效train group，另1个缺motion group：<15帧384条，15–29帧930条，30–999帧4567条，>=1000帧140条。原build_tokens按长度保留4567条（75.85%），文本空行未逐项复核。固定seed114514的cache crop没有300帧上限；保留组中210条crop超过75 token。TokenData固定tokens[:75]，随机drop-first最多使原始index75偶尔可见，更后位置不可访问。原审计`output/hhc_sota_20261016_training_length_coverage.json`不是质量评测。

`dataset/interhuman_token_windows.py`新增historical/uniform取样；新训练入口`--token-window historical|uniform`默认historical。uniform仅在原first/last-token augmentation后长度>75时均匀取75token窗口，窗口RNG独立；不扩展source长度门限，不改变码本/短序列。1024种子、真实TokenData对照CPU检查exit0：historical逐项相同，short完全相同，caption/长度/beta/EOS保留，重启确定性一致，180token例覆盖106个起点0..105。没有训练权重/FID。

预注册A=historical CFG+historical window；B=matched CFG+historical window；C=historical CFG+uniform window。先分开比较B-A、C-A，再组合有证据的因素/多seed。930条短动作及140条极长动作的门限恢复仍是后续独立因素，当前没有暗中加入。

## 本轮统一480val实测（03:06 SGT，三种基线完成）

各方法保留原生采样预算：InterMask20步/CFG2/top-k .9/T1，InterGen与TIMotion DDIM50/CFG3.5；请求、输入变换和评测器统一，推理成本并非相同。后续比较单独披露实际耗时。

| 方法 | FID↓ | Matching↓ | R1 / R2 / R3↑ | Div / GT Div |
|---|---:|---:|---|---|
| InterMask official，原生长度floor(L/4)*4 | 8.931384 | 3.795384 | .443750 / .612500 / .685417 | 8.127659 / 7.741262 |
| InterGen official，原始请求长度L | 8.762189 | 3.794905 | .425000 / .591667 / .672917 | 7.934211 / 7.741262 |
| TIMotion public Transformer+LPA，长度L | 8.721812 | 3.780101 | .460417 / .639583 / .722917 | 8.080552 / 7.741262 |

TIMotion generation/score均exit0。三者comparability审计exit0：请求内容、官方evaluator provenance一致，全部GT/text embedding逐元素完全相同（max abs error=0）。InterMask取整造成0/1/2/3帧差分别97/121/114/148条，平均1.6521帧；另两者均480条长度相同。收据`output/hhc_sota_20261016/comparability_three_methods.json`，各cache/embedding/report均SHA绑定。InterGen早期runner的initial source快照早于Path修复；子评分report保存真正执行脚本SHA51b4dbdce2b2982be3a1764454d16f59cae20479627f476edbb8ad51fca118db，不把父快照当作已执行版本。TIMotion新版runner另保存每阶段source hash。

三者batch1生成loop墙钟时间分别InterGen713.01秒、InterMask477.46秒、TIMotion386.02秒；均单4090与既有InterGen长训练共卡，排除初始化，不可作独占推理速度排名。按原96候选池保留分组的长度诊断已保存：<30帧103条，30–99帧225条，>=100帧152条；没有重组检索候选池。该分组仅作诊断，不是长度覆盖造成性能差异的因果证据，不估小子集FID。

InterGen generation/score均exit0，报告`output/hhc_sota_20261016/intergen_validation480_metrics.json`。两项均是官方发布权重本轮运行，不等同论文旧运行；尤其InterGen的Matching与论文5.108不同，应保留这个差异，不能挑其中有利值合并。480样本/512维特征的单次FID没有正式20rep CI，当前数值不能宣称统计显著或SOTA。InterMask生成短0–3帧，属于已披露的原生长度边界；正式同长度协议若改为ceil解码再裁L，必须先独立冻结协议、两臂一致，不能静默替换当前结果。

冻结480val/request内容SHA751efc9ced91e2a6ad4eec4a7b9316946761fe13f86d09af7d5c04a9226b7f93，seed20261006，raw262D，官方CPU InterCLIP，batch96检索，无特征干预，MM待评。报告`output/hhc_sota_20261016/intermask_validation480_metrics.json`。N=480单次val FID不能与论文/test20rep或HHC历史直接排序。初次在全部embedding完成后，写provenance时Path/str错误exit1；保留失败文件，修复后原cache不变、CPU独立重算v2 exit0。后续脚本hash在导入时冻结，防止运行中文件更新误绑provenance。

## 02:36 SGT 云端资产恢复与评测兼容验收

原生数据已直接在Runpod由固定raw重建：7779文件、7777有效pair，3945/4106为空；62232个原始数组逐项bitwise还原。派生特征body MPJPE均值 .0000789mm，max joint .000901mm，rotation max5.96e-7，online position max5.49e-5m，contact差8/8980112；不同runtime下的派生特征不称bitwise。manifest SHA256 `7468c59cbe010e438227ec1a13e30d1fd9a03a8fc1e8c9d0d4bfb971e02c5dc5`，motions.h5 SHA256 `16655a9a91f392c6a54579d447e79cac06a46c439c78685f598df5c2d07acaec`。完整状态在云端`output/hhc_native_restore_control_20261016/full_status.json`，builder exit0。

原valMM生成器恢复100prompt×11repeat，内容SHA `b9c2f30997389ddc263edf8c8c7aa9af4cac8ac2e4607daa114c072a300ace18`；绝对source路径使本机/云文件SHA不同。testMM由原固定种子/1098 eligible顺序重建20rep×100prompt×30repeat；与现有1056 main请求交集1908/2000条，所有共享字段一致，20个main请求文件SHA未变。旧MM文件缺失，**原文件字节身份未证**；新文件内容单独冻结，不能覆称找回原文件。

InterMask的VQ518555706B/SHA af57bbdde94ed4d4a01f3ba5f9fe24f0012c2a37b3e38c2fa57a2a91ebcc1272，MGT651546446B/SHA392ca8c9e42c2c23aeb97ba7b810e8fa4bdcc5f4839a5a909be2f56bf1698665，以及InterGen2701617525B/SHA bab341123c27af9d8c3ee7ebc43c2d341bfd0f4103425d06f8af6c717a086f8d已直接落云并核验，未收取本机。InterMask4条smoke exit0，480条完整val生成exit0、CPU评分运行中；InterGen同请求基线排队中。仅val，不将结果冒充论文20rep test。

实际云端PyTorch2.8默认weights_only变化导致InterCLIP加载失败。HHC wrapper改为先验证既定官方checkpoint SHA `4e14b8ee5c2d965778fe003fb37d98ed8e5c7e0940668515fc7dcfa2831c17f8`，再显式weights_only=False。207张量与文件完全一致、文本embedding有限、篡改在反序列化前拒绝，6项既有测试通过；官方网络、权重、特征和评分方程未变。收据`output/interhuman_baseline_restore_20261016/evaluator_compat_check.json`。

HHC VQ首次smoke在导入阶段因原Inter-X未用SMPL-X资源缺失而失败，未进入GPU。将资源导入移入原Inter-X和旧ReConsLoss构造函数后，原生SMPL仍用既定female SMPL/FK，网络/损失方程不变。v2读取3263条train motion、559094个原64帧窗口，已完成batch256/fp32四次真实GPU更新、梯度均有限：loss1.00620/1.07168/1.02091/1.02957，峰值allocated1.7913GiB，参数26789702，无checkpoint保存。报告`output/hhc_sota_20261016/cloud_native_vq_smoke.json`；仅验就绪，不是恢复185k或质量提升。RMSE、Full/Body/Hands MPJPE、Root误差、Full/Body/Hands joint FID、滑步、接触F1十项均待评；本原生SMPL无手指，未来hand项须标不可比。

TIMotion权重已云端获取1430359445B，首次获取SHA `f364605f7e67cc272dc31810f384ff0dedb564331442257bf776f8d7285ffcee`，无作者公开checksum。Transformer+LPA/5层16头512维的CPU严格加载294张量通过，无missing/unexpected。runner4780/GPU生成5047/CPU评分5263均已结束，generation/score exit0，完整480val结果见上表；MM与正式test尚未执行。

## TIMotion / CoShMDM 权重身份边界

[TIMotion固定commit README](https://github.com/AIGC-Explorer/TIMotion/blob/172580fd7f2551a7cc10fde1f15ff73315742e65/README.md)链接的公开权重ID为`15WE1kiITcrwva3QTGW86plgxA8zmufq4`，实时元数据1430359445B。公开网络为Transformer，不能以此复现论文RWKV4.702。评测脚本还覆盖YAML的配置：默认layers5、heads16、latent512、LPA显式开启；应由原评测默认值与checkpoint严格加载共同确认，不能直接用YAML的8层/1024维。作者未发布checksum，云端首次获取会冻结SHA并如实标注该局限。

[CoShMDM固定commit下载脚本](https://github.com/AliManjotho/coshmdm-code/blob/c5116ab3bead55051ffec2ec9a161a38d2ab5116/prepare/download_pretrain_model.sh)明确指向`1sYOJEdbB8hDI5pY1-X960njdFD2WcwFq`，与InterGen相同。因此该公开下载路径不能提供独立CoShMDM checkpoint复现证据。论文报告值保留在上表，公开代码运行值必须单独披露范围。已有Inter-X公共代码port也明确不等于论文VQ/shape/RL-SDF架构，不混作本轮InterHuman正式复现。

**05:03 SGT 当前公开发布的完整核查：** 官方main仍为c5116ab3，live remote仅main、无tag；[官方Releases](https://github.com/AliManjotho/coshmdm-code/releases)、[Issues](https://github.com/AliManjotho/coshmdm-code/issues)和项目页未找到备用独立checkpoint。这是所列官方范围内未发现，不能推出所有渠道均不存在。

在本机只读稀疏源码快照上运行审计exit0；InterGen参照的9个模型源、模型config、download script共11个git blob已逐一匹配官方固定commit b1750df。CoSh 9个models文件在CoShMDM→InterGen、bert_*→clip_*等显式标识符重命名后AST全部相同，其中blocks/cfg_sampler/gaussian_diffusion/layers/losses/nets/utils七个文件连字节都相同；config只改NAME和checkpoint路径。实际公开架构为262D直接动作扩散，未包含论文项目页的VQ latent/shape条件/RL-SDF模块。名称带bert的[公开文本模型](https://github.com/AliManjotho/coshmdm-code/blob/c5116ab3bead55051ffec2ec9a161a38d2ab5116/models/coshmdm.py)实际仍载入CLIP ViT-L/14@336px。

云CPU exec47681 exit0，对同下载ID已存在的2701617525B checkpoint（SHAbab341123c27af9d8c3ee7ebc43c2d341bfd0f4103425d06f8af6c717a086f8d）只做mmap/key审计，346 tensors，未完整构造/加载模型或推理：

| 原checkpoint前缀 | 公开CoSh源码要求 | 原键数 | 对应bert键数 |
|---|---|---:|---:|
| clip_transformer | bert_transformer | 144 | 0 |
| clipTransEncoder | bertTransEncoder | 24 | 0 |
| clip_ln | bert_ln | 2 | 0 |

总计170个重命名键不匹配，其中后两项是可训练文本层/LN的26键、11,029,504个参数元素。[公开infer](https://github.com/AliManjotho/coshmdm-code/blob/c5116ab3bead55051ffec2ec9a161a38d2ab5116/tools/infer.py)用strict=False且只去掉model.前缀，可能跳过这些训练参数；[公开eval](https://github.com/AliManjotho/coshmdm-code/blob/c5116ab3bead55051ffec2ec9a161a38d2ab5116/tools/eval.py)则strict=True。这里的命名不兼容由实际checkpoint keys与源码推得，没有冒称执行过完整load_state_dict。若显式改名后加载，得到的是同架构/同来源权重的InterGen，不构成独立论文CoSh复现。**不以忽略缺键、复制InterGen指标或补写未发布模块的方式填第四行。** CoSh论文4.801仍作已核对文献比较；统一复现待匹配的独立架构/权重证据。JSON原始证据及验收在`output/hhc_sota_20261016/coshmdm_*audit*.json`、`coshmdm_checkpoint_names.json`、`coshmdm_release_acceptance.json`；无新权重下载，HHC训练/队列保持。

## HHC原生VQ重建已实际开训（03:29 SGT）

旧185k仍不可恢复：本轮旧server SSH再次超时；本地权重清单仅有历史InterGen/CLIP，Drive根目录只发现已查过的项目备份目录；W&B匿名元数据请求返回project=null，不能据此取回权重，也不据此断言远端不存在文件。基于真实VQ smoke约1.8GiB的内存证据，已选择在现有云4090上共卡重建，无需暂停原InterGen，也没有新增pod或本机checkpoint下载。新GPU任务受20%进程显存上限、CPU nice10/2线程限制，evaluator在CPU；原InterGen2353仍在更新。用户若明确选择算力优先级，再按其选择调整；未将未答问题当作暂停许可。

新run为`/workspace/HHC/output/vq_hhc_native_rebuild_seed123_20261016`，driver5690/trainer5729。26.79M参数、batch256/window64、码本12288×2、body96/root64、原loss权重、999 warmup+300000主步/300k cosine；每5000主步原中心窗口480val。新入口`tools/train_hhc_native_vq_cloud.py`的validation函数、loss表达式和optimizer AST与原入口完全相同。显式独立CPU Generator随机无放回排列保留窗口分布，序列顺序不同历史RandomSampler，完整保存排列与cursor，不称历史bitwise复现。W&B disabled，JSON日志权威；隔离绘图依赖有固定requirements，不修改原InterGen运行环境。

首次启动因缺绘图依赖在导入阶段exit1，无权重更新；补齐隔离依赖和import preflight后，initial4/resume6均exit0，模型/EMA逐项重载相同，105个optimizer状态和batch cursor1024恢复。随后持续训练已越过warmup。云CPU恢复审计通过：update1999/main1000，107个模型tensor、105组optimizer全部有限且step均1999，EMA有限，保存cursor511744后的下一批可重复；342031139B/SHA `f7cb110330d681923b5de9e3cd7b816e42af833aa89c987bc2e3e67921396e1d`。仅小验收JSON收回本机，恢复包始终留云；不是质量评估或异地云备份。控制及准确恢复命令见`migration/control/hhc-vq-rebuild.json`，小时监控已纳入新任务，仍禁本机权重中转。

首个质量点为main5000。当前没有本轮HHC FID提升；RMSE、Full/Body MPJPE、root误差、Full/Body joint FID、滑步、contact F1均待评，Hands MPJPE/Hands joint FID不适用。原native VQ的训练内InterCLIP重建FID也不能替代MGT生成FID或joint FID。

`tools/freeze_hhc_rebuilt_vq_cloud.py`只在云端对val最佳权重建不可变hardlink，并绑定step/hash、normalizer、训练合同及所选验证报告。最终默认要求VQ完整训练完成；中途快照必须显式`--pilot-snapshot`，标签为intermediate_val_selected_pilot。MGT新增`--vq-spec`承接新码本来源，不把step写死185000；旧185k模式仍保留。新的MGT默认micro4累积有效64、CPU评测、最多GPU50%内存，已经通过CPU导入；真实VQ端到端MGT smoke和A/B/C仍待所选验证checkpoint。原InterGen验证复用同一GPU模型，先前另需第二完整GPU模型的推测已被源码核对纠正；启动MGT仍应核对实测显存，不能把VQ共卡成功外推为所有训练并行均安全。

03:44 SGT：main1000真实恢复包的云CPU审计exit0，342031139B/SHA f7cb110330d681923b5de9e3cd7b816e42af833aa89c987bc2e3e67921396e1d，107模型张量/105 optimizer状态/EMA有限、步数1999及batch游标511744一致、下一批可重复。首个main5000验证执行中，尚无质量结果。新MGT文本缓存使用CPU fp32单次CLIP前向，实际batch1/8/37/128（含空/长文本）的global与word输出与旧实现逐项完全相同，maxdiff0；其他device/dtype保留原路径。这是准备时间优化，不是FID收益。真实全MGT smoke入口的CPU导入exit0，尚未运行GPU。

现有`evaluate_hhc_interhuman_vq_validation.py`及geometry的checkpoint入口是旧262D模型/manifest schema，不能直接加载本次native276D新VQ。将来复用geometry的`--generations`需要先独立原生解码并按完整冻结请求转换/绑定H5；不能仅把参数路径改成新checkpoint。Full/Body joint FID尚未实现，Hand两项不适用；geometry contact是距离阈值proxy，不替代语义/mesh接触F1。

## 尚缺资产与后续执行

### 04:40 SGT：canonical-pair位置损失对照启动

原官方`data/utils.py:148–153`实际朝向只用hips[2,1]，肩关节虽列出但未用。新`utils/hhc_canonical_pair.py`从NativeSMPLFK的Y-up世界关节做每人floor、锚点初始XZ归零及锚点朝向旋转，对第二人的相对位置保留共同坐标；顺序及T-1裁剪与原代码一致。它只用于新增训练损失，原官方evaluator及推理预处理未修改，也不新增脚贴地/刚性对齐后处理。

真实cloud audit55074 exit0：480个官方请求×GT/预测×anchor1/2×swap两种共3840项，最大误差GT1.4514e-5m/pred2.1994e-5m，均值约1.9e-7m；数值等价，不声称bitwise。CPU double方向导数差5.98e-12，公共yaw/translation不变性差1.78e-15m、swap逐项相同、退化水平hip向量拒绝。冻结VQ仍step5000/原SHA；报告`output/hhc_sota_20261016/canonical_pair_parity.json`绑定新源码和官方utils SHA16cce02600004466a6b0aed45f383b248f2b8fa61577faee3caba3281864abf0。

新pilot`train_hhc_decoder_canonical_pilot_cloud.py`用同一base5000，仅decoder9.33M参数，seed20261016、batch256/window64、FP32、原AdamW、1000更新，新增3×两个锚点平均canonical joint SmoothL1；3沿用原native Joint loss系数，而非按val调参或放大上一轮无效geodesic。原RawLoss全项保留，编码器/码本/condition固定eval。对照复用已完成decoder_control1000，要求base SHA、checkpoint SHA和整段batch order digest一致；新项训练结束后才独立跑完整480val/FID/geometry/配对CI。exec20986/PID8835真实更新及冻结检查已通过，质量仍待评。

CPU源审计检查optimizer/batches/components/root diagnostic/objective/params赋值AST相同；既有23锁定源逐项SHA未变，不影响主VQ和7178/8321。原InterGen268 scratch另在20000完成480val FID24.314143/Matching3.981196/R .125/.191667/.245833/Div7.603141（GT7.741262），best20000，已继续20250；仍为独立旧实验，不称HHC改进。

04:27 SGT主VQ更新：已到main15601，最近完整val/best为main15000，重建FID3.458571、原生22joint Body117.367mm、Matching3.812677、R1/2/3 .333333/.487500/.612500、Div7.721899/GT7.715767；此前main10000为4.223619/125.507mm。协议仍固定val前480/center/max300/raw native，非上方独立官方请求几何/生成协议，不能交叉当作方法收益或SOTA。原InterGen2353、VQ5690/5729、ABC7178、text-only8321在本次/proc实查均存活；两个等待器尚无MGT任务启动。

### 04:12 SGT：独立官方请求VQ评分与误差归因

VQ仍为早期main5000/seed123快照（SHA554e6e3adf105bae838d9917271e5919ace0fa595b75be3c881ea60c608c76e5），不是最终模型。新`export_hhc_native_vq_validation_cloud.py`已完整导出与三个生成基线相同的480条请求，严格保持crop/person order/有效长度；raw输入重复末帧补到4倍数，解码后先裁掉padding，再做原官方预处理。H5仅云端，95164618B/SHA480446680691e103f4424e8da0b0a5d8dcfeaf4f9382a601de12c6cec1a505c3。恒等往返最大坐标差4.7011e-6m、接触0/365800差异；权重未变。导出exec26675、官方CPU评分31671、几何评分16687均exit0。

| VQ十项，官方480val、5000步 | 独立结果 |
|---|---|
| Joint coordinate RMSE | 352.627mm |
| Full MPJPE，22 joints含root | 372.110mm |
| Body MPJPE，21 non-root joints | 373.027mm |
| Hands MPJPE | 不适用，无手指关节 |
| Root position error | 352.851mm |
| Full joint FID | 待评，非InterCLIP FID |
| Body joint FID | 待评 |
| Hands joint FID | 不适用 |
| 滑步 | 13.696mm/frame；同GT接触足转移分母的GT参考4.258mm/frame，30Hz |
| Contact F1 | 几何距离阈值proxy .624510；语义/mesh接触F1仍待评 |

91450 person-frames，按有效帧汇总，padding不计误差；不额外刚性/root对齐。接触proxy为22×22跨人关节min距离<=0.1m，不能冒称语义接触。现有2项CPU几何检查（mask/scale/group/pooling、接触计数/固定GT足分母）直接调用通过exit0；云环境无pytest，未安装或宣称pytest执行成功。

官方旧bulk96评分：重建InterCLIP FID4.575838、Matching3.812979、R1/2/3 .381250/.543750/.612500，Div7.717589/GT7.841806（seed20260130、长度排序）。新增`score_frozen_interhuman_h5_cpu.py`与三基线完全相同的逐请求CPU编码、原始顺序/96候选池和seed20261006：FID4.575837786、Matching3.812979190、R相同、Div7.663406/GT7.741262；GT/text embeddings与InterGen基线逐项torch/NumPy相同，maxdiff0，exec48640 exit0。两种Div的seed/顺序不同，不能把GT Div不同说成GT特征改变；FID数值差约2.2e-7属编码批量数值差。本VQ是重建，不与生成8.7或论文test值排SOTA。

全480同请求原生世界MPJPE137.136mm、root98.724mm、root-relative body89.469mm。只作误差归因的GT oracle对照（不能部署/不能当改进结果）：原预测官方Full MPJPE372.110mm；仅替换root位置346.834；仅替换全局root rotation155.550；两者均换84.243；全部GT恒等0.000286mm。原预测与冻结H5逐项完全一致。结果说明该早期模型的全局朝向误差对官方处理后误差影响较大，不直接外推最终300k，也不据此更改evaluator。

GT-free边界token重复的固定32条筛查为负结果：decode0 Full MPJPE438.898mm，重复8/16 tokens分别566.053/588.489mm；first-frame全局朝向26.069°变28.954°/29.961°，未改善，不扩跑480/FID、不按少数接触值挑成功。所有原始小结果在`output/hhc_sota_20261016/native_vq5000_official/`。

下一项真实方法pilot已在云端运行：`train_hhc_decoder_root_pilot_cloud.py`，exec36269，固定5000码本/encoder/condition，仅训练decoder，两臂各1000更新/同batch256/window64/seed20261016、AdamW原参数，差别仅额外`.01 * global_root_geodesic`。它让global rotation单独获得与原全22joint rotation mean相同系数，原损失仍保留；不是用扰动探针直接设权重。两臂仍需独立480val评分，训练loss不算收益。主VQ/旧2353/A-B-C队列均未更改，所有新增权重只在云。

公平性补充：新增独立`export_hhc_mgt_validation_cloud.py`默认固定零体型同时送入MGT/VQ/FK，不使用测试GT betas，预定final40000模型；GT-shape requested模式单独标注。CPU导入exit0，GPU实际完整生成仍待MGT权重。04:22 SGT新后评测队列PID8321已经云端运行并核对/proc/STATUS为waiting_for_all_mgt，配置`migration/control/hhc-text-only-validation.json`，23项源码绑定，6项CPU mock安全检查exit0。等A/B/C三组全部完成才开始GPU工作，每组final40000的zero/requested均独立导出480H5、再用统一CPU scorer与InterGen GT/text逐项相同的门限验收。不会与MGT pilot训练争抢GPU，不改既有20源码或7178。结果仅val，完整test/MM/CI未做，不能宣称SOTA。

### 04:22 SGT：等预算decoder-only root损失负结果

完整两臂训练exec36269 exit0，decoder9,331,982参数、1000更新、固定base VQ5000、seed20261016、batch256/window64、FP32、无shape dropout，encoder/codebooks/condition完全不变；两臂所有训练batch的SHA相同：16f0a33ffe6f8bff57de0469c88932373653ef28cbcd334f5310d54d067f823d。control/root checkpoint分别SHA c6d23fbb05568edfc86a7b0ef47449498ee526561556320574689dadcd784a84 / 9091bc34a2cc74f5bebede5770ca70f09f89d5c13cf0a85ff3a18aaec5dda24d，各37341821B，只留云。独立override只严格替换decoder、验证base SHA和所有非decoder权重未改；不改原VQ spec/原5000步含义。两臂导出/统一CPU评分/geometry均exit0，GT/text同三基线逐项相同。

| 官方480val，均为重建 | base5000 | decoder control +1000 | 额外root geodesic .01 +1000 |
|---|---:|---:|---:|
| InterCLIP FID↓ | 4.575838 | 4.201122 | 4.319167 |
| Matching↓ | 3.812979 | 3.811854 | 3.812434 |
| R1/R2/R3↑ | .38125/.54375/.61250 | .36250/.53750/.63125 | .36250/.53333/.62708 |
| Div / GT Div | 7.663406/7.741262 | 7.667987/7.741262 | 7.664603/7.741262 |
| Joint coordinate RMSE mm↓ | 352.627 | 333.324 | 327.954 |
| Full MPJPE mm↓，22含root | 372.110 | 358.891 | 365.921 |
| Body non-root MPJPE mm↓ | 373.027 | 359.858 | 366.847 |
| Hands MPJPE | 不适用 | 不适用 | 不适用 |
| Root error mm↓ | 352.851 | 338.587 | 346.465 |
| Full/Body joint FID | 待评 | 待评 | 待评 |
| Hands joint FID | 不适用 | 不适用 | 不适用 |
| 滑步 mm/frame↓，GT4.258 | 13.696 | 13.771 | 14.122 |
| geometry Contact F1↑ | .624510 | .600663 | .591458 |

完整语义/mesh contact F1仍待评，proxy不能替代。主比较是最后两列：额外root项FID增加0.118045，Full MPJPE增加7.030mm，滑步增加.35085mm/frame，因此不采用、不接正式MGT；不能因RMSE单项降低宣称成功。control相对base的FID降低包含额外1000训练计算，且检索/接触变差，不把它冒称新方法或全指标提升。

480个identifier配对bootstrap10000次/seed20261016：root项减control的Full MPJPE差95%CI[-15.440,28.933]mm，root error[-15.088,30.144]mm，contact F1[-.04127,.02268]，滑步[.23013,.47340]mm/frame。只覆盖这些固定模型的val抽样不确定性，不是训练seed/generation20rep/SOTA置信区间。下一步若继续朝向方向，应先验证可微官方canonical joint loss与实际预处理一致，再做单因素decoder试验；不能由GT oracle直接推断geodesic加权有效。

03:45 SGT首轮完整VQ验证：main5000，seed123，固定val前480/center/max300/raw native协议；重建FID4.883522、Matching3.818581、R1/2/3 .352083/.487500/.591667、Div7.744356/GT7.715767、原生22joint Body MPJPE136.739mm，当前最佳step5000。这是重建指标，不能与生成基线FID8.7直接比较。

| VQ十项 | main5000状态 |
|---|---|
| RMSE | 待评 |
| Full MPJPE | 待统一geometry评测 |
| Body MPJPE | 136.739mm；原训练监控22joint含root，不混作body non-root结果 |
| Hands MPJPE | 不适用，无手指关节 |
| Root误差 | 待评 |
| Full joint FID | 待评 |
| Body joint FID | 待评 |
| Hands joint FID | 不适用 |
| 滑步 | 待评 |
| 接触F1 | 待评 |

云端pilot冻结快照115069810B/SHA554e6e3adf105bae838d9917271e5919ace0fa595b75be3c881ea60c608c76e5，freeze exit0；原VQ继续训练，快照不是最终模型。核对15553MiB空闲后运行真实全MGT smoke，exec74877 exit0：285709616训练参数，2次更新loss9.74741/6.05805，梯度有限；14/299/115帧生成3×300×524，官方evaluator嵌入3×512有限，冻结VQ所有张量未变，峰值5.4628GiB，无MGT checkpoint。此处不计算小样本FID，也不凭两次更新判断生成质量。权重始终留云端。

03:50 SGT，云端driver7178已排队等完整300k VQ，状态waiting_for_complete_vq。队列源码`tools/run_hhc_abc_after_vq_cloud.py`及配置`migration/control/hhc-abc-after-vq.json`绑定20项SHA，随后freeze最终val-best并顺序A/B/C三组各40k：A historical text+historical window；B matched global/word dropout+historical window；C historical text+uniform window。同VQ/seed114514、有效batch64/micro4、AdamW、BF16、CPU evaluator、GPU50%上限。每组10k间隔val+MM，只做val选模，无formal test。启动须测空闲>=13GiB；非零exit/源码变化/显存不足即记录并停止后续组，不自动重跑。6项mock CPU调度检查exit0，不加载模型或GPU。A/B/C尚未训练，不声称任何收益。现有小时automation已覆盖队列；原InterGen2353继续。

早期`output/hhc_sota_20261016/asset_preflight.json`是恢复前本机快照；目前native数据及MM内容已云端恢复，仍缺原HHC VQ185k/MGT。现有备份文件夹134项未找到该HHC权重。不要拿InterGen recovery冒充HHC权重，不重用失配归一化，也不把代码完成当作实验启动。

旧服务器若恢复可读，原HHC VQ185k仍只能直接传到云端并核验，本地不下载。当前fresh VQ重建已实际训练，不再把旧185k缺失当作整个目标的阻塞；历史HHC20rep结果仍仅为旧记录，不能与新分支混称同模型。

云端资源与资产齐备后先真实smoke再pilot，记录step、val FID、Matching、R1/2/3、Div及MM。确认有可重复收益后，再独立测试masked-only/step-unroll或rotation+FK的VQ改进；不同时改变多项，也不以随机扰动响应比直接设loss权重。采样CFG/T只用val校准；test不用于反复试参。

最终SOTA验收：统一请求/evaluator重跑基线、固定条件披露、val冻结模型、完整独立test/MM及95%CI，并报告穿插/接触/滑步与推理成本。对位置—旋转一致性或单seed FID改善都不能直接称SOTA。既有旋转后处理的test rep0收益约1.02%/3.95%，检索有涨跌且无MM证据，不足以替代上述流程。

## 算力与本地checkpoint政策

用户2026-10-16新要求覆盖旧滚动本地备份指令。所有GPU工作只在云端；本机仅CPU分析/源码/小结果。`migration/control/local-checkpoint-policy.json`禁止新checkpoint下载，选定检查时已收齐的step15000 recovery作为唯一后续使用副本；旧历史文件不擅自删除。已停止旧initial-backup30948，重启加载政策的Runpod controller53968；云端trainer未重启。两项回归验证“待传输任务不能绕过禁令”和“仍可读取云训练状态”均通过。

既有hhc小时automation已更新，禁止重启旧下载器或绕过新政策，要求云到云备份。云到云checkpoint备份尚未验收，不能把旧本机中转方案说成已满足新要求。后续保留云端恢复点，待配置并验证云端备份。

## 2026-10-17 公开资产路径与比较版本验收

[TIMotion作者讨论#7](https://github.com/AIGC-Explorer/TIMotion/issues/7#issuecomment-3303279274)补充了公开backbone身份的直接来源。发帖者报告的成绩属于其自己的复现，不是本任务结果；作者关于随机性的回复也不证明本任务480val等价于论文20rep。论文中[Transformer5.433与RWKV4.702](https://arxiv.org/html/2408.17135v3)保留为不同条目。作者#2表示计划上传HF不等于已发布，实际HF API查询范围和时间均已保存。

[CoShMDM项目摘要](https://alimanjotho.github.io/coshmdm/)明确text及body-shape conditioning；目前尚未独立验证其test shape取值来源，不能进一步猜成GT shape。项目三个Paper/arXiv/Video按钮href为#。其[DOI](https://doi.org/10.1109/TVCG.2026.3675725)已由Crossref出版社存入的元数据验证，volume32/issue7/pages5911–5924/2026年7月；IEEE页面本次仍为机器人验证提示，未取得完整论文。占位按钮不是论文本身不存在的证据。现有公开脚本仍沿用InterGen下载ID，未获得符合论文架构的独立权重；不借用InterGen补第四行。

本轮新目录是已有三条480val结果和文献版本之间的身份映射，并非正式test汇总。公共val报告逐文件SHA、request SHA、metrics以及原comparability receipt的0差异声明已核对；未重新读取三组embedding，未重跑任何模型，不能把旧收据检查描述成新数组实验。HHC最终text-only选模后仍须实际执行统一20rep main/MM；文献成绩单列，保留condition/版本/采样预算差异。资产范围已闭合，不在公开源未变时重复同一搜索；新增RWKV/CoSh对应权重后才能宣称复现相应架构。

原始小JSON：public_baseline_asset_routes_raw_verified_20261017.json；机器清单：public_baseline_comparison_identity_20261017.json；此前public_baseline_asset_routes_20261017.json已标明正文为assistant paraphrase，不是API原文。没有向作者发消息，没有权重/数组下载。
