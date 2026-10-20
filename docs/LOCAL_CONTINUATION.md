### 2026-10-16 22:29 SGT｜VQ300k完成，原队列接续A

- fresh VQ manifest=complete、main300000/update300999，原driver/trainer已不在进程表。最终480训练内重建FID2.637497，best170000/2.447121；Matching3.808705，R1/2/3=.358333/.516667/.614583，Div7.763293/GT7.715767；seed123/fixed_val_first480_center_max300_raw_smpl_v1。较295k FID/Matching/native Body误差及Div偏差变差，R1下降、R2/R3改善。
- 十项严格口径：RMSE、Full MPJPE、Body non-root MPJPE、root误差、Full/Body joint FID、滑步、接触F1待评；Hands MPJPE/Hands joint FID不适用。额外native22含root监控101.681892mm不替代严格Body项。
- 原7178已自动freeze_final exit0、进入A_historical child17259（实际命令同绑定配置），tokens.pt存在，尚未验到更新日志，不宣称已完成训练更新。freeze spec为completed_training_val_best，step170000、115069874B、SHA2f676f63466135385ded7016d60da9dd3b6e86e3086b85e14a201732d1a55002、origin=fresh_cloud_rebuild_not_historical185k。未手动重复launch或重启VQ。
- InterGen2353推进52050；8321/9975仍存在。SSH远端GPU16%/6451MiB，保持运行、idle_since为空；一次SSH banner超时后原地址重试成功，无算力切换。无本机checkpoint下载；权重云到云备份未验收。下一步仅观察A实际更新和后续队列，formal/test仍不启动。
### 2026-10-16 03:49 SGT｜双训练闲置保护复核

- 实际进程、日志及 manifest 复核：InterGen scratch PID2353 推进至 step19350；fresh HHC VQ driver5690/trainer5729 推进至 update7000 / 主步6001。两项均活跃，GPU利用率100%、显存8531MiB；未重启或停止任何训练。
- 已通过应用更新并读回验收 `hhc-runpod` 每15分钟保护：同时检查两项训练及其他有效评测/部署/传输，任一活跃均不得停机；仅全部闲置30分钟才停机保留卷。SSH失败本身不构成GPU闲置证据。
- 本机控制器53968继续运行；本轮无checkpoint下载、无initial-backup重启、无新增算力。云到云checkpoint备份仍未配置验收，不能宣称备份完成。此次没有新的完整验证结果。
# HHC 本机接管（2026-10-15，新加坡时间）

> **2026-10-16 02:10 SGT 最高优先级新覆盖：所有GPU训练/推理/评测只在云端；本机不再新增checkpoint下载（本任务上限一个）。migration/control/local-checkpoint-policy.json已禁下载，step15000是检查时已收齐的选定副本，旧历史文件保留不擅删。旧initial-backup30948已停，不得重启；新Runpod controller53968只做状态/恢复监控，不下载。hhc小时automation已更新。下方旧“持续收取本机恢复包”指令全部被此条覆盖。云到云备份尚未验收，不虚报成功。**

新goal为改进HHC并按统一InterHuman评测比较InterGen/InterMask/TIMotion/CoShMDM。fresh VQ300k已完整完成并冻结val-best170k；用户新指令为InterGen独占，A已保存4000步暂停并排队，待InterGen完整704k结束后继续ABC与zero/train-only shape验证；最新句柄与证据以“当前目标”和docs/hhc_sota_interhuman_20261016.md为准。公开三基线480val已验，正式生成比较尚待真实HHC选模。旧独立InterGen scratch704k云训练保留，不能冒充HHC改进或SOTA。

**02:36活跃云端句柄：** InterMask480val runner3916已生成exit0、正在CPU评分4133；InterGen baseline runner4126等待`output/interhuman_baseline_restore_20261016/gpu_validation.lock`，不因未有STATUS而重复启动。输出均在`output/interhuman_baseline_restore_20261016/{intermask,intergen}`。HHC native builder3521/3522 exit0，数据`dataset/interhuman_smpl_interx_layout_v1_20261004`，完整收据`output/hhc_native_restore_control_20261016/full_status.json`。VQ smoke4037导入未用SMPL-X失败，未进入GPU；懒加载修复后待新唯一output重试。所有权重留云，不收本地H5/ckpt。用户算力优先级async问题待答，不自行停2353。


## 当前目标

2026-10-20 Edmond 迁移已验收：官方 InterGen 从 Runpod 完整262240步状态恢复；新262592步 checkpoint（196项optimizer）再次CPU验收通过。主算力为免费 newstation / edmond-Z790-Taichi，实际目录 /mnt/8T-HDD/wenning/hhc_migration_20261020/workspace/HHC，使用同级 edmond_runtime_namespace.py 保留 /workspace 路径。trainer138991/35423489；HHC/eval等待139371/35441518、139469/35450774；watch139818/35465491；两个下游CPU等待139470、139471。所有PID以实际LAUNCH/STATUS核验。Runpod不再作为主算力，禁止自动恢复；控制面停机结果见 output/edmond_migration_20261020/runpod_stop_receipt.json 和 migration/control/compute-owner.json。两automation保持PAUSED，HHC4000与官方704k+真实20rep/MM+独立review门槛不变。后续改动须提交并推送GitHub，同步Edmond旧项目与实际runtime。Runpod停机保留本次版本；若将来获授权启用，须先拉取同一最新提交，不得为例行同步自动启动付费实例。无新FID，无本机权重/数组下载。详见 docs/edmond_migration_20261020.md。

恢复证据包official_intergen_restart_recovery_evidence_20261020.zip已验收：467222B/SHA765c48111f02e6ba1db85dd44cc84914fbd9c72cef2fddd4ee2a8ce9a770d987，85成员，云端整包和逐成员SHA均一致；Drive文件1w7fPiOddWoSmi33ChK1n1m1wvj4nLjEj的id/name/size/parent读回匹配，API未返回checksum，不声称Drive SHA。仅源码/小JSON/失败日志/文档，不是模型异地备份；本条及最终观察晚于ZIP内文档快照。收据output/hhc_sota_20261016/official_recovery_backup_receipt_20261020.json。

**2026-10-20 17:56 SGT：云端容器重启后的完整恢复已验收。** 原进程全部缺失、255552半文件已隔离保留，从完整255200恢复；新增外层Lightning ckpt_path完整状态恢复并补执行原pending epoch scheduler hook，保留原官方源码/磁盘配置和激活包装。冻结YACS CPU对照与实际首批/196项optimizer/计数通过；恢复后的新255552完整checkpoint再验通过（2701620469B/SHA cf983bfe...cab8a2），仅留云。tmpfs原压缩包SHA、每人7777文件/15558 eval链接及官方Dataset11274/352 CPU验证通过；两个未更新失败attempt均保留。当前trainer1282/3987440，HHC/eval1486/3999646、1490/4003594，watch1491/4003597，下游1506/4003941与1507/4003959，全为实测身份。后续使用tools/read_official_intergen_wait_local.py消费既有watch与实际/proc，勿依赖临时functions store。两automation只更新prompt，仍PAUSED/频率目标保持；HHC4000与final704k+真实20rep/MM+独立review门槛保持。无新FID/SOTA证据，无本机权重/数组下载。详见docs/intergen_official_reproduction_20261017.md顶部及output/hhc_sota_20261016/official_recovery_*_20261020.json。

**2026-10-18 22:52 SGT 新padding边界审计已验收：** 官方VEL/FC只检查前一帧mask，纳入有效末帧→第一padding差分。云CPU抽取真实AST，8合成+1实际train3292/80帧共9例exit0；有效预测=GT、仅首padding+1m时VEL=.374999962/padding梯度.749999924，独立双端mask候选均0。合成1cm接触例亦复现；FC受预测门控限制。旧custom已双端有效，此项不解释其FID退化；当前官方/排队绑定源未改、无新质量结果。source/report SHA独立读回通过，session32827 exit0；第一次诊断epsilon容差断言失败保留，不当模型失败。详见docs/intergen_bug_audit_20261017.md顶部和intergen_padding_velocity_cpu_20261018.json。仅云CPU/小JSON源码，本机新增权重数组0，HHC4000及官方优先保持。

**2026-10-17 23:49 SGT新增bug审计：** 镜像文本counterclockwise→countercounterclockwise，在当前官方/旧custom/HHC adapter均存在；云CPU实际全训练5637交互/16911描述影响60交互/150描述，六合成例+全描述三源对照exit0。候选仅在未绑定工具audit_intergen_mirror_text_cpu_cloud.py内，不能改官方源或直接接入HHC；共享bug不能解释FID差距。source/report SHA读回通过，报告intergen_mirror_text_cpu_20261017.json与acceptance已保存，无GPU/权重读取/本机数组下载。当前官方29Python固定提交一致，22675/25145/26133真实身份匹配，两个队列仍等待。详见docs/intergen_mirror_text_bug_20261017.md；相同来源不重复扫描，维持官方优先/HHC4000/automation PAUSED。

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

备份验收：本轮bug修复/官方baseline启动源码与小证据包intergen_bug_fix_and_official_baseline_startup_20261017.zip，534226B/SHA9620746b3f490d1ce570b3e310fa02b586b3339bab32911d14841074702f1a22，83成员（含37个与云端固定commit逐字节一致的官方源码文件）。云端归档完整字节/SHA及逐成员SHA已验收；Drive文件1sS6-KQ9RhuIeuCYFI5yhbcLIpFQBUG_l，metadata大小534226/父目录核对一致，Drive未返回checksum，不声称Drive SHA验证。只含代码、配置、文档与小JSON；没有权重/生成数组，也不是模型异地备份。归档docs快照早于本条备份回执。模型checkpoint仅云卷保留。回执output/hhc_sota_20261016/intergen_bug_startup_backup_receipt_20261017.json。官方run最近训练验收仍为13:11 UTC/step880，704 checkpoint完整性已过；下次普通观察≥10min，目标继续active。

**2026-10-17 21:11 SGT：官方基线训练与完整checkpoint已验收。** 云端CPU验收固定工具原exec64670明确exit0；取得当前可用epoch1/global_step704 checkpoint，2701618229B/SHA fbbba5ee3cbfcc50abcd1189eeb75172af83b2ee0af3fa9c40e721d239cbe5aa，云端accepted_checkpoints硬链接快照保留；输入/输出头262D、196项optimizer状态全部step704、权重与optimizer均有限、scheduler warmup10/max_iters2000通过，官方源码与运行包装SHA均匹配。仅云端存权重，不下载本机。默认官方Lightning checkpoint未含顶层RNG，不能声称bitwise恢复；此处验收权重/optimizer/scheduler/counters完整性。

13:11:32 UTC实际trainer22675活跃，epoch2/step880，LR3e-5、官方total .6334，GPU仅22675/7880MiB；首次epoch checkpoint与后续更新都已发生，不能再把此run标为仅launch。CPU观察器21986的/proc也已独立验明存活，每600秒观察不恢复。下一普通复核≥10min，仅验收/异常提前；早先root定时exec64670已收终态不重启。旧50420/cell1053只读旧run观察仍过时，不能作为官方job句柄或续训依据。

真实稀疏loss CPU复核v2已由root审源码、检查归一化/请求重放/零mask项为零/其余loss逐位不变，并对云端源码与报告SHA独立读回一致：有非零affinity的3请求，修正项/旧项335.27–415.85倍；contact2.00–4.41倍。v1曾混淆物理量/归一化且GT构造错误，已明确INVALID，不引用其数值。v2报告13359B/SHA057fab6a...d159b9，结果只量化同样本损失缩放，不是新FID，也不能断言解释所有退化。独立loss修正版未接入任何训练，官方基线始终使用官方loss。

SOTA/正式HHC生成比较未完成；HHC仍saved4000且未启动，官方baseline最终完成后的新HHC门控待安装，不能启用旧自定义704k门控。后续先维护官方baseline稳定运行/里程碑评分与新queue，再做单变量表示实验。完整证据见docs/intergen_bug_audit_20261017.md、output/hhc_sota_20261016/official_intergen_startup_checkpoint_acceptance_20261017.json及official_intergen_observer_status_20261017.json。

**2026-10-17 21:00:32 SGT 官方基线已验收实际更新。** 云端固定observer返回trainer22675/birth467827326真实活跃，epoch0/step40/inner39，LR1e-5；官方loss total5.2207、simple1.3785、DM2.765、JA.1465、RO.005，均有限。GPU仅22675，占7880MiB；没有HHC GPU作业。此为官方loss基线的训练进度，不是FID结果。旧自定义run保持103000保存暂停。

独立显存重计算probe已验证输出/170参数梯度/输入与条件梯度最大差0、RNG一致；外部包装只改变激活保存，不使用sparse-v2自写loss。CPU观察器21986按600秒观察当前LAUNCH；不自动重启，无HHC恢复器在跑。下一实质验收为首个epoch全量checkpoint及optimizer/scheduler/counters，再建立官方基线完成后的HHC新门控；不能用旧v3等待源或旧704k身份复活自定义训练。HHC仍saved4000。goal仍active、SOTA未验证。

**2026-10-17 官方基线启动验收中：** 当前云端trainer22675/birth467827326，attempt4，日志output/intergen_official_baseline_20261017/train.attempt4.log；不是旧自定义run。官方源码固定b1750df，所有Python文件与提交相同；train.yaml仅输出路径/名称和每epoch保存有改动。源码执行入口经外部run_official_intergen_checkpointed_cloud.py调用未改的官方tools/train.py，仅为TransformerBlock启用nonreentrant activation checkpointing并保留RNG。完整8层FP32/双人/dropout0.1/非零head/含padding对照：输出、170个参数及输入/条件梯度最大差均0，反向后CPU/CUDA RNG相同，probe exit0；不扩大为多步bitwise证明。wrapper4582B/SHA54a4b2759c638fd38b0a662cdef3d12b71ac14809d3a00d4fc2b769d0a2e48f3，已完整上传读回核对，运行期间禁止改写。

前三次启动均在任何optimizer更新前退出：1数据目录嵌套符号链接未被os.walk遍历；2训练注释5834非UTF8；3原始batch32/FP32超4090显存。前两项已按实际物理目录视图及官方支持的ignore_list修正，官方CPU Dataset实测11274条/5637个ID，与现有adapter完整eligible集合相同，无按指标筛数据。第三项用上述已验证显存重计算解决，当前仍须以实际训练日志确认首批更新，不能把launch当训练完成。所有旧失败日志与LAUNCH.attempt1/2/3.json保留。

bug已有可复现数值证据：稀疏pair仅1/484有效时旧加权loss .00619835而官方同mask/独立修正版3；每人1接触脚时旧7.5而官方求和60。CPU梯度/padding回归exit0；独立修正版intergen_rotation_objective_sparse_v2.py（SHAf7565755...1856）未接入任何训练。GPU生成8请求20k/103k诊断完成，直接位置/中性shape FK差74.715→47.984mm，去噪误差也改善，不能认定此项解释全部FID退化。详细docs/intergen_bug_audit_20261017.md及小JSON已有本机验收；不下载数组/权重。

CPU只读观察器21986（observe_official_intergen_baseline_cloud.py 3003B/SHA4f914426aedc4c9def3ea5702f71b56c34c15c49ee5cf17457d5e41fb2124e70）每600秒更新OBSERVER_STATUS.json，只观察不恢复/不启动HHC；首次观察21822初始化后该attempt已OOM，当前必须读取最新LAUNCH和/proc来识别22675。两项app automation保持PAUSED。旧exec50420/cell1053定时观察已过时，无权触发训练/排队。HHC仍saved4000且旧waiter20344已停，后续官方baseline验收后再建立新完成门控，禁止套用旧自定义704k配置自动续训。

**2026-10-17 用户已改变算力安排：保存并暂停自定义旋转/FK实验，先复现官方 InterGen 基线，并要求检查 bug。** 云端旧训练20042在103000步完整状态校验后有意退出（Z）；完整恢复包2188977969B/SHA bcaee8ce1088411b68d3b514cf20c6e15271b93241cc23bb4d735d129f51c651，latest729646551B/SHA d36d31b254ecf7296d0e189a32ab5af83ffdc389fd1d880e584b351efebdcec3，留在云端 output/intergen_official_baseline_transition_control_20261017。仅为云卷硬链接快照，不是独立异地备份。旧本机恢复器104244和云HHC等待器20344有意停止，不恢复旧704k运行；HHC仍保存4000等待。旧RECOVERY.json移存RECOVERY.paused_for_official_20261017.json，用于阻止旧恢复器自动重启；权重与完整恢复包保留。两项app automation维持PAUSED，无新本机权重下载。

已完成CPU实际数据对照：8个训练ID、64个镜像/锚定人物/输出顺序组合，当前前262维与未改官方Dataset.__getitem__逐位相同，mean/std亦完全一致；未覆盖长窗口裁剪分支与完整随机分布，不能扩大为全管线等价。报告37052B/SHA f6fe834ec03ff848d9ef48154dd16d49af4f5db3c87a2e844b57087838433484。代码7183B/SHA beee72e062a552ad511001638346900503d45d8d87d2ba5c21311ef39609b95d。上轮审计/本轮数据证据均属实质进展，目标尚未完成。

新任务：云端CPU官方环境/源码准备器20888，日志output/setup_official_intergen_baseline_20261017.log；GPU bug诊断21104，日志output/diagnose_custom_intergen_heads_20261017.log，仅在旧训练退出且GPU空闲后启动，保留预测root6、比较直接位置与FK位置并做teacher-forced去噪误差，尚未验收结果，不能称已找到根因。新官方源码third_party/intergen_official_baseline_20261017来自固定提交完整git archive，拟直接运行官方tools/train.py，所有训练Python源码不改；训练配置只改输出路径/名称及每epoch保存（原100epoch），保留原batch32、2000epoch、FP32、官方loss/epoch调度。准备器STATUS才是实际进度，不把准备当训练。旧定时observer exec50420/cell1053已过时，只读结果不再作为训练704k等待依据，不重启。

**2026-10-17 InterGen fidelity audit correction.** Current rotation/FK run is NOT an official reproduction with only representation changed. Nine official source files match commit b1750df4a859443b8a69329f759ce3a7503ae179 byte-for-byte on cloud. Custom trainer/data/loss, 268D, BF16, step LR schedule, and 10000-step weights-only continuation (optimizer/RNG reset) are confirmed. Export strips root6 without rebuilding evaluator positions/velocities by FK. These changes confound attribution; current 480-val FID cannot challenge an official ~6 reproduction. Evidence: output/hhc_sota_20261016/intergen_official_fidelity_audit_evidence_20261017.json. Detailed audit: docs/intergen_official_fidelity_audit_20261017.md. No live source/process/queue changes or new GPU task. Observer session50420/cell1053 remains pending; collect original cell only.

**2026-10-17 20:03 SGT：100k完整验证已验，训练仍正常，HHC继续等待。** 12:00:06 UTC原20042实际活跃、100350，GPU仅20042/10356MiB；20344仍waiting_for_intergen_complete、saved4000、无phase。原等待exec75105/cell1028最终exit0，不重复启动。100k FID35.2390586118、Matching3.9905038834、R .125/.216666667/.26875、Div7.891499996/GT7.741262436，较90k各主要项及Div差距变差，best仍20k/24.3141428796。MM待评，非HHC收益或正式test。

已用云CPU只读20/60/80/90/100k既有embeddings：GT逐元素相同，官方FID复算误差<2e-8。20→100k FID增10.924916，其中均值项增8.466891/77.5%；偏移80k已明显，90→100k均值项下降、协方差项上升，不能把回退简单归因84k性能切换。工具audit_intergen_cached_fid_drift_cpu.py 5415B/SHA2c6bdc...b4b29，源码云全字节/SHA一致、SSH exit0，不导入torch/模型或GPU、不改数组。小结果intergen_cached_fid_drift_100k_cpu_20261017.json 9052B/SHAf22015...d5d49；原100k metrics2642B/SHAdb25f4...944be，合并证据intergen100000_and_cpu_drift_evidence_20261017.json。没有权重/数组下载、训练重启或HHC GPU工作；目标active、SOTA未完成。

**下一普通质量观察：2026-10-17 14:30 UTC / 22:30 SGT，目标110k。** 本轮75105已终态；12:08 UTC已启动新的本机纯CPU定时只读任务，root原exec **50420**，orchestration cell **1053**正收取原句柄（用functions.wait继续1053，不要另行读50420造成竞争），初始delay8509.26秒。它到真实14:30才SSH调用固定observer，未完成则仅14:50/15:10/15:30补查；保留并收取原句柄，不重复启动，不将等待超时当训练退出。两项app automation仍PAUSED，不是此等待任务。无异常/新用户指令时此前不重复普通云轮询；现有小时恢复器104244真实进程存活、stderr空，最近状态记录12:02 UTC/100500，不冒充当前云步骤。HHC仍严格完整InterGen704k后才从保存4000续训。

本轮官方资产只读复核：TIMotion main仍172580fd7f2551a7cc10fde1f15ff73315742e65、CoSh main仍c5116ab3bead55051ffec2ec9a161a38d2ab5116、InterGen master仍b1750df4a859443b8a69329f759ce3a7503ae179。CoSh/InterGen下载脚本均276B/SHA1fa25981b86a93f3957a7a91854be0ea9a28d89066ee54626729a80d7a38eaac，同一Drive ID；TIMotion README仍只有原两项链接，没有新增可核验RWKV权重证据。低成本agent初步将“独立命名”误写为独立checkpoint，已要求撤回，root以两脚本逐字节SHA和官方API HEAD独立纠正；第三方PaperNotes不作为论文证据。没有新增权重或可复现行，表格标签不改。完整小证据official_baseline_assets_metadata_recheck_20261017.json为5681B/SHA5144bb548777e989fe41bb1084d595d2759bffd369325671b266bc77cd0e0bbd；仅官方HTTP元数据和源码，不下载模型，也不据有限搜索声称当前全球SOTA前沿已穷尽。

**2026-10-17 17:44 SGT：InterGen90k完整验证已收，继续同一fast训练。** 09:43:03 UTC原20042/proc活跃、91100；20344仍纯CPU等待complete704000，A保持保存4000，GPU仅20042/10356MiB。原只读observer exec42312/cell1013最终exit0，所有未来定时观察均已完成，不重启这个观察任务。

90k固定480val FID34.9960871932、Matching3.9862061183、R .145833333/.227083333/.285416667、Div7.714151859/GT7.741262436；best仍20k/24.3141428796。FID比80k差，但Matching/R/Div差距改善；不能归因于microbatch变更，不擅自改完整704k安排。MM待评；这不是HHC或正式20rep成绩。小metrics2668B/SHA1f07929f94613822f9221d54f8588a33a81130bd18304c77145c339e0caa70c7已验；请求/seed20261006/batch96/evaluator与80k相同。

84k后141个常规区间实测均值.752365秒/步（中位.726916），90k验证/保存额外约979秒；含验证每万步约2.36小时，剩完整训练估约6天。继续原云训练与严格HHC等待；不改PAUSED的两项automation、不下载权重、不新增GPU任务。下个质量节点100k；普通只读检查至少间隔10分钟，无变化保持安静。证据output/hhc_sota_20261016/intergen90000_and_fast_continuation_evidence_20261017.json。目标仍active，五候选生成选模/正式比较/SOTA未完成。

### 4090 吞吐修复验收（2026-10-17 16:12 SGT）

云端短基准比较后已从完整84000恢复包切换同一实验为micro16×accum2，effective batch仍32。新训练PID20042，权重、优化器和Python/NumPy/Torch/CUDA RNG完整恢复；原2353有意退出，不恢复旧版。训练架构、数据、学习率绝对步数日程、704000目标及原micro4有效帧归一化分组保留。较大microbatch改变dropout执行及浮点顺序，不声称bitwise replay或FID改善。

实际续训前1000步耗时767.70秒，约0.768秒/步，相对原中位约2.43倍；最近三个50步区间0.756/0.783/0.811秒。每万步纯训练约2.1小时；原每万步验证/保存还约16分钟，未来耗时仍随长度、保存和评测变化。最后8秒GPU采样为0–3%发生在85000保存附近，不作为整程利用率，也不能宣称已满载显卡。

84500新恢复包云CPU校验通过：2188977969字节，SHA256 `725a30c104a3475bbd79ec8884ba2194cea2c8a0de721062a361cbcce423d7af`，196个优化器状态、计数74500（历史10000步曾weights-only恢复，84000此次没有重置），所有权重/优化器张量有限、RNG齐备。最初身份断言失败为checkpoint的PosixPath和JSON字符串类型差异，路径相同；修正仅规范化source/output两个路径字段。验收脚本第一次误用model/rng键而报KeyError，按真实trainable_state及*_rng结构重验exit0，不隐瞒失败或冒充通过。

HHC保持A4000保存等待，不占GPU。旧CPU等待17952、20161均为有意取代；新等待器20344使用独立v3源和配置，严格要求InterGen complete704000、最终checkpoint身份与step一致且训练进程退出。v3路径规范化已用CPU正/负例及真实84500恢复包验收；预检release_gate=false，无HHC子进程启动。原29项HHC绑定源码保持。Windows监控104244运行fast恢复器；local-checkpoint-policy明确禁下载。本轮仅源码、日志和小型JSON在本机，无权重下载。

两项automation实际为PAUSED，保持其状态、原频率和目标线程；提示已指向v3并按UTF-8全文读回一致。它们目前不提供主动闲置保护。实际训练仍运行，不能停pod。

**下一次普通质量观察：2026-10-17 12:00 UTC / 20:00 SGT，目标100k完整validation。** 原90k观察器已exit0。09:48 UTC本轮已启动本机纯CPU等待任务，root原exec **75105**，初次回显delay7886.58秒；这是等待真实12:00后才SSH运行既有只读CLI，不是已经检查了100k，也不是app automation。若未完成，原任务仅12:20/12:40/13:00补查；保留并收取原句柄，不重复启动或把等待超时当训练退出。两项app automation仍PAUSED。

既有Windows monitor104244已由低成本agent即时真实进程查询确认存活，命令tools/local_runpod_control_fast.py，最近小时记录09:01:48 UTC/89000、stderr空、无新fast错误，checkpoint_downloads_enabled=false；此旧云快照不冒充当前云训练步骤。无异常或新用户指令时，不在12:00前重复普通云轮询；不得为了填写本轮报告重启训练或启动HHC。实际时间到后保留真实时间/exit code，结果缺失也不等于训练终态。

**2026-10-17 15:30 SGT：InterGen吞吐短基准已启动，主训练83300步暂挂。** 新tools/intergen_speed_probe_cloud.py 16865B/SHA14d8b05be14f482450b116043e07bd3ca6bb53ffd7476f175fed7e8bf689c916本机AST通过、云字节/SHA读回一致。两项云CPU假进程验收通过：已有worker receipt及controller写receipt前死亡的竞态，均停止dummy worker并SIGCONT dummy trainer；未碰真实2353。低成本agent静态复核发现的receipt竞态已以/proc严格argv回查补齐，watchdog退出延迟不冒充基准失败。

真实controller19712、worker19716、独立watchdog19715；output/intergen_speed_probe_20261017，root原exec session51800仍运行。原2353在83300主动SIGSTOP且/proc=T/birth451188498，原源SHA不改；显存原2353/5786MiB+benchmark19716/1654MiB（初始化阶段）。这是预期短性能测试，不是训练故障；禁止旧控制器另行重启原训练或HHC。watchdog上限15分钟，controller worker限13分钟，异常会先停止benchmark再SIGCONT原2353。worker只读原recovery.pt，不保存/覆盖原模型；HHC仍4000排队。

测同effective32的original4×8/deferred4×8/8×4/16×2/32×1，每档2warmup+8计时更新；同4×8核对输入/最终参数/优化器/CPU和CUDA RNG哈希，较大micro明确不称exact replay。benchmark独立显存上限65%，原进程显存驻留，所以OOM不能推断独占24GB不支持该batch；输入hash有少量共同测量开销。结果待验，尚未切换正式训练；本机无GPU工作或checkpoint下载。

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

**12:00 SGT：近期文献及协议分类补齐。** 已阅读InterCMDM、ARMS及Text2Interact原文，核对HINT输入条件的比较性说明；详细数值/来源在主报告新表，均为文献值，未作为本任务复现。不能只超过旧TIMotion即宣称当前SOTA；新论文的变体/初始化条件/合成数据/重复次数单列，不降低完整目标或修改已冻结评测。

InterCMDM官方资源的只读低成本子agent审计：main HEAD8a376f64ed9cd687995b3c24cb4d1a2235be9847，代码已公开；在所查README、完整tree、GitHub releases及项目页内未发现生成器预训练权重链接，release数0，README给出的下载是InterMask evaluator。只限这些官方来源的发现，不断言全网不存在；不另训该baseline或把evaluator当生成器。本轮无GPU工作/模型下载/实验源码变更，既有VQ300k及ABC/shape队列保持，已有监控agent等12:10原定单次实查。

新小证据recent_interhuman_literature_protocol_20261016.json为4215B/SHA7ecad1e062b57448c90ef7beeb7d9d7586630a2fdacc9d9570f629872a66bd36，[Drive备份](https://drive.google.com/file/d/1H7T-YJoaFFH-Plkb6KoJz9gARmc1HVAe/view) metadata大小/父目录读回一致、云checksum未暴露，收据同名.receipt.json。不是模型权重备份；本机checkpoint新增0，goal active，SOTA与完整正式生成比较仍未实现。

**11:52 SGT：130k新验证与原链路存活。** 11:50一次SSH exit0、六PID2353/5690/5729/7178/8321/9975及cmdline保持，GPU100%/8533MiB；VQ134801/update135800，latest130000 native重建FID2.5867905778，best仍115000/2.5664796810，contract16b75e...保持。130k seed123/fixed_val_first480_center_max300_raw_smpl_v1/480：Matching3.808081500、R .35625/.49375/.627083333、Div7.744163513/GT7.715767384、native22含root Body105.531544mm；较125k FID/Matching及Div差距较好、Body与R1R2较差/R3持平，非全项提升，非生成或joint FID。

130k十项RMSE、Full/Body non-root MPJPE、root、Full/Body joint FID、滑步、Contact F1待评；Hands MPJPE/FID不适用。此节点325B云JSON/SHA201bad873b460d3bd5dfea1ddc10527690246cea12c6f94417405c711d873942，本条不声称已下载/Drive备份。最近HHC完整官方480仍105k；本轮新InterMask公开VQ完整480重建对照见下方，不能替换任一生成baseline数字。InterGen scratch33200/best20000 FID24.314143；原三个queue状态仍waiting_for_complete_vq / waiting_for_all_mgt / waiting_for_zero_shape_scoring、phase_exit_codes空。

本轮新增工作是明确的同请求codec重建诊断，所有成功GPU/CPU原句柄已exit0；首启导入失败1及修正已完整记录，没有训练重启或绑定源码改动。已有agent等12:10只查一次；本机checkpoint/H5/NPZ下载0，GPU仅云端。继续完整VQ300k→ABC40k→五候选真实text-only val选模→正式20rep main/MM；整体SOTA及InterGen/InterMask/TIMotion/CoShMDM全目标仍未完成，不以重建对照替代生成比较。

**11:44 SGT：InterMask原生VQ统一480重建对照完成。** 复用已有官方源码/公开VQ ep46/iteration20608与原导出器，InterMask重建shared InterCLIP FID.985963612；当前HHC105k中间快照2.603228230，相差1.617264618。二者同冻结480/GT-text embeddings/evaluator/逐请求编码/96检索batch/metric seed20261006/91450 person-frames，GT/text maxdiff0。InterMask Matching3.793553988、R .427083333/.5875/.689583333均优于HHC105k点估计；Div7.592753887相对GT7.741262436差距大于HHC7.711113453。这里是GT motion重建，不能替换此前InterMask文本生成FID8.931384，亦不表示HHC text-to-motion输赢或SOTA。

InterMask重建十项：坐标RMSE78.177mm；Full22-joint MPJPE109.967mm；Body21-nonroot110.963mm；Hands MPJPE不适用；root89.063mm；Full joint FID待评；Body joint FID待评；Hands joint FID不适用；滑步3.212800mm/frame；关节距离接触代理F1 .586442786（semantic/mesh contact仍待评）。对照HHC105k分别为RMSE285.025/Full269.304/Body270.111/root252.367/滑步12.403673/代理F1 .670726166；HHC代理F1更高。GT参考滑步4.258247mm/frame；较低滑步不自动证明整体物理质量更好，未估跨codec差异CI。

此比较覆盖表征、量化及坐标转换整条路径，不能孤立归因某一模块。模型结构、输入及训练合同不同：Mask用normalized262D、现有导出器zero-pad到ceil(L/4)*4再trim；HHC用raw SMPL + requested GT betas、repeat-last raw pad及官方转换。released Mask与HHC中间105k不是等预算或等条件对照，iteration数字也不代表相同训练成本；重建FID差不是理论生成性能界。原VQ300k和A/B/C/shape/text-only队列保持，真实生成候选出来后再据证据决定改表征/量化/转换，不直接换baseline码本冒充HHC改进。

GPU成功原exec86749 exit0/peak1.165411GiB；shared CPU62731与geometry33974均exit0。首启误用依赖路径缺torch_geometric在模型导入前exit1，确认输出目录为空后改用已有/opt/hhc-baseline-deps:/opt/hhc-vq-deps完成；源码/模型方程未改，失败及环境路径已记入export_receipt。现有公开VQ权重af57bb...与四份官方源SHA逐项通过、strict state加载。H5 99539788B/SHA379e0fab6f2e5194e495beff428ee1584ca8452d34660e1689b52023f961d580；权重/H5/embeddings均留云。29绑定源码前后保持，未新增训练分支/Pod或重启原训练。

只收3个JSON共9988B，与云大小/SHA逐项一致，本机output/hhc_sota_20261016/intermask_vq_official480_codec；acceptance明确限制。合并原始证据intermask_hhc105k_official480_codec_evidence.json 28363B/SHA12174bfc2e9f9ead638b462948b4f1985b8837afc26d9cc9cb59334ee34b9a1d，[Drive证据备份](https://drive.google.com/file/d/1zP6l5LZ1E_Z4NKPJ-zRqQpGfl3FBCbHs/view) metadata大小/父目录读回一致、云checksum未暴露；收据同名.receipt.json。无本机checkpoint/H5/NPZ下载，不是权重备份。原agent仍等11:50单次普通核查，goal active，正式生成比较与SOTA仍未完成。

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

**10:13 SGT：105k新最佳及官方480复核进行中。** 10:10一次云SSH exit0确认六原PID live，VQ107201/update108200，latest/best105000 FID2.5744459359（旧65000/2.6810599747）；Matching3.807006137、R .345833333/.525/.622916667、Div7.757760525/GT7.715767384、native22含root Body104.842863mm。seed123/native fixed_val_first480_center_max300_raw_smpl_v1/480，非生成或joint FID。100k→105k FID/Matching/Body改善、R1持平/R2R3回退/Div差距收窄，不称全面改善。

云CPU freeze exit0，29绑定源实际SHA全匹配；新中间spec output/hhc_vq_intermediate_frozen_best105k_20261016/frozen_vq_spec.json SHA7baf4ae7a0e62dfd915d1548daf004f7344f181c4eb1ad48c497f4dbfa14aa38，冻结105000 checkpoint115069874B/SHA81314b813fa95564ca705bba6f5322aacd705610810fbe85eeaa3a752536e672；scope intermediate_val_selected_pilot，不替代完整300k final freeze。GPU可用15551MiB后启动已有exporter，原exec session9695/child13710正在官方冻结480重建，output/hhc_native_vq_official480_best105k_20261016；目前96/480，恒等坐标最大.004701mm。原句柄未结束，不重跑。

105k十项RMSE/Full/Body nonroot/root/Full-Body joint FID/滑步/Contact F1待本次统一评；Hands MPJPE/FID不适用。后续复用已有CPU shared/geometry及60k→105k paired工具，不新增算法分支。仅云端GPU，所有ckpt/H5/embeddings留云；本机新增0。InterGen30100/best20000 FID24.314143、三个queue仍waiting/phase空。继续原300k→ABC40k→真实val选模→正式main/MM，goal active。

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

75k十项：RMSE待评；Full MPJPE待评；Body non-root MPJPE待评（原生22含root监控不替代）；Hands MPJPE不适用；root待评；Full joint FID待评；Body joint FID待评；Hands joint FID不适用；滑步待评；Contact F1待评。最近完整官方冻结480仍为60k/shared FID2.577503和对应十项，不移作75k数值。

因75k验证回退，只读核查metrics完整222816B快照：391个train/15个validation/3个ready/2个resumed，非有限值0、malformed0；55–75k最近train原始行的loss从.086821到.080875，lr/perplexity/components均有限。未发现已记录的数值故障，不等于全面参数检查、泛化或原因判定；单批loss不可作训练趋势统计。原有ready/resumed为历史记录，本轮重启0。此前口头提取把20个非train事件都概括为validation不准确，以完整快照分类为准。manifest running/last75601/update76600/best65000，contract16b75e...保持；不因单次FID反弹重启、改源码或新增分支。

70k346B/SHA23e85552d81f38e9c95dac03a0d55b1bb1f55510f870ae9c562e923436eb62df；75k320B/SHAcc500b6653416f4b56a045739a2829d33684609091b820575d19d99a50f130dc；日志审计4652B/SHA5d42a9133e60bf8debe9662874656df934348326053d09a04f6fbc857a1d62ae，均与云大小/SHA匹配。合并小证据vq70000_75000_validation_and_log_evidence.json在output/hhc_sota_20261016。只收JSON，无GPU新增工作/模型或数组读取/ckpt下载，75k原观察exec71862 exit0。

08:10:49真实/proc复核六PID2353/5690/5729/7178/8321/9975 live、VQ74801/InterGen26950（best20000 24.314143）、GPU12%/8533MiB、三个queue waiting/phase空；随后75k完整节点和日志manifest继续推进。下一次普通>=08:21。继续原300k→ABC40k→五项真实text-only val→20rep正式main/MM，goal active，尚无生成SOTA证据。


**07:39 SGT 新完整65k训练内验证：** seed123/native fixed_val_first480_center_max300_raw_smpl_v1/480，latest/best65000 FID2.6810599747（旧best60000/2.6950839104）；Matching3.807880656改善，但Body native22含root108.376098mm较60k107.325754变差，R1/2/3 .345833/.506250/.610417均回退，Div7.754404/GT7.715767差距扩大。不称全项改进、生成或SOTA。334B JSON SHAb4b82bee1b8359c1942bcd118a96204d9651f2825c63907db9d3bf3963bcbe1e已与云只读SHA核对；本轮仅取小JSON，无GPU新增工作/ckpt下载/训练源码修改。

65k十项：RMSE待评；Full MPJPE待评；Body non-root MPJPE待评（native22含root监控不替代）；Hands MPJPE不适用；root待评；Full joint FID待评；Body joint FID待评；Hands joint FID不适用；滑步待评；Contact F1待评。最近一次完整官方冻结480是60k/shared FID2.577503及对应十项，不能移作65k结果。

07:38实际/proc和两次SSH exit0确认六PID2353/5690/5729/7178/8321/9975 live；VQ66001/update67000，InterGen25900/best20000 FID24.314143，GPU100%/8533MiB，三个queue保持waiting/phase空。下一次普通检查>=07:48；继续原300k→ABC40k→五候选真实val→20rep正式main/MM，无新增分支/外围代码，goal active。


07:27 SGT 60k官方480源码/小结果12payload归档migration/hhc-vq60k-official480-20261016.tar.gz，273921B/SHAd581165dd927a9b682c84b7f0677ad37cc38db63f3f571a566fec52eba1efd31，逐payload和tar读回通过；Drive1kjj4IdKdmj0KfabKGQX7aRGSEYuLN1QB metadata大小/父目录一致、未暴露云checksum，收据同名-backup-receipt.json。无checkpoint/motion H5/NPZ embedding/prior数组，不是权重备份；包中是完成前文档快照，本条完成说明另存。所有本轮CPU/GPU原句柄终态exit0、不要重跑。正式导出/主-MM评分/汇总/存储路径已备齐；除具体故障修复外，未有真实生成候选前不再追加外围代码，优先等待原300k训练与后续生成消融。07:24原六PID live/三queue等待，下一次普通检查>=07:34；本机ckpt新增0，29源不改，goal仍active。


**07:25 SGT 60k官方冻结480重建完整验收：** shared InterCLIP FID2.577503（35k2.648528），native训练内latest/best60000 FID2.695084（旧best35000/2.827235）；两种请求/crop协议分开。Full MPJPE283.358mm、root267.635mm、滑步12.514mm/frame的均值较35k变差；几何四项paired CI均跨0，未证明改善/稳定恶化。Matching/Div与GT差距未改善，不称整体成功。是GT motion+requested GT betas重建/额外25k更新，非等预算算法、text-only生成或SOTA。

freeze、GPU export54186/12096、shared CPU32269、geometry48271、paired43843均exit0；GT/text embeddings与三基线逐项相同，29绑定源保持，6个小JSON共12703B与云大小/SHA逐一一致，acceptance.json在output/hhc_sota_20261016/vq_official480_best60k。所有weights/H5/embeddings留云，本机新增0。完整十项与35k→60k配对区间见主报告新节；Hands N/A、Full/Body joint FID及semantic/mesh contact未评不补造。

07:24一次一返final的普通监控已正常返回：六PID2353/5690/5729/7178/8321/9975 live，VQ62601/update63600/latest-best60000 2.695084；InterGen25450/best20000 24.314143，GPU98%/8533MiB，三个queue仍waiting/phase空。下一次普通>=07:34。不追加训练分支，继续完整300k→ABC40k→五项真实text-only val选模→20rep正式main/MM；goal active/未达成SOTA。


**07:20 SGT 60k新最佳及官方480验收进行中：** native seed123/fixed_val_first480_center_max300_raw_smpl_v1/480的step60000重建FID2.6950839104，best刷新为60000（旧35000/2.8272352923）；native22含root Body107.325754mm、Matching3.810742760、R .358333/.510417/.622917、Div7.746598/GT7.715767。R2/R3及Matching较55k变差，不能称全项提升；非生成/SOTA。新348B JSON SHA26cc99f51ce9c01c4033201949298f45e35a338046c932b0a569ed9495cd4031。

云端中间spec output/hhc_vq_intermediate_frozen_0717_20261016/frozen_vq_spec.json，checkpoint115069874B/SHA3df73e015053f1c7a625a1eba8ae3a39f0edfc053c5875035cd74874719c214c，scope仍intermediate_val_selected_pilot，不替代300k final freeze。原GPU export exec54186/child12096 exit0，output/hhc_native_vq_official480_best60k_20261016/reconstructions.h5 95130020B/SHAb6b709f2125663ffe8e45ea6c9bba60e6c7710ee9840b771b88ca250e0ce0e49；恒等坐标max .004701mm/接触365800值0不匹配，冻结权重保持/GPU peak.127GiB，全在云端。主训/29绑定源不改，实测15551MiB GPU free后才导出。

**当前三个云CPU原句柄：** shared InterCLIP exec32269、geometry exec48271、35k→60k paired exec43843，均待收终态，不重复启动。新compare_hhc_vq_training_checkpoints_cpu.py SHA0f6cb32e3cad4b46e8186730c645f72866d307379eee6938b5685fb55cfb9fc2，校验两spec/H5 SHA、同source_run/training contract/normalization，计算metrics/10000次bootstrap AST与原工具相同；额外25000训练步，不是等预算新方法。60k十项RMSE/Full/Body nonroot/root/Full-Body joint FID/滑步/contact待统一评，Hands MPJPE/FID不适用。共享FID不能替代joint FID；原始GT motion+requested GT betas重建不可作为text-only生成。

07:13:48主线程异常接管直接/proc确认六PID2353/5690/5729/7178/8321/9975 live，VQ59801/update60800，InterGen实际STATUS.json=25150/best20000/FID24.314143，三queue waiting/phase空；下一次普通>=07:24。此前多节点监控的中间报告未送达主线程，已仅中止观察agent等待，没有停止任何云训练。InterGen无training_manifest.json，应读STATUS.json/training.jsonl；VQ validation glob需仅匹配validation_六位数字.json，排除validation_panel.json。初次接管读取路径失败已补齐，绝不能误判训练失败。后续例行检查一次一返final，不再依赖中间send_message。


**06:58 SGT 最新完整55k覆盖：** native seed123/fixed_val_first480_center_max300_raw_smpl_v1的VQ55000重建FID2.8667573289，best仍35000/2.8272352923；Body native22含root108.769865mm较50k107.314958变差，Matching3.810460与R1/2 .352083/.525也变差，R3 .627083略改善，Div7.719928/GT7.715767，不能称全项进步。55k十项RMSE/Full/Body nonroot/root/Full-Body joint FID/滑步/contact待统一评、Hands MPJPE/FID不适用；不要挪用35k官方480几何数值。新334B JSON SHAd0004d4abc73e3a724b936a214b3254bde33334130e616d9fd9c8636f8c84d1b与云一致。节点观察/proc确认原5729 live且exec立即exit0，未重复启动。

06:56普通实查六PID2353/5690/5729/7178/8321/9975 live、VQ54801/update55800、InterGen24600/best20000 FID24.314143、GPU26%/8533MiB，原三个queue仍waiting/phase空；下一次普通监控>=07:06，55k节点验收不要求补查训练。完整300k→ABC40k→五候选真实val→正式20rep main/MM仍未完成，goal active。

06:58 MM归档源码/小验收11payload包migration/hhc-formal-mm-archive-20261016.tar.gz，271243B/SHA11309072e15c098485bec35540f1623ab792787d9602e9ad263231d764452814逐payload及tar读回通过；Drive1YN0cuon7PbG_Y0E6p3fmlx6hwxd1K1qv metadata大小/父目录一致、未暴露云checksum，收据同名-backup-receipt.json。无checkpoint/motion H5/NPZ embedding/prior数组，不是权重备份。归档含完成前文档快照，55k JSON及本条完成说明在包后新增；本机新增ckpt0，29绑定源不改，不追加研究分支。


**06:55 SGT 正式MM归档路径验收：** 新tools/score_hhc_text_only_formal_mm_cpu.py只在真实五候选val选模receipt核验后做云CPU formal MM，要求相同MGT/VQ/shape/sampling及每prompt固定体型。它保留原score_fair_interhuman_mm.multimodality_from_embeddings的同样100×30×D官方encoder输出、seed1234+rep/pairs10/scale6/divisor2，并保存float32 NPZ及SHA。归档保持encoder返回顺序；没有为了对齐seed改变旧wrapper排序。新score schema interhuman_fair_mm_score_v2可由既有四方法汇总接入，显式消费先验score/H5/独立NPZ哈希；旧v1没有archive仍保留H5。未执行真实文件回收/正式test。

云CPU exec57685 exit0、32项合成集成检查：原160score汇总/CI/保护，真实MM H5请求布局与mock encoder100次，保留矩阵逐位相同、原官方MM算式完全一致、Numpy RNG保持，错误winner/GTshape/跨repeat改体型/模型SHA拒绝，归档损坏/错维度/H5别名拒绝，v2仅删自建fixture H5后仍能从NPZ复算MM；所有fixture删除，真实生成删除0。**InterCLIP encoder和真实选模receipt在此测试中mock，不能当实际HHC test或质量验收。** 已通过测试不重跑；2683B audit SHA0a724af3a2d26f85ff27fc7124e8726a702650dc9bf976237dcf3e32504fa548与云一致。新scorer SHA575bf2d149fa17c6579c6ebedd420b5ddf430784f3198dabcec6ad6a54e08651，future score adapter SHAa3a5fb44691090961b811bc8f8458570fec293fd0f518f8a49bb3351db6ad0c9，旧393c4f...源码快照保留；29活跃绑定源不改。hourly hhc已app更新并TOML读回ACTIVE、原target/rrule/禁本机ckpt下载保持。小验收mm_embedding_archive_acceptance.json。


06:47 SGT长度/时间诊断及50k小结果7payload归档migration/hhc-vq-length-time-diagnostic-20261016.tar.gz，260474B/SHAdbf63f010b05e60c3975aba7de5763d87fc1fa12e54605ce6944aeef59dff00b，逐payload/tar读回通过；Drive1AxIkZYTtCvK1kQT3OsFl1V_zz081_LHr metadata大小/父目录一致、未暴露云checksum。无weights/motion H5/embedding/prior数组；非权重备份，包内是完成前文档快照，本条完成说明与收据另存。06:46普通实查六PID live，VQ52401/latest50000 FID2.925748/best35000 2.827235，InterGen24250/best20000 24.314143，GPU99%/8533MiB；原三个队列phase空/仍waiting，v2为9975。下一次普通检查>=06:56，不改29源/不新增本机checkpoint。


**06:44 SGT root/长度诊断完成：** 云CPU exec46787 exit0，仅读已有5k/35k官方480重建，无模型加载/GPU。35k Full268.367mm、root251.220mm与已验汇总相差<1e-7mm；每帧剔除自身root误差后22joint残差75.332mm（GT oracle诊断，不能修正输出或作主成绩）。初帧root196.530mm、末帧277.063mm；归一时间四分位220.277/243.959/257.663/283.980mm，说明初始空间误差和时间累积均存在。按长度14–63/64–127/128–191/192–255/256–299分组的root为207.661/285.083/238.992/313.805/231.918mm，组间非单调且动作不同，不能宣称只扩训练窗口即可解决。新10,075B JSON与云SHA9f1dbd3f85bf0722a6664b418cfc018c11a62f8a2e954bbc322ca2f8198df706一致，29源不变，本机ckpt/H5/embedding新增0。保留原ABC/形状生成消融，暂不扩分支。


**06:40 SGT 50k完整验证与备份覆盖：** VQ最新完整50000重建FID2.9257481069，best仍35000/2.8272352923；native22含root Body107.314958mm改善，Matching3.809512/R .358333/.529167/.625000/Div7.731749 GT7.715767，非全指标改进。seed123/native fixed_val_first480_center_max300_raw_smpl_v1；其十项RMSE/Full/Body nonroot/root/Full-Body joint FID/滑步/contact待统一评，Hands MPJPE/FID不适用。不能混用35k官方480重建的完整几何指标，不能当生成/SOTA。334B小JSON SHA1d763b79bb1dc8398f7bf598e002a91502883c79c67e520bd5ba954a7bdd1f73与云一致。

本轮35k官方480源码/小结果13payload包migration/hhc-vq35k-official480-20261016.tar.gz，263587B/SHA401ab03910629a81578617fd41120b26e0ffb5f7bbf9637b490e36c0edf1eb3e，逐payload与tar读回通过，Drive1sKH5cp84IUcitutBohevAglkHLrWOSLJ metadata大小/父目录一致、未暴露云checksum；收据同名-backup-receipt.json。无checkpoint/motion H5/embedding/prior数组；不是权重备份，归档含完成前文档快照，50k JSON/本条完成说明在包后新增。原29源/训练/三个队列保持，普通监控仍>=06:46，不需因这次artifact验收补查PID。


**2026-10-16 06:40 SGT 最新覆盖：** fresh VQ35k中间快照在官方冻结480请求上完整重建验收：InterCLIP FID2.648528、Full MPJPE268.367mm、root251.220mm、滑步12.349mm/frame；对5k多训练30k步，位置/滑步paired CI不跨0，contact F1 .632373改善CI跨0。是GT motion+requested GT betas重建，不能与文本生成基线8.7比较胜负，非SOTA/最终VQ选模。5份小JSON共12151B逐项对云大小/SHA一致，本机ckpt/H5/embedding新增0，29绑定源码保持。完整十项/限制见主报告新节。

06:36普通实查六PID2353/5690/5729/7178/8321/9975 live；VQ49801/update50800，最新完整45000/FID2.940790、best35000/2.827235；InterGen23950/best20000/24.314143，GPU18%/8533MiB。7178等完整300k、8321等全部MGT、9975 waiting_for_zero_shape_scoring/phase空，原9549仍superseded不可重启。下一次普通监控>=06:46；当前没有真实final40k选模receipt，不提前formal test。

CPU训练覆盖审计：6022 train IDs中4567满足当前30<=rawN<1000；排除1314条<30、140条>=1000、1条缺失。15–29帧930条是潜在数据覆盖改进线索；长度单位需与官方处理后T-1及4帧token换算。当前不改缓存/源码/训练，不把覆盖统计当质量收益。


06:25 SGT增量包`migration/hhc-native-baseline-formal-preflight-20261016.tar.gz`297075B/SHA8e37784f43aab7a1db526afbd126444b4b5cb738ec74f0bb89fd04ab64360125、33payload已验，Drive1J3JeNmjQkZWCZGMiPKSl1t-kSUQ-i57z metadata大小/父目录读回一致，云checksum未暴露；收据同名-backup-receipt.json。包内无权重/motion H5/prior数组。新baseline模型加载须沿用真实CPU_final验收：Gen/TI全state strict，Mask VQ strict，transformer仅冻结预训练clip_允许缺键、其他缺键拒绝；不得把放宽非CLIP键当成功。已完成的所有本轮CPU/GPU测试不重跑；原训练/7178/8321/9975继续，普通监控下次>=06:32。

**06:23 SGT 最新45000覆盖：** latest重建FID2.940790、best仍35000/2.827235，native22含root Body108.947519mm较40k变差；Matching3.809457/R .358333/.533333/.616667/Div7.735402 GT7.715767，R3回退。协议seed123/native fixed_val_first480_center_max300_raw_smpl_v1，非生成/SOTA，十项缺项待评/Hands N/A。348B小JSON SHA84c08671bcc403334ec6ba5beb5a0986f6747e1bae1299f1f0b51c642b6b4a4e已对云。06:22普通实查六PID live，VQ45801、InterGen23500/best20000 FID24.314143，GPU90%/8533MiB、三个queue仍waiting；下一次普通>=06:32。hourly automation已新增经过验收的native baseline/新汇总接口，原target/rrule/9975/29源限制和禁本机ckpt下载保持。当前所有CPU/GPU baseline验收handle均已exit0，勿重跑。

**06:20 SGT exec45990已整体/三child exit0，勿重跑：** 三基线真实云GPU validation first/min/max19/299/14帧×同seed重复；9case finite/bitwise重复、CPU-CUDA RNG保持；Mask16/296/12、Gen/TI原L，峰值1.2001/.8206/.9136GiB。无GT motion/shape、无test/评分、不存motion/ckpt。三份CPU_final及GPU_smoke关键源/模型/采样一致，最终exporter SHA0d88373666edf68692ad688a378bf7585d6f19fad343baab0025a49d8699d6ce；9小JSON大小/SHA对云一致、29源保持。acceptance=`output/hhc_sota_20261016/native_baseline_formal_acceptance.json`。33/16项CPU检查及真实Mask480评分parity均exit0，不重跑；真实生成文件删除0。finalVQ spec仍不存在，无真实formal选模receipt，不提前正式生成。新源码/小结果正在本轮Drive增量备份，旧包不含这些文件；模型权重不在包内。

**06:15 SGT 最新覆盖：** VQ最新完整40000重建FID2.924769，best仍35000/2.827235；native22含root Body108.307099mm改善，但Matching3.810695/R1 .341667变差，R2/3 .516667/.625000、Div7.754080/GT7.715767。非生成/正式test/SOTA；完整十项缺项/Hands N/A须保留。06:11普通实查VQ43601、InterGen23150/best20000 FID24.314143，六PID2353/5690/5729/7178/8321/9975 live，GPU89%/8533MiB，下一次普通>=06:21。

新baseline正式入口`tools/export_interhuman_native_baselines_formal_cloud.py`、`dataset/interhuman_native_baseline_h5.py`、`tools/score_interhuman_native_baselines_cpu.py`保留native长度（Mask floor(L/4)*4），formal生成前必须有真实HHC val选模receipt及本method最终CPU验收`output/hhc_sota_20261016/native_formal_{method}_strict_cpu_final.json`。三最终CPU child/总exec83084 exit0；Mask只有CPU构造分配被重定向，GPU走未改官方代码。33协议检查/16合成汇总消费测试及真实Mask480新scorer回放全部exit0（80877，FID误差2.6e-7/GT text完全相同），无test评分或真实formal比较。原29源码不改。**当前cloud GPU val-only sampler smoke exec45990**，Gen→TI→Mask，固定验证请求first/min/max、每个seed重复、仅小JSON；需继续收取原句柄终态，不重启。其权重/实际motion均只在云、不下载。

新的`aggregate_interhuman_text_only_comparison_cpu.py`消费HHC新shared-main/fair-MM及三baseline新score格式，要求20main+20MM/方法、相同模型/官方eval/GT text及选模receipt；旧aggregate_fair_interhuman_metrics.py不是这套新score接口。`consume_interhuman_formal_generation_cloud.py`仅供后续明确执行：正式score+embedding+H5 hash验证/持久回执后回收自己的单个H5，HHC MM没有embedding保存时拒绝；未删任何真实生成文件，不能将临时fixture当实验成绩。CoSh仍仅论文/发布审计对照，不能复制InterGen。

05:50 SGT增量包`migration/hhc-formal-text-only-v2-20261016.tar.gz`263941B/SHA984c395b6965167c45ef720846e0ca8598c3b963f42e0d42c8a73f7acb797a89、17payload已验；Drive1SJO69YpKYFNzyud0Px95ulzfp6upwZLZ metadata大小/父目录读回，云checksum未暴露。只含新源码/小结果，不含权重/H5/prior数组；收据同名-backup-receipt.json。当前9975/v2与原训练保持，goal active/未达到SOTA。

**05:48 SGT 新完整VQ覆盖：** latest/best35000重建FID2.827235、native22含root Body109.042825mm，Matching3.810415、R .347917/.502083/.622917、Div7.709463/GT7.715767；R1较30k下降，非全项获益/生成FID/SOTA。真实manifest running/main37001/update38000，29绑定源SHA均保持，35k小JSON与云SHA0769af45c850c07d0a318b656b0e1e06afe0a771cde885d6e123397ba8ec7f74一致。完整十项缺项/Hands N/A见主报告。新driver9975为当前体型队列，旧9549已被替换且不可重启；formal仍无真实选模receipt、无GPU生成。小验收`output/hhc_sota_20261016/formal_text_only_v2_acceptance.json`。

**05:46 SGT 最新接管覆盖：** 原train-prior等待器9549存在H5 attrs.metadata嵌套JSON的读取缺陷；原9项mock遗漏完整成功链。独立v2已14项云CPU验收，原28文件不改；守卫实查9549 waiting/phase空/无child和输出后仅终止该等待器，/proc消失，旧STATUS=superseded。新 **9975** live/waiting，cloud control `/workspace/HHC/output/hhc_train_shape_val_v2_control_20261016`，本机配置`migration/control/hhc-train-shape-validation-v2.json`，云`migration/hhc-train-shape-validation-v2.json`，SHA a7494e3ef9d4135b8c78c23064470df79906125683b57804560c072d3d8b3857，29源=原28+v2。**不可重启9549或旧queue**，原2353/5690/5729/7178/8321命令守卫前后一致。上游仍8321完整12phase；下游A两种prior输出路径沿用原计划，尚无GPU child/质量结果。小验收`train_shape_queue_v2_cpu_audit.json`与`train_shape_queue_v2_handoff.json`在`output/hhc_sota_20261016`。

新正式入口`tools/hhc_text_only_formal_contract.py`与`tools/export_hhc_text_only_formal_cloud.py`已独立完成CPU协议预检18项（exec1613 exit0）；只允许zero/train_mean/empirical_train_pair。**当前没有真实selection receipt，不启动formal GPU/test**：先五个预登记final40k实际480val及GT/text parity，按冻结min-val-FID选模，再核对MGT/VQ/真实generation/evaluator/源码哈希；新入口在GPU加载前强制验收。MM每prompt固定体型，30次只变motion seed；正式目标20×1056/main及20×100×30/MM，合成rep0格式测试不等于真实模型端到端或质量。后续不能用旧训练CLI GT-shape formal成绩冒充pure text-only。

05:42低频检查原训练live，VQ34801、最新完整/best30000重建FID2.870298，InterGen22250、最新完整/best20000 FID24.314143。下一次普通原PID轮询>=05:52（新9975按05:44:47接管计>=05:55）；35k完整artifact可按验收节点观察。hourly automation已更新/读回9975/v2/29源，原schedule/target保持，本机仍禁新增ckpt。

**05:32 SGT 普通复核覆盖：** 原2353/5690/5729/7178/8321真实live；VQ已32801、最新完整/best30000重建FID2.870298；InterGen21950、最新完整/best20000 FID24.314143；GPU100%/8533MiB。30k后的原训练已继续，不能误判验证结束为任务退出。原训练下次普通检查>=05:42；9549最后05:29 live/waiting，下次普通>=05:39。JSON2364B/SHA d79653d7ad48695d1aac10c05cac45325cf47373739b80a01b888f8716f940bf已Drive ID14aJkkl8W8HzTLVSDDk2bpXa-co1P8dqA并metadata大小/父目录读回，收据`output/hhc_sota_20261016/vq30000_storage_protocol_backup_receipt.json`；无权重、云SHA未暴露。后续重点保持训练与原队列，未有真实MGT生成结果前不再追加训练分支。

**05:30 SGT 最新验收：** VQ30000训练内完整480val已完成，观察exec10152 exit0，latest/best30000 FID2.870298/Body22inclroot110.311079mm，R .370833/.500000/.600000、Matching3.812147（非全优），原seed123/native中心crop协议，不是生成FID。证据`output/hhc_sota_20261016/vq_rebuild_validation_030000.json` SHA cf852333ef1696be1eecc606ea74e3d3dcfc10e182cd10bae2b17756f93f0939。05:21原PID组live、VQ29801/InterGen21600，原InterGen最近完整/best仍20000 FID24.314143；05:29新9549已/proc复核live/waiting、无child phase。原训练正常轮询可在05:31后，新9549下一次普通检查>=05:39，无需补轮询。

正式协议已读回metadata：main20×1056；MM20×100×30，MM→main manifest hash一致，无test评分。**切勿直接以原train --formal-test生成的requested GT-shape成绩冒称text-only；目前独立zero/prior exporter仅val。等真实val冻结模型/策略，再独立实现formal text-only及MM latent-shape重复协议（包括重复identifier处理），不改28锁定源，不提前test选模。** 新小证据`output/hhc_sota_20261016/vq30000_storage_and_formal_readiness.json`记录未完成门限。持久卷配置80GB，当前/workspace du22831495680B；共享df不代表pod配额，不需扩容/清理。hhc小时prompt已去旧阶段日志，保留三个等待器/严格恢复/源绑定/云GPU与禁止本机ckpt下载，已读回ACTIVE及原target/rrule。

05:19源/小结果包`migration/hhc-train-shape-prior-source-20261016.tar.gz`311007B/SHAad28c32889448f3100ae75edd26da1ef50dd0a9405c62e3556f5e7f82a3c8a54（41 payload，无ckpt/H5/prior NPZ）已传Drive ID1V9xyUYzwmuMVbErFk3TxK9McKrp89EJ1，metadata大小/父目录读回；云SHA未暴露。hourly hhc已更新9549句柄与28源码限制并TOML读回ACTIVE、原target/schedule不变。不要把当前排队/CPU验收当实际生成质量完成。

**05:17 SGT 新优先覆盖：** 训练体型先验CPU统计已完成，4567训练配对/1556种不同pair，zero的逐维std缩放中心距离9.043位于98.423%分位（非严格OOD结论/FID收益）。先验云`output/hhc_train_shape_prior_20261016/{prior_spec.json,train_shape_pairs.npz}`，145147B NPZ未下载，train-only数组/manifest/缓存source全部绑定。新增helper/exporter及真实CPU FK/no-GT-H5/请求只读id-seed/全局RNG/480顺序分块检查exit0；9项队列mock exit0。

新queue PID9549已确认/proc live/`waiting_for_zero_shape_scoring`，config本机`migration/control/hhc-train-shape-validation.json`、云`migration/hhc-train-shape-validation.json`，configSHA c852d16bb747b2f6b37304515ffd83d377edf181c42521e2142ee3d7e8f17386。control`/workspace/HHC/output/hhc_train_shape_val_control_20261016/STATUS.json`。它等原8321全部12phase完成后，只对A_historical final40000作train_mean/empirical_train_pair两个单因素val导出+共享CPU评分，以原A_zero作控制，核对同MGT/VQ SHA及GT/text embeddings逐项相同。**28绑定源不要改，不重复launch，不把尚未生成的候选当质量收益**。父训练/原两个队列继续，无新pod/本地checkpoint下载。验收JSON`output/hhc_sota_20261016/train_shape_prior_acceptance.json`。后续先看实测FID/R/Div及物理质量，再决定是否推广其他组及冻结独立test/MM；本轮没有做formal test。

最近普通状态05:10：VQ26801，最新完整及best25000重建FID3.143828，Body22含root115.774mm（较20k112.941变差，非全优）；InterGen21250，最近完整/best仍20000/FID24.314143；旧三个driver5690/7178/8321均活跃。下次普通检查>=05:20，新增9549刚于05:15单独验收，不需重复轮询。

05:05小证据包`migration/hhc-coshmdm-release-audit-20261016.tar.gz`281959B/SHA986a301a70defadfbe4b54454c13ee9b96fffd597551166796ca196bc5d41b9e（28个payload，含当前报告、源码和小JSON，无权重/H5）已Drive备份，ID18eazyoYGb6bKLhKVRXU_Z4DhjYG6XohU，metadata大小/父目录读回。收据`migration/hhc-coshmdm-release-audit-20261016.receipt.json`；不宣称云checksum或checkpoint云到云备份已验收。

**05:03 SGT 公平比较新证据：** 当前CoShMDM官方main=c5116ab3，源码稀疏快照`migration/public-source/coshmdm-c5116ab3`（无ckpt）。CPU source审计exit0，11个InterGen参照git blob匹配官方b1750df，9个models AST经显式标识符改名均相同、配置仅NAME/路径不同且下载ID相同。云CPU审计exec47681 exit0：现有InterGen checkpoint346 tensors中缺CoSh重命名所需170个bert键，含26个可训练text encoder/LN键、11,029,504元素；仅key检查，没有完整模型load/推理/GPU。公开infer strict=False可漏加载训练过的文本层，eval strict=True则命名不兼容。不要把忽略缺键运行或改名后的InterGen当作独立CoSh复现；论文4.801单列，暂无官方独立权重。机器收据`output/hhc_sota_20261016/coshmdm_release_acceptance.json`，新工具`tools/audit_coshmdm_checkpoint_names_cloud.py`；23锁定源码未变。此问题只限制CoSh独立统一复现，HHC训练继续，不是整个goal blocked。

最近普通监控04:59：原2353=20950，VQ5729=24801，最近完整/best仍各20000/24.314143与20000/3.244720，两个等待器7178/8321 live/waiting。下次普通监控>=05:09；25k正式验收节点可单独读完成artifact，但不要把训练内val当生成FID或重启作业。

本轮04:53证据包`migration/hhc-canonical-pilot-20261016.tar.gz`529192B/SHA4ea275985b668912bc181cdd540ad4c952f9899d889a36676181c03dc40a13b6（136源码/小结果，无权重/H5）已Drive上传并folder listing+metadata大小/父目录读回，ID1dHr2-xUKMqvTvI6E_gxx2y0d2fk4Oxzx。收据`migration/hhc-canonical-pilot-20261016.receipt.json`；云SHA未暴露，云到云checkpoint备份仍未验收。

**04:50 SGT 最新覆盖：** canonical-pair额外3×joint SmoothL1短pilot及全480导出/共享评分/geometry/paired bootstrap全部exit0。相同base5000、1000 decoder更新/seed20261016及完全相同训练batch下，候选Full MPJPE306.399 vs control358.891mm、root285.135 vs338.587mm；但FID4.312691 vs4.201122、滑步14.916 vs13.771mm/frame、geometry contact F1 .564069 vs.600663。**不晋升、不改主VQ/ABC/text-only队列**，不用位置改善替代总体质量；下一步重点是已排队的CFG/window生成实验，不继续凭oracle或少数指标扩大早期损失试验。新源码`utils/hhc_canonical_pair.py`、`tools/audit_hhc_canonical_pair_cloud.py`、`tools/train_hhc_decoder_canonical_pilot_cloud.py`和结果已保存；本地CPU证据账本`migration/record_hhc_canonical_pilot_20261016.py`exit0，8个小JSON对云SHA全部通过，23绑定源未变。权重/H5仍全在云，接受记录`output/hhc_sota_20261016/decoder_canonical_pilot/acceptance.json`。各原始exec终态已收，不重跑同目录。

04:49低频状态：原2353仍训练20600，最新/best20000 FID24.314143；VQ5729已20601，最新完整训练内480/best20000 FID3.244720、Body22inclroot112.941mm（独立原生协议，非生成/SOTA）；5690/7178/8321均live。A/B/C等300k、text-only等全部40k，普通轮询>=10分钟。旧记录的15000/3.458571和“canonical未评分”均已被本段覆盖。

**04:22 SGT 最新覆盖：** 原VQ/2353继续，A/B/C7178仍等待300k。新的text-only后评测队列8321已核对/proc命令和STATUS为waiting_for_all_mgt，配置本机`migration/control/hhc-text-only-validation.json`、云`/workspace/HHC/migration/hhc-text-only-validation.json`，状态`/workspace/HHC/output/hhc_text_only_val_control_20261016/STATUS.json`。它等A/B/C全完成后各导出final40000的zero shape(主比较、无GT betas)及requested GT shape(附加条件对照)，统一CPU逐请求评分/seed20261006，与InterGen baseline GT/text embedding逐项相同才能通过。23源码绑定，6项CPU调度mock exit0；不要修改绑定源码或重复launch。GPU阶段尚未执行，正式test/MM另行冻结。

本轮真实decoder-only短pilot完整完成exit0（根VQ5000、两臂1000步、同batch256/seed20261016/orderSHA一致、encoder/codebook/condition不变）。额外`.01*global root geodesic`较等预算control变差：FID4.319167 vs4.201122，Full MPJPE365.921 vs358.891mm，滑步14.122 vs13.771mm/frame。候选不采用、不改主VQ/ABC；只在云留权重。GT oracle全局朝向诊断有效但不是推理方案，简单decoder边界重复也已负结果淘汰。详细十项、恒等坐标审计、固定模型paired CI在主报告及`output/hhc_sota_20261016`子目录。

下一项VQ方法应先核对可微的官方canonical joint几何与原官方处理的一致性，再考虑decoder-only相同预算试验；不要因oracle有效直接增大geodesic权重或把GT替换当生成。新MGT零shape代码已CPU导入，实际完整生成仍待40k训练权重。

**03:50 SGT 最新覆盖：** rebuilt MGT smoke exec74877 exit0，285.71M真实参数、2次更新、14/299/115帧生成、官方embedding有限、VQ未变，GPU peak5.46GiB；JSON `output/hhc_sota_20261016/rebuilt_mgt_smoke_005000.json`。A/B/C已云端排队，driver7178确认live/`waiting_for_complete_vq`，状态`/workspace/HHC/output/hhc_abc_after_vq_control_20261016/STATUS.json`，配置`migration/control/hhc-abc-after-vq.json`（本机）及`/workspace/HHC/migration/hhc-abc-after-vq.json`（云端）。不要重复launch，不修改20项绑定源码。等VQ完整300k后自动freeze最终val-best、依次A/B/C各40k，同VQ/seed，仅val；原2353保持。driver遇失败写确切exit并停后续组，恢复前核对原状态/latest而非从0回放。6项CPU mock队列检查exit0，只验调度安全不验训练质量。当前仍未达到SOTA；正式test/MM独立冻结后另启。

**03:45 SGT 最新验收：** VQ main5000完成480val：重建FID4.883522、Body MPJPE136.739mm，最佳5000。不能与生成FID比较。已云freeze至`output/hhc_vq_pilot_frozen_005000_20261016/frozen_vq_spec.json`，SHA554e6e3adf105bae838d9917271e5919ace0fa595b75be3c881ea60c608c76e5，selection_scope为intermediate_val_selected_pilot。真实全MGT smoke用此spec执行，output `output/hhc_rebuilt_mgt_smoke_005000_20261016`，主线程exec74877；已有两次更新后进入生成，待确切exit，不重复启动同output。VQ继续300k，A/B/C40k尚未启动；只收JSON，不下载权重。

**03:44 SGT 验收补充：** 首个main5000 VQ验证执行中，尚无完成指标；验证采用原循环/CPU官方InterCLIP，日志静默不是失败。`output/hhc_sota_20261016/cloud_vq_rebuild_recovery_audit.json`是真实main1000恢复包的CPU完整性验收，exit0，权重仍云端。新MGT入口已接CPU fp32 CLIP单前向缓存，batch1/8/37/128与旧两次前向逐项相同；cloud import预检通过。`tools/smoke_hhc_rebuilt_mgt_cloud.py`已上传，需先等首个完整480val及best_fid.pth，再freeze `--pilot-snapshot`，核对显存余量后运行。此smoke只验真实模型2次更新/长度边界生成/evaluator有限嵌入，不算质量评测。原2353验证复用同一GPU模型，先前“另需第二份完整GPU模型”的推测不成立，仍按实测显存安排共卡。

**2026-10-16 03:25 SGT 重要新覆盖：不再因旧185k缺失/优先级未答而把整个HHC停住。** 真实smoke证明VQ只需约1.8GiB，因此已在现有云4090共卡、nice10、GPU20%上限启动fresh VQ重建，无需暂停旧InterGen2353。driver5690/trainer5729，run`/workspace/HHC/output/vq_hhc_native_rebuild_seed123_20261016`；999warmup后update1000/main1已验收。initial4/resume6 exit0、模型/EMA/RNG/batch游标恢复验证通过。后续每5000 main步骤质量验证；300k主阶段，原损失，batch256/window64，W&B disabled，日志metrics.jsonl。准确恢复命令与环境见`migration/control/hhc-vq-rebuild.json`；只在旧trainer/driver终态确认后恢复，不从0重启，不原地改合同sources。本机仍禁新ckpt下载。

HHC MGT新接口`--vq-spec`明确新码本step/hash/origin，历史VQ185k模式仍默认；冻结工具`tools/freeze_hhc_rebuilt_vq_cloud.py`只创建云端不可变hardlink。中间验证模型只能显式`--pilot-snapshot`并标中间pilot，最终需完成VQ后val选模。A/B/C尚未启动，不宣称提升；MGT显存较大，启动时须保留旧InterGen20k验证的额外显存余量，不能直接因VQ共卡成功就假设三训同时安全。用户若答优先级再调整，当前没暂停旧长训。

最终验证小包`migration/hhc-sota-validation-complete-20261016.tar.gz`341706B/hash9e7526509a79e1e91d153afd6424f9a0334681ab9a9549685be67953412fe7c4已上传Drive ID11NsR9-diJbM0QOPFpIcEAtccXEoyot55并metadata大小读回，无checkpoint。三个基线/就绪smoke已完成，所有CLI终态已收；不能再次启动同路径。后续目标仍为真实HHC训练改进+独立test/MM，而非重复验证部署。

**2026-10-16 03:06 SGT 最新覆盖：** 三个baseline 480val已全完成，InterGen8.762189、InterMask8.931384、TIMotion Transformer+LPA8.721812（均非正式test/MM）；比较收据`output/hhc_sota_20261016/comparability_three_methods.json`证实GT/text embedding完全相同，长度/采样预算差异明示。VQ smoke已完成，三基线runner与smoke均退出，不重启。原长训2353继续，用户priority问题未答。HHC VQ185k未找回；若重训VQ须另立合同，不能伪称恢复旧185k。A/B/C候选代码/CPU检查完成，未测新HHC提升，goal未达SOTA，不可标complete。下一步先确认待答优先级/旧VQ云恢复，再启动真实HHC对照；不得重拉本地ckpt。

**2026-10-16 02:56 SGT 最新覆盖：** InterGen baseline4126已generation/score exit0，FID8.762189/R1 .425000；报告已本机收取。VQsmoke4282已完成4次batch256真实更新，loss/梯度有限、显存1.7913GiB，无ckpt，十项质量待评。TIMotion runner4780/GPU child5047现运行，云端`output/interhuman_baseline_restore_20261016/timotion/STATUS.json`，保持同锁不要重启。唯一待收基线为TIMotion完整结果；HHC185k仍缺，不能擅自暂停旧长训2353。所有新权重留云，所有本机收取仅小JSON。下一阶段仍先处理用户待答的算力优先级和真实HHC tokenizer，A/B/C未启动，不能当SOTA。

**2026-10-16 02:48 SGT 新状态覆盖02:36段落：** InterMask480val v2评分exit0，FID8.931384/Matching3.795384/R1 .443750，报告本机`output/hhc_sota_20261016/intermask_validation480_metrics.json`。初次score exit1（provenance Path/str），失败文件保留。InterGen runner4126/gen4322仍运行；HHC VQ smoke4282等待同GPU锁；TIMotion runner4780也排队，不因缺STATUS重启。TIMotion权重直接云获取并CPU strict294 tensors通过。随机token-window候选已加入新CFG入口（默认historical），1024种子真实TokenData CPU检查通过，原训练未启动。小JSON已收本机，无新增ckpt；旧HHC185k仍缺，用户算力优先级待答。下一步收这三项的确切终态，保留原长训2353。

**2026-10-15 用户纠正后的最高优先级：旧服务器连接超时即切换Runpod，不再把“最新完整optimizer/RNG包未齐”作为部署阻塞。当前算力owner=Runpod，pod 3aj1czzq0a9x64，SSH root@157.157.221.29:33923（本机已有newstation私钥），单4090安全云US$0.89/GPU小时，80GB持久存储另计。旧服务器monitor已受控停止且源码按compute-owner.json禁止自动拉起；不要按旧heartbeat再次拉起旧monitor。原initial-backup可继续收原partial救回全状态，不改变Runpod优先级。**

Runpod迁移及真实更新已验收，2026-10-16 00:12 SGT实际12650。使用已验证scratch best10000作明确weights-only续训分支，仍为scratch起源，不读官方生成权重；optimizer/RNG重新初始化已记录，不能冒称恢复了原服务器全状态。新run独立路径output/intergen_rotation_scratch_runpod_20261015，704000步/rotation-FK/冻结请求不变。原服务器训练当前未知，不伪称已停止；如恢复连接，停止本聊天旧tmux后仅收备份，禁止双重自动续训。用户已明确要求在此不确定情况下继续切换，不再等旧机恢复。

闲置保护heartbeat hhc-runpod已ACTIVE，每15分钟检查本聊天pod，无训练/有效部署传输连续30分钟则pod_action stop保留卷；无变化安静。必须检查真实训练日志/更新而非只看RUNNING。训练用CLIP ViT-L/14@336与评测用ViT-B/32均已在Runpod及本机校验；额外训练CLIP也已完成Drive分片备份，详细收据见最新记录。

本机 `D:\projects\HHC` 是后续代码、实验设计、记录和备份的权威工作目录。服务器只承担计算。SSH 配置 `newstation` 指向 edmond-personal（100.101.52.38；hostname edmond-Z790-Taichi），原项目 `/mnt/8T-HDD/wenning/HHC`。初次直接用 edmond-personal 别名因没有 SSH 用户/密钥配置而失败；采用已有且主机密钥已验证的 newstation 配置成功连接。

核对实际主会话 `01a08732-f5bd-7640-a0e0-d8621c0b7447` 的用户消息：10-08开始旋转主导 InterGen 训练，10-09要求每小时监控/崩溃恢复，10-10改为从头训练并要求自主启动。本机延续这项最新任务；不依据旧交接重启已收束实验。

训练保持：rotation6D/root6D 为核心、对应 shape 骨架上的 FK position 等损失；root旋转随镜像/anchor同步变换；268D生成评测时删除末尾root6，还原262D。仅冻结CLIP保留预训练；不读取InterGen生成权重。704000步、lr1e-4、micro4×acc8、每10000步480val、每500步恢复点。val选模后独立test/MM仍待完成。

## 接管时发现

- 原 scratch STATUS 停在400；训练和旧守护均已退出。现存 recovery 与其内模型均为step0，optimizer state为空；恢复核验exit0。从step0重跑未保存的400步，不声称从400恢复。
- 原8TB盘仅余115MiB，系统盘约4.5GiB；不删除用户旧实验。
- 61GiB物理RAM、31GiB `/dev/shm` 可用；两块RTX4090空闲。采用内存盘计算工作目录，解决新的大恢复包原子保存需要额外空间的问题。
- SSH通过伦敦DERP中继，大文件慢。公网22直连超时；未修改用户代理、VPN或Tailscale配置。
- 原微调已完成10000步，最佳480val为6000/FID8.943854；这与当前scratch不同，不混报为scratch结果。

## 活跃计算及控制

- 远程计算目录 `/dev/shm/hhc_local_control_20261015`，run `output/intergen_rotation_scratch_20261010`。
- tmux `hhc_local_scratch_20261015`；本次trainer PID114898、pane PID114514（后续恢复会变）。启动日志 `resume_1791994996.log`；本机最新复核已到step850/loss1.659450有限。step500原子恢复包2188976033B、SHA256 `73dfc2722062586ead1b5ca82f247fa22f1e47afc9614ed7d2502783139821dc`，独立CPU读取检查exit0：模型与optimizer步数一致，模型/196个optimizer状态均有限。
- 原服务器源码/数据通过链接用于计算；本机修改后的训练入口部署在内存盘。本次只新增 `--write-recovery-receipt`，原子保存后生成step/bytes/SHA256小收据；不改loss或数据转换。
- 本机 `tools/local_compute_control.py monitor` 每3600秒核对状态，无训练时仅恢复此聊天创建的tmux；训练文件锁阻止重复进程。服务器重启使stage消失时，不伪报自动恢复成功，需要从本机备份重新部署。
- 本机 `tools/local_compute_control.py initial-backup` 使用SFTP reget续传。先拉原持久step0恢复文件，再按小时捕获原子recovery的不可变硬链接快照，下载核验后才释放自己创建的临时传输链接。阶段恢复文件与模型同step；未保存dataloader游标，非bitwise恢复。
- 后台Windows进程隐藏运行；状态/PID、事件及stdout/stderr在 `migration/control`。本机休眠或关机会暂停控制/收取。
- 小时heartbeat负责补充网盘上传、故障处理和有意义结果通知；后台监控已有原句柄时不得重复启动。

## 保存与备份状态

- 已下载核心源码/文档/测试：`migration/core-source.tar.gz`，2441248B，SHA256 `bd7a4db57b76b5ce94841f6c35bec9cb38c39dc57c326cf8454d30e7b56977b3`。快照排除了大型生成配置、数据和权重；不能称已复制全部874GB项目。
- Google Drive目录：<https://drive.google.com/drive/folders/1lkcZR-s9z6LAhe9mF93PLB6TEpkvM50r>。
- 核心源码已上传并读回大小：file ID `1Zzt5xylxomb3lcEZvXh8qVIT2Y2Ui20T`，<https://drive.google.com/file/d/1Zzt5xylxomb3lcEZvXh8qVIT2Y2Ui20T/view>。
- Git历史、正确中文用户history索引、冻结请求及原训练/评测小记录已落盘；`history-state.tar.gz`完整23991350B、SHA256 `70d45954d9bf4ce66f724f5128a415e30f786cd26cc8e66e15e0a94fab57851f`，已上传Drive ID `19YGEkqpHT5MEoUzlWTClPdZq5aevGXf3`。`migration/pull_messages.py`另外保存主会话5517条消息（含1592条user角色记录/环境消息，原文某些中文已在服务器rollout中损坏，以正确中文history索引交叉核对）；省去大工具输出，不复制Codex auth.json/config凭据。消息压缩包429348B、SHA256 `f60168afef4a3cdfd8039bef28f5e5eb2e63692eb404e511cb37a34c7ddc4221`，已上传Drive ID `1qM-dA9hHx0kgfVx_s_GDx6leAfP_5wSl`。
- step0 `recovery.pt` 完整729659053B，SHA256 `100d13d3a3e014647688f074cdfa704f4ef3e524fcdf8d591fe71783ce4a6116`。本机下载及独立再次校验已完成；已按64MiB分成11片上传Drive、逐片读回大小，manifest `1URxgXMaKG-ACxd0D0DTdINqnPUApUxkR`（5635B已读回），receipt标记uploaded_parts。按manifest顺序拼接后须核验整包hash。这不是当前训练最新checkpoint；新恢复点仍在后台收取。
- `.partial` 均是未验收传输，不得用于恢复或冒称备份成功。两个早期全量捕获尝试因夹带大型生成配置已停止，残留partial保留为未完成文件。
- 后续recovery/best与新完整val指标须收取并上传；Google Drive上传后读回元数据才标记成功。不要上传临时半写checkpoint。

## 备用算力

Runpod已做实时只读认证检查：账号无pod，接口可用；尚未创建付费资源，因为服务器当前可连接并已恢复训练。连接持续失败时按用户授权启用Runpod，创建前报实时单GPU小时价并设闲置停机；不可重复跑两份。

备用依赖尚未完整复制到本机，不能称可立即离线恢复：InterHuman原始motions约1.2G、processed约4.2G；冻结InterMask预处理/统计、InterCLIP约1.51GB、CLIP ViT-B-32约338MB、合法已有female SMPL约237MB，以及third_party固定commit。公开原始数据链接及校验下载程序见 `docs/interhuman_raw_conversion_20261003.md`、`tools/download_interhuman_raw.py`。从原服务器收取许可证限定的已有模型；不替换统计或评测器，不重新抽取冻结val/test。先核验数据、依赖与最新完整恢复点才启动备用训练。

## 下一步

### 2026-10-16 01:24 SGT 更新后全状态12000已云备份

recovery step12000的33片及manifest16537B全部metadata核对，Drive ID1o4H7FFIBOTmkAws4Q_P6UNFr0dcxUM6E，receipt uploaded_parts；完整源2188976561B/hash ad87b6136dfbc7cb00ddd99f241e9323061fda7de68f9b4e49b90eaed1eef11a，恢复时有序拼接后核验。它包含本Runpod分支更新后的optimizer/RNG，仍未保存dataloader cursor，非bitwise恢复；不要重复上传该包。当前训练14750、无新完整val，下个20000，继续后台收后续恢复点/新best。旧server14k partial未动，旧monitor禁用。

### 2026-10-16 CLIP固定源码另存

实际运行CLIP源码commit d05afc436d78f1c48dc0dbf8e5980a9d471f35f6的完整公开归档已下载并安全展开到third_party/CLIP-d05afc436d78f1c48dc0dbf8e5980a9d471f35f6，代码中训练模型URL hash与实际冻结ViT-L/14@336匹配。原归档migration/clip-source-d05afc436d78.tar.gz 4312596B/hash553c80a6364ac7350dd7800f535531175f22c11fa742bda1b50b62121eb5f40f已上传并metadata读回Drive ID1TAw7tbQ3nOWpPb3E0hlrMSmNHm6QHfZs，receipt uploaded。bootstrap目前仍用精确commit Git URL；若外网Git不可用可把此已验证源码迁入实例并从本地目录安装，不能宣称已制作全部Python依赖离线镜像。没有改当前训练或正在运行的脚本。

### 2026-10-16 00:12 SGT 本轮云备份完成

额外训练CLIP934088680B/hash3035c92b350959924f9f00213499208652fc7ea050643e8b385c2dac08641f02的14片全部metadata核对，manifest6648B已上传读回ID1uyLmRgGbILuKPOIxXz17skwZFMxnjxLq，原receipt uploaded_parts。新Runpod初始10000恢复包11片及manifest5712B已完成，ID1WIVhyXaFhEia399v7jcaq23o9UPykAuZ。两项不重复上传。真实训练12650/目标704000，无新完整480val；controller31608当前固定收12000全状态2188976561B/hashad87b6136dfbc7cb00ddd99f241e9323061fda7de68f9b4e49b90eaed1eef11a，runpod-transfer.json与同一partial保留，receipt前不可称本机/云完成。

### 新Runpod初始恢复包10000已云备份

初始10000 recovery已11片全验收、最终manifest5712B metadata读回，ID1WIVhyXaFhEia399v7jcaq23o9UPykAuZ，receipt uploaded_parts。不要重复上传此包；source=新分支reset后的初始状态，不是最新全状态12000，更不是原服务器14k。CLIP训练大模型另14片尚在上传，按其独立manifest续传。

### 23:50 SGT 两项云备份执行

额外训练CLIP本机934088680B/hash3035c92b350959924f9f00213499208652fc7ea050643e8b385c2dac08641f02成功receipt已生成，31392正常完成不重启；新Runpod分支初始10000 recovery729659645B/hash eb206d8e519fcbaa4d1a7db975006f31920e7be24f214eb5b55220ac9c20fb05也已本机校验。二者.parts/manifest.json共11+14片正在逐片上传核对，不重复上传已有cloud_id，uploaded_unverified要先metadata补验。全部片及最终manifest核对后再receipt uploaded_parts。本机完整源码/数据/SMPL/两种CLIP/InterCLIP均已备齐，训练环境依赖仍按Runpod freeze/bootstrap重建，不等同于GPU离线环境镜像完成。实际12000及该步2.19GB新全状态尚在Runpod，下个后台周期会收取；不能把当前10000云备份误报12000。

### 22:55 SGT 补充本机训练CLIP后台

tools/local_training_clip_download.py PID31392独立收取runtime_assets/clip/ViT-L-14-336px.pt，官方Azure URL的固定934088680B/SHA3035c92b350959924f9f00213499208652fc7ea050643e8b385c2dac08641f02已由Runpod真实文件/官方包URL核对。锁/PID/log/err/status在migration/control/training-clip.*，同一partial；完成正常退出不要重启。成功receipt、大小/SHA通过后按64MiB分片上传Drive并读回metadata再标uploaded_parts；尚未完成时如实保留pending，不阻塞当前Runpod训练。Runpod已10250，无新val。

### 22:53 SGT 最新句柄

训练已10150，loss0.31806685，grad4.901733有限。当前本机Runpod controller为31608，旧24216为加载新增bootstrap恢复入口已受控退出；同一传输计划/partial保留，训练未重启。后续只读runpod-control.pid.json。实例重启若SSH地址变化，先get_pod读真实direct host/port再更新本机controller及闲置保护，不能把旧地址超时认作训练消失；/opt环境和shm依赖由tools/runpod_bootstrap.sh重建，已固定CLIP源码commit及关键Python依赖。当前尚无新Runpod完整val。

### 22:50 SGT 首个Runpod实际更新验收

实际训练到10050，50次更新93.55秒，loss0.52555495、grad_norm5.737349有限；新STATUS已training/10050。下一完整480val20000；继承best10000/FID44.620449不是新验证。后台controller24216已开始收取新分支初始10000恢复包729659645B/hash eb206d8e519fcbaa4d1a7db975006f31920e7be24f214eb5b55220ac9c20fb05，固定transfer_step010000.pt与runpod-transfer.json，保留partial后核验，不重复另起下载。当前计算owner=Runpod，旧server monitor不要再拉起。

### 22:49 SGT Runpod已接入实际训练/新后台

当前训练run=/workspace/HHC/output/intergen_rotation_scratch_runpod_20261015，日志/workspace/hhc-training.log；GPU已参与计算，等待10050首条更新，不把旧preparing/step0 STATUS当作真实当前step。明确从scratch best10000权重继续，optimizer/RNG新初始化，config.continuation记录来源step/SHA并在后续resume中保留；下一完整480val20000，best10000/FID44.620449来自迁入权重，不是新Runpod结果。

本机tools/local_runpod_control.py PID24216，锁/PID/状态/传输计划在migration/control/runpod-*；隐藏日志runpod-control.log/.err。按小时收原子恢复包的不可变硬链接快照并断点收取，大小/整包SHA通过后receipt pending，后续heartbeat上传Drive。只恢复本聊天Runpod训练入口，不拉旧机；旧机可达时定向停止本聊天旧trainer/tmux并写old-compute-fenced.json，不删任何旧实验。旧initial-backup30948仍救原14k partial。不要再重启旧monitor。

Runpod公共数据/模型及权重哈希已验证，全部480val输入回放通过。额外训练CLIP ViT-L/14@336 SHA3035c92b350959924f9f00213499208652fc7ea050643e8b385c2dac08641f02已在Runpod校验，本机尚未收取这项大模型，不要再次误报所有冻结资产本机齐。环境/opt/hhc-env，持久卷/workspace/HHC，快速展开数据/dev/shm/hhc_inputs（58GiB shm）；重启后必须重新展开，原始zip/raw tar在卷中。runtime freeze记录本机migration/runpod-runtime-freeze.txt，非原New_MMM环境bitwise复现。

### 22:05 SGT 本机后台已恢复

旧monitor37724/initial-backup57948均已退出，命令行全量搜索无同任务进程，原因未知。确认退出后仅各启动一份隐藏后台，新monitor30724、initial-backup30948；以后以各.pid.json为准。日志改用monitor-recovered.log/.err与initial-backup-recovered.log/.err。固定14k partial1053358080B及rolling-transfer.json保持原样；未创建重复下载。SSH独立复核仍超时，17700仅旧状态，不称当前训练存活或已恢复。无新val，不重启已正常完成的runtime-assets及数据下载器。

### 06:53 SGT InterCLIP云备份已完成

InterCLIP本机整包固定SHA通过，23片全部上传并metadata大小核对，manifest11206B已上传读回ID191GD3KFIBR07sszc3p5bIDzDIfaGc101；原receipt标uploaded_parts、有序cloud_parts及manifest URL。不要重复上传已成功分片，不要重启正常完成的runtime-assets62896。Runpod仍未创建，原因详见readiness：最新完整14k recovery未齐、旧服务器训练状态无法确认/未隔离；资产齐全不等于备用训练成功。monitor/backup继续原断点，不改scratch、不静默用model-only best替代全状态恢复。

### 06:38 SGT 冻结资产已本机收齐

官方HTTP Range已沿原partial补齐InterCLIP1513626777B，整包SHA256与原资产固定4e14b8ee5c2d965778fe003fb37d98ed8e5c7e0940668515fc7dcfa2831c17f8一致，receipt与runtime-assets.json均已更新。62896正常完成，不重启。23个64MiB分片及manifest在interclip.ckpt.parts，逐片metadata读回再标uploaded；目前正在云备份，receipt完成前不可宣称成功。所有备用输入/冻结资产本机完整，但最新训练全状态仍partial，旧服务器停止证明与Runpod环境/闲置停机尚未验收。

### 06:29 SGT 服务器失联 / Runpod预检

服务器多次SSH/SFTP连接超时，旧trainer是否仍运行未知；monitor37724、initial-backup57948存活，不重启。Runpod社区RTX4090实时报价US$0.34/GPU小时、库存LOW，官方runpod-torch-v280模板读回；尚无付费资源，不能未解决双重训练风险就启动新trainer，也不能用best10000静默丢弃optimizer/RNG替代完整recovery。切换门槛详见migration/control/runpod-readiness.json。

官方InterGen评测器下载脚本给出的Drive文件1bJv5lTP7otJleaBYZ2byjru_k_wCsGvH实际metadata大小1513626777B匹配冻结InterCLIP。local_runtime_asset_download.py增加SSH失败后的官方HTTP Range备用（同一个已有partial，8MiB请求，必须整包固定SHA通过）；首次HTTP探测读超时，不能称已补齐。确认旧57740暂无SFTP子进程，受控停止并证实退出后启动新62896加载备用；以后读runtime-assets.pid.json，不依据旧PID重启。monitor/backup未停、14k固定快照partial未换。公开源不是换评测器，最终SHA仍4e14b8ee5c2d965778fe003fb37d98ed8e5c7e0940668515fc7dcfa2831c17f8。

### 05:33 SGT 数据及固定源码就绪

processed下载/校验/解压已完整结束：ZIP2462028219B、固定SHA256通过，15554文件通过ZIP CRC检查；状态complete，正常退出不要重启61212。原始7779文件MD5通过，全部480冻结验证请求所需raw/annots/person1/person2文件本机存在，请求内容未改。

`third_party/intergen_base_20261008`已恢复独立Git metadata，HEAD为固定b1750df4a859443b8a69329f759ce3a7503ae179；源码/config与Git对象diff为空，仅readme三项媒体未迁移，不影响训练。不能在缺少nested Git时把HHC父库commit当作InterGen源码身份。

best10000本机收取已验收，59628正常退出不重启；云端分片正在执行，receipt uploaded_parts之前不冒称完成。InterCLIP仍在续传，全部备用依赖仍未齐。最新训练异常是16850处Aborted，已自动从16500恢复到17700；未确认根因，不将它凭空诊断成同一SIGILL。

### 04:33 SGT 续传修复与公开数据

原始motions7779文件已全部按原MD5清单核验完成，状态在`output/interhuman_raw_download_local_20261015.json`，进程正常结束，不要重启。processed ZIP已查官方Drive目录，ID`1qpFij33tVZhUkk9-kDaDy1FuQb0LJSEf`、2462028219B；固定SHA256`ad951d24f95fff550d964c8f977eb99da47264b81e3080312c95046e69fd6325`与服务器原完整文件一致，HTTP Range206探测通过。后台`tools/local_processed_data_download.py` PID61212，日志/锁/PID/状态在`migration/control/processed-data.*`，按64MiB范围续传；整包SHA核验后才重命名，逐成员路径边界及ZIP CRC检查后才解压。`processed-data-status.json` complete之前不称备用数据齐全。

滚动恢复包收取修复已生效：`migration/control/rolling-transfer.json`保存step/不可变remote/bytes/hash，超时或进程重启先恢复同一个快照，不跳到新step。旧backup53732及其SFTP做明确受控维护并证实退出后，新backup57948加载修复，继续原step14000 partial。模拟超时和重启回归通过；原step7000未完成partial仍保留，未声称它已验收。monitor37724及训练未重启。后续检查先读`initial-backup.pid.json`取得当前PID，不依据历史记录53732判定失活。

processed downloader和best/runtime assets均只在已退出且尚未complete/存在有效receipt时考虑重启，保留同一partial；完成正常退出不算故障。若已有结果SHA不符，记录错误，不能覆盖用户文件或冒称恢复成功。

### 首个完整验证及best收取

best10000已在本机/Drive完整保存：729644528B，11个64MiB分片逐片大小读回；重组manifest ID`1OAc3tMjGG-096kA1Z0amdN-CKN0RBgjY`（5566B已核对），原receipt已uploaded_parts。按顺序拼接后核验整包SHA256。它只含模型训练权重/配置/归一化，不含完整optimizer/RNG；最新完整恢复点仍在独立rolling backup中，不能宣称全状态续训资产都已云备份。

step10000完整480val已验收：FID44.620449（也是首个best）、Matching3.996044、R1/2/3=.102083/.191667/.247917、Div7.122461、GT7.741262，seed20261006；仅训练内验证，独立test/MM待评。本机报告/embedding在`output/local_compute_intergen_scratch_20261015/validation_010000`，包已备份Drive ID`1nrzAVzsVYF4ZBHZt_a-g2BC0YlEARw4k`。训练继续，下次完整val20000。

后台`tools/download_pinned_best.py`以独立锁收取已硬链接冻结的best10000，remote=`/dev/shm/hhc_local_control_20261015/output/intergen_rotation_scratch_20261010/best_transfer_step010000.pth`，729644528B/SHA256`e432c7a2ed82ae3fbfbb7bcdb00525e241eee0e44d35297a9b23e66d0454c2dd`。PID/日志/错误在`migration/control/best010000.*`。完成后退出；只有receipt且大小/hash通过才能分片上传，不能因该进程正常退出而重启重复传输。本机目标`checkpoints/local_backups/intergen_scratch_20261010_step010000/best.pth`；当前尚未完成模型备份。

### 03:39 SGT 接管修复与备用资产状态

验证入口已显式复用`data.adapter`，避免SSH cwd=/home/wenning引起默认相对数据路径错误；全部480冻结请求CPU回放exit0。待原进程exit1后才部署修正，主10589 exit0，新日志`resume_1792006402.log`、trainer121970从9500恢复，现已进入真实480生成。尚未获得有效评分前不报告FID。保留之前失败/中断缓存，未改loss、统计或请求内容。

SMPL/CLIP已完成本机和Drive分片备份，原receipt标记uploaded_parts；重组manifest ID分别`1NUDvuHqVUDUXx5F28InauJIMi1Yf-_s-`和`1ixV_jvnFn9R5bLE6_wpeir4mSXazPdCY`。InterCLIP仍由runtime-assets后台收取，不能将这两项完成当作全部备用依赖就绪。

`tools/local_public_data_download.py` PID62800后台下载官方原始motions，以原固定7779条size/MD5清单核验，日志/锁/PID在`migration/control/public-data.*`，本机状态单独保存在`output/interhuman_raw_download_local_20261015.json`，不覆盖原服务器完成清单。下载依赖requests位于`migration/download_packages`（pip --target），不修改bundled或训练环境；使用本机已有127.0.0.1:7892代理仅限该下载进程。仅在进程已退出且本机manifest仍partial时重启，已有MD5通过文件自动跳过。processed下载尚未开始。

### Google Drive 大文件限制

2026-10-15连接器入口上限512MiB，但实际传输层又拒绝256MiB分片，提示上限104857600B（100MiB）。有效上限按100MiB处理：大文件使用`tools/split_for_drive.py <已校验完整文件>`生成64MiB顺序分片及`.parts/manifest.json`（整包/每片字节数及SHA256）。逐片upload_file并metadata读回大小，及时在manifest对应行保存cloud_id/cloud_url/cloud_status，避免重传已成功分片。所有片通过后上传最终manifest，再在原receipt标记cloud_status=`uploaded_parts`、cloud_manifest_id/url及ordered parts。恢复时按index二进制拼接，并核验整包字节数/SHA256后才能torch.load；不要将单片当checkpoint。部分上传必须保持pending，不可声称完整云端备份成功。旧256MiB分片没有上传成功，不纳入最终manifest。

### 备用依赖队列补充（2026-10-15 01:34 SGT）

冻结InterMask小源码/统计/eval配置已保存到`third_party/InterMask`，官方split/annots已保存到`dataset/interhuman_official`。原始motions官方清单在`output/interhuman_raw_download_20261003.json`；不能将清单中服务器当时的complete状态误认为本机数据下载完整。三项大资产路径、字节数及SHA256在`migration/runtime-assets.json`。

元数据包`migration/dependency-metadata.tar.gz`（1587867B、SHA256 `56955c3b3f98e006abd308edfc81042dfb0da0d8db2f24ee67515a9aa1f3a816`）已上传并读回Drive ID `1nnHhkB0AW7mXk6sHiYQSMT7qTUo7U93V`。

后台`tools/local_runtime_asset_download.py` PID57740按小时等待step0 recovery的成功receipt后，顺序续传SMPL/CLIP/InterCLIP，避免抢占尚未完成的初始恢复包传输。锁与PID在`migration/control/runtime-assets.*`；只在已证实退出时重启。每个大资产校验后生成同名`.receipt.json`，后续heartbeat也须将这些已核验资产上传现有Drive目录并读回metadata，随后标记receipt uploaded。本机公开motions/processed下载尚未开始；bundled Python不含requests，已有Anaconda的导入探测长时间未返回已停止，不影响训练/备份后台。后续用独立迁移环境安装下载依赖，不改GPU训练环境。

1. 完成初始恢复包本机下载，上传Drive并验证（Git/会话记录已保存）。
2. 首次step500恢复收据/模型/optimizer已核验；继续按小时收取后续恢复点，新完整val报告当前及best FID、Matching/R/Div。
3. 补齐备用依赖和best权重收取，使服务器失联时可部署Runpod。
4. 训练完成后冻结best做独立test/MM；完整验收前不宣称性能改善成立。
