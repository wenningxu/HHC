2026-10-22 17:07 UTC：关闭tcache的官方trainer283224仍在约471210步SIGABRT退出，GDB返回255；同类fastbins堆损坏检测于Python list分配，不能归因为list。该调整未解决故障。完整470976/epoch1337 checkpoint已CPU验收196optimizer/model finite/scheduler/loops，SHA5a4a85bc0db2a568feb402fe3ab37d575b4c83237ccfe58f3c5c7dce95830ff5，云native_abort_20261022_1659硬链接保留；本机小证据no_tcache_abort_20261022_1659.json，owner已标记停止。旧watch283384仍存活，任何下次恢复必须先核实停止它，不能重复watch；HHC283349/eval283383仍等待。当前新独立诊断unit hhc-ddp-loopback-memcheck-20261022，目录native_abort_20261022_1659/ddp_loopback_memcheck：Valgrind+原DDP/真实数据/loss，NCCL_SOCKET_IFNAME=lo、NCCL_IB_DISABLE=1、NCCL_DEBUG=INFO尝试隔离此前初始化停滞；470976完整恢复，回调限32步/900秒、禁生产checkpoint，未出结论。下次读取该unit/launch/实际PID，勿重放。正式主训练停止；HHC4000及704k+实际20rep/MM+独立review不变，无新FID/本机权重。

2026-10-22 12:59 UTC：官方GPU1已从完整449504恢复，实际读回449660；trainer283224/birth53283846，HHC283349/53296331、eval283383/53296494、唯一watch283384/53296499。unit hhc-official-no-tcache-20261022，Restart=no，云native_abort_20261022_1203。保持原DDP/模型/loss/FP32/batch32/worker0/激活包装与完整状态恢复，GDB及原PYTHONMALLOCdebug/trace保留，仅新增GLIBC_TUNABLES=glibc.malloc.tcache_count=0作堆诊断；不是根因修复证明。首批epoch/scheduler1277及196optimizer通过，约.67秒/步。旧两个纯等待队列核实后停止并重绑，旧watch缺失且未重复启动。此前Valgrind完整DDP诊断卡NCCL bootstrap/首次broadcast，采栈后明确停止；单卡v1策略auto不兼容Lightning1.9.1而退出；v2的max_steps触发iteration-based恢复，令手动scheduler补偿重复，32步仅作内存负复现。v3改回调限步，实际449504→449536、scheduler1277通过；返回91唯一错误为纯print Python文件也复现的启动wcscat非法读取，无训练非法写入，不能排除长期/DDP故障。8条现存正式续训首批scheduler/LR均匹配。所有诊断输出独立、无生产checkpoint写入。小证据single_device_memcheck_v3_result_20261022.json、production_resume_scheduler_audit_20261022.json、no_tcache_recovery_20261022.json。后续按10分钟间隔监控同一新LAUNCH，下一完整checkpoint再CPU验收。HHC4000与704k+真实20rep/MM+独立review不变，无新FID/本机权重。

2026-10-22 12:26 UTC：官方主训练当前停止，最新279167再次SIGABRT（fastbins堆损坏），末日志449710；完整449504/epoch1276检查点CPU验收196optimizer与finite通过，SHA135dd50b022eb757a2adf4e160bc5c5db793ffbb1292d273212d1556fa71a3ac，留云native_abort_20261022_1203。此前GDB分别在autograd RNG state分配和NumPy einsum分配检测损坏，均不是已定位的写坏来源。PYTHONMALLOCdebug未提前捕获。独立原Dataset Valgrind1000样本0errors；八层模型16步完成，唯一非法读取在纯print Python文件启动也复现，不能认定模型根因。证据output/hhc_sota_20261016/model_memcheck_startup_control_20261022.json。当前完整训练路径Valgrind诊断unit hhc-full-train-memcheck-20261022/MainPID281077、实际memcheck281079/birth53095762已确认运行；目录native_abort_20261022_1203/full_train_memcheck。独立日志/禁用checkpoint/logger，保留原数据/loss/DDP/FP32/batch32/激活重计算及worker0，从449504恢复，最多32步或900秒；需读取同一句柄，未通过不声称恢复成功，不重放。主LAUNCH仍旧终态，HHC4000及704k+真实20rep/MM+review门槛保持。无新FID，无本机权重下载。下一步收取完整路径诊断结果并定位故障；普通主训练监控此时不适用。

2026-10-21 21:46 UTC 原Dataset独立CPU/GDB预处理600000次、507.9秒通过，原gdb返回0/CUDA未初始化/两源码SHA不变；该负复现不排除模型运行时堆损坏。真实训练改由GDB监督，新增inferior登记包装保证LAUNCH PID是真实Python而非gdb；CPU正常/abort smoke分别0/255并捕获SIGABRT原生栈。原模型、loss、数据transform、batch32及worker0隔离配置保持。完整371360恢复首批/epoch1055/scheduler1055/196optimizer通过。trainer259281/birth47807457，实际TracerPid259271匹配；HHC259413/47817428、eval259451/47818769、唯一watch259452/47818774。systemd hhc-official-native-debug-20261021，Restart=no；云recovery_native_debug_20261021/train.log将捕获原生栈和sharedlibrary；若异常保留原GDB输出，不把gdb返回255解释为训练自发exit255。换LAUNCH前已确认旧watch数量0。记录371510实际更新，约.62秒/步。新故障根因仍未知；无新FID/HHC放行/本机权重。证据本机native_abort_cpu_result_20261021.json与edmond_gdb_training_recovery_20261021.json。

2026-10-21 最新：single_loader251846再次SIGABRT，运行约179分钟，末日志371670；取消多进程未解决故障。faulthandler当前线程在原utils.process_motion_np:146的np.einsum（NumPy1.23.5/torch1.13.1），仍只是检测点，不能归因为einsum或硬件。完整371360 CPU验收196optimizer，SHAcd79de89428c82be023ff1162be6b06ca2342fe2820c86c897554a3daaf6d6d2，硬链接留云native_abort_audit_20261021。当前GPU训练终态，先做云CPU原Dataset预处理复现；systemd hhc-native-cpu-audit-20261021/MainPID258291，GDB批处理+native backtrace，命令/child/exit/progress/result均在该目录，最多600000样本或600秒处理（不含加载）。工具audit_official_intergen_native_abort_cpu.py；不含训练模型、不能把通过当根因排除或质量提高。下一步读取同一unit/child/progress，待终态收取cpu_exit.json与cpu_result.json或GDB栈，再决定恢复/定位；不要把训练监控assert当新故障，不盲目重启。若恢复，须先核实停止所有本任务旧observer再换LAUNCH。HHC4000与704k+实际20rep/MM+独立review不变。GPU仅云端、本机未下载权重。

2026-10-21 20:01 UTC 修复重复observer竞争：旧249535/birth46401116在上次trainer退出后未及时检查终态，恢复后继续跟随新LAUNCH，覆盖当前252063/birth46648710的OBSERVER_STATUS，触发身份一致性assert。已按/proc与历史身份核实后仅SIGTERM旧observer，云duplicate_observer_stop_intent/receipt保留；当前trainer251846、队列252028/252062和observer252063未动。下一当前observer自然写入后独立验收363230通过，无训练中断/新FID。今后任何恢复在换LAUNCH前都必须检查并停止本任务旧observer（不能因原trainer缺失就推断observer也退出），再单次启动新watch，避免共享状态竞争。

2026-10-21 18:31 UTC 第二次SIGABRT已捕获堆损坏：原249116末日志354570，malloc(): invalid next->prev_inuse (unsorted)，faulthandler主线程在multiprocessing接收DataLoader数据。该栈只定位检测点，不能证明写坏内存的根因。完整354464 checkpoint CPU验收通过（196optimizer/SHA812fb16d3ad9cbf3385a5f9d1143ccb409ea190a8853665c588a91729bba083c）。新增外层single_loader driver，仅目标InterHumanDataset的DataLoader num_workers2→0，保留官方源码字节、模型/loss/batch32/shuffle/drop_last/scheduler；明确随机增强RNG轨迹改变，不称bitwise。云CPU实际官方train_dataloader方法+11274合成确定索引/352批次一致，函数签名与非目标loader保持。实际首批354464/epoch1007/scheduler1007/196optimizer通过，runtime loader确认0worker/352batch；约.62秒/步。trainer251846/46638034；HHC252028/46648536、eval252062/46648701、watch252063/46648710。systemd hhc-official-single-loader-20261021，Restart=no，故障栈保留。云recovery_single_loader_20261021；本机edmond_heap_corruption_loader_isolation_20261021.json和edmond_single_loader_recovery_20261021.json。仍是稳定性隔离措施，尚未证明根因修复。HHC4000及704k+实际20rep/MM+独立review保持，无新FID/本机权重。

2026-10-21 17:50 UTC 捕获官方训练SIGABRT：systemd监督记录原trainer244699/birth45967185 returncode=-6/signal6（17:39:33UTC，末日志351460），service exit134；无Python栈，日志末少量非UTF8字节，限定时间kernel无OOM/Xid/MCE记录，根因仍未知，不能宣称已修复。完整351296 checkpoint CPU验收196 optimizer/模型/scheduler/loops通过，SHA e8a6a9c60ea414cdf46686113753dcdd17a693e8f4112fab1475d472430b5f21。新systemd单次服务hhc-official-gpu1-20261021-1743，Restart=no，新增PYTHONFAULTHANDLER=1/TORCH_SHOW_CPP_STACKTRACES=1诊断环境，学习协议未改。trainer249116/birth46377254，首批351296/epoch998/scheduler998；HHC249500/46400944、eval249534/46401107、watch249535/46401116。GPU1 UUID实际匹配；记录351370实际更新，队列纯等待。云证据recovery_gpu1_20261021_1743；本机edmond_sigabrt_20261021.json与edmond_sigabrt_recovery_20261021.json。HHC4000/704k+实际20rep/MM+独立review保持，无新FID/本机权重。

2026-10-21 正式fair-test设备准备：两个尚未运行的export_hhc_text_only_formal_cloud.py与export_interhuman_native_baselines_formal_cloud.py仅将CUDA_VISIBLE_DEVICES门控0改1及提示文本，逻辑cuda:0保持（可见设备重编号）。AST反向替换后与原版完全一致，语法通过；无模型导入/GPU执行。云端核实四现行队列配置均不绑定这两个脚本，尚无selection receipt。将来正式生成前须重新产生与新版exporter SHA匹配的公开基线CPU acceptance并正常冻结真实val选择；不能复用旧exporter指纹receipt。当前训练、选择规则、20rep/MM门槛未改。

2026-10-21 16:38 UTC GPU1官方训练恢复：原228708在345510日志后消失，无异常栈；journal仅会话1253结束，无已确认退出原因。完整345312 checkpoint CPU验收通过（196 optimizer，SHA a75eee4d4bd9e94ecf7d449d1eff63d9051f1e1c2a60cabcc3dff6df2a19e4b6）。新trainer244699/birth45967185由用户systemd单次服务hhc-official-gpu1-20261021-1632托管，Restart=no；supervisor记录子进程终态，服务保留退出信息。首批345312/epoch981/scheduler981一致，物理GPU1 UUID匹配，约0.60秒/步。原官方源码、loss与学习协议未改。证据云recovery_gpu1_20261021_1632。HHC仍4000，704k+实际20rep/MM+独立review门槛保持；无新FID，无本机权重下载。

2026-10-21 用户明确迁到GPU1继续官方基线：GPU0其他用户任务使步时约0.60→1.49秒；仅迁本任务。完整332288 checkpoint CPU验收196项optimizer/model/scheduler/loops，SHA99367b4ebef260155c9cafd9692067cb087294c005c585518224c203e70f60ba。新独立gpu1恢复driver调用原激活包装函数与未改官方train.py，原包装字节及探针凭据保留；不绕改模型/loss。trainer228708/birth45115067；首批332288/epoch944/scheduler944，GPU1实际PID匹配，约0.60秒/步。四等待器设备取config cuda_visible_devices=1，AST差异仅设备选取；nvidia-smi运行时查询范围同步GPU1。HHC230205/eval230206/text230207/shape230208/watch230209，birth45141472；HHC4000、704k+真实20rep/MM+review门槛保持。证据云gpu1_transition_20261021、本机edmond_gpu1_transition_20261021.json。勿重放旧launch/freeze intent。无新FID，无本机权重下载。

2026-10-21 Edmond官方训练中断后已从完整316448恢复。CPU验收196项optimizer、model finite、scheduler/loops通过；checkpoint SHA3a49440bedd273fb92e6354f705667a42eedcf1bc910bb742cfbe06a4487978b。原日志末316630无异常栈，journal会话退出但原因未证实；系统盘989MB、HDD2.3T空闲。首次恢复因新增TMPDIR路径太长导致AF_UNIX错误，首批前终止并保留记录；第二次短路径/workspace/tmp成功。trainer211103/birth44085334；HHC211405、eval211406、watch211407/birth均44106721。原等待器因LAUNCH指纹变动退出，核实无phase/child/intent后归档状态并显式重绑。官方源码/学习协议未改，HHC4000及704k+真实20rep/MM+独立review门槛保持。证据见云output/intergen_official_baseline_20261017/recovery_20261021与本机output/hhc_sota_20261016/edmond_recovery_20261021.json。无新质量结果，无本机权重下载。

### 2026-10-17 00:58 SGT｜HHC已保存暂停，InterGen独占

- 远端保存控制器确认A step4000检查点3429290971B、SHA256 fb1b5a2c606ffaa68fcff878f1038d122713d652b1767de35c64acd0f8614963，模型/optimizer/scheduler/RNG存在。原A17259 exit -15、ABC7178终态是用户优先级要求的有意暂停，不恢复旧driver。
- InterGen2353继续56800/704000；恢复等待器17952等待完整704000及身份/最终checkpoint核验，不因空闲或异常提前恢复HHC。远端GPU31%/5796MiB，保持pod运行。权重留云、无新增下载或GPU任务。

### 2026-10-16 22:44 SGT｜A组真实更新已确认

- 原队列7178/child17259实际命令保持历史text/window、final frozen VQ、seed114514、micro4/40k、CPU evaluator/GPU50%。A_historical.log已记录step1/2/200，step200 loss8.928966、gradient norm1.452371有限，GPU allocated峰值5.692966GiB；不是质量验证结果。STATUS.md仍step0，不用旧状态判断死亡或重启。
- 原VQ manifest complete/300k保持，未重启5690/5729；InterGen2353推进52500。8321/9975仍存在，SSH远端GPU60%/12207MiB。保护保持运行、idle_since为空，无新增资源、重启、停机或本机checkpoint下载。
### 2026-10-16 22:29 SGT｜VQ300k完成，原队列接续A

- fresh VQ manifest=complete、main300000/update300999，原driver/trainer已不在进程表。最终480训练内重建FID2.637497，best170000/2.447121；Matching3.808705，R1/2/3=.358333/.516667/.614583，Div7.763293/GT7.715767；seed123/fixed_val_first480_center_max300_raw_smpl_v1。较295k FID/Matching/native Body误差及Div偏差变差，R1下降、R2/R3改善。
- 十项严格口径：RMSE、Full MPJPE、Body non-root MPJPE、root误差、Full/Body joint FID、滑步、接触F1待评；Hands MPJPE/Hands joint FID不适用。额外native22含root监控101.681892mm不替代严格Body项。
- 原7178已自动freeze_final exit0、进入A_historical child17259（实际命令同绑定配置），tokens.pt存在，尚未验到更新日志，不宣称已完成训练更新。freeze spec为completed_training_val_best，step170000、115069874B、SHA2f676f63466135385ded7016d60da9dd3b6e86e3086b85e14a201732d1a55002、origin=fresh_cloud_rebuild_not_historical185k。未手动重复launch或重启VQ。
- InterGen2353推进52050；8321/9975仍存在。SSH远端GPU16%/6451MiB，保持运行、idle_since为空；一次SSH banner超时后原地址重试成功，无算力切换。无本机checkpoint下载；权重云到云备份未验收。下一步仅观察A实际更新和后续队列，formal/test仍不启动。
### 2026-10-16 21:58 SGT｜VQ主步295000训练内480val

- fresh VQ seed123、fixed_val_first480_center_max300_raw_smpl_v1：重建FID2.517825，best仍170000/2.447121；Matching3.807691，R1/2/3=0.368750/0.506250/0.606250，Div7.762134/GT7.715767。较290k FID/R2/R3/native Body误差及Div偏差变差，Matching/R1改善；非生成FID或正式test。
- 十项严格口径：RMSE、Full MPJPE、Body non-root MPJPE、root误差、Full/Body joint FID、滑步、接触F1均待评；Hands MPJPE/Hands joint FID不适用。额外native22含root监控Body MPJPE101.115503mm不能替代严格Body non-root项。
- 实际/proc、日志及manifest确认InterGen51100、VQ296601活跃，三个队列存在。SSH远端GPU99%/8533MiB，保持运行、idle_since为空；无重启、停机或本机checkpoint下载。
### 2026-10-16 21:43 SGT｜VQ主步290000训练内480val

- fresh VQ seed123、fixed_val_first480_center_max300_raw_smpl_v1：重建FID2.508909，best仍170000/2.447121；Matching3.808893，R1/2/3=0.350000/0.527083/0.616667，Div7.742551/GT7.715767。较285k FID/Matching/R2/native Body误差及Div偏差改善，R1/R3下降；非生成FID或正式test。
- 十项严格口径：RMSE、Full MPJPE、Body non-root MPJPE、root误差、Full/Body joint FID、滑步、接触F1均待评；Hands MPJPE/Hands joint FID不适用。额外native22含root监控Body MPJPE100.944840mm不能替代严格Body non-root项。
- 实际/proc、日志及manifest确认InterGen50600、VQ293001活跃，三个队列存在。SSH远端GPU99%/8533MiB，保持运行、idle_since为空；无重启、停机或本机checkpoint下载。
### 2026-10-16 21:28 SGT｜InterGen50k与VQ285k完整480val

- InterGen rotation scratch step50000：FID26.835826，best仍20000/24.314143；Matching3.964856，R1/2/3=0.125000/0.208333/0.256250，Div7.420342/GT7.741262。480固定val/seed20261006/DDIM50/CFG3.5/官方InterCLIP，artifact complete；非独立test，MM待评。较40k FID/Matching/R改善，Div偏差变差。
- fresh VQ seed123、fixed_val_first480_center_max300_raw_smpl_v1：主步285000重建FID2.705952，best仍170000/2.447121；Matching3.809445，R1/2/3=0.364583/0.512500/0.629167，Div7.751274/GT7.715767。较280k FID/Matching/R2/native Body误差变差，R1/R3及Div偏差改善；非生成FID或正式test。
- VQ十项严格口径：RMSE、Full MPJPE、Body non-root MPJPE、root误差、Full/Body joint FID、滑步、接触F1均待评；Hands MPJPE/Hands joint FID不适用。额外native22含root监控Body MPJPE106.390874mm不能替代严格Body non-root项。
- 实际/proc、日志及manifest确认InterGen50150、VQ289201活跃，三个队列存在。SSH远端GPU98%/8533MiB，保持运行、idle_since为空；无重启、停机或本机checkpoint下载。
### 2026-10-16 20:58 SGT｜VQ主步280000训练内480val

- fresh VQ seed123、fixed_val_first480_center_max300_raw_smpl_v1：重建FID2.565684，best仍170000/2.447121；Matching3.807963，R1/2/3=0.350000/0.525000/0.625000，Div7.752593/GT7.715767。较275k FID/R2/R3及Div偏差改善，Matching/R1/native Body误差变差；非生成FID或正式test。
- 十项严格口径：RMSE、Full MPJPE、Body non-root MPJPE、root误差、Full/Body joint FID、滑步、接触F1均待评；Hands MPJPE/Hands joint FID不适用。额外native22含root监控Body MPJPE101.723517mm不能替代严格Body non-root项。
- 实际/proc、日志及manifest确认InterGen49750、VQ280001活跃，三个队列存在。SSH远端GPU98%/8533MiB，保持运行、idle_since为空；无重启、停机或本机checkpoint下载。
### 2026-10-16 20:43 SGT｜VQ主步275000训练内480val

- fresh VQ seed123、fixed_val_first480_center_max300_raw_smpl_v1：重建FID2.616986，best仍170000/2.447121；Matching3.807415，R1/2/3=0.362500/0.514583/0.610417，Div7.764685/GT7.715767。较270k FID/R2/R3/native Body误差及Div偏差变差，Matching/R1改善；非生成FID或正式test。
- 十项严格口径：RMSE、Full MPJPE、Body non-root MPJPE、root误差、Full/Body joint FID、滑步、接触F1均待评；Hands MPJPE/Hands joint FID不适用。额外native22含root监控Body MPJPE101.259959mm不能替代严格Body non-root项。
- 实际/proc、日志及manifest确认InterGen49300、VQ276201活跃，三个队列存在。SSH远端GPU93%/8533MiB，保持运行、idle_since为空；无重启、停机或本机checkpoint下载。
### 2026-10-16 20:28 SGT｜VQ主步270000训练内480val

- fresh VQ seed123、fixed_val_first480_center_max300_raw_smpl_v1：重建FID2.519880，best仍170000/2.447121；Matching3.807768，R1/2/3=0.360417/0.529167/0.616667，Div7.759356/GT7.715767。较265k FID/Matching/R1/R3/native Body误差改善，R2及Div偏差变差；非生成FID或正式test。
- 十项严格口径：RMSE、Full MPJPE、Body non-root MPJPE、root误差、Full/Body joint FID、滑步、接触F1均待评；Hands MPJPE/Hands joint FID不适用。额外native22含root监控Body MPJPE101.052840mm不能替代严格Body non-root项。
- 实际/proc、日志及manifest确认InterGen48850、VQ272601活跃，三个队列存在。SSH远端GPU89%/8533MiB，保持运行、idle_since为空；无重启、停机或本机checkpoint下载。
### 2026-10-16 20:12 SGT｜VQ主步265000训练内480val

- fresh VQ seed123、fixed_val_first480_center_max300_raw_smpl_v1：重建FID2.662802，best仍170000/2.447121；Matching3.808388，R1/2/3=0.358333/0.531250/0.612500，Div7.758129/GT7.715767。较260k FID/Matching/R3/native Body误差变差，R1/R2及Div偏差改善；非生成FID或正式test。
- 十项严格口径：RMSE、Full MPJPE、Body non-root MPJPE、root误差、Full/Body joint FID、滑步、接触F1均待评；Hands MPJPE/Hands joint FID不适用。额外native22含root监控Body MPJPE101.713999mm不能替代严格Body non-root项。
- 实际/proc、日志及manifest确认InterGen48350、VQ268401活跃，三个队列按上游等待、phase空；本机controller53968存在。SSH远端nvidia-smi为100%/8533MiB，保持运行、idle_since为空；无重启、停机或本机checkpoint下载。云到云权重备份仍未验收。
### 2026-10-16 19:58 SGT｜VQ主步260000训练内480val

- fresh VQ seed123、fixed_val_first480_center_max300_raw_smpl_v1：重建FID2.553868，best仍170000/2.447121；Matching3.807034，R1/2/3=0.356250/0.516667/0.620833，Div7.767519/GT7.715767。较255k FID/Matching/R3/native Body误差改善，R1/R2及Div偏差变差；非生成FID或正式test。
- 十项严格口径：RMSE、Full MPJPE、Body non-root MPJPE、root误差、Full/Body joint FID、滑步、接触F1均待评；Hands MPJPE/Hands joint FID不适用。额外native22含root监控Body MPJPE101.380678mm不能替代严格Body non-root项。
- 实际/proc、日志及manifest确认InterGen47900、VQ最新记录264801活跃，三个队列存在。GPU37%/8533MiB，保持运行、idle_since为空；无重启、停机或本机checkpoint下载。
### 2026-10-16 19:28 SGT｜VQ主步255000训练内480val

- fresh VQ seed123、fixed_val_first480_center_max300_raw_smpl_v1：重建FID2.561431，best仍170000/2.447121；Matching3.808424，R1/2/3=0.370833/0.522917/0.614583，Div7.760464/GT7.715767。较250k FID/R1/R2改善，Matching/R3/native Body误差及Div偏差变差；非生成FID或正式test。
- 十项严格口径：RMSE、Full MPJPE、Body non-root MPJPE、root误差、Full/Body joint FID、滑步、接触F1均待评；Hands MPJPE/Hands joint FID不适用。额外native22含root监控Body MPJPE101.501718mm不能替代严格Body non-root项。
- 实际/proc、日志及manifest确认InterGen46950、VQ256001活跃，三个队列存在。GPU97%/8533MiB，保持运行、idle_since为空；无重启、停机或本机checkpoint下载。
### 2026-10-16 19:12 SGT｜VQ主步250000训练内480val

- fresh VQ seed123、fixed_val_first480_center_max300_raw_smpl_v1：重建FID2.570775，best仍170000/2.447121；Matching3.807431，R1/2/3=0.358333/0.516667/0.618750，Div7.754082/GT7.715767。较245k FID/Matching/R1/R3/native Body误差及Div偏差改善，R2下降；非生成FID或正式test。
- 十项严格口径：RMSE、Full MPJPE、Body non-root MPJPE、root误差、Full/Body joint FID、滑步、接触F1均待评；Hands MPJPE/Hands joint FID不适用。额外native22含root监控Body MPJPE101.391918mm不能替代严格Body non-root项。
- 实际/proc、日志及manifest确认InterGen46450、VQ251801活跃，三个队列按上游等待、phase空；本机controller53968存在。GPU96%/8533MiB，保持运行、idle_since为空；无重启、停机或本机checkpoint下载。云到云权重备份仍未验收。
### 2026-10-16 18:57 SGT｜VQ主步245000训练内480val

- fresh VQ seed123、fixed_val_first480_center_max300_raw_smpl_v1：重建FID2.590852，best仍170000/2.447121；Matching3.808852，R1/2/3=0.341667/0.522917/0.612500，Div7.754686/GT7.715767。较240k FID/R1/R3/native Body误差及Div偏差变差，Matching/R2改善；非生成FID或正式test。
- 十项严格口径：RMSE、Full MPJPE、Body non-root MPJPE、root误差、Full/Body joint FID、滑步、接触F1均待评；Hands MPJPE/Hands joint FID不适用。额外native22含root监控Body MPJPE101.804265mm不能替代严格Body non-root项。
- 实际/proc、日志及manifest确认InterGen46000、VQ248401活跃，三个队列存在。GPU99%/8533MiB，保持运行、idle_since为空；无重启、停机或本机checkpoint下载。
### 2026-10-16 18:42 SGT｜VQ主步240000训练内480val

- fresh VQ seed123、fixed_val_first480_center_max300_raw_smpl_v1：重建FID2.536262，best仍170000/2.447121；Matching3.808947，R1/2/3=0.358333/0.512500/0.614583，Div7.748394/GT7.715767。较235k FID/R1/R2及Div偏差改善，Matching/R3/native Body误差变差；非生成FID或正式test。
- 十项严格口径：RMSE、Full MPJPE、Body non-root MPJPE、root误差、Full/Body joint FID、滑步、接触F1均待评；Hands MPJPE/Hands joint FID不适用。额外native22含root监控Body MPJPE101.626441mm不能替代严格Body non-root项。
- 实际/proc、日志及manifest确认InterGen45500、VQ244601活跃，三个队列存在。GPU99%/8533MiB，保持运行、idle_since为空；无重启、停机或本机checkpoint下载。
### 2026-10-16 18:11 SGT｜VQ主步235000训练内480val

- fresh VQ seed123、fixed_val_first480_center_max300_raw_smpl_v1：重建FID2.619981，best仍170000/2.447121；Matching3.806505，R1/2/3=0.354167/0.510417/0.618750，Div7.763340/GT7.715767。较230k FID/Matching/native Body误差及Div偏差改善，R1/R2/R3下降；非生成FID或正式test。
- 十项严格口径：RMSE、Full MPJPE、Body non-root MPJPE、root误差、Full/Body joint FID、滑步、接触F1均待评；Hands MPJPE/Hands joint FID不适用。额外native22含root监控Body MPJPE101.121360mm不能替代严格Body non-root项。
- 实际/proc、日志及manifest确认InterGen44550、VQ235201活跃，三个队列按上游等待、phase空；本机controller53968存在。GPU93%/8533MiB，保持运行、idle_since为空；无重启、停机或本机checkpoint下载。云到云权重备份仍未验收。
### 2026-10-16 17:57 SGT｜VQ主步230000训练内480val

- fresh VQ seed123、fixed_val_first480_center_max300_raw_smpl_v1：重建FID2.643147，best仍170000/2.447121；Matching3.807224，R1/2/3=0.364583/0.525000/0.627083，Div7.774072/GT7.715767。较225k FID及Div偏差变差，Matching/R1/R2/R3/native Body误差改善；非生成FID或正式test。
- 十项严格口径：RMSE、Full MPJPE、Body non-root MPJPE、root误差、Full/Body joint FID、滑步、接触F1均待评；Hands MPJPE/Hands joint FID不适用。额外native22含root监控Body MPJPE101.401290mm不能替代严格Body non-root项。
- 实际/proc、日志及manifest确认InterGen44100、VQ232001活跃，三个队列存在。GPU99%/8533MiB，保持运行、idle_since为空；无重启、停机或本机checkpoint下载。
### 2026-10-16 17:42 SGT｜VQ主步225000训练内480val

- fresh VQ seed123、fixed_val_first480_center_max300_raw_smpl_v1：重建FID2.580982，best仍170000/2.447121；Matching3.808372，R1/2/3=0.354167/0.504167/0.618750，Div7.767467/GT7.715767。较220k FID/Matching/R1/R3/native Body误差改善，R2及Div偏差变差；非生成FID或正式test。
- 十项严格口径：RMSE、Full MPJPE、Body non-root MPJPE、root误差、Full/Body joint FID、滑步、接触F1均待评；Hands MPJPE/Hands joint FID不适用。额外native22含root监控Body MPJPE101.519599mm不能替代严格Body non-root项。
- 实际/proc、日志及manifest确认InterGen43650、VQ228401活跃，三个队列存在。GPU100%/8533MiB，保持运行、idle_since为空；无重启、停机或本机checkpoint下载。
### 2026-10-16 17:27 SGT｜VQ主步220000训练内480val

- fresh VQ seed123、fixed_val_first480_center_max300_raw_smpl_v1：重建FID2.587893，best仍170000/2.447121；Matching3.809700，R1/2/3=0.337500/0.506250/0.604167，Div7.746142/GT7.715767。较215k FID及Div偏差改善，Matching/R1/R2/R3/native Body误差变差；非生成FID或正式test。
- 十项严格口径：RMSE、Full MPJPE、Body non-root MPJPE、root误差、Full/Body joint FID、滑步、接触F1均待评；Hands MPJPE/Hands joint FID不适用。额外native22含root监控Body MPJPE102.306241mm不能替代严格Body non-root项。
- 实际/proc、日志及manifest确认InterGen43200、VQ224601活跃，三个队列存在。GPU99%/8533MiB，保持运行、idle_since为空；无重启、停机或本机checkpoint下载。
### 2026-10-16 16:57 SGT｜VQ主步215000训练内480val

- fresh VQ seed123、fixed_val_first480_center_max300_raw_smpl_v1：重建FID2.595186，best仍170000/2.447121；Matching3.808122，R1/2/3=0.350000/0.516667/0.622917，Div7.760356/GT7.715767。较210k FID/native Body误差改善，Matching/R1/R2/R3及Div偏差变差；非生成FID或正式test。
- 十项严格口径：RMSE、Full MPJPE、Body non-root MPJPE、root误差、Full/Body joint FID、滑步、接触F1均待评；Hands MPJPE/Hands joint FID不适用。额外native22含root监控Body MPJPE102.181846mm不能替代严格Body non-root项。
- 实际/proc、日志及manifest确认InterGen42250、VQ215601活跃，三个队列存在。GPU99%/8533MiB，保持运行、idle_since为空；无重启、停机或本机checkpoint下载。
### 2026-10-16 16:42 SGT｜VQ主步210000训练内480val

- fresh VQ seed123、fixed_val_first480_center_max300_raw_smpl_v1：重建FID2.689062，best仍170000/2.447121；Matching3.807462，R1/2/3=0.356250/0.520833/0.625000，Div7.758219/GT7.715767。较205k FID/R2/R3/native Body误差变差，Matching/R1及Div偏差改善；非生成FID或正式test。
- 十项严格口径：RMSE、Full MPJPE、Body non-root MPJPE、root误差、Full/Body joint FID、滑步、接触F1均待评；Hands MPJPE/Hands joint FID不适用。额外native22含root监控Body MPJPE102.277618mm不能替代严格Body non-root项。
- 实际/proc、日志及manifest确认InterGen41800、VQ212001活跃，三个队列存在。GPU99%/8533MiB，保持运行、idle_since为空；无重启、停机或本机checkpoint下载。
### 2026-10-16 16:27 SGT｜VQ主步205000训练内480val

- fresh VQ seed123、fixed_val_first480_center_max300_raw_smpl_v1：重建FID2.607886，best仍170000/2.447121；Matching3.808267，R1/2/3=0.354167/0.529167/0.645833，Div7.769372/GT7.715767。较200k FID及Div偏差变差，Matching/R2/R3/native Body误差改善；非生成FID或正式test。
- 十项严格口径：RMSE、Full MPJPE、Body non-root MPJPE、root误差、Full/Body joint FID、滑步、接触F1均待评；Hands MPJPE/Hands joint FID不适用。额外native22含root监控Body MPJPE101.823750mm不能替代严格Body non-root项。
- 实际/proc、日志及manifest确认InterGen41300、VQ208401活跃，三个队列存在。GPU99%/8533MiB，保持运行、idle_since为空；无重启、停机或本机checkpoint下载。
### 2026-10-16 16:11 SGT｜VQ主步200000训练内480val

- fresh VQ seed123、fixed_val_first480_center_max300_raw_smpl_v1：重建FID2.549937，best仍170000/2.447121；Matching3.809034，R1/2/3=0.354167/0.518750/0.625000，Div7.748456/GT7.715767。较195k FID/R1/R2及Div偏差改善，Matching/R3/native Body误差变差；非生成FID或正式test。
- 十项严格口径：RMSE、Full MPJPE、Body non-root MPJPE、root误差、Full/Body joint FID、滑步、接触F1均待评；Hands MPJPE/Hands joint FID不适用。额外native22含root监控Body MPJPE101.851631mm不能替代严格Body non-root项。
- 实际/proc、日志及manifest确认InterGen40750、VQ204201活跃，三个队列按上游等待、phase均空；本机controller53968存在。GPU100%/8533MiB，保持运行、idle_since为空；无重启、停机或本机checkpoint下载。云到云权重备份仍未验收。
### 2026-10-16 15:58 SGT｜InterGen40000完整480val

- 原InterGen rotation scratch step40000：FID39.267600，较30k35.852000进一步变差；best仍20000/24.314143。Matching4.003752，R1/2/3=0.104167/0.172917/0.235417，Div7.759773/GT7.741262。480条固定validation、seed20261006、DDIM50/CFG3.5、官方InterCLIP、batch96，artifact status=complete；MM待评、非独立test。Matching/Div偏差改善，R2/R3下降。
- 实际/proc、持续日志与manifest确认InterGen40300、VQ最新记录199801活跃；VQ200k完整artifact尚无，最新仍195k，best170k/2.447121。三个队列进程存在，GPU13%/8533MiB，保持运行、idle_since为空；无重启、停机或本机checkpoint下载。
### 2026-10-16 15:42 SGT｜VQ主步195000训练内480val

- fresh VQ seed123、fixed_val_first480_center_max300_raw_smpl_v1：重建FID2.578474，best仍170000/2.447121；Matching3.808635，R1/2/3=0.337500/0.514583/0.633333，Div7.751590/GT7.715767。较190k FID/Matching/R1/R2变差，R3/native Body误差及Div偏差改善；非生成FID或正式test。
- 十项严格口径：RMSE、Full MPJPE、Body non-root MPJPE、root误差、Full/Body joint FID、滑步、接触F1均待评；Hands MPJPE/Hands joint FID不适用。额外native22含root监控Body MPJPE101.433007mm不能替代严格Body non-root项。
- 实际/proc、日志及manifest确认InterGen40000验证阶段、VQ195401活跃，三个队列存在。InterGen40k完整评分未确认，best仍20k/24.314143。GPU100%/8533MiB，保持运行、idle_since为空；无重启、停机或本机checkpoint下载。
### 2026-10-16 15:27 SGT｜VQ主步190000训练内480val

- fresh VQ seed123、fixed_val_first480_center_max300_raw_smpl_v1：重建FID2.544361，best仍170000/2.447121；Matching3.808058，R1/2/3=0.354167/0.522917/0.614583，Div7.766973/GT7.715767。较185k FID/Matching/R2/native Body误差改善，R1/R3及Div偏差变差；非生成FID或正式test。
- 十项严格口径：RMSE、Full MPJPE、Body non-root MPJPE、root误差、Full/Body joint FID、滑步、接触F1均待评；Hands MPJPE/Hands joint FID不适用。额外native22含root监控Body MPJPE101.581232mm不能替代严格Body non-root项。
- 实际/proc、日志及manifest确认InterGen39950、VQ191801活跃，三个队列存在。GPU99%/8533MiB，保持运行、idle_since为空；无重启、停机或本机checkpoint下载。
### 2026-10-16 15:10 SGT｜VQ主步185000训练内480val

- fresh VQ seed123、fixed_val_first480_center_max300_raw_smpl_v1：重建FID2.621966，best仍170000/2.447121；Matching3.808715，R1/2/3=0.358333/0.506250/0.631250，Div7.758382/GT7.715767。较180k FID/Matching/R1/R3/native Body误差改善，R2及Div偏差变差；非历史185k恢复、非生成FID或正式test。
- 十项严格口径：RMSE、Full MPJPE、Body non-root MPJPE、root误差、Full/Body joint FID、滑步、接触F1均待评；Hands MPJPE/Hands joint FID不适用。额外native22含root监控Body MPJPE101.903804mm不能替代严格Body non-root项。
- 实际/proc、日志及manifest确认InterGen39400、VQ187401活跃，三个队列分别按上游等待、phase均空。本机controller53968存在。GPU98%/8533MiB，保持运行、idle_since为空；无重启、停机或本机checkpoint下载。云到云权重备份仍未验收。
### 2026-10-16 14:55 SGT｜VQ主步180000训练内480val

- fresh VQ seed123、fixed_val_first480_center_max300_raw_smpl_v1：重建FID2.700768，best仍170000/2.447121；Matching3.809502，R1/2/3=0.343750/0.514583/0.610417，Div7.736070/GT7.715767。较175k FID/Matching/R1/R2/R3/native Body误差变差，Div偏差改善；非生成FID或正式test。
- 十项严格口径：RMSE、Full MPJPE、Body non-root MPJPE、root误差、Full/Body joint FID、滑步、接触F1均待评；Hands MPJPE/Hands joint FID不适用。额外native22含root监控Body MPJPE107.634513mm不能替代严格Body non-root项。
- 实际/proc、日志及manifest确认InterGen38950、VQ183601活跃，三个队列存在。GPU96%/8533MiB，保持运行、idle_since为空；无重启、停机或本机checkpoint下载。
### 2026-10-16 14:40 SGT｜VQ主步175000训练内480val

- fresh VQ seed123、fixed_val_first480_center_max300_raw_smpl_v1：重建FID2.653234，best仍170000/2.447121；Matching3.808141，R1/2/3=0.350000/0.518750/0.627083，Div7.766299/GT7.715767。较170k FID/Matching/R1/native Body误差及Div偏差变差，R2/R3改善；非生成FID或正式test。
- 十项严格口径：RMSE、Full MPJPE、Body non-root MPJPE、root误差、Full/Body joint FID、滑步、接触F1均待评；Hands MPJPE/Hands joint FID不适用。额外native22含root监控Body MPJPE102.949831mm不能替代严格Body non-root项。
- 实际/proc、日志及manifest确认InterGen38450、VQ最新记录179801活跃，三个队列存在。GPU26%/8533MiB，保持运行、idle_since为空；无重启、停机或本机checkpoint下载。
### 2026-10-16 14:10 SGT｜VQ主步170000训练内480val

- fresh VQ seed123、fixed_val_first480_center_max300_raw_smpl_v1：重建FID2.447121，刷新best=170000/2.447121；Matching3.807248，R1/2/3=0.356250/0.512500/0.614583，Div7.756771/GT7.715767。较165k FID/native Body误差及Div偏差改善，Matching/R1/R2/R3变差；非生成FID或正式test。
- 十项严格口径：RMSE、Full MPJPE、Body non-root MPJPE、root误差、Full/Body joint FID、滑步、接触F1均待评；Hands MPJPE/Hands joint FID不适用。额外native22含root监控Body MPJPE102.289230mm不能替代严格Body non-root项。
- 实际/proc、日志及manifest确认InterGen37550、VQ171001活跃，三个队列存在。GPU99%/8533MiB，保持运行、idle_since为空；无重启、停机或本机checkpoint下载。
### 2026-10-16 13:55 SGT｜VQ主步165000训练内480val

- fresh VQ seed123、fixed_val_first480_center_max300_raw_smpl_v1：重建FID2.666095，best仍150000/2.519049；Matching3.807127，R1/2/3=0.360417/0.527083/0.629167，Div7.772364/GT7.715767。较160k FID/native Body误差及Div偏差变差，Matching/R1/R2/R3改善；非生成FID或正式test。
- 十项严格口径：RMSE、Full MPJPE、Body non-root MPJPE、root误差、Full/Body joint FID、滑步、接触F1均待评；Hands MPJPE/Hands joint FID不适用。额外native22含root监控Body MPJPE102.475987mm不能替代严格Body non-root项。
- 实际/proc、日志及manifest确认InterGen37050、VQ167401活跃，三个队列存在。GPU99%/8533MiB，保持运行、idle_since为空；无重启、停机或本机checkpoint下载。
### 2026-10-16 13:40 SGT｜VQ主步160000训练内480val

- fresh VQ seed123、fixed_val_first480_center_max300_raw_smpl_v1：重建FID2.618327，best仍150000/2.519049；Matching3.808372，R1/2/3=0.354167/0.508333/0.612500，Div7.766007/GT7.715767。较155k FID/native Body误差改善，Matching/R2及Div偏差变差；非生成FID或正式test。
- 十项严格口径：RMSE、Full MPJPE、Body non-root MPJPE、root误差、Full/Body joint FID、滑步、接触F1均待评；Hands MPJPE/Hands joint FID不适用。额外native22含root监控Body MPJPE102.459898mm不能替代严格Body non-root项。
- 实际/proc、日志及manifest确认InterGen36600、VQ163601活跃，三个队列存在。GPU99%/8533MiB，保持运行、idle_since为空；无重启、停机或本机checkpoint下载。
### 2026-10-16 13:25 SGT｜VQ主步155000训练内480val

- fresh VQ seed123、fixed_val_first480_center_max300_raw_smpl_v1：重建FID2.690010，best仍150000/2.519049；Matching3.807979，R1/2/3=0.354167/0.520833/0.612500，Div7.763233/GT7.715767。较150k FID/R3/native Body误差及Div偏差变差，Matching/R1/R2改善；非生成FID或正式test。
- 十项严格口径：RMSE、Full MPJPE、Body non-root MPJPE、root误差、Full/Body joint FID、滑步、接触F1均待评；Hands MPJPE/Hands joint FID不适用。额外native22含root监控Body MPJPE103.416279mm不能替代严格Body non-root项。
- 实际/proc、日志及manifest确认InterGen36150、VQ最新记录159801活跃，三个队列存在；160k验证artifact尚无，不作失败。GPU68%/8533MiB，保持运行、idle_since为空；无重启、停机或本机checkpoint下载。
### 2026-10-16 12:55 SGT｜VQ主步150000训练内480val

- fresh VQ seed123、fixed_val_first480_center_max300_raw_smpl_v1：重建FID2.519049，刷新best=150000/2.519049；Matching3.808302，R1/2/3=0.337500/0.508333/0.620833，Div7.749679/GT7.715767。较145k FID/Matching/R2/native Body误差及Div偏差改善，R1下降；非生成FID或正式test。
- 十项严格口径：RMSE、Full MPJPE、Body non-root MPJPE、root误差、Full/Body joint FID、滑步、接触F1均待评；Hands MPJPE/Hands joint FID不适用。额外native22含root监控Body MPJPE102.948434mm不能替代严格Body non-root项。
- 实际/proc、日志及manifest确认InterGen35200、VQ151001活跃，三个队列存在。GPU90%/8533MiB，保持运行、idle_since为空；无重启、停机或本机checkpoint下载。
### 2026-10-16 12:40 SGT｜VQ主步145000训练内480val

- fresh VQ seed123、fixed_val_first480_center_max300_raw_smpl_v1：重建FID2.729096，best仍115000/2.566480；Matching3.808605，R1/2/3=0.339583/0.495833/0.620833，Div7.779064/GT7.715767。较140k FID/R1/R2/native Body误差及Div偏差变差，Matching改善；非生成FID或正式test。
- 十项严格口径：RMSE、Full MPJPE、Body non-root MPJPE、root误差、Full/Body joint FID、滑步、接触F1均待评；Hands MPJPE/Hands joint FID不适用。额外native22含root监控Body MPJPE103.118212mm不能替代严格Body non-root项。
- 实际/proc、日志及manifest确认InterGen34700、VQ147201活跃，三个队列存在。GPU99%/8533MiB，保持运行、idle_since为空；无重启、停机或本机checkpoint下载。
### 2026-10-16 12:25 SGT｜VQ主步140000训练内480val

- fresh VQ seed123、fixed_val_first480_center_max300_raw_smpl_v1：重建FID2.665649，best仍115000/2.566480；Matching3.809026，R1/2/3=0.347917/0.497917/0.620833，Div7.750368/GT7.715767。较135k FID/Matching/R2/R3变差，R1/native Body误差及Div偏差改善；非生成FID或正式test。
- 十项严格口径：RMSE、Full MPJPE、Body non-root MPJPE、root误差、Full/Body joint FID、滑步、接触F1均待评；Hands MPJPE/Hands joint FID不适用。额外native22含root监控Body MPJPE102.771148mm不能替代严格Body non-root项。
- 实际/proc、日志及manifest确认InterGen34250、VQ143601活跃，三个队列存在。GPU96%/8533MiB，保持运行、idle_since为空；无重启、停机或本机checkpoint下载。
### 2026-10-16 12:10 SGT｜VQ主步135000训练内480val

- fresh VQ seed123、fixed_val_first480_center_max300_raw_smpl_v1：重建FID2.663697，best仍115000/2.566480；Matching3.806766，R1/2/3=0.329167/0.512500/0.627083，Div7.768274/GT7.715767。较130k FID/R1及Div偏差变差，Matching/R2/native Body误差改善；非生成FID或正式test。
- 十项严格口径：RMSE、Full MPJPE、Body non-root MPJPE、root误差、Full/Body joint FID、滑步、接触F1均待评；Hands MPJPE/Hands joint FID不适用。额外native22含root监控Body MPJPE103.844796mm不能替代严格Body non-root项。
- 实际/proc、日志及manifest确认InterGen33800、VQ139801活跃，三个队列存在。GPU100%/8533MiB，保持运行、idle_since为空；无重启、停机或本机checkpoint下载。
### 2026-10-16 11:40 SGT｜VQ主步130000训练内480val

- fresh VQ seed123、fixed_val_first480_center_max300_raw_smpl_v1：重建FID2.586791，best仍115000/2.566480；Matching3.808081，R1/2/3=0.356250/0.493750/0.627083，Div7.744164/GT7.715767。较125k FID/Matching及Div偏差改善，R1/R2/native Body误差变差；非生成FID或正式test。
- 十项严格口径：RMSE、Full MPJPE、Body non-root MPJPE、root误差、Full/Body joint FID、滑步、接触F1均待评；Hands MPJPE/Hands joint FID不适用。额外native22含root监控Body MPJPE105.531544mm不能替代严格Body non-root项。
- 实际/proc、日志及manifest确认InterGen32850、VQ130801活跃，三个队列存在。GPU92%/8533MiB，保持运行、idle_since为空；无重启、停机或本机checkpoint下载。
### 2026-10-16 11:25 SGT｜VQ主步125000训练内480val

- fresh VQ seed123、fixed_val_first480_center_max300_raw_smpl_v1：重建FID2.856941，best仍115000/2.566480；Matching3.808280，R1/2/3=0.358333/0.525000/0.627083，Div7.782913/GT7.715767。较120k FID/Matching/R3/native Body误差及Div偏差变差，R1/R2改善；非生成FID或正式test。
- 十项严格口径：RMSE、Full MPJPE、Body non-root MPJPE、root误差、Full/Body joint FID、滑步、接触F1均待评；Hands MPJPE/Hands joint FID不适用。额外native22含root监控Body MPJPE104.126064mm不能替代严格Body non-root项。
- 实际/proc、日志及manifest确认InterGen32400、VQ127001活跃，三个队列存在。GPU98%/8533MiB，保持运行、idle_since为空；无重启、停机或本机checkpoint下载。
### 2026-10-16 11:10 SGT｜VQ主步120000训练内480val

- fresh VQ seed123、fixed_val_first480_center_max300_raw_smpl_v1：重建FID2.663476，best仍115000/2.566480；Matching3.807885，R1/2/3=0.354167/0.514583/0.631250，Div7.755010/GT7.715767。较115k FID/Matching及Div偏差变差，R1/R2/R3/native Body误差改善；非生成FID或正式test。
- 十项严格口径：RMSE、Full MPJPE、Body non-root MPJPE、root误差、Full/Body joint FID、滑步、接触F1均待评；Hands MPJPE/Hands joint FID不适用。额外native22含root监控Body MPJPE103.346982mm不能替代严格Body non-root项。
- 实际/proc、日志及manifest确认InterGen31900、VQ123401活跃，三个队列存在。GPU99%/8533MiB，保持运行、idle_since为空；无重启、停机或本机checkpoint下载。
### 2026-10-16 10:55 SGT｜VQ主步115000训练内480val

- fresh VQ seed123、fixed_val_first480_center_max300_raw_smpl_v1：重建FID2.566480，刷新best=115000/2.566480；Matching3.807768，R1/2/3=0.343750/0.512500/0.616667，Div7.735664/GT7.715767。较110k FID/native Body误差及Div偏差改善，Matching/R1/R2/R3变差；非生成FID或正式test。
- 十项严格口径：RMSE、Full MPJPE、Body non-root MPJPE、root误差、Full/Body joint FID、滑步、接触F1均待评；Hands MPJPE/Hands joint FID不适用。额外native22含root监控Body MPJPE104.184688mm不能替代严格Body non-root项。
- 实际/proc、日志及manifest确认InterGen31450、VQ119801活跃，三个队列存在。GPU92%/8533MiB，保持运行、idle_since为空；无重启、停机或本机checkpoint下载。
### 2026-10-16 10:25 SGT｜VQ主步110000训练内480val

- fresh VQ seed123、fixed_val_first480_center_max300_raw_smpl_v1：重建FID2.671110，best仍105000/2.574446；Matching3.807585，R1/2/3=0.358333/0.514583/0.629167，Div7.775691/GT7.715767。较105k FID/Matching/R2及Div偏差变差，R1/R3/native Body误差改善；非生成FID或正式test。
- 十项严格口径：RMSE、Full MPJPE、Body non-root MPJPE、root误差、Full/Body joint FID、滑步、接触F1均待评；Hands MPJPE/Hands joint FID不适用。额外native22含root监控Body MPJPE104.632332mm不能替代严格Body non-root项。
- 实际/proc、日志及manifest确认InterGen30550、VQ110801活跃，三个队列存在。GPU100%/8533MiB，保持运行、idle_since为空；无重启、停机或本机checkpoint下载。
### 2026-10-16 10:10 SGT｜InterGen30k与VQ105k完整480val

- InterGen scratch step30000：FID35.852000，较20k变差，best仍20000/24.314143；Matching4.011828，R1/2/3=0.104167/0.191667/0.239583，Div7.980249/GT7.741262。artifact status=complete、480条、seed20261006、DDIM50/CFG3.5、官方InterCLIP训练内验证；MM待评，非独立test。训练已继续30050。
- fresh VQ seed123、fixed_val_first480_center_max300_raw_smpl_v1：主步105000重建FID2.574446，刷新best=105000/2.574446；Matching3.807006，R1/2/3=0.345833/0.525000/0.622917，Div7.757761/GT7.715767。较100k FID/Matching/native Body误差改善，R2/R3下降；非生成FID或正式test。
- VQ十项严格口径：RMSE、Full MPJPE、Body non-root MPJPE、root误差、Full/Body joint FID、滑步、接触F1均待评；Hands MPJPE/Hands joint FID不适用。额外native22含root监控Body MPJPE104.842863mm不能替代严格Body non-root项。
- 实际/proc、日志及manifest确认两项训练活跃（VQ107001），三个队列存在。GPU95%/8533MiB，保持运行、idle_since为空；无重启、停机或本机checkpoint下载。
### 2026-10-16 09:55 SGT｜VQ主步100000训练内480val

- fresh VQ seed123、fixed_val_first480_center_max300_raw_smpl_v1：重建FID2.820426，best仍65000/2.681060；Matching3.808834，R1/2/3=0.345833/0.531250/0.633333，Div7.783712/GT7.715767。较95k FID/R1/native Body误差变差，Matching/R2/R3改善；非生成FID或正式test。
- 十项严格口径：RMSE、Full MPJPE、Body non-root MPJPE、root误差、Full/Body joint FID、滑步、接触F1均待评；Hands MPJPE/Hands joint FID不适用。额外native22含root监控Body MPJPE105.544894mm不能替代严格Body non-root项。
- 实际/proc、日志及manifest确认InterGen30000验证阶段、VQ103001活跃，三个队列存在。InterGen新30k完整评分尚未确认，best仍20k/24.314143。GPU98%/8533MiB，保持运行、idle_since为空；无重启、停机或本机checkpoint下载。
### 2026-10-16 09:40 SGT｜VQ主步95000训练内480val

- fresh VQ seed123、fixed_val_first480_center_max300_raw_smpl_v1：重建FID2.745579，best仍65000/2.681060；Matching3.809304，R1/2/3=0.350000/0.520833/0.622917，Div7.786695/GT7.715767。较90k FID/R2/R3改善，Matching/R1/native Body误差及Div偏差变差；非生成FID或正式test。
- 十项严格口径：RMSE、Full MPJPE、Body non-root MPJPE、root误差、Full/Body joint FID、滑步、接触F1均待评；Hands MPJPE/Hands joint FID不适用。额外native22含root监控Body MPJPE105.530840mm不能替代严格Body non-root项。
- 实际/proc、日志及manifest确认InterGen29700、VQ99401活跃，三个队列存在。GPU91%/8533MiB，保持运行、idle_since为空；无重启、停机或本机checkpoint下载。
### 2026-10-16 09:10 SGT｜VQ主步90000训练内480val

- fresh VQ seed123、fixed_val_first480_center_max300_raw_smpl_v1：重建FID2.925590，best仍65000/2.681060；Matching3.809106，R1/2/3=0.360417/0.508333/0.606250，Div7.739639/GT7.715767。较85k FID/Matching/R1/R2/native Body误差改善，R3下降；非生成FID或正式test。
- 十项严格口径：RMSE、Full MPJPE、Body non-root MPJPE、root误差、Full/Body joint FID、滑步、接触F1均待评；Hands MPJPE/Hands joint FID不适用。额外native22含root监控Body MPJPE104.771998mm不能替代严格Body non-root项。
- 实际/proc、日志及manifest确认InterGen28750、VQ90401活跃，三个队列存在。GPU100%/8533MiB，保持运行、idle_since为空；无重启、停机或本机checkpoint下载。
### 2026-10-16 08:55 SGT｜VQ主步85000训练内480val

- fresh VQ seed123、fixed_val_first480_center_max300_raw_smpl_v1：重建FID3.005204，best仍65000/2.681060；Matching3.811837，R1/2/3=0.356250/0.500000/0.612500，Div7.761118/GT7.715767。较80k FID/Matching/R2变差，R1/R3/native Body误差改善；非生成FID或正式test。
- 十项严格口径：RMSE、Full MPJPE、Body non-root MPJPE、root误差、Full/Body joint FID、滑步、接触F1均待评；Hands MPJPE/Hands joint FID不适用。额外native22含root监控Body MPJPE104.925321mm不能替代严格Body non-root项。
- 实际/proc、日志及manifest确认InterGen28300、VQ86601活跃，三个队列存在。GPU99%/8533MiB，保持运行、idle_since为空；无重启、停机或本机checkpoint下载。
### 2026-10-16 08:40 SGT｜VQ主步80000训练内480val

- fresh VQ seed123、fixed_val_first480_center_max300_raw_smpl_v1：重建FID2.747533，best仍65000/2.681060；Matching3.809055，R1/2/3=0.350000/0.502083/0.606250，Div7.736104/GT7.715767。较75k FID/Matching/native Body误差改善，R2/R3下降；非生成FID或正式test。
- 十项严格口径：RMSE、Full MPJPE、Body non-root MPJPE、root误差、Full/Body joint FID、滑步、接触F1均待评；Hands MPJPE/Hands joint FID不适用。额外native22含root监控Body MPJPE105.323285mm不能替代严格Body non-root项。
- 实际/proc、持续日志和manifest确认InterGen27850、VQ83001活跃，三个队列存在。GPU99%/8533MiB，保持运行、idle_since为空；无重启、停机或本机checkpoint下载。
### 2026-10-16 08:25 SGT｜VQ主步75000训练内480val

- fresh VQ seed123、fixed_val_first480_center_max300_raw_smpl_v1：重建FID3.102185，best仍65000/2.681060；Matching3.811388，R1/2/3=0.350000/0.506250/0.610417，Div7.746270/GT7.715767。较70k FID/Matching/R2/R3及native Body误差变差，R1改善；非生成FID或正式test。
- 十项严格口径：RMSE、Full MPJPE、Body non-root MPJPE、root误差、Full/Body joint FID、滑步、接触F1均待评；Hands MPJPE/Hands joint FID不适用。额外native22含root监控Body MPJPE112.060877mm不能替代严格Body non-root项。
- 实际/proc、日志及manifest确认InterGen27350、VQ79201活跃，三个队列存在。GPU99%/8533MiB，保持运行、idle_since为空；无重启、停机或本机checkpoint下载。
### 2026-10-16 08:10 SGT｜VQ主步70000训练内480val

- fresh VQ seed123、fixed_val_first480_center_max300_raw_smpl_v1：重建FID2.693309，best仍65000/2.681060；Matching3.809607，R1/2/3=0.333333/0.516667/0.639583，Div7.752460/GT7.715767。较65k FID/Matching/R1变差，Body误差改善；非生成FID或正式test。
- 十项严格口径：RMSE、Full MPJPE、Body non-root MPJPE、root误差、Full/Body joint FID、滑步、接触F1均待评；Hands MPJPE/Hands joint FID不适用。额外native22含root监控Body MPJPE104.785631mm不能替代严格Body non-root项，也不移用60k独立冻结评分。
- 实际/proc、持续日志及manifest确认InterGen26900、VQ最新记录74801活跃，三个队列存在。GPU瞬时8%/8533MiB，保持运行、idle_since为空；无重启、停机或本机checkpoint下载。
### 2026-10-16 07:39 SGT｜VQ主步65000完整480val

- fresh VQ seed123、fixed_val_first480_center_max300_raw_smpl_v1：重建FID2.681060，刷新best=65000/2.681060；Matching3.807881，R1/2/3=0.345833/0.506250/0.610417，Div7.754404/GT7.715767。较60k FID/Matching改善，R1/R2/R3下降、Body误差变差；非生成FID或正式test。
- 十项质量：RMSE待评；Full MPJPE待评；原生22joint Body MPJPE108.376098mm；Hands MPJPE不适用；Root误差待评；Full/Body joint FID待评；Hands joint FID不适用；滑步待评；接触F1待评。
- 实际/proc、日志和manifest确认InterGen25950、VQ66201活跃，三个队列存在。GPU98%/8533MiB，保持运行、idle_since为空；无重启、停机或本机checkpoint下载。
### 2026-10-16 07:24 SGT｜VQ主步60000完整480val

- fresh VQ seed123、fixed_val_first480_center_max300_raw_smpl_v1：重建FID2.695084，刷新best=60000/2.695084；Matching3.810743，R1/2/3=0.358333/0.510417/0.622917，Div7.746598/GT7.715767。较55k FID/Body改善，Matching及R2/R3变差；非生成FID或正式test。
- 十项质量：RMSE待评；Full MPJPE待评；原生22joint Body MPJPE107.325754mm；Hands MPJPE不适用；Root误差待评；Full/Body joint FID待评；Hands joint FID不适用；滑步待评；接触F1待评。
- 实际/proc、持续日志及manifest确认InterGen25450、VQ62401活跃，三个队列存在。GPU99%/8533MiB，保持运行、idle_since为空；无重启、停机或本机checkpoint下载。
### 2026-10-16 07:09 SGT｜VQ主步55000完整480val

- fresh VQ seed123、fixed_val_first480_center_max300_raw_smpl_v1：重建FID2.866757，best仍35000/2.827235；Matching3.810460，R1/2/3=0.352083/0.525000/0.627083，Div7.719928/GT7.715767。FID较50k改善，但Matching、R1/R2及Body误差变差；非生成FID或正式test。
- 十项质量：RMSE待评；Full MPJPE待评；原生22joint Body MPJPE108.769865mm；Hands MPJPE不适用；Root误差待评；Full/Body joint FID待评；Hands joint FID不适用；滑步待评；接触F1待评。
- /proc、持续日志及manifest确认InterGen24950、VQ58801活跃，三个队列存在。GPU98%/8533MiB，保持运行、idle_since为空；无重启、停机或本机checkpoint下载。
### 2026-10-16 06:54 SGT｜VQ主步50000完整480val

- fresh VQ seed123、fixed_val_first480_center_max300_raw_smpl_v1：重建FID2.925748，best仍35000/2.827235；Matching3.809512，R1/2/3=0.358333/0.529167/0.625000，Div7.731749/GT7.715767。FID较45k略好但未刷新最佳；非生成FID或正式test。
- 十项质量：RMSE待评；Full MPJPE待评；原生22joint Body MPJPE107.314958mm；Hands MPJPE不适用；Root误差待评；Full/Body joint FID待评；Hands joint FID不适用；滑步待评；接触F1待评。
- /proc、持续日志及manifest确认InterGen24500、VQ最新记录54801活跃，三个队列存在。GPU瞬时7%/8533MiB不代表闲置，保持运行、idle_since为空；无重启、停机或本机checkpoint下载。
### 2026-10-16 06:24 SGT｜VQ主步45000完整480val

- fresh VQ seed123、fixed_val_first480_center_max300_raw_smpl_v1：重建FID2.940790，best仍35000/2.827235；Matching3.809457，R1/2/3=0.358333/0.533333/0.616667，Div7.735402/GT7.715767。FID及Body误差较40k略变差，非全项收益；非生成FID或正式test。
- 十项质量：RMSE待评；Full MPJPE待评；原生22joint Body MPJPE108.947519mm；Hands MPJPE不适用；Root误差待评；Full/Body joint FID待评；Hands joint FID不适用；滑步待评；接触F1待评。
- 实际/proc、持续日志和manifest确认InterGen23550、VQ46201活跃，三个队列7178/8321/9975存在。GPU81%/8533MiB，保持运行、idle_since为空；无重启、停机或本机checkpoint下载。
### 2026-10-16 06:09 SGT｜VQ主步40000完整480val

- fresh VQ seed123、fixed_val_first480_center_max300_raw_smpl_v1：重建FID2.924769，较35k变差，best仍35000/2.827235；Matching3.810695，R1/2/3=0.341667/0.516667/0.625000，Div7.754080/GT7.715767。非生成FID或正式test。
- 十项质量：RMSE待评；Full MPJPE待评；原生22joint Body MPJPE108.307099mm；Hands MPJPE不适用；Root误差待评；Full/Body joint FID待评；Hands joint FID不适用；滑步待评；接触F1待评。
- /proc、持续日志及manifest确认InterGen23100、VQ42601活跃，三个队列7178/8321/9975存在。GPU98%/8533MiB，保持运行、idle_since为空；未重启、停机或下载checkpoint。
### 2026-10-16 05:54 SGT｜VQ主步35000完整480val

- fresh VQ seed123、fixed_val_first480_center_max300_raw_smpl_v1：重建FID2.827235，best=35000/2.827235；Matching3.810415，R1/2/3=0.347917/0.502083/0.622917，Div7.709463/GT7.715767。非生成FID、独立test或SOTA结论。
- 十项质量：RMSE待评；Full MPJPE待评；原生22joint Body MPJPE109.042825mm；Hands MPJPE不适用；Root误差待评；Full/Body joint FID待评；Hands joint FID不适用；滑步待评；接触F1待评。
- 实际/proc、日志及manifest核实：InterGen22600、VQ39001，其他队列7178/8321/9975存在。GPU96%/8533MiB，保持运行、idle_since为空；未启动/重启任务、未停机或下载checkpoint。
### 2026-10-16 05:24 SGT｜VQ主步30000完整480val

- fresh VQ seed123、fixed_val_first480_center_max300_raw_smpl_v1：重建FID2.870298，best=30000/2.870298；Matching3.812147，R1/2/3=0.370833/0.500000/0.600000，Div7.735841/GT7.715767。非生成FID、独立test或SOTA结论。
- 十项质量：RMSE待评；Full MPJPE待评；原生22joint Body MPJPE110.311079mm；Hands MPJPE不适用；Root误差待评；Full/Body joint FID待评；Hands joint FID不适用；滑步待评；接触F1待评。
- 实际/proc、日志及manifest核实：InterGen21700、VQ30001继续推进，其他队列进程7178/8321/9549存在。GPU91%/8533MiB；保持运行、idle_since为空，无重启/停机/本机checkpoint下载。
### 2026-10-16 05:09 SGT｜VQ主步25000完整480val

- fresh VQ seed123、fixed_val_first480_center_max300_raw_smpl_v1：重建FID3.143828，best=25000/3.143828；Matching3.811189，R1/2/3=0.358333/0.497917/0.610417，Div7.730003/GT7.715767。FID改善但Body MPJPE较20k回升，不称十项全优；非生成FID或正式test。
- 十项质量：RMSE待评；Full MPJPE待评；原生22joint Body MPJPE115.773666mm；Hands MPJPE不适用；Root误差待评；Full/Body joint FID待评；Hands joint FID不适用；滑步待评；接触F1待评。
- /proc、持续更新日志及manifest核实：InterGen21200、VQ26201；两队列进程仍存在。GPU97%/8533MiB，保持运行、idle_since为空；未重启、停机或下载checkpoint。
### 2026-10-16 04:54 SGT｜VQ主步20000完整480val

- fresh VQ seed123，fixed_val_first480_center_max300_raw_smpl_v1：重建FID3.244720，best=20000/3.244720；Matching3.813437，R1/2/3=0.372917/0.493750/0.614583，Div7.738170/GT7.715767。此为训练内重建验证，非生成FID或正式test。
- 十项质量：RMSE待评；Full MPJPE待评；原生22joint Body MPJPE112.940591mm；Hands MPJPE不适用；Root误差待评；Full/Body joint FID待评；Hands joint FID不适用；滑步待评；接触F1待评。
- 实际/proc、日志与manifest复核：InterGen20750、VQ22601，两trainer及两排队driver均存在；GPU97%/8533MiB，保持运行、闲置计时为空。未重复启动、停机或下载checkpoint。
### 2026-10-16 04:39 SGT｜两项完整480val更新

- InterGen scratch step20000：FID24.314143，best=20000/24.314143；Matching3.981196，R1/2/3=0.125000/0.191667/0.245833，Div7.603141/GT7.741262。固定480val、seed20261006、DDIM50/CFG3.5训练内验证；MM待评，非独立test。实际训练已继续至20300。
- fresh VQ seed123 main15000：重建FID3.458571，best=15000/3.458571；Matching3.812677，R1/2/3=0.333333/0.487500/0.612500，Div7.721899/GT7.715767。协议fixed_val_first480_center_max300_raw_smpl_v1，非生成FID。实际已推进至19401。
- VQ十项：RMSE待评；Full MPJPE待评；原生22joint Body MPJPE117.366906mm；Hands MPJPE不适用；Root误差待评；Full/Body joint FID待评；Hands joint FID不适用；滑步待评；接触F1待评。
- /proc、日志、manifest复核两trainer及队列/其他评测进程存在，GPU99%/10612MiB。保持pod运行，闲置计时为空；无重启、停机或本机checkpoint下载。
### 2026-10-16 04:09 SGT｜VQ主步10000完整480val

- fresh VQ seed123、fixed_val_first480_center_max300_raw_smpl_v1：重建FID4.223619，当前及best均step10000；Matching3.819290，R1/2/3=0.339583/0.472917/0.604167，Div7.749465（GT7.715767）。这是训练内重建验证，不是生成质量或正式test。
- 十项质量：RMSE待评；Full MPJPE待评；原生22joint Body MPJPE125.506712mm；Hands MPJPE不适用；Root误差待评；Full/Body joint FID待评；Hands joint FID不适用；滑步待评；接触F1待评。不能把重建FID当作joint FID。
- 实际进程/日志/manifest复核：VQ5729推进至主步11401；InterGen2353正在step20000验证生成，尚无完整新评分；队列7178及decoder root pilot7836进程存在。GPU100%、10610MiB，保护保持运行，idle_since为空。未停止/重启任何任务，未下载checkpoint。
### 2026-10-16 03:49 SGT｜双训练闲置保护复核

- 实际进程、日志及 manifest 复核：InterGen scratch PID2353 推进至 step19350；fresh HHC VQ driver5690/trainer5729 推进至 update7000 / 主步6001。两项均活跃，GPU利用率100%、显存8531MiB；未重启或停止任何训练。
- 已通过应用更新并读回验收 `hhc-runpod` 每15分钟保护：同时检查两项训练及其他有效评测/部署/传输，任一活跃均不得停机；仅全部闲置30分钟才停机保留卷。SSH失败本身不构成GPU闲置证据。
- 本机控制器53968继续运行；本轮无checkpoint下载、无initial-backup重启、无新增算力。云到云checkpoint备份仍未配置验收，不能宣称备份完成。此次没有新的完整验证结果。
# HHC 简短进度

更新时间：2026-10-05。后续实质进展更新本文件；新记录放在前面，不记录重复的无变化轮询。

## 当前状态

2026-10-20 原185k VQ已从恢复连接的Edmond找回并CPU验收，115068263B/SHA08ba7d0d...2f8d608；107状态张量严格加载通过，normalizer与当前170k逐字节相同。原训练清单10项源码指纹中7项匹配现版，3项差异已从迁移前归档找回匹配版本。历史native重建FID2.4238196未重跑，不是新生成质量结果。当前170k冻结spec及A4000绑定不改，不直接替换码本，不重复旧失联资产搜索。仅云CPU，未下载本机权重/数组。详见 docs/historical_vq185k_recovery_20261020.md。

2026-10-20 Edmond 迁移已验收：官方 InterGen 从 Runpod 完整262240步状态恢复；新262592步 checkpoint（196项optimizer）再次CPU验收通过。主算力为免费 newstation / edmond-Z790-Taichi，实际目录 /mnt/8T-HDD/wenning/hhc_migration_20261020/workspace/HHC，使用同级 edmond_runtime_namespace.py 保留 /workspace 路径。trainer138991/35423489；HHC/eval等待139371/35441518、139469/35450774；watch139818/35465491；两个下游CPU等待139470、139471。所有PID以实际LAUNCH/STATUS核验。Runpod不再作为主算力，禁止自动恢复；控制面停机结果见 output/edmond_migration_20261020/runpod_stop_receipt.json 和 migration/control/compute-owner.json。两automation保持PAUSED，HHC4000与官方704k+真实20rep/MM+独立review门槛不变。后续改动须提交并推送GitHub，同步Edmond旧项目与实际runtime。Runpod停机保留本次版本；若将来获授权启用，须先拉取同一最新提交，不得为例行同步自动启动付费实例。无新FID，无本机权重/数组下载。详见 docs/edmond_migration_20261020.md。

恢复证据包official_intergen_restart_recovery_evidence_20261020.zip已验收：467222B/SHA765c48111f02e6ba1db85dd44cc84914fbd9c72cef2fddd4ee2a8ce9a770d987，85成员，云端整包和逐成员SHA均一致；Drive文件1w7fPiOddWoSmi33ChK1n1m1wvj4nLjEj的id/name/size/parent读回匹配，API未返回checksum，不声称Drive SHA。仅源码/小JSON/失败日志/文档，不是模型异地备份；本条及最终观察晚于ZIP内文档快照。收据output/hhc_sota_20261016/official_recovery_backup_receipt_20261020.json。

**2026-10-20 17:56 SGT：云端容器重启后的完整恢复已验收。** 原进程全部缺失、255552半文件已隔离保留，从完整255200恢复；新增外层Lightning ckpt_path完整状态恢复并补执行原pending epoch scheduler hook，保留原官方源码/磁盘配置和激活包装。冻结YACS CPU对照与实际首批/196项optimizer/计数通过；恢复后的新255552完整checkpoint再验通过（2701620469B/SHA cf983bfe...cab8a2），仅留云。tmpfs原压缩包SHA、每人7777文件/15558 eval链接及官方Dataset11274/352 CPU验证通过；两个未更新失败attempt均保留。当前trainer1282/3987440，HHC/eval1486/3999646、1490/4003594，watch1491/4003597，下游1506/4003941与1507/4003959，全为实测身份。后续使用tools/read_official_intergen_wait_local.py消费既有watch与实际/proc，勿依赖临时functions store。两automation只更新prompt，仍PAUSED/频率目标保持；HHC4000与final704k+真实20rep/MM+独立review门槛保持。无新FID/SOTA证据，无本机权重/数组下载。详见docs/intergen_official_reproduction_20261017.md顶部及output/hhc_sota_20261016/official_recovery_*_20261020.json。

**2026-10-18 22:52 SGT 新padding边界审计已验收：** 官方VEL/FC只检查前一帧mask，纳入有效末帧→第一padding差分。云CPU抽取真实AST，8合成+1实际train3292/80帧共9例exit0；有效预测=GT、仅首padding+1m时VEL=.374999962/padding梯度.749999924，独立双端mask候选均0。合成1cm接触例亦复现；FC受预测门控限制。旧custom已双端有效，此项不解释其FID退化；当前官方/排队绑定源未改、无新质量结果。source/report SHA独立读回通过，session32827 exit0；第一次诊断epsilon容差断言失败保留，不当模型失败。详见docs/intergen_bug_audit_20261017.md顶部和intergen_padding_velocity_cpu_20261018.json。仅云CPU/小JSON源码，本机新增权重数组0，HHC4000及官方优先保持。

**2026-10-17 23:49 SGT｜用户要求继续查bug：新增确认镜像文本方向错误。** 三处真实源码均将counterclockwise错误变为countercounterclockwise；云CPU全5637交互/16911描述中影响60交互/150描述，按镜像抽样规则约0.4435%。六合成例及全训练描述三源表达式对照通过，97960明确exit0；源码上传/结果原始字节SHA读回通过，CUDA未初始化、未读checkpoint。此为官方继承bug，不解释custom/official差距；候选仅在独立审计工具中，绑定源未改。当前官方29项Python与固定提交逐字节相同；22675/25145/26133实际PID+birth正确，两个等待器仍waiting_for_official_training。旧custom暂停/HHC4000/automation PAUSED保持，无新FID或SOTA结果。详见docs/intergen_mirror_text_bug_20261017.md；已确认稀疏/接触分母及时间reduction错误仍以原数值证据为准，不重复同一CPU反例。

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

### 2026-10-17 20:03 SGT｜InterGen100k完整验证与云CPU漂移核对

- 12:00:06 UTC原20042实际/proc活跃、训练100350，CPU ticks1583556；20344活跃、A仍4000纯等待、phase_exit_codes空，GPU仅20042/10356MiB。预定12:00前未重复普通云轮询；原等待exec75105/cell1028最终exit0，已无待收观察句柄。没有重启训练、HHC GPU工作、配置切换或本机权重下载。
- 100k固定480val：FID **35.2390586118**，Matching3.9905038834，R1/2/3=.125000/.216666667/.268750，Div7.891499996/GT7.741262436；best仍20k/FID24.3141428796。相较90k，FID、Matching、三个R及Div距GT均变差。MM仍待评；请求SHA751efc...26b7f93、seed20261006、batch96、evaluator provenance与90k完全一致。原始2642B/SHAdb25f46051e3529a3a705b6644c163758460837ab1d85bdc9f0685ec9b5944be只收小JSON。
- 新独立只读CPU工具tools/audit_intergen_cached_fid_drift_cpu.py，5415B/SHA2c6bdc03f967a7481f444e7d82a479365281e7dbf6ad94cbf8619a1d597b4b29，云完整字节/SHA读回通过，实际SSH exit0。只读20/60/80/90/100k既有NPZ与报告，不导入torch、不加载模型、不做前向、不改数组；五快照GT embeddings逐元素相同，原官方指标源码哈希固定，五次FID复算与报告误差<2e-8。
- FID均值项/协方差项：20k 6.978053/17.336090，60k 8.892387/15.737315，80k 13.199120/18.960648，90k 16.099012/18.897075，100k 15.444944/19.794114。20k→100k总增10.924916，其中均值项增8.466891（77.50%）；明显偏移在84k性能切换前已存在。90k→100k均值项反而下降，协方差项增加使总FID上升。因此不把变化单因归于加速或更换GT/evaluator；这是数值分解，不证明训练原因、根位置偏差或无模式缺失，不据此做特征平移/指标修饰。CPU小结果9052B/SHAf22015f92f86b1dfbb638a43b81196c685084e9a1b7c23f6b49c8248dc5d5d49。
- 继续用户已明确的完整InterGen704k与HHC保存4000等待；真实HHC五候选生成/选模、正式20rep main/MM、四类方法比较及SOTA仍未完成。下一普通质量节点14:30 UTC/22:30 SGT，目标110k；现有小时恢复监控保留，app automations仍PAUSED。上一goal轮是90k结果进展，本轮是100k真实结果、缓存诊断及verified wait，goal active。合并证据output/hhc_sota_20261016/intergen100000_and_cpu_drift_evidence_20261017.json。

100k来源/结果闭合包 `intergen100000_evidence_and_drift_source_20261017.zip` 为377950B/SHA100551ddfdf22660471a774eccc7a4a05d1b14476e41899880a4da761b5c38d6，云整包及11成员完整字节/SHA通过；本收尾段之前的三文档快照、独立CPU工具和小JSON，无权重/NPZ/H5/token数组。[Drive备份](https://drive.google.com/file/d/102fVohFj01qJ9RBCPbOy5n0g8Ki_Vk-X/view?usp=drivesdk) 大小/父目录读回通过、checksum未暴露；回执 `output/hhc_sota_20261016/intergen100000_packet_receipt_20261017.json`。这是来源与结果备份，不是模型权重备份。

### 2026-10-17 17:44 SGT｜InterGen90k验证与持续吞吐验收

- 09:43:03 UTC只读SSH exit0：原fast训练20042真实/proc活跃，已91100；等待器20344活跃、A仍保存4000等待、phase_exit_codes空，GPU唯一compute app为20042/10356MiB。本轮观察器原exec42312、orchestration cell1013最终exit0；08:43/09:03/09:23/09:43均为实际时间观察，不把90k验证期间步数不变视为退出。未重启训练、启动HHC、改绑定源码或新增本机checkpoint。
- 90k固定480训练内val：FID **34.9960871932**，Matching3.9862061183，R1/2/3=.145833333/.227083333/.285416667，Div7.714151859/GT7.741262436；best仍20k/FID24.3141428796。相较80k，FID变差2.83632，但Matching、三个R及Div距GT改善；不宣称全项改善，也不能把这一次变化归因于84k后的microbatch切换。MM仍待评。原始2668B/SHA1f07929f94613822f9221d54f8588a33a81130bd18304c77145c339e0caa70c7已只收小JSON；request、seed20261006、batch96和evaluator provenance与80k逐项相同。
- fast84k→91150的141个常规50更新区间：每步中位.726916秒、均值.752365秒；90k验证及保存额外估计978.95秒，非精确拆分profiling。每万步含一次验证约2.36小时；剩612850步与62次验证按此段估算约145小时/6天，非完成保证。此前“约2.1小时/万步”只指纯训练，不含验证。
- 这是独立InterGen rotation/FK scratch实验，不是公开权重成绩、HHC生成收益或20rep正式test。HHC仍按用户优先级等完整704k；五候选真实text-only val、正式main/MM和SOTA未完成。上一goal轮为verified wait，本轮新增90k完整证据及真实进程等待；目标保持active。合并证据：output/hhc_sota_20261016/intergen90000_and_fast_continuation_evidence_20261017.json。

90k闭合白名单来源包 `intergen90000_evidence_and_docs_20261017.zip`：393572B、SHA3355885195d4d52802636135780482f611bda69ac7d3db57b27155bcd4e8e5fb，云端整包及14成员完整字节/SHA验收。[Drive备份](https://drive.google.com/file/d/1FMCSpo5fR9DutZfp_B4ESx4Qu8j0WkkN/view?usp=drivesdk) 大小/父目录读回一致，checksum未暴露；回执 `output/hhc_sota_20261016/intergen90000_packet_receipt_20261017.json`。包含本收尾段之前的文档快照及小JSON，无权重/动作/token数组，不是模型权重备份。

### 4090 吞吐修复验收（2026-10-17 16:12 SGT）

云端短基准比较后已从完整84000恢复包切换同一实验为micro16×accum2，effective batch仍32。新训练PID20042，权重、优化器和Python/NumPy/Torch/CUDA RNG完整恢复；原2353有意退出，不恢复旧版。训练架构、数据、学习率绝对步数日程、704000目标及原micro4有效帧归一化分组保留。较大microbatch改变dropout执行及浮点顺序，不声称bitwise replay或FID改善。

实际续训前1000步耗时767.70秒，约0.768秒/步，相对原中位约2.43倍；最近三个50步区间0.756/0.783/0.811秒。每万步纯训练约2.1小时；原每万步验证/保存还约16分钟，未来耗时仍随长度、保存和评测变化。最后8秒GPU采样为0–3%发生在85000保存附近，不作为整程利用率，也不能宣称已满载显卡。

84500新恢复包云CPU校验通过：2188977969字节，SHA256 `725a30c104a3475bbd79ec8884ba2194cea2c8a0de721062a361cbcce423d7af`，196个优化器状态、计数74500（历史10000步曾weights-only恢复，84000此次没有重置），所有权重/优化器张量有限、RNG齐备。最初身份断言失败为checkpoint的PosixPath和JSON字符串类型差异，路径相同；修正仅规范化source/output两个路径字段。验收脚本第一次误用model/rng键而报KeyError，按真实trainable_state及*_rng结构重验exit0，不隐瞒失败或冒充通过。

HHC保持A4000保存等待，不占GPU。旧CPU等待17952、20161均为有意取代；新等待器20344使用独立v3源和配置，严格要求InterGen complete704000、最终checkpoint身份与step一致且训练进程退出。v3路径规范化已用CPU正/负例及真实84500恢复包验收；预检release_gate=false，无HHC子进程启动。原29项HHC绑定源码保持。Windows监控104244运行fast恢复器；local-checkpoint-policy明确禁下载。本轮仅源码、日志和小型JSON在本机，无权重下载。

两项automation实际为PAUSED，保持其状态、原频率和目标线程；提示已指向v3并按UTF-8全文读回一致。它们目前不提供主动闲置保护。实际训练仍运行，不能停pod。

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


- **2026-10-16 06:25 SGT 比较路径源码/小结果备份验收：** 33个payload逐项大小/SHA及tar读回通过，`migration/hhc-native-baseline-formal-preflight-20261016.tar.gz`297075B/SHA8e37784f43aab7a1db526afbd126444b4b5cb738ec74f0bb89fd04ab64360125，Drive ID1J3JeNmjQkZWCZGMiPKSl1t-kSUQ-i57z metadata大小/父目录读回一致，云checksum未暴露。含新baseline/score/汇总/消费/CPU和GPU smoke代码、原I/O源码快照、9份实际验收、40k/45k小JSON及当前报告/配置，无checkpoint/motion H5/prior数组；不声称模型权重已云到云备份。当前六训练/等待器保留，所有本轮baseline验收已exit0，正式HHC val选模/test/MM/SOTA仍未完成。

- **2026-10-16 06:23 SGT 新完整VQ45000验收：** 06:22原六PID再次按>=10min实查live；VQ45801/update46800、InterGen23500/best20000 FID24.314143，GPU90%/8533MiB。VQ latest45000 FID2.940790、best仍35000/2.827235，native22含root Body108.947519mm（较40k108.307变差）；Matching3.809457、R .358333/.533333/.616667、Div7.735402/GT7.715767，R3回退。seed123/native fixed_val_first480_center_max300_raw_smpl_v1；非生成/test/SOTA。十项RMSE/Full/Body nonroot/root/Full-Body joint FID/滑步/contact待评、Hands两项N/A。348B JSON SHA84c08671bcc403334ec6ba5beb5a0986f6747e1bae1299f1f0b51c642b6b4a4e对云一致。7178/8321/9975等待不变，无新分支/重启；hourly hhc已app更新新baseline/汇总入口，不重跑已完成验收，不提前test，原schedule/target和禁本机checkpoint下载保留。

- **2026-10-16 06:20 SGT 三基线真实GPU验收已完成：** 原exec45990及Gen/TI/Mask三个child均exit0；验证集19/299/14帧各seed重复两次，共9case均finite、同seed bitwise一致、外层CPU/CUDA RNG保持。Gen/TI原长度，Mask实际16/296/12；GPU峰值1.2001/.8206/.9136GiB，全部在原云4090，无GT motion/shape输入、无test/质量评分、不保存motion/ckpt。CPU_final与GPU_smoke的模型/源catalogue/采样/bridge逐项相同（Mask仅CPU初始化device redirect显式例外，真实GPU未patch），最终exporter SHA0d88373666edf68692ad688a378bf7585d6f19fad343baab0025a49d8699d6ce。9小JSON大小/SHA与云一致，29源保持，finalVQ spec仍无；acceptance=`output/hhc_sota_20261016/native_baseline_formal_acceptance.json`。33协议/16合成汇总消费/真实Mask480评分parity均已exit0，不重跑；真实生成文件删除0。本机无checkpoint新增。当前仍等完整VQ、A/B/C及五项text-only val选模，再正式比较，SOTA未达成。

- **2026-10-16 06:15 SGT VQ40k与基线正式比较路径进展：** VQ当前43601/update44600（06:11正常低频实查），最新完整40000重建FID2.924769，best仍35000/2.827235；Body22含root108.307099mm改善，但Matching3.810695、R1 .341667较35k变差，R2/3 .516667/.625000，Div7.754080/GT7.715767。seed123/native fixed_val_first480_center_max300_raw_smpl_v1；非生成/正式test/SOTA。十项RMSE、Full、Body non-root、root、Full/Body joint FID、滑步、contact待统一评，Hands两项N/A。331B小JSON SHA35e87b53a4650b0e13052c63ed6d4f747334b7b2e0bee6f8df80676fb2f1473e与云一致。InterGen23150/best20000 FID24.314143；六PID2353/5690/5729/7178/8321/9975均live，GPU89%/8533MiB；队列等待未变。

  新独立native-baseline H5/export/CPU评分保留InterMask floor(L/4)*4与Gen/TI原L，不补帧改变基线。三公开模型新入口已云CPU加载验收，最终source版串行inspection exec83084整体及三child exit0。InterMask VQ原构造器硬编码.cuda，首CPU-hidden尝试96723 exit1；仅CPU检查重定向此分配后通过，正式GPU不patch，官方源未改。其云树无.git，按已验收transformer/options+VQ四源SHA固定并记录全Python catalogue，不伪称git revision。

  native协议exec54194 exit0、33项（所有40份请求hash/count，三模型main/MM合成H5、native lengths/RNG/padding/错误拒绝）通过，后续loader修改未动I/O/RNG函数AST。已有真实InterMask480缓存经新scorer回放exec80877 exit0：GT/text embedding逐项相同、gen最大差2.98e-7、FID delta -2.5984e-7，其余主指标一致；只有float32 raw-normalized往返，非新生成/模型收益。新四方法20rep汇总/paired CI和安全正式H5消费工具已16项云CPU合成集成验收exec4788 exit0，160份fixture已删除，真实生成文件删除0；HHC MM无embedding archive则保持H5。当前仍无真实formal选模receipt/test比较，CoSh不得用InterGen改名代替。

  **当前验收句柄exec45990**：在现有云GPU逐个对Gen/TI/Mask验证集first/min/max长度作真实sampler smoke（无test/质量评分、不保存motion/ckpt），每seed重复并查外层RNG。尚待终态，不重复启动。原训练/29源保持，本机无新增checkpoint。

- **2026-10-16 05:50 SGT 正式入口/v2修复增量备份验收：** 17个payload逐项大小/SHA及tar读回通过，`migration/hhc-formal-text-only-v2-20261016.tar.gz`263941B/SHA984c395b6965167c45ef720846e0ca8598c3b963f42e0d42c8a73f7acb797a89；Drive ID1SJO69YpKYFNzyud0Px95ulzfp6upwZLZ metadata大小/父目录读回一致，云checksum未暴露。含新formal源码/CPU审计、v2/handoff源码与证据、配置、35k小JSON及当前报告，无checkpoint/motion H5/prior NPZ，不声称模型权重已备份。9975等待与原长训练保留，未重复launch；后续真实val选模/正式生成/SOTA仍待完成。

- **2026-10-16 05:48 SGT 新完整VQ35000验收：** 最新完整及best更新为35000/重建FID2.827235（旧30000/2.870298）；native22含root Body109.042825mm（旧110.311079）。Matching3.810415，R1/2/3=.347917/.502083/.622917，Div7.709463/GT7.715767；R1较30k .370833下降，不称全指标获益。seed123、fixed_val_first480_center_max300_raw_smpl_v1；非官方冻结480生成或SOTA。RMSE/Full/Body non-root/root/Full-Body joint FID/滑步/contact待统一评；Hands两项N/A。小JSON SHA0769af45c850c07d0a318b656b0e1e06afe0a771cde885d6e123397ba8ec7f74已与云一致，manifest实际37001/update38000、running/best35000。此次为完整35k验收节点，不重复启动。云29绑定源SHA均未变；3份formal/v2小JSON大小/SHA已与云逐项一致，无权重/H5下载。

- **2026-10-16 05:46 SGT 元数据集成缺陷修复、v2等待器与正式入口CPU验收：** 发现原9549按顶层读取shape_policy/MGT/VQ SHA，但两个MGT val exporter将这些存入H5 attrs.metadata JSON，原成功路径会KeyError。此前9项mock覆盖失败门限，遗漏真实嵌套布局的完整成功链；已用独立v2修复，原28绑定源未改。新14项云CPU mock exit0，新增完整4phase成功、nested/flat解析、冲突拒绝、checkpoint不符停后续、不重放。05:44:47守卫脚本再次确认9549/proc命令、waiting、phase空/无child/无输出，SIGTERM后/proc消失；只替换此等待器，原2353/5690/5729/7178/8321命令逐项一致。新driver **9975** / `output/hhc_train_shape_val_v2_control_20261016` live/waiting，配置`migration/control/hhc-train-shape-validation-v2.json` SHA a7494e3ef9d4135b8c78c23064470df79906125683b57804560c072d3d8b3857、29源（原28+v2）；旧9549标记superseded，不可重启。真实旧exit_code未获wait状态，未伪称143。两个train-prior候选仍只排队，无GPU生成成绩。

  新独立`tools/hhc_text_only_formal_contract.py` / `export_hhc_text_only_formal_cloud.py`要求五个预登记text-only候选真实final40k固定480val均完成后，按min-val-FID冻结模型/体型策略/采样/源码；核实实际generation与checkpoint SHA、相同VQ、官方evaluator和GT/text parity。现无真实选模receipt，未启动test/GPU正式导出。MM预先固定每prompt同一train-only体型，30次只变motion seed。云CPU audit exec1613 exit0、18项通过：真实请求rep0主1056及MM100×30写合成H5并经既有正式validator，顺序/seed/shape/padding与拒绝重放；GT shape/test分数/缺候选/改源码或receipt拒绝。合成fixture已删除，不是实际MGT生成/质量证明，后续真实GPU端到端仍待验收。小JSON仅1816/1042/1615B已收取，无权重/H5下载。

  05:42低频只读检查：原五PID及当时9549均live，VQ34801/update35800，最新完整/best仍30000/FID2.870298；InterGen22250，最新完整/best20000/FID24.314143，GPU29%/8533MiB。未出现新完整35k。hourly hhc已app更新并TOML读回ACTIVE、原schedule/target保持，包含9975/v2/29源/旧9549禁止重启/正式门限；原闲置保护不变。

- **2026-10-16 05:32 SGT 30k后训练推进及小结果备份：** 原2353/5690/5729/7178/8321再次按>=10min普通间隔实查/proc与日志均live；VQ32801/update33800，最新完整/best30000 FID2.870298，InterGen21950/最新完整best20000 FID24.314143；GPU100%/8533MiB。没有重复训练或新分支。9549已05:29单独确认live/waiting，本次未重复查。新30k/storage/formal-metadata JSON2364B/SHA d79653d7ad48695d1aac10c05cac45325cf47373739b80a01b888f8716f940bf已Drive ID14aJkkl8W8HzTLVSDDk2bpXa-co1P8dqA上传并metadata大小/父目录读回；不含权重、云checksum未暴露，收据`output/hhc_sota_20261016/vq30000_storage_protocol_backup_receipt.json`。原28源保持、不下本机ckpt；新HHC生成test/MM/SOTA仍未完成。

- **2026-10-16 05:30 SGT VQ30000完整验收与正式协议边界：** 原trainer5729在观察开始已/proc确认live，等待`validation_030000.json`的原exec10152终态exit0；480条训练内val完成，最新及best30000/FID2.870298（旧25000/3.143828），Body22含root110.311079mm（旧115.773666），Matching3.812147、R .370833/.500000/.600000、Div7.735841/GT7.715767。R3较25k下降、Matching略变差，不称全指标提升。seed123/native fixed_val_first480_center_max300_raw_smpl_v1；完整十项缺项/Hands N/A见主报告。小JSON SHA cf852333ef1696be1eecc606ea74e3d3dcfc10e182cd10bae2b17756f93f0939已与云观察结果一致；没有新checkpoint下载或GPU分支。

  正常05:21观察VQ29801、InterGen21600；2353/5690/5729/7178/8321都live，原InterGen最新完整/best仍20000/FID24.314143。05:29单独复核9549实际/proc live且waiting_for_zero_shape_scoring、无child phase；28绑定源未改。Runpod live读回persistent `/workspace` size80GB/RUNNING，`du -sx`本卷22831495680B、HHC约19GiB；不把共享`df`的487TB视为本pod配额，没有扩容/清理。

  正式请求metadata确认main=20×1056（每模型21120），MM=20×100×30（每模型60000生成），MM绑定main manifest hash一致；不是main每rep2000。只核对协议JSON，没有计算test分数/用test选模。**现有训练CLI --formal-test仍使用requested GT shape，不能当pure text-only；zero/prior独立exporter目前仅val，正式text-only/MM入口及latent shape重复协议须在真实val选模后独立冻结实现**。不得编辑28活跃源码来提前绕过此门限。hourly hhc prompt已收束为当前唯一任务/三个队列及正式范围限制，app更新和TOML读回ACTIVE，原schedule/target及禁止下载保留。证据`output/hhc_sota_20261016/vq30000_storage_and_formal_readiness.json`。

- **2026-10-16 05:19 SGT 体型候选源码/结果备份与保护更新：** 41个payload逐一大小/SHA及归档读回通过，`migration/hhc-train-shape-prior-source-20261016.tar.gz`311007B/SHAad28c32889448f3100ae75edd26da1ef50dd0a9405c62e3556f5e7f82a3c8a54；Drive ID1V9xyUYzwmuMVbErFk3TxK9McKrp89EJ1 metadata大小/父目录读回一致，云checksum未暴露。含28绑定源码、额外queue mock、配置/25k和prior小JSON/报告；不含模型checkpoint、motion H5或train prior NPZ。hourly hhc已通过app更新并TOML读回ACTIVE，9549/28源绑定/源与权重限制及原schedule/target保留。原长训及三个等待器继续，不把CPU/排队完成当生成或SOTA。

- **2026-10-16 05:17 SGT train-only体型推理对照已排队：** 云CPU统计builder exec84936 exit0，严格复用MGT cache的train/30<=L<1000/文本非空门限，4567行配对betas、1556种不同pair、20维均变化；未读取val/test betas或motion数组。零beta到训练均值的逐维std缩放欧氏距离9.04293，训练median3.87615/p95=6.57825，零点距离位于98.423%分位；这是条件分布差异线索，非FID收益或严格OOD判定。145147B prior NPZ只在云（SHA4c5d547d9602abc3fd5b361c4689a3d5ba10207a1022994515e9037b802cfd64），本机只收spec/小结果。

  新独立exporter不改原23源，TrainPriorScorer构造阶段禁H5仍通过真实CPU FK，继承原generate不变；480请求只取identifier/sample_seed，mean与empirical两策略不改全局RNG、顺序/分块不变，完整生成loop AST一致。CPU策略审计exit0，9项调度mock exit0（上游live等待/死亡/失败/配置不符、完整12phase、显存/多GPU、child exit9停后续、不重放）。新云driver9549已/proc核实live、STATUS=`waiting_for_zero_shape_scoring`；本机配置`migration/control/hhc-train-shape-validation.json`、云`migration/hhc-train-shape-validation.json`，control`output/hhc_train_shape_val_control_20261016`，28源SHA绑定，**不要修改绑定文件/重复launch**。它只等原8321完成后对A_historical final40000作train_mean/empirical_train_pair两次480val，共用原zero A control、相同checkpoint/VQ/采样seed/原版CPU evaluator，GT/text逐项一致后才比较；无GT shape、formal test=false，尚无实际MGT生成或质量结果。原VQ/2353/7178/8321保留，不新增pod、不下载权重。验收`output/hhc_sota_20261016/train_shape_prior_acceptance.json`。

- **05:10 SGT 新完整VQ25000验收：** 实际26801，最新完整及best为25000/FID3.143828（旧20000/3.244720）；Matching3.811189，R .358333/.497917/.610417，Div7.730003/GT7.715767，原生22joint含root Body115.773666mm较20k112.940591变差，非全指标收益。协议fixed_val_first480_center_max300_raw_smpl_v1、seed123、重建非生成。RMSE/Full MPJPE/Body non-root/root/Full-Body joint FID/滑步/接触待统一评；Hands两项不适用。原InterGen21250，最新完整/best20000 FID24.314143。2353/5690/5729/7178/8321均live，GPU91%/8533MiB；普通监控下次>=05:20。

- **2026-10-16 05:05 SGT CoSh发布审计归档：** 28个源码/小结果payload已逐项大小/SHA及归档读回通过，`migration/hhc-coshmdm-release-audit-20261016.tar.gz`281959B/SHA986a301a70defadfbe4b54454c13ee9b96fffd597551166796ca196bc5d41b9e。Drive ID18eazyoYGb6bKLhKVRXU_Z4DhjYG6XohU已上传并metadata读回大小/父目录一致，云checksum未暴露。包内无checkpoint/H5；云端只读取已有checkpoint、零本机权重下载。主训练及7178/8321未改，普通监控下一次>=05:09。

- **2026-10-16 05:03 SGT CoShMDM独立比较缺口进一步核实：** 官方main仍c5116ab3，`git ls-remote --heads --tags`仅main、无tag；官方Releases/Issues/项目页未发现备用独立checkpoint。本机稀疏clone只取源码，无权重；原scope auditor增加可选repo/output/InterHuman范围及InterGen AST比较，CPU审计exit0，11个InterGen参照源码git blob已与官方固定b1750df API逐项核对。CoSh的9个models文件仅显式标识符重命名后AST全相同，配置仅NAME/CHECKPOINT不同，下载ID同InterGen。云CPU key审计exec47681 exit0，原已在云的2701617525B/SHAbab341...86f8d checkpoint346 tensors中，bert命名所需170键全缺（其中可训练两层text encoder+LN26键/11,029,504元素）；未运行完整模型load或推理。公开infer `strict=False`可能跳过这些可训练层，eval `strict=True`不兼容此命名；不可忽略缺键再伪造第四行独立CoSh复现，也不可复制InterGen成绩。论文4.801继续单列、独立统一评测待作者匹配架构/权重；不影响HHC主训练。JSON `coshmdm_public_release_audit_verified.json`/`coshmdm_checkpoint_names.json`/`coshmdm_release_acceptance.json`位于`output/hhc_sota_20261016`，23绑定源未改、零GPU审计/本机ckpt下载。

- **04:59 SGT 一次低频存活复核：** 2353/5690/5729/7178/8321实际进程均live；VQ24801，最新完整及best仍20000/重建FID3.244720（Body22inclroot112.941mm）；InterGen20950，最新完整及best仍20000/FID24.314143。7178 waiting_for_complete_vq、8321 waiting_for_all_mgt；GPU99%/8533MiB。未有新完整质量结果，不重启。后续正常轮询>=10分钟。

- **2026-10-16 04:53 SGT 小型证据归档：** `migration/hhc-canonical-pilot-20261016.tar.gz`含136个源码/小结果payload，529192B，SHA4ea275985b668912bc181cdd540ad4c952f9899d889a36676181c03dc40a13b6；本机逐payload大小/SHA及归档读回均通过。Drive ID1dHr2-xUKMqvTvI6E_gxx2y0d2fk4Oxzx已上传，直接folder listing及metadata读回大小529192B/父目录一致；工具未提供云checksum，不冒称云SHA验收。包内无checkpoint/H5。这只是源码/结果备份，云到云checkpoint备份仍未配置验收。

- **2026-10-16 04:50 SGT canonical-pair真实pilot完整验收：** trainer exec20986、480导出38366、共享CPU评分28897、geometry85711、paired bootstrap76336均exit0。base VQ5000冻结encoder/codebook/condition，仅decoder+1000更新、seed20261016/batch256/window64；所有训练批次与先前control完全相同。候选Full MPJPE306.399 vs358.891mm、root285.135 vs338.587mm，改善分别52.492/53.452mm；但FID4.312691 vs4.201122、滑步14.916 vs13.771mm/frame、geometry contact F1 .564069 vs.600663变差，**不接入主VQ/MGT，不宣称总体改进或SOTA**。配对480 identifier/10000 bootstrap位置改善CI不跨0，接触/滑步恶化CI亦不跨0；仅为固定早期重建模型val不确定性。GT/text embeddings与三基线逐项相同，23绑定源未改；8个小JSON的本机/云SHA一致，零新增checkpoint/H5下载。十项/原始JSON/决策见主报告及`output/hhc_sota_20261016/decoder_canonical_pilot/acceptance.json`。停止对这版早期损失的扩跑，重点等待完整VQ后的CFG/window生成对照。

- **2026-10-16 04:49 SGT 正常低频复核：** 2353/5690/5729/7178/8321均活跃，单4090利用率99%、8533/24564MiB。fresh native VQ已20601，最新完整480训练内val及best为step20000/FID3.244720（旧15000/3.458571）、22joint含root Body监控112.941mm，协议`fixed_val_first480_center_max300_raw_smpl_v1`；不可与生成FID或官方frozen480重建混排。原InterGen rotation scratch20600，最新完整/best仍20000/FID24.314143。A/B/C队列7178及text-only后评测8321继续等待，不重启、不改23源、不增pod，下一次普通状态检查至少10分钟后。

- **2026-10-16 04:40 SGT canonical关节损失校验/真实短训：** 新`utils/hhc_canonical_pair.py`严格按官方实际代码只取hips[2,1]定朝向，按人独立floor、锚点XZ和共同锚点旋转、保留T-1位置。云端audit55074 exit0：480条GT及预测、两anchor/两swap共3840组合，最大坐标差GT .01451mm、pred .02199mm；CPU方向导数误差5.98e-12，公共yaw/translation不变性1.78e-15m、swap逐项相同、退化hip拒绝。新canonical-only decoder pilot exec20986/PID8835已实际运行，base5000、1000更新、seed20261016/batch256/window64、只额外3×canonical SmoothL1，复用上轮同预算decoder_control；前4/100更新有限、冻结检查通过、GPU peak1.18GiB。optimizer/sampler/原loss components/root diagnostic/objective/params AST一致，23个既有锁定源未改。未有新候选质量结果，不改主VQ/ABC/text-only队列。

- **2026-10-16 04:40 SGT 既有旋转/FK新完整val已核实：** 原InterGen268 scratch分支20000完整480val FID24.314143、Matching3.981196、R1/2/3 .125/.191667/.245833、Div7.603141/GT7.741262、MM待评；best更新为20000（旧10000/44.620449）。真实进程2353已继续20250。采样DDIM50/CFG3.5，生成不输入GT shape/root；这是旧独立实验，不冒充HHC改进或SOTA。小JSON已收取，无ckpt。主VQ检查到19001，最近完整及best仍15000/重建FID3.458571；两级等待器均保留。

- **15k验证小结果补充备份验收：** 独立JSON322B/SHAf391915e98023531772db8db9a53fc954777ffd3a7aef611fad2d1e7c1f81168已传Drive ID1WDewcVz57rJp8OZ0Omvghz9NC0X44mYZ，metadata大小322读回一致；不是checkpoint备份，也不包含在先前517439B包中。

- **2026-10-16 04:27 SGT 主VQ新完整验证/等待器实查：** /proc确认2353/5690/5729/7178/8321原命令均live。VQ实际update16600/main15601，完整main15000固定480 center训练内验证：重建FID3.458571、Body117.367mm、Matching3.812677、R1/2/3 .333333/.487500/.612500、Div7.721899/GT7.715767；最新与best均15000（此前10000/4.223619）。检索非全项提升，不与生成/正式test比较。原InterGen仍20000流程、best10000/FID44.620449，未有新完成480val证据，不重启。7178等待完整VQ；8321 waiting_for_all_mgt，未开始GPU后评测。新15k小JSON单独收取，发生在517439B归档完成之后，未冒称已在该包内；本地仍无ckpt新增。

- **2026-10-16 04:27 SGT 新原生VQ评测/负结果/公平评测队列已云备份：** `migration/hhc-native-vq-root-pilot-20261016.tar.gz`517439B/SHA130d056c6e42f65af748162ee85bfce21870639e9d691aebebc0b542495bc19f，119项内容逐一字节数/SHA复核，Drive ID1B7vkE0LWl9RAzKpu5lSxYc3LCPcMih-M，metadata大小读回一致、云checksum未暴露。含新源码、原exporter source快照、三份480重建/geometry、decoder等预算负结果/配对CI和text-only队列配置；无权重/H5。hhc小时automation已更新并读回ACTIVE、包含8321队列；主训练/两级等待器继续。本机仍零新增ckpt，不冒称最新模型权重已经云到云备份。

- **2026-10-16 04:22 SGT decoder根朝向pilot完成，未采用；文本-only评测已排队：** exec36269两臂各1000训练exit0，9.33M decoder参数，encoder/codebooks/condition不变，两臂完整batch SHA16f0a33ffe6f8bff57de0469c88932373653ef28cbcd334f5310d54d067f823d相同；control/root模型37.34MB均只在云。两臂完整480导出34183/21463、统一评分31949/15419、geometry58594/93975及配对CPU70688均exit0。对照/root项：FID4.201122/4.319167，Full MPJPE358.891/365.921mm、root338.587/346.465mm、sliding13.771/14.122mm/frame、contact proxy F1 .600663/.591458；root额外项无收益，不以RMSE单项改善宣称成功，不进入正式MGT。480个identifier配对bootstrap10000次，Full delta+7.030mm CI[-15.440,28.933]，滑步delta+.35085 CI[.23013,.47340]；仅固定模型val抽样不确定性。新PID8321已实查live/waiting_for_all_mgt，23项源码锁定、6项CPU mock安全检查exit0，等原A/B/C三组全完成后顺序各导出final40k的zero/requested shape，再用统一CPU评分及GT/text逐项相同检查；无test、无新增GPU资源、本地无ckpt。配置`migration/control/hhc-text-only-validation.json`。VQ最新已验main13401，最新/最佳10k重建FID4.223619、Body125.507mm；旧InterGen20000正在后续流程，best仍10000/FID44.620449，无新完整val。

- **2026-10-16 04:12 SGT 官方480val新VQ独立评分/归因完成：** 同三基线请求的native VQ5000导出26675、几何16687、旧bulk评分31671、新逐请求统一评分48640均exit0。重建FID4.575838/Matching3.812979/R .38125/.54375/.6125，统一seed20261006的Div7.663406/GT7.741262；GT/text embedding与InterGen基线逐项相同maxdiff0。Full/Body(non-root)MPJPE372.110/373.027mm，坐标RMSE352.627mm，root352.851mm，滑步13.696mm/frame(GT4.258)，几何contact F1 .624510；Hand两项不适用，Full/Body joint FID及语义接触F1待评。重建指标不与生成FID比较。恒等转换max0.0047mm/接触0差异。全480同请求native世界MPJPE137.136，GT-oracle仅换全局朝向可降官方误差至155.550，但不是推理方法。GT-free边界重复8/16token在固定32条上均变差，已淘汰不扩跑。真实decoder-only等预算root损失两臂1000更新pilot已开exec36269，control首4更新/冻结编码器检查通过、峰值1.18GiB；待两臂独立val，不称提升。详细十项/协议见主报告。

- **2026-10-16 03:51 SGT 新VQ/队列源码与小结果已云备份：** `migration/hhc-vq-rebuild-source-20261016.tar.gz`375895B/SHA a615bd0129ed6ae55966070ef0341410113768dfdc719b86f693adff55094adc，76项归档内容逐项字节数/SHA复核通过；Drive ID1YNkMEO5N5EIrqpGit-XKHLPHQmZD9h5F，metadata大小读回一致，连接器未暴露云checksum。包含fresh VQ训练/恢复代码、首5000验证、真实MGT smoke、A/B/C排队配置及CPU安全审计，无checkpoint。训练与排队继续；不能说最新训练权重已云到云备份或SOTA已完成。

- **2026-10-16 03:50 SGT MGT真实链路验收完成，A/B/C排队：** exec74877确切exit0，完整285709616参数MGT使用云端VQ5000快照完成2次真实更新（loss9.74741→6.05805，梯度有限）；14/299/115帧请求输出3×300×524，官方evaluator嵌入3×512有限，冻结VQ所有张量未变，GPU peak5.4628GiB，不存MGT checkpoint。仅功能验收，不是生成质量收益。新driver7178已验明等待完整VQ300k，配置`migration/control/hhc-abc-after-vq.json`，20项源码SHA绑定，之后冻结最终val-best顺序运行A历史/历史窗口、B匹配文本dropout/历史窗口、C历史文本/uniform窗口，各40k/同seed/有效batch64/micro4、仅val选模；不使用5000中间码本做最终消融。云队列6项CPU mock安全检查通过exit0（初次仅写结果目录缺失，修复后重跑），未触发GPU或模型加载。三组尚未训练、正式test未开。小时automation已同步队列及禁本地ckpt。最近VQ实际6601/best5000重建FID4.883522；原InterGen19400，best仍10000/FID44.620449，无新完整val。

- **2026-10-16 03:45 SGT 新VQ首轮质量验证完成/MGT真实smoke启动：** main5000、seed123，480条固定center/max300训练内重建验证：FID4.883522、Matching3.818581、R1/2/3 .352083/.487500/.591667、Div7.744356/GT7.715767、原生22joint Body MPJPE136.739mm；当前及最佳均step5000。此FID是重建，不与三生成基线8.7比较，不称SOTA。完整十项：RMSE待评；Full MPJPE待评；Body MPJPE136.739mm（原生22joint含root，非独立统一geometry结果）；Hands MPJPE不适用；Root误差待评；Full/Body joint FID待评；Hands joint FID不适用；滑步/接触F1待评。云端freeze pilot exit0，115069810B/SHA554e6e3adf105bae838d9917271e5919ace0fa595b75be3c881ea60c608c76e5，非最终选模。确认余量15553MiB后，以GPU50%上限启动真实全MGT两更新+长度边界生成/evaluator smoke（exec74877），VQ与旧2353均保留，等待原句柄终态。只收JSON，未收任何权重。

- **2026-10-16 03:44 SGT HHC恢复包与MGT准备验收：** 云端CPU审计真实update1999/main1000恢复包342031139B/SHA f7cb110330d681923b5de9e3cd7b816e42af833aa89c987bc2e3e67921396e1d：107模型张量、105 optimizer状态与步数、非buffer EMA均有限，保存的batch排列可重复生成下一批，exit0；权重未下载。首个main5000验证已进入官方CPU evaluator阶段，尚无完成指标，不因验证期间训练日志暂停而重启。新MGT入口和真实全模型smoke的云端CPU导入exit0；CPU fp32 CLIP缓存改为单前向，真实batch1/8/37/128的word/global输出与旧实现torch.equal，maxdiff0，已接入独立新入口，不改运行中的VQ合同。小时automation已覆盖新VQ及禁本机ckpt下载。MGT smoke/A/B/C仍未启动。

- **2026-10-16 03:25 SGT 真实HHC VQ重建已运行（新的计算决策覆盖此前排队）：** 旧server再次SSH超时；本地权重清单/Drive根目录均无旧185k，W&B匿名元数据project=null不能恢复。根据真实smoke峰值1.79GiB，新VQ可共卡运行而不暂停旧任务，故采用现有单4090低优先级/显存20%上限重建，无新增pod、无本地新ckpt；未把用户未答问题当作暂停旧训的许可。原InterGen2353仍真实18650、best10000/FID44.620449，无新完整val。新run`output/vq_hhc_native_rebuild_seed123_20261016`，driver5690/trainer5729，前999warmup已完成，当前update1000/main step1、loss.289660有限；NVIDIA进程占用约2.7GiB（allocated峰值1.80GiB）。这是fresh rebuild，非旧185k恢复/性能提升；原模型/损失/300k cosine/验证函数保留，独立可恢复batch排列不同历史精确顺序，W&B disabled且JSON日志权威。首个质量验证main5000，十项待评/手部不可比。

- **2026-10-16 03:25 SGT 恢复验收/入口接通：** 初次因plotly依赖缺失在导入阶段exit1，无权重更新，失败记录保留；补齐隔离`/opt/hhc-vq-deps`并完整import preflight exit0后重试。initial4/resume6均exit0，4→5/6恢复验证模型/EMA逐项相同、105个optimizer状态、batch cursor1024恢复；随后从6继续长训。采样器CPU跨epoch/中断恢复审计exit0；验证函数、loss/optimizer AST与原入口相同。控制和准确恢复命令见`migration/control/hhc-vq-rebuild.json`。新MGT入口新增显式不可变VQ spec（step/hash/origin），保留历史185k默认路径，新分支不伪称旧模型；默认micro4累积有效64、CPU评测、GPU50%上限，尚未训练，后续启动需核对主训练验证的显存余量。

- **2026-10-16 03:09 SGT 三基线/就绪验收最终小包云备份：** hhc-sota-validation-complete-20261016.tar.gz 341706B/SHA9e7526509a79e1e91d153afd6424f9a0334681ab9a9549685be67953412fe7c4，Drive ID11NsR9-diJbM0QOPFpIcEAtccXEoyot55，metadata大小读回一致，云checksum未暴露。包含三基线完整480val、comparability、真实VQ smoke、源码和协议，无checkpoint；不是HHC训练完成/正式test/MM/SOTA。远端三基线及smoke进程均已退出；关闭本机两条已完成作业的悬挂SSH连接，不影响2353长训。HHC旧VQ及算力优先级待处理，goal保持未完成。

- **2026-10-16 03:06 SGT 三基线480val全完成：** TIMotion4780 generation/score均exit0：FID8.721812、Matching3.780101、R1/2/3 .460417/.639583/.722917、Div8.080552（GT7.741262）。InterGen8.762189/InterMask8.931384为同次协议结果；三者request/evaluator相同、GT/text embeddings maxdiff0，cache/report/embedding SHA关联审计通过。InterMask平均长度少1.6521帧，原生sampler预算不同，单次480val无正式20rep/MM，不声称排名显著或SOTA。小JSON均已收本机，无新ckpt。三基线和VQ smoke均已终止，无待排队job；既有InterGen长训2353仍保留。HHC对照A/B/C仍未启动，缺原VQ185k，重训/暂停长训优先级问题待用户答复。结果与说明见主报告；最终小结果归档待本轮补包。

- **2026-10-16 02:56 SGT 第二基线+真实VQ smoke完成：** InterGen runner4126的480val生成/CPU评分均exit0：FID8.762189、Matching3.794905、R1/2/3 .425000/.591667/.672917、Div7.934211（GT7.741262）。报告小JSON已收本机，与InterMask同请求/evaluator但原生生成长度不同；非论文test20rep，不排SOTA。HHC VQ4282完成batch256/window64/fp32四次真实GPU更新，loss/梯度有限、peak allocated1.7913GiB、26.79M参数、不保存ckpt。仅训练就绪，重建十项待评/手部不适用，不能称质量提升。TIMotion4780/gen5047已开始真实480val；原InterGen长训2353保留，旧HHC185k缺失及用户算力优先级仍待答。新实测不在此前320617B归档内，后续追加小结果包。

- **2026-10-16 02:53 SGT 代码/小结果云归档及资产预检：** hhc-sota-cloud-validation-20261016.tar.gz 320617B/SHA ca1ddb262c2314d46193a02fe4bd107bfa36e9437768e014eb82951c0574c247，Drive ID1tZKU-O2bCV64rhPQ8jzs-T3uN_l8qDDH，metadata大小读回一致，云checksum未暴露。包含源码/协议/CPU审计/InterMask480val结果，无checkpoint。云端资产预检只缺原VQ185k；CLIP存在/root/.cache/clip（已补预检默认缓存路径），不等于MGT运行环境全部验收。InterGen480val生成exit0、CPU评分4926运行，VQ4282/TIMotion4780仍排队，等待确切终态；原长训未停。

- **2026-10-16 02:48 SGT 首项统一val及第二HHC候选：** InterMask480val生成exit0，首次CPU评分在encoding后遇provenance Path/str错误exit1；保留失败文件、修复后v2独立重算exit0：FID8.931384、Matching3.795384、R1/2/3 .443750/.612500/.685417、Div8.127659（GT7.741262），MM待评。与论文test/HHC历史20rep口径不同，不能排SOTA。小JSON已收本机，无ckpt。TIMotion1430359445B直接云获取完成，SHA f364605f7e67cc272dc31810f384ff0dedb564331442257bf776f8d7285ffcee（作者无公开checksum）；Transformer+LPA CPU严格加载294张量无missing/unexpected，runner4780持同锁排队。InterGen4126/4322运行，VQsmoke4282等待锁，不重复启动。

- **2026-10-16 02:48 SGT token长度审计/改进代码：** 云CPU查6021个有效train group：<15帧384、15–29帧930、30–999帧4567、>=1000帧140；原缓存按长度保留75.85%，其中210个crop>75token。原crop没有300帧上限，最终TokenData固定前75。新增uniform window，1024种子对真实TokenData检查exit0：historical逐项一致、短序列/文本/长度/shape/EOS保留、长例起点0..105。与CFG分开A/B/C对照，未训练/未测收益，source长度门限不改。已核DHVAE5.015、Text2Interact5.191；HINT online3.100但首窗口输入实现待核验，4.702不能当全球SOTA充分线。主报告已更新。

- **2026-10-16 02:36 SGT 云端原生数据/基线恢复：** native SMPL全量重建3521/3522已exit0：7779文件、7777有效pair、62232原始数组bitwise还原；派生特征max joint .000901mm，contact差8/8980112，不冒称旧派生数据bitwise。manifest SHA7468c59cbe010e438227ec1a13e30d1fd9a03a8fc1e8c9d0d4bfb971e02c5dc5。valMM及20rep testMM重建完成；内容审计1908/2000共享prompt字段一致、20个main请求hash未变，旧MM文件原字节一致性未证。InterMask两权重与InterGen官方权重直接落云并SHA通过，无本机新增ckpt。InterCLIP加载兼容PyTorch2.8已修复（先验证既定SHA，再显式weights_only=False）；207张量完全相同、有限embedding、篡改拒绝及6项已有测试通过，不改评测数学。

- **2026-10-16 02:36 SGT 验证活跃句柄/缺口：** InterMask runner3916已完成480val生成exit0（child3919），CPU评分4133运行中；InterGen baseline runner4126持同一GPU锁排队，不重复启动。HHC VQ smoke4037在导入阶段缺未使用的SMPL-X资源而退出；改为仅在原Inter-X构造时懒加载资源，保留损失/网络方程，待重试。现有scratch trainer2353继续运行；用户尚未回答优先暂停InterGen还是保持长训练，不能自行暂停。仍缺旧HHC185k VQ，未有新HHC训练提升/SOTA证据。CoShMDM公开下载脚本实际指向InterGen同一个checkpoint，不能报告成独立CoShMDM复现；TIMotion公开版须标Transformer+LPA，不冒称RWKV。

- **2026-10-16 02:15 SGT 本轮研究小包已云归档：** hhc-sota-research-cpu-20261016.tar.gz 572057B/SHA eb7ba387adee63a6e094fb0d370605f9d5089d8b16709b21f3f840c492028544，Drive ID1OXdw-YVGqnBwkfpmk8VASE8BZQyxJFWA；metadata读回大小一致，连接器未暴露云checksum，不称云端SHA复核。含代码、政策、CPU探针和比较报告，无checkpoint。新controller53968已成功读到云训练16100及云恢复点16000，未下载新恢复包；最新完整val仍10000/FID44.620449，无新生成质量结论。

- **2026-10-16 02:10 SGT 新goal研究与用户算力/备份限制：** 用户要求所有GPU工作仅云端、本地不下载超过一个ckpt。已写agents及local-checkpoint-policy.json，禁止所有新增checkpoint下载；旧initial-backup30948确认归属后停止，Runpod controller31608受控重载为53968，仅保留云端状态/训练恢复，不下载。两项回归exit0证明待传输任务不能绕过禁令、监控仍可运行；云trainer未重启。hhc既有小时automation已更新覆盖旧下载指令。检查时已有step15000完整receipt，选作后续唯一本地使用副本；旧历史文件未擅删，云到云备份尚未配置/验收。

- **2026-10-16 02:10 SGT HHC改进研究实测：** CPU完成冻结32val全有效长度14–271帧、4方向/2能量级的P/V/P+一致V/合法SO3/6D Gaussian五臂探针（37363 exit0）。epsilon.02时SO3/P响应比.4640，动作聚类bootstrap95%CI[.4192,.5121]；Gaussian/SO3为1.5760[1.4882,1.6660]，不能把旧4条Gaussian“旋转更敏感”外推成loss权重或模型排名。找到原HHC训练只drop global文本、CFG采样同时drop word/global的差异；独立historical/matched单因素40k入口完成，真实拓扑tiny CPU路由/梯度检查exit0，无新HHC训练/FID。缺HHC VQ185k、native数据、MM冻结文件，旧server SSH超时，Drive134项无对应权重；后续资产直接传云，不经本机ckpt下载。四类基线论文表与HHC历史20rep结果已整理；HHC8.701897±.108161仍非SOTA，TIMotion需区分RWKV4.702与Transformer5.433。完整证据/协议/下一步：docs/hhc_sota_interhuman_20261016.md。

- **2026-10-16 01:24 SGT step12000完整恢复包云备份完成：** 2188976561B/SHA ad87b6136dfbc7cb00ddd99f241e9323061fda7de68f9b4e49b90eaed1eef11a，33片逐片上传并metadata大小核对，最终manifest16537B也已读回，Drive ID1o4H7FFIBOTmkAws4Q_P6UNFr0dcxUM6E；原receipt已uploaded_parts，记录manifest URL与有序分片ID。此包保存新Runpod分支12000时的模型/optimizer/RNG，dataloader cursor未保存，不能称bitwise恢复；区别于初始10000 reset状态，也不称当前最新14750状态已云备份。训练正常14750/loss0.25819496，暂无新完整480val；best仍继承10000/FID44.620449。当前后台未重启，继续704000目标。

- **固定CLIP源码也已本机/云归档：** 已按实际运行commit d05afc436d78f1c48dc0dbf8e5980a9d471f35f6从官方公开源收完整源码，安全路径解压并核对冻结模型URL hash；归档4312596B/SHA553c80a6364ac7350dd7800f535531175f22c11fa742bda1b50b62121eb5f40f上传并metadata读回Drive ID1TAw7tbQ3nOWpPb3E0hlrMSmNHm6QHfZs。它补齐源码重建入口，不是全部Python环境离线镜像。当前step12000完整恢复包33片继续上传，尚未完成云验收。

- **2026-10-16 01:00 SGT 首个更新后全状态备份已本机验收：** Runpod step12000恢复包2188976561B/SHA ad87b6136dfbc7cb00ddd99f241e9323061fda7de68f9b4e49b90eaed1eef11a已完整收取，有成功receipt；本轮再次整包/分片校验通过，33个64MiB分片准备上传并逐片读回，全部片/最终manifest完成前保持cloud pending。这是新Runpod分支真实训练更新后的optimizer/RNG全状态；区别于此前初始10000的fresh状态。训练刚由闲置保护确认13900，无新完整480val，不重复报告新FID。controller31608和旧initial-backup30948均存活，不重启/不另开下载，旧monitor依owner禁用。

- **2026-10-16 00:12 SGT 本轮两项云备份完成：** 新Runpod初始10000 recovery729659645B的11片及manifest5712B均读回，ID1WIVhyXaFhEia399v7jcaq23o9UPykAuZ；训练CLIP934088680B的14片及manifest6648B也全部读回，ID1uyLmRgGbILuKPOIxXz17skwZFMxnjxLq。整包/各片SHA与receipt绑定，两原receipt已uploaded_parts并保存manifest URL/有序分片ID，不重复传输。当前训练12650/loss0.21804528，无新完整val；12000新全状态2188976561B/hashad87b6136dfbc7cb00ddd99f241e9323061fda7de68f9b4e49b90eaed1eef11a仍由controller31608按固定快照续传，不能冒称最新状态已云备份。旧14k partial仍保留，旧monitor禁用。

- **Runpod新分支初始10000恢复包云备份验收完成：** 729659645B/hash eb206d8e519fcbaa4d1a7db975006f31920e7be24f214eb5b55220ac9c20fb05，11个64MiB分片全部metadata读回，最终重组manifest5712B读回一致，Drive ID1WIVhyXaFhEia399v7jcaq23o9UPykAuZ，receipt已uploaded_parts并保存有序分片id/url。这是明确重置optimizer/RNG后的新分支初始状态，不能冒称旧全状态或最新12000恢复包。额外训练CLIP14片仍继续上传，完成前保持pending。

- **2026-10-15 23:50 SGT 本机新增资产验收及云备份执行：** 新分支初始10000恢复包729659645B/SHA eb206d8e519fcbaa4d1a7db975006f31920e7be24f214eb5b55220ac9c20fb05已在本机有成功receipt且再次整包/分片核验通过；额外训练CLIP ViT-L/14@336 934088680B/SHA3035c92b350959924f9f00213499208652fc7ea050643e8b385c2dac08641f02同样已完成，31392正常退出不重启。两项共11+14个64MiB分片正在并发上传，逐片metadata核对并立即持久记录id/url；最终manifest与receipt完成前不称云备份成功。Runpod真实12000，已生成带当前optimizer/RNG的新恢复包2188976561B/hash ad87b6136dfbc7cb00ddd99f241e9323061fda7de68f9b4e49b90eaed1eef11a，尚未本机收取；本次上传10000是重置optimizer/RNG后的初始分支状态，不是12000。后台31608/30948存活，不重启；旧monitor按owner禁用，无新完整480val。

- **2026-10-15 22:55 SGT 补齐额外训练CLIP的本机副本：** Runpod训练正常到10250，无新完整val；controller31608及旧initial-backup30948存活，不重启。新分支初始recovery固定快照partial在增长，尚未验收/上传。启动独立隐藏下载tools/local_training_clip_download.py，PID31392，沿官方Azure模型URL收取额外ViT-L/14@336，固定934088680B/SHA3035c92b350959924f9f00213499208652fc7ea050643e8b385c2dac08641f02；单锁、同一partial续传，整包SHA通过后才receipt。状态training-clip-status.json、PID/日志training-clip.*；完成正常退出不重启，之后按64MiB分片上传Drive并读回metadata。此前Runpod已校验这项，不影响现有训练；本机副本仍未完成，不冒称备用资产本机全部齐。

- **Runpod接管包已云备份验收：** migration/runpod-takeover-20261015.tar.gz 203326B/SHA20648ab1702516118f93e4af82ad7bf516819b9d0d010f049df8c8e1328e7c06上传并metadata大小读回，Drive ID1rqRMwX6Fb7xgqAzJv8lKQ_r4wvB559DL，receipt uploaded。包含部署/恢复/控制代码、接管文档和实际runtime freeze；不是训练checkpoint。新checkpoint仍后台固定快照收取，partial不称备份完成。

- **2026-10-15 22:53 SGT 切换运行稳定：** 实际10150，loss0.31806685/grad_norm4.901733有限；无新480val，不把loss改善当生成质量结论。新增固定依赖的runpod_bootstrap.sh，以便实例重启后重建/opt环境与内存盘数据。为加载此恢复入口，仅受控停止本聊天旧本机controller24216及其归属SFTP、确认退出后启动31608，原runpod-transfer.json及同一partial保持不变；没有重启远程训练。后续以runpod-control.pid.json为准。源代码/部署与runtime freeze清单正在打包云备份。

- **2026-10-15 22:50 SGT Runpod训练更新已验收：** 新分支实际到10050（50次更新93.55秒），total loss0.52555495、grad_norm5.737349有限，lr9.995475e-5，目标仍704000。当前/继承best为10000/FID44.620449，来自旧scratch的完整480val，未新增Runpod FID、不将训练loss当质量改善。新全状态备份controller24216存活，已固定初始10000恢复快照并沿本机partial收取；此恢复包是新分支fresh optimizer/RNG状态，不是旧14k全状态。闲置保护每15分钟检查、持续空闲30分钟停机，旧服务器自动拉起已禁止。下一完整480val20000。

- **2026-10-15 22:49 SGT Runpod真实训练入口已启动：** pod3aj1czzq0a9x64上，scratch step10000权重、SMPL与raw压缩包SHA全部匹配，processed2462028219B及InterCLIP1513626777B整包SHA通过；训练额外CLIP ViT-L/14@336 SHA3035c92b350959924f9f00213499208652fc7ea050643e8b385c2dac08641f02已补齐。480冻结验证输入全部回放通过，请求SHA751efc9ced91e2a6ad4eec4a7b9316946761fe13f86d09af7d5c04a9226b7f93不变。训练器已加载182394124个trainable参数并进入GPU计算，GPU占用约5.8GB；正在等待每50步首条更新日志，尚无新val。新分支初始恢复包10000为729659645B/SHA eb206d8e519fcbaa4d1a7db975006f31920e7be24f214eb5b55220ac9c20fb05，optimizer/RNG重置已明确写入config，不能称沿用旧全状态。Python环境在/opt/hhc-env（/workspace/hhc-env为链接）；数据展开在/dev/shm/hhc_inputs，持久卷保留zip/raw包以便重建。新本机controller24216负责Runpod状态、按小时恢复/断点备份及旧机可达后停止本聊天旧tmux；旧机当前仍未知，不声称停止。依赖Matplotlib/NumPy兼容问题已修复，freeze清单正在收取。

- **2026-10-15 22:23 SGT 按用户纠正实际切换Runpod：** 撤销“最新全状态包未齐就不部署”的自加门槛；实时社区4090无货，安全云LOW，报价US$0.89/GPU小时后已创建本聊天pod`3aj1czzq0a9x64`，80GB持久存储。SSH成功，实际RTX4090 24564MiB、Torch2.8.0+cu128；源码/冻结请求已上传，SMPL已到、best10000及raw正在上传，公开processed/evaluator/CLIP由新机校验收取。旧monitor受控停止，compute-owner.json与源码门禁禁止再次自动拉旧机训练；旧initial-backup保留原断点继续救回全状态。Runpod闲置保护hhc-runpod每15分钟检查，空闲30分钟停机保留存储。新增明确scratch weights-only续训入口，验证来源scratch/哈希/统计/参数，保留原step与best，记录optimizer/RNG reset；不是完整全状态恢复。代码编译通过，尚待真实运行验收。发现训练另需CLIP ViT-L/14@336，已纳入补齐，纠正旧资产完整性表述。

- **2026-10-15 22:05 SGT 恢复本机控制后台：** 用户要求继续后复核，旧monitor37724及initial-backup57948均已不存在，按命令行全量查找也无同任务Python进程；退出原因未确认。仅在上述退出证据确认后，隐藏启动monitor30724、initial-backup30948，各自PID文件已更新且进程存活；日志为migration/control/monitor-recovered.*与initial-backup-recovered.*。step14000固定快照partial仍1053358080/2188976673B，rolling-transfer.json的remote/SHA未变，未重开重复下载。独立SSH仍超时；remote-status中的17700是旧快照，不能代表当前训练存活，无新完整480val。Runpod切换仍受旧trainer状态未知和最新全状态包未齐限制；未声称训练已恢复。

- **2026-10-15 06:53 SGT 冻结InterCLIP云备份完成：** 1513626777B完整文件已按原SHA验收，23个64MiB分片逐片上传并metadata大小读回，最终manifest11206B同样核对通过；Drive ID`191GD3KFIBR07sszc3p5bIDzDIfaGc101`，receipt已uploaded_parts并记录所有有序分片ID。runtime-assets62896正常完成，不重启。服务器复核仍连接超时，monitor37724/backup57948存活，14k全状态包保留原partial；最新已确认训练仍17700（历史快照，不代表当前存活），无新完整val。Runpod尚未创建/训练：US$0.34社区4090报价已告知，需完整恢复点、旧trainer停止/隔离证明及闲置停机保护，不能把备用资产齐全等同于已成功切换。

- **2026-10-15 06:38 SGT 冻结评测器本机验收完成：** 官方HTTP Range成功沿原partial补齐InterCLIP1513626777B，整包SHA256 4e14b8ee5c2d965778fe003fb37d98ed8e5c7e0940668515fc7dcfa2831c17f8与原服务器资产完全一致；原receipt生成，runtime-assets62896正常完成，不重启。固定源码、原始/processed数据、female SMPL、CLIP、InterCLIP现已本机收齐，GPU环境/端到端恢复尚未验收。23个64MiB分片正在上传Drive，云完成前保持pending。最新14k optimizer/RNG包仍partial、旧机是否仍训练未知，尚未启动Runpod或改scratch协议。

- **2026-10-15 06:29 SGT 心跳：服务器连续失联，Runpod预检进入备用准备。** monitor、initial-backup均仍存活，独立SSH及后台多次连接超时；没有证据证明旧trainer已停止，不能启动第二份训练。Runpod实时社区4090为US$0.34/GPU小时、availability LOW，官方PyTorch模板已读回；未创建付费pod。完整14k recovery仍为partial，best10000仅模型权重，不能静默重置optimizer/RNG声称完整续训。冻结InterCLIP官方发布ID1bJv5lTP7otJleaBYZ2byjru_k_wCsGvH metadata大小1513626777B与原资产吻合，HTTP首次探测读超时。为已有资产worker加入官方HTTP Range备用，复用同一个partial、整包固定SHA通过后才验收；确认旧57740无SFTP子进程且受控退出后启动62896，monitor/backup不动，未重复下载。代码编译通过。Runpod切换门槛记录于migration/control/runpod-readiness.json；依赖与完整恢复点、旧机停止证明、闲置停机保护尚未齐备，备用未宣称成功。暂无新完整val。

- **2026-10-15 05:33 SGT 检查后best10000云备份完成：** 首个已验证best模型729644528B（SHA e432c7a2ed82ae3fbfbb7bcdb00525e241eee0e44d35297a9b23e66d0454c2dd）本机校验后分为64MiB×11片上传，每片metadata大小核对；最终manifest5566B也已读回，Drive ID`1OAc3tMjGG-096kA1Z0amdN-CKN0RBgjY`，receipt标记uploaded_parts。这是模型权重备份，不是带optimizer/RNG的最新训练恢复包；14k完整recovery仍在固定快照续传，InterCLIP仍未齐。备用源码Git/输入状态包20436839B/SHA d6068637fd90ca7b13f9e9d985c8ceed213ed1a9201535d4be945684ac08105d已上传读回，ID`1hJID8imIBmHe-CvDfDrKu0MSh9_AefPk`。训练正常恢复至17700，尚无第二个完整val，不更新或重报新FID。

- **2026-10-15 05:33 SGT 检查与输入迁移验收：** 原trainer124657在16850报Aborted/core dumped（此次不是仅凭推测记SIGILL，具体原因未确认）；小时监控从16500恢复，新trainer126483已到17700。仍无20000完整val，最新/最佳仍10000/FID44.620449，不重复报新效果。processed ZIP2462028219B整包SHA ad951d24f95fff550d964c8f977eb99da47264b81e3080312c95046e69fd6325通过，15554个文件完成ZIP CRC/路径检查解压；原始7779文件MD5已通过，全部480冻结验证输入本机存在，请求未变。补齐独立third_party InterGen Git身份：核对服务器官方remote/commit后fetch精确b1750df4a859443b8a69329f759ce3a7503ae179，code/config diff为空，仅未迁移readme媒体；避免Git向上误识别HHC主库commit。best10000本机729644528B/SHA e432c7a2ed82ae3fbfbb7bcdb00525e241eee0e44d35297a9b23e66d0454c2dd验收通过、后台正常退出不重启，正在11片上传；未标为云完成前不冒称成功。monitor37724、修复backup57948、runtime-assets57740均存活，InterCLIP仍partial（本次1211335680/1513626777B），备用评测器尚未齐。

- **2026-10-15 04:33 SGT 小时检查与备用数据推进：** trainer121970在14600再次SIGILL/exit132，监控已从完整14500恢复，新trainer124657已到15700；当前最新完整val仍10000/FID44.620449，也是best，未新增评测。不动驱动/不重复训练。公开原始motions7779文件/1245872625B全部本机MD5通过（failures0，下载进程正常完成不重启）。发现旧rolling backup超时后会换到新step留下旧partial，已改为`rolling-transfer.json`持久固定同一不可变快照；模拟超时+重启回归通过。为加载修复做受控维护：终止归属53732的SFTP及旧备份进程，确认旧进程退出后仅启动新backup57948，保留14k原partial；快照2188976673B/hash991ec2b154cc962b419f7a0d2a93b1f60af9ebac0462a9d706315453fb83aa49。monitor37724/训练均未停止。官方processed ZIP已从Drive实际目录确认ID1qpFij33tVZhUkk9-kDaDy1FuQb0LJSEf、2462028219B，对照服务器完整原ZIP SHA ad951d24f95fff550d964c8f977eb99da47264b81e3080312c95046e69fd6325，Range206/1MiB/PK签名探测通过；隐藏下载61212已启动64MiB断点请求，完成后整包SHA/ZIP CRC及路径检查再解压，不覆盖不匹配已有文件。best10000和InterCLIP仍未验收/上传；备用环境尚未齐。

- **2026-10-15 03:43 SGT 首次完整scratch 480val验收：** 修正路径后10000步DDIM50/CFG3.5完整生成与CPU冻结InterCLIP评分成功，seed20261006、固定480请求、batch96；FID44.620449（当前/历史最佳均step10000）、Matching3.996044、R1/2/3=.102083/.191667/.247917、Div7.122461（GT7.741262）。这仅是scratch首个训练内val点，不称性能改善；MM及独立test待评。评分JSON+embedding本机收取，embedding SHA256与报告匹配；评分包1828513B/SHA `9d216cbc93c5e33d75e99fe122e6417000a1ba2ee6cbede68233ad11fb19b2e0`已上传Drive并读回，ID`1nrzAVzsVYF4ZBHZt_a-g2BC0YlEARw4k`。训练已继续10100，best10000/recovery10000均已生成。已固定best10000不可变硬链接并核验step/hash，单独后台`download_pinned_best.py`收取729644528B（SHA `e432c7a2ed82ae3fbfbb7bcdb00525e241eee0e44d35297a9b23e66d0454c2dd`），避免后续best替换使传输失配；权重本机/云端尚未完成，评分记录备份不等于模型备份。继续704000预算与小时恢复，下个完整val20000。

- **2026-10-15 03:39 SGT 修复进入真实完整验证/备用资产备份：** 新trainer121970从9500恢复后达到10000并开始480条DDIM50真实生成，cache已73MiB且持续增长，未再出现验证路径错误；完整评分仍未生成，FID/best暂待评，不能以generation进度冒称验收。已有female SMPL247530000B与CLIP353976522B经整包/分片SHA256校验，分别4片/6片上传并metadata读回，重组manifest ID `1NUDvuHqVUDUXx5F28InauJIMi1Yf-_s-` / `1ixV_jvnFn9R5bLE6_wpeir4mSXazPdCY`，原receipt已标记uploaded_parts。迁移专用目录安装requests，不改训练环境；后台公开原始motions下载PID62800，固定原清单7779文件/1245872625B、MD5逐项验证，已见257个完成文件。InterCLIP、processed及新checkpoint仍在补齐，备用环境尚未完整。

- **2026-10-15 03:33 SGT 首次验证路径修复已部署：** 9500→10000触发验证后因默认adapter相对目录被解析到`/home/wenning/dataset/...`而exit1；本机训练入口改为`InterHumanValidationRequestDataset(adapter=data.adapter)`显式复用已验证训练数据根目录，不改冻结480请求/统计/hash/loss。CPU逐项回放全部480请求exit0，实际root为原官方数据路径。待旧进程明确退出后才部署文件，部署/恢复主句柄10589 exit0；新tmux日志`resume_1792006402.log`从完整9500恢复。首次有效val尚未完成，不以旧STATUS10000误报评测成功。SMPL与CLIP已本机SHA256核验，正按64MiB分片上传；InterCLIP/公开motion数据仍在补齐。

- **2026-10-15 02:26 SGT 检查后初始checkpoint云备份完成：** step0完整恢复包729659053B再次独立SHA256复核通过。连接器入口512MiB、传输层实际100MiB双重限制使整包/256MiB上传失败，已改64MiB×11片；每片上传后读回大小，整包及每片SHA256/顺序/云ID保存在重组manifest（Drive `1URxgXMaKG-ACxd0D0DTdINqnPUApUxkR`），manifest5635B也已读回，原receipt标记`uploaded_parts`。恢复须按序二进制拼接并核验整包hash，不把单片当checkpoint。**这仅是初始step0备份，较新的训练恢复点仍在后台收取，未冒称最新模型已云备份。** 本次SIGILL已由小时监控从7000自动恢复并验证继续到7300/7350；完整480val/FID仍待首次10000步验证。

- **2026-10-15 02:26 SGT 崩溃后自动恢复：** 原trainer114898在step7200报Illegal instruction/exit132；后台小时monitor自动从完整step7000 recovery恢复，trainer118561，现已见7300/7350有限loss，无重复启动。具体本次SIGILL诱因未确认，不改驱动/不影响其它作业。仍未完成首次10000步480val，FID/best待评。初始step0恢复包729659053B已续传完成且后台SHA256通过，收据待云端上传；本次正在独立复核并上传Drive。monitor37724/backup53732/runtime-assets57740仍运行。

- **2026-10-15 01:34 SGT 小时检查/备用依赖推进：** monitor37724、恢复包续传53732均存活，下载1小时超时后已自动reget续传（213073920/729659053B，未验收），未重复启动。服务器仍可连接，scratch step7200、尚无10000步完整val/FID；RAM盘28GiB余量，训练无异常。新增冻结InterMask源码/统计、官方split/annots、原始数据校验清单及三项大资产SHA256清单，依赖元数据包1587867B/SHA `56955c3b3f98e006abd308edfc81042dfb0da0d8db2f24ee67515a9aa1f3a816`已上传Drive并读回大小，ID`1nnHhkB0AW7mXk6sHiYQSMT7qTUo7U93V`。隐藏队列`local_runtime_asset_download.py` PID57740等待初始恢复包核验后按清单收取已有female SMPL/冻结CLIP/InterCLIP；大资产与公开motions/processed仍未齐，不能宣称Runpod已可立即恢复。无新增完整评测，不发性能通知。

- **2026-10-15 接管运行验收：** 本机状态复核已到scratch step850、loss1.659450有限；首个step500原子恢复包2188976033B、SHA256 `73dfc2722062586ead1b5ca82f247fa22f1e47afc9614ed7d2502783139821dc`。服务器独立CPU读取检查exit0，模型/恢复步数相同、模型全有限、196个optimizer状态均有限，恢复收据功能通过真实保存检查。完整val尚未到10000步，FID待评；不以loss下降替代性能验收。Git/历史/小评测记录与源码已完成本机/Drive保存，初始恢复包仍在续传，当前step500包仍仅在服务器RAM，云端checkpoint备份未完成。小时自动化与后台控制持续运行。

- **2026-10-15 迁移资料备份进展：** 主会话5517条消息、中文history索引、Git历史、冻结请求和原评测小记录均已本机保存；消息429348B及Git/状态包23991350B已上传Drive。源码及本机控制/接管文档继续保持本机权威。后台monitor PID37724、恢复包续传PID53732；checkpoint仍未完成本机/云端校验，不能声称全部资产已迁移。恢复后已见step150/loss5.168862有限，完整scratch val尚未到10000步、FID待评。小时heartbeat automation `hhc`已激活，只对实质变化/异常通知；服务器当前可用，Runpod暂无付费资源。

- **2026-10-15 本机接管并恢复scratch：** 权威目录改为`D:\projects\HHC`；通过SSH `newstation`成功访问edmond-personal，原项目`/mnt/8T-HDD/wenning/HHC`。核对真实历史最新指令为旋转/FK InterGen从头训练+小时自动恢复。原训练停400但持久recovery仅step0；完整恢复检查19834 exit0（模型/optimizer/RNG/config一致），从0重跑未保存部分。服务器HDD余115MiB，未清理旧实验；新计算run移到`/dev/shm/hhc_local_control_20261015/output/intergen_rotation_scratch_20261010`，tmux`hhc_local_scratch_20261015`、trainer114898，已见step1 loss6.396909有限；704000步/bs32/lr1e-4/10000步480val不变。源码本机已落盘并上传Drive核对2441248B；历史/Git和729659053B step0恢复包仍在下载，**checkpoint尚未完成网盘备份**。本机隐藏后台`local_compute_control.py`按小时恢复/续传，追加原子恢复收据便于核验。Runpod认证读取可用、无pod；当前未启用付费备用。详情、备份链接、后台句柄/限制及下一步见`docs/LOCAL_CONTINUATION.md`；内存盘不耐主机重启，需完成本机/Drive冗余后才算恢复资产保存完成。

- **10-10 用户明确授权自主启动scratch及持续监控，正式流程已启动：** 按最新指令清理旧9-02完成run的`model-000010000-0.4495.ckpt`(3943078250B)和`model-000020000-0.3532.ckpt`(3943078761B)，不可直接撤销；后期/best/latest/所有指标记录保留，释放约7.4GiB。新run `output/intergen_rotation_scratch_20261010`，tmux `intergen_scratch_hourly_20261010`、守护PID220948，日志`output/intergen_rotation_scratch_monitor_20261010/watchdog.jsonl`与独立resume日志。零InterGen权重初始化，仅冻结CLIP；704000步/lr1e-4/有效batch32，每10000步480val，每500步保存，step0也保存可恢复状态。scratch的同step模型+optimizer/RNG恢复包改存run/recovery.pt持久盘（原子替换）；每小时检查，崩溃续训，避免重复进程。当前启动并计算train root stats，尚未得出效果结论。目标create被旧unfinished目标拒绝，get_goal仍paused，工具无resume权限；没有伪称系统目标active/完成，本机守护独立运行。后续验收需要冻结best后独立test/MM，不能以启动即完成。
- **10-10 用户改为从头训练；实现/冒烟通过，正式运行因持久磁盘空间不足未启动：** `train_intergen_rotation.py --scratch`完全跳过InterGen checkpoint读取（用不存在的--checkpoint路径实测），生成网络/可训练文本适配随机初始化，仅冻结CLIP保留预训练；旋转输出bias设identity避免FK初始奇异。主96421两步microbatch4反传+DDIM50导出exit0，config checkpoint/hash均null。scratch默认lr1e-4、有效batch32、约704000步（5637×2样本、2000epoch对应预算）、warmup约10epoch，每10000步完整val；不是沿用10000步微调预算。正式新run/小时守护尚未启动。只读空间核查：/home约432MB、HDD55MB，best/latest各696MiB且hash不同，无明确重复可直接回收；未删除任何文件。需用户指定≥5GiB持久空间或批准清理过期checkpoint，再启动长训；不以RAM smoke冒称正式训练。
- **10-10 13:19 BST 训练已完成验收：** 小时守护10-09 19:08发现恢复进程183854在step3300再次SIGILL（exit -4），自动启动185132；21:08确认训练10000步exit0并停止守护。最佳完整480val为step6000 FID8.943854（step0 9.681466，下降7.62%），最终step10000 FID9.181300。最佳step6000 Matching3.804716、R1/2/3=.429167/.570833/.656250、Div8.006828；step0对应3.804834/.412500/.585417/.672917/7.991972，GTDiv7.741262。FID改善但R2/R3下降、Div偏离略增，不称全面改善；MM与独立test仍待评。下一步冻结best6000独立test及MM，当前没有训练/守护继续运行。
- **10-09 按用户要求恢复并部署每小时自动检查：** 系统journal确认原SIGILL落在`libcuda.so.570.195.03`，具体诱因未知，不重置驱动/不动其他用户GPU作业。主67200检查step2000模型/optimizer均有限且step一致；主43621完整加载模型+优化器+RNG恢复检查exit0。训练入口新增`--resume`、独占锁、跳过已完成初始化val，恢复中断val会保留原产物；后续RAM恢复包将权重和optimizer同step原子保存，避免旧两文件半写错配。后台tmux `intergen_rotation_hourly_20261009`启动`tools/watch_intergen_rotation.py`：立即恢复2000→10000，每3600秒检查一次，崩溃记录exit并续训，完成退出；不唤醒LLM逐分钟轮询。监控PID/训练PID/每次独立log见`output/intergen_rotation_fk_20261008/{watchdog.pid,trainer.pid,watchdog.jsonl,resume_*.log}`。当前最佳仍step2000/FID9.412841；下一步检查恢复后训练推进及4000完整val。限制：约300未保存step重跑、dataloader游标未保存非bitwise续训、base与恢复包仍在/dev/shm，主机重启不受此守护覆盖；/home仅约880MB可用，未清理用户文件。
- **10-09 18:05 BST 状态复核：InterGen FK训练异常中断，未完成：** 日志最后step2300/10000，shell明确`Illegal instruction (core dumped)`、exit_code=132，原PID128670不存在；主81692当前Unknown，因此以shell终态+PID核查为准，不当仍在运行。STATUS.json里的training/2300是崩溃前遗留。最后完成完整480val为step2000，也是当前最佳：FID9.681466(step0)→9.412841(-2.77%)；Matching3.804834→3.803573；R1/2/3 .412500/.585417/.672917→.437500/.587500/.662500；Div7.991972→8.060261，GT7.741262（偏离变大）。不能称全面改善；MM/test待评。best/latest文件存在，尚未核验恢复内容；此次仅状态报告，未重启/修改训练。下一步定位SIGILL并核验checkpoint后恢复。
- **10-08 用户最新任务：InterGen旋转主导训练已实现并启动正式流程：** 暂停旧旋转后处理MM扩展，优先按用户要求改InterGen训练。新增每人root6（262→268），同步原始时间采样、坐标/anchor变换、镜像/person换位；评测剥离root6，GTshape只用于训练FK骨架而非生成条件。64组合GT FK对齐最大9.86e-7m，梯度/遮罩测试70415/66620 exit0；训练两步smoke40735、microbatch4+DDIM50导出smoke84441均exit0。正式主session81692，run `/home/wenning/HHC_interhuman_runs/intergen_rotation_fk_seed20261008`；train root stats→step0完整480val→10000步官方权重微调，bs4×acc8/lr1e-5，每2000步完整val。当前启动/初始化，尚无效果结论。日志`output/intergen_rotation_fk_20261008/run.log`、终态exit_code；设计、loss权重、检查及临时checkpoint依赖详`docs/intergen_rotation_fk_20261008.md`。原baseline未修改，例行监控至少10分钟一次。
- **10-07 17:43 BST 独立test rep0已完成验收：** 原始主句柄70655（InterMask）/55717（InterGen）均收取exit0，文件exit_code也均0。冻结1056条test、seed1234、CPU评分：InterMask raw→R-only FID5.195238→5.142410（-1.02%），Matching3.792476→3.792310，R1/2/3 .442235/.592803/.688447→.439394/.593750/.686553，Div7.951096→7.948615；InterGen raw→geometric_final FID6.403511→6.150488（-3.95%），Matching3.798156→3.797202，R .434659/.618371/.690341→.435606/.616477/.693182，Div7.797199→7.791027。GTDiv7.655049。只改旋转，P/V/contact不变；独立test上小幅FID收益延续但检索有涨有跌，不能宣称全面或统计显著提升。MM尚未启动/待评，当前无运行任务；下一步补重复生成MM、多seed置信区间。完整结果`output/rotation_test_rep0_20261007/{intermask,intergen}/paired_metrics.json`。
- **10-07 独立test配对评测已启动（当前任务）：** 已在看test指标前冻结InterMask R-only step600/full、InterGen geometric_final step2000/full，权重hash及选择理由见`docs/interhuman_rotation_methods_20261007.md`。两个官方生成脚本新增test replication/GPU入口，2条全长GPU smoke主1888/41316均exit0；编译与shell语法检查通过。正式rep0为1056条、seed1234、官方采样设置，GPU0 InterMask主session70655、GPU1 InterGen主55717，启动脚本`tools/run_rotation_test_rep0_20261007.sh`。每臂自动生成→冻结校正/原始共缓存CPU编码→FID/Matching/R1/2/3/Div评分；日志与最终exit_code在`output/rotation_test_rep0_20261007/{intermask,intergen}/`，动作cache在`/dev/shm/rotation_test_rep0_20261007_*.h5`。当前仅启动，未验收，不混称20rep结果；MM重复生成尚未启动，下一步收取两原始句柄终态并补MM。仅修改本地诊断入口，未改变官方baseline源码/权重或HHC训练。
- **10-07 10:43 BST 指标优先的真实训练终态评估完成：** geometric_final主4064 exit0、完整非MM补评9208 exit0；固定2000步，内部角误差4.8596→4.8833°略差，但InterMask FID8.780644→8.721918、InterGen9.681665→9.473583，证明不能以该辅助角误差作为拒绝评测的门槛。InterMask Matching3.792391、R1/2/3=.431250/.606250/.697917、Div8.068943（R1比raw降.002083）；InterGen Matching3.805532、R=.435417/.579167/.670833、Div7.954598。第一批免训练与训练候选均已完成，仅480val探索，MM/独立test待评；下一步按FID及语义检索权衡冻结候选再独立验证，不把一致性当目标。完整表见旋转方法文档。
- **10-07 第一批免训练/训练方法统一CPU评分已完成：** P+R复核60966/R-only60507均exit0，原FID精确匹配。P+R full FID：InterMask8.722070、InterGen9.550627；R-only full：8.700338/9.507802，均比raw8.780644/9.681665小幅改善。R-only Matching/R检索未退步（详`docs/interhuman_rotation_methods_20261007.md`）；一致性仅微变，不因它未改善而淘汰。免训练VQ full为8.671397/9.813278，InterMask检索下降，不能只报FID。MM仍待评、无独立test确认。几何噪声训练旧角误差选模落step0，不视为真实训练结果；已按指标优先原则启动固定2000步真实终态候选主4064，目录`output/rotation_methods_20261007/geometric_final`，训练后自动两模型CPU完整评分。两个小校正器、数据和主HHC模型检查点均保留；主HHC训练/20rep已完成，无被新任务打断。
- **10-07 用户最新强调：目标是评估指标改善，一致性只是候选手段：** 不再把FK/角误差改善作为方法成功标准；应以验证集FID及Matching/R检索/Div的联合变化筛选，最终冻结后独立评测。当前已完成免训练VQ旋转投影（alpha1 InterMask FID8.671397、InterGen9.813278，alpha.25 InterGen9.675755；腕identity几乎无效），Matching/R/Div补评26199 exit0，MM因无repeat缓存待评。两小型训练校正器已训练2000步（原38741/49840均exit0），P+R最佳内部holdout角步200、R-only步600，GPU评分较CPU约+.006，现重载选定模型CPU复核：R-only主60507、P+R失败初始化后重试60966；首模型InterMask full FID8.700338/8.722070，均小幅改善但不是显著提升证明。几何+旋转扰动对照主27867目前按先前角误差选模落在step0，不得将identity评分冒充实际训练终态收益，也不能仅因角误差未改善否定其潜在评估收益。应补实际训练终态候选的评估，不能继续按一致性优先筛掉候选。
- **10-07 训练对修复与初始化失败记录：** 原数据构建88251/PID80974 exit0，split标签U15截断但主训练按split_index896划分正确、全部ID仅train且与val/test无交。agent重建60697/PID81925 exit0修正U16与896/128 manifest；当前npz SHA007f7c70ea76f1d8a895bd5117815158e3392a05a79f7c5bcc2503363d94b374。修复曾原位写入，导致P+R CPU复核一次BadZipFile exit1，未开始训练；文件完成后独立输出目录_retry启动60966。旧数组已不在，不能宣称两次构建逐元素实测一致；ID/seed/crop/VQ流程确定相同。原两模型已在首次完整数据上完成训练，检查点保留。数据现只读，不再改写。
- **10-07 10:17后 用户授权免训练/训练型旋转改进并行探索：** 免训练agent `rotation_vq_projection`（gpt-6-luna/low）在GPU1用冻结官方VQ重建生成输入，仅混合旋转alpha .25/.5/1，外加release腕identity独立臂；已进入CPU评分，未完成。数据agent `rotation_training_pairs`（gpt-6-luna/low）仅从train采1024×64帧GT/对应VQ重建到/dev/shm，896train/128内部holdout；主agent设计实现`tools/train_rotation_residual.py`，两模型P+R与R-only时序残差网络，零初始化+bounded .25 delta，按内部holdout旋转角选模，不用480val训练/FID损失。身份初始化/有限梯度/角误差/刚体坐标不变性自检exit0；等待训练对准备完成才启动GPU0训练。所有实验仅旋转变化，P/V/contact保持不变。
- **10-07 10:17资源核查同时发现主MGT已完整完成：** STATUS于10:12:23 complete/300k，20rep均齐，GPU空闲；自动REPORT给出val选定best40k，正式20rep test FID8.701897±.108161、Matching3.853481±.001380、R1/2/3=.320218±.005517/.447585±.005209/.523153±.004880、Div7.831259±.025797（GT7.788421±.021919）、MM.849837±.038988。当前用户请求为旋转方法探索，未改变或重新训练此主MGT；详细完整评分文件位于原run/test/REPORT.md。
- **10-07 从生成旋转初始化的IK完整480完成：** 主28807/40156均exit0，311/313秒CPU。两模型采用同一零SO3残差初始化、lr.01/60步/旋转先验.02/修正时间差.01；位置速度接触不变，无GT参与生成优化。InterMask FID8.780644→11.520416（变差），估计骨架FK残差125.97→37.55mm，局部修正8.419°，逐帧旋转角1.5616→1.8296°；InterGen FID9.681665→9.163593（改善），FK30.45→6.64mm，局部修正2.094°，逐帧角1.5340→1.5116°。GT自检改动.791°、FID.133491，明确零shape方向/平均骨长近似不完全保持GT。两raw FID与前轮精确相同；不能把FK残差当对GT的MPJPE，也不能因一致性改善声称两模型质量都改善。报告`docs/interhuman_initialized_ik_20261007.md`，各模型`ik_initialized480.json`及embeddings。未改训练或正式baseline输出。
- **10-07 IK试验改为用户强调的生成旋转初始化迭代优化：** `tools/fit_interhuman_position_ik.py`新增SO3零残差初始化（assert初始矩阵与预测旋转严格一致），Adam lr.01/60steps，位置MSE+.02旋转修正平方+.01修正的时间差分平方，保留最佳目标。P/V/contact固定；root因官方无root旋转通道从骨盆估计并固定，局部旋转从生成旋转开始，不从解析IK结果开始；叶节点保留。零shape female rest方向、骨长从各自位置序列均值估计，无GT供给生成拟合。8条GT/两模型pilot1269及4014均exit0，GT改动均.838°、FK残差19.58→2.10mm；InterMask133.20→40.56mm、InterGen33.51→7.08mm，小样本FID128.736→124.948及128.444→126.068，不作正式结论。现已各自完整480执行，输出两模型目录`ik_initialized480.json`。早先解析强IK/soft32试验无稳定FID收益，保留pilot JSON，不混入迭代结果。主训练未改动。
- **10-07 直接GT替换实验完成：** 两原句柄31337（InterMask）/21801（InterGen）均exit0，完整480/seed20261006，约114/115秒CPU；raw FID与此前输出精确一致，替换范围和未替换通道assert通过。InterMask raw8.780644→GT位置7.269357（降1.511287）/GT旋转2.744920（降6.035724）；InterGen raw9.681665→GT位置8.107611（降1.574054）/GT旋转2.560352（降7.121312）。两个baseline都是替换旋转降幅显著大于替换位置（约4.0/4.5倍），支持用户指定诊断下旋转差异影响更大；不能据此推出训练完全没有旋转监督，或将相对FID下降当可加贡献/全部噪声。GT与生成不逐帧语义对齐，保留诊断限制。结果及embedding在两模型目录`gt_swaps480.json`/`gt_swaps480.embeddings.npz`；汇总`docs/interhuman_gt_swap_fid_20261007.md`。未改当前训练。
- **10-07 用户纠正后，直接GT替换实验已启动：** 新脚本`tools/score_generated_gt_swaps.py`复用两个完整480生成cache，分别仅替换P[0:66]或R[132:258]、保持V/contact和原长度/人序不变，原始GT分布不改；同时重算raw并要求与既有结果误差<1e-4。InterMask主session31337、InterGen主21801，CPU2各自运行，目标各目录`gt_swaps480.json`；不是重跑生成，不再改做平滑实验。独立低成本agent确认两cache全480request/GT逐项一致、无NaN/Inf，InterGen同长，InterMask短0–3帧，用GT前缀对应生成长度。实际生成与GT不保证逐帧动作对齐，结果解释为用户指定的特征替换诊断。当前待评分终态。
- **10-07 09:22 BST 两baseline生成消融已完成验收：** 两tmux pane_dead=1/dead_status=0，InterGen full480.exit_code=0；两JSON均complete/limit480/frame_cap0/seed20261006且request SHA一致。InterMask于10-06 22:57完成，InterGen于10-07 00:46完成。完整480val真实生成FID：InterMask raw8.780644/R正交化8.948847/R正交化+平滑8.944336/P平滑8.780078/V重算8.780790/P平滑+V重算8.779010；InterGen对应9.681665/9.698902/9.697059/9.680271/9.683937/9.682499。所有干预对GT与生成对称执行，沿各baseline官方长度政策，非正式test/多seed。结果不支持“6D离流形或本次轻度滤波可去除的旋转噪声推高FID”，但不等价于排除旋转语义/姿态错误主导；尚未完成真实生成的P/R贡献分解，不得把此前VQ重建Shapley60.6%套用。完整JSON分别`output/intermask_generation_rotation_20261006/generated480.json`、`output/intergen_generation_rotation_20261006/full480.json`；对应embeddings已持久化。用户本次询问结果，未启动新实验或改训练。
- **10-06 22:30 BST 480重建归因完成、两baseline真实生成仍运行：** InterMask VQ配对CPU session1852/PID56522 exit0，296.27s；同轮FID PVR=.985273（历史.985952，相差.000679，不称bitwise重现），仅P=.360348、仅V=.001283、仅R=.569187；8组合诊断Shapley P=.382307(38.8%)、V=.005647(.6%)、R=.597320(60.6%)。这支持本轮VQ重建的R影响最大，不证明两个text-to-motion模型的FID由旋转噪声主导；双侧R正交化FID反升1.079522。报告`docs/intermask_feature_fid_20261006.md`及full JSON。真实InterMask 480生成→六臂评分运行于tmux `intermask_rot480_20261006`（pane57154/生成57180），输出`output/intermask_generation_rotation_20261006/generated480.*`，预计约25分钟；32全长generation/eval均exit0，仅管线诊断。InterGen strictload missing/unexpected均空，ckpt sha256 bab341123c27af9d8c3ee7ebc43c2d341bfd0f4103425d06f8af6c717a086f8d；4条smoke65872/评分15253均exit0。480生成→六臂评分运行于tmux `hhc_intergen_rotation_20261006`（pane57257/生成57259），CPU2/ddim50/CFG3.5，预计1.5–2小时；日志及自动持久结果`output/intergen_generation_rotation_20261006/full480.*`，终态`full480.exit_code`。InterGen大cache在`/dev/shm/hhc_intergen_20261006.CtYnWA/full480.h5`临时内存，重启会丢，checkpoint/source同父目录；JSON/embeddings结束自动复制到output。两任务原脚本运行中禁止修改，均不占GPU，不需用户另发评测指令。下一步收两tmux终态与完整生成FID各臂，尚不能报告两模型结论。
- **10-06 22:25 BST 用户要求真实baseline旋转/FID归因，实验已运行：** 不再仅研究GT噪声敏感性。InterMask配对VQ480统计完成：MPJPE109.926mm、旋转角误差5.115°，正在CPU2跑完整480的8种P/V/R混合输入及双侧旋转正交化（agent session1852/PID56522）；这些是重建诊断而非text-to-motion结果。InterMask真实生成4条冒烟session69021 exit0，32条完整长度生成session72280/PID56767，通过后自动480生成+同样干预。官方6D为first-two-columns交错布局，不能用连续3+3解释identity；已统一两agent。InterGen官方ckpt经Drive元数据核实2701617525字节，连接器413大小上限，改官方gdown至`/dev/shm/hhc_intergen_20261006.CtYnWA/intergen.ckpt`（主43083 exit0），官方源码同目录source clone主95968 exit0；未占用训练保存磁盘，agent `/root/intergen_generation_ablation` 正做strictload及真实生成冒烟。已有HHC output/fair_eval/intermask_official_exact为InterX，已排除。当前主训练日志仍推进至234600，无改动；实验CPU低线程不占GPU。最终需分别报告baseline生成诊断及限制，不能由VQ或小样本FID替代两个模型结论。
- **10-06 evaluator偏差研究初步完成：** 主agent已阅读子agent完整报告`docs/interhuman_evaluator_feature_bias_20261006.md`。released InterCLIP实际看每人位置66D/速度66D/旋转126D，删除接触4D；两baseline均完整表示重建+位置派生附加loss，并非只监督位置。CPU探针63855 exit0（4条val×64帧×2方向）等标准化总扰动能量下，旋转embedding响应约位置1.3倍、速度约位置1/20；contact通道扰动响应严格0。局部结果不支持“旋转与速度共同主导”，不能证明FID差距或系统性方法偏差；尚未对baseline生成输出实测。未使用GPU/改训练。建议下一步32条合法旋转与P/V一致性扰动，并做HHC/InterMask同GT重建误差归因。
- **10-06 用户新增 evaluator 偏差研究，已委派：** 子agent `/root/interhuman_evaluator_bias` 独立审查InterHuman evaluator的位置/旋转/速度/接触输入、归一化及训练目标，并以InterGen/InterMask实际表示与loss为baseline核对。用户关于特征主导的判断作为待验证假设，区分实现证据与扰动实验；不改当前训练、不占用训练GPU。报告目标`docs/interhuman_evaluator_feature_bias_20261006.md`，尚未得出结论；如有低开销CPU验证则一并报告，否则给出最小消融方案。
- **10-06 12:24 BST 状态核查：** tmux hhc_smpl_mgt_20261006/PID34624仍运行，日志115400/300000，latest115k；loss.065507、lr2e-4、峰值15.426GiB，磁盘余5.7GiB。最新110k训练内480val：FID12.152609、Matching3.868224、R1/2/3=.310417/.431250/.500000、Div7.883273（GT7.751709）、MM.910640；best仍40k/FID10.165793。相比100k FID12.046650略差、MM.841652提升，未刷新best，不以训练loss下降声称生成质量改善。正式test未开始，保持原300k自动流程；下一验证120k。
- **10-06 09:11 BST 续训恢复验收通过：** 校验会话9982 exit0：latest77k模型/optimizer有限，scheduler77k/lr2e-4、RNG齐全，绑定source/request/VQ hashes一致，JSONL完整；磁盘余5.63GiB高于原4GiB保存门槛。使用原脚本--resume，在独立tmux `hhc_smpl_mgt_20261006`（PID34624，非exec session）恢复同W&B djxe8qmw，日志为run目录`resume_20261006.log`。09:10:38实际完成77200更新，loss=.1482641100883484、gradnorm=.3903447985649109及各accuracy与中断前77200逐值一致。未改源码/超参；原300k→val best→20rep test/MM自动流程继续，下一验证80k；最近70k FID10.992630/MM.847834，best40k/FID10.165793，均训练内验证。独立低成本agent核查W&B及19:45–20:10系统日志，未发现明确OOM/kill/GPU异常，中断原因仍未知。tmux隔离交互终端，不能保证跨系统重启续跑；常规监控保持≥10分钟。
- **10-06 09:05 BST 用户询问恢复状态，实查再次中断：** 昨日恢复后日志推进至77200（10-05 19:51），latest记录77000；当前PID8058及全局MGT训练进程均不存在，旧session70755为Unknown，无确切exit_code，中断原因未确定，不能以陈旧STATUS的training声称仍运行。最近70k训练内480val FID10.992630、MM0.847834，历史best40k/FID10.165793；正式test未开始。本次为只读状态核查，尚未再次启动；下一步核查中断原因及77k checkpoint后恢复原流程。
- **10-05 16:00 BST 恢复验收通过：** 新主session70755/PID8058、同W&B djxe8qmw已从30k恢复并实际算至30200。复算30200 loss=.23514299094676971、gradnorm=.6813701391220093、valid/masked accuracy=.9230974316596985/.9113401770591736，与中断前该点逐值一致（不扩大为全程bitwise保证）；LR2e-4，峰值15.386GiB。W&B仅拒绝旧step30200重复上传，属回放已有记录，并非训练失败。无源码/超参改动，下一验证40k；最近30k FID10.893690、best20k/10.765910，均为480val训练内结果，formal test仍未开始。后续沿原300k→best→20rep main/MM自动流程，≥10分钟低频检查，实时STATUS由后台维护。
- **10-05 15:57 BST 中断核查与同轮恢复：** 旧PID52658不存在，旧主50085已Unknown（无exit_code，不声称正常退出）；系统14:14重启，具体原因无证据。日志最后完整训练30200，尾407个NUL；原文件备份`metrics.before_resume_20261005_1557.jsonl`后仅截去零尾，157条完整JSON原样保留。主28007 exit0验证latest30k和best20k可完整加载、模型/optimizer有限、scheduler30k/lr2e-4、RNG齐全，所有绑定source/request/VQ hash一致。30k训练内480val FID10.893690/MM.941595，best20k FID10.765910；test未开始。已使用原脚本--resume启动恢复，无模型/超参/源码修改；需验收到新更新后才算恢复成功。原自动300k→val best→20rep test/MM任务继续。
- **10-05 08:47 BST 正式MGT已确认训练：** 主50085保持运行，online W&B `djxe8qmw`（https://wandb.ai/edmond_gla/NewInter/runs/djxe8qmw），物理batch64首2步loss9.60967/9.59821、梯度范数3.44078/1.96665（clip1）、峰值15.25GiB；已不是排队或仅初始化。无正式val FID，首次验证在10k。中文实时状态`/home/wenning/HHC_interhuman_runs/mgt_interhuman_smpl_interx_shape_seed114514_20261005/STATUS.md`；完成后报告为同目录`test/REPORT.md`，现尚未产生。主线程不逐分钟轮询，后台自动执行剩余训练、选模、20rep评测。
- **10-05 新SMPL MGT正式自动流程已启动（初始化待确认）：** 主session50085，目录`/home/wenning/HHC_interhuman_runs/mgt_interhuman_smpl_interx_shape_seed114514_20261005`；一个进程依次300k训练→冻结val最低FID选模→20轮test main与MM→自动`test/REPORT.md`，不需要用户另行触发。完整smoke65750 exit0；主复跑单测71355 exit0，滤除冻结VQ的MGT checkpoint严格重载/RNG回放47888 exit0；独立审阅确认两通道映射/AdaBlock/CFG，训练与生成条件合同显式切换。保留原真实无warmup MultiStep（agent实查旧W&B前1k LR恒2e-4，修正先前仅凭args猜warmup说法）。今后不原地改运行中绑定源码；低频状态文件`STATUS.md`/manifest/metrics.jsonl自动维护，正式结果只有20rep齐全且summary complete才算完成。详情`docs/interhuman_smpl_mgt_20261005.md`。
- **10-05 用户授权最新VQ→MGT→评测自动完成，开训前验证中：** 锁定新SMPL VQ best185k（不改VQ）；新`models/mgt_interhuman_smpl.py`复用原Inter-X有shape旧拓扑（1024/16heads、2 cross+7 Ada base+9 head、dropout.1），仅Body96/Root64、12288码本、Root→Body。原历史MultiStep分支源码无warmup执行（不能把未使用args1000当已执行），AdamW2e-4/.5,.9/wd1e-6，150k×.05，300k/BF16/effective64。保留原token-cache single/single/double及token头尾删1策略；数据需要将容量38→76以覆盖InterHuman300帧。冻结val-only选模，不复制旧test侧效应。完整物理batch64 smoke主65750/PID50930，4567训练动作/13540文本，285709616可训练参数；前2步loss9.60967/9.59821且有限梯度，峰值allocated15.25GiB，正在做真实50步/480val生成评分。自动pipeline已实现训练→best重载→20×1056 test与20×100×30 MM流式评分，省去约47.5GiB动作H5；尚未正式启动或声称评分完成。当前主要文件`train/train_mgt_interhuman_smpl.py`、`utils/interhuman_smpl_mgt.py`，独立回归测试并行中。
- **10-05 用户询问mean/std及训练状态，已实查训练完成：** 原session29332收exit0，PID39707已退出，manifest complete/300999总更新/300000主步；10-05 05:40 BST完成，约9小时22分。mean/std确为新Y-up/pelvis/raw276表示的3263条train>=64序列逐帧重新统计，val仅复用train统计；不是Inter-X旧统计或含root速度的归档统计。实读normalization.npz与latest300k/best185k/175k三个ckpt，mean/std逐元素完全一致，有限且std>0，六个padding槽mean0/std1。60次480-val日志无NaN/Inf；最佳FID2.42381963@185k，body MPJPE100.996529mm；末态300k FID2.53196412/MPJPE100.200449mm；175k FID2.53301816/MPJPE101.595081mm；最低MPJPE99.686719mm@275k（非FID选模）。协议fixed_val_first480_center_max300_raw_smpl_v1/seed123，训练内InterCLIP重建FID，不是joint FID/正式test；RMSE、全身MPJPE、root、Full/Body joint FID、滑步、接触F1待评，手部MPJPE/Hands joint FID不适用。未启动后续训练或正式评测。
- **10-04 20:18 BST 正式训练已实际运行：** 主session29332，online W&B `knk4gll9`（https://wandb.ai/edmond_gla/NewInter/runs/knk4gll9），run目录`/home/wenning/HHC_interhuman_runs/vq_interhuman_smpl_raw_interx_handless_seed123_20261004`。已完成首4次warmup优化更新，loss/梯度有限，尚无正式5000-step验证或最佳FID；999 warmup后300k主阶段。后台训练每5k自动验证/记录，主线程不逐分钟轮询；异常自动写manifest failed，后续状态以该目录metrics.jsonl/training_manifest.json为准。当前只训练VQ，未启动MGT。正式入口源码启动后不再原地编辑。
- **10-04 开训门槛通过并启动正式进程：** smoke主35468明确exit0，batch256/12288码本4步反传+480条val完整链路通过；低成本agent独立测试最终实跑exit0，覆盖坐标/FK往返、raw空槽、shape作用、encode/decode/full、EMA状态严格重载、loss有限反传及继承公式。现启动`train/train_vq_interhuman_smpl_raw.py`，正式目录`/home/wenning/HHC_interhuman_runs/vq_interhuman_smpl_raw_interx_handless_seed123_20261004`，等待online W&B与首批更新确认；不是已完成训练。详细合同见`docs/interhuman_interx_raw_training_20261004.md`，原raw root速度补零，已删除手分支；smoke的step0高FID仅链路检查，不当模型质量结论。
- **10-04 用户授权开始严格Inter-X无手训练：** 新训练视图纠正为原`raw`合同：22关节双人6D+root位置，速度槽全零，不把上一版归档的速度送模型。保留原body96/root64通道、删除hand96及其码本，shape条件8维/原mask0.1、12288码本、FP32、batch256/window64及历史999 warmup+300k cosine/原loss权重。新增独立模型/视图/训练入口，未改旧模型。原始Z-up→Y-up与translation→pelvis变换可逆；7778抽验FK对原身体坐标最大2.38e-7m。完整batch冒烟主session35468运行，目录`/home/wenning/HHC_interhuman_runs/vq_interhuman_smpl_raw_smoke_20261004`；先验证反传及完整480-val评分再正式启动。硬盘仅1.6GiB，训练输出放/home（11GiB），仅保留latest/best/175k及原子保存预算。此处为新InterHuman训练，不声称复刻原数据或正式评分协议。
- **10-04 SMPL双人格式全量完成：** 主53034 exit0；`dataset/interhuman_smpl_interx_layout_v1_20261004`含7779归档ID/7777有效对，62232原始数组逐字节恢复。经过实际6D编码/解码，body MPJPE均值0.0000754133mm、最大单关节0.0008821941mm；在线位置最大9.77516e-6m，contact 8/8980112不同，明确非bitwise官方特征一致。主61417 exit0验证正式loader、全部加载ID文本和split原样保留、shape20维及裁窗速度，窗口>=8的train/val/test=5978/575/1172。数据3.1GiB、盘余1.6GiB；shape已经参与数据FK，但新shape条件VQ/MGT尚未接入/未训练，旧基线未改。下一步若训练则需独立276维body/root模型及存储预算，不能直接加载旧524维ckpt。格式/无损边界见新格式文档。
- **10-04 全量配对数据构建运行中：** 主session53034，独立输出`dataset/interhuman_smpl_interx_layout_v1_20261004`，已过3750/7779；原数据不覆盖，当前磁盘余2.9GiB。详细槽位/单位/无损边界见`docs/interhuman_smpl_pair_format_20261004.md`。下一步收终态、检查正式loader及split/text，尚不启动训练。
- **10-04 原生SMPL配对格式开始落地：** 按用户澄清不转SMPL-X，仅借Inter-X双人组织方式。新增独立`dataset/interhuman_smpl_pair.py`与构建器，保留全速AA/translation及真实betas，30FPS视图22关节双人6D+独立root位置/后向速度（276维），另存shaped FK关节位置/速度。32对pilot主62130 exit0，256个原始数组逐字节还原，最大关节误差0.00060858mm、23000 contact值零差异。新回归测试主17967 exit0，覆盖裁窗/批量速度、shape确实改变FK、有限非零梯度和官方524维转换。旧672/524维模型与归一化不可直接复用，尚未接入新shape条件训练，未改旧基线或恢复旧MGT；正在启动全量构建，完成后报告真实全量误差，不能以pilot宣称无损验收。
- **10-03 最终独立复核与源码归档：** 低成本agent只读复核完整下载manifest、7777对JSON及两个空raw，确认覆盖/误差/9处contact边界与主结论一致，不扩大到全492。源码快照`output/interhuman_raw_conversion_sources_20261003.tar.gz`含4工具+1测试，5文件读回逐字节一致、SHA dc35a03740dabef85f6d4b0d3008a42c82d2abd1bb8372cb7a3a8c455e69930d；当前重建脚本SHA与全量结果provenance一致，最终py_compile通过。下载/全量转换/边界诊断均已收exit0，无遗留本次运行任务。
- **10-03 本次原始数据任务全量验收完成：** 下载49128 exit0/7779文件全部官方size+MD5通过；全量转换25979 exit0，7777有效对/15554人次/2260582 person-frames，无帧数/人物对应错误。female SMPL+原始betas+floor(k×59.94/30)+序列化wrist identity，body192平均MPJPE0.00005675mm、最坏单关节0.00092195mm、6D最大差7.15e-7；不是VQ重构成绩。官方在线262接触有9/8980112差异，主20884 exit0逐项证实8处高度阈值浮点舍入、1处速度阈值舍入，未改阈值/标签；不声称bitwise一致。全部原始gender标签neutral、3202种beta向量、官方手部旋转全为固定默认姿态。完整492中的手部位置未重建，当前HHC所用身体表示已复现。详见`docs/interhuman_raw_conversion_20261003.md`和三个output JSON。仅去重旧run完全相同的last/latest，保留全部模型内容，释放3.67GiB、下载后HDD余4.6GiB；未动/home、未重训/改数据入口、未恢复暂停的长期goal或294k MGT。
- **10-03 原始motion下载已验收完成：** 主49128明确exit0，完整manifest status=complete、7779/7779 size+MD5 verified、failures=[]，总1245872625bytes；路径`dataset/interhuman_official/motions`。仅对旧run字节完全相同的last/latest去重释放3.67GiB，保留所有模型内容与路径，未动/home。全量转换验收主25979已从排队进入运行，目标7777有效对（2个raw空动作无processed），当前完成1408对，尚未取得终态；不能把已下载当转换全验通过。
- **10-03 原始/processed缺失ID原因已确认：** 原始完整清单编号连续1–7779；processed压缩包两人各7777，缺3945/4106；这两个原始pkl下载后实读frames=0且两人动作数组均0帧，故没有有效对应，并非解压遗漏/人物编号错配。新usercontent下载49128已过4250/7779、0失败，当前有效全量目标为7777对；自动评测队列现为25979（等待PID19336），尚未启动。
- **10-03 18:43 UTC 下载异常恢复：** media API在1107个已验文件后停滞，独立测试同一缺失文件1113的公开usercontent/uc端点均HTTP200；已终止旧PID19034，主15600明确exit143。旧全量排队96998按完整性门槛exit1（未开始评测），没有误用不完整数据。停止后修改下载器使用公开usercontent直链、20秒超时/5次重试，复用已完成的完整metadata而不重复枚举，主新session49128/6workers，前1000缓存校验0失败。此前已验1107文件均保留，完整下载与全量重建仍待完成。
- **10-03 18:39 UTC 全量验收已排队：** 下载主15600/PID19034（8workers，公开media、官方MD5）继续；旧67516已主动TERM/exit143。排队主session96998等该PID结束，随后要求下载manifest status=complete/verified_count=count才运行全7779对body192及online262检查，输出`output/interhuman_raw_body_reconstruction_full_20261003.json`；未满足则失败退出，不伪称完成。新增两项本地回归测试主23170 exit0，覆盖路径安全、59.94floor采样、7778官方数值对应以及joint-only FK与完整SMPL forward一致；py_compile通过。当前系统goal实查为paused，不能声称旧研究目标在自动续跑；本次用户新授权的数据下载/诊断按普通任务继续，未恢复MGT。
- **10-03 原始→官方身体表示已在128对完整序列复现：** root session85853/10091均exit0。64对128人性别×shape对照全部female+原始betas最佳；128对256人重建192维无帧数错误，平均MPJPE0.00005405mm、最坏单关节0.00061077mm、6D最大差4.77e-7；再走原官方262维预处理，位置最大差2.50e-6m、101000个contact值零差异。关键原因：59.94→30使用floor时间索引，不是[::2]；raw标neutral但实际身体匹配female模板；必须保留原始betas；发布6D的左右wrist为identity。只复现HHC所用身体部分，不声称全492/手指恢复。详见`docs/interhuman_raw_conversion_20261003.md`。
- **10-03 下载切换：** 原session67516到500项出现8次失败，已确认PID18608后主动TERM，主收exit143；未修改运行中源。改用已实测MD5一致的公开Drive media API，保留校验通过的文件，恢复主session15600/8workers，当前750/7779、0失败，尚未完成。历史partial清单文档删除，完整元数据保留output清单；无模型数据损失，未清理更多ckpt。
- **10-03 原始 InterHuman 下载与对应关系排查开始（用户最新优先级）：** 用户授权仅清理 `/mnt/8T-HDD/wenning`，不动 `/home`，休息期间自主推进。完成旧 raw12288_v2 run 的 last.ckpt/latest-000300000.ckpt 字节级 SHA256 相等核验（c1c3c35bcbb027c1d1c6dc95e39f511f55c1e6b08c9db8e596e9bfb73043eb31），将 last 改为 latest 的硬链接，保留两路径/全部模型内容，释放3943090318bytes≈3.67GiB，磁盘可用5.8GiB；该完成run若将来续训，应另立输出目录，避免原地覆盖共享inode。原始Drive文件夹完整分页列出7810项，31项同名同MD5重复，7779个唯一pkl共1245872625bytes；下载主session67516，每个文件校验官方size/MD5，目录`dataset/interhuman_official/motions`，清单`output/interhuman_raw_download_20261003.json`，尚未完成。前次15998在完整下载前因重复文件名保护断言退出，已确认重复均同内容后只下载一份、保存别名ID。
- **10-03 首样本7778对应关系初诊：** raw确含每人betas10/gender/pose_body63/root_orient/trans，157帧@59.94；processed每人79×492，不是训练时262维。旋转逐帧最近邻及明确floor(k×59.94/30)均得到[0,1,3,5,...155]，直接[::2]不对应；按前两列交错展平重建body旋转平均绝对误差约1.1e-7，最大约.001，需继续查数值转换零旋转细节。直接用本地SMPL-X重建22关节仍有约146/152mm原坐标MPJPE，不声称已恢复完整转换；官方issue43转述作者使用SMPL-H（62×3+51×6），正在定位模型/平移差异。原294k MGT未重启，当前优先用户新任务。
- **10-03 原始 InterHuman 数据位置核查（只读）：** 当前 `dataset/interhuman_official` 仅发现 motions_processed、文本、split 及相应压缩包；InterMask/data/InterHuman 和 coshmdm-code/data/interhuman_processed 均链接到此目录。硬盘有限深度目录/归档查找未发现可确认的 InterHuman 原始 SMPL-X 参数包，不能据此断言任意未标明目录均不存在。纠正表述：当前训练未读取 shape，不等于官方原始数据没有 shape；未下载、未改训练。另本日实查 MGT latest 保存 step294000/best100000 FID18.734051，当前无 train_mgt_interhuman.py 进程，尚无300k完成证据，未重启。
- **10-02约11:42 UTC续训290k验证：** agent实查PID176730/launcher176729活、W&B hdsosznv running/latest293260、LR1e-5；val@290000 FID20.606772，best仍100k/18.734051；R1/2/3=.366667/.514583/.585417、Matching3.854004、Div7.686494（GT7.751709）、MM.361471（同100×11/10pairs）、val loss7.670480。FID未刷新best；nonfinite grad/weights=0、磁盘16GiB。原29466继续最后300k终态，完成后主收exit_code并审计/严格重载，不把末态或启动评分当验收；正式test未启动。
- **10-02约10:40 UTC续训280k验证：** agent实查PID176730活、W&B hdsosznv running/latest282200、LR1e-5；val@280000 FID20.036703，best仍100k/18.734051；R1/2/3=.368750/.502083/.552083、Matching3.855466、Div7.674364（GT7.751709）、MM.346195（同100×11/10pairs）、val loss7.680395，valid/body/root/masked acc=.135382/.116003/.154762/.066798。FID较270k回升、未刷新best；nonfinite grad/weights=0、磁盘16GiB、无异常栈。原29466继续最后20k固定训练，完成后核验完整记录/选模再放行正式评分，当前正式test未启动，≥30分钟监控。
- **10-02约09:34 UTC续训270k验证：** agent实查PID176730活、W&B hdsosznv running/latest270960、LR1e-5；val@270000 FID18.957331，best仍100k/18.734051；R1/2/3=.358333/.479167/.577083、Matching3.848890、Div7.681527（GT7.751709）、MM.357096（同100×11/10pairs）、val loss7.689875，valid/body/root/masked acc=.133929/.115933/.151925/.066376。FID/MM较260k改善但未刷新best；nonfinite grad/weights=0、磁盘余16GiB、无异常栈。原29466继续最后30k固定训练，正式test未启动，≥30分钟监控。
- **10-02约09:03 UTC续训260k验证：** agent实查PID176730活、W&B hdsosznv running/latest266900、LR1e-5；val@260000 FID19.444688，best仍100k/18.734051；R1/2/3=.372917/.510417/.570833、Matching3.850403、Div7.616699（GT7.751709）、MM.326243（同100×11/10pairs）、val loss7.650991，valid/body/root/masked acc=.137254/.119068/.155440/.068530。检索有提升但FID未刷新best，MM仅小幅波动；nonfinite grad/weights=0、磁盘16GiB、无异常栈。原29466继续固定300k收尾，正式test未启动，≥30分钟监控。
- **10-02约07:59 UTC续训250k验证：** agent实查PID176730活、W&B hdsosznv running/latest255720、LR1e-5；val@250000 FID19.301525，best仍100k/18.734051；R1/2/3=.358333/.493750/.558333、Matching3.850356、Div7.626090（GT7.751709）、MM.316754（同100×11/10pairs）、val loss7.697487，valid/body/root/masked acc=.136759/.118946/.154572/.068677。FID较240k回落但未刷新best，MM近似持平；nonfinite grad/weights=0、磁盘16GiB、无异常栈。原29466继续固定300k收尾，保持验证集选模，正式test未启动，≥30分钟监控。
- **10-02约06:56 UTC续训240k验证：** agent实查PID176730活、W&B hdsosznv running/latest244600、LR1e-5；val@240000 FID20.748279，best仍100k/18.734051；R1/2/3=.362500/.483333/.550000、Matching3.855331、Div7.676972（GT7.751709）、MM.317725（同100×11/10pairs）、val loss7.624835，valid/body/root/masked acc=.133695/.114381/.153008/.066248。FID再次回升、未刷新best，末态不能替代100k候选；nonfinite grad/weights=0、磁盘16GiB、日志无异常栈。原29466继续既定300k收尾，正式test未启动，≥30分钟监控。
- **10-02约05:53 UTC续训230k验证：** agent实查PID176730活、W&B hdsosznv running/latest233460、LR1e-5；val@230000 FID19.673849，best仍100k/18.734051；R1/2/3=.356250/.489583/.583333、Matching3.850047、Div7.666222（GT7.751709）、MM.333647（同100×11/10pairs）、val loss7.669530，valid/body/root/masked acc=.138312/.118383/.158241/.067989。FID回升、MM较220k下降，未刷新best；nonfinite grad/weights=0、磁盘16GiB、日志仅已知警告。原29466继续固定300k与既定选模，正式test未启动，≥30分钟监控。
- **10-02约04:50 UTC续训220k验证：** agent实查PID176730活、W&B hdsosznv running/latest222400、LR1e-5；val@220000 FID19.181627，best仍100k/18.734051；R1/2/3=.362500/.475000/.552083、Matching3.848768、Div7.674816（GT7.751709）、MM.388463（同100×11/10pairs）、val loss7.640284，valid/body/root/masked acc=.138063/.118155/.157970/.067805。FID/MM较210k改善但未刷新best；nonfinite grad/weights=0、磁盘16GiB、无异常栈。原29466继续固定300k及既定选模，正式test未启动，≥30分钟监控。
- **10-02约03:46 UTC续训210k验证：** agent实查PID176730活、W&B hdsosznv running/latest211380、LR1e-5；val@210000 FID19.309681，best仍100k/18.734051；R1/2/3=.347917/.470833/.560417、Matching3.848654、Div7.660457（GT7.751709）、MM.321619（同100×11/10pairs）、val loss7.672739，valid/body/root/masked acc=.138931/.117284/.160578/.068261。FID/MM较200k略改善，仍未刷新best；nonfinite grad/weights=0、磁盘16GiB、无异常栈。原29466继续固定300k及最低val FID选模，正式test未启动，≥30分钟监控。
- **10-02约02:42 UTC续训200k验证：** agent实查PID176730活、W&B hdsosznv running/latest200300、LR1e-5；val@200000 FID19.493065，best仍100k/18.734051；R1/2/3=.352083/.483333/.547917、Matching3.851726、Div7.639468（GT7.751709）、MM.288934（同100×11/10pairs）、val loss7.640986，valid/body/root/masked acc=.137741/.118118/.157364/.067009。FID未刷新best、MM再降；nonfinite grad/weights=0、磁盘余16GiB、无异常栈。原29466继续剩余固定预算至300k，保持已保存best，不拿末态替代验证集选模；正式test未启动，≥30分钟监控。
- **10-02约02:12 UTC续训190k验证：** agent实查PID176730活、W&B hdsosznv running/latest196350、LR1e-5；val@190000 FID19.099938，best仍100k/18.734051；R1/2/3=.368750/.493750/.566667、Matching3.849276、Div7.704075（GT7.751709）、MM.365982（同100×11/10pairs）、val loss7.705803，valid/body/root/masked acc=.137886/.116219/.159553/.065438。FID较180k小幅下降、MM回升但未刷新best；nonfinite grad/weights=0、磁盘余16GiB、无异常栈。原29466继续固定300k，正式test未启动，≥30分钟监控。
- **10-02约01:09 UTC续训180k验证：** agent实查PID176730活、W&B hdsosznv running/latest185350、LR1e-5；val@180000 FID19.135883，best仍100k/18.734051；R1/2/3=.362500/.502083/.566667、Matching3.848698、Div7.668920（GT7.751709）、MM.300395（同100×11/10pairs）、val loss7.680228，valid/body/root/masked acc=.138853/.114705/.163001/.065588。FID较170k改善但未刷新best，MM继续下降，不把检索提升当整体质量提升；nonfinite grad/weights=0、磁盘余16GiB、日志无异常栈。原29466继续300k固定预算，正式test未启动，≥30分钟监控。
- **10-02约00:06 UTC续训170k验证：** agent实查PID176730活、W&B hdsosznv running/latest174340、LR1e-5；val@170000 FID19.410439，best仍100k/18.734051；R1/2/3=.360417/.477083/.581250、Matching3.850744、Div7.637316（GT7.751709）、MM.347506（同100×11/10pairs）、val loss7.655218，valid/body/root/masked acc=.140207/.116572/.163843/.065193。降率后FID/MM相对160k近似持平，未刷新best；nonfinite grad/weights=0、磁盘16GiB、无异常栈。原29466继续既定300k，正式test未启动，≥30分钟监控。
- **10-01约23:03 UTC续训160k验证（本地10-02）：** agent实查PID176730活、W&B hdsosznv running/latest163380、实际LR1e-5；val@160000 FID19.325268，best仍100k/18.734051；R1/2/3=.364583/.483333/.570833、Matching3.849412、Div7.650041（GT7.751709）、MM.343261（同100×11/10pairs）、val loss7.591332，valid/body/root/masked acc=.142932/.117030/.168833/.065984。降率后首个完整10k区间未刷新best，MM继续偏低，不声称降率已解决泛化；nonfinite grad/weights=0、磁盘余16GiB、无异常栈。原29466继续固定300k，训练内val非正式test，≥30分钟监控。
- **10-01约22:00 UTC续训150k验证与降率确认：** agent实查PID176730活、W&B hdsosznv running/latest152360；实际train LR=1e-5@152360（148460仍2e-4），原150k×.05调度已生效。val@150000 FID19.343642，best仍100k/18.734051；R1/2/3=.347917/.470833/.572917、Matching3.847837、Div7.545980（GT7.751709）、MM.371456（同100×11/10pairs）、val loss7.494313，valid/body/root/masked acc=.148772/.119109/.178434/.066821。此验证是降率边界，不称已体现降率后训练效果；nonfinite grad/weights=0、磁盘16GiB、日志无异常栈。原29466继续固定300k，后续160k起观察降率后轨迹，不改参数/正式test仍未启动。
- **10-01约20:57 UTC续训140k验证：** agent实查PID176730活、W&B hdsosznv running/latest141350；val@140000 FID20.558655，best仍100k/18.734051；R1/2/3=.331250/.470833/.537500、Matching3.854497、Div7.643063（GT7.751709）、MM.383349（同100×11/10pairs）、val loss7.321104，valid/body/root/masked acc=.148448/.120655/.176241/.068578。FID回升、MM较130k近似持平，未刷新best；nonfinite grad/weights=0、磁盘16GiB，日志仅已知警告。原29466继续，下一关注原定150k学习率×.05及后续验证，不提前改参；正式test未启动。
- **10-01约19:54 UTC续训130k验证：** agent实查PID176730活、W&B hdsosznv running/latest130200；val@130000 FID19.936190，best仍100k/18.734051；R1/2/3=.350000/.487500/.570833、Matching3.849769、Div7.708548（GT7.751709）、MM.386787（同100×11/10pairs）、val loss7.091151，valid/body/root/masked acc=.153156/.124187/.182125/.068867。检索/Matching改善但FID未刷新best、MM较120k下降，不称全面改善；nonfinite grad/weights=0、磁盘余16GiB。原29466继续固定300k/150k降率，正式test未启动，≥30分钟监控。
- **10-01约19:24 UTC续训120k验证：** agent实查PID176730活、W&B hdsosznv running/latest126200；val@120000 FID19.666681，best仍100k/18.734051；R1/2/3=.337500/.462500/.539583、Matching3.854421、Div7.613599（GT7.751709）、MM.413583（同100×11/10pairs）、val loss7.178305，valid/body/root/masked acc=.154942/.121577/.188307/.063338。MM较110k回升但FID未刷新best；nonfinite grad/weights=0，磁盘余16GiB，无异常栈。原29466继续固定300k及150k降率，训练内val非正式test，≥30分钟监控。
- **10-01约18:20 UTC续训110k验证：** agent实查PID176730活、W&B hdsosznv running/latest115020；val@110000 FID19.303998，best仍100k/18.734051；R1/2/3=.333333/.464583/.537500、Matching3.851299、Div7.620108（GT7.751709）、MM.342015（同100×11/10pairs）、val loss7.087099，valid/body/root/masked acc=.155923/.121687/.190158/.066657。MM较100k .419030明显下降，FID未刷新best，不宣称整体改善；nonfinite grad/weights=0、磁盘16GiB、无异常栈。原29466继续固定预算与原150k降率计划，正式test未启动，≥30分钟监控。
- **10-01约17:19 UTC续训100k验证刷新best：** agent实查PID176730活、W&B hdsosznv running/latest103960；val@100000 FID18.734051，新best100k；R1/2/3=.341667/.460417/.545833、Matching3.854473、Div7.655377（GT7.751709）、MM.419030（同100×11/10pairs）、val loss7.119988，valid/body/root/masked acc=.156951/.114356/.199546/.065486。FID改善但MM较90k下降、token loss继续升高，不能只报FID掩盖取舍。nonfinite grad/weights=0、磁盘余16GiB；原29466继续300k固定预算，源码/配置不变，正式test未启动，≥30分钟监控。
- **10-01约16:16 UTC续训90k验证：** agent实查PID176730活、W&B hdsosznv running/latest92900；val@90000 FID20.032278，best仍60k/19.640771；R1/2/3=.335417/.472917/.537500、Matching3.858401、Div7.686671（GT7.751709）、MM.445322（同100×11/10pairs）、val loss6.877112，valid/body/root/masked acc=.162753/.124514/.200992/.069158。近期FID约20附近波动、MM约.45，未刷新best，loss继续升高；nonfinite grad/weights=0，磁盘16GiB，日志无异常栈。原29466继续固定训练，不改参数/重启；训练内val非正式test，后续≥30分钟。
- **10-01约15:15 UTC远端状态自行恢复、80k验证已出：** ≥600s异常复核确认W&B hdsosznv回running/latest81850、trainer176730活，未干预/重启训练或修改远端状态。val@80000 FID20.482711，best仍60k/19.640771；R1/2/3=.354167/.489583/.568750、Matching3.852289、Div7.629570（GT7.751709）、MM.453689（同100×11/10pairs）、val loss6.827942，valid/body/root/masked acc=.164928/.119408/.210448/.065393。nonfinite grad/weights=0；前次crashed是短暂远端状态，不等于训练退出，具体触发机制未确认。原29466继续，恢复≥1800s监控。
- **10-01约15:03 UTC远端W&B状态异常待复核：** 既定监控报hdsosznv远端`crashed`、latest79980、无80k val，但trainer176730仍活。root实际收29466仍active，ps176730 Rl、wandb-core176757 Ssl、GPU0利用率97%/17603MiB；本地W&B文件15:02UTC仍更新，internal仅启动INFO、训练log无异常栈，latest.pth修改14:48UTC。不能据远端state判训练退出或重启，暂视计算与远端状态不一致、原因未定；保留原会话，安排≥600s异常复核80k验证/同步是否恢复，之后回≥1800s。不手动更改W&B状态、不重新启动，磁盘仍16GiB。
- **10-01约14:31 UTC续训70k验证：** agent实查trainer176730活、W&B hdsosznv running/latest74900；val@70000 FID19.876493，best仍60k/19.640771；R1/2/3=.337500/.450000/.531250、Matching3.857727、Div7.673244（GT7.751709）、MM.452811（100×11/10pairs）、val loss6.604572，valid/body/root/masked acc=.170830/.118952/.222707/.065041。FID较60k轻微回升，MM近似稳定，loss继续上升；不据单节点改参。nonfinite grad/weights=0，磁盘余16GiB；原29466继续固定300k预算，正式test未启动，≥30分钟监控。
- **10-01约13:28 UTC续训60k验证：** 低频agent实查trainer176730存活、同W&B hdsosznv running/latest63780；val@60000 FID19.640771，新best60k（50k20.216120），R1/2/3=.312500/.456250/.541667、Matching3.854398、Div7.640451（GT7.751709）、MM.447953（100×11/10pairs）、val loss6.459633，valid/body/root/masked acc=.185365/.135766/.234964/.075014。FID继续降低、MM相对50k近似持平，但val token loss仍上升且检索有波动，不称泛化问题已消失。nonfinite grad/weights=0，磁盘余16GiB；原29466继续固定300k训练与≥30分钟监控，正式评分未启动。
- **10-01约12:27 UTC续训已越过50k：** 低频agent实查主trainer176730存活，原W&B hdsosznv running/latest optimizer52460，nonfinite grad/weights=0；磁盘余16GiB，未触发5GiB警戒，未删除文件。原manifest仍保留50k记录，不据此重启；主29466实查仍运行，首续训验证60k待出，best仍50k/FID20.216120。此后≥30分钟只报新val/异常/终态。
- **10-01 12:16 UTC dropout03_v2原run续训已启动：** 主session29466/GPU0，日志`/home/wenning/HHC_interhuman_runs/mgt_interhuman_dropout03_v2_continue_20261001.log`；原trainer原登记参数仅resume latest/终点300000，恢复50k优化器/调度器/RNG/同W&B hdsosznv，不重初始化、不改配置/源码。保留原150k×.05降率及最低val FID选模，先只执行训练，不自动串正式评分（磁盘余16GiB，正式20rep容量未验）。启动前GPU0 39MiB，50k manifest及resume路径确认；首真实>50k step待≥600s监控，之后≥1800s，磁盘<5GiB节点告警但不自行删除。注意resume期间原trainer可能保留pilot manifest直至终态，以实际PID/step判运行，不据旧manifest重启。
- **dropout03_v2 50k源归档验收：** 原run `source_snapshot_pilot_v2_20261001.tar.gz`，14 sources+50k pilot manifest共15项预核/读回SHA全匹配，create/list/readback exit0，50833bytes，SHAdda5eb8f837609198bdeef49819d0ce319d66a31ba6853974857750c1377d181。保留pilot审计所绑定旧manifest，不把该归档称300k完成。
- **10-01 12:14 UTC dropout03_v2 50k终态与下一决策：** 主75242明确exit0，manifest pilot_decision_ready/current50000，best50k FID20.2161204072、ckpt SHA6b412b4e1ee7138fa8eb90ea0e5f768e62006aa40bb2c12c5df5774b2e1fb189；主60486 pilot W&B审计exit0/7 checks全true，主51913 selected绑定exit0。50k R1/2/3=.322917/.466667/.560417、Matching3.856360、Div7.666643（GT7.751709）、MM.446163、val loss6.310857；同v1选定50k FID21.170514/MM.472920/loss6.313080。FID约改善4.5%，MM约下降5.7%，loss基本不变，不能称过拟合已解决或全面优于v1。停止dropout搜索，按已登记最低val FID选择v2作为第③项完整预算候选，拟原run不重初始化续至300k（含原150k降率），全程保留最佳val checkpoint；不把单个pilot当第③项完成。低成本agent先归档v2 50k source+manifest，完成后才续训，当前尚未启动续训。NVMe余16GiB够训练checkpoint峰值但正式20rep容量未放行，下一续训仅训练、不自动接正式评分；不删除旧实验、不用test选模。
- **10-01约11:43 UTC dropout03_v2 40k验证：** agent实查trainer169928存活、W&B running/latest46900；current/best40k FID21.273116，R1/2/3=.306250/.456250/.541667、Matching3.862246、Diversity7.619466（GT7.751709）、MM.492277（同100×11/10pairs）、val loss6.107007。与同40k v1 FID21.264795几乎相同、MM低于v1 .543428，val loss也未优于v1 6.047837，不支持泛化已修复。nonfinite grad/weights=0；原75242待50k终态，不自动续训、不追加参数点，训练内val非正式test。
- **10-01约10:42 UTC dropout03_v2 30k验证：** 低频agent实查PID169928存活、W&B hdsosznv running/latest35550；current30k FID22.291048，best仍20k/22.024733；R1/2/3=.306250/.441667/.520833、Matching3.873469、Diversity7.683964（GT7.751709）、MM.584181（同100×11/10pairs）、val loss5.762236，valid/body/root/masked acc=.214891/.146052/.283730/.083731。同30k v1 FID22.917066稍差，但v1 MM.618646/Matching3.865775/检索更好；此前20k的MM优势未保持，新run loss仍持续升高，不能声称泛化问题已解决。nonfinite grad/weights=0；继续原75242至登记50k决策，不扩大sweep，训练内val非正式test。
- **10-01约09:41 UTC dropout03_v2 20k验证：** 低频agent实查PID169928存活、W&B hdsosznv running/latest24160；current/best20k FID22.024733，R1/2/3=.287500/.435417/.510417、Matching3.882470、Diversity7.628753（GT7.751709）、MM.771665（100×11/10pairs）、val objective loss5.238704。较自身10k FID改善，但对同20k v1 FID21.625773/Matching3.870628/R=.327083/.447917/.529167仍略差；MM优于v1 .710235、loss低于v1 5.461804，呈取舍而非全面改善。nonfinite grad/weights=0；继续预登记50k原75242，≥30分钟监控，训练内val不是正式test，MPJPE未评。
- **10-01 dropout03_v2首个10k验证：** 低频agent实查PID169928存活、W&B hdsosznv running/latest13050；current/best10k FID25.179602，R1/2/3=.283333/.385417/.454167、Matching3.906055、Diversity7.512132（GT7.751709）、MM.914483（100×11/10pairs）、val objective loss4.607866。同10k v1为FID21.277377、R=.283333/.431250/.493750、Matching3.883060、Div7.666836、MM.926542、loss4.728200；dropout03虽然token loss略低，但生成FID/检索并未改善，不能提前宣布正则化有效。维持预登记50k同预算对照，不追加参数点；nonfinite grad/weights=0，主75242未终态，后续≥30分钟。训练内验证，非正式20rep，MPJPE未评。
- **10-01 dropout03_v2首个≥10分钟节点已确认：** 低频agent实查trainer169928存活、W&B hdsosznv running/latest optimizer2300，nonfinite grad/weights=0、gradient L2约1.6083，GPU0 15713MiB，无异常栈。首val仍在10k，FID/R/Matching/Diversity/MM尚未评；主75242继续，后续≥1800秒仅新val/异常/终态通知，保持源码与配置冻结。
- **10-01 07:28 UTC MGT dropout单变量对照已启动：** 独立`hhc_interhuman_mgt_dropout03_v2`，主session75242/GPU0，日志`/home/wenning/HHC_interhuman_runs/mgt_interhuman_dropout03_v2_20261001.log`；新登记config仅model.dropout .1→.3（附实验说明），模型尺寸/初始seed/data/loss/optimizer/sampling/50k暂停预算均不变，仍冻结v3 VQ。复用trainer加`--contract`并绑定实际配置SHA，launcher以`HHC_INTERHUMAN_MGT_VARIANT=dropout03_v2`隔离run与正式评分目录；baseline默认评分路径保留。36文件127检查48867 exit0及shell语法通过；初次9075 exit1捕获baseline评分路径改变，恢复原路径后全量重过，未改测试掩盖问题。新增断言仅dropout差异、同seed小模型初始state逐值相同且dropout概率实际传递。启动前GPU0 39MiB、NVMe余22GiB足够本pilot检查点峰值，正式评分容量另查；首次≥600s后≥1800s由低成本agent监控，尚未把启动称真实step/质量验收。运行源码/配置再次冻结。
- **MGT v1源快照已验并保留边界：** `source_snapshot_pilot_v1_20261001.tar.gz`在原run目录，14 sources+pilot manifest共15项全部预核/读回SHA一致、exit0、50407bytes、tar SHA187867c897bc3983a4638a8d761e1c674493c8007968bec07e32695c26a7029e；归档后才修改MGT trainer配置入口。旧v1 manifest/checkpoints/配置/评分未改，旧v1严格源码重载需对应归档版本，不绕过校验；VQ绑定源完全未动。v1为50k pilot而非300k完成。
- **10-01 MGT train/val差距实测诊断完成：** 新独立`tools/diagnose_interhuman_mgt_modes.py`复用严格loader/数据/编码/mask/loss；主10694明确exit0，50k ckpt同每split64样本、8微批、同批token/mask配对比较train/eval，无optimizer，完整模型含VQ前后state SHA一致。train样本 eval/train模式loss1.041171/1.144945、masked acc.582069/.554727；val样本6.128066/6.043848、masked acc.072324/.075206。eval开关无法解释差距，当前更支持泛化退化/过拟合而非数值或模式切换故障；这只是64例诊断，不代替480val或正式test。JSON在当前MGT run `mode_gap_diagnostic.json`。低成本agent单次3000行history提取每节点前1000步（60记录）：train loss10k→50k=3.163726/2.306016/1.794989/1.502451/1.264556，对应val=4.728200/5.461804/5.848290/6.047837/6.313080；train mask比例.638–.644 vs固定val .653336，仅约1个百分点差，不足以单独解释巨大gap。数据路径源码均复用同adapter normalize与在线token/mask/loss。50k暂不续，下一步选择单变量泛化改进候选并保留此基线，不能归因全部FID差距或用test选参。
- **10-01约07:19 UTC MGT 50k pilot正常终态：** root收原5003明确exit0；manifest `pilot_decision_ready/current50000`（非full300k完成），selected50k/FID21.1705144562、checkpoint SHA1e28cdc45ff181a6ea86afaf7145a6d0c23aa9a1fce5fbffc456a4a28c92d758。50k val R1/2/3=.325/.458333/.547917、Matching3.856653、Diversity7.548625（GT7.751709）、MM.472920；固定val objective loss6.313080、valid/body/root/masked accuracy=.171462/.124006/.218918/.065556，最新nonfinite grad/weights=0。主53398 pilot W&B audit exit0/7项全true，主52589 `--allow-pilot` selected绑定校验exit0；W&B finished，旧周期监控已停止。FID相对10k仅小幅改善、MM接近减半且val token loss连续升高，暂不盲目续300k；下一步只针对train/val路径与泛化差距诊断，先由低成本agent单次提取5段train统计及对应val，主检查实现/决定配对诊断。保留全部现有产物，不test选模、不声称stage③完成。
- **10-01约06:48 UTC MGT 40k验证：** 低频agent实查trainer161544存活、W&B running/latest47220；val@40000 FID21.264795，新best40k（10k为21.277377，差距很小），R1/2/3=.333333/.466667/.537500、Matching3.862029、Diversity7.548853（GT7.751709）、MM.543428（同100×11/10pairs）。FID未继续恶化但MM继续下降，不凭极小FID改善称整体成功。nonfinite grad/weights=0；原5003未终态，等待50k预登记暂停及完整轨迹决策，不自动续训。训练内val，非正式20rep。
- **10-01约05:47 UTC MGT 30k验证：** 低频agent实查trainer161544存活、W&B1x34pp2c running/latest35920；val@30000 FID22.917066，best仍10k/21.277377；R1/2/3=.320833/.456250/.545833、Matching3.865775、Diversity7.670763（GT7.751709）、MM.618646（同100×11/10pairs）。FID连续两个节点变差、MM从.9265→.7102→.6186，Matching则改善；这提示质量/多样性风险，不能以检索提升掩盖，也不能凭这些指标单独归因为VQ或MGT塌缩。暂保持已登记50k pilot，不中途重启/改参，到暂停点联合全部轨迹决定是否续训/诊断。nonfinite grad/weights=0，主5003仍运行；训练内val，非正式20rep验收。
- **10-01约04:46 UTC MGT 20k验证：** 低频agent实查trainer161544存活、W&B1x34pp2c running/latest24680；val@20000 FID21.625773，历史最佳仍10k/21.277377。R1/2/3=.327083/.447917/.529167、Matching3.870628、Diversity7.586453（GT7.751709）、MM.710235（同val100×11/10pairs）。相对10k检索/Matching改善，但FID未降且MM下降，不能宣称生成整体改善；保持登记50k决策点，暂不追加采样/模型改动。最新nonfinite grad/weights=0；原5003未终态，继续≥30分钟监控。训练内验证不等于正式20rep评分，MPJPE未评。
- **10-01约03:45 UTC MGT首个10k验证已出：** 低频agent实查trainer161544存活，W&B1x34pp2c running/latest13320；val@10000 InterCLIP FID21.277377（当前唯一/最佳step10000），R1/2/3=.283333/.431250/.493750、Matching3.883060、Diversity7.666836（GT7.751709）、MM.926542（val100 prompts×11 repeats/10 pairs）。此为训练内固定480val，不是20rep正式test；未产生MPJPE，不填0也不混同VQ重构成绩。最新nonfinite grad/weights=0、GPU0 16749MiB；继续冻结配置到预登记50k决策点，不因单个早期节点改采样或调参。原主5003仍负责终态，后续≥30分钟监控。
- **10-01正式MGT评测链与容量预检：** 主只读核对common50 finalizer→val-FID选模校验→main/MM评分→20rep汇总，CPU定向14检查（3文件）主60562明确exit0；测试不等于实际生成验收。低成本agent核验冻结请求manifest/文件SHA及evaluator资产存在：20rep，每rep main1056、MM100×30（10 pairs），最终输出FID/Matching/R1–3/Diversity及GT/MM的均值与95%CI；训练中100×11 MM仅诊断，不冒充正式协议。静态motion有效数据合计16,800,152,640B=15.65GiB，固定300帧未压缩47.51GiB；实际LZF+零padding大小尚未实测。当前/home余24GiB、HDD余2.1GiB，正式输出沿用/home；50k决策与正式评分启动前重查磁盘余量、计入剩余checkpoint，不凭假设压缩比放行。未改运行源码/配置，未删产物，原5003继续，质量仍待实际val。
- **10-01约02:43 UTC MGT首个实际训练节点已确认：** 低频agent实查trainer161544存活，W&B run1x34pp2c running/latest optimizer2300，GPU0占用15713/24564MiB，无异常栈；nonfinite gradients/weights均0，gradient L2约1.7921、clip hit rate1（初期裁剪不等于失败）。尚无val event，FID/R/Matching/Diversity/MM均未评，首验证在10k；不报虚构质量指标。latest正写入未读，主5003仍负责原50k pilot。后续监控≥30分钟，只在新val/异常/终态汇报。
- **10-01 02:32 UTC InterHuman MGT首次正式pilot已启动：** 新run`hhc_interhuman_mgt_pilot_v1`，主session5003/GPU0，日志`/home/wenning/HHC_interhuman_runs/mgt_interhuman_pilot_v1_20261001.log`；绑定v3 selected29000（SHA74c608ce…c2cd3a1）及完整30k audit，恢复的VQ源17/17严格匹配。沿用2通道body/root、75位置、229,367,296可训练MGT参数、无shape/contact、seed114514、AdamW2e-4/150k降率、固定full300k，50k暂停决策；每10k同480val+100×11 MM诊断，最终test后置。训练首次启动前据资源测量将登记config/CLI/launcher同步为physical64×accum1（effective64不变），不是已训练run中途改batch；35文件125测试60024 exit0。首10分钟低频确认实际step，之后≥30分钟监控，原句柄由root收；仅启动，首step与生成质量尚待确认，严禁修改已运行shell或绑定source。
- **MGT真实GPU预检验收与物理batch选择：** agent52736明确exit0、GPU1结束释放17MiB；v3严格重载、8条真实Dataset/collate在线token[8,75,2]、实际缓存CLIP global[8,512]/word[8,77,512]与InterCLIP同驻；正式MGT BF16 306/306梯度张量有限，VQ完整state逐值不变且无梯度。两份FP32 Adam moments已分配但未optimizer.step/W&B/checkpoint；重复真实8条仅资源对照：8×8累积第二轮0.2625s、allocated/reserved6.205/6.297GiB；physical64第二轮0.2183s、14.176/14.713GiB，无OOM。选择64×1减少约17%实测前反向时间；这不是64条独立数据质量实验，不承诺bitwise梯度/整训练吞吐等价，optimizer.step/数据加载等开销不在该时间内。NVMe余27GB，足以容纳既定latest/best/last及原子写峰值，未删除旧产物。
- **MGT真实资源预检已执行：** 子agent原session52736，GPU1启动前17MiB；严格v3 selected29000 + 实际8条数据 + 缓存TextCLIP/InterCLIP + 正式MGT前反向及Adam moments驻留。比较8×8累积与64×1物理batch的wall/peak只作资源选择，不optimizer.step、不写训练W&B/检查点、不宣称梯度bitwise等价；主等待确切exit_code和结果后才定正式batch与启动。此时无正式MGT run。
- **10-01约02:24 UTC恢复v3严格加载、MGT绑定已切换：** 在v4快照18项读回通过且所有旧训练终态后，仅撤回自己为v4加入的trainer路由/倍率校验及shell variant分支，恢复v3 trainer逐字节SHA；主91289明确exit0，17份源哈希全部匹配、selected29000严格重载和W&B审计绑定通过（未跳过校验）。MGT launcher现明确指向`hhc_interhuman_vq_bodyroot_ema_v3`，仍75×2/4倍压缩；v4配置/检查点/归档保留，活动入口不再续其候选。新增launcher绑定断言、v4测试保留配置/模型形状检查并注明归档状态，35文件125检查主18963 exit0，两个shell语法通过。实际GPU1冻结VQ+真实CLIP+MGT前反向/显存与8×8对64×1吞吐预检交低成本agent执行；尚未启动MGT，物理batch是否调整需先看实测，保持有效batch64。
- **10-01 v4 pilot源快照已验：** `vq_interhuman/hhc_interhuman_vq_bodyroot_dense_v4/source_snapshot_pilot_v4_20261001.tar.gz`，17 source+pilot manifest共18项预核及读回SHA全部通过，打包/核验exit0，40640bytes，SHAab8d64753c9095527ed0f76fbd127fe48633c36be0852298b0399a946998fe67。仍pilot10000，不称30k完成。监控已停止，后续恢复v3匹配源码不影响此独立归档。
- **10-01约02:20 UTC v4 10k收尾与不续训决定：** 主34542明确exit0，manifest pilot_decision_ready/10000（非30k complete），best10000/FID9.895569987；主64296 pilot审计明确exit0、7项checks全true/覆盖10000。root从v3/v4两份审计逐点读回：同10k FID10.083237/9.895570，但相同10k预算内按预登记最低FID选模v3 best9k/9.639050优于v4 best10k/9.895570；同10k MPJPE185.094/188.809mm、MAE.368219/.379312。v4并非所有step更差（1/2/5/7/8/10k FID更低），但未改善选模FID且末态几何/特征误差略差，不支持双倍token预算，按登记停止v4，不续30k、不追加VQ sweep。v4末态R1/2/3=.333333/.452083/.54375、Matching3.839829、Diversity7.570561。旧v3完整30k仍为当前最优两部分HHC基线；下一步进入其MGT端到端pilot准备，不把它称官方VQ复现成功，也不把追平不同结构官方VQ变成无限追加结构实验的自设门槛。先完成v4源归档并恢复v3匹配源码、严格重载及MGT绑定/真实资源复核，未启动MGT。
- **10-01约02:08 UTC v4首次低频节点：** agent实查launcher/trainer存活，W&B kuhamhwz running、telemetry4750、稳定checkpoint/val4000。训练内FID14.778482（best4000/同值）、MPJPE213.444mm、MAE.408657、R1/2/3=.28750/.42917/.51458、Matching3.860857、Diversity7.452688；同4k v3为FID14.654793/MPJPE202.969mm/MAE.393255，v4此时未显示重构优势，不能因token变密预设改进。FP32 EMA有限、count_min .739700/.790148、max每码norm240.293/142.236；最新训练usage .75977/.96484、perplexity286.945/317.519，非有限梯度/权重0。保持登记10k决策点，不中途改配置/追加参数，主34542仍运行。
- **10-01 02:03 UTC MGT压缩率接口只读清单已核对：** 低成本agent盘点、root复读实际代码：`utils/interhuman_mgt_batch.py:30`固定75槽/floor(length/4)/最大74/contact占位75；`utils/interhuman_mgt_training.py:40`形状75；`models/interhuman_transformer.py:177`采样容量4*(block_size-1)与ceil(length/4)写死。若v4最终选中需同步为由VQ因子2推导150槽/最多149有效token，并登记两个MGT配置及覆盖4/2两种测试；END/PAD ID不变，预留位置仍必要。`utils/interhuman_mgt_evaluation.py`生产路径已显式传VQ动态因子，位置embedding按block_size构建。此次未改MGT或训练配置，不提前放行；v4原34542已复核运行中，继续≥600秒监控。
- **10-01 01:57 UTC v4原生2倍时间压缩pilot已启动：** 新run`hhc_interhuman_vq_bodyroot_dense_v4`，主session34542/GPU0，日志`/home/wenning/HHC_interhuman_runs/vq_bodyroot_dense_v4_20261001.log`；同seed123/batch64/64帧/FP32量化器/loss/data/固定30k cosine，10k暂停。唯一架构设置down_t2→1，编码64帧由16→32个位置×2通道；原生实现随之减少采样/残差块，参数22,337,548→13,154,828、感受野变化，不声称纯bitrate隔离。新config`configs/interhuman_vq_pilot_dense_v4.json`，训练入口根据已登记down_t选择配置、校验压缩倍率，shell显式`HHC_INTERHUMAN_VQ_VARIANT=dense_v4`隔离run。35文件125检查主64670 exit0；真实batch GPU预检67998 exit0（64×64×524→64×32×2，有限loss/grad、FP32 EMA、peak249049600B，不含optimizer/evaluator）；首次41528 exit1因诊断误用不存在的.encode接口，修为实际forward(type=encode)后完整重跑通过，未改变模型解决报错。NVMe余29GB。仅已启动，首step待≥600s监控；主保留句柄，不将启动当验收。
- **v4版本边界：** 已完成v3全部评分且源快照读回后才修改trainer的配置路由/倍率校验；旧v3绑定的trainer SHA因此与当前版本不同，旧检查点、manifest和评分未改，严格重载会拒绝不匹配源码，复演需对应已归档源版本，不静默跳过校验。所有模型/损失源码未改。若v4最终被选中，MGT现有75位置预算需按2倍时间压缩重新验证并登记，不能直接套旧4倍压缩MGT配置；目前不修改/启动MGT。
- **10-01约01:53 UTC剩余误差分解完成：** 新独立`tools/diagnose_interhuman_root_pose.py`复用严格loader/官方H5校验/valid-frame输入检查；共同平移、反向双人位移、局部关节误差、padding与非有限拒绝合成检查exit0。真实同480val主8094明确exit0，45725pair-frames/state未变，未对模型训练；HHC/官方(mm)：raw MPJPE156.765/109.926、root108.038/89.002、双人root相对向量145.158/123.246、root中心80.692/66.650、去共同中心后全身135.694/98.805、逐人去root平移后body115.502/80.845。raw值和分母与两份冻结几何报告逐值复核通过。GT平移对齐仅诊断、不作部署成绩，各范数不能相加为贡献率；结果排除“仅整体坐标偏移”的解释，局部姿态仍有显著差距。下一项选择仍保持body/root两部分、仅将时间压缩4→2的独立受控pilot；尚未注册/改训练/启动，需先验参数与资源，不同时改变loss或shape。
- **10-01 v3终态源码快照已验：** agent完成`vq_interhuman/hhc_interhuman_vq_bodyroot_ema_v3/source_snapshot_completed_v3_20261001.tar.gz`，18项含17source+终态manifest，逐项SHA通过，40515bytes，SHA054727f3a1a00bbf77b3191fe5fdb576a4b3fcc61668f07ae7312260a35beadb。首次tar因manifest路径错误exit2，修正归档路径后的打包与逐项读回均exit0；源文件/manifest/检查点未改。
- **10-01约01:48 UTC v3全流程完成，改善但仍有明显质量差距：** 主94307和9165均明确exit0；manifest complete/final30000，selected29000（训练seed123、固定480val评分seed20260130），checkpoint SHA74c608ce5c507668026de07d24e740492cd1538ebb94d386ce3e28439c2cd3a1；最终manifest SHA8f1b22e9d8b4de205c39ca3ba9f41ea949ded119b3dde0efa4c52b1d9c330f7c。W&B审计pass/7项checks全true，17source SHA仍一致；同request/evaluator独立重载确认InterCLIP FID6.623693（非joint FID）、MPJPE156.764992mm、MAE.342476、R1/2/3=.339583/.477083/.5625、Matching3.826673、Diversity7.615552（GT7.751709）。补充几何RMSE112.619120mm、normalized feature RMSE.649496、full/body非root/root误差156.764985/159.085324/108.037859mm，GT接触帧脚滑5.415380mm/frame（共同GT4.258247）、10cm几何contact F1 .5386725（TP2455/FP1238/FN2967）。Hands MPJPE/FID不适用；Full/Body joint FID待评、无匹配joint evaluator；语义mesh F1未评。官方同协议FID.985952/MPJPE109.925614mm，仍明显更好，不能把量化器修复成功称官方质量复现。v2全部重建主指标均改善，仍保留v2失败对照。子agent归档v3终态源快照，停止旧监控；MGT未启动，下一步先分解剩余root/局部姿态误差以决定表示/压缩调整，不重启第①项。
- **10-01约01:35 UTC v3训练内25k节点：** agent实查trainer149195存活、W&B c4fwrj21 running，val至25000/telemetry25080；FID6.936626（best25000/同值）、MPJPE161.200mm、MAE.344886、R1/2/3=.35000/.50625/.57500、Matching3.828323、Diversity7.619122。最新batch body/root perplexity244.171/428.991、使用码数468/624；最新行非有限字段空，最近有值24980为0/0，不把空值当0。仍训练内验证，原94307与9165未终态，剩余至30k及独立完整评分继续。
- **10-01约01:25 UTC v3训练内20k节点：** agent实查trainer149195存活、step20k checkpoint/W&B同run running，latest telemetry20160；val@20000 FID7.326702（best20000/同值）、MPJPE166.180mm、MAE.349701、R1/2/3=.3375/.475/.57708、Matching3.830791、Diversity7.569805，非有限梯度/权重0。EMA sum/count仍FP32有限、count_min .754719/.780274，max每码norm246.496/115.639。质量继续改善、无旧式数值漂移；原94307训练至30k及9165最终几何评分继续，未作最终质量验收。
- **10-01 01:17 UTC同16k对照验证修复持续有效：** 子agent39762明确exit0，root实际读回v3 probe complete/step16000/state未变，同480val/91450person-frames/每通道11616token。v2→v3 body active151→636、perplexity57.806→301.375；root76→1006、35.357→709.267。MPJPE(mm)量化/body连续/root连续/全连续：v2=225.301/223.192/218.950/246.157，v3=176.271/175.064/161.699/180.031。v3在4k/8k/16k未重现旧固定val码本收缩，支持FP32 EMA+有限窗口reset组合修复确实改善利用率和质量；两项机制未单独消融，不分别归因，也不称全部官方差距已解决。诊断到此停止追加节点，保持30k训练及原94307/9165收尾；GPU1已释放给最终评分队列。
- **10-01约01:15 UTC v3续训已越过15k：** 低频agent实查trainer149195存活、step15000 checkpoint、同W&B c4fwrj21 running/telemetry15250。训练内val@15000 FID8.420053（当前best15000/同值）、MPJPE172.542mm、MAE.356946、R1/2/3=.33333/.46667/.55、Matching3.836275、Diversity7.590683；nonfinite grads/weights=0。EMA sum/count均FP32有限，count_min body/root .770043/.786587，码向量max L2 249.039/89.989，无旧式数值漂移。实际续训成功而非仅启动；仍按30k固定终点运行，主94307与几何队列9165未收终态，未启动MGT。
- **10-01约01:05 UTC v3补充物理评分已排队：** 实查续训trainer149195存活（149282等是DataLoader workers）；主新session9165等待该PID退出，严格检查manifest complete/final30000后，在GPU1重新加载最终selected做同480val几何/物理评分，输出`interhuman_vq_validation/hhc_interhuman_vq_bodyroot_ema_v3/hhc_selected_geometry_physical.json`，队列日志`/home/wenning/HHC_interhuman_runs/vq_bodyroot_ema_v3_geometry_queue_20261001.log`。训练失败则gate报错退出，不重启。94307负责原训练/审计/InterCLIP独立评分及官方对比；9165仅补充评分，二者尚无终态，不将排队当完成。保持source冻结直至两条收尾。
- **10-01约01:03 UTC通过10k续训决策、同v3已启动至30k：** 原pilot67334 exit0；完整1k–10k验证到齐，latest10k FID10.083237、best9k/9.639050、MPJPE185.094mm、MAE.368219、R1/2/3=.33542/.45833/.54167、Matching3.839800、Diversity7.554070。EMA sum/count两分支FP32有限、count_min均.785678、max每码norm207.911/94.765；训练非有限梯度/权重0。依据持续质量改善及码本数值健康，续同run/c4fwrj21至原固定30k，主新session94307，日志`/home/wenning/HHC_interhuman_runs/vq_bodyroot_ema_v3_continue_20261001.log`，未改结构/loss/数据/源SHA/schedule。仅启动，尚未验证超过10k；≥600秒agent继续监控，主收最终审计+独立评分+对比终态，MGT仍未放行。
- **10-01约01:02 UTC v3固定8k诊断完成：** 子agent40507明确exit0，root读回step8000/model_state_unchanged=true；同480val/91450person-frames，各通道11616token。v2→v3 body active436→631、perplexity185.535→298.619；root183→1000、88.947→666.383。MPJPE(mm)量化/body连续/root连续/全连续：v2=217.746/203.599/177.402/185.196，v3=183.070/186.386/164.372/189.111；两路使用恢复且量化质量改善，但连续绕过不是部署成绩，尚不能把剩余官方差距归于单一机制。结果`interhuman_vq_validation/quantization_probe_v3_step8000.json`，v2沿用旧文件未重跑。
- **10-01约01:01 UTC v3 10k pilot正常结束：** 主原67334明确exit0，manifest pilot_decision_ready/current10000，best训练内step9000/FID9.639050170；root重新核验17份source SHA及selected checkpoint SHA匹配，NVMe余34GB。尚未等同最终质量验收或MGT放行；低频监控agent收完整1k–10k轨迹/末态EMA，另agent GPU1复用既有probe做v3固定8k对比旧v2的已完成8k结果。收齐后决定同run固定30k续训，不另开参数实验、不改训练源。
- **10-01约00:56 UTC同4k EMA数值对照补齐：** agent CPU读取固定检查点，v2 body/root每码最大L2范数652.144/202.861、EMA sum BF16/count FP32、count_min均3.5088e-18；v3为258.309/87.318、sum/count均FP32、count_min .739700/.774469。纠正此前整矩阵norm比较口径，支持该节点无旧式EMA漂移，但不外推30k稳定性。主线程原训练67334收取仍返回同session运行中，无终态、不重启；8k快照尚未生成，固定8k诊断留至节点到达后执行。
- **10-01 00:54 UTC同4k固定验证集量化诊断完成：** 复用既有probe、GPU1/同480val/每通道11616有效token，独立输出`interhuman_vq_validation/quantization_probe_{v2,v3}_step4000.json`，root实读complete/step4000/model_state_unchanged=true。v2→v3 body active632→596、perplexity305.845→287.384；root active265→978、perplexity179.832→625.158，改善主要在root而非所有通道。MPJPE(mm)全量化/仅body连续/仅root连续/全连续：v2=222.437/213.420/197.789/211.477；v3=202.969/204.516/187.578/212.923。绕过latent可能偏离decoder分布，仅诊断不选模、不称可部署得分。子agent原v3 session83771 exit0；v2原17320首次poll未保留exit_code后句柄Unknown，不能编造exit0，root另查无probe进程、产物完整且量化MPJPE与训练内评分吻合，未重跑。训练GPU0/原67334继续，下一步10k决策及后续同8k对照，不据单个4k节点宣称完全修复。
- **10-01约00:50 UTC v3首次10分钟节点：** agent实查trainer141207存活、W&B c4fwrj21 running、最新训练telemetry5150，checkpoint与最新val4000。训练内FID14.654793（best4000/同值）、MPJPE202.969mm、normalized MAE.393255、R1/2/3=.29375/.43125/.510417、Matching3.859703、Diversity7.354924；root从旧审计读v2同4k为15.602493/222.437mm/.408528，三项初步改善，非最终独立score。最新batch body/root perplexity235.352/386.232、使用码数431/639，nonfinite grads/weights=0。checkpoint两分支codebook/code_sum/code_count均FP32；agent报整矩阵norm，不能与旧每码最大范数直接比较，下节点明确max per-code。维持固定10k pilot、主67334，不提前续训或启动MGT，≥600秒监控。
- **10-01 00:49 UTC审计重复来源已定位：** 低成本agent只读安装版W&B SDK并在线小窗复现：keyed `SampledHistoryScan`以1000步分页时，min0/max2001返回val/fid@1000及@2000各两次；unkeyed `HistoryScan`同窗筛选各一次。SDK sampledHistory相邻页共享边界，服务端在两页均返回边界事件；全历史60/30现象一致，不是训练真实验证了60次。保留单次完整历史扫描，指标值/step与旧审计去重结果一致，原审计归档不覆盖；此为审计性能及计数修复，不改变v2质量判定或v3训练。
- **10-01 00:47 UTC子agent疑点复核完成（只读）：** G026已有独立方向修订与物理评分，但接触未恢复，不算修复验收；G013右手/角色问题有证据，踢击目标及其逐关系时段仍未确认，提案禁止应用；G023现有资料不足以确定缓存错误。均未覆盖生产缓存，也未新增优化实验。沿用已有诊断产物，不因用户授权而全量重标。
- **10-01 00:46 UTC缓存复核授权与边界：** 用户说明配对及按距离自动划分的接触帧基本正确、但允许GPT子agent修复疑点。派`cache_repair_review`以实际可用gpt-6-luna low只读核对已有G026/G013/G023证据与修订产物，不重做全量、不覆盖缓存；确认配对后再独立检查逐关系时段，不以优化后的距离反选标签。生产替换前需保留版本并重优化、共同标签评分；证据不足保留待审。当前InterHuman v3训练不受此复核影响。
- **10-01 00:46 UTCW&B单次历史扫描复核：** 新审计38146 exit0、18项测试92152 exit0；新文件另存`v2/wandb_history_audit_single_scan.json`，旧审计未覆盖。严格JSON比较未完全相等：27项train仅count少30；16项val由60条变30条，旧events去除逐值重复后与新events逐值相同；所有指标极值/首末step、step轴及审计checks一致。故不能声称原始历史计数完全等价，训练事件重复来源仍待进一步确认；新扫描约22秒。当前v3绑定17份source SHA全部重新核验通过，没有改训练代码。
- **10-01 00:38 UTC量化器修复v3独立pilot已发起：** 新run`hhc_interhuman_vq_bodyroot_ema_v3`，主session67334、GPU0，日志`/home/wenning/HHC_interhuman_runs/vq_bodyroot_ema_v3_20261001.log`。新注册配置`configs/interhuman_vq_pilot_ema_v3.json`保持seed123/22,337,548参数/384-128 latent/1024 codes/batch64/损失/数据/固定30k schedule，仅显式quantizer=`fp32_window_reset_v1`，从头10k暂停决策，不复用漂移的旧EMA。模型缺少该参数时仍默认legacy，Inter-X源码未改；新量化器源文件加入训练SHA契约，结果路径按run隔离，旧JSON/检查点保留。完整34文件123测试主6898 exit0；正式64×64×524 GPU检查19473 exit0，BF16模型+FP32几何loss，所有loss/梯度有限、EMA三状态FP32。此前20316失败是诊断命令误将几何loss放入BF16，36803定位后按真实训练边界修正，不是修改loss规避错误。新run尚未确认optimizer首step，交≥600秒监控，不把启动当成功。
- **10-01 00:35 UTC旧v2全流程终态与正式指标已验：** 主85304明确exit0，32249此前exit0；W&B审计pass、全部7项checks真/无retry error，完整历史覆盖30k。严格独立重载selected18000的InterCLIP FID12.8482742604、MPJPE219.784629mm、MAE.397802659、R1/2/3=.28125/.397916667/.483333333、Matching3.859456889、Diversity7.433530331（GT7.751708984）；官方FID.985952136、MPJPE109.925614mm，其余对比全部落后。共同request/evaluator绑定通过，v2判为质量失败对照；不是joint FID。源码快照与RNG迁移记录已保留。确认旧shell/评分均结束后才接入v3，旧run不再监控/不续训。
- **10-01约00:30 UTC v2训练30k结束、补充评分完成：** manifest实读complete/final30000，selected step18000/best训练内FID12.848274；主32249明确exit0，严格重新加载selected/SHA/旧source通过，同480val/91450person-frames。HHC/官方：坐标RMSE150.979638/78.152762mm；normalized feature RMSE .731822763/.306380773；全身MPJPE219.784631/109.925617mm；body非root MPJPE221.873263/110.921957mm；root误差175.923361/89.002478mm；GT接触帧滑步6.997105/3.212843mm/frame（共同GT4.258247，257461transition）；10cm几何contact F1 .472474977/.586124650（HHC TP2077/FP1293/FN3345）。手MPJPE/Hands joint FID不适用，Full/Body joint FID未评（没有匹配InterHuman evaluator），几何F1不冒充语义/mesh接触。上述均未支持质量验收。selected SHA315279fb65a0c23d160484755a1b1f26dcde7aab156dd5118ee30f27ec7a29b5；几何产物hhc_v2_selected_geometry_physical.json。原85304尚运行W&B审计（agent实查审计子进程），未取得最终InterCLIP重载score/对比；继续原句柄，暂不改绑定source。
- **10-01 00:19 UTC训练内26k/最终几何评分已排队：** 低频agent实查trainer135221/launcher134840存活，W&B running最新optimizer26500，最新val@26000 FID13.323130、best18000/12.848274，MPJPE223.497mm、MAE.401833、R1/2/3=.26458/.39375/.47917、Matching3.859950、Diversity7.393869，非有限计数0。当前尚无30k最终score。主另起排队session32249，等待已确认存活的trainer135221退出，再严格检查manifest complete/final30000，读取selected并用原严格loader在GPU1做几何/物理评分，输出`interhuman_vq_validation/hhc_v2_selected_geometry_physical.json`；若训练失败则退出，不重启。主85304负责原训练+W&B审计+官方InterCLIP重载评分，32249仅补充评分，当前均未取得终态。必须两条均收终态并归档后才能改变旧source；不修改运行中的shell脚本。
- **10-01约00:12 UTC旧实验源码快照完成：** 低成本agent生成run内`source_snapshot_pre_quantizer_fix_20261001.tar.gz`，16份manifest绑定source+当前manifest快照共17文件/40139bytes；归档前16/16 SHA匹配，tar逐项读回0失败，命令exit0。tar SHA972e44726d0dc792e01e952ce65a82c919577ebde6a8ce25ac221470fccc14b2。仅源契约备份，不是完整环境/依赖打包；snapshot manifest仍历史pilot_decision_ready/10000，不能当训练终态。现有源码、训练、产物均未改或删除；为旧30k评分后接入新量化器保留可审计历史。
- **10-01约00:08 UTC训练内val至21k：** 低成本agent实查trainer135221/launcher134840存活、step21000 checkpoint及W&B同run running。current FID13.508147，历史best更新step18000/FID12.848274；MPJPE222.355mm、MAE.400551、R1/2/3=.27917/.40625/.49375、Matching3.858296、Diversity7.301474。body/root batch perplexity27.4058/35.2964、utilization .05664/.05078，非有限梯度/权重均0，无异常栈。数值有限不等于EMA机制健康，原85304仍至30k并独立评分。修复后对照拟保持原seed123、两分支384/128、1024codes、batch64、loss/数据/30k schedule，仅更换已测InterHuman FP32有限窗口EMA量化器，从头独立run；未接入/未启动，不续用已漂移EMA状态，不提前MGT。
- **10-01 00:06 UTCInterHuman专用量化器隔离修复已测试、未接入：** 新`models/interhuman_quantizer.py::InterHumanQuantizeEMAReset`继承原实现，forward/encode选码/EMA禁用autocast并保持FP32；保留21更新窗口，但窗口结束清usage，对实际重置码同时重建code_sum/count，防下一步又回旧中心；拒绝静默读取BF16旧EMA。3项CPU测试exit0（精度/梯度、过期used码恢复与EMA一致、下一次更新resume逐值相同）。真实GPU1正式1024×384/128两码本BF16前反向+84更新/4窗口55453 exit0，状态全部FP32有限，count_min=1，max code norm19.5959/11.3137对应测试常量输入，无漂移。全部当前训练source SHA实查不变：尚未接入VQ/model_tools或launcher，不影响原85304及Inter-X；不能以单测宣称训练质量已修好。待旧30k结束/独立评分归档后，再接入独立新实验，保持其它结构/损失/数据不变检验这一机制修复。
- **10-01约00:04 UTC定位到EMA数值机制，尚未改训练：** step16k固定前96val/2114有效token诊断96313 exit0，同BF16 encoder值仅将选码运算改FP32，body/root assignment差2.8855%/.8988%，真实平方量化误差2998.711→2997.453/370.635→370.433，故不能用瞬时选码误差解释全部收缩。CPU agent27121 exit0读取8k/16k状态：usage均1024/1024，而16k count>=1仅body107/root64，历史used但当前count<1有917/960；共享quantizer累积self.usage且reset窗口后不清空，已used闲置码不会再被重置。root进一步实读两通道code_sum BF16、code_count FP32；16k count最小7.00649e-44、body/root码向量最大范数1,284,238/741,455。独立零新分配EMA衰减复现实测exit0（初始sum=count=1，mu=.99）：BF16sum/FP32count在1k/8k/16k变1.6678/83.2945/65536，同FP32对照恒1，证明混精度累计可使闲置码漂移，并非仅相关性；但尚未用修复训练证明它解释所有质量差距。当前原85304固定30k继续作为有问题的对照，不将其质量验收为成功；后续应以InterHuman专用FP32 EMA/距离及正确有限窗口reset修复，不直接改变已接受的Inter-X共享训练行为，不原地改正在运行的源码。
- **09-30 23:59 UTC量化瓶颈诊断完成：** 新独立`tools/diagnose_interhuman_vq_quantization.py`在GPU1只读step8000/16000固定快照、同480val，原6538/78427均exit0；恒等/计数/不改权重测试通过，所有VQ state逐值未变，无训练。MPJPE(mm)按全量化/仅body连续/仅root连续/全连续：8k=217.746/203.599/177.402/185.196，16k=225.301/223.192/218.950/246.157。全量化与训练内MPJPE吻合。全val每通道11616token（末块ceil，排除纯padding）；8k body/root active436/183、perplexity185.535/88.947；16k降至151/76、57.806/35.357，证明码本有效使用在同一固定数据上明显收缩，不只是训练batch统计波动。绕过root在8k改善较大但16k很小，全连续16k更差，故不能把差距简单归为最近码误差；连续latent可能偏离decoder训练分布，不把该oracle干预当可部署质量或塌缩因果证明。产物`interhuman_vq_validation/quantization_probe_step{8000,16000}.json`位于NVMe run根。保持30k固定实验继续；若最终仍明显落后，不因资源预检通过就启动MGT，应先诊断VQ优化/码本收缩。
- **09-30约23:57 UTC训练内val至16k：** 低成本agent实查trainer135221存活/checkpoint16000/W&B running同run，current FID13.422204、best8000/13.273319，MPJPE225.301mm、MAE.398710、R1/2/3=.26875/.400/.500、Matching3.860354、Diversity7.416669；训练遥测body/root perplexity60.9856/40.4568、utilization .104492/.0625，非有限梯度/权重计数0。manifest仍停留pilot状态，不能据其判断进程退出；原85304继续、监控≥600秒。
- **09-30约23:55 UTC真实CLIP驻留预检通过：** GPU1低成本agent原89686 exit0，复用缓存ViT-B/32与官方InterCLIP，无下载；正式冻结TextCLIP真实caption输出[8,512]/[8,77,512]均有限，在线VQ+正式MGT BF16前反向及全部梯度有限，VQ state SHA前后不变。CLIP/InterCLIP/Adam moments同时驻留时peak allocated4,876,596,224B(~4.542GiB)、reserved4,970,250,240B(~4.629GiB)，全程8.093s含加载；官方evaluator以合成batch8得到有限[8,512]text/motion embeddings。此为资源/接口验收，不是完整训练或96例验证峰值，更不是质量评分。未optimizer.step/W&B/checkpoint写入，GPU1已恢复20MiB，GPU0原续训未改；正式MGT启动仍等待VQ最终验收。
- **09-30约23:51 UTC MGT正式尺寸资源预检通过（非训练）：** 低成本agent GPU1原81901 exit0；正式22,337,548参数冻结VQ+229,367,296可训练MGT，micro8、300帧/length297经真实online encode得到[8,75,2]，正式512D global/77word随机CLIP输入，BF16 logits[8,75,2,1024]。loss7.196882与全部306张量/229,367,296梯度元素有限；VQ无梯度且前后state SHA一致。未optimizer.step/未W&B/未写checkpoint；只分配真实Adam两个moment，峰值allocated约3.539GiB、reserved约3.598GiB，全程2.196s（非训练吞吐）。未加载真实CLIP模型，故不是完整训练峰值；24GiB卡资源余量充足但正式启动仍需真实CLIP/数据核验。预检只读旧step8000备份作形状/资源测试，不是最终选模或质量评分。GPU1结束恢复20MiB，GPU0续训未触碰；MGT仍未训练，等待VQ验收。
- **09-30 23:46 UTC RNG修复后的实际续训已验：** agent实查trainer135221存活、step11000 checkpoint在位、同W&B exxnl1l7 running/last11000，确实超过10k，不再只是启动声明。训练内val@11000 FID15.061368，历史best8000/13.273319；MPJPE232.474mm、MAE.396713、R1/2/3=.2875/.425/.5、Matching3.865232、Diversity7.328153；body/root batch perplexity130.5832/59.7929、utilization .196289/.096680。无异常栈。RNG修复只验收恢复行为，质量仍未达官方基准；沿用主85304至30k及独立评分，监控恢复≥600秒。此前12786明确exit0，MGT训练契约9项+新几何2项共11项回归通过。
- **09-30 23:42 UTC续训启动失败已修复并重发：** 低频agent发现旧trainer133801退出，root收54730明确exit1；`torch.load(map_location=cuda)`把CPU RNG ByteTensor也移到GPU，restore报错，step仍10000，故23:28的启动未产生任何有效续训。VQ/MGT两个restore函数仅加`.cpu()`恢复CPU/CUDA状态，不改训练目标；真实CUDA回放50824 exit0，两函数CPU/GPU后续随机数逐值一致。停机后备份旧源码/manifest/latest/best到run内`rng_resume_fix_20260930`；严格仅允许此两行差异的一次性migration49774 exit0，latest/best除training_contract_sha256外所有嵌套字段逐值一致，原文件完整保留；新contract26524911f8f6f9800c287bd6a890a40c79a4eeb54645dbdc581e8c96ccaec36a，迁移记录migration.json，非静默绕过哈希。现主85304、trainer135221实查存活，仍从10k续30k同W&B run，新日志`/home/wenning/HHC_interhuman_runs/vq_bodyroot_evalfix_v2_continue_rngfix_20260930.log`；尚未证明超过10k。agent约23:45做一次异常修复启动验收，之后恢复≥600秒监控；root收85304终态。新几何工具/训练契约回归另跑，MGT尚未启动。
- **09-30 23:36 UTC独立物理诊断基准完成：** 在新geometry工具内复用官方脚点[7,10,8,11]及GT接触标准(speed²<.001m²/frame²、y<[.12,.05,.12,.05]m)，统计预测xz位移而非训练loss；固定GT分母，预测抬脚不能规避计数。另定义22×22最近关节距≤.1m的逐帧双人几何contact F1，双方同阈值，明确是伪标签代理、不等于文本语义/mesh contact，且不是官方论文指标。2项合成测试通过，覆盖TP/FP/FN、padding、抬脚不改分母、无transition返回null。官方完整480val CPU15012 exit0：257461接触脚transition、滑步3.212843mm/frame，GT参考4.258247；45725pair-frames，TP2826/FP1395/FN2596、几何F1 .586124650。更低滑步不能单独证明更好（可能平滑）；坐标误差与前次逐值一致。新产物`/home/wenning/HHC_interhuman_runs/interhuman_vq_validation/official_intermask_geometry_physical.json`，保留旧coordinate JSON；HHC待最终selected同协议补评，训练链未改。
- **09-30 23:34 UTC独立坐标诊断实现并完成官方基准：** 新`tools/score_interhuman_vq_geometry.py`复用冻结480val/H5校验/最终selected加载器，不改训练链；测试已知误差、std缩放、padding屏蔽、分组/加权及非法输入均exit0。官方重构CPU评分13230 exit0，91450 person-frames：xyz坐标RMSE78.152762mm、normalized 0:258 feature RMSE .306380773（不同量纲，分开列）；full MPJPE109.925617mm、body非root21关节110.921957mm、root位置误差89.002478mm。无刚性/root对齐；body包含wrist但不冒充手指。与原正式评分full MPJPE差<1e-7m、有效帧数一致，所有训练绑定source SHA复核未变。产物`/home/wenning/HHC_interhuman_runs/interhuman_vq_validation/official_intermask_geometry.json`；HHC需待30k最终选定checkpoint后同入口评分。滑步/双人contact F1/Full-Body joint FID仍待评，手部两项N/A；不宣称十项完成。
- **09-30 23:32 UTC完整VQ报告覆盖审计：** 低成本agent只读核对，现有`utils/interhuman_vq_evaluation.py::reconstruction_errors`仅全身22关节MPJPE与normalized feature MAE，MAE不能替代RMSE。RMSE/身体分组MPJPE/root误差/独立滑步/接触F1仍待评；训练foot loss不是滑步评分，足接触flag F1也不是双人接触F1。InterHuman只有22关节到wrist，无手指，手部MPJPE及Hands joint FID不适用。Full/Body joint FID目前无匹配InterHuman评估器；Inter-X SMPL-X joint evaluator不能直接套用，当前FID一律标官方InterCLIP。下一步可在冻结训练代码之外对反归一化22关节补充坐标误差及明确协议的物理诊断，缺项不填造数、不用有利子集宣称完整验收。续训54730仍按原监控安排。
- **09-30 23:28 UTC 10k正常结束并续至30k：** 主25153 exit0，manifest pilot_decision_ready/10000，W&B exxnl1l7 finished；最终训练内val FID15.121286（best8000/13.273319）、MPJPE228.583mm、MAE.397891、R1/2/3=.275/.406250/.483333、Matching3.864225、Diversity7.351795。末段回升，不宣称质量验收；loss/权重/梯度均有限，body/root batch perplexity136.9737/64.4043，utilization .215820/.108398。root检查全部绑定source SHA、latest step/contract、有限权重及optimizer/EMA/RNG存在均exit0。综合前期下降与末段波动，保持预注册容量/loss/30k cosine不变、保留best，续训检验后期趋势；不启动MGT。新主session54730、trainer133801/conda133419已实查存活，GPU0同run从latest10000恢复，日志`/home/wenning/HHC_interhuman_runs/vq_bodyroot_evalfix_v2_continue_20260930.log`。尚未确认恢复后首step；原低成本agent首次≥23:38:47、之后≥600秒监控，root收54730终态；30k训练后shell还有W&B审计/独立加载评分/官方对比，不把训练退出当全流程完成。
- **09-30 23:27 UTC第二轮低频检查（训练内val至9000）：** agent实查trainer130801/launcher130401存活，W&B `exxnl1l7` running，主25153未退出。当前step9000 FID13.714373，历史最佳step8000/FID13.273319；MPJPE219.629mm、normalized MAE.394579、R1/2/3=.266667/.435417/.500000、Matching3.855515、Diversity7.394946（GT参考7.751709）。step1k至9k FID依次42.111890/22.963221/19.423080/15.602493/14.101801/15.784526/13.770760/13.273319/13.714373；MPJPE(mm)331.846/249.003/237.017/222.437/219.194/219.864/218.351/217.746/219.629。最新body perplexity/utilization153.2362/.224609，root73.0746/.116211；这是遥测批次值，不能直接作为全数据码本塌缩判据。近期改善放缓但较初期明显改善，无异常栈；待10k终态与最后一轮指标，无新增异常则按原协议续30k，不提前改容量/训练预算或启动MGT。
- **09-30约23:20 UTC补充VQ验收差异，未改变实验：** 只读核对官方opt/train/data/encoder，CPU载入发布权重实测主88375 exit0：64帧单人encoder形状[1,512,5,16]、80个code，即双人160 token；HHC同64帧双人32 token，token数量压缩强5倍，不是同瓶颈容量复现。官方batch256为单人窗口，当前batch64为双人窗口；按满batch估算官方20608步曝光5,275,648人窗口，HHC10k为1,280,000、30k为3,840,000，故不能用步数更多声称训练曝光超过官方。官方warmup约1/4 epoch+70%/85%里程碑衰减，当前1000步warmup+固定30k cosine亦不同。几何loss项/权重已数值核对，但这不等于架构或全训练流程相同。保持当前10k决策/30k预注册协议，先看曲线与码本利用率，暂不把差距归因于某一项，也不擅自加训练预算。
- **09-30 23:16 UTC v2首轮低频训练验证：** 低成本agent约23:14只读W&B run `exxnl1l7`，state=running、已见step4000；训练内480val当前FID15.602493，历史最佳同step4000/FID15.602493，MPJPE0.222437m=222.437mm，R1/2/3=.270833/.404167/.477083、Matching3.868612。相较同协议官方VQ FID.985952/MPJPE109.925614mm仍明显落后，不作质量验收，也不混称joint FID。root约23:15实查trainer130801存活、step5000 checkpoint已落盘；原25153继续，不重启。下一步10k pilot终态后看完整曲线决定是否续训30k，监控继续≥600秒。
- **09-30 23:10 UTC完整CPU回归与MGT容量核算完成：** 主79196 exit0，显式执行全部30个`tests/*interhuman*.py`文件的115项无fixture测试；修正测试中残留三分支配置预期为512/12、384/128、2 codebooks与22,337,548参数，未修改运行中训练绑定源码。低成本agent正式配置meta构建18447 exit0、容量核算12447 exit0：MGT可训练229,367,296参数，每份FP32权重+Adam checkpoint约2.564GiB；代码实际覆盖latest/best/last，含原子临时副本峰值约10.255GiB，并非保存300份。NVMe当前约44.5GiB可用，继续使用`HHC_INTERHUMAN_OUTPUT_ROOT=/home/wenning/HHC_interhuman_runs`，无需修改保存策略或删除旧产物。v2仍沿用主25153；新验证结果待≥600秒监控，不以测试通过代替训练质量验收。
- **09-30 23:04:37 UTC当前唯一VQ训练句柄：** 新v2主session25153、trainer PID130801(Rl)/launcher130401，初始化中；低成本agent已改监控新目录，首次≥600秒后。旧49910已exit130，旧“运行”记录均历史。GPU1官方VQ/GT评分均已结束，无评分队列。新v2尚未取得训练内FID，之后只用修正wrapper结果判断。
- **09-30约23:04 UTC评测输入根因修复，VQ以独立v2重启：** GT自评R@1=.15625暴露评测归一化错误；核读官方`eval.py`先normalize=True生成、后False评分，EvaluationDataset对生成做normalizer.backward。旧HHC wrapper直接送normalized输入，故旧FID/retrieval/Diversity无效；MPJPE已有反归一化不受此问题影响。已核实并SIGINT旧trainer128635，主49910明确exit130；最后W&B step1080、checkpoint step1000、旧FID44.279无效，全部旧文件保留且添加ABORTED说明。修复共享InterHuman wrapper对GT/生成统一反归一化一次，并绑定stats及wrapper SHA；20项评测/launcher/contract检查通过（含调用原官方wrapper方法比对）。官方VQ原H5复用、无效JSON改名`.invalid_normalized_inputs.json`保留，修正评分62006 exit0；GT复算30505 exit0。官方VQ step20608/epoch46，同480val/seed20260130：FID .985952、MPJPE109.925614mm、MAE .174309、R1/2/3 .429167/.587500/.691667、Matching3.793986、Diversity7.731043(GT参考7.751709)；GT self FID -1.002e-6数值误差、R .422917/.600000/.697917、Matching3.785759、MPJPE/MAE0。以上是官方InterCLIP FID，不是joint FID；其余VQ物理/分部指标待评，不宣称完整十项验收。对比器另修Diversity应接近GT、拒绝NaN/Inf，4测试通过。现新run`hhc_interhuman_vq_bodyroot_evalfix_v2`从头启动GPU0，固定原训练参数/10k gate，日志`/home/wenning/HHC_interhuman_runs/vq_bodyroot_evalfix_v2_20260930.log`，旧run不续训/不改写manifest。
- **09-30 22:57 UTC pilot活跃句柄：** 主session49910，launcher PID128235，trainer PID128635；首次ps确认存活、数据初始化中，尚未确认optimizer step或validation。已有低成本agent `/root/full6238_artifact_audit`按≥600秒只读监控首step/新val/异常，主agent负责原49910终态；不重复轮询或因日志空重启。当前run目录`/home/wenning/HHC_interhuman_runs/vq_interhuman/hhc_interhuman_vq_pilot_v1`，旧HDD实验全部保留。
- **09-30 22:57 UTC第③项VQ启动前验收完成并发起pilot：** GPU0正式尺寸64×64×524 BF16真实train batch前反向均有限，预检30096 exit0，无optimizer step；训练数据3242pairs/458928windows匹配contract，初始化耗106.17s，前反向0.605s、峰值allocated0.300GiB；官方96例evaluator得到96×512有限text/motion embedding，整段峰值1.367GiB（不是完整FID评分）。发现并修复EMA checkpoint helper固定访问第三quantizer的残留；两通道下一次forward/EMA恢复逐值相同，原三通道恢复/legacy fallback回归67895 exit0。旧launcher错误要求胜过Interact2AR，已按最新顺序改为SHA绑定的完成shape overlay/6238分母/>60%交付门槛，两launcher支持HHC_INTERHUMAN_OUTPUT_ROOT且契约测试通过。W&B只读认证edmond_gla/NewInter成功。现使用NVMe `/home/wenning/HHC_interhuman_runs`启动GPU0 10k pilot/固定30k schedule，batch64、384/128 latent、1024 codes、在线W&B；日志`/home/wenning/HHC_interhuman_runs/vq_pilot_20260930.log`。已发起不等于已完成或质量验收；下一步确认真实step并交≥600s低频监控，运行期间冻结训练绑定源码。
- **09-30第③项两通道训练/评测链回归通过：** online batch与decode改75×2，复用mask算法支持两次独立tau排序(root mask≤body)，保留三通道原RNG/公式；修复旧分支accuracy混算其它通道。恢复原preregistered historical训练text dropout支持，推理仍paper_text_cfg，不暗改训练协议。低成本agent同步4份配置/W&B profile，主agent同步训练默认384,128及data contract；VQ正式尺寸实测22,337,548参数。10个测试文件共57项主2744 exit0，覆盖real VQ/MGT前反向、采样/解码、两/三通道mask、artifact绑定、训练gate/恢复与配置指标；此为代码/协议检查，不是训练质量验收。GPU0/1均空闲；HDD仅2.1GB，而VQ32份full checkpoint按参数+Adam粗估8.58GB，NVMe所在根分区47GB。下一步正式尺寸GPU显存/数据/evaluator/W&B预检，选择充足存储路径再启动pilot，不删除旧产物。
- **09-30第③项两通道MGT核心接通：** `models/interhuman_transformer.py`复用现有attention blocks/head/text dropout实现独立body/root嵌入与root→body交错流，attention mask和输出均2通道，seeded/unseeded采样不再退回三通道；shared Inter-X transformer未修改。初始化显式排除冻结VQ，防止继承的递归初始化改写编码器权重。VQ/MGT测试文件全部11项主76815 exit0，paper与historical两种topology均验证梯度有限、VQ权重逐值不变/无梯度、固定seed单条/批量采样一致、END/PAD及无seed路径；补最终入口限制后定向回归exit0。仍未训练：batch bridge、mask ratios、分支accuracy、评测及配置尚待改成2，完整训练协议未验收。
- **09-30缓存子agent复核完成：** G026坐腿第二配对方向有文本/视觉支持，但已做诊断并未恢复接触；G013扇脸左右手有文本/mesh支持，踢击目标仍不确定；G023漏拍背有证据但手侧/时段边界未完全确定；互惠三例仍属提案。复核未发现可不经重优化/评分直接替换生产的已验产物，原缓存和过滤后增强不变。详细证据沿用`docs/shape_text_contact_review_20260912.md`及`docs/shape_pair_repair_results_20260912.md`，不重复历史GPU实验。
- **09-30第③项两分支VQ定向验证通过：** 当前实现为两个真实encoder/quantizer，body/root输入512/12、默认latent384/128，输出完整524维，tokens最后一维2；旧类名仅导入别名，不兼容旧三分支权重。6项feature partition检查+5项VQ/Inter-X宽度回归检查exit0；真实动作3292接官方几何loss，前反向、两encoder参数更新、encode/decode有限性通过，连同3项官方loss数值/权重/双人对称检查主90357 exit0。共15项定向检查，不是完整测试套件，不证明训练复现。尚未启动训练：InterHuman MGT base/forward、采样、batch、训练指标和配置仍残留三通道，须全部贯通并验收；本轮不改共享Inter-X transformer。Ponytail复用现有encoder/decoder/loss，无新增依赖。
- **09-30用户补充缓存修订原则：** 配对及按距离自动划分的接触帧段基本可信，但允许GPT subagent修复明确疑点。复用已有低成本GPT agent只读核对历史修订证据，不重复已完成实验；仅修证据支持的关系，受影响帧段须依据原动作逐关系距离复核，不能凭文本臆造或用优化后距离反向定义验收标签。原缓存/已交付增强与冻结评分保持不变；独立修订经主agent验收、重新优化和统一标签评分后才可替换。当前第③项两分支改动尚未完整测试，不宣称训练已启动。
- **09-30进入第③项，训练前发现旧方案不符：** 现有InterHuman representation/VQ/MGT准备代码与pilot仍是body/arms/root三通道(320,192,12)，不符合用户要求body/root两部分，尚未启动旧配置。下一步将arms并入body，保留全部524D，分支512/12，root→body；连同token batch、采样、评测及配置一起改，避免只改VQ留下三通道接口。loss实际对齐测试78119 exit0：六几何项对官方InterMask逐项数值一致、默认加权与双人平均/commitment只计一次共3项均通过。环境无pytest，直接运行现有无fixture测试函数；此前unittest收集0项(exit5)不算通过。低成本agent只读核对数据/evaluator/已有实验资产，主agent负责结构修改及训练验收，按ponytail复用现有训练评测流程，不动Inter-X指标代码。
- **09-30全量增强打包验收完成：** 主28765 exit0，`dataset/interx_shape_ssp62_physical_full6238_reviewfiltered_20260930`完整内容读回通过，4148增强+9110原训练=13258；6238请求分母、两份gate/源文件/排除表绑定均验证，7已知风险名均未进入增强名单。没有启动训练。root复算全6238平均foot 6.049045→5.547326 mm/frame，Eq6 0.146984298→0.027224764 m²/frame；严格单项foot4823、Eq6 5330、联合4154。7278有效接触关系（5666样本，572无有效关系）等权距离original/new/opt=103.776/107.016/96.205mm，opt相对new改善4425/7278，2cm帧占比47.496%/43.089%/49.810%。该接触指标基于旧自动时段与缓存，非全面文本正确性认证。更严格要求双方各自foot observations均>0时联合4150/6238（敏感性，不改正式gate）。第②项速度/物理门槛及可用数据交付已实测完成；语义局限保留，下一步按顺序进入第③项InterHuman body/root双分支实验，勿重启①。
- **09-30 22:35 UTC全量物理评分/gate完成：** 恢复76361明确exit0，remaining6138最终完整；标准gate重建33297 exit0，4080/6138通过。root完整selection核验15878 exit0：与first100合并6238唯一名称、位置0..6237，旧4879行逐值保留，恢复绑定匹配；物理4154/6238=66.591856%，已知7语义风险过滤后4148/6238=66.495672%，分母不缩小。全量selection已保存`output/ssp62_full6238_reviewfiltered_selection_20260930.json`。打包前空间3,499,941,888 bytes，双H5未压缩motion保守估计1,470,838,656 bytes+256MiB余量满足；开始现有materializer全量新overlay及标准完整内容读回，尚未训练/宣称文本全正确。完整contact汇总由低成本agent提取中。
- **09-30恢复评分已启动且断点重算通过：** 按ponytail新增独立`tools/resume_shape_support_transfer.py`复用原评分器（原文件及指标公式未改），旧4879行partial只读保留。核验floor/全部ground bindings、模型checkpoint/Volume实现、prefix contact/timing/NPZ绑定后，重算末例`G026T004A023R015`完整row并以rtol1e-6/atol1e-10检验一致，已通过；旧记录数值仍原样保留，物理gate仍无容忍。恢复主session76361/PID122199，GPU1，日志`output/ssp62_physical_remaining_assessment_resume_20260930.log`，新输出`...remaining6138_assessment_recovered_20260930.json`，1260次计算含1次重放+1259缺失。Python故障栈开启，禁新巨型core转储。恢复边界测试51080 exit0、原pair评分测试15667 exit0；首条`Recovered total 4879/6138; complete=False`不是验收完成。旧partial未绑定自身scorer哈希：以冻结首100报告/current code及重放交叉印证，恢复provenance明确保留该历史证据限制。旧3965/28329已终止，不自动重启打包，待新评分终态后重建gate/收尾。
- **09-30崩溃系统证据：** 低成本agent读取kernel日志，20:46:05 PID104291报`segfault ... in nvidiactl`；同次检索无OOM/GPU Xid记录，`coredumpctl`未安装（exit127），无可用调用栈，尚不能确定根因。不得将其直接归因为磁盘不足或某个样本损坏；后续恢复需保留原检查点、启用Python故障栈并验证断点附近复算一致性。
- **09-30评分段错误，原队列已终止：** root收原3965明确exit139；最后日志4879/6138，PID104291段错误。原收尾28329 exit1，因remaining gate不存在，未打包/训练。`...remaining6138_assessment_20260912.partial.json`仍为可解析56,075,687字节、4879行，每行含Volume/contact，名称严格匹配生产清单前缀；下一例`G019T005A027R013__shapeaug_s00000123`，剩余1259，不能把partial当最终结果。磁盘3.4GB、GPU1已释放。低成本agent查系统core/kernel证据；root诊断现有评分器恢复路径，优先复用已绑定检查点，保持旧产物/评分公式不变。当前无运行中的评分/收尾队列，旧3965/28329不再是活跃句柄。
- **09-30评分I/O等待复核：** 低频监控在4395/6138处捕获PID104291由Rl变D；root只读ps为Dl、wchan=`rq_qos_wait`，同时日志继续到4398/6138，不能据此判为卡死。磁盘剩余3.4GB，未删除文件或重启评分；最终assessment/gate仍待产出，打包前须关注可用空间，原3965/28329保留。
- **09-30公开shape来源说明补齐：** 只读核对生产contract所指plan，`donor_pool_kind=ssp3d_transferred_smplx`，不是AMASS/HumanML3D。作者[SSP-3D仓库](https://github.com/akashsengupta1997/SSP-3D)说明311条图像标签包含SMPL shape/gender，shape是多帧拟合的pseudo-ground-truth而非扫描真值；仓库[LICENSE](https://github.com/akashsengupta1997/SSP-3D/blob/master/LICENSE)为MIT，保留版权/许可声明。该事实不等于SMPL/SMPL-X模型或原图也可无限制再分发；当前仅本地研究使用，不发布这些资产。生产/评分/打包输入未改，仍等待完整物理评分。
- **09-30完整floor已完成、独立Volume阶段开始：** 低成本监控发现正式remaining6138_floor JSON与“Volumetric completed 72/6138”日志；root核验12276行恰覆盖6138例的original/new两个条件，completion SHA一致，exit0。原3965继续独立评分，未产生最终assessment/gate，不从72例推算通过率；全量收尾28329继续原等待链，冻结配置。
- **09-30全量产物完整性审计通过：** 低成本agent最终New_MMM session1445 exit0；6238唯一名称/位置0..6237、6238 sidecar、12476NPZ SHA全匹配，37428个float数组全有限；12476份betas以float32逐值等donor且逐人不同source，8个实现文件hash匹配，坏样本0。该审计仅确认生成/数据绑定，不代替后续滑步、穿透、contact评分；3965与28329继续等待后续节点。
- **09-30全6238生成完成：** 原生成session25321明确exit0，remaining6138清单complete=true，合并first100共6238唯一样本。root读取全部sidecar统计optimizer调用均值4.192379s、中位4.516103、330/6238>5s；样本处理均值4.451407s、中位4.766455、1230/6238>5s（不含整次启动/评分与部分日志写入）。速度均值达目标，但不是每例<5秒。评分队列95830已自动进入floor核验；完整质量通过率尚未产生。另启动低成本gpt-6-luna全6238/12476NPZ一次性完整性审计，生成完成不等于通过验收。
- **09-30低频监控交接：** 已有低成本gpt-6-luna子agent `/root/partial_w1_inventory`负责≥600秒间隔只读监控，仅阶段切换/异常通知。其09:04 UTC实际ps核实95782 Ssl与95830 S存活，remaining5153/6138、complete=false，评分尚未开始；子agent等待session30234，不接管root25321/3965/28329。主线程停止重复例行轮询，原句柄终态仍由root收取。
- **09-30全量收尾队列建立：** 主session28329等待现评分队列PID95830结束，随后重建first100/remaining6138两份物理gate、断言选择位置完整0..6237及6238唯一请求、应用已绑定7例语义风险表；保存`output/ssp62_full6238_reviewfiltered_selection_20260930.json`，只有过滤后仍>60%才打包到`dataset/interx_shape_ssp62_physical_full6238_reviewfiltered_20260930`，最后全内容读回。任一步失败即停，不自动训练、不把排队称验收。当前活跃生成25321/PID95782、评分3965/PID95830、收尾28329；冻结本轮打包器/selection helper/风险表直到收尾终态。磁盘实查5.1GB余量；未删除历史产物。
- **09-30过滤打包真实验证完成：** 原86436 exit0，新`dataset/interx_shape_ssp62_physical_first100_reviewfiltered_20260930`实际打包68增强样本+9110原训练样本=9178；标准完整内容校验读回通过，quality保留100请求/74物理通过/68过滤保留及文件SHA，7风险名无一进入增强名单。旧74 overlay未改，未启动训练；此仅验证安全打包链，最终6238仍待完整评分，剩余语义正确性不因过滤而被认证。
- **09-30过滤打包验证句柄：** 主session86436，CPU串行first100新overlay打包及全内容校验；预期保留68/100（74物理通过中6例命中7例风险表，另一例本已物理不通过）。这是风险过滤后的保留数，不是人工语义正确率；最终结果待收。
- **09-30已知语义风险过滤入口：** 按ponytail复用现有quality-selection链，为materializer添加可选`--semantic-exclusions`；共享校验器打包/读回均重建物理gate并验证排除文件SHA，保持原请求分母，同时保留physical_accepted_names与过滤后的accepted_names；默认行为不变。7例名单`docs/shape_semantic_exclusions_20260930.json`只隔离旧缓存产物，不改原数据/缓存/timing，不认证剩余文本全正确。6项单元测试11689 exit0（分母不缩水、物理名单不丢、篡改/未知/重复/空原因拒绝）。开始first100真实打包+完整读回验证，输出独立reviewfiltered目录，不训练、不作为全6238验收；生产与评分核心未改。
- **09-30同200例weight1最终验收完成：** 原61031收exit0，root复算脚滑147、独立Eq6 172、联合126/200=63%；同weight10为144/154/112=56%。200名字/原shape完整Volume/配对/时段/角色全部逐值一致。weight1脚滑5.367354mm/frame、Eq6 .032080626m²/frame，224关系接触90.884046mm（142改善）、最近5%102.753128mm（132改善）、2cm比例52.0368%；调用4.134360s、5/200>5s。weight10虽接触71.178759mm更好但联合下降7个百分点，主要伴随穿透通过数下降，不推广。weight1开发100+额外200合计200/300=66.67%，仍不等同全量/文本语义验收；保留当前生产weight1、停止权重探索，完成全量独立gate和语义问题清单后再打包。
- **09-30weight1补基线200优化已完成：** report200行，调用均值4.134360s、5/200>5s；低成本agent实际读400NPZ有限/双方donor且与source不同，200输入SHA和顺序全部通过（最终exit0）。主session61031进入独立评分，实查PID96358 Rl；未有最终assessment，因此不报通过率，不将生成验收冒充物理验收。
- **09-30恢复越过断点且7例重算已验：** 低频实查95782 Ssl、remaining3940/6138、complete=false，累计4040/6238仅生成数。root实际读取7份新sidecar及14NPZ，SHA全部匹配，exit0，旧21备份仍保留。weight1补基线61031仍运行，尚无最终report/assessment。旧20260923对照低成本审计：只有前9完整，后7三文件均0字节，其余184缺失，不能拼成完整验收。
- **09-30启动后实查：** 恢复生成95782从初始磁盘等待进入Rsl，日志出现真实50步优化迭代；新weight1 holdout200日志亦有实际迭代。25321仍运行、不是退出；后续按≥10分钟低频检查，不因初始日志为空再启动。
- **09-30weight1补对照句柄：** 主session61031，GPU0串行完整200优化与评分；尚未完成。当前主活跃为全量25321、其评分队列3965、独立200补基线61031。
- **09-30同200例weight1补对照重新启动：** 旧20260923目录仅16份逐例JSON、无最终report/assessment且没有活跃进程；保留原目录，不把零散产物拼成已验收报告。新独立目录`output/ssp62_matched_attract1_iter50_holdout200_20260930`完整重跑200（沿用同inputs/floor/50步/.001/weight1），之后接独立评分。此为中断补基线，不增参数点；恢复全量25321仍沿用、不重启。
- **09-30评分队列句柄：** session3965等待恢复生成PID95782，复用原完整性检查→floor→独立Volume/contact→strict gate脚本；未自动训练。主线程负责收25321/3965终态。
- **09-30重新恢复全量：** 实查无旧实验进程，weight1 holdout200没有report/最终assessment，旧34070不当活跃或已完成。remaining清单仍3769/6138、complete=false；此前7例已存在21个`.unverified_20260923`备份，当前源文件缺失，本轮改名预检exit1因源已隔离，未重复移动/删除。原入口无overwrite续跑session25321/PID95782（实查Dsl），仅缺失项重算；日志`output/ssp62_remaining6138_recover7_20260930.log`。继续weight1/原50步/.001/地面方案，不推广holdout56%的weight10。评分队列对新PID重新建立，终态待收。
- **09-23新增活跃句柄：** 同holdout200 weight1优化→独立评分主session34070，已启动GPU0；它是补基线，不是重启weight10或全量任务。全量恢复尚待完整性审计，不重复启动。
- **09-23恢复核对与weight10扩样失败：** 旧50572句柄Unknown，旧PID36864/38265已不在；最终assessment真实存在并完整200行，不重跑已完成实验、不编造旧exit_code。严格脚滑144、Eq6 154、联合112/200=56%，未达到>60；开发100+额外200合计177/300=59%。均值原→优化脚滑5.802534→5.387037mm/frame、Eq6 .165585370→.034044708m²/frame；224关系原/换shape/优化接触99.884587/101.171692/71.178759mm，175改善；最近5%112.209488→81.942496mm、169改善，2cm比例43.1791→64.5302%。均值改善不代替逐例失败，不推广weight10。
- **09-23原全量确已中断：** 旧生成12484与评分12632均不在、没有相关活跃进程；remaining清单complete=false、3769/6138，未有最终gate。低成本gpt-6-luna只读核验所有已登记sidecar/NPZ及源码绑定后再决定恢复；不把旧日志“waiting”当活跃。GPU均空闲。另启动同holdout200 weight1对照以区分批次难度与吸引权重效应，不增加权重点；沿用原floor/shape/50步/.001，输出ssp62_matched_attract1_iter50_holdout200_*20260923。
- **09-12额外200优化完成、评分运行：** 原50572下优化已结束，独立评分38265 Rl；root与低成本agent实际读取400NPZ，有限/200份sidecar SHA/顺序/400份指定donor且不同source全部通过，root exit0。调用均值4.148529s、中位4.553831、P95 4.895066、5/200>5s；不含准备与评分。最终assessment未出，不提前报通过率。全量recipe与核心保持冻结。
- **09-12额外200已进入优化：** 低频实查父PID36864、优化37138 Rl；floor200已完成，原session50572继续，尚无最终优化report/assessment。GPU1生产12484与评分队列12632存活。磁盘余8.6GB，未删除历史证据；不因日志未出最终文件重启任务。
- **09-12额外200输入已完成并隔离核验：** 50572返回真实“holdout200 disjoint and optimizer inputs matched”，已进入全序列floor准备；root另读300份sidecar确认100/200个source_name也完全无交集（不只是augmented_name不同），exit0。不把random-only输入计为优化完成或质量通过。
- **09-12权重10完整100例终态与扩大验证：** 原62588 exit0；root复算严格联合65/100（脚滑77、独立Eq6 85不增，双方观测正），脚滑5.814842mm/frame、Eq6 .023799623m²/frame。124关系接触78.998588mm、101改善；双向最近5%91.454218mm、95改善；2cm比例62.67698%。同原shape完整Volume和标签/角色/timing绑定逐值一致。平均调用4.210925s、7例>5s。冻结weight10，不再追加权重点；主session50572已启动offset100/limit200的独立输入→全序列物理floor→同weight10优化→独立评分，输出ssp62_matched_attract10_iter50_holdout200_*。输入为random-only只用于取原source/donor/crop，不使用旧优化结果；checker重新恢复固定物理地面；执行前断言与开发100名字无交集、影响优化的配置一致。尚未有200例通过率，不改GPU1生产recipe。
- **09-12漏配对pat67对照结束：** 原12913 exit0，control/corrected与独立物理、共同正确标签contact均完成。正确角色swapped=true，旧crop47–66共20帧；原shape/换shape/control/corrected距离2.630821/2.202378/5.542323/3.624027mm。补配对改善相对control但仍比初始化大；两者均物理通过（原脚滑3.054240mm/frame/Eq6 .146809386，control2.978077/.000409013，corrected2.966867/.000616907）。属于合理近接触内的改善，不追求自设毫米完美；原缓存仍不覆盖。
- **09-12权重10优化产物已验：** 100例优化report生成；低成本agent与root分别实际读回200NPZ，float全部有限、200份beta逐值匹配源sidecar公开donor且均不同于original，100个sidecar SHA与名字顺序全部一致，root exit0。调用均值4.210925s、中位4.582596、P95 5.037190、7/100>5。原session62588进入独立评分，尚无最终100例通过率；pat67 session12913仍按队列等待，不重复生成。
- **09-12接触指标交叉检查：** 既有100例表的contact mm是逐关系、按时段加权的最小顶点距离（124关系等权），不是整片表面平均。补算更广的双向最近5%指标：原shape123.061835mm、换shape127.035103；weight1优化118.386606（75/124改善），weight30优化78.670361（109/124改善）。支持接触收益并非只改善一个最近顶点，但不能证明文本/动作语义全正确。weight10优化report已生成，完整assessment仍待原62588，不用部分评分估计成功率。
- **09-12拍背对照活跃句柄：** 主session12913已建立，等待weight10父33143退出且最终assessment存在后执行；当前排队，不是优化已完成。主agent负责收终态，生产6495/评分36631与weight10 62588均沿用原句柄。
- **09-12拍背漏配对确认与隔离提案：** G023T002A011R011（first100 offset67）caption拍背但原pickle为空。主agent看骨架，原shape全crop网格90669 exit0：P2右掌→P1背最近1.171162mm/source65，<5cm为source55–83共29帧；旧自动timing20正帧，不把其当人工真值。低成本GPT生成独立`docs/contact_pair_correction_pat67_20260912.json`，both palms→back保留未指定左右、9:1；root批准diagnostic-only并从实际文件通过reviewed_contacts校验25472 exit0，SHA68fdd0d72c02464b9d988bd20fcfc5ca42a07e165d5891746c427060a0d21c0b。未改源缓存/时段；control/corrected同50步/.001/weight1/地面对照排在weight10父PID33143后，输出ssp62_pat67_*，接独立物理和共同正确标签contact，尚未验收。
- **09-12已知语义疑点与物理通过交集：** 当前7例待复核名单中，weight1物理74例包含6例、weight30物理60例包含5例；剩余68/55仅“未命中已知疑点”，不是已通过语义审核或新的成功率。保留物理指标原定义，不直接用于训练。
- **09-12权重10/100例单因素对照已启动：** GPU0空闲后主session62588串行优化与独立Volume/脚滑/contact评分；同100例、floor/50步/.001/原配对时段/地面200/1/.5，仅attract=10。输出`output/ssp62_matched_attract10_iter50_first100_20260912`及对应assessment。动机为weight30完整100例仅60%边界，检查较保守吸引是否能兼顾接触与>60%物理通过；未启动其他参数点或生产更新。
- **09-12低频全量复核：** 低成本agent实际ps确认GPU1生成12484及评测队列12631/12632存活；remaining2892/6138、complete=false，累计2992/6238为生成数，不是通过数。继续原6495/36631，不重启。
- **09-12权重30/100例完整验收结果：** session50170 exit0；脚滑75/100不增、独立Eq6 79/100不增、交集60/100，恰在60%边界，不能称稳定超过目标。脚滑均值5.876432mm/frame、Eq6 .028094015（原6.279624/.140836875）；124条有效关系接触原shape111.473877、换shape114.287273、优化66.619748mm，113/124改善，2cm比例68.0390%。调用4.209581s均值、6/100>5s。已知语义风险未解除，不启动训练/第③项，不原地替换生产recipe。
- **09-12权重30/100例产物读回通过：** 主agent实际读取200个NPZ，100名字/顺序与manifest一致、浮点字段全部有限、双方betas逐值等于公开donor且均不同于原shape；没有原样fallback。独立评分PID31918实查Rl，原session50170继续，质量验收仍待最终报告。
- **09-12权重30的100例优化完成：** 最终report包含100行、生成PID31083已退出；optimizer-call均值4.209581s、6/100>5s，不含源/floor准备和CPU/独立评分。原整体session50170仍在后续评分阶段，尚无100例联合通过率/完整contact结果；不把优化完成当验收。
- **09-12低频复核与汇总：** 实际ps确认GPU1生成12484、评分队列12631/12632存活；剩余分片2620/6138、complete=false，累计2720/6238仅生成数。GPU0权重30的100例优化PID31083活跃，评分尚未启动，原session50170继续。已把所有完成的25例权重/LR/原生基线统一收录`docs/shape_pair_repair_results_20260912.md`，避免分散指标混用；无新的100例质量结论。
- **09-12权重30扩大100例已启动：** 主线程session50170，GPU0同floor100/50步/.001/原配对时段/合理地面、attract30，优化后接独立Volume+脚滑+contact；输出`output/ssp62_matched_attract30_iter50_first100_20260912`及对应assessment。为统一floor100绑定并复核运行波动，包含原前25例重跑；尚无100例通过率，已知语义问题不因物理通过而解除。拍背时段84019已全部exit0，未修改缓存。
- **09-12权重30小批量达到数值目标：** session77615 exit0，25例联合17/25=68%（脚滑20、Eq6 21）；均值脚滑5.851390mm/frame、Eq6 .008749772，接触58.822080mm、29/32改善、2cm比例60.0572%。调用均值4.275073s，3/25>5；较weight1接触106.436mm约减半，但通过率84→68%。仅25例，不是全量验收；下一步扩大100例，先让既有拍背时段84019完成，不混改地面/标注。
- **09-12拍背逐关系时段入口已实现：** 新隔离`tools/run_geometry_relation_timing_pilot.py`复用checker和MeshGroundSupportLoss，不改仍运行的核心；从绑定原shape的逐帧union距离生成<=1cm权重1、1–3cm线性降至0的几何伪时段，可扩展旧mask外，仅改变语义项，aux/地面维持原时段与配置。小测试41889 exit0，真实样本待运行；排在weight30父PID28621结束并有最终评分之后，不称修复已验收。
- **09-12三例配对验证完成：** session76353 exit0。互抓臂/互拍背两例新旧配对均物理通过，但正确标签第二关系没有恢复；原global时段把两关系混在一起。互拍背原geometry第二关系<1cm有18帧，其中10帧不在旧mask，支持仅对该例重建逐关系几何伪时段。全零时段打脸例实际原右手—头最近49.621/85.575mm，尚无强证据应补造真实接触。weight30沿用77615队列。
- **09-12配对修订首例结果：** G045T002A010R003的control/corrected物理评分均完成，脚滑6.999574mm/frame≤原8.486184、Eq6双方0，但接触评分为空；主agent查原完整88帧timing全0，非裁剪丢失，所以修订没有实际吸引生效，不作为语义成功。三例队列76353继续余两例，其旧mask分别131/16正帧。权重30仍排队。
- **09-12权重100/1000配对已全部结束：** session40887 exit0。weight1000最终25例：脚滑不增11、Eq6不增14、联合7/25=28%；均值脚滑6.115023mm/frame、Eq6 .049931418、32关系接触31.757503mm，31/32改善、2cm接近82.4919%。调用均值4.273481s、3/25>5。比weight100更贴近但物理通过进一步下降，不采用；三例修订76353、随后weight30 77615继续已有队列，不再追加参数点。
- **09-12失败分解与下个诊断点：** weight100的11失败为脚滑9、Eq6 6、交集4。14物理通过样本的19条关系，weight1/100接触均距77.998/36.252mm；失败组13条关系147.999/57.462mm，收益不是只来自不合格样本。追加同25例weight30，排在三例配对修订队列PID27739后，沿用50步/.001/地面不变；未开始计时或宣称通过。weight1000优化report已有，完整评分待收。
- **09-12权重100完整25例已评分：** 最终assessment25行、绑定同旧baseline；联合通过14/25=56%（脚滑16、Eq6 19），尚未达到>60%。脚滑6.062541mm/frame、Eq6 .014099973；32条关系接触44.868561mm、29/32优于换shape初始化、2cm比例71.7132%。调用均值4.276852s、3/25>5s。相较weight1的84%/106.436mm，接触大幅改善但物理通过下降；不能晋级。weight1000及原session40887继续，三例修订队列76353等其结束。
- **09-12 18:00 UTC修订验证已排队：** 三例互相动作提案每entry已批准diagnostic-only，主agent直接读取文件经真实reviewed_contacts校验20299 exit0，SHA `1baa592c946c84a21702f8e4db04ca26e360b37bd5e801cefd1a014c0d21edf7`。GPU0队列session76353等待25例权重实验父PID27113结束且两份最终评分存在，再对offset57/91/94逐例control/corrected优化、独立物理评分、共同正确标签contact评分；不改缓存/timing。权重session40887仍待完整终态，weight100生成report已有，评分尚无最终文件，不当作失败。
- **09-12吸引尺度验证有实质进展：** 正确坐腿weight1000/50步优化86495、共同contact54943、物理82579均exit0。臀腿40.746→10.529mm、臂手30.458→4.831mm；脚滑1.920611mm/frame高于原shape1.230146，单例不通过，Eq6 .060804433仍低于原shape .275296351。说明不必200步也能拉近接触，但存在物理权衡。已启动与旧lr消融同25例/同floor/50步/.001的weight100与1000配对及完整评分，不因一例失败直接宣判全体；原标注文本问题仍独立保留。
- **09-12 17:55 UTC梯度诊断完成：** checker可选gradient-audit在epoch0/24/49记录真实加权项梯度，测试19788、实样本1721均exit0。epoch49身体姿态地面梯度范数P1/P2为28.101/40.295，吸引仅.0084/.0300，方向余弦约.001/0；提示尺度悬殊，不是强反向抵消，但尚不能从总体范数断言Adam每坐标更新原因。地面配置不动，按实测数量级启动独立weight1000/50步试验，物理/接触需联合评分。
- **09-12本轮诊断完成：** 逐关系时段weight10实验40826、共同评分44015、物理90038均exit0；臀腿40.417mm、臂手28.797mm，比同weight10无重分权40.243/26.925差，暂不采用。逐关系loss40219 exit0：newshape臀腿吸引.104536、臂手.082241，臀腿不是吸引项整体为零。已排除本例“只需更多步数”“只修配对/时段”“臀腿winner选错导致40mm”“臀腿吸引整体消失”几种单一解释；下一步用实际梯度判断损失竞争和有限步长，不动已固定的地面方案。
- **09-12 17:49 UTC区域/时段定位：** 独立region诊断28714 exit0，小测试10562 exit0。坐腿newshape获胜[9,19]时段均距42.211mm，整个臀腿union40.746mm，约1.5mm差距不是40mm间隙主因。原动作crop22首个有效帧臀腿32.492mm但臂手225.769mm；统一时段确实给尚未靠近的手臂施加吸引。复用已有原几何relative timing=.03，保持旧总mask，仅逐关系分权；同weight10/50步/.001实验40826已启动，尚未评完。
- **09-12低频生成复核：** agent实际ps确认12484 Rl、评测队列12632及父12631存活，剩余分片2182/6138、complete=false，累计2282/6238仅生成数，不是通过率。继续原6495/36631，不重复启动。
- **09-12 17:45 UTC吸引权重实测完成：** 50步weight10优化85506、共同contact66753、独立物理75213均exit0；正确坐腿标签臀腿40.746→40.243mm，仅微改善，臂手30.458→26.925mm改善。脚滑1.204743mm/frame、Eq6 .003421699均不高于原shape，单例物理通过；调用6.614s不达5s。说明加强吸引对手臂有效，但不能解决本例坐腿间隙，不能推广为全量方案。下一步检查实际选中region及逐关系时段，不继续只堆步数；生产缓存和地面保持不变。
- **09-12吸引定位新结果：** 正确坐腿配对100步39900、200步3480均exit0，共同contact43747 exit0；同样lr.001与合理地面，臀腿距离50/100/200步41.468/40.342/40.141mm，未恢复原shape4.733mm。200步日志两关系均锁定失败；代码仍置is_anchored=True，epoch101起未锁定关系不再吸引。不是50步弱吸引的解释，也不能靠延长步数解决本例。诊断入口新增独立attraction-weight，测试90650 exit0；50步weight10实验85506运行，生产runner/loss/地面不改。
- **09-12 17:40 UTC新增完成：** 原MMM生成49378与独立评分8970均exit0；同25例实际shape/物理输入、原生200步/.01，严格脚滑+Eq6交集8/25（32%）、optimizer-call均值28.513s。当前50步/.001为21/25（84%）、4.284s，但接触距离原生48.317mm明显优于当前106.436mm，不能称吸引质量已保留。坐腿修订优化/物理评分61010、共同contact评分4866均exit0；修正配对后物理通过，但正确标签下坐腿距离40.746→41.468mm，吸引仍未达标。详见`docs/shape_pair_repair_results_20260912.md`。
- **最新用户澄清/授权：** 接触时段历史上按配对距离自动划分，基本正确但可能随错误配对而错；此前文档/代码中“manual timing”不能解释为人工逐事件真值。用户授权GPT subagent修复配对，已交已有低成本GPT agent生成4例独立修订提案，不覆盖运行缓存；主agent复核后仅重算受影响关系时段并小批量比较，保留基本正确部分，不全量推翻。未知kick目标不补造stomach标签。
- **三臂lr配对完成：** session69249 exit0。同25例/50步/冻结当前物理方案，主agent复算.001/.003/.01联合脚滑+独立Eq6无容忍通过21/11/0（分母25），优化脚滑均值5.671/6.100/8.735mm/frame，接触距离106.436/101.082/100.152mm（32关系等权）。只说明当前方案50步下大lr损害脚滑，不能替代原算法结论。原MMM原生200步/.01基线session49378已在GPU0启动，目录`output/ssp62_native_mmm_lr001_iter200_first25_20260912`（名称lr001表示.01，实际report以.01为准）；原算法评分待完成。
- **恢复已越过旧断点：** 17:15 UTC实查PID12484 Rl，剩余分片1803/6138、complete=false，已超过中断前1739；累计1903/6238为生成数不是通过数。lr串行session69249确认仍运行，.003优化已有实际输出，未把缺少assessment文件误判为任务退出。8例恢复后的哈希另行复核。
- **新增已证实语义问题：** G013T007A030R001的exact绑定caption说右手扇脸、被打者右腿反踢；实际contact pickle却是P1左手→P2头及P1右腿→P2腹。左右手与跨关系施动者不符，全局角色交换不能修复。文本哈希绑定不等于标注正确，不能将距离下降或74%物理通过视为语义验收。保持运行缓存不变，候选不晋级；详见`docs/shape_text_contact_review_20260912.md`。
- **学习率正式配对补评已启动：** floor25 session21034 exit0；GPU0空闲核验后session69249串行运行同25例、相同shape/seed1234/50步/模型复用/地面配置的lr .001/.003/.01，每臂后接完整crop独立Volume+脚滑+三阶段contact评分，任何失败即停。输出`output/ssp62_matched_lr{0.001,0.003,0.01}_iter50_first25_20260912`及对应assessment；这不是原算法200步基线，后者仍待补。CPU/双GPU并发可能影响耗时，不把本轮时间直接当无负载速度。文本审查清单由低成本agent提取，视觉动作正确性仍未验收。
- **09-12恢复校验异常与新增验收要求：** 续跑98339因sidecar JSON损坏exit1，评测队列54477亦exit1。1739登记样本中1731双NPZ哈希匹配，8个sidecar损坏（7空、1无效）；8例sidecar与16个未经哈希核验NPZ已原目录改名保留为`.corrupt_20260912`/`.unverified_20260912`，未删除。重新运行原入口只重算缺失项，日志`output/ssp62_remaining6138_recover8_20260912.log`，终态待收。用户新增：必须补lr=.01同协议对照、原算法基线、contact mismatch三阶段改善与文本角色/部位/动作/时序正确性检查；现全量仅候选，74%不代表这些新增要求已通过。
- **09-12 17:04 UTC恢复：** 旧生成4182906、队列4185539均已消失，旧session45170/54095均Unknown；机器uptime起点16:27:22，重启事实已确认，但14:08停止写入原因未定。剩余分片停在1739/6138，累计1839/6238。绑定源码hash全部一致，复用原入口校验后跳过完整产物（不加overwrite），续跑session98339/PID10970，日志`output/ssp62_remaining6138_resume_20260912_1704.log`；独立评测队列已对新PID重新排队。旧“运行”句柄均为历史，尚无全量通过率。
- **当前验收：** 生产100优化24286、独立评分84971、严格gate82940均exit0；实际输出联合通过74/100，双方换shape且标准脚滑和独立Volume Eq6均不高于原shape物理GT。调用均值4.159844s，样本处理均值4.412644s、20/100>5s。打包96986与全74例读回89670均exit0；全6238未完成，不启动第③项或新训练。
- **当前运行（09-12 12:03 UTC）：** 剩余6138例新物理坐标优化GPU1 session45170/PID4182906，目录`output/ssp62_production_physical_remaining6138_20260912`；冻结offset100/selection_seed20260911及已验收参数。前100例不重做；仅生成已启动，剩余质量评分/全量通过率未完成。运行期间不改runner及其依赖。
- **最新用户覆盖：** 地面只需合理范围，目的是正确共同平面与ground loss坐标，不追求毫米级完美。当前地面参数冻结，停止为1mm等自设阈值反复调参；转入固定标注接触吸引实际效果检查。73/100新坐标脚滑+nearest穿透代理交集已测，不冒充独立体积验收或生产完成。
- **最新地面试验：** ground200/temporal1/horizontal.5/50步完整100优化及独立评估已完成。固定支撑滑动不增42→75/100（2例无观测不计通过），标准脚滑73→79/100；支撑高度P95 1.857→2.403mm。最大入地0.625mm，平均optimizer-call5.444s、73/100>5s。候选保留但尚未地面+脚滑+人体穿透联合验收；不改吸引、生产路径或训练数据。
- **最新用户指令：开始第②项 shape augmentation。** 第①项以已报告的采样折中结果收束，不宣称严格质量全不降。先优化距离/穿透计算，再使用合法公开shape进行增强；验收必须对比原始shape GT，且确实换shape，原样fallback不算通过。
- **最新覆盖：用户明确一个一个问题来，先地面关系，再吸引优化。** 当前只诊断/修复地面坐标、ground loss与导出往返；暂停扩展接触/角色/吸引实验。地面未验收前不启动原算法配对优化或第③项。
- **已确认的地面问题：** 官方预处理在neutral beta0下逐人按关节最低点落地，再共同减P1首帧origin；不是双方各自root都归零。实际beta/gender改变脚底高度，共同平移不能补偿；关节最低点也不等于mesh脚底。旧70.52%proxy/200例67%交集仅是旧坐标协议诊断，不证明物理有效增强，不据此晋级或训练。
- **当前实验：** 全序列实际shape mesh floor恢复+共同origin导出+原shape脚部高度参考+连续高度衰减+竖直修正二阶平滑，ground200/temporal1/lr.001/50步。100例优化及独立支撑评估均完成；最大入地0.644mm，原支撑高度偏差中位/P95/最大0.574/1.857/16.737mm。双方支撑可观测且偏差≤1mm为63/100，仅地面诊断，不是增强通过率。优化调用均值5.397s、71/100>5s（含模型初始化，不含source/floor与评估），速度目标未达。正在复核时间突变实例；生产runner/loader尚未接入，不把诊断NPZ直接用于训练。
- **旧产物保留但暂停晋级：** 134例旧overlay未用于新VQ/MGT训练，原数据未覆盖；旧全6238已生成，Volume候选评分继续（shard0原session86163，shard1迁移session24091），最终旧gate仅诊断。原自动打包链因旧shard1人工中断不会成功收尾。第③④项未启动。

## 历史状态（以下不是当前运行列表）

- **最新验收：有/无shape原采样与排序加噪完整验证已完成。** 有shape session `55740` / monitor `1743` exit0；不追加探索，完整结果见 `docs/shape_no_shape_sampling_validation_20260911.md`。以下“运行/待评”为历史节点，不代表当前状态。
- **最新用户指令（09-11）：停止继续探索，优先验证有/无 shape checkpoint 的原采样 vs confidence_all 排序加噪。** 不再以任何单项退化直接淘汰候选；完整展示质量与 MM 权衡。
- weight-decay 训练 PID `4025534` 已按指令 SIGINT，原 session `79369` 回收 exit1（人工中断，不是新训练故障）；已存 checkpoint 保留，20k timer `1730` 已终止。旧 GPU0 10k 冻结评分 `48301`/monitor `1729` 不追加任务，允许收尾；此前暂停的训练不恢复。
- 有 shape：既有选定 260k（SHA `58b976b8cb579e4a687b9614789bbf088b73e4496eb6b6c06ef7190e80f0d072`）；无 shape：既有选定 300k。比较各 checkpoint 内采样改动，不把不同训练步数直接作 shape 单因素因果结论。共同 VQ175k、冻结544质量样本、100×11 MM、50步/T0.5/CFG3.5、requested shape、无contact guidance。
- 新入口 `tools/run_hhc_shape_confidence_validation.sh` 已 bash -n 通过，GPU1 session `55740` 已启动（尚待完整结果）：有shape原采样与全分支noise1，CPU joint32及valid_frames physical。无shape复用已完成同协议结果，先核验绑定；不重复昂贵生成。

- **用户已接受 VQ 质量“基本可以复现”，解除严格10项逐项不退化对探索的阻断；不声称 bitwise 或十项全优。** 当前 MModality 质量保持、采样/条件训练/token压缩/零shape在 scope，第②—④项后置。
- 联合文本 dropout MGT 已完成50k训练与冻结质量/MM评分：多样性明显提高，但 Full/Hands FID 退化，未晋级；原训练 PID `3537043` 已正常退出。
- span v2 已完成10k/20k/30k/40k/50k冻结配对评测：50k滑步明显退化且 Hands FID 变差，未晋级、不追加 MM、不延长训练。原 session `28378` 已在 step50000 validation 后 clean stop，exit0。
- root replay 的100×11 MM有增益，但主质量与三额外 seed 未过；posthoc token restoration、generated sliding gate、CFG4.5 均已完成且未晋级。
- circular-span v2 已到终态，不再写作“继续到下一验证里程碑”。max_probability sampler/exporter/MM metadata 已实现且默认行为不变；session `96986` 的 cascade reference、sampling trace、root replay 共12测试 exit0。下一步仅扩展既有 trace 工具做 control/max_probability 的8×4实模型诊断，尚未启动。

- 优先级：①质量不降地提高 MModality → ②公开 shape augmentation 与优化加速 → ③InterHuman 两部分 VQ → ④其他方法复现。
- 当前回到第①项 MModality 质量保持；MGT/采样恢复执行，第②—④项仍后置。shape beta factorial val16 已完成；四臂重复率 requested/requested `68/96`、MGT-zero `73/96`，没有改善信号。
- 已完成训练与评分：stride4/stride8 均完成 10000 步，配对训练 session `80349` 正常退出（exit 0）；stride8 10k 三项冻结样本评分均已核验。
- text-dropout 50k 冻结质量与MM均已完成；不再延长该训练。当前只继续 circular-span v2 到下一验证里程碑。

- shape 两端 beta policy 拆分代码及测试已完成，未改训练源码；factorial 仅为16 prompt×4 repeat 小试验，不是正式 MM 评测或质量结论。

## 最近完成

### 09-12｜100例完整评分终态与严格结果

- 主线程50170 exit0，最终`output/ssp62_matched_attract30_iter50_first100_assessment_20260912.json`100行；主agent核对名字顺序与weight1生产100一致，原shape所有Volume指标逐值相同。200NPZ先前已独立读回验证finite、双方donor betas精确且均改变。
- weight1/weight30同100例联合74/60，脚滑80/75、Eq6 92/79分别不增；均值脚滑5.767546/5.876432mm/frame、Eq6 .021187844/.028094015m²/frame，均低于原shape但逐例失败不能省略。原要求严格双不增，不加epsilon、不用均值代替逐例计数。
- weight30接触共124条有效关系（零时段/无关系不造评分）：原shape111.473877mm、换shape114.287273、优化66.619748；113/124相对初始化改善，2cm接近帧比例原45.6492%、换shape40.9391%、优化68.0390%。接近不是人工语义真值；已发现错角色/时段问题仍独立待修。
- 平均调用4.209581s、6/100>5s，不含准备与评分；通过率从25例68%降为100例60%，说明扩样有必要。暂不据此宣布全量目标完成或原接触质量无损。

### 09-12｜拍背逐关系伪时段实测完成，权重30扩到100例

- 84019完整优化/独立物理/共同contact全部exit0，真实CLI验证了证据SHA、角色绑定及现有逐关系loss接入。用同一原几何伪时段重评全部条件：原shape两关系4.656001/3.125646mm，换shape8.987710/7.165582，旧global时段优化12.710342/21.781703，新逐关系优化7.008146/17.904074。
- 新时段较旧优化改善，但第二关系仍劣于换shape未优化，不能称完全恢复接触。脚滑5.344231mm/frame、独立Eq6 .007840368m²/frame不高于原shape5.429719/.042002448；只单例诊断，源缓存/ground不变，不全量采用。
- 权重30的25例已达到68%与4.275s均值且接触改善，扩大到100例是必要验收节点。session50170串行优化+评分，冻结当前所有依赖，不再改checker/scorer直至终态。25例小试不是全量成功，也不启动第③项或新训练。

### 09-12｜权重30的25例完整评分完成

- 原session77615收exit0。主agentassert25名字、原shape完整Volume指标、contact/timing SHA与weight1基线一致，读取最终`output/ssp62_matched_attract30_iter50_first25_assessment_20260912.json`。
- 严格脚滑+独立Eq6不增17/25=68%，双方正观测，无容忍；脚滑20/25、Eq6 21/25分别不增，均值5.851390mm/frame/.008749772m²/frame。接触32关系58.822080mm，相对换shape初始化29/32改善，2cm帧比例60.0572%。调用均值4.275073s、3/25>5s，含初始化与并发，不含独立评测。
- 这是目前小批量速度/通过率/接触的折中候选，不把25例推断为全量稳定达标，也不宣称优于原MMM的每项接触质量。需扩大100例同协议评测、保留已知标注问题的独立状态，再决定生产更新。

### 09-12｜独立逐关系几何时段诊断实现

- 主线程队列session84019，等待weight30父PID28621结束后执行；当前排队，完整终态由主agent回收。
- 按Ponytail复用已有优化/导出/独立评分，仅新增小入口。严格限定单例<=100步，校验evidence状态、名字/sidecar/review/timing SHA、绑定输出文件SHA与floor SHA、实际旧pickle/caption/修正contacts、原geometry数组维度；全局角色仍由原初始化确定，必须等于evidence绑定角色才安装逐关系权重。没有几何支持的关系拒绝，而非造事件。
- 权重定义为clip((.03-original_distance)/.02,0,1)，不是历史标签阈值或人工真值；仅接触语义逐关系时段改变，auxiliary/global mask保持旧值，合理地面配置不变。报告保存完整权重、evidence SHA及入口SHA，源缓存不覆盖。
- `tests/test_geometry_relation_times.py`41889 exit0覆盖线性权重、负/NaN/空输入和无支持事件拒绝。真实CLI与角色/几何绑定路径待排队实测，不能以小测试代替实样本验收。
- 队列依次优化拍背offset94、独立物理评分、同正确标签全局时段评分+逐帧输出；后续用同一原几何伪时段对control/new两条件再算同口径，不能以变更评分时段制造改进。输出`output/ssp62_backpat_geometry_times_20260912`及`..._physical_20260912.json`、`..._common_contact_20260912.json`。

### 09-12｜三例修订完成及拍背逐关系时段证据

- 76353 exit0；互抓臂G001T003A019R024正确标签两距离：原shape29.848/127.244mm，换shape23.904/130.085，旧配对优化25.210/141.738，修正23.651/139.428。新旧均标准脚滑/Eq6不高于原shape（修正8.587405mm/frame、.002572743，原9.208356/.031715627）。第二关系原本也较远，全crop最小51.931mm；不能仅凭caption强行认作已贴合。
- 互拍背G053T002A011R007：原shape35.104/167.693mm，换shape34.598/164.706，旧优化23.979/176.748，修正21.176/175.722。修正脚滑5.371252mm/frame、Eq6 .008891489（原5.429719/.042002448），物理通过但第二接触尚未恢复。上述接触均按同一旧global mask，不把小幅改善视作语义已验收。
- 互拍背原geometry第一/第二关系最近1.089mm(source89)/1.059mm(source128)。旧mask仅source84–91、109–110、117–119、125–127共16帧；第一关系<1cm36帧，其中23在mask外，第二关系<1cm18帧，其中10在mask外。原mesh拟合接近不等于人工真值，但已足以支持下一步独立逐关系时段伪标注诊断，不复用一个global mask强拉两段动作。
- 全零时段打脸例原geometry74088 exit0：P1右手→P2头49.621mm/source35、仅1帧<5cm；P2右手→P1头85.575mm/source11、无<5cm帧，左手均>500mm。不能把旧mask全0直接叫漏标，更不能为了产生接触而人为补帧；该例暂不修timing。

### 09-12｜空时段首例与拥抱第99例补查

- 第一例G045T002A010R003原timing完整88帧全0，crop0:88；其共同contact结果所有条件fixed_regions=[]。原/修订两优化输出物理指标相同，不代表修订正确性已验收，也不计接触改善。后续要查原几何是否真有接近，不凭caption造接触时段。其余两例G001T003A019R024 crop21:171、131正帧，G053T002A011R007 crop16:166、16正帧，具备进一步接触对照条件。
- 另完成pos99 G019T006A000R023可视化8469 exit0，`output/ssp62_text_visual_hug99_20260912`。原动作可见红P2双臂靠蓝P1头部，稀疏骨架不能证明蓝P1拍红P2背；文本说另一人拍背，原两关系head→arms/back→hands从语义上仍同一施动者，统一全局交换不能修复。agent再次把逐关系部位合理误当跨关系角色合理，root未采纳。
- 原shape实际mesh全crop几何16377 exit0：P1头→P2臂最小.862mm/source64、<5cm帧40–100；P1手→P2背最小63.189mm/source60、无<5cm帧；旧P1背→P2手最小.205mm/source58、<5cm帧40–98。5cm只是接近诊断阈值，不能证明拍背绝对不存在；但几何支持旧第二关系可能属于同一个拥抱，而非caption中另一人拍背。此例保持文本/动作一致性未决，暂不把纯文本反向修订写入缓存。

### 09-12｜权重1000完成，强吸引不能直接晋级

- 原串行session40887收取exit0；主agent核对weight1000与基线floor SHA、25名字、原shape全部Volume指标、contact/timing SHA一致。最终报告`output/ssp62_matched_attract1000_iter50_first25_assessment_20260912.json`，未混用partial。
- weight1/100/1000同25例严格联合通过21/14/7（84%/56%/28%），接触32关系等权106.435756/44.868561/31.757503mm，调用均值4.284014/4.276852/4.273481s。权重1000脚滑11/25、Eq6 14/25不增，脚滑均值6.115023mm/frame、Eq6 .049931418m²/frame；均值优于原版不代表逐例通过。
- weight1000接触31/32相对换shape初始化改善，2cm比例82.4919%，但物理验收失败率72%，不用于全量augmentation。不以更高接触分数替代用户的脚滑/穿透双不增要求。
- 等待已有三例修订与weight30队列；若中间点仍不能同时兼顾，下一步需改变吸引更新方式而非继续加大固定权重。此为后续决策，不意味着已实现或缩小目标。

### 09-12｜权重100失败类型复核

- weight30队列主线程session77615，等待PID27739结束且三份最终共同contact报告齐全；当前排队，终态待主agent回收，不能误作已运行/已评完。
- 失败11/25：仅脚滑5、仅独立Eq6 2、两者4；不能只针对脚滑而忽视强吸引导致的人体穿透。
- 按weight100物理结果固定分组、同一组对比weight1/100：通过14例19关系接触77.997642→36.251926mm；失败11例13关系147.999153→57.462104mm。此分组只是解释，不缩小成功率分母，也不作后验筛选后宣称60%达标。
- 选择中间权重30作一次约束折中验证，排在原三例修订队列之后，不抢占GPU；原所有帧/地面/shape/seed/50步/.001与25例基线一致，完整独立Volume+脚滑+contact验收，输出`output/ssp62_matched_attract30_iter50_first25_20260912`及对应assessment。需真实结果后才决定扩到100例，不改变>60%/5s目标。

### 09-12｜权重100的25例完整结果

- 主agent读取最终`output/ssp62_matched_attract100_iter50_first25_assessment_20260912.json`，assert25名字顺序、original Volume全部指标、contact/timing SHA与旧weight1同25例逐值一致；未用partial凑结论。
- 原shape均值脚滑6.279881mm/frame、Eq6 .126672192；weight100为6.062541/.014099973。逐例脚滑不增16/25、Eq6不增19/25、交集14/25=56%，双方有正接触观测，不加容忍。均值改善不代表逐例达标。
- 接触32条等权最小距离：换shape未优化117.850619mm、weight1 106.435756、weight100 44.868561；29/32严格改善，2cm接近比例71.7132%、5cm79.1094%。旧缓存语义仍有独立风险，不等同真实接触正确率。
- 调用均值4.276852s、中位4.528991s、3/25>5；含初始化与并发负载，不含完整评测或源数据准备。原生MMM200步/.01同25例为32%/48.316602mm/28.512935s，不能把不同loss/步数比较当单一权重因果。
- 当前weight100未到>60%物理目标，weight1000尚待评分。只收一个完成节点，不重新启动/打断串行40887。

### 09-12 18:00 UTC｜三例配对修订真实加载通过，隔离实验排队

- G045T002A010R003互相右掌打脸、G001T003A019R024互相抓左右臂、G053T002A011R007互相拍背：每条修订保持原第一关系，仅第二关系按一致人物归属反向。冻结review SHA `1baa592c946c84a21702f8e4db04ca26e360b37bd5e801cefd1a014c0d21edf7`，今后不可在队列运行中修改。
- 低成本agent第一次误把状态放入额外映射而非entries，主agent拒绝临时注入状态的校验，最终亲自apply_patch写入每entry并删除冗余映射。20299直接以实际文件entry、原pickle SHA/内容及生产caption调用现有helper，3/3全部exit0；并未覆盖原缓存。
- 主agent核对manifest offset57/91/94准确。session76353在PID27113后依次执行每例旧/新配对50步/.001/attract1/固定地面及同shape，同一修正标签固定初始化角色共同评分，任何失败即停。输出`output/ssp62_reciprocal_{source}_{control,corrected}_20260912`、各自`_physical.json`及`output/ssp62_reciprocal_{source}_common_contact_20260912.json`。仍是诊断，不将文本配对校验当成运动语义已通过。
- 当前等待原25例权重实验评分，不重复启动、不改变其运行依赖；session40887/76353终态由主agent回收。

### 09-12｜强吸引验证与25例配对启动

- 主线程串行句柄session40887，终态待回收；不能以尚无最终assessment文件判断退出，不重复启动。
- weight1000实样本86495/54943/82579均exit0；共同正确标签两关系距离10.529379/4.831461mm，2cm接近帧比例91.453%/95.726%。相对初始化40.745798/30.457918是显著改善，不是此前毫米以下数值波动。原shape距离4.733487/15.887650mm，臀腿仍未完全恢复。
- 物理脚滑1.920611mm/frame（原1.230146），Eq6 .060804433m²/frame（原.275296351），该例联合不通过；地面最小Y P1=-.987mm、P2=2.864mm，未更改共同地面方案。调用6.617s含初始化，不能作为5s达标。
- 下一组已有入口GPU0串行weight100/1000，25例与旧lr.001/50步baseline一致，floor`ssp62_lr_ablation_production_floor25_20260912.json`，地面200/1/.5不变、模型复用、seed1234；每臂后接独立Volume+脚滑+同原标签contact。输出`output/ssp62_matched_attract{100,1000}_iter50_first25_20260912`及对应assessment，任一失败即停；不启动训练、不改生产核心。

### 09-12 17:55 UTC｜真实梯度与重跑数值波动

- 按Ponytail仅复用checker工厂注册可选forward hook，autograd.grad(retain_graph=True)读取加权attract/pene/ground/skate/reg/smooth/acc/total，assert分项和等于total；记录P1/P2各参数梯度范数与对吸引的cosine。测试19788 exit0验证不会写入.grad且后续backward正确；低成本agent独立核对epoch参数位置、权重覆盖通过。
- 实样本1721 exit0：epoch0身体pose ground/attract范数P1=21.404/.0090、P2=22.791/.0161；epoch49=28.101/.0084、40.295/.0300，total对吸引方向cosine接近0。位移epoch49 ground55.113/61.701、attract.0497/.0805。尚无每坐标Adam步贡献证据，不把范数差直接视为最终因果证明。
- 诊断输出与旧基线不逐值一致，P1位移最大差1.156mm、body axis-angle分量差.003882rad。无诊断重跑3081 exit0也与旧基线不同：P1位移1.196mm、P2 body最大分量.005269rad。实证存在运行波动；尚未证明诊断完全不影响数值轨迹，不声称bitwise。此前.2mm级单例差异不宜判断参数优劣。
- 新weight1000试验是依据实测梯度比值检查接触能否驱动姿态，不修改合理地面或生产loss，也不是全量推荐。所有diagnostic计时含额外backward，不用于5s验收。

### 09-12｜时段试验未晋级，逐关系吸引不为零

- 相同weight10/50步/.001，只加入原几何relative timing=.03：旧全局mask共同评分臀腿40.416871/臂手28.797002mm；基线40.243356/26.925450。用同一原几何伪时段给所有条件重评，试验40.657020/7.941729，基线40.475844/7.672854mm，仍未优。没有用换评分口径制造改进。
- 试验标准脚滑1.213095mm/frame、独立Eq6 .003691610m²/frame仍不高于原shape，但比无reltime的1.204743/.003421699更差。优化40826、评分44015/90038均exit0，输出`output/ssp62_sit_reltime003_{common_contact,physical}_20260912.json`。不采用此参数、不全量改时段。
- 复用评分工具新增可选逐关系吸引输出，并assert单关系均值与整组损失一致，实样本40219 exit0。newshape臀腿.104536/臂手.082241；原shape.001562/.048465；试验输出.105305/.073857。不能把臀腿间隙归因于该关系整体被signed ReLU截为零；尚未检查每个顶点/参数的实际梯度，不能过度排除局部梯度问题。
- 文件`output/ssp62_sit_relation_attraction_20260912.json`；该工具随后仅补空contacts诊断均值为0的边界防护，生产依赖未动。下一步检查真实加权梯度方向/量级，以决定接触参数化或优化调度，不继续盲扫单例权重。

### 09-12｜获胜region与自动时段诊断

- Ponytail复用共同评分工具增加可选`--region-diagnostics`，记录离线重新选择的region及逐帧unsigned距离，不冒充整个优化轨迹；新增小测试验证获胜区域与union区别、缺失winner安全输出，10562 exit0。生产loss/runner/缓存不改。
- 28714输出`output/ssp62_sit_region_diagnostics_20260912.json`：正确标签newshape臀腿winner[9,19]均距42.211、union40.746mm；原shapewinner[9,46]均距8.523、union4.733mm；weight10输出winner41.714、union40.243mm。手臂winner区域比union偏差较大，但臀腿区域竞争不是本例间隙主因。
- 用户说明的旧自动mask文件182帧，crop10:160；117个正帧位于crop22–38及42–141（闭区间），首argmax22。主agent从原几何逐帧报告复算首帧臀腿32.492/臂手225.769mm；有效帧内臀腿全部117帧<5cm，臂手104帧<5cm。时段整体基本合理但两关系不应全程同权。
- 既有relative_relation_timings(.03m)仅在原mask内重分权，原几何有效权重和臀腿109.826/臂手101.495（旧均117）；首帧手臂权重.001592，臀腿1。不是人工标签，也不能恢复旧mask外漏事件。冻结weight10/50步/地面/lr.001，仅打开该既有选项，40826待终态和共同contact/物理评分。

### 09-12 17:45 UTC｜weight10单例完成

- 同50步/lr.001，仅attract1→10；共同正确标签臀腿41.468→40.243mm、臂手34.371→26.925mm（两者均指优化输出比较）。相对换shape初始40.746/30.458mm，主要收益在臂手；臀腿仍远于原shape4.733mm。
- 标准脚滑1.204743mm/frame、独立Volume Eq6 .003421699m²/frame，相比原shape1.230146/.275296351均不增；相较weight1的1.165885/.000318749有退化，不隐瞒权衡。50步调用6.614s（单例含初始化，不是25例均值），未达5s。
- 新权重测试90650 exit0。旧review测试直接执行缺PYTHONPATH时报ModuleNotFoundError(exit1)，以项目根PYTHONPATH重跑13923 exit0；不是测试断言失败。生产核心依赖未修改。

### 09-12｜坐腿步数诊断完成，单因素吸引权重试验

- 正确标签臀腿/臂手距离：50步41.468/34.371mm；100步40.342/32.825；200步40.141/36.083。原shape4.733/15.888；换shape未优化40.746/30.458。没有恢复接触。
- 独立物理评分第一次42371 exit1：误将同名样本两条件传给只接收不重复样本的scorer，assert正确拒绝。改为每条件独立调用26421 exit0：100步脚滑1.066648mm/frame、Eq6 .000350040；200步.958592/.000205644，均不高于原shape1.230146/.275296351。失败日志保留，不改scorer。
- 源码主agent核对：semantic未锁定吸引除时段权重和，锁定后平方距离未做该归一化；epoch100先算未锁定loss再freeze，101起才用锁定。200步日志两关系FAILED后仍整体is_anchored=True，未锁定关系被跳过。低成本agent辅助核对的行号与数值示例有误，不采用其数值尺度比。
- 按Ponytail复用独立checker工厂，仅新增正有限attraction-weight（默认1不变）及小测试，90650 exit0。weight10/50步/.001/同shape和正确配对单因素实验85506，输出`output/ssp62_sit_pair_corrected_attract10_20260912`；需共同contact和独立物理评分后判断，不按loss下降验收。

### 09-12 17:40 UTC｜原生基线、坐腿修订实测与互相动作提案

- 原MMM25例脚滑不增10、独立Volume Eq6不增20、交集8；脚滑6.279881→7.027359mm/frame，Eq6 .126672192→.016583884m²/frame。32条原标签接触距离换shape117.850619→48.316602mm，26条改善、2cm接近比例71.4766%。这是真原算法/同物理输入，不是原示例预处理全流程复刻；28.513s含原生模型初始化及并发负载。
- 坐腿G026T001A009R022：control/corrected均脚滑+Eq6不高于原shape；正确标签两关系距离原shape4.733/15.888mm，换shape40.746/30.458，control41.497/34.427，corrected41.468/34.371。仅修配对几乎不改变结果，尚未恢复接触，不能把修订合法或物理通过当作语义验收。
- root实际复核前100 caption/pickle发现额外互相动作角色错误pos57/91/94；低成本GPT生成独立`docs/contact_pair_corrections_reciprocal_20260912.json`，root原pickle SHA、old_contacts、caption和mapper逐值assert全部exit0。提案未应用、不改原缓存/timing。pos51两次踩脚允许一致全局角色，撤回agent误报；pos99拥抱/拍背疑似角色错误需视觉确认，pos67空配对需检查是否实际避开。
- 下一步固定合理地面，隔离检查吸引未恢复原因及受影响逐关系自动时段；基本正确缓存不整体推翻，全量候选仍不用于训练。

### 09-12｜扩展前100文本配对审查

- 主agent实际查gate：已证实存在标注问题的G013T007A030R001与G026T001A009R022均在旧74例物理通过名单。因此74%确实不是语义通过率，已有pilot overlay仍不得用于新训练。part0 agent初稿把50条全列待审，未完成有效语义初筛，已退回；不采纳其0错误计数。
- 两个低成本GPT agent分别检查冻结生产100的0–49/50–99，输出`docs/contact_review_first100_part{0,1}_20260912.md`；只作带证据初筛，允许统一角色置换，明确矛盾与歧义分开，主agent验收，不将初筛数直接当错误率。运行缓存和已批准坐腿提案均冻结不动。
- 原MMM session49378本轮仍运行，尚无终态；继续原session与既有评分队列，不另起重复基线。

### 09-12｜坐腿共同接触评分接入队列

- 新独立`tools/score_reviewed_contact_comparison.py`复用既有几何、共同origin解码、fixed_region_contact与已审修订校验；旧/修正各自一套标签，每套仅在new-shape初始化选一次角色，固定评价original/new/control/corrected四条件。旧标签评分与正确标签评分分开，不优化后重新选角色、不修改原独立Volume scorer。
- 坐腿优化+物理评分队列session61010/PID20024后接CPU共同接触评分，输出`output/ssp62_sit_pair_common_contact_assessment_20260912.json`。完整实样本运行检查待依赖完成；未称评分工具已实测通过。原MMM优化session49378本轮仍运行。

### 09-12｜坐腿配对修订通过绑定检查，隔离对照排队

- 主agent重跑4例原pickle SHA/old_contacts/new映射逐值校验全部通过。仅坐腿G026T001A009R022批准`approved_for_diagnostic_only`；R001缺kick目标保持禁止应用。
- 复用独立checker新增可选`--contact-review`，校验批准状态、未禁用、caption/旧pickle SHA/旧contacts匹配、新部位ID与权重精确映射；不修改生产runner、loss、缓存或timing。新增`tests/test_reviewed_contact_override.py` session1568 exit0，覆盖批准路径及禁止/未批准/hash/caption/空关系/不完整kick拒绝。
- 等原MMM评分队列PID17971退出且有最终报告后，在GPU0依次重跑坐腿control/corrected（同offset31、同shape、原timing、50步/.001/地面配置），每臂独立脚滑/Volume评分。输出`output/ssp62_sit_pair_{control,corrected}_20260912`与physical报告；正确配对下的共同contact评估还需补，不能用旧标注分数验收修订。

### 09-12｜修订提案复核与反踢目标几何定位

- 主agent实际运行逐值检查：4例原pickle SHA均匹配，但GPT提案中R001/R003旧权重小数、R003新权重与mapper不精确一致，已要求修正并assert；尚未应用。R001单独省略kick不能冒称完整修复，需保留明确未解决状态。
- 原shape右脚最近目标定位session62781 exit0：124–131帧对应P2左手/左臂顶点，非腹部；可能阻挡，几何接近不是攻击意图。后续只能明确标为原几何接触保持关系，不能造新文本目标。
- 历史timing源追踪未找到生成脚本：当前运行时joint-distance函数及默认/显式参数不证明旧npy生成规则。沿用用户“自动距离划分”的来源说明，保留旧基本正确标签，新增受影响关系时段需另标几何推断。

### 09-12｜原MMM评分排队；历史timing来源继续核查

- 原生baseline PID17391实查运行，独立评分队列session8970等待该PID退出后调用既有完整crop脚滑/nearest/Volume/contact scorer，输出`output/ssp62_native_mmm_lr001_iter200_first25_assessment_20260912.json`。生成session49378与评分8970终态均待主agent回收。
- 输入绑定检查exit0：原生准备与三lr臂同25名字、同sidecar SHA、同floor SHA；三lr臂optimizer config除learning_rate外逐值相同。
- 发现plan字段`manual_frame_annotation_file`仅为历史命名，实际文件`/home/wenning/projects/MMM/dataset/interx/contact/G013T007A030R001.npy`，按用户澄清视为自动距离标签。当前通用函数默认.1/.3阈值不能直接认作该历史文件生成参数，已交独立低成本agent查原保存脚本，未改时段。

### 09-12｜同协议学习率消融完成，原生MMM基线启动

- 三臂完整独立评分均25例，session69249 exit0。lr .001/.003/.01：脚滑不增22/12/1，Eq6不增24/24/24，交集21/11/0；均值脚滑5.670683/6.099710/8.735482mm/frame，Eq6 .003288767/.002150782/.001377224m²/frame。原shape均值脚滑6.279881mm/frame、Eq6 .126672192。
- 接触32关系等权最小距离106.435756/101.081502/100.152430mm；相对换shape未优化严格改善20/21/20条，2cm接近帧比例44.2237/47.6588/45.4550%。optimizer-call均值4.284014/4.346094/4.284968s（有并发负载，不用作纯算子提速对比）。接触近一点不能抵消逐例脚滑验收失败；标注语义问题独立待修。
- 新隔离工具`tools/run_native_mmm_shape_baseline.py`仅准备输入与调用原仓库函数，不改原loss/BodyModel。准备session85400 exit0：25例实际物理输入/shape/timing NPZ逐项读回与哈希通过。worker阶段只导入MMM依赖并assert路径，200步/.01/seed1234，输出保留物理world坐标（decode offset0），session49378。后续用同一独立scorer，不把原示例joint-floor预处理作为输入差异混入。

### 09-12｜原仓库基线独立导入验证通过

- 在`/home/wenning/projects/MMM`工作目录执行只读import，session32442 exit0，实际optimize_full_body、DyadicInteractionLoss、BodyModel三个来源均为MMM原仓库，非HHC替代。可以用隔离进程直接跑原代码；后续准备相同物理输入与200步/.01配置，保留原损失与模型实现，避免将HHC加速实现冒称原仓库耗时。尚未启动该GPU基线。
- 逐关系时段现有MeshGroundSupportLoss已有实验内部路径，但它目前只在原manual mask内分权，不能自动修复R001 mask外踢击。修订小实验须显式区分人工文本修正与几何伪时段；当前不改生产依赖。

### 09-12｜R001逐帧几何证实右手/反向右脚事件及timing缺口

- CPU只读几何session49636 exit0，原shape/fullsequence mesh floor，逐帧cKDTree区域最近顶点距离：P2右手→P1头最小0.111mm、5cm接近26–103帧，左手最小617mm；P1右脚→P2全身最小1.524mm、5cm接近124–131帧，全部在原timing26–103外。支持原文本右手及另一人右脚的顺序，确认缓存左右/角色错误，并证实timing未覆盖后续几何接近。whole-body目标不能证明踢中腹部，仍不造stomach标签。
- 后续应保留正确角色并按关系拆时段，不能简单扩大统一mask使扇脸和踢人同时生效；当前原缓存保持冻结，lr实验用于原协议对照而非语义已修复。

### 09-12｜8例损坏样本恢复校验完成

- 低成本独立检查exit0：8例sidecar均可解析，16个NPZ哈希均匹配，各自旧.corrupt与.unverified备份仍存在。恢复只证明产物完整性，不代表这些样本物理或语义通过。原续跑继续处理剩余数据，未修改缓存。

### 09-12｜坐腿样本跨关系角色不一致；避免把抬腿直接当漏接触

- 主agent查看另外两例图并实际读G026T001A009R022 pickle：butt(P1)→legs(P2)，随后hands(P1)→arms(P2)，与caption“坐腿者由另一人手扶上臂”角色组合不符；图中蓝人坐向红人腿、红手靠近蓝上臂，支持该解读但不证明mesh接触。不调整缓存或用lr强行满足。
- 推腹R030图保留伸手与后退过程，稀疏图无法判断文本“fall backward”是否真实完成或哪只手，暂不判文本通过。R001帧119抬腿处于timing外，仅说明应核查事件范围，不足以直接断言实际脚部接触漏标。

### 09-12｜新lr=.001配对25例评分完成；接触改善不是全通过

- 独立完整assessment已写出。主agent直接按Volume报告的样本脚滑（观测加权）+Eq6、<=无容忍与正脚观测复算：脚滑22/25不增，Eq6 24/25不增，交集21/25。原→优化脚滑6.279881→5.670683mm/frame，Eq6 .126672192→.003288767m²/frame；不是agent错误的23/25或逐人等权滑步。三臂串行session69249尚未终态，.003/.01仍待完成。
- 32条关系等权：原/换shape未优化/优化后最小距离115.894/117.851/106.436mm，2cm命中比例39.346/36.167/44.224%，5cm命中48.588/48.255/52.390%。20/32关系距离严格改善；不是32/32接触达成，也不是语义正确率。优化调用均值4.284s，存在并行负载。
- 重点3例图session25533 exit0。主agent查看R001两个视角，-60度可见红人伸手到蓝人头部、后续蓝人抬腿，支持文本的跨人事件切换；稀疏图不能判左右手或踢击真正接触。可见抬腿帧119已超该crop timing非零26–103，时段覆盖也需密集检查，不仅修部位。

### 09-12｜重点3例可视化启动；批次错误率未定

- floor100 session52635 exit0；以独立参考启动已发现矛盾R001、推腹R030、坐腿R022三例三阶段双视角图，输出`output/ssp62_text_visual_flagged3_20260912`，可视化未完成。
- 主agent复读32条完整caption+parsed contacts，否决agent初报18/32：它错误地把可由整组角色交换解释的反向关系算成错误，且分类重复。已要求撤回并按统一角色映射复核；现不报告该错误率。已确认R001之外R000左右手相反，A030R003互相拍脸却两条同向，A025R003肩臂及手臂两关系角色不一致；视觉真实性另查。

### 09-12｜文本标注错误溯源与影响范围检查

- G013T007A030R001错误已存在于batch `1bee17d10477b66d`保存的模型parsed_response，不是后续部位ID映射产生。role_ratio仅方向loss权重，不能补偿跨关系角色错误；主agent复读loss整组角色选择实现确认。低成本agent继续独立审查同batch32条，明确矛盾与歧义分开。
- 为8例重点动作可视化重算生产前100完整floor，CPU session52635、输出`output/ssp62_text_review_production_floor100_20260912.json`；未改变运行中生成或lr实验代码/缓存。此为扩展视觉检查准备，不是新增优化或通过率结果。

### 09-12｜文本动作可视化首例完成；原算法ground核对

- 新独立`tools/render_shape_contact_review.py`复用现有骨架绘图、SMPL-X几何和共同origin解码；对source/floor/sidecar/NPZ做哈希校验。首例G033T002A014R027、三条件/同6帧/两视角session11588 exit0，图在`output/ssp62_text_visual_first1_20260912`。主agent查看-60度图：均保留接近、交汇、分开的动作过程，但稀疏骨架不足以证明肩部碰撞，不能据此判文本完全正确；需要重点事件的密集mesh检查。
- 原MMM ground确实存在：loss权重20，relu(-vertsY)加入总loss。纠正只读agent把“无新ground reference参数”误说成“无ground loss”的判断；原算法基线应保留released ground，而非关闭地面项。lr对照69249继续使用原checker，无修改运行依赖。

### 09-12｜损坏产物保留后续跑；lr=.01对照准备

- 原入口恢复session6495/PID12484，评测按该PID重新排队，日志`ssp62_physical_remaining_assessment_recover8_20260912.log`。没有修改优化源码；完整样本仍逐项验证后跳过。
- 原物理floor报告所绑定completion已无法在现有顶层ssp目录匹配，不能直接冒用。以已验收生产100前25重算独立floor，session21034，输出`output/ssp62_lr_ablation_production_floor25_20260912.json`；接下来同入口/同25例运行.001/.003/.01、50步对照，独立Volume与contact评分后再判断。文本可视化正确性尚待检查。

### 09-12 14:00 UTC｜用户进度说明：累计生成1720/6238（27.6%）

- 实查剩余分片1620/6138、complete=false，生成PID4182906运行中；加已完成前100共1720/6238。评分队列4185539仍等待生成结束，未开始剩余全量评分。按当前吞吐粗估还需约5.5小时生成，不含独立评分。
- 当前是冻结地面与吸引参数后的批量执行，不再调学习率。前100严格联合通过74%，optimizer均值4.16s、样本处理均值4.41s（20/100超过5s）；全量通过率尚未知。之后自动重算地面、独立评分并筛选，完成第②项后才进入InterHuman。

### 09-12 13:25 UTC｜用户请求进度：全量生成累计1240/6238

- 主agent实查PID4182906 Rsl、运行1:23:12；剩余分片1140/6138、complete=false，结合前100累计1240/6238（19.9%）。队列4185539仍等待生成，未开始全量独立评分。按当前墙钟吞吐粗估剩余生成约6小时，不含后续评分，不保证完成时间。
- 当前等待计算而非新一轮调参；不把生成完成数算通过数。实际已验收仅前100的74例，全部6238通过率待评分；第③④项仍后置。

### 09-12｜姿态归一化峰值定位：小std放大，不等于大角度扭曲

- 同74例同cache crop只读CPU检查exit0：峰值69.96296位于G037T005A025R005__shapeaug_s00000123、plan crop帧50、P1 SMPL-X joint15、6D分量0（总维180）。原6D .99989927→优化 .99830866，冻结std仅2.3939674e-5；该关节实际旋转差约2.712°，原normalized -3.52056。因此该峰值主要由很小的历史std放大，不能直接解释为69倍物理旋转，也不据此保证模型无影响。
- 全74配对crop、两人55关节按帧关节池化旋转差：均值.42031°、P99 2.80837°、最大5.51086°（float32矩阵测量）。保持统计量和优化参数不变；未添加限制或新训练。生成4182906实测Rsl、后续评分队列4185539实测S，均非终态。

### 09-12｜全量独立评分已排队；完成配对归一化分布检查

- 生成PID4182906实测Rsl仍运行；新增单次串行`tools/queue_ssp62_physical_remaining_assessment.sh`（bash -n exit0，字段检查实际run_contract通过），队列session54095：等待现有PID退出→严格核验6138完成/位置100..6237→从原H5重算全序列floor→完整接触/nearest/Volume评分→严格gate。flock防重复、set-e失败即停、不重启生成、不覆盖报告、不自动训练；当前只是等待，不宣称评分已开始。队列日志`output/ssp62_physical_remaining_assessment_queue_20260912.log`。运行期间不改该脚本/评分源码。
- 主agent实际CPU同74接受样本配对计算exit0：源H5先取plan crop，再按相同augmented_name/seed1234/unit8取同一cache crop，原/优化均转6D并用冻结historical_root stats。root absmax321.0313→287.5958，abs P99 139.0255→122.6270；pose absmax28.6160→69.9630、abs P99 2.9944→3.4971。root大值并非新增强独有，但姿态分布有变化，不据此保证训练质量。
- 287.5958定位G004T009A023R007__shapeaug_s00000123、plan crop帧146、维662（P1 root Z）：原2.782516m→优化2.492738m，mean .000204437、std .008666792，对应原normalized321.0313。本次不改历史normalization契约。全量评分完成前不启动训练；后续训练效果需另验收。

### 09-12｜开始剩余6138生成，复用前100；全量评分不依赖旧优化坐标

- 预检32277 exit0：6138例，5772例timing positive、5575例semantic active，其余仍保留在分母。90575 exit0核对冻结selection恰为前100+剩余6138，无重叠、位置0..6237完整；不筛掉零timing样本。
- GPU1实测仅20MiB占用后启动45170/PID4182906，已看到实际50步优化输出；GPU0旧评分/暂停训练未动。磁盘可用8.5GiB，沿用小样本约0.8GiB的全NPZ容量估计，不声称实际全量大小。
- 独立ground工具直接读取新生产smoke2 sidecar所绑定的源H5重新算全序列原/new mesh floors：71656 exit0；与旧独立floor参考逐项精确一致。接同一生产目录做support评分13261 exit0，证明后续可从新产物+原H5独立重建参考，不需生成另一套旧优化结果。完整6138评分尚未启动，不用sidecar里缓存的floor冒充独立验证。
- 主agent复算生产100（样本等权）：脚滑6.279624→5.767546 mm/frame（80不增）；Eq6 .140836875→.021187844 m²/frame（92）；depth sum3.854285→.632460 m/frame（93）；mean per-sample max depth45.433→33.053mm（80）；conditional depth14.732→9.267mm（82）；vertex fraction .00843493→.00181392（93）；collision frame fraction .462595→.425363（67）。主脚滑+Eq6交集74/100。
- 低成本agent已运行完整materialization validator exit0；raw tokenizer单batch只证明一个batch可取，不能声称遍历74例。已要求按manifest声明historical_root tokenizer遍历全部74例再报告，不启动训练。
- historical_root全74例CPU loader已由agent完成并由主agent独立重跑exit0：cache_seed1234、unit_length8、stats_dir=overlay、strict_assets；74/74 motion/beta/contact有限、名字完整、beta逐项等于捐赠shape文件。源码to_rot_6d保留原translation、只crop与normalize，无再次逐人归零。仅证明加载链，不证明训练效果；agent观察normalized最大幅值287.60，正式训练前仍应检查与原始数据同协议分布，不能只凭finite判定训练质量。

### 09-12｜实际生产100严格通过74%，启动隔离打包验证

- 原独立评测session84971收取明确exit0、完整100行；真实gate入口82940 exit0，`output/ssp62_production_physical_first100_gate_20260912.json`接受74/100，无容差、双方betas改变、原/优化均有脚滑观测。此为冻结神经SDF估计器，不称精确mesh交集真值；不外推全6238通过率。
- 清除质量策略测试遗留的全局builder置None（首轮79078因此NameError exit1）；New_MMM重跑29994 exit0，6个模块共16测试通过，包括新旧策略、混用拒绝、全分母及生产绑定。
- 复用现有materializer对74例创建隔离目录`dataset/interx_shape_ssp62_physical_first100_20260912`，session96986；不覆盖原数据、不启动训练。下一步核验H5姿态/平移、beta、接触mask及统计量，再准备全量新物理坐标流程。
- 打包96986已exit0；独立读回89670 exit0：两个H5全部74例姿态/平移逐元素等于实际NPZ，beta/gender及timing mask一致；26个拒绝样本不在H5，9110原训练样本为external links；val split和四份normalization逐字节未变，质量分母仍100。尚未实际训练loader验收或全6238生成。

### 09-12｜排程与打包兼容复核，旧产物不晋级

- 新生产100评分84971/PID4177681继续运行，最近读取至38/100。低成本agent补充质量选择旧/新策略、混用拒绝、完整分母、名单篡改的分支测试，主agent指出初稿mock与fixture问题，待其实际修正跑通，不把未完成测试报通过。
- GPU0实测19124MiB/99%，旧坐标candidate评分PID4126281、4130516仍Rl；旧VQ训练PID3573100等仍暂停Tsl，未恢复/停止。新增强后续先用GPU1，不假定GPU0空闲。
- 只读查看旧总控4093560及脚本：其评分子流程1日志结尾Terminated、当前仅score0子shell4106197存活；wait_pair任一非零会阻止后续旧自动materialize。迁移评分父shell4109283只接旧gate1_cuda0，无新训练。旧结果继续仅历史诊断，不作新坐标验收。

### 09-12｜新生产质量gate实现，拒绝未完成报告与fallback

- 按Ponytail新增`build_physical_shape_quality_gate.py`，直接消费新评分结果并复用production_ground_rows核验completion/contract/sidecar/NPZ、源motion/plan/floor、评分实现与权重hash，要求完整同序覆盖当前completion；不重复体积评分，不拼旧坐标缓存。
- 严格criteria：原/优化完整crop一致且主指标有限非负；双方float32 betas都改变；原/优化pair均有脚滑观测；标准脚滑和Eq6体积穿透平方和都<=原shape物理GT，不加容差。分母为全部请求，fallback或不观测不计通过。
- `validated_quality_selection`可重建新schema gate，旧双估计器策略保留；禁止新旧坐标策略混合，保持全请求分母。训练overlay尚未生成。此接入尚待完整生产报告实际重建/打包验证，不能仅靠单元测试宣称完成。
- 主agent补齐两种指标epsilon退化、任一人未换shape、原/优化各自无观测场景，unittest9268 exit0（3项含子场景）。对当前partial真实运行gate入口82741预期exit1，明确拒绝未完成报告，未产生输出gate。生产评分84971继续运行，尚无正式通过名单。

### 09-12｜生产100完成：含地面准备和NPZ写盘均值4.41秒

- 原session24286明确exit0；100名字顺序与固定旧plan前100逐项一致。调用均值4.159844s、中位4.520396、最大6.355346、P95 4.980834，5/100>5s。样本处理均值4.412644s、中位4.794260、最大6.658394、P95 5.273141，20/100>5s；包含源读入/检查、full-sequence floor、优化、NPZ写盘，不含启动与sidecar/completion写盘、评分，不宣称完整进程每例<5秒。
- 对新生产产物启动同100接触/脚滑/nearest+独立Volumetric评测GPU1 session84971，`output/ssp62_production_physical_first100_assessment_20260912.json`。必须使用这一批实际NPZ，不能沿用诊断输出72%作为生产通过率。
- 旧build_volume_gate要求旧GT独立NPZ/metrics.json等schema，不能直接套用新报告。后续直接对新报告建立可重建的质量gate及来源绑定，保留全请求分母；不为迎合旧格式重复昂贵评分，也不把只筛过的样本当完整请求。

### 09-12｜独立评测适配生产产物，2例真实读取通过

- 按Ponytail扩展既有`diagnose_shape_support_transfer.py --ground-dir`：诊断目录仍读report，生产目录校验完整completion与run_contract及其canonical hash；验证source/plan绑定、sidecar record/crop/shape一致、原/新floors一致、两NPZ hash。记录sidecar/manifest来源hash后进入同一评分路径。
- packed poses与原split-AMASS通过既有npz_to_amass_dict统一，随后按sidecar共同origin解码，无重新落地；读取后再次核对输出hash。真实生产smoke2 CPU评测session17664 exit0，`output/ssp62_production_smoke2_support_20260912.json`。低成本agent补充不完整/错误contract/输出hash/floor拒绝测试；未替代100独立评分。
- 生产100原session24286继续运行（PID4176190），未修改其runner或依赖实现。粗略空间检查：2例NPZ合计约0.257MiB，按其均值外推6238约0.783GiB，仅容量估计非实际全量大小。
- 主agent修正agent测试中的origin自比较和同目录自引用，改用独立旧candidate/floor报告，实际比对origin和NPZ SHA；不完整completion、错误contract hash、输出hash、floor均拒绝。重跑session33226 exit0（2项含4个拒绝子场景）；原产物未修改。

### 09-12｜生产新地面模式接入，2例冒烟与真实导出检查通过

- 按Ponytail复用现有函数，runner新增显式`--physical-mesh-ground`固定已验证50步/lr.001/nearest/frame_sum/metric-foot配置；完整source的原/新actual mesh floor（不算normal）→裁剪→原shape参考ground200/temporal1/horizontal.5→共享静态模型优化→共同P1 origin导出。旧模式仍走旧还原路径，新模式不再restore逐人旧offset。优化器新增physical_ground_reference参数选择已验证loss。
- 新模式manifest/sidecar记录共同origin、双方原/新floor、固定base seed、实际ground配置、实现hash。样本计时移至源读入前，至两份NPZ写盘后；仍不含进程启动/模型准备与sidecar/completion写盘、独立评分，不能称完整进程端到端。
- 预检66045 exit0（positive-only两个样本，仅预检）；实际冒烟不筛timing，与固定first2一致，session74482 exit0，`output/ssp62_production_physical_smoke2_alltiming_20260912`。优化调用均值4.276456s，样本处理均值4.483796s、最大6.609661s。原/新floor与旧完整序列报告精确一致；导出重新decode后shape保持、四人体最大入地0.322700mm，CPU验证25484 exit0。
- mock测试35103 exit0，主agent补齐完整5帧而非2帧crop断言后15879 exit0。生产100例已启动（同selection seed20260911、不筛timing），目录`output/ssp62_production_physical_first100_20260912`；待完整耗时及生产导出独立质量检查，未写训练overlay、未启动第③项。

### 09-12｜加速接口正式透传与无normal地面扫描通过小检查

- 按Ponytail最小修改：`optimize_full_body(..., static_body_models=None)`透传到loss；checker改为显式通过此接口传字典，避免partial默认值被覆盖。`PeneCalculator._compute_geometry(...,compute_normals=True)`默认不变，仅floor扫描显式False；穿透/接触调用仍计算法线。所有前一轮评分确切终态后才编辑。
- CPU floor扫描session66210 exit0，输出`output/ssp62_floor_no_normals_first3_20260912.json`；3例×原/新shape×双人12个完整序列floor与旧报告精确相等，最大序列mesh导出往返误差1.19209e-7m。
- 正式optimizer接口GPU两例50步冒烟session86185 exit0，`output/ssp62_reuse_api_smoke_first2_20260912`，复用字典传入；这不是重新完成100例质量验收或端到端计时。低成本agent补充mock默认None/字典身份透传与默认normal/跳过normal测试。
- 生产runner全序列mesh floor/原shape参考/共同origin/manifest契约尚待接入，训练数据未修改；第②项仍进行中。
- 主agent复核新mock测试并实际运行session40636 exit0（2项），未用mock结果替代上面的真实CPU/GPU检查。

### 09-12｜完整100独立体积验收完成：联合72%，平均优化调用4.17秒

- 原next75会话17109明确exit0，合并前25得到100唯一名字、同原/优化帧数，全部汇总指标有限；报告`output/ssp62_reuse_physical_volumetric_first100_summary_20260912.json`绑定两份来源hash并保存严格联合通过名单。
- 原GT→优化（样本等权）：Eq6双向穿透平方和0.140836875→0.021203935m²/frame，不增92/100；深度和3.854285325→0.632404637m/frame，不增93/100；穿透顶点条件深度14.731897→9.260605mm，不增82/100；样本最大深度均值45.433421→33.089088mm，不增80/100。
- 穿透顶点比例0.843493%→0.181235%，不增93/100；碰撞帧比例46.259491%→42.587770%，不增66/100；标准脚滑6.279624→5.768301mm/frame，不增78/100。Eq6+脚滑严格联合72/100（72%），不加容差、不以fallback算通过。不是所有物理指标逐例改善。
- nearest代理穿透91/100，独立体积92/100；3例判定不同（G047T000A000R004/G013T002A009R025/G047T003A015R003），联合数量碰巧同为72，不声称评测等价。
- 与平均optimizer-call4.173646s合并看，固定100例达到小批量平均调用<5s、联合>60%目标；仍有6例调用>5s，且不含完整floor准备/导出。生产坐标链、全量增强/严格gate/打包与端到端计时仍待完成；第③④项不提前启动。

### 09-12｜定位地面准备冗余法线，CPU小批量验证精确不变

- 继续第②项端到端速度检查：现有floor准备走PeneCalculator._compute_geometry，求最低Y却额外计算vertex normals。原路径first3单次完整原/新shape四人体floor耗时0.873/0.220/0.408s，CPU模型初始化1.364s一次；说明4.174s优化调用不能直接当完整sample处理时间。
- 仅在独立进程mock掉normal计算，未修改生产/运行中评分代码。3例（169/71/123完整帧）预热后交替3次原路径与3次无normal，floor数值精确相等；四人体floor中位耗时0.573905/0.231437/0.437293→0.198124/0.080383/0.163154s。session42851 exit0，报告`output/ssp62_floor_normals_cpu_first3_20260912.json`。不是整流程加速验收，不把3例外推全数据。
- 独立体积next75原session17109仍运行，主线程最近已收至24/75。下一步完整体积结果后，将无需法线的floor准备与已验证静态模型复用一并接入生产，而不改变接触/穿透所需法线。

### 09-12｜纠正跨agent句柄可见性，主线程继续收尾

- 子agent用正确session17109仍返回Unknown process id；主agent同句柄成功读取至14/75，PID4172791 Rsl，确认为会话不可跨agent读取，而非任务失败。主agent接回原会话，未重启。已写入AGENTS规则，后续子agent只提取终态文件指标，不重复尝试接管该句柄。
- next75独立体积尚未完成，完整100验收结果不变为“已通过”；当前仅first25独立联合20/25已证实。生产/训练数据未修改。

### 09-12｜独立评分委托低成本收尾，明确生产坐标接入点

- next75独立体积session17109、PID4172791已核验存活；交给既有gpt-5.6-luna低成本agent `lr_report_extract`持有原句柄并收取确切终态/合并100指标。主agent不重复轮询同一session；未完成前不宣布100体积通过率。
- 主agent只读确认`run_interx_shape_augmentation.py`约745行仍按crop joint floor对齐，822/825行分别restore旧raw坐标；这不是已验证的full-sequence actual-shape mesh floor/共同origin方案。`run_paper_contact_optimization.output_payload`和`materialize_interx_shape_augmentation.output_raw`保留trans，不自行recenter；无需改训练raw布局，但必须修正上游坐标链并绑定sidecar/manifest契约。
- 现有`diagnose_shape_physical_ground.py`仅CPU、已计算前100完整序列mesh floors，尚不是全6238可恢复的生产floor阶段。未改生产数据或启动新训练；完整体积验收后再集成。

### 09-12｜前25独立VolumetricSMPL通过20/25，继续完整100验收

- 原session94181 exit0；报告`output/ssp62_reuse_physical_volumetric_first25_20260912.json`，25条数值有限。同完整crop、原shape物理GT对解码候选、默认编码1000/query512、同名seed；未复用旧坐标GT指标。
- 原GT→候选（样本等权）：Eq6双向穿透平方和0.126672192→0.003292068m²/frame，不增24/25；穿透深度和3.157679323→0.179975184m/frame，不增24/25；穿透顶点条件均值14.213718→8.316012mm，不增23/25；样本最大深度均值43.270081→26.607587mm，不增24/25。
- 穿透顶点比例0.648458%→0.088686%，不增25/25；碰撞帧比例42.4903%→38.0698%，不增18/25。标准脚滑6.279881→5.675726mm/frame，不增21/25。以Eq6平方和+脚滑联合判定20/25（80%），与同批nearest proxy逐例穿透判定一致；不能称所有物理指标逐例改善。
- 其余75例独立体积评分GPU1 session17109已启动，无需重新优化；完整100与生产/训练格式集成仍待完成。

### 09-12｜复用100例优化完成，平均调用4.174秒

- next75原session14555 exit0；与first25合并100例，名字/顺序/optimizer与地面数值配置逐项核对一致。平均optimizer-call4.173646s、中位4.516912、最大6.333156，6/100>5s（包含两批各自首例初始化）。原无复用100均值5.444217s。最大入地0.813603mm；非端到端速度，不声称每例<5秒。
- next75独立标准脚滑+nearest穿透评测GPU1 session17792启动，报告目标`output/ssp62_reuse_lr001_iter50_physical_next75_20260912.json`；完整100通过率待合并。前100原/新400个人gender全部male/female，crop36–150帧，独立体积输入条件已核对。
- 收尾：17792 exit0。100标准脚滑原GT6.279624→候选5.768301mm/frame；nearest穿透原GT0.226241849、新shape0.635432311、优化0.036891482m²/frame。脚滑不增78/100、穿透91/100、交集72/100（无复用79/91/73）。严格不加阈值；仅proxy。
- 前25独立体积评分session94181启动；复用原评分函数、默认1000编码采样/query512、相同sample seed、完整crop、原GT与解码后候选。目标`output/ssp62_reuse_physical_volumetric_first25_20260912.json`，partial标为running_not_acceptance。未完成不报体积通过率。

### 09-12｜准备独立体积验收入口，未冒充已评分

- 遵循Ponytail扩展既有`diagnose_shape_support_transfer.py --volumetric`，直接复用`score_volumetric_smpl_npz.score_pair`。原shape GT按已验证全序列mesh floor恢复，候选仅decode共同origin；同名字seed1234规则、完整crop、不额外150帧截断、不重新落地。保存两条件体积/脚滑指标和checkpoint/实现hash，每完成一例写partial（非验收）。默认不启用体积评分。
- 既有自检查和CLI help session76088 exit0；低成本agent补充mock完整151帧/同seed/坐标保持与空帧拒绝检查。体积真实GPU评分尚未启动，等next75释放GPU1；该任务原session14555、PID4170752已复核仍运行，不重启。
- 主agent将mock原/优化设置为不同平移与beta，核对实际传给scorer的双方所有字段与seed函数参数，防止误用原输入也能过测试；unittest session9296 exit0（2项）。仅入口契约验证，不是体积模型真实评分。

### 09-12｜模型复用first25完成：平均4.23秒，联合代理通过20/25

- session80302/33053均exit0；optimizer配置、样本同序与原50步对照一致。平均optimizer-call5.471943→4.232302s（约22.65%下降），中位4.457036、最大6.333156s，2/25>5s；首例6.333156包含模型初始化，其余24均值4.144766。仍不含source/floor准备、导出与独立评分，不声称端到端全部<5s。
- 接触最小距离106.443438→106.443501mm，最近5%120.436996→120.420114mm，2cm接近帧比例44.2016%→44.2866%；标准脚滑均值5.670398→5.675726mm/frame（GT6.279881）。nearest穿透均值0.006680677→0.006683559m²/frame，穿透不增均24/25，脚滑22→21/25、交集21→20/25。报告`output/ssp62_reuse_lr001_iter50_physical_contacts_first25_20260912.json`，不是体积验收。
- 跨边界样本G055T006A030R000：GT3.878691、原优化3.856202、复用3.894788mm/frame。严格计失败，不用容差提高通过率；已知CUDA重复波动，但不能直接断言本次差异原因。最大入地0.814mm，地面仍未出现大偏移。
- next75固定同设置session14555运行，补齐100例。独立VolumetricSMPL权重实存已复核；后续使用新物理原shape GT配对，不能复用旧坐标GT评分冒充验收。

### 09-12｜静态模型复用可选接入，开始真实optimizer对照

- 按Ponytail在现有DyadicInteractionLoss增加caller-owned `static_body_models`，默认None每次独立创建；checker增加`--reuse-body-models`，仅其进程持有字典。第一次创建仍在计时内；之后复用neutral/male/female，检查设备和无参数。每例loss、角色、接触、支撑缓存仍新建；无全局cache，生产入口默认不变。
- 已核实前一实验终态后才编辑loss/checker。25例GPU1 session80302运行，lr.001/50步、地面/吸引不变，对照此前同25例5.47194s。不得从CPU初始化开销推算实际GPU提速；完成后独立评测并考虑已知CUDA重复误差。
- 既有地面参考测试session39644 exit0，diff-check通过；低成本agent仅补充mock模型复用/样本状态隔离检查。接入不等于已验证通过率或<5秒。
- 新增mock测试主agent复核补齐female对象和winner/weights隔离，unittest session35979 exit0（3项）。默认两次6模型、shared两次3模型、不完整字典拒绝均通过。存在原有pkl直接open的ResourceWarning，当前运行中不改loss；不是测试失败。

### 09-12｜100步对照完成：迭代不足确实影响吸引，但速度仍不达标

- 原session36276 exit0，25同名同序，optimizer仅iterations50→100不同。独立全crop接触/脚滑/nearest穿透session23036 exit0，报告`output/ssp62_lr001_iter100_physical_contacts_first25_20260912.json`。
- 固定lr.001：32关系等权最小距离106.443→94.384mm，最近5%距离120.437→108.408mm，2cm接近帧比例44.202%→49.182%；单关系距离102.583→94.770，多关系110.818→93.947mm。支持50步预算不足是原因之一，不证明所有接触残差都由迭代造成。
- 标准脚滑原GT6.279881、50步5.670398、100步5.335472mm/frame；不增均22/25。nearest穿透原GT0.181434、50步0.006681、100步0.004964m²/frame；不增均24/25。交集均21/25（84%），仍仅proxy验收。
- 平均optimizer-call5.47194→9.60291s，最大入地0.625→0.415mm。100步是诊断收益，不是<5秒生产方案；不直接扩大为全量。
- 三性别静态模型复用GPU检查session3549 exit0，`output/ssp62_body_model_reuse_runnable_cuda1_20260912.json`：A4/B7/A4与新实例的vertices/joints、五组输入梯度全部最大误差0，buffers不变。下一步仅复用静态模型，保留每样本独立loss/接触缓存，再测整段optimizer时间与物理指标；目前没有实现或宣称加速。

### 09-12｜静态模型复用：补齐可运行验证，CPU通过

- 新增`tools/check_body_model_reuse.py`，复用现有BodyModel和atomic_json；三种性别分别测试4帧A→7帧B→4帧A、与新实例比较。检查vertices/joints及body/hand/root/trans/betas五组输入梯度，buffer不变、无模型参数、数值有限。
- CPU session57611 exit0，报告`output/ssp62_body_model_reuse_runnable_cpu_20260912.json`；重复与新实例误差全部0。尚未GPU验证，不代表整段optimizer等价或已实现加速。遵循Ponytail仅加独立检查，未改正在运行且记录hash的loss/checker。
- lr.001/100步原session36276、PID4167788已核实仍运行；不重启。下一步收取终态并独立评测，再安排GPU复用验证。

### 09-12｜学习率对照完成：吸引改善，但脚滑代价过大

- 同名同序25例，optimizer配置仅lr .001→.003；独立报告`output/ssp62_lr003_physical_contacts_first25_20260912.json`。32条有时段关系等权平均最小距离106.443→101.029mm，最近5%距离120.437→115.464mm，2cm接近帧比例44.202%→47.472%。单/多关系子集均略改善；不是接触语义正确率。
- 标准脚滑原GT6.279881mm/frame，lr.001候选5.670398，lr.003候选6.107759；逐样本不增22→12/25。nearest穿透proxy原GT0.181434，新shape0.157877，优化0.006681→0.004655 m²/frame；不增均24/25。脚滑+proxy交集21→11/25（84%→44%），不是独立体积验收。
- 最大入地0.625→0.677mm；平均optimizer-call5.47194→5.46272s；最大身体轴角修正范数0.07515→0.13675rad。地面仍合理，但更大学习率明显损害脚滑逐例通过，不采用。只能说明lr对吸引有效，不能把未吸引充分全部归因于lr。
- 下一对照固定lr.001增加至100步，GPU1 session36276已启动，区分迭代预算影响；仅诊断，不冒充<5秒方案。复用既有评测入口，无新评分实现。

### 09-12｜学习率单因素对照启动：.001→.003，50步不变

- 按Ponytail仅给独立checker增加learning-rate override，默认None沿用manifest，非有限/非正数拒绝，report使用实际config。生产optimizer实现、loss权重、原标注不变；既有运行已收尾后才编辑脚本。
- GPU1启动前20MiB；固定first25新session98955，`output/ssp62_ground_horizontal05_lr003_iter50_first25_20260912`。lr.003、50步、ground200/temporal1/horizontal.5、relation_scale0，对照既有lr.001/50步baseline。评测继续原全局timing的固定区域接近、标准脚滑、nearest穿透proxy、地面/姿态变化与耗时；不只比较loss。
- 已知多关系全局timing存在语义限制，另分单/多关系查看，不通过本试验宣称修复时段标注。低成本agent核验CLI拒绝非法lr及传参与报告路径。

### 09-12｜软时段25例独立评估：整体未改善，不扩大

- GPU76462 exit0，`output/ssp62_relation005_physical_contacts_first25_20260912.json`。沿用原全局mask、32关系：最近距离均值106.443→108.504mm，2cm接近比例44.202→44.261%；多关系子集110.818→115.317mm、50.778→50.902%。新旧物理proxy：穿透不增24→24/25，标准脚滑22→21/25，交集21→20/25。存在已实测重复运行波动，不能以微小比例变化声称增益；不扩大该原型。
- G031两关系原全局时段均值49.065/549.087→55.569/570.529mm，但2cm帧比例70.37/0→74.07/7.41%。为区别撤去不对应时段吸引与有效接触改善，另按事先已核实的人工早段12:37、晚段55:57检查，单例GPU19766启动，输出`output/ssp62_relation_timing_effect_G031_20260912.json`。不把重加权分数替代完整质量验收。
- GPU19766 exit0：摸头早段均值13.302→15.914mm、≤2cm帧19→20/25；推手晚段56.755→55.967mm、≤2cm仍0/2。新增两帧手手≤2cm出现在早段而不是预期晚段，因此不能说软时段修好了该反例。此原型不晋级、不扩大；下一步回到固定地面方案下的学习率/迭代数直接对照，时段缺口保留为限制而非宣称已解决。

### 09-12｜软时段25例优化完成；单关系重复运行存在数值波动

- 原GPU79636 exit0，25例report完整，与旧horizontal.5 baseline的optimizer及名单/sidecar SHA一致，最大入地0.636mm。此前agent发现7个已完成单关系样本不bitwise；root逐字段确认0.008503最大值是手部轴角rad，不是8.5mm位移，不能混单位。
- 单例G029T000A003R021（offset9）三次同seed/同50步运行session19709 exit0，输出`ssp62_repeat_G029_{zero_a,zero_b,relation}_20260912`。关闭relation两次：最大trans约1.003mm、手部轴角分量0.007092rad；关闭vs开启单关系：约0.924mm、0.005014rad。至少证明旧算法重复运行本身不bitwise，不足以定位具体算子或证明所有新旧差异仅为噪声。
- 只读源码：每例torch.manual_seed1234，未设置deterministic算法；vertex_normals有index_add_并行累加，但没有实验锁定它为根因。不将微小改善作单次因果结论。
- 新25独立GPU全物理proxy+原口径contacts启动，输出`output/ssp62_relation005_physical_contacts_first25_20260912.json`；评估不使用pseudo时段替换原mask。地面不再调毫米阈值，待多关系实际效果再进入学习率/步数对照。

### 09-12｜逐关系软时段接入独立25例优化

- 按Ponytail扩展现有实验loss/checker，relation_timing_scale默认0保持旧路径；新.05m只对多关系构造原shape物理网格逐帧距离软权重。先用原全局mask选择初始角色，再按固定角色产生原reference权重；每条关系单独调用原semantic loss并按关系均值，未改原非接触loss或权重。单关系保留原路径。
- 明确禁止进入anchored阶段（checker iterations<=100，loss遇is_anchored拒绝），避免单关系调用复用错误锚点索引。未覆盖人工timing/区域，不把pseudo权重当真标签；距离比较仍用不变的原评测口径。
- root旧ground/reference及timing测试10987 exit0；逐关系mock/有限梯度63808 exit0。agent此次未实际补集成测试，root已在既有tests/test_relation_timings.py补两列独立mask/关系均值/默认原路径/anchored拒绝/有限梯度，94632 exit0。
- GPU1启动前20MiB，固定first25 session79636启动，`output/ssp62_ground_horizontal05_relation005_iter50_first25_20260912`；ground200/temporal1/horizontal.5/50步/lr.001均不变。完成后评估原全局时段接触指标、关系时段局部效果、脚滑/穿透和地面，不能只比重加权loss。

### 09-12｜原GT几何逐关系软时段原型验证，尚未优化

- 按Ponytail在既有contact诊断中添加`relative_relation_timings`：w[t,k]=manual[t]*exp(-(d[t,k]-min_k d[t,k])/0.05m)，距离固定来自原shape物理几何，不使用优化结果。不开新人工接触帧；单关系/等距关系保留原mask，同时接触不强制互斥。局限是不能识别所有关系都错的帧，明确为pseudo而非人工标签。
- root CPU62929 exit0，`output/ssp62_relation_timing_pseudo_G031_20260912.json`绑定原逐帧报告/实现SHA，保存110帧权重。摸头早/晚总权重23.7128/0.000123；推手3.3470/2.0000。重加权后的距离只用于理解mask，不能称优化收益或与旧整体均值作质量提升比较。
- 低成本agent补`tests/test_relation_timings.py`并exit0；root复读重跑。当前semantic loss只接收全局[T] mask，不能直接塞[T,K]导致广播错位；下一步需逐关系调用/归一化的明确实验接入，且保持baseline角色选择及50步未锚定路径。原标注/优化器尚未修改。
- root首次直接运行发现测试缺少repo sys.path（ModuleNotFoundError），且未真正断言列交换。已仅修测试入口并补列交换、非等距零manual和T=0边界；root重跑88037 exit0。不能把agent初版PASS当直接可运行证据。

### 09-12｜接触反例确认：两条顺序关系错误共用全局时段

- 文本G031T008A032R012描述先P1右手摸P2头，再P2左手推开P1右手；LLM区域分别为[23,26]↔[39,51,52,66]及[23,26]↔[17,33]，目前两者共用同一binary timing。agent原口头计数28有误，root实际np统计12–36和55–56合计27帧。
- root GPU80551 exit0，`output/ssp62_contact_timing_roles_G031_20260912.json`保存完整110帧距离及两个角色顺序，绑定floor/contact/timing SHA。原shape物理坐标：早段手头88.546mm、手手616.545mm；晚段手头558.985mm、手手73.554mm。交换角色手头/手手的全序列最小距离仍713/618mm，原方向56.574/52.099mm，反向更差。
- 因而共用mask把顺序接触误当同时接触是该例明确缺口；不是只要加大吸引或lr即可解决。现有原GT最近仍约5cm，不能把所有误差归因时段。下一步应在不覆盖人工全局timing/区域标注的前提下，验证独立的逐关系时段分配；几何推断时段须标为pseudo，不冒充人工标签。地面方案保持冻结。

### 09-12｜接受合理地面偏差；100例新坐标proxy完成，转接触诊断

- 用户明确无需完美地面，只需合理物理范围；已写AGENTS规则。不把诊断1mm条件作为继续调参的阻断，冻结ground200/temporal1/horizontal.5/50步；不宣称恢复原始捕捉绝对世界坐标或所有支撑标签。
- GPU43737 exit0，`output/ssp62_ground_horizontal05_physical_proxy_first100_20260912.json`：完整100例原/新shape分别用自身full-sequence mesh floor，优化只加共同origin；标准脚滑不增79、nearest-vertex穿透不增91、交集73/100。穿透均值原0.226242/换shape0.635432/优化0.036891 m²/frame。精确crop、双向平方和均值、无timing过滤，仍是proxy而非独立VolumetricSMPL验收。
- 按Ponytail复用现有fixed_region_contact与实际角色匹配方法，给support-transfer加可选contact-dir。在物理new-shape初始化上选一次角色，原/新/优化共用；固定标注region union和原timing>0帧，报告动态attraction与无符号最近距离/5%距离/1、2、5cm邻近比例，避免仅看loss下降。GPU first25 session16172启动，`output/ssp62_ground_horizontal05_contacts_first25_20260912.json`；不改优化器或训练数据。
- 接触session16172 exit0；25例中24例有标注及有效时段，共32条关系；动态吸引下降16/24、固定最近距离下降20/32。原/换shape/优化的关系等权平均最近距离115.894/117.851/106.443mm，双向最近5%距离128.586/131.248/120.437mm，2cm帧接近比例39.35/36.17/44.20%。这些是proximity诊断，不能叫真实接触召回。
- 最坏剩余关系G031T008A032R012：原576.324mm、换shape634.746mm、优化549.087mm；原GT已很远。下一步先核标注角色/时段/语义而非盲加吸引，低成本agent只读提取该例文本、region及timing元数据。agent对contact路径只读审查未见明确混人/混坐标问题。

### 09-12｜horizontal0.5完整100例收齐：候选有效但尚未联合验收

- ground_max_check报告GPU进程终态、next75 report75完整及CPU评估exit0；root读取真实first25/next75并与baseline核对100唯一名字同序。待agent补GPU确切退出码/CPU句柄与sidecar独立审计，不把其GPU_EXIT_CONFIRMED字符串当完整退出码证据。
- 旧→新固定原支撑水平位移均值2.925→2.360mm/frame（原3.010，仅98例可观测），逐样本不增42→75/100；标准脚滑5.764→5.766（原6.280），不增73→79/100。不能把固定mask诊断替代标准指标，也不能用平均值替代逐样本验收。
- 支撑高度偏差中位/P95/最大0.574/1.857/16.737→0.573/2.403/16.704mm；越1cm观测182→201（原观测10324）。最大足高一阶/二阶误差19.079/7.917→19.079/7.965（mm/frame、mm/frame²）。新最大入地0.625mm、optimizer-call均值5.444217s，最大7.619906s、73/100>5s，不含source/floor准备和CPU评估。
- 双方支撑可观测且高度偏差≤1/2/5mm为57/74/82；与固定支撑滑动不增交集43/55/62，与标准脚滑不增交集45/59/67（分母100）。这些阈值仅保留敏感性结果，不能因5mm超过60便宣布通过；新坐标人体间penetration未联合评估。水平约束保留为候选，不再盲加权，地面尚未生产接入。
- agent补CPU session31488 exit0及100名单/实际sidecarSHA/config核验exit0；其承认仅ps观察GPU终态、未拿原句柄退出码，root随后直接回收原session39970核验，不复启任务。

### 09-12｜三性别BodyModel复用CPU等价检查通过

- CPU session57493 exit0，产物`output/ssp62_body_model_reuse_three_gender_cpu_20260912.json`绑定三性别资产及实现SHA。neutral/male/female各做随机非零body/hand/root/trans/betas、4帧A→7帧B→4帧A，并与新建实例比较：vertices、joints及五类输入梯度均逐元素完全相同；buffers不变、模型无参数。
- 范围仅CPU FP32短序列模型前向/反向，不是GPU计时、完整优化或生产缓存验收。未改运行代码、未启用缓存；next75优化及后续CPU评估仍由ground_max_check独占监控，主agent不重复轮询。

### 09-12｜等待地面next75期间只读定位静态模型初始化开销

- next75 session39970/PID4160129核实存活。监控与完成后CPU support-transfer已交给既有低成本ground_max_check，主agent不重复轮询/启动；预定CPU输出`output/ssp62_support_standard_foot_temporal1_horizontal05_w200_iter50_next75_20260912.json`，须GPU明确exit0且report75完整才启动。
- 按Ponytail只读追踪：每样本loss ctor重建neutral/male/female三个BodyModel并读取region表；不能整对象缓存loss，因其包含支撑mask、角色/锚点/当前区域等样本状态。未修改任何运行源码。
- CPU实测session91950 exit0，`output/ssp62_body_model_reuse_cpu_probe_20260912.json`：三次loss构造1.336/1.350/1.357s，三BodyModel构造几乎占满。男性模型A→B→A变帧数/shape，A的输出与输入梯度完全一致，buffers未变且无可训练参数。仅CPU单性别复用初查，不是三性别/GPU/完整优化等价验收；不从CPU数字直接扣除GPU耗时宣称<5s。地面验证后可单独测试静态模型复用。

### 09-12｜horizontal0.5前25独立评估：滑动改善、地面尾部有代价

- CPU68966 exit0；root同序比较旧50步前25与新25，agent核对名单/sidecar/floor/optimizer/ground配置一致，仅horizontal0→.5。固定原支撑水平位移均值2.803→2.289mm/frame（原2.973），不增12→20/25；标准脚滑5.656→5.670mm/frame（原6.280），不增20→22/25。
- 原支撑高度偏差中位/P95/最大0.526/1.642/7.291→0.551/2.220/7.102mm；越1cm观测29→41；最大入地0.403→0.625mm。足高一阶误差最大7.236→6.988、二阶6.306→6.768（mm/frame、mm/frame²）。不是各项全优，不能只报脚滑改善。
- 双方支撑可观测且高度≤1/2/5mm的样本17/20/21→16/19/21；同时固定支撑水平不增交集10/10/10→13/14/16，标准脚滑不增交集14/16/17→14/17/19（分母25）。诊断阈值未变，尚未加人体间penetration验收。
- 固定其余75例offset25/limit75启动，输出`output/ssp62_ground_original_soft_temporal1_horizontal05_w200_iter50_next75_20260912`；同ground200/temporal1/horizontal.5/50步/lr.001，不再叠加约束、不改吸引，验证100例分布与取舍后再决定。

### 09-12｜horizontal0.5前25优化完成，独立指标评估启动

- 原GPU session21926回收exit0，完整25例；最大入地0.625mm，optimizer-call均值5.471943s（不含source/floor与CPU评估），速度仍未达标。独立support-transfer评估已启动，输出`output/ssp62_support_standard_foot_temporal1_horizontal05_w200_iter50_first25_20260912.json`；待固定原支撑/标准脚滑/高度/时间对照，不提前扩大试验。
- root发现低成本agent新增mock虽PASS，却使用全高于地面的空mask且仅返回1个非ground项，未证明其声称的完整覆盖。按Ponytail在现有测试内修正：两人不同左右非空mask、不同reference speed、实际swapped cache、前4项保持及有效水平ground贡献=5的数值断言；root session78146 exit0。未改正在运行的优化模块或checker，测试不参与运行产物hash。

### 09-12｜固定原近地支撑水平超速惩罚：25例单因素试验

- 按Ponytail仅扩展实验`MeshGroundSupportLoss`及独立checker：原shape mesh脚高≤1cm冻结区间起点，左/右脚映射踝趾7/10、8/11；附加mean ReLU(current XZ speed−original XZ speed)，不把正常原动作速度强制为0。缓存原geometry并同步角色交换；horizontal默认0保持旧路径，正权重要求原shape参考，拒绝非有限/负值。生产默认优化未接入。
- 原高度衰减/ground200/temporal1/lr.001/50步和所有原非ground项不变；新horizontal=.5，有效附加权重100。可能与高度保持、动作姿态发生竞争，须配对验证，不先宣称改善。root内置水平超速/不超速/无支撑/梯度检查及旧角色缓存测试均exit0（11575/18344）；低成本agent独立mock角色及边界复核中。
- GPU1启动前仅20MiB占用；固定first25新输出`output/ssp62_ground_original_soft_temporal1_horizontal05_w200_iter50_first25_20260912`，不覆盖旧50步baseline。完成后用既有独立support-transfer检查地面、时间、固定支撑滑动及标准脚滑，未开始吸引调参。
- 实际优化句柄session21926。低成本agent独立CPU检查exit0：XZ/原速以下不罚/超速线性/无支撑/T1/有限梯度；其报告角色与horizontal0旧行为一致。仍须实际样本验收。

### 09-12｜标准脚滑100例完成：与原支撑口径存在35例分歧

- CPU78774 exit0。复用标准脚滑公式/HBP几何，双人observations加权、样本等权均值：原/换shape初始/优化=6.280/6.442/5.764mm/frame；优化≤原73/100。与双方原支撑可观测且高度偏差≤1/2/5mm交集为47/57/62，阈值仅敏感性诊断，不因5mm交集>60就据此放宽验收；人体间penetration仍待新坐标协议评测。
- 35例标准指标改善但固定原mesh支撑水平位移增加。最大分歧G025T000A000R002：标准7.152→6.321mm/frame，但原mesh支撑0.791→1.486。下一步不盲加ground权重：CPU session38568按同一原动作冻结标准mask，再拆原mesh支撑/其余区间，区分mask变化与平均稀释，输出`output/ssp62_ground_mask_disagreement_G025_20260912.json`。不改优化或吸引项，按Ponytail复用现有几何函数。
- session38568 exit0：标准mask共1089观测，其中原mesh近地仅192；优化后标准mask减少24观测。即便冻结原标准mask，总位移仍7.788485→7.565247m下降，但其中原mesh近地位移0.151943→0.285365m上升；不是单纯mask改变，较宽统计区间的改善掩盖了近地子集恶化。人2近地160观测位移0.122294→0.252137m。该例支持下一步小批量尝试对原mesh支撑的水平运动单独约束，不能外推35例全是相同原因；保持吸引项不变。

### 09-12｜固定支撑滑动揭示逐样本退化，补标准脚滑口径

- CPU44435 exit0，完整100例horizontal报告已生成。双人按sum/observations合并，再对98个有观测样本等权平均：原/换shape初始/优化=3.010/3.136/2.925mm/frame；另2对无观测，不能算通过。优化不高于原仅42/98；同时双方有支撑且高度偏差≤1/2/5mm的交集为30/34/36（总分母100）。平均下降不能代替逐样本通过率，也不能将此诊断冒充官方指标。
- 最坏水平增量G037T005A025R005：原0.496→优化1.488mm/frame。尚不能由此单独归因ground loss；需要同批标准脚滑及后续配对ground消融，不盲目修改吸引或加大权重。
- 按Ponytail复用`score_volumetric_smpl_npz.foot_sliding`加入同一CPU诊断，固定已有公式（各候选关节floor/高度mask），明确几何来自HBP，非VolumetricSMPL后端验收。原44435已退出后才修改脚本；新CPU session78774启动，输出`output/ssp62_support_standard_foot_temporal1_w200_iter50_first100_20260912.json`，附公式来源SHA，不重跑优化。

### 09-12｜固定原支撑时段的水平滑动诊断启动

- 按Ponytail扩展既有`diagnose_shape_support_transfer.py`，复用FOOT_JOINT_INDICES=(7,10,8,11)，在原shape mesh左/右脚≤1cm的区间起点上，比较原/换shape初始/优化后踝与趾XZ位移；不按候选高度或速度剔除观测。记录sum/mean/max（m/frame）与计数，无观测为None；不是官方footskating指标，不替代最终验收。
- 内置小张量检查含映射计数、无支撑与单帧，root session26195 exit0。低成本agent独立检查左右脚映射、仅XZ和常量平移不变。旧评估32534已exit0、无同脚本存活进程后才修改；未改生产优化或吸引项。
- CPU完整100例session44435启动，输出`output/ssp62_original_support_horizontal_temporal1_w200_iter50_first100_20260912.json`；复用已完成的50步first25/next75，不重复优化。待结果检查地面稳定子集是否以增加滑动为代价。
- 低成本agent独立CPU检查exit0，左右映射、仅XZ、固定原mask、平移不变、空支撑及T=1全部通过；root复核实际PID4156879存活（CPU286%、elapsed00:32），评估尚未完成，不重复启动。

### 09-12｜50步100例原支撑评估收齐，复核时间突变

- 原CPU session32534回收exit0。root合并first25/next75，低成本agent独立复算：100对、200人体；原支撑观测10324，换shape初始化/优化后越1cm观测920→182，17人体原本无观测，不计支撑验收通过。双方可观测且最大高度偏差≤1/2/5mm为63/77/82对，分母100，阈值仅诊断。
- 支撑高度偏差中位/P95/最大（mm）初始化1.744/8.317/47.068→优化0.574/1.857/16.737。足高一阶误差中位/P95/最大（mm/frame）1.503/7.177/20.506→1.863/6.352/19.079；二阶误差（mm/frame²）1.414/6.206/10.194→2.170/6.307/7.917。中位时间误差变差，不能凭尾部下降宣称整体连续性改善，更不是footskating结果。
- 最大支撑偏差G019T000A000R005人2为16.737mm；独立CPU session87158检查最大二阶误差G004T009A023R007人2的原/初始化/优化足底、踝、root局部曲线。按Ponytail复用已有几何与解码函数，不扩充生产流程；地面未验收，吸引仍暂停。
- 局部检查首次87158因geometry返回3项却解包2项exit1、未输出；修正调用解包后85599 exit0，产物`output/ssp62_temporal50_worst_G004_20260912.json`。人2右脚源221帧二阶变化：原/初始化/优化=-8.371/-6.337/-0.454mm/frame²；踝=2.875/2.984/2.881，root=-1.195/-1.195/-0.255。该最大误差来自脚底高度曲线变平，不是此前那种踝部新增尖峰；不能把与原轨迹差异直接称为抖动。仍需检查通过子集的实际支撑稳定性与其余尾部，不能据单例全局验收，也不为缩小该误差盲目加大平滑。

### 09-12｜50步完整100例优化结束，原支撑与时序评估收尾

- next75 session64833回收exit0，和first25合计100唯一样本；最大入地0.644mm。optimizer-call均值5.396911s、最大7.453329s、71/100>5s，不含source/floor准备及CPU/export评估，不能宣称速度达标。
- next75独立评估session32534启动，输出`output/ssp62_original_support_original_soft_temporal1_w200_iter50_next75_20260912.json`；完成后与first25支持时序报告合并，不能用旧new-shape support字段替代原shape参考。低成本agent核对名单、配置和真实sidecar SHA中。
### 09-12｜训练转换边界复核：平移保持，生产接入尚未完成

- 50步剩余75例session64833/PID4153469核实存活，未改运行代码。按Ponytail只读追踪output_payload→materializer.output_raw→tokenize.to_rot_6d→raw normalizer：这些转换不做逐人重新归零。当前诊断NPZ是root_orient/pose_body/pose_hand分离格式，materializer要求poses，因此不能直接打包；生产runner仍有旧restore路径，尚需显式接入新坐标/共同origin元数据，不能把诊断结果写作已可训练。
- 低成本agent新增`tests/test_physical_ground_training_encoding.py`；root发现初版未真正测试反归一化/从恢复数据解码，要求修正后又补直接运行的sys.path位置，最终root重跑exit0。测试实际走临时NPZ、raw、6D、normalizer forward/backward并验证两人相对trans及共同origin恢复，未初始化完整loader或写生产dataset。
- 真实50步first25首样本CPU往返：首次session87833因5微米mesh断言失败，未当通过；root session32250改为报告实际差异，exit0，`output/ssp62_physical_training_conversion_roundtrip_one_20260912.json`。相对trans误差1.1921e-7m，raw normalize反转误差5.9605e-8，旋转矩阵误差3.93e-4/3.35e-4，mesh最大误差0.090/0.163mm。平移约定保持；包含轴角/6D转换的网格非bitwise，未改历史旋转实现或VQ流程。仍仅转换诊断，不是生产接入验收。
### 09-12｜50步收敛改善但5秒目标未达，扩大剩余75例

- 50步优化session71510 exit0，独立评估session77751 exit0，`output/ssp62_original_support_original_soft_temporal1_w200_iter50_first25_20260912.json`。同25例30→50：最大原支撑脚高偏差20.062→7.291mm；越1cm观测49→29；最大足高速度误差6.763→7.236mm/frame，二阶差误差6.651→6.306mm/frame²；50步最深入地0.403mm。
- 50步optimizer-call（含模型/损失初始化，不含source/floor准备和CPU/export评估）均值5.424s、最大6.834s，19/25>5s。不能宣称5秒目标达成。更多步数确有收敛收益，后续若地面稳定再做等价加速。
- 补看30步逐样本分布：23/25的已观测原支撑脚高最大偏离<2mm，仅2例明显失败（5.968/20.062mm）；这里忽略无原支撑观测字段，不能当双方全已验，更不是最终foot skating+penetration通过率。研究目标允许过滤失败增强，不必追求所有样本成功，但合格样本必须守住地面和最终双指标。
- 同soft/originalref/ground200/temporal1/lr.001/50步配置，GPU1 offset25/limit75 session64833已启动，输出`output/ssp62_ground_original_soft_temporal1_w200_iter50_next75_20260912`。地面未最终验收，吸引仍暂停。
### 09-12｜时间平滑抑制尖峰但脚高恢复退化；50步收敛对照启动

- temporal1优化session4729 exit0，独立评估session47792 exit0，`output/ssp62_original_support_original_soft_temporal1_w200_first25_20260912.json`。同25 soft→soft+temporal1：最大原支撑脚高偏差5.644→20.062mm（退化），越1cm观测30→49；速度误差18.451→6.763mm/frame、二阶差误差27.857→6.651mm/frame²，接近换shape初始化6.557/6.491。最大入地1.057→0.573mm。不能只凭平滑提升晋级。
- 低成本agent temporal CPU边界测试exit0（常量/线性修正零二阶惩罚、非线性原动作自身不罚、spike/短序列/有限梯度）。另一次只读确认旧6238两个Volume候选PID4126281/4130516仍活跃，gate0/gate1_cuda0未生成，不打包。
- 按Ponytail仅给独立checker增加iterations override与实际optimizer-call计时，报告记录实际配置，默认仍原manifest；计时包含优化器模型初始化，不含source/floor准备和CPU/export评估。新对照只将30→50步，soft/originalref/ground200/temporal1/lr.001不变，前25 GPU1 session71510启动，输出`output/ssp62_ground_original_soft_temporal1_w200_iter50_first25_20260912`。用于检验收敛不足，不先宣称5s达标。
### 09-12｜soft残余尖峰定位到腿脚姿态；测试修正量二阶平滑

- root CPU session11855 exit0，`output/ssp62_soft_temporal_worst_G008_20260912.json`。`G008T001A005R001`人1左脚源第37帧：原/初始化/hard/soft足高二阶差分别-4.189/-3.796/-46.063/-32.046mm/frame²；踝Y为-1.043/-1.663/-52.373/-39.312；root Y为+0.526/+0.526/+1.177/+4.576。soft尖峰确有额外腿脚姿态变化，不能解释为仅与原轨迹不同或导出origin误差。
- 按Ponytail在实验ground项新增显式temporal_weight（默认0保持旧模式），惩罚脚部顶点高度修正(current-original reference)的二阶差：逐帧顶点最大绝对二阶差再时间平均。不直接把原动作加速度压零、不改水平轨迹/吸引/其他loss。新试验ground200、temporal1，等效该附加项也乘ground200，其余soft/30步/lr.001不变。
- 内置spike数值与有限梯度、角色参考测试root exit0；新前25 GPU1 session4729启动，`output/ssp62_ground_original_soft_temporal1_w200_first25_20260912`。低成本agent补常量/线性修正、短序列和非线性原轨迹CPU测试中。地面与时间连续性均待复核，未晋级。
### 09-12｜平滑参考25例减少突变，但仍未验收

- soft优化session51996 exit0，独立原支撑+时序CPU session94576 exit0，产物`output/ssp62_original_support_original_soft_w200_first25_temporal_20260912.json`。低成本agent确认height_decay在1cm边界连续、高处candidate=reference时零损失、同偏差随高度减权、有限梯度和旧frame_max保持，CPU exit0。
- 同25硬→软参考：原支撑最大偏差4.929→5.644mm（略差）；越1cm观测39→30；最大足高速度误差29.781→18.451mm/frame、最大二阶差误差41.874→27.857mm/frame²；最深入地1.628→1.057mm。只是局部改善，不作最终地面/足滑/穿透验收。
- 硬参考next75原支撑/时序CPU session54129已exit0；硬参考完整100例合计最大原支撑偏差17.015mm，速度误差57.275mm/frame，二阶差误差91.020mm/frame²，越1cm观测170，证明硬参考仍有尾部失败。
- 下一步只定位soft剩余最大时间突变，决定是否需要约束竖直修正的时间变化；暂不继续扩大soft100或恢复吸引调参。
### 09-12｜平滑高度衰减支撑约束25例启动；补时间连续性指标

- 原硬参考next75 session8048回收exit0后才改实验共享代码，完整产物保留。独立support-transfer评估补原/候选/误差的一阶二阶足部最低Y差分（m/frame、m/frame²）；内置及低成本agent CPU检查exit0，覆盖短序列None、常量offset抵消、单帧突起。硬参考first25完整时序评估session9770 exit0，最大速度误差29.781mm/frame、最大二阶差误差41.874mm/frame²（相对原shape，不是foot skating指标）。
- 按Ponytail只扩展实验损失一个显式`height_decay`分支：权重exp(-relu(original_vertex_y-0.01)/0.02)，逐帧加权顶点最大偏离再全帧平均，目标仍是原shape逐帧顶点高度，不把抬脚强压y=0。1cm内权重1，之后按2cm尺度连续衰减；归一化由有硬支撑帧均值变为全帧均值，报告明确记录，不能称数值等价。ground权重200和其他loss/30步/lr.001保持。
- soft损失数值/有限梯度内置及角色参考测试root重跑exit0。新`mesh_support_original_soft`前25 GPU1 session51996启动，输出`output/ssp62_ground_original_soft_w200_first25_20260912`。硬参考next75原支撑/时间诊断CPU session54129另行启动，输出`output/ssp62_original_support_original_ref_w200_next75_temporal_20260912.json`，不会覆盖旧报告；地面仍未验收。
### 09-12｜网格轨迹复核发现硬支撑边界产生突变

- 当前75例session8048/PID4149837确认存活，未改配置。root CPU session98748 exit0，重建最坏`G052T000A004R027`人2四条件网格并实际查看`output/ssp62_ground_reference_case_G052_20260912.png`，附同名JSON。源第25帧左脚最低Y：原shape9.460mm、新shape初始化50.341mm、新shape参考优化62.078mm、原shape参考优化4.611mm。投影仅几何诊断，不作完整视觉形变验收。
- 图中原shape参考在进入/退出近地支撑段时有陡变。root额外CPU session89968 exit0，`output/ssp62_ground_reference_case_G052_temporal_20260912.json`：原shape→原参考优化，足部最低Y最大逐帧差7.162→30.772mm/frame、最大二阶差4.137→29.725mm/frame²；root Y最大二阶差3.199→14.847mm/frame²；踝Y最大二阶差11.198→18.123mm/frame²。常量导出origin在差分中抵消，非坐标解码误差。
- 原支撑高度虽恢复，但硬mask开关带来时间连续性问题，当前方案不能仅按静态地面误差晋级。下一步地面约束需要平滑支撑边界或平滑竖直修正，并保留原shape轨迹参考，不能逐帧强制所有脚贴y=0。吸引仍暂停。

### 09-12｜原shape支撑参考25例显著改善，扩大剩余75例

- 优化session43554 exit0，原支撑时序CPU评估session33361 exit0，`output/ssp62_original_support_original_ref_w200_first25_20260912.json`。与旧w200前25名字及sidecar SHA相同；低成本agent独立NPZ核对50/50 beta与gender完全一致（exit0）。
- 新shape参考→原shape参考：原支撑时刻最大高度绝对偏差52.619→4.929mm；原2835个近地观测中优化后越1cm数量190→39。最坏仍`G052T000A004R027`人2，但最大向上偏差降到2.259mm、原71支撑观测中越界60→6。全25最大入地0.521→1.628mm，有退化，尚不宣布验收。
- 全身body轴角参数差最大范数0.053750→0.056005rad；仅参数调整幅度，不是旋转geodesic或视觉形变证明。保持原shape参考/weight200/30步/lr.001等配置，GPU1剩余offset25/limit75已启动session8048，输出`output/ssp62_ground_original_ref_w200_next75_20260912`；不重复前25，不恢复吸引实验。

### 09-12｜原shape支撑参考接入实验并启动25例

- w200 next75 session46229回收exit0；与first25合计100唯一名字。按旧new-shape support口径，最深入地2.433mm、最大支撑偏离14.281mm、双方支撑可观测且两项≤1mm为70/100；因原支撑时序盲区，不作地面验收或augmentation通过率。原支撑时序next75额外CPU诊断session9208启动，输出`output/ssp62_original_support_w200_next75_20260912.json`。
- 按Ponytail在既有实验类增加可选reference_params；仅`mesh_support_original`模式传入原shape按自身full-sequence mesh floor恢复后的参数，约束原shape近地脚部顶点的高度与时序。权重200、frame_max、30步、lr.001及其他loss不变，默认旧模式不变。按self.swapped同步交换外部原shape参考；无外部参考时沿用已经交换的init，不二次交换。此为原shape网格参考，不是真实测量接触标签。
- 低成本agent新增mock-only `tests/test_mesh_ground_reference.py`；root修正直接执行import路径并补None+swapped True，重跑exit0，覆盖角色顺序、缓存、旧init及非ground tuple保持。GPU1启动前20MiB空闲，新25例session43554，输出`output/ssp62_ground_original_ref_w200_first25_20260912`。之后必须用独立original-support诊断评估，checker旧after字段仍是new-shape参考不能混用。生产路径/旧产物未改。
- CPU next75原支撑诊断session9208随后回收exit0，完整产物存在；新GPU session43554已取得前几个样本有效输出，仍运行。

### 09-12｜原shape固定支撑诊断揭示新shape自洽指标遗漏

- 按Ponytail复用已有mesh/坐标/原始payload函数，独立新增`tools/diagnose_shape_support_transfer.py`，固定原shape足部最低Y≤1cm的时序，不随候选脚抬高而取消。检查source/floor/sidecar绑定、双目录重复名拒绝，最终NPZ只加对应common_export_origin。内置及低成本agent CPU边界检查exit0（原mask、跨阈值、无support、非有限/shape不匹配）。
- w200前25例CPU session93164 exit0，`output/ssp62_original_support_w200_first25_20260912.json`。原有2835个近地足部观测，换shape初始化丢失225，优化后仍丢失190；这些是阈值观测数，不是真实接触标签。
- 最坏`G052T000A004R027`人2：原有71个近地观测，初始化/优化后均60个越界，原支撑时刻最大脚高50.341→62.078mm；相对原shape高度最大偏离40.881→52.619mm。说明new-shape-only参考存在盲区，先前w200新shape稳定性改善不能证明保住了原支撑。下一组应修正支撑参考，非继续只加权；吸引调参仍暂停。

### 09-12｜无支撑17人体复核：2人体换shape后丢失近地标记

- 75例w200原session46229/PID4147096确认运行，未改配置。等待期间root仅CPU对17个新shape初始无近地支撑人体，重建同crop原shape/新shape，各用自己的已验证full-sequence mesh floor。初次脚本误将source_shapes对象按列表读，session84643 exit1且无产物；修正为betas/genders字段后session66242 exit0，17行输出`output/ssp62_no_initial_support17_shape_audit_20260912.json`绑定旧报告/floor SHA。
- 其中15人体在原shape下也无≤1cm足部观测，不能仅凭此断言真正悬空；2人体原shape有而新shape没有：`G029T006A026R006`人1原19观测、最低6.262→13.536mm；`G013T000A001R010`人2原29观测、最低7.843→14.939mm。证实new-shape-only阈值会丢失原有近地标记，强权重不能约束未进入mask的时刻。
- 下一步地面方案须对照原shape支撑时序，不可仅以新shape自洽判成功。1cm是几何诊断阈值，不是真实触地标签；不通过随意放宽阈值提高通过率。当前75例继续作为固定协议对照，地面验收仍未完成。

### 09-12｜地面权重200固定25例改善，剩余75例验证启动

- session71509回收exit0。root对照w20完整100报告前25，顺序及input sidecar SHA逐一相同；w20→w200最深入地5.104→0.521mm，最大同帧支撑偏离12.539→1.425mm。双人均有支撑观测且入地/偏离≤1mm为14→19/25；两模式都3/50人体无支撑观测，不计作通过。不是用户最终足滑/穿透通过率。
- 独立低成本agent对当时19对已完成样本（38人体）读取pose_body，最大轴角分量调整的逐人均值0.030360→0.029970rad、RMS均值0.005427→0.005216rad；仅参数幅度初查，不是geodesic、视觉质量或完整25例形变验收。权重constructor CPU mock检查exit0，默认/其他权重保持且拒绝NaN/inf/非正值。
- 按Ponytail仅在既有checker增加offset及完整数量/唯一性/floor覆盖断言，避免重跑25例。原固定100中offset25/limit75、同w200配置在GPU1 session46229启动，输出`output/ssp62_ground_support_max_w200_next75_20260912`；与first25合并后仍需验证名单无漏无重。地面尚未最终验收，吸引保持暂停。

### 09-12｜100例地面检查完成失败；权重单因素25例启动

- session23171已回收exit0，`output/ssp62_ground_support_max_first100_20260912/report.json`完整100例/200人体。root复核最大入地20.408mm、最大支撑高度绝对偏离26.155mm；低成本agent统计有效支撑偏离中位0.560mm、P95 13.762mm。17/200人体（15样本）crop内没有初始≤1cm支撑观测，不能按“没有漂移”验收。
- 双人都有支撑观测且入地/支撑偏离均≤1/2/5mm的样本分别53/61/66（分母始终100），仅地面诊断阈值，不是用户foot skating+inter-person penetration的augmentation通过率。当前方法不晋级。
- 按Ponytail复用实验类，增加显式ground_weight，默认20不变且拒绝非正/非有限值。新实验仅20→200，其余frame_max/30步/lr.001/其他loss不变；增加root Y最大位移和body轴角差最大范数用于形变复核（不是正式质量指标）。内置CPU检查exit0；生产优化器未改。固定前25例GPU1 session71509已启动，`output/ssp62_ground_support_max_w200_first25_20260912`，未覆盖旧实验。等待其结果，不恢复吸引调参。

### 09-12｜12.54mm抬脚反例定位：支撑未漏标，优化仍抬高root

- 子agent初次CPU复核混淆初始化与导出origin，未取得有效帧级证据；其将after最高18.725mm称作initial也是错误，不能沿用。root按source H5+donor shape+full-sequence mesh floor重建初始化，最终NPZ仅加自己的common_export_origin，独立CPU session19962 exit0。
- `G046T005A026R010__shapeaug_s00000123`第二人，crop第112帧/源第162帧，左脚最低Y 5.947→18.486mm，抬升12.539mm。该帧左脚有16个初始化≤1cm的受约束顶点，其最大绝对偏离13.057mm、平均上升12.698mm；右脚无近地支撑顶点。因此该反例不是support漏标，也不是另一脚抢占max；约束存在但优化仍违反。该帧root Y上升15.802mm（XYZ变化7.306/15.802/-14.342mm），支持主要竖直平移参与抬脚，但不排除姿态变化。
- 100例session23171继续原配置，不中途改变权重或覆盖报告。后续地面修正须针对实际约束违反，而非继续只调整mask；等待完整分布后设计下一组。

### 09-12｜100例地面检查运行中已发现逐帧反例

- 原session23171/PID4144494核实存活，未重启。前13例已落盘时，最大同帧同脚支撑高度偏离12.539mm，来自`G046T005A026R010__shapeaug_s00000123`第二人；这一批最深入地1.814mm。只是中途反例，不是100例最终统计或通过率。
- 当前frame_max方案仍不足以地面验收；低成本ground_max_check进行该反例CPU网格定位，区分初始化support遗漏与约束力度不足。主agent核对forward：角色交换时init/current同步交换，ground项确实进入总loss，未被temporal mask整体关闭。保持其余项不变，等待完整样本分布再决定地面约束改法。

### 09-12｜支撑损失改为逐帧最大偏离：抬升改善，仍未验收

- 上一轮为责任/状态说明，未推进修复；本轮继续地面单因素试验。保留mean模式，新增显式`mesh_support_max`，每帧仅对初始化近地脚部顶点取最大绝对高度偏离，再对有支撑帧平均；其余loss/权重/30步/lr.001保持不变。Ponytail最小改动，仅实验模块与诊断入口，生产路径未改。内置CPU检查exit0，独立低成本agent复核中。
- GPU1 session83149 exit0，`output/ssp62_ground_support_max_first4_20260912/report.json`。8人体最深入地1.800mm（mean模式1.393mm），原支撑观测最高12.965mm（mean模式31.843mm，released33.080mm）；输入该人体最高9.973mm。不能把整体最大值之差当逐帧最大漂移，也不能声称地面完全修复。输出网格往返误差最大1.1921e-7m。
- 下一步固定100例扩大检查，并补同帧同脚支撑高度delta的max abs/up/min，防止只看聚合最大值。地面未验收，不恢复吸引实验或晋级旧增强。
- 固定100例已启动session23171，输出`output/ssp62_ground_support_max_first100_20260912`；GPU1启动前20MiB空闲。低成本ground_max_check确认CPU边界检查exit0（部分帧支撑、未支撑顶点排除、无support、梯度、mean兼容）；直接文件执行需PYTHONPATH，推荐已验证`python -m optimization.mesh_ground_support`。

### 09-12｜地面约束试验改善入地，但局部支撑抬升仍未过关

- 仅地面试验：新增独立`optimization/mesh_ground_support.py`，用每帧最深mesh入地的时间均值替代全部顶点均值，并对初始化距y=0≤1cm的脚部顶点保持其初始高度（绝对误差平均）。原ground权重20及其他loss/30步/lr.001均不变。按Ponytail在独立诊断进程用scoped patch替换loss类，生产优化器/旧评分源文件没有改动；新模式`--ground-policy mesh_support`显式写入报告。
- 内置零参考、梯度、单点入地不被顶点数稀释检查exit0；低成本physical_ground_test独立检查无support、增10倍地上顶点、上下偏离及梯度均通过。4例session97895 exit0，产物`output/ssp62_ground_support_first4_20260912/report.json`。
- 8人体优化后最低meshY（mm）：-0.021/-0.132、+0.086/-0.343、-0.056/+0.085、-1.393/-0.158；最深入地由12.406降为1.393mm。但样本4第二人原支撑观测最高仍31.843mm（旧33.080），局部漂移未解决，不能晋级。其余7人体原支撑观测最高8.434–10.048mm；输出往返断言通过。
- 下一步仍只查地面约束：支撑顶点平均误差可能稀释局部漂移，需要以局部最坏支撑偏离复核，而不是只看mean ground loss或把新结果当完全修复；吸引优化继续暂停。

### 09-12｜实际优化地面检查未通过：初始化修正后仍可入地/浮空

- 新独立`tools/check_shape_ground_after_optimization.py`复用原优化器；只替换为已验初始full-sequence mesh floor，其余30步/lr.001/metric-foot/frame_sum/released-aux/nearest等保持原配置，未调吸引、不汇报吸引收益。GPU1检查时空闲，4例session16665 exit0，产物`output/ssp62_ground_optimization_first4_20260912/report.json`及编码NPZ。不是原算法200步比较，也不是公平计时实验。
- 8人体优化后最低mesh Y（mm）：-0.616/+0.397、-12.406/-3.107、+2.898/+0.067、-3.128/-10.392。最大原支撑观测高度33.080mm（同一人体初始最大9.973mm）；支撑mask仅初始化足部region最低Y≤1cm的诊断定义，不宣称真实脚触地标签。未做输出再落地投影掩盖问题。
- 最深12.4mm入地人体的ground loss仅8.44e-7，结合源码对全部顶点/帧取均值，提示局部入地惩罚会被稀释；现有ground仅单侧地下惩罚，也没有支撑脚高度保持项。下阶段仍只处理地面约束，不能仅以初始化恢复通过宣称地面已修好。
- 全部编码NPZ重新读出、用保存的共同origin解码，与优化末态mesh最大差≤5.96046e-8m，故这些入地/浮空来自优化而非本轮序列化；尚未接生产训练loader。已立即向用户报告未通过，旧增强继续不晋级。

### 09-12｜固定100例初始化网格地面与序列化往返通过

- 原CPU session33887回收exit0，完整产物`output/ssp62_mesh_ground_roundtrip_first100_20260912.json`。root核对200条件行、400人体（100例×原/新shape×2），最大解码后网格最低Y绝对值1.1920929e-7m，最大网格往返误差同值，最大初始ground loss9.98277e-14；最低顶点400/400在脚部区域，姿态/beta及帧间位移保持断言全通过。
- 证明范围仅“完整序列固定网格平移能去除该100例的初始化地面坐标伪惩罚，且共同origin的NPZ往返保持几何”。尚未接入生产runner/训练loader，也未验证全损失优化后的支撑高度、浮空/入地及导出约定，因此地面任务尚未完成；继续只处理地面，不恢复吸引调参、不晋级旧增强。
- 已向用户明确此前失误：本轮用共同平移替换逐人落地却没有补齐地面物理验收，不能以旧raw穿透改善宣称正确；旧70.52%等仅旧坐标诊断，未用于新VQ/MGT训练，原数据未覆盖。后续实质坐标/损失语义改动须在采用前明确说明风险和验收范围。

### 09-12｜网格地面初始恢复及共同原点往返4例通过，扩展100例

- 仅处理地面，按Ponytail新增独立`utils/interx_physical_ground.py`及`tools/diagnose_shape_physical_ground.py`，不改旧评分/优化源文件。完整序列、真实gender/beta网格最低Y决定每人常量竖直平移；编码仅共同减P1首帧origin，解码加同一origin，不分别恢复旧高度。原始姿态/beta保持、时间差分保持。完整序列最低点是支撑面假设，仍需识别悬空/物体支撑，不宣称恢复原始捕捉绝对世界坐标。
- 低成本physical_ground_test新增最小测试，root复读重跑exit0；覆盖不同floor、姿态/beta/输入不变、帧间位移、双方相对坐标和往返、NaN/Inf/空trans拒绝。实际CPU4例session46420 exit0，`output/ssp62_mesh_ground_roundtrip_first4_20260912.json`：4序列×原/新shape×2人=16人体，重新SMPL解码最低网格Y全部0，ground loss全部0，最大mesh往返误差1.19209e-7m；最低点16/16在脚部区域。
- 以上只验证初始化坐标恢复和NPZ序列化往返，不是优化后地面约束、实际训练loader集成或全量验收。固定同名单前100扩大CPU验证session33887已启动，输出`output/ssp62_mesh_ground_roundtrip_first100_20260912.json`，仍不做接触/吸引实验。旧全量评分仅保留诊断，晋级保持暂停。

### 09-12｜按用户顺序只处理地面；确认关节落地制造地下网格

- 用户明确“先看地面关系再看吸引优化”，接触扩展停止；此前只读接触检查2294已exit0，不再追该方向。当前仅地面问题，旧全量评分保留诊断，不晋级。
- root只读CPU session98618 exit0：固定selection前4个完整序列、原shape与新shape各2人，以全10475顶点和55关节分别取最低Y。若按每人最低关节置0，网格最低点仍在地下28.19–52.50mm；新shape8个人体为32.67/32.41、52.50/44.25、32.92/31.58、52.40/35.59mm。此为4例诊断，非全量统计。
- 这与`ground_loss=mean(ReLU(-verts_y))`存在明确坐标定义冲突：最低关节不等于脚底表面，初始化便会人为触发地面惩罚。接下来只核对网格支撑面恢复：每人整序列固定竖直偏移、共用y=0物理地面，不改姿态/逐帧高度，不把关节中心强压到平面；验证原shape/新shape、优化前后及导出往返，地面验收后再回吸引问题。

### 09-12｜固定标注区域距离揭示loss下降不等于接触改善

- 按Ponytail扩展同一CPU工具，固定每条标注的p1/p2 RID union、raw新shape初始角色及timing>0帧，新增无符号最近顶点距离、双向最近5%距离、timing加权1/2/5cm接触帧比例。它们仍是proximity诊断，不是穿透/标注正确性验收，不动态切换标注union。内置单点15mm/零timing检查通过；低成本contact_evidence额外CPU非均匀权重、共同平移、空contact和未知RID拒绝检查exit0。
- 实际4例session75686 exit0，产物`output/ssp62_fixed_contact_coordinates_first4_20260912.json`。raw坐标的每条接触平均最近距离随机→优化：样本1 112.19→123.35mm（变差，5cm帧比例33.3%→0）；样本2 245.19→221.53mm（仍无5cm接触）；样本3 8.26→2.77mm（改善，1cm比例81.25%→100%）；样本4的两关系3.08→5.78mm（2cm比例100%→95.45%）和309.82→266.92mm（仍无5cm接触）。所有样本动态吸引loss下降并不保证固定接触目标改善。
- 全序列逐人floor诊断下同样存在不一致：样本1 112.10→123.30mm、样本2 243.63→220.03mm、样本3 9.03→3.00mm、样本4 3.22→6.00及308.95→266.09mm。不是换坐标即可消失的问题。
- 多条标注关系原GT本来就远（raw样本2约262mm、样本4第二关系约359mm）；下一步优先核对标注语义/角色/每条接触时段，不能把所有远距离都归因lr或强行吸引错误关系。旧全量结果仍不晋级，不宣称4例代表全量原算法对比。

### 09-12｜4例CPU接触吸引诊断完成：有下降，不等于接触达成

- 按Ponytail复用原`DyadicInteractionLoss._calc_semantic_contact_loss`，新增只读产物诊断入口`tools/diagnose_shape_contact_coordinates.py`；不改优化器。source/records/contact/timing/候选hash绑定，角色固定为raw新shape初始化的选择；完整序列按32帧算每人joint floor，GT用自身shape、随机/优化共享新shape初始floor，不对最终输出重新落地。内置平移往返检查，实际4例session29140 exit0，产物`output/ssp62_contact_coordinates_first4_20260912.json`。
- fixed selection前4例均有接触。原始raw坐标下随机shape→30步优化的动态区域吸引loss：0.112215→0.105522、0.504724→0.456472、0.054802→0.008030、0.279820→0.267634。完整序列逐人joint-floor诊断坐标对应0.115286→0.105504、0.501440→0.453763、0.022384→0.010019、0.273680→0.267774。4例均下降，否定“30步完全关闭吸引”的说法，但下降幅度不一，不能称接触已达成或全量有效。
- root细读实际语义路径发现使用ReLU(signed_distance)后top-k/区域竞争，不是绝对无符号接触距离；穿透负值被置零、区域可改变，因此loss下降仍不等于固定接触部位正确贴合。纠正此前子agent引用另一显式contact分支abs公式的适用范围。该诊断不含接触召回/固定顶点对达成率，不能替代后续验收；joint-floor恢复同样仍是诊断假设，不直接升级为生产恢复方案。

### 09-12｜官方高度预处理及实际H5中性体型重建核对

- 低成本`interx_height_source`定位，root完整复读`/home/wenning/projects/Inter-X/preprocess/1_prepare_data.py`（另一checkout内容相同）：先忽略gender/betas，用neutral零shape计算整序列关节，逐人减其全序列最低Y（64行）；随后双方共同减P1首帧完整trans（70–74行）。并非双方root逐帧/分别置零；P2仍保留相对P1首帧位置。split仅复制H5，不重新归一化。
- 只读CPU session6411 exit0，固定selection前4个原始完整序列（169/71/123/133帧）用neutral beta0重建，双方55关节全序列最低Y差为0、0、1.192e-7、0m；P1首帧rootY全0，P2分别0.03908/-0.21126/-0.01570/0.04695m。支持这4例H5与官方中性体型共同地面约定一致，不外推所有资产已查完。
- 因此诊断进一步收敛：原H5在neutral体型下确有共同地面，但应用原始真实beta/gender或SSP新shape后支撑高度改变；common shift不能补偿shape引入的逐人高度差，原始真实shape GT的raw物理评分也须复核。不能把先前raw坐标穿透下降直接归因于物理修复。后续区分全序列地面恢复与crop内最低点、原shape/新shape支撑校正，并保证优化输出往返训练表示不再次破坏接触关系。
- 未改算法或评测合同，旧全量评分仍仅诊断；继续保留地面/接触验收晋级暂停条件。

### 09-12｜用户指出共同地面及接触吸引验收缺口

- root复读地面变换/评分/loss源码：common shift仅令两人联合最低点为0，保留两人各自最低点差；ground loss是ReLU(-vertex_y)，不约束悬空；foot评分每人独立估计floor，不验证共同地面。此前将“保留raw相对高度”视作恢复正确物理关系依据不足，不能继续据此声称shared-ground改动已物理正确。
- 只读CPU几何检查session86719、38621均exit0，固定selection前4例、全crop/55关节。仅换shape→优化后两人最低关节高度差依次0.596→0.603mm、7.330→5.488mm、13.339→15.700mm、1.252→15.144mm；最低关节不是脚底接触真值，4例不是全量结论，但已证实算法没有强制共同floor。前100例raw root crop均值中位数两人约-0.144/-0.657mm，非每帧严格0，不能凭此还原数据预处理全过程。
- 低成本contact_evidence只读检查：现有sidecar/生成日志没有attraction/contact距离前后值；root确认attraction项仍加到total，但30步不会触发freeze_epoch=100的锚点锁定。小lr不等于loss关掉，也不能据足滑/穿透通过率证明接触拉近有效。下一步先查坐标来历和正确共同地面解码，再在固定接触部位/时序上比较GT、换shape、优化后接触距离/召回与loss；不将当前实验晋级为已验收增强。未修改正在运行的算法/评分、未启动新GPU任务。

### 09-12｜全6238例原GT Volume完成，两路均评分候选

- root核对`volume_gt0/metrics.json`与`volume_gt1_cuda0/metrics.json`均status=complete、sample_count=3119，原GT合计6238完整。shard1原GT PID4109284退出，父4109283/session24091自动启动候选PID4130516，输出`volume_candidate1_cuda0`。
- shard0候选PID4126281及两父4093560/4109283本轮ps均活跃；检查时candidate0=786、candidate1=39，两个候选manifest和最终gate尚未完成。HDD剩余8.7GB。继续原队列，尚不能报告最终交集或打包/开始原算法基线。

### 09-12｜原算法100例配对名单预检通过

- root只读CPU检查exit0：同全量计划records hash、selection seed20260911/offset0/limit100，100/100与shard0完成清单前100顺序完全一致；逐例plan_record hash、原名字、crop起止、timing hash、两人donor beta/gender/shape_id全部吻合。该名单96例positive、4例zero，不按通过率挑选。
- 有序名字JSON（紧凑分隔）SHA256=`e0f653053d4b559237b3ceb9f9e22f6a7976d20fceffc61bf340d56e13fec995`。这只证明待运行基线输入可与现有结果配对，非原算法已运行或已取得成功率；仍待全量评分完成后再跑。

### 09-12｜全量shard0原GT Volume完成，转入候选评分

- 低频复查确认`volume_gt0/metrics.json`已生成、samples完整3119例；原GT PID4107127退出，原父4093560/session86163仍活跃并自动进入candidate评分，新PID4126281，输出`volume_candidate0`（检查时71例）。不是全pipeline已完成。
- shard1父4109283/session24091及GT PID4109284仍活跃，`volume_gt1_cuda0`检查时2449/3119，之后仍顺序执行candidate1。gate0/gate1_cuda0均待生成，最终全量交集仍待评；未重复评分、未提前启动原算法基线。

### 09-12｜原算法基线源码核对完成

- 低成本agent只读核对，root复读确认：`/home/wenning/projects/MMM/optimize_full_body.py`函数默认200步/lr0.01，但其文件末尾示例实际100步；两人分别ground alignment，freeze_epoch=100，原始loss权重attract/align/pene/reg/smooth/skate/ground=1/1/5/1/0.1/100/20。主基线明确标为“原算法函数默认200步”，不把它称作原示例100步配置。
- HHC runner调用本地优化器且保留KNN和模型NPZ读取加速，因此后续耗时必须标“原算法配置、本地加速实现”，不能声称原仓库原始wall time或纯算子加速倍率。配对实验保留统一接触输入及原坐标导出规则，比较此augmentation任务中的算法效果，不声称逐字复跑原示例。
- 全量父4093560/4109283及GT Python4107127/4109284本轮ps仍活跃。原算法100例尚未启动，按用户要求等全量评分结束后执行。

### 09-12｜用户追加：全量评分后比较原算法小批量基线

- 在现有全量评分结束后、进入第③项前，补原始optimization配对基线。预定固定全量selection seed20260911、offset0、limit100（不按成功/失败挑样），复用同62-donor计划，逐例核对名字、crop、两人shape和原GT完全相同；当前算法结果取全量对应100例，不重生成。
- 报告原始GT→仅换shape→原算法/当前算法的foot和penetration绝对值、变化、严格逐例通过率，nearest-vertex与Volume交集分开；时间区分纯optimizer与运行/I/O，完整分母包含失败样本。100例仅作小批量对比，不外推全量原算法通过率。
- runner默认显示原设置200步/lr0.01、per-person ground、released_dynamic foot、penetrating_vertex_mean和released_broadcast aux；低成本`original_optimizer_audit`核对上游源码与计时可比性后再冻结基线，不能只凭默认参数或metadata标签宣称完全原版。按Ponytail复用现有工具，不改正在运行的脚本、不提前占GPU。

### 09-12｜全量评分低频监控交接

- 用户询问时间/通过率后已完整报告：平均纯优化3.861s、含主要I/O的run均值3.981s；全量proxy4399/6238=70.52%，不是尚待完成的全量双估计器交集。246/6238处理耗时超过5s，不宣称逐例达标。
- 本轮ps确认原父4093560、新父4109283及GT评分4107127/4109284均仍活跃；HDD约8.7GB可用。不重复启动、不修改运行脚本。简单监控交给低成本`full6238_monitor`（gpt-5.6-luna/low），至少10分钟检查一次，仅阶段切换/异常通知；root负责最终gate0+gate1_cuda0验收和打包。当前仍第②项，未提前启动第③项。

### 09-12｜迁移评分71例跨设备结果完全一致

- 新shard1 GT评分PID4109284在cuda0活跃约11分钟，已写95例；旧cuda1停止前保留71例（此前18分钟检查为63例）。这是不同时间窗的进度，不能直接当成严格同负载测速；当前两个评分进程共享GPU0，shard0也会受资源竞争影响。GPU0显存18683/24564MiB，HDD约8.8GB剩余。
- root逐一比对旧cuda1与新cuda0交集71例：输入p1/p2 hash、seed、frames全部一致；foot、平方穿透、深度和、条件深度、碰撞帧率、最大深度、穿透顶点比例七指标最大绝对差全部0。CPU检查exit0，支持当前迁移不改变这些已比对结果；不外推所有剩余例已一致，不拼接旧目录评分。
- 有效评分句柄仍原session86163（shard0）及新session24091（shard1），全部GT/candidate Volume未齐，继续等待。

### 09-12｜GPU1评分严重竞争，迁移本任务shard1到GPU0重新配对评分

- 低频检查：GPU0 GT Volume32分23秒715例；GPU1 GT Volume18分06秒63例，速度差明显。只读追踪本地backend、评分器、已安装attach_volume主路径，模型/输入显式传device，未发现主路径硬编码跨GPU计算证据。按Ponytail未改评分公式、模型或现有运行脚本。
- 精确确认PID4107856为本任务`references1/original_gt → volume_gt1 --device cuda:1`后，仅对此PID发送SIGTERM；随后ps确认消失，原score1.log记录Terminated。旧`volume_gt1`部分产物完整保留，不删除、不混入新设备评分。GPU1其他run_9fold及GPU0暂停训练均未操作。
- 新session24091、shell PID4109283、GT评分Python PID4109284，统一cuda0顺序执行shard1的GT→candidate→双估计器gate。新路径`output/ssp3d_all62_full6238_20260911/volume_gt1_cuda0`、`volume_candidate1_cuda0`、`gate1_cuda0.json`；日志score1_cuda0.log。复用同references1/同候选、全3119例、同seed/算法，仅device合同变更，因此GT/candidate一起重评，不拼接旧cuda1分数。是否提速待后续实测，不提前承诺。
- shard0原GT PID4107127和父session86163/PID4093560保持运行。原父任务会先等shard0完整评分/gate0，再因shard1被人工终止而退出失败，不会自动打包；后续须以gate0+gate1_cuda0汇总全6238分母/交集>60%后手动接续既有materializer和完整validator。生成耗时与已生成动作不受此次评分调度影响。

### 09-12｜全量proxy70.52%通过；Volume继续，确认GPU1并发负载

- 两片proxy报告完整，各3119，合计6238唯一请求；严格输入绑定gate session93625 exit0，产物`output/ssp3d_all62_full6238_20260911/proxy_gate{0,1}.json`。双指标2204+2195=4399/6238=70.5194%；足滑不增5202、穿透不增5273。全体均值原GT→随机shape→优化：足滑6.049045→6.157461→5.081298mm/frame；平方穿透proxy0.2586283033→0.6886273105→0.0982436606m²/frame。
- 分层：positive4146/5868=70.6544%，zero253/370=68.3784%。zero足滑4.416567→3.707269mm/frame，但平均穿透0.0170035412→0.0303353662m²/frame反而增加（353例不增，17例退化造成均值增加）；不能宣称所有分层均值均改善。最终仍仅纳入逐例双估计器都通过者，分母保留全部6238。
- 两片均转入原GT Volume评分：PID4107127(cuda0)、4107856(cuda1)，父session86163仍活跃；完整Volume和交集未出，不把proxy通过率当最终结果。磁盘约9GB剩余。
- 低成本agent一次只读资源检查：GPU1另有PID4104848 `run_9fold.py ... hetclassnn,udfgma`，约2326MiB/pmon SM56%，本任务PID4107856约1088MiB/SM42%；GPU1总util100%、非显存耗尽。确有算力竞争，可解释部分慢速但单快照不能证明全部长尾原因。未杀/暂停/重启任何其他进程；现有实际计时保留，不作独占GPU性能声称。

### 09-12｜全量shard0近邻评分完成，Volume评分继续

- 原父session86163/PID4093560仍活跃。shard0完整proxy报告`output/ssp3d_all62_full6238_20260911/gt_diagnostic0.json`已生成；root核对3119行与完成清单顺序一致，双指标2204/3119≈70.7%。这只是一个分片的proxy结果，非6238例总体或双估计器交集验收。
- shard0原队列已完成参考导出，当前GT Volume进程PID4107127(cuda0)运行；shard1原proxy进程PID4106200仍运行。磁盘约10GB剩余，未重复启动评分，最终需两片GT/candidate Volume及绑定gate齐全。

### 09-12｜全6238例生成完成，已进入完整物理评分

- 两生成PID4093583/4093584均退出；父session86163/PID4093560仍运行且已越过成功`wait_pair`进入评分（不是整个pipeline exit0）。完成清单各3119/3119、complete=true。root逐6238个sidecar验证唯一完整集合、覆盖62 donor、370零timing、两人共同floor、30步/lr.001、两人真实shape变化；CPU核验exit0。
- GPU0 optimizer均值3.764853s，sample processing均值3.854307、p95=4.334760、p99=4.518883、最大5.212599s，2/3119例>5s；GPU1 optimizer均值3.957583s，processing均值4.046960、p95=6.258969、p99=7.118816、最大11.399404s，244/3119例>5s。GPU1长尾出现在full-selection位置5234..6237，原因尚未确认，不能删异常值或声称每例<5s。run阶段均值分别3.884287/4.077183s（不含导入/预检/评分），平均速度目标达成，逐例硬上限未达成。
- 原队列新评分子任务shell PID4106197/4106198，当前proxy评分Python PID4106199(cuda0)/4106200(cuda1)，输出`output/ssp3d_all62_full6238_20260911/gt_diagnostic{0,1}.json`，日志score{0,1}.log。随后原队列自动导出GT、Volume GT/candidate、gate，再检查全量交集>60%并打包/验证。尚无完整6238例通过率，不以200例67%代替。

### 09-11｜134例真实tokenizer读取验证；纠正旧交接优先级

- 全量父PID4093560及生成PID4093583/4093584在本轮启动检查中均活跃，保持原session86163，不重复启动。全量尚未评分，暂无新增通过率结论。
- 低成本agent只读核对训练调用链，root用实际`dataset_interx_tokenize_historical_root.Text2MotionDatasetV2HHI`读取134例overlay（unit4/seed114514、root统计、strict_assets）。真实CPU命令exit0：134/134进入dataset，所有beta20维/gender精确匹配donor资产，17672裁剪后归一化帧全部finite、contact长度与token stride一致；原beta目录不存在这134个augmented名字，因此不是回退原shape。
- cache阶段从overlay all.h5取动作，训练阶段读新cache及overlay beta；原样本允许同名官方beta fallback，增强文件则在训练入口由materialization validator逐个hash/内容强制检查。已有机制足够，按Ponytail未修改loader、未启动MGT训练；val仍570条原名字、无增强。CPU读取验证不等于增强训练收益已验证。
- `AGENTS.md`及交接文档顶部旧“回到第①项”已被最新用户顺序覆盖，保留历史段落、明确当前第②项及进度文档位置，避免未来接手重启已收束探索。

### 09-11｜200例双估计器交集67%，134例训练overlay验证通过；全6238例启动

- Volume后处理session71955/89669均exit0，各GT/candidate完整100例；双估计器gate session25590 exit0。旧100：proxy76、Volume74、交集73；新100：proxy64、Volume65、交集61；合计140/139/134，分母固定200。两种穿透均是估计，不声称精确mesh交集真值。与4-donor轮97例crop不同的对比限制保留。
- Volume完整200均值原GT→候选：足滑5.829630→4.944827mm/frame（167例不增）；平方深度和0.1489088394→0.0271538832m²/frame（167例不增）；深度和3.958264→0.865516m/frame（171）；条件平均穿透深度14.444542→9.577427mm（152）；碰撞帧比例0.420397→0.384972（132）；最大深度41.527635→33.301796mm（140）；穿透顶点比例0.00846606→0.00246284（171）。额外指标不是每例全不退化。
- 按Ponytail复用materializer与训练侧validator：新增显式`--quality-gate`（每shard一个），shared helper重算绑定gate并保留全部requested/accepted/rejected。仅此显式模式支持原始非零起点/稀疏selection位置，旧无gate路径保持原连续性要求；不修改completed_names、不伪造全通过。gate补GT/candidate目录字段，便于重新验证完整评分证据。低成本agent补CPU反例，root重跑exit0：正常分母2、hash篡改、重复gate、请求集不一致、改decision再重hash均拒绝。
- materialize session89575 exit0，输出`dataset/interx_shape_aug_ssp62_confirm200_gated_20260911`（42MB）；完整训练侧validator session75359 exit0，包括motion内容hash、split、shape、原始外链、gate重算。200请求/134接受/66拒绝，原训练9110+134=9244，all11388+134=11522，val不变；只是当前200例的可用增强产物，不称全量完成。
- 新入口`tools/run_ssp62_full_shape_augmentation.sh` bash -n exit0。全计划root计数6238唯一动作（5868正timing、370零timing；纠正此前粗记372），冻结shared-ground/metric-foot/frame_sum/released-aux/lr.001/30步，全部换shape优化，不给零timing原样fallback。两个shard各3119，full-selection offset0/3119；不同于pilot正timing专属排列，全量各例仍与自身同crop原GT比较。
- 全量session86163、父PID4093560；生成Python PID4093583(cuda0)/4093584(cuda1)，日志`output/ssp3d_all62_full6238_20260911/generate{0,1}.log`。等待两生成都成功后才双GPU评分（避免污染另一GPU生成测速），日志score{0,1}.log；全量双估计器交集>60%才自动打包并完整验证到`dataset/interx_shape_aug_ssp62_full6238_gated_20260911`，不达标保留评分/失败名单。脚本已运行，不原地编辑、不重复启动。预计生成约3.4小时，评分另需数小时，按至少10分钟间隔监控。
- 空间预检HDD约13GB剩余，按实测候选35MB/200、参考66MB/200、overlay42MB/134估算全链约5–6GB，启动前要求>8GiB；未删除用户资产。GPU0原暂停训练16652MiB保留，GPU1启动前空闲。后续报告全量positive/zero分层通过率与实际速度，不从200例直接外推成功。

### 09-11｜62-donor生成完成、proxy通过70%；补双估计器验收

- 原生成session1392/57789均exit0，各100/100；后处理session71955/89669已完成proxy gate、参考导出及原GT Volume，当前运行候选Volume。proxy旧100足滑85/穿透88/双指标76通过，新100为82/79/64，全部200双指标140通过。原GT→候选均值：足滑5.829630→4.944827mm/frame，平方穿透proxy0.2227808133→0.0443085620m²/frame。
- sample processing GPU0均值3.928666/最大4.494154s，GPU1均值3.871697/最大4.709350s；run创建至完成摊销分别3.943798/3.886830s。run不含导入、预检、离线shape拟合和物理评分，不称整个pipeline端到端小于5秒。
- 重要比较限制：root逐sidecar核对发现与4-donor轮同名200例中97例crop不同。plan把crop与donor抽样共用rng，池大小变化会影响后续随机序列。当前每轮内部仍使用严格同crop原GT，70%有效；不能将66.5%→70%解释为扩大donor池的单因素收益。此前“名单/顺序相同”不等于crop相同。
- 新增`tools/build_shape_volume_quality_gate.py`，复用proxy gate，核验完整分母、参考/源/完成清单及NPZ hash、schema、评分协议、seed、帧数和有限非负指标；分别保留proxy、Volume通过率，打包候选取交集，不把神经SDF称作精确mesh交集真值。低成本agent补CPU检查，root修订隔离负例并加schema/评分帧数检查后session19042 exit0，py_compile exit0。
- gate队列session25590等待两组候选Volume PID4086423/4086457结束，再依次生成`output/ssp3d_all62_confirm200_shard{0,1}_two_estimator_gate_20260911.json`；尚未验收，不提前计通过。低成本agent只读核查既有materializer/validator如何最小接入质量筛选，不另建打包框架（Ponytail）。

### 09-11｜62-donor确认覆盖全部shape，完整200例Volume交叉评分已排队

- root检查双分片run contract：各100例、互不重叠，合并名单/顺序与此前4-donor确认200例完全一致；冻结optimizer设置相同。按新plan取这200例的donor IDs，实际覆盖全部62个shape，不只是池中有62但只采到少数。
- 原生成GPU0 session1392/PID4086213、GPU1 session57789/PID4086250确认活跃。各GPU后处理队列session71955/89669均等待两组生成PID终态，再依次做：100例原GT proxy评分→严格绑定proxy gate→导出原GT/random参考→100例原GT Volume→100例候选Volume。无需改运行中的代码、不打断暂停训练。
- 后处理输出前缀`output/ssp3d_all62_confirm200_shard0`/`shard1`，各附`_gt_diagnostic_20260911.json`、`_proxy_gate_20260911.json`、`_references_20260911`、`_volume_gt_20260911`、`_volume_candidate_20260911`。完整200例Volume用于检查proxy通过结论是否可靠，尚未完成、不提前晋级。

### 09-11｜62-donor冻结配置200例确认已启动

- full-train计划`configs/interx_shape_augmentation/ssp3d_all62_fulltrain_20260911/plan_manifest.json`生成session69556、root validator session7623均exit0：6238 eligible/5868 positive timing/62 donor。
- GPU0 session1392：offset25/limit100，输出`output/ssp3d_all62_confirm200_sharedground_lr001_iter30_shard0_20260911`；GPU1 session57789：offset125/limit100，输出`output/ssp3d_all62_confirm200_sharedground_lr001_iter30_shard1_20260911`。同selection seed20260911、shared-ground/metric-foot/frame_sum/released-aux/lr=.001/30步，主要实验变化是donor pool4→62；两GPU分片的计时分别报告。GPU0原暂停训练约16.65GB显存保持不变，优化单进程此前约2.7GB，未终止/恢复任何训练。

### 09-11｜共享地面200例66.5%通过；62组公开shape全池转换通过

- 生成session39265 exit0，200/200；评分+25例Volume队列session25793 exit0；绑定验收清单session58987 exit0（133/200=66.5%）。root复核全部shared offset相同、200条唯一完整且严格原GT双指标：旧100例71/100（此前逐人方案54/100），全新100例62/100。冻结GT脚mask双指标分别73/100、65/100，合计138/200；不混作主判定。
- 旧100原GT→候选：足滑5.986→5.122mm/frame、penetration proxy0.238394→0.142935m²/frame；新100为5.759→4.871mm/frame、0.206722→0.045128m²/frame。合计足滑5.872→4.997mm/frame、pene0.222558→0.094032m²/frame；足滑不增159/200，穿透不增166/200。完整结果`output/ssp3d_fulltrain_confirm200_sharedground_iter30_gt_diagnostic_20260911.json`及`output/ssp3d_confirm200_sharedground_proxy_quality_gate_20260911.json`。
- 200例处理均值3.864s、最大4.679s，run阶段摊销3.890s/sample，不含导入/预检/离线拟合/评估。新100仅略高于60%，仍4个probe donor，不能外推全shape池。
- 共享25例Volume复核双指标19/25，足滑24/25、平方穿透19/25不增；平方穿透均值0.181571→0.003335m²/frame、深度总和4.225525→0.193872m/frame、条件平均深度11.618→6.496mm、碰撞帧比例0.320227→0.294687、最大深度35.010→22.913mm。均值方向一致，不宣称每例每项全优。
- male补跑session79556 exit0；全池collector session78322 exit0，62/62接受、0拒绝（female21/male41）。root验证唯一shape ID、有限62×10beta、完整62源覆盖、NPZ hash、RMSE≤1cm；均值5.792674mm、最大8.971889mm。输出`output/ssp3d_donors_all62_20260911`。已启动生成62-donor full-train计划，保持现有冻结优化配置，下一步验证不依赖4-donor小池。

### 09-11｜62组shape转换全池审核已排队

- 上轮为实质进展，本轮原句柄确认200例生成session39265/PID4083318和male补跑session79556/PID4076733活跃。低频检查male已有34/41、female21/21个fit文件，未等同转换通过。
- 新队列session78322等待原male PID终态后，使用现有collector审核全部62组（GPU0可见、CPU主导，不抢GPU1生成测速）：有限精确10beta、全顶点RMSE≤1cm、每性别至少2个donor，接受/拒绝记录均保留；输出`output/ssp3d_donors_all62_20260911`。若补跑失败或缺文件，collector会失败而非将部分池误标完整。

### 09-11｜补齐评分输入绑定与严格proxy验收清单

- 200例生成仍运行，评分Python进程尚未启动；未改运行中的runner或shell。只给待运行的`diagnose_shape_optimization_gt.py`增加输入provenance：每条件完成清单hash、每例sidecar/p1/p2 hash、评分实现hash，不改几何/指标/通过规则。
- 新增`tools/build_shape_gt_quality_gate.py`：复用现有hash/原子JSON，重新核验源H5、完成清单、完整有序样本集、crop、metadata/NPZ；按严格candidate≤original-GT双指标重算决定，所有requested samples保留分母，输出accepted/rejected名单并明确仅nearest-vertex proxy。旧无输入绑定报告拒绝，不能追补hash伪称历史评分已绑定。
- 低成本agent补`tools/check_shape_gt_quality_gate.py`，root CPU重跑exit0：正常1/2、GT=0拒正穿透、缺行/重复/分母不符、NaN、错误decision、缺绑定、NPZ/metadata篡改均覆盖。首次fixture遗漏sidecar输出hash已补齐后通过，未修改生产检查使测试绕过。
- CPU验收清单任务session58987等待原评分队列PID4083520终态后执行，输出`output/ssp3d_confirm200_sharedground_proxy_quality_gate_20260911.json`；不提前宣称200例通过、不产生训练overlay。

### 09-11｜200例分组已冻结核对，串行评分排队

- root检查run contract：200个唯一motion均排除开发25例，前100与旧验证100名单/顺序完全相同，后100与旧100不重叠。原生成session39265/PID4083318确认活跃，不以partial manifest判断完成。
- GPU1队列session25793等待该PID终态后依次运行200例原GT配对proxy评分、共享地面对齐开发25例的完整VolumetricSMPL交叉评分；后者复用既有25例原GT volume结果，保持同seed/帧数/全顶点协议。输出分别为`output/ssp3d_fulltrain_confirm200_sharedground_iter30_gt_diagnostic_20260911.json`和`output/ssp3d_probe25_volume_sharedground30_20260911`。
- 等待期间只读检查后续H5 overlay打包入口，不提前训练或把未验收样本当正式增强；GPU0原补跑PID4076733仍活跃，未重启。

### 09-11｜共享地面对齐25例80%通过，冻结参数确认200例

- 生成session86735、配对评分session4773均exit0，25/25完成；root核验同样本，optimizer仅地面对齐策略/对应ablation标签改变，25例双方恢复offset完全相同。逐人→共享：足滑5.874→5.649mm/frame，穿透proxy0.028148→0.007443m²/frame；足滑不增均24/25，穿透不增16→20/25，双指标16→20/25=80%，冻结GT接地mask也是20/25。结果`output/ssp3d_fulltrain_probe25_sharedground_iter30_gt_diagnostic_20260911.json`。
- 共享组处理均值3.663s/最大4.235s，run阶段摊销3.709s/sample（不含模块导入/预检/离线拟合/评估）。开发集证据支持修正坐标不一致，但不是泛化验收。
- 冻结shared-ground/metric-foot/frame_sum/released-aux/lr=.001/30步，启动200例确认（selection seed20260911、offset25、limit200）；前100为此前失败验证集复核，后100是未参与调参的新motion。排除开发25例，后续分开报告旧100、新100及合计，不将全部200称全新独立集。仍4个probe donor，非全池泛化。

### 09-11｜发现优化/验收相对高度不一致，启动共享地面对齐

- 25例逐人floor偏移差绝对值均值6.803cm、最大15.786cm；独立逐人平移改变相对高度。agent诊断session62558及root严格全帧复跑session90256均exit0：同一候选优化对齐坐标穿透均值0.006792，恢复raw坐标0.028148m²/frame（约4.14倍），最大单例绝对差0.134954。root验证25例raw数值/帧数与已有独立评分逐一完全一致。
- `tools/diagnose_shape_alignment_gap.py`已加强为拒绝配对/姿态帧数不一致，禁止min长度截短；全帧结果`output/ssp3d_probe25_alignment_gap_strict_20260911.json`。这说明优化与验收坐标不一致是明确问题，但其对通过率的因果影响仍须单因素实测。
- 按Ponytail复用既有地面对齐函数，增加显式pair_shared_joint_min_restore_raw：双方减同一个floor，保持两人相对几何；旧默认逐人策略不变。CPU检查旧路径逐值一致、共享相对几何不变、restore往返、输入未修改、非法策略拒绝，exit0；runner metadata/ablation标签与策略一致。
- GPU1启动相同25例/metric-foot/frame_sum/lr=.001/30步，仅改变地面对齐；输出`output/ssp3d_fulltrain_probe25_sharedground_metricfoot_framesum_lr001_iter30_20260911`，质量待评分。暂缓候选保留机制，优先消除已证实的目标坐标差异。GPU0补跑PID4076733仍活跃，现有male24/41文件（未等同验收）。

### 09-11｜步长加倍穿透均值改善但通过率下降，不晋级

- 生成session25273、评分session80174均exit0，25/25；root核验相同样本、optimizer仅learning_rate由.001→.002（旧清单缺省aux字段按released_broadcast解释）。足滑均值5.874→5.939mm/frame，穿透proxy0.028148→0.019626m²/frame；足滑不增24→20/25、穿透不增16→15/25，双指标16→13/25，冻结GT脚mask双指标16→14/25。不以均值变好替代通过率。
- lr=.002处理均值3.664s/最大4.232s，run阶段摊销3.710s/sample；未晋级。结果`output/ssp3d_fulltrain_probe25_lr001_vs002_iter30_gt_diagnostic_20260911.json`。下一步检查末步选择是否丢失已满足双指标的候选；候选始终固定更换后的shape，不能回退原shape或放松GT阈值。

### 09-11｜aux逐帧修正25例退化，不晋级；检查有限步数的步长

- 生成session50000、同GT配对评分session79666均exit0；root核验同25例/同源hash，除声明的aux weighting外optimizer配置一致。旧权重→per_frame：足滑5.874→5.666mm/frame，穿透proxy0.028148→0.029009m²/frame；足滑不增24→22/25、穿透不增16→13/25，双指标16→11/25。冻结GT接地mask双指标16→13/25，不能用动态mask改善掩盖退化。结果`output/ssp3d_fulltrain_probe25_aux_perframe_iter30_gt_diagnostic_20260911.json`。
- 新组处理均值3.663s/最大4.218s，run阶段摊销3.708s/sample；无实际加速收益。保留可选修正及检查作为诊断，不晋级、不直接扩到100例；修正降低acc尺度后，原权重不再是相同正则强度。
- 继续使用较好的released_broadcast权重，仅将lr=.001→.002、保持30步，同25例进行步长单因素；检验短迭代位移幅度不足的可能性，结果待评。旧100例失败分布为仅足滑8、仅穿透27、两者11，主要瓶颈仍是穿透，不修改通过阈值或删失败样本。

### 09-11｜25例完整Volume交叉评分完成；隔离aux广播修正

- session5475 exit0：25例原GT/30步候选同名、同帧数、同scoring contract；Volume深度平方和均值0.181571→0.017638m²/frame，17/25不增；足滑6.855→5.874mm/frame，24/25不增，双指标17/25（近邻proxy16/25，3例判定不同）。深度总和4.225525→0.802617m/frame、穿透点条件平均深度11.618→8.265mm、最大深度35.010→28.920mm，但碰撞帧比例0.320227→0.327084。保留所有项，不宣称全面物理不退化；此开发集结果不推翻独立100例54%失败。
- root在shared aux损失中加入显式`aux_temporal_weighting=per_frame`：mixed 2D root/trans、3D pose正则分别按数据维数reshape时间权重；速度/加速度权重固定[Tdiff,1,1]，消除误广播的额外时间维。默认仍released_broadcast作历史对照；参数/ablation标签贯通runner→optimizer→loss，不涉及VQ/MGT训练代码。修正改变加速度尺度，不标成仅等价加速。
- 原foot独立CPU检查session31395 exit0，三文件py_compile通过；低成本agent补实际aux数值/梯度检查，完成后回到25例开发集单因素验证，而非直接在独立100例上挑参数。GPU0男性转换原session79556/PID4076733仍活跃，未重启。
- root强化`tools/check_aux_temporal_weighting.py`：非线性joints确保acc非零，逐帧循环独立参考公式，mixed 2D/3D参数梯度与joint梯度均核对，maskNone/T2覆盖。强化时旧参考公式漏了production分母.float()导致微小差异，修正测试参考后exit0；未放宽容差或修改生产公式迎合测试。已启动per_frame同25例/lr=.001/30步单因素，结果待验收。

### 09-11｜独立100例仅54%通过，暂不扩大正式增强

- 生成session70601 exit0，100/100；评分与单例volume队列session70915 exit0。root核验100例与调参25例不重叠，源数据hash和optimizer完全相同。处理均值3.878s、最大4.417s，run清单创建→完成摊销3.932s/sample（不含导入/预检/离线拟合/评分）。
- 原GT/随机shape/冻结30步：足滑5.986/6.255/5.162mm/frame，穿透proxy0.238394/0.865955/0.259281m²/frame。候选滑步不增81/100、穿透不增62/100、双指标54/100；冻结GT接地mask双指标55/100。独立验证未达>60%，25例64%不能外推；不筛掉失败样本、不宣称正式达标。完整结果`output/ssp3d_fulltrain_holdout100_framesum_iter30_gt_diagnostic_20260911.json`。
- 单例Volume交叉检查同sample/同contract/71帧/每人10475顶点：原GT→候选深度平方和0.00069691→0.00022012m²/frame，足滑11.845→9.637mm/frame；条件穿透点平均深度6.235→8.353mm，并非所有物理项均改善。下一步把这两条件评分扩展到全部25例，检验proxy判断一致性；单例不能代表总体。
- 女性21个离线fit由agent及root重复核验（root session92188 exit0），全部有限10beta且RMSE≤1cm，均值5.184835mm/最大6.579314mm；不等于男性补跑结束或正式donor池完成。

### 09-11｜准备原GT交叉评分，100例完成后串行验收

- 新增`tools/export_shape_gt_references.py`复用现有原子写入/参数加载函数；从已完成候选绑定的H5按精确crop导出原GT和随机换shape各25对，不改变坐标、不截短帧。session19522 exit0：源码H5/候选NPZ哈希匹配，shape确实变化，全部导出后重载与原评分参数逐字段一致；输出`output/ssp3d_probe25_gt_references_20260911`。拒绝覆盖旧参考目录。
- root确认VolumetricSMPL male/female checkpoint实际在`/home/wenning/.cache/torch/hub/checkpoints`，均存在。该模型是独立神经SDF估计器，不是真实体积GT；现有评分上限150帧，参考导出已断言所有crop≤150且配对等长。
- GPU1队列session70915等待原生成PID4075976退出后，先运行独立100例GT proxy评分，再依次跑25例中按名称排序首例的原GT/30步候选VolumetricSMPL全顶点smoke；不与生成并行干扰测速。100例原session70601仍运行（partial completion文件不等于完成），总体质量待评。GPU0补跑原session79556/PID4076733确认活跃，按低频监控不重复启动。

### 09-11｜仅恢复32个缺失SSP转换

- root复核通过：现有30个fit的betas严格(1,10)、vertices(1,10475,3)且有限；32个symlink均指向原male mesh且没有同名现有pkl。独立retry配置仅改变输入目录，拟合设置不变。结构通过不等于RMSE通过。
- GPU0补跑session79556已启动，配置`output/ssp3d_official_transfer_all62_20260911/configs/male_missing_retry1.yaml`；未覆盖现有fit、未改外部拟合代码。结束后全62组统一执行collector审核；GPU1独立100例仍为session70601。采用低成本agent准备、root核对后启动。

### 09-11｜30步SSP25达16/25；冻结参数验证独立100例

- 配对评分session94883 exit0：30步足滑5.874mm/frame、穿透proxy0.028148m²/frame，分别24/25、16/25不高于原GT，双指标16/25=64%；20步为15/25。冻结GT接地mask足速均值5.850mm/frame，24/25不增。完整结果`output/ssp3d_fulltrain_probe25_framesum_iter20_vs30_gt_diagnostic_20260911.json`。
- 冻结metric-foot/frame_sum/lr=.001/30步参数，GPU1启动session70601，输出`output/ssp3d_fulltrain_holdout100_metricfoot_framesum_lr001_iter30_20260911`。同full-plan与selection seed，offset25/limit100排除调参25例；这是独立motion验证，仍使用4个probe donor，并非全donor泛化或真实体积验收。不得删掉失败样本或在结果后改分母。

### 09-11｜30步完成待评分；62组转换异常，保留已有文件补跑

- 上轮属实质进展。本轮30步session11140 exit0，25/25完成：处理均值3.624s、最大4.182s，run清单创建→完成摊销3.668s/sample；20/30步同GT配对评分session94883已启动，尚无质量结论。
- 62组离线转换原session73615确切exit132，PID4066459已不存在；female21/21、male9/41有fit文件，32组缺失。无错误堆栈，内核日志无读取权限，不能断言退出原因。低成本agent准备校验现有文件、独立缺失项配置，仅补跑未完成male，不覆盖旧结果；全池未验收。

### 09-11｜穿透归一化使SSP25双proxy通过率44%→60%

- 生成session32008、配对评分session44927均exit0；25/25确实更换shape，同source/crop/donor并核验NPZ哈希、有限指标。原GT/metricfoot20/framesum20：足滑均值6.855/6.323/6.315mm/frame，穿透平方和均值0.293803/0.379953/0.041903m²/frame；双指标不增11→15/25，冻结GT脚接地mask复核也为15/25。
- 20步新组处理均值2.916s、最大3.276s；run清单创建→完成74.115s/25=2.965s/sample（含运行阶段读写，不含导入/预检/离线拟合/评分）。结果`output/ssp3d_fulltrain_probe25_framesum_lr001_iter20_gt_diagnostic_20260911.json`。60%只是本25例近邻proxy，不视为正式质量达标。
- 保持源文件不变，启动相同25对30步检查（session11140，输出`output/ssp3d_fulltrain_probe25_metricfoot_framesum_lr001_iter30_20260911`），随后需独立更大样本以避免在这25例上调参后夸大泛化。低成本agent另发现aux时间权重多一次unsqueeze导致velocity/acc广播成[L,L,J,3]，root核对源码一致；暂未改动，避免混入本次单因素。

### 09-11｜穿透reduction单因素试验

- 上轮为实质进展（44%配对结果已验收为诊断）。新增显式`penetration_loss_reduction=frame_sum`：复用原距离/符号/阈值计算，仅将按穿透点平均改为总量除以帧数；默认仍`penetrating_vertex_mean`。0.2/0.5m优化阈值保留，故还不是与评测10m阈值完全一致的loss。
- `tools/check_penetration_reduction.py` session82912 exit0：原数值不变，frame sum值和梯度正确，重复帧不变、重复穿透点总量增加、无碰撞为零。参数及实际ablation标识已贯通runner→optimizer→loss，并写入run contract/sidecar。
- 启动相同25对/4 donor/metric-foot/lr=.001/20步，唯一实验变量为穿透reduction；输出`output/ssp3d_fulltrain_probe25_metricfoot_framesum_lr001_iter20_20260911`。不减少顶点、帧或筛除失败样本；目标仍>60%且原GT双指标不增。

### 09-11｜防滑目标对齐使SSP25通过率4%→44%，瓶颈转为穿透

- session84587生成、session91268四条件评分均exit0：同25样本/4个SSP donor/lr=.001/20步，新foot policy双proxy通过11/25；动态足滑不增21/25，冻结GT接地mask足速亦21/25不增，penetration proxy不增12/25。旧20步对应1/25、1/25、1/25、14/25。
- GT/旧20/新foot：水平足滑均值6.855/7.805/6.323mm/frame；穿透proxy均值0.293803/0.375153/0.379953m²/frame。足滑改善明显，但穿透总体仍高于GT，不能晋级正式增强。结果`output/ssp3d_fulltrain_probe25_metricfoot_lr001_iter20_gt_diagnostic_20260911.json`。
- 计时口径纠正：既有sample_processing计时不含读取/落盘。旧20组处理3.154s，但run清单创建→完成平均5.228s/sample；新foot组处理3.282s、同口径墙钟3.373s/sample。不能将此前“全部处理低于5s”扩大成旧组端到端达标。新组首个sidecar距run清单创建5.423s，后续24个sidecar间隔最大4.302s（含I/O的近似诊断，未代替独立计时器）。
- 下一步拟检验penetration归一化：优化当前按穿透点数量平均，验收按逐帧穿透深度平方求和，二者不等价；保持已有效的foot目标，单独比较穿透reduction，不靠筛掉失败样本提高25对通过率。

### 09-11｜SSP25旧防滑目标失败，启动目标对齐单因素

- session1160 exit0：GT平均水平足滑6.855mm/frame；随机换shape7.043，优化20步7.805。随机双proxy通过2/25，优化仅1/25；优化足滑不增1/25、穿透不增14/25。完整数据`output/ssp3d_fulltrain_probe25_lr001_iter20_gt_diagnostic_20260911.json`，速度过而质量失败，未扩大全量增强。
- 低成本agent及root核对：旧skate目标是速度阈值筛过的3D速度平方，新评测是所有接地脚水平速度；旧快滑可被排除且空mask可能NaN。复用既有metric_aligned_foot_sliding模块新增显式可选policy（默认仍released），对齐floor、水平速度，冻结初始接地mask与当前mask取并集；不移动penetration几何、不改其余loss或权重。新目标单位由速度平方改为水平速度，数值尺度也改变，不能称纯mask替换。
- policy已贯通runner→optimizer→loss，独立ablation metadata/hash避免误标原MMM。正在执行CPU小检查后再跑相同25样本/4 donor/lr=.001/20步对照。按Ponytail复用已有损失与入口，不新建优化框架。
- CPU小检查session34793 exit0：快滑有梯度、整体y平移不变、初始mask防抬脚规避、空mask/T=1安全、其他四项aux loss逐值不变。首版测试漏传生产路径必有的temporal_mask导致旧分支广播错误，补齐fixture后通过；未改旧分支公式来让测试过。
- GPU1对齐目标25对试验session84587已启动，输出`output/ssp3d_fulltrain_probe25_metricfoot_lr001_iter20_20260911`；与旧20步使用完全相同full-plan/selection seed/shape/步长/步数，仅foot policy不同。终态后仍须逐样本双指标评分，不能以小检查代替实际验收。
- 62组拟合低频复核：GPU0 session73615对应female进程仍活跃，已有17/21女性fit文件、male尚未开始，不代表已通过RMSE审核。

### 09-11｜SSP25短迭代生成完成，完整sample全部低于5s

- session38830 exit0，25/25完成；optimizer均值3.072s，sample均值3.154s/中位3.281s/最大3.763s。包含每sample模型初始化、地面对齐与优化调用的处理计时；不包含整次运行的plan校验/模块导入、离线donor拟合或评分，不夸大为全流程墙钟。
- 已启动原GT/随机换shape/20步优化的25对评分，需通过率结果才能判断候选；目前只确认这组短迭代的sample处理时间达到目标。采用已过转换审计的4个probe donor，不代表62组全池。

### 09-11｜SSP pilot启动前split绑定检查拦截

- 修正后的GPU1 pilot句柄session38830，输出`output/ssp3d_fulltrain_probe25_lr001_iter20_20260911`；25条从完整计划正timing集合固定抽取，区别于此前子split的25条，不混用两份样本清单。下一步终态后用原GT诊断脚本测双指标及完整sample计时。
- session53866已回收exit1，未开始优化：25条子split与既有contact cache绑定的完整train split不同。保留严格缓存绑定检查，不绕过；session2042 exit0冻结完整train的SSP计划`ssp3d_probe_fulltrain_20260911`（6238记录/5868正timing），runner以selection seed20260911抽25条。原25条计划仅保留作validator正/反例测试，不宣称已运行增强。

### 09-11｜SSP25短步长增强pilot启动

- 旧plan兼容审计session73735 exit0：6238记录/118原donor未被SSP扩展破坏。实际120帧×10475点、正确CUDA设备同步计时KNN各版本16.3–34.5ms，自动版17.2ms；无数量级收益，未强制切内核。第一次events误记默认GPU0的微秒计时已识别废弃，不作证据。
- 观察旧lr=.01前几步acceleration loss急升，GPU1 session53866启动固定SSP25、lr=.001、20迭代pilot，完整输出`output/ssp3d_probe25_lr001_iter20_20260911`。这是有意检验更小步长的速度/质量折中，未进入100步anchor阶段；必须做原GT双指标核验，不能用少迭代直接宣称5s/60%。62组离线拟合仍在GPU0，暂不轮询。

### 09-11｜SSP 10维probe全通过，62组拟合启动

- 上轮为实质进展。session63047 exit0，root session97756 exit0核验4个10维完整拟合：女性RMSE5.192/4.972mm，男性8.981/5.288mm，全部≤10mm。`probe10_audit.json`绑定实际fit/mesh/入口hash；维数确为(1,10)，不是先前16维结果。
- session70468 exit0准备62组（女21/男41）；GPU0 session73615运行官方全池拟合，不动暂停训练。该阶段为离线donor转换，不计入后续每sample优化耗时；不得宣称62组已验收。
- `collect_ssp3d_smplx_donors.py` session10909 exit0产出4组已过转换门槛的probe donor池，明确SSP来源、source IDs及fit/hash；未伪标HumanML3D/AMASS。计划生成器/validator新增SSP分支，强制同gender，保留旧计划语义。
- 固定seed20260911从旧timing-positive train记录无放回选25对，冻结新计划`configs/interx_shape_augmentation/ssp3d_probe25_20260911`；25/25有手工timing。session91132 exit0：25对同gender全通过，篡改为异gender donor被拒绝；尚未运行SSP augmentation。
- 原Chamfer200步重复session17596 exit0，sample均值22.582s；singleKNN旧组16.606s（均无profile/TQDM disabled，但CPU并发不同，速度比例仅诊断）。物理对照session43898 exit0：第三例原实现proxy从.01306425波动至.01481936（约13.4%），加速版.01388663落在其中；不能将此前差异全部归因加速，也不凭两次重复宣称统计等价。三臂proxy通过均1/3。

### 09-11｜原始GT双指标诊断完成，SSP拟合接口修复

- `tools/diagnose_shape_optimization_gt.py` session82690 exit0：核对源H5/hash、exact crop、输出betas/gender及文件hash；原GT、换shape未优化、reference200、singleKNN200四条件评分。参考与加速版均仅1/3满足相对原GT的动态足滑和nearest-vertex穿透proxy不增，不能当60%或真实mesh通过率。另报告冻结GT接地mask的水平足速，防止接地mask变少伪改善。
- 三样本reference→singleKNN足滑(m/frame)：.00265838→.00265848、.00601348→.00601346、.00752487→.00753271；penetration proxy(m²/frame)：.00030156→.00030193、.10187286→.10187708、.01306425→.01388663。第三例穿透proxy约+6.3%，当前只称候选加速，需复跑原实现估计轨迹波动及扩大物理核验。
- 官方transfer直接入口session6902 exit1（namedtuple输出与字典接口冲突）；独立兼容入口session79025 exit0，但审计发现nested beta count被底层默认16覆盖，输出不可作为HHC10维donor。首例4.846mm RMSE属于16维诊断，不是10维验收。
- 已修复独立入口输出容器及显式num_betas绑定/断言（不修改官方checkout、不动损失/优化器），session63047重新启动男女各2例10维probe，输出独立`fitted10/{female,male}`，保留16维旧产物；当前任务仍第②项，未开始增强训练或InterHuman。

### 09-11｜第②项：完整200迭代热点与SSP转换实测

- 上轮为实质进展。本轮GPU1三样本200迭代profile session96559 exit0：sample均值22.861s；600步中penetration累计45.28s，是主要热点。不是5s达标。
- `point2point_signed`原Chamfer内部计算双向KNN却只取单向索引；改为同库精确单向KNN，不减少顶点/帧/损失。`tools/check_signed_distance_knn_parity.py` session8343 exit0：小点云/整SMPL-X顶点、重合/重复点、signed/unsigned六组值和索引精确相同，梯度通过容差检查。
- 修改后相同3样本200步session61541 exit0，均值16.606s（baseline带profile且进度条设置不同，不能作严格加速比例）。长轨迹参数最大差0.1163，尚不能称完整优化等价；需与原实现重复运行的数值波动及物理指标对照，不能只凭单步测试宣称质量保持。
- `tools/diagnose_ssp3d_shape_transfer.py` session36863、5482均exit0：62组位移RMSE均值7.805mm，37组≤10mm；绝对模板+shape+平移拟合均值13.578mm，0组≤10mm。全为diagnostic，未晋级donor。已复用官方SMPL-X transfer fitter及ProsePose matrix准备男女各2个probe，female session6902运行中；不改阈值掩盖残差。
- 低成本agent确认旧shape gates均无原始GT逐样本双指标门槛；旧PeneCalculator为nearest-vertex proxy且默认截前150帧，不能直接当完整mesh穿透验收。后续需要准确crop原GT与同坐标的增强输出配对评分。

### 09-11｜重复解压修复三样本复跑完成

- session2510 exit0：相同3样本×5迭代，完整optimizer均值5.073→1.944s，样本处理5.187→2.046s（约60.6%降低；旧组带cProfile，后续需无profile配对计时）。输出各数值参数最大绝对差2.384e-6，GPU优化非bitwise；模型级CPU输出/梯度检查精确相同。
- 此结果仅证明短迭代性能收益，不等于完整optimization已低于5s，更没有验收60%通过率。下一步测完整迭代计算热点，并验证SSP shape位移转换误差，再按原始shape GT对比滑步/穿透。

### 09-11｜第②项：定位并修复模型重复解压

- 三样本profile完整产物已生成：`output/shape_speed_baseline_20260911_multi3.prof`；原句柄76334已不可用，未重启。5迭代样本处理均值5.187s，loss构造均值4.344s，主要为NPZ解压。不是正式优化通过率。
- `BodyModel` 原来对shapedirs/posedirs反复索引NPZ，每次重新解压；现仅在单次构造内各读取一次并关闭文件，不共享可变loss状态，不改变loss/距离/穿透计算。按Ponytail仅修实际瓶颈，未添加缓存框架。
- `tools/check_body_model_npz_loading.py` session34369 exit0：三gender全部buffer、顶点、关节及beta梯度与git HEAD原实现精确相同；模型构造2.235/2.308/2.019s降到0.669/0.632/0.423s。已启动相同三样本复跑 `output/shape_speed_npz_once_20260911_multi3`，待核对端到端时间和优化输出。
- 低成本agent核实SSP-3D有311条、62组独立gender+SMPL10beta（男41/女21）。SMPL及SMPL-X模型、ProsePose全顶点对应matrix齐全；其转换代码拟合shape位移，并非绝对模板，下一步区分模板差异与shape转移残差。仍无通过转换验收的外部donor池。

### 09-11 19:23｜优化速度基线首样本完成

- session `53749` exit0，旧plan固定样本 `G005T003A016R004__shapeaug_s00000123`，nearest_vertex仅5迭代：循环日志0.759s，但同步测得完整optimizer调用5.3347s、sample处理5.5134s。不能用循环耗时冒充低于5s/sample，也不是正式质量/通过率结果。
- 现行loss每sample构造三种gender BodyModel；首样本表明初始化/循环外成本值得先拆分测量，下一步测多样本warm运行并profile geometry/nearest/penetration/backward。公开外部donor仍待转换验收。

### 09-11｜第②项启动：资产盘点与优化速度基线

- 主agent重新核对现行 `run_interx_shape_augmentation.py` → `optimize_full_body` → `DyadicInteractionLoss`。旧fast selective gate以random_shape_mismatch为基准，不能沿用为用户要求的60%通过率；旧smoke还有过时参数，未直接运行。
- GPU1启动单样本5迭代nearest_vertex速度基线，输出 `output/shape_speed_baseline_20260911_probe5`；复用旧Inter-X donor plan仅做性能诊断，不作为公开外部shape增强的最终结果。尚无速度结论。
- 低成本agent `shape_asset_inventory` 只读盘点合法公开donor与SMPL→SMPL-X转换资产；现有AMASS contract记录原档缺失，SSP-3D已有下载但转换未验收。未直接复制不同模型空间的betas，不做全量增强。

### 09-11｜核对Interact2Ar论文非AR参考

- 直接核对 https://arxiv.org/html/2512.19692v1#S4.T2 ：Table2星号Interact2Ar*明确为非自回归版本，不是关闭GT首帧的AR模型。NAR Full/Body/Hands FID=`.556/5.728/.906`，MM=`2.743/3.967/3.206`；R@1=`.436/.426/.382`，Matching=`3.246/4.190/3.563`。
- 与无shape300k原采样test均值比较：HHC各部分FID/Matching/R@1/2/3数值更优，MM远低；最新加噪仍仅validation，不能宣称同test击败NAR。论文还列PJ/AUJ，HHC尚无对应时序拼接实验，不能以足滑替代。仅报告，未启动新实验。

### 09-11｜无shape与Interact2Ar指标对照核对

- 用户仅要求报告，不启动新评测。root确认无shape300k原采样test20×1696 joint summary与 `output/interact2ar_joint_evaluator_suite/pre_shape_summary.json` 的 evaluator_provenance完全相同；后者确有Interact2Ar三部分joint，不能误称只有672D指标。
- 最新排序加噪仍是validation544 / MM100×11，不能与test20rep直接判输赢；未找到Interact2Ar相同validation544生成评分，GT self-score不能充当方法结果。
- 同协议test原采样HHC vs Interact2Ar无GT首帧：joint Full FID `.098106 vs .702577`，MM `.057011 vs 2.968545`。论文参考与无GT首帧复跑须分列；旧test physical仅rep00且v1，不能混作当前validation valid_frames v2。本轮未恢复研究目标或启动实验。

### 09-11｜有/无shape采样验证全部完成

- 有shape加噪MM F/B/H=`.958955/1.168498/1.330081`，原=`.031193/.037229/.047619`；重复率65.44%→7.11%。root `8ae18b` exit0核验实际H5/JSON SHA、100×11完整finite、两臂names/lengths/评测协议/ckpt/shape一致；pipeline `55740` / monitor `1743` 正常exit0。
- 有shapeFID F/B/H=`.241860/.214220/.403981`→`.244934/.194259/.390061`；Full R@1下降3.31pp、滑步增加1.58%、接触F1下降0.65pp。无shape结果亦已完整核验。两类均有大幅MM增益，但不是严格质量全不降；保留为折中方案。
- 完整数据及限制写入 `docs/shape_no_shape_sampling_validation_20260911.md`；按用户要求停止继续探索，不启动下一组实验。

### 09-11｜有shape加噪544主质量完成，MM继续

- root `8c3578` exit0：实际H5/JSON/physical SHA绑定、完整finite、两臂names/lengths、共同ckpt/CPU协议均通过。FID F/B/H 原`.241860/.214220/.403981`→加噪`.244934/.194259/.390061`；滑步`.290324→.294906`，接触F1`.588763→.582243`。完整Matching/R@K/Diversity已写数据表，Full R@1下降3.31个百分点，不只报FID。
- monitor `1743` 最新加噪MM repeat4已到96/100；原session `55740` 仍运行。只剩该臂MM及最终配对验收，不追加实验。

### 09-11｜有shape原采样MM完成，加噪生成推进

- 原采样100×11 MM Full/Body/Hands=`.031193/.037229/.047619`，重复3599/5500（65.4364%），平均2.71种动作/prompt。root audit `7133c7` exit0：实际H5/JSON SHA绑定、完整finite、shape/ckpt/requested beta/seed核验；完整数据表已更新。
- monitor `1743` 确认 control 臂完成；原session `55740` 已进入 confidence_all，加噪主生成416/544。继续原任务，不重复启动，尚无加噪评分。

### 09-11｜有shape原采样544主质量完成

- root audit `6276a4` exit0：H5实际SHA/JSON/physical绑定、shape=true、ckpt SHA、544×1完整finite、CPU joint共同协议通过。FID Full/Body/Hands=`.241860/.214220/.403981`，滑步=`.290324` m/s，接触F1=`.588763`；完整Matching/R@K/Diversity已补入 `docs/shape_no_shape_sampling_validation_20260911.md`。
- 原session `55740`/monitor `1743` 继续原采样100×11 MM，最新repeat1开始；加噪臂尚待运行，不作采样收益结论。

### 09-11 17:10｜有/无 shape 采样验证收束

- 无shape既有两臂已由root重新核验：4个H5实际SHA/JSON绑定、完成标志、finite、physical绑定和joint协议一致，session `95010` exit0。CPU口径 Full/Body/Hands FID：原 `.240967/.234382/.210853` → 加噪 `.203157/.209598/.265925`；MM `.043909/.047873/.083011` → `1.160418/1.228123/1.814558`。
- 有shape session `55740` 已 exact load/requested VQ preserved、生成32/544；monitor `1743` 每600秒独占回收。仍待两臂完整结果，不重复启动。
- 完整数据与状态见 `docs/shape_no_shape_sampling_validation_20260911.md`。本次 confidence_all 是 probability ranking + noise1，不能误标为 log_odds。

### 09-11｜weight-decay pilot 10k checkpoint 固定，冻结评测已启动

- PID `4025534` 仍运行（native 10k eval 当时7/17）；10k ready 后仅终止 timer `1724`，训练 session `79369` 继续。固定 hardlink=`output/hhc_weight_decay0p01_matched_validation_20260911/step_000010000.ckpt`；root audit `25237` exit0（chunk `f3e63c`）：global_step=`10000`、epoch=`74`、optimizer step=`10000`、wd=`.01`、model floats finite，SHA=`1dc7671d6bc7e52d774fc9268a8c04960326d81c4834bd35ac2d83a598873d0e`。
- GPU0 544 pipeline `48301` 已启动，monitor `1729` 每600秒独占回收；启动 exact model load/requested VQ preserved，已生成 `32/544`，输出同目录 `step_000010000_main.h5/_cpu.json/_physical_per_sample.json`。评分未完成，不追加 MM；monitor `1730` 每600秒只读等待 training latest20k 或 PID `4025534` 消失，session `79369` 保持。

### 09-11｜weight-decay pilot 5k checkpoint 验收

- root audit `92776` exit0（chunk `7b2329`）：`latest-000005000.ckpt` SHA=`2f1365eeed5bdddc94545e44276aa23cc70955568c0f1a17873f59db3c0b4446`，global_step=`5000`、epoch=`37`；AdamW 单 group wd=`.01`、lr=`.0002`、betas=`(.5,.9)`，447 params/309 state，optimizer step 全为5000，模型 float/optimizer finite，args loss=`historical_all_valid`、stop=`50k`一致。
- 主 PID `4025534` 仍运行（约5718 steps，epoch42 batch90），非停滞；已恢复每5000 checkpoint 间隔。旧 timer `1717` 仅在节点后终止，不结束训练；monitor `1724` 每600秒等待 10k checkpoint 或 PID 消失。尚无10k质量结果。

### 09-11｜weight-decay pilot 启动核验通过

- run=`output/t2i_trans/2026-09-11-16-20-27_hhc_no_shape_weight_decay0p01_pilot50k_seed114514_v1`，W&B `8v983xtw`，session `79369` 已到 epoch0 batch45（约4.8 it/s）。root audit `4b08ef` exit0：相对 control training_manifest_at_50k，resolved args 仅 wd `1e-6→.01`、GPU0→GPU1、run/output/W&B 不同；cache/dataset/loader manifests 逐值一致，其余 source-derived 配置保持。
- monitor cell `1717` 每600秒独占 `79369`，等待任意 latest checkpoint 做 optimizer state 验收或训练终态；启动回收 cell `1713` 已完成但训练 session 仍 live。optimizer 尚未由 checkpoint 核验，不重复监控。

### 09-11｜historical38 window probe 完成，weight-decay pilot 已启动

- pipeline `79729`/cell `1708` exit0；audit `3edb77` exit0，actual checkpoint/manifest/split/script SHA、旧 sample hash、window38 payload37 均核验。fully masked train 30k→300k NLL B/H/R=`2.71286/2.98286/3.86607→.17197/.18340/.20020`，accuracy=`45.672/41.89461/31.20189→95.65423/95.28715/95.02664%`；validation NLL=`10.66111/11.66873/11.06442→13.30775/14.67705/14.36496`，accuracy=`1.77903/.38624/.88951→2.04822/.74906/1.49813%`，maxprob=`.38313/.33585/.31138→.61079/.57392/.59061`。窗口不能解释泛化差距，仍非单一根因证明。
- weight-decay 单因素 50k pilot 原 session `79369` 已在 GPU1 启动；使用现有 launcher，目标 wd=`0.01`（control `1e-6`），其余 VQ/cache/结构/mask/dropout/seed114514/300k schedule、50k stop、10k eval保持。manifest/optimizer 尚待实际核验，不称配置已验收，不另开 monitor。

### 09-11｜historical38 token-window probe 已启动

- `audit_mgt_mask_exposure_gap.py` 新增显式 `--token-window`，默认仍为 model block_size-1；实际 window/max_payload/script SHA 写入输出，`make_batch` 增加窗口校验。既有 `test_mgt_logit_sharpness` 新增38/50 prefix/end/padding/非法窗口测试，共5项 session `77719` exit0。
- GPU1 pipeline `79729` 已启动：validation256→training256、v4 30k/300k、seed90123、batch16、Tsharp=1、CFG3.5、paper_text_cfg、NLL grid `.5/1/1.5/2`，唯一新因子 `--token-window38`；输出 `output/hhc_mgt30k300k_calibration_validation256_window38_20260911.json` 及 training 同名。仍为纯 mask/GT-context probe，非完整训练 corruption；无训练改动、无质量结论。

### 09-11｜calibration code-support 缺口核查

- root CPU audit `89328` exit0（chunk `71238a`）：按历史训练前37 payload 计数，train8617 各287161/part、active codes B/H/R=`9749/7121/6884`；calibration val256 实际9808 tokens，train 未见仅 B/H/R=`59/8/23`（`.602/.082/.235%`），historical38 whole val534 未见=`113/14/29`。因此 code support 不能解释 validation <2% accuracy。
- train/val names 无交集；val256 首次 caption 精确匹配 train 任意 caption `22/256`。training samplehash=`c79d5cc3e1405d9d4aa7e1cc0df9faec3657971fa5bd353cab3c4346ce4ebc2c`、val=`c2c0f24ca2db312ff9c0bedeb87d4fe484900d344c2120488fa5a300047a0fcd` 与正确 calibration JSON 一致；旧 val32 比较说法作废。
- 现有 audit `make_batch` 按 block_size-1 截断49 payload，和 historical38 的37 payload不同；结果有效但非完全同窗口。下一步先做显式38-window probe，不启动新训练。

### 09-11｜MGT mask-exposure calibration 完成

- pipeline `54586` exit0；两 JSON 齐全、进程退出，root 仅回收 timer `1695`。audit `7f32b8` exit0，split/manifest/checkpoint actual SHA 已核验。
- fully masked/noCFG/T1 的 training 30k→300k：NLL B/H/R=`3.6174/3.9641/4.6721→1.3662/1.4396/1.3536`，accuracy=`40.3213/36.1717/27.6050%→84.2875/83.3299/83.6594%`；validation NLL=`10.6222/11.5429/11.0630→12.7670/13.9538/13.8460`，accuracy=`1.6517/.3874/.8666→2.0086/.7239/1.4682%`，mean maxprob=`.3525/.3073/.2894→.5574/.5202/.5448`。teacher-GT cascade temperature grid 亦显示 train/val gap。
- 结果支持晚期置信增长伴随强 train/val 泛化差距，但不等于纯低熵=坏；尚未排除 code support、文本或 subject 分布混杂。下一步只读排查全 train code frequency、val unseen-code ratio、首次文本 train match，并复核 val256 sample hash；不启动新训练。

### 09-11｜MGT mask-exposure calibration 诊断已启动（尚无结果）

- 已核对 masked-only comparison 与 train loss 分支：50k 质量失败，10k 恢复缺 RNG 证据，不能作完全 clean 因果。复用 `tools/audit_mgt_mask_exposure_gap.py`，pipeline `54586`（GPU1）对 v4 MGT 30k (`model-000030000-0.6630.ckpt`) 与300k，先 validation256 再 training256，seed90123、batch16、paper_text_cfg、CFG3.5、sharpness_temperature=1，NLL T grid=`.5/1/1.5/2`。
- validation256 JSON 已完成；两个 checkpoint exact load 且 VQ preserved。training256 仍运行，由 cell `1695` 每600秒独占回收，尚无完整配对诊断结论。
- 初读 validation 全 mask、未 CFG NLL：Body/Hand/Root 30k→300k=`10.6222/11.5429/11.0630→12.7670/13.9538/13.8460`，置信变高但 NLL 变差；待 training 对照，暂不归因过拟合，也不把绝对 token 低 accuracy 当动作差（存在多模态标签与近似 codes）。
- 限制：`paper_scheduled_mask` 是纯 mask、非历史 random replacement；`body_after_gt_root/hand_after_gt_root_body` 是 GT-context probe、非实际生成；主比较同 split/同 samples/同 mask 的30k vs300k，train/val 不逐样本配对；NLL 最佳温度不等于生成质量最优。

### 09-11｜early-noise10 full544 评分完成（未晋级）

- pipeline `17976` exit0；仅在评分文件齐全后终止 timer `1680` 并回收 pipeline，未终止训练任务。audit `31302` exit0：H5 SHA=`cb867e7986bea7106afa188b0e8333c9b458f8381a3ded9aeadd11f3c35f2202`，544 finite/complete、names/lengths、checkpoint/VQ/sampler/REQ/GT hashes、noise1 cutoff10 metadata、eval/GT/length policy/seed/physical protocol 均一致；双方 JSON 均缺 runtime，不声称该字段一致。
- baseline→candidate FID Full/Body/Hands=`.240967/.234382/.210853→.270059/.208631/.291210`；slide=`.303470→.306257 m/s`，F1=`.557580→.567495`。paired bootstrap slide delta `.002787` CI `[-.003356,.008976]`、F1 delta `.009915` CI `[-.009795,.029818]` 均跨0；Full/Hands FID 分别约+12.1%/+38.1%，未晋级、无正式 MM、不追加 noise 网格，返回训练侧诊断。

### 09-11｜early-noise10 trace 完成，full544 质量评测已启动

- trace session `77111`/cell `1675` exit0，audit `89750e` exit0；provenance/artifact SHA、names/lengths/finite、counts、轨迹衔接/final IDs 均核验，control NPZ 与上轮 maxprob control bitwise 一致。motion duplicate `42→12/48`；Body/Hands/Root token duplicate `43/42/43→16/19/16`；各1000 accepted、31508 proposals，nonargmax accepted `2/1/5→6/2/4`，proposals `798/939/822→888/938/810`。
- 结果支持 commitment-order 机制存在多样性信号，但不是正式 MM、无质量证明。full544 pipeline session `17976` 已启动（GPU1 export→CPU joint32→physical valid_frames），monitor cell `1680` 每600秒独占回收；启动 exact model load/requested VQ preserved，已生成 `32/544`。固定 noise1 cutoff10、其余50/T=.5/CFG3.5、seed500123、no-shape/no-contact/requested betas/save IDs；尚未评分。
- baseline control H5 SHA=`3827dacb07d81c37e9a67cb7269bca0671b083ade66459dce40535e6ad504e0a` 与 CPU/physical 文件绑定核验通过；GT diversity、seed/batch/steps/T/CFG、no-shape/contact、ckpt/VQ 一致，但 baseline JSON 缺 `evaluator_runtime`，不称该字段完全一致。

### 09-11｜early ranking-noise cutoff trace 已启动（尚无结果）

- 新增 `confidence_noise_until_step` 规则：默认0保持原 full schedule；本实验 cutoff=`10`，仅 step≤10 使用原50步线性幅度、不重缩放，其余采样不变。14项 unittest session `9011` exit0，覆盖旧 golden token/RNG、full endpoint、noise=0、2-step 恰好6次 noise draw、前2步轨迹、invalid/resume 防混；MM scorer self-check exit0。
- trace session `77111`（GPU1）已启动，输出 `output/hhc_early_noise10_trace_val8x4_20260911`；原 v4 300k+VQ175k、8 length quantiles×4 repeats、seed700123、50/T=.5/CFG3.5，arms control/log_probability_noise1、cutoff10（仅 control 无噪声）。尚无结果；默认行为不变，生成停止后才考虑共享 source 变更。

### 09-11｜max_probability 8×4 trace 诊断完成（无改善信号）

- trace session `56635` exit0；audit `63a13b` exit0，REQ/checkpoint/VQ/script/sampler SHA、8 artifact SHA、names/lengths/finite、轨迹衔接/final IDs/stored counts 全部核验。8×4 两臂 motion duplicate 均42/48；Body/Hands/Root token duplicate control=`43/42/43`→maxprob=`45/42/42`。
- 每 part 1000 accepted；nonargmax accepted control=`2/1/5`→candidate=`1/4/6`，nonargmax proposals=`798/939/822`→`796/940/823`（各31508 proposals）。8×4 无多样性改善信号，不追加544质量/MM，不宣称全域无效。
- 结果提示仅解除 drawn-token 概率惩罚不足，高置信位置本身仍尖锐主导。下一假设尚未实施：仅早期确定位置引入排序随机性，后期回归原排序；第1项仍未达质量不降目标。

### 09-11｜max_probability trace 诊断已启动（尚无结果）

- 12项重跑 session `72977` exit0，新增 max_probability observer token/RNG parity；MM scorer `--self-check` exit0。
- trace session `56635` 已启动，GPU1，输出 `output/hhc_max_probability_trace_val8x4_20260911`；monitor cell `1664` 每600秒独占回收。模型 exact load、requested VQ preserved 已确认，control repeat0/1 各150 stages 完成；全流程及评分尚未完成。

### 09-11｜max_probability sampler 契约与回归测试完成

- max_probability sampler/exporter/MM metadata 已实现，默认行为保持不变；session `96986` 三组测试共12项 exit0，覆盖旧默认 golden token/RNG、合成两位置次优候选保留、padding、reproducibility 与 noise 组合。
- 正在扩展已有 trace 工具，准备 control/max_probability 两臂 8×4 实模型诊断；尚未启动，不作质量或 MM 结论。

### 09-11｜span v2 训练 clean stop，native 50k 验证收尾

- session `28378` 已回收 exit0，step50000 validation 后 clean stop。native 50k Full FID=`.5167`、div=`9.0198`、R=`.42830882/.62316176/.71875`、matching=`3.769199609756`；最终100-subset MM=`.1848`、subset FID=`.86011`、div=`9.23729`、R=`.48/.63/.74`、matching=`3.484842538834`。native best 仍30k Full FID=`.41091`；冻结50k失败结论不变，不延长训练。
- 主 agent 正在实施 max-probability 单因素排序，尚未通过测试或生成结果。

### 09-11｜max-probability 排序假设登记（尚未实施）

- 代码梳理确认当前 `confidence=softmax(channel_logits).gather(predictions)`，非 argmax proposal 会被排序额外惩罚；现有 noise 线性退火到0，尚未测 candidate-independent confidence。
- 拟新增单因素 max-probability 排序：位置置信度取该位置 max probability，不依赖抽到哪码；保留 Gumbel/温度/CFG/steps/RNG draws/mask counts 不变。先 baseline parity+synthetic selection self-check，再小规模生成与完整质量决定是否 MM；当前未验证提升、未实施。待 monitor `1647` 终态后才改共享 sampler/exporter，保留50k失败结论。

### 09-11｜span v2 50k 冻结评分完成（未晋级）

- pipeline `59684`/monitor `1646` exit0；audit `76606` exit0。candidate H5 SHA=`d2faba837792778f7ca6b299c17304701fa674e444b1c665f45c434100ac7dc0`，control H5 SHA=`2616217ccea4d4ae97d284492cf5ac4697bfb5f16269ef1101fdee2c5e3fd0cd`，control checkpoint SHA=`0e2308c493124ad87f69528606175c30dcd1cc0324031db8ec5c0c778724e055`；544 complete/finite、REQ/eval/GT/physical 一致。old control 缺 beta_policy/save_token_ids 诊断 attrs，其余采样 attrs 一致，不能称全部 attrs 相等。
- control→candidate FID Full/Body/Hands=`.366109/.370827/.438952→.364234/.219427/.450787`；slide=`.326886→.352584 m/s`、F1=`.519287→.533215`。bootstrap slide delta `+.02569763` CI `[+.01333403,+.03769201]`，F1 delta `+.01392776` CI `[-.01806294,+.04586271]`；滑步明显退化、Hands FID 变差，未晋级、不追加 MM、不延长 span 训练。
- native 50k Full 17/17 FID=`.5167`、div=`9.0198`、R=`.42830882/.62316176/.71875`、matching=`3.769199609756`；30k `.41091` 仍 best，MM `0` 是中途 placeholder、真实子循环1/4。后续回到 MGT 预测尖锐/采样机制定向诊断，不盲扩 span。

### 09-11｜span v2 50k checkpoint 固定，冻结评测已启动

- monitor `1613` exit0 返回 `STEP_50000_CHECKPOINT_READY`；root session `7502` exit0，固定 hardlink=`output/hhc_circular_span_matched_validation_20260911/step_000050000.ckpt`，global_step=`50000`、epoch=`373`，hyper_parameters.args 相对40k逐值无差异（含 runtime span SHA），checkpoint SHA=`398326d8f914381e1acfac6eb28a0ca6621f0e1620961611397936a45d21323f`。
- pipeline `59684` 已启动，monitor `1646` 每600秒独占回收；544 export→CPU joint32→physical valid_frames/per-sample，固定原 VQ175k、seed500123、50/T=.5/CFG3.5、requested betas、no-shape/no-contact/noise/replay/token IDs。尚无50k结果，不追加 MM；native 50k eval 当时仅5/17。monitor `1647` 等 PID 消失，训练未宣称正常结束。

### 09-11｜span v2 native 40k 验证收尾

- native 40k MM 真实完成=`.1852`，100-subset FID=`1.15428`、div=`8.94210`、R=`.43/.69/.78`、matching=`3.4038772583`；此前 `0` 仅为中途 placeholder。Full native 40k FID=`.6075`，30k `.41091` 仍 best；subset 指标不等于 frozen MM。
- PID `3868579` 已恢复至 epoch337 batch25；monitor `1613` 继续每600秒等待50k，不能称50k已完成。

### 09-11｜原 VQ175k action-group distance retention 诊断完成

- pipeline `83819`/monitor `1637` exit0；audit `78992` exit0，GT/recon/REQ/checkpoint/script/evaluator/report 哈希、544 names、两 gap pair selection、finite distance arrays 与 ratio/Pearson/normRMSE 重算均一致。主 gap≤8：1529 pairs/492 samples；Full/Body/Hands ratio=`1.051025/1.081848/1.038781`，Pearson=`.902140/.835363/.882855`，normRMSE=`.267482/.343116/.269800`。
- gap≤150 敏感性：3668 pairs/544 samples，ratio=`1.055022/1.082391/1.044518`，Pearson=`.906222/.845240/.893207`，normRMSE=`.265346/.334292/.259850`。结果不支持原 VQ 普遍把同 action 类内差异压成同一动作；平均距离稍扩张也不是额外有效多样性。非同文本 MM、pair 不独立，不提供伪 CI；结合 active-code/固定权重采样对照，优先定位 MGT 预测尖锐与采样，不盲重训 VQ。50k monitor `1613` 仍 live。

### 09-11｜VQ pair-distance retention 诊断已启动

- 已定位原 VQ175k 175k 重构报告：544 samples，checkpoint SHA=`fdf5af...`，原评测 codebook usage Body/Hand/Root=`2741/3217/4379`（各17895 tokens）；这是冻结评测范围，不与 historical38、8617 train-cache 统计混同。重构 H5 SHA=`d5ee5d9d9cf98c54df9b6603fc68abc22ce0168acba9529951040eda87df2534`，本轮未重算距离。
- 新工具 `tools/diagnose_vq_pair_distance_retention.py` self-check chunk `449c29` exit0；CPU4线程 pipeline `83819` 已启动，monitor `1637` 每600秒独占回收。输入原 VQ175k reconstruction report，先核验 GT/recon/REQ/checkpoint SHA，再 embed544 GT/重构；输出 `output/hhc_original_vq175k_pair_distance_retention_20260911.json`，尚无结果。
- 诊断按相同 action ID 配对，主 gap≤8、敏感性≤150，报告三部位距离均值比、Pearson、normalized-distance RMSE 与逐 pair 距离；这不是同文本 MM，pair 不独立，也不证明语义完整或质量通过。monitor `1613` 仍等50k。

### 09-11｜VQ active-code 与冻结请求结构补证

- `measure_hhc_token_compression.py --self-check --training-manifest` 重算 exit0：8617 active train samples，各通道287161 historical38 payload；used codes Body/Hand/Root=`9749/7121/6884`（各12288），全局有效词汇=`4950.7649/4436.8763/4441.1641`，与旧统计一致。不能据此称 VQ 仅激活1–2码，也不能把缓存全局熵等同 MGT 逐位置条件预测熵或证明语义完全保留。
- 544 冻结请求 normalized captions exact，重复文本组数为0、每文本仅1样本；因此无 GT→VQ 同文本重复的直接实证。下一步 action-group 内距离仅标为 proxy，不作正式同文本 MM；原 monitor `1613` 等50k不变。

### 09-11｜span v2 40k 冻结评分完成（未晋级）

- pipeline `70984`/monitor `1612` exit0；audit `17834` exit0，候选 H5 SHA=`0f31b1ccfcf12846bccad93df15c74d83c9c8df357464267e9d55edc573f6904`，control H5 SHA=`48ee89f9ebf5ce3d1f8acf6e2a5611c152a7821692c97fe74d58e93d15aa0915`，control checkpoint SHA=`124cd42604c2d8d2d526530e058a0b09e120faf698864621e536f07d7be3b804`；544 complete/finite、REQ/eval/GT/physical 一致。除 checkpoint 外，old control 缺 beta_policy/save_token_ids 诊断字段，不能声称所有 attrs 逐字相同。
- control→candidate FID Full/Body/Hands=`.423530/.337148/.486570→.394411/.250502/.478037` 改善；slide=`.327072→.348083 m/s`、F1=`.523550→.562119`，matching/部分 R 退化，未质量晋级、不额外 MM。bootstrap slide delta `+.02101111` CI `[+.00886898,+.03329428]`，F1 delta `+.03856969` CI `[+.00391674,+.07349191]`。
- native 40k Full FID=`.6075`（30k `.41091` 仍 best），MM `0` 仍 placeholder、真实子循环1/4；不混 frozen MM。继续既定50k pilot，monitor `1613` 每600秒等50k/PID消失，原 session `28378` 未终态。

### 09-11｜span v2 40k checkpoint 固定，冻结评测已启动

- monitor `1578` exit0 返回 `STEP_40000_CHECKPOINT_READY`；固定 hardlink=`output/hhc_circular_span_matched_validation_20260911/step_000040000.ckpt`。root session `19658` exit0：global_step=`40000`、epoch=`298`，SHA=`a223e1f23eeb6bab280b0a4b899d2065fdd898003dcecc184bd065b6f85d2f81`；audit `30149` exit0，hyper_parameters.args 相对30k无变化，runtime span contract 与 launcher/helper SHA 一致。
- pipeline `70984` 已启动，monitor `1612` 每600秒独占回收；544 export→CPU joint32→CPU physical(valid_frames/per-sample)，固定原 VQ175k、seed500123、50/T=.5/CFG3.5、requested betas、no-shape/no-contact/noise/replay/save-token-ids。尚无40k评分，不追加 MM；monitor `1613` 改为等待50k。

### 09-11｜span v2 30k paired MM 完成（无一致增益）

- root audit `75050` exit0；初次 `54270` exit1 仅误把布尔 `completed_repeats` 当计数11，纠正后两 H5 的100×11 completed/finite、REQ names/lengths、文件/JSON/checkpoint SHA、H5 attrs、协议/REQ/evaluator/runtime 全部一致（仅 checkpoint metadata 不同）。control/candidate H5 SHA=`5f11a009045143feae7f70906cd485ccd413e3cc4a656235190ce208f7a1d11e` / `f1ded42872a162b4a1190894e623ea32ab0a0908f6f8d398697c40b28b81cc37`。
- MM Full/Body/Hands=`.166925/.223767/.261704→.192432/.198570/.272696`；duplicate pairs=`711/5500→724/5500`，mean unique=`7.49→7.62`，root 独立重算一致。all-pairs bootstrap（10k、seed90210）delta/CI：Full `.05430294` `[-.04666430,.16907067]`、Body `-.02962807` `[-.11851366,.06222363]`、Hands `.00106874` `[-.07138280,.07170228]`，均跨0。
- 没有一致多样性改善或重复坍缩缓解；不追加30k grid，继续既定50k pilot观察40/50k。pipeline worker 已退出、JSON complete；原 session `22914` 已由 monitor `1585` 正常回收，exit0。

### 09-11｜span v2 native 30k 验证完成

- native Full FID=`.41091`，为新 best30k（20k `.50341`）；div=`9.1539`，R1/2/3=`.43933824/.63051471/.72610294`，matching=`3.7564303875`。
- native 100-subset MM 真实4/4=`.1765`，subset FID=`.83023`、div=`9.20940`、R=`.37/.66/.76`、matching=`3.6190307140`；此前 `0` 只是中途 placeholder，不替代 frozen MM。PID `3868579` 已恢复至 epoch236 batch94，仍存活；冻结配对 MM `22914`/monitor `1585` 仍运行，monitor `1578` 等40k。

### 09-11｜span v2 30k 冻结评分完成，MM 方向诊断已启动

- pipeline `25229`/monitor `1577` exit0；root audit `87790` exit0，candidate H5 SHA=`8fed9107fe1f34bcea8a4687b2eab57ace9ab22762d4add07613e747148cab16`，两 H5 544 complete/finite、names/lengths、REQ/GT/evaluator/seed/length policy、SHA/attrs/floor 协议一致。
- control→candidate FID Full/Body/Hands=`.339169/.437286/.477634→.272544/.290904/.356063`，slide=`.335857→.322701 m/s`，但 matching 三部位退化、F1=`.551843→.546104`；未晋级。bootstrap slide delta `-.01315563` CI `[-.02511621,-.00146350]`，F1 delta `-.00573927` CI `[-.04012491,.02762252]`。
- 因三 FID+slide 同向改善，100×11 MM pipeline `22914` 已启动、monitor `1585` 独占回收，仅判断多样性方向；训练 monitor `1578` 仍等40k，原 session `28378` 未终态。

### 09-11｜span v2 30k checkpoint 固定，冻结评测已启动

- monitor `1564` exit0；固定 hardlink=`output/hhc_circular_span_matched_validation_20260911/step_000030000.ckpt`。root audit `11396` exit0：global_step=`30000`、epoch=`223`，runtime span 配置/源 SHA 保留，文件 SHA=`6b4cb02f3a5f6e248d6cd65f1df44e8456c2882c0eb7297d949b65e038bfde22`。
- pipeline `25229` 已启动，monitor `1577` 每600秒独占回收；固定50/T=.5/CFG3.5、seed500123、batch32、requested beta、no-shape/no-contact/noise/replay。尚无30k质量/MM结果，不先追加 MM；control30k 文件待候选核验后复用。monitor `1578` 每600秒等待40k或 PID 消失。

### 09-11｜native 20k 验证收尾

- native 20k Full FID=`.50341` 仍为 best20k；MM 真实完成4/4=`.1818`，100-subset FID=`1.32655`、div=`9.14739`、R=`.49/.65/.74`、matching=`3.4359824657`。此前 `0` 是中途 placeholder；subset 指标不混作 frozen MM。20k质量保持仍失败，继续50k pilot，monitor `1564` 等30k。

### 09-11｜span v2 20k 冻结配对评分完成（未晋级）

- pipeline `44555` 经 monitor `1563` exit0；candidate H5 SHA=`4989db19eb7b6a8aef33e85906713a914911c0018b3a97fee714bf5ba48539cd`。Full/Body/Hands FID=`.558887/.573045/.617900→.352108/.508324/.401563`，但 slide=`.325428→.388338 m/s`、F1=`.539508→.531824`，未质量保持、未晋级、不额外 MM。
- root audit `12808` exit0：两 H5 544 complete/finite、names/lengths 与 REQ 一致，H5 SHA 与 joint/physical、checkpoint attrs、evaluator/GT/REQ/seed/effective_length_policy、GT diversity、physical floor/protocol 全部绑定一致；首次错误使用不存在的 `complete` 键 exit1，修正为 `completed_repeats` 后通过。
- pooled-rate bootstrap（10000、seed90210）：slide delta `+.0629104062` CI `[+.05173478,+.07429599]`；F1 delta `-.0076837595` CI `[-.04467526,+.02829080]`。native 20k FID=`.50341`（当前 native best20k，10k `.59884`）、div=`9.0366`、R=`.38970588/.58455882/.69117647`、matching=`3.85239673`；MM placeholder=`0`，真实子循环1/4，不能混作 frozen MM。继续既定50k pilot，monitor `1564` 等30k，原 session `28378` 未终态。

### 09-11｜span v2 20k checkpoint 固定，冻结评测已启动

- root monitor `1543` exit0 返回 20k ready；冻结 hardlink=`output/hhc_circular_span_matched_validation_20260911/step_000020000.ckpt`。torchload session `35113` exit0：global_step=`20000`、epoch=`149`，runtime_training_mask_intervention 配置/源 SHA 保留；sha256sum session `26281` exit0，SHA=`21fd03d5c595f648c3e0ab719ca3c88e5f84d54a8ec353fef05b2902040421c4`。
- 544 样本 GPU0 export→CPU joint→CPU physical(valid_frames, per-sample) 串行 pipeline `44555` 已启动，monitor `1563` 每600秒独占回收；固定50/T=.5/CFG3.5、seed500123、batch32、requested beta、no-shape/no-contact/no-noise/no-replay。尚无20k质量结果，不做额外 MM；原 session `28378` 未终态。

### 09-11｜20k control 来源核对

- 已定位 `output/hhc_joint_text_dropout_p0p2_matched_validation_20260910/control_step_000020000_cpu_confirmed_20260911.json` 与同目录 `control_step_000020000_physical_per_sample.json`；control H5 SHA=`c15082dd9eb765a26944dc4fe6b198f20c1c07ece218bb64a03265fb855cce8c`，544/valid_frames。
- 与 span 目录 control10k CPU 的 GT diversity 完全一致：Full `9.104939460754395`、Body `9.10047435760498`、Hands `8.466240882873535`。候选出结果后再复核来源并复用，暂不重复评分。

### 09-11｜span v2 native 10k 验证收尾，等待20k冻结节点

- monitor `1525` exit0；`model-000010000-0.5988.ckpt` 存在，native Full FID=`.59884` 仍为10k best。native 100-subset MM=`.2407`、subset FID=`1.19564`、div=`9.3171`、R=`.39/.60/.72`、matching=`3.74691224`；这些不能混作 frozen joint MM。
- PID `3868579` 仍 live（epoch76 batch97）。固定 rolling 10k 与 `model-000010000-0.5988.ckpt` 的896 state tensors 逐值一致，global_step均10000，CPU对比 exit0。持续 monitor `1543` 已建立，每600秒检查 PID 与20k latest/model checkpoint；不再轮询 `1525`。

### 09-11｜span v2 10k 配对冻结评分完成（未晋级）

- control CPU 复算 `82838` exit0；新 control/candidate JSON 的 GT diversity、REQ/GT/evaluator/seed/effective_length/samples 均逐值一致，H5 SHA 已核验。
- control→span：Full/Body/Hands FID=`.468141/1.228750/.817906→.479971/.818115/.493728`；slide=`.291672→.327429 m/s`、F1=`.522846→.528829`。Body/Hands FID 早期改善但 Full FID/滑步变差，未质量保持、未晋级；不做额外 frozen MM，继续既定50k pilot，20k 再同 step 验证。
- pooled-rate bootstrap（10000、seed90210）：slide delta `+.035756` CI `[+.024448,+.047084]`；F1 delta `+.005982` CI `[-.027237,+.039862]`。native MM 独立运行，placeholder 不替代冻结 MM。

### 09-11｜span v2 native 10k 验证进行中

- W&B 首次 native 10k：FID=`.59884`（当前 native best 10k），diversity=`8.90847`，R1/2/3=`.41544118/.59926471/.70036765`，matching=`3.85747`。
- native MM 显示 `0.0000` 仍是 placeholder，实际 MM 子循环当时 `1/4`；PID `3868579` 仍运行，native 10k validation 尚未收尾。native 与冻结 joint 指标不可混用。

### 09-11｜span v2 10k candidate 冻结评分完成，control 同 CPU 重算中

- pipeline `10473` 经 monitor `1532` 回收 exit0；root 核验 candidate 544 complete/finite、REQ names/lengths，H5 SHA=`fbed29668ccfcbb7fd982f882113311c8fd0511275795c8ab1610be28d0748d7`，physical 协议/实际 H5 hash/REQ/floor valid_frames 一致。candidate Full/Body/Hands FID=`.479971/.818115/.493728`，slide=`.327429 m/s`、F1=`.528829`。
- 旧 control 10k JSON 的 GT diversity 与当前 runtime 略不同，已启动同 CPU batch32 control 重算 session `82838`，正在运行；control physical 可复用 slide=`.291672 m/s`、F1=`.522846`。待 control 确认再报告配对结果，不以 candidate 部分 FID 宣布成功、不启动 MM。
- monitor `1532` 已终态；训练 cell `1525` 仍等待 native 10k model checkpoint，原 session `28378` 未终态。

### 09-11｜span v2 原生 10k validation 与冻结评测已启动

- v2 stdout 已到 `[Validation] Starting evaluation at iter10000`，PID `3868579` 仍运行。root 将 `latest10000` 固定 hardlink 至 `output/hhc_circular_span_matched_validation_20260911/step_000010000.ckpt`；audit `7891` exit0，global_step=`10000`、epoch=`74`、runtime span contract 存在，SHA=`1a079b2114f9c031942fd24682454aaf2b766286b76cdd691a6db24b4ca0e046`。
- GPU0 冻结544评测 pipeline `10473` 已启动，monitor `1532` 独占回收；输出 `step_000010000_main.h5/_cpu.json/_physical_per_sample.json`，固定50/T=.5/CFG3.5、seed500123、batch32、requested beta、no-shape/no-contact/noise/no-replay，CPU joint+valid_frames physical。尚无新质量指标，不先跑新 MM。
- 持续训练监控 cell `1525` 仍等待 native 10k model checkpoint（不是 rolling latest），训练未终态。

### 09-11｜span v2 step2000 导出集成 smoke 通过

- 初次 audit session `78867` exit1 仅因 rolling `latest-1000` 已被2000替换，加载前 FileNotFound，未产生 H5；root 将 step2000 hardlink 固定为 `output/hhc_span_v2_step2000_export_smoke2_20260911.ckpt`，保留 inode 防滚动竞态。
- GPU0 两样本官方 export session `12277` exit0；audit `30008` exit0：global_step=`2000`、runtime span contract 存在，H5 两条 complete/finite，motions=`(2,1,150,672)`、tokens=`(2,1,50,3)`，checkpoint SHA=`cf751d02b5bafbdb90dc4a112742bd5e88403467102ab9a8effefd67add0dc21` 与 H5 attrs 一致，VQ SHA 与原175k一致。
- 这是评测入口兼容性 smoke，不是质量指标；不跑两条 FID、不把 batch2 与正式 batch32 比较。训练尚未到10k validation，未来同 step 正式 checkpoint 需先固定再评。

### 09-11｜span v2 1000-step checkpoint 完整性验收

- session `73930` exit0；`latest-000001000.ckpt` CPU 可完整读取，global_step=`1000`、epoch=`7`，1823 model/optimizer tensors finite，optimizer_states 与 lr_schedulers 存在，runtime intervention 与 manifest 一致。bytes=`3401000630`，SHA=`79eb1e06601fe5304e9a51b08966a4fbbf5943e38727055235aa837325f79d65`；两 VQ namespace 各140个非FK tensors 与原175k 逐值一致。
- launcher 明确拒绝 resume；本项仅为保存状态完整性验收，不代表断点续训或全轨迹恢复通过。若恢复，须先实现带 span intervention 的验证入口；当前无质量验证，未改 live 源码。

### 09-11｜CFG4.5 校准完成（未恢复质量）

- session `83664` 经 monitor `1499` 回收 exit0；root 核验 544 names/lengths/finite、REQ/evaluator/GT、H5/joint/physical 绑定一致。候选 H5 SHA=`5b6fb2202360185be29e2de0de62e01dcd222be676f41c70cd68d72c6915a01e`。
- Full/Body/Hands FID=`.621480/.304352/.809347`，matching=`3.340484/3.373029/3.328315`，R1/2/3=`.400735/.619485/.724265`、`.428309/.632353/.731618`、`.363971/.568015/.689338`，diversity=`9.178613/8.621680/8.416808`；slide=`.329123 m/s`、contact F1=`.524506`。
- 相对 CFG3.5 三部位 FID 与 slide 全部更差，校准未恢复质量；停止此 arm，不做 MM、不扩网格。span v2 仍运行，overnight 监控 cell `911`。

### 09-11｜circular-span pilot v2 受控重试已训练（非终态）

- v2 session `28378`、PID `3868579`、W&B `256nu4hq` 当前约8057 optimizer steps（epoch60 batch17），Train/all=`1.200`、约4.47 it/s；已验收2000步历史 checkpoint，当前有 `latest-000008000.ckpt`/`last.ckpt`，尚无10k checkpoint/validation。已超过 v1 约4379步崩溃位置，但不能称底层故障已修复；cell `1525` 每600秒继续等待10k，225/2000 仅历史节点。
- 相同 50k/300k 日程与学习设置，仅 checkpoint interval=`1000`、加入 `-X faulthandler`；overnight 已分配真实600秒监控，cell ID 待回报。v1 exit132 证据保留。

### 09-11｜circular-span pilot 异常终止，待受控 v2 重试

- 原 pilot session `96186` / PID `3854556` 已回收 exit132；kernel journal 记录 `python3.12` 于 `10:35:21` invalid opcode。最终到 epoch32 batch91（约4379 optimizer steps，loss 约8.99–9.09），无10k validation、无5000 checkpoint；目录仅 manifest/source。无 OOM 或算法/硬件归因，底层原因未知。
- 计划相同 recipe、seed-controlled v2，仅加 `-X faulthandler`，checkpoint interval 5000→1000 以降低丢进度风险，不改学习设置/source；旧 cell `907` 已终态。CFG4.5 session `83664` / cell `1499` 仍 live。

### 09-11｜50k CFG4.5 单臂采样校准诊断（已完成）

- 因 50k joint-dropout MM 提升但 Full/Hands FID 变差，预定只测试 CFG=`4.5`（原 `3.5`），同 candidate50k/VQ175k、544 冻结请求、seed500123、50步/T=.5、batch32、paper_text_cfg、requested MGT/VQ beta、no-shape/no-contact、无 noise/root-replay。
- root exec session `83664` 已由 monitor cell `1499` 回收 exit0；Transformer exact architecture load、requested VQ preserved，结果见上方。相对 CFG3.5 全部 FID 与 slide 更差，未恢复质量，不做 MM。

### 09-11｜50k frozen MM 配对完成（多样性改善，质量未晋级）

- 原 session `2537`/monitor `1430` exit0、cell completed；两组100×11 completed/finite，names/lengths、H5/JSON/checkpoint SHA、requests/evaluator/runtime/sampling expectation 全部一致，generation metadata 仅 checkpoint/path SHA 不同。candidate H5 SHA=`0f4ea3c928a3066e4491c5d7174a15d25b4d3c17a696ba8494566d73f27b03cc`，control=`9af9e9b27b162bfe09a61f1c050c593d4cb81cb7209be3d2d0a7f2bc7a516f56`。
- Full/Body/Hands MM `.202199/.242318/.255215→.271381/.348422/.418696`（+34.2/+43.8/+64.1%）；duplicate `819/5500→370/5500`，unique `7.65→9.38`。root paired all-pairs bootstrap（10000 draws、seed90210）delta/CI：Full `.153285` `[.016216,.303090]`、Body `.106132` `[-.035510,.247937]`、Hands `.167461` `[.055416,.292561]`；仅辅助统计，不是正式 MM CI。
- 确有多样性信号，但 50k Full/Hands FID `.366109/.438952→.429040/.588571` 仍退化，未质量晋级；不延长该 candidate，继续已运行的 span 单因素 pilot。

### 09-11｜v4→span 基线差异与 sampler 局部回归核查

- v4 历史 manifest 与 span manifest 的 cache、dataset aggregates、loader contract、source-derived configuration、paper configuration 相等；resolved 参数仅 run/path/resume/W&B 与 50k diagnostic stop 不同。源码存在已知新增分支/可选项（InterHuman、temporal_stride、text_drop_prob、shapeaug hybrid、latent_dims fallback/采样策略）。
- root CPU 对照历史 launch sampler 与当前 `sample_intra_cas`：2 seeds×CFG(-1,3.5)、50 steps、T=.5，在合成 logits 下 token IDs 与 RNG 逐值一致，exec exit0。仅为局部回归证据，不代表真实训练全轨迹 bitwise 一致或质量结论；继续复用现有 export/scorer，同 step 评测。

### 09-11｜50k control frozen MM 评分完成

- `control_step_000050000_mm_cpu.json` 已完成；root 核验 H5 SHA=`9af9e9b27b162bfe09a61f1c050c593d4cb81cb7209be3d2d0a7f2bc7a516f56`、100×11 repeats 完成且 motions finite。
- Full/Body/Hands MM=`.202199488878/.242317542434/.255214601755`；duplicate=`819/5500=.1489090909`，平均 `7.65` unique motions。candidate 尚待完成，本次不下配对或质量结论；新版 `sampling_expectation` 字段与旧300k JSON不同，不作跨 step 同协议比较。cell `1430` 仍 live，不重启。

### 09-11｜circular-span matched pilot（已异常终止，历史记录）

- `tools/train_mgt_circular_span.py` 已将自身与 `audit_mgt_span_copy_shortcut.py` 保存到 run `source/span_runtime`，hash 写入并复核 manifest/checkpoint args；CPU Lightning integration session `66342` exit0。
- GPU1 正式 pilot 曾发生 optimizer 更新至约4379 steps（epoch32 batch91，loss约8.99–9.09），run=`output/t2i_trans/2026-09-11-10-20-01_hhc_no_shape_circular_span_pilot50k_seed114514_v1`，PID `3854556`，W&B `r4v70h1p`；随后 exit132，未完成 validation/无5000 checkpoint。
- root 只读复核 manifest 与 `source/span_runtime` 两文件和实际源码逐字节一致：launcher SHA=`de95069d5d41d287eb12fb56e5b3ee0821f32fed557b55d6d09328bb23cc20bd`，helper SHA=`f0222696d12ebda76c2ab45d52f5d148622aeb99c75dc9527bca3029a240e0e1`。无质量/MM 结论；旧 cell `907` 已终态，旧 joint-dropout 50k 配对 MM 已完成。

### 09-11｜circular-span 入口 CPU integration 通过，GPU smoke 已启动

- `tests/test_mgt_circular_span_integration.py` 真实 CPU Lightning 两 training steps + sanity/validation session `74675` exit0：mask 仅在 scope 内启用、validation helper identity 保持原版、weight 更新、bound method/`Trainer.fit` 恢复、checkpoint args 与 manifest 的 runtime intervention 标识一致均 PASS。
- 原 MGT PID `3537043` 已退出，GPU1 空闲；session `78308` 已回收 exit0。旧 smoke session `72348` 因 main 启动规则拒绝 `wandb disabled` exit1，未绕过规则；改为 `wandb online` 后 session `62679` exit0，两步 loss `9.63→9.61`。root checkpoint 验收 `58744` exit0：global_step2、浮点/optimizer finite、runtime intervention args/manifest 一致，非 FK 原 VQ tensors 与 net175000 逐值一致，SHA=`30e1c7031d9ddade75ec05dd0e87c70e2bc665497e8bb29b4754c47a7e23cee6`。长训练尚未启动，下一步先绑定独立入口 source snapshot；50k MM 仍由 cell `1430` 运行。

### 09-11｜circular-span 训练入口实现并通过 self-check（未启动训练）

- 新入口 `tools/train_mgt_circular_span.py` 仅在 `Trainer.fit` 期间、每次 `training_step` 临时替换 `utils.motion_masks.sample_kinematic_cascade_mask`，`finally` 恢复；validation/生成不变，原 Bernoulli 替换、ratio/count/global RNG 消耗保持，span 使用 CPU generator seed=`seed+1000000+global_step`。
- manifest/checkpoint args 标记 `runtime_intervention`，不支持 resume 或预载 MGT；self-check `19979`、py_compile 均 exit0，覆盖 count/ratio/tau/global RNG/tail/共同 visible 输入/异常恢复/恢复后精确重放。尚未做真实 Lightning 集成、GPU 冒烟或长训练；50k MM 仍由 cell `1430` 运行。

### 09-11｜mask span-copy 暴露诊断完成（非模型评测）

- 原手写 tau 统计因 body/hand 错位且无脚本/artifact，两个版本均不采纳；root 改用 `tools/audit_mgt_span_copy_shortcut.py` 复用真实 mask helper，并断言随机替换、每 part count、hand≥body≥root、END/PAD 与共享 visible 输入。self-check `70612`、实际 val32×64 draws session `59206` 均 exit0；JSON sample hash=`adf4b1c443ffb549cc3f2a15498905d6576c4023539c56538880d34c501d9221`，masked counts B/H/R=`53474/71372/28391`，两臂一致。
- copyable fraction（原→span）：±1 B/H/R=`11.4635/2.9661/19.1892%→1.6756/0.7118/3.0890%`；±2=`16.2471/4.3729/26.6317%→3.1698/1.2904/5.7131%`。这是缓存暴露诊断，无模型/训练/MM/质量结论；支持继续 mask-layout 单因素验证，原50k 配对 MM 仍由 cell `1430`/session `2537` 运行。

### 09-11｜text-dropout 50k candidate 冻结评分完成（未晋级）

- session `8789` exit0，root 验收 `82459` exit0；544 complete/finite、names/lengths、same evaluator/request/GT/source checkpoint 与 H5/joint/physical 一致均 PASS。candidate H5 SHA=`90286f5e5a465fe18583a6ad9e97c3201cd50416f3be59a36435fa6d97f4817b`。
- control→candidate：Full FID `.366109→.429040`、Hands `.438952→.588571` 明显变差，Body `.370827→.267876` 改善；三部 matching 改善，R 多数提升；slide `.326885894795→.308150647252 m/s`、F1 `.519286903240→.527835454479` 改善。整体未质量晋级，MM 仍由 cell `1430` 运行，不能覆盖 main 质量结论。

### 09-11｜50k frozen MM 诊断 queue 已排队

- queue `2537` 等待 GPU0 exporter PID `3847378` 退出且 candidate main H5 complete/finite，随后按 control→candidate 串行：100 frozen `mm_indices`×11 repeats、seed600123、batch32、50/T=.5/CFG3.5、原 VQ175k、no-MGT-shape、paper_text_cfg、requested betas、无 contact/root-replay并存 tokens；每臂 GPU0 export 后 CPU scorer32。
- 输出为 matched_validation 目录下 `control/candidate_step_000050000_mm.h5` 与 `_mm_cpu.json`；目的仅区分训练改动的多样性信号，不覆盖 main 质量取舍，不与300k/native MM混用。50k main 仍由 monitor `1417` 回收；gate cell `1410` 已完成。

### 09-11｜50k frozen MM queue 已进入运行

- root 已核验 control 50k MGT 严格加载通过、requested VQ 保留；queue `2537` 的 control repeat0 已生成 `64/100`。50k main pipeline 已完成并由 root 验收，详见最近完成；MM 仍由 cell `1430` 独占运行。
- root 只读核对 `utils/motion_masks.py`：随机替换先于 kinematic mask，训练/validation 共用 helper，未改源。下一项仅做 CPU 邻居复制诊断（固定 val32、相同随机替换与每 part mask 数量，对比原 mask/环形连续 span，统计 masked 目标能否从 ±1/±2 可见 token 复制）；尚未运行，无 MM 或 mask 改善结论。

### 09-11｜generated sliding gate 三 seed 收尾（未晋级）

- cell `1410` completed，queue `55670` exit0；三 seed 均独立核验 fixed gate mask、整条 motions/tokens exact、544 complete/finite、names/lengths、H5/joint/physical 哈希及 eval/GT/requests 一致。
- 算术均值 control→gate（FID/matching/R1/R2/R3/diversity）：Full `.244061/3.039665/.436275/.645833/.765931/9.645012`→`.243178/3.040629/.435049/.642770/.765931/9.644324`；Body `.235810/3.072519/.455882/.644608/.781863/9.197671`→`.236956/3.074170/.455882/.645221/.781250/9.198364`；Hands `.213729/3.079655/.422794/.613971/.739583/8.842378`→`.214882/3.079599/.420956/.612132/.738971/8.843904`；slide `.303646→.303155 m/s`、F1 `.557692→.558174`。
- Body/Hands FID 三 seed 均变差，整体质量保持未通过、未晋级；不再扩大或调 gate 阈值，不做 gate MM。第一项继续 50k text-dropout，50k candidate monitor `1417` 仍独占。

### 09-11｜50k control root 验收完成

- root 验收 session `7928` exit0：544 complete/finite/order names/lengths、H5/joint/physical actual SHA、checkpoint/request 一致均 PASS。control H5 SHA=`2616217ccea4d4ae97d284492cf5ac4697bfb5f16269ef1101fdee2c5e3fd0cd`，checkpoint SHA=`0e2308c493124ad87f69528606175c30dcd1cc0324031db8ec5c0c778724e055`；overnight 原始 control CPU session `8997`、physical `27270` 均 exit0。
- 50k candidate 原 session `8789` 仍由 monitor cell `1417` 独占回收；gate 三 seed queue `55670` 已完成并 exit0，不再轮询。

### 09-11｜generated sliding gate seed500125 评分完成（未晋级）

- root 独立验收 fixed gate mask、逐条 motions/token exact、H5 actual SHA=`2e7c517623c01fa8e9ec2d6a3d6c027c9d9a33268a7a12ec3ef7831e4f6b350c` 及 scores/eval/requests/GT 一致，均 PASS。
- control→gated：Full FID `.244283→.242395` 改善，Body `.234251→.234982`、Hands `.215322→.216213` 略差；slide `.303833447997→.303403546143 m/s`、F1 `.556776373420→.557052694851` 改善。两 seed 现均 Body/Hands FID 略差，未晋级。
- gate 三 seed已完成并未通过质量保持；50k candidate session `8789` 仍由 cell `1417` 独占，不重复启动。

### 09-11｜text-dropout 50k candidate pipeline 已完成

- checkpoint 保留 session `28520` exit0；`candidate_step_000050000.ckpt` 与 `latest-50000` same inode，CPU `global_step=50000`、epoch=`373`、bytes=`3401001523`，SHA=`1333fc1ed71c0381ff963bbc31215d378891b45cd818480bcd66e9c4da73449a`。
- candidate 50k GPU0 export544 + CPU joint32 + physical16 valid_frames pipeline session `8789` 已完成并验收；overnight control CPU/physical 也已回收完成。
- native 50k Full FID=`.5737`（30k `.5415` 仍 best）为 agent 核查，不等于 frozen 结果；MM 尚未齐。

### 09-11｜generated sliding gate seed500124 评分完成（未晋级）

- monitor cell `1407` 已完成，原 pipeline `12176` exit0；root 独立核验 gated H5 实际 SHA、joint/physical 同源及 eval/requests/GT 一致。
- control→gated：Full FID `.242136→.240200`，Body `.238888→.240367`，Hands `.213007→.213873`；三部 matching 均改善，但 Full R2、Hands R1/R2 下降。slide `.303541606539→.303119794334 m/s`，F1 `.559557202965→.559617167339`。未质量晋级，MM 未测。
- 原 gate 规则不调参；seed500124–126 已由 queue `55670` 串行生成并评分，三 seed 汇总见最近完成。

### 09-11｜generated sliding gate seed500124 混合 H5 已生成，评分待定

- `tools/apply_hhc_generated_sliding_gate.py` 经 root 修正原 SyntaxError，并补齐 physical/H5 names-request binding；self-check/py_compile 均 exit0。session `84839` exit0，生成 `output/hhc_generated_sliding_gate_validation_20260911/gated_seed500124.h5`，SHA=`92795b495ffc1ee492d400c095567d3fd19433d8794a2047926fb53a91cdeadd`。
- root 独立核验逐条 mask/motions/tokens exact、544 complete/finite、names/lengths；accepted `320` 中真正变化仅 `36`，全544样本保留，不能把320称为真正多样化样本。仅使用生成滑步门槛，不用 GT contact/FID 选样。
- CPU joint32+physical16 valid_frames 串行 pipeline session `12176` 已启动，输出 `gated_seed500124_cpu.json` / `_physical_per_sample.json`；尚无评分结果或 MM。

### 09-11｜三 seed generated-motion sliding gate 可行性统计（非完整评测）

- 固定规则：仅按生成动作 `foot speed sum<=对应生成 control` 且 `grounded transitions>=control` 接受 candidate，否则完整回退 control；保留全部样本。不用 GT contact/FID/embedding 选样，labels 仅事后评估。
- 只读统计：seed500124 accepted `320`、changed `36/260`，slide `.303541606539→.303119794334`、事后 F1 `.559557202965→.559617167339`；500125 `321/37/260`，`.303833447997→.303403546143`、`.556776373420→.557052694851`；500126 `334/47/257`，`.303563661217→.302942036184`、`.556742390051→.557851446913`。
- 以上来自原 per-sample counts，非重新完整评分，MM 未知；`apply_hhc_generated_sliding_gate.py` 已由 token agent 实现、待 root 审核，尚未启动评测。当前仍属任务1探索。

### 09-11｜token restoration 两臂消融评分完成

- session `68433` exit0；restore_hands physical 与 restore_body joint/physical 全部完成，root 独立核验 H5/joint/physical SHA、eval/requests/GT 一致。restore_hands physical：slide `.3037087260 m/s`、F1 `.5591020694`；restore_body：slide `.3036233598 m/s`、F1 `.5594002307`；两者仍未恢复到 control 的 `.3035416065/.5595572030`。
- restore_body H5 SHA=`4f46268a43118f712d5598beee4a29ae82c1895af70263edee4b0497bdd69fb0`。四臂消融已结束，不再写作评分运行中；token 恢复未消除质量取舍。
- 后续仅有可行性筛查：seed500124 的 320 条 speed/grounded 门槛样本中仅36条真正变化，混合滑步 `.3031197943` vs control `.3035416065`；未输出混合 H5，无新 MM/质量结论，未启动少量质量守门采样。

### 09-11｜token restoration restore_hands joint 完成，physical 参数故障已修正

- 原 session `47722` exit2：physical 命令误用 `--floor-height-policy`，argparse 拒绝；restore_hands joint 已成功写出，未重复 joint。其 H5 SHA=`a8eab12d09df18d3018feed2c516e365d91ee14414537086191a63ea461b4b59`，root 独立核验 eval/requests/GT 一致。
- restore_hands physical、restore_body joint、restore_body physical 已由新串行 session `68433` 继续；当前仅 restore_hands joint 有结果，不作质量全过结论。

### 09-11｜full544 decoder coupling RMS 诊断

- 以 valid frames 和 `HHI_BODY_MASK/HHI_HAND_MASK/HHI_ROOT_MASK_OLD` 聚合，与 fresh control 比较：fresh candidate body rotation6d RMS=`.0113282943`、hands=`.0192964787`、root translation=`.0033399059 m`；restore_hands=`.0104026787/.0061921978/.0027953785 m`，restore_body=`.0046162455/.0182269865/.0019064828 m`。
- 结果说明固定手部 token 仍会产生来自身体 token 的手部输出差异，固定 root token 仍有 root 平移变化；这是 decoder 耦合诊断，不是 MPJPE/FID 质量改善证明。session `47722` 本轮仍活、无结果；45k checkpoint 已存在，50k 尚无（训练状态为 overnight 核查）。

### 09-11｜text-dropout 40k candidate 冻结评分完成（未晋级）

- session `18879`/验收 `94621` 均 exit0；候选 544 complete/finite、names/lengths、同 eval/GT/requests、实际 H5/checkpoint SHA 均 PASS。候选 H5 SHA=`4af4ea779e21c4a3e14bd6901a6871178cef25c34cbd89bb161a3d3ee057336b`。
- 相对 control，Full/Body/Hands FID=`.379335/.284926/.476173` vs `.423530/.337148/.486570`，matching=`3.138947/3.219336/3.109669` vs `3.236657/3.233866/3.219543`；slide=`.342347` vs `.327072 m/s`，contact F1=`.536920` vs `.523550`。Full R2/R3、Hands R3 下降；三项 FID 与 matching 改善但滑步恶化，未达质量不降，未晋级。frozen MM 未测；token restoration session `47722` 尚无结果。

### 09-11｜posthoc token restoration 解码归因启动

- `tools/decode_hhc_token_restoration.py` 已经 root 审核修正并 self-check PASS；smoke8 session `65847` exit0，fresh candidate/control 各8/8 与 seed500124 archive bitwise exact、`maxabs=0`，hands/body 两臂 mixed token IDs、finite/completion/zero-padding 均 PASS。
- full544 四臂 B1 解码 session `2841` exit0；root 独立核验 fresh candidate/control 均544/544、与原 archive bitwise exact（`maxabs=0`），两恢复臂 mixed IDs、names/lengths、finite/completion/raw-zero padding 全部 PASS。复用旧 candidate/control 评分，不重评一模一样 motions。
- 两恢复臂 CPU joint(batch32)+physical(batch16 valid_frames per-sample) 串行 pipeline session `47722` 已启动，输出原目录 `*_cpu.json` / `*_physical_per_sample.json`，尚无结果。
- 40k 原 pipeline session `18879` 已 export `544/544`，进入 CPU 评分阶段，未终态；此前 control 验收 session `6435` exit0，但候选尚未验收，不报告配对结论。

### 09-11｜root replay 三 seed（500124–126）队列完成

- cell `1339` 已完成回收，session `64522` exit0；三 seed 均独立核验 544 complete/finite、source/eval/GT/requests、control token/root IDs exact 及 H5/joint/physical 一致。三 seed 算术均值：Body FID `.229670` vs control `.235810` 均改善；Hands FID `.217530` vs `.213729`、滑步 `.303918` vs `.303646 m/s` 均退化；Full FID `.244121` vs `.244061` 几乎不变，整体不达质量不降，未晋级。
- 不把开发 seed 混入三 seed 结论；MM 提升仍仅限既有 100×11 实验，无新三 seed MM 结论。所有队列结束，当前无 cell `1339` 运行。

### 09-11｜root replay seed500125 完整对照已核验（未晋级）

- root 独立核验 544 complete/finite、provenance/eval/GT/requests、H5 SHA=`1925de65cfb019795663b973bbc5e2537a07466833e87c2666bef4022bd02e01`、candidate root-replay token IDs 与旧 control 全相等及 root IDs 相等，均 PASS。
- candidate→control：Full/Body/Hands FID=`.248630/.224001/.219286` vs `.244283/.234251/.215322`，matching=`3.040329/3.072072/3.073315` vs `3.036593/3.072344/3.077598`；slide=`.303991` vs `.303833 m/s`，F1=`.559202` vs `.556776`。整体仍未晋级；本 seed 无新 MM 评估。seed500126 正在原队列导出。
- 另一训练线 native 40k：MM `.3970`、subset FID `1.9291`，不可与 root-replay frozen MM/质量混用。

### 09-11｜root replay seed500124 完整评分完成（未晋级）

- seed500124 候选 H5/CPU joint/physical 已完成；root 独立核验 544 finite、同 eval/request-GT、候选 H5 SHA=`4c75ba606e4b8e420732ca2e146f6d764af985b1e47fa9f88d642fbc8f2cb058`，root-replay token IDs 与旧 control 全相等、root IDs 相等。首 seed set-e 链路通过，但总 session `64522` 未结束；seed500125 已生成文件。
- candidate→control：Full/Body/Hands FID=`.237796/.235066/.213476` vs `.242136/.238888/.213007`，matching=`3.035654/3.084247/3.080662` vs `3.040499/3.074780/3.079741`；slide=`.303857` vs `.303542 m/s`，F1=`.559169` vs `.559557`。混合结果，未晋级；本 seed 无新增 MM。
- native 40k 日志：Full FID `.5762`、matching `3.6598021984`、R `.41911765/.62683824/.71875`、diversity `8.9921`；30k `.5415` 仍为 best，40k MM 仍运行中。

### 09-11｜root replay MM paired-bootstrap 辅助诊断

- root 独立以100 prompts同序、seed90210、10000 draws、三部位共用同一重采样索引矩阵重算 all-pairs（55 pairs）delta：Full/Body/Hands=`.05815/.07718/.06447`，三者 CI 均为正；仅为辅助诊断，不是官方随机配对 MM CI 或质量非劣性证据。

### 09-11 08:41 BST｜三 seed root replay 已开始实际导出

- 首个 seed500124 exporter PID `3808108` 已实际运行，目标 `output/hhc_root_replay_validation_20260911/root_replay_seed500124.h5` 已创建；其余两 seed 按原 session `64522` 串行执行，cell `1339` 继续独占回收。当前不是评分完成或质量通过。

### 09-11｜text-dropout 30k candidate 冻结 CPU/physical 评分完成

- session `32506` exit0；root 独立核验候选 544×1 complete/finite、request names/lengths、checkpoint SHA、seed500123、batch32、no-MGT-shape/no-root-replay 及 CPU evaluator/GT provenance 均通过；候选 H5 SHA=`fc24955602b43e69841db492ebdb1416b6edbbcfceaae9fc886138c580965f05`。
- 相对 control，Full/Body FID=`.399425/.455314` vs `.339169/.437286` 变差，Hands matching/R 变差；Hands FID=`.333738`、滑步=`.311264 m/s` 改善，但 contact F1=`.534756` vs `.551843` 变差。冻结 MM 尚未测，不混用 native `.3661`；训练继续原50k计划，root replay 三 seed 仍由 cell `1339` 回收。

### 09-11｜native 监控 ownership 收敛

- native monitor cell `1339` 独占回收原 session `32506`（30k）再 `64522`（三 seed）；每次原 session 仅 `write_stdin` 60 秒、不唤醒模型，只有终态才通知，失败即停止监控。两队列均已 set-e。
- 后续 main 只使用 `functions.wait` 等待 cell `1339`，不直接 poll `32506/64522`，不启动重复任务；30k 仍按600秒调度间隔等待。control seed500124 CPU 评分当前由 PID `3793565` 运行。

### 09-11｜root replay 三 seed 队列已创建（尚未运行）

- native queue session `64522` 已创建，等待 30k pipeline PID `3784161` 退出；之后每600秒检查并串行执行 seed500124/125/126 的 544×1 root-replay 导出、CPU joint32 与 valid_frames physical16。
- 输出分别为 `root_replay_seed500124.h5/_cpu.json/_physical_per_sample.json` 等；set-e 失败即停，已有 H5 拒绝覆盖。当前仅排队，未声称任何 seed 完成；control 三 seed 由 overnight_metrics 负责。

### 09-11｜root replay full544 MM/质量评分完成

- root MM cell `1298`：export session `35016` exit0（1674.82s），CPU scorer exit0；候选 H5 SHA 为 `4e6d4373d9d463b19610ded5c1a9d3ae8f2bc6efe447ae2ed88ab5ee12da32b4`。候选 MM Full/Body/Hands=`.079413/.114994/.146442`，exact duplicate rate=`.482`（2651/5500）；control=`.043909/.047873/.083011`、`.678545`（3732/5500），MM 与多样性改善。
- root 已独立核验 requests/order/runtime/evaluator/checkpoints/sampling 一致。完整生成质量仍未通过逐项门槛：Full/Body/Hands FID candidate→control=`.230924/.236343/.211511` vs `.240967/.234382/.210853`，slide=`.304015` vs `.303470 m/s`，contact F1=`.558232` vs `.557580`；仅部分指标改善。后续 seed500124–126 待 root 排队，未启动。

### 09-11｜joint text-dropout mask-exposure CPU 诊断

- session `20663` exit0；配置/完整指标见 `output/hhc_joint_text_dropout_10k_30k_cfg_entropy_val32_cpu_20260911.json`（32 validation、seed90123、CPU batch4、CFG3.5、T=.5、`paper_text_cfg`）。10k→30k 条件 fully-masked 有效候选数 body `4.46→5.45`、hand `4.86→6.73`、root `11.21→11.04`；guided 约 `1.47/1.48/1.70→1.48/1.48/1.59`。
- 不能据此说条件分支整体变尖；CFG 首步高度集中，但 body/hand 几乎不变，首步熵不足以解释整体 MM 下降，后续需看 rollout。该诊断不是完整生成/MM验收。

### 09-11｜root replay exporter 完成

- exporter session `35016` exit0，实际耗时 `1674.82s`；100×11 finite、frozen names/lengths/root IDs 检查 PASS。cell `1298` 的自动 CPU 评分仍进行中。

### 09-11｜text-dropout native 30k validation 已核对

- `wandb/run-20260910_135522-781f0cgp/files/output.log`：544 样本 Full native FID=`.5415`（当前 best；10k `.66997`），matching=`3.7104296684`，R=`.42279412/.57904412/.67647059`，diversity=`8.8504`。
- 同轮 100×11 native MM=`.3661`，低于 20k `.4979`、10k `.6344`；其 subset FID=`1.4992`，不可当作 544 样本 Full FID。训练已继续至 epoch226，未停止；冻结 30k candidate session `32506` 仍等待 root MM 导出。

### 09-11｜30k 候选冻结质量评测已排队

- 原生串行任务 session `32506` 已启动；每600秒检查当前 root replay MM 导出 PID `3779487`，退出后才使用 GPU0。当前为等待，不是导出完成。
- 复用现有 exporter：已核验的 `candidate_step_000030000.ckpt`、原 VQ175k、544 requests×1、seed500123、batch32、50步/T0.5/CFG3.5、no-MGT-shape、requested beta、无 contact guidance、默认采样（无 root replay）；目标 `candidate_step_000030000.h5`，存在则拒绝覆盖。
- 导出成功后自动串行 CPU joint（batch32）及 physical（batch16、valid_frames、per-sample）评分，任何失败即停止；OMP/MKL均4。结果位于 `output/hhc_joint_text_dropout_p0p2_matched_validation_20260910/`，尚未取得候选30k冻结指标，不宣称通过。

### 09-11｜root replay 三 seed 控制 CPU joint 对照完成

- 三个既有 544-sample control H5 独立核验通过：H5 SHA、seed、300k checkpoint SHA、request SHA、batch32、完成标记、有限值、names/lengths 全部一致；physical per-sample 结果复用，未重跑。
- CPU joint sessions `86574`、`86407`、`44372` 均 exit0，输出分别为 `output/hhc_root_replay_validation_20260911/control_seed500124_cpu.json`、`control_seed500125_cpu.json`、`control_seed500126_cpu.json`；Full FID 分别为 `0.2421364950`、`0.2442833415`、`0.2457627900`。
- root replay 候选尚未完成三 seed 对比，此处不作质量比较或结论。

### 09-11｜no-shape v4 control 30k 冻结评分

- 已对 `output/hhc_no_shape_v4_pilot50k_official_val/step_000030000.h5` 完成 CPU joint/physical 评分；H5 为 `544×1×150×672`、finite、names/lengths 与 frozen request 一致，SHA256=`9a1cf882dfc35562b8ddac8d5134b21818b47496d0060f48f47fef7a3a7abe1d`。请求 SHA256=`ade004f1e586a6c94afa41c1dd15defc90d1493c09ae5b9b968f2fecb1e99175`。
- joint session `79603` exit0：Full FID/matching=`.3391693063/3.3770283671`，Body=`.4372858205/3.5676630946`，Hands=`.4776343683/3.2313685417`；输出 `control_step_000030000_cpu_confirmed_20260911.json`。physical session `44480` exit0：contact F1=`.5518434583`，slide=`.3358568049 m/s`，输出 `control_step_000030000_physical_per_sample.json`。
- 该 H5 直接元数据记录 source checkpoint=`model-000030000-0.6630.ckpt`、`checkpoint_sha256=cfefb2d51e5da12360bfaec7497c51b86610363b7bd94c5d37651096e83baf4a`；与 `step_000030000.json` 的 `generation_metadata` 及当前 checkpoint 独立 SHA256 核验一致。

### 09-11｜MGT 30k checkpoint 已保留

- 旧 watcher 因入口仅固定 pre/post callback `step20000`，未找到 `prevalidation-000030000.ckpt`；watcher PID `3782037` 已 TERM，session `6457` exit143。训练本身未终止。
- 实际 `latest-000030000.ckpt` 已 CPU 核验并以 no-overwrite hardlink 保留至 `output/hhc_joint_text_dropout_p0p2_matched_validation_20260910/candidate_step_000030000.ckpt`：session `59509` exit0，`global_step=30000`、`epoch=223`、bytes `3401000693`、SHA256 `263b52dfdcce59434998306bd0cc087b1fa9fa03895a40fabaac36febb023acb`、same inode。native Full FID=`.5415`（当前优于10k `.66997`）；native MM 仍运行，不与冻结评测混比。root MM cell `1298` verified running PID `3779487`。

### 09-11｜MGT 40k checkpoint 已保留

- CPU `torch.load` 核验 `global_step=40000`、`epoch=298`；连续两次 10 秒 stat 的 size/mtime 稳定，bytes=`3401001012`，SHA256=`918734ed1ec85e581bf56c666db7972a0d16640a95f5f57fa5547dc3d630e03d`。
- 以 no-overwrite hardlink 保留至 `output/hhc_joint_text_dropout_p0p2_matched_validation_20260910/candidate_step_000040000.ckpt`，与源 inode=`126683786` 相同。40k native validation 刚开始，尚无冻结质量分数，不作推断；原训练 session exit code 未在本次核验中回收。

### 09-11｜no-shape v4 control 40k CPU 评分

- `output/hhc_no_shape_v4_pilot50k_official_val/step_000040000.h5` 核验通过：544×1×150×672、finite、completed、seed500123、batch32、no-MGT-shape、request SHA 一致；checkpoint 为 model-000040000-0.5986.ckpt，attrs 含 checkpoint SHA。
- CPU joint session `49560` exit0，输出 `output/hhc_joint_text_dropout_p0p2_matched_validation_20260910/control_step_000040000_cpu_confirmed_20260911.json`；Full/Body/Hands FID=`0.4235303550/0.3371483307/0.4865695500`。
- CPU physical session `66343` exit0，输出 `control_step_000040000_physical_per_sample.json`；sliding=`0.3270723395` m/s，contact F1=`0.5235497052`。仅为 control 对照，未启动 40k candidate 导出。

### 09-11｜no-shape v4 control 50k CPU 评分

- `step_000050000.h5` 核验通过：544×1×150×672、complete、finite、seed500123、batch32、no-MGT-shape、request SHA 一致；H5 checkpoint attrs 与实际 checkpoint SHA 一致。
- CPU joint session `8997` exit0，输出 `control_step_000050000_cpu_confirmed_20260911.json`；Full/Body/Hands FID=`0.3661092417/0.3708269550/0.4389521348`。
- CPU physical session `27270` exit0，输出 `control_step_000050000_physical_per_sample.json`；sliding=`0.3268858948` m/s，contact F1=`0.5192869032`。仅为 control 对照，未启动 candidate 导出。

### 09-11｜root replay MM 监控交接（评分未完成）

- root 当前由 functions cell `1298` 独占监控原始 export session `35016`，每60秒检查且不唤醒模型；其他 agent/root 调用不要重复轮询，只用 `functions.wait` 获取新输出。
- export 完成后由同一 cell 核验 100×11 的 finite、name/length 与 root IDs，再串行启动 CPU `score_hhc_validation_mm.py --root-replay-control --confidence-noise-scale 1 --confidence-ranking log_odds --no-mgt-shape --repeats 11 --generation-seed 600123 --batch-size 32`，输出 `root_replay_log_odds_noise1_mm_cpu.json`。
- 状态持久化键为 `root_replay_mm_monitor_state` / `root_replay_mm_score_session`，耐久时间记录 `.../mm_export_time.txt`；当前没有评分完成结论，其他任务不变。

### 09-11｜text-dropout native validation 10k/20k 日志核对

- 原生训练验证日志 `wandb/run-20260910_135522-781f0cgp/files/output.log` 显示 Full native FID：10k=`.6700`（best），20k=`.7886`；20k matching=`3.72060037`，R=`.430147/.617647/.707721`，diversity=`9.1278`。
- 独立 100-subset×11 的 native MM diagnostic：10k=`.6344`，20k=`.4979`；同段 20k subset FID=`1.1259`，不是 Full FID，且首个完整评估的 MM 显示为 placeholder `0.0000`。这些 native 数值不可与 frozen joint 协议混比，也不覆盖已有 20k frozen joint 结果。

### 09-11｜root replay full-quality prereg（尚未运行）

- 新评分开关 `--root-replay-control` 默认关闭；计划每 batch 两次 pass：先生成 probability/noise=0 control，再生成 log-odds/noise=1 candidate，并按 stage replay control root。恢复 control batch RNG chain，candidate 不消耗下一 control batch 的 RNG；control/candidate token IDs 均归档。
- 预注册完整配置：544、seed500123、batch32、50 steps、T=.5、CFG3.5、no-shape、300k×4、原 VQ175k、无 contact、requested betas；后续 CPU joint32/physical16，并与现有 CPU-confirmed control 对比。首轮32条 self-replay parity gate 由 overnight 先跑；当前无 MM/质量结果。
- 两 pass 与 debugger overhead 需先测成本，未形成生产结论；共享训练/模型未修改。

### 09-11｜root replay 32 条 parity gate 完成

- session `5488` exit0；root 独立 `np.array_equal` 核对旧 control 前32条两组 motions，以及 candidate/control token IDs，均通过，`maxabs=0.0`。
- `output/hhc_root_replay_validation_20260911/source/` 四个 source 文件 SHA 全部与登记值一致。frozen544、noise1/log-odds full replay 当前由 PID `3762168` 在 GPU0 运行（`/usr/bin/time` PID `3762167`）；原始 agent session 仍待回报，不伪造其句柄。
- 质量指标仍未测量，frozen 配置未改变；10 个 unittest、self-check 与 5 个 plain checks 已通过。

### 09-11｜root replay 完成（仅 token/解码诊断）

- session `50120` exit0；`output/hhc_root_replay_length_quantiles_val8_20260911/summary.json` 记录四个 repeat 均通过 archived control token equality、fresh self-replay token 与 CPU/device RNG 校验，paired root IDs 全部精确匹配 control。
- replay identical token pairs 为 body/hands/root=`28/37/43`；duplicate motion pairs `28/48`（旧 control `42/48`），unclamped log-odds `20/48`（交叉计数仍在核对）。fresh decoded root feature RMS=`0.002751/0.001564/0.002359/0.001980`，单位须按 root translation mask 复核；该结果只说明 root tokens/解码诊断，不作 FID、MM 或质量结论。20k 完整结果已由 overnight 记录。

### 09-11｜interleaved cascade 分阶段 replay 诊断登记（尚未运行）

- root核对实现确认：即使关闭 root noise，body/hands 改变仍会影响后续 root logits；VQ 联合解码也意味着固定 root token 不等于固定世界坐标 root 几何。
- 下一诊断拟对 paired-repeat control 做 per-stage replay：相同8个长度分位×4 repeats，重放 control root token trajectory，再在剩余 streams 应用 noise1/log-odds，保存归档 traces；使用隔离脚本，不改共享模型/训练。
- 当前仅实现审查，未 GPU launch；不宣称 MM 增益或生产方法。20k candidate 评分 session `30914` 另行进行中。

### 09-11｜text-dropout control20k CPU joint/physical 复核完成

- control20k 现有 H5 的 CPU joint 复核 session `33779` exit0，新增 `control_step_000020000_cpu_confirmed_20260911.json`；physical CPU 复核已完成并写入 `control_step_000020000_physical_per_sample.json`。两者均使用冻结544请求/GT；physical 为71026 valid frames。
- Joint Full/Body/Hands：FID `.5588873449/.5730450995/.6179003124`，matching `3.5425904779/3.7218300735/3.3737083603`，R@1/2/3 分别 `(.3805147,.5716912,.6911765)/(.4025735,.5882353,.6966912)/(.3694853,.5477941,.6911765)`，diversity `9.2039394/8.7279549/8.2123671`。
- Physical：滑步 `.3254276125 m/s`、接触 F1 `.5395078553`；仅补齐 control20k 对照，不代表 candidate20k 已评测。

### 09-11｜confidence noise0.5 physical paired bootstrap

- 已保存 `output/hhc_no_shape_confidence_validation_20260910/log_odds_noise0p5_paired_prompt_bootstrap.json`：544 对齐 prompts、seed90210、10000 次同索引有放回抽样。Slide delta（noise0.5-control）`-0.00238322 m/s`，95% CI `[-0.00613865,+0.00086511]`；contact F1 delta `-0.00446482`，95% CI `[-0.01661966,+0.00682458]`。
- 两区间均包含 0；这是 prompt bootstrap 不确定性，不是训练 seed 方差，也不是 non-inferiority 证据。GPU0 candidate 20k 四臂导出仍待完成。

### 09-11｜log-odds noise 0.5 paired CPU joint/physical confirmation

- `log_odds_noise0p5_main.h5` passed the frozen 544-request completeness check. CPU joint scorer sessions `40121` (control) and `53119` (candidate) both exited 0; outputs are `control_main_cpu_confirmed_20260911.json` and `log_odds_noise0p5_main_cpu.json`.
- Full/Body/Hands FID: control `0.240967/0.234382/0.210853`, candidate `0.214715/0.229177/0.219536`. Matching: control `3.028678/3.081421/3.084399`, candidate `3.040173/3.108250/3.093625`. Candidate R@1/2/3: Full `0.433824/0.647059/0.750000`, Body `0.457721/0.647059/0.773897`, Hands `0.417279/0.621324/0.735294`.
- CPU physical session `36066` exited 0: candidate sliding `0.301087 m/s`, contact F1 `0.553115`; control `0.303470 m/s`, `0.557580`. MM is unmeasured; this is not a promotion or quality-preservation claim.

### 09-11｜true-shape beta factorial val16 diagnostic complete

- All four 16-request × 4-repeat arms completed with frozen names/lengths, finite motions, and request-bound metadata. Control requested/requested H5 was completed by root session `17937`; serial zero/requested, requested/zero, and zero/zero exports exited `0` in sessions `39158`, `45021`, and `75862`.
- Exact byte-duplicate motion pairs per 96 repeat pairs: requested/requested `68`, zero/requested `73`, requested/zero `68`, zero/zero `73`; mean unique motions/prompt: `1.6875`, `1.625`, `1.6875`, `1.625`. This diagnostic does not justify expanding zero arms to official MM.
- Valid-token difference versus requested/requested: MGT-zero arms `4650/7296 = 0.6373355263`; VQ-zero with requested MGT `0/7296 = 0`. The latter confirms decoder-only beta routing for token IDs; no MM/FID claim is made.

### 09-11｜matched 20k candidate full544 CPU pairing

- Candidate export session `30914` exited 0; H5 is complete/finite with shape `(544,1,150,672)` and frozen request lengths. Joint CPU session `79740` and physical CPU session `87334` both exited 0.
- Candidate Full/Body/Hands FID `0.422868/0.384182/0.729064`, Matching `3.352120/3.470081/3.328590`, R@1/2/3 Full `0.406250/0.604779/0.720588`, Body `0.406250/0.606618/0.711397`, Hands `0.356618/0.564338/0.681985`; diversity `9.439331/8.863404/8.434019`.
- Matched control CPU: FID `0.558887/0.573045/0.617900`, Matching `3.542590/3.721830/3.373708`, diversity `9.203939/8.727955/8.212367`; physical candidate/control sliding `0.333730/0.325428 m/s`, contact F1 `0.527806/0.539508`. These are quality results only; no MM score or promotion is implied.

### 09-11｜保留 text-dropout MGT 20k checkpoint

- CPU `torch.load` 核验源 checkpoint `global_step=20000`、`epoch=149`；未启动 GPU/export。已将其硬链接保留为 `output/hhc_joint_text_dropout_p0p2_matched_validation_20260910/candidate_step_000020000.ckpt`，与源文件同 inode、3,401,000,246 bytes，SHA256 `ae9867bc90bc79199725fe1a80542e57e0ff23f736996a0ff4a86169669e3822`。
- 仅完成文件保留，尚未对该 20k checkpoint 生成或评分。

### 09-11｜真正 shape-conditioned MGT 零 shape 四臂筛查登记（尚未运行）

- 固定既有选定 MGT `.../model-000260000-0.1246.ckpt`（SHA256 `58b976b8cb579e4a687b9614789bbf088b73e4496eb6b6c06ef7190e80f0d072`）、原 VQ 175k，manifest 明确 `shape=true`；四臂为 requested/requested、zero/requested、requested/zero、zero/zero（分别 MGT/VQ）。
- 固定 gender/text/length、seed600123、50 steps、temperature 0.5、CFG 3.5、`paper_text_cfg`、关闭 contact guidance；第一阶段只冻结 mm_indices 前16条×4重复，保存 token_ids，诊断 token 重复/解码差异，不称官方 MM/FID或在小样本宣称质量不降；有价值后再完整同协议评测。
- 当前仅工程拆分两端 beta policy 与测试，尚未生成；不启动新运行。

### 09-11｜低强度采样导出恢复后完成

- 原session3616由root确认exit0，log_odds_noise0p5_main.h5已导出544/544；既有低成本metrics子agent接续完整性检查、同GPU joint与CPU physical评分，不重复导出。质量/MM结论待评分，尚未启动额外MM生成。

### 09-11｜part thinning 固定解码器诊断（val32）

- `output/hhc_vq_part_thinning_val32_20260911.json`：六臂加权 retained token ratio 均为 `0.8342541436464088`。相对 baseline 的 Full MSE：even/odd body `+1.624%/+1.714%`，hands `+4.368%/+5.593%`，root `+0.587%/+0.575%`。
- 复用原probe新增六臂；root独立运行6项测试exit0、真实CPU诊断session68836 exit0，复核样本/哈希/原baseline相符，并按实际长度重算保留率一致。未修改训练源码。
- 对应被 thinning 部位 MSE：body `+2.462%/+3.408%`，hands `+8.573%/+11.165%`，root `+20.812%/+86.455%`（even/odd）。这是固定 decoder token 干预，没有 joint FID、MPJPE、物理或 MM 实测，不能宣称质量通过或解码速度提高；三流 concat 后联合解码，单流变化可影响其它部位。body-only 可作为后续训练候选；root-only Full MSE 较低但 root 自身退化大，不应因 feature 权重低选它。

### 09-11｜seed123 95k 完整冻结评分完成

- 三评分 session `7228/82620/37279` 均 exit0；root独立核对 checkpoint/H5 实际 SHA、544/71026、H5 finite/completed、names/lengths、evaluator/physical 协议及物理公式重算通过。完整10项对比见审计文档末尾表格。
- 95k 相对原175k滑步/F1更好，但三项 joint FID 更差；95k 是原生验证选优诊断，不是冻结 joint 最优，不替代175k门槛，也不恢复改善训练。

### 09-11｜175k验收边界复核与原日志证据补充

- 三组175k joint JSON均为 complete：Full FID 原→新 `.11086373865→.11707720860`、Hands `.13444848918→.13870270468`，较接近；Body `.31124308622→.38108504414` 仍有差距。因此严格逐项不退化验收未通过，不能宣称复现或恢复改善；完整历史流程仍未认证。
- root核对 Jan 原 `run.log` 的独立 val-loss 记录：5000 行112、175k 行1083、300k 行1775；额外 val-loss 确曾存在。Feb 恢复入口与 July eval 仍不能冒充 Jan 完整源码；seed123 95k 统一评估已安排，结果未齐。

### 09-11｜训练内验证结果更新

- seed123 最新225k训练内 FID `.534559`，历史最佳95k为 `.433196`；seed124最近20k/25k/30k/35k/40k为 `.762986/.720977/.634228/.631321/.545567`，其中40k为当前最佳。
- 以上是训练内验证协议，不能与原Jan `.2619`或冻结joint FID数值混比；主agent已安排用现有三工具CPU重评seed123 95k，仅作原生验证选优诊断，不替代原175k基准、候选175k失败结论，也不恢复改善训练。

### 09-11｜原 VQ-VAE 最佳 checkpoint 基准确认

- 主agent核对原 Jan `run.log` 全部62条 Eva：原协议 FID 最低为 175000 步 `.2619`；其后为 60k `.2929`、50k `.2947`、110k `.3013`、225k `.3055`，300k `.3241`。统一协议重评的 5k/10k/50k/175k/300k 中，175k Full FID `.110863739` 最低，300k `.129022458` 较差。
- 两套 FID 不可数值混比，也不宣称175k每项指标均最佳；175k仍是唯一既定复现基准，300k仅作收敛诊断，不能替换基线。

### 09-11 02:43 BST｜seed124 5k 三评分完成

- 后台日志记录 `completed seed124 step000005000`；主agent读取指标并独立核验 checkpoint/H5 SHA256；子agent核验544样本/71026帧、H5 finite/completed、请求/GT/evaluator/physical 一致性及聚合重算，全部通过。
- 相对原5k，seed124十项指标均改善：RMSE `.09259686006→.09047130811`，MPJPE `.17217093971→.15988216800`，Full/Body/Hands FID `.30399025691/1.59568329074/.14720189047→.25118920621/1.41393658130/.12159951915`，滑步 `.33883610340→.33463698603`，F1 `.67852820764→.68672116439`。
- 产物前缀 `output/hhc_original_vq_reproduction_20260910/candidate_seed124_step000005000`；checkpoint SHA256 `716925f453df3b8fd61a17e2db2d2abf2faed6694177591631d2cb8527bb2f4d`，H5 SHA256 `2a658b06a8e33be7f0e7949780a7a04daa3be406bbcd5b2fcfd180c4b0ccfe81`。相对seed123 5k Full/Body FID略差，不能替代175k验收；seed123 175k失败保留，改善继续暂停，下一预设seed124 50k由原链等待。

### 09-11｜原配置复现早期主训练曲线（三步趋势记录）

- 排除原日志 warm-up 的同名 `Train.Iter 200/400/600/800`：原序列为 `Eva.Iter 0 → warm-up → 第二次 Eva.Iter 0 → 主训练计数重置`。主训练早期 `Recons`（原 / seed123 / seed124）为：1000 `0.16704 / 0.16763 / 0.16580`；2000 `0.13890 / 0.13938 / 0.14025`；3000 `0.11720 / 0.11957 / 0.12042`。
- 这只是早期趋势记录，不能据此验收，也不能将差异归因于 RNG；seed123 的 175k 失败结论保留不变。

### 09-11｜固化175k验收与seed124里程碑自动评分

- 新增check_vq_reproduction_gate.py：固定原175k checkpoint/三评分JSON哈希及请求/GT；核对同step、协议、评分关联H5哈希及物理计数，再逐项检查原有10指标，不加容忍幅度、不自动恢复训练。对seed123现有175k运行exit1：5项通过/5项失败，与人工验收一致；结果candidate_step000175000_gate.json。
- 低成本子agent新增seed124四里程碑串行CPU评分入口，flock防重，文件mtime≥60秒后核对step/seed，固定完整artifacts指纹。主agent发现并修正check-only在就绪时仍评分的问题，动态stub验证check-only仅1次checkpoint检查、实际评分路径4次调用。
- 可重复CPU回归check_vq_reproduction_tools.py已exit0，涵盖指标方向、非有限值拒绝、就绪check-only不评分；未触碰真实checkpoint。已交overnight_metrics替换seed124原监控pane，待部署核验，不增加并行监控。
- 续：同一tmux hhc_vq_seed124_monitor已替换为里程碑等待评分；pane PID3707686、脚本PID3707697由主agent实查存活，pane_dead=0。唯一当前日志/tmp/hhc_vq_seed124_milestone_scores.log（旧事件日志不再更新），每600秒等待，不重复安装或启动评分链。

### 09-11｜只读整理MModality结论边界，保留零shape想法

- 未恢复任何改善训练/生成，也未读取暂停半成品。复核已完成zero_beta评分metadata及MGT条件分支：该模型mgt_shape_conditioning=false，置零shape主要改变VQ解码条件，不能泛化为“有shape输入的MGT上置零也无效”。
- 已将限制及真正shape-conditioned MGT后续需区分采样端/解码端/两端置零的对照写到hhc_mmodality_shape_condition_diagnosis.md顶部，并标明旧live描述仅为历史记录。实际执行仍等VQ复现通过，不跨过用户前置门槛。

### 09-11 01:50 BST｜启动seed124同配置复现，检验训练随机性

- 先登记实验边界与175k固定验收，再启动tools/run_vq_mmm_reproduction_seed124_20260911.sh；session93757，单步预检PASS，实际进入999次warm-up，首次读取已到28次更新，无启动OOM。与seed123同GPU会竞争吞吐，不据墙钟速度比较曲线。
- 主agent比较两份manifest：全部数据/源码artifacts完全相同，resolved_arguments只差seed、exp_name、checkpoint_dir。seed124 contract d84e96686d035cccba451caa1ef7a34d6a02ca42d43117dcfe31e7a71ce65dcf；W&B NewInter/u0v9rucd。run.log在output/vq-interx-sep/2026-09-11-01-50-40_hhc_original_mmm_vq_reproduction_seed124_300k_fp32_20260911_v1/。
- 此实验同时改变初始化/采样/mask随机流，只检验整体随机性，不能单独归因评价时序；不以择优seed宣称跨seed稳定复现。保持FP32/batch256/原loss/300k调度，不动三暂停任务；启动资源复核交低成本子agent。
- 启动资源复核完成：seed124 PID3703449，显存5724MiB，无OOM；seed123 PID3599262仍运行、5726MiB，三暂停进程仍T状态。启动源码/manifest等21个文件已归档output/hhc_original_vq_reproduction_20260910/launch_seed124_sources_20260911.tar.gz；后续至少600秒监控，不继续启动更多seed。
- seed124唯一监控已实查：tmux hhc_vq_seed124_monitor，pane PID3704631、sleep600子进程3704646均存活；事件日志/tmp/hhc_vq_seed124_monitor.log，只记录首次5k/50k/175k/300k及异常。主agent用合成Train.Iter5000检查实际awk解析表达式，输出5000通过；未重复查询训练日志。

### 09-11｜候选300k自动评分入口已审核

- 低成本子agent新增tools/score_vq_candidate300k_20260911.sh，复用既有CPU重构/Interact2AR joint/physical工具；主agent修正初稿的错误joint入口并审核最终固定环境与参数。
- 评分前强制v2 manifest complete、optimizer step300999、固定contract、最终checkpoint路径/SHA256、内部iter300000；flock防重复，任何目标存在即拒绝覆盖。--wait每600秒静默等待；--check-only只核验不评分。
- 主agentbash -n通过；实际--check-only因当前manifest running退出1，安全拒绝评分。已交overnight_metrics接入现有唯一tmux监控pane，尚待接入核验；不改训练/不恢复改善实验。另已询问用户是否有一月原始训练源码备份。
- 续：已接入原tmux hhc_vq_monitor_3599262，pane PID3702963、wait脚本PID3702974；主agent实查二者存活及pane_dead=0。日志/tmp/hhc_vq_candidate300k_wait_score.log记录启动/最终退出码；此链是候选300k唯一等待与评分入口，不再追加并行监控。

### 09-11｜175k逐样本滑步与码本覆盖诊断

- 主agent读取已保存逐样本计数，544条一一对齐且接地分母均>0：候选滑步改善254条、退化290条、相同0条。以每条speed_sum/全体transition_sum分解官方总差，重算完全一致；最大10条仅占全部正向退化贡献约17.2%，并非单个异常动作主导。
- 同协议重构码本统计（各17895 tokens）：body活跃码2741→2617、perplexity1722.32→1658.13；hand3217→3166、2213.41→2189.17；root4379→4402、2769.28→2777.80。body覆盖略低，但不能据此声称码本崩塌或解释FID差距。
- 配对prompt bootstrap已交低成本子agent，仅用于诊断抽样不确定性，不代表训练seed方差、不更改验收门槛；未新跑模型。
- 续：子agent统计完成，主agent用排序name、seed1234、10000次同索引配对抽样独立重算官方聚合指标。新-原滑步95%区间[0.00000752,0.01135435]m/s，F1区间[-0.01226627,0.02931272]；滑步下界极近0，不夸大显著性，F1改善亦不能认作稳定优势。逐样本滑步差中位数+0.00453665m/s。


### 09-11｜原300k三项参考评分完成：较原175k多项退化

- 子agent原始会话：重构16506、joint80947、physical41054均exit0。主agent已核验三协议、544条H5完整性、逐样本物理计数求和及重算指标，均通过。
- 原175k→300k Full/Body/Hands FID：0.110864/0.311243/0.134448→0.129022/0.342154/0.139323；滑步0.343131→0.344308m/s；接触F1 0.680395→0.663814，均退化。结合重构结果，不能把300k视为天然更强参考。
- 175k既定门槛不变，也不使用较弱原300k让候选175k“过关”。300k参考已齐备，当前候选到终点后只做同一步数的收敛诊断；自己的改善仍暂停。

### 09-11｜原300k重构参考已核验，未见全面优于原175k

- 原300k reconstruction JSON/H5已生成；主agent核验请求/GT/FK/表示协议一致、544名称与长度一致、completed_repeats全部完成、(544,1,150,672)全部有限。首次核验脚本误用completed字段报错，核实真实schema后以completed_repeats重跑通过。
- 原175k→原300k：RMSE 0.089197529→0.090266077；MPJPE 0.157910202→0.157893861m；body 0.121704519→0.121066490；hand 0.187272159→0.187796351；root 0.081053401→0.080370275。更晚检查点并非全面改善；不能仅等训练更久就假定复现差距会消失。
- joint CPU评分进程PID3699369已实际启动并存活，后续物理评分由同一子agent接续；尚未宣称三项完成。监控同一tmux已修正为读取最近Train.Iter，退出时重查完成步数，仍每600秒检查。

### 09-11｜安排最终收敛参考评分，并修复失效监控

- 已交低成本token_compression_statistics执行原MMM 300k检查点的544请求CPU重构/joint/physical评分；目前为已分配待结果，不宣称完成。仍使用既有请求/GT/评价器及valid_frames floor协议；300k用于后续收敛诊断，不替换已失败的175k门槛。
- 主agent核验原net_300000.pth存在，SHA256 680e8bca961f28d859683c1fea97162b4b6f38d0dcee9d5b0e74f94d16d140ac；原日志确有300000训练与评价记录，旧日志FID不能直接当新协议FID。
- overnight_metrics实查训练PID3599262存活，最新177800；但后台monitor3695436已退出，无完成事件。已要求查因并用持久会话修复唯一600秒监控，不能以旧启动PID冒充持续监控。未重启训练。
- 续：新监控已迁入tmux hhc_vq_monitor_3599262，主agent实查pane PID3699145、pane_dead=0；事件文件/tmp/hhc_vq_monitor_3599262_tmux.log。子agent实查训练推进到181600，并非停滞。旧nohup消失的具体系统原因未证实，仅确认监控已失效。

### 09-11｜真实训练首批的随机流影响已测量

- 复用check_vq_pretrain_rng.py增加--real-data；session90266 exit0。真实raw train数据、batch256，在同一seed123诊断起点，只增加一次未隔离验证loader遍历（17批），首批256个片段与基线无任何相同内容；恢复torch RNG则整批逐字节一致。
- 只覆盖当前loader、0 worker和受控随机数起点，不是重建一月真实首批，也不说明新首批质量更差。已足以确认不能仅凭相同seed认为两次训练轨迹相同。
- 后续受控复训应显式冻结首批/后续采样及初始化随机流，区分轨迹差异与代码差异；不能靠继续增加同输入parity测试证明质量复现。原训练未改动。

### 09-11｜实际训练配置与build_model路径一致性通过

- 扩展现有parity工具--training-manifest，读取当前候选resolved_arguments，HHC侧实际调用build_model(interx_vq)，MMM侧直接构造参考类；shape_mask_prob=0.1、raw、FP32及架构参数来自运行清单，不再仅测试手写模型参数。
- 运行22步，session85966 exit0，全部逐元素一致，reset_steps=[21]；对应contract c0386308489c7e3d5c2e8726c72f7509340c339a95582c7872f426b8ba2ce7bb。排除了当前构建路径遗漏参数这一测试缺口。
- 仍是CPU固定小批次、固定LR的计算路径测试，未覆盖完整训练随机流/调度或一月源码。175k验收仍失败，不恢复改善训练。

### 09-11｜GPU数值环境核对：未发现已记录的主要版本差异

- 子agent提取、主agent复核两次W&B：torch2.5.1/numpy2.2.6/scipy1.16.3/Python3.12.12及RTX4090一致；Linux内核6.14.0-36→-37。入口及相关models/utils/dataset/options未见显式TF32/matmul_precision/cuDNN benchmark覆盖。
- 当前训练/proc环境未见NVIDIA_TF32_OVERRIDE、TORCH_ALLOW_TF32_CUBLAS_OVERRIDE、CUBLAS_WORKSPACE_CONFIG。新解释器默认matmul TF32=false、cuDNN TF32=true、benchmark=false、deterministic=false；这不是对运行中进程内部状态或一月设置的认证。
- 本次没有支持“版本/显式TF32开关变更导致差距”的证据；也不能将FP32称作全算子禁用TF32。未修改环境、依赖或训练。

### 09-11｜跨EMA重置周期的22步一致性通过

- 复用tools/check_vq_mmm_fp32_parity.py，新增--steps；运行--steps 22，session73694 exit0。四个真实64帧片段、CPU FP32，同输入/连续同RNG，当前MMM与HHC每步output、7项loss、梯度、参数、AdamW、EMA、tokens、RNG均逐元素一致。
- 实测第21步三个量化器reset_count归零，覆盖原单步测试未经过的周期重置分支。测试复用固定小批次、固定LR，不覆盖完整DataLoader/评价时序或一月源码；175k质量失败结论不变。
- 按ponytail只扩展现有诊断脚本，未改共享模型/训练代码，未新增长训练；下一排查重点仍是历史流程与随机流，而非盲改loss/EMA。

### 09-11｜复现差距分解到八项训练loss

- 低成本子agent提取、主agent独立awk复核165k–174.8k的50个共同点：Recons +3.60%、Commit +3.39%、Pos +3.76%、Vel +2.47%、Root +2.04%、Rot +2.27%、FC +1.34%、DistanceMap +9.93%（绝对差仅0.000009）。140k窗口亦全部略高，未见单项loss独有异常。
- 核查git量化器历史：EMA/reset核心规则未变；只发现零codebook的初始设备迁移变化。不能据此认证未提交的一月源码，也不能将质量差距直接归因随机数。
- 分项表已写入VQ_MMM_REPRODUCTION_AUDIT.md。下一步应做首批/量化轨迹的受控诊断，而非凭百分比改loss；原训练和暂停状态不变。

### 09-11｜训练前评价的随机数机制检查通过

- 新增并运行tools/check_vq_pretrain_rng.py，CPU exit0：实际QuantizeEMAReset在eval模式下不初始化/更新codebook；但未隔离的验证DataLoader迭代改变训练首批及首次codebook，保存/恢复CPU RNG则完全一致。仅为合成机制验证，不能归因实际FID差距。
- 原一月run.log确有warm-up前Eva.Iter 0；二月恢复入口在warm-up前/后均评价。主agent复核MMM提交a01d5ac^的Inter-X评价函数入口net.eval、出口net.train，未隔离RNG；该源码晚于一月，不能冒充一月原件。2024提交51669d20仅单人版，不能混用其日期认证Inter-X函数。
- 未更改训练或恢复自己的改善实验。后台唯一监控PID3695436，每600秒检查，事件日志/tmp/hhc_vq_monitor_3599262.log；主线程不重复监控。

### 09-11｜固化低频监控规则

- 已将至少10分钟一次、低成本subagent接管、仅重要事件汇报的规则写入AGENTS.md；继续交由overnight_metrics负责，要求复用或建立唯一非阻塞后台监控，避免模型逐分钟醒来。
- 不改变训练日志与验证频率，不重启175k评分；175k复现验收未通过的结论不变。

### 09-11｜175k失败后的FK调用链诊断

- 已检查单步测试、恢复的历史训练入口及当前MMM兼容导出：loss复用net.fk_layers，并非另建一份FK；当前vqvae_sep导出同一VQVAE_SEP3_shape类，但不能据此认证一月源码。
- 低成本子agent只读比较原/新175k保存的全部FK张量：唯一key fk_layers.unified_positions_lookup.weight，shape(118,165)，完全相等、最大差0。保存的骨架偏移不同这一假设未获支持；不撤销非原地FK修复。
- 下一步重点检查单步测试未覆盖的训练前评价、随机数消耗及codebook初始化时序；未改运行中的训练，未恢复改善实验。

### 09-11｜175k固定验收未通过

- 原等待/重构session6102、joint48126、physical15075均exit0。主agent独立核验原/新协议一致、544条H5名称/长度/完成标记及全部motion有限，逐样本重算滑步和接触F1一致。
- 原→新：RMSE 0.089198→0.090967；Full/Body/Hands FID 0.110864/0.311243/0.134448→0.117077/0.381085/0.138703；滑步0.343131→0.348822m/s，以上未通过。
- MPJPE 0.157910→0.154307m、body/hand MPJPE及root误差改善；接触F1 0.680395→0.688774改善，但不能抵消其他验收项失败。因此不能宣称原VQ已复现，自己的改善实验继续暂停。
- 后续继续定位复现差异，不放宽既定门槛。原训练保留300k调度，175k评分已完成，不重启等待链。

### 09-10｜150k前曲线对比：重建损失仍高约3.67%

- 低成本子agent提取、主agent用awk独立复核140000–149800区间的50个共同日志点：原Recons均值0.0344608000，新0.0357258000（+3.67%）。曲线尚未重合，不能宣称复现成功。
- 保持175k固定同协议验收，不改训练或放宽标准；自己的改进实验继续暂停。沿用监控session6102，不新增评分链。

### 09-10 21:11｜100k检查点状态审计通过（非质量验收）

- 低成本子agent实际加载100k检查点：iter及scheduler.last_epoch均100000；137个AdamW参数状态step全部100999，符合独立999次warm-up；quantizer_training_state的6个code_sum/code_count张量全部FP32且有限。
- 主agent确认文件464774615 bytes，所在磁盘剩余105G；目前无需清理。此检查仅说明保存状态符合调度/精度约定，不证明重建质量复现。
- 继续原175k等待链与固定验收，未增加模型评分或恢复自己的改进实验。

### 09-10｜补核对数据采样实现（只读诊断）

- 当前HHC/MMM数据类完整diff仅归一化显式选择及空行；片段索引、8-worker随机打乱、drop_last及cycle相同。raw训练分支不执行vel随机交换，未发现新增裁剪/采样分布差异。
- 共同的窗口边界索引行为保留，不夹带修正；验证消耗随机数的既知差异仍披露，不能仅凭这些检查解释约3%的曲线差距。详见复现审计，未改训练代码。

### 09-10 20:27｜补查80k曲线：重建损失差距仍在

- 主agent用awk独立按main step对齐80000–85800的30个共同日志点：Recons均值原0.0402600000、新0.0414933333（约+3.06%）；较30k–35.8k的+4.53%缩小，但没有重合，不能替代175k同协议质量验收。
- 子agent19:25 UTC实查训练PID3599262为Rsl、87200步，175k等待session6102存活；本次读日志已到87400步。未改训练配置或启动额外模型评分，自己的改进仍暂停。

### 09-10 18:26｜50k同协议评分完成：身体FID追平，手部仍落后

- reconstruction/joint/physical会话65412/53196/88065均exit0；主agent核验同一请求/GT/评价器协议、544条H5完整名称长度及全部有限值，并独立重算滑步/F1。
- 原→新50k：RMSE 0.085591→0.087284；MPJPE 0.148543→0.150089m；Root误差0.075900→0.077265m。
- Full/Body/Hands FID：0.124576/0.342572/0.109796→0.129850/0.336539/0.147476。身体FID略好，但手部FID约差34.32%；滑步0.340069→0.343096m/s，接触F1 0.693009→0.692161。
- 结论：不能验收复现。训练实际推进至51k，保持原配置；下一固定主验收175k由低成本子agent接管等待及同协议评分，自己的改进仍暂停。
- 175k唯一等待/重建会话为`6102`（overnight_metrics持有）；已核验主训练存活且无重复175k评分链。后续沿用此会话，稳定写入并确认iter=175000后评分，不重复启动。

### 09-10 17:42｜降低监控频率，交给低成本子agent

- 按用户要求，主线程停止逐分钟轮询；由既有低成本 overnight_metrics 接管训练及50k评估等待链，每10分钟最多检查一次。
- 仅在50k评分完成、训练异常或进程退出时汇报；沿用现有进程，不重复启动训练/评估。复现验收前仍不恢复改进实验。

### 09-10 17:38｜应用户要求对齐原版训练曲线：尚未复现

- 对齐原2026-01-30运行与本次训练的main step。5k/10k/20k/30k/35k的Recons（原→新）：0.09774→0.09983、0.07540→0.07670、0.05996→0.06044、0.05373→0.05453、0.05009→0.05299。
- 30k–35.8k共同30个日志窗口均值：Recons原0.051297、新0.053619（+4.53%）；Commit +4.26%、Pos +6.74%、Vel +3.87%、Root +7.40%、FC +3.76%。趋势相似但近期持续偏高，不能说训练曲线已重合；这些是日志均值，不是独立重复实验置信区间。
- 旧日志没有总Loss，PPL字段也不同，不直接混比；历史native FID协议不同，仍只用已冻结的同协议评估验收。10k Body FID仍退化约13.36%，因此尚未通过复现。原配置继续至50k趋势评估和175k主验收，自己的改进保持暂停。

### 09-10 17:33｜35k保存正常，native监测回落

- 35k权重完整加载通过：35999次AdamW更新、35000次cosine更新、有限FP32 EMA。
- 固定native验证FID从30k的0.678419回落至35k的0.640761，仍略高于25k的0.633170；不据此宣称官方质量复现。
- 原配置训练与50k等待链继续，未恢复改进实验。

### 09-10 17:16｜30k保存正常，native验证小幅回升

- 30k权重完整加载通过：30999次AdamW更新、30000次cosine更新、有限FP32 EMA。
- 固定native验证FID由25k的0.633170升至30k的0.678419，不能只报告此前下降；目前仅是一次监测回升，不据此修改训练配置。
- 继续既定训练与50k冻结评估等待链，不把native分数当作官方复现验收。

### 09-10 17:00｜25k检查点及训练内验证正常

- 25k权重完整加载通过：25999次AdamW更新、25000次cosine更新、有限FP32 EMA。
- 固定native验证FID从20k的0.835139降至25k的0.633170；训练重建损失0.05662。仍只作同协议训练内趋势，不替代冻结官方评估。
- 原配置训练继续，50k等待链仍存活；未恢复改进，未改变175k主验收门槛。

### 09-10 16:43｜20k保存及训练内验证正常

- 20k权重已完整加载核验：20999次AdamW更新、20000次cosine更新、EMA状态有限FP32。
- 固定native验证FID从15k的1.217301降至20k的0.835139，重建训练损失0.06044。仅为同一训练内监测趋势，不混同官方joint评估或175k复现验收。
- 训练继续，50k评估等待链保留，未改变训练配置或恢复改进。

### 09-10 16:27｜15k保存与训练内验证正常

- 15k checkpoint可完整加载，实查15999次AdamW更新、15000次cosine更新，三个EMA状态均有限FP32。
- 固定训练内验证FID从10k的1.365958降至15k的1.217301；这是native监测，不与官方joint FID混比，也不是最终复现验收。
- 训练继续，50k评分等待PID3607755仍存活；未新增改进实验或改变配置。

### 09-10 16:17｜原50k同协议对照全部完成

- reconstruction/joint/physical分别session74264/39177/63401，均exit0。主agent验证完整输出、同评价协议及逐样本物理聚合。
- 原50k RMSE0.085591、MPJPE0.148543m、Root误差0.075900m；Full/Body/Hands FID0.124576/0.342572/0.109796；滑步0.340069m/s，F1 0.693009。
- 新50k等待/重建链session65412、PID3607755已建立，尚未就绪；主agent直接验证该PID存活（子session在主agent不可见，不视为退出、不重启）。主训练仍存活，未改变配置。三个改进进程仍为暂停T状态，磁盘剩余113GB。

### 09-10 16:13｜新10k完整对照：接近原运行，但仍未全部达到

- reconstruction/joint/physical分别session74651/58011/16768，均exit0。主agent验证544条输出完成/有限/名称长度一致，评价器/GT/请求/seed相同；物理计数重新聚合通过。
- checkpoint确认10999次AdamW更新、10000次cosine更新、有限FP32 EMA，训练继续运行，未改配置。
- 新/原10k：RMSE0.087188/0.086489，MPJPE0.153295/0.151605m，Root误差0.084430/0.085247m。
- Full/Body/Hands FID新0.241902/1.177011/0.125569，原0.243602/1.038323/0.129246；Full/Hands略好、Body较差。滑步新0.325560 vs 原0.322576m/s；F1新0.685206 vs 原0.692086。
- 结论：5k手部FID差异未持续，但其他指标仍有差距，不能验收；继续既定训练到后续50k趋势检查与175k主验收，不恢复改进、不放宽门槛。

### 09-10 16:00｜原10k完整对照收齐

- 原10k reconstruction/joint/physical分别session21660/6325/22181，均exit0。joint Full/Body/Hands FID=0.243602/1.038323/0.129246；主agent核验与原5k完全相同的评价器、GT、请求和seed。
- 新10k等待/重建链session74651已启动；训练仍为原PID3599262，主阶段6800步以上。后续沿用原句柄，不重复启动。

### 09-10 15:59｜原10k对照部分评分核验

- 原10k重建文件的544条名称/长度、完成标记和有限值已检查；RMSE0.086489、MPJPE0.151605m、Body/Hands MPJPE0.120671/0.176812m、Root误差0.085247m。
- 从逐样本计数独立复算：滑步0.322576m/s、接触F1 0.692086；joint评分待收齐。
- 新训练已到6600步，PID3599262存活。原10k对照不是新模型结果，尚不能作新10k验收。

### 09-10 15:57｜核对原训练曲线并安排10k配对评估

- 主训练6000步仍正常。原日志/新日志重建损失：2000步0.13890/0.13938，5000步0.09774/0.09983，6000步0.09270/0.09261；commit及各项物理loss量级也接近。说明当前训练趋势接近原运行，但不替代质量验收。
- 已安排原10k CPU重评，再等待新10k稳定权重做同协议重建/joint/physical对照。保留5k手部FID略差的事实，不变更配置或放宽门槛。

### 09-10 15:56｜5k完整评分收齐：整体接近，但手部FID略差

- reconstruction/joint/physical分别session17199/31962/8357，均exit0。主agent核对joint评价器provenance、GT、请求、seed完全相同。
- 新5k Full/Body/Hands FID=0.250770/1.353980/0.150673；原5k=0.303990/1.595683/0.147202。Full/Body改善，但Hands增加0.003471，不能说全部指标复现。
- 当前结论：重建及物理点估计改善、手部FID仍有差异；继续原配置训练与后续里程碑，不调整验收标准，不恢复自己的改进。

### 09-10 15:54｜复现候选5k检查点与首批评估通过完整性核验

- 训练保存5k并继续主阶段；checkpoint实查iter5000、scheduler5000、AdamW累计5999更新，符合独立999 warm-up；三个量化器的EMA sum/count均FP32且有限。
- 新5k相对原5k：RMSE 0.091186 vs 0.092597；MPJPE 0.162998 vs 0.172171m；Body/Hands MPJPE 0.131117/0.189178 vs 0.139119/0.199315m；Root误差0.094189 vs 0.101477m。
- 滑步0.335303 vs 0.338836m/s，接触F1 0.682552 vs 0.678528；是早期点估计改善，不代表最终复现或显著性结论。
- 主agent已验证同GT/请求/FK协议、544条输出完成标记/名称/长度/有限数值，并从逐样本计数复算F1。joint FID仍待收齐；不能以5k代替175k主验收。

### 09-10 15:39｜原模型完整对照评分收齐

- 原175k joint CPU评分完成（session54173 exit0），Full/Body/Hands FID=0.110864/0.311243/0.134448；主agent核验评价器来源完全一致、全部指标与此前同协议重评逐值相等。
- 原5k/175k的重建、joint、physical共六项评分均正常完成；physical只代表关节级滑步/接触，不含mesh penetration。
- 新复现训练主阶段已超过400步，原PID3599262仍在运行。安排等待新5k完整checkpoint后做同协议CPU评分，未到里程碑前不读取半写入权重、不重启、不恢复改进。

### 09-10 15:38｜复现训练完成 warm-up，进入主阶段

- session41269/主PID3599262仍在运行；完成999次独立warm-up，主阶段超过200步。主200步Loss0.27473、Recons0.20283，日志各项有限；约5步/秒，175k主验收预计约10小时训练时间，另加评估开销。
- warm-up重建损失200/400/600/800步为0.37024/0.33159/0.26486/0.24155，原日志为0.37099/0.32838/0.26684/0.24272；趋势接近，但不是复现验收。
- 原5k/175k重建H5已核验完成标记、544条名称/长度、71026帧及有限数值。物理指标已逐样本重新聚合核验：原5k滑步0.338836m/s、F1 0.678528；原175k滑步0.343131m/s、F1 0.680395。该工具不测mesh penetration，不将接触指标称为penetration。
- 原5k同协议joint FID Full/Body/Hands=0.303990/1.595683/0.147202；原175k完整joint评分继续收尾。下一里程碑为新5k同协议对照，不能据短期loss恢复改进实验。

### 09-10 15:34｜正式复现候选进入训练

- 入口 `tools/run_vq_mmm_reproduction_20260910.sh`；FP32、原12288码本/stride4、batch256、seed123、wd2e-5、独立999 warm-up+300k cosine，无EMA/压缩改进。已到warm-up200，Loss0.71994，未出现OOM。
- 明确限制：一月完整源码/随机轨迹不可认证；当前val监测隔离随机状态，不复刻后期源码test监测。目标是用实际同协议重建证明质量复现，不能把开跑当作验收。
- 主验收固定为新175k对原175k，同544请求/71026帧，原始特征误差、各部位MPJPE、joint FID及物理质量不能退化；详见审计文档，禁止事后换有利checkpoint/评价协议。
- 原5k/175k CPU重建已完成并核验哈希与样本数：5k RMSE0.092597/MPJPE0.172171m；175k RMSE0.089198/MPJPE0.157910m。joint/physical评分继续补齐。
- 启动源码已归档 `output/hhc_original_vq_reproduction_20260910/launch_sources_20260910_v1.tar.gz`，SHA256 `482090bd8e0720557d618b0405d0bd1c76df9030e2be6fdd79ca6d7afa627ce1`。此前三个改进进程仍暂停。

### 09-10｜整模型 FP32 单步数值对照通过

- 新增可重复运行的 `tools/check_vq_mmm_fp32_parity.py`，session44721 正常退出（exit0）。
- 原架构、相同初始化及4段真实64帧输入：当前MMM/HHC输出、七项loss、全部参数梯度、AdamW更新、EMA、token与随机状态均逐元素一致。
- `output_emb_width` 256/512 在该类中不改变初始化或单步结果；不能把这个表面配置差异认作训练退化原因。
- 仍未通过原VQ训练复现验收。下一步核对训练/验证的随机状态与调度流程，固定验收协议后从头复现；自己的改进进程继续暂停。

### 09-10 15:24｜形成原 VQ 复现差异表

- 审计表已写入 `docs/VQ_MMM_REPRODUCTION_AUDIT.md`，区分已确认设置、实测一致项和来源限制。
- 另一份 IDE 历史库哈希与已恢复的冻结库一致，没有新增一月源码；不重复恢复。
- 原运行 seed123、weight decay2e-5、300k horizon 与此前 pilot 不同；当前模型/数据入口还需模型级对照，不能仅改训练参数就宣布修复。
- 当前 encoder/decoder 与 quantizer 源文件、raw 均值/方差已核对一致；完整模型与训练流程仍未验收。

### 09-10｜原 VQ 来源与 loss 数值核验

- 4 段真实验证动作、64 帧、CPU FP32：当前 MMM 与 HHC 的七项 loss 及预测张量梯度完全相等，均有限；使用相同绝对骨架的独立 FK，避免旧 FK 原地修改污染比较。这不是一月源码或整套训练的复现证明。
- 复用历史恢复工具，在临时副本上恢复出 `train_vq.py`（session `27957`，exit0），原始历史库未修改。
- 恢复的是 2026-02-26 修改前的内容，含 `vel` 数据入口及后期实验名称，尚不能当作一月 raw-12288 成功运行源码；已保留来源和哈希，继续追溯，不据此启动训练。
- 恢复文件：`output/historical_mmm_original_vq_train_recovery_20260910/`。现有三个改进进程仍暂停。

### 09-10｜按用户要求纠正前置验收顺序

- 暂停三个改进进程，保存已有状态与结果；不把此前 stride4/8 10k 对照当作原 MMM 复现。
- 重新检查已有历史审计：它已经记录原成功运行的独立 warm-up、300k cosine、weight decay 等差异，不能忽略后直接改进。
- 当前已知：本轮 pilot 的 integrated 10k schedule、weight decay 1e-6 不等于历史成功设置；loss/数据/模型一致性仍需重新逐项验证，不能仅凭旧审计宣布一致。
- 已找到原成功运行：MMM `output/vq-sep/2026-01-30-22-34-14_vqvae_shape_sep3_256/`，包含训练日志和每5k原始权重；对应本地 W&B 配置可读，明确 seed123、batch256、window64、code12288、总迭代300k、weight decay2e-5。
- 重要来源风险：W&B 记录的 Git commit `8c3e7ad` 实际仍是旧单人训练代码，不能冒充当时未提交的双人训练源码；后来的 `75814e5` 也不是一月源码快照。正在从现有归档/Local History 追溯。
- 验收顺序：原始来源与差异表 → 同输入模型/loss/梯度/更新核验 → 从头训练并与原 checkpoint 在同一重建协议下比较。三个阶段均不能跳过，目前仍在来源与差异审计。

### 09-10 15:14｜补齐采样对照条件

- 发现已有 log-odds 强度0与原始排序的 FID 略不同，不能视为同一控制。
- 安排强度0.5导出完成后，用相同GPU0/batch32同时评分原始控制、log-odds强度0和新候选；旧评分保留。
- 导出 PID `3590738` 已确认运行，评分需等进程结束且 H5 完成标记通过后才开始；暂不做 MM。

### 09-10 15:12｜补做较弱顺序扰动采样

- 此前强度 1 提高 MM，但额外种子下有质量取舍；本次仅将 log-odds 扰动降至 0.5。
- 使用原 300k MGT/175k VQ、544 条冻结请求、seed500123、50 步、T0.5、CFG3.5；严格加载已通过，原 VQ 保留。
- 这是新强度的筛选，不是已改善 MM 的结论；先查质量，再决定是否花计算量测 MM。

### 09-10 15:10｜运行状态补充复核

- 两块 RTX 4090 均在工作：GPU0 为 FP32 EMA 单因素 VQ，GPU1 为联合文本 dropout MGT；原 session 句柄对应进程均存活。
- 联合文本 dropout 已保存 15k 检查点；目前只有检查点，尚无 15k 冻结官方评分。

### 09-10 15:10｜stride8 10k 部分评分验收

- 544 条输出的名称、长度、完成标记及有限数值检查通过。
- stride8 5k → 10k：MPJPE 0.1679 → 0.1733 m，滑步 0.3157 → 0.3034 m/s，接触 F1 0.6729 → 0.6698，仍是质量取舍。
- 同为 10k 时，重建误差、滑步和接触 F1 优于发生明显退化的 stride4，但不能据此认定压缩改善整体质量；继续验证 EMA 稳定性。
- 最终 joint FID 已收齐：Full/Body/Hands 为 0.5131/1.5161/0.1783，三者均比自身 5k 变差。没有通过质量门槛，暂不升级 stride8 MGT。
- EMA 对照已到 600 步，日志损失有限；仍需等待 5k/10k 实际评估。

### 09-10 15:08｜训练交接完成

- stride8 10k 训练正常完成，EMA 对照已接手 GPU0 开始更新，不再是排队状态。
- 对照 manifest 已核验：除实验名称/目录及 EMA 干预标识、哈希外，所有记录的训练参数和数据/共享源码哈希均与 stride4 基线一致。
- 等待实际评估结果，尚不能认定 FP32 累积改善生成质量。

### 09-10 15:03｜建立简短工作记录

- 按用户要求建立本文件，便于 tmux 查看；后续记录做了什么、结果和下一步。

### 09-10｜stride8 5k 配对评估完成

- 用与 stride4 相同的 544 条冻结样本评估，已核验样本名称、长度、完整性和有限数值。
- token 减少 49.6%；Full/Body FID 变差，Hands FID 略好，文本匹配距离均略差。
- 滑步从 0.3275 降至 0.3157 m/s；接触 F1 从 0.6816 降至 0.6729。配对 bootstrap 支持滑步减少，但不能证明接触质量不降。
- 决策：暂不启动 stride8 MGT，等待 10k 与训练稳定性对照。

### 09-10｜定位 VQ 数值问题，准备单因素对照

- stride4 的训练和评估均在后期退化，不只是长序列或验证集问题。
- GPU 检测发现 bf16 距离计算会改变最近邻 token；但仅修改推理精度没有恢复重建。
- 固定输入实验发现 bf16 EMA 累积可使码本漂移；实际检查点的累积量也确为 bf16。
- 已实现独立 FP32 累积实验入口，保留原始 token 选择和聚合方式；单元检查及实际 VQ 前向/反向检查通过。未修改运行中的共享源码。

### 09-10｜联合文本 dropout 10k 评估完成

- Body FID 改善，但 Full/Hands FID、滑步变差，当前检查点未通过质量门槛；继续看训练趋势。
- 原生验证里的 MM 与冻结官方指标协议不同，不混用。

## 详细证据位置

- 诊断记录：`docs/hhc_mmodality_shape_condition_diagnosis.md`
- VQ 配对结果：`output/hhc_vq_stride_compression_paired10k_20260910/`
- 文本 dropout 结果：`output/hhc_joint_text_dropout_p0p2_matched_validation_20260910/`
- EMA 实验入口：`tools/train_vq_fp32_ema_accumulator.py`

### 09-11｜root replay full-quality CPU 验收完成

- `output/hhc_root_replay_validation_20260911/root_replay_log_odds_noise1_main.h5`：544/544 complete、finite；首32条 root replay token IDs 与 archived parity 控制完全相等。导出进程已结束，但原始 exporter exec exit code/walltime 未回收，不能伪造为 exit0。
- CPU joint session `50374` exit0；physical session `2327` exit0。候选 FID Full/Body/Hands=`0.2309235324/0.2363433037/0.2115106628`，Matching=`3.0380189699/3.0771992768/3.0815968373`，Full R@1/2/3=`0.4319852941/0.6433823529/0.7591911765`，diversity=`9.6455364227`。
- 对照 FID=`0.2409670257/0.2343818610/0.2108525635`、Matching=`3.0286780806/3.0814206881/3.0843989008`；Full FID改善，Body/Hands FID与Full Matching略变差。physical 候选 sliding=`0.3040152326` m/s、contact F1=`0.5582317226`，对照=`0.3034704498`、`0.5575800105`；无 MM，不晋级。
- 配对 prompt bootstrap（10000 draws，seed90210，诊断 JSON `output/hhc_root_replay_validation_20260911/root_replay_paired_prompt_bootstrap.json`）给 candidate-control sliding `+0.0005447827` m/s，95% CI `[-0.0001088478,+0.0012573575]`；contact F1 `+0.0006517121`，95% CI `[-0.0021792958,+0.0036187368]`。两者均跨0，不构成非劣性证明。

2026-09-11：SSP3D male 缺失补跑准备完成；核对 all62 为 male 41 个 mesh、已有 fitted male 9 个，缺失 32 个。新增 `output/ssp3d_official_transfer_all62_20260911/meshes_missing_male`（32 个指向原 male mesh 的符号链接）及独立配置 `configs/male_missing_retry1.yaml`，输出仍指向原 fitted/male；未启动 GPU、未覆盖已有 fit。现有 fitted 共 30 个 pkl，结构检查目标为 betas `(1,10)`、vertices `(1,10475,3)` 且有限。

2026-09-11：新增并完成 GPU1 诊断 `tools/diagnose_shape_alignment_gap.py`（session 62558，exit0）。对 25 对输出逐 sidecar 校验 SHA256，使用全部帧计算 raw/恢复地面对齐后的穿透均值；结果 raw `0.0281479852`、aligned `0.0067915517`、gap 均值 `-0.0213564334`、最大绝对 gap `0.1349539720`。结果写入 `output/ssp3d_probe25_alignment_gap_20260911.json`。
