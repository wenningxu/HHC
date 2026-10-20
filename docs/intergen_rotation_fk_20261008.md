# InterGen：root旋转扩维与旋转主导FK训练

## 目标与状态

**10-10 19:01 BST scratch正式训练已启动：** 用户授权自主推进后，删除旧9-02 run的10k/20k两个非最佳早期权重共7886157011B，保留后期/best/latest及指标，约7.4GiB可用空间。run=`output/intergen_rotation_scratch_20261010`，trainer PID220949，已到step50、loss有限；无官方InterGen checkpoint读取（config checkpoint/hash=null）。tmux=`intergen_scratch_hourly_20261010`，watchdog PID220948，每小时自动检查/崩溃恢复。704000步/lr1e-4/bs32，每10000步完整val；step0与每500步持久保存recovery.pt（权重+optimizer/RNG同step原子替换），不再依赖RAM恢复包。新评测尚未产生，MM/test仍待后续冻结验收。系统goal接口仍有旧paused unfinished目标且不能由工具resume；本机训练守护已独立运行，不将系统goal伪报active。

**10-10 新优先级：从头训练。** 已加入`--scratch`，跳过所有InterGen checkpoint初始化；仅CLIP冻结预训练保留。官方随机初始化的输出层为0，为避免6D/FK退化，局部/root旋转bias设置成归一化identity，其余遵循原随机初始化。用`--checkpoint /does-not-exist`成功完成两步训练与DDIM50导出（主96421 exit0），证明scratch入口不读取原生成checkpoint。scratch默认lr1e-4、704000步、micro4×acc8、约10epoch warmup、每10000步完整val；沿用现有旋转/FK目标，不声称严格原论文训练复现。正式训练尚未启动：/home432MB、HDD55MB，持久checkpoint所需空间不足，等待空间/清理授权；旧微调结果全部保留。小时守护需在正式新run启动时按新预算配置，旧10000步守护已完成，不能声称仍在监控新实验。

**10-10 13:19 BST完成复核：** 小时守护10-09 19:08检测183854在step3300再次SIGILL/exit -4并自动恢复185132，21:08确认10000步正常exit0、守护结束。完整480val FID按step0/2000/4000/6000/8000/10000依次为9.681466/9.412841/9.481942/8.943854/9.202316/9.181300。最佳step6000较step0下降7.62%；Matching3.804716，R1/2/3=.429167/.570833/.656250，Div8.006828（GT7.741262）。FID有改善但R2/R3下降，不能称全面提升。MM与独立test未评；下一步冻结best6000验证。

**10-09 恢复与小时守护：** journal定位原SIGILL指令位于NVIDIA libcuda.so.570.195.03，非已证实的loss错误；未确定更深层根因。模型/optimizer都为2000且全有限，67200 exit0；恢复实装检查43621 exit0，已加载权重、optimizer和RNG。`tools/train_intergen_rotation.py --resume`保留原config/normalization，严格核对目标步数/批量/lr/loss/data/basehash，训练加文件锁；checkpoint将同step权重与optimizer一起原子备份到RAM。非bitwise恢复，原dataloader游标未保存。

后台tmux `intergen_rotation_hourly_20261009`执行`tools/watch_intergen_rotation.py`，立即恢复训练，每小时检查一次，退出则带独立日志重新执行resume；达到10000并正常complete后结束。状态与最新完整val指标进入`output/intergen_rotation_fk_20261008/watchdog.jsonl`，PID文件同目录。不终止/重置别人的GPU工作。自动化不覆盖主机重启或驱动修复；RAM中的base/recovery重启会丢失。/home约880MB余量，保存696MB权重的临时文件空间较紧，未擅自清理其它run。

**10-09 18:05 BST复核：未完成，已异常退出。** 最后训练日志step2300，shell报Illegal instruction/core dumped，exit_code132、PID128670不存在；原session81692已Unknown。STATUS.json的training是遗留状态，不代表进程存活。best/latest存在，恢复完整性未检查，未重启。

完整480val（seed20261006）step0→step2000：FID9.681466→9.412841，Matching3.804834→3.803573，R1 .412500→.437500，R2 .585417→.587500，R3 .672917→.662500，Div7.991972→8.060261（GT7.741262）。最佳step2000；FID改善2.77%，但R3下降、Div偏离增大。MM和独立test待评。

2026-10-08用户授权：修改InterGen训练loss，旋转为核心，监督旋转FK后的位置；加入对应原始root旋转并同步变换，评测时去掉新增通道。目标仍是评测指标改善，不以一致性代替验收。

正式流程主session **81692**；启动脚本`tools/run_intergen_rotation_fk_20261008.sh`。日志`output/intergen_rotation_fk_20261008/run.log`，终态`exit_code`。Run：`/home/wenning/HHC_interhuman_runs/intergen_rotation_fk_seed20261008`。先train-only root统计→step0完整val→10000步微调，每2000步480val。启动不等于已完成训练/效果验证。

## 实现

- 不覆盖baseline源码/权重；官方commit `b1750df4a859443b8a69329f759ce3a7503ae179`复制至`third_party/intergen_base_20261008`，工作树保持干净。复用原InterGen网络、1000步cosine diffusion、x0 prediction、10% CFG条件dropout及DDIM50/CFG3.5采样，wrapper替换训练目标。
- 每人原262维不变，末尾新增6维root旋转，双人536维。只扩展共享motion_embed输入列及out.linear输出行/bias；旧权重保留，新增输入列/输出权重为0，新增bias为归一化identity。并非从零训练或声称复现原训练流程。
- 原始SMPL root_orient按原frame_rate→30fps的floor时间索引重采样。镜像为M R M；世界坐标转换及anchor朝向以Q_anchor T左乘root。双方最终位于同一anchor坐标；root随crop/anchor/output person swap同步。局部关节旋转沿用官方值，不错误左乘全局变换。
- FK骨架使用对应训练样本真实betas的female SMPL rest joints；镜像时offset同步左右换位并反射。根位置使用预测位置通道中的pelvis，不使用GT root位置做预测。GT betas/offset只作训练loss中的骨架信息，不作为生成条件；评测不读取测试GT shape/root。
- 评测按新268维mean/std反归一化，然后只去掉末尾root6，保留原262位置/速度/局部旋转/contact。原位置通道仍是预测head，并非用GT shape重建后替换；有较弱aux监督。

## Loss与训练设置

`tools/intergen_rotation_objective.py`：归一化局部6D MSE×1、root6D MSE×1、局部/root旋转矩阵MSE各×.5、FK位置m² MSE×4、pelvis归一化MSE×1；直接位置/速度/contact归一化MSE各×.1。FK速度×30、FK足接触速度×30、FK跨人距离×3、近接吸引×3。几何项只在diffusion t≤700启用；时间差分仅有效相邻帧，所有loss排除padding。骨长由FK骨架决定，不再额外惩罚独立位置head骨长。接触使用GT contact监督，距离mask用GT距离；不是原loss的逐项等价复刻。

训练入口`tools/train_intergen_rotation.py`。AdamW、wd2e-5、lr1e-5/100-step warmup/cosine至0.1倍，microbatch4×accumulation8，有效batch32；bf16网络forward、fp32 FK/loss、clip grad .5。初始官方checkpoint SHA `bab341123c27af9d8c3ee7ebc43c2d341bfd0f4103425d06f8af6c717a086f8d`。

保持原262 stats以兼容预训练权重；新增root6统计遍历全部有效train IDs，每ID一个固定seed20261008的≤300帧crop、交替mirror/anchor，std下限.05。完整ID列表/计数保存normalization.json；没有使用val/test统计。

每500步保存latest可训练权重（冻结CLIP从指定base恢复），best按480val FID选取，同时报告Matching/R1/2/3/Div；不声称只看FID就是全面改善。optimizer/RNG滚动备份在`/dev/shm/intergen_rotation_fk_seed20261008_optimizer.pt`，不是持久磁盘备份，未保存dataloader游标，不声称bitwise恢复。base原始checkpoint也仍在/dev/shm，恢复需该hash文件；source已持久化。MM/test后续待评。

## 已完成检查

- `PYTHONPATH=. .../New_MMM/bin/python dataset/interhuman_root_rotation.py`：主70415 exit0；8个train动作×8镜像/anchor/person组合，GT FK vs官方位置最大9.862965e-7m；局部/root旋转梯度均非零且有限。
- `PYTHONPATH=. .../New_MMM/bin/python tools/intergen_rotation_objective.py`：主66620 exit0；FK loss不经独立位置head绕过旋转，padding数值变化不影响loss、padding梯度0。
- microbatch2两步smoke40735 exit0；microbatch4两步+DDIM50采样smoke84441 exit0，导出形状[1,32,2,262]且有限；烟雾测试不是训练效果。
- Python编译/shell语法检查通过。评分入口`tools/score_intergen_rotation_cache.py`核对全部480冻结request、官方encoder/hash与输入维数，复用现有完整非MM指标评分脚本。
