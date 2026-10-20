# 最新 SMPL VQ → Inter-X 风格无手 MGT → 自动评测

用户授权在出行期间自动完成训练与评测。当前任务不重启旧Inter-X探索、不重训VQ、不覆盖旧524维InterHuman实验。

## 固定合同

- VQ：`vq_interhuman_smpl_raw_interx_handless_seed123_20261004/best_fid.pth`，step185000，加载真实mean/std，冻结全部权重/码本；构造MGT前后逐tensor验证未被初始化覆盖。
- 以`2026-09-02-11-49-08_hhc_historical_plus_paper_mask_ratio_raw12288_300k_seed114514_v2`为原Inter-X有shape参照，不误用512维paper模型或旧InterHuman无shape模型。
- MGT：embed1024、16heads、2个文本cross-attention+7个shape AdaBlock，head9层，FF×4、dropout.1；删除hand，仅body96/root64，两个12288码本，存储body/root、生成Root→Body。shape/gender条件按原48→1024→1024 MLP。
- shape原beta20维和female计算模板传入MGT与VQ解码；global CLIP与shape训练drop各.2。生成切`paper_text_cfg`，只去文本、保留shape。无接触指导。
- 训练seed114514、有效/物理batch64、BF16、AdamW2e-4/(.5,.9)/wd1e-6，150k×.05，300k updates，clip norm1。原historical_multistep源码没有实际warmup，不能把保留args中的warm_up_iter1000当作执行证据。
- 直接复用排序cosine kinematic mask + pkeep.5随机替换，以及all-valid NLL（不改masked-only）。原single/single/double静态cache裁剪、每次随机caption、概率1/3删首尾1 token保留；4567条train动作/13540种文本，min30/max999帧，只有train。
- 数据必要差异：时间容量38→76以容纳InterHuman300帧；最长75动作token+END，仍为4帧/token。固定val-only选模和隔离RNG取代旧历史test侧效应。

## 评测

- 每10k：已有冻结480条val + 100×11 MM诊断，50轮、T.5、CFG3.5，原始confidence排序（noise0），仅最低val FID选model。
- 训练后自动严格重载best；已有正式test main20×1056、MM20×100×30（10pairs）；逐rep存指标、embedding及request/model/evaluator/source hash，再汇总均值±95% CI。
- 重建原SMPL→官方body192→官方预处理262×2，经过eval标准化一次后交给原InterCLIP。原始生成需要多一帧供速度，token长度为ceil((requested_processed_length+1)/4)，转换后裁到精确有效长度，不用零填充冒充有效尾帧。
- 只读取请求ID的betas作为shape条件，不用GT pose/trajectory/contact修补生成。评测的anchor、左右镜像、person order遵循冻结请求，shape仍对应原person身份。GT动作只供评分。
- **这是shape-conditioned实验**，用了请求的真实betas；与无shape InterMask不能作严格同条件因果比较。指标是生成InterCLIP FID/检索/MM，不是VQ重构joint FID。
- 流式评分避免约47.5GiB动作H5；保留约0.25GiB embedding供复算，生成轨迹可用固定种子重生成。MM小批排序先恢复repeat顺序再用原整prompt等长排序，避免batch拆分改变评分输入顺序。

## 自动运行和资源

入口`train/train_mgt_interhuman_smpl.py`，一个后台进程完成训练→选模→test/MM→`test/REPORT.md`，不需要用户回来另下命令。`STATUS.md`记录中文当前阶段，`metrics.jsonl`记录实测，`training_manifest.json`含PID、状态、最佳点及错误。

输出仅latest完整optimizer续训态和best模型，两份均原子保存，不积累每10k大checkpoint；冻结VQ不重复塞进MGT ckpt。输出放/home原训练目录（约11GiB可用），未删除任何历史文件。保存前检查空间；异常退出保留最近有效checkpoint并标failed。支持显式`--resume`，恢复sampler步号、optimizer/scheduler/RNG与W&B run；不自动把重复失败冒充继续成功。训练及验证源码启动后不原地修改。

冒烟：主65750 exit0，285709616可训练参数，batch64两步有限loss/梯度，峰值allocated15.25GiB；完整480val/真实50轮采样+2prompt×11 MM通过（step2 FID101.9985只验管线，不是质量结论）。独立CPU tiny架构测试exit0，覆盖VQ冻结、loss反传、CFG保留shape、固定种子采样、END/PAD。

正式启动状态以`docs/PROGRESS.md`顶部为准；最终报告只有在test/summary.json status=complete且20个rep都齐全时才算完成。

正式进程已于10-05 08:47 BST完成首2步更新，主session50085，W&B `djxe8qmw`。运行目录`/home/wenning/HHC_interhuman_runs/mgt_interhuman_smpl_interx_shape_seed114514_20261005`；中文实时`STATUS.md`和最后`test/REPORT.md`均由后台自动更新。命令：

```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 /home/wenning/miniconda3/envs/New_MMM/bin/python -u train/train_mgt_interhuman_smpl.py --output /home/wenning/HHC_interhuman_runs/mgt_interhuman_smpl_interx_shape_seed114514_20261005
```
