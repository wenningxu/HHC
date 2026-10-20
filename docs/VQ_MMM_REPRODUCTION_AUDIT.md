# 原 MMM VQ 复现前置审计

状态：用户已接受当前 VQ 质量“基本可以复现”，解除必须等待严格10项逐项不退化的探索阻断；Full/Hands较接近、Body仍有差距，严格门槛未通过与完整历史流程未认证的证据保留。
后续探索回到 MModality 质量保持，采样/条件训练/token压缩/零shape在 scope；不声称 bitwise 或十项全优，暂停进程是否恢复待 root 操作确认。
用户要求先实际复现，再恢复自己的改进；不能用代码检查或加载旧权重代替训练复现。

## 参考运行

- 原运行：`/home/wenning/projects/MMM/output/vq-sep/2026-01-30-22-34-14_vqvae_shape_sep3_256/`。
- 配置：`/home/wenning/projects/MMM/wandb/run-20260130_223414-vqvae_shape_sep3_256/files/config.yaml`；同运行 `run.log` 可交叉核对。
- 原权重：每 5k 检查点可用，目标权重 `net_175000.pth`；HHC 导入文件 SHA256 为 `fdf5afaa5d3513f933254fafc831f6046f7c38b5e649db08fb6be7e0ce1006bb`，需在最终验收重新核验。

## 已确认的配置差异

| 项目 | 原成功运行记录 | 本轮已暂停 pilot |
| --- | --- | --- |
| seed | 123 | 114514 |
| codebook / latent | 12288 / 256 | 相同 |
| batch / window | 256 / 64 | 相同 |
| width / depth / stride | 512 / 3 / 4 | stride4 相同；stride8 为改进实验 |
| weight decay | 2e-5 | 1e-6 |
| 主训练 horizon | 300000 | 10000 |
| warm-up | 日志有 0→800→0 的独立阶段；后期恢复源码明确 999 次独立更新 | 1000 次包含在总 10000 内 |
| precision | 恢复的后期训练入口无 autocast；一月配置未单独记录此项，不能仅凭缺失字段宣称已证明 | bf16 |

配置中的 `output_emb_width=256` 与当前入口强制512也不同；必须检查实际构造函数是否使用该参数，而非直接假设影响架构。

## 已执行的实现核验

- 可运行整模型检查：`/home/wenning/miniconda3/envs/New_MMM/bin/python tools/check_vq_mmm_fp32_parity.py`，session44721 exit0，PASS。
- CPU FP32、原架构12288/256、seed123初始化、4段真实64帧输入、seed90210训练随机状态：输出、七项loss、全部参数梯度、AdamW单步后的参数/优化器状态、EMA状态、编码token和CPU RNG均逐元素相等。两侧分别传入output_emb_width256/512，证明该参数差异在此模型构造中不改变结果。
- 测试抽取当前MMM的HHI类，使用独立参考FK/损失；encoder、resnet、quantizer、骨架配置和旋转工具逐文件一致检查，相关HHI masks逐元素检查。原MMM整套数据/训练/评价流程不在此测试范围；尤其不是一月源码认证。
- 4 段真实验证动作、64 帧、seed90210，加固定0.03高斯扰动作为预测输入，CPU FP32。
- 当前 MMM 与 HHC 的 `ReConsLossInter` 七项结果完全相等；对预测张量的总梯度逐元素相等且有限。
- 两侧分别使用当前 MMM/HHC FK 实现，传入独立克隆的同一绝对骨架，避免旧构造函数原地修改污染另一侧。
- 上述四段扰动输入的loss检查本身不覆盖整模型；后续整模型检查已覆盖当前参考的全部参数梯度及单步更新，但两者均不能证明一月源码一致或训练质量复现。
- 当前模型文件差异含 shape-condition 参数化、shape-mask 参数化、输出张量分配dtype及 stride 标记；原配置下的模型级同输入/初始化/更新测试已通过，不能外推到其他配置。
- 当前数据类差异：HHC 显式按 `motion_rep` 选择归一化，MMM 当前类调用默认归一化。后期默认值不能用于推断一月 raw 训练。
- 补查当前两侧完整`dataset_interx.py`差异：除上述归一化选择及文件末尾空行外一致。训练片段按累计`L-window_size`索引枚举，不是getitem内随机裁剪；这包含两侧共同的边界索引行为，本次不顺手修正。两侧DataLoader默认8 workers、shuffle=True、drop_last=True，cycle行为相同。raw分支不会进入vel随机交换，正常train.txt也不会进入`kkkk`条件分支。因此未发现当前两侧新增的裁剪/采样分布实现差异，但仍不能认证一月数据类版本。

## 源码来源限制

- W&B 的 Git commit `8c3e7ad` 是2025年旧提交，里面仍是单人训练源码，不是当时未提交的双人实现。
- `75814e5` 是2026年7月版本，不能冒充一月快照。
- 一月运行的 `source/` 为空，不能据目录存在声称源码已保存。
- 已用现有 Java 工具从历史库临时副本恢复 `train_vq.py`，session27957 exit0。恢复文件位于 `output/historical_mmm_original_vq_train_recovery_20260910/`，SHA256 `dfcf7c39ba2c8b5439ccc19bb9cb792689ff18e44700f4636036f4fe55845008`。
- 内容是2026-02-26修改前的版本，包含后期 `vel` 入口/实验名称，不能认证为一月版本。当前 IDE 历史库与冻结库的 changes.storageData 哈希相同，没有新增更早记录。

## 后续验收门槛

1. 完成数据、模型、loss、优化器、EMA、调度和验证流程差异表；所有来源不确定项明示，不能悄悄使用当前默认值。
2. 修复复现配置/实现，补可运行的同输入模型输出、loss、梯度及单步更新检查；恢复原设置，不夹带改进。
3. 实际从头训练，对照原运行已有中间检查点；最终与原175k在相同样本、归一化、FK、评价器、设备和裁剪协议下重评。
4. 复现表现验收通过前，不能将新 tokenizer 接入 MGT，也不能恢复压缩/EMA/采样改进。数值容差与质量判据须在正式重训结果出来前明确，不能事后放宽。

暂停进程与简短状态见 `docs/PROGRESS.md`。

## 首次正式复现候选（结果出来前固定）

- 入口 `tools/run_vq_mmm_reproduction_20260910.sh`：原记录架构/seed123/AdamW2e-4、wd2e-5/独立999 warm-up+300k cosine，原量化器，无EMA/压缩等改进，FP32。FP32依据后期恢复源码，是明示的来源推断而非一月精度元数据认证。
- 已知流程差异保留并披露：当前使用val而非后期恢复入口的test做监测；验证随机状态隔离，无原入口两次iter0验证及额外随机val-loss遍历。因此目标是原VQ质量的经验复现，不宣称bitwise训练轨迹复现。数据/模型初始化的完整历史随机轨迹无法从现存一月记录确定。
- 正式训练以原175k为预先固定的主验收里程碑，300k为原学习率horizon，不能改成175k cosine。5k/10k/50k只用于检查趋势，不以短期通过替代验收，也不事后挑选有利checkpoint。
- 统一冻结请求：`output/interx_official_val_requests/validation.npz`（SHA256 `ade004f1e586a6c94afa41c1dd15defc90d1493c09ae5b9b968f2fecb1e99175`），544条/71026帧；GT `output/interx_validation_raw_ground_truth.h5`（SHA256 `ea8ed90af73293f6487bda51fb96b1ca41ad307a93d45ae47aa36a02b82e9804`）。原175k与新175k均以同一CPU FP32/batch16重建、raw归一化、修正后的非原地FK评估。joint评价使用同一冻结CPU评价器，禁止混用历史native FID。
- 主门槛：新175k的raw RMSE、全身/身体/手MPJPE、root误差及Full/Body/Hands joint FID均不高于同协议原175k；物理滑步不升、接触F1不降。只允许重复同一确定性评估所测得的数值抖动，不预留任意5%宽容带。任何一项不满足，都不能自动宣布验收，须记录差距并继续诊断；显著性/误差区间作为补充，不用于掩盖点估计退化。
- 启动训练仅代表复现候选进入实测，不代表已修复全部来源不确定项或通过用户门槛。原5k/175k同协议重建已安排单独CPU评分。

## 已重评的原始对照

同一544条冻结请求，所有任务exit0；主agent重新检查H5名称/长度/完成标记/有限值，独立聚合物理逐样本计数。

| 原checkpoint | raw RMSE | 全身MPJPE(m) | Body/Hands MPJPE(m) | Root误差(m) | Full/Body/Hands joint FID | 滑步(m/s) | 接触F1 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 5k | 0.0925968601 | 0.1721709397 | 0.1391192141 / 0.1993153815 | 0.1014773226 | 0.3039902569 / 1.5956832907 / 0.1472018905 | 0.3388361034 | 0.6785282076 |
| 175k | 0.0891975290 | 0.1579102017 | 0.1217045187 / 0.1872721594 | 0.0810534007 | 0.1108637386 / 0.3112430862 / 0.1344484892 | 0.3431309910 | 0.6803952211 |

结果位于 `output/hhc_original_vq_reproduction_20260910/`。原175k joint评价器provenance及全部metrics与已有corrected-FK CPU重评完全相等；确认协议稳定，不是训练候选通过证明。

## 候选5k结果（非最终验收）

- 三评分全部exit0，544请求完整性及相同评价协议已复核。checkpoint明确5999次AdamW更新、5000次cosine更新，EMA sum/count为有限FP32。
- raw RMSE0.0911864096，全身/Body/Hands MPJPE0.1629984079/0.1311168902/0.1891775391m，Root误差0.0941893721m，均优于原5k点估计。
- 滑步0.3353025079m/s、接触F1 0.6825515760，均略优于原5k。
- Full/Body/Hands joint FID0.2507699819/1.3539798629/0.1506732790；前两者改善，Hands比原5k差0.0034713886。没有通过“各项均不退化”，也不能替代新175k对原175k的主验收。保持配置继续训练，不事后改评价标准。

## 候选10k结果（非最终验收）

- 全部三评分exit0，输出及同协议核验通过；checkpoint确认为10999次AdamW/10000次cosine更新，EMA状态有限FP32。
- 新/原10k：raw RMSE0.0871884734/0.0864885692，MPJPE0.1532946950/0.1516048931m；Root误差0.0844295209/0.0852465001m。
- Full/Body/Hands FID：新0.2419023551/1.1770105591/0.1255694403，原0.2436022877/1.0383233897/0.1292461062。手部不再劣于同阶段参考，但Body仍较差。
- 滑步新0.3255601266 vs 原0.3225756897m/s；接触F1新0.6852062913 vs 原0.6920861538。没有满足所有指标不退化，不能宣布复现。继续固定配置，后续50k趋势及175k主验收，不事后选择有利指标。

## 候选50k结果（非最终验收）

- 三项评分及主agent完整性/同协议核验已完成，详见进度记录09-10 18:26；不是新增训练或重新评分。
- 原→新：RMSE 0.08559058539→0.08728407359，MPJPE 0.14854335768→0.15008931432m，Root误差0.07589974274→0.07726489372m。
- Full/Body/Hands FID：0.12457567944/0.34257155102/0.10979626973→0.12985013369/0.33653853789/0.14747629437。Body略好，但Hands退化约34.32%。
- 滑步0.34006941999→0.34309554124m/s，接触F1 0.69300866073→0.69216108512。尚未验收；保留175k主门槛及既定协议，不恢复改善训练。

## 候选175k主验收：失败

- 三评分已完成，544条完整性及同协议核验通过；详见PROGRESS.md“175k固定验收未通过”。原→新：RMSE 0.089197529→0.090966670，Full/Body/Hands FID 0.110863739/0.311243086/0.134448489→0.117077209/0.381085044/0.138702705，滑步0.343130991→0.348821678m/s，均未通过。
- MPJPE与接触F1改善不能抵消失败项。原300k调度仍保留用于诊断，不能用更晚checkpoint回填175k验收。自己的改善实验不恢复。

## 训练前评价时序诊断（09-11）

- 原一月日志记录warm-up前Eva.Iter 0。二月恢复训练入口174/245行分别在warm-up前/后评价，而当前复现入口没有这两次非隔离评价。
- MMM git a01d5accca54676694402b962de60ef1a23d54a7^:utils/eval_trans.py第24/232行在评价前后eval/train切换，验证DataLoader迭代未隔离RNG。此为七月删除前版本，不认证一月源码；51669d20的2024版本是单人函数，不能混用出处。
- tools/check_vq_pretrain_rng.py使用真实QuantizeEMAReset及合成CPU数据：评价不会初始化codebook，但DataLoader迭代消耗RNG，可改变首批/codebook；隔离RNG则首批与codebook完全一致。运行exit0。
- 这排查了直接“评价初始化codebook”的机制，并证明随机流差异是需要控制的变量；没有证明它导致实际质量差距。仍需历史证据或受控训练才能作因果结论，不能以此宣布复现成功。

## 训练loss分项对比（09-11）

低成本子agent提取两个窗口；主agent用独立awk复核165000–174800的全部数值。每窗口50个共同main训练点，排除warm-up/validation。以下为日志中已加权的分项，Commit尚需乘0.02才计入总loss；不比较定义不同的PPL。

| 分项 | 165k–174.8k 原均值 | 新均值 | 相对差 |
|---|---:|---:|---:|
| Recons | 0.0329482 | 0.0341340 | +3.60% |
| Commit | 1.4442430 | 1.4931364 | +3.39% |
| Pos | 0.0026176 | 0.0027160 | +3.76% |
| Vel | 0.0028198 | 0.0028894 | +2.47% |
| Root | 0.0003334 | 0.0003402 | +2.04% |
| Rot | 0.0006254 | 0.0006396 | +2.27% |
| FC | 0.0004622 | 0.0004684 | +1.34% |
| DistanceMap | 0.0000906 | 0.0000996 | +9.93% |

- 140k–149.8k也为八项全部略高：Recons +3.67%、Commit +3.44%、Pos +3.80%、Vel +2.49%、Root +1.79%、Rot +1.97%、FC +1.02%、DistanceMap +3.59%。这些是相关训练窗口，不是独立seed置信区间。
- 差距非单一loss尖峰；DistanceMap百分比最大但绝对差仅0.000009，且日志只保留5位小数，不应凭其百分比优先改距离loss。尚不能区分随机轨迹与未恢复的历史实现差异。
- MMM git 2fd1e10→75814e5中QuantizeEMAReset只将codebook零张量构造从立即.cuda()改为随模型迁移，EMA/reset核心规则未变；当前文件无工作树改动。此证据不涵盖未提交过的一月历史版本。

## 跨重置周期多步一致性（09-11）

- 命令：`/home/wenning/miniconda3/envs/New_MMM/bin/python tools/check_vq_mmm_fp32_parity.py --steps 22`，session73694 exit0。
- 原有CPU、4个真实64帧片段、12288code架构不变；固定小批次连续22次AdamW更新，每步清梯度并让两侧共享该步RNG起点，下一步从一致的结束RNG继续。
- 每步逐元素比较output/losses/gradients/state/ema/tokens/optimizer/rng，22步全部PASS；实际reset_steps=[21]，覆盖周期重置分支。
- 参考模型SHA256仍为8e85bfd0d63d2d819e98eaf8630c32da5be61480c5592b64f91917477d840fba，共享量化器SHA256仍为8e580c1c4ab84e9d0f470f3926a2b90bca0ed7449a58de7236328883e18380e7。
- 边界：当前MMM源码、同输入/同RNG、CPU与固定LR；不证明一月源码、GPU全批次、完整采样/评价流水线或最终质量复现。不能据此恢复自己的改善训练。

## GPU运行环境对照（09-11）

- 原W&B run-20260130_223414-vqvae_shape_sep3_256及新run-20260910_153311-e3yghsxx的requirements/metadata：torch2.5.1、numpy2.2.6、scipy1.16.3、Python3.12.12、RTX4090均相同；Linux内核6.14.0-36→-37，glibc2.41不变。
- 训练入口、恢复入口、启动脚本及相关models/utils/dataset/options搜索未发现显式TF32、matmul精度或cuDNN benchmark/deterministic设置。只读取当前训练PID3599262环境中相关白名单键，未见NVIDIA_TF32_OVERRIDE/TORCH_ALLOW_TF32_CUBLAS_OVERRIDE/CUBLAS_WORKSPACE_CONFIG；OMP/MKL均4线程。
- 新解释器实测torch CUDA build12.1、cuDNN90100、matmul_precision=highest、matmul.allow_tf32=false、cudnn.allow_tf32=true、benchmark=false、deterministic=false。新解释器不等于live训练内存状态，历史运行也未保存这些开关；不能声称两次GPU计算严格确定或全算子TF32禁用。
- 未找到主要版本或显式精度覆盖差异证据，未更改任何运行中设置。

## 实际manifest与模型构建路径（09-11）

- parity工具新增--training-manifest；读取当前候选training_manifest.json的resolved_arguments（仅device改CPU），HHC侧用生产build_model(args,'interx_vq')，参考侧用同架构参数直接实例化当前MMM类。
- `python tools/check_vq_mmm_fp32_parity.py --steps 22 --training-manifest checkpoints/hhc_original_mmm_vq_reproduction_seed123_300k_fp32_20260910_v1/training_manifest.json`：session85966 exit0，22步所有比较PASS，reset_steps=[21]。
- 对应清单contract c0386308489c7e3d5c2e8726c72f7509340c339a95582c7872f426b8ba2ce7bb；此调用确认shape_mask_prob=0.1与模型构建参数未导致计算分歧。固定小批次、固定LR/AdamW配置、CPU测试边界不变，不认证整条历史训练流水线。

## 真实数据首批随机流检查（09-11）

- `python tools/check_vq_pretrain_rng.py --real-data`，session90266 exit0。当前raw train数据含9002 motions/1044551 snippets，batch256；验证集实际遍历17批。只遍历数据，不执行训练/模型评价。
- 同一诊断seed123起点，无预评价/未隔离验证遍历/隔离验证遍历三组首批motion SHA256分别为a3c867738dac36778a6566502c39e67c573ee71414e7db83b52b5dff621b543f、32fbcc2fab8a1625bbed5a205cc8b2ba18491221be5c1eba8bfbd228d462304c、a3c867738dac36778a6566502c39e67c573ee71414e7db83b52b5dff621b543f。前两组相同位置片段0/256、共同唯一片段0；第三组与第一组完全一致。
- 边界：0 worker、当前验证split与人工固定RNG起点，不重建历史test loader、模型初始化之后的真实RNG或一月首批；不由此判断哪一组收敛更好。后续因果实验应控制实际训练输入/随机轨迹，不能将更多相同输入parity测试替代质量复现。

## 原300k最终检查点参考（09-11，非替换175k门槛）

- 重构session16506、joint80947、physical41054均exit0。主agent独立检查同请求/GT/FK/评价器/长度/物理协议，544条H5名称长度完成标记及全部motion有限，逐样本物理计数求和、滑步/F1重算通过。
- checkpoint SHA256 680e8bca961f28d859683c1fea97162b4b6f38d0dcee9d5b0e74f94d16d140ac；H5 SHA256（评分报告）260bfe7abdcb443d1b559ae4c0dee1114d2bfa4e52e3de1d25c2784fa640cc0a。文件前缀output/hhc_original_vq_reproduction_20260910/original_step000300000。
- RMSE0.09026607653；MPJPE/body/hand/root为0.15789386083/0.12106649037/0.18779635146/0.08037027460m。
- Full/Body/Hands FID0.12902245848/0.34215433008/0.13932309112；滑步0.34430770893m/s，接触F1 0.66381396297。三FID、滑步、F1较原175k均退化；训练更久不保证验证改善。
- 此结果仅供当前候选最终300k同step收敛诊断，不更换175k基线、不跨step制造验收通过，不解除用户要求的复现前置门槛。

## 175k逐样本物理差异与码本覆盖（09-11）

- 544条name严格配对，全部两侧接地分母>0；滑步改善254、退化290、相同0，中位差+0.00453664592m/s。按每条speed_sum/对应全体transition_sum分解总差，重算与官方值一致；最大10条贡献0.00729506162，全部正向贡献0.04231926769（约17.2%），并非孤立一个异常样本。
- 主agent独立prompt配对bootstrap：排序name，np.default_rng(1234)，10000次，各次两侧使用相同544个有放回索引，分别聚合speed/transition及TP/FP/FN后求指标，不平均逐样本F1。滑步新-原95%CI=[0.00000751957,0.01135435209]m/s；F1 CI=[-0.01226626593,0.02931271831]。
- 上述仅为固定模型的prompt重采样诊断，未建模同主体/动作相关性，不是训练seed方差；滑步下界极近0，不作强显著性声明，也不修改验收门槛。子agent以原记录顺序抽样得到略不同Monte Carlo端点，记录采用主agent明确排序的版本。
- 同协议、每part17895 tokens：body活跃码2741→2617、perplexity1722.31715→1658.13222；hand3217→3166、2213.41047→2189.16520；root4379→4402、2769.28142→2777.80487。body覆盖略低，但不能据此认定码本崩塌、因果解释FID或恢复改进训练。

## 启动前登记：seed124同配置独立复现（09-11）

- 目的：确定当前约3%的训练曲线差距是否随训练随机性改变。只将seed123改为124及独立输出路径；模型、raw数据、loss、FP32、batch256、999 warm-up、300k余弦调度、held-out评价均不变。10个manifest训练源码哈希启动前逐一验证未变；不声称恢复一月轨迹。
- 这是原配置复现实验，不是自己的模型/采样改进。seed同时改变初始化、采样及mask随机流，不能单独归因训练前评价；两seed也不足以精准估计训练方差。
- 固定主验收：seed124的175k对原175k，沿用既有544请求/GT/评分与各项不退化门槛；seed123结果继续完整披露，不以择优seed宣称跨seed稳定复现。5k、50k仅为早期趋势，300k仍是终点诊断，不跨step换基线。
- 使用GPU0，启动前空闲约10220MiB，原配置训练占5726MiB；不终止/恢复暂停任务，不降低batch或改精度。与seed123并行会竞争吞吐，不以墙钟速度比较收敛；若OOM则停止新候选排队，不修改旧训练。
- 启动入口tools/run_vq_mmm_reproduction_seed124_20260911.sh；此登记写入时尚未启动，不宣称训练完成或通过验收。

### seed124启动与归档核验

- 实际启动session93757，PID3703449，W&B NewInter/u0v9rucd；已进入warm-up，启动资源核验5724MiB、无OOM；seed123和三暂停任务状态保持。
- 主agent核验两manifest全部artifacts相同，resolved_arguments差集恰为seed/exp_name/checkpoint_dir；seed124 contract d84e96686d035cccba451caa1ef7a34d6a02ca42d43117dcfe31e7a71ce65dcf。
- 归档21个入口/模型/数据实现/损失/审计/manifest文件：output/hhc_original_vq_reproduction_20260910/launch_seed124_sources_20260911.tar.gz，SHA256 9e07f3e08c8b8207f54866e7f522eb3ce565ac6a5edb1505db874755790634cb。归档不含数据实体，数据指纹见manifest，避免重复拷贝。

## seed124 5k 早期结果（09-11 02:43 BST）

后台评分链记录 `completed seed124 step000005000`；主agent读取指标并独立核验 checkpoint/H5 SHA256；子agent核验544样本/71026帧、H5 完整且 finite、请求/GT/evaluator/physical 一致及聚合重算均通过。相对原5k，十项指标均改善（RMSE `.09259686006→.09047130811`；MPJPE `.17217093971→.15988216800`；Full/Body/Hands FID `.30399025691/1.59568329074/.14720189047→.25118920621/1.41393658130/.12159951915`；滑步 `.33883610340→.33463698603`；F1 `.67852820764→.68672116439`）。

产物前缀为 `output/hhc_original_vq_reproduction_20260910/candidate_seed124_step000005000`；checkpoint/H5 SHA256 分别为 `716925f453df3b8fd61a17e2db2d2abf2faed6694177591631d2cb8527bb2f4d` / `2a658b06a8e33be7f0e7949780a7a04daa3be406bbcd5b2fcfd180c4b0ccfe81`。相对 seed123 5k 的 Full/Body FID 略差，不能宣称全面优于 seed123 或替代175k门槛；seed123 175k失败保留，改善继续暂停，下一预设 seed124 50k 由原链等待。

## seed123 95k 与 175k 完整十指标对比（09-11）

三评分 session `7228/82620/37279` 均 exit0；checkpoint SHA256 为 `868e3cd1940162f143591449c01a3a939c015a825f454f2f4df1b0a0b3794886`，H5 SHA256 为 `ce4083fb0da2fb32382949302357ff03fdda6287c65430966c9e3de96550f022`。统一冻结请求/GT、CPU FP32、raw 表示与 Interact2Ar joint evaluator 协议；MPJPE/root 转为 mm，滑步为 m/s，RMSE 为原始特征单位。

| 指标（step / seed / 协议） | 原175k / 123 / 冻结完整评分 | 新175k / 123 / 冻结完整评分 | 新95k / 123 / 冻结完整评分 |
| --- | ---: | ---: | ---: |
| RMSE | 0.0891975290 | 0.0909666696 | 0.0887637484 |
| 全身 MPJPE (mm) | 157.9102 | 154.3072 | 151.4977 |
| 身体 MPJPE (mm) | 121.7045 | 118.9985 | 116.0535 |
| 手部 MPJPE (mm) | 187.2722 | 183.0369 | 180.3671 |
| root 误差 (mm) | 81.0534 | 79.1811 | 75.7644 |
| Full joint FID | 0.1108637386 | 0.1170772086 | 0.1529466178 |
| Body joint FID | 0.3112430862 | 0.3810850441 | 0.3901429466 |
| Hands joint FID | 0.1344484892 | 0.1387027047 | 0.1376559190 |
| 滑步 (m/s) | 0.3431310 | 0.3488217 | 0.3405370 |
| 接触 F1 | 0.6803952 | 0.6887736 | 0.6914253 |

95k 的滑步/F1优于原175k，但三项 joint FID 更差；这是原生验证选优诊断，不是冻结 joint 最优，不能替代175k逐项门槛或恢复改善。Jan 日志直接证实两次 iter0 评价及周期额外 val-loss（5000 行112、175k 行1083、300k 行1775）；Feb 入口还显示 test split、非隔离 RNG、两验证 loader，而当前为 val/0-worker/shuffleFalse/固定 RNG且无额外 loss 遍历。恢复这些流程不能冒充 Jan 完整源码已认证；原配置为 raw，Feb 恢复入口硬编码 vel，不能盲抄 vel 用于原 raw 复现。
