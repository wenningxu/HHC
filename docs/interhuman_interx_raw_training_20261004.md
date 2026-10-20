# InterHuman：严格 Inter-X raw 无手版 VQ

用户要求开始训练，严格沿用原Inter-X格式，仅去掉手。

## 相对上一版数据视图的纠正

原`train/train_vq_interx.py`强制`motion_rep=raw`，不是vel。训练现在不使用归档里root速度槽；输入[T,23,12]，前22行是两人的6D旋转（矩阵前两行），末行为`[p1 XYZ,0,0,0,p2 XYZ,0,0,0]`。模型只编码其中264维body和6维root，输出也将6个占位槽补零。速度监督来自FK后的差分，位置/速度不再自定义混用。

原生SMPL仍保留，shape真实参与FK和decoder条件，不转换SMPL-X。为了沿用原loss的Y轴高度/XZ地面语义，对全局root旋转及位置统一作`(x,y,z)→(x,z,-y)`，两人用同一变换。root位置是骨盆关节位置，不是SMPL平移参数；通过真实beta计算的rest pelvis可逆换回原translation。局部body旋转保持不变。不分别居中或逐人/逐帧落地。归档H5和原数据均不覆盖，训练视图内存构建，无额外大数据复制。

## 模型/训练合同

- 原Encoder/Decoder、QuantizeEMAReset直接复用；不是旧524维InterHuman模型。
- 原code_dim256分配body96/hand96/root64；**仅删除hand96**，保留body96/root64，decoder条件8维，总输入168维。不将手部容量补给body。
- 两码本各12288，shape projection20→40、gender2×4、condition MLP48→64→8、shape mask0.1，保留原结构。gender=1是匹配官方的female计算模板，不把raw neutral标签改成女性身份。
- width512/depth3/down_t2/stride2/dilation3；FP32、batch256/window64、seed123。
- AdamW lr2e-4、betas(.9,.99)、weight decay2e-5；EMA .99。历史独立999次warmup后300000 cosine，未改成总300k含warmup。
- SmoothL1 main1、Joint3、Velocity100、Rotation.01、Root.1、FootContact10、DistanceMap.1、commit.02。不新增损失。
- Root/Velocity/DistanceMap/Joint直接继承原loss方程；Rotation只改22关节reshape。FootContact仅为空mask添加零值防护，非空仍按原方向/权重计算；不保留原空mask可能产生NaN的缺陷。原`normalize_floor`实际不改变坐标的行为不暗改。
- 必要数据差异：用beta-dependent SMPL FK替换原SMPL-X group骨架；本数据独立train统计，不能使用Inter-X统计值。训练采用原window枚举（包括历史末窗口遗漏/首窗口重复），无手增强/镜像等额外策略。
- 共3263条长度>=64训练序列、559094个历史式窗口。短序列不补帧凑64。val有效>=16帧535条。所有split仍按原文件。

## 验证与存储

每5000 main steps固定InterHuman val前480个有效ID，中心窗口、最长300帧、长度4倍数、原首条文本。预测经逆坐标变换+SMPL+官方预处理还原524维，再用原InterCLIP。记录FID、检索、Matching、diversity、原坐标body MPJPE。固定清单写入run；隔离validation RNG且eval模式禁止EMA更新。

这是新的**训练内验证面板**，不是原Inter-X evaluator，也不是旧InterHuman随机裁窗480协议或正式test，不能直接拿数值宣称优于旧模型。正式十项VQ报告仍需冻结后另评：RMSE、全身/身体MPJPE、root误差、Full/Body joint FID、滑步、接触F1为待评；手部MPJPE/Hands joint FID为不适用（无手）。InterCLIP FID不冒充joint FID。

训练在cuda:0，验证evaluator在cuda:1。输出放/home现有训练目录，仅保留latest（含optimizer/EMA/RNG）、best val FID和175k模型，避免每5k累计大checkpoint；保留完整小体积指标记录。低频监控依赖后台训练每5k验证与manifest更新，不用模型逐分钟轮询。异常写failed状态，保存前检查磁盘余量。当前脚本不提供自动resume，不声称含DataLoader游标的bitwise续训。

## 启动验收

smoke主35468 exit0，原batch256/12288码本4次更新全有限，完整480验证流程通过；其step0 FID162.64885、body MPJPE984.164mm仅为几乎未训练的链路检查，不是训练结果。峰值训练allocated1.789GiB。
独立`tests/test_interhuman_interx_raw.py`已实跑exit0，含坐标/FK、raw空槽、VQ编码解码、shape作用、BytesIO模型/EMA严格重载和loss反传。

正式训练已于2026-10-04 20:18 BST实际启动，主session29332，W&B `knk4gll9`：https://wandb.ai/edmond_gla/NewInter/runs/knk4gll9 。输出`/home/wenning/HHC_interhuman_runs/vq_interhuman_smpl_raw_interx_handless_seed123_20261004`，首4个warmup update确认有限loss/梯度，不从smoke或旧524维模型继续。尚无主阶段验证成绩。

启动命令：

```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 /home/wenning/miniconda3/envs/New_MMM/bin/python -u train/train_vq_interhuman_smpl_raw.py --output /home/wenning/HHC_interhuman_runs/vq_interhuman_smpl_raw_interx_handless_seed123_20261004
```
