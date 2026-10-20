# InterMask 真实重建误差与 InterCLIP FID 消融

2026-10-06。实验对象是 released InterMask VQ 的真实配对重建，不是人工随机噪声，也不是 text-to-motion 生成。后者由主线程另行实验，不能混用结论。

## 数据与实现合同

- 输入：`/home/wenning/HHC_interhuman_runs/interhuman_vq_validation/official_intermask_reconstruction.h5`，480条冻结 InterHuman validation 请求，91450 person-frames；与GT逐项核对identifier、caption、length、request及normalization hashes。
- H5 schema=`interhuman_fair_generation_v1`，method=`intermask_official_vq_reconstruction`，split=`val`，status=`complete`；checkpoint epoch46 / iteration20608。
- `output/fair_eval/intermask_official_exact/run_manifest.json` 是 **InterX**，本实验明确排除。
- 复用官方 motion encoder、released checkpoint，strict 加载motion参数，输出包含学习到的latent_scale；FID继续使用官方 `calculate_activation_statistics(...,6)` 与 `calculate_frechet_distance`。
- 实验代码：[ablate_intermask_feature_fid.py](/mnt/8T-HDD/wenning/HHC/tools/ablate_intermask_feature_fid.py)。仅CPU2线程，禁用CUDA，没有改训练源码或参数。

重要布局核查：官方InterHuman存储的6D是 `matrix[..., :2].reshape(...,6)`，即first-two-columns的**交错布局**，identity为`[1,0,0,1,0,0]`。不能拿另一种连续3+3表示的解码函数直接处理。最初严格退化检查及时发现此问题（PID56165，exit1），修正后按全部GT验证。当前投影保留两列顺序，Gram–Schmidt得到SO(3)，重新交错编码；对投影幂等、旋转保持、P/V未变分别有assert。

InterMask官方 `models/losses.py:99` 调用 `data/quaternion.py:320` 的连续3+3解码，且接收标准化旋转块，不能直接作为这里的真实GT旋转几何合同。这属于公开实现与数据语义的差别，不是GT自身退化或“噪声”。已有HHC正确转换证据见 `dataset/interhuman_smpl_pair.py:100`。

## 已完成的全480配对误差

以下是按有效帧与joint累计的均值，非各序列等权均值。旋转角误差先将重建6D按正确布局投影后与GT旋转比较；角度不是原始6D欧氏误差。

| 指标 | GT | InterMask reconstruction |
|---|---:|---:|
| MPJPE | — | 109.926 mm |
| 速度向量L2误差 | — | 6.508 mm/frame |
| 关节旋转角误差 | — | 5.115° |
| 6D两列向量长度偏离1的绝对值 | 1.392e-8 | 0.005493 |
| 6D两列内积绝对值 | 1.194e-8 | 0.005966 |
| `V-ΔP` 向量L2残差 | 0.0000523 mm/frame | 4.439 mm/frame |

标准化每维MSE为P=0.0177171、V=0.1020461、R=0.1294752；不同模态维数、单位、冗余和decoder自由度均不同，不能仅凭R的MSE最大判断R主导FID。6D合法化的每维均方改变量为GT=8.66e-16、重建=6.41e-5，因此GT投影理论上近乎恒等，但重建确实有轻微离流形分量。

统计文件：[pilot JSON](/mnt/8T-HDD/wenning/HHC/output/intermask_feature_fid_20261006_pilot.json) 已包含全部480统计。修正后的pilot session55757 / PID56382，exit0，用时7.2秒；16条embedding只用于检查pipeline，原始FID5.17888、两侧旋转合法化FID5.43120，不作为正式总体证据。GT自FID约-5e-5为奇异小样本协方差的数值误差，保留原值而不擅自裁零。

## 完整480特征替换实验

完整run：session1852 / PID56522，**exit_code=0**，296.27秒。完整结果：[full JSON](/mnt/8T-HDD/wenning/HHC/output/intermask_feature_fid_20261006_full.json)，[全部embedding缓存](/mnt/8T-HDD/wenning/HHC/output/intermask_feature_fid_20261006_full.embeddings.npz)。输入H5实算SHA256=`065b0afe76c7a3aca35f2a26845faa26e6c80f2267f9022c7f59c34d528e3ba0`，与历史评分记录一致；本次执行脚本SHA256=`895eab57f2b9d8a7857cddbd86f58a83856afd98877c2ab90e0dc41791e3a7c9`。

8个条件从GT出发，分别将P、V、R的任意子集替换为对应的**同一GT的真实重建块**。空集GT为参考，PVR为实际重建（接触不影响encoder）。另外GT和PVR两侧分别投影R，检查离SO(3)流形是否解释误差。所有条件有效长度、crop、人序和normalization完全一致。

| 从GT替换的真实重建块 | 对raw GT的FID | 平均配对embedding L2 |
|---|---:|---:|
| 无（GT） | -0.00000110 | 0 |
| P | 0.360348 | 0.163650 |
| V | 0.001283 | 0.011935 |
| R | 0.569187 | 0.312919 |
| PV | 0.359889 | 0.164511 |
| PR | 0.975331 | 0.363283 |
| VR | 0.581075 | 0.313813 |
| PVR（完整真实重建） | **0.985273** | 0.364102 |

历史独立评分为FID=0.985952；本次CPU逐序列执行结果差-0.000679，约0.069%。这里没有声称bitwise重现历史GPU/batch96评分；具体执行/低秩协方差数值差异未逐项隔离。所有干预使用**本次相同CPU路径及同一参考**，差异不能通过引用历史分数来放大或缩小。

基于全部8种组合的诊断Shapley（从全GT到全重建的FID增量分摊）：P=0.382307（38.80%）、V=0.005647（0.57%）、R=0.597320（60.62%），和为0.985274（扣除GT数值残差）。这是指定混合特征实验下的平均边际贡献，不是任意自然运动的唯一因果归因；一般可以为负。本例R是最大项，同时P接近四成，不能说位置几乎不被评分。

直观的同一重建干预也一致：将R恢复为配对GT而保留预测PV，FID由0.985273降到0.359889（-63.5%）；将P恢复GT而保留预测VR，降到0.581075（-41.0%）。两个改善率不能相加：交互效应和FID的分布统计使其非可加。P/V及位置/旋转之间也可能因替换而不一致，所以这是特征误差诊断，不是可直接发布的自然生成质量提升。

## 旋转离流形是否就是FID的“噪声”来源

两侧都按正确布局合法化R后，FID为 **1.079522**，相对raw/raw的0.985273 **上升0.094248（9.57%）**。GT自身投影近乎恒等（embedding L2=5.31e-7，FID数值残差约-1.1e-6）；在此GT参照下，只投影重建得到1.079522也几乎相同。投影前后P和V数值完全不变。

所以此处需要区分两个命题：

1. **“实际旋转重建误差对该evaluator分布距离的影响大于实际位置重建误差”**：本次480条配对VQ消融支持，约61%对39%的指定Shapley分摊。
2. **“旋转6D不合法/随机噪声把位置质量掩盖了，去掉这些噪声FID应显著改善”**：本次SO(3)合法化不支持；它使FID变差。合法化只移除非正交表示分量，不去除合法旋转中的真实角误差或时间抖动，不能把这个阴性结果扩大成“任何旋转噪声都不重要”。

这仍不能回答InterMask **MGT生成** 或 InterGen的同一命题，亦不能证明方法间的不公平。真实位置MPJPE约110mm已经单独记录；FID偏重某组真实误差，不代表该样本的位置本身已很好。后续生成实验应使用真实生成cache，并对GT及生成对称干预；不允许将同caption但未对齐的GT旋转逐帧替换进生成来声称“真实旋转误差修正”。

运行命令：

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
/home/wenning/miniconda3/envs/New_MMM/bin/python -u tools/ablate_intermask_feature_fid.py \
  --limit 480 --coalitions --output output/intermask_feature_fid_20261006_full.json
```
