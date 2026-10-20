# InterHuman 原生 SMPL 的 Inter-X 风格组织

不进行 SMPL→SMPL-X 转换；保留真实每人10维betas。构建入口为
`tools/build_interhuman_smpl_pair.py`，适配器为`dataset/interhuman_smpl_pair.py`。
独立输出目录：`dataset/interhuman_smpl_interx_layout_v1_20261004`。

## 数据合同

H5每个数字ID一组，人物顺序严格保持person1/person2：

- `axis_angle_full [T,23,6]`：原始59.94FPS，前22行是root orientation及21个body关节轴角；最后行为translation。每行前3维person1，后3维person2。
- `betas [2,10]`：原始shape参数，不归零、不转SMPL-X。
- `motion30 [T30,23,12]`：前22行每人6D旋转（矩阵前两行展平）；最后行为`[p1位置3,p1速度3,p2位置3,p2速度3]`。速度是后向差分，单位m/frame，首帧0；裁窗后重新计算首帧速度。
- `joints30/joint_velocities30 [T30,22,6]`：实际betas参与female SMPL FK的关节位置/后向速度，双人XYZ拼接。
- `frame_indices30`：`floor(k*59.94/30)`。原始全速帧另存，不能把30FPS降采样称为无损。
- 原始fps、gender标签、帧数及文件MD5保存在组属性。原始标签neutral不改写；female是已全量验证匹配官方身体几何的**计算模板**，不是推断受试者性别。

保留原始坐标，不重新落地、不分别居中。身体分支264维（含root orientation），root分支12维；shape独立20维条件。此处“格式相同”指双人排列与物理量组织，不是沿用旧55关节/672维合同；也不是旧Inter-X `vel`模式中丢弃绝对平移的编码。

split文件逐字节复制，文本链接原始annots，始终按同一数字ID绑定。normalization仅由train统计；两个原始空动作3945/4106只归档不参与训练。正式loader拒绝未完成或pilot数据。

## 评测路径和无损边界

生成的未归一化motion30 + betas → 原生female SMPL FK → 官方body192 → 原官方在线处理 → 双人524维。调用`to_interhuman`；输出长度为输入T-1，与官方速度/接触预处理一致。评测需要归一化时显式传`normalize=True`。不读取GT动作、残差或接触标签来修补生成结果。

原始轴角、translation、betas可以逐字节恢复。6D转换、FK和官方在线特征只能保证经过实测的浮点精度，尤其接触阈值可能受舍入影响，不能承诺bitwise一致。官方完整492维含额外手部位置，本版本只恢复评测器实际使用的身体部分，不伪造手指动作。

旧524维InterHuman checkpoint及旧672维Inter-X checkpoint不能直接吃276维数据；本次适配器并不等于已完成新VQ/MGT shape条件训练。按ponytail最小实现原则将改动隔离为数据/转换层，复用既有SMPL和官方预处理，旧基线保持不变。

## 验证状态

32对pilot已完成（主session62130，exit0）：256原始数组逐字节相同；最大关节误差0.00060858mm；contact 0/23000不同。
测试`tests/test_interhuman_smpl_pair.py`已通过（17967，exit0）：原始还原、时间索引、裁窗/批量速度、shape改变FK、motion/beta反传、官方身体对应及524维转换。
全量构建主session53034 exit0：7779个ID归档，7777有效对，空3945/4106；62232个原始数组逐字节恢复。实际经过6D编码/解码后，平均body MPJPE为0.0000754133mm，最大单关节误差0.0008821941mm，旋转最大差5.96046e-7；官方在线位置最大差9.77516e-6m，contact有8/8980112个不同（因此不是严格无损官方特征转换）。完整逐样本结果见输出目录manifest.json。

正式loader验收主61417 exit0：split文件和所有载入ID的文本逐字节一致，裁窗后归一化/反归一化首帧速度检查通过，betas返回20维。窗口至少8帧时train/val/test分别5978/575/1172；这是短片过滤后的数量，不是修改官方split。输出约3.1GiB，磁盘剩1.6GiB，后续训练前需要另行检查存储预算。本次未启动训练。
