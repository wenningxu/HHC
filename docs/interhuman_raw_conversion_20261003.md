# InterHuman 原始数据下载与转换排查

## 范围与状态

用户最新任务：仅在 `/mnt/8T-HDD/wenning` 清理无用 checkpoint，下载官方原始 motion，研究其与官方 processed 数据不能逐项对应的原因。未改变训练数据、模型配置或重启 294k MGT。

**最终状态（10-03）：下载全部验收；7777对有效动作的身体/当前HHC输入转换已完成全帧验证。不是bitwise完全相同，也没有宣称重建全部492维。**

| 全量检查 | 结果 |
|---|---|
| 下载 | 7779/7779文件size+MD5通过，0失败，主49128 exit0 |
| 有效动作 | 7777对、15554人次、2260582 person-frames；2个空raw跳过 |
| 帧/人物对应 | 同编号同person、floor时间采样，全量无帧数错误，无人工配准 |
| 身体MPJPE | 按人次均值0.00005675mm；按帧加权0.00005797mm |
| 最坏单关节误差 | 0.00092195mm（7758/person2） |
| 身体6D最大绝对差 | 7.15e-7 |
| 在线262维位置/速度最大差 | 9.78e-6m / 2.15e-6m每帧 |
| 足部接触 | 9/8980112个值不同，约0.0001002%，均已定位阈值舍入 |
| 原始字段 | 全部只有betas/gender/pose_body/root_orient/trans，全部59.94FPS、gender=neutral |
| shape/手部 | 有效动作中3202个不同beta向量（不是身份数）；全部手部6D旋转为同一固定姿态 |

主全量25979 exit0，详细输出 `output/interhuman_raw_body_reconstruction_full_20261003.json`。这里的MPJPE是格式转换误差，**不是VQ或MGT指标**。

## 空间清理

- 完成 run：`output/t2i_trans/2026-09-02-11-49-08_hhc_historical_plus_paper_mask_ratio_raw12288_300k_seed114514_v2`。
- `last.ckpt` 与 `latest-000300000.ckpt` 各 3,943,090,318 bytes，SHA256 同为 `c1c3c35bcbb027c1d1c6dc95e39f511f55c1e6b08c9db8e596e9bfb73043eb31`。
- 已将 last 替换为 latest 的硬链接，两个路径与模型内容均保留；释放约 3.67 GiB，HDD 可用从约 2.1 增至 5.8 GiB。没有删除最佳点、对照实验或独有权重。
- 两路径现在共享 inode，不能原地覆盖其中一个；该历史 run 若续训，应使用新输出目录。空间去重可逆（有空间时复制回独立文件）。

## 下载

- 官方目录：https://drive.google.com/drive/folders/1oyozJ4E7Sqgsr7Q747Na35tWo5CjNYk3
- 原始 motions：https://drive.google.com/drive/folders/1cr46yZHEgdlaMR61TJojjZ05Hlq9DC70
- 全部分页清单：7,810 条目，31 条同名同 size/MD5 重复，7,779 个唯一 `.pkl`，总计 1,245,872,625 bytes（约 1.16 GiB）。保留重复 ID 溯源。
- 目标：`dataset/interhuman_official/motions`；完整清单和下载状态：`output/interhuman_raw_download_20261003.json`。
- 工具：`tools/download_interhuman_raw.py`，文件先写 `.part`，官方 MD5/size 校验通过才改为 `.pkl`；不覆盖不匹配的现有文件。
- 下载 session49128已exit0，完整清单status=complete/verified_count=7779/failures=[]。Drive连接器只列前1000项、gdown文件夹接口最多50项，因此完整枚举改用公开Drive网页所用API分页；不读取账户凭证，不保存网页client key。下载本体最终使用公开usercontent端点，保留MD5校验。
- 历史67516（gdown）到500项8失败后主动TERM/exit143；15600（media API）到1107项停滞后主动TERM/exit143；均先收终态再改源，已校验文件保留。旧全量队列96998因下载尚未complete而exit1，未错误启动评测。替代队列25979已在下载验收后完成全量评测，exit0。

## 官方证据与必须区分的两级预处理

- [官方 README](https://github.com/tr3e/InterGen)：motions 是 SMPL 参数，motions_processed 为位置/6D旋转。
- [作者 issue27 回复](https://github.com/tr3e/InterGen/issues/27)：492维对应 SMPL-H 的 `62×3 + 51×6`。
- [issue23](https://github.com/tr3e/InterGen/issues/23)已有原始与processed时间步不匹配的提问，当前未找到作者提供的完整离线转换脚本。
- 已发布 `utils/preprocess.py::load_motion` 不是原始 pkl 转换器：它从492维选22关节位置及21关节旋转，即192维。
- 随后在线 `process_motion_np` 做坐标轴变换、落地、初始XZ平移、朝向归一化、速度与足部接触，变成262维且少最后一帧；再把person2放回相对person1的坐标系。不可拿这一输出直接与原始参数FK逐帧对比。

## 首样本7778：已确认的数值证据（尚未全量外推）

- raw：157帧，59.94 FPS；每人 trans、root_orient、pose_body(63)、betas(10)、gender=`neutral`。两人betas不同且非零。
- processed：两人均79×492。
- 帧索引不是 `[::2]`，而是 `floor(k × 59.94 / 30)`，前几项 `[0,1,3,5,7,...]`，最后155。身体旋转最近邻独立验证同一索引。
- 身体6D旋转采用矩阵前两列按行展开（交错列元素）。用该索引和SciPy Rodrigues，平均绝对误差约1.1e-7，最大约0.001；零旋转数值约定尚待排查。
- 同编号/同人物/同原坐标，无任何刚体或root对齐，原始betas与本地 **female SMPL** 模型重建前22关节，person1/person2 MPJPE约 **0.000083 / 0.000053 mm**。
- 照raw的neutral标签使用SMPL，误差约27.39/42.00mm；使用male约88.54/79.86mm；直接用SMPL-X neutral约145.57/151.51mm。
- 因此首样本不对应的已证实原因包括：取帧规则，以及原始gender标签不能直接指导所需身体模板。SMPL前22关节精确对应，不等于已重建全部SMPL-H手部/62个位置。
- 同一7778额外验证female SMPL-X仍有134.17/138.96mm误差（session64976 exit0），所以只把SMPL-X的gender改female还不够；不能把这套SMPL-family betas直接当SMPL-X betas使用并期望相同关节。
- 扩样程序：`tools/diagnose_interhuman_raw_correspondence.py`；64对初诊运行 session85853，结果待收。

## 验收范围与限制

1. 下载、ID覆盖、帧数、body192及online262已全量验证；不再需要为本次下载删除更多checkpoint，下载后HDD可用约4.6GiB。
2. 使用female模板是数值反推并全量验证的转换约定；没有找到作者完整离线转换源码，不推断raw的neutral标签为何如此填写。
3. 全492维中额外40个手部位置未重建；作者说明采用SMPL-H，本机未找到对应SMPL-H体模。不能用SMPL-X或补零假装精确复现全部492维。当前HHC实际读取的身体部分已覆盖。
4. 未接入训练、未给模型新增shape条件、未重启294k MGT；长期研究goal仍保持原paused状态。

## 扩样结果（下载尚在进行，不是全量结论）

- 64对/128人，6种性别模板×shape对照：全部最优为female+原始betas，平均MPJPE 0.00005665mm；female零shape24.857mm、male原始93.146mm、neutral原始36.215mm。结果 `output/interhuman_raw_correspondence_probe64_20261003.json`，session85853 exit0。
- 旋转残差已定位到两个wrist字段：这64对的官方wrist旋转全部identity，原始wrist轴角却有小量非零；其余19个body旋转误差最大约2.96e-7。不是普遍的6D转换误差。官方30个手部旋转在这批样本中固定为同一非零默认姿态，不能据此说全数据都有真实手指动态。
- 独立实现 `tools/reconstruct_interhuman_raw_body.py`：female SMPL、原始betas、floor时间采样、body22 FK、序列化wrist旋转置identity，直接生成官方模型实际读取的192维；**不伪造全492维**，不覆盖官方文件。
- 128对/256人、完整序列验证（session10091 exit0）：无帧数不匹配；平均MPJPE **0.00005405mm**，最大单关节误差 **0.00061077mm**，6D最大差 **4.77e-7**。
- 再复用原官方配对预处理转262维：位置最大差2.50e-6m，速度9.54e-7m/frame，归一化全特征最大差7.33e-5；101000个足部接触值 **0处不一致**。
- 初步结论：用户遇到的不对应，在身体及HHC实际输入层面已经可以数值复现；关键是59.94→30的floor索引、female模板（不能照neutral标签）、保留原始betas、wrist序列化置零，以及区分离线492与在线262。尚待全量检查例外。

## 文件级对应的额外差异

- 原始唯一ID 7779，processed person1/person2各7777；缺失的是 **3945、4106**。
- 已检查本地官方 `motions_processed.zip` 的中央目录：它本身也没有这两个ID，不是解压漏文件。其它所有processed ID均能在原始清单找到。
- 原始3945/4106下载后进一步实查：两文件`frames=0`，两人的trans/root_orient/pose_body均0帧。因此它们没有有效processed对应，不是编号错配；重建函数明确拒绝空动作。
- 原始Drive里31个同名重复条目均同size/MD5；这不代表31条新动作。下载以唯一ID为单位，保留重复Drive ID记录。

## 使用重建函数

`tools/reconstruct_interhuman_raw_body.py::reconstruct_person(person, frame_rate, model)` 返回未归一化192维及其原始帧索引；`load_model()`加载已核验female SMPL。双人结果交给 `OfficialInterHumanAdapter.process_loaded_pair` 得到与当前HHC一致的262维，是否归一化由参数显式控制。若训练需要随机crop，应与官方一样先crop未在线处理的192维，再对该窗口执行在线配对预处理。

此实现只是独立重建/诊断入口，没有接入当前训练，未把betas作为新增模型条件。两项回归测试覆盖7778数值、时间索引以及优化FK与标准SMPL forward一致性，session23170 exit0。

这些差异解释的是“自己从raw重建为何不匹配”，不意味着当前HHC把已发布processed文件读错：现有训练一直使用官方processed和官方在线预处理。也需要修正早先口头表述：当前模型没有显式接收betas条件，但processed关节位置本身保留了原始shape的几何影响，不能说动作数据完全不含shape信息。

本报告MPJPE是确定性数据格式重建误差，不是VQ-VAE重构质量，也不是MGT生成指标。本次没有重训/替换任何模型，不能用这些微小转换误差宣称原VQ质量问题已经解决。

其它容易混淆的细节：`trans`是体模的平移参数，不应直接替代最终pelvis关节；应由正确模板/shape做FK再加trans。与离线492维比较时不要提前落地、去初始XZ或转朝向；这些属于后续在线262维阶段。6D身体旋转的排列是矩阵前两列交错展开 `[r00,r01,r10,r11,r20,r21]`，不是常见的前两行展平。

## 9个接触值差异的逐项核查

`tools/diagnose_interhuman_contact_roundoff.py` / 主20884 exit0；输出 `output/interhuman_raw_contact_roundoff_20261003.json`，断言恰好覆盖全部9处。

- ID：1564（2处）、1569、2033、2788、3677、3678、3979、7013。
- 8处为高度阈值0.05/0.12m两侧的浮点差异，重建与官方高度之差约5.96e-8～1.19e-7m。
- 7013/person2/frame189为速度平方阈值0.001两侧：重建0.000999995740，官方0.001000005519；相差约9.78e-9。
- 未调整任何阈值、未替换标签来消除差异。全特征归一化最大差2.188来自这9个离散bit；连续通道的全量归一化最大差保守上界为2.90e-4（用各连续分组实际最大差/该分组最小std计算，不冒充精确最大值）。

## 结果校验摘要

- download manifest SHA256：`4df6ded9ffabbea3b3d755f2acea1329dea166b7a143218d354612bfc76bef65`
- 全量转换结果 SHA256：`d00dcfcf5edd4deba4da2ec1b08dad4b84ccbee3bbabab2ed54709566783c67e`
- 接触边界结果 SHA256：`047d83f6e4355835aba17cdfcc850e22c9ae6f9ae99b2c1db9d2a5fcb0824aa7`
- female SMPL模型SHA256：`6d4a1791b6b94880397e1a3a4539b703a228d2150c57de7b288389a8115f4ef0`。
- 4个工具与回归测试的源码快照：`output/interhuman_raw_conversion_sources_20261003.tar.gz`，5文件逐项读回一致；SHA256 `dc35a03740dabef85f6d4b0d3008a42c82d2abd1bb8372cb7a3a8c455e69930d`。当前重建脚本SHA与全量结果provenance一致；最终5文件py_compile通过。
