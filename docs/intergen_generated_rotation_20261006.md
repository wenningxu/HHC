# InterGen 真实生成的旋转/FID干预诊断（2026-10-06）

4条冒烟生成及六臂评分已通过；完整480条正在运行，目前不能据此判断旋转噪声是否主导 FID。

## 资产与协议

- 官方源码 `tr3e/InterGen` commit `b1750df4a859443b8a69329f759ce3a7503ae179`，临时目录 `/dev/shm/hhc_intergen_20261006.CtYnWA/source`。
- 官方 `intergen.ckpt`，2701617525 bytes，SHA256 `bab341123c27af9d8c3ee7ebc43c2d341bfd0f4103425d06f8af6c717a086f8d`。严格加载成功，missing/unexpected 均为 `[]`。
- 复用 `/home/wenning/.cache/clip/ViT-L-14-336px.pt`；不下载新CLIP、不占GPU，CPU 2线程。
- 原版 `InterGen.forward_test`，DDIM50、CFG 3.5、batch 1；seed 20261006。未使用 `infer.py` 的可视化位置平滑，保留原始模型输出，遵循 `datasets/evaluator.py` 不启用平滑的评分路径。
- 共享冻结 val 请求 `output/interhuman_validation_requests/manifest.json`，request SHA256 `751efc9ced91e2a6ad4eec4a7b9316946761fe13f86d09af7d5c04a9226b7f93`。生成长度遵循InterGen原版不取整；InterMask有floor(length/4)*4，跨模型比较须保留此区别。
- 官方 mean/std 与冻结adapter精确一致，SHA256分别为 `2f97b88889c773551cd597c32e88bfa070c2fb1721a992a80bfb7cf4c22e03b6` / `dd5988b979e9803786ab497f5945316c2d52b97c90b593d0427220fd49b03e4b`。
- H5保存反归一化float32 `[L,2,262]`，GT与生成逐请求保存；缓存上限200 MB。

## 干预

复用 `tools/ablate_intermask_generated_rotation.py` 的同一评分函数：raw、R_project、R_smooth、P_smooth、V_consistent、P_smooth_V_consistent。每个干预同时施加于生成与GT。R为first-two-columns interleaved布局，identity为`[1,0,0,1,0,0]`；P/R平滑核为`.25,.5,.25`；V一致化使用逐帧位置差，末帧保留。不存在用未对齐GT替换生成旋转的“修复”。

这是冻结val的单seed归因诊断，不是官方多seed基准；小样本FID有明显估计偏差，干预FID变化不可相加成贡献百分比，平滑也会删除真实运动。

## 活跃任务

- 冒烟：前4条、64帧上限，session `65872` / PID `56915`；已结束exit_code=0，生成耗时42.88秒，真实长度19/64/64/44。
- 冒烟评分：session `15253` / PID `57110`，exit_code=0。GT旋转列范数/正交误差均约`1e-8`，支持interleaved布局；生成对应误差`.00657/.00892`，逐帧旋转角GT/生成为`1.2875/1.3246`度。小样本raw/R_project/R_smooth FID为`138.925/140.710/140.550`，仅作管线检查，不用于总体归因。
- 完整480直接启动（跳过重复生成32）：tmux `hhc_intergen_rotation_20261006`，pane PID `57257`、shell `57258`、生成Python `57259`。启动命令exit_code=0，任务尚未终止。
- 已生成前2条：19/115帧，累计6.20/28.34秒；Python RSS约3.26 GB、CPU约173%。按480条平均95.26帧估计完整生成约1.5–2小时（仅估算）。
- 完整缓存路径 `/dev/shm/hhc_intergen_20261006.CtYnWA/full480.h5`，191.68 MB未压缩数据量，使用LZF且运行时检查200 MB上限。内存临时文件不保证重启后保留；不会删除。HDD当时剩1.5 GB、RAM available45 GB、`/dev/shm`剩29 GB，因此大缓存放RAM。
- 日志 `output/intergen_generation_rotation_20261006/full480.log`；终态文件 `full480.exit_code`；tmux设置remain-on-exit。480生成成功后自动调用同一六臂评分，再把 `full480.json` 与 `full480.embeddings.npz` 复制到上述output持久目录。执行脚本 `run480.sh` 运行期间不可修改。
- 下一步：读取完整480终态与持久结果；估计CPU耗时约1–2小时，不修改DDIM50/CFG3.5。单种子480仍是诊断，不是正式多种子基准。
