# InterMask 真实生成旋转干预（2026-10-06）

状态：4条64帧上限冒烟、32条完整请求长度生成及评分均已完成；480条已启动并串联评分。该实验是真实 text-to-motion，不是 VQ 重建。

协议：冻结 `output/interhuman_validation_requests/manifest.json` 前N条，caption/请求顺序固定，seed=20261006；官方 Transformer epoch336 + VQ epoch46/iteration20608，CPU两线程，逐条batch1。原采样参数20 steps、CFG2、topk阈值0.9、temperature1、multinomial；生成长度按官方 `floor(L/4)*4`，GT保留请求原长度L。每一干预臂使用完全相同的两组样本及各自长度；缓存明确记录两种长度。

模型SHA256：

- Transformer `392ca8c9e42c2c23aeb97ba7b810e8fa4bdcc5f4839a5a909be2f56bf1698665`
- VQ `af57bbdde94ed4d4a01f3ba5f9fe24f0012c2a37b3e38c2fa57a2a91ebcc1272`
- 请求内容 `751efc9ced91e2a6ad4eec4a7b9316946761fe13f86d09af7d5c04a9226b7f93`

脚本：`tools/ablate_intermask_generated_rotation.py`。所有干预对GT和gen两侧对称执行：原始、R正交化、R正交化后(.25,.5,.25)时间滤波再投影、P同核滤波、V←ΔP、P滤波且V←ΔP。R使用正确3×2交错列格式，identity为[1,0,0,1,0,0]；V末帧因缺下一帧保留，双侧一致。投影只改变132:258，P对照只改变0:66；脚本有幂等及通道隔离检查。官方motionencoder严格加载，embedding归一化乘latent_scale=4.317841，再使用官方activation scale6计算FID。

冒烟实际生成长度16/64/64/44，6.92s；生成session69021/PID56421、评分session16508/PID56664均exit_code=0。缓存 `output/intermask_generation_rotation_20261006/smoke4x64.h5` 是float32原始262D。

|4条冒烟，仅管线验证|FID|
|---|---:|
|原始|139.4997|
|R正交化|139.9309|
|R温和滤波|139.7030|
|P温和滤波|139.4463|
|V←ΔP|139.4564|
|P滤波+V←ΔP|139.4289|

32条完整长度生成81.98s（session72280/PID56767，exit0）、评分20.32s（session81404/PID57033，exit0）：

|32条试验，仍是小样本|FID|
|---|---:|
|原始|61.32365|
|R正交化|61.41680|
|R温和滤波|61.32967|
|P温和滤波|61.30666|
|V←ΔP|61.30839|
|P滤波+V←ΔP|61.28762|

480条运行句柄：tmux `intermask_rot480_20261006`（remain-on-exit），shell PID57154、生成PID57180。日志 `output/intermask_generation_rotation_20261006/generated480.log`；原始缓存 `generated480.h5`，成功后自动评分写 `generated480.json` / `generated480.embeddings.npz`。准确终态由 `tmux list-panes -t intermask_rot480_20261006 -F '#{pane_dead} #{pane_dead_status}'` 收取，不把暂时无新输出当退出。启动时目录约11MB，480原始未压缩理论191.7MB，使用lzf压缩；预计生成约20–25min、评分约5min。

不能把4/32条FID当作生成质量结论。FID不可加，差值不是某组特征的百分比贡献；温和滤波也可能抹去真实动作。未使用不对齐GT逐帧替换。下一步：收取480单seed验证集诊断（仍不是多次test benchmark）。
