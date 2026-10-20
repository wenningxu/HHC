# InterHuman official VQ rotation projection — 2026-10-07

免训练诊断：固定 InterMask 发布版 InterHuman VQ checkpoint（SHA256 `af57bbdde94ed4d4a01f3ba5f9fe24f0012c2a37b3e38c2fa57a2a91ebcc1272`），对两份真实生成的 480 条 validation cache 分别重建双人序列。原始 raw 262D 按官方 mean/std 归一化；序列 edge-pad 到 4 的倍数后解码并裁回生成长度。仅将反归一化 VQ 重建的旋转与原始旋转做 6D 线性 blend，alpha 为 0.25、0.5、1.0；没有额外正交化。位置、速度、contact 原值逐位保留。另测发布数据约定中的 wrist identity：只在 GT 全量检查通过时，对生成的 `[246:258]` 替换为 `[1,0,0,1,0,0] × 2，不读取或复制 GT 动作值。

4 条 smoke（每模型）均通过；VQ 输出长度等于 edge-pad 长度，之后按原生成长度裁剪。全量 CUDA 重建 PID 78612，句柄 49174，exit code 0；设备 `cuda:1`。独立 CPU2 evaluator PID 80739，句柄 11962，exit code 0。评分沿用官方 MotionEncoder、模型 hash、latent_scale `4.317841053009033`、activation scale 6 与官方 Frechet 实现。两个模型 raw FID 重算均精确匹配已有 baseline。

| Validation generation | Raw FID | VQ alpha .25 | VQ alpha .5 | VQ alpha 1 | Gen-only wrist identity |
|---|---:|---:|---:|---:|---:|
| InterMask | 8.780644 | 8.747113 | 8.717303 | 8.671397 | 8.780609 |
| InterGen | 9.681665 | 9.675755 | 9.702095 | 9.813278 | 9.681665 |

对应 FID 相对 raw 的改善量（raw − intervention）：InterMask alpha `.25/.5/1` 为 `0.033531/0.063341/0.109247`；InterGen 为 `0.005910/-0.020431/-0.131614`。两模型 GT wrist identity 槽的最大绝对误差均为 `7.45e-9`（阈值 `1e-5`），identity-arm 对两边 FID 的影响近于零。VQ alpha=1 在 InterMask validation 上改善、在 InterGen 上恶化；单一验证 generation 不能支持普遍有效结论，也不是 formal test。

FID 是独立 hybrid intervention 的诊断值；不同 alpha 的变化不可相加。线性 6D blend 可能不在 SO(3) 流形上，且 VQ 重建与未改动的 P/V/contact 可能不一致。

Artifacts: `output/rotation_methods_20261007/vq_projection_inter*.{json,embeddings.npz}`. 仅旋转 VQ 输出暂存于 `/dev/shm/hhc_vq_projection_20261007/`，未生成完整大 motion cache。
