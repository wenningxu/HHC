# 两baseline真实生成：直接GT替换位置/旋转

2026-10-07。按用户明确要求，用相同请求的GT块替换生成结果，其余通道保持不变。不是旋转正交化或时间平滑。

## 结果

| 模型 | 原始FID | GT替换位置 | GT替换旋转 | 位置替换降幅 | 旋转替换降幅 |
|---|---:|---:|---:|---:|---:|
| InterMask | 8.780644 | 7.269357 | 2.744920 | 1.511287 | 6.035724 |
| InterGen | 9.681665 | 8.107611 | 2.560352 | 1.574054 | 7.121312 |

两模型在该替换诊断下，旋转比位置的FID影响更大；替换旋转的降幅分别约为替换位置的4.0倍、4.5倍。这是实际生成结果，不是此前VQ配对重建实验。

## 固定协议与检查

- 各480条冻结validation请求，生成seed20261006；沿用两个官方checkpoint与此前生成cache，无重新生成、挑选样本或参数调优。
- 输入raw `[T,2,262]`。位置只替换`[0:66]`，旋转只替换`[132:258]`，速度与接触均保留生成值。逐样本assert确认未替换通道完全不变。
- InterMask生成长度为floor(L/4)*4，取同一GT的对应前缀用于替换；InterGen生成与GT同长。不做时间扭曲或重新配对；参考GT仍保留原完整长度。
- 独立核查两cache全部请求及GT逐元素一致，无NaN/Inf；原始FID本次重算与此前JSON精确一致。
- 使用同一released InterCLIP motion encoder、latent_scale=4.317841053及官方embedding scale6的FID函数。CPU2线程，各任务约114/115秒。
- InterMask原session31337、InterGen原21801，均确切exit_code=0。

## 限制

文本生成与同caption GT不保证逐帧动作对齐，混合特征可能破坏位置—旋转或位置—速度的一致性。FID下降是指定干预的分布效应，不是可加的因果贡献比例；支持旋转差异重要，不能单凭此证明“没有旋转监督”或“旋转差异全是噪声”。单seed480验证集诊断也不是正式多rep test结果。

## 产物

- 代码：`tools/score_generated_gt_swaps.py`。
- InterMask：`output/intermask_generation_rotation_20261006/gt_swaps480.json`及同名`.embeddings.npz`。
- InterGen：`output/intergen_generation_rotation_20261006/gt_swaps480.json`及同名`.embeddings.npz`。
- JSON包含原始cache、checkpoint/request标识、各FID及降幅；embedding保存GT/raw/GT_position/GT_rotation四组，可复算。
