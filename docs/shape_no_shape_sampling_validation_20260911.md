# 有/无 shape checkpoint：原采样与排序加噪

用户要求停止探索，仅验证两类既有 checkpoint。2026-09-11 启动。
两组验证现已完成并核验；有shape pipeline session `55740` / monitor `1743` 正常结束，exit0。不追加训练或采样网格。
加噪方案为 `confidence_ranking=probability`、`confidence_noise_scale=1`、scope=all、linear_to_zero；不是 log_odds ranking，不是直接给坐标加噪。

共同设置：原 VQ175k；50 steps、T=.5、CFG3.5、paper_text_cfg；requested betas、无 contact guidance；544 条冻结 validation 质量样本（seed500123），100×11 MM（seed600123）；CPU32 joint，valid_frames physical。这是 validation 对照，不是独立 test 结果或多 seed 置信区间。

## 无 shape：300k，已完成

Checkpoint：`output/t2i_trans/2026-09-07-11-49-54_hhc_historical_paper_mask_no_mgt_shape_pilot50k_seed114514_v4/model-000300000-0.3973.ckpt`。
结果目录：`output/hhc_no_shape_confidence_validation_20260910`，采用两臂 `*_main_cpu.json`、`*_mm_cpu_detail.json`；physical 为 `control_physical_per_sample.json` / `confidence_all_physical_v2.json`。

| 指标 | 原采样 | 排序加噪 |
|---|---:|---:|
| Full FID ↓ | .240967 | .203157 |
| Body FID ↓ | .234382 | .209598 |
| Hands FID ↓ | .210853 | .265925 |
| Full Matching ↓ | 3.028678 | 3.039146 |
| Body Matching ↓ | 3.081421 | 3.062837 |
| Hands Matching ↓ | 3.084399 | 3.110037 |
| Full R@1/2/3 ↑ | .433824/.643382/.766544 | .439338/.641544/.753676 |
| Body R@1/2/3 ↑ | .454044/.647059/.781250 | .443015/.652574/.761029 |
| Hands R@1/2/3 ↑ | .422794/.613971/.735294 | .411765/.613971/.755515 |
| Full/Body/Hands Diversity | 9.646897/9.208961/8.839807 | 9.722510/9.080671/8.816910 |
| Full MM ↑ | .043909 | 1.160418 |
| Body MM ↑ | .047873 | 1.228123 |
| Hands MM ↑ | .083011 | 1.814558 |
| 精确重复对比例 ↓ | 67.8545% | 1.9818% |
| 足滑 m/s ↓ | .303470 | .313003 |
| 接触 F1 ↑ | .557580 | .541148 |

GT Diversity Full/Body/Hands = 9.104939/9.100474/8.466241。接触 F1 不代表穿透；本表未测 mesh penetration。生成任务没有唯一目标动作，不用配对 MPJPE 代替生成质量指标。

## 有 shape：260k，两臂已完成

| 指标 | 原采样 | 排序加噪 |
|---|---:|---:|
| Full/Body/Hands FID ↓ | .241860/.214220/.403981 | .244934/.194259/.390061 |
| Full/Body/Hands Matching ↓ | 3.170896/3.291101/3.224519 | 3.243371/3.226206/3.257848 |
| Full R@1/2/3 ↑ | .398897/.612132/.727941 | .365809/.579044/.724265 |
| Body R@1/2/3 ↑ | .430147/.615809/.740809 | .428309/.636029/.738971 |
| Hands R@1/2/3 ↑ | .378676/.591912/.702206 | .378676/.584559/.698529 |
| Full/Body/Hands Diversity | 9.442641/8.934847/8.599360 | 9.409704/8.929835/8.637741 |
| Full/Body/Hands MM ↑ | .031193/.037229/.047619 | .958955/1.168498/1.330081 |
| 精确重复对比例 ↓ | 65.4364%（3599/5500） | 7.1091%（391/5500） |
| 足滑 m/s ↓ | .290324 | .294906 |
| 接触 F1 ↑ | .588763 | .582243 |

加噪主质量root audit `8c3578` exit0：H5/JSON/physical实际SHA绑定、544×1完整finite、两臂names/lengths一致、共同checkpoint及CPU评测协议相同，metadata差异限噪声/诊断字段。H5 SHA=`9ab03212d22829ae5f668a73f33305408bb5416719fa775da502536707618c7c`。Full FID +1.27%，Body -9.32%，Hands -3.45%；Full Matching +2.29%，Full R@1 -3.31个百分点，足滑 +1.58%，接触F1 -0.65个百分点。

加噪MM最终root audit `8ae18b` exit0：实际H5/JSON SHA绑定、100×11完整finite、两臂names/lengths、evaluator provenance/runtime、requests、共同ckpt和requested beta一致；SHA=`9a9f751f864940757806695d181acc775acef328b936f8923fda648b85a773d4`。平均9.78种精确不同动作/prompt。两类checkpoint均恢复大量多样性；有shape Full MM约30.74倍，无shape约26.43倍，但不是全部质量指标不降。作为质量/多样性折中结果保留，不声称严格非劣性或多seed显著性。

原采样544主质量完成；root独立验收 chunk `6276a4` exit0：H5/JSON/physical实际SHA绑定、544×1完整/finite、shape=true、checkpoint SHA及CPU joint共同协议一致。H5 SHA=`6c109ef32c25ebb85990a440c17f4f739660a61f880d717add71103fe5a1f862`。

原采样100×11 MM完成；root audit `7133c7` exit0，H5实际SHA绑定、完整finite、shape/ckpt/requested beta/seed600123核验。MM H5 SHA=`cca9a0ca379286651e4fdb48250b97eb6cbd777f0c3def91c32bfe362eaf74ca`；CPU32，平均每prompt 2.71种精确不同动作。monitor1743确认control完成，加噪主生成已到416/544；尚不能判断加噪收益。

Checkpoint：`output/t2i_trans/2026-09-02-11-49-08_hhc_historical_plus_paper_mask_ratio_raw12288_300k_seed114514_v2/model-000260000-0.1246.ckpt`。
实际 SHA256 已核验：`58b976b8cb579e4a687b9614789bbf088b73e4496eb6b6c06ef7190e80f0d072`；manifest shape=true、原 VQ175k；运行已 exact load 并保留 requested VQ。

入口 `tools/run_hhc_shape_confidence_validation.sh`，GPU1 session `55740`，monitor cell `1743` 每600秒独占回收，现均结束exit0。输出 `output/hhc_shape_confidence_validation_20260911`。原采样和加噪两臂的主质量、物理及 MM 均已完成。

两个 checkpoint 训练步数和历史不同：各自内部的采样对照成立，但跨 checkpoint 差异不能归因为 shape 单因素。
