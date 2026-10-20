# 官方 InterGen 切换后的监控与指标归属修复

2026-10-17 23:17 SGT。本轮没有改变任何训练或已绑定实验源码，修复的是旧监控任务与状态清单尚未跟上官方基线切换的问题。

## 发现与实际改动

两项既有automation：hhc（HHC 训练与网盘备份）和hhc-runpod（HHC Runpod闲置保护）均处于PAUSED，但原提示仍命令用local_runpod_control_fast.py/runpod_fast_bootstrap.sh恢复自定义旋转/FK实验，并采用已废弃的HHC after-fast门控。旧run已经按用户要求保存103000并暂停，故这些提示需要更新。

已通过应用工具automation_update的update模式修改原id，使用当前官方训练、final704000完整验收、未改原始20rep/MM以及真实独立review门控。25145负责最终checkpoint验收，26133随后评测，HHC恢复仍由25145在实际review通过后执行；提示明确不自行重放launch intent、不生成虚假review收据、不调用旧fast恢复器。未手写automation TOML或创建新任务。

更新前后逐字段解析本机TOML：仅prompt及updated_at变化，name、kind、id、status=PAUSED、原频率、原target_thread_id、created_at全部精确保留。新提示为实际换行，Unicode replacement字符0。原目标聊天保持01a13b25-20f2-7ab3-ba8e-112909f02e6f，没有将任务重定向到本goal聊天。

[官方Scheduled文档](https://learn.chatgpt.com/docs/automations?surface=app)说明可在聊天内更新既有任务；本次实际效果以工具返回和本机TOML完整读回为证据。按OpenAI Docs技能核对了官方页面，没有因技能要求新增批准流程或启用调度。

## 指标与owner归属

本机migration/control/compute-owner.json的run/active_experiment已经是官方baseline，但last_validation_fid=35.23905861184335和best_validation_fid=24.314142879586825仍来自旧custom固定480val。current_hhc_waiter_pid为空、hhc_waiter_control也指旧fast。这些字段会造成跨实验误报。

已把4个旧last/best step/FID原值保存在previous_custom_validation_metrics，并明确run为旧rotation_scratch_runpod_20261015、saved103000、固定480val。官方run的顶层last/best validation及baseline_fid均为null，表示没有已完成评测。没有删掉或改写旧成绩、将训练loss冒充FID，亦没有声称官方复现达到论文约6。

本机状态同时更新为真实HHC等待25145/birth468066055、官方eval等待26133/birth468171108、observer21986/birth467800611，以及当前control/config路径。last_checkpoint_step704及SHA仍明确指已CPU验收的startup snapshot；最近训练checkpoint的文件名/bytes另列为观察记录，没有重新打开或声称完整验收最新权重。

本机runpod-idle.json从旧2353/旧ABC历史记录更新为官方9780及四个CPU等待器的实际观察，idle_since=null、keep_running、resource_action_executed=false。纯等待器和观察器本身不算活跃训练。既有30分钟真实全闲置后stop并保留卷的规则未改变。

这些metadata文件实际仅在本机控制目录。首次读取远端migration/compute-owner.json返回FileNotFoundError，核实两个远端副本均不存在后，仅更新本机原文件；未凭空创建远端控制状态。远端证据包只备份原始数据，不用作实时启动或放行依据。冻结的旧hhc-compute-priority.json仍是现行队列pin的一部分，未改其字节。

## 验收

- 应用两次update均返回status PAUSED，本机before/after解析仅两字段改变。
- 新hhc prompt SHA5756898ca87142ff65718837e896ea7979795a8e8d436b7aabdb1f1ad21c7e11。
- 新idle prompt SHA5e7c81817a6607a3d15fca6b6d447de8fbe4dea4b2f7f9f1aff0db95b7b80422。
- 完整新请求migration/control/official-intergen-automation-updates.json，13351B/SHAdf5d6ad8dc086d1029cb6ad32e15e588e52058c0579df8f34a3d37d699b29664。
- owner3437B/SHAf5e0f8596253f30435342bd7d3c326d17341eebe969037ec251b32b6369ba558；idle3419B/SHA021f59a27ec3e89c14639ab909301b77b804244c0acce5e1b33e270e75907242。
- 两项现行queue config、官方priority、两queue源码及训练wrapper共6项源/config，本机及云端SHA均仍与此前绑定值相同。
- /proc的22675、25145、26133、21986均与实际birth/命令一致、S；25145/26133均waiting，无phase/child/intent；8321/9975也已实际验明纯等待。Windows旧三个localcontroller文件名匹配的Python进程数0。
- 原始证据official_monitor_transition_before_20261017.json、official_monitor_automations_after_20261017.json、official_monitor_metadata_after_20261017.json、official_monitor_transition_acceptance_20261017.json。

15:09:23 UTC固定observer官方epoch27/9780、LR1e-4/total.185，GPU只有22675/7880MiB；未有FID。下一普通训练观察≥15:19:23 UTC，异常/验收另核。HHC仍保存4000，所有GPU工作仅云端，本机checkpoint/生成或token数组新增0，未停机/重启/新建资源。此次修复不构成质量提升，HHC真实选模、正式生成test/MM与SOTA仍未验证。
