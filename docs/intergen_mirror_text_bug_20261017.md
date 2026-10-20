# InterGen 镜像文本替换 bug（2026-10-17）

本轮新增确认：镜像文本的顺序替换先处理 `clockwise`，会把 `counterclockwise` 内的子串一起替换，最终产生 `countercounterclockwise`，没有正确翻转为 `clockwise`。该问题同时存在于固定官方提交、自定义旋转数据集和 HHC adapter，**不能作为自定义实验相对官方 FID 差距的解释**。

## 实际影响与验证

云端 CPU 扫描当前全部 5637 个可训练交互的 16911 条描述，其中 60 个交互、150 条描述含该问题。按均匀交互、均匀镜像位、交互内均匀描述采样计算，错误描述约占训练抽样的 0.4435%；这是采样规则下的概率，并非读取运行日志实测的出现频率。

直接从三处真实源码的 AST 抽取文本表达式执行；六个合成用例与全部训练描述均验证三处结果一致。例如：`counterclockwise` → 原表达式 `countercounterclockwise`，离线候选 `clockwise`。候选只把两个旋转方向改为一次正则替换，保留历史 left/right 行为；六个用例验证了方向结果及镜像两次恢复。未声称修复大小写、其他拼写或全部语言现象。

审计工具：`tools/audit_intergen_mirror_text_cpu_cloud.py`，6830B，SHA256 `bbe45c2e67579223cd886b52574933f0e5d75b76db80370bd75a1315070836b6`。云端原执行会话 97960，SSH 和审计均 exit 0；未加载 checkpoint、未初始化 CUDA。完整源码上传后按大小/SHA 读回，结果 JSON 7741B/SHA256 `301f4defbf92a6ba7a75d9d62a9a64a360408cdc77e1e7d02ca315aa68211c73`，本机取得同一原始字节后独立验证。证据位于 `output/hhc_sota_20261016/intergen_mirror_text_{cpu,acceptance,upload}_20261017.json`。

## 当前运行保持与其余问题

没有改动任何训练绑定源，也没有把候选接入官方或 HHC。官方复现保持原始增强行为；以后验证此修正时应作为独立消融，不能暗中改入基线。新独立 SSH 比较确认当前官方目录的 29 个 Python 文件全部逐字节等于 `b1750df4a859443b8a69329f759ce3a7503ae179`，激活重计算包装 SHA 与已验收版本相同。

实际 `/proc` 确认官方 trainer 22675/birth467827326、HHC waiter 25145/birth468066055、eval waiter 26133/birth468171108 存活；两个等待器仍为 waiting_for_official_training。该核查没有重新测训练步数、FID 或 GPU 利用率。旧自定义实验暂停、HHC saved4000 的策略不变，没有新增本机权重或数组。

本轮亦复核了已确认问题的源码与原 CPU 报告，而未重复运行相同测试：稀疏距离/亲和分母、接触按人归一化、变长序列 reduction，以及训练强 FK/评测弱位置头的口径差异。前三项有数值反例；位置头差异属于已量化的设计不一致，不能据此断言解释全部退化。稀疏修正版 v2 仍只修空间分母，不能视为完整官方 loss。详见 `docs/intergen_bug_audit_20261017.md` 与 `docs/intergen_temporal_reduction_20261017.md`。

证据备份：`intergen_mirror_text_bug_evidence_20261017.zip`，18096B/SHA256 `92107c5dadf4990bee59ea06210591d90e05880671b0d8a63909c2195fb8c98d`，云端整包及 7 个成员验证通过；Drive 文件 `1EuKxlZWnqWhdJRzTjoHkIWvqQPZ6vhK6` 的 id/大小/父目录读回一致，API 未返回 checksum。仅源码、小 JSON 与文档；包内文档为加上本段备份说明之前的快照，未含权重或数组。
