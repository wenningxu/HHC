# InterHuman 前沿方法：初始化与公开评测源码核验

2026-10-17 23:01 SGT。整体目标仍为真实改进HHC生成指标、完成指定方法与当前可比前沿的比较；本次只消除一部分输入条件不确定性，没有新增生成质量结果。

| 方法 | 本轮取得的直接证据 | 仍缺少的证据 |
|---|---|---|
| HINT | 作者OpenReview note和supplementary字段、arXiv正文 | 附件端点403，未读代码；初始history是否GT未知 |
| ARMS默认InterHuman评测入口 | 固定源码只向生成器传文本、长度、采样设置；默认context为空，首窗噪声，后窗取自己生成历史；CPU字面AST控制流通过 | 真实checkpoint与resolved训练配置、真实VAE/网络推理和统一20rep/MM |
| ARMS可选MM | 默认关闭；启用时CPU替身执行原wrapper，batch2/repeat3索引越界 | 后续独立prompt/repeat导出器及真实100×30 MM |

## HINT：代码入口已找到，初始化仍未验证

[HINT正文](https://arxiv.org/html/2601.20383v1)说明代码随补充材料提供。OpenReview公共搜索精确匹配标题取得作者note CGALrQfkYi，其中supplementary_material值为/attachment/79a8c937cf24c5be1621fc10a5eff9d64079e571.zip。按[官方API文档](https://docs.openreview.net/reference/api-v2/openapi-definition)请求attachment?id=CGALrQfkYi&name=supplementary_material，API及网页端HEAD均403；直接把存储字段拼作网页路径则404。未下载附件，没有绕过访问限制，公共搜索也未取得有关初始化的作者讨论。

公开arXiv和作者实验室页面没有提供已取得的代码仓库链接。没有据此断言所有渠道都没有代码。HINT的3.100±.035继续标为论文online/history条件记录、初始prefix未知；不只引用其他论文的GT-first-frame标签认定实现事实，也不从前沿讨论中删掉这个更低分。

原始来源保存在hint_primary_initialization_routes_20261017.json、hint_openreview_attachment_metadata_20261017.json、hint_openreview_public_discussion_search_20261017.json。最初全文搜索返回宽泛OR结果；仅保留精确标题note，其他论文记录已剔除，响应SHA仍说明原始HTTP响应，不能将过滤后JSON当作原响应字节。

## ARMS：默认评测入口没有GT前缀

[论文](https://arxiv.org/html/2607.05733v1)给出的[官方仓库](https://github.com/kk9six/arms)固定为e3819179209d9f3d4b717673af4f839f5b2bdf44；从commit取精确tree47d09f05abc3fe5a5a50c3cdcafa34163642fb23，对12份源码逐一核验git blob SHA1及文件SHA256。没有下载仓库中的二进制。

[EvaluationDataset的调用](https://github.com/kk9six/arms/blob/e3819179209d9f3d4b717673af4f839f5b2bdf44/model/evaluator.py#L58)只有text、ids_length、cond_scale及采样kwargs，不传GT动作或context。[DiT.generate](https://github.com/kk9six/arms/blob/e3819179209d9f3d4b717673af4f839f5b2bdf44/model/dit.py#L211)的context1/context2默认None；curr_L=0，首段使用torch.randn，后段历史来自刚生成的x1/x2。该代码仍允许其他调用者显式传context，因此结论仅限这里检查的InterHuman默认评测入口，不泛化为任意demo或任意调用。

独立工具audit_arms_eval_contract_cpu_cloud.py直接抽取并执行原EvaluationDataset类、DiT.generate方法及mask/schedule helpers的AST。tiny CPU、无真实网络/权重，替换GT动作数值后模型调用参数和替身输出相同；实际generate首段没有context，第二窗context与前窗输出一致。sampler trace用CFG1及简化尺寸，wrapper路由传CFG2.5；未验证真实CFG2.5神经网络行为、GPU执行、VAE/坐标转换数值或生成质量。decoder/denoiser/condition encoder/坐标转换均明确用替身，不把此结果冒称端到端复现。

作者论文InterHuman full-window4.436±.069与streaming4.444±.068继续分列文献；两个对应source preset与联合HumanML3D+InterHuman without-shape demo不是同一个训练设置。HF README列出kksix/arms入口，但本次未认证metadata及tree请求401，未获取真实checkpoint或训练config。云hhc-env没有huggingface_hub包，本轮没有查询或使用认证token；401不能证明repo不存在。实际权重验身份、训练数据范围、预处理/归一化及统一请求生成仍待完成。

## 公开MM分支不能直接复用

[配置](https://github.com/kk9six/arms/blob/e3819179209d9f3d4b717673af4f839f5b2bdf44/config/eval_interhuman.yaml#L3)默认mm_num_samples=0；[evaluation](https://github.com/kk9six/arms/blob/e3819179209d9f3d4b717673af4f839f5b2bdf44/eval/eval.py#L148)也注释了evaluate_multimodality及其汇总。开启此可选路径后，原wrapper将repeat-major输出拼成B×R行，却继续用原B条motion_lens/text索引；batch2、repeat3真实控制流报index2超出size2。此外i in mm_idxs判断的是batch编号，非单prompt编号。

这不说明论文FID错，也没有证据论文使用该可选MM实现。未来如纳入ARMS共同100×30 MM，须从固定prompt/repeat请求独立导出并检查归属，不能直接改一个MM开关。当前3方法正式exporter/HHC选模队列均不改。

## 验收、失败和当前计算

第一轮90937退出1，因诊断feature-conversion替身返回NumPy、原wrapper需要Tensor；不是公共模型失败。已保留失败源码和sidecar，替身修正后8586退出0。第一次独立SHA验收29778因本机重序列化JSON产生CRLF而退出1；解析内容相同，保留副本并改为云原始字节后独立验收通过，没有更改数值。所有失败均不作为方法质量结果。

- 有效CPU报告3701B/SHA099658693ede1f8e8255974d116895b93b8dda61886271c9da5b919a3f97b93a。
- 工具10051B/SHA707223e3ca35fd353645aae1db90160991b9f0367caaa90df41c8c86141ec36f。
- 冻结源码元数据193344B/SHAb07324951445ad5ae96ccd71d7884fd6df8dbbef5a29178eca06117cbc92a527。
- 结构化前沿清单recent_interhuman_frontier_source_audit_20261017.json，5658B/SHA49cc285a3fc6b8cc5e5a4abfac795c3b73ec1272fead3a6502df10ec69c9468c。

GPU仍只有官方InterGen trainer22675。14:58:06 UTC观察epoch25/8920、LR1e-4、total.1623，显存7880MiB；这是训练loss而非FID。HHC保持4000等待，两项automation PAUSED，无新GPU任务、本机checkpoint/数组下载。下一普通观察≥15:08:06 UTC。整体goal active、SOTA尚未验证。
