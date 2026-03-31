# Aspect-KB 第二阶段收口总结

更新时间：2026-03-30

本文件用于汇总 Aspect-KB 检索章节当前已经冻结、可以直接复用到论文写作中的正式材料。当前阶段收口 `E6 + E7 + E8`，不提前进入 PEFT，也不把当前 fallback 接入主流程。

## 一、当前阶段已经完成什么

- `E6`：方面引导检索 vs 朴素召回，正式结果已冻结
- `E7`：reranker 消融，正式结果已冻结
- `E8`：主通道 + fallback，对照结果已冻结
- 检索主配置已经冻结为 `aspect_main_no_rerank`
- 当前 `reranker` 与 `fallback` 保留为对照/边界结果，不进入后续默认主流程

## 二、正式实验设置

- 评测 query：`40`
- 评测单元：`80` 个 `query_id + target_aspect + target_role`
- qrels：`817` 条人工标注证据
- 候选集合：同城全部 `test` 酒店
- 主指标：`@5`
- 默认检索配置：
  - `retrieval_mode = aspect_main_no_rerank`
  - `fallback_enabled = false`

## 三、E6：方面引导 vs 朴素召回

### 1. 结果表

| Group | Aspect Recall@5 | nDCG@5 | MRR@5 | Precision@5 | Avg latency (ms) |
|---|---:|---:|---:|---:|---:|
| plain_city_test_rerank | 0.7000 | 0.3307 | 0.5750 | 0.3350 | 338.847 |
| aspect_main_rerank | 0.8500 | 0.6378 | 0.7842 | 0.6375 | 333.471 |

### 2. 结论

`aspect_main_rerank` 在四个主指标上都显著优于 `plain_city_test_rerank`，且平均延迟没有明显增加。这说明 Aspect-KB 提供的方面标签与方面过滤，不只是“让检索更像按标签查表”，而是真正提升了证据相关性、首条命中质量与 Top-5 排序质量。

### 3. 可直接写入论文的案例

案例 A：`q079 / focus:quiet_sleep`

- 朴素召回时，Top-5 基本被泛化句和错方面句占据，`nDCG@5 = 0.0`
- 方面过滤后，quiet_sleep 相关句被直接提升到 Top-1，`nDCG@5 = 1.0`
- 含义：对于多方面 query，Aspect-KB 可以显著降低“语义近但方面错”的召回噪声

案例 B：`q042 / focus:service`

- `aspect_main_rerank` 相对朴素召回 `ΔnDCG@5 = 0.6608`
- 含义：`service` 这类高频方面在引入方面过滤后，能更快集中到真正可验证的服务证据，而不是落入泛化酒店好评

## 四、E7：reranker 消融

### 1. 结果表

| Group | nDCG@5 | MRR@5 | Precision@5 | Avg latency (ms) |
|---|---:|---:|---:|---:|
| aspect_main_no_rerank | 0.6457 | 0.7781 | 0.6450 | 133.256 |
| aspect_main_rerank | 0.6378 | 0.7842 | 0.6375 | 345.117 |

### 2. 结论

当前 reranker 没有带来稳定收益。它只在少量 query 上改善了首条命中位置，因此 `MRR@5` 略高；但整体 `nDCG@5` 与 `Precision@5` 反而略低，而且平均延迟约为 dense-only 的 `2.6x`。因此，当前阶段不能把 reranker 写成“已验证有效”的主配置。

这不是坏结果。它恰好说明当前实验控制了变量，并验证了一个重要边界：在已有方面过滤和固定候选集合的前提下，cross-encoder 并不一定天然带来增益。

### 3. 可直接写入论文的案例

案例 C：`q048 / avoid:value`

- reranker 把负向 value 证据 `Not worth the price per night.` 提前，`ΔnDCG@5 = 0.6309`
- 含义：reranker 在个别 `avoid` 场景里确实可能帮助捕捉方向更准确的负向证据

案例 D：`q079 / focus:service`

- dense-only 的 Top-5 基本都是强服务证据，`nDCG@5 = 0.9445`
- reranker 后混入了更泛、更弱的服务句，`nDCG@5` 降到 `0.5303`
- 含义：reranker 当前对“泛化但流畅”的句子存在偏好，不足以稳定提升整体排序质量

## 五、E8：主通道 + fallback

### 1. 结果表

| Group | nDCG@5 | Precision@5 | evidence_insufficiency_rate | fallback_activation_rate | fallback_noise_rate |
|---|---:|---:|---:|---:|---:|
| aspect_main_rerank | 0.6378 | 0.6375 | 0.0500 | 0.0000 | 0.0000 |
| aspect_main_fallback_rerank | 0.6378 | 0.6375 | 0.0500 | 0.0500 | 1.0000 |

### 2. 结论

当前 fallback 规则没有带来任何主指标收益。它只在 `4/80` 个单元上触发，而且触发后返回的句子全部是无效噪声，因此 `fallback_noise_rate = 1.0`。这说明当前问题不在“是否要兜底排序”，而在于候选集合内部本身缺乏目标方面证据。

因此，`E8` 应当作为“边界结果”写入论文，而不是作为正向收益结果写入主流程。

### 3. 代表性边界案例

案例 E：`q021 / q022 / q023 / q081` 的 `Honolulu focus:quiet_sleep`

- 触发原因一致：`sentence_count < 2` 且 `unique_reviews < 2`
- fallback 返回的多为 general / value / service / location_transport 句子
- 没有任何一条 fallback 句被 qrels 判为 relevant
- 含义：当候选集合内部本来就缺少 quiet_sleep 证据时，aspect-free fallback 不能凭空修复证据覆盖问题

## 六、当前最稳妥的章节结论

当前 `E6-E8` 可以支撑的最稳妥结论是：

1. Aspect-KB 的方面引导检索显著优于朴素召回，这是当前检索章节最强的正结果。
2. 当前 cross-encoder reranker 未表现出稳定收益，因此不应作为后续默认配置。
3. 当前 fallback 只暴露了证据稀疏边界，并未提供有效修复，因此不应接入主流程。
4. 检索主配置当前应冻结为 `aspect_main_no_rerank`，以便后续 `E3/E4/E5` 在固定后端上开展行为实验。

## 七、当前边界与风险

- `avoid` 单元整体仍弱于 `focus`，说明“问题导向证据”召回尚未稳定
- `quiet_sleep` 是当前最脆弱方面，既受主标签限制，也受 test split 覆盖稀疏影响
- 因此当前不能写成“Aspect-KB 检索链路全面优于所有基线”，而应写成“方面引导明确有效，reranker 与 fallback 在当前设置下未验证出额外收益”

## 八、下一阶段顺序

建议固定为：

1. 将本文件与三个正式 run 的 `summary.csv / analysis.md` 整理进论文检索章节
2. 冻结系统默认检索配置为 `aspect_main_no_rerank`
3. 启动 `E3`：偏好解析
4. 启动 `E4`：澄清触发
5. 启动 `E5`：中文输入 -> 英文检索表达桥接

当前不建议启动：

- PEFT
- `G1-G4`
- 将当前 fallback 方案接入主流程
