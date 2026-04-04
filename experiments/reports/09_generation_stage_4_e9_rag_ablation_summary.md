# 生成阶段第四轮正式材料：E9 有无 RAG 正式对比汇总

更新时间：2026-04-04

本文件用于冻结 `E9` 的正式有无 RAG 对比结果。当前正式 run 为：

- 四组正式 run：
  - `experiments/runs/e9_8449c12a50585e42_20260404T081010+0000/`
- 早期 `E9` 第二轮正式 run 继续保留为生成稳定性冻结材料：
  - `experiments/runs/e9_ecbcdbab690dc503_20260401T025012+0000/`

本轮主比较固定为：

- with RAG：
  - `B_grounded_generation`
- without RAG：
  - `D_no_evidence_generation`

`A_free_generation` 与 `C_grounded_generation_with_verifier` 继续保留在同一 run 中，但只作为上下文组，不作为本文件的主 compare 结论主体。

## 1. 一句话摘要

在统一 `Qwen/Qwen3.5-4B`、统一 `api` 后端、统一 `aspect_main_no_rerank` 与统一 `E2 B_final_aspect_score Top5` 候选集的前提下，引入 RAG 后系统的主要收益不是平凡地提高 citation，而是**更稳定地保留可审计推荐**：`B_grounded_generation` 的 `recommendation_coverage = 0.95`，显著高于 `D_no_evidence_generation` 的 `0.825`，同时 `schema_valid_rate` 也从 `0.975` 提升到 `1.0`。

## 2. 当前正式结果表

| Group | Query Count | Citation Precision | Evidence Verifiability Mean | Unsupported Honesty Rate | Schema Valid Rate | Recommendation Coverage | Avg Latency (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|
| `A_free_generation` | 40 | 0.9437 | 1.9773 | 1.0000 | 1.0000 | 0.9500 | 2180.300 |
| `B_grounded_generation` | 40 | 0.9500 | 1.9924 | 1.0000 | 1.0000 | 0.9500 | 2182.405 |
| `C_grounded_generation_with_verifier` | 40 | 0.9500 | 1.9924 | 1.0000 | 1.0000 | 0.9500 | 2185.010 |
| `D_no_evidence_generation` | 40 | 0.0000 | 0.0000 | 1.0000 | 0.9750 | 0.8250 | 1052.184 |

### 2.1 主 compare：`B` vs `D`

| Compare | Recommendation Coverage | Schema Valid Rate | Citation Precision | Evidence Verifiability Mean |
|---|---:|---:|---:|---:|
| `B_grounded_generation` | 0.9500 | 1.0000 | 0.9500 | 1.9924 |
| `D_no_evidence_generation` | 0.8250 | 0.9750 | 0.0000 | 0.0000 |
| Delta (`B - D`) | +0.1250 | +0.0250 | +0.9500 | +1.9924 |

解释口径固定为：

- 主指标：
  - `recommendation_coverage`
  - `schema_valid_rate`
- 辅助指标：
  - `citation_precision`
  - `evidence_verifiability_mean`

原因：

- `D_no_evidence_generation` 本来就不看证据，因此 citation / evidence 指标只作辅助解释，不能把它们写成主结论。

## 3. 样本级对比结论

### 3.1 总体分布

在 `40` 条 query 中：

- `B` 的推荐数高于 `D`：
  - `28` 条
- `B` 与 `D` 推荐数相同：
  - `11` 条
- 表面上 `D` 推荐数高于 `B`：
  - `1` 条

推荐总数对比：

- `B_grounded_generation`
  - `68`
- `D_no_evidence_generation`
  - `39`

这说明 no-RAG 的主要问题不是偶发 citation 失误，而是**系统性地更保守、更容易减少推荐覆盖面**。

### 3.2 代表性 recovery cases

最有代表性的 `B > 0` 且 `D = 0` 样本包括：

- `q003`
- `q008`
- `q013`
- `q033`
- `q043`
- `q081`

这些样本的共同模式是：

- `B` 组能够基于当前 `EvidencePack` 保留 `1-2` 家 grounded 推荐
- `D` 组在没有证据时更容易退回到 `0 recommendation`

这正是本轮最核心的 RAG 主效应：

> RAG 的主要价值不只是让理由“有来源”，而是让系统在复杂偏好 query 上更稳定地保留可审计推荐。

### 3.3 matched abstention：`q023`

`q023` 是本轮最重要的 matched-abstain 例子：

- `B` 组：
  - `0 recommendation`
  - 明确指出缺少安静睡眠证据
- `D` 组：
  - `0 recommendation`
  - 明确指出当前没有评论证据

含义：

- 不是所有空输出都应视为系统失败
- 当前 grounded 流程在 evidence gap 情况下能够诚实 abstain

### 3.4 看似 no-RAG 胜出的 `q021` 实际无效

本轮唯一一个表面上 `D > B` 的 query 是：

- `q021`

但深入核查 `results.jsonl` 可见：

- `D_no_evidence_generation`
  - `recommendations = 1`
  - `schema_valid = False`
  - `response_error_type = schema_invalid`

因此：

- `q021` 不应被视为有效的 no-RAG 正向样本
- 严格按有效输出计算，本轮 **没有有效的 no-RAG win**

## 4. 这轮结果对旧 `E9` 主结论的影响

这轮新增 `D` 组后，原先 `A/B/C` 三组的正式质量没有被破坏，反而略有提升：

- `B_grounded_generation`
  - `citation_precision`
    - `0.9437 -> 0.9500`
- `C_grounded_generation_with_verifier`
  - `citation_precision`
    - `0.9250 -> 0.9500`
  - `retry_trigger_rate`
    - `0.025 -> 0.000`
  - `fallback_to_honest_notice_rate`
    - `0.025 -> 0.000`

因此当前最稳妥的解释是：

- 新增 `D` 组与 `B vs D` compare 并没有破坏既有 `E9` 正式结果
- 它只是把“有无 RAG”这一部分补成了可以单独写进论文的正式证据

## 5. 当前正式结论

当前 `E9` 的论文表述建议固定为：

1. `E9` 当前已经具备两层正式结论：
   - 生成层已达到可审计、可冻结状态
   - 有无 RAG 的正式主比较已经补齐
2. `B_grounded_generation` 相比 `D_no_evidence_generation` 的主要优势是：
   - 更高的 `recommendation_coverage`
   - 更稳定的 `schema_valid_rate`
3. `citation_precision` 与 `evidence_verifiability_mean` 在 `B vs D` 中只作辅助解释，不作为主结论，因为 `D` 本来就不看证据。
4. `q003 / q008 / q013 / q033 / q043 / q081` 可作为代表性 recovery cases。
5. `q021` 不是有效的 no-RAG 胜例，因为其 `D` 组输出本身是 `schema_invalid`。

## 6. 后续使用方式

后续文档与论文中建议这样引用：

- 若要说明 `E9` 生成层总体已经稳定：
  - 引用 `06_generation_stage_1_e9_formal_summary.md`
- 若要说明“有无 RAG”主效应已经补齐：
  - 引用本文件

这两份材料共同构成当前 `E9` 的正式冻结口径。
