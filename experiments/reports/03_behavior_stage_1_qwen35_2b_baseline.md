# Qwen3.5-2B 行为实验第一阶段基线归档

更新时间：2026-03-31

本文件用于冻结当前 `Qwen/Qwen3.5-2B` 在 `E3/E4` 上的第一轮正式结果。该轮结果保留为“弱基线 / 负结果”对照，不删除、不覆盖，后续所有 `v2` 优化都必须在新的 run 目录中追加。

## 1. 本轮基线配置

- 模型：`Qwen/Qwen3.5-2B`
- 调用方式：`api`
- 推理模式：`non-thinking`
- 温度：`0`
- 默认检索后端：`aspect_main_no_rerank`
- fallback：`false`
- `E3` run：`experiments/runs/e3_244aca8abf6345ad_20260331T072527+0000/`
- `E4` run：`experiments/runs/e4_4a15a89128a90d11_20260331T073016+0000/`

## 2. E3 指标摘要

| Group | Exact-Match | Unsupported Recall | City Slot F1 | Focus Slot F1 | Unsupported Slot F1 |
|---|---:|---:|---:|---:|---:|
| `A_rule_parser` | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| `B_base_llm_structured` | 0.5698 | 0.0667 | 0.9186 | 0.8258 | 0.1250 |

当前解读：

- `2B` 在 `city` 与核心 `focus_aspects` 上已经有一定解析能力，不能视为完全不可用。
- 但 `unsupported_requests` 几乎失效，说明它会把预算、距离、入住日期等条件吸收到已支持语义里。
- 因此当前 `2B` 只能作为“轻量模型能力下界”保留，不能直接作为正式可用主模型。

## 3. E4 指标摘要

| Group | Accuracy | Precision | Recall | F1 | Under-clarification |
|---|---:|---:|---:|---:|---:|
| `A_rule_clarify` | 0.9767 | 0.8889 | 1.0000 | 0.9412 | 0.0000 |
| `B_base_llm_clarify` | 0.8140 | 0.0000 | 0.0000 | 0.0000 | 1.0000 |

当前解读：

- `B_base_llm_clarify` 在 16 条应澄清正例上全部判成 `clarify_needed=false`。
- 这说明问题不在 JSON 格式，而在于 `v1` prompt 让小模型直接塌缩到了默认负类。
- 因此当前 `E4` 结果应明确写成“负结果”，不能直接复用到论文主结论中。

## 4. 代表性失败案例

### 案例 A：`E3` 城市其实识别对了，但被后处理误伤

- 代表 query：`q002 / q003`
- 现象：模型会输出 `Anaheim:CA`、`New Orleans:LA`
- 当前问题：`v1` 后处理只接受精确城市名，导致这些值被清空成 `city = null`
- 结论：这是后处理过严，不是纯语义失败

### 案例 B：`E3` 把 unsupported 条件吸收到已支持方面

- 代表 query：`q004`
- 原始 query：`帮我找Anaheim预算在 600 元以内，而且位置交通不错的酒店。`
- 现象：模型更容易输出 `value` 或 `location_transport`，但漏掉 `unsupported_requests=["budget"]`
- 结论：`E3 v2` 必须显式减轻任务负担，并把 unsupported 标签口径写死

### 案例 C：`E4` 的 `missing_city` 正例全漏判

- 代表 query：`q051 / q052`
- 金标：`clarify_needed = true`
- 基线输出：`clarify_needed = false`
- 结论：`E4 v1` prompt 对 `2B` 来说过于抽象，必须改成“先分三类，再映射字段”的分类优先设计

### 案例 D：`E4` 的 `aspect_conflict` 正例同样全漏判

- 代表 query：`q057 / q058`
- 金标：`clarify_needed = true`, `clarify_reason = aspect_conflict`
- 基线输出：空问题 + `false`
- 结论：小模型没有稳定学会“focus 与 avoid 同方面冲突必须澄清”的决策边界

## 5. 当前最稳妥结论

1. `Qwen3.5-2B` 可以作为 `E3/E4` 行为实验的弱基线保留。
2. 当前 `E3` 结果说明小模型已有初步解析能力，但后处理与 unsupported 识别仍明显不足。
3. 当前 `E4` 结果属于明确负结果，不能直接写成可用方案。
4. 下一步应优先做 `E3/E4 v2`，先在冻结诊断子集上验证，再决定是否继续用 `2B` 跑全量，还是切换到 `4B / 9B`。

## 6. 基线审计快照

当前全局 `E4` 审计文件会在后续 rerun 时刷新，因此本轮额外冻结：

- `experiments/labels/e4_clarification/clarification_question_audit_e4_4a15a89128a90d11_baseline.csv`

后续约定：

- 每轮新的 `E4` run 都在自己的 run 目录内写 `clarification_question_audit.csv`
- `experiments/labels/e4_clarification/clarification_question_audit.csv` 只保留最新副本，方便人工继续填写
