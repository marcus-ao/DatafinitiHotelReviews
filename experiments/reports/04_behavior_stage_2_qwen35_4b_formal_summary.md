# Qwen3.5-4B 行为实验第二阶段正式结果归档

更新时间：2026-03-31

本文件用于冻结当前 `Qwen/Qwen3.5-4B` 在 `E3/E4` 上的正式结果，并将其与前面的 `Qwen3.5-2B` baseline 和中间诊断 run 串成一条完整的行为实验演进线。当前阶段结论是：`2B` 适合作为弱基线保留，`4B` 已经足够支撑 `E3/E4` 的正式论文结果。

## 1. 当前冻结范围

本阶段保留并引用这几组行为实验结果：

- `2B baseline`
  - `experiments/runs/e3_244aca8abf6345ad_20260331T072527+0000/`
  - `experiments/runs/e4_4a15a89128a90d11_20260331T073016+0000/`
- `2B v2 diagnostic`
  - `experiments/runs/e3_da541f84770ed8ed_20260331T090311+0000/`
  - `experiments/runs/e4_96e0e4afb24dab2d_20260331T091021+0000/`
- `4B diagnostic`
  - `experiments/runs/e3_f62d907e600cfc14_20260331T120756+0000/`
  - `experiments/runs/e4_f928a37444c1bf52_20260331T121012+0000/`
- `4B formal full`
  - `experiments/runs/e3_14928d821d811e86_20260331T122611+0000/`
  - `experiments/runs/e4_55c8021e1119fb77_20260331T122648+0000/`

当前正式主模型固定为：

- 模型：`Qwen/Qwen3.5-4B`
- 推理方式：`api`
- 模式：`non-thinking`
- 温度：`0`
- 默认检索后端：`aspect_main_no_rerank`
- fallback：`false`

## 2. E3 正式结果

### 2.1 正式结果表

| Group | Query Count | Exact-Match | Unsupported Recall | City Slot F1 | Focus Slot F1 | Avoid Slot F1 | Unsupported Slot F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `A_rule_parser` | 86 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| `B_base_llm_structured` | 86 | 0.9767 | 1.0000 | 1.0000 | 1.0000 | 0.9474 | 0.9836 |

正式 run：

- `experiments/runs/e3_14928d821d811e86_20260331T122611+0000/`

### 2.2 相比 2B baseline 的提升

| Model / Run | Exact-Match | Unsupported Recall | City Slot F1 | Focus Slot F1 | Avoid Slot F1 | Unsupported Slot F1 |
|---|---:|---:|---:|---:|---:|---:|
| `2B baseline full` | 0.5698 | 0.0667 | 0.9186 | 0.8258 | 1.0000 | 0.1250 |
| `4B formal full` | 0.9767 | 1.0000 | 1.0000 | 1.0000 | 0.9474 | 0.9836 |

当前解读：

- `4B` 已经把 `city`、`focus_aspects` 和 `unsupported_requests` 三个最关键的行为槽位基本做稳。
- 之前 `2B` 最明显的两个问题：
  - `City:ST` 后处理误伤
  - unsupported 被吸收到已支持语义
  已经在 `4B` 上被实质性解决。
- 当前 `E3` 只剩极少数边界错例，已经足以作为正式结果写入论文。

### 2.3 剩余边界案例

案例 A：`q048`

- query：`我在Seattle想住得安静一点，但不要性价比太差的酒店。`
- gold：`avoid_aspects=["value"]`
- `4B` 预测：`avoid_aspects=[]`, `unsupported_requests=["value"]`
- 含义：`value` 的负向约束在少数情况下仍会被错当成 unsupported 约束

案例 B：`q062`

- query：`我想在New Orleans找一家性价比很好，但又最好别太强调性价比的酒店。`
- gold：`focus=["value"], avoid=["value"]`
- `4B` 预测：`focus=["value"], avoid=[]`, `unsupported=["budget"]`
- 含义：`value` 同时作为 focus + avoid 的冲突型表达仍然是 `E3` 最脆弱的残留点

## 3. E4 正式结果

### 3.1 正式结果表

| Group | Query Count | Accuracy | Precision | Recall | F1 | Over-Clarification | Under-Clarification |
|---|---:|---:|---:|---:|---:|---:|---:|
| `A_rule_clarify` | 86 | 0.9767 | 0.8889 | 1.0000 | 0.9412 | 0.0286 | 0.0000 |
| `B_base_llm_clarify` | 86 | 0.9884 | 0.9412 | 1.0000 | 0.9697 | 0.0143 | 0.0000 |

正式 run：

- `experiments/runs/e4_55c8021e1119fb77_20260331T122648+0000/`

### 3.2 相比 2B baseline 的提升

| Model / Run | Accuracy | Precision | Recall | F1 | Over-Clarification | Under-Clarification |
|---|---:|---:|---:|---:|---:|---:|
| `2B baseline full` | 0.8140 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 |
| `4B formal full` | 0.9884 | 0.9412 | 1.0000 | 0.9697 | 0.0143 | 0.0000 |

当前解读：

- `2B` 在 `E4` 上属于明确负结果，而 `4B` 已经达到正式可用水平。
- `4B` 不仅解决了“全判 false”的塌缩，还在全量上比规则组少一个误澄清，因此主指标略优于规则组。
- 当前 `E4` 已经可以作为正式论文结果写入，不需要继续依赖 `2B` 作为主模型。

### 3.3 剩余边界案例

案例 C：`q013`

- query：`我在Chicago想住得安静一点，但不要安静睡眠太差的酒店。`
- gold：不需要澄清
- 规则组：误判为 conflict
- `4B`：仍然误判为 conflict
- 含义：当前 gold 口径下，这类“同一方面的正负修饰”仍然容易被系统误看成冲突

案例 D：`q043`

- query：`我在San Francisco想住得安静一点，但不要安静睡眠太差的酒店。`
- gold：不需要澄清
- 规则组：误判为 conflict
- `4B`：正确判为 `none`
- 含义：`4B` 已经开始在部分“表面像冲突、实际不是冲突”的句子上优于规则组

## 4. 诊断 run 的作用与最终定位

中间两轮诊断 run 的作用已经完成：

- `2B v2 diagnostic`
  - 证明 `v2` prompt 与后处理方向是对的
  - 但 `E4` 召回仍不足，因此不能继续把 `2B` 当成主模型
- `4B diagnostic`
  - 证明在完全相同的 `v2` 设计下，`4B` 已经可以稳定掌握当前结构化行为任务

因此当前最稳妥的模型定位是：

1. `Qwen3.5-2B`：弱基线 / 下界模型
2. `Qwen3.5-4B`：当前正式行为实验主模型
3. `Qwen3.5-9B`：可选扩展或附加对比，不是当前论文主线阻塞项

## 5. 当前最稳妥的章节结论

当前 `E3/E4` 能支撑的最稳妥结论是：

1. 在固定检索后端 `aspect_main_no_rerank` 下，`4B` 已经可以稳定完成偏好解析与澄清触发两类行为任务。
2. `2B` 可以作为行为层弱基线，体现轻量模型的能力下界，但不适合作为正式主模型。
3. `E3` 的主要残留问题已经收缩到极少数 `value` 负向约束边界例。
4. `E4` 在全量上已经达到正式可写论文的水平，并在误澄清率上略优于规则组。

## 6. 当前归档文件

本轮额外冻结：

- `experiments/labels/e4_clarification/clarification_question_audit_e4_55c8021e1119fb77_qwen35_4b_full.csv`

当前最值得优先引用的材料：

- `experiments/reports/03_behavior_stage_1_qwen35_2b_baseline.md`
- `experiments/reports/04_behavior_stage_2_qwen35_4b_formal_summary.md`
- `experiments/runs/e3_14928d821d811e86_20260331T122611+0000/analysis.md`
- `experiments/runs/e4_55c8021e1119fb77_20260331T122648+0000/analysis.md`
