# 生成阶段第二轮正式材料：E10 v1 Base vs PEFT 正式结果汇总

更新时间：2026-04-02

本文件用于冻结 `E10 v1` 的正式对照结果，并给出下一阶段 `E10 v2` 的数据方案动机。当前正式结果固定为：

- base formal：
  - `experiments/runs/e10_0dc5c2e6f867c66f_20260402T015230+0000/`
- peft formal：
  - `experiments/runs/e10_0ef381420c1bd19a_20260402T020120+0000/`
- compare formal：
  - `experiments/runs/e10cmp_28598dfb8434c1ba_20260402T020734+0000/`

## 1. 一句话摘要

在统一的 `local/HF` 推理后端、统一冻结 `E9` eval units、统一 `Qwen3.5-4B` 基座与统一输出协议下，`PEFT exp01` 没有优于 base；其主要退化集中在 `quiet_sleep` 与 `focus+avoid` 约束样本上，说明当前 v1 训练目标与最终 grounded recommendation 任务之间存在明显错位。

## 2. 正式结果表

| Group | Query Count | Citation Precision | Evidence Verifiability Mean | Schema Valid Rate | Reasoning Leak Rate | Auditable Query Rate | Avg Latency (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|
| `A_base_4b_grounded` | 40 | 0.9688 | 1.9704 | 1.0000 | 0.0000 | 1.0000 | 5776.447 |
| `B_peft_4b_grounded` | 40 | 0.9250 | 1.9915 | 1.0000 | 0.0000 | 0.9750 | 5440.052 |

当前正式 compare 结果补充：

- `latency_formally_comparable = yes`
- `citation_precision`：
  - base 更好，差值为 `+0.0438`
- `evidence_verifiability_mean`：
  - PEFT 略高，差值为 `+0.0211`
- `schema_valid_rate`：
  - 两组都为 `1.0`
- `reasoning_leak_rate`：
  - 两组都为 `0.0`

## 3. 样本级对比结论

### 3.1 结果模式

- 共 `40` 条 query 中：
  - 明确改进：`1` 条
  - 明确退化：`2` 条
  - 其余 `37` 条在主指标上无变化
- 推荐数量对比：
  - `(2 -> 2)`：`26` 条
  - `(1 -> 1)`：`7` 条
  - `(2 -> 1)`：`4` 条
  - `(1 -> 0)`：`2` 条
  - `(0 -> 0)`：`1` 条

这说明 `PEFT exp01` 的主要变化不是格式崩坏，而是**在部分约束 query 上更保守，容易减少推荐覆盖面**。

### 3.2 代表性改进：`q079`

- base：
  - `citation_precision = 0.75`
- peft：
  - `citation_precision = 1.0`

含义：

- PEFT 在个别多方面均衡 query 上学会了更稳定地选择 pack 内合法 citation。
- 这证明 PEFT 并非全面退化，而是对某些 grounded citation 细节确实有修正能力。

### 3.3 代表性退化：`q013` 与 `q023`

`q013`

- 类型：
  - `quiet_sleep` 相关约束 query
- 现象：
  - base 保留了 `1` 条可验证推荐
  - peft 直接收缩为 `0` 推荐并输出更强的证据不足说明

`q023`

- 类型：
  - `focus quiet_sleep + avoid location_transport`
- 现象：
  - base 提供了 `1` 家可验证推荐
  - peft 收缩为 `0` 推荐

统一含义：

- `PEFT exp01` 在 `quiet_sleep` 和 `focus+avoid` 组合约束上变得更保守。
- 这种保守并未转化为整体 citation 指标收益，反而拉低了 `citation_precision`。

## 4. 方法解释：为什么会出现这种负结果

当前 `v1` 训练 manifest 只包含四类任务：

- `preference_parse`
- `clarification`
- `constraint_honesty`
- `feedback_update`

当前 **不包含**：

- `grounded_recommendation`

因此 `v1` 的微调重点是：

- 结构化偏好理解
- 澄清决策
- unsupported honesty
- 单次反馈更新

而 `E10` 正式评测真正要求的是：

- 在固定 `EvidencePack` 上输出可验证、可引用、覆盖适当的 grounded recommendation

所以 `v1` 的负结果可以被解释为：

> 仅强化“前置行为能力”并不会自动转化成更强的最终 grounded recommendation；当训练目标缺少 grounded recommendation supervision 时，模型更可能在复杂约束 query 上表现出过度保守。

## 5. 当前正式结论

当前最稳妥的论文表述应固定为：

1. `E10 v1` 在同后端、同冻结资产条件下已经完成正式对照，结果可信。
2. `PEFT exp01` 未能优于 base，不应替代当前正式主系统。
3. 负结果并非来自运行时失真；`reasoning_leak_rate = 0.0` 说明本轮 compare 已排除了推理模式泄漏。
4. 当前最合理的下一步不是调 retrieval，也不是直接调大超参，而是补齐 grounded recommendation 监督数据。

## 6. E10 v2 的直接动机

`E10 v2` 的目标固定为：

- 保持 `Qwen3.5-4B + QLoRA + 同一评测协议` 不变
- 只修改训练数据构成
- 补入 `grounded_recommendation` 任务
- 定向覆盖：
  - `quiet_sleep`
  - `focus+avoid`
  - `partial-abstain / 0-recommendation`
  - 多方面均衡推荐

实现说明：

- 当前仓库中“非官方 `E9` 且无需澄清的可直接执行 query”数量为 `0`
- 因此 `v2` grounded 数据池会使用：
  - 非官方 `E9`
  - 但已有人类 `gold slot`
  - 且具备 `city + focus/avoid aspects` 的 query
- 训练输入中显式提供 `user_preference_gold`，以保证 grounded recommendation supervision 可执行，同时避免与官方 `E9` eval units 重叠

这样后续如果 `v2` 取得提升，论文中可以明确归因为：

- **数据设计改进**
- 而不是训练配方、推理后端或评测协议变化
