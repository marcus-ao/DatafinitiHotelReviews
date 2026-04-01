# 生成阶段第一轮正式材料：E9 第二轮正式结果汇总

更新时间：2026-04-01

本文件用于把当前 `E9` 第二轮正式结果整理成可直接复用到论文正文中的材料。当前正式冻结 run 为：

- `experiments/runs/e9_ecbcdbab690dc503_20260401T025012+0000/`

第一轮 run 仅保留为诊断对照：

- `experiments/runs/e9_80e05af30f45b1f2_20260401T021215+0000/`

## 1. 一句话摘要

在固定检索后端 `aspect_main_no_rerank`、`fallback=false`、固定 `Qwen/Qwen3.5-4B` 和固定 `E2 B_final_aspect_score Top5` 候选集的前提下，第二轮 `E9` 已经把生成层稳定到可审计、可复现、可冻结的水平。

## 2. 第二轮正式结果表

| Group | Query Count | Citation Precision | Evidence Verifiability Mean | Unsupported Honesty Rate | Schema Valid Rate | Avg Latency (ms) |
|---|---:|---:|---:|---:|---:|---:|
| `A_free_generation` | 40 | 0.9437 | 1.9697 | 1.0000 | 1.0000 | 2187.283 |
| `B_grounded_generation` | 40 | 0.9437 | 1.9773 | 1.0000 | 1.0000 | 2181.737 |
| `C_grounded_generation_with_verifier` | 40 | 0.9250 | 1.9922 | 1.0000 | 1.0000 | 2248.381 |

补充：

- `C` 组：
  - `retry_trigger_rate = 0.025`
  - `fallback_to_honest_notice_rate = 0.025`

## 3. 第一轮 vs 第二轮对比

| Group | Metric | First Run | Second Run | Delta |
|---|---|---:|---:|---:|
| `A_free_generation` | Citation Precision | 0.3000 | 0.9437 | +0.6437 |
| `A_free_generation` | Schema Valid Rate | 0.2500 | 1.0000 | +0.7500 |
| `B_grounded_generation` | Citation Precision | 0.4500 | 0.9437 | +0.4937 |
| `B_grounded_generation` | Schema Valid Rate | 0.4750 | 1.0000 | +0.5250 |
| `C_grounded_generation_with_verifier` | Citation Precision | 0.5500 | 0.9250 | +0.3750 |
| `C_grounded_generation_with_verifier` | Retry Trigger Rate | 0.5250 | 0.0250 | -0.5000 |
| `C_grounded_generation_with_verifier` | Honest Fallback Rate | 0.4000 | 0.0250 | -0.3750 |

解读：

- 第二轮的主要收益来自生成稳定性而不是 retrieval 变化。
- `A/B/C` 三组 `schema_valid_rate` 全部达到 `1.0`，说明第一轮的截断与错层级问题已经基本解决。
- `C` 组 verifier 仍有收益，但不再通过大规模清空输出换取可控性。

## 4. 代表性案例

### 案例 A：`q021` 是证据覆盖边界的诚实暴露

- Query：`q021`
- 需求：`Honolulu + quiet_sleep`
- 现象：三组都输出空推荐，并在 `unsupported_notice` 中明确说明当前缺少安静睡眠证据。
- 含义：这类结果不应被视为系统失败，而应被视为 evidence gap honesty，与 `E8` 的主结论一致。

### 案例 B：`q023` 说明“有位置证据”不等于“有安静睡眠证据”

- Query：`q023`
- 需求：`Honolulu + focus quiet_sleep + avoid location_transport`
- 现象：模型能够正确识别位置证据存在，但仍因缺少安静睡眠证据而拒绝推荐。
- 含义：当前系统已经能把“有部分方面证据”与“满足完整偏好”区分开。

### 案例 C：`q079` 是 verifier 过严的残留误杀

- Query：`q079`
- 需求：`Chicago + service + quiet_sleep + location_transport`
- `A` 组：`citation_precision = 1.0`
- `B` 组：`citation_precision = 0.75`
- `C` 组：因 verifier 检测到单个非法 `sentence_id`，最终降级为诚实空输出
- 含义：当前 verifier 的主要残留问题已不是系统性失控，而是对单点 citation typo 仍偏保守。

### 案例 D：`q084` 说明 free generation 仍可能出现 citation drift

- Query：`q084`
- 需求：`San Diego + cleanliness + room_facilities + value`
- `A` 组：存在 1 条 citation drift，`citation_precision = 0.75`
- `B/C` 组：`citation_precision = 1.0`
- 含义：即使第二轮稳定性显著提升，free generation 在少量复杂多方面 query 上仍可能出现 citation 漂移。

## 5. 统一章节结论

当前 `E9` 最稳妥的论文表述应固定为：

1. 第二轮 `E9` 已经把生成层稳定到可审计、可复现、可冻结的水平。
2. 当前更强的正结果是“生成已可控、可审计”，而不是“grounded 约束在 citation precision 上远远碾压 free generation”。
3. `q021 / q023` 这类空输出属于 evidence gap honesty，不视为系统失败。
4. `q079` 是 verifier 过严的单点残留边界，应作为误差分析保留，但不再触发 retrieval 主线变更。
5. 当前 `E9` 已可作为论文正式生成阶段结果冻结，项目主线应转入 `E10 / PEFT`。
