# 下一步该做什么

更新时间：2026-03-31

## 当前推荐推进顺序

请按下面顺序推进，不建议跳步：

1. 确认并保持默认检索配置固定为 `aspect_main_no_rerank`
2. 保留当前 `Qwen3.5-2B` baseline，不覆盖、不删除
3. 先在云端重跑 `E3 v2` 诊断子集
4. 再重跑 `E4 v2` 诊断子集
5. 只有诊断通过后，才重跑 `Qwen3.5-2B` 全量 `E3/E4`
6. 若 `2B v2` 仍明显不足，再切到 `Qwen3.5-4B / 9B`
7. 整理 `E3/E4/E5` 论文材料
8. 再进入更后面的 `E9/E10`、PEFT 与主实验矩阵

## 第一优先级：立即要做

### 任务 1：保持 baseline 归档状态不变

目标：

- 将第一轮 `Qwen3.5-2B` 的 `E3/E4` run 继续视为冻结基线，不覆盖原目录

完成标准：

- `experiments/runs/e3_244aca8abf6345ad_20260331T072527+0000/` 保持不变
- `experiments/runs/e4_4a15a89128a90d11_20260331T073016+0000/` 保持不变
- `experiments/reports/03_behavior_stage_1_qwen35_2b_baseline.md` 已作为阶段总结保留

### 任务 2：先跑 E3 v2 诊断子集

目标：

- 在云端 `Qwen3.5-2B` 服务不变的前提下，先验证 `e3_v2_cn_slots_only` 是否解决后处理误伤与 unsupported 漏检

完成标准：

- 使用：
  - `experiments/assets/e3_diagnostic_query_ids.json`
- 产生一轮新的 `experiments/runs/e3_*/`
- 输出：
  - `run_meta.json`
  - `results.jsonl`
  - `summary.csv`
  - `analysis.md`
- 重点验收：
  - 不再出现因 `City:ST` 格式导致的 `city_missing`
  - `unsupported_detection_recall >= 0.50`
  - `exact_match_rate` 相比 baseline 明显提升

### 任务 3：再跑 E4 v2 诊断子集

目标：

- 验证 `e4_v2_cn_decision_label_fewshot` 是否解决“全判 false”的塌缩

完成标准：

- 使用：
  - `experiments/assets/e4_diagnostic_query_ids.json`
- 产生一轮新的 `experiments/runs/e4_*/`
- 输出：
  - `run_meta.json`
  - `results.jsonl`
  - `summary.csv`
  - `analysis.md`
- 额外生成：
  - run 内部的 `clarification_question_audit.csv`
  - 最新副本 `experiments/labels/e4_clarification/clarification_question_audit.csv`
- 重点验收：
  - 不再出现 `16` 条正例全部判 `false`
  - 诊断子集 `recall >= 0.75`
  - 平衡子集 `F1 >= 0.70`

### 任务 4：只有诊断通过后，才重跑 2B 全量 E3/E4

目标：

- 在同一版 `v2` 设计上重跑 `86` 条全量 query

完成标准：

- 产生新的全量 `e3_*` / `e4_*` run
- 新 run 与 baseline run 并列存在
- 若 `2B v2` 仍明显塌缩，则不继续打磨 `2B`，直接切到 `4B`

### 任务 5：按需切换到 4B / 9B

触发条件：

- `E4 v2` 诊断子集仍严重塌缩
- 或 `E3 v2` 的 unsupported 识别仍明显不达标

执行要求：

- 保持同一版 `v2` prompt 和同一批诊断 query
- 不在切模型时顺手改 prompt 或评测口径

## 第二优先级：E3/E4 跑稳后立刻做

### 任务 6：整理 E3/E4/E5 论文材料

目标：

- 将三组行为实验写成可直接复用的结果表、错误分析与设计结论

建议最少形成：

- `1` 张 E3 结果表
- `1` 张 E4 结果表
- `1` 张 E5 结果表
- `2-3` 组典型错误/边界案例
- `1` 段“固定后端条件下的行为层结论”

### 任务 7：把行为实验与前面的 Aspect-KB 章节串起来

建议整体叙述顺序：

1. `E1`：标签可靠性
2. `E2`：画像候选缩圈
3. `E6-E8`：检索组织方式与边界
4. `E5`：中英桥接必要性
5. `E3/E4`：偏好解析与澄清触发

## 当前不建议启动的内容

在下面这些条件未满足前，不建议提前进入：

- `E3/E4` 还没正式跑完前：不要启动 PEFT
- 行为实验章节还没收口前：不要启动 `G1-G4`
- 当前阶段：不要把 `reranker` 或 `fallback` 再接回默认主流程
- 当前阶段：不要写“系统全链路已经充分验证优于所有基线”这一类过强结论

## 一句话版本

你现在最该做的，是保留好已经跑出的 `Qwen3.5-2B` baseline，然后先在云端用同一个 `2B` 服务重跑 `E3/E4 v2` 诊断子集；只有诊断达标后再跑全量，不达标才切到 `4B / 9B`。
