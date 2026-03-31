# 下一步该做什么

更新时间：2026-03-31

## 当前推荐推进顺序

请按下面顺序推进，不建议跳步：

1. 确认并保持默认检索配置固定为 `aspect_main_no_rerank`
2. 确认当前正式行为模型固定为 `Qwen/Qwen3.5-4B`
3. 保留已有 baseline、诊断 run 和 `4B` 全量正式 run，不覆盖、不删除
4. 先人工审阅最新 `E4` 审计文件
5. 再把 `E3 / E4 / E5` 收口成统一的行为实验论文材料
6. 只有在需要模型规模附录时，才追加 `Qwen3.5-9B` 对比
7. 行为章节收口后，再进入 `E9/E10`、PEFT 与主实验矩阵

## 第一优先级：立即要做

### 任务 1：保持当前归档与正式状态不变

目标：

- 将 `2B` baseline、两轮诊断 run 和 `4B` 全量正式 run 一并视为冻结资产，不覆盖原目录

完成标准：

- `experiments/runs/e3_244aca8abf6345ad_20260331T072527+0000/` 保持不变
- `experiments/runs/e4_4a15a89128a90d11_20260331T073016+0000/` 保持不变
- `experiments/runs/e3_da541f84770ed8ed_20260331T090311+0000/` 保持不变
- `experiments/runs/e4_96e0e4afb24dab2d_20260331T091021+0000/` 保持不变
- `experiments/runs/e3_f62d907e600cfc14_20260331T120756+0000/` 保持不变
- `experiments/runs/e4_f928a37444c1bf52_20260331T121012+0000/` 保持不变
- `experiments/runs/e3_14928d821d811e86_20260331T122611+0000/` 保持不变
- `experiments/runs/e4_55c8021e1119fb77_20260331T122648+0000/` 保持不变
- `experiments/reports/03_behavior_stage_1_qwen35_2b_baseline.md` 已作为阶段总结保留
- `experiments/reports/04_behavior_stage_2_qwen35_4b_formal_summary.md` 已作为当前正式总结保留
- `configs/params.yaml` 与 `experiments/assets/frozen_config.yaml` 继续保持：
  - `default_retrieval_mode = aspect_main_no_rerank`
  - `behavior.base_model = Qwen/Qwen3.5-4B`

### 任务 2：人工审阅最新 E4 审计文件

目标：

- 确认 `Qwen3.5-4B` 在应澄清 query 上提出的问题是否足够可答、聚焦、可写入论文

完成标准：

- 打开：
  - `experiments/labels/e4_clarification/clarification_question_audit.csv`
  - `experiments/runs/e4_55c8021e1119fb77_20260331T122648+0000/clarification_question_audit.csv`
- 至少完成全部应澄清正例的：
  - `answerable_score`
  - `targeted_score`
  - `notes`
- 明确记录当前最值得写进论文的正例和边界例

### 任务 3：整理 E3 / E4 / E5 论文材料

目标：

- 将行为层实验收口成一组可直接进入论文初稿的表格、结论和案例材料

完成标准：

- 至少形成：
  - `1` 张 `E3` 结果表
  - `1` 张 `E4` 结果表
  - `1` 张 `E5` 结果表
  - `2-3` 组代表性案例
  - `1` 段“固定检索后端下的行为层结论”
- 当前主结论需明确写清：
  - `Qwen3.5-2B` 是弱基线
  - `Qwen3.5-4B` 是当前正式主模型
  - `E5` 说明中英桥接是必要条件

### 任务 4：仅在需要附录时追加 9B 对比

触发条件：

- 论文或答辩需要补一组“模型规模继续增大是否还有收益”的对比
- 或你希望把 `4B` 的正式主结论再做一个更强上界参照

执行要求：

- 保持完全相同的：
  - query 集
  - prompt 版本
  - schema
  - 默认检索配置
- 不在切到 `9B` 时顺手改评测口径
- `9B` 结果定位为附录或扩展，不覆盖 `4B` 当前主结果

## 第二优先级：E3/E4 跑稳后立刻做

### 任务 5：把行为实验与前面的 Aspect-KB 章节串起来

建议整体叙述顺序：

1. `E1`：标签可靠性
2. `E2`：画像候选缩圈
3. `E6-E8`：检索组织方式与边界
4. `E5`：中英桥接必要性
5. `E3/E4`：偏好解析与澄清触发

### 任务 6：行为章节收口后再进入后续扩展

建议顺序：

1. 先确认 `E3/E4/E5` 已经形成论文材料
2. 再决定是否需要 `Qwen3.5-9B` 附录对比
3. 行为层结论稳定后，再进入 `E9/E10`
4. 最后再进入 PEFT 与主实验矩阵

## 当前不建议启动的内容

在下面这些条件未满足前，不建议提前进入：

- 行为实验章节还没收口前：不要启动 `G1-G4`
- 当前阶段：不要把 `reranker` 或 `fallback` 再接回默认主流程
- 当前阶段：不要写“系统全链路已经充分验证优于所有基线”这一类过强结论
- 当前阶段：不要为了追求更高指标而覆盖 `2B` baseline 或 `4B` 正式 run

## 一句话版本

你现在最该做的，是把已经跑出的 `Qwen3.5-4B` 正式 `E3/E4` 结果审计并收口成论文材料，同时保留 `Qwen3.5-2B` 作为弱基线；只有当你确实需要模型规模附录时，再追加 `Qwen3.5-9B` 对比。
