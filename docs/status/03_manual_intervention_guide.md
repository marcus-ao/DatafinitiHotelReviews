# 人工介入详细指南手册

更新时间：2026-03-31

本手册只写“现在仍然需要你亲自处理”的内容，并尽量细到可以一步一步照着做。

## 当前结论

当前最可能的人工阻塞点，已经从“补跑行为实验”切换成了“复核已完成的 `E4` 审计、把行为章节写进论文，以及决定是否还要追加 `Qwen3.5-9B` 附录对比”。

也就是说：

- `E1 / E2 / E5 / E6 / E7 / E8` 都已经有正式结果
- `Qwen3.5-2B` 的第一轮 `E3/E4` baseline 已经归档
- `Qwen3.5-4B` 的全量 `E3/E4` 正式结果已经完成
- 最新 `E4` 审计已经补齐，并生成了 reviewed 冻结副本
- `E3 / E4 / E5` 的章节材料已经整理完第一版
- 现在最需要你决定的是：要不要补 `9B` 作为附录；若不补，就准备进入 `E9`

## 手册 A：你现在最该先做什么

### 第一步：不要再改动当前行为主线冻结资产

你现在不要做的是：

- 不要覆盖：
  - `experiments/runs/e3_244aca8abf6345ad_20260331T072527+0000/`
  - `experiments/runs/e4_4a15a89128a90d11_20260331T073016+0000/`
  - `experiments/runs/e3_14928d821d811e86_20260331T122611+0000/`
  - `experiments/runs/e4_55c8021e1119fb77_20260331T122648+0000/`
- 不要删除：
  - `experiments/reports/03_behavior_stage_1_qwen35_2b_baseline.md`
  - `experiments/reports/04_behavior_stage_2_qwen35_4b_formal_summary.md`
  - `experiments/reports/05_behavior_stage_3_chapter_materials.md`

### 第二步：优先复核已经填好的 `E4` 审计

你现在最值得亲自打开的文件是：

- `experiments/labels/e4_clarification/clarification_question_audit.csv`
- `experiments/labels/e4_clarification/clarification_question_audit_e4_55c8021e1119fb77_qwen35_4b_reviewed.csv`
- `experiments/runs/e4_55c8021e1119fb77_20260331T122648+0000/clarification_question_audit.csv`
- `experiments/reports/05_behavior_stage_3_chapter_materials.md`

你要重点确认三件事：

- `q051` 这一类缺城市提问是否足够简洁
- `q057` 这一类冲突澄清是否自然且聚焦
- `q062` 这一类边界例是否值得写进论文

### 第三步：如果你需要重新审计或微调分数，按固定步骤做

1. 打开：
   - `experiments/labels/e4_clarification/clarification_question_audit.csv`
   - `experiments/runs/e4_55c8021e1119fb77_20260331T122648+0000/clarification_question_audit.csv`
2. 逐行查看：
   - `query_id`
   - `group_id`
   - `question`
   - `answerable_score`
   - `targeted_score`
   - `notes`
3. 如果你决定调整，只改最新副本：
   - `clarification_question_audit.csv`
4. 改完后，再导出新的 reviewed 副本，不覆盖：
   - baseline 快照
   - run 内原始文件

打分标准固定为：

- `answerable_score`
  - `2`：明确 可直接回答
  - `1`：基本可答 但问法一般
  - `0`：不清楚 难回答或过绕
- `targeted_score`
  - `2`：高度聚焦真正缺的槽位
  - `1`：基本相关 但不够精准
  - `0`：没有问到真正关键点

### 第四步：把行为章节材料直接用进论文

当前最建议直接复用的文件是：

- `experiments/reports/05_behavior_stage_3_chapter_materials.md`

建议你优先把里面这些内容搬进论文：

- `E3` 结果表
- `E4` 结果表
- `E5` 结果表
- `q048 / q062 / q013 / q043` 案例
- `2B 是弱基线 4B 是正式主模型` 这句主结论

## 手册 B：如果你确实要补 `9B` 附录，该怎么做

只有在你明确需要“模型规模附录”时，才做这一步。

推荐顺序：

1. 按 `docs/deployment/01_autodl_qwen35_behavior_runbook.md` 启动 `Qwen3.5-9B`
2. 先做 API 冒烟验证
3. 跑 `E3`
4. 跑 `E4`
5. 把新的 run 目录同步回本地
6. 新增 `experiments/reports/06_behavior_stage_4_qwen35_9b_appendix.md`
7. 只回答：
   - `9B` 是否显著优于 `4B`
   - 收益是否足以改变当前论文主结论

建议命令：

```bash
export BEHAVIOR_MODEL_ID=Qwen/Qwen3.5-9B
python -m scripts.evaluation.run_experiment_suite --task e3_preference
python -m scripts.evaluation.run_experiment_suite --task e4_clarification
```

注意：

- 不要因为换到 `9B` 就顺手改 prompt
- 不要改 query 集
- 不要把 `reranker` 或 `fallback` 临时接回主流程
- `9B` 结果只能作为附录，不覆盖 `4B` 当前正式结论

## 手册 C：如果你不补 `9B`，下一步人工动作是什么

如果你决定不先补 `9B`，那么下一步人工动作只有两件：

1. 用 `05_behavior_stage_3_chapter_materials.md` 把行为章节写进论文
2. 按 `docs/plans/03_generation_and_peft_phase_plan.md` 开始进入 `E9`

在进入 `E9` 之前，你当前不需要再手动做：

- 重跑 `Qwen3.5-2B`
- 重跑 `Qwen3.5-4B`
- 重新标 E6 qrels
- 重新标 E1 gold
- 立刻启动 PEFT

## 手册 D：如果云端再跑不起来，你要怎么判断问题

### 情况 1：云端环境一开始就报模型加载错误

这通常说明：

- 云端环境没有准备好当前要跑的 `Qwen3.5-9B`
- 或 Hugging Face 权限 / 下载过程失败

这时优先记录为：

- “云端附录模型不可用”

### 情况 2：`vLLM` 服务能通，但实验脚本仍失败

这通常说明真正阻塞的是：

- `OPENAI_BASE_URL` 配错
- `BEHAVIOR_MODEL_ID` 与当前启动模型不一致
- `BEHAVIOR_ENABLE_THINKING` 没按实验要求关闭

### 情况 3：新增 run 有了，但结论不清楚

这时不要立刻改 prompt，而是先回到：

- `experiments/reports/05_behavior_stage_3_chapter_materials.md`
- 新增的 `06_behavior_stage_4_qwen35_9b_appendix.md`

判断 `9B` 的收益到底是不是足以改变主结论。

## 手册 E：一句话版

当前行为主线已经完成并审计完毕；你现在最该做的是把 `05_behavior_stage_3_chapter_materials.md` 用进论文，并决定是否真的需要 `9B` 附录。若不补 `9B`，项目主线就应直接按 `03_generation_and_peft_phase_plan.md` 进入 `E9`，而不是立刻跳进 PEFT。
