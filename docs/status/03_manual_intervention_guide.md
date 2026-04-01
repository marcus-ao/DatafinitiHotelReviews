# 人工介入详细指南手册

更新时间：2026-04-01

本手册只写“现在仍然需要你亲自处理”的内容，并尽量细到可以一步一步照着做。

## 当前结论

当前最可能的人工阻塞点，已经从“补跑行为实验”切换成了“按新实现入口执行 `E9`、完成 `E9` 审计，并决定是否还要追加 `Qwen3.5-9B` 附录对比”。

也就是说：

- `E1 / E2 / E5 / E6 / E7 / E8` 都已经有正式结果
- `Qwen3.5-2B` 的第一轮 `E3/E4` baseline 已经归档
- `Qwen3.5-4B` 的全量 `E3/E4` 正式结果已经完成
- 最新 `E4` 审计已经补齐，并生成了 reviewed 冻结副本
- `E3 / E4 / E5` 的章节材料已经整理完第一版
- `E9 / E10` 的代码入口、schema 和 runner 骨架已于 `2026-04-01` 实现
- `e10_prepare_manifests` 已成功生成 `sft_train_manifest.jsonl` 与 `sft_dev_manifest.jsonl`
- `e9_freeze_assets --limit-queries 2` 已通过本地 smoke
- 现在最需要你做的是：按固定顺序执行 `E9` full assets 冻结与 `E9` 正式评测

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

### 第二步：优先把 `E9` 的执行入口按顺序跑起来

你现在最值得亲自打开和执行的是：

- `scripts/evaluation/evaluate_e9_e10_generation.py`
- `experiments/assets/sft_train_manifest.jsonl`
- `experiments/assets/sft_dev_manifest.jsonl`
- `experiments/labels/e9_generation/README.md`
- `docs/plans/03_generation_and_peft_phase_plan.md`

你当前要按下面顺序执行：

1. 先在仓库根目录激活环境：

```bash
source venv/bin/activate
```

2. 先确认 `E10` 的 SFT manifest 已可重生成：

```bash
python -m scripts.evaluation.run_experiment_suite --task e10_prepare_manifests
```

3. 再正式冻结 `E9` full assets：

```bash
python -m scripts.evaluation.run_experiment_suite --task e9_freeze_assets
```

4. full assets 成功后，再执行 `E9` 正式 run：

```bash
python -m scripts.evaluation.run_experiment_suite --task e9_generation_constraints
```

### 第三步：`E9` 跑完后，按固定步骤做人工审计

1. 打开：
   - `experiments/runs/e9_*/summary.csv`
   - `experiments/runs/e9_*/analysis.md`
   - `experiments/runs/e9_*/citation_verifiability_audit.csv`
   - `experiments/labels/e9_generation/citation_verifiability_audit.csv`
2. 逐行查看：
   - `query_id`
   - `group_id`
   - `hotel_id`
   - `sentence_id`
   - `reason_text`
   - `citation_exists`
   - `in_current_evidence_pack`
   - `support_score`
   - `notes`
3. 如果你决定补人工 reviewed 结果，只改：
   - `experiments/labels/e9_generation/citation_verifiability_audit.csv`
4. 不要覆盖 run 内原始文件；run 内原始文件保留为当轮自动导出快照

打分标准固定为：

- `citation_exists`
  - `1`：该 `sentence_id` 在当前冻结 evidence 索引中真实存在
  - `0`：该 `sentence_id` 不存在
- `in_current_evidence_pack`
  - `1`：该 `sentence_id` 属于当前 query + hotel 的 `EvidencePack`
  - `0`：该 `sentence_id` 不属于当前 pack，属于越权引用
- `support_score`
  - `2`：证据明确支持当前理由
  - `1`：基本相关但支持偏弱
  - `0`：证据不支持或明显越权

### 第四步：执行 `E9` 前后，你最该注意什么

你现在最该记住的固定约束是：

- 不要改 `aspect_main_no_rerank`
- 不要把 `fallback` 接回默认主流程
- 不要把 `candidate_hotels` 改成 `city_test_all`
- `E9` 当前固定使用 `E2 B_final_aspect_score Top5`
- `E9` 当前正式基座模型固定为 `Qwen/Qwen3.5-4B`
- `E9` 当前优先只读本地 embedding 缓存；若本地没有 `BAAI/bge-small-en-v1.5` 缓存，会导致 `e9_freeze_assets` 失败

如果 `e9_freeze_assets` 报错，优先按下面方式判断：

1. 如果报 embedding 模型离线加载失败：
   - 说明本地没有缓存 `BAAI/bge-small-en-v1.5`
   - 先在可联网环境把该模型缓存好，再回到当前仓库重跑
2. 如果 assets 冻结成功，但 `e9_generation_constraints` 失败：
   - 优先检查 `OPENAI_BASE_URL`
   - 检查 `OPENAI_API_KEY`
   - 检查当前推理端点是否确实提供 `Qwen/Qwen3.5-4B`

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

如果你决定不先补 `9B`，那么下一步人工动作固定是下面四件：

1. 用 `05_behavior_stage_3_chapter_materials.md` 把行为章节写进论文
2. 运行 `e10_prepare_manifests`
3. 运行 `e9_freeze_assets`
4. 运行 `e9_generation_constraints` 并完成 `E9` 审计

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

当前行为主线已经完成并审计完毕；你现在最该做的是保持已冻结主线不动，然后按已经实现的入口依次执行 `e10_prepare_manifests -> e9_freeze_assets -> e9_generation_constraints`。若不补 `9B`，项目主线就应先把 `E9` 正式跑通并完成审计，而不是立刻跳进 PEFT。
