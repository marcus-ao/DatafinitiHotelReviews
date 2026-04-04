# 人工介入详细指南手册

更新时间：2026-04-04

本手册只写“现在仍然需要你亲自处理”的内容，并尽量细到可以一步一步照着做。

## 当前结论

当前最可能的人工阻塞点，已经从“补跑行为实验”切换成了“引用已冻结的 `E9` 正式结果与有无 RAG 对比，并继续推进后续 `E10` 主线实验归档与复评”。

也就是说：

- `E1 / E2 / E5 / E6 / E7 / E8` 都已经有正式结果
- `Qwen3.5-2B` 的第一轮 `E3/E4` baseline 已经归档
- `Qwen3.5-4B` 的全量 `E3/E4` 正式结果已经完成
- 最新 `E4` 审计已经补齐，并生成了 reviewed 冻结副本
- `E3 / E4 / E5` 的章节材料已经整理完第一版
- `E9 / E10` 的代码入口、schema、runner 与 compare 入口已经接通
- `E10 v1` 正式 compare 已完成并冻结
- `E9` 第二轮正式结果已完成并冻结为：
  - `experiments/runs/e9_ecbcdbab690dc503_20260401T025012+0000/`
- `E9` 有无 RAG 正式 compare 已完成并冻结为：
  - `experiments/runs/e9_8449c12a50585e42_20260404T081010+0000/`
- `E10 v1` 正式 run 已完成并冻结为：
  - `experiments/runs/e10_0dc5c2e6f867c66f_20260402T015230+0000/`
  - `experiments/runs/e10_0ef381420c1bd19a_20260402T020120+0000/`
  - `experiments/runs/e10cmp_28598dfb8434c1ba_20260402T020734+0000/`
- 现在最需要你做的是：准备 `E10 v3` 的 grounded manifest、训练 `exp03`、重跑 PEFT v3 并生成新的 compare

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

### 第二步：优先把 `E10 v3` 的执行入口按顺序跑起来

你现在最值得亲自打开和执行的是：

- `experiments/reports/07_generation_stage_2_e10_formal_summary.md`
- `experiments/reports/08_generation_stage_3_e10_v2_iteration_summary.md`
- `scripts/evaluation/evaluate_e9_e10_generation.py`
- `experiments/assets/e10_train_config.qwen35_4b_peft_v3.json`
- `docs/deployment/02_e10_peft_runbook.md`
- `docs/plans/03_generation_and_peft_phase_plan.md`

你当前要按下面顺序执行：

1. 先在仓库根目录激活环境：

```bash
source venv/bin/activate
```

2. 在云端 strongest base 环境下生成 `v3` manifest：

```bash
python -m scripts.evaluation.run_experiment_suite --task e10_prepare_manifests_v3
python -m scripts.evaluation.run_experiment_suite --task e10_validate_manifest_v3
```

补充说明：

- `v3` 继续保留非官方 judged grounded 来源
- 同时新增基于 `train` 酒店证据 pack 的 synthetic grounded query
- synthetic query 只服务于：
  - `partial_support_keep_recommendation`
  - `multi_hotel_pack_boundary`
  - `focus_and_avoid`
- 当前目的是修复：
  - `q018 / q022`
  - `q085`

3. 用 `v3` config 训练 `exp03`：

```bash
accelerate launch -m scripts.training.train_e10_peft \
  --config experiments/assets/e10_train_config.qwen35_4b_peft_v3.json
```

4. merge `exp03` 后，只重跑 PEFT：

```bash
python -m scripts.training.merge_e10_peft_adapter \
  --base-model-path /root/autodl-tmp/models/base/Qwen3.5-4B \
  --adapter-path /root/autodl-tmp/models/adapters/qwen35_4b_qlora/exp03 \
  --merged-output-path /root/autodl-tmp/models/merged/qwen35_4b_merged_exp03 \
  --report-dir /root/autodl-tmp/training/reports/qwen35_4b_qlora/exp03 \
  --repo-metadata-path experiments/assets/e10_adapter_metadata.qwen35_4b_peft_v3.json

python -m scripts.evaluation.run_experiment_suite --task e10_base_vs_peft --group-id B_peft_4b_grounded
```

5. 最后复用已冻结的 base formal run，生成 compare：

```bash
python -m scripts.evaluation.run_experiment_suite \
  --task e10_compare_runs \
  --base-run-dir /abs/path/to/e10_0dc5c2e6f867c66f_20260402T015230+0000 \
  --peft-run-dir /abs/path/to/new_peft_v3_run
```

### 第三步：`E10 v3` 跑完后，按固定步骤做人工审计

1. 打开：
   - 新的 `experiments/runs/e10_*/summary.csv`
   - 新的 `experiments/runs/e10_*/analysis.md`
   - 新的 `experiments/runs/e10_*/citation_verifiability_audit.csv`
   - 新的 `experiments/runs/e10cmp_*/analysis.md`
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
3. 如需补人工 reviewed 结果，可在 `E10` 的 labels 目录中新增 reviewed 副本，不覆盖 run 内原始文件
4. 不要改动 `E9` 冻结资产与第二轮正式 run

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

### 第四步：执行 `E10 v3` 时，你最该注意什么

你现在最该记住的固定约束是：

- 不要改 `aspect_main_no_rerank`
- 不要把 `fallback` 接回默认主流程
- 不要把 `candidate_hotels` 改成 `city_test_all`
- `E9` 当前正式冻结输入继续固定使用 `E2 B_final_aspect_score Top5`
- `E10 v3` 当前 base 组直接复用：
  - `e10_0dc5c2e6f867c66f_20260402T015230+0000`
- `E10 v3` 当前只重跑：
  - `B_peft_4b_grounded`
- `E10 v3` 当前新增训练任务：
  - `grounded_recommendation`

如果 `e10_base_vs_peft` 报错，优先按下面方式判断：

1. 如果 `e10_prepare_manifests_v3` 报 `grounded pool` 为空：
   - 说明当前 judged + synthetic 两路来源都没有产出合格 silver target
   - 先检查：
     - `BEHAVIOR_MODEL_ID`
     - `BEHAVIOR_ENABLE_THINKING=false`
     - strongest base 是否已经就绪
2. 如果报 base_model_id 不匹配：
   - 说明 adapter 不是从当前主线 `Qwen/Qwen3.5-4B` 训练出来的
3. 如果 `grounded_recommendation` 样本里继续出现 `sentence_id=null` 或 “无直接证据支持” 写进 `reasons[]`：
   - 说明 `partial_support_keep_recommendation` 样本仍然不够
   - 先检查 `v3` synthetic 样本是否真正进入 manifest
4. 如果 `PEFT v3` compare 仍落后于 base 超过 `0.01`：
   - 将 `v3` 记为“数据+约束修复仍不足”
   - 这时再考虑进入“数据+训练配方”路线，而不是在同一轮里继续混改

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

1. 用 `07_generation_stage_2_e10_formal_summary.md` 和 `08_generation_stage_3_e10_v2_iteration_summary.md` 把 `E10 v1 / v2` 结论写进论文
2. 运行 `e10_prepare_manifests_v3`
3. 按 `02_e10_peft_runbook.md` 训练 `exp03`
4. 重跑 `B_peft_4b_grounded` 并生成 compare

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

当前行为主线、`E9` 第二轮正式结果与 `E10 v1` 正式 compare 都已经完成并冻结；你现在最该做的是保持这些冻结资产不动，生成 `v3` grounded manifest、训练 `exp03` 并复评，而不是再回头改 retrieval 主线。
