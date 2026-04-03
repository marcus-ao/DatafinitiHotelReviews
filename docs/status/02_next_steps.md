# 下一步该做什么

更新时间：2026-04-02

## 当前推荐推进顺序

请按下面顺序推进，不建议跳步：

1. 确认并保持默认检索配置固定为 `aspect_main_no_rerank`
2. 确认当前正式行为模型固定为 `Qwen/Qwen3.5-4B`
3. 保留已有 baseline、诊断 run 和 `4B` 全量正式 run，不覆盖、不删除
4. 以 `experiments/reports/05_behavior_stage_3_chapter_materials.md` 为主入口，完成行为章节正文写作
5. 仅在论文或答辩需要时，追加 `Qwen3.5-9B` 附录对比
6. 将 `E9` 第二轮结果视为当前正式冻结结果，不再改 retrieval 主线
7. 冻结 `E10 v1` 正式负结果，不再把 `PEFT exp01` 视为待确认结果
8. 冻结 `E10 v2` 阶段性结果，进入 `E10 v3` 数据+约束修复，生成 `v3` manifest、训练 `exp03` 并复评

## 第一优先级：立即要做

### 任务 1：保持当前主线冻结状态不变

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
- `experiments/labels/e4_clarification/clarification_question_audit_e4_55c8021e1119fb77_qwen35_4b_reviewed.csv` 已作为 reviewed 副本保留
- `configs/params.yaml` 与 `experiments/assets/frozen_config.yaml` 继续保持：
  - `default_retrieval_mode = aspect_main_no_rerank`
  - `behavior.base_model = Qwen/Qwen3.5-4B`

### 任务 2：用 `05_behavior_stage_3_chapter_materials.md` 收口行为章节

目标：

- 将 `E3 / E4 / E5` 的结果和案例直接映射到论文正文，不再分散地从多个 run 目录手工拼材料

完成标准：

- `E3` 表、`E4` 表、`E5` 表都已从 `05` 号汇总文档进入论文草稿
- 至少引用：
  - `q048`
  - `q062`
  - `q013`
  - `q043`
- 当前章节主结论固定为：
  - `2B` 是弱基线
  - `4B` 是当前正式主模型
  - `E5` 证明中英桥接是必要条件

### 任务 3：决定是否真的需要 `9B` 附录

默认选择：

- 不先补 `9B`

只有在以下情况满足时才执行：

- 论文需要“模型规模继续增大是否还有收益”的附录
- 或答辩需要更强上界参照

执行要求：

- 不改 prompt
- 不改 query 集
- 不改 schema
- 不改检索配置
- 仅替换 `BEHAVIOR_MODEL_ID=Qwen/Qwen3.5-9B`

若执行，则额外产物固定为：

- 新的 `e3_*` / `e4_*` run
- `experiments/reports/06_behavior_stage_4_qwen35_9b_appendix.md`

## 第二优先级：行为章节收口后立刻做

### 任务 4：冻结 `E9` 第二轮正式结果

目标：

- 将 `e9_ecbcdbab690dc503_20260401T025012+0000` 视为当前正式 `E9` 结果
- 不再继续改 retrieval 主线
- 把 `E9` 的主结论写进状态文档、阶段计划和论文章节材料

当前固定结果：

- 正式 run：
  - `experiments/runs/e9_ecbcdbab690dc503_20260401T025012+0000/`
- 诊断 run：
  - `experiments/runs/e9_80e05af30f45b1f2_20260401T021215+0000/`
- 推荐直接引用：
  - `experiments/reports/06_generation_stage_1_e9_formal_summary.md`

当前默认解读：

- `q021 / q023` 是证据缺口诚实暴露，不视为系统失败
- `q079` 是 verifier 误杀残留边界，不再触发 retrieval 主线变更
- `E9` 当前正式口径继续固定为：
  - `aspect_main_no_rerank`
  - `fallback=false`
  - `Qwen/Qwen3.5-4B`
  - `E2 B_final_aspect_score Top5`

### 任务 5：冻结 `E10 v2` 阶段性结果并转入 `E10 v3`

目标：

- 将 `E10 v2` 固定为“比 `v1` 更好但仍未超过 base”的阶段性结果
- 在不改 retrieval、不改评测协议的前提下，推进 `E10 v3` 数据+约束修复与复训

当前推荐顺序为：

1. 先冻结并引用当前正式 run：
   - `e10_0dc5c2e6f867c66f_20260402T015230+0000`
   - `e10_0ef381420c1bd19a_20260402T020120+0000`
   - `e10cmp_28598dfb8434c1ba_20260402T020734+0000`
   - `e10_a2dd1a0bd73c57b5_20260402T073127+0000`
   - `e10cmp_7cf0c9c0a9830796_20260402T074331+0000`
2. 确认以下 `v3` 资产/入口存在且内容正确：
   - `experiments/assets/e10_train_config.qwen35_4b_peft_v3.json`
   - `scripts/evaluation/run_experiment_suite.py` 中的 `e10_prepare_manifests_v3`
   - `docs/plans/03_generation_and_peft_phase_plan.md`
   - `experiments/reports/08_generation_stage_3_e10_v2_iteration_summary.md`
3. 在云端基于 strongest base 生成 `v3 grounded` silver manifest：

```bash
python -m scripts.evaluation.run_experiment_suite --task e10_prepare_manifests_v3
```

4. 在云端训练 `exp03`：

```bash
accelerate launch -m scripts.training.train_e10_peft \
  --config experiments/assets/e10_train_config.qwen35_4b_peft_v3.json
```

5. merge `exp03` 后，只重跑：

```bash
python -m scripts.evaluation.run_experiment_suite \
  --task e10_base_vs_peft \
  --group-id B_peft_4b_grounded
```

6. 最后复用 base formal run，生成 compare：

```bash
python -m scripts.evaluation.run_experiment_suite \
  --task e10_compare_runs \
  --base-run-dir /abs/path/to/e10_0dc5c2e6f867c66f_20260402T015230+0000 \
  --peft-run-dir /abs/path/to/new_peft_v2_run
```

当前固定约束：

- 不改 retrieval 主线
- 不改 `E9` eval units
- 不改正式 base baseline
- 不改评测 prompt 与 compare 协议
- `v3` 训练配方保持不变，只扩展/修复训练数据
- `v3` task_types 继续为：
  - `preference_parse`
  - `clarification`
  - `constraint_honesty`
  - `feedback_update`
  - `grounded_recommendation`
- `v3` 优先修复：
  - `q018 / q022` 的 partial-support schema 问题
  - `q085` 的 multi-hotel pack boundary 问题

## 当前不建议启动的内容

在下面这些条件未满足前，不建议提前进入：

- 行为章节还没真正写进论文前：不要启动 `G1-G4`
- 在未生成 `sft_train_manifest_v2.jsonl / sft_dev_manifest_v2.jsonl` 前：不要直接启动 `exp02`
- 当前阶段：不要把 `reranker` 或 `fallback` 再接回默认主流程
- 当前阶段：不要为了追求更高指标而覆盖 `2B` baseline、`4B` 正式 run、或 `E9` 第二轮正式 run

## 一句话版本

你现在最该做的，是把 `E10 v2` 作为阶段性改进结果写入材料，然后用 `v3` 的数据+约束修复方案继续推进，而不是回头修改 retrieval 或更换评测协议。
