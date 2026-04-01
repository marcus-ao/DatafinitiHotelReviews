# 下一步该做什么

更新时间：2026-04-01

## 当前推荐推进顺序

请按下面顺序推进，不建议跳步：

1. 确认并保持默认检索配置固定为 `aspect_main_no_rerank`
2. 确认当前正式行为模型固定为 `Qwen/Qwen3.5-4B`
3. 保留已有 baseline、诊断 run 和 `4B` 全量正式 run，不覆盖、不删除
4. 以 `experiments/reports/05_behavior_stage_3_chapter_materials.md` 为主入口，完成行为章节正文写作
5. 仅在论文或答辩需要时，追加 `Qwen3.5-9B` 附录对比
6. 若不先补 `9B`，则先按当前仓库已实现入口冻结 `E9` full assets
7. 再跑 `E9` 三组生成约束对照与人工审计
8. `E9` 稳定后，再进入 `E10 / PEFT`

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

### 任务 4：进入 `E9` 证据约束生成

目标：

- 先回答“生成是否真正受证据约束”，不先碰 PEFT

执行入口：

- `docs/plans/03_generation_and_peft_phase_plan.md`

当前默认要求：

- 使用固定 `UserPreference`
- 使用固定 `EvidencePack`
- 不让 retrieval 波动混入 `E9`
- 默认基座模型固定为 `Qwen/Qwen3.5-4B`

截至 `2026-04-01`，当前仓库中已经可直接执行的顺序为：

1. 先确认本地 `venv` 可用，并优先从仓库根目录执行命令
2. 先跑：

```bash
python -m scripts.evaluation.run_experiment_suite --task e10_prepare_manifests
```

3. 再跑：

```bash
python -m scripts.evaluation.run_experiment_suite --task e9_freeze_assets
```

4. `E9` full assets 成功冻结后，再跑：

```bash
python -m scripts.evaluation.run_experiment_suite --task e9_generation_constraints
```

5. 生成完 run 后，立刻打开并审计：
   - `experiments/runs/e9_*/summary.csv`
   - `experiments/runs/e9_*/analysis.md`
   - `experiments/runs/e9_*/citation_verifiability_audit.csv`
   - `experiments/labels/e9_generation/citation_verifiability_audit.csv`

### 任务 5：`E9` 稳定后再进入 `E10 / PEFT`

目标：

- 用 `Base 4B vs PEFT 4B` 回答 `RQ2`

当前固定约束：

- 训练样本只来自 `train` 酒店
- 只做四类 SFT：
  - `preference_parse`
  - `clarification`
  - `constraint_honesty`
  - `feedback_update`
- 多轮能力只先做：
  - 单次澄清
  - 单次反馈更新

## 当前不建议启动的内容

在下面这些条件未满足前，不建议提前进入：

- 行为章节还没真正写进论文前：不要启动 `G1-G4`
- `E9` 还没稳定前：不要直接启动 `E10 / PEFT` 正式结果
- 当前阶段：不要把 `reranker` 或 `fallback` 再接回默认主流程
- 当前阶段：不要为了追求更高指标而覆盖 `2B` baseline 或 `4B` 正式 run

## 一句话版本

你现在最该做的，是先保留已冻结的行为主线不动，然后按仓库已经实现的入口依次执行 `e10_prepare_manifests -> e9_freeze_assets -> e9_generation_constraints`，把 `E9` 正式跑通并完成人工审计，之后再进入 `E10 / PEFT`。
