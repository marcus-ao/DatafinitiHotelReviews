# 下一步该做什么

更新时间：2026-04-01

## 当前推荐推进顺序

请按下面顺序推进，不建议跳步：

1. 确认并保持默认检索配置固定为 `aspect_main_no_rerank`
2. 确认当前正式行为模型固定为 `Qwen/Qwen3.5-4B`
3. 保留已有 baseline、诊断 run 和 `4B` 全量正式 run，不覆盖、不删除
4. 以 `experiments/reports/05_behavior_stage_3_chapter_materials.md` 为主入口，完成行为章节正文写作
5. 仅在论文或答辩需要时，追加 `Qwen3.5-9B` 附录对比
6. 将 `E9` 第二轮结果视为当前正式冻结结果，不再改 retrieval 主线
7. 进入 `E10 / PEFT` 评测骨架，准备 adapter metadata 与云端训练
8. adapter 准备好后，运行 `e10_base_vs_peft`

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

### 任务 5：进入 `E10 / PEFT` 评测骨架

目标：

- 在固定 `E9` eval units 与固定证据条件下，完成 `Base 4B vs PEFT 4B` 的评测入口准备
- 当前先做 adapter-ready 评测骨架，不在仓库里直接启动正式训练

当前推荐顺序为：

1. 先确认本地 `venv` 可用，并优先从仓库根目录执行命令
2. 先确认以下文件存在且内容正确：
   - `experiments/assets/sft_train_manifest.jsonl`
   - `experiments/assets/sft_dev_manifest.jsonl`
   - `experiments/assets/e10_train_config_template.json`
   - `experiments/assets/e10_adapter_metadata.template.json`
   - `docs/deployment/02_e10_peft_runbook.md`
3. 在云端完成 adapter 训练并回传后，准备 adapter metadata
4. 再跑：

```bash
python -m scripts.evaluation.run_experiment_suite --task e10_base_vs_peft
```

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
- 不改 retrieval 主线
- 不重生成 `E9` eval units

## 当前不建议启动的内容

在下面这些条件未满足前，不建议提前进入：

- 行为章节还没真正写进论文前：不要启动 `G1-G4`
- 在未准备好 adapter metadata 前：不要直接启动 `e10_base_vs_peft`
- 当前阶段：不要把 `reranker` 或 `fallback` 再接回默认主流程
- 当前阶段：不要为了追求更高指标而覆盖 `2B` baseline、`4B` 正式 run、或 `E9` 第二轮正式 run

## 一句话版本

你现在最该做的，是把 `E9` 第二轮结果视为当前正式冻结结果，然后准备 `E10 / PEFT` 的 adapter metadata 与云端训练产物，最后在固定 `E9` eval units 上运行 `e10_base_vs_peft`。
