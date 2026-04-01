# Experiments Workspace

本目录集中存放毕业设计当前阶段的实验资产、人工标注入口、正式结果、论文材料和下一阶段规划入口。

## 目录分层

- `assets/`
  冻结配置、切分、query gold、rubric 等长期复用资产。
- `assets/e3_diagnostic_query_ids.json`
  `E3 v2` 历史诊断子集。
- `assets/e4_diagnostic_query_ids.json`
  `E4 v2` 历史诊断子集。
- `labels/e1_aspect_reliability/`
  `E1` 的样本、正式 gold、审计说明和正式结果。
- `labels/e4_clarification/`
  `E4` 的人工问题质量审计资产目录，当前已补齐 `4B` 正式 run 的首轮人工评分。
- `labels/e9_generation/`
  `E9` 的 citation / evidence verifiability 审计目录，当前已创建字段模板与说明文件。
- `labels/e6_qrels/`
  `E6-E8` 共用的 qrels 标注入口、冻结 qrels 和标注说明。
- `reports/`
  阶段性论文材料汇总。
- `runs/`
  保留正式运行结果，以及少量具有长期引用价值的行为基线 / 诊断 run；每个 run 目录统一包含 `run_meta.json`、`results.jsonl`、`summary.csv`、`analysis.md`。

## 当前保留的正式实验覆盖

- `E1`：方面/情感标注可靠性
- `E2`：候选缩圈有效性
- `E3`：偏好解析正式结果，当前正式主模型为 `Qwen3.5-4B`
- `E4`：澄清触发正式结果，当前正式主模型为 `Qwen3.5-4B`
- `E5`：中文输入到英文检索表达桥接
- `E6`：方面引导检索 vs 朴素召回
- `E7`：reranker 消融
- `E8`：主通道与 fallback 边界

当前下一官方阶段：

- `E9`：证据约束生成
- `E10`：Base vs PEFT 行为对照

当前实现状态：

- `E9 / E10` 的代码入口已于 `2026-04-01` 接入仓库
- `e9_freeze_assets` 已通过 `limit-queries=2` 的本地 smoke 验证
- `e10_prepare_manifests` 已成功生成 `sft_train_manifest.jsonl` 与 `sft_dev_manifest.jsonl`
- `E9` 第二轮正式结果已冻结为：
  - `runs/e9_ecbcdbab690dc503_20260401T025012+0000/`
- `E10` 当前下一步是 adapter-ready 评测骨架，不是立即启动正式训练

对应规划文档：

- `../docs/plans/03_generation_and_peft_phase_plan.md`

云端行为实验操作手册：

- `../docs/deployment/01_autodl_qwen35_behavior_runbook.md`
  用于 `Qwen3.5-2B / 4B / 9B` 的 AutoDL 部署、API 冒烟、`E3/E4` 正式运行和结果回传。

## 冻结资产

- `assets/frozen_config.yaml`
- `assets/frozen_split_manifest.json`
- `assets/judged_queries.jsonl`
- `assets/slot_gold.jsonl`
- `assets/clarify_gold.jsonl`
- `assets/annotation_rubrics.md`
- `assets/sft_train_manifest.jsonl`
- `assets/sft_dev_manifest.jsonl`

`E9` 新增冻结资产入口：

- `assets/e9_generation_eval_units.jsonl`
- `assets/e9_generation_eval_query_ids.json`

说明：

- 这两个 `E9` 资产文件已经作为正式冻结评测资产使用
- 当前 `E9` 正式 run 固定复用这份 assets，不再边跑边重新检索

## 正式标注目录

### `labels/e1_aspect_reliability/`

- `aspect_sentiment_eval_sample.csv`
- `aspect_sentiment_gold.csv`
- `aspect_sentiment_gold_notes.md`
- `e1_metrics.json`
- `e1_report.md`

### `labels/e6_qrels/`

- `qrels_pool.csv`
- `qrels_evidence.jsonl`
- `e6_labeling_guide.md`
- `e6_labeling_log.md`

### `labels/e4_clarification/`

- `clarification_question_audit.csv`
  当前最新一轮 `E4` 审计副本，当前已经完成首轮人工评分。
- `clarification_question_audit_e4_4a15a89128a90d11_baseline.csv`
  `Qwen3.5-2B` baseline 的冻结快照，不随 rerun 覆盖。
- `clarification_question_audit_e4_55c8021e1119fb77_qwen35_4b_full.csv`
  `Qwen3.5-4B` 全量正式 run 的原始快照。
- `clarification_question_audit_e4_55c8021e1119fb77_qwen35_4b_reviewed.csv`
  `Qwen3.5-4B` 全量正式 run 的人工 reviewed 冻结副本。

### `labels/e9_generation/`

- `README.md`
  `E9` 审计口径说明。
- `citation_verifiability_audit.csv`
  当前最新 `E9` 审计副本。
- `citation_verifiability_audit_e9_ecbcdbab690dc503_qwen35_4b_reviewed.csv`
  `E9` 第二轮正式结果的 reviewed 冻结快照。

## 当前保留的正式主结果 run

- `runs/e2_770d3e0e2f4ded57_20260329T124258+0000/`
- `runs/e3_14928d821d811e86_20260331T122611+0000/`
- `runs/e4_55c8021e1119fb77_20260331T122648+0000/`
- `runs/e5_9a94daa5a6a31d8a_20260330T155246+0000/`
- `runs/e6_a98d0773422d5f2f_20260330T150721+0000/`
- `runs/e7_4511a1b33877007a_20260330T151015+0000/`
- `runs/e8_b610a6c12fd1195d_20260330T151133+0000/`
- `runs/e9_ecbcdbab690dc503_20260401T025012+0000/`

## 当前保留的历史 / 诊断行为 run

- `runs/e3_244aca8abf6345ad_20260331T072527+0000/`
- `runs/e3_da541f84770ed8ed_20260331T090311+0000/`
- `runs/e3_f62d907e600cfc14_20260331T120756+0000/`
- `runs/e4_4a15a89128a90d11_20260331T073016+0000/`
- `runs/e4_96e0e4afb24dab2d_20260331T091021+0000/`
- `runs/e4_f928a37444c1bf52_20260331T121012+0000/`
- `runs/e9_80e05af30f45b1f2_20260401T021215+0000/`

## 当前论文材料入口

- `reports/01_aspect_kb_stage_1_summary.md`
- `reports/02_aspect_kb_stage_2_summary.md`
- `reports/03_behavior_stage_1_qwen35_2b_baseline.md`
- `reports/04_behavior_stage_2_qwen35_4b_formal_summary.md`
- `reports/05_behavior_stage_3_chapter_materials.md`
- `reports/06_generation_stage_1_e9_formal_summary.md`

## 最小运行入口

建议统一从仓库根目录执行：

```bash
source venv/bin/activate
```

### 重生成冻结资产

```bash
python -m scripts.evaluation.prepare_experiment_assets
```

### 运行 E1

```bash
python -m scripts.evaluation.evaluate_e1_aspect_reliability
```

### 运行 E2

```bash
python -m scripts.evaluation.run_experiment_suite --task e2_candidates
```

### 运行 E6 / E7 / E8

```bash
python -m scripts.evaluation.run_experiment_suite --task e6_retrieval
python -m scripts.evaluation.run_experiment_suite --task e7_reranker
python -m scripts.evaluation.run_experiment_suite --task e8_fallback
```

### 运行 E5

```bash
python -m scripts.evaluation.run_experiment_suite --task e5_query_bridge
```

### 运行 E3 / E4

```bash
python -m scripts.evaluation.run_experiment_suite --task e3_preference
python -m scripts.evaluation.run_experiment_suite --task e4_clarification
```

### 冻结 E9 评测资产

```bash
python -m scripts.evaluation.run_experiment_suite --task e9_freeze_assets
```

如果只想先做最小 smoke：

```bash
python -m scripts.evaluation.run_experiment_suite --task e9_freeze_assets --limit-queries 2
```

### 运行 E9 生成约束评测

```bash
python -m scripts.evaluation.run_experiment_suite --task e9_generation_constraints
```

### 生成 E10 SFT manifests

```bash
python -m scripts.evaluation.run_experiment_suite --task e10_prepare_manifests
```

### 运行 E10 Base vs PEFT 骨架评测

```bash
python -m scripts.evaluation.run_experiment_suite --task e10_base_vs_peft
```

### 运行 E3 / E4 诊断子集

```bash
python -m scripts.evaluation.run_experiment_suite --task e3_preference --query-id-file experiments/assets/e3_diagnostic_query_ids.json
python -m scripts.evaluation.run_experiment_suite --task e4_clarification --query-id-file experiments/assets/e4_diagnostic_query_ids.json
```

补充说明：

- `Qwen3.5-2B` 的第一轮 `E3/E4` baseline 已归档
- `Qwen3.5-4B` 的全量 `E3/E4` 正式结果已经完成，并已成为当前行为实验主结果
- 最新 `E4` 审计已完成首轮人工评分，并冻结为 reviewed 副本
- `E3 / E4 / E5` 的章节材料已整理为 `reports/05_behavior_stage_3_chapter_materials.md`
- 当前默认 prompt 版本已切到：
  - `E3 = e3_v2_cn_slots_only`
  - `E4 = e4_v2_cn_decision_label_fewshot`
- 当前默认行为模型配置已冻结为 `Qwen/Qwen3.5-4B`
- `Qwen3.5-9B` 目前只作为可选附录模型，不是当前主线阻塞项
- `E9` 的 `candidate_hotels` 当前固定使用 `E2 B_final_aspect_score Top5`
- `E9` 资产冻结当前优先要求本地已有 `BAAI/bge-small-en-v1.5` 缓存；若本地缓存缺失，需先在可联网环境缓存 embedding 模型
- `E9` 第二轮正式结果已冻结；第一轮 `e9_80e05af30f45b1f2_20260401T021215+0000/` 仅保留为诊断对照
- `e10_base_vs_peft` 当前已实现 adapter-ready 评测骨架，但默认要求你先提供 adapter metadata
- 推荐在 AutoDL 云端 GPU 环境执行，再将正式 run 同步回本仓库
- 当前行为实验脚本已经支持 OpenAI-compatible API backend，云端配置方式详见 `../docs/deployment/01_autodl_qwen35_behavior_runbook.md`

## 推荐阅读顺序

- `reports/05_behavior_stage_3_chapter_materials.md`
- `reports/04_behavior_stage_2_qwen35_4b_formal_summary.md`
- `runs/e3_14928d821d811e86_20260331T122611+0000/analysis.md`
- `runs/e4_55c8021e1119fb77_20260331T122648+0000/analysis.md`
- `runs/e5_9a94daa5a6a31d8a_20260330T155246+0000/analysis.md`
- `../docs/plans/03_generation_and_peft_phase_plan.md`
- `../docs/repo/01_repository_structure_and_commit_guide.md`
