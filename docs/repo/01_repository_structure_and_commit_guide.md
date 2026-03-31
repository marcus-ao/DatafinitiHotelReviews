# 仓库结构与提交说明

更新时间：2026-03-31

本文件说明当前仓库的整体架构、目录分层、长期保留策略，以及这次 Git 提交前建议重点检查的内容。

## 当前仓库总体结构

```text
DatafinitiHotelReviews/
├── configs/
│   ├── db.example.yaml
│   └── params.yaml
├── docs/
│   ├── plans/
│   ├── repo/
│   └── status/
├── experiments/
│   ├── assets/
│   ├── labels/
│   │   ├── e1_aspect_reliability/
│   │   ├── e4_clarification/
│   │   └── e6_qrels/
│   ├── reports/
│   └── runs/
├── scripts/
│   ├── evaluation/
│   ├── pipeline/
│   └── shared/
├── sql/
├── tests/
├── README.md
├── requirements.txt
└── .gitignore
```

## 功能分块说明

### `configs/`

- `params.yaml`：主配置文件
- `db.example.yaml`：数据库配置模板
- `db.yaml`：本地私密配置，只本地保留，不进入 Git

### `scripts/pipeline/`

数据处理与知识库构建主链路，共 9 个步骤：

- `load_and_filter_reviews.py`
- `clean_and_dedupe_reviews.py`
- `split_reviews_into_sentences.py`
- `classify_sentence_aspects.py`
- `classify_aspect_sentiment.py`
- `build_hotel_aspect_profiles.py`
- `build_evidence_vector_index.py`
- `load_kb_to_postgres.py`
- `validate_kb_assets.py`

### `scripts/evaluation/`

实验与评测入口：

- `prepare_experiment_assets.py`
- `evaluate_e1_aspect_reliability.py`
- `evaluate_e2_candidate_selection.py`
- `evaluate_e3_e5_behavior.py`
- `evaluate_e6_e8_retrieval.py`
- `run_experiment_suite.py`

### `scripts/shared/`

- `project_utils.py`：项目通用工具函数与路径解析
- `experiment_utils.py`：实验目录常量、哈希与读写工具
- `experiment_schemas.py`：Pydantic schema 与 run log 结构

### `experiments/assets/`

冻结实验资产：

- `frozen_config.yaml`
- `frozen_split_manifest.json`
- `judged_queries.jsonl`
- `slot_gold.jsonl`
- `clarify_gold.jsonl`
- `annotation_rubrics.md`

### `experiments/labels/`

- `e1_aspect_reliability/`：E1 正式样本、gold 与结果
- `e4_clarification/`：E4 人工问题质量审计资产
- `e6_qrels/`：E6-E8 共用 qrels 标注入口与冻结结果

### `experiments/reports/`

阶段性论文材料汇总：

- `01_aspect_kb_stage_1_summary.md`
- `02_aspect_kb_stage_2_summary.md`

### `experiments/runs/`

当前只保留正式 run：

- `e2_770d3e0e2f4ded57_20260329T124258+0000`
- `e5_9a94daa5a6a31d8a_20260330T155246+0000`
- `e6_a98d0773422d5f2f_20260330T150721+0000`
- `e7_4511a1b33877007a_20260330T151015+0000`
- `e8_b610a6c12fd1195d_20260330T151133+0000`

### `docs/`

- `docs/plans/`：研究计划与实施说明
- `docs/status/`：当前进度、下一步与人工介入手册
- `docs/repo/`：仓库结构与提交说明

### `tests/`

当前有效测试：

- `test_pipeline_sentence_splitting.py`
- `test_pipeline_aspect_classification.py`
- `test_shared_project_utils.py`

## 已清理内容

本轮已移除以下低价值或无长期保留意义内容：

- `.DS_Store`
- 仓库源码目录内的 `__pycache__/`
- 空 `notebooks/`
- `tests/_tmp_db.yaml`
- `experiments/runs/.gitkeep`
- 未完成 run：`e3_a80ffb26befca274_20260330T160203+0000`
- 早期 E2 bootstrap / 冒烟 / 重复 run

同时已完成文档边界收口：

- 旧的仓库外计划文档已并入 `docs/plans/`
- 旧的仓库外状态文档已并入 `docs/status/`
- `docs/` 已成为仓库内唯一长期文档中心

## 本地保留但不提交的内容

以下内容继续保留在本地以便复现，但不应进入本次 Git 提交：

- `raw_data/`
- `data/intermediate/`
- `data/chroma_db/`
- `venv/`
- `configs/db.yaml`

这些内容已经通过 `.gitignore` 做路径级排除。

## 当前推荐提交策略

推荐分成两次提交，便于历史清晰：

1. `chore(repo): prune local artifacts and retain only official experiment outputs`
2. `docs(repo): consolidate docs and normalize repository structure`

如果只做一次提交，推荐消息：

- `chore(repo): clean workspace and normalize repository structure before thesis-stage commit`

## 提交前手动检查清单

建议你在真正提交前手动执行并确认：

```bash
git status --short --ignored
git diff --stat
git add -n .
```

重点确认：

- `raw_data/`、`data/intermediate/`、`data/chroma_db/`、`venv/`、`configs/db.yaml` 没有被意外纳入
- `experiments/runs/` 只剩 5 个正式 run
- 根 README、`experiments/README.md` 和 `docs/status/` 的路径表述已经对齐
