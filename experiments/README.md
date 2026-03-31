# Experiments Workspace

本目录集中存放毕业设计当前阶段的实验资产、人工标注入口、正式结果和阶段总结。

## 目录分层

- `assets/`
  冻结配置、切分、query gold、rubric 等长期复用资产。
- `labels/e1_aspect_reliability/`
  `E1` 的样本、正式 gold、审计说明和正式结果。
- `labels/e4_clarification/`
  `E4` 的人工问题质量审计资产目录。
- `labels/e6_qrels/`
  `E6-E8` 共用的 qrels 标注入口、冻结 qrels 和标注说明。
- `reports/`
  阶段性论文材料汇总。
- `runs/`
  只保留正式运行结果，每个 run 目录统一包含 `run_meta.json`、`results.jsonl`、`summary.csv`、`analysis.md`。

## 当前保留的正式实验覆盖

- `E1`：方面/情感标注可靠性
- `E2`：候选缩圈有效性
- `E5`：中文输入到英文检索表达桥接
- `E6`：方面引导检索 vs 朴素召回
- `E7`：reranker 消融
- `E8`：主通道与 fallback 边界

当前尚未保留正式 run 的实验：

- `E3`
- `E4`

## 冻结资产

- `assets/frozen_config.yaml`
- `assets/frozen_split_manifest.json`
- `assets/judged_queries.jsonl`
- `assets/slot_gold.jsonl`
- `assets/clarify_gold.jsonl`
- `assets/annotation_rubrics.md`

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
  由 `E4` 正式运行后生成；当前目录用于承接后续人工审计资产。

## 当前保留的正式 run

- `runs/e2_770d3e0e2f4ded57_20260329T124258+0000/`
- `runs/e5_9a94daa5a6a31d8a_20260330T155246+0000/`
- `runs/e6_a98d0773422d5f2f_20260330T150721+0000/`
- `runs/e7_4511a1b33877007a_20260330T151015+0000/`
- `runs/e8_b610a6c12fd1195d_20260330T151133+0000/`

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

补充说明：

- `E3/E4` 的 Base 组不建议在当前本地机器下载模型
- 推荐在云端 GPU 环境执行，再将正式 run 同步回本仓库

## 推荐阅读顺序

- `reports/01_aspect_kb_stage_1_summary.md`
- `reports/02_aspect_kb_stage_2_summary.md`
- `runs/e5_9a94daa5a6a31d8a_20260330T155246+0000/analysis.md`
- `../docs/repo/01_repository_structure_and_commit_guide.md`
