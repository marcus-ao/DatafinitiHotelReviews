# 基于酒店评论知识库的推荐智能体研究

本科毕业设计主仓库，当前聚焦“评论知识组织 + 检索实验 + 行为实验脚手架”的可复现研究底座。

## 当前状态

- 数据处理与知识库构建主链路已完成，验证结果为 `28/28`
- `E1`、`E2`、`E5`、`E6`、`E7`、`E8` 已完成正式结果并保留在仓库中
- `E3/E4` 代码与评测链路已准备完成，云端 Base 组计划切换为 `Qwen3.5-2B / 4B / 9B` 对比
- 默认检索配置已冻结为 `aspect_main_no_rerank`

## 仓库结构

```text
DatafinitiHotelReviews/
├── configs/
├── docs/
│   ├── deployment/
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

本地保留但默认不提交 Git 的目录：

- `raw_data/`
- `data/intermediate/`
- `data/chroma_db/`
- `venv/`
- `configs/db.yaml`

## 快速开始

### 1. 环境准备

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 2. 数据库配置

```bash
cp configs/db.example.yaml configs/db.yaml
psql -U <db_user> -d hotel_reviews_kb -f sql/init_schema.sql
```

### 3. 运行数据流水线

```bash
python -m scripts.pipeline.load_and_filter_reviews
python -m scripts.pipeline.clean_and_dedupe_reviews
python -m scripts.pipeline.split_reviews_into_sentences
python -m scripts.pipeline.classify_sentence_aspects
python -m scripts.pipeline.classify_aspect_sentiment
python -m scripts.pipeline.build_hotel_aspect_profiles
python -m scripts.pipeline.build_evidence_vector_index
python -m scripts.pipeline.load_kb_to_postgres
python -m scripts.pipeline.validate_kb_assets
```

### 4. 运行实验入口

```bash
python -m scripts.evaluation.prepare_experiment_assets
python -m scripts.evaluation.evaluate_e1_aspect_reliability
python -m scripts.evaluation.run_experiment_suite --task e2_candidates
python -m scripts.evaluation.run_experiment_suite --task e5_query_bridge
```

`E3/E4` 的下一轮正式云端 Base 组计划使用 `Qwen3.5-2B / 4B / 9B` 对比。行为实验脚本现在已经支持通过 OpenAI-compatible API 调用云端 `vLLM`，部署和执行步骤见 `docs/deployment/01_autodl_qwen35_behavior_runbook.md`。

## 文档入口

- 研究与实施计划：`docs/plans/`
- 云端部署与行为实验手册：`docs/deployment/01_autodl_qwen35_behavior_runbook.md`
- 当前进度与下一步：`docs/status/`
- 仓库结构与提交说明：`docs/repo/01_repository_structure_and_commit_guide.md`
- 实验资产与运行说明：`experiments/README.md`

## 当前稳定事实

- 数据规模：`10` 个城市、`146` 家酒店、`5947` 条评论、`51813` 条句子
- 默认下游检索模式：`aspect_main_no_rerank`
- 仓库只保留正式实验资产与正式 run，不保留 bootstrap、冒烟和未完成目录
