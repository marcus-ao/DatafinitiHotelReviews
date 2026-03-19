# 基于酒店评论知识库与高效语言模型微调的推荐智能体构建研究

> 本科毕业设计 · 数据处理与知识库构建模块

## 项目概述

本项目以 [Datafiniti Hotel Reviews](https://www.kaggle.com/datasets/datafiniti/hotel-reviews) 公共数据集为基础，构建细粒度酒店评论知识库，支撑基于 RAG 的推荐智能体与 LLM 微调研究。

**数据规模（v3.0）**

| 指标 | 数值 |
|------|------|
| 原始评论 | 10,000 条 |
| 实验城市 | 10 个 / 8 个州 |
| 实验酒店 | ~146 家（≥5 条评论）|
| 实验评论 | ~5,850 条 |
| 句子粒度 | ~44,000 句 |
| 方面标签 | ~37,000 个 |

**覆盖城市**：San Diego · San Francisco · New Orleans · Atlanta · Orlando · Seattle · Chicago · Honolulu · Dallas · Anaheim

## 技术栈

| 组件 | 选型 |
|------|------|
| 关系数据库 | PostgreSQL 16（kb schema）|
| 向量数据库 | ChromaDB 0.5+ |
| 分句 | spaCy `en_core_web_sm` |
| 方面分类 | `MoritzLaurer/deberta-v3-base-zeroshot-v2.0` |
| 情感分析 | `nlptown/bert-base-multilingual-uncased-sentiment` |
| 向量编码 | `BAAI/bge-small-en-v1.5`（384d）|
| Reranker | `BAAI/bge-reranker-base` |

## 目录结构

```
DatafinitiHotelReviews/
├── raw_data/                    # 原始数据（不提交 git）
│   └── Datafiniti_Hotel_Reviews.csv
├── data/
│   ├── intermediate/            # 中间处理文件（不提交 git）
│   └── chroma_db/               # ChromaDB 持久化（不提交 git）
├── scripts/
│   ├── utils.py                 # 公共工具函数
│   ├── 01_load_filter.py        # 数据加载与城市过滤
│   ├── 02_clean_reviews.py      # 清洗管理者回复 + 噪声
│   ├── 03_split_sentences.py    # spaCy 分句
│   ├── 04_classify_aspects.py   # 方面分类（规则 + 零样本）
│   ├── 05_classify_sentiment.py # 句子级情感分析
│   ├── 06_build_profiles.py     # 酒店方面画像聚合
│   ├── 07_build_vector_index.py # BGE 编码 + ChromaDB 索引
│   ├── 08_export_to_postgres.py # 导出到 PostgreSQL
│   └── 09_validate.py           # 数据质量验收
├── sql/
│   └── init_schema.sql          # PostgreSQL DDL
├── configs/
│   ├── params.yaml              # 所有可调参数
│   └── db.yaml                  # 数据库连接配置（不提交 git）
├── notebooks/                   # 探索分析 Notebook
├── plans/                       # 规划文档
│   ├── plan.md                  # 战略计划 v3.0
│   └── implementation-guide.md  # 从 0 到 1 实施指南
├── requirements.txt
├── .gitignore
└── README.md
```

## 快速开始

### 1. 环境准备

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/Mac
source .venv/bin/activate

pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 2. 配置数据库连接

```bash
cp configs/db.yaml.example configs/db.yaml
# 编辑 configs/db.yaml 填入 PostgreSQL 连接信息
```

### 3. 初始化 PostgreSQL Schema

```bash
psql -U hotel_user -d hotel_reviews_kb -f sql/init_schema.sql
```

### 4. 运行数据处理流水线

```bash
python scripts/01_load_filter.py
python scripts/02_clean_reviews.py
python scripts/03_split_sentences.py
python scripts/04_classify_aspects.py   # GPU 推荐，约 30min
python scripts/05_classify_sentiment.py
python scripts/06_build_profiles.py
python scripts/07_build_vector_index.py
python scripts/08_export_to_postgres.py
python scripts/09_validate.py
```

**GPU 总耗时**: ~40 分钟  
**CPU 总耗时**: ~4 小时（瓶颈在 Step 4 零样本分类）

### 5. 验收指标

`09_validate.py` 会自动检查：

- [ ] hotel 表：≥120 家（≥5 评论）
- [ ] review 表：≥5,000 条（去重后）
- [ ] sentence 表：≥35,000 句
- [ ] aspect_sentiment：≥30,000 标签
- [ ] unclassified 率：< 15%
- [ ] 日期有效率：100%
- [ ] ChromaDB 条目数与 sentence 表一致
- [ ] PostgreSQL 外键完整性：0 违例

## 参考文档

- [战略计划 v3.0](plans/plan.md)
- [从 0 到 1 实施指南](plans/implementation-guide.md)
- [Datafiniti Hotel Reviews Dataset](https://www.kaggle.com/datasets/datafiniti/hotel-reviews)
