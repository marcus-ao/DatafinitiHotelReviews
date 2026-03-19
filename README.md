# 基于酒店评论知识库的推荐智能体研究

> 本科毕业设计 · 数据处理与知识库构建模块

## 项目状态

当前数据处理主链路已经在本地完成端到端跑通，PostgreSQL 与 ChromaDB 均通过验收。

## 项目概述

本项目基于 [Datafiniti Hotel Reviews](https://www.kaggle.com/datasets/datafiniti/hotel-reviews) 公共数据集，构建可检索、可聚合、可验证的酒店评论知识库，为后续推荐智能体、RAG 检索和对话式推荐实验提供数据底座。

当前实现采用：

- PostgreSQL 保存结构化实体、评论、句子、方面情感标签和酒店方面画像
- ChromaDB 保存句子级向量索引与元数据过滤能力
- `MoritzLaurer/deberta-v3-base-zeroshot-v2.0` 做方面补充分类
- `nlptown/bert-base-multilingual-uncased-sentiment` 做句子级情感分类
- `BAAI/bge-small-en-v1.5` 做证据句向量编码

## 实测结果

### 最终统计

| 指标 | 实测结果 |
|------|----------|
| 原始评论 | 10,000 |
| 城市过滤后评论 | 6,171 |
| 清洗后评论 | 5,947 |
| 覆盖城市 | 10 |
| 覆盖州 | 8 |
| 有效酒店 | 146 |
| 句子数 | 51,813 |
| 方面标签数 | 63,085 |
| 酒店方面画像行数 | 876 |
| evidence_index 行数 | 51,813 |
| ChromaDB 记录数 | 51,813 |
| `09_validate.py` | 28/28 通过 |

### 城市分布

| 城市 | 评论数 |
|------|--------|
| San Diego | 1172 |
| San Francisco | 784 |
| New Orleans | 782 |
| Atlanta | 732 |
| Orlando | 716 |
| Seattle | 566 |
| Chicago | 449 |
| Honolulu | 295 |
| Dallas | 228 |
| Anaheim | 223 |

### 时间桶分布

| recency_bucket | 评论数 |
|----------------|--------|
| older | 4374 |
| recent_2y | 950 |
| recent_1y | 569 |
| recent_90d | 54 |

### 方面与情感分布

| 方面 | 标签数 |
|------|--------|
| general | 19644 |
| room_facilities | 15713 |
| service | 9070 |
| location_transport | 8784 |
| value | 3743 |
| cleanliness | 3581 |
| quiet_sleep | 2550 |

| 情感 | 标签数 |
|------|--------|
| positive | 40900 |
| negative | 11298 |
| neutral | 10887 |

| 标签来源 | 数量 |
|----------|------|
| rule | 60967 |
| zeroshot | 2118 |

### 验收记录

本地最终验收通过的关键结果如下：

```text
[OK] ChromaDB 写入完成: 51813 条记录
[OK] PostgreSQL 数据加载完成
验证结果: 28/28 通过 [OK] 全部通过！
```

`08_load_to_postgres.py` 最终导入结果：

```text
kb.hotel: 146 行
kb.review: 5947 行
kb.sentence: 51813 行
kb.aspect_sentiment: 63085 行
kb.hotel_aspect_profile: 876 行
kb.evidence_index: 51813 行
```

`09_validate.py` 最终通过的关键检查包括：

- 中间产物文件齐全
- `aspect_labels` 与 `aspect_sentiment` 覆盖全部句子
- `hotel_aspect_profile = 酒店数 × 6`
- ChromaDB 记录数与 `evidence_index` 一致
- ChromaDB 支持 `city + aspect + sentiment` 过滤查询
- PostgreSQL 6 张核心表行数与中间产物一致

## 目录结构

```text
DatafinitiHotelReviews/
├── raw_data/
│   └── Datafiniti_Hotel_Reviews.csv
├── data/
│   ├── intermediate/
│   └── chroma_db/
├── scripts/
│   ├── utils.py
│   ├── 01_load_filter.py
│   ├── 02_clean_and_dedupe.py
│   ├── 03_split_sentences.py
│   ├── 04_classify_aspects.py
│   ├── 05_classify_sentiment.py
│   ├── 06_build_profiles.py
│   ├── 07_build_vector_index.py
│   ├── 08_load_to_postgres.py
│   └── 09_validate.py
├── sql/
│   └── init_schema.sql
├── configs/
│   ├── params.yaml
│   └── db.yaml
├── tests/
├── notebooks/
├── plans/
│   ├── plan.md
│   ├── implementation-guide.md
│   └── chat.md
├── requirements.txt
├── .gitignore
└── README.md
```

## 中间产物契约

当前流水线固定输出链路如下：

```text
city_filtered.pkl
-> cleaned_reviews.pkl
-> sentences.pkl
-> aspect_labels.pkl
-> aspect_sentiment.pkl
-> hotel_profiles.pkl
-> evidence_index.pkl
```

其中：

- `city_filtered.pkl`: 城市过滤后的原始评论子集
- `cleaned_reviews.pkl`: 清洗、去重、日期标准化和酒店过滤后的评论主表
- `sentences.pkl`: 句子切分结果
- `aspect_labels.pkl`: 长表形式的方面标签
- `aspect_sentiment.pkl`: 长表形式的方面情感标签
- `hotel_profiles.pkl`: 146 家酒店的 6 个核心方面画像
- `evidence_index.pkl`: 句子级证据元数据，和 ChromaDB 一一对应

## 快速开始

### 1. 环境准备

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux / Mac
source .venv/bin/activate

pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 2. 数据库配置

```bash
cp configs/db.yaml.example configs/db.yaml
```

然后编辑 `configs/db.yaml`，填入：

- `host`
- `port`
- `dbname`
- `user`
- `password`
- `schema`

如需临时覆盖密码，可设置环境变量：

```powershell
$env:HOTEL_DB_PASSWORD="your_password"
```

### 3. 初始化 PostgreSQL Schema

```bash
psql -U <db_user> -d hotel_reviews_kb -f sql/init_schema.sql
```

### 4. 运行完整流水线

```bash
python scripts/01_load_filter.py
python scripts/02_clean_and_dedupe.py
python scripts/03_split_sentences.py
python scripts/04_classify_aspects.py
python scripts/05_classify_sentiment.py
python scripts/06_build_profiles.py
python scripts/07_build_vector_index.py
python scripts/08_load_to_postgres.py
python scripts/09_validate.py
```

如果 `04` 和 `05` 在本地过慢，可以在 Colab 或 GPU 服务器运行，再把生成的中间产物同步回本地继续执行 `07-09`。

## 跨平台运行说明

### Colab / 本地混合运行

- 如果 `04_classify_aspects.py` 和 `05_classify_sentiment.py` 在本地 CPU 上过慢，建议在 Colab 跑 `04-06`，再把 `aspect_labels.pkl`、`aspect_sentiment.pkl`、`hotel_profiles.pkl` 拉回本地。
- `07_build_vector_index.py` 推荐最终在本地重跑一次，保证 ChromaDB 持久化目录与本机环境一致。

### 常见问题

- Colab 生成的部分 `.pkl` 在本地读取时可能依赖 `pyarrow`，因此 `requirements.txt` 已显式加入 `pyarrow>=23.0.1`。
- 若 Windows 下直接读取从 Colab 拷回的 `data/chroma_db/` 出现 SQLite 或 Chroma `disk I/O error`，推荐删除或备份旧目录后，在本地重跑 `python scripts/07_build_vector_index.py`。
- `03_split_sentences.py` 已支持在找不到 `en_core_web_sm` 时回退到 `spacy.blank("en") + sentencizer`，但论文最终实验建议安装官方模型后再重跑，以获得更稳定的分句质量。
- Windows 下 HuggingFace 会提示 symlink warning，这通常不影响运行，只会增加缓存空间占用。

## 当前数据库结构

当前 PostgreSQL `kb` schema 下包含 6 张核心表和 2 个视图：

- `hotel`
- `review`
- `sentence`
- `aspect_sentiment`
- `hotel_aspect_profile`
- `evidence_index`
- `v_hotel_overview`
- `v_evidence_full`

## 参考文档

- [最终实施计划](plans/plan.md)
- [当前实施指南](plans/implementation-guide.md)
- [Datafiniti Hotel Reviews Dataset](https://www.kaggle.com/datasets/datafiniti/hotel-reviews)
