# 当前实施指南

> 版本：v2.0-final
> 前置文档：`plans/plan.md`
> 目标：用当前仓库和当前脚本，稳定复现已通过验收的数据处理模块

---

## 第0步：环境准备

### 0.1 Python 环境

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux / Mac
source .venv/bin/activate

pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

说明：

- `requirements.txt` 已包含 `pyarrow`
- 若未安装 `en_core_web_sm`，`03_split_sentences.py` 会自动回退到 `sentencizer`

### 0.2 HuggingFace 模型预下载（可选）

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sentence_transformers import SentenceTransformer

AutoTokenizer.from_pretrained("MoritzLaurer/deberta-v3-base-zeroshot-v2.0")
AutoModelForSequenceClassification.from_pretrained("MoritzLaurer/deberta-v3-base-zeroshot-v2.0")

AutoTokenizer.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")
AutoModelForSequenceClassification.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")

SentenceTransformer("BAAI/bge-small-en-v1.5")
```

### 0.3 数据库配置

```bash
cp configs/db.yaml.example configs/db.yaml
```

填写 `host / port / dbname / user / password / schema`。  
如需临时覆盖密码，可设置环境变量 `HOTEL_DB_PASSWORD`。

### 0.4 初始化 PostgreSQL

```bash
psql -U <db_user> -d hotel_reviews_kb -f sql/init_schema.sql
```

### 0.5 当前关键配置说明

```yaml
aspect:
  zeroshot_model: "MoritzLaurer/deberta-v3-base-zeroshot-v2.0"
  zeroshot_threshold: 0.45
  zeroshot_min_char_len: 30
  zeroshot_batch_size: 32

sentiment:
  model: "nlptown/bert-base-multilingual-uncased-sentiment"
  batch_size: 64
  # 5分类 → 3分类映射: 1-2→negative, 3→neutral, 4-5→positive

recency:
  reference_date: "2019-01-30"
  buckets:
    recent_90d: 1.2
    recent_1y: 1.0
    recent_2y: 0.9
    older: 0.8

embedding:
  model: "BAAI/bge-small-en-v1.5"
  chroma_collection: "hotel_evidence"
  chroma_persist_dir: "data/chroma_db"
```

---

## 第一步：加载与城市筛选

**脚本**: `scripts/01_load_filter.py`  
**输入**: `raw_data/Datafiniti_Hotel_Reviews.csv`  
**输出**: `data/intermediate/city_filtered.pkl`

当前实现完成：

- 读取原始 10,000 条评论
- 统一字段命名
- 过滤 10 个实验城市
- 生成 `hotel_id` 与 `review_id`

当前实测结果：

```text
原始评论: 10000
城市过滤后: 6171
唯一酒店: 211
输出文件: data/intermediate/city_filtered.pkl
```

---

## 第二步：评论清洗

**脚本**: `scripts/02_clean_and_dedupe.py`  
**输入**: `data/intermediate/city_filtered.pkl`  
**输出**: `data/intermediate/cleaned_reviews.pkl`

当前实现完成：

- 无效评论过滤
- 管理者回复切分与清洗
- 精确去重
- 日期、评分与时间桶标准化
- 酒店级质量过滤

当前实测结果：

```text
输入评论: 6171
有效评论: 6170
管理者回复检测: 2223 / 6170 (36.0%)
去重后: 6086
最终保留评论: 5947
最终保留酒店: 146
```

---

## 第三步：分句

**脚本**: `scripts/03_split_sentences.py`  
**输入**: `data/intermediate/cleaned_reviews.pkl`  
**输出**: `data/intermediate/sentences.pkl`

当前实现完成：

- 使用 spaCy 分句
- 先分句、再合并短碎片、最后做最短长度过滤
- 若缺少 `en_core_web_sm`，自动回退到 `sentencizer`

当前实测结果：

```text
输入评论: 5947
输出句子: 51813
均值: 8.7 句 / 评论
```

---

## 第四步：方面分类

**脚本**: `scripts/04_classify_aspects.py`  
**输入**: `data/intermediate/sentences.pkl`  
**输出**: `data/intermediate/aspect_labels.pkl`

当前实现：

- 先用关键词规则分类
- 未命中句子再做 zero-shot 补充
- 输出长表，每行一个 `(sentence_id, aspect)`

当前实测结果：

```text
输入句子: 51813
规则命中: 30161 / 51813 (58.2%)
zero-shot 处理: 18622
输出标签: 63085
```

---

## 第五步：情感分类

**脚本**: `scripts/05_classify_sentiment.py`  
**输入**: `data/intermediate/aspect_labels.pkl`  
**输出**: `data/intermediate/aspect_sentiment.pkl`

当前实现：

- 先对唯一句子做一次情感推理
- 再按 `sentence_id` 回填到长表标签

当前实测结果：

```text
输出标签: 63085
positive: 40900
negative: 11298
neutral: 10887
sentence vs review 一致率: 66.4%
```

---

## 第六步：聚合酒店方面画像

**脚本**: `scripts/06_build_profiles.py`  
**输入**: `data/intermediate/aspect_sentiment.pkl` + `cleaned_reviews.pkl`  
**输出**: `data/intermediate/hotel_profiles.pkl`

当前实现：

- 仅聚合 6 个核心方面
- 结合 `review_date / recency_bucket` 做时间加权
- 输出 `pos_count / neg_count / neu_count / final_aspect_score`

当前实测结果：

```text
标签数: 63085
评论数: 5947
画像行数: 876
final_aspect_score 均值: 19.749
```

---

## 第七步：向量编码与 ChromaDB 构建

**脚本**: `scripts/07_build_vector_index.py`  
**输入**: `sentences.pkl` + `aspect_sentiment.pkl` + `cleaned_reviews.pkl`  
**输出**: `evidence_index.pkl` + `data/chroma_db/`

当前实现：

- 先构造 `evidence_index.pkl`
- 再用 `BAAI/bge-small-en-v1.5` 编码全部句子
- 最后写入 ChromaDB 持久化目录
- 写入后立即做一次检索 smoke test

当前实测结果：

```text
句子数: 51813
标签数: 63085
evidence_index: 51813
ChromaDB: 51813
```

---

## 第八步：写入 PostgreSQL 数据库

**脚本**: `scripts/08_load_to_postgres.py`  
**输入**: 当前所有中间 `.pkl`  
**输出**: PostgreSQL `kb` schema

当前实现：

- 使用 `configs/db.yaml`
- 使用 `psycopg2`
- 先 `TRUNCATE` 再按外键顺序导入

导入顺序：

1. `hotel`
2. `review`
3. `sentence`
4. `aspect_sentiment`
5. `hotel_aspect_profile`
6. `evidence_index`

当前实测结果：

```text
kb.hotel: 146
kb.review: 5947
kb.sentence: 51813
kb.aspect_sentiment: 63085
kb.hotel_aspect_profile: 876
kb.evidence_index: 51813
```

---

## 第九步：验证与质量报告

**脚本**: `scripts/09_validate.py`

当前检查项：

- 中间文件存在性
- 数据量和字段契约
- 方面与情感分布
- ChromaDB 记录数与过滤查询
- PostgreSQL 6 张表的行数一致性

当前实测结果：

```text
验证结果: 28/28 通过 [OK] 全部通过！
```

---

## 执行顺序速查表

```bash
# 本地完整顺序
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

如采用 Colab / 本地混合运行，推荐顺序为：

```bash
# Colab 或 GPU 服务器
python scripts/04_classify_aspects.py
python scripts/05_classify_sentiment.py
python scripts/06_build_profiles.py

# 回到本地
python scripts/07_build_vector_index.py
python scripts/08_load_to_postgres.py
python scripts/09_validate.py
```

最终验收以 `09_validate.py = 28/28 通过` 为准。
