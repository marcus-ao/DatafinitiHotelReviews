# E6 Qrels Labeling Guide

本文件只解释 `experiments/labels/e6_qrels/qrels_pool.csv` 该怎么标。

当前正式口径：

- 只标 `qrels_pool.csv`
- 标完后再冻结成 `qrels_evidence.jsonl`
- `qrels_evidence.jsonl` 会被 `E6/E7/E8` 共用

## 1. 你要打开哪个文件

- `experiments/labels/e6_qrels/qrels_pool.csv`

这是当前唯一正式人工标注入口。

## 2. 哪些列不要改

不要改这些列：

- `query_id`
- `city`
- `query_type`
- `target_aspect`
- `target_role`
- `query_text_zh`
- `query_en_target`
- `hotel_id`
- `hotel_name`
- `sentence_id`
- `sentence_text`
- `sentence_aspect`
- `sentence_sentiment`
- `review_id`
- `pooled_from`

你只需要填写：

- `relevance`
- `aspect_match`
- `polarity_match`
- `notes`

## 3. 四个关键字段怎么理解

### 3.1 `target_aspect`

当前这条评测单元的目标方面。

例如：

- `cleanliness`
- `service`
- `quiet_sleep`

### 3.2 `target_role`

表示这条单元来自哪类约束：

- `focus`：用户想重点要这个方面
- `avoid`：用户想避免这个方面的问题

### 3.3 `pooled_from`

这句证据是从哪些官方检索模式里合并进来的。

你不用改它，它只是帮你审计来源。

### 3.4 `sentence_aspect`

这是系统当前给这句打的主方面标签。

它只能当参考，不能当人工真值。

## 4. 需要填写的三列

### 4.1 `relevance`

可选值固定为：

- `0`
- `1`
- `2`

填写规则沿用总 rubric：

- `0`：基本无关，不能作为当前单元的证据
- `1`：有一点相关，但信息弱、泛、或只部分相关
- `2`：高度相关，能直接作为当前单元的证据

### 4.2 `aspect_match`

可选值：

- `1`
- `0`

填写规则：

- `1`：句子确实在说 `target_aspect`
- `0`：句子没稳定说到 `target_aspect`

不要因为系统的 `sentence_aspect` 是某个标签，就直接跟着填。

### 4.3 `polarity_match`

可选值：

- `1`
- `0`

#### 对 `focus`

当 `target_role = focus` 时：

- `1`：句子的语义方向和用户想要的偏好一致
- `0`：句子的语义方向和用户想要的偏好冲突，或看不出支持

例子：

- 想要 `cleanliness`
  - “room was very clean” -> `1`
  - “bathroom was dirty” -> `0`

#### 对 `avoid`

当 `target_role = avoid` 时，当前第一轮正式口径固定为“问题导向解释”：

- `1`：句子明确在说用户想避免的那类问题或风险
- `0`：句子虽然提到该方面，但没有落到用户想避免的问题表达上

例子：

- `avoid service`
  - “front desk was rude and unhelpful” -> `1`
  - “staff were friendly” -> `0`
- `avoid room_facilities`
  - “air conditioner was broken” -> `1`
  - “room was spacious” -> `0`
- `avoid quiet_sleep`
  - “street noise kept us awake” -> `1`
  - “room was quiet” -> `0`

这个口径的目的，是让 `avoid` 单元先稳定评估“系统能不能召回风险证据”，而不是同时混入“正向 reassurance”。

## 5. 推荐的实际标注顺序

每一行建议按这个顺序看：

1. 先看 `query_text_zh`
2. 再看 `target_aspect + target_role`
3. 再读 `sentence_text`
4. 先判断 `aspect_match`
5. 再判断 `polarity_match`
6. 最后给 `relevance`

不要先凭系统标签或 `pooled_from` 猜答案。

## 6. 一个实用判断原则

如果你犹豫一条句子到底该不该给高分，可以问自己：

- 这句如果直接放到论文里的“证据示例”里，读者会不会一眼看懂它和当前目标方面有关？

如果不会，一般不要给 `2`。

## 7. 标完后下一步

标完 `qrels_pool.csv` 后，运行：

```bash
cd <repo-root>
source venv/bin/activate
python -m scripts.evaluation.run_experiment_suite --task e6_freeze_qrels
```

成功后再继续：

```bash
python -m scripts.evaluation.run_experiment_suite --task e6_retrieval
python -m scripts.evaluation.run_experiment_suite --task e7_reranker
python -m scripts.evaluation.run_experiment_suite --task e8_fallback
```
