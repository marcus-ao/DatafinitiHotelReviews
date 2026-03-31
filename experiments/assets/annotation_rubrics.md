# Annotation Rubrics

## 1. Aspect / Sentiment Gold for E1

### 1.1 `aspect_gold`

- 标注单位：单条 `sentence_id`
- 标注格式：多个方面用英文分号分隔，例如 `service;value`
- 可选方面：
  - `location_transport`
  - `cleanliness`
  - `service`
  - `room_facilities`
  - `quiet_sleep`
  - `value`
  - `general`

### 1.2 `is_multi_aspect`

- `1`：句子同时明显涉及两个及以上核心方面
- `0`：句子只涉及一个核心方面，或只能稳定归到 `general`

### 1.3 `sentiment_gold`

- 可选值：`positive` / `negative` / `neutral` / `unclear`
- 规则：
  - `positive`：明确表达满意、推荐、正向体验
  - `negative`：明确表达不满、抱怨、反向体验
  - `neutral`：描述性、信息性表达，无明显情绪倾向
  - `unclear`：句子语义不完整、讽刺难判、或正负同时强烈混杂

## 2. Evidence Relevance / Verifiability

### 2.1 Relevance (`0/1/2`)

- `0`：与 query 或目标方面无关
- `1`：部分相关，但信息弱、泛泛而谈，或仅弱相关
- `2`：与目标方面高度相关，且可直接作为推荐依据

### 2.2 Evidence Verifiability (`0/1/2`)

- `0`：证据不支持对应理由，或理由与句子明显不一致
- `1`：证据部分支持，但较模糊、不完整，或只支持其中一部分说法
- `2`：证据明确支持推荐理由，且方面与情绪方向一致

## 3. Clarification Quality

### 3.1 `clarify_needed`

- `true`：当前输入不足以形成可执行查询，或包含明显冲突/主要依赖不支持约束
- `false`：当前槽位已足以形成候选过滤与最小检索

### 3.2 Clarification Question Quality (`1/2/3`)

- `1`：问题模糊、无针对性，不能有效补全缺失信息
- `2`：问题基本合理，但表达较泛，信息增益一般
- `3`：问题聚焦关键缺口，可直接帮助系统进入下一步推荐

## 4. 统一标注要求

- 所有标注结果必须保留 `query_id` / `sentence_id` / `hotel_id`
- 若无法稳定判断，优先记录在 `notes`，不要强行猜测
- 对同一批样本的复标必须保持同一口径，不临时改 rubric
