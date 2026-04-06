# 全量正式实验重跑前审计协议（E1-E10, G1-G4）

更新时间：2026-04-05

本文件用于在最终正式重跑 `E1-E10` 与 `G1-G4` 之前，对整个实验体系进行一轮系统性的**严谨性审计（experiment audit）**。其目标不是简单确认“代码能跑”，而是确认：

1. 每个实验的**研究目标**清晰且不漂移；
2. 每个实验的**数据选用、评测范围、qrels/gold、split、资产路径**完全正确；
3. 每个实验中的**控制变量**真正被控制；
4. 每个实验的**指标口径、统计方式、结果用途**一致且可追溯；
5. 最终产出的结果可以无保留地正式写入论文，而不是仅作为中间材料或 supporting evidence。

---

## 1. 审计总原则

本轮审计必须遵守以下五条原则：

1. **研究问题先于脚本执行**：先明确实验究竟在回答什么问题，再确认脚本是否与该问题严格一致。  
2. **控制变量先于结果对比**：若某组对比中有两个及以上变量同时变化，则该对比不应直接用于主结论。  
3. **资产冻结先于全量重跑**：query 集、gold/qrels、split、generation assets、PEFT manifest、adapter metadata 等关键资产必须先冻结，再运行正式实验。  
4. **同名指标口径必须唯一**：同一指标在 `summary / compare / pairwise_tests / chapter_report` 中不得发生语义漂移；若聚合尺度不同，必须显式标注。  
5. **Supporting evidence 与 decisive evidence 明确分层**：`E5-E10` 与 `G1-G4` 的角色不同，严禁在论文中混写成同等级证据。

---

## 2. 审计对象分层

### 2.1 前置验证层（Chapter 4）

| 实验 | 目标 | 当前定位 |
|---|---|---|
| `E1` | Aspect 分类与情感标注可靠性 | 前置验证，本轮正式重跑 |
| `E2` | 候选酒店缩圈可行性 | 前置验证，本轮正式重跑 |
| `E3` | 偏好解析行为稳定性 | 前置验证，本轮正式重跑 |
| `E4` | 澄清触发行为稳定性 | 前置验证，本轮正式重跑 |

### 2.2 RAG supporting evidence 层（Chapter 5）

| 实验 | 目标 | 当前定位 |
|---|---|---|
| `E5` | 中文直检 vs 英文桥接 | supporting evidence |
| `E6` | Aspect-guided vs Plain retrieval | supporting evidence |
| `E7` | Reranker 消融 | supporting evidence |
| `E8` | Fallback 消融 | supporting evidence |
| `E9` | 有无证据约束的生成差异 | supporting evidence |

### 2.3 PEFT supporting evidence 层（Chapter 6）

| 实验 | 目标 | 当前定位 |
|---|---|---|
| `E10 v1-v4` | 训练目标/数据设计对 PEFT 结果的影响 | supporting evidence |

### 2.4 决定性统一对比层（Chapter 7）

| 实验 | 目标 | 当前定位 |
|---|---|---|
| `G1-G4` | 在统一 2×2 矩阵下回答 `RQ1/RQ2/RQ3` | **decisive evidence** |
| `LLM Judge` | 自动盲评辅助证据 | decisive supplementary evidence |
| `Human Blind Review` | 人工盲评辅助证据 | decisive supplementary evidence |
| `Pairwise Statistical Tests` | 组间显著性/效应量 | decisive supplementary evidence |

---

## 3. 审计 Phase 设计

本轮审计建议严格按以下 `5` 个 Phase 推进，禁止跳步。

---

### Phase 0：建立实验审计主表

#### 目标

把 `E1-E10, G1-G4` 以及对应的 Judge / blind review / stats 统一登记到一张主表里，形成后续所有执行与写作的唯一索引。

#### 必备字段

每个实验至少记录：

- 实验编号
- 所属章节
- 对应研究问题
- 实验定位（前置验证 / supporting evidence / decisive evidence）
- query scope
- gold/qrels/labels 来源
- split 依据
- retrieval backend / candidate policy
- generation backend / model
- PEFT adapter / manifest
- 指标集合
- 是否需要统计检验
- 是否允许复用旧结果
- 本次是否必须重跑
- 论文最终用途

#### 完成标准

- 任意一个实验都能回答：**“它在回答什么问题、用什么数据、最终写到论文哪里去”**。

---

### Phase 1：数据与资产口径审计

#### Phase 1.1 Query 集与评测范围

#### 必查项

1. `E1/E2` 是否与 query 扩展无关；  
2. `E3/E4` 是否严格基于完整 `judged_queries.jsonl` 的可执行行为样本；  
3. `E5-E8` 是否固定使用 `40 core queries / 80 target units`；  
4. `E9/E10` supporting evidence 是否固定使用 `40`；  
5. `G1-G4` 是否固定使用当前正式冻结的 decisive query 集（当前为 `68 = 39 core + 29 robustness`，`q021 / q024` 作为 supporting boundary cases 排除）；  
6. 是否有任何路径会误混入：
   - `conflict`
   - `missing_city`
   - 旧 `E9 query_ids`
   - 非目标 query type。

#### 关键资产

- `experiments/assets/judged_queries.jsonl`
- `experiments/assets/g_eval_query_ids_68.json`
- `experiments/assets/e9_generation_eval_query_ids.json`
- retrieval / generation eval units 相关资产

#### 完成标准

- 每个实验的 query scope 被唯一化、文档化；
- supporting evidence 与 decisive evidence 的 query scope 不混淆。

---

#### Phase 1.2 Split 与数据泄漏审计

#### 必查项

1. 所有 retrieval / generation / PEFT 训练是否统一基于 `frozen_split_manifest.json`；  
2. `E10 v1-v4` 的 train/dev manifest 是否严格不包含 test 酒店；  
3. `G1-G4` generation eval units 是否只基于 test split 酒店候选；  
4. `G` qrels 标注是否不会反向污染训练侧。

#### 完成标准

- 可明确证明：**训练数据与评测酒店没有交叉泄漏**；
- `G1-G4` 四组共享完全相同的评测酒店范围。

---

#### Phase 1.3 Gold / qrels / labels 资产审计

#### 必查项

1. `E1 gold`、`E6 qrels`、`G qrels` 是否物理隔离；  
2. `G qrels` 是否真的覆盖 `70` 条 query；  
3. robustness 30 条是否按最新决定纳入正式 qrels；  
4. `validate_g_qrels()` 是否足以检查 coverage 与基本 contract；  
5. qrels/gold 的 freeze 时间、路径、用途是否明确。

#### 完成标准

- 形成一份 “gold / qrels 资产登记表”；
- 不同实验之间不存在资产覆盖或错用。

---

### Phase 2：控制变量审计

#### Phase 2.1 Retrieval 对照变量审计（E5-E8, G retrieval）

#### 审查目标

确认 retrieval 层实验只改变了被研究变量。

#### 必查维度

- retrieval_mode
- candidate_policy
- candidate hotel scope
- qrels source
- dense / reranker top-k
- reranker on/off
- fallback on/off
- config_hash

#### 特别要求

- `E5` 只能改变 query bridge；
- `E6` 只能改变 retrieval strategy；
- `E7` 只能改变 reranker；
- `E8` 只能改变 fallback；
- `G1/G3` 共用 plain retrieval 正式结果；
- `G2/G4` 共用 aspect retrieval 正式结果。

#### 完成标准

- 输出一张 retrieval-controlled variables matrix。

---

#### Phase 2.2 Generation 对照变量审计（E9/E10/G1-G4）

#### 必查项

所有 generation compare 必须逐项核查：

- eval units 是否一致
- EvidencePack 来源是否一致
- model backend 是否一致
- 推理参数（temperature / max_new_tokens）是否一致
- prompt schema 是否一致
- recommendation / reasons 数量上限是否一致
- parse / repair / schema validate 路径是否一致

#### 关键 pair 必须满足的控制要求

- `E9 B vs D`：只改 evidence availability
- `E10 base vs PEFT`：只改 adapter
- `G1 vs G2`：只改 retrieval variant
- `G1 vs G3`：只改 PEFT
- `G2 vs G4`：只改 PEFT
- `G3 vs G4`：只改 retrieval variant

#### 完成标准

- 任一主 compare 不允许存在“两个变量同时变化”的情况。

---

#### Phase 2.3 PEFT 路线变量审计

#### 必查项

- `exp01 / exp02 / exp03 / exp04`
- `manifest v1-v4`
- base model
- qlora hyperparams
- train/dev split
- inference backend
- compare protocol

#### 目标

证明 `E10 v1-v4` 的差异主要来自**训练数据构成差异**，而不是训练脚本、推理协议或评测资产漂移。

#### 完成标准

- 可形成 “PEFT 迭代控制变量说明表”，用于论文第六章。

---

### Phase 3：指标与统计口径审计

#### Phase 3.1 指标定义总表

必须为以下指标逐个建立定义记录：

##### 检索层
- Aspect Recall@5
- nDCG@5
- Precision@5
- MRR@5
- Evidence Diversity@5
- Retrieval Latency

##### 生成层
- Citation Precision
- Evidence Verifiability Mean
- Schema Valid Rate
- Recommendation Coverage
- Aspect Alignment Rate
- Hallucination Rate
- Unsupported Honesty Rate

##### Judge / Human
- Relevance
- Traceability
- Fluency
- Completeness
- Honesty
- Overall Quality
- Evidence Credibility
- Practical Value

每个指标至少注明：

- 定义
- 计算函数
- 聚合维度
- higher_is_better
- 适用实验
- 结果文件来源

---

#### Phase 3.2 summary / compare / stats / report 一致性核查

#### 必查项

对每个关键指标，逐一检查其在以下产物中的定义与聚合是否一致：

- `summary.csv`
- `compare summary.csv`
- `group_score_map.json`
- `pairwise_tests.csv`
- `g_generation_summary.csv`
- `analysis.md`

#### 重点高风险指标

- `unsupported_honesty_rate`
- `hallucination_rate`
- `schema_valid_rate`
- `evidence_verifiability_mean`

#### 完成标准

- 同名指标若存在不同聚合口径，必须显式写出差异；
- 不允许出现“同名同表述但不同语义”的情况。

---

#### Phase 3.3 统计检验配对与样本对齐审计

#### 必查项

- `compute_pairwise_tests()` 是否按 `query_id` 对齐；
- unsupported-only 指标是否按其 query 子集独立配对；
- 多重比较校正是否输出；
- effect size 是否具备解释字段；
- 是否有 dropped query / overlap size 的记录。

#### 完成标准

- 每个 pairwise result 都能回答：
  - 用了多少 query
  - 对齐后剩多少 query
  - 为什么某些指标是 `ns`

---

### Phase 4：结果用途与论文口径审计

#### Phase 4.1 Supporting evidence 与 decisive evidence 分层审计

#### 必须明确冻结

##### Supporting evidence
- `E5-E10`
- 主要作用：说明机制、消融、边界、迭代教训

##### Decisive evidence
- `G1-G4`
- 主要作用：统一回答 `RQ1/RQ2/RQ3`

#### 完成标准

- 形成一张“论文使用边界表”：
  - 哪些结果只能进第五章 / 第六章
  - 哪些结果可进入第七章
  - 哪些表必须标注 `n=40`
  - 哪些表必须标注当前正式 decisive sample size（当前为 `n=68`）

---

#### Phase 4.2 结论强度审计

对每个核心结论标注其证据强度：

- 强支持
- 中等支持
- 方向支持
- 不能强说

#### 推荐至少覆盖以下结论

- RAG improves evidence verifiability
- RAG improves traceability / evidence credibility
- PEFT improves honesty / alignment
- PEFT cannot replace RAG
- G4 is overall best
- G4 is best in human preference

#### 完成标准

- 形成“论文主张强度表”，避免答辩或写作中 overclaim。

---

### Phase 5：最终重跑前冻结清单

这是今天正式开跑前必须完成的最后一步。

#### 必须冻结的资产

- `judged_queries.jsonl`
- `g_eval_query_ids_68.json`
- `e9_generation_eval_query_ids.json`
- `frozen_split_manifest.json`
- `E6 qrels`
- `G qrels`
- `sft_train/dev manifest v1-v4`
- `exp02 metadata`

#### 必须冻结的配置

- retrieval main config
- candidate_policy
- reranker on/off
- fallback on/off
- behavior model
- PEFT adapter
- generation prompt / parse settings
- judge model / API mode
- blind review seed / sample size

#### 必须冻结的输出契约

- 每个 run 应写哪些文件
- 每个 compare 应写哪些文件
- 每个 closure bundle 应写哪些文件

#### 完成标准

- 形成正式 rerun freeze snapshot；
- 之后所有正式结果都可追溯到同一轮冻结配置。

---

## 4. 建议执行顺序（今日版本）

### Step 1：先完成审计主表
优先级：最高  
原因：先把实验对象、用途、边界全部登记清楚。

### Step 2：优先审查 G1-G4 与 E9/E10 的 generation / stats / closure 契约
优先级：最高  
原因：这部分是决定性证据，也是最近口径修复最多的区域。

### Step 3：审查 retrieval supporting evidence 与 G retrieval final evidence 的变量控制
优先级：高  
原因：防止 retrieval 层不同阶段结果混用。

### Step 4：审查 E10 v1-v4 的训练-评测隔离与变量一致性
优先级：高  
原因：PEFT 章节最容易被问“究竟变了什么”。

### Step 5：审查 E1-E4 的正式重跑输入是否齐备
优先级：中  
原因：通常较稳定，但仍需核实与最新 freeze 一致。

### Step 6：所有审计项通过后，再正式重跑全部实验
优先级：最后  
原因：避免“先跑一整天，最后才发现口径或变量不合格”。

---

## 5. 当前已知的高风险点（需优先盯住）

### 风险 1：supporting evidence 与 decisive evidence 写混

当前项目最大的论文风险不是脚本报错，而是：

- 把 `E5-E10` 的 `n=40` supporting evidence 当成第七章主结论；
- 把 `G1-G4` 的统一结果反向覆盖第五/六章原有消融逻辑。

### 风险 2：同名指标跨层口径漂移

尤其是：

- `unsupported_honesty_rate`
- `hallucination_rate`
- `evidence_verifiability_mean`

必须防止它们在 `summary / compare / stats / report` 中语义不同却名称相同。

### 风险 3：PEFT 章节的变量归因不纯

如果 `v1-v4` 的差异被 retrieval、eval units、inference backend 或 compare 协议污染，那么第六章的“训练数据决定效果”的主张会变弱。

### 风险 4：G retrieval 新 formal eval 链路虽然已修正，但仍需完整冻结后再跑

包括：

- `G qrels` 是否完整覆盖当前正式 decisive query 集（当前为 `68 query`）
- formal plain/aspect retrieval eval 是否都产出正确 summary
- `gchapter` 是否全部依赖 formal retrieval run 而不是 proxy path

---

## 6. 建议产物

本轮审计建议最终至少形成以下 3 份文档：

1. `11_final_experiment_audit_protocol.md`（本文件）  
2. `12_experiment_audit_master_table.md`  
3. `13_experiment_control_and_metric_contracts.md`  
4. （可选）`14_final_rerun_execution_protocol.md`

---

## 7. 当前建议

最稳妥的策略不是立即无差别重跑全部实验，而是：

1. 先按本协议完成审计主表与变量/口径冻结；
2. 再对 `G1-G4` 与 `E10` 这两个高风险区域做优先审查；
3. 最后再开始今天的正式全量重跑。

只有这样，最终产出的结果才能真正满足“**论文可正式写入**”的标准，而不是仅仅“脚本顺利跑完”。
