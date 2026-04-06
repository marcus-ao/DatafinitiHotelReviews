# 实验控制变量与指标口径冻结清单

更新时间：2026-04-05

本文件是 `11_final_experiment_audit_protocol.md` 与 `12_experiment_audit_master_table.md` 的配套文档，专门用于冻结以下两类内容：

1. **控制变量契约**：每类实验允许改变什么、不允许改变什么；
2. **指标口径契约**：每个核心指标在 summary / compare / stats / report 中的正式定义与适用边界。

其目的不是重复代码实现，而是把“实验的严格可比性”写成一份正式的、在论文与重跑流程中都必须服从的契约文档。

---

## 1. 控制变量冻结原则

### 1.1 通用原则

对任何正式 compare，都必须满足：

1. **只允许一个主因素变化**；
2. 若存在两个及以上因素共同变化，则必须在文档中明确写为“bundled operational contrast”，不能写成单因素因果结论；
3. 所有固定变量必须在 run config / manifest / report 中可追溯；
4. Supporting evidence 与 decisive evidence 的控制变量体系必须分开冻结，不能混用。

---

## 2. Retrieval 系列控制变量契约

### 2.1 E5：中文直检 vs 英文桥接

| 维度 | 必须固定 | 唯一允许变化 |
|---|---|---|
| query set | `40 core / 80 units` | 中文原句 vs structured query_en |
| qrels | E6 qrels | 无 |
| retrieval backend | dense no-rerank | query 表达方式 |
| candidate hotel scope | 同城 test 酒店 | 无 |
| top-k | 固定 `@5` | 无 |

#### 写作约束

- `E5` 只能得出“桥接必要性”结论；
- 不能写成“Aspect-guided retrieval 主结论”。

---

### 2.2 E6：Aspect vs Plain retrieval

| 维度 | 必须固定 | 唯一允许变化 |
|---|---|---|
| query set | `40 core / 80 units` | retrieval strategy |
| qrels | E6 qrels | 无 |
| hotel split | frozen test hotels | 无 |
| candidate scope | same city test hotels | 无 |
| reranker | 保持为 E6 设计的官方对比配置 | retrieval mode |

#### 写作约束

- `E6` 是 retrieval 正结果的主要正式来源之一；
- 第七章若需讲 retrieval 正式差异，应优先回引 `E6`，而不是依赖 G retrieval proxy 表。

---

### 2.3 E7：Reranker ablation

| 维度 | 必须固定 | 唯一允许变化 |
|---|---|---|
| query set | `40 core / 80 units` | reranker on/off |
| qrels | E6 qrels | 无 |
| candidate scope | frozen | 无 |
| retrieval main path | aspect-guided | reranker only |

#### 写作约束

- 不允许把 `E7` 写成“RAG 主配置变化”；
- 它只能服务于“当前 reranker 未验证出稳定收益”的边界结论。

---

### 2.4 E8：Fallback ablation

| 维度 | 必须固定 | 唯一允许变化 |
|---|---|---|
| query set | `40 core / 80 units` | fallback on/off |
| qrels | E6 qrels | 无 |
| aspect main path | 保持一致 | fallback only |

#### 写作约束

- 只能写“当前 fallback 未验证出额外收益”；
- 不得将其解释为 RAG 主路线变化。

---

### 2.5 G retrieval 正式输入

| 对比用途 | formal retrieval 输入 |
|---|---|
| `G1` | plain retrieval formal run |
| `G2` | aspect retrieval formal run |
| `G3` | plain retrieval formal run（与 G1 共用） |
| `G4` | aspect retrieval formal run（与 G2 共用） |

#### 冻结要求

1. `G1/G3` 共用同一 plain retrieval run 是**控制变量设计**，不是遗漏；  
2. `G2/G4` 共用同一 aspect retrieval run 也是控制变量设计；  
3. 若 chapter 报告中 retrieval 表为 formal run summary，则必须显式说明其来源是“shared upstream retrieval input”，而不是 per-group independently rerun retrieval effect。

---

## 3. Generation 系列控制变量契约

### 3.1 E9：有无证据约束 / 有无 RAG 对比

| 维度 | 必须固定 | 唯一允许变化 |
|---|---|---|
| query set | `40` | evidence availability / prompt evidence usage |
| eval units | E9 frozen eval units | 无 |
| retrieval backend | `aspect_main_no_rerank` | 不作为比较因素 |
| candidate policy | `E2 B_final_aspect_score Top5` | 无 |
| model | `Qwen3.5-4B` | group behavior mode |
| output schema | RecommendationResponse 固定 | 无 |

#### 写作约束

- `E9` 只用于 supporting evidence；
- 若 compare 是 `B vs D`，主结论是 coverage / schema / evidence-availability influence；
- 不得把 `E9` 直接写成第七章统一矩阵证据。

---

### 3.2 E10：Base vs PEFT 迭代链

| 维度 | 必须固定 | 唯一允许变化 |
|---|---|---|
| query set | `40` | adapter / training data design |
| eval units | E9-derived frozen generation units | 无 |
| retrieval | 固定，不作为因素 | 无 |
| base model | `Qwen3.5-4B` | adapter identity |
| compare protocol | base vs one adapter at a time | manifest / adapter only |

#### 额外硬约束

1. `v1-v4` 的主解释必须是“训练数据设计差异”；  
2. 不允许同时改变 retrieval、eval units、backend、compare 协议；  
3. `exp02` 是 G3/G4 的 canonical PEFT adapter，其 lineage 必须在论文与运行环境中一致。

---

### 3.3 G1-G4：统一 2×2 生成矩阵

| Compare | 必须固定 | 唯一允许变化 |
|---|---|---|
| `G1 vs G2` | query scope、split、backend、prompt、PEFT=off | retrieval variant |
| `G1 vs G3` | query scope、split、backend、prompt、retrieval=plain | PEFT on/off |
| `G2 vs G4` | query scope、split、backend、prompt、retrieval=aspect | PEFT on/off |
| `G3 vs G4` | query scope、split、backend、prompt、PEFT=exp02 | retrieval variant |

#### 额外硬约束

1. `G1-G4` 必须全部使用同一 `68` query scope；  
2. Judge、blind review、pairwise stats 必须都基于这同一轮 group run；  
3. `gchapter` 不得回退到 proxy retrieval summary；  
4. `exp02 metadata` 必须可追溯到实际 served adapter。

---

## 4. 核心指标口径冻结表

## 4.1 检索层指标

| 指标 | 正式定义 | 聚合维度 | higher_is_better | 适用实验 | 备注 |
|---|---|---|---|---|---|
| `Aspect Recall@5` | Top-5 是否覆盖目标方面 | per-target-unit -> mean | 是 | E5-E8、formal G retrieval | retrieval judged metrics |
| `nDCG@5` | qrels graded relevance 排序质量 | per-target-unit -> mean | 是 | E5-E8、formal G retrieval | retrieval 主指标 |
| `Precision@5` | Top-5 relevant ratio | per-target-unit -> mean | 是 | E5-E8、formal G retrieval | retrieval 主指标 |
| `MRR@5` | 首个 relevant 命中位置 | per-target-unit -> mean | 是 | E5-E8、formal G retrieval | retrieval 辅助排序指标 |
| `Evidence Diversity@5` | Top-5 中酒店/方面多样性 | per-target-unit -> mean | 是 | E5-E8、formal G retrieval | 支持“证据覆盖结构”分析 |
| `Retrieval Latency` | retrieval 端到端耗时 | run-level mean | 否 | E5-E8、formal G retrieval | 必须说明 lower is better |

### 口径约束

1. retrieval 指标只应在有 qrels 的正式 retrieval eval 上解释；  
2. 若 G retrieval 表仍为 proxy，则不得用这些指标名称承载 formal retrieval claim；  
3. robustness 30 条若未正式纳入 qrels，则 retrieval 指标的 denominator 不得口头写成 `70`。

---

## 4.2 生成层指标

| 指标 | 正式定义 | 聚合维度 | higher_is_better | 适用实验 | 风险提示 |
|---|---|---|---|---|---|
| `Citation Precision` | 合法 citation 比例 | query-level -> run mean | 是 | E9、E10、G1-G4 | 对 no-evidence 组只能作辅助解释 |
| `Evidence Verifiability Mean` | citation 是否真正支撑 reason | query-level -> run mean | 是 | E9、E10、G1-G4 | groundedness 核心指标 |
| `Schema Valid Rate` | 输出是否通过 schema 校验 | query-level -> run mean | 是 | E9、E10、G1-G4 | 容易出现 ceiling |
| `Recommendation Coverage` | 是否产出非空推荐 | query-level -> run mean | 是 | E9、E10、G1-G4 | coverage 不是 groundedness 本身 |
| `Aspect Alignment Rate` | reason 是否回应 focus_aspects | query-level -> run mean | 是 | E9、E10、G1-G4 | 需依赖 eval_unit gold |
| `Hallucination Rate` | 无证据支撑声明比例 | query-level -> run mean | 否 | E9、E10、G1-G4 | lower is better；必须统一 support_score 逻辑 |
| `Unsupported Honesty Rate` | unsupported query 上是否诚实告知 | unsupported-query subset mean | 是 | E9、E10、G1-G4 | **只在 unsupported 子集解释** |

### 口径约束

1. `Unsupported Honesty Rate` 在 `summary / pairwise_tests` 中必须显式说明是 unsupported 子集；  
2. `Hallucination Rate` 若使用 `citation_exists == 0 or support_score == 0` 逻辑，则所有 compare / score map / report 都必须一致；  
3. 对 `D_no_evidence_generation` 这类天然不使用证据的组，`Citation Precision / Evidence Verifiability` 只能做辅助解释。

---

## 4.3 Judge / Human 指标

| 指标 | 聚合方式 | 适用范围 | 备注 |
|---|---|---|---|
| `Relevance` | query-level judge mean | G1-G4 | 只服务统一矩阵 |
| `Traceability` | query-level judge mean | G1-G4 | 应与 Evidence Verifiability 配合解释 |
| `Fluency` | query-level judge mean | G1-G4 | 不得单独主导模型优劣结论 |
| `Completeness` | query-level judge mean | G1-G4 | 需与 Aspect Alignment 一起解释 |
| `Honesty` | query-level judge mean | G1-G4 | 与 unsupported honesty 方向相关但不等价 |
| `Overall Quality` | item-level blind review mean | G1-G4 | 人工评审总体验 |
| `Evidence Credibility` | item-level blind review mean | G1-G4 | 与 RAG 主效应强相关 |
| `Practical Value` | item-level blind review mean | G1-G4 | 人类偏好层的重要结果 |

### 口径约束

1. Judge 与 human blind review 只能作为 supplementary decisive evidence；  
2. 若 Judge 支持 `G4 > G2`，但 human 未显著支持，论文结论必须写成“自动维度最优但人类偏好未显著拉开”。

---

## 5. 统计解释冻结规则

### 5.1 允许的结论强度

| 强度 | 允许条件 |
|---|---|
| 强支持 | paired stats 显著 + effect size 有意义 + 多源证据方向一致 |
| 中等支持 | paired stats 不稳定或 effect size 中等，但 summary/Judge/human 方向一致 |
| 方向支持 | ceiling effect / query 子集过小 / note 指向 `insufficient_non_zero_pairs` |
| 不能强说 | 指标为 proxy、样本量不匹配、或 source contract 不完整 |

### 5.2 明确禁止的 overclaim

1. 用 `G retrieval proxy` 写 retrieval 主结论；  
2. 把 `unsupported_honesty_rate` 的方向性改善写成强显著 PEFT 优势；  
3. 把 `G4` 写成“在人类偏好上显著全面优于 G2”，若 blind review 未支持；  
4. 把 `E5-E10` supporting evidence 写成第七章统一对比的决定性依据。

---

## 6. 立刻需要执行的冻结动作

### 6.1 结果层冻结

- supporting evidence 结果表必须全部带上 `n=40` 或对应 sample size；
- G1-G4 unified matrix 结果表必须明确带上当前正式 decisive sample size（当前为 `n=68 = 39 core + 29 robustness`，`q021 / q024` 作为 supporting boundary cases 单列说明）；
- retrieval formal / proxy 状态必须在 report 中单独标注。

### 6.2 运行层冻结

- fixed split manifest
- fixed qrels paths
- fixed generation eval units paths
- fixed adapter metadata path
- fixed judge model / API mode
- fixed blind review seed

### 6.3 解释层冻结

- RQ1：RAG improves evidence quality 允许强说；
- RQ2：PEFT improves behavior quality 只能中等/方向支持；
- RQ3：mechanistic complementarity but weak additive effect 是默认结论表述。

---

## 7. 最终用途

本文件的直接用途是：

1. 作为今天正式全量重跑前的**控制变量与指标口径冻结依据**；  
2. 作为后续 `14_final_rerun_execution_protocol.md` 的约束输入；  
3. 作为论文写作阶段的“**不得越界解释**”规则表。
