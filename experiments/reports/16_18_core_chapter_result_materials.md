# 第五章—第七章核心结果正文材料（RAG / PEFT / 统一矩阵）

更新时间：2026-04-07

本文件用于把当前 canonical 结果转化为可直接进入论文正文的结果分析材料，覆盖：

- 第五章：基于 RAG 的推荐增强
- 第六章：基于 PEFT 的推荐增强
- 第七章：RAG 与 PEFT 的统一对比

其目标不是替代详细的 `analysis.md` 或 `summary.csv`，而是为正文提供一套已经经过结果治理和风险控制的稳定表述。

---

## 1. 写作总原则

第五章到第七章的正文应遵循以下原则：

1. **E5-E10 是 supporting evidence**，用于解释机制、消融、负结果与边界；  
2. **G1-G4 是 decisive evidence**，用于回答 `RQ1 / RQ2 / RQ3`；  
3. 不要把 Judge、blind review、hard metrics 混成“单一总分”；  
4. 不要为了论证“最优组”而掩盖负结果或分歧；  
5. 所有强结论都必须建立在：
   - hard metrics
   - pairwise stats
   - supplementary evidence
   三者至少部分一致的基础上。

---

## 2. 第五章正文材料：RAG 路线

### 2.1 本章核心论点

第五章最稳妥的核心论点应写成：

> Aspect-guided retrieval 的主要收益不是简单增加 citation 数量，而是提高推荐理由与证据之间的真实支撑关系。因此，RAG 的贡献主要体现在 `Evidence Verifiability`、`Traceability` 以及人工感知的 `Evidence Credibility` 上，而不是单纯的输出格式或推荐覆盖率。

### 2.2 E5-E8 的正文级写法建议

#### E5：桥接必要性

> `E5` 说明，当知识库以英文表达组织时，直接以中文查询进行检索会导致明显的跨语言匹配损失；结构化英文桥接能够显著改善 retrieval quality。因此，桥接机制不是额外修饰，而是当前 RAG 管线成立的必要前提之一。

#### E6：Aspect-guided retrieval 的主正结果

> `E6` 是 retrieval supporting evidence 中最重要的一组正结果。它表明 Aspect-guided retrieval 相比 plain dense retrieval，在 `nDCG@5`、`Precision@5` 与 evidence support quality 上具有稳定优势。这说明方面约束并不是简单做过滤，而是在 retrieval 早期就显著改变了证据分布，使模型后续生成更 grounded。

#### E7：Reranker 负结果

> `E7` 显示，在当前 retrieval 配置下，增加 reranker 并没有带来核心 retrieval 指标的净提升，反而显著增加延迟成本。因此，本研究将 reranker 结果作为负结果保留，并据此冻结不带 reranker 的官方主线路径。

#### E8：Fallback 负结果

> `E8` 进一步表明，fallback 机制在当前 aspect mainline 下虽会被触发，但无法稳定改善 retrieval 主指标，且伴随较高噪声风险。因此，fallback 不应被描述为当前正式主线路径的有效增强。

### 2.3 G2 vs G1 的正文级结论

> 在统一矩阵中，`G2` 相比 `G1` 显著提升了 grounded recommendation 的 evidence support quality。特别是在 `Evidence Verifiability Mean` 上，`G2` 明显优于 `G1`，并在 Judge 的 `Traceability` 维度上获得更高分。这说明 RAG 的主要作用是改善输入证据质量，而不是仅仅改变输出风格。

### 2.4 第五章统一收束

> 因此，第五章最稳妥的结论不是“RAG 让所有指标都变好”，而是：RAG 主要提高证据质量与可追溯性，其收益集中体现在 groundedness 维度；而 reranker 与 fallback 在当前协议下未形成正式正收益，应如实作为负结果保留。

---

## 3. 第六章正文材料：PEFT 路线

### 3.1 本章核心论点

第六章的核心论点不应写成“PEFT 显著优于 base”，而应写成：

> 在当前数据规模、模型规模与 grounded recommendation 任务设定下，PEFT 的收益更多体现在行为层局部改进，而不是形成稳定的整体净收益。E10 的多轮迭代因此更适合被解释为“训练目标与数据设计的边界探索”，而不是一条稳定胜出的优化路线。

### 3.2 E10 的正文级写法建议

#### E10_base vs exp02

> 当前 canonical E10 compare 显示，`exp02` 并未在 schema validity、evidence verifiability 与 hallucination behavior 上形成稳定净收益。虽然它在局部行为对齐指标上存在提升趋势，但这些收益不足以支持“PEFT 已稳定优于 base”的强结论。

#### v1-v4 的章节逻辑

> `v1-v4` 更应被写成一条完整的“训练目标对齐消融链”：
- `v1` 暴露了缺少 grounded data 时的训练目标错位；
- `v2` 提供了最可接受的平衡点，因此被选作 G3/G4 的 canonical PEFT 路线；
- `v3/v4` 则展示了在小数据条件下，试图继续修补局部问题时会出现“按下葫芦浮起瓢”的现象。

### 3.3 unsupported honesty 的写法

> 第六章中 `unsupported_honesty_rate` 不应被当作 E10 的主证据指标，因为当前 E10 query slice 中该指标并不适用。正式结果文件已经通过 `unsupported_honesty_applicable` 显式标注了这一点。因此，关于 unsupported honesty 的主要讨论应放在 E9 与 G 统一矩阵中，而不是强行在 E10 里展开。

### 3.4 第六章统一收束

> 第六章最稳妥的结论是：PEFT 在当前设定下更适合被理解为一种行为层调节机制，而不是能够稳定提升 grounded recommendation 核心质量的普适增强路线。它提供了有价值的负结果和边界经验，这本身也是本研究的重要发现。

---

## 4. 第七章正文材料：统一矩阵与研究问题回答

### 4.1 RQ1：RAG 是否提升推荐质量？

建议写法：

> `RQ1` 得到强支持。统一矩阵结果表明，RAG 路线在 `Evidence Verifiability Mean` 等 groundedness 关键指标上带来显著改善，并在 LLM Judge 的 `Traceability` 维度上获得更高评价。这说明 Aspect-guided retrieval 的主要作用在于提高证据质量，而不是仅仅改变表层输出形式。

### 4.2 RQ2：PEFT 是否提升行为稳定性？

建议写法：

> `RQ2` 仅得到有限支持。当前结果不支持“PEFT 稳定提高整体推荐质量”的强结论，但支持将其理解为一种行为层调节机制：它可能在局部对齐、诚实性或结构一致性上产生影响，但这些影响尚不足以形成稳定净收益。

### 4.3 RQ3：RAG 与 PEFT 是否互补？

建议写法：

> `RQ3` 的答案应写成“机制互补，但结果弱叠加”。Judge 更偏好 `G4`，而 blind review 更偏好 `G2`，说明两条证据链都支持 aspect retrieval 的优势，但对 `G2` 与 `G4` 的细粒度排序并不一致。因此，最稳妥的结论不是“G4 全面最好”，而是：RAG 决定 grounded evidence quality，PEFT 提供有限的行为层修饰，二者在当前设定下未形成压倒性的协同净收益。

### 4.4 第七章必须保留的显式说明

正文中建议明确加入如下提醒：

> 需要指出的是，LLM Judge 与 sampled human-verified blind review 对 `G2` 和 `G4` 的细粒度排序存在分歧。因此，本研究不将任何单一评价来源视为唯一的终局排序依据，而是将 hard metrics、pairwise statistical tests、LLM Judge 与 blind review 共同作为补充性证据来解释最终结论。

### 4.5 第七章统一收束段

> 综合统一矩阵的结果，本研究认为：RAG 是 grounded recommendation 质量提升的主要来源，其收益集中体现在 evidence quality 与 traceability 上；PEFT 的作用更多体现在局部行为层修饰，而非形成稳定的整体净收益。二者在机制上互补，但在当前数据与协议下并未形成无争议的强协同增益。这一发现比简单宣称“联合模型最优”更具方法论价值。

---

## 5. 直接写作时的禁忌清单

### 禁忌 1
不要写：
> G4 全面最好

应写：
> G4 在 Judge 侧表现最好，但 blind review 对 G2 与 G4 的细粒度排序存在分歧。

### 禁忌 2
不要写：
> PEFT 已稳定提升推荐质量

应写：
> PEFT 在当前设定下未形成稳定净收益，但提供了有价值的行为层与边界分析线索。

### 禁忌 3
不要写：
> E7 / E8 证明 reranker / fallback 有效

应写：
> E7 / E8 在当前协议下未形成正式正收益，应如实保留为负结果或 mixed finding。

### 禁忌 4
不要写：
> blind review 是完全独立的大规模人工评审

应写：
> blind review 是 sampled human-verified blind-review evidence。

---

## 6. 本文件用途

本文件建议作为：

- 第五章—第七章正文写作底稿；
- 答辩时针对负结果与分歧结果的解释提纲；
- 与 `gchapter/.../analysis.md` 配套的章节正文级转写材料。
