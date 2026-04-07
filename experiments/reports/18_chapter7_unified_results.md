# 第七章正文材料：RAG 与 PEFT 的统一对比分析

更新时间：2026-04-07

本文件用于将当前 `G1-G4` 的 canonical 统一矩阵结果，转写为更接近论文正文风格的分析材料。第七章的目标，不是把所有评价来源压成一个简单的“总分排序”，而是在统一协议下，以 hard metrics、pairwise statistical tests、LLM Judge 与 sampled human-verified blind review 共同回答 `RQ1 / RQ2 / RQ3`。

## 1. 第七章的章节角色

第七章是全文的统一高潮，因此它最重要的不是重复 supporting evidence，而是把所有关键证据放回同一矩阵中，完成最终对比。它的写作原则应当固定为：先讲统一矩阵，再讲 supporting evidence；先讲 hard metrics 与 pairwise tests，再讲 Judge 与 blind review；最后必须显式承认不同评价层之间的细粒度分歧。只有这样，第七章才能既给出明确结论，又保持学术诚实性。

## 2. 当前统一矩阵最稳固的事实

从当前 canonical `G chapter` 结果看，最稳固的事实主要有四类。第一，retrieval 层的差异非常清晰：`G1/G3` 代表的 plain retrieval 与 `G2/G4` 代表的 aspect retrieval 在正式 qrels-based evaluation 上存在稳定差距，`aspect_recall_at_5`、`nDCG@5`、`precision@5` 与 `mrr_at_5` 都显示 aspect 路线更强。第二，generation 层的 groundedness 差异同样明确：`G2` 与 `G4` 在 `Evidence Verifiability Mean` 上显著高于 `G1`，说明 retrieval 层提升已经传导到 recommendation 端。第三，Judge 层更偏向 `G4`，其 `overall_mean = 4.2197` 为当前最高。第四，blind review 层对 `G2` 与 `G4` 的细粒度排序并不与 Judge 完全一致：blind review item mean 中 `G2` 略高于 `G4`，且 blind pairwise 里 `G2 > G4 = 9`，`G4 > G2 = 2`。也就是说，第七章从一开始就必须接受一个事实：当前 unified matrix 并不存在一个“所有评价来源都一致认定的无争议最优组”。

## 3. RQ1：RAG 是否提升推荐质量？

对于 `RQ1`，当前 unified matrix 提供的是一组相对强而稳定的支持证据。最稳妥的正文写法应是：Aspect-guided RAG 路线在 groundedness 关键维度上带来了显著收益。具体而言，`G2` 相比 `G1` 在 `Evidence Verifiability Mean` 上明显提升，并在 Judge 的 `Traceability` 维度上获得更高评价。这表明 RAG 的主要价值不是让模型“生成更多内容”，而是让模型获得更可靠、更能支撑推荐理由的证据输入。换言之，第五章中的 retrieval supporting evidence 解释了为什么 retrieval 层会变好，而第七章中的 unified matrix 则进一步证明：这种 retrieval 改善已经真实传导到 generation 端，并转化成更 grounded 的推荐结果。

## 4. RQ2：PEFT 是否提升行为稳定性？

对于 `RQ2`，正文必须保持克制。当前 unified matrix 并不支持“PEFT 稳定提升整体推荐质量”的强结论。更准确的表达是：PEFT 在一些局部指标上可能改善对齐性、表述一致性或行为层稳定性，但这些局部收益不足以在 groundedness、schema 与 hallucination 等核心维度上形成稳定净收益。因此，第七章中的 `RQ2` 最稳妥写法应该与第六章保持一致：PEFT 的作用更像行为层调节，而不是已经被统一矩阵正式证明胜出的增强路线。

## 5. RQ3：RAG 与 PEFT 是否互补？

`RQ3` 的最稳妥答案应写成：**机制互补，但结果弱叠加**。Judge 结果更偏向 `G4`，blind review 结果更偏向 `G2`，说明两条证据链都支持 aspect retrieval 路线更强，但对 `G2` 与 `G4` 的细粒度排序并不一致。因此，最安全的结论不是“G4 无争议最好”，而是：RAG 决定 evidence quality，PEFT 提供有限的行为层修饰，二者在当前协议下并未形成无争议的强协同增益。也正因为如此，第七章最有价值的地方并不是宣布一个唯一赢家，而是诚实展示不同增强路线在不同评价维度上的贡献与边界。

## 6. 第七章必须显式保留的分歧说明

第七章中必须保留一段单独的分歧说明。最稳妥的写法是：LLM Judge 与 sampled human-verified blind review 对 `G2` 与 `G4` 的细粒度排序存在差异。Judge 更偏好 `G4`，而 blind review 对 `G2` 显示出更明显的局部偏好。因此，本研究不将任何单一评价来源视为唯一终局排序依据，而是将 hard metrics、pairwise statistical tests、LLM Judge 与 blind review 共同作为补充性证据来解释最终结论。这段文字的意义不只是“补充说明”，而是第七章最重要的学术诚实性保险丝。

## 7. 第七章统一收束段

综合当前 unified matrix 的结果可知，RAG 是 grounded recommendation 质量提升的主要来源，其收益集中体现在 evidence quality 与 traceability 上；PEFT 的作用更多体现在局部行为层修饰，而非形成稳定的整体净收益。Judge 与 human-verified blind review 对 `G2` 与 `G4` 的细粒度排序存在分歧，这进一步说明当前最稳妥的结论不是简单宣称“联合模型最优”，而是承认：RAG 与 PEFT 在机制上互补，但在当前协议与数据条件下只形成了弱叠加，而没有产生无争议的协同最优。

## 8. 本章写作禁忌

第七章正式转写时应避免以下写法：

1. 不要写 `G4` 全面最好；  
2. 不要把 Judge 当成唯一最终排序；  
3. 不要把 blind review 写成大规模独立人工外审；  
4. 不要把 `G2` 与 `G4` 的分歧硬压成一个统一总分；  
5. 不要把第七章写成“PEFT 证明自己胜出”的章节。
