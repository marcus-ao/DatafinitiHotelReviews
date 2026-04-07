# 第五章正文材料：基于 RAG 的 LLM 推荐增强

更新时间：2026-04-07

本文件用于将当前 canonical supporting evidence 与统一矩阵中属于 RAG 路线的结果，整理为更接近论文正文风格的写作材料。第五章的核心任务不是证明“所有 retrieval 配置都有效”，而是围绕同一个更重要的问题展开：Aspect-guided RAG 是否通过改善证据输入质量，提升了 grounded recommendation 的可信度、可追溯性与可验证性。因此，本章需要特别注意区分桥接、主正结果与负结果的角色，避免将 retrieval 管线中的所有组件都包装成有效增强。

## 1. 桥接机制：E5 的角色应如何写

第五章的第一步不是直接论证 Aspect-guided retrieval 的优越性，而是先说明为什么跨语言桥接在当前场景下不可或缺。由于知识库存储与 evidence 组织主要建立在英文表达之上，中文原始查询如果直接进入 retrieval，会受到明显的跨语言匹配损失。`E5` 正是在这个意义上构成了第五章的前提性证据。最稳妥的正文写法应是：结构化英文桥接并不是 retrieval 主增益本身，而是让后续 retrieval 对比具备可解释性的必要前提。换言之，没有桥接，后续 `E6-E8` 与 `G2/G4` 的结果就会被跨语言失配噪声污染；有了桥接，后续 retrieval 差异才可以被更清晰地解释为检索策略本身的影响。

## 2. Retrieval 主正结果：E6 的解释重点

`E6` 是第五章 retrieval 层最核心的一组 supporting evidence。它的价值并不只在于某几个指标更高，而在于它解释了为什么这些指标会更高。当前更稳妥的解释应该是：Aspect-guided retrieval 在 retrieval 早期引入了明确的方面约束，从而显著减少了“虽然语义相关、但不足以支撑当前推荐理由”的 evidence-topic drift。也就是说，它改变的不仅是命中数量，而是**证据的分布质量**。

因此，第五章对 `E6` 的正式写法不应停留在“Aspect-guided retrieval 的 nDCG@5 和 Precision@5 更高”，而应进一步指出：这些提升意味着模型在 generation 阶段将接收到更贴近用户关注方面、更能形成 grounded recommendation 的 supporting evidence。换句话说，`E6` 的真正意义在于为后续 generation 端的 groundedness 提供解释基础，而不是单独构成一个与 generation 无关的 retrieval 排名结果。

## 3. 负结果必须保留：E7 与 E8

第五章中最值得保持学术诚实性的地方，是对 `E7` 和 `E8` 的处理。当前 canonical 结果表明，reranker 并没有带来核心 retrieval 指标的净提升，反而显著增加了延迟开销：`aspect_main_no_rerank` 的 `nDCG@5 = 0.6457`，而 `aspect_main_rerank` 下降为 `0.6378`，平均延迟则从 `199.788 ms` 大幅上升到 `750.527 ms`。因此，`E7` 的正式结论应写成：在当前协议下，reranker 是一个负结果或至少是一个无净收益结果，它不应进入当前 retrieval 主线路径。

`E8` 同样应以相同的诚实方式处理。当前 fallback 版本虽然确实被触发，但主 retrieval 指标几乎没有改善，同时 `fallback_noise_rate = 1.0`。这意味着 fallback 在当前 aspect mainline 下并没有形成可接受的正式增益，相反，它更像是一个噪声风险较高的边界尝试。因此，第五章必须明确说明：当前 canonical 结果不支持把 fallback 写成有效增强组件。

## 4. Generation 端主证据：G2 vs G1

第五章真正最强的一条证据链，来自统一矩阵中的 `G2 vs G1`。如果说 `E6` 说明了 retrieval 层为什么会变好，那么 `G2 vs G1` 则进一步证明：这种 retrieval 层的改善已经真实传导到了 generation 端，并最终表现为 grounded recommendation 质量的提高。

当前统一矩阵显示，`G1` 的 `Evidence Verifiability Mean = 1.5404`，而 `G2` 提升到了 `1.9522`；与此同时，Judge 侧的 `Traceability` 也从 `3.5809` 提升到 `4.4029`。因此，第五章最稳妥的正文表达应是：RAG 路线的核心收益不在于让模型“说得更多”，而在于让模型获得了更可验证、更能支撑推荐理由的证据输入。也正因为如此，第五章最终回答 `RQ1` 的关键证据，并不只是 retrieval supporting evidence 本身，而是 retrieval 改善与 generation groundedness 提升之间的连贯传导关系。

## 5. 第五章统一收束段

综合 `E5-E9` 与统一矩阵中的 `G2 vs G1` 可知，第五章最稳妥的结论不是“RAG 让所有指标都变好”，而是：RAG 路线的主要收益集中体现在 grounded evidence quality 的提升上。桥接机制保证了跨语言 retrieval 的可解释性，Aspect-guided retrieval 提供了 retrieval 层的主正结果，而 reranker 与 fallback 在当前协议下未形成正式正收益，应如实作为负结果保留。最终，`G2` 相比 `G1` 在 generation 端 groundedness 的显著改善，构成了回答 `RQ1` 的关键证据链。

## 6. 本章写作禁忌

第五章正式转写时，应避免以下几类写法：

1. 不要把 `E5` 写成 retrieval 主正结果；  
2. 不要把 `E7/E8` 包装成“局部有效但总体也值得保留”的正收益；  
3. 不要把 retrieval 层优势解释成“只是找到了更多句子”，而应强调“证据更能支撑推荐理由”；  
4. 不要把第五章收束成“RAG 让所有指标都变好”的简单乐观叙事。
