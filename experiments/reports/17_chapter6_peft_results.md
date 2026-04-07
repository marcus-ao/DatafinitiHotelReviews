# 第六章正文材料：基于 PEFT 的 LLM 推荐增强

更新时间：2026-04-07

本文件用于将当前 `E10` canonical 结果与其 supporting evidence 角色，整理为更接近论文正文风格的写作材料。第六章不应被写成一个“PEFT 已经稳定胜出”的章节，而应被写成：在 grounded recommendation 任务中，PEFT 在当前数据规模与协议设定下到底能提供什么、不能提供什么，以及它究竟暴露了哪些训练目标与数据构造的边界。

## 1. 第六章的章节角色

第六章的真正问题不是“PEFT 是否一定优于 base”，而是：在 grounded recommendation 这一具体任务中，PEFT 是否能改善模型的行为稳定性、对齐性与约束遵守能力；如果没有形成稳定净收益，那么这种失败本身揭示了什么方法论问题。从这个角度看，第六章的价值不仅在于是否存在正结果，更在于能否把 `v1-v4` 的实验结果组织成一条训练目标对齐的消融链。

## 2. canonical E10 compare 的最稳妥写法

当前 canonical compare（base vs exp02）显示：`citation_precision` 为 `0.9688 -> 0.9688`，保持不变；`evidence_verifiability_mean` 从 `1.9187` 降到 `1.8813`；`schema_valid_rate` 从 `1.00` 降到 `0.95`；`hallucination_rate` 从 `0.0063` 升到 `0.0250`；只有 `aspect_alignment_rate` 从 `0.8792` 提升到 `0.9167`。因此，第六章最稳妥的正文写法应是：当前结果并不支持“PEFT 在 grounded recommendation 上形成稳定净收益”的强结论。更准确的说法是，`exp02` 展现出一定的局部行为对齐收益，但这些局部收益不足以抵消其在 schema、evidence quality 与 hallucination behavior 上的退化。

也正因为如此，第六章不能被写成“PEFT 已经胜出”的成功叙事，而应写成：在当前协议下，PEFT 更像是一种行为层调节尝试，其主要研究价值来自它暴露出的 trade-off 与边界，而不是来自一个全面优于 base 的 final outcome。

## 3. `v1-v4`：一条真正有价值的消融链

`v1-v4` 的价值，恰恰不在于它们像工程优化日志那样一路变好，而在于它们构成了一条训练目标对齐消融链。`v1` 清楚暴露了在缺少 grounded recommendation 监督时，PEFT 容易偏向形式上合理、但任务目标并不对齐的行为模式；`v2` 则通过补入 grounded data 取得了当前最可接受的平衡，因此才被选作 G3/G4 的 canonical PEFT 路线；而 `v3 / v4` 进一步表明，在小数据条件下，如果继续尝试修补局部问题，往往会在 schema、evidence 与 hallucination 之间引入新的 trade-off。换句话说，第六章最重要的并不是“找到了最终最优 PEFT 版本”，而是：已经用一条完整的消融链展示了小数据 PEFT 的能力边界。

## 4. unsupported honesty 在第六章中的正确处理方式

当前 canonical E10 compare 中，`unsupported_honesty_rate` 对应的 `unsupported_honesty_applicable = False`。因此，这个指标在第六章中的角色应被明确限定为：**当前 query slice 下不适用**。正文不应把它写成漏算、缺失或隐藏问题，也不应勉强用它来证明 PEFT 在 unsupported honesty 上的优劣。关于 unsupported honesty 的正式论证，应当主要转移到 `E9` 与第七章统一矩阵中去完成，而不是在 E10 中被过度解读。

## 5. 第六章统一收束段

综合 `E10 v1-v4` 的结果，可以将第六章最稳妥地收束为：在当前 grounded recommendation 任务中，PEFT 并未形成稳定的整体净收益，但它提供了一条具有方法论价值的训练目标对齐消融链。通过这条链条，本研究得以更清楚地说明：在小数据与严格 grounded 协议下，PEFT 更适合被理解为行为层调节机制，而不是一个已经被正式验证能稳定优于 base 的增强路线。

## 6. 本章写作禁忌

第六章正式转写时应避免：

1. 把 `exp02` 写成已经正式胜出的 PEFT 版本；  
2. 把 `unsupported_honesty_rate` 的空白写成“漏算”，它当前是明确的 `not applicable`；  
3. 把 `v1-v4` 写成单调上升的优化故事；  
4. 把第六章写成“PEFT 已被证明有效”的正结果章节。
