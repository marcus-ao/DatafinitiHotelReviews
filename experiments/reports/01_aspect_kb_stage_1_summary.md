# Aspect-KB 第一阶段收口总结

更新时间：2026-03-30

本文件用于汇总 Aspect-KB 章节当前已经冻结、可以直接复用到论文写作中的首轮材料。当前阶段只收口 `E1 + E2`，不提前推进 PEFT 或 `G1-G4`。

## 一、当前阶段已经完成什么

- `E1` 资产已完成对齐，正式 gold 只使用 `experiments/labels/e1_aspect_reliability/aspect_sentiment_gold.csv`
- `E1` 正式评估已完成，并已冻结 `e1_metrics.json` 与 `e1_report.md`
- `E2` 首轮正式 run 已完成，并已补全失败案例分析
- Aspect-KB 章节已经具备一版可写的实验依据，但结论仍需保持克制

## 二、E1 正式结果

### 1. 评估口径

- 抽样总数：`360`
- 商家回复排除：`16`
- 正式可用 gold：`344`
- 严格合并后用于评估的样本数：`344`
- 其中单方面主类别评估样本：`243`
- 多方面困难集样本：`101`

### 2. 核心结果表

| Strategy | Aspect macro-F1 | Difficult-set Jaccard |
|---|---:|---:|
| rule_only | 0.5107 | 0.7812 |
| zeroshot_only | 0.0932 | 0.0099 |
| hybrid | 0.4960 | 0.7911 |

补充：

- Sentiment macro-F1: `0.4458`

### 3. 指标解释

`hybrid` 明显优于 `zeroshot_only`，说明单独依赖当前 zero-shot 方案不足以支撑 Aspect-KB 的可靠标注；但 `hybrid` 并没有在单方面主类别的 `Aspect macro-F1` 上超过 `rule_only`，只是在多方面困难集的 `Difficult-set Jaccard` 上略高。这意味着当前混合策略的真实价值主要体现在“补多方面句”，而不是“全面替代规则基线”。

### 4. 最易混淆的方面

当前 `hybrid` 下最突出的混淆对为：

- `service -> room_facilities`：`26`
- `room_facilities -> quiet_sleep`：`11`
- `room_facilities -> location_transport`：`9`

这说明涉及“住宿体验”的句子在服务、房间条件与睡眠体验之间仍有较强边界重叠。

### 5. 可直接写入论文的错误案例

案例 A：

- 句子：`Staff was great and accommodations were excellent!!`
- gold：`service;room_facilities`
- hybrid 预测：`service`
- 含义：多方面句被压缩成单方面，说明当前策略在并列赞扬句上仍会漏掉次要方面。

案例 B：

- 句子：`It's old and smells a bit musty, but I don't mind that.`
- gold：`cleanliness;room_facilities`
- hybrid 预测：`cleanliness`
- 含义：当句子同时包含“老旧”和“异味”时，系统更容易优先抓住显性的清洁信号，而忽略设施老旧这一方面。

## 三、E2 首轮正式结果

### 1. 核心结果表

| Group | Candidate Hit@5 (proxy) | Avg latency (ms) |
|---|---:|---:|
| A_rating_review_count | 0.9 | 111.641 |
| B_final_aspect_score | 0.9 | 102.223 |

### 2. 指标解释

当前 proxy 指标下，`A` 与 `B` 的命中率持平，`B_final_aspect_score` 只表现出轻微的延迟优势。因此这轮 E2 更适合支撑“Aspect-KB 画像已经能进入候选缩圈链路，并可能带来一定效率收益”，还不足以支撑“Aspect-KB 明显优于弱基线”的强结论。

### 3. 失败案例总结

两组共同失败 query 为：`q021 / q022 / q023 / q081`。它们都集中在 Honolulu，且都要求 `quiet_sleep` 支持。

当前失败的主因是：

- 冻结切分下，Honolulu 在 `test split` 里只有 `1` 家酒店
- 这家酒店正是 `Ramada Plaza By Wyndham Waikiki`
- 它在当前证据索引中 `quiet_sleep = 0`，因此不可能通过当前 `2` 句 / `2` review 的代理阈值

这说明 E2 的失败首先来自**切分后的城市候选稀疏**。

同时也存在一个次级现象：该酒店有一句 `Quieter, less congested and easy to walk to where you want along the beach.` 被标为 `location_transport`，而不是 `quiet_sleep`。这反映出当前主标签机制在“安静感受”和“位置/拥挤度”交叉表达上还会损失一部分召回。

### 4. 方法边界

当前 E2 仍是 proxy 评测，不是完整 qrels 评测；因此其作用是“支持 Aspect-KB 候选缩圈可行性”，而不是“给出最终检索质量定论”。

## 四、当前最稳妥的章节结论

Aspect-KB 章节当前可以写出的最稳妥结论是：

1. 当前混合标注策略相较纯 zero-shot 明显更可靠，但尚未在主类别整体上稳定超过 rule-only。
2. 酒店方面画像已经能够支撑候选缩圈实验，但在当前 proxy 指标下只显示出轻微效率优势，尚未形成明显命中率优势。
3. `quiet_sleep` 相关失败暴露出 test split 覆盖与主标签机制的双重限制，因此下一阶段应优先推进检索链路实验，而不是提前进入 PEFT。

## 五、下一阶段顺序

建议固定为：

1. 先将本文件、`experiments/labels/e1_aspect_reliability/e1_report.md` 与 E2 正式 `analysis.md` 整理进论文小节
2. 启动 `E6`: 方面引导 vs 朴素召回
3. 启动 `E7`: reranker 消融
4. 启动 `E8`: 主通道 + 兜底通道

当前不建议启动：

- PEFT
- `G1-G4`
- “Aspect-KB 已充分优于基线”这类过强结论
