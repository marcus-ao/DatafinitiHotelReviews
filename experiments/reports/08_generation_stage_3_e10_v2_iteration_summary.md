# 生成阶段第三轮材料：E10 v2 迭代结果与 E10 v3 修复动机

更新时间：2026-04-02

本文件用于归档 `E10 v2` 的阶段性结果。它的定位是：

- 比 `E10 v1` 更好
- 已验证 `grounded_recommendation` 数据路线有效
- 但仍未超过当前正式 base，因此不能升级为最终正结果

当前固定引用目录：

- `E10 v2` PEFT formal：
  - `experiments/runs/e10_a2dd1a0bd73c57b5_20260402T073127+0000/`
- `E10 v2` compare formal：
  - `experiments/runs/e10cmp_7cf0c9c0a9830796_20260402T074331+0000/`

## 1. 一句话结论

`E10 v2` 明显优于 `v1`，已经把 `citation_precision` 从 `0.9250` 拉回到 `0.9688`，并修复了 `q013 / q023` 的保守退化；但它同时引入了 `q018 / q022` 的 schema 失稳和 `q085` 的 pack 边界错误，因此当前仍不能替代 base。

## 2. 与 v1 / base 的关系

### 相比 `v1`

- `citation_precision`
  - `0.9250 -> 0.9688`
- `auditable_query_rate`
  - `0.9750 -> 1.0000`
- 代表性退化 `q013 / q023`
  - 已被修复

这说明：

- 补入 `grounded_recommendation` supervision 是有效方向
- `v1` 的主要问题确实是训练目标与最终 grounded recommendation 任务错位

### 相比 base

- `citation_precision`
  - `0.9688 vs 0.9688`
  - 当前打平
- `schema_valid_rate`
  - `1.0000 -> 0.9500`
  - `v2` 反而下降
- `evidence_verifiability_mean`
  - `1.9704 -> 1.9403`
  - `v2` 略低

这说明：

- `v2` 已经接近 base，但还没有形成“正式可替代”的稳定正结果

## 3. 当前需要解释的三个关键样本

### `q013 / q023`

这两条是 `v1` 的主要退化点，集中表现为：

- `quiet_sleep`
- `focus + avoid`
- 复杂约束下过度保守

`v2` 已经把这类问题修回来了，说明 grounded supervision 的引入确实能改善复杂约束下的推荐收缩问题。

### `q018 / q022`

这两条是 `v2` 的新问题，模式高度一致：

- 推荐保留了
- 但模型把“某方面缺少直接证据”的说明写进了 `reasons[]`
- 最终形成：
  - `sentence_id = null`
  - 或伪造“缺证据说明”型 reason

这类错误不是 retrieval 问题，而是：

- partial-support 情况下
- 输出 schema 与 grounded reasoning 边界没有完全学稳

### `q085`

`q085` 暴露的是另一类不同错误：

- 多酒店、多方面均衡推荐场景
- citation 落在了错误酒店的 `EvidencePack`
- 本质上是 pack boundary 漏学

因此 `q085` 不能靠继续放大 `quiet_sleep` slice 解决，而需要专门补：

- 多酒店
- 多方面
- 每家酒店引用只能来自自己的 pack

## 4. 当前正式定位

当前论文中对 `v2` 的定位建议固定为：

1. `v2` 是比 `v1` 更好的第二轮结果
2. 它证明了“补 grounded training data”是有效方向
3. 但它尚未超过 base，不应升级为新的正式主系统
4. 下一步应进入 `E10 v3`，继续走“数据 + 约束修复”路线，而不是同时混入训练超参变量

## 5. E10 v3 的直接动机

`v3` 的修复目标不再是泛化补数据，而是精确对准当前错误模式：

- `q018 / q022`
  - `partial_support_keep_recommendation`
  - 允许保留酒店，但 `reasons[]` 里只能保留有证据支持的 aspect
  - 缺证据说明只能写到根级 `unsupported_notice`
- `q085`
  - `multi_hotel_pack_boundary`
  - 每家酒店的 `sentence_id` 只能来自自己的 `EvidencePack`
- `zero_recommendation`
  - 只保留真正 evidence gap 的 grounded abstain
  - 不再把 unsupported-request 驱动的 abstain 混进 grounded recommendation 正例

为保证论文归因清晰，`v3` 继续保持：

- 不改 retrieval
- 不改 `E9` eval units
- 不改正式 base baseline
- 不改 compare 协议
- 不重跑 base

唯一变化是：

- `manifest v3`
- `exp03`
- 更严格的 grounded 数据切片与 synthetic query 设计
