# 实验审计主表（E1-E10, G1-G4）

更新时间：2026-04-05

本文件是 `11_final_experiment_audit_protocol.md` 的配套主表，用于把 `E1-E10` 与 `G1-G4` 全部实验的**研究目标、query scope、gold/qrels、split、变量控制、指标口径、重跑要求、论文用途**统一收束到一份可执行、可核查、可追溯的正式审计清单中。

本表的作用不是替代各实验自己的报告，而是提供一张“**最终正式重跑前的唯一索引**”：任何一个实验如果不能在本表中清楚回答“用什么数据、回答什么问题、结果写到哪里去”，就不应直接进入今天的正式重跑。


> Override Notice (2026-04-05): For the current final-rerun campaign, all historical `reuse` labels below are treated as role-classification only.
> The execution decision for this round is a full formal rerun of `E1-E10` and `G1-G4` after archival reset.

---

## 1. 使用说明

### 1.1 字段解释

| 字段 | 含义 |
|---|---|
| 实验编号 | `E1-E10` 或 `G1-G4` |
| 章节 / 层级 | 论文中的章节定位与证据层级 |
| 研究目标 | 该实验究竟想回答什么问题 |
| 研究问题映射 | 对应 `RQ1/RQ2/RQ3`，若无则写前置验证 |
| query scope | 实验实际使用的 query 范围或样本范围 |
| gold / qrels / labels | 正式评测依赖的人工标注或 gold 资产 |
| split 依据 | train/dev/test 或 test-only 的冻结依据 |
| retrieval / candidate | 检索配置或候选集合口径 |
| generation / PEFT | 生成模型、推理方式、适配器或 manifest |
| 核心指标 | 该实验真正应该看的指标 |
| 控制变量 | 必须保持不变的变量 |
| 可变变量 | 本实验唯一允许变化的因素 |
| 结果状态 | 本轮正式重跑 |
| 论文用途 | 写到哪一章、承担什么论证角色 |

### 1.2 证据层级约定

| 层级 | 含义 |
|---|---|
| 前置验证 | 证明数据底座或工作流基础设施成立 |
| supporting evidence | 机制、消融、边界或迭代教训，不作为最终统一结论的唯一依据 |
| decisive evidence | 第七章统一对比的决定性证据 |

---

## 2. 总表

> 说明：表格较宽，建议在编辑器中横向查看；其中 `结果状态` 与 `论文用途` 是本次全量重跑前最关键的两列。

| 实验 | 章节 / 层级 | 研究目标 | 研究问题映射 | query scope / 样本范围 | gold / qrels / labels | split 依据 | retrieval / candidate | generation / PEFT | 核心指标 | 控制变量 | 可变变量 | 结果状态 | 论文用途 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `E1` | Chapter 4 / 前置验证 | 验证方面分类与情感标注质量是否足以支撑知识库 | 前置验证 | `344` 条正式 gold 句子（与 query 扩展无关） | `experiments/labels/e1_aspect_reliability/aspect_sentiment_gold.csv` | 不依赖 G query scope；独立标注集 | 无检索对比 | 无 generation；标注可靠性任务 | `Aspect macro-F1`、`Difficult-set Jaccard`、`Sentiment macro-F1` | 标注集、评估脚本、标签集合 | 标注策略（rule / zeroshot / hybrid） | **本轮正式重跑** | Chapter 4 知识库质量验证 |
| `E2` | Chapter 4 / 前置验证 | 验证酒店候选缩圈链路是否可用 | 前置验证 | 独立候选评测集；不依赖 G decisive query scope | E2 自身 proxy hit 标记 / analysis 结果 | 基于冻结 split 的 test hotel candidate 集 | `A_rating_review_count` vs `B_final_aspect_score` | 无 generation | `Candidate Hit@5 (proxy)`、`Avg latency` | 冻结 split、城市 test 酒店范围 | 候选打分策略 | **本轮正式重跑** | Chapter 4 候选筛选可行性验证 |
| `E3` | Chapter 4 / 前置验证 | 验证偏好解析是否稳定可靠 | 前置验证 | `86` 条 judged query 中可执行行为样本 | `slot_gold.jsonl` | judged query 全集；非 retrieval/generation test split | 默认检索配置冻结为 `aspect_main_no_rerank`，但 E3 核心不比较 retrieval | `Qwen3.5-4B` behavior runtime | `Exact-Match`、`Unsupported Recall`、slot-level F1 | 模型、prompt v2、后处理规则、judge query universe | 行为模型规模/版本（历史上 2B vs 4B） | **本轮正式重跑** | Chapter 4 行为基础设施验证 |
| `E4` | Chapter 4 / 前置验证 | 验证澄清触发是否可靠 | 前置验证 | `86` 条 judged query | `clarify_gold.jsonl`、clarification audit CSV | judged query 全集；不纳入 G dialog-excluded query | 默认检索配置冻结但非主比较变量 | `Qwen3.5-4B` behavior runtime | `Accuracy`、`Precision`、`Recall`、`F1`、`Over/Under-clarification` | gold 口径、prompt 版本、query scope | 行为模型规模/版本 | **本轮正式重跑** | Chapter 4 澄清模块验证 |
| `E5` | Chapter 5 / supporting evidence | 验证中文直检 vs 结构化英文桥接的必要性 | Supporting for `RQ1` | `40 core queries / 80 target units` | `E6 qrels` 体系下的 target units 与 judged labels | 同城 test 酒店；固定 query 子集 | `dense_no_rerank`，只变 query 表达（中文 vs query_en） | 无 generation | `Aspect Recall@5`、`nDCG@5`、`MRR@5`、`Precision@5` | 候选集合、retrieval backend、qrels、@5 口径 | query bridge 表达方式 | **本轮正式重跑** | Chapter 5 证明桥接必要性 |
| `E6` | Chapter 5 / supporting evidence | 验证 Aspect-guided retrieval 是否优于 plain retrieval | `RQ1` retrieval supporting evidence | `40 core queries / 80 target units` | `experiments/labels/e6_qrels/*` | 同城全部 test 酒店；固定 `40` query | `plain_city_test_rerank` vs `aspect_main_rerank` | 无 generation | 检索层 6 指标（正式 retrieval 证据主来源之一） | qrels、target units、hotel split、@5 口径 | retrieval strategy | **本轮正式重跑** | Chapter 5 retrieval 主正结果 |
| `E7` | Chapter 5 / supporting evidence | 验证 reranker 是否带来稳定收益 | `RQ1` retrieval ablation | `40 core queries / 80 target units` | `E6 qrels` | 同城全部 test 酒店；固定 `40` query | `aspect_main_no_rerank` vs `aspect_main_rerank` | 无 generation | 检索层 6 指标，重点是 `nDCG@5 / MRR@5 / latency` | query scope、qrels、candidate set | reranker on/off | **本轮正式重跑** | Chapter 5 配置消融 |
| `E8` | Chapter 5 / supporting evidence | 验证 fallback 是否带来收益 | `RQ1` retrieval ablation | `40 core queries / 80 target units` | `E6 qrels` | 同城全部 test 酒店；固定 `40` query | `aspect_main_rerank` vs `aspect_main_fallback_rerank` | 无 generation | `nDCG@5`、`Precision@5`、fallback 激活/噪声率 | query scope、qrels、retrieval backend | fallback on/off | **本轮正式重跑** | Chapter 5 边界消融 |
| `E9` | Chapter 5 / supporting evidence | 验证有无 RAG / 有无证据约束对推荐生成的影响 | `RQ1` generation supporting evidence | `40` 条 E9 官方 generation query | `e9_generation_eval_query_ids.json`、`e9_generation_eval_units.jsonl`、citation audit | 固定 retrieval backend、固定 E9 eval units | 固定 `aspect_main_no_rerank` + `E2 B_final_aspect_score Top5` | `Qwen3.5-4B`，A/B/C/D groups | `Recommendation Coverage`、`Schema Valid Rate`、`Citation Precision`、`Evidence Verifiability Mean` | eval units、model、prompt schema、candidate assets | evidence availability / verifier | **本轮正式重跑** | Chapter 5 端到端 RAG 生成对比 |
| `E10 v1` | Chapter 6 / supporting evidence | 验证 exp01 是否优于 base，并识别训练目标错位 | `RQ2` supporting chain | `40` 条 E10 官方 compare query | 固定 E9-derived eval units + citation audit | test-only generation eval；训练数据按 hotel split 切分 | retrieval 不变，沿用 grounded assets | base vs `PEFT exp01` | `Citation Precision`、`Evidence Verifiability Mean`、`Schema Valid Rate`、`Auditable Query Rate` | eval units、retrieval、backend、compare 协议 | adapter（exp01） | **本轮正式重跑** | Chapter 6 负结果起点 |
| `E10 v2` | Chapter 6 / supporting evidence | 验证补入 grounded training data 是否修复 exp01 退化 | `RQ2` supporting chain | 同 `40` query compare scope | `manifest v2`、`sft_manifest_v2_report.json` | train/dev/test hotel split 冻结 | retrieval 不变 | base vs `PEFT exp02` | 同 E10 核心指标，额外关注 schema / pack boundary | base、retrieval、eval units、compare 协议 | 训练数据构成（v2） | **本轮正式重跑** | Chapter 6 阶段性改进证据 |
| `E10 v3` | Chapter 6 / supporting evidence | 验证更严格 grounded 数据设计的收益与副作用 | `RQ2` supporting chain | 同 `40` query compare scope | `manifest v3`、`sft_manifest_v3_report.json` | train/dev/test hotel split 冻结 | retrieval 不变 | base vs `PEFT exp03` | 同 E10 指标 | retrieval、eval units、backend 不变 | 训练数据构成（v3） | **本轮正式重跑** | Chapter 6 继续迭代 / 退化分析 |
| `E10 v4` | Chapter 6 / supporting evidence | 验证更细切片数据与 DeepSeek 辅助路线的最终边界 | `RQ2` supporting chain | 同 `40` query compare scope | `manifest v4`、`seed specs`、review logs、accepted grounded data | train/dev/test hotel split 冻结 | retrieval 不变 | base vs `PEFT exp04` | 同 E10 指标 | retrieval、eval units、compare 协议不变 | 训练数据构成（v4） | **本轮正式重跑** | Chapter 6 “按下葫芦浮起瓢”证据 |
| `G1` | Chapter 7 / decisive evidence | Plain Retrieval + Base，作为统一矩阵基线 | `RQ1/RQ2/RQ3` | `68 = 39 core + 29 robustness` | `g_eval_query_ids_68.json`、G generation eval units、G qrels（retrieval core / robustness 按设计处理） | 同城 test 酒店；统一 G split / eval assets | `plain_city_test_rerank` + `G_plain_retrieval_top5` | Base `Qwen3.5-4B` | 检索层 6 指标、生成层 7 指标、Judge、盲评、统计检验 | query scope、split、candidate policy、prompt、backend | retrieval plain；no PEFT | **本轮正式重跑** | Chapter 7 统一矩阵基线 |
| `G2` | Chapter 7 / decisive evidence | Aspect Retrieval + Base，验证 RAG 主效应 | `RQ1/RQ3` | 同 `68` query | 同 G 统一资产；retrieval 侧需明确 formal vs proxy 来源 | 同 G split | `aspect_main_no_rerank` + `G_aspect_retrieval_top5` | Base `Qwen3.5-4B` | 同 G 统一指标 | query scope、split、backend、PEFT disabled | retrieval aspect；no PEFT | **本轮正式重跑** | Chapter 7 RAG 主效应 |
| `G3` | Chapter 7 / decisive evidence | Plain Retrieval + PEFT exp02，验证 PEFT 主效应 | `RQ2/RQ3` | 同 `68` query | G generation assets + `exp02` metadata | 同 G split；adapter lineage 必须冻结 | plain retrieval 作为共享输入 | `PEFT exp02` | 同 G 统一指标 | query scope、split、retrieval plain、backend、prompt | PEFT on/off | **本轮正式重跑** | Chapter 7 PEFT 主效应 |
| `G4` | Chapter 7 / decisive evidence | Aspect Retrieval + PEFT exp02，验证联合效果与互补性 | `RQ1/RQ2/RQ3` | 同 `68` query | G generation assets + `exp02` metadata | 同 G split | aspect retrieval 作为共享输入 | `PEFT exp02` | 同 G 统一指标 | query scope、split、backend、prompt | retrieval aspect + PEFT | **本轮正式重跑** | Chapter 7 联合增强 / 互补性 |

---

## 3. 附表：Judge、盲评与统计检验的审计定位

### 3.1 LLM Judge

| 项目 | 说明 |
|---|---|
| 作用 | 对 `G1-G4` 回复做盲评，补充自动质量感知证据 |
| 适用范围 | 只服务第七章统一对比，不回写替代 E9/E10 supporting evidence |
| query scope | `70` |
| 评审维度 | `Relevance / Traceability / Fluency / Completeness / Honesty` |
| 关键控制变量 | 同一 judge model、同一 blind prompt、同一解析逻辑 |
| 论文用途 | supplementary decisive evidence |

### 3.2 Human Blind Review

| 项目 | 说明 |
|---|---|
| 作用 | 提供人类偏好层的补充证据 |
| 适用范围 | 只服务 G1-G4 统一对比 |
| 抽样范围 | `4 × 15-20` query response |
| 关键控制变量 | blind mapping、同一 worksheet、统一评分维度 |
| 论文用途 | decisive supplementary evidence |

### 3.3 Pairwise Statistical Tests

| 项目 | 说明 |
|---|---|
| 作用 | 对 G1-G4 组间差异做 paired inferential support |
| 输入 | `extract_g_group_score_map()` 生成的 query-level payload |
| 关键控制变量 | `query_id` 对齐、Holm/Bonferroni、effect size 解释字段 |
| 风险提示 | 对 ceiling metrics 与子集指标必须避免 overclaim |
| 论文用途 | decisive supplementary evidence |

---

## 4. 今日重跑前必须先确认的 stop/go 清单

### Stop（未通过则不应正式重跑）

1. 任何实验无法明确说清楚自己的 query scope；  
2. `G1-G4` 的 generation assets、retrieval assets、qrels、adapter metadata 未冻结；  
3. `E10 v1-v4` 的变量控制表无法说明“变化主要来自训练数据构成”；  
4. retrieval supporting evidence 与 G decisive evidence 的论文用途边界不清；  
5. 同名指标在不同结果文件中的聚合口径仍不一致。

### Go（满足后可进入正式重跑）

1. 所有实验在本表中的字段都已补齐；  
2. `12 / 13 / 14` 三份审计文档形成闭环；  
3. 当前正式重跑只会改变“需要新跑”的实验，不会破坏 supporting evidence 的历史角色。

---

## 5. 最终说明

本主表的作用，不是要求所有实验今天都“重做一遍”，而是要求：

> 每一个最终会进入论文的实验结果，都必须在“研究目标、数据范围、控制变量、指标口径、论文角色”这五个维度上有且只有一种正式解释。

只有在这个前提下，今天的全量重跑才是“**研究级别的正式重跑**”，而不是“工程层面的批量脚本执行”。
