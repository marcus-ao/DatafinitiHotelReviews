# 基于酒店评论知识库与参数高效微调的有状态会话式酒店推荐工作流研究

## 论文总体规划稿（v3.2 同步版）

更新时间：2026-04-05

---

## 一、研究聚焦

**在使用 LLM 进行酒店推荐的场景下，构建酒店评论知识库，探究 RAG 与 PEFT 两种策略对 LLM 推荐质量的影响，分析二者的效果差异与互补关系。**

### 1.1 三个研究问题

| 编号 | 研究问题 | 核心对比 |
| --- | --- | --- |
| RQ1 | RAG（Aspect 引导检索）能否提升 LLM 酒店推荐的证据质量与推荐覆盖率？ | G2 vs G1, G4 vs G3 |
| RQ2 | PEFT（QLoRA 微调）能否提升 LLM 酒店推荐的行为稳定性与约束遵守能力？ | G3 vs G1, G4 vs G2 |
| RQ3 | RAG 与 PEFT 是互补还是替代关系？二者在不同维度上的增益是否可叠加？ | G4 vs max(G2, G3) |

### 1.2 研究假设

- H1：Aspect 引导检索在证据覆盖率、检索排序质量和引用精确度上显著优于通用检索，且在复杂多方面查询上优势更大。
- H2：PEFT 的收益主要体现在行为层（偏好解析准确率、格式合规率、约束遵守率），而非事实知识记忆。
- H3：RAG 改善"输入质量"（更好的证据），PEFT 改善"模型质量"（更好的行为），二者改善维度不同，存在互补空间。

---

## 二、实验数据规模设计

### 2.1 查询集扩展方案

当前 E9/E10 仅使用 40 条查询（4 种类型 × 10 城市），偶然性风险较高。

扩展方案：在原 40 条核心查询基础上，新增 30 条含不支持约束的查询作为"鲁棒性评估层"，原设计总计 70 条。当前正式执行中，出于 frozen aspect mainline 的 evidence-backed candidate 可行性约束，`q021 / q024` 已按 Protocol A 从 decisive G scope 中对称剔除，因此当前正式 decisive matrix 使用 `68` 条查询（`39 core + 29 robustness`），而 `q021 / q024` 作为 supporting boundary cases 保留分析价值。

| 查询层 | 查询类型 | 数量 | 评估目的 |
| --- | --- | --- | --- |
| 核心层（40 条） | single_aspect, multi_aspect, focus_and_avoid, multi_aspect_strong | 40 | 检索与推荐的核心能力 |
| 鲁棒性层（30 条） | unsupported_budget, unsupported_distance, unsupported_heavy | 30 | 边界约束处理与诚实告知能力 |

不纳入 conflict（10 条）和 missing_city（6 条），因为这些属于对话管理层面的测试，与 RAG/PEFT 对推荐质量的影响这一主题关系不大。

### 2.2 统计检验要求

所有涉及组间对比的实验均需引入统计检验：

- 配对 Wilcoxon 符号秩检验（适用于小样本非参数比较）
- Bootstrap 置信区间（95%，1000 次重采样）
- 效应量报告（Cohen's d 或 r）
- 在结果表格中标注显著性水平（p < 0.05 / p < 0.01 / p < 0.001）

---

## 三、评价指标体系

### 3.1 检索层指标（6 项）

| 指标 | 含义 | 计算方式 |
| --- | --- | --- |
| Aspect Recall@5 | 检索结果是否覆盖用户关注的全部方面 | 在 Top-5 中命中的目标方面数 / 目标方面总数 |
| nDCG@5 | 检索结果的排序质量 | 按 qrels 相关性计算归一化折损累积增益 |
| Precision@5 | 检索结果中相关句子的比例 | Top-5 中被 qrels 标注为相关的句子数 / 5 |
| MRR@5 | 首个相关结果出现的位置 | 1 / 首个相关结果在 Top-5 中的排名 |
| Evidence Diversity | 检索结果的多样性 | Top-5 中涉及的不同酒店数 + 不同方面数 |
| Retrieval Latency | 检索效率 | 从查询编码到返回 Top-5 的端到端耗时（ms） |

### 3.2 生成层指标（7 项）

| 指标 | 含义 | 计算方式 |
| --- | --- | --- |
| Citation Precision | 推荐理由中引用的 sentence_id 是否合法 | 合法引用数 / 总引用数 |
| Evidence Verifiability | 引用句子是否真正支撑推荐理由 | 人工评分（0=不支持, 1=部分, 2=明确支持）取均值 |
| Schema Valid Rate | 输出是否符合 RecommendationResponse 格式 | 通过 Pydantic 校验的查询比例 |
| Recommendation Coverage | 有非空推荐输出的查询比例 | 含 ≥1 条推荐的查询数 / 总查询数 |
| Aspect Alignment Rate | 推荐理由是否回应了用户的关注方面 | 推荐理由涉及的方面 ∩ 用户 focus_aspects / 用户 focus_aspects |
| Hallucination Rate | 推荐理由中无证据支撑的声明比例 | 人工判定无证据支撑的 reason 数 / 总 reason 数 |
| Unsupported Honesty Rate | 对不支持约束的诚实告知率 | 正确标记不支持约束的查询数 / 含不支持约束的查询数 |

### 3.3 非计算性评价体系

#### 3.3.1 LLM-as-Judge 自动评分

使用 DeepSeek-Reasoner 作为评审模型，对每条推荐回复从以下 5 个维度进行 1-5 分评分：

| 维度 | 评分标准 |
| --- | --- |
| 相关性（Relevance） | 推荐是否回应了用户查询中的核心需求 |
| 事实可追溯性（Traceability） | 推荐理由是否可以追溯到具体的评论证据 |
| 表达流畅性（Fluency） | 推荐文本是否自然、通顺、无语法错误 |
| 完整性（Completeness） | 是否覆盖了用户关注的全部方面维度 |
| 诚实性（Honesty） | 对证据不足或不支持约束是否做了诚实说明 |

评分采用盲评方式：LLM 评审时不提供组别标签，仅提供用户查询和推荐回复。每组每条查询的 5 维分数取平均作为该组的 LLM Judge Score。

#### 3.3.2 人工专家盲评

从 G1-G4 四组中各随机抽取 15-20 条查询的推荐回复，由人工评审进行盲评。评审流程：

- 评审者不知道回复来自哪个实验组
- 对每条回复评分（1-5 分）：整体推荐质量、证据可信度、实用价值
- 对两两配对的回复进行偏好选择（A 优于 B / 无差异 / B 优于 A）
- 汇总计算偏好胜率和平均质量得分

---

## 四、论文章节架构与详细规划

### 第一章 摘要

中英文各 300-500 字，聚焦 RAG vs PEFT 对比主题、核心实验结果和主要结论。

### 第二章 绪论

#### 2.1 研究背景与意义

**规划意义**：建立研究的出发点，说明为什么要研究这个问题。

- 2.1.1 LLM 在推荐系统中的应用趋势
  - 说明 LLM 正在从"列表推荐"转向"会话推荐"，以及这种转变带来的新能力和新问题
- 2.1.2 LLM 推荐面临的核心挑战
  - 聚焦三个具体问题：幻觉（推荐无中生有的酒店信息）、证据不足（推荐理由无法追溯）、行为不稳定（输出格式不合规、约束不遵守）
- 2.1.3 两条增强路线：RAG vs PEFT
  - 提出 RAG 是"改善输入"路线（给模型更好的证据上下文），PEFT 是"改善模型"路线（让模型本身更稳定可控），引出对比研究的必要性

**对论文的作用**：让读者理解为什么这个课题有研究价值，以及为什么选择 RAG 和 PEFT 作为对比对象。

#### 2.2 国内外研究现状

**规划意义**：定位本研究在现有文献中的位置，找到研究空白。

- 2.2.1 基于评论的推荐系统
  - 从协同过滤到评论文本利用的演进，方面级情感分析在推荐中的应用
- 2.2.2 RAG 在推荐系统中的应用
  - RAG 原始架构（Lewis 等, 2020）、在推荐解释生成中的应用、Aspect 引导检索的研究空白
- 2.2.3 PEFT 在对话与推荐系统中的应用
  - LoRA/QLoRA 的发展、在指令遵循和对话系统中的应用、在推荐场景下的探索不足
- 2.2.4 RAG 与 PEFT 的对比研究
  - 指出当前文献中极少有在同一推荐场景下系统对比 RAG 和 PEFT 的工作，这是本研究填补的空白

**对论文的作用**：通过文献综述证明 RAG vs PEFT 对比在推荐场景下的研究空白确实存在。

#### 2.3 研究问题与技术路线

**规划意义**：将研究问题正式化，建立后续章节的逻辑锚点。

- 2.3.1 三个研究问题的提出（RQ1/RQ2/RQ3）
- 2.3.2 2×2 实验矩阵设计思路——为什么需要交叉对比而不是单独评测
- 2.3.3 技术路线概述——知识库构建 → RAG 增强 → PEFT 增强 → 统一对比

**对论文的作用**：让读者知道本研究用什么方法回答什么问题。

#### 2.4 论文结构安排

各章内容简介与逻辑关系。

---

### 第三章 相关技术与理论基础

#### 3.1 检索增强生成（RAG）

**规划意义**：为第五章（RAG 增强）提供技术原理支撑。

- 3.1.1 稠密检索原理
  - 双塔编码器架构、嵌入空间的语义匹配、与稀疏检索（BM25）的对比
  - BGE 嵌入模型的特点（384 维，归一化向量）
- 3.1.2 向量数据库
  - ChromaDB 的存储与检索机制、近似最近邻（ANN）算法
- 3.1.3 交叉编码器重排序
  - 与双塔编码器的区别、精排在检索管线中的位置
- 3.1.4 RAG 工作流架构
  - 检索 → 增强 → 生成的三阶段流程、"知识外置"思想的技术实现

**对论文的作用**：读者读完本节后应能理解第五章中 Aspect 引导检索的设计原理。

#### 3.2 参数高效微调（PEFT）

**规划意义**：为第六章（PEFT 增强）提供技术原理支撑。

- 3.2.1 LoRA 原理
  - 低秩分解假设、ΔW = BA 的数学表达、可训练参数量分析
- 3.2.2 QLoRA 原理
  - NF4 量化方案、4-bit 基座模型 + BFloat16 适配器的混合精度架构
- 3.2.3 SFT 训练策略
  - 监督微调的数据组织、Chat 格式模板、Packing 策略
- 3.2.4 PEFT 与全参数微调的对比
  - 参数效率、存储开销、对基座知识的保留

**对论文的作用**：读者读完本节后应能理解第六章中 QLoRA 微调方案的设计依据。

#### 3.3 酒店评论方面级分析（简述）

**规划意义**：为第四章（知识库构建）提供简要技术背景。

- 3.3.1 方面级情感分析概述
- 3.3.2 零样本分类的 NLI 转化思想

**对论文的作用**：让读者理解知识库中方面标签的生成原理，但不展开为核心贡献。

---

### 第四章 酒店评论知识库构建与推荐工作流基础

**章节定位**：前置条件章，非核心贡献。目标是用尽可能精简的篇幅证明"数据底座可靠、工作流基础设施可用"，为后续三个核心章节扫清前提。

#### 4.1 数据概况与预处理

**规划意义**：说明数据来源、规模和清洗过程，让读者了解知识库的原始材料。

- 数据来源（Datafiniti Hotel Reviews）、城市筛选（10 城市）、清洗去重、句子切分
- 精简为约 500-800 字 + 1 张数据规模汇总表

**对论文的作用**：让读者知道知识库建立在什么数据之上。不展开工程细节。

#### 4.2 知识库存储架构

**规划意义**：说明知识库的双引擎架构，这是 RAG 检索的物理基础。

- PostgreSQL 关系型存储（5 表 2 视图概述）
- ChromaDB 向量索引（BGE 嵌入、384 维）
- 六维 Aspect 体系与方面画像公式

**对论文的作用**：让读者理解后续 RAG 检索的物理层。

#### 4.3 知识库质量验证

**规划意义**：用实验数据证明知识库的标注和筛选质量可接受。

- E1 方面分类可靠性：hybrid Jaccard=0.7911, macro-F1=0.4960
  - 必须补充：zeroshot F1=0.0932 低于随机基线的原因分析
  - 必须补充：hybrid 模式下零样本处理的句子占比和质量
- E2 候选酒店筛选：Hit@5=0.9
- 情感分析 macro-F1=0.4458 的局限性讨论

**实验安排合理性**：E1 和 E2 共 2 组实验用于验证知识库的两个核心功能（标注准确性和筛选有效性），对于"前置条件验证"的定位是充分的。不需要在这一章追加更多实验。

#### 4.4 推荐工作流基础设施验证

**规划意义**：证明偏好解析和澄清触发这两个基础模块是可靠的，因此后续 RAG 和 PEFT 实验的上游输入不存在系统性偏差。

- E3 偏好解析：4B Exact-Match=0.9767（简述，不展开）
- E4 澄清决策：4B F1=0.9697（简述，不展开）
- 基座模型选型依据：Qwen3.5-4B vs 2B 的对比结论

**实验安排合理性**：E3 和 E4 各 1 组实验用于验证两个基础模块。这些不是本研究的核心贡献，简要呈现结果即可。

---

### 第五章 基于 RAG 的 LLM 推荐增强（核心章 ①）

**章节定位**：第一个核心贡献章。目标是完整论证 RAG（特别是 Aspect 引导检索）对 LLM 推荐质量的提升效果。

**本章实验安排总览（6 组实验 + 1 组统计检验）**：

| 实验编号 | 对比内容 | 评价层次 | 在本章的角色 |
| --- | --- | --- | --- |
| E5 | 中文直接检索 vs 结构化英文检索 | 检索层 | 验证查询桥接的必要性 |
| E6 | Aspect 引导检索 vs 通用稠密检索 | 检索层 | RAG 核心正结果 |
| E7 | 加 Reranker vs 不加 Reranker | 检索层 | 配置消融（负结果） |
| E8 | 加 Fallback vs 不加 Fallback | 检索层 | 配置消融（负结果） |
| E9 B vs D | 有证据约束 vs 无证据约束 | 生成层 | RAG 对推荐生成的端到端影响 |
| G2 vs G1 | Aspect 检索 + Base vs Plain 检索 + Base | 检索层 + 生成层 | RAG 主效应的统一框架验证 |
| 统计检验 | Wilcoxon + Bootstrap CI | 全部指标 | 排除偶然性 |

**实验充分性评估**：6 组实验覆盖了 RAG 的三个层次——(1) 查询桥接（E5），(2) 检索策略（E6/E7/E8 消融），(3) 端到端生成影响（E9 消融 + G2 vs G1）。加上统计检验，对于回答 RQ1 是充分且严谨的。

#### 5.1 Aspect 引导检索策略设计

**规划意义**：提出本研究的 RAG 改进方案。

- 5.1.1 通用稠密检索在酒店推荐中的局限
  - 方面不匹配问题的具体例子
- 5.1.2 Aspect 引导检索的设计
  - 方面维度过滤 → 向量检索的两阶段流程
- 5.1.3 三种检索模式的设计与对比
  - aspect_main_no_rerank / aspect_main_rerank / aspect_main_fallback_rerank
- 5.1.4 证据包（EvidencePack）构建
  - 每家候选酒店的证据封装格式

**对论文的作用**：定义 RAG 策略的技术方案，为后续实验建立评测对象。

#### 5.2 中英文查询桥接机制

**规划意义**：解决"中文用户 + 英文知识库"的跨语言检索问题。

- 5.2.1 跨语言检索问题分析
- 5.2.2 结构化归一 → 英文检索表达方案
- 5.2.3 实验验证：E5 桥接效果
  - nDCG@5: 0.2643 → 0.6457（+38.1pp）
  - 代表性案例分析（q043）

**对论文的作用**：证明桥接是 RAG 管线的必要组件。

#### 5.3 检索质量对比实验

**规划意义**：通过消融实验确定最优检索配置。

- 5.3.1 E6：Aspect 引导 vs 通用稠密检索（核心正结果）
  - nDCG@5: 0.3307 → 0.6378（+30.7pp）
  - 6 项检索层指标完整呈现
  - 统计检验结果
- 5.3.2 E7：Reranker 消融（负结果）
  - nDCG@5 无增益，延迟 2.6x
- 5.3.3 E8：Fallback 消融（负结果）
  - 噪声率 100%
- 5.3.4 检索配置冻结决策
  - 选择 aspect_main_no_rerank 的定量依据

**对论文的作用**：提供检索层的完整证据链。

#### 5.4 RAG 增强下的推荐生成质量

**规划意义**：从检索层上升到生成层，验证 RAG 对端到端推荐质量的影响。

- 5.4.1 E9 RAG 消融实验（B_grounded vs D_no_evidence）
  - coverage: 0.95 vs 0.825（+12.5pp）
  - 7 条 RAG 恢复案例的详细分析
  - q021 异常案例（无 RAG 产生虚假推荐）
  - 7 项生成层指标完整呈现
- 5.4.2 G2 vs G1 对比（Aspect 检索 + Base vs Plain 检索 + Base）
  - 统一评估框架下的检索层 + 生成层双层指标
  - 统计检验结果
  - LLM-as-Judge 评分对比

**对论文的作用**：从生成质量角度回答 RQ1。

#### 5.5 RAG 增强的效果边界分析

**规划意义**：诚实讨论 RAG 的局限。

- 证据稀疏时的系统行为（Honolulu quiet_sleep 案例）
- RAG 的延迟代价（~2x）
- 方面标签单一化的影响

**对论文的作用**：展示学术诚实性，预防答辩质疑。

---

### 第六章 基于 PEFT 的 LLM 推荐增强（核心章 ②）

**章节定位**：第二个核心贡献章。目标是论证 PEFT 对 LLM 推荐行为的影响，包括正面改进和负面教训。

**本章实验安排总览（5 组实验 + 迭代消融链）**：

| 实验编号 | 对比内容 | 评价层次 | 在本章的角色 |
| --- | --- | --- | --- |
| E10 v1 | Base vs PEFT exp01 | 生成层 | 训练目标错位的负结果（消融） |
| E10 v2 | Base vs PEFT exp02 | 生成层 | 数据修正后的阶段性改进 |
| E10 v3 | Base vs PEFT exp03 | 生成层 | 进一步迭代（v3 退化分析） |
| E10 v4 | Base vs PEFT exp04 | 生成层 | schema 恢复但 CP 退化 |
| G3 vs G1 | PEFT + Plain 检索 vs Base + Plain 检索 | 检索层 + 生成层 | PEFT 主效应的统一框架验证 |
| 统计检验 | Wilcoxon + Bootstrap CI | 全部指标 | 排除偶然性 |

**实验充分性评估**：E10 的 4 轮迭代（v1→v2→v3→v4）形成了一条完整的训练目标消融链——训练数据如何影响微调效果。G3 vs G1 在统一框架下验证 PEFT 主效应。5 组实验 + 统计检验对于回答 RQ2 是充分的。

**PEFT 适配器选择建议**：在 G3/G4 中使用 exp02（v2 适配器），因其 Citation Precision（0.9688）追平 base，是所有迭代中综合最优。exp01/exp03/exp04 作为消融对照。

#### 6.1 QLoRA 微调方案设计

**规划意义**：说明 PEFT 的技术实现细节。

- 6.1.1 基座模型选择（Qwen3.5-4B）
- 6.1.2 QLoRA 超参配置（r=16, α=32, dropout=0.05, 7 个目标模块, 4-bit 量化）
- 6.1.3 训练基础设施（AutoDL GPU, accelerate launch, TRL SFTTrainer）

**对论文的作用**：提供可复现的微调配置。

#### 6.2 训练数据构建与迭代策略

**规划意义**：这是理解 PEFT 效果差异的关键——训练数据的质量比超参调优更重要。

- 6.2.1 五类 SFT 样本设计
  - preference_parse, clarification, constraint_honesty, feedback_update, grounded_recommendation
- 6.2.2 四版 Manifest 的演进
  - v1：4 类行为样本，无 grounded（234 条 train）
  - v2：+grounded_recommendation（390 条 train）
  - v3：数据修复 + 约束收紧（391 条 train）
  - v4：进一步修复（394 条 train）
- 6.2.3 训练数据防泄漏原则
  - 严格按酒店切分，test 酒店数据不进入训练

**对论文的作用**：让读者理解后续消融实验中不同适配器表现差异的根因。

#### 6.3 PEFT 增强下的推荐生成质量

**规划意义**：呈现 PEFT 的最终效果。

- 6.3.1 E10 v2 正式结果（CP=0.9688 追平 base，schema=0.95）
- 6.3.2 G3 vs G1 对比（统一框架）
- 6.3.3 LLM-as-Judge 评分对比

**对论文的作用**：提供 PEFT 增强效果的定量证据。

#### 6.4 训练目标对齐的消融分析

**规划意义**：这是本章最有学术价值的部分——通过四轮迭代的消融链，揭示训练数据如何决定微调效果。

- 6.4.1 E10 v1 负结果分析
  - CP 从 0.9688 降到 0.9250：缺少 grounded 监督导致过度保守
  - q013/q023 坍缩为 0 推荐的根因
- 6.4.2 E10 v2 阶段改进
  - 补入 grounded 数据后 CP 恢复，但 schema 退化（q018/q022/q085）
- 6.4.3 E10 v3/v4 的进一步迭代
  - v3：schema 进一步退化（0.9），q013 重新坍缩
  - v4：schema 恢复但 CP 降到 0.9（q022/q033/q043 退化）
  - 迭代过程中的"按下葫芦浮起瓢"现象
- 6.4.4 消融链的总结性结论
  - 训练数据的场景覆盖比训练超参更重要
  - 小数据（~400 样本）+ 小模型（4B）条件下 PEFT 的天花板
  - 当前 base 模型在 grounded 推荐任务上的能力已接近 PEFT 能达到的上界

**对论文的作用**：将"负结果"转化为有方法论价值的消融分析。

| 适配器 | CP | Schema | 训练数据特点 | 教训 |
| --- | --- | --- | --- | --- |
| Base | 0.9688 | 1.0 | — | 基座已相当稳定 |
| exp01 (v1) | 0.9250 | 1.0 | 无 grounded | 训练目标错位 |
| exp02 (v2) | 0.9688 | 0.95 | +grounded | 数据修正有效 |
| exp03 (v3) | 0.9250 | 0.9 | 约束收紧 | 过度修复反伤 |
| exp04 (v4) | 0.9000 | 1.0 | 再次修复 | 按下葫芦浮起瓢 |

#### 6.5 PEFT 增强的效果边界分析

- 小数据条件下的训练震荡
- "行为内化"与"事实知识记忆"的边界
- 对更大训练数据规模和更大模型规模的展望

---

### 第七章 RAG 与 PEFT 的对比分析（核心章 ③ / 全文高潮）

**章节定位**：第三个核心贡献章，也是全文的高潮。目标是在统一框架下正面回答三个研究问题。

**本章实验安排总览**：

| 实验 | 内容 | 评价维度 |
| --- | --- | --- |
| G1 vs G2 vs G3 vs G4 | 2×2 完整矩阵 | 检索层 6 指标 + 生成层 7 指标 |
| 统计检验 | 所有组间配对比较 | Wilcoxon + Bootstrap CI + Cohen's d |
| LLM-as-Judge | 四组盲评 | 5 维度 × 70 查询 |
| 人工专家盲评 | 四组抽样盲评 | 3 维度 × 15-20 查询 |

#### 7.1 统一评估框架设计

**规划意义**：建立严格的实验协议，保证四组对比的公平性。

- 7.1.1 G1-G4 矩阵定义
- 7.1.2 变量控制冻结清单（查询集、候选酒店、推理参数、Prompt、Schema）
- 7.1.3 评价指标体系（检索层 6 项 + 生成层 7 项 + LLM Judge 5 维 + 人工 3 维）
- 7.1.4 统计检验方案

**对论文的作用**：让评审确信实验对比是公平的。

#### 7.2 四组实验结果

**规划意义**：完整呈现实验数据。

- 7.2.1 四组核心指标汇总表
- 7.2.2 检索层指标分组对比
- 7.2.3 生成层指标分组对比
- 7.2.4 LLM-as-Judge 评分对比
- 7.2.5 人工盲评结果

**对论文的作用**：提供回答 RQ1/RQ2/RQ3 的原始证据。

#### 7.3 RAG vs PEFT：效果差异分析

**规划意义**：回答 RQ1 和 RQ2。

- 7.3.1 RAG 的效果维度
  - G2 vs G1：RAG 在 base 模型上的提升
  - G4 vs G3：RAG 在 PEFT 模型上的提升
  - RAG 主要改善的指标维度（检索质量、证据覆盖、引用精确度）
- 7.3.2 PEFT 的效果维度
  - G3 vs G1：PEFT 在 Plain 检索上的提升
  - G4 vs G2：PEFT 在 Aspect 检索上的提升
  - PEFT 主要改善的指标维度（行为稳定性、格式合规、约束遵守）
- 7.3.3 两者效果的维度差异
  - RAG 更影响"证据质量"，PEFT 更影响"行为质量"
  - 或者两者在当前数据规模下差异不显著——这本身也是有价值的发现

**对论文的作用**：这是论文的核心贡献段落。

#### 7.4 RAG + PEFT：互补性分析

**规划意义**：回答 RQ3。

- 7.4.1 G4 vs max(G2, G3)：叠加效果是否优于单独最优
- 7.4.2 如果互补：各自贡献了什么——RAG 贡献证据维度，PEFT 贡献行为维度
- 7.4.3 如果不互补：可能的原因分析（如当前 PEFT 效果不足以产生叠加增益）

**对论文的作用**：提供 RQ3 的实证答案。

#### 7.5 讨论

**规划意义**：将实验结论提升到方法论层面。

- 何时优先使用 RAG（数据丰富但模型不可调时）
- 何时优先使用 PEFT（模型可调但数据稀疏时）
- 在资源受限条件下的策略建议
- 对其他推荐场景的可推广性讨论

---

### 第八章 结论与展望

#### 8.1 研究总结

- 回答 RQ1：RAG 的效果总结
- 回答 RQ2：PEFT 的效果总结
- 回答 RQ3：二者关系的总结

#### 8.2 研究创新点

- 创新点一：提出 Aspect 引导检索策略并通过受控实验验证其相对于通用检索的显著优势
- 创新点二：通过四轮 PEFT 迭代消融揭示了训练目标对齐的重要性和小数据微调的能力边界
- 创新点三：在统一评估框架下首次对比 RAG 和 PEFT 在酒店推荐场景中的效果差异与互补关系

#### 8.3 不足与展望

- 数据规模局限（10 城市、146 酒店）
- 情感分析精度不足（F1=0.4458）的下游影响
- PEFT 训练数据规模（~400 样本）的天花板
- 未来方向：更大数据规模、更大模型规模、多语言知识库、端到端工作流原型

---

## 五、实验总量清点

### 5.1 各章实验分布

| 章节 | 已有实验 | 需新增实验 | 实验总量 | 充分性判定 |
| --- | --- | --- | --- | --- |
| 第四章（前置验证） | E1, E2, E3, E4 | 无 | 4 组 | 充分（前置条件验证不需要过多实验） |
| 第五章（RAG） | E5, E6, E7, E8, E9 消融 | G1, G2 | 7 组 | 充分（覆盖桥接/检索/消融/生成四层） |
| 第六章（PEFT） | E10 v1/v2/v3/v4 | G3 | 5 组 | 充分（四轮消融链 + 统一框架验证） |
| 第七章（对比） | — | G1-G4 统一分析 + LLM Judge + 人工盲评 | 3 类评估 | 充分（统一框架 + 多维度评价） |
| **合计** | 13 组已有 | 3 组需新增 + 2 类评估 | **16 组实验 + 2 类评估** | — |

### 5.2 需新增的实验清单

| 编号 | 实验 | 所需条件 | 预估工作量 |
| --- | --- | --- | --- |
| G1 | Plain Retrieval + Base 模型推荐生成（当前正式 decisive scope：68 条） | AutoDL GPU + Plain RAG 代码 | 3-4 小时 |
| G2 复跑 | Aspect 检索 + Base 模型推荐生成（当前正式 decisive scope：68 条） | AutoDL GPU | 3-4 小时 |
| G3 | Plain Retrieval + PEFT exp02 推荐生成（当前正式 decisive scope：68 条） | AutoDL GPU + exp02 适配器 | 3-4 小时 |
| G4 复跑 | Aspect 检索 + PEFT exp02 推荐生成（当前正式 decisive scope：68 条） | AutoDL GPU | 3-4 小时 |
| LLM Judge | DeepSeek 盲评 4×68 条回复 | DeepSeek API | 2-3 小时 |
| 人工盲评 | 4×15-20 条抽样盲评 | 人工标注 | 3-5 小时 |
| 统计检验 | 所有组间 Wilcoxon + Bootstrap CI | 本地计算 | 1-2 小时 |

截至 `2026-04-05` 的工作区状态是：

- `G1-G4` 仍然没有正式云端结果，因此“实验结果”维度依然视为未完成
- 但围绕 `G1-G4` 的代码底座已经不是空白状态，而是进入了“代码基本就绪、正式结果待跑”的阶段
- 当前已落地的模块包括：
  - `g_eval_query_ids_68.json`
  - `g_run_generation / g_compare_runs`
  - `statistical_tests.py`
  - `llm_judge.py`
  - `blind_review_export.py`
  - `g_workflow_closure.py`
- 当前仍阻塞正式闭环的关键点主要是：
  - 正式 `g_plain_generation_eval_units.jsonl / g_aspect_generation_eval_units.jsonl` 尚未同步回资产目录
  - `exp02 / v2` adapter metadata 尚未补齐到本地资产目录
  - `g_workflow_closure.py` 目前仍以 helper 方式存在，尚未完全接入统一 runner

---

## 六、代码与实验适配方案

由于数据规模（40→70 查询）、评价指标（新增多项）和评价方法（新增 LLM Judge + 人工盲评）的调整，现有实验代码和结果不能直接沿用。本节对每项现有产物进行影响评估，明确"可复用 / 需后处理 / 需重跑"的分类。

### 6.1 现有实验结果适用性评估

| 实验 | 结果状态 | 查询规模 | 适用性判定 | 处理方式 |
| --- | --- | --- | --- | --- |
| E1 方面分类 | 已完成 | 344 标注句（与查询无关） | **直接复用** | 无需改动 |
| E2 候选筛选 | 已完成 | 独立评测框架 | **直接复用** | 无需改动 |
| E3 偏好解析 | 已完成 | 86 条查询（已全覆盖） | **直接复用** | 无需改动 |
| E4 澄清决策 | 已完成 | 86 条查询（已全覆盖） | **直接复用** | 无需改动 |
| E5 查询桥接 | 已完成 | 40 条查询 + 80 评测单元 | **作为 Chapter 5 辅助证据保留** | 不扩展；主对比由 G1-G4 承担 |
| E6 Aspect vs Plain | 已完成 | 40 条查询 + 80 评测单元 | **作为 Chapter 5 辅助证据保留** | 不扩展；检索层对比已充分 |
| E7 Reranker 消融 | 已完成 | 40 条查询 + 80 评测单元 | **作为配置消融保留** | 不扩展 |
| E8 Fallback 消融 | 已完成 | 40 条查询 + 80 评测单元 | **作为配置消融保留** | 不扩展 |
| E9 RAG 消融 (B vs D) | 已完成 | 40 条查询 | **作为 Chapter 5 辅助证据保留** | 不扩展；端到端对比由 G2 vs G1 承担 |
| E10 v1/v2/v3/v4 | 已完成 | 40 条查询 | **作为 Chapter 6 消融链保留** | 不扩展；PEFT 主效应由 G3 vs G1 承担 |
| **G1-G4** | **未完成** | **当前正式 decisive scope：68 条查询** | **必须新跑** | 核心实验，全部使用新指标体系 |

**设计逻辑**：E5-E10 的已有结果作为各核心章节的"辅助/消融证据"保留（不浪费已完成的工作），G1-G4 在当前正式 decisive scope（68 条查询）+ 完整指标体系上运行，作为第七章统一对比的"决定性证据"。这样既保留了已有工作的价值，又确保核心对比（G1-G4）使用最严格且已通过 formal asset validation 的评估标准；`q021 / q024` 不计入 decisive matrix，但保留为 supporting boundary cases。

#### 6.1.1 按实验划分的代码优化调整清单

下表用于回答一个更具体的问题：**在当前工作区与新版论文规划并存的前提下，哪些实验需要真的改代码，哪些实验只保留结果不再继续改实现。**

| 实验 / 实验组 | 当前工作区状态 | 是否需要调整实验实现 | 是否需要同步新指标口径 | 调整级别 | 需要调整的核心内容 | 处理结论 |
| --- | --- | --- | --- | --- | --- | --- |
| E1 方面分类 | 已完成，且与查询规模扩展无直接耦合 | 否 | 否 | 无 | 保持 `macro-F1 / Jaccard / sentiment F1` 这一任务原生指标，不引入检索层/生成层新指标 | 本轮正式重跑 |
| E2 候选筛选 | 已完成，且不依赖 E9/E10 的查询规模 | 否 | 否 | 无 | 保持候选筛选阶段原生指标，不强行套用检索句级或生成层指标 | 本轮正式重跑 |
| E3 偏好解析 | 已完成，86 条查询已覆盖完整 judged query 集 | 否 | 否 | 无 | 保持 `Exact-Match / Slot-F1 / Unsupported Detection Recall` 口径 | 本轮正式重跑 |
| E4 澄清决策 | 已完成，86 条查询已覆盖完整 judged query 集 | 否 | 否 | 无 | 保持 `Accuracy / Precision / Recall / F1 / Over-clarification / Under-clarification` 口径 | 本轮正式重跑 |
| E5 查询桥接 | 已完成，40 条查询 | 否 | 是 | 报告层同步 | 作为检索层实验，应补齐到新版 `6` 项检索指标口径与统一统计呈现方式 | 保留现有结果；必要时后处理补指标 |
| E6 Aspect vs Plain 检索 | 已完成，40 条查询 + 80 检索单元 | 否 | 是 | 报告层同步 | 作为检索层核心正结果，应统一输出新版 `6` 项检索指标与统计检验 | 保留现有结果；必要时后处理补指标 |
| E7 Reranker 消融 | 已完成 | 否 | 是 | 报告层同步 | 作为检索层消融，应同步到新版 `6` 项检索指标，保证与 E6/G1-G4 可横向比较 | 保留现有结果；必要时后处理补指标 |
| E8 Fallback 消融 | 已完成 | 否 | 是 | 报告层同步 | 作为检索层消融，应同步到新版 `6` 项检索指标，保证与 E6/G1-G4 可横向比较 | 保留现有结果；必要时后处理补指标 |
| E9 RAG 消融（B vs D） | 已完成，且当前代码已支持 `D_no_evidence_generation` | 否 | 是 | 局部优化 | 共享生成评测层需补 `70` 条查询加载、统一 `G` 系列报告字段、`Aspect Alignment Rate`、`Hallucination Rate`、Judge/人工评测衔接字段 | 保留现有 `n=40` 结果；代码按 G1/G2 共用需求升级 |
| E10 v1/v2/v3/v4 迭代链 | 已完成，`v2/v3/v4` 结果已同步回本地 | 否 | 是 | 局部优化 | compare 汇总逻辑需扩展为 `G3/G4` 共用底座，支持新版生成层指标与统一矩阵输出 | 本轮正式重跑 |
| G1 | 检索资产冻结与 generation 运行入口已实现，但正式 `70` 条 run 未产出 | 是 | 是 | 新增实现已基本落地 | `Plain Retrieval + Base` 的资产冻结、运行入口和生成侧汇总已接通；当前缺正式资产与正式云端 run | 必须新跑 |
| G2 | `Aspect + Base` 的 `40` 条辅助结果已存在，`70` 条统一框架代码已接通 | 是 | 是 | 中等扩展已基本落地 | Aspect 路线已具备扩展到 `70` 条的资产冻结和运行入口；当前缺正式复跑结果 | 必须复跑 |
| G3 | `Plain Retrieval + PEFT exp02` 的代码路径已实现，但受 `exp02` metadata 缺失阻塞 | 是 | 是 | 新增实现已基本落地 | `Plain Retrieval + PEFT exp02` 的运行入口、compare 与后处理已接通；当前缺 `v2/exp02` metadata 与正式 run | 必须新跑 |
| G4 | `Aspect + PEFT exp02` 的 `40` 条辅助结果已存在，`70` 条统一框架代码已接通 | 是 | 是 | 中等扩展已基本落地 | Aspect + PEFT 路线已具备统一框架入口；当前缺正式 `70` 条复跑与 `exp02` metadata | 必须复跑 |
| 统计检验 | 已实现 | 是 | 是 | 新增实现已完成 | `statistical_tests.py` 已支持 Wilcoxon / Bootstrap CI / Cohen's d / rank-biserial / 显著性标注 / 多重比较校正 | 待接正式 G1-G4 结果 |
| LLM-as-Judge | 已实现 | 是 | 是 | 新增实现已完成 | `llm_judge.py` 已支持 blind prompt、单组批量评分与 group-level 聚合 | 待真实 API 运行 |
| 人工盲评抽样导出 | 已实现，且已开始补 blind review 汇总 | 是 | 是 | 轻量新增已完成 | `blind_review_export.py` 已可导出匿名 blind pack，`g_workflow_closure.py` 已补 blind review 结果聚合 | 待正式人工标注 |
| G1-G4 工作流收口 | 已开始实现 | 是 | 是 | 新增实现进行中 | `g_workflow_closure.py` 已包含 `exp02` metadata 校验、score map 提取、批量 Judge、blind review 聚合和章节报告生成；当前仍待统一 runner 接线 | 需继续收口 |

**据此形成的总清单如下：**

- `[A] 继续使用任务原生指标，但本轮仍正式重跑`：`E1 / E2 / E3 / E4`
- `[B] 不改实验主体逻辑，但需要同步新版指标口径与报告模板`：`E5 / E6 / E7 / E8 / E9 / E10 v1 / v2 / v3 / v4`
- `[C] 必须新增或重构代码并重新运行`：`G1 / G2 / G3 / G4`
- `[D] 已完成代码实现但待正式运行/待正式接线`：`统计检验 / LLM Judge / 人工盲评导出 / G 工作流收口`

**当前代码改造的优先级应固定为：**

1. 先补齐 `exp02 / v2` metadata，并正式同步 `G` 系列 retrieval assets。
2. 再正式运行 `G1 / G2 / G3 / G4` 四组云端实验。
3. 然后把 `g_workflow_closure.py` 这层收口任务真正跑完：统计检验、Judge、盲评与章节总报告。
4. 最后再进入一轮针对结果薄弱项的定向优化，而不是继续扩展旧实验链。

### 6.2 查询集扩展的实施细节

#### 6.2.1 当前查询集分层

```
完整查询集（86 条，judged_queries.jsonl）
├── 核心层（40 条）—— 已有 qrels + E9 eval units
│   ├── single_aspect × 10 城市
│   ├── multi_aspect × 10 城市
│   ├── focus_and_avoid × 10 城市
│   └── multi_aspect_strong × 10 城市
├── 鲁棒性层（30 条）—— 需生成正式 eval units，不需 qrels
│   ├── unsupported_budget × 10 城市
│   ├── unsupported_distance × 10 城市
│   └── unsupported_heavy × 10 城市
└── 对话管理层（16 条）—— 不纳入 G1-G4
    ├── conflict × 10 城市
    └── missing_city × 6
```

#### 6.2.2 鲁棒性层的特殊处理

30 条鲁棒性层查询包含 unsupported 约束（预算、距离、入住日期），这些查询的评测重点不在于检索质量（因为它们也有 focus_aspects，检索仍可执行），而在于：

- 系统是否正确识别了不支持约束并诚实告知（→ Unsupported Honesty Rate）
- 推荐在存在不支持约束时是否仍然给出了合理的基于已支持约束的结果（→ Recommendation Coverage）

因此鲁棒性层**不需要新建 qrels**（qrels 用于检索排序质量评估，鲁棒性层的主要评价维度是生成层指标）。只需新建 30 条 EvidencePack（对 G1 用 Plain 检索，对 G2 用 Aspect 检索）。

#### 6.2.3 扩展后的评测查询集文件

当前 `experiments/assets/g_eval_query_ids_68.json` 是正式 decisive scope 资产，当前正式内容为 `68` 条 query_id（`39 core + 29 robustness`）；其中 `q021 / q024` 已在 asset 中通过 `excluded_query_ids` 显式记录为 supporting boundary cases。

### 6.3 新增指标的代码实现方案

#### 6.3.1 检索层新增指标

| 指标 | 代码现状 | 实现方案 |
| --- | --- | --- |
| MRR@5 | **已有**（evaluate_e6_e8_retrieval.py `mrr_at_5`） | 直接复用 |
| Evidence Diversity@5 | **已有**（`evidence_diversity_at_5`） | 直接复用 |
| Retrieval Latency | **已有**（各 run 的 `avg_latency_ms`） | 直接复用 |

检索层的 6 项指标中，3 项已实现（MRR、Diversity、Latency），3 项已有（Recall、nDCG、Precision）。**无需新增代码**。

#### 6.3.2 生成层新增指标

| 指标 | 代码现状 | 实现方案 | 实现难度 |
| --- | --- | --- | --- |
| Aspect Alignment Rate | **已实现** | 生成侧已支持从 `response + eval_unit` 计算 focus-aspect 命中率，并进入 E9/E10/G compare summary | 已完成 |
| Hallucination Rate | **已实现** | 生成侧已支持从 `audit_rows` 统计 citation 不存在或 support 为 0 的比例 | 已完成 |
| Unsupported Honesty Rate | **已有**（在 E9 metric row 中） | 直接复用 | 无 |
| Recommendation Coverage | **已有** | 直接复用 | 无 |

生成层 7 项指标当前已经全部在共享生成评测底座中落地，重点已经从“补指标”切换到“正式运行 G1-G4 并把结果汇总进章节报告”。

#### 6.3.3 统计检验代码

当前已实现 `scripts/evaluation/statistical_tests.py`，核心能力包括：

```
def wilcoxon_signed_rank(group_a_scores, group_b_scores) -> dict:
    """配对 Wilcoxon 符号秩检验"""

def bootstrap_ci(scores, n_resamples=1000, ci=0.95) -> tuple:
    """Bootstrap 置信区间"""

def cohens_d(group_a_scores, group_b_scores) -> float:
    """效应量 Cohen's d"""

def compute_pairwise_tests(g1, g2, g3, g4, metrics) -> pd.DataFrame:
    """G1-G4 所有配对的统计检验汇总"""
```

当前版本已额外补齐：

- `query_id` 对齐配对
- `significance` / `significance_adj` 标注
- `Holm / Bonferroni` 多重比较校正
- `rank_biserial` 非参数效应量
- `better_group / higher_is_better` 结果解释字段

因此统计检验模块当前不再是“待实现项”，而是“待接正式 G1-G4 结果并实际运行”的状态。

#### 6.3.4 LLM-as-Judge 代码

当前已实现 `scripts/evaluation/llm_judge.py`，核心能力包括：

```
def build_judge_prompt(query_text, recommendation_response) -> str:
    """构建 DeepSeek 评审 Prompt（盲评，不含组别标签）"""

def score_single_response(query, response, api_client) -> dict:
    """对单条回复评分，返回 5 维度 1-5 分"""

def run_llm_judge(results_dir, output_path, model="gpt-4o") -> pd.DataFrame:
    """批量评审一个 run 的全部结果"""

def aggregate_judge_scores(g1_scores, g2_scores, g3_scores, g4_scores) -> pd.DataFrame:
    """汇总四组 LLM Judge 分数"""
```

另外，当前还已实现：

- `blind_review_export.py`
- `g_workflow_closure.py` 中的批量 Judge、blind review 聚合和章节报告 helper

因此 Judge 侧当前也不是“待从零实现”，而是“代码已基本具备，待真实 API 批量执行”的状态。

#### 6.3.5 G 工作流收口代码

当前工作区已新增 `scripts/evaluation/g_workflow_closure.py`，用于承接之前规划里分散的 6 个收口任务，当前已包含：

- `extract_g_group_score_map`
- `run_g_batch_llm_judge`
- `aggregate_blind_review_results`
- `validate_exp02_metadata`
- `ensure_exp02_metadata_placeholder`
- `build_g_chapter_report`

当前缺口不再是“没有收口代码”，而是：

- 该文件尚未完全接入统一 runner
- 其输出还没有和正式 `G1-G4` 云端产物形成一轮完整闭环

### 6.4 G1-G4 实验的完整实施路径

```
Phase 1: 代码准备（本地）
│
├── 1.1 扩展查询集
│   ├── `g_eval_query_ids_68.json` 已生成（39 核心 + 29 鲁棒性，`q021/q024` 已排除）
│   └── 共享加载与校验逻辑已支持当前正式 decisive scope（68 条查询）
│
├── 1.2 实现 Plain RAG 基线
│   ├── 新增 build_evidence_pack_plain_for_candidate（~30 行）
│   ├── 在 run_experiment_suite.py 注册 g_plain_rag_freeze_assets task
│   └── 测试：本地 dry-run 确认 EvidencePack 格式正确
│
├── 1.3 补齐生成层指标
│   ├── Aspect Alignment Rate（已完成）
│   └── Hallucination Rate（已完成）
│
├── 1.4 统计检验模块（已完成）
│
└── 1.5 LLM-as-Judge 模块（已完成）

Phase 2: 实验运行（AutoDL 云端）
│
├── 2.1 冻结资产
│   ├── 为当前正式 decisive scope（68 条查询）生成 Aspect-Guided EvidencePacks
│   └── 为当前正式 decisive scope（68 条查询）生成 Plain EvidencePacks
│
├── 2.2 运行四组实验
│   ├── G1: Plain RAG + Base Qwen3.5-4B（68 条）
│   ├── G2: Aspect RAG + Base Qwen3.5-4B（68 条）
│   ├── G3: Plain RAG + PEFT exp02（68 条）
│   └── G4: Aspect RAG + PEFT exp02（68 条）
│
└── 2.3 生成引用审计
└── 4 组 × 68 条的 citation_verifiability_audit.csv

Phase 3: 评估与分析（本地 + API）
│
├── 3.1 计算硬指标（检索层 6 项 + 生成层 7 项）
├── 3.2 运行统计检验（Wilcoxon + Bootstrap CI + Cohen's d）
├── 3.3 运行 LLM-as-Judge（DeepSeek，4 × 70 = 280 条评审）
├── 3.4 组织人工盲评（4 × 15-20 条抽样）
└── 3.5 生成 G1-G4 统一对比报告
```

截至 `2026-04-05`，这条路径的当前进度可概括为：

- `Phase 1`：已基本完成
- `Phase 2`：代码入口已具备，但四组正式 run 尚未产出
- `Phase 3`：模块已基本具备，但正式统计/Judge/盲评/章节报告尚未基于真实四组结果跑完

### 6.5 已有 E5-E10 结果的论文使用规范

为避免"已有 40 条结果"和"当前正式 decisive scope 结果"产生口径混淆，论文中需遵循以下规范：

| 论文位置 | 使用的数据 | 查询规模标注 |
| --- | --- | --- |
| 第五章 5.2-5.3 节（E5/E6/E7/E8） | 已有 40 条结果 | 表格标注 "n=40 / 80 units" |
| 第五章 5.4 节（E9 消融） | 已有 40 条结果 | 表格标注 "n=40" |
| 第六章 6.3-6.4 节（E10 迭代链） | 已有 40 条结果 | 表格标注 "n=40" |
| **第七章 7.2 节（G1-G4 对比）** | **新跑当前正式 decisive scope 结果** | **表格标注 "n=68 (39 core + 29 robustness)"，并脚注说明 `q021 / q024` 为 supporting boundary cases** |
| **第七章 7.2 节（统计检验）** | **新跑当前正式 decisive scope 结果** | **标注 p 值和 CI** |
| **第七章 7.2 节（LLM Judge）** | **新跑当前正式 decisive scope 结果** | **标注 5 维度均分** |

在第七章开头需明确说明：第五/六章的辅助实验使用 40 条核心查询；第七章的统一对比实验原设计为 70 条查询，但当前正式 decisive execution 因 frozen aspect mainline 的 evidence-backed candidate 可行性约束，对 `q021 / q024` 做了对称剔除，因此正式 decisive matrix 使用 `68` 条查询。这不是方法漂移，而是一次 pre-run、对称、基于 formal asset validation 的协议级修正；`q021 / q024` 仍作为 supporting boundary cases 保留在讨论与误差分析中。

### 6.6 各项代码改动的工作量汇总

| 改动项 | 当前状态 | 主要文件 | 当前结论 |
| --- | --- | --- | --- |
| 查询集扩展 | 已完成 | `evaluate_e6_e8_retrieval.py` + `g_eval_query_ids_68.json` | 已可供 G 系列共用 |
| Plain / Aspect Retrieval Assets | 已完成入口，待正式落地资产 | `evaluate_e6_e8_retrieval.py` | 正式 `g_plain/g_aspect` assets 仍待同步 |
| G 系列 generation 入口 | 已完成 | `evaluate_e9_e10_generation.py` + `run_experiment_suite.py` | 可运行，待正式云端结果 |
| 生成层 7 指标 | 已完成 | `evaluate_e9_e10_generation.py` | 已进入 E9/E10/G 共用底座 |
| 统计检验模块 | 已完成 | `statistical_tests.py` | 待接正式 G1-G4 结果 |
| LLM-as-Judge 模块 | 已完成 | `llm_judge.py` | 待真实 API 执行 |
| 人工盲评导出 | 已完成 | `blind_review_export.py` | 待正式人工标注 |
| G 工作流收口 / 章节报告 | 进行中 | `g_workflow_closure.py` | helper 已具备，待统一接线与正式运行 |

---

## 七、设计原则（保留）

1. **知识外置**：酒店事实由 PostgreSQL 和 ChromaDB 提供，不混入模型参数。
2. **行为内化**：LLM 负责偏好解析、澄清、证据组织与输出表达。
3. **证据优先**：推荐理由必须回溯到真实 sentence_id。
4. **边界诚实**：不支持的约束显式告知。
5. **实验可复现**：关键配置冻结，所有 run 保留可追溯。

---

## 八、关键文件路径索引

| 用途 | 路径 |
| --- | --- |
| 全局配置 | configs/params.yaml |
| 数据库 Schema | sql/init_schema.sql |
| 检索评测（含 Plain RAG） | scripts/evaluation/evaluate_e6_e8_retrieval.py |
| 生成评测 | scripts/evaluation/evaluate_e9_e10_generation.py |
| 实验调度器 | scripts/evaluation/run_experiment_suite.py |
| PEFT 训练 | scripts/training/train_e10_peft.py |
| 冻结查询集 | experiments/assets/e9_generation_eval_query_ids.json |
| 完整查询集（86 条） | experiments/assets/judged_queries.jsonl |
| 酒店切分 | experiments/assets/frozen_split_manifest.json |
| E9 RAG 消融 run | experiments/runs/e9_8449c12a50585e42_20260404T081010+0000/ |
| E10 base formal run | experiments/runs/e10_0dc5c2e6f867c66f_20260402T015230+0000/ |
| E10 PEFT v2 run | experiments/runs/e10_a2dd1a0bd73c57b5_20260402T073127+0000/ |
| E10 PEFT v3 run | experiments/runs/e10_927c1d0a2fbc5870_20260403T083734+0000/ |
| E10 PEFT v4 run | experiments/runs/e10_d749ec0796f7b365_20260404T024800+0000/ |
| 论文章节 | chapters/ |
| 论文管理 | plan/ |

---

## 九、当前实现中的真实数据流分层说明

为避免论文叙事与工程实现脱节，当前项目必须明确区分“知识库构建层 / 冻结资产层 / 正式评测层”三层，而不能笼统写成“所有实验都在线访问 PostgreSQL 与 ChromaDB”。

### 9.1 知识库构建层

- 原始酒店评论首先经过清洗、切句、方面标注与画像聚合，生成 `data/intermediate/*.pkl` 中间产物。
- 在这一层中：
  - PostgreSQL 承担关系型知识库存储，保存酒店、评论、句子、方面画像和证据索引等结构化事实。
  - ChromaDB 承担句子级向量索引存储，保存可检索的证据向量及其元数据过滤条件。
- 关键脚本包括：
  - `scripts/pipeline/build_evidence_vector_index.py`
  - `scripts/pipeline/load_kb_to_postgres.py`
  - `scripts/pipeline/validate_kb_assets.py`

### 9.2 冻结资产层

- 为保证控制变量、可复现性与正式 compare 的严谨性，实验并不总是直接在线查询知识库，而是先从知识库与中间产物中冻结实验资产。
- 当前冻结资产层的典型产物包括：
  - query scopes
  - split manifests
  - qrels / gold labels
  - `GenerationEvalUnit`
  - `EvidencePack`
  - PEFT manifests
- `E2 / E5-E8` 的检索实验仍会直接访问 ChromaDB，但同时也显式依赖 `data/intermediate/*.pkl`。
- `E9 / E10 / G1-G4` 更典型的实现路径是：先利用 ChromaDB 构造 `EvidencePack / GenerationEvalUnit / manifest`，再将这些资产写入 `experiments/assets/*.json / *.jsonl`，供后续正式评测复用。

### 9.3 正式评测层

- 一旦冻结资产生成完成，后续正式实验主线主要消费的是静态实验资产与运行产物，而不是持续在线访问知识库。
- 当前正式评测层主要消费：
  - `results.jsonl`
  - `summary.csv`
  - `audit csv`
  - frozen eval assets / manifests
- 因此在正式实验热路径中：
  - PostgreSQL 几乎不直接参与；
  - ChromaDB 的直接参与主要集中在上游资产构建，而不是下游 compare / stats / Judge / blind review / chapter report。

### 9.4 对论文表述的直接约束

- 论文中的准确表述应为：
  - “本研究首先构建由 PostgreSQL 与 ChromaDB 组成的外置知识库；在正式实验阶段，为保证控制变量与可复现性，进一步将检索结果、证据包和评测单元冻结为静态实验资产，后续生成评测与组间比较均基于这些冻结资产完成。”
- 不应表述为：
  - “所有正式实验均直接在线访问 PostgreSQL 与 ChromaDB 运行。”

### 9.5 E1 路径口径冻结

- `E1` 的唯一正式 gold 路径固定为：
  - `experiments/labels/e1_aspect_reliability/aspect_sentiment_gold.csv`
- 历史上的 `experiments/E1/` 只属于旧副本位置，不再作为正式输入口径引用。
