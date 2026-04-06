# 最终正式重跑执行协议（E1-E10, G1-G4）

更新时间：2026-04-06

本文件是当前论文结果生产阶段唯一有效的正式执行协议。

## 1. 当前轮次的正式决策

本轮执行不再复用仓库内任何历史 run 作为论文正式结果。

正式政策固定为：

- `E1-E10` 全部正式重跑
- `G1-G4` 全部正式重跑
- 旧结果仅保留在仓库外归档目录中
- 正式结果只认本轮新生成的 canonical run

对应基线与登记文件：

- `experiments/assets/final_rerun_baseline.json`
- `experiments/assets/final_rerun_registry.json`

## 2. 本轮的唯一上位文档

执行时必须同时服从以下文档：

- `experiments/reports/11_final_experiment_audit_protocol.md`
- `experiments/reports/12_experiment_audit_master_table.md`
- `experiments/reports/13_experiment_control_and_metric_contracts.md`
- `experiments/reports/14_final_rerun_execution_protocol.md`

约定关系如下：

- `11` 负责审计维度
- `12` 负责实验总表与角色定位
- `13` 负责控制变量与指标口径冻结
- `14` 负责执行顺序与结果登记

## 3. 正式重跑前必须满足的条件

### 3.1 冻结输入必须存在

至少应确认以下资产存在且可读：

- `experiments/assets/frozen_split_manifest.json`
- `experiments/assets/judged_queries.jsonl`
- `experiments/assets/slot_gold.jsonl`
- `experiments/assets/clarify_gold.jsonl`
- `experiments/assets/e5_e8_core_query_ids.json`
- `experiments/assets/e9_generation_eval_query_ids.json`
- `experiments/assets/g_eval_query_ids_68.json`
- `experiments/labels/e1_aspect_reliability/aspect_sentiment_gold.csv`
- `experiments/labels/e6_qrels/qrels_evidence.jsonl`
- `experiments/labels/g_qrels/g_qrels_evidence.jsonl`

### 3.2 代码基线必须固定

在第一条正式 run 之前，必须记录：

- 当前 `HEAD SHA`
- `requirements.txt` 指纹
- `.env.example` 指纹
- 本轮 archive 根目录

这些信息已经登记在：

- `experiments/assets/final_rerun_baseline.json`

### 3.3 工作区结果目录必须保持干净

开始正式重跑前，`experiments/runs/` 不应保留旧结果目录。

允许存在的仅有：

- `.gitkeep`
- 本轮刚开始后新生成的 run 目录

## 4. 三层数据流约束

正式执行必须遵守以下三层边界：

### 4.1 知识库构建层

由 PostgreSQL、ChromaDB 和 `data/intermediate/*.pkl` 组成。

这一层负责：

- 构建结构化知识
- 构建句子级向量索引
- 验证 KB 数据契约

### 4.2 冻结资产层

这一层负责：

- 冻结 query scopes
- 冻结 gold / qrels
- 冻结 `EvidencePack`
- 冻结 `GenerationEvalUnit`
- 冻结 SFT manifests

### 4.3 正式评测层

这一层负责：

- retrieval eval
- generation eval
- compare
- pairwise tests
- LLM Judge
- blind review
- chapter closure

正式评测默认消费冻结资产，而不是把 PostgreSQL 和 ChromaDB 当作每一步都在线读取的热路径。

## 5. 正式重跑顺序

## Phase A：前置验证层

目标：

- 重跑 `E1-E4`
- 建立 Chapter 4 的正式结果

包含：

- `E1`
- `E2`
- `E3`
- `E4`

每个实验完成后，必须产出：

- `run_meta.json`
- `summary.csv`
- `analysis.md`

## Phase B：RAG supporting evidence

目标：

- 重跑 `E5-E8`
- 固定 retrieval 层 6 指标口径

包含：

- `E5`
- `E6`
- `E7`
- `E8`

## Phase C：生成 supporting evidence

目标：

- 重跑 `E9`
- 重跑 `E10 base`
- 重跑 `E10 v1-v4`

包含：

- `E9`
- `E10_base`
- `E10_v1`
- `E10_v2`
- `E10_v3`
- `E10_v4`

说明：

- `exp02` 继续作为后续 `G3/G4` 的 canonical PEFT 候选
- 但本轮仍需把 `E10 v1-v4` 全链正式补齐

## Phase D：Chapter 7 统一矩阵

目标：

- 重跑 `G1-G4`
- 产出 retrieval / generation / stats / judge / blind review / gchapter 全闭环正式结果

包含：

- `G` query assets
- `G` retrieval assets
- `G qrels`
- `G retrieval formal eval`
- `G1`
- `G2`
- `G3`
- `G4`
- `G pairwise stats`
- `G LLM Judge`
- `G blind review`
- `G chapter closure`

## 6. 每个阶段完成后的最低检查

每个 canonical run 至少应具备：

- `run_meta.json`
- `summary.csv`
- `analysis.md`

如果实验属于 generation / compare / judge / closure 类型，还应按需要具备：

- `results.jsonl`
- `citation_verifiability_audit.csv`
- `comparison.jsonl`
- `pairwise_tests.csv`
- `judge_summary.csv`

## 7. 结果登记规则

本轮所有正式 canonical 结果都必须登记到：

- `experiments/assets/final_rerun_registry.json`

登记字段至少包括：

- 实验编号
- canonical run 目录
- summary 路径
- analysis/report 路径
- query scope
- metric contract version
- 论文用途

未登记到 registry 的结果，不应直接写入论文。

## 8. Manifest 与运行映射规则

以下文件在正式重跑开始前应保持占位状态，不应写入旧路径：

- `experiments/assets/g_run_dirs.json`
- `experiments/assets/g_closure_manifest.json`

只有在本轮新的 `G1-G4` 与 `G` 闭环结果全部跑出后，才允许回填：

- 新的 `G1-G4` run 目录
- 新的 retrieval formal run 目录
- 新的 pairwise tests 路径
- 新的 judge summary 路径
- 新的 blind review summary 路径

## 9. 写作约束

本轮论文写作时，只允许引用：

- 本轮新 canonical run
- `experiments/assets/final_rerun_registry.json` 中登记过的结果

不得再把仓库外归档结果直接当作正式引用对象。

## 10. Human Blind Review 约束

Human Blind Review 必须与 LLM Judge 分开叙述。

如果某一轮 blind review 由模型自动填充、研究者复核或内部校对得到，则必须显式标明其性质，不能误写为“完全独立外部人类评审”。

只有满足独立真人复审要求的 blind review 结果，才可在论文中作为正式 human evaluation 强结论使用。

## 11. Stop / Go 规则

### Stop

出现以下任一情况时，不应继续推进下一阶段：

1. query scope 不清楚
2. gold / qrels 未冻结
3. split 口径不清楚
4. formal retrieval summary 缺失
5. generation 指标在 summary / stats / chapter 中口径不一致
6. 新结果尚未登记到 `final_rerun_registry.json`

### Go

只有当以下条件同时满足，才可进入下一阶段：

1. 输入资产已冻结
2. 关键控制变量已对齐
3. 指标口径已与 `13_experiment_control_and_metric_contracts.md` 对齐
4. 当前阶段产物已写入新的 run 目录
5. 当前阶段 canonical 结果已登记到 registry

## 12. 最终原则

本轮重跑的目标不是“尽快把所有脚本再跑一遍”，而是：

> 在统一协议、统一资产、统一口径、统一登记规则下，重新生成一套可以直接进入论文写作的正式实验结果。
