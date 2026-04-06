# Experiments Workspace

本目录现在已经切换到“论文正式全量重跑前”的干净状态。

当前约定是：

- 旧的 `E1-E10`、`G1-G4` 结果已经整体归档到仓库外。
- 仓库内只保留正式重跑所需的协议文档、冻结输入资产、标注资产、配置模板和空的 `runs/` 目录。
- 后续所有新的正式结果，都应重新写入 `experiments/runs/`，并登记到 `experiments/assets/final_rerun_registry.json`。

## 当前目录结构

- `assets/`
  正式重跑所需的冻结输入、query scopes、gold/qrels 入口、manifest 占位文件、最终重跑基线与登记表。
- `labels/`
  正式评测仍需复用的人类标注资产，例如 `E1` gold、`E6` qrels、`G` qrels 等。
- `reports/`
  当前只保留本轮正式重跑的协议与审计文档。
- `runs/`
  当前应保持为空目录，直到本轮正式重跑真正开始。

## 当前唯一有效的正式文档

- `reports/00_archived_results_notice.md`
- `reports/11_final_experiment_audit_protocol.md`
- `reports/12_experiment_audit_master_table.md`
- `reports/13_experiment_control_and_metric_contracts.md`
- `reports/14_final_rerun_execution_protocol.md`

其中：

- `11` 负责说明“重跑前要审计什么”
- `12` 负责给出 `E1-E10` 与 `G1-G4` 的总览索引
- `13` 负责冻结控制变量与指标口径
- `14` 是当前唯一正式执行协议

## 当前保留的关键资产

以下文件不属于“旧结果”，而是正式重跑输入，当前必须保留：

- `assets/frozen_split_manifest.json`
- `assets/judged_queries.jsonl`
- `assets/slot_gold.jsonl`
- `assets/clarify_gold.jsonl`
- `assets/annotation_rubrics.md`
- `assets/e5_e8_core_query_ids.json`
- `assets/e9_generation_eval_query_ids.json`
- `assets/g_eval_query_ids_68.json`
- `assets/final_rerun_baseline.json`
- `assets/final_rerun_registry.json`
- `assets/g_run_dirs.json`
- `assets/g_closure_manifest.json`
- `labels/e1_aspect_reliability/*`
- `labels/e6_qrels/*`
- `labels/g_qrels/*`

## 三层真实数据流

### 1. 知识库构建层

这一层负责把原始酒店评论数据转成可检索、可校验的知识底座：

- PostgreSQL：关系型知识库
- ChromaDB：句子级向量索引
- `data/intermediate/*.pkl`：中间快照

关键脚本：

- `scripts/pipeline/build_evidence_vector_index.py`
- `scripts/pipeline/load_kb_to_postgres.py`
- `scripts/pipeline/validate_kb_assets.py`

### 2. 冻结资产层

这一层负责把实验真正要用的输入冻结出来，避免正式比较时继续受数据库状态漂移影响。

典型资产包括：

- query id scopes
- split manifests
- gold / qrels
- `EvidencePack`
- `GenerationEvalUnit`
- `SFT manifests`

### 3. 正式评测层

这一层负责正式运行、统计检验、Judge、盲评和章节报告生成。

一旦冻结资产准备完成，正式实验主线主要消费：

- `results.jsonl`
- `summary.csv`
- `analysis.md`
- `audit csv`
- 各类 `json/jsonl` 冻结资产

因此，论文中的准确表述应是：

> 本研究首先构建由 PostgreSQL 与 ChromaDB 组成的外置知识库；在正式实验阶段，为保证控制变量与可复现性，进一步将检索结果、证据包和评测单元冻结为静态实验资产，后续生成评测与组间比较均基于这些冻结资产完成。

而不应写成：

> 所有正式实验均直接在线访问 PostgreSQL 与 ChromaDB 运行。

## E1 路径口径

`E1` 的唯一正式 gold 路径固定为：

- `experiments/labels/e1_aspect_reliability/aspect_sentiment_gold.csv`

历史上的 `experiments/E1/` 旧副本目录已废弃，不再作为正式输入口径。

## 当前工作区状态

当前工作区已经完成：

- 旧结果归档
- 正式重跑基线冻结
- 协议文档切换到 `11-14`
- `runs/` 清空
- `g_run_dirs.json` 和 `g_closure_manifest.json` 重置为占位状态

这意味着：

- 你现在看到的 `experiments/` 是“准备正式重跑”的起点
- 不是上一轮调试结果的延续目录

## 本轮正式重跑顺序

### Phase A

- `E1`
- `E2`
- `E3`
- `E4`

### Phase B

- `E5`
- `E6`
- `E7`
- `E8`

### Phase C

- `E9`
- `E10 base`
- `E10 v1`
- `E10 v2`
- `E10 v3`
- `E10 v4`

### Phase D

- `G` query / retrieval assets
- `G qrels`
- `G retrieval formal eval`
- `G1-G4`
- `pairwise tests`
- `LLM Judge`
- `blind review`
- `gchapter`

## 结果登记规则

本轮正式重跑完成后，所有 canonical 结果都应同步登记到：

- `assets/final_rerun_registry.json`

每条 canonical 结果至少应记录：

- 实验编号
- run 目录
- summary 路径
- report 路径
- query scope
- metric contract version
- 论文用途

## 归档说明

上一轮混合调试期结果已移至仓库外归档根目录，具体位置写在：

- `assets/final_rerun_baseline.json`

不要再把旧调试结果手工搬回 `experiments/runs/`。
