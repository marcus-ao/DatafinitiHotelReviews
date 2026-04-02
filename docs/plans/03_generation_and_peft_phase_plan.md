# 证据约束生成与 PEFT 阶段推进方案

更新时间：2026-04-02

本文件用于冻结行为章节收口后的后续主线。当前顺序固定为：`E9` 第二轮正式结果冻结后，完成 `E10 v1` Base vs PEFT 正式对照，再进入 `E10 v2` 数据方案。`Qwen/Qwen3.5-4B` 是当前默认基座模型；`Qwen/Qwen3.5-9B` 只作为可选附录上界，不进入默认训练与实现主线。

## 1. 当前阶段判断

目前已经成立的前提：

- `E1 / E2 / E5 / E6 / E7 / E8` 已完成正式结果
- `E3 / E4` 已完成 `Qwen/Qwen3.5-4B` 全量正式结果
- 默认检索后端已冻结为 `aspect_main_no_rerank`
- `fallback_enabled = false`
- 行为层当前正式主模型已冻结为 `Qwen/Qwen3.5-4B`

因此后续主线不再是“继续救 `2B`”或“先跑满三模型横评”，而是：

1. `E9` 第二轮结果已证明当前生成层已经达到可审计、可冻结状态
2. `E10 v1` 已经用 `4B base vs 4B PEFT` 回答了 `RQ2` 的首轮问题
3. `E10 v2` 的目标固定为：只改训练数据，不改训练配方与评测协议

## 2. E9：证据约束生成

### 2.1 目标

`E9` 只回答一个问题：

- 当前生成层是否真正受当前 `EvidencePack` 约束

不在 `E9` 中引入 PEFT，不在 `E9` 中让 retrieval 波动成为自变量。

### 2.2 固定对照

- `A_free_generation`
  - 给固定候选酒店与证据文本
  - 不强制引用 `sentence_id`
- `B_grounded_generation`
  - 输入固定 `EvidencePack`
  - 要求每条推荐理由必须显式引用 `sentence_id`
- `C_grounded_generation_with_verifier`
  - 与 `B` 相同
  - 但输出后经过 Citation Verifier
  - 若引用不存在或不属于当前 `EvidencePack`，则重试一次；仍失败则降级为诚实说明

### 2.3 固定资产与实现入口

当前仓库中已新增：

- `scripts/evaluation/evaluate_e9_e10_generation.py`
- 在 `scripts/evaluation/run_experiment_suite.py` 中新增：
  - `e9_freeze_assets`
  - `e9_generation_constraints`
  - `e10_prepare_manifests`
  - `e10_base_vs_peft`

`E9` 的正式冻结资产固定为：

- `experiments/assets/e9_generation_eval_units.jsonl`
  - 官方评测集固定复用当前 `40` 条可执行 query
  - 每条记录至少包含：
    - `query_id`
    - `query_text_zh`
    - `user_preference_gold`
    - `unsupported_requests`
    - `candidate_hotels`
    - `evidence_packs`
    - `retrieval_mode`
    - `config_hash`
- `experiments/assets/e9_generation_eval_query_ids.json`
  - 与上面资产一一对应的 query id 清单

资产生成规则固定为：

- `user_preference_gold` 直接来自 `slot_gold.jsonl`
- `candidate_hotels` 先固定为 `E2 B_final_aspect_score Top5`
- `evidence_packs` 再由当前默认后端 `aspect_main_no_rerank` 生成
- 生成一次后立即冻结
- `E9` 正式评测不得边跑边重新检索

当前正式状态补充：

- 第二轮正式 run 已冻结为：
  - `experiments/runs/e9_ecbcdbab690dc503_20260401T025012+0000/`
- 第一轮 run 保留为诊断对照：
  - `experiments/runs/e9_80e05af30f45b1f2_20260401T021215+0000/`
- 当前正式结论固定为：
  - `q021 / q023` 是 evidence gap honesty
  - `q079` 是 verifier 过严残留边界
  - retrieval 主线不再继续调整

### 2.4 复用与新增的数据契约

`E9` 复用已有：

- `UserPreference`
- `HotelCandidate`
- `SentenceCandidate`
- `EvidencePack`
- `WorkflowState`

并新增：

- `RecommendationReason`
  - `aspect`
  - `reason_text`
  - `sentence_id`
- `RecommendationItem`
  - `hotel_id`
  - `hotel_name`
  - `reasons[]`
- `RecommendationResponse`
  - `query_id`
  - `group_id`
  - `summary`
  - `recommendations[]`
  - `unsupported_notice`
  - `schema_valid`
  - `raw_response`
- `CitationVerificationResult`
  - `query_id`
  - `group_id`
  - `citation_precision`
  - `invalid_sentence_ids[]`
  - `out_of_pack_sentence_ids[]`
  - `retry_triggered`
  - `fallback_to_honest_notice`

### 2.5 指标与人工审计

`E9` 正式指标固定为：

- `Citation Precision`
- `Evidence Verifiability`
- `Unsupported Honesty Rate`
- `Schema Valid Rate`
- `avg_latency_ms`

新增人工资产目录：

- `experiments/labels/e9_generation/`

至少包含：

- `citation_verifiability_audit.csv`
- `README.md`

`citation_verifiability_audit.csv` 固定字段：

- `query_id`
- `group_id`
- `hotel_id`
- `sentence_id`
- `reason_text`
- `citation_exists`
- `in_current_evidence_pack`
- `support_score`
- `notes`

评分口径固定：

- `citation_exists`
  - `0/1`
- `in_current_evidence_pack`
  - `0/1`
- `support_score`
  - `2` = 证据明确支持理由
  - `1` = 基本相关但支持偏弱
  - `0` = 证据不支持或明显越权

### 2.6 输出目录

`E9` 正式输出固定写入：

- `experiments/runs/e9_*/`

每轮必须包含：

- `run_meta.json`
- `results.jsonl`
- `summary.csv`
- `analysis.md`

## 3. E10：Base vs PEFT 行为对照

### 3.1 目标

`E10` 回答的核心问题固定为：

- 在固定输入、固定偏好、固定证据的前提下，PEFT 是否主要提升行为能力，而不是酒店知识记忆

### 3.2 固定设计

`E10` 只允许一个主要自变量：

- 模型：`Base 4B` vs `PEFT 4B`

其他条件必须全部固定：

- 同一 `query_id`
- 同一 `UserPreference`
- 同一 `EvidencePack`
- 同一输出 schema
- 同一 prompt 版本
- 同一检索配置

`E10` 官方组名固定为：

- `A_base_4b_grounded`
- `B_peft_4b_grounded`

### 3.3 默认基座模型与原因

默认 PEFT 基座固定为：

- `Qwen/Qwen3.5-4B`

原因固定写法：

- `2B` 已被正式实验验证为弱基线，不适合作为主微调基座
- `9B` 训练与推理成本过高，不适合作为当前主线默认基座
- `4B` 已经是当前正式行为主模型，也是最适合继续进入 `E10 / PEFT` 的稳定起点

### 3.4 SFT 数据边界

训练数据来源边界必须冻结为：

- 只允许来自 `train` 酒店
- `dev` 酒店只用于调参与早停
- `test` 酒店绝不进入训练样本

SFT 样本类型固定为四类：

1. `preference_parse`
2. `clarification`
3. `constraint_honesty`
4. `feedback_update`

多轮能力本阶段只允许两类：

- 单次澄清
- 单次反馈更新

不做更复杂多轮规划，不引入开放式 Agent 行为。

### 3.5 训练与评测产物

建议新增受版本管理的训练清单：

- `experiments/assets/sft_train_manifest.jsonl`
- `experiments/assets/sft_dev_manifest.jsonl`

当前实现补充：

- 这两份 manifest 已可由 `e10_prepare_manifests` 自动生成
- `v1` manifest 只包含四类行为样本：
  - `preference_parse`
  - `clarification`
  - `constraint_honesty`
  - `feedback_update`
- `v2` 将新增：
  - `grounded_recommendation`
- `E10 v1` 正式 compare 已完成并冻结为：
  - `experiments/runs/e10cmp_28598dfb8434c1ba_20260402T020734+0000/`

### 3.6 E10 v1 正式结论

当前正式结论固定为：

- `PEFT exp01` 在同后端、同冻结资产下未优于 base
- 主要退化集中在：
  - `quiet_sleep`
  - `focus + avoid`
- 该负结果说明：
  - 当前 v1 训练目标与最终 grounded recommendation 任务不匹配

推荐直接引用：

- `experiments/reports/07_generation_stage_2_e10_formal_summary.md`

### 3.7 E10 v2 固定路线

`E10 v2` 的原则固定为：

- 保持 `Qwen3.5-4B + QLoRA + 同一 compare 协议`
- 不改 retrieval
- 不改 `E9` eval units
- 不改正式 base baseline
- 仅扩展训练数据，新增：
  - `grounded_recommendation`

`v2` 的 grounded 数据池固定来自：

- 非官方 `E9` query pool
- `judged_queries.jsonl + slot_gold.jsonl + clarify_gold.jsonl`
- train/dev 酒店 split 上各自生成的候选酒店与 evidence pack

当前实现说明：

- 仓库内“非官方 `E9` 且无需澄清的可直接执行 query”数量为 `0`
- 因此 `v2` grounded pool 实际采用：
  - 非官方 `E9` query 中，所有已经具备 `city + focus/avoid aspects` gold slot 的 query
  - 并在训练输入中显式提供 `user_preference_gold`
- 这样可以避免与官方 `E9` eval units 重叠，同时不牺牲 grounded supervision 的可执行性

`v2` 的重点 slice 固定为：

- `quiet_sleep`
- `focus + avoid`
- `partial-abstain / 0-recommendation`
- 多方面均衡推荐
- `e10_base_vs_peft` 当前已实现 adapter-ready 评测骨架，默认要求提供 adapter metadata

模型与训练大文件继续只保留在云端或本地忽略目录，不进 Git。需要被纳入版本管理的是：

- `train_config.json`
- `adapter_metadata.json`
- 训练日志摘要
- `E10` 结果 run

`E10` 正式指标固定为：

- `Schema Compliance`
- `Instruction Adherence`
- `Citation Precision`
- `Unsupported Honesty Rate`
- `Faithfulness / Groundedness`
- `avg_latency_ms`

`E10` 正式输出固定写入：

- `experiments/runs/e10_*/`

## 4. 建议推进顺序

后续顺序固定为：

1. 冻结 `E9` 第二轮正式结果与审计快照
2. 固定 `sft_train_manifest.jsonl / sft_dev_manifest.jsonl`
3. 准备 `e10_train_config_template.json` 与 adapter metadata
4. 在云端完成 `4B` 的 QLoRA / PEFT 训练
5. 回传 adapter 与 metadata
6. 跑 `E10` 的 `Base 4B vs PEFT 4B`

当前不再回到 `E9` 主线继续修改 retrieval 配置。

## 5. 验收标准

`E9` 完成标准：

- `e9_generation_constraints` 可完整跑通
- `experiments/assets/e9_generation_eval_units.jsonl` 已冻结
- `experiments/labels/e9_generation/citation_verifiability_audit.csv` 已生成
- `experiments/runs/e9_*/summary.csv` 与 `analysis.md` 成功产出

`E10` 完成标准：

- `sft_train_manifest.jsonl` 与 `sft_dev_manifest.jsonl` 已冻结
- `4B` adapter 可被统一推理入口加载
- `Base 4B` 与 `PEFT 4B` 在同一 `EvidencePack` 上可直接对照
- `experiments/runs/e10_*/summary.csv` 与 `analysis.md` 成功产出

截至 `2026-04-01` 的当前状态：

- `E9 / E10` 的代码入口、schema 与 runner 骨架已实现
- `sft_train_manifest.jsonl` 与 `sft_dev_manifest.jsonl` 已生成
- `E9` 第二轮正式结果已完成并冻结
- 下一步主线切换为 `E10 / PEFT`

## 6. 当前默认假设

- 当前主线不补 `9B` 正式结果 除非论文或答辩需要附录
- 当前生成与微调阶段默认基座都从 `Qwen/Qwen3.5-4B` 出发
- 生成层和行为层都继续坚持“知识外置 行为内化”的职责边界
- `E9` 回答生成是否受证据约束
- `E10` 回答 PEFT 是否提升行为能力
