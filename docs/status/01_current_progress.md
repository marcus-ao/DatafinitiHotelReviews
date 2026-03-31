# 当前工作进度

更新时间：2026-03-31

## 总体阶段判断

当前项目已经完成 Aspect-KB 章节的两轮核心收口，并完成行为实验主线的首轮正式结果。

目前整体处于：

- 数据底座：已完成
- 实验底座：已完成并冻结
- `E1`：已完成正式评估
- `E2`：已完成首轮正式结果与失败分析
- `E6-E8`：已完成正式检索评测并已收口为论文材料
- 默认检索配置：已冻结为 `aspect_main_no_rerank`
- `E5`：已完成正式首轮结果
- `E3/E4`：`Qwen3.5-4B` 全量正式结果已完成，`Qwen3.5-2B` 第一轮结果作为弱基线归档保留
- 当前正式行为模型：已冻结为 `Qwen/Qwen3.5-4B`
- `Qwen3.5-9B`：可作为附录或扩展对比，当前不是主线阻塞项
- PEFT / 主实验矩阵：尚未开始

## 已完成的核心内容

### 1. 数据底座

已完成 `scripts/pipeline/` 下的 9 步数据流水线，并通过验证。

当前确认结果：

- 覆盖城市：10
- 覆盖州：8
- 酒店数：146
- 清洗后评论数：5947
- 句子数：51813
- 方面情感标签数：63085
- 酒店方面画像：876
- `python -m scripts.pipeline.validate_kb_assets`：`28/28` 通过

### 2. 实验底座

已完成并冻结：

- 冻结实验配置
- 酒店级 train/dev/test 切分
- 中文 query 集
- `slot_gold`
- `clarify_gold`
- 标注 rubric
- 实验 schema
- 最小 batch runner

关键产物位置：

- `experiments/assets/frozen_config.yaml`
- `experiments/assets/frozen_split_manifest.json`
- `experiments/assets/judged_queries.jsonl`
- `experiments/assets/slot_gold.jsonl`
- `experiments/assets/clarify_gold.jsonl`
- `experiments/assets/annotation_rubrics.md`

### 3. E1：方面/情感标注可靠性对照

当前状态：正式结果已完成

已完成内容：

- `sample` 与正式 gold 已严格对齐到 `360 / 344 / 344`
- `aspect_sentiment_gold.csv` 已确认为唯一正式 gold
- `e1_metrics.json` 与 `e1_report.md` 已成功生成

当前正式结果：

- `rule_only`
  - `Aspect macro-F1 = 0.5107`
  - `Difficult-set Jaccard = 0.7812`
- `zeroshot_only`
  - `Aspect macro-F1 = 0.0932`
  - `Difficult-set Jaccard = 0.0099`
- `hybrid`
  - `Aspect macro-F1 = 0.4960`
  - `Difficult-set Jaccard = 0.7911`
- `Sentiment macro-F1 = 0.4458`

当前解读：

- `hybrid` 明显优于 `zeroshot_only`
- `hybrid` 在多方面困难集上优于 `rule_only`
- `hybrid` 还没有在主类别整体 `Aspect macro-F1` 上超过 `rule_only`
- 最突出混淆为 `service -> room_facilities`

### 4. E2：画像驱动候选缩圈有效性对照

正式结果目录：

- `experiments/runs/e2_770d3e0e2f4ded57_20260329T124258+0000/`

当前正式结果：

- `A_rating_review_count`
  - `Candidate Hit@5 (proxy) = 0.9`
  - `avg_latency_ms = 111.641`
- `B_final_aspect_score`
  - `Candidate Hit@5 (proxy) = 0.9`
  - `avg_latency_ms = 102.223`

当前解读：

- `B_final_aspect_score` 只表现出轻微延迟优势，没有拉开命中率差距
- 当前不能写成“Aspect-KB 明显优于基线”
- 共同失败的 `q021 / q022 / q023 / q081` 集中在 Honolulu 的 `quiet_sleep`
- 失败主因是切分后的候选稀疏，其次才是主标签机制对 `quiet_sleep` 的召回损失

### 5. E6-E8：检索评测实验

当前状态：三组正式检索实验已完成，并已收口为论文材料

已完成内容：

- `scripts/evaluation/evaluate_e6_e8_retrieval.py` 已实现四种官方检索模式
- `scripts/evaluation/run_experiment_suite.py` 已接入：
  - `e6_qrels_pool`
  - `e6_freeze_qrels`
  - `e6_retrieval`
  - `e7_reranker`
  - `e8_fallback`
- `E6` 标注池与正式 qrels 已冻结：
  - `40` 条可执行 query
  - `80` 个 `query-aspect` 单元
  - `817` 条人工标注证据
- `experiments/reports/02_aspect_kb_stage_2_summary.md` 已形成正式汇总材料

当前正式结果：

- `E6`
  - `plain_city_test_rerank -> aspect_main_rerank`
  - `Aspect Recall@5: 0.7000 -> 0.8500`
  - `nDCG@5: 0.3307 -> 0.6378`
  - `MRR@5: 0.5750 -> 0.7842`
  - `Precision@5: 0.3350 -> 0.6375`
- `E7`
  - `aspect_main_no_rerank`
    - `nDCG@5 = 0.6457`
    - `MRR@5 = 0.7781`
    - `Precision@5 = 0.6450`
    - `avg_latency_ms = 133.256`
  - `aspect_main_rerank`
    - `nDCG@5 = 0.6378`
    - `MRR@5 = 0.7842`
    - `Precision@5 = 0.6375`
    - `avg_latency_ms = 345.117`
- `E8`
  - `aspect_main_rerank` 与 `aspect_main_fallback_rerank` 的 `nDCG@5 / Precision@5` 完全相同
  - `fallback_activation_rate = 0.05`
  - `fallback_noise_rate = 1.0`

当前解读：

- `E6` 是明确正结果，证明方面引导检索有效
- `E7` 是负结果，说明当前 reranker 没有带来稳定收益，且延迟更高
- `E8` 是边界结果，说明当前 fallback 暴露的是证据覆盖不足，而不是排序问题
- 因此后续默认检索配置已冻结为 `aspect_main_no_rerank`
- 当前 `fallback` 不进入默认主流程

### 6. E3-E5：行为实验第一轮

当前状态：共享行为评测底座已实现，`E5` 已完成正式结果，`E3/E4` 已完成 `Qwen3.5-4B` 全量正式结果

已完成内容：

- `scripts/evaluation/evaluate_e3_e5_behavior.py` 已实现三类任务：
  - `e3_preference`
  - `e4_clarification`
  - `e5_query_bridge`
- `scripts/shared/experiment_schemas.py` 已新增：
  - `PreferenceParseResult`
  - `ClarificationDecision`
  - `BridgeQueryRecord`
- `WorkflowState` 已扩展：
  - `retrieval_mode`
  - `fallback_enabled`
  - `run_config_hash`
- 运行默认配置已冻结：
  - `workflow.default_retrieval_mode = aspect_main_no_rerank`
  - `workflow.enable_fallback = false`
  - `behavior.llm_backend = api`
  - `behavior.base_model = Qwen/Qwen3.5-4B`

`E5` 正式结果目录：

- `experiments/runs/e5_9a94daa5a6a31d8a_20260330T155246+0000/`

`E5` 当前正式结果：

- `A_zh_direct_dense_no_rerank`
  - `Aspect Recall@5 = 0.625`
  - `nDCG@5 = 0.2643`
  - `MRR@5 = 0.4233`
  - `Precision@5 = 0.2200`
- `B_structured_query_en_dense_no_rerank`
  - `Aspect Recall@5 = 0.850`
  - `nDCG@5 = 0.6457`
  - `MRR@5 = 0.7781`
  - `Precision@5 = 0.6450`

`E5` 当前解读：

- 结构化英文检索表达显著优于中文直检
- 当前桥接收益已经被正式跑出来，可以作为后续系统设计的重要依据
- `avoid` 与 `quiet_sleep` 仍然偏弱，说明其中一部分瓶颈仍然是证据稀疏，而不只是语言桥接

`E3/E4` 当前状态说明：

- `Qwen3.5-2B` baseline run 已冻结：
  - `experiments/runs/e3_244aca8abf6345ad_20260331T072527+0000/`
  - `experiments/runs/e4_4a15a89128a90d11_20260331T073016+0000/`
- `Qwen3.5-2B` baseline 归档总结已写入：
  - `experiments/reports/03_behavior_stage_1_qwen35_2b_baseline.md`
- 中间诊断 run 已保留：
  - `experiments/runs/e3_da541f84770ed8ed_20260331T090311+0000/`
  - `experiments/runs/e4_96e0e4afb24dab2d_20260331T091021+0000/`
  - `experiments/runs/e3_f62d907e600cfc14_20260331T120756+0000/`
  - `experiments/runs/e4_f928a37444c1bf52_20260331T121012+0000/`
- 当前正式全量 run 已完成：
  - `experiments/runs/e3_14928d821d811e86_20260331T122611+0000/`
  - `experiments/runs/e4_55c8021e1119fb77_20260331T122648+0000/`
- 当前正式汇总材料已写入：
  - `experiments/reports/04_behavior_stage_2_qwen35_4b_formal_summary.md`
- 当前最新 `E4` 审计快照已冻结：
  - `experiments/labels/e4_clarification/clarification_question_audit_e4_55c8021e1119fb77_qwen35_4b_full.csv`
- 当前代码已稳定使用 `v2` 设计：
  - `E3 = e3_v2_cn_slots_only`
  - `E4 = e4_v2_cn_decision_label_fewshot`
- 云端部署与执行总手册见 `docs/deployment/01_autodl_qwen35_behavior_runbook.md`

`E3` 当前正式结果：

- `A_rule_parser`
  - `Exact-Match Rate = 1.0000`
  - `Unsupported Detection Recall = 1.0000`
- `B_base_llm_structured`
  - `Exact-Match Rate = 0.9767`
  - `Unsupported Detection Recall = 1.0000`
  - `City Slot F1 = 1.0000`
  - `Focus Slot F1 = 1.0000`
  - `Avoid Slot F1 = 0.9474`
  - `Unsupported Slot F1 = 0.9836`

`E4` 当前正式结果：

- `A_rule_clarify`
  - `Accuracy = 0.9767`
  - `Precision = 0.8889`
  - `Recall = 1.0000`
  - `F1 = 0.9412`
- `B_base_llm_clarify`
  - `Accuracy = 0.9884`
  - `Precision = 0.9412`
  - `Recall = 1.0000`
  - `F1 = 0.9697`
  - `Over-clarification Rate = 0.0143`
  - `Under-clarification Rate = 0.0000`

当前解读：

- `Qwen3.5-2B` 适合作为行为层弱基线保留，但不再作为正式主模型
- `Qwen3.5-4B` 已经足够支撑 `E3/E4` 的正式论文结果
- `E3` 的剩余错例已收缩到极少数 `value` 负向约束边界例，如 `q048 / q062`
- `E4` 当前只剩 `q013` 这类易被误看成 conflict 的边界过澄清案例

## 当前已形成的可写材料

当前已经具备：

- `1` 组 E1 正式结果表
- `1` 组 E2 首轮结果表
- `1` 份 Aspect-KB 第一阶段汇总材料
- `1` 份 Aspect-KB 第二阶段检索汇总材料
- `1` 组 E5 正式桥接结果表
- `1` 份 `Qwen3.5-2B` 行为弱基线归档材料
- `1` 份 `Qwen3.5-4B` 行为正式结果归档材料
- 多组 E1 / E2 / E6-E8 代表性案例
- 多组 E3 / E4 行为案例与审计文件

推荐优先阅读：

- `experiments/reports/01_aspect_kb_stage_1_summary.md`
- `experiments/reports/02_aspect_kb_stage_2_summary.md`
- `experiments/reports/03_behavior_stage_1_qwen35_2b_baseline.md`
- `experiments/reports/04_behavior_stage_2_qwen35_4b_formal_summary.md`
- `experiments/runs/e3_14928d821d811e86_20260331T122611+0000/analysis.md`
- `experiments/runs/e4_55c8021e1119fb77_20260331T122648+0000/analysis.md`
- `experiments/runs/e5_9a94daa5a6a31d8a_20260330T155246+0000/analysis.md`

## 当前明确尚未完成的内容

- 最新一轮 `E4` 审计文件的人工审阅
- 将 `E3 / E4 / E5` 收口为统一的行为实验论文材料
- 视论文篇幅决定是否追加 `Qwen3.5-9B` 扩展对比
- `E9/E10`
- `G1-G4`
- PEFT

## 当前最值得关注的事实

1. Aspect-KB 章节已经完成两轮核心收口，当前主结论已经稳定。
2. 默认检索后端仍固定为 `aspect_main_no_rerank`，`reranker` 与 `fallback` 不回到主流程。
3. 行为实验当前正式主模型已经切换为 `Qwen/Qwen3.5-4B`，`2B` 保留为弱基线。
4. `E5` 已经证明“结构化英文检索表达”显著优于“中文直检”，因此行为层实验建立在固定桥接与固定检索后端之上。
5. 当前阶段更像“审计、收口和写作阶段”，而不是“继续救火调试阶段”；最直接的下一步是补齐 `E4` 审计并整理 `E3 / E4 / E5` 论文材料。
