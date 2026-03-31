# 当前工作进度

更新时间：2026-03-31

## 总体阶段判断

当前项目已经完成 Aspect-KB 章节的两轮核心收口，并进入行为实验第一轮。

目前整体处于：

- 数据底座：已完成
- 实验底座：已完成并冻结
- `E1`：已完成正式评估
- `E2`：已完成首轮正式结果与失败分析
- `E6-E8`：已完成正式检索评测并已收口为论文材料
- 默认检索配置：已冻结为 `aspect_main_no_rerank`
- `E5`：已完成正式首轮结果
- `E3/E4`：共享评测引擎已实现，但 Base 组正式运行改为放在云端 GPU 环境执行
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

当前状态：共享行为评测底座已实现，`E5` 已完成正式结果，`E3/E4` 待正式运行

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
  - `behavior.base_model = Qwen/Qwen2.5-3B-Instruct`

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

- 规则组与 Base 组实验逻辑都已写好
- Base 组固定模型仍是 `Qwen/Qwen2.5-3B-Instruct`
- 但该模型后续将放在云端有 GPU 的环境执行，不再在当前本地机器下载或缓存

## 当前已形成的可写材料

当前已经具备：

- `1` 组 E1 正式结果表
- `1` 组 E2 首轮结果表
- `1` 份 Aspect-KB 第一阶段汇总材料
- `1` 份 Aspect-KB 第二阶段检索汇总材料
- `1` 组 E5 正式桥接结果表
- 多组 E1 / E2 / E6-E8 代表性案例

推荐优先阅读：

- `experiments/reports/01_aspect_kb_stage_1_summary.md`
- `experiments/reports/02_aspect_kb_stage_2_summary.md`
- `experiments/runs/e5_9a94daa5a6a31d8a_20260330T155246+0000/analysis.md`

## 当前明确尚未完成的内容

- `E3` 正式 A/B 运行与结果分析
- `E4` 正式 A/B 运行与结果分析
- `experiments/labels/e4_clarification/clarification_question_audit.csv` 的人工审阅
- `E9/E10`
- `G1-G4`
- PEFT

## 当前最值得关注的事实

1. Aspect-KB 章节已经完成两轮核心收口，当前主结论已经稳定。
2. `E6` 证明了方面引导检索有效，`E7/E8` 作为负结果与边界结果同样有论文价值。
3. 后续系统默认检索配置已经冻结为 `aspect_main_no_rerank`，不再把 reranker 或 fallback 当作主流程默认项。
4. `E5` 已经证明“结构化英文检索表达”显著优于“中文直检”。
5. 当前最直接的下一步，不是 PEFT，而是让 `E3/E4` 在固定检索后端上完成正式运行与分析。
