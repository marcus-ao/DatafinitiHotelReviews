# 当前工作进度

更新时间：2026-04-01

## 总体阶段判断

当前项目已经完成 Aspect-KB 章节与行为实验主线的首轮正式收口，并进入“`E9` 第二轮正式结果已冻结 + `E10 / PEFT` 评测骨架推进”阶段。

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
- 最新 `E4` 审计：已完成首轮人工评分并冻结 reviewed 快照
- 行为章节材料：已形成 `E3 / E4 / E5` 汇总初稿
- `Qwen3.5-9B`：可作为附录或扩展对比，当前不是主线阻塞项
- `E9 / E10 / PEFT`：主线方案已冻结，其中 `E9` 第二轮正式结果已完成，`E10` 进入 adapter-ready 评测骨架阶段

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

### 3. Aspect-KB 阶段结果

`E1` 正式结果：

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

`E2` 正式结果：

- `A_rating_review_count`
  - `Candidate Hit@5 (proxy) = 0.9`
  - `avg_latency_ms = 111.641`
- `B_final_aspect_score`
  - `Candidate Hit@5 (proxy) = 0.9`
  - `avg_latency_ms = 102.223`

`E6-E8` 正式结果：

- `E6`
  - `Aspect Recall@5: 0.7000 -> 0.8500`
  - `nDCG@5: 0.3307 -> 0.6378`
  - `MRR@5: 0.5750 -> 0.7842`
  - `Precision@5: 0.3350 -> 0.6375`
- `E7`
  - `aspect_main_no_rerank`
    - `nDCG@5 = 0.6457`
    - `MRR@5 = 0.7781`
    - `Precision@5 = 0.6450`
  - `aspect_main_rerank`
    - `nDCG@5 = 0.6378`
    - `MRR@5 = 0.7842`
    - `Precision@5 = 0.6375`
- `E8`
  - `aspect_main_rerank` 与 `aspect_main_fallback_rerank` 的 `nDCG@5 / Precision@5` 完全相同
  - `fallback_activation_rate = 0.05`
  - `fallback_noise_rate = 1.0`

当前解读：

- `E6` 是明确正结果，证明方面引导检索有效
- `E7` 是负结果，说明当前 reranker 没有带来稳定收益
- `E8` 是边界结果，说明当前 fallback 暴露的是证据覆盖不足而不是排序问题
- 因此后续默认检索配置已冻结为 `aspect_main_no_rerank`

## 4. 行为实验主线当前状态

### `E5` 正式结果

正式结果目录：

- `experiments/runs/e5_9a94daa5a6a31d8a_20260330T155246+0000/`

当前正式结果：

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

当前解读：

- 结构化英文检索表达显著优于中文直检
- 桥接收益已经被正式验证，可以直接写入论文主线

### `E3 / E4` 正式结果

`Qwen3.5-2B` baseline run 已冻结：

- `experiments/runs/e3_244aca8abf6345ad_20260331T072527+0000/`
- `experiments/runs/e4_4a15a89128a90d11_20260331T073016+0000/`

`Qwen3.5-4B` 正式全量 run 已完成：

- `experiments/runs/e3_14928d821d811e86_20260331T122611+0000/`
- `experiments/runs/e4_55c8021e1119fb77_20260331T122648+0000/`

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
- `E3` 的剩余误差已收缩到极少数 `value` 边界例
- `E4` 的残留边界已收缩到 `q013` 这类少数误澄清样本

### 最新审计与材料

当前新增并已完成：

- 最新 `E4` 审计表：
  - `experiments/labels/e4_clarification/clarification_question_audit.csv`
- `4B` 原始快照：
  - `experiments/labels/e4_clarification/clarification_question_audit_e4_55c8021e1119fb77_qwen35_4b_full.csv`
- `4B` reviewed 快照：
  - `experiments/labels/e4_clarification/clarification_question_audit_e4_55c8021e1119fb77_qwen35_4b_reviewed.csv`
- 行为章节材料汇总：
  - `experiments/reports/05_behavior_stage_3_chapter_materials.md`
- 后续阶段规划：
  - `docs/plans/03_generation_and_peft_phase_plan.md`

## 当前已形成的可写材料

当前已经具备：

- `1` 组 E1 正式结果表
- `1` 组 E2 首轮结果表
- `1` 份 Aspect-KB 第一阶段汇总材料
- `1` 份 Aspect-KB 第二阶段检索汇总材料
- `1` 份 `Qwen3.5-2B` 行为弱基线归档材料
- `1` 份 `Qwen3.5-4B` 行为正式结果归档材料
- `1` 份 `E3 / E4 / E5` 行为章节材料汇总
- 多组 E1 / E2 / E6-E8 代表性案例
- 多组 E3 / E4 行为案例与审计文件

推荐优先阅读：

- `experiments/reports/05_behavior_stage_3_chapter_materials.md`
- `experiments/reports/04_behavior_stage_2_qwen35_4b_formal_summary.md`
- `experiments/runs/e3_14928d821d811e86_20260331T122611+0000/analysis.md`
- `experiments/runs/e4_55c8021e1119fb77_20260331T122648+0000/analysis.md`
- `experiments/runs/e5_9a94daa5a6a31d8a_20260330T155246+0000/analysis.md`
- `docs/plans/03_generation_and_peft_phase_plan.md`

## 当前明确尚未完成的内容

- 决定是否需要追加 `Qwen3.5-9B` 附录对比
- `E10` Base vs PEFT 正式行为对照
- QLoRA / PEFT 训练基础设施
- `G1-G4` 主实验矩阵

## 5. `E9 / E10` 当前实现进展

截至 `2026-04-01`，仓库内已经新增并接通：

- `scripts/evaluation/evaluate_e9_e10_generation.py`
- `scripts/evaluation/run_experiment_suite.py` 中的：
  - `e9_freeze_assets`
  - `e9_generation_constraints`
  - `e10_prepare_manifests`
  - `e10_base_vs_peft`
- `scripts/shared/experiment_schemas.py` 中的：
  - `RecommendationReason`
  - `RecommendationItem`
  - `RecommendationResponse`
  - `CitationVerificationResult`
  - `GenerationEvalUnit`
  - `SFTManifestRecord`
- `experiments/labels/e9_generation/README.md`
- `experiments/labels/e9_generation/citation_verifiability_audit.csv`
- `experiments/assets/sft_train_manifest.jsonl`
- `experiments/assets/sft_dev_manifest.jsonl`
- `tests/test_e9_e10_generation.py`

当前已经完成的验证：

- `E9` query 选择逻辑已由单元测试确认仍对应 `40` 条可执行 query
- citation verifier 的合法 / 不存在 / 越权三类情况已由单元测试覆盖
- verifier 二次失败后的 honest fallback 已由单元测试覆盖
- `e10_prepare_manifests` 已成功生成 `train/dev` 两份 manifest
- `e9_freeze_assets --limit-queries 2` 已成功跑通本地 smoke

当前已经完成并冻结的部分：

- `experiments/assets/e9_generation_eval_units.jsonl`
- `experiments/assets/e9_generation_eval_query_ids.json`
- `experiments/runs/e9_ecbcdbab690dc503_20260401T025012+0000/`
- `experiments/labels/e9_generation/citation_verifiability_audit.csv`

当前仍未完成的部分：

- `citation_verifiability_audit_e9_ecbcdbab690dc503_qwen35_4b_reviewed.csv`
  以外的更多人工复核扩展
- `E10` 的正式 Base vs PEFT 行为对照
- PEFT 训练、adapter 导出与云端执行

当前新增实现的固定口径：

- `E9` 检索模式固定为 `aspect_main_no_rerank`
- `fallback=false`
- 行为基座固定为 `Qwen/Qwen3.5-4B`
- `candidate_hotels` 固定使用 `E2 B_final_aspect_score Top5`
- `E9` 资产冻结当前优先只读本地 embedding 缓存，不新增 live PostgreSQL 依赖

## 6. `E9` 第二轮正式结果

当前正式 `E9` run 固定为：

- `experiments/runs/e9_ecbcdbab690dc503_20260401T025012+0000/`

第一轮 run 保留为历史诊断对照：

- `experiments/runs/e9_80e05af30f45b1f2_20260401T021215+0000/`

第二轮正式指标：

- `A_free_generation`
  - `Citation Precision = 0.9437`
  - `Evidence Verifiability Mean = 1.9697`
  - `Schema Valid Rate = 1.0000`
- `B_grounded_generation`
  - `Citation Precision = 0.9437`
  - `Evidence Verifiability Mean = 1.9773`
  - `Schema Valid Rate = 1.0000`
- `C_grounded_generation_with_verifier`
  - `Citation Precision = 0.9250`
  - `Evidence Verifiability Mean = 1.9922`
  - `Schema Valid Rate = 1.0000`
  - `retry_trigger_rate = 0.025`
  - `fallback_to_honest_notice_rate = 0.025`

当前正式解读固定为：

- 第二轮 `E9` 已经解决第一轮的主要格式稳定性问题
- `q021 / q023` 属于证据覆盖边界的诚实空输出，不视为系统失败
- `q079` 是 verifier 过严的单点残留边界，应保留到误差分析
- 当前 `E9` 已达到可审计、可复现、可冻结的稳定度

## 当前最值得关注的事实

1. Aspect-KB 章节已经完成两轮核心收口，主结论已经稳定。
2. 默认检索后端仍固定为 `aspect_main_no_rerank`，`reranker` 与 `fallback` 不回到主流程。
3. 行为实验当前正式主模型已经切换为 `Qwen/Qwen3.5-4B`，`2B` 保留为弱基线。
4. 最新 `E4` 审计已经补齐，行为章节材料已经具备写论文的基本条件。
5. 当前阶段更像“冻结 `E9` 正式结果并进入 `E10 / PEFT` 评测骨架阶段”，而不是“继续追着 retrieval 或 `E9` prompt 大改阶段”。
