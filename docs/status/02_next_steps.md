更新时间：2026-04-05

## 当前推荐推进顺序

请按下面顺序推进，不建议跳步：

1. 先确认全量正式重跑基线（registry / manifest / frozen assets / decisive scope）
2. 正式重跑 `E1-E4`
3. 正式重跑 `E5-E8`
4. 正式重跑 `E9-E10`
5. 正式重跑 `G1-G4`
6. 回本地完成统计检验、Judge、盲评导出与章节总报告
7. 在首轮全流程结果产出并登记后，再进入一天的定向优化

## 8 小时目标：跑通完整实验闭环

### 任务 1：确认本轮正式重跑基线

目标：

- 确认 `final_rerun_baseline.json`
- 确认 `final_rerun_registry.json` 为空但结构正确
- 确认当前正式 decisive scope 为 `68`
- 确认 `exp02` metadata、frozen split、qrels、query assets 全部存在

完成标准：

- Stop/Go 预检查全绿

### 任务 2：正式重跑 `E1-E4`

目标：

- 产出本轮新的前置验证 canonical runs

完成标准：

- `E1-E4` 各自具备新的 `run_meta.json / summary.csv / analysis.md`

### 任务 3：正式重跑 `E5-E8`

目标：

- 基于 `e5_e8_core_query_ids.json` 重新产出 retrieval supporting evidence canonical runs

完成标准：

- `E5-E8` 全部重跑完成，并使用 `40` query frozen scope

### 任务 4：正式重跑 `E9-E10`

目标：

- 重新产出 `E9`
- 重新产出 `E10 base / v1 / v2 / v3 / v4`

完成标准：

- `E9-E10` supporting evidence 层形成新的 canonical runs

### 任务 5：正式冻结并检查 `G` 系列 retrieval assets

目标：

- 生成并验证：
  - `experiments/assets/g_plain_generation_eval_units.jsonl`
  - `experiments/assets/g_aspect_generation_eval_units.jsonl`

完成标准：

- 两份资产都存在
- 检索模式与 candidate policy 和组定义一致
- 当前正式 decisive query 集（`68` 条）覆盖完整，不缺、不重；`q021 / q024` 作为 supporting boundary cases 单独保留

### 任务 6：云端跑完 `G1-G4`

目标：

- 生成四个正式 `ggen_*` 运行目录

当前四组定义固定为：

- `G1 = Plain Retrieval + Base`
- `G2 = Aspect Retrieval + Base`
- `G3 = Plain Retrieval + PEFT exp02`
- `G4 = Aspect Retrieval + PEFT exp02`

完成标准：

- `G1-G4` 全部有 `results.jsonl / summary.csv / citation_verifiability_audit.csv / analysis.md`
- 四组 query 集一致，均使用当前正式 decisive scope（`68` 条）

### 任务 7：生成核心 pairwise compare

目标：

- 至少产出以下 compare：
  - `G2 vs G1`
  - `G4 vs G3`
  - `G3 vs G1`
  - `G4 vs G2`

建议额外产出：

- `G4 vs G3`
- `G4 vs G1`

完成标准：

- 每组 compare 都有独立 `gcmp_*` 目录
- compare 结果能直接回答 `RQ1 / RQ2 / RQ3`

### 任务 8：完成统计检验、Judge 与盲评材料导出

目标：

- 从四组正式 run 中提取 score map
- 跑完 pairwise tests
- 跑完 LLM Judge
- 导出人工盲评包

当前代码状态：

- `statistical_tests.py` 已可用
- `llm_judge.py` 已可用
- `blind_review_export.py` 已可用
- `g_workflow_closure.py` 已实现：
  - score map 提取
  - 批量 Judge
  - blind review 结果聚合
  - 章节报告生成

完成标准：

- `pairwise_tests.csv`
- `judge_scores.csv / judge_summary.csv`
- `blind_review_pack.csv`
- `blind_review_worksheet.csv`

### 任务 9：生成 G1-G4 统一章节报告并回填 registry/manifest

目标：

- 把四组 summary、统计检验、Judge 结果和盲评结果整合成 chapter-ready 输出

完成标准：

- 至少生成：
  - `g_retrieval_summary.csv`
  - `g_generation_summary.csv`
  - `analysis.md`

## 32 小时目标：产出论文首版全流程结果

在 8 小时内跑通完整闭环之后，再用接下来的时间做定向优化：

1. 先看 `E1-E10 + G1-G4` 首轮正式结果是否满足论文级可用性
2. 若某一组明显异常，只对对应资产、metadata 或运行配置做最小修复
3. 若 Judge 或统计检验暴露出明显短板，再做第二轮精修
4. 不把定向优化回写成“旧 archive 结果仍可直接作为本轮 canonical 结果”的口径

## 当前不建议做的事

以下内容当前都不建议插队：

- 不以“历史冻结结果可直接沿用”为由跳过本轮正式 rerun
- 不再追加新的 PEFT 轮次作为主线
- 不把 `reranker` 或 `fallback` 接回默认主流程
- 不为了跑快而跳过 `G` 资产冻结与 metadata 校验
- 不在没有四组正式 run 的情况下先写第七章最终结论

## 一句话版本

当前最重要的，是把 `E1-E10 + G1-G4` 的正式 canonical rerun 按统一协议全部跑通，并完成 registry、统计检验、Judge、盲评和章节总报告的闭环。
