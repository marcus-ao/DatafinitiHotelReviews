更新时间：2026-04-05

## 当前推荐推进顺序

请按下面顺序推进，不建议跳步：

1. 继续保持 `E1-E10` 历史 run 与报告冻结，不再回头改旧实验主体
2. 补齐 `exp02 / v2` adapter metadata 的本地资产
3. 正式生成 `G` 系列 retrieval assets
4. 在云端跑完 `G1 / G2 / G3 / G4`
5. 回本地完成统计检验、Judge、盲评导出与章节总报告
6. 在首轮全流程结果产出后，再进入一天的定向优化

## 8 小时目标：跑通完整实验闭环

### 任务 1：补齐 `exp02` metadata

目标：

- 在本地资产目录中补齐 `experiments/assets/e10_adapter_metadata.qwen35_4b_peft_v2.json`

原因：

- `G3 / G4` 默认使用 `exp02`
- 当前本地缺少 `v2` metadata，会直接阻塞 PEFT 组运行

完成标准：

- `scripts/evaluation/g_workflow_closure.py` 中的 `validate_exp02_metadata` 能通过
- `G3 / G4` 不再因 metadata 缺失而无法启动

### 任务 2：正式冻结 `G` 系列 retrieval assets

目标：

- 生成并写入：
  - `experiments/assets/g_plain_generation_eval_units.jsonl`
  - `experiments/assets/g_aspect_generation_eval_units.jsonl`

完成标准：

- 两份资产都存在
- 检索模式与 candidate policy 和组定义一致
- 当前正式 decisive query 集（`68` 条）覆盖完整，不缺、不重；`q021 / q024` 作为 supporting boundary cases 单独保留

### 任务 3：云端跑完 `G1-G4`

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

### 任务 4：生成核心 pairwise compare

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

### 任务 5：完成统计检验、Judge 与盲评材料导出

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

### 任务 6：生成 G1-G4 统一章节报告

目标：

- 把四组 summary、统计检验、Judge 结果和盲评结果整合成 chapter-ready 输出

完成标准：

- 至少生成：
  - `g_retrieval_summary.csv`
  - `g_generation_summary.csv`
  - `analysis.md`

## 32 小时目标：产出论文首版全流程结果

在 8 小时内跑通完整闭环之后，再用接下来的时间做定向优化：

1. 先看 `G1-G4` 首轮结果是否满足论文级可用性
2. 若某一组明显异常，只对对应资产、metadata 或运行配置做最小修复
3. 若 Judge 或统计检验暴露出明显短板，再做第二轮精修
4. 保持 `E5-E10` 历史结果冻结，不把“优化”扩散回旧实验主线

## 当前不建议做的事

以下内容当前都不建议插队：

- 不回头继续扩写 `E10 v3 / v4`
- 不再追加新的 PEFT 轮次作为主线
- 不把 `reranker` 或 `fallback` 接回默认主流程
- 不为了跑快而跳过 `G` 资产冻结与 metadata 校验
- 不在没有四组正式 run 的情况下先写第七章最终结论

## 一句话版本

当前最重要的，不是继续做历史实验，而是把 `G1-G4` 的资产、运行、统计检验、Judge、盲评和章节总报告完整接成一条线，然后一次性上云跑通。
