本目录用于集中维护当前毕业设计的“活文档”状态板，目标是让你在任何时刻都能快速回答这三个问题：

1. 现在做到哪里了
2. 下一步该做什么
3. 哪些事情必须由你手动介入完成

## 文件说明

- `01_current_progress.md`
  当前工作进度总览，记录已经完成的模块、冻结结果、最新代码实现状态和当前真实阻塞项。
- `02_next_steps.md`
  下一步执行路线图，按“8 小时跑通实验闭环 / 32 小时产出论文首版结果”两个时间窗口组织。
- `03_manual_intervention_guide.md`
  当前需要你亲自处理的命令、环境变量、云端动作和结果回收步骤。
- `04_update_rules.md`
  说明这组文件后续如何持续维护与更新。

## 当前推荐先读

- `01_current_progress.md`
  先看当前代码、资产、历史 run 和 G 系列实验框架已经落地到哪里。
- `../plans/01_thesis_overall_plan.md`
  当前论文主线的唯一总计划，已切换到 “E5-E10 保留 + G1-G4 统一框架”。
- `02_next_steps.md`
  当前最重要的不是继续扩写 E10，而是补齐 G 系列闭环并准备上云。
- `03_manual_intervention_guide.md`
  当前人工动作已经切换到 G1-G4 资产冻结、云端运行、统计检验、Judge 和盲评汇总。
- `../../experiments/reports/05_behavior_stage_3_chapter_materials.md`
  第五章行为层与桥接层现成可写材料。
- `../../experiments/reports/06_generation_stage_1_e9_formal_summary.md`
  第五章生成层 RAG 约束实验的正式冻结材料。
- `../../experiments/reports/09_generation_stage_4_e9_rag_ablation_summary.md`
  第五章 “有无 RAG” 正式 compare 的冻结结论。
- `../../experiments/runs/e10_0dc5c2e6f867c66f_20260402T015230+0000/`
  E10 base formal 冻结 run。
- `../../experiments/runs/e10_a2dd1a0bd73c57b5_20260402T073127+0000/`
  E10 v2 / exp02 阶段性最佳 PEFT run。

## 当前维护原则

- 本目录中的文件以“当前真实状态”为准，不以早期设想为准。
- 如果代码已经落地，但还没有正式云端运行，会明确写成“代码已实现，结果待正式运行”。
- 如果某项工作只存在 helper/library，尚未接入统一 runner，也会明确说明，不把“可调用”包装成“已形成正式实验闭环”。
- 历史 `E1-E10` run 与报告保持只读，不覆盖、不回写。

## 当前阶段一句话判断

当前项目已经从 “E10 v3/v4 迭代评估” 正式切换到 “G1-G4 统一框架闭环前的最终工程收口” 阶段：

- `E1-E10` 作为论文第五章和第六章的辅助证据已经基本冻结
- `G1-G4` 的检索、生成、统计检验、LLM Judge、人工盲评导出等代码底座已大体接通
- 但 `G1-G4` 的正式统一结果、统计检验结果、Judge 结果和章节总报告尚未正式产出；当前正式 decisive scope 已按 Protocol A 调整为 `68` 条（`39 core + 29 robustness`），其中 `q021 / q024` 作为 supporting boundary cases 排除

## 最近更新时间

- 2026-04-05
