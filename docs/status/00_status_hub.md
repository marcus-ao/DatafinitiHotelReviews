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
  当前论文主线的唯一总计划；本轮正式执行口径已统一为“全量重跑 `E1-E10 + G1-G4`，旧结果仅作 archive 参考”。
- `02_next_steps.md`
  当前最重要的是按统一协议正式重跑 `E1-E10 + G1-G4`，再完成统计检验、Judge、盲评和章节收口。
- `03_manual_intervention_guide.md`
  当前人工动作围绕“全量正式重跑 `E1-E10 + G1-G4`”展开，G 系列闭环只是其中最后一段 decisive pipeline。
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
- 历史 `E1-E10` run 与报告保持只读，不作为本轮正式 canonical 结果；本轮正式结果只认新 rerun 产物。

## 当前阶段一句话判断

当前项目已经从“历史实验结果冻结展示”切换到“全量正式重跑 `E1-E10 + G1-G4` 并形成 canonical 论文结果”的执行阶段：

- `E1-E10` 与 `G1-G4` 的代码底座、冻结资产与执行协议已经基本准备好
- 旧的 `E1-E10` 结果只作为 archive 参考，不再直接作为本轮正式论文结果引用
- 当前本轮正式结果仍待完整重跑并登记；其中 `G1-G4` 的正式 decisive scope 已按 Protocol A 调整为 `68` 条（`39 core + 29 robustness`），`q021 / q024` 作为 supporting boundary cases 排除

## 最近更新时间

- 2026-04-05
