# Graduation Design Status Hub

本目录用于集中维护当前毕业设计的“活文档”状态板，目标是让你在任何时刻都能快速回答这三个问题：

1. 现在做到哪里了
2. 下一步该做什么
3. 哪些事情必须由你手动介入完成

## 文件说明

- `01_current_progress.md`
  当前工作进度总览，记录已经完成的模块、当前实验状态、最新产物和关键结论。
- `02_next_steps.md`
  下一步执行路线图，按优先级列出后续应该推进的任务。
- `03_manual_intervention_guide.md`
  需要你亲自手动处理的内容，采用详细手册式说明。
- `04_update_rules.md`
  说明这组文件后续如何持续维护与更新。

## 当前推荐先读

- `01_current_progress.md`
  先看当前已经完成到哪里，尤其是 `E9` 第二轮正式结果和有无 RAG 正式对比都已经冻结这一事实。
- `../../experiments/reports/05_behavior_stage_3_chapter_materials.md`
  当前行为章节最值得直接拿来写论文的汇总文档。
- `../../experiments/reports/06_generation_stage_1_e9_formal_summary.md`
  当前生成章节最值得直接复用到论文正文的 `E9` 汇总文档。
- `../../experiments/reports/09_generation_stage_4_e9_rag_ablation_summary.md`
  当前 `E9` 有无 RAG 正式对比的冻结结论，重点回答 `B_grounded_generation` vs `D_no_evidence_generation`。
- `../../experiments/reports/07_generation_stage_2_e10_formal_summary.md`
  当前 `E10 v1` 正式对照的冻结结论与 `v2` 数据方案动机。
- `02_next_steps.md`
  再看接下来按什么顺序推进，当前主线已经从“E9 正式冻结”转向“进入 E10 / PEFT 评测骨架”。
- `03_manual_intervention_guide.md`
  最后看当前需要你亲手做的动作，主要是准备 adapter metadata、云端训练与运行 `E10`。
- `../plans/03_generation_and_peft_phase_plan.md`
  当你准备进入 `E9 -> E10 / PEFT` 时，直接按这份路线图推进。
- `../deployment/01_autodl_qwen35_behavior_runbook.md`
  当你需要在云端复现实验或追加 `Qwen3.5-9B` 对比时，再回看这份总手册。

## 当前维护原则

- 本目录中的文件以“当前真实状态”为准，不以早期设想为准。
- 后续只要我继续在这个项目上推进工作，我会同步更新这里的内容。
- 若某项工作尚未真正完成，会明确写成“待完成”或“待人工处理”，不会包装成已完成。

## 最近更新时间

- 2026-04-04
