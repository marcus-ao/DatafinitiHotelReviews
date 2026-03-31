# 更新规则

更新时间：2026-03-31

本文件说明 `docs/status/` 目录中的文件如何持续维护。

## 一、更新原则

每次项目有实质进展时，应同步更新以下文件：

- `01_current_progress.md`
- `02_next_steps.md`
- `03_manual_intervention_guide.md`

如果进展涉及云端部署、模型对比方案或 AutoDL 操作步骤，还应同步更新：

- `../deployment/01_autodl_qwen35_behavior_runbook.md`

## 二、什么时候必须更新

以下场景必须更新：

1. 新实验跑出正式结果
2. 关键脚本逻辑发生变化
3. 当前优先级发生变化
4. 需要你手动做的事情发生变化
5. 某项任务从“待做”变成“已完成”
6. 新的人工标注入口、冻结资产或统一评测口径被建立
7. 默认检索配置或默认模型配置发生变化

## 三、各文件更新分工

### `01_current_progress.md`

更新内容：

- 当前阶段判断
- 已完成内容
- 最新实验结果
- 尚未完成项

### `02_next_steps.md`

更新内容：

- 下一步优先级
- 不建议提前启动的内容
- 当前最关键任务

### `03_manual_intervention_guide.md`

更新内容：

- 需要你手动操作的步骤
- 标注、检查、运行命令
- 常见错误提醒

## 四、当前维护约定

从现在开始，只要我继续在这个毕业设计项目上推进工作，我会同步维护这组文件。

换句话说，这个目录会作为：

- 你的当前进度总入口
- 你的下一步行动清单
- 你的人工介入操作手册

## 五、当前阶段的特别约定

当前阶段重点推进 `E3-E5`，其中：

- `E5` 已经完成正式结果
- `E3/E4` 的实现已经完成，但正式 A/B 运行改为在云端 GPU 环境执行
- 默认检索配置已经冻结为 `aspect_main_no_rerank`
- 当前 `fallback` 不进入主流程

因此只要下面这些状态发生变化，就要同步更新：

1. `Qwen3.5-2B / 4B / 9B` 的云端执行环境是否已经准备好
2. `E3` 是否已经跑出正式 `summary.csv` 与 `analysis.md`
3. `E4` 是否已经跑出正式 `summary.csv` 与 `analysis.md`
4. `experiments/labels/e4_clarification/clarification_question_audit.csv` 是否已经生成并被人工审计
5. `E3/E4/E5` 是否已经整理成论文可复用的行为章节材料
6. 默认检索配置或行为实验配置是否再次变化

## 六、建议你的使用方式

你可以把这组文件当作固定阅读顺序：

1. 先看 `01_current_progress.md`
2. 再看 `02_next_steps.md`
3. 真正开始手动操作前看 `03_manual_intervention_guide.md`
