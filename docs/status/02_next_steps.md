# 下一步该做什么

更新时间：2026-03-31

## 当前推荐推进顺序

请按下面顺序推进，不建议跳步：

1. 确认并保持默认检索配置固定为 `aspect_main_no_rerank`
2. 按云端手册准备 `Qwen3.5-2B / 4B / 9B` 的 AutoDL 执行环境
3. 完成 `E3` 正式多模型对比运行
4. 完成 `E4` 正式多模型对比运行与问题质量审计
5. 整理 `E3/E4/E5` 论文材料
6. 再进入更后面的 `E9/E10`、PEFT 与主实验矩阵

## 第一优先级：立即要做

### 任务 1：准备云端 Qwen3.5 多模型执行环境

目标：

- 按 `docs/deployment/01_autodl_qwen35_behavior_runbook.md` 在 AutoDL 上准备 `Qwen3.5-2B / 4B / 9B` 的轮流执行环境

完成标准：

- 明确设备为 `vGPU-48GB`
- 数据盘至少扩到 `100GB`
- 三个模型都能完成 `vLLM` 服务冒烟验证
- `enable_thinking=False` 的控制方式已确认

### 任务 2：跑通 E3 正式实验

目标：

- 在已经冻结的检索后端上，完成规则组与 `Qwen3.5-2B / 4B / 9B` 的正式对照

完成标准：

- 产生一轮正式 `experiments/runs/e3_*/`
- 输出：
  - `run_meta.json`
  - `results.jsonl`
  - `summary.csv`
  - `analysis.md`
- `summary.csv` 至少包含：
  - `Slot-F1`
  - `Exact-Match Rate`
  - `Unsupported Detection Recall`
  - `Schema Valid Rate`

### 任务 3：跑通 E4 正式实验

目标：

- 完成规则组与 `Qwen3.5-2B / 4B / 9B` 的正式对照，并生成问题质量审计入口

完成标准：

- 产生一轮正式 `experiments/runs/e4_*/`
- 输出：
  - `run_meta.json`
  - `results.jsonl`
  - `summary.csv`
  - `analysis.md`
- 额外生成：
  - `experiments/labels/e4_clarification/clarification_question_audit.csv`

### 任务 4：检查 E3/E4 的云端执行链路是否稳定

当前需要特别关注：

- 不再把“本地模型不可用”当作待解决项
- 现在真正需要确认的是：
  - 云端环境是否能稳定加载 `Qwen3.5`
  - `vLLM` 服务是否能稳定返回 non-thinking 输出
  - 行为实验脚本是否能通过 API backend 稳定落盘
  - run 目录是否能同步回本仓库

## 第二优先级：E3/E4 跑完后立刻做

### 任务 5：整理 E3/E4/E5 论文材料

目标：

- 将三组行为实验写成可直接复用的结果表、错误分析与设计结论

建议最少形成：

- `1` 张 E3 结果表
- `1` 张 E4 结果表
- `1` 张 E5 结果表
- `2-3` 组典型错误/边界案例
- `1` 段“固定后端条件下的行为层结论”

### 任务 6：把行为实验与前面的 Aspect-KB 章节串起来

建议整体叙述顺序：

1. `E1`：标签可靠性
2. `E2`：画像候选缩圈
3. `E6-E8`：检索组织方式与边界
4. `E5`：中英桥接必要性
5. `E3/E4`：偏好解析与澄清触发

## 当前不建议启动的内容

在下面这些条件未满足前，不建议提前进入：

- `E3/E4` 还没正式跑完前：不要启动 PEFT
- 行为实验章节还没收口前：不要启动 `G1-G4`
- 当前阶段：不要把 `reranker` 或 `fallback` 再接回默认主流程
- 当前阶段：不要写“系统全链路已经充分验证优于所有基线”这一类过强结论

## 一句话版本

你现在最该做的，是先按云端手册把 `Qwen3.5-2B / 4B / 9B` 的 AutoDL 执行环境准备好，然后在已冻结的默认检索后端上跑出 `E3/E4` 正式多模型结果；`E5` 已经完成，后面再把 `E3/E4/E5` 一起收口成行为章节材料。
