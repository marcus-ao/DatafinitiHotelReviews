# 人工介入详细指南手册

更新时间：2026-03-31

本手册只写“现在仍然需要你亲自处理”的内容，并尽量细到可以一步一步照着做。

## 当前结论

当前最可能的人工阻塞点，已经从“qrels 标注”切换成了“云端 Base 模型执行安排”。

也就是说：

- `E1 / E2 / E5 / E6 / E7 / E8` 都已经有正式结果
- `E3 / E4` 的代码实现已经完成
- 现在最需要确认的是：如何按云端手册配置 AutoDL，并让 `Qwen3.5-2B / 4B / 9B` 进入正式对比
- 你已经明确要求：本地不再下载或缓存 `Qwen`

## 手册 A：你现在最该先做什么

### 第一步：不要再在本地下载任何 Qwen Base 模型

你需要明确一件事：

- `E3/E4` 的下一轮正式 Base 组已经切换为 `Qwen3.5-2B / 4B / 9B`
- 三个模型都将在云端有 GPU 的设备执行
- 当前本地缓存已经清理，不需要再重复下载

### 第二步：先按总手册把云端环境搭起来

- `docs/deployment/01_autodl_qwen35_behavior_runbook.md`

优先完成：

- 租 `vGPU-48GB`
- 扩数据盘
- 配缓存目录
- 安装 `vLLM` 和 `openai`
- 跑通 `Qwen3.5-2B / 4B / 9B` 的 API 冒烟验证

### 第三步：确认行为实验脚本已经接上 API backend

在正式跑 `E3/E4` 之前，你要先确认一件关键事实：

- 当前行为实验脚本现在已经支持通过 OpenAI-compatible API 调用云端 `vLLM`
- 你真正需要确认的是：云端环境变量和 `configs/params.yaml` 是否与服务地址一致
- 在正式跑之前，先做一轮最小 API 冒烟验证，确保返回结果里没有 `<think>`

### 第四步：接着正式跑 E3 与 E4

当 API backend 已接好后，再按统一顺序执行：

1. 启动 `Qwen3.5-2B`
2. 跑 `E3`
3. 跑 `E4`
4. 保存日志和 run 目录
5. 停服务
6. 换 `Qwen3.5-4B`
7. 重复同样步骤
8. 再换 `Qwen3.5-9B`

## 手册 B：E4 跑完后你需要人工看什么

### 步骤 1：打开 `clarification_question_audit.csv`

重点文件：

- `experiments/labels/e4_clarification/clarification_question_audit.csv`

你需要重点看这几列：

- `query_id`
- `group_id`
- `question`
- `answerable_score`
- `targeted_score`
- `notes`

### 步骤 2：只审计应澄清的正例

本轮设计里，这份表主要用于审计“该问的时候，问得好不好”。

你的关注点是：

- 这个问题能不能被用户直接回答
- 这个问题是不是正好问到了缺失槽位
- 有没有问得太泛、太绕或跑偏

### 步骤 3：简单打分口径

建议先用这个最小口径：

- `answerable_score`
  - `2`：用户一看就能答
  - `1`：勉强能答，但问法一般
  - `0`：不容易答，或问题本身不清楚
- `targeted_score`
  - `2`：准确问到了真正缺的槽位
  - `1`：部分相关，但不够聚焦
  - `0`：没有问到真正关键点

`notes` 里只需要写最短原因，例如：

- `问到了城市，且句子简洁`
- `问法太泛，没有指出冲突方面`
- `虽然应澄清，但问题没对准 city`

## 手册 C：如果 E3/E4 跑不起来，你要怎么判断问题

### 情况 1：云端环境一开始就报模型加载错误

这通常说明：

- 云端环境没有准备好 `Qwen3.5`
- 或 Hugging Face 权限 / 下载过程失败

这时不要继续怀疑实验脚本逻辑，优先把问题判断为：

- “云端 Base 模型不可用”

### 情况 2：`vLLM` 服务能通，但实验脚本仍失败

这通常说明：

- 云端模型服务本身可能没问题
- 当前真正阻塞的更可能是：
  - `OPENAI_BASE_URL` 配错
  - `BEHAVIOR_MODEL_ID` 与当前启动模型不一致
  - `BEHAVIOR_ENABLE_THINKING` 没按实验要求关闭

这时应该记录为：

- “部署已完成，但行为实验运行配置仍需校正后补跑”

### 情况 3：E4 跑完但没有生成审计文件

这就不是模型问题，而是结果落盘链路有问题。

这种情况需要立即回来看：

- `scripts/evaluation/evaluate_e3_e5_behavior.py`
- `experiments/labels/e4_clarification/`

## 手册 D：你现在暂时不用手动做的事

下面这些当前不用你马上手动做：

- 重新标 E1 gold
- 重新标 E6 qrels
- 重跑 E6/E7/E8
- 启动 PEFT
- 启动 `G1-G4`
- 把 fallback 接回主流程

## 手册 E：一句话版

不要再在本地下载 `Qwen`；先按 `docs/deployment/01_autodl_qwen35_behavior_runbook.md` 配好 AutoDL 上的 `Qwen3.5-2B / 4B / 9B`，确认 API 冒烟通过后就可以正式跑 `E3` 和 `E4`，跑完再人工审一下 `clarification_question_audit.csv`。
