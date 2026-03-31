# 人工介入详细指南手册

更新时间：2026-03-31

本手册只写“现在仍然需要你亲自处理”的内容，并尽量细到可以一步一步照着做。

## 当前结论

当前最可能的人工阻塞点，已经从“云端诊断重跑”切换成了“最新 `E4` 审计文件的人工复核，以及是否还要追加 `Qwen3.5-9B` 扩展对比”。

也就是说：

- `E1 / E2 / E5 / E6 / E7 / E8` 都已经有正式结果
- `Qwen3.5-2B` 的第一轮 `E3/E4` baseline 已经归档
- `Qwen3.5-4B` 的全量 `E3/E4` 正式结果已经完成
- 现在最需要确认的是：把最新 `E4` 审计文件补齐，并决定是否还需要 `9B` 作为附录对比
- 你已经明确要求：本地不再下载或缓存 `Qwen`

## 手册 A：你现在最该先做什么

### 第一步：不要再在本地下载任何 Qwen Base 模型

你需要明确一件事：

- `E3/E4` 当前正式 Base 组已经固定为 `Qwen3.5-4B`
- 若要补做 `Qwen3.5-9B`，也仍然应在云端 GPU 设备执行
- 当前本地缓存已经清理，不需要再重复下载

### 第二步：保持当前 baseline 和正式 run 都不动

你现在不要做的是：

- 不要覆盖：
  - `experiments/runs/e3_244aca8abf6345ad_20260331T072527+0000/`
  - `experiments/runs/e4_4a15a89128a90d11_20260331T073016+0000/`
- 不要覆盖：
  - `experiments/runs/e3_14928d821d811e86_20260331T122611+0000/`
  - `experiments/runs/e4_55c8021e1119fb77_20260331T122648+0000/`
- 不要删除：
  - `experiments/reports/03_behavior_stage_1_qwen35_2b_baseline.md`
  - `experiments/reports/04_behavior_stage_2_qwen35_4b_formal_summary.md`

### 第三步：先人工审阅最新 E4 审计文件

你现在最值得亲自处理的文件是：

- `experiments/labels/e4_clarification/clarification_question_audit.csv`
- `experiments/runs/e4_55c8021e1119fb77_20260331T122648+0000/clarification_question_audit.csv`
- `experiments/labels/e4_clarification/clarification_question_audit_e4_55c8021e1119fb77_qwen35_4b_full.csv`

优先完成：

- 补全 `answerable_score`
- 补全 `targeted_score`
- 在 `notes` 中记下最值得进论文的案例

### 第四步：确认当前默认配置不要再漂移

在继续任何实验或写作之前，你要先确认这几个事实仍成立：

- 默认检索模式：`aspect_main_no_rerank`
- fallback：`false`
- 当前正式行为模型：`Qwen/Qwen3.5-4B`
- `E3` prompt：`e3_v2_cn_slots_only`
- `E4` prompt：`e4_v2_cn_decision_label_fewshot`

### 第五步：如果你确实还想补模型规模对比，再回到云端追加 9B

只有在你明确需要“模型规模附录”时，才做这一步。

建议顺序：

1. 按 `docs/deployment/01_autodl_qwen35_behavior_runbook.md` 启动 `Qwen3.5-9B`
2. 先做 API 冒烟验证
3. 跑 `E3`
4. 跑 `E4`
5. 把新的 run 目录同步回本地
6. 与当前 `4B` 正式结果横向比较

建议命令：

```bash
export BEHAVIOR_MODEL_ID=Qwen/Qwen3.5-9B
python -m scripts.evaluation.run_experiment_suite --task e3_preference
python -m scripts.evaluation.run_experiment_suite --task e4_clarification
```

注意：

- 不要因为换到 `9B` 就顺手改 prompt
- 不要把 `reranker` 或 `fallback` 临时接回主流程
- `9B` 结果只应作为附录或扩展对比，不覆盖 `4B` 正式结论

## 手册 B：E4 跑完后你需要人工看什么

### 步骤 1：打开 `clarification_question_audit.csv`

重点文件：

- 最新副本：`experiments/labels/e4_clarification/clarification_question_audit.csv`
- 当前正式 run 真值文件：`experiments/runs/e4_55c8021e1119fb77_20260331T122648+0000/clarification_question_audit.csv`
- 当前冻结快照：`experiments/labels/e4_clarification/clarification_question_audit_e4_55c8021e1119fb77_qwen35_4b_full.csv`

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

- 云端环境没有准备好当前要跑的 `Qwen3.5-4B` 或 `Qwen3.5-9B`
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
- 对应新的 `experiments/runs/e4_*/clarification_question_audit.csv`

## 手册 D：你现在暂时不用手动做的事

下面这些当前不用你马上手动做：

- 重新标 E1 gold
- 重新标 E6 qrels
- 重跑 E6/E7/E8
- 重跑 `Qwen3.5-2B` 诊断子集
- 启动 `G1-G4`
- 把 fallback 接回主流程
- 立刻启动 PEFT

## 手册 E：一句话版

不要再在本地下载 `Qwen`；先保留好已经归档的 `Qwen3.5-2B` baseline 和已经完成的 `Qwen3.5-4B` 正式 run，优先把最新 `E4` 审计文件补齐并整理行为实验论文材料，只有确实需要规模附录时再按云端手册追加 `Qwen3.5-9B` 对比。
