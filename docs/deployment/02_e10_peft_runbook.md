# E10 / PEFT 执行手册

更新时间：2026-04-01

本手册只覆盖 `E10` 当前需要的最小执行闭环：

1. 准备 `SFT manifest`
2. 在云端训练 `Qwen/Qwen3.5-4B` 的 PEFT adapter
3. 回传 adapter 与 metadata
4. 在本地运行 `e10_base_vs_peft`

当前不覆盖：

- 自动提交云端训练任务
- 仓库内训练脚本
- bitsandbytes / QLoRA pipeline 的代码实现

## 1. 当前固定前提

- `E9` 第二轮正式结果已冻结为：
  - `experiments/runs/e9_ecbcdbab690dc503_20260401T025012+0000/`
- `E9` 冻结评测资产固定为：
  - `experiments/assets/e9_generation_eval_units.jsonl`
  - `experiments/assets/e9_generation_eval_query_ids.json`
- `E10` 不修改 retrieval 主线：
  - `aspect_main_no_rerank`
  - `fallback=false`
  - `E2 B_final_aspect_score Top5`
- `E10` 的 base 模型固定为：
  - `Qwen/Qwen3.5-4B`

## 2. 本地先做什么

在仓库根目录执行：

```bash
source venv/bin/activate
python -m scripts.evaluation.run_experiment_suite --task e10_prepare_manifests
```

确认以下文件存在：

- `experiments/assets/sft_train_manifest.jsonl`
- `experiments/assets/sft_dev_manifest.jsonl`
- `experiments/assets/e10_train_config_template.json`
- `experiments/assets/e10_adapter_metadata.template.json`

## 3. 云端训练前要固定的输入

训练时只允许使用：

- `experiments/assets/sft_train_manifest.jsonl`
- `experiments/assets/sft_dev_manifest.jsonl`

训练目标只允许四类：

- `preference_parse`
- `clarification`
- `constraint_honesty`
- `feedback_update`

当前不加入：

- `grounded_recommendation`

## 4. 云端训练后要回传什么

训练完成后，至少回传：

- adapter 导出目录
- 最终使用的训练配置
- 一个填写完成的 adapter metadata 文件

metadata 可从：

- `experiments/assets/e10_adapter_metadata.template.json`

复制一份后填写。

必须保证：

- `base_model_id = Qwen/Qwen3.5-4B`
- `served_model_id` 为当前推理端点实际可调用的 PEFT model id
- `adapter_path` 指向真实 adapter 导出目录
- `backend = api`

## 5. 本地如何运行 E10

先准备环境变量：

```bash
export OPENAI_BASE_URL=http://127.0.0.1:8000/v1
export OPENAI_API_KEY=EMPTY
export BEHAVIOR_ADAPTER_METADATA_PATH=/absolute/path/to/final_adapter_metadata.json
```

然后运行：

```bash
python -m scripts.evaluation.run_experiment_suite --task e10_base_vs_peft
```

## 6. 运行后检查什么

成功后会产出：

- `experiments/runs/e10_*/run_meta.json`
- `experiments/runs/e10_*/results.jsonl`
- `experiments/runs/e10_*/summary.csv`
- `experiments/runs/e10_*/analysis.md`
- `experiments/runs/e10_*/citation_verifiability_audit.csv`

重点检查：

- `A_base_4b_grounded` 与 `B_peft_4b_grounded` 是否都产出完整结果
- `summary.csv` 中五项主指标是否可直接对照：
  - `citation_precision`
  - `evidence_verifiability_mean`
  - `unsupported_honesty_rate`
  - `schema_valid_rate`
  - `avg_latency_ms`

## 7. 常见报错判断

### 情况 1：缺少 adapter metadata

说明：

- `BEHAVIOR_ADAPTER_METADATA_PATH` 没设
- 或 metadata 文件路径错误

### 情况 2：base model 不匹配

说明：

- metadata 中的 `base_model_id` 不是 `Qwen/Qwen3.5-4B`
- 当前 adapter 不属于本主线

### 情况 3：served model 不可用

说明：

- `served_model_id` 与当前 API 实际部署的名称不一致
- 或 PEFT served model 尚未成功部署

## 8. 一句话版

当前 `E10` 的正确推进方式是：固定 `E9` 资产不动，在云端训练 `Qwen/Qwen3.5-4B` 的 PEFT adapter，回传并填写 adapter metadata，然后在本地直接运行 `e10_base_vs_peft` 做正式对照。
