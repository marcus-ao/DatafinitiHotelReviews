# E10 / PEFT 执行手册

更新时间：2026-04-02

本手册只覆盖 `E10` 当前需要的最小执行闭环：

1. 准备 `SFT manifest`
2. 在云端训练 `Qwen/Qwen3.5-4B` 的 PEFT adapter
3. 回传 adapter 与 metadata
4. 在云端按同后端协议分别运行 base / peft
5. 在本地或云端生成正式 compare 报告
6. 在 `E10 v1` 正式负结果基础上，准备 `E10 v2` grounded manifest 并训练 `exp02`

当前不覆盖：

- 自动提交云端训练任务
- 多节点 / 多卡训练编排
- merged model 自动部署脚本

当前正式状态补充：

- `E10 v1` 正式 compare 已冻结为：
  - `experiments/runs/e10cmp_28598dfb8434c1ba_20260402T020734+0000/`
- 当前正式结论：
  - `PEFT exp01` 未优于 base
  - 下一步进入 `E10 v2` 的数据方案，而不是直接改训练超参

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
- `scripts/training/train_e10_peft.py`
- `scripts/training/training_utils.py`

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

当前推荐训练框架固定为：

- `transformers`
- `peft`
- `trl`
- `bitsandbytes`
- `accelerate`

当前正式训练入口固定为：

```bash
accelerate launch -m scripts.training.train_e10_peft \
  --config experiments/assets/e10_train_config.qwen35_4b_peft_v1.json
```

## 4. 云端训练后要回传什么

训练完成后，至少回传：

- adapter 导出目录
- 最终使用的训练配置
- 一个填写完成的 adapter metadata 文件
- `adapter_metadata.json`
- `train_summary.json`

metadata 可从：

- `experiments/assets/e10_adapter_metadata.template.json`

复制一份后填写。

必须保证：

- `base_model_id = Qwen/Qwen3.5-4B`
- `served_model_id` 为当前 PEFT 实验标识
- `adapter_path` 指向真实 adapter 导出目录
- metadata 仅用于标识 adapter，不再要求正式结果通过 API serving 路径完成

## 5. 云端如何正式启动训练

先进入项目目录并激活环境：

```bash
cd /root/autodl-tmp/workspace/DatafinitiHotelReviews
source .venv/bin/activate
```

安装训练依赖：

```bash
pip install -U accelerate bitsandbytes peft trl datasets
```

初始化 accelerate：

```bash
accelerate config default
```

建议先做 dry-run：

```bash
accelerate launch -m scripts.training.train_e10_peft \
  --config experiments/assets/e10_train_config.qwen35_4b_peft_v1.json \
  --dry-run
```

dry-run 没问题后，再正式训练：

```bash
accelerate launch -m scripts.training.train_e10_peft \
  --config experiments/assets/e10_train_config.qwen35_4b_peft_v1.json
```

训练完成后，重点检查：

```bash
ls -lah /root/autodl-tmp/models/adapters/qwen35_4b_qlora/exp01
ls -lah /root/autodl-tmp/training/checkpoints/qwen35_4b_qlora_exp01
ls -lah /root/autodl-tmp/training/logs/qwen35_4b_qlora_exp01
ls -lah /root/autodl-tmp/training/reports/qwen35_4b_qlora_exp01
```

其中至少应看到：

- adapter 权重目录
- `adapter_metadata.json`
- `train_summary.json`

## 6. 正式 E10 如何运行

正式论文结果只接受同后端协议：

- 两组都在云端 `.venv-train` 中执行
- 两组都使用 `BEHAVIOR_LLM_BACKEND=local`
- 两组都使用同一套冻结 `E9` eval units

### Base formal

```bash
cd /root/autodl-tmp/workspace/DatafinitiHotelReviews
source .venv-train/bin/activate
export BEHAVIOR_LLM_BACKEND=local
export BEHAVIOR_MODEL_ID=/root/autodl-tmp/models/base/Qwen3.5-4B
unset BEHAVIOR_ADAPTER_METADATA_PATH
unset OPENAI_BASE_URL
unset OPENAI_API_KEY
python -m scripts.evaluation.run_experiment_suite --task e10_base_vs_peft --group-id A_base_4b_grounded
```

### PEFT formal

```bash
cd /root/autodl-tmp/workspace/DatafinitiHotelReviews
source .venv-train/bin/activate
export BEHAVIOR_LLM_BACKEND=local
export BEHAVIOR_MODEL_ID=/root/autodl-tmp/models/merged/qwen35_4b_merged_exp01
export BEHAVIOR_ADAPTER_METADATA_PATH=experiments/assets/e10_adapter_metadata.qwen35_4b_peft_v1.json
unset OPENAI_BASE_URL
unset OPENAI_API_KEY
python -m scripts.evaluation.run_experiment_suite --task e10_base_vs_peft --group-id B_peft_4b_grounded
```

### Compare 报告

```bash
python -m scripts.evaluation.run_experiment_suite \
  --task e10_compare_runs \
  --base-run-dir /abs/path/to/base_run \
  --peft-run-dir /abs/path/to/peft_run
```

## 6.1 E10 v2 如何开始

先在 strongest base 环境下生成 `v2` manifest：

```bash
python -m scripts.evaluation.run_experiment_suite --task e10_prepare_manifests_v2
```

然后在云端训练：

```bash
accelerate launch -m scripts.training.train_e10_peft \
  --config experiments/assets/e10_train_config.qwen35_4b_peft_v2.json
```

训练完成并 merge 后，只重跑 PEFT 组，再复用当前 base formal run 做 compare。

## 7. 运行后检查什么

单组运行成功后会产出：

- `experiments/runs/e10_*/run_meta.json`
- `experiments/runs/e10_*/results.jsonl`
- `experiments/runs/e10_*/summary.csv`
- `experiments/runs/e10_*/analysis.md`
- `experiments/runs/e10_*/citation_verifiability_audit.csv`

正式 compare 成功后会新增：

- `experiments/runs/e10cmp_*/run_meta.json`
- `experiments/runs/e10cmp_*/comparison.jsonl`
- `experiments/runs/e10cmp_*/summary.csv`
- `experiments/runs/e10cmp_*/analysis.md`

重点检查：

- 两个单组 run 是否都完成且 `schema_valid_rate > 0`
- `reasoning_leak_rate` 是否为 `0.0`
- compare 报告中以下主指标是否可直接对照：
  - `citation_precision`
  - `evidence_verifiability_mean`
  - `unsupported_honesty_rate`
  - `schema_valid_rate`
  - `avg_latency_ms` 仅在 compare 报告明确标记 `latency_formally_comparable=yes` 时才纳入正式结论

## 8. 常见报错判断

### 情况 1：云端训练命令启动失败

说明：

- 缺少 `accelerate / bitsandbytes / peft / trl / datasets`
- 或 base model 路径不存在

### 情况 2：缺少 adapter metadata

说明：

- `BEHAVIOR_ADAPTER_METADATA_PATH` 没设
- 或 metadata 文件路径错误

### 情况 3：base model 不匹配

说明：

- metadata 中的 `base_model_id` 不是 `Qwen/Qwen3.5-4B`
- 当前 adapter 不属于本主线

### 情况 4：PEFT 本地直载输出 Thinking Process

说明：

- 本地 backend 没有正确关闭 thinking / reasoning 输出
- 当前 run 只能视为诊断结果，不能进入正式 compare
- 或 PEFT served model 尚未成功部署

## 9. 一句话版

当前 `E10` 的正确推进方式是：固定 `E9` 资产不动，在云端直接运行
`accelerate launch -m scripts.training.train_e10_peft --config experiments/assets/e10_train_config.qwen35_4b_peft_v1.json`
训练 `Qwen/Qwen3.5-4B` 的 PEFT adapter，回传并填写 adapter metadata。由于当前单卡显存和 `merged PEFT + vLLM` 兼容性限制，`E10` 正式对照改为：

- `Base` 组：云端 `vLLM` 服务 + 本地评测
- `PEFT` 组：云端本地直载 merged 模型直接评测

## 10. E10 单服务分时运行

当前默认不要同时启动两个服务。

原因：

- `Qwen/Qwen3.5-4B` base 服务已经会占用绝大部分 48GB 显存
- merged PEFT 服务再起第二个实例时会因为显存不足而失败

所以当前标准流程固定为：

1. 启动 Base 服务
2. 本地只跑 `A_base_4b_grounded`
3. 停掉 Base 服务
4. 启动 PEFT merged 服务
5. 本地只跑 `B_peft_4b_grounded`

### Base 组

云端启动：

```bash
source .venv-serve/bin/activate
export OPENAI_API_KEY=EMPTY

vllm serve /root/autodl-tmp/models/base/Qwen3.5-4B \
  --served-model-name Qwen/Qwen3.5-4B \
  --port 8000 \
  --tensor-parallel-size 1 \
  --max-model-len 8192 \
  --reasoning-parser qwen3 \
  --language-model-only
```

本地运行：

```bash
export OPENAI_BASE_URL=http://<cloud-host>:8000/v1
export OPENAI_API_KEY=EMPTY
unset BEHAVIOR_ADAPTER_METADATA_PATH
unset BEHAVIOR_MODEL_ID

python -m scripts.evaluation.run_experiment_suite \
  --task e10_base_vs_peft \
  --group-id A_base_4b_grounded
```

### PEFT 组

当前不再要求启动 `merged PEFT` 的 `vLLM` 服务。

在云端 `.venv-train` 中直接评测：

```bash
cd /root/autodl-tmp/workspace/DatafinitiHotelReviews
source .venv-train/bin/activate
export BEHAVIOR_LLM_BACKEND=local
export BEHAVIOR_MODEL_ID=/root/autodl-tmp/models/merged/qwen35_4b_merged_exp01
export BEHAVIOR_ADAPTER_METADATA_PATH=experiments/assets/e10_adapter_metadata.qwen35_4b_peft_v1.json

python -m scripts.evaluation.run_experiment_suite \
  --task e10_base_vs_peft \
  --group-id B_peft_4b_grounded
```

说明：

- 这一步直接在云端生成 `B_peft_4b_grounded` 的 `e10_*` run
- 跑完后，把该 run 同步回本地工作区即可
