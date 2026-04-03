# E10 / PEFT 执行手册

更新时间：2026-04-02

本手册只覆盖 `E10` 当前需要的最小执行闭环：

1. 准备 `SFT manifest`
2. 在云端训练 `Qwen/Qwen3.5-4B` 的 PEFT adapter
3. 回传 adapter 与 metadata
4. 在云端按同后端协议分别运行 base / peft
5. 在本地或云端生成正式 compare 报告
6. 在 `E10 v1` 正式负结果与 `E10 v2` 阶段性结果基础上，准备 `E10 v3` grounded manifest 并训练 `exp03`

当前不覆盖：

- 自动提交云端训练任务
- 多节点 / 多卡训练编排
- merged model 自动部署脚本

当前正式状态补充：

- `E10 v1` 正式 compare 已冻结为：
  - `experiments/runs/e10cmp_28598dfb8434c1ba_20260402T020734+0000/`
- 当前正式结论：
  - `PEFT exp01` 未优于 base
  - `PEFT exp02` 已追平 citation 但仍存在 schema 边界问题
  - 下一步进入 `E10 v3` 的数据+约束修复，而不是直接改训练超参

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

## 6.1 E10 v2 当前定位

`E10 v2` 当前已经完成，定位固定为：

- 阶段性改进结果
- 不是最终正结果
- 主要新问题：
  - `q018 / q022` 的 partial-support schema 失稳
  - `q085` 的 multi-hotel pack boundary 错误

可引用目录：

- `experiments/runs/e10_a2dd1a0bd73c57b5_20260402T073127+0000/`
- `experiments/runs/e10cmp_7cf0c9c0a9830796_20260402T074331+0000/`

## 6.2 E10 v3 如何开始

先在 strongest base 环境下生成 `v3` manifest：

```bash
python -m scripts.evaluation.run_experiment_suite --task e10_prepare_manifests_v3
```

然后在云端训练：

```bash
accelerate launch -m scripts.training.train_e10_peft \
  --config experiments/assets/e10_train_config.qwen35_4b_peft_v3.json
```

训练完成并 merge 后，只重跑 PEFT 组，再复用当前 base formal run 做 compare。

当前 `v3` 固定修复目标为：

- `q018 / q022`
  - `partial_support_keep_recommendation`
- `q085`
  - `multi_hotel_pack_boundary`
- grounded abstain
  - 只保留真正 evidence gap，不再混入 unsupported-request 驱动 abstain

## 6.3 v3 的最短执行顺序

```bash
python -m scripts.evaluation.run_experiment_suite --task e10_prepare_manifests_v3
accelerate launch -m scripts.training.train_e10_peft \
  --config experiments/assets/e10_train_config.qwen35_4b_peft_v3.json
python -m scripts.evaluation.run_experiment_suite \
  --task e10_base_vs_peft \
  --group-id B_peft_4b_grounded
python -m scripts.evaluation.run_experiment_suite \
  --task e10_compare_runs \
  --base-run-dir /abs/path/to/e10_0dc5c2e6f867c66f_20260402T015230+0000 \
  --peft-run-dir /abs/path/to/new_peft_v3_run
```

## 6.4 E10 v2 如何回看
先在 strongest base 环境下回看 `v2` manifest：

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

## 9. E10 v2 完整云端执行手册

### 9.0 v2 与 v1 的唯一差异

v2 只变数据，不变训练配方：

| 项目 | v1 (exp01) | v2 (exp02) |
| --- | --- | --- |
| 训练任务 | 4 类 (无 grounded) | 5 类 (加 grounded_recommendation) |
| 训练 manifest | `sft_train_manifest.jsonl` | `sft_train_manifest_v2.jsonl` |
| QLoRA 配方 | r=16, α=32, lr=2e-4 | 完全相同 |
| 评测资产 | 40 条冻结 E9 query | 完全相同 |
| 正式 base 对照组 | `e10_0dc5c2e6f867c66f` | 直接复用，不重跑 |

### 9.1 v2 本地验证状态（已完成）

以下内容已在本地验证通过：

- v2 grounded query pool：40 条可用 query，覆盖 10 城市、6 方面
- quiet_sleep 覆盖：6 条
- focus+avoid 覆盖：10 条
- 80 个 query-split pair 全部有候选酒店
- train 平均 4.8 候选/query，dev 平均 2.9 候选/query
- 51 个单元测试全部通过

### 9.2 第一步：云端同步仓库

```bash
cd /root/autodl-tmp/workspace/DatafinitiHotelReviews
git pull origin main
```

确认以下文件存在：

- `experiments/assets/e10_train_config.qwen35_4b_peft_v2.json`
- `scripts/evaluation/evaluate_e9_e10_generation.py`（含 `prepare_e10_manifests_v2`）
- `scripts/evaluation/run_experiment_suite.py`（含 `e10_prepare_manifests_v2` task）
- `experiments/assets/e9_generation_eval_units.jsonl`（冻结 E9 资产）
- `experiments/assets/e9_generation_eval_query_ids.json`

### 9.3 第二步：生成 v2 manifest

v2 manifest 生成需要 LLM 推理（用 base 模型为每条 query 生成 silver grounded_recommendation target），因此必须在 GPU 环境中运行。

```bash
source .venv-train/bin/activate
export BEHAVIOR_LLM_BACKEND=local
export BEHAVIOR_MODEL_ID=/root/autodl-tmp/models/base/Qwen3.5-4B
export BEHAVIOR_ENABLE_THINKING=false
unset BEHAVIOR_ADAPTER_METADATA_PATH
unset OPENAI_BASE_URL
unset OPENAI_API_KEY

python -m scripts.evaluation.run_experiment_suite --task e10_prepare_manifests_v2
```

成功后会生成三个文件：

- `experiments/assets/sft_train_manifest_v2.jsonl`
- `experiments/assets/sft_dev_manifest_v2.jsonl`
- `experiments/assets/sft_manifest_v2_report.json`

**检查要点：**

```bash
python3 -c "
import json
report = json.loads(open('experiments/assets/sft_manifest_v2_report.json').read())
print(json.dumps(report, indent=2, ensure_ascii=False))
"
```

确认：

- `train_grounded_record_count_raw > 0`（grounded 原始样本非空）
- `train_grounded_share_of_final_manifest >= 0.40`（grounded 占比达标）
- `dropped_reason_counts` 中无异常大量丢弃
- `train_grounded_slice_distribution` 中 `quiet_sleep`、`focus_avoid`、`partial_abstain` 均有覆盖

**如果 grounded pool 为空或过少：**

- 检查 `BEHAVIOR_MODEL_ID` 是否正确
- 检查 `BEHAVIOR_ENABLE_THINKING=false` 是否已设置
- 检查 base 模型是否能正常推理

### 9.4 第三步：训练 exp02

先做 dry-run：

```bash
accelerate launch -m scripts.training.train_e10_peft \
  --config experiments/assets/e10_train_config.qwen35_4b_peft_v2.json \
  --dry-run
```

确认输出中：

- `train_sample_count` 大于 v1 的 234（因为新增了 grounded 样本）
- `dev_sample_count` 大于 v1 的 54
- `output_paths` 指向 `exp02` 而不是 `exp01`

正式训练：

```bash
accelerate launch -m scripts.training.train_e10_peft \
  --config experiments/assets/e10_train_config.qwen35_4b_peft_v2.json
```

训练完成后检查：

```bash
ls -lah /root/autodl-tmp/models/adapters/qwen35_4b_qlora/exp02/
ls -lah /root/autodl-tmp/training/reports/qwen35_4b_qlora_exp02/
```

至少应看到：

- `adapter_config.json`、`adapter_model.safetensors`（adapter 权重）
- `adapter_metadata.json`（adapter 元信息）
- `train_summary.json`（训练汇总）

### 9.5 第四步：Merge adapter 并验证

```bash
python3 -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

base_path = '/root/autodl-tmp/models/base/Qwen3.5-4B'
adapter_path = '/root/autodl-tmp/models/adapters/qwen35_4b_qlora/exp02'
merged_path = '/root/autodl-tmp/models/merged/qwen35_4b_merged_exp02'

tokenizer = AutoTokenizer.from_pretrained(base_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(base_path, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map='cpu')
model = PeftModel.from_pretrained(model, adapter_path)
model = model.merge_and_unload()
model.save_pretrained(merged_path)
tokenizer.save_pretrained(merged_path)
print(f'[OK] Merged model saved to {merged_path}')
"
```

验证 merged 模型可加载：

```bash
python3 -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
path = '/root/autodl-tmp/models/merged/qwen35_4b_merged_exp02'
tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(path, trust_remote_code=True, torch_dtype='auto', device_map='auto')
print(f'[OK] Merged model loaded: {model.device}')
"
```

### 9.6 第五步：准备 v2 adapter metadata

从训练报告中获取 metadata 路径：

```bash
cat /root/autodl-tmp/training/reports/qwen35_4b_qlora_exp02/adapter_metadata.json
```

将该文件复制到仓库实验资产目录：

```bash
cp /root/autodl-tmp/training/reports/qwen35_4b_qlora_exp02/adapter_metadata.json \
   experiments/assets/e10_adapter_metadata.qwen35_4b_peft_v2.json
```

确认 metadata 中以下字段正确：

- `base_model_id` 包含 `Qwen3.5-4B`
- `task_types` 包含 `grounded_recommendation`
- `adapter_path` 指向 `exp02`

### 9.7 第六步：运行 PEFT v2 评测（只跑 B 组）

```bash
export BEHAVIOR_LLM_BACKEND=local
export BEHAVIOR_MODEL_ID=/root/autodl-tmp/models/merged/qwen35_4b_merged_exp02
export BEHAVIOR_ADAPTER_METADATA_PATH=experiments/assets/e10_adapter_metadata.qwen35_4b_peft_v2.json
export BEHAVIOR_ENABLE_THINKING=false
unset OPENAI_BASE_URL
unset OPENAI_API_KEY

python -m scripts.evaluation.run_experiment_suite \
  --task e10_base_vs_peft \
  --group-id B_peft_4b_grounded
```

成功后记录新的 run 目录路径：

```bash
NEW_PEFT_RUN=$(ls -td experiments/runs/e10_* | head -1)
echo "New PEFT v2 run: $NEW_PEFT_RUN"
```

### 9.8 第七步：生成 v2 compare 报告

复用 v1 已冻结的 base formal run：

```bash
python -m scripts.evaluation.run_experiment_suite \
  --task e10_compare_runs \
  --base-run-dir experiments/runs/e10_0dc5c2e6f867c66f_20260402T015230+0000 \
  --peft-run-dir "$NEW_PEFT_RUN"
```

### 9.9 第八步：检查 v2 compare 结果

```bash
NEW_CMP=$(ls -td experiments/runs/e10cmp_* | head -1)
cat "$NEW_CMP/analysis.md"
cat "$NEW_CMP/summary.csv"
```

重点关注：

| 指标 | 期望方向 |
| --- | --- |
| `citation_precision` | v2 PEFT >= base（或差距 <= 0.01） |
| `evidence_verifiability_mean` | v2 PEFT >= v1 PEFT |
| `schema_valid_rate` | 保持 1.0 |
| `reasoning_leak_rate` | 保持 0.0 |
| `auditable_query_rate` | >= 0.975 |

### 9.10 第九步：同步结果回本地

```bash
# 在云端打包新增结果
tar czf e10_v2_results.tar.gz \
  experiments/assets/sft_train_manifest_v2.jsonl \
  experiments/assets/sft_dev_manifest_v2.jsonl \
  experiments/assets/sft_manifest_v2_report.json \
  experiments/assets/e10_adapter_metadata.qwen35_4b_peft_v2.json \
  "$NEW_PEFT_RUN" \
  "$NEW_CMP"
```

在本地：

```bash
# 解压到仓库根目录
tar xzf e10_v2_results.tar.gz -C /path/to/DatafinitiHotelReviews/
```

### 9.11 v2 结果解读决策树

```text
v2 compare 结果
  ├── citation_precision 提升 >= +0.01
  │     → 正结果：v2 grounded data 有效
  │     → 写入论文：PEFT + grounded supervision 提升了引用准确性
  │
  ├── citation_precision 在 base ±0.01 范围内
  │     → 边界结果：v2 未明显退化也未明显提升
  │     → 写入论文：grounded supervision 消除了 v1 的退化，但提升有限
  │
  └── citation_precision 仍低于 base > 0.01
        → 第二轮负结果
        → 写入论文：当前 SFT 数据规模和质量不足以通过 PEFT 提升 grounded recommendation
        → 不再继续迭代训练，转入论文写作
```

## 10. 一句话版

当前 `E10 v2` 的正确推进方式是：在云端生成含 `grounded_recommendation` 的 v2 manifest，训练 `exp02`，merge 后只重跑 PEFT 组，复用已冻结的 base run 做 compare。

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
