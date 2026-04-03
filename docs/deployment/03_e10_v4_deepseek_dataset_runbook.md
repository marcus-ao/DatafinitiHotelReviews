# E10 v4 人工设计 + DeepSeek Reasoner 辅助生成数据集执行手册

更新时间：2026-04-03

本手册只覆盖 `E10 v4` 的数据集侧执行闭环，不包含正式 compare 解读。

当前 `v4` 的基本定位固定为：

- 人工定义 slice、边界和质检规则
- DeepSeek Reasoner 辅助生成 query draft / target draft
- accepted grounded 只有通过自动质检和人工复核后才能进入正式 manifest
- 不改 retrieval、不改正式 base、不改 compare 协议

## 1. 先生成 v4 seed specs 和资产骨架

如果你之前已经生成过带 `glm` 痕迹的 `v4` 资产，先执行一次无损迁移：

```bash
source venv/bin/activate
python -m scripts.evaluation.run_experiment_suite --task e10_migrate_deepseek_assets_v4
```

这一步会把现有 `seed_specs / query_requests / target_requests / drafts / report` 中残留的
历史字段统一迁移为 DeepSeek 命名，不会清空你已经生成的草稿。

在仓库根目录执行：

```bash
source venv/bin/activate
python -m scripts.evaluation.run_experiment_suite --task e10_prepare_seed_specs_v4
```

执行后应生成：

- `experiments/assets/e10_v4_seed_specs.jsonl`
- `experiments/assets/e10_v4_gold_patch.jsonl`
- `experiments/assets/e10_v4_deepseek_drafts.jsonl`
- `experiments/assets/e10_v4_review_log.csv`
- `experiments/assets/e10_v4_accepted_grounded.jsonl`
- `experiments/assets/e10_v4_deepseek_prompt_templates.json`

其中：

- `seed_specs`
  - 是这轮 `v4` 的固定配额与证据输入边界
- `gold_patch`
  - 用于人工直写 gold grounded 样本
- `deepseek_drafts`
  - 用于保存 DeepSeek 生成的草稿
- `review_log`
  - 用于保存双层质检记录
- `accepted_grounded`
  - 只保留最终接受样本

## 1.1 生成可直接喂给 DeepSeek 的请求包

先生成 query 阶段请求：

```bash
python -m scripts.evaluation.run_experiment_suite --task e10_prepare_deepseek_query_requests_v4
```

输出：

- `experiments/assets/e10_v4_deepseek_query_requests.jsonl`

然后直接用 DeepSeek Reasoner 批量生成 query draft：

```bash
python -m scripts.evaluation.run_e10_v4_deepseek_generation \
  --stage query \
  --input-path experiments/assets/e10_v4_deepseek_query_requests.jsonl
```

如果你当前只跑 pilot：

```bash
python -m scripts.evaluation.run_e10_v4_deepseek_generation \
  --stage query \
  --input-path experiments/assets/e10_v4_deepseek_query_requests.pilot.jsonl
```

生成结果会写回：

- `experiments/assets/e10_v4_deepseek_drafts.jsonl`

其中已接受的 query 草稿需标记：

- `source_mode = silver_deepseek`
- `review_status = query_accepted`
- `seed_id`
- `query_text_zh`

然后生成 target 阶段请求：

```bash
python -m scripts.evaluation.run_experiment_suite --task e10_prepare_deepseek_target_requests_v4
```

输出：

- `experiments/assets/e10_v4_deepseek_target_requests.jsonl`

再调用 DeepSeek Reasoner 生成 target draft：

```bash
python -m scripts.evaluation.run_e10_v4_deepseek_generation \
  --stage target \
  --input-path experiments/assets/e10_v4_deepseek_target_requests.jsonl
```

## 2. DeepSeek 生成时的固定原则

使用：

- `experiments/assets/e10_v4_seed_specs.jsonl`
- `experiments/assets/e10_v4_deepseek_prompt_templates.json`

固定两阶段：

1. `query_draft`
2. `target_draft`

要求：

- `query_draft` 只生成自然中文 query
- `target_draft` 只生成 `RecommendationResponse` 兼容 JSON
- `DeepSeek` 型号名必须记录到：
  - `provenance.generator_model_name`

### 2.1 DeepSeek 配置来源

批量脚本：

- `scripts/evaluation/run_e10_v4_deepseek_generation.py`

会自动尝试加载：

- 当前工作目录下 `.env`
- 项目根目录 `.env`
- 项目上一层目录 `.env`

并读取以下环境变量：

- `DEEPSEEK_API_KEY`
- `DEEPSEEK_BASE_URL`
- `DEEPSEEK_REASONER_MODEL`

兼容回退：

- `DEEPSEEK_MODEL`
- `OPENAI_API_KEY`
- `OPENAI_BASE_URL`

默认值：

- `base_url = https://api.deepseek.com`
- `model = deepseek-reasoner`

也就是说：

- API Key / Base URL / Model 不需要写进仓库配置文件
- 只要你的 `.env` 或 shell 环境中有这些值，脚本就会直接使用

## 3. 人工质检与 accepted 入库

accepted 样本必须写入：

- `experiments/assets/e10_v4_accepted_grounded.jsonl`

review 记录必须写入：

- `experiments/assets/e10_v4_review_log.csv`

accepted 数据必须满足：

- `review_status = accepted`
- `accepted_version = v4`
- `source_mode = gold_manual | silver_deepseek`
- `primary_slice` 属于固定 6 类之一
- `accepted_target_payload` 已完成最终修订

当前固定第二轮 review 规则：

- 全部 accepted 样本必须有 `r1`
- 至少 `20%` accepted 样本必须有 `r2`
- `partial_support_keep_recommendation`
- `multi_hotel_pack_boundary`
  - 两类样本至少 `30%` 进入 `r2`

## 4. 从 accepted 生成正式 v4 manifest

先执行：

```bash
python -m scripts.evaluation.run_experiment_suite --task e10_prepare_manifests_v4
```

再验证：

```bash
python -m scripts.evaluation.run_experiment_suite --task e10_validate_manifest_v4
```

成功后应生成：

- `experiments/assets/sft_train_manifest_v4.jsonl`
- `experiments/assets/sft_dev_manifest_v4.jsonl`
- `experiments/assets/sft_manifest_v4_report.json`

## 5. 训练 exp04

云端训练前，先把 `v4` manifest 与 config 同步到云端。

训练命令固定为：

```bash
accelerate launch -m scripts.training.train_e10_peft \
  --config experiments/assets/e10_train_config.qwen35_4b_peft_v4.json
```

训练完成后：

1. merge `exp04`
2. 只重跑：
   - `B_peft_4b_grounded`
3. 复用当前 base formal run 做 compare

## 6. 当前最重要的检查项

在进入训练前，优先确认：

- `dataset_profile`
  - `pilot` 或 `full`
- `accepted_count`
- `primary_slice_distribution`
- `source_mode_distribution`
- `review_round_2_coverage`
- `slice_review_round_2_coverage`
- `max_accepted_per_seed`
- `hotel_split_distribution.test = 0`

如果这些检查没有通过，不要启动 `exp04`。
