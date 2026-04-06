更新时间：2026-04-05

本手册只写“现在仍然需要你亲自处理”的内容，并尽量细到可以一步一步照着做。

## 当前结论

当前最需要你亲自处理的工作，已经从“保留旧实验、只补 G”切换成了“按统一协议正式重跑 `E1-E10 + G1-G4`”。

> 当前正式 decisive G scope 已按 Protocol A 调整为 `68` 条查询（`39 core + 29 robustness`），`q021 / q024` 因在 frozen aspect mainline 下无法形成 evidence-backed candidate，已转为 supporting boundary cases，不再纳入 decisive matrix。

也就是说：

- `E1-E10` 历史结果继续保留，但仅作为 archive 参考
- `E1-E10 + G1-G4` 本轮都需要重新产出 canonical 结果
- `G` 系列闭环仍然是最后一段 decisive pipeline，但不是唯一需要执行的部分

## 手册 A：上云前必须先确认什么

### 第一步：确认 `exp02` metadata

当前 `G3 / G4` 默认使用 `exp02 / v2`，所以你需要先确认：

- `experiments/assets/e10_adapter_metadata.qwen35_4b_peft_v2.json`

如果这个文件还没同步回来：

- 先不要启动 `G3 / G4`
- 先从云端或历史归档中补回真实 metadata

当前仓库里已存在的只是：

- `e10_adapter_metadata.qwen35_4b_peft_v1.json`
- `e10_adapter_metadata.qwen35_4b_peft_v3.json`
- `e10_adapter_metadata.qwen35_4b_peft_v4.json`

### 第二步：确认 G 查询资产

当前应先确认：

- `experiments/assets/g_eval_query_ids_68.json`

如果你怀疑它不是最新版本，可重新生成：

```bash
python -m scripts.evaluation.run_experiment_suite --task g_build_query_ids_70
```

### 第三步：确认正式 G retrieval assets 是否已存在

当前目标资产是：

- `experiments/assets/g_plain_generation_eval_units.jsonl`
- `experiments/assets/g_aspect_generation_eval_units.jsonl`

如果这两份还不存在，就先生成：

```bash
python -m scripts.evaluation.run_experiment_suite --task g_freeze_plain_retrieval_assets
python -m scripts.evaluation.run_experiment_suite --task g_freeze_aspect_retrieval_assets
```

## 手册 B：云端如何按顺序跑 `E1-E10 + G1-G4`

### 建议顺序

1. 先在本地完成 `E1-E10` 所需冻结资产与配置确认
2. 本地或云端重跑 `E1-E4`
3. 本地或云端重跑 `E5-E8`
4. 本地或云端重跑 `E9-E10`
5. 最后在云端跑 `G1-G4`

### 为什么 G 仍然放在最后

因为本轮正式结果中，`G1-G4` 要承担 decisive matrix 角色，而 `E1-E10` 也必须先形成新的 canonical supporting / prerequisite 结果，后续 registry 与论文引用才能完全统一。

## 手册 C：云端如何跑 G1-G4

### G1 / G2：Base 组

在云端进入项目并激活环境后，先设置 base 模型：

```bash
cd /root/autodl-tmp/workspace/DatafinitiHotelReviews
source .venv-train/bin/activate

export BEHAVIOR_LLM_BACKEND=local
export BEHAVIOR_MODEL_ID=/root/autodl-tmp/models/base/Qwen3.5-4B
export BEHAVIOR_ENABLE_THINKING=false
unset BEHAVIOR_ADAPTER_METADATA_PATH
unset OPENAI_BASE_URL
unset OPENAI_API_KEY
```

然后依次运行：

```bash
python -m scripts.evaluation.run_experiment_suite --task g_run_generation --group-id G1
python -m scripts.evaluation.run_experiment_suite --task g_run_generation --group-id G2
```

### G3 / G4：PEFT 组

先确保 `exp02` merged model 与 metadata 都在云端可用，然后设置：

```bash
export BEHAVIOR_LLM_BACKEND=local
export BEHAVIOR_MODEL_ID=/root/autodl-tmp/models/merged/qwen35_4b_merged_exp02
export BEHAVIOR_ADAPTER_METADATA_PATH=experiments/assets/e10_adapter_metadata.qwen35_4b_peft_v2.json
export BEHAVIOR_ENABLE_THINKING=false
unset OPENAI_BASE_URL
unset OPENAI_API_KEY
```

然后运行：

```bash
python -m scripts.evaluation.run_experiment_suite --task g_run_generation --group-id G3
python -m scripts.evaluation.run_experiment_suite --task g_run_generation --group-id G4
```

## 手册 D：云端如何生成关键 compare

至少要跑下面四组：

```bash
python -m scripts.evaluation.run_experiment_suite --task g_compare_runs --left-run-dir /abs/path/to/G1_run --right-run-dir /abs/path/to/G2_run --left-label G1 --right-label G2
python -m scripts.evaluation.run_experiment_suite --task g_compare_runs --left-run-dir /abs/path/to/G3_run --right-run-dir /abs/path/to/G4_run --left-label G3 --right-label G4
python -m scripts.evaluation.run_experiment_suite --task g_compare_runs --left-run-dir /abs/path/to/G1_run --right-run-dir /abs/path/to/G3_run --left-label G1 --right-label G3
python -m scripts.evaluation.run_experiment_suite --task g_compare_runs --left-run-dir /abs/path/to/G2_run --right-run-dir /abs/path/to/G4_run --left-label G2 --right-label G4
```

建议额外补跑：

```bash
python -m scripts.evaluation.run_experiment_suite --task g_compare_runs --left-run-dir /abs/path/to/G3_run --right-run-dir /abs/path/to/G4_run --left-label G3 --right-label G4
python -m scripts.evaluation.run_experiment_suite --task g_compare_runs --left-run-dir /abs/path/to/G1_run --right-run-dir /abs/path/to/G4_run --left-label G1 --right-label G4
```

## 手册 E：如何做统计检验、Judge 和盲评材料导出

### 1. 先准备四组 run 映射文件

建议先手写一个 JSON，例如 `experiments/assets/g_run_dirs.json`：

```json
{
  "G1": "/abs/path/to/G1_run",
  "G2": "/abs/path/to/G2_run",
  "G3": "/abs/path/to/G3_run",
  "G4": "/abs/path/to/G4_run"
}
```

### 2. 统计检验

当前正式协议优先使用 runner 已接入的标准 task，而不是临时 Python 调用。

```bash
python - <<'PY'
from scripts.evaluation.g_workflow_closure import extract_g_group_score_map
run_dirs = {
    "G1": "/abs/path/to/G1_run",
    "G2": "/abs/path/to/G2_run",
    "G3": "/abs/path/to/G3_run",
    "G4": "/abs/path/to/G4_run",
}
extract_g_group_score_map(run_dirs, output_path="experiments/assets/g_score_map.json")
PY
```

然后再跑正式统计检验 task：

```bash
python -m scripts.evaluation.run_experiment_suite \
  --task g_compute_pairwise_tests \
  --input-path experiments/assets/g_score_map.json
```

### 3. LLM Judge

同样，当前推荐先通过 `g_workflow_closure.py` 做四组批量汇总：

```bash
python - <<'PY'
from scripts.evaluation.g_workflow_closure import run_g_batch_llm_judge
run_dirs = {
    "G1": "/abs/path/to/G1_run",
    "G2": "/abs/path/to/G2_run",
    "G3": "/abs/path/to/G3_run",
    "G4": "/abs/path/to/G4_run",
}
run_g_batch_llm_judge(run_dirs, output_dir="experiments/runs/g_judge_bundle", model="gpt-4o")
PY
```

### 4. 人工盲评导出

先导出 blind pack：

```bash
python -m scripts.evaluation.run_experiment_suite \
  --task g_export_blind_review_pack \
  --input-path experiments/assets/g_run_dirs.json \
  --sample-size 20 \
  --seed 42
```

当前如果还需要 worksheet，可继续调用：

```bash
python - <<'PY'
import pandas as pd
from scripts.evaluation.blind_review_export import export_blind_review_worksheet
blind_rows = pd.read_csv("experiments/runs/<latest_gblind_run>/blind_review_pack.csv").to_dict(orient="records")
export_blind_review_worksheet(blind_rows, "experiments/runs/<latest_gblind_run>/blind_review_worksheet.csv")
PY
```

## 手册 E：如何回收盲评结果并生成章节报告

### 1. 聚合 blind review 结果

```bash
python - <<'PY'
from scripts.evaluation.g_workflow_closure import aggregate_blind_review_results
aggregate_blind_review_results(
    "experiments/runs/<latest_gblind_run>/blind_review_worksheet_filled.csv",
    output_dir="experiments/runs/g_blind_review_summary"
)
PY
```

### 2. 生成 G1-G4 统一章节报告

```bash
python - <<'PY'
from scripts.evaluation.g_workflow_closure import build_g_chapter_report
run_dirs = {
    "G1": "/abs/path/to/G1_run",
    "G2": "/abs/path/to/G2_run",
    "G3": "/abs/path/to/G3_run",
    "G4": "/abs/path/to/G4_run",
}
build_g_chapter_report(
    run_dirs,
    pairwise_tests_path="experiments/runs/<latest_gstats_run>/pairwise_tests.csv",
    judge_summary_path="experiments/runs/g_judge_bundle/judge_summary.csv",
    blind_review_summary_dir="experiments/runs/g_blind_review_summary",
    output_dir="experiments/runs/g_chapter_report"
)
PY
```

## 手册 F：当前不要再做什么

当前不建议插队做这些事：

- 不再回头推进 `E10 v3 / v4` 作为当前主线
- 本轮按最终协议正式重跑 `E1-E10` 与 `G1-G4`
- 不把 `reranker` 或 `fallback` 接回默认主流程
- 不在 `G3 / G4` 前使用 `v3 / v4` adapter 替换 `exp02`
- 不在缺少正式 `G` 资产和 `exp02` metadata 的情况下直接启动云端四组运行

## 一句话版

当前最重要的人工动作只有一条主线：

先补齐 `exp02` metadata 和正式 G assets，再上云跑 `G1-G4`，然后回本地完成统计检验、Judge、盲评和章节总报告。
