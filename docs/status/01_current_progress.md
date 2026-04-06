更新时间：2026-04-05

## 总体阶段判断

当前项目已经完成两类工作：

- 历史实验层：`E1-E10` 的主要结果已经形成可冻结、可写论文的基础材料
- 新计划工程层：面向 `G1-G4` 的统一检索/生成/统计检验/Judge/盲评导出底座已经大体接通

当前真正所处阶段不是“继续改 E10”，而是：

- 保留 `E5-E10` 作为第五章和第六章的辅助证据
- 用 `G1-G4` 跑出当前正式 decisive scope（`68` 条查询）下的统一矩阵结果；`q021 / q024` 已从 decisive matrix 中剔除并转为 supporting boundary cases
- 在 8 小时内完成一轮完整实验闭环
- 在后续 32 小时内形成一版可写入论文的完整结果集

## 已完成的冻结实验结果

### 数据与实验底座

以下底座已经完成并继续视为冻结输入：

- `experiments/assets/frozen_config.yaml`
- `experiments/assets/frozen_split_manifest.json`
- `experiments/assets/judged_queries.jsonl`
- `experiments/assets/slot_gold.jsonl`
- `experiments/assets/clarify_gold.jsonl`
- `experiments/assets/annotation_rubrics.md`

### 检索与行为层历史实验

以下实验继续作为论文辅助证据保留：

- `E1` 方面分类
- `E2` 候选筛选
- `E3` 偏好解析
- `E4` 澄清决策
- `E5` 查询桥接
- `E6` Aspect vs Plain 检索
- `E7` reranker 消融
- `E8` fallback 消融

默认冻结主线继续保持：

- 检索模式：`aspect_main_no_rerank`
- 正式行为模型：`Qwen/Qwen3.5-4B`

### 生成层历史实验

以下 run 已同步回本地并作为冻结证据保留：

- `E9` 正式 compare：
  - `experiments/runs/e9_8449c12a50585e42_20260404T081010+0000/`
- `E10` base formal：
  - `experiments/runs/e10_0dc5c2e6f867c66f_20260402T015230+0000/`
- `E10` PEFT v2 / exp02：
  - `experiments/runs/e10_a2dd1a0bd73c57b5_20260402T073127+0000/`
- `E10` PEFT v3 / exp03：
  - `experiments/runs/e10_927c1d0a2fbc5870_20260403T083734+0000/`
- `E10` PEFT v4 / exp04：
  - `experiments/runs/e10_d749ec0796f7b365_20260404T024800+0000/`

当前固定结论继续保持：

- `E9`：RAG 的核心收益体现在更稳定的可审计推荐，而不仅仅是 citation 表面提升
- `E10`：`exp02 / v2` 仍是当前综合最优 PEFT 适配器候选
- `exp03` 与 `exp04` 保留为迭代链分析材料，不作为 G3/G4 的默认适配器

## 已完成的代码侧升级

### 检索与生成共享评测底座

当前工作区已经具备：

- 检索层新版 `6` 指标输出能力
- 生成层新版 `7` 指标输出能力
- 通用 generation summary / compare / post-hoc 重汇总能力
- `G1-G4` 的 generation 入口与双组 compare 入口

关键文件：

- `scripts/evaluation/evaluate_e6_e8_retrieval.py`
- `scripts/evaluation/evaluate_e9_e10_generation.py`
- `scripts/evaluation/run_experiment_suite.py`

### G 系列资产与运行入口

当前已经落地：

- `experiments/assets/g_eval_query_ids_68.json`
- `g_build_query_ids_70`
- `g_freeze_plain_retrieval_assets`
- `g_freeze_aspect_retrieval_assets`
- `g_run_generation`
- `g_compare_runs`

当前实际状态是：

- 代码入口已存在
- retrieval/generation 契约已经打通
- 但正式 `G` 资产与正式 `G1-G4` 结果还没有写入工作区

### 后处理与闭环分析模块

以下模块已经在代码层落地：

- `scripts/evaluation/statistical_tests.py`
- `scripts/evaluation/llm_judge.py`
- `scripts/evaluation/blind_review_export.py`
- `scripts/evaluation/g_workflow_closure.py`

当前已覆盖能力：

- Wilcoxon + Bootstrap CI + Cohen's d / rank-biserial
- LLM Judge 单组评审与汇总
- 盲评匿名抽样导出
- `exp02` metadata 校验/占位
- `G1-G4` score map 提取
- blind review 结果聚合
- G 章节统一报告生成 helper

## 当前真实阻塞项

以下几项是上云前仍需明确处理的真实缺口：

### 1. 正式 G 资产尚未落地

当前资产目录中：

- 已有：`experiments/assets/g_eval_query_ids_68.json`
- 尚未看到正式同步回仓库的：
  - `experiments/assets/g_plain_generation_eval_units.jsonl`
  - `experiments/assets/g_aspect_generation_eval_units.jsonl`

这意味着：

- `G` 系列 asset freeze 代码已就位
- 但正式运行前还需要先生成/同步这两份资产

### 2. `exp02` metadata 尚未补齐到本地资产目录

当前本地存在：

- `e10_adapter_metadata.qwen35_4b_peft_v1.json`
- `e10_adapter_metadata.qwen35_4b_peft_v3.json`
- `e10_adapter_metadata.qwen35_4b_peft_v4.json`

当前本地缺少：

- `experiments/assets/e10_adapter_metadata.qwen35_4b_peft_v2.json`

这会直接阻塞：

- `G3`
- `G4`

### 3. `g_workflow_closure.py` 尚未统一接入 runner

当前 `g_workflow_closure.py` 已存在，但还属于 helper/library 层：

- 可以调用
- 但尚未形成统一 CLI task

这意味着：

- 统计 payload 提取、批量 Judge、blind review 结果聚合、章节报告生成已经有代码
- 但正式上云/回本地闭环时，仍需通过脚本调用或继续补 runner 接线

## 当前验证状态

当前已经完成的验证包括：

- 与 `G` 系列整合直接相关的单测集通过
- `statistical_tests.py` 的强化版测试通过
- `llm_judge.py` / `blind_review_export.py` 的单测通过
- `G` 资产 freeze 的临时 smoke 已验证 plain/aspect 两路都可生成合规 `GenerationEvalUnit`
- 对真实本地 `E9/E10` run 的后处理 smoke 已验证可直接重汇总/重 compare

注意：

- 更大范围的全套测试在当前 Windows 环境下出现过 `D:\\Temp` 权限问题
- 该问题当前表现为环境层面的临时目录权限异常，不等同于 `G` 系列闭环逻辑本身失败

## 当前一句话结论

如果只看代码实现，项目已经从“缺能力”进入“缺正式运行产物”的阶段。

当前最重要的不是继续扩写历史实验，而是：

1. 补齐 `exp02` metadata
2. 正式冻结 `G` 系列资产
3. 云端跑完 `G1-G4`
4. 回本地完成统计检验、Judge、盲评和章节总报告
