# AutoDL 云端 Qwen3.5 部署与行为实验操作手册

更新时间：2026-03-31

本手册是当前仓库里关于 AutoDL 云端部署、`Qwen3.5-2B / 4B / 9B` 推理对比，以及 `E3/E4` 行为实验执行流程的唯一权威说明。后续如果云端执行方式发生变化，请优先更新本文件，再同步其他入口文档。

## 1. 你现在为什么要上云

当前本地仓库已经完成：

- `E1`
- `E2`
- `E5`
- `E6`
- `E7`
- `E8`

同时，`E3/E4` 的共享评测代码已经准备好，但 Base 组不再继续在本地机器下载和缓存模型权重。

你现在要上云，不是因为“实验还没想清楚”，而是因为当前下一阶段的真正目标已经很明确：

- 用云端 GPU 环境完成 `E3` 和 `E4`
- 在统一后端条件下比较 `Qwen3.5-2B / 4B / 9B`
- 为后续论文中的行为实验章节沉淀正式结果

当前主问题不是“要不要做实验”，而是“如何把云端推理链路稳定跑通，并把结果规范地带回本仓库”。

成功标准：

- 云端模型服务能稳定启动
- 模型输出能稳定走到行为实验脚本
- `E3/E4` 能按正式口径产出 `runs/` 结果目录

常见错误：

- 一上云就急着跑完整实验，没有先做 API 冒烟验证
- 还把“本地下载模型”当成默认路径
- 在模型、prompt、检索配置同时变化的情况下做比较

## 2. 实验设计冻结

本轮云端行为实验的正式主对照组固定为：

- `A_rule_baseline`
- `B_qwen35_2b_nonthinking`
- `C_qwen35_4b_nonthinking`
- `D_qwen35_9b_nonthinking`

冻结规则如下：

- 三个 Qwen 模型统一使用 non-thinking 模式
- 三个模型统一使用同一批 query
- 三个模型统一使用同一 prompt 版本
- 三个模型统一使用同一输出 schema
- 三个模型统一使用同一默认检索后端：
  - `retrieval_mode = aspect_main_no_rerank`
  - `fallback_enabled = false`
- 一次只启动一个模型服务，不同时挂三套服务
- 先完成 inference 对比，再决定是否进入 PEFT

为什么必须统一关闭 thinking：

- 官方模型卡显示，[Qwen3.5-2B](https://huggingface.co/Qwen/Qwen3.5-2B) 默认是 non-thinking
- [Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B) 和 [Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B) 默认带有 thinking 行为
- 如果不统一 `enable_thinking=False`，那你比较的就不只是“模型规模”，还混入了“推理模式”变量

所以本轮正式结论只回答一个问题：

- 在相同非思维模式、相同 prompt、相同 query、相同检索后端下，不同模型规模对 `E3/E4` 的行为效果是否有差异

成功标准：

- 三个模型的实验配置字段完全一致
- 唯一主要变量是模型规格

常见错误：

- 2B 用 non-thinking，4B/9B 却保留默认 thinking
- 每个模型都顺手微调 prompt
- 边跑边改 query 集或 schema

## 3. 云服务器规格选择

本轮固定推荐：

- 设备：`vGPU-48GB`
- 数据盘：至少 `100GB`
- 推荐数据盘：`150GB`

不推荐继续使用默认 `50GB` 数据盘，原因很简单：

- 三个模型会轮流下载和缓存
- `vLLM` 自身也有缓存
- 你还需要保留环境、日志和实验产物

为什么这次优先选 `vGPU-48GB`：

- `48GB` 显存足够覆盖 `2B / 4B / 9B` 单模型逐个部署
- 这轮任务是结构化行为实验，不需要多卡并行
- `9B` 在 `48GB` 上更宽松，调试空间更大
- 同价位下，`48GB` 比 `32GB` 更适合后续扩展到 LoRA / QLoRA 或更长上下文试验

建议租机时顺手确认：

- GPU 驱动和 CUDA 已经预装可用
- 数据盘容量不是默认最小值
- 机器支持长期运行和断线后恢复

成功标准：

- 机器能稳定启动单个 `Qwen3.5` 模型服务
- 磁盘不会因为缓存而很快爆满

常见错误：

- 只看显存，不看数据盘容量
- 一开始用最小磁盘，下载到一半失败
- 把三种模型长期同时留在服务中占资源

## 4. 哪些文件要同步到云端

推荐主路径：

- 直接在云端 `git clone` 整个仓库

这样最好，因为：

- 路径结构最稳定
- 文档、脚本、实验资产不会漏
- 后续同步结果更清楚

如果你必须手动上传，最小集合固定为：

- `configs/`
- `scripts/`
- `experiments/assets/`
- `experiments/labels/e6_qrels/`
- `experiments/reports/`
- `requirements.txt`
- `README.md`
- `docs/` 建议一并保留

如果你后续在云端还要复跑 `E5` 或更多检索实验，再额外同步：

- `data/intermediate/evidence_index.pkl`
- `data/chroma_db/`

这轮行为实验默认不需要上传：

- `venv/`
- `raw_data/`
- 本地 Hugging Face 缓存
- 本地 `.DS_Store`
- `configs/db.yaml`
- 与当前行为实验无关的大体量中间数据

成功标准：

- 云端仓库能独立看到脚本、配置、冻结资产和说明文档
- 不把本地环境垃圾一起带上云

常见错误：

- 手动拷文件时漏掉 `experiments/assets/`
- 把整个本地 `venv/` 也打包上传
- 忘记 `E6` 冻结 qrels 资产，导致 `E5` 或相关评测无法复用

## 5. AutoDL 上的目录规划

建议在云端固定使用下面这套目录：

```text
/root/autodl-tmp/
├── workspace/
│   └── DatafinitiHotelReviews/
├── hf-cache/
├── vllm-cache/
├── logs/
└── outputs/
```

推荐先执行：

```bash
mkdir -p /root/autodl-tmp/workspace
mkdir -p /root/autodl-tmp/hf-cache
mkdir -p /root/autodl-tmp/vllm-cache
mkdir -p /root/autodl-tmp/logs
mkdir -p /root/autodl-tmp/outputs
```

然后固定环境变量：

```bash
export HF_HOME=/root/autodl-tmp/hf-cache
export HUGGINGFACE_HUB_CACHE=/root/autodl-tmp/hf-cache
export VLLM_CACHE_ROOT=/root/autodl-tmp/vllm-cache
export OPENAI_API_KEY=EMPTY
export OPENAI_BASE_URL=http://127.0.0.1:8000/v1
```

建议把这些 `export` 写进云端用户自己的 shell 初始化文件里，避免每次重登都重新设置。

成功标准：

- 模型缓存、服务缓存、日志、仓库目录彼此分开
- 重启 shell 后环境变量可恢复

常见错误：

- 直接把缓存写到系统盘
- 缓存目录和仓库目录混在一起
- 忘了设置 `OPENAI_BASE_URL`

## 6. 基础环境初始化命令

推荐从 AutoDL 终端按顺序执行：

```bash
cd /root/autodl-tmp/workspace
git clone <your-repo-url> DatafinitiHotelReviews
cd DatafinitiHotelReviews

python -m venv .venv
source .venv/bin/activate

pip install -U pip setuptools wheel
pip install -r requirements.txt
pip install -U openai
pip install vllm --torch-backend=auto --extra-index-url https://wheels.vllm.ai/nightly
```

额外提醒：

- 当前 `Qwen3.5` 更建议优先走 API 服务方式
- `vLLM` 尽量保持较新的版本
- 行为实验是纯文本任务，后续服务命令统一使用 `--language-model-only`

如果你只打算先验证部署是否成功，也建议至少完成：

- Python 虚拟环境
- `requirements.txt`
- `openai`
- `vllm`

成功标准：

- `python`、`pip`、`vllm` 都能正常调用
- 环境中没有明显版本冲突

常见错误：

- 还没激活虚拟环境就安装
- 只装了仓库依赖，没有装 `openai` 和 `vllm`
- 把 `vllm` 装到了系统 Python

## 7. 三种模型的标准服务启动命令

本轮统一：

- 端口：`8000`
- `tensor-parallel-size = 1`
- `max-model-len = 8192`
- 文本模式

### Qwen3.5-2B

```bash
vllm serve Qwen/Qwen3.5-2B \
  --port 8000 \
  --tensor-parallel-size 1 \
  --max-model-len 8192 \
  --language-model-only
```

### Qwen3.5-4B

```bash
vllm serve Qwen/Qwen3.5-4B \
  --port 8000 \
  --tensor-parallel-size 1 \
  --max-model-len 8192 \
  --reasoning-parser qwen3 \
  --language-model-only
```

### Qwen3.5-9B

```bash
vllm serve Qwen/Qwen3.5-9B \
  --port 8000 \
  --tensor-parallel-size 1 \
  --max-model-len 8192 \
  --reasoning-parser qwen3 \
  --language-model-only
```

为什么不是官方示例里常见的 `262144`：

- 当前任务是结构化行为实验，不是长上下文能力测试
- `8192` 足够覆盖本轮 prompt 和 query
- 更短的上下文上限能显著降低显存和 KV cache 压力
- 你现在追求的是稳定对比，不是极限吞吐和超长上下文

建议把服务日志保存在：

```bash
vllm serve ... 2>&1 | tee /root/autodl-tmp/logs/qwen35_2b_vllm.log
```

成功标准：

- 服务端口正常监听
- 模型能完成一次请求响应
- 日志里没有持续滚动的报错

常见错误：

- 服务起了但端口被别的进程占用
- 4B/9B 没带 `--reasoning-parser qwen3`
- 忘了保存启动日志

## 8. 服务冒烟验证

在开始正式实验前，必须先做一次最小 API 验证。

仓库里已经提供了统一的冒烟脚本：

`scripts/evaluation/smoke_test_qwen_api.py`

推荐直接这样跑：

```bash
export OPENAI_API_KEY=EMPTY
export OPENAI_BASE_URL=http://127.0.0.1:8000/v1
export BEHAVIOR_MODEL_ID=Qwen/Qwen3.5-4B
export BEHAVIOR_ENABLE_THINKING=false

python scripts/evaluation/smoke_test_qwen_api.py
```

这个脚本会读取当前环境变量并发起最小请求，因此切换 `2B / 4B / 9B` 时只需要改：

- `BEHAVIOR_MODEL_ID`
- `BEHAVIOR_ENABLE_THINKING`

你需要检查两件事：

- 服务是否正常返回
- 返回内容中是否还出现 `<think>`

为什么这里必须显式传 `enable_thinking=False`：

- 这是 `4B / 9B` 公平对比所必需
- 如果输出仍然带 `<think>`，就说明你实际上没有把 thinking 模式关掉
- 一旦 thinking 没关掉，本轮正式实验就不能直接开跑

成功标准：

- 响应成功返回
- 输出内容简洁
- 不出现 `<think>`

常见错误：

- 只看“能返回”，不检查有没有 `<think>`
- 2B 验证过了，就默认 4B/9B 也一定没问题
- 请求参数没把 `extra_body` 传进去

## 9. 当前代码接线方式必须单独看清

这一步非常重要。

当前仓库中的行为实验脚本已经支持通过 OpenAI-compatible API 调用云端 `vLLM`，但你仍然需要明确它是如何接线的：

- [`scripts/evaluation/evaluate_e3_e5_behavior.py`](../../scripts/evaluation/evaluate_e3_e5_behavior.py) 目前仍然使用本地 `transformers` 方式直载模型
- 同时它现在也支持通过配置或环境变量切换到 `api` 后端
- 也就是说，这份仓库已经具备“本地直载 / 云端 API”双模式

当前正式云端运行时，推荐固定使用 `api` 后端，并确保以下字段已经对齐：

- `BEHAVIOR_LLM_BACKEND=api`
- `BEHAVIOR_MODEL_ID=Qwen/Qwen3.5-2B`
- `OPENAI_BASE_URL=http://127.0.0.1:8000/v1`
- `OPENAI_API_KEY=EMPTY`
- `BEHAVIOR_ENABLE_THINKING=false`

脚本当前已经会把以下关键信息写入 `run_meta.json`：

- `model_id`
- `behavior_backend`
- `behavior_api_base_url`
- `enable_thinking`
- `max_new_tokens`

建议提前冻结这组环境变量接口：

```bash
export BEHAVIOR_LLM_BACKEND=api
export BEHAVIOR_MODEL_ID=Qwen/Qwen3.5-2B
export OPENAI_BASE_URL=http://127.0.0.1:8000/v1
export OPENAI_API_KEY=EMPTY
export BEHAVIOR_ENABLE_THINKING=false
```

请把这一段理解成“云端部署与脚本接线现在都已经打通，但正式实验仍然必须检查运行配置是否对齐”。

成功标准：

- 你能明确区分“服务部署成功”和“正式实验配置已经对齐”是两回事
- `run_meta.json` 中能看到实际模型、后端和 thinking 状态

常见错误：

- 看到 `vLLM` 能起服务，就误以为模型名、端口和 thinking 状态一定都正确
- 没检查 `run_meta.json` 是否真的记录了实际运行配置

## 10. 正式运行顺序

本轮正式执行顺序固定为：

1. 部署 `Qwen3.5-2B`
2. 做 API 冒烟验证
3. 跑 `E3`
4. 跑 `E4`
5. 保存 run 目录和日志
6. 停掉 `2B` 服务
7. 启动 `Qwen3.5-4B`
8. 重复相同步骤
9. 再换 `Qwen3.5-9B`
10. 三组都完成后再整理横向对比

如果未来 API backend 已经接好，正式运行命令再统一类似为：

```bash
source .venv/bin/activate
export BEHAVIOR_LLM_BACKEND=api
export BEHAVIOR_MODEL_ID=Qwen/Qwen3.5-2B
export BEHAVIOR_ENABLE_THINKING=false
python -m scripts.evaluation.run_experiment_suite --task e3_preference
python -m scripts.evaluation.run_experiment_suite --task e4_clarification
```

执行过程中不要做这些事：

- 不要在三种模型之间更改 prompt
- 不要在三种模型之间更改 query 集
- 不要把 reranker 或 fallback 临时接回主流程
- 不要边跑边改评测口径

成功标准：

- 每个模型都完整跑过同一条流程
- 每个模型都留下对应日志与输出目录

常见错误：

- `E3` 用一种配置，`E4` 又换了模型或 prompt
- 先跑完 2B 后，4B 顺手改了 prompt
- 一个模型只跑了 `E3` 没跑 `E4`

## 11. E9 有无 RAG 对比如何在 AutoDL 云端执行

这一节只覆盖当前新增的 `E9` 正式对比：

- with RAG：
  - `B_grounded_generation`
- without RAG：
  - `D_no_evidence_generation`

这里必须特别记住：

- `E9` 的 `Qwen3.5-4B` 运行也属于 **云端行为实验**
- 继续使用 **AutoDL 上本地启动的 OpenAI-compatible / vLLM 服务**
- 不要在本地桌面直接跑正式 `e9_generation_constraints`
- `frozen_config.yaml` 里的 `http://127.0.0.1:8000/v1` 指的是 **云端容器内的本地服务地址**

### 11.1 先在云端启动 4B 服务

推荐命令：

```bash
cd /root/autodl-tmp/workspace/DatafinitiHotelReviews
source .venv/bin/activate

export HF_HOME=/root/autodl-tmp/hf-cache
export HUGGINGFACE_HUB_CACHE=/root/autodl-tmp/hf-cache
export VLLM_CACHE_ROOT=/root/autodl-tmp/vllm-cache
export OPENAI_API_KEY=EMPTY
export OPENAI_BASE_URL=http://127.0.0.1:8000/v1
export BEHAVIOR_LLM_BACKEND=api
export BEHAVIOR_MODEL_ID=Qwen/Qwen3.5-4B
export BEHAVIOR_ENABLE_THINKING=false

vllm serve Qwen/Qwen3.5-4B \
  --served-model-name Qwen/Qwen3.5-4B \
  --host 127.0.0.1 \
  --port 8000 \
  --dtype auto \
  --gpu-memory-utilization 0.90 \
  --max-model-len 4096 \
  --trust-remote-code \
  2>&1 | tee /root/autodl-tmp/logs/qwen35_4b_e9_vllm.log
```

如果你用的是另一套已经验证过的 `vllm serve` 参数，可以沿用，但必须保证：

- 服务地址仍是 `http://127.0.0.1:8000/v1`
- `served-model-name` 与 `BEHAVIOR_MODEL_ID` 一致
- `enable_thinking=false`

### 11.2 新开一个终端做 API 冒烟

不要在服务启动的同一个终端里直接跑实验。

新开终端后执行：

```bash
cd /root/autodl-tmp/workspace/DatafinitiHotelReviews
source .venv/bin/activate

export OPENAI_API_KEY=EMPTY
export OPENAI_BASE_URL=http://127.0.0.1:8000/v1
export BEHAVIOR_LLM_BACKEND=api
export BEHAVIOR_MODEL_ID=Qwen/Qwen3.5-4B
export BEHAVIOR_ENABLE_THINKING=false

python scripts/evaluation/smoke_test_qwen_api.py
```

成功标准：

- 请求能返回
- 输出里不出现 `<think>`

### 11.3 正式运行 E9 generation constraints

确认冒烟成功后，在同一个新终端里执行：

```bash
cd /root/autodl-tmp/workspace/DatafinitiHotelReviews
source .venv/bin/activate

export OPENAI_API_KEY=EMPTY
export OPENAI_BASE_URL=http://127.0.0.1:8000/v1
export BEHAVIOR_LLM_BACKEND=api
export BEHAVIOR_MODEL_ID=Qwen/Qwen3.5-4B
export BEHAVIOR_ENABLE_THINKING=false

python -m scripts.evaluation.run_experiment_suite --task e9_generation_constraints
```

如果你只想先做一次短冒烟，可以先限制 query 数：

```bash
python -m scripts.evaluation.run_experiment_suite \
  --task e9_generation_constraints \
  --limit-queries 3
```

### 11.4 跑完后应该检查什么

进入最新的 `e9_*` 目录，至少确认这些文件存在：

```bash
cd /root/autodl-tmp/workspace/DatafinitiHotelReviews
ls -lt experiments/runs | head -n 10
```

然后查看新目录中的：

- `summary.csv`
- `analysis.md`
- `rag_ablation_summary.csv`
- `rag_ablation_comparison.jsonl`
- `rag_ablation_analysis.md`

建议命令：

```bash
cat /root/autodl-tmp/workspace/DatafinitiHotelReviews/experiments/runs/最新e9目录/summary.csv
sed -n '1,260p' /root/autodl-tmp/workspace/DatafinitiHotelReviews/experiments/runs/最新e9目录/analysis.md
cat /root/autodl-tmp/workspace/DatafinitiHotelReviews/experiments/runs/最新e9目录/rag_ablation_summary.csv
sed -n '1,260p' /root/autodl-tmp/workspace/DatafinitiHotelReviews/experiments/runs/最新e9目录/rag_ablation_analysis.md
```

成功标准：

- `summary.csv` 里能看到四组：
  - `A_free_generation`
  - `B_grounded_generation`
  - `C_grounded_generation_with_verifier`
  - `D_no_evidence_generation`
- `rag_ablation_summary.csv` 能直接比较：
  - `B_grounded_generation`
  - `D_no_evidence_generation`
- `rag_ablation_analysis.md` 包含：
  - `Primary Conclusion`
  - `Recommendation Recovery Cases`
  - `Matched Abstentions`
  - `Suspicious No-RAG Wins`

### 11.5 同步回本地

跑完后，把新的 `e9_*` 目录同步回本地：

```bash
rsync -av \
  /root/autodl-tmp/workspace/DatafinitiHotelReviews/experiments/runs/最新e9目录/ \
  /path/to/local/DatafinitiHotelReviews/experiments/runs/最新e9目录/
```

如果你是从本地拉取云端：

```bash
rsync -av \
  <SSH_USER>@<SSH_HOST>:/root/autodl-tmp/workspace/DatafinitiHotelReviews/experiments/runs/最新e9目录/ \
  /Users/marcusao/vscode/GraduationDesign/DatafinitiHotelReviews/experiments/runs/最新e9目录/
```

### 11.6 这一步最容易犯的错

- 在本地桌面直接跑 `e9_generation_constraints`
- 云端没先起 `Qwen3.5-4B` 服务就直接跑脚本
- `OPENAI_BASE_URL` 不是 `http://127.0.0.1:8000/v1`
- `BEHAVIOR_MODEL_ID` 和 `served-model-name` 不一致
- 看到旧的 `E9` 目录还在，就误以为本轮 `rag_ablation_*` 已经生成

## 11. 结果回传与归档

每轮实验结束后，请把新的结果目录同步回本地仓库。

至少要回传：

- `experiments/runs/e3_*/`
- `experiments/runs/e4_*/`

同时建议一并保存：

- `vLLM` 启动日志
- 模型名
- 显卡类型
- 显存占用截图或文字记录
- 实验日期

建议主路径统一用 `rsync`，例如：

```bash
rsync -avz \
  /root/autodl-tmp/workspace/DatafinitiHotelReviews/experiments/runs/e3_*/ \
  <local-user>@<local-host>:/path/to/local/DatafinitiHotelReviews/experiments/runs/
```

如果你更习惯先打包，也可以先压缩再传，但最终本地仓库中应保持原始 run 目录结构。

建议在本地额外整理一份“云端运行记录”附录，记录：

- 模型
- 设备
- 显卡
- 日期
- 运行说明
- 日志路径

这样后续论文和答辩时更容易追溯。

成功标准：

- 结果目录成功回传
- 本地与云端能对应上同一轮实验
- 日志和模型信息没有丢

常见错误：

- 只传 `summary.csv`，没传完整 run 目录
- 结果传回来了，但没留服务日志
- 本地不知道某个 run 是哪个模型跑出来的

## 12. 常见问题排查

### 情况 1：启动时报 OOM

优先处理：

- 降低 `--max-model-len`
- 确认没有并行残留服务
- 确认显卡上没有其他占用进程

### 情况 2：数据盘爆满

优先处理：

- 检查 `/root/autodl-tmp/hf-cache`
- 检查 `/root/autodl-tmp/vllm-cache`
- 清理不再需要的旧模型缓存

### 情况 3：API 能通，但结果里仍出现 `<think>`

优先处理：

- 检查是否真的传了 `enable_thinking=False`
- 分别对 4B 和 9B 单独再做一次最小请求验证

### 情况 4：仓库脚本仍尝试本地下载模型

这通常意味着：

- 当前行为脚本还没有切到 API backend
- 或环境变量没有生效

这时不要继续怀疑云端服务本身，优先回到代码适配层处理。

### 情况 5：结果目录没生成

优先检查：

- 实验脚本是否真的跑完
- `experiments/runs/` 写盘路径是否正确
- 运行日志中是否出现 schema 或落盘异常

### 情况 6：4B / 9B 输出更长但不更稳

这本身可能就是实验发现的一部分。

请不要为了“看起来更聪明”就临时修改 prompt 抹平差异。正式对比的价值恰恰在于：

- 统一设置下，不同模型规模可能呈现不同稳定性和结构化输出质量

成功标准：

- 知道每类故障首先该查哪里
- 先排部署链路，再排脚本接线，再排模型行为

常见错误：

- 一遇到错误就同时改动服务命令、prompt、脚本和环境变量
- 把模型行为差异误判成部署故障

## 最短执行版

如果你只想快速回忆主线，请记这 8 步：

1. 租 `vGPU-48GB`
2. 扩数据盘到至少 `100GB`，建议 `150GB`
3. `git clone` 仓库到 `/root/autodl-tmp/workspace/`
4. 建虚拟环境并安装 `requirements.txt + openai + vllm`
5. 用 `vllm serve` 依次启动 `Qwen3.5-2B / 4B / 9B`
6. 冒烟验证 `enable_thinking=False`
7. 等 API backend 接好后再正式跑 `E3/E4`
8. 回传 `experiments/runs/e3_*/e4_*` 和日志，整理横向对比
