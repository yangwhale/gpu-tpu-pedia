# Qwen3.5-397B-A17B-FP8 Inference on vLLM TorchTPU

**中文** ｜ 路线：[vllm-torchtpu](../) （PyTorch 前端）　对照：[tpu-inference 版](../../tpu-inference/Qwen3.5-397B-A17B-FP8/)（JAX 前端）

---

## 📌 文档状态

| | |
|---|---|
| 版本 | **v0 — 配置与基线整理完成，端到端复现待做** |
| 数据来源 | `vllm-project/vllm-torchtpu` 主干的 CI baseline 与 benchmark config（2026-08-12 读取） |
| 本目录自测 | **尚未进行**，见文末[待实测清单](#-待实测清单) |

> 本文中所有性能与精度数字均**引用上游 CI baseline**，不是本目录跑出来的。它们的价值在于：这是仓库持续守护的回归基线，说明这条路被跑通过、且退化会被拦住。但**「上游 CI 能跑」和「你在自己集群上能跑」之间隔着环境、权重、配额三件事**，所以下面的步骤仍需实测校准。凡本目录未验证的地方都已显式标注。

---

## 🎯 上游 CI 实测性能

模型 `Qwen/Qwen3.5-397B-A17B-FP8`，FP8 权重 + FP8 KV cache，EP 开启，`ATTENTION_BACKEND=CUSTOM`。
每晚扫三种并行布局 × 2 种 ISL/OSL × 3 档并发。

### TPOT（毫秒，越低越好）

| 负载 | TP1×DP8 | TP2×DP4 | TP8×DP1 |
|---|---|---|---|
| 1k in / 8k out · c64 | 25.3 | 23.3 | **22.1** |
| 1k in / 8k out · c256 | 42.1 | 41.5 | **40.6** |
| 1k in / 8k out · c512 | **47.1** | 48.0 | 58.7 |
| 8k in / 1k out · c64 | 78.8 | 61.8 | **39.0** |
| 8k in / 1k out · c256 | **92.4** | 96.7 | 96.2 |
| 8k in / 1k out · c512 | 111.2 | 118.1 | **111.1** |

### 总吞吐（tok/s，越高越好）

| 负载 | TP1×DP8 | TP2×DP4 | TP8×DP1 |
|---|---|---|---|
| 1k in / 8k out · c64 | 2,719 | 2,972 | **3,184** |
| 1k in / 8k out · c256 | 6,146 | 6,309 | **6,416** |
| 1k in / 8k out · c512 | 9,123 | **9,162** | 6,617 |
| 8k in / 1k out · c64 | 7,428 | 9,349 | **14,411** |
| 8k in / 1k out · c256 | **22,356** | 21,551 | 19,749 |
| 8k in / 1k out · c512 | **29,351** | 27,476 | 19,667 |

> **TTFT 为什么不在表里**：高并发 cell 的 `median_ttft_ms` 被 `check_regression.py` 有意从 baseline 中移除——并发饱和时它反映的是排队位置的算术，不是延迟信号。参见[目录级说明](../#ttft-在高并发-cell-上是被有意丢弃的)。

---

## 🧭 并行布局怎么选

这是本模型最值得先看懂的一件事。三种布局的 **`DP × MAX_NUM_SEQS` 恒等于 512**，所以并发扫描在三者上探的是**相同的相对负载点**——数字之间可以直接比。

| 布局 | TP | DP | `MAX_NUM_SEQS` | `MAX_NUM_BATCHED_TOKENS` |
|---|---|---|---|---|
| golden | 1 | 8 | 64 | 1,024 |
| 中间态 | 2 | 4 | 128 | 2,048 |
| 全 TP | 8 | 1 | 512 | 8,192 |

从上面两张表能读出一条很干净的规律：

- **低并发选 TP 大。** `8k in / c64` 这一格，TP8×DP1 的 39.0 ms 比 TP1×DP8 的 78.8 ms **快一倍**——长 prompt + 低并发时，prefill 是瓶颈，把一个请求摊到 8 路上算最划算。
- **高并发选 DP 大。** `8k in / c512` 反过来，TP1×DP8 的 29,351 tok/s 比 TP8×DP1 的 19,667 tok/s **高 49%**——请求足够多时，8 个独立副本各自吃满远比 8 路同步一个请求高效。
- **交叉点在 c256 附近**，且随 ISL/OSL 移动。1k in / 8k out（decode 重）的交叉点更靠后，8k in / 1k out（prefill 重）更靠前。

**选型建议**：
- 交互式低延迟场景（对话、代码补全）→ **TP8×DP1**
- 批量高吞吐场景（离线评测、数据生成）→ **TP1×DP8**
- 不确定 / 混合负载 → TP2×DP4，但先读下面的已知问题

---

## 🐛 已知问题

### TP2×DP4 是双峰的（上游已知，未解）

配置文件里原话：

> This layout is bimodal run-to-run (~40.6 vs ~45 ms TPOT at 1k/8k c256, and c512 throughput swings ~6%) — the DP=4 router appears to settle into one of two load patterns.

上游的处理是把回归门禁放宽到 `PERF_TOLERANCE="0.08"`（其余布局用默认）。

**这意味着**：拿 TP2×DP4 做 A/B 对比时，**单次测量不可信**，6% 以内的差异分不清是改动带来的还是路由抽签抽到了另一个模式。要么多跑几次取分布，要么换 TP8×DP1 / TP1×DP8 做对比基准。根因（DP=4 时 router 为何会锁定在两种负载模式之一）尚未定位。

---

## ⚠️ 必读约束

### A. 这是多模态模型，纯文本推理必须显式关掉视觉

```bash
--language-model-only
--limit-mm-per-prompt {"image":0,"video":0}
```

两个都要给。不给的话会为视觉塔预留资源，白占 HBM。

### B. Thinking 默认要关

```bash
--default-chat-template-kwargs {"enable_thinking":false}
```

上游 benchmark 全部在 thinking OFF 下测得。开着 thinking 的话上面所有数字都不可比——输出 token 数完全是另一个量级。

### C. MoE 走 SparseCore（两个环境变量）

```bash
export USE_MOE_SPARSE_CORE="1"
export ONEHOT_MOE_PERMUTE_THRESHOLD="32768"
```

这是 Qwen3.5 架构专属的 kernel 优化路径——512 专家 + 1 共享专家的 dispatch 走 TPU 的 SparseCore 单元（对应 `kernels/sparse_core/` 那 4,734 行 ragged gather/scatter）。**没设这两个变量时的性能不在上表的口径内。**

### D. benchmark 必须 `temperature=0`

`vllm bench serve` 默认服务端采样（temp 0.7），在 decode-heavy 的 cell 上明显更慢。上游 config 显式设 `BENCHMARK_TEMPERATURE="0"`。

### E. `--block-size 256`

不是默认值。上表全部在这个 block size 下测得。

---

## ⚡ Quick Start

> ⚠️ 以下命令由 `scripts/vllm/benchmarking/configs/qwen3.5-397b-fp8-tp8-dp1-ep.sh` 还原而来，**本目录尚未实测**。首次执行请逐条核对输出。

### 0. 环境

`vllm-torchtpu` 依赖私有 registry（`torch-tpu`、`torch` 等包），需要有读权限的公司账号：

```bash
gcloud auth login
gcloud auth application-default login
gcloud auth list        # 期望：active 账号是有 registry 读权限的那个

# 推荐 uv（比 pip 快很多）
uv venv --python 3.12 ~/uv_venv
source ~/uv_venv/bin/activate
```

完整安装步骤见 `vllm-torchtpu` 仓库 README 的 Installation 一节（registry index URL 与 keyring 认证）。

### 1. 直接用仓库的 benchmark runner（推荐先走这条）

```bash
cd <vllm-torchtpu>
./scripts/vllm/benchmarking/run_benchmarks.sh \
    --config qwen3.5-397b-fp8-tp8-dp1-ep
# 期望：拉起 server → 逐个 cell 跑 benchmark_serving → 结果落本地
# 三个可选 config：
#   qwen3.5-397b-fp8-tp1-dp8-ep   (golden，高吞吐)
#   qwen3.5-397b-fp8-tp2-dp4-ep   (双峰，慎用作基准)
#   qwen3.5-397b-fp8-tp8-dp1-ep   (低延迟)

# 先看要跑什么而不真跑：
./scripts/vllm/benchmarking/run_benchmarks.sh --config qwen3.5-397b-fp8-tp8-dp1-ep --dry-run
# 服务器跑完不关，方便手动验证：
./scripts/vllm/benchmarking/run_benchmarks.sh --config qwen3.5-397b-fp8-tp8-dp1-ep --keep-alive
```

用现成 runner 的好处：`RANGE_RATIO_STYLE` 翻译、temperature、并发扫描都已经对齐 baseline 口径，出来的数能直接跟上表比。

### 2. 手工拉起 server（需要自定义时）

```bash
export USE_MOE_SPARSE_CORE="1"
export ONEHOT_MOE_PERMUTE_THRESHOLD="32768"

vllm serve Qwen/Qwen3.5-397B-A17B-FP8 \
  --tensor-parallel-size 8 \
  --data-parallel-size 1 \
  --enable-expert-parallel \
  --quantization fp8 \
  --kv-cache-dtype fp8 \
  --max-model-len 10240 \
  --max-num-batched-tokens 8192 \
  --max-num-seqs 512 \
  --gpu-memory-utilization 0.92 \
  --block-size 256 \
  --language-model-only \
  --limit-mm-per-prompt '{"image":0,"video":0}' \
  --default-chat-template-kwargs '{"enable_thinking":false}'
# 上游 config 给 server 就绪留了 120 分钟上限（SERVER_READY_WAIT_MIN=120）——
# 这是上限不是预期值，但说明冷启动可能很长，别过早判定挂了。
```

切换布局时**三个参数要一起改**，保持 `DP × MAX_NUM_SEQS = 512`：

| 布局 | `--tensor-parallel-size` | `--data-parallel-size` | `--max-num-seqs` | `--max-num-batched-tokens` |
|---|---|---|---|---|
| TP1×DP8 | 1 | 8 | 64 | 1024 |
| TP2×DP4 | 2 | 4 | 128 | 2048 |
| TP8×DP1 | 8 | 1 | 512 | 8192 |

### 3. 冒烟验证

```bash
curl -s localhost:8000/health -o /dev/null -w "%{http_code}\n"
# 期望：200

curl -s localhost:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model":"Qwen/Qwen3.5-397B-A17B-FP8",
  "messages":[{"role":"user","content":"The capital of France is"}],
  "temperature":0,"max_tokens":16
}' | python3 -c "import json,sys; d=json.load(sys.stdin)['choices'][0]; print(repr(d['message']['content']),'|',d['finish_reason'])"
# 期望：包含 Paris 的短句 | stop
# ⚠️ 若 finish_reason 是 length 且内容重复打转 → 先确认 enable_thinking 是否真的关掉了
```

---

## 🔬 精度基线（上游 CI，TP1×DP8）

| 评测 | 指标 | 分数 |
|---|---|---|
| MMLU（llama 格式） | exact_match, strict | **89.84%** ±0.39 |
| MMLU-Pro | exact_match, custom-extract | **82.29%** ±1.00 |
| HumanEval+ | pass@1 | **92.07%** ±2.12 |
| MBPP+ | pass@1 | **79.10%** ±2.09 |

回归容差 `EVAL_TOLERANCE="0.02"`。MMLU-Pro 跑的时候上游设了 `MMLU_PRO_DISABLE_MULTITURN_ARGS=true`。

```bash
./scripts/vllm/benchmarking/run_eval_flow.sh --config qwen3.5-397b-fp8-tp1-dp8-ep
```

---

## 📋 待实测清单

本目录接下来要做的事，按顺序：

- [ ] **跑通 TP8×DP1 单机**，记录真实 cold start 时长（上游只给了 120 分钟的等待上限，不是预期值）
- [ ] **复现 6 个 cell 的 TPOT / 吞吐**，与上表逐格对比，偏差 >5% 的格子单独查
- [ ] **验证不设 `USE_MOE_SPARSE_CORE` 的退化幅度** —— 量化 SparseCore MoE 到底值多少
- [ ] **确认 TP2×DP4 双峰**：同配置连跑 5 次看 TPOT 分布是不是真的双峰，能否找到触发条件
- [ ] **补 TTFT 的正确测法**：低并发 cell 下 TTFT 是有意义的，需要单独设计一组低并发测量
- [ ] **跑一遍 MMLU + HumanEval+**，确认精度基线可复现
- [ ] **profile 一次**（`CAPTURE_PROFILE=1`），定位 GDN 层与 MoE dispatch 各占多少
- [ ] **与 [tpu-inference 版](../../tpu-inference/Qwen3.5-397B-A17B-FP8/) 同硬件对比**，量化两条路线的差距

---

## 📎 与 tpu-inference 版的关系

同一个模型在本仓库有两份文档，因为它们是两条独立的技术路线：

| | tpu-inference 版 | 本文档（vllm-torchtpu 版） |
|---|---|---|
| 前端 | JAX | PyTorch（vLLM 原样） |
| 模型定义 | 需要 JAX 侧实现 | 直接用 vLLM 上游的 |
| 已验证内容 | 单机 / PD 分离 / Multi-host TP=16，含多个必需 patch | 上游 CI 三种并行布局；本目录待复现 |
| 适合 | 已在 JAX 生态 | 想让现有 PyTorch 栈零改动上 TPU |

两份都值得读——tpu-inference 那份记录了这个模型在 TPU 上的**模型侧坑**（thinking 行为、chat 路径稳定性等），那些坑与前端语言无关，在本路线上大概率同样存在。
