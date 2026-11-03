# GLM-5.3-Flash on Google Cloud G4（8× RTX PRO 6000）— 推理测试报告

**照着做一次跑通 → [RUNBOOK.md](RUNBOOK.md)**　｜　测试过程与决策记录 → [JOURNAL.md](JOURNAL.md)　｜　原始数据 → [results/](results/)

> **状态（2026-11-03）：测试完成。** 基础 7 个场景 + 8K 输入 / 1K 输出并发扫描（1–128）全部跑完，请求成功率 100%。

## 结论先行

| 项 | 结论 |
|---|---|
| 能不能跑 | **能**。vLLM nightly（`0.33.0rc1.dev484+g76c6f3f2a`）原生支持，零补丁，自动选用 `FLASHINFER_MLA_SPARSE_SM120` 后端 |
| 单路 decode | **约 100 tok/s**（TPOT 9.3 ms），输入从 1K 到 32K 基本不变 |
| 吞吐 | 1K 输入 / 512 输出、64 并发：**992 输出 tok/s**，3,001 总 tok/s |
| 8K 输入 / 1K 输出 | 单路 98.8 tok/s、TTFT 0.83 s；**并发 128 时输出 837 tok/s、总 7,541 tok/s**，每路约 7 tok/s（见 5.2） |
| 长输入 prefill | 32K 输入单路 TTFT 2.46 s（约 13K tok/s） |
| KV 容量 | 4,441,368 token；128K 上下文可同时容纳约 34 个请求 |
| 冷启动 | 权重加载 31 s，整个引擎初始化（含编译和 CUDA graph）318 s |

## 1. 测试对象

| 项 | 内容 |
|---|---|
| 模型 | `zai-org/GLM-5.3-Flash`（MIT 许可） |
| 规模 | 321B 总参 / 18B 激活，MoE 288 专家 top-8 |
| 结构 | 45 层 = 34 层 KDA 线性注意力 + 11 层 sparse MLA（rope-free / NoPE）；1 层 MTP |
| 权重 | 原生 FP8，HuggingFace 上实测 305.8 GiB |
| 最长上下文 | 1M token |

模型背景出处：vLLM 官方 recipe（recipes.vllm.ai/zai-org/GLM-5.3-Flash）。

## 2. 测试环境

| 项 | 内容 |
|---|---|
| 机型 | `g4-standard-384`（Spot），us-central1-b |
| GPU | 8× NVIDIA RTX PRO 6000 Blackwell **Server Edition**，每卡 96 GB（`nvidia-smi` 报 97,887 MiB），计算能力 SM120 |
| CPU / 内存 | 384 vCPU，2 个 NUMA 节点；1,417 GiB 内存 |
| 卡间互联 | **只有 PCIe**，无 NVLink。拓扑见下 |
| 系统盘 | 2,000 GB Hyperdisk Balanced（权重放这里） |
| 镜像 | `deeplearning-platform-release / common-cu129-ubuntu-2404-nvidia-580` |
| OS / 内核 | Ubuntu 24.04.5 LTS / 7.0.0-1014-gcp |
| 驱动 | NVIDIA 580.178.04 |
| 推理引擎 | `vllm/vllm-openai:nightly`，拉取于 2026-11-03，vLLM `0.33.0rc1.dev484+g76c6f3f2a`（digest 见第 5 节） |

GPU 拓扑（`nvidia-smi topo -m`）：

```
      GPU0 GPU1 GPU2 GPU3 GPU4 GPU5 GPU6 GPU7  NUMA
GPU0   X   PIX  NODE NODE SYS  SYS  SYS  SYS    0
GPU1  PIX   X   NODE NODE SYS  SYS  SYS  SYS    0
GPU2  NODE NODE  X   PIX  SYS  SYS  SYS  SYS    0
GPU3  NODE NODE PIX   X   SYS  SYS  SYS  SYS    0
GPU4  SYS  SYS  SYS  SYS   X   PIX  NODE NODE   1
GPU5  SYS  SYS  SYS  SYS  PIX   X   NODE NODE   1
GPU6  SYS  SYS  SYS  SYS  NODE NODE  X   PIX    1
GPU7  SYS  SYS  SYS  SYS  NODE NODE PIX   X     1
```

两两一对挂在同一个 PCIe 交换芯片下（PIX），四卡一个 NUMA 节点，跨 NUMA 走 SYS。
TP8 的 all-reduce 必然跨 NUMA，这是 G4 上做大 TP 的主要代价。

## 3. 方案选择

截至 2026-11-03，vLLM 官方 recipe 只覆盖 Hopper 和数据中心 Blackwell（B200 / GB200），**不包含 SM120**。
阻塞点是模型的 rope-free sparse MLA 在 SM120 上缺 kernel 和 KV 路径，跟踪于 vLLM issue #53963。社区有两条已跑通的路：

| | 路线 A：vLLM · FP8 · TP8 | 路线 B：SGLang · NVFP4 · TP2 |
|---|---|---|
| 权重 | 官方 FP8 原生权重 | 社区 NVFP4 量化 checkpoint |
| 卡数 | 8 卡 1 副本 | 2 卡 1 副本（8 卡可放 4 副本） |
| 长上下文 | 原作者验证到 1M | 原作者只验证到 64K |
| 依赖 | 社区补丁 / nightly | 社区 SGLang 构建 + 社区权重 |
| 本次 | **已测，见第 5 节** | 后续 |

**本次先测路线 A，理由**：权重是官方原版，不引入量化误差，客户复现时不依赖第三方 checkpoint。

### 镜像为什么换成 vLLM nightly

原作者报告里的镜像 `vllm/vllm-openai:glm53-sm120nope` **在 Docker Hub 上不存在**（2026-11-03 查询返回 404），是作者本地构建时起的名字，客户无法直接拉取。

替代选择：`vllm/vllm-openai:nightly`。依据是同一作者 2026-10-06 的跟进报告：nightly（`0.30.1rc1.dev539`）会自动选用原生 `FLASHINFER_MLA_SPARSE_SM120` 后端，**不开 MTP 时输出正确，1M 大海捞针通过**；开 MTP 时接受率为 0%，所以本次**不开 MTP**。

备选镜像（nightly 起不来时用）：社区 overlay `cstechdev/vllm:glm53-flash-nope-sm120-cu130-20260826-r1`（issue #53963 中有 TP4 端到端解码的报告）。

## 4. 测试方法

服务参数：

```
--tensor-parallel-size 8 --max-model-len 131072 --max-num-seqs 256
--gpu-memory-utilization 0.85 --kv-cache-dtype fp8
--tool-call-parser glm47 --reasoning-parser glm45
```

压测工具：镜像自带的 `vllm bench serve`，OpenAI chat 接口，随机数据集，`--ignore-eos` 保证每条请求都生成满长度。

| 场景 | 输入 token | 输出 token | 并发 | 请求数 |
|---|---|---|---|---|
| 短输入单路 | 1,024 | 512 | 1 | 8 |
| 短输入低并发 | 1,024 | 512 | 4 | 16 |
| 短输入中并发 | 1,024 | 512 | 16 | 64 |
| 短输入高并发 | 1,024 | 512 | 64 | 256 |
| 8K 输入单路 | 8,192 | 512 | 1 | 8 |
| 8K 输入中并发 | 8,192 | 512 | 16 | 64 |
| 32K 输入单路 | 32,768 | 256 | 1 | 8 |

## 5. 测试结果

每个场景的完整输出在 `results/bench-in<输入>-out<输出>-c<并发>.txt`，服务日志 `results/server.log`，环境信息 `results/env.txt`。

镜像 digest：`vllm/vllm-openai@sha256:88cc2118dab845fec1ff759c2fc8afa517cf8a668162ed8683b2f1b781372f57`（vLLM `0.33.0rc1.dev484+g76c6f3f2a`）。

### 5.1 基础场景

| 场景（输入 / 输出 / 并发） | 输出 tok/s | 总 tok/s | 平均 TTFT | P99 TTFT | 平均 TPOT | P99 ITL |
|---|---|---|---|---|---|---|
| 1K / 512 / 1 | 100.0 | 302 | 327 ms | 1,327 ms | 9.38 ms | 18.8 ms |
| 1K / 512 / 4 | 271.9 | 822 | 704 ms | 1,856 ms | 13.36 ms | 25.7 ms |
| 1K / 512 / 16 | 579.6 | 1,752 | 1,364 ms | 2,016 ms | 24.98 ms | 44.8 ms |
| 1K / 512 / 64 | 992.6 | 3,001 | 2,820 ms | 7,760 ms | 59.02 ms | 841.9 ms |
| 8K / 512 / 1 | 89.9 | 1,531 | 941 ms | 1,661 ms | 9.30 ms | 9.7 ms |
| 8K / 512 / 16 | 352.6 | 6,003 | 3,294 ms | 7,690 ms | 38.94 ms | 660.2 ms |
| 32K / 256 / 1 | 53.1 | 6,850 | 2,457 ms | 3,221 ms | 9.27 ms | 9.4 ms |

所有场景请求成功率 100%。

怎么读这张表：

- **单路 TPOT 稳定在 9.3 ms 左右**，输入从 1K 涨到 32K 几乎不变。这是混合结构的好处：34 层 KDA 线性注意力的状态大小固定，不随上下文变长。
- **并发上去后 P99 ITL 会跳到几百毫秒**（c64 时 842 ms）。原因是新请求的 prefill 和正在生成的请求挤在同一批里，生成被打断一下。对首字延迟不敏感、对流式平滑敏感的业务，可以调小 `--max-num-batched-tokens` 换更平滑的 ITL。
- 32K 单路的输出 tok/s 只有 53，是因为每条请求只生成 256 个 token，2.5 s 的 prefill 占了大头；看 TPOT 就知道 decode 本身没变慢。

### 5.2 8K 输入 / 1K 输出并发扫描

脚本 [`scripts/bench-8k1k.sh`](scripts/bench-8k1k.sh)；原始输出在 `results/8k1k/`（`.txt` 摘要 + `.json` 逐请求明细）。每档请求数 = 并发 × 4（至少 8），全部 `--ignore-eos` 生成满 1,024 token。整轮扫描用时 30 分钟。

| 并发 | 请求数 | 输出 tok/s | 总 tok/s | 每路 tok/s（1000 / TPOT） | TTFT 中位数 | TTFT P99 | 平均 TPOT | P99 ITL | 平均端到端 |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 8 | 98.8 | 891 | 107 | 0.83 s | 0.83 s | 9.32 ms | 10.7 ms | 10.4 s |
| 2 | 8 | 162.9 | 1,468 | 90 | 1.22 s | 1.64 s | 11.09 ms | 10.9 ms | 12.6 s |
| 4 | 16 | 247.3 | 2,229 | 71 | 2.31 s | 3.11 s | 14.11 ms | 13.6 ms | 16.6 s |
| 8 | 32 | 338.4 | 3,050 | 48 | 2.77 s | 6.00 s | 20.83 ms | 19.0 ms | 24.2 s |
| 16 | 64 | 440.6 | 3,971 | 31 | 2.68 s | 12.2 s | 32.76 ms | 655 ms | 37.1 s |
| 32 | 128 | 537.8 | 4,846 | 18 | 2.69 s | 23.9 s | 54.37 ms | 663 ms | 60.8 s |
| 64 | 256 | 652.5 | 5,880 | 11 | 2.73 s | 48.2 s | 89.61 ms | 716 ms | 100.2 s |
| 128 | 512 | 836.8 | 7,541 | 7 | 2.77 s | 97.3 s | 137.90 ms | 746 ms | 155.9 s |

失败请求数全部为 0。

怎么读这张表：

- **吞吐到 128 并发还在涨**（653 → 837 tok/s），没有到顶；代价是每路速度降到 7 tok/s。选并发要看业务能接受的每路速度：要「人眼阅读速度」（约 20 tok/s 以上）就把并发控制在 **32 以内**，此时整机输出约 540 tok/s。
- **TTFT 的 P99 随并发线性变大，中位数却稳定在约 2.7 s**。这是压测方式造成的：每档开始时一次性涌进「并发数」条请求，第一波 128 × 8K ≈ 1M token 要排队做 prefill。按 32K 单路测得的约 13K tok/s prefill 速度估算，最后一条要等约 80 s，和实测 P99 97 s 同一量级（推导值，非单独测量）。真实流量是陆续到达的，看中位数更接近实际体验。
- **从 16 并发起 P99 ITL 跳到 650 ms 以上**，原因同 5.1：新请求的 prefill 插进正在生成的批里。对流式平滑度敏感的场景可调小 `--max-num-batched-tokens`，或把 prefill 和 decode 拆到不同副本。
- 单路 TPOT 9.3 ms 与 5.1 中 1K、32K 输入时一致，再次说明 decode 速度基本不随上下文长度变化。

### 5.3 PD 分离：1P1D

> 进行中。两台 `g4-standard-384`：P（prefill，TP8）+ D（decode，TP8），vLLM `NixlConnector`，KV 走 TCP。
> 两机间实测带宽 173 Gbit/s（16 流）/ 49.7 Gbit/s（单流）。步骤见 RUNBOOK 第 9 节。

### 5.4 NVFP4（计划中）

> 下一步：同样的 1P1D 部署、同样的压测，把权重换成 NVFP4，再用 GSM8K 对比 FP8 与 NVFP4 的模型能力。

| 项 | 选择 | 理由 |
|---|---|---|
| NVFP4 权重 | `RedHatAI/GLM-5.3-Flash-NVFP4`（184.3 GiB，compressed-tensors 格式） | 只量化 MoE 专家的权重和激活，其余层和 MTP 保持 FP8；模型卡标明面向 vLLM。备选：`nvidia/GLM-5.3-Flash-NVFP4`（ModelOpt 量化，190.4 GiB） |
| 推理引擎 | vLLM，与 FP8 同一个镜像 digest | 引擎、PD 方式、压测参数都不变，**精度和性能的差别只来自权重**；SGLang 在 SM120 上目前只有社区构建 |
| 额外参数 | `--no-enable-flashinfer-autotune` | RedHatAI 模型卡给的参数；社区报告不加时 FlashInfer autotune 可能卡死 |
| 能力评测 | GSM8K 全量 1,319 题，5-shot，贪心解码，lm-evaluation-harness `0.4.13` | 两种精度用完全相同的设置，各自通过自己的 1P1D 部署评测 |

## 6. 已知限制与注意事项

- **社区阶段**：SM120 支持还在 vLLM nightly 里，没进正式版。上生产前要锁定镜像 digest，不要用浮动的 `nightly` 标签。
- **MTP 先别开**：nightly 原生 SM120 后端上 MTP 接受率 0%（原作者报告），开了反而更慢。
- **G4 是 Server Edition，跨机没有 GPUDirect RDMA**，所以不要做跨机 TP/PP，一台机器装下一个副本。
- **Spot 会被抢占**：压测或演示用 Spot 省钱，生产用按需或预留。注意 `g4-standard-48` 的预留是单卡机，装不下这个模型。
- 社区报告里的性能数字都是在**自建 RTX PRO 6000 机器**上测的，不是 G4 实测，以本报告的数字为准。

## 7. 参考

- vLLM issue #53963（SM120 支持的完整讨论，各方部署报告）
- vLLM PR #53906（模型支持）、#55738（NoPE sparse prefill 修复）、#55277（NoPE KV-cache writer）
- FlashInfer PR #5075
- Helix 博客：Running GLM-5.3-Flash on 2, 4 or 8 RTX PRO 6000 GPUs（路线 B）
- helixml/ramjet 仓库 `deploy/glm53_flash`（路线 B 部署配方）
