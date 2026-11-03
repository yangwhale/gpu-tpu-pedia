# GLM-5.3-Flash on Google Cloud G4（8× RTX PRO 6000）— 推理测试报告

**照着做一次跑通 → [RUNBOOK.md](RUNBOOK.md)**　｜　测试过程与决策记录 → [JOURNAL.md](JOURNAL.md)　｜　原始数据 → [results/](results/)

> **状态（2026-11-03）：** FP8 单机、FP8/NVFP4 1P1D、NVFP4 容量拆分、2P2D 与不分离对比、GSM8K 能力对比已完成；按到达速率的 TTFT 测试进行中。
> **给客户讲的 HTML 报告**由 [`scripts/report/build_report.py`](scripts/report/build_report.py) 从 `results/` 直接生成。

## 结论先行

| 项 | 结论 |
|---|---|
| 能不能跑 | **能**。vLLM nightly（`0.33.0rc1.dev484+g76c6f3f2a`）原生支持，零补丁，自动选用 `FLASHINFER_MLA_SPARSE_SM120` 后端 |
| 单路 decode | **约 100 tok/s**（TPOT 9.3 ms），输入从 1K 到 32K 基本不变 |
| 吞吐 | 1K 输入 / 512 输出、64 并发：**992 输出 tok/s**，3,001 总 tok/s |
| 8K 输入 / 1K 输出 | 单路 98.8 tok/s、TTFT 0.83 s；**并发 128 时输出 837 tok/s、总 7,541 tok/s**，每路约 7 tok/s（见 5.2） |
| PD 分离 1P1D（16 卡） | 甜点约 64 并发：930 tok/s，P99 ITL 161 ms（单机同负载 650 ms+），TTFT 中位数 1.7 s；纯吞吐不如两台单机各跑各的，优势在延迟平稳（见 5.3） |
| NVFP4（1P1D） | 8–64 并发吞吐比 FP8 高 23–30%、TPOT 低约 30%，KV 池翻倍；单路略慢 3%；甜点约 32 并发（见 5.4） |
| NVFP4 能力 | GSM8K 全量：FP8 93.33% / NVFP4 92.95%，差 0.38 个百分点，配对检验 p = 0.65，**无显著差别**（见 5.5，含局限说明） |
| P:D 配比（NVFP4，16 卡） | prefill 不随 TP 增长，TP4 每卡效率是 TP8 的 2 倍；P、D 必须同 TP；按容量配成 2P2D 全 TP4，128 并发 2,107 tok/s（1P1D 的 1.65 倍），P99 ITL 122 ms；同卡数不分离 4×TP4 吞吐最高 2,380 tok/s 但 P99 ITL 573 ms（见 5.6–5.7） |
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

### 5.3 PD 分离：1P1D（FP8）

两台 `g4-standard-384`：P（prefill，TP8）+ D（decode，TP8），vLLM `NixlConnector`，KV 走 TCP。两机间实测带宽 173 Gbit/s（16 流）/ 49.7 Gbit/s（单流）。
其余参数与单机完全相同；压测同 5.2（8K 输入 / 1K 输出，每档并发 × 4 条请求）。步骤见 RUNBOOK 第 9 节，原始数据在 `results/pd-1p1d/`。全部 0 失败。

| 并发 | 输出 tok/s | 总 tok/s | 每路 tok/s | TTFT 中位数 | TTFT P99 | 平均 TPOT | P99 ITL | 平均端到端 |
|---|---|---|---|---|---|---|---|---|
| 1 | 93.6 | 843 | 104 | 1.04 s | 1.95 s | 9.57 ms | 9.8 ms | 10.9 s |
| 2 | 164.4 | 1,481 | 92 | 1.18 s | 2.10 s | 10.82 ms | 13.4 ms | 12.4 s |
| 4 | 258.9 | 2,334 | 74 | 1.27 s | 4.38 s | 13.54 ms | 28.8 ms | 15.5 s |
| 8 | 382.9 | 3,451 | 54 | 1.27 s | 6.45 s | 18.52 ms | 43.7 ms | 20.9 s |
| 16 | 528.7 | 4,765 | 38 | 1.38 s | 12.8 s | 26.32 ms | 86.5 ms | 29.8 s |
| 32 | 691.0 | 6,227 | 25 | 1.46 s | 24.3 s | 39.77 ms | 145 ms | 45.2 s |
| 64 | 930.3 | 8,384 | 17 | 1.68 s | 48.3 s | 57.22 ms | 161 ms | 66.6 s |
| 128 | 1,196.8 | 10,785 | 16 | 31.2 s | 95.9 s | 63.80 ms | 105 ms | 102.3 s |

（每路 tok/s = 1000 / 平均 TPOT。）

#### 和单机比：要用同样的 16 张卡比

1P1D 用了 16 张卡，单机 TP8 只用 8 张。公平的对照是**两台单机各自独立服务、流量平分**（DP2）：总并发 c 时每台承担 c/2。下表 DP2 一列是用 5.2 单机数据换算的（吞吐 × 2，TPOT 取单机 c/2 档），**不是单独实测**。

| 总并发 | 1P1D 输出 tok/s | DP2 输出 tok/s（换算） | 1P1D TPOT | DP2 TPOT（换算） | 1P1D P99 ITL | DP2 P99 ITL（单机 c/2 档） |
|---|---|---|---|---|---|---|
| 16 | 528.7 | 676.8 | 26.3 ms | 20.8 ms | 86.5 ms | 19.0 ms |
| 32 | 691.0 | 881.2 | 39.8 ms | 32.8 ms | 145 ms | 655 ms |
| 64 | 930.3 | 1,075.6 | 57.2 ms | 54.4 ms | 161 ms | 663 ms |
| 128 | 1,196.8 | 1,305.0 | 63.8 ms | 89.6 ms | 105 ms | 716 ms |

结论：

- **纯看吞吐，同样 16 张卡，1P1D 在 64 并发以内不如两台单机各跑各的**（低 14–22%）。这个负载里 decode 才是大头（每条请求 1K 输出），P 机器大部分时间闲着。
- **1P1D 的价值在延迟质量**：
  - **生成不再被打断**：P99 ITL 在全部并发下都 ≤ 161 ms；单机从 16 并发起就跳到 650 ms 以上（新请求的 prefill 插进生成批次）。对流式输出、语音合成这类「卡一下就明显」的场景，这是决定性的差别。
  - **首字更快**：64 并发以内 TTFT 中位数 1.0–1.7 s，单机约 2.7 s。
  - **高并发时 TPOT 更低**：128 并发 63.8 ms 对 DP2 的约 89.6 ms。按 TPOT ≤ 65 ms 的服务目标算，1P1D 可以做到 1,197 tok/s，DP2 在 54→90 ms 之间线性插值约 1,150 tok/s（插值估计），1P1D 略好。
- **甜点在 64 并发左右**：930 tok/s、每路 17 tok/s、TTFT 中位数 1.7 s、P99 ITL 161 ms。
  到 128 并发，TTFT 中位数从 1.7 s 跳到 31 s，**瓶颈换到了 P**：D 侧每秒完成约 1.17 条请求（1,197 ÷ 1,024），对应 8K 输入要约 9.6K tok/s 的 prefill，已接近单台 P 的能力（单机 32K 输入实测约 13K tok/s，推导值）。要把并发再往上推，应该加 P（2P1D），而不是加 D。

### 5.4 NVFP4：1P1D 性能

权重换成 `RedHatAI/GLM-5.3-Flash-NVFP4`，其余全部不变：同一个镜像 digest、同样的两台机器和 1P1D 部署、同样的压测。唯一多加的参数是 `--no-enable-flashinfer-autotune`。原始数据在 `results/pd-1p1d-nvfp4/`。全部 0 失败。

| 项 | 选择 | 理由 |
|---|---|---|
| NVFP4 权重 | `RedHatAI/GLM-5.3-Flash-NVFP4`（184.3 GiB，compressed-tensors 格式） | 只量化 MoE 专家的权重和激活，其余层和 MTP 保持 FP8；模型卡标明面向 vLLM。备选：`nvidia/GLM-5.3-Flash-NVFP4`（ModelOpt 量化，190.4 GiB），本次未测 |
| 推理引擎 | vLLM，与 FP8 同一个镜像 digest | 引擎、PD 方式、压测参数都不变，**性能和精度的差别只来自权重**；SGLang 在 SM120 上目前只有社区构建 |
| MoE kernel | 日志显示自动选用 `FLASHINFER_CUTLASS` NvFp4 MoE 后端 | |

显存与容量（每卡，日志原文）：

| | FP8 | NVFP4 |
|---|---|---|
| 权重占用 | 39.38 GiB | **23.69 GiB** |
| 可用 KV 显存 | 38.14 GiB | **54.51 GiB** |
| KV 池（D 节点，整机） | 3,563,302 token | **7,126,605 token**（2 倍） |
| 权重加载时间 | 31.5 s | 21.4 s |

压测结果（8K 输入 / 1K 输出）：

| 并发 | NVFP4 输出 tok/s | FP8 输出 tok/s | 提升 | NVFP4 TPOT | FP8 TPOT | NVFP4 TTFT 中位数 | FP8 TTFT 中位数 | NVFP4 P99 ITL |
|---|---|---|---|---|---|---|---|---|
| 1 | 90.8 | 93.6 | −3% | 9.94 ms | 9.57 ms | 0.99 s | 1.04 s | 10.2 ms |
| 2 | 164.1 | 164.4 | 0% | 10.89 ms | 10.82 ms | 1.16 s | 1.18 s | 21.6 ms |
| 4 | 293.9 | 258.9 | +14% | 11.78 ms | 13.54 ms | 1.23 s | 1.27 s | 30.5 ms |
| 8 | 472.4 | 382.9 | +23% | 14.54 ms | 18.52 ms | 1.29 s | 1.27 s | 40.5 ms |
| 16 | 671.9 | 528.7 | +27% | 19.99 ms | 26.32 ms | 1.32 s | 1.38 s | 60.4 ms |
| 32 | 896.0 | 691.0 | +30% | 29.38 ms | 39.77 ms | 1.41 s | 1.46 s | 103.7 ms |
| 64 | 1,186.9 | 930.3 | +28% | 40.76 ms | 57.22 ms | 4.54 s | 1.68 s | 68.3 ms |
| 128 | 1,274.1 | 1,196.8 | +6% | 42.84 ms | 63.80 ms | 50.3 s | 31.2 s | 68.7 ms |

结论：

- **中高并发吞吐高 23–30%，TPOT 低约 30%**（8–64 并发）。NVFP4 的 MoE 权重只有 FP8 的一半，decode 读权重更快，省下的显存又让 KV 池翻倍。
- **单路不快反而略慢**（−3%）：1 并发时 TPOT 9.94 ms 对 9.57 ms。低并发场景没有收益，不要因为「4 bit 更小」就预期单路更快。
- **P 成为瓶颈的点提前了**：D 变快以后，64 并发时 TTFT 中位数已经到 4.5 s（FP8 1P1D 是 1.7 s），128 并发时吞吐只比 64 并发多 7%。按 D 侧每秒完成约 1.16 条请求（1,187 ÷ 1,024）算，需要约 9.5K tok/s 的 prefill，已到单台 P 的上限（推导值）。
- **NVFP4 1P1D 的甜点约在 32 并发**：896 tok/s、每路 34 tok/s、TTFT 中位数 1.4 s、P99 ITL 104 ms。想在 64 并发以上继续吃到 NVFP4 的吞吐，需要重新配 P:D 比例，见 5.6。

### 5.5 FP8 与 NVFP4 的模型能力对比（GSM8K）

**设置**：GSM8K 测试集全量 1,319 题，5-shot，贪心解码（temperature=0），lm-evaluation-harness `0.4.13`，走 `/v1/completions` 接口。
FP8 和 NVFP4 各自通过自己的 1P1D 部署评测，除权重外其余完全相同。脚本 [`scripts/eval/gsm8k.sh`](scripts/eval/gsm8k.sh)，原始结果与逐题记录在 `results/eval/`。

| 指标 | FP8 | NVFP4 | 差值 |
|---|---|---|---|
| strict-match | **93.33%** ± 0.69% | **92.95%** ± 0.71% | −0.38 个百分点 |
| flexible-extract | **93.40%** ± 0.68% | **93.03%** ± 0.70% | −0.37 个百分点 |

（± 为 lm-eval 报告的标准误。）

**逐题配对比较**（同一道题两种精度的对错，strict-match；flexible-extract 结果几乎一样，见 `results/eval/paired-analysis.txt`）：

| | 题数 |
|---|---|
| 两边都对 | 1,190 |
| 只有 FP8 对 | 41 |
| 只有 NVFP4 对 | 36 |
| 两边都错 | 52 |
| 抽取出的答案完全相同 | 1,227（93.0%） |

结论：

- **在 GSM8K 上看不出 NVFP4 掉能力**。差 0.38 个百分点，小于一个标准误；配对 McNemar 精确检验 p = 0.65，远达不到统计显著。两边「只有自己对」的题数（41 对 36）基本对称，说明差别是随机波动，不是 NVFP4 系统性地做错一类题。
- **两种精度确实给出不完全相同的输出**（7% 的题答案不同），但对错的总量持平。这符合预期：量化改变了数值，greedy 解码会在个别分叉点走向不同路径。
- 第三方参照：RedHatAI 在模型卡上报告这份 NVFP4 权重 GSM8K Platinum 97.74%、MATH-500 94.87%、AIME 2025 86.67%、GPQA Diamond 90.57%（他们的评测设置，与本次不可直接比较）。

**这个评测的局限，给客户讲时要说清楚**：

- GSM8K 是**偏简单、接近饱和**的数学应用题，对量化误差不够敏感；本次又是 5-shot、**不开思考**的补全方式，没覆盖模型最常用的长思考模式。
- 想更有把握，建议再加两类更敏感的测试：长思考的难题（AIME / MATH-500 / GPQA Diamond，开默认思考档，多 seed 取平均），以及客户自己的业务样本做 A/B 盲评。脚本结构可以直接复用。

### 5.6 P 和 D 怎么配：容量拆分测试（NVFP4）

1P1D 吞吐上不去，是因为 P 和 D 的能力不匹配。把两种能力拆开单独测（不开 PD，单个实例）：脚本 [`scripts/capacity.sh`](scripts/capacity.sh)，原始数据 `results/capacity/`。

| 实例 | prefill 吞吐（8K 输入，1 token 输出） | decode c32 | decode c64 | decode c128 | decode c256 |
|---|---|---|---|---|---|
| NVFP4 TP8（8 卡） | 9,454 tok/s | 1,145 tok/s · 26 ms | 1,554 · 38 ms | 2,658 · 45 ms | 3,569 · 67 ms |
| NVFP4 TP4 #1（4 卡） | 9,381 tok/s | 934 · 32 ms | 1,314 · 45 ms | 1,905 · 62 ms | 2,490 · 93 ms |
| NVFP4 TP4 #2（同机并行） | 9,039 tok/s | 929 · 32 ms | 1,307 · 45 ms | 1,897 · 62 ms | 2,490 · 93 ms |

（decode 列为「输出 tok/s · 平均 TPOT」；decode 测试用 8,184 token 的共享前缀命中缓存，prefill 近乎免费。）

- **prefill 能力不随 TP 增长**：TP4 和 TP8 都约 9.0–9.5K tok/s，一台机器拆成两个 TP4 做 prefill，能力翻倍（约 18.4K tok/s）。
- **1P1D 的 D 没吃饱**：一个 TP8 的 D 单独能在 128 并发出 2,658 tok/s，1P1D 实测只有 1,274。
- **配比**：8K/1K 每输出 1 token 要 prefill 8 token。一台 2×TP4 的 P 机器 ≈ 18.4K ÷ 8.2K ≈ 2.2 req/s；一台 D 机器在 TPOT 约 45 ms 时 ≈ 2,620 ÷ 1,024 ≈ 2.5 req/s → **按机器 P:D ≈ 1:1**。
- **P、D 必须同 TP**：试过「2 个 TP4 的 P + 1 个 TP8 的 D」，全部请求失败（KV 块大小随 TP 变化，NIXL 握手失败，见 RUNBOOK 排障）。所以两台机器的合理配置是 **2P2D 全 TP4**。

### 5.7 2P2D 与不分离（DP4）对比（NVFP4，16 卡）

两种部署都是两台机器、4 个 TP4 实例；不分离时每个实例兼做 prefill 和 decode，2P2D 时 2 个只做 prefill、2 个只做 decode。原始数据 `results/pd-2p2d-nvfp4/`、`results/dp4-nvfp4-colocated/`。

| 总并发 | 1P1D 输出 tok/s | 2P2D 输出 tok/s | 不分离 输出 tok/s | 2P2D TTFT 中位数 | 不分离 TTFT 中位数 | 2P2D P99 ITL | 不分离 P99 ITL | 2P2D TPOT | 不分离 TPOT |
|---|---|---|---|---|---|---|---|---|---|
| 16 | 671.9 | 790.8 | 973.1 | 1.44 s | 2.40 s | 49 ms | 14 ms | 17.8 ms | 14.3 ms |
| 32 | 896.0 | 1,123.6 | 1,431.7 | 1.50 s | 2.52 s | 60 ms | 31 ms | 24.7 ms | 19.8 ms |
| 64 | 1,186.9 | 1,524.5 | 1,895.7 | 1.54 s | 2.59 s | 111 ms | 558 ms | 35.7 ms | 30.3 ms |
| 128 | 1,274.1 | 2,106.9 | 2,379.8 | 1.74 s | 2.44 s | 122 ms | 573 ms | 49.5 ms | 48.8 ms |
| 256 | — | 2,478.4 | 2,927.7 | 33.19 s | 2.35 s | 92 ms | 659 ms | 58.9 ms | 79.3 ms |

- **配对比例后 PD 吞吐 +65%**：128 并发 2P2D 2,107 tok/s，对 1P1D 1,274。
- **但仍比不分离低 11–20%**：同样 16 张卡，不分离的 4×TP4 吞吐最高。PD 换来的是延迟质量：64 并发起 P99 ITL 111–122 ms 对 558–573 ms，TTFT 中位数 1.5–1.7 s 对 2.4–2.6 s。
- 256 并发时 2P2D 的 TTFT 中位数跳到 33 s：P 又到顶了，再往上需要 P 多于 D。
- 2P2D 并发 1 档（40.6 tok/s、TPOT 22.9 ms）含新起服务的预热，不代表真实单路；并发 2 档单路约 98 tok/s。
- 不分离的数据：4 个实例各配一个压测客户端同步加压，吞吐为四者相加，TTFT 中位数与 TPOT 为平均，P99 取最大。

### 5.8 按到达速率测 TTFT（2P2D 与不分离）

> 进行中（泊松到达，0.5–2.5 req/s）。第一次跑时每档用了同一个随机种子，后面的速率重发了前面的 prompt、命中前缀缓存，TTFT 偏低，已作废重测（见 JOURNAL）。

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
