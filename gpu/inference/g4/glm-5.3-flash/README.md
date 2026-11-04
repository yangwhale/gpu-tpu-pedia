# GLM-5.3-Flash on Google Cloud G4（8× RTX PRO 6000）— 推理测试报告

**复现步骤 → [RUNBOOK.md](RUNBOOK.md)**　｜　给客户讲的一页式报告 → [report.html](report.html)（细节以本文和 results/ 为准）　｜　原始数据 → [results/](results/)　｜　测试过程记录 → [JOURNAL.md](JOURNAL.md)

> 2026-11-03 实测。本文只保留最终配置的结果；配置不对或已被替代的中间数据在 `results/_archive/`，不用于结论。

## 结论

| 问题 | 结论 |
|---|---|
| 能不能跑 | **能，不用打补丁**。vLLM nightly `0.33.0rc1.dev484+g76c6f3f2a`（镜像 digest 固定）原生支持 SM120，单路约 100 tok/s |
| FP8 还是 NVFP4 | **NVFP4**。同延迟门槛下每卡吞吐约 1.4 倍（TP8 对 TP8）；GSM8K 93.33% 对 92.95%，配对检验 p = 0.65，无显著差别 |
| TP8 还是 TP4 | **TP4，每台机器跑 2 个实例**。prefill 能力不随 TP 增长，TP4 每卡效率是 TP8 的 2 倍；同门槛下每卡吞吐约 1.8 倍 |
| 不分离还是 PD 分离 | **看延迟门槛**。要首字快 / 宽松门槛：不分离；要生成不卡 / 严格门槛：2P2D（P99 ITL 约 60–70 ms 对 650 ms） |
| P:D 怎么配 | 8K 输入 / 1K 输出按机器 **1:1**（2P2D）；P 和 D **必须同 TP** |
| 对比 H200 | **G4 便宜的前提是用 FP4**：G4 NVFP4 每百万输出 token 约 $4.0–4.2，H200 FP8 约 $5.8（G4 便宜约 30%）；**两边都用 FP8 时 H200 更划算**（$5.8 对 G4 $10.3） |
| PD 网络 | **必须把 UCX TCP 分段调到 1m**，否则 KV 传输慢 3 倍；调优后单网卡可跑满 176 Gbps，暂不需要双网卡 |

## 推荐配置

| 配置 | 机器 | 每台机器 | 适用 |
|---|---|---|---|
| **A. NVFP4 不分离** | 1 台起 | 2 个 TP4 实例 + 负载均衡 | 吞吐优先、首字要快 |
| **B. NVFP4 2P2D** | 2 台一组 | 一台 2 个 TP4 的 P，一台 2 个 TP4 的 D | 生成不能卡（流式语音、数字人） |
| **C. NVFP4 4P2D** | 3 台一组 | 两台各 2 个 P，一台 2 个 D | 既要不卡、又要首字快 |
| 基线. FP8 单机 TP8 | 1 台 | 1 个 TP8 实例 | 精度对照 |

步骤见 [RUNBOOK.md](RUNBOOK.md) 第 3–6 节。

## 测试环境

| 项 | 内容 |
|---|---|
| 机型 | `g4-standard-384`（Spot）×1–3，us-central1-b |
| GPU | 8× RTX PRO 6000 Blackwell Server Edition（96 GB GDDR7，1,597 GB/s，SM120） |
| 卡间 / 机间 | PCIe Gen5，无 NVLink；两卡一对挂同一交换芯片、四卡一个 NUMA / 单网卡 iperf 173 Gbit/s（机型上限 2 块网卡 400 Gbps） |
| 镜像 / 驱动 | DLVM `common-cu129-ubuntu-2404-nvidia-580`，Ubuntu 24.04.5，驱动 580.178.04 |
| 模型 | `zai-org/GLM-5.3-Flash`：321B 总参 / 18B 激活，34 层 KDA + 11 层 sparse MLA，288 专家 top-8 |
| 权重 | FP8 官方原生 305.8 GiB；NVFP4 `RedHatAI/GLM-5.3-Flash-NVFP4` 184.3 GiB（只量化 MoE 专家） |
| 引擎 | `vllm/vllm-openai@sha256:88cc2118dab845fec1ff759c2fc8afa517cf8a668162ed8683b2f1b781372f57` |
| PD | vLLM NixlConnector，NIXL 1.5.2，UCX over TCP |
| 负载 | 8K 输入 / 1K 输出，随机 token，`--ignore-eos`；并发扫描看吞吐上限，**泊松速率扫描看真实延迟和性价比** |

## 1. 单机基线（FP8 TP8）

单路 TPOT 在 1K / 8K / 32K 输入下都约 9.3 ms（KDA 状态大小固定）。8K/1K 并发扫描：1 并发 98.8 tok/s，128 并发 837 tok/s；从 16 并发起 P99 ITL 超过 650 ms（prefill 插进生成批次）。数据：`results/fp8-tp8-single/`。

## 2. FP8 与 NVFP4

- **性能**（单机、不分离、泊松到达）：0.5 req/s 时 TPOT FP8 52.0 ms、NVFP4 30.6 ms；0.75 req/s 时 88.6 对 58.7 ms。每卡权重 39.38 → 23.69 GiB，可用 KV 38.14 → 54.51 GiB。数据：`results/rate-colocated-fp8-tp8/`、`results/rate-colocated-nvfp4-tp8/`。
- **能力**（GSM8K 全量 1,319 题，5-shot，贪心，lm-eval 0.4.13）：strict-match 93.33% 对 92.95%，flexible-extract 93.40% 对 93.03%；逐题只有 FP8 对 41 题、只有 NVFP4 对 36 题，McNemar p = 0.65。局限：GSM8K 偏简单且未开思考，上线前建议补长思考难题与业务样本 A/B。数据：`results/eval/`。

## 3. TP4 与 P:D 配比

| 实例（NVFP4） | prefill 吞吐（8K 输入） | decode：TPOT ≈ 45 ms 时 |
|---|---|---|
| TP8（8 卡） | 9,454 tok/s | 2,658 tok/s（并发 128） |
| TP4（4 卡），同机 2 个 | 9,381 + 9,039 tok/s | 1,314 + 1,307 tok/s（各并发 64） |

prefill 不随 TP 增长 → 每台机器 2×TP4。8K/1K 每输出 1 token 要 prefill 8 token：一台 P 机器 ≈ 2.2 req/s，一台 D 机器（TPOT ≈ 45 ms）≈ 2.5 req/s → **P:D ≈ 1:1**。P=TP4 / D=TP8 混合时 NIXL 握手失败（KV 块大小随 TP 变），**必须同 TP**。数据：`results/capacity/`。

## 4. 不分离与 PD 分离（NVFP4，全 TP4，PD 已做 UCX 调优）

| 总并发 | 每卡输出 tok/s（不分离 / 2P2D / 4P2D） | 平均 TPOT ms | P99 ITL ms | TTFT 中位数 s（并发扫描） |
|---|---|---|---|---|
| 16 | 61 / 52 / 36 | 14 / 17 / 17 | 14 / 31 / 26 | 2.4 / 1.2 / 1.1 |
| 32 | 89 / 76 / 53 | 20 / 22 / 22 | 31 / 48 / 48 | 2.5 / 1.3 / 1.2 |
| 64 | 118 / 106 / 75 | 30 / 31 / 32 | 558 / 59 / 60 | 2.6 / 1.4 / 1.2 |
| 128 | 149 / 146 / 108 | 49 / 40 / 43 | 573 / 68 / 75 | 2.4 / 6.7 / 1.3 |
| 256 | 183 / 157 / 158 | 79 / 42 / 56 | 659 / 70 / 89 | 2.3 / 51.1 / 1.4 |

- 2P2D 与不分离每卡吞吐接近，TPOT 更低，P99 ITL 低一个数量级；128 并发起两个 P 饱和。
- 4P2D 首字不排队（256 并发仍约 1.4 s），每卡吞吐低于 2P2D——8K/1K 下 2:1 的 P 有富余。

**泊松到达，每台机器 1 req/s 时**：不分离 TTFT 中位数 0.92 s / TPOT 40.9 ms / P99 ITL 653 ms；2P2D 1.56 s / 33.7 ms / 62 ms；4P2D 1.16 s / 44.3 ms / 74 ms。首字不分离最快，生成 2P2D 最稳。

数据：`results/colocated-4xtp4-nvfp4/`、`results/rate-colocated-nvfp4-2xtp4/`、`results/pd-2p2d-nvfp4/`、`results/rate-2p2d-nvfp4/`、`results/pd-4p2d-nvfp4/`、`results/rate-4p2d-nvfp4/`。

## 5. PD 网络调优

独立 NIXL 读取基准（`scripts/net/nixl_bench.py`，80 MB 拆 300 块）：

| 配置 | 80 MB 一次 | 吞吐 |
|---|---|---|
| CPU 内存，UCX 默认 | 28 ms | 24 Gbps |
| GPU 显存，UCX 默认 | 128 ms | 5.3 Gbps |
| GPU 显存，`UCX_TCP_TX_SEG_SIZE=1m UCX_TCP_RX_SEG_SIZE=1m` | 24.9 ms | 27 Gbps |
| 同上，8 对 agent 并行，持续约 10 s | — | 网卡计数器中位数 176 Gbps、峰值 183 Gbps（单网卡打满） |

放回 vLLM 后每次 KV 传输（每 rank 约 80–90 MB）从 379 ms 降到 121 ms。服务时一个 8K 请求约 0.35 GB（TP4），每秒 2 个请求只需约 6 Gbps，**暂不需要双网卡**。数据：`results/nixl-bench/`。

## 6. 按延迟门槛的性价比（每卡吞吐）

门槛（8K/1K）：**S1 严格流式** TTFT P90 ≤ 2 s、TPOT ≤ 25 ms、P99 ITL ≤ 200 ms；**S2 标准对话** TTFT P90 ≤ 3 s、TPOT ≤ 40 ms；**S3 宽松** TTFT P90 ≤ 5 s、TPOT ≤ 60 ms。泊松速率扫描逐档加压，取最后一个满足门槛的负载（与第一个不满足的负载间线性插值）。单元格：每卡输出 tok/s / 每卡总 tok/s；* 为最高速率仍满足。脚本 `scripts/report/goodput.py`。

| 部署 | 卡数 | S1 严格流式 | S2 标准对话 | S3 宽松 |
|---|---|---|---|---|
| FP8 单机 TP8（不分离） | 8 | **30** / 274 | **45** / 407 | **61** / 547 |
| NVFP4 单机 TP8（不分离） | 8 | **32** / 287 | **63** / 567 | **84** / 757 |
| NVFP4 单机 2×TP4（不分离） | 8 | **63** / 568 | **112** / 1,013 | **141** / 1,267 |
| NVFP4 2P2D（两机，全 TP4） | 16 | **74** / 667 | **116** / 1,047 | **132** / 1,194 |
| NVFP4 4P2D（三机，全 TP4） | 24 | **48** / 435 | **93** / 841 | **123** / 1,109* |
| H200 FP8 单机 TP8（不分离） | 8 | **106** / 952 | **203** / 1,829 | **234** / 2,112 |
| H200 FP8 单机 2×TP4（不分离） | 8 | **149** / 1,344 | **304** / 2,735 | **339** / 3,052* |

**只追吞吐**：不分离 4×TP4 在每实例 64 并发时每卡 183 tok/s（TPOT 79 ms）；甜点在每实例 32–64 并发（每卡约 150–180 tok/s）。

## 7. 对比 H200（A3 Ultra，FP8）

同一套脚本与负载，在 `a3-ultragpu-8g`（8 × H200）上实测 FP8（H200 不支持 FP4）。数据：`results/h200-fp8-tp8/`、`results/rate-h200-fp8-tp8/`、`results/h200-fp8-2xtp4/`、`results/rate-h200-fp8-2xtp4/`。

| 部署 | 每卡 Spot 价（us-central1） | S2 每卡输出 tok/s | 每百万输出 token | 单用户速度（1000/TPOT） |
|---|---|---|---|---|
| G4 NVFP4 单机 2×TP4 | $1.68 | 112 | $4.15 | 约 97 tok/s |
| G4 NVFP4 2P2D | $1.68 | 116 | $4.02 | — |
| G4 FP8 单机 TP8 | $1.68 | 45 | $10.33 | 约 107 tok/s |
| H200 FP8 单机 TP8 | $6.36 | 203 | $8.71 | 约 173 tok/s |
| H200 FP8 单机 2×TP4 | $6.36 | 304 | $5.82 | — |

- **同精度（FP8 对 FP8）H200 更划算**：G4 FP8 TP8 每百万输出 token $10.33，H200 FP8 TP8 $8.71、2×TP4 $5.82（G4 FP8 切 TP4 后每卡权重约 79 GB，KV 空间太小，未测）。
- **各用最优配置时 G4 便宜约 30%**：G4 NVFP4 $4.02–4.15 对 H200 FP8 $5.82。G4 的成本优势来自 FP4。
- H200 每卡吞吐约是 G4 最优配置的 2.6 倍，每卡价格 3.8 倍；单用户速度 H200 更快。
- H200 上同样是 2×TP4 比 TP8 每卡吞吐高约 50%。H200 KV 容量：TP8 整机 1,190 万 token。
- 价格取自 Google Cloud Spot 价格页 us-central1（2026-11-04）：A3 Ultra $50.87/小时、G4 $13.45/小时；H200 实测机器在 us-west1 / europe-west4（Spot 容量所限）。

## 已知限制

- SM120 支持在 vLLM nightly 中，未进正式版，生产请固定镜像 digest。
- 未开 MTP（社区报告 nightly 原生 SM120 后端上 MTP 接受率为 0）。
- 每个数据点一次运行，未重复取平均；Spot 机器，单 zone。
- NVFP4 为第三方量化（RedHatAI），NVIDIA 官方 `nvidia/GLM-5.3-Flash-NVFP4` 未测。

## 参考

- vLLM issue #53963（SM120 支持讨论）、PR #53906 / #55738 / #55277；FlashInfer PR #5075
- RedHatAI/GLM-5.3-Flash-NVFP4 模型卡
