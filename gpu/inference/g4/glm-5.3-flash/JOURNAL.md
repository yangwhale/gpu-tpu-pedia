# 测试过程记录

按时间记录每一步做了什么、为什么这么做、遇到什么。时间为 UTC（括号内为 HKT）。

## 2026-11-03

**02:35（10:35）调研结论**：vLLM 官方 recipe 不含 SM120；社区两条路线（vLLM FP8 TP8 / SGLang NVFP4 TP2）。选路线 A 先测，理由见 README 第 3 节。

**02:38（10:38）查预留**：项目里唯一的 G4 预留是 5 台 `g4-standard-48`（每台 1 卡），且 `specificReservationRequired: true`。
单卡装不下 306 GiB 权重，G4 跨机又没有 RDMA，几台单卡机拼不成 TP，所以这份预留用不上，改开 Spot `g4-standard-384`。

**02:40（10:40）建机**：us-central1-b 第一次就拿到 Spot 容量。8 卡、驱动 580.178.04、Ubuntu 24.04.5。

**02:41（10:41）镜像不存在**：原作者报告的 `vllm/vllm-openai:glm53-sm120nope` 在 Docker Hub 返回 404。同仓库存在的相关标签：

| 标签 | 最后更新 | 说明 |
|---|---|---|
| `nightly` | 2026-11-02 | 原生 SM120 后端，**本次采用** |
| `glm53-flash` / `glm53-flash-x86_64-cu130` | 2026-09-09 | 官方 GLM-5.3-Flash 镜像；社区报告在 SM120 上报 `pe_dim must be 64 for fp8_ds_mla` |

社区 overlay `cstechdev/vllm:glm53-flash-nope-sm120-cu130-20260826-r1`（2026-08-26）留作备选。

**02:43（10:43）开跑**：`scripts/run-all.sh` 后台执行：装 Docker → 拉镜像与下载权重并行 → 起服务 → 冒烟 → 压测。

**02:51（10:51）第一次失败**：下载步骤报 `No module named pip` → `No module named 'huggingface_hub'`。DLVM 的 Ubuntu 24.04 镜像里系统 python3 不带 pip。
改用 uv 建独立虚拟环境装 `huggingface_hub`（RUNBOOK 第 4 步、`run-all.sh` 已同步修改）。镜像这时已拉完（`nightly`，解压后 35.5 GB）。

**02:53（10:53）重跑**：下载速度约 2 GB/s，开头 60 秒下了 115 GB。`run-all.sh` 同时加了 `env.txt` 采集（镜像 digest、vLLM 版本、驱动、拓扑）。

**02:55（10:55）权重下完**：HuggingFace 下载 306 GiB 用时约 2.5 分钟（02:52:58 开始，02:55:28 进入起服务）。模型公开，不需要 HF token。

**02:57（10:57）放进同区域 GCS 桶**：从 VM 上 `gcloud storage rsync` 到 us-central1 的桶，77 秒传完（328,366,173,639 字节，与 HuggingFace 上合计 305.8 GiB 一致）。
随后实测从桶拷回 VM：**49 秒，平均 6.3 GiB/s**，比 HuggingFace 快约 3 倍。RUNBOOK 第 4 步加了 4B，`run-all.sh` 支持 `MODEL_URI`。
这期间服务已在加载：权重加载后每卡占用约 42.9 GB，正在做 JIT kernel 预热。

**03:02（11:02）服务就绪**：从 `docker run` 到可用约 6.5 分钟。日志确认自动选用 `FLASHINFER_MLA_SPARSE_SM120` + `fp8_ds_mla` KV，每卡权重 39.38 GiB、KV 38.14 GiB，KV 池共 4,441,368 token。预热时多次出现 shared memory broadcast 60 秒超时提示，没有影响。
vLLM 版本 `0.33.0rc1.dev484+g76c6f3f2a`，比原作者测过的 `0.30.1rc1` 新，**不需要任何补丁或环境变量**。

**03:02（11:02）冒烟**：思考模式下 200 token 全用在 reasoning，`content` 为空；加 `chat_template_kwargs.enable_thinking=false` 后直接返回正确中文答案。写进 RUNBOOK 第 6 步，避免客户误判为服务故障。

**03:02–03:10（11:02–11:10）基础压测完成**：7 个场景全部 100% 成功。单路 decode 约 100 tok/s，与原作者在自建机器上的 ~100 tok/s 一致；1K/512 c64 输出 992 tok/s。

**03:10（11:10）8K / 1K 并发扫描开始**，并发 1–128，脚本 `scripts/bench-8k1k.sh`。

**03:10–03:40（11:10–11:40）8K / 1K 并发扫描完成**：8 档全部成功，0 失败。吞吐 128 并发时 837 输出 tok/s，仍未饱和。高并发下 TTFT P99 很大（c128 时 97 s），中位数稳定在约 2.7 s，判断为每档开头一次性涌入请求造成的 prefill 排队（按 13K tok/s prefill 估算约 80 s，量级吻合）。
插曲：盯扫描的 watch 探针把 UTC 日志时间戳和 HKT 当前时间直接相减，误报「已跑约 3 小时」，实际 16 分钟。之后给探针写明时间戳为 UTC。

**03:50（11:50）开始 PD 分离（1P1D）**：新开第二台 `g4-standard-384`（同 zone）做 D，原机器做 P，各 TP8。镜像固定到前面测过的 digest。
确认镜像内 NIXL 1.5.2 已支持 hybrid（Mamba/KDA）模型的状态传输（`nixl/base_scheduler.py` 里有 `MambaSpec` 处理），不需要打补丁。代理用 vLLM 官方 NIXL 示例 `toy_proxy_server.py`（与镜像同 commit）。

**03:54（11:54）第二台机器准备**：从桶拷权重约 80 秒。`apt-get install nvidia-container-toolkit` 报版本不匹配（DLVM 预装并 hold 了 1.19.1），不影响 GPU 容器；但同一条命令里的 `iperf3` 被一起拖失败，已拆开。

**03:57（11:57）两机带宽**：iperf3 16 流 173 Gbit/s，单流 49.7 Gbit/s。

**03:58（11:58）P、D 启动**，日志确认 `NIXL is available`。

**04:03（12:03）1P1D 第一次失败**：P、D 都在引擎初始化末尾退出，根因 `AssertionError: 3-read Mamba conv transfer requires DS conv state layout. Set VLLM_SSM_CONV_STATE_LAYOUT=DS`。
日志里同时有大量 `CUDACachingAllocator ... OOM` 警告，容易误判为显存不够（盯任务的探针就这么误判了），但单机健康启动的日志里也有同样的警告，是 profile 阶段的正常现象。
修复：两边都加 `VLLM_SSM_CONV_STATE_LAYOUT=DS`，其余参数不变，重新启动。

**04:13（12:13）1P1D 起来了，但冒烟判失败**：P、D、代理都正常，请求能走通 P→D。冒烟请求 200 token 全花在思考上（`finish_reason: length`），脚本因 `content` 为空判失败。
细查 chat template：**这个模型没有 `enable_thinking` 开关**，传了被忽略；思考档位由 `reasoning_effort` 控制（`low` / `high`，默认 `max`）。早上单机冒烟加了 `enable_thinking=false` 能出答案，只是那次思考恰好短，不是开关生效。
改为 `"reasoning_effort": "low"`：实测 reasoning_tokens 为 0，直接出答案。RUNBOOK 第 6 步、排障表和 `run-1p1d.sh` 已更正。压测用 `--ignore-eos` 固定生成长度，不受影响。

**04:15–04:38（12:15–12:38）FP8 1P1D 并发扫描**进行中（结果待回填）。

**04:34（12:34）规划 NVFP4 + 能力对比**：客户关心 NVFP4 会不会掉能力。决定继续用 vLLM（同一镜像），只换权重为 `RedHatAI/GLM-5.3-Flash-NVFP4`，这样性能和精度的差别都只来自量化。
能力评测用 GSM8K（lm-evaluation-harness 0.4.13，5-shot，全量 1,319 题，temperature=0），FP8 和 NVFP4 都通过各自的 1P1D 部署跑，顺带验证 PD 链路的输出正确性。
流水线：FP8 扫描结束 → FP8 GSM8K（同时两台机器下载 NVFP4 权重）→ 切 NVFP4 起 1P1D → 并发扫描 → NVFP4 GSM8K。

**04:38（12:38）FP8 1P1D 扫描完成**：8 档 0 失败。甜点约 64 并发；128 并发时 TTFT 中位数跳到 31 s，判断为单台 P 的 prefill 能力到顶。同卡数下纯吞吐不如 DP2（换算），优势在 P99 ITL 平稳。

**04:38–04:43（12:38–12:43）FP8 GSM8K 第一次失败**：`lm_eval` 的 API 模型需要 `transformers` 加载 tokenizer，`lm_eval[api]` 不会自动带上（`ModuleNotFoundError: No module named 'transformers'`）。`eval/gsm8k.sh` 已补装。由于流水线已切到 NVFP4，FP8 GSM8K 改到 NVFP4 测完后切回 FP8 1P1D 再跑。
同时两台机器从 HuggingFace 下完 NVFP4 权重（185 GiB），并上传一份到同区域桶。

**04:44（12:44）NVFP4 1P1D 启动**，开始并发扫描。

**04:44–05:09（12:44–13:09）NVFP4 1P1D 扫描完成**：8 档 0 失败，vLLM 自动选 `FLASHINFER_CUTLASS` NvFp4 MoE 后端，没有遇到社区报告的 autotune 卡死（已按模型卡加 `--no-enable-flashinfer-autotune`）。
流水线随后误报 `NVFP4_FAILED`：判断语句经过 ssh 包装里的 `grep -v`，空输出时返回 1，把成功判成了失败，导致 NVFP4 GSM8K 被跳过。扫描数据本身完整。改为 FP8 GSM8K 跑完后再切回 NVFP4 补跑。

**05:09–05:28（13:09–13:28）GSM8K 能力对比完成**：切回 FP8 1P1D 跑 GSM8K（约 4 分钟），再切回 NVFP4 补跑。FP8 strict-match 93.33%、NVFP4 92.95%，差 0.38 个百分点；逐题配对 McNemar 精确检验 p = 0.65，无显著差别。局限：GSM8K 偏简单、本次不开思考，建议后续补长思考难题和客户业务样本。

**05:36–06:03（13:36–14:03）容量拆分测试**（为算 P:D 配比）：单独测一个实例的 prefill 能力（8K 输入、1 token 输出）和 decode 能力（8K 共享前缀命中缓存、1K 输出）。NVFP4 TP8：prefill 约 9.3K tok/s；TP4：约 9.0K tok/s——**prefill 能力不随 TP 增长**，一台机器拆两个 TP4 做 prefill 能力翻倍。decode（TPOT 约 45 ms）：TP8 c128 2,658 tok/s，2×TP4 c64 合计 2,620 tok/s。
按 8K/1K 每个输出 token 需 8 个 prefill token 推算，P:D 按机器约 1:1，即「一台 2×TP4 做 P + 一台做 D」。
插曲：chat 接口发 1 token 请求时，这个 token 可能落到 reasoning 字段，benchmark 判为失败（`Never received a valid chunk`），容量脚本改用 `/v1/completions`。

**06:14–06:30（14:14–14:30）不开 PD 的公平基线 DP4**：两台机器共 4 个 NVFP4 TP4 实例各自服务，每实例一个压测客户端同步加压。总并发 64 时输出 1,896 tok/s、128 时 2,380 tok/s——**明显高于 NVFP4 1P1D**（同卡数 1,187 / 1,274）。

**06:30–06:50（14:30–14:50）2P1D 第一次失败**：TP4 的 P 实例在 PD 模式下报 `No available memory for the cache blocks`（同样的 TP4 不开 PD 能正常起）。给 P 加 `--max-num-seqs 64 --gpu-memory-utilization 0.92` 后启动成功。盯任务的探针当时报的是 `SERVER TIMEOUT`，根因要看容器日志。
