# GB300 (A4X Max) · vLLM DeepSeek-V4 · Benchmark 对标报告

> **本文性质**：vLLM 在 GB300 上服务 DeepSeek-V4 的**官方 recipe + 已公开 benchmark 的对标整理**，与同目录 [`sglang-v4-gb300-benchmark.md`](./sglang-v4-gb300-benchmark.md)（含**我们实测跑通**的 8,903 output/decode-GPU）配套阅读。
> ⚠️ **与 SGLang 那份的关键区别**：SGLang 文档是我们在 GB300 上**亲自端到端跑通 + 复现**的；本 vLLM 文档的数字来自 **vLLM 官方 blog + SemiAnalysis InferenceX 公开榜单/recipe**，**尚未在本环境实跑**。要落地实跑另行安排（recipe 已就绪，见 §3）。

> 资料来源：vLLM 官方 blog「DeepSeek V4 in vLLM」(2026-04-24)、「DeepSeek-V3.2 on GB300」(2026-02-13)、vLLM recipe 站 `recipes.vllm.ai/deepseek-ai/DeepSeek-V4-Pro`、SemiAnalysis InferenceX `benchmarks/.../vllm/deepseek-v4/8k1k/*.yaml` + InferenceX 公开对比榜单。

---

## TL;DR

| 项 | 结论 |
|---|---|
| **vLLM 有无 GB300 对标** | ✅ 有，且很全：官方 V4 blog + recipe 站 + InferenceX 全套 GB300 vLLM recipe（Dynamo+vLLM）+ 公开 benchmark 榜单 |
| **官方口径最优（InferenceX，output÷decode-GPU）** | GB300 NVL72 DeepSeek-V4-Pro FP4 8K/1K：**68 tok/s/user 点 = 9,759 tok/s/GPU**；121 点 3,816；173 点 895（Dynamo+vLLM，2026-07-14 榜单） |
| **对标 SGLang** | SGLang 官方 11,200 output/decode-GPU @~50 tok/s/user（MTP 曲线）；两者操作点不同，**InferenceX 榜单是唯一 apples-to-apples 的跨框架来源** |
| **vLLM 满配拓扑** | disagg：`4p1d/5p1d/6p1d-dep4-dep8`（24/28/32 GPU，conc 4096）、`7p2d-dep4-dep16`（60 GPU 宽 EP，conc 3072）；均 Dynamo 编排 |
| **KV 传输** | **NixlConnector**（vLLM 自己的；SGLang 用 mooncake） |
| **MoE backend** | **deep_gemm_amxf4_mega_moe**（MXFP4 MegaMoE；对应 SGLang 的 megamoe） |
| **⭐ 本环境实跑最优做法（见 §8 P2d）** | NixlConnector + **NVLink cuda_ipc**（`UCX_TLS=cuda_copy,cuda_ipc,tcp` + `UCX_CUDA_IPC_ENABLE_MNNVL=y` + `--enable-cumem-allocator`）+ **vllm-router**（`--vllm-pd-disaggregation`）。KV 从 200MB/s→7-167 GB/s。**GB300 上 KV 走 NVLink，不是 RDMA/dmabuf/peermem** |

**一句话**：vLLM 对 DeepSeek-V4 在 GB300 上是 Day-0 支持、有完整 disagg recipe 和公开榜单。架构层（V4 的 CSA+HCA 注意力、FP4 MoE、MTP）与 SGLang 同源；差异在**运行时栈**：编排都用 Dynamo，但 vLLM 走 NixlConnector + deep_gemm_amxf4_mega_moe + DP/EP，SGLang 走 mooncake + megamoe + dep。

---

## 0. vLLM 的 V4 支持与架构实现（官方 blog 摘要）

vLLM 从 2026-04-24 Day-0 支持 DeepSeek-V4-Pro（1.6T）+ Flash（285B），1M context，原生 FP4 MoE 权重。V4 的长上下文注意力（与 SGLang 文档 §0 同源，模型层面）：

- **共享 K/V**（2× 省显存）+ 对注意力输出做 **inverse RoPE** 保正确性。
- **KV 跨 token 压缩**：`c4a`（8 token 加权、stride 4，≈1/4 压缩）+ `c128a`（128 token、stride 128，≈1/128）。
- **DSA 稀疏注意力**（c4a top-k=512，c128a top-k=8192）限制注意力算力。
- **短滑动窗口 128**（uncompressed 局部信息）。
- 效果：1M context BF16 KV 仅 **9.62 GiB/seq**（比 V3.2 式 61 层的 83.9 GiB 小 **8.7×**）；实际用 fp4 indexer + fp8 attention 再省 ~2×。

vLLM 实现要点（工程）：单一逻辑 block（256 native 位）统一五类 KV cache 到 **3 个 page-size 桶**；压缩器状态当 SWA KV 管理（复用 prefix-cache / disagg / CUDA graph / MTP）；kernel fusion（compressor+RMSNorm+RoPE+insert ~1.4-3×、inverse-RoPE+fp8 ~2-3×、Q norm+KV RoPE+K insert 10-20×）+ 多流并行。规划中：DeepGEMM MegaMoE kernel、paged prefill kernel。

---

## 1. 单节点 quickstart（8×B300 或 8×B200，官方 blog）

最简单的起步（单节点 DP8，含 FP4 indexer + MTP 可选优化）：

```bash
docker run --gpus all --ipc=host -p 8000:8000 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  vllm/vllm-openai:deepseekv4-cu130 deepseek-ai/DeepSeek-V4-Pro \
  --trust-remote-code --kv-cache-dtype fp8 --block-size 256 \
  --enable-expert-parallel --data-parallel-size 8 \
  --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE","custom_ops":["all"]}' \
  --attention_config.use_fp4_indexer_cache=True \
  --tokenizer-mode deepseek_v4 --tool-call-parser deepseek_v4 \
  --enable-auto-tool-choice --reasoning-parser deepseek_v4
```

Flash 版把 `deepseek-ai/DeepSeek-V4-Flash` + `--data-parallel-size 4`，可在 4×B200/B300 跑。**这是原型/冒烟用**；冲吞吐要上 disagg（§3）。

---

## 2. GB300 满配栈（vLLM vs SGLang 对照）

| 维度 | vLLM（本文） | SGLang（我们实跑那份） |
|---|---|---|
| 容器 | `vllm/vllm-openai:dsv4-megamoe-mxfp4-arm64-cu130-*` | `lmsysorg/sglang:nightly-dev-cu13-*` |
| 编排 | **Dynamo** + 专用 NATS/etcd 节点 | **Dynamo** + NATS/etcd（同） |
| KV 传输 | **NixlConnector**（`kv_role: kv_both`） | **mooncake** over 域内 NVLink（`MC_FORCE_MNNVL`） |
| MoE backend | **deep_gemm_amxf4_mega_moe**（MXFP4） | **megamoe** W4A4（`USE_FP4_ACTS`） |
| 并行 | DP + EP（`enable-expert-parallel`，TP1/PP1） | dep = TP/DP/EP 同号 + dp-attention |
| KV cache dtype | `fp8` | fp8（decode） |
| CUDA graph | prefill `enforce-eager`；decode `cudagraph_mode: FULL_DECODE_ONLY` | prefill cuda-graph-max-bs 1024；decode 1280 |
| tokenizer | `tokenizer-mode: deepseek_v4` | `--reasoning-parser deepseek-v4` + DSV4 tokenizer |
| 关键 env | `NCCL_NVLS_ENABLE`/`TORCH_SYMMMEM=NVSHMEM`/`VLLM_USE_NCCL_SYMM_MEM`/`VLLM_DSV4_MEGA_FP8_COMBINE` | `MC_FORCE_MNNVL`/`NCCL_MNNVL_ENABLE`/`SGLANG_OPT_DEEPGEMM_MEGA_MOE_*` |
| 特有优化 | `enable-ep-weight-filter`、`enable-sleep-mode`、`use_fp4_indexer_cache`、`safetensors-load-strategy: prefetch` | SWA opts、MTP EAGLE、SGLANG_OPT_SWA_* |

**共同点**：都是 Dynamo 编排 + FP4 MoE + DP/EP wide-EP + sa-bench 8K/1K 开环压测 + `output÷decode-GPU` 口径。**差异**主要在 KV 连接器（Nixl vs mooncake）和 MoE kernel 命名（本质都是 MXFP4 MegaMoE）。

---

## 3. GB300 disagg recipe（InferenceX，照抄即可）

InferenceX 的 vLLM DeepSeek-V4-Pro GB300 recipe（`benchmarks/multi_node/srt-slurm-recipes/vllm/deepseek-v4/8k1k/`），拓扑命名 `NpMd-depX-depY`：

| Recipe | Prefill | Decode | GPU 数 | conc | 用途 |
|---|---|---|---|---|---|
| `disagg-gb300-4p1d-dep4-dep8-24-c4096` | 4×dep4 | 1×dep8 | 24 | 4096 | max-tpt 入门 |
| `disagg-gb300-5p1d-dep4-dep8-28-c4096` | 5×dep4 | 1×dep8 | 28 | 4096 | |
| `disagg-gb300-6p1d-dep4-dep8-32-c4096` | 6×dep4 | 1×dep8 | 32 | 4096 | **max-tpt 点** |
| `disagg-gb300-7p2d-dep4-dep16` | 7×dep4 | 2×dep16 | 60 | 3072 | 宽 EP decode |
| `disagg-gb300-1p6d-dep4-tp4` / `1p9d-tep4-tp4` | 1 | 6/9 | — | — | 低延迟点 |

**共同参数**（prefill / decode）：
- 模型 `deepseek-v4-pro`（FP4），`kv-cache-dtype fp8`，`block-size 256`，`gpu-memory-utilization 0.9`，`tokenizer-mode deepseek_v4`，`trust-remote-code`。
- `tensor-parallel-size 1` `pipeline-parallel-size 1` `enable-expert-parallel` + `data-parallel-size = <dep 号>`（prefill dep4→dp4，decode dep8→dp8 / dep16→dp16）。
- `kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_both"}'`。
- `moe-backend deep_gemm_amxf4_mega_moe`，`enable-ep-weight-filter`，`enable-sleep-mode`，`no-enable-prefix-caching`，`no-enable-flashinfer-autotune`，`no-disable-hybrid-kv-cache-manager`。
- **prefill 专属**：`enforce-eager: true`（不做 decode cuda graph）、`max-num-seqs 16~256`、`max-num-batched-tokens 16384`、`no-async-scheduling`。
- **decode 专属**：`max-num-seqs 384~512`、`compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","mode":0}'`、`max-cudagraph-capture-size 512`、`stream-interval 50`；7p2d decode 还加 `all2all-backend flashinfer_nvlink_one_sided`。
- env（P+D 都加）：`VLLM_USE_NCCL_SYMM_MEM=1 NCCL_CUMEM_ENABLE=1 NCCL_MNNVL_ENABLE=1 NCCL_NVLS_ENABLE=1 TORCH_SYMMMEM=NVSHMEM VLLM_DSV4_MEGA_FP8_COMBINE=1`。
- benchmark：`sa-bench`，ISL 8192 / OSL 1024，`req_rate inf`（开环），`use_chat_template`，`tokenizer_mode deepseek_v4`。

**编排前置**（同 SGLang §3.3）：专用 NATS/etcd 节点 + `pip install ai-dynamo`（wheel `1.2.0.dev20260426`）+ worker 走 Dynamo。**Dynamo 单 frontend**（`enable_multiple_frontends: false`）——注意这跟我们 SGLang 实测「多 frontend +34%」不同，vLLM recipe 默认单 frontend，是否也受单 frontend CPU-bound 限制值得实测验证。

---

## 4. 已公开 benchmark（InferenceX 榜单）

SemiAnalysis InferenceX **DeepSeek-V4-Pro 1.6T · FP4 · 8K/1K · GB300 NVL72 · Dynamo+vLLM**（2026-07-14 更新，**output÷decode-GPU** 口径）：

| 交互点 (tok/s/user) | 吞吐 (tok/s/GPU) | 成本 ($/M tok) | 并发 |
|---|---|---|---|
| 68 | **9,759** | $0.075 | ~1024 |
| 121 | 3,816 | $0.196 | ~315 |
| 173 | 895 | $0.841 | ~29 |

对照 B200（同榜单）：68 点 B200 只有 3,177 → **GB300 比 B200 高 207% tok/s/GPU、便宜 124%/token**。GB300 288GB HBM（vs B200 192GB）能塞更宽的 prefill+decode recipe，是关键。

> ⚠️ **口径提示**（InferenceX 官方注）：disagg 配置按 **per-decode-GPU** 算，与 aggregated 配置的 per-total-GPU **不可直接比**。

---

## 5. 对标 SGLang：怎么看这两个数

| | vLLM（InferenceX 榜单） | SGLang（pytorch 官方 + 我们实跑） |
|---|---|---|
| 最优 output/decode-GPU | 9,759 @ **68** tok/s/user | 官方 11,200 @ **~50** tok/s/user；我们实跑 **8,903**（80%） |
| 操作点 | 交互性更高（68 vs 50） | 交互性略低、吞吐更高 |
| MTP | GB300 8k1k recipe **默认未开**（gb200 有 mtp2 变体） | dep8-**MTP** 是 11,200 的核心杠杆 |
| 编排 | Dynamo 单 frontend | Dynamo 多 frontend（我们实测 +34%） |

**关键洞察**：
1. **不能直接说"谁更快"**——9,759@68 和 11,200@50 是曲线上不同交互点；同一 tok/s/user 下才可比，得看 InferenceX 同图叠加。InferenceX 榜单是唯一 apples-to-apples 的跨框架来源。
2. **SGLang 11,200 靠 MTP**（@50 低交互点吞吐拉满）；vLLM GB300 8k1k recipe 默认没开 MTP，走高交互点。若 vLLM 也上 MTP + 压到 50 tok/s/user，理论上吞吐还能上探——可作为实跑实验。
3. **两者架构同源**（V4 CSA+HCA + FP4 MoE + Dynamo），gap 更多在**运行时成熟度 + 配方调优**，跟我们 SGLang 那份「剩余 20% = 镜像内核成熟度」的结论一致。

---

## 6. 若要在本环境实跑 vLLM（下一步）

复用 SGLang 那份的 GB300 fleet（同 18 节点、node-local SSD、Dynamo NATS/etcd），换成 vLLM：
1. **换容器**：`vllm/vllm-openai:dsv4-megamoe-mxfp4-arm64-cu130-*`（arm64/cu130），重跑 bootstrap（GIB/DOCA + `pip install ai-dynamo`）。
2. **权重**：原版 `deepseek-ai/DeepSeek-V4-Pro`（FP4，同 SGLang，node-local SSD 复用）。
3. **起 workers**：照 §3 的 6p1d-dep8（32 GPU）或按我们 fleet 规模选 recipe，`python -m dynamo`/vLLM disagg + NixlConnector。
4. **压测**：同一 sa-bench（ISL 8192/OSL 1024/req_rate inf/use_chat_template/DSV4 tokenizer），**output÷decode-GPU** 口径，直接和 SGLang 的 8,903 同口径对比。
5. **可做的实验**：① 开 MTP（`--speculative-config.method mtp --num_speculative_tokens 1`）压到 ~50 tok/s/user 看能否逼近/超过 SGLang；② 多 frontend（vLLM recipe 默认单 frontend，验证是否也吃 +34%）；③ NixlConnector vs mooncake 的 KV 传输开销对比。

---

## 7. 实跑准备：踩坑蒸馏 + 消融矩阵（目标：一次跑对）

把 SGLang 实跑那份的全部坑蒸馏过来，让 vLLM 首跑就避开；再列覆盖性消融矩阵，从最简单到满配，逐项隔离每个改动的影响。

### 7.1 SGLang 踩坑蒸馏 → vLLM

**A. 栈无关坑（直接适用，vLLM 一样会遇到）**：
1. **decode 死进程泄漏 GPU 显存**（`kill -9` 回收不了、`--query-compute-apps` 为空）→ `kubectl delete pod --force` 重建（模型在 node SSD 持久）。
2. **`pkill python3` 会连 frontend 一起杀** → frontend 必须在 worker 全稳定后**统一最后起**，别边起边补。
3. **Dynamo「circuits open」真根因常是僵尸进程霸占端口** → `/proc/net/tcp` 反查 PID `kill -9` 后再起。
4. **自愈部署循环（一次成功的关键）**：启动后进「校验-重试」循环，没加载的 pod 等 **≥90s** 让内核 reap 掉 D-state 再 killport 重启，2-3 轮自清、无需重建；**别急着重试**（<10s 连撞会误判）。frontend 统一最后起。
5. **选域**：18 节点 NVL72、按「无 GPU-pod 空闲节点数」扫，别只看 Ready+label；ComputeDomain 只收敛 15/18 → 删卡住 pod 重建；节点 DRA 是 ipvlan → 重建该节点。
6. **存储 + bootstrap**：权重放 node-local SSD RAID；换容器必重跑 GIB/DOCA/ai-dynamo bootstrap。
7. **ready 判据**：用 GPU mem（prefill loaded=mem>200G）+ 日志注册串，别信 stdout 缓冲的 "fired up"。

**B. vLLM 特有、首跑必验证的点（SGLang 没有的差异）**：
1. **启动方式**：disagg worker 走 `python3 -m dynamo.vllm`（对应 dynamo.sglang），vllm_config args 透传（见 §3）。**首跑确认模块名 + 透传格式**。
2. **NixlConnector**（vLLM 的 KV 传输，非 mooncake）：需每节点设 `VLLM_NIXL_SIDE_CHANNEL_HOST=<本节点IP>`；可选 `UCX_NET_DEVICES` 指定 RDMA 网卡。`kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_both"}'`。
3. **端口**：vLLM 用 `data-parallel-rpc-port 13345` 等（非 SGLang 的 40236/5000）；端口僵尸机制类似，但 hex 值不同，killport 参数要换。
4. **容器 bootstrap**：`vllm/vllm-openai:dsv4-megamoe-mxfp4-arm64-cu130-*` 是否自带 GIB/DOCA 待确认；不带则同 SGLang 流程补。
5. **prefill `enforce-eager`**（不做 cuda graph）→ 启动更快但 prefill 峰值可能略低；decode 用 `FULL_DECODE_ONLY` cuda graph。启动行为与 SGLang 不同，ready 判据相应调整。

### 7.2 消融测试矩阵（从最简单到满配，逐项隔离）

> 原则同 SGLang §3.0：**锁死拓扑做 baseline，一次只翻一个开关**，拿干净的 per-step delta。全程官方口径（8K/1K、sa-bench 开环、output÷decode-GPU）。

| Phase | 配置 | 目的 / 测什么 |
|---|---|---|
| **P0 单节点冒烟** | 8×B300 DP8（§1 docker 命令） | 证明容器能加载 + 生成（"Paris"）。最简单起步。 |
| **P1 单节点 benchmark** | 单节点 DP8，sa-bench 8K/1K conc 扫描 | 拿单节点 per-GPU baseline；对照 SGLang Phase 2（2,794）。 |
| **P1 消融** | 在 P1 上逐项翻：① MoE backend（deep_gemm_amxf4_mega_moe vs 默认 flashinfer）② FP4 indexer（`use_fp4_indexer_cache` on/off）③ MTP（`--speculative-config.method mtp --num_speculative_tokens 1` on/off）④ kv-cache-dtype（fp8 vs bf16）| **每招值多少 tok/s** |
| **P2 最小 disagg** | 1 prefill dep4 + 1 decode dep8（12 GPU）+ Dynamo + NixlConnector | 打通 PD 链路（这步验证 §7.1-B 全部 vLLM 特有点）。 |
| **P3 拓扑扫描** | 4p1d / 5p1d / **6p1d**-dep4-dep8（conc 4096）；再 7p2d-dep4-dep16（conc 3072） | **prefill 数收敛点**（对照 SGLang 14→16 收敛）；dep8 vs dep16 宽 EP 的 per-decode-GPU 差异。 |
| **P4 满配参数消融** | 固定 P3 最优拓扑，逐项翻：① MTP on/off（**vLLM GB300 recipe 默认关，这是最大问号**）② multi-frontend（recipe 默认单，验证是否吃 SGLang 那 +34%）③ all2all-backend（`flashinfer_nvlink_one_sided`）④ prefill 扩到 8/12/16（看是否像 SGLang 需要更多 prefill 喂饱 dep8）| 定位每个参数对满配吞吐的影响 |
| **P5 对标定稿** | 最优 vLLM 配置 vs SGLang 8,903（同交互点、同口径） | 得出 vLLM↔SGLang 在本环境的真实对比 + 各自最优配方 |

### 7.3 一次跑对的执行方式

复用 SGLang 的自愈部署脚本骨架（改 vLLM 启动命令 + 端口 + NixlConnector env）：清空 → 一把启动全 worker → 90s 轮询校验-重试 → decode 就绪 → frontend 统一起验 → 探活 → sa-bench。**从 P0 单节点开始**，每 Phase 跑通再进下一个，避免一上来满配踩多个坑难定位。

---

## 8. 本环境实跑记录（进行中，2026-07-22）

> 从 §7 消融矩阵按 Phase 推进，本环境（GB300 A4X Max，4卡/节点）实测。

### P0 单节点冒烟 ✅
- 容器 `vllm/vllm-openai:dsv4-megamoe-mxfp4-arm64-cu130-4ba0a72`（arm64）单节点 **DP4+EP**（4卡，非 8卡 DP8——A4X Max 每节点仅 4 卡），原版 `/mnt/ssd/DeepSeek-V4-Pro`（806G FP4）+ `moe-backend deep_gemm_amxf4_mega_moe` + `kv-cache-dtype fp8`，生成正确（"Paris"，fingerprint `dp4-ep`）。**单节点不需要 GIB/DOCA bootstrap**（无跨节点 RDMA）。
- **⭐ 新坑（vLLM V4 特有）**：冷启动（torch.compile inductor + DeepGEMM warmup 1666 shapes + cuda graph capture 51 sizes）**超默认 600s 超时**，ApiServer 报 `TimeoutError: engine core ... Waited 600s` 后 `died with exit code 1`。**必须 `export VLLM_ENGINE_READY_TIMEOUT_S=3600`**。compile cache 指到 SSD（`VLLM_CACHE_ROOT=/mnt/ssd/vllm-cache`）→ 第二次启动快很多。

### P1 单节点 benchmark（DP4，8K/1K，`vllm bench serve`，口径 total in+out ÷ 4 GPU）

| 并发 | Total tok/s | **tok/s/GPU (÷4)** | Output tok/s | Median TPOT |
|---|---|---|---|---|
| 1 | 528 | 132 | 59 | 16.8 ms |
| 16 | 4,991 | 1,248 | 554 | 27.4 ms |
| 64 | 12,252 | **3,063** | 1,361 | 43.7 ms |

**vs SGLang Phase 2 单节点 TP4**（同 8K/1K 同口径）：conc1 209 / conc16 1,898 / conc64 2,794。

**结论（单节点交叉）**：**低并发 SGLang 快**（conc1 209 vs 132、conc16 1,898 vs 1,248——TP 延迟优势）；**高并发 conc64 vLLM 反超 +10%**（3,063 vs 2,794——DP+EP 的吞吐扩展比纯 TP 好）。这跟 vLLM 官方"V4 初版、优化进行中"一致，且 vLLM 的 wide-EP 天然更吃高并发。

### P2 disagg 最小验证 1p1d-dep4 ✅（跨节点跑通，2026-07-22）

**架构**：`v4v-p0` prefill(dep4, node A) + `v4v-d0` decode(dep4, node B) + `v4v-d1` frontend。KV 从 prefill 跨节点传到 decode，走 CX-8。

**配置要点**（照官方 InferenceX `4p1d-dep4-dep8` recipe 缩到 1p1d-dep4）：
- **NIXL 配法**：这个 build 把 connector 重构成 `vllm.distributed.kv_transfer.kv_connector.v1.nixl` 子包（导出 `NixlConnector`）。用 vLLM 原生 `--kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_both"}'`，**不是** dynamo `--connector` flag。旧路径 `...v1.nixl_connector` 已不存在。
- **NATS/etcd 复用**：集群 `dynamo-nats`(4222)/`dynamo-etcd`(2379) 服务在，worker 靠 `NATS_SERVER` + `ETCD_ENDPOINTS` env 连、经 etcd 自动发现。
- **ai-dynamo**：recipe pin 的 `1.2.0.dev20260426` 已从 PyPI 下架 → 用 **`ai-dynamo==1.2.1`**。
- **side channel**：`VLLM_NIXL_SIDE_CHANNEL_HOST=$(hostname -i)` + `VLLM_NIXL_SIDE_CHANNEL_PORT`。
- **端到端验证**：frontend `curl /v1/chat/completions` → prefill → KV 跨节点 → decode → 正确返回。

**⭐ 核心发现：vanilla 容器就能跨节点 disagg，不需要 GIB/DOCA**
- 容器**自带完整 RDMA userspace**：`libibverbs.so`/`libmlx5.so`（rdma-core 39.0）+ `ibv_reg_dmabuf_mr` 符号 + `/dev/infiniband/uverbs0-7`（8× CX-8 经 mrdma DRA）。此前"容器缺 RDMA 栈"的判断是错的。
- 缺的只有 **GIB**（那是 NCCL 用的，NIXL 不依赖）+ 诊断工具。

**⭐⭐ 但 KV transfer 是瓶颈：~200 MB/s（无 GPUDirect）**

| 项目 | 值 |
|---|---|
| 每次 KV transfer | 81.5 MB（8K token, FP8） |
| **实测吞吐** | **140–310 MB/s**（CX-8 线速 ~50 GB/s，慢 ~200×） |
| 每次 xfer 耗时 | 200–800 ms（直接拉高 TTFT） |
| disagg 1p1d-dep4 吞吐 | conc16 = 574 / conc64 = 1,693 output tok/s |

**根因**：NIXL 回退到 **cuda_copy 主机中转**（GPU→host→RDMA→host→GPU），没走 GPUDirect RDMA 零拷贝。日志：`GDAKI not supported, please load nvidia_peermem` + `mlx5_0~7: GPU-direct RDMA is not available (GDA_DMABUF_ENABLE=try)`。

**GPUDirect 三次验证（全指向同一结论）**
1. vanilla：200 MB/s（cuda_copy）
2. 加 `UCX_IB_GPU_DIRECT_RDMA=yes` + `UCX_TLS=^gdaki` 重启：**KV 仍 200 MB/s，零改善**（conc16 610 ≈ baseline 574）
3. 结论：**纯 env 调优无法开 GPUDirect**。NIXL 自带 UCX 通过 GDAKI 探测 GPUDirect，需 nvidia_peermem 或可用 GDA_DMABUF；vanilla 容器 GPU 驱动是 nvidia-open 580（理论支持 dmabuf 导出）+ rdma-core 有 dmabuf 符号，但 UCX 的 GDAKI 探测失败 → 禁用 GDR

**下一步（要真吞吐，二选一，均需 bootstrap）**
- **A. nvidia_peermem**：host 节点 `modprobe nvidia_peermem`（需 node 级访问/特权 DaemonSet；容器内无模块文件，`modprobe` 不可用）
- **B. DOCA OFED**：容器内装 DOCA OFED userspace，提供 GDA/dmabuf 可用的 mlx5 栈让 UCX GDAKI 探测通过

**踩坑记录**
- **pod 重建丢 pip 包**：`ai-dynamo` 装在容器 root FS，pod 重建（换新容器）后消失（模型在 hostPath `/mnt/ssd` 持久，pip 包不持久）→ 重建后必须重装。教训：dynamo 应进镜像或 initContainer。
- **pkill 留 GPU 显存**：`pkill -f dynamo.vllm` 只杀 launcher，multiproc 的 Worker 子进程（进程名不含 dynamo.vllm）不被杀，仍各占 ~256 GiB → 下次启动 OOM。**正确清法**：`nvidia-smi --query-compute-apps=pid` 拿 GPU 占用 PID 逐个 `kill -9` + `pkill -9 -f dynamo`，几秒后显存归 0。**仅当进程已死但 `ps` 查不到、显存仍不回收（真僵尸 CUDA context）才需重建 pod**（容器 PID ns 销毁内核回收）。
- **nodeName 破 DRA**：重建时用 `nodeName` 钉节点会**绕过 scheduler**，DRA ResourceClaim 卡 pending（DRA 分配靠 scheduler）。必须用 `nodeAffinity`（`kubernetes.io/hostname`）钉节点，保留 scheduler。

### P2b MooncakeConnector（NVLink KV，对齐 SGLang，2026-07-22）

**为什么换 mooncake**：NixlConnector 走 UCX RDMA 需 nvidia_peermem 内核模块（DOCA userspace 补不了），KV 卡 200MB/s。GB300 NVL72 的正解是 KV 走**域内 NVLink**（mooncake，同 SGLang）。DOCA OFED + GIB 是 mooncake 前置（见 §7 bootstrap）。

**打通 mooncake 的三个坑**：
1. **libcudart.so.12**：`mooncake-transfer-engine 0.3.9` wheel 是 CUDA 12 编译，vLLM 容器 CUDA 13 → `from mooncake.engine import TransferEngine` 报缺 `libcudart.so.12`。修：`pip install nvidia-cuda-runtime-cu12`，`LD_LIBRARY_PATH` 加 `.../nvidia/cuda_runtime/lib`。
2. **MC_FORCE_MNNVL=1**：只配 `mooncake_protocol=rdma` 会 `Mooncake Transfer Engine initialization failed`。GB300 必须 `export MC_FORCE_MNNVL=1`（+ `NCCL_MNNVL_ENABLE=1 NCCL_CUMEM_ENABLE=1`）走域内 NVLink → init 成功（日志 `nvlink_transport.cpp` + `Mooncake Transfer Engine Scheduler`）。
3. **connector 配置**：prefill `kv_role=kv_producer` / decode `kv_role=kv_consumer`（**不是** Nixl 的 kv_both）+ `kv_connector_extra_config.mooncake_protocol=rdma`。

**⭐ 架构卡点：dynamo.vllm ⊥ MooncakeConnector**。dynamo 的 disagg 靠 prefill 返回 `res.kv_transfer_params`（NixlConnector 填，走 response）；MooncakeConnector 不填（用自己的 bootstrap server 带外协调）→ dynamo `handlers.py:_build_disaggregated_params` 拿 None → decode 报 `missing disaggregated_params`，生成 500。**mooncake engine 层已 work（NVLink init 成功），但 dynamo 编排层对不上**。

**解法：vLLM 原生 mooncake proxy**（弃 dynamo）。mooncake pip 包自带 `vllm_v1_proxy_server.py`。prefill/decode 用纯 `vllm serve`（非 dynamo.vllm）各暴露 /v1，proxy `--prefiller-hosts/-ports` + `--decoder-hosts/-ports` 路由。**端到端跑通**（"Paris"正确）。

**⭐ 实测结果（1p1d-dep4，8K/1K）— mooncake NVLink 反而更慢**：

| | Mooncake+NVLink+toyproxy | NixlConnector+dynamo |
|---|---|---|
| conc16 output tok/s | 502 | 574 |
| conc16 TTFT | 4746 ms | 1751 ms |
| conc64 output tok/s | 1333 | 1693 |
| conc64 TTFT | 9132 ms | 3219 ms |
| conc64 TPOT | 38.9 ms | 32.9 ms |

**根因不是 KV 传输**（TPOT decode 速度接近），而是 **`vllm_v1_proxy_server.py` 是 toy proxy**（注释写明从 NIXL 测试拷来）：串行 HTTP——先整个 prompt 发 prefill 等完、再发 decode，每请求多两跳 HTTP + mooncake per-request handshake → TTFT 撑高 2-3×。dynamo 的路由优化过但跟 mooncake 不兼容。

**⭐ 核心 gap**：高效编排器（dynamo）跟 mooncake 不兼容；兼容 mooncake 的 proxy（toy）不高效。当前 mooncake+toyproxy 净性能不如 Nixl+dynamo。要发挥 mooncake NVLink 需**生产级 mooncake-兼容编排器**（如 SGLang 自己的 PD router），是更大工程。

**mooncake 启动配置（复现用）**：
- prefill/decode 纯 `vllm serve` + `--kv-transfer-config '{"kv_connector":"MooncakeConnector","kv_role":"kv_producer|kv_consumer","kv_connector_extra_config":{"mooncake_protocol":"rdma"}}'` + `--host 0.0.0.0 --port 8000`
- env: `MC_FORCE_MNNVL=1`（GB300 走域内 NVLink，必需）+ `NCCL_MNNVL_ENABLE=1 NCCL_CUMEM_ENABLE=1` + `LD_LIBRARY_PATH` 含 cuda12 runtime（`pip install nvidia-cuda-runtime-cu12`）+ GIB source + `VLLM_MOONCAKE_BOOTSTRAP_PORT`
- proxy: `python vllm_v1_proxy_server.py --host 0.0.0.0 --port 8000 --prefiller-hosts <p-ip> --prefiller-ports 8000 --decoder-hosts <d-ip> --decoder-ports 8000`（先杀掉占 8000 的旧 frontend）

### P2c 为什么 dynamo 配 SGLang-mooncake 行、配 vLLM-mooncake 不行 + DMABUF 尝试（2026-07-22）

**dynamo 是分框架集成的**：
- `dynamo.sglang` 有 `compute_bootstrap_address`——把 prefill 的 mooncake **bootstrap 地址 (host/port/room)** 传给 decode，正好对上 mooncake 的 bootstrap 协调模型 → SGLang+dynamo+mooncake 能配。
- `dynamo.vllm` 完全照 **NixlConnector** 建（`do_remote_prefill`/`remote_engine_id`/`remote_block_ids` 塞 response，args.py 只有 Nixl/LMCache/FlexKV），**没有 bootstrap 地址传递通路** → 接不住 vLLM MooncakeConnector（用 bootstrap server 带外）。**是 dynamo 两套集成的差异，非 mooncake 本身问题**。

**DMABUF 路径（GKE COS 只支持 dmabuf 不支持 peer_mem）**：NIXL 的 UCX backend 理论上能用 dmabuf 走 GPUDirect。实测装 DOCA OFED（libmlx5→MOFED 1.25.58 支持 `ibv_reg_dmabuf_mr`）后重测 Nixl+dynamo：**KV 仍 200MB/s、吞吐没变**（conc16 612 / conc64 1677，vs vanilla 574/1693 噪声内）。UCX 的 GDAKI dmabuf 探测仍 `not available`，标准 rc_mlx5 也没走 dmabuf → 仍 cuda_copy。**COS 有 dmabuf 但这套 UCX+nvidia-open 驱动组合没成功启用它**。本镜像 NIXL 插件只有 UCX/GPUNETIO/GDS/LIBFABRIC，无 mooncake 插件。

**⭐ 最终三方对比（1p1d-dep4，8K/1K）**：

| 配置 | conc16 out | conc64 out | KV | 编排 |
|---|---|---|---|---|
| Nixl+dynamo (vanilla / MOFED) | 574 / 612 | 1693 / 1677 | 200MB/s cuda_copy | 高效 ✓ |
| Mooncake NVLink + toy proxy | 502 | 1333 | NVLink（快）但 proxy 拖慢 | toy ✗ |

**当时结论（已被 P2d 推翻）**：以为拿不到"高效编排 + 快 KV"组合。**真相**：错在一直想用 RDMA/dmabuf GPUDirect（COS 不支持 peermem，dmabuf 又没在这套 UCX+驱动上启用）。**正解是让 NIXL 走 NVLink 的 cuda_ipc，根本不碰 RDMA** —— 见 P2d。

### P2d ⭐ 最终解法：NixlConnector + NVLink cuda_ipc + vllm-router（2026-07-22 跑通）

**这才是对的做法** —— KV 不走 RDMA，走**域内 NVLink 的 UCX cuda_ipc**。三件套：

1. **UCX transport 只留 NVLink 路径**：`UCX_TLS=cuda_copy,cuda_ipc,tcp`（**删掉 rdma/rc**）+ `UCX_CUDA_IPC_ENABLE_MNNVL=y`（让 cuda_ipc 跨节点走多机 NVLink）+ `NCCL_MNNVL_ENABLE=1 NCCL_CUMEM_ENABLE=1`。
2. **`--enable-cumem-allocator`**：vLLM 用 VMM(cuMem) 分配 KV cache，块才能通过 IPC handle 共享（不加则退回 cuda_copy）。
3. **prefill/decode 同 subblock**（podAffinity `gce-topology-subblock`）→ 同 NVLink 域，cuda_ipc 直接 GPU↔GPU 搬 KV。

**编排器 = `vllm-router`**（弃 dynamo/toy proxy）：`pip install vllm-router` → `vllm-router --policy round_robin --vllm-pd-disaggregation --prefill http://<p>:8001 --decode http://<d>:8002 --port 30000`。原生支持 NixlConnector 的 producer/consumer PD 握手。

**connector**：prefill `kv_role=kv_producer` / decode `kv_role=kv_consumer`（`kv_load_failure_policy=fail`），纯 `vllm serve`（不是 dynamo.vllm）。

**⭐ 实测 KV transfer：200 MB/s → 7,000–167,000 MB/s（7–167 GB/s），提速 100–800×**，xfer 时间 200–800ms → 2–10ms。NVLink 路径坐实。

**端到端（1p1d-dep4→TP4，8K/1K）**：conc16 out 487 / TTFT 5749ms；conc64 out 1534 / **TTFT 1171ms（全配置最优）** / TPOT 30ms。KV 已不是瓶颈，此规模瓶颈转到 prefill 计算（dep4/TP4 慢官方 3.7×）。

**关键教训**：GB300 NVL72 上 vLLM disagg 的 KV 应走 **NVLink cuda_ipc**，不是 RDMA/dmabuf/peermem。之前 DOCA OFED / mooncake / dmabuf / peermem 的探索全属方向错误——NIXL 本身就能走 NVLink，只差 `UCX_CUDA_IPC_ENABLE_MNNVL=y` + `--enable-cumem-allocator`。方法来源：厂商验证过的 GB300 vLLM recipe。

**下一步吞吐杠杆**：DSpark 投机解码（`--speculative-config method=dspark num_speculative_tokens=7`，需 DeepSeek-V4-Pro-DSpark 模型）+ 扩 P/D 拓扑。

## 9. DSpark 端到端复现（官方镜像 + NVLink KV + vllm-router，1p1d）

在 P2d（NixlConnector + NVLink cuda_ipc + vllm-router）基础上叠加 **DSpark 投机解码**。全程官方组件、可复现。

### 9.0 DSpark 是什么
DeepSeek 2026-06 开源的投机解码框架（arxiv 2607.05147）：**semi-autoregressive draft head（Markov 头）+ 按负载调度的验证**，一次草稿多个 token、target 一次前向验证。相比单层 MTP 提速 51–400%。**不是新模型** —— `DeepSeek-V4-Pro-DSpark` = V4-Pro 同 checkpoint（FP8）+ baked-in DSpark draft 模块（config 里 `dspark_block_size`/`dspark_markov_rank`/`dspark_target_layer_ids`）。

### 9.1 前置
- **镜像**：官方 **`vllm/vllm-openai:v0.25.1-aarch64`**（2026-07-13 稳定版，带 dspark + `deep_gemm_mega_moe` + mxfp4）。**不要用私有自建镜像**，保复现性。**⚠️ 不要用 `nightly-aarch64`** —— 它 tag 日期虽新但实际报 `0.23.1rc1.dev1373`（版本反而旧），有 **kv_block_zeroer 断言 bug**：NixlConnector disagg 调度 `new_block_ids_to_zero` 但 `_init_kv_zero_meta()` 只在 `needs_kv_cache_zeroing=True` 时调、导致 `assert self.kv_block_zeroer is not None` 崩（`model_runner.py:883`），生成即 500。v0.25.1 已修（对齐厂商验证过的版本）。
- **模型**：`deepseek-ai/DeepSeek-V4-Pro-DSpark`（HF，FP8，~893GB / 66 shards）。`hf download` 到一节点 → `gcloud storage cp` 上 GCS → 各节点从 GCS 拉到 local SSD。
- **拓扑**：1 prefill + 1 decode，各 1 节点 4 GPU，**同 subblock**（podAffinity `gce-topology-subblock` → 同 NVLink 域）。

### 9.2 KV over NVLink（同 P2d，关键三件套）
```
UCX_TLS=cuda_copy,cuda_ipc,tcp          # 删 rdma/rc
UCX_CUDA_IPC_ENABLE_MNNVL=y             # cuda_ipc 跨节点走多机 NVLink
NCCL_MNNVL_ENABLE=1 NCCL_CUMEM_ENABLE=1 VLLM_USE_NCCL_SYMM_MEM=0
```
+ vllm serve 加 `--enable-cumem-allocator`（VMM 分配 KV 才能 IPC 共享）。

### 9.3 Prefill（kv_producer，TP4，enforce-eager）
```
vllm serve /models/DeepSeek-V4-Pro-DSpark \
  --served-model-name deepseek-ai/DeepSeek-V4-Pro-DSpark \
  --trust-remote-code --enable-cumem-allocator --kv-cache-dtype fp8 --block-size 256 \
  --port 8001 --tensor-parallel-size 4 --enforce-eager \
  --max-num-seqs 16 --max-num-batched-tokens 16384 --no-enable-prefix-caching \
  --no-disable-hybrid-kv-cache-manager \
  --moe-backend deep_gemm_mega_moe --enable-expert-parallel --tokenizer-mode deepseek_v4 \
  --speculative-config '{"method":"dspark","num_speculative_tokens":7,"draft_sample_method":"greedy"}' \
  --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_producer","kv_load_failure_policy":"fail"}'
```
env 另加：`VLLM_NIXL_SIDE_CHANNEL_PORT=5557` + `VLLM_NIXL_SIDE_CHANNEL_HOST=<prefill-ip>`。

### 9.4 Decode（kv_consumer，TP4，FULL_DECODE_ONLY）
同上，改：`--kv-role kv_consumer`、`--port 8002`、去掉 `--enforce-eager`、`--max-num-seqs 1024`、`--max-cudagraph-capture-size 1024`、`--compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[8,16,...,1024]}'`、`--gpu-memory-utilization 0.9`；`VLLM_NIXL_SIDE_CHANNEL_PORT=5558`。**decode 先等 prefill 的 5557 side channel 就绪再起**。

### 9.5 Router（vllm-router）
```
pip install vllm-router
vllm-router --policy round_robin --vllm-pd-disaggregation \
  --prefill http://<prefill-ip>:8001 --decode http://<decode-ip>:8002 \
  --host 0.0.0.0 --port 30000 --intra-node-data-parallel-size 1
```

### 9.6 验证 + benchmark
- 冒烟：`curl :30000/v1/completions` prompt "The capital of France is" → "Paris"。
- 压测：`vllm bench serve --backend openai --endpoint /v1/completions --base-url http://<router>:30000 ...`。**⚠️ tokenizer 用 HF repo id（`--tokenizer deepseek-ai/DeepSeek-V4-Pro-DSpark`）不要用本地路径** —— v0.25.1 的 transformers 把本地路径当 repo id 报错。
- **⚠️⚠️ 不能用 `--dataset-name random --ignore-eos` 测 DSpark**：随机 token 下 DSpark Markov 草稿头接受率 ~0，投机纯开销、反而更慢（实测 conc16 357/conc64 893 output tok/s，TPOT 36/57ms，**低于无 DSpark 的 P2d** 487/1693、20/33ms）。DSpark 的 51–400% 提速只在**真实连贯数据**上体现（draft 被接受）。测 DSpark 必须用 **sa-bench / sharegpt + chat template**（同厂商）。

### 9.8 实跑结论（2026-07-22）
- **✅ DSpark 端到端跑通**：官方 `v0.25.1-aarch64` + DeepSeek-V4-Pro-DSpark（FP8）+ NixlConnector NVLink KV + vllm-router，`--speculative-config method=dspark num_speculative_tokens=7`，prefill/decode 各加载 DSpark draft（96 params），生成正确（"Paris. The capital of Germany is Berlin..."）。
- **踩坑**：(1) `nightly-aarch64` 有 kv_block_zeroer bug（见 §9.1），换 `v0.25.1-aarch64` 解决；(2) bench tokenizer 用 HF repo id；(3) 随机数据测不出 spec 收益。
### 9.9 规模+压力扫描（真实数据 sharegpt，2026-07-22 起）

**测法**：`vllm bench serve --backend openai-chat --endpoint /v1/chat/completions --dataset-name sharegpt --dataset-path ShareGPT_V3.json`（真实对话，自然生成，DSpark draft 才有接受率）。tokenizer 用 HF repo id。

**1p1d-TP4（2 节点，基线）**：

| 并发 | Output tok/s | TTFT | TPOT |
|---|---|---|---|
| 32 | 228 | 17,989 ms | 15.2 ms |
| 128 | **1,026**（峰值） | 7,725 ms | 30.6 ms |
| 256 | 1,002（饱和） | 14,967 ms | 84.7 ms |

- **⭐ DSpark 真实数据生效**：`Mean acceptance length 3.0–3.2`（每 decode step 吐 ~3 token，≈3× decode yield），per-position 接受率 0.69/0.46/0.32/…，avg draft 接受 ~29%。**随机数据接受率~0 测不出，真实数据才现**。
- **⭐ 1p1d 瓶颈 = prefill 单节点**：TTFT 高达 8–18s，output 卡在 ~1026 tok/s。→ **提吞吐的杠杆是加 prefill 节点喂饱 decode**（同厂商 4p1d/5p1d/6p1d 思路）。

**4p1d-TP4（5 节点：4 prefill + 1 decode，同 subblock NVLink 域）**：

| 并发 | Output tok/s | TTFT | TPOT |
|---|---|---|---|
| 128 | 970 | 18,760 ms | 19.7 ms |
| 256 | 1,987 | 4,058 ms | 86.2 ms |
| 512 | **3,004**（峰值） | 8,414 ms | 76.8 ms |

- **⭐ 4 prefill 喂 1 decode → 吞吐 ~3× 于 1p1d**（3,004 vs 1,026）。瓶颈从 prefill 转到 **decode**（TPOT 77–86ms，decode 单节点 dep4/TP4 饱和）。DSpark acceptance length 仍 ~3.0。
- **scaling 规律**：1p1d prefill-bound(1026) → 4p1d decode-bound(3004)。**下一步扩 decode（第 2 个 decode 节点）**解 decode 瓶颈。

**4p2d-TP4（6 节点：4 prefill + 2 decode，同 subblock NVLink 域）**：

| 并发 | Output tok/s | TTFT | TPOT |
|---|---|---|---|
| 256 | 1,114 | 12,490 ms | 228.0 ms |
| 512 | 4,297 | 6,062 ms | 62.3 ms |
| 1024 | **5,826**（峰值） | 12,372 ms | 77.8 ms |

- **⭐ decode 从 1→2 节点 → 峰值 3,004 → 5,826（~1.9×）**，几乎线性——**证实 4p1d 确为 decode-bound**。TPOT 恢复到 62–78ms（单 decode 时同 concurrency 已饱和）。
- C=256 首轮偏低（1,114，router 刚重启的 warmup 伪影，TTFT/P99 异常高）；有效峰值看 C=1024。
- **瓶颈再平衡**：C=1024 时 TTFT 回升到 ~12s，说明 4 prefill 又开始吃紧 → 下一步加 prefill（6p2d）验证是不是 prefill 限制。

**6p2d-TP4（8 节点：6 prefill + 2 decode，同 subblock NVLink 域）**：

| 并发 | Output tok/s | TTFT | TPOT |
|---|---|---|---|
| 512 | 1,612 | 23,664 ms | 34.7 ms |
| 1024 | 5,612 | 8,542 ms | 82.9 ms |
| 2048 | **6,190**（峰值） | 42,279 ms | 83.6 ms |

- **⭐⭐ 加 prefill（4p→6p）吞吐几乎没涨**（5,826 → 6,190，+6%），只把 TTFT 从 12s 降到 8.5s（C=1024）。C=2048 硬堆并发才多挤出一点，代价是 TTFT 飙到 42s（过饱和）。
- **决定性结论**：**吞吐天花板 = decode 节点数 × ~3,000 tok/s**（1 decode ≈ 3,004；2 decode ≈ 5,800–6,200）。prefill 只决定 TTFT（喂料速度），不决定稳态吞吐。→ **要提吞吐必须加 decode，不是加 prefill**。
- C=512 首档 1,612 又是 router 重启的 warmup 伪影（P99 TTFT 140s 但 median 仅 1.4s）。

**6p3d-TP4（9 节点：6 prefill + 3 decode，同 subblock NVLink 域）**：

| 并发 | Output tok/s | TTFT | TPOT | 失败 |
|---|---|---|---|---|
| 1024 | 8,046 | 5,040 ms | 76.6 ms | 0 |
| 2048 | **9,153**（峰值） | 19,666 ms | 78.1 ms | 0 |

- **⭐⭐⭐ decode 加到 3 节点 → 峰值 9,153**，0 失败，TPOT 稳定 76–78ms。**decode 线性 scaling 坐实**：1→3,004 / 2→5,826 / 3→9,153，≈ **3,050 tok/s per decode 节点**。
- 加 prefill（4p→6p）配 3 decode 后 TTFT 也压得住（C=1024 仅 5s）。

**scaling law 总结**：

| 拓扑 | decode 节点 | 峰值 Output tok/s | 每 decode | 瓶颈 |
|---|---|---|---|---|
| 1p1d | 1 | 1,026 | — | prefill |
| 4p1d | 1 | 3,004 | 3,004 | decode |
| 4p2d | 2 | 5,826 | 2,913 | decode |
| 6p2d | 2 | 6,190 | 3,095 | decode（prefill 过量） |
| 6p3d | 3 | **9,153** | 3,051 | decode |

**⭐ 核心结论**：DSpark disagg 在 GB300 NVL72 上 **吞吐随 decode 节点数线性增长（~3,050 tok/s/decode-TP4）**。prefill 只决定 TTFT（喂料速度），不决定稳态吞吐——2 decode 配 4 或 6 prefill 峰值几乎一样。**扩容优先加 decode**，prefill 按 TTFT SLA 配（经验 ~2 prefill : 1 decode 已够）。

### 9.9.1 踩坑：decode 容器 OOM（exit 137）

跑 6p4d（4 decode）时 **decode 容器被 exit 137 杀掉**（`ContainerStatusUnknown`，K8s 显示 `terminated exitCode 137`），router 报 `tcp connect error deadline elapsed`，那一档吞吐反跌到 3,597 且 102 请求失败。根因：**decode pod `memory: 600Gi` limit 在高并发（KV cache + activation + NIXL buffer）下被击穿触发 OOM-kill**。换节点重建 decode 后 6p3d 干净跑通。**教训**：GB300 decode pod 内存 limit 要留足头寸（或调低 `--max-num-seqs`/KV cache 占比），高压下别贴着 limit 跑；benchmark 前先确认所有 decode `kubectl get pod` 是 `Running` 且 `curl /health` 通。

### 9.7 GCS 传输 auth 坑
GKE 节点 compute SA 对模型 bucket **OAuth scope 未授权**。上传/下载用：本机 `gcloud auth application-default print-access-token` → cp token 进 pod → `CLOUDSDK_AUTH_ACCESS_TOKEN=<token> gcloud storage cp ... --billing-project=<project>`（`gcloud auth login --cred-file` 不吃 authorized_user ADC；只有 `CLOUDSDK_AUTH_ACCESS_TOKEN` 能让 gcloud CLI 用上）。用完删 token。

---

*SGLang 实跑对标见 [`./sglang-v4-gb300-benchmark.md`](./sglang-v4-gb300-benchmark.md)。榜单值（§4）仍为官方/InferenceX 公开数据。*
