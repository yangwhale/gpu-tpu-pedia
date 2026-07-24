# GB300 (A4X Max) · vLLM DeepSeek-V4 · Benchmark 对标报告

> **本文性质**：vLLM 在 GB300 上服务 DeepSeek-V4 的**官方 recipe + 已公开 benchmark 的对标整理**，与同目录 [`sglang-v4-gb300-benchmark.md`](./sglang-v4-gb300-benchmark.md)（含**我们实测跑通**的 8,903 output/decode-GPU）配套阅读。
> ✅ **已在本环境端到端跑通 + 复现**（2026-07-23）：照抄厂商官方 vLLM recipe（deepgemm 镜像 1p1d），4k1k **Total token throughput 24,358 tps = 厂商 22,000 基线的 111%**，复现成功。**完整照抄步骤见 §9.10**（权威复现 checklist）。前半部分（§0–§8）的 blog/榜单数据仍为参考；§9 是本环境实跑记录。

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

**架构**：`prefill`(dep4, node A) + `decode`(dep4, node B) + `frontend`(router)。KV 从 prefill 跨节点传到 decode，走 CX-8。

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
> ⚠️ **镜像重要更正（2026-07-23）**：本节及 §9.2–§9.9 用的通用 `vllm/vllm-openai:v0.25.1-aarch64` 只**够功能验证（生成正确）**，但**性能腰斩**（`deep_gemm_mega_moe` 静默 fallback 慢 kernel）。**要复现厂商 22,000 tps 性能，必须换 deepgemm 专用镜像 `vllm-openai-deepgemm:v0.25.1-sm100-aarch64`**（它是官方 vLLM CI 构建、非私有 fork，见 §9.10 Step 0）。下面这句"用通用镜像/别用私有镜像"仅对**功能冒烟**成立，性能复现看 §9.10。

- **镜像（功能验证用）**：官方 **`vllm/vllm-openai:v0.25.1-aarch64`**（2026-07-13 稳定版，带 dspark + `deep_gemm_mega_moe` + mxfp4）。**⚠️ 不要用 `nightly-aarch64`** —— 它 tag 日期虽新但实际报 `0.23.1rc1.dev1373`（版本反而旧），有 **kv_block_zeroer 断言 bug**：NixlConnector disagg 调度 `new_block_ids_to_zero` 但 `_init_kv_zero_meta()` 只在 `needs_kv_cache_zeroing=True` 时调、导致 `assert self.kv_block_zeroer is not None` 崩（`model_runner.py:883`），生成即 500。v0.25.1 已修（对齐厂商验证过的版本）。
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

> ⚠️ **本节（9.9–9.9.5）所有绝对吞吐数字跑在通用镜像上，已作废**（`deep_gemm_mega_moe` fallback 慢 kernel）。scaling **规律**（吞吐随 decode 节点线性、prefill 只决定 TTFT、DSpark 需真实数据）仍成立且有价值；但**绝对 tok/s 一律以 deepgemm 镜像的 §9.9.6 + §9.10 为准**。

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
| 6p4d | 4 | (预测 ~12,000) | — | decode 不稳定，见 §9.9.1 |

**⭐ 核心结论**：DSpark disagg 在 GB300 NVL72 上 **吞吐随 decode 节点数线性增长（~3,050 tok/s/decode-TP4）**，1→3 decode 完美线性（3,004 / 5,826 / 9,153）。按此外推 4 decode ≈ 12,000。prefill 只决定 TTFT（喂料速度），不决定稳态吞吐——2 decode 配 4 或 6 prefill 峰值几乎一样。**扩容优先加 decode**，prefill 按 TTFT SLA 配（经验 ~2 prefill : 1 decode 已够）。**当前稳定实测峰值 = 6p3d 9,153 tok/s**（4 decode 因 decode 稳定性问题未跑通，见下）。

### 9.9.1 踩坑：4-decode 拓扑下 decode 不稳定（两次失败）

6p4d（4 decode）**连续两次跑失败**，两种崩法：

1. **decode 容器 OOM（exit 137）**：第一次跑，一个 decode 容器被 `exit 137` 杀掉（`ContainerStatusUnknown`），router 报 `tcp connect error deadline elapsed`，吞吐反跌到 3,597 + 102 失败。→ `memory: 600Gi` limit 在高压下被击穿触发 OOM-kill。
2. **decode 引擎 crash（500 → engine 退出）**：换新节点重建 decode 后再跑，一个**在 6p3d 里跑得完美（0 失败）的 decode**，连跑第二轮 sweep 时引擎崩溃——先返回 `500 Internal Server Error`，随后 `MPClient: stopping engine manager` / `engine manager stopped`，API port 挂掉，router 报 `Connection refused (os error 111)`，吞吐 2,810 + 363 失败。此时 K8s pod 仍显示 `ready=true`（容器主进程刚退、restartPolicy 还没触发，健康状态滞后）。

**共同规律**：DSpark decode pod 在**持续多轮重压测**后内存累积/不稳定，最终 OOM 或引擎 crash。fresh decode 单轮跑没问题（6p3d 干净），跑过多轮的 decode 容易崩。

**教训 / 缓解**：
- 内存留头寸：调低 `--max-num-seqs`（1024→512）或 KV cache 占比，别贴着 `600Gi` limit 跑；或直接提高 pod memory limit。
- benchmark 前逐一确认所有 decode `kubectl get pod` 为 `Running` **且** `curl /health` 通（pod ready 状态会滞后于引擎崩溃，别只信 `kubectl get pod`）。
- 长时间多轮压测的 decode 建议每轮前重建（fresh pod）以排除累积效应。
- **结论**：4 decode 的吞吐（外推 ~12,000）需先解决 decode 稳定性才能实测；当前**稳定可复现峰值为 6p3d = 9,153 tok/s**。

### 9.9.2 decode 改 DP-attention + dep8 宽 EP（✅✅ 实测 2026-07-23，吞吐炸裂）

> ⚠️ **本节数字口径已被 §9.9.3 修正**：下面 "1,983 tok/s/GPU、2.6× TP4" 是 **ShareGPT 闭环**（prefill-limited）测的，**不能对标官方 11,200**（官方是 sa-bench 开环 + 8192/1024 workload）。dep8 相对 TP4-dep4 的每卡效率优势（架构层面）成立，但绝对吞吐的正确 decode-saturation 复测见 **§9.9.3**（官方口径峰值 1,458/GPU，瓶颈在 prefill 不在 decode）。

**⭐⭐⭐ dep8 实测结果（5 TP4-prefill + 1 dep8-decode，ShareGPT）**：

| 并发 | Output tok/s | Total tok/s | TTFT | TPOT |
|---|---|---|---|---|
| 1024 | 6,399 | 13,964 | 6,392 ms | 139 ms |
| 2048 | **15,861** | 34,255 | 4,240 ms | 94 ms |

- **单个 dep8 decode（8 GPU）C=2048 = 15,861 output tok/s**，远超之前外推的 12,000，也远超 6p3d 基线（3×TP4-dep4 decode = 12 GPU = 9,153）。
- **per-decode-GPU 对比**：dep8 = **1,983 tok/s/GPU** vs TP4-dep4 = 763 tok/s/GPU → **~2.6× 每卡效率**！
- **根因**：dep8（TP1 + DP8-attention + EP8）是 MLA-MoE decode 的正确配置——① DP-attention 每 rank 各存各请求 KV、**不复制 MLA latent**（TP4 会复制）；② EP8 把 384 expert 摊到每卡 48 个、省 HBM → 更大 batch；③ attention/dense 权重只存一份。wide-EP 在高并发（C=2048）尤其发力（batch 大、EP 摊得开）。
- **教训固化**：**别再用 TP4-dep4 扩副本那套**——那是被 TP4 的 MLA-KV 复制 + EP4 厚 expert 拖累的次优解。DeepSeek MLA-MoE 的 decode 就该 DP-attention + 宽 EP（dep8/dep16），跟 SGLang 一致。之前 §9.9 的 1p1d→6p3d scaling law（~3050/decode）是在次优 TP4 配置下测的，dep8 直接把天花板拉高 2.6×。

**✅ 端到端验证过程（2026-07-23）**：不动 prefill（保持 TP4），只新建 dep8 decode，测 **TP4-prefill → DP8-decode 的 KV 跨并行度传输**：
- **多节点 vLLM DP8 启动成功**：head（node A，DP rank 0-3）+ headless worker（node B，DP rank 4-7），`--data-parallel-size 8 --data-parallel-size-local 4 --data-parallel-address <head> --data-parallel-rpc-port 13345`（worker 加 `--data-parallel-start-rank 4 --headless`），world_size=8 跨 2 节点连通。
- **dep8 不 OOM**：EP8 把 384 expert 摊到每卡 48 个，`--gpu-memory-utilization 0.85` 舒适装下（对比 dep4 prefill 因 DP4 复制权重 268GB/卡 OOM）。
- **⭐ TP4-prefill → DP8-decode KV 传输 work**：NixlConnector 跨并行度传输成功，生成正确。根因：**MLA 的 KV 是所有头共享的完整 latent（512+64），block 布局与 TP/DP 并行度无关**（都是 per-rank 完整 latent），所以 TP4 producer 传给 DP8 consumer 天然兼容。→ **「只改 decode、不动 prefill」路线成立**，省掉全栈改造。
- **踩坑**：清旧 TP4 进程时 `pkill -f 'vllm serve'` 漏杀 `VLLM::Worker` 子进程（进程名不含 "vllm serve"），278GB×4 显存不释放 → 新实例 OOM。必须按进程名 `kill VLLM::/EngineCore`。另 GB300 驱动/GIB 保留 ~40GB/卡，util 要留头寸。

**⚠️ prefill 并行度选项（2026-07-23 实测）**：vLLM 对 DeepSeek V4 的 prefill 权重分片**只有 TP 可用**。
- **PP 不支持**：`--pipeline-parallel-size 4` 直接 `NotImplementedError: Pipeline parallelism is not supported for this model`（模型未实现 `SupportsPP` 接口）。**框架差异**：SGLang 支持 PP4 prefill（他们就用这个），vLLM 不支持。
- **DP 会 OOM**：`--data-parallel-size 4` 复制 attention/dense 权重 → 268GB/卡 → OOM（见上）。
- **→ 结论**：vLLM prefill 就用 **TP4**（唯一可行的权重分片方式）。TP4 vs PP4 只能跨框架比（vLLM-TP4 vs SGLang-PP4）。

**⭐ SGLang prefill TP4 vs PP4 干净对比（2026-07-23，同节点/同模型 R1-NVFP4/同配置 standalone，仅换并行度）**：

| 测试 | TP4 | PP4 |
|---|---|---|
| 8K 输入 conc1（纯 prefill TTFT） | Mean **197ms** | Mean **296ms** |
| input1024/out512 conc16 | Mean TTFT 357ms / P90 594ms / 吞吐 800 tok/s | Mean TTFT 292ms / P90 420ms / 吞吐 474 tok/s |

- **8K 单条 prefill：TP4 快 ~33%**（197 vs 296ms）。conc16 的 TTFT 两者**同一量级**（PP4 mean 还略低、尾部更稳；TP4 吞吐高但那是 decode 侧）。
- **⚠️ 重要纠正**：SGLang 历史 R2（1P1D disagg，PP4，input1024/conc16）Mean TTFT = **7360ms**，而**同参数 PP4 standalone 只有 292ms —— 差 25×**。唯一区别是 disagg vs standalone。**所以那 7360ms 不是 PP4 并行度的锅**，是 1P1D disagg 里单个 prefill-only 节点在 conc16 下被打爆排队 + KV 传输 + decode 侧堆积。
- **结论**：PP4 本身**不拉跨**（standalone 292ms 正常），TP4 只在单条长 prefill 略优。SGLang 生产 14P:2D 那种拉跨，病根在 **disaggregation 的 prefill 节点吞吐/配置**（chunked-prefill/节点数/单节点并发能力），**不在 TP-vs-PP 的选择**。换 TP4 能小赚单请求延迟，但救不了 14:2。别被 standalone-vs-disagg 的表面数字骗。

**本轮 decode 用的是 TP4 + EP4（dep4）**（照 DSpark 1p1d-dep4 基线扩副本），但这对 MLA-MoE 是次优：

- **MLA 在 TP 下 KV 不分片、只复制**：MLA 的 KV 是所有头共享的压缩 latent（512+64），每个 TP rank 都要存完整 latent → TP4 把 KV cache 复制 4 份，零节省。TP 只帮到权重分片 + EP + 按头分 attention 计算。
- **官方 vLLM V4-Pro recipe 用的是 dep8**（`6p1d-dep4-dep8`）：decode = **TP1 + DP8-attention + EP8**（`--tensor-parallel-size 1 --data-parallel-size 8 --enable-expert-parallel` + dp-attention）。DP-attention 每 rank 各存各请求的 KV → 天然不复制；EP8 把 384 expert 摊到每卡 48 个 → 省 HBM、更大 batch；attention/dense 权重只存一份。这也是 SGLang dep8 的思路。
- **下一轮**：decode 切成 DP-attention + dep8（跨 2 节点 8 卡宽 EP），对齐官方 recipe + SGLang，做 apples-to-apples；预期逼近外推 ~12,000，且宽 EP 每卡 expert 少、内存压力小，可能顺带绕开 §9.9.1 的 4-decode 内存崩坑。跑前需确认 vLLM EP8 + NixlConnector disagg + NVLink KV 在跨节点 8 卡宽度成立。

### 9.9.3 ⭐⭐⭐ decode-saturation 正确口径复测：sa-bench 开环 + 官方 workload（2026-07-23）

> ⚠️ **本节及 §9.9.4/9.9.5 的绝对吞吐数字已作废**：均跑在**通用镜像**上（`deep_gemm_mega_moe` 静默 fallback 慢 kernel），非架构/prefill 上限。正确基线见 **§9.9.6**（deepgemm 镜像复现厂商 22,000）。本节方法学（开环 sa-bench + 官方 workload + output÷decode-GPU 口径）仍成立，仅绝对值受镜像拖累。

**为什么要重测**：§9.9.2 的 "dep8 = 1,983 tok/s/GPU、2.6× TP4" 是 **ShareGPT 闭环**测的，对标官方 11,200 / SGLang 8,993 时**差一个数量级**。根因不是 decode 弱，是**口径错**：ShareGPT 输入短（~200 tok）、输出短，请求大部分时间耗在排队/TTFT，**decode 队列压根没喂满**——测的是整条 PD 流水线，不是 decode 天花板。官方 11,200 是用 **sa-bench 开环 + random 8192/1024 固定长度** workload 专门把 decode 压满测的。本轮换成同一把尺子。

**修正后两轮扫描（8×TP4-prefill + 1×dep8-decode，同一栈）**：

**A. ShareGPT 闭环补测（8 prefill，`vllm bench serve`，口径 output÷8-decode-GPU）**

| 并发 | Output tok/s | per-GPU (÷8) | TTFT | TPOT |
|---|---|---|---|---|
| 1024 | 11,642 | 1,455 | 3,485 ms | 51.5 ms |
| 2048 | 13,213 | 1,652 | 5,040 ms | 104.7 ms |
| 3072 | 15,918 | **1,990** | 9,692 ms | — |

→ 8 prefill 相比旧 5 prefill 峰值几乎没变（1,990 vs 1,983），且 TTFT 一路爬到 9.7s、并发翻倍吞吐只涨 13%——**证实 ShareGPT 闭环卡在 prefill/排队，不是 decode**。换汤不换药。

**B. sa-bench 开环 + 官方 workload（random 8192/1024 range-ratio 1.0，`sglang.bench_serving --backend sglang-oai` 打 vllm-router，官方口径 output÷8-decode-GPU）**

| 并发 | Output tok/s | per-GPU (÷8) | TTFT | TPOT |
|---|---|---|---|---|
| 64 | 5,937 | 742 | 797 ms | **6.48 ms** |
| 256 | 10,010 | 1,251 | 13,106 ms | 7.37 ms |
| 512 | 11,144 | 1,393 | 34,038 ms | 7.31 ms |
| 1024 | 11,665 | **1,458** | 75,946 ms | ~7 ms |

**⭐ 结论：瓶颈在 prefill，不在 decode。**

- **decode 全程有富余**：TPOT 从 conc64 到 conc1024 **稳在 ~7ms 纹丝不动**（DSpark 投机解码健康，draft 在 random prompt 上照样被接受——draft 猜的是 target 自己的输出分布，与 prompt 自不自然无关）。7ms TPOT = 每 user ~143 tok/s，**远快于官方 11,200 的运行点 ~50 tok/s/user（20ms TPOT）**，说明 decode 还能扛更多并发。
- **prefill 是硬墙**：TTFT 从 797ms → 13s → 34s → 76s **指数爆炸**，output 在 conc512 就撞停（512→1024 只涨 5%）。8×TP4-prefill 消化不了 8192 长输入的 prefill 洪流。
- **对标**：官方 workload 下峰值 **1,458 tok/s/GPU** vs SGLang 8,993（16 prefill）vs 官方 11,200。gap **几乎全在 prefill 容量**——SGLang 打 8,993 用了 **16 个 prefill**，我们只有 8 个。这跟 §7.2-P4 "prefill 扩到 8/12/16 看是否像 SGLang 需要更多 prefill 喂饱 dep8" 的预判完全吻合。
- **下一步**：要真正逼近 decode 天花板 / 对标 8,993，需把 prefill 从 8 扩到 ~16（decode 侧 dep8 不用动，它有大量余量）。这是**拓扑/资源问题，不是 decode 架构或内核问题**。dep8 decode 本身（TPOT 7ms）是有竞争力的。

> **口径对齐说明**：A 表 ShareGPT（短输入）峰值 1,990 反而比 B 表 sa-bench（8192 长输入）峰值 1,458 高——因为 ShareGPT prefill 便宜、请求流得快。但**只有 B 表的 8192/1024 才是官方 11,200 的同 workload**，A 表不可与官方直接比。对标一律用 B 表。

### 9.9.6 ⭐⭐⭐⭐⭐ 官方 1p1d 复现成功 = 厂商 4k1k 22,000 tps 基线（2026-07-23）

**结论先行**：照抄厂商官方 vLLM recipe（`fp8_1p1d_nixl_dspark_s7`）跑 1p1d（1×TP4 prefill + 1×TP4 decode = 8 GPU），4k1k（ISL 4096 / OSL 1024）**Total token throughput 复现并反超厂商 22,000 tps 基线**：

| 并发 | Total tok/s (in+out) | vs 22,000 | Median TPOT | Median TTFT |
|---|---|---|---|---|
| 256 | **23,120** | **105%** | 12.6 ms | 37 s |
| 512 | **24,358** | **111%** | 13.2 ms | 87 s |

（TTFT 随并发爬升是单 prefill 在 4k 输入下的排队，不影响总吞吐；峰值出在 conc512。）

**⭐ 最关键发现：镜像是之前所有 dep8 数字拉胯的根因，不是架构。**

- 官方 recipe 用的是**专用镜像 `vllm-openai-deepgemm:v0.25.1-sm100-aarch64`**，启动日志可见 `expert_dtype resolved to 'fp4'` + `Selected DeepGemmFp8BlockScaledMMKernel` + `DeepGEMM PDL/E8M0 enabled`——**FP4 专家 + DeepGEMM 优化 kernel 全激活**。
- 之前 §9.9.1–§9.9.5 全部跑在**通用镜像 `vllm-openai:v0.25.1-aarch64`** 上，`--moe-backend deep_gemm_mega_moe` 在通用镜像上**静默 fallback 到慢 kernel**。**因此 §9.9.2 的 "1,983"、§9.9.3 的 "1,458"、以及所有 dep8/多-frontend 绝对吞吐数字对标官方无效**（是镜像被阉割，不是 dep8 架构或 prefill 数量的问题）。绝对性能一律以本节（deepgemm 镜像）为准。

**官方 recipe 关键要素（照抄，缺一不可）**：
1. **镜像**：`vllm-openai-deepgemm`（非通用 `vllm-openai`）——最大变量
2. `--attention_config.use_fp4_indexer_cache=True`（FP4 indexer 缓存，prefill+decode 都加）
3. DSV4 parser：`--tool-call-parser deepseek_v4 --enable-auto-tool-choice --reasoning-parser deepseek_v4`
4. decode = **TP4**（`--max-num-seqs 1024 --max-num-batched-tokens 8192` + `FULL_DECODE_ONLY` cudagraph，capture sizes `[8,16,24,...,1024]`，`--gpu-memory-utilization 0.9`）
5. DSpark s7（`{"method":"dspark","num_speculative_tokens":7,"draft_sample_method":"greedy"}`）+ NixlConnector NVLink KV（`UCX_TLS=cuda_copy,cuda_ipc,tcp` + `UCX_CUDA_IPC_ENABLE_MNNVL=y` + `--enable-cumem-allocator`）
6. prefill = TP4 enforce-eager，`--max-num-seqs 16 --max-num-batched-tokens 16384`，kv_producer

> **冷启动注意**：deepgemm 镜像会做 **DeepGEMM kernel warmup**（prefill ~2484 kernel / decode ~1666）+ TileLang JIT + DSpark cudagraph capture，比通用镜像慢（~8-12 min），但这正是优化 kernel 在编译。

**⚠️ 运维大坑：`vllm-router` 进程名是 `vllm::router`（带冒号）**。所以 `pkill -f vllm-router` **永远匹配不到**，杀不掉旧 router → 僵尸 router 累积占住 Prometheus 端口 → 新 router 起不来报 `FailedToCreateHTTPListener("Address already in use")`。**正确清理：`pkill -9 -f 'vllm::router'`**。这一个坑消耗了大量排查时间（所有"router 起不来/503/端口冲突"都源于此）。

### 9.10 ⭐⭐⭐⭐⭐ 完整复现 checklist（照抄跑出 4k1k 22-24K tps · 2026-07-23 验证）

> **这是本文档的权威复现路径**。前面 §9.9.1–§9.9.5 是通用镜像上的探索过程（绝对吞吐已作废，只留方法学）；要**复现厂商 22,000 tps 基线**，照本节即可。全程 8 GPU（1 prefill + 1 decode），单 NVL72 subblock。

#### Step 0 · 镜像（最关键，别用错）

- **必须用 deepgemm 专用镜像**：`vllm-openai-deepgemm:v0.25.1-sm100-aarch64`（sm100=Blackwell，aarch64=GB300 Grace）。
- **镜像来源（已核实）**：**官方 vLLM 构建**，非个人自制——走 vLLM 官方 buildkite `release-v2` CI（build 3803）编译，基于 vllm-project/vllm commit `752a3a5`（2026-07-12）；DeepSeek-V4 + DSpark 模型代码在 `vllm/models/deepseek_v4/nvidia/`（**NVIDIA 贡献**）；内置 `deep_gemm 2.5.0`。厂商只是把它转存到私有 registry 做部署暂存，**没有 fork 改码**。
- **为什么不能用通用 `vllm/vllm-openai:v0.25.1-aarch64`**：通用镜像**能跑通（生成正确）但性能腰斩**——`--moe-backend deep_gemm_mega_moe` 静默 fallback 到慢 kernel，测不到 FP4/DeepGEMM 优化。deepgemm 镜像启动日志会显式打印 `expert_dtype resolved to 'fp4'` + `Selected DeepGemmFp8BlockScaledMMKernel` + `DeepGEMM PDL/E8M0 enabled`（**认准这几行 = 优化 kernel 已激活**）。

#### Step 1 · 前置
- 模型 `DeepSeek-V4-Pro-DSpark`（FP8，~832GB/66 shards）在两节点 local SSD（hostPath），容器内挂到 `/models`。
- 两节点同 subblock（`gce-topology-subblock` podAffinity → 同 NVLink 域），各 4 GPU，`imagePullSecrets` 能拉 deepgemm 镜像。

#### Step 2 · Prefill（TP4，kv_producer，enforce-eager）
env：`VLLM_USE_NCCL_SYMM_MEM=0 NCCL_CUMEM_ENABLE=1 NCCL_MNNVL_ENABLE=1 NCCL_NVLS_ENABLE=1` + `UCX_NET_DEVICES=all UCX_CUDA_IPC_ENABLE_MNNVL=y UCX_TLS=cuda_copy,cuda_ipc,tcp` + `VLLM_NIXL_SIDE_CHANNEL_PORT=5557 VLLM_NIXL_SIDE_CHANNEL_HOST=<prefill-ip>` + `PYTHONHASHSEED=0`
```bash
vllm serve /models/DeepSeek-V4-Pro-DSpark --served-model-name deepseek-ai/DeepSeek-V4-Pro-DSpark \
  --trust-remote-code --enable-cumem-allocator --kv-cache-dtype fp8 --block-size 256 \
  --port 8001 --tensor-parallel-size 4 --enforce-eager \
  --max-num-seqs 16 --max-num-batched-tokens 16384 --no-disable-hybrid-kv-cache-manager \
  --attention_config.use_fp4_indexer_cache=True \
  --moe-backend deep_gemm_mega_moe --enable-expert-parallel --tokenizer-mode deepseek_v4 \
  --tool-call-parser deepseek_v4 --enable-auto-tool-choice --reasoning-parser deepseek_v4 \
  --speculative-config '{"method":"dspark","num_speculative_tokens":7,"draft_sample_method":"greedy"}' \
  --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_producer","kv_load_failure_policy":"fail"}'
```

#### Step 3 · Decode（TP4，kv_consumer，FULL_DECODE_ONLY cudagraph）
env 同 prefill，但 `VLLM_NIXL_SIDE_CHANNEL_PORT=5558`。**先等 prefill 的 5557 side channel 就绪再起 decode**。
```bash
vllm serve /models/DeepSeek-V4-Pro-DSpark --served-model-name deepseek-ai/DeepSeek-V4-Pro-DSpark \
  --trust-remote-code --enable-cumem-allocator --kv-cache-dtype fp8 --block-size 256 \
  --port 8002 --tensor-parallel-size 4 \
  --max-num-seqs 1024 --max-num-batched-tokens 8192 --max-cudagraph-capture-size 1024 \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[8,16,24,32,40,48,56,64,96,128,192,256,384,512,768,1024]}' \
  --gpu-memory-utilization 0.9 --no-disable-hybrid-kv-cache-manager \
  --attention_config.use_fp4_indexer_cache=True \
  --moe-backend deep_gemm_mega_moe --enable-expert-parallel --tokenizer-mode deepseek_v4 \
  --tool-call-parser deepseek_v4 --enable-auto-tool-choice --reasoning-parser deepseek_v4 \
  --speculative-config '{"method":"dspark","num_speculative_tokens":7,"draft_sample_method":"greedy"}' \
  --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_consumer","kv_load_failure_policy":"fail"}'
```
> **冷启动 ~8-12 min**：deepgemm 镜像做 DeepGEMM warmup（prefill ~2484 / decode ~1666 kernel）+ TileLang JIT + DSpark cudagraph capture（`Capturing dspark CUDA graphs (FULL)`）。decode ready 判据：`curl localhost:8002/health` = 200。

#### Step 4 · Router（vllm-router，PD-disagg）
```bash
pip install vllm-router   # deepgemm 镜像已带; 没有再装
vllm-router --policy round_robin --vllm-pd-disaggregation --host 0.0.0.0 --port 30000 \
  --prefill http://<prefill-ip>:8001 --decode http://<decode-ip>:8002 --intra-node-data-parallel-size 1
```
> **⚠️ 运维大坑：router 进程名是 `vllm::router`（带冒号）**。`pkill -f vllm-router` **永远杀不掉**，僵尸 router 累积占 Prometheus 端口 → 新 router 报 `FailedToCreateHTTPListener("Address already in use")`。**清理必须用 `pkill -9 -f 'vllm::router'`**（今晚所有"router 起不来/503"都源于此）。ready 判据：`curl localhost:30000/health` = 200（若 503「Prefill policy failed to select a worker」= 后端还没注册好，等几秒或查 prefill/decode :8001/:8002 是否 200）。

#### Step 5 · Benchmark（4k1k，官方口径 Total token throughput）
用 SGLang `sa-bench`（`bench_serving`）打 router，**random 8192? 不，4k1k = ISL 4096 / OSL 1024**，`--random-range-ratio 1.0`（sglang 语义 = 固定长度；**vLLM bench 语义相反、要用 0.0**）：
```bash
python3 -m sglang.bench_serving --backend sglang-oai --host <router-ip> --port 30000 \
  --model deepseek-ai/DeepSeek-V4-Pro-DSpark --dataset-name random \
  --random-input-len 4096 --random-output-len 1024 --random-range-ratio 1.0 \
  --num-prompts <2*conc> --max-concurrency <conc>   # 扫 conc 256/512
```

#### 预期结果（复现验证 2026-07-23）
| 并发 | Total tok/s (in+out) | vs 厂商 22,000 | TPOT |
|---|---|---|---|
| 256 | 23,120 | 105% | 12.6 ms |
| 512 | **24,358** | **111%** | 13.2 ms |

达到 ≥22,000 = 复现成功（说明镜像 + config + 环境全对）。若远低于此：99% 是**用错了通用镜像**（回 Step 0 认那三行 kernel 日志）。

**2p1d 对比（同一 TP4 decode，加第 2 个 TP4 prefill，2026-07-24）**：

| 拓扑 | conc256 Total | conc512 Total | 结论 |
|---|---|---|---|
| 1p1d | 23,120 | 24,358 | 单 prefill 略喂不满单 decode |
| 2p1d | 27,803 | **31,499** | 加 prefill **+29%**（4k1k 下第 2 prefill 把单 decode 喂更足） |

→ 4k1k 输入下，**加 prefill 仍有收益**（decode 未被单 prefill 完全喂饱）；比 SGLang 8k 长输入时"加 prefill 无用/decode-bound"更靠 prefill 一侧（输入短 → prefill 相对便宜、瓶颈更偏 decode，但单 prefill 仍不足）。⚠️ 测量注意：benchmark **必须带 warmup**——2p1d 无 warmup 首跑 conc256 只有 13,585（DeepGEMM JIT 未热 + router 冷），warmup 后 27,803，差 2×，别被冷跑数字骗。

#### 复现 checklist（逐条核对）
- [ ] 镜像是 `vllm-openai-deepgemm`（不是通用 `vllm-openai`）
- [ ] 启动日志有 `expert_dtype resolved to 'fp4'` + `DeepGemmFp8BlockScaledMMKernel`
- [ ] prefill/decode 都带 `--attention_config.use_fp4_indexer_cache=True` + DSV4 parser
- [ ] KV over NVLink 三件套（`UCX_TLS=cuda_copy,cuda_ipc,tcp` + `UCX_CUDA_IPC_ENABLE_MNNVL=y` + `--enable-cumem-allocator`）
- [ ] 两节点同 subblock（同 NVLink 域）
- [ ] 清 router 用 `pkill -9 -f 'vllm::router'`（带冒号）
- [ ] benchmark 4k1k，sglang bench `--random-range-ratio 1.0`（或 vllm bench `0.0`）

### 9.11 ⭐⭐⭐ deepgemm dep8 + prefill 数扫描：找 decode 饱和点（2026-07-24）

**目的**：dep8（TP1+DP8+EP8 dp-attention decode，**8 GPU**）在正确的 deepgemm 镜像下，逐个加 prefill（TP4）喂料，看 output÷8-decode-GPU 何时被打满（decode-bound）。workload = **8k1k**（ISL 8192 / OSL 1024，官方 dep8 workload），conc1024，sa-bench 开环。

| prefill 数 | Output tok/s | **Output÷8 /GPU** | Total tok/s (in+out) | Median TPOT | 增量/prefill |
|---|---|---|---|---|---|
| 1 | 2,453 | 307 | 22,082 | 5.4 ms | — |
| 2 | 4,773 | 597 | 42,956 | 5.6 ms | +290 |
| 3 | 6,982 | 873 | 62,836 | 8.1 ms | +276 |
| 4 | 8,920 | 1,115 | 80,283 | 7.8 ms | +242 |
| 5 | 11,005 | 1,376 | 99,046 | 9.7 ms | +261 |
| 6 | 12,369 | **1,546** | 111,317 | 10.6 ms | +170 |

**⭐ 核心结论：dep8 在 1–6 prefill 全程 prefill-feed-limited，decode 根本没喂饱。**

- **output÷8 近似线性增长**（每加 1 prefill ~+250/GPU），到 6 prefill 才 1,546/GPU = 官方 11,200 的 **14%**。N=5→6 增量收窄到 +170（decode 开始有一点感觉），但**远未饱和**。
- **decode 有巨大余量**：TPOT 全程 5–11ms（DSpark 在 deepgemm 上极健康），即便 6 prefill 也只 10.6ms（对比官方运行点 ~20ms/50 tok/s-user 还快一倍）。decode 不是瓶颈，**prefill 喂料才是**。
- **要打满 dep8 需 ≫6 prefill**：按线性外推，逼近 11,200/GPU 需要远多于 6 个 prefill（SGLang 官方 8,993 用了 **16 prefill**，且是 8 GPU decode 同规模）。本环境只有 6 个 deepgemm prefill pod，够看清趋势但喂不满 dep8。
- **对比 TP4 decode（§9.10/§9.11-1p1d/2p1d，4 GPU decode，4k1k）**：TP4 decode 用 1–2 prefill 就接近喂饱（decode GPU 少、输入短）；dep8（8 GPU decode + 8k 长输入）需要**成倍的 prefill** 才能喂饱——decode 越宽、输入越长，越吃 prefill。**扩 dep8 吞吐的杠杆 = 加 prefill 到 ~2.5:1 (prefill:decode-GPU) 甚至更高**，不是加 decode。

> **一句话**：dep8 decode 本身（deepgemm 镜像 + FP4 + DSpark，TPOT ~7ms）性能是够的；限制 dep8 打不到 11,200 的是 **prefill 喂料容量**（8k 输入 + 8 decode GPU 需要 ≫6 prefill）。这与 §9.9 早期"decode 线性、prefill 决定 TTFT"的规律一致，只是绝对值在正确镜像下高得多。

#### 9.11.1 ⭐⭐⭐⭐ 同一 dep8 在 4k1k 下重扫：验证"输入减半 = 喂料翻倍"（2026-07-24）

**动机**：§9.11 用 8k1k，每 prefill 只喂 ~2,453 output。1p1d（§9.10）在 **4k1k** 下单 TP4 prefill 却喂到 ~5,056 output（Total 25,281）。差异纯粹来自**输入长度**：8192 输入 = 2× prefill 算力/请求 = 一半的请求速率 = 一半喂料。为验证并找 dep8 更真实的 decode 上限，把**同一套 dep8**（不重启，同一 head endpoint）换成 4k1k，重扫 1–4 prefill。

| prefill 数 | Output tok/s | **Output÷8 /GPU** | Total tok/s (in+out) | Median TPOT | vs 8k1k 同档 output |
|---|---|---|---|---|---|
| 1 | 5,056 | 632 | 25,281 | 6.0 ms | 2,453 → **2.06×** |
| 2 | 9,902 | 1,238 | 49,508 | 6.9 ms | 4,773 → 2.07× |
| 3 | 13,245 | 1,656 | 66,225 | 9.3 ms | 6,982 → 1.90× |
| 4 | 17,775 | **2,222** | 88,877 | 9.0 ms | 8,920 → 1.99× |

**⭐ 核心结论：**

1. **"输入减半→喂料翻倍"精确成立**：4k1k 每档 output 恰好是 8k1k 的 ~2×（2.06/2.07/1.90/1.99）。"输入减半 → 每 prefill 喂料翻倍"的直觉方向对——4k 下每 prefill 顶 8k 的两个。
2. **但 dep8 比预想更能扛**：即便 4k1k + 4 prefill（output 17,775，2,222/GPU），dep8 decode **仍未饱和**——output 随 prefill 近似线性（增量 ~4,900/4,950/4,415/4,531），TPOT 只从 6.0ms 微升到 9.0ms（N=3 起才有一点感觉）。2 prefill 只到 1,238/GPU，离打满还很远。
3. **全程仍 prefill-feed-limited**：TTFT 从 197s（1p）降到 40s（4p），说明请求还在排队等 prefill；TPOT 全程 <10ms 说明 decode 一直有富余。dep8 的 decode 天花板在 4k1k / 4 prefill 内**没探到**，要真正打满需继续加 prefill（本环境仅 6 个 deepgemm prefill pod）。
4. **杠杆确认**：无论 4k 还是 8k，扩 dep8 吞吐的杠杆都是**加 prefill**，不是加 decode GPU。4k 把每 prefill 的喂料效率翻倍，所以要逼近 11,200/GPU，4k 需要的 prefill 数约是 8k 的一半，但绝对数量依然 ≫4。

> **对比一句话**：8k1k 6 prefill 才 1,546/GPU；4k1k 4 prefill 已 2,222/GPU。同样的 dep8，输入短一半，同样 prefill 数下 output 翻倍、TPOT 只多几毫秒。dep8 decode 是"猛兽"，瓶颈永远在 prefill 侧的喂料带宽。

#### 9.11.2 ⭐⭐⭐⭐⭐ 扫到 16 prefill 找 dep8 天花板 + KV 失败策略崩溃坑（2026-07-24）

**动机**：§9.11 只到 6 prefill（1,546/GPU），距 SGLang 官方 dep8 8k1k 的 **~9,000 tok/s/chip** 还差 6×。补 10 个 prefill 节点（本地盘已有权重，无需下载），凑齐 **16 prefill**（= SGLang 官方 dep8 拓扑）跑满 8k1k，看 output÷8 能否逼近 9,000。

**完整 1→16 prefill 曲线（8k1k, conc1024, output÷8/GPU）：**

| prefill 数 | Output tok/s | **Output÷8 /GPU** | Median TPOT | 状态 |
|---|---|---|---|---|
| 1 | 2,453 | 307 | 5.4 ms | 干净 |
| 2 | 4,773 | 597 | 5.6 ms | 干净 |
| 4 | 8,920 | 1,115 | 7.8 ms | 干净 |
| 6 | 12,369 | 1,546 | 10.6 ms | 干净 |
| 8 | 17,614 | 2,202 | 10.7 ms | 干净 |
| 10 | 21,433 | 2,679 | 15.1 ms | 干净 |
| **12** | **25,500** | **3,187** | 14.3 ms | **干净峰值** |
| 14 | — | — | — | ❌ **dep8 崩溃** |
| 16 | — | — | — | ❌ dep8 已死 |

**⭐ 核心发现 1：dep8 8k1k 稳定峰值 ≈ 3,187/GPU @ 12 prefill = SGLang 9,000/chip 的 35%。**

曲线 1→12 prefill 近似线性上升（307→3,187/GPU），TPOT 全程 5–15ms 健康。**12 prefill 是最后一个干净点**，此时 output 还在涨、TPOT 才 14ms，dep8 明明有余量——但 14 prefill 时直接崩了。

**⭐ 核心发现 2（重要坑）：`kv_load_failure_policy: fail` 让 dep8 在高 prefill fan-in 下脆崩。**

- **现象**：14 prefill 时 dep8 `EngineDeadError` 整个引擎死亡（ApiServer_0 died），16 prefill router 全是 `Connection refused`（dep8 已死）。首轮 14p output 反降、TPOT 跳 40ms、16p 仅 320/4096 成功——全部作废。
- **真凶**：decode 日志 `Failed to notify KV connector about rejected request` + `kv_load_failure_policy: fail`。14+ prefill 并发往 dep8 推 KV（NixlConnector over MNNVL），decode 侧 KV 块吃紧时某个 KV transfer 被 **reject**，`fail` 策略把这个**瞬时 reject 升级成致命 engine 死亡**。**不是算力天花板，是错误处理策略太脆。**
- **修复尝试**：`kv_load_failure_policy` 合法值只有 `recompute` / `fail`（`vllm/config/kv_transfer.py:69`）。改 `recompute`（KV 失败时 decode 本地重算而非崩溃）后重启 dep8：
  - 12p recompute：2,645/GPU，TPOT 24.8ms（比 fail 的 3,187 低 17%——recompute 税：KV 失败重算吃 decode 算力）
  - 14p recompute：2,883/GPU，但只 3,152/4096 成功，**测后 dep8 仍崩溃**
- **结论**：recompute 缓解但没根治。dep8 在 8k1k / 14 prefill fan-in 处是**硬崩溃墙**——KV transfer 容量 + decode KV 块在 14+ prefill 同时灌入时耗尽，即便 recompute 也扛不住。

**⭐ 核心发现 3：距 SGLang 9,000/chip 的 3× 差距既不是精度、也不是 MTP——两边都开。**

> ⚠️ **更正（2026-07-24）**：早前版本误写"本配置无 MTP"。实测 **MTP 一直开着**：模型 config `num_nextn_predict_layers: 1`（1 层 MTP/nextn 模块），加载日志 `Resolved architecture: DeepSeekV4MTPModel`，`--speculative-config {method:dspark, num_speculative_tokens:7}` 底层就是用 MTP head 投机 draft 7 token。**DSpark = DeepSeek 的 MTP 投机解码**，不是另一回事。

dep8 峰值 3,187/GPU vs SGLang 9,000/chip，同 checkpoint（FP4 experts + FP8 dense）、同开 MTP，差距真正来源：

1. **spec 配置节流 decode（关键嫌疑）**：启动日志 warning `max_num_scheduled_tokens is set to 2048 based on the speculative decoding settings. This may lead to suboptimal performance.`——`num_speculative_tokens=7` + `max_num_seqs=1024` + `max_num_batched_tokens=8192` 这组合把 decode 调度 batch 掐到 2048，vLLM 自己判定次优。draft 7 token（单 MTP head autoregressive 调 7 次）也可能过头，接受率在后几个 token 掉。SGLang 的 MTP token 数 + batch 配置更优。
2. **KV 传输鲁棒性 + 崩溃**：SGLang（Mooncake）高 fan-in 下比 vLLM NixlConnector 稳；vLLM dep8 在 14 prefill 就崩（见发现 2），够不到 SGLang 的 16-prefill 运行点，最优点被截断在 12p。
3. **待验证**：EPLB 专家负载均衡、`max_num_batched_tokens` 调大给 spec draft 留槽、`num_speculative_tokens` 下调到 3–4。

> **一句话**：dep8 8k1k 在本配置下稳定峰值 ≈ 3,187/GPU @ 12 prefill（SGLang 9,000 的 35%）。差距**不是精度、不是 MTP**（两边都有），而是 (a) spec 配置把 decode batch 节流到 2048、(b) 14 prefill 起 `kv_load_failure_policy: fail`（+ KV 块耗尽）硬崩溃够不到满配运行点。要逼近 SGLang 需调 spec batch 槽（增 max-num-batched-tokens / 减 num_speculative_tokens）+ 更鲁棒 KV 传输 + 更保守 KV 显存。**本轮两大工程收获：fail→recompute 崩溃坑 + DSpark 即 MTP 的正名 + spec batch 节流 warning。**

#### 9.11.3 ⭐⭐⭐⭐ 调优轮：抬 decode cap（batch 8192→16384）（2026-07-24）

**改动**：`max-num-batched-tokens` 8192→16384（decode 调度 cap 2048→10240，spec 节流 warning 消失）+ `kv_load_failure_policy: recompute` + spec7 + mem 0.85。
> 注：首次尝试 `gpu-mem-util 0.85→0.92` **启动即 CUDA OOM**（276GB 用了 268GB，再要 8GB 就爆）——0.92 + 大 batch 激活缓冲挤爆显存。退回 0.85。**教训：GB300 上 batch 与 mem-util 要联动，加 batch 必须留显存。**

| prefill | Output÷8 /GPU | TPOT | 成功数 | 测后 dep8 | vs 基线 |
|---|---|---|---|---|---|
| 12 | 3,011 | 22.7 ms | 4095/4096 | ✅ 存活 | recompute 12p 2,645 → **+14%** |
| 14 | **3,222** | 21.5 ms | 3572/4096 | ❌ 崩溃 | 超 fail-12p 峰值 3,187 |
| 16 | — | — | — | ❌ 已死 | 跳过 |

**结论**：
1. **抬 cap 确实提吞吐**：recompute 12p 2,645→3,011（+14%），14p 冲到 **3,222/GPU 新峰值**（超过之前 fail 策略 12p 的 3,187）——证明之前 decode 确实被 2048 cap 节流。
2. **但没根治崩溃墙**：14p 仍崩（3572/4096 后 dep8 死）。KV 块在 14-prefill fan-in 下耗尽，batch/mem 调不动这个墙——是 NixlConnector KV 传输 + decode KV 块分配的结构问题。
3. **新峰值 3,222/GPU = SGLang 9,000 的 36%**（小幅提升，崩溃墙未破）。

> **下一步旋钮**（未验证）：(a) 降 `max_num_seqs` 1024→512（减并发换 KV 块余量抗 fan-in 崩溃）；(b) 降 `num_speculative_tokens` 7→3（减每 seq 的 KV/算力）；(c) router 侧限 prefill fan-in 速率；(d) 换更鲁棒 KV 传输（对标 SGLang Mooncake）。**根治崩溃是打到 SGLang 9,000 的前提——只有稳住 16 prefill 满配才够得着。**

#### 9.11.4 ⭐⭐⭐⭐⭐ 崩溃根因彻底定位：两个独立失效模式（2026-07-24）

开 `PYTHONFAULTHANDLER=1` + `core_pattern→/mnt/ssd` 复现后，崩溃拆成**两个完全独立的 bug**：

**失效模式 A — ephemeral storage 耗尽驱逐（14p killer，已修复）**

- **根因链**：decode worker 崩溃时吐 **7.5–9.4GB core dump 到容器根目录 `/`（ephemeral 系统盘）**。`core_pattern` 默认 `/core.%e.%p.%t` 写容器根。多次崩溃 core 累积（+ tmp/inductor cache 也写小盘）→ 节点 ephemeral 超阈值（threshold 10GB，实测占 51GB）→ **kubelet 驱逐 decode-worker pod（`Evicted` exit 137）** → head 等不到 remote DP worker（decode 第二节点） → dep8 判死。**之前所有 `EngineDeadError` / `Failed to notify KV connector about rejected request` 全是下游假象。**
- **修复**：`ulimit -c unlimited` + `core_pattern→/mnt/ssd`（12TB RAID）+ `TMPDIR`/`XDG_CACHE_HOME`/`TORCHINDUCTOR_CACHE_DIR→/mnt/ssd`。
- **验证**：修复后 **14p 干净通过 4096/4096，output 26,568=3,321/GPU，dep8 存活，0 core**（此前 14p 必崩）。**14p 3,321/GPU 是当前最优稳定点。**
- 诚实注明：修复是治真病根（磁盘耗尽），非关 core dump 掩盖。

**失效模式 B — CUDA device-side assert（16p，未修复）**

修复 A 后冲 16p 仍崩，但 **decode-worker pod=Running（未被驱逐）**——证明是**另一个 bug**，不是 ephemeral：

- **首崩**：`Worker_DP1` 在 `model_runner.py:1073 sample` → `RuntimeError: Triton Error [CUDA]: device-side assert triggered`（sampling kernel 设备侧断言）。
- **级联**：DP1 死 → DP-attention 每步的 gloo all_reduce padding 同步（`dp_utils.py sync_cudagraph_and_dp_padding`）在其他 rank 报 `Connection closed by peer` → `ProcessGroupNCCL::HeartbeatMonitor` 检测 TCPStore 断连 → **SIGABRT 级联全 8 DP rank**（faulthandler 显示各 rank 死在 `cumem.py release_pools`/GC 的 teardown 路径）→ dep8 死。
- **结果**：16p 3529/6144 partial，output 22,389=2,799/GPU。core（9.3GB/rank）已安全落 /mnt/ssd。
- **推断**：device-side assert 大概率是 sampling 数值边缘（logits NaN/inf 或 invalid sample index），16p 更高并发统计上更易触发，可能与 DSpark spec + FP8/FP4 numerics 相关。async CUDA error 使上报栈定位到 `sample`，确切 assert 需 `CUDA_LAUNCH_BLOCKING=1 + TORCH_USE_CUDA_DSA=1` 重跑锁定。gdb 分析 core 未成（镜像 apt 源无 gdb 包）。

**当前结论**：
| prefill | output÷8/GPU | 状态 |
|---|---|---|
| 12（ephemeral 修复前 fail 策略）| 3,187 | 干净 |
| **14（ephemeral 修复后）** | **3,321** | **干净 4096/4096，稳定最优** |
| 16（ephemeral 修复后）| 2,799 | ❌ CUDA device-side assert 崩（partial） |

> **一句话**：dep8 崩溃是**两个叠加 bug**——(A) core dump 撑爆 ephemeral 驱逐 pod（已修：core/tmp/cache→/mnt/ssd，14p 现稳达 3,321/GPU）；(B) 16p 下 sampling kernel CUDA device-side assert 触发 gloo/NCCL 级联 abort（未修，需 CUDA_LAUNCH_BLOCKING 锁定确切断言）。**14p 3,321/GPU = SGLang 9,000 的 37%，是当前可稳定复现的最优。** 逼近 SGLang 仍需 (B) 的数值修复 + AFD 级架构变更（见 [[concepts/afd-attn-ffn-disaggregation]]）。

#### 9.11.5 ⭐⭐⭐⭐⭐ decode MFU + HBM 实测：memory-bound 实锤（2026-07-24）

14p 稳定档（output 26,568 tok/s = 3,321/GPU）下 `nvidia-smi` 实测两个 decode 节点全 8 GPU：

| 指标 | 空载 | 14p 负载 |
|---|---|---|
| GPU util | 0% | **100%**（个别 dip 70–92% = DP 不均衡/async output）|
| 功耗 | ~220 W | **1,000–1,120 W**（近满载）|
| SM 时钟 | — | **2,070 MHz（最大 boost）** |
| HBM 占用 | 263.5 GB | 263.5 GB（KV 启动即预分配，负载不变）|
| HBM 总量 | 284 GB（276 GiB 可用）| **占用 92.7%，仅剩 ~20 GB** |
| KV cache | — | **~90 GB/GPU**（97.2 GB, 64358 blocks）|

**算力利用率：**
- **MFU ≈ 6%**：achieved(accepted-token GEMM) = 2×37e9×26,568 = 246 TFLOP/s/GPU；混合峰值（experts FP4 dense 5,000 + attn/dense FP8 2,500，加权 ~4,000 TFLOP/s/GPU）→ 6.2%。（注：MTP spec draft+verify + 8k attention 使原始 forward 计算 > accepted-token，硬件实际更忙，故 util 100%。）

**⭐ 核心判断：decode 是 memory-bandwidth-bound，不是"空转饿死"。**

- **100% util + 满功耗满时钟 + 仅 6% MFU** = SM 全忙但卡在 HBM 读取（小 batch 下 expert GEMM 算术强度太低 + 8k KV 读取吃带宽）。这不是"idle 等 token"，是"忙着读内存、算得少"——**memory-bound 的典型签名**。
- **HBM 92.7% 占满（KV 90GB + 权重挤同一块 HBM），仅剩 20GB** → 无法通过加大 `max_num_seqs`/batch 来提高每 expert 的 token 数（一加就 OOM，实测 mem 0.92 启动即 OOM 印证）。
- **这实锤了 [[concepts/afd-attn-ffn-disaggregation]] 的论断**：coexist decode 下 KV 与 FFN 权重争抢同一 HBM，长上下文(8k)让 KV 吃满显存 → MoE batch 缩小 → 算术强度低 → memory-bound → MFU 仅 6%。

**优化方向（数据指向）：**
1. **AFD（attn/ffn 分离）**：把 FFN 权重挪到独立 GPU，attention GPU 释放出 HBM → 更大 KV/batch → 每 expert 更多 token → 提 MFU。这是结构性大招（见 wiki 页）。
2. **压缩 KV**：DeepGEMM PR#304 的 FP4 Indexer 把 V4 KV 从 132→68 byte/token(-48%)——KV 从 90GB 砍半可腾出 HBM 加 batch。
3. **不能简单加 max_num_seqs**：HBM 已 93% 满。

> **一句话**：dep8 decode 在 14p 是 **100% util / 6% MFU / HBM 93% 满 / KV 90GB** 的 memory-bandwidth-bound 状态——compute 有 94% 余量但被内存墙锁死，且 HBM 无空间加 batch。这就是 vLLM coexist dep8 只有 SGLang 37% 的物理根因，也印证了 AFD "长上下文饿死 MoE"。真正提升要么 AFD 释放 HBM、要么压缩 KV 腾空间。

### 9.7 GCS 传输 auth 坑
GKE 节点 compute SA 对模型 bucket **OAuth scope 未授权**。上传/下载用：本机 `gcloud auth application-default print-access-token` → cp token 进 pod → `CLOUDSDK_AUTH_ACCESS_TOKEN=<token> gcloud storage cp ... --billing-project=<project>`（`gcloud auth login --cred-file` 不吃 authorized_user ADC；只有 `CLOUDSDK_AUTH_ACCESS_TOKEN` 能让 gcloud CLI 用上）。用完删 token。

---

*SGLang 实跑对标见 [`./sglang-v4-gb300-benchmark.md`](./sglang-v4-gb300-benchmark.md)。榜单值（§4）仍为官方/InferenceX 公开数据。*
