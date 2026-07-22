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

*基于公开来源整理（vLLM blog / recipe 站 / SemiAnalysis InferenceX），2026-07-22。**本环境未实跑**，数字为官方/榜单值；实跑配套见 §6。SGLang 实跑对标见 [`./sglang-v4-gb300-benchmark.md`](./sglang-v4-gb300-benchmark.md)。*
