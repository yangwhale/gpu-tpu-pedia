# Kimi K3 · PD 分离 —— 实验设计（BACKLOG，本轮不做）

> **状态：`[未开始 · 非官方外推]`**
>
> 官方（vLLM day-0 博客 / preview / recipes）**没有公布任何 K3 的 PD 吞吐数字或 P:D 配比**，
> 只提了一句已验证拓扑。本文的方法论**全部外推自 vLLM 2026-07-23 的 GLM-5.2 PD 专文**。
>
> ⚠️ **跨模型搬运 PD 经验风险很高**：K3 是 KDA 混合缓存 + DSpark 投机，
> GLM-5.2 是 DSA 稀疏注意力 + MTP，两者在 PD 交接处的细节完全不同。
> **先把 [主 Runbook](./VLLM-K3-RUNBOOK.md) §0–§5 官方路径跑通、拿到本环境基线**，
> 再回来做这里的事。否则分不清问题出在 PD、出在模型、还是出在环境。

## 缺口与设计

### 1 先说清楚官方给了什么、没给什么

翻遍 day-0 博客、preview 博客、recipes 页，**关于 PD 分离，vLLM 只给了这些**：

| 给了 | 内容 |
|---|---|
| 一个已验证拓扑 | **TEP8 prefill → DEP16 decode**，NIXL 传 KV |
| Pareto 两个端点 | 高吞吐端 **2K+ TPGS**（tokens/GPU/s），低延迟端 **100+ TPS/user** |
| 一张 Pareto 曲线图 | 只有图，无数据表 |

**没给的**（也就是你要的那些）：

- ❌ P:D 配比扫描 —— 几 P 几 D 是拐点
- ❌ PD 下的 total token TPS
- ❌ PD 下的 generate speed / TPOT / TTFT 分档
- ❌ 不同 ISL（8K / 32K / 128K / 1M）下配比怎么变

所有公布的 tok/s 数字（111 / 118 / 331 / 370）**全是 bs=1 单用户、非 PD 的 TP8/TP16**。
换句话说：**K3 的 PD 吞吐曲线目前是空白，谁先测谁先有。**

### 2 可借的方法论：vLLM 三天前的 GLM-5.2 PD 专文

同一批人 2026-07-23 发了
[GLM-5.2 on 24× B300 的 PD 实战](https://vllm.ai/blog/2026-07-23-glm-5.2-nvfp4-b300-pd)，
**卡数正好也是 24**，方法论可以直接搬。要点：

| 维度 | GLM-5.2 的做法 |
|---|---|
| 拓扑 | **4 Prefill + 1 Decode**：prefill `TP1 DP4 EP`×4 实例（16 卡）+ decode `TP1 DP8 EP`×1（8 卡） |
| 目标 | **不是峰值吞吐**，而是 SLA 内的最大吞吐：mean TTFT ≤ 2.5 s、mean TPOT ≤ 20 ms |
| 批大小 | **不固定**。按 request rate 打，并发是 SLA 允许下的自然结果（8K 时约 700 并发，16K 约 300，256K 约 25） |
| 压测注入 | `--request-rate = 目标 TPS ÷ (input_len + output_len) × 调节系数` |
| 度量 | 配置用的卡数不同，所以看 **TGS（每 GPU 吞吐）**，不看绝对值 |

**三条可以直接抄的结论：**

**① 最大的一个坑在 PD 与投机解码的交界处，不在任何单个 kernel 里。**
请求刚从 prefill 交接过来时只算 1 个 token，而 decode 上已有的请求是 1+N（开了投机），
形状不一致 → 批次变成 mixed batch → 掉出 full-CUDA-Graph 快路径。
DP 下更糟：任何一个 rank 收到新请求，其余 rank 全都跟着走慢路径，
而稳态 PD 下新请求是持续到达的，等于**慢路径常驻**。
修法是 decode 侧对新请求做 **speculative padding** 补到 1+N。
**这一个改动把 TPOT 从 ~40 ms 干到 ~22 ms**，是整个优化里最大的一笔。

> K3 同样是 PD + 投机（DSpark），**这个坑几乎肯定同样存在**。
> 上游修复是 PR #45237，确认镜像里已包含。

**② 不要选 TGS 最高的配置。** GLM 侧 `TP1 DP2 EP` 每卡吞吐最好，
但每实例只有 2 卡、KV cache 装不下长上下文，最终发的是 `TP1 DP4 EP`，
**用 8% 的每卡效率换 KV 容量**。K3 是 1M 上下文，这个权衡只会更极端。

**③ 小规模上 EP 是负收益。** GLM 侧实测 `TP2 + EP` 比纯 `TP2` 还慢——
两卡规模下 all-to-all 开销盖过收益。EP 需要足够多专家摊到足够多设备上才划算。
K3 有 896 个专家，理论上比 GLM 更吃得住 EP，但**小实例上仍要实测验证**。

### 3 建议的扫描方案（GB300，4 GPU/节点）

从官方那个已验证拓扑起步，它恰好也是 24 卡：

**基线：1× TEP8 prefill（2 节点）+ 1× DEP16 decode（4 节点）= 6 节点 / 24 卡**

然后固定总卡数扫 P:D：

| 轮次 | Prefill | Decode | 总卡 | 观察什么 |
|---|---|---|---|---|
| A（基线） | TEP8 × 1（8 卡） | DEP16 × 1（16 卡） | 24 | 官方唯一背书的拓扑，先复现 |
| B | TEP8 × 2（16 卡） | DEP8 × 1（8 卡） | 24 | prefill 加倍：是否 TTFT 富余、decode 变瓶颈 |
| C | TEP4 × 2（8 卡） | DEP16 × 1（16 卡） | 24 | prefill 拆小实例，测 EP 在 4 卡规模是否仍正收益 |
| D | TEP8 × 1（8 卡） | DEP8 × 2（16 卡） | 24 | decode 拆两实例 vs 一个大实例 |

**每轮固定 ISL 扫三档：8K / 32K / 128K**，因为 GLM 的经验是
**并发随 ISL 上升而自然下降**，配比拐点会跟着 ISL 移动——
本仓库 V4 的教训完全一致（`--swa-full-tokens-ratio` 最优值跟 ISL 绑定，4K→0.15、8K→0.10）。

**判据用 SLA 而不是峰值**，建议先沿用 GLM 那套：mean TTFT ≤ 2.5 s、mean TPOT ≤ 20 ms，
在此约束下最大化 total token TPS。同时记 **TGS（每 GPU 吞吐）**，因为各轮卡数分配不同。

### 4 PD 启动命令骨架 `[GLM 模板 + K3 参数，未实测]`

**Prefill 节点**（TEP8：attention TP + MoE EP，序列并行自动开）：

```bash
export VLLM_USE_V2_MODEL_RUNNER=1        # MoE 模型默认不开，必须显式打开（GLM 侧 -11% TPOT）
export VLLM_USE_RUST_FRONTEND=1
export VLLM_ALLREDUCE_USE_FLASHINFER=1
export UCX_TLS="rc,cuda_copy"            # 走 RDMA 时必设，否则 KV 传输不走 RDMA

vllm serve moonshotai/Kimi-K3 \
  --trust-remote-code \
  --enable-prefix-caching \
  --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_producer"}' \
  -ep -tp 8 \
  --moe-backend deep_gemm_mega_moe \
  --all2all-backend flashinfer_nvlink_one_sided \
  --load-format fastsafetensors \
  --gpu-memory-utilization 0.92 \
  --enable-auto-tool-choice --tool-call-parser kimi_k3 --reasoning-parser kimi_k3 \
  --shutdown-timeout 300
```

**Decode 节点**（DEP16）：

```bash
export VLLM_USE_V2_MODEL_RUNNER=1
export VLLM_USE_RUST_FRONTEND=1
export UCX_TLS="rc,cuda_copy"

vllm serve moonshotai/Kimi-K3 \
  --trust-remote-code \
  --enable-prefix-caching \
  --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_consumer"}' \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
  --max-num-batched-tokens 1024 \
  -ep -tp 1 -dp 16 \
  --moe-backend deep_gemm_mega_moe \
  --all2all-backend flashinfer_nvlink_one_sided \
  --kv-cache-dtype fp8 \
  --attention-config '{"use_prefill_query_quantization":true}' \
  --gpu-memory-utilization 0.90 \
  --speculative-config '{"model":"Inferact/Kimi-K3-DSpark","method":"dspark","num_speculative_tokens":7,"attention_backend":"FLASHINFER_MLA","draft_sample_method":"probabilistic","rejection_sample_method":"block"}' \
  --shutdown-timeout 300
```

**几处相对 GLM 模板的 K3 专属改动，别漏：**

- `--all2all-backend`：**NVLink 用 `flashinfer_nvlink_one_sided`**（GLM 那篇用的是 two_sided，
  因为发文时 one_sided 还没有；K3 recipes 明确推荐 one_sided）
- `--moe-backend deep_gemm_mega_moe` —— DEP 环境的官方推荐；TP>1 才用 `flashinfer_trtllm`
- `--enable-prefix-caching` —— K3 默认关，必加
- 投机解码用 **DSpark**（`num_speculative_tokens: 7`），不是 MTP。
  GLM 那边 prefill 侧用 1、decode 侧用 3 的**非对称设法值得借鉴**：
  prefill 只管尽快交出 KV，深度投机没意义；decode 在延迟关键路径上才值得深投

### 5 压测注入方式 `[GLM 方法]`

PD 场景不能再用 `--max-concurrency 1` 那种单流打法，要按速率注入：

```bash
vllm bench serve \
  --backend openai-chat \
  --model moonshotai/Kimi-K3 \
  --endpoint /v1/chat/completions \
  --dataset-name random \
  --random-input-len 16384 \
  --random-output-len 1000 \
  --request-rate <目标TPS ÷ (16384+1000) × 调节系数> \
  --percentile-metrics ttft,tpot,itl,e2el \
  --metric-percentiles 50,90 \
  --save-result
```

random 数据集**没有前缀缓存命中**，所以 TTFT 是最坏情况的纯计算延迟——
这点要在报数时说明，否则跟开了 prefix cache 的生产数字对不上。

### 6 必须盯的四组监控 `[GLM 经验]`

PD 分离的可观测性比单实例难，因为延迟被劈成两个池：TTFT 归 prefill，TPOT 归 decode，中间夹一次 KV 传输。
出问题时用户只看到「变慢了」。要分池看：

1. **分池的 TTFT / TPOT 分位数** —— 判断问题属于 prefill 还是 decode，第一个要看的
2. **投机接受率与平均接受长度** —— 最容易忽略但最早预警。接受率下滑**不报错**，只是 TPOT 慢慢劣化。GLM 那边把它当一级告警指标
3. **KV cache 利用率 vs GPU 利用率**，两侧都看 —— PD 的核心价值就是两类资源独立伸缩，这两条曲线的相对高低就是扩容信号
4. **KV 传输延迟与队列深度** —— 判断是不是跨节点网络成了瓶颈

> **另一个只有长跑才看得见的坑**：GLM 那边多日连续跑发现 vLLM 进程 RSS 线性增长
> （721 GiB → 800 GiB，几十小时不收敛），GPU 侧指标全正常，且因为 `EngineCore` 启动时调了
> `gc.freeze()`，`tracemalloc` 也看不见。根因是 KV block id 列表只记录不清空。
> 已在上游修复。**验收要跑多日连续测试，不能只跑几分钟 benchmark。**
