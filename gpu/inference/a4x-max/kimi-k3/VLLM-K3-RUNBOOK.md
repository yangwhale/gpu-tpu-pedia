# vLLM · Kimi K3 (2.8T) · GB300 NVL72 端到端 Runbook

> 从 0 开始：起 pod → 建 RAID → 拉 1.4 TB 权重 → 起 TP8 服务 → 压测 → 复现官方 331/370 tok/s。
>
> **每节的状态标记含义**
> - `[官方]` —— 逐字来自 vLLM day-0 博客 / recipes，未在本环境跑过
> - `[本环境]` —— 复用本仓库已验证过的环境步骤（V4 那套），对 K3 做了参数替换
> - `[待验证]` —— 需要跑通后回填的位置
>
> 全文尚无 `[已验证]`。跑通一节就把该节改标，并在 §10 记录。

---

## ⚠️ 开跑前必须知道的三件事

**一、现在只能用 Docker 镜像。** 官方原话：由于依赖复杂（含 FlashInfer 等多个 pre-release 依赖），
**目前只有 Docker 镜像可用**，pip 装不起来。镜像 tag 见
<https://recipes.vllm.ai/moonshotai/Kimi-K3>。

**二、`--enable-prefix-caching` 不加就没有。** vLLM 对绝大多数模型默认开启 prefix caching，
**唯独 K3 默认关闭**。不加这个 flag，KDA 混合缓存那一整套优化等于没有。

**三、TP8 在 GB300 上是跨 2 个节点。** GB300 NVL72 每节点 4 GPU，
所以官方 recipe 里的 `--tensor-parallel-size 8` 必须配 `--nnodes 2`。
两个节点**必须在同一 NVL72 subblock 内**（KV 走域内 NVLink，跨域不行）——
这条是本环境 V4 runbook 的血泪教训，K3 同样适用。

---

## 0. TL;DR `[官方]`

最简起法（单节点 8×B300，或 GB300 上凑够 8 卡）：

```bash
vllm serve moonshotai/Kimi-K3 \
  --tensor-parallel-size 8 \
  --trust-remote-code \
  --load-format fastsafetensors \
  --enable-prefix-caching \
  --enable-auto-tool-choice \
  --tool-call-parser kimi_k3 \
  --reasoning-parser kimi_k3
```

要低延迟就加 DSpark 投机解码（单流解码约 3 倍）：

```bash
--speculative-config '{"model":"Inferact/Kimi-K3-DSpark","method":"dspark","num_speculative_tokens":7,"attention_backend":"FLASHINFER_MLA","draft_sample_method":"probabilistic","rejection_sample_method":"block"}'
```

---

## 1. 前置条件 `[本环境]`

```bash
# ① 两个节点在同一 NVL72 subblock（manifest 的 podAffinity 保证）
#    K3 的 TP8 跨节点走 NVLink，跨域会直接崩性能

# ② RAID 挂载 —— 先查这个
#    正常 12T；若显示 256K 说明 RAID 没挂，1.4 TB 权重放不进去
df -h /mnt/disks/raid/0

# ③ 能拉 vLLM K3 镜像（imagePullSecrets）
#    GB300 集群有 CronJob 自动刷 ar-pull-secret，YAML 里只需 imagePullSecrets

# ④ 没有孤儿 ComputeDomain 占着 DRA channel
kubectl get computedomain -A
```

> RAID 建不起来看 [gb300-local-ssd-raid0-SETUP.md](../deepseek-v4/gb300-local-ssd-raid0-SETUP.md)，
> 注意里面 `md0 → md127` 那个陷阱。

---

## 2. 部署 pod fleet `[本环境]`

复用 V4 的 manifest，改镜像和副本数即可：

```bash
# 参考 ../deepseek-v4/manifests/vllm-fleet.yaml
# 需要改的：
#   - image: 换成 vLLM K3 day-0 镜像
#   - replicas: TP8 → 2；TP16 → 4
#   - podAffinity: 保持同 subblock
kubectl apply -f manifests/k3-fleet.yaml
kubectl get pods -o wide | grep k3
```

---

## 3. 拉权重 `[本环境]`

**体量：2.8T 参数 × MXFP4 ≈ 1.4 TB。** 比 V4-Pro 的 832G 还大 70%，
内存盘绝对放不下，必须落 RAID。

```bash
# 别用 gcloud —— vLLM 镜像里没装。用 curl + GCS JSON API 的并行脚本
# 参数化版本已在 V4 目录：../deepseek-v4/scripts/pull-gcs-model.sh
BUCKET=<your-bucket> PREFIX=Kimi-K3 JOBS=16 \
  bash scripts/pull-gcs-model.sh /mnt/disks/raid/0/Kimi-K3

# DSpark draft 模型（小，直接 HF 或 GCS 都行）
#   Inferact/Kimi-K3-DSpark

# ⚠️ access token 只有 1 小时有效期，1.4 TB 拉不完就会断 —— 拉之前先刷新
# ⚠️ 重建任何 pod 之后都要重新校验 shard 数（hostPath 不跟着 pod 走）
du -sh /mnt/disks/raid/0/Kimi-K3
ls /mnt/disks/raid/0/Kimi-K3/*.safetensors | wc -l   # [待验证] 填实际 shard 数
```

---

## 4. 启动服务 · TP8 + DSpark `[官方]`

这一段是**官方 reproduce recipe 的逐字复刻**，只把 `HEAD_ADDR` 换成本环境地址。

### 4.1 环境变量

```bash
export NCCL_DMABUF_ENABLE=0
export VLLM_ALLREDUCE_USE_FLASHINFER=1
export VLLM_USE_RUST_FRONTEND=1
export VLLM_ENGINE_READY_TIMEOUT_S=3600      # 2.8T 加载慢，别用默认超时
export HEAD_ADDR=<node-0 IP>
```

### 4.2 起服务（两个节点各跑一次，只有 `--node-rank` 不同）

```bash
vllm serve moonshotai/Kimi-K3 \
  --enable-prefix-caching \
  --tensor-parallel-size 8 \
  --nnodes 2 \
  --node-rank 0 \
  --moe-backend auto \
  --trust-remote-code \
  --load-format fastsafetensors \
  --max-num-seqs 512 \
  --gpu-memory-utilization 0.9 \
  --max-model-len auto \
  --max-cudagraph-capture-size 256 \
  --kv-cache-dtype fp8 \
  --attention-config '{"mla_prefill_backend":"FLASHINFER","use_prefill_query_quantization":true}' \
  --speculative-config '{"model":"Inferact/Kimi-K3-DSpark","method":"dspark","num_speculative_tokens":7,"attention_backend":"FLASHINFER_MLA","draft_sample_method":"probabilistic","rejection_sample_method":"block"}'
```

node-1 上把 `--node-rank 0` 改成 `--node-rank 1`，其余完全相同。

### 4.3 参数逐条解释

| 参数 | 为什么 |
|---|---|
| `--enable-prefix-caching` | **K3 默认关**，不加则混合 KDA 缓存全部失效 |
| `--tensor-parallel-size 8` + `--nnodes 2` | GB300 每节点 4 卡，TP8 = 2 节点 |
| `--moe-backend auto` | 官方 recipe 用 auto；如需手动：TP>1 用 `flashinfer_trtllm`，DEP 用 `deep_gemm_mega_moe` |
| `--load-format fastsafetensors` | 1.4 TB 权重，普通 loader 太慢 |
| `--kv-cache-dtype fp8` | 压 KV，为 1M 上下文留空间 |
| `--max-model-len auto` | 让引擎按显存自算，别硬写 1048576 |
| `--attention-config` | MLA prefill 走 FlashInfer + prefill query 量化 |
| `--speculative-config` | DSpark，`num_speculative_tokens: 7` |
| `VLLM_ENGINE_READY_TIMEOUT_S=3600` | 2.8T 模型加载 + CUDA graph capture 很久 |

### 4.4 就绪判据 `[待验证]`

```bash
curl -s http://${HEAD_ADDR}:8000/health
# 别用 ss 判端口 —— 镜像里通常没装（V4 踩过）
```

> **[待验证]** 补充：启动日志里应出现的 MoE backend / KDA kernel 选择行，
> 用来判断「跑起来了」和「跑对了」——参考 V4 vLLM runbook 里认 DeepGEMM 那四行的做法。
> 所有健康信号全绿但性能腰斩是常态。

---

## 5. 压测 · 复现官方数字 `[官方]`

两条命令对应两个官方数字。

### 5.1 无投机解码基线（8K/1K random，bs=1）

```bash
vllm-bench \
  --backend openai \
  --base-url "http://${HEAD_ADDR}:8000" \
  --model moonshotai/Kimi-K3 \
  --dataset-name random \
  --random-input-len 8192 \
  --random-output-len 1024 \
  --random-range-ratio 0.8 \
  --prompt-token-ids \
  --ignore-eos \
  --sweep-max-concurrency 1 \
  --sweep-num-prompts-factor 10 \
  --seed 42 \
  --percentile-metrics "ttft,tpot,itl,e2el" \
  --metric-percentiles "50,90,99" \
  --save-result
```

**官方目标：TP8 111 tok/s/user，TP16 118 tok/s/user。**

### 5.2 投机解码（SPEED Bench，bs=1）

```bash
vllm-bench \
  --backend openai \
  --base-url "http://${HEAD_ADDR}:8000" \
  --model moonshotai/Kimi-K3 \
  --dataset-name speed-bench \
  --speed-bench-config throughput_16k \
  --speed-bench-max-input-len 10240 \
  --speed-bench-category low_entropy \
  --output-len 1536 \
  --num-prompts 10 \
  --no-oversample \
  --max-concurrency 1 \
  --temperature 1.0 \
  --top-p 0.95 \
  --save-result \
  --save-detailed
```

**官方目标：TP8 331 tok/s/user，TP16 370 tok/s/user（3.14×）。**

> ⚠️ **注意这里官方用的是 `--temperature 1.0`，不是 0。**
> 本仓库 V4 的教训是「压测不发 temperature=0，投机解码接受率崩到 1%」——
> 但那是因为 V4 的 draft 是 greedy 而服务端默认采样。
> K3 的 DSpark 用 `draft_sample_method: probabilistic`，与温度采样匹配，
> 所以官方才敢用 1.0。**换任何采样参数前先确认 draft 侧的采样方式**。

### 5.3 官方接受率参考

| 任务类型 | 每步接受 token 数 |
|---|---|
| 代码等低熵任务 | ~4.73 |
| 创作等高熵任务 | ~2.61 |

实测接受率明显低于这个区间，先查采样参数是否与 draft 匹配。

---

## 6. 扩展：TP16 与 PD 分离 `[官方]`

### 6.1 TP16（4 节点）

把 §4.2 的 `--tensor-parallel-size 16 --nnodes 4`，`--node-rank` 依次 0/1/2/3。
这是拿到 **370 tok/s** 的配置。

### 6.2 PD 分离（官方已验证拓扑）

**TEP8 prefill → DEP16 decode，NIXL 做 KV 传输。**

要点：
- prefill 用 **TEP**（attention TP + MoE EP），比纯 TP 通信更省，专家 GEMM 形状更好
- **序列并行在 TEP 下默认开启**，无需额外 flag（用 TP + MegaMoE，或 TP+DP+EP 时自动生效）
- NIXL connector 把共享 KV page 当**两个逻辑视图**：token 级 MLA cache + request 级 KDA 状态（含卷积与循环状态），握手时交换两套 metadata、分别建传输描述符
- 异构 TP 下 prefill / decode 块大小不同，connector 会做逻辑→物理块映射并把未传输的尾部清零，防止上一个请求的脏数据从 padding 漏出来

> V4 的教训在这里同样成立：**重启 prefill 必须同时重启 decode**（NIXL 句柄会失效）。

---

## 7. PD 分离：官方**没给**数，这是我们要自己测的部分 `[缺口 + 实验设计]`

### 7.1 先说清楚官方给了什么、没给什么

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

### 7.2 可借的方法论：vLLM 三天前的 GLM-5.2 PD 专文

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

### 7.3 建议的扫描方案（GB300，4 GPU/节点）

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

### 7.4 PD 启动命令骨架 `[GLM 模板 + K3 参数，未实测]`

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

### 7.5 压测注入方式 `[GLM 方法]`

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

### 7.6 必须盯的四组监控 `[GLM 经验]`

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

## 8. Agentic 场景：KDA 缓存保留策略 `[官方]`

K3 的 KDA 状态**不随序列长度增长**，但单个状态很大（约等于几千 token 的 MLA cache）。
不能每个 token 都存，于是 vLLM 给了两条互补策略：

**区间保留** —— 每隔固定 token 数打一个 checkpoint：

```bash
export VLLM_PREFIX_CACHE_RETENTION_INTERVAL=32768   # 例：每 32K 一个
export VLLM_PREFIX_CACHE_RETENTION_INTERVAL=0       # 只保留 prompt 结尾
```

`0` 表示关闭周期性 checkpoint、只保留 prompt 末尾状态 —— **多轮对话为主的负载用这个最划算**，
因为下一轮通常从重放上一轮 prompt 开始，prompt 末尾正是最可能被复用的位置（vLLM 会自动识别保留）。

**Marconi 式选择性保留** —— 规则是「第二次命中才缓存」：第一次说明这个前缀存在，
第二次才说明它真的被共享。一次性前缀不占缓存，热前缀自动晋升。适合系统提示词 /
仓库快照 / 工具定义这类**不落在 prompt 边界上**的共享前缀。

---

## 9. 故障速查 `[待验证]`

| 现象 | 先查 |
|---|---|
| prefix cache 命中率为 0 | 是否漏了 `--enable-prefix-caching`（K3 默认关） |
| `tool_calls` 返回空 | 官方已知问题，与 prompt 和运行相关。加 schema 校验 + 重试，或改用 structured tool calling |
| 评测分数偏低 | 先看是不是被截断（K3 思考很长），调大 `max_tokens` 和 reasoning effort |
| 投机解码加速不明显 | 查采样参数是否与 draft 的 `probabilistic` 匹配；查接受率是否远低于 4.73 / 2.61 |
| 视觉相关报错 / TP 除不尽 | ViT `head_size=12`，TP8 除不尽，确认 `--mm-encoder-tp-mode=data`（默认开） |
| 加载超时 | `VLLM_ENGINE_READY_TIMEOUT_S=3600` |
| 跨节点性能腰斩 | 两节点是否在同一 NVL72 subblock |
| RAID 只有 256K | RAID 没挂，见 V4 的 SETUP 文档 md0→md127 陷阱 |

---

## 10. 官方精度基线 `[官方]`

| GSM8K | GPQA-Diamond | OCRBench | MMMU Pro Vision |
|---|---|---|---|
| 0.976 | 0.939 | 0.889 | 0.818 |

均为 vLLM OpenAI 兼容端点、max reasoning effort 下实测。

---

## 11. 验证记录

> **[待填]** 本环境尚未跑过。每跑通一轮，按 V4 runbook §10 的格式记录：
> 轮次 / 日期 / 是否清空环境从零重跑 / 实测数字 / 与官方差多少 / 撞到的文档缺陷。

| 轮次 | 日期 | 配置 | 实测 tok/s (bs=1) | vs 官方 | 备注 |
|---|---|---|---|---|---|
| — | — | TP8 无投机 | — | /111 | 待跑 |
| — | — | TP8 + DSpark | — | /331 | 待跑 |
| — | — | TP16 + DSpark | — | /370 | 待跑 |

---

## 12. 官方 roadmap（会影响后续复现基线）

- **DCP（Decode Context Parallelism）** —— 原型显示选定负载下比 TP8 高 **40%** 吞吐，即将上游
- **EPLB** 性能改进
- **Confidence-based scheduling** —— 用 DSpark 的 confidence head 剪掉不会被接受的 draft token
- **RL 支持** —— vLLM rollout 已加，正在对接 RL 生态
- 更广的 AMD ROCm 调优

---

## 来源

- vLLM day-0 博客（本文主要来源）：<https://vllm.ai/blog/2026-07-27-k3>
- 架构与 kernel 预告：<https://vllm.ai/blog/2026-07-22-kimi-k3-preview>
- 官方 recipes / Docker 镜像：<https://recipes.vllm.ai/moonshotai/Kimi-K3>
- 模型卡：<https://huggingface.co/moonshotai/Kimi-K3>
- DSpark draft：<https://huggingface.co/Inferact/Kimi-K3-DSpark>
