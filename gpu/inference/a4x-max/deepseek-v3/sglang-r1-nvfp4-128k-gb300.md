# SGLang × DeepSeek-R1 NVFP4 — GB300 NVL72 长上下文大规模 EP 推理复现

> 复现对象：LMSYS 官方博客 [*Deploying DeepSeek on GB300 NVL72: Big Wins in Long-Context Inference*](https://lmsys.org/blog/2026-02-19-gb300-longctx/)（2026-02-19）
> 官方端到端复现 issue：[sgl-project/sglang#18703 — Instructions & Recipes for DeepSeek NVFP4 on GB300](https://github.com/sgl-project/sglang/issues/18703)
> 目标：在我们现有的 GB300 (A4X Max) NVL72 环境上，按本文一步步跑通 DeepSeek-R1-NVFP4 的 128K/8K 长上下文、PD 分离 + Wide-EP 推理。

---

## 0. 博客成果（我们要复现的目标数字）

| 指标 | GB300 | 对比 GB200 |
|------|-------|-----------|
| 长上下文(128K/8K)峰值吞吐 | **226.2 TPS/GPU** | 1.53× (GB200 147.9) |
| 开 MTP 后单用户吞吐 | TPS/User 23→43 (+87%) | 1.87× |
| 匹配延迟下 TPS/GPU | — | 1.4×–1.6× |
| 128K prefill TTFT | **8.6s**（32K dynamic chunking） | 1.07×–1.23× 更低 |
| decode 并发 (DEP16) | 36 req/GPU (576 并发) | GB200 20 req/GPU (320) |
| FMHA kernel | 205ms | 1.35× 快 (GB200 277ms) |
| 精度 (LongBench-v2) | 57.2% (non-MTP) / 56.9% (MTP) | 对齐官方 56.7% |

**用满的新 feature**：NVFP4 GEMM/dispatch、FP8 attention + FP8 KV cache、PD 分离、Wide-EP（EP≤32）+ DeepEP + DeepGEMM、MTP (Spec-V2 overlap scheduler)、chunked PP prefill + dynamic chunking、NVIDIA Dynamo 编排、Blackwell Ultra 硬件（2× SFU softmax、1.5× HBM 288GB、1.5× NVFP4）。

---

## 1. 需要多少卡？测试规模一览（重要）

配置命名规则：`ctxN`=N 个 prefill worker，`ctx_pp4`=每个 prefill 用 PP4（4 GPU），`gen1`=1 个 decode 实例，`depN`=decode expert-parallel = N GPU，`batchN`=decode batch，`mtpN`=MTP draft 层数。

| 实验档 | 配置 | Prefill | Decode | **总 GPU** | 节点数(4 GPU/节点) |
|--------|------|---------|--------|-----------|-------------------|
| Prefill TTFT (1P1D) | `ctx1_pp4_gen1_tp4` | PP4 = 4 | TP4 = 4 | **8** | 2 |
| Max-tput 小档 | `ctx3_pp4_gen1_dep8` | 3×PP4 = 12 | DEP8 = 8 | **20** | 5 |
| Mid-curve | `ctx5_pp4_gen1_dep16` | 5×PP4 = 20 | DEP16 = 16 | **36** | 9 |
| **Max-tput 大档** | `ctx8_pp4_gen1_dep32` | 8×PP4 = 32 | DEP32 = 32 | **64** | **16** |

### 结论：一个 NVL72 域就够

- **最大档 ctx8_dep32 = 64 GPU = 正好一个 NVL72 域（我们的一个 subblock = 16 节点 × 4 GPU）。**
- 小档（8/20/36 GPU）在一个域里绰绰有余。
- **所以单个 NVL72 域（16 节点）足够跑通整套 sweep**，不需要 4 个域。
- 我们有 4 个域 → 可以**并行跑 4 个不同规模/配置**，或留 1 个域跑推理、其余跑训练。
- PD 分离要求 prefill↔decode 之间高速传 KV cache；放在**同一个 NVL72 域内**走 NVLink/MNNVL 最优，别跨域。

> 建议起步：先用 **1P1D = 8 GPU（2 节点）** 跑通 prefill TTFT，验证环境；再上 **ctx3_dep8 = 20 GPU** 验证 PD + Wide-EP；最后 **ctx8_dep32 = 64 GPU** 冲峰值吞吐（占满一个域）。

---

## 2. 环境依赖（严格对齐官方复现）

| 组件 | 版本 / 来源 |
|------|------------|
| SGLang 代码 | 分支 [`YAMY1234/sglang@gb300_blog`](https://github.com/YAMY1234/sglang/tree/gb300_blog) @ commit `a046758`（优化已全部合入 main，固定 commit 为稳定复现） |
| 容器 | [`lmsysorg/sglang:v0.5.7-cu130-runtime`](https://hub.docker.com/layers/lmsysorg/sglang/v0.5.7-cu130-runtime/) |
| sgl-kernel | 需在容器内重装 |
| flashinfer | **≥ v0.6.1**（sm103 的 cutedsl 支持） |
| nvshmem | `nvshmem-cu13` 版本 **3.3.24** |
| DeepEP | 从 [`fzyzcjy/DeepEP`](https://github.com/fzyzcjy/DeepEP) 源码编译（构建命令见 [Dockerfile#L228](https://github.com/YAMY1234/sglang/blob/gb300_blog/docker/Dockerfile#L228)） |
| 模型 | **DeepSeek-R1-0528-NVFP4-v2**（FP4 量化 checkpoint，需从 HF 下载到本地） |
| 编排 | 官方用 **srt-slurm + NVIDIA Dynamo**（Slurm 系统）；我们 GKE 见第 4 节适配 |

> **容器最省事的路子：直接从 `YAMY1234/sglang@gb300_blog` 的 [Dockerfile](https://github.com/YAMY1234/sglang/blob/gb300_blog/docker/Dockerfile) build**（里面已把 DeepEP 源码编译、nvshmem 3.3.24、flashinfer≥0.6.1、sgl-kernel 全串好，见 L228）。不要指望 `v0.5.7-cu130-runtime` 开箱即用——它需要重装这些才能在 sm103 上跑 cutedsl。
>
> **模型下载**（~350GB，671B×0.5B/param FP4）— 真实 repo 已确认：
> ```bash
> huggingface-cli download nvidia/DeepSeek-R1-0528-NVFP4-v2 \
>   --local-dir /mnt/models/DeepSeek-R1-0528-NVFP4-v2
> # nvidia 官方 modelopt NVFP4 checkpoint, MIT license
> # 放共享存储（GCS gcsfuse / Lustre），所有 pod 挂载同一路径
> ```

> ### ✅ 本环境已验证的硬件事实（2026-07-18 实查 yw pod）
> - **GPU**：`NVIDIA GB300, compute_cap 10.3` = **sm_103a** → 必须 flashinfer≥0.6.1 的 cutedsl kernel ✓
> - **RDMA HCA**：`mlx5_0` ~ `mlx5_7`（8 张 CX-8）→ `--disaggregation-ib-device mlx5_0,mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7`
> - **RDMA 网口**：`gpu0ipvlan0/1` ~ `gpu3ipvlan0/1`（8 个 ipvlan，2/GPU）+ `eth0` 管理网

---

## 3. 官方复现路径（srt-slurm + Dynamo，Slurm 环境）

> 我们环境是 GKE 不是 Slurm，这一节仅作**权威参照**；实际在我们环境跑走第 4 节。

```bash
# 1. 拉复现分支
git clone -b gb300_blog https://github.com/YAMY1234/srt-slurm.git
cd srt-slurm
# 按 srt-slurm README 的 quick-start 装 srtctl

# 2. 配 srtslurm.yaml（SLURM 账号/分区/模型路径/容器）
#   default_partition: "gb300"
#   model_paths.dsfp4: "/path/to/DeepSeek-R1-0528-NVFP4-v2"
#   containers.v0.5.7-cu130: "/path/to/sglang-v0.5.7-cu130-runtime.sqsh"

# 3. 提交一个配置（示例：ctx8_dep32 最大档）
python -m srtctl.cli.submit apply \
  -f recipes/gb300-128k8k-blog/1-max-tput/gb300-maxthroughput-ctx8_ctx_pp4_gen1_dep32_batch8_eplb0_mtp0.yaml \
  --setup-script gb300-fp4-128k8k-setup.sh
```

- 全部实验配置：<https://github.com/YAMY1234/srt-slurm/tree/gb300_blog/recipes/gb300-128k8k-blog>
- 顶配置 PR：<https://github.com/ishandhanani/srt-slurm/pull/156/changes>
- 三个子目录：`1-max-tput` / `2-peak-capacity-vs-latency-constaints` / `3-prefill-latency`

---

## 4. 我们 GKE 环境的适配路径（直接起 SGLang PD server）

我们没有 Slurm。两条路：

- **路 A（推荐先试）**：跳过 srt-slurm 编排层，用我们训练同款的 **pod-0 SSH fanout** 直接在 pod 上起 SGLang 的 prefill / decode server，手动接 PD（bootstrap port + nixl transfer）。
- **路 B（生产级）**：部署 **NVIDIA Dynamo Kubernetes stack**（官方支持 GB200/GB300，带 inference-aware autoscaling + topology-aware 调度），由 Dynamo 做 KV-aware 路由和 PD 编排。

下面给出**从 recipe 提取的真实 SGLang server 参数**（以 `ctx3_dep8` 档为例，decode DEP8 + 3×PP4 prefill = 20 GPU）。

### 4.1 关键环境变量（prefill + decode 通用）

```bash
export MC_FORCE_MNNVL=1                 # 强制走 MNNVL（GB300 NVL72 域内）
export NCCL_MNNVL_ENABLE=1
export NCCL_CUMEM_ENABLE=1
export SGLANG_NVFP4_CKPT_FP8_GEMM_IN_ATTN=1
export SGLANG_PER_TOKEN_GROUP_QUANT_8BIT_V2=1
export SGLANG_MOONCAKE_CUSTOM_MEM_POOL=True
export SGLANG_DISABLE_TP_MEMORY_INBALANCE_CHECK=1
export SGLANG_FLASHINFER_FP4_GEMM_BACKEND=cutlass
export FLASHINFER_DISABLE_VERSION_CHECK=1
export FLASHINFER_WORKSPACE_BASE=/tmp/flashinfer-cache
export SGLANG_DG_CACHE_DIR=/configs/deepgemm-cache
export TORCH_DISTRIBUTED_DEFAULT_TIMEOUT=7200
# PD 心跳/超时放宽（长上下文 bootstrap 慢）
export SGLANG_DISAGGREGATION_HEARTBEAT_MAX_FAILURE=100000
export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=100000
export SGLANG_DISAGGREGATION_WAITING_TIMEOUT=100000
export SGLANG_HACK_SEQ_BOOTSTRAP_ROOM=1
# decode 端额外：
export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=1024
export SGLANG_MOE_NVFP4_DISPATCH=1      # NVFP4 dispatch，all-to-all 流量降 4×
export SGLANG_NVFP4_CKPT_FP8_NEXTN_MOE=1
```

### 4.2 Decode server 参数（DEP8：2 节点 × 4 GPU，data-parallel=8）

```
--model-path <DeepSeek-R1-0528-NVFP4-v2>
--served-model-name deepseek-ai/DeepSeek-R1
--quantization modelopt_fp4
--attention-backend trtllm_mla
--kv-cache-dtype fp8_e4m3
--context-length 136001
--disaggregation-mode decode
--disaggregation-transfer-backend nixl
--disaggregation-bootstrap-port 30001
--enable-dp-attention --enable-dp-lm-head --enable-symm-mem
--data-parallel-size 8
--tensor-parallel-size 8
--expert-parallel-size 8
--pipeline-parallel-size 1
--moe-a2a-backend deepep --deepep-mode low_latency
--moe-runner-backend flashinfer_cutedsl
--moe-dense-tp-size 1 --disable-shared-experts-fusion
--ep-dispatch-algorithm static --ep-num-redundant-experts 32 --eplb-algorithm deepseek
--cuda-graph-max-bs 256 --max-running-requests 256
--mem-fraction-static 0.80
--chunked-prefill-size -1 --disable-radix-cache
--prefill-round-robin-balance
--scheduler-recv-interval 1 --stream-interval 10
--trust-remote-code --watchdog-timeout 1000000
```

### 4.3 Prefill server 参数（每 worker PP4：1 节点 × 4 GPU，3 个 worker）

```
--model-path <DeepSeek-R1-0528-NVFP4-v2>
--served-model-name deepseek-ai/DeepSeek-R1
--quantization modelopt_fp4
--attention-backend trtllm_mla
--kv-cache-dtype fp8_e4m3
--context-length 136001
--disaggregation-mode prefill
--disaggregation-transfer-backend nixl
--disaggregation-bootstrap-port 30001
--data-parallel-size 1
--tensor-parallel-size 1
--expert-parallel-size 1
--pipeline-parallel-size 4                # chunked PP prefill
--moe-runner-backend flashinfer_trtllm
--moe-dense-tp-size 1
--load-balance-method round_robin
--max-running-requests 32
--mem-fraction-static 0.72
--chunked-prefill-size -1 --disable-radix-cache
--scheduler-recv-interval 1 --stream-interval 10
--trust-remote-code --watchdog-timeout 1000000
```

> `ctx8_dep32` 大档：decode 改 `--data-parallel-size 32 --tensor-parallel-size 32 --expert-parallel-size 32`（DEP32 = 8 节点 × 4），prefill 起 8 个 PP4 worker（8 节点）。开 MTP 的档加 `--speculative-algorithm` 相关 MTP 参数（见 recipe `*_mtp2.yaml`）。

### 4.4 多节点启动 + PD 路由（关键，dry-run 走查补充）

> 这一节是我第一版漏掉、但**没它根本跑不起来**的核心机制。

**(a) 多节点 SGLang server**：任何跨 >1 节点的 server（DEP8=2 节点、DEP16=4、DEP32=8）都要在**每个节点**上起一个进程，靠这三个参数组成一个分布式 server：

```
--nnodes <N>                 # 该 server 占几个节点 (DEP8→2, DEP16→4, DEP32→8)
--node-rank <r>              # 本节点在该 server 内的 rank (0..N-1)
--dist-init-addr <head_ip>:<port>   # 该 server rank0 的地址
```
（prefill 每个 worker 是 PP4=1 节点 4 GPU，单节点即可，不需要 --nnodes；decode 才跨节点。）

**(b) PD 路由（prefill↔decode 怎么连）**：不是"对齐 bootstrap-port"就完事。三种方式：
- **SGLang 原生 router / Model Gateway**：在前面起一个 router，把请求按策略分发到 prefill 池和 decode 池。prefill server 加 `--disaggregation-mode prefill`，decode 加 `--disaggregation-mode decode`，router 知道两个池的地址。
- **NVIDIA Dynamo**（博客用的）：KV-aware router + 生命周期管理，有 K8s stack。
- 传输后端：recipe 用 `nixl`（`--disaggregation-transfer-backend nixl`）；也可用 Mooncake。

**(c) KV 传输后端 — 我们单域部署可优于官方 recipe**：

官方 recipe 用 `nixl`（因为 Slurm 跨机架）。但**我们所有配置都塞在一个 NVL72 域内**（最大 ctx8_dep32=64 GPU=一个 subblock），所以 KV cache 可以直接走 **NVLink（Mooncake NVLINK pool）**，比 RoCE 更快更省事：
```
# 方案 1（推荐，单域内）：Mooncake + NVLink KV 传输
--disaggregation-transfer-backend mooncake
export SGLANG_MOONCAKE_CUSTOM_MEM_POOL=NVLINK   # NVL72 域内 KV 走 NVLink

# 方案 2（对齐官方 / 跨域时）：nixl + CX-8 RDMA
--disaggregation-transfer-backend nixl
--disaggregation-ib-device mlx5_0,mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7  # 已实查
```
> 先试方案 1（NVLink），单域内 KV 传输不出域、绕开 RoCE/UCX 在 GKE 上的调参。跑不通再退方案 2。

### 4.5 MTP（EAGLE spec decode，`*_mtp2` 档的真实参数）

```
--speculative-algorithm EAGLE
--speculative-num-steps 2
--speculative-eagle-topk 1
--speculative-num-draft-tokens 3
--speculative-moe-runner-backend deep_gemm
--speculative-moe-a2a-backend deepep
# 配套 env:
export SGLANG_NVFP4_CKPT_FP8_NEXTN_MOE=1
```

### 4.6 Benchmark（长上下文）

官方用 SemiAnalysis 的 `sa-bench`（可能非公开）。**我们可用 SGLang 自带 `bench_serving` 替代**：

```bash
python -m sglang.bench_serving \
  --backend sglang --host <router_ip> --port <port> \
  --dataset-name random \
  --random-input-len 128000 --random-output-len 8000 \
  --max-concurrency 512 --num-prompts 512
```
对标 ISL=128000 / OSL=8000 / concurrency=512，看 TPS/GPU 和 TTFT。

---

## 5. GB300-on-GKE 适配要点（我们环境专属）

1. **MNNVL / NVLink 域**：`MC_FORCE_MNNVL=1 NCCL_MNNVL_ENABLE=1`，且所有 EP rank 必须在**同一个 subblock**（一个 NVL72 域），跟训练时 EP 落域内的逻辑一致。
2. **RDMA / DRA 已就绪**：我们 GB300 GKE 的 GIB + DRA + asapd-lite 已跑通（见 `../../a4x-max/12-self-managed-k8s/`），跨节点 KV 传输（nixl / Mooncake）走 CX-8 RDMA。
3. **模型下载**：DeepSeek-R1-0528-NVFP4-v2 从 HF 拉到共享存储（Lustre / GCS），~400GB FP4 权重，pod 挂载。
4. **容器**：用 `lmsysorg/sglang:v0.5.7-cu130-runtime`，进容器重装 sgl-kernel + flashinfer≥0.6.1 + DeepEP(fzyzcjy) + nvshmem-cu13 3.3.24；或自己 bake 一个（对齐 `YAMY1234/sglang@gb300_blog` 的 Dockerfile）。
5. **PD 编排缺口（最大适配工作量）**：srt-slurm 是 Slurm 的。GKE 上三选一，按轻→重：
   - **① SGLang 原生 router + SSH fanout（推荐先试）**：用我们训练同款 pod-0 SSH fanout 起 prefill/decode server，前面挂一个 `sglang_router`（Model Gateway）做 PD 分发。最轻、复用已验证的 fanout 基建。
   - **② NVIDIA Dynamo K8s stack**：跟 GKE 最契合、生产级（KV-aware 路由 + autoscaling），但要新起一套 Dynamo。
   - ③ 手动对齐 bootstrap-port 无 router：只适合 1P1D 冒烟，不可规模化。
6. **共享存储前置**：GB300 训练用的是 mock data，没配模型存储。350GB NVFP4 权重需要共享盘 —— 我们已有 **GCS gcsfuse**（CC Pages 在用）可直接挂，或配 Managed Lustre。**这是开跑前必须先解决的**。
7. **cutedsl 需 sm103**：已实查 GB300 = compute_cap 10.3 = sm103a，flashinfer≥0.6.1 才有 cutedsl fp4 kernel，务必确认版本。

---

## 6. 复现步骤（GKE，路 A：直接起 SGLang PD，从 1P1D 起步）

1. **准备**：一个 NVL72 域打标签（16 节点足够，1P1D 只需 2 节点）；建 SSH-enabled pod 池（同训练用的 `yw-pool` 模式，但换 SGLang 容器）；模型下载到共享存储并挂载（`/mnt/models/...`）；deepgemm-cache / flashinfer-workspace 挂可写卷。
2. **构建/装依赖**（不是"验证"）：从 `gb300_blog` Dockerfile build 容器（含 DeepEP 源码编译 + nvshmem 3.3.24 + flashinfer≥0.6.1 + sgl-kernel）；进 pod 确认 `import deep_ep` OK、`flashinfer.__version__>=0.6.1`、GPU 是 **sm103a**（`nvidia-smi --query-gpu=compute_cap`）。
3. **起 1P1D（8 GPU）冒烟**：节点 A 起 prefill server（4.3 参数，PP4=单节点 4 GPU）；节点 B 起 decode server（4.2 参数改小档 TP4/DP1/EP1，单节点）；前面起一个 **SGLang router**（4.4b）把两者接起来；套 4.1 env + `--disaggregation-transfer-backend nixl` + `--disaggregation-ib-device`（4.4c）。
4. **发一条 128K 请求**（打 router 地址）验证 PD 通路（prefill→KV 传→decode 出 token）。
5. **上 ctx3_dep8（20 GPU）**：3 个 prefill worker（各 PP4 单节点）+ decode DEP8（**2 节点，用 4.4a 的 `--nnodes 2 --node-rank 0/1 --dist-init-addr`**）+ router；跑 4.6 的 `bench_serving` ISL=128K/OSL=8K，看 TPS/GPU。
6. **冲 ctx8_dep32（64 GPU，占满一个域）**：8 prefill + decode DEP32（**8 节点分布式 server**）+ router；对比博客 226 TPS/GPU。
7. **开 MTP**（4.5 的 EAGLE 参数）看 TPS/User 提升（博客 23→43）。

> 每档跑通后把实测 TPS/GPU、TTFT 填进本文档，跟博客数字对比。

---

## 6.5 Dry-run 走查发现（第一版 → 修订）

对照 LMSYS 博客 + issue 18703 + SGLang 官方 PD 文档，逐步"假装执行"抓出的坑：

| # | 第一版的问题 | 修订 |
|---|-------------|------|
| 1 | 依赖写成"验证"，其实容器不自带 | 改为**从 gb300_blog Dockerfile 构建**（DeepEP 源码编译等），见第 2 节 |
| 2 | **完全没写多节点启动** | 补 4.4a：`--nnodes/--node-rank/--dist-init-addr`（DEP8=2/16=4/32=8 节点） |
| 3 | PD 只说"对齐 bootstrap-port" | 补 4.4b：**必须有 router**（SGLang Model Gateway 或 Dynamo）在前面分发 |
| 4 | 没提 RDMA HCA / NVL72 KV 传输 | 补 4.4c：`--disaggregation-ib-device` + NVLink KV pool env |
| 5 | 模型下载无命令、大小写 400GB | 补 `huggingface-cli download`，改 ~350GB（671B×0.5） |
| 6 | benchmark 只写 sa-bench（非公开） | 补 4.6：**用 SGLang 自带 `bench_serving`** 替代 |
| 7 | MTP 参数含糊 | 补 4.5：确切 EAGLE 参数（num-steps 2 / topk 1 / draft-tokens 3） |
| 8 | 没提可写卷挂载 | 补 deepgemm-cache / flashinfer-workspace / 模型路径挂载 |

**仍待实测确认的不确定点**（Round 2/3 后收敛）：
- ~~HCA 命名~~ → **已实查解决**：`mlx5_0`~`mlx5_7`。
- **KV 传输走 NVLink(Mooncake) 还是 RoCE(nixl)**：单域内理论上 NVLink 更优，但 SGLang Mooncake NVLINK pool 在 GB300 GKE 上是否即插即用待验；跑不通退 nixl。
- `bench_serving` 的 random dataset 能否代表 sa-bench 的 128K 长上下文分布（结果对比时要注明差异）。
- DEP32 分布式 decode 跨 8 节点时，MNNVL 域内 all-to-all 是否需额外对齐（类比训练侧 hybridep `NUM_OF_HYBRID_EP_RANKS` 的坑）。
- SGLang 原生 router 的 PD 分发在多 prefill worker（ctx3/5/8）下的负载均衡行为待观察。

## 7. 状态

**前置（开跑前必须）**
- [ ] 共享存储就位（GCS gcsfuse / Lustre）+ 下载 `nvidia/DeepSeek-R1-0528-NVFP4-v2`（~350GB）
- [ ] 从 `gb300_blog` Dockerfile build SGLang 容器（DeepEP/nvshmem 3.3.24/flashinfer≥0.6.1）
- [ ] 建 SGLang SSH-enabled pod 池（yw-pool 模式换容器）

**测试**
- [ ] 1P1D (8 GPU) 冒烟：PD 通路通 + 一条 128K 请求出 token
- [ ] ctx3_dep8 (20 GPU)：PD + Wide-EP + router，bench_serving 128K/8K
- [ ] ctx8_dep32 (64 GPU，占满一个域)：峰值吞吐 vs 博客 226 TPS/GPU
- [ ] MTP 档（EAGLE）：TPS/User vs 博客 23→43

*（待测，跑通后回填实测数字）*

*（待测，跑通后回填实测数字）*
