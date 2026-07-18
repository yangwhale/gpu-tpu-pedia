# DeepSeek V3 (671B) 256 GPU 训练 — GB300 NVL72 (A4X Max) GKE

GB300 (A4X Max) GKE 集群上的 DeepSeek V3 (671B, 61 层) 256 GPU 预训练 benchmark，基于 NVIDIA Megatron Bridge r0.5.0 官方 recipe。

**核心成果**：2026-07-17 首次在 GB300 256 GPU 上跑通 **full_iteration CUDA graph**，GBS=4096 达到 **~1618 MODEL TFLOP/s/GPU**（对标官方 1648，98.2%）。

> Qwen3 235B-A22B 的 GKE 训练文档见同级 `07d-gb300-qwen3-235b-gke/`（测试中）。

**参考**：
- [NVIDIA Megatron Bridge Performance Summary](https://docs.nvidia.com/nemo/megatron-bridge/latest/performance-summary.html)
- [AI-Hypercomputer/gpu-recipes](https://github.com/AI-Hypercomputer/gpu-recipes)
- Bridge 版本：r0.5.0（NeMo 容器 `nemo-gb300-ready:26.06-v1` 内置）

---

## 成果总览

| 模型 | 规模 | 精度 | Graph | GBS | 稳态 TFLOP/s/GPU | Step Time | 状态 |
|------|------|------|-------|-----|-----------------|-----------|------|
| DeepSeek V3 (61L, 671B) | 256 GPU (4×16) | FP8_MX | **full_iteration** | 2048 (V1) | ~1553 | 5.47s | 通过 (30 步) |
| DeepSeek V3 (61L, 671B) | 256 GPU (4×16) | FP8_MX | **full_iteration** | 4096 (V2) | **~1618** | 10.53s | 通过 (30 步) |
| DeepSeek V3 (61L, 671B) | 256 GPU (4×16) | FP8_MX | **full_iteration** | **15360** (`--global_batch_size`) | **~1658** | 38.5s | 通过 (30 步) |
| DeepSeek V3 **scale-in (31L)** | **128 GPU (4×8)** | FP8_MX | **full_iteration** | 7680 | **~1550**（全规模 93.5%） | 22.2s | 通过 (30 步) |
| Qwen3 235B-A22B | 256 GPU (4×16) | FP8_MX | full_iteration | 8192 (V2) | 待测 | — | 计划中 |

### 对标 NVIDIA 官方 (DeepSeek V3 256 GPU MXFP8)

| System | GBS | TP/PP/CP/VP/EP | Tokens/s/GPU | Model TFLOP/s/GPU |
|--------|-----|---------------|-------------|-------------------|
| **DGX-GB300** | 4096 | 1/2/1/8/32 | 6338 | **1648** |
| DGX-GB300 | 15360 | 1/2/1/8/32 | 6422 | 1670 |
| DGX-GB200 | 4096 | 1/4/1/4/64 | 4969 | 1292 |
| DGX-B300 | 4096 | 1/8/1/n-a/8 | 3541 | 920 |

> 我们 GBS=4096 实测 **1618～1620** vs 官方 **1648** = **~98.3%**。并行配置 (TP=1/PP=2/VP=8/EP=32) 与官方完全一致。
>
> **vboost 实验结论（2026-07-18 复测）**：单独开 `nvidia-smi boost-slider --vboost 1` 后稳态仍是 **~1615-1620**（best 1620，typical 1615），**没有拉近到 1648**。说明 GBS=4096 档剩下 ~2% 差距**不是 vboost 造成的**。vboost 对本 recipe 无明显增益，可不必单独设。
>
> **GBS=15360 实验结论（2026-07-18，对标官方第二档 1670）**：把 `--global_batch_size 15360`（CLI 覆盖 v2 的 4096）后，稳态冲到 **~1658（best 1659）= 官方 1670 的 99.3%**，step time 38.5s，60 microbatch，无 OOM。**大 GBS 摊薄 pipeline bubble 的收益实打实**：1618(GBS4096) → 1658(GBS15360)，+2.5%。这是目前最接近官方的配置。
> - 注意：`4096 / 15360` 是 **global batch size**，不是 sequence length（seq_length 两档都固定）。`--global_batch_size` 优先级高于 `-cv` 版本（`utils.py:186`）。
> - GBS 必须能被 `MBS × DP` 整除：15360 / (2 × 128) = 60 microbatch（整除）。

---

## 集群架构：4-domain 256 GPU

GB300 一个 NVLink Domain (NVL72) = 一个 subblock ≤ 18 台机器。256 GPU = 64 节点，必须**跨 4 个 subblock**。

**关键**：每个 subblock 用一个**独立的 ComputeDomain**（IMEX 通道只能在单个 NVLink domain 内建立）。绝不能用一个 ComputeDomain 横跨 4 个 subblock。

```
256 GPU = 4 domain × 16 节点 × 4 GPU
┌─────────────┬─────────────┬─────────────┬─────────────┐
│ subblock A  │ subblock B  │ subblock C  │ subblock D  │
│ CD: yw-cd-a │ CD: yw-cd-b │ CD: yw-cd-c │ CD: yw-cd-d │
│ 16 node     │ 16 node     │ 16 node     │ 16 node     │
│ NVLink(MNNVL)│ NVLink     │ NVLink      │ NVLink      │
└──────┬──────┴──────┬──────┴──────┬──────┴──────┬──────┘
       └── CX-8 RDMA 跨 domain 互联 (PP / DP) ──┘
```

- **组内 (16 节点 64 GPU)**：NVLink，走 ComputeDomain IMEX，承载 EP=32 的 all-to-all
- **组间 (4 domain)**：CX-8 SuperNIC RDMA，承载 PP / DP 通信

---

## Part 1 — 完整可跑通步骤 (DeepSeek V3, 1553 TFLOP/s)

集群 `gb300-gke-test` (Regional us-central1)。所有 `kubectl` 在 gLinux 上执行（context = `gke_tencent-gcp-taiji-poc_us-central1_gb300-gke-test`）。

### Step 0 — 确认 4 个 subblock 各有 16 节点

```bash
# 每个 GB300 node pool = 一个 subblock。查每个 subblock 的节点数
kubectl get nodes -L cloud.google.com/reservation-subblocks -l team=yangwhale --no-headers \
  | awk '{print $NF}' | sort | uniq -c
```

若某 subblock 不足 16 节点，给同 subblock 的空闲节点补标签：

```bash
kubectl label node <NODE_NAME> team=yangwhale --overwrite
```

### Step 1 — 创建 4-CD sleep-infinity pod 池

用 sleep-infinity 常驻池，训练通过 `kubectl exec` 启动，避免每次重建 pod 的开销。

关键点（每组 a/b/c/d 一份，钉在不同 node pool = 不同 subblock）：
- `ComputeDomain` `numNodes: 0`（动态发现节点），每组独立
- `podAffinity` on `cloud.google.com/gce-topology-subblock`：保证同组 16 pod 落同一 subblock
- `podAntiAffinity` on `kubernetes.io/hostname` (job=yw)：每节点仅 1 pod
- MRDMA `count: 8`（CX-8 8 PF）
- 共享一个 headless Service `yw`

完整 YAML 见 `yw-pool-256.yaml`（同目录）。应用：

```bash
kubectl apply -f yw-pool-256.yaml
# 等 64 pod Running（新标签节点首次拉镜像 ~2-3 min）
kubectl get pods -l job=yw --no-headers | grep -c Running   # 应为 64
kubectl get computedomains | grep yw                        # 4 个 Ready
```

### Step 2 — 准备启动脚本（严格对齐 Bridge）

每个 pod 跑 `run-dsv3-yw.sh`（见同目录）。它做三件事：
1. 从 hostname (`yw-{a,b,c,d}-N`) 算全局 `node_rank` 0-63
2. **导出完整 env**（见 Part 2，缺一不可）
3. torchrun 起 native recipe（**不覆盖 cuda_graph_impl**）

训练命令（内层 per-local-rank worker）：

```bash
numactl --cpunodebind=$((LOCAL_RANK/2)) --membind=$((LOCAL_RANK/2)) \
python scripts/performance/run_script.py \
  -m deepseek -mr deepseek_v3 --task pretrain \
  -g gb300 -c fp8_mx -ng 256 \
  --data mock --max_steps 30 \
  --log_dir /tmp/nemo-results \
  -cv v1 \
  logger.log_throughput=true
```

> **绝不加** `model.cuda_graph_impl=...` / `model.cuda_graph_scope=...` 覆盖。让 recipe 用原生 `full_iteration` + 自动启用 `moe_paged_stash`。

### Step 3 — 分发脚本 + 启动

torchrun 直跑模式下 run_script.py **不会**自动设 perf env（那是 nemo-run launcher 侧 plugin 的活），所以 env 必须在脚本里手动 export（Part 2）。

```bash
# 分发（base64 避免 stdin 管道产生 symlink 的坑）
B64=$(base64 -w0 run-dsv3-yw.sh)
for g in a b c d; do for i in $(seq 0 15); do echo yw-$g-$i; done; done \
  | xargs -P 16 -I {} kubectl exec {} -- bash -c \
    "echo $B64 | base64 -d > /tmp/run-dsv3-yw.sh && chmod +x /tmp/run-dsv3-yw.sh"

# 全部 64 pod 并行启动
for g in a b c d; do for i in $(seq 0 15); do echo yw-$g-$i; done; done \
  | xargs -P 32 -I {} kubectl exec {} -- bash -c \
    "nohup /tmp/run-dsv3-yw.sh > /tmp/dsv3-run.log 2>&1 &"
```

### Step 4 — 监控

```bash
kubectl exec yw-a-0 -- bash -c 'grep "Step Time" /tmp/dsv3-run.log | tail -6'
# 期望：Step Time : 5.4Xs GPU utilization: 155X.X MODEL_TFLOP/s/GPU
```

时间线（首次运行）：rendezvous → megatron init (~38s) → optimizer init → warmup 3 步 → **graph capture** → 稳态 30 步。每 5 步一次 manual GC，step time 从 5.47s 短暂升到 6.1s（正常）。

---

## Part 2 — 关键环境变量（full graph 能跑通的根本原因）

以下 env 由 Bridge 的 `perf_plugins.py` 在 nemo-run launcher 侧设置。**torchrun 直跑时不会自动生效，必须手动 export**。历史上一直失败就是漏了其中的 full-graph 专属项。

来源：`scripts/performance/utils/executors.py` (`PERF_ENV_VARS` base) + `scripts/performance/perf_plugins.py`（增量）。

### full_iteration graph 专属（历史遗漏的关键 2 项）

| env | 值 | 作用 |
|-----|-----|------|
| `PYTORCH_CUDA_ALLOC_CONF` | `expandable_segments:True,graph_capture_record_stream_reuse:True` | 允许 graph capture 期间复用 record stream，NCCL 通信 stream 能并入 capture DAG |
| `TORCH_NCCL_AVOID_RECORD_STREAMS` | `0` | full graph 下必须 0（base 默认 1），配合上一项解决 `StreamCaptureUnjoined` |

> 这两项是 `perf_plugins.py:302-306` 在 `cuda_graph_impl == "full_iteration"` 时才追加的。之前手写脚本没有 → graph capture 崩 `cudaErrorStreamCaptureUnjoined`。

### HybridEP NVL domain (EP=32, GB300 NVL72)

| env | 值 | 说明 |
|-----|-----|------|
| `NVLINK_DOMAIN_SIZE` | `72` | GB300 NVL72 |
| `USE_MNNVL` | `1` | 多节点 NVLink |
| `NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN` | `32` | **== EP size**（不是 8！`perf_plugins.py:383` 对 GB200/GB300 直接 `= ep_size`） |
| `NUM_OF_TOKENS_PER_CHUNK_COMBINE_API` | `128` | unfused combine 性能 workaround |

### CUDA / LayerNorm / cutedsl / deepseek

| env | 值 | 条件 |
|-----|-----|------|
| `CUDA_DEVICE_MAX_CONNECTIONS` | `32` | hybridep + sm100+ |
| `NVTE_FWD_LAYERNORM_SM_MARGIN` | `20` | hybridep（非 hybridep 是 16） |
| `NVTE_BWD_LAYERNORM_SM_MARGIN` | `20` | 同上 |
| `NVTE_CUTEDSL_FUSED_GROUPED_MLP` | `1` | cutedsl_fused_grouped_mlp=True |
| `CUDNNFE_CLUSTER_OVERLAP_MARGIN` | `8` | cutedsl + moe_a2a_overlap |
| `NVTE_ALLOW_NONDETERMINISTIC_ALGO` | `0` | deepseek |
| `NVTE_NORM_FWD_USE_CUDNN` | `1` | deepseek fp8_mx gb300 保留 cudnn LN（del_cudnn_ln=False） |
| `NVTE_NORM_BWD_USE_CUDNN` | `1` | 同上 |

### base NCCL / GIB (GKE GB300)

| env | 值 |
|-----|-----|
| `NCCL_GRAPH_REGISTER` | `0`（=1 在 GB300 GIB 下导致 rendezvous 挂死，见坑 G） |
| `NCCL_CONF_FILE` | `/usr/local/gib/configs/nccl.a4xmax.conf` |
| `NCCL_NVLS_ENABLE` | `0`（省显存） |
| `NCCL_IB_SPLIT_DATA_ON_QPS` | `1` |
| `NCCL_CTA_POLICY` | `1` |
| `TORCH_NCCL_HIGH_PRIORITY` | `1` |
| `GLOO_SOCKET_IFNAME` / `NCCL_SOCKET_IFNAME` | `eth0` |
| `HF_HUB_DISABLE_XET` | `1` |

---

## Part 3 — 踩过的坑（试过不行的）

| # | 配置 | 卡数 | 结果 | 根因 |
|---|------|------|------|------|
| A | PP=1 EP=64 full graph | 64 | **OOM** | 无 PP 切分，61 层 + graph buffer 放不下 288GB |
| B | PP=2 VP=1 full graph | 64 | **crash** | VP=1 → overlap_p2p 强制 False → batch_p2p_sync 调 `synchronize()` → `StreamCaptureUnsupported` |
| C | PP=2 VP=12 full graph | 64 | **crash** | 缺 full-graph env（见 Part 2）→ NCCL p2p stream 未并入 capture → `StreamCaptureUnjoined` |
| D | PP=4 VP=12 full graph | 64 | **OOM** | warmup 3 步 OK，第 4 步 capture 时显存不够 |
| E | PP=4 VP=12, HYBRID_EP=32 | 256 | **hang** | 借了 GB200 的 VPP，且 4-domain 布局不对 |
| F | PP=2 VP=8, 手动覆盖 TE graph | 256 | **crash (optimizer init)** | 覆盖成 TE graph 绕过 full_iteration+paged_stash；且个别节点瞬时网络断连 |
| G | full graph + `NCCL_GRAPH_REGISTER=1` | 256 | **hang** | GB300 GIB 环境下 =1 导致 torchrun rendezvous 25min 无 worker |
| H | 借 GB200 VPP=3 到 GB300 | 256 | — | GB300 原生 recipe 不设这些，不要跨机型借参数 |

**共性教训**：
1. **不要手动覆盖 `cuda_graph_impl`** — 覆盖成 TE graph 会绕过 `moe_paged_stash` / `moe_expert_rank_capacity_factor` 等 full graph 必需机制（`set_full_iter_cg_configs`）。
2. **不要跨机型借参数** — GB200 的 VPP=3 不适用 GB300；照搬 GB300 原生 recipe。
3. **不要漏 perf env** — run_script.py torchrun 直跑不设 env，必须手动补全 Part 2 全部变量。
4. **HYBRID_EP_RANKS = EP size**，不是固定 8（GB200/GB300 NVL72 场景）。
5. **ComputeDomain 每 subblock 一个**，不横跨。

---

## Part 4 — Bridge 原生 recipe 参数参考

### DeepSeek V3 GB300 FP8_MX (256 GPU) — 已验证 1553 TFLOP/s

来源：`scripts/performance/configs/deepseek/deepseek_workload_base_configs.py` `DEEPSEEK_V3_PRETRAIN_CONFIG_GB300_FP8_MX_V1`

| 参数 | 值 |
|------|-----|
| num_layers | 61 (full 671B) |
| TP / PP / VPP / EP | 1 / 2 / 8 / 32 |
| pp_layout | `Et*4\|(t*4\|)*14tmL` |
| MBS / GBS | 1 / **4096 (V2, 对标官方 1648)**；V1 = 2048 |
| cuda_graph_impl | `full_iteration` |
| moe_a2a_overlap | True |
| moe_flex_dispatcher_backend | hybridep |
| cutedsl_fused_grouped_mlp | True |
| fp8_dot_product_attention | True |
| moe_paged_stash | True（full_iteration 自动启用） |
| mtp_num_layers | 1 |

### Qwen3 235B-A22B GB300 FP8_MX (256 GPU)

来源：`scripts/performance/configs/qwen/qwen3_workload_base_configs.py` `QWEN3_235B_A22B_PRETRAIN_CONFIG_GB300_FP8_MX_V2`

| 参数 | 值 |
|------|-----|
| TP / PP / VPP / EP | 1 / 4 / None / 32 |
| MBS / GBS | 2 / 8192 |
| cuda_graph_impl | `full_iteration` |
| moe_a2a_overlap | True |
| cutedsl_fused_grouped_mlp | True |
| fp8_dot_product_attention | True |

> Qwen3 V2 用 PP=4 + VP=None + full_iteration。启动命令把 `-m deepseek -mr deepseek_v3` 换成 `-m qwen -mr qwen3_235b_a22b`，`-cv v2`，其余 env 同 Part 2。

---

## Part 5 — 参数调优记录（DeepSeek V3, 2026-07-18）

### 已扫过的配置 → 实测吞吐

| 配置 | GBS | MBS | microbatch | step time | 稳态 TFLOP/s | 对标官方 |
|------|-----|-----|-----------|-----------|-------------|---------|
| `-cv v1` | 2048 | 1 | 8 | 5.47s | ~1553 | — |
| `-cv v2` | 4096 | 2 | 16 | 10.53s | **~1618** | 1648 的 98.2% |
| `-cv v2 --global_batch_size 15360` | 15360 | 2 | 60 | 38.5s | **~1658 (best 1659)** | **1670 的 99.3%** |

**结论：GBS 是目前收益最大的旋钮。** GBS 越大 → microbatch 越多 → pipeline bubble 摊得越薄 → MFU 越高。4096→15360 拿到 +2.5%（1618→1658），基本追平官方。

### 可调旋钮清单（按预期收益排序）

| 旋钮 | 怎么调 | 预期 | 已测? |
|------|--------|------|-------|
| **GBS** | `--global_batch_size N`（须被 MBS×DP 整除，DP=128） | 越大越接近官方，15360 已达 99.3% | ✅ 1658 |
| GPU 锁频 | 官方 perf plugin `_set_lock_gpu_freq`（本 recipe 未开） | 稳定态测量再稳一点，可能 +0.x% | ❌ 待测 |
| MBS | `--micro_batch_size N`（1/2/4） | 内存↔气泡 tradeoff，2 已较优 | 部分 |
| vboost | `nvidia-smi boost-slider --vboost 1` | 本 recipe **无明显增益**（实测 1615-1620 没变） | ✅ 无效 |
| seq_length | `--seq_length N`（默认 4096） | 改的是**工作负载**（长上下文），非官方 1670 那档，attention 平方级涨 | ❌ 另类测试 |

### 关键澄清

- 官方对标表的 `4096 / 15360` 是 **global batch size**，**不是 sequence length**。seq_length 两档都固定 4096。
- `--global_batch_size` CLI 优先级高于 `-cv` 版本内置的 GBS（`utils.py:186`），可直接覆盖。
- **GBS 必须能被 `MBS × DP` 整除**：DSV3 256GPU 下 DP = 256/(TP1×PP2) = 128，MBS=1（FP8_MX）→ 15360/128=120 microbatch ✓；若报错先检查整除性。

---

## Part 6 — Scale-in 测试：小规模能否复现 MFU（2026-07-18）

**目的**：层数 + 节点等比例减半，验证能否在少量机器上复现全规模的 TFLOP/s，用于**低成本调试**（先在小集群验证想法，再上全规模）。

### 配置对比（严格等比例减半）

| 维度 | 全规模 | Scale-in |
|------|--------|----------|
| GPU | 256 (4×16) | **128 (4×8)** |
| 层数 | 61 (3 dense+58 MoE+1 MTP) | **31 (3 dense+28 MoE+1 MTP)** |
| pp_layout | `Et*4\|(t*4\|)*14tmL` | `Et*2\|(t*2\|)*14tmL` |
| GBS | 15360 | **7680** |
| TP/PP/VPP/EP | 1/2/8/32 | 1/2/8/32（**不变**）|
| microbatch | 120 | 120（**不变**）|
| 稳态 TFLOP/s | 1658 | **~1550 (best 1551)** |

> Scale-in 命令在 15360 版基础上加：`-ng 128 --global_batch_size 7680 --num_layers 31 --first_k_dense_replace 3 --pipeline_model_parallel_layout "Et*2|(t*2|)*14tmL" -tp 1 -pp 2 -vp 8 -ep 32`；torchrun `--nnodes=32`，node_rank offset 改 a=0/b=8/c=16/d=24。集群侧 `kubectl scale statefulset yw-{a,b,c,d} --replicas=8`。

### 结论

- **复现了 ~93.5% 的 MFU**（1550 / 1658），**不是 100%**。
- **差 ~6.5% 的原因**：层数减半后每 pipeline stage 只剩 2 层（全规模 4 层/stage），但 **embedding / MTP / EP all-to-all 等固定开销不随层数缩小**，占比变大 → MFU 掉一点。
- **对调试足够有代表性**：128 GPU 能复现绝大部分性能特征（1550 vs 1658），验证并行策略 / 环境变量 / 新想法完全够用，省一半机器。
- **若要更贴近全规模 MFU**：保持"每 stage 4 层"（减 VPP 而非减层/stage），让固定开销占比不变——但那样层数减不了太多。**减层/stage 换来的是内存和层数都小、更快迭代**，是 tradeoff。

---

## 运维踩坑：DRA / ComputeDomain clique 卡死（2026-07-17 实战）

**背景**：反复删建 ComputeDomain（多次 apply / delete pool、force-delete pod）后，pool 再也拉不满 64，卡在 ~37/64，`kubectl exec` 也只能连上一半 pod。折腾数小时才定位。

### 症状

- pod 大量 Pending，pool 卡在某个数（如 37/64），**分布不均**（如 a=11 b=11 c=10 d=5）。
- Pending pod **没有硬调度失败**，`describe` 只见 `NotTriggerScaleUp`（autoscaler 软提示），节点全 Ready。
- 部分 pod 反复被删重建（claim 状态 `pending → allocated,reserved → deleted` 循环）。
- 最后几个 pod 卡 `FailedPrepareDynamicResources: ResourceClaim not created yet` / `no relationship found between node and this object`。
- `kubectl exec` 只能连约一半 pod（konnectivity 流被占）——是并发 exec + churn 的副作用，不是主因。

### 根因（实查，非猜测）

1. 删 ComputeDomain 后，`nvidia-dra-driver-gpu` 命名空间残留大量 **`computedomain-daemon-*` pod 卡在 Terminating**（数十个，几小时不退）。
2. 每个 CD 组一个 **IMEX clique**；节点上的 daemon 要加入 clique 才算 CD 的一个可用节点。僵尸 daemon 在 clique 里留了 **stale 条目占位**，新 daemon 加不进 → CD `total nodes` 只到 5-15（不足 16）。
3. CD 节点不够 → pod 的 `compute-domain-channel` DRA claim 无法 admission → 反复 thrash。
4. 叠加：一天内创建/删除上万 claim，**kube resourceclaim controller 积压滞后**，最后几个 pod 卡 "ResourceClaim not created yet"。

### 诊断命令

```bash
CTX=gke_tencent-gcp-taiji-poc_us-central1_gb300-gke-test
# 1. 看僵尸 daemon（卡 Terminating 的数量）
kubectl --context $CTX get pods -n nvidia-dra-driver-gpu --no-headers | grep computedomain-daemon | awk '{print $3}' | sort | uniq -c
# 2. 看每个 CD 实际认到几个节点（controller 日志）
CTRL=$(kubectl --context $CTX get pods -n nvidia-dra-driver-gpu --no-headers | grep controller | awk '{print $1}')
kubectl --context $CTX logs -n nvidia-dra-driver-gpu $CTRL --tail=60 | grep CDStatusSync | tail
# 3. 看 pending pod 的真实卡点
kubectl --context $CTX describe pod <pending-pod> | grep -A6 Events
```

### 解法

```bash
# 强删所有卡 Terminating 的 computedomain-daemon → 触发 controller CliqueCleanup 清 stale 条目
kubectl --context $CTX get pods -n nvidia-dra-driver-gpu --no-headers | grep computedomain-daemon | grep Terminating \
  | awk '{print $1}' | xargs -P 8 -I {} kubectl --context $CTX delete pod {} -n nvidia-dra-driver-gpu --grace-period=0 --force
# 观察 controller 日志出现 "CliqueCleanup: successfully removed N stale daemon entries"，CD total nodes 会爬回 16
# 对仍卡 "ResourceClaim not created yet" 的个别 pod，强删让其重建（controller 追上后新 claim 秒建）
kubectl --context $CTX delete pod <stuck-pod> --grace-period=0 --force
```

### 预防

- **不要反复 churn ComputeDomain**。改配置尽量 patch StatefulSet，别动 CD。
- 删 pool 用 `kubectl delete -f pool.yaml`（带 CD），删完**等 computedomain-daemon 全部退干净**（`grep computedomain-daemon | grep -c Terminating` 归 0）再重建。
- force-delete pod（`--grace-period=0 --force`）会留下没清干净的 daemon/claim，是这次坑的放大器 —— 非必要不用。
- 大规模启动/分发**不要用并行 `kubectl exec`**（走 konnectivity 会被限流卡半数）；用 **pod-0 SSH + hostfile** fanout（见 `run-*-yw.sh` 同款 SSH-enabled pool，走集群内网 eth0，绕开 API）。

---

## 运维踩坑：节点池"物理坏死" + clique 死结 → 整池征用绕过（2026-07-17 夜实战）

**背景**：4-domain 池要凑满 64（每域 16）。跑到一半发现 **2 个域各卡 15/16**，2 个 pod 长期 Pending（`describe` 只见 `NotTriggerScaleUp`）。**两个域卡的原因完全不同，且都不是上一节那个 churn 僵尸 daemon 坑**。

### 坑 1：node pool 物理坏死 —— GCE 建不出第 16 台（原池不可修复）

- 现象：pool（= subblock）恒为 15 个节点，autoscaler 一直加不出第 16 台。
- **实查根因（gcloud，非猜测）**：

```bash
# MIG 建实例的错误历史
gcloud compute instance-groups managed list-errors <mig-name> --zone <zone> --project tencent-gcp-taiji-poc
# → INTERNAL_ERROR: Instance creation failed: Internal error. (Code: '-5430573231130294902')，已持续 ~24h
# reservation 本身健康：
gcloud compute reservations describe <res> --zone <zone> --project tencent-gcp-taiji   # count/inUseCount/assuredCount 正常，有余量，0 degraded
```

- reservation **健康有余量**（不是配额/预留耗尽），是 GCE 对该 subblock 某实例的**基础设施故障**。**这种坑无法在原池修复**，只能换池。

### 坑 2：健康池上的 DRA clique 软件死结（空节点也调度不上）

- 现象：pool 有满 16 个 `team=yangwhale` 且 `Ready` 的节点，其中 1 台**空着无任何工作负载**，但那个 Pending pod 就是落不上去，`describe` 只见 `NotTriggerScaleUp`，无硬调度失败。
- 根因：该空节点没进 ComputeDomain 的 IMEX clique（daemon 没在其上跑起 → `compute-domain-channel` claim 无法 admission），鸡生蛋死结。与上一节 clique 坑同源，但表现为"单个空节点进不了 clique"。

### 解法：整池征用一个干净未用的 node pool，把该域整体搬过去

> **关键认知**：域内 16 pod 用 `podAffinity topologyKey: cloud.google.com/gce-topology-subblock` **锁死在同一 subblock**，即 **pool 与 subblock 一一对应，不能跨池拼节点**。修不了就**整池换**，不要东拼西凑散节点。

```bash
CTX=gke_tencent-gcp-taiji-poc_us-central1_gb300-gke-test
# 1. 找干净未用的池（team 无标签、节点数 >=16、全 Ready）
for p in 0002 0010 0011 0012; do
  none=$(kubectl --context $CTX get nodes -l cloud.google.com/gke-nodepool=gb300-pool-$p \
    -o custom-columns=T:.metadata.labels.team --no-headers | grep -cw '<none>')
  echo "pool-$p 闲置节点=$none"
done
# 2. 给目标池打 16 个 team=yangwhale
N=$(kubectl --context $CTX get nodes -l cloud.google.com/gke-nodepool=gb300-pool-0012 --no-headers \
    | grep -w Ready | awk '{print $1}' | head -16)
kubectl --context $CTX label node $N team=yangwhale --overwrite
# 3. 改 StatefulSet nodeSelector 池号 + 删旧 STS/CD + 重 apply（nodeSelector 变更需重建 pod）
sed -i 's/gb300-pool-0008/gb300-pool-0012/' yw-pool.yaml
kubectl --context $CTX delete statefulset yw-d
kubectl --context $CTX delete computedomain yw-cd-d
kubectl --context $CTX apply -f yw-pool.yaml    # 重建到新池，全新 CD → 干净 clique
# 4. 腾出的旧坏池摘标签回收（回到闲置 <none>）
kubectl --context $CTX label node -l cloud.google.com/gke-nodepool=gb300-pool-0008 team-
```

### 判断口诀

- **先分清"池物理坏"还是"clique 软件死结"**：`gcloud ...list-errors` 有 GCE `INTERNAL_ERROR` = 物理坏，只能换池；节点齐全但空节点调度不上 = clique 死结，可修 CD 也可换池。
- **有富余闲置池时，整池征用 + 全新 ComputeDomain 是最稳的一步到位**，省去 clique 考古。
- **pool 与 subblock 一一对应**：换池是"换整块 16 节点"，不是补单节点。

---

## 成功经验：换池后"晾一宿让 DRA/RDMA 收敛"才是真解（2026-07-18 实战）

这是本轮最反直觉、也最值钱的一条教训。

### 失败时间线（当晚，换池后立刻硬跑）

刚把 2 个坏域换到干净池、64/64 刚凑齐，就连着拉起 DSV3，**3 次全失败**：

| 次数 | 现象 | 当时判断 |
|------|------|---------|
| 1 | 进到 NCCL，rank 195(新域) collective timeout 崩 | 以为首跑 flakiness |
| 2 | `ncclRemoteError` + `Message truncated: 128120 vs 120120`，新域节点容器 GPU hang，exec 都进不去 | 僵尸进程 bootstrap 撞车 |
| 3 | workers 起来后**卡 rendezvous 静默 hang**，GPU 全程 0%，日志 8 分钟不增长 | 新节点 RDMA 没就绪 |

同时观察到：**反复删建 ComputeDomain 有传染性**——连没动过的好域 (yw-c/pool-0007) 都开始掉 pod、掉 team 标签。当晚越修越乱。

### 成功（次日早晨，同配置同池，一把过 1620）

环境**静置约 6 小时**后，第二天早上：
1. DRA 僵尸 daemon 自动清零（`Terminating` 归 0）。
2. 只需补回 pool-0007 过夜掉标签的 3 台节点 → 64/64。
3. **同一个 `run-dsv3-opt.sh`、同样 4 个池**，干净拉起 → 越过 init/NCCL/capture → 稳态 **1620 TFLOP/s/GPU**，30 步一把过，零崩溃。

### 核心教训

> **重度 churn（反复删建 ComputeDomain）+ 新打标签的闲置节点之后，DRA / IMEX clique / RDMA ipvlan 子系统需要时间自我收敛。此时不要连环硬拉训练——每次失败的 GPU hang 又留下僵尸容器，进一步污染 bootstrap，越修越糟（负反馈）。正确做法是停手、让它静置（几十分钟到数小时）收敛，再干净拉一次。**

配套小教训（都在当晚踩到）：
- **卡 `Terminating` 的僵尸 pod（GPU hang）会堵住新 pod 调度** → `--grace-period=0 --force` 强删即解。
- **新打标签节点的 `device gpuXipvlanY not found in store`（DRANET ipvlan race）是暂时性的** → 删 pod 重建一次 sandbox 就好，不是坏节点。
- **节点会 flap（Ready↔NotReady，如 ps6b）** → 直接换池里另一台健康的 team=NONE 节点，别指望它稳。
- **churn 会连累好域**：迭代换池时盯着所有域的 pod 数，别只盯在换的那个。

---

## SSH 免密 fanout 启动（替代并行 kubectl exec）

大规模（64 pod）启动训练时，**不要用并行 `kubectl exec`**（走 konnectivity 会限流，只能连约一半）。改用 **pod-0 SSH + hostfile fanout**，走集群内网 eth0，绕开 k8s API。

### 方式一（当前）：容器启动时运行 `yw-node-init.sh`

在 image 尚未 bake 依赖前，把 `yw-node-init.sh` 作为 StatefulSet 的容器 command，每次启动装好 sshd + 免密密钥 + dllogger（幂等、apt 带重试）。前置：k8s Secret `yw-ssh`（`id_ed25519` + `authorized_keys`）挂载到 `/etc/yw-ssh`。

```bash
# 建共享密钥 Secret（一次）
ssh-keygen -t ed25519 -N "" -f yw_ssh_key -C yw-pool
kubectl create secret generic yw-ssh \
  --from-file=id_ed25519=yw_ssh_key --from-file=authorized_keys=yw_ssh_key.pub
# StatefulSet 挂 Secret + command 跑 yw-node-init.sh（见该脚本头部注释）
```

启动训练（pod-0 fanout）：

```bash
kubectl exec yw-a-0 -- bash -c '
  > /tmp/hostfile; for g in a b c d; do for i in $(seq 0 15); do echo yw-$g-$i.yw >> /tmp/hostfile; done; done
  # 分发脚本 + 启动，全走 SSH 内网
  cat /tmp/hostfile | xargs -P 32 -I H scp -q /tmp/run-dsv3-yw.sh H:/tmp/
  cat /tmp/hostfile | xargs -P 32 -I H ssh H "nohup /tmp/run-dsv3-yw.sh > /tmp/run.log 2>&1 &"
'
```

> ⚠️ **坑：SSH 启动丢容器整套 ENV（2026-07-17 实战，踩了两次）**
> SSH 会话是**全新 login shell**，**完全不继承容器镜像的 ENV**（PATH、LD_LIBRARY_PATH、CUDA_HOME、CPATH…全丢）。`kubectl exec` 会继承容器 ENV，所以之前用 exec 启动一直没暴露这坑；一改 SSH 启动就连环踩：
> - **丢 PATH**（`/opt/venv/bin` 不在）→ `python` 变系统 `/usr/bin/python` → **`ModuleNotFoundError: No module named 'nemo_run'`**
> - **丢 CUDA env**（CUDA_HOME/include）→ Triton JIT 编译找不到头文件 → **`fatal error: cuda.h: No such file or directory`** → 崩
> - 只补 PATH 是打地鼠，补完 PATH 又冒出 cuda.h。
>
> **根治（推荐）**：run 脚本开头**加载容器 PID 1 的完整 ENV**，一次性拿全（PATH/LD_LIBRARY_PATH/CUDA_HOME/…）：
> ```bash
> # SSH 启动必加：继承容器完整 env（login shell 会丢）
> if [ -r /proc/1/environ ]; then
>   while IFS= read -r -d '' __e; do export "$__e" 2>/dev/null || true; done < /proc/1/environ
> fi
> ```
> 之后脚本再显式 export 训练专用 env（NCCL_*、full-graph 两项等）覆盖即可。
> **备选/root fix**：bake image 时写 `/etc/profile.d/*.sh` 或 sshd `PermitUserEnvironment` + `~/.ssh/environment`，让 SSH 会话自动继承（`yw-node-init.sh` 已写 `/etc/profile.d/yw-env.sh`，但非交互 SSH 未必 source profile.d，故 run 脚本内的 `/proc/1/environ` 加载是最稳的双保险）。

> 排查口诀：SSH 启动报 `No module named X` / `cuda.h: No such file` / `command not found` / 找错 python → 基本都是**丢了容器 ENV**。对比 `kubectl exec pod -- bash -c 'echo $PATH'` vs `kubectl exec pod -- ssh 自己.yw 'echo $PATH'` 立见分晓。

### 方式二（目标）：bake 进 image

把 sshd + dllogger + 共享密钥 + host keys + sshd_config 全 bake 进镜像，pod 启动只需 `/usr/local/bin/yw-start.sh`（起 sshd + sleep）。Dockerfile 见 `Dockerfile.yw-ssh`。

```bash
docker buildx build --platform linux/arm64 \
  -t us-central1-docker.pkg.dev/tencent-gcp-taiji-poc/gb300-images/nemo-gb300-ready:26.06-v2-ssh \
  --push .
```

> **构建环境坑（2026-07-17 未跑通）**：基础镜像 18.4GB/ARM64。gLinux(x86) 无 docker daemon；Cloud Build 默认 compute SA 缺 cloudbuild 源 bucket 的 storage 权限（403）。待项目管理员给 Cloud Build SA 授权，或用原生 ARM 构建器后再 bake。设计（Dockerfile + 依赖清单）已就绪。
> **安全提示**：`Dockerfile.yw-ssh` 会把共享私钥 bake 进镜像——仅限 benchmark/POC 集群，勿用于生产。

---

## 附：文件清单

| 文件 | 说明 |
|------|------|
| `yw-pool-256.yaml` | 4-CD sleep-infinity pod 池（256 GPU on team=yangwhale；含 SSH-enabled 变体的免密 fanout 基础） |
| `run-dsv3-yw.sh` | DSV3 单 pod 启动脚本（env + rank 计算 + torchrun native recipe） |
| `yw-node-init.sh` | 容器启动初始化（装 sshd + 注入免密密钥 + dllogger，幂等带重试）— image bake 前的替代 |
| `Dockerfile.yw-ssh` | 将上述依赖 bake 进镜像的 Dockerfile（待有 ARM 构建环境后使用） |

---

*2026-07-17 · GB300 256 GPU full_iteration graph 首次跑通 · DeepSeek V3 ~1553 TFLOP/s*
