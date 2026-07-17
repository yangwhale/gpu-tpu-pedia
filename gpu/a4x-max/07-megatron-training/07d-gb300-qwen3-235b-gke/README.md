# GB300 NVL72 (A4X Max) 256 GPU MoE 训练 — GKE

GB300 (A4X Max) GKE 集群上的 256 GPU MoE 大模型预训练 benchmark，基于 NVIDIA Megatron Bridge 官方 recipe。覆盖 DeepSeek V3 (671B) 和 Qwen3 235B-A22B。

**核心成果**：2026-07-17 首次在 GB300 256 GPU 上跑通 **full_iteration CUDA graph**，DeepSeek V3 达到 **~1553 MODEL TFLOP/s/GPU**（30 步稳态）。

**参考**：
- [NVIDIA Megatron Bridge Performance Summary](https://docs.nvidia.com/nemo/megatron-bridge/latest/performance-summary.html)
- [AI-Hypercomputer/gpu-recipes](https://github.com/AI-Hypercomputer/gpu-recipes)
- Bridge 版本：r0.5.0（NeMo 容器 `nemo-gb300-ready:26.06-v1` 内置）

---

## 成果总览

| 模型 | 规模 | 精度 | Graph | 稳态 TFLOP/s/GPU | Step Time | 状态 |
|------|------|------|-------|-----------------|-----------|------|
| DeepSeek V3 (61L, 671B) | 256 GPU (4×16) | FP8_MX | **full_iteration** | **~1553** | 5.47s | 通过 (30 步) |
| Qwen3 235B-A22B | 256 GPU (4×16) | FP8_MX | full_iteration | 待测 | — | 计划中 |

> DeepSeek V3 官方无公开 256 GPU 数字对标；Qwen3 235B 官方 GB300 256 GPU V2 = 1335 TFLOP/s。

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
| MBS / GBS | 1 / 2048 (V1)；V2 = 4096 |
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

## 附：文件清单

| 文件 | 说明 |
|------|------|
| `yw-pool-256.yaml` | 4-CD sleep-infinity pod 池（256 GPU on team=yangwhale） |
| `run-dsv3-yw.sh` | DSV3 单 pod 启动脚本（env + rank 计算 + torchrun native recipe） |

---

*2026-07-17 · GB300 256 GPU full_iteration graph 首次跑通 · DeepSeek V3 ~1553 TFLOP/s*
