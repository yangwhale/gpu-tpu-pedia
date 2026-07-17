# Qwen3 235B-A22B 256 GPU 训练 — GB300 NVL72 (A4X Max) GKE

GB300 (A4X Max) GKE 集群上的 Qwen3 235B-A22B (94 层, 128 专家) 256 GPU 预训练 benchmark，基于 NVIDIA Megatron Bridge r0.5.0 官方 recipe。

**核心成果**：2026-07-17 在 GB300 256 GPU 上跑通 **full_iteration CUDA graph**，GBS=8192 达到 **~1360 MODEL TFLOP/s/GPU**（对标官方 1335，**102%**）。

> **关键修正**：Qwen3 GB300 V2 原生 recipe **漏设 VPP**（`virtual_pipeline_model_parallel_size=None`），而 PP=4 + EP A2A overlap 的 assertion 强制要求 VPP。必须手动补 `model.virtual_pipeline_model_parallel_size=2`。这是与 DeepSeek V3（recipe 自带 VPP=8）最大的区别。

> DeepSeek V3 (671B) 的 GKE 训练文档见同级 `07e-gb300-deepseekv3-671b-gke/`。基础设施（4-domain pool、env 变量分析）两者共享。

**参考**：
- [NVIDIA Megatron Bridge Performance Summary](https://docs.nvidia.com/nemo/megatron-bridge/latest/performance-summary.html)
- Bridge 版本：r0.5.0（NeMo 容器 `nemo-gb300-ready:26.06-v1` 内置）

---

## 成果总览

| 项 | 值 |
|----|-----|
| 模型 | Qwen3 235B-A22B (94 层, 128 专家, topk=8) |
| 规模 | 256 GPU (4 domain × 16 节点 × 4 GPU) |
| 精度 | FP8_MX |
| Graph | full_iteration |
| 并行 | TP=1, PP=4, **VPP=2 (手动补)**, EP=32, ETP=1 |
| MBS / GBS | 2 / 8192 |
| 稳态 | 14.26s/step, **~1360 MODEL TFLOP/s/GPU** |
| 显存 | max-reserved **277 / 288 GB (96%)** |
| 状态 | 通过 (30 步) |

### 对标 NVIDIA 官方 (Qwen3 235B GB300 256 GPU MXFP8 V2)

| 来源 | Model TFLOP/s/GPU |
|------|-------------------|
| **本文实测 (VPP=2)** | **~1360** |
| NVIDIA 官方 V2 | 1335 |

> 实测 **1360** vs 官方 **1335** = **102%**。首次尝试 VPP=2 即达官方水平。

---

## 集群架构：4-domain 256 GPU

与 DeepSeek V3 完全相同：256 GPU = 64 节点跨 4 个 subblock，**每 subblock 一个独立 ComputeDomain**（IMEX 只能在单 NVLink domain 内建立）。组内 NVLink 承载 EP=32 all-to-all，组间 CX-8 RDMA 承载 PP/DP。

sleep-infinity pod 池 YAML 见同目录 `yw-pool-256.yaml`（team=yangwhale, node pool 0003/0005/0007/0008）。详见 `07e-gb300-deepseekv3-671b-gke/README.md` 的架构章节。

---

## Part 1 — 完整可跑通步骤

集群 `gb300-gke-test` (Regional us-central1)，context = `gke_tencent-gcp-taiji-poc_us-central1_gb300-gke-test`，所有 `kubectl` 在 gLinux 执行。

### Step 0 — 4-domain pool 就绪

若 pool 未建，`kubectl apply -f yw-pool-256.yaml`，等 64 pod Running + 4 个 ComputeDomain Ready（详见 07e 文档 Step 0-1）。

### Step 1 — 启动脚本（严格对齐 Bridge + VPP 修正）

每 pod 跑 `run-qwen3-yw.sh`（见同目录）。内层训练命令：

```bash
numactl --cpunodebind=$((LOCAL_RANK/2)) --membind=$((LOCAL_RANK/2)) \
python scripts/performance/run_script.py \
  -m qwen -mr qwen3_235b_a22b --task pretrain \
  -g gb300 -c fp8_mx -ng 256 \
  --data mock --max_steps 30 \
  --log_dir /tmp/nemo-results \
  -cv v2 \
  model.virtual_pipeline_model_parallel_size=2 \
  logger.log_throughput=true
```

> **关键**：`-cv v2`（256 GPU full graph 配置）+ `model.virtual_pipeline_model_parallel_size=2`（补 recipe 漏掉的 VPP）。不覆盖 cuda_graph_impl，让 recipe 用原生 full_iteration。

### Step 2 — 分发 + 启动

```bash
# 分发 (base64 避免 stdin symlink 坑)
B64=$(base64 -w0 run-qwen3-yw.sh)
for g in a b c d; do for i in $(seq 0 15); do echo yw-$g-$i; done; done \
  | xargs -P 16 -I {} kubectl exec {} -- bash -c \
    "echo $B64 | base64 -d > /tmp/run-qwen3-yw.sh && chmod +x /tmp/run-qwen3-yw.sh"

# 全部 64 pod 并行启动
for g in a b c d; do for i in $(seq 0 15); do echo yw-$g-$i; done; done \
  | xargs -P 32 -I {} kubectl exec {} -- bash -c \
    "nohup /tmp/run-qwen3-yw.sh > /tmp/qwen3-run.log 2>&1 &"
```

### Step 3 — 监控

```bash
kubectl exec yw-a-0 -- bash -c 'grep "Step Time" /tmp/qwen3-run.log | tail -6'
```

时间线：rendezvous → init → warmup → **graph capture（第 1 步 ~236s）** → settling（第 2 步 ~50s）→ 稳态（14.26s/step, ~1360 TFLOP/s）。总计 ~12 分钟跑完 30 步。

---

## Part 2 — 关键环境变量

与 DeepSeek V3 **绝大部分相同**（EP=32，同一套 full-graph + hybridep env）。完整表见 `07e-gb300-deepseekv3-671b-gke/README.md` Part 2。**Qwen3 与 DSV3 的差异**：

| env | DSV3 | Qwen3 | 原因 |
|-----|------|-------|------|
| `NVTE_ALLOW_NONDETERMINISTIC_ALGO=0` | 设 | **不设** | perf plugin 只对 deepseek 设 |
| `NVTE_NORM_FWD_USE_CUDNN=1` | 设 (保留) | **不设 (popped)** | qwen fp8_mx gb300 → del_cudnn_ln=True |
| `NVTE_NORM_BWD_USE_CUDNN=1` | 设 (保留) | **不设 (popped)** | 同上 |
| `HF_TOKEN` | 不需要 (NullTokenizer) | **必须** | Qwen3 用 `AutoBridge.from_hf_pretrained("Qwen/Qwen3-235B-A22B")` + HF tokenizer |

其余全部相同（full-graph 两项、hybridep 四项、CUDA_DEVICE_MAX_CONNECTIONS=32、layernorm margin=20、cutedsl、NCCL GIB base 等）。

---

## Part 3 — 踩过的坑（Qwen3 特有）

| # | 配置 | 结果 | 根因 / 修正 |
|---|------|------|-------------|
| Q1 | 原生 recipe (VPP=None) | **crash** | `AssertionError: If enabling EP A2A overlap, virtual_pipeline_model_parallel_size must be specified when pipeline_model_parallel_size > 1`。Qwen3 GB300 V2 recipe 漏设 VPP（PP 从 V1 的 1 改成 V2 的 4，但没补 VPP），而 GB200 行有 VPP=3、H100 行有 VPP=4。**修正：override `model.virtual_pipeline_model_parallel_size=2`** |
| Q2 | 手动覆盖 TE graph | crash (optimizer init) | 同 DSV3：覆盖 cuda_graph_impl 绕过 full_iteration+paged_stash 机制 |
| Q3 | 缺 HF_TOKEN | 拉模型 config/tokenizer 失败 | Qwen3 需 `HF_TOKEN` 从 HuggingFace 拉 `Qwen/Qwen3-235B-A22B` |
| Q4 | 沿用 DSV3 的 cudnn LN env | (退化) | Qwen3 del_cudnn_ln=True，不应设 `NVTE_NORM_*_USE_CUDNN`（DSV3 才保留） |

**Qwen3 vs DSV3 关键差异总结**：
1. **VPP**：DSV3 recipe 自带 VPP=8（开箱即跑）；Qwen3 GB300 recipe **漏设**，必须手动补 VPP（本文用 2）。
2. **tokenizer**：DSV3 用 NullTokenizer（无需 HF）；Qwen3 需 HF_TOKEN。
3. **cudnn LN / 非确定算法 env**：deepseek 专属，Qwen3 不设。
4. **显存**：Qwen3 (94 层, PP=4, VPP=2) max-reserved 277/288GB (96%)，比 DSV3 更吃紧，MBS 上调空间小。

> DSV3 通用坑（A-H：full graph 的 stream env、HYBRID_EP_RANKS=EP、CD 每 subblock 一个、NCCL_GRAPH_REGISTER=0 等）同样适用，见 07e 文档 Part 3。

---

## Part 4 — Bridge 原生 recipe 参数参考

来源：`scripts/performance/configs/qwen/qwen3_workload_base_configs.py` `QWEN3_235B_A22B_PRETRAIN_CONFIG_GB300_FP8_MX_V2`

| 参数 | 原生值 | 本文实测值 |
|------|--------|-----------|
| num_layers | 94 | 94 |
| num_moe_experts / topk | 128 / 8 | 128 / 8 |
| TP / PP / EP / ETP | 1 / 4 / 32 / 1 | 同 |
| **VPP** | **None (漏设 → 崩)** | **2 (手动补)** |
| MBS / GBS | 2 / 8192 | 同 |
| seq_length | 4096 | 4096 |
| cuda_graph_impl | full_iteration | 同 |
| moe_a2a_overlap | True | 同 |
| moe_flex_dispatcher_backend | hybridep | 同 |
| cutedsl_fused_grouped_mlp | True | 同 |
| fp8_dot_product_attention | True | 同 |

> 对比：GB200 V2 = PP=8 + **VPP=3** + EP=32；H100 V1 = PP=8 + **VPP=4** + EP=32。唯 GB300 V2 漏 VPP。

---

## 扫描式调优结果 (2026-07-17)

以 VPP=2/MBS=2/GBS=8192 (1360 TFLOP/s) 为基线，扫描 VPP 与 MBS：

| 配置 | 稳态 TFLOP/s | 显存 max-reserved | 结论 |
|------|-------------|-------------------|------|
| **VPP=2, MBS=2** | **~1360** | 277 / 288 GB | ✅ **最优** |
| VPP=4, MBS=2 | ~1325 | 254 GB | 慢 2.6% |
| VPP=8, MBS=2 | — | OOM (capture) | ❌ 显存爆 |
| VPP=4, MBS=4 | — | hang (capture 死锁) | ❌ 不可用 |

**结论：VPP=2（recipe 默认思路）就是 Qwen3 235B 的最优配置。**

**为什么"VPP 越大越好"在这里不成立**：
1. **bubble 本来就小**：num_microbatches = GBS/(MBS×DP) = 8192/(2×64) = 64，PP=4 的 bubble ≈ (PP-1)/(VPP×microbatches)。VPP=2 时已只有 2.3%，VPP=4 降到 1.2%——只省 1%，却让 P2P 通信开销上升，净慢 2.6%。
2. **显存非单调**：VPP=2→4 因 stage 变细 + moe_paged_stash 使单块 buffer 更省（277→254GB）；但 VPP=4→8 时 in-flight microbatch 的激活 stash 份数暴增 + capture 图内存池开销反超，直接 OOM。拐点在 VPP=4 与 8 之间。
3. **MBS 上不去**：MBS 必须整除 GBS/DP=128，故只能 1/2/4/8。MBS=4 激活翻倍，即便配 VPP=4 省显存也在 capture 阶段死锁（NCCL hang）。

> 对比 DeepSeek V3（PP=2, VPP=8 可行）：DSV3 层数/microbatch 结构不同，高 VPP 才划算；Qwen3 (PP=4, microbatch=64) 高 VPP 无收益。**不要跨模型套 VPP 经验。**

---

## 运维踩坑

- **DRA / ComputeDomain clique 卡死**（pool 拉不满 64、卡 ~37、pod Pending 无硬报错）：反复删建 ComputeDomain 留下僵尸 `computedomain-daemon` 占 IMEX clique → 强删僵尸 daemon 触发 CliqueCleanup 恢复。完整诊断 + 解法 + 预防见 `07e-gb300-deepseekv3-671b-gke/README.md` 的「运维踩坑：DRA / ComputeDomain clique 卡死」章节。
- **大规模启动别用并行 kubectl exec**（konnectivity 限流卡半数）；用 pod-0 SSH + hostfile fanout（SSH-enabled pool，走 eth0 内网）。
- **SSH 启动丢容器整套 ENV**：SSH 会话不继承镜像 ENV → 连环踩 `ModuleNotFoundError: nemo_run`（丢 PATH）+ `cuda.h: No such file`（丢 CUDA env）。根治：run 脚本开头从 `/proc/1/environ` 加载容器完整 env。详见 07e 同名坑。

---

## 附：文件清单

| 文件 | 说明 |
|------|------|
| `yw-pool-256.yaml` | 4-CD sleep-infinity pod 池（与 DSV3 共享） |
| `run-qwen3-yw.sh` | Qwen3 235B 单 pod 启动脚本（含 VPP=2 override + HF_TOKEN） |

---

*2026-07-17 · GB300 256 GPU Qwen3 235B full_iteration graph 跑通 · ~1360 TFLOP/s (官方 1335 的 102%) · VPP=2*
