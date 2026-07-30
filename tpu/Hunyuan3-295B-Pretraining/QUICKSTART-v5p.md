# 混元 3（295B-A21B）在 TPU v5p 上预训练 — Quick Start

用 MaxText 在 TPU v5p 上跑腾讯混元 3 的完整方案。从零到拿到基线，两条命令。

| | |
|---|---|
| 模型 | Tencent Hunyuan 3，295B 总参 / 21B 激活，80 层 MoE |
| 平台 | TPU v5p，256 芯片（`4x8x8`） |
| 框架 | MaxText（nnx），代码在 [`yangwhale/maxtext` 的 `hunyuan3` 分支](https://github.com/yangwhale/maxtext/tree/hunyuan3) |
| 精度 | BF16 计算 / FP32 主权重 |
| **实测基线** | **step 63.17 s · 161.0 TFLOP/s/chip · MFU 35.07% · 265,588 tok/s** |

> 这份文档只讲**当前可复现的那一条路径**。完整的移植过程、失败轮次、
> 十二个 bug 的复盘在 [README.md](README.md)，需要追溯时再去查。

---

## 1. Hy3 是什么：一半 Qwen3，一半 DeepSeek V3

理解这一点，后面所有配置就都讲得通了。

Hy3 的结构可以精确拆成两半，而这两半在 MaxText 里**都已经有现成实现**：

| 组成部分 | 血统 | MaxText 里的现成实现 |
|---|---|---|
| Attention：GQA 64q/8kv + QK-LayerNorm + 无 bias | **Qwen3** | `qwen3.self_attention_with_norm` |
| MoE：sigmoid 路由 + 专家偏置 + 共享专家 | **DeepSeek V3** | `moe.get_routed_and_shared_moe` |
| 第 0 层 dense、1–79 层 MoE 的分层扫描 | **DeepSeek V3** | `decoders.py` 的 `first_num_dense_layers` 机制 |
| MTP（多 token 预测）1 层 | 通用 | `layers/multi_token_prediction.py` |

**所以新写的代码只有装配逻辑，零新数学。** 模型文件 `models/hunyuan3.py`
一共 161 行有效代码，两个类各约 40 行，全部是调用上面这些现成组件。

### 1.1 为什么不能直接用现成的 decoder block

MaxText 有 `qwen3_moe` 和 `deepseek` 两个 block，各缺一半：

| Hy3 需要 | `qwen3_moe` | `deepseek` |
|---|---|---|
| GQA attention | ✅ | ❌ 硬编码 MLA |
| sigmoid 路由 + 专家偏置 | ✅ | ✅ |
| 共享专家 ×1 | ❌ 拿不到 | ✅ |
| 第 0 层 dense | ❌ 不支持 | ✅ |

于是新增了一个 `decoder_block: "hunyuan3"`，把两边各取一半接起来。

### 1.2 与 DeepSeek V3 的一处关键差异

**Hy3 没有 device-limited routing。** DSV3 把 256 个专家分成 8 组，
一个 token 先选 4 个组、再在组内选 top-8，目的是压缩 all-to-all 扇出。
Hy3 直接在全部 192 个专家里做全局 top-8。

> ⚠️ **配置里不要设 `n_routing_groups` / `topk_routing_group`。**
> MaxText 默认 `-1`（禁用），正好匹配 Hy3。照搬 DSV3 配方把这两项加进来，
> 路由行为就变了 —— 而且不报错。

### 1.3 结构参数

| 项 | 值 | | 项 | 值 |
|---|---|---|---|---|
| 层数 | 80（第 0 层 dense） | | routed experts | 192 |
| hidden_size | 4096 | | top-k | 8 |
| ffn_hidden_size（dense 层） | 13312 | | moe_ffn_hidden_size | 1536 |
| attention heads | 64 | | shared experts | 1 |
| KV groups（GQA） | 8 | | 路由打分 | sigmoid |
| head_dim | 128 | | 专家偏置（aux-loss-free） | 启用 |
| vocab_size | 120832 | | routed scaling factor | 2.826 |
| rope theta | 11158840.0 | | MTP 层数 | 1 |
| QK LayerNorm | 是 | | tie embeddings | 否 |

**框架报告的参数量：298.786 B** = 模型本体 294.9 B + MTP 头 3.886 B。
两个平台逐位一致，可以拿它当第一道健康检查。

### 1.4 参数量分布决定并行策略

| 组成 | 参数量 | 占比 |
|---|---|---|
| **路由专家** | **286.2 B** | **97.0%** |
| 共享专家 | 1.49 B | 0.5% |
| Attention | 6.04 B | 2.0% |
| Dense FFN（第 0 层） | 0.16 B | 0.1% |
| Embedding + LM head | 0.99 B | 0.3% |

**97% 的参数在专家里**，直接推出三条：

1. **TP 无用** —— attention 只占 2%，切它纯亏通信。配 `ici_tensor_parallelism=1`。
2. **不用 EP** —— 在 TPU 上专家并行是负优化，实测 `EP=64 / FSDP=4` 直接超显存
   326 GB。v5p 的 3D torus + SparseCore 卸载能把 FSDP 的集合通信藏得很好，
   反而是纯 FSDP 最快。配 `ici_fsdp_parallelism=-1`（吃满 256 路）。
3. **显存压力主要来自专家权重** —— 优化显存要从优化器状态和激活入手，
   见 §6。

---

## 2. 代码从哪来

### 2.1 唯一真相：fork 的分支

```
https://github.com/yangwhale/maxtext  分支 hunyuan3
```

基于上游 main，三个 commit：

| commit | 内容 |
|---|---|
| `Resolve the loss-free-balancing bias path per decoder block` | 上游 bug 修复（与 Hy3 无关，任何非 DeepSeek 模型开 aux-loss-free 均衡都会撞） |
| `Add Tencent Hunyuan 3 (295B-A21B)` | 模型本体 + 注册 |
| `Let Hunyuan3 use the SwiGLU activation bound too` | 激活截断白名单 |

跟上游用 `git rebase upstream/main`。

### 2.2 新增 3 个文件

| 文件 | 说明 |
|---|---|
| [`src/maxtext/models/hunyuan3.py`](https://github.com/yangwhale/maxtext/blob/hunyuan3/src/maxtext/models/hunyuan3.py) | `Hunyuan3DenseLayer` + `Hunyuan3MoELayer`，只做接线 |
| [`src/maxtext/configs/models/hunyuan3-295b.yml`](https://github.com/yangwhale/maxtext/blob/hunyuan3/src/maxtext/configs/models/hunyuan3-295b.yml) | 正式配置，全文见附录 A |
| `src/maxtext/configs/models/hunyuan3-smoke.yml` | 4 层缩层配置，结构与正式版完全一致 |

### 2.3 改动上游 12 个文件

MaxText 里几乎每个「这个模型该走哪条路」的判断，都是一张**按模型家族名字写死的表**。
加一个新模型不是改一处，是把这类表全部找齐、逐个问「Hy3 该不该在这里」。
12 个文件分三类：

**A. 身份登记（4 处，纯机械）**

| 文件 | 做什么 |
|---|---|
| `common/common_types.py` | 加 `HUNYUAN3` 枚举 |
| `configs/types.py` | pydantic `Literal` 白名单加两个模型名 |
| `layers/decoders.py` | 分派表 ×2（linen 侧） |
| `layers/nnx_decoders.py` | 分派表 ×1（nnx 侧，**最容易漏的一张**） |

**B. 行为归类（7 处，每处都是一道判断题）**

| 文件 | 判断的是什么 | Hy3 的答案 |
|---|---|---|
| `layers/moe.py` | 路由数学、共享专家、SwiGLU 截断 | ✅ 跟 DSV3 一路 |
| `layers/linears.py` | 专家宽度 | ✅ |
| `layers/multi_token_prediction.py` | MTP 的 batch 重分片 | ✅ |
| `utils/maxtext_utils.py` | **FLOP 统计公式** | ✅ 不加会让 MFU 虚高约 5 倍 |
| `utils/generate_param_only_checkpoint.py` | 权重导出时的分组展开 | ✅ 不加会导出错误结构 |
| `utils/layerwise_quantization.py` | 逐层量化白名单 | ✅ |
| `experimental/rl/grpo_utils.py` | RL 参数同步分组 | ✅ |

**C. 上游 bug（1 处）**

`trainers/pre_train/train.py` 把 DeepSeek 的模块属性名 `DeepSeekMoeBlock_0`
写死在无梯度专家偏置的更新路径里。任何不叫这个名字的模型只要开
`routed_bias_update_rate`，配置能过、训练能起、**第一步 AttributeError**。
改成按 `decoder_block` 查表。

---

## 3. 环境准备

四件事，缺一件都跑不起来。

### 3.1 GKE 集群 + TPU 节点池

```bash
# 256 芯片（64 台 × 4）—— 主力
gcloud container node-pools create np-v5p-256 \
  --cluster=CLUSTER --project=PROJECT --region=us-central1 \
  --node-locations=us-central1-a \
  --machine-type=ct5p-hightpu-4t --tpu-topology=4x8x8 \
  --num-nodes=64 --spot --scopes=cloud-platform

# 4 芯片小池 —— 冒烟用，改一行代码几十秒验一轮
gcloud container node-pools create np-v5p-dev \
  --cluster=CLUSTER --project=PROJECT --region=us-central1 \
  --node-locations=us-central1-a \
  --machine-type=ct5p-hightpu-4t --tpu-topology=2x2x1 \
  --num-nodes=1 --spot --scopes=cloud-platform
```

- `--num-nodes` = 芯片数 ÷ 4，且必须与 `--tpu-topology` 相乘一致
- 256 芯片池实测 **8 分 26 秒**建完
- **v5p 在 us-central1 只有 `-a` 区有货**；集群是区域级的，节点池自己指定 zone 即可
- Spot 配额在控制台查不到不代表没有 —— v5p 不走 `PREEMPTIBLE_TPU_LITE_PODSLICE_V5`
  那组老 metric，Cloud Quotas API 返回空。**只能试**，报错会直接告诉你是配额还是容量

### 3.2 JobSet CRD

```bash
kubectl apply --server-side -f \
  https://github.com/kubernetes-sigs/jobset/releases/download/v0.11.1/manifests.yaml
kubectl wait --for=condition=Available deploy/jobset-controller-manager \
  -n jobset-system --timeout=180s
```

v0.11.1 自带证书，**不需要 cert-manager**。新集群默认没有这个 CRD，
不装的话提交训练时 `kubectl apply` 会找不到 `jobset.x-k8s.io/v1alpha2`。

### 3.3 暂存桶 + 跨项目授权

```bash
gcloud storage buckets create gs://YOUR-STAGE-BUCKET --location=US

NODE_SA=<集群项目号>-compute@developer.gserviceaccount.com
gcloud storage buckets add-iam-policy-binding gs://YOUR-STAGE-BUCKET \
  --member="serviceAccount:$NODE_SA" --role=roles/storage.objectViewer

# 镜像在别的项目时，节点 SA 还要能拉
gcloud artifacts repositories add-iam-policy-binding gcr.io --location=us \
  --project=IMAGE_PROJECT --member="serviceAccount:$NODE_SA" \
  --role=roles/artifactregistry.reader
```

### 3.4 网络（新项目常踩）

共享项目的 default VPC 是 auto 模式，`10.128.0.0/9` 被各 region 子网占满，
`10.0.0.0/9` 又被切碎，GKE 凑不出一整块 `/14` 给 pod：

```
The network "default" does not have available private IP space in
10.0.0.0/9 to reserve a /14 block for pods
```

**建自己的 custom VPC 最省事**，顺带能把 MTU 开到 8896：

```bash
gcloud compute networks create NAME-vpc --subnet-mode=custom --mtu=8896
gcloud compute networks subnets create NAME-uc1 --network=NAME-vpc \
  --region=us-central1 --range=10.124.0.0/22 \
  --secondary-range=pods=10.125.0.0/16,services=10.124.16.0/20 \
  --enable-private-ip-google-access
```

三段全压在 `10.124.0.0/15` 内，将来做 VPC peering 只需让对方避开这一段。
选网段前先跑 `gcloud network-connectivity internal-ranges list` 拿权威占用表 ——
不要只看 `clusters list` 的 `clusterIpv4Cidr`，那漏掉了子网自身占的地址。

---

## 4. 跑起来

```bash
cd maxtext-hunyuan3/
gcloud container clusters get-credentials CLUSTER --region=REGION --project=PROJECT

export GCS_STAGE=gs://YOUR-STAGE-BUCKET/hy3
export IMAGE=us-docker.pkg.dev/YOUR-PROJECT/gcr.io/YOUR-maxtext-latest:runner

# ① 准备代码（只有改了代码才要重跑；换参数不用）
bash prep.sh

# ② 起训练
PLATFORM=v5p bash run.sh myrun

# ③ 看日志
kubectl logs -f job/hy3-myrun-slice-job-0 -c jax-tpu
```

### 4.1 两个脚本各做什么

| 脚本 | 动作 |
|---|---|
| `prep.sh` | clone `hunyuan3` 分支 → **8 项自检** → `tar` 整棵 `src/maxtext` → 传 GCS |
| `run.sh` | 提交 JobSet；pod 里 `rm -rf /deps/src/maxtext` 后**整棵解包覆盖** |

**注意是整棵覆盖，不是只注入改动文件。** 只注入的话，测的是
「我的改动 + 容器里的旧基座」，不是分支本身。

`prep.sh` 的 8 项自检挡的是「分支自己少东西」：三个新增文件在不在、
白名单两个模型名全不全、枚举有没有 `HUNYUAN3`、`train.py` 补丁在不在、
`Hunyuan3MoeBlock_0` 这个属性名在模型文件和训练循环两边对不对得上。

### 4.2 先跑冒烟

```bash
NODES=1 TOPO=2x2x1 PLATFORM=v5p MODEL=hunyuan3-smoke STEPS=8 \
  bash run.sh smoke per_device_batch_size=1 max_target_length=2048
```

4 层缩层，结构与 295B 完全一致（192 专家、top-8、sigmoid、专家偏置、
共享专家、GQA、QK-norm、fp32 路由、MTP 全是满配），只砍层数。
单步约 0.82 s，8 步 loss 应从 13.45 降到 10.35。

**为什么 4 层能代表 80 层**：MaxText 按**类型**分组做 `scan`，
79 个 MoE 层共用同一份编译产物，层与层的差别只在权重数值上，不在代码路径上。
所以冒烟测的不是「抽样几层」，而是**那个被复用 79 次的唯一函数**。

冒烟覆盖不到的：显存压力、大规模切分、80 层累积的数值误差、
完整 XLA flag 集、收敛质量、以及全部性能。它证明的是「代码路径都对」。

### 4.3 读日志的四条规矩

1. **先确认 `64/64 Running` 再看日志。** TPU 切片全有全无，人不齐时活着的
   pod 会报 `GetSliceInfo can only be invoked after a slice is built` ——
   那是症状不是病因。
2. **判错看最早那条，不是日志尾。** 配置非法会先把 TPU 拉起来再退，
   真正的报错（`MAXTEXT CONFIG ERROR` / pydantic 的 `Value error`）在日志上方。
3. **step 0 含编译，step 1/2 是 JAX 异步派发的假读数**，稳态取 step ≥ 3。
   v5p 上编译约 9 分钟。
4. **v5p 是 MegaCore，1 device = 1 chip**，日志里的 `TFLOP/s/device`
   不需要换算，`MFU = TFLOP/s/device ÷ 459`。

---

## 5. v5p 基线

### 5.1 硬件

| | 值 |
|---|---|
| 节点池 | `np-v5p-256`，64 台 `ct5p-hightpu-4t`，拓扑 `4x8x8` |
| 芯片数 | **256**（注意这在 Google 命名法里叫 `v5p-512`） |
| JAX device 数 | 256（MegaCore，1 device = 1 chip） |
| HBM / chip | 95.74 GB HBM2e |
| BF16 峰值 / chip | 459 TFLOPS |
| 总算力 | 117.5 PFLOPS BF16 |

> v5p **没有 FP8 加速**，FP8 峰值等于 BF16。所以这个平台上不必考虑 FP8 训练。

### 5.2 实测指标

```
number parameters: 298.786 billion
completed step: 3, seconds: 63.174, TFLOP/s/device: 160.974, loss: 13.200
completed step: 4, seconds: 63.176, TFLOP/s/device: 160.969, loss: 13.129
completed step: 5, seconds: 63.174, TFLOP/s/device: 160.974, loss: 13.072
completed step: 6, seconds: 63.171, TFLOP/s/device: 160.981, loss: 13.029
completed step: 7, seconds: 63.170, TFLOP/s/device: 160.984, loss: 12.998
```

| 指标 | 值 |
|---|---|
| 参数量（框架报） | **298.786 B** |
| 稳态 step | **63.17 s** |
| TFLOP/s / chip | **160.98** |
| **MFU** | **35.07%** |
| 整机吞吐 | **265,588 tok/s**（1,037 tok/s/chip） |
| 每步 token | 16,777,216（256 × 8 × 8192） |
| 每卡 HBM 峰值 | 贴近 95.74 GB 上限 |
| step 抖动 | 毫秒级（63.170–63.176） |

**这个数字换过项目、换过 VPC、换过集群从零复现过**，
代码从 GitHub 分支现 clone 现打包，与既有水位相差 +0.05%。

> **参照**：同一个模型在 GB300 上 64 GPU BF16 的 MFU 是 31.6%。
> v5p 的单卡算力只有 GB300 的 1/5.9，MFU 反而更高 ——
> 256 芯片的 3D torus 加 SparseCore 集合通信卸载，
> 把 MoE 那些碎通信藏得比 NVLink 域还干净。
> 代价是要 256 张卡才换来 GB300 64 卡约七成的整机吞吐。

### 5.3 完整参数集

以下就是 `run.sh` 的 v5p 分支在用的全部参数。

**并行与模型**

```
model_name=hunyuan3-295b
override_model_config=True
ici_fsdp_parallelism=-1          # 吃满 256 路 FSDP
ici_tensor_parallelism=1         # TP 无用，attention 只占 2% 参数
```

**MoE**

```
megablox=True                    # 分组矩阵乘
sparse_matmul=True
use_custom_sort_vjp=True         # 「按专家排序」那步的自定义反向传播
```

**batch 与序列**

```
per_device_batch_size=8          # 已到显存上限，再上 OOM
max_target_length=8192
```

**Attention（splash kernel 块大小，全部 2048）**

```
attention=flash
sa_block_q=2048  sa_block_kv=2048  sa_block_kv_compute=2048
sa_block_q_dkv=2048  sa_block_kv_dkv=2048  sa_block_kv_dkv_compute=2048
sa_block_q_dq=2048  sa_block_kv_dq=2048
sa_use_fused_bwd_kernel=False    # v5p 上关掉才快
```

**重计算与 offload**

```
scan_layers=True
remat_policy=custom
decoder_layer_input=offload
out_proj=remat                   # 官方 DSV3 配方用 offload，Hy3 上要改回 remat
```

**精度与其他**

```
dtype=bfloat16
weight_dtype=float32
allow_split_physical_axes=True
tokenizer_type=tiktoken
tokenizer_path=src/maxtext/assets/tokenizer_llama3.tiktoken
```

**MoE tile 参数（18 个）**

新版 MaxText 把旧版的 3 个 tile 参数拆成了 6 条通路各 3 维：

```
{wi,wo}_tile_{fwd,dlhs,drhs}_{batch_seq,embed_dim,mlp_dim}
```

当前全部取同值 —— `batch_seq=512`、`embed_dim=1024`、`mlp_dim=1024`。
新版允许六条通路各配各的，**这是一个还没扫过的调优面**。

### 5.4 XLA flag（25 个）

通过 `LIBTPU_INIT_ARGS` 传入。v5p 上最值钱的是 SparseCore 集合通信卸载那一组。

```
--xla_tpu_dvfs_p_state=3
--xla_tpu_scoped_vmem_limit_kib=65472

# SparseCore 集合通信卸载 —— v5p 上这一组值 4.07 pp MFU
--xla_tpu_enable_sparse_core_reduce_scatter_v2=true
--xla_tpu_enable_sparse_core_collective_offload_all_gather=true
--xla_tpu_enable_sparse_core_collective_offload_2d_all_gather=true
--xla_tpu_enable_sparse_core_collective_offload_all_reduce=true
--xla_tpu_enable_sparse_core_collective_offload_reduce_scatter=true
--xla_tpu_enable_sparse_core_collective_aggregator=true
--xla_tpu_enable_concurrent_sparse_core_offloading=true
--xla_tpu_enable_offloading_gather_to_sparsecore=true
--xla_tpu_enable_sparse_core_offload_queuing_in_lhs=true
--xla_tpu_sparse_core_all_gather_latency_multiplier=1
--xla_tpu_sparse_core_reduce_scatter_latency_multiplier=3
--xla_tpu_use_tc_device_shape_on_sc=True
--xla_sc_disable_megacore_partitioning=True
--xla_sc_enable_instruction_fusion=false
--xla_sc_disjoint_spmem=false
--xla_tpu_enable_all_gather_offload_tracing=true

# 异步 all-gather
--xla_enable_async_all_gather=true
--xla_tpu_prefer_async_allgather_to_allreduce=true
--xla_tpu_enable_async_collective_fusion_fuse_all_gather=false

# 调度
--xla_tpu_enable_latency_hiding_layer_scheduler=true
--xla_tpu_scheduler_percent_shared_memory_limit=150
--xla_tpu_aggressive_opt_barrier_removal=true
--xla_tpu_pcie_bandwidth_multiplier=0.03
```

> ⚠️ **libtpu 对不认识的 flag 是硬失败**（`Unknown command line flag`，进程直接退）。
> 官方 DSV3 v5p 配方里还有一个 `--2a886c8_chip_config_name=megachip_tccontrol`，
> 新版 libtpu 已经摘掉，带上会起不来。**换镜像必须重过一遍 flag 集。**

---

## 6. 每个选择值多少

从官方 DeepSeek3 v5p 配方出发做的消融，按贡献排序。这张表回答
「这些参数为什么是这个值」，也告诉你哪些不能动。

| 改动 | ΔMFU | 说明 |
|---|---|---|
| **序列长度 8192**（而非 4096） | **+9.20 pp** | 最大的单项 |
| **不用 dropping**（不设 `capacity_factor`） | **+7.85 pp** | 192 专家 top-8，丢 token 代价很高 |
| **`use_custom_sort_vjp=True`** | **+6.15 pp** | 默认 `False`，一个布尔量快五分之一 |
| **`per_device_batch_size` 4 → 8** | **+5.16 pp** | 已到显存顶，再上 OOM |
| **25 个 XLA flag**（而非只留 2 个） | **+4.07 pp** | 主要来自 SparseCore 卸载那一组 |
| `out_proj=remat`（而非 offload） | +0.22 pp | 官方 DSV3 用 offload，Hy3 上略亏 |
| 保留 `tile_*` 参数 | +0.31 pp | |

**不要开的三个开关**（都实测过）：

| 开关 | 后果 |
|---|---|
| `EP=64 / FSDP=4` | **OOM**，超 326 GB。TPU 上专家并行是负优化 |
| `shard_exp_on_fsdp=True` | **OOM**。要拿一半 FSDP 宽度去换，净亏 |
| `shard_optimizer_over_data=True` | 优化器状态 sharding 退化成全复制，`device_put` 直接拒 |

> **最大的一条经验**：起点是自己攒的配置，MFU 只有 **2.45%**；
> 照抄官方 DeepSeek3 v5p 配方、只换模型名，**一步跳到 31.56%（12.9 倍）**。
> 上表那些调参加起来，才把它从 31.56% 推到现在的水位。
> **移植新模型时，先找官方同类配方，再谈调参** —— 顺序反了会浪费很多轮。

---

## 7. 优化空间

按预期收益 / 实施成本排序。

### 7.1 优化器状态降到 BF16（低成本，有确定余量）

当前每卡贴着 95.74 GB 跑。FSDP=256 下每芯片 46.69 亿参数，
fp32 优化器是 weights 4 + grads 4 + mu 4 + nu 4 = 16 B/param = 18.7 GB 静态。

把 Adam 一阶动量和梯度降到 bf16：

```
mu_dtype=bfloat16 grad_dtype=bfloat16
```

16 → 12 B/param，静态 **18.7 → 14.0 GB，腾出 4.7 GB/chip**。

- `nu_dtype`（二阶动量）optax 不支持单独设，恒随 `weight_dtype`
- **主权重仍是 fp32**，所以这是三份状态里最温和的一个改动
- 腾出来的显存可以拿去减 remat 或者加 batch

这一项在 v7 分支上已经开着，**v5p 从没试过**。

### 7.2 `scan(unroll=N)`（需要改代码，机理清楚）

`jax.lax.scan` 有个 `unroll` 参数，`unroll=5` 就是「循环体里放 5 层、
转 79/5 圈」，介于全循环和全展开之间。MaxText 目前没用（走默认值 1）。

**为什么在 TPU 上可能有效**：整个 step 是一个 XLA 程序，`scan` 编译成
HLO 的 `while` 循环，**没有 per-layer launch 开销** —— 所以收益不是「少 launch」。
真正的点在于 **`while` 循环体是 XLA 的调度边界，跨迭代不能重排**。
循环体里只有 1 层时，第 N 层的 all-gather 没法藏进第 N−1 层的计算；
`unroll=N` 把这 N 层放进同一个调度域，延迟隐藏调度器才有发挥空间。

改动约 10 行（加一个 config 字段透传给 `jax.lax.scan`）。
建议按 1/2/4/8 扫一轮找拐点，同时记录**吞吐、编译时间、HBM 峰值**三条曲线。

### 7.3 MoE tile 参数逐通路调优（零成本，未探索）

18 个 tile 参数现在全取同值，纯粹是从旧版 3 个参数「形式对齐」过来的。
新版允许 `{wi,wo} × {fwd,dlhs,drhs}` 六条通路各配各的 —— **这个面完全没扫过**。

### 7.4 抓 xplane trace 定位瓶颈（诊断，不是优化）

参数扫描的收益已经明显变平。继续盲扫期望很低，
需要 trace 直接回答「时间花在路由 / 分组重排 / all-to-all / offload 还是 GEMM」。
注意 profiler 自身有开销，要给它更长的窗口才能跑出稳态。

---

## 8. 已知限制

| 项 | 状态 |
|---|---|
| 数据集 | 目前是 `dataset_type=synthetic`。**loss 下降只证明「能算且不发散」，不是收敛证据** |
| HF 权重 → MaxText Orbax 转换 | 未做。只跑吞吐基线可以不碰；要 SFT 必须做 |
| SFT 时冻结 `gate.bias` | 未做。上游有偏置更新规则，但 SFT 需要的是**冻结**它，当前无 `trainable_parameters_mask` |
| `initializer_range: 0.006` | 未配。只影响 from-scratch 预训练的初始化；加载权重或 SFT 不受影响 |
| 完整 loss 曲线 | 目前只跑到 step 9。建议补一条 30 步以上的 |

---

## 附录 A：`hunyuan3-295b.yml` 全文

```yaml
decoder_block: "hunyuan3"
base_num_decoder_layers: 80
base_emb_dim: 4096
base_mlp_dim: 13312            # dense layer 0 only
base_num_query_heads: 64
base_num_kv_heads: 8           # GQA 8:1
head_dim: 128
vocab_size: 120832
mlp_activations: ["silu", "linear"]
normalization_layer_epsilon: 1.0e-5   # HF rms_norm_eps；注意不是 qwen3/deepseek 用的 1e-6
enable_dropout: False
logits_via_embedding: False    # 等价于 HF 的 untie_embeddings_and_output_weights: True
use_qk_norm: True
attention_bias: False
rope_max_timescale: 11158840
rope_type: "default"
max_position_embeddings: 262144
num_experts: 192
num_experts_per_tok: 8
base_moe_mlp_dim: 1536
shared_experts: 1
first_num_dense_layers: 1
routed_score_func: "sigmoid"
routed_bias: True
routed_scaling_factor: 2.826
norm_topk_prob: False          # route_norm 已在 deepseek_scale_weights 里做过
float32_gate_logits: True      # 192 个专家，bf16 分不开 sigmoid 打分
routed_bias_update_rate: 0.001 # aux-loss-free 均衡，gamma 取自 DSV3
mtp_num_layers: 1
mtp_loss_scaling_factor: 0.1
```

> 两处容易配错、且**不报错**的：
> `normalization_layer_epsilon` 必须是 `1.0e-5`（HF 原文），
> 抄 qwen3/deepseek 的 `1e-6` 不会有任何提示；
> `float32_gate_logits: True` 不能省，192 个专家在 bf16 下分不开 sigmoid 打分。

---

## 附录 B：延伸阅读

| 文档 | 内容 |
|---|---|
| [README.md](README.md) | 完整实验记录：移植全过程、三条战线的全部轮次、12 个 bug 的复盘 |
| [MAXTEXT-PORTING-GUIDE.md](MAXTEXT-PORTING-GUIDE.md) | 把**别的**模型移植到 MaxText 的通用范式 |
| [maxtext-hunyuan3/](maxtext-hunyuan3/) | `prep.sh` / `run.sh` 两个脚本 |
