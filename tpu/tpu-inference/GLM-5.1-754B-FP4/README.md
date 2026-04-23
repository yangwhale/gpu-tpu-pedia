# GLM-5.1 754B FP4 Inference on TPU v7x-8

> 端到端指南：在单节点 TPU v7x-8 上运行 GLM-5.1 754B（FP4 量化）推理，
> 包含环境搭建、权重下载、FP4 转换、vLLM 服务启动。
>
> **基于 DeepSeek R1 671B FP4 推理经验**，GLM-5.1 与 DeepSeek V3/R1 架构高度同源（MLA + MoE），
> 量化策略、cache 生成流程、启动方式完全一致。
>
> **代码仓库**: https://github.com/yangwhale/tpu-inference (branch: `feature/moe-fp4-weight-cache`)
>
> **实现方案**: https://cc.higcp.com/pages/glm51-tpu-inference-plan-20260422.html
>
> **HuggingFace 模型**: [zai-org/GLM-5.1-FP8](https://huggingface.co/zai-org/GLM-5.1-FP8)（142 safetensors, 705 GB, FP8 e4m3 128×128 block 量化）

本文档提供两种部署方式：

| 方式 | 适用场景 | 跳转 |
|------|----------|------|
| **Part A: GKE + Docker** | 生产环境，GKE 集群已有 TPU node pool | [Part A](#part-a-gke--docker) |
| **Part B: TPU VM 裸机** | 开发测试，直接在 TPU VM 上安装运行（**推荐**） | [Part B](#part-b-tpu-vm-裸机安装) |

两种方式共享相同的推理和测试步骤（Step 3-7）。

### MoE Cache 生成策略

与 DeepSeek R1 完全一致，推荐 CPU 并行直转：

| 方式 | 步骤 | 实测耗时 | 存储需求 | 推荐 |
|------|------|----------|----------|------|
| **CPU 并行直转** ⭐ | safetensors → FP4（纯 CPU, 12 并发） | **28 min** | 模型 ~705G + FP4 ~757G = **~1.5 TB** | ✅ |
| TPU FP4 直转 | safetensors → FP4（TPU JIT, 串行） | ~114 min（预估） | 同上 | 备选 |
| FP8→FP4 两步法 | safetensors → FP8 cache → FP4 转换 | ~80 min（预估） | 模型 + FP8 + FP4 = **~2.0 TB** | 备选 |

> **已验证（2026-04-23, GKE E2E pod, v7x-8, 224 vCPU, 944 GB RAM）**：
> - CPU 并行直转 76 层 MoE，12 workers，**28 min 完成**（平均 245.5s/层，最快 204s，最慢 285s）
> - 前 2 批（layer 3-26）每层 ~258s（safetensors 冷读 ~55s），后续批次 ~225s（Linux page cache 命中后 load 降至 ~7s）
> - Non-MoE 权重提取：2292 keys，21.54 GB，**109 秒**完成
> - FP8 权重格式：128×128 2D block 量化，e4m3 dtype，scale key `weight_scale_inv`（与 DeepSeek R1 完全一致）
> - `gen_fp4_cache_cpu_parallel.py` 可直接使用，无需修改
>
> **内存注意**：12 workers 峰值内存 ~300 GB。如果 /dev/shm 已有数据（如之前的 cache），
> 会挤占进程可用内存导致 OOM（exit code 137，整个 pod 被 kill）。
> **务必确保 /dev/shm 为空或有足够剩余内存再启动转换。**

**CPU 并行直转**（`gen_fp4_cache_cpu_parallel.py`）是最快方式：纯 numpy 实现，不需要 TPU/JAX，
使用 ProcessPoolExecutor 12 workers 并行处理 76 层 MoE experts。

---

## 架构对比：GLM-5.1 vs DeepSeek R1

GLM-5.1 与 DeepSeek V3/R1 **~90% 架构相同**（MLA + MoE + SwiGLU），主要差异是超参：

| 参数 | DeepSeek R1 | GLM-5.1 | 影响 |
|------|-------------|---------|------|
| architectures | DeepseekV3ForCausalLM | **GlmMoeDsaForCausalLM** | 模型注册名 |
| hidden_size | 7168 | **6144** | 更窄（所有线性层维度） |
| num_hidden_layers | 61 | **78** | 更深（含 1 层 MTP） |
| MoE 层数 | 58 (layer 3-60) | **76 (layer 3-78)** | 含 MTP layer 78 |
| Dense 前几层 | 3 (layer 0-2) | 3 (layer 0-2) | first_k_dense_replace=3 |
| num_attention_heads | 128 | **64** | attention 并行度减半 |
| q_lora_rank | 1536 | **2048** | MLA Q 压缩维度更大 |
| qk_nope_head_dim | 128 | **192** | 非 RoPE 部分维度更大 |
| v_head_dim | 128 | **256** | Value head 维度 2x |
| kv_lora_rank | 512 | 512 | 相同 |
| qk_rope_head_dim | 64 | 64 | 相同 |
| n_routed_experts | 256 | 256 | 相同 |
| num_experts_per_tok | 8 | 8 | 相同 |
| moe_intermediate_size | 2048 | 2048 | 相同 |
| n_group | 8 | **1** | 不分组路由 |
| rope_theta | 10000 (YaRN) | **1000000** | 位置编码策略不同 |
| rope_interleave | false | **true** | 交错 RoPE 维度排列 |
| vocab_size | 129280 | **154880** | Embedding 更大 |
| num_nextn_predict_layers | 0 | **1** | MTP（Multi-Token Prediction） |
| 总参数量 | ~671B | **~754B** | 更窄更深，总量更大 |

> **关键结论**：量化策略（FP4 MoE + FP8 非 MoE）、cache 生成脚本、vLLM 启动流程与 DeepSeek R1 完全一致。
> 脚本中的维度由数据自动推导，MoE 层范围 3-78（含 MTP layer，共 76 层 MoE）。
> **已验证**：权重 key 命名与 DeepSeek R1 完全一致（`model.layers.{i}.mlp.experts.{j}.gate_proj.weight` / `weight_scale_inv`）。

---

## 硬件与模型概览

### 硬件要求

| 项目 | 要求 |
|------|------|
| TPU | v7x-8（2x2x1 拓扑，4 chips = 8 devices） |
| HBM | 94.75 GB/device，总计 758 GB |
| 主机内存 | ≥940 GB（模型加载 + /dev/shm 缓存，比 DeepSeek R1 需求更高） |
| 存储 | ≥2.0 TB（模型 ~705 GB + FP4 cache ~724 GB） |

### 模型概览

| 参数 | 值 |
|------|-----|
| 模型 | GLM-5.1 754B ([zai-org/GLM-5.1-FP8](https://huggingface.co/zai-org/GLM-5.1-FP8)) |
| 架构名 | `GlmMoeDsaForCausalLM` |
| 架构 | MoE（256 experts, top-8 routing）+ MLA + DSA（Phase 1 不启用）+ MTP（Phase 1 不启用） |
| 总参数量 | ~754B |
| 总层数 | 78（layer 0-77 标准 + layer 78 MTP） |
| MoE 层数 | 76（layer 3-78，含 MTP layer） |
| Dense 前几层 | 3（layer 0-2） |
| FP8 量化 | 128×128 2D block, e4m3, key: `weight_scale_inv` |
| 量化方案 | FP4（float4_e2m1fn）MoE experts + FP8 attention |
| safetensors | 142 文件, 119,028 keys, 705 GB |
| FP4 MoE HBM | ~45.3 GB/device（8 devices 可放下） |
| 非 MoE HBM | ~21 GB/device（replicate，TP=1） |
| 固定开销 | ~71 GB/device（含激活值 + 系统开销） |
| 可用 KV cache | ~24 GB/device |

### HBM 核算（FP4 MoE + FP8 非 MoE，EP=8, TP=1）

| 组件 | GLM-5.1 | DeepSeek R1 | 精度 |
|------|---------|-------------|------|
| MoE Expert（EP=8 分片） | **45.3 GB** | 40.9 GB | FP4 |
| 非 MoE 权重（replicate） | **~21 GB** | ~23 GB | FP8 |
| 激活值峰值 | ~2 GB | ~2 GB | BF16/FP32 |
| 系统开销 | ~3 GB | ~3 GB | - |
| **固定开销小计** | **~71 GB** | ~69 GB | |
| **可用于 KV cache** | **~24 GB** | ~26 GB | |

> **结论**：v7x-8 单节点 FP4 完全放得下。每 device 剩余 ~24 GB 给 KV cache，
> DP Attention 下总 KV 预算 ~188 GB，8K 上下文可支撑 ~274 并发。

---

### MTP Layer（Multi-Token Prediction）

GLM-5.1 包含 1 层 MTP（`num_nextn_predict_layers: 1`），位于 layer 78：

| 组件 | 描述 |
|------|------|
| `model.layers.78.mlp.experts.*` | MTP 层的 MoE experts（256 个，与标准层相同结构） |
| `model.layers.78.mlp.gate` | MTP 层的 Router |
| `model.layers.78.self_attn.*` | MTP 层的 MLA attention |
| `model.enorm` / `model.hnorm` | MTP 专属的 embedding/hidden norm |
| `model.eh_proj` | MTP 专属的 embedding→hidden 投影 |
| `model.layers.78.shared_head.norm` | MTP 层的共享 head norm |

**MTP 工作原理**：训练时同时预测 N+1 和 N+2 token，提升训练效率。推理时可选开启 speculative decoding（用 MTP head 投机预测下一个 token）。

**Phase 1 策略：跳过 MTP**
- vLLM 的 DeepSeek V3/R1 代码已有 MTP skip 逻辑
- 设置 `--num-nextn-predict-layers 0` 或不传此参数即可
- **不影响推理正确性**，仅失去 speculative decoding 加速
- MTP 层的 MoE 权重仍需转 FP4（layer 78 在转换范围内），但推理时不会被加载到 HBM

### DSA（Deep Sparse Attention）

GLM-5.1 独有的稀疏注意力机制，通过 indexer 选择性关注 key tokens：

| Key | Shape | 描述 |
|-----|-------|------|
| `self_attn.indexer.wk` | (6144, 1024) | Indexer key 投影 |
| `self_attn.indexer.wq_b` | (1024, 1024) | Indexer query 投影 |
| `self_attn.indexer.weights_proj` | (64, 4, 1) | Indexer 权重投影 |
| `self_attn.indexer.k_norm` | (1024,) | Indexer key norm |

**Phase 1 策略：跳过 DSA**，使用标准 full attention。DSA indexer 权重为非 MoE 参数，
会被 `extract_non_moe_weights.py` 自动提取到 `non_moe_weights.safetensors`，但推理时不使用。

---

# Part A: GKE + Docker

适用于已有 GKE 集群和 Docker 镜像的场景。与 DeepSeek R1 的 GKE 部署流程完全一致。

## A-1: 创建 GKE TPU Pod

```bash
kubectl apply -f - <<'EOF'
apiVersion: v1
kind: Pod
metadata:
  name: vllm-glm51
spec:
  containers:
  - name: vllm
    image: <YOUR_DOCKER_REGISTRY>/vllm-tpu:latest
    resources:
      limits:
        google.com/tpu: 8
    volumeMounts:
    - name: data
      mountPath: /data
    - name: dshm
      mountPath: /dev/shm
  volumes:
  - name: data
    persistentVolumeClaim:
      claimName: data-pvc
  - name: dshm
    emptyDir:
      medium: Memory
      sizeLimit: 800Gi         # /dev/shm 需要 ≥724 GB
  restartPolicy: Never
  nodeSelector:
    cloud.google.com/gke-tpu-topology: 2x2x1
    cloud.google.com/gke-tpu-accelerator: tpu-v7x-lite-podslice
EOF
```

> **存储需求**：模型 ~705 GB + FP4 cache ~724 GB = ~1.4 TB。推荐 Hyperdisk Extreme 4 TB。

## A-2: 更新 tpu-inference 到 FP4 分支

与 DeepSeek R1 完全一致：

```bash
cd /workspace/tpu_inference
git fetch origin
git checkout feature/moe-fp4-weight-cache
python3 -c "import tpu_inference; print('OK')"
```

完成后跳转到 [Step 3: 下载模型权重](#step-3-下载模型权重)。

---

# Part B: TPU VM 裸机安装

与 DeepSeek R1 裸机安装完全一致，仅存储空间需求更大。

## B-1: 创建 TPU VM + 数据盘

```bash
export PROJECT=<PROJECT_ID>
export ZONE=us-central1-c
export TPU_NAME=my-glm51-vm

gcloud alpha compute tpus queued-resources create ${TPU_NAME}-qr \
  --node-id $TPU_NAME \
  --project $PROJECT \
  --zone $ZONE \
  --accelerator-type tpu7-8 \
  --runtime-version v2-alpha-tpu7-ubuntu2404 \
  --scopes=https://www.googleapis.com/auth/cloud-platform \
  --metadata=enable-oslogin=false \
  --service-account <SA_EMAIL>

# 创建 3TB 数据盘（GLM-5.1 需要比 DeepSeek R1 更多存储）
gcloud compute disks create ${TPU_NAME}-data \
  --project $PROJECT --zone $ZONE \
  --type hyperdisk-balanced \
  --size 3TB \
  --provisioned-iops 40000 \
  --provisioned-throughput 2400

gcloud alpha compute tpus tpu-vm attach-disk $TPU_NAME \
  --disk ${TPU_NAME}-data \
  --zone $ZONE --project $PROJECT --mode read-write

gcloud compute tpus tpu-vm ssh $TPU_NAME --project $PROJECT --zone $ZONE
```

进入 VM 后格式化挂载：

```bash
sudo mkfs.ext4 -m 0 -E lazy_itable_init=0,lazy_journal_init=0 /dev/nvme1n1
sudo mkdir -p /data
sudo mount -o discard,defaults /dev/nvme1n1 /data
sudo chown $USER:$USER /data
echo "/dev/nvme1n1 /data ext4 discard,defaults 0 2" | sudo tee -a /etc/fstab

# 扩大 /dev/shm（需要 ≥800G 放 FP4 cache 724G + non-MoE 21G）
sudo mount -o remount,size=800G /dev/shm
```

## B-2 ~ B-5: 安装环境

与 DeepSeek R1 **完全一致**，参考 [DeepSeek R1 README](../DeepSeek-R1-671B-FP4/README.md) 的 B-2 到 B-5 步骤。

```bash
# 系统依赖
sudo apt-get update && sudo apt-get install -y libopenmpi-dev libomp-dev git curl

# uv + venv
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
uv venv ~/vllm_env --python 3.12
source ~/vllm_env/bin/activate

# tpu-inference（FP4 分支）
cd ~
git clone https://github.com/yangwhale/tpu-inference.git
cd tpu-inference && git checkout feature/moe-fp4-weight-cache

# vLLM
export VLLM_COMMIT_HASH="$(cat .buildkite/vllm_lkg.version | tr -d '[:space:]')"
cd ~ && git clone https://github.com/vllm-project/vllm.git
cd vllm && git checkout "${VLLM_COMMIT_HASH}"
uv pip install -r requirements/tpu.txt --torch-backend=cpu
VLLM_TARGET_DEVICE="tpu" uv pip install -e .
cd ~

# ⚠️ 修复 JAX 版本
uv pip install jax==0.9.2 jaxlib==0.9.2 libtpu==0.0.39 flax==0.12.4

# tpu-inference
cd ~/tpu-inference && uv pip install -e . --no-deps && cd ~
```

设置路径：

```bash
export STORAGE=/data
export TI_DIR=~/tpu-inference
```

---

# 通用步骤（GKE 和 TPU VM 共用）

## Step 3: 下载模型权重

GLM-5.1 FP8 模型约 705 GB（142 safetensors 文件）：

```bash
uv pip install huggingface_hub

# 下载 GLM-5.1 FP8 量化版本
huggingface-cli download zai-org/GLM-5.1-FP8 \
  --local-dir $STORAGE/models/GLM-5.1-FP8

# 验证文件数量
ls $STORAGE/models/GLM-5.1-FP8/*.safetensors | wc -l
# 预期：142
```

> **已验证**：
> - HuggingFace repo: `zai-org/GLM-5.1-FP8`（142 safetensors, 705 GB）
> - safetensors 已包含 FP8 e4m3 量化权重 + 128×128 block scale（`weight_scale_inv`）
> - 权重 key 命名与 DeepSeek R1 完全一致，`gen_fp4_cache_cpu_parallel.py` 可直接使用
> - 下载速度参考：E2E pod（GKE, us-central1）约 6 分钟完成

设置模型路径：

```bash
export MODEL=$STORAGE/models/GLM-5.1-FP8
```

---

## Step 4: 生成 FP4 MoE Cache（推荐：CPU 并行直转）

> **与 DeepSeek R1 完全相同的转换流程**，脚本已适配 GLM-5.1 的 76 层 MoE（layer 3-78，含 MTP layer）。

```bash
source ~/vllm_env/bin/activate

ls $MODEL/*.safetensors | wc -l
df -h $STORAGE                     # 需要 ~724 GB 可用空间

# 启动 CPU 并行 FP4 cache 生成
python3 -u gen_fp4_cache_cpu_parallel.py \
  --model-dir $MODEL \
  --cache-dir $STORAGE/moe-cache/ep8_tp1_gmm_ep_fp4e2m1_bsNone \
  --workers 12
```

> **脚本说明**：`gen_fp4_cache_cpu_parallel.py` 已从 DeepSeek R1 版本适配：
> - MoE 层范围：`range(3, 79)`（76 层，vs DeepSeek R1 的 `range(3, 61)` 58 层）
> - 含 MTP layer 78（虽然 Phase 1 不使用 MTP，但 MoE 权重仍需转换以保持 cache 完整性）
> - 维度由数据自动推导，无需硬编码（hidden_size=6144, moe_intermediate=2048 从 safetensors 读取）
> - FP8 格式已验证：128×128 block 量化，e4m3 dtype，scale key `weight_scale_inv`
> - 其余逻辑（FP8 block dequant → FP32 → FP4 per-channel quant → GMM_EP layout）完全一致

**工作原理**：

```
safetensors (FP8 e4m3fn)
    │
    │  load_layer_experts(): 逐层加载 256 experts 的 gate/up/down 权重
    ▼
FP8 weights (256, 4096, 6144) + block scale (256, 32, 48)
    │
    │  dequant_fp8_blocked(): 128×128 block 反量化
    ▼
FP32 weights (256, 4096, 6144)
    │
    │  quantize_to_fp4(): per-channel 量化（axis=2）
    ▼
FP4 weights → GMM_EP layout (swapaxes) → npy 文件
```

**输出 shapes**（GLM-5.1）：

```
w13_weight:       (256, 6144, 4096) float4_e2m1fn
w13_weight_scale: (256, 1, 1, 4096) float32
w2_weight:        (256, 2048, 6144) float4_e2m1fn
w2_weight_scale:  (256, 1, 1, 6144) float32
```

**验证 1: Shape 和层数检查**

```bash
ls $STORAGE/moe-cache/ep8_tp1_gmm_ep_fp4e2m1_bsNone/ | grep model_layers | wc -l
# 预期：76

python3 -c "
import numpy as np
d = '$STORAGE/moe-cache/ep8_tp1_gmm_ep_fp4e2m1_bsNone/model_layers_3_mlp_experts'
for name in ['w13_weight', 'w13_weight_scale', 'w2_weight', 'w2_weight_scale']:
    a = np.load(f'{d}/{name}.npy')
    print(f'{name}: {a.shape} {a.dtype}')
"
# 预期：
#   w13_weight:       (256, 6144, 4096) |V1    (float4_e2m1fn)
#   w13_weight_scale: (256, 1, 1, 4096) float32
#   w2_weight:        (256, 2048, 6144) |V1
#   w2_weight_scale:  (256, 1, 1, 6144) float32
```

**验证 2: FP4 量化质量抽查**

核心思路：将 FP4 cache 反量化回 FP32，与原始 FP8 反量化的 FP32 对比，检查误差分布。

```bash
python3 validate_weights.py $MODEL $STORAGE/moe-cache/ep8_tp1_gmm_ep_fp4e2m1_bsNone
```

验证脚本 `validate_weights.py` 做两件事：

1. **Non-MoE bit-exact 检查**：从合并的 `non_moe_weights.safetensors` 和原始 safetensors 各取 15 个 key
   （embedding、attention、layernorm、router 等），逐元素比较，应完全一致（未经量化转换）。

2. **FP4 MoE 量化质量检查**：抽查 3 层（early/middle/late）× 3 个 expert（0/127/255），对每个 expert：
   - 原始路径：safetensors FP8 → `dequant_fp8_blocked()` → FP32
   - Cache 路径：npy FP4 → `.view(float4_e2m1fn)` → reverse GMM_EP layout → `.astype(float32)` × scale → FP32
   - 对比指标：max/mean abs error、relative error（mean + p99）、cosine similarity、零值比例、分布统计（min/max/mean/std）

> **注意**：npy 文件中 FP4 存储为 `|V1`（1 字节 void type），需要 `.view(ml_dtypes.float4_e2m1fn)`
> 才能正确转为 float32，直接 `.astype(np.float32)` 会报 ValueError。

**实测验证结果（2026-04-23）**：

| 检查项 | 结果 | 备注 |
|--------|------|------|
| Non-MoE bit-exact | **15/15 PASS** ✅ | embedding/attention/norm/router/lm_head 全部精确匹配 |
| FP4 cosine similarity | **0.992-0.993** ✅ | layer 3 和 layer 40 各 3 个 expert，全部 >0.99 |
| FP4 max abs error | ~0.013-0.017 | FP4 精度范围内，正常 |
| FP4 mean abs error | ~0.0016 | 很小 |
| FP4 零值比例 | ~12-13% | 原始 FP8 近似 0%；FP4 只有 8 个可表示值，小值被量化为 0，属正常行为 |
| FP4 分布 min/max | 与原始一致 | 极值未丢失 |
| Scale 非零率 | 100% | 所有 scale 值均非零 |
| 总 cache 大小 | **756.8 GB** | 76 层 FP4 npy + non-MoE safetensors |

> **量化质量说明**：FP4（float4_e2m1fn）只有 8 个可表示值 `{0, 0.5, 1, 1.5, 2, 3, 4, 6}`（加符号位）。
> ~12% 零值和 cosine similarity 0.993 是 FP4 精度的正常表现，与 DeepSeek R1 实测结果一致。
> Cosine similarity 衡量的是权重矩阵的"方向"相似度（1.0 = 完美还原），
> 比绝对误差更能反映量化对推理结果的影响。>0.99 即为合格。

---

## Step 5: 预拷贝 FP4 Cache + Non-MoE 权重到 /dev/shm

与 DeepSeek R1 完全一致。**强烈建议**，避免 vLLM MoE prefetch deadlock。

### Step 5a: 拷贝 FP4 cache

```bash
df -h /dev/shm
# 需要 ≥800 GB（FP4 cache 735G + non-MoE 21.5G ≈ 757G）

time cp -r $STORAGE/moe-cache/ep8_tp1_gmm_ep_fp4e2m1_bsNone /dev/shm/
export MOE_WEIGHT_CACHE_DIR=/dev/shm
```

> **已验证**：GLM-5.1 FP4 cache 实测 **~735 GB**（76 层 npy），加 non-MoE 21.5 GB = **~757 GB**。
> v7x-8 的 944 GB RAM 够用（/dev/shm 800G，实际占用 757G）。
> 比 DeepSeek R1（~610 GB）大 ~147 GB。

### Step 5b: 提取并拷贝 Non-MoE 权重

```bash
python3 extract_non_moe_weights.py \
  --model-dir $MODEL \
  --output $STORAGE/moe-cache/ep8_tp1_gmm_ep_fp4e2m1_bsNone/non_moe_weights.safetensors

cp $STORAGE/moe-cache/ep8_tp1_gmm_ep_fp4e2m1_bsNone/non_moe_weights.safetensors \
   /dev/shm/ep8_tp1_gmm_ep_fp4e2m1_bsNone/

ls -lh /dev/shm/ep8_tp1_gmm_ep_fp4e2m1_bsNone/non_moe_weights.safetensors
# 预期：~21 GB
```

---

## Step 6: 启动 vLLM 推理服务

> ⚠️ **三个环境变量缺一不可**（与 DeepSeek R1 完全一致）：
>
> | 环境变量 | 值 | 漏设后果 |
> |---------|-----|---------|
> | `MOE_REQUANTIZE_WEIGHT_DTYPE=float4_e2m1fn` | **必须设** | 查找 FP8 目录 → cache miss → OOM |
> | `NEW_MODEL_DESIGN=1` | **必须设** | MLA 模型强制要求 |
> | `MOE_WEIGHT_CACHE_DIR=/dev/shm` | **必须设** | 找不到 FP4 cache |

```bash
export MOE_REQUANTIZE_WEIGHT_DTYPE=float4_e2m1fn
export NEW_MODEL_DESIGN=1
export MOE_WEIGHT_CACHE_DIR=/dev/shm

cd /tmp

python3 -m vllm.entrypoints.openai.api_server \
  --model $MODEL \
  --tensor-parallel-size 8 \
  --quantization fp8 \
  --enforce-eager \
  --enable-prefix-caching \
  --enable-chunked-prefill \
  --max-model-len 4096 \
  --trust-remote-code \
  --additional-config '{
    "sharding": {
      "sharding_strategy": {
        "enable_dp_attention": true,
        "expert_parallelism": 8,
        "tensor_parallelism": 1
      }
    },
    "replicate_attn_weights": "True",
    "sparse_matmul": "True"
  }'
```

等待日志显示 `Application startup complete`。

> **JAX 模型已实现**（branch: `feature/glm51-inference`）：
> - `GlmMoeDsaForCausalLM` 已注册到 `model_loader.py` 的 `_MODEL_REGISTRY`
> - JAX 实现 `glm_moe.py` 基于 `deepseek_v3.py` copy+modify
> - MTP: `load_weights()` 动态跳过 layer 78 权重
> - DSA: indexer 权重通过 `skip_substrs` 过滤，不加载不报错
> - RoPE: 新增 `InterleavedRotaryEmbedding`（theta=1M, even/odd 交错, 无 YaRN）
>
> **待验证**：
> 1. 5 层最小测试 `--hf-overrides '{"num_hidden_layers":5}'` 启动是否正常
> 2. MLA kernel 对新维度（qk_nope=192, v=256）是否兼容
> 3. 全量 78 层推理结果

---

## Step 7: 验证推理

```bash
curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "'"$MODEL"'",
    "messages": [{"role": "user", "content": "What is 2+3? Answer with just the number."}],
    "max_tokens": 256
  }' | python3 -m json.tool

# 健康检查
curl -s http://localhost:8000/health
```

---

## FP4 Cache 磁盘 / SHM 占用

| 组件 | GLM-5.1（实测） | DeepSeek R1（参考） |
|------|---------|---------------------|
| FP4 cache（磁盘） | **~735 GB** | ~610 GB |
| Non-MoE consolidated | **21.5 GB** | ~23 GB |
| 总 cache 大小 | **756.8 GB** | ~633 GB |
| SHM 预加载 | **~757 GB** | ~633 GB |
| Host RAM 需求（SHM + vLLM） | **~930 GB** | ~810 GB |

> GLM-5.1 的 FP4 cache 比 DeepSeek R1 大 ~124 GB（76 层 vs 58 层 MoE）。
> v7x-8 的 944 GB RAM 够用（930 GB 需求），GKE pod memory limit 建议设为 940 GB。

---

## KV Cache 与并发容量

MLA KV cache 计算与 DeepSeek R1 相同（kv_lora_rank=512, rope_dim=64），但 GLM-5.1 层数更多（78 vs 61，Phase 1 跳过 MTP layer 78 则为 77 层）：

| 上下文长度 | 每请求 KV | GLM-5.1 最大并发 | DeepSeek R1 | 计算 |
|-----------|-----------|-----------------|-------------|------|
| 4K | 343 MB | ~548 | ~753 | 188 GB ÷ 343 MB |
| 8K | 686 MB | **~274** | ~377 | 188 GB ÷ 686 MB |
| 16K | 1.37 GB | ~137 | ~188 | 188 GB ÷ 1.37 GB |
| 32K | 2.74 GB | ~69 | ~94 | 188 GB ÷ 2.74 GB |
| 128K | 10.97 GB | ~17 | ~24 | 188 GB ÷ 10.97 GB |

> GLM-5.1 并发约为 DeepSeek R1 的 **~73%**（层数更多 → 每 token KV 更大：87.75 KB vs 68.6 KB）。

---

## 已知风险与待验证项

### 已验证 ✅

1. ~~**HuggingFace 权重格式**~~：**已确认** — `zai-org/GLM-5.1-FP8`，FP8 e4m3 + 128×128 block scale（`weight_scale_inv`），与 DeepSeek R1 格式完全一致，`gen_fp4_cache_cpu_parallel.py` 可直接使用
2. ~~**权重 key 命名**~~：**已确认** — 与 DeepSeek R1 完全一致（`model.layers.{i}.mlp.experts.{j}.gate_proj.weight` / `weight_scale_inv`）
3. ~~**FP4 cache 维度**~~：**已确认** — 脚本的 block dequant 维度由数据自动推导，适配 GLM-5.1 的 (6144, 2048)
4. ~~**FP4 量化质量**~~：**已验证（2026-04-23）** — cosine similarity 0.992-0.993（>0.99 合格），Non-MoE bit-exact 15/15 通过
5. ~~**FP4 cache 生成耗时**~~：**已验证** — CPU 并行（12 workers）28 min，平均 245.5s/层
6. ~~**Non-MoE 提取**~~：**已验证** — 2292 keys，21.54 GB，109 秒
7. ~~**模型注册**~~：**已完成** — `GlmMoeDsaForCausalLM` 注册到 `model_loader.py`，JAX 实现在 `glm_moe.py`（branch: `feature/glm51-inference`）

### 待验证

1. **推理测试**：5 层最小测试 `--hf-overrides '{"num_hidden_layers":5}'` 验证模型加载和基本推理
2. **MLA kernel 维度兼容**：qk_nope=192（非 128 的倍数）是否需要 padding
3. **全量推理**：78 层完整模型启动 + benchmark

### 踩坑记录

所有 DeepSeek R1 的踩坑经验（22 条）均适用于 GLM-5.1，参见 [DeepSeek R1 README](../DeepSeek-R1-671B-FP4/README.md#踩坑记录)。

**GLM-5.1 新增踩坑**：

23. **/dev/shm 占用导致 FP4 cache 生成 OOM**：如果 /dev/shm 有旧数据（如上次推理的 cache），
    会挤占容器内存。12 workers 峰值 ~300 GB，/dev/shm 占 631 GB 时只剩 ~250 GB，直接 OOM kill 整个 pod（exit 137）。
    **解法**：生成 FP4 cache 前确保 /dev/shm 为空，或用新 pod。

24. **FP4 npy 文件的 dtype 是 `|V1` 不是 `float4_e2m1fn`**：numpy 将 ml_dtypes.float4_e2m1fn 存储为
    1 字节 void 类型。加载后必须 `.view(ml_dtypes.float4_e2m1fn)` 才能正确转换为 float32，
    直接 `.astype(np.float32)` 会报 `ValueError: could not convert string to float`。

---

## 参考资料

| 资源 | 链接 / 路径 |
|------|------------|
| GLM-5.1 TPU 推理方案 | [cc.higcp.com/pages/glm51-tpu-inference-plan-20260422.html](https://cc.higcp.com/pages/glm51-tpu-inference-plan-20260422.html) |
| DeepSeek R1 FP4 推理指南 | [../DeepSeek-R1-671B-FP4/README.md](../DeepSeek-R1-671B-FP4/README.md) |
| GLM-5.1 HuggingFace (FP8) | [zai-org/GLM-5.1-FP8](https://huggingface.co/zai-org/GLM-5.1-FP8) |
| GLM-5.1 PR #2324 | vllm-project/tpu-inference#2324 |
| tpu-inference FP4 分支 | [github.com/yangwhale/tpu-inference](https://github.com/yangwhale/tpu-inference) branch: feature/moe-fp4-weight-cache |
| MoE FP4 Cache 优化报告 | [cc.higcp.com/assets/moe-fp4-cache-full-report-20260421.html](https://cc.higcp.com/assets/moe-fp4-cache-full-report-20260421.html) |
