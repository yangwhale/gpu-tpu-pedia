# DeepSeek R1 671B FP4 Inference on TPU v7x-8

> 端到端指南：在单节点 TPU v7x-8 上运行 DeepSeek R1 671B（FP4 量化）推理，
> 包含环境搭建、权重缓存生成、FP4 转换、vLLM 服务启动、以及 GSM8K 准确性验证。
>
> **代码仓库**: https://github.com/yangwhale/tpu-inference (branch: `feature/moe-fp4-weight-cache`)

本文档提供两种部署方式：

| 方式 | 适用场景 | 跳转 |
|------|----------|------|
| **Part A: GKE + Docker** | 生产环境，GKE 集群已有 TPU node pool | [Part A](#part-a-gke--docker) |
| **Part B: TPU VM 裸机** | 开发测试，直接在 TPU VM 上安装运行（**推荐**） | [Part B](#part-b-tpu-vm-裸机安装) |

两种方式共享相同的推理和测试步骤（Step 3-7）。

### MoE Cache 生成策略

有三种方式生成 FP4 MoE cache：

| 方式 | 步骤 | 耗时 | 存储需求 | 推荐 |
|------|------|------|----------|------|
| **CPU 并行直转** ⭐ | safetensors → FP4（纯 CPU, 12 并发） | **~28 min** | 模型 700G + FP4 540G = **~1.2 TB** | ✅ |
| TPU FP4 直转 | safetensors → FP4（TPU JIT, 串行） | ~87 min | 模型 700G + FP4 610G = **~1.3 TB** | 备选 |
| FP8→FP4 两步法 | safetensors → FP8 cache → FP4 转换 | ~60 min | 模型 700G + FP8 404G + FP4 610G = **~1.7 TB** | 备选 |

**CPU 并行直转**（`gen_fp4_cache_cpu_parallel.py`）是最快方式：纯 numpy 实现，不需要 TPU/JAX，
使用 ProcessPoolExecutor 12 workers 并行处理 58 层 MoE experts。
模型从磁盘读取，全部 RAM 留给转换过程的中间 FP32 数组。

TPU FP4 直转通过设置 `MOE_REQUANTIZE_WEIGHT_DTYPE=float4_e2m1fn` 实现，跳过 FP8 中间步骤，节省 ~400 GB 存储。

> **cache 子目录名**由 `_get_config_cache_subdir()` 自动生成，格式 `ep{EP}_tp{TP}_{backend}_{dtype}_bs{block}`。
> 当前验证通过的配置 backend 均为 `gmm_ep`（EP=8, TP=1），子目录名如 `ep8_tp1_gmm_ep_fp4e2m1_bsNone`。

---

## 硬件与模型概览

### 硬件要求

| 项目 | 要求 |
|------|------|
| TPU | v7x-8（2x2x1 拓扑，4 chips = 8 devices） |
| HBM | 94.75 GB/device，总计 758 GB |
| 主机内存 | ≥920 GB（模型加载 + /dev/shm 缓存） |
| 存储 | ≥1.5 TB（模型 700 GB + FP4 cache 610 GB；两步法需额外 404 GB FP8 cache） |

### 模型概览

| 参数 | 值 |
|------|-----|
| 模型 | DeepSeek R1 671B |
| 架构 | MoE（256 experts, top-8 routing）+ MLA |
| 总参数量 | 671B |
| MoE 层数 | 58（layer 3-60） |
| 量化方案 | FP4（float4_e2m1fn）MoE experts + FP8 attention |
| FP4 MoE 显存 | ~60.9 GB/device（8 devices 可放下） |
| FP8 MoE 显存 | ~101.8 GB/device（超出 HBM，必须用 FP4） |

---

# Part A: GKE + Docker

适用于已有 GKE 集群和 Docker 镜像的场景。

## A-1: 创建 GKE TPU Pod

```bash
kubectl apply -f - <<'EOF'
apiVersion: v1
kind: Pod
metadata:
  name: vllm-deepseek-r1
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
      claimName: data-pvc      # 见下方 PVC 创建说明
  - name: dshm
    emptyDir:
      medium: Memory
      sizeLimit: 800Gi         # /dev/shm 需要 ≥610 GB
  restartPolicy: Never
  nodeSelector:
    cloud.google.com/gke-tpu-topology: 2x2x1
    cloud.google.com/gke-tpu-accelerator: tpu-v7x-lite-podslice
EOF
```

存储方案（二选一）：

```bash
# 方案 1: Hyperdisk Extreme PVC（推荐）
kubectl apply -f - <<'EOF'
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: data-pvc
spec:
  accessModes: [ReadWriteOnce]
  storageClassName: hyperdisk-extreme
  resources:
    requests:
      storage: 4Ti
      iops: "20000"
EOF

# 方案 2: 如果集群已有 Lustre，直接用 Lustre PVC
# 把上面 Pod 的 data-pvc 替换为 lustre-pvc，mountPath 改为 /lustre
```

> **存储需求**：模型 ~700 GB + FP4 cache ~610 GB = ~1.3 TB。Hyperdisk Extreme 4 TB + 20K IOPS 提供充足的空间和顺序读写性能。

进入 Pod：
```bash
kubectl exec -it vllm-deepseek-r1 -- bash
```

## A-2: 更新 tpu-inference 到 FP4 分支

Docker 镜像中的 `tpu_inference` 是 editable install，直接切分支即可：

```bash
cd /workspace/tpu_inference
git fetch origin
git checkout feature/moe-fp4-weight-cache
```

验证：
```bash
python3 -c "import tpu_inference; print('OK')"
```

完成后跳转到 [Step 3: 下载模型权重](#step-3-下载模型权重)。

---

# Part B: TPU VM 裸机安装

适用于直接在 TPU VM 上开发测试，不使用 Docker。

## B-1: 创建 TPU VM + 数据盘

```bash
export PROJECT=<PROJECT_ID>
export ZONE=us-central1-c
export TPU_NAME=my-deepseek-vm

# 创建 TPU VM（注意 scopes 和 oslogin）
gcloud alpha compute tpus queued-resources create ${TPU_NAME}-qr \
  --node-id $TPU_NAME \
  --project $PROJECT \
  --zone $ZONE \
  --accelerator-type tpu7-8 \
  --runtime-version v2-alpha-tpu7-ubuntu2404 \
  --scopes=https://www.googleapis.com/auth/cloud-platform \
  --metadata=enable-oslogin=false \
  --service-account <SA_EMAIL>

# 等待状态变为 ACTIVE（约 3-5 分钟）
watch -n 10 gcloud alpha compute tpus queued-resources describe ${TPU_NAME}-qr \
  --project $PROJECT --zone $ZONE --format="value(state.state)"

# 创建 2TB Hyperdisk Balanced 数据盘（顺序读写 2.4 GB/s）
gcloud compute disks create ${TPU_NAME}-data \
  --project $PROJECT --zone $ZONE \
  --type hyperdisk-balanced \
  --size 2TB \
  --provisioned-iops 40000 \
  --provisioned-throughput 2400

# 挂载数据盘到 TPU VM
gcloud alpha compute tpus tpu-vm attach-disk $TPU_NAME \
  --disk ${TPU_NAME}-data \
  --zone $ZONE --project $PROJECT --mode read-write

# SSH 连接
gcloud compute tpus tpu-vm ssh $TPU_NAME --project $PROJECT --zone $ZONE
```

进入 VM 后，格式化并挂载数据盘：

```bash
# 数据盘通常是 /dev/nvme1n1
lsblk | grep nvme

# 格式化 + 挂载
sudo mkfs.ext4 -m 0 -E lazy_itable_init=0,lazy_journal_init=0 /dev/nvme1n1
sudo mkdir -p /data
sudo mount -o discard,defaults /dev/nvme1n1 /data
sudo chown $USER:$USER /data

# 写入 fstab（重启自动挂载）
echo "/dev/nvme1n1 /data ext4 discard,defaults 0 2" | sudo tee -a /etc/fstab

# 扩大 /dev/shm（默认只有内存的一半，不够放 FP4 cache）
sudo mount -o remount,size=800G /dev/shm
df -h /dev/shm
# 预期：800G
```

> **实测数据（TPU v7-8 VM）**
> - TPU VM 创建：~3 分钟（queued-resources）
> - Hyperdisk 格式化：~1 分钟（2TB ext4）
> - 磁盘性能：顺序读写 2416 MiB/s (2.53 GB/s)，随机 4K IOPS 40.3K — 均打满 provisioned 上限

## B-2: 安装系统依赖

```bash
sudo apt-get update && sudo apt-get install -y \
  libopenmpi-dev \
  libomp-dev \
  git \
  curl
```

## B-3: 安装 Python 环境 + vLLM + tpu-inference

推荐使用 `uv`（比 pip 快 10-100x）从源码安装，以便切换到 FP4 分支：

```bash
# 安装 uv
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env

# 创建 Python 3.12 虚拟环境
uv venv ~/vllm_env --python 3.12
source ~/vllm_env/bin/activate

# 克隆 tpu-inference（使用 FP4 分支）
cd ~
git clone https://github.com/yangwhale/tpu-inference.git
cd tpu-inference
git checkout feature/moe-fp4-weight-cache

# 获取 vLLM pinned 版本并克隆（注意 trim 尾部空格）
export VLLM_COMMIT_HASH="$(cat .buildkite/vllm_lkg.version | tr -d '[:space:]')"
cd ~
git clone https://github.com/vllm-project/vllm.git
cd vllm
git checkout "${VLLM_COMMIT_HASH}"

# 安装 vLLM（TPU target）
uv pip install -r requirements/tpu.txt --torch-backend=cpu
VLLM_TARGET_DEVICE="tpu" uv pip install -e .
cd ~

# ⚠️ 关键：修复 JAX 版本（vLLM 安装会降级 JAX 到 0.8.0）
uv pip install jax==0.9.2 jaxlib==0.9.2 libtpu==0.0.39 flax==0.12.4

# 安装 tpu-inference（--no-deps 避免再次降级 JAX）
cd ~/tpu-inference
uv pip install -e . --no-deps
cd ~
```

> **为什么从源码安装？**
> 因为 FP4 MoE cache 的改动在 `feature/moe-fp4-weight-cache` 分支，
> 尚未合入 PyPI 的 `vllm-tpu` 包。从源码安装可以直接使用该分支的代码。

## B-4: 验证安装

```bash
source ~/vllm_env/bin/activate

python3 -c '
import jax
import importlib.metadata
from vllm.platforms import current_platform

print(f"vllm: {importlib.metadata.version(\"vllm\")}")
print(f"tpu_inference: {importlib.metadata.version(\"tpu_inference\")}")
print(f"jax: {jax.__version__}")
print(f"platform: {current_platform.get_device_name()}")
print(f"devices: {len(jax.devices())} x {jax.devices()[0].platform}")
'
```

实测输出：
```
vllm: 0.19.1rc1.dev321+g6dc949140.tpu
tpu_inference: 0.0.0
jax: 0.9.2
platform: TPU V7X
devices: 8 x tpu
```

> **安装耗时实测**：系统依赖 ~1 分钟，uv + venv ~10 秒，vLLM TPU requirements ~3 分钟，
> vLLM editable install ~3 秒，tpu-inference editable install ~2 秒。总计 ~5 分钟。

## B-5: 设置存储路径

数据盘已在 B-1 步骤中挂载到 `/data`，直接使用：

```bash
export STORAGE=/data
export TI_DIR=~/tpu-inference
```

> **存储规划**（2TB Hyperdisk Balanced）
>
> | 内容 | 大小 | 路径 |
> |------|------|------|
> | 模型权重 | ~700 GB | `/data/models/DeepSeek-R1` |
> | FP4 cache | ~610 GB | `/data/moe-cache/ep8_tp1_gmm_ep_fp4e2m1_bsNone` |
> | **合计** | **~1.3 TB** | 2 TB 盘剩余 ~700 GB |
>
> 注意：使用 FP4 直转时不需要 FP8 中间 cache。两步法额外需要 ~404 GB。

完成后继续 Step 3。

---

# 通用步骤（GKE 和 TPU VM 共用）

以下步骤在两种部署方式中通用。路径变量说明：
- **GKE (Part A)**：`$STORAGE` 取决于挂载的存储类型，代码在 `/workspace/tpu_inference`
- **TPU VM (Part B)**：`$STORAGE` 取决于 B-5 的选择，代码在 `~/tpu-inference`

```bash
# 设置通用变量（根据实际情况调整）
# GKE（按实际挂载选一个）:
export STORAGE=/data       # Hyperdisk Extreme PVC
# export STORAGE=/lustre   # Lustre PVC
# export STORAGE=/gcs      # GCS FUSE
export TI_DIR=/workspace/tpu_inference

# TPU VM:
export STORAGE=/data       # Hyperdisk（B-1 挂载）
export TI_DIR=~/tpu-inference
```

## Step 3: 下载模型权重

模型权重约 700 GB：

```bash
# 安装 huggingface_hub（如果未安装）
pip install huggingface_hub

# 下载 DeepSeek R1
huggingface-cli download deepseek-ai/DeepSeek-R1 \
  --local-dir $STORAGE/models/DeepSeek-R1

# 验证文件完整性
ls $STORAGE/models/DeepSeek-R1/*.safetensors | wc -l
# 预期：163 个 safetensors 分片
```

设置模型路径（后续步骤都会用到）：

```bash
export MODEL=$STORAGE/models/DeepSeek-R1
```

---

## Step 4: 生成 FP4 MoE Cache（推荐：CPU 并行直转）

> **推荐方案**：纯 CPU numpy 实现（`gen_fp4_cache_cpu_parallel.py`），无需 TPU/JAX，
> 12 workers 并行，58 层仅需 **~28 min**。直接从 safetensors 转 FP4，不需要中间 FP8 cache。

```bash
# 确保 vllm_env 中有 ml_dtypes、safetensors、torch
source ~/vllm_env/bin/activate

# 确认模型和存储
ls $MODEL/*.safetensors | wc -l   # 预期: 163
df -h $STORAGE                     # 需要 ~620 GB 可用空间

# 启动 CPU 并行 FP4 cache 生成
# --workers: 并行数，根据可用 RAM 调整（每 worker peak ~70 GB）
# 例: 944 GB RAM 的 v7x-8 机器可开 12 workers
cd /tmp
python3 -u gen_fp4_cache_cpu_parallel.py \
  --model-dir $MODEL \
  --cache-dir $STORAGE/moe-cache/ep8_tp1_gmm_ep_fp4e2m1_bsNone \
  --workers 12

# 中断后可续跑（自动跳过已完成的层，无需 --force）
```

**工作原理**：

```
safetensors (FP8 e4m3fn)
    │
    │  load_layer_experts(): 逐层加载 256 experts 的 gate/up/down 权重
    ▼
FP8 weights (256, 4096, 7168) + block scale (256, 32, 56)
    │
    │  dequant_fp8_blocked(): 128×128 block 反量化（reshape → broadcast multiply）
    ▼
FP32 weights (256, 4096, 7168)
    │
    │  quantize_to_fp4(): per-channel 量化（axis=2, abs_max / fp4_max）
    ▼
FP4 weights (256, 4096, 7168) + scale (256, 4096, 1)
    │
    │  GMM_EP layout: swapaxes(1,2) + expand_dims
    ▼
FP4 cache npy:
  w13_weight:       (256, 7168, 4096) float4_e2m1fn
  w13_weight_scale: (256, 1, 1, 4096) float32
  w2_weight:        (256, 2048, 7168) float4_e2m1fn
  w2_weight_scale:  (256, 1, 1, 7168) float32
```

**关键设计决策**：

| 决策 | 原因 |
|------|------|
| ProcessPoolExecutor（非 Thread） | 进程退出时 OS 完全回收内存，避免 glibc malloc arena 碎片累积 OOM |
| max_tasks_per_child=1 | 每层处理完后杀掉 worker 进程重建，确保内存不泄漏 |
| in-place numpy 操作 | `w_fp32 *= scale_inv`, `np.clip(..., out=)` 避免分配大临时数组 |
| w13/w2 串行处理 | 先完成 w13 全流程并释放内存，再处理 w2，降低 peak 从 ~130GB 到 ~70GB/worker |
| 模型从磁盘读（非 /dev/shm） | 释放 800G RAM 给 workers，load 增加 ~20s/层但可开更多并发 |

**workers 数量选择**：

| 机器 RAM | 推荐 workers | 依据 |
|----------|-------------|------|
| 944 GB（v7x-8） | 12 | 800G available ÷ 70G peak/worker |
| 500 GB | 6 | 留 80G 给系统 |
| 250 GB | 3 | 最低配置 |

验证生成结果：
```bash
# 应有 58 个 layer 目录
ls $STORAGE/moe-cache/ep8_tp1_gmm_ep_fp4e2m1_bsNone/ | grep model_layers | wc -l
# 预期：58

# 验证 shape
python3 -c "
import numpy as np
d = '$STORAGE/moe-cache/ep8_tp1_gmm_ep_fp4e2m1_bsNone/model_layers_3_mlp_experts'
for name in ['w13_weight', 'w13_weight_scale', 'w2_weight', 'w2_weight_scale']:
    a = np.load(f'{d}/{name}.npy')
    print(f'{name}: {a.shape} {a.dtype}')
"
# 预期：
#   w13_weight:       (256, 7168, 4096) |V1    (float4_e2m1fn)
#   w13_weight_scale: (256, 1, 1, 4096) float32
#   w2_weight:        (256, 2048, 7168) |V1
#   w2_weight_scale:  (256, 1, 1, 7168) float32
```

---

## Step 5: 预拷贝 FP4 Cache 到 /dev/shm（强烈推荐）

**强烈建议**将 cache 拷贝到 `/dev/shm`（tmpfs）。除了加速 21x，还能**避免 vLLM 的 MoE prefetch deadlock**（见[踩坑 #13](#13-vllm-moe-prefetch-deadlock-cache-hit-路径)）。从磁盘加载时 prefetch 线程长时间占用 semaphore，可能导致全部线程死锁：

```bash
# FP4 cache 约 610 GB，需要 /dev/shm 有足够空间
df -h /dev/shm
# TPU VM 默认 /dev/shm = 主机内存的一半（v7x-8 约 473 GB）
# GKE Pod 可配置 emptyDir.sizeLimit: 800Gi

# 如果空间足够：
time cp -r $STORAGE/moe-cache/ep8_tp1_gmm_ep_fp4e2m1_bsNone /dev/shm/
# 约 12 分钟

# 后续启动时指向 /dev/shm
export MOE_WEIGHT_CACHE_DIR=/dev/shm
```

> **TPU VM 注意**：默认 /dev/shm 约 473 GB，放不下 610 GB FP4 cache。
> 可通过 `sudo mount -o remount,size=800G /dev/shm` 扩大，但会压缩可用内存。
> 如果 shm 空间不够，直接从磁盘加载也可以正常工作（慢一些）。

---

### （备选）Step 4-alt-1: vLLM TPU 在线 FP4 直转

<details>
<summary>展开 vLLM 在线 FP4 直转步骤（~45 min，需要 TPU）</summary>

首次启动 vLLM 时，系统自动从 safetensors 提取 MoE 权重并生成 FP4 缓存。
较慢（~45 分钟），但不需要额外脚本。

```bash
export MOE_WEIGHT_CACHE_DIR=$STORAGE/moe-cache
export MOE_REQUANTIZE_WEIGHT_DTYPE=float4_e2m1fn
export NEW_MODEL_DESIGN=1

cd /tmp
python3 -m vllm.entrypoints.openai.api_server \
  --model $MODEL \
  --tensor-parallel-size 8 \
  --quantization fp8 \
  --enforce-eager \
  --max-model-len 128 \
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

等待日志出现 `Application startup complete`，然后 `Ctrl+C` 停止。
Cache 在后台异步写入磁盘，等待全部完成后再继续。

> **注意**：子目录名（如 `ep8_tp1_gmm_ep_fp4e2m1_bsNone`）由 EP/TP/backend/dtype 配置自动决定。
> 如果 cache 不足 58 层，用 `gen_missing_fp4_cache.py` 补齐（见 [FAQ](#q-cache-生成不足-58-层怎么办)）。

</details>

---

### （备选）Step 4-alt-2: FP8→FP4 两步法

如果 FP4 直转遇到问题，可以先生成 FP8 cache 再离线转换。需要额外 ~404 GB 存储。

<details>
<summary>展开两步法详细步骤</summary>

#### Step 4a: 生成 FP8 Cache

```bash
export MOE_WEIGHT_CACHE_DIR=$STORAGE/moe-cache
export MOE_REQUANTIZE_WEIGHT_DTYPE=float8_e4m3fn
export NEW_MODEL_DESIGN=1

cd /tmp
python3 -m vllm.entrypoints.openai.api_server \
  --model $MODEL \
  --tensor-parallel-size 8 \
  --quantization fp8 \
  --enforce-eager \
  --max-model-len 128 \
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

验证 58 层 FP8 cache 已生成：
```bash
ls $MOE_WEIGHT_CACHE_DIR/ep8_tp1_gmm_ep_fp8e4m3_bsNone/ | grep model_layers | wc -l
# 预期：58
```

#### Step 4b: FP8 → FP4 离线转换

FP4 将 MoE 权重体积减半。纯 CPU 运算，无需 TPU。

```bash
python3 $TI_DIR/scripts/convert_fp8_to_fp4.py \
  --fp8-cache $MOE_WEIGHT_CACHE_DIR/ep8_tp1_gmm_ep_fp8e4m3_bsNone \
  --fp4-cache $MOE_WEIGHT_CACHE_DIR/ep8_tp1_gmm_ep_fp4e2m1_bsNone \
  --workers 10
```

预期：~30 分钟（10 workers）。

</details>

---

## Step 6: 启动 vLLM 推理服务

```bash
export MOE_REQUANTIZE_WEIGHT_DTYPE=float4_e2m1fn
export NEW_MODEL_DESIGN=1
# MOE_WEIGHT_CACHE_DIR 沿用 Step 4/5 的设置

# 重要：cd 到非 ~/vllm 的目录，避免 Python namespace 冲突
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

等待日志显示：
```
INFO:     Uvicorn running on http://0.0.0.0:8000
```

### 关键参数说明

| 参数 | 说明 |
|------|------|
| `--tensor-parallel-size 8` | 使用全部 8 个 TPU devices |
| `--quantization fp8` | 启用 FP8 量化 schema（MoE 部分实际用 FP4） |
| `--enforce-eager` | Eager 模式，避免 XLA tracing 开销 |
| `--enable-prefix-caching` | 启用 KV cache 前缀复用 |
| `--enable-chunked-prefill` | 分块预填充 |
| `--max-model-len 4096` | 最大序列长度 |
| `expert_parallelism: 8` | EP=8，每个 device 处理 32 experts |
| `tensor_parallelism: 1` | TP=1（attention 用 DP 代替） |
| `sparse_matmul: True` | 稀疏矩阵乘法优化 |

---

## Step 7: 验证推理

在另一个终端发送测试请求：

```bash
# 简单测试
curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "'"$MODEL"'",
    "messages": [{"role": "user", "content": "What is 2+3? Answer with just the number."}],
    "max_tokens": 256
  }' | python3 -m json.tool
```

预期 DeepSeek R1 会展示思维链推理过程，最终给出正确答案。

```bash
# 健康检查
curl -s http://localhost:8000/health
# 预期：{"status":"ok"}

# 查看模型信息
curl -s http://localhost:8000/v1/models | python3 -m json.tool
```

---

## Step 8: GSM8K 准确性测试

GSM8K (Grade School Math 8K) 是评估数学推理能力的标准 benchmark。

### 方法 1：使用 tpu_inference 集成测试脚本

```bash
# 确保 vLLM 服务正在运行（Step 6）

# 安装 lm_eval
pip install lm_eval

# 运行准确性测试
python -m pytest -rP \
  $TI_DIR/scripts/vllm/integration/test_accuracy.py::test_lm_eval_accuracy_v1_engine \
  --tensor-parallel-size=8 \
  --model-name=$MODEL \
  --expected-value=0.75
```

参数说明：
- `--expected-value=0.75`：预期 GSM8K exact_match 准确率 ≥75%（3% 容差）
- 如果准确率低于 `expected_value - 0.03`，测试 FAIL

### 方法 2：使用 test_accuracy.sh 一键脚本

```bash
export TEST_MODEL=$MODEL
export TENSOR_PARALLEL_SIZE=8
export MINIMUM_ACCURACY_THRESHOLD=0.75

bash $TI_DIR/tests/e2e/benchmarking/test_accuracy.sh \
  -r $(dirname $TI_DIR)
```

### 方法 3：直接调用 lm_eval

```bash
lm_eval \
  --model vllm \
  --model_args "pretrained=$MODEL,tensor_parallel_size=8,max_model_len=2048,max_num_seqs=64" \
  --tasks gsm8k \
  --batch_size auto
```

输出示例：
```
|Tasks|Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr|
|-----|------:|----------------|-----:|-----------|---|-----:|---|-----:|
|gsm8k|      3|exact_match     |     5|exact_match|↑  |0.7950|±  |0.0111|
|     |       |strict-match    |     5|exact_match|↑  |0.7950|±  |0.0111|
```

---

## 性能数据

以下数据在 TPU v7x-8 单节点（chrisya-tpu7-8-02）上实测。

### 权重加载时间

| 阶段 | 耗时 | 说明 |
|------|------|------|
| vLLM 初始化 | ~30s | 模型配置解析 + JAX 初始化 |
| Safetensors 非 MoE 权重 | ~6:00 | 70/163 shards（93 MoE shards 被 cache 跳过） |
| MoE Cache → TPU（/dev/shm） | **~0:42** | 58 层 FP4 cache, tmpfs zero-copy mmap, ~0.7s/层 |
| **总计（/dev/shm）** | **~7:00** | 从 Docker 启动到 Application startup complete |

### MoE Cache 大小

| 格式 | 磁盘大小 | HBM 占用/device | 说明 |
|------|----------|-----------------|------|
| FP8 cache | ~404 GB | ~101.8 GB（超 HBM） | 58 层 × ~7 GB/层 |
| FP4 cache | ~610 GB | ~60.9 GB | 58 层 × ~10.5 GB/层（含 FP32 scale） |

> **注意**：FP4 cache 文件比 FP8 大，因为 FP4 原生存储需要额外的 scale 精度（FP32 scale）。
> 但加载到 TPU HBM 后，FP4 权重占用的显存只有 FP8 的一半。
> 使用 FP4 直转时不需要 FP8 cache。

### FP4 Cache 生成

| 方式 | 耗时 | 每层 | 并发 | 说明 |
|------|------|------|------|------|
| **gen_fp4_cache_cpu_parallel.py** ⭐ | **~28 min** | ~324s | 12 | 纯 CPU numpy，ProcessPoolExecutor，无需 TPU |
| gen_fp4_cache_optimized.py | ~87 min | ~87s | 1 | TPU JIT，prefetch + async save |
| vLLM 在线生成 | ~5 h | ~5 min | 1 | safetensors 全量加载 + requant |
| FP8→FP4 两步法 | ~60 min | - | 1 | FP8 生成 ~30 min + FP4 转换 ~30 min |

> **CPU 并行 vs TPU 串行**：CPU 方案单层慢 ~4 倍（324s vs 87s），但 12 workers 并行总时间快 3 倍。
> 关键约束是内存：每 worker peak ~70 GB FP32 中间数组，需要 `max_tasks_per_child=1` 防止内存碎片。
> 模型从磁盘读取（非 /dev/shm），load 约 7-55s/层（取决于 page cache），全部 RAM 留给计算。

### FP4 Cache 拷贝到 /dev/shm

| 项目 | 数据 |
|------|------|
| 拷贝时间 | ~12 分钟 |
| 数据量 | 610 GB |
| 吞吐 | ~850 MB/s（PD → tmpfs） |

### GSM8K 准确性

| Metric | Score | Stderr |
|--------|-------|--------|
| flexible-extract | **94.92%** | ±0.60% |
| strict-match | **94.84%** | ±0.61% |

> FP4 量化精度损失极小，远超 75% 阈值。
> 测试用时 ~23 分钟，1319 题，batch_size=16。

---

## 环境变量参考

| 变量 | 说明 | 示例值 |
|------|------|--------|
| `MOE_WEIGHT_CACHE_DIR` | MoE 权重缓存根目录 | `$STORAGE/moe-cache` |
| `MOE_REQUANTIZE_WEIGHT_DTYPE` | MoE 目标量化类型 | `float4_e2m1fn` |
| `NEW_MODEL_DESIGN` | 启用 MLA 模型设计 | `1` |
| `MOE_REQUANTIZE_BLOCK_SIZE` | 量化块大小 | `512`（可选） |
| `MOE_PARALLEL_WORKERS` | 并行 requant 线程数 | `1`（默认） |

---

## FAQ

### Q: 为什么不直接用 FP8，要转 FP4？

FP8 MoE 权重每 device 需要 ~101.8 GB HBM，超出 v7x 单 device 的 94.75 GB 限制。
FP4 将 MoE 权重减半到 ~60.9 GB/device，加上 attention 权重后总计 ~70 GB/device，留出足够的 KV cache 空间。

### Q: FP4 转换会损失精度吗？

转换使用 dequant→rescale→requant 流程（而非简单截断）：
1. FP8 × scale → FP32（恢复真实值）
2. 计算新的 per-channel scale = abs_max / 6.0（FP4 最大值）
3. FP32 / new_scale → clip → FP4

GSM8K 测试可验证精度是否在可接受范围内。

### Q: 首次启动后可以删除 FP8 cache 吗？

可以。FP4 cache 已包含所有需要的信息。但建议保留以备回退。

### Q: TPU VM 的 /dev/shm 不够大怎么办？

TPU VM 默认 /dev/shm = 主机内存一半（v7x-8 约 473 GB），放不下 610 GB FP4 cache。两种解决方式：
1. `sudo mount -o remount,size=800G /dev/shm` 扩大（会压缩可用内存）
2. 直接从磁盘加载（不用 shm），每层加载时间从 ~1.3s 变为 ~28s，总共慢约 25 分钟

### Q: GKE 和 TPU VM 有什么区别？

| 对比 | GKE + Docker | TPU VM 裸机 |
|------|-------------|-------------|
| 环境隔离 | Docker 容器，环境一致 | 直接用系统 Python |
| /dev/shm | 可配 `emptyDir.sizeLimit: 800Gi` | 默认 473 GB，需手动扩大 |
| 存储 | Lustre PVC 或 PD | GCS/PD/Lustre 自行挂载 |
| 适合场景 | 生产部署 | 开发调试、快速测试 |

### Q: PyPI 安装和源码安装怎么选？

- **PyPI (`pip install vllm-tpu`)**：最简单，但 FP4 分支改动尚未发布到 PyPI
- **源码安装**：当前唯一方式，因为需要 `feature/moe-fp4-weight-cache` 分支的改动
- FP4 合入主线后，可直接 `pip install vllm-tpu` 使用

### Q: 裸机（TPU VM）上能直接跑吗？

**可以。** 裸机和 Docker 都已验证可用。关键注意事项：

1. **cd /tmp**：必须从非 `~/vllm` 的目录启动，否则 Python 会把 `~/vllm/` 当作 namespace package
2. **JAX 版本**：安装 vLLM 后必须手动升级 JAX 到 0.9.2（见[踩坑 #1](#1-jax-版本被-requirementstputxt-降级)）
3. **`from vllm import SamplingParams` 会报错**：这是循环导入 bug，但不影响 `python3 -m vllm.entrypoints.openai.api_server`

### Q: Cache 生成不足 58 层怎么办？

vLLM 进程可能提前退出或 async cache 写入未完成。已生成的 cache 是完好的，用 `gen_missing_fp4_cache.py` 补齐：

**关键：权重不能转置，保持 safetensors 原始 shape。** `process_fp8_moe_weights` 内部 convention：
- w13 = `(E, 2*intermediate, hidden)` = (256, 4096, 7168) — gate+up 在 axis=1 concat
- w2 = `(E, hidden, intermediate)` = (256, 7168, 2048)

```python
# gen_missing_fp4_cache.py — 关键部分
os.environ["MOE_REQUANTIZE_WEIGHT_DTYPE"] = "float4_e2m1fn"  # 必须在 import 前设置

# 不要转置！safetensors 的原始 layout 就是正确的：
gate_stack = torch.stack([gate_w[i] for i in range(E)])   # (E, 2048, 7168)
up_stack = torch.stack([up_w[i] for i in range(E)])       # (E, 2048, 7168)
w13_weight = torch.cat([gate_stack, up_stack], dim=1)     # (E, 4096, 7168)
w2_weight = torch.stack([down_w[i] for i in range(E)])    # (E, 7168, 2048)
```

完整脚本见本目录下的 [`gen_fp4_cache_optimized.py`](gen_fp4_cache_optimized.py)（带 prefetch + async save 优化）。

```bash
# 生成 GMM_EP 格式 cache（默认写入 /dev/shm）
cd /tmp && source ~/vllm_env/bin/activate
python3 gen_fp4_cache_optimized.py --backend gmm_ep

# 指定自定义路径
python3 gen_fp4_cache_optimized.py --backend gmm_ep --cache-dir /data/moe-cache/ep8_tp1_gmm_ep_fp4e2m1_bsNone

# 强制重新生成所有层
python3 gen_fp4_cache_optimized.py --force
```

旧版 [`gen_missing_fp4_cache.py`](gen_missing_fp4_cache.py) 仍可用（串行版，用于参考）。

### Q: Docker 方式还能用吗？

**可以用。** Docker 适合需要环境隔离的场景。命令示例：

```bash
# 构建 Docker 镜像
cd ~/tpu-inference
export VLLM_COMMIT_HASH="$(cat .buildkite/vllm_lkg.version | tr -d '[:space:]')"
sudo docker build -t tpu-inference:fp8-gen \
  --build-arg VLLM_COMMIT_HASH=$VLLM_COMMIT_HASH .

# Docker FP4 直转
sudo docker run --rm --privileged --net=host \
  --shm-size=800g \
  -v /data:/data \
  -e MOE_WEIGHT_CACHE_DIR=/data/moe-cache \
  -e MOE_REQUANTIZE_WEIGHT_DTYPE=float4_e2m1fn \
  -e NEW_MODEL_DESIGN=1 \
  tpu-inference:fp8-gen \
  python3 -m vllm.entrypoints.openai.api_server \
    --model /data/models/DeepSeek-R1 \
    --tensor-parallel-size 8 --quantization fp8 --enforce-eager \
    --max-model-len 128 --trust-remote-code \
    --additional-config '{"sharding":{"sharding_strategy":{"enable_dp_attention":true,"expert_parallelism":8,"tensor_parallelism":1}},"replicate_attn_weights":"True","sparse_matmul":"True"}'
```

> **注意**：Docker 需要 `--net=host`（TPU runtime 固定端口）和 `--privileged`（/dev/vfio 访问）。

---

## 踩坑记录

### 1. JAX 版本被 requirements/tpu.txt 降级

**现象**：`uv pip install -e .`（vLLM）后 JAX 从 0.9.2 降到 0.8.0，TPU v7 初始化失败（`No ba16c7433 device found`）

**原因**：`requirements/tpu.txt` 依赖 `tpu-inference==0.12.0`（PyPI），它拉入旧版 JAX 0.8.0

**修复**：
```bash
uv pip install jax==0.9.2 jaxlib==0.9.2 libtpu==0.0.39 flax==0.12.4
```
**教训**：每次 `pip install -e .` 后务必检查 `python3 -c "import jax; print(jax.__version__)"`

### 2. flax 版本不兼容

**现象**：`cannot import name 'mutable_array' from 'jax._src.core'`

**原因**：旧 flax（0.10.x/0.11.x）使用了 JAX 0.9.2 中已移除的内部 API

**修复**：`uv pip install flax==0.12.4`（与 Docker 镜像一致）

### 3. `_load_weights` shape mismatch（上游 Bug）

**现象**：`weights.shape=(256, 2048, 7168) param.value.shape=(256, 7168, 2048)`

**原因**：`jax_array_from_reshaped_torch` 只对 2D tensor 做自动转置（`permute_dims=(1,0)`）。添加 expert dim `reshape_dims=(1,) + shape` 后变成 3D，跳过了转置逻辑

**影响**：首次 FP8 cache 生成的 Docker 进程在处理完 54/58 层 MoE cache 后，加载非 MoE 权重时触发 shape mismatch 崩溃

**workaround**：
- FP8 cache 生成用 Docker（cache 在 MoE 处理阶段生成，不受此 bug 影响）
- 崩溃后缺失的 4 层用 `gen_missing_cache.py` 补齐

### 4. 权重转置方向

**现象**：`quantize_tensor` 报 `axis=2 of tensor.shape=(256, 7168, 6144) is not divisible by block=4096`

**原因**：手动拼接 MoE 权重时错误地对 gate/up 做了 `.T`（转置），导致 w13 shape 从正确的 `(E, 4096, 7168)` 变成 `(E, 7168, 4096)`。`quantize_moe_weights` 的 padding 逻辑对反转的维度进行了错误对齐

**教训**：safetensors 里的权重格式（gate/up: `(intermediate, hidden)`，down: `(hidden, intermediate)`）已经匹配 `process_fp8_moe_weights` 的输入 convention。**不需要任何转置**。关键代码注释在 `fp8.py:1013-1016`

### 5. Docker 必须 `--net=host`

**现象**：`Failed to connect to [::]:8353`（TPU runtime 端口）

**原因**：TPU runtime 使用固定端口通信，Docker 默认网络隔离阻断了连接

**修复**：`docker run --privileged --net=host`

### 6. Docker 写入的文件 owner 是 root

**现象**：Docker 内生成的 FP8 cache 文件 owner 是 root，裸机 Python 无法写入同目录

**修复**：`sudo chown -R $USER:$USER /data/moe-cache/`

### 7. vLLM namespace package 冲突

**现象**：`vllm.__path__` 指向 `~/vllm/`（git repo 根目录）而非 site-packages 的 editable install

**原因**：Python 从 `~/` 启动时，`~/vllm/` 被识别为 namespace package，覆盖了正确的 import 路径

**修复**：启动 vLLM 前 `cd /tmp`（或任何不含 `vllm/` 子目录的位置）

### 8. libtpu lockfile 残留

**现象**：`ABORTED: Internal error when accessing libtpu multi-process lockfile`

**原因**：上一个 JAX 进程异常退出，lockfile 未清理

**修复**：`sudo rm /tmp/libtpu_lockfile`

### 9. TPU device busy

**现象**：`TPU initialization failed: open(/dev/vfio/4): Device or resource busy`

**原因**：另一个进程（Docker 容器或 Python 进程）正在占用 TPU

**修复**：`sudo docker stop <container>` 或 `kill <pid>`，然后 `sudo rm /tmp/libtpu_lockfile`

### 10. Async cache 写入不完整

**现象**：`Application startup complete` 后 cache 目录只有 15/58 层，且不再增长

**原因**：vLLM 的 MoE cache 写入是异步的（后台线程）。如果进程收到推理请求或被 `Ctrl+C` 中断，async writer 可能被阻塞或提前终止

**修复**：
- 等 `Application startup complete` 后，**不要立刻发推理请求**，先等 cache 写完
- 用 `ls ... | wc -l` 监控直到 58 层全部到齐
- 如果进程已退出导致 cache 不全，用 `gen_missing_fp4_cache.py` 补齐

### 11. max-model-len 128 太小无法用 Chat API

**现象**：Chat Completions API 返回 `maximum context length is 128 tokens ... your prompt contains 62 characters`

**原因**：`--max-model-len 128` 主要用于 cache 生成（最小化 KV cache 内存占用）。Chat API 的 chat template 会添加系统 tokens，很容易超过 128

**修复**：cache 生成用 `--max-model-len 128`，正式推理用 `--max-model-len 4096` 或更大

### 12. FP4 直转不触发 shape mismatch bug

**发现**：FP4 直转（`MOE_REQUANTIZE_WEIGHT_DTYPE=float4_e2m1fn`）时全部 58 层（包括 57-60）都能正常处理，不触发踩坑 #3 中的 `_load_weights` shape mismatch bug

**原因**：FP4 直转在 `process_weights_after_loading` 中执行 requantization，此时权重已经正确 reshape。Shape mismatch 发生在后续 `_load_weights` 将 non-MoE 权重加载到 JAX array 时，而 FP4 直转的代码路径在此之前就完成了 MoE cache 保存

**意义**：这是 FP4 直转优于两步法的另一个理由 — 不仅更简单，还绕开了已知 bug

### 13. vLLM MoE prefetch deadlock（cache hit 路径）

**现象**：从磁盘加载已有 FP4 cache 时，vLLM 在加载第 8 层后永久挂起。所有 1600+ 线程阻塞在 `futex_wait_queue`，RSS 不增长。三次复现。

**原因**：`fp8.py` 的 prefetch 机制使用 `Semaphore(_MOE_PREFETCH_AHEAD=8)` 控制并发：
1. `_start_moe_prefetch()` 将所有 58 个 cache 文件提交到线程池
2. 前 8 个 prefetch 线程 acquire semaphore 成功，加载 cache
3. 后续线程阻塞在 `acquire()` 等待 permit 释放
4. `_get_prefetched_cache()` 在主线程消费后才 release()
5. 但主线程在 `block_until_ready()` 等 TPU DMA，无法调用 `_get_prefetched_cache()`
6. → 经典循环等待 deadlock

```
prefetch threads → acquire() → blocked (8 permits exhausted)
        ↑                              ↓
main thread → block_until_ready() → can't call _get_prefetched_cache() → no release()
```

**修复**：将 cache 放在 `/dev/shm`（tmpfs），prefetch 线程秒完不卡 semaphore，打破死锁。或用独立 gen 脚本直接生成到 `/dev/shm`

### 14. NEW_MODEL_DESIGN=1 必填

**现象**：不设 `NEW_MODEL_DESIGN=1` 时 vLLM 启动报错：`MLA models require both the NEW_MODEL_DESIGN=1 environment variable...`

**原因**：vLLM 代码更新，MLA 架构模型（DeepSeek V3/R1）强制要求 `NEW_MODEL_DESIGN=1` + DP attention

**修复**：设置 `NEW_MODEL_DESIGN=1` 并加 `--additional_config` 的 DP attention 参数。Backend 仍为 `GMM_EP`（EP=8, TP=1）

### 15. EngineCore 孤儿进程持续占用 TPU

**现象**：kill vLLM 主进程后，TPU 仍然 "device busy"。`fuser /dev/vfio/*` 显示 EngineCore 子进程存活

**原因**：vLLM V1 engine 通过 `multiprocessing` spawn EngineCore 子进程。`kill` 主进程不一定会发信号给子进程

**修复**：
```bash
# 查找并杀死 EngineCore
ps aux | grep EngineCore | grep -v grep
sudo fuser -k /dev/vfio/*
# 清理 lockfile
sudo rm /tmp/libtpu_lockfile
```

**教训**：每次停止 vLLM 后，必须检查并清理 EngineCore 孤儿进程和 lockfile

### 16. gen 脚本 async save 导致 HBM 泄漏

**现象**：`gen_fp4_cache_optimized.py` 处理 3-4 层后 OOM：`RESOURCE_EXHAUSTED: Attempting to allocate 66.50G`

**原因**：async save 的 `ThreadPoolExecutor` 持有 `output_weights`（FusedMoEWeights 含 JAX device arrays）的引用。Python GC 无法回收在 Future 中被引用的 device arrays，HBM 累积泄漏

**修复**：在提交 async save 前，先在主线程完成 device→host 拷贝（`np.asarray()`），再将纯 numpy 数据提交给 save 线程。JAX arrays 立即释放

```python
# 错误（泄漏）：
save_pool.submit(save_layer_cache, out_dir, output_weights)  # holds device refs

# 正确：
np_data = weights_to_numpy(output_weights)  # copy to host
del output_weights  # release device memory
save_pool.submit(save_layer_numpy, out_dir, np_data)  # numpy only
```

### 17. CPU 并行 FP4 生成的内存管理（3 个 OOM 教训）

**教训 1：不要把模型放 /dev/shm 再跑转换**

**现象**：模型 673G 占满 /dev/shm，剩余 RAM 不够 workers 的 FP32 中间数组

**修复**：模型放磁盘读（/data/models/），load 慢 ~20s/层但释放 800G RAM 给计算。
12 workers × 70G peak = 840G ≈ 可用 RAM

**教训 2：glibc malloc 不归还大块内存**

**现象**：ThreadPoolExecutor 前 3 批层正常，第 4 批 OOM。`del array` + `gc.collect()` 后 RSS 不降

**原因**：glibc malloc arena 碎片。numpy 分配的大 FP32 数组（30 GB）释放后，arena 不释放回 OS

**修复**：用 `ProcessPoolExecutor(max_tasks_per_child=1)` — 每层处理完杀掉 worker 进程重建，
OS 直接回收所有内存。代价是每层多 ~1s 的进程启动开销，可忽略

**教训 3：numpy 临时数组要用 in-place 操作**

**现象**：预估每 worker peak ~50 GB，实际 ~130 GB

**原因**：`np.abs(w)` → 30G 拷贝，`w * scale_inv` → 30G，`np.clip(result)` → 30G。
quantize_to_fp4 内部创建了 3 个全尺寸 FP32 临时数组

**修复**：
```python
# 优化前（3× 临时数组 = 90 GB）：
abs_max = np.max(np.abs(w_fp32), axis=2, keepdims=True)
fp4 = np.clip(w_fp32 * scale_inv, fp4_min, fp4_max).astype(fp4_dtype)

# 优化后（1× 临时数组 = 30 GB）：
abs_buf = np.abs(w_fp32)           # 1 copy
abs_max = np.max(abs_buf, ...)
del abs_buf                         # 立即释放
w_fp32 *= scale_inv                 # in-place
np.clip(w_fp32, ..., out=w_fp32)    # in-place
fp4 = w_fp32.astype(fp4_dtype)
```

---

## 端到端时间线（实测）

从零开始到推理验证通过的完整时间线（CPU 并行 FP4 直转路径）：

| 步骤 | 耗时 | 备注 |
|------|------|------|
| TPU VM 创建 | ~3 min | queued-resources |
| Hyperdisk 2TB 创建+挂载 | ~2 min | 格式化 ~1 min |
| 系统依赖 + Python 环境 | ~5 min | uv + vLLM + tpu-inference |
| 模型下载（HuggingFace） | 视网速 | 700 GB |
| **FP4 cache 生成（CPU 并行）** | **~28 min** | gen_fp4_cache_cpu_parallel.py, 12 workers |
| FP4 cache 拷贝到 /dev/shm | ~12 min | 610 GB, ~850 MB/s |
| vLLM FP4 启动 | ~7 min | 含 safetensors + MoE cache |
| GSM8K 测试 | ~23 min | 1319 题, exact_match 94.9% |
| **总计（不含模型下载）** | **~80 min** | 比旧方案（~2h）快 33% |

### 已验证的软件版本组合

| 组件 | 裸机 | Docker |
|------|------|--------|
| JAX | 0.9.2 | 0.9.2 |
| jaxlib | 0.9.2 | 0.9.2 |
| libtpu | 0.0.39 | 0.0.39 |
| flax | 0.12.4 | 0.12.4 |
| torch | 2.9.0+cpu | 2.10.0 |
| vLLM | 0.19.1rc1.dev321+g6dc949140.tpu | 同左 |
| tpu-inference | feature/moe-fp4-weight-cache | 同左 |
| Python | 3.12 | 3.12 |
| TPU runtime | v2-alpha-tpu7-ubuntu2404 | 同左 |
