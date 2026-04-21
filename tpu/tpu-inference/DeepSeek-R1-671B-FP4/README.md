# DeepSeek R1 671B FP4 Inference on TPU v7x-8

> 端到端指南：在单节点 TPU v7x-8 上运行 DeepSeek R1 671B（FP4 量化）推理，
> 包含环境搭建、权重缓存生成、FP4 转换、vLLM 服务启动、以及 GSM8K 准确性验证。
>
> **代码仓库**: https://github.com/yangwhale/tpu-inference (branch: `feature/moe-fp4-weight-cache`)

本文档提供两种部署方式：

| 方式 | 适用场景 | 跳转 |
|------|----------|------|
| **Part A: GKE + Docker** | 生产环境，GKE 集群已有 TPU node pool | [Part A](#part-a-gke--docker) |
| **Part B: TPU VM 裸机** | 开发测试，直接在 TPU VM 上安装运行 | [Part B](#part-b-tpu-vm-裸机安装) |

两种方式共享相同的推理和测试步骤（Step 3-7）。

---

## 硬件与模型概览

### 硬件要求

| 项目 | 要求 |
|------|------|
| TPU | v7x-8（2x2x1 拓扑，4 chips = 8 devices） |
| HBM | 94.75 GB/device，总计 758 GB |
| 主机内存 | ≥920 GB（模型加载 + /dev/shm 缓存） |
| 存储 | ≥1.5 TB（模型 700 GB + FP8 cache 404 GB + FP4 cache 610 GB） |

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
    - name: lustre
      mountPath: /lustre
    - name: dshm
      mountPath: /dev/shm
  volumes:
  - name: lustre
    persistentVolumeClaim:
      claimName: lustre-pvc    # 替换为实际 PVC
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

# 安装 tpu-inference
cd ~/tpu-inference
uv pip install -e .
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
> | FP8 cache | ~404 GB | `/data/moe-cache/ep8_tp1_gmm_ep_fp8e4m3_bsNone` |
> | FP4 cache | ~610 GB | `/data/moe-cache/ep8_tp1_gmm_ep_fp4e2m1_bsNone` |
> | **合计** | **~1.7 TB** | 2 TB 盘剩余 ~300 GB |

完成后继续 Step 3。

---

# 通用步骤（GKE 和 TPU VM 共用）

以下步骤在两种部署方式中通用。路径变量说明：
- **GKE (Part A)**：`$STORAGE` 通常是 `/lustre`，代码在 `/workspace/tpu_inference`
- **TPU VM (Part B)**：`$STORAGE` 取决于 B-5 的选择，代码在 `~/tpu-inference`

```bash
# 设置通用变量（根据实际情况调整）
# GKE:
export STORAGE=/lustre
export TI_DIR=/workspace/tpu_inference

# TPU VM:
export STORAGE=/mnt/data    # 或 /gcs 或 /lustre
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

## Step 4: 首次启动 — 生成 FP8 MoE Cache

首次启动 vLLM 时，系统自动从 safetensors 提取 MoE 权重并生成 FP8 缓存。
较慢（~30-60 分钟），但只需执行一次。

```bash
export MOE_WEIGHT_CACHE_DIR=$STORAGE/moe-cache
export MOE_REQUANTIZE_WEIGHT_DTYPE=float8_e4m3fn
export NEW_MODEL_DESIGN=1

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

等待日志出现 `[MoE cache saved]` 表示缓存已写入。启动完成后 `Ctrl+C` 停止。

验证 FP8 cache：
```bash
# 应有 58 个 layer 目录
ls $MOE_WEIGHT_CACHE_DIR/ep8_tp1_gmm_ep_fp8e4m3_bsNone/ | grep model_layers | wc -l
# 预期：58

# 每个 layer 目录包含 5 个文件
ls $MOE_WEIGHT_CACHE_DIR/ep8_tp1_gmm_ep_fp8e4m3_bsNone/model_layers_3_mlp_experts/
# 预期：meta.json  w13_weight.npy  w13_weight_scale.npy  w2_weight.npy  w2_weight_scale.npy
```

> **注意**：子目录名（如 `ep8_tp1_gmm_ep_fp8e4m3_bsNone`）由 EP/TP/backend/dtype 配置自动决定。

---

## Step 5: FP8 → FP4 离线转换

FP4 将 MoE 权重体积减半，使 671B 模型放入 v7x-8 的 HBM。纯 CPU 运算，无需 TPU。

```bash
python3 $TI_DIR/scripts/convert_fp8_to_fp4.py \
  --fp8-cache $MOE_WEIGHT_CACHE_DIR/ep8_tp1_gmm_ep_fp8e4m3_bsNone \
  --fp4-cache $MOE_WEIGHT_CACHE_DIR/ep8_tp1_gmm_ep_fp4e2m1_bsNone \
  --workers 10
```

预期输出：
```
Converting 58 layers (FP8 -> native FP4)
  Input:   .../ep8_tp1_gmm_ep_fp8e4m3_bsNone
  Output:  .../ep8_tp1_gmm_ep_fp4e2m1_bsNone
  Workers: 10
  FP4 range: [-6.0, 6.0]
  model_layers_3_mlp_experts: 42.1s
  ...
Done: 58 layers in 2520.3s (42.0 min)
```

验证 FP4 cache：
```bash
python3 -c "
import json
with open('$MOE_WEIGHT_CACHE_DIR/ep8_tp1_gmm_ep_fp4e2m1_bsNone/model_layers_3_mlp_experts/meta.json') as f:
    m = json.load(f)
    print('storage_format:', m.get('_storage_format'))
    print('w13_dtype:', m.get('w13_weight_dtype'))
    print('w2_dtype:', m.get('w2_weight_dtype'))
"
# 预期：
#   storage_format: native_fp4
#   w13_dtype: float4_e2m1fn
#   w2_dtype: float4_e2m1fn
```

### （可选）预拷贝 FP4 Cache 到 /dev/shm

如果存储 I/O 是瓶颈，将 FP4 cache 拷贝到内存文件系统可加速 21x：

```bash
# FP4 cache 约 610 GB，需要 /dev/shm 有足够空间
df -h /dev/shm
# TPU VM 默认 /dev/shm = 主机内存的一半（v7x-8 约 473 GB）
# GKE Pod 可配置 emptyDir.sizeLimit: 800Gi

# 如果空间足够：
time cp -r $MOE_WEIGHT_CACHE_DIR/ep8_tp1_gmm_ep_fp4e2m1_bsNone /dev/shm/
# 约 5-6 分钟

# 后续启动时指向 /dev/shm
export MOE_WEIGHT_CACHE_DIR=/dev/shm
```

> **TPU VM 注意**：默认 /dev/shm 约 473 GB，放不下 610 GB FP4 cache。
> 可通过 `sudo mount -o remount,size=800G /dev/shm` 扩大，但会压缩可用内存。
> 如果 shm 空间不够，直接从磁盘加载也可以正常工作（慢一些）。

---

## Step 6: 启动 vLLM 推理服务

```bash
export MOE_REQUANTIZE_WEIGHT_DTYPE=float4_e2m1fn
export NEW_MODEL_DESIGN=1
# MOE_WEIGHT_CACHE_DIR 沿用 Step 5 的设置

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

以下数据在 TPU v7x-8 单节点上测得。

### 权重加载时间

| 阶段 | 耗时 | 说明 |
|------|------|------|
| vLLM 初始化 | ~30s | 模型配置解析 + JAX 初始化 |
| Safetensors 非 MoE 权重 | ~4:25 | 23 GB / 1367 个 tensor |
| MoE Cache → TPU（磁盘） | ~27:00 | 直接从 Lustre/PD mmap 读取 |
| MoE Cache → TPU（/dev/shm） | **~1:16** | tmpfs zero-copy mmap |
| **总计（/dev/shm 加速）** | **~5:41** | **对比磁盘的 ~33 min，加速 5.8x** |

### MoE Cache 大小

| 格式 | 大小 | 说明 |
|------|------|------|
| FP8 cache | ~404 GB | 58 层 × ~7 GB/层 |
| FP4 cache | ~610 GB | 58 层 × ~10.5 GB/层（含 FP32 scale） |

> **注意**：FP4 cache 文件比 FP8 大，因为 FP4 原生存储需要额外的 scale 精度（FP32 scale）。
> 但加载到 TPU HBM 后，FP4 权重占用的显存只有 FP8 的一半。

### FP8→FP4 转换

| 项目 | 数据 |
|------|------|
| 转换时间 | ~42 分钟（10 workers） |
| CPU 内存峰值 | ~30 GB |
| 输入 | FP8 npy cache（404 GB） |
| 输出 | Native FP4 npy cache（610 GB） |

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
