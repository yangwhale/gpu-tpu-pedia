# GLM-5.1 754B FP4 Inference on TPU v7x — TPU VM 版

> TPU VM 端到端部署指南：从创建 VM 到完成 benchmark。
>
> **模型**: [zai-org/GLM-5.1-FP8](https://huggingface.co/zai-org/GLM-5.1-FP8)（142 safetensors, ~705 GB）
>
> **架构**: 754B 总参 / **MoE**（256 experts, top-8）+ MLA + DSA + MTP / 78 layers / FP4 MoE + FP8 Attn + BF16 non-MoE
>
> **代码仓库**: [yangwhale/tpu-inference](https://github.com/yangwhale/tpu-inference)（branch: `feature/glm51-inference`）
>
> **模型存储**: `gs://aidc-tpu-data/models`（GCS 对象存储，模型权重统一存放于此）
>
> GKE 版见同目录 [README.md](README.md)。

---

### 关键前提

> **GLM-5.1 推理需要 3 步额外准备**（Qwen3.5/Qwen3-Coder 不需要）：
>
> 1. **生成 FP4 MoE Cache**：`gen_fp4_cache_cpu_parallel.py`，CPU 纯 numpy，~28 min
> 2. **合并 Non-MoE 权重**：`extract_non_moe_weights.py`，~2 min
> 3. **拷贝 Cache 到 /dev/shm**：~4 min（并行拷贝）
>
> 这 3 步是 **首次部署**独有的。后续重启只需确认 /dev/shm 中 cache 仍在即可跳过。

### 三个必设环境变量

> **每次启动 vLLM 前必须设置，缺一不可！**

| 环境变量 | 值 | 漏设后果 |
|---------|-----|---------|
| `MOE_REQUANTIZE_WEIGHT_DTYPE=float4_e2m1fn` | **必须设** | 默认查找 FP8 cache → cache miss → **HBM OOM** |
| `NEW_MODEL_DESIGN=1` | **必须设** | MLA 模型强制要求，不设直接报错退出 |
| `MOE_WEIGHT_CACHE_DIR=/dev/shm` | **必须设** | 找不到 FP4 cache，触发在线 requantization |

`MOE_REQUANTIZE_WEIGHT_DTYPE` 是最容易遗漏也最致命的：它控制 cache 子目录名。不设时默认为 `fp8`，子目录变成 `ep8_tp1_gmm_ep_fp8e4m3_bsNone`，而 FP4 cache 在 `ep8_tp1_gmm_ep_fp4e2m1_bsNone`，导致全部 cache miss → OOM。

### 与 Qwen3.5 的关键区别

| 维度 | Qwen3.5-397B | **GLM-5.1-754B** |
|------|-------------|-----------------|
| 架构 | Hybrid GDN+Attn | **纯 MoE + MLA** |
| 量化 | FP8 native | **FP4 MoE + FP8 Attn + BF16 non-MoE** |
| FP4 Cache | 不需要 | **必须**（~705 GB，含生成+拷贝流程） |
| 并行策略 | TP=8 | **EP=8, TP=1**（`--additional-config` JSON 控制） |
| 代码分支 | `main` | **`feature/glm51-inference`**（yangwhale fork） |
| vLLM 入口 | `vllm serve` | `vllm serve`（必须用 CLI 入口，`python3 -m` 会触发循环导入） |
| Chat 稳定性 | 5-shot only | **Chat 正常**（自称 GLM / Z.ai） |
| 数据盘需求 | ≥500 GB | **≥2 TB**（模型 705 GB + FP4 cache 705 GB） |
| /dev/shm 用途 | 存放模型权重 | **存放 FP4 cache**（模型留在磁盘） |
| PD Connector | TPUConnectorHMA | **TPUConnector**（非 hybrid） |

---

## 目录

- [Part 1: 单机推理](#part-1-单机推理)
- [Part 2: PD 分离 (1P1D)](#part-2-pd-分离-1p1d)
- [Part 3: 多节点推理 (EP=16)](#part-3-多节点推理-ep16)

---

## 环境变量

```bash
export PROJECT_ID=<your-project-id>
export ZONE=us-central1-c
export RESERVATION_NAME=<your-reservation>
export VPC_NAME=<your-vpc-name>
export SUBNET_NAME=<your-subnet-name>
export HF_TOKEN=<your-hf-token>
export MODEL_BUCKET=gs://aidc-tpu-data/models         # 模型权重 GCS 路径
export MODEL_NAME=GLM-5.1-FP8                          # 模型目录名（与 GCS 一致）
```

## 硬件要求

| 项目 | 单机 (Part 1 & 2) | 多节点 (Part 3) |
|------|-------------------|----------------|
| 机型 | `tpu7x-standard-4t` | `tpu7x-standard-4t` × 2 |
| TPU | v7x-8（4 chips, 8 devices） | v7x-16（8 chips, 16 devices） |
| HBM | 768 GB | 1,536 GB |
| 主机内存 | 944 GB | 944 GB × 2 |
| 启动盘 | 1 TB Hyperdisk Balanced | 1 TB × 2 |
| 数据盘 | **≥2 TB**（模型 705 GB + FP4 cache 705 GB） | ≥2 TB × 2 |
| /dev/shm | **≥800 GB**（FP4 cache ~705 GB） | 待定（见 Part 3 说明） |

---

# Part 1: 单机推理

## Step 1: 创建数据盘和 VM

### 1.1 创建 Hyperdisk ML 数据盘

```bash
gcloud compute disks create glm51-data-01 \
    --project=${PROJECT_ID} \
    --zone=${ZONE} \
    --type=hyperdisk-ml \
    --size=4TB \
    --provisioned-throughput=2500
```

> **为什么 4 TB**：模型 ~705 GB + FP4 cache ~705 GB + Non-MoE 合并文件 ~21 GB + 临时文件 ≈ 1.5 TB。4 TB 提供充足余量。如果 Hyperdisk ML 配额不足，可改用 Hyperdisk Balanced 2 TB。

### 1.2 创建 TPU VM

```bash
gcloud compute instances create glm51-vm-01 \
    --project=${PROJECT_ID} \
    --zone=${ZONE} \
    --machine-type=tpu7x-standard-4t \
    --network-interface=network=${VPC_NAME},subnet=${SUBNET_NAME} \
    --maintenance-policy=TERMINATE \
    --provisioning-model=RESERVATION_BOUND \
    --reservation-affinity=specific \
    --reservation=${RESERVATION_NAME} \
    --create-disk=auto-delete=yes,boot=yes,size=1000GB,type=hyperdisk-balanced,image=projects/ubuntu-os-accelerator-images/global/images/family/ubuntu-accel-2404-amd64-tpu-tpu7x \
    --disk=name=glm51-data-01,device-name=data-disk,mode=rw,auto-delete=no \
    --no-shielded-secure-boot \
    --scopes=cloud-platform
```

### 1.3 SSH 连接

```bash
VM_IP=$(gcloud compute instances describe glm51-vm-01 \
    --project=${PROJECT_ID} --zone=${ZONE} \
    --format="value(networkInterfaces[0].accessConfigs[0].natIP)")
ssh -o StrictHostKeyChecking=no -i ~/.ssh/google_compute_engine ${USER}@${VM_IP}
```

---

## Step 2: 格式化挂载数据盘

```bash
sudo mkfs.ext4 -m 0 -E lazy_itable_init=0,lazy_journal_init=0,discard /dev/nvme1n1
sudo mkdir -p /mnt/data
sudo mount -o discard,defaults /dev/nvme1n1 /mnt/data
sudo chmod a+w /mnt/data
echo "/dev/disk/by-id/google-data-disk /mnt/data ext4 discard,defaults,nofail 0 2" | sudo tee -a /etc/fstab
```

---

## Step 3: 环境准备

### 3.1 系统配置

```bash
sudo sysctl -w vm.max_map_count=8388608
echo 'vm.max_map_count=8388608' | sudo tee -a /etc/sysctl.conf

if [ -f /sys/module/vfio_iommu_type1/parameters/dma_entry_limit ]; then
  echo 2000000 | sudo tee /sys/module/vfio_iommu_type1/parameters/dma_entry_limit
fi
```

### 3.2 安装系统依赖

```bash
sudo apt-get update -qq
sudo apt-get install -y -qq libopenmpi-dev libomp-dev git curl
```

### 3.3 安装 vLLM + tpu-inference（裸机）

> **关键差异**：GLM-5.1 使用 `yangwhale/tpu-inference` 的 `feature/glm51-inference` 分支，不是 upstream `vllm-project/tpu-inference` 的 main 分支。

```bash
# 安装 uv
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

# 创建 Python 3.12 虚拟环境
uv venv ~/vllm_env --python 3.12
source ~/vllm_env/bin/activate

# 克隆 tpu-inference（GLM-5.1 分支）
cd ~
git clone https://github.com/yangwhale/tpu-inference.git
cd tpu-inference
git checkout feature/glm51-inference

# 获取 pinned vLLM commit hash
VLLM_VERSION_FILE=".buildkite/vllm_lkg.version"
if [ -f "$VLLM_VERSION_FILE" ]; then
    export VLLM_COMMIT_HASH="$(cat $VLLM_VERSION_FILE | tr -d '[:space:]')"
    echo "Pinned vLLM commit: ${VLLM_COMMIT_HASH}"
else
    echo "No pinned version found, using latest vLLM main"
    export VLLM_COMMIT_HASH=""
fi

# 克隆 vLLM 并 checkout 到 pinned 版本
cd ~
git clone https://github.com/vllm-project/vllm.git
cd vllm
if [ -n "${VLLM_COMMIT_HASH}" ]; then
    git checkout "${VLLM_COMMIT_HASH}"
fi

# 安装 vLLM（TPU target）
uv pip install -r requirements/tpu.txt --torch-backend=cpu
VLLM_TARGET_DEVICE="tpu" uv pip install -e .

# 修复 JAX 版本（vLLM 安装会降级 JAX，必须装回 0.9.2）
uv pip install jax==0.9.2 jaxlib==0.9.2 libtpu==0.0.39 flax==0.12.4

# 安装 tpu-inference（--no-deps 避免再次降级 JAX）
cd ~/tpu-inference
uv pip install -e . --no-deps
```

### 3.4 验证安装

```bash
source ~/vllm_env/bin/activate
python3 << 'PYEOF'
import jax
import importlib.metadata
from vllm.platforms import current_platform
print(f"vllm: {importlib.metadata.version('vllm')}")
print(f"tpu_inference: {importlib.metadata.version('tpu_inference')}")
print(f"jax: {jax.__version__}")
print(f"platform: {current_platform.get_device_name()}")
print(f"devices: {len(jax.devices())} x {jax.devices()[0].platform}")
PYEOF
```

预期输出（版本号以实际为准）：
```
vllm: 0.20.x
tpu_inference: 0.0.0
jax: 0.9.2
platform: TPU V7X
devices: 8 x tpu
```

> **关于 JAX 版本**：vLLM 的 `requirements/tpu.txt` 会安装 JAX 0.8.0，但 TPU v7x 需要 JAX 0.9.2 + libtpu 0.0.39。安装 vLLM 后必须手动覆盖安装正确版本。
>
> **关于 `--no-deps`**：tpu-inference 的 `pyproject.toml` 依赖 jax/jaxlib，不加 `--no-deps` 会再次触发 JAX 降级。

### 3.5 设置运行时环境变量

```bash
# 基础环境变量
export HF_TOKEN=${HF_TOKEN}
export JAX_PLATFORMS=tpu,cpu
export TPU_BACKEND_TYPE=jax
export PJRT_DEVICE=TPU
export MODEL_IMPL_TYPE=flax_nnx
export USE_MOE_EP_KERNEL=0
export USE_BATCHED_RPA_KERNEL=0

# ⚠️ GLM-5.1 三个必设环境变量
export MOE_REQUANTIZE_WEIGHT_DTYPE=float4_e2m1fn   # 控制 FP4 cache 查找，漏设 = OOM
export NEW_MODEL_DESIGN=1                           # MLA 模型必须
export MOE_WEIGHT_CACHE_DIR=/dev/shm                # 指向 FP4 cache 根目录
```

### 3.6 下载模型权重

模型权重可从 GCS 或 HuggingFace 下载到数据盘。

```bash
# 方案 A：从 GCS 拷贝（推荐，速度最快）
gcloud storage cp -r ${MODEL_BUCKET}/${MODEL_NAME} /mnt/data/

# 方案 B：从 HuggingFace 下载
pip install -U "huggingface_hub[hf_transfer]"
HF_HUB_ENABLE_HF_TRANSFER=1 huggingface-cli download \
  zai-org/GLM-5.1-FP8 --local-dir /mnt/data/GLM-5.1-FP8

# 验证
ls /mnt/data/GLM-5.1-FP8/*.safetensors | wc -l   # 应为 142
du -sh /mnt/data/GLM-5.1-FP8                      # 应为 ~705 GB
```

设置模型路径变量（后续步骤都会用到）：

```bash
export MODEL=/mnt/data/GLM-5.1-FP8
```

> **与 Qwen3.5 的区别**：Qwen3.5 模型 378 GB 可放 /dev/shm 加速加载。GLM-5.1 模型 705 GB + FP4 cache 705 GB 合计 1.4 TB，远超 /dev/shm 容量，因此**模型留在数据盘，/dev/shm 只存 FP4 cache**。
>
> **首次上传模型到 GCS**：如果 GCS 桶里还没有模型权重，先在任意机器上从 HuggingFace 下载后上传：
> ```bash
> gcloud storage cp -r /mnt/data/GLM-5.1-FP8 ${MODEL_BUCKET}/
> ```

---

## Step 4: 生成 FP4 Cache + Non-MoE 合并

> **首次部署独有**。FP4 MoE Cache 和 Non-MoE 合并文件生成一次即可，后续重启直接使用。

### 4.1 下载脚本

```bash
cd /mnt/data
curl -LO https://raw.githubusercontent.com/yangwhale/gpu-tpu-pedia/main/tpu/tpu-inference/GLM-5.1-754B-FP4/gen_fp4_cache_cpu_parallel.py
curl -LO https://raw.githubusercontent.com/yangwhale/gpu-tpu-pedia/main/tpu/tpu-inference/GLM-5.1-754B-FP4/extract_non_moe_weights.py
```

### 4.2 确保 /dev/shm 为空

```bash
df -h /dev/shm
ls /dev/shm/

# ⚠️ 如果有旧数据，必须清理！12 workers 峰值内存 ~70 GB/worker，残留数据会导致 OOM kill
rm -rf /dev/shm/*
```

### 4.3 生成 FP4 MoE Cache（~28 min）

```bash
source ~/vllm_env/bin/activate

python3 -u /mnt/data/gen_fp4_cache_cpu_parallel.py \
  --model-dir $MODEL \
  --cache-dir /mnt/data/moe-cache/ep8_tp1_gmm_ep_fp4e2m1_bsNone \
  --workers 12

# 中断后可续跑（自动跳过已完成的层）
```

> **workers 数量**：根据可用 RAM 调整（每 worker 峰值 ~70 GB）。v7x-8 机器 944 GB RAM → 最多 12 workers。
>
> **纯 CPU 操作**：不需要 TPU/JAX，纯 numpy 计算。可在任何有足够 RAM 的机器上运行。

### 4.4 提取 Non-MoE 权重（~2 min）

将散落在 142 个 safetensors 中的非 MoE 权重合并为单个文件，**加载从 4m26s → 21s**：

```bash
python3 /mnt/data/extract_non_moe_weights.py \
  --model-dir $MODEL \
  --output /mnt/data/moe-cache/ep8_tp1_gmm_ep_fp4e2m1_bsNone/non_moe_weights.safetensors
```

### 4.5 验证

```bash
# 检查 MoE 层数
ls /mnt/data/moe-cache/ep8_tp1_gmm_ep_fp4e2m1_bsNone/ | grep model_layers | wc -l
# 预期：76（layer 3-78）

# 检查 non-MoE 文件
ls -lh /mnt/data/moe-cache/ep8_tp1_gmm_ep_fp4e2m1_bsNone/non_moe_weights.safetensors
# 预期：~21 GB

# 检查 FP4 shape
python3 -c "
import numpy as np
d = '/mnt/data/moe-cache/ep8_tp1_gmm_ep_fp4e2m1_bsNone/model_layers_3_mlp_experts'
for name in ['w13_weight', 'w13_weight_scale', 'w2_weight', 'w2_weight_scale']:
    a = np.load(f'{d}/{name}.npy')
    print(f'{name}: {a.shape} {a.dtype}')
"
# 预期输出：
#   w13_weight:       (256, 6144, 4096) |V1    (float4_e2m1fn)
#   w13_weight_scale: (256, 1, 1, 4096) float32
#   w2_weight:        (256, 2048, 6144) |V1
#   w2_weight_scale:  (256, 1, 1, 6144) float32
```

---

## Step 5: 拷贝 Cache 到 /dev/shm

将 FP4 cache + Non-MoE 权重预加载到 `/dev/shm`（tmpfs），**大幅加速启动 + 避免 MoE prefetch deadlock**。

```bash
# 扩容 /dev/shm（默认 ~472 GB，需容纳 705 GB FP4 cache）
sudo mount -o remount,size=800G /dev/shm

SRC=/mnt/data/moe-cache/ep8_tp1_gmm_ep_fp4e2m1_bsNone
DST=/dev/shm/ep8_tp1_gmm_ep_fp4e2m1_bsNone

mkdir -p $DST

# 拷贝 non-MoE 权重
cp $SRC/non_moe_weights.safetensors $DST/

# 并行拷贝 76 层 MoE cache（8 workers，~4 min）
ls -d $SRC/model_layers_* | xargs -P 8 -I {} cp -r {} $DST/

# 验证
ls $DST/ | grep model_layers | wc -l   # 预期：76
ls -lh $DST/non_moe_weights.safetensors  # 预期：~21 GB
df -h /dev/shm                           # 预期占用 ~705 GB
```

> **不要用 `cp -r` 单线程**！单线程 ~8 min，`xargs -P 8` 并行 ~4 min。
>
> **总占用**：FP4 cache + non-MoE ≈ **~705 GB**，/dev/shm 800 GB 够用。
>
> **⚠️ /dev/shm 是 tmpfs**：VM 重启后数据丢失，需重新从数据盘拷贝（Step 5 本步骤）。

### 可选优化：注释 `jax.clear_caches()`

`weight_utils.py` 中的 `jax.clear_caches()` 导致每个 tensor 的 `jax.device_put()` 重新编译。2292 个 non-MoE tensor 只有 ~25 种 unique shape，但每次都重编译。

```bash
source ~/vllm_env/bin/activate
TPI_DIR=$(python3 -c "import tpu_inference; import os; print(os.path.dirname(tpu_inference.__file__))" 2>/dev/null)

grep -n 'jax.clear_caches()' ${TPI_DIR}/models/jax/utils/weight_utils.py
# 注释掉所有出现的 jax.clear_caches() 行
sed -i 's/^        jax.clear_caches()/#        jax.clear_caches()/' \
  ${TPI_DIR}/models/jax/utils/weight_utils.py

# 清理 pycache
find ${TPI_DIR} -name '__pycache__' -type d -exec rm -rf {} + 2>/dev/null || true
```

> **效果**：Non-MoE 加载从 6m29s → ~25-108s（实测 10x 加速）。
> **风险**：无。cache 存的是 compiled transfer programs，~25 种 shape × 几 KB ≈ 不到 1 MB。

---

## Step 6: 启动 vLLM（~4-11 min）

> **重要**：必须 `cd /tmp` 后再运行 vLLM，否则 `~/vllm/` 或 `~/tpu-inference/` 目录会被 Python 当作 namespace package，导致 import 错误。

```bash
source ~/vllm_env/bin/activate
cd /tmp && mkdir -p /tmp/vllm-logs

# ⚠️ 确认三个必设环境变量
echo "MOE_REQUANTIZE_WEIGHT_DTYPE=${MOE_REQUANTIZE_WEIGHT_DTYPE}"  # 应为 float4_e2m1fn
echo "NEW_MODEL_DESIGN=${NEW_MODEL_DESIGN}"                        # 应为 1
echo "MOE_WEIGHT_CACHE_DIR=${MOE_WEIGHT_CACHE_DIR}"                # 应为 /dev/shm

nohup env \
  PJRT_DEVICE=TPU TPU_BACKEND_TYPE=jax JAX_PLATFORMS=tpu,cpu \
  SKIP_JAX_PRECOMPILE=1 VLLM_XLA_CHECK_RECOMPILATION=0 \
  MODEL_IMPL_TYPE=flax_nnx HF_HUB_OFFLINE=1 \
  MOE_REQUANTIZE_WEIGHT_DTYPE=float4_e2m1fn \
  NEW_MODEL_DESIGN=1 \
  MOE_WEIGHT_CACHE_DIR=/dev/shm \
  vllm serve $MODEL \
    --served-model-name GLM-5.1-FP8 \
    --tensor-parallel-size 8 \
    --quantization fp8 \
    --enforce-eager \
    --enable-prefix-caching \
    --enable-chunked-prefill \
    --max-model-len 4096 \
    --trust-remote-code \
    --additional-config '{"sharding":{"sharding_strategy":{"enable_dp_attention":true,"expert_parallelism":8,"tensor_parallelism":1}},"replicate_attn_weights":"True","sparse_matmul":"True"}' \
    --host 0.0.0.0 --port 8000 \
  > /tmp/vllm-logs/serve.log 2>&1 &

# 等待就绪（约 4-11 min）
until curl -s -o /dev/null -w '%{http_code}\n' http://localhost:8000/health | grep -q 200; do
  date; tail -1 /tmp/vllm-logs/serve.log 2>/dev/null; sleep 30
done
echo "Server ready"
```

### 参数说明

| 参数 | 实际含义 | 容易误解的点 |
|------|----------|-------------|
| `--tensor-parallel-size 8` | 总设备数 | **不是 TP=8**。实际 TP=1，EP=8（由 additional-config 控制） |
| `--quantization fp8` | vLLM 量化 schema 名 | **不是 FP8 推理**。MoE 的 FP4 由环境变量控制 |
| `expert_parallelism: 8` | EP=8 | 256 experts ÷ 8 = 每 device 32 experts |
| `tensor_parallelism: 1` | TP=1 | attention 权重用 replicate 代替切分 |
| `--enforce-eager` | 禁用 ahead-of-time 编译 | MoE 模型必须，否则编译超时 |

> **启动时间**：
> - **未优化**：~11 min（non-MoE 加载 6m29s）
> - **优化后**（注释 `jax.clear_caches()`）：~4-6 min

---

## Step 7: 验证推理

在**另一个终端** SSH 到 VM：

```bash
# 测试 1: 数学计算
curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "GLM-5.1-FP8",
    "messages": [{"role": "user", "content": "What is 2+3? Answer with just the number."}],
    "max_tokens": 256
  }' | python3 -m json.tool

# 测试 2: 中文对话
curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "GLM-5.1-FP8",
    "messages": [{"role": "user", "content": "你是谁？用一句话介绍自己。"}],
    "max_tokens": 128
  }' | python3 -c "import sys,json; r=json.load(sys.stdin); print(r['choices'][0]['message']['content'])"

# 健康检查
curl -s http://localhost:8000/health
# 预期：{"status":"ok"}
```

> **GLM-5.1 是 thinking 模型**：输出可能包含 `<think>...reasoning...</think>` 推理过程，这是正常行为。

### 实测验证结果（2026-04-24, GKE E2E pod, v7x-8）

| 测试 | 结果 |
|------|------|
| 2+3 数学计算 | ✅ 正确回答 5（含思维链推理过程） |
| 中文自我介绍 | ✅ 自称 "Z.ai 创建的大语言模型"，输出流畅 |
| 英文逻辑推理 | ✅ 正确推理 |
| HBM 占用 | 58.43/94.75 GiB per device（61.6%） |
| MoE cache | 76/76 层全部 hit（FP4） |

---

## Step 8: Benchmark

> **注意**：TPU VM 裸机没有安装 PyTorch，`vllm bench serve` 命令不可用。使用以下 Python 脚本替代。

```bash
python3 << 'PYEOF'
import requests, time, concurrent.futures

URL = "http://localhost:8000/v1/chat/completions"
MODEL = "GLM-5.1-FP8"
BASE = "The quick brown fox jumps over the lazy dog. "

def make_prompt(target_tokens):
    return BASE * (target_tokens // 10)

def send_request(prompt, output_len, rid):
    t0 = time.time()
    r = requests.post(URL, json={
        "model": MODEL, "messages": [{"role": "user", "content": prompt}],
        "max_tokens": output_len, "temperature": 0.7, "ignore_eos": True})
    data = r.json()
    t1 = time.time()
    if "error" in data:
        return {"error": data["error"]["message"][:200]}
    u = data.get("usage", {})
    return {"prompt": u.get("prompt_tokens",0), "completion": u.get("completion_tokens",0),
            "time": t1-t0, "tps": u.get("completion_tokens",0)/(t1-t0)}

def bench(input_tok, output_tok, conc, n):
    print("\n" + "="*60)
    print("P%d/D%d  concurrency=%d  requests=%d" % (input_tok, output_tok, conc, n))
    print("="*60)
    prompt = make_prompt(input_tok)
    t0 = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=conc) as ex:
        results = [f.result() for f in
            [ex.submit(send_request, prompt, output_tok, i) for i in range(n)]]
    ok = [r for r in results if "error" not in r]
    if not ok: print("  ALL FAILED"); return
    total_t = time.time() - t0
    total_c = sum(r["completion"] for r in ok)
    print("  Avg prompt tokens:     %d" % (sum(r["prompt"] for r in ok)/len(ok)))
    print("  Per-request tok/s:     %.1f" % (sum(r["tps"] for r in ok)/len(ok)))
    print("  Aggregate tok/s:       %.1f" % (total_c / total_t))

# Warmup
send_request(make_prompt(128), 32, -1)

bench(1024, 1024, 1, 3)
bench(1024, 1024, 4, 8)
bench(1024, 1024, 16, 32)
bench(1024, 1024, 64, 128)
bench(1024, 1024, 128, 256)
PYEOF
```

### 预期性能参考（GKE 实测值, 2026-04-24）

> TPU VM 上的性能应与 GKE Pod 一致（同硬件、同软件栈）。

| 并发 | Throughput (tok/s) | tok/s/chip | TTFT (s) | TPOT (ms) |
|---:|---:|---:|---:|---:|
| 1 | 28.4 | 7.1 | 0.534 | 35 |
| 4 | 130.5 | 32.6 | 0.510 | 30 |
| 16 | 444.8 | 111.2 | 1.016 | 35 |
| 64 | 1,570 | 392.5 | 3.174 | 38 |
| 256 | 3,873 | 968.3 | 8.869 | 57 |
| **1,024** | **6,504** | **1,626** | 31.38 | 125 |

**关键操作点：**

| 操作点 | 并发 | Throughput | tok/s/chip | 适用场景 |
|--------|-----|-----------|-----------|---------|
| Max Throughput | 1,024 | 6,504 tok/s | 1,626 | 离线批处理 |
| Balanced | 64 | 1,570 tok/s | 393 | 中等负载在线服务 |
| Low Latency | 4 | 130 tok/s | 33 | 交互式对话 |

---

# Part 2: PD 分离 (1P1D)

> ⚠️ **本章节为理论设计，尚未实测。** 参数基于 GKE 单机验证 + Qwen3-Coder PD 分离经验推导。
>
> 2 台 TPU v7x-8 VM：一台跑 Prefill（kv_producer），一台跑 Decode（kv_consumer），通过 VPC 内网传输 KV cache。
>
> **架构说明**：PD 分离 **不需要 Ray**。两个 vLLM 实例完全独立运行，通过 TPUConnector 直接 P2P 传输 KV cache，`toy_proxy_server.py` 负责请求路由。

### GLM-5.1 PD 必读差异（vs Qwen3.5 PD）

| 项 | Qwen3.5 (hybrid GDN+Attn) | **GLM-5.1 (纯 MoE + MLA)** |
|---|---|---|
| `kv_connector` | `TPUConnectorHMA` | **`TPUConnector`** |
| `kv_connector_module_path` | `tpu_inference.distributed.tpu_connector_hma` | **`tpu_inference.distributed.tpu_connector`** |
| Hybrid KV cache flag | 需 `--no-disable-hybrid-kv-cache-manager` | **不需要**（纯 MoE，无 hybrid） |
| HMA connector 部署 | 需手动下载 | **不需要**（TPUConnector 已在代码中） |
| FP4 cache | 不需要 | **两台 VM 都需要各自的 /dev/shm FP4 cache** |

## Step 1: 创建 2 台 VM

复用 Part 1 的方式，创建 Prefill 和 Decode 两台 VM：

```bash
# Prefill VM
gcloud compute disks create glm51-data-prefill \
    --project=${PROJECT_ID} --zone=${ZONE} \
    --type=hyperdisk-ml --size=4TB --provisioned-throughput=2500

gcloud compute instances create glm51-prefill \
    --project=${PROJECT_ID} --zone=${ZONE} \
    --machine-type=tpu7x-standard-4t \
    --network-interface=network=${VPC_NAME},subnet=${SUBNET_NAME} \
    --maintenance-policy=TERMINATE \
    --provisioning-model=RESERVATION_BOUND \
    --reservation-affinity=specific \
    --reservation=${RESERVATION_NAME} \
    --create-disk=auto-delete=yes,boot=yes,size=1000GB,type=hyperdisk-balanced,image=projects/ubuntu-os-accelerator-images/global/images/family/ubuntu-accel-2404-amd64-tpu-tpu7x \
    --disk=name=glm51-data-prefill,device-name=data-disk,mode=rw,auto-delete=no \
    --no-shielded-secure-boot \
    --scopes=cloud-platform

# Decode VM
gcloud compute disks create glm51-data-decode \
    --project=${PROJECT_ID} --zone=${ZONE} \
    --type=hyperdisk-ml --size=4TB --provisioned-throughput=2500

gcloud compute instances create glm51-decode \
    --project=${PROJECT_ID} --zone=${ZONE} \
    --machine-type=tpu7x-standard-4t \
    --network-interface=network=${VPC_NAME},subnet=${SUBNET_NAME} \
    --maintenance-policy=TERMINATE \
    --provisioning-model=RESERVATION_BOUND \
    --reservation-affinity=specific \
    --reservation=${RESERVATION_NAME} \
    --create-disk=auto-delete=yes,boot=yes,size=1000GB,type=hyperdisk-balanced,image=projects/ubuntu-os-accelerator-images/global/images/family/ubuntu-accel-2404-amd64-tpu-tpu7x \
    --disk=name=glm51-data-decode,device-name=data-disk,mode=rw,auto-delete=no \
    --no-shielded-secure-boot \
    --scopes=cloud-platform
```

> **省盘方案**：如果第一台 VM 已生成 FP4 cache，可将数据盘设为 `READ_ONLY_MANY` 同时挂载到两台 VM（只读）。或者用一台 VM 生成完 cache 后 `gcloud storage cp` 到 GCS，第二台从 GCS 拷贝。

## Step 2: 两台 VM 分别执行环境准备

在两台 VM 上分别执行：
1. Part 1 Step 2（格式化挂载数据盘）
2. Part 1 Step 3.1 ~ 3.5（系统配置 + 安装 vLLM/tpu-inference + 设置环境变量）
3. Part 1 Step 3.6（下载模型权重到 `/mnt/data/`）
4. Part 1 Step 4（生成 FP4 Cache + Non-MoE 合并）— 第二台可从 GCS/第一台拷贝 cache
5. Part 1 Step 5（拷贝 Cache 到 /dev/shm）

> **两台 VM 都需要独立的 /dev/shm FP4 cache**。FP4 cache 在各自的 /dev/shm 中分别加载。

## Step 3: 获取内网 IP

```bash
PREFILL_IP=$(gcloud compute instances describe glm51-prefill \
    --project=${PROJECT_ID} --zone=${ZONE} \
    --format="value(networkInterfaces[0].networkIP)")
DECODE_IP=$(gcloud compute instances describe glm51-decode \
    --project=${PROJECT_ID} --zone=${ZONE} \
    --format="value(networkInterfaces[0].networkIP)")
echo "Prefill: ${PREFILL_IP}, Decode: ${DECODE_IP}"
```

## Step 4: 启动 Prefill 实例

SSH 到 Prefill VM：

```bash
source ~/vllm_env/bin/activate
cd /tmp && mkdir -p /tmp/vllm-logs

nohup env \
  PJRT_DEVICE=TPU TPU_BACKEND_TYPE=jax JAX_PLATFORMS=tpu,cpu \
  SKIP_JAX_PRECOMPILE=1 VLLM_XLA_CHECK_RECOMPILATION=0 \
  MODEL_IMPL_TYPE=flax_nnx HF_HUB_OFFLINE=1 \
  MOE_REQUANTIZE_WEIGHT_DTYPE=float4_e2m1fn \
  NEW_MODEL_DESIGN=1 \
  MOE_WEIGHT_CACHE_DIR=/dev/shm \
  vllm serve /mnt/data/GLM-5.1-FP8 \
    --served-model-name GLM-5.1-FP8 \
    --tensor-parallel-size 8 \
    --quantization fp8 \
    --enforce-eager \
    --enable-prefix-caching \
    --enable-chunked-prefill \
    --max-model-len 16384 \
    --trust-remote-code \
    --gpu-memory-utilization 0.70 \
    --additional-config '{"sharding":{"sharding_strategy":{"enable_dp_attention":true,"expert_parallelism":8,"tensor_parallelism":1}},"replicate_attn_weights":"True","sparse_matmul":"True"}' \
    --host 0.0.0.0 --port 8000 \
    --kv-transfer-config '{"kv_connector":"TPUConnector","kv_connector_module_path":"tpu_inference.distributed.tpu_connector","kv_role":"kv_producer"}' \
  > /tmp/vllm-logs/prefill.log 2>&1 &
```

关键差异（vs Part 1 单机）：
- `--gpu-memory-utilization=0.70`（留 30% HBM 给 KV transfer buffer）
- `kv_role=kv_producer`
- `--max-model-len=16384`（PD 模式支持更长 context）
- 使用 `TPUConnector`（不是 `TPUConnectorHMA`，GLM-5.1 非 hybrid）

> 启动需 5~12 分钟。用 `tail -f /tmp/vllm-logs/prefill.log` 观察进度。

## Step 5: 启动 Decode 实例

SSH 到 Decode VM：

```bash
source ~/vllm_env/bin/activate
cd /tmp && mkdir -p /tmp/vllm-logs

nohup env \
  PJRT_DEVICE=TPU TPU_BACKEND_TYPE=jax JAX_PLATFORMS=tpu,cpu \
  SKIP_JAX_PRECOMPILE=1 VLLM_XLA_CHECK_RECOMPILATION=0 \
  MODEL_IMPL_TYPE=flax_nnx HF_HUB_OFFLINE=1 \
  MOE_REQUANTIZE_WEIGHT_DTYPE=float4_e2m1fn \
  NEW_MODEL_DESIGN=1 \
  MOE_WEIGHT_CACHE_DIR=/dev/shm \
  vllm serve /mnt/data/GLM-5.1-FP8 \
    --served-model-name GLM-5.1-FP8 \
    --tensor-parallel-size 8 \
    --quantization fp8 \
    --enforce-eager \
    --enable-prefix-caching \
    --enable-chunked-prefill \
    --max-model-len 16384 \
    --trust-remote-code \
    --gpu-memory-utilization 0.90 \
    --additional-config '{"sharding":{"sharding_strategy":{"enable_dp_attention":true,"expert_parallelism":8,"tensor_parallelism":1}},"replicate_attn_weights":"True","sparse_matmul":"True"}' \
    --host 0.0.0.0 --port 9000 \
    --kv-transfer-config '{"kv_connector":"TPUConnector","kv_connector_module_path":"tpu_inference.distributed.tpu_connector","kv_role":"kv_consumer"}' \
  > /tmp/vllm-logs/decode.log 2>&1 &
```

关键差异（vs Prefill）：`--gpu-memory-utilization=0.90`（Decode 不需要预留 transfer buffer），`kv_role=kv_consumer`，`port=9000`。

## Step 6: 验证两端就绪 + 启动 Proxy

在 Prefill VM 上执行：

```bash
source ~/vllm_env/bin/activate
export DECODE_IP=<decode-vm-internal-ip>

# 确认两个实例都已 ready
curl -s http://localhost:8000/v1/models | python3 -m json.tool
curl -s http://${DECODE_IP}:9000/v1/models | python3 -m json.tool
```

两个都返回模型信息后，启动 proxy：

```bash
python3 ~/tpu-inference/examples/disagg/toy_proxy_server.py \
  --host 0.0.0.0 --port 7000 \
  --prefiller-hosts localhost --prefiller-ports 8000 \
  --decoder-hosts ${DECODE_IP} --decoder-ports 9000
```

Smoke test（通过 proxy 端口 7000）：

```bash
curl -s http://localhost:7000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "GLM-5.1-FP8",
    "messages": [{"role": "user", "content": "What is 2+3? Answer with just the number."}],
    "max_tokens": 50
  }' | python3 -m json.tool
```

> **首次请求延迟**：第一次请求会触发 XLA 编译，Prefill 和 Decode 各需约 2~3 分钟。后续请求命中编译缓存。

## Step 7: PD 分离 Benchmark

```bash
python3 << 'PYEOF'
import requests, time, concurrent.futures

URL = "http://localhost:7000/v1/chat/completions"
MODEL = "GLM-5.1-FP8"
BASE = "The quick brown fox jumps over the lazy dog. "

def make_prompt(target_tokens):
    return BASE * (target_tokens // 10)

def send_request(prompt, output_len, rid):
    t0 = time.time()
    r = requests.post(URL, json={
        "model": MODEL, "messages": [{"role": "user", "content": prompt}],
        "max_tokens": output_len, "temperature": 0.7, "ignore_eos": True})
    data = r.json()
    t1 = time.time()
    if "error" in data:
        return {"error": data["error"]["message"][:200]}
    u = data.get("usage", {})
    return {"prompt": u.get("prompt_tokens",0), "completion": u.get("completion_tokens",0),
            "time": t1-t0, "tps": u.get("completion_tokens",0)/(t1-t0)}

def bench(input_tok, output_tok, conc, n):
    print("\n" + "="*60)
    print("P%d/D%d  concurrency=%d  requests=%d" % (input_tok, output_tok, conc, n))
    print("="*60)
    prompt = make_prompt(input_tok)
    t0 = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=conc) as ex:
        results = [f.result() for f in
            [ex.submit(send_request, prompt, output_tok, i) for i in range(n)]]
    ok = [r for r in results if "error" not in r]
    if not ok: print("  ALL FAILED"); return
    total_t = time.time() - t0
    total_c = sum(r["completion"] for r in ok)
    print("  Avg prompt tokens:     %d" % (sum(r["prompt"] for r in ok)/len(ok)))
    print("  Per-request tok/s:     %.1f" % (sum(r["tps"] for r in ok)/len(ok)))
    print("  Aggregate tok/s:       %.1f" % (total_c / total_t))

# Warmup
send_request(make_prompt(128), 32, -1)

bench(1024, 1024, 1, 3)
bench(8192, 1024, 4, 8)
bench(1024, 8192, 64, 256)
PYEOF
```

### ⚠️ PD 分离已知风险点

> 以下是理论分析的潜在失败点，实测时需逐一验证：

| # | 风险 | 排查方向 |
|---|------|---------|
| 1 | TPUConnector 是否兼容 `--additional-config` 中的 EP+DP sharding | 检查 Prefill/Decode log 中 sharding 初始化是否正常 |
| 2 | FP4 cache 在 PD 模式下 /dev/shm 空间是否足够（cache 705 GB + KV transfer buffer） | 监控 `df -h /dev/shm` 和 HBM 使用 |
| 3 | `--enable-prefix-caching` 和 `--enable-chunked-prefill` 是否与 PD 兼容 | 如果 Prefill 报错，尝试去掉这两个 flag |
| 4 | KV transfer 带宽是否足够支撑长 context（16K tokens） | 如果 TTFT 异常高，降低 `--max-model-len` |

---

# Part 3: 多节点推理 (EP=16)

> ⚠️ **本章节为理论设计，尚未实测。** 参数基于 GKE 单机验证 + Qwen3.5 multi-host 经验推导。
>
> 2 台 TPU v7x-8 VM 组成 v7x-16 slice（8 chips, 16 devices），通过 ICI 高速互联。
>
> **关键区别**：GLM-5.1 multi-host 使用 **EP=16**（不是 TP=16）。256 experts ÷ 16 devices = 16 experts/device（vs 单机 32/device），MoE 部分 HBM 占用减半。

### Multi-host vs Single-host 关键差异

| # | Multi-host 特定变化 | 说明 |
|---|---|---|
| 1 | `expert_parallelism: 16`（additional-config） | 256 experts 分到 16 devices |
| 2 | `--tensor-parallel-size 16` | 总 device 数，实际 TP=1 不变 |
| 3 | `--distributed-executor-backend ray` | 需要 Ray 集群 |
| 4 | FP4 cache 目录名变化 | `ep16_tp1_gmm_ep_fp4e2m1_bsNone`（从 ep8 变为 ep16） |
| 5 | 无 mrope/hybrid patches | GLM-5.1 不需要 Qwen3.5 的 3 个 patches |

## Step 1: 创建 v7x-16 TPU Slice

TPU7x multi-host slice 必须通过 **Workload Policy + Instance Template + MIG** 三件套创建，确保物理 ICI 互联。

### 1.1 创建 Workload Policy

```bash
SLICE_NAME=glm51-slice

gcloud compute resource-policies create workload-policy ${SLICE_NAME}-wp \
    --type=HIGH_THROUGHPUT \
    --accelerator-topology=2x2x2 \
    --project=${PROJECT_ID} \
    --region=${ZONE%-*}
```

### 1.2 创建 Instance Template

```bash
gcloud compute instance-templates create ${SLICE_NAME}-it \
    --project=${PROJECT_ID} \
    --machine-type=tpu7x-standard-4t \
    --network-interface=network=${VPC_NAME},subnet=projects/${PROJECT_ID}/regions/${ZONE%-*}/subnetworks/${SUBNET_NAME},nic-type=GVNIC \
    --create-disk=auto-delete=yes,boot=yes,size=1000GB,type=hyperdisk-balanced,image=projects/ubuntu-os-accelerator-images/global/images/family/ubuntu-accel-2404-amd64-tpu-tpu7x \
    --reservation-affinity=specific \
    --reservation=${RESERVATION_NAME} \
    --provisioning-model=RESERVATION_BOUND \
    --instance-termination-action=DELETE \
    --maintenance-policy=TERMINATE \
    --scopes=cloud-platform
```

> **注意**：`--instance-termination-action=DELETE` 是 RESERVATION_BOUND + MIG 的必需参数。subnet 必须用完整路径。

### 1.3 创建 MIG（TPU Slice）

```bash
gcloud compute instance-groups managed create ${SLICE_NAME}-mig \
    --project=${PROJECT_ID} \
    --zone=${ZONE} \
    --template=${SLICE_NAME}-it \
    --size=2 \
    --default-action-on-vm-failure=do-nothing \
    --workload-policy=projects/${PROJECT_ID}/regions/${ZONE%-*}/resourcePolicies/${SLICE_NAME}-wp
```

### 1.4 获取 VM 名称和 IP

```bash
MIG_VMS=$(gcloud compute instance-groups managed list-instances ${SLICE_NAME}-mig \
    --project=${PROJECT_ID} --zone=${ZONE} --format="value(name)")
echo "VMs: ${MIG_VMS}"

HOST0_VM=$(echo "${MIG_VMS}" | head -1)
HOST1_VM=$(echo "${MIG_VMS}" | tail -1)

HOST0_IP=$(gcloud compute instances describe ${HOST0_VM} \
    --project=${PROJECT_ID} --zone=${ZONE} \
    --format="value(networkInterfaces[0].networkIP)")
HOST1_IP=$(gcloud compute instances describe ${HOST1_VM} \
    --project=${PROJECT_ID} --zone=${ZONE} \
    --format="value(networkInterfaces[0].networkIP)")
HOST0_EXT=$(gcloud compute instances describe ${HOST0_VM} \
    --project=${PROJECT_ID} --zone=${ZONE} \
    --format="value(networkInterfaces[0].accessConfigs[0].natIP)")
HOST1_EXT=$(gcloud compute instances describe ${HOST1_VM} \
    --project=${PROJECT_ID} --zone=${ZONE} \
    --format="value(networkInterfaces[0].accessConfigs[0].natIP)")

echo "Host 0: ${HOST0_VM} (${HOST0_IP} / ${HOST0_EXT})"
echo "Host 1: ${HOST1_VM} (${HOST1_IP} / ${HOST1_EXT})"
```

### 1.5 验证 ICI 互联

SSH 到任一 VM：

```bash
curl -s -H "Metadata-Flavor: Google" \
  http://metadata.google.internal/computeMetadata/v1/instance/attributes/accelerator-type
# 应输出: v7x-16

curl -s -H "Metadata-Flavor: Google" \
  http://metadata.google.internal/computeMetadata/v1/instance/attributes/worker-id
# Host 0 = 0, Host 1 = 1

curl -s -H "Metadata-Flavor: Google" \
  http://metadata.google.internal/computeMetadata/v1/instance/attributes/tpu-env | grep TOPOLOGY
# 应输出: TOPOLOGY: 2x2x2
```

## Step 2: 环境准备 + 模型/Cache 拷贝

在两台 VM 上分别执行：

1. Part 1 Step 3.1 ~ 3.5（系统配置 + 安装 vLLM/tpu-inference + 设置环境变量）
2. 附加数据盘并格式化挂载（Part 1 Step 1 + Step 2），或者不使用附加数据盘而将模型放在 boot disk

### 模型和 FP4 Cache 拷贝（两台 VM 都执行）

```bash
# 拷贝模型权重到数据盘（或 boot disk ~/models/）
gcloud storage cp -r ${MODEL_BUCKET}/${MODEL_NAME} /mnt/data/
export MODEL=/mnt/data/GLM-5.1-FP8

# 验证
ls ${MODEL}/*.safetensors | wc -l   # 应为 142
```

### FP4 Cache 处理

Multi-host 的 /dev/shm 需要与 Ray Object Store 共存。有两种方案：

**方案 A（推荐）：FP4 cache 放 /dev/shm，限制 Ray Object Store**

```bash
# 扩容 /dev/shm
sudo mount -o remount,size=850G /dev/shm

# 第一台 VM 生成 FP4 cache（如果已有，从 GCS 或其他 VM 拷贝）
python3 -u /mnt/data/gen_fp4_cache_cpu_parallel.py \
  --model-dir $MODEL \
  --cache-dir /mnt/data/moe-cache/ep8_tp1_gmm_ep_fp4e2m1_bsNone \
  --workers 12

python3 /mnt/data/extract_non_moe_weights.py \
  --model-dir $MODEL \
  --output /mnt/data/moe-cache/ep8_tp1_gmm_ep_fp4e2m1_bsNone/non_moe_weights.safetensors

# 拷贝到 /dev/shm（EP=16 目录名）
# Cache 内容与 EP=8 完全相同，只是目录名不同（tpu-inference 按 EP 值查找目录）
SRC=/mnt/data/moe-cache/ep8_tp1_gmm_ep_fp4e2m1_bsNone
DST=/dev/shm/ep16_tp1_gmm_ep_fp4e2m1_bsNone

mkdir -p $DST
cp $SRC/non_moe_weights.safetensors $DST/
ls -d $SRC/model_layers_* | xargs -P 8 -I {} cp -r {} $DST/

# 验证
ls $DST/ | grep model_layers | wc -l   # 预期：76
df -h /dev/shm                           # 预期占用 ~705 GB
```

> **EP=16 vs EP=8 目录名**：tpu-inference 按 `ep{EP}_tp{TP}_gmm_ep_{dtype}_bsNone` 格式查找 cache 目录。EP=16 时查找 `ep16_tp1_...`。Cache 内容完全相同（所有 256 experts），sharding 在加载时完成。

**方案 B：FP4 cache 放磁盘（如果 /dev/shm 空间不足）**

```bash
# cache 直接放数据盘，不拷贝到 /dev/shm
# 为 EP=16 创建 symlink
ln -s /mnt/data/moe-cache/ep8_tp1_gmm_ep_fp4e2m1_bsNone \
      /mnt/data/moe-cache/ep16_tp1_gmm_ep_fp4e2m1_bsNone

# MOE_WEIGHT_CACHE_DIR 指向磁盘
export MOE_WEIGHT_CACHE_DIR=/mnt/data/moe-cache
```

> ⚠️ **方案 B 风险**：磁盘加载比 tmpfs 慢约 100x。如果 cache 目录不完整（缺少 meta.json），可能触发 MoE prefetch deadlock。确保所有 76 个 layer 目录都有完整的 `.npy` 文件和 `meta.json`。

## Step 3: 设置 TPU 拓扑环境变量

### Host 0（Ray Head）

```bash
source ~/vllm_env/bin/activate

# 基础环境变量
export PJRT_DEVICE=TPU
export TPU_BACKEND_TYPE=jax
export JAX_PLATFORMS=
export MODEL_IMPL_TYPE=flax_nnx
export USE_MOE_EP_KERNEL=0
export USE_BATCHED_RPA_KERNEL=0
export HF_HUB_OFFLINE=1
export SKIP_JAX_PRECOMPILE=1
export VLLM_XLA_CHECK_RECOMPILATION=0

# ⚠️ GLM-5.1 三个必设环境变量
export MOE_REQUANTIZE_WEIGHT_DTYPE=float4_e2m1fn
export NEW_MODEL_DESIGN=1
export MOE_WEIGHT_CACHE_DIR=/dev/shm   # 方案 A；方案 B 改为磁盘路径

# Multi-host TPU 拓扑变量（替换 <HOST0_IP> 和 <HOST1_IP>）
export TPU_WORKER_HOSTNAMES="<HOST0_IP>,<HOST1_IP>"
export TPU_WORKER_ID=0
export TPU_PROCESS_ADDRESSES="<HOST0_IP>:8471,<HOST1_IP>:8471"
export TPU_PROCESS_PORT=8471
export TPU_HOST_BOUNDS="1,1,2"
export TPU_CHIPS_PER_HOST_BOUNDS="2,2,1"
export TPU_TOPOLOGY="2x2x2"
export TPU_ACCELERATOR_TYPE="tpu7x-16"
export TPU_SKIP_MDS_QUERY=true
export TPU_MULTIHOST_BACKEND=ray
export VLLM_HOST_IP=<HOST0_IP>
```

### Host 1（Ray Worker）

```bash
source ~/vllm_env/bin/activate

# 基础环境变量（同上）
export PJRT_DEVICE=TPU
export TPU_BACKEND_TYPE=jax
export JAX_PLATFORMS=
export MODEL_IMPL_TYPE=flax_nnx
export USE_MOE_EP_KERNEL=0
export USE_BATCHED_RPA_KERNEL=0
export HF_HUB_OFFLINE=1
export SKIP_JAX_PRECOMPILE=1
export VLLM_XLA_CHECK_RECOMPILATION=0

# ⚠️ GLM-5.1 三个必设环境变量
export MOE_REQUANTIZE_WEIGHT_DTYPE=float4_e2m1fn
export NEW_MODEL_DESIGN=1
export MOE_WEIGHT_CACHE_DIR=/dev/shm

# Multi-host TPU 拓扑变量
export TPU_WORKER_HOSTNAMES="<HOST0_IP>,<HOST1_IP>"
export TPU_WORKER_ID=1
export TPU_PROCESS_ADDRESSES="<HOST0_IP>:8471,<HOST1_IP>:8471"
export TPU_PROCESS_PORT=8471
export TPU_HOST_BOUNDS="1,1,2"
export TPU_CHIPS_PER_HOST_BOUNDS="2,2,1"
export TPU_TOPOLOGY="2x2x2"
export TPU_ACCELERATOR_TYPE="tpu7x-16"
export TPU_SKIP_MDS_QUERY=true
export TPU_MULTIHOST_BACKEND=ray
export VLLM_HOST_IP=<HOST1_IP>
```

> **关键差异**：
> - `TPU_WORKER_ID=0` vs `TPU_WORKER_ID=1`
> - `VLLM_HOST_IP` 分别设为各自 IP
> - `JAX_PLATFORMS=`（空）— multi-host 必须设为空

## Step 4: 启动 Ray 集群 + vLLM

### Host 0（先启动 Ray Head）

```bash
# --object-store-memory 限制 Ray plasma store 为 50 GB（留更多给 FP4 cache）
RAY_memory_monitor_refresh_ms=0 ray start --head \
  --port=6379 \
  --node-ip-address=${VLLM_HOST_IP} \
  --resources='{"TPU": 4}' \
  --object-store-memory=53687091200

sleep 20
ray status
```

### Host 1（启动 Ray Worker）

```bash
RAY_memory_monitor_refresh_ms=0 ray start \
  --address=<HOST0_IP>:6379 \
  --node-ip-address=${VLLM_HOST_IP} \
  --resources='{"TPU": 4}' \
  --object-store-memory=53687091200
```

### Host 0（确认集群就绪后启动 vLLM）

```bash
ray status   # 确认 2 nodes, 8 TPU

cd /tmp && mkdir -p /tmp/vllm-logs

nohup env \
  PJRT_DEVICE=TPU TPU_BACKEND_TYPE=jax JAX_PLATFORMS= \
  MODEL_IMPL_TYPE=flax_nnx USE_MOE_EP_KERNEL=0 USE_BATCHED_RPA_KERNEL=0 \
  HF_HUB_OFFLINE=1 SKIP_JAX_PRECOMPILE=1 VLLM_XLA_CHECK_RECOMPILATION=0 \
  MOE_REQUANTIZE_WEIGHT_DTYPE=float4_e2m1fn \
  NEW_MODEL_DESIGN=1 \
  MOE_WEIGHT_CACHE_DIR=/dev/shm \
  RAY_memory_monitor_refresh_ms=0 \
  vllm serve /mnt/data/GLM-5.1-FP8 \
    --served-model-name GLM-5.1-FP8 \
    --tensor-parallel-size 16 \
    --distributed-executor-backend ray \
    --quantization fp8 \
    --enforce-eager \
    --enable-prefix-caching \
    --enable-chunked-prefill \
    --max-model-len 16384 \
    --trust-remote-code \
    --gpu-memory-utilization 0.9 \
    --additional-config '{"sharding":{"sharding_strategy":{"enable_dp_attention":true,"expert_parallelism":16,"tensor_parallelism":1}},"replicate_attn_weights":"True","sparse_matmul":"True"}' \
    --host 0.0.0.0 --port 8000 \
  > /tmp/vllm-logs/serve.log 2>&1 &

# 等待就绪
until curl -s -o /dev/null -w '%{http_code}\n' http://localhost:8000/health | grep -q 200; do
  date; tail -1 /tmp/vllm-logs/serve.log 2>/dev/null; sleep 60
done
echo "Server ready"
```

> **关键参数说明**：
>
> | 参数 | 值 | 说明 |
> |------|-----|------|
> | `--tensor-parallel-size` | `16` | 总 device 数（2 nodes × 8 devices） |
> | `expert_parallelism` | `16` | 256 experts ÷ 16 = 每 device 16 experts |
> | `tensor_parallelism` | `1` | attention 权重仍然 replicate |
> | `--object-store-memory` | `53687091200` (50 GB) | 减小 Ray plasma store，为 FP4 cache 让出 /dev/shm |
> | `RAY_memory_monitor_refresh_ms=0` | | 禁用 Ray OOM monitor |
>
> **注意**：multi-host 不支持 `--async-scheduling`（Ray executor 限制）。

## Step 5: 验证和 Benchmark

### Smoke test

```bash
curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "GLM-5.1-FP8",
    "messages": [{"role": "user", "content": "What is 2+3? Answer with just the number."}],
    "max_tokens": 50
  }' | python3 -m json.tool
```

### Benchmark

```bash
python3 << 'PYEOF'
import requests, time, concurrent.futures

URL = "http://localhost:8000/v1/chat/completions"
MODEL = "GLM-5.1-FP8"
BASE = "The quick brown fox jumps over the lazy dog. "

def make_prompt(target_tokens):
    return BASE * (target_tokens // 10)

def send_request(prompt, output_len, rid):
    t0 = time.time()
    r = requests.post(URL, json={
        "model": MODEL, "messages": [{"role": "user", "content": prompt}],
        "max_tokens": output_len, "temperature": 0.7, "ignore_eos": True})
    data = r.json()
    t1 = time.time()
    if "error" in data:
        return {"error": data["error"]["message"][:200]}
    u = data.get("usage", {})
    return {"prompt": u.get("prompt_tokens",0), "completion": u.get("completion_tokens",0),
            "time": t1-t0, "tps": u.get("completion_tokens",0)/(t1-t0)}

def bench(input_tok, output_tok, conc, n):
    print("\n" + "="*60)
    print("P%d/D%d  concurrency=%d  requests=%d" % (input_tok, output_tok, conc, n))
    print("="*60)
    prompt = make_prompt(input_tok)
    t0 = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=conc) as ex:
        results = [f.result() for f in
            [ex.submit(send_request, prompt, output_tok, i) for i in range(n)]]
    ok = [r for r in results if "error" not in r]
    if not ok: print("  ALL FAILED"); return
    total_t = time.time() - t0
    total_c = sum(r["completion"] for r in ok)
    print("  Avg prompt tokens:     %d" % (sum(r["prompt"] for r in ok)/len(ok)))
    print("  Per-request tok/s:     %.1f" % (sum(r["tps"] for r in ok)/len(ok)))
    print("  Aggregate tok/s:       %.1f" % (total_c / total_t))

# Warmup
send_request(make_prompt(128), 32, -1)

bench(1024, 1024, 1, 3)
bench(1024, 1024, 4, 8)
bench(1024, 1024, 8, 16)
bench(1024, 1024, 16, 32)
bench(8192, 1024, 1, 3)
bench(8192, 1024, 4, 8)
PYEOF
```

### Multi-host 性能预估

> 基于 Qwen3.5 multi-host 实测的性能衰减规律（-15% ~ -21%），预估如下：

| 场景 | 预估 tok/s | 单机参考 | 预估 vs 单机 |
|------|----------:|--------:|------------:|
| P1K/D1K c=1 | ~22 | 28.4 | ~-22% |
| P1K/D1K c=4 | ~100 | 130.5 | ~-23% |
| P1K/D1K c=64 | ~1,250 | 1,570 | ~-20% |
| P1K/D1K c=1024 | ~5,200 | 6,504 | ~-20% |

> Multi-host 优势不在于更高吞吐，而在于 **更大 KV cache 容量**（1,536 GB HBM → 支持 16K+ context）和 **更低 MoE 内存压力**（EP=16 → 每 device 16 experts vs 32）。

### ⚠️ Multi-host 已知风险点

| # | 风险 | 排查方向 |
|---|------|---------|
| 1 | `--additional-config` 的 EP=16 是否在 Ray multi-host 下正确工作 | 检查 vLLM log 中 sharding 初始化 |
| 2 | FP4 cache 目录命名是否匹配 `ep16_tp1_...` | 如果 cache miss，检查实际查找路径 |
| 3 | `/dev/shm` 空间是否足够（FP4 cache 705 GB + Ray 50 GB） | 监控 `df -h /dev/shm`，如果 OOM 改用方案 B |
| 4 | `--enforce-eager` 在 Ray executor 下是否兼容 | 如果编译报错，尝试去掉 |
| 5 | `--enable-prefix-caching` / `--enable-chunked-prefill` 与 Ray 兼容性 | 如果启动挂起，去掉这两个 flag |

---

## 防火墙规则

PD 分离和多节点推理需要 VM 间内网通信：

```bash
gcloud compute firewall-rules create allow-vllm-internal \
    --project=${PROJECT_ID} \
    --network=${VPC_NAME} \
    --allow=tcp \
    --source-ranges=10.0.0.0/8 \
    --description="Allow all internal TCP for vLLM/Ray/TPU communication"
```

主要端口参考：

| 端口 | 用途 |
|------|------|
| 6379 | Ray GCS server（multi-host） |
| 7000 | PD 分离 proxy |
| 8000 | vLLM API（Prefill / 单机 / Host 0） |
| 8471 | libtpu coordinator（multi-host ICI） |
| 9000 | vLLM API（Decode） |
| 动态 | TPUConnector KV transfer / ZMQ side-channel |

---

## 资源清理

```bash
# 停止 vLLM / Ray
pgrep -f 'vllm|EngineCore|ray' | xargs -r kill -9

# 删除单机 VM + 数据盘
gcloud compute instances delete glm51-vm-01 --project=${PROJECT_ID} --zone=${ZONE} --quiet
gcloud compute disks delete glm51-data-01 --project=${PROJECT_ID} --zone=${ZONE} --quiet

# 删除 PD 分离 VM + 数据盘
gcloud compute instances delete glm51-prefill glm51-decode --project=${PROJECT_ID} --zone=${ZONE} --quiet
gcloud compute disks delete glm51-data-prefill glm51-data-decode --project=${PROJECT_ID} --zone=${ZONE} --quiet

# 删除 multi-host slice（MIG → Template → Workload Policy）
SLICE_NAME=glm51-slice
gcloud compute instance-groups managed delete ${SLICE_NAME}-mig --project=${PROJECT_ID} --zone=${ZONE} --quiet
gcloud compute instance-templates delete ${SLICE_NAME}-it --project=${PROJECT_ID} --quiet
gcloud compute resource-policies delete ${SLICE_NAME}-wp --project=${PROJECT_ID} --region=${ZONE%-*} --quiet
```

---

## Troubleshooting

| 症状 | 根因 | 修复 |
|---|---|---|
| **`CompileTimeHbmOom: Used 651G of 94.75G hbm`** | `MOE_REQUANTIZE_WEIGHT_DTYPE` 未设置，查找 FP8 cache → miss → OOM | `export MOE_REQUANTIZE_WEIGHT_DTYPE=float4_e2m1fn` |
| **`MLA models require NEW_MODEL_DESIGN=1`** | 缺 `NEW_MODEL_DESIGN` 环境变量 | `export NEW_MODEL_DESIGN=1` |
| **vLLM 卡死不动（0% CPU，线程全在 futex_wait）** | MoE prefetch deadlock：cache 从磁盘加载 或 cache 目录不完整 | 确保 cache 在 /dev/shm（tmpfs），且所有 76 层都有完整文件 |
| **FP4 cache 生成时 OOM Kill（exit 137）** | /dev/shm 有旧数据，挤占 worker 内存 | `rm -rf /dev/shm/*` 后重新生成 |
| **TPU device busy** | 上次 vLLM 异常退出，孤儿进程占 TPU | `pgrep -f 'EngineCore\|vllm' \| xargs -r kill -9` + `rm -f /tmp/libtpu_lockfile` |
| **`/dev/shm` 中出现多个 cache 目录** | 同时存在 FP4 和 FP8 cache | 删除 `ep8_tp1_gmm_ep_fp8e4m3_bsNone`，只保留 `fp4e2m1` |
| **vLLM import 报错 / namespace package** | 在 `~/vllm/` 或 `~/tpu-inference/` 目录下运行 | `cd /tmp` 后再运行 |
| **JAX 版本不对 / libtpu 报错** | vLLM 安装降级了 JAX | `uv pip install jax==0.9.2 jaxlib==0.9.2 libtpu==0.0.39 flax==0.12.4` |
| **PD: KV transfer 失败** | 防火墙未开放内网端口 | 检查 VPC 防火墙规则，允许 `10.0.0.0/8` TCP 全端口 |
| **Multi-host: Ray worker 被 kill** | Ray OOM monitor 误杀 | 设 `RAY_memory_monitor_refresh_ms=0` |
| **Multi-host: /dev/shm OOM** | FP4 cache + Ray Object Store 超出 /dev/shm | 减小 `--object-store-memory` 或 改用磁盘方案 B |

---

## 端到端流程总结

```
Step 1: 创建数据盘（≥2 TB）+ VM
    ↓
Step 2: 格式化挂载数据盘
    ↓
Step 3: 环境准备（uv + vLLM + tpu-inference feature/glm51-inference）
    ↓
Step 4: 生成 FP4 Cache（~28 min）+ 合并 Non-MoE（~2 min）   ← GLM-5.1 独有
    ↓
Step 5: 拷贝 Cache 到 /dev/shm（~4 min）+ 可选优化
    ↓
Step 6: 启动 vLLM（⚠️ 三个环境变量！~4-11 min）
    ↓
Step 7: curl 验证推理
    ↓
Step 8: Benchmark
```

> **首次部署总耗时**（不含模型下载）：FP4 生成 28 min + 合并 2 min + 拷贝 4 min + 启动 ~11 min ≈ **~45 min**
>
> **后续重启**：只需 Step 6-7（如果 /dev/shm cache 还在），**~4-11 min**

---

## 参考资料

| 资源 | 链接 |
|------|------|
| GLM-5.1 GKE 部署指南 | [README.md](README.md) |
| DeepSeek R1 FP4 推理指南 | [../DeepSeek-R1-671B-FP4/README.md](../DeepSeek-R1-671B-FP4/README.md) |
| Qwen3.5 TPU VM 部署指南 | [../Qwen3.5-397B-A17B-FP8/README-TPU-VM.md](../Qwen3.5-397B-A17B-FP8/README-TPU-VM.md) |
| GLM-5.1 HuggingFace 模型 | [zai-org/GLM-5.1-FP8](https://huggingface.co/zai-org/GLM-5.1-FP8) |
| tpu-inference 代码 | [yangwhale/tpu-inference](https://github.com/yangwhale/tpu-inference) branch: `feature/glm51-inference` |
