# Qwen3-Coder-480B FP8 Inference on TPU v7x — TPU VM 版

> TPU VM 端到端部署指南：从创建 VM 到完成 benchmark。
>
> **模型**: [Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8](https://huggingface.co/Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8)（~480 GB）
>
> **代码仓库**: [vllm-project/tpu-inference](https://github.com/vllm-project/tpu-inference)（main 分支）
>
> **模型存储**: `gs://aidc-tpu-data`（GCS 对象存储，模型权重和 cache 统一存放于此）
>
> GKE 版见同目录 [README.md](README.md)。

## 目录

- [Part 1: 单机推理](#part-1-单机推理)
- [Part 2: PD 分离 (1P1D)](#part-2-pd-分离-1p1d)
- [Part 3: 多节点推理 (TP=16)](#part-3-多节点推理-tp16)

---

## 环境变量

```bash
export PROJECT_ID=<your-project-id>
export ZONE=us-central1-c
export RESERVATION_NAME=<your-reservation>
export VPC_NAME=<your-vpc-name>
export SUBNET_NAME=<your-subnet-name>
export HF_TOKEN=<your-hf-token>
export MODEL_BUCKET=gs://aidc-tpu-data          # 模型权重 & cache 的 GCS 存储桶
export MODEL_NAME=qwen3-coder-480b-fp8           # 模型目录名
```

## 硬件要求

| 项目 | 单机 (Part 1 & 2) | 多节点 (Part 3) |
|------|-------------------|----------------|
| 机型 | `tpu7x-standard-4t` | `tpu7x-standard-4t` × 2 |
| TPU | v7x-8（4 chips, 8 devices） | v7x-16（8 chips, 16 devices） |
| HBM | 768 GB | 1,536 GB |
| 主机内存 | 944 GB | 944 GB × 2 |
| 数据盘 | ≥600 GB（模型 ~480 GB） | 共享盘或各自挂载 |

---

# Part 1: 单机推理

## Step 1: 创建数据盘和 VM

### 1.1 创建 Hyperdisk ML 数据盘

```bash
gcloud compute disks create qwen3-data-01 \
    --project=${PROJECT_ID} \
    --zone=${ZONE} \
    --type=hyperdisk-ml \
    --size=2TB \
    --provisioned-throughput=2500
```

### 1.2 创建 TPU VM

```bash
gcloud compute instances create qwen3-vm-01 \
    --project=${PROJECT_ID} \
    --zone=${ZONE} \
    --machine-type=tpu7x-standard-4t \
    --network-interface=network=${VPC_NAME},subnet=${SUBNET_NAME} \
    --maintenance-policy=TERMINATE \
    --provisioning-model=RESERVATION_BOUND \
    --reservation-affinity=specific \
    --reservation=${RESERVATION_NAME} \
    --create-disk=auto-delete=yes,boot=yes,size=500GB,type=hyperdisk-balanced,image=projects/ubuntu-os-accelerator-images/global/images/ubuntu-accel-2404-amd64-tpu-tpu7x-v20260422 \
    --disk=name=qwen3-data-01,device-name=data-disk,mode=rw,auto-delete=no \
    --no-shielded-secure-boot \
    --scopes=cloud-platform
```

### 1.3 SSH 连接

```bash
VM_IP=$(gcloud compute instances describe qwen3-vm-01 \
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

使用 `uv` 创建 Python 3.12 虚拟环境，安装 vLLM（TPU 版）和 tpu-inference：

```bash
# 安装 uv
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"  # uv 0.11+ 不再生成 ~/.local/bin/env，直接 export PATH

# 创建 Python 3.12 虚拟环境
uv venv ~/vllm_env --python 3.12
source ~/vllm_env/bin/activate

# 克隆 tpu-inference（获取 pinned vLLM 版本）
cd ~
git clone https://github.com/vllm-project/tpu-inference.git
cd tpu-inference

# 获取 pinned vLLM commit hash
export VLLM_COMMIT_HASH="$(cat .buildkite/vllm_lkg.version | tr -d '[:space:]')"
echo "vLLM commit: ${VLLM_COMMIT_HASH}"

# 克隆 vLLM 并 checkout 到 pinned 版本
cd ~
git clone https://github.com/vllm-project/vllm.git
cd vllm
git checkout "${VLLM_COMMIT_HASH}"

# 安装 vLLM（TPU target）
uv pip install -r requirements/tpu.txt --torch-backend=cpu
VLLM_TARGET_DEVICE="tpu" uv pip install -e .

# 修复 JAX 版本（vLLM 安装会降级 JAX 到 0.8.0，必须装回 0.9.2）
uv pip install jax==0.9.2 jaxlib==0.9.2 libtpu==0.0.39 flax==0.12.4

# 安装 tpu-inference（--no-deps 避免再次降级 JAX）
cd ~/tpu-inference
uv pip install -e . --no-deps
```

### 3.4 验证安装

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

预期输出（实际版本号会随 pinned vLLM commit 漂移）：
```
vllm: 0.20.x          # 例如 0.20.1rc1.dev14+gfd74c90d9.tpu（pinned commit fd74c90d9 实测值）
tpu_inference: 0.0.0   # 本地 install -e 时 pyproject.toml 的默认版本号
jax: 0.9.2
platform: TPU V7X      # 或包含 "TPU" 字串
devices: 8 x tpu
```

> **关于 GCE Metadata 404 ERROR**：执行验证脚本或启动 vLLM 时，可能看到 `tpu_info.py:40] Unable to poll TPU GCE Metadata. Got status code: 404 ...` 的多行 ERROR 日志，但**紧接着的 INFO 行会正确显示** `tpu_type=tpu7x-8 | num_chips=8`。这是 v7x VM 不暴露旧版 GCE TPU metadata endpoint 导致的非致命警告，可忽略。
>
> **关于 JAX 版本**：TPU v7x 需要 JAX 0.9.2 + libtpu 0.0.39。某些 vLLM commit 安装时会把 JAX 降级到 0.8.0（pinned commit `fd74c90d9` 实测**不会**降级，但旧/新版本可能会），所以保留 `uv pip install jax==0.9.2 ...` 这一步作为防御性安装：JAX 已是 0.9.2 时为 no-op，被降级时则恢复回 0.9.2。
>
> **关于 `--no-deps`**：tpu-inference 的 `pyproject.toml` 依赖 jax/jaxlib，不加 `--no-deps` 会再次触发 JAX 降级。

### 3.5 设置运行时环境变量

```bash
# 建议写入 ~/.bashrc 或每次启动前 source
export HF_TOKEN=${HF_TOKEN}
export JAX_PLATFORMS=tpu,cpu
export TPU_BACKEND_TYPE=jax
export PJRT_DEVICE=TPU
export MODEL_IMPL_TYPE=vllm
export USE_MOE_EP_KERNEL=0
export USE_BATCHED_RPA_KERNEL=0
```

### 3.6 选择模型存放位置（**根据机器 RAM 大小决定**）

模型权重统一存放在 GCS，每次启动 VM 后从 GCS 拷贝到本地。落点有两种方案，**必须根据机器 RAM 大小选择**。

> ⚠️ **必读 — 主机 RAM OOM 风险**：
> vLLM 加载 480GB 模型时，EngineCore 进程会占用 **~510 GiB anon RSS**（实测 `anon-rss=511 GiB`，因 safetensors 被 copy 到进程 heap 而非纯 mmap，加上 MoE re-quantization 产生的临时 buffer）。如果模型同时也在 `/dev/shm`（tmpfs，占 ~450 GiB RAM），合计 ~960 GiB，会**超出 944 GiB RAM 的标准 `tpu7x-standard-4t` 触发 OOM**：
> ```
> Out of memory: Killed process VLLM::EngineCore total-vm:690813156kB anon-rss:511342372kB
> ```
> vLLM 自己也会在 weight loading 阶段警告（容易被淹）：
> ```
> INFO weight_utils.py:934] Auto-prefetch is disabled because the filesystem (TMPFS)
> is not a recognized network FS and the checkpoint size (449.04 GiB) exceeds 90%
> of available RAM (477.97 GiB).
> ```
> 看到这条警告就说明应该走方案 B。
>
> ⚠️ 注意：常见问题 6.1 提到加 `--enable-expert-parallel` 防 OOM，那是针对 **TPU HBM OOM**。本案是**主机 RAM OOM**，必须通过控制 `/dev/shm` 占用解决，加 EP 参数无效。

#### 方案 A: RAM ≥ 1.5 TiB 的机器 — 用 /dev/shm（最快）

模型放 tmpfs，加载 ~10 秒：

```bash
# 扩容 /dev/shm（默认 ~472 GB，需要容纳 480 GB 模型 + vLLM IPC）
sudo mount -o remount,size=700G /dev/shm

# 从 GCS 拷贝模型权重到 /dev/shm（必须用 gcloud storage cp，最快）
gcloud storage cp -r ${MODEL_BUCKET}/${MODEL_NAME} /dev/shm/

export MODEL_DIR=/dev/shm/${MODEL_NAME}
```

#### 方案 B: RAM < 1.5 TiB（含默认 `tpu7x-standard-4t` 944 GiB） — 用 /mnt/data

模型放持久化 nvme 数据盘，加载 ~3.5 min（含 JAX 编译总启动 ~10 min），但能跑通且重启免重拷：

```bash
# 不要扩 /dev/shm（保持默认 ~472 GB，让出主机 RAM 给 vLLM 进程）

# 从 GCS 拷贝模型权重到 /mnt/data
gcloud storage cp -r ${MODEL_BUCKET}/${MODEL_NAME} /mnt/data/

export MODEL_DIR=/mnt/data/${MODEL_NAME}
```

#### 验证（两种方案通用）

```bash
ls ${MODEL_DIR}/*.safetensors | wc -l   # 应为 49
ls ${MODEL_DIR}/{tokenizer.json,vocab.json,tokenizer_config.json}
du -sh ${MODEL_DIR}                      # 应为 ~450 GB
```

> **为什么 /dev/shm 最快**：内存文件系统读取速度 ~50 GB/s，比 Hyperdisk ML（2.4 GB/s）快 20 倍，模型加载时间从 ~3.5 min 缩短到 ~10 秒。
>
> **为什么用 `gcloud storage cp`**：自动多线程分片传输，比 `gsutil cp` 快 2-3 倍。实测吞吐：GCS → /dev/shm 平均 **10.5 GiB/s**（449 GiB 用 48 秒）；GCS → /mnt/data 平均 **3.3 GiB/s**（用 2m20s，受 hyperdisk 写入带宽限制）。
>
> **注意**：`/dev/shm` 是 tmpfs，VM 重启后数据会丢失需要重拷；`/mnt/data` 持久化，重启后无需重拷。

---

## Step 4: 验证模型权重

模型已在 Step 3.6 从 GCS 拷贝到 `${MODEL_DIR}`（方案 A: `/dev/shm/${MODEL_NAME}`，方案 B: `/mnt/data/${MODEL_NAME}`）。验证：

```bash
ls ${MODEL_DIR}/*.safetensors | wc -l   # 应为 49
ls ${MODEL_DIR}/{tokenizer.json,vocab.json,tokenizer_config.json}
```

> **首次上传模型到 GCS**：如果 GCS 桶里还没有模型权重，先在任意机器上从 HuggingFace 下载后上传：
> ```bash
> pip install -U "huggingface_hub[hf_transfer]"
> HF_HUB_ENABLE_HF_TRANSFER=1 huggingface-cli download \
>   Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8 --local-dir /tmp/qwen3-coder-480b-fp8
> gcloud storage cp -r /tmp/qwen3-coder-480b-fp8 ${MODEL_BUCKET}/
> ```

---

## Step 5: 启动 vLLM

> **重要**：必须 `cd /tmp` 后再运行 vLLM，否则 `~/vllm/` 目录会被 Python 当作 namespace package，导致 import 错误。

```bash
source ~/vllm_env/bin/activate
cd /tmp && mkdir -p /tmp/vllm-logs

# MODEL_DIR 已在 Step 3.6 设置（方案 A: /dev/shm/...，方案 B: /mnt/data/...）
export HF_HUB_OFFLINE=1

nohup env \
  SKIP_JAX_PRECOMPILE=1 \
  VLLM_XLA_CHECK_RECOMPILATION=0 \
  MODEL_IMPL_TYPE=vllm \
  HF_HUB_OFFLINE=1 \
  vllm serve ${MODEL_DIR} \
    --served-model-name Qwen3-Coder-480B-FP8 \
    --seed 42 \
    --max-model-len 10240 \
    --max-num-batched-tokens 8192 \
    --max-num-seqs 512 \
    --no-enable-prefix-caching \
    --tensor-parallel-size 8 \
    --kv-cache-dtype fp8 \
    --gpu-memory-utilization 0.95 \
    --async-scheduling \
    --enable-expert-parallel \
    --port 8000 --host 0.0.0.0 \
  > /tmp/vllm-logs/serve.log 2>&1 &

# 等待就绪
#   方案 A (/dev/shm): 约 7-15 min
#   方案 B (/mnt/data): 约 10-15 min（实测 637s / 10.6 min，含 JAX 首次编译）
until curl -s -o /dev/null -w '%{http_code}\n' http://localhost:8000/health | grep -q 200; do
  date; sleep 30
done
echo "✅ Server ready"
```

---

## Step 6: 验证推理

```bash
# Smoke test
curl -s http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen3-Coder-480B-FP8",
    "prompt": "def fibonacci(n):",
    "max_tokens": 50, "temperature": 0.0
  }' | python3 -c "import sys,json; print(json.load(sys.stdin)['choices'][0]['text'])"

# Chat API
curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen3-Coder-480B-FP8",
    "messages": [{"role": "user", "content": "用中文写一个判断质数的 Python 函数。"}],
    "max_tokens": 512
  }' | python3 -m json.tool
```

---

## Step 7: Benchmark

### 7.1 安装 benchmark 工具

```bash
cd ~
git clone https://github.com/kimbochen/bench_serving.git
```

### 7.2 1K/1K Benchmark（CI 标准配置）

```bash
python3 bench_serving/benchmark_serving.py \
  --model="Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8" \
  --backend=vllm \
  --host=127.0.0.1 --port=8000 \
  --dataset-name=random \
  --random-input-len=1024 --random-output-len=1024 \
  --random-range-ratio=0.8 \
  --num-prompts=320 --max-concurrency=64 \
  --request-rate=inf --ignore-eos
```

### 7.3 1K/8K Benchmark（长输出）

```bash
python3 bench_serving/benchmark_serving.py \
  --model="Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8" \
  --backend=vllm \
  --host=127.0.0.1 --port=8000 \
  --dataset-name=random \
  --random-input-len=1024 --random-output-len=8192 \
  --random-range-ratio=0.8 \
  --num-prompts=128 --max-concurrency=64 \
  --request-rate=inf --ignore-eos
```

### 7.4 8K/1K Benchmark（长输入）

```bash
python3 bench_serving/benchmark_serving.py \
  --model="Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8" \
  --backend=vllm \
  --host=127.0.0.1 --port=8000 \
  --dataset-name=random \
  --random-input-len=8192 --random-output-len=1024 \
  --random-range-ratio=0.8 \
  --num-prompts=128 --max-concurrency=64 \
  --request-rate=inf --ignore-eos
```

### 7.5 全并发扫描

```bash
for c in 1 4 16 64; do
  echo "=== Concurrency $c ==="
  vllm bench serve \
    --model="Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8" \
    --num-warmups 3 \
    --dataset-name random \
    --random-input-len 1024 --random-output-len 1024 \
    --num-prompts $((c * 4)) \
    --max-concurrency $c \
    --request-rate inf --ignore-eos \
    --host localhost --port 8000 \
    --result-file qwen3coder_1k1k_c${c}.json
  sleep 30
done
```

### 7.6 预期性能参考（GKE 实测值）

| Workload | c=1 | c=4 | c=16 | c=64 |
|----------|----:|----:|-----:|-----:|
| 1K/1K | 48 tok/s | 177 | 602 | 1478 |
| 1K/8K | 47.5 | 178 | 621 | 1623 |
| 8K/1K | 46.4 | 162 | 483 | 943 |

> TPU VM 上的性能应与 GKE Pod 一致（同硬件、同软件栈）。

---

# Part 2: PD 分离 (1P1D)

> 2 台 TPU v7x-8 VM：一台跑 Prefill（kv_producer），一台跑 Decode（kv_consumer），通过 VPC 内网传输 KV cache。
>
> **架构说明**：PD 分离 **不需要 Ray**。两个 vLLM 实例完全独立运行，通过 TPUConnector（JAX transfer server + ZMQ side-channel）直接 P2P 传输 KV cache，`toy_proxy_server.py` 负责请求路由。

## Step 1: 创建 2 台 VM

复用 Part 1 的方式，创建 Prefill 和 Decode 两台 VM：

```bash
# Prefill VM
gcloud compute disks create qwen3-data-prefill \
    --project=${PROJECT_ID} --zone=${ZONE} \
    --type=hyperdisk-ml --size=2TB --provisioned-throughput=2500

gcloud compute instances create qwen3-prefill \
    --project=${PROJECT_ID} --zone=${ZONE} \
    --machine-type=tpu7x-standard-4t \
    --network-interface=network=${VPC_NAME},subnet=${SUBNET_NAME} \
    --maintenance-policy=TERMINATE \
    --provisioning-model=RESERVATION_BOUND \
    --reservation-affinity=specific \
    --reservation=${RESERVATION_NAME} \
    --create-disk=auto-delete=yes,boot=yes,size=500GB,type=hyperdisk-balanced,image=projects/ubuntu-os-accelerator-images/global/images/ubuntu-accel-2404-amd64-tpu-tpu7x-v20260422 \
    --disk=name=qwen3-data-prefill,device-name=data-disk,mode=rw,auto-delete=no \
    --no-shielded-secure-boot \
    --scopes=cloud-platform

# Decode VM
gcloud compute disks create qwen3-data-decode \
    --project=${PROJECT_ID} --zone=${ZONE} \
    --type=hyperdisk-ml --size=2TB --provisioned-throughput=2500

gcloud compute instances create qwen3-decode \
    --project=${PROJECT_ID} --zone=${ZONE} \
    --machine-type=tpu7x-standard-4t \
    --network-interface=network=${VPC_NAME},subnet=${SUBNET_NAME} \
    --maintenance-policy=TERMINATE \
    --provisioning-model=RESERVATION_BOUND \
    --reservation-affinity=specific \
    --reservation=${RESERVATION_NAME} \
    --create-disk=auto-delete=yes,boot=yes,size=500GB,type=hyperdisk-balanced,image=projects/ubuntu-os-accelerator-images/global/images/ubuntu-accel-2404-amd64-tpu-tpu7x-v20260422 \
    --disk=name=qwen3-data-decode,device-name=data-disk,mode=rw,auto-delete=no \
    --no-shielded-secure-boot \
    --scopes=cloud-platform
```

> **省盘方案**：如果不需要读写分离，可以用 **1 块 Hyperdisk ML** 设为 `READ_ONLY_MANY` 模式同时挂载到两台 VM（只读），前提是模型权重已提前写好。参见 [TPU-VM README](../TPU-VM/README.md) 的 Hyperdisk ML 读写模式说明。

## Step 2: 两台 VM 分别执行环境准备

在两台 VM 上分别执行 Part 1 的 Step 2（挂载盘）+ Step 3（系统配置 + 裸机安装 vLLM + 拷贝模型）。

> ⚠️ **同样适用 Part 1 Step 3.6 的 RAM 大小判断**：默认 `tpu7x-standard-4t` 944 GiB 走 /dev/shm 会 OOM，需用 `/mnt/data`。下方示例假设走方案 A（/dev/shm），如果你的机器是 944 GiB，请改为方案 B。

```bash
# 方案 A 示例（机器 RAM ≥ 1.5 TiB）：
sudo mount -o remount,size=700G /dev/shm
gcloud storage cp -r ${MODEL_BUCKET}/${MODEL_NAME} /dev/shm/
export MODEL_DIR=/dev/shm/${MODEL_NAME}

# 方案 B 示例（机器 RAM < 1.5 TiB，含默认 944 GiB）：
# gcloud storage cp -r ${MODEL_BUCKET}/${MODEL_NAME} /mnt/data/
# export MODEL_DIR=/mnt/data/${MODEL_NAME}
```

## Step 3: 获取内网 IP

```bash
PREFILL_IP=$(gcloud compute instances describe qwen3-prefill \
    --project=${PROJECT_ID} --zone=${ZONE} \
    --format="value(networkInterfaces[0].networkIP)")
DECODE_IP=$(gcloud compute instances describe qwen3-decode \
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
  SKIP_JAX_PRECOMPILE=1 VLLM_XLA_CHECK_RECOMPILATION=0 \
  MODEL_IMPL_TYPE=vllm HF_HUB_OFFLINE=1 \
  vllm serve /dev/shm/qwen3-coder-480b-fp8 \
    --served-model-name Qwen3-Coder-480B-FP8 \
    --seed 42 \
    --max-model-len 10240 \
    --max-num-batched-tokens 8192 \
    --max-num-seqs 512 \
    --no-enable-prefix-caching \
    --tensor-parallel-size 8 \
    --kv-cache-dtype fp8 \
    --gpu-memory-utilization 0.70 \
    --async-scheduling \
    --enable-expert-parallel \
    --port 8000 --host 0.0.0.0 \
    --kv-transfer-config '{"kv_connector":"TPUConnector","kv_connector_module_path":"tpu_inference.distributed.tpu_connector","kv_role":"kv_producer"}' \
  > /tmp/vllm-logs/serve.log 2>&1 &
```

关键差异：`gpu-memory-utilization=0.70`（留 30% 给 KV transfer buffer），`kv_role=kv_producer`。

## Step 5: 启动 Decode 实例

SSH 到 Decode VM：

```bash
source ~/vllm_env/bin/activate
cd /tmp && mkdir -p /tmp/vllm-logs

nohup env \
  SKIP_JAX_PRECOMPILE=1 VLLM_XLA_CHECK_RECOMPILATION=0 \
  MODEL_IMPL_TYPE=vllm HF_HUB_OFFLINE=1 \
  vllm serve /dev/shm/qwen3-coder-480b-fp8 \
    --served-model-name Qwen3-Coder-480B-FP8 \
    --seed 42 \
    --max-model-len 10240 \
    --max-num-batched-tokens 8192 \
    --max-num-seqs 512 \
    --no-enable-prefix-caching \
    --tensor-parallel-size 8 \
    --kv-cache-dtype fp8 \
    --gpu-memory-utilization 0.90 \
    --async-scheduling \
    --enable-expert-parallel \
    --port 9000 --host 0.0.0.0 \
    --kv-transfer-config '{"kv_connector":"TPUConnector","kv_connector_module_path":"tpu_inference.distributed.tpu_connector","kv_role":"kv_consumer"}' \
  > /tmp/vllm-logs/serve.log 2>&1 &
```

关键差异：`gpu-memory-utilization=0.90`，`kv_role=kv_consumer`，`port=9000`。

## Step 6: 启动 Proxy

在 Prefill VM 上启动 proxy，连接两个实例：

```bash
source ~/vllm_env/bin/activate

# 设置 Decode VM 内网 IP（Step 3 获取的 networkIP）
export DECODE_IP=<decode-vm-internal-ip>

python3 ~/tpu-inference/tpu_inference/examples/disagg/toy_proxy_server.py \
  --host 0.0.0.0 --port 7000 \
  --prefiller-hosts localhost --prefiller-ports 8000 \
  --decoder-hosts ${DECODE_IP} --decoder-ports 9000
```

> **注意**：`${DECODE_IP}` 使用 VPC 内网 IP（Step 3 获取的 `networkIP`），不是外网 IP。确保两台 VM 在同一 VPC 且防火墙允许端口 8000/9000/7000。

## Step 7: PD 分离 Benchmark

在 Prefill VM 上对 proxy 端口发请求：

```bash
# 1K/1K c=1
vllm bench serve \
  --model Qwen3-Coder-480B-FP8 \
  --dataset-name random \
  --random-input-len 1024 --random-output-len 1024 \
  --num-prompts 4 --max-concurrency 1 \
  --request-rate inf --ignore-eos \
  --host localhost --port 7000

# 8K/1K c=4
vllm bench serve \
  --model Qwen3-Coder-480B-FP8 \
  --dataset-name random \
  --random-input-len 8192 --random-output-len 1024 \
  --num-prompts 16 --max-concurrency 4 \
  --request-rate inf --ignore-eos \
  --host localhost --port 7000

# 1K/8K c=64
vllm bench serve \
  --model Qwen3-Coder-480B-FP8 \
  --dataset-name random \
  --num-warmups 10 \
  --random-input-len 1024 --random-output-len 8192 \
  --num-prompts 256 --max-concurrency 64 \
  --request-rate inf --ignore-eos \
  --host localhost --port 7000 \
  --metric-percentiles 90,99
```

### PD 分离预期性能参考（GKE 实测值）

| 配置 | TTFT (med) | TPOT (med) | Output tok/s | vs 单实例 |
|------|----------:|----------:|------------:|----------|
| 单实例 1K/1K c=1 | 95 ms | 20.6 ms | 48 | baseline |
| 1P1D 1K/1K c=1 | 281 ms | 18.3 ms | 53.8 | TPOT -11% |
| 单实例 8K/1K c=4 | 1495 ms | 23.2 ms | 162 | baseline |
| 1P1D 8K/1K c=4 | 2908 ms | 20.6 ms | 170 | TPOT -11% |

---

# Part 3: 多节点推理 (TP=16)

> 2 台 TPU v7x-8 VM 组成 v7x-16（8 chips, 16 devices），通过 Ray 分布式 + DCN 跨节点通信。
>
> **注意**：GKE 实测结论是 multi-host TP=16 的 output throughput 全场景比单机 v7x-8 差 17~63%，不推荐生产用。此部分主要用于验证和测试。

## Step 1: 创建 2 台 VM

与 Part 2 相同方式创建 2 台 VM（qwen3-host0, qwen3-host1），确保在同一 VPC 同一 zone。

```bash
for i in 0 1; do
  gcloud compute disks create qwen3-data-host${i} \
      --project=${PROJECT_ID} --zone=${ZONE} \
      --type=hyperdisk-ml --size=2TB --provisioned-throughput=2500

  gcloud compute instances create qwen3-host${i} \
      --project=${PROJECT_ID} --zone=${ZONE} \
      --machine-type=tpu7x-standard-4t \
      --network-interface=network=${VPC_NAME},subnet=${SUBNET_NAME} \
      --maintenance-policy=TERMINATE \
      --provisioning-model=RESERVATION_BOUND \
      --reservation-affinity=specific \
      --reservation=${RESERVATION_NAME} \
      --create-disk=auto-delete=yes,boot=yes,size=500GB,type=hyperdisk-balanced,image=projects/ubuntu-os-accelerator-images/global/images/ubuntu-accel-2404-amd64-tpu-tpu7x-v20260422 \
      --disk=name=qwen3-data-host${i},device-name=data-disk,mode=rw,auto-delete=no \
      --no-shielded-secure-boot \
      --scopes=cloud-platform
done
```

## Step 2: 环境准备

在两台 VM 上分别执行 Part 1 的 Step 2 + Step 3（挂载盘、系统配置、裸机安装 vLLM、GCS 拷贝模型到 SHM）。

## Step 3: 获取内网 IP

```bash
HOST0_IP=$(gcloud compute instances describe qwen3-host0 \
    --project=${PROJECT_ID} --zone=${ZONE} \
    --format="value(networkInterfaces[0].networkIP)")
HOST1_IP=$(gcloud compute instances describe qwen3-host1 \
    --project=${PROJECT_ID} --zone=${ZONE} \
    --format="value(networkInterfaces[0].networkIP)")
echo "Host0: ${HOST0_IP}, Host1: ${HOST1_IP}"
```

## Step 4: 设置 Multi-host TPU 环境变量（两台 VM 都执行）

Multi-host 推理需要手动设置 TPU 拓扑环境变量，让两台 VM 的 TPU 识别为一个 v7x-16 集群。

### Host 0（Ray Head + vLLM API Server）

```bash
source ~/vllm_env/bin/activate

# 基础环境变量（同 Part 1 Step 3.5）
export PJRT_DEVICE=TPU
export TPU_BACKEND_TYPE=jax
export JAX_PLATFORMS=
export MODEL_IMPL_TYPE=vllm
export USE_MOE_EP_KERNEL=0
export USE_BATCHED_RPA_KERNEL=0
export HF_HUB_OFFLINE=1
export SKIP_JAX_PRECOMPILE=1
export VLLM_XLA_CHECK_RECOMPILATION=0

# Multi-host TPU 拓扑变量
export TPU_WORKER_HOSTNAMES="${HOST0_IP},${HOST1_IP}"
export TPU_WORKER_ID=0
export TPU_PROCESS_ADDRESSES="${HOST0_IP}:8471,${HOST1_IP}:8471"
export TPU_PROCESS_PORT=8471
export TPU_HOST_BOUNDS="1,1,2"
export TPU_CHIPS_PER_HOST_BOUNDS="2,2,1"
export TPU_TOPOLOGY="2x2x2"
export TPU_ACCELERATOR_TYPE="tpu7x-16"
export TPU_SKIP_MDS_QUERY=true
export TPU_MULTIHOST_BACKEND=ray
export VLLM_HOST_IP=${HOST0_IP}
```

### Host 1（Ray Worker）

```bash
source ~/vllm_env/bin/activate

# 基础环境变量（同上）
export PJRT_DEVICE=TPU
export TPU_BACKEND_TYPE=jax
export JAX_PLATFORMS=
export MODEL_IMPL_TYPE=vllm
export USE_MOE_EP_KERNEL=0
export USE_BATCHED_RPA_KERNEL=0
export HF_HUB_OFFLINE=1
export SKIP_JAX_PRECOMPILE=1
export VLLM_XLA_CHECK_RECOMPILATION=0

# Multi-host TPU 拓扑变量
export TPU_WORKER_HOSTNAMES="${HOST0_IP},${HOST1_IP}"
export TPU_WORKER_ID=1
export TPU_PROCESS_ADDRESSES="${HOST0_IP}:8471,${HOST1_IP}:8471"
export TPU_PROCESS_PORT=8471
export TPU_HOST_BOUNDS="1,1,2"
export TPU_CHIPS_PER_HOST_BOUNDS="2,2,1"
export TPU_TOPOLOGY="2x2x2"
export TPU_ACCELERATOR_TYPE="tpu7x-16"
export TPU_SKIP_MDS_QUERY=true
export TPU_MULTIHOST_BACKEND=ray
export VLLM_HOST_IP=${HOST1_IP}
```

> **关键差异**：`TPU_WORKER_ID=0` vs `TPU_WORKER_ID=1`，`VLLM_HOST_IP` 分别设为各自 IP。

## Step 5: 启动 Ray 集群 + vLLM

### Host 0（先启动 Ray Head，再启动 vLLM）

```bash
# 启动 Ray Head（daemon 模式，不加 --block）
ray start --head --port=6379 --node-ip-address=${VLLM_HOST_IP} --resources='{"TPU": 4}'
sleep 20
ray status   # 确认 head 启动

# 等 Host 1 的 Ray Worker 加入后再启动 vLLM
# 检查 ray status 显示 2 个 node, 总共 8 TPU
```

### Host 1（启动 Ray Worker）

```bash
# 加入 Ray 集群（--block 保持前台）
ray start --address=${HOST0_IP}:6379 --node-ip-address=${VLLM_HOST_IP} --resources='{"TPU": 4}' --block
```

### Host 0（确认集群就绪后启动 vLLM）

```bash
# 确认 2 nodes, 8 TPU
ray status

# 启动 vLLM（TP=16, Ray executor）— 必须 cd /tmp 避免 namespace package 问题
cd /tmp
vllm serve /dev/shm/qwen3-coder-480b-fp8 \
  --served-model-name Qwen3-Coder-480B-FP8 \
  --seed 42 \
  --tensor-parallel-size 16 \
  --distributed-executor-backend ray \
  --max-model-len 10240 \
  --max-num-batched-tokens 8192 \
  --max-num-seqs 512 \
  --no-enable-prefix-caching \
  --kv-cache-dtype fp8 \
  --gpu-memory-utilization 0.9 \
  --enable-expert-parallel \
  --host 0.0.0.0 --port 8000
```

> **注意**：multi-host 不支持 `--async-scheduling`（Ray executor 限制）。启动时间约 12-13 min。

等待日志输出 `Application startup complete`。

## Step 6: 验证和 Benchmark

在 Host 0 上执行：

```bash
# Smoke test
curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"Qwen3-Coder-480B-FP8","messages":[{"role":"user","content":"用一句话介绍 TPU v7"}],"max_tokens":100}' | python3 -m json.tool

# Warmup
for inp in 1024 8192; do
  for out in 64 1024; do
    vllm bench serve --backend vllm \
      --model Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8 \
      --served-model-name Qwen3-Coder-480B-FP8 \
      --base-url http://localhost:8000 --endpoint /v1/completions \
      --dataset-name random --random-input-len $inp --random-output-len $out \
      --num-prompts 2 --max-concurrency 1 --ignore-eos > /dev/null 2>&1
  done
done

# Benchmark
for scenario in "1024 1024 1" "1024 1024 4" "1024 1024 16" "8192 1024 4" "8192 1024 16"; do
  read inp out conc <<< "$scenario"
  echo "=== ${inp}/${out} c=${conc} ==="
  vllm bench serve --backend vllm \
    --model Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8 \
    --served-model-name Qwen3-Coder-480B-FP8 \
    --base-url http://localhost:8000 --endpoint /v1/completions \
    --dataset-name random --random-input-len $inp --random-output-len $out \
    --num-prompts $((conc * 4)) --max-concurrency $conc \
    --ignore-eos 2>&1 | tail -22
done
```

### Multi-host 预期性能参考（GKE 实测值）

| 场景 | 单机 v7x-8 (TP=8) | 多机 v7x-16 (TP=16) | 差异 |
|------|------------------:|-------------------:|----:|
| 1K/1K c=1 | 48 | 37.5 | -22% |
| 1K/1K c=4 | 177 | 98.4 | -44% |
| 1K/1K c=16 | 602 | 220 | -63% |
| 8K/1K c=4 | 162 | 134 | -17% |
| 8K/1K c=16 | 483 | 223 | -54% |

> **结论**：Multi-host TP=16 因跨节点 DCN 通信开销，throughput 全面低于单机。推荐用 2 × v7x-8 Data Parallel 替代。

---

## 防火墙规则

PD 分离和多节点推理需要 VM 间内网通信。建议允许内网全端口 TCP（TPUConnector 的 KV transfer 和 ZMQ side-channel 使用动态端口）：

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
| 8471 | libtpu coordinator（multi-host DCN） |
| 9000 | vLLM API（Decode） |
| 动态 | TPUConnector KV transfer / ZMQ side-channel |

---

## 资源清理

```bash
# 删除 VM（数据盘设了 auto-delete=no，不会自动删）
for vm in qwen3-vm-01 qwen3-prefill qwen3-decode qwen3-host0 qwen3-host1; do
  gcloud compute instances delete $vm \
      --project=${PROJECT_ID} --zone=${ZONE} --quiet 2>/dev/null
done

# 删除数据盘（确认不再需要）
for disk in qwen3-data-01 qwen3-data-prefill qwen3-data-decode qwen3-data-host0 qwen3-data-host1; do
  gcloud compute disks delete $disk \
      --project=${PROJECT_ID} --zone=${ZONE} --quiet 2>/dev/null
done
```

---

## 常见问题

### 1. vLLM 启动 OOM

**TPU HBM OOM**（每个 device 显存超）：必须加 `--enable-expert-parallel`，否则 experts 在每个 device 都 replicate。

**主机 RAM OOM**（被 oom-killer 杀掉、`anon-rss=511 GiB` 类似日志）：是模型同时占用 `/dev/shm`（450 GiB）+ vLLM 进程 anon RAM（510 GiB）超出物理内存。**`--enable-expert-parallel` 不能解决本案**。详见 Step 3.6 的方案 A vs 方案 B 选择。

### 2. vLLM import 报错 / namespace package 冲突

如果在 `~/vllm/` 目录下运行 `vllm serve`，Python 会把 `~/vllm/` 当作 namespace package，导致 import 错误。解决：

```bash
cd /tmp   # 必须离开 ~/vllm/ 目录
vllm serve ...
```

### 3. JAX 版本不对 / libtpu 报错

vLLM 安装会把 JAX 降级到 0.8.0，必须手动覆盖安装：

```bash
source ~/vllm_env/bin/activate
uv pip install jax==0.9.2 jaxlib==0.9.2 libtpu==0.0.39 flax==0.12.4
python3 -c "import jax; print(jax.__version__)"  # 应为 0.9.2
```

### 4. libtpu lockfile 报错

```bash
pkill -9 -f "vllm|EngineCore"; sleep 3; rm -f /tmp/libtpu_lockfile /tmp/.vllm_ipc_*
```

### 5. PD 分离 Decode 连不上 Prefill

检查 VPC 内网 IP 和防火墙规则，确保端口 8000/9000 可达：

```bash
# 在 Decode VM 上测试
curl -s http://${PREFILL_IP}:8000/health
```

### 6. Multi-host AttributeError: d.coords

TPU 环境变量没设对。确认 `TPU_WORKER_HOSTNAMES` 包含两台 VM 的 IP，且 `TPU_WORKER_ID` 在两台分别为 0 和 1。

### 7. 模型权重缺 tokenizer

```bash
cd /dev/shm/qwen3-coder-480b-fp8/
for f in tokenizer.json tokenizer_config.json vocab.json; do
  curl -sL -o $f https://huggingface.co/Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8/resolve/main/$f
done
```
