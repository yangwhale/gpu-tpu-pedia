# Qwen3-Coder-480B FP8 Inference on TPU v7x — TPU VM 版

> TPU VM 端到端部署指南：从创建 VM 到完成 benchmark。
>
> **模型**: [Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8](https://huggingface.co/Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8)（~480 GB）
>
> **代码仓库**: [vllm-project/tpu-inference](https://github.com/vllm-project/tpu-inference)（main 分支）
>
> **模型存储**: `gs://aidc-tpu-data/models`（GCS 对象存储，模型权重统一存放于此）
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
export MODEL_BUCKET=gs://aidc-tpu-data/models    # 模型权重 GCS 路径
export MODEL_NAME=Qwen3-Coder-480B-A35B-FP8      # 模型目录名（与 GCS 一致）
```

## 硬件要求

| 项目 | 单机 (Part 1 & 2) | 多节点 (Part 3) |
|------|-------------------|----------------|
| 机型 | `tpu7x-standard-4t` | `tpu7x-standard-4t` × 2 |
| TPU | v7x-8（4 chips, 8 devices） | v7x-16（8 chips, 16 devices） |
| HBM | 768 GB | 1,536 GB |
| 主机内存 | 944 GB | 944 GB × 2 |
| 启动盘 | 1 TB Hyperdisk Balanced | 1 TB × 2 |
| 数据盘 | ≥600 GB（模型 ~480 GB） | 不需要（模型拷到 boot disk ~/models/） |

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
    --create-disk=auto-delete=yes,boot=yes,size=1000GB,type=hyperdisk-balanced,image=projects/ubuntu-os-accelerator-images/global/images/family/ubuntu-accel-2404-amd64-tpu-tpu7x \
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
source $HOME/.local/bin/env

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

预期输出：
```
vllm: 0.9.x
tpu_inference: 0.1.0
jax: 0.9.2
platform: TpuDevice
devices: 8 x tpu
```

> **关于 JAX 版本**：vLLM 的 `requirements/tpu.txt` 会安装 JAX 0.8.0，但 TPU v7x 需要 JAX 0.9.2 + libtpu 0.0.39。安装 vLLM 后必须手动覆盖安装正确版本。
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

### 3.6 扩容 /dev/shm 并从 GCS 拷贝模型

模型权重统一存放在 GCS，每次启动 VM 后拷贝到 `/dev/shm`（内存文件系统）以获得最快的加载速度。

```bash
# 扩容 /dev/shm（默认 ~472 GB，需要容纳 480 GB 模型 + vLLM IPC）
sudo mount -o remount,size=700G /dev/shm

# 从 GCS 拷贝模型权重到 /dev/shm（必须用 gcloud storage cp，最快）
gcloud storage cp -r ${MODEL_BUCKET}/${MODEL_NAME} /dev/shm/

# 验证
ls /dev/shm/${MODEL_NAME}/*.safetensors | wc -l   # 应为 49
ls /dev/shm/${MODEL_NAME}/{tokenizer.json,vocab.json,tokenizer_config.json}
du -sh /dev/shm/${MODEL_NAME}                      # 应为 ~450 GB
```

> **为什么用 /dev/shm**：内存文件系统读取速度 ~50 GB/s，比 Hyperdisk ML（2.4 GB/s）快 20 倍，模型加载时间从 ~3.5 min 缩短到 ~10 秒。
>
> **为什么用 `gcloud storage cp`**：这是 GCS 下载最快的命令，自动多线程分片传输，比 `gsutil cp` 快 2-3 倍。
>
> **注意**：`/dev/shm` 是 tmpfs，VM 重启后数据会丢失，需要重新从 GCS 拷贝。
>
> **首次上传模型到 GCS**：如果 GCS 桶里还没有模型权重，先在任意机器上从 HuggingFace 下载后上传：
> ```bash
> pip install -U "huggingface_hub[hf_transfer]"
> HF_HUB_ENABLE_HF_TRANSFER=1 huggingface-cli download \
>   Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8 --local-dir /tmp/Qwen3-Coder-480B-A35B-FP8
> gcloud storage cp -r /tmp/Qwen3-Coder-480B-A35B-FP8 ${MODEL_BUCKET}/
> ```

---

## Step 4: 启动 vLLM（约 7-15 min）

> **重要**：必须 `cd /tmp` 后再运行 vLLM，否则 `~/vllm/` 目录会被 Python 当作 namespace package，导致 import 错误。

```bash
source ~/vllm_env/bin/activate
cd /tmp && mkdir -p /tmp/vllm-logs

export MODEL=/dev/shm/Qwen3-Coder-480B-A35B-FP8
export HF_HUB_OFFLINE=1

nohup env \
  SKIP_JAX_PRECOMPILE=1 \
  VLLM_XLA_CHECK_RECOMPILATION=0 \
  MODEL_IMPL_TYPE=vllm \
  HF_HUB_OFFLINE=1 \
  vllm serve $MODEL \
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

# 等待就绪（约 7-15 min）
until curl -s -o /dev/null -w '%{http_code}\n' http://localhost:8000/health | grep -q 200; do
  date; sleep 30
done
echo "✅ Server ready"
```

---

## Step 5: 验证推理

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

## Step 6: Benchmark

### 6.1 安装 benchmark 工具

```bash
cd ~
git clone https://github.com/kimbochen/bench_serving.git
```

### 6.2 1K/1K Benchmark（CI 标准配置）

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

### 6.3 1K/8K Benchmark（长输出）

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

### 6.4 8K/1K Benchmark（长输入）

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

### 6.5 全并发扫描

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

### 6.6 预期性能参考（GKE 实测值）

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
    --create-disk=auto-delete=yes,boot=yes,size=1000GB,type=hyperdisk-balanced,image=projects/ubuntu-os-accelerator-images/global/images/family/ubuntu-accel-2404-amd64-tpu-tpu7x \
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
    --create-disk=auto-delete=yes,boot=yes,size=1000GB,type=hyperdisk-balanced,image=projects/ubuntu-os-accelerator-images/global/images/family/ubuntu-accel-2404-amd64-tpu-tpu7x \
    --disk=name=qwen3-data-decode,device-name=data-disk,mode=rw,auto-delete=no \
    --no-shielded-secure-boot \
    --scopes=cloud-platform
```

> **省盘方案**：如果不需要读写分离，可以用 **1 块 Hyperdisk ML** 设为 `READ_ONLY_MANY` 模式同时挂载到两台 VM（只读），前提是模型权重已提前写好。参见 [TPU-VM README](../TPU-VM/README.md) 的 Hyperdisk ML 读写模式说明。

## Step 2: 两台 VM 分别执行环境准备

在两台 VM 上分别执行：
1. Part 1 Step 2（格式化挂载数据盘）
2. Part 1 Step 3.1 ~ 3.5（系统配置 + 安装 vLLM/tpu-inference + 设置环境变量）
3. 扩容 /dev/shm 并拷贝模型（同 Part 1 Step 3.6）：

```bash
sudo mount -o remount,size=700G /dev/shm
gcloud storage cp -r ${MODEL_BUCKET}/${MODEL_NAME} /dev/shm/
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
  vllm serve /dev/shm/Qwen3-Coder-480B-A35B-FP8 \
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
  vllm serve /dev/shm/Qwen3-Coder-480B-A35B-FP8 \
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

> 2 台 TPU v7x-8 VM 组成 v7x-16 slice（8 chips, 16 devices），通过 ICI 高速互联。
>
> **注意**：multi-host TP=16 通过 ICI Slice 互联，性能比单机 v7x-8 低 15~21%（实测数据见 Step 6）。仍不如单机，主要用于大模型无法单机装下的场景。

## Step 1: 创建 v7x-16 TPU Slice

TPU7x multi-host slice 必须通过 **Workload Policy + Instance Template + MIG** 三件套创建，确保物理 ICI 互联。
单独创建两台 GCE VM 只能走 DCN（数据中心网络），无法获得 ICI 高速互联。

### 1.1 创建 Workload Policy

Workload Policy 告诉 Compute Engine 按指定拓扑分配**物理 ICI 互联**的 TPU chips。

```bash
SLICE_NAME=qwen3-slice

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
    --reservation=projects/${PROJECT_ID}/reservations/${RESERVATION_NAME} \
    --provisioning-model=RESERVATION_BOUND \
    --instance-termination-action=DELETE \
    --maintenance-policy=TERMINATE \
    --scopes=cloud-platform
```

> **注意**：`--instance-termination-action=DELETE` 是 RESERVATION_BOUND + MIG 的必需参数。subnet 必须用完整路径 `projects/.../subnetworks/...`，因为 instance template 是全局资源，不会自动推断 region。

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

> **三个关键参数**（缺一不可，否则只会创建独立 VM 而非 ICI slice）：
>
> | 参数 | 作用 |
> |------|------|
> | `--workload-policy` | 指定 ICI 拓扑，MIG 按此拓扑分配物理互联的 chips |
> | `--default-action-on-vm-failure=do-nothing` | 禁止 MIG 自动修复单个 VM（会破坏 slice 拓扑） |
> | Target Size Policy = BULK | gcloud 在检测到 workload-policy 时自动设置；REST API 须手动指定 `targetSizePolicy.mode: BULK`，确保所有 VM 原子性同时分配 |
>
> **验证 slice 生效**：VM 创建后检查 metadata，成功的 slice 应显示 `TOPOLOGY: 2x2x2`、`HOST_BOUNDS: 1,1,2`、`TPU_ACCELERATOR_TYPE: tpu7x-16`，且 `worker-network-endpoints` 包含两台 VM。如果看到 `TOPOLOGY: 2x2x1`、`HOST_BOUNDS: 1,1,1`，说明 workload policy 没有正确关联。

MIG 会创建 2 台 VM，物理上通过 ICI 互联组成一个 v7x-16 slice（8 chips, 16 devices）。

### 1.4 查看创建的 VM

```bash
gcloud compute instance-groups managed list-instances ${SLICE_NAME}-mig \
    --project=${PROJECT_ID} --zone=${ZONE}
```

记下两台 VM 的名称（后续 SSH 和配置时需要）。

## Step 2: 环境准备

在两台 VM 上分别执行 Part 1 的 Step 3（裸机安装 vLLM + tpu-inference + JAX 0.9.2 fix）。

> **注意**：multi-host **不要**用 `/dev/shm` 存模型。Ray 的 Object Store 默认占用 `/dev/shm` 约 30-40%，会与模型文件冲突。模型改拷到 boot disk `~/models/`。

SSH 到 MIG 中的 VM（直连外网 IP，与 Part 1 Step 1.3 一致）：

```bash
# 获取两台 VM 的外网 IP
gcloud compute instance-groups managed list-instances ${SLICE_NAME}-mig \
    --project=${PROJECT_ID} --zone=${ZONE} --format="table(instance.basename(),instance.status)"

# 查看具体 VM 的 IP
VM_NAME=<vm-name-from-above>
VM_IP=$(gcloud compute instances describe ${VM_NAME} \
    --project=${PROJECT_ID} --zone=${ZONE} \
    --format="value(networkInterfaces[0].accessConfigs[0].natIP)")
ssh -o StrictHostKeyChecking=no -i ~/.ssh/google_compute_engine ${USER}@${VM_IP}
```

### 模型拷贝（两台 VM 都执行）

multi-host 的模型必须存放在 boot disk，不能用 `/dev/shm`：

```bash
mkdir -p ~/models
gcloud storage cp -r ${MODEL_BUCKET}/${MODEL_NAME} ~/models/

# 验证
ls ~/models/${MODEL_NAME}/*.safetensors | wc -l   # 应为 49
du -sh ~/models/${MODEL_NAME}                      # 应为 ~450 GB
```

> **为什么不用 /dev/shm**：Ray Object Store 默认用 `/dev/shm` 的 30%（~280 GB），加上模型 450 GB + vLLM worker 内存，总 RAM 用量超过物理内存导致 OOM。将模型放在 boot disk 上，虽然加载速度稍慢（~3 min vs ~10 s），但 RAM 使用可控。

## Step 3: 获取内网 IP

```bash
# 替换为 Step 1.4 中获取的实际 VM 名称
HOST0_VM=<mig-vm-name-0>
HOST1_VM=<mig-vm-name-1>

HOST0_IP=$(gcloud compute instances describe ${HOST0_VM} \
    --project=${PROJECT_ID} --zone=${ZONE} \
    --format="value(networkInterfaces[0].networkIP)")
HOST1_IP=$(gcloud compute instances describe ${HOST1_VM} \
    --project=${PROJECT_ID} --zone=${ZONE} \
    --format="value(networkInterfaces[0].networkIP)")
echo "Host0: ${HOST0_IP}, Host1: ${HOST1_IP}"
```

> **注意**：MIG 创建的 VM 名称是自动生成的（如 `qwen3-slice-mig-xxxx`），需从 Step 1.4 的输出中获取。选择 `TPU_WORKER_ID=0` 的 VM 作为 Host 0（Ray Head + vLLM API Server）。

## Step 4: 设置 Multi-host TPU 环境变量（两台 VM 都执行）

Multi-host 推理需要设置 TPU 拓扑环境变量。Workload Policy + MIG 创建的 slice 已具备物理 ICI 互联，这里的环境变量让 JAX runtime 识别拓扑。

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

> **关键差异**：
> - `TPU_WORKER_ID=0` vs `TPU_WORKER_ID=1`
> - `VLLM_HOST_IP` 分别设为各自 IP
> - `JAX_PLATFORMS=`（空）而非单机的 `tpu,cpu` — multi-host 必须设为空，让 `PJRT_DEVICE=TPU` 控制设备选择，否则 JAX 无法正确初始化跨节点拓扑

## Step 5: 启动 Ray 集群 + vLLM

### Host 0（先启动 Ray Head）

```bash
# 启动 Ray Head（daemon 模式）
# --object-store-memory 限制 Ray plasma store 为 100 GB，避免占满 /dev/shm
RAY_memory_monitor_refresh_ms=0 ray start --head \
  --port=6379 \
  --node-ip-address=${VLLM_HOST_IP} \
  --resources='{"TPU": 4}' \
  --object-store-memory=107374182400

sleep 20
ray status   # 确认 head 启动，应显示 100.0 GiB object store
```

### Host 1（启动 Ray Worker）

```bash
# 加入 Ray 集群
RAY_memory_monitor_refresh_ms=0 ray start \
  --address=${HOST0_IP}:6379 \
  --node-ip-address=${VLLM_HOST_IP} \
  --resources='{"TPU": 4}' \
  --object-store-memory=107374182400

# 可选：--block 保持前台，或不加 --block 以 daemon 运行
```

### Host 0（确认集群就绪后启动 vLLM）

```bash
# 确认 2 nodes, 8 TPU, 每个 node 100 GB object store
ray status

# 启动 vLLM（TP=16, Ray executor）— 必须 cd /tmp 避免 namespace package 问题
cd /tmp && mkdir -p /tmp/vllm-logs

nohup env \
  PJRT_DEVICE=TPU TPU_BACKEND_TYPE=jax JAX_PLATFORMS= \
  MODEL_IMPL_TYPE=vllm USE_MOE_EP_KERNEL=0 USE_BATCHED_RPA_KERNEL=0 \
  HF_HUB_OFFLINE=1 SKIP_JAX_PRECOMPILE=1 VLLM_XLA_CHECK_RECOMPILATION=0 \
  RAY_memory_monitor_refresh_ms=0 \
  vllm serve ~/models/Qwen3-Coder-480B-A35B-FP8 \
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
    --host 0.0.0.0 --port 8000 \
  > /tmp/vllm-logs/serve.log 2>&1 &

# 等待就绪（约 40 min，含模型加载 + 多轮 XLA 编译）
until curl -s -o /dev/null -w '%{http_code}\n' http://localhost:8000/health | grep -q 200; do
  date; tail -1 /tmp/vllm-logs/serve.log 2>/dev/null; sleep 60
done
echo "✅ Server ready"
```

> **关键参数说明**：
>
> | 参数 | 作用 |
> |------|------|
> | `--object-store-memory=107374182400` | 限制 Ray plasma store 为 100 GB（默认占 /dev/shm 30%~280 GB，会挤占模型和 worker 内存） |
> | `RAY_memory_monitor_refresh_ms=0` | 禁用 Ray OOM monitor（模型加载期间 RAM 使用高峰会触发 worker 被 kill） |
> | `~/models/...` 而非 `/dev/shm/...` | 模型放 boot disk，避免 tmpfs RAM 双重计数导致 OOM |
>
> **注意**：multi-host 不支持 `--async-scheduling`（Ray executor 限制）。启动时间约 40 min（含模型从 boot disk 加载 ~3 min + 多轮 XLA 编译 ~35 min）。

## Step 6: 验证和 Benchmark

在 Host 0 上执行：

### Smoke test

```bash
python3 -c "
import requests, json
r = requests.post('http://localhost:8000/v1/chat/completions', json={
    'model': 'Qwen3-Coder-480B-FP8',
    'messages': [{'role': 'user', 'content': 'What model are you? Reply in one sentence.'}],
    'max_tokens': 100, 'temperature': 0.7
})
print(json.dumps(r.json(), indent=2))
"
```

### Benchmark

> **注意**：TPU VM 裸机没有安装 PyTorch，`vllm bench serve` 命令不可用。使用以下 Python 脚本替代。

```bash
python3 << 'PYEOF'
import requests, time, concurrent.futures

URL = "http://localhost:8000/v1/chat/completions"
MODEL = "Qwen3-Coder-480B-FP8"
BASE = "The quick brown fox jumps over the lazy dog. "  # ~10 tokens/repeat

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

### Multi-host 性能参考（TPU VM ICI Slice 实测值）

| 场景 | Per-req tok/s | Aggregate tok/s | 单机 v7x-8 参考 | vs 单机 |
|------|-------------:|----------------:|---------------:|-------:|
| P1K/D1K c=1 | 37.7 | 37.7 | 48 | -21% |
| P1K/D1K c=4 | 35.8 | 143.2 | 177 | -19% |
| P1K/D1K c=8 | 34.1 | 272.9 | — | — |
| P1K/D1K c=16 | 32.1 | 513.8 | 602 | -15% |
| P8K/D1K c=1 | 36.7 | 36.7 | 46.4 | -21% |
| P8K/D1K c=4 | 34.0 | 136.0 | 162 | -16% |

> **结论**：ICI Slice（Workload Policy + MIG）多机性能比单机低 15-21%。高并发下差距更小（c=16 仅 -15%）。
>
> **启动耗时**：模型加载 ~5 min（boot disk） + XLA 编译 ~29 min（backbone 10 轮 + compute_logits） + 初始化 ~6 min = 总计约 40 min。首次启动无 XLA cache，后续可通过持久化 cache 加速。

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
| 8471 | libtpu coordinator（multi-host ICI） |
| 9000 | vLLM API（Decode） |
| 动态 | TPUConnector KV transfer / ZMQ side-channel |

---

## 资源清理

```bash
# 删除单机 / PD 分离 VM
for vm in qwen3-vm-01 qwen3-prefill qwen3-decode; do
  gcloud compute instances delete $vm \
      --project=${PROJECT_ID} --zone=${ZONE} --quiet 2>/dev/null
done

# 删除数据盘（确认不再需要）
for disk in qwen3-data-01 qwen3-data-prefill qwen3-data-decode; do
  gcloud compute disks delete $disk \
      --project=${PROJECT_ID} --zone=${ZONE} --quiet 2>/dev/null
done

# 删除 multi-host slice（MIG + Template + Workload Policy）
SLICE_NAME=qwen3-slice
gcloud compute instance-groups managed delete ${SLICE_NAME}-mig \
    --project=${PROJECT_ID} --zone=${ZONE} --quiet
gcloud compute instance-templates delete ${SLICE_NAME}-it \
    --project=${PROJECT_ID} --quiet
gcloud compute resource-policies delete ${SLICE_NAME}-wp \
    --project=${PROJECT_ID} --region=${ZONE%-*} --quiet
```

---

## 常见问题

### 1. vLLM 启动 OOM

必须加 `--enable-expert-parallel`，否则 experts 在每个 device 都 replicate。

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

### 7. Multi-host Ray OOM / 模型文件消失

Ray Object Store 默认占 `/dev/shm` 的 30%（944 GB × 30% = ~280 GB）。如果模型也在 `/dev/shm`（450 GB），两者合计超过 `/dev/shm` 容量，导致：
- 模型文件被 Ray plasma store 覆盖"消失"
- 或 Ray worker 加载模型时 RAM 双重计数触发 OOM

**解决方案**：
```bash
# 1. 模型放 boot disk，不要放 /dev/shm
mkdir -p ~/models && gcloud storage cp -r ${MODEL_BUCKET}/${MODEL_NAME} ~/models/

# 2. Ray 启动时限制 object store（100 GB 足够）
ray start --head ... --object-store-memory=107374182400

# 3. 禁用 Ray OOM monitor（加载期间高峰不误杀）
export RAY_memory_monitor_refresh_ms=0
```

### 8. 模型权重缺 tokenizer

```bash
cd /dev/shm/Qwen3-Coder-480B-A35B-FP8/   # 或 ~/models/Qwen3-Coder-480B-A35B-FP8/ (multi-host)
for f in tokenizer.json tokenizer_config.json vocab.json; do
  curl -sL -o $f https://huggingface.co/Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8/resolve/main/$f
done
```
