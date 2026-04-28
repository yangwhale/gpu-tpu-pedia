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

### 3.2 安装 Docker

```bash
sudo apt-get update -qq
sudo apt-get install -y -qq docker.io
sudo usermod -aG docker $USER
newgrp docker
```

### 3.3 拉取 vLLM TPU 镜像

```bash
docker pull vllm/vllm-tpu:nightly
```

### 3.4 扩容 /dev/shm 并从 GCS 拷贝模型

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

### 3.5 启动容器

```bash
docker run -d --name vllm \
    --privileged --net=host --ipc=host \
    -v /mnt/data:/data \
    -v /dev:/dev \
    -e HF_TOKEN=${HF_TOKEN} \
    -e JAX_PLATFORMS=tpu,cpu \
    -e TPU_BACKEND_TYPE=jax \
    -e PJRT_DEVICE=TPU \
    -e MODEL_IMPL_TYPE=vllm \
    -e USE_MOE_EP_KERNEL=0 \
    -e USE_BATCHED_RPA_KERNEL=0 \
    vllm/vllm-tpu:nightly sleep infinity
```

> **`--ipc=host`**：让容器共享宿主机的 `/dev/shm`，容器内可以直接访问 `/dev/shm/${MODEL_NAME}`。

进入容器：

```bash
docker exec -it vllm bash
```

---

## Step 4: 验证模型权重

模型已在 Step 3.4 从 GCS 拷贝到 `/dev/shm`。在容器内验证：

```bash
ls /dev/shm/${MODEL_NAME}/*.safetensors | wc -l   # 应为 49
ls /dev/shm/${MODEL_NAME}/{tokenizer.json,vocab.json,tokenizer_config.json}
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

```bash
cd /tmp && mkdir -p /tmp/vllm-logs

export MODEL=/dev/shm/qwen3-coder-480b-fp8
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
cd /workspace
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

在两台 VM 上分别执行 Part 1 的 Step 2（挂载盘）+ Step 3（系统配置 + Docker + GCS 拷贝模型到 SHM）。

```bash
# 在每台 VM 上执行（SSH 进去后）
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

SSH 到 Prefill VM，进入容器：

```bash
docker exec -it vllm bash
```

启动 vLLM（kv_producer 模式）：

```bash
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

SSH 到 Decode VM，进入容器：

```bash
docker exec -it vllm bash
```

启动 vLLM（kv_consumer 模式）：

```bash
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

在 Prefill VM 的容器内启动 proxy，连接两个实例：

```bash
# 设置 Decode VM 内网 IP（Step 3 获取的 networkIP）
export DECODE_IP=<decode-vm-internal-ip>

python3 /workspace/tpu_inference/examples/disagg/toy_proxy_server.py \
  --host 0.0.0.0 --port 7000 \
  --prefiller-hosts localhost --prefiller-ports 8000 \
  --decoder-hosts ${DECODE_IP} --decoder-ports 9000
```

> **注意**：`${DECODE_IP}` 使用 VPC 内网 IP（Step 3 获取的 `networkIP`），不是外网 IP。确保两台 VM 在同一 VPC 且防火墙允许端口 8000/9000/7000。

## Step 7: PD 分离 Benchmark

在 Prefill VM 容器内对 proxy 端口发请求：

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

在两台 VM 上分别执行 Part 1 的 Step 2 + Step 3（挂载盘、系统配置、Docker、GCS 拷贝模型到 SHM）。

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

## Step 4: 启动容器（两台 VM 都执行）

两台 VM 的 Docker 启动命令需要额外设置 multi-host TPU 环境变量：

### Host 0（Ray Head + vLLM API Server）

```bash
docker run -d --name vllm \
    --privileged --net=host --ipc=host \
    -v /mnt/data:/data \
    -v /dev:/dev \
    -e HF_TOKEN=${HF_TOKEN} \
    -e PJRT_DEVICE=TPU \
    -e TPU_BACKEND_TYPE=jax \
    -e JAX_PLATFORMS= \
    -e MODEL_IMPL_TYPE=vllm \
    -e USE_MOE_EP_KERNEL=0 \
    -e USE_BATCHED_RPA_KERNEL=0 \
    -e HF_HUB_OFFLINE=1 \
    -e SKIP_JAX_PRECOMPILE=1 \
    -e VLLM_XLA_CHECK_RECOMPILATION=0 \
    -e TPU_WORKER_HOSTNAMES="${HOST0_IP},${HOST1_IP}" \
    -e TPU_WORKER_ID=0 \
    -e TPU_PROCESS_ADDRESSES="${HOST0_IP}:8471,${HOST1_IP}:8471" \
    -e TPU_PROCESS_PORT=8471 \
    -e TPU_HOST_BOUNDS="1,1,2" \
    -e TPU_CHIPS_PER_HOST_BOUNDS="2,2,1" \
    -e TPU_TOPOLOGY="2x2x2" \
    -e TPU_ACCELERATOR_TYPE="tpu7x-16" \
    -e TPU_SKIP_MDS_QUERY=true \
    -e TPU_MULTIHOST_BACKEND=ray \
    -e VLLM_HOST_IP=${HOST0_IP} \
    vllm/vllm-tpu:nightly sleep infinity
```

### Host 1（Ray Worker）

```bash
docker run -d --name vllm \
    --privileged --net=host --ipc=host \
    -v /mnt/data:/data \
    -v /dev:/dev \
    -e HF_TOKEN=${HF_TOKEN} \
    -e PJRT_DEVICE=TPU \
    -e TPU_BACKEND_TYPE=jax \
    -e JAX_PLATFORMS= \
    -e MODEL_IMPL_TYPE=vllm \
    -e USE_MOE_EP_KERNEL=0 \
    -e USE_BATCHED_RPA_KERNEL=0 \
    -e HF_HUB_OFFLINE=1 \
    -e SKIP_JAX_PRECOMPILE=1 \
    -e VLLM_XLA_CHECK_RECOMPILATION=0 \
    -e TPU_WORKER_HOSTNAMES="${HOST0_IP},${HOST1_IP}" \
    -e TPU_WORKER_ID=1 \
    -e TPU_PROCESS_ADDRESSES="${HOST0_IP}:8471,${HOST1_IP}:8471" \
    -e TPU_PROCESS_PORT=8471 \
    -e TPU_HOST_BOUNDS="1,1,2" \
    -e TPU_CHIPS_PER_HOST_BOUNDS="2,2,1" \
    -e TPU_TOPOLOGY="2x2x2" \
    -e TPU_ACCELERATOR_TYPE="tpu7x-16" \
    -e TPU_SKIP_MDS_QUERY=true \
    -e TPU_MULTIHOST_BACKEND=ray \
    -e VLLM_HOST_IP=${HOST1_IP} \
    vllm/vllm-tpu:nightly sleep infinity
```

> **关键差异**：`TPU_WORKER_ID=0` vs `TPU_WORKER_ID=1`，`VLLM_HOST_IP` 分别设为各自 IP。

## Step 5: 启动 Ray 集群 + vLLM

### Host 0（先启动 Ray Head，再启动 vLLM）

```bash
docker exec -it vllm bash

# 启动 Ray Head（daemon 模式，不加 --block）
ray start --head --port=6379 --node-ip-address=${VLLM_HOST_IP} --resources='{"TPU": 4}'
sleep 20
ray status   # 确认 head 启动

# 等 Host 1 的 Ray Worker 加入后再启动 vLLM
# 检查 ray status 显示 2 个 node, 总共 8 TPU
```

### Host 1（启动 Ray Worker）

```bash
docker exec -it vllm bash

# 加入 Ray 集群（--block 保持前台）
ray start --address=${HOST0_IP}:6379 --node-ip-address=${VLLM_HOST_IP} --resources='{"TPU": 4}' --block
```

### Host 0（确认集群就绪后启动 vLLM）

```bash
# 确认 2 nodes, 8 TPU
ray status

# 启动 vLLM（TP=16, Ray executor）
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

在 Host 0 容器内执行：

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

必须加 `--enable-expert-parallel`，否则 experts 在每个 device 都 replicate。

### 2. Docker 看不到 TPU

确保 `docker run` 使用 `--privileged` 且挂载了 `/dev`。验证：

```bash
docker exec vllm ls /dev/vfio/
```

### 3. libtpu lockfile 报错

```bash
docker exec vllm bash -c 'pkill -9 -f "vllm|EngineCore"; sleep 3; rm -f /tmp/libtpu_lockfile /tmp/.vllm_ipc_*'
```

### 4. PD 分离 Decode 连不上 Prefill

检查 VPC 内网 IP 和防火墙规则，确保端口 8000/9000 可达：

```bash
# 在 Decode VM 上测试
curl -s http://${PREFILL_IP}:8000/health
```

### 5. Multi-host AttributeError: d.coords

TPU 环境变量没设对。确认 `TPU_WORKER_HOSTNAMES` 包含两台 VM 的 IP，且 `TPU_WORKER_ID` 在两台分别为 0 和 1。

### 6. 模型权重缺 tokenizer

```bash
cd /dev/shm/qwen3-coder-480b-fp8/
for f in tokenizer.json tokenizer_config.json vocab.json; do
  curl -sL -o $f https://huggingface.co/Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8/resolve/main/$f
done
```
