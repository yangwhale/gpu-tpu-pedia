**English** | [中文](./README-TPU-VM.md)

# Qwen3-Coder-480B FP8 Inference on TPU v7x — TPU VM Edition

> End-to-end TPU VM deployment guide: from creating the VM to completing the benchmark.
>
> **Model**: [Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8](https://huggingface.co/Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8) (~480 GB)
>
> **Code repository**: [vllm-project/tpu-inference](https://github.com/vllm-project/tpu-inference) (main branch)
>
> **Model storage**: `gs://aidc-tpu-data/models` (GCS object storage; model weights are stored here uniformly)
>
> For the GKE edition, see [README.md](README.md) in the same directory.

## Table of Contents

- [Part 1: Single-host Inference](#part-1-单机推理)
- [Part 2: PD Disaggregation (1P1D)](#part-2-pd-分离-1p1d)
- [Part 3: Multi-node Inference (TP=16)](#part-3-多节点推理-tp16)

---

## Environment Variables

```bash
export PROJECT_ID=<your-project-id>
export ZONE=us-central1-c
export RESERVATION_NAME=<your-reservation>
export VPC_NAME=<your-vpc-name>
export SUBNET_NAME=<your-subnet-name>
export HF_TOKEN=<your-hf-token>
export MODEL_BUCKET=gs://aidc-tpu-data/models    # Model weights GCS path
export MODEL_NAME=Qwen3-Coder-480B-A35B-FP8      # Model directory name (matches GCS)
```

## Hardware Requirements

| Item | Single-host (Part 1 & 2) | Multi-node (Part 3) |
|------|-------------------|----------------|
| Machine type | `tpu7x-standard-4t` | `tpu7x-standard-4t` × 2 |
| TPU | v7x-8 (4 chips, 8 devices) | v7x-16 (8 chips, 16 devices) |
| HBM | 768 GB | 1,536 GB |
| Host memory | 944 GB | 944 GB × 2 |
| Boot disk | 1 TB Hyperdisk Balanced | 1 TB × 2 |
| Data disk | ≥600 GB (model ~480 GB) | Not needed (model copied to boot disk ~/models/) |

---

# Part 1: Single-host Inference

## Step 1: Create the Data Disk and VM

### 1.1 Create the Hyperdisk ML Data Disk

```bash
gcloud compute disks create qwen3-data-01 \
    --project=${PROJECT_ID} \
    --zone=${ZONE} \
    --type=hyperdisk-ml \
    --size=2TB \
    --provisioned-throughput=2500
```

### 1.2 Create the TPU VM

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

### 1.3 SSH Connection

```bash
VM_IP=$(gcloud compute instances describe qwen3-vm-01 \
    --project=${PROJECT_ID} --zone=${ZONE} \
    --format="value(networkInterfaces[0].accessConfigs[0].natIP)")
ssh -o StrictHostKeyChecking=no -i ~/.ssh/google_compute_engine ${USER}@${VM_IP}
```

---

## Step 2: Format and Mount the Data Disk

```bash
sudo mkfs.ext4 -m 0 -E lazy_itable_init=0,lazy_journal_init=0,discard /dev/nvme1n1
sudo mkdir -p /mnt/data
sudo mount -o discard,defaults /dev/nvme1n1 /mnt/data
sudo chmod a+w /mnt/data
echo "/dev/disk/by-id/google-data-disk /mnt/data ext4 discard,defaults,nofail 0 2" | sudo tee -a /etc/fstab
```

---

## Step 3: Environment Preparation

### 3.1 System Configuration

```bash
sudo sysctl -w vm.max_map_count=8388608
echo 'vm.max_map_count=8388608' | sudo tee -a /etc/sysctl.conf

if [ -f /sys/module/vfio_iommu_type1/parameters/dma_entry_limit ]; then
  echo 2000000 | sudo tee /sys/module/vfio_iommu_type1/parameters/dma_entry_limit
fi
```

### 3.2 Install System Dependencies

```bash
sudo apt-get update -qq
sudo apt-get install -y -qq libopenmpi-dev libomp-dev git curl
```

### 3.3 Install vLLM + tpu-inference (Bare Metal)

Use `uv` to create a Python 3.12 virtual environment and install vLLM (TPU edition) and tpu-inference:

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env

# Create a Python 3.12 virtual environment
uv venv ~/vllm_env --python 3.12
source ~/vllm_env/bin/activate

# Clone tpu-inference (to get the pinned vLLM version)
cd ~
git clone https://github.com/vllm-project/tpu-inference.git
cd tpu-inference

# Get the pinned vLLM commit hash
export VLLM_COMMIT_HASH="$(cat .buildkite/vllm_lkg.version | tr -d '[:space:]')"
echo "vLLM commit: ${VLLM_COMMIT_HASH}"

# Clone vLLM and check out the pinned version
cd ~
git clone https://github.com/vllm-project/vllm.git
cd vllm
git checkout "${VLLM_COMMIT_HASH}"

# Install vLLM (TPU target)
uv pip install -r requirements/tpu.txt --torch-backend=cpu
VLLM_TARGET_DEVICE="tpu" uv pip install -e .

# Fix the JAX version (the vLLM install downgrades JAX to 0.8.0; you must reinstall 0.9.2)
uv pip install jax==0.9.2 jaxlib==0.9.2 libtpu==0.0.39 flax==0.12.4

# Install tpu-inference (--no-deps to avoid downgrading JAX again)
cd ~/tpu-inference
uv pip install -e . --no-deps
```

### 3.4 Verify the Installation

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

Expected output:
```
vllm: 0.9.x
tpu_inference: 0.1.0
jax: 0.9.2
platform: TpuDevice
devices: 8 x tpu
```

> **About the JAX version**: vLLM's `requirements/tpu.txt` installs JAX 0.8.0, but TPU v7x requires JAX 0.9.2 + libtpu 0.0.39. After installing vLLM, you must manually override and install the correct versions.
>
> **About `--no-deps`**: tpu-inference's `pyproject.toml` depends on jax/jaxlib; without `--no-deps`, the JAX downgrade is triggered again.

### 3.5 Set Runtime Environment Variables

```bash
# Recommended to add to ~/.bashrc, or source before each startup
export HF_TOKEN=${HF_TOKEN}
export JAX_PLATFORMS=tpu,cpu
export TPU_BACKEND_TYPE=jax
export PJRT_DEVICE=TPU
export MODEL_IMPL_TYPE=vllm
export USE_MOE_EP_KERNEL=0
export USE_BATCHED_RPA_KERNEL=0
```

### 3.6 Expand /dev/shm and Copy the Model from GCS

Model weights are stored uniformly in GCS. After each VM startup, copy them to `/dev/shm` (an in-memory filesystem) for the fastest loading speed.

```bash
# Expand /dev/shm (default ~472 GB; needs to hold the 480 GB model + vLLM IPC)
sudo mount -o remount,size=700G /dev/shm

# Copy the model weights from GCS to /dev/shm (must use gcloud storage cp; fastest)
gcloud storage cp -r ${MODEL_BUCKET}/${MODEL_NAME} /dev/shm/

# Verify
ls /dev/shm/${MODEL_NAME}/*.safetensors | wc -l   # should be 49
ls /dev/shm/${MODEL_NAME}/{tokenizer.json,vocab.json,tokenizer_config.json}
du -sh /dev/shm/${MODEL_NAME}                      # should be ~450 GB
```

> **Why /dev/shm**: An in-memory filesystem reads at ~50 GB/s, 20x faster than Hyperdisk ML (2.4 GB/s), shortening model load time from ~3.5 min to ~10 seconds.
>
> **Why `gcloud storage cp`**: This is the fastest command for GCS downloads, automatically doing multi-threaded sharded transfers — 2-3x faster than `gsutil cp`.
>
> **Note**: `/dev/shm` is tmpfs; data is lost after a VM restart, so you must re-copy it from GCS.
>
> **First-time upload of the model to GCS**: If the model weights are not yet in the GCS bucket, first download them from HuggingFace on any machine and upload:
> ```bash
> pip install -U "huggingface_hub[hf_transfer]"
> HF_HUB_ENABLE_HF_TRANSFER=1 huggingface-cli download \
>   Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8 --local-dir /tmp/Qwen3-Coder-480B-A35B-FP8
> gcloud storage cp -r /tmp/Qwen3-Coder-480B-A35B-FP8 ${MODEL_BUCKET}/
> ```

---

## Step 4: Start vLLM (about 7-15 min)

> **Important**: You must `cd /tmp` before running vLLM; otherwise the `~/vllm/` directory will be treated by Python as a namespace package, causing import errors.

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

# Wait until ready (about 7-15 min)
until curl -s -o /dev/null -w '%{http_code}\n' http://localhost:8000/health | grep -q 200; do
  date; sleep 30
done
echo "✅ Server ready"
```

---

## Step 5: Verify Inference

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

### 6.1 Install the Benchmark Tool

```bash
cd ~
git clone https://github.com/kimbochen/bench_serving.git
```

### 6.2 1K/1K Benchmark (CI Standard Configuration)

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

### 6.3 1K/8K Benchmark (Long Output)

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

### 6.4 8K/1K Benchmark (Long Input)

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

### 6.5 Full Concurrency Sweep

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

### 6.6 Expected Performance Reference (Measured on GKE)

| Workload | c=1 | c=4 | c=16 | c=64 |
|----------|----:|----:|-----:|-----:|
| 1K/1K | 48 tok/s | 177 | 602 | 1478 |
| 1K/8K | 47.5 | 178 | 621 | 1623 |
| 8K/1K | 46.4 | 162 | 483 | 943 |

> Performance on a TPU VM should match a GKE Pod (same hardware, same software stack).

---

# Part 2: PD Disaggregation (1P1D)

> 2 TPU v7x-8 VMs: one runs Prefill (kv_producer), the other runs Decode (kv_consumer), transferring the KV cache over the VPC internal network.
>
> **Architecture note**: PD disaggregation **does not require Ray**. The two vLLM instances run completely independently, transferring the KV cache directly P2P via the TPUConnector (JAX transfer server + ZMQ side-channel), with `toy_proxy_server.py` handling request routing.

## Step 1: Create 2 VMs

Reusing the approach from Part 1, create the Prefill and Decode VMs:

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

> **Disk-saving option**: If you do not need read/write separation, you can use **a single Hyperdisk ML** set to `READ_ONLY_MANY` mode and mount it on both VMs simultaneously (read-only), provided the model weights have been written in advance. See the Hyperdisk ML read/write mode notes in the [TPU-VM README](../TPU-VM/README.md).

> **Fallback when there is no Hyperdisk ML quota**: If Hyperdisk ML quota is insufficient, you can skip creating a data disk and store the model weights on the boot disk instead. Simply remove the `gcloud compute disks create` and `--disk=name=...` lines. Copy the model directly onto the boot disk (e.g., `/home/${USER}/`).

## Step 2: Run Environment Preparation on Each of the Two VMs

On each of the two VMs, run:
1. Part 1 Step 2 (format and mount the data disk) — skip if there is no data disk
2. Part 1 Steps 3.1 ~ 3.5 (system configuration + install vLLM/tpu-inference + set environment variables)
3. Copy the model (same as Part 1 Step 3.6); the model storage location depends on your disk configuration:

```bash
# Option A (machine RAM ≥ 1.5 TiB + space available in /dev/shm):
sudo mount -o remount,size=700G /dev/shm
gcloud storage cp -r ${MODEL_BUCKET}/${MODEL_NAME} /dev/shm/
export MODEL_DIR=/dev/shm/${MODEL_NAME}

# Option B (Hyperdisk ML data disk mounted at /mnt/data):
# gcloud storage cp -r ${MODEL_BUCKET}/${MODEL_NAME} /mnt/data/
# export MODEL_DIR=/mnt/data/${MODEL_NAME}

# Option C (no data disk, use the boot disk):
# mkdir -p /home/${USER}/${MODEL_NAME}
# gcloud storage cp -r ${MODEL_BUCKET}/${MODEL_NAME}/* /home/${USER}/${MODEL_NAME}/
# export MODEL_DIR=/home/${USER}/${MODEL_NAME}
```

> ⚠️ **The same RAM-size judgment from Part 1 Step 3.6 applies**: the default `tpu7x-standard-4t` 944 GiB will OOM if you use /dev/shm; use Option B or C instead.
>
> Regardless of which option you choose, subsequent steps uniformly reference the model path via `${MODEL_DIR}`.

## Step 3: Get the Internal IPs

```bash
PREFILL_IP=$(gcloud compute instances describe qwen3-prefill \
    --project=${PROJECT_ID} --zone=${ZONE} \
    --format="value(networkInterfaces[0].networkIP)")
DECODE_IP=$(gcloud compute instances describe qwen3-decode \
    --project=${PROJECT_ID} --zone=${ZONE} \
    --format="value(networkInterfaces[0].networkIP)")
echo "Prefill: ${PREFILL_IP}, Decode: ${DECODE_IP}"
```

## Step 4: Start the Prefill Instance

SSH to the Prefill VM:

```bash
source ~/vllm_env/bin/activate
cd /tmp && mkdir -p /tmp/vllm-logs

nohup env \
  PJRT_DEVICE=TPU TPU_BACKEND_TYPE=jax JAX_PLATFORMS= \
  SKIP_JAX_PRECOMPILE=1 VLLM_XLA_CHECK_RECOMPILATION=0 \
  MODEL_IMPL_TYPE=vllm HF_HUB_OFFLINE=1 \
  vllm serve ${MODEL_DIR} \
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
  > /tmp/vllm-logs/prefill.log 2>&1 &
```

Key differences (vs Part 1 single-host): `gpu-memory-utilization=0.70` (reserves 30% of HBM for the KV transfer buffer), `kv_role=kv_producer`.

> Startup takes 8~12 minutes (model loading + MoE requantization + XLA compilation). Use `tail -f /tmp/vllm-logs/prefill.log` to watch progress; it is ready when you see `Application startup complete`.

## Step 5: Start the Decode Instance

SSH to the Decode VM:

```bash
source ~/vllm_env/bin/activate
cd /tmp && mkdir -p /tmp/vllm-logs

nohup env \
  PJRT_DEVICE=TPU TPU_BACKEND_TYPE=jax JAX_PLATFORMS= \
  SKIP_JAX_PRECOMPILE=1 VLLM_XLA_CHECK_RECOMPILATION=0 \
  MODEL_IMPL_TYPE=vllm HF_HUB_OFFLINE=1 \
  vllm serve ${MODEL_DIR} \
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
  > /tmp/vllm-logs/decode.log 2>&1 &
```

Key differences (vs Prefill): `gpu-memory-utilization=0.90` (Decode does not need to reserve a transfer buffer), `kv_role=kv_consumer`, `port=9000`.

> You can start it at the same time as Prefill; the two load the model independently. Likewise, use `tail -f /tmp/vllm-logs/decode.log` to watch.

## Step 6: Verify Both Ends Are Ready + Start the Proxy

Run all of the following operations on the Prefill VM:

```bash
source ~/vllm_env/bin/activate

# Set the Decode VM internal IP (the networkIP obtained in Step 3)
export DECODE_IP=<decode-vm-internal-ip>
```

Confirm that both instances are ready:

```bash
curl -s http://localhost:8000/v1/models | python3 -m json.tool
curl -s http://${DECODE_IP}:9000/v1/models | python3 -m json.tool
```

Once both return model information, start the proxy:

```bash
python3 ~/tpu-inference/examples/disagg/toy_proxy_server.py \
  --host 0.0.0.0 --port 7000 \
  --prefiller-hosts localhost --prefiller-ports 8000 \
  --decoder-hosts ${DECODE_IP} --decoder-ports 9000
```

> **Path note**: The proxy script is located at `tpu-inference/examples/disagg/` (not `tpu_inference/examples/`).
>
> **Note**: `${DECODE_IP}` uses the VPC internal IP (the `networkIP` obtained in Step 3), not the external IP. Make sure the two VMs are in the same VPC and the firewall allows ports 8000/9000/7000.

After the proxy starts, first send a smoke test to verify the full pipeline:

```bash
curl http://localhost:7000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"Qwen3-Coder-480B-FP8","messages":[{"role":"user","content":"What is 2+2?"}],"max_tokens":32}'
```

> ⚠️ **First-request latency**: The first request triggers XLA compilation, taking about 1~2 minutes each for Prefill and Decode. This is normal; subsequent requests hit the compilation cache and latency drops to the second range.

## Step 7: PD Disaggregation Benchmark

> **Note**: The bare-metal TPU VM does not have PyTorch installed, so the `vllm bench serve` command is unavailable. Use the following Python script instead (sending requests to proxy port 7000).

```bash
python3 << 'PYEOF'
import requests, time, concurrent.futures

URL = "http://localhost:7000/v1/chat/completions"
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
bench(8192, 1024, 4, 8)
bench(1024, 8192, 64, 256)
PYEOF
```

### PD Disaggregation Expected Performance Reference (Measured on GKE)

| Configuration | TTFT (med) | TPOT (med) | Output tok/s | vs Single Instance |
|------|----------:|----------:|------------:|----------|
| Single instance 1K/1K c=1 | 95 ms | 20.6 ms | 48 | baseline |
| 1P1D 1K/1K c=1 | 281 ms | 18.3 ms | 53.8 | TPOT -11% |
| Single instance 8K/1K c=4 | 1495 ms | 23.2 ms | 162 | baseline |
| 1P1D 8K/1K c=4 | 2908 ms | 20.6 ms | 170 | TPOT -11% |

---

# Part 3: Multi-node Inference (TP=16)

> 2 TPU v7x-8 VMs form a v7x-16 slice (8 chips, 16 devices), interconnected via high-speed ICI.
>
> **Note**: multi-host TP=16 is interconnected over an ICI Slice; performance is 15~21% lower than a single-host v7x-8 (see Step 6 for measured data). It is still slower than a single host and is mainly used in scenarios where a large model cannot fit on a single host.

## Step 1: Create the v7x-16 TPU Slice

A TPU7x multi-host slice must be created via the **Workload Policy + Instance Template + MIG** trio to ensure physical ICI interconnect.
Creating two GCE VMs separately can only use DCN (data center network) and cannot achieve high-speed ICI interconnect.

### 1.1 Create the Workload Policy

The Workload Policy tells Compute Engine to allocate TPU chips with **physical ICI interconnect** according to the specified topology.

```bash
SLICE_NAME=qwen3-slice

gcloud compute resource-policies create workload-policy ${SLICE_NAME}-wp \
    --type=HIGH_THROUGHPUT \
    --accelerator-topology=2x2x2 \
    --project=${PROJECT_ID} \
    --region=${ZONE%-*}
```

### 1.2 Create the Instance Template

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

> **Note**: `--instance-termination-action=DELETE` is a required parameter for RESERVATION_BOUND + MIG. The subnet must use the full path `projects/.../subnetworks/...`, because an instance template is a global resource and will not automatically infer the region.

### 1.3 Create the MIG (TPU Slice)

```bash
gcloud compute instance-groups managed create ${SLICE_NAME}-mig \
    --project=${PROJECT_ID} \
    --zone=${ZONE} \
    --template=${SLICE_NAME}-it \
    --size=2 \
    --default-action-on-vm-failure=do-nothing \
    --workload-policy=projects/${PROJECT_ID}/regions/${ZONE%-*}/resourcePolicies/${SLICE_NAME}-wp
```

> **Two key parameters** (both are required; otherwise it will create independent VMs rather than an ICI slice):
>
> | Parameter | Purpose |
> |------|------|
> | `--workload-policy` | Specifies the ICI topology; the MIG allocates physically interconnected chips according to it |
> | `--default-action-on-vm-failure=do-nothing` | Prevents the MIG from auto-repairing a single VM (which would break the slice topology) |
>
> **Note**: When gcloud detects `--workload-policy`, it automatically sets `targetSizePolicy.mode: BULK` (ensuring all VMs are allocated atomically at the same time). If you use the REST API instead of the gcloud CLI, you must specify this field manually.
>
> **Verify the slice is in effect**: After the VMs are created, check the metadata. A successful slice should show `TOPOLOGY: 2x2x2`, `HOST_BOUNDS: 1,1,2`, `TPU_ACCELERATOR_TYPE: tpu7x-16`, and `worker-network-endpoints` containing both VMs. If you see `TOPOLOGY: 2x2x1`, `HOST_BOUNDS: 1,1,1`, it means the workload policy was not correctly associated.

The MIG creates 2 VMs, physically interconnected via ICI to form a single v7x-16 slice (8 chips, 16 devices).

### 1.4 View the Created VMs

```bash
gcloud compute instance-groups managed list-instances ${SLICE_NAME}-mig \
    --project=${PROJECT_ID} --zone=${ZONE}
```

Note down the names of the two VMs (needed for subsequent SSH and configuration).

## Step 2: Environment Preparation

On each of the two VMs, run Part 1's Step 3 (bare-metal install of vLLM + tpu-inference + JAX 0.9.2 fix).

> **Note**: For multi-host, **do not** use `/dev/shm` to store the model. Ray's Object Store by default occupies about 30-40% of `/dev/shm`, which would conflict with the model files. Copy the model to the boot disk `~/models/` instead.

SSH to a VM in the MIG (directly via external IP, the same as Part 1 Step 1.3):

```bash
# Get the external IPs of the two VMs
gcloud compute instance-groups managed list-instances ${SLICE_NAME}-mig \
    --project=${PROJECT_ID} --zone=${ZONE} --format="table(instance.basename(),instance.status)"

# View the IP of a specific VM
VM_NAME=<vm-name-from-above>
VM_IP=$(gcloud compute instances describe ${VM_NAME} \
    --project=${PROJECT_ID} --zone=${ZONE} \
    --format="value(networkInterfaces[0].accessConfigs[0].natIP)")
ssh -o StrictHostKeyChecking=no -i ~/.ssh/google_compute_engine ${USER}@${VM_IP}
```

### Copy the Model (run on both VMs)

The multi-host model must be stored on the boot disk and cannot use `/dev/shm`:

```bash
mkdir -p ~/models
gcloud storage cp -r ${MODEL_BUCKET}/${MODEL_NAME} ~/models/

# Verify
ls ~/models/${MODEL_NAME}/*.safetensors | wc -l   # should be 49
du -sh ~/models/${MODEL_NAME}                      # should be ~450 GB
```

> **Why not /dev/shm**: Ray Object Store by default uses 30% of `/dev/shm` (~280 GB); combined with the 450 GB model + vLLM worker memory, total RAM usage exceeds physical memory and causes OOM. Putting the model on the boot disk is slightly slower to load (~3 min vs ~10 s), but keeps RAM usage controllable.

## Step 3: Get the Internal IPs

```bash
# Replace with the actual VM names obtained in Step 1.4
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

> **Note**: The MIG-created VM names are auto-generated (e.g., `qwen3-slice-mig-xxxx`) and must be obtained from the Step 1.4 output. Choose the VM with `TPU_WORKER_ID=0` as Host 0 (Ray Head + vLLM API Server).

## Step 4: Set the Multi-host TPU Environment Variables (run on both VMs)

Multi-host inference requires setting the TPU topology environment variables. The slice created by Workload Policy + MIG already has physical ICI interconnect; these environment variables let the JAX runtime recognize the topology.

### Host 0 (Ray Head + vLLM API Server)

```bash
source ~/vllm_env/bin/activate

# Base environment variables (same as Part 1 Step 3.5)
export PJRT_DEVICE=TPU
export TPU_BACKEND_TYPE=jax
export JAX_PLATFORMS=
export MODEL_IMPL_TYPE=vllm
export USE_MOE_EP_KERNEL=0
export USE_BATCHED_RPA_KERNEL=0
export HF_HUB_OFFLINE=1
export SKIP_JAX_PRECOMPILE=1
export VLLM_XLA_CHECK_RECOMPILATION=0

# Multi-host TPU topology variables
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

### Host 1 (Ray Worker)

```bash
source ~/vllm_env/bin/activate

# Base environment variables (same as above)
export PJRT_DEVICE=TPU
export TPU_BACKEND_TYPE=jax
export JAX_PLATFORMS=
export MODEL_IMPL_TYPE=vllm
export USE_MOE_EP_KERNEL=0
export USE_BATCHED_RPA_KERNEL=0
export HF_HUB_OFFLINE=1
export SKIP_JAX_PRECOMPILE=1
export VLLM_XLA_CHECK_RECOMPILATION=0

# Multi-host TPU topology variables
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

> **Key differences**:
> - `TPU_WORKER_ID=0` vs `TPU_WORKER_ID=1`
> - `VLLM_HOST_IP` set to each host's own IP
> - `JAX_PLATFORMS=` (empty) rather than the single-host `tpu,cpu` — multi-host must set it empty so that `PJRT_DEVICE=TPU` controls device selection; otherwise JAX cannot correctly initialize the cross-node topology

## Step 5: Start the Ray Cluster + vLLM

### Host 0 (start the Ray Head first)

```bash
# Start the Ray Head (daemon mode)
# --object-store-memory limits the Ray plasma store to 100 GB to avoid filling up /dev/shm
RAY_memory_monitor_refresh_ms=0 ray start --head \
  --port=6379 \
  --node-ip-address=${VLLM_HOST_IP} \
  --resources='{"TPU": 4}' \
  --object-store-memory=107374182400

sleep 20
ray status   # confirm the head started; should show a 100.0 GiB object store
```

### Host 1 (start the Ray Worker)

```bash
# Join the Ray cluster
RAY_memory_monitor_refresh_ms=0 ray start \
  --address=${HOST0_IP}:6379 \
  --node-ip-address=${VLLM_HOST_IP} \
  --resources='{"TPU": 4}' \
  --object-store-memory=107374182400

# Optional: --block to stay in the foreground, or omit --block to run as a daemon
```

### Host 0 (start vLLM after confirming the cluster is ready)

```bash
# Confirm 2 nodes, 8 TPU, and a 100 GB object store per node
ray status

# Start vLLM (TP=16, Ray executor) — must cd /tmp to avoid the namespace package issue
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

# Wait until ready (about 40 min, including model loading + multiple rounds of XLA compilation)
until curl -s -o /dev/null -w '%{http_code}\n' http://localhost:8000/health | grep -q 200; do
  date; tail -1 /tmp/vllm-logs/serve.log 2>/dev/null; sleep 60
done
echo "✅ Server ready"
```

> **Key parameter notes**:
>
> | Parameter | Purpose |
> |------|------|
> | `--object-store-memory=107374182400` | Limits the Ray plasma store to 100 GB (default occupies 30% of /dev/shm ~280 GB, which would crowd out the model and worker memory) |
> | `RAY_memory_monitor_refresh_ms=0` | Disables the Ray OOM monitor (the RAM-usage peak during model loading would trigger workers being killed) |
> | `~/models/...` instead of `/dev/shm/...` | Put the model on the boot disk to avoid OOM from tmpfs RAM double-counting |
>
> **Note**: Multi-host does not support `--async-scheduling` (a Ray executor limitation). Startup takes about 40 min (including ~3 min to load the model from the boot disk + ~35 min of multiple rounds of XLA compilation).

## Step 6: Verify and Benchmark

Run on Host 0:

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

> **Note**: The bare-metal TPU VM does not have PyTorch installed, so the `vllm bench serve` command is unavailable. Use the following Python script instead.

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

### Multi-host Performance Reference (Measured on TPU VM ICI Slice)

| Scenario | Per-req tok/s | Aggregate tok/s | Single-host v7x-8 reference | vs Single-host |
|------|-------------:|----------------:|---------------:|-------:|
| P1K/D1K c=1 | 37.7 | 37.7 | 48 | -21% |
| P1K/D1K c=4 | 35.8 | 143.2 | 177 | -19% |
| P1K/D1K c=8 | 34.1 | 272.9 | — | — |
| P1K/D1K c=16 | 32.1 | 513.8 | 602 | -15% |
| P8K/D1K c=1 | 36.7 | 36.7 | 46.4 | -21% |
| P8K/D1K c=4 | 34.0 | 136.0 | 162 | -16% |

> **Conclusion**: An ICI Slice (Workload Policy + MIG) multi-host setup is 15-21% slower than a single host. The gap is smaller under high concurrency (only -15% at c=16).
>
> **Startup time**: model loading ~5 min (boot disk) + XLA compilation ~29 min (backbone 10 rounds + compute_logits) + initialization ~6 min = about 40 min total. The first startup has no XLA cache; subsequent startups can be accelerated by persisting the cache.

---

## Firewall Rules

PD disaggregation and multi-node inference require internal network communication between VMs. It is recommended to allow all-port internal TCP (the TPUConnector's KV transfer and ZMQ side-channel use dynamic ports):

```bash
gcloud compute firewall-rules create allow-vllm-internal \
    --project=${PROJECT_ID} \
    --network=${VPC_NAME} \
    --allow=tcp \
    --source-ranges=10.0.0.0/8 \
    --description="Allow all internal TCP for vLLM/Ray/TPU communication"
```

Main ports reference:

| Port | Purpose |
|------|------|
| 6379 | Ray GCS server (multi-host) |
| 7000 | PD disaggregation proxy |
| 8000 | vLLM API (Prefill / single-host / Host 0) |
| 8471 | libtpu coordinator (multi-host ICI) |
| 9000 | vLLM API (Decode) |
| Dynamic | TPUConnector KV transfer / ZMQ side-channel |

---

## Resource Cleanup

```bash
# Delete single-host / PD disaggregation VMs
for vm in qwen3-vm-01 qwen3-prefill qwen3-decode; do
  gcloud compute instances delete $vm \
      --project=${PROJECT_ID} --zone=${ZONE} --quiet 2>/dev/null
done

# Delete the data disks (confirm they are no longer needed)
for disk in qwen3-data-01 qwen3-data-prefill qwen3-data-decode; do
  gcloud compute disks delete $disk \
      --project=${PROJECT_ID} --zone=${ZONE} --quiet 2>/dev/null
done

# Delete the multi-host slice (MIG + Template + Workload Policy)
SLICE_NAME=qwen3-slice
gcloud compute instance-groups managed delete ${SLICE_NAME}-mig \
    --project=${PROJECT_ID} --zone=${ZONE} --quiet
gcloud compute instance-templates delete ${SLICE_NAME}-it \
    --project=${PROJECT_ID} --quiet
gcloud compute resource-policies delete ${SLICE_NAME}-wp \
    --project=${PROJECT_ID} --region=${ZONE%-*} --quiet
```

---

## FAQ

### 1. vLLM startup OOM

You must add `--enable-expert-parallel`; otherwise the experts are replicated on every device.

### 2. vLLM import error / namespace package conflict

If you run `vllm serve` inside the `~/vllm/` directory, Python will treat `~/vllm/` as a namespace package, causing import errors. Fix:

```bash
cd /tmp   # must leave the ~/vllm/ directory
vllm serve ...
```

### 3. Wrong JAX version / libtpu error

The vLLM install downgrades JAX to 0.8.0; you must manually override the install:

```bash
source ~/vllm_env/bin/activate
uv pip install jax==0.9.2 jaxlib==0.9.2 libtpu==0.0.39 flax==0.12.4
python3 -c "import jax; print(jax.__version__)"  # should be 0.9.2
```

### 4. libtpu lockfile error

```bash
pkill -9 -f "vllm|EngineCore"; sleep 3; rm -f /tmp/libtpu_lockfile /tmp/.vllm_ipc_*
```

### 5. PD disaggregation Decode cannot connect to Prefill

Check the VPC internal IPs and firewall rules to ensure ports 8000/9000 are reachable:

```bash
# Test on the Decode VM
curl -s http://${PREFILL_IP}:8000/health
```

### 6. Multi-host AttributeError: d.coords

The TPU environment variables are not set correctly. Confirm that `TPU_WORKER_HOSTNAMES` contains the IPs of both VMs and that `TPU_WORKER_ID` is set to 0 and 1 on the two hosts respectively.

### 7. Multi-host Ray OOM / model files disappear

The Ray Object Store by default occupies 30% of `/dev/shm` (944 GB × 30% = ~280 GB). If the model is also in `/dev/shm` (450 GB), the two together exceed the `/dev/shm` capacity, causing:
- Model files "disappearing" because they are overwritten by the Ray plasma store
- Or OOM triggered by RAM double-counting when a Ray worker loads the model

**Solution**:
```bash
# 1. Put the model on the boot disk, not /dev/shm
mkdir -p ~/models && gcloud storage cp -r ${MODEL_BUCKET}/${MODEL_NAME} ~/models/

# 2. Limit the object store when starting Ray (100 GB is sufficient)
ray start --head ... --object-store-memory=107374182400

# 3. Disable the Ray OOM monitor (avoid false kills during the loading peak)
export RAY_memory_monitor_refresh_ms=0
```

### 8. Model weights missing the tokenizer

```bash
cd /dev/shm/Qwen3-Coder-480B-A35B-FP8/   # or ~/models/Qwen3-Coder-480B-A35B-FP8/ (multi-host)
for f in tokenizer.json tokenizer_config.json vocab.json; do
  curl -sL -o $f https://huggingface.co/Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8/resolve/main/$f
done
```
