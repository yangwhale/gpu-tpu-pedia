**English** | [中文](./README-TPU-VM.md)

# GLM-5.1 754B FP4 Inference on TPU v7x — TPU VM Edition

> End-to-end deployment guide for TPU VM: from creating the VM to completing the benchmark.
>
> **Model**: [zai-org/GLM-5.1-FP8](https://huggingface.co/zai-org/GLM-5.1-FP8) (142 safetensors, ~705 GB)
>
> **Architecture**: 754B total params / **MoE** (256 experts, top-8) + MLA + DSA + MTP / 78 layers / FP4 MoE + FP8 Attn + BF16 non-MoE
>
> **Code repository**: [yangwhale/tpu-inference](https://github.com/yangwhale/tpu-inference) (branch: `feature/glm51-inference`)
>
> **Model storage**: `gs://aidc-tpu-data/models` (GCS object storage; all model weights are stored here)
>
> For the GKE edition, see [README.md](README.md) in the same directory.

---

### Key Prerequisites

> **GLM-5.1 inference requires 3 extra preparation steps** (not needed for Qwen3.5/Qwen3-Coder):
>
> 1. **Generate FP4 MoE Cache**: `gen_fp4_cache_cpu_parallel.py`, pure CPU numpy, ~28 min
> 2. **Merge Non-MoE weights**: `extract_non_moe_weights.py`, ~2 min
> 3. **Copy Cache to /dev/shm**: ~4 min (parallel copy)
>
> These 3 steps are unique to the **first deployment**. On subsequent restarts, you only need to confirm the cache is still in /dev/shm to skip them.

### Three Mandatory Environment Variables

> **Must be set before every vLLM launch — none can be omitted!**

| Environment Variable | Value | Consequence If Missing |
|---------|-----|---------|
| `MOE_REQUANTIZE_WEIGHT_DTYPE=float4_e2m1fn` | **Must be set** | Defaults to looking up FP8 cache → cache miss → **HBM OOM** |
| `NEW_MODEL_DESIGN=1` | **Must be set** | Mandatory for MLA models; without it, exits with an error |
| `MOE_WEIGHT_CACHE_DIR=/dev/shm` | **Must be set** | FP4 cache not found, triggers online requantization |

`MOE_REQUANTIZE_WEIGHT_DTYPE` is the most easily overlooked and most fatal: it controls the cache subdirectory name. When unset, it defaults to `fp8`, so the subdirectory becomes `ep8_tp1_gmm_ep_fp8e4m3_bsNone`, whereas the FP4 cache is in `ep8_tp1_gmm_ep_fp4e2m1_bsNone`, causing all cache misses → OOM.

### Key Differences from Qwen3.5

| Dimension | Qwen3.5-397B | **GLM-5.1-754B** |
|------|-------------|-----------------|
| Architecture | Hybrid GDN+Attn | **Pure MoE + MLA** |
| Quantization | FP8 native | **FP4 MoE + FP8 Attn + BF16 non-MoE** |
| FP4 Cache | Not needed | **Required** (~705 GB, includes generation + copy workflow) |
| Parallelism strategy | TP=8 | **EP=8, TP=1** (controlled by `--additional-config` JSON) |
| Code branch | `main` | **`feature/glm51-inference`** (yangwhale fork) |
| vLLM entry point | `vllm serve` | `vllm serve` (must use the CLI entry; `python3 -m` triggers a circular import) |
| Chat stability | 5-shot only | **Chat works** (identifies itself as GLM / Z.ai) |
| Data disk requirement | ≥500 GB | **≥2 TB** (model 705 GB + FP4 cache 705 GB) |
| /dev/shm purpose | Stores model weights | **Stores FP4 cache** (model stays on disk) |
| PD Connector | TPUConnectorHMA | **TPUConnector** (non-hybrid) |

---

## Table of Contents

- [Part 1: Single-Node Inference](#part-1-single-node-inference)
- [Part 2: PD Disaggregation (1P1D)](#part-2-pd-disaggregation-1p1d)
- [Part 3: Multi-Node Inference (EP=16)](#part-3-multi-node-inference-ep16)

---

## Environment Variables

```bash
export PROJECT_ID=<your-project-id>
export ZONE=us-central1-c
export RESERVATION_NAME=<your-reservation>
export VPC_NAME=<your-vpc-name>
export SUBNET_NAME=<your-subnet-name>
export HF_TOKEN=<your-hf-token>
export MODEL_BUCKET=gs://aidc-tpu-data/models         # Model weights GCS path
export MODEL_NAME=GLM-5.1-FP8                          # Model directory name (matches GCS)
```

## Hardware Requirements

| Item | Single-node (Part 1 & 2) | Multi-node (Part 3) |
|------|-------------------|----------------|
| Machine type | `tpu7x-standard-4t` | `tpu7x-standard-4t` × 2 |
| TPU | v7x-8 (4 chips, 8 devices) | v7x-16 (8 chips, 16 devices) |
| HBM | 768 GB | 1,536 GB |
| Host memory | 944 GB | 944 GB × 2 |
| Boot disk | 1 TB Hyperdisk Balanced | 1 TB × 2 |
| Data disk | **≥2 TB** (model 705 GB + FP4 cache 705 GB) | ≥2 TB × 2 |
| /dev/shm | **≥800 GB** (FP4 cache ~705 GB) | TBD (see Part 3 notes) |

---

# Part 1: Single-Node Inference

## Step 1: Create Data Disk and VM

### 1.1 Create Hyperdisk ML Data Disk

```bash
gcloud compute disks create glm51-data-01 \
    --project=${PROJECT_ID} \
    --zone=${ZONE} \
    --type=hyperdisk-ml \
    --size=4TB \
    --provisioned-throughput=2500
```

> **Why 4 TB**: model ~705 GB + FP4 cache ~705 GB + merged Non-MoE file ~21 GB + temporary files ≈ 1.5 TB. 4 TB provides ample headroom. If you have insufficient Hyperdisk ML quota, you can switch to Hyperdisk Balanced 2 TB.

### 1.2 Create TPU VM

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

### 1.3 SSH Connection

```bash
VM_IP=$(gcloud compute instances describe glm51-vm-01 \
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

> **Key difference**: GLM-5.1 uses the `feature/glm51-inference` branch of `yangwhale/tpu-inference`, not the main branch of upstream `vllm-project/tpu-inference`.

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

# Create a Python 3.12 virtual environment
uv venv ~/vllm_env --python 3.12
source ~/vllm_env/bin/activate

# Clone tpu-inference (GLM-5.1 branch)
cd ~
git clone https://github.com/yangwhale/tpu-inference.git
cd tpu-inference
git checkout feature/glm51-inference

# Get the pinned vLLM commit hash
VLLM_VERSION_FILE=".buildkite/vllm_lkg.version"
if [ -f "$VLLM_VERSION_FILE" ]; then
    export VLLM_COMMIT_HASH="$(cat $VLLM_VERSION_FILE | tr -d '[:space:]')"
    echo "Pinned vLLM commit: ${VLLM_COMMIT_HASH}"
else
    echo "No pinned version found, using latest vLLM main"
    export VLLM_COMMIT_HASH=""
fi

# Clone vLLM and checkout the pinned version
cd ~
git clone https://github.com/vllm-project/vllm.git
cd vllm
if [ -n "${VLLM_COMMIT_HASH}" ]; then
    git checkout "${VLLM_COMMIT_HASH}"
fi

# Install vLLM (TPU target)
uv pip install -r requirements/tpu.txt --torch-backend=cpu
VLLM_TARGET_DEVICE="tpu" uv pip install -e .

# Fix JAX version (installing vLLM downgrades JAX; must reinstall 0.9.2)
uv pip install jax==0.9.2 jaxlib==0.9.2 libtpu==0.0.39 flax==0.12.4

# Install tpu-inference (--no-deps to avoid downgrading JAX again)
cd ~/tpu-inference
uv pip install -e . --no-deps

# Fix vLLM DSA buffer TPU compatibility issue
# In deepseek_v2.py, topk_indices_buffer uses device="tpu", which PyTorch does not recognize
# This buffer is only used by the GPU SparseAttnIndexer; on TPU, change it to "cpu"
sed -i 's/device=self\.device,/device="cpu" if self.device == "tpu" else self.device,/' \
  ~/vllm/vllm/model_executor/models/deepseek_v2.py
```

### 3.4 Verify Installation

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

Expected output (actual version numbers may vary):
```
vllm: 0.20.x
tpu_inference: 0.0.0
jax: 0.9.2
platform: TPU V7X
devices: 8 x tpu
```

> **About the JAX version**: vLLM's `requirements/tpu.txt` installs JAX 0.8.0, but TPU v7x requires JAX 0.9.2 + libtpu 0.0.39. After installing vLLM, you must manually override and install the correct version.
>
> **About `--no-deps`**: tpu-inference's `pyproject.toml` depends on jax/jaxlib; without `--no-deps`, it triggers a JAX downgrade again.
>
> **About `MODEL_IMPL_TYPE=flax_nnx`**: **Do not set it to `vllm`**. The `vllm` path uses `DefaultModelLoader`, which loads all 142 safetensors (705 GB) into host memory. Meanwhile, the FP4 cache in /dev/shm already occupies ~705 GB, leaving insufficient RAM → OOM Kill. The `flax_nnx` path uses JAX's `GlmMoeForCausalLM.load_weights()`, with built-in `_filter_moe_shards()` logic: once it detects the FP4 cache in /dev/shm, it skips all pure-MoE safetensors and loads only the non-MoE weights (~21 GB), reading the MoE part directly from cache.
>
> **About `vllm serve`**: **Do not use `python3 -m vllm.entrypoints.openai.api_server`**. The module entry point triggers a partial initialization of `vllm.__init__`, causing `from vllm import SamplingParams` to fail with a circular import. `vllm serve` CLI goes through the full startup path and has no such issue.

### 3.5 Set Runtime Environment Variables

```bash
# Base environment variables
export HF_TOKEN=${HF_TOKEN}
export JAX_PLATFORMS=tpu,cpu
export TPU_BACKEND_TYPE=jax
export PJRT_DEVICE=TPU
export MODEL_IMPL_TYPE=flax_nnx    # ⚠️ Must be flax_nnx, not vllm (see note below)
export USE_MOE_EP_KERNEL=0
export USE_BATCHED_RPA_KERNEL=0

# ⚠️ GLM-5.1 three mandatory environment variables
export MOE_REQUANTIZE_WEIGHT_DTYPE=float4_e2m1fn   # Controls FP4 cache lookup; omitting = OOM
export NEW_MODEL_DESIGN=1                           # Required for MLA models
export MOE_WEIGHT_CACHE_DIR=/dev/shm                # Points to the FP4 cache root directory
```

### 3.6 Download Model Weights

Model weights can be downloaded to the data disk from GCS or HuggingFace.

```bash
# Option A: Copy from GCS (recommended, fastest)
gcloud storage cp -r ${MODEL_BUCKET}/${MODEL_NAME} /mnt/data/

# Option B: Download from HuggingFace
pip install -U "huggingface_hub[hf_transfer]"
HF_HUB_ENABLE_HF_TRANSFER=1 huggingface-cli download \
  zai-org/GLM-5.1-FP8 --local-dir /mnt/data/GLM-5.1-FP8

# Verify
ls /mnt/data/GLM-5.1-FP8/*.safetensors | wc -l   # Should be 142
du -sh /mnt/data/GLM-5.1-FP8                      # Should be ~705 GB
```

Set the model path variable (used in subsequent steps):

```bash
export MODEL=/mnt/data/GLM-5.1-FP8
```

> **Difference from Qwen3.5**: The Qwen3.5 model is 378 GB and can be placed in /dev/shm to accelerate loading. GLM-5.1's model (705 GB) + FP4 cache (705 GB) total 1.4 TB, far exceeding /dev/shm capacity, so the **model stays on the data disk and /dev/shm only stores the FP4 cache**.
>
> **First-time upload of the model to GCS**: If the GCS bucket does not yet have the model weights, first download them from HuggingFace on any machine and upload:
> ```bash
> gcloud storage cp -r /mnt/data/GLM-5.1-FP8 ${MODEL_BUCKET}/
> ```

---

## Step 4: Generate FP4 Cache + Merge Non-MoE

> **First deployment only**. The FP4 MoE Cache and merged Non-MoE file only need to be generated once; subsequent restarts use them directly.

### 4.1 Download Scripts

```bash
cd /mnt/data
curl -LO https://raw.githubusercontent.com/yangwhale/gpu-tpu-pedia/main/tpu/tpu-inference/GLM-5.1-754B-FP4/gen_fp4_cache_cpu_parallel.py
curl -LO https://raw.githubusercontent.com/yangwhale/gpu-tpu-pedia/main/tpu/tpu-inference/GLM-5.1-754B-FP4/extract_non_moe_weights.py
```

### 4.2 Ensure /dev/shm Is Empty

```bash
df -h /dev/shm
ls /dev/shm/

# ⚠️ If there is old data, you must clean it! 12 workers peak at ~70 GB/worker; leftover data causes OOM kill
rm -rf /dev/shm/*
```

### 4.3 Generate FP4 MoE Cache (~28 min)

```bash
source ~/vllm_env/bin/activate

python3 -u /mnt/data/gen_fp4_cache_cpu_parallel.py \
  --model-dir $MODEL \
  --cache-dir /mnt/data/moe-cache/ep8_tp1_gmm_ep_fp4e2m1_bsNone \
  --workers 12

# Resumable after interruption (automatically skips completed layers)
```

> **Number of workers**: Adjust based on available RAM (each worker peaks at ~70 GB). A v7x-8 machine with 944 GB RAM → at most 12 workers.
>
> **Pure CPU operation**: No TPU/JAX needed; pure numpy computation. Can run on any machine with enough RAM.

### 4.4 Extract Non-MoE Weights (~2 min)

Merge the non-MoE weights scattered across 142 safetensors into a single file, **reducing load time from 4m26s → 21s**:

```bash
python3 /mnt/data/extract_non_moe_weights.py \
  --model-dir $MODEL \
  --output /mnt/data/moe-cache/ep8_tp1_gmm_ep_fp4e2m1_bsNone/non_moe_weights.safetensors
```

### 4.5 Verify

```bash
# Check the number of MoE layers
ls /mnt/data/moe-cache/ep8_tp1_gmm_ep_fp4e2m1_bsNone/ | grep model_layers | wc -l
# Expected: 76 (layer 3-78)

# Check the non-MoE file
ls -lh /mnt/data/moe-cache/ep8_tp1_gmm_ep_fp4e2m1_bsNone/non_moe_weights.safetensors
# Expected: ~21 GB

# Check FP4 shape
python3 -c "
import numpy as np
d = '/mnt/data/moe-cache/ep8_tp1_gmm_ep_fp4e2m1_bsNone/model_layers_3_mlp_experts'
for name in ['w13_weight', 'w13_weight_scale', 'w2_weight', 'w2_weight_scale']:
    a = np.load(f'{d}/{name}.npy')
    print(f'{name}: {a.shape} {a.dtype}')
"
# Expected output:
#   w13_weight:       (256, 6144, 4096) |V1    (float4_e2m1fn)
#   w13_weight_scale: (256, 1, 1, 4096) float32
#   w2_weight:        (256, 2048, 6144) |V1
#   w2_weight_scale:  (256, 1, 1, 6144) float32
```

---

## Step 5: Copy Cache to /dev/shm

Preload the FP4 cache + Non-MoE weights into `/dev/shm` (tmpfs) to **greatly accelerate startup + avoid MoE prefetch deadlock**.

```bash
# Expand /dev/shm (default ~472 GB; needs to hold the 705 GB FP4 cache)
sudo mount -o remount,size=800G /dev/shm

SRC=/mnt/data/moe-cache/ep8_tp1_gmm_ep_fp4e2m1_bsNone
DST=/dev/shm/ep8_tp1_gmm_ep_fp4e2m1_bsNone

mkdir -p $DST

# Copy non-MoE weights
cp $SRC/non_moe_weights.safetensors $DST/

# Parallel-copy 76 layers of MoE cache (8 workers, ~4 min)
ls -d $SRC/model_layers_* | xargs -P 8 -I {} cp -r {} $DST/

# Verify
ls $DST/ | grep model_layers | wc -l   # Expected: 76
ls -lh $DST/non_moe_weights.safetensors  # Expected: ~21 GB
df -h /dev/shm                           # Expected usage: ~705 GB
```

> **Do not use single-threaded `cp -r`**! Single-threaded takes ~8 min; `xargs -P 8` in parallel takes ~4 min.
>
> **Total usage**: FP4 cache + non-MoE ≈ **~705 GB**; /dev/shm at 800 GB is sufficient.
>
> **⚠️ /dev/shm is tmpfs**: Data is lost after a VM restart; you must re-copy it from the data disk (this Step 5).

### Optional Optimization: Comment Out `jax.clear_caches()`

The `jax.clear_caches()` call in `weight_utils.py` causes each tensor's `jax.device_put()` to recompile. The 2292 non-MoE tensors have only ~25 unique shapes, but each is recompiled every time.

```bash
source ~/vllm_env/bin/activate
TPI_DIR=$(python3 -c "import tpu_inference; import os; print(os.path.dirname(tpu_inference.__file__))" 2>/dev/null)

grep -n 'jax.clear_caches()' ${TPI_DIR}/models/jax/utils/weight_utils.py
# Comment out all occurrences of jax.clear_caches()
sed -i 's/^        jax.clear_caches()/#        jax.clear_caches()/' \
  ${TPI_DIR}/models/jax/utils/weight_utils.py

# Clean pycache
find ${TPI_DIR} -name '__pycache__' -type d -exec rm -rf {} + 2>/dev/null || true
```

> **Effect**: Non-MoE loading goes from 6m29s → ~25-108s (measured 10x speedup).
> **Risk**: None. The cache stores compiled transfer programs, ~25 shapes × a few KB ≈ less than 1 MB.

---

## Step 6: Launch vLLM (~4-11 min)

> **Important**: You must `cd /tmp` before running vLLM; otherwise the `~/vllm/` or `~/tpu-inference/` directory will be treated by Python as a namespace package, causing import errors.

```bash
source ~/vllm_env/bin/activate
cd /tmp && mkdir -p /tmp/vllm-logs

# ⚠️ Confirm the three mandatory environment variables
echo "MOE_REQUANTIZE_WEIGHT_DTYPE=${MOE_REQUANTIZE_WEIGHT_DTYPE}"  # Should be float4_e2m1fn
echo "NEW_MODEL_DESIGN=${NEW_MODEL_DESIGN}"                        # Should be 1
echo "MOE_WEIGHT_CACHE_DIR=${MOE_WEIGHT_CACHE_DIR}"                # Should be /dev/shm

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

# Wait until ready (about 4-11 min)
until curl -s -o /dev/null -w '%{http_code}\n' http://localhost:8000/health | grep -q 200; do
  date; tail -1 /tmp/vllm-logs/serve.log 2>/dev/null; sleep 30
done
echo "Server ready"
```

### Parameter Notes

| Parameter | Actual Meaning | Common Misunderstanding |
|------|----------|-------------|
| `--tensor-parallel-size 8` | Total number of devices | **Not TP=8**. Actually TP=1, EP=8 (controlled by additional-config) |
| `--quantization fp8` | vLLM quantization schema name | **Not FP8 inference**. MoE FP4 is controlled by environment variables |
| `expert_parallelism: 8` | EP=8 | 256 experts ÷ 8 = 32 experts per device |
| `tensor_parallelism: 1` | TP=1 | Attention weights are replicated instead of sharded |
| `--enforce-eager` | Disables ahead-of-time compilation | Required for MoE models; otherwise compilation times out |

> **Startup time**:
> - **Unoptimized**: ~11 min (non-MoE loading 6m29s)
> - **Optimized** (commenting out `jax.clear_caches()`): ~4-6 min

---

## Step 7: Verify Inference

In **another terminal**, SSH into the VM:

```bash
# Test 1: Math computation
curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "GLM-5.1-FP8",
    "messages": [{"role": "user", "content": "What is 2+3? Answer with just the number."}],
    "max_tokens": 256
  }' | python3 -m json.tool

# Test 2: Chinese conversation
curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "GLM-5.1-FP8",
    "messages": [{"role": "user", "content": "你是谁？用一句话介绍自己。"}],
    "max_tokens": 128
  }' | python3 -c "import sys,json; r=json.load(sys.stdin); print(r['choices'][0]['message']['content'])"

# Health check
curl -s http://localhost:8000/health
# Expected: {"status":"ok"}
```

> **GLM-5.1 is a thinking model**: The output may contain a `<think>...reasoning...</think>` reasoning process; this is normal behavior.

### Measured Verification Results (2026-04-24, GKE E2E pod, v7x-8)

| Test | Result |
|------|------|
| 2+3 math computation | ✅ Correctly answered 5 (with chain-of-thought reasoning) |
| Chinese self-introduction | ✅ Identifies itself as a "large language model created by Z.ai", fluent output |
| English logical reasoning | ✅ Correct reasoning |
| HBM usage | 58.43/94.75 GiB per device (61.6%) |
| MoE cache | 76/76 layers all hit (FP4) |

---

## Step 8: Benchmark

> **Note**: A bare-metal TPU VM does not have PyTorch installed, so the `vllm bench serve` command is unavailable. Use the following Python script instead.

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

### Expected Performance Reference (GKE measured values, 2026-04-24)

> Performance on a TPU VM should match the GKE Pod (same hardware, same software stack).

| Concurrency | Throughput (tok/s) | tok/s/chip | TTFT (s) | TPOT (ms) |
|---:|---:|---:|---:|---:|
| 1 | 28.4 | 7.1 | 0.534 | 35 |
| 4 | 130.5 | 32.6 | 0.510 | 30 |
| 16 | 444.8 | 111.2 | 1.016 | 35 |
| 64 | 1,570 | 392.5 | 3.174 | 38 |
| 256 | 3,873 | 968.3 | 8.869 | 57 |
| **1,024** | **6,504** | **1,626** | 31.38 | 125 |

**Key operating points:**

| Operating Point | Concurrency | Throughput | tok/s/chip | Use Case |
|--------|-----|-----------|-----------|---------|
| Max Throughput | 1,024 | 6,504 tok/s | 1,626 | Offline batch processing |
| Balanced | 64 | 1,570 tok/s | 393 | Medium-load online serving |
| Low Latency | 4 | 130 tok/s | 33 | Interactive conversation |

---

# Part 2: PD Disaggregation (1P1D)

> ⚠️ **Measured and verified: the PD pipeline works but has stability issues.** The full Prefill→KV transfer→Decode pipeline was verified on a TPU v7x-8 VM, but the DPScheduler (GLM-5.1 MLA forces `enable_dp_attention` on) and JAX multithreading have a fork deadlock issue, causing the engine to hang after handling about 2-3 PD requests. **Usable for short demos, but unstable for long-running serving.**
>
> Two TPU v7x-8 VMs: one runs Prefill (kv_producer), the other runs Decode (kv_consumer), transferring KV cache over the VPC internal network.
>
> **Architecture note**: PD disaggregation **does not require Ray**. The two vLLM instances run completely independently, transferring KV cache directly P2P via TPUConnector, with `toy_proxy_server.py` handling request routing.

### GLM-5.1 PD Must-Read Differences (vs Qwen3.5 PD)

| Item | Qwen3.5 (hybrid GDN+Attn) | **GLM-5.1 (pure MoE + MLA)** |
|---|---|---|
| `kv_connector` | `TPUConnectorHMA` | **`TPUConnector`** |
| `kv_connector_module_path` | `tpu_inference.distributed.tpu_connector_hma` | **`tpu_inference.distributed.tpu_connector`** |
| Hybrid KV cache flag | Needs `--no-disable-hybrid-kv-cache-manager` | **Not needed** (pure MoE, no hybrid) |
| HMA connector deployment | Needs manual download | **Not needed** (TPUConnector is already in the code) |
| FP4 cache | Not needed | **Both VMs need their own /dev/shm FP4 cache** |

## Step 1: Create 2 VMs

Reusing the Part 1 approach, create the Prefill and Decode VMs:

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

> **Disk-saving approach**: If the first VM has already generated the FP4 cache, you can set the data disk to `READ_ONLY_MANY` and mount it on both VMs simultaneously (read-only). Or, use one VM to generate the cache, then `gcloud storage cp` it to GCS, and have the second VM copy from GCS.

## Step 2: Run Environment Preparation on Both VMs

On both VMs, run:
1. Part 1 Step 2 (format and mount the data disk)
2. Part 1 Step 3.1 ~ 3.5 (system config + install vLLM/tpu-inference + set environment variables)
   - ⚠️ **tpu-inference must use the `yangwhale/feature/glm51-inference` branch** (contains the 20+ necessary commits for GLM-5.1's MoE cache, FP4 quantization, etc.)
   - ⚠️ **The tpu-inference versions on both VMs must match**, otherwise the TPUConnector KV transfer protocol may be incompatible
   - ⚠️ **The deepseek_v2.py patch must be applied on both VMs** (the sed command in Part 1 Step 3.3)
3. Part 1 Step 3.6 (download model weights to `/mnt/data/`)
4. Part 1 Step 4 (generate FP4 Cache + merge Non-MoE) — the second VM can copy the cache from GCS/the first VM
5. Part 1 Step 5 (copy Cache to /dev/shm)
6. **(Required) DPScheduler PD patch** — see Step 2.1 below

> **Both VMs need their own /dev/shm FP4 cache**. The FP4 cache is loaded separately in each VM's /dev/shm.

> ⚠️ **GCE TPU VM /dev/shm mount namespace isolation issue**: Each SSH session on a GCE TPU VM has its own mount namespace. This means data written to /dev/shm in one SSH session is **not visible** to a vLLM process started in another SSH session. **You must complete the /dev/shm cache copy and the vLLM launch in the same SSH session.** If the VM restarts or you re-SSH, you need to re-copy the FP4 cache to /dev/shm.

### Step 2.1: DPScheduler PD Patch (needed on both VMs)

GLM-5.1's MLA architecture requires `enable_dp_attention=true`, which makes vLLM use the DPScheduler (instead of the regular Scheduler). The DPScheduler creates 8 sub-scheduler processes via `multiprocessing.fork`, but the main scheduler's `self.connector` is hardcoded to `None`, so the KV connector metadata is not passed to the model runner, triggering `AssertionError: scheduler_output.kv_connector_metadata is not None` in PD mode.

Run the following patch on **both VMs**:

```bash
cd ~/tpu-inference

# Backup
cp tpu_inference/core/sched/dp_scheduler.py tpu_inference/core/sched/dp_scheduler.py.bak

# Find where DPSchedulerOutput is created in the _combine_scheduler_outputs method,
# and add kv_connector_metadata merge logic before the return
python3 << 'PATCH'
import re

with open('tpu_inference/core/sched/dp_scheduler.py', 'r') as f:
    content = f.read()

# Check whether already patched
if 'combined_kv_meta' in content:
    print('Patch already applied')
    exit(0)

# Insert the KV metadata merge code before 'return result' in the _combine_scheduler_outputs method
old = '        return result'
# Find the 'return result' in the _combine_scheduler_outputs method
# Need to precisely match the return within that method

patch_code = '''        # Combine kv_connector_metadata from sub-schedulers (PD disagg support)
        combined_kv_meta = None
        for output in rank_outputs:
            if output.kv_connector_metadata is not None:
                if combined_kv_meta is None:
                    combined_kv_meta = output.kv_connector_metadata
                else:
                    if hasattr(combined_kv_meta, 'reqs_to_send'):
                        combined_kv_meta.reqs_to_send.update(
                            output.kv_connector_metadata.reqs_to_send)
                    if hasattr(combined_kv_meta, 'reqs_to_load'):
                        combined_kv_meta.reqs_to_load.update(
                            output.kv_connector_metadata.reqs_to_load)
        if combined_kv_meta is not None:
            result.kv_connector_metadata = combined_kv_meta
        return result'''

# Replace only the return in _combine_scheduler_outputs
# Find the method and its return
idx = content.find('def _combine_scheduler_outputs')
if idx == -1:
    print('ERROR: _combine_scheduler_outputs not found')
    exit(1)

# Find 'return result' after that method
ret_idx = content.find('        return result', idx)
if ret_idx == -1:
    print('ERROR: return result not found in method')
    exit(1)

content = content[:ret_idx] + patch_code + content[ret_idx + len('        return result'):]

with open('tpu_inference/core/sched/dp_scheduler.py', 'w') as f:
    f.write(content)

print('Patch applied successfully')
PATCH
```

> **Principle**: The DPScheduler's 8 sub-schedulers each create a KV connector and generate `kv_connector_metadata`, but the main scheduler's `_combine_scheduler_outputs()` method only merges the scheduling outputs, losing the metadata. This patch merges the sub-schedulers' metadata and attaches it to the `DPSchedulerOutput`.

## Step 3: Get Internal IPs

```bash
PREFILL_IP=$(gcloud compute instances describe glm51-prefill \
    --project=${PROJECT_ID} --zone=${ZONE} \
    --format="value(networkInterfaces[0].networkIP)")
DECODE_IP=$(gcloud compute instances describe glm51-decode \
    --project=${PROJECT_ID} --zone=${ZONE} \
    --format="value(networkInterfaces[0].networkIP)")
echo "Prefill: ${PREFILL_IP}, Decode: ${DECODE_IP}"
```

## Step 4: Launch the Prefill Instance

SSH into the Prefill VM:

```bash
source ~/vllm_env/bin/activate
cd /tmp && mkdir -p /tmp/vllm-logs

# ⚠️ VLLM_HOST_IP must be set to this machine's internal IP (TPUConnector uses this address to establish the KV transfer server)
export VLLM_HOST_IP=${PREFILL_IP}

nohup env \
  PJRT_DEVICE=TPU TPU_BACKEND_TYPE=jax JAX_PLATFORMS=tpu,cpu \
  SKIP_JAX_PRECOMPILE=1 VLLM_XLA_CHECK_RECOMPILATION=0 \
  MODEL_IMPL_TYPE=flax_nnx HF_HUB_OFFLINE=1 \
  MOE_REQUANTIZE_WEIGHT_DTYPE=float4_e2m1fn \
  NEW_MODEL_DESIGN=1 \
  MOE_WEIGHT_CACHE_DIR=/dev/shm \
  VLLM_HOST_IP=${VLLM_HOST_IP} \
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

Key differences (vs Part 1 single-node):
- `VLLM_HOST_IP`: **Must be set to this machine's internal IP**. TPUConnector uses this address to start the KV transfer server (default port 9100) and ZMQ side channel (default port 9600); the Decode instance pulls KV cache through these ports
- `--gpu-memory-utilization=0.70` (leave 30% HBM for the KV transfer buffer)
- `kv_role=kv_producer`
- `--max-model-len=16384` (PD mode supports longer context)
- Uses `TPUConnector` (not `TPUConnectorHMA`; GLM-5.1 is non-hybrid)

> Startup takes 5~12 minutes. Use `tail -f /tmp/vllm-logs/prefill.log` to monitor progress.
>
> ⚠️ **You must confirm the /dev/shm FP4 cache is visible in the same SSH session before launching vLLM.** If you open a new SSH session, first run `ls /dev/shm/ep8_tp1_gmm_ep_fp4e2m1_bsNone/model_layers_10_mlp_experts/meta.json` to confirm the cache exists. If it does not exist, you need to re-copy it (Part 1 Step 5).

## Step 5: Launch the Decode Instance

SSH into the Decode VM (**note: you must complete the /dev/shm copy and vllm launch in the same SSH session**):

```bash
source ~/vllm_env/bin/activate
cd /tmp && mkdir -p /tmp/vllm-logs

# ⚠️ VLLM_HOST_IP must be set to the Decode VM's own internal IP
export VLLM_HOST_IP=${DECODE_IP}

nohup env \
  PJRT_DEVICE=TPU TPU_BACKEND_TYPE=jax JAX_PLATFORMS=tpu,cpu \
  SKIP_JAX_PRECOMPILE=1 VLLM_XLA_CHECK_RECOMPILATION=0 \
  MODEL_IMPL_TYPE=flax_nnx HF_HUB_OFFLINE=1 \
  MOE_REQUANTIZE_WEIGHT_DTYPE=float4_e2m1fn \
  NEW_MODEL_DESIGN=1 \
  MOE_WEIGHT_CACHE_DIR=/dev/shm \
  VLLM_HOST_IP=${VLLM_HOST_IP} \
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

Key differences (vs Prefill): `--gpu-memory-utilization=0.90` (Decode does not need to reserve a transfer buffer), `kv_role=kv_consumer`, `port=9000`.

> **Network requirement**: The Decode VM needs to access the Prefill VM's ports **9100** (KV transfer) and **9600** (ZMQ side channel). Within the same VPC internal network, no additional firewall rules are usually needed, but if you have a Network Firewall Policy, confirm these two ports are allowed.

## Step 6: Verify Both Ends Ready + Launch Proxy

On the Prefill VM, run:

```bash
source ~/vllm_env/bin/activate
export DECODE_IP=<decode-vm-internal-ip>

# Confirm both instances are ready
curl -s http://localhost:8000/v1/models | python3 -m json.tool
curl -s http://${DECODE_IP}:9000/v1/models | python3 -m json.tool
```

After both return model information, launch the proxy:

```bash
python3 ~/tpu-inference/examples/disagg/toy_proxy_server.py \
  --host 0.0.0.0 --port 7000 \
  --prefiller-hosts localhost --prefiller-ports 8000 \
  --decoder-hosts ${DECODE_IP} --decoder-ports 9000
```

Smoke test (via proxy port 7000):

```bash
curl -s http://localhost:7000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "GLM-5.1-FP8",
    "messages": [{"role": "user", "content": "What is 2+3? Answer with just the number."}],
    "max_tokens": 50
  }' | python3 -m json.tool
```

> **First-request latency**: The first request triggers XLA compilation, taking about 2~3 minutes each for Prefill and Decode. Subsequent requests hit the compilation cache.

## Step 7: PD Disaggregation Benchmark

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

### ⚠️ Known PD Disaggregation Issues (Measured)

| # | Issue | Status | Details |
|---|------|------|------|
| 1 | DPScheduler does not pass kv_connector_metadata | **Fixed** | Resolved by the patch in Step 2.1. Without the patch, you get `AssertionError: scheduler_output.kv_connector_metadata is not None` |
| 2 | /dev/shm mount namespace isolation | **Note required** | Each SSH session on a GCE TPU VM has its own mount namespace; the FP4 cache copy and vLLM launch must be in the same session |
| 3 | DPScheduler + JAX fork deadlock | **Unresolved** | `enable_dp_attention=true` (forced by MLA) → DPScheduler creates 8 sub-processes via `multiprocessing.fork` → fork is unsafe in a JAX multithreaded environment → the engine hangs after handling about 2-3 PD requests (futex deadlock, 2000+ threads). The root cause is the `RuntimeWarning: os.fork() was called. os.fork() is incompatible with multithreaded code, and JAX is multithreaded` in the vLLM log |
| 4 | TPUConnector compatibility with EP+DP sharding | **Verified** | KV transfer works correctly under EP=8, DP_attention=true |
| 5 | `--enable-prefix-caching` + `--enable-chunked-prefill` with PD | **Verified** | Both flags are compatible with PD; no need to remove them |
| 6 | FP4 cache + KV transfer buffer space | **Verified** | /dev/shm 800G is sufficient (cache 705G + buffer); no HBM conflict |
| 7 | First-request XLA compilation latency | **Expected behavior** | Prefill and Decode each need about 3 minutes for the first XLA compilation; subsequent requests hit the cache (Prefill <1s, Decode <1s) |

---

# Part 3: Multi-Node Inference (EP=16)

> ⚠️ **This section is a theoretical design and has not yet been measured.** Parameters are derived from GKE single-node verification + Qwen3.5 multi-host experience.
>
> Two TPU v7x-8 VMs form a v7x-16 slice (8 chips, 16 devices), interconnected via high-speed ICI.
>
> **Key difference**: GLM-5.1 multi-host uses **EP=16** (not TP=16). 256 experts ÷ 16 devices = 16 experts/device (vs 32/device single-node), halving the MoE HBM usage.

### Key Differences Between Multi-host and Single-host

| # | Multi-host-specific change | Notes |
|---|---|---|
| 1 | `expert_parallelism: 16` (additional-config) | 256 experts distributed across 16 devices |
| 2 | `--tensor-parallel-size 16` | Total number of devices; actual TP=1 unchanged |
| 3 | `--distributed-executor-backend ray` | Requires a Ray cluster |
| 4 | FP4 cache directory name changes | `ep16_tp1_gmm_ep_fp4e2m1_bsNone` (from ep8 to ep16) |
| 5 | No mrope/hybrid patches | GLM-5.1 does not need Qwen3.5's 3 patches |

## Step 1: Create a v7x-16 TPU Slice

A TPU7x multi-host slice must be created via the **Workload Policy + Instance Template + MIG** trio to ensure physical ICI interconnection.

### 1.1 Create Workload Policy

```bash
SLICE_NAME=glm51-slice

gcloud compute resource-policies create workload-policy ${SLICE_NAME}-wp \
    --type=HIGH_THROUGHPUT \
    --accelerator-topology=2x2x2 \
    --project=${PROJECT_ID} \
    --region=${ZONE%-*}
```

### 1.2 Create Instance Template

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

> **Note**: `--instance-termination-action=DELETE` is a required parameter for RESERVATION_BOUND + MIG. The subnet must use the full path.

### 1.3 Create MIG (TPU Slice)

```bash
gcloud compute instance-groups managed create ${SLICE_NAME}-mig \
    --project=${PROJECT_ID} \
    --zone=${ZONE} \
    --template=${SLICE_NAME}-it \
    --size=2 \
    --default-action-on-vm-failure=do-nothing \
    --workload-policy=projects/${PROJECT_ID}/regions/${ZONE%-*}/resourcePolicies/${SLICE_NAME}-wp
```

### 1.4 Get VM Names and IPs

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

### 1.5 Verify ICI Interconnection

SSH into either VM:

```bash
curl -s -H "Metadata-Flavor: Google" \
  http://metadata.google.internal/computeMetadata/v1/instance/attributes/accelerator-type
# Should output: v7x-16

curl -s -H "Metadata-Flavor: Google" \
  http://metadata.google.internal/computeMetadata/v1/instance/attributes/worker-id
# Host 0 = 0, Host 1 = 1

curl -s -H "Metadata-Flavor: Google" \
  http://metadata.google.internal/computeMetadata/v1/instance/attributes/tpu-env | grep TOPOLOGY
# Should output: TOPOLOGY: 2x2x2
```

## Step 2: Environment Preparation + Model/Cache Copy

On both VMs, run:

1. Part 1 Step 3.1 ~ 3.5 (system config + install vLLM/tpu-inference + set environment variables)
2. Attach and format/mount the data disk (Part 1 Step 1 + Step 2), or skip the additional data disk and place the model on the boot disk

### Model and FP4 Cache Copy (run on both VMs)

```bash
# Copy model weights to the data disk (or boot disk ~/models/)
gcloud storage cp -r ${MODEL_BUCKET}/${MODEL_NAME} /mnt/data/
export MODEL=/mnt/data/GLM-5.1-FP8

# Verify
ls ${MODEL}/*.safetensors | wc -l   # Should be 142
```

### FP4 Cache Handling

In multi-host, /dev/shm needs to coexist with the Ray Object Store. There are two options:

**Option A (recommended): Place FP4 cache in /dev/shm, limit the Ray Object Store**

```bash
# Expand /dev/shm
sudo mount -o remount,size=850G /dev/shm

# Generate the FP4 cache on the first VM (if already present, copy from GCS or another VM)
python3 -u /mnt/data/gen_fp4_cache_cpu_parallel.py \
  --model-dir $MODEL \
  --cache-dir /mnt/data/moe-cache/ep8_tp1_gmm_ep_fp4e2m1_bsNone \
  --workers 12

python3 /mnt/data/extract_non_moe_weights.py \
  --model-dir $MODEL \
  --output /mnt/data/moe-cache/ep8_tp1_gmm_ep_fp4e2m1_bsNone/non_moe_weights.safetensors

# Copy to /dev/shm (EP=16 directory name)
# The cache content is identical to EP=8, only the directory name differs (tpu-inference looks up the directory by EP value)
SRC=/mnt/data/moe-cache/ep8_tp1_gmm_ep_fp4e2m1_bsNone
DST=/dev/shm/ep16_tp1_gmm_ep_fp4e2m1_bsNone

mkdir -p $DST
cp $SRC/non_moe_weights.safetensors $DST/
ls -d $SRC/model_layers_* | xargs -P 8 -I {} cp -r {} $DST/

# Verify
ls $DST/ | grep model_layers | wc -l   # Expected: 76
df -h /dev/shm                           # Expected usage: ~705 GB
```

> **EP=16 vs EP=8 directory name**: tpu-inference looks up the cache directory using the format `ep{EP}_tp{TP}_gmm_ep_{dtype}_bsNone`. With EP=16, it looks up `ep16_tp1_...`. The cache content is identical (all 256 experts); sharding happens at load time.

**Option B: Place FP4 cache on disk (if /dev/shm space is insufficient)**

```bash
# Place the cache directly on the data disk, do not copy to /dev/shm
# Create a symlink for EP=16
ln -s /mnt/data/moe-cache/ep8_tp1_gmm_ep_fp4e2m1_bsNone \
      /mnt/data/moe-cache/ep16_tp1_gmm_ep_fp4e2m1_bsNone

# Point MOE_WEIGHT_CACHE_DIR to the disk
export MOE_WEIGHT_CACHE_DIR=/mnt/data/moe-cache
```

> ⚠️ **Option B risk**: Disk loading is about 100x slower than tmpfs. If the cache directory is incomplete (missing meta.json), it may trigger a MoE prefetch deadlock. Make sure all 76 layer directories have complete `.npy` files and `meta.json`.

## Step 3: Set TPU Topology Environment Variables

### Host 0 (Ray Head)

```bash
source ~/vllm_env/bin/activate

# Base environment variables
export PJRT_DEVICE=TPU
export TPU_BACKEND_TYPE=jax
export JAX_PLATFORMS=
export MODEL_IMPL_TYPE=flax_nnx
export USE_MOE_EP_KERNEL=0
export USE_BATCHED_RPA_KERNEL=0
export HF_HUB_OFFLINE=1
export SKIP_JAX_PRECOMPILE=1
export VLLM_XLA_CHECK_RECOMPILATION=0

# ⚠️ GLM-5.1 three mandatory environment variables
export MOE_REQUANTIZE_WEIGHT_DTYPE=float4_e2m1fn
export NEW_MODEL_DESIGN=1
export MOE_WEIGHT_CACHE_DIR=/dev/shm   # Option A; for Option B, change to the disk path

# Multi-host TPU topology variables (replace <HOST0_IP> and <HOST1_IP>)
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

### Host 1 (Ray Worker)

```bash
source ~/vllm_env/bin/activate

# Base environment variables (same as above)
export PJRT_DEVICE=TPU
export TPU_BACKEND_TYPE=jax
export JAX_PLATFORMS=
export MODEL_IMPL_TYPE=flax_nnx
export USE_MOE_EP_KERNEL=0
export USE_BATCHED_RPA_KERNEL=0
export HF_HUB_OFFLINE=1
export SKIP_JAX_PRECOMPILE=1
export VLLM_XLA_CHECK_RECOMPILATION=0

# ⚠️ GLM-5.1 three mandatory environment variables
export MOE_REQUANTIZE_WEIGHT_DTYPE=float4_e2m1fn
export NEW_MODEL_DESIGN=1
export MOE_WEIGHT_CACHE_DIR=/dev/shm

# Multi-host TPU topology variables
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

> **Key differences**:
> - `TPU_WORKER_ID=0` vs `TPU_WORKER_ID=1`
> - `VLLM_HOST_IP` set to each host's own IP
> - `JAX_PLATFORMS=` (empty) — must be empty for multi-host

## Step 4: Launch the Ray Cluster + vLLM

### Host 0 (start the Ray Head first)

```bash
# --object-store-memory limits the Ray plasma store to 50 GB (leaving more for the FP4 cache)
RAY_memory_monitor_refresh_ms=0 ray start --head \
  --port=6379 \
  --node-ip-address=${VLLM_HOST_IP} \
  --resources='{"TPU": 4}' \
  --object-store-memory=53687091200

sleep 20
ray status
```

### Host 1 (start the Ray Worker)

```bash
RAY_memory_monitor_refresh_ms=0 ray start \
  --address=<HOST0_IP>:6379 \
  --node-ip-address=${VLLM_HOST_IP} \
  --resources='{"TPU": 4}' \
  --object-store-memory=53687091200
```

### Host 0 (launch vLLM after confirming the cluster is ready)

```bash
ray status   # Confirm 2 nodes, 8 TPU

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

# Wait until ready
until curl -s -o /dev/null -w '%{http_code}\n' http://localhost:8000/health | grep -q 200; do
  date; tail -1 /tmp/vllm-logs/serve.log 2>/dev/null; sleep 60
done
echo "Server ready"
```

> **Key parameter notes**:
>
> | Parameter | Value | Notes |
> |------|-----|------|
> | `--tensor-parallel-size` | `16` | Total number of devices (2 nodes × 8 devices) |
> | `expert_parallelism` | `16` | 256 experts ÷ 16 = 16 experts per device |
> | `tensor_parallelism` | `1` | Attention weights are still replicated |
> | `--object-store-memory` | `53687091200` (50 GB) | Shrink the Ray plasma store to free up /dev/shm for the FP4 cache |
> | `RAY_memory_monitor_refresh_ms=0` | | Disable the Ray OOM monitor |
>
> **Note**: multi-host does not support `--async-scheduling` (a Ray executor limitation).

## Step 5: Verification and Benchmark

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

### Multi-host Performance Estimate

> Based on the performance degradation pattern measured for Qwen3.5 multi-host (-15% ~ -21%), the estimates are as follows:

| Scenario | Estimated tok/s | Single-node reference | Estimated vs single-node |
|------|----------:|--------:|------------:|
| P1K/D1K c=1 | ~22 | 28.4 | ~-22% |
| P1K/D1K c=4 | ~100 | 130.5 | ~-23% |
| P1K/D1K c=64 | ~1,250 | 1,570 | ~-20% |
| P1K/D1K c=1024 | ~5,200 | 6,504 | ~-20% |

> The advantage of multi-host is not higher throughput, but **larger KV cache capacity** (1,536 GB HBM → supports 16K+ context) and **lower MoE memory pressure** (EP=16 → 16 experts per device vs 32).

### ⚠️ Multi-host Known Risk Points

| # | Risk | Troubleshooting Direction |
|---|------|---------|
| 1 | Whether `--additional-config`'s EP=16 works correctly under Ray multi-host | Check sharding initialization in the vLLM log |
| 2 | Whether the FP4 cache directory naming matches `ep16_tp1_...` | If cache miss, check the actual lookup path |
| 3 | Whether `/dev/shm` space is sufficient (FP4 cache 705 GB + Ray 50 GB) | Monitor `df -h /dev/shm`; if OOM, switch to Option B |
| 4 | Whether `--enforce-eager` is compatible under the Ray executor | If compilation errors, try removing it |
| 5 | Compatibility of `--enable-prefix-caching` / `--enable-chunked-prefill` with Ray | If startup hangs, remove these two flags |

---

## Firewall Rules

PD disaggregation and multi-node inference require internal communication between VMs:

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
| 8000 | vLLM API (Prefill / single-node / Host 0) |
| 8471 | libtpu coordinator (multi-host ICI) |
| 9000 | vLLM API (Decode) |
| Dynamic | TPUConnector KV transfer / ZMQ side-channel |

---

## Resource Cleanup

```bash
# Stop vLLM / Ray
pgrep -f 'vllm|EngineCore|ray' | xargs -r kill -9

# Delete the single-node VM + data disk
gcloud compute instances delete glm51-vm-01 --project=${PROJECT_ID} --zone=${ZONE} --quiet
gcloud compute disks delete glm51-data-01 --project=${PROJECT_ID} --zone=${ZONE} --quiet

# Delete the PD disaggregation VMs + data disks
gcloud compute instances delete glm51-prefill glm51-decode --project=${PROJECT_ID} --zone=${ZONE} --quiet
gcloud compute disks delete glm51-data-prefill glm51-data-decode --project=${PROJECT_ID} --zone=${ZONE} --quiet

# Delete the multi-host slice (MIG → Template → Workload Policy)
SLICE_NAME=glm51-slice
gcloud compute instance-groups managed delete ${SLICE_NAME}-mig --project=${PROJECT_ID} --zone=${ZONE} --quiet
gcloud compute instance-templates delete ${SLICE_NAME}-it --project=${PROJECT_ID} --quiet
gcloud compute resource-policies delete ${SLICE_NAME}-wp --project=${PROJECT_ID} --region=${ZONE%-*} --quiet
```

---

## Troubleshooting

| Symptom | Root Cause | Fix |
|---|---|---|
| **`CompileTimeHbmOom: Used 651G of 94.75G hbm`** | `MOE_REQUANTIZE_WEIGHT_DTYPE` not set, looks up FP8 cache → miss → OOM | `export MOE_REQUANTIZE_WEIGHT_DTYPE=float4_e2m1fn` |
| **`MLA models require NEW_MODEL_DESIGN=1`** | Missing `NEW_MODEL_DESIGN` environment variable | `export NEW_MODEL_DESIGN=1` |
| **vLLM hangs (0% CPU, all threads in futex_wait)** | MoE prefetch deadlock: cache loaded from disk, or cache directory incomplete | Ensure the cache is in /dev/shm (tmpfs), and that all 76 layers have complete files |
| **OOM Kill during FP4 cache generation (exit 137)** | /dev/shm has old data, squeezing out worker memory | `rm -rf /dev/shm/*` then regenerate |
| **TPU device busy** | Previous vLLM exited abnormally, orphan process holding the TPU | `pgrep -f 'EngineCore\|vllm' \| xargs -r kill -9` + `rm -f /tmp/libtpu_lockfile` |
| **Multiple cache directories appear in `/dev/shm`** | Both FP4 and FP8 caches exist | Delete `ep8_tp1_gmm_ep_fp8e4m3_bsNone`, keep only `fp4e2m1` |
| **OOM Kill with `MODEL_IMPL_TYPE=vllm`** | `DefaultModelLoader` loads all 705 GB of safetensors; the /dev/shm cache already occupies ~705 GB, leaving insufficient RAM | Change to `MODEL_IMPL_TYPE=flax_nnx` (the JAX path skips MoE shards and loads only the ~21 GB non-MoE) |
| **`ImportError: cannot import name 'SamplingParams'`** | Using `python3 -m vllm.entrypoints.openai.api_server` causes a circular import | Switch to the `vllm serve` CLI entry |
| **`RuntimeError: Expected one of cpu, cuda...` (torch.empty)** | The DSA buffer in `deepseek_v2.py` uses `device="tpu"`, which PyTorch does not recognize | Modify `deepseek_v2.py` to change the buffer device to `"cpu"` (this buffer is only used by the GPU path) |
| **vLLM import error / namespace package** | Running inside the `~/vllm/` or `~/tpu-inference/` directory | `cd /tmp` before running |
| **Wrong JAX version / libtpu error** | Installing vLLM downgraded JAX | `uv pip install jax==0.9.2 jaxlib==0.9.2 libtpu==0.0.39 flax==0.12.4` |
| **PD: KV transfer failure** | Firewall not opening internal ports | Check the VPC firewall rules; allow `10.0.0.0/8` TCP on all ports |
| **Multi-host: Ray worker killed** | Ray OOM monitor false kill | Set `RAY_memory_monitor_refresh_ms=0` |
| **Multi-host: /dev/shm OOM** | FP4 cache + Ray Object Store exceeds /dev/shm | Reduce `--object-store-memory` or switch to disk Option B |

---

## End-to-End Workflow Summary

```
Step 1: Create data disk (≥2 TB) + VM
    ↓
Step 2: Format and mount the data disk
    ↓
Step 3: Environment preparation (uv + vLLM + tpu-inference feature/glm51-inference)
    ↓
Step 4: Generate FP4 Cache (~28 min) + merge Non-MoE (~2 min)   ← GLM-5.1 only
    ↓
Step 5: Copy Cache to /dev/shm (~4 min) + optional optimization
    ↓
Step 6: Launch vLLM (⚠️ three environment variables! ~4-11 min)
    ↓
Step 7: curl to verify inference
    ↓
Step 8: Benchmark
```

> **Total first-deployment time** (excluding model download): FP4 generation 28 min + merge 2 min + copy 4 min + startup ~11 min ≈ **~45 min**
>
> **Subsequent restarts**: only Step 6-7 (if the /dev/shm cache is still present), **~4-11 min**

---

## References

| Resource | Link |
|------|------|
| GLM-5.1 GKE deployment guide | [README.md](README.md) |
| DeepSeek R1 FP4 inference guide | [../DeepSeek-R1-671B-FP4/README.md](../DeepSeek-R1-671B-FP4/README.md) |
| Qwen3.5 TPU VM deployment guide | [../Qwen3.5-397B-A17B-FP8/README-TPU-VM.md](../Qwen3.5-397B-A17B-FP8/README-TPU-VM.md) |
| GLM-5.1 HuggingFace model | [zai-org/GLM-5.1-FP8](https://huggingface.co/zai-org/GLM-5.1-FP8) |
| tpu-inference code | [yangwhale/tpu-inference](https://github.com/yangwhale/tpu-inference) branch: `feature/glm51-inference` |
