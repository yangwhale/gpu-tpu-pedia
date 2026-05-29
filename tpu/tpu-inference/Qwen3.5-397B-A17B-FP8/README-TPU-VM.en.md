**English** | [中文](./README-TPU-VM.md)

# Qwen3.5-397B-A17B-FP8 Inference on TPU v7x — TPU VM Edition

> End-to-end TPU VM deployment guide: from creating the VM to completing the benchmark.
>
> **Model**: [Qwen/Qwen3.5-397B-A17B-FP8](https://huggingface.co/Qwen/Qwen3.5-397B-A17B-FP8) (94 safetensors, ~378 GiB, FP8 native)
>
> **Architecture**: 397B total params / 17B activated / **hybrid GDN+Attention** (45 GDN + 15 Standard Attn) + 512 routed experts + FP8 native
>
> **Code repository**: [vllm-project/tpu-inference](https://github.com/vllm-project/tpu-inference) (main branch ≥ 2026-05-15, includes [PR #2366](https://github.com/vllm-project/tpu-inference/pull/2366) + [PR #2577](https://github.com/vllm-project/tpu-inference/pull/2577) commit `04077875`)
>
> **Model storage**: `gs://aidc-tpu-data/models` (GCS object storage; model weights are stored uniformly here)
>
> For the GKE edition, see [README.md](README.md) in the same directory.

---

### Known Critical Limitations (historical bugs, already fixed)

> **The chat infinite-loop bug was fixed on 2026-05-15** ([vllm-project/tpu-inference PR #2577](https://github.com/vllm-project/tpu-inference/pull/2577), commit `04077875`).
>
> **You MUST use a main-branch image/code with commit ≥ `04077875` (date ≥ 2026-05-15).**
>
> #### Historical symptoms (versions < 2026-05-15)
> - **Chat infinite loop**: with thinking OFF, output is `about about about...` or garbled languages; with thinking ON, output is a `\n/**\n/**` infinite loop or blank content
> - **Stable fallback**: 5-shot Q/A completion + `enable_thinking:false` (this is the pattern used for GSM8K 93.93%)
>
> #### Root cause
> The GDN (Gated Linear Attention) recurrent scan kernel is numerically unstable at bf16 precision, causing consecutive tokens to degenerate into a repeated pattern.
>
> #### Fix contents (PR #2577)
> - `tpu_inference/kernels/gdn/recurrent_scan_v2.py`: upcast internal computation to `jnp.float32` (5 places)
> - `tpu_inference/layers/common/gdn_attention.py`: `chunk_size` 64→32
> - **Also takes effect for TP=8 users** (not just DP attention mode)
>
> #### Verification tests (chat path, after PR #2577)
> | Test | Output | finish_reason |
> |------|------|---------------|
> | `/v1/completions` `"The capital of France is"` | `" Paris."` | `stop` ✅ |
> | `/v1/chat/completions` same prompt | `"The capital of France is **Paris**."` | `stop` ✅ |
> | `/v1/chat/completions` GSM8K Janet ducks question | `"$18"` (correct answer, with reasoning) | `stop` ✅ |
>
> #### How to upgrade
> - **TPU VM**: re-run `pip install vllm tpu-inference` (mind the nightly index) so that it is ≥ 2026-05-15
> - **GKE**: pull `vllm/vllm-tpu:nightly` then use `sha256:d39995c6193e012967d57409c5a5d1e20a2e5242fbced458ee2ee210fb1e8bc0` or a newer digest
> - **Verify**: see [Step 4: Verify patches](#step-4-验证-patch-pr-2366--pr-2577)

---

## Table of Contents

- [Part 1: Single-host inference](#part-1-单机推理)
- [Part 2: PD disaggregation (1P1D)](#part-2-pd-分离-1p1d)
- [Part 3: Multi-node inference (TP=16)](#part-3-多节点推理-tp16)

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
export MODEL_NAME=Qwen3.5-397B-A17B-FP8               # Model directory name (matches GCS)
```

## Hardware Requirements

| Item | Single-host (Part 1 & 2) | Multi-node (Part 3) |
|------|-------------------|----------------|
| Machine type | `tpu7x-standard-4t` | `tpu7x-standard-4t` × 2 |
| TPU | v7x-8 (4 chips, 8 devices) | v7x-16 (8 chips, 16 devices) |
| HBM | 768 GB | 1,536 GB |
| Host memory | 944 GB | 944 GB × 2 |
| Boot disk | 1 TB Hyperdisk Balanced | 1 TB × 2 |
| Data disk | ≥500 GB (model ~378 GiB) | Not needed (model copied to boot disk ~/models/) |

---

# Part 1: Single-host Inference

## Step 1: Create the data disk and VM

### 1.1 Create the Hyperdisk ML data disk

```bash
gcloud compute disks create qwen35-data-01 \
    --project=${PROJECT_ID} \
    --zone=${ZONE} \
    --type=hyperdisk-ml \
    --size=2TB \
    --provisioned-throughput=2500
```

### 1.2 Create the TPU VM

```bash
gcloud compute instances create qwen35-vm-01 \
    --project=${PROJECT_ID} \
    --zone=${ZONE} \
    --machine-type=tpu7x-standard-4t \
    --network-interface=network=${VPC_NAME},subnet=${SUBNET_NAME} \
    --maintenance-policy=TERMINATE \
    --provisioning-model=RESERVATION_BOUND \
    --reservation-affinity=specific \
    --reservation=${RESERVATION_NAME} \
    --create-disk=auto-delete=yes,boot=yes,size=1000GB,type=hyperdisk-balanced,image=projects/ubuntu-os-accelerator-images/global/images/family/ubuntu-accel-2404-amd64-tpu-tpu7x \
    --disk=name=qwen35-data-01,device-name=data-disk,mode=rw,auto-delete=no \
    --no-shielded-secure-boot \
    --scopes=cloud-platform
```

### 1.3 SSH connection

```bash
VM_IP=$(gcloud compute instances describe qwen35-vm-01 \
    --project=${PROJECT_ID} --zone=${ZONE} \
    --format="value(networkInterfaces[0].accessConfigs[0].natIP)")
ssh -o StrictHostKeyChecking=no -i ~/.ssh/google_compute_engine ${USER}@${VM_IP}
```

---

## Step 2: Format and mount the data disk

```bash
sudo mkfs.ext4 -m 0 -E lazy_itable_init=0,lazy_journal_init=0,discard /dev/nvme1n1
sudo mkdir -p /mnt/data
sudo mount -o discard,defaults /dev/nvme1n1 /mnt/data
sudo chmod a+w /mnt/data
echo "/dev/disk/by-id/google-data-disk /mnt/data ext4 discard,defaults,nofail 0 2" | sudo tee -a /etc/fstab
```

---

## Step 3: Environment preparation

### 3.1 System configuration

```bash
sudo sysctl -w vm.max_map_count=8388608
echo 'vm.max_map_count=8388608' | sudo tee -a /etc/sysctl.conf

if [ -f /sys/module/vfio_iommu_type1/parameters/dma_entry_limit ]; then
  echo 2000000 | sudo tee /sys/module/vfio_iommu_type1/parameters/dma_entry_limit
fi
```

### 3.2 Install system dependencies

```bash
sudo apt-get update -qq
sudo apt-get install -y -qq libopenmpi-dev libomp-dev git curl
```

### 3.3 Install vLLM + tpu-inference (bare metal)

Use `uv` to create a Python 3.12 virtual environment and install vLLM (TPU edition) and tpu-inference:

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

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

# Fix the JAX version (installing vLLM downgrades JAX to 0.8.0; you must reinstall 0.9.2)
uv pip install jax==0.9.2 jaxlib==0.9.2 libtpu==0.0.39 flax==0.12.4

# Install tpu-inference (--no-deps to avoid downgrading JAX again)
cd ~/tpu-inference
uv pip install -e . --no-deps
```

### 3.4 Verify the installation

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

Expected output (version numbers depend on the actual install):
```
vllm: 0.20.x
tpu_inference: 0.0.0
jax: 0.9.2
platform: TPU V7X
devices: 8 x tpu
```

> **Note**: a source install of `tpu_inference` shows `0.0.0`; this is normal. On a GCE VM, `tpu_info` may report a 404 error (`Unable to poll TPU GCE Metadata`); this does not affect functionality.

> **About the JAX version**: vLLM's `requirements/tpu.txt` installs JAX 0.8.0, but TPU v7x requires JAX 0.9.2 + libtpu 0.0.39. After installing vLLM, you must manually override-install the correct versions.
>
> **About `--no-deps`**: tpu-inference's `pyproject.toml` depends on jax/jaxlib; without `--no-deps`, it triggers a JAX downgrade again.

### 3.5 Set runtime environment variables

```bash
export HF_TOKEN=${HF_TOKEN}
export JAX_PLATFORMS=tpu,cpu
export TPU_BACKEND_TYPE=jax
export PJRT_DEVICE=TPU
export MODEL_IMPL_TYPE=vllm
export USE_MOE_EP_KERNEL=0
export USE_BATCHED_RPA_KERNEL=0
```

### 3.6 Expand /dev/shm and copy the model from GCS

The model weights are stored uniformly in GCS. After starting the VM each time, copy them to `/dev/shm` (an in-memory filesystem) for the fastest load speed.

```bash
# Expand /dev/shm (default ~472 GB; needs to hold the 378 GiB model + vLLM IPC)
sudo mount -o remount,size=600G /dev/shm

# Copy the model weights from GCS to /dev/shm
gcloud storage cp -r ${MODEL_BUCKET}/${MODEL_NAME} /dev/shm/

# Verify
ls /dev/shm/${MODEL_NAME}/*.safetensors | wc -l   # should be 94
du -sh /dev/shm/${MODEL_NAME}                      # should be ~378 GiB
```

> **Why /dev/shm**: an in-memory filesystem reads at ~50 GB/s, 20× faster than Hyperdisk ML (2.4 GB/s), reducing model load time from ~3.5 min to ~10 seconds.
>
> **RAM size considerations**: by default `tpu7x-standard-4t` has 944 GiB RAM. After expanding `/dev/shm` to 600G, the 378 GiB model + vLLM runtime memory is about 400 GiB, leaving ~544 GiB for the system. If you hit OOM, switch to the Hyperdisk ML data disk (option B) or the boot disk (option C).
>
> **Alternative option B** (model on the data disk):
> ```bash
> gcloud storage cp -r ${MODEL_BUCKET}/${MODEL_NAME} /mnt/data/
> export MODEL_DIR=/mnt/data/${MODEL_NAME}
> ```
>
> **Alternative option C** (no data disk, use the boot disk):
> ```bash
> mkdir -p ~/models/${MODEL_NAME}
> gcloud storage cp -r ${MODEL_BUCKET}/${MODEL_NAME}/* ~/models/${MODEL_NAME}/
> export MODEL_DIR=~/models/${MODEL_NAME}
> ```
>
> The default option A model path is `/dev/shm/${MODEL_NAME}`; subsequent steps use this as the example. If you use option B/C, substitute the corresponding path.
>
> **First-time upload of the model to GCS**: if the GCS bucket does not yet have the model weights, first download them from HuggingFace on any machine and then upload:
> ```bash
> pip install -U "huggingface_hub[hf_transfer]"
> HF_HUB_ENABLE_HF_TRANSFER=1 huggingface-cli download \
>   Qwen/Qwen3.5-397B-A17B-FP8 --local-dir /tmp/Qwen3.5-397B-A17B-FP8
> gcloud storage cp -r /tmp/Qwen3.5-397B-A17B-FP8 ${MODEL_BUCKET}/
> ```

---

## Step 4: Verify patches (PR #2366 + PR #2577)

The Qwen3.5 hybrid GDN+Attention model requires two patches to take effect together; neither can be missing:

| Patch | Target | Symptoms without the patch |
|-------|----------|-------------------|
| **PR #2366** | `runner/kv_cache_manager.py` — hybrid KV cache page size | gibberish output / OOM / EngineCore crash |
| **PR #2577** | `kernels/gdn/recurrent_scan_v2.py` — GDN scan bf16→fp32 upcast | chat infinite loop / `about about about...` / `\n/**\n/**` |

An install from main branch ≥ 2026-05-15 should already include both fixes.

### Verify PR #2366 (KV cache)

```bash
KCM_PATH=$(python3 -c "import tpu_inference; import os; print(os.path.join(os.path.dirname(tpu_inference.__file__), 'runner', 'kv_cache_manager.py'))" 2>/dev/null | tail -1)
grep -c '_hybrid_uniform_page_size_bytes' "$KCM_PATH"
# output 7 = already included, skip the patch steps below
# output 0 = manual patch required
```

If the output is 0, manually download the fixed version from GitHub main:

```bash
curl -sf https://raw.githubusercontent.com/vllm-project/tpu-inference/main/tpu_inference/runner/kv_cache_manager.py \
  -o /tmp/kv_cache_manager_patched.py
grep -c '_hybrid_uniform_page_size_bytes' /tmp/kv_cache_manager_patched.py   # should output 7
cp $KCM_PATH ${KCM_PATH}.bak
cp /tmp/kv_cache_manager_patched.py $KCM_PATH
```

> **Why the patch is required**: the vLLM hybrid allocator shares 1 `KVCacheTensor` across 4 layers (a GPU byte-level optimization), but TPU `jax.Array` is strongly typed and must duplicate per-layer. Without the patch → the vLLM scheduler's block_id pool is ~3.5× larger than the TPU's actual capacity → block_id out of bounds → multi-request state collapse → **gibberish output / OOM / EngineCore crash**.

### Verify PR #2577 (GDN fp32 upcast)

```bash
RS_PATH=$(python3 -c "import tpu_inference; import os; print(os.path.join(os.path.dirname(tpu_inference.__file__), 'kernels', 'gdn', 'recurrent_scan_v2.py'))" 2>/dev/null | tail -1)
grep -c 'jnp.float32' "$RS_PATH"
# output ≥ 5 = PR #2577 already included
# output 0 = upgrade the image / reinstall vllm + tpu-inference (not recommended to manually patch a single file, because PR #2577 also changed gdn_attention.py's chunk_size)

# Also verify chunk_size = 32
GA_PATH=$(python3 -c "import tpu_inference; import os; print(os.path.join(os.path.dirname(tpu_inference.__file__), 'layers', 'common', 'gdn_attention.py'))" 2>/dev/null | tail -1)
grep -n 'chunk_size' "$GA_PATH" | head -5
# should show chunk_size=32 (before PR #2577 it was 64)
```

> **Why the patch is required**: the GDN (Gated Linear Attention) recurrent scan kernel is numerically unstable at bf16 precision; during consecutive token generation, accumulated error causes the output to degenerate into a repeated pattern (chat infinite loop). PR #2577 upcasts the internal scan computation to `jnp.float32` (5 places) and lowers chunk_size from 64 to 32 to further reduce accumulated error. **Both the TP=8 and DP attention modes benefit.**

---

## Step 5: Launch vLLM (about 7-10 min)

> **Important**: you must `cd /tmp` before running vLLM, otherwise the `~/vllm/` directory will be treated by Python as a namespace package, causing import errors.

```bash
source ~/vllm_env/bin/activate
cd /tmp && mkdir -p /tmp/vllm-logs

nohup env \
  PJRT_DEVICE=TPU TPU_BACKEND_TYPE=jax JAX_PLATFORMS=tpu,cpu \
  SKIP_JAX_PRECOMPILE=1 VLLM_XLA_CHECK_RECOMPILATION=0 \
  MODEL_IMPL_TYPE=vllm HF_HUB_OFFLINE=1 \
  vllm serve /dev/shm/Qwen3.5-397B-A17B-FP8 \
    --served-model-name Qwen3.5-397B-FP8 \
    --seed 42 \
    --max-model-len 4096 \
    --max-num-batched-tokens 4096 \
    --max-num-seqs 256 \
    --no-enable-prefix-caching \
    --tensor-parallel-size 8 \
    --enable-expert-parallel \
    --kv-cache-dtype fp8 \
    --block-size 256 \
    --gpu-memory-utilization 0.9 \
    --async-scheduling \
    --reasoning-parser qwen3 \
    --limit-mm-per-prompt '{"image":0,"video":0}' \
    --trust-remote-code \
    --port 8000 --host 0.0.0.0 \
  > /tmp/vllm-logs/serve.log 2>&1 &

# Wait until ready (about 7-10 min)
until curl -s -o /dev/null -w '%{http_code}\n' http://localhost:8000/health | grep -q 200; do
  date; tail -1 /tmp/vllm-logs/serve.log 2>/dev/null; sleep 30
done
echo "Server ready"
```

**Wait to see the key logs (markers of PR #2366 + completed startup)**:
```
Hybrid KV cache: padding every layer spec to 23289856 bytes     <- PR #2366 padding
regular_attn_shape=(num_blocks, (1280, 8, 4, 256))              <- block_size 1280 (wrongly 4352 before the patch)
num_gpu_blocks_override=945
INFO: Application startup complete.
```

> **Key parameter notes (vs Qwen3-Coder-480B)**:
>
> | Parameter | Qwen3.5 | Qwen3-Coder | Reason |
> |------|---------|-------------|------|
> | `--max-model-len` | `4096` | `10240` | single-host KV cache capacity limit |
> | `--max-num-batched-tokens` | `4096` | `8192` | CI accuracy test default |
> | `--max-num-seqs` | `256` | `512` | hybrid model scheduler capacity |
> | `--block-size` | `256` | default | CI default |
> | `--reasoning-parser` | `qwen3` | none | parses the `<think>` tag |
> | `--limit-mm-per-prompt` | `'{"image":0,"video":0}'` | none | skips the vision encoder |

---

## Step 6: Verify inference (chat test)

> **After PR #2577 the chat path is stable** (see ["Known Critical Limitations"](#已知关键限制历史-bug已修复) at the top of the document). Two verification methods are provided below:
> - **A. Single-prompt chat test** (recommended; closest to real usage scenarios)
> - **B. 5-shot Q/A pattern** (the baseline template for few-shot evals such as GSM8K)

### A. Single-prompt chat test (recommended)

```bash
curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen3.5-397B-FP8",
    "messages": [{"role": "user", "content": "The capital of France is"}],
    "max_tokens": 50,
    "temperature": 0
  }' | python3 -c "
import json, sys
d = json.load(sys.stdin)
m = d['choices'][0]
print('content:', repr(m['message']['content']))
print('finish:', m['finish_reason'])
"
```

**Expected**: `content` should contain the word "Paris" (e.g. `'The capital of France is **Paris**.'`) / `finish: stop`

> ⚠️ If the output is still an `about about about...` or `\n/**\n/**` infinite loop, your image/code commit is < `04077875` (2026-05-15), and you **must upgrade** (see ["How to upgrade"](#已知关键限制历史-bug已修复)).

### B. 5-shot Q/A pattern (GSM8K eval baseline)

```bash
python3 -c "
import requests, json

SHOTS = ('Question: Capital of Japan?\nAnswer: Tokyo.\n\n'
         'Question: Capital of Germany?\nAnswer: Berlin.\n\n'
         'Question: Capital of Italy?\nAnswer: Rome.\n\n'
         'Question: Capital of Spain?\nAnswer: Madrid.\n\n'
         'Question: Capital of Brazil?\nAnswer: Brasilia.\n\n')

r = requests.post('http://localhost:8000/v1/chat/completions', json={
    'model': 'Qwen3.5-397B-FP8',
    'messages': [{'role': 'user', 'content': SHOTS + 'Question: Capital of France?\nAnswer:'}],
    'max_tokens': 50, 'temperature': 0,
    'chat_template_kwargs': {'enable_thinking': False}
})
data = r.json()
m = data['choices'][0]['message']
print('content:', repr(m['content']))
print('reasoning_len:', len(m.get('reasoning') or ''))
print('finish:', data['choices'][0]['finish_reason'])
"
```

**Expected**: `content: 'Paris.'` / `reasoning_len: 0` / `finish: stop` (the GSM8K 93.93% accuracy uses exactly this pattern)

---

## Step 7: Benchmark

> **Note**: the bare-metal TPU VM does not have PyTorch installed, so the `vllm bench serve` command is unavailable. Use the following Python script instead.

```bash
python3 << 'PYEOF'
import requests, time, concurrent.futures

URL = "http://localhost:8000/v1/chat/completions"
MODEL = "Qwen3.5-397B-FP8"
SHOTS = ("Question: Capital of Japan?\nAnswer: Tokyo.\n\n"
         "Question: Capital of Germany?\nAnswer: Berlin.\n\n"
         "Question: Capital of Italy?\nAnswer: Rome.\n\n"
         "Question: Capital of Spain?\nAnswer: Madrid.\n\n"
         "Question: Capital of Brazil?\nAnswer: Brasilia.\n\n")
BASE = "The quick brown fox jumps over the lazy dog. "

def make_prompt(target_tokens):
    return BASE * (target_tokens // 10)

def send_request(prompt, output_len, rid):
    t0 = time.time()
    r = requests.post(URL, json={
        "model": MODEL, "messages": [{"role": "user", "content": prompt}],
        "max_tokens": output_len, "temperature": 0.7, "ignore_eos": True,
        "chat_template_kwargs": {"enable_thinking": False}})
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

### Single-host performance measurements (TPU VM, v7x-8, TP=8, 2026-04-29)

> Test conditions: `tpu7x-standard-4t`, model on `/dev/shm`, `--gpu-memory-utilization=0.9`, `--max-model-len=4096`.
> Data taken from the second run with a warm XLA compilation cache (the first run includes compilation overhead and is excluded).

| Concurrency | Latency | Throughput | Per-user tok/s |
|---:|---:|---:|---:|
| P1 | 21.1 s | 48.6 tok/s | 48.6 |
| P4 | 22.5 s | 182.3 | 45.6 |
| P16 | 26.3 s | 622.5 | 39.0 |
| P64 | 47.3 s | 1383.2 | 21.8 |
| **P128** | 66.2 s | **1969.0 tok/s** | 15.7 |

> **Peak is at P128** (~1969 tok/s). Higher concurrency (P256) causes throughput to drop because of scheduler preempt jitter.
>
> **First-run compilation note**: each new (batch_size, seq_len) combination triggers an XLA compilation (1-2 min) the first time it appears, causing first-run P1 latency of ~118s. After the second run hits the compilation cache, it drops to 21s. Before benchmarking, it is recommended to first trigger compilation of all target shapes with warmup requests.

---

# Part 2: PD Disaggregation (1P1D)

> 2 TPU v7x-8 VMs: one runs Prefill (kv_producer), one runs Decode (kv_consumer), transferring the KV cache over the VPC internal network.
>
> **Architecture note**: PD disaggregation **does not require Ray**. The two vLLM instances run completely independently and transfer the KV cache directly P2P via TPUConnectorHMA (which supports hybrid GDN+Attention KV transfer); `toy_proxy_server.py` is responsible for request routing.

### Must-read Qwen3.5 PD differences (vs Qwen3-Coder PD)

| Item | Qwen3-Coder (pure attention) | **Qwen3.5 (hybrid GDN+Attn)** |
|---|---|---|
| `kv_connector` | `TPUConnector` | **`TPUConnectorHMA`** |
| `kv_connector_module_path` | `tpu_inference.distributed.tpu_connector` | **`tpu_inference.distributed.tpu_connector_hma`** |
| Hybrid KV cache manager flag | not needed | **required** `--no-disable-hybrid-kv-cache-manager` |
| HMA connector file | already in nightly | **manual deployment needed** (Step 4) |

**Why `--no-disable-hybrid-kv-cache-manager` is required**: when vLLM sees a `kv_transfer_config`, it disables the hybrid KV cache manager by default. But Qwen3.5's 60 layers (45 GDN + 15 Attn) cannot be unified → `ValueError: Hybrid KV cache manager is disabled but failed to convert the KV cache specs to one unified type` → EngineCore crash.

## Step 1: Create 2 VMs

Reusing the approach from Part 1, create the Prefill and Decode VMs:

```bash
# Prefill VM
gcloud compute disks create qwen35-data-prefill \
    --project=${PROJECT_ID} --zone=${ZONE} \
    --type=hyperdisk-ml --size=2TB --provisioned-throughput=2500

gcloud compute instances create qwen35-prefill \
    --project=${PROJECT_ID} --zone=${ZONE} \
    --machine-type=tpu7x-standard-4t \
    --network-interface=network=${VPC_NAME},subnet=${SUBNET_NAME} \
    --maintenance-policy=TERMINATE \
    --provisioning-model=RESERVATION_BOUND \
    --reservation-affinity=specific \
    --reservation=${RESERVATION_NAME} \
    --create-disk=auto-delete=yes,boot=yes,size=1000GB,type=hyperdisk-balanced,image=projects/ubuntu-os-accelerator-images/global/images/family/ubuntu-accel-2404-amd64-tpu-tpu7x \
    --disk=name=qwen35-data-prefill,device-name=data-disk,mode=rw,auto-delete=no \
    --no-shielded-secure-boot \
    --scopes=cloud-platform

# Decode VM
gcloud compute disks create qwen35-data-decode \
    --project=${PROJECT_ID} --zone=${ZONE} \
    --type=hyperdisk-ml --size=2TB --provisioned-throughput=2500

gcloud compute instances create qwen35-decode \
    --project=${PROJECT_ID} --zone=${ZONE} \
    --machine-type=tpu7x-standard-4t \
    --network-interface=network=${VPC_NAME},subnet=${SUBNET_NAME} \
    --maintenance-policy=TERMINATE \
    --provisioning-model=RESERVATION_BOUND \
    --reservation-affinity=specific \
    --reservation=${RESERVATION_NAME} \
    --create-disk=auto-delete=yes,boot=yes,size=1000GB,type=hyperdisk-balanced,image=projects/ubuntu-os-accelerator-images/global/images/family/ubuntu-accel-2404-amd64-tpu-tpu7x \
    --disk=name=qwen35-data-decode,device-name=data-disk,mode=rw,auto-delete=no \
    --no-shielded-secure-boot \
    --scopes=cloud-platform
```

> **Disk-saving option**: if you do not need read/write separation, you can use **1 Hyperdisk ML** set to `READ_ONLY_MANY` mode and mount it to both VMs (read-only), provided the model weights have been written ahead of time.
>
> **Alternative when there is no Hyperdisk ML quota**: if the Hyperdisk ML quota is insufficient, you can skip creating a data disk and store the model weights on the boot disk instead. Simply remove the `gcloud compute disks create` and `--disk=name=...` lines. Copy the model directly onto the boot disk (e.g. `/home/${USER}/`).

## Step 2: Run environment preparation on each of the two VMs

On each of the two VMs, perform:
1. Part 1 Step 2 (format and mount the data disk) — skip if there is no data disk
2. Part 1 Steps 3.1 ~ 3.5 (system configuration + install vLLM/tpu-inference + set environment variables)
3. Part 1 Step 4 (verify the PR #2366 patch)
4. Copy the model (same as Part 1 Step 3.6); the model storage location depends on your disk configuration:

```bash
# Option A (machine RAM ≥ 1.5 TiB + space in /dev/shm):
sudo mount -o remount,size=600G /dev/shm
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

> Regardless of the option, subsequent steps uniformly reference the model path via `${MODEL_DIR}`.

## Step 3: Get the internal IPs

```bash
PREFILL_IP=$(gcloud compute instances describe qwen35-prefill \
    --project=${PROJECT_ID} --zone=${ZONE} \
    --format="value(networkInterfaces[0].networkIP)")
DECODE_IP=$(gcloud compute instances describe qwen35-decode \
    --project=${PROJECT_ID} --zone=${ZONE} \
    --format="value(networkInterfaces[0].networkIP)")
echo "Prefill: ${PREFILL_IP}, Decode: ${DECODE_IP}"
```

## Step 4: Deploy the HMA connector (run on both VMs)

Qwen3.5 PD disaggregation requires `TPUConnectorHMA` (which supports hybrid GDN+Attention KV transfer). The current nightly image does not include this file, so it must be deployed manually from tpu-inference main.

```bash
source ~/vllm_env/bin/activate

# Get the tpu_inference install directory (2>/dev/null suppresses the metadata 404 logs)
TPI_DIR=$(python3 -c "import tpu_inference; import os; print(os.path.dirname(tpu_inference.__file__))" 2>/dev/null)
echo "tpu_inference directory: ${TPI_DIR}"

# Download the HMA connector from GitHub main
curl -sf https://raw.githubusercontent.com/vllm-project/tpu-inference/main/tpu_inference/distributed/tpu_connector_hma.py \
  -o ${TPI_DIR}/distributed/tpu_connector_hma.py

# Verify
grep -c 'TPUConnectorHMA' ${TPI_DIR}/distributed/tpu_connector_hma.py
# should output ≥18
```

> If `tpu-inference` is already on the latest main branch (after `git pull`), you can skip this step — the file is already at `distributed/tpu_connector_hma.py`.

## Step 5: Launch the Prefill instance

SSH to the Prefill VM:

```bash
source ~/vllm_env/bin/activate
cd /tmp && mkdir -p /tmp/vllm-logs

nohup env \
  PJRT_DEVICE=TPU TPU_BACKEND_TYPE=jax JAX_PLATFORMS=tpu,cpu \
  SKIP_JAX_PRECOMPILE=1 VLLM_XLA_CHECK_RECOMPILATION=0 \
  MODEL_IMPL_TYPE=vllm HF_HUB_OFFLINE=1 \
  vllm serve ${MODEL_DIR} \
    --served-model-name Qwen3.5-397B-FP8 \
    --seed 42 \
    --max-model-len 16384 \
    --max-num-batched-tokens 4096 \
    --max-num-seqs 256 \
    --no-enable-prefix-caching \
    --tensor-parallel-size 8 \
    --enable-expert-parallel \
    --kv-cache-dtype fp8 \
    --block-size 256 \
    --gpu-memory-utilization 0.70 \
    --async-scheduling \
    --no-disable-hybrid-kv-cache-manager \
    --reasoning-parser qwen3 \
    --limit-mm-per-prompt '{"image":0,"video":0}' \
    --trust-remote-code \
    --port 8000 --host 0.0.0.0 \
    --kv-transfer-config '{"kv_connector":"TPUConnectorHMA","kv_connector_module_path":"tpu_inference.distributed.tpu_connector_hma","kv_role":"kv_producer"}' \
  > /tmp/vllm-logs/prefill.log 2>&1 &
```

Key differences (vs Part 1 single-host): `--gpu-memory-utilization=0.70` (reserve 30% HBM for the KV transfer buffer), `kv_role=kv_producer`, `--max-model-len=16384` (PD supports longer context), `--no-disable-hybrid-kv-cache-manager` (required for hybrid model PD).

> Startup takes 8~12 minutes (model loading + MoE requantization + XLA compilation). Use `tail -f /tmp/vllm-logs/prefill.log` to watch progress; it is ready when you see `Application startup complete`.

## Step 6: Launch the Decode instance

SSH to the Decode VM:

```bash
source ~/vllm_env/bin/activate
cd /tmp && mkdir -p /tmp/vllm-logs

nohup env \
  PJRT_DEVICE=TPU TPU_BACKEND_TYPE=jax JAX_PLATFORMS=tpu,cpu \
  SKIP_JAX_PRECOMPILE=1 VLLM_XLA_CHECK_RECOMPILATION=0 \
  MODEL_IMPL_TYPE=vllm HF_HUB_OFFLINE=1 \
  vllm serve ${MODEL_DIR} \
    --served-model-name Qwen3.5-397B-FP8 \
    --seed 42 \
    --max-model-len 16384 \
    --max-num-batched-tokens 4096 \
    --max-num-seqs 256 \
    --no-enable-prefix-caching \
    --tensor-parallel-size 8 \
    --enable-expert-parallel \
    --kv-cache-dtype fp8 \
    --block-size 256 \
    --gpu-memory-utilization 0.90 \
    --async-scheduling \
    --no-disable-hybrid-kv-cache-manager \
    --reasoning-parser qwen3 \
    --limit-mm-per-prompt '{"image":0,"video":0}' \
    --trust-remote-code \
    --port 9000 --host 0.0.0.0 \
    --kv-transfer-config '{"kv_connector":"TPUConnectorHMA","kv_connector_module_path":"tpu_inference.distributed.tpu_connector_hma","kv_role":"kv_consumer"}' \
  > /tmp/vllm-logs/decode.log 2>&1 &
```

Key differences (vs Prefill): `--gpu-memory-utilization=0.90` (Decode does not need to reserve a transfer buffer), `kv_role=kv_consumer`, `port=9000`.

> Can be started at the same time as Prefill; both sides load the model independently. Likewise use `tail -f /tmp/vllm-logs/decode.log` to watch.

## Step 7: Verify both ends are ready + launch the Proxy

Perform all of the following operations on the Prefill VM:

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

Once both return model information, launch the proxy:

```bash
python3 ~/tpu-inference/examples/disagg/toy_proxy_server.py \
  --host 0.0.0.0 --port 7000 \
  --prefiller-hosts localhost --prefiller-ports 8000 \
  --decoder-hosts ${DECODE_IP} --decoder-ports 9000
```

> **Path note**: the proxy script is located at `tpu-inference/examples/disagg/` (not `tpu_inference/examples/`).
>
> **Note**: `${DECODE_IP}` uses the VPC internal IP (the `networkIP` obtained in Step 3), not the external IP. Make sure the two VMs are in the same VPC and that the firewall allows ports 8000/9000/7000.

After the proxy starts, first send a smoke test to verify the complete pipeline:

```bash
python3 -c "
import requests, json
SHOTS = ('Question: Capital of Japan?\nAnswer: Tokyo.\n\n'
         'Question: Capital of Germany?\nAnswer: Berlin.\n\n'
         'Question: Capital of Italy?\nAnswer: Rome.\n\n'
         'Question: Capital of Spain?\nAnswer: Madrid.\n\n'
         'Question: Capital of Brazil?\nAnswer: Brasilia.\n\n')
r = requests.post('http://localhost:7000/v1/chat/completions', json={
    'model': 'Qwen3.5-397B-FP8',
    'messages': [{'role': 'user', 'content': SHOTS + 'Question: Capital of France?\nAnswer:'}],
    'max_tokens': 50, 'temperature': 0,
    'chat_template_kwargs': {'enable_thinking': False}
})
data = r.json()
m = data['choices'][0]['message']
print('content:', repr(m['content']), '| finish:', data['choices'][0]['finish_reason'])
"
# Expected: content: '\n\nParis.' | finish: stop (output through the proxy carries a \n\n prefix)
```

> **First-request latency**: the first request triggers XLA compilation; Prefill and Decode each need about 2~3 minutes (about 5 minutes total). Subsequent requests hit the compilation cache and latency drops to the second level.

## Step 8: PD Disaggregation Benchmark

> **Note**: the bare-metal TPU VM does not have PyTorch installed, so the `vllm bench serve` command is unavailable. Use the following Python script instead (sending requests to proxy port 7000).

```bash
python3 << 'PYEOF'
import requests, time, concurrent.futures

URL = "http://localhost:7000/v1/chat/completions"
MODEL = "Qwen3.5-397B-FP8"
BASE = "The quick brown fox jumps over the lazy dog. "

def make_prompt(target_tokens):
    return BASE * (target_tokens // 10)

def send_request(prompt, output_len, rid):
    t0 = time.time()
    try:
        r = requests.post(URL, json={
            "model": MODEL, "messages": [{"role": "user", "content": prompt}],
            "max_tokens": output_len, "temperature": 0.7, "ignore_eos": True,
            "chat_template_kwargs": {"enable_thinking": False}}, timeout=1800)
        data = r.json()
        t1 = time.time()
        if "error" in data:
            return {"error": str(data.get("error",""))[:200]}
        u = data.get("usage", {})
        return {"prompt": u.get("prompt_tokens",0), "completion": u.get("completion_tokens",0),
                "time": t1-t0, "tps": u.get("completion_tokens",0)/(t1-t0)}
    except Exception as e:
        return {"error": str(e)[:200]}

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
bench(8192, 1024, 1, 3)
bench(8192, 1024, 4, 8)
bench(1024, 8192, 1, 3)
bench(1024, 8192, 4, 8)
PYEOF
```

### PD disaggregation performance measurements (TPU VM, 1P1D, v7x-8 × 2, 2026-04-29)

> Test conditions: Prefill `gpu-mem=0.70`, Decode `gpu-mem=0.90`, `--max-model-len=16384`.
> Data taken from the second run with a warm XLA compilation cache (the first run includes compilation overhead and is excluded).

| Config | Latency | Per-req tok/s | Agg tok/s | vs single-host |
|------|--------:|--------------:|----------:|--------:|
| P1K/D1K c=1 | 22.1 s | 46.3 | 46.3 | 0.95x |
| P1K/D1K c=4 | 24.3 s | 42.1 | 167.0 | 0.92x |
| P8K/D1K c=1 | 23.4 s | 43.8 | 43.8 | — |
| P8K/D1K c=4 | 27.1 s | 37.8 | 148.2 | — |
| P1K/D8K c=1 | 172.1 s | 47.6 | 47.6 | — |
| P1K/D8K c=4 | 181.5 s | 45.1 | 180.3 | — |

> **PD vs single-host**: in the P1K/D1K scenario, the per-request throughput of PD disaggregation is about 92-95% of single-host,
> the slight loss coming from KV cache network transfer (~349 MB per request via TPUConnectorHMA).
> The real advantage of PD disaggregation is that Prefill and Decode can scale independently and it supports a longer `max-model-len` (16384 vs 4096 single-host).
>
> **P8K long prompt**: Prefill processing 8K tokens only adds ~1s of latency (23.4s vs 22.1s), showing that TPU prefill computation is very efficient.
>
> **D8K long generation**: a single request generating 8192 tokens takes ~172s, yet per-request tok/s is slightly higher (47.6 vs 46.3),
> because the first-token latency is amortized. `timeout` must be set to ≥1800s.

---

# Part 3: Multi-node Inference (TP=16)

> 2 TPU v7x-8 VMs form a v7x-16 slice (8 chips, 16 devices), interconnected via high-speed ICI.
>
> **Note**: Qwen3.5 multi-host requires **3 patches** (2 more mrope bypass patches than single-host).

### Multi-host vs Single-host key differences

| # | Multi-host-specific fix | Consequence if not fixed |
|---|---|---|
| 1 | **`--max-num-batched-tokens=16384`** (≥ Qwen3.5 `max_tokens_per_mm_item`) | silent hang in init_device, worker SIGSEGV |
| 2 | **PR #2366 patch (kv_cache_manager.py)** | KV init AssertionError or OOM |
| 3 | **tpu_runner.py patch**: when `disable_mm_from_limits=True`, set `self.uses_mrope=False` | first request TypeError `Qwen3VL.get_mrope_input_positions()` |
| 4 | **persistent_batch_manager.py patch**: defensive None check for mrope fn | PersistentBatchManager calls None → TypeError |

## Step 1: Create the v7x-16 TPU Slice

A TPU7x multi-host slice must be created via the trio of **Workload Policy + Instance Template + MIG** to ensure physical ICI interconnect.
Creating two GCE VMs separately can only use DCN (data center network) and cannot obtain the high-speed ICI interconnect.

### 1.1 Create the Workload Policy

```bash
SLICE_NAME=qwen35-slice

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

> **Note**: `--instance-termination-action=DELETE` is a required parameter for RESERVATION_BOUND + MIG. The subnet must use the full path `projects/.../subnetworks/...`, because the instance template is a global resource and does not automatically infer the region.

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

### 1.4 Get the VM names and IPs

```bash
# The VM names created by the MIG carry a random suffix
MIG_VMS=$(gcloud compute instance-groups managed list-instances ${SLICE_NAME}-mig \
    --project=${PROJECT_ID} --zone=${ZONE} --format="value(name)")
echo "VMs: ${MIG_VMS}"

# Get each IP (in creation order; the first one is Host 0)
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

### 1.5 Verify the ICI interconnect

SSH to either VM and check the metadata to confirm the ICI slice configuration:

```bash
# List all instance attributes
curl -s -H "Metadata-Flavor: Google" \
  http://metadata.google.internal/computeMetadata/v1/instance/attributes/ | tr '\n' ' '
# Should include: accelerator-type  tpu-env  worker-id  worker-network-endpoints  etc.

# Verify the accelerator type
curl -s -H "Metadata-Flavor: Google" \
  http://metadata.google.internal/computeMetadata/v1/instance/attributes/accelerator-type
# Should output: v7x-16 (note: no "tpu" prefix)

# Verify worker-id (Host 0 = 0, Host 1 = 1)
curl -s -H "Metadata-Flavor: Google" \
  http://metadata.google.internal/computeMetadata/v1/instance/attributes/worker-id

# Verify the topology (included in the tpu-env YAML)
curl -s -H "Metadata-Flavor: Google" \
  http://metadata.google.internal/computeMetadata/v1/instance/attributes/tpu-env | grep TOPOLOGY
# Should output: TOPOLOGY: 2x2x2
```

## Step 2: Prepare the environment on both VMs + copy the model

On each of the two VMs, perform:
1. Part 1 Steps 3.1 ~ 3.5 (system configuration + install vLLM/tpu-inference + set environment variables)

Then copy the model to the boot disk (**do not use /dev/shm**, the Ray Object Store will conflict):

```bash
mkdir -p ~/models
gcloud storage cp -r ${MODEL_BUCKET}/${MODEL_NAME} ~/models/

# Verify
ls ~/models/${MODEL_NAME}/*.safetensors | wc -l   # should be 94
du -sh ~/models/${MODEL_NAME}                      # should be ~378 GiB
```

> **Note**: for multi-host, **do not** store the model in `/dev/shm`. Ray's Object Store by default occupies about 30-40% of `/dev/shm` and will conflict with the model files. Copy the model to the boot disk `~/models/` instead.

## Step 3: Apply the 3 patches (run on both VMs)

Multi-host requires 3 patches. Patch 1 (PR #2366) is already included in the latest main branch. Patches 2 and 3 are mrope bypasses, injected inline with `sed`; **do not replace the entire file** (the tpu-inference API version may differ, and whole-file replacement causes `TypeError: cannot unpack non-iterable ModelInterface object`).

```bash
source ~/vllm_env/bin/activate

# Get the tpu_inference install directory (2>/dev/null suppresses the metadata 404 logs)
TPI_DIR=$(python3 -c "import tpu_inference; import os; print(os.path.dirname(tpu_inference.__file__))" 2>/dev/null)
RUNNER_DIR=${TPI_DIR}/runner
echo "Runner dir: ${RUNNER_DIR}"

# === Patch 1: PR #2366 (kv_cache_manager.py) — verify it is already included ===
KV_COUNT=$(grep -c '_hybrid_uniform_page_size_bytes' ${RUNNER_DIR}/kv_cache_manager.py 2>/dev/null || echo 0)
echo "PR #2366 check: ${KV_COUNT} (expect 7)"
# If not 7, the tpu-inference version is too old and needs to be updated to main branch

# === Patch 2: tpu_runner.py — mrope bypass ===
# Inject after the disable_mm_from_limits check: set uses_mrope=False to avoid multi-host TypeError
if ! grep -q "PATCH" ${RUNNER_DIR}/tpu_runner.py 2>/dev/null; then
    sed -i '/and not disable_mm_from_limits)/a\
\
        # PATCH: disable mrope path when user explicitly disables mm via --limit-mm-per-prompt\
        # Otherwise update_states triggers mrope code with old API → TypeError on multi-host\
        if disable_mm_from_limits:\
            self.uses_mrope = False\
            self.get_mrope_input_positions_fn = None' ${RUNNER_DIR}/tpu_runner.py
    echo "Patch 2 applied: $(grep -c 'PATCH' ${RUNNER_DIR}/tpu_runner.py) (expect 1)"
else
    echo "Patch 2 already applied"
fi

# === Patch 3: persistent_batch_manager.py — mrope None guard ===
# Add a fn is not None check to the if self.uses_mrope condition
if ! grep -q "PATCH" ${RUNNER_DIR}/persistent_batch_manager.py 2>/dev/null; then
    sed -i 's/            if self.uses_mrope:/            if self.uses_mrope and get_mrope_input_positions_fn is not None:  # PATCH: guard against None fn/' \
        ${RUNNER_DIR}/persistent_batch_manager.py
    echo "Patch 3 applied: $(grep -c 'PATCH' ${RUNNER_DIR}/persistent_batch_manager.py) (expect 1)"
else
    echo "Patch 3 already applied"
fi

# Clean __pycache__
find ${TPI_DIR} -name '__pycache__' -type d -exec rm -rf {} + 2>/dev/null || true
echo "Done. Pycache cleaned."
```

> **Why use sed instead of whole-file replacement?** The return type of tpu-inference's `get_model()` changes between versions (the old version returns a tuple, the new version returns a `ModelInterface` object). The complete files in `scripts/multihost-patches/` may be incompatible with the version you installed. An inline sed patch only modifies the lines that need changing and is compatible with any version.

## Step 4: Set the TPU topology environment variables

### Host 0 (Ray Head)

```bash
source ~/vllm_env/bin/activate

# Base environment variables
export PJRT_DEVICE=TPU
export TPU_BACKEND_TYPE=jax
export JAX_PLATFORMS=
export MODEL_IMPL_TYPE=vllm
export USE_MOE_EP_KERNEL=0
export USE_BATCHED_RPA_KERNEL=0
export HF_HUB_OFFLINE=1
export SKIP_JAX_PRECOMPILE=1
export VLLM_XLA_CHECK_RECOMPILATION=0

# Ray executor env var propagation (by default only VLLM_/HF_/NCCL_ prefixes are passed; you must explicitly add TPU/PJRT/JAX)
export VLLM_RAY_EXTRA_ENV_VAR_PREFIXES_TO_COPY="TPU_,PJRT_,JAX_,SKIP_"

# Multi-host TPU topology variables (replace <HOST0_IP> and <HOST1_IP> with the internal IPs obtained in Step 1.4)
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
export MODEL_IMPL_TYPE=vllm
export USE_MOE_EP_KERNEL=0
export USE_BATCHED_RPA_KERNEL=0
export HF_HUB_OFFLINE=1
export SKIP_JAX_PRECOMPILE=1
export VLLM_XLA_CHECK_RECOMPILATION=0

# Ray executor env var propagation (same as Host 0)
export VLLM_RAY_EXTRA_ENV_VAR_PREFIXES_TO_COPY="TPU_,PJRT_,JAX_,SKIP_"

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
> - `JAX_PLATFORMS=` (empty) instead of the single-host `tpu,cpu` — for multi-host it must be empty so that `PJRT_DEVICE=TPU` controls device selection; otherwise JAX cannot correctly initialize the cross-node topology
> - `VLLM_RAY_EXTRA_ENV_VAR_PREFIXES_TO_COPY` — **required**: the vLLM Ray executor by default only propagates environment variables with the `VLLM_*`, `HF_*`, `NCCL_*` prefixes to worker nodes. Without setting this variable, the worker-side topology info such as `TPU_WORKER_HOSTNAMES`, `TPU_TOPOLOGY` is all lost, causing TPU device initialization to fail

## Step 5: Launch the Ray cluster + vLLM

### Host 0 (start the Ray Head first)

```bash
# Start the Ray Head (daemon mode)
# --object-store-memory limits the Ray plasma store to 100 GB to avoid filling /dev/shm
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
  --address=<HOST0_IP>:6379 \
  --node-ip-address=${VLLM_HOST_IP} \
  --resources='{"TPU": 4}' \
  --object-store-memory=107374182400
```

### Host 0 (after confirming the cluster is ready, launch vLLM)

```bash
# Confirm 2 nodes, 8 TPU, and a 100 GB object store on each node
ray status

# Launch vLLM (TP=16, Ray executor) — you must cd /tmp to avoid the namespace package issue
cd /tmp && mkdir -p /tmp/vllm-logs

nohup env \
  PJRT_DEVICE=TPU TPU_BACKEND_TYPE=jax JAX_PLATFORMS= \
  MODEL_IMPL_TYPE=vllm USE_MOE_EP_KERNEL=0 USE_BATCHED_RPA_KERNEL=0 \
  HF_HUB_OFFLINE=1 SKIP_JAX_PRECOMPILE=1 VLLM_XLA_CHECK_RECOMPILATION=0 \
  VLLM_RAY_EXTRA_ENV_VAR_PREFIXES_TO_COPY="TPU_,PJRT_,JAX_,SKIP_" \
  RAY_memory_monitor_refresh_ms=0 \
  vllm serve ~/models/Qwen3.5-397B-A17B-FP8 \
    --served-model-name Qwen3.5-397B-FP8 \
    --tensor-parallel-size 16 \
    --distributed-executor-backend ray \
    --max-model-len 16384 \
    --max-num-batched-tokens 16384 \
    --max-num-seqs 256 \
    --no-enable-prefix-caching \
    --kv-cache-dtype fp8 \
    --gpu-memory-utilization 0.9 \
    --enable-expert-parallel \
    --limit-mm-per-prompt '{"image":0,"video":0}' \
    --trust-remote-code \
    --host 0.0.0.0 --port 8000 \
  > /tmp/vllm-logs/serve.log 2>&1 &

# Wait until ready (about 11-30 min, including model loading + multiple rounds of XLA compilation)
until curl -s -o /dev/null -w '%{http_code}\n' http://localhost:8000/health | grep -q 200; do
  date; tail -1 /tmp/vllm-logs/serve.log 2>/dev/null; sleep 60
done
echo "Server ready"
```

> **Key parameter notes**:
>
> | Parameter | Effect |
> |------|------|
> | `--max-num-batched-tokens=16384` | **required** ≥ Qwen3.5's `max_tokens_per_mm_item`, otherwise init_device silent hangs + SIGSEGV |
> | `--object-store-memory=107374182400` | limits the Ray plasma store to 100 GB (default occupies 30% of /dev/shm ~280 GB, which squeezes the model and worker memory) |
> | `VLLM_RAY_EXTRA_ENV_VAR_PREFIXES_TO_COPY` | **required**: makes the Ray executor propagate `TPU_*`/`PJRT_*`/`JAX_*`/`SKIP_*` environment variables to worker nodes (default only passes `VLLM_*`/`HF_*`/`NCCL_*`) |
| `RAY_memory_monitor_refresh_ms=0` | disables the Ray OOM monitor (the RAM usage spike during model loading would trigger workers being killed) |
> | `~/models/...` instead of `/dev/shm/...` | model on the boot disk, to avoid tmpfs RAM double-counting causing OOM |
>
> **Note**: multi-host does not support `--async-scheduling` (a Ray executor limitation), nor does it use `--reasoning-parser` or `--block-size`.

## Step 6: Verify and Benchmark

Perform on Host 0:

### Smoke test (5-shot Q/A)

```bash
python3 -c "
import requests, json
SHOTS = ('Question: Capital of Japan?\nAnswer: Tokyo.\n\n'
         'Question: Capital of Germany?\nAnswer: Berlin.\n\n'
         'Question: Capital of Italy?\nAnswer: Rome.\n\n'
         'Question: Capital of Spain?\nAnswer: Madrid.\n\n'
         'Question: Capital of Brazil?\nAnswer: Brasilia.\n\n')
for country in ['France', 'Italy', 'Australia', 'Canada', 'Brazil']:
    r = requests.post('http://localhost:8000/v1/chat/completions', json={
        'model': 'Qwen3.5-397B-FP8',
        'messages': [{'role': 'user', 'content': SHOTS + f'Question: Capital of {country}?\nAnswer:'}],
        'max_tokens': 50, 'temperature': 0,
        'chat_template_kwargs': {'enable_thinking': False}
    })
    data = r.json()
    m = data['choices'][0]['message']
    print(f'{country}: {repr(m[\"content\"])} | {data[\"choices\"][0][\"finish_reason\"]}')
"
```

**Expected**: 5/5 all hit, all finish=stop
```
France: ' Paris.' | stop
Italy: ' Rome.' | stop
Australia: ' Canberra.' | stop
Canada: ' Ottawa.' | stop
Brazil: ' Brasilia.' | stop
```

### Benchmark

> **Note**: the bare-metal TPU VM does not have PyTorch installed, so the `vllm bench serve` command is unavailable. Use the following Python script instead.

```bash
python3 << 'PYEOF'
import requests, time, concurrent.futures

URL = "http://localhost:8000/v1/chat/completions"
MODEL = "Qwen3.5-397B-FP8"
BASE = "The quick brown fox jumps over the lazy dog. "

def make_prompt(target_tokens):
    return BASE * (target_tokens // 10)

def send_request(prompt, output_len, rid):
    t0 = time.time()
    r = requests.post(URL, json={
        "model": MODEL, "messages": [{"role": "user", "content": prompt}],
        "max_tokens": output_len, "temperature": 0.7, "ignore_eos": True,
        "chat_template_kwargs": {"enable_thinking": False}})
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

### Multi-host performance reference (v7x-16, TP=16, measured 2026-04-28)

**Startup time**: ~12 min (model loading ~8 min + first XLA compilation ~4 min)

> **XLA compilation note**: each new (batch_size, seq_len) combination triggers an XLA compilation (2-5 min) the first time it appears;
> subsequent requests with the same shape are not recompiled. Before benchmarking, it is recommended to first trigger compilation with warmup requests.

| Scenario | Per-req tok/s | Aggregate tok/s | Single-host reference | vs single-host |
|------|-------------:|----------------:|---------:|--------:|
| P1K/D1K c=1 | 35.5 | 35.5 | 48.6 | 0.73x |
| P1K/D1K c=4 | 33.1 | 132 | 182.3 | 0.72x |
| P1K/D1K c=8 | 32.0 | 256 | — | — |
| P1K/D1K c=16 | 30.3 | 485 | — | — |

> **Multi-host vs single-host**: multi-host (TP=16) per-request throughput is about ~72% of single-host (TP=8),
> due to cross-node ICI communication overhead. The advantage of multi-host is the larger KV cache capacity (1536 GB HBM),
> supporting `--max-model-len 16384` (single-host only 4096), suitable for long-context scenarios.

---

## Firewall Rules

PD disaggregation and multi-node inference require inter-VM internal communication. We recommend allowing internal all-port TCP (the TPUConnectorHMA KV transfer and ZMQ side-channel use dynamic ports):

```bash
gcloud compute firewall-rules create allow-vllm-internal \
    --project=${PROJECT_ID} \
    --network=${VPC_NAME} \
    --allow=tcp \
    --source-ranges=10.0.0.0/8 \
    --description="Allow all internal TCP for vLLM/Ray/TPU communication"
```

Main port reference:

| Port | Purpose |
|------|------|
| 6379 | Ray GCS server (multi-host) |
| 7000 | PD disaggregation proxy |
| 8000 | vLLM API (Prefill / single-host / Host 0) |
| 8471 | libtpu coordinator (multi-host ICI) |
| 9000 | vLLM API (Decode) |
| dynamic | TPUConnectorHMA KV transfer / ZMQ side-channel |

---

## Resource Cleanup

```bash
# Stop vLLM / Ray
pgrep -f 'vllm|EngineCore|ray' | xargs -r kill -9

# Delete the single-host VM + data disk
gcloud compute instances delete qwen35-vm-01 --project=${PROJECT_ID} --zone=${ZONE} --quiet
gcloud compute disks delete qwen35-data-01 --project=${PROJECT_ID} --zone=${ZONE} --quiet

# Delete the PD disaggregation VMs + data disks
gcloud compute instances delete qwen35-prefill qwen35-decode --project=${PROJECT_ID} --zone=${ZONE} --quiet
gcloud compute disks delete qwen35-data-prefill qwen35-data-decode --project=${PROJECT_ID} --zone=${ZONE} --quiet

# Delete the multi-host slice (MIG → Template → Workload Policy)
SLICE_NAME=qwen35-slice
gcloud compute instance-groups managed delete ${SLICE_NAME}-mig --project=${PROJECT_ID} --zone=${ZONE} --quiet
gcloud compute instance-templates delete ${SLICE_NAME}-it --project=${PROJECT_ID} --quiet
gcloud compute resource-policies delete ${SLICE_NAME}-wp --project=${PROJECT_ID} --region=${ZONE%-*} --quiet
```

---

## Troubleshooting

| Symptom | Root cause | Fix |
|---|---|---|
| **Garbled output / OOM / EngineCore silent crash under multiple concurrency** | Missing PR #2366 (KV cache state corruption) | Follow [Step 4](#step-4-验证-patch-pr-2366--pr-2577); the `_hybrid_uniform_page_size_bytes` grep should output 7 |
| **weight load 80s/shard (vs normal 2s)** | Insufficient `/dev/shm` residual RAM → vLLM skips auto-prefetch | Clean `/dev/shm`: `rm -rf /dev/shm/sem.* /dev/shm/wrk_*` |
| **`libtpu lockfile` / `TPU device busy`** | The last vLLM exited abnormally; an orphan process holds the TPU | `pgrep -f 'EngineCore\|vllm' \| xargs -r kill -9` + `rm -f /tmp/libtpu_lockfile` |
| **PD mode: `ValueError: Hybrid KV cache manager is disabled`** | Missing `--no-disable-hybrid-kv-cache-manager` | Add the flag ([must-read PD differences](#qwen35-pd-必读差异vs-qwen3-coder-pd)) |
| **PD mode: `ModuleNotFoundError: tpu_connector_hma`** | HMA connector not deployed | Follow [Part 2 Step 4](#step-4-部署-hma-connector两台-vm-都执行) |
| **Multi-host: `TypeError: ... mrope ...`** | Missing mrope bypass patches | Follow [Part 3 Step 3](#step-3-应用-3-个-patches两台-vm-都执行) |
| **Multi-host: init_device 14 min with no log + SIGSEGV** | `--max-num-batched-tokens` too small | Set it to `16384` (≥ `max_tokens_per_mm_item`) |
| **Chat output infinite loop / garbled languages** | GDN recurrent scan bf16 numerical instability (missing PR #2577) | Upgrade to commit ≥ `04077875` (2026-05-15) which includes PR #2577 ([Known Limitations](#已知关键限制历史-bug已修复)) |
| **Ray worker killed** | Ray OOM monitor false-kill | Set `RAY_memory_monitor_refresh_ms=0` |
| **Multi-host: worker reports `No TPU devices found` / topology variables not taking effect** | The vLLM Ray executor by default does not propagate `TPU_*`/`PJRT_*` prefixed environment variables to workers | Set `VLLM_RAY_EXTRA_ENV_VAR_PREFIXES_TO_COPY="TPU_,PJRT_,JAX_,SKIP_"` |
