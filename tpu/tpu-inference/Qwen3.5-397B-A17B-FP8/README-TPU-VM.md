# Qwen3.5-397B-A17B-FP8 Inference on TPU v7x — TPU VM 版

> TPU VM 端到端部署指南：从创建 VM 到完成 benchmark。
>
> **模型**: [Qwen/Qwen3.5-397B-A17B-FP8](https://huggingface.co/Qwen/Qwen3.5-397B-A17B-FP8)（94 safetensors, ~378 GiB, FP8 native）
>
> **架构**: 397B 总参 / 17B 激活 / **hybrid GDN+Attention**（45 GDN + 15 Standard Attn） + 512 routed experts + FP8 native
>
> **代码仓库**: [vllm-project/tpu-inference](https://github.com/vllm-project/tpu-inference)（main branch ≥ 2026-04-23, 含 PR #2366）
>
> **模型存储**: `gs://aidc-tpu-data/models`（GCS 对象存储，模型权重统一存放于此）
>
> GKE 版见同目录 [README.md](README.md)。

---

### 已知关键限制

> 当前部署 **不适合 conversational chatbot**。
> - **Chat 路径 broken**: thinking OFF 输出语言错乱/死循环；thinking ON 解释/闲聊类问题 content 输出空 / `Thinking\n` 死循环
> - **唯一稳定路径**: 5-shot Q/A completion pattern + `enable_thinking:false`（GSM8K 93.93% 就是用这个）
> - **适合用例**: batch eval, structured generation, few-shot completion, code gen
> - 高 GSM8K accuracy ≠ chat ready — **不要被误导**

---

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
export MODEL_BUCKET=gs://aidc-tpu-data/models         # 模型权重 GCS 路径
export MODEL_NAME=Qwen3.5-397B-A17B-FP8               # 模型目录名（与 GCS 一致）
```

## 硬件要求

| 项目 | 单机 (Part 1 & 2) | 多节点 (Part 3) |
|------|-------------------|----------------|
| 机型 | `tpu7x-standard-4t` | `tpu7x-standard-4t` × 2 |
| TPU | v7x-8（4 chips, 8 devices） | v7x-16（8 chips, 16 devices） |
| HBM | 768 GB | 1,536 GB |
| 主机内存 | 944 GB | 944 GB × 2 |
| 启动盘 | 1 TB Hyperdisk Balanced | 1 TB × 2 |
| 数据盘 | ≥500 GB（模型 ~378 GiB） | 不需要（模型拷到 boot disk ~/models/） |

---

# Part 1: 单机推理

## Step 1: 创建数据盘和 VM

### 1.1 创建 Hyperdisk ML 数据盘

```bash
gcloud compute disks create qwen35-data-01 \
    --project=${PROJECT_ID} \
    --zone=${ZONE} \
    --type=hyperdisk-ml \
    --size=2TB \
    --provisioned-throughput=2500
```

### 1.2 创建 TPU VM

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

### 1.3 SSH 连接

```bash
VM_IP=$(gcloud compute instances describe qwen35-vm-01 \
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
export PATH="$HOME/.local/bin:$PATH"

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

> **注意**：`tpu_inference` 源码安装显示 `0.0.0`，这是正常的。GCE VM 上 `tpu_info` 可能报 404 错误（`Unable to poll TPU GCE Metadata`），不影响功能。

> **关于 JAX 版本**：vLLM 的 `requirements/tpu.txt` 会安装 JAX 0.8.0，但 TPU v7x 需要 JAX 0.9.2 + libtpu 0.0.39。安装 vLLM 后必须手动覆盖安装正确版本。
>
> **关于 `--no-deps`**：tpu-inference 的 `pyproject.toml` 依赖 jax/jaxlib，不加 `--no-deps` 会再次触发 JAX 降级。

### 3.5 设置运行时环境变量

```bash
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
# 扩容 /dev/shm（默认 ~472 GB，需要容纳 378 GiB 模型 + vLLM IPC）
sudo mount -o remount,size=600G /dev/shm

# 从 GCS 拷贝模型权重到 /dev/shm
gcloud storage cp -r ${MODEL_BUCKET}/${MODEL_NAME} /dev/shm/

# 验证
ls /dev/shm/${MODEL_NAME}/*.safetensors | wc -l   # 应为 94
du -sh /dev/shm/${MODEL_NAME}                      # 应为 ~378 GiB
```

> **为什么用 /dev/shm**：内存文件系统读取速度 ~50 GB/s，比 Hyperdisk ML（2.4 GB/s）快 20 倍，模型加载时间从 ~3.5 min 缩短到 ~10 秒。
>
> **RAM 大小判断**：默认 `tpu7x-standard-4t` 有 944 GiB RAM。`/dev/shm` 扩容到 600G 后，378 GiB 模型 + vLLM 运行时内存约 400 GiB，剩余 ~544 GiB 给系统。如果 OOM，改用 Hyperdisk ML 数据盘（方案 B）或 boot disk（方案 C）。
>
> **备选方案 B**（模型放数据盘）：
> ```bash
> gcloud storage cp -r ${MODEL_BUCKET}/${MODEL_NAME} /mnt/data/
> export MODEL_DIR=/mnt/data/${MODEL_NAME}
> ```
>
> **备选方案 C**（无数据盘，用 boot disk）：
> ```bash
> mkdir -p ~/models/${MODEL_NAME}
> gcloud storage cp -r ${MODEL_BUCKET}/${MODEL_NAME}/* ~/models/${MODEL_NAME}/
> export MODEL_DIR=~/models/${MODEL_NAME}
> ```
>
> 默认方案 A 的模型路径为 `/dev/shm/${MODEL_NAME}`，后续步骤以此为例。如用方案 B/C，替换相应路径。
>
> **首次上传模型到 GCS**：如果 GCS 桶里还没有模型权重，先在任意机器上从 HuggingFace 下载后上传：
> ```bash
> pip install -U "huggingface_hub[hf_transfer]"
> HF_HUB_ENABLE_HF_TRANSFER=1 huggingface-cli download \
>   Qwen/Qwen3.5-397B-A17B-FP8 --local-dir /tmp/Qwen3.5-397B-A17B-FP8
> gcloud storage cp -r /tmp/Qwen3.5-397B-A17B-FP8 ${MODEL_BUCKET}/
> ```

---

## Step 4: 验证 PR #2366 patch

Qwen3.5 是 hybrid GDN+Attention 模型，vLLM 的 hybrid KV cache allocator 有一个已知 bug（PR #2366 修复）。从 2026-04-23 之后的 main branch 安装应已包含此修复。

```bash
# 验证 PR #2366 已包含
KCM_PATH=$(python3 -c "import tpu_inference; import os; print(os.path.join(os.path.dirname(tpu_inference.__file__), 'runner', 'kv_cache_manager.py'))" 2>/dev/null | tail -1)
grep -c '_hybrid_uniform_page_size_bytes' "$KCM_PATH"
# 输出 7 = 已包含，跳过下面的 patch 步骤
# 输出 0 = 需要手动 patch
```

如果输出 0，手动从 GitHub main 下载修复版：

```bash
curl -sf https://raw.githubusercontent.com/vllm-project/tpu-inference/main/tpu_inference/runner/kv_cache_manager.py \
  -o /tmp/kv_cache_manager_patched.py
grep -c '_hybrid_uniform_page_size_bytes' /tmp/kv_cache_manager_patched.py   # 应输出 7
cp $KCM_PATH ${KCM_PATH}.bak
cp /tmp/kv_cache_manager_patched.py $KCM_PATH
```

> **为什么必须 patch**：vLLM hybrid allocator 把 4 layers 共享 1 个 `KVCacheTensor`（GPU byte-level 优化），但 TPU `jax.Array` strongly typed 必须 duplicate per-layer。不 patch → vLLM scheduler 的 block_id pool 比 TPU 实际容量大 ~3.5× → block_id 越界 → 多 request 状态塌陷 → **gibberish output / OOM / EngineCore crash**。

---

## Step 5: 启动 vLLM（约 7-10 min）

> **重要**：必须 `cd /tmp` 后再运行 vLLM，否则 `~/vllm/` 目录会被 Python 当作 namespace package，导致 import 错误。

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

# 等待就绪（约 7-10 min）
until curl -s -o /dev/null -w '%{http_code}\n' http://localhost:8000/health | grep -q 200; do
  date; tail -1 /tmp/vllm-logs/serve.log 2>/dev/null; sleep 30
done
echo "Server ready"
```

**等待看到关键 log（PR #2366 + 启动完成的标志）**：
```
Hybrid KV cache: padding every layer spec to 23289856 bytes     <- PR #2366 padding
regular_attn_shape=(num_blocks, (1280, 8, 4, 256))              <- block_size 1280 (patch 前是错的 4352)
num_gpu_blocks_override=945
INFO: Application startup complete.
```

> **关键参数说明（vs Qwen3-Coder-480B）**：
>
> | 参数 | Qwen3.5 | Qwen3-Coder | 原因 |
> |------|---------|-------------|------|
> | `--max-model-len` | `4096` | `10240` | 单机 KV cache 容量限制 |
> | `--max-num-batched-tokens` | `4096` | `8192` | CI accuracy test 默认值 |
> | `--max-num-seqs` | `256` | `512` | hybrid 模型 scheduler 容量 |
> | `--block-size` | `256` | 默认 | CI 默认值 |
> | `--reasoning-parser` | `qwen3` | 无 | 解析 `<think>` tag |
> | `--limit-mm-per-prompt` | `'{"image":0,"video":0}'` | 无 | 跳过 vision encoder |

---

## Step 6: 验证推理（5-shot Q/A）

> **重要**：Qwen3.5 的 chat 路径不稳定。必须使用 5-shot Q/A pattern + `enable_thinking:false` 验证（见文档顶部"已知关键限制"）。

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

**预期**：`content: 'Paris.'` / `reasoning_len: 0` / `finish: stop`

---

## Step 7: Benchmark

> **注意**：TPU VM 裸机没有安装 PyTorch，`vllm bench serve` 命令不可用。使用以下 Python 脚本替代。

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

### 单机性能实测（TPU VM, v7x-8, TP=8, 2026-04-29）

> 测试条件：`tpu7x-standard-4t`，模型在 `/dev/shm`，`--gpu-memory-utilization=0.9`，`--max-model-len=4096`。
> 数据取自 XLA 编译缓存已热的第二轮运行（首轮含编译开销，不计入）。

| 并发 | Latency | Throughput | Per-user tok/s |
|---:|---:|---:|---:|
| P1 | 21.1 s | 48.6 tok/s | 48.6 |
| P4 | 22.5 s | 182.3 | 45.6 |
| P16 | 26.3 s | 622.5 | 39.0 |
| P64 | 47.3 s | 1383.2 | 21.8 |
| **P128** | 66.2 s | **1969.0 tok/s** | 15.7 |

> **Peak 在 P128**（~1969 tok/s）。更高并发（P256）因 scheduler preempt 抖动，吞吐反降。
>
> **首轮编译注意**：每种新的 (batch_size, seq_len) 组合首次出现时会触发 XLA 编译（1-2 min），导致首轮 P1 延迟 ~118s。第二轮命中编译缓存后降至 21s。Benchmark 前建议先用 warmup 请求触发所有目标 shape 的编译。

---

# Part 2: PD 分离 (1P1D)

> 2 台 TPU v7x-8 VM：一台跑 Prefill（kv_producer），一台跑 Decode（kv_consumer），通过 VPC 内网传输 KV cache。
>
> **架构说明**：PD 分离 **不需要 Ray**。两个 vLLM 实例完全独立运行，通过 TPUConnectorHMA（支持 hybrid GDN+Attention 的 KV transfer）直接 P2P 传输 KV cache，`toy_proxy_server.py` 负责请求路由。

### Qwen3.5 PD 必读差异（vs Qwen3-Coder PD）

| 项 | Qwen3-Coder (pure attention) | **Qwen3.5 (hybrid GDN+Attn)** |
|---|---|---|
| `kv_connector` | `TPUConnector` | **`TPUConnectorHMA`** |
| `kv_connector_module_path` | `tpu_inference.distributed.tpu_connector` | **`tpu_inference.distributed.tpu_connector_hma`** |
| Hybrid KV cache manager flag | 不需要 | **必须** `--no-disable-hybrid-kv-cache-manager` |
| HMA connector 文件 | nightly 已含 | **需手动部署**（Step 4） |

**为什么必须 `--no-disable-hybrid-kv-cache-manager`**：vLLM 看到 `kv_transfer_config` 时默认 disable hybrid KV cache manager。但 Qwen3.5 60 layer (45 GDN + 15 Attn) 无法 unify → `ValueError: Hybrid KV cache manager is disabled but failed to convert the KV cache specs to one unified type` → EngineCore 崩。

## Step 1: 创建 2 台 VM

复用 Part 1 的方式，创建 Prefill 和 Decode 两台 VM：

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

> **省盘方案**：如果不需要读写分离，可以用 **1 块 Hyperdisk ML** 设为 `READ_ONLY_MANY` 模式同时挂载到两台 VM（只读），前提是模型权重已提前写好。
>
> **无 Hyperdisk ML 配额时的备选方案**：如果 Hyperdisk ML 配额不足，可以不创建 data disk，改用启动盘存放模型权重。去掉 `gcloud compute disks create` 和 `--disk=name=...` 行即可。模型直接拷贝到启动盘上（如 `/home/${USER}/`）。

## Step 2: 两台 VM 分别执行环境准备

在两台 VM 上分别执行：
1. Part 1 Step 2（格式化挂载数据盘）—— 如果没有 data disk 则跳过
2. Part 1 Step 3.1 ~ 3.5（系统配置 + 安装 vLLM/tpu-inference + 设置环境变量）
3. Part 1 Step 4（验证 PR #2366 patch）
4. 拷贝模型（同 Part 1 Step 3.6），模型存放位置取决于你的磁盘配置：

```bash
# 方案 A（机器 RAM ≥ 1.5 TiB + /dev/shm 有空间）：
sudo mount -o remount,size=600G /dev/shm
gcloud storage cp -r ${MODEL_BUCKET}/${MODEL_NAME} /dev/shm/
export MODEL_DIR=/dev/shm/${MODEL_NAME}

# 方案 B（有 Hyperdisk ML data disk 挂载在 /mnt/data）：
# gcloud storage cp -r ${MODEL_BUCKET}/${MODEL_NAME} /mnt/data/
# export MODEL_DIR=/mnt/data/${MODEL_NAME}

# 方案 C（无 data disk，使用启动盘）：
# mkdir -p /home/${USER}/${MODEL_NAME}
# gcloud storage cp -r ${MODEL_BUCKET}/${MODEL_NAME}/* /home/${USER}/${MODEL_NAME}/
# export MODEL_DIR=/home/${USER}/${MODEL_NAME}
```

> 无论哪种方案，后续步骤统一用 `${MODEL_DIR}` 引用模型路径。

## Step 3: 获取内网 IP

```bash
PREFILL_IP=$(gcloud compute instances describe qwen35-prefill \
    --project=${PROJECT_ID} --zone=${ZONE} \
    --format="value(networkInterfaces[0].networkIP)")
DECODE_IP=$(gcloud compute instances describe qwen35-decode \
    --project=${PROJECT_ID} --zone=${ZONE} \
    --format="value(networkInterfaces[0].networkIP)")
echo "Prefill: ${PREFILL_IP}, Decode: ${DECODE_IP}"
```

## Step 4: 部署 HMA connector（两台 VM 都执行）

Qwen3.5 PD 分离需要 `TPUConnectorHMA`（支持 hybrid GDN+Attention KV transfer），当前 nightly 镜像不含此文件，需从 tpu-inference main 手动部署。

```bash
source ~/vllm_env/bin/activate

# 获取 tpu_inference 安装目录（2>/dev/null 抑制 metadata 404 日志）
TPI_DIR=$(python3 -c "import tpu_inference; import os; print(os.path.dirname(tpu_inference.__file__))" 2>/dev/null)
echo "tpu_inference 目录: ${TPI_DIR}"

# 从 GitHub main 下载 HMA connector
curl -sf https://raw.githubusercontent.com/vllm-project/tpu-inference/main/tpu_inference/distributed/tpu_connector_hma.py \
  -o ${TPI_DIR}/distributed/tpu_connector_hma.py

# 验证
grep -c 'TPUConnectorHMA' ${TPI_DIR}/distributed/tpu_connector_hma.py
# 应输出 ≥18
```

> 如果 `tpu-inference` 已经是最新的 main branch（`git pull` 后），这一步可以跳过——文件已在 `distributed/tpu_connector_hma.py`。

## Step 5: 启动 Prefill 实例

SSH 到 Prefill VM：

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

关键差异（vs Part 1 单机）：`--gpu-memory-utilization=0.70`（留 30% HBM 给 KV transfer buffer），`kv_role=kv_producer`，`--max-model-len=16384`（PD 支持更长 context），`--no-disable-hybrid-kv-cache-manager`（hybrid 模型 PD 必须）。

> 启动需 8~12 分钟（模型加载 + MoE requantization + XLA 编译）。用 `tail -f /tmp/vllm-logs/prefill.log` 观察进度，看到 `Application startup complete` 即就绪。

## Step 6: 启动 Decode 实例

SSH 到 Decode VM：

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

关键差异（vs Prefill）：`--gpu-memory-utilization=0.90`（Decode 不需要预留 transfer buffer），`kv_role=kv_consumer`，`port=9000`。

> 可以与 Prefill 同时启动，两边独立加载模型。同样用 `tail -f /tmp/vllm-logs/decode.log` 观察。

## Step 7: 验证两端就绪 + 启动 Proxy

在 Prefill VM 上执行以下所有操作：

```bash
source ~/vllm_env/bin/activate

# 设置 Decode VM 内网 IP（Step 3 获取的 networkIP）
export DECODE_IP=<decode-vm-internal-ip>
```

确认两个实例都已 ready：

```bash
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

> **路径说明**：proxy 脚本位于 `tpu-inference/examples/disagg/`（不是 `tpu_inference/examples/`）。
>
> **注意**：`${DECODE_IP}` 使用 VPC 内网 IP（Step 3 获取的 `networkIP`），不是外网 IP。确保两台 VM 在同一 VPC 且防火墙允许端口 8000/9000/7000。

Proxy 启动后，先发一个 smoke test 验证完整链路：

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
# 期望: content: '\n\nParis.' | finish: stop（通过 proxy 的输出会带 \n\n 前缀）
```

> **首次请求延迟**：第一次请求会触发 XLA 编译，Prefill 和 Decode 各需约 2~3 分钟（总计约 5 分钟）。后续请求命中编译缓存，延迟降到秒级。

## Step 8: PD 分离 Benchmark

> **注意**：TPU VM 裸机没有安装 PyTorch，`vllm bench serve` 命令不可用。使用以下 Python 脚本替代（对 proxy 端口 7000 发请求）。

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

### PD 分离性能实测（TPU VM, 1P1D, v7x-8 × 2, 2026-04-29）

> 测试条件：Prefill `gpu-mem=0.70`，Decode `gpu-mem=0.90`，`--max-model-len=16384`。
> 数据取自 XLA 编译缓存已热的第二轮运行（首轮含编译开销，不计入）。

| 配置 | Latency | Per-req tok/s | Agg tok/s | vs 单机 |
|------|--------:|--------------:|----------:|--------:|
| P1K/D1K c=1 | 22.1 s | 46.3 | 46.3 | 0.95x |
| P1K/D1K c=4 | 24.3 s | 42.1 | 167.0 | 0.92x |
| P8K/D1K c=1 | 23.4 s | 43.8 | 43.8 | — |
| P8K/D1K c=4 | 27.1 s | 37.8 | 148.2 | — |
| P1K/D8K c=1 | 172.1 s | 47.6 | 47.6 | — |
| P1K/D8K c=4 | 181.5 s | 45.1 | 180.3 | — |

> **PD vs 单机**：P1K/D1K 场景下 PD 分离的 per-request 吞吐约为单机的 92-95%，
> 轻微损耗来自 KV cache 网络传输（每请求 ~349 MB via TPUConnectorHMA）。
> PD 分离的真正优势在于：Prefill 和 Decode 可独立扩缩容，且支持更长的 `max-model-len`（16384 vs 单机 4096）。
>
> **P8K 长 prompt**：Prefill 处理 8K tokens 仅增加 ~1s 延迟（23.4s vs 22.1s），说明 TPU 的 prefill 计算非常高效。
>
> **D8K 长生成**：单请求生成 8192 tokens 耗时 ~172s，per-request tok/s 反而略高（47.6 vs 46.3），
> 因为首 token 延迟被摊薄。`timeout` 需设 ≥1800s。

---

# Part 3: 多节点推理 (TP=16)

> 2 台 TPU v7x-8 VM 组成 v7x-16 slice（8 chips, 16 devices），通过 ICI 高速互联。
>
> **注意**：Qwen3.5 multi-host 需要 **3 个 patches**（比单机多 2 个 mrope bypass patch）。

### Multi-host vs Single-host 关键差异

| # | Multi-host 特定 fix | 不修后果 |
|---|---|---|
| 1 | **`--max-num-batched-tokens=16384`** (≥ Qwen3.5 `max_tokens_per_mm_item`) | silent hang in init_device, worker SIGSEGV |
| 2 | **PR #2366 patch (kv_cache_manager.py)** | KV init AssertionError 或 OOM |
| 3 | **tpu_runner.py patch**: `disable_mm_from_limits=True` 时 set `self.uses_mrope=False` | first request TypeError `Qwen3VL.get_mrope_input_positions()` |
| 4 | **persistent_batch_manager.py patch**: defensive None check for mrope fn | PersistentBatchManager call None → TypeError |

## Step 1: 创建 v7x-16 TPU Slice

TPU7x multi-host slice 必须通过 **Workload Policy + Instance Template + MIG** 三件套创建，确保物理 ICI 互联。
单独创建两台 GCE VM 只能走 DCN（数据中心网络），无法获得 ICI 高速互联。

### 1.1 创建 Workload Policy

```bash
SLICE_NAME=qwen35-slice

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

### 1.4 获取 VM 名称和 IP

```bash
# MIG 创建的 VM 名称带随机后缀
MIG_VMS=$(gcloud compute instance-groups managed list-instances ${SLICE_NAME}-mig \
    --project=${PROJECT_ID} --zone=${ZONE} --format="value(name)")
echo "VMs: ${MIG_VMS}"

# 分别获取 IP（按创建顺序，第一台做 Host 0）
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

SSH 到任一 VM，检查 metadata 确认 ICI slice 配置：

```bash
# 列出所有 instance attributes
curl -s -H "Metadata-Flavor: Google" \
  http://metadata.google.internal/computeMetadata/v1/instance/attributes/ | tr '\n' ' '
# 应包含: accelerator-type  tpu-env  worker-id  worker-network-endpoints 等

# 验证 accelerator 类型
curl -s -H "Metadata-Flavor: Google" \
  http://metadata.google.internal/computeMetadata/v1/instance/attributes/accelerator-type
# 应输出: v7x-16（注意：不含 "tpu" 前缀）

# 验证 worker-id（Host 0 = 0, Host 1 = 1）
curl -s -H "Metadata-Flavor: Google" \
  http://metadata.google.internal/computeMetadata/v1/instance/attributes/worker-id

# 验证拓扑（包含在 tpu-env YAML 中）
curl -s -H "Metadata-Flavor: Google" \
  http://metadata.google.internal/computeMetadata/v1/instance/attributes/tpu-env | grep TOPOLOGY
# 应输出: TOPOLOGY: 2x2x2
```

## Step 2: 两台 VM 环境准备 + 模型拷贝

在两台 VM 上分别执行：
1. Part 1 Step 3.1 ~ 3.5（系统配置 + 安装 vLLM/tpu-inference + 设置环境变量）

然后拷贝模型到 boot disk（**不要用 /dev/shm**，Ray Object Store 会冲突）：

```bash
mkdir -p ~/models
gcloud storage cp -r ${MODEL_BUCKET}/${MODEL_NAME} ~/models/

# 验证
ls ~/models/${MODEL_NAME}/*.safetensors | wc -l   # 应为 94
du -sh ~/models/${MODEL_NAME}                      # 应为 ~378 GiB
```

> **注意**：multi-host **不要**用 `/dev/shm` 存模型。Ray 的 Object Store 默认占用 `/dev/shm` 约 30-40%，会与模型文件冲突。模型改拷到 boot disk `~/models/`。

## Step 3: 应用 3 个 patches（两台 VM 都执行）

Multi-host 需要 3 个 patch。Patch 1 (PR #2366) 在最新 main branch 已包含。Patch 2 和 3 是 mrope bypass，用 `sed` 内联注入，**不要替换整个文件**（tpu-inference API 版本可能不同，整文件替换会导致 `TypeError: cannot unpack non-iterable ModelInterface object`）。

```bash
source ~/vllm_env/bin/activate

# 获取 tpu_inference 安装目录（2>/dev/null 抑制 metadata 404 日志）
TPI_DIR=$(python3 -c "import tpu_inference; import os; print(os.path.dirname(tpu_inference.__file__))" 2>/dev/null)
RUNNER_DIR=${TPI_DIR}/runner
echo "Runner dir: ${RUNNER_DIR}"

# === Patch 1: PR #2366 (kv_cache_manager.py) — 验证是否已包含 ===
KV_COUNT=$(grep -c '_hybrid_uniform_page_size_bytes' ${RUNNER_DIR}/kv_cache_manager.py 2>/dev/null || echo 0)
echo "PR #2366 check: ${KV_COUNT} (expect 7)"
# 如果不是 7，说明 tpu-inference 版本太旧，需要更新到 main branch

# === Patch 2: tpu_runner.py — mrope bypass ===
# 在 disable_mm_from_limits 判断后注入：设 uses_mrope=False，避免 multi-host TypeError
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
# 在 if self.uses_mrope 条件中增加 fn is not None 检查
if ! grep -q "PATCH" ${RUNNER_DIR}/persistent_batch_manager.py 2>/dev/null; then
    sed -i 's/            if self.uses_mrope:/            if self.uses_mrope and get_mrope_input_positions_fn is not None:  # PATCH: guard against None fn/' \
        ${RUNNER_DIR}/persistent_batch_manager.py
    echo "Patch 3 applied: $(grep -c 'PATCH' ${RUNNER_DIR}/persistent_batch_manager.py) (expect 1)"
else
    echo "Patch 3 already applied"
fi

# 清理 __pycache__
find ${TPI_DIR} -name '__pycache__' -type d -exec rm -rf {} + 2>/dev/null || true
echo "Done. Pycache cleaned."
```

> **为什么用 sed 而不是整文件替换？** tpu-inference 的 `get_model()` 返回类型在不同版本间变化（旧版返回 tuple，新版返回 `ModelInterface` object）。`scripts/multihost-patches/` 中的完整文件可能与你安装的版本不兼容。sed 内联 patch 只修改需要改的行，与任何版本兼容。

## Step 4: 设置 TPU 拓扑环境变量

### Host 0（Ray Head）

```bash
source ~/vllm_env/bin/activate

# 基础环境变量
export PJRT_DEVICE=TPU
export TPU_BACKEND_TYPE=jax
export JAX_PLATFORMS=
export MODEL_IMPL_TYPE=vllm
export USE_MOE_EP_KERNEL=0
export USE_BATCHED_RPA_KERNEL=0
export HF_HUB_OFFLINE=1
export SKIP_JAX_PRECOMPILE=1
export VLLM_XLA_CHECK_RECOMPILATION=0

# Multi-host TPU 拓扑变量（替换 <HOST0_IP> 和 <HOST1_IP> 为 Step 1.4 获取的内网 IP）
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
export MODEL_IMPL_TYPE=vllm
export USE_MOE_EP_KERNEL=0
export USE_BATCHED_RPA_KERNEL=0
export HF_HUB_OFFLINE=1
export SKIP_JAX_PRECOMPILE=1
export VLLM_XLA_CHECK_RECOMPILATION=0

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
  --address=<HOST0_IP>:6379 \
  --node-ip-address=${VLLM_HOST_IP} \
  --resources='{"TPU": 4}' \
  --object-store-memory=107374182400
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

# 等待就绪（约 11-30 min，含模型加载 + 多轮 XLA 编译）
until curl -s -o /dev/null -w '%{http_code}\n' http://localhost:8000/health | grep -q 200; do
  date; tail -1 /tmp/vllm-logs/serve.log 2>/dev/null; sleep 60
done
echo "Server ready"
```

> **关键参数说明**：
>
> | 参数 | 作用 |
> |------|------|
> | `--max-num-batched-tokens=16384` | **必须** ≥ Qwen3.5 的 `max_tokens_per_mm_item`，否则 init_device silent hang + SIGSEGV |
> | `--object-store-memory=107374182400` | 限制 Ray plasma store 为 100 GB（默认占 /dev/shm 30%~280 GB，会挤占模型和 worker 内存） |
> | `RAY_memory_monitor_refresh_ms=0` | 禁用 Ray OOM monitor（模型加载期间 RAM 使用高峰会触发 worker 被 kill） |
> | `~/models/...` 而非 `/dev/shm/...` | 模型放 boot disk，避免 tmpfs RAM 双重计数导致 OOM |
>
> **注意**：multi-host 不支持 `--async-scheduling`（Ray executor 限制），也不使用 `--reasoning-parser`、`--block-size`。

## Step 6: 验证和 Benchmark

在 Host 0 上执行：

### Smoke test（5-shot Q/A）

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

**预期**：5/5 全 hit，全 finish=stop
```
France: ' Paris.' | stop
Italy: ' Rome.' | stop
Australia: ' Canberra.' | stop
Canada: ' Ottawa.' | stop
Brazil: ' Brasilia.' | stop
```

### Benchmark

> **注意**：TPU VM 裸机没有安装 PyTorch，`vllm bench serve` 命令不可用。使用以下 Python 脚本替代。

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

### Multi-host 性能参考（v7x-16, TP=16, 2026-04-28 实测）

**启动时间**：~12 min（模型加载 ~8 min + 首次 XLA 编译 ~4 min）

> **XLA 编译注意**：每种新的 (batch_size, seq_len) 组合首次出现时会触发 XLA 编译（2-5 min），
> 后续相同 shape 的请求不再编译。Benchmark 前建议先用 warmup 请求触发编译。

| 场景 | Per-req tok/s | Aggregate tok/s | 单机参考 | vs 单机 |
|------|-------------:|----------------:|---------:|--------:|
| P1K/D1K c=1 | 35.5 | 35.5 | 48.6 | 0.73x |
| P1K/D1K c=4 | 33.1 | 132 | 182.3 | 0.72x |
| P1K/D1K c=8 | 32.0 | 256 | — | — |
| P1K/D1K c=16 | 30.3 | 485 | — | — |

> **Multi-host vs 单机**：Multi-host (TP=16) 单请求吞吐约为单机 (TP=8) 的 ~72%，
> 因为 ICI 跨节点通信开销。Multi-host 优势在于更大的 KV cache 容量（1536 GB HBM），
> 支持 `--max-model-len 16384`（单机仅 4096），适合长上下文场景。

---

## 防火墙规则

PD 分离和多节点推理需要 VM 间内网通信。建议允许内网全端口 TCP（TPUConnectorHMA 的 KV transfer 和 ZMQ side-channel 使用动态端口）：

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
| 动态 | TPUConnectorHMA KV transfer / ZMQ side-channel |

---

## 资源清理

```bash
# 停止 vLLM / Ray
pgrep -f 'vllm|EngineCore|ray' | xargs -r kill -9

# 删除单机 VM + 数据盘
gcloud compute instances delete qwen35-vm-01 --project=${PROJECT_ID} --zone=${ZONE} --quiet
gcloud compute disks delete qwen35-data-01 --project=${PROJECT_ID} --zone=${ZONE} --quiet

# 删除 PD 分离 VM + 数据盘
gcloud compute instances delete qwen35-prefill qwen35-decode --project=${PROJECT_ID} --zone=${ZONE} --quiet
gcloud compute disks delete qwen35-data-prefill qwen35-data-decode --project=${PROJECT_ID} --zone=${ZONE} --quiet

# 删除 multi-host slice（MIG → Template → Workload Policy）
SLICE_NAME=qwen35-slice
gcloud compute instance-groups managed delete ${SLICE_NAME}-mig --project=${PROJECT_ID} --zone=${ZONE} --quiet
gcloud compute instance-templates delete ${SLICE_NAME}-it --project=${PROJECT_ID} --quiet
gcloud compute resource-policies delete ${SLICE_NAME}-wp --project=${PROJECT_ID} --region=${ZONE%-*} --quiet
```

---

## Troubleshooting

| 症状 | 根因 | 修复 |
|---|---|---|
| **多并发输出乱码 / OOM / EngineCore silent crash** | 缺 PR #2366 (KV cache 状态损坏) | 走 [Step 4](#step-4-验证-pr-2366-patch)，grep 应输出 7 |
| **weight load 80s/shard (vs 正常 2s)** | `/dev/shm` 残留 RAM 不足 → vLLM 跳过 auto-prefetch | 清 `/dev/shm`：`rm -rf /dev/shm/sem.* /dev/shm/wrk_*` |
| **`libtpu lockfile` / `TPU device busy`** | 上次 vLLM 异常退出，孤儿进程占 TPU | `pgrep -f 'EngineCore\|vllm' \| xargs -r kill -9` + `rm -f /tmp/libtpu_lockfile` |
| **PD 模式: `ValueError: Hybrid KV cache manager is disabled`** | 缺 `--no-disable-hybrid-kv-cache-manager` | 加 flag（[PD 必读差异](#qwen35-pd-必读差异vs-qwen3-coder-pd)） |
| **PD 模式: `ModuleNotFoundError: tpu_connector_hma`** | 未部署 HMA connector | 走 [Part 2 Step 4](#step-4-部署-hma-connector两台-vm-都执行) |
| **Multi-host: `TypeError: ... mrope ...`** | 缺 mrope bypass patches | 走 [Part 3 Step 3](#step-3-应用-3-个-patches两台-vm-都执行) |
| **Multi-host: init_device 14 min 无 log + SIGSEGV** | `--max-num-batched-tokens` 太小 | 设为 `16384`（≥ `max_tokens_per_mm_item`） |
| **Chat 输出死循环 / 语言错乱** | Qwen3.5 chat 路径 broken | 使用 5-shot Q/A pattern + `enable_thinking:false`（[已知限制](#已知关键限制)） |
| **Ray worker 被 kill** | Ray OOM monitor 误杀 | 设 `RAY_memory_monitor_refresh_ms=0` |
