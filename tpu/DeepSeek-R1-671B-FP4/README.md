# DeepSeek R1 671B FP4 Inference on TPU v7x-8

> 端到端指南：在单节点 TPU v7x-8 上运行 DeepSeek R1 671B（FP4 量化）推理，
> 包含环境搭建、权重缓存生成、FP4 转换、vLLM 服务启动、以及 GSM8K 准确性验证。

## 目录

- [硬件与软件要求](#硬件与软件要求)
- [模型概览](#模型概览)
- [Step 1: 创建 GKE TPU Pod](#step-1-创建-gke-tpu-pod)
- [Step 2: 下载模型权重](#step-2-下载模型权重)
- [Step 3: 首次启动 — 生成 FP8 MoE Cache](#step-3-首次启动--生成-fp8-moe-cache)
- [Step 4: FP8 → FP4 离线转换](#step-4-fp8--fp4-离线转换)
- [Step 5: 启动 vLLM 推理服务](#step-5-启动-vllm-推理服务)
- [Step 6: 验证推理](#step-6-验证推理)
- [Step 7: GSM8K 准确性测试](#step-7-gsm8k-准确性测试)
- [性能数据](#性能数据)
- [环境变量参考](#环境变量参考)
- [FAQ](#faq)

---

## 硬件与软件要求

### 硬件

| 项目 | 要求 |
|------|------|
| TPU | v7x-8（2x2x1 拓扑，4 chips = 8 devices） |
| HBM | 94.75 GB/device，总计 758 GB |
| 主机内存 | ≥920 GB（模型加载 + /dev/shm 缓存） |
| 存储 | Lustre 或 Persistent Disk，≥1.5 TB |

### 软件

| 组件 | 版本 |
|------|------|
| Docker 镜像 | 含 vLLM + tpu_inference 的 TPU 推理镜像 |
| tpu_inference | branch: `feature/moe-fp4-weight-cache` |
| vLLM | v0.19.x（TPU 支持） |
| Python | 3.10+ |
| JAX | TPU 版本（镜像内预装） |

---

## 模型概览

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

## Step 1: 创建 GKE TPU Pod

创建一个 v7x-8 TPU Pod，挂载 Lustre 或 PD 存储：

```bash
# 创建 TPU Pod（根据实际环境替换参数）
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

---

## Step 2: 下载模型权重

模型权重约 700 GB，建议下载到 Lustre 或大容量 PD：

```bash
# 安装 huggingface_hub（镜像内可能已有）
pip install huggingface_hub

# 下载 DeepSeek R1 到 Lustre
huggingface-cli download deepseek-ai/DeepSeek-R1 \
  --local-dir /lustre/models/DeepSeek-R1

# 验证文件完整性
ls /lustre/models/DeepSeek-R1/*.safetensors | wc -l
# 预期：163 个 safetensors 分片
```

设置模型路径环境变量（后续步骤都会用到）：

```bash
export MODEL=/lustre/models/DeepSeek-R1
```

---

## Step 3: 首次启动 — 生成 FP8 MoE Cache

首次启动 vLLM 时，系统会自动从 safetensors 提取 MoE 权重并生成 FP8 缓存。
这个过程较慢（~30-60 分钟），但只需执行一次。

```bash
# 设置缓存目录
export MOE_WEIGHT_CACHE_DIR=/lustre/moe-cache
export MOE_REQUANTIZE_WEIGHT_DTYPE=float8_e4m3fn
export NEW_MODEL_DESIGN=1

# 首次启动（会自动生成 FP8 cache）
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
# 预期输出：58

# 每个 layer 目录包含 5 个文件
ls $MOE_WEIGHT_CACHE_DIR/ep8_tp1_gmm_ep_fp8e4m3_bsNone/model_layers_3_mlp_experts/
# 预期：meta.json  w13_weight.npy  w13_weight_scale.npy  w2_weight.npy  w2_weight_scale.npy
```

> **注意**：FP8 cache 的子目录名（如 `ep8_tp1_gmm_ep_fp8e4m3_bsNone`）由 EP/TP/backend/dtype 配置
> 自动决定。如果你的并行策略不同，子目录名也会不同。

---

## Step 4: FP8 → FP4 离线转换

FP4 将 MoE 权重体积减半，使 671B 模型可以放入 v7x-8 的 HBM。

转换使用 `tpu-inference` 仓库中的脚本，纯 CPU 运算，无需 TPU：

```bash
# 确保 tpu_inference 代码在 feature/moe-fp4-weight-cache 分支
cd /workspace/tpu_inference
git checkout feature/moe-fp4-weight-cache

# 运行 FP8 → FP4 转换
python3 scripts/convert_fp8_to_fp4.py \
  --fp8-cache $MOE_WEIGHT_CACHE_DIR/ep8_tp1_gmm_ep_fp8e4m3_bsNone \
  --fp4-cache $MOE_WEIGHT_CACHE_DIR/ep8_tp1_gmm_ep_fp4e2m1_bsNone \
  --workers 10
```

预期输出：
```
Converting 58 layers (FP8 -> native FP4)
  Input:   /lustre/moe-cache/ep8_tp1_gmm_ep_fp8e4m3_bsNone
  Output:  /lustre/moe-cache/ep8_tp1_gmm_ep_fp4e2m1_bsNone
  Workers: 10
  FP4 range: [-6.0, 6.0]
  model_layers_3_mlp_experts: 42.1s
  model_layers_4_mlp_experts: 43.5s
  ...

Done: 58 layers in 2520.3s (42.0 min)
Output: 58/58 layer dirs in /lustre/moe-cache/ep8_tp1_gmm_ep_fp4e2m1_bsNone
```

验证 FP4 cache 的 `meta.json` 标记：
```bash
cat $MOE_WEIGHT_CACHE_DIR/ep8_tp1_gmm_ep_fp4e2m1_bsNone/model_layers_3_mlp_experts/meta.json \
  | python3 -m json.tool | grep -E "storage_format|weight_dtype"
# 预期输出：
#   "_storage_format": "native_fp4",
#   "w13_weight_dtype": "float4_e2m1fn",
#   "w2_weight_dtype": "float4_e2m1fn"
```

### （可选）预拷贝 FP4 Cache 到 /dev/shm

如果存储 I/O 是瓶颈（如 Lustre 随机读慢），可将 FP4 cache 拷贝到内存文件系统：

```bash
# FP4 cache 约 610 GB，需要 /dev/shm 有足够空间
time cp -r $MOE_WEIGHT_CACHE_DIR/ep8_tp1_gmm_ep_fp4e2m1_bsNone /dev/shm/
# 约 5-6 分钟

df -h /dev/shm
# 确认已占用 ~610 GB，剩余 ≥100 GB

# 后续启动时将 MOE_WEIGHT_CACHE_DIR 指向 /dev/shm
export MOE_WEIGHT_CACHE_DIR=/dev/shm
```

---

## Step 5: 启动 vLLM 推理服务

```bash
# 设置环境变量
export MODEL=/lustre/models/DeepSeek-R1
export MOE_WEIGHT_CACHE_DIR=/lustre/moe-cache          # 或 /dev/shm（如果做了 Step 4 可选步骤）
export MOE_REQUANTIZE_WEIGHT_DTYPE=float4_e2m1fn        # 使用 FP4
export NEW_MODEL_DESIGN=1

# 启动 vLLM
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

## Step 6: 验证推理

在另一个终端（或同一 Pod 新开 shell）发送测试请求：

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

## Step 7: GSM8K 准确性测试

GSM8K (Grade School Math 8K) 是评估数学推理能力的标准 benchmark。

### 方法 1：使用 tpu_inference 集成测试脚本

```bash
# 确保 vLLM 服务正在运行（Step 5）

# 安装 lm_eval（如果未安装）
pip install lm_eval

# 运行准确性测试
cd /workspace/tpu_inference
python -m pytest -rP \
  scripts/vllm/integration/test_accuracy.py::test_lm_eval_accuracy_v1_engine \
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

bash /workspace/tpu_inference/tests/e2e/benchmarking/test_accuracy.sh \
  -r /workspace
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
| MoE Cache → TPU（Lustre） | ~27:00 | 直接从 Lustre mmap 读取 |
| MoE Cache → TPU（/dev/shm） | **~1:16** | tmpfs zero-copy mmap |
| **总计（/dev/shm 加速）** | **~5:41** | **对比 Lustre 的 ~33 min，加速 5.8x** |

### MoE Cache 大小

| 格式 | 大小 | 说明 |
|------|------|------|
| FP8 cache | ~404 GB | 58 层 × ~7 GB/层 |
| FP4 cache | ~610 GB | 58 层 × ~10.5 GB/层（含 scale） |

> **注意**：FP4 cache 文件比 FP8 大，因为 FP4 原生存储需要额外的 scale 精度（FP32 scale），
> 而 FP8 cache 的 scale 是原始量化时就有的。但加载到 TPU HBM 后，FP4 权重占用的显存只有 FP8 的一半。

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
| `MOE_WEIGHT_CACHE_DIR` | MoE 权重缓存根目录 | `/lustre/moe-cache` |
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

### Q: /dev/shm 不够大怎么办？

直接从 Lustre 加载也可以工作，只是每层加载时间从 ~1.3s 变为 ~28s。
确保 Pod 的 `emptyDir.sizeLimit` 设置足够大（≥800Gi）。

### Q: 如何重建 Docker 镜像？

确保 `tpu_inference` 使用 `feature/moe-fp4-weight-cache` 分支：
```bash
cd /workspace/tpu_inference
git fetch origin
git checkout feature/moe-fp4-weight-cache
pip install -e .
```

无需重建 Docker 镜像，editable install 会直接使用本地代码。
