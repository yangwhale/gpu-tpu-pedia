# GLM-5.1 754B FP4 Inference on TPU v7x-8

> 端到端指南：在单节点 TPU v7x-8 上运行 GLM-5.1 754B（FP4 量化）推理。
> 新手按照步骤走即可完成全流程。
>
> **代码仓库**: https://github.com/yangwhale/tpu-inference (branch: `feature/glm51-inference`)
>
> **模型**: [zai-org/GLM-5.1-FP8](https://huggingface.co/zai-org/GLM-5.1-FP8)（142 safetensors, 705 GB）

## 🎯 关键性能（实测 2026-04-24, TPU v7x-8 4 chips, FP4）

| 操作点 | 并发 | Throughput | tok/s/chip | tok/s/user | TPOT |
|--------|-----|-----------|------------|-----------|------|
| 🚀 Max Throughput | 128 | **2,569 tok/s** | **642** | 20.1 | 45 ms |
| ⚖️ Balanced | 64 | 1,570 tok/s | 393 | 24.6 | 38 ms |
| 💨 Low Latency | 4 | 130 tok/s | 33 | 32.6 | 30 ms |

> **vs DeepSeek R1**: GLM-5.1 在 p16-p128 范围 **快 2.7-3.5×**（详见下方 Benchmark 章节）

---

## ⚠️ 三个必设环境变量

> **每次启动 vLLM 前必须设置，缺一不可！**

| 环境变量 | 值 | 漏设后果 |
|---------|-----|---------|
| `MOE_REQUANTIZE_WEIGHT_DTYPE=float4_e2m1fn` | **必须设** | 默认查找 FP8 cache → cache miss → **HBM OOM**（651 GB vs 94.75 GB/device） |
| `NEW_MODEL_DESIGN=1` | **必须设** | MLA 模型强制要求，不设直接报错退出 |
| `MOE_WEIGHT_CACHE_DIR=/dev/shm` | **必须设** | 找不到 FP4 cache，触发在线 requantization |

**`MOE_REQUANTIZE_WEIGHT_DTYPE` 是最容易遗漏也最致命的**：它控制 cache 子目录名。
不设时默认为 `fp8`，子目录变成 `ep8_tp1_gmm_ep_fp8e4m3_bsNone`，
而 FP4 cache 在 `ep8_tp1_gmm_ep_fp4e2m1_bsNone`，导致全部 cache miss → OOM。
且错误信息 (`CompileTimeHbmOom`) **不提示根因是环境变量**，排查极其困难。

---

## 硬件与模型概览

| 项目 | 要求 |
|------|------|
| TPU | v7x-8（2x2x1 拓扑，4 chips = 8 devices） |
| HBM | 94.75 GB/device，总计 758 GB |
| 主机内存 | ≥940 GB（模型加载 + /dev/shm 缓存） |
| 存储 | ≥2.0 TB（模型 ~705 GB + FP4 cache ~735 GB） |

| 模型参数 | 值 |
|---------|-----|
| 架构 | MoE（256 experts, top-8）+ MLA + DSA + MTP |
| 总参数量 | ~754B |
| 总层数 | 78（layer 0-77 标准 + layer 78 MTP） |
| MoE 层数 | 75（layer 3-77，MTP layer 78 推理时跳过） |
| Dense 前几层 | 3（layer 0-2） |
| 量化方案 | FP4 MoE experts + FP8 attention + BF16 non-MoE |
| FP4 MoE HBM | ~27.5 GB/device（EP=8 分片，220 GB total ÷ 8） |
| Non-MoE HBM | ~21 GB/device（replicate 在 8 个 device） |
| 总 HBM 占用 | **58.43 GB/device (61.6%)**，剩 36 GB 给 KV cache |

### 与 DeepSeek R1 的关键差异

| 参数 | DeepSeek R1 | GLM-5.1 |
|------|-------------|---------|
| hidden_size | 7168 | **6144** |
| num_hidden_layers | 61 | **78** |
| MoE 层数 | 58 | **76** |
| num_attention_heads | 128 | **64** |
| q_lora_rank | 1536 | **2048** |
| qk_nope_head_dim | 128 | **192** |
| v_head_dim | 128 | **256** |
| rope_theta | 10000 (YaRN) | **1000000** (无 YaRN) |
| rope_interleave | false | **true** |
| 总参数量 | ~671B | **~754B** |

> 量化策略、cache 生成流程、vLLM 启动方式与 DeepSeek R1 **完全一致**。

---

## Step 1: 创建 GKE TPU Pod

在已有 GKE 集群（需要 TPU v7x node pool）中创建 Pod：

```bash
kubectl apply -f - <<'EOF'
apiVersion: v1
kind: Pod
metadata:
  name: vllm-glm51
spec:
  containers:
  - name: main
    image: <YOUR_DOCKER_REGISTRY>/vllm-tpu:latest
    resources:
      limits:
        google.com/tpu: 8
    volumeMounts:
    - name: data
      mountPath: /data
    - name: dshm
      mountPath: /dev/shm
  volumes:
  - name: data
    persistentVolumeClaim:
      claimName: data-pvc      # Lustre PVC 或 Hyperdisk PVC
  - name: dshm
    emptyDir:
      medium: Memory
      sizeLimit: 800Gi         # /dev/shm 需要 ≥757 GB
  restartPolicy: Never
  nodeSelector:
    cloud.google.com/gke-tpu-topology: 2x2x1
    cloud.google.com/gke-tpu-accelerator: tpu-v7x-lite-podslice
EOF
```

> **存储需求**：模型 ~705 GB + FP4 cache ~735 GB ≈ 1.4 TB。推荐 Lustre PVC 或 Hyperdisk Extreme 4 TB。

等待 Pod Running 后进入：

```bash
kubectl exec -it vllm-glm51 -- bash
```

---

## Step 2: 准备代码

Docker 镜像中的 `tpu_inference` 是 editable install，切到 GLM 分支即可：

```bash
cd /workspace/tpu_inference
git fetch origin
git checkout feature/glm51-inference

# 验证
python3 -c "import tpu_inference; print('OK')"
```

---

## Step 3: 下载模型权重

```bash
# 安装下载工具（如果未安装）
pip install huggingface_hub

# 下载 GLM-5.1 FP8 量化模型（~705 GB）
huggingface-cli download zai-org/GLM-5.1-FP8 \
  --local-dir /data/models/GLM-5.1-FP8

# 验证
ls /data/models/GLM-5.1-FP8/*.safetensors | wc -l
# 预期：142
```

设置模型路径（后续步骤都会用到）：

```bash
export MODEL=/data/models/GLM-5.1-FP8
```

---

## Step 4: 生成 FP4 MoE Cache

> 推荐 **CPU 并行直转**：纯 numpy，不需要 TPU/JAX，12 workers 并行，75 层仅需 **~28 min**。

### 4a: 确保 /dev/shm 为空

```bash
# 查看 /dev/shm 占用
df -h /dev/shm
ls /dev/shm/

# ⚠️ 如果有旧 cache，必须清理！否则 12 workers 内存不够会 OOM kill 整个 Pod
rm -rf /dev/shm/*
```

### 4b: 运行 FP4 转换

```bash
# gen_fp4_cache_cpu_parallel.py 与本 README 同目录
# 如需下载：
# curl -LO https://raw.githubusercontent.com/yangwhale/gpu-tpu-pedia/main/tpu/tpu-inference/GLM-5.1-754B-FP4/gen_fp4_cache_cpu_parallel.py

python3 -u gen_fp4_cache_cpu_parallel.py \
  --model-dir $MODEL \
  --cache-dir /data/moe-cache/ep8_tp1_gmm_ep_fp4e2m1_bsNone \
  --workers 12

# 中断后可续跑（自动跳过已完成的层）
```

> **workers 数量**：根据可用 RAM 调整（每 worker 峰值 ~70 GB）。
> v7x-8 机器 944 GB RAM → 最多 12 workers。

### 4c: 提取 Non-MoE 权重

将散落在 142 个 safetensors 中的非 MoE 权重合并为单个文件，**加载从 4:26 → 21s**：

```bash
python3 extract_non_moe_weights.py \
  --model-dir $MODEL \
  --output /data/moe-cache/ep8_tp1_gmm_ep_fp4e2m1_bsNone/non_moe_weights.safetensors
```

### 4d: 验证

```bash
# 检查层数
ls /data/moe-cache/ep8_tp1_gmm_ep_fp4e2m1_bsNone/ | grep model_layers | wc -l
# 预期：75（layer 3-77，MTP layer 78 不需要）

# 检查 non-MoE 文件
ls -lh /data/moe-cache/ep8_tp1_gmm_ep_fp4e2m1_bsNone/non_moe_weights.safetensors
# 预期：~21 GB

# 检查 FP4 shape
python3 -c "
import numpy as np
d = '/data/moe-cache/ep8_tp1_gmm_ep_fp4e2m1_bsNone/model_layers_3_mlp_experts'
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
SRC=/data/moe-cache/ep8_tp1_gmm_ep_fp4e2m1_bsNone
DST=/dev/shm/ep8_tp1_gmm_ep_fp4e2m1_bsNone

mkdir -p $DST

# 拷贝 non-MoE 权重
cp $SRC/non_moe_weights.safetensors $DST/

# 并行拷贝 75 层 MoE cache（8 workers，~4 min）
ls -d $SRC/model_layers_* | xargs -P 8 -I {} cp -r {} $DST/

# 验证
ls $DST/ | grep model_layers | wc -l   # 预期：75（layer 3-77，MTP layer 78 不需要）
ls -lh $DST/non_moe_weights.safetensors  # 预期：~21 GB
df -h /dev/shm                           # 预期占用 ~757 GB
```

> **不要用 `cp -r` 单线程**！单线程 ~8 min，`xargs -P 8` 并行 ~4 min。
>
> **总占用**：FP4 cache ~735 GB + non-MoE 21 GB ≈ **757 GB**，/dev/shm 800 GB 够用。

---

## Step 6: 启动 vLLM 推理服务

> ⚠️ **再次提醒：三个环境变量缺一不可！**

```bash
# ⚠️ 三个必设环境变量
export MOE_REQUANTIZE_WEIGHT_DTYPE=float4_e2m1fn   # 控制 FP4 cache 查找，漏设 = OOM
export NEW_MODEL_DESIGN=1                           # MLA 模型必须
export MOE_WEIGHT_CACHE_DIR=/dev/shm                # 指向 cache 根目录

# ⚠️ 必须 cd 到非 tpu-inference 的目录，避免 Python namespace 冲突
cd /tmp

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

等待日志显示 `Application startup complete`。

> **启动时间**：
> - 未优化：**~10 min**（non-MoE 加载 6m29s，`jax.clear_caches()` 导致每个 tensor 重复编译）
> - 优化后：**~3-4 min**（注释掉 `weight_utils.py` 中的 `jax.clear_caches()`，详见下方[性能优化](#性能优化可选)）

### 参数说明

| 参数 | 实际含义 | 容易误解的点 |
|------|----------|-------------|
| `--tensor-parallel-size 8` | 总设备数 | **不是 TP=8**。实际 TP=1，EP=8（由 additional-config 控制） |
| `--quantization fp8` | vLLM 量化 schema 名 | **不是 FP8 推理**。MoE 的 FP4 由环境变量控制 |
| `expert_parallelism: 8` | EP=8 | 256 experts ÷ 8 = 每 device 32 experts |
| `tensor_parallelism: 1` | TP=1 | attention 权重用 replicate 代替切分 |

---

## Step 7: 验证推理

在**另一个终端**（`kubectl exec -it vllm-glm51 -- bash`）发送请求：

```bash
# 测试 1: 数学计算
curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/data/models/GLM-5.1-FP8",
    "messages": [{"role": "user", "content": "What is 2+3? Answer with just the number."}],
    "max_tokens": 256
  }' | python3 -m json.tool

# 测试 2: 中文对话
curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/data/models/GLM-5.1-FP8",
    "messages": [{"role": "user", "content": "你是谁？用一句话介绍自己。"}],
    "max_tokens": 128
  }' | python3 -c "import sys,json; r=json.load(sys.stdin); print(r['choices'][0]['message']['content'])"

# 健康检查
curl -s http://localhost:8000/health
# 预期：{"status":"ok"}
```

### 实测验证结果（2026-04-24, GKE E2E pod, v7x-8）

| 测试 | 结果 |
|------|------|
| 2+3 数学计算 | ✅ 正确回答 5（含思维链推理过程） |
| 中文自我介绍 | ✅ 自称 "Z.ai 创建的大语言模型"，输出流畅 |
| 英文逻辑推理 (60km/h × 2.5h) | ✅ 正确推理 150 km |
| 健康检查 /health | ✅ 正常 |
| HBM 占用 | 58.43/94.75 GiB per device（61.6%） |
| KV cache | 2,309,120 tokens |
| MoE cache | 75/75 层全部 hit（FP4） |

### 质量评测：GSM8K 数学推理（2026-04-24，全量 1319 题）

| 指标 | 值 |
|------|-----|
| **测试集** | GSM8K test (1,319 题) |
| **准确率 (flexible-extract)** | **87.49% ± 0.91%** |
| 评测工具 | lm-evaluation-harness 0.4.9.2 |
| Prompt | 0-shot CoT (gsm8k_cot_zeroshot) |
| 生成参数 | greedy (temp=0), max_gen_toks=3500 |
| 量化 | FP4 (E2M1) MoE + FP8 attention |
| 评测耗时 | ~20 min (concurrency=64) / ~32 min (concurrency=16) |

**输出格式说明**：GLM-5.1 是 thinking 模型，输出格式为 `<think>...</think> The answer is N`。
- ✅ flexible-extract（提取最后数字）：**87.49%** — 真实分数
- ❌ strict-match（要求 `#### N` 格式）：11.68% — 无效，GLM 不用此格式

**二次验证（v4，2026-04-24）**：
- 用 `--reasoning-parser glm45` 启动的新 vLLM + concurrency=16 重跑全量 1319 题
- flexible-extract 结果：**0.8749052312357847**（与 full run 字符级一致）→ 数字稳定收敛、可重现
- strict-match 降至 2.05%（reasoning-parser 改变 content 格式导致 regex 不匹配，不影响 flexible 提取）
- 高并发 (64) 在新 vLLM 上会触发 retry storm（v3 失败案例）；concurrency=16 是稳定 baseline

**Caveat：max_gen_toks 截断**：
- 11.4% 题目输出长度 > 3,000 chars（接近 max_gen_toks=3500 上限）
- max-model-len=4096 限制了 reasoning 完整展开空间
- 假设截断题目一半答错，"无截断"理论上限约 92%；要彻底验证需重启 vLLM 调大 max-model-len（成本 ~14 min cold start）

**对比**：DeepSeek R1 671B FP8（NVIDIA 官方测试）GSM8K 94.92%，GLM-5.1 754B FP4 与之差距约 7 个百分点，符合两个模型定位差异（R1 是 reasoning-first 旗舰，GLM-5.1 定位 agentic engineering — 在 SWE-Bench Pro / CyberGym / BrowseComp 上是开源 SOTA）。

---

## 常见问题排查

### OOM: `CompileTimeHbmOom: Used 651.83G of 94.75G hbm`

**原因**：`MOE_REQUANTIZE_WEIGHT_DTYPE` 未设置，默认查找 FP8 cache 目录 → cache miss → OOM。

**修复**：确认三个环境变量都设了：
```bash
echo $MOE_REQUANTIZE_WEIGHT_DTYPE  # 应为 float4_e2m1fn
echo $NEW_MODEL_DESIGN             # 应为 1
echo $MOE_WEIGHT_CACHE_DIR         # 应为 /dev/shm
```

### 报错: `MLA models require NEW_MODEL_DESIGN=1`

**修复**：`export NEW_MODEL_DESIGN=1`

### vLLM 卡死不动（0% CPU，线程全在 futex_wait）

**原因**：MoE prefetch deadlock。cache 从磁盘加载时 semaphore 死锁。

**修复**：确保 cache 在 `/dev/shm`（tmpfs），不要从磁盘加载。参见 Step 5。

### TPU device busy

**原因**：上一个 vLLM 进程的 EngineCore 子进程还活着。

**修复**：
```bash
# 找到并杀死所有相关进程
ps aux | grep python | grep -v grep
kill -9 <PIDs>

# 确认 TPU 设备释放
fuser /dev/vfio/*

# 清理 lockfile
rm -f /tmp/libtpu_lockfile /tmp/.vllm_ipc_*
```

### `/dev/shm` 中出现多个 cache 目录

如果同时存在 `ep8_tp1_gmm_ep_fp4e2m1_bsNone` 和 `ep8_tp1_gmm_ep_fp8e4m3_bsNone`：

```bash
# 删除不需要的 FP8 cache 残留
rm -rf /dev/shm/ep8_tp1_gmm_ep_fp8e4m3_bsNone

# 只保留 FP4 cache
ls /dev/shm/
# 应只有 ep8_tp1_gmm_ep_fp4e2m1_bsNone/
```

### FP4 cache 生成时 Pod 被 OOM Kill（exit 137）

**原因**：/dev/shm 已有旧数据，挤占了 worker 内存。

**修复**：生成前清空 /dev/shm：`rm -rf /dev/shm/*`

---

## 性能数据（实测 2026-04-24）

### 启动时间（cache + non-MoE 均在 /dev/shm）

| 阶段 | 未优化 | 优化后（预估） | 说明 |
|------|--------|--------------|------|
| JAX 初始化 + Abstract model | ~1m44s | ~1m44s | mesh 创建、77 层模型配置 |
| Non-MoE 权重加载 | **6m29s** | **~25-108s** | 未优化时 `jax.clear_caches()` 每个 tensor 重编译 |
| MoE Cache 从 /dev/shm 加载 | ~1m51s | ~1m51s | 75 层 mmap + device_put |
| Server init + warmup | ~25s | ~25s | KV cache + DPScheduler |
| **总启动时间** | **~10m44s** | **~4-6 min** | 优化项见下方 |

### HBM 占用（实测 GKE E2E pod）

| 组件 | 每 device | 备注 |
|------|----------:|------|
| MoE Expert（FP4, EP=8 分片） | ~27.5 GB | 220 GB total ÷ 8 devices |
| 非 MoE 权重（replicate） | ~21 GB | attention + embedding + dense |
| 系统开销（KV cache 元数据等） | ~10 GB | XLA scratch, runtime |
| **已占用** | **58.43 GB (61.6%)** | 实测值 |
| **可用 KV cache** | **~36 GB** | 94.75 − 58.43 |
| **支持 KV tokens** | **2,309,120** | 实测，max-model-len=4096 |
| HBM 总量 / device | 94.75 GB | 8 devices × 94.75 = 758 GB usable |

> **说明**：4 chips × 192 GB raw HBM = 768 GB，但每 device 暴露 94.75 GB（runtime 占用差额）。
> Non-MoE 权重在 8 个 device 上各保留一份完整副本（共 168 GB across 8 devices），不分片。

### FP4 Cache 生成

| 方式 | 耗时 | 说明 |
|------|------|------|
| **CPU 并行直转** ⭐ | **~28 min** | 纯 numpy，12 workers |
| Non-MoE 提取 | ~2 min | 2292 keys → 21 GB |
| 拷贝到 /dev/shm | ~4 min | xargs -P 8 并行 |

---

## 推理性能 Benchmark（实测 2026-04-24，TPU v7x-8）

> **测试工具**: EvalScope perf v1.6.0 &nbsp;|&nbsp; **方法**: 每个并发先跑 1 轮预热（discarded）再跑 1 轮 recording

### 1K input / 1K output（短对话场景）

| Concurrency | Output tok/s | tok/s/chip | TTFT (s) | TPOT (ms) | tok/s/user | Latency (s) |
|------------:|-------------:|-----------:|---------:|----------:|-----------:|------------:|
|           1 |         28.4 |        7.1 |    0.534 |        35 |       28.4 |       36.07 |
|           2 |         56.6 |       14.1 |    0.531 |        35 |       28.3 |       36.19 |
|           4 |        130.5 |       32.6 |    0.510 |        30 |       32.6 |       31.39 |
|           8 |        254.4 |       63.6 |    0.528 |        31 |       31.8 |       32.20 |
|          16 |        444.8 |      111.2 |    1.016 |        35 |       27.8 |       36.83 |
|          32 |        788.2 |      197.1 |    1.974 |        39 |       24.7 |       41.53 |
|          64 |    **1,570** |  **392.5** |    3.174 |        38 |       24.6 |       41.67 |
|     **128** | **2,569.1**  |  **642.3** |    4.870 |        45 |       20.1 |       50.89 |

**关键 Pareto 操作点：**

| 操作点 | 并发 | Throughput | tok/s/user | TPOT | 适用场景 |
|--------|-----|-----------|-----------|------|---------|
| 🚀 **Max Throughput** | 128 | 2,569 tok/s (642/chip) | 20.1 | 45 ms | 离线批处理 |
| ⚖️ **Balanced** | 64 | 1,570 tok/s (393/chip) | 24.6 | 38 ms | 中等负载在线服务 |
| 💨 **Low Latency** | 4 | 130 tok/s (33/chip) | 32.6 | 30 ms | 交互式对话 |

**关键发现：**

- ✅ **Throughput 线性扩展** — p1 → p128 throughput 增长 90×，对应并发增长 128×，scaling 效率 70%
- ✅ **未饱和** — p128 仍在线性区间，预期 p256/p512 还有提升空间
- ✅ **TTFT 可预测** — 即使 p128 也只有 4.87s，无 R1 在中等并发的 TTFT 异常
- ✅ **100% 成功率** — 所有 8 个并发级别全部 success rate 100%

### 与 DeepSeek R1 671B FP4 对比（同一 TPU v7x-8 硬件）

| Parallel | GLM-5.1 (tok/s/chip) | DeepSeek R1 (tok/s/chip) | GLM/R1 |
|---------:|--------------------:|-----------------------:|:------:|
|        1 |                 7.1 |                    9.4 |  0.76× |
|        4 |                32.6 |                   36.7 |  0.89× |
|       16 |               111.2 |                   41.2 | **2.70×** |
|       64 |               392.5 |                  113.3 | **3.46×** |
|      128 |               642.3 |                  230.7 | **2.78×** |

> **观察**：GLM-5.1 在中等并发 (p16-p128) 全面优于 R1，与 GLM 架构"更瘦长"（hidden 6144 vs 7168, 层数 78 vs 61）一致 — 单 token 计算量小、batch 调度效率高。R1 优势在 p256+ 的 throughput 上限（p2048 达 1,827 tok/s/chip）。

### 长上下文场景（待测）

| 场景 | 状态 |
|------|------|
| 8K input / 1K output | ⏳ 待测试（需重启 vLLM 调大 max-model-len 至 16384+） |
| 1K input / 8K output | ⏳ 待测试 |

---

## 性能优化（可选）

### 注释 `jax.clear_caches()` — Non-MoE 加载快 10x

`weight_utils.py` 中的 `jax.clear_caches()` 导致每个 tensor 的 `jax.device_put()` 重新编译 transfer program。
2292 个 non-MoE tensor 只有 ~25 种 unique shape，但每次都重新编译，白白浪费 6 分钟。

```bash
# 在 Pod 内执行
cd /workspace/tpu_inference
grep -n 'jax.clear_caches()' tpu_inference/models/jax/utils/weight_utils.py
# 注释掉所有出现的 jax.clear_caches() 行
sed -i 's/^        jax.clear_caches()/#        jax.clear_caches()/' \
  tpu_inference/models/jax/utils/weight_utils.py
```

> **效果**：Non-MoE 加载从 6m29s → ~25-108s（DeepSeek R1 实测 10x 加速）。
> **风险**：无。cache 存的是 compiled transfer programs，~25 种 shape × 几 KB ≈ 不到 1 MB。

详见 [DeepSeek R1 踩坑 #18](../DeepSeek-R1-671B-FP4/README.md#18-jaxclear_caches-性能-bug--注释一行快-10x)。

---

## 端到端流程总结

```
Step 1: 创建 Pod（kubectl apply）
    ↓
Step 2: 切代码分支（git checkout feature/glm51-inference）
    ↓
Step 3: 下载模型（huggingface-cli download，~705 GB）
    ↓
Step 4: 生成 FP4 cache（gen_fp4_cache_cpu_parallel.py，~28 min）
       + 提取 non-MoE 权重（extract_non_moe_weights.py，~2 min）
    ↓
Step 5: 拷贝 cache 到 /dev/shm（xargs -P 8，~4 min）
    ↓
Step 6: 启动 vLLM（⚠️ 设三个环境变量！）
    ↓
Step 7: curl 验证推理
```

> **首次部署总耗时**（不含模型下载）：FP4 生成 28 min + 提取 2 min + 拷贝 4 min + 启动 ~11 min ≈ **~45 min**
>
> **后续重启**：只需 Step 6-7（如果 /dev/shm 的 cache 还在），**~11 min**（优化后 ~4-6 min）。

---

## 环境变量参考

| 变量 | 说明 | 值 | 必填 |
|------|------|-----|------|
| `MOE_REQUANTIZE_WEIGHT_DTYPE` | MoE 目标量化类型，控制 cache 子目录名 | `float4_e2m1fn` | ⚠️ **必填** |
| `NEW_MODEL_DESIGN` | 启用 MLA 模型设计 | `1` | ⚠️ **必填** |
| `MOE_WEIGHT_CACHE_DIR` | MoE 权重缓存根目录 | `/dev/shm` | ⚠️ **必填** |
| `MOE_REQUANTIZE_BLOCK_SIZE` | 量化块大小 | （可选） | 可选 |

---

## 参考资料

| 资源 | 链接 |
|------|------|
| DeepSeek R1 FP4 推理指南（完整版） | [../DeepSeek-R1-671B-FP4/README.md](../DeepSeek-R1-671B-FP4/README.md) |
| GLM-5.1 HuggingFace 模型 | [zai-org/GLM-5.1-FP8](https://huggingface.co/zai-org/GLM-5.1-FP8) |
| tpu-inference 代码 | [github.com/yangwhale/tpu-inference](https://github.com/yangwhale/tpu-inference) branch: `feature/glm51-inference` |
