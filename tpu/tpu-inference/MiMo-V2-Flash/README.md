# MiMo-V2-Flash (FP8) 推理 on TPU v7x-8

> 端到端指南：在单台 TPU v7x-8（8 devices / 4 chips / 768 GB HBM）上运行 MiMo-V2-Flash FP8 推理。
>
> **架构**：256 routed experts / 8 activated / Hybrid SWA（9 Full Attention + 39 SWA）/ FP8 原生量化
>
> **推理框架**: [sglang-jax](https://github.com/sgl-project/sglang-jax)（single-host, TP=8, DP=2, EP=8）
>
> **模型**: [XiaomiMiMo/MiMo-V2-Flash](https://huggingface.co/XiaomiMiMo/MiMo-V2-Flash)（HuggingFace，~20 GB/chip FP8）

---

## 硬件与资源

| 项 | 值 |
|---|---|
| TPU | 1× v7x-8（8 devices, single-host） |
| HBM | 96 GB/device × 8 = 768 GB |
| 模型显存 | ~20 GB/chip（FP8），单机轻松放下 |
| 推理框架 | sglang-jax |
| 并行策略 | TP=8, DP=2, EP=8, moe-backend=epmoe |
| 容器镜像 | `us-docker.pkg.dev/cloud-tpu-images/jax-ai-image/tpu:jax0.8.1-rev1` |

---

## 部署步骤

### Step 0: 创建 GKE Pod

提交一个 Job 让 node pool autoscaler 拉起 TPU 机器：

```yaml
# /tmp/tpu-e2e-pod.yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: e2e-tpu-test
  namespace: default
spec:
  backoffLimit: 0
  template:
    spec:
      restartPolicy: Never
      nodeSelector:
        cloud.google.com/gke-tpu-accelerator: tpu7x
        cloud.google.com/gke-tpu-topology: 2x2x1
      tolerations:
        - key: "google.com/tpu"
          operator: "Exists"
          effect: "NoSchedule"
      containers:
        - name: tpu-shell
          image: us-docker.pkg.dev/cloud-tpu-images/jax-ai-image/tpu:jax0.8.1-rev1
          command: ["sleep", "infinity"]
          resources:
            requests:
              google.com/tpu: "4"
            limits:
              google.com/tpu: "4"
          volumeMounts:
            - name: dev-shm
              mountPath: /dev/shm
      volumes:
        - name: dev-shm
          emptyDir:
            medium: Memory
            sizeLimit: 500Gi
```

```bash
kubectl apply -f /tmp/tpu-e2e-pod.yaml

# 等待 Pod Running（Spot 机器 ~3-5 min 拉起）
kubectl get pods -l job-name=e2e-tpu-test -w
```

### Step 1: 进入 Pod

```bash
POD=$(kubectl get pods -l job-name=e2e-tpu-test -o jsonpath='{.items[0].metadata.name}')
kubectl exec -it $POD -- bash
```

### Step 2: 安装 sglang-jax (从源码)

> **必须从源码安装**: pip 版本 (0.0.2) 缺少 `--ep-size` / `--moe-backend` 等 MoE 关键参数。

```bash
pip install uv
uv venv --python 3.12 /opt/sglang-env && source /opt/sglang-env/bin/activate

cd /opt
git clone https://github.com/sgl-project/sglang-jax.git
cd sglang-jax
uv pip install -e "python[all]"

# 验证
python -m sgl_jax.launch_server --help | grep -q "ep-size" && echo "OK" || echo "FAIL"
```

### Step 3: 下载模型权重

模型从 HuggingFace 自动下载，首次启动时 sglang-jax 会自动拉取。

如需预下载（可选）：

```bash
uv pip install huggingface_hub
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('XiaomiMiMo/MiMo-V2-Flash', local_dir='/tmp/MiMo-V2-Flash')
"
```

预下载后启动时使用本地路径 `--model-path /tmp/MiMo-V2-Flash`。

### Step 4: 启动 sglang-jax 推理服务

```bash
# HF_HOME 放 /dev/shm 避免 boot disk 满被 evict (模型文件 ~70GB)
export HF_HOME=/dev/shm/hf_cache
export JAX_COMPILATION_CACHE_DIR=/dev/shm/jit_cache

uv run python -u -m sgl_jax.launch_server \
    --model-path XiaomiMiMo/MiMo-V2-Flash \
    --trust-remote-code \
    --tp-size 8 --dp-size 2 --ep-size 8 \
    --moe-backend epmoe \
    --host 0.0.0.0 --port 30271 \
    --page-size 256 --context-length 262144 \
    --chunked-prefill-size 4096 \
    --dtype bfloat16 --mem-fraction-static 0.95 \
    --swa-full-tokens-ratio 0.25 --skip-server-warmup \
    --max-running-requests 128 \
    --attention-backend fa
```

**关键参数说明**：

| 参数 | 值 | 说明 |
|------|---|------|
| `--tp-size 8` | 8 | Tensor Parallel = 8 devices |
| `--dp-size 2` | 2 | Data Parallel (attention TP = 8/2 = 4) |
| `--ep-size 8` | 8 | Expert Parallel = 8 devices |
| `--moe-backend epmoe` | epmoe | 单机 v7x-8 比 fused 快 18-26% |
| `--swa-full-tokens-ratio 0.25` | 0.25 | Full Attention 占 KV cache 25%，SWA 占 75% |
| `--page-size 256` | 256 | SWA eviction 效率最优 |
| `--context-length 262144` | 256K | 最大上下文长度 |
| `--chunked-prefill-size 4096` | 4096 | Prefill 分块大小 |
| `--mem-fraction-static 0.95` | 0.95 | KV cache 占 HBM 比例 |
| `--attention-backend fa` | FlashAttention | 高性能 attention kernel |

### Step 5: 等待 Cold Start + Health Check

冷启动包括模型下载（首次）、权重加载、FP8 dequant、XLA 编译。

```bash
# 另开一个 terminal exec 进 Pod 监控
watch -n 10 'curl -sf http://localhost:30271/health && echo "READY" || echo "NOT READY"'
```

或轮询等待：

```bash
for i in $(seq 1 120); do
  CODE=$(curl -sf -o /dev/null -w "%{http_code}" http://localhost:30271/health 2>/dev/null)
  echo "T+$((i*10))s HTTP ${CODE:-000}"
  [ "$CODE" = "200" ] && echo "Server READY!" && break
  sleep 10
done
```

### Step 6: Smoke Test

```bash
# 简单数学题
curl -s http://localhost:30271/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "MiMo-V2-Flash",
    "messages": [{"role": "user", "content": "What is 2+3? Answer with just the number."}],
    "max_tokens": 50,
    "temperature": 0
  }' | python3 -c 'import sys,json; r=json.load(sys.stdin); print("Answer:", r["choices"][0]["message"]["content"])'
# 期望: Answer: 5

# 连续 3 次请求验证 KV cache 不乱码
for i in 1 2 3; do
  echo "--- Request $i ---"
  curl -s http://localhost:30271/v1/chat/completions \
    -H 'Content-Type: application/json' \
    -d '{
      "model": "MiMo-V2-Flash",
      "messages": [{"role": "user", "content": "What is the capital of France? Answer in one word."}],
      "max_tokens": 20,
      "temperature": 0
    }' | python3 -c 'import sys,json; r=json.load(sys.stdin); print(r["choices"][0]["message"]["content"])'
done
# 期望: 3 次都输出 Paris
```

### Step 7: Benchmark（可选）

```bash
# 吞吐测试: 256 prompts × 16K input / 1K output, concurrency 64
# 参数与 SGLang-JAX 官方文档完全一致
source /opt/sglang-env/bin/activate
python -m sgl_jax.bench_serving \
    --backend sgl-jax \
    --port 30271 \
    --dataset-name random \
    --num-prompts 256 \
    --random-input 16384 \
    --random-output 1024 \
    --max-concurrency 64 \
    --random-range-ratio 1 \
    --warmup-requests 0 \
    --tokenizer XiaomiMiMo/MiMo-V2-Flash
```

**Benchmark 结果**

测试条件：256 prompts × 16K input / 1K output, concurrency 64, v7x-8 single-host

| moe-backend | dp | Output tok/s | Median ITL | Mean TPOT | Mean TTFT |
|-------------|---|--------------|-----------|-----------|-----------|
| **fused** | **2** | **782** | **25.8 ms** | **53.2 ms** | 29.4s |
| fused | 1 | 721 | 31.0 ms | 59.4 ms | 30.2s |
| epmoe | 2 | 636 | 32.6 ms | 65.8 ms | 35.7s |
| epmoe | 1 | 523 | 33.4 ms | 77.2 ms | 46.4s |

> **最佳配置: fused + dp=2**，输出吞吐 782 tok/s，Median ITL 25.8 ms。
>
> **与 SGLang-JAX 官方 benchmark 对比**（commit `b787fdef`, dp=1）：
> 官方报告 epmoe 480 tok/s > fused 382 tok/s，结论是"epmoe 单机优于 fused"。
> 我们在最新代码（dev533）上实测结果相反：**fused 全面优于 epmoe**（+23% dp=1, +23% dp=2）。
> fused Pallas kernel 可能已在后续版本中得到优化。
> 官方数据来源：[sglang-jax PR #931](https://github.com/sgl-project/sglang-jax/pull/931)。

**准确率参考** (来源: SGLang-JAX 官方文档)

| Model | Dataset | Metric | Score |
|-------|---------|--------|-------|
| MiMo-V2-Flash | GSM8K | AverageAccuracy | 0.9401 (n=1319) |

---

## Troubleshooting

| 症状 | 根因 | 修复 |
|------|------|------|
| OOM | 并发请求过多 | 降低 `--max-running-requests` 或 `--mem-fraction-static` |
| SWA Pool 耗尽 | SWA token 被全部占满 | 降低并发或调大 `--swa-full-tokens-ratio` |
| XLA 编译超时 | 无 JIT cache | 设置 `JAX_COMPILATION_CACHE_DIR` 持久化缓存 |
| 后续请求乱码 | SWA cache bug（V2.5-Pro 遇到过） | V2-Flash 未报告此问题，若遇到参考 V2.5-Pro 的 5 个 patches |
| Spot 抢占 Pod 消失 | GCP Spot VM 被回收 | 重新 apply Job manifest，autoscaler 会拉新机器 |

---

## 模型核心参数

| 字段 | 值 |
|------|---|
| 架构 | MoE + Hybrid SWA (9 Full + 39 SWA) |
| 总参数 | ~? (FP8 原生量化) |
| 激活参数 | ~? (8 experts activated / 256 total) |
| Hidden dim | 4096 |
| Num layers | 48 (9 full attn + 39 SWA) |
| KV heads | 8 |
| Head dim (K) | 192 |
| Head dim (V) | 128 |
| Sliding window | 128 tokens (SWA 层) |
| 最大位置 | 262,144 |
| Top-K experts | 8 / 256 |
| Scoring | softmax + noaux_tc correction bias |

## 与 MiMo-V2.5-Pro 对比

| | MiMo-V2-Flash | MiMo-V2.5-Pro |
|---|---|---|
| 精度 | FP8 原生 | BF16 |
| 单机可跑 | v7x-8 (1 host) | 需 2× v7x-8 |
| 显存/chip | ~20 GB | ~96 GB |
| SWA patches | 暂未报告需要 | 5 个必须 |
| Cold start | ~15 min (实测) | ~50 min |
| Expert 数 | 256 | 384 |
| MoE backend | epmoe (单机最优) | epmoe |

## 参考

- [sglang-jax MiMo V2 Flash 官方文档](https://github.com/sgl-project/sglang-jax/blob/main/docs/basic_usage/mimo_v2_flash.md)
- [MiMo-V2-Flash HuggingFace Model Card](https://huggingface.co/XiaomiMiMo/MiMo-V2-Flash)
- 同体系 README: [MiMo-V2.5-Pro BF16](../MiMo-V2.5-Pro-BF16/) · [DeepSeek R1 FP4](../DeepSeek-R1-671B-FP4/) · [Qwen3.5 FP8](../Qwen3.5-397B-A17B-FP8/)
