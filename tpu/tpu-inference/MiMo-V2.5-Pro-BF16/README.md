# MiMo-V2.5-Pro (BF16) 推理 on 2× TPU v7x-8

> 🌐 **Languages** | **语言**: **中文**

> 端到端指南：在 2× TPU v7x-8（16 devices, multi-host）上运行 MiMo-V2.5-Pro BF16 推理。
>
> **架构**：~1T 总参 / 42B 激活 / **Hybrid SWA**（60 SWA + 10 Full Attention）+ 384 routed experts + BF16
>
> **推理框架**: [sglang-jax](https://github.com/sgl-project/sglang)（**不是 vLLM**）— multi-host TP=8, EP=2
>
> **模型**: MiMo-V2.5-Pro（~1T 总参，BF16）

---

## 🚨 已知关键限制

> 当前部署是 **POC 验证阶段**，5 个 SWA cache patches 必需。

- **Multi-host only**：~1T 总参 BF16 需要 2× v7x-8（2 hosts × 8 devices），单机 OOM
- **5 个 SWA cache patches 必需**：sglang-jax 的 SWA radix cache 有多个 accounting bug，不打 patch 会导致 KV cache 损坏 → 后续请求输出乱码
- **低并发**：SWA pool（7908 tokens）+ context_length=4096 仅容纳 ~1.9 个完整序列
- **冷启动 ~50 min**：25 min 权重加载 + 5 min FP8 QKV dequant + 15 min XLA 编译 + 7 min DECODE PRECOMPILE
- **Benchmark 未完成**：smoke test 通过，吞吐 / 延迟 / 准确率评测待做

---

## ⚠️ 必读约束

### A. 5 个 SWA Cache Patches（全部必须）

sglang-jax 的 Hybrid SWA（Sliding Window Attention）radix cache 有多个 accounting bug，会导致 KV cache 损坏。**5 个 patches 缺一不可**：

| # | Patch 文件 | 目标文件 | 修复内容 |
|---|-----------|---------|---------|
| 1 | `patch_swa_cache_leak.py` | `swa_radix_cache.py` | `inc_lock_ref` / `dec_lock_ref` 在 lock/unlock 时 `_swa_eff_len()` 值不同（node split 导致），snapshot eff_len 修复 `swa_protected_size_` 漂移 |
| 2 | `patch_swa_cache_leak.py` | `swa_radix_cache.py` | `_delete_tombstone_leaf` 用 `len(node.key)` 而非 `len(node.value)` 导致 `full_evictable_size_` 计错 |
| 3 | `patch_sanity_check_tolerant.py` | `swa_radix_cache.py` | `sanity_check()` evictable size 不一致时 `assert` → `warning`（容忍 patch 未完全修复的残留 drift） |
| 4 | `patch_check_memory_tolerant.py` | `scheduler.py` | `check_memory()` leak 检测 `ValueError` → `warning`（同上） |
| 5 | `patch_disable_evict_swa_v2.py` | `schedule_batch.py` | **核心 fix**：`maybe_evict_swa()` 完全禁用 — `_evict_swa()` 在 decode 和 extend 路径都会 free SWA slots 而不通知 tree cache，导致 prefix cache hit 时读到 stale KV 数据 |

**根因分析（Patch #5）**：
- `_evict_swa()` 调用 `free_swa()` 释放 sliding_window 外的 SWA slots
- 但**从不通知 tree cache**（`notify_swa_mapping_freed()` 是 no-op）
- 释放的 slots 被分配给新请求
- 旧请求的 tree node 仍引用已释放的 SWA positions → **attention 读到垃圾数据**
- Decode 和 extend 两条路径都会触发（v1 patch 只禁 decode 不够）

### B. 关键启动参数

| 参数 | 值 | 作用 |
|------|---|------|
| `--tp-size` | `8` | Tensor Parallel = 8 devices/host |
| `--ep-size` | `2` | Expert Parallel = 2 (跨 2 hosts) |
| `--nnodes` | `2` | Multi-host 2 节点 |
| `--context-length` | `4096` | 受 SWA pool 容量限制 |
| `--max-total-tokens` | `8192` | 总 token 预算 |
| `--mem-fraction-static` | `0.95` | KV cache 占 HBM 比例 |
| `--enable-cache-report` | — | 输出 cache 使用情况 |

---

## 🧭 部署步骤

### 前提

- GKE 集群 + 2× TPU v7x-8 节点（multi-host）
- Lustre 或 GCS 挂载的模型权重路径 `/lustre/models/MiMo-V2.5-Pro`
- sglang-jax container image（含 `/opt/sglang-jax/`）
- StatefulSet + Headless Service（2 个 pod 互相发现）

### Step 1: 准备 Pod + 验证模型

```bash
CTX=<your-gke-context>
POD_0=mimo-benchmark-0
POD_1=mimo-benchmark-1
MODEL=/lustre/models/MiMo-V2.5-Pro

# 验证 pods running
kubectl --context=$CTX get pods | grep mimo-benchmark
# 期望: mimo-benchmark-0  1/1  Running
#        mimo-benchmark-1  1/1  Running

# 验证模型权重
kubectl --context=$CTX exec $POD_0 -- bash -c "ls $MODEL/*.safetensors | wc -l"
```

### Step 2: 应用 5 个 Patches（两个 pod 都要）

将 4 个 patch 脚本 cp 到两个 pod 并执行：

```bash
PATCHES=(
  patch_swa_cache_leak.py
  patch_sanity_check_tolerant.py
  patch_check_memory_tolerant.py
  patch_disable_evict_swa_v2.py
)

for POD in $POD_0 $POD_1; do
  for P in "${PATCHES[@]}"; do
    kubectl --context=$CTX cp /tmp/$P $POD:/tmp/$P
    kubectl --context=$CTX exec $POD -- python3 /tmp/$P
  done
  echo "=== $POD patched ==="
done
```

**验证 patches 生效**（在任一 pod 上）：

```bash
kubectl --context=$CTX exec $POD_0 -- bash -c "
  echo 'Patch 1 (eff_len_at_lock):' \$(grep -c '_swa_eff_len_at_lock' /opt/sglang-jax/python/sgl_jax/srt/mem_cache/swa_radix_cache.py)
  echo 'Patch 2 (tombstone fix):' \$(grep -c 'len(node.value)' /opt/sglang-jax/python/sgl_jax/srt/mem_cache/swa_radix_cache.py)
  echo 'Patch 3 (sanity tolerant):' \$(grep -c 'evictable mismatch (tolerant)' /opt/sglang-jax/python/sgl_jax/srt/mem_cache/swa_radix_cache.py)
  echo 'Patch 4 (check_memory tolerant):' \$(grep -c '_leak_count' /opt/sglang-jax/python/sgl_jax/srt/managers/scheduler.py)
  echo 'Patch 5 (evict_swa disabled):' \$(grep -c 'DISABLED: _evict_swa frees SWA slots' /opt/sglang-jax/python/sgl_jax/srt/managers/schedule_batch.py)
"
# 期望: 所有输出 ≥1
```

### Step 3: 启动 sglang-jax（File-based launcher，两个 pod）

⚠️ **必须用 file-based launcher**：`kubectl exec` 的 nohup 会被 SIGKILL。

```bash
cat > /tmp/launch_sglang_multihost.sh <<'LAUNCHER'
#!/bin/bash
set -e

NNODES=2
NPROC_PER_NODE=8
EP_SIZE=2
MODEL_PATH=/lustre/models/MiMo-V2.5-Pro
DIST_ADDR="mimo-benchmark-0.mimo-bench-headless-svc:5000"

# 从 hostname 提取 node rank (mimo-benchmark-0 → 0, mimo-benchmark-1 → 1)
NODE_RANK=$(hostname | grep -oP '\d+$')

echo "Starting sglang-jax: node_rank=$NODE_RANK, nnodes=$NNODES, ep=$EP_SIZE"

pgrep -f 'sglang\|srt' | xargs -r kill -9 2>/dev/null || true
sleep 2
rm -f /tmp/libtpu_lockfile

setsid nohup python3 -m sgl_jax.launch_server \
  --model-path "$MODEL_PATH" \
  --tp-size $NPROC_PER_NODE --ep-size $EP_SIZE \
  --nnodes $NNODES --node-rank "$NODE_RANK" \
  --dist-init-addr "$DIST_ADDR" \
  --context-length 4096 --max-total-tokens 8192 \
  --mem-fraction-static 0.95 \
  --port 30271 --host 0.0.0.0 \
  --enable-cache-report --trust-remote-code \
  >> /tmp/sglang_mimo.log 2>&1 < /dev/null &
disown
echo "launched pid=$!"
exit 0
LAUNCHER

# cp + launch on both pods
for POD in $POD_0 $POD_1; do
  kubectl --context=$CTX cp /tmp/launch_sglang_multihost.sh $POD:/tmp/launch_sglang_multihost.sh
  kubectl --context=$CTX exec $POD -- bash /tmp/launch_sglang_multihost.sh
done
```

### Step 4: 等待 Cold Start (~50 min) + Health Check

```bash
# 轮询 health（60s 间隔，最多 60 min）
for i in $(seq 1 60); do
  C=$(kubectl --context=$CTX exec $POD_0 -- curl -sf -o /dev/null -w "%{http_code}" http://localhost:30271/health 2>/dev/null)
  C=${C:-000}
  echo "T+$((i*60))s HTTP $C"
  [ "$C" = "200" ] && break
  sleep 60
done
```

**Cold start 阶段时间参考**：
| 阶段 | 耗时 | 日志标志 |
|------|------|---------|
| 权重加载 | ~25 min | `Loading model weights...` |
| FP8 QKV dequant | ~5 min | `dequantize` 相关 log |
| XLA 编译 | ~15 min | `XLA compilation` |
| DECODE PRECOMPILE | ~7 min | `DECODE PRECOMPILE` |
| **总计** | **~50 min** | `Application startup complete` / health 200 |

### Step 5: Smoke Test

```bash
# 简单 completion 测试
kubectl --context=$CTX exec $POD_0 -- curl -s http://localhost:30271/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"MiMo-V2.5-Pro","messages":[{"role":"user","content":"What is 2+3? Answer with just the number."}],"max_tokens":50,"temperature":0}' \
  | python3 -c 'import sys,json; r=json.load(sys.stdin); print("content:", repr(r["choices"][0]["message"]["content"]), "| finish:", r["choices"][0]["finish_reason"])'
# 期望: content: '5' | finish: stop

# 多次请求验证（关键：验证后续请求不乱码）
for i in 1 2 3; do
  echo "--- Request $i ---"
  kubectl --context=$CTX exec $POD_0 -- curl -s http://localhost:30271/v1/chat/completions \
    -H 'Content-Type: application/json' \
    -d '{"model":"MiMo-V2.5-Pro","messages":[{"role":"user","content":"What is the capital of France? Answer in one word."}],"max_tokens":20,"temperature":0}' \
    | python3 -c 'import sys,json; r=json.load(sys.stdin); print(repr(r["choices"][0]["message"]["content"]))'
done
# 期望: 3 次都输出 'Paris' 或包含 Paris 的连贯回答
```

**验证标准**：
- ✅ 3/3 请求输出连贯（不是乱码）
- ✅ 0 LEAK warnings（`grep LEAK /tmp/sglang_mimo.log`）
- ✅ 0 sanity check warnings（`grep "sanity check" /tmp/sglang_mimo.log`）
- ✅ SWA token 计数始终为正（`grep "swa token" /tmp/sglang_mimo.log | head -5`）

---

## Troubleshooting

| 症状 | 根因 | 修复 |
|------|------|------|
| **后续请求输出乱码** | `maybe_evict_swa` 未禁用，SWA slots 被回收但 tree cache 不知道 | 确认 patch #5 已应用：`grep -c 'DISABLED: _evict_swa' schedule_batch.py` 应输出 1 |
| **SWA token 计数为负** (`#swa token: -224`) | decode 或 extend 路径的 `_evict_swa` 释放了不该释放的 slots | 同上，确认完整禁用（v2 patch，不是 v1） |
| **AssertionError: evictable_size != lru_list_evictable_size** | SWA cache accounting drift | patch #3（sanity_check tolerant）降级为 warning |
| **ValueError: token_to_kv_pool_allocator memory leak** | SWA protected_size 漂移 | patch #1（eff_len snapshot）修复 + patch #4（check_memory tolerant）兜底 |
| **冷启动 >60 min** | `/dev/shm` 残留占用 RAM | 清理 `/dev/shm`：`rm -rf /dev/shm/sem.* /dev/shm/wrk_*` |
| **两个 pod 无法互联** | Headless service DNS 未就绪 | 确认 `mimo-bench-headless-svc` service 存在，`nslookup mimo-benchmark-0.mimo-bench-headless-svc` 解析正常 |
| **kubectl exec 启动后进程消失** | 非 file-based launcher，被 SIGKILL | 必须用 file-based launcher（Step 3） |

---

## 模型核心参数

| 字段 | 值 |
|------|---|
| 架构 | MoE + **Hybrid SWA** (60 SWA + 10 Full Attention) |
| 参数 | ~1T 总 / 42B 激活 / 384 routed experts |
| 维度 | Hidden 6144, 8 KV heads, 70 layers |
| 量化 | BF16（无量化） |
| Sliding Window | 128 tokens（SWA 层） |
| 最大位置 | 1,048,576 |

## 硬件门槛

| 项 | 要求 |
|---|---|
| TPU | **2× v7x-8**（16 devices, multi-host 必须） |
| HBM | 96 GB/device × 16 = 1,536 GB |
| 推理框架 | sglang-jax（**不是 vLLM**） |
| 并行策略 | TP=8, EP=2 |

## 待完成

- [ ] P1-3: R3 FusedMoE + R5 Prefetch 组合测试
- [ ] P1-4: BSZ Sweep (64/128/256/512)
- [ ] P1-5: `exp/skip-padding-tokens` 分支测试
- [ ] GSM8K / MMLU 准确率评测
- [ ] 吞吐 / 延迟 benchmark 数据

## 参考

- 同体系 README: [DeepSeek R1 FP4](../DeepSeek-R1-671B-FP4/) · [Qwen3.5 FP8](../Qwen3.5-397B-A17B-FP8/) · [GLM-5.1 FP4](../GLM-5.1-754B-FP4/)
