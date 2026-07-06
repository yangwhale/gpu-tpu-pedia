# 基于 SGLang-Jax 的 MiMo-V2-Flash PD 分离性能报告

> 在 TPU v7x 上使用 SGLang-Jax 对 MiMo-V2-Flash 进行 Prefill/Decode 分离（PD disaggregation）的性能测试记录。
>
> 测试配置：MiMo-V2-Flash，16K input / 4K output，非 MTP。

**性能结论（先给结论）：**

- 在 **2 台 TPU v7x 4 chips** 资源下，PD **1P1D** 在 **batch 128** 高压场景达到 **13.64K total tok/s**，相比 two-pod non-PD 的 **12.78K total tok/s** 高约 **6.8%**。
- 在 **3 台 TPU v7x 4 chips** 资源下，PD **2P1D** 达到本轮最高吞吐 **15.31K total tok/s**，相比 two-pod non-PD 高约 **19.8%**（但多使用 1 台 prefill pod）。
- **batch 64** 场景下，two-pod non-PD serve-level DP 更好。因此 **PD 分离的价值主要体现在更高并发、长上下文、prefill/decode 互相干扰明显的场景**。

---

## 目录

- [前言](#前言)
  - [软件说明](#软件说明)
  - [硬件说明](#硬件说明)
  - [测试口径](#测试口径)
- [性能测试](#性能测试)
  - [总体结果](#总体结果)
  - [结果记录](#结果记录)
  - [稳态能力](#稳态能力)
- [与非 PD 的对比](#与非-pd-的对比)
  - [batch 128 场景](#batch-128-场景)
- [结论](#结论)
  - [适用场景](#适用场景)
  - [性能判断](#性能判断)
- [未来提升方向](#未来提升方向)
- [附录 A: 运行环境和命令](#附录-a-运行环境和命令)

---

## 前言

### 软件说明

**SGLang-Jax**

- repo: `https://github.com/primatrix/sglang-jax.git`
- 分支: `epic/mimo-pd-disggragation`
- 远端 benchmark commit: `c6105f1cb09119ce40462d9f65776198a312737b`
- 本报告基于的前序本地报告 commit: `d3ac42493745dcf4d9e141ef0a2a351029d7cc25`

**模型**

- `/models/MiMo-V2-Flash`

**评测工具**

- `sgl_jax.bench_serving`
- `evalscope==0.17.1`

### 硬件说明

本轮测试使用 **TPU v7x 2x2x2（4 chips, 8 devices）**，每个 server 内部使用 `tp-size=8`、`ep-size=8`、`dp-size=2`。

| 配置 | 资源规模 | 说明 |
|------|----------|------|
| PD 1P1D | 2 台 TPU pod | 1 台 prefill，1 台 decode |
| PD 2P1D | 3 台 TPU pod | 2 台 prefill，1 台 decode |
| non-PD serve-level DP | 2 台 TPU pod | 2 个普通 non-PD server，通过轻量 round-robin proxy 分发 |

### 测试口径

本轮主测试为 **`16K input / 4K output`**，**`request_rate=inf`**，即一次性并发压测。这类测试适合观察高压下的吞吐上限和系统尾部行为，**但不是严格的 SLO goodput 测试**。

需要特别注意：

- **client TTFT 包含并发排队**，不应直接当作单请求 prefill 能力。
- **prefill 能力** 更适合看 serve log 中的 `active input tok/s`。
- **decode 能力** 更适合看 serve log 中的 `highwater output tok/s` 和 client `ITL`。

---

## 性能测试

### 总体结果

核心结果如下：

| 模式 | 资源 | 并发 | total tok/s | input tok/s | output tok/s | mean ITL |
|------|------|------|-------------|-------------|--------------|----------|
| non-PD serve-level DP | 2 pods | batch 64 | 11.70K | 9.36K | 2.34K | 23.89ms |
| PD 1P1D | 2 pods | batch 64 | 10.55K | 8.44K | 2.11K | 24.13ms |
| PD 2P1D | 3 pods | batch 64 | 10.83K | 8.66K | 2.17K | 25.84ms |
| non-PD serve-level DP | 2 pods | batch 128 | 12.78K | 10.22K | 2.56K | 34.56ms |
| PD 1P1D | 2 pods | batch 128 | 13.64K | 10.91K | 2.73K | 28.73ms |
| PD 2P1D | 3 pods | batch 128 | **15.31K** | 12.25K | 3.06K | 35.08ms |

> 说明：这里更多是体现整体端到端趋势变化，并没有考虑到 SLO 这些信息。稳态情况看下面的“稳态能力”会更有参考价值。

### 结果记录

**PD 1P1D, batch 128**

- total token throughput: `13.64K tok/s`
- input token throughput: `10.91K tok/s`
- output token throughput: `2.73K tok/s`
- peak output token throughput: `3.48K tok/s`
- mean TTFT: `57.60s`
- mean ITL: `28.73ms`

```
============ Serving Benchmark Result ============
Backend:                                 sgl-jax
Traffic request rate:                    inf
Max request concurrency:                 128
Successful requests:                     384
Benchmark duration (s):                  576.49
Total input tokens:                      6291456
Total generated tokens:                  1572864
Request throughput (req/s):              0.67
Input token throughput (tok/s):          10913.33
Output token throughput (tok/s):         2728.33
Peak output token throughput (tok/s):    3483.00
Total token throughput (tok/s):          13641.66
Mean TTFT (ms):                          57601.52
Mean ITL (ms):                           28.73
==================================================
```

**PD 2P1D, batch 128**

- total token throughput: `15.31K tok/s`
- input token throughput: `12.25K tok/s`
- output token throughput: `3.06K tok/s`
- peak output token throughput: `3.97K tok/s`
- mean TTFT: `20.11s`
- mean ITL: `35.08ms`

```
============ Serving Benchmark Result ============
Backend:                                 sgl-jax
Traffic request rate:                    inf
Max request concurrency:                 128
Successful requests:                     384
Benchmark duration (s):                  513.78
Total input tokens:                      6291456
Total generated tokens:                  1572864
Request throughput (req/s):              0.75
Input token throughput (tok/s):          12245.51
Output token throughput (tok/s):         3061.38
Peak output token throughput (tok/s):    3968.00
Total token throughput (tok/s):          15306.89
Mean TTFT (ms):                          20114.33
Mean ITL (ms):                           35.08
==================================================
```

**Non-PD Two-pod Serve-Level DP, batch 128**

- total token throughput: `12.78K tok/s`
- input token throughput: `10.22K tok/s`
- output token throughput: `2.56K tok/s`
- peak output token throughput: `5.02K tok/s`
- mean TTFT: `46.49s`
- mean ITL: `34.56ms`

```
============ Serving Benchmark Result ============
Backend:                                 sgl-jax
Traffic request rate:                    inf
Max request concurrency:                 128
Successful requests:                     383
Benchmark duration (s):                  613.79
Total input tokens:                      6275072
Total generated tokens:                  1568768
Request throughput (req/s):              0.62
Input token throughput (tok/s):          10223.43
Output token throughput (tok/s):         2555.86
Peak output token throughput (tok/s):    5017.00
Total token throughput (tok/s):          12779.29
Mean TTFT (ms):                          46493.20
Mean ITL (ms):                           34.56
==================================================
```

> 说明：这里 non-PD TTFT 时间偏长，是因为 batch size 128 并发会在 router 层堆积（对应长 prompt request，serve 只能串行处理 prefill）。

### 稳态能力

从 serve log 的稳态窗口看，PD 1P1D 在 batch 128 下的 prefill 和 decode 基本匹配。含义是：

- **2 台机器时**，PD 1P1D 已经能把 prefill 和 decode 的处理速率匹配到接近平衡。
- **3 台机器时**，增加 prefill 能明显提升 burst 吞吐和 TTFT，但 decode 侧开始成为更紧的资源。
- 后续如果继续扩大资源，应**优先验证更多 decode 节点或 decode MTP，而不是只增加 prefill**。

---

## 与非 PD 的对比

### batch 128 场景

batch 128 时，PD 分离收益体现在：

- PD 1P1D 相比 two-pod non-PD 高约 **6.8%**。
- PD 2P1D 相比 two-pod non-PD 高约 **19.8%**，但多使用 1 台 prefill pod。
- PD 1P1D 的 mean ITL 为 **28.73ms**，优于 two-pod non-PD 的 **34.56ms**。

原因是高并发长上下文下，non-PD 的每个 replica 都需要同时承担 prefill 和 decode。当 prefill 较重时，同一设备上的 prefill/decode 会互相干扰；PD 分离把两类负载放到不同设备上，更容易保持 decode 的持续输出能力。

稳态窗口下的 prefill / decode 拆分对比：

| 模式 | prefill active input tok/s | prefill req/s @16K | decode highwater output tok/s | decode req/s @4K | 说明 |
|------|----------------------------|--------------------|-------------------------------|------------------|------|
| PD 1P1D, batch 128 | 12.79K | 0.780 | 3.18K | 0.776 | P/D 基本匹配，是当前最干净的 2 pod 对比 |
| PD 2P1D, batch 128 | 15.71K | 0.959 | 3.63K | 0.886 | prefill 余量更大，decode 开始更紧 |
| non-PD two-pod, batch 128 | 11.64K | 0.711 | 4.67K | 1.140 | decode rank-local 峰值较高，但稳态窗口不完全对齐 |

---

## 结论

### 适用场景

- 如果目标 workload 是**中等并发**（例如 batch 64 左右），且主要追求简单部署，**two-pod non-PD serve-level DP 更合适**。
- 如果目标 workload 是**长上下文、高并发**（例如 `16K/4K batch 128`），**PD 分离有明确价值**。在 2 pod 公平对比下，PD 1P1D 已经有 6.8% 总吞吐优势，并且 ITL 更好。
- 如果允许**多 P 单 D**配置，**PD 2P1D 是本轮最高吞吐配置**，总吞吐达到 `15.31K tok/s`，但 decode 侧开始更紧，继续扩容需要谨慎评估 P/D 配比以及对应的 mesh 配置。

### 性能判断

- **prefill 侧**：长输入情况下，当前 prefill 能力本身和 non-PD 不会有太大差异，都是串行 chunk prefill；优势主要体现在**不会 prefill/decode 相互干扰**，且能够**动态补充 prefill 节点**。
- **decode 侧**：长输出阶段仍主要取决于 decode pod 能力。
- **transfer 侧**：当前 PD 分离的 with-overlap 能力并没有完全提供，尤其是 **decode prealloc 和 KV transfer 的 overlap**。

---

## 未来提升方向

1. 增加 **open-loop request-rate sweep**，用明确的 TTFT/ITL SLO 计算 goodput。
2. 验证 `1P:2D`、`1P:3D`，确认长输出 workload 下更多 decode 是否优于更多 prefill。
3. 尝试 **decode MTP**，观察 output token/s 和 ITL 是否能进一步改善。
4. 继续优化 PD transfer path，重点关注 **prealloc 和 KV transfer 的重叠**。
5. 保留 precompile cache 作为启动优化；运行时吞吐优化仍应聚焦 **P/D 配比**和 **transfer overlap**。

---

## 附录 A: 运行环境和命令

### 运行环境

```bash
export JAX_COMPILATION_CACHE_DIR=/tmp/tpu_logs/jit_cache
export LIBTPU_INIT_ARGS=--xla_tpu_dvfs_p_state=7
export PYTHONPATH=/tmp/tpu-raiden-cached/tpu-raiden:/tmp/sglangjax:${PYTHONPATH:-}
export SGLANG_JAX_USE_RAIDEN=1
```

### PD 1P1D（1 台 prefill pod + 1 台 decode pod）

核心 server 参数如下：

```bash
# bootstrap
/usr/local/bin/python -m sgl_jax.srt.disaggregation.run_bootstrap \
  --pod 0.0.0.0 --port 8998

# prefill
/usr/local/bin/python -m sgl_jax.launch_server \
  --model-path /models/MiMo-V2-Flash --trust-remote-code \
  --tp-size 8 --ep-size 8 --moe-backend fused_v2 \
  --page-size 256 --context-length 262144 \
  --disable-radix-cache --chunked-prefill-size 2048 --max-prefill-tokens 16384 \
  --dtype bfloat16 --mem-fraction-static 0.84 --swa-full-tokens-ratio 0.2 \
  --skip-server-warmup --max-running-requests 256 \
  --dp-size 2 --dp-schedule-policy round_robin \
  --precompile-bs-paddings 1 4 8 16 32 64 128 256 \
  --precompile-token-paddings 4096 \
  --disaggregation-enable-d2h --disaggregation-use-raiden \
  --enable-metrics --enable-request-time-stats-logging \
  --pod 0.0.0.0 --port 10000 \
  --disaggregation-mode prefill \
  --disaggregation-bootstrap-url http://localpod:8998 \
  --disaggregation-max-inflight-transfers 8

# decode
/usr/local/bin/python -m sgl_jax.launch_server \
  --model-path /models/MiMo-V2-Flash --trust-remote-code \
  --tp-size 8 --ep-size 8 --moe-backend fused_v2 \
  --page-size 256 --context-length 262144 \
  --disable-radix-cache --chunked-prefill-size 2048 --max-prefill-tokens 16384 \
  --dtype bfloat16 --mem-fraction-static 0.84 --swa-full-tokens-ratio 0.2 \
  --skip-server-warmup --max-running-requests 256 \
  --dp-size 2 --dp-schedule-policy round_robin \
  --precompile-bs-paddings 1 4 8 16 32 64 128 256 \
  --precompile-token-paddings 4096 \
  --disaggregation-enable-d2h --disaggregation-use-raiden \
  --enable-metrics --enable-request-time-stats-logging \
  --pod 0.0.0.0 --port 10001 \
  --disaggregation-mode decode \
  --disaggregation-max-inflight-transfers 32

# router
/usr/local/bin/python -m sgl_jax.srt.disaggregation.launch_router \
  --pd-disaggregation --mini-lb \
  --prefill http://<prefill-pod>:10000 8998 \
  --decode http://localpod:10001 \
  --prefill-bootstrap-pod <prefill-pod> \
  --max-concurrent-requests 256 \
  --pd-prefill-max-inflight-requests 4 \
  --pd-router-admission-poll-ms 50 \
  --pod 0.0.0.0 --port 30000
```

### Benchmark 命令

```bash
/usr/local/bin/python -m sgl_jax.bench_serving \
  --backend sgl-jax \
  --base-url http://localpod:30000 \
  --model /models/MiMo-V2-Flash \
  --tokenizer /models/MiMo-V2-Flash \
  --dataset-name random \
  --random-input-len 16384 \
  --random-output-len 4096 \
  --random-range-ratio 1.0 \
  --num-prompts 384 \
  --request-rate inf \
  --max-concurrency 128 \
  --warmup-requests 0 \
  --seed 12345 \
  --output-details \
  --extra-request-body '{"sampling_params":{"temperature":0.1,"top_p":0.95,"max_new_tokens":4096,"ignore_eos":true}}'
```

### Non-PD two-pod serve-level DP

使用两个普通 server：

```bash
/usr/local/bin/python -m sgl_jax.launch_server \
  --model-path /models/MiMo-V2-Flash --trust-remote-code \
  --tp-size 8 --ep-size 8 --moe-backend fused_v2 \
  --page-size 256 --context-length 262144 \
  --disable-radix-cache --chunked-prefill-size 2048 --max-prefill-tokens 16384 \
  --dtype bfloat16 --mem-fraction-static 0.84 --swa-full-tokens-ratio 0.2 \
  --skip-server-warmup --max-running-requests 256 \
  --dp-size 2 --dp-schedule-policy round_robin \
  --precompile-bs-paddings 1 4 8 16 32 64 128 256 \
  --precompile-token-paddings 4096 \
  --enable-metrics --enable-request-time-stats-logging \
  --pod 0.0.0.0 --port 30010
```

non-PD proxy 仅做 round-robin 转发：

```
backend[0] = http://<rank0-pod>:30010
backend[1] = http://127.0.0.1:30010
listen     = http://0.0.0.0:30000
```
