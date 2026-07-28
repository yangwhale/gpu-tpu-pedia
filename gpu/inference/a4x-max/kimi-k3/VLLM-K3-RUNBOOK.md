# vLLM · Kimi K3 (2.8T) · GB300 NVL72 端到端 Runbook

> 从 0 开始：起 pod → 建 RAID → 拉 1.4 TB 权重 → 起 TP8 服务 → 压测 → 复现官方 331/370 tok/s。
>
> **每节的状态标记含义**
> - `[官方]` —— 逐字来自 vLLM day-0 博客 / recipes，未在本环境跑过
> - `[本环境]` —— 复用本仓库已验证过的环境步骤（V4 那套），对 K3 做了参数替换
> - `[待验证]` —— 需要跑通后回填的位置
>
> 全文尚无 `[已验证]`。跑通一节就把该节改标，并在 §10 记录。

---

## ⚠️ 开跑前必须知道的三件事

**一、现在只能用 Docker 镜像。** 官方原话：由于依赖复杂（含 FlashInfer 等多个 pre-release 依赖），
**目前只有 Docker 镜像可用**，pip 装不起来。镜像 tag 见
<https://recipes.vllm.ai/moonshotai/Kimi-K3>。

**二、`--enable-prefix-caching` 不加就没有。** vLLM 对绝大多数模型默认开启 prefix caching，
**唯独 K3 默认关闭**。不加这个 flag，KDA 混合缓存那一整套优化等于没有。

**三、TP8 在 GB300 上是跨 2 个节点。** GB300 NVL72 每节点 4 GPU，
所以官方 recipe 里的 `--tensor-parallel-size 8` 必须配 `--nnodes 2`。
两个节点**必须在同一 NVL72 subblock 内**（KV 走域内 NVLink，跨域不行）——
这条是本环境 V4 runbook 的血泪教训，K3 同样适用。

---

## 0. TL;DR `[官方]`

最简起法（单节点 8×B300，或 GB300 上凑够 8 卡）：

```bash
vllm serve moonshotai/Kimi-K3 \
  --tensor-parallel-size 8 \
  --trust-remote-code \
  --load-format fastsafetensors \
  --enable-prefix-caching \
  --enable-auto-tool-choice \
  --tool-call-parser kimi_k3 \
  --reasoning-parser kimi_k3
```

要低延迟就加 DSpark 投机解码（单流解码约 3 倍）：

```bash
--speculative-config '{"model":"Inferact/Kimi-K3-DSpark","method":"dspark","num_speculative_tokens":7,"attention_backend":"FLASHINFER_MLA","draft_sample_method":"probabilistic","rejection_sample_method":"block"}'
```

---

## 1. 前置条件 `[本环境]`

```bash
# ① 两个节点在同一 NVL72 subblock（manifest 的 podAffinity 保证）
#    K3 的 TP8 跨节点走 NVLink，跨域会直接崩性能

# ② RAID 挂载 —— 先查这个
#    正常 12T；若显示 256K 说明 RAID 没挂，1.4 TB 权重放不进去
df -h /mnt/disks/raid/0

# ③ 能拉 vLLM K3 镜像（imagePullSecrets）
#    GB300 集群有 CronJob 自动刷 ar-pull-secret，YAML 里只需 imagePullSecrets

# ④ 没有孤儿 ComputeDomain 占着 DRA channel
kubectl get computedomain -A
```

> RAID 建不起来看 [gb300-local-ssd-raid0-SETUP.md](../deepseek-v4/gb300-local-ssd-raid0-SETUP.md)，
> 注意里面 `md0 → md127` 那个陷阱。

---

## 2. 部署 pod fleet `[本环境]`

复用 V4 的 manifest，改镜像和副本数即可：

```bash
# 参考 ../deepseek-v4/manifests/vllm-fleet.yaml
# 需要改的：
#   - image: 换成 vLLM K3 day-0 镜像
#   - replicas: TP8 → 2；TP16 → 4
#   - podAffinity: 保持同 subblock
kubectl apply -f manifests/k3-fleet.yaml
kubectl get pods -o wide | grep k3
```

---

## 3. 拉权重 `[本环境]`

**体量：2.8T 参数 × MXFP4 ≈ 1.4 TB。** 比 V4-Pro 的 832G 还大 70%，
内存盘绝对放不下，必须落 RAID。

```bash
# 别用 gcloud —— vLLM 镜像里没装。用 curl + GCS JSON API 的并行脚本
# 参数化版本已在 V4 目录：../deepseek-v4/scripts/pull-gcs-model.sh
BUCKET=<your-bucket> PREFIX=Kimi-K3 JOBS=16 \
  bash scripts/pull-gcs-model.sh /mnt/disks/raid/0/Kimi-K3

# DSpark draft 模型（小，直接 HF 或 GCS 都行）
#   Inferact/Kimi-K3-DSpark

# ⚠️ access token 只有 1 小时有效期，1.4 TB 拉不完就会断 —— 拉之前先刷新
# ⚠️ 重建任何 pod 之后都要重新校验 shard 数（hostPath 不跟着 pod 走）
du -sh /mnt/disks/raid/0/Kimi-K3
ls /mnt/disks/raid/0/Kimi-K3/*.safetensors | wc -l   # [待验证] 填实际 shard 数
```

---

## 4. 启动服务 · TP8 + DSpark `[官方]`

这一段是**官方 reproduce recipe 的逐字复刻**，只把 `HEAD_ADDR` 换成本环境地址。

### 4.1 环境变量

```bash
export NCCL_DMABUF_ENABLE=0
export VLLM_ALLREDUCE_USE_FLASHINFER=1
export VLLM_USE_RUST_FRONTEND=1
export VLLM_ENGINE_READY_TIMEOUT_S=3600      # 2.8T 加载慢，别用默认超时
export HEAD_ADDR=<node-0 IP>
```

### 4.2 起服务（两个节点各跑一次，只有 `--node-rank` 不同）

```bash
vllm serve moonshotai/Kimi-K3 \
  --enable-prefix-caching \
  --tensor-parallel-size 8 \
  --nnodes 2 \
  --node-rank 0 \
  --moe-backend auto \
  --trust-remote-code \
  --load-format fastsafetensors \
  --max-num-seqs 512 \
  --gpu-memory-utilization 0.9 \
  --max-model-len auto \
  --max-cudagraph-capture-size 256 \
  --kv-cache-dtype fp8 \
  --attention-config '{"mla_prefill_backend":"FLASHINFER","use_prefill_query_quantization":true}' \
  --speculative-config '{"model":"Inferact/Kimi-K3-DSpark","method":"dspark","num_speculative_tokens":7,"attention_backend":"FLASHINFER_MLA","draft_sample_method":"probabilistic","rejection_sample_method":"block"}'
```

node-1 上把 `--node-rank 0` 改成 `--node-rank 1`，其余完全相同。

### 4.3 参数逐条解释

| 参数 | 为什么 |
|---|---|
| `--enable-prefix-caching` | **K3 默认关**，不加则混合 KDA 缓存全部失效 |
| `--tensor-parallel-size 8` + `--nnodes 2` | GB300 每节点 4 卡，TP8 = 2 节点 |
| `--moe-backend auto` | 官方 recipe 用 auto；如需手动：TP>1 用 `flashinfer_trtllm`，DEP 用 `deep_gemm_mega_moe` |
| `--load-format fastsafetensors` | 1.4 TB 权重，普通 loader 太慢 |
| `--kv-cache-dtype fp8` | 压 KV，为 1M 上下文留空间 |
| `--max-model-len auto` | 让引擎按显存自算，别硬写 1048576 |
| `--attention-config` | MLA prefill 走 FlashInfer + prefill query 量化 |
| `--speculative-config` | DSpark，`num_speculative_tokens: 7` |
| `VLLM_ENGINE_READY_TIMEOUT_S=3600` | 2.8T 模型加载 + CUDA graph capture 很久 |

### 4.4 就绪判据 `[待验证]`

```bash
curl -s http://${HEAD_ADDR}:8000/health
# 别用 ss 判端口 —— 镜像里通常没装（V4 踩过）
```

> **[待验证]** 补充：启动日志里应出现的 MoE backend / KDA kernel 选择行，
> 用来判断「跑起来了」和「跑对了」——参考 V4 vLLM runbook 里认 DeepGEMM 那四行的做法。
> 所有健康信号全绿但性能腰斩是常态。

---

## 5. 压测 · 复现官方数字 `[官方]`

两条命令对应两个官方数字。

### 5.1 无投机解码基线（8K/1K random，bs=1）

```bash
vllm-bench \
  --backend openai \
  --base-url "http://${HEAD_ADDR}:8000" \
  --model moonshotai/Kimi-K3 \
  --dataset-name random \
  --random-input-len 8192 \
  --random-output-len 1024 \
  --random-range-ratio 0.8 \
  --prompt-token-ids \
  --ignore-eos \
  --sweep-max-concurrency 1 \
  --sweep-num-prompts-factor 10 \
  --seed 42 \
  --percentile-metrics "ttft,tpot,itl,e2el" \
  --metric-percentiles "50,90,99" \
  --save-result
```

**官方目标：TP8 111 tok/s/user，TP16 118 tok/s/user。**

### 5.2 投机解码（SPEED Bench，bs=1）

```bash
vllm-bench \
  --backend openai \
  --base-url "http://${HEAD_ADDR}:8000" \
  --model moonshotai/Kimi-K3 \
  --dataset-name speed-bench \
  --speed-bench-config throughput_16k \
  --speed-bench-max-input-len 10240 \
  --speed-bench-category low_entropy \
  --output-len 1536 \
  --num-prompts 10 \
  --no-oversample \
  --max-concurrency 1 \
  --temperature 1.0 \
  --top-p 0.95 \
  --save-result \
  --save-detailed
```

**官方目标：TP8 331 tok/s/user，TP16 370 tok/s/user（3.14×）。**

> ⚠️ **注意这里官方用的是 `--temperature 1.0`，不是 0。**
> 本仓库 V4 的教训是「压测不发 temperature=0，投机解码接受率崩到 1%」——
> 但那是因为 V4 的 draft 是 greedy 而服务端默认采样。
> K3 的 DSpark 用 `draft_sample_method: probabilistic`，与温度采样匹配，
> 所以官方才敢用 1.0。**换任何采样参数前先确认 draft 侧的采样方式**。

### 5.3 官方接受率参考

| 任务类型 | 每步接受 token 数 |
|---|---|
| 代码等低熵任务 | ~4.73 |
| 创作等高熵任务 | ~2.61 |

实测接受率明显低于这个区间，先查采样参数是否与 draft 匹配。

---

## 6. 扩展：TP16 与 PD 分离 `[官方]`

### 6.1 TP16（4 节点）

把 §4.2 的 `--tensor-parallel-size 16 --nnodes 4`，`--node-rank` 依次 0/1/2/3。
这是拿到 **370 tok/s** 的配置。

### 6.2 PD 分离 —— **本轮不做**

官方只提了一句已验证拓扑（TEP8 prefill → DEP16 decode，NIXL 传 KV），
**没有公布任何 P:D 配比或吞吐数字**。相关分析与实验设计另存
[PD-BACKLOG.md](./PD-BACKLOG.md)，等主线 §0–§5 跑通、有了本环境基线之后再开。

> 不在本轮做的理由：那份设计是从 vLLM 的 GLM-5.2 PD 专文外推的，
> **跨模型搬运 PD 经验正是最容易踩坑的地方**（K3 是 KDA 混合缓存 + DSpark，
> GLM 是 DSA + MTP，PD 交接的细节完全不同）。先把官方给了数的路径复现干净。

---

## 7. Agentic 场景：KDA 缓存保留策略 `[官方]`

K3 的 KDA 状态**不随序列长度增长**，但单个状态很大（约等于几千 token 的 MLA cache）。
不能每个 token 都存，于是 vLLM 给了两条互补策略：

**区间保留** —— 每隔固定 token 数打一个 checkpoint：

```bash
export VLLM_PREFIX_CACHE_RETENTION_INTERVAL=32768   # 例：每 32K 一个
export VLLM_PREFIX_CACHE_RETENTION_INTERVAL=0       # 只保留 prompt 结尾
```

`0` 表示关闭周期性 checkpoint、只保留 prompt 末尾状态 —— **多轮对话为主的负载用这个最划算**，
因为下一轮通常从重放上一轮 prompt 开始，prompt 末尾正是最可能被复用的位置（vLLM 会自动识别保留）。

**Marconi 式选择性保留** —— 规则是「第二次命中才缓存」：第一次说明这个前缀存在，
第二次才说明它真的被共享。一次性前缀不占缓存，热前缀自动晋升。适合系统提示词 /
仓库快照 / 工具定义这类**不落在 prompt 边界上**的共享前缀。

---

## 8. 故障速查 `[待验证]`

| 现象 | 先查 |
|---|---|
| prefix cache 命中率为 0 | 是否漏了 `--enable-prefix-caching`（K3 默认关） |
| `tool_calls` 返回空 | 官方已知问题，与 prompt 和运行相关。加 schema 校验 + 重试，或改用 structured tool calling |
| 评测分数偏低 | 先看是不是被截断（K3 思考很长），调大 `max_tokens` 和 reasoning effort |
| 投机解码加速不明显 | 查采样参数是否与 draft 的 `probabilistic` 匹配；查接受率是否远低于 4.73 / 2.61 |
| 视觉相关报错 / TP 除不尽 | ViT `head_size=12`，TP8 除不尽，确认 `--mm-encoder-tp-mode=data`（默认开） |
| 加载超时 | `VLLM_ENGINE_READY_TIMEOUT_S=3600` |
| 跨节点性能腰斩 | 两节点是否在同一 NVL72 subblock |
| RAID 只有 256K | RAID 没挂，见 V4 的 SETUP 文档 md0→md127 陷阱 |

---

## 9. 官方精度基线 `[官方]`

| GSM8K | GPQA-Diamond | OCRBench | MMMU Pro Vision |
|---|---|---|---|
| 0.976 | 0.939 | 0.889 | 0.818 |

均为 vLLM OpenAI 兼容端点、max reasoning effort 下实测。

---

## 10. 验证记录

> **[待填]** 本环境尚未跑过。每跑通一轮，按 V4 runbook §10 的格式记录：
> 轮次 / 日期 / 是否清空环境从零重跑 / 实测数字 / 与官方差多少 / 撞到的文档缺陷。

| 轮次 | 日期 | 配置 | 实测 tok/s (bs=1) | vs 官方 | 备注 |
|---|---|---|---|---|---|
| — | — | TP8 无投机 | — | /111 | 待跑 |
| — | — | TP8 + DSpark | — | /331 | 待跑 |
| — | — | TP16 + DSpark | — | /370 | 待跑 |

---

## 11. 官方 roadmap（会影响后续复现基线）

- **DCP（Decode Context Parallelism）** —— 原型显示选定负载下比 TP8 高 **40%** 吞吐，即将上游
- **EPLB** 性能改进
- **Confidence-based scheduling** —— 用 DSpark 的 confidence head 剪掉不会被接受的 draft token
- **RL 支持** —— vLLM rollout 已加，正在对接 RL 生态
- 更广的 AMD ROCm 调优

---

## 来源

- vLLM day-0 博客（本文主要来源）：<https://vllm.ai/blog/2026-07-27-k3>
- 架构与 kernel 预告：<https://vllm.ai/blog/2026-07-22-kimi-k3-preview>
- 官方 recipes / Docker 镜像：<https://recipes.vllm.ai/moonshotai/Kimi-K3>
- 模型卡：<https://huggingface.co/moonshotai/Kimi-K3>
- DSpark draft：<https://huggingface.co/Inferact/Kimi-K3-DSpark>
