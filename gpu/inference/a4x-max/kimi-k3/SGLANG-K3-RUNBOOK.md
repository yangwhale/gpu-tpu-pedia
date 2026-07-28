# SGLang · Kimi K3 (2.8T) · GB300 NVL72 端到端 Runbook

> **本文定位**：跟 [VLLM-K3-RUNBOOK.md](./VLLM-K3-RUNBOOK.md) 对应的 SGLang 侧手册。
>
> **来源有三层，全文逐节标注，别混**：
> - `[本环境·已验证]` —— 来自本仓库 **[DeepSeek-V4-Pro SGLang runbook](../deepseek-v4/SGLANG-V4PRO-RUNBOOK.md)**，
>   这是本环境 SGLang-on-GB300 的 **Golden Truth**：端到端、反复验证十几遍、推倒重写过，
>   **凡与本文冲突，一律以它为准**。辅以 [DeepSeek-R1 3P2D 指南](../deepseek-v3/sglang-r1-nvfp4-gb300-3p2d-DEPLOY-GUIDE.md)（PD 与 RDMA 部分）。
>   这一层与模型无关，对 K3 同样成立 —— **不要因为「K3 是新模型」就重新发明**
> - `[K3官方]` —— 来自 SGLang K3 cookbook / LMSYS day-0 博客，**官方自己标着 Not Verified**
> - `[待测]` —— 本环境还没跑，留空等填
>
> **一句话**：环境和流程照我们自己的经验走，K3 专属参数照官方，两边冲突时**以我们踩过的为准**。

---

## ⚠️ 最重要的三条（每条都花过我们一整轮部署）

### 一、`--mamba-full-memory-ratio` 就是 K3 版的 `swa-full-tokens-ratio`

V4 那轮最贵的教训：**KV 池预算划错，一个参数值决定 54% 的吞吐**，而所有健康信号都是绿的。

K3 上同一个位置的旋钮是 `--mamba-full-memory-ratio`（默认 `0.86`），
它划的是 **KDA 状态池 vs MLA KV 池**的比例。官方 cookbook 甚至配了个计算器，
说明这个值同样是**跟负载绑定、不是常数**。

**判据照搬 V4 的方法**：跑一轮看两个池谁先到 0.9+，谁先满就给谁加预算，目标是两边同时落在 0.88–0.93。
K3 上启动后要回读的是 `max_total_num_tokens`（KV 侧）和**准入请求上限**（状态侧）。

> V4 那次我在 ratio 设错的情况下测出「batch 上不去」，把原因全归给了 prefill —— 方向没错，
> **权重完全估错**。K3 上别重复这个错误：**调参之前先把两个池的占用读出来**。

### 二、`--max-running-requests` 不显式设，开投机后会被重置成 48

`[K3官方]` cookbook 原话：投机解码开着时，SGLang 把未设置的 `--max-running-requests`
**重置为 48**。48 并发在 GB300 上等于什么都没跑。

这条跟 V4 的「三个健康信号全绿但只有 1/3 算力」是同一类陷阱：**不报错，只是慢**。

### 三、`pkill -9` 一个满载的 SGLang 进程会泄漏 ~97 GB 显存/卡

`[本环境·已验证]` V4 实测：`pkill -9` 之后 pod 里全是 zombie、RSS=0，
但 `nvidia-smi` 每张卡还挂着 97 GB。下一次启动必 `Not enough memory`，
**而且报错会把人往「调大 mem-fraction」这个完全相反的方向带**。

**正确做法：删 pod 让 StatefulSet 重建**（实测 56 秒显存归零），不要 pkill。

> 推测原因是 GB300 走 MNNVL/IMEX，导出到 fabric 的显存在 SIGKILL 时不走正常回收路径，
> 而容器 PID 1 是 `sleep infinity` 不 reap zombie。**这个泄漏会污染后续所有实验**——
> V4 那轮我因此连续三次误判「MTP steps=2 内存不够」。

---

## 0. TL;DR

| 项 | 值 |
|---|---|
| 目标配置（起步） | **Unified · TP8 · DSPARK**，2 节点 8 卡 |
| 硬件口径 `[K3官方]` | GB300 = **2×4**（每节点 4 卡），TP8 / DCP8，MNNVL 与 cuMem 自动识别 |
| 模型 | `moonshotai/Kimi-K3`，MXFP4 权重，**约 1.4 TB**，放 Local SSD RAID |
| Draft | **`RadixArk/Kimi-K3-DSpark`** ⚠️ 跟 vLLM 用的 `Inferact/...` **不是同一个，别混** |
| 关键参数 | `--mamba-full-memory-ratio`（见文首第一条）、`--max-running-requests`（见第二条） |
| 官方数 `[K3官方]` | bs=1 无投机 **~113 tok/s**；+DSPARK **~423 tok/s**；PD 前沿 **2,808 tok/s/GPU** |
| 本环境实测 | **`[待测]`** |

**最短路径**：§1 前置 → §2 起 fleet → §3 RAID + 模型 → §4 分发 → §5 启动 → §6 就绪判据 →
§7 压测（**第一轮当 warmup 丢掉**）。撞到怪事先翻 **§10**，那里每一行都是真踩过的。

---

## 1. 前置条件 `[本环境·已验证]`

```bash
# ① 同域节点数 —— PD / 跨节点 TP 的 KV 走 MNNVL，跨域会退化到 RDMA 并显著变慢
kubectl get nodes -l cloud.google.com/gke-nodepool=gb300-pool-0002 --no-headers | wc -l

# ② DRA GPU driver
kubectl get pods -A | grep -c dra-driver-nvidia-gpu    # >0

# ③ 没有孤儿 ComputeDomain 占着 channel
kubectl get computedomain -A
```

**同域是硬要求。** 同 nodepool 通常同域。

---

## 2. 部署 pod fleet `[本环境·已验证]`

复用 V4 的 manifest 骨架，**三个关键设计一个都不能省**：

1. **用 StatefulSet，不要裸 pod + `nodeName`** —— DRA 的 ComputeDomain channel 必须由 scheduler 预留，
   `nodeName` 绕过调度器会 `FailedPrepareDynamicResources`
2. **StatefulSet 的稳定 DNS 名是白赚的** —— 跨节点要 `--dist-init-addr`，pod IP 每次重建都变
3. **两个 DRA claim 都要** —— mrdma（8 张 CX-8）+ ComputeDomain channel（MNNVL）

**pod 内存 limit**：`[本环境·已验证]` decode 600Gi 够（KV 在 HBM）；
**长上下文 prefill 要 700Gi**（激活峰值 + 加载缓冲）。节点 909Gi allocatable。
K3 是 1M 上下文，**建议 prefill 直接给 700Gi 起步**。

> ⚠️ `[本环境·已验证]` **16 pod 同时申请 DRA 会滞后**，部分 pod 卡 `ContainerCreating`
> 报 `ResourceClaim not created yet`。删掉卡住的 pod + `apply` 重触发即可，可能重试 1–2 轮。

---

## 3. RAID 与模型 `[本环境·已验证]`

### 3.1 ⚠️ 先查 RAID，再查模型（`md0` → `md127` 陷阱）

**「模型缺失」十有八九不是模型没拷，是那台的 Local SSD RAID 根本没挂上。**

```bash
for i in $(seq 0 7); do
  printf "sgl-%s: " $i
  kubectl exec sgl-$i -- df -h /mnt/ssd | tail -1 | awk '{print $2, $5}'
done
# 正常 12T；看到 256K 100% → RAID 没挂
```

**根因**：节点重启后内核把已存在的阵列自动组装成 `/dev/md127` 而不是 `/dev/md0`，
而 DaemonSet 脚本 `grep -q "md0"` 判定「没有阵列」→ 去 create → 盘已被占 → 连环失败。

**表现极具迷惑性**：hostPath 用 `DirectoryOrCreate`，kubelet 会在只读根文件系统上建出目录、落到 tmpfs，
**pod 正常起、`/mnt/ssd` 存在、但只有 256K**，写模型时静默失败（`curl -o` 写出 0 字节，退出码还是 0）。

修复见 [gb300-local-ssd-raid0-SETUP.md](../deepseek-v4/gb300-local-ssd-raid0-SETUP.md)，
要点是**动态识别 md 号**：`MD=$(awk '/^md[0-9]+ : active/{print $1; exit}' /proc/mdstat)`，
且**有 fs 就别格式化**。挂好后不用重启 pod（`mountPropagation: HostToContainer` 会传播进去）。

### 3.2 模型 `[本环境·已验证]` + `[待测]`

**约 1.4 TB**，比 V4-Pro 的 806G 大 70%。**必须放 Local SSD RAID，不放内存盘**（RAM 留给别的）。

```bash
# 校验（分片数 [待测]，跑通后填）
for i in $(seq 0 7); do
  echo -n "sgl-$i: "
  kubectl exec sgl-$i -- bash -c "du -sh /mnt/ssd/Kimi-K3 | cut -f1; \
    ls /mnt/ssd/Kimi-K3/*.safetensors | wc -l" | tr '\n' ' '; echo
done
```

> ⚠️ **每次重建 fleet 都要重新校验，哪怕上一轮刚跑完。** 节点数比 pod 数多时调度器会换节点，
> 上一轮的空闲节点这轮可能被占，那个 pod 就是空的。V4 审计轮 1 就是这么抓到 `sgl-0` 模型缺失的。

**缺失时怎么补**（优先级从高到低）：

**① pod→pod 直传** —— 集群内网远快于 GCS，V4 实测 **3.6 GB/s**（806G 约 4 分钟），
K3 的 1.4 TB 约 7 分钟。源端 `python3 -m http.server`，目标端 `xargs -P 6` + `curl`。

**② 从 GCS 拉** —— ⚠️ **`lmsysorg/sglang` 镜像里没有 `gcloud`**。要么先装，
要么用镜像里已有的 `google-cloud-storage` Python SDK，要么走 ①。

---

## 4. 分发启动脚本 `[本环境·已验证]`

```bash
for i in $(seq 0 7); do
  kubectl exec -i sgl-$i -- bash -c "cat > /tmp/serve.sh && chmod +x /tmp/serve.sh" < scripts/sgl-serve-tp8-dspark.sh
done
kubectl exec sgl-0 -- wc -l /tmp/serve.sh   # ★ 必须校验非空
```

> ⚠️ **必须 `kubectl exec -i`**。少了 `-i` 时 stdin 不透传，容器里得到**空文件，且不报错**。
> `kubectl cp` 同理会静默失败，cp 完必须 `wc -l`。

---

## 5. 启动 `[K3官方]` 参数 + `[本环境·已验证]` 启动纪律

### 5.1 起步配方：Unified · TP8 · DSPARK（GB300 2×4）

```bash
sglang serve \
  --trust-remote-code \
  --model-path /mnt/ssd/Kimi-K3 \
  --tp-size 8 \
  --disable-custom-all-reduce \
  --enable-symm-mem \
  --mem-fraction-static 0.85 \
  --reasoning-parser kimi_k3 \
  --tool-call-parser kimi_k3 \
  --mamba-full-memory-ratio 0.86 \
  --max-running-requests <按目标并发设，别留空！> \
  --host 0.0.0.0 --port 30000 \
  --speculative-algorithm DSPARK \
  --speculative-draft-model-path RadixArk/Kimi-K3-DSpark \
  --speculative-dspark-block-size 7 \
  --enable-linear-replayssm-spec
```

### 5.2 参数逐条解释

| 参数 | 为什么 | 来源 |
|---|---|---|
| `--mamba-full-memory-ratio 0.86` | KDA 状态池 vs MLA KV 池的划线。**见文首第一条** | `[K3官方]` |
| `--max-running-requests` | **不设会被重置成 48**。见文首第二条 | `[K3官方]` |
| `--enable-linear-replayssm-spec` | ReplaySSM：存原始输入而非快照，draft window 512 KB → 16 KB（约 32×） | `[K3官方]` |
| `--speculative-dspark-block-size 7` | 提 7 个 draft token | `[K3官方]` |
| `--disable-custom-all-reduce --enable-symm-mem` | GB300 对称内存路径 | `[K3官方]` |
| **不要设** `--moe-runner-backend` | Blackwell 上自动选 FlashInfer MXFP4（trtllm-gen SiTU）；H100/H200 才 pin Marlin | `[K3官方]` |
| **不要设** 三个 attention backend | K3 把 prefill / decode / verify 作为一组解析；**设了任何一个就取消其余的自动解析** | `[K3官方]` |

> ⚠️ **「不要设」这类建议要留个心眼。** V4 那轮最阴的坑就是**依赖默认值**：
> 旧文档没写 `--moe-runner-backend deep_gemm`，nightly 镜像更新后 `auto` 改选了 flashinfer，
> 整条 megamoe 路径直接崩。**写文档时能跑 ≠ 三个月后能跑。**
> 建议：先按官方不设，但**从启动日志里把实际选中的 backend 记下来**，写进 §11 验证记录。
> 一旦哪天性能异常，第一件事就是对这行日志。

### 5.3 启动纪律 `[本环境·已验证]`

```bash
# ① 先单个冒烟，再批量 —— 8 个一起错的排查成本远高于先验 1 个
kubectl exec sgl-0 -- bash -c "setsid nohup bash /tmp/serve.sh > /tmp/srv.log 2>&1 </dev/null & sleep 4"

# ② 一个 pod 只启一次。反复启动会堆多个 python 进程叠加 host 内存 → OOM
#    → 容器重启清空 /tmp（脚本和日志一起消失）→ 更乱
#    重启前先清干净：
kubectl exec sgl-0 -- bash -c "pkill -9 -f 'sglang[.]launch_server'"   # ★ 括号必须有
```

> ⚠️ **`pkill -f` 会自杀。** `kubectl exec sgl-N -- bash -c "pkill -9 -f sglang.launch_server; ..."`
> 这条命令行**自身就含有那串字符**，于是把自己杀了（exit 137，后面的语句一条都不执行，而且不报错）。
> 用 `'sglang[.]launch_server'` 括号转义，或 `pkill -9 python`。**本项目在两个框架上踩了四次。**

> ⚠️ **`setsid nohup ... &` 后面要 `sleep 4`。** `kubectl exec` 返回后会关掉 exec 流，
> 子进程会在 detach 完成前被带走 —— 表现是**日志文件根本不生成，且完全不报错**。
> V4 压测时裸 `for` 循环起 14 路，实测只有 6 路活下来。

### 5.4 时序预期 `[待测]`

| 阶段 | V4-Pro (806G) 实测 | K3 (1.4 TB) 预期 | 本环境实测 |
|---|---|---|---|
| 权重加载 | 4–5 min | 更久 | `[待测]` |
| CUDA graph capture | 3–5 min | ? | `[待测]` |
| **单实例总计** | **8–12 min** | ? | `[待测]` |

> `[本环境·已验证]` **别在 5 分钟时下结论。** V4 审计轮 1 就是因为文档写 180s、
> 实际要 300s，误判成「起不来」。
>
> `[本环境·已验证]` **decode 起来前会刷 `DeepGEMM warmup: 0/65536`，
> 初始 ETA 显示几十小时是误导** —— JIT 一热就到 ~1000 it/s，实际约 1 分钟。别被吓到。

---

## 6. 就绪判据 `[本环境·已验证]` 的方法 + `[待测]` 的具体判据

**V4 最贵的一课：三个看起来最自然的健康信号全是绿的，系统却只有 1/3 的算力。**

| 层 | 判据 | 能不能信 |
|---|---|---|
| ① | `nvidia-smi` HBM 高 | ❌ SGLang **先预分配显存池**，权重可能随后加载失败 |
| ② | 日志出现 `Load weight end` | ❌ 之后还要建 ZMQ / 起 scheduler / 注册，任一步崩都不改这行 |
| ③ | **服务真的能出 token** | ✅ |

```bash
# 最低限度的端到端验证 —— 不做这一步，压测可能全 0
kubectl exec sgl-0 -- curl -s localhost:30000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"Kimi-K3","messages":[{"role":"user","content":"The capital of France is"}],"max_tokens":32}'
```

**`[待测]`**：K3 启动日志里应该出现的关键行（MoE runner 选择 / KDA kernel / DCP backend），
跑通后填进来，作为「跑对了」而不只是「跑起来了」的判据。

> `[本环境·已验证]` **存活判据和就绪判据要用相反的信号**：
> 判「起来没」用注册/服务响应，判「还活着没」用 `nvidia-smi` 显存。
> V4 那轮我用注册数判存活，误判「14 台全程在线」，实际早就全崩了。

> `[本环境·已验证]` **`/tmp/srv.log` 是二进制**（含 NUL），`grep` 会直接返回 `binary file matches`。
> 所有分析先 `tr -d '\000'`，或用 `grep -a`。

---

## 7. 压测 `[本环境·已验证]` 方法

### 7.1 ★ 第一轮必须当 warmup 丢掉

`[本环境·已验证]` **重启后首轮比热态低 6.5–7%，不是噪声，高度可复现**：

| | 冷（重启后第 1 轮） | 热（第 2 轮） | 差 |
|---|---|---|---|
| V4 审计轮 1 | 8,520 | 9,118 | +7.0% |
| V4 审计轮 2 | 8,552 | 9,108 | +6.5% |
| 两轮之间偏差 | ±0.2% | ±0.1% | — |

冷热**各自都稳定到 ±0.2%**，说明是确定性的 warmup 成本。**这笔开销在进程内**
（首次遇到各 M shape 时的 kernel 选择 / autotune），**跟磁盘 JIT 缓存在不在无关**——
V4 验证过：`SGLANG_DG_CACHE_DIR` 在节点盘、跨 pod 保留，冷跑照样低 6.5%。

**流程：跑两轮，报第二轮。**

> `[本环境·已验证]` V3 那边还撞到过更极端的：sweep 第一档撞上首次 JIT 编译，
> **TTFT 冲到 67s、总吞吐掉到 438**，看起来像配置全错。第二遍 warm 就正常了。

### 7.2 压并发，别用小 conc 汇报数字

`[本环境·已验证]` V3 的教训：conc=8 只用到 DEP8 容量的 ~3%。K3 上同理。

⚠️ 但注意 K3 的 `--max-running-requests` 陷阱（文首第二条）——**先确认它不是 48**。

### 7.3 口径必须先对齐

`[本环境·已验证]` V4 的教训：官方数字往往是 **output tok/s ÷ decode-GPU 数**，
分子只算 output、分母只算 decode 卡。**同一次测量换个口径能差一倍**
（V4 的 dep16 实验：一个口径 11,880 超标，另一个 5,270 腰斩）。

SGLang K3 官方那几个数的口径：

| 数字 | 口径 |
|---|---|
| ~113 / ~423 tok/s | **单用户 bs=1 decode**（前者无投机，后者 DSPARK） |
| 2,808 tok/s/GPU | **PD 前沿**，1× PP8 prefill 喂 1× TP8 decode |
| 2,633 tok/s/GPU | 2× PP8 prefill 喂 2× DCP8 decode |
| 541 tok/s | agentic 回放，48 并发会话，DCP8 |

> ⚠️ 求和多路结果的前提是**各路时间窗重叠**。V4 踩过：某几路晚启动几十秒，
> 各自吞吐都是在「独占更多算力」的窗口里测的，直接相加会显著高估。
> **核对各路 `Benchmark duration` 相近再求和。**

---

## 8. 官方指标（背景） `[K3官方]`

### 8.1 单用户 decode（bs=1）

| 阶段 | tok/s |
|---|---|
| bring-up 基线（Marlin W4A16 MoE） | 44.3 |
| 15 级 kernel 优化之后 | **112.5** |
| **+ DSPARK 投机解码** | **~423** |

那 15 级优化里最大的四块：**通信融合 +27.6**（NVLS in-switch reduction，
把 residual add 和 RMSNorm 塞进 collective 里）、**launch/copy 消除 +19.9**、
**NVIDIA 定制 kernel +10.3**、**overlap 与 prologue 融合 +10.4**。

> 官方总结的方法论值得记：**all-reduce 是同步点，省一微秒一比一变成 step 时间；
> 而在别的流里 overlap 的 kernel 只按十分之一折算。写 kernel 前先在 trace 里
> 确认它在不在关键路径上——这是整轮优化里杠杆最高的习惯。**

### 8.2 PD 前沿（在 2×4 GB300 上测的，跟我们硬件一致）

| 拓扑 | 每 GPU 吞吐 |
|---|---|
| **1× PP8 prefill → 1× TP8 decode**（fp4 arm） | **2,808 tok/s/GPU** |
| 2× PP8 prefill → 2× DCP8 decode | 2,633 tok/s/GPU |
| 1 prefill 喂 2 / 3 / 4 个 decode 实例 | 拿总吞吐换单用户速度，**推到 116+ tok/s/user** |

### 8.3 三个架构结论

**① prefill 用深度 PP，不用 TEP。** `--pp-size 8 --tp-size 1`。
实测 **PP8×TP1 约为 TEP8 上限的 1.7 倍**，TTFT 还更低；单个 PP8 prefill 节点
有 TEP8 节点 **1.45–1.72 倍**的 prefill 能力。理由是 TP 每层结尾都有 AllReduce 挡着，
PP 的 stage 间传递能被下一个 chunk 的计算盖住（K3 上隐藏了 91%）。

> ⚠️ **必须用满 8 个 stage**。浅切（PP4×TP2）还要付 TP2 的 all-reduce，benchmark 上打不过 TEP8。
> 而且 **DSPARK 与 PP 互斥**（要求 `pp_size == 1`）。

**② decode 用 DCP 按 token 位置切 KV。** `--dcp-size 8`。
MLA 只有一个 KV head，TP 切不动、每 rank 存全量副本。DCP8 把**逻辑 KV 从 1.5M 拉到 12.2M token（约 7.9×）**。
agentic 回放实测：TP8 在 16 并发就崩，DCP8 扛到 **48 并发 / 541 tok/s**。

**③ KDA 状态池是并发天花板。** **DP、EP、DCP 都不切它** ——
只有 attention-TP 宽度、SSM dtype、cache 策略能改每 GPU 的账单。MLA KV 反而好压（fp8）好去重（DCP）。

---

## 9. 本环境待测指标 `[待测]`

跑通后填。**口径按 §7.3 对齐，冷轮丢掉只报热轮。**

### 9.1 单实例 TP8

| 场景 | 官方 | 本环境实测 | 达成率 |
|---|---|---|---|
| bs=1 无投机 | ~113 tok/s | | |
| bs=1 + DSPARK | ~423 tok/s | | |
| 高并发 total tok/s | — | | |
| TPOT 中位 | — | | |
| TTFT 中位 | — | | |

### 9.2 内存池划分（对应文首第一条）

| `--mamba-full-memory-ratio` | KDA 状态池占用 | MLA KV 占用 | 准入上限 | 吞吐 | 备注 |
|---|---|---|---|---|---|
| 0.86（默认） | | | | | |
| | | | | | |

> **调参判据**：哪个先到 0.9+ 就给哪个加预算，目标两边同时 0.88–0.93。

### 9.3 DCP 消融

| 配置 | 逻辑 KV 容量 | 最大并发 | 吞吐 | ITL |
|---|---|---|---|---|
| TP8（无 DCP） | | | | |
| TP8 + DCP8 | | | | |

### 9.4 PD 分离（**K3 官方有真实数据，可以直接对标**）

| 拓扑 | 官方 | 本环境实测 |
|---|---|---|
| 1× PP8 prefill → 1× TP8 decode | 2,808 tok/s/GPU | |
| 2× PP8 prefill → 2× DCP8 decode | 2,633 tok/s/GPU | |
| 1 prefill → 2/3/4 decode | 116+ tok/s/user | |

### 9.5 vs vLLM 同环境对比

| | vLLM | SGLang |
|---|---|---|
| bs=1 无投机（官方） | 111 (TP8) | ~113 |
| bs=1 + 投机（官方） | 331 (TP8) | ~423 |
| bs=1 无投机（本环境） | | |
| bs=1 + 投机（本环境） | | |
| PD 每 GPU（本环境） | | |

---

## 10. 故障速查

**上半部分继承自 V4 / V3，与模型无关，对 K3 同样成立；下半部分是 K3 专属。**

### 10.1 环境与流程类 `[本环境·已验证]`

| 现象 | 根因 | 处理 |
|---|---|---|
| **跑通了但吞吐只有一半**，健康信号全绿 | 实际在服务的实例数 < 预期 | 用 §6 的端到端判据重查，别信显存和日志 |
| `/mnt/ssd` 只有 256K | RAID 没挂（`md0`→`md127`） | §3.1，动态识别 md 号 |
| 容器里脚本是空文件 | `kubectl exec` 少了 `-i`；或 `kubectl cp` 静默失败 | 加 `-i`，之后必须 `wc -l` 校验 |
| **日志文件根本不生成、也不报错** | exec 关流太快，`setsid` 还没 detach 完 | exec 里加 `sleep 4` + 外层校验重试 |
| `kubectl exec` 自己 exit 137、后续语句没执行 | `pkill -f <pat>` 匹配到 exec 自身命令行 | `'sglang[.]launch_server'` 括号转义 |
| **换个参数就 `Not enough memory`，改回原参数也起不来** | 上次 `pkill -9` 泄漏了 ~97 GB/卡 | **删 pod 让 StatefulSet 重建**（56s），别 pkill |
| 重建 fleet 后某 pod 模型目录是空的 | 调度器换了节点 | §3.2 每次重建都校验，缺就 pod→pod 补 |
| 容器里 `gcloud: command not found` | `lmsysorg/sglang` 镜像不含 gcloud | pod→pod 直传，还快 4× |
| `FailedPrepareDynamicResources` | 裸 pod + `nodeName` 绕过 scheduler | 用 StatefulSet |
| pod 卡 `ContainerCreating` / `ResourceClaim not created yet` | 多 pod 同时申请 DRA，controller 滞后 | 删卡住的 pod + `apply` 重触发，重试 1–2 轮 |
| `OOMKilled` (exit 137) 加载时 | host 侧加载缓冲峰值超限 | 内存 request ≥600Gi，长上下文 prefill 700Gi；**一 pod 只启一次** |
| pod `Evicted` DiskPressure | 大镜像顶爆 boot 盘 | fresh 节点池 + 删重建 |
| `grep` 返回 `binary file matches` | `srv.log` 含 NUL | 先 `tr -d '\000'` 或用 `grep -a` |
| 首轮压测数字异常低（TTFT 几十秒） | 首次 JIT 编译 | §7.1，跑两轮报第二轮 |
| `DeepGEMM warmup 0/65536` ETA 几十小时 | JIT 未热时的误导性估算 | 实际约 1 分钟，别被吓到 |

### 10.2 PD 相关 `[本环境·已验证]`，K3 上大概率同样成立

| 现象 | 根因 | 处理 |
|---|---|---|
| 单请求 60s 超时 / `KVTransferError` | nixl 走 RoCE 在 GKE 上调不通（RoCE v2 over IPv6，netdev 名 `gpuNipvlanM`） | **改走 NVLink**：`--disaggregation-transfer-backend mooncake` + `SGLANG_MOONCAKE_CUSTOM_MEM_POOL=NVLINK` + `MC_FORCE_MNNVL=1`。成功标志是 decode 日志出现 `Using cross-node NVLink transport (MC_FORCE_MNNVL)` |
| `NIXL_ERR_BACKEND` / RDMA backend 创建失败 | 官方 Ubuntu 镜像缺 CX-8 的 mlx5 verbs | 装 `doca-ofed-userspace` |
| `Decode handshake failed` | **PD 两侧 `--context-length` 必须一致** | 改就两边一起改 |
| 重启 decode 后 prefill 全崩 | disagg 对端消失，prefill scheduler 自杀 | **躲不掉**，做 decode 实验就要预算上 prefill 的重建时间 |
| 重启 server 后单条 curl 挂死 | router/frontend 缓存了旧 instance 连接 | **router 必须跟着一起重启**，然后跑一次 e2e 确认 |

### 10.3 K3 专属 `[K3官方]` + GitHub

| 现象 | 根因 | 处理 |
|---|---|---|
| **并发上不去，卡在 48** | 开投机后 `--max-running-requests` 未设被重置成 48 | 显式设。见文首第二条 |
| 并发上不去（非 48） | **KDA 状态池是天花板**，DP/EP/DCP 都不切它 | 调 attention-TP 宽度 / SSM dtype / cache 策略，别指望加 DP |
| DSPARK 崩 `TypeError: 'NoneType' object is not callable in top_k_renorm_prob` | ⚠️ **SGLang 已知 open bug** [#32569](https://github.com/sgl-project/sglang/issues/32569)（2026-07-27 开，尚未关闭） | 先跑 NOSPEC 基线；确认镜像是否已含修复 |
| 上了 DCP 之后报错 / decode graph 不对 | **DCP 下不能用 `--enable-symm-mem`**（为 decode graph 正确性强制禁用） | 去掉该 flag |
| 长上下文 prefill 想用 DSPARK | **DSPARK 要求 `pp_size == 1`**，与 Deep PP 互斥 | 二选一 |
| HiCache 与 DCP 组合异常 | host 层还不完全 DCP-aware：**L3 一律、L1+L2 在开投机时，都要去掉 DCP flag 跑纯 TP** | 按 cookbook 提示降级 |
| PD 只有 decode 注册上 | `--prefill` 后面那个位置参数 `8998` 必须等于 `--disaggregation-bootstrap-port` | 对齐两者 |
| PD decode 并发行为诡异 | `--disaggregation-decode-extra-slots` 未固定：**低于 32 请求时默认两倍 batch，高于则为零** | 显式 pin 住 |
| 视觉输入报错 | 开源 serving 契约**只支持图像**，拒绝视频和音频 | — |
| 评测分数偏低 | K3 思考很长，多半是**被截断**不是答错 | 放大 `max_tokens`、调高 `reasoning_effort` |

---

## 11. 验证记录 `[待填]`

> 按 V4 runbook §10 的格式记：轮次 / 日期 / **是否清空环境从零重跑** / 实测数字 / 与官方差多少 / 撞到的文档缺陷。

| 轮次 | 日期 | 是否从零 | 配置 | 实测 | vs 官方 | 撞到的坑 |
|---|---|---|---|---|---|---|
| — | — | — | TP8 NOSPEC | — | /113 | |
| — | — | — | TP8 + DSPARK | — | /423 | |
| — | — | — | PD PP8→TP8 | — | /2,808 | |

**启动日志关键行**（跑通后填，用于日后判断「跑对了」）：

```
[待填] MoE runner backend 实际选中：
[待填] Attention backend 实际解析：
[待填] max_total_num_tokens：
[待填] 准入请求上限：
```

> **为什么一定要做从零审计**：V4 那份 runbook 写完后清空环境照文档重跑了 3 轮，
> 抓出 8 个文档缺陷，其中 2 个是「上一轮跑通了、写下来了、看着也对」的东西 ——
> 有一条甚至是我自己刚写下的错误建议，靠 review 文档发现不了。
> 还有一次 **n=1 的观察被写成了因果**（「错开 8s 能降低崩溃率」），第二轮直接证伪。

---

## 12. 与 vLLM 侧的技术路线差异 `[K3官方]`

同一个模型，两家的最优解不一样，**别把一边的经验直接搬到另一边**：

| | vLLM | SGLang |
|---|---|---|
| prefill | **TEP8**（attention TP + MoE EP） | **PP8×TP1 深度流水**（实测 1.7× TEP8） |
| decode KV 去重 | 靠 PD + 分页 | **DCP 按 token 位置切**（逻辑容量 7.9×） |
| 投机时的 KDA 状态 | 引擎内处理 | **ReplaySSM**：存输入不存快照（32×） |
| MoE backend | `deep_gemm_mega_moe`(DEP) / `flashinfer_trtllm`(TP>1) | Blackwell 自动选 FlashInfer MXFP4；短上下文批量用 MegaMoE |
| Draft 模型 | `Inferact/Kimi-K3-DSpark` | **`RadixArk/Kimi-K3-DSpark`** |
| 关键内存旋钮 | `--kv-cache-dtype` + `--max-model-len` | **`--mamba-full-memory-ratio`** |
| 公布 PD 数据 | ❌ 无 | ✅ 2,808 tok/s/GPU |
| 命令验证状态 | 给了实测 reproduce recipe | **cookbook 每格标 Not Verified** |

> **两个 draft 模型不是同一个，别混用。**

---

## 来源

- SGLang / Miles day-0 博客：<https://www.lmsys.org/blog/2026-07-27-kimi-k3-day0-support>
- SGLang K3 cookbook：<https://lmsysorg.mintlify.app/cookbook/autoregressive/Moonshotai/Kimi-K3>
- 本仓库 **[DeepSeek-V4-Pro SGLang runbook](../deepseek-v4/SGLANG-V4PRO-RUNBOOK.md)** —— **Golden Truth**，
  端到端验证十几遍并重写过，本文环境与流程部分全部继承自它
- 本仓库 [DeepSeek-R1 3P2D 部署指南](../deepseek-v3/sglang-r1-nvfp4-gb300-3p2d-DEPLOY-GUIDE.md)（PD 与 RDMA 经验）
- 模型卡：<https://huggingface.co/moonshotai/Kimi-K3>
