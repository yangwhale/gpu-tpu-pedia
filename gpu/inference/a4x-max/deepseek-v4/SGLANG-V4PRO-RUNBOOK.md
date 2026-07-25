# SGLang · DeepSeek-V4-Pro · GB300 NVL72 复现 Runbook

> **本文定位**：**只讲怎么跑通**的操作手册。所有命令可直接复制，所有引用的脚本/manifest 都已入库（`manifests/` + `scripts/`）。
> 原理分析、benchmark 演进史、被推翻的假设 → 见 `sglang-v4-gb300-benchmark.md`（旧版，按时间顺序记录，信息全但需自行导航）。
>
> **文档状态约定**：每节标注 `[已验证]` / `[待验证]` / `[已知问题]`，**不用星标**。标 `[已验证]` 的都在本文末 §10 有实跑记录。

---

## ⚠️ 最重要的一条：模型与 MoE backend 必须配套

**这是复现失败的头号原因，且报错完全指不到根因。** 两套配方互斥，混用必挂：

| checkpoint | MoE runner backend | a2a backend | 备注 |
|---|---|---|---|
| `deepseek-ai/DeepSeek-V4-Pro`（**官方原装**，FP4 MoE + FP8 attn） | `deep_gemm` | **`megamoe`** | 高性能路径，旧文档 8,993 就是这个 |
| `nvidia/DeepSeek-V4-Pro-NVFP4`（NVFP4 MoE + FP8 attn） | **`flashinfer_trtllm_routed`** | 不用 megamoe | megamoe 没有为它注册融合 kernel |

**混错的两种典型报错**：

- NVFP4 权重 + megamoe → `NotImplementedError: Runner backend FLASHINFER_TRTLLM_ROUTED requires a fused func for a2a backend megamoe, but none is registered`
- NVFP4 权重 + `deep_gemm` + megamoe → `RuntimeError: The size of tensor a (384) must match the size of tensor b (48)`（384 = `n_routed_experts`，48 = 384/ep_size，随 ep 缩放）

> **先确认你手上是哪份权重**：`python3 -c "import json;print(json.load(open('<model>/config.json')).get('quantization_config'))"`。
> 旧文档把这两套配方分散在 §1（checkpoint 说明）和 §13.A（启动脚本）两处，中间无任何交叉提示——照 §13.A 抄、配 NVFP4 权重，必然失败。

---

## 0. TL;DR

| 项 | 值 |
|---|---|
| 目标配置 | **14 prefill（每个 dep4）+ 1 decode（dep8 跨 2 节点）+ MTP** |
| 节点需求 | 16 节点 GB300，**必须同一 NVL72 域**（本文用 `gb300-pool-0002`） |
| 模型 | `/mnt/ssd/DeepSeek-V4-Pro`（**官方原装**，806G，节点本地 SSD）—— 不是 `-NVFP4` 那份 |
| 镜像 | `lmsysorg/sglang:nightly-dev-cu13-20260720-b3570a45` |
| 编排 | Dynamo（`dynamo.sglang` worker + `dynamo.frontend`）+ NATS + etcd |
| 参考成绩 | 旧文档基线 8,993；**本文实测 9,168**（conc600 同工作点）output tok/s ÷ decode-GPU = 官方 11,200 的 82%。拉高并发最多到 9,745（87%）但 TTFT 飙到 253s，不可用。见 §10.2 |

**为什么是 14 prefill**：实测 14→16 prefill 只涨 2%（8,809 → 8,993），已收敛。14 是性价比拐点。

---

## 1. 前置条件 `[已验证]`

```bash
# ① 同域节点数（16 起）
kubectl get nodes -l cloud.google.com/gke-nodepool=gb300-pool-0002,team=yangwhale --no-headers | wc -l

# ② Dynamo 基础设施（服务发现，worker 靠它自注册）
kubectl get pods | grep -E 'dynamo-nats|dynamo-etcd'   # 两个都要 Running

# ③ DRA GPU driver
kubectl get pods -A | grep -c dra-driver-nvidia-gpu    # >0
```

**同域是硬要求**：PD 分离的 KV 传输走 mooncake + MNNVL，跨域会退化到 RDMA 并显著变慢。同 nodepool 通常同域。

---

## 2. 部署 pod fleet `[已验证]`

```bash
kubectl apply -f manifests/sgl-fleet.yaml
kubectl get pods -l app=sgl -w        # 等 16/16 Running（实测 ~2min）
```

**角色分配**（由跑哪个脚本决定，pod 本身同构）：

| Pod | 角色 |
|---|---|
| `sgl-0` / `sgl-1` | decode，dep8 跨 2 节点（rank 0 / rank 1）|
| `sgl-2` … `sgl-15` | prefill，各自 dep4 |

**manifest 的三个关键设计**：

1. **用 StatefulSet，不要裸 pod + `nodeName`**。DRA 的 ComputeDomain channel 必须由 scheduler 预留，`nodeName` 绕过调度器会导致 `FailedPrepareDynamicResources`。
2. **StatefulSet 送的稳定 DNS 名是白赚的**。decode 跨节点要 `--dist-init-addr`，有稳定名比每次查 IP 可靠（pod IP 每次重建都变）。
3. **两个 DRA claim 都要**：`sgl4-mrdma`（8 张 CX-8，mooncake KV 传输）+ `sgl4-ch`（ComputeDomain channel，MNNVL）。

---

## 3. 检查 / 铺设模型 `[已验证]`

本文用的是**官方原装** `deepseek-ai/DeepSeek-V4-Pro`（806G，64 个 safetensors），不是 `nvidia/...-NVFP4`。理由见文首告警——只有原装权重能走 megamoe 高性能路径。

```bash
# 校验：每节点 806G、64 个分片
for i in $(seq 0 15); do
  echo -n "sgl-$i: "
  kubectl exec sgl-$i -- bash -c "du -sh /mnt/ssd/DeepSeek-V4-Pro 2>/dev/null | cut -f1; \
    ls /mnt/ssd/DeepSeek-V4-Pro/*.safetensors 2>/dev/null | wc -l" 2>/dev/null | tr '\n' ' '
  echo
done
```

模型在**节点本地 SSD**（hostPath `/mnt/disks/raid/0`），删 pod 不丢。若缺失，从 GCS 并行拉（实测 ~850MB/s，全 fleet 并行约 25min）：

```bash
for i in $(seq 0 15); do
  kubectl exec sgl-$i -- bash -c "mkdir -p /mnt/ssd/DeepSeek-V4-Pro && \
    gcloud storage rsync -r gs://chrisya-gb300-models/DeepSeek-V4-Pro /mnt/ssd/DeepSeek-V4-Pro" &
done; wait
```

> ⚠️ **两份权重同名易混**。原装目录叫 `DeepSeek-V4-Pro`，NVFP4 版叫 `DeepSeek-V4-Pro-NVFP4`。启动前用 `config.json` 里的 `quantization_config` 二次确认自己拿的是哪份（见文首告警）。

---

## 4. 分发启动脚本 `[已验证]`

```bash
for i in $(seq 0 15); do
  ( kubectl exec -i sgl-$i -- bash -c "cat > /tmp/decode-dep8.sh"  < scripts/decode-dep8.sh
    kubectl exec -i sgl-$i -- bash -c "cat > /tmp/prefill-dep4.sh && chmod +x /tmp/*.sh" < scripts/prefill-dep4.sh ) &
done; wait
kubectl exec sgl-5 -- wc -l /tmp/decode-dep8.sh /tmp/prefill-dep4.sh   # 校验非空
```

> ⚠️ **必须 `kubectl exec -i`**。少了 `-i` 时 stdin 不透传，容器里会得到**空文件**，而且不报错。

---

## 5. 启动 workers `[已验证]`

**先单个冒烟，再批量铺开**——14 个一起错的排查成本远高于先验 1 个。

```bash
# ① 单 prefill 冒烟（~3min 后 HBM 应 >200G）
kubectl exec sgl-2 -- bash -c "setsid nohup bash /tmp/prefill-dep4.sh > /tmp/srv.log 2>&1 </dev/null &"
sleep 180 && kubectl exec sgl-2 -- nvidia-smi --query-gpu=memory.used --format=csv,noheader | head -1

# ② 通过后起 decode + 剩余 13 prefill
D0=$(kubectl get pod sgl-0 -o jsonpath='{.status.podIP}')
kubectl exec sgl-0 -- bash -c "setsid nohup bash /tmp/decode-dep8.sh 0 $D0:5000 > /tmp/srv.log 2>&1 </dev/null &"
kubectl exec sgl-1 -- bash -c "setsid nohup bash /tmp/decode-dep8.sh 1 $D0:5000 > /tmp/srv.log 2>&1 </dev/null &"
for i in $(seq 3 15); do
  kubectl exec sgl-$i -- bash -c "setsid nohup bash /tmp/prefill-dep4.sh > /tmp/srv.log 2>&1 </dev/null &" &
done; wait
```

### 5.1 ⚠️ 就绪判据有三层，前两层都会骗人

这是本次复测**代价最大的一课**：我用错判据，误以为满配跑通，压出来的数只有目标的一半，追了几小时才发现 14 个 prefill 里只有 5 个真在服务。

| 层 | 判据 | 说明什么 | 能不能信 |
|---|---|---|---|
| ① | `nvidia-smi` HBM > 200G | SGLang **预分配了显存池** | ❌ 权重可能随后加载失败、显存归零 |
| ② | `grep 'Load weight end' /tmp/srv.log` | 权重读完了 | ❌ 之后还要建 ZMQ / 起 scheduler / 注册，任一步崩都不改这行日志 |
| ③ | **etcd 里有 `v1/instances/dynamo/prefill/generate/<id>`** | worker **已进服务池、frontend 会路由给它** | ✅ **唯一权威** |

**权威判据一条命令**（数必须等于 prefill 数 / decode 数）：

```bash
kubectl exec sgl-2 -- bash -c "curl -s http://dynamo-etcd:2379/v3/kv/range -X POST \
  -d '{\"key\":\"AA==\",\"range_end\":\"AA==\",\"keys_only\":true}'" | python3 -c "
import sys,json,base64
ks=[base64.b64decode(k['key']).decode() for k in json.load(sys.stdin).get('kvs',[])]
print('prefill workers:', len([k for k in ks if k.startswith('v1/instances/dynamo/prefill/generate')]))
print('decode  workers:', len([k for k in ks if k.startswith('v1/instances/dynamo/backend/generate')]))"
# 期望： prefill workers: 14 / decode workers: 1
```

> **为什么 ①② 会骗人**：SGLang 启动是「分配显存池 → 读权重 → 建 ZMQ 端口 → 起 scheduler → 向 etcd 注册」。40236 ZMQ 端口僵尸（见 §6）发生在读权重**之后**，所以 `Load weight end` 已经打出来了，进程才崩，显存随即归零。而 frontend **一律返回 200**（它只要连上 NATS/etcd 就健康），也不会告诉你后端池子空了。**三个看起来最自然的健康信号全是绿的，系统却只有 1/3 的算力。**

**症状对照**（怀疑池子不满时先看这个）：

| 症状 | 含义 |
|---|---|
| TPOT 26–35ms（健康）但总吞吐只有目标的 40–60% | prefill 池子不满 → decode 饿着 |
| TTFT 中位 > 30s，而 TPOT 正常 | 同上，请求堵在 prefill 队列 |
| frontend 全 200、GPU 有些卡显存为 0 | 典型的「注册数 < 节点数」 |

---

## 6. 自愈循环（**一次成功的关键**）`[已验证]`

单个 worker 起不来是**常态**，不是异常（本次 14 个里 9 个中招）。手动逐个救每次都漏，正确姿势是脚本化「校验→重试」，**且校验必须用 §5.1 的 etcd 判据**：

```bash
NEED=14
for round in 1 2 3 4 5; do
  BAD=""
  for i in $(seq 2 15); do
    M=$(kubectl exec sgl-$i -- bash -c "nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits|head -1" 2>/dev/null|tr -d ' ')
    [ "${M:-0}" -lt 200000 ] 2>/dev/null && BAD="$BAD $i"
  done
  [ -z "$BAD" ] && { echo "显存全绿，转 etcd 终检"; break; }
  echo "round$round 未就绪:$BAD → 清进程"
  for i in $BAD; do kubectl exec sgl-$i -- bash -c "pkill -9 -f dynamo.sglang" 2>/dev/null; done
  sleep 95                                    # ★ 必须 ≥90s
  for i in $BAD; do kubectl exec sgl-$i -- bash -c "setsid nohup bash /tmp/prefill-dep4.sh > /tmp/srv.log 2>&1 </dev/null &"; done
  sleep 300                                   # 权重加载 ~4-5min
done
# ★★ 收尾必做：etcd 注册数终检（见 §5.1），不等于 14 就继续救
```

**两个必守的细节**：

- **每轮间隔 ≥90 秒**。40236 ZMQ 端口僵尸的持有者是 D-state 进程，`kill -9` 杀不掉，但**内核会在其 GPU 驱动调用返回后自动回收**。急着重试（<10s）会连撞，让人误判"必须重建 pod"。实测本次 round1 就救回 8/9。
- **`pkill` 模式别写 `-f sglang`**。那会匹配到 `kubectl exec` 自己那条含 `sglang` 的 `bash -c` 命令行，把自己杀掉（表现为 exit 137）。只 `pkill -9 -f dynamo.sglang`。

---

## 7. 启动 frontend `[已验证]`

**必须在所有 worker 稳定之后统一起**——因为 `pkill python3` 会连 frontend 一起杀。

```bash
for i in $(seq 2 15); do
  kubectl exec sgl-$i -- bash -c "NATS_SERVER=nats://dynamo-nats:4222 ETCD_ENDPOINTS=http://dynamo-etcd:2379 \
    setsid nohup python3 -m dynamo.frontend --http-port 8001 > /tmp/fe.log 2>&1 </dev/null &"
done
# 探活：必须 owned_by=nvidia + context_window=1048576，否则是僵尸 sglang::router 占端口
kubectl exec sgl-2 -- curl -s localhost:8001/v1/models
```

**多 frontend 是吞吐关键（+34%）**：单个裸 frontend 在高并发下 CPU-bound，是编排瓶颈。

> **frontend 返回 200 不代表后端池子健康**——它只要连上 NATS/etcd 就 200。后端有几个 worker 必须查 etcd（§5.1）。

---

## 8. 压测 `[已验证]`

### 8.1 口径（必须先对齐，否则数字没法比）

官方 11,200 的口径是 **output tok/s ÷ decode-GPU 数**：

- 分子**只算 output token**，input token 不进；
- 分母**只算 decode GPU**（本配置 8 张），14 个 prefill 节点的 56 张卡**不进分母**。

所以聚合方式 = 各路 `Output token throughput` **求和 ÷ 8**。

> ⚠️ **求和的前提是各路时间重叠**。如果某几路晚启动几十秒，各自的 `Output token throughput` 都是在「独占更多算力」的时间窗里测的，直接相加会**显著高估**。所以 §8.3 的启动器要确保 14 路同时在跑，并核对各路 `Benchmark duration` 相近。

### 8.2 装 sa-bench（pod 里默认没有）

用的是 SemiAnalysis 的 InferenceX（不是 SGLang 自带的 `sglang.bench_serving`，两者口径不同）：

```bash
git clone https://github.com/SemiAnalysisAI/InferenceX /tmp/InferenceX
kubectl exec sgl-2 -- mkdir -p /mnt/ssd
tar czf /tmp/ix.tgz -C /tmp InferenceX
for i in $(seq 2 15); do
  kubectl cp /tmp/ix.tgz sgl-$i:/tmp/ix.tgz
  kubectl exec sgl-$i -- bash -c "cd /mnt/ssd && tar xzf /tmp/ix.tgz" &
done; wait
# 校验 14/14（漏装的节点会静默不参与压测，见 §8.3 的坑）
for i in $(seq 2 15); do
  printf "%s:%s " $i "$(kubectl exec sgl-$i -- bash -c 'test -f /mnt/ssd/InferenceX/utils/bench_serving/benchmark_serving.py && echo Y||echo N')"
done; echo
```

只有 184K，很快。**注意 `/mnt/ssd` 是节点本地盘**——工具是按节点铺的，不是按 pod。

### 8.3 启动 14 路（带校验重试，别裸 for 循环）

```bash
CONC=600; NP=1800; TAG=r1
launch(){ kubectl exec sgl-$1 -- bash -c "rm -f /tmp/sab-$TAG.log; cd /mnt/ssd/InferenceX && \
  setsid nohup python3 utils/bench_serving/benchmark_serving.py \
  --backend openai --host 127.0.0.1 --port 8001 \
  --model deepseek-ai/DeepSeek-V4-Pro --tokenizer /mnt/ssd/DeepSeek-V4-Pro \
  --dataset-name random --random-input-len 8192 --random-output-len 1024 --random-range-ratio 0.8 \
  --num-prompts $NP --max-concurrency $CONC --request-rate inf --ignore-eos --dsv4 --use-chat-template \
  > /tmp/sab-$TAG.log 2>&1 </dev/null & sleep 4"; }        # ★ sleep 4 见下

for att in 1 2 3; do
  TODO=""
  for i in $(seq 2 15); do
    kubectl exec sgl-$i -- bash -c "test -s /tmp/sab-$TAG.log" 2>/dev/null || TODO="$TODO $i"
  done
  [ -z "$TODO" ] && break
  for i in $TODO; do launch $i & done; wait
  sleep 8
done
```

**★ 为什么每个 exec 里要 `sleep 4`**：`kubectl exec` 返回后会关掉 exec 流。如果 `setsid nohup ... &` 刚 fork 就关流，子进程会在 detach 完成前被带走——表现是**日志文件根本不生成，且完全不报错**。裸 `for` 循环批量起 14 路，实测只有 6 路活下来。加 4 秒让进程站稳，再配外层「校验→重试」，才稳。

**参数说明**：`--request-rate inf` = 开环（对标官方必须开环，闭环会严重低估）；`--ignore-eos` 保证每条真出满 1024 token；`--dsv4 --use-chat-template` 必须成对出现（只给 `--dsv4` 会报错）；`--backend` 只能是 `openai`（没有 `sglang-oai` 这个值）。

### 8.4 收结果

```bash
TOT=0
for i in $(seq 2 15); do
  v=$(kubectl exec sgl-$i -- bash -c "grep 'Output token throughput' /tmp/sab-$TAG.log|awk '{print \$NF}'")
  d=$(kubectl exec sgl-$i -- bash -c "grep 'Benchmark duration' /tmp/sab-$TAG.log|awk '{print \$NF}'")
  echo "sgl-$i out=$v dur=$d"          # ★ dur 必须彼此相近，否则不能求和
  [ -n "$v" ] && TOT=$(python3 -c "print($TOT+$v)")
done
python3 -c "print('output/decode-GPU:', round($TOT/8,1))"
```

### 8.5 读数：怎么判断瓶颈在哪一侧

| 指标 | 健康值 | 偏离的含义 |
|---|---|---|
| **TPOT 中位** | 21–35ms | 高于此 → decode 自身过载 |
| **TTFT 中位** | < 10s | 远高于此（30s+）而 TPOT 正常 → **prefill 喂料不足**，请求堵在 prefill 队列 |
| **各路 duration** | 彼此 ±15% | 差异大 → 没有真正并发，聚合数不可信 |

**反推 decode 实际并发**：`聚合 output tok/s × TPOT(s)` ≈ 正在 decode 的序列数。拿它跟你 offer 的总并发比——差得远就说明请求都堵在 prefill 侧。这一步能在几秒内区分「decode 到顶」和「prefill 饿着 decode」，比盯 GPU 利用率快得多。

> `ITL` 和 `TPOT` 会差两个数量级（如 ITL 2200ms vs TPOT 21ms），这是 `--stream-interval 60` 把 60 个 token 攒一个 chunk 发的结果，**不是异常**。判断延迟用 TPOT，不要用 ITL。

---

## 9. 故障速查

| 现象 | 根因 | 处理 | 状态 |
|---|---|---|---|
| **跑通了但吞吐只有一半**（TPOT 正常、TTFT 60s+、frontend 全 200） | **prefill worker 大量没注册进 etcd**，但显存/日志/frontend 三个信号全绿 | §5.1 用 etcd 判据重查，§6 自愈补齐 | `[已定位并修复]` |
| `NotImplementedError: Runner backend FLASHINFER_TRTLLM_ROUTED requires a fused func for a2a backend megamoe` | `--moe-runner-backend` 默认 `auto`，新 nightly 的 auto 选了 flashinfer，与 megamoe 不配套 | **显式加 `--moe-runner-backend deep_gemm`**（脚本已含） | `[已定位并修复]` |
| `RuntimeError: The size of tensor a (384) must match the size of tensor b (48)` | NVFP4 权重走了 megamoe 路径（384=`n_routed_experts`，48=384/ep） | 换官方原装权重，见文首告警 | `[已定位并修复]` |
| 容器里脚本是空文件 | `kubectl exec` 少了 `-i` | 见 §4 | `[已修复]` |
| **压测日志文件根本不生成、也不报错** | `kubectl exec` 关流太快，`setsid` 还没 detach 完子进程就被带走 | exec 里加 `sleep 4` + 外层校验重试，见 §8.3 | `[已定位并修复]` |
| 压测某几路一直没结果 | 那几个**节点**没铺 `/mnt/ssd/InferenceX`（工具按节点铺，不按 pod） | §8.2 校验 14/14 | `[已定位并修复]` |
| `FailedPrepareDynamicResources` | 裸 pod + `nodeName` 绕过 scheduler | 用 StatefulSet | `[已修复]` |
| prefill HBM 一直 0 + `scheduler died (exit -3)` | 40236 ZMQ 端口僵尸（瞬态） | §6 自愈循环，间隔 ≥90s | `[已验证有效]` |
| `kubectl exec` 自己 exit 137 | `pkill -f sglang` 匹配到 exec 自身的 `bash -c` 命令行 | 改用 `pkill -9 -f dynamo.sglang` | `[已定位并修复]` |
| decode 显存 199G 不释放、`kill -9` 无效 | 真 D-state 卡死 | `kubectl delete pod --force` 重建（模型在节点盘，不丢） | — |

---

## 9. 验证记录

**2026-07-25 复测（本 runbook 首次成文）**，环境：`gb300-pool-0002` 17 节点，16 pod fleet，官方原装 `DeepSeek-V4-Pro`。

| 步骤 | 结果 | 实测 |
|---|---|---|
| §1 前置检查 | ✅ | 17 节点带标签；dynamo-nats/etcd Running |
| §2 部署 fleet | ✅ | 16/16 Running，一节点一 pod，~2min |
| §3 模型 | ✅ | 16/16 节点 `DeepSeek-V4-Pro` 806G / 64 分片（GCS 拉取 ~850MB/s） |
| §4 分发脚本 | ✅ | 16 pod 全部写入成功 |
| §5 单 prefill 冒烟 | ✅ | 加 `--moe-runner-backend deep_gemm` 后，3min HBM 260GB |
| §5 批量 14 prefill | ⚠️ **5/14** | 9 个撞 40236 僵尸；**显存和日志判据全绿，骗过了第一次验收** |
| §5.1 etcd 判据 | ✅ | 揭穿上一行：`prefill/generate` 只有 5 个 |
| §6 自愈循环 | ✅ | round1 救回 8/9，剩 1 个单独重启，最终 **14/14 注册** |
| §5 decode dep8 | ✅ | 8 rank 全 `Load weight end` + `registration succeeded` |
| §7 frontend | ✅ | 14/14 返回 200，`owned_by=nvidia`、`context_window=1048576` |
| 端到端推理 | ✅ | `curl /v1/completions` → 正确回答 "Paris"，PD 链路全通 |
| §8 满配压测 | ✅ | **9,168 output/decode-GPU**，见下 |

### 10.1 压测结果

`14 prefill(dep4) + 1 decode(dep8+MTP)`，sa-bench 开环 14 路 × conc600 × 1800 prompts，ISL 8192 / OSL 1024 / range 0.8：

| 轮次 | 实际在池 prefill | 聚合 output tok/s | **output/decode-GPU** | TPOT 中位 | TTFT 中位 |
|---|---|---|---|---|---|
| 首测（误判就绪） | **5** | 36,502 | 4,563（51%）| 27ms | **61s** |
| 降并发排除过载 | **5** | 35,841 | 4,480（50%）| 27ms | 高 |
| **修满 14 后** | **14** | **73,347** | **9,168** | 59–61ms | — |

- 对老文档基线 8,993 = **102%**（±2% 噪声内，判定复现成功）
- 对官方 11,200 = **81.9%**
- 各路 `Benchmark duration` 289–328s（±7%），真并发，聚合有效

**首测两轮的教训值得单列**：TPOT 27ms 完全健康、frontend 全 200、降并发也不改善——三个信号都指向「系统正常，只是天花板低」。真相是算力只有 1/3。**排查顺序应该是先数后端 worker，再调参数**；我反过来了，白跑两轮。

### 10.2 并发扫描（14 prefill 全在池）

| 每路 conc | 总 offered | 时长 | 聚合 output tok/s | **output/decode-GPU** | TPOT 中位 | **TTFT 中位** | 增量 |
|---|---|---|---|---|---|---|---|
| 400 | 5,600 | 224s | 70,216 | 8,777（78.4%）| 58–59ms | — | 基线 |
| **600** | 8,400 | 300s | 73,347 | **9,168（81.9%）** | 59–61ms | **45s** | +4.5% |
| 900 | 12,600 | 470s | 74,239 | 9,280（82.9%）| 62–63ms | — | +1.2% |
| 1200 | 16,800 | 618s | 75,534 | 9,442（84.3%）| 62ms | **150s** | +1.7% |
| 1800 | 25,200 | 955s | 77,960 | **9,745（87.0%）** | 60ms | **253s** | +3.2% |

**三个必须一起看的事实**：

1. **吞吐一直在涨，但涨得很慢** —— offered 从 5,600 拉到 25,200（4.5×），吞吐只涨 11%。
2. **TPOT 全程钉在 58–63ms 不动** —— decode 早在 conc400 就已经饱和（`--max-running-requests 8192` 是硬上限），多出来的并发全在 prefill 侧排队，没进 decode。
3. **延迟代价是灾难性的** —— TTFT 中位从 45s 涨到 **253s**。conc1800 那个 9,745 是「拿 5.6 倍排队延迟换 6% 吞吐」买来的，**不是可用工作点**。

> 还有个**测量效应**要扣掉：并发越高单轮跑越久（224s → 955s），稳态占比越大、启停摊薄越多，本身就会让聚合数字虚高一点。

**结论**：对标口径应取 **conc600 = 9,168**（与旧文档 8,993 同工作点，判定复现成功）。**~9,200–9,400 是 `dep8 + MTP + nightly 镜像` 的实际天花板**，剩下 ~1.2× 到官方 11,200 是**单卡内核成熟度差**——不是并发点没找对、也不是 prefill 数量不够（旧文档 §14 已实测 full autotune 无提升、EPLB 与 megamoe 不兼容）。要摸 11,200 只剩「对齐官方 pinned 镜像（commit `14f81a67`）」一条路。

### 10.3 本次复测发现的旧文档缺陷（均已在本文修正）

1. **模型与 MoE backend 配方错配（最严重）** —— 两套互斥配方分散在旧文档 §1 和 §13.A，中间无交叉提示。照 §13.A 抄 + 手上是 NVFP4 权重 = 必挂，且报错（tensor 384 vs 48）完全指不到根因。→ 提到文首告警。
2. **就绪判据不足以判就绪** —— 旧文档给的「HBM>200G」和本文初稿给的 `Load weight end` **都会骗人**。崩溃发生在读权重之后，两个信号都已经变绿了。唯一权威是 etcd 注册数。→ §5.1。
3. **pod 生成器未入库** —— 旧文档引用 `gen18-0001.py`，仓库里没有，第一步就卡死。→ `manifests/sgl-fleet.yaml`。
4. **漏 `--moe-runner-backend deep_gemm`** —— 旧文档强调"一个字都不能漏"，却依赖 `auto` 的默认行为。nightly 镜像更新后 auto 改选 flashinfer，整条 megamoe 路径直接崩。**「依赖默认值」型埋雷**：写文档时能跑 ≠ 半年后能跑。
5. **裸 pod 与 DRA 冲突未说明** —— 生成器批量创建裸 pod，与 DRA channel 预留机制冲突。→ StatefulSet。
6. **GCS bucket 名写成占位符** —— 全文 `gs://<bucket>`，真名 `gs://chrisya-gb300-models`。
7. **压测工具来源未记** —— 只说用 InferenceX `sa-bench`，没说从哪来、怎么装、装到哪（**按节点不按 pod**）。→ §8.2。
8. **压测启动方式没写对** —— 裸 `for` 循环 `kubectl exec` 批量起，实测 14 路只活 6 路且不报错。→ §8.3 的 `sleep 4` + 校验重试。

> **方法论**：本文每一节都是「先在集群上真跑一遍，再写进来」。上面 8 条全部是执行中撞出来的，读旧文档发现不了。**其中第 2 条是最贵的**——它不让部署失败，只让性能腰斩，而所有常规健康检查都是绿的。
