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

模型在**节点本地 SSD**（hostPath `/mnt/disks/raid/0`），删 pod 不丢。

> ⚠️ **这一步不能跳过，哪怕上一轮刚跑完**。节点数（17）比 pod 数（16）多，重建 StatefulSet 时调度器**会换节点**——上一轮的空闲节点这轮可能被占用，那个 pod 就是空的。

### 3.1 ⚠️ 先查 RAID 挂载，再查模型（`md0` → `md127` 陷阱）

**「模型缺失」十有八九不是模型没拷，是那台的 Local SSD RAID 根本没挂上。** 一条命令先排除：

```bash
for i in $(seq 0 15); do
  printf "sgl-%s: " $i
  kubectl exec sgl-$i -- df -h /mnt/ssd | tail -1 | awk '{print $2, $5}'
done
# 正常：12T / 15-20%。若看到 **256K 100%** → RAID 没挂，见下
```

**根因**：节点重启后内核会把已存在的 RAID 阵列**自动组装成 `/dev/md127`**，而不是创建时的 `/dev/md0`。而常见的 RAID DaemonSet 脚本写的是：

```bash
if ! grep -q "md0" /proc/mdstat; then mdadm --create /dev/md0 ... ; fi
tune2fs -l /dev/md0 || mkfs.ext4 -F /dev/md0
mount /dev/md0 /mnt/disks/raid/0
```

`"md127"` 里不含 `"md0"` → 判定「没有阵列」→ 去 create → 盘已被 md127 占用 → `mdadm: Device or resource busy` → 后面 `mkfs` / `mount` 连环失败。

**失败后的表现极具迷惑性**：hostPath 用 `DirectoryOrCreate`，kubelet 会在 COS 只读根文件系统上建出这个目录，落到 tmpfs 上 —— **pod 正常起、`/mnt/ssd` 存在、但只有 256K**。写模型时静默失败（`curl -o` 写出 0 字节文件，退出码还是 0）。

**修复**（数据无损，阵列和 ext4 都还在，只是没挂）：

```bash
# 在该节点的 raid DaemonSet pod 里执行（需 mountPropagation: Bidirectional 才能传播到宿主）
MD=$(awk '/^md[0-9]+ : active/{print $1; exit}' /proc/mdstat)   # 动态识别，别写死 md0
tune2fs -l /dev/$MD >/dev/null 2>&1 || mkfs.ext4 -F /dev/$MD    # ★ 有 fs 就别格式化
mkdir -p /mnt/disks/raid/0
mountpoint -q /mnt/disks/raid/0 || mount -o discard,defaults /dev/$MD /mnt/disks/raid/0
chmod a+w /mnt/disks/raid/0
```

挂好后**不用重启 sgl pod** —— sgl 的 volumeMount 带 `mountPropagation: HostToContainer`，宿主机上的新挂载会直接传播进运行中的容器。实测 5 台修复后 pod 内立刻看到 12T 和 64 个分片，模型数据一个没丢。

> 永久修复见 `manifests/raid-pool-0002.yaml`（已把 `md0` 硬编码改成动态识别，并在结尾加了「仍是 tmpfs 就报 FAILED」的自检）。

> ⚠️ **两份权重同名易混**。原装目录叫 `DeepSeek-V4-Pro`，NVFP4 版叫 `DeepSeek-V4-Pro-NVFP4`。启动前用 `config.json` 里的 `quantization_config` 二次确认自己拿的是哪份（见文首告警）。

**缺失时怎么补**（优先级从高到低）：

**① pod→pod 直传（最快，实测 3.6 GB/s，806G 约 4 分钟）**

集群内网带宽远高于 GCS，且**镜像里没有 `gcloud`**（见下），所以这是首选：

```bash
SRC=1; DST=0                                   # 从 sgl-1 补给 sgl-0
SRCIP=$(kubectl get pod sgl-$SRC -o jsonpath='{.status.podIP}')
# 源端起 HTTP 服务（python3 -m http.server 自 3.7 起是多线程的）
kubectl exec sgl-$SRC -- bash -c "cd /mnt/ssd && setsid nohup python3 -m http.server 8899 >/dev/null 2>&1 </dev/null &"
# 生成文件清单（含 assets/ inference/ 等子目录，共 276 个文件）
kubectl exec sgl-$SRC -- bash -c "cd /mnt/ssd/DeepSeek-V4-Pro && find . -type f | sed 's|^\./||' > /tmp/fl.txt"
kubectl exec sgl-$SRC -- cat /tmp/fl.txt > /tmp/fl.txt && kubectl cp /tmp/fl.txt sgl-$DST:/tmp/fl.txt
# 目标端 6 路并行拉
kubectl exec sgl-$DST -- bash -c "mkdir -p /mnt/ssd/DeepSeek-V4-Pro && setsid nohup bash -c '
  cd /mnt/ssd/DeepSeek-V4-Pro
  xargs -P 6 -I{} bash -c \"mkdir -p \\\$(dirname {}); curl -sf -o {} http://$SRCIP:8899/DeepSeek-V4-Pro/{}\" < /tmp/fl.txt
  echo DONE > /tmp/copy.done' >/dev/null 2>&1 </dev/null &"
# 轮询：du 到 806G 且 /tmp/copy.done 存在即完成
```

**② 从 GCS 拉**（全 fleet 都缺时用，~850MB/s）：

```bash
gcloud storage rsync -r gs://chrisya-gb300-models/DeepSeek-V4-Pro /mnt/ssd/DeepSeek-V4-Pro
```

> ⚠️ **`lmsysorg/sglang` 镜像里没有 `gcloud`**（`command not found`），所以上面这条**不能直接在容器里跑**。要么先装 gcloud，要么用容器里已有的 `google-cloud-storage` Python SDK，要么走 ① 的 pod→pod。旧文档默认容器有 gcloud（那是它在 bootstrap 阶段装的），照抄会卡住。

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
# ① 单 prefill 冒烟（★ 等 300s，不是 180s）
kubectl exec sgl-2 -- bash -c "setsid nohup bash /tmp/prefill-dep4.sh > /tmp/srv.log 2>&1 </dev/null &"
sleep 300 && kubectl exec sgl-2 -- nvidia-smi --query-gpu=memory.used --format=csv,noheader | head -1
#   期望 ~260 GiB。若为 0，看 /tmp/srv.log 尾部

# ② decode 先起（它最慢，graph capture 要 8-10min，跟 prefill 并行）
D0=$(kubectl get pod sgl-0 -o jsonpath='{.status.podIP}')
kubectl exec sgl-0 -- bash -c "setsid nohup bash /tmp/decode-dep8.sh 0 $D0:5000 > /tmp/srv.log 2>&1 </dev/null &"
kubectl exec sgl-1 -- bash -c "setsid nohup bash /tmp/decode-dep8.sh 1 $D0:5000 > /tmp/srv.log 2>&1 </dev/null &"

# ③ 剩余 13 prefill —— 错开起（摊开 I/O；注意这不能避免 §5.1 的崩溃）
for i in $(seq 3 15); do
  kubectl exec sgl-$i -- bash -c "setsid nohup bash /tmp/prefill-dep4.sh > /tmp/srv.log 2>&1 </dev/null &"
  sleep 8
done
```

> **错开 8s 是便宜的保险，但别指望它解决 40236 崩溃**。实测两轮：
> - 轮 1 同时起 13 个（`&`+`wait`）→ 12 崩
> - 轮 2 错开 8s 起 13 个 → **仍然 9 崩**
>
> 失败率是 60–85%，与是否错开**没有显著相关**（根因见 §5.1，是 pod 内部竞态）。唯一有效的手段是 §6 的重试循环。错开的实际好处只是把 14 个 pod 的启动 I/O 摊开，不至于同时读 14×806G。

**时序预期**（从启动算起）：

| 阶段 | 耗时 | 观测点 |
|---|---|---|
| 权重加载 | ~4–5min | HBM 从 0 涨到 ~260 GiB |
| CUDA graph capture | 再 ~3–5min | HBM 稳定不动 |
| 向 etcd 注册 | 之后 | §5.2 的计数开始涨 |
| **总计** | **8–12min** | 别在 5 分钟时下结论 |

### 5.1 40236 崩溃的真实根因（**不是**跨节点争抢）

**失败率 60–85%，是本流程最大的时间黑洞**，所以值得讲清楚它到底是什么。

报错长这样，出现在**权重已经加载完之后**：

```
Load weight end. elapsed=367.51 s, ...          ← 4 个 DP rank 全部成功
...
File "sglang/srt/entrypoints/engine.py", line 257, in __init__
    self.send_to_rpc = get_zmq_socket(...)
zmq.error.ZMQError: Address already in use (addr='tcp://127.0.0.1:40236')
```

三个关键事实：

1. **地址是 `127.0.0.1`** —— pod 内部 loopback，**跟别的节点、别的 pod 完全无关**。同域并发启动不是原因。
2. **崩的是主进程，不是 scheduler** —— 4 个 `sglang::scheduler_DP*` 子进程都已 `Load weight end`，是主进程回来 bind rpc socket 时撞车。健康 pod 上 `grep 9D2C /proc/net/tcp` 能看到 40236 有 5 条记录（主进程 listen + 4 个 DP rank 连入）。
3. **崩完留 4 个 defunct 僵尸**（`[sglang::schedul] <defunct>`，PPID=1），端口随后被内核释放。

即：**`--enable-dp-attention` + `dp_size=4` 下，主进程与 DP rank 之间对固定 rpc 端口 40236 的初始化竞态**。谁先 bind 是时序决定的，所以它表现为按 pod 独立的掷硬币，**重试就能过**。

> **实操结论**：别去调启动顺序、别去改并发度、别怀疑网络——**只有重试有用**（§6）。每轮之间留 ≥90s，让内核收掉僵尸持有的端口。

### 5.2 ⚠️ 就绪判据有三层，前两层都会骗人

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

单个 worker 起不来是**常态**，不是异常（本次 14 个里 9 个中招）。手动逐个救每次都漏，正确姿势是脚本化「校验→重试」，**且校验必须用 §5.2 的 etcd 判据**：

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
  for i in $BAD; do kubectl exec sgl-$i -- bash -c "pkill -9 -f 'dynamo[.]sglang'" 2>/dev/null; done   # ★ 括号必须有
  sleep 95                                    # ★ 必须 ≥90s
  for i in $BAD; do                           # ★ 错开 8s，别一把梭
    kubectl exec sgl-$i -- bash -c "setsid nohup bash /tmp/prefill-dep4.sh > /tmp/srv.log 2>&1 </dev/null &"
    sleep 8
  done
  sleep 330                                   # 权重加载 + capture
done
# ★★ 收尾必做：etcd 注册数终检（见 §5.2），不等于 14 就继续救
```

**两个必守的细节**：

- **每轮间隔 ≥90 秒**。40236 ZMQ 端口僵尸的持有者是 D-state 进程，`kill -9` 杀不掉，但**内核会在其 GPU 驱动调用返回后自动回收**。急着重试（<10s）会连撞，让人误判"必须重建 pod"。实测本次 round1 就救回 8/9。
- **`pkill -f` 必须用括号转义**：写成 `pkill -9 -f 'dynamo[.]sglang'`。`pkill -f` 匹配的是**整条命令行**，而 `kubectl exec sgl-N -- bash -c "pkill -9 -f dynamo.sglang; ..."` 这条命令行**自身就含有 `dynamo.sglang` 这串字符**——于是它把自己杀了（exit 137，且后面的重启语句根本没执行）。`dynamo[.]sglang` 作为正则仍匹配真实进程，但字面串不等于自身，就不会自杀。同理别写 `-f sglang`。

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

> **frontend 返回 200 不代表后端池子健康**——它只要连上 NATS/etcd 就 200。后端有几个 worker 必须查 etcd（§5.2）。

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

### 8.4 ★ 第一轮必须当 warmup 丢掉

**重启后的首轮压测比热态低 ~7%，不是噪声，而且高度可复现。** 两次独立的「清空→重建→压测」：

| | 冷（重启后第 1 轮） | 热（紧接着第 2 轮） | 差 |
|---|---|---|---|
| 审计轮 1 | 8,520 | **9,118** | +7.0% |
| 审计轮 2 | 8,552 | **9,108** | +6.5% |
| 两轮之间偏差 | ±0.2% | ±0.1% | — |

冷态和热态**各自都稳定到 ±0.2%**，说明这是确定性的 warmup 成本，不是随机波动。

> **注意归因**：这**不是** DeepGEMM 磁盘 JIT 缓存的锅。`SGLANG_DG_CACHE_DIR=/mnt/ssd/dg-cache` 在节点盘、跨 pod 重建保留，轮 2 开始时它已经是热的，**冷跑照样低 6.5%**。所以这笔开销发生在**进程内**（首次遇到各 M shape 时的 kernel 选择 / 内存池 / autotune 状态）——**每次重启 server 都要重新付一次**，跟磁盘缓存在不在无关。

> **流程：跑两轮，报第二轮。** 只跑一轮就报数会系统性低估 ~7%，而且这个偏差**只在重启后出现**，极易被误读成「这次部署有问题」。

### 8.5 收结果

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

### 8.6 读数：怎么判断瓶颈在哪一侧

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
| **跑通了但吞吐只有一半**（TPOT 正常、TTFT 60s+、frontend 全 200） | **prefill worker 大量没注册进 etcd**，但显存/日志/frontend 三个信号全绿 | §5.2 用 etcd 判据重查，§6 自愈补齐 | `[已定位并修复]` |
| `NotImplementedError: Runner backend FLASHINFER_TRTLLM_ROUTED requires a fused func for a2a backend megamoe` | `--moe-runner-backend` 默认 `auto`，新 nightly 的 auto 选了 flashinfer，与 megamoe 不配套 | **显式加 `--moe-runner-backend deep_gemm`**（脚本已含） | `[已定位并修复]` |
| `RuntimeError: The size of tensor a (384) must match the size of tensor b (48)` | NVFP4 权重走了 megamoe 路径（384=`n_routed_experts`，48=384/ep） | 换官方原装权重，见文首告警 | `[已定位并修复]` |
| 容器里脚本是空文件 | `kubectl exec` 少了 `-i` | 见 §4 | `[已修复]` |
| **压测日志文件根本不生成、也不报错** | `kubectl exec` 关流太快，`setsid` 还没 detach 完子进程就被带走 | exec 里加 `sleep 4` + 外层校验重试，见 §8.3 | `[已定位并修复]` |
| 压测某几路一直没结果 | 那几个**节点**没铺 `/mnt/ssd/InferenceX`（工具按节点铺，不按 pod） | §8.2 校验 14/14 | `[已定位并修复]` |
| 重建 fleet 后某个 pod 模型目录是空的 | 17 节点 16 pod，重建时调度器换了节点，落到了没铺模型的备用节点 | §3 每次重建都要校验，缺就 pod→pod 补 | `[已定位并修复]` |
| 容器里 `gcloud: command not found` | `lmsysorg/sglang` 镜像不含 gcloud | 用 §3 的 pod→pod 直传（还更快） | `[已定位并修复]` |
| `FailedPrepareDynamicResources` | 裸 pod + `nodeName` 绕过 scheduler | 用 StatefulSet | `[已修复]` |
| prefill HBM 一直 0 + `scheduler died (exit -3)` | 40236 ZMQ 端口僵尸（瞬态） | §6 自愈循环，间隔 ≥90s | `[已验证有效]` |
| `kubectl exec` 自己 exit 137、后续语句没执行 | `pkill -f <pat>` 匹配到 exec 自身的 `bash -c` 命令行（那行文本里就含 `<pat>`） | 改用 `pkill -9 -f 'dynamo[.]sglang'` 括号转义 | `[已定位并修复]` |
| decode 显存 199G 不释放、`kill -9` 无效 | 真 D-state 卡死 | `kubectl delete pod --force` 重建（模型在节点盘，不丢） | — |

---

## 10. 验证记录

**2026-07-25 复测（本 runbook 首次成文）**，环境：`gb300-pool-0002` 17 节点，16 pod fleet，官方原装 `DeepSeek-V4-Pro`。

| 步骤 | 结果 | 实测 |
|---|---|---|
| §1 前置检查 | ✅ | 17 节点带标签；dynamo-nats/etcd Running |
| §2 部署 fleet | ✅ | 16/16 Running，一节点一 pod，~2min |
| §3 模型 | ✅ | 16/16 节点 `DeepSeek-V4-Pro` 806G / 64 分片（GCS 拉取 ~850MB/s） |
| §4 分发脚本 | ✅ | 16 pod 全部写入成功 |
| §5 单 prefill 冒烟 | ✅ | 加 `--moe-runner-backend deep_gemm` 后，3min HBM 260GB |
| §5 批量 14 prefill | ⚠️ **5/14** | 9 个撞 40236 僵尸；**显存和日志判据全绿，骗过了第一次验收** |
| §5.2 etcd 判据 | ✅ | 揭穿上一行：`prefill/generate` 只有 5 个 |
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

### 10.3 审计轮 1：全清空 → 只照本文重建（2026-07-26）

把 StatefulSet 连 pod 全删，然后**只用本文的命令**重建一遍，不参考任何历史命令。

| 步骤 | 结果 | 实测 / 偏差 |
|---|---|---|
| §2 部署 fleet | ✅ | 16/16 Running **60s**（文档写 ~2min，偏保守） |
| §3 模型校验 | ⚠️ **抓到问题** | `sgl-0` 落到备用节点，模型缺失 → pod→pod 补齐 806G/4min |
| §4 分发脚本 | ✅ | 16/16 写入成功 |
| §5 单 prefill 冒烟 | ❌ **文档错** | 180s 时 HBM 还是 0，实际要 ~300s |
| §5 批量起 13 prefill | ❌ **文档错** | `&`+`wait` 一把梭 → **12/14 崩** |
| §6 自愈（错开 8s） | ✅ | round1 修 7/12、round2 修 5/5，**2 轮全绿** |
| §5.2 etcd 终检 | ✅ | prefill 14 / decode 1 |
| §7 frontend | ✅ | 14/14 → 200，`owned_by=nvidia`、`context_window=1048576` |
| 端到端推理 | ✅ | `"The capital of France is"` → `" Paris. The capital of Italy is Rome."` |
| §8 压测（冷） | ⚠️ | 8,520（比首轮低 7%）|
| §8 压测（热） | ✅ **通过** | **9,118 = 首轮 9,168 的 99.5%** |

**结论：可复现（±0.5%）**。但暴露了 5 个文档缺陷，全部已修：

1. **仓库脚本仍指向 `-NVFP4`** —— 与文首告警自相矛盾，照抄必挂。（还没开跑就抓到）
2. **重建会换节点** —— 17 节点 16 pod，调度器重排，落到备用节点的 pod 没有模型。→ §3 加了强制校验说明。
3. **容器里没有 `gcloud`** —— 文档给的 GCS 拉取命令跑不了。→ §3 改推 pod→pod 直传（还快 4×）。
4. **`pkill -9 -f dynamo.sglang` 会自杀** —— 我上一版**刚写进文档的"安全"建议是错的**：`pkill -f` 匹配整条命令行，而 `kubectl exec ... bash -c "pkill -9 -f dynamo.sglang; ..."` 这行本身就含该字符串。→ 改 `'dynamo[.]sglang'`。
5. **批量启动必须错开** —— 同时起 13 个 → 12 崩；错开 8s → 2 轮全绿。→ §5③。

> **审计的价值在这里**：#4 和 #5 都是「上一轮跑通了、写下来了、看着也对」的东西。只有真正从零再跑一遍才会暴露——#4 甚至是我自己刚写的错误建议，靠 review 文档发现不了。

### 10.4 审计轮 2：用修正后的文档再从零跑一遍（2026-07-26）

| 步骤 | 结果 | 实测 |
|---|---|---|
| §2 部署 fleet | ✅ | 16/16 Running 60s |
| §3 模型校验 | ✅ | 16/16 齐全（这轮没换到空节点）|
| §4 分发脚本 | ✅ | — |
| §5① 冒烟 300s | ✅ | sgl-2 HBM 260,942 MiB（**验证了轮 1 把 180s 改 300s 是对的**）|
| §5③ 错开 8s 起 13 个 | ❌ **证伪轮 1 结论** | **仍崩 9/14** |
| §6 自愈 | ✅ | 3 轮全绿 → etcd prefill 14 / decode 1 |
| §7 frontend + e2e | ✅ | 14/14 → 200；`" Paris. The capital of Italy is Rome."` |
| §8 压测 冷 / 热 | ✅ | **8,552 / 9,108** |

**三次独立热态测量的一致性**：

| 测量 | output/decode-GPU |
|---|---|
| 首轮（非审计） | 9,168 |
| 审计轮 1 | 9,118 |
| 审计轮 2 | 9,108 |
| **离散度** | **±0.7%** |

**本轮唯一的新发现，是推翻了上一轮的一个结论**：

- 轮 1 我从单次观察归纳出「错开 8s 启动能大幅降低 40236 崩溃率」。轮 2 照做，**仍崩 9/14**——失败率 60–85%，与是否错开无关。
- 顺着这条线挖到了**真根因**（§5.1）：报错地址是 `127.0.0.1:40236`，pod 内 loopback；崩的是主进程 bind rpc socket，而 4 个 DP scheduler 早已 `Load weight end`。是 `--enable-dp-attention` + `dp_size=4` 下主进程与 DP rank 抢固定端口的**进程内竞态**，跟跨节点并发毫无关系。
- 教训：**n=1 的观察不要写成因果**。「实测 X 时出现 Y」和「X 导致 Y」是两回事，后者要再测一次才配写。

### 10.5 本次复测发现的旧文档缺陷（均已在本文修正）

1. **模型与 MoE backend 配方错配（最严重）** —— 两套互斥配方分散在旧文档 §1 和 §13.A，中间无交叉提示。照 §13.A 抄 + 手上是 NVFP4 权重 = 必挂，且报错（tensor 384 vs 48）完全指不到根因。→ 提到文首告警。
2. **就绪判据不足以判就绪** —— 旧文档给的「HBM>200G」和本文初稿给的 `Load weight end` **都会骗人**。崩溃发生在读权重之后，两个信号都已经变绿了。唯一权威是 etcd 注册数。→ §5.2。
3. **pod 生成器未入库** —— 旧文档引用 `gen18-0001.py`，仓库里没有，第一步就卡死。→ `manifests/sgl-fleet.yaml`。
4. **漏 `--moe-runner-backend deep_gemm`** —— 旧文档强调"一个字都不能漏"，却依赖 `auto` 的默认行为。nightly 镜像更新后 auto 改选 flashinfer，整条 megamoe 路径直接崩。**「依赖默认值」型埋雷**：写文档时能跑 ≠ 半年后能跑。
5. **裸 pod 与 DRA 冲突未说明** —— 生成器批量创建裸 pod，与 DRA channel 预留机制冲突。→ StatefulSet。
6. **GCS bucket 名写成占位符** —— 全文 `gs://<bucket>`，真名 `gs://chrisya-gb300-models`。
7. **压测工具来源未记** —— 只说用 InferenceX `sa-bench`，没说从哪来、怎么装、装到哪（**按节点不按 pod**）。→ §8.2。
8. **压测启动方式没写对** —— 裸 `for` 循环 `kubectl exec` 批量起，实测 14 路只活 6 路且不报错。→ §8.3 的 `sleep 4` + 校验重试。

> **方法论**：本文每一节都是「先在集群上真跑一遍，再写进来」，写完又清空环境**照文档从零重跑了 3 轮**（§10.3 / §10.4 / §10.6）。上面 8 条全部是执行中撞出来的，读文档发现不了。
>
> **两个最贵的**：第 2 条（就绪判据）不让部署失败、只让性能腰斩，而所有常规健康检查都是绿的；而审计轮 2 推翻的那条错误归因，是**我自己在轮 1 刚写下的**——它读起来完全合理，只有真跑第二遍才暴露。**这就是为什么复现文档必须自己审计自己。**

---

## 11. 冲击官方 11,200：调研结论 + 实验 `[进行中]`

我们稳定在 **9,032–9,168**（四次热态测量，均值 9,107，±0.8%）。这一节记录「差的那 20% 到底在哪」的调研和实验。

### 11.1 调研推翻了两个原有假设

| 原假设 | 调研结论 | 出处 |
|---|---|---|
| 「我们用 nightly，官方用更成熟的 pinned 镜像，差在内核成熟度」 | **方向反了**。官方 pinned 是 `nightly-dev-20260527-14f81a67`（2026-05-27），我们用的 `20260720` **新两个月** | commit `14f81a67` = sgl-project/sglang *"bump sglang-kernel to 0.4.3 (#26421)"* |
| 「dep8 + MTP 是官方那条曲线的配置」 | **不是**。官方 8K/1K 的 frontier 全是 **wide-EP（dep16/24/32/40）**，`dep8` 已被 [InferenceX PR #1586](https://github.com/SemiAnalysisAI/InferenceX/pull/1586) 作为 *"dominated … superseded by wide-EP frontier"* **删除** | InferenceX `benchmarks/multi_node/srt-slurm-recipes/sglang/deepseek-v4/8k1k/` |

**11,200 的原始出处**：[PyTorch 官方博客 2026-06-23](https://pytorch.org/blog/serving-deepseek-v4-on-gb300-with-sglang-5x-higher-throughput-at-the-same-interactivity-since-day-0)，原文是「the June 2026 MTP curve delivers ~11,200 tok/s/GPU **at roughly 50 tok/s/user**」。**它是一条 Pareto 曲线上的点，不是某个单一配置**，而且限定了交互性（50 tok/s/user ≈ TPOT 20ms）。

> **我们的工作点完全不同**：TPOT 钉在 58–63ms ≈ **16 tok/s/user**。也就是说我们在**交互性差 3 倍**的点上，吞吐还低 18%——两个轴都在人家里面，不是简单的 throughput/latency 取舍。

### 11.2 口径复核

InferenceX [`utils/process_result.py`](https://github.com/SemiAnalysisAI/InferenceX/blob/main/utils/process_result.py) L136-163 定义了三个口径：

```python
output_tput_denominator = decode_gpus if decode_gpus > 0 else total_gpus
'tput_per_gpu':        total_token_throughput / total_gpus       # (in+out) ÷ 全部 GPU
'output_tput_per_gpu': output_throughput / output_tput_denominator   # out ÷ decode GPU ← 本文用的
'input_tput_per_gpu':  (total - output) / prefill_gpus
```

我们的数按两种口径：

| 口径 | 我们 | vs 11,200 |
|---|---|---|
| `output_tput_per_gpu`（out ÷ 8 decode 卡） | 9,032 | 80.6% |
| `tput_per_gpu`（(in+out) ÷ 64 卡） | **10,180** | **90.9%** |

博客标题那个 11,200 到底指哪个字段，公开材料没写死。**先记下这个不确定性**——它可能一下子把 gap 从 19% 缩到 9%。

### 11.3 官方 recipe（16 节点可复刻的两种）

官方跑 18 节点 / 72 GPU（满 NVL72）。我们 16 节点，按同比例可摆两种：

| 配置 | prefill | decode | 官方对应 | 建议 conc |
|---|---|---|---|---|
| A | 12 × dep4（48 卡） | 1 × **dep16**（4 节点 16 卡） | `14p1d-dep4-dep16-18-c8192` | 8192 |
| B | 8 × dep4（32 卡） | 1 × **dep32**（8 节点 32 卡） | `10p1d-dep4-dep32-18-c2500` | 2500 |

**注意官方每个拓扑只用一个并发点**（dep16 用 8192、dep32 用 2500），不是我们这种 14 路 × 600 平均分。

### 11.4 实验记录

#### Exp A：dep8 + MTP + 官方 decode 参数 → ❌ **配置冲突，不成立**

改动：`mem-fraction-static 0.85→0.94`、`max-running-requests 8192→18432`、`swa-full-tokens-ratio 0.1→0.20`、加 `SGLANG_OPT_USE_ONLINE_COMPRESS=1`。

结果：decode 启动即崩。

```
AssertionError: online c128 does not support MTP
```

> **这是本轮最有价值的一条**：**KV 压缩 V2（online c128）与 MTP 互斥，只能二选一**。它直接解释了为什么官方五个 wide-EP 8K/1K recipe **全都没有 speculative 配置**——不是「MTP 在饱和时收益为负」这么模糊的理由，而是**开了压缩就用不了 MTP**，官方选了压缩。
>
> 所以 Exp A / Exp B 不是两个独立实验，**它们被强制合并**。

#### Exp B / B2：把官方 decode 参数搬到 dep8 → ❌ **三次失败，路本身是死的**

| 尝试 | 配置 | 结果 |
|---|---|---|
| A | dep8 + MTP + `0.94` / `18432` / online-compress | `AssertionError: online c128 does not support MTP` |
| B | 去 MTP，其余同上 | decode 起来了、etcd 注册成功，**但压测 100% 失败**：`batch.retract_decode()` → `NotImplementedError` |
| B2 | 再把 `max-running-requests` 退回 `8192` | `torch.OutOfMemoryError`（276.62 GiB 卡上只剩 244 MiB） |

**三个失败各自暴露一条硬约束**：

1. **online c128 与 MTP 互斥** —— 只能二选一。
2. **online c128 没实现 `retract_decode`** —— 一旦并发超过 KV 容量、需要抢占回退，就直接抛 `NotImplementedError`，**所有在途请求全挂**。所以开压缩时 `max-running-requests` 必须保守到**永不触发抢占**。官方那个 `18432` 是给 wide-EP 的（dep16/32 的 KV 池大得多）。
3. **`mem-fraction-static 0.94` 是 wide-EP 专用** —— dep8 时模型只摊在 8 张卡上，单卡权重占用是 dep32 的 4 倍，0.94 直接把激活空间挤没。

> **结论**：官方那套 decode 参数是**跟 wide-EP 共同设计的一个整体**，拆开逐项搬到 dep8 上会以三种不同方式失败。**dep8 + 官方参数这条路不存在**——要用官方参数就必须同时换成 wide-EP 拓扑。这反过来印证了 PR #1586 把 dep8 判为 dominated 并删除的决定。

#### Exp C：wide-EP decode dep16（12 prefill + 4 节点 decode）→ ⚠️ **总吞吐 +16.7%，但 per-GPU 腰斩**

全量干净部署（`sgl-0..3` = decode dep16 跨 4 节点，`sgl-4..15` = 12 prefill），12 路 × conc 683 ≈ 8196（对齐官方 dep16 的 `c8192`）：

| | dep8 baseline | **dep16 wide-EP** | 变化 |
|---|---|---|---|
| 聚合 output tok/s | 72,256 | **84,321** | **+16.7%** |
| output ÷ decode-GPU | 9,032（÷8）| **5,270（÷16）** | −41.7% |
| TPOT 中位 | 60ms | **33ms** | **−45%** |
| TTFT 中位 | 45s | 54s | +20% |
| 反推 decode batch | 4,335 seq | **2,783 seq** | **−36%** |
| `tput_per_gpu`（(in+out)÷64） | 10,180 | **11,880** | +16.7% |

**怎么读这组数**：

1. **wide-EP 本身是有效的** —— 同样 64 张卡，总输出吞吐涨 16.7%，TPOT 从 60ms 砍到 33ms（交互性接近翻倍）。MoE 摊得更宽确实更高效。
2. **但 per-decode-GPU 腰斩，原因是 decode 没吃饱** —— 反推 batch 从 4,335 掉到 2,783。加宽 decode 到 4 节点，prefill 就从 14 个掉到 12 个，而 **prefill 本来就是瓶颈**（TTFT 45s → 54s）。decode 卡数翻倍、喂进来的活反而少了，per-GPU 自然崩。
3. **这是节点数的硬约束，不是配置问题** —— 官方 `14p1d-dep4-dep16` 用 **18 节点 / 72 卡**（14×4 prefill + 16 decode）。我们 16 节点 / 64 卡，摆 dep16 只剩 12 prefill。**在 16 节点上复刻官方 wide-EP 配比在数学上就不成立。**

> **口径再次变得关键**：按 `tput_per_gpu` 算，dep16 是 **11,880 > 官方 11,200**。按 `output_tput_per_gpu` 算只有 5,270。同一次测量，两个口径一个超标一个腰斩——**在确定官方到底用哪个字段之前，不要再拿这个数字做决策**。

#### Exp D：dep12（3 节点 decode）+ 13 prefill

在 16 节点约束下找 prefill/decode 最优配比（对应官方 `15p1d-dep12`）。结果见下（进行中）。

### 11.5 待验证清单（按 收益/成本 排序）

| # | 动作 | 成本 | 预期 |
|---|---|---|---|
| 1 | Exp B：官方 decode 参数 + 去 MTP | 只重启 decode，~35min | 未知，先拿到 |
| 2 | **wide-EP decode dep16 / dep32** | 重排 fleet，~50min | **调研认为主要 gap 在这** |
| 3 | 按官方单一并发点压测（dep16 用 8192 / dep32 用 2500） | 免费 | 口径对齐 |
| 4 | 切官方 pinned 镜像 `nightly-dev-20260527-14f81a67` | 拉镜像 + 重建 | 低（我们的更新） |
| 5 | EPLB + Waterfill（cookbook 称支持 megamoe，与旧文档结论矛盾） | 中 | 待验证 |
