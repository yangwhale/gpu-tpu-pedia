# SGLang · DeepSeek-V4-Pro · GB300 NVL72 复现 Runbook

> **本文定位**：**只讲怎么跑通**的操作手册。所有命令可直接复制，所有引用的脚本/manifest 都已入库（`manifests/` + `scripts/`）。
> 旧版探索文档 `sglang-v4-gb300-benchmark.md` 已删除；其中仍成立的实测数据（架构代际对比 / PD 配比公式 / 历史扫描 / 已证无效的尝试）蒸馏在 **§12**。
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
> 已删除的旧文档把这两套配方分散在它的 §1（checkpoint 说明）和 §13.A（启动脚本）两处，中间无任何交叉提示 —— 照那份启动脚本抄、手上又是 NVFP4 权重，必然失败。（本文没有 §13，这里说的是旧文档的章节号。）

---

## 0. TL;DR

| 项 | 值 |
|---|---|
| 目标配置 | **14 prefill（每个 dep4）+ 1 decode（dep8 跨 2 节点）+ MTP** |
| 节点需求 | 16 节点 GB300，**必须同一 NVL72 域**（本文用 `gb300-pool-0002`） |
| 模型 | `/mnt/ssd/DeepSeek-V4-Pro`（**官方原装**，806G，节点本地 SSD）—— 不是 `-NVFP4` 那份 |
| 镜像 | `lmsysorg/sglang:nightly-dev-cu13-20260720-b3570a45` |
| 编排 | Dynamo（`dynamo.sglang` worker + `dynamo.frontend`）+ NATS + etcd |
| 关键参数 | `--swa-full-tokens-ratio` **按 ISL 调**：4K→`0.15`，8K→`0.10`。这一个值决定 54% 的吞吐，见 §11.5 |
| 最好成绩 | ISL 4096：端到端 **10,704** per-decode-GPU（官方 11,200 的 **95.6%**）／ decode 自报峰值 **12,070**（**107.8%**）。照本文默认参数跑是 10,614 / 12,063，差 1% 以内 |
| | ISL 8192：端到端 **9,354**（83.5%）|
| 部署耗时 | 全 fleet 从零 **16–24min**（40236 自愈占大头）／ 只换 decode 参数 **16–18min**（§6.1）|

**为什么是 14 prefill**：实测 14→16 prefill 只涨 2%（8,809 → 8,993），已收敛。14 是性价比拐点。

**最短路径**：§1 前置 → §2 起 fleet → §3 查模型（**含 §3.1 RAID 挂载**）→ §4 分发 → §5 起 worker → §6 自愈到 etcd 全绿 → §7 起 frontend → §8 压测（**第一轮当 warmup 丢掉**）。
调参数不用重来一遍，看 **§6.1**。撞到怪事先翻 **§9 故障速查**，那里每一行都是真踩过的。

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
#    ★ SWA_RATIO 按你要跑的 ISL 设：4096→0.15（脚本默认），8192→0.10。这一个值决定 54% 的吞吐，见 §11.5
D0=$(kubectl get pod sgl-0 -o jsonpath='{.status.podIP}')
kubectl exec sgl-0 -- bash -c "SWA_RATIO=0.15 setsid nohup bash /tmp/decode-dep8.sh 0 $D0:5000 > /tmp/srv.log 2>&1 </dev/null &"
kubectl exec sgl-1 -- bash -c "SWA_RATIO=0.15 setsid nohup bash /tmp/decode-dep8.sh 1 $D0:5000 > /tmp/srv.log 2>&1 </dev/null &"

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

单个 worker：

| 阶段 | 耗时 | 观测点 |
|---|---|---|
| 权重加载 | ~4–5min | HBM 从 0 涨到 ~260 GiB |
| CUDA graph capture | 再 ~3–5min | HBM 稳定不动 |
| 向 etcd 注册 | 之后 | §5.2 的计数开始涨 |
| **单 worker 总计** | **8–12min** | 别在 5 分钟时下结论 |

整个 fleet（含 §6 自愈轮次，实测多次）：

| 场景 | 耗时 | 备注 |
|---|---|---|
| 全 fleet 从零（16 pod） | **16–24min** | 波动来自 40236 中招的台数 |
| 只换 decode 参数 | **16–18min** | 见 §6.1 —— **不需要重建 pod fleet** |
| 只补几台崩掉的 prefill | ~8min/轮 | §6 自愈循环 |

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

⚠️ **同一个竞态有两种报错文本，自愈脚本必须两个都匹配**：

```
zmq.error.ZMQError: Address already in use (addr='tcp://127.0.0.1:40236')
ValueError: rpc_port at 40236 is not available in 30 seconds. rpc_port is used by
            a process already. process.name()='python3' process.cmdline()=[...]
```

第二种出现在**上一次崩溃的残留进程还占着端口**时（僵尸没被内核收掉就重启了）。我的自愈循环只 grep 了第一种，结果两台 prefill 卡了 5 分钟没人管。正确的匹配：

```bash
grep -aE 'Address already in use|rpc_port at 40236 is not available' /tmp/srv.log
```

碰到第二种要**先 `pkill -9 -f 'dynamo[.]sglang'` 清残留再重启**，光重启会继续撞同一个占用者。

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

> ⚠️ **etcd 只在「判就绪」这个方向上权威，反过来「判存活」会骗人**：worker 崩掉之后 lease TTL 还没过期，它的 key 会继续挂在 etcd 里好几分钟。我据此误判过「14 台 prefill 全程存活」，实际全崩了（§11.6 #8）。
>
> | 要判断的事 | 用什么 |
> |---|---|
> | 起来了没（readiness）| **etcd 注册数** ✅（显存和日志会提前变绿）|
> | 还活着没（liveness）| **`nvidia-smi` 显存**  ✅（etcd 会滞后虚报）|
>
> 两个方向刚好用相反的判据，别混用。§6 的自愈循环正是「显存查死亡 → etcd 做终检」。

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

### 6.1 只改 decode 参数时的正确流程（16–18 分钟，别重建整个 fleet）`[已验证]`

做参数实验（`swa-full-tokens-ratio`、MTP 深度、`mem-fraction-static`）时，**不需要把 16 个 pod 全删重来**。但也**不能只重启 decode 就完事** —— 有两个坑必须一起处理：

- **decode 一消失，14 台 prefill 会全部自杀**（§11.6 #6）。躲不掉，只能预算上重建时间。
- **`pkill -9` 一个满载的 decode 会泄漏 ~97 GB 显存/卡**（§11.6 #7），下一次启动必 OOM。所以**要删 pod 让 StatefulSet 重建，不要 pkill**。

四步，实测 16–18 分钟：

```bash
# ① 删 decode pod（唯一能真正还显存的办法），等到显存归零 —— 实测 56s
kubectl delete pod sgl-0 sgl-1 --wait=false
until [ "$(kubectl get pod sgl-0 sgl-1 --no-headers | grep -c Running)" = 2 ] &&       [ "$(kubectl exec sgl-0 -- nvidia-smi --query-gpu=memory.used            --format=csv,noheader,nounits | head -1 | tr -d ' ')" -lt 5000 ]; do sleep 10; done

# ② 起 decode（~11min 到 etcd 注册）
for i in 0 1; do kubectl exec -i sgl-$i -- bash -c "cat > /tmp/dec.sh" < scripts/decode-dep8.sh; done
D0=$(kubectl get pod sgl-0 -o jsonpath='{.status.podIP}')
kubectl exec sgl-0 -- bash -c "SWA_RATIO=0.15 setsid nohup bash /tmp/dec.sh 0 $D0:5000 > /tmp/srv.log 2>&1 </dev/null &"
kubectl exec sgl-1 -- bash -c "SWA_RATIO=0.15 setsid nohup bash /tmp/dec.sh 1 $D0:5000 > /tmp/srv.log 2>&1 </dev/null &"
# 等 §5.2 的 decode workers = 1

# ③ 重起 14 台 prefill（它们已经被 ② 带崩了）+ §6 自愈循环，~6min
# ④ 重起 14 个 frontend（§7 的告警），跑单条 e2e 确认
```

**每一步都要验，跳过任何一步的失败都会伪装成「参数不行」**：

| 步 | 验什么 | 跳过的后果 |
|---|---|---|
| ① | `nvidia-smi` 归零 | 下一次启动 OOM，被误判成「这个参数内存不够」|
| ② | etcd `decode workers: 1` | — |
| ③ | 显存 >200G ×14 **且** etcd `prefill: 14` | 池子不满，压出来的数腰斩 |
| ④ | 单条 `curl /v1/completions` 有输出 | 压测全 0 |

> 本轮实测：跳过 ① 直接 pkill，导致 MTP `steps=2` 连续 OOM 三次，我一度判定「MTP2 内存不够」还降了 `cuda-graph-max-bs` —— 清干净之后它一次就起来了。**被污染环境下得出的失败结论一律作废。**

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

⚠️ **只要重启过任何 prefill / decode server，14 个 frontend 必须全部重启**（§11.6 #9）。frontend 缓存了旧 instance 的连接，不会自己重新发现：etcd 里 `prefill=14 decode=1` 全绿、单条 `curl` 却挂死超时，decode 日志停在重启那一刻并刷几百条 `Attempting to reconnect to <prefill>:30001`。

```bash
for i in $(seq 2 15); do kubectl exec sgl-$i -- bash -c "pkill -9 -f 'dynamo[.]frontend'" & done; wait
sleep 8
# 再照上面的循环重起，然后一定要跑一次单条 e2e 验证
```

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

**先明确你在哪个工作点** —— 同一套系统的「健康值」在两个工作点上差 3 倍：

| 工作点 | TPOT 中位 | TTFT 中位 | 说明 |
|---|---|---|---|
| **低延迟**（官方 11,200 那条曲线的位置）| 20–35ms | < 10s | ≈ 50 tok/s/user，batch 小 |
| **最大吞吐**（本文 §11.5 配置 #3 / #7）| 58–85ms | 60–85s | batch 撑到 KV 池上限，用延迟换吞吐 |

**两个都是正常的，选哪个取决于 SLA。** 下面这张表判断的是「有没有故障」，不是「快不快」：

| 指标 | 异常信号 | 含义 |
|---|---|---|
| TPOT 正常但吞吐只有目标 40–60% | — | **prefill 池子不满**，先查 §5.2 的 etcd 注册数 |
| TTFT 冲到分钟级而 TPOT 反而偏低 | — | prefill 喂料不足，decode 在饿着 |
| `full token usage` 与 `swa token usage` 差 0.3 以上 | — | **KV 池预算划错**，见 §11.5 结论 1（这条影响 54% 吞吐）|
| 各路 `Benchmark duration` 差 >15% | — | 没有真正并发，聚合数不可信 |

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
| **换了个 decode 参数就 `RuntimeError: Not enough memory`，改回原参数也一样起不来** | 上一次 `pkill -9` 满载 decode 泄漏了 ~97 GB/卡，环境已被污染 | **删 pod 让 StatefulSet 重建**（56s，显存归零），别 pkill。见 §6.1 / §11.6 #7 | `[已定位并修复]` |
| **etcd 显示 14 台全在线，压测却完全没输出** | lease TTL 未过期，死掉的 worker 还挂在注册表里 | 用 `nvidia-smi` 显存判存活，etcd 只用来判就绪。见 §5.2 / §11.6 #8 | `[已定位并修复]` |
| **重启 server 后单条 `curl` 挂死超时**，decode 日志刷 `Attempting to reconnect to <ip>:30001` | frontend 缓存了旧 instance 连接，不会自动重新发现 | 14 个 frontend 全部重启，见 §7 | `[已定位并修复]` |
| prefill 报 `rpc_port at 40236 is not available in 30 seconds` | 40236 竞态的第二种报错文本（上次崩溃的残留进程还占着端口） | 先 `pkill -9 -f 'dynamo[.]sglang'` 清残留再重启，见 §5.1 | `[已定位并修复]` |

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

**结论**：对标口径应取 **conc600 = 9,168**（与旧文档 8,993 同工作点，判定复现成功）。

> ⚠️ **本节的扫描全部跑在 `swa-full-tokens-ratio = 0.1` 上，这个值对 ISL 8192 恰好接近最优，但整段的天花板结论后来被 §11.5 推翻了。**
>
> 我当时写的是「~9,200–9,400 是 dep8 + MTP + nightly 镜像的实际天花板，剩下的 gap 是单卡内核成熟度差，只剩换 pinned 镜像一条路」。**错了**：同一个镜像、同一套硬件，只把 `swa-full-tokens-ratio` 改成 4096 场景的最优值，端到端就到了 **10,704**、decode 峰值 **12,070**（超过官方标称）。
>
> **别把「在某组参数下扫出来的上界」当成系统天花板。** 并发扫描只在你已经把其他参数调对的前提下才测得出天花板。

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

### 10.5 审计轮 3：swa-ratio 消融（2026-07-26 下午）

前两轮审计验的是「照文档能不能跑通」，这一轮验的是「文档给的参数对不对」。**结论是不对** —— 见 §11.5 的 7 组消融。

| 步骤 | 结果 | 实测 |
|---|---|---|
| 全 fleet 从零重建 | ✅ | 16–24min（40236 自愈 2–4 轮）|
| §6.1 只换 decode 参数 | ✅ | **16–18min**，比全量重建省 1/3 |
| swa-ratio 0.20 / 0.15 + MTP steps 1/2 共 7 组 | ✅ | §11.5 完整表 |
| 最优配置复现 | ✅ | 端到端 **10,704** / 峰值 **12,070** |

**本轮新抓到 4 个坑**（全部进了 §11.6）：`pkill` 满载 decode 泄漏 97 GB/卡、etcd 注册数不能判存活、server 重启后 frontend 必须一起重启、40236 竞态的第二种报错文本。

**本轮自己推翻自己两次**：

1. 我先写下「只重启 decode 不影响 prefill，14 台全程在线」（依据是 etcd 显示 14）。十分钟后查显存 —— **14 台全是 0，早就崩光了**。etcd lease TTL 在骗人。
2. 我先判定「MTP `steps=2` 内存不够」（连续 OOM 三次，还降了 `cuda-graph-max-bs` 重试）。清掉显存泄漏后 **一次就起来了** —— 三次失败全跑在被污染的环境里。

> 两次都是同一类错误：**拿一个「看起来很可靠的信号」当结论，没去查它成立的前提**。etcd 注册数在启动时确实权威，显存 OOM 报错也确实是真的 —— 但前者不适用于崩溃检测，后者的因不是我以为的那个。

### 10.6 本次复测发现的旧文档缺陷（均已在本文修正）

1. **模型与 MoE backend 配方错配（最严重）** —— 两套互斥配方分散在旧文档的 §1 和 §13.A 两处，中间无交叉提示。照那份启动脚本抄 + 手上是 NVFP4 权重 = 必挂，且报错（tensor 384 vs 48）完全指不到根因。→ 提到文首告警。
2. **就绪判据不足以判就绪** —— 旧文档给的「HBM>200G」和本文初稿给的 `Load weight end` **都会骗人**。崩溃发生在读权重之后，两个信号都已经变绿了。唯一权威是 etcd 注册数。→ §5.2。
3. **pod 生成器未入库** —— 旧文档引用 `gen18-0001.py`，仓库里没有，第一步就卡死。→ `manifests/sgl-fleet.yaml`。
4. **漏 `--moe-runner-backend deep_gemm`** —— 旧文档强调"一个字都不能漏"，却依赖 `auto` 的默认行为。nightly 镜像更新后 auto 改选 flashinfer，整条 megamoe 路径直接崩。**「依赖默认值」型埋雷**：写文档时能跑 ≠ 半年后能跑。
5. **裸 pod 与 DRA 冲突未说明** —— 生成器批量创建裸 pod，与 DRA channel 预留机制冲突。→ StatefulSet。
6. **GCS bucket 名写成占位符** —— 全文 `gs://<bucket>`，真名 `gs://chrisya-gb300-models`。
7. **压测工具来源未记** —— 只说用 InferenceX `sa-bench`，没说从哪来、怎么装、装到哪（**按节点不按 pod**）。→ §8.2。
8. **压测启动方式没写对** —— 裸 `for` 循环 `kubectl exec` 批量起，实测 14 路只活 6 路且不报错。→ §8.3 的 `sleep 4` + 校验重试。

> **方法论**：本文每一节都是「先在集群上真跑一遍，再写进来」，写完又清空环境**照文档从零重跑了 3 轮**（§10.3 / §10.4 / §10.5）。上面 8 条全部是执行中撞出来的，读文档发现不了。
>
> **两个最贵的**：第 2 条（就绪判据）不让部署失败、只让性能腰斩，而所有常规健康检查都是绿的；而审计轮 2 推翻的那条错误归因，是**我自己在轮 1 刚写下的**——它读起来完全合理，只有真跑第二遍才暴露。**这就是为什么复现文档必须自己审计自己。**

---

## 11. 冲击官方 11,200：调研结论 + 实验 `[已完成]`

起点是稳定的 **9,032–9,168**（四次热态测量，均值 9,107，±0.8%），差官方 11,200 约 20%。这一节记录「那 20% 到底在哪」的完整调研和实验。

**结论先行**：gap 的大头是 **KV 池预算划错**（`swa-full-tokens-ratio`，一个参数值 +54%），不是镜像、不是拓扑、也不是 prefill 数量。最终 ISL 4096 端到端 **10,704**（官方的 95.6%）、decode 自报峰值 **12,070**（**107.8%**）。直接看 **§11.5**。

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

| 口径 | 当时（ratio 未调优） | **最终（§11.5 配置 #7）** | vs 11,200 |
|---|---|---|---|
| `output_tput_per_gpu`（out ÷ 8 decode 卡） | 9,032（80.6%）| **10,704** | **95.6%** |
| `tput_per_gpu`（(in+out) ÷ 64 卡） | 10,180（90.9%）| — | — |

博客标题那个 11,200 到底指哪个字段，公开材料没写死。**这个不确定性最后没派上用场** —— 调对参数后，即使按最保守的 `output_tput_per_gpu` 口径也已经到 95.6%，不需要靠换口径去缩 gap。

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

#### Exp E：攒 batch 组合 → ✅ **+4.3%**（本节四个实验里唯一有效的，但很快被 §11.5 的 ratio 调优盖过）

两个我一直用着默认值的开关：

| 参数 | 默认 | 设成 | 作用 |
|---|---|---|---|
| `--disaggregation-decode-polling-interval` | **1** | **8** | decode 每 8 个 forward pass 才去 prefill 侧取一次已传输的 KV，而不是每步都取。请求攒起来形成更大的 decode batch |
| `--enable-prefill-delayer` | False | **on** | DP attention 下延迟 prefill、减少 rank 空转（配 `--prefill-delayer-max-delay-passes 30 --prefill-delayer-queue-min-ratio 0.5 --prefill-delayer-max-delay-ms 3000`）|

| | baseline dep8 | **+攒 batch**（两次独立测量）|
|---|---|---|
| output/decode-GPU | 9,107（4 次均值）| **9,502.6** / **9,409.1** |
| vs 官方 11,200 | 81.3% | **84.8% / 84.0%** |
| TPOT 中位 | 60ms | **57ms** |
| TTFT 中位 | 45s | 45s |

**+3.3~4.3%（两次独立测量都复现）且 TPOT 反而降了 3ms** —— 不是拿延迟换吞吐，是把 decode 原本的空转填上了。相比之下 conc 扫描、wide-EP、官方 decode 参数三条路全是死的。

> 第二次测量走的是完整「清空 → 重建 → 冷跑 → 热跑」流程，冷 8,908 / 热 9,409，冷热差 +5.6%（与 §8.4 的 +6.5~7.0% 同量级）。

### 11.5 ⭐ 决定性发现：唯一真正有效的旋钮是 `swa-full-tokens-ratio`

#### 先说怎么读数

**光看 sa-bench 的聚合数字会一直被 prefill 拖着，看不到 decode 自己的能力。** decode 引擎每隔几百毫秒自报一次瞬时速率，两个数要一起看：

```bash
# 峰值（decode 引擎的能力上限）—— srv.log 含 NUL，必须先 tr -d
kubectl exec sgl-0 -- bash -c "tr -d '\000' < /tmp/srv.log | grep 'gen throughput' | \
  sed -E 's/.*#running-req: ([0-9]+).*full token usage: ([0-9.]+).*swa token usage: ([0-9.]+).*accept len: ([0-9.]+).*gen throughput \(token\/s\): ([0-9.]+).*/\1 \2 \3 \4 \5/' | \
  awk 'NF==5' | sort -k5 -n | tail -1"
# 输出: running-req  full占用  swa占用  accept-len  gen-tok/s
```

> **这个数是 per-DP-rank（= per GPU），不是引擎总和**。交叉验证：某轮 p50 = 9,587 × 8 rank = 76,696，同轮 sa-bench 实测聚合 output = 74,833，差 2.5%。若是引擎总和则会差 8 倍。

#### 完整消融表

统一条件：14× prefill dep4 + dep8 decode（2 节点 8 GPU），sa-bench random 数据，`--disable-radix-cache`，OSL 1024。

**ISL 4096 / 每 frontend conc 1050 × 14**

| # | MTP | `swa-ratio` | **端到端 per-decode-GPU** | TPOT | **峰值 tok/s/GPU** | 峰值 req/rank | full / swa 占用 | accept len |
|---|---|---|---|---|---|---|---|---|
| 1 | steps1 draft2 | 0.20 | 6,887 | 58.0 ms | 11,851 | 727 | 0.93 / 0.63 | 1.88 |
| 2 | steps2 draft3 | 0.20 | 8,802 | 72.4 ms | 10,279 | 708 | 0.90 / 0.52 | 2.36 |
| 3 | **steps1 draft2** | **0.15** | **10,614** | 77.7 ms | **12,063** | **886** | **0.92 / 0.89** | 1.88 |
| 7 | steps1 draft2 | 0.15 + `mem-frac 0.88` | **10,704** | 85.3 ms | **12,070** | **999** | 0.92 / 0.67 | 1.88 |

**ISL 8192 / conc 600 × 14**

| # | MTP | `swa-ratio` | **端到端 per-decode-GPU** | TPOT | 峰值 tok/s/GPU | 峰值 req/rank | full / swa 占用 | accept len |
|---|---|---|---|---|---|---|---|---|
| 4 | steps1 draft2 | **0.10** | **9,354** | 57.1 ms | 10,490 | 588 | 0.78 / **0.96** | 1.87 |
| 5 | steps1 draft2 | 0.15 | 9,053 | 48.8 ms | 10,035 | 410 | 0.80 / 0.37 | 1.89 |
| 6 | steps2 draft3 | 0.20 | 8,077 | 43.5 ms | 9,217 | 373 | 0.93 / 0.37 | 2.39 |

**对标官方 11,200**：4K 端到端 10,614 = **94.8%**，4K 峰值 12,063 = **107.7%**。

#### 结论 1：最优点 = 两个 KV 池同时饱和

`swa_tokens = full_tokens × ratio`，两个池吃同一份预算（实测总量恒定 ≈ 4.01 M token）。**调 ratio 就是在两个池之间划线，划错一边就有一半浪费**：

| | full 占用 | swa 占用 | 结果 |
|---|---|---|---|
| 4K @ ratio 0.20 | 0.93 | 0.63 | full 先满，SWA 池浪费 37% → batch 卡在 727 |
| **4K @ ratio 0.15** | **0.92** | **0.89** | **两边同时到顶 → batch 冲到 886** |
| 8K @ ratio 0.15 | 0.80 | 0.37 | SWA 池浪费 63% → batch 只有 410 |
| 8K @ ratio 0.10 | 0.78 | 0.96 | swa 先满，但已接近平衡 → batch 588 |

**最优 ratio 跟 ISL 绑定，不是常数**：序列越长 full-attn KV 占比越高（SWA 只留窗口、full 留全部），所以长 ISL 要更低的 ratio。实测甜点 **ISL 4096 → 0.15，ISL 8192 → 0.10~0.12**。

> **调参方法**：跑一轮看 `full token usage` 和 `swa token usage`，**哪个先到 0.9+ 就给哪个加预算**（swa 先满 → 调高 ratio；full 先满 → 调低）。目标是两个数同时落在 0.88–0.93。

#### 结论 1b：batch 到 ~890 就算力饱和，再堆没用

把 `mem-fraction-static` 从 0.85 提到 0.88（KV 总预算 +13%）：

| | batch/rank | 峰值 tok/s/GPU | 端到端 |
|---|---|---|---|
| mem-frac 0.85 | 886 | 12,063 | 10,614 |
| mem-frac 0.88 | **999** (+13%) | 12,070 (**+0.06%**) | 10,704 (+0.8%) |

**batch 涨 13%，吞吐纹丝不动。** 说明在 ~890/rank 之后 decode 已经从「KV 容量受限」转成「算力受限」—— **12,070 tok/s/GPU 就是 dep8 在 GB300 上的硬天花板**，kernel 的 1024/rank 上限（999 已经摸到 97%）根本不是真正的约束。

> 推论：想再往上只有两条路 —— 更快的 attention/MoE kernel，或者换拓扑摊薄每卡计算量。**继续调 batch 相关的参数（`max-running-requests`、`cuda-graph-max-bs`、`mem-fraction`）已经没有意义。**

#### 结论 2：MTP `steps=2` 峰值亏 13%、端到端赚 28%（两个指标方向相反）

| | running-req | accept len | 反推单步耗时 | 峰值 tok/s/GPU | 端到端 |
|---|---|---|---|---|---|
| steps=1 draft=2 | 727 | 1.88 | **115.3 ms** | **11,851** | 6,887 |
| steps=2 draft=3 | 708 | 2.36 | 162.5 ms | 10,279 | **8,802** |

（单步耗时 = running-req × accept-len ÷ gen-throughput）

accept len +26%，但单步耗时 **+41%** —— 多那一轮 draft forward 在大 batch 下不划算。**这是官方 recipe 高并发下坚持 `steps=1` 的量化理由**，cookbook 只含糊写了「饱和时收益为负」。

而端到端反而 +28%，因为「每步多吐 token」的收益一直在，「每步更慢」的代价只在 batch 撑到顶时才显著；prefill 喂不满时 batch 大部分时间不在顶点。

> **两个指标不矛盾，选哪个取决于目标**：对标 decode 引擎能力用峰值 → `steps=1`；对标真实业务吞吐用端到端 → prefill 受限的部署里 `steps=2` 可能更好。**本文最终推荐配置 #3（`steps=1` + ratio 按 ISL 调）**，因为它在两个指标上同时最优。

#### 结论 3：之前认定的「瓶颈全在 prefill」只对了一半

早期（§11.4）我判定 gap 全在 prefill，依据是 ISL 8192 稳态 batch 只有 588/rank 而峰值能到 933。**这个判断在当时的参数下成立，但它掩盖了一个大得多的旋钮**：同样 14 台 prefill、同样并发，只把 `swa-ratio` 从 0.20 改到 0.15，端到端就从 6,887 涨到 10,614（**+54%**）。

真实的因果分层：

| 因素 | 量级 | 可修性 |
|---|---|---|
| **KV 池分配失衡** | **+54%** | 一个参数值 |
| prefill 供给不足 | +4.3%（攒 batch） | 要加机器 |
| kernel 1024/rank 硬顶 | 还没摸到（886/1024 = 87%） | 要改上游 |

> **教训**：「瓶颈在 X」这类结论必须标注它成立的参数前提。我在 ratio 设错的情况下测出「batch 上不去」，然后把原因全归给了 prefill —— 方向没错，权重完全估错了。

#### Exp D：dep12（3 节点 decode）+ 13 prefill

在 16 节点约束下找 prefill/decode 最优配比（对应官方 `15p1d-dep12`）。**已放弃**——上表证明该调的是 KV 池划分，不是 decode 拓扑。

### 11.6 硬约束清单（都是官方 recipe 里看不出来、只有真跑才撞得到的）

按撞到的顺序，每条都花了一整轮部署（30–50 分钟）才定位：

| # | 约束 | 报错 | 为什么隐蔽 |
|---|---|---|---|
| 1 | **KV 压缩 V2 与 MTP 互斥** | `AssertionError: online c128 does not support MTP` | 这才是官方 wide-EP recipe 全不开 MTP 的真正原因，cookbook 只含糊说「饱和时收益为负」|
| 2 | **online c128 未实现 `retract_decode`** | `NotImplementedError` | decode 起得来、etcd 注册成功、单条 e2e 也通过；**只有压到超 KV 容量才炸**，且一炸就是全部在途请求 |
| 3 | **`mem-fraction-static 0.94` 是 wide-EP 专用** | `torch.OutOfMemoryError` | dep8 单卡扛 1/8 模型，是 dep32 的 4 倍，0.94 把激活空间挤没 |
| 4 | **attention plan kernel 硬顶 1024 请求/rank** | `c_plan.cuh:522: GPU plan only support batch size up to 1024` | SGLang 自己 hardcode 的 `kMaxPrefillBatchSize`，源于「单 CUDA block + 每线程一请求」的实现（block 线程上限就是 1024）+ 静态 shared memory 数组。**不是硬件限制，是实现取舍**（为避开 MTP/graph capture 时的 host sync）|
| 5 | **PD 两侧 `context-length` 必须一致** | `Decode handshake failed` | 只改 decode 会让 KV 布局对不上。**decode 照常注册、frontend 全 200、单条 e2e 也过**，只有压测才暴露。而且这个改动本身多余——SGLang 的 KV 页按需分配，短请求本就不占满上限 |
| 6 | **重启 decode 会把 14 台 prefill 全带崩** | prefill 侧 `No live scheduler processes found` → `kill_process_tree(include_parent=True)` | disagg 对端一消失，prefill 的 scheduler 自杀。**pkill 重启和删 pod 重建都一样，躲不掉**。做 decode 实验就要预算上 prefill 的重建时间 |
| 7 | **`pkill -9` 一个满载的 decode 会泄漏 ~97 GB 显存/卡** | 下次启动 `RuntimeError: Not enough memory`，**参数一个字没改也起不来** | 见下方专述。报错还会把人往「调大 mem-fraction」这个完全相反的方向带 |
| 8 | **etcd 注册数不能当存活判据** | 14 台 prefill 全崩了，etcd 仍然显示 `prefill=14` | lease TTL 没过期，死进程的 key 还挂着好几分钟。**必须看显存**（见下方） |
| 9 | **重启任何 server 之后，frontend 必须一起重启** | e2e curl 挂死超时，decode 日志停在重启前那一刻、几百条 `Attempting to reconnect to <prefill>:30001` | frontend 缓存了旧 instance 的连接，etcd 里 `prefill=14 decode=1` 全绿也没用。**14 个 frontend 全部 `pkill -9 -f 'dynamo[.]frontend'` 重起，e2e 立刻恢复** |
| 10 | **`swa-full-tokens-ratio` 的方向很容易搞反** | 调到 0.056 后 `full token usage: 0.95` @ 仅 356 running-req，rank 挂掉 | **源码定义**：`swa_full_tokens_ratio = SWA池 ÷ full池`（`swa_tokens = full_tokens × ratio`，`pool_configurator.py:387`）。所以 **ratio ↑ = SWA 池变大 / full 池变小**，ratio ↓ 反之。我一度以为反了，把本已吃紧的 SWA 又砍一半。**实测最优跟 ISL 绑定：4K→0.15，8K→0.10**，判据见 §11.5 结论 1 |

#### 约束 7 专述：SIGKILL 满载 decode = 泄漏 97 GB/卡

真实机制：

```bash
# pkill -9 之后，pod 里一个活进程都没有（全是 zombie，RSS=0）
$ kubectl exec sgl-0 -- ps -eo pid,stat,rss,comm --sort=-rss | head
  PID STAT   RSS COMMAND
    1 Ss    2432 sleep            ← PID 1 是 sleep infinity，不 reap 子进程
  357 Z        0 python3          <defunct>
  645 Z        0 sglang::schedul  <defunct>

# 但显存没还
$ kubectl exec sgl-0 -- nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader
0, 97486 MiB, 284208 MiB      ← 4 张卡全是 97 GB
1, 97486 MiB, 284208 MiB
```

284 GB 减掉 97 GB 只剩 187 GB，而 `mem-fraction-static 0.85` 要 241 GB → 下一次启动必 OOM。**而且报错是 `Not enough memory. Please try to increase --mem-fraction-static.`，会把人往「调大 mem-fraction」这个完全相反的方向带。**

推测原因：GB300 走 MNNVL/IMEX，导出到 fabric 的显存在进程被 SIGKILL 时不走正常回收路径；容器 PID 1 是 `sleep infinity` 不 reap，zombie 一直挂着。

**正确做法 —— 删 pod 让 StatefulSet 重建，不要 pkill**：

```bash
kubectl delete pod sgl-0 sgl-1 --wait=false
# 等 Running 且显存归零（实测 56 秒）
until [ "$(kubectl get pod sgl-0 sgl-1 --no-headers | grep -c Running)" = 2 ] && \
      [ "$(kubectl exec sgl-0 -- nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' ')" -lt 5000 ]; do sleep 10; done
```

删 pod 到显存归零实测 **56 秒**，比等泄漏自己消失快得多（等了 20 分钟没动静）。

> **这个泄漏会污染后续所有实验**。本轮 MTP `steps=2` 连续 OOM 三次，我一度判定是 MTP 本身内存不够、还降了 `cuda-graph-max-bs` 重试 —— 其实三次全跑在被第一次 pkill 留下的 97 GB 污染的环境里。清干净之后 **MTP2 一次就起来了**。
>
> **判据：每次 decode 启动前先看 `nvidia-smi` 显存是否归零，不归零的失败结论一律作废。**

#### 约束 8 专述：存活判据必须看显存，不能看 etcd

`§5.2` 说「etcd 注册数是权威判据」，那是针对**启动阶段**（能注册 = 真起来了）。**崩溃检测反过来完全不成立**：

```bash
# 14 台 prefill 全部已死
$ for i in $(seq 2 15); do kubectl exec sgl-$i -- nvidia-smi \
    --query-gpu=memory.used --format=csv,noheader,nounits | head -1; done
0 0 0 0 0 0 0 0 0 0 0 0 0 0          ← 显存全空 = 全崩

$ curl etcd .../v1/instances/dynamo/prefill/generate | wc -l
14                                     ← 仍然显示 14 台在线
```

lease TTL 没到期，死进程的 key 会在 etcd 里再挂几分钟。我据此误判「prefill 全程存活」，把错误结论写进了这份文档，直到去查显存才发现。

**正确的存活判据**（两个都要）：

```bash
# 1. 显存 > 200 GB（prefill dep4 满载）
MEM(){ kubectl exec sgl-$1 -- bash -c "nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits|head -1"; }
# 2. 日志里没有 include_parent=True 的自杀记录
kubectl exec sgl-$i -- bash -c "tr -d '\000' < /tmp/srv.log | grep -c 'kill_process_tree.*include_parent=True'"
```

> 注意 `/tmp/srv.log` 是**二进制**（含 NUL），`grep` 会直接返回 `binary file matches`。所有分析都要先 `tr -d '\000'`，或者用 `grep -a`。

> **参数语义务必读源码再动手**。本项目在 `swa-full-tokens-ratio` 上栽了两次：第一次照搬 dep32 的值、第二次把方向理解反了。判据是 decode 日志里的两个 usage：
>
> ```
> #full token: 6715392, full token usage: 0.78,   ← full 池
> #swa  token:  505088, swa  token usage: 0.96    ← SWA 池（先满 = 瓶颈在这）
> ```
>
> **哪个先到 ~0.96 就给哪个加预算**：SWA 先满 → 调高 ratio；full 先满 → 调低。

> **一个反复出现的模式**：官方 recipe 里的参数值是**跟拓扑绑定的整体**（#3 `mem-fraction 0.94`、#10 `swa-ratio 0.056` 都是 wide-EP 专用）。逐项摘出来搬到 dep8，每一项都会以不同方式失败。**要么整套换拓扑，要么一个都别动。**

> **共同点**：第 2、5、6 条都能通过所有常规健康检查（进程在、显存满、etcd 注册、frontend 200、单条推理正确），**只有真正加压才暴露**。这类故障没法靠 review 配置发现。

### 11.7 结论与剩余路径

**一句话**：gap 的大头是 **KV 池预算划错**（一个参数值 +54%），prefill 供给不足只是次要因素。

最终成绩（16 节点，14× prefill dep4 + dep8 decode，`steps=1 draft=2`）：

| 指标 | 实测（最好，配置 #7）| 官方 11,200 |
|---|---|---|
| ISL 4096 端到端 per-decode-GPU | **10,704** | 95.6% |
| ISL 4096 decode 自报峰值 | **12,070** | **107.8%** |
| ISL 8192 端到端（`swa-ratio 0.10`）| 9,354 | 83.5% |

> **推荐用配置 #3，不是 #7**。#7 只比 #3 多改了 `mem-fraction-static 0.85→0.88`，收益在噪声内（10,614 → 10,704，+0.8%），却把 OOM 风险抬高了。**照 §5 的默认值跑（`mem-frac 0.85` + `SWA_RATIO` 按 ISL 设）就能拿到 99% 的成绩。**

**结论链**：

1. **decode 引擎本身超过官方标称** —— 峰值 12,070 vs 11,200 = 107.8%。
2. **端到端 10,704 与峰值 12,070 的差距（11%）才是 prefill 供给不足的真实份额** —— 而不是早期估计的 20%。
3. **早期把 gap 全归给 prefill 是权重估错**（详见 §11.5 结论 3）。真正的排序是：KV 池划分 +54% ≫ 攒 batch +4.3% ≫ 其他。
4. **换镜像 / wide-EP / 照搬官方 decode 参数三条路全是死的**（§11.4）。

**天花板已经摸到了，而且不是 kernel 的 1024/rank**：把 KV 预算加大 13% 后 batch 冲到 999/rank（上限的 97%），吞吐却只涨 0.06%（§11.5 结论 1b）。**dep8 在 GB300 上的算力天花板 = 12,070 tok/s/GPU**，batch 超过 ~890 就不再是瓶颈。

**剩余可做的**（按收益/成本）：

| # | 动作 | 成本 | 说明 |
|---|---|---|---|
| 1 | ~~`mem-fraction-static` 0.85 → 0.88~~ | 已试 | **基本无效**：batch +13%，峰值 +0.06%、端到端 +0.8%。0.88 本身能跑（0.94 才 OOM，§11.6 #3），但把 OOM 风险抬高换不到东西 |
| 2 | **ISL 8192 上把 ratio 调到 0.10~0.12** | 一轮部署 | 8K 在 0.15 时 SWA 浪费 63%，0.10 时 swa 0.96 略过头，中间还有空间 |
| 3 | 加 prefill 节点（需 >16 节点） | 要机器 | 16 节点已是 14+2，加不动。官方 18 节点正为此 |
| 4 | 改 kernel 突破 1024/rank | 上游 PR | 把「单 block + 每线程一请求」改成 grid-stride + 动态 shared memory |
| 5 | 真实流量下开 prefix cache + wide-EP | 换负载 | wide-EP 摊薄权重腾出的 HBM 主要价值是装更多 prefix cache（KV hash），命中率上去 prefill 就少干活。**但 sa-bench 用 random 数据、且本文全程 `--disable-radix-cache`，这条在合成 benchmark 上赚不到** —— 这也解释了官方 frontier 曲线与我们实测的矛盾 |

> **给复现者的一句话**：如果只想抄一个能跑出好数的配置，用 §5 的 `decode-dep8.sh` 并把 `--swa-full-tokens-ratio` 按你的 ISL 设成 **4K→0.15 / 8K→0.10**，其余参数别动。


---

## 12. 历史实测数据（从旧文档 `sglang-v4-gb300-benchmark.md` 蒸馏，原文已删）

旧文档记录了 2026-07-20~22 的探索过程，绝大部分结论已被本文取代或推翻。下面是**仍然成立、且不可再生**的部分。

### 12.1 为什么 V4 能上万而 R1 上不了（架构代际）

我们实测 R1 短上下文（8K/1K）峰值 **1,359 tok/s/GPU**，官方 V4-Pro 同 workload **11,200**，差约 8×。根因是模型代际，不是调优不到位：

| 维度 | R1 | V4 |
|---|---|---|
| 注意力 | 全注意力 MLA，KV 留全历史 | **hybrid CSA + HCA**，@1M 时 KV 仅 V3.2 的 ~10%；等效滑动窗口 |
| decode `max-running-requests` | 2048（KV 大，塞不下更多）| **18432**（KV 薄，并发拉 9×）← 吞吐上万的直接原因 |
| MoE 量化 | W4A8 | **W4A4 MegaMoE**（激活也 4bit，矩阵乘快 ~2×）|
| KV 压缩 | 无 | **online compress**（C4/C128 压缩态池）|

**一句话**：V4 靠 CSA+HCA 把 KV 打薄 → decode 并发从 2K 拉到 18K → 吞吐堆上万。这是架构层解访存瓶颈，全注意力天生追不上。

### 12.2 PD 流水线配比公式（可复用）

```
需要的 prefill worker 数 = (decode 每秒完成请求数 × 输入长度) ÷ 单 prefill worker 吞吐
```

以官方 11,200 那个点（dep8 / 8K1K / 50 tok/s/user）推演：

1. 每张 decode 卡在 50 tok/s/user 服务 `11,200 ÷ 50 ≈ 224` 并发用户 → **decode 卡数 = 目标用户数 ÷ 224**（规模决策，不是性能决策）
2. dep8 = 8 × 224 ≈ **1,792 有效用户**（recipe 灌 conc 8192 是 offered load，多的在排队）
3. decode 完成率 = 8 × 11,200 ÷ 1024 ≈ **87.5 req/s** → prefill 须供 87.5 × 8192 ≈ **71.7 万 input tok/s**
4. 单 prefill worker（dep4）≈ 4 × 18,200 ≈ 7.28 万 → 需 **8–10 个**。官方 `high-conc-8p1d` 正是 8 个，对得上。

> **口径的隐含代价**：`output ÷ decode-GPU` **把 prefill 成本藏起来了** —— 堆再多 prefill 喂一个小 decode，per-decode-GPU 都好看。它是「解码效率」指标，**不是整机 TCO**。

### 12.3 官方 recipe 全表

| recipe | prefill | decode | MTP | 场景 |
|---|---|---|---|---|
| `mid-curve-1p1d-dep8` | 1 | dep8 | ✓ | 低并发交互 |
| `mid-curve-4p1d-dep8`（steps 3）| 4 | dep8 | ✓ | conc 1024 |
| `high-conc-8p1d-dep8`（steps 1）| 8 | dep8 | ✓ | conc 8192 ← **11,200 最可能在此** |
| `10p1d-dep32` c2500 | 10 | dep32 | ✗ | 大规模吞吐 |
| `15p1d-dep12` c12000 | 15 | dep12 | ✗ | 超高并发 |

> 官方博客"How to Reproduce"贴的是 `10p1d-dep32-c2500`（**no-MTP**），但 11,200 出自 **MTP 曲线**——博客拿它当流程示范，误导性很强。

### 12.4 历史扫描数据（本文未重测，仍可参考）

**多 frontend**（单 frontend 是 Python 进程，高并发 CPU-bound）：

| frontend × conc | 聚合 output | output/decode-GPU |
|---|---|---|
| 1 × 2500 | 40,481 | 5,060 |
| 4 × 625 | 48,934 | 6,117 |
| 8 × 625 | 54,300 | **6,788（+34%）** |

**prefill 数量**：8 → 6,788；**14 → 8,809（+30%）**；16 → 8,993（+2%，收敛）。

**单 prefill worker 并发扫描**：峰值 **15,196 input tok/s/卡 = 官方 83%**。早期"prefill 慢 3.7×"是 conc4 极低并发下的测量假象，**已推翻**。

**满载 GPU 利用率**（16p 满配，nvidia-smi 采样）：

| 角色 | util | HBM | 功耗（TDP ~1400W）|
|---|---|---|---|
| prefill | 99–100% | 271–277 GiB | 1137 W（81%）|
| decode | 75–97% | 268–275 GiB | 960–1038 W（71%）|

### 12.5 已验证无效 / 有害的尝试（别重做）

| 尝试 | 结果 |
|---|---|
| DeepGEMM full autotune（关 `FAST_WARMUP`）| 热态 9,018 vs 8,993，**噪声内**。冷→热那 +12% 是一次性 JIT，不是 autotune 的功劳。代价是巨大启动开销 |
| EPLB | 与 megamoe 不兼容（三种失败模式）。但 cookbook 称 Waterfill 变体支持，见 §11.7 |
| **`slow_down` / 预留 decode token「攒批再放」hack** | `SGLANG_HACK_PD_DECODE_NUM_RESERVED_DECODE_TOKENS=1026` → 吞吐**降到 2,984**，过度预分配，**有害**。⚠️ 注意与 §11.4 Exp E 区分：官方的 `--disaggregation-decode-polling-interval` 是**有效**的（+4.3%），这个 env hack 是**有害**的，两者别混 |
| 闭环 `bench_serving` + `--max-concurrency` | 闭环已限死在途请求数，再开 `router-queue-threshold` 只增延迟不增吞吐（**−13%**）。对标官方必须**开环** `--request-rate inf` |

### 12.6 其他硬约束（旧文档记录，本文未复现）

- **合法 EP 必须整除 256**（V4 有 256 个专家）。`dep40` 会直接 `assert num_physical_experts % ep_size == 0` 崩溃，**必须配 EPLB 加冗余专家凑整**（如 256+24=280，280/40=7）。所以对 dep40 而言 EPLB 不是优化、是启动前提。合法值：8 / 16 / 32 / 64…
- **DeepEP dispatch buffer**：`max-running-requests × MTP_draft_tokens ≤ SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK`，否则稳态负载炸 `deep_ep.cpp:1105`。**调并发时三个值要一起动。**
- `SGLANG_DSV4_COMPRESS_STATE_DTYPE=bf16`（默认 fp32）—— 压缩态池省显存，可换更多 decode slot。本文未测。
- **镜像**：`v0.5.15.post1`（R1 时代）不支持 V4，必须 nightly；换镜像后要重验 `sm_103a`。
