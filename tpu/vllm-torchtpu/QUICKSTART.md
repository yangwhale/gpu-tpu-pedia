# 从零到第一个 benchmark 数字

> **这份文档的目标**：让你在 TPU v7 上跑出第一个 vllm-torchtpu 的 benchmark 结果，
> 且**不需要排查任何问题**。0.6B 约 15 分钟，35B 约 23 分钟（实测，逐阶段耗时见下表）。
>
> **本文档已按自己写的步骤完整复测过一遍**（2026-08-13），复测中又修正了 3 处错误：
> taint 说明、内存盘容量、耗时估计。下面的数字都是实测值，不是估算。
>
> 下面每一个 `apt-get` 包、每一行 pod spec、每一个环境变量，都对应一个我们实测踩过的坑
> （2026-08-12 一整夜，15 轮，6 个坑，每个坑废掉一个 60 分钟窗口）。
> **照抄即可，不要精简。**想知道每条为什么存在，见
> [`Qwen3.5-397B-A17B-FP8/RUNLOG-20260812.md`](./Qwen3.5-397B-A17B-FP8/RUNLOG-20260812.md)。

---

## 0. 你需要什么

| | |
|---|---|
| 硬件 | TPU v7（`tpu7x`）**4 芯片 = 8 device**，拓扑 `2x2x1` |
| 权限 | 能读 `us-docker.pkg.dev/ml-oss-artifacts-transient`（镜像 + pip 源） |
| 时间 | **按模型差别很大**：0.6B 约 15 分钟，35B 约 23 分钟（逐阶段见下表） |

**实测各阶段耗时**（v7x 4 芯片，内存盘，`hf_transfer` 开启）：

| 阶段 | 0.6B / TP=1 | 35B / TP=4 | 397B / TP=8 冷 | **397B 热（缓存命中）** |
|---|---|---|---|---|
| 起 pod | 38 s | 38 s | 38 s | 38 s |
| 装环境 | 4.4 min | 4.4 min | ~5 min | **3 min 08 s** |
| 权重 | ~1 min（1.2 GB） | ~2 min（35 GB） | 下载 ~8 min（406 GB） | **零下载，加载 4 min** |
| 编译到 server ready | ~7 min | ~15 min | **19 min** | **0（新 graph 数 = 0）** |
| **合计到可服务** | 约 13 min | 约 22 min | **约 29 min** | **约 18 min** |
| 跑 benchmark | 2.3 min | 1 min | — | **单 cell 26 min 42 s** |
| **窗口够不够** | ✅ | ✅ | ❌ 连一个 cell 都跑不完 | **✅ 45 min，塞得进 55 min** |

> ⚠️ **编译是最大的一块准备成本**，但增长比想象中温和：397B 编译 19 分钟，
> 只比 35B 多 4 分钟，远非按参数量线性外推。
>
> **397B 冷启动跑不完，热启动可以。**冷启动 29 分钟准备后只剩 24 分钟，
> 而单个 cell 要 26 分 42 秒——差一点点。做完缓存持久化后准备降到 18 分钟，
> 45 分钟跑完一个 cell，塞得进 55 分钟窗口。**做法见 [3.5 节](#35-跑大模型必读把权重和编译缓存放持久盘)。**
> 注意 `tp8-dp1` 这个 config 共 6 个 cell，跑全要 6 个窗口或申请独占。

> **注意 chip 与 device 的口径**：v7x 是 1 chip = 2 device。所以「4 芯片」= 8 device，
> 配置里 `TP × DP = 8` 指的是 **device 数**，对应的就是这一台机器。
> vLLM 日志会打 `num_chips=8`，那其实是 device 数，别按字面理解成 8 颗芯片。

---

## 1. 起 Pod（4 芯片）

直接用 [`manifests/v7-4chip-dev.yaml`](./Qwen3.5-397B-A17B-FP8/manifests/v7-4chip-dev.yaml)。
里面有四处**不能删**的东西：

```yaml
volumes:
- name: dshm
  emptyDir: {medium: Memory, sizeLimit: 32Gi}     # ① vLLM 多进程 RPC 要 ≥160 MiB，K8s 默认只给 64 MiB
- name: workdir
  emptyDir: {medium: Memory, sizeLimit: 320Gi}    # ② 权重/HF cache/编译缓存全放内存，躲开节点磁盘配额
                                                  #    ⚠️ 320Gi 只够到 ~35B。按模型调，见下表
...
    volumeMounts:
    - {name: dshm,    mountPath: /dev/shm}
    - {name: workdir, mountPath: /work}
    resources:
      # ③ limits 必须写在同一行里。分成两行会让后者覆盖前者，
      #    google.com/tpu 丢掉 limit → Job 直接被 API server 拒绝创建
      limits:   {google.com/tpu: 4, ephemeral-storage: 32Gi}
      requests: {google.com/tpu: 4, cpu: "8", memory: 48Gi, ephemeral-storage: 16Gi}
    tolerations:                                   # ④ TPU 节点的 taint，见下表
    - {key: google.com/tpu, operator: Equal, value: present, effect: NoSchedule}
```

**④ 的 taint 要按集群类型给，给多了无害、给少了永远 Pending：**

| 集群类型 | 需要的 toleration | 说明 |
|---|---|---|
| 共享 NAP 集群（如 `bodaborg-tpu7x-nap`） | 只需 `google.com/tpu=present` | 实测节点只有这一个 taint |
| 自有项目的 **DWS / queued provisioning** node pool | 还要加 `cloud.google.com/gke-queued="true"` | 这类 pool 有第二个 taint，**且光加 toleration 不够**，必须走 `ProvisioningRequest` 才会真正分配节点 |

查一下省得猜：

```bash
kubectl get nodes -l cloud.google.com/gke-tpu-accelerator=tpu7x \
  -o jsonpath='{.items[0].spec.taints}'
```

**`workdir` 的 `sizeLimit` 必须按模型调**（节点有 944 GB RAM，放心给）：

| 模型 | 权重约 | 建议 `sizeLimit` |
|---|---|---|
| Qwen3-0.6B | 1.2 GB | 320Gi（默认即可） |
| Qwen3.5-35B-A3B-FP8 | 35 GB | 320Gi（默认即可） |
| Qwen3-Coder-480B-FP8 | ~480 GB | **640Gi** |
| **Qwen3.5-397B-A17B-FP8** | **~400 GB** | **560Gi** ← 默认的 320Gi **装不下**，会在下载中途爆 |

```bash
kubectl apply -f manifests/v7-4chip-dev.yaml
kubectl get pods -n <你的 namespace> -w
# 期望：约 20-60 秒变 Running。共享集群若需 NAP 扩容可能到 5 分钟。
```

**验证拿到的确实是 4 芯片：**

```bash
kubectl exec $POD -- bash -c 'ls /dev/vfio | tr "\n" " "; echo; env | grep TPU_'
# 期望：/dev/vfio 里有 0..7（8 个 device = 4 chip）
#       TPU_TOPOLOGY=2x2x1   TPU_CHIPS_PER_HOST_BOUNDS=2,2,1
```

> ⚠️ Pod spec 改完**一定要 grep 一遍 `resources:` 段确认没有重复键**。
> `kubectl apply --dry-run=server` **检查不出来**——它只校验 JobSet 本身，
> 不校验它将来生成的 Job。我们在这上面白跑了 9 个任务。

---

## 2. 装环境（一条命令，约 5 分钟）

```bash
# 本机：把源码 + vLLM tarball 一起打包（不要让 pod 自己去 GitHub 下，实测两次失败）
git clone https://github.com/vllm-project/vllm-torchtpu.git
curl -L -o vllm.tgz https://github.com/vllm-project/vllm/archive/refs/tags/v0.26.1rc0.tar.gz
tar czf pkg.tgz vllm-torchtpu vllm.tgz

kubectl exec -i $POD -- bash -c 'mkdir -p /work && cd /work && tar xzf -' < pkg.tgz
kubectl cp Qwen3.5-397B-A17B-FP8/scripts/setup-in-pod.sh $NS/$POD:/work/setup.sh

kubectl exec $POD -- bash -c \
  "AR_TOKEN='$(gcloud auth print-access-token)' bash /work/setup.sh /work/vllm-torchtpu"
```

**期望最后三行：**

```
  platform     : TpuPlatform
  tpu devices  : 8
SETUP_OK
```

脚本内部做了这些，每条都是必须的：

| 步骤 | 为什么 |
|---|---|
| `apt install git cmake build-essential ninja-build **curl**` | base 镜像这五个全没有。**curl 最致命**：benchmark runner 用它探 `/health`，没有 curl 就永远认为 server 没起来，而 server 其实是好的 |
| 先装死版本矩阵（jax 0.10.2 / libtpu 0.0.44.1 / torch 2.13.0 / torch-tpu 08-04 nightly） | 镜像自带的旧一档，`jax 0.9.2` 没有 `pallas.tpu.BufferType`，一跑就崩 |
| 用 `pip` **不用 `uv`** 装 | uv 不读 `PIP_INDEX_URL`，会报「registry 里找不到 torch-tpu」而实际上包一直在 |
| vLLM 装完再 `--no-deps` 装 vllm_torchtpu | 不加 `--no-deps` 时 pip 会按 `vllm @ git+...` 重建 vLLM，且构建隔离环境里 `VLLM_TARGET_DEVICE` 落回 `cpu` 编译失败，整步回滚 |
| `SETUPTOOLS_SCM_PRETEND_VERSION_FOR_VLLM_TORCHTPU=0.1.0` | tarball 传源码没有 `.git`，setuptools_scm 取不到版本 |
| `sed -i '/tpu-inference/d' vllm/requirements/tpu.txt` | 否则上游 tpu-inference plugin 会跟 vllm-torchtpu 装到一起 |

---

## 3. 跑 benchmark

```bash
kubectl exec $POD -- bash -c '
  export HF_HOME=/work/hf HF_HUB_ENABLE_HF_TRANSFER=1 PYTHONUNBUFFERED=1
  export SERVER_READY_WAIT_MIN=25          # ← 必须小于 pod 寿命，理由见下
  cd /work/vllm-torchtpu
  bash ./scripts/vllm/benchmarking/run_benchmarks.sh --config qwen3-0.6b-tp1
'
```

> ⚠️ **`SERVER_READY_WAIT_MIN` 默认是 90 分钟**，而共享集群的 pod 通常只活 60 分钟。
> runner 唯一会打印 `server.log` 的代码在「等待超时」分支里——默认值下**那段永远执行不到**，
> 于是 server 出任何问题你都只会看到一串 `Waiting for server...`。
> **把它设成小于 pod 寿命**，这是本次调试中最值钱的一个环境变量。
> （397B 的 config 里写的是 `120`，跑之前记得覆盖。）

**期望时间线**（0.6B，首次冷启动）：

```
[cmd] vllm serve ...                              ← 立刻
Waiting for server... (60s ~ 420s elapsed)        ← 正常，在编译
Starting main benchmark run...                    ← 约 7 分钟后
============ Serving Benchmark Result ============ ← 再 2-3 分钟
```

编译期间想看它在干嘛（**默认看不到，要手动 tail**）：

```bash
kubectl exec $POD -- bash -c \
  'tail -f $(ls -td /work/vllm-torchtpu/benchmark_runs/*/ | head -1)/server.log'
# 会看到：backbone 按 num_seqs 桶编译（7 个桶约 41 秒），
#         然后每个 token range 一个 graph（各约 12 秒）
```

---

## 3.5 跑大模型必读：把权重和编译缓存放持久盘

**对 397B 这一档，缓存持久化不是优化项，是前置条件。**没有它，一个 60 分钟窗口
连一个 cell 都跑不完；有了它，一个窗口稳定产出一个 cell。

实测对比（同一个 397B、同一份 pod spec）：

| 阶段 | 冷启动 | 热启动（缓存命中） |
|---|---|---|
| 装环境 | 5 min | 3 min 08 s |
| 权重 | **下载 461 s（406 GB）** | **零下载**，从盘加载 94 shards **4 min** |
| **编译** | **19 min** | 大部分从缓存反序列化，**但不是 0**，见下 |
| **到 server ready** | **29 min** | **16 min 16 s** |

热启动那 16 分 16 秒的精确归因（方法：扫 `server.log` 相邻行的最大时间空隙，
不靠猜里程碑）：

| 阶段 | 耗时 | 能不能压 |
|---|---|---|
| 装环境 | 3 min 12 s | 可与权重下载并行 |
| 进程启动 + 插件加载 | 46 s | 固定开销，压不动 |
| 权重加载 | **4 min 00 s** | **可压到 69 s**，见 §3.6 |
| 编译 / 缓存加载 | 7 min 32 s | 含 16 次无效重编，见下 |

### 怎么做

**① 建缓存盘**（[`manifests/cache-pvc.yaml`](./Qwen3.5-397B-A17B-FP8/manifests/cache-pvc.yaml)）：

```bash
kubectl apply -f Qwen3.5-397B-A17B-FP8/manifests/cache-pvc.yaml
# hyperdisk-balanced 1000Gi RWO。397B 实测占 423 GB（权重 379 + 编译缓存 44）
```

**② 用带缓存挂载的 pod**（[`manifests/v7-4chip-cached.yaml`](./Qwen3.5-397B-A17B-FP8/manifests/v7-4chip-cached.yaml)），它比基础版多挂三处：

```yaml
- {name: cachedisk, mountPath: /cache}                        # 持久盘
- {name: workdir, mountPath: /root/.cache, subPath: rootcache} # 编译缓存默认位置，导去内存盘
- {name: workdir, mountPath: /tmp,         subPath: tmproot}   # TPU 日志同理
```

**③ 两个环境变量把东西引到持久盘**：

```bash
export HF_HOME=/cache/hf            # 权重
export VLLM_CACHE_ROOT=/cache/vllm  # 编译缓存
```

> `VLLM_CACHE_ROOT` 是 vllm-torchtpu 真正读的变量
> （`runner/tpu_runner.py:4666`、`compilation/tpu_compiler.py:259`）。

**④ 第一轮预热**：照常跑一次，权重和编译产物会自动落到 `/cache`。跑完检查：

```bash
kubectl exec $POD -- bash -c 'du -sh /cache/hf /cache/vllm; find /cache/vllm -name "artifact_compile_range*" | wc -l'
# 397B 期望：/cache/hf 379G   /cache/vllm 44G   88 个 artifact
```

**⑤ 之后每轮**加上 `export HF_HUB_OFFLINE=1`（权重已就位，杜绝任何偷偷下载）。
验证缓存真命中，看这两个数：

```bash
D=$(ls -td /work/vllm-torchtpu/benchmark_runs/*/|head -1)
grep -c "Compiling a graph for compile range" $D/server.log   # 期望 0
grep -c "pickle.load" $D/server.log                            # 期望 >0（在反序列化已编译产物）

# ⚠️ 上面两条不够。真正的缓存失效走的是另一个字符串，只查上面两条会得到虚假的安心：
grep -c "Loading compiled executable" $D/server.log   # 从缓存读回的可执行体数
grep -ci "Compiling model again"      $D/server.log   # 读不回、退化成重编的数量 —— 这条才是关键
```

> ⚠️ **别只看 "Waiting for server" 的时长就断定缓存没生效**——热启动仍要花 4 分钟
> 加载 400 GB 权重、再反序列化编译产物，总共约 16 分钟。判据是上面那几个计数。

> **已知上游缺陷（2026-08-14 实测，397B / tp8-dp1）**：
> 缓存**不会全命中**。88 个可执行体正常读回，**16 个失败后重新编译**，8 个 worker 每个中招 2 次：
>
> ```
> WARNING [decorators.py:321] Compiling model again due to a load failure from
>   .../torch_aot_compile/<hash>/rank_N_0/model,
>   reason: a bytes-like object is required, not 'BundledAOTAutogradResult'
> ```
>
> 这是序列化路径的类型错配：写进去的是 `BundledAOTAutogradResult` 对象，读出来按 `bytes` 处理。
> 后果是这部分编译时间**每次启动都白付**，缓存盘做得再持久也救不回来。
> 排查时如果只查 `"Compiling a graph for compile range"`（它确实是 0），
> 会误判成「缓存全命中」而漏掉这 16 次重编。

### 为什么不用 GCS + gcsfuse

试过，**没走通**：pod 卡在 `gke-gcsfuse/bucket-scan-pending`，因为 namespace 的
`default` KSA 没绑 Workload Identity，gcsfuse 认证不了桶；而共享集群的桶不便改 IAM。

不过就算能用，对**单 pod 串行**跑 benchmark 来说 hyperdisk 也更合适——编译产物是大量
小文件，真磁盘远好过 gcsfuse。**gcsfuse 的价值在并行**：RWX 能让多个 pod 共享同一份
400 GB 权重，不必各下一遍。要并行铺开时再解决 WI 授权。

### ⚠️ 结果文件也要落持久盘

benchmark 结果写在 `/work/.../benchmark_runs/<ts>/isl*.json`，而 `/work` 是内存盘。
pod 一到期变 `Completed`，`kubectl exec` 就报
`cannot exec into a container in a completed pod`，**文件拿不回来**。

```bash
# 跑完立刻拷走
cp -r /work/vllm-torchtpu/benchmark_runs /cache/results/
```

我们在这上面栽过：只能靠 `kubectl logs` 从 stdout 的文本摘要里重建指标，JSON 原文丢了。

---

## 3.6 如果你的 pod 不受 60 分钟限制，上面那套请全部推翻

§3.5 那一整套（持久盘、单 cell 切分、跑完立刻拷结果）**是为了绕开共享集群 60 分钟
pod 寿命而生的**，不是什么最佳实践。在专属集群 / DWS 节点上 pod 可以长期存活，
硬搬这套反而每个 cell 白付一次十几分钟的启动。

**换成这样：**

### ① 权重放 tmpfs，不放磁盘

`tpu7x-standard-4t` 节点有 **963 GB 内存**，378 GiB 权重放内存盘绰绰有余。

| 权重落点 | 加载耗时 |
|---|---|
| EXT4 hyperdisk | 240 s |
| **tmpfs** | **69 s** |

**快 3.5 倍，每次启动省 171 秒。**

```yaml
volumes:
- name: work
  emptyDir: {medium: Memory, sizeLimit: 640Gi}
```

> ⚠️ 同时必须把 `/tmp` 和 `/root/.cache` 也挂到 tmpfs。
> 节点 **ephemeral-storage 只有 43.8 GiB**，pip 编译 vLLM 和 TPU 日志往容器盘写会撞
> **节点级**驱逐 —— 而容器里 `df` 看着一片宽裕，因为那是整节点共享配额，容器 `df` 看不见。

### ② 权重从同区 GCS 拉，别走 HF

| 来源 | 速率 |
|---|---|
| HuggingFace | 967 MB/s |
| **同 region GCS** | **231 MiB/s × N 节点，聚合 990 MiB/s** |

镜像里没有 gcloud SDK，用 `google-cloud-storage` 自己控并发即可，
见 [`scripts/fetch-weights.py`](./Qwen3.5-397B-A17B-FP8/scripts/fetch-weights.py)。

> **进度条要统计已落盘字节，不能统计已完成文件。**
> 94 个 4 GB 分片 / 48 并发时，前 20 分钟一个文件都不会完成，
> 按文件计数的进度条会一直显示 `0 MiB/s` 像卡死，ETA 也算得离谱。
> `du -sh` 才是真值 —— 那时盘上其实已经有 359 GiB。

### ③ 装环境和拉权重并行跑

一个打 pypi、一个打 GCS，不抢同一个瓶颈。实测环境安装只要 **2 min 50 s**，
完全被 28 分钟的权重下载掩盖。串行的话白多花近 3 分钟。

### ④ 一次跑完整个 config，不要切 cell

一次 server 启动的固定成本是十几分钟。同一节点上的多个 cell **必须共用一次启动**。

### ⑤ 有多机 slice 时，切开当多个单机用

2x2x4 的多机 slice 上，把 TPU env 降回单机视图，可以干净地只用本机 8 个 device。
**这不是虚拟化也不是资源划分，是「拒绝组队」** —— 硬件本来就是 4 台各插各的 4 颗芯片，
「16 卡是一个整体」是 libtpu 按环境变量去会合才形成的，不去会合就各干各的。

```bash
export TPU_WORKER_ID=0
export TPU_PROCESS_ADDRESSES=localhost:8471
export TPU_WORKER_HOSTNAMES=localhost
export TPU_HOST_BOUNDS=1,1,1            # ← 核心：GKE 注的是 1,1,4
export TPU_CHIPS_PER_HOST_BOUNDS=2,2,1  # ← 不用改，每台本来就是 4 颗
export TPU_ACCELERATOR_TYPE=tpu7x-8     # ← 8 是 device 数，不是 chip 数
unset TPU_MULTIHOST_BACKEND             # ← 设过 ray 不清掉会走多机分支
```

三个易错点：
**①** `TPU_PROCESS_ADDRESSES`（带端口）和 `TPU_WORKER_HOSTNAMES`（纯主机名）是一对，
只改一个会对不上。
**②** `TPU_CHIPS_PER_HOST_BOUNDS` 保持 `2,2,1` 不动，真正变的只有 `TPU_HOST_BOUNDS`。
**③** v7x 上 **1 chip = 2 device**，所以多机是 `tpu7x-32`（16×2），单机是 `tpu7x-8`（4×2）。
写错会报拓扑不匹配，但错误信息不会提示你是单位搞错了。

验证：`python3 -c "import jax; print(len(jax.devices()))"`
返回 **8** 就对了；返回 32 是变量没生效；**卡住不返回**是它还在等另外 3 台会合。

**实测互不干扰**（同一 cell，4 台同时跑 vs 单机独占）：

| | median TPOT | output tok/s |
|---|---|---|
| 单机独占 | 39.62 ms | 5833.5 |
| 4 台同时（同一 slice） | 39.54 ms | 5877.9 |
| 差异 | −0.22% | +0.76% |

噪声级别。**一个 16 chip slice 可以当 4 个独立 4 chip 环境用，吞吐不打折。**
需要跑实验矩阵时，申请一个大 slice 切开用，比申请 4 个小 slice 排 4 次队快得多，
而且拿到的是同一批硬件，横向对比没有机器差异。

实测效果：18 个 cell 铺在 4 个节点上，约 90 分钟墙钟跑完。
按 §3.5 那套要 18 个 60 分钟窗口。

---

## 4. 参考结果

`Qwen3-0.6B / TP=1 / ISL 1024 / OSL 1024 / 并发 8`，v7x 4 芯片实测：

| 指标 | 值 |
|---|---|
| Successful requests | 320 / 320 |
| Output token throughput | 2,422 tok/s |
| Median TPOT | 3.19 ms |
| Median TTFT | 20.57 ms |
| Duration | 135.5 s |

拿到接近的数字，说明你的环境是对的。

> 这个数**不是**性能基准（0.6B 在上游没有 baseline，且 runner 未强制 `temperature=0`）。
> 它只是「环境正确」的对照。

### 可对标的那份：Qwen3.5-35B-A3B-FP8 / TP=4 / EP / ISL 1024 / OSL 1024 / 并发 64

| 指标 | 上游 baseline | 我们实测 | 偏差 |
|---|---|---|---|
| completed | 320 | 320 | ±0 |
| median TPOT | 12.2 ms | **9.84 ms** | −19.6% |
| output token throughput | 4,673.7 tok/s | **5,534.5 tok/s** | +18.4% |
| total token throughput | 9,329.6 tok/s | **11,047.9 tok/s** | +18.4% |

口径已逐项核对一致（含 `benchmark_temperature=0`）。环境版本：
`torch 2.13.0 / jax 0.10.2 / libtpu 0.0.44.1 / torch-tpu 0.1.1.dev20260804130134 / vllm 0.26.1rc0+tpu`。

### 满配那份：Qwen3.5-397B-A17B-FP8 / TP=8 DP=1 EP / ISL 1024 / OSL 8192 / 并发 64

| 指标 | 上游 baseline | 我们实测 | 偏差 |
|---|---|---|---|
| completed | 640 | **640**（failed 0） | ±0 |
| median TPOT | 22.1 ms | **21.09 ms** | −4.6% |
| output token throughput | 2,831.3 tok/s | **2,955.5 tok/s** | +4.4% |
| total token throughput | 3,184.2 tok/s | **3,323.8 tok/s** | +4.4% |

实测补充（基线无对应字段）：median TTFT 106.6 ms · P99 TPOT 21.59 ms ·
Median ITL 20.50 ms · duration 1,602 s。
交叉验证：`(590,085 input + 4,734,705 output) / 1602.0 s = 3,323.8 tok/s` ✅

> ⚠️ **这些 +4% / +18% 都是线索不是结论**：单次测量、未验证重复性；也不知道上游各自的
> baseline 记录于哪个版本组合。注意 35B 快 18.4% 而 397B 只快 4.4%，差异明显，
> 这本身就提示存在未受控变量。若要对外引用，请先各跑 3 次取分布并与上游核对版本。
> 用它做「环境是否正确」的判据是够的——**你的数应该落在这个量级，不该差一个数量级**。

---

## 5. 接下来跑什么

上游 `scripts/vllm/benchmarking/baselines/perf/` 里有可对标的基线：

| config | 基线（ISL/OSL 1024, c64） | 备注 |
|---|---|---|
| `qwen3.5-35b-fp8-tp4-ep` | TPOT 12.2 ms / total 9,330 tok/s / completed 320 | ✅ 真实数据，推荐第一个对标 |
| `qwen3-coder-480b-fp8-tp8-ep` | 有 | 权重大，注意时间 |
| `qwen3.5-397b-fp8-{tp1dp8,tp2dp4,tp8dp1}` | 6 个负载点各有 | 权重约 400 GB |
| `qwen3.5-35b-fp8-dp8-ep` / `dp4-tp2-ep` | ⚠️ **空壳**（`completed=0`，全 0） | **不要拿来对标** |

**跑大模型前先解决两件事**，否则 60 分钟窗口不够：

1. **内存盘要够 + 权重预置**。397B 实测 **406.2 GB / 95 个 shard**，pod 本地盘（74 GB）
   放不下；内存盘装得下但**默认 320Gi 不够，要 620Gi**（实测占 379 GB，余量充足）。
   下载本身很快：**420 秒下完 406 GB ≈ 967 MB/s**（`hf_transfer` + 32 workers + 内存盘），
   所以下载不是瓶颈，**编译才是**。
   预下载务必注意路径：

   ```python
   # 对：不传 cache_dir，跟着 HF_HOME 走
   snapshot_download("Qwen/Qwen3.5-397B-A17B-FP8", max_workers=32)
   # 错：cache_dir=$HF_HOME 会落到 $HF_HOME/ 而 vLLM 找 $HF_HOME/hub/，导致再下一遍把盘撑爆
   ```
   ```bash
   export HF_HUB_OFFLINE=1   # 权重就位后起 server，杜绝任何偷偷下载
   ```
   并行跑多配置时把权重放 GCS 用 gcsfuse 只读共享，省下每 pod 一次的 400 GB。
2. **编译缓存持久化**（对 397B 是**前置条件**，不是优化项）。缓存在
   `/root/.cache/vllm/torch_compile_cache/`。实测编译耗时随规模快速上升：
   0.6B 每图约 12 秒、约 7 分钟就绪；35B 的 `compile range (8192,8192)` 单图就要 50 秒、
   四个 TP worker 各编一份、**约 15 分钟就绪**。397B 参数量是 35B 的 11 倍，
   **60 分钟窗口大概率不够**，必须先把缓存落到持久卷或 GCS 复用。

---

## 6. 出问题时，先看这张表

**这些症状我们全部遇到过，直接对号入座，不要花时间排查。**

| 症状 | 真因 | 处理 |
|---|---|---|
| 一直 `Waiting for server...` 到 pod 被杀 | **镜像里没有 curl**，runner 探活永远失败（server 其实是好的） | `apt install curl` |
| 调高并发后 TTFT 暴涨一个数量级，**吞吐持平甚至下降** | **`DATA_PARALLELISM=1` 撞调度墙** —— 只有一个 engine、一个连续批处理循环，请求全挤在一个队列里。与 `MAX_NUM_BATCHED_TOKENS` 无关（tp1-dp8 的预算只有 1024，比 tp8-dp1 的 8192 更小，反而是高并发下最快的） | 高并发要吞吐就换 DP。实测 397B / ISL 8192 / 并发 512：tp1-dp8 出 3316 tok/s，tp8-dp1 只有 2146 且 median TTFT 达 **108 s** |
| PVC 挂不上，报 `exit status 19 / No such device` | **TPU 节点的 COS 镜像没有 lustre 客户端模块**（只有 lnet/libcfs）。CSI pod 显示 `2/2 Running` 是假象 | 进节点 `modprobe lustre` 确认；确认后改用 tmpfs 或 GCS，别再查网络和 PVC 绑定 |
| 容器里 `df` 显示磁盘很空，pod 却被驱逐 | ephemeral-storage 是**整节点共享配额**，容器 `df` 看不见。节点只有 43.8 GiB | 把 `/work`、`/tmp`、`/root/.cache` 全挂 tmpfs |
| JobSet `--dry-run=server` 全绿，实际提交后没有 pod | dry-run **只校验 JobSet CRD，不校验它生成的 Job**。典型元凶是 YAML 里两个 `limits` 键，后者覆盖前者导致 `google.com/tpu` 丢掉限额 | 把 pod template 抽成独立 Job 再 dry-run 一次；真因只在 `jobset-system` 控制器日志里 |
| `RuntimeError: Insufficient space in /dev/shm: 160 MiB required, 64 MiB free` | K8s 默认 `/dev/shm` 只有 64 MiB | 挂 `emptyDir: {medium: Memory}` 到 `/dev/shm` |
| Pod 被 evict，`ephemeral-storage` 不足 | 容器里 `df` 看到的不是你的配额，节点是共享的 | `/work` 挂内存盘 + 声明 `ephemeral-storage` |
| JobSet 创建了但一个 Job 都没有 | pod spec 里有**两个 `limits` 键**，`google.com/tpu` 丢了 limit | 合并成一行；去 `jobset-system` 看 controller 日志 |
| `AttributeError: pallas.tpu has no attribute 'BufferType'` | jax 版本旧了 | 装 `jax==0.10.2 jaxlib==0.10.2` |
| `torch-tpu was not found in the package registry` | 你用的是 `uv`，它不读 `PIP_INDEX_URL` | 换 `python3 -m pip` |
| `'vllm bench serve' not available` | setup 阶段其实失败了（常见于 GitHub 下载 503），脚本却继续往下跑 | 检查 setup 日志；vLLM 源码随包传入 |
| `Engine core initialization failed ... Failed core proc(s): {}` | 这是**转述**，不是真因 | 去 `benchmark_runs/<ts>/server.log` 找真正的 traceback |
| pod `Completed`、`exitCode 0`、但日志是空的 | 日志写在容器内文件里，pod 一死就没了 | 输出 tee 到 `/proc/1/fd/1` 走容器 stdout |
| `LookupError: setuptools-scm was unable to detect version` | tarball 传源码丢了 `.git` | `export SETUPTOOLS_SCM_PRETEND_VERSION_FOR_VLLM_TORCHTPU=0.1.0` |
| `AttributeError: module 'vllm' has no attribute '__version__'` | 你在 `/work/vllm` 源码目录里 import，串到本地目录了 | `cd /tmp` 再 import |
| 明明预下载过权重，server 启动时又下一遍、然后盘满 | `snapshot_download(cache_dir=$HF_HOME)` 落在 `$HF_HOME/`，而 HF 默认缓存是 **`$HF_HOME/hub`**，差一层目录 | 预下载别传 `cache_dir`（跟着 `HF_HOME` 走），并在起 server 前 `export HF_HUB_OFFLINE=1` |

### 排障通用顺序

vLLM 起不来时，**不要在 runner 的输出里找答案**，它只是转述。按这个顺序往上走：

```
pod 日志  →  benchmark_runs/<ts>/server.log  →  kubectl describe pod
         →  kubectl get jobset -o json（conditions / replicatedJobsStatus）
         →  kubectl logs -n jobset-system -l control-plane=controller-manager
```

我们有一次 9 个任务全失败，真因只出现在最后那一层。
