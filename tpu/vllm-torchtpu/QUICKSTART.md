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

| 阶段 | Qwen3-0.6B / TP=1 | Qwen3.5-35B / TP=4 |
|---|---|---|
| 起 pod | 38 s | 38 s |
| 装环境 | 4 分 24 秒 | 4 分 24 秒 |
| 拉权重 | ~1 min（1.2 GB） | **~2 min（35 GB）** |
| 编译到 server ready | **~7 min** | **~15 min** |
| 跑 benchmark | 2.3 min | 1 min |
| **合计** | **约 15 分钟** | **约 23 分钟** |

> ⚠️ **编译是主要成本，且随模型规模快速上升。**0.6B 每个图约 12 秒，
> 35B 的 `compile range (8192, 8192)` **单个就要 50 秒**，且 TP=4 时四个 worker 各编一份。
> **397B 大概率塞不进 60 分钟窗口**——除非先做编译缓存持久化（见第 5 节）。

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

> ⚠️ **这个 +18% 是线索不是结论**：单次测量、未验证重复性；也不知道上游 baseline 是用哪个
> 版本组合跑的。若要引用，请先连跑 3 次取分布，并跟上游确认 baseline 的版本。
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
