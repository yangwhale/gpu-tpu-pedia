# 从零到第一个 benchmark 数字

> **这份文档的目标**：让你在 **35 分钟内**在 TPU v7 上跑出第一个 vllm-torchtpu 的
> benchmark 结果，且**不需要排查任何问题**。
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
| 时间 | 首次约 35 分钟（装环境 5 + 拉模型 2 + 编译 7 + 跑 benchmark 3，其余是余量） |

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
...
    volumeMounts:
    - {name: dshm,    mountPath: /dev/shm}
    - {name: workdir, mountPath: /work}
    resources:
      # ③ limits 必须写在同一行里。分成两行会让后者覆盖前者，
      #    google.com/tpu 丢掉 limit → Job 直接被 API server 拒绝创建
      limits:   {google.com/tpu: 4, ephemeral-storage: 32Gi}
      requests: {google.com/tpu: 4, cpu: "8", memory: 48Gi, ephemeral-storage: 16Gi}
    tolerations:                                   # ④ 共享集群的 TPU 节点有两个 taint
    - {key: google.com/tpu,             operator: Equal, value: present, effect: NoSchedule}
    - {key: cloud.google.com/gke-queued, operator: Equal, value: "true",  effect: NoSchedule}
```

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
> 它只是「环境正确」的对照。真正的对标数据用有 baseline 的 config，见下。

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

1. **权重预置**。397B 约 400 GB，pod 本地盘只有 74 GB 放不下；内存盘（`/work` 已挂 320Gi）
   装得下，但每个 pod 都要重下一次。**并行跑多配置时把权重放 GCS 用 gcsfuse 只读共享。**
2. **编译缓存持久化**。缓存在 `/root/.cache/vllm/torch_compile_cache/`，
   0.6B 的 201 秒就绪时间里绝大部分是编译（权重才 1.2 GB）。397B 的图更大更多，
   不复用缓存的话每轮都要重付。

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

### 排障通用顺序

vLLM 起不来时，**不要在 runner 的输出里找答案**，它只是转述。按这个顺序往上走：

```
pod 日志  →  benchmark_runs/<ts>/server.log  →  kubectl describe pod
         →  kubectl get jobset -o json（conditions / replicatedJobsStatus）
         →  kubectl logs -n jobset-system -l control-plane=controller-manager
```

我们有一次 9 个任务全失败，真因只出现在最后那一层。
