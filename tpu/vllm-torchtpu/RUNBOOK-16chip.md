# RUNBOOK：16 chip TPU v7x 跑 Qwen3.5-397B 推理测试

> **契约**：你没碰过 TPU、没用过 vllm-torchtpu，照做能得到 18 格性能矩阵
> （TTFT / TPOT / ITL / E2EL / 吞吐），且不需要自己排查任何问题。
>
> 每步格式：**命令 → 期望看到什么 → 不对时怎么办**。标 ⚠️ 的都是实测踩过的坑，不要精简。
>
> 同目录另两份：[QUICKSTART](./QUICKSTART.md) 是 4 chip 最短路径；
> [RUNLOG](./Qwen3.5-397B-A17B-FP8/RUNLOG-20260814-16chip.md) 是数据与规律分析。

| 产出 | 3 并行策略 × 2 序列形状 × 3 并发 = **18 格** |
|---|---|
| 硬件 | 4 节点 × `tpu7x-standard-4t` = 16 chip / 32 device，拓扑 2x2x4 |
| 耗时 | **约 2 小时 20 分** = 准备 30 分（装环境与拉权重并行）+ 跑测 90 分（4 节点并行）+ 收数 20 分 |
| 前提 | pod 不受 60 分钟寿命限制（共享集群请改用 QUICKSTART） |

---

## 0. 前置检查

```bash
for c in kubectl gcloud python3; do command -v $c >/dev/null && echo "✓ $c" || echo "✗ $c"; done
kubectl get crd jobsets.jobset.x-k8s.io      # 没有则先装 JobSet operator
```

还需要能读 `us-docker.pkg.dev/ml-oss-artifacts-transient`（镜像 + pip 私有源）。

**定义变量。** 占位符用带引号的 `CHANGE_ME_*`，**不要写 `<...>`** ——
`<` 是 bash 重定向符，照抄粘贴会报语法错，而报错信息指不到真正原因。

```bash
export CTX="CHANGE_ME_context"     NS="default"
export POOL="CHANGE_ME_nodepool"   # TPU 节点池名
export IMG="us-docker.pkg.dev/ml-oss-artifacts-transient/torch-tpu-docker-container/torch-tpu:nightly-20260726"

for v in CTX NS POOL IMG; do case "${!v}" in ""|*CHANGE_ME*) echo "✗ $v 未改";; *) echo "✓ $v";; esac; done
kubectl --context="$CTX" get ns "$NS" >/dev/null && echo "✓ 连通"
```

**摸清节点规格**（决定后面所有容量参数）：

```bash
kubectl --context=$CTX get nodes -l cloud.google.com/gke-nodepool=$POOL -o custom-columns=\
'NAME:.metadata.name,TPU:.status.allocatable.google\.com/tpu,MEM:.status.allocatable.memory,\
EPH:.status.allocatable.ephemeral-storage,TOPO:.metadata.labels.cloud\.google\.com/gke-tpu-topology'
```

期望 4 行：`TPU=4`、`MEM=963568656Ki`（919 GiB）、`EPH=47060071478`、`TOPO=2x2x4`。

> ⚠️ **`EPH` 只有 43.8 GiB。** pip 编译 vLLM、`/tmp`、pip cache 都往容器盘写，会撞**节点级**驱逐。
> 危险在于容器里 `df` 看着宽裕 —— 那是整节点共享配额，**容器 `df` 看不见**。
> 第 1 步的 pod spec 已把这些目录挪到 tmpfs，别删。

> ⚠️ **有 Lustre PVC 也别用。** TPU 节点的 COS 镜像装了 `lnet`/`libcfs` 但**没有 lustre 客户端模块**，
> 挂载报 `exit status 19 / No such device`；`lustre-csi-node` 显示 `2/2 Running` 是假象。
> 进节点 `modprobe lustre` 会看到 `Module lustre not found in /lib/modules/6.12.68+`。
> 本 runbook 走 **tmpfs + 同区 GCS**。

---

## 1. 拿到 TPU 且不弄丢

DWS flex-start 节点 `BookingExpired=True` 后，**空闲即是 autoscaler 回收候选**。
删占位与提交真任务之间不能有空窗 —— 实测被回收后重新排队等了 38 小时。

**要点不是「手快」，是把会失败的事全部挪到删除之前。**

### 1.1 先预热镜像（零风险）

拉镜像不需要 TPU 资源，可与占位 Job 共存。这一步提前消化掉切换时最大的失败模式。

```bash
for N in $(kubectl --context=$CTX get nodes -l cloud.google.com/gke-nodepool=$POOL -o name | cut -d/ -f2); do
cat <<EOF | kubectl --context=$CTX apply -f -
apiVersion: v1
kind: Pod
metadata: {name: prepull-${N##*-}, namespace: $NS}
spec:
  restartPolicy: Never
  nodeName: $N
  containers: [{name: c, image: $IMG, command: ["bash","-c","echo PREPULL_OK; sleep 20"],
                resources: {requests: {cpu: 100m, memory: 512Mi}}}]
  tolerations: [{operator: Exists}]
EOF
done
```

期望约 2 分钟后全部 `Completed`。若 `ImagePullBackOff` → 权限问题，**先解决，不要动占位**。

### 1.2 三层校验

```bash
cd CHANGE_ME_repo/tpu/vllm-torchtpu   # 先进到本仓库这个目录
export RB=$PWD                       # 后面所有脚本路径都基于它
[ -d "$RB/Qwen3.5-397B-A17B-FP8/scripts" ] && echo "✓ RB=$RB" || echo "✗ 目录不对，请 cd 到 tpu/vllm-torchtpu"

cp $RB/Qwen3.5-397B-A17B-FP8/manifests/v7x-16chip-jobset.yaml jobset.yaml
sed -i "s#hy3-v7-16-dws-2#$POOL#" jobset.yaml     # 换成你的节点池名
```

四处**不能改**的地方：

| | 值 | 为什么 |
|---|---|---|
| `failurePolicy.maxRestarts` | **3** | TPU `SLICE_FAILURE` 反复重试会耗尽配额，**DWS 节点被整批回收**。别设 10 |
| `resources` | 每种资源**只出现一次** `limits` / 一次 `requests` | 两个 `limits` 块时后者覆盖前者，`google.com/tpu` 静默丢限额 |
| `/tmp`、`/root/.cache` | 挂 tmpfs | 否则撞节点级 ephemeral-storage 驱逐 |
| `/dev/shm` | 32Gi（开 MTP 要更大） | 见 5.3 |

```bash
python3 $RB/Qwen3.5-397B-A17B-FP8/scripts/validate-jobset.py jobset.yaml --context $CTX
# 期望：三层全过，可以提交
```

> ⚠️ **第 3 层不能省。** JobSet 的 `--dry-run=server` **只校验 JobSet CRD 本身，
> 不校验它生成的 Job**。真出问题时 dry-run 一路绿灯，真因只在
> `kubectl logs -n jobset-system -l control-plane=controller-manager` 里。
> 实测这个疏漏废掉过一整夜 9 个任务。

### 1.3 切换

```bash
kubectl --context=$CTX get jobs -n $NS          # 找占位 Job 名。删 pod 没用，会被重建
export PLACEHOLDER_JOB="CHANGE_ME_job_name"

kubectl --context=$CTX delete job "$PLACEHOLDER_JOB" -n $NS --wait=false; \
kubectl --context=$CTX apply -f jobset.yaml
```

> ⚠️ 两条命令必须在同一行、用 `;` 连接。中间**不要**加 `sleep` 或判断。实测切换 2 秒。

```bash
export PODS=$(kubectl --context=$CTX get pods -n $NS -o name | grep ttpu16 | cut -d/ -f2)
export P0=$(echo $PODS | awk '{print $1}')
kubectl --context=$CTX get pods -n $NS -o wide | grep ttpu16
```

期望约 10 秒后 4 个 `Running`，`NODE` 列互不相同（镜像已预热所以这么快）。

**节点空闲、无占位时**：跳过 1.1 和 delete，直接 apply。

---

## 2. 确认容器环境

```bash
kubectl --context=$CTX exec $P0 -n $NS -- bash -c 'env | grep "^TPU_" | sort; df -h /work /dev/shm /tmp'
```

期望 GKE 已注入多机 mesh：`TPU_ACCELERATOR_TYPE=tpu7x-32`、`TPU_HOST_BOUNDS=1,1,4`、
`TPU_TOPOLOGY=2x2x4`、`TPU_PROCESS_ADDRESSES` 有 4 个地址；`/work` 和 `/tmp` 各 640G、`/dev/shm` 32G。

> **口径**：v7x 上 **1 chip = 2 device**。`tpu7x-32` 的 32 是 device 数。
> vLLM 日志打的 `num_chips=8` 其实也是 device 数，别按字面理解。

---

## 3. 装环境 + 拉权重（并行）

一个打 pypi、一个打 GCS，不抢同一个瓶颈。实测装环境 2 分 50 秒，**完全被 28 分钟的下载掩盖**；
串行要白多花近 3 分钟。

```bash
# 本机打包。⚠️ 不要让 pod 自己去 GitHub 下 vLLM —— 实测踩过 503 和 RemoteDisconnected，各废一个窗口
git clone https://github.com/vllm-project/vllm-torchtpu.git
curl -L -o vllm-src.tgz https://github.com/vllm-project/vllm/archive/refs/tags/v0.26.1rc0.tar.gz
tar czf vtt.tgz --exclude='.git' vllm-torchtpu

D=$RB/Qwen3.5-397B-A17B-FP8/scripts     # $RB 在 1.2 步已定义
for P in $PODS; do (
  for F in vtt.tgz vllm-src.tgz; do kubectl --context=$CTX cp $F $NS/$P:/work/$F; done
  kubectl --context=$CTX cp $D/fetch-weights.py     $NS/$P:/work/fetch-weights.py
  kubectl --context=$CTX cp $D/bootstrap-16chip.sh  $NS/$P:/work/bootstrap.sh
  kubectl --context=$CTX cp $D/run-config.sh        $NS/$P:/work/run-config.sh
  kubectl --context=$CTX exec $P -n $NS -- bash -c 'cd /work && tar xzf vtt.tgz'
) & done; wait
```

**权重走同区 GCS，不走 HuggingFace**：HF 967 MB/s，同区 GCS 单节点 231 MiB/s、
4 节点聚合 990 MiB/s。先确认 pod 能读你的桶：

```bash
kubectl --context=$CTX exec $P0 -n $NS -- python3 -c "
import urllib.request as u,json
h={'Metadata-Flavor':'Google'}
t=json.load(u.urlopen(u.Request('http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/token',headers=h),timeout=5))['access_token']
print('✓ token 长度',len(t))"
```

不通就给 KSA 绑 Workload Identity，或把 `fetch-weights.py` 改成 `snapshot_download` 走 HF（多约 8 分钟）。

```bash
TOK=$(gcloud auth print-access-token)     # ⚠️ 只活 1 小时
for P in $PODS; do
  kubectl --context=$CTX exec $P -n $NS -- bash -c \
    "AR_TOKEN='$TOK' nohup bash -c 'bash /work/bootstrap.sh 2>&1 | tee /work/bootstrap.log' >/dev/null 2>&1 &"
done
```

**看进度**：

```bash
for P in $PODS; do kubectl --context=$CTX exec $P -n $NS -- bash -c \
  'echo "$(hostname) $(du -sh /work/models/qwen3.5-397b 2>/dev/null|cut -f1)"'; done
```

> ⚠️ **看 `du`，不要看进度条。** 94 个 4 GB 分片、48 并发时，前 20 分钟一个文件都不会「完成」，
> 按文件计数的进度条会一直显示 `0 MiB/s` 像卡死 —— 而那时盘上已经有 359 GiB。
> **大文件批量传输的进度必须按已落盘字节算。**

**完成判据**（4 个都满足才往下走）：

```bash
for P in $PODS; do kubectl --context=$CTX exec $P -n $NS -- bash -c \
  'grep -q BOOTSTRAP_OK /work/bootstrap.log && echo "$(hostname) OK $(du -sb /work/models/qwen3.5-397b|cut -f1) bytes $(ls /work/models/qwen3.5-397b/*.safetensors|wc -l) 分片"'; done
```

期望 4 行 `OK`，**字节数完全相同**，分片数都是 94。

> **重跑是安全的。** `fetch-weights.py` 按**字节数**校验而不是只看文件在不在，
> 所以缺失的和内容损坏的都会被补下，已完整的跳过。
> 实测：删掉 2 个分片 + 把第 3 个截断成 1000 字节后重跑，**41 秒**恢复到逐字节一致。
> 中途失败直接重跑即可，不必从头来。
版本矩阵应为 torch 2.13.0 / jax 0.10.2 / libtpu 0.0.44.1 / torch-tpu 0.1.1.dev20260804130134 /
vllm 0.26.1rc0+tpu，`platform: TpuPlatform`、`tpu devices: 8`。

> ⚠️ 镜像自带 torch 2.11 / jax 0.9.2 / libtpu 0.0.41，**比 `pyproject.toml` 要求的旧一档，必须升**
> （`jax 0.9.2` 没有 `pallas.tpu.BufferType`，一跑就崩）。
> 且**必须用 `pip` 不能用 `uv`** —— uv 不读 `PIP_INDEX_URL`，会报「找不到 torch-tpu」而包一直都在。

---

## 4. 把 16 chip 切成 4 个独立 4 chip 环境

18 格串行要 6 小时，切成 4 份并行**约 90 分钟**。

**原理：这不是虚拟化，是「拒绝组队」。** 硬件本来就是 4 台机器各插各的 4 颗芯片；
「16 卡是一个整体」是 libtpu 启动时按环境变量去跟其他 3 台会合才形成的。改掉变量就不去会合。
不需要隔离机制 —— 每台物理上本来就摸不到别人的芯片。

| 变量 | GKE 注入 | 改成 |
|---|---|---|
| `TPU_WORKER_ID` | 0/1/2/3 | `0` |
| `TPU_PROCESS_ADDRESSES` | 4 个 DNS 名 `:8471` | `localhost:8471` |
| `TPU_WORKER_HOSTNAMES` | 同上 4 个 | `localhost` |
| **`TPU_HOST_BOUNDS`** | **`1,1,4`** | **`1,1,1`** ← 核心只有这一个 |
| `TPU_CHIPS_PER_HOST_BOUNDS` | `2,2,1` | `2,2,1`（**不变**） |
| `TPU_ACCELERATOR_TYPE` | `tpu7x-32` | `tpu7x-8` |

再加 `unset TPU_MULTIHOST_BACKEND`。三个易错点：
① 地址表和主机名表是一对，**只改一个会对不上**；
② `TPU_CHIPS_PER_HOST_BOUNDS` 不用改，每台本来就是 4 颗；
③ **1 chip = 2 device**，写错报拓扑不匹配，**而错误信息不会提示是单位搞错了**。

```bash
kubectl --context=$CTX exec $P0 -n $NS -- bash -c 'cd /tmp
export TPU_WORKER_ID=0 TPU_PROCESS_ADDRESSES=localhost:8471 TPU_WORKER_HOSTNAMES=localhost
export TPU_HOST_BOUNDS=1,1,1 TPU_CHIPS_PER_HOST_BOUNDS=2,2,1 TPU_ACCELERATOR_TYPE=tpu7x-8
python3 -c "import jax; print(len(jax.devices()))"'
```

| 返回 | 含义 |
|---|---|
| **8** | 对了 |
| 32 | 变量没生效，仍在多机视图 |
| **卡住不返回** | 还在等另外 3 台会合 —— 地址表没改干净 |

> **代价**：跨机 ICI 链路完全闲置。只适合**单机装得下**的模型 ——
> 397B FP8 权重 378 GiB，单机 768 GB HBM 装得下。

（`run-config.sh` 的 `single` 模式已内置这组变量，直接用即可。）

---

## 5. 跑 benchmark

### 5.1 分派：按编译缓存亲和性

**同一节点始终跑同一并行策略**，否则吃不到编译缓存，每次重编 19 分钟。

> ⚠️ **自定义 config 必须分发到每个要用它的 pod。** config 存在容器本地的
> `scripts/vllm/benchmarking/configs/`，在 A 节点上生成的 config，B 节点看不见。
> 实测漏了这步，B 节点直接 `RUN_FAILED: 找不到 config`，白等一个窗口。
> 生成后先逐台确认：
>
> ```bash
> for P in $PODS; do kubectl --context=$CTX exec $P -n $NS -- bash -c \
>   'ls /work/vllm-torchtpu/scripts/vllm/benchmarking/configs/'"$CFG"'.sh 2>/dev/null || echo "$(hostname) 缺配置"'; done
> ```

```bash
run(){ kubectl --context=$CTX exec $1 -n $NS -- bash -c \
  "nohup bash -c 'bash /work/run-config.sh $2 single $3 $4; echo CHAIN_DONE' > /work/run.log 2>&1 &"; }
set -- $PODS
run $1 qwen3.5-397b-fp8-tp8-dp1-ep '"1024:8192"' '"64 256 512"'
run $2 qwen3.5-397b-fp8-tp8-dp1-ep '"8192:1024"' '"64 256 512"'
run $3 qwen3.5-397b-fp8-tp2-dp4-ep '""' '""'      # 空 = 跑 config 自带的全部 6 格
run $4 qwen3.5-397b-fp8-tp1-dp8-ep '""' '""'
```

### 5.2 起跑 3 分钟后做一次健康检查（能省一小时）

```bash
for P in $PODS; do kubectl --context=$CTX exec $P -n $NS -- bash -c \
  'L=$(ls -t /work/vllm-torchtpu/benchmark_runs/*/server.log|head -1); echo "$(hostname) $(wc -l < $L) 行"
   grep -h "Loading weights took" "$L" 2>/dev/null'; done
```

期望 `Loading weights took 69.28 seconds`（tmpfs）。**若是 240 秒左右**，说明权重落在磁盘不是内存盘，
回头检查 pod spec 的 volume 配置。

### 5.3 ⚠️ 要开投机解码（MTP）？两个坑

397B checkpoint 里有 3096 个 `mtp.*` 权重，投机解码可用，实测接受率 **93–95%**。
但会踩两个坑，**第二个尤其反直觉**。

#### 坑一：`/dev/shm` 撑爆

397B checkpoint 里有 3096 个 `mtp.*` 权重，投机解码可行，但 32 GiB 的 `/dev/shm` **一定会被撑爆**。

> **这个坑的伪装最强**：报出来的是 **SIGBUS（信号 7）**，栈顶停在 `determine_available_memory`，
> 看起来像 HBM 不够。实际是 tmpfs 写满 —— mmap 时成功、访问时才 fault，看不到干净的 `ENOSPC`。
> **诊断先跑 `df -h /dev/shm`，别急着调 `gpu-memory-utilization`。**

不重启 pod 的修法（重启要重下 378 GB 权重 + 重编缓存，代价 30 分钟）：

```bash
kubectl --context=$CTX exec $P -n $NS -- bash -c '
N=$(pgrep python3 | wc -l)   # 用 wc 不用 pgrep -c：数量为 0 时 pgrep -c 返回码非 0，配 || 会输出两行      # ⚠️ 别用 pgrep -f 加模式串：模式串就在本命令行里，会匹配到执行它的 bash 自己
[ "${N:-0}" -gt 0 ] && { echo "还有 python 在跑"; exit 1; }
rm -rf /dev/shm/torch_tpu_cache && mkdir -p /work/ttc
ln -sfn /work/ttc /dev/shm/torch_tpu_cache      # /work 同样是内存盘但有 640 GiB
df -h /dev/shm'
```

**不开 MTP 也要留意**：常规跑测下 `/dev/shm` 会涨到 **69–78%**。建议起手就把 `sizeLimit` 提到 64Gi。

#### 坑二：编译缓存全命中时 MTP 反而会失败

同一份 serve 配置，跑 4 次的结果完全由缓存状态决定：

| 缓存加载 | 新编图 | 动态形状告警 | 结果 |
|---|---|---|---|
| 8 | 20 | 8 | ✓ 成功 |
| **168** | **0** | 8 | ✗ 失败 |
| 全命中 | 0 | — | ✗ **失败（51 秒即崩）** |
| 0 | 21 | 8 | ✓ 成功（接受率 93.2%） |

报错是 `NotImplementedError: TPU backend: does not support dynamic shape`
（`torch_tpu/_internal/compile/_backend.py`）。

> **别被这个错误名带偏。** 成功的那两次**同样报了 8 次动态形状告警** ——
> 那不是失败原因。真正的差别是：能重新编译时（新编图 20–21 个），
> 撞到动态形状后会走回退路径重编；而缓存全命中时一个图都不编，
> **回退路径没机会触发，告警就升级成致命错误**。
>
> **处置：开 MTP 前先清编译缓存** —— `rm -rf $VLLM_CACHE_ROOT/* /work/ttc/*`。
> 代价是多花约 20 分钟冷编译，但这是目前唯一稳定的路径。

#### MTP 到底值不值（口径对齐后的实测）

同为 tp8-dp1 / ISL 1024 / OSL 1024 / 640 prompts：

| 并发 | 不开 MTP | 开 MTP | 差 |
|---|---|---|---|
| 16 | 1237 tok/s | **1397** | **+12.9%** |
| 64 | **2631** | 2406 | **−8.6%** |

**低并发有收益、高并发反而拖累**，交叉点在并发 16 与 64 之间。
这与 GPU 侧「MTP 是单个最大吞吐杠杆」的结论**不一致** ——
那边的参考场景是小模型（27B dense）在 B200 上算力吃不满，
而 397B MoE 在 v7x 上高并发时本就算力饱和，投机解码的额外前向反成负担。

`--speculative-config` 的 JSON 引号必须转义，否则内层双引号会截断外层字符串：

```bash
# 对： --speculative-config {\"method\":\"mtp\",\"num_speculative_tokens\":1}
# 错： --speculative-config {"method":"mtp","num_speculative_tokens":1}
source your-config.sh   # 写完一定 source 后实际解析验一遍，别只看源码
python3 -c "import json,sys;print(json.loads(sys.argv[1]))" \
  "$(echo "$EXTRA_SERVE_ARGS" | grep -o '{"method".*num_speculative_tokens":1}')"
```

### 5.4 口径（不满足就没有可比性）

| 项 | 值 | 为什么 |
|---|---|---|
| `BENCHMARK_TEMPERATURE` | **0** | golden client 默认 temp 0，而 `vllm bench serve` 默认走 server 端 sampling（模型 `generation_config` 是 temp 0.7），decode 重的 cell 上明显更慢，两者不可比 |
| `NUM_PROMPTS` | 640（并发 ≤32 时降到 64） | 并发 4 跑 640 条要几小时 |
| `RANDOM_RANGE_RATIO` | 0.8，`min` 风格 | 采样长度落在 [0.8·len, len]，保证不超 `max-model-len` |

逐台确认，不要假设：

```bash
for P in $PODS; do kubectl --context=$CTX exec $P -n $NS -- bash -c \
  'grep -h "^BENCHMARK_TEMPERATURE" /work/vllm-torchtpu/scripts/vllm/benchmarking/configs/*.sh|head -1'; done
```

---

## 6. 收结果

```bash
python3 $RB/Qwen3.5-397B-A17B-FP8/scripts/collect-results.py \
  --context $CTX --ns $NS --hours 4 --repo ./vllm-torchtpu
```

期望 18 行，每行 `640/640`，吞吐差异全部落在 ±5% 内。

> ⚠️ **`--hours` 不能省。** 结果目录里会混进上一轮的 run 目录 ——
> `run-config.sh` 的后台 copier 是 `cp -r benchmark_runs/*`，会把历史全拷进去，
> 不按时间过滤就会把旧数当新数。

**每格必须记全**：TTFT / TPOT / ITL / E2EL 各自的 mean·median·p99，
`total_input_tokens`、`total_output_tokens`、`request_throughput`、`duration`，
**以及产生它们的 ISL / OSL / 并发 / 并行配置** —— 脱离配置的吞吐数字没有意义。

---

## 7.（可选）要和 GPU 公开数据对比，先补三件事

直接拿本轮数字比 B200 **会得出错误结论**：

| 维度 | 本轮 | 公开 B200 数据 | 怎么补 |
|---|---|---|---|
| 序列形状 | 1024:8192 / 8192:1024 | InferenceX 用 **1K/1K**，GCloud 博客用 1024/512 | 改 `ISL_OSL_CONFIGS` |
| 交互速度 | 21–47 tok/s/user | 报的区间是 **59–272** | 并发压到 4/16/32，同时 `NUM_PROMPTS` 降到 64 |
| 投机解码 | **没开** | 开了 MTP / EAGLE | 见 5.3 |

> 第三条影响最大：同一篇 Google Cloud 实测博客量化过，关掉 MTP 吞吐掉三分之一，
> 原话是「MTP 是单个最大的吞吐杠杆」。不对齐等于白让三成。

---

## 故障速查

| 症状 | 真因 | 处理 |
|---|---|---|
| 一直 `Waiting for server...` 到超时 | **镜像里没 curl**，runner 探 `/health` 永远失败（server 其实是好的） | `apt install curl`。这一个包废掉过三个窗口 |
| `SIGBUS` / `signal 7`，栈顶 `determine_available_memory` | **`/dev/shm` 写满**，不是 HBM 不够 | 先 `df -h /dev/shm`，按 5.3 处理 |
| PVC 挂不上 `exit status 19 / No such device` | TPU 节点 COS 镜像**无 lustre 客户端模块**；CSI pod `2/2 Running` 是假象 | `modprobe lustre` 确认后改用 tmpfs + GCS |
| 容器 `df` 很空但 pod 被驱逐 | ephemeral-storage 是**整节点共享配额**，容器 `df` 看不见 | `/work`、`/tmp`、`/root/.cache` 全挂 tmpfs |
| `--dry-run=server` 全绿但没有 pod | dry-run **只校验 JobSet CRD，不校验它生成的 Job** | 跑 `validate-jobset.py`；真因在 `jobset-system` 控制器日志 |
| 校验脚本第 2 层报 `Forbidden ... applying patch` | 同名 JobSet 已存在，dry-run 走 patch 撞 immutable 字段 | 脚本已自动改名规避；手工验时也要改名 |
| 进度条 `0 MiB/s` 像卡死 | 按**完成文件数**计数，而 4 GB 分片前 20 分钟一个都完不成 | 看 `du -sh` |
| 热启动没吃到编译缓存 | **缓存 key 含完整 serve 配置**，差一个参数全废 | 用 runner 重跑，别手写 `vllm serve` |
| 查了「缓存全命中」但启动仍慢 | 只查了 `"Compiling a graph for compile range"`（确实是 0） | 还要查 `"Compiling model again"` —— 实测 104 个产物里 **16 个每次重编**，上游 bug |
| `pgrep -f "vllm serve"` 报有进程但实际没有 | 模式串在本命令行里，**匹配到执行它的 bash 自己** | 用 `pgrep -c python3` |
| 脚本报成功但没有结果文件 | 脚本最后一行是 `echo`，**退出码永远 0**，内部失败传不出来 | `run-config.sh` 已加 `exit $RC`；自己写编排脚本时同样要透出失败码 |
| MTP 报 `does not support dynamic shape` | **编译缓存全命中**导致回退重编路径没触发，不是真的不支持 | 清 `$VLLM_CACHE_ROOT` 和 `/work/ttc` 后重跑，见 5.3 |
| 报 TPU 拓扑不匹配 | `tpu7x-N` 的 N 是 **device 数不是 chip 数** | 4 chip 写 `tpu7x-8`，16 chip 写 `tpu7x-32` |

---

**已实测产出**：18/18 格完成，640/640 零失败，吞吐 14 胜 4 负 vs baseline（全在 ±4.3% 内）。
数据与规律见 [RUNLOG](./Qwen3.5-397B-A17B-FP8/RUNLOG-20260814-16chip.md)。
