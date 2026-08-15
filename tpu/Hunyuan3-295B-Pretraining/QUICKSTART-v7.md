> 🌐 **中文** | [English](QUICKSTART-v7.en.md)

# 混元 3（295B-A21B）在 TPU v7 (Ironwood) 上预训练 — Quick Start

用 MaxText 在 TPU v7 上跑腾讯混元 3。**照着 [§2 配方](#二直接可抄的配方) + [§3 六步](#三跑起来六步)走，第一次就能拿到当前最高水位。**

| | |
|---|---|
| 模型 | Tencent Hunyuan 3，295B 总参 / 21B 激活，**完整 80 层** MoE |
| 平台 | TPU v7 Ironwood（`tpu7x-standard-4t`，**2 device / chip**） |
| 框架 | MaxText（nnx），代码在 [`yangwhale/maxtext` 的 `hunyuan3` 分支](https://github.com/yangwhale/maxtext/tree/hunyuan3) |
| 数据 | 合成数据、seq 4096、稳态取 step 4–7（只测吞吐，不测收敛） |

---

## 一、性能是怎么一路上去的

> [!important] 2026-08-15 更新：BF16 水位从 630 提到 **662**，配方也换了
> **推荐路径已从「tokamax + `tkcfg.py` 注入」改为「默认 megablox + 18 个配置参数」。**
>
> v7 上有两条都能拿到 tile 收益的路。下表第 1 行走的是 tokamax 路径
> （`use_tokamax_gmm=True` + monkeypatch）。但那个开关在 v7 上有死锁风险
> （[TUNING-v7 §6.7](TUNING-v7.md)），所以 `run.sh` 一直没开它 ——
> **同时也没传另一条路要的 18 个配置参数**，结果实际跑在默认 tile 上，白丢 26%。
>
> 而且这个状态不会自己暴露：`tkcfg.py` 挂上去也没用（它打在没启用的那条路上），
> 加计数器实测 **「被调用 0 次」，却照常打印 `[tkcfg] patched`**。
>
> 换成 18 个配置参数 `{wi,wo}_tile_{fwd,dlhs,drhs}_{batch_seq,embed_dim,mlp_dim}`
> = `(512, 2048, 1536)`，**64 芯片实测**：
>
> | 配方 | per-chip | MFU | 备注 |
> |---|---:|---:|---|
> | 下表第 3 行（tokamax + 注入） | 630 | 27.31% | 需要开 `use_tokamax_gmm` |
> | **18 参数 + `pdbs=12`** | **662.3** | **28.71%** | **不需要开它，无死锁风险** |
> | **18 参数 + `pdbs=13`** | **666.6** | **28.89%** | `pdbs` 上限，14 差 1.79 G 装不下 |
>
> 快出来的 5.1% 猜测来自反向两条路径（`dlhs`/`drhs`）也被 tile 了 —— **未验证**。
> tile 值本身两条路径最优点相同，扫描与调法见 [TUNING-v7 §3.4.5/3.4.6](TUNING-v7.md)。
> **下表 0-4 行是历史线，保留原样不改；新水位见上表。**

**全部在 64 芯片（16 节点 / 128 device）上实测，一条线走到底 —— 每一行只加一件事。**

| # | 加了什么 | step | **TFLOP/s/chip** | tok/s/chip | 增量 | **累计** |
|---|---|---|---|---|---|---|
| 0 | 起点：`megablox` + pdbs 8 | 19.92 s | 457 | 3,290 | — | — |
| 1 | **tokamax `tile(512,2048,1536)`** | 16.75 s | **543** | 3,912 | +18.8% | **+18.8%** |
| 2 | **`per_device_batch_size` 8 → 12** | 23.54 s | **580** | 4,176 | +6.8% | **+26.9%** |
| 3 | **`--xla_tpu_dvfs_p_state=7`** | 21.67 s | **630** | 4,536 | +8.6% | **+37.9%** |
| 4 | **FP8 + QAG**（pdbs 降到 7） | **11.81 s** | **674** | **4,854** | +7.0% | **+47.5%** |
| 5 | **FP8 换 tile 入口**（2026-08-15） | **7.85 s** | **1,014.8** | **7,308** | **+50%** | **+122%** |

> ⚠️ **step 时间这一列不能横着比大小** —— 每行的 batch 不一样。
> 第 2 步 step 从 16.75 涨到 23.54，是因为 batch 从 8 加到 12，**单步算的 token 多了 50%**；
> 第 4 步 step 砍半到 11.81，是因为 FP8 把 batch 降回了 7。
> **唯一可以横比的是 `tok/s/chip`**，它已经把 batch 和 step 都归一掉了。

**每一步在做什么、为什么值这么多：**

| # | 一句话 | 代价 | 详情 |
|---|---|---|---|
| 1 | MaxText 不暴露 tokamax tile，**默认值回退到 `128³`，慢 12.4 倍**。6 行 monkeypatch 注入正确 tile | 显存 +1.1 G | [§3.4](#34-注入-tokamax-tile必做收益最大的一步) |
| 2 | 显存换吞吐，74 → 92 G，逼近 94.74 上限 | 14 会 OOM | [§4.1](#41-完整参数集) |
| 3 | 把芯片锁在最高频率档 | **零代价**，HBM 一字节没涨 | 见下方折叠区 |
| 4 | 计算换精度；QAG（量化后再 all-gather）省 4.5–11 G 通信显存 | MFU 分母变 4614，pdbs 只能到 7 | [§2](#二直接可抄的配方) |

**对照参考：GB300 = 6,242 tok/s/GPU**（同为 seq 4096，可直接比）。
v7 单芯片从起点的 **52.7%** 走到现在的 **77.8%**。

<details>
<summary><b>关于 dvfs_p_state</b> —— 四轮对照、档位单调、7 已顶格</summary>

同一批节点、同一个常驻 pod、同一份代码，只差这一个 flag：

| flag | step | TFLOP/s/chip | MFU | 峰值 HBM |
|---|---|---|---|---|
| 不带 | 23.537 s | 579.9 | 25.14% | 91.94 G |
| `=3` | 23.539 s | 579.9 | 25.14% | 91.94 G |
| `=5` | 22.438 s | 608.4 | 26.37% | 91.94 G |
| **`=7`** | **21.670 s** | **629.9** | **27.31%** | 91.94 G |

- **默认档就是 3**（`=3` 与不带逐位一致），每档约 +2.4%，单调无拐点
- **合法范围 `[0,7]`**，`=9` 直接 `INVALID_ARGUMENT`，7 已顶格
- v7 专用，v6e 不支持
- **它提的不是整颗芯片的频率**：同一 flag 在 BF16 上 +8.6%、在 FP8 上 +8.0%，
  五个负载的收益与 compute-bound 程度相关（r=0.967）。反推核心频率约 +12.5%，
  Hy3 有约 29% 的 step 时间卡在 HBM/ICI 上，频率买不动。
  推导见 [EXPERIMENT-LOG](EXPERIMENT-LOG.md)

</details>

---

## 二、直接可抄的配方

三组参数，其余部分完全一致（见 [§4.1 完整参数集](#41-完整参数集)）。

### ⚠️ 64 芯片 FP8 → 1,014.8（**存疑，暂勿用于生产**）

> [!warning] 2026-08-15 20:35：这个配方存在正确性疑点
> profile 显示 native 路径每卡只拿 **3** 个专家（tokamax 是 192，比值正好等于 FSDP 宽度 64），
> 而本模型的 MoE 设计是「token 不搬家、每卡 gather 全部专家」。**可能是算少了。**
> loss 10 步对得上（差 ≤0.001）与之矛盾，但 10 步可能太短。
> **30 步以上的对拍未做。定论前请继续用下面的旧配方（677）。**
> 完整调查见 [TUNING-v7 §3.4.8](TUNING-v7.md)。

在 BF16 那套 18 个 tile 配置参数的基础上，加这几项：

```
ici_data_parallelism=2           # QAG 的整除锁：192 个专家只能 FSDP=64
ici_fsdp_parallelism=64
per_device_batch_size=7          # 上限就是 7，AOT 实测 8 要 98.05 G
use_qwix_quantization=True
quantization=fp8_full
shard_exp_on_fsdp=True
weight_quantization_calibration_method=fixed,-224,224
act_quantization_calibration_method=fixed,-224,224
# ⚠️ 不要开 use_tokamax_gmm —— 开了反而慢 50%，见下
```

> [!important] 2026-08-15：FP8 换 tile 入口后 674 → **1,014.8**，快 **49.9%**
> | 配方 | step | per-chip | MFU<sub>4614</sub> | tok/s/chip | 峰值 HBM |
> |---|---:|---:|---:|---:|---:|
> | 旧：`use_tokamax_gmm` + `tkcfg.py` 注入 | 11.761 s | 677.0 | 14.67% | 4,876 | 92.42 G |
> | **新：默认 megablox + 18 个 tile 配置参数** | **7.847 s** | **1,014.8** | **21.99%** | **7,308** | 91.40 G |
>
> **loss 前两步完全相同，之后每步差 ≤0.001**（打印精度的最后一位，见下方核验），峰值 HBM 还低 1 G。
>
> **这里跟 BF16 那条不一样，值得单独说清楚：**
> FP8 这条路上 `tkcfg.py` **是真的生效的** —— 加计数器实测被调用，
> 而且打印出 tokamax 的默认启发式是 `128,128,128`，印证了「不注入会回退到 128³」。
> 所以旧的 674 不是「没调 tile」，是**调了，但那条 kernel 路径本身就慢 50%**。
>
> ⚠️ **FP8 不带任何 tile 参数会直接崩**，不是变慢：
> `AssertionError: v=1536 bv=1024 s=1536` —— 默认 `mlp_dim` tile 是 1024，除不尽 1536。
> （BF16 走 `jax.lax.ragged_dot`，那条路会 `min()` 裁剪所以不崩；
> FP8 走 `mblx.gmm`，直接断言。**同一个默认值，两条路一个崩一个只是慢。**）

### ⚡ 64 芯片 BF16 → **662**（2026-08-15 起的推荐配方）

```
ici_fsdp_parallelism=-1          # 自动吃满 128 路，等价 DP=1
ici_tensor_parallelism=1
per_device_batch_size=12         # 91.94 G；13 也能跑（92.57 G，+0.66%），14 OOM
megablox=True                    # ⚠️ 不要开 use_tokamax_gmm（§6.7 死锁）
--xla_tpu_dvfs_p_state=7         # 去掉这行约 -8.6%
# 18 个 tile 参数，{wi,wo} × {fwd,dlhs,drhs} × (512, 2048, 1536)
wi_tile_fwd_batch_seq=512  wi_tile_fwd_embed_dim=2048  wi_tile_fwd_mlp_dim=1536
wi_tile_dlhs_batch_seq=512 wi_tile_dlhs_embed_dim=2048 wi_tile_dlhs_mlp_dim=1536
wi_tile_drhs_batch_seq=512 wi_tile_drhs_embed_dim=2048 wi_tile_drhs_mlp_dim=1536
wo_tile_fwd_batch_seq=512  wo_tile_fwd_embed_dim=2048  wo_tile_fwd_mlp_dim=1536
wo_tile_dlhs_batch_seq=512 wo_tile_dlhs_embed_dim=2048 wo_tile_dlhs_mlp_dim=1536
wo_tile_drhs_batch_seq=512 wo_tile_drhs_embed_dim=2048 wo_tile_drhs_mlp_dim=1536
```

> `maxtext-hunyuan3/run.sh` 的 v7 分支已默认带上这一整套，直接跑即可。
> **漏掉那 18 行会掉到 525（−26%），而且不报错。**

### ⚡ 旧配方（tokamax 注入路径，630）—— 保留备查，不再推荐

```
megablox=True use_tokamax_gmm=True     # ⚠️ 有死锁风险
TK_TM=512 TK_TK=2048 TK_TN=1536        # 靠 tkcfg.py monkeypatch 注入
```

### 256 芯片 BF16 → **599**

```
ici_data_parallelism=2
ici_fsdp_parallelism=256         # 加宽 FSDP，每卡静态分片减半，省 13 G
ici_tensor_parallelism=1
per_device_batch_size=16         # 省下的显存全喂给 batch，92.33 G
megablox=True use_tokamax_gmm=True
TK_TM=512 TK_TK=2048 TK_TN=1536
--xla_tpu_dvfs_p_state=7         # 256 芯片上未复测，64 芯片实测 +8.6%
```

**两条切分规律：**

1. **默认把 FSDP 宽度固定在 128，多出来的 device 全给 DP。** 64 芯片正好 128 device 所以 `DP=1`，
   256 芯片 512 device 所以 `DP=4`。
2. **≥256 芯片时可以换一条路**：FSDP 加宽到 256，用省下的 13 G 换更大 batch（580 → 599，+3.3%）。
   64 芯片没有更宽的 FSDP 可选，所以在**分片维度上**到 580 为止 —— 但频率（第 3 步）和精度（第 4 步）
   是另外两个正交维度，64 芯片最终反超 256 芯片。

**EP（专家并行）不要用 —— 两个规模都测过，方向一致。**

| 规模 | EP | 结果 |
|---|---|---|
| 16 芯片 | EP=4 | −71% |
| **64 芯片** | **EP=2** | **−39.6%**（380 vs 630） |

代价不只是 AllToAll 在 3D torus 上多跳转发，更致命的是**它逼着 FSDP 减半**
（128 device ÷ EP 2 = 64），每卡静态分片翻倍；再加上非专家参数在 EP 轴上只能复制
（实测 24% 未分片），batch 被从 12 压到 6。

> ⛔ **EP + FP8 在当前 MaxText 上直接不可用**：`mblx.gmm` 的反向规则不支持专家维度被切，
> 报 `Custom VJP bwd rule ... bfloat16[96,...] vs bfloat16[192,...]`。这是 kernel 实现限制，绕不过去。

**TP（张量并行）也不要用，同样测过。** `TP=2` 确实省 25.96 G 显存（92.42 → 66.46 G，比 QAG 省得还多），
但同 batch 慢 **30.8%**，把省下的显存全换成 batch（7 → 12）只补回 8%，**净亏 25.3%**（503 vs 674）。
根因**不是**跨芯片通信 —— 实测 `create_device_mesh` 默认就把 EP/TP 这个宽度为 2 的维度
映射到**同一颗芯片的两个 chiplet**（64/64 行，走 D2D 1.2 TB/s，零跳），这一层已经是最优了。
真正的代价是**每引入一个并行维度就要从 FSDP 借宽度**：FSDP 从 128 掉到 64，每卡静态分片翻倍。

> 试过 `custom_mesh_and_rule` 让 TP **只切 MoE、不切 attention**（想省下那 2% 参数的通信），
> 结果**更慢 20.2%**（371.8 vs 466.2）—— TP 切 attention 同时也是**计算分摊**，
> 摘掉后每卡要算全量 attention。**参数占比 2% ≠ 计算占比 2%。**

---

## 三、跑起来（六步）

> 自建集群还要先做节点池等前置，见文末折叠区**「从零搭环境」**。
> 用别人的托管 / Kueue 集群可直接从这里开始。

### 3.1 起一个常驻 pod（不要用一次性 Job）

三个理由：**占住 slice**（共享集群里放手 30 秒就被抢）、**复用编译缓存**（第二轮起从 10+ 分钟降到秒级）、代码只拉一次。

```yaml
apiVersion: jobset.x-k8s.io/v1alpha2
kind: JobSet
metadata: {name: hy3-dev, namespace: <NS>}
spec:
  failurePolicy: {maxRestarts: 3}          # ⚠️ 不要设 10，见踩坑区
  replicatedJobs:
  - name: slice-job
    replicas: 1
    template:
      spec:
        parallelism: 16                     # 芯片数 ÷ 4；256 芯片写 64
        completions: 16
        backoffLimit: 0
        template:
          spec:
            restartPolicy: Never
            priorityClassName: medium       # ⚠️ 不写 = prio 0，随时被抢占
            nodeSelector:
              cloud.google.com/gke-tpu-accelerator: tpu7x
              cloud.google.com/gke-tpu-topology: "4x4x4"    # 256 芯片写 4x8x8
              # 托管集群通常还要加 reservation-name / queue-name
            hostNetwork: true
            dnsPolicy: ClusterFirstWithHostNet
            tolerations: [{operator: "Exists"}]
            containers:
            - name: jax-tpu
              image: <MAXTEXT_RUNNER_IMAGE>
              securityContext: {privileged: true}
              ports: [{containerPort: 8471}, {containerPort: 8080}]
              command: ["bash","-c"]
              args: ["gcloud storage cp <GCS_STAGE>/hy3-maxtext.tgz /tmp/p.tgz &&
                      cd /deps && rm -rf src/maxtext && tar xzf /tmp/p.tgz && sleep infinity"]
              resources: {limits: {google.com/tpu: 4}}
              volumeMounts: [{mountPath: /dev/shm, name: dshm}]
            volumes: [{name: dshm, emptyDir: {medium: Memory}}]
```

代码包由 `prep.sh` 生成（clone `hunyuan3` 分支 → 8 项自检 → tar **整棵** `src/maxtext` → 传 GCS）。
**整棵覆盖，不是只注入改动文件** —— 只注入的话，测的是「我的改动 + 容器里的旧基座」。

### 3.2 齐步执行脚本

多机 TPU 必须所有 pod 同时起。存成 `hy3-run.sh`：

```bash
#!/bin/bash
CMD=${1:?}; NS=${NS:-default}; JS=${JS:-hy3-dev}; NP=${NP:-16}
mapfile -t PODS < <(kubectl get pods -n $NS -l jobset.sigs.k8s.io/jobset-name=$JS \
  --field-selector status.phase=Running --no-headers | awk '{print $1}' | sort)
[ ${#PODS[@]} -eq $NP ] || { echo "需要 $NP 个 Running pod, 现在 ${#PODS[@]}"; exit 1; }
echo "[hy3-run] ${#PODS[@]} 个 pod 并行执行（下方是 worker-0 实时输出）"
TMP=$(mktemp -d); trap 'rm -rf $TMP' EXIT
for i in "${!PODS[@]}"; do
  if [ "$i" -eq 0 ]; then
    # worker-0 必须实时 tee，否则 6–30 分钟里无法区分「在编译」和「已卡死」
    timeout -k 30 2700 kubectl exec "${PODS[$i]}" -n $NS -c jax-tpu -- \
      bash -c "$CMD" 2>&1 | tee "$TMP/0.out" &
  else
    timeout -k 30 2700 kubectl exec "${PODS[$i]}" -n $NS -c jax-tpu -- \
      bash -c "$CMD" > "$TMP/$i.out" 2>&1 &
  fi
done
wait
grep -lE "^Traceback" $TMP/*.out | sed 's/^/⚠ 报错: /'
grep -ohE "SLICE_FAILURE_[A-Z_]+" $TMP/*.out | sort -u | sed 's/^/🔴 硬件故障: /'
```

### 3.3 冒烟（必做，不对就别往下走）

```bash
NS=<ns> JS=<jobset> NP=16 bash hy3-run.sh 'python3 -c "import jax;print(jax.device_count())"'
```

| 规模 | `NP` | 应返回 |
|---|---|---|
| 16 节点 / 64 chip | 16 | **128** |
| 64 节点 / 256 chip | 64 | **512** |

> ⚠️ **绝对不要单独在某一个 pod 里 `import jax`。** 那个进程会抓住 `/dev/vfio/*` 不放，
> 之后所有训练都报 `Device or resource busy; Couldn't open iommu group`，只能重建 pod。
> 一定要用上面的齐步脚本。

### 3.4 注入 tokamax tile（必做，收益最大的一步）

```python
# tkcfg.py —— 在 import train 之前 exec
import os, dataclasses
from tokamax._src.ops.ragged_dot import pallas_mosaic_tpu as P
_TM, _TK, _TN = (int(os.environ[k]) for k in ("TK_TM", "TK_TK", "TK_TN"))
_orig = P.PallasMosaicTpuRaggedDot._get_heuristics_config
def _patched(self, ba):
    c = _orig(self, ba)
    k, n = ba.arguments["rhs"].shape[-2], ba.arguments["rhs"].shape[-1]
    return dataclasses.replace(c, tile_m=_TM, tile_k=min(_TK, k), tile_n=min(_TN, n))
P.PallasMosaicTpuRaggedDot._get_heuristics_config = _patched
print("[tkcfg] patched")
```

`kubectl cp` 到每个 pod 的 `/tmp/tkcfg.py`（**并行 cp**，串行会很慢）：

```bash
for p in $(kubectl get pods -n $NS -l jobset.sigs.k8s.io/jobset-name=$JS \
           --field-selector status.phase=Running --no-headers | awk '{print $1}'); do
  kubectl cp tkcfg.py $NS/$p:/tmp/tkcfg.py -c jax-tpu &
done; wait
```

> ⚠️ **pod 一旦重建，`/tmp/tkcfg.py` 和编译缓存全没了，必须重铺。**

`(512, 2048, 1536)` 的来历：

| 维度 | 最优 | 说明 |
|---|---|---|
| `tile_n` | **1536** | 必须 `= base_moe_mlp_dim`。1024 不整除会 `AssertionError`，512 能整除但切三刀更慢 |
| `tile_k` | **2048** | 甜点。不是抄表的 1024，也不是越大越好（4096 直接 OOM） |
| `tile_m` | **512** | 表里 `m` 落在 1024 档，实测 512 快 3.9%。**抄表是好起点，不是终点** |

### 3.5 跑一轮

```bash
XLA='--xla_tpu_dvfs_p_state=7 --xla_tpu_scoped_vmem_limit_kib=65472 --xla_enable_async_all_gather=true --xla_tpu_enable_sparse_core_reduce_scatter_v2=true --xla_tpu_enable_sparse_core_collective_offload_all_gather=true --xla_tpu_enable_sparse_core_collective_offload_2d_all_gather=true --xla_tpu_enable_sparse_core_collective_offload_all_reduce=true --xla_tpu_enable_sparse_core_collective_offload_reduce_scatter=true --xla_tpu_enable_sparse_core_collective_offload_3d_all_gather=true --xla_tpu_enable_sparse_core_collective_aggregator=true --xla_tpu_use_tc_device_shape_on_sc=True --xla_sc_disable_megacore_partitioning=True --xla_tpu_enable_latency_hiding_layer_scheduler=true --xla_tpu_scheduler_percent_shared_memory_limit=150 --xla_tpu_enable_layer_scheduler_for_dependent_collectives=true --xla_tpu_enable_multi_compute_overlap_in_layer_scheduler=false'

NP=16 bash hy3-run.sh "
export LIBTPU_INIT_ARGS='$XLA' JAX_PLATFORMS=tpu,cpu
export TK_TM=512 TK_TK=2048 TK_TN=1536
cd /deps && python3 -c 'exec(open(\"/tmp/tkcfg.py\").read()); import runpy; runpy.run_module(\"src.maxtext.trainers.pre_train.train\", run_name=\"__main__\")' \
  src/maxtext/configs/base.yml model_name=hunyuan3-295b override_model_config=True \
  ici_fsdp_parallelism=-1 ici_tensor_parallelism=1 \
  per_device_batch_size=12 max_target_length=4096 \
  megablox=True use_tokamax_gmm=True sparse_matmul=True use_custom_sort_vjp=True \
  scan_layers=True remat_policy=custom decoder_layer_input=offload out_proj=remat \
  attention=flash use_tokamax_splash=True sa_use_fused_bwd_kernel=True \
  sa_block_q=2048 sa_block_kv=2048 sa_block_kv_compute=2048 \
  sa_block_q_dkv=2048 sa_block_kv_dkv=2048 sa_block_kv_dkv_compute=2048 \
  sa_block_q_dq=2048 sa_block_kv_dq=2048 \
  opt_type=adamw mu_dtype=bfloat16 grad_dtype=bfloat16 use_iota_embed=True \
  allow_split_physical_axes=True dtype=bfloat16 weight_dtype=float32 \
  tokenizer_type=tiktoken tokenizer_path=src/maxtext/assets/tokenizer_llama3.tiktoken \
  dataset_type=synthetic enable_checkpointing=False steps=8 jax_cache_dir=/tmp/jcn \
  base_output_directory=<GCS_OUT> run_name=myrun
"
```

上面是 **64 芯片 BF16 版**。换另外两组配方时，命令里要动的就下面这些，其余一字不动：

| 换成 | 删掉 | 改成 / 加上 |
|---|---|---|
| **64 芯片 FP8 + QAG** → 674 | `ici_fsdp_parallelism=-1` | `ici_data_parallelism=2 ici_fsdp_parallelism=64`<br>`per_device_batch_size=7`（原 12）<br>**新增**：`use_qwix_quantization=True quantization=fp8_full shard_exp_on_fsdp=True weight_quantization_calibration_method=fixed,-224,224 act_quantization_calibration_method=fixed,-224,224` |
| **256 芯片 BF16** → 599 | `ici_fsdp_parallelism=-1` | `ici_data_parallelism=2 ici_fsdp_parallelism=256`<br>`per_device_batch_size=16`（原 12）<br>`NP=64`（原 16） |

**每轮之间必须清理**，否则下一轮起不来：

```bash
NP=16 bash hy3-run.sh 'pkill -9 -f "pre_train[.]train"; rm -f /tmp/libtpu_lockfile'
```

> 模式串写 `'pre_train[.]train'` 而不是 `'pre_train.train'` —— 后者会匹配到执行这条命令的
> shell 自己的命令行，把自己杀掉。

### 3.6 取数与换算

```bash
grep -oE "completed step: [4-7], seconds: [0-9.]+"        $LOG   # step（稳态取 4–7）
grep -oE "completed step: [4-7].*TFLOP/s/device: [0-9.]+"  $LOG   # per-device
grep -ohE "Total hbm usage >= [0-9.]+G"                    $LOG   # 峰值 HBM
```

**v7 是 2 device / chip**（v5p 是 1:1），框架日志一律按 device 报 —— 这是最容易出错的一步：

```
per-chip TFLOP/s = 日志里的 TFLOP/s/device × 2
MFU              = per-chip ÷ 2307      （FP8 用 4614）
tok/s            = device 数 × per_device_batch_size × max_target_length ÷ step
tok/s/chip       = tok/s ÷ (device 数 / 2)
```

> 方向搞反就是 4 倍误差。同理 `per_device_batch_size` 里的 "device" 在 v7 上是**半个芯片**，
> 跟 v5p 对比时要减半才等价。

**两个开跑就能查的健康检查**（不对说明代码没打全）：

| 日志字段 | 应为 |
|---|---|
| `number parameters` | **298.786 billion**（跨平台恒定） |
| `Total TFLOPs` | seq 4096 下约 **4547**。若是 5 倍左右，说明 FLOP 公式没加 `HUNYUAN3`，MFU 会虚高 |

---

## 四、参数与 flag

### 4.1 完整参数集

**并行**（唯一随规模/精度变的部分，见 [§2](#二直接可抄的配方)）

```
ici_tensor_parallelism=1         # TP 无用，attention 只占 2% 参数
```

**MoE（含最大收益项）**

```
megablox=True
use_tokamax_gmm=True             # ← 配合 tkcfg.py 注入 tile 才有意义
sparse_matmul=True
use_custom_sort_vjp=True
TK_TM=512  TK_TK=2048  TK_TN=1536      # 环境变量
```

**batch 与序列**

```
per_device_batch_size=12         # BF16；FP8+QAG 用 7
max_target_length=4096           # 实测 seq 8192 + pdbs 4 与之等价（451 vs 453），不必换
```

**Attention**

```
attention=flash
use_tokamax_splash=True          # v7 特有
sa_use_fused_bwd_kernel=True     # ⚠️ v5p 上这一项要设 False
sa_block_q=2048  sa_block_kv=2048  sa_block_kv_compute=2048
sa_block_q_dkv=2048  sa_block_kv_dkv=2048  sa_block_kv_dkv_compute=2048
sa_block_q_dq=2048  sa_block_kv_dq=2048
```

**重计算 / offload / 优化器**

```
scan_layers=True                 # 也让编译时间不随层数涨
remat_policy=custom
decoder_layer_input=offload
out_proj=remat

opt_type=adamw
mu_dtype=bfloat16                # Adam 一阶动量降 bf16
grad_dtype=bfloat16
use_iota_embed=True
```

优化器状态从 16 B/param 降到 12 B/param。`nu_dtype` optax 不支持单独设，恒随 `weight_dtype`；**主权重仍是 fp32**。

**精度与基线条件**

```
dtype=bfloat16  weight_dtype=float32  allow_split_physical_axes=True
tokenizer_type=tiktoken
tokenizer_path=src/maxtext/assets/tokenizer_llama3.tiktoken

dataset_type=synthetic           # 只测吞吐不测收敛
enable_checkpointing=False       # 避免 I/O 干扰读数
steps=8                          # 取 step 4–7 稳态
```

### 4.2 XLA flag（16 个）

```
# 频率（1）—— 值 +8.6%
--xla_tpu_dvfs_p_state=7

# 基础（2）
--xla_tpu_scoped_vmem_limit_kib=65472
--xla_enable_async_all_gather=true

# SparseCore 卸载组（9）—— v7 上收益 ±0，保留只为与官方配方对齐
--xla_tpu_enable_sparse_core_reduce_scatter_v2=true
--xla_tpu_enable_sparse_core_collective_offload_all_gather=true
--xla_tpu_enable_sparse_core_collective_offload_2d_all_gather=true
--xla_tpu_enable_sparse_core_collective_offload_all_reduce=true
--xla_tpu_enable_sparse_core_collective_offload_reduce_scatter=true
--xla_tpu_enable_sparse_core_collective_offload_3d_all_gather=true
--xla_tpu_enable_sparse_core_collective_aggregator=true      # ← 这个不能删
--xla_tpu_use_tc_device_shape_on_sc=True
--xla_sc_disable_megacore_partitioning=True

# 调度器组（4）—— 值 +6.6%
--xla_tpu_enable_latency_hiding_layer_scheduler=true
--xla_tpu_scheduler_percent_shared_memory_limit=150
--xla_tpu_enable_layer_scheduler_for_dependent_collectives=true
--xla_tpu_enable_multi_compute_overlap_in_layer_scheduler=false
```

> SparseCore 那 9 个里有 8 个确实零收益，但第 9 个（`collective_aggregator`）是层调度器的硬依赖，
> 删了直接 `INVALID_ARGUMENT: Latency hiding layer scheduler requires sparse core collective aggregator`。
> **裁剪 flag 要成组，不能逐个删。**
>
> ⚠️ libtpu 对不认识的 flag 是**硬失败**（`Unknown command line flag`，进程直接退）。**换镜像必须重过一遍 flag 集。**

---

## 五、模型与代码

**一句话：attention 是 Qwen3 的，MoE 是 DeepSeek V3 的。** MaxText 里这两半都已有现成实现，
本项目只写了装配逻辑，零新数学。

| | |
|---|---|
| 结构 | 80 层，第 0 层 dense、1–79 层 MoE |
| Attention | GQA 64q / 8kv，head_dim 128，QK-LayerNorm，无 bias |
| MoE | **192** routed experts top-8 + 1 shared，sigmoid 路由 + 专家偏置 |
| 其他 | MTP 1 层，vocab 120832，routed scaling 2.826 |
| 参数分布 | **97% 在路由专家里**，attention 只占 2% |

参数分布直接决定并行策略：**TP 无用**、**EP 是负优化**、**FSDP 是主力**。

> ⚠️ **192 不是 2 的幂，这个数字会反复咬你。** 选并行度时先检查整除 ——
> `shard_exp_on_fsdp` 在 FSDP=128 上直接 `IndivisibleError`（192 % 128 ≠ 0），只能取 64。

**代码唯一真相是 fork 的分支**，本仓不留副本：`https://github.com/yangwhale/maxtext` 分支 `hunyuan3`。
基于上游 main 三个 commit：模型本体 + 注册、上游 loss-free-balancing bias 路径修复、SwiGLU 激活截断白名单。
新增 3 个文件（`models/hunyuan3.py` 161 行 + 两个 yml），改动上游 12 个文件全是「让框架认识这个模型」，
**没有一处算法实现**。v5p 和 v7 用同一份代码、同一个镜像。

---

## 六、已知限制

| 项 | 状态 |
|---|---|
| 数据集 | `synthetic`。**loss 下降只证明「能算且不发散」，不是收敛证据** |
| 完整 loss 曲线 | 未记。建议补一条 30 步以上的 |
| HF 权重 → Orbax 转换 | 未做。只跑吞吐可以不碰；要 SFT 必须做 |
| BF16 调参空间 | **2026-08-15 被顶穿**：换 tile 入口后到 662（`pdbs=13` 时 666.6），超出原定 600–630 区间。tile / batch 两条线已再次见底 |
| FP8 调参空间 | **2026-08-15 被顶穿**：换 tile 入口后 674 → **1,014.8（+50%）**。此前「已见底」的结论成立于 tokamax 注入那条路，换条路就不成立了。batch 仍卡在 7（AOT 实测 8 要 98.05 G） |
| 256 芯片 + `dvfs=7` | 未复测。64 芯片上 +8.6%，预期等幅但没验 |
| 容量 | tpu7x 抢手，全球仅 4 个 zone 有机型 |

---

<details>
<summary><b>从零搭环境（自建集群才需要）</b></summary>

用别人已建好的托管 / Kueue 集群的话，整节跳过 —— 那种集群不让你自己建 workload policy 和节点池。

### workload policy（v7 特有，必须先建）

v5p 建节点池一条命令就完了，tpu7x 会直接被拒：

```
Creation of a managed instance group with tpu7x-standard-4t machine type
with placement policy is not supported. Use workload policy instead.
```

```bash
P=YOUR-PROJECT
for TOPO in 4x4x4 4x8x8 2x2x1; do
  gcloud compute resource-policies create workload-policy wp-$TOPO \
    --project=$P --region=us-central1 \
    --type=HIGH_THROUGHPUT --accelerator-topology=$TOPO
done
```

- `--accelerator-topology` **必须带**。只给 `--type` 会报
  `does not support TPU topology with group placement policy and workload policy at the same time`
- **一个 policy 对应一个拓扑**，用几种拓扑就建几个

gcloud 缺这个子命令时的 REST 写法：

```bash
TOK=$(gcloud auth application-default print-access-token)
curl -s -X POST -H "Authorization: Bearer $TOK" -H "Content-Type: application/json" \
 "https://compute.googleapis.com/compute/v1/projects/$P/regions/us-central1/resourcePolicies" \
 -d "{\"name\":\"wp-$TOPO\",\"workloadPolicy\":{\"type\":\"HIGH_THROUGHPUT\",\"acceleratorTopology\":\"$TOPO\"}}"
```

### TPU 节点池

```bash
# 64 芯片（16 台）—— 拓扑 4x4x4
gcloud container node-pools create np-v7x-64 \
  --cluster=CLUSTER --project=$P --region=us-central1 --node-locations=us-central1-c \
  --machine-type=tpu7x-standard-4t --tpu-topology=4x4x4 \
  --placement-policy=wp-4x4x4 --num-nodes=16 \
  --disk-type=hyperdisk-balanced --disk-size=200 --scopes=cloud-platform

# 256 芯片（64 台）：--tpu-topology=4x8x8 --placement-policy=wp-4x8x8 --num-nodes=64
```

四个跟 v5p 不一样的地方：

| | 说明 |
|---|---|
| `--placement-policy` | 指向上面建的那个，**拓扑要对得上** |
| `--disk-type=hyperdisk-balanced` | **v7 不接受普通 pd** |
| zone | v7 在 **`us-central1-c`**，v5p 在 `-a` |
| `--num-nodes` | = 芯片数 ÷ 4，且必须与 `--tpu-topology` 相乘一致 |

**想长时间稳定占用（不被抢占）用 DWS flex-start**，三个参数绑死：
`--flex-start` + `--num-nodes=0` + `--enable-autoscaling --min-nodes=0 --max-nodes=N`，
少一个各报各的错。上限用 `--max-nodes`，写成 `--total-max-nodes` 会被判成 0。排队按 **20 小时**预期，不是分钟级。

> ⚠️ **不要顺手加 `--enable-queued-provisioning`。** gcloud 原文：
> *"all new nodes can be obtained **only** through queuing via ProvisioningRequest API"* ——
> 加了之后普通 Job / Deployment 就再也拉不起节点，必须装 Kueue 提 `ProvisioningRequest`。
> 想让 `sleep infinity` 直接触发扩容，就只用 `--flex-start`。

### JobSet CRD

```bash
kubectl apply --server-side -f \
  https://github.com/kubernetes-sigs/jobset/releases/download/v0.11.1/manifests.yaml
kubectl wait --for=condition=Available deploy/jobset-controller-manager \
  -n jobset-system --timeout=180s
```

v0.11.1 自带证书，**不需要 cert-manager**。

### 暂存桶 + 跨项目授权

```bash
gcloud storage buckets create gs://YOUR-STAGE-BUCKET --location=US

NODE_SA=<集群项目号>-compute@developer.gserviceaccount.com
gcloud storage buckets add-iam-policy-binding gs://YOUR-STAGE-BUCKET \
  --member="serviceAccount:$NODE_SA" --role=roles/storage.objectViewer

# 镜像在别的项目时，节点 SA 还要能拉
gcloud artifacts repositories add-iam-policy-binding gcr.io --location=us \
  --project=IMAGE_PROJECT --member="serviceAccount:$NODE_SA" \
  --role=roles/artifactregistry.reader
```

### 网络

共享项目的 default VPC 是 auto 模式，`10.128.0.0/9` 被各 region 子网占满，GKE 凑不出一整块 `/14` 给 pod。
**建自己的 custom VPC 最省事**，顺带能把 MTU 开到 8896：

```bash
gcloud compute networks create NAME-vpc --subnet-mode=custom --mtu=8896
gcloud compute networks subnets create NAME-uc1 --network=NAME-vpc \
  --region=us-central1 --range=10.124.0.0/22 \
  --secondary-range=pods=10.125.0.0/16,services=10.124.16.0/20 \
  --enable-private-ip-google-access
```

### 建池之前先看一眼容量

**配额决定你能不能申请，容量决定你能不能拿到，这是两个独立的闸门。**
池子停在 `PROVISIONING` 且 MIG 没有 error，那就是纯粹排不到机器，**换项目、提配额都无效**。

```bash
gcloud compute instances list --project=ANY-PROJECT-IN-SAME-ZONE \
  --filter="machineType~tpu7x AND status=RUNNING" \
  --format='value(zone,scheduling.provisioningModel)' | sort | uniq -c
```

> `4x4x4`（64 芯片）是**原子切片**，没有 48 / 56 这种中间档 —— **凑不满等于拿不到**。

</details>

<details>
<summary><b>踩坑速查</b></summary>

### 保命五条

1. **先确认 pod 全 Running 再看日志。** TPU 切片全有全无，人不齐时活着的 pod 会报
   `GetSliceInfo can only be invoked after a slice is built` —— 那是症状不是病因。
2. **判错看最早那条，不是日志尾。** 配置非法会先把 TPU 拉起来再退，真正的报错
   （`MAXTEXT CONFIG ERROR` / pydantic `Value error`）在日志上方。
3. **step 0 含编译，step 1/2 是 JAX 异步派发的假读数**，稳态取 step ≥ 3。
4. **日志必须落盘。** pod 一旦被删、被抢占、或被集群时间上限杀掉，`kubectl logs` 就再也读不到。
5. 🔴 **撞到 `SLICE_FAILURE_*` 立刻整体中止，绝不重试。** 这是硬件故障，每次重启都会在同一层崩，
   耗尽 `maxRestarts` 后 JobSet 进 `Failed`，**没有消费方了 autoscaler 会把节点缩回 0，卡直接没了**。
   判据：`completed step: 0` 出现过、峰值 HBM 远低于上限、**每次崩的 worker 不同**。
   所以 `maxRestarts` 设 3，不要设 10。

```bash
grep -q "SLICE_FAILURE" $LOG && { echo "TPU 硬件故障，换一批节点"; exit 2; }
```

### 共享集群上作业被抢占

**症状**：`kubectl exec` 的进程突然 `exit code 137`，但 pod 还在 Running；日志停在编译刚结束处；
没有任何 OOM 字样。

**别急着归因为内存不足。** 查这两处：

```bash
kubectl get jobset <NAME> -n <NS> -o jsonpath='{.status.conditions}'   # 找 "jobset is resumed"
kubectl get events -n <NS> --sort-by=.lastTimestamp | grep -iE "preempt|FailedScheduling"
```

看到 `jobset is resumed` 就说明它刚被 suspend 过一轮 —— **Kueue 抢占，pod 被删重建，
你 exec 进去的进程跟着被 SIGKILL**。附带后果：`/tmp` 下的 `tkcfg.py` 和编译缓存全没了，必须重铺。

**`priorityClassName: medium`（prio 500）挡不住。** 不写这个字段则是 prio 0，任何人都能抢。

### `exclusive-topology` 注解会让 autoscaler 少看见需求

它要求 leader pod 先落地、follower 才能抄它的 `gke-nodepool` 选择器。没有空节点时 leader 落不了地，
现象是 `parallelism=16` 但 `status.active=1`，**没有任何报错**。
绕法：直接在 `nodeSelector` 写死 `gke-nodepool`，或去掉该注解。

### 托管（Kueue）集群上扩不出节点

**`kubectl` 层面看不到真正原因。** pod 事件只会给你 `Pod didn't trigger scale-up: ... in backoff after failed scale-up`。
真原因在 MIG 里：

```bash
gcloud container clusters describe <CLUSTER> --region <REGION> \
  --format='value(nodePools[].instanceGroupUrls)' | tr ';' '\n' | grep tpu
gcloud compute instance-groups managed list-errors <MIG> --zone <ZONE> \
  --format='value(error.code,error.message)'
```

一次真实输出：`QUOTA_EXCEEDED  Quota 'HDB_TOTAL_GB' exceeded. Limit: 40960.0` —— **区域磁盘配额打满**。
TPU 节点启动盘是 hyperdisk-balanced（100 GB/台），跟别人 PVC 抢同一个区域配额；
别人留下的 `Released` PV 会一直占着。

排查顺序（**直接看 pod 会误判**）：

| 步骤 | 看什么 | 卡在这说明 |
|---|---|---|
| ① | `kubectl get clusterqueue` / `get workload` | 配额记账没过 |
| ② | MIG `list-errors` | **配额/容量，扩不出机器** |
| ③ | `kubectl get nodes` | 机器建出来了但没注册 |
| ④ | `kubectl get pods` | 常规问题：镜像、配置 |

> **被 admit ≠ 有机器。** 队列放行只代表记账通过。反过来也成立：**账面满 ≠ 机器忙**
> —— 见过记账 364/375 已占、实际只跑 60 芯片的情况。

### 其他单点

| 坑 | 后果 | 处理 |
|---|---|---|
| 单独在一个 pod 里 `import jax` | 占住 `/dev/vfio/*`，之后全报 `Device or resource busy` | 只能重建 pod；一律用齐步脚本 |
| `pkill -f "pre_train.train"` | 匹配到执行它的 shell 自己，把自己杀掉 | 写成 `'pre_train[.]train'` |
| 删掉 `collective_aggregator` | `Latency hiding layer scheduler requires sparse core collective aggregator` | flag 要成组裁剪 |
| 换镜像沿用旧 flag | `Unknown command line flag`，进程直接退 | 换镜像必须重过 flag 集 |
| 只开 `shard_exp_on_fsdp` | **静默失效**，不报错也不变快 | calibration 必须写 `fixed,-224,224` |
| `tile_n` 用 1024 | `AssertionError: v=1536 bv=1024 s=1536` | 必须 `= base_moe_mlp_dim` = 1536 |
| `quantization=fp8` | `AttributeError: Fp8Quantization 无 quant_dg`（NVIDIA 专用类） | TPU 正路是 `fp8_full` + qwix |

</details>

<details>
<summary><b>深入：数字背后的原理</b></summary>

### 硬件

| | 64 芯片 | 256 芯片 |
|---|---|---|
| 节点 | 16 台 `tpu7x-standard-4t` | 64 台 |
| 拓扑 | `4x4x4` | `4x8x8` |
| JAX device | **128** | **512** |
| HBM | 192 GB / chip；**单 device 可用 94.74 GB** | 同 |
| BF16 峰值 / chip | **2,307 TFLOPS**（FP8 是 4,614） | 同 |
| 总算力 BF16 | 147.6 PFLOPS | 590.6 PFLOPS |

### 64 芯片和 256 芯片跑出同一个数，不是巧合

**580 vs 580，MFU 25.14% vs 25.12%，峰值 HBM 91.94 G vs 91.94 G —— 一字节不差。**

`DP=4 × FSDP=128` 就是**四个独立的 64 芯片作业**：组内做 FSDP 集合通信，组间只在每步末尾同步一次梯度。
每卡的分片形状、tile 匹配、激活占用完全相同，所以 per-chip 性能必然相同。

⇒ **不必为了性能去抢大 slice，16 节点就能拿到全部收益，调优也可以在 16 节点上做**
（只要改的开关不改变分片形状）。

### 扩展性：weak scaling 100%，strong scaling 掉 11%

| 扩展方式 | 512 device 怎么切 | 每卡工作量 | global batch | per-chip | 相对 64 芯片 |
|---|---|---|---|---|---|
| **Weak**（加卡也加 batch） | `DP=4 × FSDP=128` | 不变（pdbs 12） | **4×** | **580** | **100.0%** |
| **Strong**（加卡不加 batch） | `DP=1 × FSDP=512` | 缩到 1/4 | 1× | 404 | 89% |

**为什么 DP 方向能到 100%** —— 组间那次 all-reduce 便宜到可以忽略：

```
Hy3 梯度（bf16）           ≈ 590 GB
每卡梯度分片（FSDP=128）    = 590 / 128 ≈ 4.6 GB
ring all-reduce 传输量      = 2(p−1)/p × 4.6 GB = 6.9 GB   （p = DP = 4）
v7 ICI 单芯片双向带宽       = 1,200 GB/s
                            ──────────────────────────────
理论耗时                    ≈ 12 ms   ← 占 step 23.54 s 的 0.05%
```

即使按 1/6 带宽利用率保守估计（35 ms）也只有 0.15%，完全淹没在 ±3% 噪声里，而且还能跟反向传播尾部重叠。
对比之下**组内 FSDP 每层两次、80 层就是 160 次** —— 量级差两个数量级。这就是「DP 便宜、FSDP 贵」的根本原因。

**为什么 strong scaling 掉 11%**：`FSDP=512` 把同一份权重摊到 4 倍卡上，每卡分片缩到 1/4。
集合通信**次数没变，每次的有效载荷只有 1/4** —— 固定开销（同步、启动延迟、torus 多跳转发）摊不动了。

⇒ **加卡的时候要同时加 batch。**

三条边界（别过度外推）：只测到 DP=4；这是单 slice 内的结论（**跨 slice 走 DCN，带宽低一个数量级以上**）；
前提是每卡工作量不变。

### 显存怎么算：两参数模型（有失效区间）

用同基座两个实测点解 `HBM = 静态 + 斜率 × pdbs`：

```
DP4×FSDP128:  74.20 G @ pdbs 8 ，91.93 G @ pdbs 12
              → 静态 38.7 G ，斜率 4.43 G / pdbs
DP2×FSDP256:  静态 25.9 G（FSDP 翻倍，静态减半），斜率相同
```

| 基座 | pdbs 8 | pdbs 10 | pdbs 12 | pdbs 14 | pdbs 16 |
|---|---|---|---|---|---|
| `DP4×FSDP128` 预测 | 74.2 | 84.1 | 91.9 | 100.8 | 109.6 |
| `DP4×FSDP128` **实测** | **74.20** ✅ | **84.06** ✅ | **91.94** ✅ | — | **OOM** ✅ |
| `DP2×FSDP256` 预测 | 61.4 | 73.5 | 79.1 | 87.9 | 96.8 → 判 OOM |
| `DP2×FSDP256` **实测** | **61.36** ✅ | — | **78.27** ✅ | **89.56** ✅ | **92.33** ❌ **预测错，实际跑通** |

> ⚠️ **线性模型在 pdbs ≥ 14 之后失效。** 实测 `DP2×FSDP256` 的逐段斜率：
> `8→12` 是 4.23 G/pdbs，`12→14` 是 5.65，`14→16` 骤降到 1.39 —— 高 batch 区间激活增长明显**次线性**
> （大概率是 XLA 在显存压力下改变了 remat / offload 调度），线性外推会**系统性高估**。
>
> 用法：只在已测区间 ±2 个 pdbs 内插值；预测 OOM 的配置**仍然值得实跑一次**
> （两次判 OOM 都错了）；只有预测「远超上限」（如 109.6 G，超 15%）才可以直接排除。

### 缩层冒烟（改代码后先跑这个）

```bash
NODES=1 TOPO=2x2x1 MODEL=hunyuan3-smoke STEPS=8 \
  bash run.sh smoke per_device_batch_size=1 max_target_length=2048
```

4 层缩层，结构与 295B 完全一致（192 专家、top-8、sigmoid、专家偏置、共享专家、GQA、QK-norm、
fp32 路由、MTP 全是满配），只砍层数。

| | v7 实测 | v5p 同命令 |
|---|---|---|
| 参数量 | **16.139 B** | 16.139 B（**必须一致**） |
| `total_weights` | 16384 | 8192 |
| loss（8 步） | 13.411 → 11.091 | 13.453 → 10.354 |
| NaN / skipped | 0 | 0 |

> 两个平台 loss 序列不同**是对的**：同样 4 芯片，v7 有 8 个 device 而 v5p 只有 4 个，
> `pdbs=1` 之下 global batch 差一倍。**跨平台恒定、可当硬标准的只有参数量 16.139 B。**

**为什么 4 层能代表 80 层**：MaxText 按**类型**分组做 `scan`，79 个 MoE 层共用同一份编译产物，
层与层的差别只在权重数值上。冒烟测的是**那个被复用 79 次的唯一函数**。
它覆盖不到的：显存压力、大规模切分、完整 XLA flag 集、收敛质量、以及全部性能。

</details>

<details>
<summary><b>其他规模的实测数据</b></summary>

### 256 芯片：从 453 到 599 每一步值多少

同批次测得（同一组常驻 pod，编译缓存复用），**未叠 `dvfs=7`**：

| 步骤 | 增量 | chip | MFU | tok/s | 峰值 HBM | 累计 |
|---|---|---|---|---|---|---|
| 起点 | `DP4×FSDP128` / pdbs 8 / megablox | 453 | 19.64% | 835,131 | 74.20 G | — |
| **+tile** | tokamax `tile(512, 2048, 1536)` | **532** | 23.04% | 979,893 | 75.33 G | **+17.4%** |
| **+batch** | pdbs 8 → 10 | **564** | 24.45% | 1,039,573 | 84.06 G | **+24.5%** |
| **+batch** | pdbs 10 → 12 | **580** | 25.12% | 1,068,372 | 91.94 G | **+28.0%** |
| +换基座 | `DP2×FSDP256` 省 13 G，pdbs → 14 | 585 | 25.37% | 1,078,802 | 89.56 G | +29.1% |
| **+再加 batch** | 同基座 pdbs → 16 | **599** | **25.96%** | **1,103,757** | 92.33 G | **+32.2%** |

### 64 芯片 BF16 逐点（对照 256 芯片同配方）

| 配置 | step | chip | MFU | tok/s/chip | 峰值 HBM | 256 芯片同配置 |
|---|---|---|---|---|---|---|
| megablox / pdbs 8 | 19.92 s | 457 | 19.80% | 3,290 | 74.20 G | 453 |
| tile + pdbs 8 | 16.75 s | 543 | 23.55% | 3,912 | 75.33 G | 532 |
| tile + pdbs 10 | 20.28 s | 561 | 24.32% | 4,040 | **84.06 G** | 564（HBM 84.06 G） |
| **tile + pdbs 12** | **23.54 s** | **580** | **25.14%** | **4,176** | **91.94 G** | **580**（HBM 91.94 G） |
| tile + pdbs 12 + **dvfs 7** | **21.67 s** | **630** | **27.31%** | **4,536** | 91.94 G | 未复测 |
| **18 个配置参数 + pdbs 12 + dvfs 7** 🏆 | **20.61 s** | **662.3** | **28.71%** | **4,770** | 91.94 G | 未测 |
| 18 个配置参数 + pdbs 13 + dvfs 7 | 22.19 s | **666.6** | **28.89%** | 4,799 | 92.57 G | 未测 |

> 两个规模在同配方下逐点吻合，pdbs 10 和 12 两处峰值 HBM 一字节不差（84.06 / 91.94 G）。
> 这是「DP 层不改变单步计算图」最硬的证据。

### FP8 全部实测

| 规模 | 配方 | step | chip | MFU<sub>FP8 4614</sub> | tok/s/chip | 峰值 HBM |
|---|---|---|---|---|---|---|
| 256 chip | 无 QAG，`DP2×FSDP256` pdbs 16 | 29.46 s | 618 | 13.39% | 4,449 | 92.80 G |
| 64 chip | 无 QAG，`DP1×FSDP128` pdbs 10 | 19.15 s | 594 | 12.87% | 4,281 | 86.20 G |
| 64 chip | **+QAG**，`DP2×FSDP64` pdbs 7 | 12.76 s | 624 | 13.53% | 4,495 | 92.42 G |
| **64 chip** | **+QAG +dvfs 7**（tokamax 注入） | **11.81 s** | 674 | 14.61% | 4,854 | 92.42 G |
| 64 chip | 同上，2026-08-15 复现 | 11.76 s | 677.0 | 14.67% | 4,876 | 92.42 G |
| **64 chip** 🏆 | **+QAG +dvfs 7 + 18 个 tile 配置参数** | **7.85 s** | **1,014.8** | **21.99%** | **7,308** | **91.40 G** |

> **FP8 的 MFU 分母是 4614，不是 BF16 的 2307** —— 别拿它跟 BF16 那几行直接比 MFU 大小，
> 要比就比 **tok/s/chip**。DSV3 官方同口径是 743.5 / 16.1%。
>
> 256 experts 的探索（645 / `DP1×FSDP128` + QAG + pdbs 11）改了模型，与上表不可横比，见 TUNING-v7。

</details>

<details>
<summary><b>验证记录</b></summary>

**这份文档的每个数字都是照它自己的步骤跑出来的。** 记录每一轮审计的偏差 ——
包括被改掉的错误，因为「文档哪里会坑人」比「文档说了什么」更值钱。

### 轮 1（2026-08-04）：两个规模同时复测

**方法**：不销毁 pod，`hy3-run.sh` 和 `tkcfg.py` 用脚本从本文档 markdown 里**原样抠出**，
训练命令按 §3.5 拼，**不使用任何历史脚本**。

| 规模 | 配方 | 文档写的 | 实测 | 偏差 |
|---|---|---|---|---|
| 64 chip | `DP1×FSDP128` + tile + pdbs 12 | step 23.54 / **580** | step **23.538** / **579.97** | 0.005% |
| 256 chip | `DP2×FSDP256` + tile + pdbs 16 | step 30.40 / **599** | step **30.399** / **598.74** | 0.04% |

**抓到 4 个缺陷**：①§0 与 §3.4 对 64 卡 pdbs 的说法自相矛盾（旧版残留）；
②删 JobSet 想从零重建，30 秒内整池被抢走，审计中断；③`hy3-run.sh` 跑完前日志 0 字节，
无法区分编译与卡死；④冒烟只给了 64 节点的预期值。**全部已修**，②转为规矩：审计不重跑抢卡步骤。

### 轮 2（2026-08-11）：DVFS 与 FP8 叠加

同一常驻 pod 内做 A/B，只改一个 flag。BF16 四档对照 + FP8 两组对照，
**不带 dvfs 的 FP8 那轮跑出 624.1，与三天前记录的 625.4 差 −0.20%** ——
隔了三天、换了 pod、换了节点，说明配方与环境可复现，对照组可信。

</details>

---

## 附录 A：与 v5p 的差异速查

两个平台**共用同一份代码、同一个镜像**，以下是全部差异：

| | v7 | v5p |
|---|---|---|
| 机型 / 拓扑 | `tpu7x-standard-4t`，`4x4x4` / `4x8x8`，us-central1-**c** | `ct5p-hightpu-4t`，`4x8x8`，us-central1-**a** |
| 建池前置 | **必须先建 workload policy** | 无 |
| 磁盘 | **hyperdisk-balanced** | 默认 |
| **device : chip** | **2 : 1** | 1 : 1 |
| MFU 分母 | **2,307** | 459 |
| `max_target_length` | **4096** | 8192 |
| `sa_use_fused_bwd_kernel` | **True** | **False** |
| `use_tokamax_splash` | **True** | 不设 |
| `use_tokamax_gmm` + tile 注入 | ~~True，+17.4%~~ **已弃用**，改走配置参数 | 负收益，不用 |
| `xla_tpu_dvfs_p_state=7` | **True，+8.6%** | 未测（v5p 上默认档未知） |
| `opt_type` / `mu_dtype` / `grad_dtype` | **adamw / bf16 / bf16** | 默认 / fp32 / fp32 |
| MoE tile 参数（18 个） | **全设 (512,2048,1536)** —— 2026-08-15 改，比注入路径快 5.1% | 全设 (512,1024,1024) |
| XLA flag | 16 个 | 25 个 |
| SparseCore 卸载组收益 | **±0** | **+4.07 pp** |
| **最高水位** | **1,014.8**（FP8+QAG+dvfs+18 个 tile 参数）／ **666.6**（BF16 同左） | 161.0（已收敛，MFU 35.07%） |

> **同一个开关在两个平台上可以反号。** `sa_use_fused_bwd_kernel`、SparseCore 卸载组、
> `use_tokamax_gmm` 三处都是。**别把一个平台的调优结论直接搬到另一个。**

## 附录 B：延伸阅读

| 文档 | 内容 |
|---|---|
| [TUNING-v7.md](TUNING-v7.md) | 调优实践：为什么是这个水位、全部消融表、失败项 |
| [QUICKSTART-v5p.md](QUICKSTART-v5p.md) | v5p 版，含架构完整拆解；基线 MFU 35.07% |
| [EXPERIMENT-LOG.md](EXPERIMENT-LOG.md) | 完整实验档案：全部轮次、12 个 bug 复盘、DVFS 时钟域分析 |
| [MAXTEXT-PORTING-GUIDE.md](MAXTEXT-PORTING-GUIDE.md) | 把别的模型移植到 MaxText 的通用范式 |
| [maxtext-hunyuan3/](maxtext-hunyuan3/) | `prep.sh` / `run.sh` |
