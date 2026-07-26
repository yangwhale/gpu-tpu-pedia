# vLLM · DeepSeek-V4-Pro-DSpark · GB300 NVL72 复现 Runbook

> **本文定位**：**只讲怎么跑通**的操作手册。所有命令可直接复制，引用的脚本/manifest 都已入库（`manifests/` + `scripts/`）。
> 探索过程、被推翻的假设、公开榜单对标 → 见 `vllm-v4-gb300-benchmark.md`（旧版，按时间顺序，信息全但需自行导航）。
> SGLang 侧对照 → [`SGLANG-V4PRO-RUNBOOK.md`](./SGLANG-V4PRO-RUNBOOK.md)。
>
> **文档状态约定**：每节标注 `[已验证]` / `[待验证]` / `[已知问题]`。标 `[已验证]` 的都在 §10 有实跑记录。

---

## ⚠️ 最重要的一条：镜像用错会「跑得通但性能腰斩」

**这是 vLLM 侧复现失败的头号原因，而且不报错、不告警、生成结果还完全正确。**

| 镜像 | 能跑吗 | 性能 |
|---|---|---|
| `…/gb300-images/vllm-openai-deepgemm:v0.25.1-sm100-aarch64` | ✅ | **正确**（4k1k 24,358 tps）|
| `vllm/vllm-openai:v0.25.1-aarch64`（通用）| ✅ 生成正确 | ❌ **腰斩** —— `--moe-backend deep_gemm_mega_moe` **静默 fallback 到慢 kernel** |

**唯一可靠的判据是启动日志里这三行**（认这个，别认镜像 tag）：

```
expert_dtype resolved to 'fp4'
Selected DeepGemmFp8BlockScaledMMKernel
DeepGEMM PDL/E8M0 enabled
```

三行齐全 = FP4/DeepGEMM 优化 kernel 已激活。缺任何一行 = 你在测一个假的配置。

> **镜像来源已核实**：走 vLLM 官方 buildkite `release-v2` CI（build 3803）编译，基于 vllm-project/vllm commit `752a3a5`（2026-07-12），DeepSeek-V4 + DSpark 模型代码在 `vllm/models/deepseek_v4/nvidia/`（NVIDIA 贡献），内置 `deep_gemm 2.5.0`。**不是个人自制 fork**，厂商只是转存到私有 registry。

---

## 0. TL;DR

| 项 | 值 |
|---|---|
| 基线配置 | **1 prefill(TP4) + 1 decode(TP4)**，8 GPU / 2 节点，单 NVL72 subblock |
| 扩展配置 | **N prefill(TP4) + 1 decode(dep8)**，decode = TP1 + DP8-attention + EP8 跨 2 节点 |
| 模型 | `DeepSeek-V4-Pro-DSpark`（FP8，~832GB / 66 shards），节点本地 SSD |
| 镜像 | `us-central1-docker.pkg.dev/tencent-gcp-taiji-poc/gb300-images/vllm-openai-deepgemm:v0.25.1-sm100-aarch64`（见文首告警）|
| KV 传输 | **NixlConnector + NVLink cuda_ipc**（不是 RDMA/dmabuf/peermem）|
| 编排 | **`vllm-router`**（不是 Dynamo；SGLang 那边才用 Dynamo）|
| MoE backend | `deep_gemm_mega_moe`（对应 SGLang 的 `megamoe`）|
| 投机解码 | DSpark，`num_speculative_tokens=7` |
| 参考成绩 | **4k1k 1p1d = 24,358 total tok/s = 厂商基线 22,000 的 111%**；2p1d = 31,499 |

**与 SGLang 的关键架构差异**（同一台机器、同一个模型）：

| | vLLM | SGLang |
|---|---|---|
| KV 传输 | NixlConnector + UCX cuda_ipc | mooncake |
| 编排 | vllm-router | Dynamo（`dynamo.sglang` + `dynamo.frontend`）|
| MoE backend | `deep_gemm_mega_moe` | `megamoe` |
| prefill 并行 | **只能 TP**（PP 未实现、DP 会 OOM）| TP / PP 都行 |
| AFD（attn-FFN 分离）| ❌ 无原生支持（[RFC #22799](https://github.com/vllm-project/vllm/issues/22799) 仍 open）| ✅ |

---

## 1. 前置条件 `[待验证]`

```bash
# ① 同 NVL72 subblock 的节点（KV 走域内 NVLink，跨域不行）
kubectl get nodes -l cloud.google.com/gke-nodepool=gb300-pool-0002,team=yangwhale --no-headers | wc -l

# ② RAID 挂载（⚠️ 先查这个，见 SGLANG runbook §3.1 的 md0→md127 陷阱）
#    正常 12T；若 256K 说明 RAID 没挂，模型放不进去
kubectl get pods -l app=vllm --no-headers -o custom-columns=N:.metadata.name | while read p; do
  echo -n "$p: "; kubectl exec $p -- df -h /mnt/ssd | tail -1 | awk '{print $2, $5}'
done

# ③ 能拉 deepgemm 镜像（imagePullSecrets）
kubectl get secret ar-pull-secret >/dev/null && echo "pull secret OK"
```

**同 subblock 是硬要求**：KV 走 UCX `cuda_ipc` + MNNVL，只在同一 NVLink 域内成立。manifest 里用 `gce-topology-subblock` 的 podAffinity 保证。

---

## 2. 部署 pod fleet `[待验证]`

```bash
kubectl apply -f manifests/vllm-fleet.yaml
kubectl get pods -l app=vllm -w        # 等全部 Running
```

**与 SGLang fleet 的差异**：

1. 镜像换成 deepgemm 专用
2. **必须加 `cloud.google.com/gce-topology-subblock` podAffinity** —— cuda_ipc 只在同 NVLink 域内工作（旧文档只提了要同 subblock，没给标签全名和 manifest 写法）
3. **core dump / tmp 要重定向到 `/mnt/ssd`**（见 §9 崩溃根因 A）

---

## 3. 模型 `[待验证]`

用的是 **`DeepSeek-V4-Pro-DSpark`**（FP8，~832GB，66 shards），**不是** SGLang 那份 `DeepSeek-V4-Pro`（806G / 64 shards）。DSpark 版本自带投机解码用的 draft 头。

```bash
for p in $(kubectl get pods -l app=vllm -o name | sed 's|pod/||'); do
  echo -n "$p: "
  kubectl exec $p -- bash -c "du -sh /mnt/ssd/DeepSeek-V4-Pro-DSpark 2>/dev/null|cut -f1; \
    ls /mnt/ssd/DeepSeek-V4-Pro-DSpark/*.safetensors 2>/dev/null|wc -l" | tr '\n' ' '; echo
done
```

**从 GCS 补**（⚠️ 有 auth 坑）：GKE 节点的 compute SA 对模型 bucket **OAuth scope 未授权**。必须：

```bash
# 本机取 token → 拷进 pod → 用 CLOUDSDK_AUTH_ACCESS_TOKEN 跑 gcloud
gcloud auth application-default print-access-token > /tmp/tok
kubectl cp /tmp/tok <pod>:/tmp/tok
kubectl exec <pod> -- bash -c "CLOUDSDK_AUTH_ACCESS_TOKEN=\$(cat /tmp/tok) \
  gcloud storage cp -r gs://chrisya-gb300-models/DeepSeek-V4-Pro-DSpark /mnt/ssd/ \
  --billing-project=<project>; rm /tmp/tok"
```

> `gcloud auth login --cred-file` **不吃** authorized_user ADC，只有 `CLOUDSDK_AUTH_ACCESS_TOKEN` 能让 gcloud CLI 用上。用完删 token。
> 也可以走 pod→pod 直传（更快，见 SGLANG runbook §3 ①）。

---

## 4. ⭐ KV over NVLink 三件套 `[待验证]`

**GB300 上 vLLM disagg 的 KV 必须走 NVLink cuda_ipc，不是 RDMA。** 这三样缺一不可：

```bash
export UCX_TLS=cuda_copy,cuda_ipc,tcp      # ① 只留 NVLink 路径，删掉 rdma/rc
export UCX_CUDA_IPC_ENABLE_MNNVL=y         #    让 cuda_ipc 跨节点走多机 NVLink
export UCX_NET_DEVICES=all
export NCCL_MNNVL_ENABLE=1 NCCL_CUMEM_ENABLE=1
# ② vllm serve 加 --enable-cumem-allocator
#    （用 VMM/cuMem 分配 KV，块才能通过 IPC handle 共享；不加会退回 cuda_copy）
# ③ prefill / decode 必须同 subblock（manifest 的 podAffinity 保证）
```

**实测效果：KV transfer 200 MB/s → 7,000–167,000 MB/s（7–167 GB/s），提速 100–800×**，单次 xfer 从 200–800ms 降到 2–10ms。

> **历史教训**：之前在 DOCA OFED / mooncake / dmabuf / nvidia-peermem 上折腾很久，**全是方向错误**。NIXL 本身就能走 NVLink，只差 `UCX_CUDA_IPC_ENABLE_MNNVL=y` + `--enable-cumem-allocator` 两个开关。

---

## 5. 启动 prefill / decode `[待验证]`

```bash
# 分发脚本
for p in $(kubectl get pods -l app=vllm -o name | sed 's|pod/||'); do
  kubectl exec -i $p -- bash -c "cat > /tmp/vllm-prefill-tp4.sh"  < scripts/vllm-prefill-tp4.sh
  kubectl exec -i $p -- bash -c "cat > /tmp/vllm-decode-tp4.sh && chmod +x /tmp/*.sh" < scripts/vllm-decode-tp4.sh
done

# ① 先起 prefill（它要先把 5557 side channel 开出来）
PIP=$(kubectl get pod vllm-0 -o jsonpath='{.status.podIP}')
kubectl exec vllm-0 -- bash -c "setsid nohup bash /tmp/vllm-prefill-tp4.sh $PIP > /tmp/srv.log 2>&1 </dev/null &"

# ② ★ 等 prefill 的 side channel 就绪，再起 decode
until kubectl exec vllm-0 -- bash -c "ss -tlnp 2>/dev/null | grep -q :5557"; do sleep 10; done
kubectl exec vllm-1 -- bash -c "setsid nohup bash /tmp/vllm-decode-tp4.sh $PIP > /tmp/srv.log 2>&1 </dev/null &"
```

**顺序是硬约束**：decode 起来会去连 prefill 的 NIXL side channel（5557），prefill 没就绪就起 decode 会握手失败。

**冷启动 ~8–12 分钟**，期间在做：DeepGEMM warmup（prefill ~2484 / decode ~1666 个 kernel）+ TileLang JIT + DSpark cudagraph capture（日志 `Capturing dspark CUDA graphs (FULL)`）。

**就绪判据**：

| 角色 | 判据 |
|---|---|
| prefill | `curl localhost:8001/health` = 200 |
| decode | `curl localhost:8002/health` = 200 |
| **优化 kernel 生效** | 启动日志有文首那三行（**这条最容易漏**）|

### 5.1 prefill 只能用 TP

vLLM 对 DeepSeek-V4 的 prefill 权重分片**只有 TP 可用**：

- **PP 不支持**：`--pipeline-parallel-size 4` → `NotImplementedError: Pipeline parallelism is not supported for this model`（模型未实现 `SupportsPP`）。**框架差异**：SGLang 支持 PP4 prefill。
- **DP 会 OOM**：`--data-parallel-size 4` 复制 attention/dense 权重 → 268GB/卡 → OOM。

### 5.2 decode 扩展：dep8（TP1 + DP8-attention + EP8）

TP4 decode 对 MLA-MoE 是**次优**的：**MLA 的 KV 是所有头共享的压缩 latent（512+64），TP 下不分片、只复制** —— TP4 把 KV cache 复制 4 份，零节省。

dep8 才是对的：DP-attention 每 rank 各存各请求的 KV（天然不复制）+ EP8 把 384 expert 摊到每卡 48 个（省 HBM → 更大 batch）+ attention/dense 权重只存一份。**实测每卡效率 2.6×**（1,983 vs 763 tok/s/GPU，ShareGPT 闭环口径）。

```bash
# head（node A，DP rank 0-3）
--tensor-parallel-size 1 --data-parallel-size 8 --data-parallel-size-local 4 \
--data-parallel-address <head-ip> --data-parallel-rpc-port 13345 --enable-expert-parallel
# headless worker（node B，DP rank 4-7）：上面基础上加
--data-parallel-start-rank 4 --headless
```

> **TP4-prefill → DP8-decode 的 KV 传输天然兼容** —— MLA 的 block 布局与 TP/DP 并行度无关（都是 per-rank 完整 latent）。所以「只改 decode、不动 prefill」路线成立，省掉全栈改造。

---

## 6. Router `[待验证]`

```bash
pip install vllm-router     # deepgemm 镜像已带
vllm-router --policy round_robin --vllm-pd-disaggregation \
  --host 0.0.0.0 --port 30000 \
  --prefill http://<prefill-ip>:8001 --decode http://<decode-ip>:8002 \
  --intra-node-data-parallel-size 1
```

> ### ⚠️ 运维大坑：router 进程名是 `vllm::router`（**带冒号**）
>
> **`pkill -f vllm-router` 永远杀不掉它。** 僵尸 router 会累积占住 Prometheus 端口，新 router 报 `FailedToCreateHTTPListener("Address already in use")`。
>
> **正确清理**：`pkill -9 -f 'vllm::router'`
>
> 同类坑还有 **`pkill -f 'vllm serve'` 漏杀 `VLLM::Worker` 子进程**（进程名不含 "vllm serve"），278GB×4 显存不释放 → 新实例 OOM。**必须按 `VLLM::` / `EngineCore` 进程名杀。**
>
> 参见 SGLANG runbook §6 的姊妹坑（`pkill -f dynamo.sglang` 会匹配到 `kubectl exec` 自身的命令行而自杀）—— **这一类「pkill 模式写不对」的故障在本项目里出现了三次**。

**ready 判据**：`curl localhost:30000/health` = 200。若 503「Prefill policy failed to select a worker」= 后端还没注册好，等几秒，或先查 `:8001` / `:8002` 是否 200。

---

## 7. 端到端验证 `[待验证]`

```bash
curl -s -X POST http://<router-ip>:30000/v1/completions -H 'Content-Type: application/json' \
  -d '{"model":"deepseek-ai/DeepSeek-V4-Pro-DSpark","prompt":"The capital of France is","max_tokens":10,"temperature":0}'
```

---

## 8. 压测 `[待验证]`

用 SGLang 的 `bench_serving` 打 router（跨框架同口径）：

```bash
python3 -m sglang.bench_serving --backend sglang-oai --host <router-ip> --port 30000 \
  --model deepseek-ai/DeepSeek-V4-Pro-DSpark --dataset-name random \
  --random-input-len 4096 --random-output-len 1024 --random-range-ratio 1.0 \
  --num-prompts $((2*CONC)) --max-concurrency $CONC     # 扫 conc 256 / 512
```

> ### ⚠️ `--random-range-ratio` 在两个 bench 里语义**相反**
>
> | 工具 | 固定长度 | 变长 |
> |---|---|---|
> | `sglang.bench_serving` | `1.0` | `0.8` |
> | `vllm bench serve` | **`0.0`** | — |
>
> 写错会把「固定 4096」变成「随机 0–4096」，平均输入长度腰斩，吞吐数字虚高一倍。

> ### ⚠️ 必须 warmup，冷跑差 2×
>
> 实测 2p1d 无 warmup 首跑 conc256 只有 **13,585**，warmup 后 **27,803**。原因是 DeepGEMM JIT 未热 + router 冷。**跑两轮，报第二轮**（同 SGLANG runbook §8.4）。

**参考结果**（2026-07-23 验证，4k1k）：

| 拓扑 | conc256 Total | conc512 Total | vs 厂商 22,000 |
|---|---|---|---|
| 1p1d | 23,120 | **24,358** | **111%** |
| 2p1d | 27,803 | **31,499** | 143%（加 prefill +29%）|

**达到 ≥22,000 = 复现成功。** 若远低于此：99% 是**用错了通用镜像**（回文首告警认那三行 kernel 日志）。

---

## 9. 故障速查

| 现象 | 根因 | 处理 |
|---|---|---|
| **跑得通但吞吐腰斩** | 用了通用镜像，`deep_gemm_mega_moe` 静默 fallback | 认启动日志三行；换 deepgemm 镜像 |
| `FailedToCreateHTTPListener("Address already in use")` | 僵尸 `vllm::router` 占 Prometheus 端口 | `pkill -9 -f 'vllm::router'`（带冒号）|
| 新实例 OOM、显存不释放 | `pkill -f 'vllm serve'` 漏杀 `VLLM::Worker` | 按 `VLLM::` / `EngineCore` 进程名杀 |
| KV 传输只有 ~200 MB/s | 走了 cuda_copy 不是 cuda_ipc | §4 三件套，特别是 `--enable-cumem-allocator` |
| decode 握手失败 | prefill 的 5557 side channel 还没就绪 | §5 的启动顺序 |
| router 503「Prefill policy failed to select a worker」 | 后端未注册 | 等几秒；查 8001/8002 |
| **pod 被驱逐（Evicted）** | **core dump 撑爆 ephemeral storage** | core pattern / TMPDIR 重定向到 `/mnt/ssd` |
| 高并发下崩溃 | DSpark spec `vectorized_gather` 索引越界 | 消融确认；禁 spec 规避 |
| gcloud 拉模型 403 | 节点 SA 对 bucket OAuth scope 未授权 | §3 的 `CLOUDSDK_AUTH_ACCESS_TOKEN` 方案 |
| `--pipeline-parallel-size` 报 NotImplementedError | vLLM 对 V4 未实现 `SupportsPP` | prefill 只能 TP（§5.1）|
| prefill DP4 OOM 268GB/卡 | DP 会复制 attention/dense 权重 | 同上，用 TP4 |

---

## 10. 验证记录

`[待实跑]` —— 本文骨架从 `vllm-v4-gb300-benchmark.md` 提炼，**尚未按本文步骤从零跑过**。计划同 SGLang：清空环境 → 只照本文命令重建 → 记录每步偏差 → 修 → 再跑一轮确认。

---

## 11. 性能结论

### 11.1 dep8 全程 prefill-feed-limited（与 SGLang 同结论）

8k1k、conc1024、sa-bench 开环，逐个加 prefill：

| prefill 数 | Output tok/s | **Output÷8 /GPU** | Median TPOT | 增量/prefill |
|---|---|---|---|---|
| 1 | 2,453 | 307 | 5.4 ms | — |
| 3 | 6,982 | 873 | 8.1 ms | +276 |
| 6 | 12,369 | **1,546** | 10.6 ms | +170 |

**output÷8 近似线性增长**（每加 1 prefill ~+250/GPU），到 6 prefill 才 1,546/GPU。**TPOT 全程 5–11ms** —— decode 有巨大余量，瓶颈完全在 prefill。

**4k1k 重扫同一套 dep8**（不重启）：

| prefill 数 | Output÷8 /GPU | vs 8k1k 同档 |
|---|---|---|
| 1 | 632 | **2.06×** |
| 2 | 1,238 | 2.07× |
| 4 | **2,222** | 1.99× |

**「输入减半 → 每 prefill 喂料翻倍」精确成立**（2.06 / 2.07 / 1.90 / 1.99）。但即便 4k1k + 4 prefill，dep8 decode **仍未饱和**（TPOT 才 9.0ms）。

> **扩 dep8 吞吐的杠杆是加 prefill，不是加 decode GPU。** decode 越宽、输入越长，越吃 prefill。
> **与 SGLang 侧完全一致** —— 见 [`SGLANG-V4PRO-RUNBOOK.md`](./SGLANG-V4PRO-RUNBOOK.md) §11.5：SGLang decode 自报峰值 11,113 = 官方 99.2%，稳态只有 9,587 也是因为 prefill 喂不满。**两个框架在同一台机器上撞到同一堵墙。**

### 11.2 vs SGLang：差多少、为什么

| 里程碑 | 结果 |
|---|---|
| 官方 1p1d 复现（4k1k）| **24,358 total tps = 厂商 22,000 的 111%** ✅ |
| dep8 8k1k 稳定峰值 | **3,321/GPU @ 14 prefill** |
| 同环境 SGLang dep8 8k1k | **~9,500/GPU** |
| 比值 | vLLM ≈ SGLang 的 **37%** |

**吞吐天花板真因**：decode **SM-bound 于低-FLOP 的 MoE 开销**（EP 通信 + 小 GEMM），不是喂料、不是带宽。MegaMoE 已开，MFU 仍只有 6%。

**⚠️ 诚实修正**：本环境 2.7× 的差距**偏大**，说明这套 vLLM dep8 **未调到 vLLM 最优**（缺 Wide-EP 满配 + KV/batch 平衡未极致）。公开对比里 SGLang 典型领先 **20–29%**（localaimaster 2026-02 +29%；gpustack H200 6635 vs 5482），极端案例 ~2×。**「3× 差距」不应作为 vLLM 的永久结论。**

**结构性差距在 AFD**：SGLang 有 attention-FFN 分离，vLLM **无原生支持**（[RFC #22799](https://github.com/vllm-project/vllm/issues/22799) 仍 open）。学界业界共识一致：
- FastAFD 官方 blog 原话："Long context starves the MoE layer by shrinking the KV-capped decode batch"
- MegaScale-Infer（arxiv 2504.02263）：分离 attention/FFN，MoE 推理提速 **1.9×**

> **收官定论**：vLLM 在本 GB300 环境**功能完整、能复现厂商 1p1d 22K 基线**；dep8 宽-EP decode 稳定峰值 ~3,400/GPU，受限于 coexist 架构的小-batch MoE 开销，**非 bug、非配置疏漏可根治**。逼近 SGLang 的 9,000+ 需要 AFD 级架构。
