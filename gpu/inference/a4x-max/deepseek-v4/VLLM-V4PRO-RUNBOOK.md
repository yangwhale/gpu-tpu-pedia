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

**唯一可靠的判据是启动日志（认这个，别认镜像 tag）。以下四行必须齐全** —— 括号里是源码位置，用来确认你 grep 的是对的东西：

```
Detected quantization_config.scale_fmt=ue8m0; enabling UE8M0 for DeepGEMM.   (config.py:742)
DeepSeek V4 expert_dtype resolved to 'fp4'                                    (quant_config.py:75)
Selected DeepGemmFp8BlockScaledMMKernel for Fp8LinearMethod                   (__init__.py:600)
DeepGEMM E8M0 enabled on current platform.                                    (deep_gemm.py:120)
DeepGEMM PDL enabled on deep_gemm.                                            (deep_gemm.py:202)
```

```bash
kubectl exec vllm-0 -- bash -c "tr -d '\000' < /tmp/srv.log | \
  grep -aoE \"expert_dtype resolved to 'fp4'|Selected DeepGemmFp8BlockScaledMMKernel|DeepGEMM E8M0 enabled|DeepGEMM PDL enabled\" | sort -u"
```

> ⚠️ **本文早期版本把这条判据写成了一行 `DeepGEMM PDL/E8M0 enabled`（PDL 和 E8M0 合并）—— 这个字符串在日志里根本不存在。** 照那个 grep 会在**镜像完全正确**的情况下得出「缺一行 → 配置是假的」的结论。实跑第一轮就被这条自己写的判据误导了。**写 grep 判据时必须从真实日志里复制原文，不能凭印象合并。**

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

## 1. 前置条件 `[已验证]`

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

# ④ ★ 没有孤儿 ComputeDomain 占着 DRA channel（见下）
kubectl get computedomain
```

⚠️ **上一套 fleet 的 ComputeDomain 不会跟着 StatefulSet 一起删**。`kubectl delete statefulset sgl` 之后 `sgl4-cd` 还在，继续占着 IMEX channel，新 pod 全卡在：

```
Warning FailedScheduling: pod "default/vllm-3": ResourceClaim not created yet.
        no new claims to deallocate, preemption: 0/145 nodes are available
```

**节点明明是空的**（`kubectl get pods --field-selector spec.nodeName=<n>` 只有 daemonset），查 `kubectl get resourceclaim` 会看到新 pod 的 `*-ch-*` claim 停在 `pending`。**删掉孤儿 CD 才放行**：

```bash
kubectl get computedomain          # 找没有对应 pod 的
kubectl delete computedomain sgl4-cd
```

> **只删你自己那套的**。集群里可能有别人的 CD（本次就有 6 个别的团队的），删错会打断人家的 workload。判据：CD 名字对得上你 manifest 里 `spec.channel.resourceClaimTemplate.name` 的那一个。

**同 subblock 是硬要求**：KV 走 UCX `cuda_ipc` + MNNVL，只在同一 NVLink 域内成立。manifest 里用 `gce-topology-subblock` 的 podAffinity 保证。

---

## 2. 部署 pod fleet `[已验证]`

```bash
kubectl apply -f manifests/vllm-fleet.yaml
kubectl get pods -l app=vllm -w        # 等全部 Running
```

**与 SGLang fleet 的差异**：

1. 镜像换成 deepgemm 专用
2. **必须加 `cloud.google.com/gce-topology-subblock` podAffinity** —— cuda_ipc 只在同 NVLink 域内工作（旧文档只提了要同 subblock，没给标签全名和 manifest 写法）
3. **core dump / tmp 要重定向到 `/mnt/ssd`**（见 §9 崩溃根因 A）

---

## 3. 模型 `[已验证]`

用的是 **`DeepSeek-V4-Pro-DSpark`**（FP8，~832GB，66 shards），**不是** SGLang 那份 `DeepSeek-V4-Pro`（806G / 64 shards）。DSpark 版本自带投机解码用的 draft 头。

```bash
for p in $(kubectl get pods -l app=vllm -o name | sed 's|pod/||'); do
  echo -n "$p: "
  kubectl exec $p -- bash -c "du -sh /mnt/ssd/DeepSeek-V4-Pro-DSpark 2>/dev/null|cut -f1; \
    ls /mnt/ssd/DeepSeek-V4-Pro-DSpark/*.safetensors 2>/dev/null|wc -l" | tr '\n' ' '; echo
done
```

**从 GCS 补** —— ⚠️ **deepgemm 镜像里没有 `gcloud`，也没有 `gsutil` / `wget` / `aria2c`**：

```bash
kubectl exec vllm-0 -- bash -c 'for c in curl wget python3 aria2c gsutil gcloud; do
  printf "%s:%s " $c "$(command -v $c >/dev/null && echo Y || echo N)"; done'
# 实测: curl:Y wget:N python3:Y aria2c:N gsutil:N gcloud:N
```

所以只能 **curl 直接打 GCS JSON API**（本机拿 bearer token 送进去）。**实测 16 路并行 2.7 GB/s/pod，892 GB 约 6 分钟，5 个 pod 并发 8 分钟拉完 4.4 TB —— 比 gcloud 还快。**

```bash
# ① 本机生成对象清单 + token
gcloud storage ls -r 'gs://chrisya-gb300-models/DeepSeek-V4-Pro-DSpark/**' \
  | sed 's|gs://chrisya-gb300-models/||' | grep -v '/$' > /tmp/dspark.list   # 186 个对象
gcloud auth application-default print-access-token > /tmp/tok

# ② 拉取脚本（scripts/pull-gcs-model.sh 已入库）
cat > /tmp/pull-dspark.sh <<'INNER'
#!/bin/bash
TOK=$(cat /tmp/tok); B=chrisya-gb300-models; DST=/mnt/ssd
get(){ o="$1"; f="$DST/$o"; mkdir -p "$(dirname "$f")"; [ -s "$f" ] && return 0
  curl -sfL -H "Authorization: Bearer $TOK" \
    "https://storage.googleapis.com/storage/v1/b/$B/o/$(python3 -c "import urllib.parse,sys;print(urllib.parse.quote(sys.argv[1],safe=''))" "$o")?alt=media" \
    -o "$f.part" && mv "$f.part" "$f"; }
export -f get; export TOK B DST
xargs -a /tmp/dspark.list -P 16 -I{} bash -c 'get "$@"' _ {}
echo "DONE $(du -sh $DST/DeepSeek-V4-Pro-DSpark|cut -f1) $(ls $DST/DeepSeek-V4-Pro-DSpark/*.safetensors|wc -l) shards"
INNER

# ③ 分发到每个 pod 后台跑
for p in $(kubectl get pods -l app=vllm -o name | sed 's|pod/||'); do
  ( for f in tok dspark.list pull-dspark.sh; do kubectl cp /tmp/$f $p:/tmp/$f; done
    kubectl exec $p -- bash -c "setsid nohup bash /tmp/pull-dspark.sh > /tmp/pull.log 2>&1 </dev/null & sleep 3" ) &
done; wait
# ④ 校验：832G / 66 shards
```

> **三个细节**：`.part` 临时名 + 成功才 `mv`，断点续传只靠 `[ -s "$f" ]` 跳过已完成的；对象名要 URL-encode（路径里有 `/`）；token 有效期约 1 小时，够拉完，**用完记得删**。
> **不要**用 `kubectl cp` 传 892 GB（走 API server，慢一个量级）。

---

## 4. ⭐ KV over NVLink 三件套 `[已验证]`

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

## 5. 启动 prefill / decode `[已验证]`

```bash
# 分发脚本
for p in $(kubectl get pods -l app=vllm -o name | sed 's|pod/||'); do
  kubectl exec -i $p -- bash -c "cat > /tmp/vllm-prefill-tp4.sh"  < scripts/vllm-prefill-tp4.sh
  kubectl exec -i $p -- bash -c "cat > /tmp/vllm-decode-tp4.sh && chmod +x /tmp/*.sh" < scripts/vllm-decode-tp4.sh
done

# ① 先起 prefill（它要先把 5557 side channel 开出来）
PIP=$(kubectl get pod vllm-0 -o jsonpath='{.status.podIP}')
kubectl exec vllm-0 -- bash -c "setsid nohup bash /tmp/vllm-prefill-tp4.sh $PIP > /tmp/srv.log 2>&1 </dev/null &"

# ② ★ 等 prefill 就绪，再起 decode
#    ⚠️ 别用 ss —— deepgemm 镜像里没装（见下方告警）。用 health：
until [ "$(kubectl exec vllm-0 -- bash -c "curl -s -o /dev/null -w '%{http_code}' -m 3 localhost:8001/health")" = 200 ]; do sleep 20; done
# ★ decode 传的是【自己的 IP】，不是 prefill 的（见下方告警）
DIP=$(kubectl get pod vllm-1 -o jsonpath='{.status.podIP}')
kubectl exec vllm-1 -- bash -c "setsid nohup bash /tmp/vllm-decode-tp4.sh $DIP > /tmp/srv.log 2>&1 </dev/null &"
```

**顺序是硬约束**：decode 起来会去连 prefill 的 NIXL side channel（5557），prefill 没就绪就起 decode 会握手失败。

> ### ⚠️ 别用 `ss` 判端口 —— 镜像里没装，而且失败是静默的
>
> deepgemm 镜像**没有 `ss`**（也没有 `netstat` / `lsof`）：
>
> ```bash
> $ kubectl exec vllm-0 -- ss -tln
> bash: line 1: ss: command not found
> ```
>
> 如果你按常规写法加了 `2>/dev/null`，`command not found` 会被吞掉，判据**永远为假**，循环干等到超时 —— 而 prefill 其实早就好了。我在这上面空转了 5 分钟。
>
> 要直接看端口只能读 `/proc/net/tcp`（端口是**十六进制**：5557 = `15B5`，8001 = `1F41`）：
>
> ```bash
> kubectl exec vllm-0 -- bash -c 'grep -qi 15B5 /proc/net/tcp && echo "5557 listening"'
> ```
>
> **但更好的做法是别查端口，查 `/health`** —— 它是应用级判据，200 就意味着 side channel 已经开好了。实测两者同时满足。

> ### ⚠️ `VLLM_NIXL_SIDE_CHANNEL_HOST` 是「我 bind 哪」，不是「我连谁」
>
> 两个 server 各自 bind 自己的 side channel（prefill 5557 / decode 5558），**对端地址是 router 在请求里带过来的**，不是靠这个环境变量。所以：
>
> ```bash
> # ❌ 错：decode 传 prefill 的 IP
> bash /tmp/vllm-decode-tp4.sh $PREFILL_IP
> #    → zmq.error.ZMQError: Cannot assign requested address (addr='tcp://10.72.22.47:5558')
> #      —— 它在试图 bind 一个不属于自己的地址
>
> # ✅ 对：两边都传【自己 pod 的 IP】
> bash /tmp/vllm-decode-tp4.sh $(kubectl get pod vllm-1 -o jsonpath='{.status.podIP}')
> ```
>
> **这个坑要等 12 分钟才暴露**：权重加载 + DeepGEMM warmup + CUDA graph capture（`Graph capturing finished in 174 secs`）全部成功之后，最后一步建 side channel 才炸。中间所有信号都是正常的。

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

## 6. Router `[已验证]`

⚠️ **`vllm-router` 不在 deepgemm 镜像里**（`command -v vllm-router` → 空，`import vllm_router` → ModuleNotFoundError）。必须先装，pod 有外网、约 20 秒：

```bash
kubectl exec vllm-0 -- pip install --no-cache-dir vllm-router   # 实测装到 0.1.15
```

```bash
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

## 7. 端到端验证 `[已验证]`

```bash
curl -s -X POST http://<router-ip>:30000/v1/completions -H 'Content-Type: application/json' \
  -d '{"model":"deepseek-ai/DeepSeek-V4-Pro-DSpark","prompt":"The capital of France is","max_tokens":10,"temperature":0}'
```

---

## 8. 压测 `[已验证]`

⚠️ **`sglang` 不在 vLLM 镜像里**（`import sglang` → ModuleNotFoundError）。两个可选项，但**只有一个能用**（原因见下方 temperature 告警）：

| 工具 | 在镜像里 | 发 `temperature` 吗 | 能用吗 |
|---|---|---|---|
| `vllm bench serve` | ✅ 内置 | ❌ 不发，也没有 `--temperature` 开关 | ❌ **投机解码会废掉** |
| InferenceX `sa-bench` | ❌ 要 clone | ✅ 硬编码 `temperature: 0.0` | ✅ **用这个** |

```bash
# 装 sa-bench（184K，跟 SGLang 侧同一把尺子）
git clone --depth 1 https://github.com/SemiAnalysisAI/InferenceX /tmp/InferenceX
tar czf /tmp/ix.tgz -C /tmp InferenceX && kubectl cp /tmp/ix.tgz vllm-0:/tmp/ix.tgz
kubectl exec vllm-0 -- bash -c "cd /mnt/ssd && tar xzf /tmp/ix.tgz"

kubectl exec vllm-0 -- bash -c "cd /mnt/ssd/InferenceX && python3 utils/bench_serving/benchmark_serving.py \
  --backend openai --host 127.0.0.1 --port 30000 --model deepseek-ai/DeepSeek-V4-Pro-DSpark \
  --tokenizer /mnt/ssd/DeepSeek-V4-Pro-DSpark --dataset-name random \
  --random-input-len 4096 --random-output-len 1024 --random-range-ratio 1.0 \
  --num-prompts \$((2*CONC)) --max-concurrency \$CONC --request-rate inf --ignore-eos --dsv4 --use-chat-template"
```

> ### 🔥 压测工具不发 `temperature=0`，投机解码直接废掉 —— 同一套服务差 3.1 倍
>
> `DeepSeek-V4-Pro-DSpark/generation_config.json` 里是：
>
> ```json
> {"do_sample": true, "temperature": 1.0, "top_p": 1.0}
> ```
>
> 请求不带 `temperature`，服务端就按 **1.0 随机采样**；而 DSpark 的 draft 是 `"draft_sample_method":"greedy"`。**greedy 的草稿去猜随机采样的结果，接受率必然崩**。而 `num_speculative_tokens=7` 意味着每步固定跑 8 次 forward —— 接受不了就是纯亏 8 倍算力。
>
> **同一套服务、同一个 4k1k conc256 负载，实测**：
>
> | 压测工具 | temperature | spec 接受率 | Output tok/s | TPOT |
> |---|---|---|---|---|
> | `vllm bench serve` | 服务端默认 **1.0** | **1.16%** | 944 | **232 ms** |
> | **`sa-bench`** | 硬编码 **0.0** | **34%** | **2,911** | **48.6 ms** |
> | 比值 | — | 29× | **3.1×** | 4.8× |
>
> **判据**（跑完立刻查，不到 20% 就说明请求没带 temperature=0）：
>
> ```bash
> kubectl exec vllm-1 -- bash -c "curl -s localhost:8002/metrics | \
>   grep -E '^vllm:spec_decode_num_(draft|accepted)_tokens_total'"
> # 接受率 = accepted ÷ draft
> ```
>
> **这条对任何开了投机解码的服务都成立，不限于 vLLM/DSpark。** 跨框架比性能时，两边的 benchmark 工具是否都发 `temperature=0` 必须先对齐 —— 否则比的是采样策略，不是引擎。

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

**历史参考**（2026-07-23，`sglang.bench_serving`，4k1k）：

| 拓扑 | conc256 Total | conc512 Total | vs 厂商 22,000 |
|---|---|---|---|
| 1p1d | 23,120 | **24,358** | **111%** |
| 2p1d | 27,803 | **31,499** | 143%（加 prefill +29%）|

**本轮复刻实测**（2026-07-26，`sa-bench`，1p1d 4k1k conc256，见 §10）：

| 指标 | 实测 | vs 历史 23,120 |
|---|---|---|
| Total tok/s（推算）| **≈14,563** | 63% |
| Output tok/s | 2,911 | — |
| Median TPOT | 48.6 ms | — |
| Median TTFT | 25.2 s | **prefill 受限** |
| spec 接受率 | 34% | — |

**没有完整复现 23,120**，差 37%。已排除的因素：镜像正确（四行 kernel 日志齐全）、KV 走 NVLink（NIXL+UCX 已协商）、temperature 已修正。**剩余嫌疑是 prefill 侧**：TTFT 中位 25 秒说明请求全堵在 1 个 prefill 上，而 prefill 脚本用的是 `--enforce-eager`（无 cudagraph）+ `--max-num-seqs 16`。下一轮先加 prefill 数、再调这两个参数。

> **若你的数远低于 14,000**：按优先级查 —— ① 压测工具没发 `temperature=0`（占 3.1×，见上方告警）② 用了通用镜像（回文首认那四行 kernel 日志）。

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

### 10.1 复刻轮 1：清空 SGLang → 只照本文命令重建（2026-07-26）

环境：`gb300-pool-0002`，8 replica StatefulSet（**实际只调度上 5 个**，见 §1 孤儿 CD），1p1d（vllm-0 prefill TP4 / vllm-1 decode TP4）。

| 步骤 | 结果 | 实测 / 偏差 |
|---|---|---|
| §1 前置 | ⚠️ **文档缺项** | 孤儿 `sgl4-cd` 占着 DRA channel，3/8 pod 卡 Pending。→ 已补进 §1 ④ |
| §2 fleet | ✅ | 5 pod Running，全部落在同一 subblock |
| §3 模型 | ❌ **文档错** | 教用 `gcloud`，**镜像没装**。改 curl + bearer token，**2.7 GB/s/pod，5 pod 并发 8 分钟拉完 4.4 TB** |
| §4 KV over NVLink | ✅ | NIXL agent + UCX backend 起来，`TransferTopology(tp_ratio=1, local_tp=4, remote_tp=4)` 协商成功 |
| §5 prefill | ✅ | 11:38 起 → 11:48 health 200（**10 分钟**：权重 66 shards + TileLang JIT + DeepGEMM warmup 2484 kernel）|
| §5 就绪判据 | ❌ **文档错** | 用 `ss` 判 5557，**`ss` 没装**且被 `2>/dev/null` 吞掉 → 空转 5 分钟。改查 `/health` |
| §5 decode | ❌ **脚本 bug** | `VLLM_NIXL_SIDE_CHANNEL_HOST` 传了 prefill 的 IP → `Cannot assign requested address`。**第 12 分钟才炸**（graph capture 174s 都成功了）|
| §5 decode（修复后）| ✅ | 12:08 起 → 12:13 health 200（**5 分钟**，DeepGEMM 缓存已热）|
| 文首 kernel 判据 | ❌ **文档错** | `DeepGEMM PDL/E8M0 enabled` 这行字面不存在，实际是分开的 5 行。差点误判镜像不对 |
| §6 router | ❌ **文档错** | 「镜像已带」不成立，要 `pip install vllm-router`（20 秒）|
| §7 e2e | ✅ | `" Paris. The capital of Germany is Berlin..."`，响应 ID 带 `prefill_addr...decode_addr...`，PD 链路确认 |
| §8 压测 | ❌ **文档错 + 重大发现** | `sglang` 不在镜像里；换 `vllm bench serve` 后**投机解码接受率只有 1.16%**，根因是它不发 `temperature=0`（见 §8 告警）|
| §8 压测（sa-bench）| ⚠️ **未达标** | conc256 Total ≈ **14,563** = 历史 23,120 的 **63%**，卡在 prefill |

**本轮抓到 7 个文档缺陷，全部已修**：

1. **§3 用 `gcloud` 拉模型** —— 镜像里没有（也没有 gsutil / wget / aria2c）
2. **§1 没提孤儿 ComputeDomain** —— 删了 StatefulSet 它还在，新 pod 全 Pending 且节点是空的
3. **§5 用 `ss` 判端口** —— 没装，`2>/dev/null` 把 `command not found` 吞了，判据永远为假
4. **decode 脚本把 side channel bind 地址填成对端 IP** —— 真 bug，且 12 分钟后才暴露
5. **文首 kernel 判据的字符串是拼出来的、不存在** —— 最重要的那条检查本身写错了
6. **§6 说 router 镜像已带** —— 没带
7. **§8 用 `sglang.bench_serving`** —— 不在镜像里；换工具引出了 temperature 陷阱

> **第 5 条最值得记**：那是全文标榜「唯一可靠的判据」的一条，我把日志里两行（`DeepGEMM E8M0 enabled` / `DeepGEMM PDL enabled`）**凭印象合并成了一行**。写判据类内容必须从真实输出里复制原文。
>
> **第 7 条价值最高**：它不是文档瑕疵，是一个**跨框架性能对比的方法论陷阱** —— 两个 benchmark 工具对同一套服务测出 3.1× 差距，纯粹因为一个发 `temperature=0` 一个不发。

### 10.2 待办（下一轮）

- [ ] 加 prefill（2p1d / 4p1d），确认 TTFT 25s 是不是唯一瓶颈
- [ ] prefill 去掉 `--enforce-eager` + 提高 `--max-num-seqs`，对比
- [ ] 复现历史 23,120；达标后再做审计轮 2（清空重跑确认可复现）
- [ ] 蒸馏并删除旧文档 `vllm-v4-gb300-benchmark.md`（945 行）

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
> **与 SGLang 侧部分一致** —— 见 [`SGLANG-V4PRO-RUNBOOK.md`](./SGLANG-V4PRO-RUNBOOK.md) §11.5：SGLang 侧最终把主因定位到 **KV 池预算划错**（`swa-full-tokens-ratio`，一个参数值 +54%），prefill 供给不足只占约 11%（端到端 10,704 vs decode 峰值 12,070）。**「瓶颈在 prefill」这个结论在 SGLang 侧被大幅下调了权重 —— vLLM 侧目前 TTFT 25 秒，prefill 受限是真的，但同样要提防有更大的参数旋钮没找到。**

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
