# GB300 (A4X Max) · SGLang DeepSeek-V4 · 端到端测试指南 + Benchmark 报告

> **状态：Phase 1（Flash）+ Phase 2（Pro）单节点 TP4 已实测通过（2026-07-20）**；Phase 3（PD-disagg 冲吞吐）待跑。
> §3.9 是**可照抄复现的完整运行手册**（建 pod → 下权重 → 备份 GCS → 启动 → 压测，每步带实测脚本）；§7 是 **Benchmark 汇总报告**（Flash / Pro / R1 三方对比）。
> 结论先行：**入门不复杂**——V4-Flash（284B）单节点 GB300 TP=4 一台机器 4 卡就能起，比 R1 的 PD-disagg 64 卡简单得多。难的是复现官方 11,200 tok/s/GPU 那套（18 节点 + MegaMoE W4A4 + SWA + Dynamo）。
>
> 资料来源：SGLang V4 cookbook、lmsys Day-0 博客（2026-04-25）、pytorch「Serving DeepSeek-V4 on GB300」（2026-06-23）、SemiAnalysis InferenceX srt-slurm recipe `disagg-gb300-10p1d-dep4-dep32-18-c2500.yaml`。

---

## 0. 为什么 V4 能上万而 R1 上不了（先理解本质）

我们实测 R1 短上下文（8K/1K）峰值 **1,359 tok/s/GPU**，官方 V4 Pro 同 workload **11,200**，差 ~8×。根因不是我们没调好，是模型代际：

| 维度 | R1（我们跑的）| V4（要跑的）|
|---|---|---|
| 注意力 | 全注意力 MLA，KV 留全历史 | **hybrid CSA + HCA**，~10% KV cache vs V3.2 @1M；等效滑动窗口 |
| decode `max-running-requests` | 2048（KV 大，塞不下更多）| **18432**（KV 薄，并发拉 9×）← 吞吐上万的直接原因 |
| MoE 量化 | W4A8（权重4bit/激活8bit）| **W4A4 MegaMoE**（激活也4bit，矩阵乘快~2×）|
| KV 压缩 | 无 | **online compress**（C4/C128 压缩态池）|
| 上下文 | 128K | **1M** |

**一句话**：V4 靠 CSA+HCA 把 KV 打薄 → decode 并发从 2K 拉到 18K → 吞吐堆上万。这是架构层解访存瓶颈，R1 的全注意力天生追不上。

---

## 1. 模型与 checkpoint

V4 两个变体（2026-04-24 发布，MIT License）：

| 变体 | 总参 | 激活 | GB300 部署 |
|---|---|---|---|
| **DeepSeek-V4-Flash** | 284B | 13B | **单节点 TP=4**（入门首选）|
| **DeepSeek-V4-Pro** | 1.6T | 49B | 单节点 TP=4；或 PD-disagg 冲吞吐 |

**checkpoint 选项**（Instruct 版才能 chat；Base 版只用于继续预训练）：
- `deepseek-ai/DeepSeek-V4-Flash` / `-Pro` —— 官方 stock，**FP4 MoE + FP8 attn/dense** 混合精度，一份覆盖所有 FP4 GPU。
- `nvidia/DeepSeek-V4-Flash-NVFP4` / `-Pro-NVFP4` —— NVFP4 混合（MoE NVFP4 + attn FP8），需 `--moe-runner-backend flashinfer_trtllm_routed`（不给会自动选）。**跟我们 R1 用的 NVFP4 一脉相承，优先试这个**。
- `sgl-project/DeepSeek-V4-*-FP8` —— 仅 Hopper 用，GB300 用不上。

推荐生成参数：`temperature=1.0, top_p=1.0`。三种推理模式：Non-think / Think High / Think Max（Think Max 需 ≥384K context）。

---

## 2. 关键差异：跟 R1 相比，什么变、什么不变

**不变（直接复用 R1 的环境，见 deepseek-v3 DEPLOY-GUIDE §1-§4）：**
- GB300 GKE 集群 + `pool-0010` + `team=yangwhale` 节点
- GIB（GCS `gib-a4xmax.tgz`）+ DOCA OFED userspace + gcloud bootstrap
- mooncake NVLINK KV pool（`MC_FORCE_MNNVL=1 NCCL_MNNVL_ENABLE=1 NCCL_CUMEM_ENABLE=1`）—— cookbook 明确 GB300 cross-pod 也要这三个
- pod YAML 模板（ComputeDomain + mrdma DRA + subblock podAffinity + 内存盘放模型）

**变（V4 新东西）：**
| 项 | R1 | V4 |
|---|---|---|
| **镜像** | `lmsysorg/sglang:v0.5.15.post1-cu130` | **需要新镜像**：`lmsysorg/sglang:latest` 或 recipe 用的 nightly `nightly-dev-cu13-20260520-425dffbd`（v0.5.15 太旧不支持 V4）|
| 模型 | R1-0528-NVFP4（385G）| V4-Flash（~150G）/ V4-Pro（~800G FP4）|
| MoE backend | deepep + flashinfer_cutedsl | **megamoe**（W4A4）|
| 注意力特有 | — | SWA opts（`SGLANG_OPT_SWA_*`）、online compress（`SGLANG_OPT_USE_ONLINE_COMPRESS=1`）、`swa-full-tokens-ratio` |
| 编排 | sglang_router | 官方用 **Dynamo**（kv router）；我们可先用 sglang_router 简化 |
| tokenizer | 标准 | V4 有专用 `encoding_dsv4` + 官方 bench 用 custom tokenizer |

---

## 3. 分阶段测试计划（一步一个脚印，从易到难）

> **存储：全程 Local SSD RAID，不用内存盘**。V4-Pro 1.6T ≈ 800G FP4，内存盘（tmpfs）放不下（800G 模型 + 运行时 > 942G 节点 RAM）；且 RAM 要留给 KV cache / HiCache CPU offload。所有 pod 模型都放 **Local SSD RAID `/mnt/disks/raid/0`**（读 14 GB/s，跨 pod 持久）。详见 [`./gb300-local-ssd-raid0-SETUP.md`](./gb300-local-ssd-raid0-SETUP.md)。

### Phase 0：Local SSD RAID 就位 + 拉镜像 + 模型 + 验证 sm_103a
1. **先确保节点池 Local SSD RAID 就位**：部署 `gke-raid-disks` DaemonSet，逐节点 `grep -c md0 /proc/mdstat` 确认全 =1（12T RAID 挂 `/mnt/disks/raid/0`）。**用干净池**（R1 实测 pool-0007 17/17 全成）；有残留 `mnt-disks-ssdN.mount` 污染的节点 RAID 会失败，需重建（见 RAID-SETUP 坑速查）。
2. 起 1 个 GB300 pod：复用 R1 gen-pods（image 改 `lmsysorg/sglang:latest`，pod=1），**ssd 卷用 hostPath `/mnt/disks/raid/0` + `mountPropagation: HostToContainer`，内存 request 600Gi**（模型在 Local SSD 不进 RAM；≥600Gi 覆盖加载峰值，200Gi 会 OOM）。
3. **验证镜像含 sm_103a**（GB300 = cc 10.3）：`cuobjdump` 查 `sgl_kernel/*.so` arch 有没有 `sm_103a`（R1 踩过这个坑，V4 新镜像必须重验）。
4. bootstrap（GIB + DOCA + gcloud，同 R1 §4）。
5. `gcloud storage cp` 拉 `nvidia/DeepSeek-V4-Flash-NVFP4` **到 Local SSD `/mnt/ssd`**（背后是 RAID；先拉小的 Flash。R1 实测 GCS→Local SSD ~5.4 GiB/s）。

### Phase 1：V4-Flash 单节点 TP=4 冒烟（最简单，先跑通）★ 从这里开始
- **1 台 GB300 / 4 GPU，无 PD、无多节点、无 MegaMoE**。
- 最小启动（low-latency 配方，**model-path 指 Local SSD**）：
  ```bash
  python -m sglang.launch_server --model-path /mnt/ssd/DeepSeek-V4-Flash-NVFP4 \
    --tp-size 4 --trust-remote-code \
    --moe-runner-backend flashinfer_trtllm_routed \
    --reasoning-parser deepseek-v4 --tool-call-parser deepseekv4 \
    --host 0.0.0.0 --port 30000
  ```
- 验证：curl `/v1/chat/completions`，确认返回 + `reasoning_content` 分离正常。
- **目标**：先证明 V4 能在我们的 GB300 上从 Local SSD 加载 + 生成。跑通即 Phase 1 成功。

> **✅ Phase 1 实测通过（2026-07-20）**：`lmsysorg/sglang:latest` = **0.5.15.post1**（就含 `deepseek_v4.py`/`deepseek_v4_nextn.py`，**不用 nightly**，且有 sm_103a）。V4-Flash-NVFP4（157G/59 文件）从 Local SSD 加载，日志出现 V4 专属 `DeepSeek V4 MHC prenorm prewarm` + `DeepseekV4AttnBackend`，chat 生成正确（"capital of France"→"Paris"）。**坑**：load 慢（~12min）因 autotune 默认开 + 上传抢 CPU——冒烟加 `--disable-flashinfer-autotune` 快很多。
>
> **权重下载 + GCS 备份（关键，避免多节点重复从 HF 拉）**：
> 1. **下载用 `hf download`**（新版 CLI；`huggingface-cli download` 已废、只打 help 不干活）：`HF_HUB_ENABLE_HF_TRANSFER=1 hf download nvidia/DeepSeek-V4-Flash-NVFP4 --local-dir /mnt/ssd/DeepSeek-V4-Flash-NVFP4`（hf_transfer ~840MB/s，157G ~3min）。
> 2. **传 GCS 走 R1 那套 ADC+SDK 法**（node scope 只读、org 禁 SA key、gcloud 不认 ADC——见 R1 RUNLOG 坑5）：kubectl cp glinux 的 `~/.config/gcloud/application_default_credentials.json` 进 pod，用 python `google-cloud-storage` SDK（`GOOGLE_APPLICATION_CREDENTIALS=adc.json` + ThreadPoolExecutor(16)）上传，**传完删 adc.json**。V4-Flash 168GB / 75 文件 ~196s。GCS: `gs://chrisya-gb300-models/DeepSeek-V4-Flash-NVFP4`。
> 3. 以后各节点 bootstrap 直接 `gcloud storage cp -r gs://.../DeepSeek-V4-Flash-NVFP4 /mnt/ssd/`（读只需 ro scope，节点默认有）。
>
> **Phase 1 压测（单节点 TP4 / 4 GPU / 8K-1K / warm）**：
>
> | 并发 | 总吞吐 tok/s | 总/GPU(÷4) | output tok/s | TPOT median | TTFT median |
> |---|---|---|---|---|---|
> | 1 | 1520 | 380 | 169 | 5.71 ms | 220 ms |
> | 16 | 12477 | 3119 | 1386 | 8.06 ms | 1709 ms |
> | 64 | **34162** | **8540** | 3796 | 13.0 ms | 3102 ms |
>
> - **单节点 4 卡 conc64 就到 8540 tok/s/GPU（in+out）**——比我们 R1 64 卡 8K/1K 的 1359 高 **6.3×**。V4-Flash 284B/13B 激活 + SWA，每 token 效率碾压 R1 671B 全注意力。
> - conc1 单用户 TPOT **5.71ms（≈175 tok/s/user）**，交互极快。conc64 TTFT 3.1s 仍可用，未见顶——可继续压更高并发。
> - 这是**单节点 Flash**（非 Pro、非 18 节点 PD）。官方 11,200 是 V4-Pro 8K/1K + 18 节点 Dynamo + MegaMoE W4A4——Phase 3 再冲。

### Phase 2：V4-Pro 单节点 TP=4
- cp `nvidia/DeepSeek-V4-Pro-NVFP4`（~800G）到 **Local SSD**，`--model-path /mnt/ssd/DeepSeek-V4-Pro-NVFP4`（单节点 4×277G=1108G HBM 放得下）。**这里就是 Local SSD 的价值所在**：800G 模型放 Local SSD 不吃 RAM；若放内存盘，800G tmpfs + 运行时直接爆 942G 节点内存。
- 同 Phase 1 启动，`--tp-size 4`。验证加载 + 生成。

> **✅ Phase 2 实测通过（2026-07-20）**：`nvidia/DeepSeek-V4-Pro-NVFP4`（**851G / 76 文件**）从 HF 下载到 Local SSD（hf_transfer ~0.9 GB/s，851G 约 15 min），单节点 TP4 加载：
> - **权重加载 <1 min**（Local SSD 读满速，`avail mem=274GB/GPU`），autotune + DeepGEMM warmup（32768 kernels）+ CUDA graph 约 12 min 到 ready。冒烟可加 `--disable-flashinfer-autotune` 提速。
> - chat 生成正确（"capital of France"→"Paris"）。GCS 备份 `gs://chrisya-gb300-models/DeepSeek-V4-Pro-NVFP4/`（913G，ADC+SDK 上传 668s；ADC 用后即删）。
>
> **Phase 2 压测（单节点 TP4 / 4 GPU / 8K-1K / warm）**：
>
> | 并发 | Total tok/s（in+out）| **tok/s/GPU** | Output tok/s | Median TPOT | Median TTFT |
> |---|---|---|---|---|---|
> | 1 | 838 | 209 | 93 | 10.44 ms | 295 ms |
> | 16 | 7594 | 1898 | 844 | 16.55 ms | 2193 ms |
> | 64 | **11177** | **2794** | 1242 | 33.74 ms | 9845 ms |
>
> - **单节点 4 卡 conc64 到 2794 tok/s/GPU（in+out）**——比同 workload 的 V4-Flash（8540）低 **3.1×**，符合预期：Pro 1.6T/49B 激活 vs Flash 284B/13B 激活，模型大 5.6× / 激活大 3.8×。
> - conc1 单用户 TPOT **10.44ms（≈96 tok/s/user）**，比 Flash（175 tok/s/user）慢约一半，Pro 更重但仍交互流畅。
> - conc64 TTFT 9.8s 偏高——单节点 4 卡跑 1.6T prefill 压力大，这正是 Phase 3 上 PD-disagg（prefill 独立扩展）要解决的。官方 11,200 tok/s/GPU 是 Pro + 18 节点 Dynamo + MegaMoE W4A4，单节点做不到，Phase 3 再冲。

### 3.9 可复测运行手册（Phase 1 + Phase 2 单节点实录）★ 照抄即可复现

> 全程 `gcloud`/`kubectl` **在 gLinux 跑**（本机 Context Aware Access 被拦）：`ssh glinux` 后 `bash -l -c "export PATH=\$HOME/google-cloud-sdk/bin:\$PATH; <cmd>"`。集群凭证：`gcloud container clusters get-credentials gb300-gke-test --region us-central1 --project tencent-gcp-taiji-poc`。

**前置**：干净节点池 Local SSD RAID 就位（见 [`./gb300-local-ssd-raid0-SETUP.md`](./gb300-local-ssd-raid0-SETUP.md)，pool-0007 实测 17/17）。

#### Step 1 — 起单节点 pod（Local SSD hostPath，600Gi）

`v4-pro.yaml`（Flash 同理，改 name/nodepool；Pro 需 ≥600Gi 覆盖加载峰值）：
```yaml
apiVersion: v1
kind: Pod
metadata: {name: v4-pro, labels: {app: v4-pro}}
spec:
  nodeSelector: {cloud.google.com/gke-nodepool: gb300-pool-0007, team: yangwhale}
  tolerations: [{operator: Exists}]
  containers:
  - name: sglang
    image: lmsysorg/sglang:latest       # =0.5.15.post1，含 deepseek_v4.py + sm_103a
    securityContext: {privileged: true}
    command: ["sleep","infinity"]
    resources:
      limits: {nvidia.com/gpu: 4, memory: 600Gi}
      requests: {memory: 600Gi}
    volumeMounts:
    - {name: ssd, mountPath: /mnt/ssd, mountPropagation: HostToContainer}   # 关键
    - {name: shm, mountPath: /dev/shm}
  volumes:
  - {name: ssd, hostPath: {path: /mnt/disks/raid/0, type: Directory}}       # RAID
  - {name: shm, emptyDir: {medium: Memory, sizeLimit: 64Gi}}
```
```bash
kubectl apply -f v4-pro.yaml
kubectl exec v4-pro -- df -h /mnt/ssd   # 应见 /dev/md0 12T（RAID 挂上了）
```

#### Step 2 — 下权重到 Local SSD（`hf download`，不是 `huggingface-cli`）

```bash
kubectl exec v4-pro -- bash -c '
export HF_HUB_ENABLE_HF_TRANSFER=1
pip install -q hf_transfer huggingface_hub google-cloud-storage
nohup hf download nvidia/DeepSeek-V4-Pro-NVFP4 \
  --local-dir /mnt/ssd/DeepSeek-V4-Pro-NVFP4 > /mnt/ssd/dl.log 2>&1 &'
# 实测：Pro 851G/76 文件 ~15min（hf_transfer ~0.9GB/s）；Flash 157G/59 文件 ~3min
```

#### Step 3 — 备份 GCS（ADC + python SDK；node scope 只读 + org 禁 SA key，只能这样）

> **为什么不能直接 `gcloud storage cp`**：GKE 节点 OAuth scope 是 storage **只读**（VM 级封顶，IAM 授权也不解），且 org policy 禁建 SA key。唯一可写法：把 gLinux 的**用户 ADC** 拷进 pod，用 python `google-cloud-storage` SDK（SDK 认 `GOOGLE_APPLICATION_CREDENTIALS` 用户凭证，绕开节点 scope）。**传完立即删 ADC**（敏感）。详见 R1 RUNLOG 坑5。

```bash
# 1) 拷用户 ADC 进 pod
kubectl cp ~/.config/gcloud/application_default_credentials.json v4-pro:/mnt/ssd/adc.json
# 2) SDK 并发上传（v4sdkup.py）
cat > /tmp/v4sdkup.py <<'PY'
import os, glob, time, concurrent.futures
os.environ["GOOGLE_APPLICATION_CREDENTIALS"]="/mnt/ssd/adc.json"
from google.cloud import storage
client=storage.Client(project="tencent-gcp-taiji-poc")
bucket=client.bucket("chrisya-gb300-models")
src="/mnt/ssd/DeepSeek-V4-Pro-NVFP4"
files=[f for f in glob.glob(src+"/**",recursive=True) if os.path.isfile(f) and "/.cache/" not in f]
def up(f):
    rel=os.path.relpath(f,src)
    bucket.blob(f"DeepSeek-V4-Pro-NVFP4/{rel}").upload_from_filename(f); return rel
t0=time.time()
with concurrent.futures.ThreadPoolExecutor(16) as ex: n=len(list(ex.map(up,files)))
print(f"UP_DONE {n} files in {int(time.time()-t0)}s",flush=True)
PY
kubectl cp /tmp/v4sdkup.py v4-pro:/root/v4sdkup.py
kubectl exec v4-pro -- python /root/v4sdkup.py     # Pro 913G ~668s；Flash 168G ~196s
# 3) 删 ADC（务必）
kubectl exec v4-pro -- rm -f /mnt/ssd/adc.json
# 以后别的节点直接读（只读 scope 够）：gcloud storage cp -r gs://chrisya-gb300-models/DeepSeek-V4-Pro-NVFP4 /mnt/ssd/
```

#### Step 4 — 启动单节点 TP4 + 验证生成

```bash
kubectl exec v4-pro -- bash -c '
export SGLANG_DG_CACHE_DIR=/mnt/ssd/dg-cache FLASHINFER_WORKSPACE_BASE=/mnt/ssd/fi-cache FLASHINFER_DISABLE_VERSION_CHECK=1
nohup python -m sglang.launch_server --model-path /mnt/ssd/DeepSeek-V4-Pro-NVFP4 \
  --served-model-name deepseek-ai/DeepSeek-V4-Pro --tp-size 4 --trust-remote-code \
  --moe-runner-backend flashinfer_trtllm_routed \
  --reasoning-parser deepseek-v4 --tool-call-parser deepseekv4 \
  --host 0.0.0.0 --port 30000 > /mnt/ssd/server.log 2>&1 &'
# 权重加载 <1min（Local SSD 满速），autotune+DeepGEMM warmup(32768)+CUDA graph ~12min 到 ready
# 冒烟想快：加 --disable-flashinfer-autotune
kubectl exec v4-pro -- bash -c 'until curl -sf -m3 localhost:30000/health; do sleep 15; done; echo READY'
kubectl exec v4-pro -- curl -s localhost:30000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"deepseek-ai/DeepSeek-V4-Pro","messages":[{"role":"user","content":"Capital of France? one word"}],"max_tokens":10}'
# → "Paris" ✓
```

#### Step 5 — Benchmark 8K/1K（口径：total in+out ÷ 4 GPU，warm）

```bash
kubectl exec v4-pro -- bash -c 'cd /sgl-workspace/sglang
# warmup
python -m sglang.bench_serving --backend sglang-oai --port 30000 \
  --model deepseek-ai/DeepSeek-V4-Pro --dataset-name random \
  --random-input-len 8192 --random-output-len 1024 --random-range-ratio 1.0 \
  --num-prompts 16 --max-concurrency 8 >/dev/null 2>&1
for C in 1 16 64; do echo "=== conc=$C ==="
  python -m sglang.bench_serving --backend sglang-oai --port 30000 \
    --model deepseek-ai/DeepSeek-V4-Pro --dataset-name random \
    --random-input-len 8192 --random-output-len 1024 --random-range-ratio 1.0 \
    --num-prompts $((C*4)) --max-concurrency $C 2>&1 \
    | grep -iE "Total token throughput|Median TTFT|Median TPOT|^Concurrency"
done'
```

---

### Phase 3：PD-disagg + MegaMoE W4A4 + SWA（冲吞吐）
- 拓扑参考官方 `10P1D-dep4-dep32`（10 prefill 各 TP4/DP4/EP4 + decode DEP32 8节点）。**我们可先缩小到 4P1D 或 8P + DEP32 试**（跟 R1 Round4 同规模，省节点）。
- decode 关键参数（来自官方 recipe）：
  ```
  --moe-a2a-backend megamoe --enable-dp-attention --enable-dp-lm-head \
  --tp-size 32 --dp-size 32 --ep-size 32 --swa-full-tokens-ratio 0.20 \
  --context-length 9216 --mem-fraction-static 0.94 \
  --max-running-requests 18432 --cuda-graph-max-bs 1280 \
  --disaggregation-mode decode --disaggregation-transfer-backend mooncake
  ```
- W4A4 MegaMoE env：`SGLANG_OPT_DEEPGEMM_MEGA_MOE_USE_FP4_ACTS=1 SGLANG_OPT_DEEPGEMM_MEGA_MOE_USE_MXF4_KIND=1 SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK=8192`
- SWA env：`SGLANG_OPT_SWA_SPLIT_LEAF_ON_INSERT=1 SGLANG_OPT_SWA_EVICT_DROP_PAGE_MARGIN=1 SGLANG_OPT_USE_ONLINE_COMPRESS=1`
- + mooncake NVLINK（同 R1）。
- **编排**：先用 sglang_router（我们熟）跑通；要严格复现官方再上 Dynamo。

### Phase 4：Benchmark 8K/1K + 三方对比
- 同口径（total in+out /GPU，random 8192/1024）：
  | 对象 | TPS/GPU |
  |---|---|
  | 我们 R1 8K/1K（已测）| 1,359 |
  | 我们 V4 8K/1K（本计划目标）| ? |
  | 官方 V4 Pro 8K/1K | 11,200 |
- 逐步开关做消融：先 no-MegaMoE → +W4A4 → +MTP，看每步收益。

---

## 4. 关键参数 / env 速查（来自 cookbook + 官方 recipe）

| 类别 | 参数 / env | 说明 |
|---|---|---|
| MegaMoE W4A4 | `--moe-a2a-backend megamoe` + `USE_FP4_ACTS=1` + `USE_MXF4_KIND=1` | Blackwell only，仅 high-throughput 配方；别手动设 `--moe-runner-backend` |
| SWA | `--swa-full-tokens-ratio 0.20` + `SGLANG_OPT_SWA_*` | 滑动窗口，KV 打薄的核心 |
| online compress | `SGLANG_OPT_USE_ONLINE_COMPRESS=1` | C4/C128 压缩态 |
| compress dtype | `SGLANG_DSV4_COMPRESS_STATE_DTYPE=bf16` | 默认 fp32；bf16 省显存塞更多 slot |
| MTP | `--speculative-algorithm EAGLE --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4` | low-latency；high-throughput 时关掉（verify 成本 > 收益）|
| NVFP4 ckpt | `--moe-runner-backend flashinfer_trtllm_routed` | nvidia NVFP4 checkpoint 必须 |
| MNNVL | `MC_FORCE_MNNVL=1 NCCL_MNNVL_ENABLE=1 NCCL_CUMEM_ENABLE=1` | GB300 cross-pod KV over NVLink |
| KV offload | HiCache L2/L3（GPU→CPU→storage）| 多轮/长上下文才需要，冲吞吐不用 |

---

## 5. 已知坑（cookbook 明列，开跑前记牢）

1. **DeepEP dispatch buffer 约束**：必须 `max-running-requests × MTP_draft_tokens ≤ SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK`，否则稳态负载炸 `deep_ep.cpp:1105`。调并发时三个值一起动。
2. **镜像版本**：v0.5.15.post1（R1 用的）**不支持 V4**，必须换 `latest` 或 nightly。换了要重验 sm_103a。
3. **mooncake 走 NVLink**：GB300 cross-pod 报 `nvlink_transport.cpp:497 ... not found` 就加 MNNVL 三连（同 R1 的坑）。
4. **MegaMoE 限制**：只在 Blackwell + high-throughput 配方生效；low-latency/balanced 看不到。
5. **生成器默认值保守**：cookbook 说默认 `max-running-requests`/`cuda-graph-max-bs` 偏保守，要往实际峰值并发调。
6. **模型大**：V4-Pro 1.6T FP4 ~800G，内存盘（tmpfs）要够；单节点 4×277G HBM 放权重没问题，但 tmpfs 放 checkpoint + pod 内存要算好（参考 R1 内存坑）。

---

## 6. 开跑前 checklist

- [ ] **用干净节点池**（无 `mnt-disks-ssdN.mount` 残留污染；R1 实测 pool-0007 干净）
- [ ] **Local SSD RAID 就位**：`gke-raid-disks` DaemonSet 部署，逐节点 `grep -c md0 /proc/mdstat` 全 =1
- [ ] pod ssd 卷 = hostPath `/mnt/disks/raid/0`（HostToContainer），内存 request 600Gi
- [ ] 确认 `lmsysorg/sglang:latest`（或 nightly-dev-cu13）含 sm_103a
- [ ] 集群池有空闲节点（Phase1 只需 1 台；Phase3 需 ~9-18 台）
- [x] V4-Flash / Pro NVFP4 checkpoint 备份到 GCS（`gs://chrisya-gb300-models/DeepSeek-V4-Flash-NVFP4` 168G / `-Pro-NVFP4` 913G），bootstrap 时 `gcloud cp` **到 Local SSD `/mnt/ssd`**
- [x] Phase 1（Flash）+ Phase 2（Pro）单节点冒烟 + 压测通过 → 下一步 Phase 3 规模
- [ ] benchmark 用同口径（total in+out /GPU，8K/1K，warm 值）；`input-len +1 BOS` 且 `input+output ≤ context-length`

---

## 7. Benchmark 汇总报告（实测，2026-07-20）

**统一口径**：SGLang `bench_serving`，random 数据集 **8192 input / 1024 output**（`--random-range-ratio 1.0` 固定长度），warm（先 warmup 再测），单节点 4 GPU。`tok/s/GPU = Total token throughput(in+out) ÷ 4`。

### 7.1 三方对比（conc=64 峰值）

| 部署 | 模型 | 规模 | Total tok/s | **tok/s/GPU** | conc1 TPOT | 相对 |
|---|---|---|---|---|---|---|
| 本次 V4-Flash | 284B / 13B 激活 | 单节点 4 GPU | 34162 | **8540** | 5.71 ms | 基准 |
| 本次 V4-Pro | 1.6T / 49B 激活 | 单节点 4 GPU | 11177 | **2794** | 10.44 ms | Flash 的 0.33× |
| 我们 R1（前期）| 671B 全注意力 | 64 GPU 8P+DEP32 | — | 1359 | — | Flash 的 0.16× |
| 官方 V4-Pro | 1.6T | **18 节点 PD** + MegaMoE W4A4 | — | 11200 | — | 本次 Pro 的 4.0× |

### 7.2 逐并发明细

**V4-Flash（单节点 TP4）**
| 并发 | Total tok/s | tok/s/GPU | Output tok/s | Median TPOT | Median TTFT |
|---|---|---|---|---|---|
| 1 | 1520 | 380 | 169 | 5.71 ms | 220 ms |
| 16 | 12477 | 3119 | 1386 | 8.06 ms | 1709 ms |
| 64 | 34162 | **8540** | 3796 | 13.0 ms | 3102 ms |

**V4-Pro（单节点 TP4）**
| 并发 | Total tok/s | tok/s/GPU | Output tok/s | Median TPOT | Median TTFT |
|---|---|---|---|---|---|
| 1 | 838 | 209 | 93 | 10.44 ms | 295 ms |
| 16 | 7594 | 1898 | 844 | 16.55 ms | 2193 ms |
| 64 | 11177 | **2794** | 1242 | 33.74 ms | 9845 ms |

### 7.3 结论

1. **Flash vs Pro 差 3.1×**（8540 vs 2794 tok/s/GPU @conc64），符合模型代差：Pro 总参大 5.6×、激活大 3.8×，单 token 计算量更大。
2. **Flash 单节点就碾压 R1 64 卡 6.3×**（8540 vs 1359）——V4 架构（CSA+HCA 打薄 KV + SWA）在同 workload 下每 token 效率远超 R1 全注意力。
3. **Pro 单节点距官方 11,200 差 4.0×**——差距全在部署形态：官方是 18 节点 PD-disagg（prefill 独立扩展消化长 input）+ MegaMoE W4A4（激活也 4bit，矩阵乘快 ~2×）+ SWA。单节点 4 卡扛 1.6T 的 prefill，conc64 TTFT 已到 9.8s（瓶颈在 prefill），这正是 Phase 3 要解的。
4. **交互性**：Pro conc1 TPOT 10.44ms ≈ 96 tok/s/user，Flash 5.71ms ≈ 175 tok/s/user，都远快于人眼阅读速度，单用户体验流畅。

---

*2026-07-20（Local SSD based）。Phase 1（Flash）+ Phase 2（Pro）单节点 TP4 已实测通过 + 压测，§3.9 为可复现手册、§7 为 benchmark 报告。存储全程 Local SSD RAID（不用内存盘，见 gb300-local-ssd-raid0-SETUP.md）。R1 端到端见 `../deepseek-v3/`。下一步 Phase 3 = Pro + MegaMoE W4A4 + SWA + PD-disagg（官方 10P1D-dep4-dep32 / Dynamo）冲 11K。*
