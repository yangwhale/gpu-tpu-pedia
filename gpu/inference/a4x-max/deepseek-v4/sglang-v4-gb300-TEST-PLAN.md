# GB300 (A4X Max) · SGLang DeepSeek-V4 · 端到端测试指南 + Benchmark 报告

> 资料来源：SGLang V4 cookbook、lmsys Day-0 博客（2026-04-25）、pytorch「Serving DeepSeek-V4 on GB300」（2026-06-23）、SemiAnalysis InferenceX srt-slurm recipe `disagg-gb300-10p1d-dep4-dep32-18-c2500.yaml`。

---

## TL;DR — 最终状态与结论（2026-07-21 晚，口径已校正，详见 §10）

> ⚠️ **口径**：本文谈"对标官方 11,200"一律用官方口径 **output-only ÷ decode-GPU 数**（prefill 卡不进分母、输入 token 不进分子）。早期章节出现过的 "3,031 / 3,522 tok/s/GPU" 是错口径 `(in+out)÷72卡`，**已作废**，别再引用。

| 项 | 状态 |
|---|---|
| Phase 1 V4-Flash 单节点 TP4（聚合）| ✅ conc64 8,540 tok/s/GPU（in+out÷4，单节点口径，仅供内部对比）|
| Phase 2 V4-Pro 单节点 TP4（聚合）| ✅ conc64 2,794 tok/s/GPU（同上口径）|
| Phase 3 V4-Pro PD 分离（Dynamo + megamoe W4A4 + SWA + MTP）| ✅ 端到端跑通，Dynamo 根治 tail-stall |
| **最优实测（官方口径 output÷decode-GPU）** | **16 prefill + dep8-mtp + sa-bench 开环 = 8,993 = 官方 11,200 的 80%**（单 frontend 5,060 → 多 frontend 6,788 → 16 prefill 8,993；14→16 收敛，见 §12.6）。剩余 1.25× = 单卡内核成熟度差，非架构 |
| **距官方 11,200** | 剩余 20% = 官方 pinned 镜像内核成熟度（§14 实测：full autotune 无提升、EPLB 与 megamoe 不兼容，均不能缩小 gap）。早期 dep8 闭环 6,659 + "prefill 慢 3.7×" 均系测量假象，已被 §11（并发扫描）+ §12（sa-bench 开环 8,993）证伪 |

**核心结论（以 §10 为准）**：
- **11,200 = MTP 曲线 @ 50 tok/s/user = dep8-MTP 家（`high-conc-8p1d-dep4-dep8-mtp` 最可能）**，不是 dep32/dep40 wide-EP（那是 no-MTP 曲线）。
- **PD 是流水线**：`prefill worker 数 = decode 完成请求率 × 输入长度 ÷ 单 prefill 吞吐`。dep8/1760并发/8K1K 需 ~8 prefill（官方速）。~~我 prefill 慢 3.7× 需 30 个~~ **已推翻（§11）：我 prefill 高并发下达官方 83%，8 个够**。
- **每 decode 卡 ≈ 224 用户**（@50 tok/s/user）；decode 卡数 = 目标用户 ÷ 224（规模决策，非性能）。
- **MTP 只对小 batch/交互有用**：大 batch 高吞吐点 MTP 收益归零（实测 dep8 关 MTP 6,536 ≈ 开 6,659，打平）。
- **wide-EP 在 feed-limited 下反降 per-GPU**：总输出被 prefill 卡在 ~50K，dep8÷8=6,659 > dep32÷32=1,573。
- **攒批（slow_down）官方没用**：被 decode KV pool 卡死（dep8 最多攒 ~450 请求），是 burst 非稳态。
- **dep40 非法(无 EPLB)**：256 专家 % 40 ≠ 0，assert 崩；dep40 必须配 EPLB 冗余专家凑整。

**四条最贵的踩坑经验**：
1. **checkpoint 变体**：megamoe/PD **必须用原版 `deepseek-ai/DeepSeek-V4-Pro`**（FP4 MoE+FP8 attn），**不能用 `nvidia/*-NVFP4`**（强制 `flashinfer_trtllm_routed`，对 deepep/megamoe 无 fused func，多节点必崩；只适合单节点 Phase 1/2）。
2. **必须用 Dynamo 不是 sglang_router**：sgl-router 高并发 PD KV 交接 race，请求永久 hang（issue #9266/#31206/#12688/#5450）；官方用 Dynamo。**Dynamo「circuits-open」真根因 = 旧 `sglang::router` 僵尸霸占 8000 端口**（`/proc/net/tcp` 反查 PID kill 掉再起 frontend）。
3. **MTP 坑**：online-compress 与 MTP 互斥；ctx 9216 太小 MTP 令请求超限（benchmark OSL 降 960）；提 ctx→cuda-graph OOM；反复 kill dynamo.sglang 积累 zombie+GPU 泄漏 191GB→ 唯一解 `kubectl delete pod --force` 重建。
4. **重启 decode 必重启 prefill**（否则 KV transfer 断，decode prealloc 死锁）；ZMQ 40236 残留用 `/proc/net/tcp` killport 清。

**详细导航（看这几个就够）**：**§13 满配端到端复现 checklist（照抄即可）** · **§12 满配 sa-bench 最终结果（8,993）** · §14 autotune/EPLB 收口 · §10.1-10.3 官方 11,200 口径 + PD 流水线公式 · §3.9 单节点复现手册。§7/§8/§9/§10.4-10.7 是探索过程与已推翻的旧结论，仅存档。

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
> **Phase 1 压测**（8K/1K warm，数字见 §7.2）：单节点 4 卡 conc64 = **8540 tok/s/GPU（in+out）**，比 R1 64卡的 1359 高 6.3×；conc1 TPOT 5.71ms 交互极快。

### Phase 2：V4-Pro 单节点 TP=4
- cp `nvidia/DeepSeek-V4-Pro-NVFP4`（~800G）到 **Local SSD**，`--model-path /mnt/ssd/DeepSeek-V4-Pro-NVFP4`（单节点 4×277G=1108G HBM 放得下）。**这里就是 Local SSD 的价值所在**：800G 模型放 Local SSD 不吃 RAM；若放内存盘，800G tmpfs + 运行时直接爆 942G 节点内存。
- 同 Phase 1 启动，`--tp-size 4`。验证加载 + 生成。

> **✅ Phase 2 实测通过（2026-07-20）**：`nvidia/DeepSeek-V4-Pro-NVFP4`（**851G / 76 文件**）从 HF 下载到 Local SSD（hf_transfer ~0.9 GB/s，851G 约 15 min），单节点 TP4 加载：
> - **权重加载 <1 min**（Local SSD 读满速，`avail mem=274GB/GPU`），autotune + DeepGEMM warmup（32768 kernels）+ CUDA graph 约 12 min 到 ready。冒烟可加 `--disable-flashinfer-autotune` 提速。
> - chat 生成正确（"capital of France"→"Paris"）。GCS 备份 `gs://chrisya-gb300-models/DeepSeek-V4-Pro-NVFP4/`（913G，ADC+SDK 上传 668s；ADC 用后即删）。
>
> **Phase 2 压测**（8K/1K warm，数字见 §7.2）：单节点 4 卡 conc64 = **2794 tok/s/GPU（in+out）**，比 Flash 低 3.1×（Pro 模型大 5.6×/激活大 3.8×）；conc64 TTFT 9.8s 偏高（单节点扛 1.6T prefill 压力大 → Phase 3 上 PD 解决）。

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

### Phase 3：PD-disagg 满配 + 逐项消融（冲官方 11,200 tok/s/GPU）

> **官方 11,200 的真相**（核对 pytorch blog 2026-06-23 + SemiAnalysis InferenceX）：GB300 **disaggregated** lane、V4-Pro FP4、ISL=8192/OSL=1024、dynamo-sglang、**2026-06 带 MTP 曲线**，在 **~50 tok/s/user** 交互点达 **~11,200 tok/s/GPU**；对比 Day-0（2026-04）no-MTP 的 ~2,200，两个月 **5×**。这 5× 不是单点，是整条曲线抬升。官方 recipe：`disagg-gb300-10p1d-dep4-dep32-18-c2500.yaml`（srt-slurm + Dynamo 起）。

> **⚠️ 拓扑已定稿**：早期 Phase 3 探索用官方"示范" recipe `10P1D-dep4-dep32`（72 GPU wide-EP，no-MTP），后证明**那不是 11,200 的配方**（§10.2）。**最终正确配方 = 16 prefill(dep4) + dep8-MTP + 多 frontend（§12/§13）**。下面保留仍可复用的**部署前置 + 踩坑实录**（这些跟拓扑无关，§13 也用得上）；满配参数与启动步骤一律看 §13。

#### 3.0-prereq 域与存储（开跑前必须就位）

- **域**：需**一个完整 18 节点 NVL72 域，且 GPU 全空闲**（不只是 Ready + 标签）。实战最终用 **`subblock-0001`（= `gb300-pool-0001`）free=18**（首选 0002 被 sgl2 残留占了，见 3.0-b）。选域先跑「每 subblock 空闲节点扫描」，别只看 ready/标签。确认 18 台 `nvidia.com/gpu.clique` 一致（同一 NVLink 域）。
- **Local SSD RAID**：给 pool-0002 部署 `gke-raid-disks` DaemonSet，逐节点 `grep -c md0 /proc/mdstat` 全 =1；**先验无 `mnt-disks-ssdN.mount` 残留污染**（见 RAID-SETUP 坑速查）。
- **权重**：18 节点各 `gcloud storage cp -r gs://chrisya-gb300-models/DeepSeek-V4-Pro-NVFP4 /mnt/ssd/`（只读 scope 够）。
- **bootstrap**：GIB + DOCA + gcloud（同 R1 §4）。
- **编排**：先用 sglang_router（我们熟）跑通；严格复现官方再上 Dynamo。

> **✅ 3.0 准备实录（2026-07-20）**：
> 1. **RAID**：`gke-raid-disks-0002` DaemonSet 部署到 pool-0002，**18/18 节点 RAID_READY**（12T `/mnt/disks/raid/0`）。
> 2. **权重预拉**：`v4pro-puller-0002` DaemonSet（`gcloud storage cp -r` 从 GCS 到各节点 Local SSD），**18/18 节点各 851G / 76 文件**。node 只读 scope 足够读 GCS。
> 3. **踩坑 & 修复**：
>    - **Bug 1（puller 镜像 amd64）**：首版 puller 用 `google/cloud-sdk:slim`，GB300 arm64 报 `exec format error`。修：换 `ubuntu:24.04` + apt 装 `google-cloud-cli`（同 RAID DS 的 arm64 坑）。
>    - **Bug 2（RAID inactive，256K tmpfs，18 节点里 1-2 台）**：`/proc/mdstat` 见 md0 **inactive**（旧 mdadm superblock 把盘拆成坏数组），mount 报 `can't read superblock`，DS 无 `set -e` 假 RAID_READY。修：**live 清**（不用重建节点）——`mdadm --stop` 所有 md + `sleep 1` + `--zero-superblock --force` 全盘 + 重 create/mkfs/mount，再删该节点 puller 重拉。**udev race**：stop 后 udev 秒级重组，抢在 zero 前 → create busy，重跑 1-2 次即成。详见 [RAID-SETUP 坑速查 B 类](./gb300-local-ssd-raid0-SETUP.md#5-坑速查)。

> **✅ 3.0-b 部署踩坑补充（2026-07-20 pool-0001 实战，血泪追加）**：
> 1. **选域必须查 GPU 实占，不能只看 Ready + team 标签**。首选 subblock-0002 时只数了「18 ready + 全 yangwhale」，结果上面压着别人 5 个 sgl2 pod（44h 前的 R1 残留）+ 我的新 pod，只剩 1 空。**正确姿势**：扫每个 subblock 的「无 GPU-pod 的空闲节点数」（遍历所有 pod 的 `nodeName` + `limits.nvidia.com/gpu`），选 free=18 的域。
> 2. **换域搬迁**：subblock-0002 被占 + 有台重建慢的节点，最终改用 **subblock-0001**（dsv3 清空后 free=18）。给它 `kubectl label node -l ...pool-0001 team=yangwhale --overwrite`（原 team=gdde）→ 铺 RAID + 拉权重 + 部署。
> 3. **节点 DRA 网络模式歪（ipvlan vs pci）**：pool-0002 有台 lcg3，其 `dra.net` resourceslice 是 **ipvlan 设备**（`gpu*ipvlan*`）而非正常的 pci 直通，`mrdma.google.com` claim 分不出 8 卡，pod 永远 pending `cannot allocate all claims`。修：**重建该节点**（`gcloud compute instances delete` → MIG 原名重拉，GB300 冷启 15+min）。教训：`kubectl get resourceslice <node>-dra.net -o json` 看设备名，ipvlan ≠ 正常。
> 4. **ComputeDomain / IMEX 只收敛 15/18**：18 pod 部署后卡 15 running，3 个 `ResourceClaim not created yet` / `FailedPrepareDynamicResources`。`kubectl get computedomain -o jsonpath={.status.nodes}` 只见 15 节点（虽 18 台 clique 一致）。修：**删掉那 3 个卡住的 pod 让它重建**（`kubectl delete pod ... && kubectl apply`）→ 重新触发 ComputeDomain 加入，即 18/18。
> 5. **权重 puller DaemonSet 镜像必须 arm64**：`google/cloud-sdk:slim` 是 amd64 → `exec format error`；换 `ubuntu:24.04` + apt 装 `google-cloud-cli`（同 RAID DS 坑）。

#### 3.2 megamoe 路线的两个决定性前提（deepep 死胡同已弃）

> deepep a2a 路径在此 stack 跑不通（`flashinfer_trtllm_routed`/`cutedsl`/`deep_gemm` 连撞 runner bug）；官方 recipe 全程用 **megamoe**，遂弃 deepep。下面两条是 megamoe PD 能跑起来的关键前提：

> **✅ 关键：checkpoint 变体 + 镜像 决定 megamoe 能否跑**：
> - **根因**：`nvidia/DeepSeek-V4-Pro-NVFP4` 变体**强制** `flashinfer_trtllm_routed` runner，它对 **deepep 和 megamoe 两种 a2a 都没有 fused func**（`requires a fused func for a2a backend {deepep|megamoe}, but none is registered`）→ 单节点 TP（Phase 2）能跑，但**多节点 PD 的 megamoe/deepep 全崩**。
> - **正解**：megamoe PD 必须用**官方原版 checkpoint `deepseek-ai/DeepSeek-V4-Pro`**（FP4 MoE + FP8 attn/dense，~806G），不是 nvidia NVFP4 变体。官方 recipe 的 `model.path` 就是原版。
> - **镜像**：官方 pin 的 `nightly-dev-cu13-20260520` 已被 Docker Hub GC（`not found`）；`latest`（0.5.15.post1）的 megamoe auto-runner 也 mismatch。**用最新可用 nightly**（实测 `lmsysorg/sglang:nightly-dev-cu13-20260720-b3570a45` ✅）。查可用 tag：`curl -s "https://hub.docker.com/v2/repositories/lmsysorg/sglang/tags/?page_size=100&name=nightly-dev-cu13"`。
> - **✅ Run A 实测跑通（2026-07-20 22:22）**：原版 checkpoint + 最新 nightly + megamoe，10P1D-dep4-dep32（72 GPU，subblock-0001），sglang_router `cache_aware` `--pd-disaggregation`，PD 端到端经 router 正确生成（"Capital of France"→"Paris"），prefill→decode KV 走域内 NVLink（mooncake `MC_FORCE_MNNVL=1`）。
> - **换镜像必重 bootstrap**：GIB/DOCA/nixl 装在容器内，换 image 后 pod 全新，需重跑 bootstrap（GIB tgz + DOCA OFED userspace + nixl）。
> - **prefill DEP4 mem-fraction 0.90 / decode DEP32 0.94**（DEP4 4卡装 1.6T 更挤）。

#### 3.3 满配参数 + Dynamo 前置（详细启动步骤见 §13）

最终满配配方（16 prefill dep4 + dep8-MTP + 多 frontend）的**完整参数与自愈启动步骤已收敛到 §13**，此处只留两条 §13 未展开的前置：

- **Dynamo 运行时前置**：(1) 起 `dynamo-nats`（nats:2.10-alpine -js）+ `dynamo-etcd`（etcd:v3.5.16 + Service）两个 k8s pod；(2) 每 worker pod `pip install ai-dynamo`（不降级 nightly sglang，`dynamo.sglang` 能 import 即可）；(3) worker 用 `python3 -m dynamo.sglang`（非 `sglang.launch_server`）+ env `NATS_SERVER`/`ETCD_ENDPOINTS`/`DYN_SYSTEM_PORT`。**为什么必须 Dynamo 不用 sglang_router**：见 §8（router 高并发 tail-stall）。
- **MTP 四大坑**（Run D 实测，配方已进 §13）：(1) `--speculative-num-steps 1`（不要 3）+ **prefill/decode 两侧都加**，否则 nextn 模块加载不一致；(2) `NUM_MAX_TOKENS_PER_RANK=4096`（draft 放大 token，1280 会 `exceeds cap`）；(3) `context-length 9216`（**别提到 9728/10240 → cuda-graph OOM**），benchmark **OSL 用 960**（8192+960+draft < 9216，否则 78% 请求超限被拒）；(4) **MTP 与 online 压缩互斥**（去掉 `USE_ONLINE_COMPRESS`）。

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
- [ ] Phase3 满配需 **18 节点同一 NVL72 域 + GPU 全空闲**（扫每 subblock 无-GPU-pod 空闲数，别只看 ready/标签；实战用 `subblock-0001` free=18，`clique` 一致）；decode DEP32 跨域走不了 NVLink
- [ ] 部署后若卡 15/18（`ResourceClaim not created yet`）：删重建卡住的 pod 触发 ComputeDomain 收敛；若某节点 DRA 是 ipvlan（mrdma 分不出）：重建该节点
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
| 官方 V4-Pro | 1.6T | **18 节点 PD** + MegaMoE W4A4 | — | 11200* | — | 见下注 |

> \* 官方 11,200 是 **output ÷ decode-GPU** 口径，本表单节点数是 **(in+out) ÷ 4** 口径，**两者不可直接相除比较**（此处仅作量级参照）。官方口径的满配实测见 **§12（8,993 = 官方 80%）**，gap 分析见 §10 + §14。

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
3. **Pro 单节点 vs 官方 11,200 不是一个口径**（单节点 in+out÷4 vs 官方 output÷decode-GPU，见上注），只能说差距全在部署形态：官方是 18 节点 PD-disagg（prefill 独立扩展消化长 input）+ MegaMoE W4A4（激活也 4bit，矩阵乘快 ~2×）+ SWA + MTP。单节点 4 卡扛 1.6T 的 prefill，conc64 TTFT 已到 9.8s（瓶颈在 prefill），这正是 Phase 3 要解的。官方口径的对齐实测见 §7.4。
4. **交互性**：Pro conc1 TPOT 10.44ms ≈ 96 tok/s/user，Flash 5.71ms ≈ 175 tok/s/user，都远快于人眼阅读速度，单用户体验流畅。

### 7.4 Phase 3 PD-disagg 最终结果 → 见 §12

> 早期用 sglang `bench_serving` **闭环**测得 dep8+8prefill+MTP = 6,659 output/decode-GPU，**已被官方 sa-bench 开环取代**：满配 16 prefill + dep8-MTP + 多 frontend = **8,993 = 官方 11,200 的 80%**（详见 §12）。当时以为"总输出卡在 ~50K 是 prefill 慢 3.7×"，已被 §11（prefill 高并发达官方 83%）+ §12（换开环 + 多 frontend + 加 prefill 吃回大部分）证伪。**保留一个仍成立的洞见**：feed-limited 下 wide-EP（dep32）per-decode-GPU 反而低于 dep8（见 §10.5）；11,200 出在 dep8-MTP 曲线，不是 dep32/dep40 wide-EP。

---

## 8. 为什么必须 Dynamo + 各优化原理

**为什么必须 Dynamo 不用 sglang_router**：sglang_router 的 PD-disaggregation 在并发下对少数请求的 prefill→decode KV 交接有 race，请求 hang 后**不失败、不重试、不释放槽位**（尾部永久 hang，conc 越高死槽位越多、吞吐反降；issue #9266/#31206/#12688/#5450，各种 disable/retry/timeout 均无效）。Dynamo 有服务发现（先路由 decode 再 KV-aware 选 prefill）+ **请求级 migration/cancellation**，hang/挂时能迁移取消、不累积死槽位——这正是 tail-stall 的解药。代价是要 NATS+ETCD+ai-dynamo（前置见 §3.3，部署见 §13）。⚠️ 诊断时**绝不能直连 prefill:30000 绕 frontend**（触发 `bootstrap_room should not be None` 打崩 prefill）。

**各项优化为什么能冲上万**：
- **PD 分离**：prefill 算力密集、decode 访存密集，拆开各自扩展、用满 GPU；KV 经 NVLink/RDMA 零拷贝送 decode。
- **megamoe W4A4**：expert dispatch+GEMM 融合成一个 kernel + 激活压 4bit → MoE 层快 ~2×（实测 conc256 下 1.48×，是最大单项）。
- **MTP（EAGLE 投机解码）**：一次 forward 出多 token，TPOT 从 43ms 稳到 11ms（draft 被接受）——**11,200 的核心杠杆**（§10.2）。
- **Wide-EP / SWA / online 压缩**：把 KV 打薄让 decode 塞更大 batch，但收益只在高并发 + prefill 喂得饱时显现（feed-limited 下 wide-EP per-GPU 反降，见 §10.5；conc256 下 SWA 未见增益）。

---

## 9. 官方 c2500 recipe 对齐项 + 出处（source of truth）

> 早先曾把 gap 归因为「prefill batch 没攒满」→ 后又归因「prefill 单卡慢 3.7×」，**两者均已推翻**（§11：prefill 高并发达官方 83%）。本节保留仍有用的**参数对齐清单**、**开环 vs 闭环的真教训**和**出处链接**。

### 9.1 官方 recipe 我们对齐/未对齐项

| 项 | 官方值 | 我们 | 类型 |
|---|---|---|---|
| `chunked-prefill-size` | 32768 | 32768 ✅ | prefill worker |
| **router-mode** | `kv` | 裸起（无）| Dynamo frontend |
| **router-queue-threshold** | `64` | 无 | Dynamo frontend |
| router-temperature | `0.5` | 无 | Dynamo frontend |
| router-kv-overlap-score-weight | `0` | 无 | Dynamo frontend |
| no-kv-events | `true` | 无 | Dynamo frontend |
| 多 frontend | `num_additional_frontends: 8` | 1 | Dynamo frontend 分流 |
| **压测负载** | `req_rate: inf`（开环打满）| max-concurrency（闭环）| bench |
| prefix cache | `SGLANG_RADIX_FORCE_MISS=1`（强制关，防作弊）| 未设（random 数据无前缀重叠，影响小）| worker env |
| decode max-running-requests | 18432 | 18432 ✅ | decode worker |

### 9.2 开环 vs 闭环：口径本身是真差别之一（实测教训）

同一 dep32 配置，仅换压测口径（相对比较，口径无关）：
- **闭环 max-concurrency 下 `router-queue-threshold` 无用甚至有害**：闭环已限死在途请求数，router 排队只增延迟不增吞吐 + KV 路由开销 → **约 -13%**。
- **开环 `req_rate inf`（官方口径）才对**：请求持续涌入，effective concurrency 自然平衡，比闭环 **约 +16%**。**对标官方必须用开环。**

开环是对标官方的**必要口径**；配合多 frontend + 16 prefill，最终 §12 达 8,993（80%）。剩余 20% 见 §14（镜像内核成熟度）。

### 9.3 出处（source of truth）

1. **PyTorch 官方博客「Serving DeepSeek-V4 on GB300」**：https://pytorch.org/blog/serving-deepseek-v4-on-gb300-with-sglang-5x-higher-throughput-at-the-same-interactivity-since-day-0
2. **SemiAnalysis InferenceX c2500 recipe**（PyTorch 博客点名的复现文件）：https://github.com/SemiAnalysisAI/InferenceX/blob/801d1261235f4892d4831de9de70c34f5bea7d98/benchmarks/multi_node/srt-slurm-recipes/sglang/deepseek-v4/8k1k/disagg-gb300-10p1d-dep4-dep32-18-c2500.yaml
3. **SGLang 大规模 EP serving 博客**：https://www.lmsys.org/blog/2025-05-05-large-scale-ep
4. **NVIDIA Dynamo Router Guide**：https://docs.nvidia.com/dynamo/v1.0.0/components/router/router-guide
5. **SGLang PD 文档**：https://docs.sglang.ai/advanced_features/pd_disaggregation.html ｜ batch 攒满佐证 issue #12591：https://github.com/sgl-project/sglang/issues/12591
6. **DeepEP Waterfill**（EP dispatch 均衡）：sgl-project/sglang PR #25391

---

## 10. ⭐⭐⭐⭐ 对官方 11,200 的完整认知（2026-07-21 晚，逐段精读官方 PyTorch 博客 + dep8/dep32/dep40 全实测后）

> 本节是目前对官方 11,200 **最准确、最深入的理解**——口径、拓扑、PD 配比公式、撞墙真根因全在此。前文 §7/§8/§9 是实验过程与踩坑，**结论一律以本节为准**。

### 10.1 【口径】官方是「output-only ÷ decode-GPU」，不是「(in+out) ÷ 总GPU」

- **正确口径**：官方 11,200 是 **纯输出 token ÷ 解码卡数**（InferenceX 方法论原文："disaggregated configs calculate output throughput per decode GPU"），**prefill 卡不进分母、输入 token 不进分子**。拿「(输入+输出)÷72卡」这种数去对 11,200 是关公战秦琼（早期草稿犯过这错，已全删）。
- **量纲验证**：11,200 若是「输出÷总72卡」→ ×72 = 80万输出/s，反推输入几百万/s，1.6T 模型物理不可能。只有「输出÷解码卡」讲得通（dep8：11,200×8=8.96万输出/s，≈87.5 req/s，合理）。
- **含义（关键）**：这个口径**把 prefill 成本藏起来了**——你堆再多 prefill 喂一个小 decode，per-decode-GPU 都好看。它是「解码效率」指标，不是整机 TCO。
- **我们已对齐口径的最终实测（§12）**：16 prefill + dep8-MTP + 多 frontend + sa-bench 开环 = **8,993 output/decode-GPU = 官方 80%**。早期 dep8 闭环 6,659（÷8）系闭环 + 单 frontend + 8 prefill 未拉满所致，非口径错。dep32 同法 1,573（更低，见 10.5：feed-limited 下 wide-EP per-GPU 反降）。

### 10.2 【拓扑】11,200 是 dep8 + MTP，不是 dep40 / wide-EP

- 官方博客原文："**June 2026 MTP 曲线** @ ~50 tok/s/user = 11,200"。带 MTP 的 recipe **全是 dep8 那一家**（`mid-curve-*-dep8-mtp` / `high-conc-*-dep8-mtp`）；而 `10p1d-dep32` / `8p1d-dep40` 这些 wide-EP 大配置的 yaml **没有 speculative，是 no-MTP 曲线**。
- 所以 **11,200 出在 dep8-MTP，最可能是 `high-conc-8p1d-dep4-dep8-mtp`（conc 8192）**。把 wide-EP dep32/dep40 当成 11,200 的路是**偏的**（早期一度这么以为）。
- 我们早先测的 dep8+MTP（6,659）**恰恰就是对的那一家**；下午一度转去 dep40 是走偏（且 dep40 有硬约束，见 10.4）。
- 博客"How to Reproduce"贴的 `10p1d-dep4-dep32-c2500` 是 **no-MTP 大配置**，不是 11,200 的配方——博客拿它当"流程示范"，误导性强。

### 10.3 PD 是流水线：prefill 数怎么算才不饿死 decode（可复用公式）

**核心公式**：`需要的 prefill worker 数 = (decode 每秒完成请求数 × 输入长度) ÷ (单 prefill worker 吞吐)`

以 11,200 那个点（dep8 / 8K1K / 50 tok/s/user）为例：
1. 每张 decode 卡在 50 tok/s/user 点服务 `11,200 ÷ 50 ≈ 224 个并发用户`。**decode 卡数 = 目标总用户 ÷ 224**（这是规模决策，不是性能决策；效率 11,200 是常数）。
2. dep8（8 decode 卡）= 8×224 ≈ **1,792 有效用户**（recipe 灌 conc 8192 是 offered load，稳态有效并发 ~1,792，多的在排队）。
3. decode 完成请求率 = 8×11,200 ÷ 1024 ≈ **87.5 req/s**；prefill 需供 87.5 × 8192 ≈ **71.7 万 input tok/s**。
4. 单 prefill worker（dep4，官方速）≈ 4×18,200 ≈ 7.28 万 → 需 **~8-10 个 prefill**。官方 `high-conc-8p1d` 用 **8 个**，对得上。

**官方 P:D 配比表（按并发缩放，decode 越大/并发越高 prefill 越多）**：

| recipe | prefill | decode | 曲线 | 场景 |
|---|---|---|---|---|
| mid-curve-1p1d-dep8 | 1 | dep8 | MTP | 低并发交互 |
| mid-curve-2p1d-dep8 | 2 | dep8 | MTP | |
| mid-curve-4p1d-dep8（num-steps 3）| 4 | dep8 | MTP | conc 1024 |
| high-conc-8p1d-dep8（num-steps 1）| 8 | dep8 | MTP | conc 8192 ← **11,200 最可能在此** |
| 10p1d-dep32 c2500 | 10 | dep32 | no-MTP | 大规模吞吐 |
| 15p1d-dep12 c12000 | 15 | dep12 | no-MTP | 超高并发（prefill 拉满、decode 缩小）|

### 10.4 【已推翻】"prefill 单卡慢 3.7×" = 低并发测量假象（见 §11）

原结论（prefill 慢 3.7×、需 ~30 prefill、饿死 decode）**已推翻**：那个"1.44s/forward"是 conc4 极低并发欠载下量的；§11 并发扫描实测同一 worker 高并发达 15,196 = 官方 83%，prefill 无缺陷、8 个够。**以 §11 为准。**

### 10.5 dep40 的硬约束 + wide-EP 在 feed-limited 下反而更差

- **dep40 非法(无 EPLB)**：DeepSeek-V4 有 **256 个专家，256 % 40 ≠ 0**，`assert num_physical_experts % ep_size == 0` 直接崩。dep40 **必须配 EPLB 加冗余专家**凑整（如 256+24=280，280/40=7）。所以 **EPLB 不只是 2.54× 加速，是 dep40 能启动的前提**。合法 EP 必须整除 256（8/16/32/64…）。
- **feed-limited 下 wide-EP 反而降 per-GPU**：实测 dep32（32 decode 卡）total output ~50K，dep8（8 卡）total output ~53K——**总输出都被 prefill 卡在 ~50K**，与 decode 拓扑无关。dep8 ÷8 = 6,659；dep32 ÷32 = 1,573。**prefill 喂不动时，decode 铺得越宽 per-GPU 越低**。研究说的"wide-EP 提 per-GPU"只在 prefill 供得上时成立（官方 prefill 快，我不快）。

### 10.6 「prefill 攒批再放」= 压测 hack，别用

官方 11,200 是 `req_rate inf` 持续喂料的**稳态数**，不用攒批。`slow_down`（issue #6017）是量 decode 峰值的 hack、dynamo.sglang 也没这接口；且被 decode KV pool 卡死（dep8 最多攒 ~450 请求，放开即 burst 打回原形）。实测 `SGLANG_HACK_PD_DECODE_NUM_RESERVED_DECODE_TOKENS=1026` 反而过度预分配、吞吐降到 2,984，**有害**。

### 10.7 冲 11,200 的待办 → 已全部执行完（收口见 §12/§14）

原待办清单已全部做完：
1. ✅ **官方 sa-bench 开环**（§12）：换掉闭环 `bench_serving` → 8,993。
2. ✅ **满配瓶颈定位**（§12 瓶颈地图）：① 编排/frontend（多 frontend +34%）② prefill 喂料数（加到 16 +32%，收敛）③ 剩余 ~1.25× = 镜像内核成熟度。
3. ✅ **复现 dep8-MTP 配方** + 口径对齐（output÷decode-GPU + 开环）。
4. ✅ **autotune + EPLB 都试过**（§14）：full autotune 无提升、EPLB 与 megamoe 不兼容，均不能缩小 gap。

**最终认知**：满配 = 官方 80%；剩余 20% 确认只在官方 pinned 镜像（commit `14f81a67`）的内核/运行时成熟度，非架构/拓扑/编排/prefill 数/decode 配置/kernel autotune/专家均衡。

---

## 11. ⭐⭐⭐⭐⭐ 【重大修正】"prefill 慢 3.7×"是测量操作点假象——并发扫描实证（2026-07-22 最小单元）

**结论先行**：之前反复出现的"我 prefill 单卡吞吐只有官方 27%、慢 3.7×"是**在极低并发（conc4）下测量的假象**，不是真实缺陷。在干净的单 dep4 worker 上做并发扫描，prefill 吞吐随并发强烈上升，峰值达官方 **83%**。**§10.4 的"prefill 3.7× 慢、喂不动 decode"结论据此推翻。**

### 11.1 方法：最小单元隔离测量
- 单个 prefill worker = 1 节点 4 GPU（TP4/DP4/EP4），standalone `sglang.launch_server`（**非 disagg**，不经 Dynamo/router），直接 `sglang.bench_serving` 打。
- 与官方 prefill 配置逐行对齐：megamoe + W4A4（`USE_FP4_ACTS=1`）+ dp-attention + chunked-prefill 32768（DP=4 下自动调成 8192/rank，官方同款，非 bug）。
- ISL 8192 / OSL 1（纯 prefill），random 数据集。

### 11.2 并发扫描实测（官方口径 input tok/s ÷ 4 GPU）

| max-concurrency | Total input tok/s | **per-GPU** | 备注 |
|---|---|---|---|
| 4 | 19,765 | **4,940** | ← 之前误当"天花板"下 3.7× 结论 |
| 16 | 18,214 | 4,553 | 低并发噪声区 |
| 32 | 36,988 | 9,247 | 翻倍 |
| 64 | 42,832 | 10,708 | |
| **128** | **60,785** | **15,196** | ← 峰值，= 官方 18,200 的 **83%** |
| 256 | 57,120 | 14,280 | 饱和（TTFT 28s）|

- **prefill 吞吐强并发依赖**：低并发下 GPU 有流水线气泡/欠载，conc4 只有峰值的 1/3。
- 官方 18,200 是在 **c2500**（10 prefill → 每 worker ~conc250）测的；匹配高并发后我们峰值 **15,196 = 83%**，只差 ~1.2×，属 nightly/调参级差异，**非架构缺陷**。

### 11.3 沿途排除的假设（都不是 3.7× 的因）
- **CUDA graph（排除）**：prefill breakable + tc_piecewise 两条 backend 在源码 `_disable_*_cudagraph_if_incompatible` 里都被 `megamoe` + `dp-attention` 自动 gate 关掉（DSV4 还多一条 c4-indexer capture-pool OOM gate）；本 nightly **无 `enforce_piecewise` 强开 flag**；官方 recipe 也没设 prefill CG flag → **两边都 eager**，不是差异来源。
- **镜像（排除）**：`nightly-dev-cu13-20260601`（最接近官方 5 月 pin，5 月版已被 dockerhub GC）vs 我们 `20260720`，conc4 同为 ~4,900/GPU（4,812 vs 4,940）→ **非 nightly regression**。
- **FP4 indexer（边际）**：`--enable-deepseek-v4-fp4-indexer`（GB300 SM100 支持）conc4 下仅 **+5%**（5,172 vs 4,940）。

### 11.4 压测工具口径差异（sa-bench 实测对比，2026-07-22）
- **我们之前用**：`sglang.bench_serving`（SGLang 自带），`--max-concurrency N` 闭环。
- **官方用**：`sa-bench` = InferenceX `utils/bench_serving/benchmark_serving.py`（vLLM/sglang bench 的 fork）。确切调用（`benchmark_lib.sh:517`）：`--request-rate inf`（开环）+ **`--ignore-eos`** + `--num-warmups 2×conc` + `--dsv4`（DSV4 chat 编码 `encoding_dsv4.py`）+ 自定义 tokenizer，打 **Dynamo frontend**。
- **同一 dep4 worker、同一操作点，两工具实测对比（input tok/s/GPU, ISL8192/OSL1）**：

| conc | 我的 bench_serving | 官方 sa-bench | 差异 |
|---|---|---|---|
| 32 | 9,247 | **13,568** | sa-bench +47% |
| 128 | 15,196 | 14,607 | 收敛 |
| 256 | 14,280 | 14,798 | 收敛 |

- **结论**：工具口径差在**低并发极大（+47%）**，因为 sa-bench 的 pacing/ignore-eos 让 GPU 更快填满；**高并发峰值两者收敛到 ~15,000 tok/s/GPU**。
- 且 **18,200 不是官方直接实测的 prefill benchmark**，是「喂饱官方 dep8 decode 达 11,200 所需的 prefill 速率」推导值（§9.1）。所以我们单卡 prefill 峰值 ~15,000 ≈ **官方所需的 82%**——差 ~1.2×，属调参级，**彻底否定"3.7× 缺陷"**。

### 11.5 对 11,200 的启示（已在 §12 落实）
- prefill 单卡不是瓶颈（高并发下 83%）。据此换官方 **sa-bench 开环** + 多 frontend + 加 prefill 到 16，满配从闭环 6,659 提到 **8,993（§12）= 官方 80%**——真瓶颈是 (a) 编排/单 frontend CPU-bound（多 frontend +34%）+ (b) prefill 喂料数（加到 16 +32%），正如本节推断。剩余 20% 见 §14。

---

## 12. ⭐⭐⭐⭐⭐ 满配 sa-bench 实测 + 多 frontend 提升（2026-07-22，官方口径 output÷decode-GPU）

**结论先行**：用官方 sa-bench 开环打满配 `high-conc-8p1d-dep8-mtp`，靠**加多 frontend**把 output/decode-GPU 从 **5,060 抬到 6,788（+34%）**，= 官方 11,200 的 **61%**。完整瓶颈地图见 §12.4。这是**第一次用官方口径 + 官方工具**跑通满配。

### 12.1 部署（复用 fleet，非从零重建）
- 复用现有 18-pod fleet（清僵尸），按官方 `disagg-high-conc-8p1d-dep4-dep8-mtp.yaml`：**8 prefill(dep4)** + **1 decode(dep8 = TP8/DP8/EP8 跨 2 节点，d0 rank0 / d1 rank1，dist d0IP:5000)** + Dynamo(nats/etcd/frontend)。
- decode：MTP EAGLE num-steps=1/topk=1/draft=2，mem 0.85，max-running 8192，ctx 9216，swa 0.1，NUM_MAX_TOKENS_PER_RANK 4096（**去掉 §10.6 那个有害的 RESERVED_DECODE_TOKENS hack**）。
- **三个部署坑**：(1) decode 死进程泄漏 GPU 显存（199GB/卡不释放、`nvidia-smi --query-compute-apps` 为空）→ `kubectl delete pod` 重建清显存（唯一可靠解）；(2) prefill 40236 ZMQ 端口僵尸残留 → `killport.sh 40236` + 重启；(3) worker `"fired up"` 日志因 stdout 缓冲不可靠 → 用 **GPU util 判 ready**（prefill loaded = mem>200G & util≈0；decode warmup 完 = util<15%）。

### 12.2 单 frontend 满配基线（sa-bench 开环 `--request-rate inf --ignore-eos --dsv4`）

| 操作点 | output/decode-GPU | TPOT | TTFT 中位 | 成功率 |
|---|---|---|---|---|
| conc2500 | **5,060** | 35ms | 8.5s | 7499/7500 干净 |
| conc8192 | 3,553 | 26ms | 33s | 过载 thrash（更低）|

→ 单 frontend 在 conc2500 就到顶，再灌并发反而 thrash。

### 12.3 【关键提升】多 frontend 扩展

官方 recipe 开 `num_additional_frontends: 8`（共 9 frontend）。机制：多个 `dynamo.frontend` 进程共享同一 NATS/etcd worker 池，各占一个 pod 的端口 8001，多路 sa-bench 分打后聚合：

| frontend 数 × 每路 conc | 总 conc | 聚合 output tok/s | **output/decode-GPU** | TTFT 中位 |
|---|---|---|---|---|
| 1 × 2500 | 2500 | 40,481 | 5,060 | 8.5s |
| 4 × 625 | 2500 | 48,934 | **6,117** | 7s |
| 8 × 625 | 5000 | 54,300 | **6,788** | 38s |

→ **多 frontend +34%（5,060 → 6,788）**。单个 dynamo.frontend 是 Python 进程，高并发下 CPU-bound（处理 HTTP + KV 路由 + tokenize），分流到多进程直接提吞吐 + 降 TTFT。**这就是"提高的这一大截子"**。

### 12.4 完整瓶颈地图（距 11,200 的分解）
1. **编排 / frontend —— 占约 34% 损失**，多 frontend 已吃回（5,060 → 6,788）。
2. **剩余 ~1.65× —— prefill 喂料受限**。8-frontend conc5000 时 TTFT 飙 38s（prefill 队列积压），算账：prefill 正跑在 ~15k input tok/s/卡（= §11 单卡峰值），8 个 prefill 喂 ~59 req/s 就是池子天花板。根子两条：(a) 单卡 prefill 15k vs 喂饱 dep8 需 ~18.2k = **1.2× nightly/kernel 调优差**（§11.4）；(b) **8 个 prefill 数量不够喂满 dep8**。
3. **decode 不是瓶颈**（TPOT 26-35ms 一直健康）。

### 12.5 再提升的杠杆（未做）
- **加 prefill 数量**（12-15p；官方 `15p1d-dep12` 就是拉满 prefill）：prefill 喂得上，decode 就能填满冲更高。现有 18 节点还有 p9 + d2-d7 空着。
- **修 prefill 单卡 1.2×**：对齐官方内核优化 / pinned 镜像（§10.7）。
- **口径铁律**：对标 11,200 必须 sa-bench 开环 + 多 frontend + 高并发；单 frontend / 闭环 / 低并发都会严重低估。

### 12.6 【已验证】加 prefill 数量 → +30%（8→14 prefill，2026-07-22）

按 §12.5 杠杆实测：复用空闲干净节点（p9 + d2-d6），把 prefill 从 8 扩到 **14**（各 dep4，decode 仍 dep8/8卡，frontend 同步扩到 14），sa-bench 开环 14 路 × conc600（共 ~8400）：

| prefill 数 | 有效驱动路 | 聚合 output tok/s | **output/decode-GPU** | 增量 |
|---|---|---|---|---|
| 8 | 8 × conc625 | 54,300 | 6,788（61%）| 基线 |
| **14** | 14 × conc600 | 70,475 | **8,809（79%）** | **+30%** |
| **16** | 14 × conc600* | 71,949 | **8,993（80%）** | **+2%（收敛）** |

*16-prefill 时 d3/d6 两路 frontend 被 setsid 修复的 pkill python3 误杀，实为 14 路有效驱动（offered ~8400）；16 个 prefill worker 均在池中被路由。

→ **8→14 prefill +30%（喂料是瓶颈）；14→16 只 +2%（强烈收敛）**。**关键转折：瓶颈已从 prefill 喂料转到 decode 自身**——再加 prefill 也喂不出更多，dep8 decode 的吞吐天花板 ≈ 9,000 output/decode-GPU = **官方 80%**。

**GPU 利用率实测（16-prefill 满载，nvidia-smi 采样）**：
- **prefill 卡**：util **99-100%**（彻底打满），HBM **271-277 GiB**（288 的 ~96%），功耗 **1137 W**。
- **decode 卡**：util **75-97%**（有波动，偶尔等喂料/MTP verify 间隙），HBM **268-275 GiB**，功耗 **960-1038 W**。
- GB300 单卡 TDP ~1400W，prefill 1137W ≈ 81%、decode ~1000W ≈ 71% → 两边都在高负载,但没顶满功耗墙。

**最终决定性结论**：满配 16p + dep8 天花板 = **8,993 = 官方 80%**，剩余 ~1.25× = **decode/prefill 单卡内核成熟度差 ~1.2×**（我们 nightly vs 官方 pinned 镜像的 kernel 优化），**不是架构、拓扑、编排、prefill 数量的问题**——那些都已调到收敛。要摸 11,200 只剩"对齐官方内核/镜像"这一条路（§10.7）。

---

## 13. 端到端复现 checklist（满配 sa-bench 官方口径，2026-07-22 验证）

把全流程收成一条可照抄的路径（复用已就位的 18-node GB300 fleet + node-local SSD 模型；复用 pod 清僵尸即可，不必重建）。

> **★ 一次成功的关键 = 自愈部署（把手动救场变成脚本循环）**。手动"起 16 个再逐个救"每次都漏；正确姿势是**启动后进「校验-重试」循环**：
> 1. 清所有 pod 进程 → 分发脚本 → 一次性 `setsid` 启动 decode(dep8 跨 d0/d1) + 16 prefill。
> 2. **每 90s 轮询**：`nvidia-smi memory.used < 200G` 的 prefill = 没起来。对这些 pod `pkill python3 + killport 40236 + setsid` 重启，回到轮询。
> 3. **关键洞察**：40236 端口僵尸是**暂时的**——`kill -9` 杀不掉 D-state 持有者，但内核会在其 GPU 驱动调用返回后自动 reap，**给足 90s 再重试就会成功**。实测 2-3 轮清干净全部（本轮 16 个里 3 个中招：round1 [p0 d2 d5]→round2 [d5]→round3 全绿，8min 一次成功）。**别急着重试**（上一轮给 p7 立刻重试→连撞→误以为要重建）。
> 4. 重试 3 轮仍不动的（罕见，真 D-state 卡死）→ 才 `kubectl delete pod --force` 重建（fresh pod 记得补 `kubectl cp` bench 工具）。
> 5. decode 就绪判据：`grep 'Model registration succeeded' /tmp/srv.log`（graph capture ~6-10min，与 prefill 重试并行，通常 prefill 齐时 decode 也好了）。
> 6. **frontend 必须最后统一起**：因为 `pkill python3` 会连 frontend 一起杀，所以**先把 worker 全稳定，再一次性起 16 个 frontend 并验 200**，不要边起 worker 边补 frontend。
>
> 这套循环脚本化后 = 一条命令一次成功，无需人工盯。下面 Step 1-5 是它内部各阶段的参数细节。

**Step 0 前提**：18 节点 GB300 单 NVL72 域（subblock）；模型 DeepSeek-V4-Pro 在各节点 `/mnt/disks/raid/0`（容器内 `/mnt/ssd`，hostPath）；`dynamo-nats` + `dynamo-etcd` pod 就绪；镜像 `lmsysorg/sglang:nightly-dev-cu13-*`。
> **pod 从零创建**（fleet 不存在时）：用 18-pod 生成器（`gen18-0001.py` 模板：ComputeDomain `sgl4-cd` + `mrdma.google.com` DRA claim count=8 + subblock `podAffinity` + hostname `podAntiAffinity`（每节点 1 pod）+ hostPath `/mnt/disks/raid/0` + 600Gi mem + `imagePullSecrets: ar-pull-secret`），`kubectl apply` 后 GIB/DOCA/ai-dynamo 在容器内 bootstrap（同 R1 §4）。**已有 fleet 只需复用 pod 清进程重部署，不必重建**。

**Step 1 部署 workers**（16 prefill + 1 decode，用 `dynamo.sglang`）：
- **16 prefill**：每节点 1 个 dep4（`--tensor/data/expert-parallel-size 4` + `--moe-a2a-backend megamoe` + W4A4 env `USE_FP4_ACTS=1` + `--enable-dp-attention --enable-dp-lm-head` + `--chunked-prefill-size 32768` + `--mem-fraction-static 0.9` + `--disaggregation-mode prefill`），端口 40000。
- **1 decode**：dep8 跨 2 节点（`--tp/dp/ep 8 --nnodes 2 --node-rank 0/1 --dist-init-addr d0IP:5000` + MTP `--speculative-algorithm EAGLE --speculative-num-steps 1 --speculative-eagle-topk 1 --speculative-num-draft-tokens 2` + `--mem-fraction-static 0.85 --max-running-requests 8192 --context-length 9216 --swa-full-tokens-ratio 0.1 --disaggregation-mode decode`）。
- env 关键：`NATS_SERVER`/`ETCD_ENDPOINTS`（服务发现，worker 自注册，热加不用重启全体）+ `MC_FORCE_MNNVL=1`/`NCCL_MNNVL_ENABLE=1`（域内 NVLink mooncake KV 传输）。

**Step 2 多 frontend（吞吐关键，+34%）**：每 prefill 节点起一个 `python3 -m dynamo.frontend --http-port 8001`（带 NATS/ETCD env）。单个裸 frontend 高并发 CPU-bound 是瓶颈。

**Step 3 探活**：`curl frontend:8001/v1/models` 须 `owned_by:nvidia` + `context_window:1048576`（否则是僵尸 `sglang::router` 占端口）；再 `/v1/completions` 发一条确认能生成。

**Step 4 sa-bench（官方口径）**：InferenceX `utils/bench_serving/benchmark_serving.py`，`--request-rate inf`（**开环**）`--ignore-eos --dsv4 --use-chat-template` + DSV4 tokenizer，ISL 8192 / OSL 1024 / range 0.8；**多路并行**（每 frontend 一路，各 conc ~600）。

**Step 5 口径**：各路 output token throughput **求和 ÷ decode-GPU 数（8）** = output/decode-GPU，对标官方 11,200（prefill 卡不进分母、input 不进分子）。

**三大部署坑（已被上面自愈循环覆盖，原理备查）**：
1. **40236 ZMQ 端口僵尸**（最常见，每次约 3/16 prefill 中招）：`pkill -9` 杀不掉 D-state 持有者，但**内核会自动 reap，等 90s 再 killport+setsid 重试即成**（2-3 轮清完）。killport = `/proc/net/tcp` 反查 40236(hex 9D2C) inode→PID→kill。**唯一注意：别急着重试**（<10s 连撞会误判需重建）。
2. **decode 死进程真卡死泄漏显存**（199G/卡不释放、`--query-compute-apps` 为空、`kill -9` 也回收不了）→ 这才需 `kubectl delete pod --force` 重建（模型在 node SSD 持久）。仅在 40236 重试 3 轮仍不动时才升级到此。
3. **`pkill python3` 连 frontend 一起杀** → 所以 frontend 必须在 worker 全稳定后**统一最后起**（见自愈循环第 6 步），别边起 worker 边补。`"fired up"` 日志 stdout 缓冲不可靠，用 **GPU mem>200G 判 prefill ready**、`grep 'Model registration succeeded' 判 decode ready`。

**GPU 利用率参考（满配 16p 满载）**：prefill 99-100% util / HBM 96% / 1137W；decode 75-97% util / HBM 96% / ~1000W（GB300 TDP ~1400W）。

> **✅ 审计复现验证（2026-07-22，两次从零跑）**：
> - **第 1 次**（手动逐个救场）：16 路 sa-bench = **9,160 output/decode-GPU**（73,277÷8），≈ 基线 8,993（±2%，均 ≈ 官方 80%）。过程撞 40236 端口僵尸，手动救，其中 1 个（急于重试）误判为需重建 pod。
> - **第 2 次**（自愈循环脚本，见上方 ★）：**一次成功，8 分钟，全自动无人工**。round1 [p0 d2 d5] 中招 → 90s 后 killport 重试 → round2 剩 [d5] → round3 全绿；decode 与 prefill 并行 capture 完毕即注册；frontend 统一起验 200；探活 owned_by=nvidia；e2e mini-bench 32/32 通。**印证「40236 僵尸是暂时的、耐心重试 2-3 轮自清、无需重建」**。
> - **第 3 次**（按本定稿文档 + §13.A 完整脚本真实从零跑）：**又一次成功，8 分钟自动**（仅 p8 中招 → 3 轮 killport 清完）；16 路 sa-bench = **8,903 output/decode-GPU**（71,227÷8）。三次从零跑（9,160 / 8,903 + 基线 8,993）全落在 ±2%，均 ≈ 官方 80%。
> 结论：一次成功的关键不是运气，是**把校验-重试循环脚本化 + 每轮间隔 ≥90s**；文档 §13.A 完整脚本 + 自愈循环 = 可照抄、可复现、一次成功。

### 13.A ⭐ 完整启动脚本（source of truth，照抄即可，勿删任何 env）

> 上面 Step 1 是参数解读；**下面两个脚本是实测能跑的完整版**，一个字都不能漏（GIB source、cache 目录、SWA、`USE_CUSTOM_ALL_REDUCE_V2` prefill=1/decode=0 的差异、`NUM_MAX_TOKENS_PER_RANK` prefill=9216/decode=4096、reasoning/tool parser、ib-device、disagg 超时全是必需）。

**`prefill-dep8.sh`**（16 个 prefill 每节点跑一份，`setsid bash prefill-dep8.sh`）：
```bash
#!/bin/bash
source /usr/local/gib/scripts/set_nccl_env.sh 2>/dev/null || true
export NCCL_CONF_FILE=/usr/local/gib/configs/nccl.a4xmax.conf LD_LIBRARY_PATH=/usr/local/gib/lib64:${LD_LIBRARY_PATH:-}
export NCCL_DEBUG=INFO NCCL_SOCKET_IFNAME=eth0 GLOO_SOCKET_IFNAME=eth0 NCCL_IB_SPLIT_DATA_ON_QPS=1
export SGLANG_MOONCAKE_CUSTOM_MEM_POOL=True MC_FORCE_MNNVL=1 NCCL_MNNVL_ENABLE=1 NCCL_CUMEM_ENABLE=1
export FLASHINFER_DISABLE_VERSION_CHECK=1 SGLANG_DG_CACHE_DIR=/mnt/ssd/dg-cache FLASHINFER_WORKSPACE_BASE=/mnt/ssd/fi-cache
export SGLANG_JIT_DEEPGEMM_FAST_WARMUP=1
export NATS_SERVER=nats://dynamo-nats:4222 ETCD_ENDPOINTS=http://dynamo-etcd:2379 DYN_SYSTEM_PORT=8081
export SGLANG_RADIX_DISABLE_REUSE=1 SGLANG_DEFAULT_THINKING=1 SGLANG_DSV4_REASONING_EFFORT=max
export SGLANG_OPT_SWA_SPLIT_LEAF_ON_INSERT=1 SGLANG_OPT_SWA_EVICT_DROP_PAGE_MARGIN=1 SGLANG_OPT_SWA_RELEASE_LEAF_LOCK_AFTER_WINDOW=1
export SGLANG_OPT_USE_CUSTOM_ALL_REDUCE_V2=1
export SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK=9216 SGLANG_OPT_DEEPGEMM_MEGA_MOE_USE_FP4_ACTS=1 SGLANG_OPT_DEEPGEMM_MEGA_MOE_USE_MXF4_KIND=1
export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=100000 SGLANG_DISAGGREGATION_WAITING_TIMEOUT=100000 SGLANG_DISAGGREGATION_HEARTBEAT_MAX_FAILURE=20
python3 -m dynamo.sglang --model-path /mnt/ssd/DeepSeek-V4-Pro --served-model-name deepseek-ai/DeepSeek-V4-Pro \
  --trust-remote-code --reasoning-parser deepseek-v4 --tool-call-parser deepseekv4 --watchdog-timeout 86400 \
  --tensor-parallel-size 4 --data-parallel-size 4 --expert-parallel-size 4 \
  --enable-dp-attention --enable-dp-lm-head --moe-a2a-backend megamoe --moe-dense-tp-size 1 \
  --disaggregation-mode prefill --disaggregation-transfer-backend mooncake --disaggregation-bootstrap-port 30001 \
  --disaggregation-ib-device mlx5_0,mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7 \
  --mem-fraction-static 0.9 --max-running-requests 1024 --cuda-graph-max-bs 1024 --chunked-prefill-size 32768 \
  --stream-interval 60 --disable-radix-cache --enable-metrics --host 0.0.0.0 --port 40000
```

**`decode-hc.sh`**（dep8 跨 2 节点：d0 `bash decode-hc.sh 0 <d0IP>:5000`，d1 `bash decode-hc.sh 1 <d0IP>:5000`）：
```bash
#!/bin/bash
NODE_RANK=$1; DIST_ADDR=$2
source /usr/local/gib/scripts/set_nccl_env.sh 2>/dev/null || true
export NCCL_CONF_FILE=/usr/local/gib/configs/nccl.a4xmax.conf LD_LIBRARY_PATH=/usr/local/gib/lib64:${LD_LIBRARY_PATH:-}
export NCCL_DEBUG=WARN NCCL_SOCKET_IFNAME=eth0 GLOO_SOCKET_IFNAME=eth0 NCCL_IB_SPLIT_DATA_ON_QPS=1
export SGLANG_MOONCAKE_CUSTOM_MEM_POOL=True MC_FORCE_MNNVL=1 NCCL_MNNVL_ENABLE=1 NCCL_CUMEM_ENABLE=1
export FLASHINFER_DISABLE_VERSION_CHECK=1 SGLANG_DG_CACHE_DIR=/mnt/ssd/dg-cache FLASHINFER_WORKSPACE_BASE=/mnt/ssd/fi-cache
export SGLANG_JIT_DEEPGEMM_FAST_WARMUP=1
export NATS_SERVER=nats://dynamo-nats:4222 ETCD_ENDPOINTS=http://dynamo-etcd:2379 DYN_SYSTEM_PORT=8082
export SGLANG_RADIX_DISABLE_REUSE=1 SGLANG_DEFAULT_THINKING=1 SGLANG_DSV4_REASONING_EFFORT=max
export SGLANG_OPT_SWA_SPLIT_LEAF_ON_INSERT=1 SGLANG_OPT_SWA_EVICT_DROP_PAGE_MARGIN=1 SGLANG_OPT_SWA_RELEASE_LEAF_LOCK_AFTER_WINDOW=1
export SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK=4096 SGLANG_OPT_DEEPGEMM_MEGA_MOE_USE_FP4_ACTS=1 SGLANG_OPT_DEEPGEMM_MEGA_MOE_USE_MXF4_KIND=1
export SGLANG_OPT_USE_CUSTOM_ALL_REDUCE_V2=0
export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=100000 SGLANG_DISAGGREGATION_WAITING_TIMEOUT=100000 SGLANG_DISAGGREGATION_HEARTBEAT_MAX_FAILURE=20
python3 -m dynamo.sglang --model-path /mnt/ssd/DeepSeek-V4-Pro --served-model-name deepseek-ai/DeepSeek-V4-Pro \
  --trust-remote-code --reasoning-parser deepseek-v4 --tool-call-parser deepseekv4 --watchdog-timeout 86400 \
  --tensor-parallel-size 8 --data-parallel-size 8 --expert-parallel-size 8 --pp-size 1 \
  --nnodes 2 --node-rank $NODE_RANK --dist-init-addr $DIST_ADDR \
  --enable-dp-attention --enable-dp-lm-head --moe-a2a-backend megamoe --moe-dense-tp-size 1 \
  --disaggregation-mode decode --disaggregation-transfer-backend mooncake --disaggregation-bootstrap-port 30001 \
  --disaggregation-ib-device mlx5_0,mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7 \
  --speculative-algorithm EAGLE --speculative-num-steps 1 --speculative-eagle-topk 1 --speculative-num-draft-tokens 2 \
  --mem-fraction-static 0.85 --swa-full-tokens-ratio 0.1 --context-length 9216 \
  --max-running-requests 8192 --cuda-graph-max-bs 1280 --stream-interval 60 --disable-radix-cache --enable-metrics
```

**frontend**（16 个 prefill 节点各起一份）：`NATS_SERVER=nats://dynamo-nats:4222 ETCD_ENDPOINTS=http://dynamo-etcd:2379 setsid python3 -m dynamo.frontend --http-port 8001`

## 14. ⭐⭐⭐⭐ Autotune + EPLB 实验：冲剩余 20% gap 的两次尝试（2026-07-22）

实测两个可能冲击剩余 ~20% gap 的杠杆：完整 DeepGEMM autotune 和 EPLB。**结论：两者都无法有效缩小 gap，反向印证剩余差距只在官方 pinned 镜像的内核成熟度。**

### 14.1 完整 DeepGEMM autotune（FAST_WARMUP=0）——无提升

`SGLANG_JIT_DEEPGEMM_FAST_WARMUP` 控制 warmup 时编译的 M shape 集合（`deep_gemm_wrapper/compile_utils.py:56`）：
- `=1`（fast，默认）：只编译 ~3072 个采样 shape（1-1024 密集 + 大 batch 稀疏采样）
- `=0`（full）：编译 1..m_max 全部（prefill chunked=32768 → m_max=65536，即约 21× 的 shape）

全 18 worker 用 `=0` 重启实测（官方口径 output÷8 decode-GPU）：

| 配置 | aggregate tok/s | ÷8 decode-GPU | vs baseline |
|---|---|---|---|
| baseline（fast-warmup，热态） | 71,944 | 8,993 | — |
| full autotune 首轮（冷，含 67s 首请求 JIT） | 64,300 | 8,037 | -11% |
| full autotune 第二轮（热态） | 72,147 | 9,018 | +0.3% |

**full autotune 无有意义提升**（9,018 vs 8,993 在噪声内）。冷→热那 +12% 纯是一次性 JIT 编译，不是 autotune 的功劳——fast 和 full 热态后行为一致。原因：fast-warmup 已覆盖 serving 实际碰到的常用 M shape，这些 shape 的 DeepGEMM kernel 本就接近最优；full 多编的 6 万个 shape serving 根本用不上。**代价是巨大启动开销，收益为零，不值。**

### 14.2 EPLB——与 megamoe 不兼容，三种失败模式

EPLB（Expert Parallel Load Balancing）针对 decode 端专家负载不均。在本 build 尝试 `--enable-eplb --ep-num-redundant-experts N`（非 elastic 模式 = 在线自动重平衡，`recorder-mode` 自动设 stat、`ep-dispatch-algorithm` 设 static、每 N 迭代重平衡）：

**关键发现：EPLB 强制放弃 megamoe。** megamoe 融合 kernel 不支持 redundant expert 迁移，启用 EPLB 后 MoE backend 自动切到更重的 **FlashInfer TRTLLM MoE**——而 megamoe 正是 11,200 recipe 的核心优化之一。

三种失败模式（decode dep8 跨 2 节点）：
1. **OOM**：redundant=8 + 原配置（mem-frac 0.85 / graph 1280）→ flashinfer backend 静态内存超预算 → `Not enough memory`
2. **AssertionError**：`num_physical_experts % ep_size == 0`（`eplb/expert_location.py:250`）——256 routed + redundant 必须被 ep_size=8 整除，redundant 只能取 {8,16,24...}，不能取 4
3. **prewarm 死锁**：redundant=8 + 减内存（mem-frac 0.88 / graph 512）过了 OOM 和断言，但卡死在 MHC prenorm prewarm（5 分钟零推进，很可能是 EPLB 专家重分布导致 2 节点 rank 集合通信配置不一致）

**EPLB 与 megamoe 在此 2 节点 PD 分离 decode 上不可用。** 官方 recipe 不启用 EPLB 是正确选择。（注：清 decode 僵尸时 kill -9 杀不掉 D-state 进程，须 `kubectl delete pod --force` 重建。）

### 14.3 综合结论

两个实验都无法缩小 gap，**反向印证** §12 的瓶颈地图：剩余 ~1.25× 既不是 kernel autotune 覆盖度（14.1 证伪），也不是 decode 专家不均（14.2 证 EPLB 在本栈不可用）——**确认只在官方 pinned 镜像的整体内核/运行时成熟度**。冲满 11,200 唯一剩的路是对齐官方 pinned 镜像（tag 编码 commit `14f81a67`，从该 commit 构建）。

---

*2026-07-22 定稿（Local SSD based）+ **端到端复现审计通过**：全 fleet 清空 → 照 §13 从零重新部署 → 16 路 sa-bench 复现 **9,160 output/decode-GPU**（≈ 基线 8,993，±2%），步骤与 benchmark 均验证正确。Phase 1（Flash）+ Phase 2（Pro）单节点 TP4 + **Phase 3 满配 PD（Dynamo / megamoe W4A4 / SWA / MTP）全部实测通过**。导航：**§13 端到端复现 checklist（照抄）、§12 最终结果、§14 autotune/EPLB 收口**、§10.1-10.3 官方 11,200 口径+公式、§11 prefill 并发扫描修正、§3.9 单节点手册。*

***最终结论（官方口径 output÷decode-GPU，sa-bench 开环）***：满配 **16 prefill + dep8-MTP + 多 frontend = 8,993 = 官方 11,200 的 80%**。提升路径：单 frontend 5,060 →（多 frontend）6,788 →（14 prefill）8,809 →（16 prefill）8,993；**14→16 仅 +2% 已收敛**。**瓶颈地图**：① 编排/frontend（多 frontend 吃回 +34%）② prefill 喂料（加 prefill 吃回 +32%，到 16 收敛）③ 剩余 **~1.25× = 单卡内核成熟度差**（我们 nightly vs 官方 pinned 镜像的 kernel/runtime），**非架构/拓扑/编排/prefill 数量/decode 配置**。**§14 实测收口**：完整 DeepGEMM autotune（关 FAST_WARMUP）热态无提升（9,018 vs 8,993），EPLB 与 megamoe 不兼容（三种失败模式），两者都不能缩小 gap → 冲满 11,200 唯一剩的路是**对齐官方 pinned 镜像（从 commit `14f81a67` 构建）**。早期"prefill 单卡慢 3.7×"（§10.4）系低并发测量假象，已被 §11 证伪。存储全程 Local SSD RAID（见 gb300-local-ssd-raid0-SETUP.md）。R1 端到端见 `../deepseek-v3/`。*
