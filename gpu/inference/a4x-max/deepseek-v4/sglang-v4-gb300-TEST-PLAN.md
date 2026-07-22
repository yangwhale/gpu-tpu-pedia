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
| **最优实测（官方口径 output÷decode-GPU）** | **8-frontend + high-conc-8p1d-dep8-mtp + sa-bench 开环 = 6,788 = 官方 11,200 的 61%**（多 frontend 从 5,060 提 +34%，见 §12）|
| **距官方 11,200** | 满配 dep8-MTP 稳态 6,659 output/decode-GPU。**曾以为根因=prefill 单卡慢 3.7×，已推翻**（§11：那是 conc4 低并发测量假象，同一 worker 高并发峰值 15,196=官方 83%，prefill 非瓶颈）。真因待用官方 sa-bench 开环重测（编排/decode 侧）|

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

**详细导航**：§3.9 单节点可复现手册 · §3.3+§3.4 PD+Dynamo 复现步骤 · §7 单节点 benchmark · **§10 官方 11,200 深度解析 + gap 根因（最权威，看这个）**。

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

### Phase 3：PD-disagg 满配 + 逐项消融（冲官方 11,200 tok/s/GPU）

> **官方 11,200 的真相**（核对 pytorch blog 2026-06-23 + SemiAnalysis InferenceX）：GB300 **disaggregated** lane、V4-Pro FP4、ISL=8192/OSL=1024、dynamo-sglang、**2026-06 带 MTP 曲线**，在 **~50 tok/s/user** 交互点达 **~11,200 tok/s/GPU**；对比 Day-0（2026-04）no-MTP 的 ~2,200，两个月 **5×**。这 5× 不是单点，是整条曲线抬升。官方 recipe：`disagg-gb300-10p1d-dep4-dep32-18-c2500.yaml`（srt-slurm + Dynamo 起）。

#### 3.0 消融方法论（本阶段最高原则）

- **锁死拓扑做 baseline**：整个消融**全程固定** `10P1D-dep4-dep32`（P/D 数量、并行度、workload 全不变），**只逐项翻软件开关**。绝不拿不同 P/D 数量的两个 run 对比——那不可比。
- **一次只加一样**，拿到干净的 per-step delta，每招值多少 tok/s 心里有数。
- 全程同口径：8K/1K（random 8192/1024，range-ratio 1.0）、同并发扫描、warm 值、`total(in+out) ÷ 72 GPU`。

#### 3.1 拓扑（固定，官方 10P1D-dep4-dep32）

| 角色 | 实例 | 每实例并行 | 节点 | GPU |
|---|---|---|---|---|
| Prefill | 10 | TP4 / DP4 / EP4（1 节点 4 GPU）| 10 | 40 |
| Decode | 1 | DEP32（TP32 / DP32 / EP32）| 8 | 32 |
| **合计** | — | — | **18** | **72** |

- **P:D = 10:8（机器）**——prefill 多，因 8K 输入是算力瓶颈，多铺机器摊 prefill；decode 集中成一坨 wide-EP 吃并发。
- **decode 32 卡走域内 NVLink**（mooncake `MC_FORCE_MNNVL=1`），**18 节点必须同一 NVL72 域**，否则跨域 NVLink KV 传输不通。

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

#### 3.2 消融 run 序列（固定 3.1 拓扑，逐项叠加）

> **⚠️ 2026-07-20 重大方向修正**：原计划 baseline 用 `deepep` a2a，实测在 `lmsysorg/sglang:latest` 上 V4 prefill 的 deepep 路径**根本跑不通**（连撞 4 个 runner bug，见 3.2-b）。拉了官方 recipe 原文（`gh api SemiAnalysisAI/InferenceX .../disagg-gb300-10p1d-dep4-dep32-18-c2500.yaml`）确认：**官方 prefill+decode 全程用 `megamoe`，不是 deepep**，且用**特定 nightly 镜像** `lmsysorg/sglang:nightly-dev-cu13-20260520-425dffbd`（非 latest）。故 baseline 改从 **megamoe W4A8** 起——这才是 V4 真正跑得通的路。

| Run | 相对上一步新增 | MoE backend | 激活精度 | SWA opt/压缩 | MTP | 目的 |
|---|---|---|---|---|---|---|
| **A** baseline | megamoe（官方镜像）| megamoe | W4A8（不设 `USE_FP4_ACTS`）| 关 | 关 | V4 可跑底线 |
| **B** +W4A4 | `USE_FP4_ACTS=1` `USE_MXF4_KIND=1` | megamoe | **W4A4** | 关 | 关 | 激活也 4bit（官方最大一跳）|
| **C** +SWA/压缩 | `SGLANG_OPT_SWA_*` + `USE_ONLINE_COMPRESS=1` | megamoe | W4A4 | **开** | 关 | SWA 预算 + online 压缩，decode 更大 batch |
| **D** +MTP（=满配终极）| `--speculative-algorithm EAGLE ...` | megamoe | W4A4 | 开 | **开** | 官方满配，对标 11,200 |

- **镜像**：全程 `lmsysorg/sglang:nightly-dev-cu13-20260520-425dffbd`（官方 recipe 指定；`latest` 的 deepep/V4 路径有 bug）。
- **每个 run 输出**：conc 扫描（1 / 16 / 64 / 128 …推到 c2500）的 tok/s/GPU + TPOT + TTFT，记录到 §7 消融表。
- **W4A8 vs W4A4**：由 env `SGLANG_OPT_DEEPGEMM_MEGA_MOE_USE_FP4_ACTS`（0/不设 = W4A8，1 = W4A4）切换；`NUM_MAX_TOKENS_PER_RANK` prefill=8192 / decode=1280（官方值）。

> **✅ 3.2-b Run A 起服务踩坑（2026-07-20，deepep 路线弃用前的血泪）**：`latest` 镜像下 V4 prefill 依次撞：(1) `flashinfer_trtllm_routed` runner + `deepep` a2a 不兼容（`requires a fused func for a2a backend deepep, but none is registered`）；(2) 换 `flashinfer_cutedsl` → prefill `deepep normal` 返回 5-tuple 但 cutedsl 要 6-tuple（`not enough values to unpack (expected 6, got 5)`，cutedsl 只吃 decode 的 low_latency）；(3) 换 `deep_gemm`（注意是下划线，`deepgemm` 是 invalid choice）→ `tensor a (384) vs b (96)` DP/EP shape 不匹配。**结论：V4 的 deepep prefill 在此 stack 未调通，官方走 megamoe，遂弃 deepep 改 megamoe baseline。** 另：prefill DEP4（4 卡装 1.6T）`mem-fraction-static` 需 ≥0.90（0.8 报 `no GPU memory for KV cache`）；decode DEP32（32 卡摊薄）0.94 即可。

> **✅ 3.2-c 关键突破：checkpoint 变体 + 镜像 决定 megamoe 能否跑（2026-07-20，Run A 跑通根因）**：
> - **根因**：`nvidia/DeepSeek-V4-Pro-NVFP4` 变体**强制** `flashinfer_trtllm_routed` runner，它对 **deepep 和 megamoe 两种 a2a 都没有 fused func**（`requires a fused func for a2a backend {deepep|megamoe}, but none is registered`）→ 单节点 TP（Phase 2）能跑，但**多节点 PD 的 megamoe/deepep 全崩**。
> - **正解**：megamoe PD 必须用**官方原版 checkpoint `deepseek-ai/DeepSeek-V4-Pro`**（FP4 MoE + FP8 attn/dense，~806G），不是 nvidia NVFP4 变体。官方 recipe 的 `model.path` 就是原版。
> - **镜像**：官方 pin 的 `nightly-dev-cu13-20260520` 已被 Docker Hub GC（`not found`）；`latest`（0.5.15.post1）的 megamoe auto-runner 也 mismatch。**用最新可用 nightly**（实测 `lmsysorg/sglang:nightly-dev-cu13-20260720-b3570a45` ✅）。查可用 tag：`curl -s "https://hub.docker.com/v2/repositories/lmsysorg/sglang/tags/?page_size=100&name=nightly-dev-cu13"`。
> - **✅ Run A 实测跑通（2026-07-20 22:22）**：原版 checkpoint + 最新 nightly + megamoe，10P1D-dep4-dep32（72 GPU，subblock-0001），sglang_router `cache_aware` `--pd-disaggregation`，PD 端到端经 router 正确生成（"Capital of France"→"Paris"），prefill→decode KV 走域内 NVLink（mooncake `MC_FORCE_MNNVL=1`）。
> - **换镜像必重 bootstrap**：GIB/DOCA/nixl 装在容器内，换 image 后 pod 全新，需重跑 bootstrap（GIB tgz + DOCA OFED userspace + nixl）。
> - **prefill DEP4 mem-fraction 0.90 / decode DEP32 0.94**（DEP4 4卡装 1.6T 更挤）。

#### 3.3 decode / prefill 关键参数（官方 recipe）

decode（DEP32）：
```
--moe-a2a-backend megamoe --enable-dp-attention --enable-dp-lm-head \
--tp-size 32 --dp-size 32 --ep-size 32 --swa-full-tokens-ratio 0.20 \
--context-length 9216 --mem-fraction-static 0.94 \
--max-running-requests 18432 --cuda-graph-max-bs 1280 \
--disaggregation-mode decode --disaggregation-transfer-backend mooncake
```
- W4A4 env：`SGLANG_OPT_DEEPGEMM_MEGA_MOE_USE_FP4_ACTS=1 SGLANG_OPT_DEEPGEMM_MEGA_MOE_USE_MXF4_KIND=1 SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK=8320`
- SWA env：`SGLANG_OPT_SWA_SPLIT_LEAF_ON_INSERT=1 SGLANG_OPT_SWA_EVICT_DROP_PAGE_MARGIN=1 SGLANG_OPT_USE_ONLINE_COMPRESS=1`
- **MTP（Run D，实测跑通 = 这套）**：`--speculative-algorithm EAGLE --speculative-num-steps 1 --speculative-eagle-topk 1 --speculative-num-draft-tokens 2`（DeepSeek V4 原生 nextn=1，**prefill + decode 两侧都要加**，否则 nextn 模块加载不一致）。**不要**用 `num-steps 3`（未验证）。decode 侧配套：`NUM_MAX_TOKENS_PER_RANK=4096`（draft 放大 token 数，1280 会 `exceeds cap`）、`mem-fraction 0.90`、`context-length 9216`（**别提到 9728/10240——会 cuda-graph OOM**）。benchmark 侧 **OSL 用 960**（8192+960+draft < ctx 9216，否则 MTP 令 8194+1024=9218 超限，78% 请求被拒）。**MTP 与 online 压缩互斥**（去掉 `USE_ONLINE_COMPRESS`）。
- mooncake NVLINK（同 R1）：`MC_FORCE_MNNVL=1 NCCL_MNNVL_ENABLE=1 NCCL_CUMEM_ENABLE=1`（prefill + decode 都加）

**✅ 实测跑通的 PD 启动要点（Run A，原版 checkpoint + megamoe）**：
- **model-path 用原版** `/mnt/ssd/DeepSeek-V4-Pro`（非 NVFP4），**不设** `--moe-runner-backend`（megamoe 自选）。
- prefill（每节点 DEP4）：`--tp 4 --dp 4 --ep 4 --enable-dp-attention --enable-dp-lm-head --moe-a2a-backend megamoe --moe-dense-tp-size 1 --deepep-config '{normal_dispatch...}' --disaggregation-mode prefill --mem-fraction-static 0.90 --chunked-prefill-size 32768 --cuda-graph-max-bs 512`。
- decode（8 节点 DEP32）：`--tp 32 --dp 32 --ep 32 --nnodes 8 --node-rank $R --dist-init-addr $D0IP:5000 --enable-dp-attention --enable-dp-lm-head --moe-a2a-backend megamoe --moe-dense-tp-size 1 --disaggregation-mode decode --disaggregation-decode-polling-interval 8 --mem-fraction-static 0.94 --swa-full-tokens-ratio 0.20 --context-length 9216 --max-running-requests 18432 --cuda-graph-max-bs 1280`。
- **router**（sglang_router，**仅冒烟用，高并发有 tail-stall**）：`python -m sglang_router.launch_router --pd-disaggregation --prefill http://<pIP>:30000 30001 ×10 --decode http://<d0IP>:30000 --policy cache_aware --host 0.0.0.0 --port 8000`。
- **启动耗时**：8 节点 decode NCCL rendezvous + megamoe warmup ~7min 到 ready；**首次 benchmark warmup 极慢**（megamoe/deepgemm 对每个 8K prefill shape 首次 JIT 编译，单请求可达十几分钟），JIT 缓存后正常。

#### 3.4 ⭐ Dynamo 编排（官方用的，高并发唯一可用；实测跑通 = 这套）

sglang_router 高并发 tail-stall（§8.3），**必须换 Dynamo**。完整步骤：

1. **NATS + ETCD**（k8s pod，default-pool）：`kubectl apply -f nats-etcd.yaml`（`dynamo-nats` = nats:2.10-alpine -js；`dynamo-etcd` = quay.io/coreos/etcd:v3.5.16 + Service）。
2. **18 worker 装 ai-dynamo**：每 pod `pip install ai-dynamo`（**不降级 nightly sglang**，`dynamo.sglang` 能 import 即可）。
3. **worker 启动**：把 §3.3 的 `python -m sglang.launch_server` 换成 **`python3 -m dynamo.sglang`**（透传所有 megamoe/W4A4/DEP32/MTP args），每 pod 加 env：`NATS_SERVER=nats://dynamo-nats:4222 ETCD_ENDPOINTS=http://dynamo-etcd:2379 DYN_SYSTEM_PORT=8081`(prefill)/`8082`(decode) + `--enable-metrics`。prefill 加 `--host 0.0.0.0 --port 40000`。
4. **等 worker 全 ready 再起 frontend**（关键，见下）：decode head（d0）health 200 + 日志 `Model registration succeeded` + `spec decode runtime metadata:{'nextn':1,'method':'EAGLE'}`（MTP 确认）；prefill 各 `:8081/health`=200。
5. **frontend**（在任一 prefill pod）：`NATS_SERVER=... ETCD_ENDPOINTS=... python3 -m dynamo.frontend --http-port 8000`。验证 `/v1/models` 返回 `owned_by:nvidia`+`context_window:1048576`（若返 `owned_by:local` 说明打到了僵尸 sgl-router，见坑）。
6. **benchmark**：`bench_serving --backend sglang-oai --port 8000 --model deepseek-ai/DeepSeek-V4-Pro --dataset-name random --random-input-len 8192 --random-output-len 960 --random-range-ratio 1.0 --num-prompts N --max-concurrency C`（MTP 时 OSL 用 960）。

**⭐ Dynamo 启动/重启六大坑（全趟过，照做避坑）**：
1. **frontend 必须在 worker 完全 ready 后起**：warmup 期起会触发熔断 open 且不恢复 → `No available prefill workers (circuits open)`。
2. **「circuits open」真根因常是僵尸 `sglang::router` 霸占 8000**（它 `/v1/models` 也返 200 但 `owned_by:local`）→ dynamo.frontend bind 失败静默崩。pod 内无 `ss`/`lsof`，用 `/proc/net/tcp` 反查监听 8000 的 PID（hex 端口 `1F40`）→ `kill -9` → 再起 frontend。
3. **换 image/重启必须彻底杀** `dynamo.sglang` + `sglang::`（否则占 dist port 5000 / ZMQ 40236）。
4. **ZMQ `40236 Address already in use`**（prefill 重启常见）：老进程 socket 残留 → `/proc/net/tcp` 反查 40236(hex `9D2C`) 的 PID kill 掉再起。
5. **反复 kill+relaunch 会积累 zombie 进程 + GPU 显存泄漏不释放**（`sleep infinity` 的 PID1 不 reap 僵尸，undead CUDA context 卡住显存到 ~191GB）→ 新进程必 cuda-graph OOM。**唯一可靠解 = `kubectl delete pod <d*> --force --grace-period=0` 重建 pod**（模型在 node-local SSD 持久，podAffinity 保 subblock 不变，重建后 GPU 归 0）。重建 decode pod 后 **IP 变了，prefill 需重启重连**（否则 decode 日志刷 `Lost connection with prefill instance`）。
6. **重启 decode 后 frontend 也要重起**（worker 重新注册）。

### Phase 4：汇总消融报告 + 三方对比
- 把 Run A→E 的 per-step delta 汇成消融表（见 §7.4 模板），得出「每招值多少 tok/s/GPU」。
- 与官方 11,200 对标，分析剩余 gap（Dynamo vs router、c2500 并发是否压满、EPLB/Waterfill 等未开项）。

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

> \* 官方 11,200 是 **output ÷ decode-GPU** 口径，本表单节点数是 **(in+out) ÷ 4** 口径，**两者不可直接相除比较**（此处仅作量级参照）。官方口径的对齐实测见 §7.4（dep8 = 6,659 output/decode-GPU），gap 分析见 §10。

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

### 7.4 Phase 3 PD-disagg + Dynamo 最终结果（官方口径 output ÷ decode-GPU）

> ⚠️ **口径**：以下一律用官方口径 **纯输出 token ÷ decode-GPU 数**（对标官方 11,200）。早期草稿里的 "3,031 / 3,522 tok/s/GPU" 是错口径 (in+out)÷72卡，已删。完整 gap 分析见 §10。

**最优实测（稳态，客户端 output ÷ decode-GPU）**：

| 配置 | decode 卡 | 客户端 output tok/s | **output/decode-GPU** | 对标官方 11,200 |
|---|---|---|---|---|
| dep8 + 8 prefill + MTP-3 | 8 | 53,272 | **6,659** | 0.59×（差 ~1.7×）|
| dep32 + 10 prefill（no-MTP）| 32 | 50,351 | 1,573 | 更低（见 §10.5）|

**要点**：
- **dep8 那一家才是官方 11,200 的配方**（MTP 曲线 @~50 tok/s/user）；dep32/dep40 是 no-MTP 大吞吐曲线，per-decode-GPU 反而低。
- **MTP 是真杠杆**：TPOT 从无 MTP 43ms 稳到 11ms，draft 被接受、per-user 速度不掉的同时吞吐翻倍，复现官方「2,200→11,200 靠 MTP」机制。
- **总输出被 prefill 卡在 ~50K/s**（dep8 53K、dep32 50K 几乎一样）→ decode 铺再宽也白搭。~~瓶颈在 prefill 单卡吞吐慢官方 3.7×~~ **此结论已被 §11 推翻**（prefill 单卡高并发达官方 83%）；满配总输出卡在 ~50K 的真因是 PD 编排没把高并发投到 prefill，需 sa-bench 开环重测。

对标：官方 GB300 disagg V4-Pro FP4 8K/1K @~50 tok/s/user = **11,200 output/decode-GPU**（Day-0 no-MTP ~2,200）。

---

## 8. 方案、测量法、Router 对比与原理（2026-07-21 凌晨深挖）

### 8.1 我用的方案
- **拓扑**：官方 `10P1D-dep4-dep32`（10 prefill×TP4/DP4/EP4 = 40 GPU + 1 decode×DEP32 = 32 GPU，共 18 节点 72 GPU，subblock-0001 单 NVL72 域）。
- **模型**：官方原版 `deepseek-ai/DeepSeek-V4-Pro`（FP4 MoE + FP8 attn，806G，**非** nvidia NVFP4 变体）。
- **镜像**：`lmsysorg/sglang:nightly-dev-cu13-20260720-b3570a45`（最新 nightly；官方 pin 的 0520 已被 GC）。
- **MoE**：`megamoe` a2a backend；W4A8→W4A4 由 env `USE_FP4_ACTS` 切换。
- **KV 传输**：mooncake over 域内 NVLink（`MC_FORCE_MNNVL=1`）。
- **编排（关键分歧点）**：我用 **sglang_router**（`--pd-disaggregation --policy cache_aware`）；**官方用 NVIDIA Dynamo**。

### 8.2 测量法（绕开 bench 客户端 tail-stall）
bench_serving 在高并发下**尾部少数请求 hang → 出不了 100% summary**。两个可靠替代：
1. **完成率 × 时间**：取 tqdm 进度（如 979/1024 @107s = 9.15 req/s），×9216 tok ÷72 GPU。
2. **decode 服务端日志**：对 32 个 DP rank 各取最新 `gen throughput`，去重求和（噪声较大，作交叉验证）。

### 8.3 ⭐ sglang_router 出了什么问题（tail-stall 根因）
- **现象**：任何并发 >1，每波尾部约 1-2 个请求**永久 hang**（conc4 卡 6/8、conc16 卡 62/64、conc256 卡 985/1024）；高并发（conc1024）hung 请求累积占 KV 槽 → **吞吐不升反降**（9.15→collapse）。
- **根因（GitHub 实锤）**：sglang_router 的 PD-disaggregation 在并发下对少数请求的 prefill→decode KV 交接有 race，请求 hang 后**不失败、不重试、不释放槽位**。相关 issue：#9266（高并发 sgl-router KV 传输失败）、#31206（circuit breaker per-leg：prefill 熔断后仍派 decode → decode 等永不来的 KV）、#12688（"PD requests failed with sgl-router，try upgrading"）、#5450（KV transfer 高并发变慢，未修）。
- **试过无效**：`--disable-circuit-breaker`（2→1 略好）、`--retry-max-retries 3`、`SGLANG_DISAGGREGATION_WAITING_TIMEOUT=90`（卡住请求不是 KV-wait 态所以 timeout 不触发）、`--disable-stream`。
- **自伤坑**：诊断时「直连 prefill:30000 绕 router」会触发 `AssertionError: bootstrap_room should not be None` 把 prefill controller 打崩（PD prefill 只能经 router 收请求，**绝不能直连**）。

### 8.4 ⭐ 官方 Router（Dynamo）好在哪
官方 recipe `frontend: type: dynamo`，用 `python -m dynamo.sglang` 起 worker + Dynamo frontend（需 NATS + ETCD 分布式运行时）。相比 sglang_router：
- **不用 load balancer**：Dynamo 有服务发现，**先路由到 decode，再 KV-aware 选 prefill**，请求生命周期由 Dynamo 统一管理。
- **请求级容错**：支持 request migration / cancellation（sgl-router 没有）——请求 hang/worker 挂时能迁移或取消，**不会累积死槽位**。这正是 sgl-router tail-stall 的解药。
- **KV-aware routing**：按前缀命中路由，减少重复 prefill。
- **代价**：要装 ai-dynamo[sglang] + NATS + ETCD + 18 worker 走 `dynamo.sglang` 重起 + frontend，是个大工程（官方 srt-slurm 自动化）。

### 8.5 原理小结（为什么 PD + 各项优化能冲上万）
- **PD 分离**：prefill 算力密集、decode 访存密集，拆开各自扩展 + 用满 GPU。KV 从 prefill 经 NVLink/RDMA 零拷贝送到 decode。
- **Wide-EP（DEP）**：decode 做 DP-attention + EP，专家摊薄、并发拉高。**但注意**：wide-EP 只在 prefill 喂得饱时提 per-decode-GPU；feed-limited 时铺得越宽 per-GPU 越低（见 §10.5）。
- **megamoe W4A4**：expert dispatch+GEMM 融合成一个 kernel，激活也压 4bit → MoE 层快 ~2×（实测 conc256 下 **1.48×**）。
- **SWA + online 压缩**：把 KV 打薄，让 decode 塞下**更大 batch**（收益在**高并发**才显现；conc256 下 KV 不是瓶颈，未见增益）。
- **MTP（EAGLE 投机解码）**：一次 forward 出多 token，per-user 提速——**11,200 的核心杠杆**（见 §10.2）。
- **要到官方 11,200**：需 Dynamo（高并发稳定）+ **dep8-MTP 配方** + 快 prefill 按 8:1 喂满 + 开环压测。完整路径见 §10。

### 8.6 从 sglang_router 到 Dynamo：踩坑与打通（历史过程，结论以 §10 为准）

这段是 2026-07-21 一夜从 sglang_router 撞墙、换 Dynamo、跑通 MTP 的过程，保留**可复用的真教训**（数字口径已按 §10 校正，早期草稿的 (in+out)÷72 错口径数已删）。

**① sglang_router 阶段的消融真相**（服务端测量，仅作趋势参考）：
- **W4A4 是最大单项**：megamoe W4A8→W4A4 约 **1.48×**（Run A→B），expert dispatch+GEMM 融合 + 激活压 4bit。
- **SWA/online 压缩在 conc256 无增益**（KV 未成瓶颈），收益要到高并发/大 batch 才显现。
- **sgl-router PD 在并发下有 tail-stall**：尾部少数请求永久 hang，不失败/不重试/不释放槽位（issue #9266/#31206/#12688/#5450）→ conc 上不去、MTP 压测出不了数。**这是必须换 Dynamo 的根因。**

**② Dynamo 打通的两个关键坑**：
- **"circuits open" 谜团真根因 = 僵尸 sglang::router 霸占 8000 端口**（不是 Dynamo 熔断 bug）。僵尸 router `/v1/models` 也返 200 迷惑人（`owned_by:local`），真 Dynamo frontend 是 `owned_by:nvidia`+`context_window:1048576`。dynamo.frontend 因端口占用静默 bind 失败崩溃，请求全打到僵尸 router 的死 prefill 熔断上。**排查法**：pod 内无 `ss`/`lsof`，用 `/proc/net/tcp` 反查监听 8000 的 PID，`kill -9` 后**换端口起全新 frontend**（熔断器初始闭合）。
- **反复 kill+relaunch dynamo.sglang → zombie 进程 + ZMQ 40236 残留 + GPU 显存泄漏 191GB 不释放**（`pkill` 杀不掉 D/Z 态，sleep-infinity PID1 不 reap 僵尸）→ 新进程必 OOM → **唯一可靠解 = `kubectl delete pod --force` 重建 pod**（模型在 node-local SSD 持久，podAffinity 保 subblock）。

**③ MTP 复现成功**（官方「2,200→11,200 靠 MTP」的核心杠杆）：
- **生效实锤**：decode 日志 `spec decode runtime metadata:{'nextn':1,'method':'EAGLE'}`；**TPOT 从无 MTP 43ms 稳到 11ms**（draft 被接受）。
- **MTP 四大坑（全解）**：(1) `online c128 does not support MTP`，online 压缩与 MTP 互斥→去掉；(2) draft token 放大每 rank token 数，`NUM_MAX_TOKENS_PER_RANK` 1280→4096；(3) ctx 9216 太小，MTP 令 8194+1024>9216 有 78% 请求被拒→benchmark OSL 降 960（提 ctx 则 cuda-graph OOM，回退）；(4) MTP draft 模型 + 投机 CUDA graph 吃显存，decode `mem-fraction 0.94→0.88` + `cuda-graph-max-bs 1280→512`。

**④ 最终天花板 + 结论**：dep8 + 8 prefill + MTP-3 稳态 = **6,659 output/decode-GPU**（官方口径），距 11,200 差 ~1.7×。~~gap 根因是 prefill 单卡慢 3.7×~~ **已被 §11 推翻**（prefill 高并发达官方 83%，非瓶颈）；真因待 sa-bench 开环重测（编排/decode 侧）。口径纠错、PD 配比公式见 §10。

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

但开环 + router 对齐只吃回 ~16%，**没吃回全部 gap**——剩余~~在 prefill 单卡慢~~ 真因见 §11（prefill 非瓶颈，待 sa-bench 开环重测编排/decode 侧）。

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
- **我们已对齐口径的真实数**：dep8 + 8 prefill + MTP-3 稳态 = 客户端 output 53,272/s ÷ 8 decode 卡 = **6,659 output/decode-GPU**（这个是对的口径）；dep32 同法 = 50,351 ÷ 32 = **1,573**（更低，见 10.3）。距 11,200 差 **~1.7×（dep8）**，不是之前写的 3.7×（那是错口径算的）。

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

### 10.4 【已推翻】曾以为"prefill 单卡慢 3.7×"——实为低并发测量假象（正确结论见 §11）

> ⚠️ 本节原结论（prefill 单卡吞吐慢官方 3.7×、18 节点配不出 P:D 比、必饿死 decode）**已被 2026-07-22 的并发扫描实测推翻**。保留于此仅作过程记录，**以 §11 为准**。

- 原观测：单个 8K prefill forward "1.44s"（官方 0.45s），据此推 prefill 慢 3.7×。**错在这个 1.44s 是在 conc4（极低并发、GPU 严重欠载）下量的**，不是 prefill 的真实吞吐上限。
- §11 实测：同一个 dep4 worker，并发从 conc4 拉到 conc128，prefill 吞吐 4,940 → **15,196 tok/s/GPU（= 官方 18,200 的 83%）**。prefill 单卡硬件/kernel **没有 3.7× 缺陷**，只是需要高并发才喂满 GPU。
- 所以"需 ~30 个 prefill worker、18 节点放不下 → 饿死 decode"这条推理**不成立**。满配跑不满 11,200 的真因需重新诊断（PD 编排是否把高并发投到每个 prefill、或瓶颈在 decode/KV），见 §11.5。

### 10.5 dep40 的硬约束 + wide-EP 在 feed-limited 下反而更差

- **dep40 非法(无 EPLB)**：DeepSeek-V4 有 **256 个专家，256 % 40 ≠ 0**，`assert num_physical_experts % ep_size == 0` 直接崩。dep40 **必须配 EPLB 加冗余专家**凑整（如 256+24=280，280/40=7）。所以 **EPLB 不只是 2.54× 加速，是 dep40 能启动的前提**。合法 EP 必须整除 256（8/16/32/64…）。
- **feed-limited 下 wide-EP 反而降 per-GPU**：实测 dep32（32 decode 卡）total output ~50K，dep8（8 卡）total output ~53K——**总输出都被 prefill 卡在 ~50K**，与 decode 拓扑无关。dep8 ÷8 = 6,659；dep32 ÷32 = 1,573。**prefill 喂不动时，decode 铺得越宽 per-GPU 越低**。研究说的"wide-EP 提 per-GPU"只在 prefill 供得上时成立（官方 prefill 快，我不快）。

### 10.6 「prefill 攒批再放」这招：官方没用，且被 decode 显存卡死

- 官方 11,200 是 **`req_rate inf` 持续喂料的稳态数**，**不用**攒批技巧。
- `slow_down` 攒批（issue #6017）是**压测 hack**（量 decode 峰值），不是产 11,200 的方法；且 **dynamo.sglang 上没有 slow_down HTTP 接口**，用不了。
- **根本上限**：prefill 过的 KV 得存进 decode 的 KV pool 等消费；pool 有限（dep8 ~400万 token → 最多攒 ~450 个 8K 请求，dep32 ~1300万 → ~1600 个）。攒满即到顶，放开后 decode drain 一波爆发就打回原形。**攒批 = 一次性 burst，被 KV pool 卡死，不可持续；稳态 11,200 绕不开"prefill 真快"**。
- 我实测 `SGLANG_HACK_PD_DECODE_NUM_RESERVED_DECODE_TOKENS=1026` 在 dynamo dep8 下**过度预分配**（499 prealloc 占满 KV、只 14 running），吞吐反降到 2,984——**这招在我环境有害**。

### 10.7 冲 11,200 的真正待办（按优先级，2026-07-22 修正）

> ⚠️ 原待办把"修 prefill 单卡吞吐"列为 #1，**已作废**（§11 证明 prefill 高并发下达官方 83%，不慢）。新优先级：

1. **换官方口径压测**：用 SemiAnalysis **sa-bench**（`req_rate inf` 开环 + conc 2500 + 自定义 DSV4 tokenizer）打 Dynamo frontend，别再用 sglang `bench_serving` 闭环 / 低并发单点。
2. **重新诊断满配瓶颈**：既然 prefill 单卡够快（§11），满配上不去 11,200 大概率在 (a) PD 编排（Dynamo router/queue）没把足够并发投到每个 prefill worker，或 (b) decode / KV 传输侧。逐项测。
3. **严格复现配方**：`high-conc-8p1d-dep8`（8 prefill : dep8 + MTP num-steps 1）+ 官方 sa-bench 开环。
4. **口径对齐**：一律「output ÷ decode-GPU」+ 开环高并发。
5. EPLB（decode 2.54×）留最后。

**一句话认知（修正版）**：**之前"prefill 单卡慢 3.7×"是低并发（conc4）测量假象——§11 实测同一 worker 高并发下达官方 83%。真正待解的是：官方口径（sa-bench 开环 c2500）下满配为什么只到 6,659——瓶颈已不在 prefill 单卡速度，要往 PD 编排 / decode 侧重新查。**

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

### 11.5 对 11,200 的启示（重新定向）
- prefill 单卡不是瓶颈（高并发下 83%）。满配 PD 之前只到 6,659 output/decode-GPU 的真因需重查：**(a) PD 编排（Dynamo router/queue）有没有把足够并发投递到每个 prefill worker；(b) 瓶颈是否在 decode / KV 传输侧**。
- 下一步：官方 **sa-bench 开环** + `high-conc-8p1d-dep8` 配方，端到端重测，定位真瓶颈。

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

---

*2026-07-22 更新（Local SSD based）。Phase 1（Flash）+ Phase 2（Pro）单节点 TP4 + **Phase 3 满配 PD（Dynamo / megamoe W4A4 / SWA / MTP）全部实测通过**。导航：§3.9 单节点手册、§3.3+§3.4 PD+Dynamo 步骤、§7 benchmark、§8 全过程、§10 官方 11,200 认知、**§11 prefill 并发扫描重大修正、§12 满配 sa-bench + 多 frontend 提升**。**当前最优（官方口径 output÷decode-GPU）= 8-frontend + high-conc-8p1d-dep8-mtp + sa-bench 开环 = 6,788 = 官方 11,200 的 61%**（从单 frontend 5,060 靠多 frontend +34% 提上来）。**瓶颈地图（§12.4）：编排/frontend 占 34%（已吃回）+ prefill 喂料受限 1.65×（单卡 1.2× kernel 差 + prefill 数量不够）；decode 非瓶颈。** 早期"prefill 单卡慢 3.7×"（§10.4）已被 §11 证伪（低并发测量假象）。存储全程 Local SSD RAID（见 gb300-local-ssd-raid0-SETUP.md）。R1 端到端见 `../deepseek-v3/`。*
