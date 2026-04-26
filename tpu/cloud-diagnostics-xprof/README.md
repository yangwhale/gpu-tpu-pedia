# Cloud Diagnostics XProf — TPU 性能 Profiling 实操指南

> **Cloud Diagnostics XProf** ([`AI-Hypercomputer/cloud-diagnostics-xprof`](https://github.com/AI-Hypercomputer/cloud-diagnostics-xprof)) 是 Google Cloud 官方的 **TPU/XLA workload profiler**。它在底层 [`openxla/xprof`](https://github.com/openxla/xprof)（前身是 TensorBoard Profiler）之上提供了 **托管 VM/GKE Pod 实例 + 一键 trace capture + 浏览器可视化** 的体验。
>
> **典型用途**：
> - 看 TPU 单机 / 多机推理的算子耗时分布（matmul / attention / collective）
> - 找 prefill / decode 时间分配（PD 分离前的诊断）
> - 看 HBM 带宽利用率、ICI 通信占比
> - 看 XLA / Pallas kernel 的实际编译结果

---

## 📖 怎么读这份文档

> **首次用 XProf**（10 分钟）→ §1 概念 → §2 5 分钟快速上手
>
> **为已有 vLLM 推理跑 profile** → §3 集成方式 → §3.3 vLLM-TPU
>
> **看 trace 不会** → §5 解读 Trace
>
> **遇到问题** → §6 常见踩坑

---

## 1. 概念 — XProf vs xprof vs xprofiler vs Cloud Diagnostics

容易混淆的三个名字：

| 名字 | 是什么 | 谁维护 |
|------|-------|--------|
| **`xprof`** ([openxla/xprof](https://github.com/openxla/xprof)) | 底层 profiler 引擎 + TensorBoard 插件，前身就是大家熟悉的 **TensorBoard Profiler** | OpenXLA |
| **`xprofiler`** (CLI 工具) | Cloud Diagnostics 提供的命令行包装器，简化"建实例 + 抓 trace + 看图" 整套流程 | Google AI-Hypercomputer |
| **Cloud Diagnostics XProf** | 整个产品的官方名字（含上面两块 + GKE 部署 + Pulumi 模板等）| Google Cloud |

**简单说**：
- `xprof` = 引擎（看图的内核）
- `xprofiler` = CLI（你日常打交道的命令）
- Cloud Diagnostics XProf = 官方品牌名

---

## 2. 🎯 5 分钟快速上手

### 2.1 安装 xprofiler 客户端

```bash
# 在 jumpbox / cloudtop / 你自己 workstation 上（不是 TPU VM）
python3 -m venv ~/.venv-xprof
source ~/.venv-xprof/bin/activate

pip install cloud-diagnostics-xprof

# 验证
xprofiler --help
```

### 2.2 准备 GCS bucket（存 trace 数据）

```bash
gcloud auth login
gcloud auth application-default login

# 给 xprofiler VM 的服务账户授权读写 bucket
PROJECT_ID=$(gcloud config get-value project)
GCS_BUCKET=gs://your-xprof-bucket          # 自己创建一个

# 默认服务账户（VM compute default SA）需要 Storage Object User
PROJECT_NUM=$(gcloud projects describe $PROJECT_ID --format='value(projectNumber)')
gcloud storage buckets add-iam-policy-binding $GCS_BUCKET \
    --member="serviceAccount:${PROJECT_NUM}-compute@developer.gserviceaccount.com" \
    --role="roles/storage.objectUser"
```

### 2.3 创建 xprofiler VM（看图用的服务器）

```bash
ZONE=us-central1-a   # 跟你的 TPU 同 region 即可
xprofiler create -z $ZONE -l $GCS_BUCKET

# 输出会给你一个 URL 和 SSH 命令：
#   1. https://<id>-dot-us-<region>.notebooks.googleusercontent.com
#   2. xprofiler connect -z <zone> -l <bucket> -m ssh
# 创建过程 ~3-5 分钟（VM 起来 + 装包）
```

> 💰 **成本**：默认 `c4-highmem-8` 实例，~$0.5/hr。**用完记得删**（见 §2.6）。

### 2.4 在 TPU workload 里加 profile collector（**关键前置**）

XProf 不会自动从 TPU 拉数据，**必须**在你的训练/推理代码里启用 collector。

```python
# JAX
import jax
jax.profiler.start_server(9012)   # 选一个端口

# PyTorch / torch_xla
import torch_xla.debug.profiler as xp
server = xp.start_server(9012)

# TensorFlow
import tensorflow.compat.v2 as tf2
tf2.profiler.experimental.server.start(9012)
```

> ⚠️ 这一步必须在 vLLM/training 启动前跑过；否则后面 capture 全拿不到数据。

### 2.5 Capture profile（抓 trace）

**方式 A: CLI 一键抓**（适合 GCE TPU VM）

```bash
xprofiler capture \
  -z $ZONE \
  -l $GCS_BUCKET/run1 \
  -f jax \
  -n my-tpu-vm-name \
  -d 2000          # 抓 2000ms = 2s

# 输出: Profile saved to gs://.../tensorboard/plugins/profile/<session_id>/
```

**方式 B: GKE Pod 抓**（vLLM-TPU 在 GKE 上跑用这个）

```bash
gcloud container clusters get-credentials <cluster> --region=<region>

xprofiler capture \
  -z $ZONE \
  -o gke \
  -l $GCS_BUCKET/run1 \
  -f jax \
  -n vllm-pod-name \
  -d 2000
```

**方式 C: TensorBoard UI 点按钮抓**（在浏览器里）— 见 §3.2。

### 2.6 看图（连 xprofiler 实例）

打开 `xprofiler create` 给的 URL，登录 Google 账号 → TensorBoard UI。
左侧 dropdown 选你的 run，看 **Overview / Trace Viewer / HLO Op / Memory Profile** 等标签。

> 🔒 **看完一定要删 VM**（不删一直计费）：
> ```bash
> xprofiler delete -z $ZONE -l $GCS_BUCKET
> ```

---

## 3. 三种 Profile Capture 方式对比

| 方式 | 适用场景 | 优点 | 缺点 |
|------|---------|------|------|
| **3.1 程序内代码 capture** | 训练循环精准抓某 step / inference 抓某请求 | **最精准**，可以选时机 | 需要改代码 |
| **3.2 TensorBoard UI 按钮** | 已部署 workload 临时调查 | 不改代码，浏览器即可 | 需要手动填 host:port |
| **3.3 `xprofiler capture` CLI** | 自动化 / CI / 多 host 同时抓 | 脚本友好 | 跟 3.2 等价，命令更长 |

### 3.1 程序内 capture（推荐生产场景）

**JAX**：
```python
# 方式 1: 显式 start/stop
jax.profiler.start_trace("gs://your-bucket/run1")
# ... 跑训练步 / 推理 ...
jax.profiler.stop_trace()

# 方式 2: context manager（推荐）
with jax.profiler.trace("gs://your-bucket/run1"):
  for step in range(10):
    train_step(...)
```

**PyTorch (torch_xla)**：
```python
xp.trace_detached(f"localhost:{9012}",
                  "gs://your-bucket/run1",
                  duration_ms=2000)

# 给每个 step 打 marker
for step, batch in enumerate(loader):
  with xp.StepTrace('train_step', step_num=step):
    train_step(batch)
```

### 3.2 TensorBoard UI 按钮 capture

1. 浏览器打开 xprofiler URL → TensorBoard
2. 左上角 dropdown 选 "**Profile**"
3. 点 "**CAPTURE PROFILE**" 按钮
4. 填 `Profile Service URL(s)` ⚠️ 这是 TPU VM 的 hostname **不是 xprofiler VM**：
   ```
   <TPU_VM_HOSTNAME>.<ZONE>.c.<PROJECT_ID>.internal:<PORT>
   ```
   例：`t1v-n-g8675e3i-w-0.us-east5-b.c.my-project.internal:9012`

5. 点 "**CAPTURE**"

> 拿到 hostname：`gcloud compute tpus tpu-vm ssh $TPU_NAME --zone=$ZONE --command="hostname"`

### 3.3 vLLM-TPU 内置 phased profiling（多机推理 daily benchmark 用）

[`vllm-project/tpu-inference`](https://github.com/vllm-project/tpu-inference) 的多机 nightly 脚本支持 `--phased-profiling-dir gs://...`，每个 phase（warmup / steady state / decode-only）自动各抓一个 trace：

```bash
# 完整脚本见 vllm-project/tpu-inference/scripts/multihost/benchmarks/torchax/xprof/
bash scripts/multihost/nightly_benchmarking.sh \
  --model-path "..." \
  --tp-size 16 \
  ...other-flags... \
  --phased-profiling-dir "gs://your-bucket/xprof/qwen3-coder/torchax/1k-8k"
```

之后用 `xprofiler create -l gs://your-bucket/xprof` 一次看所有 phase 对比。

---

## 4. ⭐ GCS 路径规范（容易踩坑）

xprofiler 对路径**有期待的格式**，搞错路径多个 run 不会聚合到同一个 view。

### 创建 instance 时

```bash
# ✅ 推荐：bucket 根路径，让 xprofiler 自动发现下面所有 run
xprofiler create -l gs://my-bucket

# ⚠️ 不推荐：路径过深，只能看一个 run
xprofiler create -l gs://my-bucket/run1/subdir/specific
```

### Capture 时

```bash
# ✅ 推荐：每个 run 一个子路径
xprofiler capture -l gs://my-bucket/run-2026-04-26-qwen3-coder ...
# capture 会自动建 .../tensorboard/plugins/profile/<session_id>/
```

### 自动产生的目录结构

```
gs://my-bucket/
├── run-2026-04-26-qwen3-coder/
│   └── tensorboard/plugins/profile/
│       ├── session_2026_04_26_07_30_15/      ← 每次 capture 一个 session
│       │   └── *.xplane.pb                    ← profile 数据
│       └── session_2026_04_26_07_45_22/
└── run-2026-04-26-deepseek-r1/
    └── tensorboard/plugins/profile/
        └── session_2026_04_26_08_12_03/
```

xprofiler UI 里左下角 dropdown 会展示成 `run-2026-04-26-qwen3-coder/session_2026_04_26_07_30_15` 等条目。

---

## 5. 解读 Trace — 6 个最常用的 view

打开 TensorBoard Profile tab，左侧 "Tools" dropdown 切换 view：

### 5.1 Overview Page（**首选**）
- **Step time breakdown**：高层显示 compute / collective / memory 占比
- **Top 10 ops**：哪些算子最耗时（一眼看出瓶颈）
- **MFU**：实际 vs peak FLOPS 占比
- 看完这个再决定下一步钻哪个 view

### 5.2 Trace Viewer
- 时序图（chrome://tracing 风格）
- **找跨 chip / 跨 host 的 collective 阻塞**：看 ICI / DCN bar
- **找 host overhead**：CPU 在干嘛 / 等谁

### 5.3 HLO Op Profile
- 按 XLA HLO op 聚合时间
- 适合**算子级**优化（如 fp8 vs bf16 matmul 哪个慢）

### 5.4 Memory Profile
- HBM 占用随时间变化
- 找 OOM 根因 / KV cache 增长曲线

### 5.5 Memory Viewer
- 静态 memory layout（程序启动时）
- 看哪些 buffer 持久占用 HBM

### 5.6 Pod Viewer（多机专用）
- 跨 host 时序对齐
- 看 ICI 和 DCN 哪个是瓶颈
- 看 multihost / PD 分离的 KV transfer

> 📚 **详细图解**：[openxla/xprof docs](https://github.com/openxla/xprof/blob/master/docs/)

---

## 6. 常见踩坑

### #0 🚨 **gLinux + LOAS2 环境 xprofiler create 不能用**（hulk 2026-04-26 实测踩过）

**症状**：`xprofiler create` 报 `ECP Proxy returned an error` / `gcloud crashed (ECPProxyError)`，setup VM 后内部 cleanup 也失败：
```
Unable to set up instance. Initiating cleanup.
ERROR: ECP Proxy returned an error
ERROR: gcloud crashed (ECPProxyError): ECP Proxy indicated an internal error: Failed to forward request
```

**根因**：xprofiler 内部用 subprocess 调 `gcloud compute instances delete`（cleanup 时）和其他 gcloud 子命令，这些子进程跟 gLinux 的 LOAS2 ECP Proxy 鉴权冲突。即使 `prodcertstatus` 显示 cert 有效，subprocess 仍会失败。手动跑 `gcloud compute instances delete` 反而成功，仅在 xprofiler subprocess 调用时 fail。

**修复**（优先级降序）：
1. ✅ **在 cloudtop 上跑 xprofiler**（不是 gLinux）— 推荐做法，cloudtop 没有 ECP 冲突
2. ✅ **在 GCE jumpbox VM 上跑 xprofiler** — 起一个小 VM (e2-small) 专门跑 xprofiler CLI
3. ⚠️ 在 gLinux 跑 — **目前不可用**，等上游修

> 💸 **代价**: hulk 实测时这个坑造成约 $0.10 的浪费 cost（4 个 VM × 几分钟 + cleanup leak），所以**强烈建议先确认环境**再跑。

### #-1 🚨 **xprofiler create 默认是交互式，`yes |` pipe 会建多个 VM**

**症状**：用 `yes | xprofiler create ...` 期望跳过确认，结果建了 N 个 VM（每个 c4-highmem-8 ~$0.5/hr）。

**根因**：xprofiler create 检测到"已有 instance for this bucket"会问"是否再建一个？"，`yes` 会无限回答 `y`，连续触发新建。

**修复**：
```bash
# ✅ 推荐：用 echo "y" 而不是 yes（只回答一次）
echo "y" | xprofiler create -z $ZONE -l $BUCKET --skip-creation-if-exists

# ❌ 千万别用 yes pipe — 实测会建 2-3 个 VM
yes | xprofiler create ...    # ❌
```

**验证不爆**：
```bash
gcloud compute instances list --filter="name~xprof"   # 应该只有 1 个
```

如果发现多个，立即并行删：
```bash
for VM in $(gcloud compute instances list --filter="name~xprof" --format="value(name)"); do
  gcloud compute instances delete $VM --zone=us-central1-a --quiet &
done
wait
```

### #-0.5 🚨 **xprofiler 客户端缺 `pyOpenSSL` 模块**

**症状**：第一次跑 xprofiler create 立刻报 `MutualTLSChannelError: No module named 'OpenSSL'`。

**根因**：`pip install cloud-diagnostics-xprof` 没把 `pyOpenSSL` 设为依赖，但 google-auth 的 mTLS 通道需要它。

**修复**：
```bash
pip install -U pyOpenSSL
# 或者直接装 extras (如果将来支持):
# pip install "cloud-diagnostics-xprof[mtls]"
```

### #-0.25 🚨 **gcloud active account 必须是 user 不是 SA**

**症状**：xprofiler 内部 gcloud 调用失败（即使 ADC 工作正常）。

**根因**：`gcloud auth list` 显示 active account 不是用户邮箱（如 `chrisya@google.com`），而是默认 SA（如 `insecure-cloudtop-shared-user@cloudtop-hk.iam.gserviceaccount.com`）。

**修复**：
```bash
gcloud auth list   # 确认 active account 是用户
gcloud config set account chrisya@google.com   # 切回 user
```

> ⚠️ 注意：在 venv 里激活后再跑 gcloud 命令，可能 active account 行为跟 host shell 不一样 — 每次都验证一遍。


### #1 ⚠️ Capture 返回成功但 TensorBoard 看不到 trace

**根因**：collector 端口（9012）没起。`jax.profiler.start_server(9012)` 漏了或没在 capture 之前跑。

**修复**：
```python
# 在 vllm serve / 训练脚本最开头加：
import jax
jax.profiler.start_server(9012)
```

### #2 ⚠️ `Profile Service URL` 填错

**症状**：UI 抓不到 trace，提示 connection refused。

**根因**：填的是 xprofiler VM 的 hostname（错），应该是 **TPU VM 自己的 hostname**。

**修复**：用 `gcloud compute tpus tpu-vm ssh ... --command="hostname"` 拿正确 hostname。

### #3 ⚠️ Profile loading 超过 3 分钟（UI 卡）

**根因**：profile size > 1 GB，默认 c4-highmem-8 不够。

**修复**：删 VM 重建用更大机器：
```bash
xprofiler delete -z $ZONE -l $GCS_BUCKET
xprofiler create -z $ZONE -l $GCS_BUCKET --machine-type=c4-highmem-32
```

### #4 ⚠️ IAM `Permission denied` on bucket

**根因**：xprofiler VM 的 SA 没有 bucket 权限。

**修复**：
```bash
gcloud storage buckets add-iam-policy-binding $GCS_BUCKET \
    --member="serviceAccount:${PROJECT_NUM}-compute@developer.gserviceaccount.com" \
    --role="roles/storage.objectUser"
```

### #5 ⚠️ GKE 上 capture pod 失败

**根因**：kubectl context 不对 / pod 名拼错。

**修复**：
```bash
kubectl config current-context           # 确认对的集群
kubectl get pods -o wide                 # 确认 pod 名
xprofiler capture -o gke -n <pod_name> ...   # 必须加 -o gke
```

### #6 ⚠️ 忘了删 VM，账单暴涨

```bash
xprofiler list -z $ZONE                   # 列出所有
xprofiler delete -z $ZONE -l $GCS_BUCKET  # 按 GCS 删
# 或按 VM 名删:
xprofiler delete -z $ZONE --vm-name xprof-<uuid>
```

---

## 7. 进阶：GKE 上部署 xprofiler

如果你的 TPU workload 在 GKE 跑（如 vLLM 单机 / multihost / PD 分离），把 xprofiler 也部到同集群可以：
- 减少跨网络延迟（同 VPC）
- 用 K8s 标准管理（Service / RBAC / 共享集群）

完整步骤参见 [上游 README "Create Xprofiler - GKE"](https://github.com/AI-Hypercomputer/cloud-diagnostics-xprof?tab=readme-ov-file#create-xprofiler---gke) 一节，含：
1. 创建 namespace + ServiceAccount
2. 配置 GCP SA / K8s SA / Workload Identity 绑定
3. 给 SA 授权 GCS
4. `xprofiler create --gke -l $GCS -z <zone>`

或者用 [Pulumi 模板一键部署](https://github.com/AI-Hypercomputer/cloud-diagnostics-xprof/blob/main/xprofiler-with-pulumi.md)。

---

## 8. 参考资料

| 资源 | 链接 |
|------|------|
| 官方 GitHub | [AI-Hypercomputer/cloud-diagnostics-xprof](https://github.com/AI-Hypercomputer/cloud-diagnostics-xprof) |
| 底层引擎 xprof | [openxla/xprof](https://github.com/openxla/xprof) |
| JAX profiling 文档 | [docs.jax.dev/.../profiling](https://docs.jax.dev/en/latest/profiling.html) |
| PyTorch on TPU profiling | [cloud.google.com/.../pytorch-xla-performance-profiling-tpu-vm](https://cloud.google.com/tpu/docs/pytorch-xla-performance-profiling-tpu-vm) |
| TensorFlow profiler | [tensorflow.org/guide/profiler](https://www.tensorflow.org/guide/profiler) |
| vLLM-TPU phased profiling 示例 | [tpu-inference/scripts/multihost/benchmarks/torchax/xprof/](https://github.com/vllm-project/tpu-inference/tree/main/scripts/multihost/benchmarks/torchax/xprof) |
| TensorBoard Profiler 历史教程 | [tensorflow.org/tensorboard/tensorboard_profiling_keras](https://www.tensorflow.org/tensorboard/tensorboard_profiling_keras) |

---

## ⚠️ 关于本文档

本文档基于 [`AI-Hypercomputer/cloud-diagnostics-xprof`](https://github.com/AI-Hypercomputer/cloud-diagnostics-xprof) 官方 README + [`vllm-project/tpu-inference`](https://github.com/vllm-project/tpu-inference) 实际用法整理 + **hulk 在 gLinux 上 dogfood 实测**（2026-04-26）。

### Dogfood 验证状态（2026-04-26）
- ✅ §2.1 安装 xprofiler — 实测通过（但发现需要补装 `pyOpenSSL`，见 §6 #-0.5）
- ✅ §2.2 创建 GCS bucket + SA 授权 — 实测通过
- ❌ §2.3 `xprofiler create` — **在 gLinux + LOAS2 环境无法工作**（ECP Proxy 错误，见 §6 #0）
- ⏳ §2.4-2.6 + §3-§5 — 因 §2.3 阻塞未实测，需在 cloudtop / GCE jumpbox 上验证

### 本次 dogfood 找到的 4 个坑（已写入 §6）
- **#0**: gLinux + LOAS2 环境 xprofiler create 不能用（最严重）
- **#-1**: `yes |` pipe 会建多个 VM（实测一次创了 3 个）
- **#-0.5**: 缺 `pyOpenSSL` 包（需手动补装）
- **#-0.25**: gcloud active account 不能是 SA

**本文档的精神**：复制粘贴跑得通 + 踩坑明示 + 把上游分散文档整理成一个客户友好的入口。

---

📅 创建：2026-04-26 by hulk
