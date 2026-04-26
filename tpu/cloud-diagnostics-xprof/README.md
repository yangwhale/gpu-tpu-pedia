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

### 1.1 XProf vs 其他 GCP profiler — 别选错工具

GCP 上还有几个名字相似的 profiler，新人很容易选错：

| 工具 | 抓什么 | 适用场景 |
|------|------|------|
| **Cloud Diagnostics XProf**（本文档） | TPU/GPU XLA workload 的算子 / HBM / 集体通信 | **TPU/GPU ML 训练或推理**的性能分析 |
| **Cloud Profiler** | 长时间运行服务的 CPU / heap / wall time / contention | 生产 server 的 always-on 性能监控（Java / Python / Go service） |
| **Cloud Trace** | 分布式 RPC 调用链 | 微服务延迟分析、找跨服务瓶颈 |
| **TensorBoard Profiler**（旧名） | 同 XProf — 就是 xprof 的前身 | 老代码 / 老教程里的叫法，现在统一叫 XProf |
| **NVIDIA Nsight Systems** | GPU SM 占用、CUDA stream、kernel launch | **GPU 上深入到 CUDA 层**的优化（XProf 看不到 SM 级细节）|

**一句话路由**：
- 性能问题在 **TPU/GPU 上算子层** → XProf
- 性能问题在 **CPU 服务进程** → Cloud Profiler
- 性能问题在 **跨服务调用** → Cloud Trace
- 想看 **GPU CUDA 内部细节**（如 warp 利用率） → Nsight

### 1.2 什么时候 NOT 用 XProf

XProf 不是万能锤子，下面这些场景应该选别的工具：

- ❌ **纯 CPU Python 程序慢** → 用 [`py-spy`](https://github.com/benfred/py-spy) 或 `cProfile`
- ❌ **Pod / 容器内存泄漏** → 用 `pprof` (Go) 或 `tracemalloc` (Python)
- ❌ **gRPC / HTTP 服务延迟** → 用 Cloud Trace 或 OpenTelemetry
- ❌ **K8s Pod 调度问题** → 看 `kubectl describe` + GKE event
- ❌ **磁盘 I/O 瓶颈** → 用 `iotop` / `iostat`，XProf 不抓 disk

> 💡 经验：XProf 是给 **"workload 已经能跑、想找 TPU/GPU 上算子层瓶颈"** 用的。如果你的瓶颈在 CPU、网络、调度、I/O，先用对应专项工具排查。

### 1.3 成本概览

| 项目 | 成本 |
|------|------|
| xprofiler VM (默认 c4-highmem-8) | **~$0.5/hr** — 看完不删会持续计费 |
| GCS 存储 trace | 单次 trace 一般几十 MB ~ 几 GB，~$0.02/GB/月 |
| 网络出口（看图时） | 同 region 内免费，跨 region 才计费 |

**省钱姿势**：
1. 看完图立刻 `xprofiler delete -z $ZONE -l $BUCKET`（最常忘的事 — 见 §6 #10）
2. 多个 run 共用一个 xprofiler 实例（只创一次 VM）
3. 不要用 `yes |` pipe 跑 create — 会创多个 VM（见 §6 #4）

---

## 2. 🎯 5 分钟快速上手

> 🚨 **开跑前先看 — 当前已知会卡住的 2 个坑**（2026-04-26 hulk dogfood 实测）：
>
> 1. **§2.3 `xprofiler create` 在 Debian 12 / 现行 cloudtop / gLinux 上 100% 失败** — VM startup script 装 `python3-distutils`，但该包在 Debian 12 已被移除。详见 §6 #1，先看完再开始。
> 2. **§2.1 `pip install cloud-diagnostics-xprof` 不会自动装 `pyOpenSSL`**，第一次跑会报 `MutualTLSChannelError`。装包时一并 `pip install pyOpenSSL` 就行。
>
> 这两个坑当前还没修，但跟着下面步骤走到对应位置，会有内联提醒。

### 2.1 安装 xprofiler 客户端

```bash
# 在 jumpbox / cloudtop / 你自己 workstation 上（不是 TPU VM）
python3 -m venv ~/.venv-xprof
source ~/.venv-xprof/bin/activate

pip install cloud-diagnostics-xprof
pip install -U pyOpenSSL    # ⚠️ 必装 — 上游漏配 mTLS 依赖，详见 §6 #2

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

> 🚨 **必读**：当前 (2026-04-26) `xprofiler create` 在 Debian 12 / cloudtop / gLinux 上**会失败**。原因 + 临时解法见 §6 #1。如果你只是想本地看截图熟悉 UI，跳到 §3.4 用本地 TensorBoard 替代。

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

### 3.4 ✅ 本地 TensorBoard + xprof plugin（**推荐 fallback** — 绕过 §6 #1 必踩坑）

如果你只是想**看 UI 熟悉 XProf**，或者 `xprofiler create` 一直起不来 VM（当前 Debian 12 必踩），用本地 TensorBoard 跑完全一样的 UI：

```bash
python3 -m venv ~/.venv-xprof-local
source ~/.venv-xprof-local/bin/activate
pip install -U tensorboard tensorboard-plugin-profile "setuptools<70"   # setuptools<70 必加，否则 pkg_resources 报错

# 把 GCS 上的 trace 拉到本地（或本地直接生成）
gsutil cp -r gs://your-bucket/run1 /tmp/local-trace

tensorboard --logdir=/tmp/local-trace --port=6006 --bind_all
# → 浏览器开 http://localhost:6006/  → 选 PROFILE tab
```

**优势**：
- 不用建 VM、不用付钱、不用等 5 分钟
- UI 和 xprofiler 实例完全一样（同一个 plugin）
- 可以直接打开 `.xplane.pb` 文件

**劣势**：
- `--bind_all` 只在本机能访问；想分享给团队还是要走 xprofiler VM 的 inverting-proxy URL
- 大 profile (>1 GB) 本机可能内存不够，xprofiler 默认 c4-highmem-8 (64 GB) 更稳

> 📚 hulk 用这个方案跑通了完整的 Demo，含 Overview / Trace Viewer / HLO Stats 截图：[https://cc.higcp.com/pages/xprof-demo-20260426.html](https://cc.higcp.com/pages/xprof-demo-20260426.html)

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

打开 TensorBoard Profile tab，左侧 "Tools" dropdown 切换 view。下面截图来自 hulk 在 gLinux 上跑一个 JAX CPU demo 的真实 profile（[完整 demo 页](https://cc.higcp.com/pages/xprof-demo-20260426.html)，含 .pb 文件下载）。**UI 和 TPU profile 完全一样**，只是 device-side view 在 CPU 下没数据。

### 5.1 Overview Page（**首选**）

![XProf Overview Page](https://cc.higcp.com/assets/xprof-overview-20260426.png)

**看这一张就能判断"该往哪钻"**：
- **Step time breakdown**：高层显示 compute / collective / memory 占比
- **Top 10 ops**：哪些算子最耗时
- **MFU** (Model FLOPs Utilization)：`实际 FLOPS / peak FLOPS`，TPU v5p 训练 60%+ 算健康，<30% 一般是被 input pipeline 或 collective 卡住
- **Run Environment**：device 类型、host 数、core 数 — 先确认你抓到了想抓的硬件

**🩺 诊断实例**：

| 你看到 | 大概率原因 | 下一步 |
|------|------|------|
| Device idle > 30% | Input pipeline 太慢喂不上 | 钻 §5.2 Trace Viewer 看 host CPU/IO bar 的位置 |
| Device collective > 20% | 跨芯片通信瓶颈 | 钻 §5.6 Pod Viewer 看 ICI/DCN 哪个慢 |
| Compilation Time 异常大 | 形状不固定 / dynamic shape 频繁重编译 | 看 HLO Op Stats 里 jit recompile 次数 |
| MFU < 30% 但 device 不 idle | 算子选择/编译效果差 | 钻 §5.3 HLO Op Profile 看具体哪个 op GFLOPs/sec 低 |

### 5.2 Trace Viewer

![XProf Trace Viewer](https://cc.higcp.com/assets/xprof-trace-viewer-expanded-20260426.png)

时序图（chrome://tracing 风格）— 横轴时间，纵轴每个 process / thread 一行。
- **找跨 chip / 跨 host 的 collective 阻塞**：看 ICI / DCN bar 是不是覆盖了大段时间
- **找 host overhead**：CPU 在干嘛 / 等谁
- 操作：拖动平移、滚轮缩放、`f` 键 fit-to-view、`m` 键标记区间

**🩺 诊断实例**：

| 时序图上看到 | 含义 |
|------|------|
| Device 行有大段空白 | TPU 在等输入或等同步，对照 host 行看是 IO 还是 collective |
| HBM ↔ device 频繁来回的小 op | KV cache 没有 layout 对齐 / 没用 `jax.jit` 包起来 |
| Step 之间间隔忽长忽短 | dynamic batching 或 prefill/decode 混跑（prefill 进来就拖累 decode）|

> 💡 **想直接交互式看 Trace？** 任何 trace 文件下都有 `*.trace.json.gz`，下载后拖到 [ui.perfetto.dev](https://ui.perfetto.dev/)，**无需装任何东西**就能在浏览器里看，比 xprofiler VM 快很多。

### 5.3 HLO Op Profile

![XProf HLO Op Stats（CPU profile 显示 No data 是预期）](https://cc.higcp.com/assets/xprof-hlo-stats-20260426.png)

按 XLA HLO op 聚合时间，适合**算子级**优化。
- 排序看哪个 op 总耗时最多
- 看 GFLOPs/sec — 实际 vs 理论峰值，差距大的 op 是优化对象
- 同时看 `Bound by`：compute-bound 还是 memory-bound

> ⚠️ 上面截图显示 "No data" — 因为是 CPU profile，HLO 是 device-side 概念。**真 TPU/GPU profile 这里会有完整 op 列表**（matmul / convert / dot / convolution 等）。

**🩺 诊断实例**：

| 看到 | 含义 |
|------|------|
| `dot` (matmul) op 单独占 50%+ 但 GFLOPs/sec 远低于 peak | bf16/fp8 dtype 没生效，或者 contracting dim 没对齐到 128 |
| `convert` op 排进 top-5 | 频繁 dtype cast，检查中间张量是不是没用统一精度 |
| `all-reduce` / `all-gather` 占比高 | 集体通信成本，考虑减小 sharding 切分粒度 |

### 5.4 Memory Profile

HBM 占用**随时间**变化（动态曲线）。
- 找 OOM 根因 — 在 OOM 之前 HBM 是怎么涨上去的
- 看 KV cache 增长曲线 — 推理时 KV 单调上涨，到达瓶颈点
- 看 activation peak — 训练时反向时是不是某个 layer activation 异常大

**🩺 诊断实例**：

| 看到 | 含义 |
|------|------|
| HBM 在某个 step 突然跳一大块 | 该 step 多了一个临时 tensor，没及时释放 |
| KV cache 涨速 > 预期 | batch 没及时释放，或者 paged attention 没生效 |
| Activation peak 出现在某 layer 反向 | recompute / gradient checkpoint 没覆盖到该 layer |

### 5.5 Memory Viewer

**静态** memory layout（程序启动时）— 看哪些 buffer 持久占用 HBM。
- 模型参数 vs activation buffer vs scratch space 的占比
- HBM 占用最大的几个 buffer 名字（HLO 起的名）

**🩺 诊断实例**：模型加载完 HBM 占 80%+ → 留给 KV cache 的空间不够 → 推理时大概率 OOM 或 batch 跑不起来。

### 5.6 Pod Viewer（多机专用）

跨 host 时序对齐 — multihost / PD 分离场景必看。
- 看 ICI（机内）和 DCN（跨机）哪个是瓶颈
- 看 multihost / PD 分离的 KV transfer 耗时
- 看不同 host 的 step 是不是对齐（不对齐说明 straggler 拖累全局）

**🩺 诊断实例**：

| 看到 | 含义 |
|------|------|
| 某个 host 的 step 始终晚 5ms+ 完成 | straggler — 该 host CPU/IO/网络异常，需要换机 |
| DCN bar 覆盖大段时间但 ICI 几乎没有 | 跨机通信是瓶颈，考虑 colocate 或减少 cross-host sharding |
| KV transfer (PD 分离) 占 prefill 30%+ | TPUConnector 配置不对，或 chunked transfer 没开 |

> 📚 **详细图解**：[openxla/xprof docs](https://github.com/openxla/xprof/blob/master/docs/)

---

## 6. 常见踩坑

### #1 🚨 **xprofiler VM 的 startup script 在 Debian 12 上跑不完**（hulk 2026-04-26 深度实测）

**症状**：`xprofiler create` 等了 ~5 分钟后报：
```
Unable to set up instance. Initiating cleanup.
```
然后 cleanup 阶段 subprocess 调 `gcloud compute instances delete` 撞 ECP Proxy error（这个是次要派生症状）。

**真根因**（深挖发现）：

xprofiler create 实际成功创建 VM（gcloud compute instances create 返回 RUNNING），但 VM 上的 startup script **失败在很早期**，没机会把 `startup_output.json` 上传到 GCS 让 xprofiler 知道 setup 完成。

xprofiler 默认 polling 16 次（每次 ~20s, 总 ~5 min）等 GCS 上的 `startup_output.json` 出现 → 永远等不到 → 宣布 "Unable to set up instance"。

**深挖出的真问题**（基于 startup script 内容推断）：

xprofiler create 的 startup script 第一步是：
```bash
apt-get install -yq git supervisor python3 python3-pip python3-distutils python3-virtualenv
```

**`python3-distutils` 在 Debian 12 (bookworm) 已经被移除**（Python 3.11 起 distutils 被废弃，包改名为 `python3-setuptools`）。`apt-get install` 报 "no installation candidate" → 整个 startup 立刻 exit → `gsutil cp startup_output.log` 这步根本没机会跑 → GCS bucket 完全空 → xprofiler 永远等不到 ready 信号。

**怎么验证**：
```bash
# 1. 不让 xprofiler 自动删 VM
xprofiler create -z $ZONE -l $BUCKET --auto-delete-on-failure-off

# 2. setup fail 后 VM 还在，看 serial console
gcloud compute instances get-serial-port-output xprof-<uuid> --zone=$ZONE | grep -E "ERROR|distutils|installation candidate"

# 3. 或 SSH 进去看 syslog
gcloud compute ssh xprof-<uuid> --zone=$ZONE -- "sudo cat /var/log/syslog | grep -i distutils"
```

**修复**（按可行性排序）：

1. 🟡 **等上游修 startup script** — [报 issue 给 AI-Hypercomputer/cloud-diagnostics-xprof](https://github.com/AI-Hypercomputer/cloud-diagnostics-xprof/issues)，建议改用 `python3-setuptools` 或显式指定 image-family 为更老的 debian-11
2. 🟡 **手动 patch startup script** — fork xprofiler 改 `create_action.py` 里 startup script 的 apt 行
3. ✅ **临时 workaround**：用 `--vm-name` + `--auto-delete-on-failure-off`，setup fail 后手动 SSH 进 VM 把缺的包装上、起 tensorboard，再继续用 — 麻烦但能跑通

> 🌐 **不是 gLinux 专属**：这个 startup script 失败跟你在哪跑 xprofiler 客户端无关（cloudtop / jumpbox / gLinux 都一样）— **VM 上的问题是普遍性的**。但 cloudtop / jumpbox 不会撞 ECP cleanup 错误，所以表现可能稍好。

### #2 🚨 **xprofiler 客户端缺 `pyOpenSSL` 模块**

**症状**：第一次跑 xprofiler create 立刻报 `MutualTLSChannelError: No module named 'OpenSSL'`。

**根因**：`pip install cloud-diagnostics-xprof` 没把 `pyOpenSSL` 设为依赖，但 google-auth 的 mTLS 通道需要它。

**修复**：
```bash
pip install -U pyOpenSSL
# 或者直接装 extras (如果将来支持):
# pip install "cloud-diagnostics-xprof[mtls]"
```

### #3 🚨 **gcloud active account 必须是 user 不是 SA**

**症状**：xprofiler 内部 gcloud 调用失败（即使 ADC 工作正常）。

**根因**：`gcloud auth list` 显示 active account 不是用户邮箱（如 `chrisya@google.com`），而是默认 SA（如 `insecure-cloudtop-shared-user@cloudtop-hk.iam.gserviceaccount.com`）。

**修复**：
```bash
gcloud auth list   # 确认 active account 是用户
gcloud config set account chrisya@google.com   # 切回 user
```

> ⚠️ 注意：在 venv 里激活后再跑 gcloud 命令，可能 active account 行为跟 host shell 不一样 — 每次都验证一遍。

### #4 🚨 **xprofiler create 默认是交互式，`yes |` pipe 会建多个 VM**

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


### #5 ⚠️ Capture 返回成功但 TensorBoard 看不到 trace

**根因**：collector 端口（9012）没起。`jax.profiler.start_server(9012)` 漏了或没在 capture 之前跑。

**修复**：
```python
# 在 vllm serve / 训练脚本最开头加：
import jax
jax.profiler.start_server(9012)
```

### #6 ⚠️ `Profile Service URL` 填错

**症状**：UI 抓不到 trace，提示 connection refused。

**根因**：填的是 xprofiler VM 的 hostname（错），应该是 **TPU VM 自己的 hostname**。

**修复**：用 `gcloud compute tpus tpu-vm ssh ... --command="hostname"` 拿正确 hostname。

### #7 ⚠️ Profile loading 超过 3 分钟（UI 卡）

**根因**：profile size > 1 GB，默认 c4-highmem-8 不够。

**修复**：删 VM 重建用更大机器：
```bash
xprofiler delete -z $ZONE -l $GCS_BUCKET
xprofiler create -z $ZONE -l $GCS_BUCKET --machine-type=c4-highmem-32
```

### #8 ⚠️ IAM `Permission denied` on bucket

**根因**：xprofiler VM 的 SA 没有 bucket 权限。

**修复**：
```bash
gcloud storage buckets add-iam-policy-binding $GCS_BUCKET \
    --member="serviceAccount:${PROJECT_NUM}-compute@developer.gserviceaccount.com" \
    --role="roles/storage.objectUser"
```

### #9 ⚠️ GKE 上 capture pod 失败

**根因**：kubectl context 不对 / pod 名拼错。

**修复**：
```bash
kubectl config current-context           # 确认对的集群
kubectl get pods -o wide                 # 确认 pod 名
xprofiler capture -o gke -n <pod_name> ...   # 必须加 -o gke
```

### #10 ⚠️ 忘了删 VM，账单暴涨

```bash
xprofiler list -z $ZONE                   # 列出所有
xprofiler delete -z $ZONE -l $GCS_BUCKET  # 按 GCS 删
# 或按 VM 名删:
xprofiler delete -z $ZONE --vm-name xprof-<uuid>
```

---

## 7. 进阶：GKE 上部署 xprofiler

如果你的 TPU workload 在 GKE 跑（如 vLLM 单机 / multihost / PD 分离），把 xprofiler 也部到同集群有两个好处：
- 减少跨网络延迟（同 VPC，看大 profile 加载更快）
- 用 K8s 标准管理（Service / RBAC / 共享集群）

⚠️ hulk 还没在 GKE 上 dogfood 过，**直接看上游官方步骤**：
- [Create Xprofiler — GKE](https://github.com/AI-Hypercomputer/cloud-diagnostics-xprof?tab=readme-ov-file#create-xprofiler---gke)（手动步骤：namespace + WI + SA 授权 + `xprofiler create --gke`）
- [Pulumi 一键模板](https://github.com/AI-Hypercomputer/cloud-diagnostics-xprof/blob/main/xprofiler-with-pulumi.md)（IaC 自动化）

⚠️ §6 #1 的 Debian 12 坑在 GKE 上**还没验证是否复现** — 上游 Pod image 可能用的是更老的 base，但没确认。如果踩到，参考 §7.5 诊断 recipe。

---

## 7.5 🔧 诊断 setup 失败的完整 recipe

如果 `xprofiler create` 报 `Unable to set up instance`，按下面 4 步**保留 VM + 看真错误**：

### Step 1: 用 `--auto-delete-on-failure-off` 起 VM
让 setup fail 时 **VM 不被自动删**，方便事后诊断：

```bash
xprofiler create \
  -z us-central1-a \
  -l gs://your-bucket \
  --skip-creation-if-exists \
  --auto-delete-on-failure-off \
  --verbose 2>&1 | tee /tmp/xprof-create.log

# 等 ~5-10 分钟后，VM 还会在
gcloud compute instances list --filter="name~xprof"
```

### Step 2: 抓 serial port output（不需要 SSH）

VM 上的 startup script 输出会进 serial console：

```bash
VM_NAME=xprof-<uuid>     # 从上一步 list 拿到
gcloud compute instances get-serial-port-output $VM_NAME \
  --zone=us-central1-a 2>&1 \
  | tee /tmp/serial.log

# 找错误关键字
grep -E 'ERROR|Error|FAIL|fail|cannot|not found|installation candidate' /tmp/serial.log | head -20
```

### Step 3: SSH 进 VM 看完整 startup log

如果 serial 不够细，SSH 进去看 syslog / startup-script log：

```bash
gcloud compute ssh $VM_NAME --zone=us-central1-a -- "
  echo '=== Last 50 lines of startup-script log ==='
  sudo journalctl -u google-startup-scripts.service -n 50 --no-pager
  echo
  echo '=== TensorBoard process ==='
  ps -ef | grep tensorboard | grep -v grep
  echo
  echo '=== Docker container ==='
  sudo docker ps -a
"
```

### Step 4: 看完后清理 VM（**别忘了，否则一直收钱**）

```bash
gcloud compute instances delete $VM_NAME --zone=us-central1-a --quiet
gcloud storage rm -r gs://your-bucket --quiet   # 也清理 bucket（如果用完了）
```

### 常见 startup script 失败模式

| 症状 | 根因 | 修复 |
|------|------|------|
| `Unable to locate package python3-distutils` | Debian 12 移除该包 | 等上游修；或 fork 改 startup script 用 `python3-setuptools` |
| `Unable to fetch some archives` (apt) | VM 在受限 VPC 没 internet | 给 VM subnet 配 NAT / Cloud NAT |
| TensorBoard 19 次 polling 全 fail | pip install tensorflow-cpu 网络 timeout | 改用预装 image / 加 PyPI mirror |
| `gcloud storage cp ... permission denied` | VM SA 没 GCS 权限（见 §2.2 授权步骤）| 重做 §2.2 IAM binding |
| Inverting-proxy docker 起不来 | gcr.io 拉镜像失败 | VM 需要 internet + Container Registry 权限 |

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
- ✅ §2.1 安装 xprofiler — 实测通过（但发现需要补装 `pyOpenSSL`，见 §6 #2）
- ✅ §2.2 创建 GCS bucket + SA 授权 — 实测通过
- ❌ §2.3 `xprofiler create` — **VM 上 startup script 在 Debian 12 失败**（真根因 `python3-distutils` 包在 Debian 12 已被移除，见 §6 #1）— 跟你在哪跑 xprofiler 客户端无关
- ⏳ §2.4-2.6 + §3 — 因 §2.3 阻塞未实测在 xprofiler VM 上跑通
- ✅ §5 解读 Trace — **本地 TensorBoard 跑通了**（绕过 xprofiler VM），生成真 .xplane.pb + 4 个 view 截图，详见 [demo 页](https://cc.higcp.com/pages/xprof-demo-20260426.html)

### 本次 dogfood 找到的 4 个坑（已写入 §6）
- **#1**: VM startup script 在 Debian 12 失败 — **真根因是 `python3-distutils` 包被移除**（最严重，所有 cloudtop / gLinux 都中招）
- **#2**: 缺 `pyOpenSSL` 包（装完 `cloud-diagnostics-xprof` 第一次跑就撞 mTLS 错）
- **#3**: gcloud active account 不能是 SA（cloudtop 默认 active 是共享 SA）
- **#4**: `yes |` pipe 会建多个 VM（实测一次创了 3 个）

**本文档的精神**：复制粘贴跑得通 + 踩坑明示 + 把上游分散文档整理成一个客户友好的入口。

---

📅 创建：2026-04-26 by hulk
