# TPU v7 Benchmark 完全指南

> ALModel (MaxText) 在 GKE TPU v7 (Ironwood) 集群上的 Benchmark 操作手册。
> 从零开始，一步步跑通 benchmark，收集性能数据。

**适用环境**: GKE 1.34+, TPU v7 (tpu7x), Kueue + JobSet, MaxText
**最后更新**: 2026-03-06
**实测数据**: ALModel 17B MoE, 32 chips (2×4×4), batch=12

---

## 目录

1. [前置条件](#1-前置条件)
2. [集群准备](#2-集群准备)
3. [安装 Kueue (首次)](#3-安装-kueue-首次)
4. [生成 JobSet YAML](#4-生成-jobset-yaml)
5. [提交 Benchmark](#5-提交-benchmark)
6. [监控和收集结果](#6-监控和收集结果)
7. [清理](#7-清理)
8. [Sync/Async 计时模式切换](#8-syncasync-计时模式切换)
9. [实测 Benchmark 数据](#9-实测-benchmark-数据)
10. [时间线预期](#10-时间线预期)
11. [常见问题和踩坑](#11-常见问题和踩坑)
12. [TFLOP/s 计算 Bug 分析（2026-03-06 发现）](#12-tflops-计算-bug-分析2026-03-06-发现)
13. [附录：完整配置参考](#13-附录完整配置参考)

---

## 1. 前置条件

### 1.1 工具安装

```bash
# gcloud CLI (已认证)
gcloud auth list

# kubectl
kubectl version --client

# xpk (用于 Kueue 安装，不用于提交任务)
pip install xpk==0.16.1

# gsutil (gcloud SDK 自带)
gsutil version
```

### 1.2 需要的资源

| 资源 | 说明 |
|------|------|
| GKE 集群 | 1.34+ 版本，带 TPU v7 节点池 |
| TPU 节点池 | tpu7x-standard-4t (每节点 4 chips)，8 节点 = 32 chips |
| GCS bucket | 存放代码和训练输出 |
| 容器镜像 | 预装 JAX + MaxText 依赖的镜像 |

### 1.3 当前配置（示例）

```yaml
项目: cloud-tpu-multipod-dev
集群: chrisya-v7x-v134
Zone: us-central1
节点池: np-tpu7x-spot-32
机器类型: tpu7x-standard-4t (4 chips/node)
节点数: 8
总芯片: 32 chips
拓扑: 2x4x4
镜像: gcr.io/cloud-tpu-multipod-dev/chrisya-maxtext-runner
代码: gs://chrisya-v7x-us-central1/ant-pretrain-kkx
输出: gs://chrisya-v7x-us-central1/almodel-training-output
```

---

## 2. 集群准备

### 2.1 设置 kubectl context

```bash
# 生成 kubeconfig
gcloud container clusters get-credentials chrisya-v7x-v134 \
  --zone us-central1 \
  --project cloud-tpu-multipod-dev

# 确认 context
export CONTEXT="gke_cloud-tpu-multipod-dev_us-central1_chrisya-v7x-v134"
```

### 2.2 Pre-flight 检查

一次性并行检查节点状态和 Kueue：

```bash
# 检查 TPU 节点数量和拓扑
CLOUDSDK_CONTEXT_AWARE_USE_CLIENT_CERTIFICATE=false kubectl --context=$CONTEXT \
  get nodes -l cloud.google.com/gke-tpu-accelerator=tpu7x \
  -o jsonpath='{range .items[*]}{.metadata.name}{"\t"}{.metadata.labels.cloud\.google\.com/gke-tpu-topology}{"\n"}{end}'
```

期望输出：8 个节点，拓扑均为 `2x4x4`。

```bash
# 检查 Kueue 是否就绪
CLOUDSDK_CONTEXT_AWARE_USE_CLIENT_CERTIFICATE=false kubectl --context=$CONTEXT \
  get resourceflavors 2>/dev/null
```

如果返回 `No resources found`，需要先安装 Kueue（见第 3 节）。

---

## 3. 安装 Kueue (首次)

> 只需执行一次。如果 `kubectl get resourceflavors` 已有结果，跳过此步。

### 3.1 用 xpk cluster adapt

```bash
xpk cluster adapt \
  --cluster chrisya-v7x-v134 \
  --project cloud-tpu-multipod-dev \
  --zone us-central1 \
  --tpu-type=tpu7x-64
```

> **⚠️ 关键踩坑：tpu-type 命名**
>
> xpk 的 `--tpu-type` 用的是 **device count**，不是 chip count！
> - 32 chips (拓扑 2×4×4) → `--tpu-type=tpu7x-64` (32 chips × 2 TCs = 64 devices)
> - 16 chips (拓扑 2×2×4) → `--tpu-type=tpu7x-32`
>
> 如果用错了（比如 `tpu7x-32`），Kueue 会创建拓扑 `2x2x4` 的 ResourceFlavor，
> 与实际节点 `2x4x4` 不匹配，Job 会一直 Pending。

### 3.2 验证安装

```bash
# 检查 ResourceFlavor 的拓扑标签
kubectl get resourceflavor -o jsonpath='{range .items[*]}{.metadata.name}: {.spec.nodeLabels}{"\n"}{end}'
```

确认输出中 `gke-tpu-topology` 的值与节点实际拓扑一致（`2x4x4`）。

---

## 4. 生成 JobSet YAML

> **为什么不用 `xpk workload create`？**
> 因为 xpk 需要 Docker daemon 来构建镜像。gLinux 上 Docker 通常不运行，
> 会直接报错。我们直接写 JobSet YAML 用 `kubectl apply`。

### 4.1 参数说明

| 参数 | 计算方式 | 32 chips 示例值 |
|------|---------|---------------|
| `RUN_NAME` | 自定义 | `bench-test-001` |
| `NUM_HOSTS` | total_chips / chips_per_host | 32 / 4 = `8` |
| `CHIPS_PER_HOST` | 节点的 `google.com/tpu` capacity | `4` |
| `TOPOLOGY` | 节点实际拓扑 | `2x4x4` |
| `ici_fsdp_parallelism` | total_chips × 2 (TCs per chip) | 32 × 2 = `64` |

### 4.2 完整 YAML 模板

将以下内容保存为 `/tmp/<RUN_NAME>.yaml`：

```yaml
apiVersion: jobset.x-k8s.io/v1alpha2
kind: JobSet
metadata:
  name: <RUN_NAME>                    # ← 替换：任务名称
  labels:
    kueue.x-k8s.io/queue-name: multislice-queue
spec:
  failurePolicy:
    maxRestarts: 0
  replicatedJobs:
  - name: worker
    replicas: 1
    template:
      spec:
        parallelism: 8               # ← 替换：NUM_HOSTS
        completions: 8               # ← 替换：NUM_HOSTS
        completionMode: Indexed
        backoffLimit: 0
        template:
          metadata:
            labels:
              app: <RUN_NAME>        # ← 替换
          spec:
            restartPolicy: Never
            nodeSelector:
              cloud.google.com/gke-tpu-accelerator: tpu7x
              cloud.google.com/gke-tpu-topology: 2x4x4    # ← 替换：TOPOLOGY
            containers:
            - name: jax-tpu
              image: gcr.io/cloud-tpu-multipod-dev/chrisya-maxtext-runner  # ← 替换：镜像
              ports:
              - containerPort: 8471
              - containerPort: 8080
              securityContext:
                privileged: true
              resources:
                limits:
                  google.com/tpu: "4"        # ← 替换：CHIPS_PER_HOST
              env:
              - name: LIBTPU_INIT_ARGS
                value: >-
                  --xla_tpu_enable_async_collective_fusion=true
                  --xla_tpu_enable_async_collective_fusion_fuse_all_gather=true
                  --xla_tpu_enable_async_collective_fusion_multiple_steps=true
                  --xla_tpu_overlap_compute_collective_tc=true
                  --xla_enable_async_all_gather=true
                  --xla_enable_async_collective_permute=true
                  --xla_tpu_enable_all_experimental_scheduler_features=true
                  --xla_tpu_scoped_vmem_limit_kib=65536
                  --xla_tpu_dvfs_p_state=7
                  --xla_tpu_use_enhanced_launch_barrier=true
                  --xla_tpu_spmd_rng_bit_generator_unsafe=true
              command:
              - bash
              - -c
              - |
                set -e
                echo "[$(hostname)] Downloading code from GCS..."
                mkdir -p /tmp/ant-pretrain-kkx
                gsutil -m -q rsync -r gs://chrisya-v7x-us-central1/ant-pretrain-kkx /tmp/ant-pretrain-kkx
                cd /tmp/ant-pretrain-kkx
                export PYTHONPATH=/tmp/ant-pretrain-kkx:/tmp/ant-pretrain-kkx/src:$PYTHONPATH
                echo "[$(hostname)] Starting training..."
                rm -f /tmp/libtpu_lockfile
                python3 -m MaxText.train src/MaxText/configs/base.yml \
                    model_name=al_model \
                    override_model_config=true \
                    run_name=<RUN_NAME> \
                    base_output_directory=gs://chrisya-v7x-us-central1/almodel-training-output \
                    dataset_type=synthetic \
                    vocab_size=157184 \
                    steps=20 \
                    per_device_batch_size=12 \
                    gradient_accumulation_steps=2 \
                    max_target_length=4096 \
                    opt_type=adamw \
                    learning_rate=3.36e-4 \
                    adam_weight_decay=0.1 \
                    gradient_clipping_threshold=1.0 \
                    warmup_steps_fraction=0.02 \
                    cosine_learning_rate_final_fraction=0.1 \
                    learning_rate_schedule_steps=-1 \
                    ici_fsdp_parallelism=64 \
                    ici_tensor_parallelism=1 \
                    ici_context_parallelism=1 \
                    dcn_fsdp_parallelism=1 \
                    dcn_tensor_parallelism=1 \
                    remat_policy=full \
                    megablox=true \
                    sparse_matmul=true \
                    enable_checkpointing=false \
                    gcs_metrics=false \
                    save_config_to_gcs=false \
                    packing=false \
                    log_period=1
                echo "[$(hostname)] Training finished with exit code $?"
```

> **注意**：代码下载用 `gsutil rsync -r`，不要用 `gsutil cp -r`！
> `cp -r` 在容器中会因为目标目录不存在而报 "Destination URL must name a directory" 错误。

---

## 5. 提交 Benchmark

```bash
# 设置变量
export RUN_NAME="bench-test-001"
export CONTEXT="gke_cloud-tpu-multipod-dev_us-central1_chrisya-v7x-v134"

# 提交 JobSet
CLOUDSDK_CONTEXT_AWARE_USE_CLIENT_CERTIFICATE=false \
kubectl --context=$CONTEXT apply -f /tmp/${RUN_NAME}.yaml
```

检查 JobSet 状态：

```bash
kubectl --context=$CONTEXT get jobset ${RUN_NAME}
```

---

## 6. 监控和收集结果

### 6.1 等待 Pods Running

```bash
# 看 pods 状态（通常 30-45 秒变 Running）
kubectl --context=$CONTEXT get pods -l app=${RUN_NAME} -w
```

### 6.2 查看训练日志

```bash
# 找到 pod-0（主日志）
POD0=$(kubectl --context=$CONTEXT get pods -l app=${RUN_NAME} \
  --sort-by=.metadata.name -o name | head -1)

# 实时看日志
kubectl --context=$CONTEXT logs ${POD0} -f
```

### 6.3 提取性能数据

训练完成后，从日志中提取关键指标：

```bash
kubectl --context=$CONTEXT logs ${POD0} | grep -E "completed step:|total_weights|TFLOP"
```

输出格式：
```
completed step: 0, seconds: 46.218, TFLOP/s/device: 156.914, Tokens/s/device: 1087.103, ...
completed step: 1, seconds: 10.315, TFLOP/s/device: 703.074, Tokens/s/device: 4873.430, ...
completed step: 2, seconds: 10.106, TFLOP/s/device: 717.539, Tokens/s/device: 4973.568, ...
...
completed step: 19, seconds: 9.912, TFLOP/s/device: 731.536, Tokens/s/device: 5070.576, ...
```

### 6.4 TensorBoard

结果也会写入 GCS，可用 TensorBoard 查看：

```bash
tensorboard --logdir=gs://chrisya-v7x-us-central1/almodel-training-output/${RUN_NAME}/tensorboard/
```

---

## 7. 清理

```bash
kubectl --context=$CONTEXT delete jobset ${RUN_NAME}
```

---

## 8. Sync/Async 计时模式切换

MaxText 默认使用 **异步 dispatch**，即 `p_train_step()` 返回的是 future（TPU 还在计算），
step time 测量的是 Python 端的 dispatch 时间而非 TPU 实际计算时间。

这导致前几步的 step time 不准确（见第 9 节数据对比）。

### 8.1 两种模式

| 模式 | 描述 | 稳态 Step Time | 适用场景 |
|------|------|---------------|---------|
| **Async（默认）** | 不修改代码，保持异步 dispatch | ~9.87s, ~734 TFLOP/s | Benchmark 报告、性能对比 |
| **Sync（诊断）** | Patch train.py，每步 block_until_ready | ~9.92s, ~731 TFLOP/s | 诊断 warmup/编译行为 |

> **重要**：稳态 step time 两种模式几乎相同（差 0.5%）。区别只在前 3 步。
> 生产 benchmark 用 Async 模式即可。

### 8.2 开启 Sync 模式

下载 → 修改 → 上传 GCS 上的 `train.py`：

```bash
# 下载
gsutil cp gs://chrisya-v7x-us-central1/ant-pretrain-kkx/src/MaxText/train.py /tmp/train.py

# 在 p_train_step() 之后、step_time_delta 之前加一行：
# 原始代码：
#   state, metrics = p_train_step(state, example_batch, nextrng)
#   step_time_delta = datetime.datetime.now() - last_step_completion
#
# 改为：
#   state, metrics = p_train_step(state, example_batch, nextrng)
#   jax.block_until_ready(state)                                   # ← 加这行
#   step_time_delta = datetime.datetime.now() - last_step_completion

# 上传
gsutil cp /tmp/train.py gs://chrisya-v7x-us-central1/ant-pretrain-kkx/src/MaxText/train.py
```

### 8.3 关闭 Sync 模式（恢复 Async）

删除 `jax.block_until_ready(state)` 那一行，重新上传：

```bash
gsutil cp gs://chrisya-v7x-us-central1/ant-pretrain-kkx/src/MaxText/train.py /tmp/train.py
# 删除 jax.block_until_ready(state) 行
gsutil cp /tmp/train.py gs://chrisya-v7x-us-central1/ant-pretrain-kkx/src/MaxText/train.py
```

> **为什么 `profile_cleanly: True` 不够？**
> `profile_cleanly` 只在 profiler 的 `activate()` / `deactivate()` 时调用 `block_until_ready()`。
> 如果 `profiler: ""`（默认），它**完全无效**。要让每步计时准确，必须手动 patch train.py。

---

## 9. 实测 Benchmark 数据

### 9.1 环境

```
模型: ALModel 17B MoE
硬件: TPU v7 × 32 chips (拓扑 2×4×4)
Batch: per_device_batch_size=12, gradient_accumulation_steps=2
并行策略: ici_fsdp_parallelism=64, ici_tensor_parallelism=1
Remat: full
数据: synthetic
日期: 2026-03-05
```

### 9.2 Sync 模式（block_until_ready 开启）

```
Step  | Time (s) | TFLOP/s/device | Tokens/s/device
------|----------|----------------|----------------
  0   |  46.218  |     156.9      |    1,087.1
  1   |  10.315  |     703.1      |    4,873.4
  2   |  10.106  |     717.5      |    4,973.6
  3   |   9.999  |     725.2      |    5,027.1
  ...
  19  |   9.912  |     731.5      |    5,070.6
```

**稳态 (Step 3-19)**: ~9.92 s/step, ~731 TFLOP/s/device

### 9.3 Async 模式（默认，无 block_until_ready）

```
Step  | Time (s) | TFLOP/s/device | Tokens/s/device
------|----------|----------------|----------------
  0   |  35.437  |     204.7      |    1,418.5
  1   |   0.449  |  16,137.8      |   111,864.3   ← 虚假的快（仅 dispatch 时间）
  2   |  31.127  |     233.0      |    1,614.8    ← 还 Step 1 的"债"
  3   |   0.610  |  11,899.4      |    82,475.4
  ...
  19  |   9.873  |     734.4      |    5,090.3
```

**稳态 (Step 5+)**: ~9.87 s/step, ~734 TFLOP/s/device

### 9.4 对比分析

```
           Sync       Async      差异
稳态 step  9.92s      9.87s      0.5% (几乎相同)
稳态 TFLOP 731        734        0.4%
Step 0     46.2s      35.4s      Sync 记录了完整编译+执行
Step 1     10.3s      0.4s       Async 仅记录 dispatch（不可信）
Step 2     10.1s      31.1s      Async 还了 Step 1 的"债"
```

**关键结论**：
- 稳态性能几乎无差异，因为 async 模式下一步的 `p_train_step(state, ...)`
  也需要读取上一步的 `state`，相当于隐式 sync
- 区别仅在前 3 步的 timing shift
- **Benchmark 报告直接用 Async 稳态数据即可**

### 9.5 JAX Async Dispatch 原理

```
时间轴 (Async 模式):

Step 0: |------ XLA 编译 (同步) + 执行 ------|
        ↑ start                              ↑ p_train_step 返回
        记录 step_time = 35.4s

Step 1: |dispatch|  ← Python 立即返回 (0.4s)
        TPU 后台: |------------ 计算中 -----------|
        记录 step_time = 0.4s (不包含实际计算！)

Step 2: |------ 等 Step 1 完成 + dispatch ------|
        记录 step_time = 31.1s (包含 Step 1 的计算时间)

Step 3+: 稳态，每步时间 ≈ 上一步 TPU 计算时间
```

加了 `block_until_ready` 后，每步记录的都是自己的实际计算时间：

```
时间轴 (Sync 模式):

Step 0: |------ XLA 编译 + 执行 ------|-- block --|
        记录 step_time = 46.2s

Step 1: |-------- TPU 计算 --------|- block -|
        记录 step_time = 10.3s (真实计时)

Step 2: |-------- TPU 计算 --------|- block -|
        记录 step_time = 10.1s (真实计时)
```

---

## 10. 时间线预期

### 10.1 单次 Benchmark 全流程 (32 chips, ALModel 17B, 20 steps)

| 阶段 | 耗时 | 说明 |
|------|------|------|
| Pre-flight 检查 | ~10s | kubectl 检查节点 + Kueue |
| 生成 YAML + 提交 | ~10s | 写文件 + kubectl apply |
| Container 拉取 + 代码下载 | ~30s | 镜像已缓存时更快 |
| XLA 编译 | ~3-5 min | 首次编译，Step 0 包含在内 |
| 训练 20 steps | ~3-4 min | 稳态 ~10s/step |
| 日志收集 + 清理 | ~10s | grep + delete jobset |
| **总计** | **~8-10 min** | 熟练操作 |

### 10.2 首次运行（含 Kueue 安装）

| 阶段 | 额外耗时 |
|------|---------|
| Kueue 安装 (`xpk cluster adapt`) | ~2-3 min |
| 调试拓扑匹配问题（如有） | ~5-10 min |

---

## 11. 常见问题和踩坑

### 11.1 Job 一直 Pending

**症状**: `kubectl get pods` 显示 Pending，不进入 Running。

**排查**:
```bash
# 检查 Kueue admission
kubectl get workloads

# 检查 ResourceFlavor 拓扑是否匹配
kubectl get resourceflavor -o yaml | grep tpu-topology
kubectl get nodes -l cloud.google.com/gke-tpu-accelerator=tpu7x \
  -o jsonpath='{.items[0].metadata.labels.cloud\.google\.com/gke-tpu-topology}'
```

**常见原因**:
- ResourceFlavor 拓扑与节点不匹配（用了错误的 `--tpu-type`）
- Spot 节点被回收，节点数不够

### 11.2 gsutil cp -r 报错

**症状**: `Destination URL must name a directory, file, or pipe`

**原因**: 容器中 `gsutil cp -r` 目标目录不存在时会报错。

**解决**: 改用 `gsutil rsync -r`，并确保目标目录已创建：
```bash
mkdir -p /tmp/ant-pretrain-kkx
gsutil -m -q rsync -r gs://bucket/path /tmp/ant-pretrain-kkx
```

### 11.3 xpk workload create 报 Docker 错误

**症状**: `Cannot connect to the Docker daemon`

**原因**: gLinux 上 Docker daemon 通常不运行。

**解决**: 不用 xpk 提交，直接写 JobSet YAML + `kubectl apply`（本文档的方式）。

### 11.4 Step Time 数据看起来不正常

**症状**: Step 1 只有 0.4s，Step 2 突然 31s。

**原因**: JAX async dispatch 导致的 timing shift（见 [9.5 节](#95-jax-async-dispatch-原理)）。

**解决**: 这是正常现象。只看稳态数据（Step 5+）即可。如果需要每步准确计时，
开启 Sync 模式（见 [第 8 节](#8-syncasync-计时模式切换)）。

### 11.5 profile_cleanly 没有效果

**症状**: 设了 `profile_cleanly: True` 但 step time 仍然有 timing shift。

**原因**: `profile_cleanly` 只在 profiler activate/deactivate 时 block。
如果 `profiler: ""`（默认），完全无效。

**解决**: 必须手动 patch `train.py` 加 `jax.block_until_ready(state)`。

### 11.6 xpk tpu-type 选错

**速查表** (TPU v7):

| 实际芯片数 | 拓扑 | xpk tpu-type | 说明 |
|-----------|------|-------------|------|
| 16 chips | 2×2×4 | `tpu7x-32` | 16 × 2 TCs = 32 devices |
| 32 chips | 2×4×4 | `tpu7x-64` | 32 × 2 TCs = 64 devices |
| 64 chips | 4×4×4 | `tpu7x-128` | 64 × 2 TCs = 128 devices |

规则: **device count = chip count × 2**（每个 chip 有 2 个 TensorCore）。

---

## 12. TFLOP/s 计算 Bug 分析（2026-03-06 发现）

> **结论**：MaxText 对 ALModel 报告的 TFLOP/s 数值**虚高**，原因是 FFN flops 被 `num_decoder_layers` 重复乘了一次。

### 12.1 Bug 位置

文件：`src/MaxText/maxtext_utils.py`，函数 `calculate_tflops_training_per_device()`

### 12.2 问题分析

ALModel 的 FFN flops 通过 `calculate_routed_and_shared_ffn_tflops_per_device()` 计算，
该函数**内部已经乘了 layer 数**：

```python
# calculate_routed_and_shared_ffn_tflops_per_device() 内部
num_dense_layers, num_moe_layers = get_dense_moe_layers(config)
dense_ffn_flops = ... * num_dense_layers     # 已乘 1 层 (dense)
moe_ffn_flops = ... * num_moe_layers         # 已乘 19 层 (MoE)
total_ffn_flops = dense_ffn_flops + moe_ffn_flops  # 返回整个模型的 FFN flops
```

但在下游 combine 分支中，`AL_MODEL` 没有被加到 DEEPSEEK/LING2 的正确分支，
落入了 `else` 分支，**再次乘了 `num_decoder_layers`**：

```python
# ❌ ALModel 错误地走了 else 分支
learnable_weight_tflops = (
    (total_ffn_flops + qkv_flops + projection_flops) * num_decoder_layers  # FFN 被多乘一次！
    + embedding_flops
) * 3 / 10**12

# ✅ DEEPSEEK/LING2 的正确做法
learnable_weight_tflops = (
    total_ffn_flops + (qkv_flops + projection_flops) * num_decoder_layers  # 只有 QKV 乘 layer 数
    + embedding_flops
) * 3 / 10**12
```

### 12.3 影响范围

- **虚高的组件**：`learnable_weight_tflops`（FFN 部分被乘了 20 倍）
- **正确的组件**：`attention_tflops`（不受影响）、step time（不受影响）
- **报告影响**：`TFLOP/s/device` = `total_tflops / step_time`，分子虚高导致报告的 TFLOP/s 虚高

### 12.4 修复

```python
# 第 594 行，加上 DecoderBlockType.AL_MODEL
elif config.decoder_block in (DecoderBlockType.DEEPSEEK, DecoderBlockType.LING2, DecoderBlockType.AL_MODEL):
```

### 12.5 修复前后对比

> **注意**：step time 完全不受影响（它是墙钟时间），只有 TFLOP/s 的计算值变了。

| 指标 | 修复前（虚高） | 修复后（正确） | 变化 |
|------|------------|------------|------|
| 稳态 TFLOP/s/device | ~731 | **~76.9** | 下降 9.5× |
| 稳态 Step Time | ~9.87s | ~9.87s | 不变 |
| Tokens/s/device | ~9,958 | ~9,958 | 不变 |

> 修复后 Total TFLOPs per step per device 从 ~7,250 降到 ~759，
> 因为 FFN flops 不再被 num_decoder_layers (20) 重复乘。
> Step time 和吞吐量完全不变，只有 FLOP 计算值变了。

### 12.6 经验教训

- 新增 `DecoderBlockType` 时，必须检查 `calculate_tflops_training_per_device()` 中所有分支
- `calculate_routed_and_shared_ffn_tflops_per_device()` 返回的是**整个模型**的 FFN flops（已含 layer 数），
  而 `qkv_flops` / `projection_flops` 是**单层**的。两者不能用相同的乘法逻辑
- TFLOP/s 是计算指标而非测量指标，它的准确性完全依赖公式正确性。
  永远要 code review FLOP 计算逻辑，不要只看最终数字

---

## 13. 附录：完整配置参考

### 13.1 ALModel 17B MoE 训练参数

```yaml
# 模型配置
model_name: al_model
override_model_config: true

# 数据
dataset_type: synthetic
vocab_size: 157184

# Batch
per_device_batch_size: 12
gradient_accumulation_steps: 2
max_target_length: 4096

# 优化器
opt_type: adamw
learning_rate: 3.36e-4
adam_weight_decay: 0.1
gradient_clipping_threshold: 1.0
warmup_steps_fraction: 0.02
cosine_learning_rate_final_fraction: 0.1
learning_rate_schedule_steps: -1

# 并行策略
ici_fsdp_parallelism: 64         # = total_chips × 2
ici_tensor_parallelism: 1
ici_context_parallelism: 1
dcn_fsdp_parallelism: 1
dcn_tensor_parallelism: 1

# 性能优化
remat_policy: full
megablox: true
sparse_matmul: true

# Benchmark 设置
enable_checkpointing: false
gcs_metrics: false
save_config_to_gcs: false
packing: false
log_period: 1
steps: 20
```

### 13.2 推荐 XLA Flags (TPU v7)

```bash
LIBTPU_INIT_ARGS="
--xla_tpu_enable_async_collective_fusion=true
--xla_tpu_enable_async_collective_fusion_fuse_all_gather=true
--xla_tpu_enable_async_collective_fusion_multiple_steps=true
--xla_tpu_overlap_compute_collective_tc=true
--xla_enable_async_all_gather=true
--xla_enable_async_collective_permute=true
--xla_tpu_enable_all_experimental_scheduler_features=true
--xla_tpu_scoped_vmem_limit_kib=65536
--xla_tpu_dvfs_p_state=7
--xla_tpu_use_enhanced_launch_barrier=true
--xla_tpu_spmd_rng_bit_generator_unsafe=true
"
```

### 13.3 快速开始 Checklist

- [ ] `gcloud auth list` 确认认证
- [ ] `kubectl get nodes -l cloud.google.com/gke-tpu-accelerator=tpu7x` 确认节点在线
- [ ] `kubectl get resourceflavors` 确认 Kueue 已安装
- [ ] 检查 ResourceFlavor 拓扑匹配
- [ ] 复制 YAML 模板，替换 `<RUN_NAME>`
- [ ] `kubectl apply -f /tmp/<RUN_NAME>.yaml`
- [ ] `kubectl get pods -l app=<RUN_NAME> -w` 等待 Running
- [ ] `kubectl logs <pod-0> -f` 看日志
- [ ] 记录稳态 step time 和 TFLOP/s
- [ ] `kubectl delete jobset <RUN_NAME>` 清理
