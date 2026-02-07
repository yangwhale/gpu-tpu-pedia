---
name: tpu-trainer
description: TPU v7 (Ironwood) 模型训练自动化。当用户说"帮我训练"、"跑一下训练"、"测试训练"等并指定 tpu-recipes 下的模型路径时触发。自动生成脚本、提交训练、收集结果并写文档。
license: MIT
---

# TPU v7 Training Automation

在 TPU v7 (Ironwood) 上自动执行 MaxText 模型训练，包括环境准备、任务提交、结果收集和文档生成。

## 触发条件

当用户的请求包含以下模式时触发：
- "帮我训练 tpu-recipes/training/ironwood/..."
- "跑一下 ...的训练"
- "测试一下 ...模型"
- 指定了 `tpu-recipes/training/ironwood/<model>/<config>/xpk` 路径

## 配置文件

隐私信息存储在 `~/.claude/skills/tpu-trainer/config.yaml`，包含：
- `project_id`: GCP 项目 ID
- `cluster_name`: GKE 集群名称
- `reservations`: TPU reservation 列表（按优先级排序）
- `default_zone`: 默认 zone（优先 us-central1-ai1a）
- `base_output_dir`: GCS 输出路径
- `workload_image`: Docker 镜像地址
- `tpu_recipes_path`: tpu-recipes 仓库本地路径

**首先读取配置文件获取这些值，不要硬编码。**

## 完整工作流程

### 第一步：解析用户请求

从用户指定的路径中解析出：
- **模型名称**: 如 `deepseek3-671b`, `qwen3-235b-a22b`
- **配置**: 如 `4k-bf16-tpu7x-4x4x8`, `4k-fp8-tpu7x-4x8x8`
- **TPU 拓扑**: 从配置名中提取，如 `tpu7x-4x4x8` (128 chips) 或 `tpu7x-4x8x8` (256 chips)

路径模式：`tpu-recipes/training/ironwood/<model>/<config>/xpk`

### 第二步：读取官方 Recipe

读取指定路径下的 `run_recipe.sh`，从中提取：
- XLA_FLAGS（完整复制）
- MAXTEXT_ARGS（完整复制）
- device-type（如 `tpu7x-4x4x8`）
- 其他 xpk workload create 参数

### 第三步：生成训练脚本

在指定路径下生成两个脚本：

#### 3a. `setup_training_env.sh` — 环境设置 + Docker 镜像构建

```bash
#!/bin/bash
# 从 config.yaml 读取的值填入
set -e

export PROJECT_ID="<from config>"
export CLUSTER_NAME="<from config>"
export ZONE="<from config>"
export WORKLOAD_IMAGE="<from config>"

# 构建 Docker 镜像（如果镜像不存在）
# 使用 maxtext_branch, jax_version, libtpu_version from config
```

注意：如果 Docker 镜像已存在（`docker manifest inspect` 能找到），跳过构建步骤。

#### 3b. `submit_<model>.sh` — 提交训练任务

```bash
#!/bin/bash
set -e

export PROJECT_ID="<from config>"
export CLUSTER_NAME="<from config>"
export ZONE="<from config>"
export BASE_OUTPUT_DIR="<from config>"
export WORKLOAD_IMAGE="<from config>"
export WORKLOAD_NAME="$(printf \"%.26s\" \"${USER//_/-}-<model-short-name>\")-$(date +%Y%m%d-%H%M)"

# XLA_FLAGS 和 MAXTEXT_ARGS 从 run_recipe.sh 完整复制
# 额外添加 profiler 配置：
# profiler=xplane profiler_steps=3 skip_first_n_steps_for_profiler=5

xpk workload create \
  --cluster=$CLUSTER_NAME \
  --project=$PROJECT_ID \
  --zone=$ZONE \
  --priority=very-high \
  --max-restarts=0 \
  --device-type=<from recipe> \
  --num-slices=1 \
  --docker-image="${WORKLOAD_IMAGE}" \
  --enable-debug-logs \
  --workload="${WORKLOAD_NAME}" \
  --command="set -e && export ENABLE_PATHWAYS_PERSISTENCE='1' && \
export LIBTPU_INIT_ARGS='${XLA_FLAGS}' && \
export JAX_PLATFORMS='tpu,cpu' && export ENABLE_PJRT_COMPATIBILITY='true' && \
python3 -m MaxText.train MaxText/configs/base.yml ${MAXTEXT_ARGS}"
```

### 第四步：确保集群就绪

1. 检查集群是否存在：`gcloud container clusters list --filter=name=<cluster>`
2. 检查是否有匹配拓扑的 TPU node pool（状态为 RUNNING）
3. 如果没有 node pool，使用 `xpk cluster adapt` 创建：
   - **必须用 xpk**，不能用 `gcloud container node-pools create`（TPU v7 需要 workload policy，gcloud 只支持 placement policy 会报错 `INVALID_ARGUMENT`）
   - 按 config.yaml 中的 reservation 优先级，直接尝试创建（不必先查容量；容量不足会返回 RESOURCE_EXHAUSTED，再切换 reservation）
   - xpk adapt 会自动配置 Kueue（ResourceFlavor、ClusterQueue、xpk configmap）

#### Node Pool 创建命令（使用 xpk）

```bash
xpk cluster adapt \
  --cluster=<cluster> \
  --project=<project_id> \
  --zone=<zone> \
  --tpu-type=tpu7x-<topology like 4x4x8> \
  --num-slices=1 \
  --reservation=<reservation_name>
```

此命令需要几分钟完成，建议在后台运行。

#### 用 gcloud beta 手动创建（xpk memory_limit bug 时的替代方案）

如果 xpk adapt 因 `memory_limit` bug 失败，可以用 `gcloud beta` 手动创建：

```bash
gcloud beta container node-pools create <np-name> \
  --cluster=<cluster> \
  --project=<project_id> \
  --location=us-central1 \
  --node-locations=<zone> \
  --machine-type=tpu7x-ultranet-4t \
  --num-nodes=<num_hosts> \
  --placement-policy=tpu7x-<num_devices>-<topology>-placement-policy \
  --reservation-affinity=specific \
  --reservation=<reservation_name> \
  --enable-gvnic \
  --scopes=storage-full,gke-default,"https://www.googleapis.com/auth/cloud-platform" \
  --max-pods-per-node=15 \
  --node-version=<cluster_node_version>
```

**关键注意事项**：
1. **必须用 `gcloud beta`**，不是 `gcloud`
2. **不要加** `--placement-type=COMPACT` 和 `--tpu-topology`（TPU v7 用 workload policy）
3. **必须加** `--placement-policy=tpu7x-<devices>-<topology>-placement-policy`（xpk workload 的 pod 用这个 label 做 node selector）
4. **不要加** `--no-enable-autoupgrade`（RAPID channel 强制开启）
5. 确保 GCE resource policy 已存在：`gcloud compute resource-policies describe tpu7x-<devices>-<topology>-placement-policy --region=us-central1`
6. Kueue 配置需要手动检查（xpk adapt 通常已配置好 configmap，但可能没配完 Kueue）

#### 手动 Kueue 配置（仅当 xpk 未自动配置时需要）

如果 xpk adapt 没有正确配置 Kueue（检查方法见下），需要手动配置：

#### TPU 拓扑 → 资源映射

| 拓扑 | Chips | Devices | Hosts/Nodes | GKE Machine Type | Kueue Flavor | Kueue Quota |
|------|-------|---------|-------------|------------------|--------------|-------------|
| tpu7x-4x4x8 | 128 | 256 | 32 | tpu7x-ultranet-4t | 1xtpu7x-256 | 128 |
| tpu7x-4x8x8 | 256 | 512 | 64 | tpu7x-ultranet-4t | 1xtpu7x-512 | 256 |
| tpu7x-8x8x8 | 512 | 1024 | 128 | tpu7x-ultranet-4t | 1xtpu7x-1024 | 512 |
| tpu7x-8x8x16 | 1024 | 2048 | 256 | tpu7x-ultranet-4t | 1xtpu7x-2048 | 1024 |

#### Kueue 配置模板

ResourceFlavor:
```yaml
apiVersion: kueue.x-k8s.io/v1beta1
kind: ResourceFlavor
metadata:
  name: 1xtpu7x-<num_devices>
spec:
  nodeLabels:
    cloud.google.com/gke-tpu-accelerator: tpu7x
    cloud.google.com/gke-tpu-topology: <topology like 4x8x8>
```

ClusterQueue 需要在 resourceGroups 中添加新的 flavor：
```yaml
- name: 1xtpu7x-<num_devices>
  resources:
  - name: google.com/tpu
    nominalQuota: <num_chips>
```

xpk configmap patch:
```bash
kubectl patch configmap <cluster>-resources-configmap --type merge \
  -p '{"data":{"tpu7x-<num_devices>":"<num_hosts>"}}'
```

### 第五步：提交训练

1. 执行 submit 脚本
2. 等待 pods 全部进入 Running 状态
3. 监控 worker 0 的日志（`kubectl logs -f <pod-name>`）
4. 等待所有 steps 完成

### 第六步：收集结果

从训练日志中提取每个 step 的指标：
```
completed step: N, seconds: X, TFLOP/s/device: Y, Tokens/s/device: Z, total_weights: W, loss: L
```

计算 per-chip 指标：
- TFLOP/s/chip = TFLOP/s/device × 2（一个 chip = 2 个 TensorCore）
- Tokens/s/chip = Tokens/s/device × 2（与 ici_fsdp_transpose_parallelism 无关，始终 × 2）
- 验证公式：Tokens/s/chip = total_weights / step_time / num_chips

稳态性能取 Step 2+ 的平均值（排除 Step 0 JIT 编译、Step 1 warmup、profiler 步）。

### 第七步：生成 README 文档

在模型根目录（如 `tpu-recipes/training/ironwood/deepseek3-671b/`）创建或更新 `README.md`。

README 模板：

```markdown
# <Model Name> 训练测试记录

## 模型概况

| 项目 | 值 |
|------|-----|
| 模型 | <model name> |
| 总参数量 | <params> |
| 硬件 | TPU v7 (Ironwood) |
| 框架 | MaxText (<branch>) |
| JAX | <version> |
| Libtpu | <version> |
| XPK | <version> |

## 测试结果

### 我的测试记录

| 日期 | 配置 | Precision | Step Time (s) | TFLOPs/s/device | TFLOPs/s/chip | Tokens/s/chip | Loss (final) | 备注 |
|------|------|-----------|...

### 详细训练日志 - <topology> (<date>)

| Step | 耗时 (s) | TFLOP/s/device | TFLOP/s/chip | Tokens/s/chip | Loss |
|------|---------|...

- **稳态性能 (Step 2+)**: ~X s/step, ~Y TFLOP/s/chip, ~Z Tokens/s/chip
- **Loss 下降**: from → to (N%)
```

如果 README 已存在，追加新的测试记录行和详细日志 section，不要覆盖已有数据。

### 第八步：清理资源

训练完成后，**主动删除** node pool 释放预留资源（不需要询问用户确认）：

1. 删除 xpk workload：
   ```bash
   xpk workload delete --workload <name> \
     --cluster=<cluster> --project=<project_id> --zone=<zone>
   ```

2. 删除 TPU node pool（释放 reservation 资源）：
   ```bash
   gcloud container node-pools delete <np-name> \
     --cluster=<cluster> \
     --region=us-central1 \
     --project=<project_id> \
     --quiet
   ```
   注意：使用 `--region=us-central1`（区域级集群），不要用 `--zone`。

3. 验证清理完毕：
   ```bash
   gcloud container node-pools list --cluster=<cluster> --region=us-central1 --project=<project_id>
   ```
   应该只剩 `default-pool`。

## 重要注意事项

### Node Pool 管理

- 使用 `--region=us-central1`（区域级集群），不要用 `--zone`
- `gcloud container operations cancel` 只能取消 node upgrade，不能取消 CREATE_NODE_POOL
- 创建 node pool 可能因 reservation 容量不足而失败（RESOURCE_EXHAUSTED），需切换到其他 reservation

### XPK 已知问题

- xpk cluster create 发现已有 node pool 会弹交互确认（无法在脚本中自动化），建议直接用 gcloud
- xpk configmap 只是前端校验，真正的调度器是 Kueue
- xpk cluster adapt v0.16.1 有 `memory_limit` bug

### 指标换算

- 1 chip = 2 devices (TensorCores)
- TFLOP/s/chip = TFLOP/s/device × 2（所有拓扑一致）
- Tokens/s/chip = Tokens/s/device × 2（所有拓扑一致，与 ici_fsdp_transpose_parallelism 无关）
- Profiler 采集期间 step time 会增大 ~3x，属正常现象

### fp8 训练注意事项

- fp8 recipe 使用 `quantization=fp8_full` 和 `use_qwix_quantization=True`
- 包含大量 tile 参数（`wi_tile_*`, `wo_tile_*`），必须从 run_recipe.sh 完整复制
- fp8 还有 `weight_quantization_calibration_method` 和 `act_quantization_calibration_method` 参数
- fp8 可能使用不同的 XLA_FLAGS（比 bf16 多更多 sparse core 相关 flags）
- fp8 实测比 bf16 快 ~22-23%（DeepSeek3-671B: 4x4x8 22.39s vs 27.42s, 4x8x8 22.02s vs 27.12s）
- fp8 的 JIT 编译比 bf16 慢 ~25%（157s vs 125s），因为 quantization 内核更复杂
- fp8 训练的 Loss 略高于 bf16（预期范围内，大规模训练中差异会缩小）

### fp8 4x8x8 vs 4x4x8 参数差异

fp8 的 4x8x8 recipe 与 4x4x8 有以下关键差异：
- `ici_fsdp_transpose_parallelism`: 4x4x8 用 1，4x8x8 用 2
- `moe_fsdp_use_two_stage_all_gather`: 仅 4x8x8 有此参数（True）
- `use_max_logit_estimate`: 4x4x8 用 -1，4x8x8 用 22
- `attn_logits_soft_cap`: 仅 4x8x8 有此参数（15）
- XLA_FLAGS: 4x8x8 比 4x4x8 多出 `data_parallel_opt`、`ici_rs_pipelining`、`impure_use_lmr_on_gxc`、`dot_dot_fusion`、`rwb_fusion` 等 flags
- **不要混用**：不同拓扑的 recipe 参数差异较大，必须从对应的 run_recipe.sh 完整复制

### 扩展效率

- 128→256 chips（4x4x8→4x8x8）呈现近乎线性扩展
- per-chip 吞吐基本不变（~1-2% 提升），总吞吐翻倍
- bf16 和 fp8 均表现出相同的扩展特性
- 实测数据（DeepSeek3-671B fp8）：4x4x8 733.1 → 4x8x8 745.3 TFLOP/s/chip（+1.7%）

### FSDP 分片上限（DeepSeek3-671B 特有）

- DeepSeek3-671B 的某个 tensor 维度大小为 512（与 MoE 专家结构相关）
- `fsdp × fsdp_transpose` 不能超过 512，否则分片无法整除 tensor 维度
- 4x8x8 (512 devices): fsdp=256, fsdp_transpose=2, 乘积=512 → 刚好达到上限
- 超过 512 devices 时，直接增加 FSDP 会报 ValueError（dimension not divisible）
- **解决方案**：引入 `ici_data_parallelism=2` 把多余设备用于数据并行
- **代价**：per-chip 效率下降约 45%（ICI 数据并行引入 all-reduce 通信开销）
- **结论**：DeepSeek3-671B 的高效扩展上限是 256 chips (512 devices)
- 超过此限制需要考虑专家并行（expert parallelism）、张量并行或多 slice DCN 方案

### 超过 FSDP 上限时的报错特征

```
ValueError: global size of its dimension 0 should be divisible by 1024,
but it is equal to 512 (full shape: (512, 3, 128, 256))
```
看到此类报错时：
1. 检查 mesh 中 `fsdp × fsdp_transpose` 的乘积
2. 确认是否超过了模型 tensor 的最小维度
3. 考虑添加 `ici_data_parallelism=2`（牺牲效率换取可运行）
4. 或改用多 slice DCN 方案（每个 slice 保持在 FSDP 上限内）

### 训练前检查清单

提交训练前，按此清单逐项检查可避免大部分调度失败：
1. **placement-policy** 存在：`gcloud compute resource-policies describe tpu7x-<devices>-<topology>-placement-policy --region=us-central1`
2. **Kueue ResourceFlavor** 存在：`kubectl get resourceflavor 1xtpu7x-<devices>`
3. **Kueue ClusterQueue** 包含对应 flavor：`kubectl get clusterqueue -o yaml | grep 1xtpu7x-<devices>`
4. **xpk configmap** 包含条目：`kubectl get configmap <cluster>-resources-configmap -o yaml | grep tpu7x-<devices>`
5. **Node pool** 状态为 RUNNING：`gcloud container node-pools list --cluster=<cluster> --region=us-central1`
6. **Submit 脚本 ZONE** 与 node pool zone 一致

### Docker 镜像

- 镜像 `chrisya-maxtext-runner` 是通用的，支持所有 MaxText 模型和精度（bf16/fp8）
- 修改 MaxText 代码后需要重新构建镜像
- 构建需要 Python 3.12 环境（与运行环境的 3.11 不同）

## 示例

### 示例 1：训练指定配置

用户："帮我训练 tpu-recipes/training/ironwood/deepseek3-671b/4k-bf16-tpu7x-4x4x8/xpk 这个"

执行：
1. 读取 config.yaml 获取项目/集群信息
2. 读取 run_recipe.sh 获取训练参数
3. 生成 setup 和 submit 脚本
4. 确保 node pool 就绪（优先 ai1a reservation）
5. 提交训练，监控日志
6. 收集结果，更新 deepseek3-671b/README.md
7. 清理资源

### 示例 2：训练新模型

用户："帮我跑一下 qwen3-235b-a22b 的 fp8 训练"

执行：
1. 定位路径：`tpu-recipes/training/ironwood/qwen3-235b-a22b/4k-fp8-tpu7x-4x8x8/xpk`
2. 同上流程

## 故障排查

### Workload scheduling validation failed

xpk configmap 缺少对应的 TPU 类型条目。解决：
```bash
kubectl patch configmap <cluster>-resources-configmap --type merge \
  -p '{"data":{"tpu7x-<devices>":"<hosts>"}}'
```

### Kueue admission failed (flavor doesn't match)

缺少 ResourceFlavor 或 ClusterQueue 配置。参考第四步创建。

### RESOURCE_EXHAUSTED

Reservation 容量不足。查看容量（用 `gcloud beta compute reservations describe`）：
```bash
gcloud beta compute reservations describe <reservation_name> \
  --zone=<zone> --project=<project_id> \
  --format="value(aggregateReservation.hostCount,aggregateReservation.inUseHostCount)"
```
可用容量 = hostCount - inUseHostCount（单位是 hosts，每 host 4 chips）。
实践中直接尝试创建也可以，容量不足会返回 RESOURCE_EXHAUSTED 或 `No available resources`。
注意：即使 hostCount - inUseHostCount > 0，如果 reservation 处于 `DEGRADED` 状态（有维护进行中），也可能无法分配资源。
切换到其他 reservation 重试。

**重要**：切换 reservation/zone 后，submit 脚本的 `ZONE` 变量也必须同步更新（xpk workload create 的 --zone 参数需要与 node pool 所在 zone 一致）。

### Reservation 与 zone 不匹配

使用 `--reservation-affinity=specific --reservation=<name>` 时，reservation 必须存在于 `--node-locations` 指定的 zone 中。如果 reservation 在 `us-central1-ai1a` 但 `--node-locations=us-central1-c`，会报错 "Reservation is incorrect for the requested resources"。

查找 reservation 所在 zone：
```bash
gcloud beta compute reservations list --project=<project_id> --filter="name~<reservation_name>" --format="table(name,zone,status)"
```

### Node pool 创建失败 (machine type 错误)

TPU v7 的 GKE machine type 是 `tpu7x-ultranet-4t`，**不是** `ct7x-4x4x8` 或类似格式。
查询可用 machine type：
```bash
gcloud compute machine-types list --filter="name~tpu7x" --zones=<zone> --project=<project_id>
```

### Node pool 卡在 PROVISIONING

等待操作完成（可能需要 60+ 分钟），无法取消。`gcloud container operations cancel` 只能取消 node upgrade，**不能**取消 CREATE_NODE_POOL。完成后删除重试。

### Node pool 处于 ERROR 状态

之前创建失败的 node pool 可能残留为 ERROR 状态，导致再次创建时报 `409 Already Exists`。解决：
```bash
gcloud container node-pools delete <np-name> \
  --cluster=<cluster> --region=us-central1 --project=<project_id> --quiet
```
删除后重新创建即可。

### Multi-Slice DCN 训练前置条件

Multi-slice（跨多个 TPU slice 的分布式训练）需要集群级网络基础设施支持，**这些配置在集群创建后不可更改**：

1. **Multi-networking**：集群创建时需启用 `--enable-multi-networking`，提供双网卡（eth0 管理 + eth1 高速 DCN）
2. **Dataplane V2**：集群需使用 `ADVANCED_DATAPATH`（Cilium），而非 LEGACY（kube-proxy）
3. **sliceControllerConfig**：GKE addon，用于多 slice TPU 编排
4. **第二个 VPC 网络**：为 DCN 高速通信提供专用网络平面
5. **TCP rmem 调优**：DaemonSet 在所有节点上设置 `tcp_rmem="4096 41943040 314572800"`
6. **MegaScale gRPC 接口配置**：在 LIBTPU_INIT_ARGS 中添加 `--megascale_grpc_interface_prefixes=eth1,eth2,lo`

**通信机制**：Multi-slice DCN 通过 host-mediated gRPC（MegaScale 协议）通信，数据经由 host CPU 而非 chip-to-chip 直连。因此需要高速的 host 间网络（双网卡 + 专用 VPC）。

**不支持后期启用**：如果集群创建时未配置上述选项，需要**重建集群**才能支持 multi-slice 训练。

**xpk 提交 multi-slice 训练**的额外参数：
- `--num-slices=2`（或更多）
- `--device-type=tpu7x-4x8x8`（单个 slice 的拓扑）
- MaxText 参数：`dcn_data_parallelism=2`, `dcn_pipeline_parallelism=1`

**机器类型说明**：
- `tpu7x-standard-4t`：标准机器类型，支持 multi-slice
- `tpu7x-ultranet-4t`：带 iRDMA（Intel RDMA + Diorite SmartNIC），可加速 DCN 传输但非必要
- 正常运行的 multi-slice 集群（如 bodaborg）使用 `tpu7x-standard-4t`，说明 iRDMA 不是 multi-slice 的必要条件

### HBM 容量与 OOM

TPU v7 每 chip 有 192GB HBM，但每 chip 有 2 个 TensorCore（device）：
- **per-chip HBM**: 192 GB
- **per-device HBM**: ~96 GB（192/2）
- **可用 per-device HBM**: ~94.75 GB（扣除系统保留 ~1.25GB）

OOM 报错中显示的是 per-device 限制（94.75GB），不是 per-chip 的 192GB。如果看到 OOM 且使用量在 95-105GB 范围内，需要减小 `per_device_batch_size` 或启用更多 offload（如 `decoder_layer_input=offload`）。

### 首次训练新拓扑的顺利路径

如果集群之前已经用 xpk adapt 配置过相同拓扑（比如 bf16 4x8x8），那么 Kueue、configmap、placement-policy 都已就绪。再次训练相同拓扑（比如 fp8 4x8x8）时：
1. 只需创建 node pool（Kueue 等配置复用）
2. 用"训练前检查清单"快速验证
3. 通常一次成功，无需额外配置
