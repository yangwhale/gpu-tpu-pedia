# TPU v7x VM 创建与存储配置

> 端到端指南：创建 TPU v7x-8 VM 实例、配置 Hyperdisk ML 高吞吐数据盘、创建 GCS 对象存储桶、
> 并通过 fio 基准测试验证磁盘性能。适用于推理和训练场景的存储准备。

## 目录

- [硬件与环境概览](#硬件与环境概览)
- [Step 1: 创建 Hyperdisk ML 数据盘](#step-1-创建-hyperdisk-ml-数据盘)
- [Step 2: 创建 TPU v7x VM 实例](#step-2-创建-tpu-v7x-vm-实例)
- [Step 3: SSH 连接到 VM](#step-3-ssh-连接到-vm)
- [Step 4: 格式化并挂载数据盘](#step-4-格式化并挂载数据盘)
- [Step 5: 安装 fio 并运行基准测试](#step-5-安装-fio-并运行基准测试)
- [Step 6: 创建 GCS 存储桶](#step-6-创建-gcs-存储桶)
- [基准测试结果（实测）](#基准测试结果实测)
- [Hyperdisk ML 读写模式说明](#hyperdisk-ml-读写模式说明)
- [附录: 磁盘类型选型参考](#附录-磁盘类型选型参考)

---

## 硬件与环境概览

| 项目 | 值 |
|------|-----|
| **机器类型** | `tpu7x-standard-4t`（4 chips = 8 devices） |
| **TPU 型号** | v7x (Ironwood) |
| **HBM** | 192 GB/chip, 总计 768 GB |
| **主机内存** | 944 GB |
| **启动盘** | 500 GB Hyperdisk Balanced |
| **数据盘** | 2 TB Hyperdisk ML（provisioned 2500 MiB/s） |

### 环境变量

后续命令中会用到以下变量，请根据实际情况修改：

```bash
export PROJECT_ID=<your-project-id>
export ZONE=us-central1-c                  # 根据 reservation 所在 zone 修改
export INSTANCE_NAME=<your-instance-name>   # 例如 my-tpu7x-01
export DATA_DISK_NAME=<your-data-disk-name> # 例如 my-tpu7x-data-01
export RESERVATION_NAME=<your-reservation>  # TPU v7x reservation 名称
export VPC_NAME=<your-vpc-name>
export SUBNET_NAME=<your-subnet-name>
export BUCKET_NAME=<your-bucket-name>       # 例如 my-tpu-data
export BUCKET_LOCATION=us-central1           # GCS 桶的区域（与 VM 同区域）
```

---

## Step 1: 创建 Hyperdisk ML 数据盘

Hyperdisk ML 是 GCP 提供的高吞吐磁盘类型，专为 AI/ML 工作负载设计。
相比 Hyperdisk Balanced，它提供更高的吞吐带宽和多机只读共享能力。

```bash
gcloud compute disks create ${DATA_DISK_NAME} \
    --project=${PROJECT_ID} \
    --zone=${ZONE} \
    --type=hyperdisk-ml \
    --size=2TB \
    --provisioned-throughput=2500
```

**参数说明**：

| 参数 | 值 | 说明 |
|------|-----|------|
| `--type` | `hyperdisk-ml` | Hyperdisk ML 磁盘类型 |
| `--size` | `2TB` | 磁盘容量 |
| `--provisioned-throughput` | `2500` | 预配置吞吐量（MiB/s）。IOPS 自动按 16:1 派生（2500 × 16 = 40K IOPS） |

> **注意**：Hyperdisk ML 的 **单机吞吐量上限为 2,400 MiB/s**（[官方文档](https://docs.cloud.google.com/compute/docs/disks/hd-types/hyperdisk-ml#perf-limits)），
> 实测 TPU v7x 上精确命中此上限（2,415 MiB/s）。provisioned-throughput 设为 2500 即可打满单机性能，
> 设更高的值不会提升性能，只会增加费用。
>
> 如果需要将磁盘共享给多台 VM（只读模式），则需按 VM 数量 × 2,400 来设置 provisioned-throughput。
>
> **修改限制**：吞吐量每 **6 小时**只能修改一次，磁盘大小每 **4 小时**只能修改一次。

验证磁盘创建：

```bash
gcloud compute disks describe ${DATA_DISK_NAME} \
    --project=${PROJECT_ID} \
    --zone=${ZONE} \
    --format="table(name,type.basename(),sizeGb,provisionedThroughput,accessMode)"
```

预期输出：

```
NAME              TYPE          SIZE_GB  PROVISIONED_THROUGHPUT  ACCESS_MODE
my-tpu7x-data-01  hyperdisk-ml  2048     2500                   READ_WRITE_SINGLE
```

---

## Step 2: 创建 TPU v7x VM 实例

```bash
gcloud compute instances create ${INSTANCE_NAME} \
    --project=${PROJECT_ID} \
    --zone=${ZONE} \
    --machine-type=tpu7x-standard-4t \
    --network-interface=network=${VPC_NAME},subnet=${SUBNET_NAME} \
    --maintenance-policy=TERMINATE \
    --provisioning-model=RESERVATION_BOUND \
    --reservation-affinity=specific \
    --reservation=${RESERVATION_NAME} \
    --create-disk=auto-delete=yes,boot=yes,size=500GB,type=hyperdisk-balanced,image=projects/ubuntu-os-accelerator-images/global/images/ubuntu-accel-2404-amd64-tpu-tpu7x-v20260422 \
    --disk=name=${DATA_DISK_NAME},device-name=data-disk,mode=rw,auto-delete=no \
    --no-shielded-secure-boot \
    --scopes=cloud-platform
```

**关键参数说明**：

| 参数 | 说明 |
|------|------|
| `--machine-type=tpu7x-standard-4t` | TPU v7x 单节点（4 chips, 8 devices） |
| `--maintenance-policy=TERMINATE` | TPU VM 必须设置为 TERMINATE（不支持 live migration） |
| `--provisioning-model=RESERVATION_BOUND` | 绑定到指定的容量预留 |
| `--reservation-affinity=specific` | 使用特定 reservation |
| `--create-disk=...image=...` | 启动盘使用 TPU v7x 专用镜像 |
| `--disk=...mode=rw` | 数据盘以读写模式挂载 |
| `--no-shielded-secure-boot` | TPU 镜像不支持 Secure Boot |

> **镜像选择**：使用 `ubuntu-os-accelerator-images` 项目中的 TPU v7x 专用镜像。
> 镜像版本号（如 `v20260422`）会定期更新，可通过以下命令查看同项目其他 VM 使用的镜像版本：
> ```bash
> gcloud compute disks describe <existing-tpu-vm-name> \
>     --project=${PROJECT_ID} --zone=${ZONE} \
>     --format="value(sourceImage)"
> ```

验证 VM 状态：

```bash
gcloud compute instances describe ${INSTANCE_NAME} \
    --project=${PROJECT_ID} \
    --zone=${ZONE} \
    --format="table(name,status,machineType.basename(),networkInterfaces[0].accessConfigs[0].natIP)"
```

等待状态从 `STAGING` 变为 `RUNNING`（约 30-60 秒）。

---

## Step 3: SSH 连接到 VM

获取 VM 的外部 IP：

```bash
VM_IP=$(gcloud compute instances describe ${INSTANCE_NAME} \
    --project=${PROJECT_ID} \
    --zone=${ZONE} \
    --format="value(networkInterfaces[0].accessConfigs[0].natIP)")
echo "VM IP: ${VM_IP}"
```

SSH 连接：

```bash
ssh -o StrictHostKeyChecking=no -i ~/.ssh/google_compute_engine ${USER}@${VM_IP}
```

> **提示**：如果使用 Google Corp 网络，可能需要加 `-o ProxyCommand=none` 绕过代理。
> 如果连接超时，等待 30 秒后重试（VM 刚启动时 SSH 服务可能尚未就绪）。

---

## Step 4: 格式化并挂载数据盘

以下命令在 **VM 内部** 执行。

### 4.1 确认数据盘设备

```bash
lsblk
```

预期输出（数据盘为 `nvme1n1`，2TB）：

```
NAME         MAJ:MIN RM   SIZE RO TYPE MOUNTPOINTS
nvme1n1      259:0    0     2T  0 disk
nvme0n1      259:1    0   500G  0 disk
├─nvme0n1p1  259:2    0   499G  0 part /
├─nvme0n1p14 259:3    0     4M  0 part
├─nvme0n1p15 259:4    0   106M  0 part /boot/efi
└─nvme0n1p16 259:5    0   913M  0 part /boot
```

也可通过 device-name 确认映射关系：

```bash
ls -la /dev/disk/by-id/ | grep google-data-disk
```

### 4.2 格式化为 ext4

```bash
sudo mkfs.ext4 -m 0 -E lazy_itable_init=0,lazy_journal_init=0,discard /dev/nvme1n1
```

参数说明：
- `-m 0`：不预留空间给 root（数据盘不需要）
- `lazy_itable_init=0,lazy_journal_init=0`：格式化时完成所有初始化，避免后台 I/O 影响性能测试
- `discard`：启用 TRIM 支持

### 4.3 挂载到 /mnt/data

```bash
sudo mkdir -p /mnt/data
sudo mount -o discard,defaults /dev/nvme1n1 /mnt/data
sudo chmod a+w /mnt/data
```

### 4.4 配置开机自动挂载

```bash
echo "/dev/disk/by-id/google-data-disk /mnt/data ext4 discard,defaults,nofail 0 2" | sudo tee -a /etc/fstab
```

验证挂载：

```bash
df -h /mnt/data
```

预期输出：

```
Filesystem      Size  Used Avail Use% Mounted on
/dev/nvme1n1    2.0T   28K  2.0T   1% /mnt/data
```

---

## Step 5: 安装 fio 并运行基准测试

### 5.1 安装 fio

```bash
sudo apt-get update -qq && sudo apt-get install -y -qq fio
fio --version  # 预期: fio-3.36 或更高
```

### 5.2 顺序读测试（验证吞吐）

```bash
sudo fio --name=seq-read \
    --filename=/dev/nvme1n1 \
    --rw=read \
    --bs=1M \
    --direct=1 \
    --ioengine=libaio \
    --iodepth=64 \
    --numjobs=4 \
    --runtime=30 \
    --time_based \
    --group_reporting
```

### 5.3 顺序写测试（验证写入吞吐）

写入测试需要使用文件路径（fio 不允许对已挂载的块设备直接写入）：

```bash
sudo fio --name=seq-write \
    --filename=/mnt/data/fio-test \
    --rw=write \
    --bs=1M \
    --direct=1 \
    --ioengine=libaio \
    --iodepth=64 \
    --numjobs=4 \
    --size=10G \
    --runtime=30 \
    --time_based \
    --group_reporting
```

### 5.4 随机读 4K 测试（验证 IOPS）

```bash
sudo fio --name=rand-read \
    --filename=/dev/nvme1n1 \
    --rw=randread \
    --bs=4k \
    --direct=1 \
    --ioengine=libaio \
    --iodepth=256 \
    --numjobs=4 \
    --runtime=30 \
    --time_based \
    --group_reporting
```

### 5.5 随机写 4K 测试（验证写入 IOPS）

```bash
sudo fio --name=rand-write \
    --filename=/mnt/data/fio-test-rw \
    --rw=randwrite \
    --bs=4k \
    --direct=1 \
    --ioengine=libaio \
    --iodepth=256 \
    --numjobs=4 \
    --size=10G \
    --runtime=30 \
    --time_based \
    --group_reporting
```

### 5.6 清理测试文件

```bash
rm -f /mnt/data/fio-test /mnt/data/fio-test-rw
```

### fio 关键参数说明

| 参数 | 说明 |
|------|------|
| `--direct=1` | 绕过 OS 页缓存，测试裸盘性能 |
| `--ioengine=libaio` | 使用 Linux 异步 I/O 引擎（必须用 libaio 而非默认的 psync，否则无法利用高队列深度） |
| `--iodepth=64/256` | I/O 队列深度。顺序读写用 64，随机 I/O 用 256 以充分打满 IOPS |
| `--numjobs=4` | 并发线程数，模拟多进程同时读写的场景 |
| `--bs=1M/4k` | 块大小。1M 用于顺序吞吐测试，4K 用于 IOPS 测试 |
| `--runtime=30` | 测试时长 30 秒 |
| `--group_reporting` | 聚合所有线程的统计数据 |

---

## Step 6: 创建 GCS 存储桶

创建一个 GCS 存储桶用于存放模型权重、数据集或 checkpoint 等大文件。
建议桶的区域与 VM 一致，减少跨区域传输延迟和费用。

```bash
gcloud storage buckets create gs://${BUCKET_NAME} \
    --project=${PROJECT_ID} \
    --location=${BUCKET_LOCATION} \
    --uniform-bucket-level-access
```

**参数说明**：

| 参数 | 说明 |
|------|------|
| `--location` | 桶的区域。GCS 桶是 **region 级别**（如 `us-central1`），不是 zone 级别 |
| `--uniform-bucket-level-access` | 启用统一桶级访问控制（推荐），禁用 ACL，仅使用 IAM 策略管理权限 |

> **注意**：GCS 桶名称全球唯一，建议使用 `{project-id}-{用途}` 的命名格式（如 `my-project-tpu-data`）。

验证桶创建：

```bash
gcloud storage buckets describe gs://${BUCKET_NAME} \
    --format="table(name,location,storageClass)"
```

预期输出：

```
NAME              LOCATION     STORAGE_CLASS
my-tpu-data       US-CENTRAL1  STANDARD
```

### 在 VM 中使用 GCS

VM 创建时已设置 `--scopes=cloud-platform`，因此可以直接使用 `gcloud storage` 命令访问桶：

```bash
# 上传文件到 GCS
gcloud storage cp /mnt/data/model-weights.safetensors gs://${BUCKET_NAME}/models/

# 从 GCS 下载文件到数据盘
gcloud storage cp gs://${BUCKET_NAME}/models/model-weights.safetensors /mnt/data/

# 并行下载大文件（自动分片）
gcloud storage cp --recursive gs://${BUCKET_NAME}/models/ /mnt/data/models/
```

---

## 基准测试结果（实测）

**测试环境**：TPU v7x-8 (tpu7x-standard-4t), 2TB Hyperdisk ML, provisioned 3000 MiB/s

| 测试项 | 实测值 | 单机上限 | 达标情况 |
|--------|--------|----------|----------|
| 顺序读吞吐 (1M bs) | **2,415 MiB/s** | 2,400 MiB/s | ✅ 达到上限 |
| 顺序写吞吐 (1M bs) | **2,416 MiB/s** | 2,400 MiB/s | ✅ 达到上限 |
| 随机读 IOPS (4K bs) | **48,300** | 38,400 | ✅ 超过标称 |
| 随机写 IOPS (4K bs) | **48,300** | 38,400 | ✅ 超过标称 |

### 关于单机吞吐量上限

根据 [Hyperdisk ML 官方文档](https://docs.google.com/compute/docs/disks/hd-types/hyperdisk-ml#perf-limits)，
**每台 VM 的 Hyperdisk ML 吞吐量上限为 2,400 MiB/s**（所有加速器优化机型一致）。
实测 2,415 MiB/s 精确命中此上限，与以下参数调整无关：

- Block size: 1M / 4M / 16M
- 并发线程: 4 / 8 / 16 jobs
- 队列深度: 16 / 64 / 128 iodepth

这是 Hyperdisk ML 的 **per-VM 架构限制**，不是磁盘本身的性能瓶颈。
如需更高的聚合吞吐，应将磁盘切换为只读模式并挂载到多台 VM（每台 VM 各自可达 2,400 MiB/s）。

> **A3-8g 例外**：A3 Ultra（H100）以只读模式挂载时可达 4,000 MiB/s。TPU v7x 文档暂未列出，但实测与 2,400 MiB/s 上限一致。

### 与 Hyperdisk Balanced 对比

| 指标 | Hyperdisk Balanced | Hyperdisk ML | 提升 |
|------|-------------------|-------------|------|
| 顺序读写吞吐 | 1,200 MiB/s | 2,415 MiB/s | **2.0x** |
| 随机 IOPS | 40,000 | 48,300 | **1.2x** |
| 多机只读共享 | ❌ 不支持 | ✅ 最多 2,500 台 | — |

---

## Hyperdisk ML 读写模式说明

Hyperdisk ML 支持两种访问模式，通过两阶段工作流使用：

### 阶段 1：单机读写（READ_WRITE_SINGLE）

创建后默认模式。挂载到单台 VM 以读写模式使用，用于：
- 下载模型权重
- 准备数据集
- 生成 FP4/FP8 cache 文件

### 阶段 2：多机只读（READ_ONLY_MANY）

数据准备完成后，可以切换为只读模式分发给多台 VM：

```bash
# 1. 先 detach 磁盘（必须在所有 VM 上卸载）
gcloud compute instances detach-disk ${INSTANCE_NAME} \
    --disk=${DATA_DISK_NAME} \
    --project=${PROJECT_ID} \
    --zone=${ZONE}

# 2. 切换 access mode（不可逆操作！）
gcloud compute disks update ${DATA_DISK_NAME} \
    --project=${PROJECT_ID} \
    --zone=${ZONE} \
    --access-mode=READ_ONLY_MANY

# 3. 以只读模式挂载到多台 VM
gcloud compute instances attach-disk ${INSTANCE_NAME} \
    --disk=${DATA_DISK_NAME} \
    --device-name=data-disk \
    --mode=ro \
    --project=${PROJECT_ID} \
    --zone=${ZONE}
```

**重要限制**：

| 规则 | 说明 |
|------|------|
| 不支持 multi-writer | 不能多台 VM 同时写入 |
| 切只读不可逆 | 一旦设为 `READ_ONLY_MANY`，**无法恢复**为读写模式 |
| 切换前必须 detach | 磁盘挂载状态下不能修改 access mode |
| 最多共享实例数 | ≤256 GiB: 2,500 台; 257 GiB-1 TiB: 600 台; 1-2 TiB: 300 台; 2-16 TiB: 128 台; >16 TiB: 30 台 |
| 共享时的吞吐要求 | 超过 20 台 VM 共享时，需为每台 VM 预配至少 100 MiB/s |

---

## 附录: 磁盘类型选型参考

TPU v7x 仅支持以下两种 Hyperdisk 类型：

| 磁盘类型 | 单机最大吞吐 | IOPS 配置 | 多机共享 | 适用场景 |
|----------|-------------|-----------|----------|----------|
| **Hyperdisk Balanced** | 实测 ~1,200 MiB/s（需手动设置 IOPS + 吞吐） | 手动设置 | ❌ | 通用读写、开发测试 |
| **Hyperdisk ML** | **2,400 MiB/s**（per-VM 硬上限） | 自动 = 吞吐 × 16 | ✅ 最多 2,500 台 | 模型权重分发、推理集群 |

> **不支持的磁盘类型**：Hyperdisk Extreme 和标准 Persistent Disk (pd-ssd/pd-balanced) 均不兼容 `tpu7x-standard-4t` 机器类型。

### 月费估算（2TB 磁盘）

| 费用项 | Hyperdisk Balanced (40K IOPS, 1200 MiB/s) | Hyperdisk ML (2500 MiB/s) |
|--------|-------------------------------------------|---------------------------|
| 存储 | 2048 GiB × $0.06 = $123 | 2048 GiB × $0.08 = $164 |
| 吞吐 | 1200 MiB/s × $0.06 = $72 | 2500 MiB/s × $0.12 = $300 |
| IOPS | 40K × $0.006 = $240 | 包含在吞吐中 |
| **合计** | **~$435/月** | **~$464/月** |

> **省钱提示**：单机场景下 provisioned-throughput 设为 2500 即可打满 2,400 MiB/s 上限，
> 比设 3000 每月省 $60（(3000-2500) × $0.12），比设 6000 每月省 $420。
> 吞吐量每 6 小时只能修改一次，建议一次设对。

---

*最后更新: 2026-04-28 | 测试人: Chris Yang*
