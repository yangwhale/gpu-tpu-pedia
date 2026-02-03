---
name: lssd-mounter
description: This skill should be used when users need to detect, format, and mount Google Cloud Local SSDs (NVMe) as a RAID0 array on /lssd. It covers SSD detection, RAID0 creation with mdadm, filesystem formatting, mounting, and HuggingFace cache directory setup. The skill is particularly useful for high-speed storage for ML model caching, checkpoints, and temporary data on Google Cloud VMs.
license: MIT
---

# LSSD Mounter

## Overview

LSSD Mounter 帮助用户在 Google Cloud VM 上检测、格式化并挂载本地 NVMe SSD 为 RAID0 阵列。挂载到 `/lssd` 后，可用于 HuggingFace 模型缓存、检查点存储和临时数据的高速存储。

Google Cloud 的 Local SSD 提供极高的 IOPS 和带宽（每块 SSD 高达 680MB/s 读取、360MB/s 写入），通过 RAID0 组合可获得数 GB/s 的吞吐量。

## When to Use This Skill

- 在 Google Cloud VM 上首次设置 Local SSD 存储
- 检查 Local SSD 挂载状态和可用空间
- 为 HuggingFace 模型下载配置高速缓存目录
- 诊断 Local SSD 相关问题

## Prerequisites

- Google Cloud VM with Local SSD attached
- sudo privileges
- mdadm package (usually pre-installed)

## Quick Start

### 检查 LSSD 状态

```bash
./scripts/check-lssd.sh
```

### 挂载 LSSD

```bash
sudo ./scripts/mount-lssd.sh
```

## Detection and Mounting Workflow

### Step 1: 检测已挂载状态

首先检查 `/lssd` 是否已挂载:

```bash
if mountpoint -q /lssd 2>/dev/null; then
    echo "LSSD 已挂载: $(df -h /lssd | tail -1 | awk '{print $2}')"
fi
```

### Step 2: 检测可用 NVMe SSD

如果未挂载，检测可用的 NVMe SSD:

```bash
# 检测 NVMe SSD 数量
NVME_COUNT=$(ls /dev/disk/by-id/ 2>/dev/null | grep -c "google-local-nvme-ssd" || echo 0)
echo "检测到 ${NVME_COUNT} 块 NVMe SSD"

# 列出所有 SSD
ls /dev/disk/by-id/google-local-nvme-ssd-*
```

常见配置:
- 1 块 SSD: 375 GB
- 4 块 SSD: 1.5 TB
- 8 块 SSD: 3 TB
- 16 块 SSD: 6 TB
- 24 块 SSD: 9 TB
- 32 块 SSD: 12 TB (a3-megagpu-8g, a3-highgpu-8g)

### Step 3: 创建 RAID0 阵列

使用 mdadm 创建 RAID0 阵列:

```bash
# 获取所有 NVMe SSD 设备
NVME_DEVICES=$(ls /dev/disk/by-id/google-local-nvme-ssd-* 2>/dev/null | sort)
NVME_COUNT=$(echo "$NVME_DEVICES" | wc -l)

# 检查 md0 是否已存在
if [ -e /dev/md0 ]; then
    echo "RAID 设备 /dev/md0 已存在"
else
    # 创建 RAID0 阵列
    sudo mdadm --create /dev/md0 --level=0 --raid-devices=$NVME_COUNT $NVME_DEVICES
fi
```

### Step 4: 格式化和挂载

```bash
# 格式化 (仅在新创建时)
sudo mkfs.ext4 -F /dev/md0

# 创建挂载点并挂载
sudo mkdir -p /lssd
sudo mount /dev/md0 /lssd

# 设置权限
sudo chmod a+w /lssd
```

### Step 5: 配置 HuggingFace 缓存

```bash
# 创建 HuggingFace 缓存目录
sudo mkdir -p /lssd/huggingface
sudo chmod a+w /lssd/huggingface

# 设置环境变量
export HF_HOME=/lssd/huggingface

# 持久化到 bashrc
if ! grep -q 'export HF_HOME=/lssd/huggingface' ~/.bashrc; then
    echo '' >> ~/.bashrc
    echo '# HuggingFace cache on LSSD (high-speed local SSD)' >> ~/.bashrc
    echo 'export HF_HOME=/lssd/huggingface' >> ~/.bashrc
fi
```

### Step 6: 预下载 DeepSeek 模型权重 (可选)

从 GCS 快速复制预缓存的 DeepSeek-V3 模型权重，避免从 HuggingFace 下载：

```bash
# 检查是否已经下载过
DEEPSEEK_PATH="/lssd/huggingface/hub/models--deepseek-ai--DeepSeek-V3"

if [ -d "$DEEPSEEK_PATH" ]; then
    echo "✓ DeepSeek-V3 权重已存在: $DEEPSEEK_PATH"
    du -sh "$DEEPSEEK_PATH"
else
    echo "下载 DeepSeek-V3 权重从 GCS..."
    gcloud storage cp -r gs://chrisya-gpu-pg-ase1/huggingface /lssd/
    echo "✓ DeepSeek-V3 权重下载完成"
fi
```

**说明:**
- GCS bucket `gs://chrisya-gpu-pg-ase1/huggingface` 包含预缓存的 DeepSeek-V3 FP8 权重
- 从 GCS 下载比从 HuggingFace 快很多（同区域带宽高）
- 权重约 ~600GB，包含完整的 safetensors 文件

## Common Scenarios

### Scenario 1: LSSD 已挂载

```
$ ./scripts/check-lssd.sh
[OK] /lssd 已挂载
容量: 11T | 已用: 2.3T | 可用: 8.7T | 使用率: 21%
RAID 状态: md0 : active raid0 nvme31n1[31] nvme30n1[30] ... [32/32] [UUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUU]
HF_HOME: /lssd/huggingface
```

### Scenario 2: 有 SSD 但未挂载

```
$ ./scripts/check-lssd.sh
[INFO] /lssd 未挂载
[INFO] 检测到 32 块 NVMe SSD
[INFO] 运行 sudo ./scripts/mount-lssd.sh 来挂载
```

### Scenario 3: 没有 Local SSD

```
$ ./scripts/check-lssd.sh
[WARN] 未检测到 NVMe SSD
[INFO] 此 VM 可能未配置 Local SSD
```

### Scenario 4: md0 已存在但未挂载

```
$ ./scripts/mount-lssd.sh
[INFO] RAID 设备 /dev/md0 已存在，跳过创建
[INFO] 挂载 /dev/md0 到 /lssd...
[OK] LSSD 挂载完成
```

## Troubleshooting

### 问题: mdadm 创建失败 - 包含了系统盘

**症状:**
```
mdadm: chunk size must be a power of 2, not 33
```
或 lsblk 显示 33 个设备（包括 2TB 系统盘）

**根因:**
使用 `ls /dev/nvme*n1` 会包含系统盘（通常是 2TB），导致 mdadm 创建失败。

**解决方案:**
仅选择 375GB 的 Local SSD，排除系统盘:

```bash
# 方法 1: 通过 google-local-nvme-ssd 符号链接（推荐）
NVME_DEVICES=$(ls /dev/disk/by-id/google-local-nvme-ssd-* 2>/dev/null | sort)

# 方法 2: 通过设备大小过滤（375GB = 402653184000 bytes）
NVME_DEVICES=$(lsblk -d -b -o NAME,SIZE | awk '$2 == 402653184000 {print "/dev/"$1}' | sort)
```

### 问题: mdadm 命令未找到

**解决方案:**
```bash
sudo apt-get update && sudo apt-get install -y mdadm
```

### 问题: mount 失败 - 文件系统损坏

**诊断:**
```bash
sudo fsck /dev/md0
```

**解决方案:**
如果数据不重要，可以重新格式化:
```bash
sudo mkfs.ext4 -F /dev/md0
sudo mount /dev/md0 /lssd
```

### 问题: RAID 阵列降级

**诊断:**
```bash
cat /proc/mdstat
```

**说明:**
Local SSD 数据在 VM 停止后会丢失，这是正常的。重新启动后需要重新创建 RAID。

### 问题: 挂载后权限问题

**解决方案:**
```bash
sudo chmod a+w /lssd
sudo chmod a+w /lssd/huggingface
```

## Verification

### 检查 RAID 状态

```bash
cat /proc/mdstat
```

期望输出:
```
md0 : active raid0 nvme31n1[31] nvme30n1[30] ...
      12497551360 blocks super 1.2 512k chunks
```

### 检查挂载状态

```bash
df -h /lssd
```

期望输出:
```
Filesystem      Size  Used Avail Use% Mounted on
/dev/md0         12T   24K   12T   1% /lssd
```

### 验证 HuggingFace 缓存

```bash
echo $HF_HOME
ls -la /lssd/huggingface
```

## Environment Variables

| Variable | Value | Description |
|----------|-------|-------------|
| HF_HOME | /lssd/huggingface | HuggingFace 模型和数据集缓存目录 |

## Performance Notes

- **读取带宽**: ~680 MB/s per SSD (RAID0: 线性扩展)
- **写入带宽**: ~360 MB/s per SSD (RAID0: 线性扩展)
- **IOPS**: ~400,000 per SSD

32 块 SSD RAID0 配置可提供:
- 读取: ~20 GB/s
- 写入: ~11 GB/s

## Resources

### scripts/

- `mount-lssd.sh` - 自动检测并挂载 Local SSD
- `check-lssd.sh` - 检查 LSSD 状态和配置

## Data Persistence Warning

**重要**: Google Cloud Local SSD 数据在以下情况会丢失:
- VM 停止 (stop)
- VM 抢占 (preemption)
- 主机维护事件

Local SSD 仅适用于:
- 临时缓存 (如 HuggingFace 模型)
- 可重新生成的检查点
- 临时数据处理

不要在 Local SSD 上存储无法恢复的重要数据。
