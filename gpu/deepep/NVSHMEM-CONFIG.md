# NVSHMEM v3.2.5 配置指南

## 📋 目录

1. [构建信息](#构建信息)
2. [关键配置项分析](#关键配置项分析)
3. [性能调优建议](#性能调优建议)
4. [调试与监控](#调试与监控)
5. [完整配置参考](#完整配置参考)
6. [最佳实践](#最佳实践)

---

## 🔧 构建信息

### 版本信息
- **NVSHMEM版本**: v3.2.5
- **CUDA API**: 12.0.90
- **CUDA Driver**: 12.0.90
- **构建时间**: Oct 27 2025 10:52:39

### 已启用特性
- ✅ **IBGDA支持** (`NVSHMEM_IBGDA_SUPPORT=ON`) - GPU直接访问InfiniBand
- ✅ **IBRC支持** (`NVSHMEM_IBRC_SUPPORT=ON`) - InfiniBand可靠连接
- ✅ **NVTX支持** (`NVSHMEM_NVTX=ON`) - 性能分析工具集成
- ✅ **GDRCopy支持** (`NVSHMEM_USE_GDRCOPY=ON`) - GPU Direct RDMA Copy加速

### 已禁用特性
- ❌ **UCX传输** (`NVSHMEM_UCX_SUPPORT=OFF`)
- ❌ **NCCL集成** (`NVSHMEM_USE_NCCL=OFF`)
- ❌ **MPI支持** (`NVSHMEM_MPI_SUPPORT=OFF`)
- ❌ **PMIx支持** (`NVSHMEM_PMIX_SUPPORT=OFF`)

### 依赖路径
```bash
CUDA_HOME=/usr/local/cuda
GDRCOPY_HOME=/opt/deepep/gdrcopy
LIBFABRIC_HOME=/usr/local/libfabric
MPI_HOME=/usr/local/ompi
NCCL_HOME=/usr/local/nccl
NVSHMEM_PREFIX=/usr/local/nvshmem
```

---

## 📊 关键配置项分析

### 1. 对称堆内存配置

#### `NVSHMEM_SYMMETRIC_SIZE`
**默认值**: `1073741824` (1GB)

**说明**: 指定每个PE的对称堆内存大小（字节）

**支持的单位后缀**:
- `k/K`: × 2^10 (KiB)
- `m/M`: × 2^20 (MiB)
- `g/G`: × 2^30 (GiB)
- `t/T`: × 2^40 (TiB)

**示例**:
```bash
export NVSHMEM_SYMMETRIC_SIZE=4G      # 4GB
export NVSHMEM_SYMMETRIC_SIZE=2048M   # 2GB
export NVSHMEM_SYMMETRIC_SIZE=0.5G    # 512MB
```

**调优建议**:
- **小规模应用**: 1-2GB
- **中等规模应用**: 2-4GB
- **大规模应用**: 4-8GB
- **最大限制**: `NVSHMEM_MAX_MEMORY_PER_GPU=137438953472` (128GB)

---

### 2. 传输层配置

#### `NVSHMEM_REMOTE_TRANSPORT`
**默认值**: `"ibrc"`

**可选值**:
- `ibrc`: InfiniBand可靠连接（推荐用于大多数场景）
- `ucx`: UCX传输（需要UCX支持，当前构建未启用）
- `libfabric`: Libfabric传输
- `ibdevx`: InfiniBand DevX（实验性）
- `none`: 仅本地通信

**示例**:
```bash
export NVSHMEM_REMOTE_TRANSPORT=ibrc
```

#### `NVSHMEM_IB_ENABLE_IBGDA`
**默认值**: `false`

**说明**: 启用GPU直接通信传输，可显著提升性能

**推荐配置**:
```bash
export NVSHMEM_IB_ENABLE_IBGDA=true  # 启用GPU直接访问
```

#### `NVSHMEM_ENABLE_NIC_PE_MAPPING`
**默认值**: `false`

**说明**:
- `false`: PE分配距离最近的NIC
- `true`: 使用轮询方式分配NIC或使用`NVSHMEM_HCA_PE_MAPPING`

**跨节点通信建议**:
```bash
export NVSHMEM_ENABLE_NIC_PE_MAPPING=true
```

---

### 3. P2P通信优化

#### `NVSHMEM_DISABLE_P2P`
**默认值**: `false`

**说明**: 控制GPU P2P连接

**推荐**: 保持为`false`以启用P2P通信

#### `NVSHMEM_DISABLE_CUDA_VMM`
**默认值**: `false`

**说明**: 控制CUDA虚拟内存管理用于P2P映射

**平台建议**:
- **x86平台**: 保持`false`（默认启用）
- **P9平台**: 默认禁用

**要求**: CUDA Runtime和Driver版本 ≥ 11.3

#### `NVSHMEM_DISABLE_MNNVL`
**默认值**: `false`

**说明**: 控制MNNVL（Multi-Node NVLink）连接

**推荐**: 保持为`false`以在支持的平台上启用

#### `NVSHMEM_DISABLE_NVLS`
**默认值**: `false`

**说明**: 控制NVLS SHARP资源用于集合通信

**推荐**: 保持为`false`以获得最佳集合通信性能

---

### 4. 集合通信配置

#### Barrier优化

**`NVSHMEM_BARRIER_DISSEM_KVAL`**
- **默认值**: `2`
- **说明**: 分散算法的基数
- **推荐值**: 2-4之间，较大值可能提升大规模barrier性能

**`NVSHMEM_BARRIER_TG_DISSEM_KVAL`**
- **默认值**: `2`
- **说明**: 线程组barrier的分散算法基数

#### Reduce操作优化

**`NVSHMEM_REDUCE_SCRATCH_SIZE`**
- **默认值**: `524288` (512KB)
- **说明**: 每个团队的对称堆内存保留空间
- **最小值**: 16B，必须是8B的倍数

**调优建议**:
```bash
# 标准配置
export NVSHMEM_REDUCE_SCRATCH_SIZE=1M

# 大规模reduce操作
export NVSHMEM_REDUCE_SCRATCH_SIZE=2M
```

**`NVSHMEM_REDUCE_NVLS_THRESHOLD`**
- **默认值**: `2048`
- **说明**: 使用allreduce one-shot算法的消息大小阈值

**`NVSHMEM_REDUCE_RECEXCH_KVAL`**
- **默认值**: `2`
- **说明**: 递归交换reduce算法的基数

#### Fcollect优化

**`NVSHMEM_FCOLLECT_LL_THRESHOLD`**
- **默认值**: `2048`
- **说明**: 使用LL算法的阈值

**`NVSHMEM_FCOLLECT_LL128_THRESHOLD`**
- **默认值**: `0`
- **说明**: 使用LL128算法的阈值

**`NVSHMEM_FCOLLECT_NVLS_THRESHOLD`**
- **默认值**: `16777216` (16MB)
- **说明**: 使用NVLS算法的阈值

**`NVSHMEM_FCOLLECT_NTHREADS`**
- **默认值**: `512`
- **说明**: fcollect集合操作每个块的线程数

#### Broadcast优化

**`NVSHMEM_BCAST_ALGO`**
- **默认值**: `0`
- **值**: `0` = 使用默认算法选择策略

**`NVSHMEM_BCAST_TREE_KVAL`**
- **默认值**: `2`
- **说明**: broadcast树算法的基数

---

### 5. 代理线程配置

#### `NVSHMEM_PROXY_REQUEST_BATCH_MAX`
**默认值**: `32`

**说明**: 代理线程在单次进度循环中处理的最大请求数

**高吞吐场景优化**:
```bash
export NVSHMEM_PROXY_REQUEST_BATCH_MAX=64   # 高吞吐
export NVSHMEM_PROXY_REQUEST_BATCH_MAX=128  # 极高吞吐（谨慎使用）
```

#### `NVSHMEM_DISABLE_LOCAL_ONLY_PROXY`
**默认值**: `false`

**说明**: 在NVLink-only配置下完全禁用代理线程

**注意**: 禁用后将无法使用：
- 设备端全局退出
- 设备端等待超时轮询

---

### 6. Bootstrap配置

#### `NVSHMEM_BOOTSTRAP`
**默认值**: `"PMI"`

**可选值**:
- `PMI`: PMI引导（默认）
- `MPI`: MPI引导
- `SHMEM`: SHMEM引导
- `plugin`: 插件引导
- `UID`: UID引导

#### `NVSHMEM_BOOTSTRAP_PMI`
**默认值**: `"PMI"`

**可选值**:
- `PMI`: PMI标准
- `PMI-2`: PMI-2标准
- `PMIX`: PMIx标准

**Slurm环境推荐**:
```bash
export NVSHMEM_BOOTSTRAP=PMI
export NVSHMEM_BOOTSTRAP_PMI=PMI-2  # 或PMIX
```

---

### 7. 内存管理配置

#### `NVSHMEM_HEAP_KIND`
**默认值**: `"DEVICE"`

**可选值**:
- `VIDMEM`: GPU显存（默认）
- `SYSMEM`: 系统内存

#### `NVSHMEM_CUMEM_GRANULARITY`
**默认值**: `536870912` (512MB)

**说明**: `cuMemAlloc`/`cuMemCreate`的粒度

**影响**: 较大的粒度可能导致内存碎片

#### `NVSHMEM_CUMEM_HANDLE_TYPE`
**默认值**: `"FILE_DESCRIPTOR"`

**可选值**:
- `FILE_DESCRIPTOR`: 文件描述符
- `FABRIC`: Fabric句柄

---

### 8. 团队管理

#### `NVSHMEM_MAX_TEAMS`
**默认值**: `32`

**说明**: 允许的最大同时团队数

**大规模应用建议**:
```bash
export NVSHMEM_MAX_TEAMS=64
```

---

## 🚀 性能调优建议

### 场景1: 节点内高性能通信

**适用**: 单节点多GPU训练，GPU数量≤8

```bash
#!/bin/bash
# 节点内优化配置

# 启用所有P2P特性
export NVSHMEM_DISABLE_P2P=false
export NVSHMEM_DISABLE_MNNVL=false
export NVSHMEM_DISABLE_NVLS=false
export NVSHMEM_DISABLE_CUDA_VMM=false

# 适中的对称堆大小
export NVSHMEM_SYMMETRIC_SIZE=2G

# 使用本地传输
export NVSHMEM_REMOTE_TRANSPORT=none

# 可选：禁用代理线程（仅NVLink）
# export NVSHMEM_DISABLE_LOCAL_ONLY_PROXY=true
```

---

### 场景2: 跨节点InfiniBand通信

**适用**: 多节点分布式训练，使用InfiniBand互连

```bash
#!/bin/bash
# 跨节点IB优化配置

# InfiniBand传输
export NVSHMEM_REMOTE_TRANSPORT=ibrc

# 启用GPU直接访问
export NVSHMEM_IB_ENABLE_IBGDA=true

# NIC-PE映射
export NVSHMEM_ENABLE_NIC_PE_MAPPING=true

# 较大的对称堆
export NVSHMEM_SYMMETRIC_SIZE=4G

# P2P优化
export NVSHMEM_DISABLE_P2P=false
export NVSHMEM_DISABLE_MNNVL=false

# 代理线程优化
export NVSHMEM_PROXY_REQUEST_BATCH_MAX=64
```

---

### 场景3: 大规模集合通信优化

**适用**: 大规模All-Reduce、Barrier等集合操作

```bash
#!/bin/bash
# 集合通信优化配置

# Barrier优化
export NVSHMEM_BARRIER_DISSEM_KVAL=4

# Reduce优化
export NVSHMEM_REDUCE_SCRATCH_SIZE=2M
export NVSHMEM_REDUCE_NVLS_THRESHOLD=4096

# NVLS优化
export NVSHMEM_DISABLE_NVLS=false
export NVSHMEM_FCOLLECT_NVLS_THRESHOLD=32M
export NVSHMEM_REDUCESCATTER_NVLS_THRESHOLD=32M

# 增加团队数量
export NVSHMEM_MAX_TEAMS=64

# 对称堆配置
export NVSHMEM_SYMMETRIC_SIZE=4G
```

---

### 场景4: 延迟敏感型应用

**适用**: 需要最低延迟的实时应用

```bash
#!/bin/bash
# 低延迟优化配置

# 启用所有直接通信特性
export NVSHMEM_IB_ENABLE_IBGDA=true
export NVSHMEM_DISABLE_P2P=false
export NVSHMEM_DISABLE_CUDA_VMM=false

# 快速barrier
export NVSHMEM_BARRIER_DISSEM_KVAL=2

# 优化代理线程
export NVSHMEM_PROXY_REQUEST_BATCH_MAX=16

# 绕过某些检查（谨慎使用）
# export NVSHMEM_ASSERT_ATOMICS_SYNC=true
# export NVSHMEM_BYPASS_FLUSH=true
```

---

### 场景5: 吞吐量优先应用

**适用**: 批处理、大数据量传输

```bash
#!/bin/bash
# 高吞吐量优化配置

# 大对称堆
export NVSHMEM_SYMMETRIC_SIZE=8G

# 大批量处理
export NVSHMEM_PROXY_REQUEST_BATCH_MAX=128

# 大阈值
export NVSHMEM_FCOLLECT_NVLS_THRESHOLD=64M
export NVSHMEM_REDUCESCATTER_NVLS_THRESHOLD=64M

# 内存粒度优化
export NVSHMEM_CUMEM_GRANULARITY=1G
```

---

## 🐛 调试与监控

### 调试配置

#### `NVSHMEM_DEBUG`
**默认值**: `""`

**可选值**:
- `VERSION`: 版本信息
- `WARN`: 警告消息
- `INFO`: 信息消息
- `ABORT`: 中止消息
- `TRACE`: 跟踪消息

**示例**:
```bash
# 启用警告和信息
export NVSHMEM_DEBUG=WARN,INFO

# 完整调试
export NVSHMEM_DEBUG=VERSION,WARN,INFO,TRACE
```

#### `NVSHMEM_DEBUG_SUBSYS`
**默认值**: `""`

**可选值**:
- `INIT`: 初始化
- `COLL`: 集合通信
- `P2P`: 点对点通信
- `PROXY`: 代理线程
- `TRANSPORT`: 传输层
- `MEM`: 内存管理
- `BOOTSTRAP`: 引导
- `TOPO`: 拓扑
- `UTIL`: 工具
- `ALL`: 所有子系统

**示例**:
```bash
# 调试传输和P2P
export NVSHMEM_DEBUG_SUBSYS=TRANSPORT,P2P

# 排除某些子系统
export NVSHMEM_DEBUG_SUBSYS=^UTIL,^TOPO

# 所有子系统
export NVSHMEM_DEBUG_SUBSYS=ALL
```

#### `NVSHMEM_DEBUG_FILE`
**默认值**: `""`

**说明**: 调试输出文件名

**占位符**:
- `%h`: 主机名
- `%p`: 进程ID

**示例**:
```bash
export NVSHMEM_DEBUG_FILE=/tmp/nvshmem_%h_%p.log
```

#### `NVSHMEM_DEBUG_ATTACH_DELAY`
**默认值**: `0`

**说明**: 初始化时的延迟秒数，用于附加调试器

**示例**:
```bash
export NVSHMEM_DEBUG_ATTACH_DELAY=10
```

---

### 性能分析 (NVTX)

#### `NVSHMEM_NVTX`
**默认值**: `"off"`

**可选值（逗号分隔）**:
- `init`: 库设置
- `alloc`: 内存管理
- `launch`: 内核启动例程
- `coll`: 集合通信
- `wait`: 阻塞点对点同步
- `wait_on_stream`: 流上的点对点同步
- `test`: 非阻塞点对点同步
- `memorder`: 内存排序（quiet, fence）
- `quiet_on_stream`: nvshmemx_quiet_on_stream
- `atomic_fetch`: 获取原子内存操作
- `atomic_set`: 非获取原子内存操作
- `rma_blocking`: 阻塞远程内存访问
- `rma_nonblocking`: 非阻塞远程内存访问
- `proxy`: 代理线程活动
- `common`: init,alloc,launch,coll,memorder,wait,atomic_fetch,rma_blocking,proxy
- `all`: 所有组
- `off`: 禁用所有NVTX

**示例**:
```bash
# 常用分析
export NVSHMEM_NVTX=common

# 完整分析
export NVSHMEM_NVTX=all

# 自定义组合
export NVSHMEM_NVTX=coll,rma_blocking,proxy
```

**使用Nsight Systems分析**:
```bash
nsys profile --trace=nvtx,cuda,osrt \
  --output=nvshmem_profile \
  ./your_application
```

---

### 错误检查

#### `NVSHMEM_ENABLE_ERROR_CHECKS`
**默认值**: `false`

**说明**: 启用错误检查（影响性能）

**开发阶段建议**:
```bash
export NVSHMEM_ENABLE_ERROR_CHECKS=true
```

**生产环境建议**: 保持为`false`

---

### 信息输出

#### `NVSHMEM_VERSION`
**默认值**: `false`

**说明**: 启动时打印库版本

```bash
export NVSHMEM_VERSION=true
```

#### `NVSHMEM_INFO`
**默认值**: `false`

**说明**: 启动时打印环境变量选项

```bash
export NVSHMEM_INFO=true
```

#### `NVSHMEM_INFO_HIDDEN`
**默认值**: `false`

**说明**: 启动时打印隐藏的环境变量选项

```bash
export NVSHMEM_INFO_HIDDEN=true
```

---

## 📝 完整配置参考

### 标准生产环境配置

```bash
#!/bin/bash
# NVSHMEM生产环境标准配置
# 适用于多节点InfiniBand集群

# ===== 基础配置 =====
export NVSHMEM_SYMMETRIC_SIZE=4G
export NVSHMEM_BOOTSTRAP=PMI
export NVSHMEM_BOOTSTRAP_PMI=PMI-2

# ===== 传输层配置 =====
export NVSHMEM_REMOTE_TRANSPORT=ibrc
export NVSHMEM_IB_ENABLE_IBGDA=true
export NVSHMEM_ENABLE_NIC_PE_MAPPING=true

# ===== P2P通信配置 =====
export NVSHMEM_DISABLE_P2P=false
export NVSHMEM_DISABLE_CUDA_VMM=false
export NVSHMEM_DISABLE_MNNVL=false
export NVSHMEM_DISABLE_NVLS=false

# ===== 集合通信配置 =====
export NVSHMEM_BARRIER_DISSEM_KVAL=2
export NVSHMEM_REDUCE_SCRATCH_SIZE=1M
export NVSHMEM_FCOLLECT_NVLS_THRESHOLD=16M
export NVSHMEM_REDUCESCATTER_NVLS_THRESHOLD=16M

# ===== 代理线程配置 =====
export NVSHMEM_PROXY_REQUEST_BATCH_MAX=64

# ===== 团队管理 =====
export NVSHMEM_MAX_TEAMS=32

# ===== 监控配置（可选）=====
export NVSHMEM_DEBUG=WARN
# export NVSHMEM_NVTX=common  # 性能分析时启用
# export NVSHMEM_INFO=true     # 调试时启用
```

---

### 开发/调试环境配置

```bash
#!/bin/bash
# NVSHMEM开发和调试配置

# ===== 基础配置 =====
export NVSHMEM_SYMMETRIC_SIZE=2G
export NVSHMEM_BOOTSTRAP=PMI

# ===== 调试配置 =====
export NVSHMEM_DEBUG=VERSION,WARN,INFO
export NVSHMEM_DEBUG_SUBSYS=INIT,COLL,P2P,TRANSPORT
export NVSHMEM_DEBUG_FILE=/tmp/nvshmem_%h_%p.log

# ===== 错误检查 =====
export NVSHMEM_ENABLE_ERROR_CHECKS=true

# ===== 信息输出 =====
export NVSHMEM_VERSION=true
export NVSHMEM_INFO=true
export NVSHMEM_INFO_HIDDEN=true

# ===== 性能分析 =====
export NVSHMEM_NVTX=all

# ===== 调试器附加延迟 =====
# export NVSHMEM_DEBUG_ATTACH_DELAY=10
```

---

### 基准测试配置

```bash
#!/bin/bash
# NVSHMEM基准测试配置
# 用于性能评估和优化

# ===== 基础配置 =====
export NVSHMEM_SYMMETRIC_SIZE=8G

# ===== 传输配置 =====
export NVSHMEM_REMOTE_TRANSPORT=ibrc
export NVSHMEM_IB_ENABLE_IBGDA=true

# ===== 性能优化 =====
export NVSHMEM_DISABLE_P2P=false
export NVSHMEM_DISABLE_NVLS=false
export NVSHMEM_PROXY_REQUEST_BATCH_MAX=128

# ===== 性能分析 =====
export NVSHMEM_NVTX=common

# ===== 最小日志 =====
export NVSHMEM_DEBUG=""
export NVSHMEM_INFO=false
```

---

## 🎯 最佳实践

### 1. 内存配置原则

✅ **推荐做法**:
- 根据应用需求设置合适的`NVSHMEM_SYMMETRIC_SIZE`
- 留出至少20%的GPU内存给其他用途
- 大规模应用使用4GB以上的对称堆

❌ **避免**:
- 将对称堆设置得过大，导致GPU内存不足
- 在小规模应用中使用过大的对称堆

### 2. 传输层选择

✅ **推荐做法**:
- **节点内**: 使用P2P + NVLink (`REMOTE_TRANSPORT=none`)
- **跨节点**: 使用InfiniBand (`REMOTE_TRANSPORT=ibrc`)
- 启用`IBGDA`以获得最佳性能

❌ **避免**:
- 在不支持的平台上强制启用某些特性
- 混用不兼容的传输配置

### 3. 集合通信优化

✅ **推荐做法**:
- 根据消息大小调整阈值参数
- 启用NVLS以加速大规模集合操作
- 增加`REDUCE_SCRATCH_SIZE`以支持大规模reduce

❌ **避免**:
- 对所有消息大小使用相同的算法
- 禁用NVLS等高性能特性

### 4. 调试与开发

✅ **推荐做法**:
- 开发阶段启用`ERROR_CHECKS`
- 使用`DEBUG_SUBSYS`精确控制调试输出
- 利用NVTX进行性能分析

❌ **避免**:
- 生产环境启用完整调试
- 忽略警告消息

### 5. 性能监控

✅ **推荐做法**:
```bash
# 定期检查NVSHMEM性能
export NVSHMEM_NVTX=common
nsys profile --trace=nvtx,cuda ./app

# 分析瓶颈
export NVSHMEM_DEBUG=WARN
export NVSHMEM_DEBUG_SUBSYS=TRANSPORT,COLL
```

### 6. 生产部署检查清单

在生产环境部署前，请确认：

- [ ] 对称堆大小合理（不超过GPU内存的80%）
- [ ] 传输层配置正确（IBRC + IBGDA）
- [ ] P2P和NVLS已启用
- [ ] Bootstrap配置与作业调度器匹配
- [ ] 调试选项已禁用（性能优先）
- [ ] 代理线程配置已优化
- [ ] 集合通信阈值已调优
- [ ] 在测试环境验证过配置

---

## ⚠️ 常见问题与注意事项

### 1. CUDA MPS配置

**问题**: 使用CUDA MPS时NVSHMEM初始化失败

**解决方案**:
```bash
# 确保总线程百分比不超过100%
# 例如，4个进程每个25%
export CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=25

# 或强制忽略（风险自负）
export NVSHMEM_IGNORE_CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=true
```

### 2. 内存碎片问题

**问题**: 频繁分配/释放导致内存碎片

**解决方案**:
```bash
# 调整内存粒度
export NVSHMEM_CUMEM_GRANULARITY=256M  # 减小粒度

# 或使用预分配策略
export NVSHMEM_SYMMETRIC_SIZE=8G  # 一次性分配足够空间
```

### 3. NCCL集成

**注意**: 当前构建**未启用NCCL支持**

如需NCCL集成，需要：
1. 安装NCCL库
2. 使用`NVSHMEM_USE_NCCL=ON`重新编译NVSHMEM
3. 设置`NCCL_HOME`环境变量

### 4. UCX传输

**注意**: 当前构建**未启用UCX支持**

如需UCX传输，需要：
1. 安装UCX库
2. 使用`NVSHMEM_UCX_SUPPORT=ON`重新编译NVSHMEM
3. 设置`UCX_HOME`环境变量

### 5. 性能下降排查

**步骤1**: 检查P2P状态
```bash
export NVSHMEM_DEBUG=INFO
export NVSHMEM_DEBUG_SUBSYS=P2P,TRANSPORT
```

**步骤2**: 验证IB连接
```bash
ibv_devinfo  # 检查IB设备
nvidia-smi topo -m  # 检查GPU拓扑
```

**步骤3**: 使用NVTX分析
```bash
export NVSHMEM_NVTX=all
nsys profile --trace=nvtx ./app
```

---

## 📚 参考资源

### 官方文档
- [NVSHMEM Programming Guide](https://docs.nvidia.com/nvshmem/)
- [NVSHMEM API Reference](https://docs.nvidia.com/nvshmem/api/)

### 性能优化
- [NVSHMEM Best Practices](https://docs.nvidia.com/nvshmem/best-practices/)
- [GPU Direct RDMA](https://docs.nvidia.com/cuda/gpudirect-rdma/)

### 工具
- [Nsight Systems](https://developer.nvidia.com/nsight-systems)
- [NVTX](https://docs.nvidia.com/nvtx/)

---

## 📄 版本历史

- **v3.2.5** (Oct 27 2025)
  - 当前版本
  - CUDA 12.0.90支持
  - IBGDA和IBRC支持
  - NVTX集成

---

## 📧 支持与反馈

如有问题或建议，请联系：
- NVIDIA NVSHMEM支持团队
- 查阅[NVSHMEM GitHub Issues](https://github.com/NVIDIA/nvshmem/issues)

---

**文档创建日期**: 2025-10-27  
**NVSHMEM版本**: v3.2.5  
**作者**: GPU-TPU-Pedia Project