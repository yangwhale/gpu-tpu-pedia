# GB300 (A4X Max) Slurm 集群部署与 NCCL Benchmark

> **方案**: GCP Cluster Toolkit + Slurm，官方推荐的 GB300 裸机编排方式
> **参考**: [GCP: Create an A4X Max Slurm cluster](https://cloud.google.com/cluster-toolkit/docs/deploy/slurm/create-a4x-max-cluster)
> **项目**: `tencent-gcp-taiji-poc`, Zone: `us-central1-b`

---

## 前提条件

| 项目 | 说明 |
|------|------|
| Cluster Toolkit | v1.62.0+，本地或 Cloud Shell 安装 |
| Reservation | GB300 reservation (subblock 级别) |
| Filestore quota | 10 TiB HIGH_SCALE_SSD (zonal) |
| GCS bucket | 用于 Terraform state |
| 节点数 | 建议 18 的倍数 (一个 NVLink domain = 18 节点 72 GPU) |

---

## 1. 安装 Cluster Toolkit

```bash
# Cloud Shell 或本地 Linux
git clone https://github.com/GoogleCloudPlatform/cluster-toolkit.git
cd cluster-toolkit
./gcluster --version  # 确认 >= v1.62.0
```

## 2. 创建 GCS Bucket (Terraform state)

```bash
PROJECT=tencent-gcp-taiji-poc
BUCKET=chrisya-gb300-slurm-tf-state

gcloud storage buckets create gs://$BUCKET \
  --project=$PROJECT \
  --default-storage-class=STANDARD \
  --location=us-central1 \
  --uniform-bucket-level-access
gcloud storage buckets update gs://$BUCKET --versioning
```

## 3. 创建部署文件

```yaml
# a4xmax-deployment.yaml
terraform_backend_defaults:
  type: gcs
  configuration:
    bucket: chrisya-gb300-slurm-tf-state
vars:
  deployment_name: chrisya-gb300-slurm
  project_id: tencent-gcp-taiji-poc
  region: us-central1
  zone: us-central1-b
  a4x_max_cluster_size: 16        # 先用 16 节点测试
  a4x_max_reservation_name: nvidia-gb300-dxkhoz4ypk4mh
```

> **节点数建议**: 18 的倍数最优 (NVLink domain = 18 节点)。16 节点也可以，但无法充分利用 NVLink 全互联。

## 4. 部署集群

```bash
cd cluster-toolkit

./gcluster deploy -d a4xmax-deployment.yaml \
  examples/machine-learning/a4x-maxgpu-4g-metal/a4xmax-bm-slurm-blueprint.yaml
```

部署约 20-30 分钟。Blueprint 自动完成:
- 构建自定义镜像 (Ubuntu 24.04 + NVIDIA 580 + DOCA OFED + Slurm)
- 创建 VPC 网络 (IDPF 管理 + RoCE Metal RDMA)
- 部署 Slurm controller + login node + compute nodes
- 安装 asapd-lite (systemd service)
- 安装 GIB NCCL plugin (systemd service)
- 配置 IMEX prolog/epilog (自动管理)

## 5. 连接集群

```bash
# 找到 login node
gcloud compute instances list --zones=us-central1-b --filter="name ~ login" --format="value(name)"

# SSH 连接
gcloud compute ssh <LOGIN_NODE> --zone=us-central1-b --tunnel-through-iap
```

## 6. NCCL 测试

### 方式一: Ramble 自动化 (推荐)

```bash
# 在 login node 上执行，自动跑 2/4/8/16 节点的 all-gather/all-reduce/reduce-scatter
nohup bash /opt/apps/system_benchmarks/run-nccl-tests-via-ramble.sh >& nccl.log &

# 监控进度
tail -f nccl.log

# 结果在 nccl-tests_<timestamp>/summary.tsv
```

### 方式二: 手动 sbatch

```bash
# 在 login node 上
srun --nodes=2 --ntasks-per-node=4 --gpus-per-node=4 --partition=a4xmax \
  /usr/local/gib/scripts/run_nccl_tests.sh \
  -t all_reduce -b 1M -e 16G -f 2 -p 22 -g 4 \
  $(scontrol show hostnames $SLURM_NODELIST | tr '\n' ' ')
```

## 7. 已知问题 (本 reservation)

| 问题 | 影响 | Workaround |
|------|------|-----------|
| **mlx5_7 rail switch 故障** | 全 block 12 subblock 的第 8 个 NIC 不可用 | Slurm blueprint 的 `nccl.a4xmax.conf` 可能需要补充 `NCCL_IB_HCA` 排除 mlx5_7 |
| **NVSwitch P2P 初始化失败** | subblock-0001, 0007 全 NS | 避免使用这两个 subblock |
| **CX-8 firmware error** | 4 个 PCI 设备 dmesg 报错 (mlx5_2/3/6/7) | mlx5_2/3/6 功能正常，仅 mlx5_7 实际不可用 |

## 8. 销毁集群

```bash
cd cluster-toolkit
./gcluster destroy chrisya-gb300-slurm
```

> 注意: Filestore 有 deletion protection，需先关闭才能删除。

---

## 9. 参考

- [GCP: Create an A4X Max Slurm cluster](https://cloud.google.com/cluster-toolkit/docs/deploy/slurm/create-a4x-max-cluster)
- [GCP: Run NCCL on Slurm clusters](https://cloud.google.com/ai-hypercomputer/docs/nccl/test-slurm)
- [GCP: Run NCCL on Compute Engine instances](https://cloud.google.com/ai-hypercomputer/docs/nccl/test-vms)
- [Cluster Toolkit Blueprint (GitHub)](https://github.com/GoogleCloudPlatform/cluster-toolkit/blob/main/examples/machine-learning/a4x-maxgpu-4g-metal/a4xmax-bm-slurm-blueprint.yaml)
- [GB300 全栈组件详解](https://cc.higcp.com/pages/gb300-rdma-stack-guide-20260713.html)

---

*基于 GCP 官方文档 + 实测经验 · 2026-07-13*
