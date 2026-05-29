**English** | [中文](./README.md)

# TPU v7x VM Creation and Storage Configuration

> End-to-end guide: creating a TPU v7x-8 VM instance, configuring a high-throughput Hyperdisk ML data disk, creating a GCS object storage bucket,
> and validating disk performance via fio benchmarks. Applicable to storage preparation for both inference and training scenarios.

## Table of Contents

- [Hardware and Environment Overview](#hardware-and-environment-overview)
- [Step 1: Create the Hyperdisk ML Data Disk](#step-1-create-the-hyperdisk-ml-data-disk)
- [Step 2: Create the TPU v7x VM Instance](#step-2-create-the-tpu-v7x-vm-instance)
- [Step 3: SSH into the VM](#step-3-ssh-into-the-vm)
- [Step 4: Format and Mount the Data Disk](#step-4-format-and-mount-the-data-disk)
- [Step 5: Install fio and Run Benchmarks](#step-5-install-fio-and-run-benchmarks)
- [Step 6: Create the GCS Bucket](#step-6-create-the-gcs-bucket)
- [Benchmark Results (Measured)](#benchmark-results-measured)
- [Hyperdisk ML Read/Write Mode Notes](#hyperdisk-ml-readwrite-mode-notes)
- [Appendix: Disk Type Selection Reference](#appendix-disk-type-selection-reference)

---

## Hardware and Environment Overview

| Item | Value |
|------|-----|
| **Machine type** | `tpu7x-standard-4t` (4 chips = 8 devices) |
| **TPU model** | v7x (Ironwood) |
| **HBM** | 192 GB/chip, 768 GB total |
| **Host memory** | 944 GB |
| **Boot disk** | 500 GB Hyperdisk Balanced |
| **Data disk** | 2 TB Hyperdisk ML (provisioned 2500 MiB/s) |

### Environment Variables

The following variables are used in the commands below; modify them according to your situation:

```bash
export PROJECT_ID=<your-project-id>
export ZONE=us-central1-c                  # Modify to match the zone of your reservation
export INSTANCE_NAME=<your-instance-name>   # e.g., my-tpu7x-01
export DATA_DISK_NAME=<your-data-disk-name> # e.g., my-tpu7x-data-01
export RESERVATION_NAME=<your-reservation>  # TPU v7x reservation name
export VPC_NAME=<your-vpc-name>
export SUBNET_NAME=<your-subnet-name>
export BUCKET_NAME=<your-bucket-name>       # e.g., my-tpu-data
export BUCKET_LOCATION=us-central1           # GCS bucket region (same region as the VM)
```

---

## Step 1: Create the Hyperdisk ML Data Disk

Hyperdisk ML is a high-throughput disk type provided by GCP, designed specifically for AI/ML workloads.
Compared to Hyperdisk Balanced, it offers higher throughput bandwidth and multi-machine read-only sharing capability.

```bash
gcloud compute disks create ${DATA_DISK_NAME} \
    --project=${PROJECT_ID} \
    --zone=${ZONE} \
    --type=hyperdisk-ml \
    --size=2TB \
    --provisioned-throughput=2500
```

**Parameter explanation**:

| Parameter | Value | Description |
|------|-----|------|
| `--type` | `hyperdisk-ml` | Hyperdisk ML disk type |
| `--size` | `2TB` | Disk capacity |
| `--provisioned-throughput` | `2500` | Provisioned throughput (MiB/s). IOPS is automatically derived at a 16:1 ratio (2500 × 16 = 40K IOPS) |

> **Note**: Hyperdisk ML's **single-machine throughput ceiling is 2,400 MiB/s** ([official docs](https://docs.cloud.google.com/compute/docs/disks/hd-types/hyperdisk-ml#perf-limits)),
> and on TPU v7x it precisely hits this ceiling in testing (2,415 MiB/s). Setting provisioned-throughput to 2500 is enough to max out single-machine performance;
> setting a higher value does not improve performance, it only increases cost.
>
> If you need to share the disk across multiple VMs (read-only mode), set provisioned-throughput to (number of VMs) × 2,400.
>
> **Modification limits**: throughput can only be modified once every **6 hours**, and disk size can only be modified once every **4 hours**.

Verify the disk creation:

```bash
gcloud compute disks describe ${DATA_DISK_NAME} \
    --project=${PROJECT_ID} \
    --zone=${ZONE} \
    --format="table(name,type.basename(),sizeGb,provisionedThroughput,accessMode)"
```

Expected output:

```
NAME              TYPE          SIZE_GB  PROVISIONED_THROUGHPUT  ACCESS_MODE
my-tpu7x-data-01  hyperdisk-ml  2048     2500                   READ_WRITE_SINGLE
```

---

## Step 2: Create the TPU v7x VM Instance

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

**Key parameter explanation**:

| Parameter | Description |
|------|------|
| `--machine-type=tpu7x-standard-4t` | TPU v7x single node (4 chips, 8 devices) |
| `--maintenance-policy=TERMINATE` | TPU VMs must be set to TERMINATE (live migration not supported) |
| `--provisioning-model=RESERVATION_BOUND` | Bound to the specified capacity reservation |
| `--reservation-affinity=specific` | Use a specific reservation |
| `--create-disk=...image=...` | Boot disk uses the TPU v7x dedicated image |
| `--disk=...mode=rw` | Data disk mounted in read-write mode |
| `--no-shielded-secure-boot` | TPU images do not support Secure Boot |

> **Image selection**: use the TPU v7x dedicated image from the `ubuntu-os-accelerator-images` project.
> The image version number (e.g., `v20260422`) is updated periodically; you can check the image version used by other VMs in the same project with:
> ```bash
> gcloud compute disks describe <existing-tpu-vm-name> \
>     --project=${PROJECT_ID} --zone=${ZONE} \
>     --format="value(sourceImage)"
> ```

Verify the VM status:

```bash
gcloud compute instances describe ${INSTANCE_NAME} \
    --project=${PROJECT_ID} \
    --zone=${ZONE} \
    --format="table(name,status,machineType.basename(),networkInterfaces[0].accessConfigs[0].natIP)"
```

Wait for the status to change from `STAGING` to `RUNNING` (about 30-60 seconds).

---

## Step 3: SSH into the VM

Get the VM's external IP:

```bash
VM_IP=$(gcloud compute instances describe ${INSTANCE_NAME} \
    --project=${PROJECT_ID} \
    --zone=${ZONE} \
    --format="value(networkInterfaces[0].accessConfigs[0].natIP)")
echo "VM IP: ${VM_IP}"
```

SSH connect:

```bash
ssh -o StrictHostKeyChecking=no -i ~/.ssh/google_compute_engine ${USER}@${VM_IP}
```

> **Tip**: if you are on a Google Corp network, you may need to add `-o ProxyCommand=none` to bypass the proxy.
> If the connection times out, wait 30 seconds and retry (the SSH service may not be ready right after the VM starts).

---

## Step 4: Format and Mount the Data Disk

The following commands are executed **inside the VM**.

### 4.1 Confirm the Data Disk Device

```bash
lsblk
```

Expected output (the data disk is `nvme1n1`, 2TB):

```
NAME         MAJ:MIN RM   SIZE RO TYPE MOUNTPOINTS
nvme1n1      259:0    0     2T  0 disk
nvme0n1      259:1    0   500G  0 disk
├─nvme0n1p1  259:2    0   499G  0 part /
├─nvme0n1p14 259:3    0     4M  0 part
├─nvme0n1p15 259:4    0   106M  0 part /boot/efi
└─nvme0n1p16 259:5    0   913M  0 part /boot
```

You can also confirm the mapping via device-name:

```bash
ls -la /dev/disk/by-id/ | grep google-data-disk
```

### 4.2 Format as ext4

```bash
sudo mkfs.ext4 -m 0 -E lazy_itable_init=0,lazy_journal_init=0,discard /dev/nvme1n1
```

Parameter explanation:
- `-m 0`: do not reserve space for root (not needed for a data disk)
- `lazy_itable_init=0,lazy_journal_init=0`: complete all initialization at format time, to avoid background I/O affecting performance tests
- `discard`: enable TRIM support

### 4.3 Mount to /mnt/data

```bash
sudo mkdir -p /mnt/data
sudo mount -o discard,defaults /dev/nvme1n1 /mnt/data
sudo chmod a+w /mnt/data
```

### 4.4 Configure Automatic Mount on Boot

```bash
echo "/dev/disk/by-id/google-data-disk /mnt/data ext4 discard,defaults,nofail 0 2" | sudo tee -a /etc/fstab
```

Verify the mount:

```bash
df -h /mnt/data
```

Expected output:

```
Filesystem      Size  Used Avail Use% Mounted on
/dev/nvme1n1    2.0T   28K  2.0T   1% /mnt/data
```

---

## Step 5: Install fio and Run Benchmarks

### 5.1 Install fio

```bash
sudo apt-get update -qq && sudo apt-get install -y -qq fio
fio --version  # Expected: fio-3.36 or higher
```

### 5.2 Sequential Read Test (validate throughput)

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

### 5.3 Sequential Write Test (validate write throughput)

The write test must use a file path (fio does not allow writing directly to an already-mounted block device):

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

### 5.4 Random Read 4K Test (validate IOPS)

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

### 5.5 Random Write 4K Test (validate write IOPS)

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

### 5.6 Clean Up Test Files

```bash
rm -f /mnt/data/fio-test /mnt/data/fio-test-rw
```

### Key fio Parameter Explanation

| Parameter | Description |
|------|------|
| `--direct=1` | Bypass the OS page cache to test raw disk performance |
| `--ioengine=libaio` | Use the Linux async I/O engine (must use libaio rather than the default psync, otherwise high queue depth cannot be utilized) |
| `--iodepth=64/256` | I/O queue depth. Use 64 for sequential read/write, 256 for random I/O to fully saturate IOPS |
| `--numjobs=4` | Number of concurrent threads, simulating multiple processes reading/writing simultaneously |
| `--bs=1M/4k` | Block size. 1M for sequential throughput tests, 4K for IOPS tests |
| `--runtime=30` | Test duration of 30 seconds |
| `--group_reporting` | Aggregate statistics across all threads |

---

## Step 6: Create the GCS Bucket

Create a GCS bucket for storing large files such as model weights, datasets, or checkpoints.
It is recommended that the bucket region match the VM, to reduce cross-region transfer latency and cost.

```bash
gcloud storage buckets create gs://${BUCKET_NAME} \
    --project=${PROJECT_ID} \
    --location=${BUCKET_LOCATION} \
    --uniform-bucket-level-access
```

**Parameter explanation**:

| Parameter | Description |
|------|------|
| `--location` | Bucket region. GCS buckets are **region-level** (e.g., `us-central1`), not zone-level |
| `--uniform-bucket-level-access` | Enable uniform bucket-level access control (recommended), disabling ACLs and using only IAM policies to manage permissions |

> **Note**: GCS bucket names are globally unique; it is recommended to use a `{project-id}-{purpose}` naming format (e.g., `my-project-tpu-data`).

Verify the bucket creation:

```bash
gcloud storage buckets describe gs://${BUCKET_NAME} \
    --format="table(name,location,storageClass)"
```

Expected output:

```
NAME              LOCATION     STORAGE_CLASS
my-tpu-data       US-CENTRAL1  STANDARD
```

### Using GCS from the VM

Since the VM was created with `--scopes=cloud-platform`, you can directly use `gcloud storage` commands to access the bucket:

```bash
# Upload a file to GCS
gcloud storage cp /mnt/data/model-weights.safetensors gs://${BUCKET_NAME}/models/

# Download a file from GCS to the data disk
gcloud storage cp gs://${BUCKET_NAME}/models/model-weights.safetensors /mnt/data/

# Parallel download of large files (automatic sharding)
gcloud storage cp --recursive gs://${BUCKET_NAME}/models/ /mnt/data/models/
```

---

## Benchmark Results (Measured)

**Test environment**: TPU v7x-8 (tpu7x-standard-4t), 2TB Hyperdisk ML, provisioned 3000 MiB/s

| Test item | Measured value | Single-machine ceiling | Result |
|--------|--------|----------|----------|
| Sequential read throughput (1M bs) | **2,415 MiB/s** | 2,400 MiB/s | ✅ At ceiling |
| Sequential write throughput (1M bs) | **2,416 MiB/s** | 2,400 MiB/s | ✅ At ceiling |
| Random read IOPS (4K bs) | **48,300** | 38,400 | ✅ Exceeds nominal |
| Random write IOPS (4K bs) | **48,300** | 38,400 | ✅ Exceeds nominal |

### About the Single-Machine Throughput Ceiling

According to the [Hyperdisk ML official docs](https://docs.google.com/compute/docs/disks/hd-types/hyperdisk-ml#perf-limits),
**the Hyperdisk ML throughput ceiling per VM is 2,400 MiB/s** (consistent across all accelerator-optimized machine types).
The measured 2,415 MiB/s precisely hits this ceiling, independent of the following parameter adjustments:

- Block size: 1M / 4M / 16M
- Concurrent threads: 4 / 8 / 16 jobs
- Queue depth: 16 / 64 / 128 iodepth

This is a **per-VM architectural limit** of Hyperdisk ML, not a performance bottleneck of the disk itself.
For higher aggregate throughput, switch the disk to read-only mode and mount it on multiple VMs (each VM can reach 2,400 MiB/s).

> **A3-8g exception**: A3 Ultra (H100) can reach 4,000 MiB/s when mounted in read-only mode. The TPU v7x docs do not list this yet, but in testing it is consistent with the 2,400 MiB/s ceiling.

### Comparison with Hyperdisk Balanced

| Metric | Hyperdisk Balanced | Hyperdisk ML | Improvement |
|------|-------------------|-------------|------|
| Sequential read/write throughput | 1,200 MiB/s | 2,415 MiB/s | **2.0x** |
| Random IOPS | 40,000 | 48,300 | **1.2x** |
| Multi-machine read-only sharing | ❌ Not supported | ✅ Up to 2,500 machines | — |

---

## Hyperdisk ML Read/Write Mode Notes

Hyperdisk ML supports two access modes, used via a two-phase workflow:

### Phase 1: Single-Machine Read/Write (READ_WRITE_SINGLE)

The default mode after creation. Mounted on a single VM and used in read-write mode, for:
- Downloading model weights
- Preparing datasets
- Generating FP4/FP8 cache files

### Phase 2: Multi-Machine Read-Only (READ_ONLY_MANY)

Once data preparation is complete, you can switch to read-only mode to distribute to multiple VMs:

```bash
# 1. First detach the disk (must be unmounted on all VMs)
gcloud compute instances detach-disk ${INSTANCE_NAME} \
    --disk=${DATA_DISK_NAME} \
    --project=${PROJECT_ID} \
    --zone=${ZONE}

# 2. Switch access mode (irreversible operation!)
gcloud compute disks update ${DATA_DISK_NAME} \
    --project=${PROJECT_ID} \
    --zone=${ZONE} \
    --access-mode=READ_ONLY_MANY

# 3. Mount in read-only mode on multiple VMs
gcloud compute instances attach-disk ${INSTANCE_NAME} \
    --disk=${DATA_DISK_NAME} \
    --device-name=data-disk \
    --mode=ro \
    --project=${PROJECT_ID} \
    --zone=${ZONE}
```

**Important limitations**:

| Rule | Description |
|------|------|
| No multi-writer support | Multiple VMs cannot write simultaneously |
| Switch to read-only is irreversible | Once set to `READ_ONLY_MANY`, it **cannot be reverted** to read-write mode |
| Must detach before switching | Access mode cannot be changed while the disk is mounted |
| Max number of sharing instances | ≤256 GiB: 2,500 machines; 257 GiB-1 TiB: 600 machines; 1-2 TiB: 300 machines; 2-16 TiB: 128 machines; >16 TiB: 30 machines |
| Throughput requirement when sharing | When shared across more than 20 VMs, you must provision at least 100 MiB/s per VM |

---

## Appendix: Disk Type Selection Reference

TPU v7x supports only the following two Hyperdisk types:

| Disk type | Max single-machine throughput | IOPS configuration | Multi-machine sharing | Use case |
|----------|-------------|-----------|----------|----------|
| **Hyperdisk Balanced** | Measured ~1,200 MiB/s (IOPS + throughput must be set manually) | Manual | ❌ | General read/write, dev & test |
| **Hyperdisk ML** | **2,400 MiB/s** (per-VM hard ceiling) | Automatic = throughput × 16 | ✅ Up to 2,500 machines | Model weight distribution, inference clusters |

> **Unsupported disk types**: Hyperdisk Extreme and standard Persistent Disk (pd-ssd/pd-balanced) are both incompatible with the `tpu7x-standard-4t` machine type.

### Monthly Cost Estimate (2TB disk)

| Cost item | Hyperdisk Balanced (40K IOPS, 1200 MiB/s) | Hyperdisk ML (2500 MiB/s) |
|--------|-------------------------------------------|---------------------------|
| Storage | 2048 GiB × $0.06 = $123 | 2048 GiB × $0.08 = $164 |
| Throughput | 1200 MiB/s × $0.06 = $72 | 2500 MiB/s × $0.12 = $300 |
| IOPS | 40K × $0.006 = $240 | Included in throughput |
| **Total** | **~$435/month** | **~$464/month** |

> **Cost-saving tip**: in single-machine scenarios, setting provisioned-throughput to 2500 is enough to max out the 2,400 MiB/s ceiling,
> saving $60/month versus setting it to 3000 ((3000-2500) × $0.12), and $420/month versus setting it to 6000.
> Throughput can only be modified once every 6 hours, so it is best to set it correctly the first time.

---

*Last updated: 2026-04-28 | Tested by: Chris Yang*
