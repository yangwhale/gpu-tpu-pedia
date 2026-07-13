> 🌐 [中文](README.md) | **English**

# 0. Architecture Overview & Core Concepts + 1. Environment Setup

> This chapter is aimed at engineers encountering GB200/A4X for the first time. It walks through the hardware architecture, GCP machine-type differences, core concepts, and the creation of VPC/subnets/firewalls/Placement Policy.

## 0. Architecture Overview & Core Concepts

**Reading recommendation**: This chapter is aimed at engineers encountering GB200/A4X for the first time. It walks through the hardware architecture, GCP machine-type differences, and the core concepts that frequently come up during deployment. Experienced readers can skip to [Chapter 1](#1-environment-setup).

### 0.1 Hardware Architecture Comparison

The table below compares the key hardware differences across three generations of GPU platforms—these differences directly determine the deployment approach and achievable performance.

| Dimension | GB200 (Blackwell) | B200 (Blackwell) | H200 (Hopper) |
|------|-------------------|------------------|---------------|
| GPU architecture | Blackwell (sm_100) | Blackwell (sm_100) | Hopper (sm_90) |
| NVLink generation | 5th-gen NVLink | 5th-gen NVLink | 4th-gen NVLink |
| NVSwitch | 5th-gen · supports MNNVL cross-node NVLink | 5th-gen · supports MNNVL (limited) | 4th-gen · intra-node NVLink only |
| NVLink domain size | **72 GPUs** (NVL72: 18 nodes cross-node interconnect) | 16 GPUs (NVL16: 2 nodes) | 8 GPUs (intra-node only) |
| CPU | Grace ARM64 (aarch64) | x86_64 | x86_64 |
| GPU memory | HBM3e | HBM3e | HBM3e |

**Key difference**: The GB200 NVL72 domain interconnects the 72 GPUs across 18 nodes via NVSwitch, with a peak bidirectional bandwidth of approximately **840 GB/s** (per GPU pair). Cross-domain communication, on the other hand, can only go over RDMA, peaking at approximately 325 GB/s—a gap of roughly 2.5x. Therefore, placing all nodes of a single training job into the same NVL72 domain is the top priority for performance optimization.

### 0.2 GCP Machine-Type Comparison

GB200/B200/H200 correspond to different Accelerator-Optimized machine-type families on GCP, with significant differences in network topology and deployment approach.

| Dimension | A4X (GB200) | A4 (B200) | A3 Ultra (H200) |
|------|-------------|-----------|-----------------|
| Machine-type example | `a4x-highgpu-4g` | `a4-highgpu-8g` | `a3-ultragpu-8g` |
| GPUs / node | 4 (GB200) | 8 (B200) | 8 (H200) |
| CPU architecture | ARM64 (Grace) | x86_64 | x86_64 |
| RDMA NIC | 4 × CX-7 400Gbps | 8 × CX-7 400Gbps | 8 × CX-7 400Gbps |
| NVLink domain | **18 nodes / 72 GPUs** (NVL72) | 2 nodes / 16 GPUs (NVL16) | N/A (intra-node NVLink only) |
| MNNVL cross-node NVLink | Supported (requires IMEX daemon) | Supported (limited, 2 nodes) | Not supported |
| Placement Policy topology | `1x72` | `1x16` | N/A |
| Container image note | Requires ARM64 / aarch64 images | Standard x86_64 images | Standard x86_64 images |

**A4X ARM64 note**: GB200 uses a Grace ARM64 CPU, so all container images (including the NCCL plugin, NVSHMEM build artifacts, etc.) must be built for the aarch64 architecture. The GIB NCCL plugin image name must carry the `-arm64` suffix.

### 0.3 Core Concepts Quick Reference

| Concept | Description |
|------|------|
| **NVL72 Domain** | A cross-node NVLink interconnect domain formed by 18 A4X nodes (72 GB200 GPUs) via 5th-gen NVSwitch. Any GPU pair within the domain can communicate directly over NVLink, with a peak bidirectional bandwidth of approximately 840 GB/s. Communication outside the domain can only go over RDMA (~325 GB/s). |
| **MNNVL** (Multi-Node NVLink) | Cross-node NVLink communication capability. The core feature of the GB200 NVL72—it lets GPUs on different physical nodes within the same domain be interconnected over NVLink just as if they were intra-node. Requires the IMEX daemon to be running to enable. |
| **IMEX Daemon** (Inter-node Memory Exchange) | A user-space daemon (`nvidia-imex`) that manages cross-node NVLink channels. The NVLS transport's `cuMulticastCreate` relies on IMEX coordination. If IMEX is not running, NCCL silently falls back to RDMA (bandwidth drops from ~840 GB/s to ~326 GB/s) and reports CUDA error 801. |
| **ComputeDomain** | A Kubernetes DRA CRD (provided by the DRA GPU Driver) used to declaratively manage the IMEX daemon lifecycle within k8s. After a ComputeDomain is created, each node in the domain automatically starts an IMEX daemon pod, and a Pod gains MNNVL capability simply by referencing the ComputeDomain via a ResourceClaimTemplate. Each node can belong to only one ComputeDomain at a time. |
| **DRA** (Dynamic Resource Allocation) | Kubernetes' native API for requesting hardware resources (GA in k8s 1.33+, API version `resource.k8s.io/v1`). A Pod declaratively requests devices such as GPUs and RDMA NICs via a ResourceClaim, with allocation coordinated between the scheduler and the DRA driver. |
| **DRANET** | A DRA-based RDMA NIC allocation driver (v1.3.0). It publishes CX-7 RDMA NICs as DRA devices to a ResourceSlice; a Pod requests RDMA devices via a ResourceClaim, enabling precise, GPU-NIC PCIe topology-aware allocation. |
| **GIB** (GPUDirect InfiniBand) | An NCCL communication plugin provided by Google (v1.1.2) that encapsulates GPUDirect RDMA optimizations. It is injected into a Pod as an init container, mounted at `/usr/local/gib`, and automatically configures NCCL environment variables via `set_nccl_env.sh`. |
| **Placement Policy** | A GCP resource policy that ensures a group of VMs is allocated to the same physical NVL72 domain. For A4X it is created with `--collocation=COLLOCATED --gpu-topology=1x72`, with each NVL72 domain bound to one Placement Policy. In production, N domains require N Policies to be created. |

### 0.4 Three-Layer Coordination Mechanism (Placement Policy → ComputeDomain → IMEX)

NVL72 cross-node NVLink communication requires three layers of coordination, none of which can be omitted:

```
Placement Policy    →  Guarantees VMs land in the same NVSwitch physical domain
       ↓
ComputeDomain CRD   →  Discovers same-domain nodes by gpu.clique label, starts IMEX daemon
       ↓
IMEX daemon         →  18 nodes handshake with each other, establishing NVLink multicast channels
       ↓
NVLS transport ready →  NCCL / NVSHMEM / DeepEP communicate over NVSwitch (~900 GB/s)
```

**What happens if one of these layers is missing?**
- Missing Placement Policy → VMs scatter across different physical domains, NVSwitch is unreachable
- Missing ComputeDomain → the IMEX daemon does not start, the software layer is unreachable
- Missing IMEX → NCCL **does not error out**, silently degrading to RDMA (900→325 GB/s). DeepEP throughput is 8x worse

**MNNVL environment variable control**: `MNNVL_ENABLE=0` forces the use of RDMA; `MNNVL_ENABLE=2` fully enables NVLink.

### 0.5 NVLink Bandwidth: Intra-node = Cross-node

All GPU communication in the GB200 NVL72 **goes through NVSwitch**, with no distinction between intra-node and cross-node:
- Intra-node 4 GPUs → through the local node's NVSwitch → 900 GB/s unidirectional
- Cross-node GPUs → through NVSwitch optical-cable interconnect → likewise 900 GB/s unidirectional

The bandwidth is fully symmetric. Cross-node only adds a few microseconds of latency over intra-node (electro-optical conversion); there is no difference in bandwidth.

### 0.6 Multiple Teams Sharing One NVL72 Domain

**Security**: A single NVL72 domain can be shared among multiple Teams. GPU isolation is guaranteed by four layers:
1. **k8s device isolation** — the NVIDIA device plugin / DRA only exposes the allocated GPUs to a Pod
2. **CUDA device visibility** — a process can only `cudaSetDevice` to GPUs visible to it
3. **Peer access must be explicitly enabled** — cross-GPU memory access requires calling `cudaDeviceEnablePeerAccess`
4. **NCCL communicator isolation** — different Teams use different unique IDs, so communicators are naturally isolated

**Bandwidth does not contend**: NVSwitch is a full-bandwidth, non-blocking switch. Team A's communication does not consume Team B's bandwidth.

**ComputeDomain constraint**: only one ComputeDomain can exist in a domain at a time. When multiple Teams share a domain, the ComputeDomain must be managed centrally—they cannot each create their own. IMEX is a control plane (coordinating channel establishment), not a data plane (it does not forward traffic), so linking them together does not affect performance.

#### Concept Relationship Diagram

```
┌─────────────────── NVL72 Domain (18 nodes / 72 GPUs) ───────────────────┐
│                                                                        │
│  ┌─ Node 1 ──┐  ┌─ Node 2 ──┐       ┌─ Node 18 ─┐                    │
│  │ 4× GB200  │  │ 4× GB200  │  ...  │ 4× GB200  │                    │
│  │ 4× CX-7   │  │ 4× CX-7   │       │ 4× CX-7   │                    │
│  └─────┬─────┘  └─────┬─────┘       └─────┬─────┘                    │
│        │               │                    │                          │
│        └───── NVSwitch (MNNVL, ~840 GB/s) ──┘                         │
│                                                                        │
│  Placement Policy (1x72) ── ensures VMs land in the same physical domain │
│  ComputeDomain CRD ── manages the IMEX daemon, enables MNNVL           │
│  DRANET ── allocates CX-7 RDMA NICs to Pods                           │
│  GIB ── NCCL communication plugin, optimizes GPU RDMA                  │
└────────────────────────────────────────────────────────────────────────┘
        │ Cross-domain communication (RDMA only, ~325 GB/s)
┌───────┴──────── Another NVL72 Domain ──────────────────────────────────┐
│  ...                                                                   │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 1. Environment Setup

### Global Variables (adjust to your actual environment)

**CIDR planning note**: The subnets below are suitable for a cluster at the scale of **1800 GPUs (450 A4X VMs)**. The primary management subnet uses /21 (2046 IPs), while the remaining subnets use /22 (1022 IPs); all can accommodate 450+ A4X VMs with headroom to spare (Control Plane, regular VMs, future expansion).

**Readers should replace the following CIDR values with the ranges assigned by their internal network team**. Only the variable values need to be changed; all subsequent commands reference them automatically.

```bash
# ===== GCP project and region =====
PROJECT="your-gcp-project"                # ← replace with your GCP project ID
REGION="us-east5"                          # ← replace with your region
ZONE="us-east5-a"                          # ← replace with your zone
RESERVATION="your-gb200-reservation"       # ← replace with your GB200 reservation name

# ===== Machine and image =====
MACHINE_TYPE="a4x-highgpu-4g"
IMAGE="tlinux-server-4-gb200-v1"           # ← replace with your VM image name
IMAGE_PROJECT="$PROJECT"                   # ← project where the image resides (change if different from the VM project)

# ===== Network names =====
GVNIC_NET="a4x-gvnic-net-0"               # ← primary GVNIC management network (MTU 8896)
GVNIC_SUB="a4x-gvnic-sub-0"               # ← primary GVNIC subnet
GVNIC_NET_1="a4x-gvnic-net-1"             # ← secondary GVNIC network
GVNIC_SUB_1="a4x-gvnic-sub-1"             # ← secondary GVNIC subnet
RDMA_NET="a4x-rdma-net"                   # ← RDMA HPC network (requires network profile)

# ===== Network CIDRs (adjust to the ranges assigned by your internal network team) =====
GVNIC_CIDR="10.0.0.0/21"                  # primary management subnet: /21 = 2046 IPs
GVNIC_1_CIDR="10.0.8.0/22"                # secondary management subnet: /22 = 1022 IPs
RDMA_CIDR_0="10.0.16.0/22"                # RDMA subnet 0 (CX-7 NIC 0)
RDMA_CIDR_1="10.0.20.0/22"                # RDMA subnet 1 (CX-7 NIC 1)
RDMA_CIDR_2="10.0.24.0/22"                # RDMA subnet 2 (CX-7 NIC 2)
RDMA_CIDR_3="10.0.28.0/22"                # RDMA subnet 3 (CX-7 NIC 3)
SUBNET_SUPERNET="10.0.0.0/16"             # ← supernet of all subnets, used for firewall rules

# ===== NVL72 Domain configuration =====
NUM_DOMAINS=2                              # ← number of NVL72 Domains to deploy (can be set to 25+ in production)
NODES_PER_DOMAIN=18                        # fixed value for A4X NVL72, do not modify
PLACEMENT_PREFIX="a4x-nvl72-domain"        # Placement Policy name prefix

# ===== Cluster naming =====
CP_NAME="gb200-cp"                         # Control Plane node name
WORKER_PREFIX="gb200"                      # Worker node name prefix

# ===== Kubernetes =====
K8S_VERSION="1.34"
POD_CIDR="10.244.0.0/16"

# ===== Component versions =====
GIB_VERSION="v1.1.2"
DEVICE_PLUGIN_VERSION="v0.17.1"
DRANET_VERSION="v1.3.0"
DRA_GPU_DRIVER_VERSION="v25.12.0"
CALICO_VERSION="v3.29.3"
PYTORCH_IMAGE="nvcr.io/nvidia/pytorch:26.05-py3"
GIB_IMAGE="us-docker.pkg.dev/gce-ai-infra/gpudirect-gib/nccl-plugin-gib-diagnostic-arm64:${GIB_VERSION}"

# ===== Shared storage =====
GCSFUSE_BUCKET="your-training-bucket"      # ← replace with your GCS bucket name
LOCAL_SSD_MOUNT="/mnt/stateful_partition"   # Local SSD RAID0 mount point
```

### 1.1 Prerequisites

- The GCP project already has a GB200 A4X reservation (DENSE or CALENDAR type)
- The `gcloud` CLI is installed and authenticated locally
- The VPC/subnet/firewall creation in section 1.2 has been completed

### 1.2 Create VPC / Subnets / RDMA Network

#### 1.2.1 Primary GVNIC Management Network (MTU 8896)

**The MTU must be set to 8896** (Jumbo Frames). The default MTU of 1460/1500 causes Lustre throughput to drop by approximately 10%. The MTU of the MRDMA network is configured automatically by the Network Profile and does not need to be set manually.

```bash
# Create the primary GVNIC network (MTU 8896)
gcloud compute networks create $GVNIC_NET \
  --subnet-mode=custom --mtu=8896 --project=$PROJECT

gcloud compute networks subnets create $GVNIC_SUB \
  --network=$GVNIC_NET \
  --region=$REGION \
  --range=$GVNIC_CIDR \
  --project=$PROJECT
```

#### 1.2.2 Secondary GVNIC Network

```bash
gcloud compute networks create $GVNIC_NET_1 \
  --subnet-mode=custom --project=$PROJECT

gcloud compute networks subnets create $GVNIC_SUB_1 \
  --network=$GVNIC_NET_1 \
  --region=$REGION \
  --range=$GVNIC_1_CIDR \
  --project=$PROJECT
```

#### 1.2.3 RDMA HPC Network (requires Network Profile)

**Network Profile**: The RDMA network must be created using `--network-profile=${ZONE}-vpc-roce`. This Network Profile automatically configures the optimal MTU and RDMA parameters via GCP.

```bash
# Create the RDMA HPC network
gcloud beta compute networks create $RDMA_NET \
  --network-profile=${ZONE}-vpc-roce \
  --subnet-mode=custom \
  --project=$PROJECT

# Create 4 RDMA subnets (one for each CX-7 NIC)
RDMA_CIDRS=($RDMA_CIDR_0 $RDMA_CIDR_1 $RDMA_CIDR_2 $RDMA_CIDR_3)
for n in 0 1 2 3; do
  gcloud compute networks subnets create ${RDMA_NET}-sub-${n} \
    --network=$RDMA_NET \
    --region=$REGION \
    --range="${RDMA_CIDRS[$n]}" \
    --project=$PROJECT
done
```

#### 1.2.4 Firewall Rules

```bash
# SSH firewall (IAP source IP range)
gcloud compute firewall-rules create ${GVNIC_NET}-allow-iap-ssh \
  --network=$GVNIC_NET \
  --allow=tcp:22 \
  --source-ranges=35.235.240.0/20 \
  --project=$PROJECT

# Primary GVNIC internal communication (includes Pod CIDR — ComputeDomain daemon pods require cross-node communication)
gcloud compute firewall-rules create ${GVNIC_NET}-allow-internal \
  --network=$GVNIC_NET \
  --allow=tcp:0-65535,udp:0-65535,icmp \
  --source-ranges=${SUBNET_SUPERNET},${POD_CIDR} \
  --project=$PROJECT

# Secondary GVNIC internal communication
gcloud compute firewall-rules create ${GVNIC_NET_1}-allow-internal \
  --network=$GVNIC_NET_1 \
  --allow=tcp:0-65535,udp:0-65535,icmp \
  --source-ranges=${SUBNET_SUPERNET} \
  --project=$PROJECT
```

**Firewall note**: Some GCP organization policies automatically delete newly created firewall rules. If SSH or Pod communication fails, first check whether the firewall rules still exist. It is recommended to use IAP SSH (`--tunnel-through-iap`) to bypass firewall restrictions.

#### Network Layout Overview

| Network | Subnet | CIDR | Purpose |
|------|------|------|------|
| `$GVNIC_NET` | `$GVNIC_SUB` | `$GVNIC_CIDR` (/21, 2046 IPs) | Primary management network (SSH, k8s API, Pod CIDR) |
| `$GVNIC_NET_1` | `$GVNIC_SUB_1` | `$GVNIC_1_CIDR` (/22, 1022 IPs) | Secondary management network |
| `$RDMA_NET` | `${RDMA_NET}-sub-0..3` | `$RDMA_CIDR_0..3` (/22 each) | GPU RDMA compute network (one subnet per NIC) |

### 1.3 Check Reservation Status

```bash
gcloud compute reservations describe $RESERVATION \
  --zone=$ZONE --project=$PROJECT \
  --format="table(specificReservation.count, specificReservation.inUseCount)"
```

### 1.4 Placement Policy (required per Domain)

**Production scale**: Each NVL72 Domain (18 nodes / 72 GPUs) requires a dedicated Placement Policy. For example, 1800 GPUs = 25 Domains, requiring 25 Policies to be created. A Placement Policy is bound one-to-one with a Domain, ensuring that the VMs under that Policy are allocated to the same NVSwitch Domain.

```bash
# Create a collocated placement policy for each Domain
for d in $(seq 1 $NUM_DOMAINS); do
  gcloud beta compute resource-policies create group-placement \
    ${PLACEMENT_PREFIX}-${d} \
    --collocation=COLLOCATED \
    --gpu-topology=1x72 \
    --project=$PROJECT --region=$REGION
done
```

#### Placement Policy FAQ

**Q: Can `--gpu-topology=1x72` only be set to 72?**
A: The value corresponds to the machine type. A4X (GB200) uses `1x72` (18 nodes, 72 GPUs), while A4 (B200) uses `1x16` (2 nodes, 16 GPUs). The value defines the minimum topology constraint—"give me a fully interconnected domain at least this large."

**Q: Must I create all 18 VMs at once to use it?**
A: No. It is perfectly fine to use only 2 or 4 VMs in a `1x72` domain. The unused slots are left empty, and machines can be added later. The ComputeDomain generates IMEX configuration for all 18 physical slots in the domain; the undeployed slots produce connection-retry logs (harmless and can be ignored).

**Q: Will multiple Placement Policies be assigned to the same physical domain?**
A: They might. The GCP scheduler allocates automatically based on available slots. If two Policies land in the same domain, the `gpu.clique` label of all VMs will be identical. In that case, the ComputeDomain selects nodes by clique, and only one ComputeDomain instance can exist in the domain—users of the multiple Policies need to coordinate on sharing it.

**Q: If someone else's VMs are already in the domain, will adding my machines cause a conflict?**
A: No. As long as there are enough free slots in the domain, your VMs will be created normally. GPU communication is isolated via NCCL communicators, and NVSwitch is full-bandwidth non-blocking, so they do not interfere with each other. However, if the other party has already created a ComputeDomain, you need to reuse it rather than create another one.

### 1.5 Create Additional Firewall Rules (optional)

```bash
# SSH firewall (IAP source IP range)
gcloud compute firewall-rules create allow-ssh-iap-k8s1341 \
  --network=$GVNIC_NET \
  --allow=tcp:22 \
  --source-ranges=35.235.240.0/20 \
  --project=$PROJECT

# Cluster internal communication (includes Pod CIDR)
gcloud compute firewall-rules create allow-internal-k8s1341 \
  --network=$GVNIC_NET \
  --allow=tcp,udp,icmp \
  --source-ranges=${SUBNET_SUPERNET},${POD_CIDR} \
  --project=$PROJECT

# Note: Calico uses VXLAN mode (UDP 4789), which is already covered by the allow-internal rule above
```

---

## Known Issues and Considerations

### TLinux 4 Characteristics

| Issue | Description |
|------|------|
| Boot disk device name is not fixed | The NVMe device numbering may change between boots; use `findmnt` to look it up dynamically |
| Boot disk partitioning | After the 50GB OS image is installed onto a 1TB disk, it does not auto-expand. The startup script uses `sgdisk` to create a 4-partition layout |
| Missing base components | sudo, gcloud CLI, perftest, and the CUDA Toolkit are not pre-installed and must be installed manually |
| Docker CE repo | The RHEL 9 baseurl must be hardcoded; TLinux 4 is not automatically recognized by Docker officially |

### VPC Management NIC MTU

The MTU of the GVNIC management network must be set to **8896** (Jumbo Frames). The default MTU of 1460/1500 causes Lustre throughput to drop by approximately 10%.

### NIC Bonding Not Applicable

GCP A4X instances **do not support** traditional Linux bonding (bond0/bond1, etc.), and no configuration is required.

| Type | Device name | Count | Purpose | Redundancy mechanism |
|------|--------|------|------|----------|
| **GVNIC** | eth0, eth1 | 2 | Management network | GCP Andromeda SDN provides HA at the virtualization layer |
| **MRDMA** | mlx5_0..mlx5_3 | 4 | GPU RDMA compute network | SR-IOV VFs are managed by GCP RDMA infrastructure and allocated via DRANET DRA |

### nvidia-fabricmanager Not Applicable

The NVSwitch on GCP A4X is managed by the **virtualization layer (hypervisor)**; the Guest VM neither needs to nor should run the Fabric Manager service. The NVSwitch is managed by a Service VM at the GCP hypervisor layer.

### nvidia_peermem Not Applicable

GPU Direct RDMA on GCP A4X is implemented via the **GIB plugin** (`nccl-plugin-gib`) and does not use the traditional `nvidia_peermem` kernel module.

### Hybrid-Cloud Scenario Considerations (self-hosted CP + GCP Workers)

If the Control Plane is deployed in an on-premises IDC while the Workers are on GCP, note the following:

- **API Server network connectivity**: requires Cloud VPN or Cloud Interconnect
- **DRA controller cross-network authentication**: the default kubeadm ClusterRole lacks DRA permissions (see [03-gpu-stack](../03-gpu-stack/) section 3.5)
- **Custom admission controller compatibility**: ensure they do not intercept DRA-related APIs (`resource.k8s.io/v1`)
- **Calico VXLAN MTU**: subtract 50 bytes of VXLAN overhead over the dedicated link

---

## VM Delivery Acceptance Checklist (Appendix A)

### System and Security Configuration

| Adaptation item | Customer standard | GCP adaptation status | GCP notes |
|--------|----------|-------------|----------|
| Operating system | TencentOS Server 4.0 ARM | Met | `tlinux-server-4-gb200-v1` custom image |
| SELinux | Disabled | Met | Set by startup script |
| Firewalld | Disabled | Met | Disabled by startup script |
| SSH port | Only allow 56000 | Met | startup script modifies sshd_config |

### Network and NIC Configuration

| Adaptation item | Customer standard | GCP adaptation status | GCP notes |
|--------|----------|-------------|----------|
| Management NIC | bond1 | Not applicable | GVNIC managed by Andromeda SDN |
| RDMA NIC | bond2-bond5 | Alternative approach | 4 MRDMA NICs, allocated via DRANET DRA |
| NIC MTU | >= 4200 | Met | GCP VPC MTU=8896 |
| NIC driver | >= 5.8 | Met | mlx5_core 25.10-1.2.2 |

### GPU Driver and Services

| Adaptation item | Customer standard | GCP adaptation status | GCP notes |
|--------|----------|-------------|----------|
| GPU driver version | >= 535.247.01 | Met | NVIDIA 580.126.20 (R580 Open) |
| nvidia_peermem | Loaded | Not applicable | Replaced by the GIB plugin |
| nvidia-fabricmanager | active | Not applicable | NVSwitch managed by the virtualization layer |
| NCCL/GIB | Includes GIB | Met | GIB v1.1.2 injected via Pod initContainer |

### Storage and Disks

| Adaptation item | Customer standard | GCP adaptation status | GCP notes |
|--------|----------|-------------|----------|
| Boot disk | 1TB | Met | Hyperdisk Balanced 1TB |
| sda1 (/) | 21.5G | Met | Partitioned by startup script |
| sda2 (/boot/efi) | 512M | Met | TLinux image default |
| sda3 (/usr/local) | 20G | Met | Separate partition by startup script |
| sda4 (/data) | ~950G | Met | Remaining space by startup script |
| Local SSD | — | Met | NVMe RAID0 mounted at /mnt/stateful_partition |
| Lustre 1PB | Delivered with GB200 | Met | GCP Parallel Store (Lustre) CSI driver |

### Startup Scripts (Appendix B.5)

Two versions of the initialization script have been prepared per customer requirements:

| Script | Purpose | Description |
|------|------|------|
| `tlinux4-customer-init.sh` | Customer delivery | Does not install docker/kubelet; SSH 56000; SELinux disabled; GDRCopy v2.6; boot disk partitioning |
| `tlinux4-internal-init.sh` | Internal testing | Includes containerd + NVIDIA Container Toolkit + kubeadm/kubelet/kubectl k8s 1.34 |

**Note**: The items marked "Not applicable" in the VM delivery acceptance checklist (bond, nvidia_peermem, nvidia-fabricmanager) are standards for physical servers / self-hosted IDC environments. On GCP A4X, the corresponding functionality is implemented through GCP-native mechanisms (Andromeda SDN, the GIB plugin, virtualization-layer NVSwitch management)—functionally equivalent but implemented differently.

### Short Hostname (Appendix B.5 — required for Megatron)

Megatron-LM uses Gloo for inter-process communication. If the Pod hostname is too long, it triggers a `File name too long` error. Solution: set the `hostname` field in the Pod spec to a short name (e.g., `mega-h1`).
