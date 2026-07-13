> 🌐 [中文](README.md) | **English**

# 2. Creating a k8s 1.34.1 Cluster

**Key changes in k8s 1.34**: DRA (Dynamic Resource Allocation) is **GA** in k8s 1.34, enabled by default, no feature gate required. The ResourceClaimTemplate/DeviceClass API version is `resource.k8s.io/v1` (not v1beta2).

## 2.1 Control Plane Nodes

CP nodes use x86_64 VMs (no GPU required), connected only to the primary GVNIC management network.

> **Machine type recommendation**: `n4-standard-8` (8 cores, 32GB) or higher. `e2-standard-4` works but is on the weak side—etcd and the API server have CPU and IOPS requirements under large clusters. A disk of 200GB+ is recommended.

> **Network reuse**: If you already have a management network (e.g. `chrisya-gvnic-net-0`), you can reuse it directly without creating a new VPC. Just make sure the target region has a subnet. The benefit of reusing an existing network is direct SSH—any VM within the same VPC can SSH directly via internal IP.

```bash
# Option A: Use an existing network (recommended, direct SSH)
gcloud compute instances create $CP_NAME \
  --project=$PROJECT --zone=$ZONE \
  --machine-type=n4-standard-8 \
  --image-family=rocky-linux-9 --image-project=rocky-linux-cloud \
  --boot-disk-size=200GB \
  --network-interface=network=$GVNIC_NET,subnet=$GVNIC_SUB \
  --scopes=cloud-platform

# Option B: Create a standalone network (requires IAP tunnel SSH)
gcloud compute instances create $CP_NAME \
  --project=$PROJECT --zone=$ZONE \
  --machine-type=n4-standard-8 \
  --image-family=rocky-linux-9 --image-project=rocky-linux-cloud \
  --boot-disk-size=200GB \
  --network-interface=network=$GVNIC_NET,subnet=$GVNIC_SUB \
  --metadata-from-file=startup-script=scripts/kubeadm-control-plane-k8s134.sh \
  --scopes=cloud-platform
```

### SSH to the CP Node

```bash
# Option A: Direct internal connection within the same VPC (inject SSH key first)
# Inject via --metadata at creation time, or after creation:
gcloud compute instances add-metadata $CP_NAME \
  --zone=$ZONE --project=$PROJECT \
  --metadata=ssh-keys="$USER:$(cat ~/.ssh/id_ed25519.pub)"
ssh $USER@<CP_INTERNAL_IP>

# Option B: IAP tunnel (use when on a different network)
gcloud compute ssh $CP_NAME --zone=$ZONE --project=$PROJECT --tunnel-through-iap
```

### Manual CP Installation Steps

If you did not use a startup script, SSH to the CP and manually perform the following 7 steps:

```bash
# Step 1: Disable swap and SELinux
sudo swapoff -a
sudo sed -i '/swap/d' /etc/fstab
sudo setenforce 0
sudo sed -i 's/^SELINUX=enforcing$/SELINUX=permissive/' /etc/selinux/config

# Step 2: Load kernel modules + sysctl
sudo tee /etc/modules-load.d/k8s.conf <<EOF
overlay
br_netfilter
EOF
sudo modprobe overlay && sudo modprobe br_netfilter

sudo tee /etc/sysctl.d/k8s.conf <<EOF
net.bridge.bridge-nf-call-iptables  = 1
net.bridge.bridge-nf-call-ip6tables = 1
net.ipv4.ip_forward                 = 1
EOF
sudo sysctl --system

# Step 3: Install containerd (Docker CE repo)
sudo dnf config-manager --add-repo https://download.docker.com/linux/centos/docker-ce.repo
sudo dnf install -y containerd.io
sudo mkdir -p /etc/containerd
sudo sh -c 'containerd config default > /etc/containerd/config.toml'
sudo sed -i 's/SystemdCgroup = false/SystemdCgroup = true/' /etc/containerd/config.toml
sudo systemctl enable --now containerd

# Step 4: Install kubeadm/kubelet/kubectl
#   Note: kubectl must be installed from pkgs.k8s.io, do not use the one bundled with google-cloud-sdk
#   (the google-cloud-sdk kubectl version does not match k8s 1.34)
sudo tee /etc/yum.repos.d/kubernetes.repo <<EOF
[kubernetes]
name=Kubernetes
baseurl=https://pkgs.k8s.io/core:/stable:/v1.34/rpm/
enabled=1
gpgcheck=1
gpgkey=https://pkgs.k8s.io/core:/stable:/v1.34/rpm/repodata/repomd.xml.key
exclude=kubectl
EOF
sudo dnf install -y kubelet kubeadm --disableexcludes=kubernetes
# Install kubectl separately from the k8s repo (avoids the google-cloud-sdk version)
sudo dnf install -y --repo=kubernetes kubectl
sudo systemctl enable --now kubelet

# Step 5: kubeadm init
#   --node-name specifies a clean hostname, avoiding the FQDN
sudo kubeadm init \
  --pod-network-cidr=10.244.0.0/16 \
  --node-name=$CP_NAME

# Step 6: Configure kubeconfig
mkdir -p $HOME/.kube
sudo cp -i /etc/kubernetes/admin.conf $HOME/.kube/config
sudo chown $(id -u):$(id -g) $HOME/.kube/config

# Step 7: Install Calico v3.29.3 CNI
kubectl create -f https://raw.githubusercontent.com/projectcalico/calico/v3.29.3/manifests/tigera-operator.yaml
# Wait for tigera-operator to be ready
sleep 10
cat <<EOF | kubectl create -f -
apiVersion: operator.tigera.io/v1
kind: Installation
metadata:
  name: default
spec:
  calicoNetwork:
    nodeAddressAutodetectionV4:
      cidrs:
      - "$MGMT_SUBNET"    # Management subnet CIDR (e.g. 10.14.0.0/24)
    ipPools:
    - blockSize: 26
      cidr: 10.244.0.0/16
      encapsulation: VXLANCrossSubnet
      natOutgoing: Enabled
      nodeSelector: all()
EOF

# Verify
kubectl get nodes  # The CP node should show Ready
kubectl get pods -n calico-system  # Calico pods should gradually become Running
# All calico-node must be 1/1 Ready — 0/1 indicates BGP peering failure
```

> **Calico multi-NIC pitfall (critical)**: A4X Workers have 6 NICs (2 GVNIC + 4 MRDMA). Calico's default `firstFound: true` IP autodetection will select an RDMA NIC (e.g. 10.10.28.x) instead of the management GVNIC (e.g. 10.14.0.x), causing BIRD BGP peering to fail → calico-node on the CP stays permanently 0/1 Not Ready → Pod DNS is completely down (CoreDNS runs on the CP, and the VXLAN tunnel is broken).
>
> Fix: You must set `nodeAddressAutodetectionV4.cidrs` to the management subnet CIDR in the Installation CRD. If you forgot to set it or set it incorrectly, patch it afterward as follows:
> ```bash
> kubectl patch installation default --type=json -p '[
>   {"op": "replace", "path": "/spec/calicoNetwork/nodeAddressAutodetectionV4",
>    "value": {"cidrs": ["10.14.0.0/24"]}}
> ]'
> kubectl delete pods -n calico-system -l k8s-app=calico-node  # Restart calico-node
> ```

> **kubectl version pitfall**: On Rocky Linux, if the google-cloud-sdk repo is configured, `dnf install kubectl` will prefer the google-cloud-sdk version (e.g. 574.0.0) over the k8s 1.34.x version. It is recommended to add `exclude=kubectl` to kubernetes.repo to avoid the conflict, then install it separately with `--repo=kubernetes`.

### Retrieving Join Information

```bash
# The last few lines of the kubeadm init output contain the join command, formatted like:
# kubeadm join <CP_IP>:6443 --token <TOKEN> --discovery-token-ca-cert-hash sha256:<HASH>

# You can also extract it afterward:
CP_IP=$(hostname -I | awk '{print $1}')
JOIN_TOKEN=$(kubeadm token list -o jsonpath='{.token}' | head -1)
JOIN_HASH=$(openssl x509 -pubkey -in /etc/kubernetes/pki/ca.crt | \
  openssl rsa -pubin -outform der 2>/dev/null | sha256sum | awk '{print $1}')
echo "CP_IP=$CP_IP JOIN_TOKEN=$JOIN_TOKEN JOIN_HASH=$JOIN_HASH"

# The token is valid for 24h; regenerate it after expiry:
kubeadm token create --print-join-command
```

## 2.2 Placement Policy and Worker Node VM Creation

**The relationship between Domain and Placement Policy**:

- Each NVL72 Domain = 18 nodes × 4 GPU = 72 GPU, determined by the physical NVSwitch topology
- Each Domain requires its own `Placement Policy` (created in [01-environment-setup](../01-environment-setup/) section 1.4); the `--resource-policies` parameter determines which Domain the VM is allocated to
- Production environment (e.g. 1800 GPU = 25 Domains): batch-create Workers for each Domain, using the corresponding Domain's Placement Policy for each batch
- Use GA `gcloud compute` (not alpha/beta), and do not add `--local-ssd` (A4X automatically mounts 12TB NVMe)
- `no-address` on MRDMA — the network profile does not allow MRDMA interfaces to have an AccessConfig
- **You do not need to create all 18 at once**—creating just 2~4 for testing is perfectly fine; fill the empty slots later

### Viewing and Reusing Existing Placement Policies

Before creating a new Policy, first review the existing Policies in the project and their usage:

```bash
# List all Placement Policies
gcloud beta compute resource-policies list \
  --project=$PROJECT --filter="region~$REGION" \
  --format="table(name,region.basename(),status)"

# View the Policy used by each A4X VM (find Policies with many free slots to reuse)
gcloud compute instances list \
  --project=$PROJECT --zones=$ZONE \
  --filter="machineType~a4x" \
  --format="table(name,status,resourcePolicies.basename())"
```

If a Policy has only 2~4 VMs (14~16 free slots), you can reuse it directly—your VM will join the same physical domain. Note that a single domain can only have one ComputeDomain (see [01-environment-setup](../01-environment-setup/) section 0.6 for details).

### Worker Image Selection

| Image | Project | Preinstalled Contents | Use Case |
|------|------|----------|----------|
| `rocky-linux-9-optimized-gcp-nvidia-580-arm64` | `rocky-linux-accelerator-cloud` | NVIDIA 580 driver + RDMA NIC driver | **Recommended**: GCP official public image; start installing containerd/kubelet/IMEX from here |
| `rocky-linux-9-optimized-gcp-nvidia-latest-arm64` | `rocky-linux-accelerator-cloud` | Latest NVIDIA driver | Use when you need the latest driver features |
| `tlinux-server-4-gb200-v4` | Project-private | NVIDIA driver + TLinux 4 customized OS | Customer-customized environment (paired with the `tlinux4-k8s134-worker.sh` startup script) |
| `rocky-linux-9-arm64` | `rocky-linux-cloud` | Clean OS, no NVIDIA driver | Use when you need to install the driver fully yourself (not recommended, GB200 driver installation is complex) |

> **Verified in practice** (2026-06-27): After creating an A4X VM with the `rocky-linux-9-optimized-gcp-nvidia-580-arm64` image, the 4 GB200 GPUs (189GB HBM each) and 4 RDMA NICs come up automatically, and `nvidia-smi` works normally.

### NIC Configuration Requirements

> **Important**: A4X NIC configuration has strict ordering requirements—**the first 2 must be GVNIC, and the last 4 must be MRDMA**. If this requirement is not met, `gcloud` will error out directly.
>
> Error example: providing only 1 GVNIC + 4 MRDMA → `On a4x-highgpu-4g, the first NIC (if present) and the second NIC (if present) must be of type GVNIC. These must be followed by 0 or 4 MRDMA NICs.`
>
> Therefore you must prepare **2 GVNIC networks** (primary management network + secondary network) and **1 RDMA network** (4 subnets).

### Batch-Creating Workers (Each Domain Uses Its Corresponding Placement Policy)

**Production example**: 25 Domains × 18 nodes/Domain = 450 Worker VMs.

```bash
# RDMA subnet names (determined by the naming used when creating 01-environment-setup)
RDMA_SUB_0="${RDMA_NET}-sub-0"
RDMA_SUB_1="${RDMA_NET}-sub-1"
RDMA_SUB_2="${RDMA_NET}-sub-2"
RDMA_SUB_3="${RDMA_NET}-sub-3"

# Loop to create Worker VMs for all Domains
for d in $(seq 1 $NUM_DOMAINS); do
  echo "=== Creating workers for Domain ${d} ==="
  for i in $(seq 0 $((NODES_PER_DOMAIN - 1))); do
    gcloud compute instances create ${WORKER_PREFIX}-d${d}-w${i} \
      --project=$PROJECT --zone=$ZONE \
      --machine-type=$MACHINE_TYPE \
      --image-family=rocky-linux-9-optimized-gcp-nvidia-580-arm64 \
      --image-project=rocky-linux-accelerator-cloud \
      --boot-disk-size=500GB --boot-disk-type=hyperdisk-balanced \
      --scopes=cloud-platform \
      --reservation-affinity=specific --reservation=$RESERVATION \
      --maintenance-policy=TERMINATE \
      --restart-on-failure \
      --resource-policies=${PLACEMENT_PREFIX}-${d} \
      --network-interface=nic-type=GVNIC,network=$GVNIC_NET,subnet=$GVNIC_SUB \
      --network-interface=nic-type=GVNIC,network=$GVNIC_NET_1,subnet=$GVNIC_SUB_1,no-address \
      --network-interface=nic-type=MRDMA,network=$RDMA_NET,subnet=$RDMA_SUB_0,no-address \
      --network-interface=nic-type=MRDMA,network=$RDMA_NET,subnet=$RDMA_SUB_1,no-address \
      --network-interface=nic-type=MRDMA,network=$RDMA_NET,subnet=$RDMA_SUB_2,no-address \
      --network-interface=nic-type=MRDMA,network=$RDMA_NET,subnet=$RDMA_SUB_3,no-address \
      --metadata=ssh-keys="$USER:$(cat ~/.ssh/id_ed25519.pub)" &
  done
  wait  # Wait for all Workers of this Domain to finish being created
done
```

> **TLinux image users**: If using a `tlinux-server-4-gb200-v*` image, you can achieve automatic installation and join via `--metadata-from-file=startup-script=scripts/tlinux4-k8s134-worker.sh` and `--metadata=cp-ip=$CP_IP,join-token=$JOIN_TOKEN,join-hash=$JOIN_HASH`.

**Naming convention**: Worker names follow the format `${WORKER_PREFIX}-d${DOMAIN}-w${INDEX}`, e.g. `gb200-d1-w0` through `gb200-d1-w17` are the 18 nodes of Domain 1.

## 2.3 Worker Hardware Tuning + Joining k8s

When Workers use the GCP official `rocky-linux-9-optimized-gcp-nvidia-580-arm64` image, the GPU driver is preinstalled, but you need to manually complete hardware tuning and k8s software installation. This is done in two phases (with one reboot in between).

### Phase 1: Hardware Tuning (Reboot Required)

```bash
# SSH to the Worker
ssh $USER@<WORKER_IP>

# === IMEX initramfs configuration ===
# Enable the IMEX channel device file (a prerequisite for cross-node NVLink)
echo "options nvidia NVreg_CreateImexChannel0=1" | sudo tee /etc/modprobe.d/nvidia.conf
sudo dracut --force --add-drivers "gve"

# === Kernel modules (k8s prerequisite) ===
sudo tee /etc/modules-load.d/k8s.conf <<EOF
overlay
br_netfilter
tcp_bbr
EOF
sudo modprobe overlay && sudo modprobe br_netfilter && sudo modprobe tcp_bbr

# === sysctl (k8s + Grace CPU tuning) ===
sudo tee /etc/sysctl.d/k8s.conf <<EOF
net.bridge.bridge-nf-call-iptables  = 1
net.bridge.bridge-nf-call-ip6tables = 1
net.ipv4.ip_forward                 = 1
EOF

sudo tee /etc/sysctl.d/90-grace-gb200.conf <<EOF
kernel.numa_balancing = 0
net.ipv4.tcp_congestion_control = bbr
net.core.default_qdisc = fq
net.core.rmem_max = 16777216
net.core.wmem_max = 16777216
net.ipv4.tcp_rmem = 4096 87380 16777216
net.ipv4.tcp_wmem = 4096 65536 16777216
vm.swappiness = 0
vm.zone_reclaim_mode = 0
EOF
sudo sysctl --system

# === GPU persist mode + disable unneeded services ===
sudo nvidia-smi -pm 1
sudo systemctl disable --now irqbalance 2>/dev/null || true

# === Reboot (a reboot is needed for the IMEX initramfs to take effect) ===
sudo reboot
```

> **GB200 reboot time**: GB200 A4X reboots much slower than an ordinary VM—GPU unload + NVSwitch release + UEFI re-initialization take about **5-8 minutes** in total. The NVIDIA Persistence Daemon's stop job may hang for 3 minutes (this is normal; do not force-kill it).

> **BBR kernel module**: The sysctl setting `tcp_congestion_control = bbr` requires the `tcp_bbr` kernel module to be loaded first, otherwise it will fall back to `cubic` after a reboot. You must add `tcp_bbr` to `/etc/modules-load.d/`.

### Phase 1 Verification (After Reboot)

```bash
ssh $USER@<WORKER_IP>

# The IMEX channel should exist
ls /dev/nvidia-caps-imex-channels/
# Expected output: channel0

# sysctl should be persistent
echo "numa=$(cat /proc/sys/kernel/numa_balancing) tcp=$(cat /proc/sys/net/ipv4/tcp_congestion_control)"
# Expected output: numa=0 tcp=bbr

# GPU
nvidia-smi --query-gpu=name,persistence_mode --format=csv,noheader | head -1
# Expected output: NVIDIA GB200, Enabled
```

### Phase 2: Install containerd + kubelet + Join the Cluster

```bash
# === containerd ===
sudo dnf config-manager --add-repo https://download.docker.com/linux/centos/docker-ce.repo
sudo dnf install -y containerd.io
sudo mkdir -p /etc/containerd
sudo sh -c 'containerd config default > /etc/containerd/config.toml'
sudo sed -i 's/SystemdCgroup = false/SystemdCgroup = true/' /etc/containerd/config.toml
sudo systemctl enable --now containerd

# === NVIDIA Container Toolkit ===
curl -s -L https://nvidia.github.io/libnvidia-container/stable/rpm/nvidia-container-toolkit.repo | \
  sudo tee /etc/yum.repos.d/nvidia-container-toolkit.repo
sudo dnf install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=containerd --set-as-default
sudo systemctl restart containerd

# === kubelet / kubeadm ===
sudo tee /etc/yum.repos.d/kubernetes.repo <<EOF
[kubernetes]
name=Kubernetes
baseurl=https://pkgs.k8s.io/core:/stable:/v1.34/rpm/
enabled=1
gpgcheck=1
gpgkey=https://pkgs.k8s.io/core:/stable:/v1.34/rpm/repodata/repomd.xml.key
EOF
sudo dnf install -y kubelet kubeadm --disableexcludes=kubernetes
# Install kubectl separately: temporarily remove the exclude, install from the specified repo, then add it back
sudo sed -i '/^exclude/d' /etc/yum.repos.d/kubernetes.repo
sudo dnf install -y --repo=kubernetes kubectl
echo "exclude=kubectl" | sudo tee -a /etc/yum.repos.d/kubernetes.repo
sudo systemctl enable kubelet

# === kubeadm join ===
# Use the join command from the CP node (obtained from the CP)
sudo kubeadm join <CP_IP>:6443 \
  --token <JOIN_TOKEN> \
  --discovery-token-ca-cert-hash sha256:<JOIN_HASH> \
  --node-name $(hostname)
```

### Phase 2 Verification (On the CP Node)

```bash
kubectl get nodes -o wide
# You should see the Worker node Ready
# Note: After a Worker first joins, kubelet needs 30-60 seconds to pull the CNI plugin before it becomes Ready
```

> **kubectl version pitfall**: On Rocky Linux, if the google-cloud-sdk repo is configured, `dnf install kubectl` will install the gcloud version (e.g. 574.0.0). You must install it separately from the kubernetes repo—add `exclude=kubectl` in kubernetes.repo, then run `dnf install -y --repo=kubernetes kubectl`.

## 2.4 Node Labels

### Method A: Manual Labeling (Small-Scale / Validation Environment)

```bash
# Label Worker nodes by Domain
for d in $(seq 1 $NUM_DOMAINS); do
  for i in $(seq 0 $((NODES_PER_DOMAIN - 1))); do
    kubectl label node ${WORKER_PREFIX}-d${d}-w${i} \
      nvl72-domain=domain-${d} --overwrite
  done
done

# Verify
kubectl get nodes -L nvl72-domain
```

### Method B: Automatic Registration via Startup Script (Production / Large-Scale Environment, Recommended)

GCP VMs can obtain physical topology information via the Metadata Server, without IAM permissions. VMs within the same NVL72 domain return identical topology hash values.

```bash
# In the Worker startup script (before kubeadm join), add:

# 1. Obtain the physical topology hash from the Metadata Server (VMs in the same domain return the same value)
TOPO_HASH=$(curl -sf -H "Metadata-Flavor: Google" \
  "http://metadata.google.internal/computeMetadata/v1/instance/attributes/physical_host_topology" \
  | md5sum | cut -c1-8)  # Take the first 8 characters as the domain identifier

# 2. Configure kubelet startup parameters to automatically attach the topology label
mkdir -p /etc/systemd/system/kubelet.service.d
cat > /etc/default/kubelet <<KUBELET_EOF
KUBELET_EXTRA_ARGS=--node-labels=nvl72-domain=domain-${TOPO_HASH}
KUBELET_EOF
systemctl daemon-reload

# 3. The node automatically carries the label at kubeadm join time
kubeadm join ${CP_IP}:6443 \
  --token "${JOIN_TOKEN}" \
  --discovery-token-ca-cert-hash "sha256:${JOIN_HASH}" \
  --node-name "${NODE_NAME}" \
  --ignore-preflight-errors=Hostname
```

**Two APIs for topology discovery**:

| Method | Endpoint | Permission Requirement | Return Value |
|------|------|----------|--------|
| Metadata Server (recommended) | `http://metadata.google.internal/.../physical_host_topology` | None (accessed directly from within the VM) | Topology hash string |
| Compute API | `gcloud compute instances describe --format='value(resourceStatus.physicalHostTopology.subblock)'` | Requires `compute.instances.get` IAM permission | Structured subblock identifier |

**Note**: The Metadata Server returns a hash value (not human-readable but stable); VMs in the same domain are guaranteed to return the same value. The Compute API returns a human-readable subblock name, but requires additional IAM configuration.
