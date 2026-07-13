> 🌐 [中文](README.md) | **English**

# 3. Install GPU Stack + ComputeDomain + 4. Shared Storage

**k8s 1.34 DRA status**: DRA is **GA (Generally Available)** in k8s 1.34, enabled by default, no feature gate required. The ResourceClaimTemplate/DeviceClass API version is `resource.k8s.io/v1`.

## 3.1 Install nvidia-device-plugin

Use the nvidia-device-plugin DaemonSet to expose GPUs to k8s as the `nvidia.com/gpu` resource.

```bash
# Install Helm (if not already installed)
curl -fsSL https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3 | bash

# Install nvidia-device-plugin
helm repo add nvdp https://nvidia.github.io/k8s-device-plugin
helm repo update
helm install nvidia-device-plugin nvdp/nvidia-device-plugin \
  --namespace kube-system \
  --wait --timeout 300s
```

> **Node label requirement**: By default, the nvidia-device-plugin DaemonSet uses nodeAffinity to match `feature.node.kubernetes.io/pci-10de.present=true` (the NVIDIA PCI vendor ID). If Node Feature Discovery (NFD) is not installed, you must manually apply this label to the GPU Workers:
> ```bash
> kubectl label node <WORKER_NAME> feature.node.kubernetes.io/pci-10de.present=true
> ```
> Without this label, the DaemonSet's DESIRED=0 and GPUs will not be discovered.

```bash
# Verify GPUs are visible
kubectl get nodes -o custom-columns="NAME:.metadata.name,GPU:.status.allocatable.nvidia\.com/gpu"
```

**Expected output**: Each Worker node shows `4` GPUs. If it shows `<none>`, check whether the device-plugin pod was scheduled onto the Worker (`kubectl get pods -n kube-system -l app.kubernetes.io/name=nvidia-device-plugin`).

## 3.2 DRA GPU Driver (includes ComputeDomain controller)

The **DRA GPU Driver** (v25.12.0+) provides the ComputeDomain CRD and controller for managing the IMEX daemon lifecycle. Installation deploys the kubelet plugin (DaemonSet) and the controller (Deployment).

```bash
# Install DRA GPU Driver via Helm (OCI registry, not a traditional repo)
# Note: the version number is 0.4.0 (not v25.12.0)
helm upgrade --install nvidia-dra-driver-gpu \
  oci://registry.k8s.io/dra-driver-nvidia/charts/dra-driver-nvidia-gpu \
  --version 0.4.0 \
  --namespace nvidia-dra-driver-gpu --create-namespace \
  --set nameOverride=nvidia-dra-driver-gpu \
  --set nvidiaDriverRoot=/ \
  --set gpuResourcesEnabledOverride=true \
  --set controller.affinity=null \
  --set controller.priorityClassName='' \
  --set kubeletPlugin.priorityClassName=''

# For v0.4.0 the CRD must be applied manually (the helm chart does not include the ComputeDomain CRD by default)
kubectl apply -f https://raw.githubusercontent.com/kubernetes-sigs/dra-driver-nvidia-gpu/v0.4.0/deployments/helm/dra-driver-nvidia-gpu/crds/resource.nvidia.com_computedomains.yaml

# Verify DRA GPU Driver components
kubectl -n nvidia-dra-driver-gpu get pods -o wide
# You should see: controller Pod (1/1) + one kubelet-plugin Pod (2/2) per GPU Worker

# Verify the ComputeDomain CRD is registered
kubectl get crd | grep computedomain
# Should show: computedomains.resource.nvidia.com + computedomaincliques.resource.nvidia.com
```

> **Verified in practice** (2026-06-27): DRA GPU Driver 0.4.0 installed successfully from the OCI registry `registry.k8s.io/dra-driver-nvidia/charts/dra-driver-nvidia-gpu`. The CRD must be applied separately.

**Prerequisite**: The Worker nodes' IMEX channels must already be created (`/dev/nvidia-caps-imex-channels/channel0` exists). This is configured in Phase 1 of the Worker startup script (`NVreg_CreateImexChannel0=1` + `dracut --force` + reboot).

**Mutual exclusivity**: ComputeDomain is mutually exclusive with the nvidia-imex systemd service. If `nvidia-imex.service` is running on a Worker, it must be stopped first: `systemctl stop nvidia-imex && systemctl disable nvidia-imex`

## 3.3 Install DRANET

DRANET v1.3.0 is the Kubernetes SIG DRA networking driver, used to allocate RDMA NICs to Pods as DRA devices.

```bash
# Install DRANET v1.3.0 via Helm
helm install dranet oci://registry.k8s.io/networking/charts/dranet \
  --version $DRANET_VERSION \
  --namespace kube-system \
  --wait --timeout 300s

# Verify the DRANET DaemonSet
kubectl -n kube-system get ds dranet

# Verify ResourceSlices (each node should discover its RDMA devices)
kubectl get resourceslice -o wide | head -20
```

## 3.4 Create the RDMA DeviceClass

```bash
kubectl apply -f yamls/k8s1341-rdma-deviceclass.yaml
```

DeviceClass contents (note the API is `resource.k8s.io/v1`):

```yaml
apiVersion: resource.k8s.io/v1
kind: DeviceClass
metadata:
  name: rdma-devices
spec:
  selectors:
  - cel:
      expression: |
        device.driver == "dra.net" &&
        has(device.attributes["dra.net"].rdma) && device.attributes["dra.net"].rdma == true
```

> **The `rdma == true` filter is mandatory**: DRANET discovers all network interfaces on the host, including Calico's `vxlan.calico` (normalized device name `net-oz4gyylofzrwc3djmnxq`). If the DeviceClass matches only on `device.driver == "dra.net"`, a Pod may be allocated the vxlan.calico interface, and when the DRANET NRI plugin configures routing it will report `fail to add route for interface vxlan.calico: network is unreachable`, leaving the Pod stuck in ContainerCreating. Adding `rdma == true` precisely filters for the Mellanox RDMA NICs. (Verified in practice 2026-06-28)

#### Verify the installation

```bash
kubectl get deviceclass
# Should show: rdma-devices

kubectl get resourceslice -o wide | head -20
```

## 3.5 Scheduler RBAC completion (required for kubeadm 1.34)

**Critical step**: The default `system:kube-scheduler` ClusterRole in kubeadm 1.34.x is severely incomplete — it lacks not only DRA permissions but even the list/watch permissions for basic resources such as pods, nodes, and services. If you skip this step:
- ComputeDomain daemon pods remain permanently in ContainerCreating (no logs, no errors)
- DRA ResourceClaims remain permanently pending (the scheduler cannot list ResourceSlices/DeviceClasses)
- Pods using podAffinity/podAntiAffinity cannot be scheduled

```bash
# Scheduler RBAC completion — must be run before deploying any GPU workload
cat <<'EOF' | kubectl apply -f -
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRole
metadata:
  name: system:kube-scheduler:full
rules:
- apiGroups: [""]
  resources: ["pods", "pods/status", "pods/binding"]
  verbs: ["get", "list", "watch", "update", "patch", "create", "delete"]
- apiGroups: [""]
  resources: ["nodes", "nodes/status"]
  verbs: ["get", "list", "watch", "update", "patch"]
- apiGroups: [""]
  resources: ["services", "endpoints", "namespaces", "configmaps"]
  verbs: ["get", "list", "watch"]
- apiGroups: [""]
  resources: ["events"]
  verbs: ["create", "patch", "update", "get", "list", "watch"]
- apiGroups: [""]
  resources: ["persistentvolumeclaims", "persistentvolumeclaims/status", "persistentvolumes"]
  verbs: ["get", "list", "watch", "update", "patch"]
- apiGroups: [""]
  resources: ["replicationcontrollers"]
  verbs: ["get", "list", "watch"]
- apiGroups: ["apps"]
  resources: ["replicasets", "statefulsets", "daemonsets"]
  verbs: ["get", "list", "watch"]
- apiGroups: ["policy"]
  resources: ["poddisruptionbudgets"]
  verbs: ["get", "list", "watch"]
- apiGroups: ["storage.k8s.io"]
  resources: ["storageclasses", "csinodes", "csidrivers", "csistoragecapacities"]
  verbs: ["get", "list", "watch"]
- apiGroups: ["resource.k8s.io"]
  resources: ["resourceclaims", "resourceclaims/status"]
  verbs: ["get", "list", "watch", "update", "patch"]
- apiGroups: ["resource.k8s.io"]
  resources: ["resourceslices", "deviceclasses"]
  verbs: ["get", "list", "watch"]
- apiGroups: ["coordination.k8s.io"]
  resources: ["leases"]
  verbs: ["get", "list", "watch", "create", "update", "patch", "delete"]
- apiGroups: ["events.k8s.io"]
  resources: ["events"]
  verbs: ["create", "patch", "update"]
---
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRoleBinding
metadata:
  name: system:kube-scheduler:full
roleRef:
  apiGroup: rbac.authorization.k8s.io
  kind: ClusterRole
  name: system:kube-scheduler:full
subjects:
- apiGroup: rbac.authorization.k8s.io
  kind: User
  name: system:kube-scheduler
EOF

# Restart the Scheduler so its informers re-sync
kubectl delete pod -n kube-system -l component=kube-scheduler
kubectl get pods -n kube-system -l component=kube-scheduler -w
# Wait for Running 1/1
```

## 3.6 Create the ComputeDomain

**ComputeDomain** is a CRD provided by the DRA GPU Driver for declaratively managing the IMEX daemon lifecycle. After creation, the controller automatically:

1. Creates a daemon DaemonSet → starts IMEX daemon pods on the nodes within the domain
2. Creates a ResourceClaimTemplate (Pods obtain an IMEX channel through this template)
3. IMEX daemon pods interconnect via the pod CIDR (port 50000) and establish an IMEX session

```bash
# Prerequisite: nvidia-imex.service must already be disabled (handled in step 3.2)

# Create a ComputeDomain for each Domain
for d in $(seq 1 $NUM_DOMAINS); do
  CD_NAME="domain-${d}-compute-domain"
  echo "=== Creating ComputeDomain: ${CD_NAME} ==="

  cat <<EOF | kubectl apply -f -
apiVersion: resource.nvidia.com/v1beta1
kind: ComputeDomain
metadata:
  name: ${CD_NAME}
spec:
  numNodes: 0
  channel:
    allocationMode: Single
    resourceClaimTemplate:
      name: ${CD_NAME}-channel
EOF

  # numNodes: 0 — recommended value (when IMEXDaemonsWithDNSNames=true); does not gate daemon startup
  # allocationMode: Single — each Pod exclusively owns one IMEX channel

  # Get the ComputeDomain UID and label the nodes within the domain
  CD_UID=$(kubectl get computedomain ${CD_NAME} -o jsonpath='{.metadata.uid}')
  echo "ComputeDomain UID: $CD_UID"

  for i in $(seq 0 $((NODES_PER_DOMAIN - 1))); do
    kubectl label node ${WORKER_PREFIX}-d${d}-w${i} \
      resource.nvidia.com/computeDomain=$CD_UID --overwrite
  done
done

# Wait for the daemon pods to start (usually 30-60 seconds)
kubectl get pods -n gpu-operator -l app=computedomain-daemon -w

# Verify the status of all ComputeDomains
kubectl get computedomain -o wide
# Each ComputeDomain's status should be Ready
```

**Expected**: ComputeDomain status is Ready, each node in the domain has one daemon pod running, and the IMEX session is established successfully.

**Note**: The daemon logs will continuously attempt to connect to non-existent peers (e.g., a domain has 18 nodes total but only 2 are in use); this is normal behavior and can be ignored.

**Production environment**: Each physical NVL72 domain requires one ComputeDomain. 1800 cards = 25 domains × 18 nodes/domain = 25 ComputeDomains + 25 Placement Policies.

**Multiple workloads within a domain**: ComputeDomain **supports** coexistence of multiple workloads within a domain. Each node belongs to at most one ComputeDomain (each node can run only one IMEX daemon), but the ResourceClaimTemplate of the same ComputeDomain can be shared by multiple Pods.

---

## 4. GCSFuse + Lustre Shared Storage

### 4.1 GCSFuse (already configured in the startup script)

Each Worker's startup script (`tlinux4-k8s134-worker.sh`) already includes GCSFuse installation and mounting:

```bash
# The startup script automatically runs:
dnf install -y gcsfuse

# Mount the GCS bucket (using Local SSD as the file cache)
mkdir -p ${LOCAL_SSD_MOUNT}/shared
gcsfuse --implicit-dirs \
  --file-cache-dir=${LOCAL_SSD_MOUNT}/gcsfuse-cache \
  $GCSFUSE_BUCKET ${LOCAL_SSD_MOUNT}/shared
```

Accessed from within Pods via hostPath:

```yaml
volumes:
- name: shared-data
  hostPath:
    path: /mnt/stateful_partition/shared    # GCSFuse mount point
- name: scratch-data
  hostPath:
    path: /mnt/stateful_partition/scratch-data  # Local SSD high-speed cache
```

### 4.2 GCSFuse v2 high-performance configuration (optional)

GCSFuse v2 supports parallel downloads and kernel caching, which can significantly speed up loading large model checkpoints.

```bash
# Enable the v2 high-performance mount in the startup script
gcsfuse \
  --implicit-dirs \
  --file-cache-dir=${LOCAL_SSD_MOUNT}/gcsfuse-cache \
  --file-cache-capacity-per-file-mb=-1 \
  --file-cache-cache-file-for-range-read \
  --file-cache-enable-parallel-downloads \
  --file-cache-parallel-downloads-per-file=16 \
  --file-cache-download-chunk-size-mb=50 \
  --file-cache-max-parallel-downloads=64 \
  --kernel-list-cache-ttl-secs=600 \
  --metadata-cache-ttl-secs=600 \
  --stat-cache-capacity=20000 \
  --type-cache-max-size-mb=32 \
  --rename-dir-limit=200000 \
  $GCSFUSE_BUCKET ${LOCAL_SSD_MOUNT}/shared
```

**Key parameter descriptions**:

- `--file-cache-enable-parallel-downloads`: Enable file-level parallel downloads (new in v2)
- `--file-cache-parallel-downloads-per-file=16`: Up to 16 parallel chunks per file
- `--file-cache-capacity-per-file-mb=-1`: Do not limit the per-file cache size
- `--kernel-list-cache-ttl-secs=600`: Cache directory listings for 10 minutes, reducing metadata API calls

**Note**: The file cache resides on the Local SSD RAID0 (`/mnt/stateful_partition/gcsfuse-cache`); the A4X ships with ~12TB of NVMe, providing extremely high cache IOPS.

### 4.3 Lustre (optional)

If the customer provides a Lustre file system, mount it using the Lustre CSI Driver + PV/PVC approach.

#### Kernel 6.6 and Lustre version compatibility

GB200 Workers run kernel 6.6 (TLinux 4 / Rocky Linux 9.x), and the **Lustre 2.14 client kernel module cannot be compiled on kernel 6.6**. 2.14 only supports kernel 4.18-5.x (the RHEL 8 era); the VFS/networking-stack API changes in 6.x kernels make the source incompatible.

| Lustre version | Highest supported kernel | Notes |
|---|---|---|
| 2.14.x | ~5.x (RHEL 8) | Cannot be compiled on kernel 6.6 |
| 2.15.7+ | 5.14 (RHEL 9.6) | Partial 6.x support |
| 2.16+ | 6.x (RHEL 9.4+) | Recommended |
| 2.17+ | 6.6+ | Full support |

If the customer's Lustre **server** is 2.14, **the client must use 2.16+ or 2.17** (backward compatible; a newer client can connect to an older server).

#### Grant deadlock: 2.17 Client + 2.14 Server pitfall (verified)

**Symptom**: After a buffered I/O write (`cp -a`, `tar`, application writes), sync hangs and all subsequent I/O — including reads — is blocked. The `dd oflag=direct` test **cannot reproduce** this (direct I/O bypasses the page cache and the grant mechanism).

**Root cause**: The mismatched client/server versions cause the grant parameters to be incompatible.

- Lustre 2.14 server: `initial_grant=8MB`, dynamic max ~60MB
- Lustre 2.17 client default: `max_dirty_mb=64MB` (per-OST dirty-page cap)
- 64MB > 60MB → the client's backlog of dirty pages exceeds the server's granted amount → grant is exhausted (`cur_grant_bytes=0`) → flush RPCs cannot be sent → permanent deadlock

**Kernel errors**:
```
LustreError: osc_announce_cached: data-OST0004-osc: dirty 16675 > dirty_max 16384
LustreError: osc_extent_wait: data-OST0007-osc: wait ext to 0 timedout, recovery in progress?
```

**Fix**: Lower the client parameters to ensure the dirty-page cap < the server grant cap.

```bash
# Apply temporarily
lctl set_param osc.*.max_dirty_mb=32       # default 64 → 32 (< server max grant ~60MB)
lctl set_param osc.*.max_rpcs_in_flight=4   # default 8 → 4 (gives the 2.14 server more processing time)
```

**Persistence**: `/etc/lctl.conf` only takes effect once, when the module is loaded, and cannot override the new OSC instances created by CSI mounts. You must use a systemd service:

```bash
cat <<'EOF' | sudo tee /etc/systemd/system/lustre-tuning.service
[Unit]
Description=Lustre client tuning for 2.14 server compatibility
After=kubelet.service

[Service]
Type=oneshot
ExecStartPre=/bin/sleep 30
ExecStart=/usr/sbin/lctl set_param osc.*.max_dirty_mb=32
ExecStart=/usr/sbin/lctl set_param osc.*.max_rpcs_in_flight=4
RemainAfterExit=yes

[Install]
WantedBy=multi-user.target
EOF
sudo systemctl daemon-reload && sudo systemctl enable lustre-tuning
```

#### GCS → Lustre data transfer

Writing directly from GCS into Lustre (`gcloud storage rsync`) generates a large number of concurrent buffered writes due to composite downloads, quickly exhausting the grant. A two-stage approach is recommended:

```bash
# 1. GCS → local SSD (~3.8 GB/s)
gcloud storage rsync -r gs://bucket/models /lssd/models

# 2. Local SSD → Lustre (parallel cp across multiple directories, ~6 GB/s)
for dir in /lssd/models/*/; do
  cp -a "$dir" /mnt/lustre/models/ &
done
wait

# 3. Confirm dirty pages are zeroed out (a non-zero value indicates a grant deadlock; requires umount -l + remount)
lctl get_param osc.*.cur_dirty_bytes | grep -v "=0$"
```

#### Loading models into vLLM from Lustre

By default vLLM uses mmap to load safetensors. When `tensor-parallel-size > 1`, multiple GPU workers trigger mmap page faults simultaneously, and Lustre's `ll_filemap_fault` serializes these faults, causing read performance to degrade exponentially with the number of shards. Solution: first `cp` the model to Local SSD or `/dev/shm`, then load it from local storage.

---

## k8s 1.34 DRA + ComputeDomain considerations

| Item | Description |
|------|------|
| DRA API version | ResourceClaimTemplate/DeviceClass use `resource.k8s.io/v1` (the GA API, not v1beta2) |
| DRA status | In k8s 1.34, DRA is **GA**, enabled by default, no feature gate required |
| kubeadm Scheduler RBAC | The default `system:kube-scheduler` ClusterRole in kubeadm 1.34.x is severely incomplete — it lacks not only DRA permissions but even basic permissions for pods/nodes/services. **Must be completed manually** (see 3.5) |
| DRANET version | v1.3.0, install: `helm install dranet oci://registry.k8s.io/networking/charts/dranet --version v1.3.0 -n kube-system` |
| ComputeDomain mutual exclusivity | **Mutually exclusive** with `nvidia-imex.service` (systemd) and the IMEX Manager DaemonSet |
| ComputeDomain node labels | After creation, nodes must be labeled manually: `kubectl label node <NODE> resource.nvidia.com/computeDomain=<UID>` |
| IMEX daemon log noise | In a <18-node environment, the daemon will continuously attempt to connect to non-existent nodes — **harmless, can be ignored** |
| NCCL test binary | The `all_reduce_perf` shipped in the pytorch image **is not linked against MPI**; the MPI version must be compiled from source |
| mpirun path | `/usr/local/mpi/bin/mpirun` (not `/usr/local/gib/bin/mpirun`) |
| Calico multi-NIC | A4X Workers have 6 NICs (2 GVNIC + 4 MRDMA); the Calico Installation must set `nodeAddressAutodetectionV4.cidrs` to the management subnet. The default `firstFound` will pick an RDMA NIC, causing BGP failure and Pod DNS breakdown (see 02-k8s-cluster Step 7 notes) |
| SSH keys | Inside the container you must use `ed25519` (`ssh-keygen -t ed25519`); RSA fails because there is no `/dev/tty` |
