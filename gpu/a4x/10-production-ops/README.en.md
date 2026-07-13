> 🌐 [中文](README.md) | **English**

# GB200 NVL72 Production Operations Knowledge Base

## 1. Overview

This section consolidates **all high-value engineering lessons** accumulated from deploying to running production operations on GB200 NVL72 (A4X) clusters, covering:

- **16 recurring production-grade pitfalls** and their fixes (already baked into scripts/yaml)
- **XID fault auto-recovery** flow (Xid 137/145/94 NVLink fabric hardware faults)
- **DRA Controller Race** detection and 2-stage fix
- **Base Image Baking** methodology (7 Phases, new VM boots in < 2 min)
- **One-click Workload deployment** wrapper (16min stuck -> 4min Ready)
- **cuBLAS GEMM Benchmark** for single-GPU peak compute validation
- Full benchmark results at a glance (cuBLAS / NCCL / Megatron / DeepEP)

**Hardware environment**: GB200 NVL72 (Grace ARM CPU + 2 Blackwell GPU superchip), 4 GPU + 4 MRDMA NIC + 2 GVNIC per node. 18 nodes form 1 NVL72 super-pod, sharing an NVLink Switch fabric (~840 GB/s MNNVL cross-node interconnect). k8s 1.34 + NVIDIA DRA Driver v0.4 ComputeDomain CRD manages cross-node IMEX sessions.

---

## 2. 16 High-Value Lessons Learned

The following 16 pitfalls **recurred** during NVL72 cluster deployment and stress testing, and have all been baked into scripts/yaml/wrappers.

### 2.1 NVIDIA DRA v0.4 controller race -- hits on every workload switch

| Item | Details |
|---|---|
| **Symptom** | After a Pod is scheduled it hangs in `PodInitializing > 5min`, event reports `FailedPrepareDynamicResources DeadlineExceeded`, daemon DS DESIRED=0 never spawns |
| **Root cause** | DRA Driver v0.4 controller internal optimistic locking failure: `Operation cannot be fulfilled on computedomains... the object has been modified (attempt 1)`, silently drops the work item and never adds the `resource.nvidia.com/computeDomain=<CD-UID>` label to the node hosting the client pod |
| **Fix (2-stage)** | Stage 1: `kubectl label node <pod-node> resource.nvidia.com/computeDomain=<CD-UID>` manually applies the label, daemon DS begins spawning; Stage 2: `ssh node 'sudo systemctl restart kubelet'` unsticks the daemon pod from ContainerCreating |
| **Already baked** | `check-k8s-dra-health.sh --fix-race` automatically runs the 2-stage fix; `prepare-workload.sh` auto-invokes fix-race after apply; `dra-cd-label-reconciler-deployment.yaml` runs as a cluster-resident controller polling every 15s to auto-apply the label |

### 2.2 reconciler idle for 5h -- `kubectl get cd` short name silent fail

| Item | Details |
|---|---|
| **Symptom** | Reconciler pod is Running but shows no reconcile activity for 5 hours |
| **Root cause** | The ConfigMap script used `kubectl get cd -A`, but the server does not register the `cd` short name (only the full `computedomain`), returning an error to stderr that parses as empty, silently skipping 1200+ times |
| **Fix** | Must use the full name `kubectl get computedomain -A`, with `grep -v "^error"` filtering |
| **Already baked** | `dra-cd-label-reconciler.sh` and the corresponding deployment yaml have been corrected |

### 2.3 GIB libnccl ABI mismatch -- DeepEP / Megatron hard assert

| Item | Details |
|---|---|
| **Symptom** | Pod import triggers `AssertionError: Invalid NCCL versions: /usr/local/gib/lib64/libnccl.so.2.30.4 (loaded) v.s. ... (expected)` |
| **Root cause** | The libnccl injected by the GIB init container has the **same version number but different binary content** compared to the image's pip `nvidia-nccl-cu13` (GIB is a self-built version); DeepEP `check_nccl_so()` uses `filecmp.cmp(shallow=False)` for strict comparison and hard fails |
| **Fix** | `export LD_PRELOAD=/usr/local/lib/python3.12/dist-packages/nvidia/nccl/lib/libnccl.so.2` to make dlopen prefer the pip libnccl; also `rm -rf /opt/DeepEP` to avoid triggering the import |
| **Already baked** | `gen-megatron-sts.py` pod entry now adds LD_PRELOAD + rm -rf |

### 2.4 Megatron multi-node iter log is in the last rank pod, not the master

| Item | Details |
|---|---|
| **Symptom** | After a multi-node run, `kubectl logs <master-pod>` has no iter log |
| **Root cause** | Megatron `training_log()` uses `is_last_rank()` = `rank == world_size - 1`; in multi-node the last rank is in the final pod (replicas=16 -> g1-15) |
| **Fix** | Collect logs from `kubectl logs <pod>-g1-{N-1}`; note that for a normally Exit 0 Complete pod, use plain `kubectl logs` and **do not add `--previous`** (that only shows the previous instance of a restarted container, and returns empty for a normal Complete) |
| **Already baked** | `prepare-workload.sh` docs + Megatron results README explicitly note this |

### 2.5 Megatron `--eval-iters 0` triggers `int // None` crash

| Item | Details |
|---|---|
| **Symptom** | When building the dataloader it reports `TypeError: unsupported operand for //: 'int' and 'NoneType'` |
| **Root cause** | Megatron-core 0.16 `get_train_valid_test_num_samples()` computes `args.train_iters // args.eval_interval`; `--eval-iters 0` does not actually run eval but `args.eval_interval` defaults to None, triggering a division error |
| **Fix** | Explicitly set `--eval-interval 1000` (any positive number, paired with eval-iters=0 no eval actually runs) |
| **Already baked** | `gen-megatron-sts.py` adds this by default |

### 2.6 `--log-throughput` multi-node + mock-data silently skips iter log

| Item | Details |
|---|---|
| **Symptom** | Adding `--log-throughput` to auto-print TFLOPs/GPU, but multi-node + mock-data skips the iter log entirely |
| **Root cause** | Megatron-core 0.16 bug: on the multi-node path the throughput computation silently swallows the iter log |
| **Fix** | Do not add `--log-throughput`; compute TFLOPs manually with the 6N formula: `TF/GPU = 6 * N_params * (gbs * seq) / (avg_iter_sec * num_gpus) / 1e12`. Adding the flag is OK for single-node |
| **Already baked** | The yaml generator has removed `--log-throughput` |

### 2.7 Xid 137/145/94 -- GPU NVLink fault, one-click auto-recovery

| Item | Details |
|---|---|
| **Symptom** | Pod CrashLoopBackOff, dmesg reports `NVRM: Xid (PCI:0000:01:00): 137, ...` (NVLink fabric error) |
| **Root cause** | NVL72 MNNVL fabric / NVLink Switch transient error; device-plugin marks the GPU Unhealthy, node `nvidia.com/gpu` decreases, workload scheduling fails |
| **Fix** | 3 escalating options: (1) `nvidia-smi --gpu-reset` (recovers most cases); (2) `lspci -d 10de: \| xargs setpci ... LNKCTL` to reset the PCI link (needed for a few cases); (3) node reboot (very rare last resort) |
| **Already baked** | `xid-fault-auto-recover.sh --apply` automatically: detect bad nodes -> verify Xid -> serial reset -> force delete stale device-plugin pod -> poll GPU=4 |

### 2.8 stale Terminating CD daemons are not auto-GC'd

| Item | Details |
|---|---|
| **Symptom** | After a ComputeDomain is deleted, daemon pods hang in `Terminating` with a finalizer in the `nvidia-dra-driver-gpu` namespace, accumulating 30+ stale pods |
| **Root cause** | The NVIDIA DRA v0.4 daemon DS finalizer races with CD GC; when the CD is deleted the daemon is not cascade-cleaned |
| **Fix** | `kubectl delete pod -n nvidia-dra-driver-gpu --force --grace-period=0` |
| **Already baked** | `prepare-workload.sh` Step 1 automatically cleans up stale Terminating daemons before every apply |

### 2.9 Calico VXLAN defaults to tailscale instead of VPC

| Item | Details |
|---|---|
| **Symptom** | Cross-node pod networking is slow (~50 MB/s instead of several GB/s over the VPC); VXLAN encapsulation went through wireguard |
| **Root cause** | The Tigera operator `nodeAddressAutodetectionV4: {firstFound: true}` (default) picks the first non-loopback NIC; if tailscale0 enumerates first it gets picked |
| **Fix** | Explicitly configure `nodeAddressAutodetectionV4: {kubernetes: NodeInternalIP}` to get the correct IP from kubelet |
| **Already baked** | The deployment docs' kubeadm init + Calico install steps now enforce this configuration |

### 2.10 tailscale MagicDNS hijacks /etc/resolv.conf, breaking all dnf install

| Item | Details |
|---|---|
| **Symptom** | After installing tailscale, `dnf install` reports `Cannot prepare internal mirrorlist: Curl error (6): Couldn't resolve host name` |
| **Root cause** | tailscale takes over /etc/resolv.conf; the MagicDNS resolver does not query internal domains (internal mirrors are NXDOMAIN on public DNS, only the GCE metadata DNS has them) |
| **Fix** | `tailscale up --accept-dns=false` to not take over resolv.conf |
| **Already baked** | The tailscale config in the startup script now adds `--accept-dns=false` |

### 2.11 alltoall hang across 2 cliques -- NVIDIA Known Issue, no workaround

| Item | Details |
|---|---|
| **Symptom** | 36-node NCCL alltoall across 2 cliques stuck, with no error/timeout/exit |
| **Root cause** | An NVIDIA-confirmed NCCL race condition (missing cross-2-clique atomic lock); no fix in the current NCCL release |
| **Fix** | (1) Restrict alltoall to a single clique (<=72 GPU); (2) across cliques run only the 3 collectives (all_reduce / all_gather / reduce_scatter); (3) 36-node chain of 4 collectives uses a split yaml (3coll + alltoall run separately) for 100% PASS |
| **Already baked** | The yaml generator provides `3coll` / `alltoall` split modes |

### 2.12 Zombie Monitor -- grep pattern only covers terminal states, misses stuck signals

| Item | Details |
|---|---|
| **Symptom** | Pod stuck in init for 16 min; Monitor only noticed at its 1h timeout |
| **Root cause** | The Monitor command `until kubectl logs \| grep "iteration 50"` only waits for the success signal; a pod stuck in init emits no iteration log, so the monitor stays silent all the way to timeout |
| **Fix** | The grep pattern must cover three categories: **progress** (`iteration X/Y`, `step N`, `TFLOPs`), **terminal** (`DONE`, `FAIL`, `Traceback`, `exit code`), and **stuck** (`FailedScheduling`, `FailedPrepareDynamicResources`, `ImagePullBackOff`, `CrashLoopBackOff`, `DeadlineExceeded`, `OOM`, `NCCL.*timeout`) |
| **Already baked** | The global ops-standards Monitor section now includes a hardened anti-zombie checklist |

### 2.13 DRA kubelet-plugin Running but CD ResourceSlice not published

| Item | Details |
|---|---|
| **Symptom** | Physically all healthy, DRA all Running, but an 18n workload has 5/18 stuck Pending, Event `5 cannot allocate all claims`, controller log `CDStatusSync: total nodes=13` capped |
| **Root cause** | The DRA driver v0.4.0 kubelet-plugin `compute-domains` container occasionally gets stuck in an informer `Watch close` reconnect; Running but not actually running `Publishing ResourceSlice` |
| **Diagnosis** | `kubectl get resourceslice --no-headers \| grep compute-domain.nvidia.com \| wc -l` (expected = worker count; fewer means some plugin didn't publish) |
| **Fix** | Precisely delete the plugin pod on the node missing its slice; the DS respawns a fresh pod (~15s) triggering a fresh publish. Much lighter than a kubelet restart (kubelet restart has a large blast radius) |
| **Already baked** | `check-k8s-dra-health.sh --fix-rs` auto-diagnoses + targeted delete + wait publish |

### 2.14 host systemd nvidia-imex.service contends with the ComputeDomain daemon for the IMEX session

| Item | Details |
|---|---|
| **Symptom** | Workload pod stuck in `ContainerCreating`, kubelet event `AssertComputeDomainReady` fails, daemon log repeatedly `NV_ERR_IN_USE Failed to allocate Imex session` |
| **Root cause** | The host's systemd `nvidia-imex.service` (enabled by default by the driver package) contends with the CD daemon pod for the same IMEX session; the driver allows only one owner |
| **Fix** | Disable + mask during worker base image baking: `systemctl disable --now nvidia-imex && systemctl mask nvidia-imex` |
| **Already baked** | `image-build.sh` Phase 6 has locked this in; existing workers can be remediated via bulk SSH |

### 2.15 DRANET doesn't inject the netlink interface -- DeepEP raw ibverbs cross-host fails

| Item | Details |
|---|---|
| **Symptom** | NCCL via GIB plugin works cross-host (326 GB/s), but DeepEP test_ep cross-host reports `ibv_modify_qp errno 110 Connection timed out` |
| **Root cause** | DRANET v1.3.0 only injects the ibverbs char devices (`/dev/infiniband/uverbs0..3`); it **does not move the netlink interface (bond2-5) into the pod ns, nor configure an IP**. Inside the pod the GID only has link-local IPv6, and RoCEv2 IPv4 is entirely empty. NCCL via GIB uses `rdma_cm` + IB CM which bypasses this limitation; DeepEP raw `ibv_modify_qp` requires a routable GID |
| **Fix** | DeepEP / raw ibverbs workloads use `hostNetwork: true` to bypass DRANET and directly use the host's bond2-5 which have IPs |
| **Already baked** | NCCL/Megatron workloads use the DRANET ResourceClaim (works); the DeepEP workload yaml uses `hostNetwork: true` |

### 2.16 GB200 kernel cmdline `iommu.passthrough=1` breaks UVM ATS bind -- CUDA errors 701 at startup

| Item | Details |
|---|---|
| **Symptom** | All CUDA apps report `cudaErrorDevicesUnavailable (701)` at startup, even with fabric Completed / imex active / driver freshly initialized. dmesg repeatedly floods `uvm_ats_sva_bind_gpu` stacks |
| **Root cause** | The kernel cmdline `iommu.passthrough=1 iommu.strict=0` (the "performance" setting recommended by the NVIDIA Grace tuning doc) **breaks UVM ATS bind**. The Grace doc's original text "might have issues with ATS" -- for GB200 Grace+Blackwell heavy-UVM-ATS scenarios this is a hard break |
| **Ruled out** | fabricmanager fail (expected on GB200 compute nodes), missing CD CR, nvidia-imex inactive, driver state cache, containerd version -- all ruled out |
| **Fix** | Remove `iommu.passthrough=1 iommu.strict=0` from the kernel cmdline, keeping only `init_on_alloc=0`. Let SMMUv3 use the default translation mode. Cost: theoretically slightly higher PCIe DMA latency, but GB200 GPU-CPU communication goes over NVLink-C2C (900 GB/s within the Grace<->Blackwell single chip), so the practical impact is minimal |
| **Already baked** | `image-build.sh` Phase 2: `grubby --update-kernel=ALL --remove-args="iommu.passthrough iommu.strict"` |

---

## 3. XID Fault Auto-Recovery

`scripts/troubleshooting/xid-fault-auto-recover.sh` implements **fully automated detection and recovery** of NVLink Xid hardware faults, unattended.

### 3.1 Handled XID error codes

| XID Code | Meaning |
|---|---|
| **137** | NVLink fabric error (cross-node MNNVL fabric communication fault, sticky) |
| **145** | NVLink device error (device-level NVLink fault, sticky) |
| **94** | NVLink/device-plugin related fault |

These Xids are **sticky** -- the GPU device-plugin marks the affected GPU Unhealthy, the node `nvidia.com/gpu` allocatable drops from 4 to 3 or lower, and new workloads cannot be scheduled.

### 3.2 Recovery flow (6 steps)

```
Step 1: Detect -- kubectl get nodes to find nodes with nvidia.com/gpu allocatable < 4
     |
Step 2: Verify -- SSH to the bad node, dmesg grep Xid 137/145/94 to confirm it's an NVLink fault
     |             (nodes without an NVLink fault are skipped, need manual investigation)
     |
Step 3: Reset -- gcloud compute instances reset (host reboot)
     |           *** serial by default, to avoid IMEX storm cascade ***
     |
Step 4: Wait -- poll node Ready=True (max 900s)
     |
Step 5: Cleanup -- force delete stale device-plugin pod (Unknown/Pending/Terminating)
     |              to let the DaemonSet spawn a fresh pod
     |
Step 6: Verify -- poll GPU allocatable=4 (max 5 min, check every 15s)
```

### 3.3 Key: why reset must be serial

**IMEX storm cascade**: the 18 nodes inside an NVL72 super-pod share the MNNVL fabric; resetting multiple nodes simultaneously causes the nvidia-imex daemons to stop at the same time, triggering a cross-cluster MNNVL fabric session race that **produces new NVLink Xid faults on nodes that were originally healthy**:

- 12 nodes reset in parallel -> 10 **new** faults (45% trigger rate)
- 10 nodes in parallel -> 3 new faults
- single node serial -> 0 cascade

Serial by default (`--parallel` can override, but it will cascade). Serial takes ~10 min per node and converges in 1 pass; parallel is faster but needs multiple passes to converge.

### 3.4 Usage

```bash
# Dry-run: detect affected nodes, no changes
bash xid-fault-auto-recover.sh

# Apply the fix (serial, recommended)
bash xid-fault-auto-recover.sh --apply

# Single-node fix
bash xid-fault-auto-recover.sh --node <worker-hostname> --apply

# Parallel fix (fast but may cascade)
bash xid-fault-auto-recover.sh --apply --parallel
```

---

## 4. DRA Controller Race Fix

The NVIDIA DRA Driver v0.4 controller has an optimistic locking race condition, the most frequent operational issue on NVL72 clusters.

### 4.1 Problem mechanism

```
workload pod requests a ComputeDomain channel ResourceClaim
     |
DRA Controller attempts to Update the computedomain CR
     |
optimistic locking fails: "the object has been modified (attempt 1)"
     |
Controller silently drops the work item (no retry)
     |
Node is missing the resource.nvidia.com/computeDomain=<CD-UID> label
     |
daemon DaemonSet DESIRED=0, daemon pod never spawns
     |
Pod permanently stuck in PodInitializing / ContainerCreating
```

### 4.2 `check-k8s-dra-health.sh` detection

Script Check 12 specifically detects this race:

1. Iterate over all active ComputeDomains
2. Check each CD's daemon DaemonSet DESIRED value
3. Count the associated client channel ResourceClaim allocated count
4. **DESIRED=0 but client_RC_alloc>0** = confirmed controller race

```bash
# Detect
bash check-k8s-dra-health.sh

# Detect + auto-fix
bash check-k8s-dra-health.sh --fix-race
```

### 4.3 2-Stage fix flow

```bash
# Stage 1: manually apply the label (to let the daemon DS schedule)
kubectl label node <node> resource.nvidia.com/computeDomain=<CD-UID> --overwrite

# Wait 30s, check whether the daemon pod is Ready
# If still ContainerCreating...

# Stage 2: restart kubelet (refresh the PrepareResource cache)
ssh <node> 'sudo systemctl restart kubelet'
```

### 4.4 Continuous auto-fix

Deploy `dra-cd-label-reconciler-deployment.yaml` to the cluster, polling every 15s:

1. `kubectl get computedomain -A` to get all CDs and their UIDs
2. `kubectl get resourceclaim -A` to find the node associated with each allocated channel
3. Check whether the node has the `resource.nvidia.com/computeDomain=<CD-UID>` label
4. If missing, automatically `kubectl label node`

---

## 5. Base Image Baking

`scripts/host/image-build.sh` writes all host-level hardware configuration into the OS image in one pass, so a new VM boots in < 2 min and can join k8s.

### 5.1 Why bake

Without an image: each VM at every boot installs ~3 GB of packages + grub update + initramfs rebuild + reboot, ~10 min/node. With the image baked: a new VM is Ready in < 2 min.

### 5.2 7 Phases

| Phase | Content | Key operations |
|---|---|---|
| **1** | OS base packages + sshd + selinux + filesystem expansion | `dnf install`, growpart, selinux disabled |
| **2** | Kernel cmdline | `grubby --remove-args="iommu.passthrough iommu.strict"` + `--args="init_on_alloc=0"` (**key**: iommu.passthrough breaks UVM ATS, see 2.16) |
| **3** | IMEX initramfs | `echo "options nvidia NVreg_CreateImexChannel0=1" > /etc/modprobe.d/nvidia.conf` + `dracut --force --add-drivers "gve"` (otherwise `/dev/nvidia-caps-imex-channels` does not exist) |
| **4** | Common prereqs | swap off, modules-load (overlay, br_netfilter), sysctl ip_forward |
| **5** | Grace persistent sysctl | Write `/etc/sysctl.d/90-grace-gb200.conf`: `kernel.numa_balancing=0`, TCP BBR, buffer 128MB, `vm.max_map_count=1048576`, etc. |
| **6** | Reapply systemd service | Register `grace-gb200-reapply.service`, auto-run at every boot: PCI ACS off, CPU governor=performance, THP=madvise, nvidia_peermem, `nvidia-smi -pm 1`; **also disable nvidia-imex** (to let the CD daemon take over, see 2.14) |
| **7** | NIC rename | PCI BDF-based systemd `.link` files, 6 NIC -> bond0..bond5 (2 GVNIC + 4 MRDMA); the A4X hardware PCI topology is fixed, so baked once it applies to all VMs of the same type |

Additional Phase 8 (NIC ring + NVMe tuning): ethtool ring rx/tx=8192, mlx5 combined channels settings, NVMe poll_queues=16 + coalescing; disable irqbalance.

### 5.3 Baked vs Runtime

| Category | Item | Source |
|---|---|---|
| **Baked into image** | OS packages / kernel cmdline / IMEX initramfs / sysctl / NIC .link rename / systemd reapply service / nvidia-imex disabled | `image-build.sh` |
| **Runtime (auto via systemd at every boot)** | PCI ACS off / CPU governor=performance / THP=madvise / nvidia_peermem / nvidia-smi -pm 1 | `grace-gb200-reapply.service` (registered in Phase 6) |
| **Customer-installed (not in image)** | Container runtime (containerd 2.x, **NRI must be enabled**) / nvidia-container-toolkit / CNI plugins / kubelet + kubeadm + kubectl / k8s join | Customer's own k8s flow |

### 5.4 Usage

```bash
# 1. Create the build VM (using a vendor base image with driver pre-installed)
gcloud compute instances create gb200-image-builder \
  --machine-type=a4x-highgpu-4g --image=<vendor-base-image> ...

# 2. Push + execute (~5 min)
gcloud compute scp scripts/host/image-build.sh gb200-image-builder:/tmp/
gcloud compute ssh gb200-image-builder --command="sudo bash /tmp/image-build.sh"

# 3. Reboot (to apply initramfs + kernel cmdline + NIC rename)
gcloud compute instances reset gb200-image-builder

# 4. Verify
gcloud compute ssh gb200-image-builder --command="sudo bash /tmp/image-build.sh --verify"
# Expected: init_on_alloc=0 / iommu none / IMEX channel0 / numa_balancing=0 / bbr /
#       nvidia-imex disabled / grace-reapply enabled / 6 .link files / bond0..5 /
#       irqbalance disabled / GPU persist Enabled / Fabric Completed

# 5. Snapshot + Create image
gcloud compute instances stop gb200-image-builder
gcloud compute disks snapshot gb200-image-builder --snapshot-names=<snap-name>
gcloud compute images create gb200-worker-base-v1 --source-snapshot=<snap-name>
```

---

## 6. One-Click Workload Deployment

`scripts/troubleshooting/prepare-workload.sh` is an idempotent wrapper that bakes in the fix flow for all known DRA v0.4 race conditions, measured to reduce a 16min stuck to 4min Ready.

### 6.1 The 4 problems it solves

1. **stale Terminating CD daemons** -- the previous workload's daemon pods stuck with a Terminating finalizer (DRA v0.4 GC race)
2. **DRA controller race** -- "object has been modified" silent drop, daemon DS DESIRED=0
3. **kubelet PrepareResource cache stale** -- after a CD label switch, kubelet caches the old state
4. **FailedPrepareDynamicResources DeadlineExceeded** -- deadline timeout on the newly-labeled node

### 6.2 Execution flow (5 steps)

```
Step 1: Clean stale Terminating daemons
        kubectl delete pod -n nvidia-dra-driver-gpu --force --grace-period=0
             |
Step 2: Apply workload yaml
        kubectl apply -f <yaml>
             |
Step 3: Wait 25s
        give the DRA controller enough time to allocate the channel
             |
Step 4: Proactive fix-race
        invoke check-k8s-dra-health.sh --fix-race
        (Stage 1 label + Stage 2 kubelet restart if needed)
             |
Step 5: Wait for master pod Running (max 6min)
        poll every 10s; if still stuck at 3min, run a second fix-race
```

### 6.3 Usage

```bash
# Apply workload + fix + wait
bash prepare-workload.sh <yaml-path> <master-pod-name>

# Delete workload + cleanup
bash prepare-workload.sh -d <yaml-path>
```

---

## 7. cuBLAS GEMM Benchmark

`scripts/host/cublas_bench_gb200.sh` runs a single-GPU GEMM peak compute test **directly** on the worker host (without starting a k8s pod).

### 7.1 How it works

The host only has the NVIDIA driver (R580), with no CUDA toolkit installed. The script **borrows** libcublas / libcublasLt / libcudart (~670 MB) from an already-pulled container image via `ctr -n k8s.io images mount` into /tmp, then runs the benchmark binary with `LD_LIBRARY_PATH`.

```bash
# Extract CUDA libs from the container image (no CUDA toolkit install needed)
sudo ctr -n k8s.io images mount "$IMAGE" "$MOUNT"
sudo cp -a "$MOUNT"/usr/local/cuda-13.2/targets/sbsa-linux/lib/libcublas* /tmp/cublas_bench/
sudo cp -a "$MOUNT"/usr/local/cuda-13.2/targets/sbsa-linux/lib/libcudart* /tmp/cublas_bench/
sudo ctr -n k8s.io images unmount "$MOUNT"

# Run the benchmark with the borrowed libs
LD_LIBRARY_PATH=/tmp/cublas_bench bash cublas_bench_gb2_3.sh
```

### 7.2 Peak results

| Dtype | GEMM (M x N x K) | Measured TFLOPs | NVIDIA reference | Difference |
|---|---|---|---|---|
| FP4  | 9728 x 16384 x 8192 | **6845** | 6507 | +5.2% |
| FP8  | 9728 x 2048 x 32768 | **3063** | 2805 | +9.2% |
| FP16 | 8192 x 9728 x 16384 | **1492** | 1372 | +8.7% |
| BF16 | 8192 x 9728 x 16384 | **1592** | 1471 | +8.2% |
| TF32 | 8192 x 9728 x 16384 | **733**  | 675  | +8.6% |
| FP32 | 8192 x 9728 x 16384 | **75**   | 75   | on par |

All 6 dtypes measured 5-9% above the NVIDIA reference (newer cuBLAS optimization).

### 7.3 Usage

```bash
# Run on the control machine, auto-SSH to the worker host
gcloud compute ssh <worker-node> --command="bash -s" < scripts/host/cublas_bench_gb200.sh \
  | tee results/cublas/<worker-node>.log
```

---

## 8. Benchmark Results at a Glance

### 8.1 cuBLAS GEMM (single-GPU peak)

- FP4 **6845** / FP8 **3063** / FP16 **1492** / BF16 **1592** / TF32 **733** / FP32 **75** TFLOPs
- All 5-9% above the NVIDIA reference

### 8.2 NCCL cluster communication (16 GiB in-place busbw, GB/s)

| Scale | GPU count | all_reduce | all_gather | reduce_scatter | alltoall |
|---|---|---|---|---|---|
| 1n (4 GPU local) | 4 | 686.73 | 668.86 | 666.88 | 689.13 |
| 2n D=1 (same super-pod MNNVL) | 8 | **840.12** | 683.37 | 693.12 | 679.93 |
| 2n D=2 (cross-clique RDMA) | 8 | **326.09** | 188.85 | 188.89 | 83.16 |
| 4n D=1 (same super-pod) | 16 | **900.99** | 687.81 | 704.56 | 681.03 |
| 4n D=2 (cross-clique mixed) | 16 | **328.66** | 194.44 | 194.13 | 35.02 |
| 18n (72 GPU MNNVL) | 72 | **905.05** | 681.38 | 702.67 | 650.96 |
| 36n 2-CD (across 2 NVL72) | 144 | **688.14** | 704.13 | 699.75 | 40.59 * |

\* 36n alltoall across 2 ComputeDomains is a known NCCL issue (chain pollution); a single pass of vanilla is fine at 40 GB/s, but running a chain of 4 collectives together fails.

**Key findings**:
- Same-ComputeDomain 18-node all_reduce **905 GB/s**, close to the MNNVL fabric theoretical ceiling (~900 GB/s)
- StatefulSet self-driven vs standalone are performance-equivalent (+/-1% noise), saving 90% of yaml lines (114 vs 1129)
- DDP multi-node scaling is linear (same-domain sts vs standalone < 1% difference)

### 8.3 DeepEP v2 (test_ep)

| Scale | Dispatch SU avg | Combine SU avg | Combine latency avg | vs 1n |
|---|---|---|---|---|
| 1n (4 GPU) | 668 | 701 | 69 us | baseline |
| 2n MNNVL=0 (RDMA only) | 78 | 78 | 1001 us | - |
| 2n MNNVL=2 (production) | **605** | **653** | **100 us** | -7% BW |
| **16n** (64 GPU MNNVL=2) | **521** | **540** | **154 us** | -23% |

- Same-clique must use MNNVL=2 (NCCL GIB routed to MNNVL fabric, 8x the performance of RDMA)
- 16n combine 540 GB/s, 83% efficient relative to 2n=653 (8x scale-up loses only 17%)
- 144/144 test cases PASS, 0 errors

### 8.4 Megatron-LM training (5 configs, mock data, 50 iter, seq=4096, bf16)

| Model | GPU | Parallelism | ms/iter | tok/s/GPU | TFLOPs/GPU | MFU |
|---|---|---|---|---|---|---|
| llama2-7b  | 4  | tp1pp1 dp4 | 9406  | 27,872 | **1127** | 45.1% |
| llama2-7b  | 8  | tp1pp1 dp8 | 4881  | 26,861 | **1086** | 43.4% |
| llama2-13b | 4  | tp1pp1 dp4 | 18978 | 13,817 | **1079** | 43.2% |
| llama2-13b | 8  | tp1pp1 dp8 | 9614  | 13,636 | **1065** | 42.6% |
| llama3-70b | 64 | tp4pp2 dp8 | 28355 | 2,312  | **971**  | **38.9%** |

**Key observations**:
- 7B/13B hold steady at **42-45% MFU**, DDP multi-node scaling 96-99% (MNNVL fabric NVLink has near-zero cross-host overhead)
- 70B tp4pp2 drops to **39% MFU** (PP=2 pipeline bubble + larger attention share on a bigger model)
- selective recompute / FP8 / longer seq not yet enabled, leaving 5-10% headroom

---

## 9. Script Index

### `scripts/host/` -- Host-level operations (no pods)

| Script | Purpose |
|---|---|
| `image-build.sh` | **GCE custom image baking** (7 Phases + verify). Bakes in kernel cmdline / IMEX / sysctl / NIC rename / systemd |
| `cublas_bench_gb200.sh` | Run cuBLAS GEMM peak directly on host (6 dtypes). Auto-borrows CUDA libs by ctr-mounting a container image |
| `create-worker-vm-prod.sh` | Bulk `gcloud compute instances create` for worker VMs (including metadata injection) |
| `startup-forrest-gb200-k8s134-prod.sh` | All-in-one prod startup reference (including IMEX initramfs + auto-join, for trimming reference only) |
| `node-join.sh` | Local node kubeadm join (for debug) |
| `refresh-join-token.sh` | Generate a new 24h kubeadm join token on the master |
| `env.sh` | Shared env vars (project / zone / cluster name) |
| `gx` | Mini jump-host tool (only supports the `k8n` alias, SSH to master via env config) |

### `scripts/post-install/` -- Install GPU Stack after k8s Ready

| Script | Purpose |
|---|---|
| `install-device-plugin.sh` | helm install NVIDIA device-plugin 0.19.2 + GPU Feature Discovery |
| `install-dranet.sh` | helm install DRANET (k8s 1.34 RDMA DRA driver) |
| `install-dra-gpu-driver.sh` | helm install NVIDIA DRA GPU Driver v0.4 (ComputeDomain CRD) |
| `apply-rdma-deviceclass.sh` | Apply RDMA DeviceClass (4 NIC bond2-5 for ResourceClaim) |
| `verify-worker.sh` | Verify label / GPU count / IMEX / fabric state across all workers |

### `scripts/benchmark/` -- Benchmark tools

| Script | Purpose |
|---|---|
| `gen-nccl-multinode-sts.py` | NCCL test yaml generator (StatefulSet). Params: N node, D domain, mode (4coll/3coll/alltoall) |
| `gen-deepep-sts.py` | DeepEP v2 yaml generator (sts + ComputeDomain). N>=2 uses outside-launch mode |
| `gen-megatron-sts.py` | Megatron-LM yaml generator. 5 configs (llama2-7b/13b + llama3-70b), self-driven torchrun |
| `run-deepep.sh` | DeepEP outside launcher (kubectl exec parallel) |
| `run-nccl-test.sh` | NCCL test manual launcher (mpirun) |
| `extract-deepep-stats.py` | DeepEP log parser (outputs min/max/avg/count + latency) |
| `extract-megatron-stats.py` | Megatron iter log parser (elapsed_ms / TFLOPs / tokens-per-sec / loss) |
| `build-megatron-image.sh` | Build Megatron image locally on a node (buildah, NGC base ~6 min) |

### `scripts/troubleshooting/` -- Day-2 Ops

| Script | Purpose |
|---|---|
| `check-k8s-dra-health.sh` | **13-item cluster health check** (control-plane / worker / GPU / device-plugin / DRA controller / kubelet-plugin / CD ResourceSlice / DRANET / clique label / residual RC-CD / daemon DS scale / Xid / workload pod). `--fix-rs` fixes ResourceSlice, `--fix-race` fixes the controller race |
| `prepare-workload.sh` | **One-click workload apply wrapper**. Cleanup stale daemon + apply + fix-race + wait Running + second retry. Measured 16min stuck -> 4min Ready |
| `xid-fault-auto-recover.sh` | **Xid 137/145/94 auto-recovery**. Detect -> verify -> serial reset -> cleanup -> verify GPU=4. Serial by default to avoid IMEX cascade |
| `dra-cd-label-reconciler.sh` | DRA CD label reconciler standalone (`--once` single run / `--interval=15` daemon mode). Also available as a cluster-resident deployment |
