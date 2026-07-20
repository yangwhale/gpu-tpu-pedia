# GB300 (A4X Max) · Local SSD RAID 0 挂载指南

> 给 V4 大模型（Pro 1.6T ~800G）当**快速本地存储**——替代内存盘（tmpfs 吃 pod RAM），用 4 块 Local NVMe SSD 做 RAID 0，读 ~14 GB/s、写 ~20 GB/s、12T 容量。
> 2026-07-20 在 `gb300-pool-0010`（`a4x-maxgpu-4g-metal`）实测跑通 + 幂等验证。

---

## 0. 问题根因（先搞清楚"为什么 GKE 没自动挂好"）

**误区**：以为 GKE 会自动把 Local SSD 格式化 + 挂载好。
**真相**：本池创建时用的是 **`localNvmeSsdBlockConfig: localSsdCount: 4`**（raw block 模式），GKE **故意**把 4 块 Local SSD 以**裸块设备**暴露（`/dev/disk/by-id/google-local-ssd-block0..3`），**不格式化、不挂载**——留给用户自己 RAID + 格式化。

> 对比：若池用 `--ephemeral-storage-local-ssd`（ephemeral 模式），GKE 才会自动 RAID + 挂到 ephemeral storage。本池是 block 模式，所以要自己弄。

物理盘实况（`lsblk`）：
```
nvme0n1  2.9T   ← google-local-ssd-block0   (裸, 无 fstype/mount)
nvme2n1  2.9T   ← google-local-ssd-block2
nvme3n1  2.9T   ← google-local-ssd-block1
nvme4n1  2.9T   ← google-local-ssd-block3
nvme1n1  100G   ← 启动盘 (含 /mnt/stateful_partition, 别碰)
```
4 × 2.9T = ~11.6T 裸容量 → RAID 0 → **12T**。

---

## 1. 关键坑：官方 DaemonSet 是 amd64，GB300 是 arm64

GKE 官方给的 RAID DaemonSet（`kubernetes-sigs/sig-storage-local-static-provisioner` 的 `gke-daemonset-raid-disks.yaml`）用镜像 `registry.k8s.io/startup-script:v2`，**只有 amd64**。GB300 是 **Grace ARM CPU**，直接跑报 `exec /bin/sh: exec format error`。

**修法**：换多架构镜像 `ubuntu:24.04`（arm64 可用）+ **`mountPropagation: Bidirectional`** 把容器内的挂载传播到 host，容器内自带 `mdadm`/`mkfs.ext4` 跑 RAID 脚本。

---

## 2. 部署（arm64 RAID DaemonSet）

`/tmp/gke-raid-ds.yaml`（scope 到 pool-0010，幂等）：
```yaml
apiVersion: apps/v1
kind: DaemonSet
metadata: {name: gke-raid-disks, namespace: default, labels: {k8s-app: gke-raid-disks}}
spec:
  selector: {matchLabels: {name: gke-raid-disks}}
  template:
    metadata: {labels: {name: gke-raid-disks}}
    spec:
      nodeSelector:
        cloud.google.com/gke-local-nvme-ssd: "true"
        cloud.google.com/gke-nodepool: gb300-pool-0010   # scope，避免碰其他池
      tolerations: [{operator: Exists}]
      hostPID: true
      containers:
      - name: raid
        image: ubuntu:24.04                # arm64 多架构，替代 amd64 的 startup-script:v2
        securityContext: {privileged: true}
        command: ["/bin/bash","-c"]
        args:
        - |
          set -o pipefail
          export DEBIAN_FRONTEND=noninteractive
          apt-get update -qq && apt-get install -y -qq mdadm e2fsprogs >/dev/null 2>&1
          devices=()
          for ssd in /dev/disk/by-id/google-local-ssd-block*; do [ -e "$ssd" ] && devices+=("$ssd"); done
          if [ "${#devices[@]}" -eq 0 ]; then echo "NO_LOCAL_SSD"; sleep infinity; fi
          # 幂等：已有 md0 就不重建
          if ! grep -q "md0" /proc/mdstat; then
            echo "y" | mdadm --create /dev/md0 --level=0 --force --raid-devices=${#devices[@]} "${devices[@]}"
          fi
          # 幂等：已格式化就不重做
          tune2fs -l /dev/md0 >/dev/null 2>&1 || mkfs.ext4 -F /dev/md0
          mkdir -p /mnt/disks/raid/0
          mountpoint -q /mnt/disks/raid/0 || mount -o discard,defaults /dev/md0 /mnt/disks/raid/0
          chmod a+w /mnt/disks/raid/0
          echo "RAID_READY: $(df -h /mnt/disks/raid/0 | tail -1)"
          sleep infinity
        volumeMounts:
        - {name: dev, mountPath: /dev}
        - {name: raid, mountPath: /mnt/disks/raid, mountPropagation: Bidirectional}  # 关键
      volumes:
      - {name: dev, hostPath: {path: /dev}}
      - {name: raid, hostPath: {path: /mnt/disks/raid, type: DirectoryOrCreate}}
```
```bash
kubectl apply -f /tmp/gke-raid-ds.yaml
kubectl get ds gke-raid-disks      # DESIRED=CURRENT=READY=17（pool 全节点）
kubectl logs <ds-pod>              # 看到 RAID_READY: /dev/md0 12T ... /mnt/disks/raid/0
```

---

## 3. 消费者 pod 用法（V4 pod 将来照此挂）

hostPath 挂 `/mnt/disks/raid/0`，propagation 用 `HostToContainer`：
```yaml
    volumeMounts:
    - {name: raid, mountPath: /data, mountPropagation: HostToContainer}
  volumes:
  - {name: raid, hostPath: {path: /mnt/disks/raid/0, type: Directory}}
```
消费者 pod 内 `df -h /data` 应见 `/dev/md0 12T`，读写正常。

---

## 4. 实测（DS 管理的 RAID，consumer pod fio）

| 指标 | 值 |
|---|---|
| 容量 | 12T（4×2.9T RAID0）|
| 顺序写（4 job 1M）| **20.2 GiB/s（21.7 GB/s）** |
| 顺序读（4 job 1M）| **13.9 GiB/s（15.0 GB/s）** |
| 加载 V4 Pro 800G 估时 | ~57s（@14GB/s 读）|

**幂等性验证**：删掉某节点的 DS pod 让它重启 → hello.txt 数据存活、md0 仍挂载、不重格式化。✅

---

## 5. 坑速查

| 现象 | 根因 | 解 |
|---|---|---|
| 以为 GKE 自动挂好，结果盘是裸的 | 池用 `local-nvme-ssd-block`（raw 模式，设计如此）| 自己 RAID + 格式化（本文）|
| DS pod `exec format error` | `startup-script:v2` 是 amd64，GB300 是 arm64 | 换 `ubuntu:24.04` + mountPropagation |
| 消费者 pod 看不到挂载 / debug pod 看不到 | 挂载在 pod 启动后建立，propagation 默认不传入 | DS 用 `Bidirectional`，消费者用 `HostToContainer` |
| RAID 跨 4 盘但选错设备 | 别把启动盘 nvme1n1 算进去 | 只用 `google-local-ssd-block*`（by-id）|
| 重启/repave 后数据丢 | Local SSD 是 ephemeral，reboot 数据不保 | 正常；DS 会自动重建 RAID，模型需从 GCS 重拉 |
| **部分节点 RAID 建不成，pod /mnt/ssd 是 256K tmpfs 不是 12T**（模型写进去变空，加载报 `Unrecognized model`）| **节点残留污染（A 类：systemd mount）**：之前的部署在这几台留了个 systemd 单元 `mnt-disks-ssdN.mount`，把一块 local SSD 单独挂在 `/mnt/disks/ssd0`，占着盘 → `mdadm --create` 报 `Device or resource busy` → RAID 建不成 → DS 脚本没 `set -e` 还 sleep infinity 掩盖失败 → 消费者 pod hostPath 落到 `/mnt/disks` 的 256K tmpfs | **干净节点无此问题**（新拉的节点 4 盘全裸，RAID 必成）。检测：`grep -c md0 /proc/mdstat` 逐节点扫，=0 即坏。修：**重建污染节点**（`gcloud compute instances delete` → 池拉新的干净节点）最可靠——live 卸载 systemd mount 会被立刻重挂、设备仍 busy，压不干净。诊断链：pod 里 `df /mnt/ssd` 见 256K tmpfs → `grep md0 /proc/mdstat` 空 → `mdadm --create` 报 block busy → host `/proc/1/mounts` 见 `/dev/nvmeXn1 /mnt/disks/ssd0` → `systemctl cat mnt-disks-ssd0.mount`。|
| **同上现象，但 `/proc/mdstat` 里 md0 存在却是 `inactive`**（2026-07-20 pool-0002 lt06 + pool-0001 61jb 实测，18 节点里各碰 1 台）| **节点残留污染（B 类：mdadm 旧 superblock）**：4 块 SSD 上有**上一次 RAID 留下的 mdadm 1.2 metadata**，开机 mdadm 自动组装时按旧 UUID 把盘**拆成坏数组**（`md0` + `md127`，全 inactive `(S)`）→ DS 脚本 `grep -q md0` 判定「已存在」跳过 create → `mount` 报 `can't read superblock on /dev/md0` → 无 `set -e` 继续 → 假 `RAID_READY`（其实挂的是 256K tmpfs）| **B 类能 live 清，不用重建节点**（区别于 A 类）。检测：`grep md0 /proc/mdstat` 见 `inactive`。修（在该节点 RAID DS pod 里跑，privileged + Bidirectional 传播回 host）：`umount`→`mdadm --stop` 所有 `/dev/md*`→**`sleep 1`**（关键，等 stop 落定）→`for d in google-local-ssd-block*; do mdadm --zero-superblock --force $d; done`（**抹掉旧 superblock**）→ `mdadm --create ... --run --raid-devices=4` + `mkfs.ext4 -F` + `mount`。修完删该节点消费者/puller pod 让它重挂。**⚠️ udev race**：`mdadm --stop` 后 udev 会**立刻**按旧 superblock 重组 inactive 数组，抢在 zero 之前 → create 报 `Device or resource busy`。对策：stop 后先 `sleep 1` 再**一次性 zero 全部 4 盘**；仍失败就重跑 1-2 次（lt06 一次成，61jb 两次成）。|

> **DS 脚本硬化建议（两类污染都能早暴露）**：把 create/mount 段前加 `set -e`，且 create 前判断应改为 **`mdadm --detail /dev/md0 | grep -q "State : clean\|active"`**（不是裸 `grep md0`——inactive 的 md0 也匹配裸 grep，导致跳过 create 掩盖 B 类污染）。这样坏节点会 CrashLoop 立刻可见，而不是假 ready。

---

## 6. 给 V4 用的下一步

- V4 pod（Flash/Pro）把模型从 GCS 拉到 `/data`（= /mnt/disks/raid/0），替代内存盘 tmpfs → 省下 800G RAM。
- V4 测试计划见 `./sglang-v4-gb300-TEST-PLAN.md`。

---

**建议硬化 DaemonSet**：脚本加 `set -e`（RAID 建失败就 crash、pod CrashLoop 可见，而不是 sleep infinity 静默掩盖）；这样污染节点会立刻暴露、便于定位。**干净节点上本 DaemonSet 100% 可靠**（pool-0007 实测 17/17 全成），失败只发生在有残留 systemd mount 的污染节点。

---

*2026-07-20 实测跑通。干净池 **pool-0007 17/17 节点 RAID 全成**；`gb300-pool-0010` 有 4 台被之前部署残留的 `mnt-disks-ssd0.mount` 污染（已 recreate）。机型 `a4x-maxgpu-4g-metal` / 4× local NVMe SSD block。*
