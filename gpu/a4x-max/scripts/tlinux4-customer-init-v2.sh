#!/bin/bash
# Compatible with: A4X/GB200, A4X MAX/GB300, H100, CPU-only instances
#
# v5.3 镜像已内置: NVIDIA driver, GVE, IDPF, IMEX, GDRCopy, nvidia-persistenced,
#                   SELinux/firewalld disabled, memlock unlimited, sudoers, SSH
#
# 本脚本仅负责: 文件系统初始化 + sshd 22/56000 端口 + 网卡自动改名 (bond0-N)
#
# 网卡改名策略 (自动适配 GB200/GB300):
#   管理网卡 (GVNIC/IDPF): 主网卡(默认路由)→bond0, 其余按PCI地址排序→bond1, ...
#   RDMA IB设备 (mlx5_X):  按设备名排序, 接续管理网卡编号→bondN, bondN+1, ...
#   GB200 结果: bond0-1 (2×GVNIC) + bond2-5 (4×CX-7)  = 6 interfaces
#   GB300 结果: bond0-1 (2×IDPF)  + bond2-9 (8×CX-8)  = 10 interfaces
set -euo pipefail
exec > >(tee /var/log/tlinux4-customer-init.log) 2>&1

INIT_STAMP=/etc/.customer-init-done

###############################################################################
# Network interface rename (auto-detect) — runs every boot
###############################################################################
if rdma link show 2>/dev/null | grep -q .; then
  echo "=== Network interface rename (auto-detect) ==="
  BOND_IDX=0

  # Detect primary NIC (carries default route)
  PRIMARY_NIC=$(ip route show default 2>/dev/null | awk '{print $5}' | head -1)

  # Collect physical ethernet NICs needing rename
  MGMT_NICS=()
  for devpath in /sys/class/net/*/device; do
    [ -e "$devpath" ] || continue
    nic=$(basename "$(dirname "$devpath")")
    [[ "$nic" == lo || "$nic" == bond* ]] && continue
    [[ -d "/sys/class/net/$nic/device/infiniband" ]] && continue
    MGMT_NICS+=("$nic")
  done

  if [ ${#MGMT_NICS[@]} -gt 0 ]; then
    # ---- First boot: rename management NICs ----

    # Primary NIC → bond0
    if [[ -n "$PRIMARY_NIC" ]]; then
      ip link set dev "$PRIMARY_NIC" down
      ip link set dev "$PRIMARY_NIC" name bond0
      ip link set dev bond0 up
      echo "  $PRIMARY_NIC → bond0 (primary)"
      BOND_IDX=1
    fi

    # Remaining NICs → bond1, bond2, ... sorted by PCI bus address
    SORTED=$(for n in "${MGMT_NICS[@]}"; do
      [[ "$n" == "$PRIMARY_NIC" ]] && continue
      pci=$(basename "$(readlink -f "/sys/class/net/$n/device")" 2>/dev/null || echo "zzzz")
      echo "$pci $n"
    done | sort | awk '{print $2}')

    for nic in $SORTED; do
      if ip link show "$nic" &>/dev/null; then
        ip link set dev "$nic" down
        ip link set dev "$nic" name "bond${BOND_IDX}"
        ip link set dev "bond${BOND_IDX}" up
        echo "  $nic → bond${BOND_IDX}"
        BOND_IDX=$((BOND_IDX + 1))
      fi
    done
    MGMT_COUNT=$BOND_IDX

    # Update NM connection profile for primary NIC
    if [[ -n "$PRIMARY_NIC" ]]; then
      for nmf in /etc/NetworkManager/system-connections/*.nmconnection; do
        [ -f "$nmf" ] || continue
        if grep -q "interface-name=${PRIMARY_NIC}" "$nmf" 2>/dev/null; then
          sed -i "s/interface-name=${PRIMARY_NIC}/interface-name=bond0/" "$nmf"
          sed -i "s/id=${PRIMARY_NIC}/id=bond0/" "$nmf"
          nmcli connection reload 2>/dev/null || true
          echo "  Updated NM profile: ${PRIMARY_NIC} → bond0"
          break
        fi
      done
    fi

    # Create udev rules for persistent naming across reboots (one-time)
    UDEV_RULE="/etc/udev/rules.d/70-a4x-bond-rename.rules"
    if [ ! -f "$UDEV_RULE" ]; then
      : > "$UDEV_RULE"
      for bidx in $(seq 0 $((MGMT_COUNT - 1))); do
        PCI=$(basename "$(readlink -f "/sys/class/net/bond${bidx}/device")" 2>/dev/null || true)
        [ -n "$PCI" ] && echo "SUBSYSTEM==\"net\", ACTION==\"add\", KERNELS==\"$PCI\", NAME=\"bond${bidx}\"" >> "$UDEV_RULE"
      done
      echo "  Created udev rules for ${MGMT_COUNT} management NICs"
    fi
  else
    # ---- Reboot: management NICs already named by udev ----
    while ip link show "bond${BOND_IDX}" &>/dev/null; do
      BOND_IDX=$((BOND_IDX + 1))
    done
    [ $BOND_IDX -gt 0 ] && echo "  Management NICs already renamed (bond0-bond$((BOND_IDX - 1)))"
    MGMT_COUNT=$BOND_IDX
  fi

  # ---- RDMA IB devices (runtime rename, needs every boot) ----
  for rdev in $(rdma dev 2>/dev/null | awk '{print $2}' | sed 's/:$//' | sort); do
    [[ "$rdev" == bond* ]] && continue
    if rdma dev set "$rdev" name "bond${BOND_IDX}" 2>/dev/null; then
      echo "  $rdev → bond${BOND_IDX}"
      BOND_IDX=$((BOND_IDX + 1))
    fi
  done

  RDMA_COUNT=$((BOND_IDX - MGMT_COUNT))
  echo "  --- Result: ${MGMT_COUNT} mgmt + ${RDMA_COUNT} RDMA = ${BOND_IDX} total ---"
  ip -br link show type ether 2>/dev/null || true
  rdma link show 2>/dev/null || true
fi

if [ -f "$INIT_STAMP" ]; then
  echo "Customer initialization already completed. Skipping."
  exit 0
fi

echo "================================================================"
echo "  TLinux 4 v5.3 — Startup Script (v2 Universal)"
echo "  $(date)"
echo "================================================================"

###############################################################################
# [1] Boot disk partitioning: /usr/local (20G) + /data (remaining)
###############################################################################
echo "=== [1] Boot disk partitioning ==="

BOOT_DEV=$(findmnt -n -o SOURCE / | sed 's/p[0-9]*$//')
PARTITION_STAMP=/etc/.partitions-created

if [ ! -f "$PARTITION_STAMP" ]; then
  echo "--- Creating boot disk partitions (1TB disk) ---"

  sgdisk -e "$BOOT_DEV" 2>/dev/null || true

  sgdisk -n 3:0:+20G -t 3:8300 "$BOOT_DEV"
  sgdisk -n 4:0:0 -t 4:8300 "$BOOT_DEV"
  partprobe "$BOOT_DEV" 2>/dev/null || true
  sleep 2

  mkfs.xfs -f "${BOOT_DEV}p3"
  mkfs.xfs -f "${BOOT_DEV}p4"

  TMPMNT=$(mktemp -d)
  mount "${BOOT_DEV}p3" "$TMPMNT"
  cp -a /usr/local/. "$TMPMNT/" 2>/dev/null || true
  umount "$TMPMNT"
  rmdir "$TMPMNT"

  mount "${BOOT_DEV}p3" /usr/local
  mkdir -p /data
  mount "${BOOT_DEV}p4" /data

  P3_UUID=$(blkid -s UUID -o value "${BOOT_DEV}p3")
  P4_UUID=$(blkid -s UUID -o value "${BOOT_DEV}p4")
  if [ -n "$P3_UUID" ] && ! grep -q "$P3_UUID" /etc/fstab; then
    echo "UUID=$P3_UUID /usr/local xfs defaults,nofail 0 2" >> /etc/fstab
  fi
  if [ -n "$P4_UUID" ] && ! grep -q "$P4_UUID" /etc/fstab; then
    echo "UUID=$P4_UUID /data xfs defaults,nofail 0 2" >> /etc/fstab
  fi

  touch "$PARTITION_STAMP"
  echo "Boot disk partitioned: p3=/usr/local(20G) p4=/data(remaining)"
fi

###############################################################################
# [2] Local SSD RAID0 → /mnt/stateful_partition
###############################################################################
echo "=== [2] Local SSD RAID0 setup ==="

SSD_DEVICES=()
for disk in /dev/nvme*n1; do
  [ -b "$disk" ] || continue
  PART_COUNT=$(lsblk -n -o NAME "$disk" 2>/dev/null | wc -l)
  if [ "$PART_COUNT" -le 1 ]; then
    SSD_DEVICES+=("$disk")
    echo "  Detected local SSD: $disk ($(lsblk -n -d -o SIZE "$disk"))"
  fi
done

NUM_SSDS=${#SSD_DEVICES[@]}
echo "  Found $NUM_SSDS local SSD device(s)"

if [ "$NUM_SSDS" -gt 0 ]; then
  MOUNT_POINT="/mnt/stateful_partition"
  mkdir -p "$MOUNT_POINT"

  if [ "$NUM_SSDS" -eq 1 ]; then
    TARGET_DEV="${SSD_DEVICES[0]}"
    mkfs.xfs -f "$TARGET_DEV"
    mount "$TARGET_DEV" "$MOUNT_POINT"
    DEV_UUID=$(blkid -s UUID -o value "$TARGET_DEV")
    if [ -n "$DEV_UUID" ] && ! grep -q "$DEV_UUID" /etc/fstab; then
      echo "UUID=$DEV_UUID $MOUNT_POINT xfs defaults,nofail 0 2" >> /etc/fstab
    fi
  else
    mdadm --stop /dev/md0 2>/dev/null || true
    mdadm --create /dev/md0 --level=0 --raid-devices="$NUM_SSDS" "${SSD_DEVICES[@]}" --force --run
    mkfs.xfs -f /dev/md0
    mount /dev/md0 "$MOUNT_POINT"
    mkdir -p /etc/mdadm
    mdadm --detail --scan >> /etc/mdadm/mdadm.conf
    DEV_UUID=$(blkid -s UUID -o value /dev/md0)
    if [ -n "$DEV_UUID" ] && ! grep -q "$DEV_UUID" /etc/fstab; then
      echo "UUID=$DEV_UUID $MOUNT_POINT xfs defaults,nofail 0 2" >> /etc/fstab
    fi
  fi

  mkdir -p "$MOUNT_POINT"/{gcsfuse-cache,scratch-data}
  echo "$MOUNT_POINT mounted (total: $(df -h "$MOUNT_POINT" | awk 'NR==2{print $2}'))"
else
  echo "WARNING: No local SSD devices found"
  mkdir -p /mnt/stateful_partition
fi

###############################################################################
# [3] sshd port 22 + 56000
###############################################################################
echo "=== [3] sshd port 22 + 56000 ==="

SSHD_CHANGED=0
if ! grep -q '^Port 22' /etc/ssh/sshd_config; then
  sed -i '1i Port 22' /etc/ssh/sshd_config
  SSHD_CHANGED=1
fi
if ! grep -q '^Port 56000' /etc/ssh/sshd_config; then
  sed -i '/^Port 22/a Port 56000' /etc/ssh/sshd_config
  SSHD_CHANGED=1
fi
if [ "$SSHD_CHANGED" -eq 1 ]; then
  systemctl restart sshd
  echo "sshd now listening on port 22 and 56000"
else
  echo "sshd ports 22 and 56000 already configured"
fi

###############################################################################
# Verification
###############################################################################
echo ""
echo "================================================================"
echo "  Verification"
echo "================================================================"

echo "--- OS ---"
cat /etc/os-release | head -3
echo "Kernel: $(uname -r)"

if lspci 2>/dev/null | grep -qi 'NVIDIA'; then
  echo ""
  echo "--- GPU ---"
  nvidia-smi -L 2>&1 | head -4
  echo "Persistence: $(nvidia-smi -q 2>/dev/null | grep 'Persistence Mode' | head -1 || echo N/A)"
  echo "IMEX: $(ls /dev/nvidia-caps-imex-channels/ 2>&1)"
fi

echo ""
echo "--- Network ---"
ip -br link show type ether 2>/dev/null || true
echo "RDMA: $(rdma link show 2>/dev/null | awk '{print $2}' | tr '\n' ', ' || echo N/A)"

echo ""
echo "--- Disk ---"
df -h /usr/local /data /mnt/stateful_partition 2>/dev/null | grep -v Filesystem

echo ""
echo "--- SSH ---"
echo "Ports: $(grep '^Port' /etc/ssh/sshd_config | awk '{print $2}' | tr '\n' ', ')"

touch "$INIT_STAMP"
echo ""
echo "================================================================"
echo "  Startup complete at $(date)"
echo "================================================================"
