#!/bin/bash
# =============================================================================
# image-build.sh — GB200 worker base image build (run-once before snapshot)
#
# 把所有 host 级硬件配置一次性写进 OS image。
# 在一台干净的 GB200 worker VM 上跑此脚本 → 重启 → snapshot disk → create custom image。
# 新 worker VM 用此 image, **无需 startup-script, 无需传 metadata**, 镜像内 systemd 服务每开机自动跑所有 tuning。
#
# 前提:
#   • Base OS: TLinux 4 ARM64 (自带 R580+ NVIDIA driver + nvidia-imex systemd unit,
#                              直接 nvidia-smi 即可,本脚本不再装 driver)
#   • 干净 VM 还未 join 任何 k8s cluster
#
# 流程 (7 个 Phase):
#   Phase 1   OS base 软件包 + sshd + selinux + filesystem 扩展
#   Phase 2   Kernel cmdline (init_on_alloc=0, 移除 iommu.passthrough)
#   Phase 3   IMEX initramfs (NVreg_CreateImexChannel0=1 + dracut --add-drivers gve)
#   Phase 4   通用 prereqs (modules-load, sysctl br_netfilter / ip_forward — 容器运行时通用)
#   Phase 5   Grace GB200 持久 sysctl (numa_balancing=0, BBR, TCP buffer, vm.*)
#   Phase 6   Grace 一次性 host tuning systemd reapply.service 注册
#               (含 PCI ACS / CPU governor / THP / nvidia_peermem / nvidia-smi -pm 1)
#   Phase 7   NIC rename → bond0..bond5 (PCI BDF-based systemd .link, 一次性 baked, A4X 硬件 BDF 固定)
#
# 不在此脚本内 (客户自己装):
#   • Container runtime (containerd / docker / podman) + NRI + CNI plugins
#   • nvidia-container-toolkit
#   • k8s 客户端 (kubelet / kubeadm / kubectl)
#   • k8s 控制平面 + DRA driver + ComputeDomain stack
#
# 无需 first-boot.sh / startup-script / metadata — 所有 host tuning 都在镜像内 systemd 服务里。
# 客户在节点起来后自己装 container runtime / kubelet, 用自己的方式 kubeadm join。
#
# 用法:
#   sudo bash image-build.sh
#   sudo reboot              # let IMEX initramfs + cmdline + NIC rename rules take effect
#   sudo bash image-build.sh --verify     # post-reboot sanity check
#   # 停 VM → snapshot disk → create custom image
#   gcloud compute disks snapshot <DISK> --snapshot-names=<NAME> --zone=<ZONE>
#   gcloud compute images create <IMAGE-NAME> --source-snapshot=<NAME>
# =============================================================================
set -euo pipefail
exec > >(tee -a /var/log/gb200-image-build.log) 2>&1

# ===== --verify mode (post-reboot sanity check, no install) =====
if [ "${1:-}" = "--verify" ]; then
  echo "=== image-build verify (post-reboot) ==="
  echo "  init_on_alloc: $(cat /proc/cmdline | grep -oE 'init_on_alloc=[0-9]' || echo MISSING)"
  echo "  iommu params:  $(cat /proc/cmdline | grep -oE 'iommu\.[a-z]+=[0-9]' || echo none ✓)"
  echo "  IMEX modprobe: $(cat /etc/modprobe.d/nvidia.conf 2>/dev/null || echo MISSING)"
  echo "  IMEX channels: $(ls /dev/nvidia-caps-imex-channels/ 2>&1 | head -1)"
  echo "  numa_balancing: $(cat /proc/sys/kernel/numa_balancing)  (expect: 0)"
  echo "  TCP congestion: $(cat /proc/sys/net/ipv4/tcp_congestion_control)  (expect: bbr)"
  echo "  nvidia-imex systemd: $(systemctl is-enabled nvidia-imex 2>/dev/null) (expect: disabled — CD daemon 接管)"
  echo "  grace-reapply: $(systemctl is-enabled grace-gb200-reapply 2>/dev/null) (expect: enabled)"
  echo "  NIC .link files: $(ls /etc/systemd/network/10-bond*.link 2>/dev/null | wc -l) (expect: 6)"
  echo "  NIC names:     $(ls /sys/class/net | grep -E '^bond[0-9]+$' | sort | tr '\n' ' ')"
  echo "  irqbalance:    $(systemctl is-enabled irqbalance 2>/dev/null) (expect: disabled)"
  echo "  IMEX channels: $(ls /dev/nvidia-caps-imex-channels/ 2>&1 | head -1)"
  echo "  GPU persist:   $(nvidia-smi --query-gpu=persistence_mode --format=csv,noheader | head -1) (expect: Enabled)"
  echo "  Fabric State:  $(nvidia-smi -q 2>/dev/null | awk '/^    Fabric/{f=1} f && /State/{print $NF; exit}') (expect: Completed)"
  exit 0
fi

###############################################################################
# Phase 1: OS base
###############################################################################
echo "=== [1] OS base 软件包 + sshd + selinux + filesystem ==="
dnf install -y sudo cloud-utils-growpart xfsprogs jq pciutils nvme-cli util-linux \
  dnf-plugins-core ethtool python3 curl 2>/dev/null

echo '%google-sudoers ALL=(ALL:ALL) NOPASSWD:ALL' >/etc/sudoers.d/google_sudoers
chmod 440 /etc/sudoers.d/google_sudoers
systemctl enable --now sshd
sed -i 's/^#*PermitRootLogin.*/PermitRootLogin yes/' /etc/ssh/sshd_config
systemctl restart sshd

cat >/etc/security/limits.d/rdma.conf <<EOF
* soft memlock unlimited
* hard memlock unlimited
EOF

setenforce 0 2>/dev/null || true
sed -i -E 's/^SELINUX=(enforcing|permissive)$/SELINUX=disabled/' /etc/selinux/config 2>/dev/null || true

# root partition growpart (idempotent)
BOOT_DEV=$(findmnt -n -o SOURCE / | sed 's/p[0-9]*$//')
PART_NUM=$(findmnt -n -o SOURCE / | grep -o '[0-9]*$')
growpart "${BOOT_DEV}" "${PART_NUM}" 2>/dev/null || true
xfs_growfs / 2>/dev/null || resize2fs "$(findmnt -n -o SOURCE /)" 2>/dev/null || true

###############################################################################
# Phase 2: Kernel cmdline (init_on_alloc=0, NO iommu.passthrough)
#   iommu.passthrough=1 break UVM ATS bind, must NOT be set for GB200
###############################################################################
echo "=== [2] Kernel cmdline: init_on_alloc=0 ==="
if command -v grubby >/dev/null 2>&1; then
  grubby --update-kernel=ALL --remove-args="iommu.passthrough iommu.strict" 2>/dev/null || true
  grubby --update-kernel=ALL --args="init_on_alloc=0"
else
  if [ -f /etc/default/grub ] && ! grep -q 'init_on_alloc=0' /etc/default/grub; then
    sed -i -E 's/iommu\.(passthrough|strict)=[01] //g' /etc/default/grub
    sed -i 's|^\(GRUB_CMDLINE_LINUX="\)|\1init_on_alloc=0 |' /etc/default/grub
    grub2-mkconfig -o /boot/grub2/grub.cfg 2>/dev/null ||
      grub2-mkconfig -o /boot/efi/EFI/tlinux/grub.cfg 2>/dev/null ||
      echo "  ⚠ grub2-mkconfig 失败, 人工 check"
  fi
fi

###############################################################################
# Phase 3: IMEX initramfs (NVreg_CreateImexChannel0=1 + dracut --add-drivers gve)
#   nvidia-imex 需要 /dev/nvidia-caps-imex-channels 设备, NVreg_* 参数让 nvidia 模块创建
#   gve 驱动加进 initramfs 让早期 boot 阶段网卡可用
###############################################################################
echo "=== [3] IMEX initramfs config + dracut rebuild ==="
echo "options nvidia NVreg_CreateImexChannel0=1" >/etc/modprobe.d/nvidia.conf
dracut --force --add-drivers "gve"
echo "  ✓ initramfs rebuilt; reboot 后 /dev/nvidia-caps-imex-channels/ 会出现"

###############################################################################
# Phase 4: k8s prereqs (modules + sysctl)
###############################################################################
echo "=== [4] k8s prereqs ==="
swapoff -a
sed -i '/swap/d' /etc/fstab

cat >/etc/modules-load.d/k8s.conf <<EOF
overlay
br_netfilter
EOF
modprobe overlay
modprobe br_netfilter

cat >/etc/sysctl.d/k8s.conf <<EOF
net.bridge.bridge-nf-call-iptables  = 1
net.bridge.bridge-nf-call-ip6tables = 1
net.ipv4.ip_forward                 = 1
EOF

###############################################################################
# Phase 5: Grace GB200 持久 sysctl (numa_balancing=0, BBR, TCP buffer, vm.*)
###############################################################################
echo "=== [5] Grace GB200 持久 sysctl ==="
cat >/etc/sysctl.d/90-grace-gb200.conf <<'EOF'
kernel.numa_balancing = 0
net.core.default_qdisc          = fq
net.ipv4.tcp_congestion_control = bbr
net.core.rmem_max               = 134217728
net.core.wmem_max               = 134217728
net.ipv4.tcp_rmem               = 4096 87380 134217728
net.ipv4.tcp_wmem               = 4096 65536 134217728
net.core.netdev_max_backlog     = 30000
net.core.somaxconn              = 4096
vm.max_map_count                = 1048576
vm.dirty_background_bytes       = 67108864
vm.dirty_bytes                  = 268435456
vm.min_free_kbytes              = 524288
fs.file-max                     = 2097152
fs.aio-max-nr                   = 1048576
fs.inotify.max_user_watches     = 524288
fs.inotify.max_user_instances   = 8192
EOF
sysctl --system >/dev/null 2>&1

###############################################################################
# Phase 6: Grace 一次性 host tuning systemd reapply.service
#   PCI ACS / CPU governor / THP / nvidia_peermem 重启不持久, 注册 systemd unit 每次开机重新跑
###############################################################################
echo "=== [6] Grace reapply systemd service ==="
cat >/usr/local/sbin/grace-gb200-reapply.sh <<'REAPPLY'
#!/bin/bash
set +e
for BDF in $(lspci -d "*:*:*" | awk '{print $1}'); do
  setpci -v -s "${BDF}" ECAP_ACS+0x6.w >/dev/null 2>&1 || continue
  setpci -v -s "${BDF}" ECAP_ACS+0x6.w=0000 >/dev/null 2>&1
done
for g in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do
  [ -w "$g" ] && echo performance > "$g"
done
echo madvise         > /sys/kernel/mm/transparent_hugepage/enabled 2>/dev/null || true
echo "defer+madvise" > /sys/kernel/mm/transparent_hugepage/defrag  2>/dev/null || true
lsmod | grep -qE nvidia_peermem || modprobe nvidia_peermem 2>/dev/null || true
nvidia-smi -pm 1 2>/dev/null || true
exit 0
REAPPLY
chmod +x /usr/local/sbin/grace-gb200-reapply.sh

cat >/etc/systemd/system/grace-gb200-reapply.service <<'UNIT'
[Unit]
Description=Reapply Grace GB200 host tuning (PCI ACS / CPU gov / THP / peermem / GPU persist)
After=multi-user.target
[Service]
Type=oneshot
ExecStart=/usr/local/sbin/grace-gb200-reapply.sh
RemainAfterExit=yes
[Install]
WantedBy=multi-user.target
UNIT
systemctl daemon-reload
systemctl enable grace-gb200-reapply.service

# Run once now (sysfs writes are runtime-effective, OK to run during bake)
bash /usr/local/sbin/grace-gb200-reapply.sh

# disable host nvidia-imex (let ComputeDomain daemon 接管 IMEX session, 避免 NV_ERR_IN_USE)
systemctl disable --now nvidia-imex 2>/dev/null || true
echo "  ✓ host nvidia-imex disabled (CD daemon 在 workload 起来时接管)"

###############################################################################
# Phase 7: NIC rename → bond0..bond5 (PCI BDF-based systemd .link, 一次性 baked)
#   原方案用 MAC 匹配 udev rule, MAC 每 VM 不同 → 必须 first-boot 生成。
#   改用 PCI BDF (Path) 匹配 systemd .link: A4X NVL72 硬件 topology 固定,
#   所有 A4X VM 的 PCI BDF 一致, image build 时一次性生成, 客户 VM 直接 work。
###############################################################################
echo "=== [7] NIC rename → bond0..bond5 (PCI BDF-based systemd .link, 一次性 baked) ==="
# 清旧 MAC 规则 (如果有遗留)
rm -f /etc/udev/rules.d/80-bond-rename.rules 2>/dev/null
rm -f /etc/systemd/network/10-bond*.link 2>/dev/null

i=0
for net_dir in $(ls -d /sys/class/net/*/device 2>/dev/null | sort -V); do
  nic_path=${net_dir%/device}
  nic_name=$(basename $nic_path)
  # skip 非 GPU host NIC
  case "$nic_name" in lo | docker* | cni* | flannel* | vxlan* | virbr* | tailscale* | bond*) continue ;; esac
  drv_link=$(readlink "$net_dir/driver" 2>/dev/null) || continue
  drv=$(basename "$drv_link")
  case "$drv" in
  mlx5_core | gve)
    pci_bdf=$(basename $(readlink -f "$net_dir"))
    LINK_FILE=$(printf '/etc/systemd/network/10-bond%d.link' $i)
    cat >"$LINK_FILE" <<EOF
[Match]
Path=pci-$pci_bdf

[Link]
Name=bond$i
EOF
    echo "  $nic_name (driver=$drv, PCI=$pci_bdf) → bond$i"
    i=$((i + 1))
    ;;
  esac
done
echo "  ✓ $i NIC .link 文件已生成 (image build 末尾 reboot 后自动生效)"

###############################################################################
# Phase 8: NIC ring + channel + NVMe coalescing
###############################################################################
echo "=== [8] NIC ring + NVMe tuning ==="
systemctl disable --now irqbalance 2>/dev/null && echo "  ✓ irqbalance disabled" || echo "  (irqbalance n/a)"
systemctl disable --now cpufrequtils 2>/dev/null && echo "  ✓ cpufrequtils disabled" || echo "  (cpufrequtils n/a)"

CORES=$(nproc)
(
  set +e
  for nic in $(ls /sys/class/net | grep -vE '^lo$|^docker|^cni|^flannel|^vxlan|^virbr|^tailscale'); do
    drv=$(readlink /sys/class/net/$nic/device/driver 2>/dev/null | awk -F/ '{print $NF}')
    case "$drv" in
    mlx5_core | gve)
      ethtool -G $nic rx 8192 tx 8192 2>/dev/null &&
        echo "    $nic ($drv): ring rx/tx=8192" ||
        echo "    $nic ($drv): ring 设置失败"
      if [ "$drv" = "mlx5_core" ]; then
        MAX_CH=$(ethtool -l $nic 2>/dev/null | awk '/Pre-set maximums:/,/Current hardware settings:/ { if ($1=="Combined:") print $2 }' | head -1)
        [ -z "$MAX_CH" ] && MAX_CH=0
        if [ "$MAX_CH" -gt 0 ]; then
          TARGET=$((CORES < MAX_CH ? CORES : MAX_CH))
          [ "$TARGET" -gt 16 ] && TARGET=16
          ethtool -L $nic combined $TARGET 2>/dev/null &&
            echo "    $nic ($drv): combined=$TARGET" || true
        fi
      fi
      ;;
    esac
  done
) || true

echo "options nvme poll_queues=16" >/etc/modprobe.d/nvme.conf
for dev in /dev/nvme[0-9]n[0-9]; do
  [ -b "$dev" ] || continue
  nvme set-feature "$dev" -f 0x8 -V 0x00000107 2>/dev/null &&
    echo "    $dev: coalescing TIME=100us THR=8" || true
done

###############################################################################
# 不包含 (客户自己装):
#   • Container runtime (containerd / docker / podman) + NRI 配置 + CNI plugins
#   • k8s 客户端 (kubelet / kubeadm / kubectl)
#   • nvidia-container-toolkit
###############################################################################

echo
echo "==========================================================="
echo "  ✅ image build 完成 @ $(date -u +%FT%TZ)"
echo "==========================================================="
echo "  下一步:"
echo "    1. sudo reboot   (let initramfs + kernel cmdline 生效)"
echo "    2. 重启后跑 sudo bash image-build.sh --verify 确认全 PASS"
echo "    3. gcloud compute instances stop <THIS-VM>"
echo "    4. gcloud compute disks snapshot <BOOT-DISK> --snapshot-names=<NAME> --zone=<ZONE>"
echo "    5. gcloud compute images create gb200-base-v1 --source-snapshot=<NAME>"
echo "    6. 客户在此 image 上自行装: container runtime / kubelet / nvidia-container-toolkit"
echo "    7. 新节点 VM 用此 image — 无需 startup-script 无需 metadata, systemd 服务自动跑 tuning"
echo "==========================================================="
