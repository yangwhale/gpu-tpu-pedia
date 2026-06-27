#!/bin/bash
# =============================================================================
# startup-forrest-gb200-k8s134-prod.sh
#   GB200 (Grace + Blackwell) worker prep — PROD 版 (无 tailscale, 自动 join)
#   k8s 1.34.1 / Calico v3.32 / TLinux 4 ARM64 / Lustre
#
# vs startup-forrest-gb200-k8s134.sh 差异:
#   ✗ 删 Phase A Tailscale (data plane 不依赖, 生产环境用 metadata 直接 join)
#   + Phase 1.8 注入 cp-ssh-pubkey 到 /home/maxwellx/.ssh/authorized_keys (master ssh 维护)
#   + Phase 9.5 auto kubeadm join (从 metadata cp-join-cmd, 失败不退出 stamp 仍写)
#
# 流程 (单次执行,IMEX 中间 reboot 一次):
#   [stamp check] DONE_STAMP 存在 → exit 0
#   Phase 0   参数解析 (metadata)
#   Phase 1   TLinux 4 base
#   Phase 1.5 NIC rename
#   Phase 1.7 Kernel cmdline (init_on_alloc=0)
#   Phase 1.8 注入 cp ssh pub key 到 maxwellx authorized_keys
#   Phase 2   IMEX initramfs → REBOOT
#   === REBOOT (NIC rename + cmdline + IMEX 一并生效) ===
#   Phase 3   k8s prereqs
#   Phase 4   Grace 持久 sysctl
#   Phase 5   Grace tuning (PCI ACS / governor / THP / peermem) + reapply.service
#   Phase 5.5 NIC ring + channel / NVMe coalescing
#   Phase 6   containerd + nvidia-container-toolkit + CNI plugins
#   Phase 7   kubelet/kubeadm/kubectl 1.34.1
#   Phase 8   GPU persist + IMEX channels verify
#   Phase 9   Lustre client install + mount /data
#   Phase 9.5 auto kubeadm join (从 metadata cp-join-cmd)
#   Phase 10  verify + 打印结果
#   [write DONE_STAMP]
#
# Required VM metadata attributes:
#   cp-ip               master VPC IP (e.g. 10.10.0.18)
#   cp-ssh-pubkey       master maxwellx user pub key (master 上 cat /home/maxwellx/.ssh/google_compute_engine.pub)
#   cp-join-cmd         完整 kubeadm join command (master 上 `kubeadm token create --print-join-command`)
#                       格式: "kubeadm join 10.10.0.18:6443 --token <T> --discovery-token-ca-cert-hash sha256:<H>"
#                       不传则跳过 auto-join (后续手动 / ansible 推 node-join.sh)
#
# Optional VM metadata:
#   lustre-ip / lustre-fs / lustre-mount  (default 10.158.224.3 / data / /data)
# =============================================================================
set -euo pipefail
exec > >(tee -a /var/log/forrest-worker-init.log) 2>&1

###############################################################################
# Stamp file: only run once. Reboot 之后再跑只走到 exit 0。
###############################################################################
DONE_STAMP=/etc/.startup-done
IMEX_STAMP=/etc/.imex-configured
NIC_RENAME_STAMP=/etc/.nic-renamed
CMDLINE_STAMP=/etc/.cmdline-set
CP_SSH_STAMP=/etc/.cp-ssh-injected

if [ -f "$DONE_STAMP" ]; then
  echo "[$(date -u +%FT%TZ)] Startup already completed at $(cat $DONE_STAMP). Exit."
  exit 0
fi

echo "==========================================================="
echo "  forrest-gb200-worker-prep PROD   $(date -u +%FT%TZ)"
echo "==========================================================="

###############################################################################
# Phase 0: 参数 (metadata)
###############################################################################
META() {
  curl -sf -H "Metadata-Flavor: Google" \
    "http://metadata.google.internal/computeMetadata/v1/$1" 2>/dev/null || true
}

CP_IP="${CP_IP:-$(META instance/attributes/cp-ip)}"
[ -z "$CP_IP" ] && { echo "FATAL: CP_IP missing (set --metadata=cp-ip=<master VPC IP>)"; exit 1; }

NODE_VPC_IP="$(META instance/network-interfaces/0/ip)"
[ -z "$NODE_VPC_IP" ] && NODE_VPC_IP=$(ip -4 -br addr show | awk '/^(bond0|eth0|ens[0-9]+)/{print $3; exit}' | cut -d/ -f1)
[ -z "$NODE_VPC_IP" ] && { echo "FATAL: NODE_VPC_IP not detected"; exit 1; }

# Prod 版: master 的 ssh pub key + kubeadm join cmd
CP_SSH_PUBKEY="${CP_SSH_PUBKEY:-$(META instance/attributes/cp-ssh-pubkey)}"
CP_JOIN_CMD="${CP_JOIN_CMD:-$(META instance/attributes/cp-join-cmd)}"
INST_NAME="${INST_NAME:-$(META instance/name)}"

LUSTRE_IP="${LUSTRE_IP:-$(META instance/attributes/lustre-ip)}"
LUSTRE_IP="${LUSTRE_IP:-10.158.224.3}"
LUSTRE_FS="${LUSTRE_FS:-$(META instance/attributes/lustre-fs)}"
LUSTRE_FS="${LUSTRE_FS:-data}"
LUSTRE_MOUNT="${LUSTRE_MOUNT:-$(META instance/attributes/lustre-mount)}"
LUSTRE_MOUNT="${LUSTRE_MOUNT:-/data}"

echo "  hostname     = ${INST_NAME}"
echo "  CP_IP        = ${CP_IP}     (master VPC IP)"
echo "  NODE_VPC_IP  = ${NODE_VPC_IP}     (kubelet --node-ip)"
echo "  cp-ssh-pubkey= ${CP_SSH_PUBKEY:0:40}... ${CP_SSH_PUBKEY:+(set)}${CP_SSH_PUBKEY:-(MISSING — master ssh 不通)}"
echo "  cp-join-cmd  = ${CP_JOIN_CMD:+(set, auto-join enabled)}${CP_JOIN_CMD:-(MISSING — skip auto-join)}"
echo "  Lustre       = ${LUSTRE_IP}@tcp:/${LUSTRE_FS} → ${LUSTRE_MOUNT}"

###############################################################################
# Phase 1: TLinux 4 base
###############################################################################
echo "=== [1] TLinux 4 base ==="
dnf install -y sudo cloud-utils-growpart xfsprogs jq pciutils nvme-cli util-linux \
  dnf-plugins-core ethtool 2>/dev/null || true
echo '%google-sudoers ALL=(ALL:ALL) NOPASSWD:ALL' > /etc/sudoers.d/google_sudoers
chmod 440 /etc/sudoers.d/google_sudoers
systemctl enable --now sshd
sed -i 's/^#*PermitRootLogin.*/PermitRootLogin yes/' /etc/ssh/sshd_config
systemctl restart sshd

cat > /etc/security/limits.d/rdma.conf <<EOF
* soft memlock unlimited
* hard memlock unlimited
EOF

setenforce 0 2>/dev/null || true
sed -i -E 's/^SELINUX=(enforcing|permissive)$/SELINUX=disabled/' /etc/selinux/config 2>/dev/null || true

BOOT_DEV=$(findmnt -n -o SOURCE / | sed 's/p[0-9]*$//')
PART_NUM=$(findmnt -n -o SOURCE / | grep -o '[0-9]*$')
growpart "${BOOT_DEV}" "${PART_NUM}" 2>/dev/null || true
xfs_growfs / 2>/dev/null || resize2fs "$(findmnt -n -o SOURCE /)" 2>/dev/null || true

###############################################################################
# Phase 1.5: NIC rename → bond0..bond5
###############################################################################
if [ ! -f "$NIC_RENAME_STAMP" ]; then
  echo "=== [1.5] NIC rename → bond0..bond5 ==="
  RULES_FILE="/etc/udev/rules.d/80-tencent-bond-rename.rules"
  rm -f /etc/systemd/network/10-bond[0-9]*.link 2>/dev/null
  echo "# Auto-generated by forrest startup script" > "$RULES_FILE"
  i=0
  for nic in $(ls /sys/class/net | grep -vE '^lo$|^docker|^cni|^flannel|^vxlan|^virbr|^bond|^tailscale' | sort); do
    MAC=$(cat /sys/class/net/$nic/address 2>/dev/null || true)
    [ -z "$MAC" ] && continue
    echo "SUBSYSTEM==\"net\", ACTION==\"add\", ATTR{address}==\"${MAC}\", NAME=\"bond${i}\"" >> "$RULES_FILE"
    echo "  ${nic} (${MAC}) → bond${i}"
    i=$((i+1))
  done
  udevadm control --reload-rules 2>/dev/null || true
  touch "$NIC_RENAME_STAMP"
  echo "  ${i} NICs scheduled to rename on next reboot"
fi

###############################################################################
# Phase 1.7: Kernel cmdline
###############################################################################
if [ ! -f "$CMDLINE_STAMP" ]; then
  echo "=== [1.7] Kernel cmdline: init_on_alloc=0 (no iommu.passthrough — breaks UVM ATS) ==="
  if command -v grubby >/dev/null 2>&1; then
    grubby --update-kernel=ALL --remove-args="iommu.passthrough iommu.strict" 2>/dev/null || true
    grubby --update-kernel=ALL --args="init_on_alloc=0" || echo "  WARN: grubby failed"
  else
    echo "  WARN: grubby not found, fallback /etc/default/grub"
    if [ -f /etc/default/grub ] && ! grep -q 'init_on_alloc=0' /etc/default/grub; then
      sed -i -E 's/iommu\.(passthrough|strict)=[01] //g' /etc/default/grub
      sed -i 's|^\(GRUB_CMDLINE_LINUX="\)|\1init_on_alloc=0 |' /etc/default/grub
      grub2-mkconfig -o /boot/grub2/grub.cfg 2>/dev/null || \
        grub2-mkconfig -o /boot/efi/EFI/tlinux/grub.cfg 2>/dev/null || \
        echo "  grub2-mkconfig failed, 人工 check"
    fi
  fi
  touch "$CMDLINE_STAMP"
fi

###############################################################################
# Phase 1.8: 注入 master cp 的 ssh pub key 到 maxwellx authorized_keys
#  目的: master 能 ssh maxwellx@<worker> 跑维护命令 (systemctl restart kubelet 等)
###############################################################################
if [ ! -f "$CP_SSH_STAMP" ] && [ -n "$CP_SSH_PUBKEY" ]; then
  echo "=== [1.8] 注入 cp ssh pub key 到 /home/maxwellx/.ssh/authorized_keys ==="
  # 确保 maxwellx 用户存在 (TLinux 4 image 默认有, 没有则 skip)
  if id maxwellx >/dev/null 2>&1; then
    MAX_HOME=$(getent passwd maxwellx | cut -d: -f6)
    [ -z "$MAX_HOME" ] && MAX_HOME=/home/maxwellx
    sudo -u maxwellx mkdir -p "${MAX_HOME}/.ssh"
    sudo -u maxwellx touch "${MAX_HOME}/.ssh/authorized_keys"
    chmod 700 "${MAX_HOME}/.ssh"
    chmod 600 "${MAX_HOME}/.ssh/authorized_keys"
    # idempotent: 只在没 hit 时 append
    KEY_HASH=$(echo "$CP_SSH_PUBKEY" | awk '{print $2}' | cut -c1-30)
    if ! grep -q "$KEY_HASH" "${MAX_HOME}/.ssh/authorized_keys" 2>/dev/null; then
      echo "$CP_SSH_PUBKEY" >> "${MAX_HOME}/.ssh/authorized_keys"
      echo "  ✓ injected cp pub key (${CP_SSH_PUBKEY:0:30}...) to ${MAX_HOME}/.ssh/authorized_keys"
    else
      echo "  ✓ cp pub key already present (skip)"
    fi
    touch "$CP_SSH_STAMP"
  else
    echo "  WARN: maxwellx user not found, skip cp ssh injection"
  fi
elif [ -z "$CP_SSH_PUBKEY" ]; then
  echo "=== [1.8] cp ssh pub key missing in metadata, skip (master ssh 维护不通) ==="
fi

###############################################################################
# Phase 2: IMEX initramfs config + REBOOT (NIC rename + cmdline + IMEX 一起生效)
###############################################################################
if [ ! -f "$IMEX_STAMP" ]; then
  echo "=== [2] IMEX initramfs config (one-time + REBOOT) ==="
  echo "options nvidia NVreg_CreateImexChannel0=1" > /etc/modprobe.d/nvidia.conf
  dracut --force --add-drivers "gve"
  touch "$IMEX_STAMP"
  echo "  IMEX initramfs ready; NIC rename + cmdline 也会一并生效"
  echo "  Rebooting in 3s..."
  sleep 3
  reboot
  exit 0
fi

###############################################################################
# Phase 3: k8s prereqs
###############################################################################
echo "=== [3] k8s prereqs ==="
swapoff -a
sed -i '/swap/d' /etc/fstab

cat > /etc/modules-load.d/k8s.conf <<EOF
overlay
br_netfilter
EOF
modprobe overlay
modprobe br_netfilter

cat > /etc/sysctl.d/k8s.conf <<EOF
net.bridge.bridge-nf-call-iptables  = 1
net.bridge.bridge-nf-call-ip6tables = 1
net.ipv4.ip_forward                 = 1
EOF

###############################################################################
# Phase 4: Grace 持久 sysctl
###############################################################################
echo "=== [4] Grace 持久 sysctl ==="
cat > /etc/sysctl.d/90-grace-gb200.conf <<'EOF'
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
echo "  ✓ numa_balancing=$(cat /proc/sys/kernel/numa_balancing)"

###############################################################################
# Phase 5: Grace 一次性 host tuning + reapply
###############################################################################
echo "=== [5] Grace 一次性 host tuning ==="
ACS_COUNT=0
for BDF in $(lspci -d "*:*:*" | awk '{print $1}'); do
  setpci -v -s "${BDF}" ECAP_ACS+0x6.w >/dev/null 2>&1 || continue
  setpci -v -s "${BDF}" ECAP_ACS+0x6.w=0000 >/dev/null 2>&1 && ACS_COUNT=$((ACS_COUNT+1))
done
echo "  ✓ PCI ACS disabled on $ACS_COUNT bridges"

GOV_COUNT=0
for cpu in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do
  [ -w "$cpu" ] && echo performance > "$cpu" && GOV_COUNT=$((GOV_COUNT+1))
done
echo "  ✓ CPU governor=performance on $GOV_COUNT CPUs"

echo madvise         > /sys/kernel/mm/transparent_hugepage/enabled 2>/dev/null || true
echo "defer+madvise" > /sys/kernel/mm/transparent_hugepage/defrag  2>/dev/null || true

if lsmod | grep -qE "nvidia_peermem"; then
  echo "  ✓ nvidia_peermem loaded"
else
  modprobe nvidia_peermem 2>&1 && echo "  ✓ nvidia_peermem loaded" || echo "  ⚠ peermem failed"
fi

cat > /usr/local/sbin/grace-gb200-reapply.sh <<'REAPPLY'
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

cat > /etc/systemd/system/grace-gb200-reapply.service <<'UNIT'
[Unit]
Description=Reapply Grace GB200 host tuning
After=multi-user.target
[Service]
Type=oneshot
ExecStart=/usr/local/sbin/grace-gb200-reapply.sh
RemainAfterExit=yes
[Install]
WantedBy=multi-user.target
UNIT
systemctl daemon-reload
systemctl enable grace-gb200-reapply.service >/dev/null 2>&1

###############################################################################
# Phase 5.5: NIC ring + channel + NVMe coalescing
###############################################################################
echo "=== [5.5] NIC ring + NVMe tuning ==="
systemctl disable --now irqbalance 2>/dev/null && echo "  ✓ irqbalance disabled" || echo "  (irqbalance n/a)"
systemctl disable --now libvirtd   2>/dev/null && echo "  ✓ libvirtd disabled"   || echo "  (libvirtd n/a)"
systemctl disable --now cpufrequtils 2>/dev/null && echo "  ✓ cpufrequtils disabled" || echo "  (cpufrequtils n/a)"

CORES=$(nproc)
( set +e
  for nic in $(ls /sys/class/net | grep -vE '^lo$|^docker|^cni|^flannel|^vxlan|^virbr|^tailscale'); do
    drv=$(readlink /sys/class/net/$nic/device/driver 2>/dev/null | awk -F/ '{print $NF}')
    case "$drv" in
      mlx5_core|gve)
        ethtool -G $nic rx 8192 tx 8192 2>/dev/null && \
          echo "    $nic ($drv): ring rx/tx=8192" || \
          echo "    $nic ($drv): ring 设置失败"
        if [ "$drv" = "mlx5_core" ]; then
          MAX_CH=$(ethtool -l $nic 2>/dev/null | awk '/Pre-set maximums:/,/Current hardware settings:/ { if ($1=="Combined:") print $2 }' | head -1)
          [ -z "$MAX_CH" ] && MAX_CH=0
          if [ "$MAX_CH" -gt 0 ]; then
            TARGET=$(( CORES < MAX_CH ? CORES : MAX_CH ))
            [ "$TARGET" -gt 16 ] && TARGET=16
            ethtool -L $nic combined $TARGET 2>/dev/null && \
              echo "    $nic ($drv): combined=$TARGET" || true
          fi
        fi
        ;;
    esac
  done
) || true

echo "options nvme poll_queues=16" > /etc/modprobe.d/nvme.conf
for dev in /dev/nvme[0-9]n[0-9]; do
  [ -b "$dev" ] || continue
  nvme set-feature "$dev" -f 0x8 -V 0x00000107 2>/dev/null && \
    echo "    $dev: coalescing TIME=100us THR=8" || true
done

###############################################################################
# Phase 6: containerd + CNI plugins + nvidia-container-toolkit + NRI
###############################################################################
echo "=== [6] containerd + CNI plugins + nvidia-container-toolkit (NRI enabled) ==="
DOCKER_ARCH=$(uname -m)
cat > /etc/yum.repos.d/docker-ce.repo <<EOF
[docker-ce-stable]
name=Docker CE Stable - ${DOCKER_ARCH}
baseurl=https://download.docker.com/linux/rhel/9/${DOCKER_ARCH}/stable
enabled=1
gpgcheck=1
gpgkey=https://download.docker.com/linux/rhel/gpg
EOF
dnf install -y containerd.io
mkdir -p /etc/containerd
containerd config default > /etc/containerd/config.toml
sed -i 's/SystemdCgroup = false/SystemdCgroup = true/' /etc/containerd/config.toml
python3 -c "
import re
p = '/etc/containerd/config.toml'
t = open(p).read()
t2 = re.sub(r'(\[plugins\.\"io\.containerd\.nri\.v1\.nri\"\][^\[]*?)disable = true', r'\1disable = false', t, flags=re.DOTALL)
open(p,'w').write(t2)
" 2>/dev/null || sed -i '/io.containerd.nri.v1.nri/,/^\s*\[/{s/disable = true/disable = false/}' /etc/containerd/config.toml
mkdir -p /var/run/nri
systemctl enable --now containerd

echo "  --- install cni-plugins-linux-arm64 to /opt/cni/bin ---"
mkdir -p /opt/cni/bin
CNI_VER=v1.5.1
curl -sL "https://github.com/containernetworking/plugins/releases/download/${CNI_VER}/cni-plugins-linux-arm64-${CNI_VER}.tgz" -o /tmp/cni.tgz
tar -xzf /tmp/cni.tgz -C /opt/cni/bin/

curl -s -L https://nvidia.github.io/libnvidia-container/stable/rpm/nvidia-container-toolkit.repo | \
  tee /etc/yum.repos.d/nvidia-container-toolkit.repo
dnf install -y nvidia-container-toolkit
nvidia-ctk runtime configure --runtime=containerd --set-as-default
systemctl restart containerd

###############################################################################
# Phase 7: kubelet/kubeadm/kubectl 1.34.1
###############################################################################
echo "=== [7] kubelet/kubeadm/kubectl 1.34.1 ==="
cat > /etc/yum.repos.d/kubernetes.repo <<EOF
[kubernetes]
name=Kubernetes
baseurl=https://pkgs.k8s.io/core:/stable:/v1.34/rpm/
enabled=1
gpgcheck=1
gpgkey=https://pkgs.k8s.io/core:/stable:/v1.34/rpm/repodata/repomd.xml.key
exclude=kubelet kubeadm kubectl cri-tools
EOF
rpm --import https://pkgs.k8s.io/core:/stable:/v1.34/rpm/repodata/repomd.xml.key
dnf install -y kubelet-1.34.1 kubeadm-1.34.1 kubectl-1.34.1 \
  conntrack socat iptables ebtables ethtool iproute-tc \
  --disableexcludes=kubernetes
systemctl enable kubelet
echo "KUBELET_EXTRA_ARGS=--node-ip=${NODE_VPC_IP}" > /etc/sysconfig/kubelet

###############################################################################
# Phase 8: GPU persist + IMEX channels verify (host systemd nvidia-imex 关, CD daemon 接管)
###############################################################################
echo "=== [8] GPU persist + IMEX channels verify ==="
nvidia-smi -pm ENABLED 2>/dev/null || true
echo "  IMEX channels (kernel /dev/nvidia-caps-imex-channels/):"
ls /dev/nvidia-caps-imex-channels/ 2>/dev/null || echo "    WARN: not found"

# Phase 10 retro fix 固化: host systemd nvidia-imex 关掉, 留给 CD daemon
systemctl disable --now nvidia-imex 2>/dev/null && echo "  ✓ systemd nvidia-imex disabled" \
                                                  || echo "  (systemd nvidia-imex 已 disabled/n/a)"

FABRIC_STATE=$(nvidia-smi -q 2>/dev/null | awk '/^    Fabric/{f=1} f && /State/{print $NF; exit}' || true)
echo "  Fabric State = ${FABRIC_STATE:-?} (expected: Completed)"

###############################################################################
# Phase 9: Lustre client + mount /data
###############################################################################
echo "=== [9] Lustre .mount unit ==="
mkdir -p "$LUSTRE_MOUNT"
MOUNT_UNIT="$(systemd-escape --suffix=mount --path "$LUSTRE_MOUNT")"
cat > /etc/systemd/system/${MOUNT_UNIT} <<EOF
[Unit]
Description=Lustre mount ${LUSTRE_MOUNT} (forrest-lustre)
After=network-online.target
Wants=network-online.target

[Mount]
What=${LUSTRE_IP}@tcp:/${LUSTRE_FS}
Where=${LUSTRE_MOUNT}
Type=lustre
Options=defaults,_netdev,noatime,nodiratime
TimeoutSec=120

[Install]
WantedBy=multi-user.target
EOF
systemctl daemon-reload
systemctl enable "${MOUNT_UNIT}" >/dev/null 2>&1

if command -v mount.lustre >/dev/null 2>&1; then
  systemctl start "${MOUNT_UNIT}" 2>&1 | tail -3 || echo "  ⚠ lustre mount 失败"
  mountpoint -q "$LUSTRE_MOUNT" && echo "  ✓ ${LUSTRE_MOUNT} mounted" || echo "  ⚠ not mounted yet"
else
  echo "  Lustre client 未装成功,跳过 mount"
fi

###############################################################################
# Phase 9.5: auto kubeadm join (从 metadata cp-join-cmd, 不阻塞 stamp)
###############################################################################
NODE_NAME=$(hostname -s 2>/dev/null || META instance/name || echo "${INST_NAME}")

if [ -n "$CP_JOIN_CMD" ] && [ ! -f /etc/kubernetes/kubelet.conf ]; then
  echo "=== [9.5] auto kubeadm join ==="
  echo "  CP_JOIN_CMD: ${CP_JOIN_CMD}"
  echo "  --node-name ${NODE_NAME}"
  set +e
  ${CP_JOIN_CMD} --node-name "${NODE_NAME}" --ignore-preflight-errors=Hostname 2>&1 | tee /var/log/forrest-kubeadm-join.log
  RC=${PIPESTATUS[0]}
  set -e
  if [ "$RC" -eq 0 ]; then
    echo "  ✓ joined cluster as ${NODE_NAME}"
  else
    echo "  ⚠ join failed (rc=$RC), 后续手动 / refresh-join-token.sh"
  fi
elif [ -f /etc/kubernetes/kubelet.conf ]; then
  echo "=== [9.5] kubelet.conf 已存在 (已 join 过), skip ==="
else
  echo "=== [9.5] cp-join-cmd missing in metadata, skip auto-join ==="
fi

###############################################################################
# Phase 10: verify + 打印结果
###############################################################################
FABRIC=$(nvidia-smi -q -i 0 2>/dev/null | grep -E "(ClusterUUID|CliqueId)" | awk '{print $NF}')
CLIQUE_ID=$(echo "$FABRIC" | head -1)
CLUSTER_UUID=$(echo "$FABRIC" | tail -1)

GPU_PRODUCT=$(nvidia-smi --query-gpu=name --format=csv,noheader -i 0 2>/dev/null | head -1 | tr ' ' '-')
GPU_COUNT=$(nvidia-smi -L 2>/dev/null | wc -l)
DRIVER_VER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader -i 0 2>/dev/null | head -1)
JOIN_STATE="NOT-joined"
[ -f /etc/kubernetes/kubelet.conf ] && JOIN_STATE="joined (kubelet.conf 在)"

echo
echo "==========================================================="
echo "  ✅ ${NODE_NAME} prepared  @ $(date -u +%FT%TZ)"
echo "  VPC IP        = ${NODE_VPC_IP}"
echo "  CP target     = ${CP_IP}:6443"
echo "  Join state    = ${JOIN_STATE}"
echo "  Driver        = ${DRIVER_VER:-?}"
echo "  ClusterUUID   = ${CLUSTER_UUID:-?}"
echo "  CliqueId      = ${CLIQUE_ID:-?}"
echo "  GPUs          = ${GPU_COUNT}× ${GPU_PRODUCT:-?}"
echo "  Lustre        = $(mountpoint -q ${LUSTRE_MOUNT} && echo mounted || echo NOT-mounted)"
echo "==========================================================="
nvidia-smi -L 2>/dev/null | head -10 || true

cat <<NEXT

===========================================================================
NEXT —
  verify on master: gx k8n k get node ${NODE_NAME} -o wide
  ssh from master:  ssh maxwellx@${NODE_NAME} 'systemctl is-active kubelet'  (cp-ssh-pubkey 已注入)
  label 全自动:     device-plugin + GFD 0.19.2 设 nvidia.com/gpu.{clique,product,count,...}
                  ComputeDomain workload 触发时 controller 设 resource.nvidia.com/computeDomain
===========================================================================
NEXT

###############################################################################
# Stamp: DONE_STAMP
###############################################################################
date -u +%FT%TZ > "$DONE_STAMP"
echo "[$(date -u +%FT%TZ)] startup-script-prod done. Stamp: $DONE_STAMP"
