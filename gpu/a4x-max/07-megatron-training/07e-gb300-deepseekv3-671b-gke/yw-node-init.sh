#!/bin/bash
# yw-node-init.sh — GB300 pod 容器启动初始化脚本
#
# 用途：在 image 尚未 bake 这些依赖前，作为 StatefulSet 的容器 command 使用。
#       把每次启动/每次跑都需要的东西一次装好：sshd + 免密 SSH + dllogger。
#       pod 起来后即可从 pod-0 用 `ssh yw-<g>-<i>.yw` 免密 fanout 启动训练。
#
# 前置：k8s Secret `yw-ssh`（含 id_ed25519 + authorized_keys）挂载到 /etc/yw-ssh。
#       创建方式：
#         ssh-keygen -t ed25519 -N "" -f yw_ssh_key -C yw-pool
#         kubectl create secret generic yw-ssh \
#           --from-file=id_ed25519=yw_ssh_key \
#           --from-file=authorized_keys=yw_ssh_key.pub
#
# 在 StatefulSet 里的用法：
#   command: ["/bin/bash","-c","<本脚本内容>"]
#   volumeMounts: - name: yw-ssh; mountPath: /etc/yw-ssh; readOnly: true
#
# 幂等：可重复执行。已装则跳过。
set -u

echo "[yw-init] $(date) starting node init on $(hostname)"

# ---------- 1. openssh-server（带重试，避免 64 pod 并发 apt 抖动）----------
if [ ! -x /usr/sbin/sshd ]; then
  for attempt in 1 2 3 4 5; do
    apt-get update -qq && apt-get install -y -qq openssh-server && break
    echo "[yw-init] apt attempt $attempt failed, retry in 10s"; sleep 10
  done
fi
[ -x /usr/sbin/sshd ] || { echo "[yw-init] FATAL: sshd install failed"; }

# ---------- 2. 免密 SSH 密钥（从挂载的 Secret 注入）----------
mkdir -p /root/.ssh /run/sshd
if [ -f /etc/yw-ssh/id_ed25519 ]; then
  cp /etc/yw-ssh/id_ed25519    /root/.ssh/id_ed25519
  cp /etc/yw-ssh/authorized_keys /root/.ssh/authorized_keys
  chmod 700 /root/.ssh
  chmod 600 /root/.ssh/id_ed25519 /root/.ssh/authorized_keys
else
  echo "[yw-init] WARN: /etc/yw-ssh secret not mounted — SSH fanout will not work"
fi

# ---------- 3. SSH client config（跨 pod 不做 host key 校验）----------
printf 'StrictHostKeyChecking no\nUserKnownHostsFile /dev/null\nLogLevel ERROR\n' > /root/.ssh/config
chmod 600 /root/.ssh/config

# ---------- 4. sshd host keys + 允许 root 公钥登录 ----------
ssh-keygen -A >/dev/null 2>&1
sed -i 's/^#\?PermitRootLogin.*/PermitRootLogin prohibit-password/' /etc/ssh/sshd_config
sed -i 's/^#\?PubkeyAuthentication.*/PubkeyAuthentication yes/' /etc/ssh/sshd_config

# ---------- 5. 启动 sshd ----------
/usr/sbin/sshd && echo "[yw-init] sshd started"

# ---------- 6. dllogger（训练脚本每次都要，提前装好）----------
python -c "import dllogger" 2>/dev/null || \
  pip install --no-cache-dir "git+https://github.com/NVIDIA/dllogger#egg=dllogger" >/dev/null 2>&1 || true

echo "[yw-init] SSH_READY $(hostname)"

# ---------- 7. 常驻 ----------
sleep infinity
