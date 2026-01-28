#!/bin/bash
#
# mount-lssd.sh - 检测、格式化并挂载 Google Cloud Local SSD 为 RAID0
#
# 用法: sudo ./mount-lssd.sh
#
# 功能:
#   1. 检测 /lssd 是否已挂载
#   2. 检测可用的 NVMe SSD 数量
#   3. 创建 RAID0 阵列 (支持任意数量的 SSD)
#   4. 格式化并挂载到 /lssd
#   5. 配置 HuggingFace 缓存目录
#

set -e

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

success() {
    echo -e "${GREEN}[OK]${NC} $1"
}

warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 检查是否以 root 运行
check_root() {
    if [ "$EUID" -ne 0 ]; then
        error "请使用 sudo 运行此脚本"
        exit 1
    fi
}

# 检查 mdadm 是否安装
check_mdadm() {
    if ! command -v mdadm &>/dev/null; then
        info "正在安装 mdadm..."
        apt-get update -qq && apt-get install -y -qq mdadm
    fi
}

# 检查是否已挂载
check_mounted() {
    if mountpoint -q /lssd 2>/dev/null; then
        local size=$(df -h /lssd | tail -1 | awk '{print $2}')
        local used=$(df -h /lssd | tail -1 | awk '{print $3}')
        local avail=$(df -h /lssd | tail -1 | awk '{print $4}')
        success "LSSD 已挂载"
        info "容量: ${size} | 已用: ${used} | 可用: ${avail}"
        return 0
    fi
    return 1
}

# 检测 NVMe SSD
detect_nvme() {
    # 获取所有 Google Local NVMe SSD
    local nvme_pattern="/dev/disk/by-id/google-local-nvme-ssd-*"

    if ! ls $nvme_pattern &>/dev/null; then
        warn "未检测到 NVMe SSD"
        info "此 VM 可能未配置 Local SSD"
        return 1
    fi

    # 获取 SSD 列表 (按数字排序)
    NVME_DEVICES=$(ls $nvme_pattern 2>/dev/null | sort -V)
    NVME_COUNT=$(echo "$NVME_DEVICES" | wc -l)

    info "检测到 ${NVME_COUNT} 块 NVMe SSD"
    return 0
}

# 创建 RAID0 阵列
create_raid() {
    if [ -e /dev/md0 ]; then
        warn "RAID 设备 /dev/md0 已存在，跳过创建"
        return 0
    fi

    info "创建 RAID0 阵列 (${NVME_COUNT} 块 NVMe SSD)..."

    # 将设备列表转换为数组
    local devices_array=($NVME_DEVICES)

    # 使用 mdadm 创建 RAID0
    mdadm --create /dev/md0 \
        --level=0 \
        --raid-devices=${NVME_COUNT} \
        "${devices_array[@]}"

    success "RAID0 阵列创建完成"
}

# 格式化文件系统
format_filesystem() {
    # 检查是否已有文件系统
    if blkid /dev/md0 &>/dev/null; then
        local fs_type=$(blkid -o value -s TYPE /dev/md0)
        if [ "$fs_type" = "ext4" ]; then
            info "文件系统已存在 (ext4)，跳过格式化"
            return 0
        fi
    fi

    info "格式化文件系统 (ext4)..."
    mkfs.ext4 -F /dev/md0
    success "文件系统格式化完成"
}

# 挂载到 /lssd
mount_lssd() {
    info "挂载 /dev/md0 到 /lssd..."

    mkdir -p /lssd

    if mount /dev/md0 /lssd; then
        success "LSSD 挂载完成"
    else
        error "挂载失败"
        return 1
    fi

    # 设置权限
    chmod a+w /lssd
}

# 设置 HuggingFace 缓存目录
setup_hf_cache() {
    info "设置 HuggingFace 缓存目录..."

    mkdir -p /lssd/huggingface
    chmod a+w /lssd/huggingface

    # 获取调用者的用户名 (sudo 情况下)
    local real_user="${SUDO_USER:-$USER}"
    local bashrc_file="/home/${real_user}/.bashrc"

    if [ -f "$bashrc_file" ]; then
        if ! grep -q 'export HF_HOME=/lssd/huggingface' "$bashrc_file" 2>/dev/null; then
            echo '' >> "$bashrc_file"
            echo '# HuggingFace cache on LSSD (high-speed local SSD)' >> "$bashrc_file"
            echo 'export HF_HOME=/lssd/huggingface' >> "$bashrc_file"
            info "已将 HF_HOME 添加到 $bashrc_file"
        else
            info "HF_HOME 已在 $bashrc_file 中配置"
        fi
    fi

    success "HuggingFace 缓存目录配置完成: /lssd/huggingface"
}

# 显示最终状态
show_status() {
    echo ""
    echo "=========================================="
    echo "          LSSD 挂载状态"
    echo "=========================================="
    echo ""

    # 挂载信息
    df -h /lssd
    echo ""

    # RAID 状态
    info "RAID 状态:"
    cat /proc/mdstat | grep -A 2 "md0"
    echo ""

    # 环境变量提示
    success "请运行以下命令使环境变量生效:"
    echo "    source ~/.bashrc"
    echo ""
    echo "或在当前 shell 中运行:"
    echo "    export HF_HOME=/lssd/huggingface"
    echo ""
}

# 主函数
main() {
    echo ""
    echo "=========================================="
    echo "  Google Cloud Local SSD RAID0 挂载工具"
    echo "=========================================="
    echo ""

    check_root
    check_mdadm

    # 检查是否已挂载
    if check_mounted; then
        setup_hf_cache
        show_status
        exit 0
    fi

    # 检测 NVMe SSD
    if ! detect_nvme; then
        exit 1
    fi

    # 创建 RAID0
    create_raid

    # 格式化
    format_filesystem

    # 挂载
    mount_lssd

    # 设置 HF 缓存
    setup_hf_cache

    # 显示状态
    show_status
}

main "$@"
