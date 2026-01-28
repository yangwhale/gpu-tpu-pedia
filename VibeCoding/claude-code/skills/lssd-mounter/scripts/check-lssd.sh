#!/bin/bash
#
# check-lssd.sh - 检查 Google Cloud Local SSD 状态
#
# 用法: ./check-lssd.sh
#
# 功能:
#   1. 检查 /lssd 挂载状态
#   2. 显示 RAID 状态
#   3. 显示可用 NVMe SSD
#   4. 显示磁盘使用情况
#   5. 检查 HuggingFace 缓存配置
#

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
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

section() {
    echo ""
    echo -e "${CYAN}=== $1 ===${NC}"
}

# 检查挂载状态
check_mount_status() {
    section "挂载状态"

    if mountpoint -q /lssd 2>/dev/null; then
        success "/lssd 已挂载"

        # 获取磁盘使用信息
        local df_output=$(df -h /lssd | tail -1)
        local size=$(echo "$df_output" | awk '{print $2}')
        local used=$(echo "$df_output" | awk '{print $3}')
        local avail=$(echo "$df_output" | awk '{print $4}')
        local use_pct=$(echo "$df_output" | awk '{print $5}')

        echo "  容量: ${size}"
        echo "  已用: ${used} (${use_pct})"
        echo "  可用: ${avail}"
        return 0
    else
        warn "/lssd 未挂载"
        return 1
    fi
}

# 检查 RAID 状态
check_raid_status() {
    section "RAID 状态"

    if [ -e /dev/md0 ]; then
        success "RAID 设备 /dev/md0 存在"

        if [ -f /proc/mdstat ]; then
            echo ""
            cat /proc/mdstat | grep -A 3 "md0" | head -4
        fi

        # 获取 RAID 详细信息
        if command -v mdadm &>/dev/null; then
            echo ""
            local raid_level=$(mdadm --detail /dev/md0 2>/dev/null | grep "Raid Level" | awk '{print $4}')
            local raid_devices=$(mdadm --detail /dev/md0 2>/dev/null | grep "Raid Devices" | awk '{print $4}')
            local array_size=$(mdadm --detail /dev/md0 2>/dev/null | grep "Array Size" | awk '{print $4, $5}')

            if [ -n "$raid_level" ]; then
                echo "  RAID Level: ${raid_level}"
                echo "  设备数量: ${raid_devices}"
                echo "  阵列大小: ${array_size}"
            fi
        fi
    else
        warn "RAID 设备 /dev/md0 不存在"
    fi
}

# 检查 NVMe SSD
check_nvme_ssds() {
    section "NVMe SSD 检测"

    local nvme_pattern="/dev/disk/by-id/google-local-nvme-ssd-*"

    if ls $nvme_pattern &>/dev/null 2>&1; then
        local nvme_count=$(ls $nvme_pattern 2>/dev/null | wc -l)
        success "检测到 ${nvme_count} 块 NVMe SSD"

        # 计算总容量 (每块 SSD 约 375GB)
        local total_gb=$((nvme_count * 375))
        local total_tb=$(echo "scale=1; $total_gb / 1000" | bc 2>/dev/null || echo "$((total_gb / 1000))")
        echo "  预计总容量: ~${total_gb} GB (~${total_tb} TB)"

        # 显示设备列表 (仅显示编号)
        echo ""
        echo "  设备编号:"
        ls $nvme_pattern 2>/dev/null | \
            sed 's/.*google-local-nvme-ssd-/  - SSD #/' | \
            sort -t'#' -k2 -n | \
            head -10

        if [ "$nvme_count" -gt 10 ]; then
            echo "  ... (共 ${nvme_count} 块)"
        fi
    else
        warn "未检测到 NVMe SSD"
        info "此 VM 可能未配置 Local SSD"
    fi
}

# 检查 HuggingFace 缓存
check_hf_cache() {
    section "HuggingFace 缓存"

    # 检查环境变量
    if [ -n "$HF_HOME" ]; then
        success "HF_HOME 已设置: $HF_HOME"
    else
        warn "HF_HOME 未设置"
    fi

    # 检查目录
    if [ -d "/lssd/huggingface" ]; then
        success "/lssd/huggingface 目录存在"

        # 计算缓存大小
        local cache_size=$(du -sh /lssd/huggingface 2>/dev/null | awk '{print $1}')
        if [ -n "$cache_size" ]; then
            echo "  缓存大小: ${cache_size}"
        fi

        # 显示缓存内容概览
        if [ -d "/lssd/huggingface/hub" ]; then
            local model_count=$(ls /lssd/huggingface/hub 2>/dev/null | grep -c "^models--" || echo 0)
            local dataset_count=$(ls /lssd/huggingface/hub 2>/dev/null | grep -c "^datasets--" || echo 0)
            echo "  已缓存模型: ${model_count}"
            echo "  已缓存数据集: ${dataset_count}"
        fi
    else
        warn "/lssd/huggingface 目录不存在"
    fi

    # 检查 bashrc 配置
    local bashrc_file="$HOME/.bashrc"
    if grep -q 'export HF_HOME=/lssd/huggingface' "$bashrc_file" 2>/dev/null; then
        success "HF_HOME 已配置在 ~/.bashrc"
    else
        warn "HF_HOME 未配置在 ~/.bashrc"
    fi
}

# 显示建议
show_recommendations() {
    section "建议"

    local mounted=false
    local has_ssds=false
    local has_raid=false

    mountpoint -q /lssd 2>/dev/null && mounted=true
    ls /dev/disk/by-id/google-local-nvme-ssd-* &>/dev/null 2>&1 && has_ssds=true
    [ -e /dev/md0 ] && has_raid=true

    if $mounted; then
        success "LSSD 已正确配置，无需操作"
        echo ""
        echo "  使用 HuggingFace 缓存:"
        echo "    export HF_HOME=/lssd/huggingface"
        echo ""
    elif $has_ssds; then
        info "检测到 NVMe SSD 但未挂载"
        echo ""
        echo "  运行以下命令挂载 LSSD:"
        echo "    sudo ./scripts/mount-lssd.sh"
        echo ""
        echo "  或手动执行:"
        echo "    sudo mdadm --create /dev/md0 --level=0 --raid-devices=N /dev/disk/by-id/google-local-nvme-ssd-*"
        echo "    sudo mkfs.ext4 -F /dev/md0"
        echo "    sudo mkdir -p /lssd && sudo mount /dev/md0 /lssd"
        echo ""
    else
        warn "未检测到 Local SSD"
        echo ""
        echo "  如需使用 Local SSD，请在创建 VM 时附加 Local SSD:"
        echo "    gcloud compute instances create INSTANCE_NAME \\"
        echo "      --local-ssd interface=nvme \\"
        echo "      ..."
        echo ""
    fi
}

# 主函数
main() {
    echo ""
    echo "=========================================="
    echo "    Google Cloud Local SSD 状态检查"
    echo "=========================================="

    check_mount_status
    check_raid_status
    check_nvme_ssds
    check_hf_cache
    show_recommendations

    echo ""
}

main "$@"
