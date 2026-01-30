#!/bin/bash
# =============================================================================
# Claude Code 安装脚本
# 适用于 Linux/macOS 系统
#
# Usage:
#   ./install-claude-code.sh [OPTIONS]
#
# Options:
#   --project-id <ID>        Google Cloud Project ID
#   --context7-key <KEY>     Context7 API Key
#   --github-token <TOKEN>   GitHub Personal Access Token
#   --github-user <USER>     GitHub Username
#   --jina-key <KEY>         Jina AI API Key
#   --yes, -y                Auto-confirm prompts (skip optional params, use current gcloud project)
#   --help, -h               Show this help message
#
# Note: On GCE VMs, uses --no-launch-browser for gcloud auth
#
# Example:
#   ./install-claude-code.sh --project-id my-project --github-token ghp_xxx --github-user myuser -y
# =============================================================================

set -e

# =============================================================================
# 命令行参数解析
# =============================================================================
PROJECT_ID=""
CONTEXT7_API_KEY=""
GITHUB_TOKEN=""
GITHUB_USERNAME=""
JINA_API_KEY=""
AUTO_YES=false

show_help() {
    head -20 "$0" | tail -18 | sed 's/^# //' | sed 's/^#//'
    exit 0
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --project-id)
            PROJECT_ID="$2"
            shift 2
            ;;
        --context7-key)
            CONTEXT7_API_KEY="$2"
            shift 2
            ;;
        --github-token)
            GITHUB_TOKEN="$2"
            shift 2
            ;;
        --github-user)
            GITHUB_USERNAME="$2"
            shift 2
            ;;
        --jina-key)
            JINA_API_KEY="$2"
            shift 2
            ;;
        --yes|-y)
            AUTO_YES=true
            shift
            ;;
        --help|-h)
            show_help
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# 脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_DIR="$SCRIPT_DIR/claude-code/config"
CLAUDE_DIR="$HOME/.claude"

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 打印带颜色的信息
info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1"
    exit 1
}

# =============================================================================
# 1. 检查系统要求
# =============================================================================
info "检查系统要求..."

# 检查操作系统
OS="$(uname -s)"
case "${OS}" in
    Linux*)     OS_TYPE=Linux;;
    Darwin*)    OS_TYPE=Mac;;
    *)          error "不支持的操作系统: ${OS}"
esac
info "操作系统: ${OS_TYPE}"

# =============================================================================
# 2. 安装 Claude Code (原生安装)
# =============================================================================
info "安装 Claude Code..."

curl -fsSL https://claude.ai/install.sh | bash

success "Claude Code 安装完成！"

# 确保 ~/.local/bin 在 PATH 中 (当前会话)
export PATH="$HOME/.local/bin:$PATH"
info "已设置 PATH: ~/.local/bin"

# 根据 shell 类型确定配置文件
if [ -n "$ZSH_VERSION" ]; then
    SHELL_RC="$HOME/.zshrc"
else
    SHELL_RC="$HOME/.bashrc"
fi

# 确保 PATH 被写入 shell 配置文件 (持久化)
if ! grep -q '.local/bin' "$SHELL_RC" 2>/dev/null; then
    echo '' >> "$SHELL_RC"
    echo '# Claude Code PATH' >> "$SHELL_RC"
    echo 'export PATH="$HOME/.local/bin:$PATH"' >> "$SHELL_RC"
    info "已添加 PATH 到 $SHELL_RC"
else
    info "PATH 已存在于 $SHELL_RC"
fi

# =============================================================================
# 3. 验证 Claude Code 安装
# =============================================================================
info "验证 Claude Code 安装..."

if command -v claude &> /dev/null; then
    success "Claude Code 安装成功！"
    info "版本: $(claude --version 2>/dev/null || echo '已安装')"
else
    error "Claude Code 安装失败，请检查错误信息"
fi

# =============================================================================
# 4. 安装 Node.js v20 (用于 MCP 和插件)
# =============================================================================
info "检查 Node.js (用于 MCP 服务器和插件)..."

install_nodejs() {
    # 检查是否有 nvm
    if command -v nvm &> /dev/null; then
        info "使用已有的 nvm 安装 Node.js v20..."
        nvm install 20
        nvm use 20
    else
        # 安装 nvm
        info "安装 nvm..."
        curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.1/install.sh | bash
        export NVM_DIR="$HOME/.nvm"
        [ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"
        info "安装 Node.js v20..."
        nvm install 20
        nvm use 20
    fi
}

if command -v node &> /dev/null; then
    NODE_VERSION=$(node -v | sed 's/v//' | cut -d. -f1)
    if [ "$NODE_VERSION" -ge 18 ]; then
        success "Node.js 版本符合要求: $(node -v)"
    else
        warn "Node.js 版本过低 ($(node -v))，需要 v18+ 来运行 MCP 和插件"
        install_nodejs
    fi
else
    info "未检测到 Node.js，正在安装 v20 (用于 MCP 和插件)..."
    install_nodejs
fi

# 验证 Node.js
if command -v node &> /dev/null; then
    success "Node.js 安装成功: $(node -v)"
fi

# =============================================================================
# 5. 配置 Vertex AI 授权
# =============================================================================
info "配置 Vertex AI..."

# 检查 gcloud 是否安装
if ! command -v gcloud &> /dev/null; then
    warn "未检测到 gcloud CLI，跳过 Vertex AI 配置"
    warn "请先安装 Google Cloud SDK: https://cloud.google.com/sdk/docs/install"
else
    info "检查 Google Cloud 认证状态..."
    
    # 检查 application-default credentials 是否存在
    ADC_FILE="$HOME/.config/gcloud/application_default_credentials.json"
    
    # 检测是否在 GCE 环境
    IS_GCE=false
    if curl -s -H "Metadata-Flavor: Google" http://169.254.169.254/computeMetadata/v1/ &>/dev/null; then
        IS_GCE=true
    fi

    # gcloud auth 辅助函数
    do_gcloud_auth() {
        if [ "$IS_GCE" = true ]; then
            info "检测到 GCE 环境，使用 --no-launch-browser 模式..."
            echo -e "${YELLOW}请在浏览器中打开下方 URL 完成认证，然后粘贴返回的 code:${NC}"
            gcloud auth application-default login --no-launch-browser
        else
            gcloud auth application-default login
        fi
    }

    if [ -f "$ADC_FILE" ]; then
        success "已检测到 Application Default Credentials"
        if [ "$AUTO_YES" = false ]; then
            echo -e "${YELLOW}是否需要重新登录? (y/N):${NC}"
            read -p "> " REAUTH
            if [[ "$REAUTH" =~ ^[Yy]$ ]]; then
                info "执行 gcloud auth application-default login..."
                do_gcloud_auth
            fi
        fi
    else
        warn "未检测到 Application Default Credentials"
        echo -e "${YELLOW}是否现在执行 gcloud auth application-default login? (Y/n):${NC}"
        read -p "> " DO_AUTH
        if [[ ! "$DO_AUTH" =~ ^[Nn]$ ]]; then
            info "执行 gcloud auth application-default login..."
            if do_gcloud_auth; then
                success "Google Cloud 认证成功！"
            else
                warn "认证失败，请稍后手动执行: gcloud auth application-default login"
            fi
        else
            warn "跳过认证，请稍后手动执行: gcloud auth application-default login"
        fi
    fi
    
    # 获取 Project ID
    if [ -z "$PROJECT_ID" ]; then
        # 尝试获取当前配置的 project
        CURRENT_PROJECT=$(gcloud config get-value project 2>/dev/null)

        if [ "$AUTO_YES" = true ] && [ -n "$CURRENT_PROJECT" ]; then
            # 自动模式：使用当前 gcloud project
            PROJECT_ID="$CURRENT_PROJECT"
            info "自动使用当前 gcloud project: $PROJECT_ID"
        else
            echo ""
            echo -e "${YELLOW}请输入你的 Google Cloud Project ID (用于 Vertex AI):${NC}"
            if [ -n "$CURRENT_PROJECT" ]; then
                echo -e "${BLUE}当前 gcloud 配置的 project: $CURRENT_PROJECT${NC}"
                echo -e "${YELLOW}直接回车使用此 project，或输入新的 project ID:${NC}"
            fi
            read -p "> " INPUT_PROJECT_ID

            # 如果用户直接回车，使用当前 project
            if [ -z "$INPUT_PROJECT_ID" ] && [ -n "$CURRENT_PROJECT" ]; then
                PROJECT_ID="$CURRENT_PROJECT"
            else
                PROJECT_ID="$INPUT_PROJECT_ID"
            fi
        fi
    else
        info "使用命令行参数 Project ID: $PROJECT_ID"
    fi
    
    if [ -z "$PROJECT_ID" ]; then
        warn "未输入 Project ID，跳过 Vertex AI 配置"
    else
        # 获取 Context7 API Key (可选)
        if [ -z "$CONTEXT7_API_KEY" ] && [ "$AUTO_YES" = false ]; then
            echo ""
            echo -e "${YELLOW}请输入 Context7 API Key (用于文档查询插件，可选，直接回车跳过):${NC}"
            echo -e "${BLUE}获取地址: https://context7.io/${NC}"
            read -p "> " CONTEXT7_API_KEY
        elif [ -n "$CONTEXT7_API_KEY" ]; then
            info "使用命令行参数 Context7 API Key"
        fi

        # 获取 GitHub Personal Access Token (可选)
        if [ -z "$GITHUB_TOKEN" ] && [ "$AUTO_YES" = false ]; then
            echo ""
            echo -e "${YELLOW}请输入 GitHub Personal Access Token (用于 GitHub 插件和 git push，可选，直接回车跳过):${NC}"
            echo -e "${BLUE}获取地址: https://github.com/settings/tokens${NC}"
            echo -e "${BLUE}需要权限: repo, read:org, read:user${NC}"
            read -p "> " GITHUB_TOKEN
        elif [ -n "$GITHUB_TOKEN" ]; then
            info "使用命令行参数 GitHub Token"
        fi

        # 获取 GitHub 用户名 (如果提供了 token)
        if [ -n "$GITHUB_TOKEN" ]; then
            if [ -z "$GITHUB_USERNAME" ] && [ "$AUTO_YES" = false ]; then
                echo -e "${YELLOW}请输入 GitHub 用户名 (用于 git push):${NC}"
                read -p "> " GITHUB_USERNAME
            elif [ -n "$GITHUB_USERNAME" ]; then
                info "使用命令行参数 GitHub Username: $GITHUB_USERNAME"
            fi

            # 保存到 git credentials
            if [ -n "$GITHUB_USERNAME" ]; then
                git config --global credential.helper store
                echo "https://${GITHUB_USERNAME}:${GITHUB_TOKEN}@github.com" > ~/.git-credentials
                chmod 600 ~/.git-credentials
                success "GitHub token 已保存到 git credentials (用于 git push)"
            fi
        fi

        # 获取 Jina AI API Key (可选)
        if [ -z "$JINA_API_KEY" ] && [ "$AUTO_YES" = false ]; then
            echo ""
            echo -e "${YELLOW}请输入 Jina AI API Key (用于网页读取和搜索 MCP，可选，直接回车跳过):${NC}"
            echo -e "${BLUE}获取地址: https://jina.ai/${NC}"
            read -p "> " JINA_API_KEY
        elif [ -n "$JINA_API_KEY" ]; then
            info "使用命令行参数 Jina AI API Key"
        fi

        # 创建 .claude 目录
        mkdir -p "$CLAUDE_DIR"
        
        # 从模板生成配置文件
        if [ -f "$CONFIG_DIR/settings.template.json" ]; then
            # 替换所有占位符
            sed -e "s/\${PROJECT_ID}/$PROJECT_ID/g" \
                -e "s/\${CONTEXT7_API_KEY}/$CONTEXT7_API_KEY/g" \
                -e "s/\${GITHUB_TOKEN}/$GITHUB_TOKEN/g" \
                "$CONFIG_DIR/settings.template.json" > "$CLAUDE_DIR/settings.json"
            success "已生成 settings.json (Project ID: $PROJECT_ID)"
            if [ -n "$CONTEXT7_API_KEY" ]; then
                success "已配置 Context7 API Key"
            fi
            if [ -n "$GITHUB_TOKEN" ]; then
                success "已配置 GitHub Token (Claude 插件)"
                if [ -n "$GITHUB_USERNAME" ]; then
                    success "已配置 Git Credentials (git push)"
                fi
            fi
        else
            warn "未找到配置模板: $CONFIG_DIR/settings.template.json"
        fi
    fi
fi

# =============================================================================
# 6. 安装 Marketplaces
# =============================================================================
info "安装插件市场..."

# Marketplace 列表
declare -A MARKETPLACES=(
    ["claude-plugins-official"]="anthropics/claude-plugins-official"
    ["everything-claude-code"]="affaan-m/everything-claude-code"
    ["awesome-claude-skills"]="ComposioHQ/awesome-claude-skills"
    ["ui-ux-pro-max-skill"]="nextlevelbuilder/ui-ux-pro-max-skill"
    ["planning-with-files"]="OthmanAdi/planning-with-files"
)

for marketplace in "${!MARKETPLACES[@]}"; do
    repo="${MARKETPLACES[$marketplace]}"
    info "添加市场: $marketplace ($repo)"
    claude plugin marketplace add "$repo" 2>/dev/null || warn "添加市场 $marketplace 失败，请手动添加"
done

success "插件市场添加完成！"

# =============================================================================
# 7. 安装 Plugins
# =============================================================================
info "安装插件..."

# 从配置文件读取插件列表
if [ -f "$CONFIG_DIR/plugins.txt" ]; then
    while IFS= read -r line || [ -n "$line" ]; do
        # 跳过注释和空行
        [[ "$line" =~ ^#.*$ ]] && continue
        [[ -z "$line" ]] && continue
        
        plugin=$(echo "$line" | tr -d '[:space:]')
        info "安装插件: $plugin"
        claude plugin install "$plugin" 2>/dev/null || warn "安装 $plugin 失败，请手动安装"
    done < "$CONFIG_DIR/plugins.txt"
    success "插件安装完成！"
else
    warn "未找到插件列表: $CONFIG_DIR/plugins.txt"
fi

# =============================================================================
# 8. 安装 Happy Coder
# =============================================================================
info "安装 Happy Coder..."

if command -v npm &> /dev/null; then
    npm install -g happy-coder && success "Happy Coder 安装成功！" || warn "Happy Coder 安装失败"
else
    warn "npm 未找到，跳过 Happy Coder 安装"
fi

# =============================================================================
# 9. 安装 MCP 服务器
# =============================================================================
info "安装 MCP 服务器..."

# Jina AI MCP (需要 API Key)
if [ -n "$JINA_API_KEY" ]; then
    info "安装 Jina AI MCP..."
    claude mcp add-json "jina-ai" "{\"command\":\"npx\",\"args\":[\"-y\",\"jina-ai-mcp-server\"],\"env\":{\"JINA_API_KEY\":\"$JINA_API_KEY\"}}" 2>/dev/null \
        && success "Jina AI MCP 安装成功！" \
        || warn "Jina AI MCP 安装失败，请手动安装"
else
    info "跳过 Jina AI MCP (未提供 API Key)"
fi

# Kubernetes MCP (支持 GKE、minikube、Rancher Desktop 等)
info "安装 Kubernetes MCP..."
claude mcp add kubernetes -- npx mcp-server-kubernetes 2>/dev/null \
    && success "Kubernetes MCP 安装成功！" \
    || warn "Kubernetes MCP 安装失败，请手动安装"

success "MCP 服务器安装完成！"

# =============================================================================
# 10. 安装自定义 Skills (使用软连接)
# =============================================================================
info "安装自定义 Skills..."

SKILLS_SRC="$SCRIPT_DIR/claude-code/skills"
SKILLS_DST="$CLAUDE_DIR/skills"

# 记录安装的 skills
INSTALLED_SKILLS=()

if [ -d "$SKILLS_SRC" ]; then
    # 如果目标已存在且不是软连接，先备份
    if [ -e "$SKILLS_DST" ] && [ ! -L "$SKILLS_DST" ]; then
        warn "发现已存在的 skills 目录，备份到 ${SKILLS_DST}.bak"
        mv "$SKILLS_DST" "${SKILLS_DST}.bak"
    fi

    # 如果已经是正确的软连接，跳过
    if [ -L "$SKILLS_DST" ] && [ "$(readlink -f "$SKILLS_DST")" = "$(realpath "$SKILLS_SRC")" ]; then
        info "Skills 软连接已存在且正确"
    else
        # 删除旧的软连接（如果存在）
        [ -L "$SKILLS_DST" ] && rm "$SKILLS_DST"

        # 创建软连接
        ln -s "$SKILLS_SRC" "$SKILLS_DST"
        success "已创建软连接: $SKILLS_DST -> $SKILLS_SRC"
    fi

    # 统计已安装的 skills
    for skill_dir in "$SKILLS_SRC"/*/; do
        if [ -d "$skill_dir" ]; then
            skill_name=$(basename "$skill_dir")
            # 跳过隐藏目录和 .DS_Store
            [[ "$skill_name" == .* ]] && continue
            INSTALLED_SKILLS+=("$skill_name")
        fi
    done

    success "自定义 Skills 安装完成！(共 ${#INSTALLED_SKILLS[@]} 个，通过软连接)"

    # 显示已安装的 skills 及其功能
    echo ""
    info "已安装的 Skills:"
    for skill in "${INSTALLED_SKILLS[@]}"; do
        case "$skill" in
            "deepep-installer")
                echo "  - deepep-installer: DeepEP 安装 (CUDA, GDRCopy, NVSHMEM, DeepEP)"
                ;;
            "sglang-installer")
                echo "  - sglang-installer: SGLang 安装和调试 (含 DeepEP 依赖检测)"
                ;;
            "paper-explainer")
                echo "  - paper-explainer: 论文解读 (中文大白话翻译)"
                ;;
            "lssd-mounter")
                echo "  - lssd-mounter: Local SSD 挂载 (RAID0, HuggingFace 缓存)"
                ;;
            "vllm-installer")
                echo "  - vllm-installer: vLLM 安装和调试 (含 LSSD/DeepEP 检测)"
                ;;
            *)
                echo "  - $skill"
                ;;
        esac
    done
else
    warn "未找到 Skills 目录: $SKILLS_SRC"
fi

# =============================================================================
# 11. 完成
# =============================================================================
echo ""
echo "=============================================="
echo -e "${GREEN}安装完成！${NC}"
echo "=============================================="
echo ""
echo "已安装组件："
echo "  - Claude Code: 原生安装"
echo "  - Node.js: v20 LTS (用于 MCP 服务器和插件)"
echo "  - Happy Coder: npm 全局安装"
echo "  - Vertex AI 配置: $CLAUDE_DIR/settings.json"
echo "  - 插件市场: ${#MARKETPLACES[@]} 个"
echo "  - MCP 服务器: Kubernetes, Jina AI (如已配置 API Key)"
echo "  - 自定义 Skills: ${#INSTALLED_SKILLS[@]} 个"
for skill in "${INSTALLED_SKILLS[@]}"; do
    echo "      * $skill"
done
echo ""
echo "下一步操作："
echo ""
echo "1. 首次运行 claude 命令进行登录认证："
echo "   $ claude"
echo ""
echo "2. 查看已安装插件："
echo "   $ claude plugin list"
echo ""
echo "3. 查看已安装 MCP 服务器："
echo "   $ claude mcp list"
echo ""
echo "4. MCP 配置文件位置："
echo "   ~/.claude/mcp_servers.json"
echo ""
echo "5. 使用自定义 Skills:"
echo "   - /deepep-installer  : 安装 DeepEP (用于 MoE 模型)"
echo "   - /sglang-installer  : 安装和调试 SGLang"
echo "   - /vllm-installer    : 安装和调试 vLLM"
echo "   - /lssd-mounter      : 挂载 Local SSD (RAID0)"
echo "   - /paper-explainer   : 论文解读"
echo ""
echo "6. Skills 目录位置:"
echo "   ~/.claude/skills/"
echo ""
echo "=============================================="
