---
name: zsh-installer
description: This skill should be used when users need to install, configure, and customize Zsh with Oh My Zsh on Linux systems. It covers Zsh installation, Oh My Zsh setup, theme configuration (agnoster), plugin management, and environment variable setup. The skill replicates the standard team shell configuration for consistent development environments across machines.
license: MIT
---

# Zsh + Oh My Zsh Installer

## Overview

在 Linux 系统上安装和配置 Zsh + Oh My Zsh，统一团队开发环境。包括 agnoster 主题、git 插件、NVM 集成和自定义颜色方案。

## When to Use This Skill

- 在新 VM 或服务器上初始化 shell 环境
- 用户要求安装 zsh 或 oh-my-zsh
- 需要统一的终端配置
- 诊断 zsh 或 oh-my-zsh 相关问题

## Prerequisites

- Ubuntu/Debian 系统 (apt-get)
- sudo 权限
- 网络连接 (下载 Oh My Zsh)

## Quick Start

### 一键安装

```bash
# 1. 安装 zsh
sudo apt-get update && sudo apt-get install -y zsh

# 2. 安装 Oh My Zsh (非交互模式)
sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)" "" --unattended

# 3. 设置 zsh 为默认 shell
sudo chsh -s $(which zsh) $(whoami)
```

## Installation Workflow

### Step 1: 安装 Zsh

```bash
sudo apt-get update && sudo apt-get install -y zsh
```

验证安装:

```bash
zsh --version
# 预期: zsh 5.9 (x86_64-ubuntu-linux-gnu) 或更高
```

### Step 2: 安装 Oh My Zsh

```bash
# 非交互安装，不自动切换 shell
sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)" "" --unattended
```

**注意:** `--unattended` 参数避免交互提示，适合脚本化安装。

### Step 3: 配置 .zshrc

安装完成后，写入标准配置:

```bash
cat > ~/.zshrc << 'ZSHRC_EOF'
# Path to your Oh My Zsh installation.
export ZSH="$HOME/.oh-my-zsh"

# Theme: agnoster (需要 Powerline 字体)
ZSH_THEME="agnoster"

# Plugins
plugins=(git)

# Agnoster theme: custom color scheme
AGNOSTER_GIT_CLEAN_BG=yellow
AGNOSTER_GIT_CLEAN_FG=black

source $ZSH/oh-my-zsh.sh

# Claude Code PATH
export PATH="$HOME/.local/bin:$PATH"

# NVM (如果已安装)
export NVM_DIR="$HOME/.nvm"
[ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"
[ -s "$NVM_DIR/bash_completion" ] && \. "$NVM_DIR/bash_completion"
ZSHRC_EOF
```

### Step 4: 设置默认 Shell

```bash
sudo chsh -s $(which zsh) $(whoami)
```

### Step 5: 验证安装

```bash
# 验证默认 shell
echo $SHELL
# 预期: /usr/bin/zsh

# 验证 Oh My Zsh
ls ~/.oh-my-zsh/oh-my-zsh.sh && echo "Oh My Zsh OK"

# 验证主题
grep 'ZSH_THEME' ~/.zshrc
# 预期: ZSH_THEME="agnoster"
```

## Configuration Details

### 主题: agnoster

agnoster 是一个信息丰富的 Powerline 风格主题，显示:
- 用户名和主机名
- 当前目录
- Git 分支和状态
- 返回值指示

自定义颜色:
```bash
# Git clean 状态使用黄色背景 + 黑色文字
AGNOSTER_GIT_CLEAN_BG=yellow
AGNOSTER_GIT_CLEAN_FG=black
```

### 插件: git

默认启用 `git` 插件，提供常用 git 别名:
- `ga` = `git add`
- `gc` = `git commit`
- `gco` = `git checkout`
- `gd` = `git diff`
- `gl` = `git pull`
- `gp` = `git push`
- `gst` = `git status`

### 环境变量

| Variable | Value | Description |
|----------|-------|-------------|
| ZSH | $HOME/.oh-my-zsh | Oh My Zsh 安装路径 |
| ZSH_THEME | agnoster | 使用的主题 |
| PATH | $HOME/.local/bin:$PATH | Claude Code 工具路径 |
| NVM_DIR | $HOME/.nvm | NVM 安装路径 (如果有) |

## Troubleshooting

### 问题: agnoster 主题显示乱码

**原因:** 缺少 Powerline 字体

**解决方案:**
```bash
# 安装 Powerline 字体
sudo apt-get install -y fonts-powerline
```

或手动安装:
```bash
git clone https://github.com/powerline/fonts.git --depth=1
cd fonts && ./install.sh
cd .. && rm -rf fonts
```

### 问题: Oh My Zsh 安装失败

**诊断:**
```bash
# 检查 curl 是否可用
which curl || sudo apt-get install -y curl

# 检查 git 是否可用
which git || sudo apt-get install -y git
```

### 问题: chsh 失败

**解决方案:**
```bash
# 确保 zsh 在 /etc/shells 中
grep -q "$(which zsh)" /etc/shells || echo "$(which zsh)" | sudo tee -a /etc/shells

# 再次尝试
sudo chsh -s $(which zsh) $(whoami)
```

### 问题: 新终端没有加载 zsh

**解决方案:**
```bash
# 检查当前 shell
echo $SHELL

# 如果不是 zsh，手动设置
sudo chsh -s /usr/bin/zsh $(whoami)

# 重新登录或手动启动 zsh
exec zsh
```

## Verification Checklist

安装完成后的验证清单:

```bash
echo "=== Zsh + Oh My Zsh 验证 ==="
echo "1. Zsh version: $(zsh --version)"
echo "2. Default shell: $SHELL"
echo "3. Oh My Zsh: $(test -d ~/.oh-my-zsh && echo 'installed' || echo 'NOT installed')"
echo "4. Theme: $(grep 'ZSH_THEME=' ~/.zshrc | head -1)"
echo "5. Plugins: $(grep 'plugins=' ~/.zshrc | head -1)"
```
