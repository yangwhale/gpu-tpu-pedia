---
name: tmux-installer
description: This skill should be used when users need to install, configure, and customize tmux with Oh My Tmux (gpakosz/.tmux) on Linux systems. It covers tmux installation, Oh My Tmux setup, Tokyo Night color theme, Powerline separators, custom keybindings (split panes, window navigation, synchronize-panes), mouse mode, and clipboard integration. The skill replicates the standard team tmux configuration.
license: MIT
---

# Tmux + Oh My Tmux Installer

## Overview

在 Linux 系统上安装和配置 Tmux + Oh My Tmux (gpakosz/.tmux)，统一团队终端复用器环境。包括 Tokyo Night 主题配色、Powerline 分隔符、自定义快捷键和鼠标支持。

## When to Use This Skill

- 在新 VM 或服务器上安装和配置 tmux
- 用户要求安装 tmux 或 oh-my-tmux
- 需要统一的 tmux 配置
- 诊断 tmux 配置相关问题

## Prerequisites

- Ubuntu/Debian 系统 (apt-get)
- sudo 权限
- git (用于 clone oh-my-tmux)
- 网络连接

## Important: 先安装 Zsh

在安装 tmux 之前，应先安装 Zsh + Oh My Zsh。请先使用 `zsh-installer` 技能完成 shell 配置。

## Quick Start

```bash
# 1. 安装 tmux
sudo apt-get update && sudo apt-get install -y tmux

# 2. 安装 Oh My Tmux
cd ~ && git clone https://github.com/gpakosz/.tmux.git
ln -s -f .tmux/.tmux.conf ~/.tmux.conf

# 3. 写入自定义配置 (见 Step 3)
```

## Installation Workflow

### Step 1: 安装 Tmux

```bash
sudo apt-get update && sudo apt-get install -y tmux
```

验证安装:

```bash
tmux -V
# 预期: tmux 3.4 或更高
```

### Step 2: 安装 Oh My Tmux

```bash
cd ~
git clone https://github.com/gpakosz/.tmux.git
ln -s -f .tmux/.tmux.conf ~/.tmux.conf
cp .tmux/.tmux.conf.local ~/.tmux.conf.local
```

**说明:**
- `.tmux/.tmux.conf` 是 Oh My Tmux 的主配置（不要修改）
- `.tmux.conf.local` 是用户自定义配置（所有个性化设置在这里）

### Step 3: 写入自定义配置

将以下内容写入 `~/.tmux.conf.local`:

```bash
cat > ~/.tmux.conf.local << 'TMUX_EOF'
# Oh my tmux! - https://github.com/gpakosz/.tmux

# -- bindings ------------------------------------------------------------------
tmux_conf_preserve_stock_bindings=false

# -- session creation ----------------------------------------------------------
tmux_conf_new_session_prompt=false
tmux_conf_new_session_retain_current_path=false

# -- windows & pane creation ---------------------------------------------------
tmux_conf_new_window_retain_current_path=true
tmux_conf_new_window_reconnect_ssh=false
tmux_conf_new_pane_retain_current_path=true
tmux_conf_new_pane_reconnect_ssh=false

# -- display -------------------------------------------------------------------
tmux_conf_24b_colour=auto

# -- theming -------------------------------------------------------------------
tmux_conf_theme=enabled

# Tokyo Night color scheme
tmux_conf_theme_colour_1="#080808"    # dark gray
tmux_conf_theme_colour_2="#303030"    # gray
tmux_conf_theme_colour_3="#8a8a8a"    # light gray
tmux_conf_theme_colour_4="#00afff"    # light blue
tmux_conf_theme_colour_5="#e0af68"    # Tokyo Night gold
tmux_conf_theme_colour_6="#080808"    # dark gray
tmux_conf_theme_colour_7="#a9b1d6"    # Tokyo Night soft white
tmux_conf_theme_colour_8="#080808"    # dark gray
tmux_conf_theme_colour_9="#e0af68"    # Tokyo Night gold
tmux_conf_theme_colour_10="#7aa2f7"   # Tokyo Night blue
tmux_conf_theme_colour_11="#9ece6a"   # Tokyo Night green
tmux_conf_theme_colour_12="#8a8a8a"   # light gray
tmux_conf_theme_colour_13="#a9b1d6"   # Tokyo Night soft white
tmux_conf_theme_colour_14="#080808"   # dark gray
tmux_conf_theme_colour_15="#080808"   # dark gray
tmux_conf_theme_colour_16="#414868"   # Tokyo Night dark gray-blue
tmux_conf_theme_colour_17="#a9b1d6"   # Tokyo Night soft white

# window style
tmux_conf_theme_window_fg="default"
tmux_conf_theme_window_bg="default"
tmux_conf_theme_highlight_focused_pane=false
tmux_conf_theme_focused_pane_bg="$tmux_conf_theme_colour_2"

# pane border style
tmux_conf_theme_pane_border_style=thin
tmux_conf_theme_pane_border="$tmux_conf_theme_colour_2"
tmux_conf_theme_pane_active_border="$tmux_conf_theme_colour_4"
%if #{>=:#{version},3.2}
tmux_conf_theme_pane_active_border="#{?pane_in_mode,$tmux_conf_theme_colour_9,#{?synchronize-panes,$tmux_conf_theme_colour_16,$tmux_conf_theme_colour_4}}"
%endif

# pane indicator colours
tmux_conf_theme_pane_indicator="$tmux_conf_theme_colour_4"
tmux_conf_theme_pane_active_indicator="$tmux_conf_theme_colour_4"

# status line style
tmux_conf_theme_message_fg="$tmux_conf_theme_colour_1"
tmux_conf_theme_message_bg="$tmux_conf_theme_colour_5"
tmux_conf_theme_message_attr="bold"

tmux_conf_theme_message_command_fg="$tmux_conf_theme_colour_5"
tmux_conf_theme_message_command_bg="$tmux_conf_theme_colour_1"
tmux_conf_theme_message_command_attr="bold"

# window modes style
tmux_conf_theme_mode_fg="$tmux_conf_theme_colour_1"
tmux_conf_theme_mode_bg="$tmux_conf_theme_colour_5"
tmux_conf_theme_mode_attr="bold"

# status line style
tmux_conf_theme_status_fg="$tmux_conf_theme_colour_3"
tmux_conf_theme_status_bg="$tmux_conf_theme_colour_1"
tmux_conf_theme_status_attr="none"

# terminal title
tmux_conf_theme_terminal_title="#h ❐ #S ● #I #W"

# window status style
tmux_conf_theme_window_status_fg="$tmux_conf_theme_colour_3"
tmux_conf_theme_window_status_bg="$tmux_conf_theme_colour_1"
tmux_conf_theme_window_status_attr="none"
tmux_conf_theme_window_status_format="#I #W#{?#{||:#{window_bell_flag},#{window_zoomed_flag}}, ,}#{?window_bell_flag,!,}#{?window_zoomed_flag,Z,}"

# window current status style
tmux_conf_theme_window_status_current_fg="$tmux_conf_theme_colour_1"
tmux_conf_theme_window_status_current_bg="$tmux_conf_theme_colour_4"
tmux_conf_theme_window_status_current_attr="bold"
tmux_conf_theme_window_status_current_format="#I #W#{?#{||:#{window_bell_flag},#{window_zoomed_flag}}, ,}#{?window_bell_flag,!,}#{?window_zoomed_flag,Z,}"

# window activity status style
tmux_conf_theme_window_status_activity_fg="default"
tmux_conf_theme_window_status_activity_bg="default"
tmux_conf_theme_window_status_activity_attr="underscore"

# window bell status style
tmux_conf_theme_window_status_bell_fg="$tmux_conf_theme_colour_5"
tmux_conf_theme_window_status_bell_bg="default"
tmux_conf_theme_window_status_bell_attr="blink,bold"

# window last status style
tmux_conf_theme_window_status_last_fg="$tmux_conf_theme_colour_4"
tmux_conf_theme_window_status_last_bg="$tmux_conf_theme_colour_2"
tmux_conf_theme_window_status_last_attr="none"

# Powerline separators
tmux_conf_theme_left_separator_main='\uE0B0'
tmux_conf_theme_left_separator_sub='\uE0B1'
tmux_conf_theme_right_separator_main='\uE0B2'
tmux_conf_theme_right_separator_sub='\uE0B3'

# status left/right content
tmux_conf_theme_status_left=" ❐ #S | ↑#{?uptime_y, #{uptime_y}y,}#{?uptime_d, #{uptime_d}d,}#{?uptime_h, #{uptime_h}h,}#{?uptime_m, #{uptime_m}m,} "
tmux_conf_theme_status_right=" #{prefix}#{mouse}#{pairing}#{synchronized}#{?battery_status,#{battery_status},}#{?battery_bar, #{battery_bar},}#{?battery_percentage, #{battery_percentage},} , %R , %d %b | #{username}#{root} | #{hostname} "

# status left style
tmux_conf_theme_status_left_fg="$tmux_conf_theme_colour_6,$tmux_conf_theme_colour_7,$tmux_conf_theme_colour_8"
tmux_conf_theme_status_left_bg="$tmux_conf_theme_colour_9,$tmux_conf_theme_colour_10,$tmux_conf_theme_colour_11"
tmux_conf_theme_status_left_attr="bold,none,none"

# status right style
tmux_conf_theme_status_right_fg="$tmux_conf_theme_colour_12,$tmux_conf_theme_colour_13,$tmux_conf_theme_colour_14"
tmux_conf_theme_status_right_bg="$tmux_conf_theme_colour_15,$tmux_conf_theme_colour_16,$tmux_conf_theme_colour_17"
tmux_conf_theme_status_right_attr="none,none,bold"

# indicators
tmux_conf_theme_pairing="⚇"
tmux_conf_theme_pairing_fg="none"
tmux_conf_theme_pairing_bg="none"
tmux_conf_theme_pairing_attr="none"

tmux_conf_theme_prefix="⌨"
tmux_conf_theme_prefix_fg="none"
tmux_conf_theme_prefix_bg="none"
tmux_conf_theme_prefix_attr="none"

tmux_conf_theme_mouse="↗"
tmux_conf_theme_mouse_fg="none"
tmux_conf_theme_mouse_bg="none"
tmux_conf_theme_mouse_attr="none"

tmux_conf_theme_root="!"
tmux_conf_theme_root_fg="none"
tmux_conf_theme_root_bg="none"
tmux_conf_theme_root_attr="bold,blink"

tmux_conf_theme_synchronized="⚏"
tmux_conf_theme_synchronized_fg="none"
tmux_conf_theme_synchronized_bg="none"
tmux_conf_theme_synchronized_attr="none"

# battery
tmux_conf_battery_bar_symbol_full="◼"
tmux_conf_battery_bar_symbol_empty="◻"
tmux_conf_battery_bar_length="auto"
tmux_conf_battery_bar_palette="gradient"
tmux_conf_battery_hbar_palette="gradient"
tmux_conf_battery_vbar_palette="gradient"
tmux_conf_battery_status_charging="↑"
tmux_conf_battery_status_discharging="↓"

# clock
tmux_conf_theme_clock_colour="$tmux_conf_theme_colour_4"
tmux_conf_theme_clock_style="24"

# -- clipboard -----------------------------------------------------------------
tmux_conf_copy_to_os_clipboard=true

# -- urlscan -------------------------------------------------------------------
tmux_conf_urlscan_options="--compact --dedupe"

# -- user customizations -------------------------------------------------------

# increase history size
set -g history-limit 50000

# start with mouse mode enabled
set -g mouse on

# 更直观的分屏快捷键
bind | split-window -h    # 用 | 左右分屏
bind - split-window -v    # 用 - 上下分屏

# 同步模式切换 (Prefix + i)
bind-key i set-window-option synchronize-panes

# CapsLock (F13 via SSH sends \e[25~) as additional prefix trigger
set -s user-keys[0] "\e[25~"
bind -n User0 switch-client -T prefix

# window navigation: prefix + h/l
bind -r h previous-window
bind -r l next-window

# display a message after toggling mouse support
bind m run "cut -c3- '#{TMUX_CONF}' | sh -s _toggle_mouse" \; display 'mouse #{?#{mouse},on,off}'

# -- tpm -----------------------------------------------------------------------
tmux_conf_update_plugins_on_launch=true
tmux_conf_update_plugins_on_reload=true
tmux_conf_uninstall_plugins_on_reload=true
TMUX_EOF
```

### Step 4: 验证安装

```bash
# 启动 tmux 验证
tmux new-session -d -s test && tmux kill-session -t test && echo "Tmux OK"

# 检查配置文件
ls -la ~/.tmux.conf ~/.tmux.conf.local
```

## Configuration Details

### 主题: Tokyo Night

配色基于流行的 Tokyo Night 编辑器主题:

| 颜色变量 | 色值 | 用途 |
|----------|------|------|
| colour_4 | #00afff | 高亮蓝 (活动边框、当前窗口) |
| colour_5 | #e0af68 | Tokyo Night 金色 (消息、模式) |
| colour_7 | #a9b1d6 | 柔白 (文字) |
| colour_9 | #e0af68 | 金色 (状态栏左段1) |
| colour_10 | #7aa2f7 | Tokyo Night 蓝 (状态栏左段2) |
| colour_11 | #9ece6a | Tokyo Night 绿 (状态栏左段3) |

### Powerline 分隔符

使用 Powerline 特殊字符作为状态栏分隔符:
- `\uE0B0` () 和 `\uE0B1` () 用于左侧
- `\uE0B2` () 和 `\uE0B3` () 用于右侧

**需要 Powerline 字体支持** (通常 Nerd Font 已包含)。

### 自定义快捷键

| 快捷键 | 功能 | 说明 |
|--------|------|------|
| `Prefix + \|` | 水平分屏 | 比默认的 `%` 更直观 |
| `Prefix + -` | 垂直分屏 | 比默认的 `"` 更直观 |
| `Prefix + i` | 切换同步模式 | 同时向所有 pane 发送输入 |
| `Prefix + h` | 上一个窗口 | 可重复 (-r) |
| `Prefix + l` | 下一个窗口 | 可重复 (-r) |
| `Prefix + m` | 切换鼠标支持 | 显示开关状态 |
| `CapsLock (F13)` | 额外 Prefix 触发 | SSH 场景下通过 F13 映射 |

### 重要设置

| 设置 | 值 | 说明 |
|------|-----|------|
| history-limit | 50000 | 滚动缓冲区大小 |
| mouse | on | 默认启用鼠标 |
| new_window_retain_current_path | true | 新窗口保留当前路径 |
| new_pane_retain_current_path | true | 新面板保留当前路径 |
| copy_to_os_clipboard | true | 复制到系统剪贴板 |

## Common Scenarios

### Scenario 1: 在新 VM 上完整安装

```bash
# 先安装 zsh (使用 zsh-installer 技能)
sudo apt-get update && sudo apt-get install -y zsh
sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)" "" --unattended
sudo chsh -s $(which zsh) $(whoami)

# 再安装 tmux + oh-my-tmux
sudo apt-get install -y tmux
cd ~ && git clone https://github.com/gpakosz/.tmux.git
ln -s -f .tmux/.tmux.conf ~/.tmux.conf
# 然后写入 .tmux.conf.local (见 Step 3)
```

### Scenario 2: 多节点批量部署

配合 `parallel-ssh` 技能，在多台机器上同时部署:

```bash
# 使用 parallel-ssh 在所有节点执行安装脚本
# 参考 parallel-ssh 技能的用法
```

### Scenario 3: 已有 tmux 但没有 Oh My Tmux

```bash
# 备份旧配置
cp ~/.tmux.conf ~/.tmux.conf.bak 2>/dev/null

# 安装 Oh My Tmux
cd ~ && git clone https://github.com/gpakosz/.tmux.git
ln -s -f .tmux/.tmux.conf ~/.tmux.conf
# 写入 .tmux.conf.local (见 Step 3)

# 重新加载
tmux source-file ~/.tmux.conf
```

## Troubleshooting

### 问题: Powerline 字符显示为方块

**原因:** 终端缺少 Powerline/Nerd Font

**解决方案:**
```bash
sudo apt-get install -y fonts-powerline
```

或在本地终端 (iTerm2/Windows Terminal) 中设置 Nerd Font 字体。

### 问题: tmux source-file 报错

**诊断:**
```bash
tmux source-file ~/.tmux.conf 2>&1
```

**常见原因:**
- `.tmux.conf.local` 语法错误
- tmux 版本过低不支持某些特性

**解决方案:**
```bash
# 检查 tmux 版本
tmux -V

# 如果低于 3.2，移除 %if 条件块
```

### 问题: 鼠标模式不工作

**诊断:**
```bash
tmux show -g mouse
# 预期: mouse on
```

**解决方案:**
```bash
# 在 tmux 内重新加载配置
# 按 Prefix + r (Oh My Tmux 默认重载快捷键)
```

### 问题: 剪贴板不工作

**原因:** Linux 需要 xsel 或 xclip

**解决方案:**
```bash
sudo apt-get install -y xsel
# 或
sudo apt-get install -y xclip
```

### 问题: CapsLock/F13 prefix 不生效

**说明:** 这个功能需要本地终端将 CapsLock 映射为 F13 并通过 SSH 发送 `\e[25~`。

**配置方法 (macOS + Karabiner):**
1. 安装 Karabiner-Elements
2. 将 CapsLock 映射为 F13
3. iTerm2 会自动发送 F13 转义序列

## Verification Checklist

```bash
echo "=== Tmux + Oh My Tmux 验证 ==="
echo "1. Tmux version: $(tmux -V)"
echo "2. Oh My Tmux: $(test -L ~/.tmux.conf && echo 'installed (symlink)' || echo 'NOT installed')"
echo "3. Config local: $(test -f ~/.tmux.conf.local && echo 'exists' || echo 'NOT found')"
echo "4. Mouse mode: $(tmux start-server \; show -g mouse 2>/dev/null || echo 'tmux not running')"
echo "5. History limit: $(tmux start-server \; show -g history-limit 2>/dev/null || echo 'tmux not running')"
```

## Status Bar Layout

状态栏布局说明:

```
┌─────────────────────────────────────────────────────────┐
│ ❐ session | ↑ 3d 5h  │  0 zsh  1 vim* │ ⌨ 14:30 11 Feb │ user │ hostname │
│ ← 金色  → 蓝色 → 绿色 │  ← 暗灰底  →  │ ← 灰蓝底 → ← 柔白底 →            │
└─────────────────────────────────────────────────────────┘
```

- **左侧段1** (金色背景): Session 名称
- **左侧段2** (蓝色背景): 运行时间
- **左侧段3** (绿色背景): (可扩展)
- **中间**: 窗口列表 (当前窗口蓝色高亮)
- **右侧段1** (暗灰背景): 指示器 + 电池
- **右侧段2** (灰蓝背景): 时间 + 日期
- **右侧段3** (柔白背景): 用户名 + 主机名
