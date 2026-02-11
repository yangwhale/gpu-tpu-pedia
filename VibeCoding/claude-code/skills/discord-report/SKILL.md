---
name: discord-report
description: 通过 Discord Bot 将任务报告发送到 Discord 频道。当用户说"discord通知我"、"ds通知我"、"发discord"、"发ds"等关键词时触发此技能。
license: MIT
---

# Discord 报告发送

当用户请求将报告发送到 Discord 时，使用此技能。通过已部署的 Discord Bot 将消息发送到指定频道。

## 触发条件

当用户的请求中包含以下关键词时触发：
- "discord通知我"
- "ds通知我"
- "发discord"
- "发ds"
- "discord发给我"
- "ds发给我"
- "发到discord"
- "发到ds"
- "用discord通知"
- "用ds通知"

## 前置条件

Discord Bot 必须已经部署并运行。检查方式：

```bash
ps aux | grep bot_simple | grep -v grep
```

如果 Bot 未运行，先使用 `discord-bot-setup` 技能部署 Bot。

## 工作流程

### 第一步：生成报告内容

根据当前会话中完成的任务，生成 Markdown 格式的报告。报告应包含：

```markdown
## 任务摘要
[简要描述完成了什么任务]

## 完成的工作
- [工作项 1]
- [工作项 2]
- [工作项 3]

## 关键变更
[列出主要的代码/配置变更]

## 注意事项
[任何需要用户注意的问题或后续步骤]
```

### 第二步：调用发送脚本

使用 Bash 工具调用发送脚本：

```bash
~/.claude/scripts/send-to-discord.sh "报告内容" "标题"
```

或者通过 stdin 传递内容：

```bash
cat << 'EOF' | ~/.claude/scripts/send-to-discord.sh "" "标题"
## 任务摘要
完成了用户认证功能的开发...

## 完成的工作
- 添加了 login 接口
- 实现了 JWT 验证
EOF
```

#### 带图片发送

支持发送网络图片或本地图片：

```bash
# 网络图片 URL
~/.claude/scripts/send-to-discord.sh "报告内容" "标题" "https://example.com/image.png"

# 本地图片文件
~/.claude/scripts/send-to-discord.sh "报告内容" "标题" "/path/to/screenshot.png"
```

### 第三步：确认发送结果

脚本会返回：
- `✅ 报告已发送到 Discord` - 成功
- `❌ 发送失败: ...` - 失败，显示错误信息

## Discord 消息格式

发送的消息显示为 Embed 卡片格式：
- 蓝色侧边条（颜色值 5814783 / #58ACFF）
- 标题栏显示报告标题
- 正文支持 Discord Markdown
- 底部注明 "Claude Code"
- 可嵌入图片

## 支持的 Markdown 语法

Discord Embed 支持以下 Markdown：
- **粗体** `**text**`
- *斜体* `*text*`
- ~~删除线~~ `~~text~~`
- `行内代码`
- 代码块（带语法高亮）
- [链接](url) `[text](url)`
- 列表（无序 `-`，有序 `1.`）
- 引用 `>`
- 标题 `##`（仅一级效果好）

**不支持**：表格（同飞书一样不支持表格渲染）

### 表格数据的替代方案

与飞书类似，Discord Embed 不支持表格语法。替代方案：

**方案 1：使用列表格式（推荐）**
```markdown
**128 并发**
- Output: 1,159 tok/s
- TTFT: 1.73s
- P99 TTFT: 6.23s
```

**方案 2：使用代码块（适合对齐数据）**
````markdown
```
并发  Output(tok/s)  TTFT(s)  TPOT(s)
64    593.7          1.20     0.105
128   1,159.0        1.73     0.106
```
````

## 内容长度限制

- 单条 Embed 描述最大 4096 字符
- 脚本会自动分片发送超长内容（每片 4000 字符）
- 分片标题自动添加"(续)"后缀

## 示例

### 示例 1：任务完成后发送报告

用户："帮我修复登录 bug，完成后用 discord 通知我"

执行：
1. 修复 bug
2. 生成报告内容
3. 调用 `send-to-discord.sh` 发送

### 示例 2：带图片发送

用户："跑个性能测试，结果截图用 ds 发给我"

执行：
1. 运行性能测试
2. 截图保存到本地
3. 调用 `send-to-discord.sh` 带图片发送

### 示例 3：发送自定义内容

用户："把今天的工作总结发到 ds"

执行：
1. 总结当前会话的所有工作
2. 格式化为 Markdown
3. 发送到 Discord

## 配置

Bot Token 和 Channel ID 已配置在脚本中：
```
~/.claude/scripts/send-to-discord.sh
```

当前配置：
- **频道**: `🥶-claude-code` (Vibe Coding 服务器)
- **Channel ID**: `1471088850712531055`

如需修改目标频道，编辑脚本中的 `CHANNEL_ID` 变量。

## 故障排查

### 发送失败

1. 检查 Bot 是否在运行：`ps aux | grep bot_simple | grep -v grep`
2. 检查 Bot Token 是否有效
3. 检查 Bot 是否有目标频道的发送权限
4. 查看返回的错误信息

### Bot 未运行

```bash
setsid python3 ~/.claude/discord-bot/bot_simple.py < /dev/null >> ~/.claude/discord-bot/bot.log 2>&1 &
disown
```

### 图片发送失败

- 本地文件：确认文件路径存在且可读
- 网络 URL：确认 URL 可公开访问（Discord 需要能下载该图片）
- 文件大小：Discord 限制上传文件最大 25MB
