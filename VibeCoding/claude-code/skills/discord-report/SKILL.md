# Discord 报告发送

当用户完成任务并请求将报告发送到 Discord 时，使用此技能。

## 触发条件

当用户的请求中包含以下关键词时触发：
- "发到 Discord"
- "Discord 发给我"
- "发 Discord"
- "Discord 通知"
- "通知 Discord"

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

### 第三步：确认发送结果

脚本会返回：
- `✅ 报告已发送到 Discord` - 成功
- `❌ 发送失败: HTTP xxx` - 失败，显示 HTTP 状态码

## Discord 消息格式

发送的消息会显示为 Embed 格式：
- 蓝色左侧色条 (color: 3447003 = #3498DB)
- 粗体标题
- Markdown 格式正文
- 底部灰色 footer "Claude Code Auto Report"
- 自动附加时间戳

## 支持的 Markdown 语法

Discord Embed 支持以下 Markdown：
- **粗体** `**text**`
- *斜体* `*text*`
- ~~删除线~~ `~~text~~`
- `行内代码`
- 代码块（带语法高亮）
- [链接](url) `[text](url)`
- 列表（有序和无序）
- > 引用块

**不支持**：表格（用代码块或列表替代）、图片内嵌（需要 image URL）

### 表格数据的替代方案

Discord 的 Markdown 渲染器**不支持表格语法**。当需要展示结构化数据时，应采用以下替代方案：

**方案 1：使用代码块（推荐，等宽字体对齐）**
````markdown
```
配置        TFLOPS    MFU
8192³       1081.3    93.7%
16384²      1113.6    96.5%
```
````

**方案 2：使用列表格式**
```markdown
**BF16 最佳性能**
- 最高 TFLOPS: **1,113.6** (M=16384)
- 最高 MFU: **96.5%**
- 理论峰值: 1,153.5 TFLOPS/chiplet
```

**方案 3：使用粗体 + 换行**
```markdown
**v7 vs v6e**
BF16: 1,114 vs 728 TFLOPS (**1.53x**)
FP32: 671 vs 583 TFLOPS (**1.15x**)
```

## 内容长度限制

- Discord Embed description 最大 **4096 字符**
- 脚本会自动拆分超长内容为多条消息（每条 4000 字符）
- 拆分时自动添加 (1/N) 编号
- 多条消息之间间隔 1 秒避免触发 Discord rate limit

## 示例

### 示例 1：任务完成后发送报告

用户："帮我修复登录 bug，完成后发 Discord"

执行：
1. 修复 bug
2. 生成报告内容
3. 调用 `send-to-discord.sh` 发送

### 示例 2：发送自定义内容

用户："把测试结果发到 Discord"

执行：
1. 总结测试结果
2. 格式化为 Markdown（避免使用表格）
3. 发送到 Discord

## 配置

Webhook URL 已配置在脚本中：
```
~/.claude/scripts/send-to-discord.sh
```

如需修改 Webhook（换频道），编辑脚本中的 `WEBHOOK_URL` 变量。

## 故障排查

### 发送失败

1. 检查网络连接
2. 确认 Webhook URL 有效（频道未删除、Webhook 未被禁用）
3. HTTP 429 = rate limit，等待几秒重试
4. HTTP 400 = payload 格式错误，检查内容是否包含非法字符

### 内容被截断

- 单条 Embed 限制 4096 字符
- 脚本自动拆分，但拆分点可能在 Markdown 语法中间
- 建议精简报告内容到 4000 字符以内

### 格式显示异常

- 避免使用表格语法（`| col |` 不会渲染）
- 代码块需要用三个反引号包围
- 嵌套列表最多支持 2 层
