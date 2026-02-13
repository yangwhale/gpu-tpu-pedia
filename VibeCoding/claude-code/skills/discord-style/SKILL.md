---
name: discord-style
description: Discord 消息格式化规则。当 Claude Code 通过 Discord Bot 运行时自动适配 Discord 平台的消息风格，避免渲染问题。在 Discord 会话中始终遵循这些规则。当用户说"discord风格"、"discord格式"、"适配discord"时触发。
---

# Discord 消息风格指南

当通过 Discord 频道与用户交互时，遵循以下规则让消息在 Discord 中正确显示且易读。

## 核心原则

Discord 是聊天平台，不是文档。简短、直接、对话式。

## 格式规则

### 可以用

- **粗体** 强调重点
- `代码` 标记技术术语
- ```代码块``` 贴代码（带语言标记）
- 无序列表 `-` 和有序列表 `1.`
- `> 引用` 引用内容
- `||剧透||` 隐藏长输出
- `<url>` 抑制链接预览

### 不要用

- **表格** — Discord 把 `| col |` 渲染成纯文本，难看
- **标题 `##`** — 仅一级 `#` 有效果，其他层级不渲染
- **嵌套列表** — Discord 不支持缩进列表
- **图片 `![]()`** — 不渲染，用文件附件替代
- **脚注 `[^1]`** — 不支持

### 表格替代方案

代码块对齐时，表头和数据列都用英文/ASCII，避免中文双宽字符导致错位：

```
Model   VRAM         FP16         FP8
A100    80GB HBM2e   312 TFLOPS   N/A
H100    80GB HBM3    989 TFLOPS   1979 TFLOPS
B200    192GB HBM3e  2250 TFLOPS  4500 TFLOPS
```

列表格式（适合少量字段或中文标签）：

**A100 SXM**
- VRAM: 80GB HBM2e
- FP16: 312 TFLOPS

**H100 SXM**
- VRAM: 80GB HBM3
- FP16: 989 TFLOPS

## 消息长度

- Discord 单条消息上限 2000 字符
- 长回复拆分成多条，在自然断点（换行、段落）处切
- 优先发核心结论，细节按需展开

## 写作风格

- 短句为主，1-3 句话说清一件事
- 不要 "我很高兴为您..." 之类的废话
- 中文为主，技术术语保留英文
- 匹配对话的语气和节奏
- 结论先行，不要铺垫

## 代码输出

- 短代码（<10行）直接贴代码块
- 长代码输出用 `||折叠||` 或建议用户看文件
- 错误信息只贴关键行，不要整段 stack trace

## 富内容页面

复杂内容（大表格、图表、报告）不适合 Discord 消息时，生成 HTML 页面到 CC Pages：

1. 写 HTML 到 `/var/www/cc/pages/{topic}-{YYYYMMDD-HHmmss}.html`
2. 页面必须包含 OG 标签（og:title, og:description, og:image）让 Discord 显示预览
3. 用 `send-to-discord.sh --plain "https://cc.higcp.com/pages/{filename}"` 发送链接
4. **必须用 `--plain` 模式**，Embed 模式下链接不会触发 Discord 的 OG 预览

### OG 标签模板

```html
<meta property="og:title" content="页面标题">
<meta property="og:description" content="简短描述">
<meta property="og:image" content="https://cc.higcp.com/assets/og-image.png">
<meta property="og:image:width" content="1200">
<meta property="og:image:height" content="630">
<meta name="twitter:card" content="summary_large_image">
<meta name="theme-color" content="#22C55E">
```

## 进度更新

长任务中主动汇报，但不要刷屏：
- 开始时：一句话说清在做什么
- 关键节点：完成了什么 / 遇到问题
- 结束时：结果 + 变更摘要
