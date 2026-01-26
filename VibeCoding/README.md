# VibeCoding - AI 编程工具资源库

本目录用于集中管理 Claude Code 的安装脚本、配置文件、插件和自定义 Skills。

## 📁 目录结构

```
VibeCoding/
├── install-claude-code.sh          # 一键安装脚本
├── README.md                        # 本文档
└── claude-code/                     # Claude Code 配置
    ├── config/                      # 配置文件
    │   ├── settings.template.json   # 设置模板（敏感信息已参数化）
    │   ├── marketplaces.json        # 插件市场列表
    │   └── plugins.txt              # 要安装的插件列表
    └── skills/                      # 自定义 Skills
        └── paper-explainer/         # 大白话论文解读 Skill
            └── SKILL.md
```

## 🚀 快速安装

```bash
# 赋予执行权限
chmod +x install-claude-code.sh

# 运行安装脚本
./install-claude-code.sh
```

## 📦 安装脚本功能

安装脚本会自动完成以下操作：

1. **安装 Claude Code** - 使用官方原生安装方式
2. **安装 Node.js v20** - 用于运行 MCP 服务器和插件
3. **配置 Vertex AI** - 交互式输入 Project ID
4. **配置 API Keys** - 可选输入 Context7 和 GitHub Token
5. **添加插件市场** - 5 个官方和社区市场
6. **安装插件** - 从 `plugins.txt` 批量安装 (18 个)
7. **安装 Happy Coder** - npm 全局安装
8. **安装自定义 Skills** - 复制到 `~/.claude/skills/`

## 🔌 预配置插件

### 官方插件 (claude-plugins-official)
- `ralph-loop` - 循环执行任务
- `explanatory-output-style` - 解释性输出风格
- `pyright-lsp` - Python 语言服务
- `context7` - 上下文增强
- `huggingface-skills` - HuggingFace 集成
- `github` - GitHub 操作
- `commit-commands` - Git 提交命令
- `playwright` - 浏览器自动化
- `Notion` - Notion 集成

### 社区技能 (awesome-claude-skills)
- `skill-creator` - 技能创建器
- `document-skills-*` - 文档处理 (docx/pdf/pptx/xlsx)
- `video-downloader` - 视频下载

### 第三方插件
- `planning-with-files` - 文件规划
- `everything-claude-code` - 综合插件集
- `ui-ux-pro-max` - UI/UX 设计增强

## 🎯 自定义 Skills

### paper-explainer - 大白话论文解读

将学术论文翻译成通俗易懂的中文解读文档，特点：
- 自动创建规范的文件结构
- 生成 SVG 配图
- 知识点补充框解释专业术语
- 示例代码和公式大白话翻译

触发方式：提供 PDF 论文并说"解读论文"或"大白话解读"

## ⚙️ 配置说明

### settings.template.json

使用 Vertex AI 模式的配置模板：

```json
{
  "env": {
    "CLAUDE_CODE_USE_VERTEX": "1",
    "CLOUD_ML_REGION": "asia-southeast1",
    "ANTHROPIC_VERTEX_PROJECT_ID": "${PROJECT_ID}",
    "ANTHROPIC_MODEL": "claude-opus-4-5@20251101",
    "CONTEXT7_API_KEY": "${CONTEXT7_API_KEY}",
    "GITHUB_PERSONAL_ACCESS_TOKEN": "${GITHUB_TOKEN}"
  }
}
```

安装时会交互式提示输入：
- **PROJECT_ID** (必需) - Google Cloud 项目 ID
- **CONTEXT7_API_KEY** (可选) - 获取地址: https://context7.io/
- **GITHUB_TOKEN** (可选) - 获取地址: https://github.com/settings/tokens (需要 repo, read:org, read:user 权限)

### 插件认证说明

| 插件 | 认证方式 | 说明 |
|------|---------|------|
| GitHub | Token | 安装时配置 `GITHUB_PERSONAL_ACCESS_TOKEN` |
| Context7 | API Key | 安装时配置 `CONTEXT7_API_KEY` |
| Notion | OAuth | 运行时在浏览器中授权 |
| Playwright | 无需认证 | 自动工作 |

### 添加新插件

编辑 `claude-code/config/plugins.txt`，每行一个插件：

```
plugin_name@marketplace_name
```

### 添加新 Skill

在 `claude-code/skills/` 下创建新目录：

```
skills/
└── my-skill/
    └── SKILL.md
```

## 🔗 相关资源

- [Claude Code 官方文档](https://docs.anthropic.com/claude-code)
- [MCP 协议规范](https://modelcontextprotocol.io/)
- [claude-plugins-official](https://github.com/anthropics/claude-plugins-official)
- [awesome-claude-skills](https://github.com/ComposioHQ/awesome-claude-skills)
