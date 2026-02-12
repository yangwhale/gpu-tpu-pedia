# VibeCoding - GPU/TPU 基础设施的 Claude Code 工具箱

一键部署 Claude Code 作为 AI 基础设施助手。包含 17 个自定义 skill、11 个插件、Vertex AI 集成，专为 Google Cloud 上的 GPU/TPU 训练和推理工作流设计。

## 目录结构

```
VibeCoding/
├── install-claude-code.sh              # 一键安装脚本
└── claude-code/
    ├── config/
    │   ├── settings.template.json      # Vertex AI 配置（已参数化）
    │   ├── marketplaces.json           # 插件市场源
    │   └── plugins.txt                 # 要安装的插件列表
    └── skills/                         # 17 个自定义 Skill
        ├── sglang-installer/           # SGLang 推理服务器
        ├── vllm-installer/             # vLLM 推理服务器
        ├── deepep-installer/           # DeepSeek Expert Parallelism
        ├── lssd-mounter/               # Google Cloud Local SSD RAID
        ├── tpu-trainer/                # TPU v7 训练自动化
        ├── discord-bot-setup/          # Discord Bot（持久 Claude 进程）
        ├── parallel-ssh/               # GPU 集群批量 SSH
        ├── tmux-installer/             # Tmux + Oh My Tmux (Tokyo Night)
        ├── zsh-installer/              # Zsh + Oh My Zsh (agnoster)
        ├── paper-explainer/            # 学术论文大白话中文解读
        ├── skill-creator/              # Anthropic 官方 Skill 创建元工具
        ├── frontend-slides/            # HTML 演示文稿生成（零依赖动画）
        ├── chip-performance-test.md    # GPU/TPU 芯片性能测试
        ├── report/                     # 客户贡献报告管理
        ├── discord-report/             # 发送报告到 Discord
        ├── feishu-report/              # 发送报告到飞书
        └── wechat-report/              # 发送报告到微信
```

## 快速安装

```bash
chmod +x install-claude-code.sh
./install-claude-code.sh
```

安装脚本会自动完成：
1. 安装 Claude Code（原生二进制）
2. 安装 Node.js v20（MCP 服务器和插件依赖）
3. 配置 Vertex AI（交互式输入 Project ID）
4. 配置 API Keys（Context7、GitHub Token、Jina AI）
5. 添加 5 个插件市场并安装 11 个插件
6. 软链接所有自定义 Skill 到 `~/.claude/skills/`

## Skill 一览

### 推理服务器

| Skill | 功能 |
|-------|------|
| **sglang-installer** | 从源码安装 SGLang v0.5.8，预编译 DeepGEMM。支持 NIXL/Mooncake 传输后端、Prefill-Decode 分离部署、DeepSeek-V3/R1 FP8/bf16。内含 30+ 常见错误修复方案。 |
| **vllm-installer** | 安装 vLLM v0.14.1 + FlashInfer 0.5.3。NIXL KV 传输、分离式 Prefill。Blackwell GPU 上跑 DeepSeek-V3 FP8 比 SGLang 更稳定。 |
| **deepep-installer** | 安装 DeepSeek Expert Parallelism（MoE 全对全通信）。10 阶段安装流程：CUDA → DOCA-OFED → NVSHMEM (IBGDA) → DeepEP。4 节点实测 RDMA 54-58 GB/s。 |

### 基础设施

| Skill | 功能 |
|-------|------|
| **lssd-mounter** | 自动检测并挂载 Google Cloud Local SSD 为 RAID0（`/lssd`），最高 20GB/s 吞吐。自动配置 HuggingFace 缓存目录。 |
| **tpu-trainer** | TPU v7 (Ironwood) 模型训练自动化。自动生成 MaxText 脚本、提交 XPK Workload、监控训练、收集结果。支持 FP8/BF16、FSDP 分片、多 Slice DCN。 |
| **parallel-ssh** | 并行在 GPU 集群节点上执行命令。一次性启动分布式推理、检查 GPU 状态、收集日志。 |

### 开发环境

| Skill | 功能 |
|-------|------|
| **tmux-installer** | 安装 tmux + Oh My Tmux，Tokyo Night 配色，Powerline 分隔符，自定义快捷键（`Prefix+\|` 分屏、`Prefix+i` 同步模式）。 |
| **zsh-installer** | 安装 Zsh + Oh My Zsh，agnoster 主题，git 插件，NVM 集成。 |
| **skill-creator** | Anthropic 官方的 Skill 创建元工具。YAML frontmatter、渐进式披露、验证脚本。 |

### 通信

| Skill | 功能 |
|-------|------|
| **discord-bot-setup** | 部署 Discord Bot，通过 Unix socketpair + stream-json 与 Claude Code 持久进程通信（与 VSCode 插件相同机制）。每用户独立 session、Whisper 语音转写、下拉菜单 session 历史切换。 |
| **discord-report** | 发送格式化 Embed 报告到 Discord 频道。 |
| **feishu-report** | 发送卡片格式报告到飞书。 |
| **wechat-report** | 通过 Server酱发送报告到微信。 |

### 创作工具

| Skill | 功能 |
|-------|------|
| **frontend-slides** | 生成零依赖、动画丰富的 HTML 演示文稿。支持从零创建和 PPT 转换。12 种预设风格（Bold Signal、Neon Cyber、Dark Botanical 等），"Show Don't Tell" 风格发现流程。 |
| **chip-performance-test** | GPU/TPU 芯片性能自动化测试。运行 GEMM benchmark、测量计算能力、生成性能报告。 |

### 研究与报告

| Skill | 功能 |
|-------|------|
| **paper-explainer** | 将学术论文 PDF 转为大白话中文解读文档，自动生成 SVG 配图、示例代码、知识点补充。 |
| **report** | 管理客户贡献报告（中英文双版本）。自动按类型分类：技术支持、订单促成、开源贡献、活动。 |

## 预配置插件

| 插件 | 来源 | 功能 |
|------|------|------|
| context7 | 官方 | 文档查询和上下文增强 |
| github | 官方 | GitHub Issues、PRs、Repos 操作 |
| huggingface-skills | 官方 | HuggingFace 模型/数据集操作 |
| playwright | 官方 | 浏览器自动化 |
| pyright-lsp | 官方 | Python 语言服务 |
| commit-commands | 官方 | Git 提交命令辅助 |
| explanatory-output-style | 官方 | 教学式输出模式 |
| skill-creator | 社区 | 创建和打包新 Skill |
| planning-with-files | 第三方 | 基于文件的任务规划 |
| everything-claude-code | 第三方 | 综合 agent 工具包 |
| ui-ux-pro-max | 第三方 | UI/UX 设计增强 |

## 配置说明

### Vertex AI 配置

`settings.template.json` 使用参数化变量：

```json
{
  "env": {
    "CLAUDE_CODE_USE_VERTEX": "1",
    "CLOUD_ML_REGION": "asia-southeast1",
    "ANTHROPIC_VERTEX_PROJECT_ID": "${PROJECT_ID}",
    "ANTHROPIC_MODEL": "claude-opus-4-5@20251101"
  }
}
```

安装时交互式输入：
- **PROJECT_ID**（必需）— Google Cloud 项目 ID
- **CONTEXT7_API_KEY**（可选）— 从 [context7.io](https://context7.io/) 获取
- **GITHUB_TOKEN**（可选）— 从 [GitHub Settings](https://github.com/settings/tokens) 获取（需要 `repo`、`read:org`、`read:user` 权限）

### 添加新 Skill

在 `skills/` 下创建目录，包含 `SKILL.md` 文件：

```
skills/my-skill/
├── SKILL.md              # 必需：YAML frontmatter + 指令
├── scripts/              # 可选：可执行脚本
├── references/           # 可选：参考文档（按需加载）
└── assets/               # 可选：模板、图片等资源
```

使用 `skill-creator` 技能（`/skill-creator`）按 Anthropic 标准格式创建。

### 添加新插件

编辑 `plugins.txt`，每行一个：

```
plugin_name@marketplace_name
```

## 架构亮点

### Discord Bot：持久进程模式

`discord-bot-setup` 技能部署的 Bot 使用与 VSCode Claude Code 插件完全相同的通信机制：

```
Discord 用户 ↔ Bot (py-cord) ↔ Unix socketpair ↔ Claude Code (stream-json) ↔ Anthropic API
```

关键发现：`--permission-prompt-tool stdio` 参数能让 Claude 进程保持存活。没有它，进程会在第一次响应后退出。这使得 Bot 能以完整交互模式运行，支持 auto memory、CLAUDE.md 加载和 skills——这是 `claude -p` 模式做不到的。

### 推理部署全栈

在多节点 GPU 集群上部署大型 MoE 模型（如 DeepSeek-V3 671B）的完整流程：

```
1. lssd-mounter      → 挂载 Local SSD 用于模型权重缓存
2. deepep-installer   → 安装 Expert Parallelism 通信层
3. sglang/vllm        → 部署推理服务器（Prefill-Decode 分离）
4. parallel-ssh       → 大规模集群运维
```

## 相关链接

- [Claude Code 官方文档](https://code.claude.com/docs)
- [MCP 协议](https://modelcontextprotocol.io/)
- [Anthropic Skills 仓库](https://github.com/anthropics/skills)
- [gpu-tpu-pedia](https://github.com/yangwhale/gpu-tpu-pedia)（父仓库）
