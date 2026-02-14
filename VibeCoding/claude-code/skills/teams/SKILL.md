---
name: teams
description: 查看 Agent Teams 状态。当用户说"teams"、"团队状态"、"看看团队"、"team status"时触发。
---

# Agent Teams 状态查看

查看当前所有活跃 Agent Team 的组织结构和任务状态。

## 触发条件

当用户输入 `/teams` 或包含以下关键词时触发：
- "teams"、"团队状态"、"看看团队"、"team status"

## 执行步骤

1. 读取 `~/.claude/teams/` 目录获取活跃 team 列表
2. 读取 `~/.claude/tasks/{team-name}/` 获取任务状态
3. 生成文字摘要发送到 Discord
4. 附上 Dashboard 链接

## 具体操作

### 步骤 1: 尝试通过 API 获取
```bash
curl -s --max-time 3 localhost:8787/api/summary
```

如果 API 返回正常（有 `summary` 字段），直接用返回的 `summary` 内容。

### 步骤 2: API 不可用时的 Fallback

如果 curl 失败或超时，直接读文件系统：
```bash
# 列出活跃 team
ls ~/.claude/teams/

# 对每个 team 读取 config 和 tasks
cat ~/.claude/teams/{name}/config.json   # 成员列表
ls ~/.claude/tasks/{name}/               # 任务文件
```

手动拼摘要：读 config.json 里的 members，读 tasks 里每个 json 的 status，汇总输出。

同时尝试重启后端服务：
```bash
tmux kill-session -t cc-dashboard 2>/dev/null
tmux new-session -d -s cc-dashboard "cd ~/cc-dashboard && python3 -m uvicorn server:app --host 127.0.0.1 --port 8787 --log-level info"
```

### 步骤 3: 输出格式

**无活跃 team 时：**
```
当前没有活跃的 Agent Team。

用 /agent-teams 可以创建新团队。
```

**有活跃 team 时：**
```
**Agent Teams 状态**

{summary 内容}

Dashboard: https://cc.higcp.com/dashboard/
```

## Dashboard

实时监控页面：`https://cc.higcp.com/dashboard/`
- 树状组织架构图
- SSE 实时状态更新
- 点击节点查看 agent 日志

## 后端服务

Dashboard API 跑在 tmux session `cc-dashboard`（port 8787）：
- 启动: `~/cc-dashboard/start.sh`
- 健康检查: `curl -s localhost:8787/api/health`
