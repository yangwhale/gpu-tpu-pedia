---
name: agent-teams
description: 创建和管理 Agent Teams 并行任务团队。当用户说"建个团队"、"拉个队伍"、"并行跑"、"create team"、"spawn agents"、"组队"时触发。
---

# Agent Teams 并行任务框架

## 概述
使用 Claude Code 的 Agent Teams 功能创建多 agent 团队，支持并行任务执行和异步 Discord 通知。

## 核心流程

```
TeamCreate → 并行 Task(background, bypassPermissions) → agent 完成后直接 Discord 通知 → lead 汇总 → shutdown → TeamDelete
```

## 创建团队

```
TeamCreate → team_name, description
```

## 启动 Teammate

每个 teammate 用 Task tool 启动，必须带以下参数：
- `team_name`: 团队名
- `name`: 成员名
- `subagent_type`: "general-purpose"
- `mode`: "bypassPermissions"（必须，否则权限弹窗卡住）
- `run_in_background`: true

## Teammate Prompt 模板

```
你是「{名字}」，{team_name} 团队成员，通用型人才。

启动后立刻做这两件事：
1. 用 Bash 执行：~/.claude/scripts/send-to-discord.sh --plain "【{名字}】已上线，待命中"
2. 用 SendMessage 告诉 team-lead 你已就位

然后等待 team-lead 分配任务。以后每次完成任务都要：先 send-to-discord.sh 通知用户，再 SendMessage 汇报 team-lead。
```

## 异步通知机制

**关键设计**：agent 完成任务后**直接** `send-to-discord.sh` 通知用户，不经 lead 中转。

通知链：
- `agent → Discord`（直达用户，可靠）
- `agent → lead`（SendMessage 记录，可能延迟）

为什么不走 lead 中转：
- lead 的 turn 队列会积压 teammate 的 idle/shutdown 消息
- 用户消息和 teammate 消息无优先级区分
- 直达 Discord 更可靠，延迟更低

## 派活方式

```
SendMessage → type: "message", recipient: "{name}", content: "任务描述"
```

## 关闭团队

```
SendMessage → type: "shutdown_request" → 每个成员
等 shutdown_approved
TeamDelete
```

## 注意事项

1. **幽灵队员**: restart 后 config 残留但进程已死，必须 `rm -rf ~/.claude/teams/{name}` 清理
2. **Team vs Sub Agent**: 独立并行任务用 sub agent；需要成员间通信协作才用 team
3. **消息错位**: 框架限制，teammate 状态消息和用户消息同队列无优先级，通过直达 Discord 缓解
4. **并行启动**: 多个 Task 调用放在同一个 message 中并行发出
