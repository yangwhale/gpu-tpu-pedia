---
name: agent-teams
description: 创建和管理 Agent Teams 并行任务团队。当用户说"建个团队"、"拉个队伍"、"并行跑"、"create team"、"spawn agents"、"组队"时触发。
---

# Agent Teams 并行任务框架

## 概述
使用 Claude Code 的 Agent Teams 功能创建多 agent 团队，支持并行任务执行和异步 Discord 通知。

## 核心流程

```
TeamCreate → 并行 Task(background, bypassPermissions) → agent 完成后通知 lead → lead 汇总/决策 → lead 通知用户 → shutdown → TeamDelete
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

**日志规则**（必须遵守）：
- 日志文件：~/.claude/teams/{team_name}/logs/{名字}.log
- 启动时创建目录：mkdir -p ~/.claude/teams/{team_name}/logs
- 每次执行操作前后都用 Bash 追加日志：
  echo "[$(date '+%H:%M:%S')] {动作描述}" >> ~/.claude/teams/{team_name}/logs/{名字}.log
- 日志内容包括：收到任务、开始执行、执行进度、完成结果、遇到错误

启动后立刻做两件事：
1. mkdir -p ~/.claude/teams/{team_name}/logs
2. echo "[$(date '+%H:%M:%S')] 【{名字}】已上线，待命中" >> ~/.claude/teams/{team_name}/logs/{名字}.log
3. 用 SendMessage 告诉 team-lead 你已就位

然后等待 team-lead 分配任务。

**禁止直接联系用户**：不要使用 send-to-discord.sh，不要直接给用户发消息。所有沟通必须经过 team-lead。

**任务完成后必须执行两步通知**（缺一不可）：
1. 写日志：echo "[$(date '+%H:%M:%S')] 任务完成：{结果摘要}" >> 日志文件
2. 通知 lead：SendMessage 给 team-lead，包含结果摘要和产出物路径

lead 会决定是否需要通知用户。你只对 lead 汇报。
```

## Teammate 日志

每个 teammate 有独立日志文件，方便实时追踪和 debug：

```
~/.claude/teams/{team_name}/logs/
├── shunshen.log
├── felix.log
└── chris.log
```

实时查看：`tail -f ~/.claude/teams/{team_name}/logs/*.log`
单人查看：`tail -f ~/.claude/teams/{team_name}/logs/shunshen.log`

## 通知机制

**核心原则**：组员 → Lead → 用户。组员**禁止**直接联系用户。

通知链：
- `agent → lead`（SendMessage，唯一合法路径）
- `lead → 用户`（lead 汇总/过滤后决定是否通知用户）

Lead 的职责：
- 收集组员汇报，掌握全局进度
- 过滤噪音，只在需要用户决策或通知最终结果时才打扰用户
- 使用 `send-to-discord.sh` 通知用户（只有 lead 有权使用）

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
