# prompts — 输出契约

每个场景一个 JSON schema，**agent 必须按 schema 返回**，前端直接渲染，不解析自由文本。

| 文件 | 场景 |
|---|---|
| `parse.json` | 命令智能识别：参数 + 置信度 + 不确定项 |
| `explain.json` | 结果解读：结论 + 依据 + 下一步建议 |
| `new_model.json` | 新模型起始配置：表单 + 命令 + 每条选择的理由 |
| `propose_rule.json` | 候选 lint 规则草稿 |

**共同字段**：`confidence`（high/medium/low）、`evidence`（每条结论的出处）、
`needs_confirmation`（要用户确认的项）。
