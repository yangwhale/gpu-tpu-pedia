# 语义记忆系统 (Mem0 + Vertex AI)

## 架构
- **Mem0 OSS** — 记忆管理层（fact 提取、去重、生命周期）
- **Gemini 3 Flash Preview** — LLM，用于 fact 提取和记忆合并（Vertex AI ADC 认证，region=global）
- **Vertex AI text-embedding-004** — Embedding（768 维）
- **Vertex AI Vector Search** — 向量存储（ScaNN，STREAM_UPDATE）
- **FastMCP** — MCP server 框架，暴露 4 个 tools 给 CC

## GCP 资源
- 项目: gpu-launchpad-playground (604327164091)
- Region: asia-southeast1 (Vector Search), global (Gemini LLM)
- Index: 187185257559097344 (chrisya-cc-memory-index-v2, STREAM_UPDATE)
- Endpoint: 1258377863850098688
- Public Domain: 2092452504.asia-southeast1-604327164091.vdb.vertexai.goog
- Deployed Index ID: cc_memory_v2

## MCP Server
- 代码: ~/mcp-memory-server/server.py
- venv: ~/mcp-memory-server/.venv/
- 注册: ~/.claude.json → mcpServers → cc-memory (user scope)
- Tools: memory_store, memory_search, memory_list, memory_delete
- 认证: Vertex AI ADC（无 API key），Gemini LLM 走 global region

## Mem0 Patches（升级 mem0ai 后需重新 apply）
1. `mem0/llms/gemini.py` — 无 API key 时 fallback 到 vertexai=True（读 GOOGLE_CLOUD_PROJECT + GOOGLE_CLOUD_LOCATION 环境变量）
2. `mem0/memory/setup.py` — 默认维度 1536→768
3. `mem0/vector_stores/vertex_ai_vector_search.py` — list() 方法 search 调用签名修复

## 去重优化
- `custom_update_memory_prompt`: 自定义 prompt 降低去重激进度，解决方案类内容优先 ADD
- `memory_store_raw`: 新增工具，`infer=False` 强制存储，绕过 LLM 去重（当 memory_store 误杀时用）
- 存储策略: 默认用 memory_store（智能提取），被拒绝时自动 fallback 到 memory_store_raw

## 已知问题
- Gemini 3 Flash Preview 仅在 `global` region 可用，其他 region（us-central1 等）返回 404
- Gemini 3 Flash Preview 偶尔产生 malformed JSON，导致 Mem0 fact 提取失败（预计 GA 后改善）

## 进度
- [x] Vertex AI Vector Search Index + Endpoint
- [x] Mem0 集成（Gemini + Vertex AI Embedding + Vector Search）
- [x] MCP server 构建 + 注册
- [x] v1 index 清理
- [x] 新 session 端到端验证
- [x] LLM 升级到 Gemini 3 Flash Preview
- [x] 开发经验教训存入向量记忆
- [x] 去重策略优化 + memory_store_raw fallback
- [x] 内容质量整理（清理低价值条目 + 补充高价值经验）
