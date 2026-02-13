"""MCP Memory Server - Semantic memory powered by Mem0 + Vertex AI."""

import os
import logging

# Suppress noisy logs
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("google_genai").setLevel(logging.WARNING)

# Must set before importing mem0
os.environ.setdefault("MEM0_TELEMETRY", "false")
os.environ.setdefault("GOOGLE_CLOUD_PROJECT", "gpu-launchpad-playground")
os.environ.setdefault("GOOGLE_CLOUD_LOCATION", "us-central1")

from fastmcp import FastMCP
from mem0 import Memory

# ── Mem0 Config ────────────────────────────────────────────────────────────

CUSTOM_UPDATE_PROMPT = """
You are a memory deduplication expert. Your task is to decide how to handle a new memory
compared to existing memories. Be CONSERVATIVE with deduplication — only mark as NONE
when the new memory is truly identical in both content AND context to an existing one.

Key rules:
- If the new memory adds ANY new detail, solution, or context not in existing memories → ADD
- If the new memory describes a DIFFERENT scenario even if it mentions the same entity → ADD
- If the new memory contains a troubleshooting solution or workaround → always ADD (solutions are high-value)
- Only use NONE when the exact same fact with the same context already exists
- Use UPDATE when the new memory is strictly a better version of an existing one
- Use DELETE only when the new memory directly contradicts an existing one

Respond with a JSON object: {"memory": "text", "event": "ADD|UPDATE|DELETE|NONE", "id": "existing_id_or_null"}
"""

MEM0_CONFIG = {
    "llm": {
        "provider": "gemini",
        "config": {
            "model": "gemini-3-flash-preview",
            "temperature": 0.1,
            "max_tokens": 2000,
        },
    },
    "embedder": {
        "provider": "vertexai",
        "config": {
            "model": "text-embedding-004",
            "embedding_dims": 768,
        },
    },
    "vector_store": {
        "provider": "vertex_ai_vector_search",
        "config": {
            "project_id": "gpu-launchpad-playground",
            "project_number": "604327164091",
            "region": "asia-southeast1",
            "endpoint_id": "1258377863850098688",
            "index_id": "187185257559097344",
            "deployment_index_id": "cc_memory_v2",
            "vector_search_api_endpoint": "2092452504.asia-southeast1-604327164091.vdb.vertexai.goog",
        },
    },
    "custom_update_memory_prompt": CUSTOM_UPDATE_PROMPT,
    "version": "v1.1",
}

DEFAULT_USER_ID = "chris"

# ── Lazy init ──────────────────────────────────────────────────────────────

_memory: Memory | None = None


def _get_memory() -> Memory:
    global _memory
    if _memory is None:
        _memory = Memory.from_config(config_dict=MEM0_CONFIG)
    return _memory


# ── MCP Server ─────────────────────────────────────────────────────────────

mcp = FastMCP("CC Memory")


@mcp.tool
def memory_store(content: str, user_id: str = DEFAULT_USER_ID, metadata: dict | None = None) -> str:
    """Store information in long-term semantic memory.

    Mem0 automatically extracts key facts, deduplicates, and manages memory lifecycle.
    Use this to save important facts, decisions, debugging experiences,
    user preferences, or any information worth remembering across sessions.

    Args:
        content: The information to remember. Be descriptive and include context.
        user_id: User identifier for memory isolation (default: chris).
        metadata: Optional metadata dict (e.g. {"category": "debugging"}).

    Returns:
        Summary of what was stored.
    """
    m = _get_memory()
    result = m.add(content, user_id=user_id, metadata=metadata or {})

    memories = result.get("results", [])
    if not memories:
        return "No new facts extracted from the input."

    lines = []
    for mem in memories:
        event = mem.get("event", "ADD")
        text = mem.get("memory", "")
        lines.append(f"[{event}] {text}")

    return f"Stored {len(memories)} memory(ies):\n" + "\n".join(lines)


@mcp.tool
def memory_store_raw(content: str, user_id: str = DEFAULT_USER_ID, metadata: dict | None = None) -> str:
    """Force-store a memory without LLM deduplication.

    Use this when memory_store rejects content due to over-aggressive dedup.
    The content is stored as-is without fact extraction or dedup checks.
    Only use as fallback — prefer memory_store for normal usage.

    Args:
        content: The exact text to store as a memory entry.
        user_id: User identifier for memory isolation (default: chris).
        metadata: Optional metadata dict.

    Returns:
        Confirmation of what was stored.
    """
    m = _get_memory()
    result = m.add(content, user_id=user_id, metadata=metadata or {}, infer=False)

    memories = result.get("results", [])
    if not memories:
        return "Failed to store memory."

    lines = []
    for mem in memories:
        event = mem.get("event", "ADD")
        text = mem.get("memory", "")
        lines.append(f"[{event}] {text}")

    return f"Force-stored {len(memories)} memory(ies):\n" + "\n".join(lines)


@mcp.tool
def memory_search(query: str, user_id: str = DEFAULT_USER_ID, limit: int = 5) -> str:
    """Search long-term semantic memory for relevant information.

    Use this to recall previously stored facts, decisions, experiences,
    or any information that might be relevant to the current task.

    Args:
        query: Natural language description of what you're looking for.
        user_id: User identifier (default: chris).
        limit: Number of results (default 5, max 20).

    Returns:
        Matching memories ranked by relevance.
    """
    limit = min(max(1, limit), 20)
    m = _get_memory()
    result = m.search(query, user_id=user_id, limit=limit)

    memories = result.get("results", [])
    if not memories:
        return "No matching memories found."

    lines = []
    for i, mem in enumerate(memories, 1):
        score = mem.get("score", 0)
        text = mem.get("memory", "")
        mem_id = mem.get("id", "?")
        lines.append(f"[{i}] (score: {score:.3f}) {text}  [id: {mem_id}]")

    return "\n".join(lines)


@mcp.tool
def memory_list(user_id: str = DEFAULT_USER_ID) -> str:
    """List all stored memories for a user.

    Args:
        user_id: User identifier (default: chris).

    Returns:
        All memories for the user.
    """
    m = _get_memory()
    result = m.get_all(user_id=user_id)

    memories = result.get("results", [])
    if not memories:
        return "No memories stored."

    lines = [f"Total: {len(memories)} memories\n"]
    for mem in memories:
        text = mem.get("memory", "")
        mem_id = mem.get("id", "?")
        lines.append(f"- {text}  [id: {mem_id}]")

    return "\n".join(lines)


@mcp.tool
def memory_delete(memory_id: str) -> str:
    """Delete a specific memory by ID.

    Args:
        memory_id: The memory ID to delete (get from memory_search or memory_list).

    Returns:
        Confirmation.
    """
    m = _get_memory()
    m.delete(memory_id)
    return f"Deleted memory: {memory_id}"


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    mcp.run()


if __name__ == "__main__":
    main()
