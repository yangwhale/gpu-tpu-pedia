#!/bin/bash
# =============================================================================
# Mem0 Patches for Vertex AI Integration
#
# 升级 mem0ai 包后需要重新运行此脚本
# Usage: ./apply-mem0-patches.sh [VENV_PATH]
#   VENV_PATH: Python venv 路径 (default: ~/mcp-memory-server/.venv)
# =============================================================================

set -e

VENV_PATH="${1:-$HOME/mcp-memory-server/.venv}"
SITE_PACKAGES=$(find "$VENV_PATH/lib" -maxdepth 1 -name 'python3.*' -type d)/site-packages

if [ ! -d "$SITE_PACKAGES/mem0" ]; then
    echo "[ERROR] mem0 not found in $SITE_PACKAGES"
    exit 1
fi

echo "[INFO] Applying Mem0 patches in $SITE_PACKAGES ..."

# ─── Patch 1: gemini.py — ADC fallback (no API key → Vertex AI mode) ─────
GEMINI_FILE="$SITE_PACKAGES/mem0/llms/gemini.py"
if grep -q 'vertexai=True' "$GEMINI_FILE" 2>/dev/null; then
    echo "[SKIP] gemini.py already patched"
else
    python3 -c "
import sys
with open(sys.argv[1]) as f:
    content = f.read()

old = '        self.client = genai.Client(api_key=api_key)'
new = '''        # Support Vertex AI mode via ADC when no API key is available
        if api_key:
            self.client = genai.Client(api_key=api_key)
        else:
            project = os.getenv(\"GOOGLE_CLOUD_PROJECT\", os.getenv(\"GCLOUD_PROJECT\"))
            location = os.getenv(\"GOOGLE_CLOUD_LOCATION\", \"us-central1\")
            self.client = genai.Client(vertexai=True, project=project, location=location)'''

if old not in content:
    print('[WARN] gemini.py: target line not found, may need manual patch')
    sys.exit(0)

content = content.replace(old, new)
with open(sys.argv[1], 'w') as f:
    f.write(content)
print('[OK] gemini.py patched')
" "$GEMINI_FILE"
fi

# ─── Patch 2: setup.py — default embedding dims 1536 → 768 ───────────────
SETUP_FILE="$SITE_PACKAGES/mem0/memory/setup.py"
if grep -q 'embedding_model_dims", 768' "$SETUP_FILE" 2>/dev/null; then
    echo "[SKIP] setup.py already patched"
else
    sed -i 's/embedding_model_dims", 1536)/embedding_model_dims", 768)/' "$SETUP_FILE"
    echo "[OK] setup.py patched (dims 1536 → 768)"
fi

# ─── Patch 3: vertex_ai_vector_search.py — list() search call fix ────────
VAS_FILE="$SITE_PACKAGES/mem0/vector_stores/vertex_ai_vector_search.py"
if grep -q 'query="", vectors=zero_vector' "$VAS_FILE" 2>/dev/null; then
    echo "[SKIP] vertex_ai_vector_search.py already patched"
else
    sed -i 's/self.search(query=zero_vector,/self.search(query="", vectors=zero_vector,/' "$VAS_FILE"
    echo "[OK] vertex_ai_vector_search.py patched (list() search signature)"
fi

echo "[DONE] All patches applied."
