#!/usr/bin/env bash
# 本机开发用。生产部署见 README §9.5（systemd + 跳板机反代）。
set -euo pipefail
cd "$(dirname "$0")/.."
exec python3 -m uvicorn tpuguru.backend.app:app --host 127.0.0.1 --port "${TPUGURU_PORT:-8820}" "$@"
