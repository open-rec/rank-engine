#!/usr/bin/env bash
set -euo pipefail

exec uvicorn server:app \
  --host "${RANK_HOST:-0.0.0.0}" \
  --port "${RANK_PORT:-8123}" \
  --workers "${RANK_WORKERS:-1}"
