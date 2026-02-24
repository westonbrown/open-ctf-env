#!/bin/bash
# =============================================================================
# Sync open-ctf-env to DGX Spark
# =============================================================================
# Pushes local source code, configs, and data to the DGX Spark host.
# Excludes build artifacts, caches, and large output directories.
#
# Usage:
#   bash scripts/deploy/sync_to_dgx.sh
#
# Environment overrides:
#   DGX_HOST   SSH target (default: abrown@100.91.175.48)
#   DGX_PATH   Remote directory (default: /home/abrown/open-ctf-env)
# =============================================================================

set -euo pipefail

DGX_HOST="${DGX_HOST:-abrown@100.91.175.48}"
DGX_PATH="${DGX_PATH:-/home/abrown/open-ctf-env}"

# Resolve the project root (parent of scripts/deploy/)
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOCAL_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

echo "[$(date '+%H:%M:%S')] Syncing ${LOCAL_DIR} -> ${DGX_HOST}:${DGX_PATH}"

# Verify DGX is reachable
if ! ssh -o ConnectTimeout=10 "${DGX_HOST}" 'echo OK' >/dev/null 2>&1; then
    echo "ERROR: Cannot reach DGX at ${DGX_HOST}"
    echo "Try: ssh -o StrictHostKeyChecking=accept-new ${DGX_HOST}"
    exit 1
fi

# Ensure target directory exists
ssh "${DGX_HOST}" "mkdir -p ${DGX_PATH}"

rsync -avz --progress \
    --exclude '.git' \
    --exclude 'references/' \
    --exclude '__pycache__' \
    --exclude '*.pyc' \
    --exclude 'outputs/' \
    --exclude '.venv/' \
    --exclude 'node_modules/' \
    --exclude '*.egg-info' \
    --exclude '.ruff_cache' \
    --exclude '.pytest_cache' \
    --exclude '.mypy_cache' \
    "${LOCAL_DIR}/" \
    "${DGX_HOST}:${DGX_PATH}/"

echo "[$(date '+%H:%M:%S')] Synced to ${DGX_HOST}:${DGX_PATH}"
