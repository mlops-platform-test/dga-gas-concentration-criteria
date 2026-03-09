#!/bin/bash
# =============================================================================
# Bootstrap Platform Repository
# =============================================================================
# ../mlops-platform이 없으면 clone, 있으면 pull
#
# 환경변수로 override 가능:
#   MLOPS_PLATFORM_REPO_URL - 플랫폼 레포 URL
#   MLOPS_PLATFORM_REF      - 브랜치/태그
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
WORKSPACE_DIR="$(dirname "$REPO_ROOT")"

# Settings (환경변수 우선, 없으면 기본값)
REPO_URL="${MLOPS_PLATFORM_REPO_URL:-https://github.com/ATG-AMS/mlops-platform.git}"
REF="${MLOPS_PLATFORM_REF:-main}"
PLATFORM_DIR="$WORKSPACE_DIR/mlops-platform"

echo "==================================================="
echo "Bootstrap MLOps Platform Repository"
echo "==================================================="
echo "Repo URL: $REPO_URL"
echo "Ref: $REF"
echo "Target: $PLATFORM_DIR"
echo ""

if [ ! -d "$PLATFORM_DIR" ]; then
    echo "[CLONE] Cloning mlops-platform..."
    git clone --depth 1 --branch "$REF" "$REPO_URL" "$PLATFORM_DIR"
    echo "[OK] Cloned successfully"
else
    if [ ! -d "$PLATFORM_DIR/.git" ]; then
        echo "[ERROR] $PLATFORM_DIR exists but is not a git repository"
        exit 1
    fi
    echo "[PULL] Updating mlops-platform..."
    cd "$PLATFORM_DIR"
    git fetch origin
    git checkout "$REF"
    git pull origin "$REF"
    echo "[OK] Updated successfully"
fi

echo ""
echo "Platform ready at: $PLATFORM_DIR"
