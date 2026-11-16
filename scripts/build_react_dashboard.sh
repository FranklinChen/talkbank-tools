#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FRONTEND_DIR="$ROOT/frontend"
TARGET_DIR="${1:-${BATCHALIGN_DASHBOARD_DIR:-$HOME/.batchalign3/dashboard}}"

if [[ -z "$TARGET_DIR" || "$TARGET_DIR" == "/" ]]; then
  echo "error: refusing to deploy dashboard to '$TARGET_DIR'" >&2
  exit 2
fi

if [[ "${BATCHALIGN_SKIP_API_SYNC:-0}" == "1" ]]; then
  python3 "$ROOT/scripts/frontend_workflow.py" build-existing
else
  python3 "$ROOT/scripts/frontend_workflow.py" build
fi

cd "$FRONTEND_DIR"
mkdir -p "$TARGET_DIR"
# ${TARGET_DIR:?} aborts if the variable is somehow empty, so the glob can
# never expand to /* (TARGET_DIR is already validated non-empty above).
rm -rf "${TARGET_DIR:?}"/*
cp -R "$FRONTEND_DIR/dist"/. "$TARGET_DIR"/

echo "React dashboard deployed to: $TARGET_DIR"
echo "Serve with batchalign server via BATCHALIGN_DASHBOARD_DIR=$TARGET_DIR (or default ~/.batchalign3/dashboard)."
