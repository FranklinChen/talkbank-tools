#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FRONTEND_DIR="$ROOT/frontend"
TARGET_DIR="${1:-${BATCHALIGN_DASHBOARD_DIR:-$HOME/.batchalign3/dashboard}}"

if [[ -z "$TARGET_DIR" || "$TARGET_DIR" == "/" ]]; then
  echo "error: refusing to deploy dashboard to '$TARGET_DIR'" >&2
  exit 2
fi

if ! command -v npm >/dev/null 2>&1; then
  echo "error: npm not found. Install Node.js + npm first." >&2
  exit 127
fi

if [[ "${BATCHALIGN_SKIP_API_SYNC:-0}" != "1" ]]; then
  "$ROOT/scripts/generate_dashboard_api_types.sh"
fi

cd "$FRONTEND_DIR"
if [[ ! -d node_modules ]]; then
  npm ci
fi

npm run build

mkdir -p "$TARGET_DIR"
rm -rf "$TARGET_DIR"/*
cp -R "$FRONTEND_DIR/dist"/. "$TARGET_DIR"/

echo "React dashboard deployed to: $TARGET_DIR"
echo "Serve with batchalign server via BATCHALIGN_DASHBOARD_DIR=$TARGET_DIR (or default ~/.batchalign3/dashboard)."
