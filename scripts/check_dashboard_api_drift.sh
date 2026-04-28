#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

"$ROOT/scripts/generate_dashboard_api_types.sh"

cd "$ROOT"
git diff --exit-code openapi.json frontend/openapi.json frontend/src/generated/api.ts

echo "Dashboard API artifacts are up to date."
