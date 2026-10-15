#!/usr/bin/env bash
# Keep the existing entry point; graph inspection and timestamps are portable.
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
exec uv run --no-sync python scripts/check_extension_freshness.py
