#!/usr/bin/env bash
# Keep the existing entry point; checks consume content-bound build evidence.
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
exec uv run --no-sync python -m scripts.check_extension_freshness
