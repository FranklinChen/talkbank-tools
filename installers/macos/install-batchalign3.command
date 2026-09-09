#!/bin/bash
# install-batchalign3.command: one-click Batchalign3 installer for macOS.
#
# Double-click in Finder. This runs the canonical GitHub-release installer,
# which installs the uv package manager if needed and then installs the
# batchalign3 CLI. There is no PyPI package; distribution is via GitHub
# releases. The first install downloads large ML dependencies.
#
# After installation, open a new Terminal window and run:
#   batchalign3 --help
set -euo pipefail

echo "============================================"
echo "  Batchalign3 Installer for macOS"
echo "============================================"
echo ""

curl --proto '=https' --tlsv1.2 -LsSf \
    https://github.com/FranklinChen/talkbank-tools/releases/latest/download/install-batchalign3.sh | sh

echo ""
echo "============================================"
echo "  Open a NEW Terminal window and run:"
echo "    batchalign3 --help"
echo "============================================"
echo ""
[ "${CI:-}" = "true" ] || read -rp "Press Enter to close..."
