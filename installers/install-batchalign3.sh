#!/usr/bin/env bash
# install-batchalign3.sh: install or update the batchalign3 CLI from the latest
# GitHub release. Bootstraps uv if absent and installs into an isolated uv tool
# environment using a uv-managed Python (default 3.13).
#
#   curl --proto '=https' --tlsv1.2 -LsSf \
#     https://github.com/FranklinChen/talkbank-tools/releases/latest/download/install-batchalign3.sh | sh
#
# Re-running upgrades an existing installation. Override the managed Python with
# BATCHALIGN3_PYTHON (for example BATCHALIGN3_PYTHON=3.13).
#
# There is no PyPI package: distribution is via GitHub releases. batchalign3's
# own dependencies still resolve from PyPI, so the first install downloads large
# ML dependencies.
set -euo pipefail

REPO="FranklinChen/talkbank-tools"
PYTHON_VERSION="${BATCHALIGN3_PYTHON:-3.13}"

die() { printf 'install-batchalign3: %s\n' "$1" >&2; exit 1; }
info() { printf 'install-batchalign3: %s\n' "$1" >&2; }

# 1. Ensure uv is available (install it if absent).
if ! command -v uv >/dev/null 2>&1; then
    info "uv not found; installing uv from astral.sh"
    curl --proto '=https' --tlsv1.2 -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi
command -v uv >/dev/null 2>&1 || die "uv installation failed; see https://docs.astral.sh/uv/"

# 2. Detect platform and map to the wheel platform tag.
os="$(uname -s)"
arch="$(uname -m)"
case "$os" in
    Darwin)
        case "$arch" in
            arm64)  plat="macosx_11_0_arm64" ;;
            x86_64) plat="macosx_10_12_x86_64" ;;
            *) die "unsupported macOS architecture: $arch" ;;
        esac ;;
    Linux)
        case "$arch" in
            x86_64)  plat="manylinux_2_28_x86_64" ;;
            aarch64) plat="manylinux_2_28_aarch64" ;;
            *) die "unsupported Linux architecture: $arch" ;;
        esac ;;
    *) die "unsupported OS: $os (on Windows use install-batchalign3.ps1)" ;;
esac

# 3. Resolve the abi3 wheel asset URL from the latest release. For testing,
# BATCHALIGN3_RELEASE_JSON_FILE supplies the release API JSON directly (so the
# resolution logic can be exercised offline), and BATCHALIGN3_RESOLVE_ONLY
# prints the resolved URL and exits before installing.
api="https://api.github.com/repos/${REPO}/releases/latest"
if [ -n "${BATCHALIGN3_RELEASE_JSON_FILE:-}" ]; then
    release_json="$(cat "$BATCHALIGN3_RELEASE_JSON_FILE")"
else
    release_json="$(curl --proto '=https' --tlsv1.2 -fsSL "$api")"
fi
wheel_url="$(printf '%s' "$release_json" | grep -o "https://[^\"]*-abi3-${plat}\\.whl" | head -1)"
[ -n "$wheel_url" ] || die "no abi3 wheel for platform ${plat} in the latest ${REPO} release"

if [ -n "${BATCHALIGN3_RESOLVE_ONLY:-}" ]; then
    printf '%s\n' "$wheel_url"
    exit 0
fi

# 4. Install or upgrade with a uv-managed Python.
info "installing ${wheel_url##*/} (Python ${PYTHON_VERSION})"
uv tool install --force --python "$PYTHON_VERSION" "$wheel_url"

# 5. Make sure the tool bin is on PATH for future shells, then report.
uv tool update-shell >/dev/null 2>&1 || true
info "done. Open a new shell and run: batchalign3 --help"
