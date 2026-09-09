#!/bin/bash
# installers/test-github-release.sh: test the GitHub Release install path.
#
# Builds the abi3 wheel, creates a draft pre-release, uploads the wheel,
# downloads it back, installs via uv into an isolated sandbox, verifies the CLI
# works and upgrades, and checks that install-batchalign3.sh resolves the right
# abi3 wheel URL from the release JSON. Cleans up the draft release on exit.
# Proves the GitHub Release distribution path without touching PyPI.
#
# Prerequisites:
#   - gh CLI authenticated with repo access (`gh auth status`)
#   - uv, Rust toolchain, and this talkbank-tools checkout
#
# Usage:
#   bash installers/test-github-release.sh              # build + test
#   bash installers/test-github-release.sh --no-build   # reuse existing wheel

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DIST_DIR="$REPO_ROOT/dist"
BUILD_WHEEL=true

TAG="v0.0.0-installer-test"
REPO="FranklinChen/talkbank-tools"

for arg in "$@"; do
    case "$arg" in
        --no-build) BUILD_WHEEL=false ;;
        *) echo "Unknown argument: $arg"; exit 1 ;;
    esac
done

# ── Preflight ───────────────────────────────────────────────────────────────

echo "=== Preflight checks ==="

if ! command -v gh &>/dev/null; then
    echo "ERROR: gh CLI not found. Install from https://cli.github.com/"
    exit 1
fi

if ! gh api user --jq .login &>/dev/null; then
    echo "ERROR: gh CLI not authenticated. Run: gh auth login"
    exit 1
fi

echo "[OK] gh CLI authenticated as $(gh api user --jq .login)"

# ── Build wheel ─────────────────────────────────────────────────────────────

if [ "$BUILD_WHEEL" = true ]; then
    echo ""
    echo "=== Building wheel ==="
    cd "$REPO_ROOT"
    uv build --wheel --out-dir "$DIST_DIR/"
fi

# ls -t picks the newest wheel by mtime; wheel filenames are tool-generated
# and contain no whitespace or special characters, so ls is safe here.
# shellcheck disable=SC2012
WHEEL="$(ls -t "$DIST_DIR"/*.whl 2>/dev/null | head -1)"
if [ -z "$WHEEL" ]; then
    echo "ERROR: No wheel found in $DIST_DIR/. Run without --no-build."
    exit 1
fi
echo "[OK] Using wheel: $(basename "$WHEEL")"

# ── Set up isolated uv environment ──────────────────────────────────────────

SANDBOX="$(mktemp -d)"
DOWNLOAD_DIR="$SANDBOX/download"
mkdir -p "$DOWNLOAD_DIR"
export UV_TOOL_DIR="$SANDBOX/tools"
export UV_TOOL_BIN_DIR="$SANDBOX/bin"
export PATH="$UV_TOOL_BIN_DIR:$PATH"

cleanup() {
    echo ""
    echo "=== Cleaning up ==="
    # Delete the draft release and its tag from GitHub.
    if gh release view "$TAG" --repo "$REPO" &>/dev/null; then
        # --cleanup-tag may warn if draft releases have no real tag; ignore.
        gh release delete "$TAG" --repo "$REPO" --cleanup-tag --yes 2>/dev/null || \
            gh release delete "$TAG" --repo "$REPO" --yes 2>/dev/null || true
        echo "[OK] Deleted draft release $TAG"
    fi
    rm -rf "$SANDBOX"
    echo "[OK] Removed sandbox $SANDBOX"
}
trap cleanup EXIT

echo ""
echo "Sandbox: $SANDBOX"

# ── Clean up any leftover test release ──────────────────────────────────────

if gh release view "$TAG" --repo "$REPO" &>/dev/null; then
    echo "[...] Deleting leftover test release $TAG"
    gh release delete "$TAG" --repo "$REPO" --cleanup-tag --yes 2>/dev/null || \
        gh release delete "$TAG" --repo "$REPO" --yes 2>/dev/null || true
fi

# ── Create draft release and upload wheel ───────────────────────────────────

echo ""
echo "=== Test 1: Create draft GitHub Release ==="
gh release create "$TAG" \
    --repo "$REPO" \
    --title "Installer Test (delete me)" \
    --notes "Automated test release, safe to delete." \
    --draft \
    --prerelease \
    "$WHEEL"
echo "[OK] Created draft release $TAG with $(basename "$WHEEL")"

# ── Verify the asset is listed ──────────────────────────────────────────────

echo ""
echo "=== Test 2: Verify release assets ==="
ASSETS="$(gh release view "$TAG" --repo "$REPO" --json assets --jq '.assets[].name')"
echo "Assets: $ASSETS"
if ! echo "$ASSETS" | grep -q "\.whl$"; then
    echo "FAIL: No .whl asset found in release"
    exit 1
fi
echo "[OK] Wheel asset present"

# ── Download the wheel from the release ─────────────────────────────────────

echo ""
echo "=== Test 3: Download wheel from GitHub Release ==="
gh release download "$TAG" \
    --repo "$REPO" \
    --dir "$DOWNLOAD_DIR" \
    --pattern "*.whl"

# Only one wheel is downloaded into this fresh dir; its name is tool-generated
# with no whitespace or special characters, so ls is safe here.
# shellcheck disable=SC2012
DOWNLOADED_WHEEL="$(ls "$DOWNLOAD_DIR"/*.whl | head -1)"
echo "[OK] Downloaded: $(basename "$DOWNLOADED_WHEEL")"

# ── Install from the downloaded wheel ───────────────────────────────────────

echo ""
echo "=== Test 4: Install from downloaded wheel ==="
uv tool install --python 3.12 "$DOWNLOADED_WHEEL"
echo "[OK] uv tool install succeeded"

# ── Verify CLI works ────────────────────────────────────────────────────────

echo ""
echo "=== Test 5: Verify batchalign3 ==="
batchalign3 --help >/dev/null
echo "[OK] batchalign3 --help exits 0"
batchalign3 version 2>&1 || true

# ── Test upgrade from release ───────────────────────────────────────────────

echo ""
echo "=== Test 6: Upgrade from downloaded wheel ==="
uv tool install --force --python 3.12 "$DOWNLOADED_WHEEL"
echo "[OK] Upgrade (--force) succeeded"

batchalign3 --help >/dev/null
echo "[OK] batchalign3 --help exits 0 after upgrade"

# ── Test installer resolution logic (offline, against the draft JSON) ────────

echo ""
# A draft release has no git tag, so the API releases/tags endpoint 404s.
# Synthesize a release JSON from the real built wheel name: this still checks
# that the installer's platform string matches maturin's actual wheel tag and
# that its grep extracts the abi3 wheel URL.
echo "=== Test 7: install-batchalign3.sh resolves the abi3 wheel URL ==="
WHEEL_NAME="$(basename "$DOWNLOADED_WHEEL")"
RELEASE_JSON="$SANDBOX/release.json"
printf '{"assets":[{"browser_download_url":"https://github.com/%s/releases/download/%s/%s"}]}\n' \
    "$REPO" "$TAG" "$WHEEL_NAME" > "$RELEASE_JSON"
RESOLVED="$(BATCHALIGN3_RELEASE_JSON_FILE="$RELEASE_JSON" BATCHALIGN3_RESOLVE_ONLY=1 \
    bash "$SCRIPT_DIR/install-batchalign3.sh")"
echo "Resolved URL: $RESOLVED"
case "$RESOLVED" in
    *-abi3-*.whl) echo "[OK] installer resolved an abi3 wheel for this platform" ;;
    *) echo "FAIL: installer did not resolve an abi3 wheel URL"; exit 1 ;;
esac

# ── Done ────────────────────────────────────────────────────────────────────

echo ""
echo "============================================"
echo "  All GitHub Release tests passed!"
echo "============================================"
