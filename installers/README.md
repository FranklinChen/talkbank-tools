# One-Click Installers

**Status:** Current
**Last updated:** 2026-04-02 07:26 EDT

Scripts for installing Batchalign3 without requiring users to open a terminal
or type commands manually.

## macOS

**File:** `macos/install-batchalign3.command`

1. Download the file.
2. Double-click it in Finder.
3. If macOS Gatekeeper blocks it:
   - Right-click the file > **Open** > **Open** in the dialog.
   - Or: System Settings > Privacy & Security > scroll down > **Open Anyway**.
4. A Terminal window will open showing installation progress.
5. When done, open a **new** Terminal window and run `batchalign3 --help`.

The script installs `uv` (if not present) and then installs `batchalign3`
via `uv tool install`. Re-running the script upgrades an existing installation.

## Windows

**File:** `windows/install-batchalign3.bat`

1. Download the file.
2. Double-click it in Explorer.
3. If Windows SmartScreen blocks it: click **More info** > **Run anyway**.
4. A Command Prompt window will open showing installation progress.
5. When done, open a **new** PowerShell or Command Prompt and run
   `batchalign3 --help`.

The script installs `uv` (if not present) via PowerShell and then installs
`batchalign3` via `uv tool install`. Re-running the script upgrades an
existing installation.

## Distribution tier assessment

| Tier | What | Status | Effort |
|------|------|--------|--------|
| 1 (current) | `.command` / `.bat` scripts | **Live** | Done |
| 1.5 | GitHub Releases with platform wheels | **Staged** (`release.yml` ready; publishing on hold) | Done |
| 2 | `.pkg` (macOS) + `.exe` (Windows) bundled installers | Not started | 4-6 days |
| 3 | Tauri desktop GUI with built-in updater | Dormant | 1-2 weeks |

**Tier 1** covers the immediate need: users who can't find Terminal/PowerShell
or follow `curl` instructions can download a file and double-click it.

**Tier 1.5** is wired in `release.yml` and tested locally via
`test-github-release.sh`, but it remains a staged release-preparation path for
now. The project is currently in **beta** (pre-release stabilization). Keep the
wheel-building and download/install flow working, but do **not** treat this as
authorization to publish new GitHub Releases or resume PyPI. Public release is
pending completion of release engineering work: signing/notarization expectations
in `../../docs/code-signing-and-distribution.md` must be wired for any direct
macOS app or CLI downloads, and the broader release gate must reopen.

**Tier 2** would provide native OS installers (macOS `.pkg` signed with Apple
Developer ID, Windows `.exe` via Inno Setup or WiX) that appear in
Applications/Programs and include uninstallers. This requires code signing
certificates and platform-specific build infrastructure.

**Tier 3** would wrap the CLI in a native desktop application with a graphical
interface. Infrastructure exists at `apps/dashboard-desktop/` (Tauri v2.8 +
React) with a CI workflow at `.github/workflows/dashboard-desktop.yml`, but
the surface is dormant and not functional for end users.

### Tauri desktop app: current status and prereqs

The [Tauri + React Dashboard ADR](../book/src/decisions/tauri-react-dashboard-adoption.md)
accepted Tauri as the long-term desktop path (2026-02-25). The ADR commits to
shipping a minimum viable desktop app before the 1.0 public release. What's
needed:

1. **File/folder picker** — Tauri native dialog feeding paths to the
   batchalign3 server
2. **Progress display** — SSE from server rendered in the React UI
3. **Auto-start server** — launch `batchalign3 serve start` on app open
4. **Auto-update** — Tauri's built-in updater (requires signing keys)
5. **Signed bundles** — notarized `.app` for macOS, signed `.exe` for Windows

PyPI release preparation follows the same staged-but-on-hold rule. Wheel builds
and installer tests should stay green, but this README documents release
engineering readiness, not permission to publish. The project remains in beta
until release engineering and stabilization work are complete.

## Testing

Both scripts support `BATCHALIGN_PACKAGE` (override package spec) and `CI=true`
(skip interactive prompts) environment variables for automated testing.

```bash
# Test the macOS installer (builds wheel, isolated sandbox, cleanup)
bash installers/test.sh

# Reuse an existing wheel in dist/
bash installers/test.sh --no-build

# Test the full GitHub Release flow (creates draft release, downloads, installs)
bash installers/test-github-release.sh
bash installers/test-github-release.sh --no-build
```

Both test scripts use `UV_TOOL_DIR`/`UV_TOOL_BIN_DIR` to install into an
isolated temp directory — they do not affect the developer's real tool
installations. The GitHub Release test creates a draft pre-release, verifies
the download + install path, then deletes the draft on cleanup.
