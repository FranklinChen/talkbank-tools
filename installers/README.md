# Batchalign install scripts

**Status:** Current
**Last updated:** 2026-06-21 19:53 EDT

The canonical way to install the `batchalign3` CLI is the release-hosted
installer script. It bootstraps [`uv`](https://docs.astral.sh/uv/) if needed,
detects the platform, installs the matching abi3 wheel from the latest GitHub
release into an isolated uv tool environment using a uv-managed Python (3.13 by
default), and re-running it upgrades in place.

```bash
# macOS / Linux
curl --proto '=https' --tlsv1.2 -LsSf https://github.com/FranklinChen/talkbank-tools/releases/latest/download/install-batchalign3.sh | sh
```

```powershell
# Windows (PowerShell)
irm https://github.com/FranklinChen/talkbank-tools/releases/latest/download/install-batchalign3.ps1 | iex
```

There is no PyPI package; distribution is via GitHub releases only. The
`batchalign3` PyPI name is reserved for the upstream fork. batchalign3's own
dependencies still resolve from PyPI, so the first install downloads large ML
dependencies.

## Files

| File | Purpose |
|---|---|
| `install-batchalign3.sh` | Canonical macOS/Linux installer (attached to every release). |
| `install-batchalign3.ps1` | Canonical Windows installer (attached to every release). |
| `macos/install-batchalign3.command` | Double-click wrapper that runs `install-batchalign3.sh`. |
| `windows/install-batchalign3.bat` | Double-click wrapper that runs `install-batchalign3.ps1`. |
| `test-github-release.sh` | End-to-end test of the GitHub-release install path. |

## Double-click helpers

For users who prefer not to use a terminal, the `.command` (macOS) and `.bat`
(Windows) files run the same canonical installer.

### macOS

1. Download `macos/install-batchalign3.command`.
2. Double-click it in Finder.
3. If macOS Gatekeeper blocks it (the file is not code-signed):
   - Right-click the file > **Open** > **Open** in the dialog, or
   - System Settings > Privacy & Security > scroll down > **Open Anyway**.
4. A Terminal window opens showing installation progress.
5. When done, open a **new** Terminal window and run `batchalign3 --help`.

### Windows

1. Download `windows/install-batchalign3.bat`.
2. Double-click it in Explorer.
3. If Windows SmartScreen blocks it: click **More info** > **Run anyway**.
4. A terminal window opens showing installation progress.
5. When done, open a **new** PowerShell or Command Prompt and run
   `batchalign3 --help`.

## Choosing the Python version

The installers use a uv-managed Python 3.13 by default. Set `BATCHALIGN3_PYTHON`
(for example `3.13`) before running to install against another supported
version (3.13 or 3.14). One abi3 wheel per platform serves all of them.

## Native installers

OS-level installer surfaces (`.pkg`, `.dmg`, `.exe`, `.msi`) are not provided;
they are blocked until signing/notarization automation exists. The release
assets are the per-platform wheels, the installer scripts, and a `sha256.sum`.

## Testing

`test-github-release.sh` proves the full path end-to-end: it builds the wheel,
creates a throwaway draft pre-release, runs the installer against it, verifies
the CLI works, and deletes the draft on cleanup. It installs into an isolated
temp directory (`UV_TOOL_DIR` / `UV_TOOL_BIN_DIR`), so it does not touch your
real tool installations.

```bash
bash installers/test-github-release.sh              # build wheel + full test
bash installers/test-github-release.sh --no-build   # reuse an existing wheel in dist/
```
