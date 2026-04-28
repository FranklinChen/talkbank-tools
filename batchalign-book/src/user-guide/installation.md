# Installation

**Status:** Current
**Last updated:** 2026-04-28 16:59 EDT

Batchalign runs on **Windows, macOS, and Linux**. Pre-built wheels are available from PyPI for all three platforms (macOS ARM + Intel, Linux x86 + ARM, Windows x86).

## System requirements

| Requirement | Details |
|------------|---------|
| Python | 3.12 (installed automatically by `uv`) |
| Disk space | ~2 GB for ML models (downloaded on first use) |
| RAM | 8 GB minimum, 16 GB recommended |
| FFmpeg | Only needed for MP4 media files |
| Platforms | macOS ARM + Intel, Windows x86, Linux x86 + ARM |

## Easiest: Download and double-click

No terminal required. Download the installer for your platform and
double-click it:

- **macOS:** [Download install-batchalign3.command](https://github.com/TalkBank/talkbank-tools/raw/main/installers/macos/install-batchalign3.command)
  — if macOS blocks it, right-click > **Open** > **Open**
- **Windows:** [Download install-batchalign3.bat](https://github.com/TalkBank/talkbank-tools/raw/main/installers/windows/install-batchalign3.bat)
  — if SmartScreen blocks it, click **More info** > **Run anyway**

See the [installers README](https://github.com/TalkBank/talkbank-tools/blob/main/installers/README.md) for detailed Gatekeeper/SmartScreen instructions.

## Install with uv

If you don't have `uv` yet, install it first — then install Batchalign:

```bash
uv tool install batchalign3
```

At this stage there is intentionally **not** a separate
`uv tool install batchalign3-server` package. The default distribution remains
one install so BA2-style local/direct use stays simple while the unreleased
server/control-plane architecture is still evolving.

### macOS

1. Open **Terminal** (search "Terminal" in Spotlight, or find it in
   Applications > Utilities).

2. Install `uv`:
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

3. **Close and reopen Terminal** so the new command is available.

4. Install Batchalign:
   ```bash
   uv tool install batchalign3
   ```

5. Verify:
   ```bash
   batchalign3 --help
   ```

### Windows

1. Open **PowerShell** (search "PowerShell" in the Start menu).

2. Install `uv`:
   ```powershell
   irm https://astral.sh/uv/install.ps1 | iex
   ```

3. **Close and reopen PowerShell** so the new command is available.

4. Install Batchalign:
   ```powershell
   uv tool install batchalign3
   ```

5. Verify:
   ```powershell
   batchalign3 --help
   ```

### Linux

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc   # or restart your terminal
uv tool install batchalign3
batchalign3 --help
```

## First run

After installing, **restart your terminal** so the `batchalign3` command is
on your PATH. The first time you run a processing command (e.g. `morphotag`),
ML models will be downloaded automatically. This is a one-time cost of ~2 GB
and may take a few minutes depending on your connection. Subsequent runs use
cached models.

**Prefer a graphical interface?** See the [Desktop App](desktop-app.md) guide
for processing files without a terminal.

## Updating

Upgrade to the latest version:

```bash
uv tool upgrade batchalign3
```

If you installed via the one-click installer, re-running the same installer
script will upgrade an existing installation.

The CLI prints a notice when a newer version is available on PyPI. You can
suppress this by setting `BATCHALIGN_NO_UPDATE_CHECK=1`.

## Offline / alternative install

All install paths use PyPI by default. If you need to install without internet
access (e.g., air-gapped machines), you can install from a local wheel file:

```bash
uv tool install ./batchalign3-1.0.0-cp312-cp312-macosx_11_0_arm64.whl
```

Wheel files for all 5 supported platforms are built by the release CI. Once
GitHub Releases is set up (see [installers README](https://github.com/TalkBank/talkbank-tools/blob/main/installers/README.md)),
they will be downloadable from the releases page.

## Built-in engines

All built-in engines, including Cantonese/HK providers, are part of the base
package:

```bash
uv tool install batchalign3
```

## Worker Python resolution

The CLI finds a Python 3.12 runtime automatically — via
`BATCHALIGN_PYTHON`, the active virtualenv, a sibling/project `.venv`, or
`python3.12` on PATH. Override explicitly:

```bash
# macOS / Linux
export BATCHALIGN_PYTHON=/path/to/venv/bin/python

# Windows (PowerShell)
$env:BATCHALIGN_PYTHON = "C:\path\to\venv\Scripts\python.exe"
```

Under `uv tool install`, the visible `batchalign3` command is still a thin
Python launcher, but it immediately `exec`s the packaged Rust CLI binary. The
wrapper also preserves the chosen Python runtime for worker subprocesses, so
`batchalign3 serve ...` and background/daemon flows still run through the same
Rust CLI/server codepath as direct invocation of the packaged binary.

## Verify the installation

Confirm the CLI is available:

```bash
batchalign3 --help
```

Confirm the chosen Python runtime can import the worker package:

```bash
$BATCHALIGN_PYTHON -c "import batchalign.worker"
```

If you are relying on `VIRTUAL_ENV` or `python3` instead of
`BATCHALIGN_PYTHON`, run the same import check with that interpreter.

## Rev.AI setup

If you plan to use the default Rev.AI-backed transcription path, initialize
`~/.batchalign.ini`:

```bash
batchalign3 setup
```

See [Rev.AI Integration](rev-ai.md) for details.

## Development install

For contributors working from a source checkout:

```bash
git clone https://github.com/talkbank/talkbank-tools.git
git clone https://github.com/talkbank/batchalign3.git
cd batchalign3
make sync
make build
```

The repo-managed `.venv` includes the same built-in engine families as the base
package. `make sync` is sufficient for Cantonese/HK providers as well.

That full dev build also rebuilds the embedded dashboard, so it requires
Node.js + npm in addition to Rust and uv. If you only need the CLI and Python
extension surfaces, use `make build-python` and `make build-rust` instead.

In a source checkout, `uv run batchalign3` is still the normal way to invoke
the installed console script. After `make build-python`, the wrapper falls
back to the repo CLI when the embedded bridge is intentionally omitted, so the
fast extension-only rebuild still leaves you with a runnable `batchalign3`
command.

For the fastest contributor loop, pair the slim extension rebuild with a one-
time CLI build:

```bash
make build-python
cargo build -p batchalign-cli
uv run batchalign3 --help
```

Reserve `uv run` for Python tools such as `pytest`, `mypy`, and `maturin`
when you are not invoking the CLI.

Common rebuilds from a dev checkout:

```bash
cargo build -p batchalign-cli        # CLI / server changes
make build-rust                      # same as above via Makefile
make build-python                    # PyO3 or shared chat-ops changes
make build                           # full dev rebuild
./target/debug/batchalign3 --help
cargo run -p batchalign-cli -- --help
cargo nextest run --workspace
cargo nextest run --manifest-path pyo3/Cargo.toml
uv run pytest
uv run mypy
```

For the fuller contributor workflow and rebuild matrix, see
[Building & Development](../developer/building.md).
