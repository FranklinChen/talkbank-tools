# Installation

**Status:** Current
**Last updated:** 2026-08-31 22:01 EDT

`batchalign3` is distributed via **GitHub releases** (there is no PyPI package).
The installer bootstraps [`uv`](https://docs.astral.sh/uv/) if needed, installs
`batchalign3` into an isolated environment using a uv-managed Python (3.13 by
default), and re-running it upgrades to the latest release.

```bash
# macOS / Linux
curl --proto '=https' --tlsv1.2 -LsSf https://github.com/FranklinChen/talkbank-tools/releases/latest/download/install-batchalign3.sh | sh
```

```powershell
# Windows (PowerShell)
irm https://github.com/FranklinChen/talkbank-tools/releases/latest/download/install-batchalign3.ps1 | iex
```

After installing, **open a new terminal** so the `batchalign3` command is on
your PATH, then:

```bash
batchalign3 --help
```

Pre-built wheels are published for all five platforms (macOS Apple Silicon +
Intel, Linux x86_64 + aarch64, Windows x86_64). One abi3 wheel per platform
covers Python 3.13 and newer. `batchalign3`'s own dependencies still resolve
from PyPI, so the first install downloads large ML dependencies.

## System requirements

| Requirement | Details |
|------------|---------|
| Python | 3.13 or 3.14 (a uv-managed 3.13 is used by default) |
| Disk space | Several GB for ML models (downloaded on first use) |
| RAM | 8 GB minimum, 16 GB recommended |
| FFmpeg | Only needed for some media formats |
| Platforms | macOS Apple Silicon + Intel, Linux x86_64 + aarch64, Windows x86_64 |

## Choosing the Python version

The installer uses a uv-managed Python 3.13 by default. To install against a
different supported version, set `BATCHALIGN3_PYTHON` before running it:

```bash
BATCHALIGN3_PYTHON=3.13 curl --proto '=https' --tlsv1.2 -LsSf \
  https://github.com/FranklinChen/talkbank-tools/releases/latest/download/install-batchalign3.sh | sh
```

## Double-click helpers

If you prefer not to use a terminal, the repo ships double-click wrappers that
run the same installer:

- **macOS:** [install-batchalign3.command](https://github.com/FranklinChen/talkbank-tools/raw/main/installers/macos/install-batchalign3.command)
- **Windows:** [install-batchalign3.bat](https://github.com/FranklinChen/talkbank-tools/raw/main/installers/windows/install-batchalign3.bat)

The downloaded helpers are not code-signed, so macOS Gatekeeper / Windows
SmartScreen may warn on first run; see the
[installers README](https://github.com/FranklinChen/talkbank-tools/blob/main/installers/README.md)
for the click-through. They install `uv` if needed and then run the canonical
installer.

## Updating

Re-run the installer one-liner; it reinstalls the latest release in place:

```bash
curl --proto '=https' --tlsv1.2 -LsSf https://github.com/FranklinChen/talkbank-tools/releases/latest/download/install-batchalign3.sh | sh
```

## Offline / manual install from a wheel

Every release attaches per-platform wheels plus a `sha256.sum`. To install
without the script (for example on an air-gapped machine), download the wheel
for your platform from the
[latest release](https://github.com/FranklinChen/talkbank-tools/releases/latest)
and install it with `uv`:

```bash
uv tool install --python 3.13 ./batchalign3-0.4.1-cp313-abi3-macosx_11_0_arm64.whl
```

## First run

The first time you run a processing command (for example `morphotag`), ML
models are downloaded automatically. This is a one-time cost of several GB and
may take a few minutes depending on your connection; subsequent runs use cached
models.

**Evaluating the experimental GUI shell?** See
[Batchalign Desktop (Experimental)](desktop-app.md). The supported first-time
user path is the `batchalign3` CLI above.

## Worker Python resolution

The CLI finds a Python 3.13 runtime automatically, via `BATCHALIGN_PYTHON`, the
active virtualenv, a sibling/project `.venv`, or `python3.13` on PATH. Override
explicitly:

```bash
# macOS / Linux
export BATCHALIGN_PYTHON=/path/to/venv/bin/python

# Windows (PowerShell)
$env:BATCHALIGN_PYTHON = "C:\path\to\venv\Scripts\python.exe"
```

The visible `batchalign3` command is a thin Python launcher that immediately
`exec`s the packaged Rust CLI binary. The launcher also preserves the chosen
Python runtime for worker subprocesses, so `batchalign3 serve ...` and
background/daemon flows run through the same Rust CLI/server codepath as direct
invocation of the packaged binary.

## Verify the installation

```bash
batchalign3 --help
```

Confirm the chosen Python runtime can import the worker package:

```bash
$BATCHALIGN_PYTHON -c "import batchalign.worker"
```

If you are relying on `VIRTUAL_ENV` or `python3` instead of `BATCHALIGN_PYTHON`,
run the same import check with that interpreter.

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
git clone https://github.com/FranklinChen/talkbank-tools.git
cd talkbank-tools
make batchalign-python-prepare    # build wheel + sync uv env + install
make build                         # cargo build --workspace --release
```

`make batchalign-python-prepare` rebuilds the wheel via the maturin backend
declared in `pyproject.toml`, runs `uv sync --group dev --no-install-project`,
and installs the freshly built wheel into the dev environment.

`make build` runs `cargo build --workspace --release`. It does not rebuild the
embedded dashboard; if you also need the React dashboard rebuilt, run
`make batchalign-dashboard-build` (which requires Node.js + npm in addition to
Rust and uv).

In a source checkout, `uv run batchalign3` is the normal way to invoke the
console script; the maturin backend's `profile = "dev"` setting means each
`uv run ...` triggers an incremental rebuild of the PyO3 extension on demand.
Reserve `uv run` for Python tools such as `pytest`, `mypy`, and `maturin` when
you are not invoking the CLI itself.

For the fastest contributor loop:

```bash
uv run batchalign3 --help        # incremental PyO3 rebuild via maturin/uv
cargo build -p batchalign         # native batchalign3 binary (debug)
./target/debug/batchalign3 --help
```

For the fuller contributor workflow and rebuild matrix, see
[Building & Development](../developer/building.md).
