# talkbank-tools

**Status:** Current
**Last updated:** 2026-06-30 13:55 EDT

[![CI](https://github.com/FranklinChen/talkbank-tools/actions/workflows/ci.yml/badge.svg)](https://github.com/FranklinChen/talkbank-tools/actions/workflows/ci.yml)
[![Batchalign Python](https://github.com/FranklinChen/talkbank-tools/actions/workflows/batchalign-python.yml/badge.svg)](https://github.com/FranklinChen/talkbank-tools/actions/workflows/batchalign-python.yml)
[![License: BSD-3-Clause](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)](LICENSE)

This repository is the home of **Batchalign3**: the [TalkBank](https://talkbank.org/)
audio and ML pipeline that produces and enriches CHAT transcripts
(transcription, forced alignment, neural morphotagging, translation, and
utterance segmentation). The `batchalign3` command-line tool, its Python
package, the PyO3 bridge, the dashboard web UI, and the experimental desktop
shell all live here.

The CHAT format itself (grammar, spec, tree-sitter parser, data model,
validation, the `chatter` CLI, and the CLAN command reference) lives in the
separate [`chatter`](https://github.com/TalkBank/chatter) repository.
talkbank-tools **consumes** chatter's published crates; it is no longer a
CHAT-format toolchain.

## Install the `batchalign3` CLI

```bash
# macOS / Linux
curl --proto '=https' --tlsv1.2 -LsSf https://github.com/FranklinChen/talkbank-tools/releases/latest/download/install-batchalign3.sh | sh
```

```powershell
# Windows (PowerShell)
irm https://github.com/FranklinChen/talkbank-tools/releases/latest/download/install-batchalign3.ps1 | iex
```

The installer bootstraps [`uv`](https://docs.astral.sh/uv/) if it is not already
present, installs `batchalign3` into an isolated environment with a uv-managed
Python (3.13 by default; override with `BATCHALIGN3_PYTHON`), and re-running it
upgrades to the latest release. The first install downloads large ML
dependencies (PyTorch, Stanza, Whisper, and others).

Distribution is via GitHub releases only; there is no PyPI package. If you
prefer a download-and-double-click path, the macOS `.command` and Windows `.bat`
helpers under [`installers/`](installers/README.md) run the same installer.

See the [Installation guide](book/src/batchalign/user-guide/installation.md) and
[Quick start](book/src/batchalign/user-guide/quick-start.md) for first-run
details (including `batchalign3 setup`).

## Usage

```bash
batchalign3 transcribe recordings/ -o transcripts/ --lang eng
batchalign3 align corpus/ -o aligned/
batchalign3 morphotag corpus/ -o tagged/
```

Full command reference:
[CLI reference](book/src/batchalign/user-guide/cli-reference.md).

## Supported platforms and Python

| Requirement | Details |
|---|---|
| Operating systems | macOS (Apple Silicon + Intel), Linux (x86_64 + aarch64), Windows (x86_64) |
| Python | 3.13 or 3.14 (the installer uses a uv-managed Python, 3.13 by default) |
| Disk | Several GB for ML models, downloaded on first use |
| RAM | 8 GB minimum, 16 GB recommended |
| FFmpeg | Needed only for some media formats |

## Documentation

User, developer, and architecture docs live in the mdBook under
[`book/`](book/):

| Section | Entry point |
|---|---|
| Install hub and quickstart | [Install](book/src/install/index.md), [Quickstart](book/src/quickstart/index.md) |
| Batchalign3 user guide, architecture, developer guide | [Batchalign3 introduction](book/src/batchalign/introduction.md) |

Build the book locally:

```bash
make book
make book-serve   # serves http://localhost:3000
```

## Repository map

| Path | What lives there |
|---|---|
| `batchalign/` | Python package for the `batchalign3` CLI |
| `crates/batchalign` | Batchalign pipeline (ASR, FA, morphotag, jobs/runner, store, local server control plane, dashboard API) |
| `crates/batchalign-transform` | Batchalign-specific CHAT transforms layered over chatter's generic transform |
| `crates/batchalign-pyo3` | PyO3 bridge and wheel build surface |
| `crates/batchalign-types` | Shared types |
| `frontend/` | React dashboard web UI |
| `apps/dashboard-desktop/` | Tauri desktop shell (experimental) |
| `installers/` | GitHub-release install scripts and double-click wrappers |
| `book/` | The mdBook (user, developer, architecture docs) |
| `docs/` | Repo-level reference and release-contract notes |

The CHAT core crates (`talkbank-model`, `talkbank-parser`,
`talkbank-parser-re2c`, `talkbank-transform`) are consumed from the public
`chatter` repository via git-tag dependencies; see `Cargo.toml`.

## Building and developing

```bash
make help                   # overview of repo-native tasks
make check                  # workspace compile check
make test                   # Rust workspace tests + doctests
make verify                 # canonical pre-merge gate (compile + batchalign + book)
make batchalign-ci-rust     # batchalign Rust / PyO3 gate
make batchalign-ci-python   # batchalign wheel / pytest / typecheck gate
make book                   # build the mdBook
make ci-local               # fast local CI approximation
make ci-full                # stricter local CI approximation
```

For lower-level build helpers:

```bash
cargo run -q -p xtask -- help
```

## License

BSD-3-Clause. Copyright (c) 2026, Carnegie Mellon University. See
[LICENSE](LICENSE).
