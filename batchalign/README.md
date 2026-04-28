# Batchalign3

**Status:** Current
**Last updated:** 2026-04-28 16:59 EDT

[![CI](https://github.com/TalkBank/talkbank-tools/actions/workflows/batchalign-python.yml/badge.svg)](https://github.com/TalkBank/talkbank-tools/actions/workflows/batchalign-python.yml)
[![PyPI](https://img.shields.io/pypi/v/batchalign3)](https://pypi.org/project/batchalign3/)

Turn audio recordings into fully annotated [CHAT](https://talkbank.org/0info/manuals/CHAT.html)
transcripts — or enrich existing transcripts — from the command line.

- **Transcribe** — speech-to-text from audio (Whisper, Rev.AI)
- **Morphotag** — morphosyntactic analysis (%mor and %gra tiers)
- **Align** — forced alignment of words to audio timestamps
- **Translate** — add translation tiers (%xtra)
- **Segment** — utterance boundary detection
- **Benchmark** — WER scoring against gold transcripts

Part of the [TalkBank](https://talkbank.org/) project. Runs on macOS,
Windows, and Linux.

## Get Started

### Easiest: Download and double-click

No terminal required. Download the installer for your platform, double-click
it, and follow the on-screen prompts:

- **macOS:** [Download install-batchalign3.command](https://github.com/TalkBank/talkbank-tools/raw/main/installers/macos/install-batchalign3.command)
  — if macOS blocks it, right-click > **Open** > **Open** in the dialog
- **Windows:** [Download install-batchalign3.bat](https://github.com/TalkBank/talkbank-tools/raw/main/installers/windows/install-batchalign3.bat)
  — if SmartScreen blocks it, click **More info** > **Run anyway**

See [`installers/README.md`](installers/README.md) for details.

### From the terminal

Install the `uv` package manager, then install Batchalign:

**macOS / Linux:**

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Windows (PowerShell):**

```powershell
irm https://astral.sh/uv/install.ps1 | iex
```

Close and reopen your terminal, then:

```bash
uv tool install batchalign3
```

All built-in engines, including Cantonese/HK providers, ship in the base
package:

```bash
uv tool install batchalign3
```

### System requirements

- **Python:** 3.12 (installed automatically by `uv`)
- **Disk:** ~2 GB for ML models (downloaded on first use)
- **RAM:** 8 GB minimum, 16 GB recommended
- **FFmpeg:** only needed for MP4 media files
- **Platforms:** macOS (ARM + Intel), Windows (x86), Linux (x86 + ARM)

See [Installation guide](book/src/user-guide/installation.md) for offline
install, worker Python resolution, and development setup.

### First run

After installing, **restart your terminal** so the `batchalign3` command is on
your PATH. Then configure your default ASR engine:

```bash
batchalign3 setup
```

This creates `~/.batchalign.ini`. You can also configure non-interactively:

```bash
batchalign3 setup --non-interactive --engine whisper
batchalign3 setup --non-interactive --engine rev --rev-key <KEY>
```

The first time you run a processing command (e.g. `morphotag`), ML models will
be downloaded automatically — this is a one-time cost of ~2 GB and may take a
few minutes depending on your connection.

See [Quick start](book/src/user-guide/quick-start.md) for a full first-run
walkthrough.

### Updating

Upgrade to the latest version:

```bash
uv tool upgrade batchalign3
```

If you installed via the one-click installer, re-running the same installer
script will upgrade an existing installation.

The CLI will print a notice when a newer version is available on PyPI.

## Usage

The safest way to run any command is with a separate output directory, so
your originals are never touched:

```bash
# Morphosyntactic analysis (%mor and %gra tiers)
batchalign3 morphotag ~/corpus/ -o ~/output/

# Forced alignment (word-level timestamps)
batchalign3 align ~/corpus/ -o ~/output/

# ASR transcription
batchalign3 transcribe ~/recordings/ -o ~/transcripts/ --lang eng

# Translation (%xtra tier)
batchalign3 translate ~/corpus/ -o ~/output/

# Utterance segmentation
batchalign3 utseg ~/corpus/ -o ~/output/

# WER benchmarking
batchalign3 benchmark ~/corpus/
```

The `-o` flag is optional — two positional arguments are treated as
`input/ output/`:

```bash
batchalign3 morphotag ~/corpus/ ~/output/    # same as -o ~/output/
```

See [CLI reference](book/src/user-guide/cli-reference.md) for the full
command list and all flags.

### In-place processing

If your corpus is tracked in Git (or you have another backup), you can skip
the output directory and write results directly back into the source files.
A single argument with no `-o` is treated as in-place:

```bash
batchalign3 morphotag ~/corpus/
batchalign3 align ~/corpus/
batchalign3 translate ~/corpus/
```

The `--in-place` flag makes this explicit, and is required when passing
multiple input paths:

```bash
batchalign3 morphotag --in-place ~/corpus1/ ~/corpus2/
```

Each `.cha` file is overwritten with the annotated version. You can then
review the changes with `git diff` and commit when satisfied.

> **Warning:** In-place processing has no undo. If your files are not under
> version control, copy the folder first or use `-o` to write to a separate
> directory.

### Verbosity and logs

```bash
batchalign3 -v morphotag ~/corpus/ -o ~/output/    # verbose
batchalign3 -vv morphotag ~/corpus/ -o ~/output/   # debug
batchalign3 logs --last                             # most recent run
```

### Server mode

By default, a local server starts automatically and stays running so ML
models only load once. If you have a more powerful machine (e.g. one with a
GPU), you can run the server there and connect to it from your desktop or
laptop:

```bash
# On the server (e.g. a GPU workstation called myserver):
batchalign3 serve start --port 9000      # default port is 8000

# From any other machine on the network (use the same port):
batchalign3 --server http://myserver:9000 morphotag ~/corpus/ -o ~/output/
```

See [Server mode](book/src/user-guide/server-mode.md) for setup details and
the remote/local tradeoffs.

## Learn more

### For users

- [Installation guide](book/src/user-guide/installation.md) — system requirements, offline install, updating
- [Quick start](book/src/user-guide/quick-start.md) — first run walkthrough
- [CLI reference](book/src/user-guide/cli-reference.md) — all commands and flags
- [Server mode](book/src/user-guide/server-mode.md) — remote dispatch, daemon management
- [Performance tips](book/src/user-guide/performance.md) — large corpus processing
- [Migrating from Batchalign2](book/src/migration/index.md) — upgrade path
- [TalkBank CHAT manual](https://talkbank.org/0info/manuals/CHAT.html) — CHAT format reference

### For developers

- [Python API](book/src/user-guide/python-api.md) — CLI-first usage, removed legacy APIs
- [Building & Development](book/src/developer/building.md) — Rust toolchain, dev rebuilds, test matrix

## Development

Requires a Rust toolchain and [uv](https://docs.astral.sh/uv/).

```bash
make sync && make build
./target/debug/batchalign3 --help
```

## Support

- Bug reports and feature requests: <https://github.com/TalkBank/talkbank-tools/issues>
- General TalkBank questions: <https://talkbank.org/>

---

Supported by NIH grant HD082736.
