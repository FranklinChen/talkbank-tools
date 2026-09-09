# Batchalign3

**Status:** Current
**Last updated:** 2026-06-21 19:53 EDT

[![CI](https://github.com/FranklinChen/talkbank-tools/actions/workflows/batchalign-python.yml/badge.svg)](https://github.com/FranklinChen/talkbank-tools/actions/workflows/batchalign-python.yml)

Turn audio recordings into fully annotated [CHAT](https://talkbank.org/0info/manuals/CHAT.html)
transcripts, or enrich existing transcripts, from the command line.

- **Transcribe**: speech-to-text from audio (Whisper, Rev.AI)
- **Morphotag**: morphosyntactic analysis (%mor and %gra tiers)
- **Align**: forced alignment of words to audio timestamps
- **Translate**: add translation tiers (%xtra)
- **Segment**: utterance boundary detection
- **Benchmark**: WER scoring against gold transcripts

Part of the [TalkBank](https://talkbank.org/) project. Runs on macOS,
Windows, and Linux.

## Get Started

Install the `batchalign3` CLI from the latest GitHub release. The installer
bootstraps [`uv`](https://docs.astral.sh/uv/) if needed, installs into an
isolated environment with a uv-managed Python (3.12 by default), and re-running
it upgrades to the latest release. There is no PyPI package.

**macOS / Linux:**

```bash
curl --proto '=https' --tlsv1.2 -LsSf https://github.com/FranklinChen/talkbank-tools/releases/latest/download/install-batchalign3.sh | sh
```

**Windows (PowerShell):**

```powershell
irm https://github.com/FranklinChen/talkbank-tools/releases/latest/download/install-batchalign3.ps1 | iex
```

Open a new terminal, then run `batchalign3 --help`. The repo-hosted
[`installers/`](../installers/README.md) directory also has macOS `.command` and
Windows `.bat` double-click wrappers that run the same installer.

### System requirements

- **Python:** 3.12, 3.13, or 3.14 (a uv-managed 3.12 is used by default)
- **Disk:** several GB for ML models (downloaded on first use)
- **RAM:** 8 GB minimum, 16 GB recommended
- **FFmpeg:** only needed for some media formats
- **Platforms:** macOS (Apple Silicon + Intel), Windows (x86_64), Linux (x86_64 + aarch64)

See the [Installation guide](../book/src/batchalign/user-guide/installation.md) for
double-click helpers, offline install, worker Python resolution, and development
setup.

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

The first time you run a processing command (for example `morphotag`), ML models
are downloaded automatically. This is a one-time cost of several GB and may take
a few minutes depending on your connection.

See [Quick start](../book/src/batchalign/user-guide/quick-start.md) for a full first-run
walkthrough.

### Updating

Re-run the installer one-liner to upgrade in place:

```bash
curl --proto '=https' --tlsv1.2 -LsSf https://github.com/FranklinChen/talkbank-tools/releases/latest/download/install-batchalign3.sh | sh
```

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

The `-o` flag is optional: two positional arguments are treated as
`input/ output/`:

```bash
batchalign3 morphotag ~/corpus/ ~/output/    # same as -o ~/output/
```

See [CLI reference](../book/src/batchalign/user-guide/cli-reference.md) for the full
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
models only load once. If you have a more powerful machine (for example one
with a GPU), you can run the server there and connect to it from your desktop
or laptop:

```bash
# On the server (for example a GPU workstation called myserver):
batchalign3 serve start --port 9000      # default port is 8000

# From any other machine on the network (use the same port):
batchalign3 --server http://myserver:9000 morphotag ~/corpus/ -o ~/output/
```

See [Server mode](../book/src/batchalign/user-guide/server-mode.md) for setup details and
the remote/local tradeoffs.

## Learn more

### For users

- [Installation guide](../book/src/batchalign/user-guide/installation.md): system requirements, offline install, updating
- [Quick start](../book/src/batchalign/user-guide/quick-start.md): first run walkthrough
- [CLI reference](../book/src/batchalign/user-guide/cli-reference.md): all commands and flags
- [Server mode](../book/src/batchalign/user-guide/server-mode.md): remote dispatch, daemon management
- [Performance tips](../book/src/batchalign/user-guide/performance.md): large corpus processing
- [Migrating from Batchalign2](../book/src/batchalign/migration/index.md): upgrade path
- [TalkBank CHAT manual](https://talkbank.org/0info/manuals/CHAT.html): CHAT format reference

### For developers

- [Python API](../book/src/batchalign/user-guide/python-api.md): CLI-first usage, removed legacy APIs
- [Building & Development](../book/src/batchalign/developer/building.md): Rust toolchain, dev rebuilds, test matrix

## Development

Requires a Rust toolchain and [uv](https://docs.astral.sh/uv/).

```bash
make batchalign-python-prepare && make build
./target/debug/batchalign3 --help
```

## Support

- Bug reports and feature requests: <https://github.com/FranklinChen/talkbank-tools/issues>
- General TalkBank questions: <https://talkbank.org/>

---

Supported by NIH grant HD082736.
