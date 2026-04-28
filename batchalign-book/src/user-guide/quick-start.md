# Quick Start

**Last modified:** 2026-03-27 11:18 EDT

This chapter covers the most common `batchalign3` workflows from the terminal.
The examples assume the `batchalign3` binary is installed and that local
processing commands can reach a Python runtime with `batchalign.worker`
available.

**Prefer a graphical interface?** See the [Desktop App](desktop-app.md) guide
instead — same commands, no terminal required.

For the full command surface, see the [CLI Reference](cli-reference.md).

## Before you start

**Model downloads:** The first time you run a processing command, Batchalign
downloads ML models (~2 GB). This is a one-time cost — subsequent runs use
cached models from disk.

**Caching:** Batchalign caches analysis results in a local SQLite database.
Re-processing the same file with the same command returns cached results
instantly. See [Caching](caching.md) for details.

**Performance:** Back-to-back runs are still much faster than first-run model
downloads because models and caches stay on disk. If you need hot in-memory
workers across repeated runs, start an explicit server with `batchalign3 serve`.
See [Performance](performance.md) for tuning tips.

## Basic command shape

```bash
batchalign3 [GLOBAL OPTIONS] COMMAND [COMMAND OPTIONS] [PATHS...]
```

- Global options go before the command.
- Most processing commands use `-o/--output` for a destination directory.
- Omitting `-o/--output` means in-place processing when the command supports it.

## Transcribe audio to CHAT

```bash
batchalign3 transcribe ~/recordings/ -o ~/transcripts/ --lang eng
```

To use OpenAI Whisper instead of the default Rev.AI engine:

```bash
batchalign3 transcribe ~/recordings/ -o ~/transcripts/ \
  --asr-engine whisper-oai --lang eng
```

To use a local Whisper model:

```bash
batchalign3 transcribe ~/recordings/ -o ~/transcripts/ \
  --asr-engine whisper --lang eng
```

Important routing note: explicit `--server` now submits shared-filesystem
`paths_mode` jobs for `transcribe`. The target server must be able to read the
same input paths and write the requested output paths.

## Align transcripts against audio

```bash
batchalign3 align ~/corpus/ -o ~/aligned/
```

Common useful flags:

```bash
batchalign3 align ~/corpus/ -o ~/aligned/ --wor
batchalign3 align ~/corpus/ -o ~/aligned/ --fa-engine whisper
batchalign3 align ~/corpus/ -o ~/aligned/ --utr-engine whisper
```

## Add morphosyntactic analysis

```bash
batchalign3 morphotag ~/corpus/ -o ~/tagged/
```

Useful variants:

```bash
batchalign3 morphotag ~/corpus/ -o ~/tagged/ --retokenize
batchalign3 morphotag ~/corpus/ -o ~/tagged/ --skipmultilang
```

Repeated runs are usually faster because Batchalign reuses its cache. For
interactive sessions where you want workers to stay warm in memory across
commands, use explicit server mode (`batchalign3 serve start` plus `--server`).

## Verbosity

```bash
batchalign3 align ~/corpus/ -o ~/aligned/
batchalign3 -v align ~/corpus/ -o ~/aligned/
batchalign3 -vv align ~/corpus/ -o ~/aligned/
batchalign3 -vvv align ~/corpus/ -o ~/aligned/
```

## Run logs

```bash
batchalign3 logs
batchalign3 logs --last
batchalign3 logs --export
batchalign3 logs --clear
```

## Remote server mode

For commands that support explicit remote dispatch:

```bash
batchalign3 --server http://yourserver:8000 morphotag ~/corpus/ -o ~/tagged/
batchalign3 --server http://yourserver:8000 align ~/corpus/ -o ~/aligned/
```

`transcribe` and `avqi` stay on the local execution path even when `--server`
is provided.

## Next steps

- [Desktop App](desktop-app.md) — process files without a terminal
- [CLI Reference](cli-reference.md)
- [Performance](performance.md)
- [Server Mode](server-mode.md)
- [Rev.AI Integration](rev-ai.md)
- [Python API](python-api.md)
