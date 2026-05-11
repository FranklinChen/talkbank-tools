# Quick Start

**Status:** Current
**Last updated:** 2026-05-11 17:25 EDT

This page gets you from zero to productive with `chatter` in five minutes.
[Install chatter first](installation.md) if you haven't already.

## Validate a CHAT file

Check a single transcript for errors:

```bash
chatter validate transcript.cha
```

If the file is valid:

```text
✓ transcript.cha is valid
```

If there are problems, you'll see rich diagnostics with the exact location
and a stable error code. For example, an utterance missing its terminator:

```text
  × error[E304]: Expected terminator not found (line 6, column 1)

 6 │ *CHI:	hello world

  help: Add a terminator at the end: Standard (. ? !), Interruption (+... +/.)
```

Every error code (`E304`, `E705`, etc.) links to
[documentation with fix guidance](validation-errors.md).

## Validate an entire corpus

Point `chatter` at a directory — it walks recursively, validates in parallel,
and caches results:

```bash
chatter validate corpus/
```

The interactive TUI shows progress and lets you browse errors per file.
Use `--format json` for machine-readable output, or `--quiet` for CI
(exit code 1 on errors).

## Run an analysis

`chatter` reimplements the CLAN command surface across validation,
analysis, transform, and format-converter categories. A handful of
legacy NLP commands (the trie-and-rule MOR/POST/MEGRASP family) are
deliberately not implemented — they emit a clear error pointing at
Batchalign for the neural replacement. See the
[CLAN command status matrix](../../clan-reference/appendices/status-matrix.md)
for the authoritative per-command counts and per-command status.

Try frequency analysis:

```bash
chatter clan freq transcript.cha
```

Output shows word frequencies by speaker. Add filters:

```bash
chatter clan freq transcript.cha --speaker CHI    # one speaker
chatter clan mlu transcript.cha --speaker CHI     # mean length of utterance
chatter clan combo transcript.cha --include-word "want"  # boolean keyword search
```

All CLAN commands support `--format json` and `--format csv` for
downstream processing.

## Convert to JSON

Get a structured representation of any CHAT file:

```bash
chatter to-json transcript.cha
```

The output conforms to the [TalkBank CHAT JSON Schema](https://talkbank.org/schemas/v0.1/chat-file.json).
Convert back with `chatter from-json`.

## Watch for changes

Edit a file and get live validation feedback:

```bash
chatter watch transcript.cha
```

Every time you save, `chatter` re-validates and shows updated diagnostics.

## What next?

- **[CLI Reference](cli-reference.md)** — all commands, flags, and output formats
- **[Validation Errors](validation-errors.md)** — every error code, with examples and fix guidance
- **[VS Code Extension](vscode-extension.md)** — live diagnostics, CLAN commands, media playback
- **[Migrating from CLAN](migrating-from-clan.md)** — flag mapping for CLAN veterans
- **[Batch Workflows](batch-workflows.md)** — corpus-scale validation and analysis
