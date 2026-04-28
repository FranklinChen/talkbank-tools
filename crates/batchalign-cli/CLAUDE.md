# batchalign-cli — CLI Client, Dispatch, and Daemon Management

**Status:** Current
**Last modified:** 2026-04-12 06:57 EDT

## Overview

Clap-based CLI that never loads ML models locally. All processing is delegated to a
batchalign server via HTTP. The CLI handles file discovery, job submission, polling,
result retrieval, and local daemon lifecycle.

## Module Map

| Module | Purpose |
|--------|---------|
| `args.rs` | Clap argument structs (`Cli`, `GlobalOpts`, `Commands`, per-command args) |
| `dispatch.rs` | Two-tier dispatch: explicit `--server` → auto-daemon |
| `client.rs` | HTTP client (health, submit, poll, download) with retry/backoff |
| `resolve.rs` | Input path resolution (`--file-list`, `--in-place`, legacy 2-arg) |
| `discover.rs` | File discovery (walk dirs, filter extension, sort by size, skip dummy) |
| `error.rs` | Typed errors with stable exit codes (2–6) for scripting |
| `daemon.rs` | Daemon lifecycle (spawn, health-check, stale-binary restart, sidecar) |
| `serve_cmd.rs` | `batchalign3 serve` (start/stop/status) |
| `output.rs` | Write results to filesystem with path traversal protection |
| `progress.rs` | Terminal progress bars (indicatif) and `ProgressSink` boundary |
| `tui/` | ratatui TUI dashboard with reducer-owned `AppState`, pipeline phase dots, health metrics |

## Key Commands

```bash
cargo nextest run -p batchalign-cli
cargo clippy -p batchalign-cli -- -D warnings
```

## Dispatch Priority

1. **Explicit `--server URL`** → single-server (content mode: POST CHAT text)
2. **Local daemon** → auto-spawn `batchalign3 serve start`, paths mode (filesystem paths)

**Content mode** (remote): reads .cha files, POSTs text, polls, writes results.
**Paths mode** (daemon): sends only filesystem paths, daemon reads/writes directly.

## Daemon Profiles

- **Main** — default daemon, system Python
- **Sidecar** — Python 3.12 for transcribe (whisper needs numba), configured via
  `BATCHALIGN_SIDECAR_PYTHON` or `~/.batchalign3/sidecar/.venv/bin/python`

Stale-binary detection: `build_hash` in `daemon.json` compared to running binary.
Mismatch triggers auto-restart (useful during development).

## Exit Codes

| Code | Meaning |
|------|---------|
| 0 | Success |
| 2 | Usage error (bad args, missing paths) |
| 3 | Config error (bad YAML) |
| 4 | Network error (unreachable server) |
| 5 | Server error (unsupported command, HTTP error) |
| 6 | Local runtime error |

---
Last Updated: 2026-04-12 06:57 EDT
