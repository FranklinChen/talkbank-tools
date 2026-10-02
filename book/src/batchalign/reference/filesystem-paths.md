# Filesystem Paths Used by batchalign3

**Status:** Current
**Last updated:** 2026-10-01 18:40 EDT

All current filesystem paths used by the public `batchalign3` runtime.

Unless otherwise noted, paths are rooted under
`batchalign::types::config::layout::ba_state_dir()` (defined at
`crates/batchalign/src/types/config/layout.rs:98`), which defaults
to `~/.batchalign3` and can be overridden with `BATCHALIGN_STATE_DIR`.

## File modes

The state directory is per user. The state files batchalign3 writes there
are owner-only: the daemon state files, the server handshakes and the worker
registry (`workers.json`, from the Rust server and from the Python worker
daemons alike) are written `0600`, as are debug artifacts (which hold
transcript content). Not every file in the directory is: the lock files
(empty) and the logs (`server.log`, `daemon.log`, `sidecar-daemon.log`) are
created with the umask's default mode. Every reader of them is the same
user's CLI, server or worker daemons; no documented reader runs under another
account, so a setup that shares a state directory between accounts is not
supported. Reports a command is asked to write elsewhere (`compare-runs`
output, `eval utr-alignment` evidence) get the mode `std::fs::write` would
give them, `0666` less the umask (`0644` under the usual `022`). The Rust
writers go through `atomic_file::write_atomically`, whose `Audience` argument
(`Owner` or `UmaskDefault`) states which.

Every read-modify-write of a state file, and every single-writer slot, holds
an exclusive OS lock on a dedicated `<file>.lock` beside it, through one
primitive (`crate::file_lock::HeldFileLock`), never a lock on the state file
itself, which each write replaces by rename. Taking a lock creates the
directory if it is missing, and dropping it unlocks explicitly, so a
release is immediate even while a forked child still shares the descriptor.

## Configuration

| Path | Purpose | Defined in |
|------|---------|------------|
| `~/.batchalign.ini` | User config shared with older tooling (for example Rev.AI credentials and default ASR selection) | `batchalign/config.py`, `crates/batchalign/src/cli/setup_cmd.rs` |
| `~/.batchalign3/server.yaml` | Server/daemon configuration | `crates/batchalign/src/types/config/` (directory module: `layout.rs`, `server.rs`, `resolve.rs`, `tests.rs`) |

## Runtime data

| Path | Purpose | Defined in |
|------|---------|------------|
| `~/.batchalign3/logs/` | Structured CLI run logs (`run-*.jsonl`) and exported log zips | `crates/batchalign/src/cli/logs_cmd.rs` |
| `~/.batchalign3/server.pid` | Handshake for the main server: the PID and the port it actually bound, written by the server after its bind succeeds | `crates/batchalign/src/server_handshake.rs` |
| `~/.batchalign3/sidecar-server.pid` | The same, for the transcribe sidecar daemon | `crates/batchalign/src/server_handshake.rs` |
| `~/.batchalign3/server.pid.lock`, `sidecar-server.pid.lock` | Lock files serializing each handshake's publish and removal; never deleted | `crates/batchalign/src/server_handshake.rs` |
| `~/.batchalign3/workers.json.lock` | Lock file serializing every read-modify-write of the worker registry, taken by the Rust server and the Python worker daemons alike | `crates/batchalign/src/worker/registry.rs`, `batchalign/worker/_registry.py` |
| `~/.batchalign3/server.log` | stderr log for manual `batchalign3 serve start` | `crates/batchalign/src/cli/serve_cmd.rs` |
| `~/.batchalign3/daemon.json` | main auto-daemon state | `crates/batchalign/src/cli/daemon.rs` |
| `~/.batchalign3/daemon.lock` | main auto-daemon startup lock | `crates/batchalign/src/cli/daemon.rs` |
| `~/.batchalign3/daemon.log` | main auto-daemon stderr log | `crates/batchalign/src/cli/daemon.rs` |
| `~/.batchalign3/sidecar-daemon.json` | sidecar-daemon state for transcribe-heavy workloads | `crates/batchalign/src/cli/daemon.rs` |
| `~/.batchalign3/sidecar-daemon.lock` | sidecar-daemon startup lock | `crates/batchalign/src/cli/daemon.rs` |
| `~/.batchalign3/sidecar-daemon.log` | sidecar-daemon stderr log | `crates/batchalign/src/cli/daemon.rs` |
| `~/.batchalign3/jobs/` | per-job staging directories | `crates/batchalign/src/lib.rs`, `crates/batchalign/src/routes/jobs/mod.rs` |
| `~/.batchalign3/jobs.db` | SQLite job persistence database | `crates/batchalign/src/db/mod.rs` |

## Caches

### Analysis cache (SQLite)

Utterance-level cache entries for morphosyntax, utterance segmentation,
translation, and forced alignment.

The default path comes from `dirs::cache_dir()` in Rust and intentionally
matches the Python-side `platformdirs` location.

| Platform example | Path |
|------------------|------|
| macOS | `~/Library/Caches/batchalign3/cache.db` |
| Linux | `~/.cache/batchalign3/cache.db` |

Override with `BATCHALIGN_ANALYSIS_CACHE_DIR`, which relocates the database to
`$BATCHALIGN_ANALYSIS_CACHE_DIR/cache.db`.

Defined in:

- `crates/batchalign/src/cache/sqlite.rs`
- `crates/batchalign/src/cli/cache_cmd.rs`

### Media cache

Directory used for cached media artifacts and exposed through
`batchalign3 cache stats|clear`.

The default path comes from `dirs::data_dir()`.

| Platform example | Path |
|------------------|------|
| macOS | `~/Library/Application Support/batchalign3/media_cache/` |
| Linux | `~/.local/share/batchalign3/media_cache/` |

Override with `BATCHALIGN_MEDIA_CACHE_DIR`.

Defined in:

- `crates/batchalign/src/ensure_wav.rs`
- `crates/batchalign/src/cli/cache_cmd.rs`

## Legacy compatibility note

BA2-era tooling used:

- `~/.batchalign.ini`
- `~/.batchalign/`
- `~/.cache/batchalign/`
- `~/Library/Application Support/batchalign/media_cache/`

The current release intentionally still shares `~/.batchalign.ini`, but the
state directory, jobs DB, logs, daemon state, and caches otherwise use the
`batchalign3` prefix.
