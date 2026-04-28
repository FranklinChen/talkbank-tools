# Async Runtime Architecture

**Status:** Current
**Last updated:** 2026-03-21 15:30

This document describes the async runtime model underlying the batchalign3
server and CLI. It covers the tokio runtime configuration, thread pool
boundaries, concurrency primitives, and the policies that govern which
work runs where.

Other documents describe *what* flows through the system
([Command Lifecycles](command-lifecycles.md),
[Dispatch System](dispatch-system.md),
[Server Architecture](server-architecture.md)). This document describes
*how* the runtime executes that work.

## Runtime Configuration

The server uses tokio's default multi-threaded runtime (work-stealing
scheduler, one worker thread per CPU core). No explicit thread count or
runtime builder configuration exists — tokio defaults are appropriate for
this workload because:

1. The heavy compute (ML inference) runs in Python subprocesses, not on
   tokio worker threads.
2. The Rust server's own work is I/O-bound: HTTP handling, IPC with
   workers, SQLite queries, file reads/writes.
3. The Mac Studio deployment target (M3 Ultra, 24 cores) benefits from
   work-stealing across many lightweight async tasks.

The CLI creates a one-off `tokio::runtime::Runtime::new()` for commands
that need async (e.g., `cache` stats via sqlx). Most CLI paths enter
async through `#[tokio::main]`.

## Thread Pool Boundaries

Three distinct thread contexts exist at runtime:

```text
┌─────────────────────────────────────────────────────────┐
│ Tokio worker threads (async executor)                   │
│                                                         │
│  HTTP handlers, SSE streams, WebSocket, IPC protocol,   │
│  sqlx queries, tokio::fs, job orchestration,            │
│  JoinSet tasks, Semaphore waits                         │
│                                                         │
│  Rule: nothing here may block for more than ~1 ms       │
├─────────────────────────────────────────────────────────┤
│ Tokio blocking thread pool (spawn_blocking)             │
│                                                         │
│  ffmpeg subprocess + fs2 file locking (ensure_wav),     │
│  ratatui TUI rendering, daemon subprocess spawn         │
│                                                         │
│  Rule: use for work that blocks a thread (subprocess    │
│  wait, file lock, synchronous terminal I/O)             │
├─────────────────────────────────────────────────────────┤
│ Python worker subprocesses (external, managed by pool)  │
│                                                         │
│  ML model loading, GPU inference, Stanza/Whisper/etc.   │
│                                                         │
│  Rule: all ML compute lives here, never on Rust threads │
└─────────────────────────────────────────────────────────┘
```

### What runs where

| Context | Examples | Key constraint |
|---------|----------|----------------|
| Async executor | HTTP routes, SSE, WebSocket, sqlx, `tokio::fs`, IPC JSON-lines, JoinSet file tasks | Must not block |
| `spawn_blocking` | ffmpeg conversion, `fs2` file locking, TUI rendering, `tailscale` CLI probe | Blocks a dedicated thread; returns `JoinHandle` |
| Python subprocess | Stanza NLP, Whisper ASR, Wave2Vec FA, speaker diarization | Managed by `WorkerPool`; communicates over stdio |

### Locked subprocess boundary

These Python worker processes are not temporary scaffolding on the way to
"all Rust." They are the intended steady-state runtime boundary for Python-only
ML libraries and SDKs.

That means future de-Pythonization work should remove Python-side orchestration
*around* the workers, not collapse model loading and inference back into tokio
threads. The Python subprocess context should stay limited to:

- direct model/SDK invocation;
- task-local model bootstrap and worker-local device selection;
- thin model-host request execution after Rust-owned stdio dispatch.

Config ownership, cache policy, prepared-audio/text construction, result
normalization, CHAT mutation, and other provider-independent workflow logic stay
on the Rust side. Already-landed BA2 compatibility shims are migration surfaces
above this runtime and are not part of the subprocess decision.

### When to use `spawn_blocking`

Use `spawn_blocking` when code does any of the following:

- **Waits for a subprocess** — `std::process::Command::output()` blocks
  until the child exits. Short-lived probes (e.g., `ffmpeg -version`)
  are borderline but should still use `spawn_blocking` if called from
  async context.
- **Acquires a file lock** — `fs2::FileExt::lock_exclusive()` can block
  indefinitely if another process holds the lock.
- **Performs CPU-intensive work** exceeding ~1 ms — fingerprint hashing
  of large buffers, compression, etc.
- **Calls synchronous terminal I/O** — ratatui draw calls block on
  terminal write.

Do **not** use `spawn_blocking` for:

- File reads/writes — use `tokio::fs` instead (see below).
- SQLite — `sqlx` is natively async with its own connection pool.
- Short `std::fs` calls in sync (non-async) functions — they're not on
  the executor, so there's nothing to block.

## File I/O Policy

**Use `tokio::fs` for all file operations in async contexts.** This
applies to `read_to_string`, `write`, `create_dir_all`, `rename`,
`remove_file`, `remove_dir_all`, and `metadata`.

`tokio::fs` delegates to `spawn_blocking` internally, so it moves file
I/O off the executor without requiring manual `spawn_blocking` at every
call site. This is the right default for an I/O-multiplexing server.

**Use `std::fs` only in these contexts:**

1. **Inside an existing `spawn_blocking` closure** — the thread is
   already dedicated, so `tokio::fs` would add needless indirection.
   Example: `ensure_wav` does fingerprinting, locking, and ffmpeg
   conversion in one `spawn_blocking` block using `std::fs` throughout.
2. **In synchronous functions** that are never called from async code —
   `clear_cache()`, `cache_stats()`, `media_fingerprint()`,
   `ffmpeg_available()`, `default_cache_dir()`.
3. **In test setup/teardown** — `#[test]` functions (not `#[tokio::test]`)
   are synchronous.
4. **One-time startup** — database directory creation in `db/mod.rs`,
   worker venv probing in `worker/python.rs`. These run once before the
   server accepts requests; blocking briefly is harmless.

Current state: all async code paths use `tokio::fs` consistently. HTTP
route handlers, dispatch code, and runner utility functions
(`preflight_validate_media`, `resolve_audio_for_chat`,
`compute_audio_identity`) all use `tokio::fs` for metadata, existence
checks, reads, and writes.

## Concurrency Primitives

### JoinSet — structured concurrent file processing

`tokio::task::JoinSet` is the primary tool for concurrent per-file
dispatch. Each dispatch shape spawns file tasks into a `JoinSet` and
collects results:

| Location | What's spawned | Bounded by |
|----------|----------------|------------|
| `dispatch_fa_infer()` | One task per CHAT file (FA) | `Semaphore(num_workers)` |
| `dispatch_transcribe_infer()` | One task per audio file | `Semaphore(num_workers)` |
| `dispatch_per_file_process()` | One task per file (opensmile, avqi) | `Semaphore(num_workers)` |
| `lib.rs` app-level | One task per running job | `max_concurrent_jobs` config |
| `queue.rs` | Queue dispatcher background task | Single task |

Pattern:
```rust
let semaphore = Arc::new(Semaphore::new(num_workers.max(1)));
let mut join_set = JoinSet::new();

for file in files {
    let permit = semaphore.clone().acquire_owned().await?;
    join_set.spawn(async move {
        let _permit = permit; // held until task completes
        process_file(file).await
    });
}

while let Some(result) = join_set.join_next().await {
    // handle result or error
}
```

The `Semaphore` permit is acquired *before* spawning and held by the
task via `acquire_owned()`. This ensures at most N files are in-flight
simultaneously, matching the number of available Python workers.

### RuntimeSupervisor — owned task lifecycle

The server no longer exposes queue-dispatch and per-job task ownership through
shared `AppState` locks. `runtime_supervisor.rs` owns:

- the queue dispatcher task
- the tracked per-job `JoinSet`
- shutdown waiting with an explicit timeout

This keeps task lifecycle control in one actor and leaves `AppState` with a
cloneable handle instead of raw task collections.

### Semaphore — bounding parallelism

Three semaphore instances exist:

| Instance | Permits | Guards |
|----------|---------|--------|
| Per-file dispatch (FA) | `num_workers` | Concurrent file tasks within a job |
| Per-file dispatch (transcribe) | `num_workers` | Same pattern, transcribe shape |
| `WorkerGroup.available` | 1 per idle worker | Async wait for worker availability |

The worker pool semaphore is unusual: permits are managed manually
(`add_permits(1)` on worker return, `permit.forget()` on checkout)
rather than via RAII guards. This is because the checkout/return
lifecycle spans multiple async operations and the permit count must
track the actual idle worker count precisely.

### Channels

| Type | Instance | Capacity | Purpose |
|------|----------|----------|---------|
| `broadcast` | `AppState.control.ws_tx` | 256 | Job progress → WebSocket/SSE clients |
| `mpsc::unbounded` | Per-file progress forwarder | Unbounded | Sub-file progress → job store |
| `mpsc::unbounded` | `JobRegistry` actor mailbox | Unbounded | JobStore/query/runner inspect+mutate requests |
| `mpsc::unbounded` | CLI TUI reducer queue | Unbounded | Polling progress sink → blocking TUI runtime |
| `oneshot` | `JobRegistry` per-call reply | 1 | Registry actor → caller result handoff |
| `oneshot` | TUI cancel signal | 1 | User keystroke → abort polling loop |

The broadcast channel uses `RecvError::Lagged(n)` handling — if a slow
WebSocket client falls behind, it skips missed events rather than
blocking the sender.

### Ownership boundary rule

Cross-request or cross-task coordination state should live behind an owned task
or actor boundary, not behind a route-visible mutex. The rule of thumb is:

1. If multiple routes, runner tasks, or background loops coordinate through the
   same mutable collection, give that state one owner and send commands to it.
2. If the state is only a tiny cell internal to one owner, a local mutex is an
   implementation detail and is acceptable.

`JobRegistry` is the main completed example of this rule. The in-memory jobs map
now lives behind one actor task (`mpsc::unbounded` + `oneshot` replies), so
callers describe transitions and projections instead of locking a shared
`HashMap`.

The deliberate bulk escape hatches are `JobRegistry::inspect_all()` and
`JobRegistry::mutate_all()`. Keep those reserved for crash recovery and other
rare whole-registry reconciliation passes; new request/runner code should
prefer named store or per-job registry methods.

### Mutex Policy

**Avoid Mutex wherever possible.** Use lock-free alternatives first:

| Need | Use | Not |
|------|-----|-----|
| Atomic swap of optional value | `ArcSwapOption` | `Mutex<Option<T>>` |
| Concurrent map | `DashMap` | `Mutex<HashMap>` |
| Lock-free counter | `AtomicUsize` / `AtomicBool` | `Mutex<usize>` |
| Lazy initialization | `OnceLock` / `LazyLock` | `Mutex<Option<T>>` |
| Work distribution | `crossbeam_channel` | `Mutex<VecDeque>` |
| Async event fan-out | `broadcast` | shared vec behind mutex |
| Async availability gate | `Semaphore` | mutex-guarded counter |
| One-shot signal | `oneshot` / `crossbeam_channel` | mutex-guarded bool |

**When Mutex is acceptable:** Sub-microsecond critical sections that never
cross an `.await` point. Always document the justification in a code comment.

### Mutex selection: tokio vs std

When mutable state grows enough behavior to deserve its own boundary, prefer a
message/actor seam instead of another shared lock. `JobRegistry` already
crossed that threshold and now uses an owned actor task
(`mpsc::UnboundedSender` + `oneshot` replies) rather than a
`tokio::sync::Mutex<HashMap<...>>`.

| Type | Used for | Hold duration | Why |
|------|----------|---------------|-----|
| `tokio::sync::Mutex` | `OperationalCounterStore.counters` | One small inspect/mutate call | Tiny store-local counter state; keeps metrics bookkeeping separate without actor overhead |
| `tokio::sync::Mutex` | `WorkerGroup.bootstrap` | Held across worker spawn `.await` (1-10 s) | Serializes model-loading spikes per key |
| `std::sync::Mutex` | `WorkerGroup.idle` (VecDeque) | ~1 μs (push/pop) | Never held across `.await`; avoids tokio fair-scheduling overhead |
| `std::sync::Mutex` | `WorkerPool.groups` (HashMap) | ~1 μs (lookup) | Same rationale |

Those remaining mutexes are deliberate owner-local exceptions, not new
coordination seams:

- `OperationalCounterStore.counters` is a tiny metrics bookkeeping cell.
- `WorkerGroup.bootstrap` serializes expensive worker startup for one key.
- `WorkerGroup.idle` and `WorkerPool.groups` protect microsecond queue/map
  bookkeeping that never crosses an `.await`.

**Rule:** Reach for an actor/message boundary first when a shared collection
starts to accumulate real behavior. Use `tokio::sync::Mutex` when a small
async-owned state cell truly needs straightforward shared mutation. Use
`std::sync::Mutex` for microsecond critical sections that never cross an
`.await`. Using `std::sync::Mutex` across `.await` would deadlock the executor;
using `tokio::sync::Mutex` for trivial operations wastes scheduling overhead.

No `RwLock` usage anywhere in the codebase. No `parking_lot` usage.

The CLI TUI does not use a mutex for render state. The blocking ratatui loop
owns `AppState` directly and receives `TuiUpdate` values over an unbounded
channel from the polling-side `ProgressSink` adapter.

### CancellationToken — cooperative cancellation

Each job gets a `CancellationToken` at creation. Cancellation is
cooperative, not preemptive:

```text
User sends DELETE /jobs/{id}
  → job.cancel_token.cancel()
    → dispatch loop checks is_cancelled() before spawning next file task
    → in-flight file tasks run to completion (no mid-inference abort)
    → worker returned to pool normally
```

The worker pool also has a pool-level `CancellationToken` that stops the
background health-check loop during graceful shutdown.

## SQLite Access

All database access uses `sqlx::SqlitePool` (native async, not
`spawn_blocking`):

- **Job store** (`db/`): WAL mode, 5 max connections, 10s busy timeout
- **Utterance cache** (`cache/sqlite.rs`): Same pool configuration
- **CLI inspection** (`cache_cmd.rs`): One-off pool for stats queries

`sqlx` manages its own connection pool and integrates directly with the
tokio runtime. Queries are truly async — they yield the thread while
waiting for SQLite's WAL lock or I/O. No `spawn_blocking` wrapper is
needed or desired.

## Subprocess Management

Two subprocess patterns coexist:

### `tokio::process::Command` — async subprocess with protocol

Used for long-lived subprocesses where the Rust code must interact with
the child asynchronously (read stdout, write stdin, wait for events):

- **Python worker spawn** (`worker/handle.rs`): Spawns `python -m
  batchalign.worker`, reads the `{"ready": true}` JSON line, then
  enters the request/response IPC loop.
- **ffprobe** (`runner/util.rs`): Queries audio duration/metadata and
  `.await`s the result.

### `std::process::Command` — blocking subprocess

Used where the Rust code simply needs to run a command and wait:

- **Inside `spawn_blocking`**: ffmpeg conversion in `ensure_wav` — the
  thread is already dedicated, and the `fs2` lock must be held across
  the conversion.
- **Fire-and-forget**: `open` command to launch browser (no wait needed).
- **Daemon spawn**: Detached child process with `setsid` — parent
  doesn't wait.
- **Sync probes**: `ffmpeg -version`, `tailscale` hostname lookup.

## Graceful Shutdown

Shutdown proceeds in order:

1. Axum signal handler fires (SIGINT/SIGTERM)
2. Server stops accepting new connections
3. In-flight HTTP requests complete (axum graceful shutdown)
4. `WorkerPool.cancel` token fires → health-check loop exits
5. App-level `JoinSet` is drained — all running jobs complete or time out
6. Queue dispatcher task completes
7. Worker processes receive `shutdown` IPC command → exit
8. SQLite pools close

Job-level `CancellationToken`s are **not** fired during graceful
shutdown — running jobs are allowed to finish. Only explicit user
cancellation (DELETE) triggers job-level cancellation.

## Timers and Intervals

| Location | Type | Period | Purpose |
|----------|------|--------|---------|
| Job lease renewal | `tokio::time::sleep` | 60s loop | Heartbeat to prevent lease expiry on long jobs |
| Worker health check | `tokio::time::interval` | 30s (configurable) | Detect crashed workers, respawn |
| Daemon startup probe | `tokio::time::sleep` | 500ms retries | Wait for daemon `/health` to respond |
| Background server start | `tokio::time::sleep` | 2s grace period | Let server initialize before returning to CLI |

The health-check interval uses `MissedTickBehavior::Skip` — if a health
check takes longer than 30s (e.g., worker respawn), the next tick fires
immediately but skipped ticks don't burst.

The lease-renewal loop now uses a typed `LeaseRenewalOutcome` from the store,
and the actual claim/renew/release mutation lives on `Job` instead of in the
heartbeat loop itself. That makes the stop/continue boundary explicit without
leaving lease state as open-coded field choreography.

## Key Files

| File | Async role |
|------|------------|
| `lib.rs` | Runtime entry, app-level JoinSet, broadcast channel, graceful shutdown |
| `runner/mod.rs` | Job task spawn, lease renewal background task |
| `runner/dispatch/` | JoinSet + Semaphore for per-file dispatch (`infer_batched.rs`, `fa_pipeline.rs`, `transcribe_pipeline.rs`, etc.) |
| `worker/pool/mod.rs` | Worker checkout semaphore, `std::sync::Mutex` idle queue |
| `worker/pool/lifecycle.rs` | Health-check interval loop with CancellationToken |
| `worker/handle.rs` | `tokio::process::Command` for worker spawn + IPC |
| `store/mod.rs` | JobStore composition: registry actor, counter store, semaphore |
| `store/registry.rs` | JobRegistry actor mailbox + `oneshot` reply boundary |
| `store/job.rs` | Per-job `CancellationToken` |
| `db/mod.rs` | sqlx pool configuration (WAL, 5 connections) |
| `cache/sqlite.rs` | sqlx pool for utterance cache |
| `ensure_wav.rs` | `spawn_blocking` for ffmpeg + file locking |
| `routes/jobs/stream.rs` | SSE via broadcast channel subscription |
| `queue.rs` | Queue dispatcher background task |
