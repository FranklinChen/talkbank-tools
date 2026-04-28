# Test Server and Worker Lifecycle

**Status:** Current
**Last updated:** 2026-03-20 17:45

## The problem

ML integration tests (golden snapshots, audio transcription, parity checks,
profile verification) need a running batchalign server with loaded Python ML
workers. Each worker loads Whisper, Stanza, or pyannote models consuming 2–5 GB
RAM. The lifecycle of these workers during testing has caused three kernel OOM
panics (2026-03-19, 2026-03-20) on a 64 GB developer machine.

### Root cause: per-binary server isolation (resolved)

Previously, each Rust integration test binary (`golden.rs`, `golden_audio.rs`,
`golden_parity.rs`, `profile_verification.rs`, etc.) was compiled as a separate
executable. Each binary included `mod common;` which compiled the shared fixture
module into its own address space. The fixture uses a `static LazyLock` for the
server backend — process-local, so 7 binaries = 7 independent worker pools.

### Implemented solution: single binary consolidation

All 7 ML binaries are now consolidated into one binary (`ml_golden.rs`) with
submodules. One binary = one process = one `LazyLock` = one `PreparedWorkers` =
one set of loaded models. Peak memory is ~8-12 GB instead of 7x that.

```
ml_golden (one binary, one process)
  → LazyLock → PreparedWorkers → python3 (Stanza, Whisper, Wave2Vec, pyannote)
  ├── ml_golden::golden           (12 tests)
  ├── ml_golden::golden_audio     (22 tests)
  ├── ml_golden::golden_parity    (16 tests)
  ├── ml_golden::live_server_fixture (5 tests)
  ├── ml_golden::profile_verification (3 tests)
  ├── ml_golden::option_receipt   (5 tests)
  └── ml_golden::error_paths      (7 tests)
```

The `LiveServerSession` fixture within the binary is well-designed:

- **One `PreparedWorkers` backend** shared across all 70 tests
- **Fresh HTTP server per session** (new port, new jobs dir, new SQLite)
- **Semaphore-gated sessions** so tests don't collide on control-plane state
- **Warm model cache** across tests — only the first test pays cold-start cost

## Architecture overview

The fixture system has three layers: the process-global backend (loaded
models), the fixture thread (session lifecycle), and per-test sessions
(isolated HTTP servers). This diagram shows the full structure:

```mermaid
graph TB
    subgraph "ml_golden process"
        LL["static LIVE_FIXTURE: LazyLock"]
        LL -->|"initializes once"| FT["Fixture Thread<br/>(dedicated OS thread + Tokio runtime)"]
        FT -->|"owns"| BS["BackendState"]
        BS -->|"Ready"| LFB["LiveFixtureBackend"]
        LFB --> PW["PreparedWorkers"]
        LFB --> SC["ServerConfig"]

        PW -->|"holds"| WP["WorkerPool"]
        WP -->|"manages"| W1["python3 -m batchalign.worker<br/>--task morphosyntax --lang eng<br/>(Stanza ~1.5 GB)"]
        WP -->|"manages"| W2["python3 -m batchalign.worker<br/>--task fa --lang eng<br/>(Whisper/Wave2Vec ~3 GB)"]
        WP -->|"manages"| W3["python3 -m batchalign.worker<br/>--task asr --lang eng<br/>(Whisper ~3 GB)"]

        FT -->|"creates/destroys"| AS["ActiveSession"]
        AS --> AX["Axum HTTP server<br/>(ephemeral port)"]
        AS --> DB["SQLite (jobs)"]
        AS --> TD["TempDir (state)"]

        subgraph "Test Threads"
            T1["golden::morphotag_eng_simple"]
            T2["golden_audio::align_eng_wav2vec"]
            T3["golden_parity::parity_morphotag_eng"]
        end

        T1 -->|"Semaphore acquire"| SEM["Semaphore(1)"]
        T2 -.->|"waits"| SEM
        T3 -.->|"waits"| SEM
        SEM -->|"Acquire command"| FT
    end

    style LL fill:#f9f,stroke:#333
    style PW fill:#bbf,stroke:#333
    style W1 fill:#fbb,stroke:#333
    style W2 fill:#fbb,stroke:#333
    style W3 fill:#fbb,stroke:#333
    style AX fill:#bfb,stroke:#333
```

## Fixture thread lifecycle

The fixture thread runs on a dedicated OS thread with its own Tokio runtime.
It processes `Acquire` and `Release` commands from test threads via an
`mpsc` channel. The backend (Python workers + models) is initialized lazily
on the first `Acquire` and cached for all subsequent sessions.

```mermaid
sequenceDiagram
    participant T as Test Thread
    participant S as Semaphore(1)
    participant FT as Fixture Thread
    participant BS as BackendState
    participant PW as PreparedWorkers

    Note over FT: Thread started by LazyLock

    T->>S: acquire_owned()
    S-->>T: OwnedSemaphorePermit

    T->>FT: Acquire { reply }

    alt BackendState::Uninitialized (first test)
        FT->>BS: ensure_backend()
        BS->>PW: prepare_workers()
        Note over PW: resolve_python()<br/>spawn workers<br/>load models<br/>(Stanza, Whisper, etc.)
        PW-->>BS: PreparedWorkers
        BS-->>FT: BackendState::Ready
    end

    FT->>FT: start_session(backend)
    Note over FT: TempDir::new()<br/>RuntimeLayout::from_state_dir()<br/>create_app_with_prepared_workers()<br/>TcpListener::bind("127.0.0.1:0")<br/>axum::serve()

    FT-->>T: SessionSnapshot { base_url, state_dir, infer_tasks }

    Note over T: Run test:<br/>POST /jobs → poll → GET /results

    T->>FT: Release { reply }
    FT->>FT: cleanup_session()
    Note over FT: server_task.abort()<br/>state.shutdown_for_reuse(5s)<br/>drop(state)<br/>drop(runtime_root)
    FT-->>T: ack

    T->>S: drop(OwnedSemaphorePermit)
    Note over S: Next test can acquire
```

## Python worker startup and model loading

When the backend initializes (first `Acquire`), `prepare_workers()` spawns
Python worker subprocesses. Each worker loads multi-GB ML models into memory
and signals readiness over its stdio JSON-lines protocol.

```mermaid
sequenceDiagram
    participant RS as Rust Server
    participant WP as WorkerPool
    participant PY as python3 -m batchalign.worker

    RS->>WP: prepare_workers(config, pool_config)

    WP->>PY: spawn(--task morphosyntax --lang eng)
    Note over PY: import stanza<br/>stanza.Pipeline("en")<br/>~1.5 GB into RAM<br/>~15-30s cold start

    PY-->>WP: stdout: {"ready": true, "pid": 12345, "transport": "stdio"}

    WP->>PY: spawn(--task fa --lang eng)
    Note over PY: import whisper<br/>whisper.load_model("base")<br/>~3 GB into RAM<br/>~30-60s cold start

    PY-->>WP: stdout: {"ready": true, "pid": 12346, "transport": "stdio"}

    Note over WP: Workers registered in pool<br/>PID files written to ~/.batchalign3/worker-pids/

    WP-->>RS: PreparedWorkers { pool, infer_tasks }
    Note over RS: Backend ready — all subsequent<br/>Acquire commands reuse these workers
```

## Per-test session lifecycle

Each test acquires a session (serialized by the semaphore), gets a fresh
HTTP server with its own jobs directory and SQLite database, runs its test
logic, then releases the session. The Python workers persist across sessions.

```mermaid
sequenceDiagram
    participant T as Test Function
    participant LSS as LiveServerSession
    participant AX as Axum Server (ephemeral)
    participant WP as WorkerPool (shared)
    participant PY as Python Worker (warm)

    T->>LSS: require_live_server(InferTask::Morphosyntax)
    LSS->>LSS: LiveServerSession::acquire()
    Note over LSS: Semaphore → Acquire → start_session()

    T->>AX: POST /jobs { command: "morphotag", files: [...] }
    AX->>AX: Parse CHAT, extract words
    AX->>WP: checkout worker for (morphosyntax, eng)
    WP-->>AX: CheckedOutWorker (reuses warm process)
    AX->>PY: stdin: {"op": "batch_infer", "items": [...]}
    PY-->>AX: stdout: {"results": [{pos, lemma, deprel}...]}
    AX->>AX: Inject %mor/%gra into CHAT AST
    AX->>AX: Serialize CHAT, store result
    AX-->>T: 200 OK { job_id }

    T->>AX: GET /jobs/{id} (poll)
    AX-->>T: { status: "completed" }

    T->>AX: GET /jobs/{id}/results
    AX-->>T: { files: [{ content: "..." }] }

    T->>T: assert / insta::assert_snapshot!

    T->>LSS: drop (or explicit close)
    LSS->>LSS: Release → cleanup_session()
    Note over LSS: HTTP server aborted<br/>SQLite + TempDir dropped<br/>Workers stay alive for next test
```

## Cleanup and safety layers

Multiple overlapping mechanisms ensure workers are cleaned up even if tests
crash or are killed.

```mermaid
graph TB
    subgraph "Normal shutdown"
        A["Test completes"] --> B["LiveServerSession::close()"]
        B --> C["cleanup_session()"]
        C --> D["server_task.abort()"]
        C --> E["state.shutdown_for_reuse(5s)"]
        E --> F["Workers returned to pool<br/>(stay warm for next test)"]
    end

    subgraph "Drop fallback"
        G["Test panics"] --> H["LiveServerSession::Drop"]
        H --> I["thread::spawn release"]
        I --> J["cleanup_session() on background thread"]
    end

    subgraph "Process exit"
        K["All tests done /<br/>binary exits"] --> L["WorkerPool::Drop"]
        L --> M["Kill all idle workers<br/>(SIGTERM)"]
        L --> N["Remove PID files"]
    end

    subgraph "Orphan recovery (next startup)"
        O["New test run starts"] --> P["PID file reaper"]
        P --> Q["Scan ~/.batchalign3/worker-pids/"]
        Q --> R{"Worker alive?<br/>Parent dead?"}
        R -->|"Yes (orphan)"| S["SIGTERM → 2s → SIGKILL"]
        R -->|"No (stale file)"| T["Remove PID file"]
        R -->|"Both alive"| U["Skip (belongs to<br/>running server)"]
    end

    subgraph "External guard"
        V["Claude Code session"] --> W["Guard hook checks<br/>pgrep batchalign.worker"]
        W -->|"Workers found"| X["Block test command<br/>(prevent double-spawn)"]
    end

    style F fill:#bfb,stroke:#333
    style M fill:#fbb,stroke:#333
    style S fill:#fbb,stroke:#333
```

## Defense-in-depth layers

These remain as additional safety nets beyond the single-binary consolidation:

| Layer | Where | What |
|-------|-------|------|
| nextest default-filter | `.config/nextest.toml` | ML binary excluded from `cargo nextest run` |
| nextest ml test group | `.config/nextest.toml` | ML binary serialized (`max-threads=1`) when opted in |
| Claude Code guard hook | `.claude/settings.local.json` | Blocks test commands when worker processes detected |
| Global worker cap | `WorkerPool` (`max_total_workers`) | Hard ceiling on total workers across all keys |
| `WorkerPool::Drop` | `pool/mod.rs` | Kills idle workers when pool dropped without `shutdown()` |
| PID file reaper | `pool/reaper.rs` | Scans `~/.batchalign3/worker-pids/` on startup, kills orphans |

## Future: shared test daemon (historical analysis)

If the test suite outgrows the single-binary approach (e.g., the binary
becomes too large to link, or test isolation requires separate processes),
the next step is a shared test daemon. This is preserved here as a future
option, not a current plan.

The idea: one long-lived server for the entire `cargo nextest run --profile
ml` invocation, with test binaries connecting as HTTP clients. The server's
autotuner and memory gate handle scheduling. Models load once and stay warm.

**Implementation options (in order of simplicity):**
1. **nextest setup script** — `[profile.ml.scripts.setup]` starts a daemon
2. **Test-managed daemon** — file-lock coordination in `common/mod.rs`
3. **Always-on dev daemon** — assume a running server, skip if absent

## Relationship to the broader worker architecture

The test lifecycle problem is a microcosm of the deployment lifecycle:

- **Development**: one developer machine, multiple concurrent test/dev sessions
- **Production (net)**: one server, multiple concurrent jobs from the fleet

The single-daemon test architecture exercises the same code paths as production:
autotuner, memory gate, worker pool, idle timeout, health checking. The
per-binary in-process approach exercises none of these, which is why it was
blindsided by the OOM crashes that production handles gracefully.

Making tests use the production dispatch path also means test failures surface
real bugs (scheduling, memory, lifecycle) rather than hiding them behind
per-test isolation.
