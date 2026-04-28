# Parallelism Architecture

**Status:** Current
**Last updated:** 2026-04-25 12:56 EDT

## Three layers of parallelism

batchalign3 has three independent parallelism controls:

```mermaid
graph TD
    A["--workers N (CLI flag)"] --> B["max_workers_per_job (ServerConfig)"]
    B --> C["compute_job_workers()"]
    C --> D["JoinSet + Semaphore\n(file-level parallelism)"]

    E["max_workers_per_key\n(PoolConfig, default: 8)"] --> F["Worker Pool\n(Python processes per profile+lang)"]

    G["gpu_thread_pool_size (default 4)"] --> H["dispatch_semaphore\n(Rust, SharedGpuWorker)"]
    G --> I["ThreadPoolExecutor\n(Python, max_workers=K)"]
    H --> J["execute_v2 in flight ≤ K\n(matches Python serving capacity)"]
    I --> J

    style A fill:#f9f,stroke:#333
    style E fill:#bbf,stroke:#333
    style G fill:#bfb,stroke:#333
    style J fill:#bfb,stroke:#333
```

Layer 3's twin nodes (`dispatch_semaphore` on the Rust side and
`ThreadPoolExecutor` on the Python side) share **one** ceiling
`K = gpu_thread_pool_size`. The Rust gate ensures a caller waiting for
an executor slot does not hold an active per-request timer; the timer
only ticks once a slot is granted and work is issued. See
`developer/worker-protocol-v2.md` § "The dispatch semaphore contract"
for the full architectural rule.

### Layer 1: File parallelism (user-facing)

**What it controls:** How many files are processed concurrently within a single job.

**User control:** `--workers N` CLI flag, or `max_workers_per_job` in `server.yaml`.

**Default behavior:**
- GPU-heavy commands (`transcribe`, `align`, `benchmark`): **1 file** (prevents OOM)
- CPU-only commands (`morphotag`, `utseg`, `translate`): auto-tuned by `compute_job_workers()` based on available RAM and CPU cores

**Implementation:** `runner/dispatch/` uses a `JoinSet` with a `Semaphore(num_workers)` to cap concurrent file processing tasks.

```mermaid
sequenceDiagram
    participant CLI as CLI (--workers N)
    participant Daemon as Auto-Daemon
    participant Server as Rust Server
    participant AutoTune as compute_job_workers()
    participant Runner as Job Runner

    CLI->>Daemon: spawn with --workers N
    Daemon->>Server: start with max_workers_per_job=N
    Note over Server: Job submitted with 10 files
    Server->>AutoTune: command, num_files, config
    AutoTune-->>Server: min(N, files, RAM/budget, CPUs)
    Server->>Runner: Semaphore(computed_workers)
    Runner->>Runner: Process files with bounded concurrency
```

### Layer 2: Worker pool (operator-facing)

**What it controls:** How many Python worker processes exist per (profile, language, engine) key.

**User control:** `max_workers_per_key` in `server.yaml` (not a CLI flag -- this is an operator concern).

**Default:** 8 per key. GPU profile: 1 process (concurrent via threads). Stanza profile: auto-tuned. IO profile: 1 process.

**Implementation:** `worker/pool/mod.rs` manages worker lifecycle. Workers are spawned lazily and cached.

### Layer 3: GPU dispatch concurrency (Rust gate + Python pool)

**What it controls:** How many `execute_v2` calls are in flight at the
same time per shared GPU worker.

**User control:** `gpu_thread_pool_size` in `server.yaml` (default: 4).
This single knob sets both the Python `ThreadPoolExecutor(max_workers=K)`
*and* the Rust-side `dispatch_semaphore` permit count, so the two sides
agree on the in-flight ceiling.

**Default:** 4. On Apple Silicon (MPS excluded for batchalign3), set to
1 — there is no compute parallelism to gain, and a higher value just
means CPU-bound inferences contending for cores.

**Implementation:**
- Rust: `worker/pool/shared_gpu/stdio.rs` and
  `worker/pool/shared_gpu/tcp.rs` each carry a
  `dispatch_semaphore: Arc<Semaphore>` with `K` permits, acquired
  *before* the per-request timeout starts.
- Python: `batchalign/worker/_protocol.py::_serve_stdio_concurrent`
  hosts a `ThreadPoolExecutor(max_workers=K)` so multiple FA/ASR/Speaker
  inferences can run concurrently when the device releases the GIL.

```mermaid
sequenceDiagram
    participant T1 as Task 1
    participant T2 as Task 2
    participant T3 as Task 3 (queued)
    participant Sem as dispatch_semaphore (K=2)
    participant Worker as SharedGpuWorker
    participant Py as Python ThreadPool (max_workers=2)

    T1->>Sem: acquire (granted)
    T2->>Sem: acquire (granted)
    T3->>Sem: acquire (waiting — no timer running yet)
    T1->>Worker: pending.insert + stdin.write + tokio::time::timeout START
    T2->>Worker: pending.insert + stdin.write + tokio::time::timeout START
    Worker->>Py: execute_v2 (id=1, id=2)
    Py-->>Worker: response(id=1)
    Worker-->>T1: success
    T1-->>Sem: release
    Sem->>T3: granted
    T3->>Worker: pending.insert + stdin.write + tokio::time::timeout START
```

Task 3's per-request timer only starts at the moment it acquires the
permit — never during queue-wait. This is the architectural contract
asserted by
`tests/gpu_concurrent_dispatch.rs::gpu_concurrent_dispatch_does_not_charge_queue_wait_against_per_request_timeout`.

## Why GPU commands default to 1 worker

Each GPU-heavy inference (Whisper ASR, Whisper FA, Wave2Vec) loads 2-5 GB of model weights into GPU/MPS memory. Processing multiple files concurrently means multiple inference requests running simultaneously, all sharing the same GPU memory pool.

On a 64 GB developer machine with MPS:
- 1 concurrent file: ~5 GB GPU memory -- safe
- 4 concurrent files: ~5 GB GPU x 4 threads = ~20 GB GPU pressure -- risky
- 8 concurrent files (old default): ~40 GB GPU pressure -- **kernel OOM crash**

The server's auto-tuner estimated available RAM but did not account for GPU memory pressure. Setting GPU commands to default to 1 file prevents this class of crash entirely.

Operators with dedicated GPU hardware (e.g., net with M3 Ultra 256 GB) can safely increase this via `--workers N` or `server.yaml`.
