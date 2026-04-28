# Worker Architecture Assessment: Frameworks vs Custom Orchestration

**Status:** Draft
**Last updated:** 2026-03-19

This report evaluates whether batchalign3 should adopt an off-the-shelf
framework for worker lifecycle management, or continue with custom
orchestration. It incorporates the full history of framework experiments from
batchalign-next (Feb 2026), the bugs discovered during the Miami transcribe
debugging session (Mar 19, 2026), and the current state of the Rust migration.

## Executive Summary

batchalign-next tried Ray, Celery+Redis, and Temporal.io in a rapid 4-day
evaluation (Feb 6-9, 2026). All three were abandoned. The winning architecture
was an in-memory JobStore with ThreadPoolExecutor — the simplest possible
solution. That architecture was ported to Rust in batchalign3 and is now
production.

Six weeks later, a single debugging session on a 47-file transcribe job
exposed **five distinct bugs** in the custom orchestrator: TOCTOU race in GPU
worker creation, pre-scale routing to the wrong worker type, engine overrides
key mismatch, pre-scale skipped for `--lang auto`, and worker ready timeout
too short for Whisper cold start. All five stem from **entangled worker
lifecycle and request dispatch** — the exact class of problem that frameworks
are designed to solve.

The question is not "were the frameworks bad?" but "have our constraints
changed enough that the trade-offs are different now?"

## What Was Tried and Why It Failed

### Ray (Feb 6-7, 2 days)

**What:** Distributed computing framework with WorkerActor pattern, Tailscale
cluster support, pipelined dispatch.

**Why it failed:**
- Ray's overhead was high for small files
- Network latency dominated processing time
- MPS (Apple Metal) doesn't work with Ray's multiprocessing
- Debugging was painful (multi-process, multi-machine logs)

**Code deleted:** ~1,500 lines (cluster.py, dispatch_ray.py, distributed/)

**Verdict then:** "Distributed computing ≠ solution to local scheduling problem"

**Verdict now:** Still correct. Ray solves the wrong problem. Our bottleneck is
not distributing work across machines — it's managing model lifecycle on a
single machine. Ray's actor model is also a poor fit for long-lived GPU workers
that share models across requests.

### Celery + Redis (Feb 8, 1 day)

**What:** Task queue with Redis backend, autoscale, loading-aware concurrency.

**Why it failed:**
- Redis was an external dependency users didn't want to run
- Celery's worker spawning was slow
- Thundering herd: all workers loaded models simultaneously
- 13 bug-fix commits in 19 hours

**Code deleted:** ~750 lines (celery_app.py, redis_store.py, tasks.py)

**Verdict then:** "Frameworks don't fix architectural issues (model lifecycle)"

**Verdict now:** **Partially wrong.** The specific objections were valid in Feb
2026 (Redis dependency, Celery's fork-based workers loading models redundantly).
But the conclusion that "frameworks don't fix model lifecycle" conflated
Celery's specific limitations with the general concept of external orchestration.
Celery was the wrong framework, not frameworks in general.

### Temporal.io (Feb 9, 4 hours)

**What:** Workflow orchestration with typed activities, durable state, retry.

**Why it failed:**
- gRPC 4MB message limit (audio metadata exceeded it)
- Memo API mismatch (SDK version differences)
- Required running a Temporal server daemon
- 5 production bugs in one session, all invisible in tests because mocks hid
  SDK behavior

**Commit message:** "It's overkill for a single lab server"

**Verdict then:** "Frameworks hide complexity; tests with mocks miss real SDK
behavior"

**Verdict now:** **Largely correct for Temporal specifically.** Temporal is
designed for multi-service, multi-team orchestration at scale. It's genuinely
overkill for a single-machine ML pipeline. The gRPC message limit is also a
real constraint for audio-heavy workflows. However, the "single lab server"
framing understated the problem — we now have 11 machines, a production
server, and reliability requirements.

### In-Memory JobStore + ThreadPoolExecutor (Feb 9 - present)

**What:** Single-process Python server with in-memory state, SQLite
write-through, ThreadPoolExecutor for GPU commands, ProcessPoolExecutor for
CPU commands.

**Why it won:**
- Simple: one process, no external daemons
- Testable: no mocks needed
- Observable: /health endpoint
- Correct: SQLite WAL + threading.Lock

**Ported to Rust as:** batchalign3's axum server + WorkerPool + SharedGpuWorker
+ SQLite JobStore

## What's Changed Since February

The Feb 2026 evaluation was done under these constraints:
- Python owned the entire stack (CHAT parsing, NLP, orchestration)
- The server was a Python FastAPI process managing Python threads/processes
- Model lifecycle was entangled with CHAT processing in every code path
- The team was in rapid prototyping mode (4 frameworks in 4 days)
- "Single lab server" was the deployment model

**Today's constraints are fundamentally different:**

| Dimension | Feb 2026 | Mar 2026 |
|-----------|----------|----------|
| Language boundary | Python-only | Rust server + Python ML workers |
| CHAT processing | Python (mixed with NLP) | Rust (completely separated) |
| Worker protocol | In-process (threads/forks) | Subprocess + stdio JSON-lines IPC |
| Deployment | Single machine experimentation | 11-machine fleet, production |
| Model lifecycle | Entangled with dispatch | Already separated (spawn + ready signal) |
| Reliability needs | Best-effort prototyping | Production — bugs = lost researcher time |

The critical change: **the Rust/Python boundary already exists.** Workers are
already separate processes communicating over IPC. The "model lifecycle
entanglement" that made frameworks useless in Feb is already solved — Python
workers are stateless inference endpoints. The Rust server already doesn't
touch model loading.

This means the Feb conclusion — "framework migration adds complexity but does
not remove the hardest part: model lifecycle" — **no longer applies.** The
hardest part is already solved. What remains is exactly what frameworks do well:
process management, health checking, readiness gating, and request routing.

## The Bugs That Frameworks Would Have Prevented

From the Mar 19 debugging session:

| Bug | Root cause | Framework equivalent |
|-----|-----------|---------------------|
| TOCTOU race in GPU worker map | Lock dropped between check and spawn | Framework owns worker registry atomically |
| Pre-scale uses wrong worker type | Caller must know GPU vs non-GPU path | Framework routes internally |
| Engine overrides key mismatch | Two code paths derive the same key differently | Framework uses declared capabilities, not key construction |
| Pre-scale skipped for lang=auto | LanguageSpec enum not handled in pre-scale | Framework doesn't need pre-scale — workers exist or they don't |
| Worker ready timeout too short | Arbitrary 120s timeout for model loading | Framework watches progress, not wall clock |

Every one of these bugs exists because we hand-rolled process lifecycle
management in ~2,000 lines of Rust (pool/mod.rs, handle.rs, shared_gpu.rs,
lifecycle.rs, auto_tune.rs). A framework replaces those 2,000 lines with
battle-tested, community-maintained code.

## Framework Re-Evaluation for Current Architecture

### Option A: Keep Custom Orchestration, Harden

**What:** Fix the five bugs, add progress-based readiness, add the
worker-as-service pattern described in the redesign plan.

**Pros:**
- No new dependencies
- Full control over every behavior
- We understand the code completely

**Cons:**
- We keep maintaining ~2,000 lines of non-core infrastructure code
- Every new feature (auto-scaling, multi-language warmup, graceful degradation)
  is more custom code
- The successor inherits custom infrastructure instead of industry-standard tooling
- We spent 6+ hours debugging infrastructure bugs instead of working on CHAT tooling

**Effort:** Medium. The worker-as-service refactor is ~1-2 weeks.

**Risk:** We continue to discover new infrastructure bugs as usage patterns change.

### Option B: Adopt a Lightweight Model Server (BentoML or similar)

**What:** Each inference module (ASR, FA, morphosyntax, utseg, translate,
speaker, etc.) becomes a BentoML "Service" that loads its model, exposes an
inference endpoint, and handles health/readiness natively. The Rust server
calls these services over HTTP instead of managing subprocess lifecycle.

**Pros:**
- Model lifecycle fully delegated (loading, health, readiness, scaling)
- Standard deployment (Docker, Kubernetes, systemd — all supported)
- Built-in batching, concurrency control, GPU memory management
- The successor sees `bentoml serve asr_service` — self-explanatory
- Eliminates: WorkerPool, SharedGpuWorker, WorkerHandle, pre-scale, auto-tune,
  ready timeout, TOCTOU race, key derivation — all of it

**Cons:**
- HTTP overhead vs stdio IPC (microseconds per request — negligible for ML inference)
- BentoML dependency (~50 MB)
- Learning curve for the team
- Deployment changes (services must be started separately or via BentoML orchestrator)

**Effort:** Medium. The Python inference modules already have the right shape
(stateless functions that load a model and return results). Wrapping them as
BentoML services is mechanical. The Rust server replaces `WorkerPool` with HTTP
client calls.

**What BentoML was NOT evaluated in Feb:** It wasn't tried at all (0 commits
mentioning it). The Feb evaluation focused on task queues (Celery) and workflow
engines (Temporal, Ray) — not model serving frameworks. BentoML solves a
different problem: model lifecycle and serving, which is exactly what we need.

### Option C: Use Systemd/Launchd + Unix Sockets (DIY Service Pattern)

**What:** Workers become standalone daemons started by the OS service manager.
Communication over Unix domain sockets. The Rust server discovers workers by
scanning a socket directory.

**Pros:**
- No framework dependency
- Uses standard OS facilities (systemd, launchd)
- Workers survive server restarts
- Simple to understand

**Cons:**
- Still custom IPC code (swap stdio for sockets)
- Still custom health checking and readiness
- Still custom service discovery
- Doesn't solve batching, scaling, or GPU memory management
- The successor inherits systemd units and custom discovery code

**Effort:** Low-medium. Transport swap + service discovery + systemd/launchd units.

**Risk:** We solve the lifecycle problem but keep all the other custom
infrastructure. Lower improvement-to-effort ratio than Option B.

### Option D: Celery Revisited (with Current Architecture)

**What:** The Rust server submits tasks to Celery workers via Redis. Each
Celery worker loads models at startup and processes inference requests. The
Rust server is a Celery client (submits tasks, polls results).

**Pros:**
- Battle-tested task queue
- Built-in retry, timeout, health checking
- Workers are long-lived (models loaded once)
- Redis is the only external dependency (trivial to run)

**Cons:**
- The Feb objections partially remain: Redis dependency, fork-based workers
- Celery's fork model means each worker loads its own models (high memory)
- Celery is Python-centric; calling it from Rust requires an HTTP bridge or
  Redis protocol implementation
- Celery is aging (less active development than newer alternatives)

**Feb objection re-evaluation:** The "users don't want to run Redis" objection
is weaker now — we're deploying to a managed fleet, not end-user machines.
The "thundering herd model loading" objection is weaker because we now
understand model profiles and could configure Celery workers per-profile. But
the fork-based memory model is still bad for GPU workers.

**Effort:** Medium-high. Rust-to-Celery bridge, Redis deployment, worker
configuration.

### Option E: Do Nothing (Just Fix the Timeout)

**What:** Bump `ready_timeout_s` to 300s, ship the pre-scale fixes, and move on.

**Pros:**
- Minimal effort
- No new dependencies
- Production unblocked today

**Cons:**
- Every bug we found today will have analogs as we add features
- The 2,000 lines of custom infrastructure remain a maintenance burden
- The timeout is a band-aid, not a fix

**Effort:** Already done (timeout bumped to 300s).

**Risk:** Next debugging session will find more infrastructure bugs.

## Recommendation

**Short term (this week):** Option E — ship the timeout fix and pre-scale
improvements. Production is unblocked.

**Medium term (before public release):** Option C — the worker-as-service
pattern with Unix sockets and OS-level service management. This eliminates the
lifecycle bugs without adding framework dependencies. It's the smallest change
that addresses the root cause.

**Evaluate for post-release:** Option B — BentoML or similar model serving
framework. This is the "correct" long-term architecture but requires more
evaluation. Key questions to answer:

1. Does BentoML work well with MPS (Apple Metal GPU)?
2. Does BentoML support the SharedGpuWorker pattern (multiple inference types
   sharing one model process)?
3. What's the HTTP overhead vs stdio IPC for our request sizes?
4. Can BentoML services be managed by launchd on macOS?
5. Does the successor benefit more from understanding BentoML (transferable
   skill) or understanding our custom code (TalkBank-specific)?

**Not recommended:** Options A (throwing more custom code at the problem) or D
(Celery — wrong tool for model serving).

## The Succession Test

The ultimate question: when the successor inherits this system in 3-5 years,
which architecture is most operable by someone who has never met us?

- **Custom orchestrator (current):** Requires reading 2,000 lines of Rust to
  understand worker lifecycle. Bugs require deep knowledge of the pool
  internals. No documentation exists outside our codebase.

- **BentoML services:** `bentoml serve whisper_service` — the successor reads
  BentoML's documentation (maintained by a company with hundreds of
  contributors) and understands the deployment model. Our custom code is
  limited to the CHAT processing pipeline, which is our actual domain expertise.

- **Systemd services with Unix sockets:** `systemctl status batchalign-gpu-worker`
  — the successor uses standard Linux tooling. Our custom code is the IPC
  protocol and service discovery, which is small and well-documented.

All three are better than the current situation where the successor must
understand our custom WorkerPool, SharedGpuWorker, TOCTOU mitigation, pre-scale
routing, engine override key derivation, ready timeout tuning, and auto-tune
memory budgeting — all of which are one-off code that exists nowhere else.

## Appendix: Key Reference Documents from batchalign-next

| Document | Location | Key finding |
|----------|----------|-------------|
| Experience Report | `EXPERIENCE_REPORT.md` lines 253-321 | Timeline of all 4 framework experiments |
| ADR: Server Orchestration | `docs/adr-server-orchestration-migration-2026-02-19.md` | "Framework migration adds complexity but does not remove model lifecycle" |
| Concurrency Assessment | `docs/concurrency-assessment-2026-02-19.md` | "Main throughput limits are ML model behavior, not queue framework choice" |
| Fleet Management Plan | `docs/fleet-management-plan.md` lines 293-304 | "Rust-native distribution only matters at much larger scale" |
| Replace Temporal commit | `1f2b6a2c` | "It's overkill for a single lab server" — 5 production bugs in one session |

## Appendix: Framework Experiment Timeline

| Date | Duration | Framework | Commits | Result |
|------|----------|-----------|---------|--------|
| Feb 6-7 | 2 days | Ray | 11 | Abandoned — overhead, MPS incompatibility |
| Feb 8 | 1 day | Celery + Redis | 13 | Abandoned — external dep, thundering herd |
| Feb 9 AM | 4 hours | Temporal.io | 8 | Abandoned — gRPC limits, 5 prod bugs |
| Feb 9 PM | shipped | JobStore + ThreadPool | 1 | Production to today |

Total framework code written and deleted: ~3,000+ lines across 32+ commits.
