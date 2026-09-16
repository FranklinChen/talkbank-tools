# batchalign: HTTP Server, Job Store, and NLP Orchestration

**Status:** Current
**Last modified:** 2026-09-15 18:27 EDT

## Overview

Axum-based REST server managing job lifecycle, Python worker dispatch, and server-side
CHAT orchestration (CHAT ownership boundary, server owns parse/cache/inject/serialize,
Python workers provide stateless NLP inference only).

## Versioning (BUILD_HASH)

Build identity is a `git describe` string assembled in `build.rs`;
staleness is judged by build identity, never semver. Details: the
repo-root `CLAUDE.md` and the book's deployment pages.

## Module Map

See `src/` directly; a hand-maintained tree here rotted (a 2026-07
audit found ten missing modules). Orientation: `runner/` (per-job
async tasks + dispatch), `command_model/` + `planning/` + `execution/`
(command registry, immutable JobPlan, recipe kernel), `worker/`
(Python worker pool + gates), `db/` (SQLite WAL), `morphosyntax/`,
`fa/`, `pipeline/`, `merge_verify/`, `chat_ops/`, `cli/`.

## Job Registry Concurrency Model

`JobRegistry` no longer exposes a shared `Mutex<HashMap<...>>` boundary.
`JobStore` creates one owned actor task with an `mpsc::UnboundedSender`
mailbox. Callers submit either:

- `Inspect` closures for read-only projections
- `Mutate` closures for in-place transitions

Each request pairs with a `oneshot` reply so callers still `await` a typed
result. Prefer the named store/registry methods for normal job-local work;
`inspect_all()` / `mutate_all()` remain the bulk escape hatches for recovery and
other collection-wide operations.

Route, query, and runner code should think in terms of job transitions and
projections, not in terms of "lock the map and poke fields."

## Route Endpoints

| Method | Path | Purpose |
|--------|------|---------|
| POST | `/jobs` | Submit job (validates command, checks conflicts) |
| GET | `/jobs`, `/jobs/{id}` | List/get jobs |
| GET | `/jobs/{id}/results[/{filename}]` | Download results |
| GET | `/jobs/{id}/stream` | SSE streaming (real-time progress) |
| POST | `/jobs/{id}/cancel`, `/jobs/{id}/restart` | Lifecycle |
| DELETE | `/jobs/{id}` | Permanent delete |
| GET | `/health` | Version, capabilities, worker state |

## Job Lifecycle and Requeue Invariant

Canonical: `book/src/batchalign/architecture/job-state-machine.md`
(exclusive-runner invariant, RunGeneration restart safety, requeue
rules, bootstrap reconciliation). Contract test:
`tests/restart_handoff.rs`. Do not restate here.

## Dispatch Routing (runner/)

Per-command dispatch modules live in `runner/dispatch/`; the routing
story is in `book/src/batchalign/architecture/command-flowcharts.md`.

## Type System

Domain newtypes are defined in `batchalign-types` using `string_id!` and `numeric_id!`:
- **`../batchalign-types/src/macros.rs`**: macro definitions (generates Deref, serde transparent, From, Borrow, etc.)
- **`../batchalign-types/src/domain/`**: `JobId`, `CommandName`, `ReleasedCommand`, `LanguageCode3`, `LanguageSpec`, `DisplayPath`, `StampSafeText` (the one stamp-safe text type, wrapped by `ReportedEngineName`), `CorrelationId`, `NumSpeakers`, `UnixTimestamp`, `DurationMs`, `MemoryMb`, etc.
- **`../batchalign-types/src/scheduling.rs`**: `AttemptId`, `WorkUnitId`
- **`types/params.rs`**: `CachePolicy`, `WorTierPolicy` enums; `MorphosyntaxParams`, `FaParams`, `AudioContext` structs
- **`pipeline/mod.rs`**: `PipelineServices` (shared infrastructure refs: pool, cache; no engine identity, each stage names its own)
- **`engine_reports.rs`**: a worker's capability report, admitted once in `WorkerPool::record_capabilities`, which stores each worker key's outcome (admitted, or refused and listed in `/health`). A report carries task support plus the FA identity only: FA's entry names the engine (or is null before FA loads), every other entry is null, and admission refuses `MissingEngineReport`, `UnadvertisedEngineReport` and `EngineNamedForNonFaTask`. `FaCacheNamespace` (a plain newtype over `ReportedEngineName`) is the only identity read from a report, only through `FaCacheNamespace::from_loaded` on a post-load `LoadedCapabilities`; translate, coref and morphosyntax name their engines on result items
- **`capability.rs`**: only `command_supported` (the worker supports the command's primary infer task; decides `/health` advertisement, job acceptance and the dispatch check) and `WorkerCapabilitySnapshot` (commands as `Vec<ReleasedCommand>` and tasks, derived behind private fields). The FA engine is read at dispatch by the forced-alignment arm in `runner/routing.rs` with `FaCacheNamespace::from_loaded`
- **`cache/mod.rs`**: typed cache API; `CacheTask<N>` constants in `cache::tasks` pair each task with its sealed `CacheNamespace` type, and `UtteranceCache` does not implement the string-keyed `CacheBackend`; `BuildOwnedNamespace` is the one representation behind `UtrAsrCacheNamespace`, `RevAsrModelRevision`, `SpeakerEvidenceModelRevision` and `SpeakerNormalizationRevision`
- **`provenance.rs`**: `StampCodec` owns the stamp grammar for the writer, `extract_provenance` and the no-op write gate, which sets aside every stamp `command_model::commands_stamped_by` derives for the job's recipe; the private `recognize` module owns "is this comment ours" in two steps (`RecognizedComment::classify` names a stamp's command, `NamedStamp::parse` reads its body), so the gate can tell a damaged stamp of its own command from one belonging to another; see `book/src/batchalign/architecture/provenance.md`

**Boundary patterns:** Raw `String` from HTTP → `JobId::from()` at handler entry. `&Path` in domain code → `to_string_lossy()` at IPC/JSON. `bool` from CLI → `CachePolicy::from()` at dispatch. See `book/src/batchalign/architecture/type-driven-design.md`.

## Admission and Eviction Gates

Admission (CPU + memory) gates every worker spawn; per-engine memory
reservations feed it; idle eviction reclaims workers. The canonical
tables and tier floors: `book/src/batchalign/developer/memory-safety.md`.
Never bypass the gates in new spawn paths.

## Middleware Stack

See `book/src/batchalign/developer/http-body-limits.md` and the
server-architecture page for the axum layer order.
