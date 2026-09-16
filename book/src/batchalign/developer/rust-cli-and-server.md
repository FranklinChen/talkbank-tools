# Rust CLI and Server

**Status:** Current
**Last updated:** 2026-09-15 18:27 EDT

This page covers the Rust control plane that powers `batchalign3`: the CLI
client, the HTTP server, and how to extend them.

The current worker-boundary replacement plan is documented separately in
[Worker Protocol V2](worker-protocol-v2.md). That spec is the source of truth
for replacing the legacy stdio JSON-lines worker contract.

## Crate Map

After the 2026-04-28 monorepo merge, batchalign source lives as a
small set of sibling crates inside this workspace:

| Crate | Role |
|-------|------|
| `crates/batchalign/` | The runtime crate: Clap CLI, dispatch router, direct-host bootstrap, Axum HTTP server, job store, worker pool, cache, and command-owned orchestration. The `chat_ops/` module owns CHAT extraction, injection, validation, ASR post-processing, and DP alignment that is batchalign-specific. |
| `crates/batchalign/src/commands/` | (submodule) released-command definitions, author-facing constructors, and the command catalog |
| `crates/batchalign-types/` | Domain newtypes, worker IPC types (V2), shared between the runtime crate and the PyO3 bridge |
| `crates/batchalign-pyo3/` | PyO3 bridge crate (`batchalign_core`); workspace member, slim dep tree (`batchalign-types` + `talkbank-transform` + pyo3/numpy/serde/tracing) |
| `crates/talkbank-{model,parser,transform,clan,...}` | CHAT data model, parser, pipelines, CLAN tools, shared across the workspace; `batchalign` depends on the first three by workspace path |

## Common Developer Commands

```bash
cargo check --workspace
cargo test --workspace
cargo check --manifest-path crates/batchalign-pyo3/Cargo.toml    # PyO3 crate (separate)
cargo test --manifest-path crates/batchalign-pyo3/Cargo.toml
```

## CLI Command Dispatch (Single Source of Truth)

`batchalign::cli::run_command()` in
`crates/batchalign/src/cli/mod.rs:251` is the
**single canonical command router**. The standalone binary (`main.rs`) calls it.
The installed `batchalign3` console command is a tiny Python wrapper
(`batchalign/_cli.py`) that finds and execs the standalone binary, either
packaged in the wheel at `batchalign/_bin/batchalign3`, or from
`target/debug/batchalign3` in a source checkout.

```rust,ignore
main.rs            → batchalign::cli::run_command(cli)
batchalign/_cli.py → os.execv(batchalign/_bin/batchalign3)  [installed]
                   → os.execv(target/debug/batchalign3)      [dev checkout]
```

`main.rs` and `batchalign/_cli.py` are thin wrappers.
No command-specific logic lives in either of them.

The CLI layer now exposes two contributor-facing named seams:

- `ReleasedCommand` in
  `crates/batchalign-types/src/domain.rs:36` is the closed released
  command vocabulary for contributor-facing Rust code. Parse external
  strings into this enum as early as possible; keep the old
  string-backed `CommandName` only at wire/storage boundaries.
- `CommandProfile` in
  `crates/batchalign/src/cli/args/mod.rs:148` keeps the command
  identity, language, file extensions, and speaker count together as
  a typed profile instead of a positional tuple.
- `DispatchRequest` in
  `crates/batchalign/src/cli/dispatch/mod.rs:42` carries the typed
  command profile, I/O settings, and runtime flags into the
  dispatcher as one named boundary object.

The dispatcher also consults
`batchalign::released_command_uses_local_audio()` and the shared released
command catalog to decide whether a requested command uses the shared-filesystem
audio path under an explicit `--server` submission or can use ordinary
content-mode submission.

On the app side, the current execution split is now:

- `ExecutionEngine`: shared command execution core
- `ServerExecutionHost`: queue/store/server-owned lifecycle behavior
- `DirectHost` / `DirectExecutionHost`: inline local execution without queueing
  or registry discovery
- `ServerBackend` / `LocalServerBackend`, route-facing server control-plane
  seam over persisted jobs, orchestration, event subscription, traces, and
  runtime shutdown
- `prepare_workers*()` vs `prepare_direct_workers()`: explicit separation
  between server worker bootstrap and direct local worker bootstrap

### Align / FA host flow

The part that was easiest to misunderstand during the recent `align` emergency
was **where forced alignment actually runs**.

`align` now has two honest host paths:

- **Direct mode** (no `--server`) does **not** start Axum, an HTTP server, a
  queue, or registry discovery. The CLI prepares local workers and runs the job
  through `DirectHost`.
- **Explicit server mode** (`--server URL`) submits a shared-filesystem
  `paths_mode` job. The server must be able to read the submitted source paths
  and write the requested output paths on the execution host.

Both paths converge on the same FA runner code in
`crates/batchalign/src/runner/dispatch/fa_pipeline.rs`.

```mermaid
flowchart TD
    cli["batchalign3 align"]
    route{"--server?"}
    prep["prepare_paths_submission()\nsource_paths + output_paths\nmedia_mapping + media_subdir"]

    direct["dispatch_direct_mode()"]
    direct_workers["prepare_direct_workers()"]
    direct_host["DirectHost::submit_submission()\nDirectHost::run_job()"]

    server["dispatch_single_server()"]
    post["POST /jobs (paths_mode)"]
    backend["ServerBackend / queue / orchestrator"]

    runner["process_one_fa_file()\nfa_pipeline.rs"]
    fa["media resolution\n→ ensure_wav\n→ %wor / incremental reuse\n→ optional UTR\n→ FA transport\n→ traces + output"]

    cli --> route
    route -->|no| direct
    route -->|yes| server
    direct --> prep --> direct_workers --> direct_host --> runner
    server --> prep --> post --> backend --> runner
    runner --> fa
```

That shared convergence is deliberate: direct mode and server mode should differ
in **host/orchestration** behavior, not in the actual forced-alignment logic.

When the CLI is polling or writing file results, `FileErrorDetail`
in `crates/batchalign/src/cli/dispatch/helpers.rs:24` keeps
file-scoped failures as a named record instead of spreading
filename/message pairs through the progress code.

The command-specific logic starts from one declaration in
`crates/batchalign/src/recipe_runner/catalog.rs`: a `CatalogEntry` per released
command, naming its stage recipe and stating its family, planner, execution
mode, capability kind, io profile, dispatch kind, worker tasks and output
policy. The shared runner dispatches from that entry, so nothing per-command
needs to import store/queue/host plumbing to reach an existing family executor.

There is no per-command authoring layer any more. A `commands/` module with one
module per command, six `declare_*_command!` macros and a `CommandDefinition`
type existed until 2026-07-28, when removing an `#![allow(dead_code)]` proved it
produced values nothing read; the compatibility views over the catalog followed
on 2026-07-29.

That is an intentional contributor contract:

- new commands should be authored direct-first and laptop-friendly
- command authors should not need to understand server internals
- command authors should usually only pick a family helper, not hand-author
  scheduling/runtime metadata
- server mode may opt into a different host/backend, but it should reuse the
  same generated command definition

`command_family.rs` keeps the small family enum used by command metadata,
`text_batch.rs` keeps reusable text-family helpers, and `runner/dispatch/`
keeps shared execution helpers. `crates/batchalign/src/runner/` should stay
focused on job lifecycle, queueing, and policy rather than becoming a second
authoring surface for commands.
HTTP routes, SSE, and WebSocket handlers should prefer `ServerBackend` over
reaching through `AppState` to raw `JobStore`, queue, or runtime internals.

One dependency-graph cleanup already landed here: the standalone binary's OTLP
telemetry stack and update-check helper are now gated behind the
`batchalign` crate's `binary-entry` feature. The PyO3 `cli_entry` path
still shares `run_command()`, but it no longer drags those binary-only
dependencies into the extension build.
The embedded CLI bootstrap path now also lives in `batchalign`
(`run_embedded_cli_from_argv()`), so `pyo3` no longer owns its own `clap`
parsing or Tokio runtime setup.

Under `uv tool install`, the Python wrapper now also primes
`BATCHALIGN_SELF_EXE` before it `exec`s the packaged Rust binary. That gives
Rust server/daemon re-exec paths one explicit source of truth for "which binary
am I?" instead of forcing them to guess from `current_exe()` or PATH.

## Server control-plane replacement guidance

The current recommendation is:

- **Do not replace Axum just to say the server is off the shelf.** The HTTP
  shell is not the main pain source.
- **Do not move the primary CLI or server into Python.** That would re-center
  orchestration around the worker/package layer instead of the typed Rust
  control plane.
- **If we replace something, replace the control-plane backend** behind
  `ServerBackend`: queued-job claims, retries, recovery, runtime supervision,
  persisted progress, and cancellation.

That keeps the architectural split honest:

- `ExecutionEngine` remains the canonical command execution core.
- `DirectHost` remains the BA2-style local/default execution path.
- `ServerBackend` becomes the place where embedded-vs-durable server behavior
  can differ without teaching commands about server internals.

For local and single-host installs, the in-process local backend is the
implemented server architecture.

Current backend contract:

- `ExecutionEngine` owns canonical command execution only.
- `DirectHost` remains the direct/local execution path.
- `ServerBackend` owns app-facing job submission, inspection, cancellation,
  traces, event subscription, and runtime shutdown.
- `LocalServerBackend` owns queued-job orchestration, restart/recovery,
  runtime supervision, and store-backed lifecycle behavior.
- The shared runner may report failures and progress, but it does not own
  higher-level server policy.

The important current design point is that direct mode stays simple while the
server control plane remains local, explicit, and fully owned by the Rust
runtime in this repository.

Validated so far:

```bash
cargo check -p batchalign -p batchalign
cargo test -p batchalign --lib -q
cargo test -p batchalign --test contract_suite json_compat:: -q
cargo test -p batchalign --lib -q
batchalign3 serve start --foreground --test-echo
batchalign3 jobs --server http://127.0.0.1:8111 <JOB_ID>
curl -X POST http://127.0.0.1:8111/jobs/<JOB_ID>/restart
curl -X DELETE http://127.0.0.1:8111/jobs/<JOB_ID>
```

Important validation caveat: existing e2e coverage already treats text-only
infer-task commands like `morphotag` as expected failures under `--test-echo`.
Use `--test-echo` to validate control-plane behavior, not infer-task success.

### First-class debuggability

Direct mode and server mode should share the **shape of the debugging handles**
they expose, but they should **not** be forced to share one live control-plane
implementation just for symmetry.

What should be shared:

- stable `job_id`
- stable staging/artifact directory
- bug-report identifiers / files
- optional persisted trace artifact file

What should remain mode-specific:

- HTTP polling / WebSocket / dashboard transport
- queue persistence and recovery model
- live event fan-out
- server-only operational state

Direct mode now persists a machine-readable `debug-artifacts.json` file inside
the per-job staging directory and exports `debug-traces.json` when traces were
captured. The CLI also prints the direct job ID and artifact directory up front,
so a human or LLM agent can later inspect a failed local run by job ID instead
of relying on transient terminal output alone.

Opt-in telemetry is still worth considering, but only as an **additive**
debugging aid for fleet/server deployments. It should not replace inspectable
local artifacts. The first-class debugging path must remain: "here is the job
ID and here are the files to inspect."

For day-to-day command work, prefer the command layer first:

1. add or extend `crates/batchalign/src/commands/<name>.rs`
2. choose the existing runner family it should reuse
3. keep the CLI argument plumbing thin
4. let runner/dispatch handle lifecycle and resource policy, not semantics

## Adding a New CLI Command

When adding a new processing command (e.g., `batchalign3 foo`), these files
must be updated:

### 1. CLI argument definition

**`crates/batchalign/src/cli/args/mod.rs`**: Add
`Commands::Foo(FooArgs)` variant to the `Commands` enum.

**`crates/batchalign/src/cli/args/commands.rs`**: Define `FooArgs`
struct with clap attributes. Include `CommonOpts` if the command
processes files.

### 2. CLI dispatch

**`crates/batchalign/src/cli/mod.rs`**: Add the match arm in
`run_command()` (defined at `cli/mod.rs:251`). For processing
commands, this typically falls through to the `cmd =>` wildcard arm
that calls `cli::dispatch::dispatch()`. For utility commands (like
`serve`, `jobs`, `models`), add an explicit arm.

### 3. Typed command options

**`crates/batchalign/src/types/options.rs`**: Add
`CommandOptions::Foo { ... }` variant to the serde-tagged enum. This is the
wire format between CLI and server.

**`crates/batchalign/src/cli/args/options.rs`**: Add the builder in
`build_typed_options()` that converts `FooArgs` → `CommandOptions::Foo`.

### 4. Server-side task routing and capability gate

**`crates/batchalign/src/recipe_runner/recipes.rs`**: add the stage recipe.

**`crates/batchalign/src/recipe_runner/catalog.rs`**: declare the
`CatalogEntry`. That is the whole registration.

`crates/batchalign/src/runner/policy.rs` answers
`command_requires_chat_infer()` straight off that entry. It is total: a
`ReleasedCommand` always has an entry (pinned by
`recipe_runner::catalog::tests::every_released_command_has_a_spec`), so it
returns no `Option` for callers to unwrap.

Availability reads two different facts at two moments:

1. `command_supported()` in `crates/batchalign/src/capability.rs`: the
   worker's admitted report (admitted once, in
   `WorkerPool::record_capabilities()`) advertises the entry's
   `capabilities.primary_infer_task`. This alone decides what `/health`
   advertises and which submissions are accepted, from any report, including
   one a lazily loading worker gave before loading anything. It is the only
   availability rule in that module, beside `WorkerCapabilitySnapshot`.
2. At dispatch (`runner/routing.rs`),
   `WorkerPool::ensure_command_capabilities()` loads the command's task on the
   selected worker (`ensure_task`) and returns `LoadedCapabilities`: the task
   it loaded and the report taken after that load. Routing runs step 1 again
   against that report. Then the forced-alignment dispatch arm, and only that
   arm, reads the FA engine with `FaCacheNamespace::from_loaded`
   (`engine_reports.rs`), because every FA cache row is namespaced by it. It
   refuses a report taken after a different task loaded (`LoadedAnotherTask`),
   an engine still unnamed after the load (`UnreportedAfterLoad`) and a worker
   that does not support FA (`NotSupported`). No catalog field declares this
   requirement; the arm that reads the namespace is the only place that says
   so.

The critical implementation rule is that **startup capability state is not
authoritative for execution**. The current server intentionally allows an
optimistic cold-start snapshot so app creation does not have to spawn a
dedicated probe worker. Execution then resolves a **live capability snapshot**
before it trusts infer-task gating:

- `WorkerCapabilitySnapshot::resolve()` in `crates/batchalign/src/capability.rs`
  prefers the pool's first admitted report over the startup view of every
  released command
- the router (`runner/routing.rs`) forces a command-appropriate live probe
  through `WorkerPool::ensure_command_capabilities()`, which returns the
  selected key's admitted report, before applying the availability rule
- `WorkerPool::discover_from_registry()` admits the report probed from a
  healthy TCP registry daemon under that daemon's worker key, so
  registry-only deployments do not start with `infer_tasks = []`

This split is deliberate. It avoids the old failure mode where lazy startup said
"we will discover capabilities later" but the first real `morphotag` or
`compare` job was still judged by an empty startup snapshot.

One implementation detail matters here: sequential TCP daemons accept one
connection at a time. Registry discovery therefore probes capabilities on the
same `TcpWorkerHandle` it already opened for the discovery health check, instead
of trying to race a second connection.

A checked-out TCP handle is a `TcpCheckout` (`worker/pool/dispatch.rs`), which
owns the handle and its group slot together. When the exchange ends, the
handle goes back to its group unless `WorkerError::worker_after_failure()`
(`worker/error.rs`) answers `Retire`. Only `WorkerResponse`, `Bootstrap`,
`MemoryGuard`, `NoWorker` and `PoolShuttingDown` leave the worker reusable;
every other error (a dead process, a protocol or I/O failure,
`CapabilitiesRefused`, and the rest) retires the handle. Retiring drops the
handle, which closes the connection, and releases the slot; a checkout dropped
without finishing (a cancelled exchange) does the same, because a half-read
stream could hand a later exchange a stale reply. The daemon itself is left
running and is adopted again by the next registry sweep.

A worker whose capability report was refused is never pooled, on any path: a
spawned worker is shut down, a registry GPU daemon is disconnected, a TCP
handle is dropped, a checked-out worker is taken out of its group, and
registry discovery (`worker/pool/discovery.rs`) does not integrate it.

The registry layer now also carries explicit daemon ownership metadata:

- `external` daemons are preserved on routine shutdown
- `server_owned` daemons are tagged with `server_instance_id` and `server_pid`
- shutdown only retires daemons owned by the current server instance
- discovery skips foreign live owners and reaps stale foreign owned daemons
- discovery refuses, without reaping it or removing its entry, a daemon whose
  entry names another build or no build. Entries carry `build_identity`, which
  the daemon reads from `BATCHALIGN_BUILD_IDENTITY`, set by the server's daemon
  spawner and by `batchalign3 worker start`. The logged refusal names the
  remedy: `batchalign3 worker stop`, then `batchalign3 worker start` with this
  build. Each sweep's refusals are listed in `/health` under
  `refused_registry_workers` (see
  [Observability](../architecture/observability.md#refused-registry-daemons))

That ownership model is the durable fix for the old orphan-daemon/kill-all
whackamole around server-spawned TCP workers.

On the Python side, you must also add the `InferTask` to `_INFER_TASK_PROBES` in
`batchalign/worker/_handlers.py`. See
[Adding Inference Providers](../developer/adding-engines.md#4-wire-dispatch-and-capability-advertisement)
for details.

### 5. Server-side dispatch shape

`crates/batchalign/src/runner/routing.rs` matches the catalog's
`RunnerDispatchKind` exhaustively. Batched text commands (`morphotag`,
`utseg`, `translate`, `coref`, `compare`) are routed by name in
`dispatch_batched_text_command()` to the recipe-owned modules under
`crates/batchalign/src/execution/` (`morphotag/`, `utseg.rs`, `translate.rs`,
`coref.rs`, and `kernel.rs` for compare); a new text command needs an arm
there. Every other kind goes to its module under
`crates/batchalign/src/runner/dispatch/`:
- `fa_pipeline.rs`: `dispatch_fa_infer()` for per-file forced alignment
- `transcribe_pipeline.rs`: `dispatch_transcribe_infer()` for audio-to-CHAT generation
- `benchmark_pipeline.rs`: `dispatch_benchmark_infer()` for transcribe + compare composition
- `media_analysis_v2.rs`: `dispatch_media_analysis_v2()` for opensmile/avqi/diarize
- `speaker_identity_pipeline.rs`: `dispatch_speaker_identity()` for speaker identification

**Recipe-driven execution (new model):** Compare has been migrated from
`runner/dispatch/` to the recipe-driven `execution/` kernel. New commands
should prefer the `execution/` model when they have multi-stage workflows.
See `crates/batchalign/src/execution/` for the `StageExecutor` trait
and `crates/batchalign/src/planning/` for `build_job_plan()`.

### 6. Orchestrator module

**`crates/batchalign/src/commands/foo.rs`**: The command-owned wrapper that
owns the command's semantic shape, shared plan selection, and materialization
policy.

**`crates/batchalign/src/foo.rs`** or `runner/dispatch/*`: Keep shared
algorithmic code and reusable runner families here when it improves clarity, but
do not make them the only obvious home of the released command.

For batch text workflows, prefer the named wrappers in
`crates/batchalign/src/text_batch.rs` over raw tuples:

- `TextBatchFileInput` keeps one file name and one owned CHAT payload together.
- `TextBatchFileResults` keeps the per-file outcome shape explicit.
- `TextWorkflowFileError` keeps file-scoped failure details separate from file
  identity instead of returning `String` error messages.

### 7. Worker support

**`batchalign/worker/_model_loading/`**: Register the dynamic batch-infer
handler for `InferTask.FOO` during worker bootstrap if the task needs loaded
runtime state or engine-specific wiring.

**`batchalign/worker/_infer.py`**: Only update this file if the task is a
pure static route that does not need bootstrap-installed runtime wiring.

**`batchalign/inference/foo.py`**: The Python inference module (pure model
invocation, no CHAT awareness).

### 8. CHAT operations (if needed)

**`crates/batchalign/src/foo.rs`**: Payload collection, cache key
computation, result injection functions used by the orchestrator.

## OpenAPI Workflow

```bash
# Generate OpenAPI schema
cargo run -q -p batchalign -- openapi --output openapi.json

# Verify schema is up to date (CI gate)
cargo run -q -p batchalign -- openapi --check --output openapi.json
```

## Relationship to the PyO3 Layer

The CLI/server workspace and the PyO3 extension are separate build targets:

- **Root workspace** (`crates/`): operational control plane (CLI + server)
- **`crates/batchalign-pyo3/`**: Python extension module (`batchalign_core`)

Both share CHAT operations through `batchalign`. The PyO3 crate
also depends on `batchalign` (for `run_command()`) and `batchalign`
(for OpenAPI types), but it now does so with `default-features = false` so the
extension path does not compile the standalone binary's OTLP stack.

See [Building & Development](building.md) for the recommended fast
local loop (one `cargo build -p batchalign` for the source-checkout
fallback; `uv run maturin develop -m crates/batchalign-pyo3/Cargo.toml
-F pyo3/extension-module` or the
`make batchalign-build-wheel` → `make batchalign-python-prepare`
chain when you need the PyO3 extension installed into the dev env).
