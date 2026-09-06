# Testing

**Status:** Current
**Last updated:** 2026-09-06 00:42 EDT

## Philosophy

The test suite is split into tiers by design. The following diagram shows
the tiers, their resource requirements, and how to invoke each.

The manually dispatched Coverage job runs Rust coverage once over the workspace,
including `batchalign-pyo3`, and excludes the experimental desktop shell:

```bash
cargo llvm-cov --workspace --exclude batchalign-dashboard-desktop --locked --lcov --output-path lcov-rust-workspace.info
```

The desktop shell needs GTK/WebKit on Linux and is outside the headless CI
package set. This exclusion does not remove PyO3 coverage; the former separate
PyO3 step repeated tests that the workspace command already selects. Coverage
remains an explicit measurement job, not an inner-loop test command.
Its native compile restores the dashboard artifact produced by the same
workflow before compiling the embedded server assets. It installs and probes
`protoc`, `ffmpeg` and `ffprobe` before the instrumented build, because coverage
also exercises real media boundaries. PyO3 uses the same virtual-environment
interpreter as the installed wheel.
The instrumented Rust coverage job sets `RUST_TEST_THREADS=1` on its small
runner. Independent CLI tests otherwise reserve host memory as separate
servers: a 16 GB runner reached 8.25 GB of pending reservations with 7 GB
available, blocking another worker for 120 seconds and timing out other jobs.
This is an execution budget for coverage, not a disabled memory guard or a
skipped test. Explicit concurrency inside a test still runs. Ordinary local
and push tests retain their existing scheduling. Reproduce the affected suite
with `cargo test -p batchalign --test cli_integration_suite --locked -- --test-threads=1`.

Coverage caches `target/llvm-cov-target` under its own shared key and saves
compiled dependencies after failed tests. The ordinary build's full cache hit
cannot stand in for those instrumented artifacts; sharing that immutable key
previously forced the dependency graph to rebuild on each coverage retry.

The memory-tier architecture test scans Rust files in process and parses the
embedded runtime TOML. It requires no `rg` subprocess or workspace-root search;
unreadable source files fail the check. This keeps ordinary test and coverage
environments equivalent without adding a search executable to every runner.

Processing-command subprocess tests own a `CliHarness`, which seeds an isolated
HOME with setup configuration. They must not inherit the developer's setup:
coverage runners correctly start without it. The detached-server worker test
checks the running child's actual `--workers` arguments, then checks that a
completed job respects that ceiling. Granted workers may be lower because CPU
and memory admission remain active. The daemon binds an OS-selected port and
publishes its handshake, so a startup failure is a failure rather than a skipped
port-collision test. Its echo daemon owns a separate memory ledger and declares
a minimal positive headroom floor, so unrelated live-model tests do not consume
its synthetic reservations. Zero is not used: YAML maps that legacy value back
to automatic sizing. An owned daemon session stops the process on both normal
completion and unwinding; failures include its log.

Both the Python wheel job and standalone PyO3 build cache the root Cargo
workspace's `target` directory. The extension lives at
`crates/batchalign-pyo3`; the former `pyo3` workspace path no longer exists.
The cache action's default root mapping follows the actual Cargo workspace
without maintaining a second, stale crate layout.

```mermaid
flowchart TD
    fast["Tier 1: Fast Tests\n(make test / cargo test)\nUnit + protocol + test-echo integration\nNo models, no GPU\n~5s, safe, fully parallel"]
    ml["Tier 2: ML Golden Tests\n(make batchalign-test-ml-golden)\nReal Whisper + Stanza + pyannote\nSkips are FAILURES, not passes\nSerialized (profile test-threads=1)\n~5min, 8-12 GB peak RAM"]
    pygolden["Tier 3: Python Golden\n(uv run pytest -m golden)\nbatchalign_core extension\n~10-30s, 1-2 GB"]

    fast -->|"routine dev loop\n(every edit)"| safe(["Safe on any machine"])
    ml -->|"opt-in only\n(pre-release, inference changes)"| danger(["Serialized, never\nrun with bare cargo test"])
    pygolden -->|"opt-in only\n(PyO3 changes)"| safe
```

1. **Fast tests**: unit tests, protocol tests, test-echo integration tests.
   No ML models, no GPU, no multi-GB processes. These run in seconds, fully
   parallel, on every edit. This is the inner development loop. It must stay
   fast and safe, a `cargo test` should never crash your machine.

2. **ML tests**: golden snapshots, audio transcription, parity checks,
   profile verification. These spawn real Python workers that load Whisper,
   Stanza, pyannote, etc. Each worker consumes 2-5 GB RAM. They are slow,
   expensive, and dangerous on developer machines.

**ML tests are excluded by default.** You must opt in explicitly, and only
when you have a reason: a change to worker dispatch, a new language, an
inference module edit, a pre-release check. Never as part of routine
edit-compile-test.

This mirrors the Python side, where `uv run pytest` excludes `golden`,
`slow`, and `integration` markers by default.

## Rust test executable topology

The ordinary Rust integration tests are compiled into three shared targets:

- `contract_suite` for wire formats, manifests, workflow helpers, and other
  fast contracts;
- `cli_integration_suite` for CLI and server behavior using the shared test
  server fixture;
- `worker_integration_suite` for worker protocol, routing, lifecycle, and pool
  behavior.

Four targets keep their own processes because isolation is part of their
contract: `relative_output_dir` mutates the process working directory,
`stress` controls concurrent server pressure, `turmoil_net` owns a simulated
network, and `gpu_concurrent_dispatch` performs parent-process worker cleanup.
The feature-gated `ml_golden` target remains a separate, opt-in executable.

This layout was measured on macOS on 2026-09-05. Before consolidation, a warm
`cargo test -p batchalign --tests -- --list` spent 15.87 seconds merely
starting 34 executables, while a no-run build was already warm in 0.42 seconds.
The consolidated layout starts nine ordinary executables, including the two
unit-test targets, and the same listing probe takes 6.61 seconds. The assertions
were retained; the removed cost was repeated linking and process startup.

The workspace continues to use plain `cargo test`. `cargo-nextest` is not
installed because it must still enumerate every executable and its earlier
eager process burst repeatedly wedged macOS `syspolicyd`. Reconsider it only
through an isolated benchmark of this consolidated topology, with constrained
process concurrency and clean plus warm measurements. Runner adoption requires
an improvement beyond the cost already removed at the source.

That benchmark was run with a temporary cargo-nextest 0.9.143 binary and four
test workers. The 52-test `contract_suite` took 3.40 seconds under nextest,
which launches one process per test, versus 2.56 seconds as one plain-Cargo
test executable. The full default suite has 2,433 runnable tests, so nextest
would exchange nine ordinary process starts for thousands. It remains rejected
for the default developer loop on both performance and macOS process-assessment
grounds.

## Fast Contributor Loop

For command-workflow edits, the shortest useful loop is usually:

```bash
cargo xtask affected-rust packages
make batchalign-python-prepare
cargo build -p batchalign
cargo test -p batchalign --test contract_suite workflow_helpers::
cargo test -p batchalign --test cli_integration_suite cli::
uv run batchalign3 --help
```

Use the workflow-layer tests when you are changing:

- command semantics
- compare / benchmark behavior
- materializers and typed intermediate bundles
- workflow-family dispatch or composition

Keep the broader ML tests for runtime changes that actually touch workers,
models, or cache behavior.

If you only changed docs or workflow metadata, start with
`cargo xtask affected-rust packages` and the narrow CLI/help checks before
running anything expensive.

### Why this matters

Kernel OOM panics have been caused by ML test binaries spawning
concurrent Whisper workers during `cargo test`. Each golden test
binary was a separate process that started its own server with its own
worker pool. Running them in parallel exhausted machine memory.

A separate kernel OOM panic was caused by running Python
`@pytest.mark.golden` tests repeatedly with the default `-n 3` from
`pytest.ini`. Each xdist worker loaded its own Stanza model instance
(~500 MB). Over multiple invocations combined with cargo builds and a
local batchalign daemon, cumulative memory pressure exceeded the
machine's RAM. This led to the three-layer Python-side OOM guard in
`conftest.py`.

See `docs/postmortems/` for incident details.

### Implemented solution: single binary

All ML tests are consolidated into one binary (`ml_golden`). One binary =
one process = one `LazyLock` = one `PreparedWorkers` = one set of loaded
models. Peak memory is ~8-12 GB (one pool) instead of 7x that.

The `LiveServerSession` fixture within the binary is well-designed:
- **One `PreparedWorkers` backend** shared across all 70 tests
- **Fresh HTTP server per session** (new port, new jobs dir, new SQLite)
- **Semaphore-gated sessions** so tests don't collide on control-plane state
- **Warm model cache** across tests, only the first test pays cold-start

### Defense-in-depth layers

These remain as additional safety nets:

| Layer | What | Catches |
|-------|------|---------|
| **`ml-golden` cargo feature** | `ml_golden` carries `required-features`, so a plain `cargo test` cannot build it | Routine dev runs |
| **`make batchalign-test-ml-golden`** | ML tests serialized via `--test-threads=1` | Explicit opt-in |
| **Global worker cap** | `max_total_workers` (RAM / 6GB, clamped to `[2, 32]`) | Multi-key pool explosion |
| **`WorkerPool::Drop`** | Kills idle workers when pool is dropped | Test cleanup on panic/exit |
| **PID file reaper** | `~/.batchalign3/worker-pids/` scanned on startup | Orphans from crashed servers |
| **pytest OOM guard (configure)** | Forces `-n 0` when `-m golden` on < 128 GB machine | Standard golden invocation |
| **pytest OOM guard (collection)** | Aborts if golden tests collected with `-n > 0` on < 128 GB | Overridden addopts |
| **pytest OOM guard (fixture)** | Per-test `_guard_golden_oom` autouse fixture fails in xdist workers on < 128 GB | Belt-and-suspenders; cannot be bypassed |
| **Claude Code guard hooks** | Block workspace `cargo test` under memory pressure, and any cargo run concurrent with another in the same workspace | AI assistant sessions |

## Quick reference

```bash
# Fast tests only (default, safe, parallel, no models)
cargo test --workspace
make test

# ML tests only (serialized, one at a time)
make batchalign-test-ml-golden

# Specific ML test (filter by submodule name)
cargo test -p batchalign --features ml-golden --test ml_golden golden:: -- --test-threads=1

# Everything (fast + ML)
make batchalign-test-ml-golden

# Python (fast only by default)
uv run pytest

# Python golden/integration
uv run pytest -m golden
uv run pytest -m integration
```

## Nextest configuration

### The ML golden suite is substantially broken (2026-07-28)

First honest run of the whole suite after giving it an entry point:
**104 passed, 86 failed, 191 total, 625 s.**

None of the 86 is the skip panic described below; every test acquired a live
session and then failed on its own merits. Observed causes, from the run log
(archived at `ml-golden-baseline-2026-07-28.log`):

- Worker death mid-job: `worker process exited unexpectedly (exit code: None)`,
  `GPU worker reader loop exited, worker process is dead`.
- Runtime teardown racing the job: `A Tokio 1.x context was found, but it is
  being shutdown`.
- Jobs returning `Failed` where `Completed` was asserted (~30).
- HTTP 400 on content-job submission (~10).
- Snapshot drift on the `compare` and `coref` goldens.
- A rejection-message assertion now stale: the test expects an
  unsupported-language message, but morphotag now rejects job-level `--lang`
  outright (the 2026-05-03 incident), so the message no longer matches.

**Do not read this as a regression introduced on 2026-07-28.** The suite had no
Makefile or CI entry point and was reachable only through nextest's
`--profile ml`, retired with nextest itself, so there is no recent green
baseline to regress from. The failures are accumulated rot that nothing was
positioned to notice.

Treat the numbers above as the BASELINE to drive down, not as a gate. Until it
is green, `make batchalign-test-ml-golden` is a diagnostic, and adding it to
`make verify` would only train people to ignore a red gate.

### ML golden tests fail rather than skip

`require_direct_session_warmed` in the ml_golden suite **panics** when a live
session cannot be acquired. It used to return `None`, and every call site did
`else { return; }`, so a test that never executed reported `ok`.

That is not hypothetical. On 2026-07-28 two newly written Italian golden tests
reported `ok` in 7.23 s having produced no output; only replacing an assertion
with a deliberate lie and watching it still pass would have told them apart.
The suite had also had NO entry point in the Makefile or CI, reachable only
through nextest's `--profile ml`, so its silence went unnoticed after nextest
was retired.

Building with `--features ml-golden` is an explicit request to run these tests.
If the environment cannot host them (no Python worker, no model weights, no
credentials), do not run the suite; the feature gate exists so a plain
`cargo test` never reaches it.

nextest was removed on 2026-07-27 (it wedged macOS `syspolicyd` by
exec'ing every test binary up front to enumerate tests). The ML exclusion
that used to live in `.config/nextest.toml` as a `default-filter` now lives
in the code as `required-features = ["ml-golden"]`, so correctness no longer
depends on which runner you use.

**Default profile:** applies a `default-filter` that excludes all ML test
binaries. `cargo test` runs only fast tests, because `ml_golden` requires the
`ml-golden` feature to build at all. This is the safe
default.

**ML profile (`--profile ml`):** the profile's `default-filter` selects
only `binary(ml_golden)`, and the profile sets `test-threads = 1` so the
suite runs serially, preventing concurrent model loading and the OOMs
that follow.

**Override the default filter for one run:**
```bash
cargo test -p batchalign --features ml-golden --test ml_golden -- --test-threads=1
```

All ML tests live in one binary (`ml_golden`) with submodules:

| Submodule | What | Models |
|-----------|------|--------|
| `golden` | Text NLP golden snapshots | Stanza |
| `golden_audio` | Audio transcription/alignment | Whisper, Wave2Vec, pyannote |
| `golden_parity` | Batchalign2 output parity | Stanza |
| `live_server_fixture` | Full server with live workers | Mixed |
| `profile_verification` | Worker pool profile grouping | Wave2Vec, Stanza |
| `option_receipt` | Option propagation differential tests | Stanza, Wave2Vec |
| `error_paths` | Graceful failure under live server | Mixed |

## Test categories

| Category | Tool | Command | Models | Runtime | Default |
|----------|------|---------|--------|---------|---------|
| Rust unit tests | cargo | `cargo test --workspace` | None | ~5s | Yes |
| PyO3 unit tests | cargo | `cargo test --manifest-path crates/batchalign-pyo3/Cargo.toml` | None | ~3s | Yes |
| Python unit tests | pytest | `uv run pytest` | None | ~2s | Yes |
| Worker protocol | cargo | `cargo test -p batchalign --test worker_integration_suite worker_protocol_matrix::` | None (test-echo) | ~5s | Yes |
| Server integration | cargo | `cargo test -p batchalign --test cli_integration_suite integration::` | None (test-echo) | ~5s | Yes |
| Network fault (turmoil) | cargo | `cargo test --test turmoil_net` | None | <1s | Yes |
| Workflow helpers | cargo | `cargo test -p batchalign --test contract_suite workflow_helpers::` | None | ~2s | Yes |
| JSON compat | cargo | `cargo test -p batchalign --test contract_suite json_compat::` | None | ~1s | Yes |
| ML tests (all) | cargo | `make batchalign-test-ml-golden` | Mixed | ~5min | **No** |
| Python golden | pytest | `uv run pytest -m golden` | batchalign_core | ~10s | **No** |
| Python integration | pytest | `uv run pytest -m integration` | Worker | ~5s | **No** |
| Cantonese ASR engines | pytest | `uv run pytest batchalign/tests/languages/cantonese/` | FunASR+ | ~2min | **No** |

## When to run ML tests

Run ML tests based on what changed, not as a habit:

| What you changed | Run |
|-----------------|-----|
| Rust unit logic (parser, DP, postprocess) | Fast tests only |
| Workflow-family modules or compare/benchmark materializers | `workflow_helpers` + focused CLI tests |
| Python inference module | `--profile ml` |
| Worker protocol or IPC types | `worker_protocol_matrix` (fast) + `--profile ml` |
| Worker pool, dispatch, or lifecycle | `--profile ml` |
| FA pipeline or UTR | `--profile ml` |
| Morphosyntax injection or retokenization | `--profile ml` |
| Pre-release or large refactor | Full `--profile ml` |
| Adding a new language | `--profile ml` |

## Python tests

```bash
uv run pytest                                           # Fast only
uv run pytest -m golden -v                              # Golden snapshots
uv run pytest -m integration -v                         # Integration
uv run pytest -m "golden or integration" -v             # Both
uv run pytest batchalign/tests/test_batch_infer_dispatch.py -v  # Specific file
```

If you changed `crates/batchalign-pyo3/` or shared Rust crates that feed `batchalign_core`,
rebuild the extension before running Python tests that import it. The
maturin build backend declared in `pyproject.toml` (`build-backend =
"maturin"`, `[tool.maturin]` block) means `uv run <anything>` does an
incremental rebuild on demand. For a clean wheel install of the freshly
built extension into the dev environment:

```bash
make batchalign-python-prepare
```

### Test doubles

Prefer explicit fake seams over `monkeypatch` when touching production code.
If a test needs to replace runtime behavior, the first question should be
whether the production boundary wants a typed injected dependency instead.

### Worker protocol V2 drift suite

```bash
uv run pytest batchalign/tests/test_worker_protocol_v2_types.py -q
uv run pytest batchalign/tests/test_worker_protocol_v2_artifacts.py -q
uv run pytest batchalign/tests/test_worker_fa_v2.py -q
cargo test -p batchalign --test contract_suite worker_protocol_v2_compat::
cargo test -p batchalign -E 'test(fa_result_v2)'
cargo test -p batchalign --test worker_integration_suite worker_v2_fa_roundtrip::
```

These tests read fixture files under `tests/fixtures/worker_protocol_v2/`
so the Rust and Python schema models stay aligned.

### Cross-language contract tests

Several test pairs look redundant but exist **intentionally**: Rust and Python
must independently verify the shared wire format. If only one side is tested, a
serialization change in the other language could silently break IPC.

| Python test | Rust counterpart | What they verify |
|-------------|------------------|-----------------|
| `test_ipc_type_conformance.py` | `scripts/check_ipc_type_drift.sh` (CI gate) | Schema field parity between Rust and Python models |
| `test_worker_ipc.py` | `worker_protocol_v2_compat.rs` | JSON roundtrip through both language's serializers |
| `test_worker_protocol_v2_types.py` | `worker_protocol_matrix.rs` | V2 protocol envelope parsing on both sides |

**Do not consolidate these pairs.** A passing Rust test does not prove the Python
side deserializes correctly, and vice versa.

## Rust tests

```bash
# PyO3 extension
cargo test --manifest-path crates/batchalign-pyo3/Cargo.toml

# Root workspace (fast tests only)
cargo test --workspace

# Workflow layer
cargo test -p batchalign --test contract_suite workflow_helpers::

# Focused modules within the consolidated suites
cargo test -p batchalign --test cli_integration_suite cli::
cargo test -p batchalign --test cli_integration_suite e2e::
cargo test -p batchalign --test cli_integration_suite integration::
cargo test -p batchalign --test contract_suite json_compat::
```

### Profile verification tests

`ml_golden/profile_verification.rs` exercises the worker profile architecture
under real model inference. Unlike golden tests (which verify output
correctness), these tests verify resource usage:

- **GPU profile sharing**: multi-file align uses a single `SharedGpuWorker`
- **Stanza profile grouping**: morphotag and utseg share one Stanza worker
- **Label regression guard**: all worker keys use `profile:*` prefix

Run with `make batchalign-test-ml-golden`.

### ML test skip behavior

Model-gated tests use `require_live_server(InferTask::Xxx, "message")`:

1. Tries to acquire a `LiveServerSession` with a warm worker pool
2. Checks if the required InferTask is available (model installed)
3. Returns `None` (test silently skips) if models are unavailable

Python uses `@pytest.mark.skipif` or `pytest.skip()` for similar gating.

Even under `--profile ml`, tests skip gracefully if models are not
installed. You won't get false failures, just silent skips.

## Worker process safety

ML tests spawn Python worker subprocesses that load multi-GB models.
Several safeguards prevent runaway resource consumption:

**Global worker cap:** The `WorkerPool` enforces a hard ceiling on total
workers across all `(profile, lang, engine)` keys. The production
formula `ram_total_mb / 6GB`, clamped to `[2, 32]` (with a fallback
of 4 if `sysinfo` reports 0, e.g. macOS undercounts), lives in
`recommend_max_total_workers()` at
`crates/batchalign/src/host_facts/recommendations.rs:244`. The
runtime value is exposed via `EffectiveConfig::max_total_workers`
and consumed by `PoolConfig` (see comments at
`crates/batchalign/src/worker/pool/mod.rs:157`). Configurable via
`max_total_workers` in `server.yaml`.

**Pool Drop:** `WorkerPool` implements `Drop` to kill all idle workers
synchronously, even when tests exit without calling `pool.shutdown()`.

**PID file reaper:** Each spawned worker writes a PID file to
`~/.batchalign3/worker-pids/{pid}` recording its parent server PID. On
pool startup, stale files (dead workers) are cleaned up and orphans
(live workers whose parent server is dead) are killed via
SIGTERM → 2s wait → SIGKILL.

## Dashboard Playwright tests

```bash
cd frontend
npm run e2e:install
npm run test:e2e
```

If Chromium has not been installed:

```bash
cd frontend
npm run test:e2e:setup
```

## Type checking

```bash
uv run mypy                       # mypy only
make batchalign-typecheck-python  # mypy under the batchalign-* target group
make lint-affected                # affected-Rust clippy + affected Python mypy
```

## CI hygiene

Release-facing CI checks cover:

- CLI/package version sync (`make ci-local` + xtask `lint-ci-hygiene`)
- Stale legacy-term detection (xtask `lint-ci-hygiene`)
- Retired package/path checks (xtask `lint-ci-hygiene`)
- Command execution path integration coverage (focused tests under
  `crates/batchalign/tests/`)

```bash
cargo xtask lint-ci-hygiene
make ci-local
```

## Coverage

There is a coverage workflow in `.github/workflows/batchalign-python.yml` (manual
`workflow_dispatch`, not a release gate).

- Python: full inference adapter surface covered
- Remaining low-coverage areas: training, worker bootstrap, test helpers

```bash
# Python coverage (non-integration)
uv run --no-sync pytest -n0 --cov=batchalign --cov-report=term \
  --disable-pytest-warnings -m 'not integration' -q batchalign/tests

# Rust coverage
cargo llvm-cov --manifest-path crates/batchalign-pyo3/Cargo.toml \
  --lcov --output-path lcov-rust.info
cargo llvm-cov --workspace \
  --lcov --output-path lcov-rust-workspace.info
```

## Structural lints (xtask)

Two lints run as xtask subcommands rather than test binaries to avoid
unnecessary integration test binary compilation:

```bash
cargo xtask lint-wide-structs     # Enforces reviewed field caps on wide structs
cargo xtask lint-ci-hygiene       # Version sync, legacy terms, retired packages
```

Both are included in `make ci-local`. Thin test proxies in
`crates/batchalign/tests/` invoke them so `cargo test` still catches
regressions.

## Deterministic simulation testing (turmoil)

Network fault testing uses [turmoil](https://github.com/tokio-rs/turmoil) to
simulate partitions, message delays, server crashes, and concurrent clients
under virtual time. Tests run in <1s with no Python or ML dependencies.

See [Deterministic Simulation (turmoil)](testing-turmoil.md) for architecture,
adapter details, and the full test catalog.

```bash
cargo test -p batchalign --test turmoil_net
```

## Known gaps

1. **No concurrent dispatch stress tests.** The worker pool, job registry, and
   media walker have complex concurrency paths exercised only by `test-echo`
   integration tests. A dedicated stress harness (multiple concurrent jobs with
   real server lifecycle) would catch race conditions earlier.
   Shuttle was evaluated but can't test our Semaphore/broadcast
   primitives (broadcast is a stub that panics, Semaphore forwards
   to real tokio with no schedule exploration); the full
   tool-evaluation note lives outside this public repo.

2. **No negative-path ML tests.** Golden tests verify happy paths. There are no
   tests for graceful degradation when models are unavailable, corrupt, or
   return malformed output under real inference.

3. **No cross-platform CI.** Tests run only on macOS (local) and Linux (CI).
   Windows is a supported platform but has no automated test coverage.

4. **Dashboard Playwright tests are opt-in.** The React frontend E2E suite
   requires manual Chromium setup and is not part of the default CI gate.

## Background test runner (`make test-bg`)

The cost function for test runs is wall-clock time spent *waiting*,
not just time spent running. `scripts/test-bg.sh` wraps any command,
runs it detached, writes structured logs, and posts a macOS desktop
notification on completion. The developer keeps working; failures
ping loudly, successes ping quietly (or silently with `--quiet`).

```bash
scripts/test-bg.sh -- cargo test --workspace
scripts/test-bg.sh -- uv run pytest -m 'golden and mwt_probe' -k fra
```

Log layout per run (under `~/.batchalign3/bg-test/<slug>/`):

| File | Meaning |
|------|---------|
| `<ts>.log` | Full stdout+stderr. Ends with `=== TEST-BG COMPLETED: exit=N duration=Ns ===`. |
| `<ts>.status` | Exit code. File's presence is the unambiguous "done" signal. |
| `<ts>.meta` | cmd, pid, ts_start, ts_end, duration_s, exit. |

The COMPLETED sentinel line lets a watcher (tail, `Monitor` tool,
etc.) detect completion without polling the filesystem. The
`.status` file is the authoritative done signal.

A Makefile glue layer (`make test-bg` / `test-bg-status` /
`test-bg-smoke`) was discussed but is not landed; `scripts/test-bg.sh`
is the current entry point.
