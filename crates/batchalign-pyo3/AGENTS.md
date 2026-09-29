# batchalign-core: Rust Worker Runtime

**Status:** Current
**Last modified:** 2026-09-22 17:57 EDT

## Overview

Slim PyO3 bridge providing the Rust worker runtime for batchalign3's Python ML
worker processes. Workers are stateless inference endpoints that load ML models,
receive structured data from the Rust server via stdio JSON-lines IPC, run
inference, and return raw results.

This crate does NOT contain CHAT parsing, AST manipulation, or pipeline
orchestration, all of that lives in the Rust server (`crates/batchalign/`)
and `batchalign`.

This crate is a regular workspace member at `crates/batchalign-pyo3/`.
The maturin build drives it via
`--manifest-path crates/batchalign-pyo3/Cargo.toml` (the cdylib half of
`crate-type = ["cdylib", "rlib"]` is what Python imports as
`batchalign_core`); `cargo ... --workspace` builds the rlib half and
runs its tests like any other crate.

## Layout

```
crates/batchalign-pyo3/src/
├── lib.rs                  # Module registration
├── error.rs                # Typed error surface for the bridge
├── worker_protocol.rs      # IPC message dispatch
├── worker_execute.rs       # Shared executor control plane: failure taxonomy + the execute operation
├── worker_asr_exec.rs      # ASR execution (Whisper, Cantonese providers)
├── worker_fa_exec.rs       # Forced alignment execution
├── worker_media_exec.rs    # Speaker diarization, OpenSMILE, AVQI
├── worker_text_exec.rs     # Batched text tasks (morphosyntax, utseg, translate, coref)
├── worker_text_results.rs  # Text task normalization + token alignment
├── worker_artifacts.rs     # Prepared artifact loading from IPC
├── cantonese_asr_bridge.rs        # Cantonese provider projection + normalization
└── py_json_bridge.rs       # Python→JSON conversion utility
```

### The executor control plane

Every `execute_*_request_v2` entry point is a thin wrapper over
`worker_execute::execute_request_v2`, which owns the whole verb: parse the
Python request, start the clock, validate attachment descriptors, run the task,
and fold the outcome into an `ExecuteResponseV2`. An executor supplies only its
`run_*` closure.

Properties worth knowing before adding a task:

- **The `run` closure receives a `ValidatedRequestV2`, a phase type only the
  control plane's `validate_request` can construct.** Task/payload agreement
  and attachment-descriptor validation happen once, there; an executor cannot
  be handed an unvalidated request, so its `extract_*` only answers "is this
  payload the one I run". Audio executors likewise get their mono channel
  check through `worker_artifacts::require_mono_prepared_audio`, whose
  `MonoPreparedAudio` return is the only route to decoded samples: the raw
  require/load/decode primitives are private to `worker_artifacts`, so the
  un-checked path has no signature an executor can reach.

- **`ExecuteFailure` is the single owner of the failure-to-protocol-code
  mapping.** Do not re-derive a `ProtocolErrorCodeV2` in an executor; return the
  variant and let the shared code decide. The three executors previously kept
  private copies of the same enum and wrote the mapping out by hand at each
  dispatch site.
- **`ArtifactFailure` records missing-versus-unreadable where the failure is
  born**, in `worker_artifacts.rs`. It used to be flattened into a message and
  recovered downstream by substring match, so rewording a diagnostic silently
  re-labelled every missing attachment as unreadable. Artifact helpers return
  `Result<_, ArtifactFailure>`; `?` carries the category into `ExecuteFailure`.

The response constructors are deliberately private to `worker_execute`. A public
primitive is an invitation to re-implement the verb, and the two things it
protects (an outcome that agrees with `result`, and an elapsed time measured
from the right instant) are exactly what hand-built responses got wrong.

Since 2026-08-21 (ruled) `ExecuteResponseV2` is STRUCTURAL in
`batchalign-types`: its interior is a validated enum, its only constructors
are the two legal shapes, and deserialization refuses a success without a
payload or an error carrying one. Outcome-agrees-with-result is an invariant
the type carries everywhere, not a property of this module's discipline.

## Which compiled extension are you testing?

**The invariant: exactly one extension is importable, and it is the one your
last command built.** Two install modes write to the same environment and
clobber each other, so before this was pinned down, which extension a test run
imported was a property of filesystem residue rather than of anything stated.

| You ran | Install mode | `import batchalign_core` resolves to | Profile |
|---|---|---|---|
| `uv run maturin develop` | editable (`.pth` to the repo root) | `batchalign_core/batchalign_core.abi3.so` in the working tree | debug |
| `make batchalign-python-prepare` | wheel into `.venv` | `site-packages/batchalign_core/batchalign_core.abi3.so` | release |

Both are legitimate and neither is redundant: develop is the inner loop and is
fast because `[tool.maturin] profile = "dev"`; the wheel is the artifact that
actually ships, with the release `batchalign3` binary bundled in. Do not
"simplify" by deleting one.

**What made this a trap.** `python-source = "."`, so the repo root is the
package source and `batchalign_core/__init__.py` shadows site-packages for
anything run from the repo root. It star-imports whichever extension its
`__path__` finds first, and the in-tree one always wins when present. A
`maturin develop` artifact therefore survives a wheel install and keeps being
imported, so "build the wheel, install it, run pytest" can test a wheel it
never loaded. On 2026-08-21 that made a whole session of PyO3 boundary
verification meaningless: the suite had been importing an extension built the
previous afternoon, and it passed either way because behaviour was preserved.

**Two mechanisms now hold the invariant, and they are complementary.**
`batchalign-python-prepare` clears any in-tree extension after installing the
wheel, so its promise is honest. `scripts/check_extension_freshness.sh`, wired
into `_batchalign-test-python`, resolves the extension Python will *actually*
import and refuses if it is older than the Rust it is built from: it proves the
invariant instead of restating it, and covers both modes plus the
no-extension-at-all case.

## Key Commands

```bash
cargo test --manifest-path crates/batchalign-pyo3/Cargo.toml
cargo build --manifest-path crates/batchalign-pyo3/Cargo.toml
cd /path/to/talkbank-tools && uv run maturin develop
```

## Rust Coding Standards

See root `AGENTS.md` for workspace-universal Rust standards (edition, error
handling, logging, file size limits, git conventions). This crate follows all
of those. Crate-specific additions below.

## Rules

- **All JSON via serde.** `#[derive(Deserialize)]`/`#[derive(Serialize)]` structs only.
- **GIL release.** All pure-Rust methods use `py.detach()` (pyo3 0.28).
- **No CHAT parsing here.** CHAT manipulation is in `batchalign` and
  the Rust server. This crate only bridges Python ML calls.

## Architecture

```
Rust Server (crates/batchalign/)
  ├── Parses CHAT, extracts payloads
  ├── Sends IPC request to Python worker (stdio JSON-lines)
  │
  └── Python Worker Process
        ├── worker_protocol.rs: dispatch IPC messages
        ├── worker_*_exec.rs: load prepared artifacts, call ML model
        ├── cantonese_asr_bridge.rs: project Cantonese provider output
        └── Returns raw results → Rust server injects into CHAT
```

**See also:** [Interface Map](../../INTERFACE_MAP.md) for unified documentation of all
Python/Rust boundaries, including Python caller locations, shared schema
definitions, and responsibility splits per boundary.
