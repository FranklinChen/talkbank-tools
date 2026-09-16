# Adding Inference Providers

**Status:** Current
**Last updated:** 2026-09-16 03:36 EDT

Batchalign3 no longer has a public entry-point plugin system. New engines are
added in-tree as built-in worker capabilities.

This page covers the current extension path.

If you are adding a new command, declare it as one `CatalogEntry` in
`crates/batchalign/src/recipe_runner/catalog.rs` (choosing its `CommandFamily`
there) with its stage recipe in `recipe_runner/recipes.rs`, and keep any
algorithmic or orchestration logic in the owning Rust module. Engine work should
support that Rust-owned command flow; it should not define the command shape on
its own. See [Adding a New Command](./adding-commands.md).

## Choose the layer first

There are two different things you might be adding:

1. A new **worker-side inference backend** such as a new ASR or FA engine.
2. A new **server command** that needs Rust-side orchestration plus, optionally,
   a new worker inference task.

Most engine work starts in Python and only touches Rust for typed IPC contracts,
command registration, and server orchestration.

## Adding a worker-side inference backend

### 1. Add the inference module

Create a built-in module under `batchalign/inference/` that exposes a pure
inference helper consumed by a typed V2 worker host:

```python
from __future__ import annotations

from batchalign.worker._types_v2 import MyTaskItemV2, MyTaskResultItemV2


def infer_my_task(items: list[MyTaskItemV2]) -> list[MyTaskResultItemV2]:
    results: list[MyTaskResultItemV2] = []
    for item in items:
        results.append(MyTaskResultItemV2(ok=True))
    return results
```

Keep these modules CHAT-free. Python workers should accept structured payloads
and return structured results only.

### 2. Add or reuse the task identifier

If this is a new live infer task, add it in the V2 IPC type definitions:

- `batchalign/worker/_types_v2.py`
- `crates/batchalign-types/src/worker_v2/` (re-exported by `crates/batchalign/src/types/worker_v2.rs`)

If you are only adding a new engine behind an existing task such as ASR or FA,
reuse the existing task and add only the new engine selector/state.

### 3. Load model state in the worker

Update `batchalign/worker/_model_loading/` so `load_worker_task()` can
initialize the new engine for the relevant infer task. This is where task-level
engine overrides are resolved and worker state is populated.

For existing command families, you usually update one of:

- `load_asr_engine()`
- `load_fa_engine()`
- `load_translation_engine()`
- `load_stanza_models()` in `worker/_stanza_loading.py`

### 4. Wire dispatch and capability advertisement

Update:

- `batchalign/worker/_execute_v2.py` to route the task or engine
- `batchalign/worker/_text_v2.py` if the task belongs to the shared batched
  text host
- `batchalign/worker/_handlers.py` to advertise `infer_tasks`, and, for forced
  alignment only, to name the engine in `engine_versions`

A capability report carries task support plus the FA identity only.
`_reported_engine()` names FA's engine and returns `None` for every other
task; Rust admission refuses a report that names an engine for any other task
(`EngineNamedForNonFaTask`). If the new engine is an FA variant, keep the task
stable and make `_reported_engine()` name the FA engine that actually loaded.
Report `None` until the process can name it (FA reports `None` until an FA
engine has loaded); never a guessed name or `"unknown"`. Set an identity where
the loader loads the model, as the translation loaders build one
`LoadedTranslation` record (backend and engine together) on `_state`. A name
must be a valid `ReportedEngineName`, a wrapper over `StampSafeText`:
non-blank, no surrounding whitespace (the Unicode `White_Space` set, written
out as `_STAMP_WHITESPACE` in `batchalign/worker/_types.py`), and none of `|`,
`;`, `]` or a line break.

Where the engine is named depends on when the server needs it:

- **On results (translate, coref, morphosyntax).** Each result item names the
  engine or model that produced it: `translated` and `resolved` items carry
  `engine`, and `analyzed` items carry `model` with the pipeline variant.
  Provenance is built from the results a file applied, so these commands never
  read the capability report for identity and never fail a job for a `None`
  report. A new engine for a text task should name itself the same way.
- **Before dispatch (forced alignment only).** FA cache rows are read under
  the FA engine's namespace before any worker runs, so FA's identity comes
  from the capability report. Two different facts are read at two moments.
  Whether the worker SUPPORTS a command's primary infer task is
  `command_supported` in `capability.rs`, which decides what `/health`
  advertises and which jobs are accepted. Whether the FA engine is NAMED is
  read at dispatch: `WorkerPool::ensure_command_capabilities` loads the task
  on the selected worker and returns `LoadedCapabilities` (the task it loaded
  and the report taken after that load), and the forced-alignment arm in
  `runner/routing.rs` reads the engine with `FaCacheNamespace::from_loaded`
  (`crates/batchalign/src/engine_reports.rs`). That constructor refuses a
  report taken after another task loaded, an engine still unnamed after the
  load, and a worker that does not support FA. So a lazily loading worker, or
  a registry daemon probed before it loaded FA, still advertises and accepts
  `align`; dispatch loads FA and reads the engine then. `FaCacheNamespace` is
  a plain newtype over `ReportedEngineName`, and that dispatch arm is the only
  place a pre-dispatch identity is required; no catalog field declares one.

**Capability gate (critical):** The `_capabilities()` function in `_handlers.py`
uses **import probes** to decide which infer tasks to advertise. If you add a new
`InferTask`, you must add it to the `_INFER_TASK_PROBES` dict with the tuple of
Python modules that must be importable, and give it an arm in the exhaustive
`_reported_engine()` match:

```python
_INFER_TASK_PROBES: dict[InferTask, tuple[str, ...]] = {
    ...
    InferTask.MY_TASK: ("my_library",),
}
```

Capabilities are detected lazily from the first real worker spawn, there is no
dedicated probe worker at startup. The capability check uses import probes, not
loaded model state. This means capability advertisement must be based on import
availability, never on `_state.my_model is not None`. If you gate on loaded
model state, your task will not be advertised and the server will silently
exclude the command. The FA engine NAME in `engine_versions` is the opposite:
it reflects what has actually loaded, and is `None` before that.

The Rust server cross-checks: commands whose primary `InferTask` is not in the
worker's `infer_tasks` list are excluded from the server's advertised
capabilities; engine names are not consulted. A report whose
`engine_versions` holds an invalid name or a key that is not a task fails to
deserialize, and one that lacks an entry for an advertised task, names a task
that was not advertised, or names an engine for a task other than FA is
refused where the pool admits it (`WorkerPool::record_capabilities`). See [Capability Discovery](../../architecture/python-rust-boundary/python-rust-boundary.md#capability-discovery)
for the full flow.

### 5. Register dependencies

Add the engine's Python dependencies to the appropriate section in
`pyproject.toml`:

- **Core engines** (expected to work out of the box): add to `dependencies`.
  All standard commands (align, transcribe, translate, morphotag, etc.) have
  their dependencies in `dependencies` so that a standard install
  gives users everything.

- **Built-in engines with extra runtime dependencies**: add them to
  `dependencies` if they are part of the supported built-in engine surface.
  Credential-gated or region-specific does not imply a separate install tier.

  Users then install `batchalign3[my-engine]`.

### Cross-cutting Rust edits for a new ASR engine variant

For an ASR engine specifically, a variant lives in **three Rust enums
that must stay in sync**: a mismatch in any one silently mis-routes
dispatch. The following tables enumerate every file/identifier you
must update when adding a new variant. Use the `whisper_hub` addition
(2026-04-22) as a worked example of every line item.

**The three-enum synchronization:**

| Enum | File | Role |
|------|------|------|
| `AsrEngineName` | `crates/batchalign/src/types/engines.rs` | User-facing type. Wire name, parsing, dispatch-key lookup. |
| `AsrBackendV2` | `crates/batchalign-types/src/worker_v2/requests.rs` | IPC contract with Python workers. Regenerate the schema after editing via `bash scripts/generate_ipc_types.sh`; the conformance test then checks the hand-written Python model against it. |
| `AsrWorkerMode` | `crates/batchalign/src/transcribe/types.rs` | Server-side dispatch selector that bridges the other two. |

**Helpers each variant must appear in:**

| Function | File | Purpose |
|---|---|---|
| `AsrEngineName::wire_name()` | `types/engines.rs` | Rust→string for JSON/SQLite. |
| `AsrEngineName::try_from_wire_name()` | `types/engines.rs` | String→Rust at boundaries. |
| `AsrEngineName::dispatch_override_name()` | `types/engines.rs` | Pool key (must equal wire_name or `None`). |
| `AsrWorkerMode::from_engine_name()` | `transcribe/types.rs` | Wire-string → worker-mode variant. |
| `AsrWorkerMode::as_v2_backend()` | `transcribe/types.rs` | Worker-mode → IPC backend. |
| `AsrBackend::provenance_name()` | `transcribe/types.rs` | Canonical engine identity for production transcript provenance and warnings. |
| `asr_backend_engine()` | `crates/batchalign/src/worker/pool/execute_v2.rs` | Maps the wire backend to an `AsrEngineName`; the pool-key string then comes from `dispatch_override_name()`, so there is no second table to keep in step. |
| *(input-source routing)* | `crates/batchalign/src/transcribe/infer.rs` | Match on `AsrWorkerMode` picks `PreparedAudio` (local model) vs `ProviderMedia` (external service). |

**Model identity obligations.** A variant present in all three enums and
every helper above is still refused at the bridge without these three:

| Obligation | Where | Why it is not optional |
|---|---|---|
| The ASR result carries `AsrModelIdentityV2` | your worker-side runner builds it; typed in `crates/batchalign-types/src/worker_v2/responses.rs` | `model` is required, not optional, so a result naming no models does not compile. The bridge admits the reported composition against the request's pin and refuses a disagreement by name; for a provider backend it does so BEFORE calling the provider, so a mismatch costs no paid request. A worker that recorded no identity is refused per engine, never handed one inferred from the request. |
| The composition is per engine, not one model | `crates/batchalign/src/model_manifest.rs` | An engine that loads more than one model names all of them: Qwen names its ASR model AND its forced aligner, Paraformer names its checkpoint plus the voice-activity and punctuation models it additionally loads. The UTR ASR cache namespace is built from that composition, so an omitted member pools rows produced by different weights. |
| A monologue speaker is `SpeakerAttributionV2` | `responses.rs::AsrMonologueV2::speaker` | A bare string cannot tell "this engine separates nobody, so it named nobody" from "it named nobody although separation was requested". That is why undiarized engines once wrote `"0"`, which became a `PAR0` tier indistinguishable from a real first speaker. |

The contract itself is specified twice, and both are worth reading before you
add a variant: the ASR `#### Result` section of
[worker-protocol-v2](worker-protocol-v2.md), and the ASR row of
`INTERFACE_MAP.md` at the repository root.

Worker-side enum (matches Rust wire name one-to-one):

- `AsrEngine` in `batchalign/worker/_types.py`: the Python enum the
  worker bootstrap stores in `_state.asr_engine`.

Request validation surface (optional, only for engines with
per-engine language constraints like the Cantonese ASR engines):

- `validate_language_support()` in `crates/batchalign/src/types/request.rs`.

**Per-language default model_id resolution.** If your engine picks a
model per language (e.g. different HF fine-tunes per language), add
entries to `batchalign/models/resolve.py` rather than inventing a new
per-engine table. `resolve("your_engine", lang_iso3)` returns the
model_id or `None`; raise a typed error on `None` unless the caller
passed an explicit override, rather than falling back to a generic
default.

**HF Whisper fine-tune gotcha.** HF community Whisper fine-tunes bake
`language` and `task` into their own `generation_config`. Passing
those again in `generate_kwargs` produces gibberish. The escape hatch
is the `skip_language_force: bool` flag on
`batchalign/inference/types.py::WhisperASRHandle`: when `True`,
`gen_kwargs()` returns ONLY `{"max_new_tokens": 444}` and omits
`task`, `language`, `generation_config`, and `repetition_penalty`.
See `batchalign/inference/whisper_hub.py` for the wiring:
pass `language="auto"` to `load_whisper_asr()` AND set
`handle.skip_language_force = True` before returning.

**Why the `max_new_tokens=444` safety cap.** With empty
`generate_kwargs`, the HuggingFace ASR pipeline can let a fine-tune
fall into a non-converging decoder state where it never predicts an
end-of-utterance token, hanging the worker for tens of minutes. The
cap is a hard upper bound on tokens-per-chunk, not a probability
override, so it is a no-op on successful runs but terminates
runaways. The value 444 is one below Whisper's legal max:
`max_target_positions = 448` includes the 3 special start tokens,
leaving 445 for new tokens, with 444 chosen for one token of margin.

## TDD discipline for engine additions

The rest of this section is a TDD checklist derived from actually
shipping (and breaking) `whisper_hub`. Every item is reactive: it
corresponds to a mistake that was made once and should be prevented
by test structure in the future.

### Test at the observable boundary, not at the function you call into

When a loader or constructor populates a stateful intermediate (a
handle, a worker state, a registry entry), do not substitute tests
that assert "the right thing was passed *into* the constructor" for
tests that assert "the object the constructor returns, when exercised
by a downstream caller, produces the right observable behavior."

Concrete example. The `whisper_hub` loader was initially tested only
by asserting that `load_whisper_asr` received `language="auto"`,
a proxy for "fine-tunes won't get their language re-forced at
`generate()` time." That assertion was true but insufficient: the
V2 inference path (`infer_whisper_prepared_audio`) calls
`handle.gen_kwargs(request_lang)` and ignores `handle.lang`
entirely. The fine-tune was receiving `task="transcribe",
language="malayalam"` at every `generate()` call and would have
produced cross-script gibberish. The unit tests did not fail because
nothing ever exercised the actual runtime path.

The fix was an additional test that constructs a
`WhisperASRHandle(skip_language_force=True)` directly, calls
`gen_kwargs("malayalam")` on it, and asserts `task` and `language`
are absent from the returned dict. That test exercises the runtime
contract that production depends on.

General rule: **for every stateful intermediate in the pipeline,
there must be tests on both sides of it.** Input-side tests verify
the construction call site. Output-side tests verify that downstream
callers, given only the constructed object, see the right behavior.

### Grep for every method that consumes the state you set

When you set a field on a shared handle or state object, grep for
every call site that reads that field OR reads a seemingly-unrelated
field that could diverge. `gen_kwargs(lang)` reading
*the caller-supplied* `lang` rather than `self.lang` is exactly that
kind of divergence: two plausibly-interchangeable data sources where
only one was the contract.

```bash
# What reads self.lang?
rg 'model\.lang|handle\.lang|\.lang =' batchalign/inference/
# What calls gen_kwargs?
rg 'gen_kwargs\(' batchalign/
```

If the same conceptual value (the "language to transcribe in") flows
through both paths, your engine-addition test must cover both.

### Add a failing runtime-behavior test before writing any loader code

For ASR engine additions, the RED test baseline is:

1. `test_<engine>_wire_roundtrip`: `AsrEngineName::<Engine>.wire_name()`
   roundtrips through `try_from_wire_name`. Already a common idiom;
   add the variant to the existing test module.
2. `test_<engine>_worker_mode_lowers_correctly`: `AsrWorkerMode`
   variant lowers to the right `AsrBackendV2` and back.
3. `test_<engine>_loader_dispatch`: worker bootstrap's
   `load_asr_engine()` routes `engine_overrides["asr"]=="<engine>"`
   to your new loader function.
4. **`test_<engine>_handle_gen_kwargs_for_concrete_language`**,
   construct a handle the way your loader would, call its
   generation-kwargs method with a concrete (non-auto) language, and
   assert the output dict matches what you expect `generate()` to
   receive. This is the test that catches the fine-tune trap above.
5. **`test_<engine>_resolves_model_id_for_seeded_language`**: if
   your engine uses per-language defaults from `resolve.py`, pin the
   seed entry with a direct assertion on `resolve("<engine>", lang)`.
6. **`test_<engine>_raises_on_unseeded_language`**: if your engine
   raises on a missing default, pin the error type and message
   fragment. Don't let the error degrade into a silent stock
   fallback.
7. **`test_<engine>_result_carries_model_identity`**: build the result
   your runner returns and assert its `model` names every model the
   engine loads, each at the revision it was OBSERVED at rather than
   the one that was requested.
8. **`test_<engine>_bridge_refuses_unreported_identity`**: a worker
   response recording no identity for your engine must be refused by
   name, and for a provider backend the refusal must happen before the
   provider is called.
9. **`test_<engine>_monologue_speaker_attribution`**: if your engine
   produces monologues, assert an undiarized result carries the
   `undiarized` attribution rather than a speaker labelled `"0"`.

Guard-rail tests must accompany any deny-list / recommendation
changes, if you redirect users from engine X to engine Y for some
language, engine Y must itself pass validation for that language.

### Rebuild the PyO3 extension: the Python worker's dispatch is Rust

`AsrBackendV2` exists in two Rust crates (`batchalign-types` for the
server, and `crates/batchalign-pyo3/src/worker_asr_exec.rs` via that crate). The PyO3
function `batchalign_core.execute_asr_request_v2(request, ...)` owns
the runtime dispatch: it pattern-matches on `AsrBackendV2` inside the
worker process and routes to the right runner. Adding a new enum
variant means you must:

1. Add a match arm inside `crates/batchalign-pyo3/src/worker_asr_exec.rs::run_asr` that
   routes the new variant. The compiler will catch the missing arm if
   you let it, but only after you rebuild the PyO3 extension.
2. Rebuild `batchalign_core`: `make batchalign-python-prepare` (which
   produces a fresh wheel via the maturin backend declared in
   `pyproject.toml` and reinstalls it into the dev environment), or run
   any `uv run …` command to trigger an incremental rebuild against
   `[tool.maturin] profile = "dev"`. Running `make build` alone (which
   does `cargo build --workspace --release`) compiles the PyO3 crate but
   does **not** install the resulting `.so` into the Python environment
   that workers import from. Without the install step, workers silently
   load the previously installed `batchalign_core` extension, which has
   no match arm for your new variant and drops the request with no
   response (the Rust server then sits in its request-timeout wait for
   ~30 minutes). Neither end logs the Pydantic / serde validation
   failure that would localize this.
3. Also update the Python `AsrBackendV2` enum in
   `batchalign/worker/_types_v2.py`. It is hand-maintained, and the
   conformance test (`test_ipc_type_conformance.py`) is what catches you
   having forgotten: it compares that enum against the regenerated
   `ipc-schema/`, so run `bash scripts/generate_ipc_types.sh` after the
   Rust change and let the test tell you what is missing.

**Same class of bug as the `gen_kwargs` trap.** Both are "state
crosses a boundary I didn't search for, so the test suite never
exercises the real runtime path." The fix is the same grep ritual:

```bash
# Before declaring a new ASR variant done:
rg "AsrBackendV2::" crates/
rg 'AsrBackendV2\.'  batchalign/
```

Every match site gets a new arm. If you can't find the grep result
again on first read, the variant has a hole.

### Compile-time vs install-time mismatch

`batchalign-pyo3` is a member of the root Cargo workspace (see
`members` in the root `Cargo.toml`), so `cargo check --workspace`
and `make build` *do* compile the PyO3 bridge and *do*
exhaustive-match-check `AsrBackendV2` arms there. Targeted
commands like `cargo check -p batchalign` skip it because
`batchalign` does not depend on `batchalign-pyo3`.

The failure mode the rebuild ritual prevents is therefore not a
missed compile error: it is an install gap. The freshly compiled
PyO3 `.so` lives in `target/`, while the Python worker process
imports `batchalign_core` from the wheel previously installed into
the active `uv` environment. Without `make batchalign-python-prepare`
(or an equivalent reinstall), the worker keeps loading the stale
`.so` and the new variant produces a silent stall on the worker's
stdin readline (no response surfaces until the 30-minute
audio-task timeout fires).

The grep-and-rebuild ritual above is the defense; the workspace
structure no longer is the gap.

(An out-of-date doc-comment in `crates/batchalign-pyo3/Cargo.toml`
describes the crate as "outside the root workspace by design",
that comment predates the move into the workspace and should be
updated when next touched.)

### Structural opportunity: `gen_kwargs` takes a string

`WhisperASRHandle.gen_kwargs(lang)` dispatches on a string: `"auto"`
is one special case, `"Cantonese"` is another, and any other string
means "force this language on `generate()`". Plus the new
`skip_language_force` flag adds a fourth behavior. This is
boolean-blindness dressed in string clothing, four distinct
generation modes hidden in two orthogonal inputs. A future refactor
should replace this with an enum such as `WhisperGenMode::{
AutoDetect, FinetunePinnedByConfig, CantoneseSpecialCase,
ForceLanguage(Name) }`, with each engine's loader picking the
variant explicitly. Out of scope for a single-engine addition, but
worth doing before the *next* whisper-family engine (WhisperX,
WhisperOai Hub variant, etc.) repeats the same mistake.

## Adding a new server command

If you are adding a new top-level command (not just a new engine for an
existing command), see the detailed 8-step checklist in
[Rust CLI and Server](rust-cli-and-server.md#adding-a-new-cli-command).

In addition to those Rust-side changes, update these Python-side surfaces:

1. **`crates/batchalign-types/src/command_spec.rs`**: Add a `CommandSpec` entry to
   `COMMAND_SPECS`. Then run `cargo xtask gen-runtime-toml` to regenerate
   `batchalign/runtime_constants.toml` (the generated file is the shared
   Rust/Python source of truth; do not edit it directly).
2. **`batchalign/worker/_handlers.py`**: Add the `InferTask` to
   `_INFER_TASK_PROBES` (at `_handlers.py:77`) so the worker
   advertises it. This is the only Python-side probe mechanism; the
   server cross-checks advertised infer-tasks against required
   command capabilities. See
   [step 4 above](#4-wire-dispatch-and-capability-advertisement) for
   details.
3. **`batchalign/worker/_model_loading/`**: Register the dynamic
   runtime host for the new task if it depends on loaded model state or
   engine-specific wiring. Reserve **`batchalign/worker/_execute_v2.py`** for
   the small task router that dispatches to those prepared hosts.

Remember: command semantics live in the command-owned Rust layer, not in the
worker bootstrap layer. The worker layer should only know how to load engines
and execute typed tasks.

## No public extension surface

There is no public Python extension surface. New engines are added
in-tree, through the steps above; there is no `batchalign.plugins`
discovery API, no `PluginDescriptor` contract, and no supported
external-package path for adding ASR / FA / morphosyntax / etc.
backends without modifying this repo.

The worker-side modules under `batchalign.worker.*` and the
`batchalign.providers` re-export module are internal implementation
detail and may change without notice.

If you want to ship an engine without making it a mandatory
dependency of the package, use an `[project.optional-dependencies]`
extra in `pyproject.toml` so the dependency installs only when
explicitly requested.

## Test expectations

At minimum, add:

- unit tests for the new inference module
- worker dispatch tests covering `_execute_v2()` or the relevant task host
- bootstrap/handler-registration tests if the task uses dynamic worker runtime
- Rust integration coverage if the new engine changes server orchestration,
  command routing, or capability gating
- doc updates for install syntax, command options, and migration notes if this
  replaces a BA2 or pre-release workflow

Relevant existing coverage lives in:

- `batchalign/tests/` (Python-side worker dispatch / handler tests)
- `crates/batchalign/tests/` (Rust-side server / IPC / CI hygiene tests)
- inline Rust tests under `crates/batchalign-pyo3/src/` for the PyO3 bridge

## Rule of thumb

If the change affects CHAT structure, it belongs in Rust.

If the change affects model inference only, it usually belongs in Python plus
the typed worker contract that Rust consumes.
