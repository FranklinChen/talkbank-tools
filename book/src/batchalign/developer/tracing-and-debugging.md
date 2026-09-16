# Tracing and Debugging

**Status:** Current
**Last updated:** 2026-09-15 07:21 EDT

This document describes the tracing and debugging strategy across the
batchalign3 stack: Rust (batchalign-core PyO3 bridge), Rust (CLI and server
control plane), and Python (pipeline engines and worker process).

## Server Log File

The server writes its own log file directly, like Nginx or Apache.
When running in `--foreground` mode (which is how both the daemon
spawn and Ansible start the server), stderr is redirected to
`~/.batchalign3/server.log` via `dup2(2)`. All `tracing` output
(WARN and above by default) is captured in this file regardless of
how the process was started.

**Log location:** `~/.batchalign3/server.log` (append mode)

**Default level:** `WARN`: captures pipeline timing, cache metrics,
heartbeat warnings, worker crashes, slow queries. Does NOT capture
per-file progress, worker spawn/ready, or routine lifecycle events.

**For debugging:** Run with `-v` to get `INFO` level (worker spawns,
job lifecycle, per-file progress) or `-vv` for `DEBUG` (full payload
details). Or set `RUST_LOG=info` environment variable.

```bash
# Normal operation (WARN only):
batchalign3 serve start

# Debugging session (INFO: shows worker spawns, job lifecycle):
batchalign3 -v serve start --foreground

# Deep debugging (DEBUG: shows payloads, IPC):
batchalign3 -vv serve start --foreground

# Override per-module:
RUST_LOG=batchalign::morphosyntax=debug batchalign3 serve start
```

**Log rotation:** Not implemented. The log file grows unbounded.
For long-running production servers, periodically truncate:
```bash
: > ~/.batchalign3/server.log  # truncate without restarting
```

## Verbosity Levels

A single `-v` / `-vv` / `-vvv` flag on the CLI controls both Rust tracing and
Python logging across the entire stack.

| Level | Rust (`tracing`) | Python (`logging`) | When to use |
|-------|-------------------|--------------------|-------------|
| 0 (default) | `WARN` | `WARNING` | Normal operation |
| 1 (`-v`) | `INFO` | `INFO` | Server start/stop, job lifecycle |
| 2 (`-vv`) | `DEBUG` | `DEBUG` | Per-file progress, engine boundary data |
| 3 (`-vvv`) | `TRACE` | `DEBUG` | Full payload dumps (truncated) |

### How verbosity propagates

```rust,ignore
CLI (main.rs)
  │
  ├─ init_tracing(verbose)          ← sets Rust filter level
  │
  └─ serve_cmd::start(args, verbose)
       │
       └─ PoolConfig { verbose, .. }
            │
            └─ WorkerConfig { verbose, .. }
                 │
                 └─ python3 ... --verbose N   ← forwarded to batchalign.worker
                      │
                      └─ logging.basicConfig(level=...)
```

In background mode (`batchalign3 serve start` without `--foreground`), the
`-v` flags are forwarded to the re-exec'd background process.

## Engine Boundary Tracing

The highest-risk surface in the stack is the Rust-Python boundary where data
crosses serialization layers. This boundary is instrumented at three points:

### 1. Morphosyntax batch orchestrator (Rust side)

The batchalign-side orchestrator at
`crates/batchalign/src/morphosyntax/worker.rs` instruments the
extract → infer → inject sequence with `debug!` traces at each stage
boundary: utterance + word counts going in, response item counts
coming back from the worker, and injection-time counts going out.
(The previous PyO3 ParsedChat callback path
`add_morphosyntax_batched_inner` in `pyo3/src/morphosyntax_ops.rs`
was retired in the 2026-03-21 PyO3 slimdown; worker-runtime pyo3
today is worker_protocol.rs + worker_*_exec.rs only.)

### 2. Python inference module (`batchalign/inference/morphosyntax.py`)

The `batch_infer_morphosyntax` function logs:

- Item count and elapsed time at `INFO` level on completion
- Sentence count mismatch warnings at `WARNING` level
- Stanza batch failure warnings at `WARNING` level

### 3. Worker IPC (`crates/batchalign/src/worker/handle/`)

Worker spawn, shutdown, health checks, and IPC dispatch are logged
at `info!` and `debug!` levels across the
`crates/batchalign/src/worker/handle/` submodules (`mod.rs`,
`config.rs`, `ipc.rs`, `lifecycle.rs`, `spawn.rs`, `protocol.rs`).
Worker stderr is captured for crash diagnostics.

## Performance

The `tracing` crate's `debug!` and `trace!` macros cost **~1-5 ns** when the
corresponding level is filtered out (the default level is `WARN`). All
instrumented functions are per-file or per-utterance, never per-word. There is
no measurable performance impact during normal operation.

Python `logging.debug()` calls are similarly inexpensive when the logger level
is `WARNING`.

## Safe AST Construction

### The problem

Raw text from NLP engines (Stanza, Whisper) must be converted to CHAT AST
nodes. Directly constructing AST nodes with `Word::new_unchecked` bypasses the
lexical validation that the parser would normally enforce, allowing malformed
words into the AST. These silently propagate until pre-serialization validation,
at which point the error is far from the root cause.

### Policy

1. **Always try `DirectParser::parse_word()` first**: if the text is valid
   CHAT syntax, the parser returns a properly validated `Word`.
2. **Only fall back to `new_unchecked` when the input is genuinely unparseable**
   (e.g., ASR returned non-CHAT characters). Log a `warn!` when this happens.
3. **Never fall back to `new_unchecked` in retokenization**: if a Stanza-split
   token can't be parsed, keep the original CHAT word unchanged.

### Implementation

Three categories of `new_unchecked` usage have been addressed:

**A. ASR transcript construction**
(`crates/batchalign-transform/src/build_chat/`):
ASR engines return raw text that must become CHAT words. The code
tries `DirectParser::parse_word()` first and only falls back to
`new_unchecked` with a `warn!` if parsing fails. Entry points:
`build_chat()` in `build_chat/mod.rs:41` and `build_chat_from_json()`
in `build_chat/bridge.rs:10`.

**B. Retokenization fallback**
(`crates/batchalign-transform/src/retokenize/`):
When Stanza splits a CHAT word into MWT sub-tokens, each sub-token
must be parsed back into a CHAT `Word`. `try_parse_token_as_word()`
at `crates/batchalign-transform/src/retokenize/parse_helpers.rs:108`
returns `Option<Word>` instead of always succeeding. On parse
failure, the original word is preserved (no invalid content enters
the AST).

**C. Temporary scaffolding**: a temporary word is used only as input
to `resolve_word_language()`
(`../chatter/crates/talkbank-model/src/validation/word/language/resolve.rs:137`)
and never injected into the AST. This is a documented acceptable use
of `new_unchecked`.

### Injection-time alignment check (`crates/batchalign-transform/src/morphosyntax/injection.rs`)

Before injecting MOR/GRA tiers into an utterance, the code validates
that the number of MOR items matches the number of alignable words
extracted from the AST. A mismatch is a bug, it means the
extraction or NLP mapping is wrong.

```rust,ignore
// crates/batchalign-transform/src/morphosyntax/injection.rs: count alignment check
let word_count = extracted.len();
let mor_count = mors.len();
if word_count != mor_count {
    tracing::warn!(word_count, mor_count, ...);
    return Err(format!("MOR item count ({mor_count}) does not match ..."));
}
```

This catches problems at the point of injection (close to root cause) rather
than deferring to the pre-serialization validation pass.

## Debugging Workflows

### Diagnosing a morphosyntax failure

1. Run with `-vv` to see per-utterance word counts and Stanza I/O:
   ```bash
   batchalign3 -vv morphotag input/ output/
   ```

2. If a specific utterance fails, the `warn!` from `inject.rs` will report the
   exact word count mismatch and utterance text.

3. Run with `-vvv` (trace) to see the full JSON payload sent to Stanza and the
   JSON response (truncated to 500 chars).

### Diagnosing a retokenization issue

When Stanza splits a word into MWT sub-tokens and one sub-token is
unparseable:

1. A `warn!` is logged: `"Token is not valid CHAT syntax; keeping original word"`.
2. The original word is preserved in the AST.
3. The MOR cursor advances past the sub-token indices to stay in sync.

### Diagnosing an ASR construction issue

When ASR returns text that isn't valid CHAT:

1. A `warn!` is logged: `"ASR word is not valid CHAT syntax; using unchecked fallback"`.
2. The unchecked word enters the AST, this is expected for non-CHAT characters.
3. Pre-serialization validation will catch any downstream issues.

### Checking worker verbosity

To verify that verbosity reaches Python workers:

```bash
batchalign3 -vv serve start --foreground
```

Worker stderr will show `DEBUG`-level messages from `batchalign.worker` and
`batchalign.inference.morphosyntax`.

## Debug Artifact Pipeline (`--debug-dir`)

The `--debug-dir PATH` flag (or `BATCHALIGN_DEBUG_DIR` env var) writes
structured CHAT/JSON artifacts at pipeline stages that are currently wired.
`align` and `transcribe` have those producers; accepting the global option on
another command is not a promise that the command emits artifacts. When
`--debug-dir` is not set, all dump operations are zero-cost no-ops.

The directory is interpreted by the **server process**. The CLI converts a
relative value to an absolute client path before submission, which is correct
for direct mode, a loopback server, or an explicitly shared filesystem. It is
not a remote artifact-transfer protocol: on a different host the same absolute
path may be absent, unwritable, or name unrelated storage. For a remote server,
pass an absolute path that is meaningful on that server and retrieve it through
normal host access. A future server-owned artifact store should replace this
cross-host path convention; current behavior must not be described as copying
debug files back to the client.

```bash
# Alignment with debug artifacts
batchalign3 align input/ output/ --lang eng --debug-dir /tmp/ba3-debug

# Transcription with debug artifacts
batchalign3 transcribe audio/ output/ --lang eng --debug-dir /tmp/ba3-debug

# Via environment variable (useful for server-side debugging)
BATCHALIGN_DEBUG_DIR=/tmp/ba3-debug batchalign3 transcribe audio/ output/
```

### Architecture

The debug artifact pipeline is built around the `DebugDumper` struct in
`crates/batchalign/src/runner/debug_dumper.rs`. It follows a zero-cost
abstraction pattern: when constructed without a directory, every method is an
immediate no-op. When constructed with a directory, methods write artifacts to
disk at each pipeline stage.

```mermaid
graph TD
    CLI["CLI: --debug-dir PATH<br>(global_opts.rs)"]
    ENV["ENV: BATCHALIGN_DEBUG_DIR"]
    CLI --> CO["CommonOptions.debug_dir<br>(options.rs)"]
    ENV --> CO
    CO --> |"job submission"| JOB["Job.dispatch.options.common().debug_dir"]
    JOB --> |"per-file task"| DD["DebugDumper::new(debug_dir)"]
    DD --> |"Some(path)"| WRITE["Write artifacts to disk"]
    DD --> |"None"| NOOP["Zero-cost no-op"]
```

### How DebugDumper threads through each pipeline

Each pipeline creates its own `DebugDumper` at the per-file dispatch level,
extracting `debug_dir` from the job options. The dumper is then threaded through
the pipeline context and called at stage boundaries.

```mermaid
flowchart TB
    subgraph "Align Pipeline (fa_pipeline.rs)"
        FA_DISPATCH["dispatch_fa_infer()"] --> FA_CTX["FaFileContext { dumper }"]
        FA_CTX --> FA1["dump_utr_input()"]
        FA1 --> FAR["Rev UTR → dump_rev_evidence()<br>(one per raw evidence key)"]
        FAR --> FA2["dump_utr_tokens()"]
        FA2 --> FA3["dump_utr_output()"]
        FA3 --> FA4["dump_fa_grouping()"]
        FA4 --> FA5["dump_fa_group_result() x N"]
        FA5 --> FAE["dump_fa_evidence()\n(versioned, fail-closed)"]
        FAE --> FA6["dump_fa_output()"]
    end

    subgraph "Transcribe Pipeline (pipeline/transcribe.rs)"
        TX_DISPATCH["dispatch_transcribe_infer()"] --> TX_CTX["TranscribePipelineContext { dumper }"]
        TX_CTX --> TX0["Rev stage_asr_infer → dump_rev_evidence()<br>(versioned, fail-closed)"]
        TX0 --> TX1["stage_asr_infer → dump_asr_response()"]
        TX1 --> TXS0["stage_speaker_diarization<br/>(when dedicated backend is needed)"]
        TXS0 --> TXS1["dump_speaker_evidence()<br>(versioned, fail-closed)"]
        TXS1 --> TXS2["dump_speaker_turns()<br>(exact consumed turns, fail-closed)"]
        TXS2 --> TX2["stage_asr_postprocess<br/>speaker projection + pre-CHAT utseg"]
        TX2 --> TXP["UtsegEvidenceSink.write(pre_chat)<br/>(model-backed path, fail-closed)"]
        TXP --> TX3["stage_build_chat → dump_post_asr_chat()"]
        TX3 --> TX4["stage_run_utseg → dump_pre_utseg_chat()"]
        TX4 --> TXE["UtsegEvidenceSink.write(post_chat)<br/>(fail-closed)"]
        TXE --> TX5["stage_run_utseg → dump_post_utseg_chat()"]
        TX5 --> TX6["stage_run_morphosyntax → dump_pre_morphosyntax_chat()"]
    end
```

### Artifact directory layout

For a transcribe job on `sample.wav` with `--debug-dir /tmp/debug`:

```text
/tmp/debug/
  # Transcribe pipeline artifacts
  sample_rev_evidence.json       # Rev media/request/cache/projection causal record
  sample_speaker_evidence.json   # Speaker request/cache/projection causal record
  sample.turns.json              # Exact normalized dedicated-speaker segments
  sample_asr_response.json       # ASR tokens + timestamps as the server received them
  sample_pre_chat_utseg_evidence.json
                                 # Exact words, assignments, model, policy evidence
  sample_post_asr.cha            # CHAT after assembly (before utseg)
  sample_pre_utseg.cha           # CHAT entering the post-CHAT utseg pass
  sample_post_chat_utseg_evidence.json
                                 # Post-CHAT words, assignments, model, policy evidence
  sample_post_utseg.cha          # CHAT after the post-CHAT utseg pass
  sample_pre_morphosyntax.cha    # CHAT entering morphosyntax

  # Align pipeline artifacts (for a file sample.cha)
  sample_utr_input.cha           # CHAT before UTR injection
  sample_utr_tokens.json         # ASR timing tokens fed to UTR
  sample_utr_output.cha          # CHAT after UTR injection
  sample_utr_result.json         # UTR injection statistics
  *-utr-rev-*_rev_evidence.json  # Rev UTR media/request/cache/projection records
  sample_fa_input.cha            # CHAT before FA (after UTR)
  sample_fa_grouping.json        # FA group plan (time windows, words)
  sample_fa_group_0.json         # Per-group words + timings
  sample_fa_group_1.json
  sample_fa_evidence.json        # Versioned score/provenance/decision evidence
  sample_fa_output.cha           # Final aligned CHAT
```

`sample_fa_evidence.json` differs from the older best-effort stage dumps: if
artifact collection was requested and this versioned evidence cannot be
serialized or written, the align job fails with a persistence error. Schema
version 2 records group identity/source/cache keys and pre-injection timing,
model score, complete boundary provenance, and the exact typed decisions that
later altered or removed timing. Decisions remain in the sidecar while CHAT
contains no review-tier projection. The `post_injection_timings` field is
intentionally empty until the complete typed per-word post-processing identity
path is connected.

The fail-closed FA evidence and same-job speaker-turn artifacts are fully
materialized before a temporary file is opened, synchronized, and atomically
renamed into place. On Unix, the containing directory is synchronized too.
When the submitted filename includes directories, the evidence filename adds
a short digest of that full identity after `sample`; this keeps equal basenames
from distinct corpus branches separate.

### What "raw" means for ASR artifacts

`*_asr_response.json` records the tokens as the server received them. For
engines hosted in the Python worker (FunASR SenseVoice and Paraformer,
Tencent, Aliyun, Qwen) that is AFTER the worker's provider adapter, which has
already removed markup, dropped punctuation units and paired each unit with
its own timestamp (see the FunASR section of the ASR token pipeline page). It
is not the engine's untouched output. Rev differs: its provider-shaped
transcript is retained in the Rev evidence cache (see the transcribe guide).

### Environment-variable ASR dumps

Two older diagnostics are enabled by environment variables instead of
`--debug-dir`. The transcribe pipeline reads them in the process that runs
it, which is the server, so set them in the server's environment rather than
the client shell.

| Variable | Writes | On failure |
| --- | --- | --- |
| `BA3_DUMP_ASR_PIPELINE=/path/file.json` | `AsrPipelineTrace`: the worker's tokens, the words after each post-processing stage, and the final utterances. A raw token's `ts` and `end_ts` are written as `null` when the provider supplied no such endpoint, so a consumer has to handle a missing bound rather than reading a zero that was never reported | Serialization and write failures are ignored |
| `BA3_DUMP_UTTERANCES=/path/file.json` | The post-processed utterances handed to CHAT construction | The file fails with a typed diagnostic error |

Each names a single path, so a multi-file job overwrites it once per file
and only the last file's dump survives. Prefer `--debug-dir`, which names
artifacts per input.

### Always-on error logging (no `--debug-dir` needed)

Even without `--debug-dir`, certain failure modes automatically log diagnostic
data at `WARN` level. These are zero-cost in the happy path and fire only when
something goes wrong:

| Failure | What is logged | Where |
|---------|---------------|-------|
| Utseg pre-validation fails (parse error in CHAT) | Full CHAT text + error details | `utseg.rs` |
| Whisper returns inverted timestamps | Warning with start/end values | `inference/asr.py` |
| MOR item count mismatch | Word count + MOR count + utterance text | `inject.rs` |
| Stanza sentence count mismatch | Expected vs actual sentence counts | `morphosyntax.py` |

The utseg CHAT dump is particularly important for transcribe pipelines: if ASR
post-processing produces CHAT that doesn't parse cleanly, the full CHAT text is
logged so you can see exactly which token caused the parse error, without
needing to reproduce the run.

### Example: Diagnosing a transcribe-to-utseg failure

This workflow illustrates the debugging path for a job where transcription
succeeds but utseg rejects the CHAT output (like job `696870c7-02b`,
maria18.wav).

```mermaid
sequenceDiagram
    participant ASR as ASR Inference
    participant PP as Rust Post-Processing
    participant BC as Build CHAT
    participant DD as DebugDumper
    participant UT as Utseg
    participant LOG as Server Logs

    ASR->>PP: raw tokens + timestamps
    PP->>BC: utterances
    BC->>DD: dump_post_asr_chat(chat_text)
    DD-->>DD: write sample_post_asr.cha
    BC->>UT: chat_text
    UT->>DD: dump_pre_utseg_chat(chat_text)
    DD-->>DD: write sample_pre_utseg.cha
    UT->>UT: parse_lenient(chat_text)
    Note over UT: parse error detected!
    UT->>LOG: warn!(chat_text, errors)
    UT-->>BC: Err("utseg pre-validation failed")
```

**Without `--debug-dir`:** check server logs for the `warn!` containing the
full CHAT text and error details.

**With `--debug-dir`:** inspect `sample_post_asr.cha` to see the exact CHAT
that was produced by the transcribe stage. Feed it to the parser locally:

```bash
# Reproduce the parse error offline
cargo run -p talkbank-cli -- validate /tmp/debug/sample_post_asr.cha
```

### Example: Diagnosing an FA grouping issue

```bash
# 1. Run alignment with debug artifacts
batchalign3 align input/ output/ --lang eng --debug-dir /tmp/ba3-debug

# 2. Inspect the UTR input and tokens
cat /tmp/ba3-debug/sample_utr_input.cha
jq . /tmp/ba3-debug/sample_utr_tokens.json

# 3. Write a test that loads the fixtures and calls inject_utr_timing directly
# (no ML model needed, the tokens are already captured)
```

### Implementation details

**`DebugDumper` struct** (`runner/debug_dumper.rs`):

- `new(dir: Option<&Path>)`: enabled dumper or zero-cost no-op
- `disabled()`: test helper, always no-op
- `ensure_dir()`: lazily creates the directory on first write
- `stem(filename)`: extracts file stem for artifact naming
- Each dump method follows the pattern: check `ensure_dir()` → serialize →
  `fs::write()` → log on failure (never panics)

**Threading pattern:**

1. Job options carry `debug_dir: Option<String>` in `CommonOptions`
2. Per-file dispatch extracts it: `job.dispatch.options.common().debug_dir`
3. Creates `DebugDumper::new(debug_dir.as_deref().map(Path::new))`
4. Passes the dumper into the pipeline context struct
5. Stage functions call dump methods at transition points

## Fine-Grained Cache Overrides (`--override-media-cache-tasks`)

For experiment-grade control, `--override-media-cache-tasks` bypasses cache only for
specific NLP tasks:

```bash
# Skip UTR ASR cache but keep the FA cache
batchalign3 align input/ output/ --override-media-cache-tasks utr_asr

# Skip both cache-backed align tasks
batchalign3 align input/ output/ \
  --override-media-cache-tasks utr_asr,forced_alignment

# Refresh Rev words while reusing paid pyannote evidence
batchalign3 transcribe input/ output/ --diarization enabled \
  --override-media-cache-tasks rev_asr_evidence

# Refresh pyannote evidence while reusing Rev words
batchalign3 transcribe input/ output/ --diarization enabled \
  --override-media-cache-tasks speaker_diarization_raw_evidence
```

The cache-backed task names currently exposed by this selective CLI are
`utr_asr`, `forced_alignment`, `rev_asr_evidence`, and
`speaker_diarization_raw_evidence`. Legacy text-task names are accepted with a
warning but do nothing because morphosyntax, utterance segmentation, and
translation are not cached.

The existing `--override-media-cache` continues to skip all cache domains.
Internally, `CacheOverrides::Tasks(BTreeSet<CacheTaskName>)` resolves per-task
via `policy_for(CacheTaskName)`. The typed `FaDispatchPlan` deliberately keeps
FA and UTR policies in distinct fields; both the initial UTR pass and the retry
fallback consume the UTR field rather than borrowing the FA policy. Transcribe
likewise carries a named `TranscribeCachePolicies` record, so Rev and speaker
refresh authority cannot be swapped or coupled through one generic policy.

For replay-only experiments, `--require-media-cache` resolves every
cache-backed task to `CachePolicy::RequireCache`. Missing evidence becomes an
error before an inference capability is constructed. Clap and HTTP admission
both reject combinations with either refresh flag, so a job cannot carry
contradictory replay and refresh instructions.

## Stanza Anomaly Detection

The morphosyntax inference module (`batchalign/inference/morphosyntax.py`)
detects several classes of Stanza misbehavior:

| Anomaly | Detection |
|---------|-----------|
| Bogus lemma | Lemma is pure punctuation for a word with letters (e.g. 哎呀 → 》) |
| Sentence count mismatch | Stanza returned a different number of sentences than input utterances |
| Batch failure | Stanza raised an exception on a batch of items |

When detected, these are logged at `WARNING` level. The bogus-lemma check is
in `_is_bogus_lemma()` and triggers substitution with a `"?"` lemma rather
than propagating the bad value.

## Debugging Async Dispatch with `tokio-console`

Symptoms this tool answers: the Rust dispatch chain is stuck,
`batchalign3` is parked at 0% CPU after a `Starting ASR inference`
log line, no progress, no errors, and the question is "which
async task is blocked, on which resource, for how long?"
`tokio-console` shows the live state of every Tokio task plus the
synchronization primitive each task is waiting on. It complements
`py-spy` (which covers the Python worker side); see
[CPU Profiling](./cpu-profiling.md) for the Python-side recipes.

### Build the debug-runtime binary

`console-subscriber` is gated behind a `debug-runtime` cargo
feature and requires `--cfg tokio_unstable` at rustc time
(the Tokio runtime instrumentation hooks are unstable APIs).
Production binaries built without this feature carry zero cost:
the dep is not linked and no gRPC server starts.

```bash
RUSTFLAGS="--cfg tokio_unstable" \
  cargo build -p batchalign --bin batchalign3 --features debug-runtime
```

Build time on first run includes downloading and compiling the
`console-subscriber` + `tonic` + `prost` dep tree (~3-5 min on a
fast machine, cached thereafter).

### Run the workload and attach

```bash
# Terminal 1: run any batchalign3 command with the debug-runtime binary.
# The console gRPC server starts on 127.0.0.1:6669 at process startup.
./target/debug/batchalign3 transcribe input/ -o out/ --lang yue \
    --asr-engine qwen --engine-overrides '{"qwen_model": "Qwen/Qwen3-ASR-0.6B-hf"}' \
    --sequential --no-server -vv

# Terminal 2: install (once) and attach the TUI client.
cargo install tokio-console
tokio-console http://127.0.0.1:6669
```

### What to look for

The TUI has four primary views:

| View | Use for |
|---|---|
| **Tasks (default `t`)** | List of all live async tasks with state (RUNNING / IDLE / SCHEDULED), `tracing::span` name, busy / idle / poll counts. Look for a task labeled with the request_id of the stuck operation. |
| **Resources (`r`)** | Every `tokio::sync::*` primitive in use: `Mutex`, `Notify`, `Semaphore`, `oneshot::Sender/Receiver`, `mpsc`, `Barrier`. Each row shows how many tasks are waiting on it and for how long. |
| **Task detail (`Enter` on a task)** | Backtrace of the most recent poll, which resource the task is waiting on, what woke it last. |
| **Resource detail (`Enter` on a resource)** | Waiter list, exactly which tasks are blocked on this primitive. |

Built-in **lints** fire automatically in the bottom pane: "task
has been blocked on the same resource for > N seconds", "task is
busy-polling without yielding", "many tasks waiting on a single
`Mutex`". For the qwen dispatch hang investigation, the relevant
lint signature was "task blocked on `oneshot::Receiver` for > 30s"
, would fire within the first minute of any reproduction.

### Span naming convention

For tokio-console to label tasks meaningfully, the async functions
on the dispatch chain are annotated with `#[tracing::instrument]`:

| Span name | File | Carries |
|---|---|---|
| `dispatch_execute_v2_with_progress` | `worker/pool/dispatch.rs` | `request_id` |
| `dispatch_gpu_execute_v2` | `worker/pool/dispatch.rs` | `target`, `lang`, `request_id` |
| `get_or_create_gpu_worker` | `worker/pool/mod.rs` | `target`, `lang` |
| `execute_v2` | `worker/pool/shared_gpu/stdio.rs` | `pid`, `request_id` |
| `shared_gpu_reader_loop` | `worker/pool/shared_gpu/stdio.rs` | `pid` |
| `write_request` / `read_response` | `worker/handle/ipc.rs` | `pid` |

When adding new dispatch surface, add `#[instrument(skip_all,
fields(...))]` with the same convention so the TUI labels stay
useful. `skip_all` is important, the default `#[instrument]`
captures every argument's `Debug` impl, which is too noisy for
large request payloads.

### Attaching from a different host

The gRPC server binds to `127.0.0.1:6669` by default, localhost
only. To attach from a workstation to a fleet host's batchalign3
process, SSH-forward the port:

```bash
ssh -L 6669:127.0.0.1:6669 operator@server
# then locally:
tokio-console http://127.0.0.1:6669
```

### When NOT to use it

- **Production binaries.** The `tokio_unstable` cfg couples to
  non-stable Tokio APIs; the gRPC server adds a real dep tree;
  neither is appropriate for production observability. Use the
  existing OpenTelemetry / `tracing-appender` server-log surface
  for production.
- **Memory or UB bugs.** Wrong category. Use `memray` / `dhat-rs`
  / `miri` instead.
- **Subprocess IPC bugs in isolation.** `tokio-console` sees the
  Rust side only. A bug that crosses into the Python worker also
  needs `py-spy dump` on the worker pid, both views together
  cover the dispatch chain end-to-end.

## Debugging Python workers with `py-spy`

See [CPU Profiling](./cpu-profiling.md) for the full reference.
Quick recipes:

```bash
sudo py-spy dump --pid <worker-pid>           # one-shot stack of every thread
sudo py-spy top --pid <worker-pid>            # live top
sudo py-spy record --native --subprocesses \
    --pid $(pgrep -f batchalign3) -o flame.svg  # flame graph
```

`py-spy dump` is the first thing to try when a Python worker is
hung at 0% CPU, replaces the old "tail the log and guess"
pattern.
