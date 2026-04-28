# Rust Core (batchalign_core)

**Status:** Current
**Last modified:** 2026-03-21 14:47 EDT

For new contributors, start with:

- [Rust Contributor Onboarding](rust-contributor-onboarding.md)
- [Rust Workspace Map](rust-workspace-map.md)

## Repository Structure

The PyO3 bridge lives in `pyo3/` within the batchalign3 repository. This is a
single-crate project (`batchalign-pyo3`) that builds the `batchalign_core`
Python extension module. It depends on `batchalign-chat-ops` and
`batchalign-types` (same workspace). It does **not** depend on `talkbank-model`
or `talkbank-parser` — all CHAT manipulation is handled by the Rust server.

The default editable build is intentionally slim. It builds only the extension
surface, leaving the embedded CLI bridge and Rev.AI bridge for the full
packaged profile. In a source checkout, the Python console-script wrapper can
still run `batchalign3` by falling back to the repo CLI.

| Crate | Location | Purpose |
|-------|----------|---------|
| `batchalign-pyo3` | `pyo3/src/` | Worker runtime — what Python imports as `batchalign_core` |
| `batchalign-chat-ops` | `crates/batchalign-chat-ops/` | CHAT lifecycle: parsing, injection, extraction, alignment, post-processing |
| `batchalign-types` | `crates/batchalign-types/` | Shared domain types, worker IPC types |
| `batchalign-app` | `crates/batchalign-app/` | Rust server — owns all CHAT orchestration |

## Module Organization

The PyO3 crate (~2,950 lines) is a slim worker runtime. All CHAT manipulation
lives in `batchalign-chat-ops` and is called directly by the Rust server.

| Module | Purpose |
|--------|---------|
| `lib.rs` | Module registration (~80 lines) |
| `worker_protocol.rs` | IPC message dispatch (health, capabilities, infer, execute_v2) |
| `worker_asr_exec.rs` | ASR execution (Whisper, HK providers) |
| `worker_fa_exec.rs` | Forced alignment execution |
| `worker_media_exec.rs` | Speaker diarization, OpenSMILE, AVQI |
| `worker_text_results.rs` | Text task normalization + `align_tokens` |
| `worker_artifacts.rs` | Prepared artifact loading from IPC attachments |
| `hk_asr_bridge.rs` | HK/Cantonese provider projection + normalization |
| `py_json_bridge.rs` | Python→JSON conversion utility |

## Key PyO3 Entry Points

### Worker Protocol

- `dispatch_protocol_message(...)` — route IPC messages to typed Python handlers

### Worker V2 Executors

| Function | Purpose |
|----------|---------|
| `execute_asr_request_v2(...)` | Load prepared audio, call Whisper/HK provider |
| `execute_forced_alignment_request_v2(...)` | Load prepared audio+text, call FA model |
| `execute_speaker_request_v2(...)` | Load prepared audio, call pyannote/NeMo |
| `execute_opensmile_request_v2(...)` | Load prepared audio, extract acoustic features |
| `execute_avqi_request_v2(...)` | Load paired audio, calculate voice quality |
| `normalize_text_task_result(...)` | Reshape BatchInferResponse → typed V2 results |

### Utilities

| Function | Purpose |
|----------|---------|
| `align_tokens(...)` | Map Stanza tokenizer output back to CHAT words |
| `normalize_cantonese(...)` | Simplified → HK traditional + domain replacements |
| `cantonese_char_tokens(...)` | Per-character tokenization for Cantonese FA |
| HK bridge functions | Project FunASR/Tencent/Aliyun output into common shapes |

### Removed APIs (2026-03-21)

The following were removed when the Rust server made them redundant:

- `ParsedChat` class and all callback methods (parse, serialize, add_morphosyntax, add_forced_alignment, etc.)
- `run_provider_pipeline()` and provider pipeline helpers
- Standalone functions: `build_chat`, `parse_and_serialize`, `extract_nlp_words`, `wer_compute`, `wer_metrics`, `dp_align`, etc.
- All inner function modules: `morphosyntax_ops`, `fa_ops`, `text_ops`, `speaker_ops`, `cleanup_ops`, `tier_ops`

All domain logic is now tested in `batchalign-chat-ops` (548 inline tests) and exercised through the Rust server.

## Tree-Sitter Grammar

The CHAT grammar definition lives in the sibling `talkbank-tools` repo at
`talkbank-tools/grammar/`. After editing it, regenerate the C parser:

```bash
cd ../talkbank-tools/grammar && tree-sitter generate
```

This regenerates `parser.c`, which the tree-sitter parser depends on.
**Forgetting this step causes the parser to use a stale grammar.**

After grammar changes, always test against real corpus data in addition to
the curated test suite.

## Building for Development

When you change the PyO3 bridge or the shared `batchalign-chat-ops` logic that
Python consumes, rebuild the extension with the repo-native command:

```bash
make build-python
```

This runs `uv run maturin develop -m pyo3/Cargo.toml --no-default-features -F
pyo3/extension-module` under the hood.

If you need to verify the packaged install surface instead of the fast dev
loop, run:

```bash
make build-python-full
```

If you also plan to validate the standalone Rust CLI directly after a shared
crate change, rebuild that binary too:

```bash
cargo build -p batchalign-cli
```

## Running Rust Tests

```bash
cargo nextest run --manifest-path pyo3/Cargo.toml
```

To run the full parser integration test suite (both parsers), use the
`talkbank-tools` workspace:

```bash
cargo nextest run --manifest-path ../talkbank-tools/Cargo.toml -p talkbank-parser-tests
```

## GIL Release Strategy

All pure-Rust `batchalign_core` methods release the Python GIL via
`py.detach()` (pyo3 0.28). This allows Python threads to run concurrently
while Rust does heavy computation.

Callback-based methods hold the GIL only during the Python callback
invocation. The pattern is:

1. Release GIL, walk the Rust AST, collect data.
2. Acquire GIL, call the Python callback.
3. Release GIL, inject results back into the AST.

CPU-bound Rust work does not block other Python threads, while the callback
(which runs Python model inference) holds the GIL as expected.

## Workflow: Adding a New CHAT Transformation

1. Add the Rust function in `batchalign-pyo3` — as `#[pymethods]` on
   `ParsedChat` (in the appropriate domain file under `parsed_chat/`) or as
   a standalone `#[pyfunction]` in `pyfunctions.rs`.
2. Add the inner logic in the corresponding `*_ops.rs` module.
3. Write Rust tests.
4. Regenerate the grammar if you changed `grammar.js`.
5. Rebuild `batchalign_core` (`make build-python` for the fast loop, or
   `make build-python-full` when you need the embedded CLI bridge too).
6. Call the new function from Python — either through the worker dispatch
   (`batchalign/worker/_infer.py`) or through `pipeline_api.py` operations.
7. Test against real corpus data, not just unit tests.
