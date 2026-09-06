# Python/Rust Interface Map

**Status:** Current
**Last updated:** 2026-09-06 08:27 EDT

This document is the unified reference for all Python/Rust interface boundaries in batchalign3.

## Interface Boundaries

### 1. Worker Protocol Dispatch (`pyo3-protocol`)
**Purpose:** IPC message routing and envelope validation

| Aspect | Location |
|--------|----------|
| Rust source of truth | `crates/batchalign-pyo3/src/worker_protocol.rs` |
| Python implementation | `batchalign/worker/_protocol.py` + `batchalign/worker/_handlers.py` |
| Shared schema | `ipc-schema/worker_v2/` (JSON Schema) |
| IPC types | `crates/batchalign-types/src/worker_v2/mod.rs` |
| Python types | `batchalign/worker/_types_v2.py` (hand-written + validated) |
| Tests | `crates/batchalign/tests/worker_protocol_v2_compat.rs`, `batchalign/tests/test_ipc_type_conformance.py` |

**Cross-references:**
- Rust: `crates/batchalign-pyo3/src/worker_protocol.rs` line 1 onward references `ipc-schema/worker_v2/`
- Python: `batchalign/worker/_protocol.py` implements protocol handlers
- See architecture docs: `book/src/batchalign/developer/worker-protocol-v2.md`

**Responsibility:**
- **Rust controls:** Message validation, routing, envelope structure, error codes
- **Python implements:** Handler dispatch, message logging, resource cleanup

Rust `prepare_protocol_message` returns either a constructor-private
`PendingProtocolRequest` or an `ImmediateProtocolReply`. The pending operation
vocabulary excludes shutdown. Native `dispatch_protocol_message` accepts only
that pending type and returns a response payload; the former always-false
shutdown return flag is gone. Python's synchronous wrapper still accepts raw
messages, while concurrent readers handle immediate replies before queueing.
The canonical stub is `stubs/batchalign_core/__init__.pyi`.

Concurrent stdio obtains `ProtocolStdin` from native `open_protocol_stdin`
(`worker_stdio.rs`). Its bounded mailbox owns line delivery, EOF, read failures,
and stopping; `stop()` wakes a blocked Python `read_line()` without requiring
parent EOF. EOF drains queued requests, whereas stopping cancels queued work.
The sole native OS reader holds no Python objects and is process-owned; shutdown
does not join an uncancellable OS stdin read.

---

### 2. ASR Execution V2 (`pyo3-asr`)
**Purpose:** Audio-to-text transcription via Whisper or HK providers

| Aspect | Location |
|--------|----------|
| Rust FFI | `crates/batchalign-pyo3/src/worker_asr_exec.rs::execute_asr_request_v2()` |
| Rust request type | `crates/batchalign-types/src/worker_v2/requests.rs::AsrRequestV2` |
| Rust response type | `crates/batchalign-types/src/worker_v2/responses.rs::ExecuteResponseV2` |
| Python executor | `batchalign/worker/_asr_v2.py::execute_asr_request_v2()` |
| Python host | `batchalign/worker/_asr_v2.py::AsrExecutionHostV2` |
| Whisper bridge | `batchalign/inference/asr.py` (local model) |
| HK providers | `batchalign/inference/languages/cantonese/` (Tencent, FunASR, Aliyun) |
| Schema | `ipc-schema/worker_v2/AsrRequestV2.json`, `ipc-schema/worker_v2/ExecuteResponseV2.json` |

**Rust/Python contract:**
- Input: `AsrRequestV2` with `prepared_audio` reference + metadata
- Output: `ExecuteResponseV2` with ASR results or error
- Side-effects: None (stateless)

**Cross-references:**
- Python calls Rust-exposed `batchalign_core.execute_asr_request_v2()`
- Rust prepares audio bytes; Python loads from attachment refs
- See: `book/src/batchalign/architecture/asr-token-pipeline.md`

**Responsibility:**
- **Rust owns:** Audio preparation, artifact management, result type structure
- **Python owns:** Model loading, inference execution, error reporting to Rust

---

### 3. Forced Alignment V2 (`pyo3-fa`)
**Purpose:** Word-level timing alignment via Whisper or Wave2Vec

| Aspect | Location |
|--------|----------|
| Rust FFI | `crates/batchalign-pyo3/src/worker_fa_exec.rs::execute_forced_alignment_request_v2()` |
| Rust request type | `crates/batchalign-types/src/worker_v2/requests.rs::ForcedAlignmentRequestV2` |
| Python executor | `batchalign/worker/_fa_v2.py::execute_forced_alignment_request_v2()` |
| Schema | `ipc-schema/worker_v2/ForcedAlignmentRequestV2.json` |

**Rust/Python contract:**
- Input: `ForcedAlignmentRequestV2` with prepared audio + normalized text
- Output: Word timing results or error
- Side-effects: None

**Cross-references:**
- Rust server caches FA results; Python is stateless

**Responsibility:**
- **Rust owns:** Text normalization, cache eviction, timing post-processing
- **Python owns:** Alignment model execution

---

### 4. Media Analysis V2: OpenSMILE (`pyo3-media-opensmile`)
**Purpose:** Acoustic feature extraction

| Aspect | Location |
|--------|----------|
| Rust FFI | `crates/batchalign-pyo3/src/worker_media_exec.rs::execute_opensmile_request_v2()` |
| Python executor | `batchalign/worker/_opensmile_v2.py::execute_opensmile_request_v2()` |
| Schema | `ipc-schema/worker_v2/OpenSmileRequestV2.json` |

**Rust/Python contract:**
- Input: `OpenSmileRequestV2` with prepared audio
- Output: Numeric feature vectors
- Side-effects: None

---

### 5. Media Analysis V2: AVQI (`pyo3-media-avqi`)
**Purpose:** Voice quality analysis (paired audio comparison)

| Aspect | Location |
|--------|----------|
| Rust FFI | `crates/batchalign-pyo3/src/worker_media_exec.rs::execute_avqi_request_v2()` |
| Python executor | `batchalign/worker/_avqi_v2.py::execute_avqi_request_v2()` |
| Schema | `ipc-schema/worker_v2/AvqiRequestV2.json` |

**Rust/Python contract:**
- Input: `AvqiRequestV2` with paired audio references
- Output: Quality score + metrics
- Side-effects: None

---

### 6. Media Analysis V2: Speaker Diarization (`pyo3-media-speaker`)
**Purpose:** Speaker boundary and identity detection

| Aspect | Location |
|--------|----------|
| Rust FFI | `crates/batchalign-pyo3/src/worker_media_exec.rs::execute_speaker_request_v2()` |
| Python executor | `batchalign/worker/_speaker_v2.py::execute_speaker_request_v2()` |
| Schema | `ipc-schema/worker_v2/SpeakerRequestV2.json` |

**Rust/Python contract:**
- Input: `SpeakerRequestV2` with prepared audio
- Output: backend-tagged evidence: completed pyannoteAI job output or local
  Pyannote/NeMo segments
- Side-effects: pyannoteAI requests upload temporary media and submit a paid
  provider job; local backends have no external service side effect

---

### 7. Text Task Execution V2 (`pyo3-text`)
**Purpose:** Rust control plane for the four batched text tasks (morphosyntax, utseg, translate, coref)

| Aspect | Location |
|--------|----------|
| Rust FFI | `crates/batchalign-pyo3/src/worker_text_exec.rs::execute_{morphosyntax,utseg,translate,coref}_request_v2()` |
| Result normalization | `crates/batchalign-pyo3/src/worker_text_results.rs::normalize_*_result()` (Rust-internal, called by the executors) |
| Rust FFI | `crates/batchalign-pyo3/src/worker_text_results.rs::align_tokens()` (tokenizer realignment, separate concern) |
| Python caller | `batchalign/worker/_text_v2.py::execute_*_request_v2()` |
| Schema | Responds with V2 ExecuteResponseV2 |

**Rust/Python contract (since 2026-08-21):** Rust owns request parsing,
task/payload agreement, prepared-batch loading and item-count checks, the
failure taxonomy, and result normalization into the typed V2 payloads. Python
keeps the host of loaded model runners plus one adapter per task, which
parses the frozen batch into typed batch items, builds `BatchInferRequest`,
manages the morphosyntax progress callback, and calls the model.

---

### 8. Worker Artifact Loading (`pyo3-artifacts`)
**Purpose:** Resolve and load Rust-prepared artifacts (audio bytes, JSON metadata)

| Aspect | Location |
|--------|----------|
| Rust-internal loaders | `crates/batchalign-pyo3/src/worker_artifacts.rs` |
| Consumers | the Rust executors in `worker_{asr,fa,media,text}_exec.rs` |

**Functions (Rust-internal since 2026-08-21; the FFI loader pyfunctions were
deleted when their last Python caller, the old text control plane, was
ported):**
- `find_attachment()`: locate artifact by ID
- `load_json_attachment_text()`: JSON batch payload as frozen text
- `load_prepared_text_json_impl()`: load prepared text payload
- `require_mono_prepared_audio()`: mono-checked audio proof (`MonoPreparedAudio`)

**Design:**
- Artifacts are **file-backed** in a per-worker temp directory
- IPC message contains artifact references (`ArtifactRefV2`)
- The Rust executors resolve refs and load bytes or JSON; Python receives typed arguments
- Future: Can migrate to shared memory without changing Python logic

---

### 9. HK/Cantonese ASR Bridges (`pyo3-hk`)
**Purpose:** Project provider-specific ASR output (FunASR, Tencent, Aliyun) into common shapes

| Aspect | Location |
|--------|----------|
| Rust FFI functions | `crates/batchalign-pyo3/src/cantonese_asr_bridge.rs` (6 functions) |
| Python callers | `batchalign/inference/languages/cantonese/`, `batchalign/worker/_asr_v2.py` |

**Functions:**
- `funaudio_segments_to_asr()`: project FunASR segment output
- `tencent_result_detail_to_asr()`: project Tencent API response
- `aliyun_sentences_to_asr()`: project Aliyun API response
- `normalize_cantonese()`: simplified ↔ traditional + domain replacements
- `cantonese_char_tokens()`: per-character tokenization for FA
- `clean_funaudio_segment_text()`: text cleanup before normalization

**Design:** These bridge the gap between provider-native formats and Batchalign `MonologueAsrResultV2`.

**Responsibility:**
- **Rust owns:** Output shape (`MonologueAsrResultV2`), char tokenization
- **Python owns:** Provider SDK invocation, raw response parsing

---

### 10. Worker V2 IPC Schema (`worker-v2-schema`)
**Purpose:** Single source of truth for all IPC message types

| Aspect | Location |
|--------|----------|
| Rust types | `crates/batchalign-types/src/worker_v2/` |
| Re-exported by | `crates/batchalign/src/types/worker_v2.rs` |
| JSON Schema (generated) | `ipc-schema/worker_v2/` |
| Python hand-written | `batchalign/worker/_types_v2.py` (with validators) |
| Conformance tests | `crates/batchalign/tests/worker_protocol_v2_compat.rs`, `batchalign/tests/test_ipc_type_conformance.py` |

**Sync process:**
1. Modify Rust struct + add `#[derive(schemars::JsonSchema)]`
2. Register in `crates/batchalign/src/ipc_schema.rs`
3. Run `cargo run -p batchalign -- ipc-schema --output ipc-schema/`
4. Run `bash scripts/generate_ipc_types.sh` to update Python
5. Conformance tests verify hand-written Python types match schema
6. CI gate: `bash scripts/check_ipc_type_drift.sh`

**See:** `book/src/batchalign/developer/ipc-type-sync.md`

---

### 11. Rust/Python Type Sync System (`types-sync`)
**Purpose:** Ensure Rust and Python types stay synchronized

| Aspect | Location |
|--------|----------|
| Rust types | `crates/batchalign-types/src/worker_v2/` |
| Schema generator | `crates/batchalign/src/ipc_schema.rs` |
| Hand-written overlays | `batchalign/worker/_types_v2.py` (validators, aliases) |
| Drift tests | `crates/batchalign/tests/worker_protocol_v2_compat.rs` |
| Conformance tests | `batchalign/tests/test_ipc_type_conformance.py` |
| Scripts | `scripts/generate_ipc_types.sh`, `scripts/check_ipc_type_drift.sh` |

**Invariant:** All Rust struct → Python model mappings are validated at test time.

---

## Documentation Map

### Architecture (Decision Record)
- `book/src/batchalign/architecture/server-architecture.md`: Server-side orchestration, job lifecycle

### Developer Reference (Implementation Spec)
- `book/src/batchalign/developer/worker-protocol-v2.md`: IPC protocol, envelope types, V1/V2 migration status
- `book/src/batchalign/developer/ipc-type-sync.md`: Type generation pipeline, conformance testing, adding new types
- `book/src/batchalign/developer/maturin-pyo3-surface.md`: PyO3 wheel packaging, editable installs

### Crate Documentation (Source Code)
- `crates/batchalign-pyo3/CLAUDE.md`: Pyo3 crate architecture, standards, rules
- `crates/batchalign/CLAUDE.md`: Server module map, job lifecycle, concurrency model
- `crates/batchalign-types/CLAUDE.md`: Domain types, newtype conventions (if exists)

### In-Code Documentation
- `crates/batchalign-pyo3/src/lib.rs`: Module registration, what each function does
- `crates/batchalign-pyo3/src/worker_protocol.rs`: IPC dispatch logic
- `crates/batchalign-pyo3/src/worker_*_exec.rs`: Per-task executors
- `batchalign/worker/_execute_v2.py`: Python dispatch logic
- `batchalign/worker/_*_v2.py`: Per-task executors

---

## When Adding a New Interface

1. **Define Rust types** with `#[derive(schemars::JsonSchema)]`
2. **Register in schema generator** (`crates/batchalign/src/ipc_schema.rs`)
3. **Generate schemas** (`cargo run -p batchalign -- ipc-schema --output ipc-schema/`)
4. **Generate Python** (`bash scripts/generate_ipc_types.sh`)
5. **Add conformance test** in `batchalign/tests/test_ipc_type_conformance.py`
6. **Document in this file** (boundary name, locations, responsibility split)
7. **Add doc comment to Rust FFI** pointing to this file
8. **Add doc comment to Python** pointing to Rust source and this file

---

## Checklist: Are Docs Synced?

For each boundary, verify:

- [ ] Rust source has doc comments explaining contract (input, output, side-effects)
- [ ] Python source has doc comments explaining contract
- [ ] Both reference each other (cross-links in comments)
- [ ] Shared schema/types are generated and conformance-tested
- [ ] This Interface Map lists the boundary with both source locations
- [ ] mdbook builds without warnings about missing links
- [ ] Running `mdbook build batchalign-book` passes
