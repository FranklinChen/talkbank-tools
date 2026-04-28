# Python/Rust Interface Reference

**Status:** Current
**Last modified:** 2026-04-02 20:47 EDT

This document describes the `batchalign_core` PyO3 module — the Rust worker
runtime extension that Python ML workers use.

## Overview

`batchalign_core` is a slim PyO3 extension (`pyo3/`, ~2,950 lines) that serves
one purpose: bridging the Rust server with Python ML workers. It exposes:

1. **Worker protocol dispatch** — routes IPC messages to typed handlers
2. **Worker V2 executors** — load prepared artifacts, call ML models, return raw results
3. **HK/Cantonese bridges** — project provider-specific ASR output into common shapes
4. **Utilities** — token alignment, Cantonese normalization, Rev.AI HTTP client
5. **CLI entry** — console_scripts bridge for the PyPI package

Python never touches raw CHAT text. All parsing, mutation, validation, and
serialization happens in the Rust server (`crates/batchalign-app/`) using
`batchalign-chat-ops` directly.

## Architecture

```
Rust Server (crates/batchalign-app/)
  ├── Parses CHAT, extracts payloads, checks cache
  ├── Sends IPC request to Python worker (stdio JSON-lines)
  │
  └── Python Worker Process
        ├── worker_protocol.rs   →  dispatch IPC messages
        ├── worker_*_exec.rs     →  load prepared artifacts, call ML model
        ├── hk_asr_bridge.rs     →  project HK provider output
        ├── align_tokens()       →  Stanza token realignment
        └── Returns raw results  →  Rust server injects into CHAT AST
```

The Rust server owns the entire CHAT lifecycle: parse → cache lookup → extract
payloads → dispatch to worker → receive raw results → normalize → inject into
AST → validate → serialize.

## De-Pythonization boundary

Python stays only for direct ML/library calls and the thinnest worker-side glue:

### Must stay Python

| Surface | Why |
|---|---|
| `batchalign/worker/` | Thin worker host for Python-native ML runtimes |
| `batchalign/inference/` | Direct model or SDK invocation (Stanza, Whisper, pyannote, etc.) |
| `batchalign/inference/hk/` | Python-only HK/Cantonese SDK and model boundaries |
| `batchalign/models/` | Training code depending on Python ML libraries |

Number expansion was moved entirely into Rust on 2026-04-26 (see
`book/src/architecture/number-expansion.md`); the Python
`_number_expansion.py` and `_expand_numbers_v2.py` modules and the
`expand_numbers` V2 IPC were removed.

### Removed (2026-03-21)

| Surface | Why removed |
|---|---|
| `ParsedChat` class + callback methods | Rust server uses `ChatFile` directly |
| `batchalign.pipeline_api` | Rust server owns pipeline orchestration |
| `batchalign.compat` | Deprecated BA2 shim, no longer needed |
| `batchalign.inference.benchmark` | WER scoring available via `batchalign3 compare` |
| Standalone `#[pyfunction]`s (build_chat, WER, extraction, etc.) | Server calls `batchalign-chat-ops` directly |

## Worker Protocol

Workers communicate via stdio JSON-lines. Each message has an `op` field:

| Op | Handler | Description |
|----|---------|-------------|
| `health` | Python | Worker health status |
| `capabilities` | Python | Available commands and tasks |
| `infer` / `batch_infer` | Python | Legacy inference (morphosyntax, utseg, etc.) |
| `execute_v2` | Rust dispatcher → Python model | Typed V2 execution with prepared artifacts |
| `shutdown` | Protocol | Clean worker shutdown |

`dispatch_protocol_message()` validates the JSON envelope in Rust, then calls
the appropriate Python handler.

## Worker V2 Executors

Each executor loads Rust-prepared artifacts from the IPC message, calls the
Python ML model, and returns raw results:

| Executor | Task | What Rust prepares | What Python does |
|----------|------|-------------------|-----------------|
| `execute_asr_request_v2` | ASR | PCM audio bytes | Run Whisper / HK provider |
| `execute_forced_alignment_request_v2` | FA | PCM audio + word JSON | Run Whisper/Wave2Vec FA |
| `execute_speaker_request_v2` | Speaker | PCM audio bytes | Run pyannote/NeMo |
| `execute_opensmile_request_v2` | OpenSMILE | PCM audio bytes | Extract acoustic features |
| `execute_avqi_request_v2` | AVQI | Paired audio bytes | Calculate voice quality |
| `normalize_text_task_result` | Text tasks | N/A | Reshape BatchInferResponse → V2 types |

## HK/Cantonese Bridges

Python HK engines call back into Rust for output projection:

| Function | Purpose |
|----------|---------|
| `funaudio_segments_to_asr` | Project FunASR segments → monologues + timed words |
| `tencent_result_detail_to_asr` | Project Tencent output → monologues + timed words |
| `aliyun_sentences_to_asr` | Project Aliyun output → monologues + timed words |
| `normalize_cantonese` | Simplified → HK traditional + domain replacements |
| `cantonese_char_tokens` | Per-character tokenization for Cantonese FA |

## Rev.AI Native HTTP Client

The shared Rust crate `crates/batchalign-revai/` provides Rev.AI HTTP calls.
The Rust server uses this crate directly for all Rev.AI operations (transcribe,
UTR, pre-submission). No Rev.AI functions are exposed to Python — the PyO3
wrappers were removed as dead code (2026-03-21).

## Module Layout (`pyo3/src/`)

```
lib.rs                  — module registration (~80 lines)
worker_protocol.rs      — IPC message dispatch
worker_asr_exec.rs      — ASR execution (Whisper, HK providers)
worker_fa_exec.rs       — forced alignment execution
worker_media_exec.rs    — speaker diarization, OpenSMILE, AVQI
worker_text_results.rs  — text task normalization + align_tokens
worker_artifacts.rs     — prepared artifact loading from IPC
hk_asr_bridge.rs        — HK/Cantonese provider projection + normalization
py_json_bridge.rs       — Python→JSON conversion utility
```

## GIL Strategy

All pure-Rust functions use `py.detach()` (PyO3 0.28) to release the GIL during
computation. The worker executors hold the GIL only during Python model
invocation.
