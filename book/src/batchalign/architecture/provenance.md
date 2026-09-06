# Processing Provenance System

**Status:** Current
**Last updated:** 2026-09-05 21:58 EDT

## Overview

The provenance system injects structured `@Comment` headers into CHAT
files recording what batchalign3 did, when, and with what engines. This
enables reproducibility, auditing, and UI display of processing history.

## Architecture

```mermaid
flowchart LR
    subgraph "Pipeline (Rust)"
        A["Parse CHAT\n(parse_lenient)"] --> B["Process\n(infer/inject)"]
        B --> C["inject_provenance()\n(provenance.rs)"]
        C --> D["Serialize\n(to_chat_string)"]
    end
    subgraph "Worker"
        W["engine_versions\n(live detection)"]
    end
    W -->|"engine version"| C
```

### Source: `crates/batchalign/src/provenance.rs`

The module provides:

- **`ProvenanceComment`**: typed builder for provenance metadata
- **`inject_provenance(&mut ChatFile, &ProvenanceComment)`**: AST-level
  injection that adds/replaces `@Comment` headers
- **`inject_provenance_into_text(&str, &ProvenanceComment) -> Result<String, ParseErrors>`**: strictly parses serialized output before adding a comment; parser recovery errors return diagnostics instead of a rewritten document
- **Per-command builders**: `morphotag_provenance()`,
  `align_provenance()`, `transcribe_provenance()`, etc.

### Comment Format

```text
[ba3 <command> | key=val ; key=val | ISO-8601-timestamp]
```

The `[ba3 ` prefix is the machine-parseable discriminator. The
bracketed format is visually distinct from user-authored comments and
greppable with `grep '\[ba3 '`.

### AST Manipulation (not string hacking)

Provenance is injected into the CHAT AST, not the serialized text:

1. Existing `@Comment` with matching `[ba3 <command> |` prefix is removed
   from `ChatFile.lines`
2. A new `Line::Header { Header::Comment { BulletContent } }` is inserted
   after the last constant participant header (`@ID`, `@Birth of`,
   `@Birthplace of`, or `@L1 of`)
3. The file is then serialized normally via `to_chat_string()`

This ensures provenance comments participate in proper CHAT serialization
(bullet handling, line wrapping, encoding).

### Re-export chain

The provenance module needs `Header`, `BulletContent`, and `Span` from
`talkbank-model`. Since `batchalign` doesn't depend on
`talkbank-model` directly, these types are re-exported through
`batchalign`:

```rust,ignore
talkbank-model::header::Header      →  batchalign::Header
talkbank-model::model::BulletContent →  batchalign::BulletContent
talkbank-model::Span                →  batchalign::Span
```

## Injection Points

Each pipeline injects provenance right before serialization:

| Command | File | Injection site |
|---------|------|---------------|
| morphotag | `pipeline/morphosyntax.rs` | `stage_serialize()`: injects into `&mut ChatFile` before `to_chat_string()` |
| utseg | `pipeline/text_infer.rs` | `run_cached_text_pipeline()`: after `apply()`, before `to_chat_string()` |
| translate | `pipeline/text_infer.rs` | Same as utseg (shared generic pipeline) |
| coref | `coref.rs` | `run_coref_impl()`: after injection, before serialize |
| align | `runner/dispatch/fa_pipeline.rs` | `AlignAudioTask::finalize_success()`: after FA result, uses fallible `inject_provenance_into_text()` |
| transcribe | `pipeline/transcribe.rs` | serialization stage: injects structured provenance and the unchecked-ASR warning through the production AST helpers |

## Engine Version Source

Engine versions come from `WorkerCapabilities.engine_versions`: a
`BTreeMap<String, String>` reported by each Python worker at spawn time.
This is live detection, not hardcoded constants.

The map is surfaced through:
- `PipelineServices.engine_version` (for morphotag/utseg/translate)
- `EngineVersion` newtype in the FA dispatch plan
- Direct backend enum matching for transcribe

Example live values:
```json
{
  "morphosyntax": "1.11.1",
  "fa": "whisper-fa-large-v2",
  "asr": "rev",
  "utseg": "1.11.1",
  "coref": "stanza",
  "translate": "googletrans-v1"
}
```

## Replacement Semantics

When the same command is run again on the same file:

1. `inject_provenance()` scans all `Line::Header` entries
2. Any `Header::Comment` whose `BulletContent` text starts with
   `[ba3 <command> |` is removed
3. The new comment is inserted after the last constant participant header

This means re-running morphotag replaces the morphotag comment but
preserves any align or transcribe comments. The processing history
accumulates across different commands but doesn't duplicate within
one command.

## Human-readable unchecked-ASR warning

Transcribe writes two comments for different audiences:

- `[ba3 transcribe | ...]` is machine-readable processing provenance; and
- the warning below is a human-visible safety statement.

```text
@Comment:	Batchalign 0.3.0, ASR Engine rev. Unchecked output of ASR model, DO NOT USE.
```

`inject_unchecked_warning()` owns this production path. It replaces any prior
unchecked-ASR warning, writes the compiling product version and actual ASR
engine identity, includes the mandated `DO NOT USE`, and uses the same
constant-header-aware AST insertion point as structured provenance.

## Regression coverage

Tests beside `provenance.rs` cover deterministic formatting, replacement,
cross-command preservation, constant-header ordering, extraction, no-op write
detection, the product-version stamp, and the explicit safety warning. Malformed
main tiers and generated morphosyntax tiers are refused at the serialized-output
boundary. The dispatcher preserves these diagnostics as `ServerError::OutputParse`,
a system failure rather than a successful file or a bad-input response. The ASR
backend matrix in `transcribe/mod.rs` proves that Rev, Whisper variants,
Tencent, Aliyun, Funaudio, and Qwen retain distinct provenance names. There is
no test-only comment implementation: tests exercise the production builder and
AST injection path.
