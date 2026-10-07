# CHAT Parsing (Rust)

**Status:** Current
**Last updated:** 2026-10-02 20:51 EDT

All CHAT parsing and serialization is handled by Rust. The CHAT lifecycle
(parsing, word extraction, result injection, validation, serialization)
runs in Rust both on the server side (`batchalign` crate) and
through the PyO3 bridge (`batchalign_core`). Python handles only ML
inference, Whisper, Stanza, wav2vec, translation models.

## Architecture

The required CHAT lifecycle is:

```text
CHAT text (.cha file)
  │
  ▼
Chatter parser → complete, command-bound admission → typed CHAT AST
  │
  ├── extract words/payloads   → structured data for ML inference
  ├── inject %mor/%gra         → from morphosyntax results
  ├── inject word timings       → from forced alignment results
  ├── inject %xtra             → from translation results
  ├── split utterances          → from utseg results
  │
  ▼
to_chat_string() / handle.serialize()
  │
  ▼
CHAT text (only after checked application and output admission)
```

These operations happen in the Rust server crate (`batchalign`) using
functions from `batchalign`. Python workers provide only raw ML
inference results; the Rust server handles all CHAT parsing, mutation, and
serialization.

## What Rust Does

### Parsing

Two parse modes:

- **`parse(text)`** -- strict mode. Rejects files with parse errors.
- **`parse_lenient(text)`** -- recovery/inspection mode. Recovery does not
  authorize inference or publication. Its historical `%mor`/`%gra` diagnostic
  exemption is inappropriate for workflows that retain those tiers.

The shared `utseg`/`translate` pipeline and the separate `coref` pipeline use Chatter's
`parse_validated_with_parser`, obtaining an immutable `ValidChatFile` before
collecting payloads. Editing consumes that proof. See the authoritative
[command contracts](command-contracts.md) for regeneration exemptions and
the current migration scope.

Both parsers use the tree-sitter grammar owned by Chatter
to produce a concrete syntax tree, which is then walked into typed Rust model
structures (`ChatFile`, `MainTier`, `WorTier`, `Terminator`, etc.).

### NLP Word Extraction

`extract_words()`
(`../chatter/crates/talkbank-transform/src/extract.rs:48`) walks the AST and
produces an ordered list of "NLP-clean" words with node indices.  It
skips retraces, events, CA markers, overlap points, and other
non-lexical content based on the requested `TierDomain`.  The node
indices let downstream code map NLP output (Stanza tokens, FA word
timings) back to AST positions in O(1) per token, no DP re-alignment
needed.

This replaced the old Python `annotation_clean` function (60+ lines of
`.replace()` calls) and eliminated O(n*m) DP alignment in the morphosyntax
and forced alignment engines.

### Tier Construction

For all NLP commands (morphosyntax, FA, translation, utseg), the pattern is:

1. Rust collects payloads from the AST (word lists, utterance texts, etc.).
2. ML inference runs (via worker IPC on the server path, or via Python
   callback on the Python API path).
3. Rust injects results back into the AST, constructing the appropriate
   dependent tiers (%mor, %gra, %wor, %xtra, etc.).

Payload collection and result injection use functions from
`batchalign` (e.g., `collect_payloads()` / `inject_results()` for
morphosyntax).

### Serialization

The `WriteChat` trait renders CHAT text from the AST; serialization alone does
not establish validity or transform completion. It handles formatting concerns:
continuation lines, escaping, bullet timestamp encoding, tier
alignment, and header ordering. Rust callers invoke `chat_file.to_chat_string()`;
there is no PyO3 `handle.serialize()` surface today (the ParsedChat
binding was retired in the 2026-03-21 pyo3 slimdown).

### Validation

Chatter's full model validator owns CHAT rules and alignment validation.
`validate_to_level` and `validate_output` are narrower workflow checks; they do
not cover the full suite of CHAT validity rules. The
[command contracts](command-contracts.md) specify retained-input admission,
completion, output checks and failure categories without treating these levels
as full-validity certificates. On the Python side, structured validation results reach callers as
`CHATValidationException.errors` (a `list[ValidationErrorEntry]`),
see [Errors, Batchalign Runtime](../../architecture/errors-and-validation/batchalign-errors.md).

## What Stays in Python

| Operation | Module | Python Library |
|-----------|--------|---------------|
| ASR transcription | `inference/asr.py` | transformers |
| Forced alignment | `inference/fa.py` | transformers, torchaudio |
| Morphosyntactic analysis | `inference/morphosyntax.py` | stanza |
| Speaker diarization | `inference/speaker.py` | nemo, pyannote |
| Utterance segmentation | `inference/utseg.py` | stanza |
| Translation | `inference/translate.py` | googletrans, seamless |
| Audio feature extraction | `inference/opensmile.py` | opensmile |
| Cantonese ASR | `inference/languages/cantonese/` | tencent/aliyun/funasr SDKs |

Each module exports a pure inference entrypoint used by the live V2 worker host
(for example the morphosyntax helper called from `_text_v2.py`, or the ASR/FA
helpers used by `execute_v2`). Python workers are stateless ML endpoints;
server/client orchestration is in Rust (axum + Rust CLI), and server-to-worker
transport is stdio IPC.

## Background

This current Rust AST architecture replaced the older Python-heavy CHAT path,
which depended much more on string manipulation, text flattening, and
post-hoc reconstruction. The durable current rule is:

- CHAT parsing, extraction, injection, validation, and serialization belong in
  Rust
- Python workers should focus on inference, not CHAT ownership

That boundary is what makes the current morphosyntax, alignment, translation,
and utterance-segmentation behavior more predictable than the older BA2-era
paths.

## Current boundaries

### Current DP posture

The important current distinction is not "DP never exists anywhere." It is:

- current CHAT parsing and tier construction no longer depend on broad
  flattened-text reconstruction
- current morphosyntax standard paths are index-driven
- current FA handling in the Rust path is deterministic and identity/index-first
- edit-distance style algorithms remain legitimate for evaluation tasks such as
  WER

If you need the public migration story for where older BA2-style DP-heavy
recovery changed, use the migration chapters rather than this architecture page.

See:
- [Python-Rust Boundary](../../architecture/python-rust-boundary/python-rust-boundary.md)
 , server-side CHAT ownership, wire protocol, capability discovery,
  worker module layout.
