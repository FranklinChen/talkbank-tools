# Errors — Batchalign Runtime

**Status:** Current
**Last updated:** 2026-05-01 17:07 EDT

How Batchalign produces, propagates, and surfaces errors specific to
the ML runtime: parse modes, ML/IPC failures, network errors,
ASR-API errors, worker-crash recovery, the Python-facing exception
hierarchy, and the CLI failure summary. For the CHAT-core error
infrastructure (codes, sinks, severities) see
[talkbank-tools-errors](talkbank-tools-errors.md). For the typed
boundary between Python and Rust workers see
[python-rust-errors](python-rust-errors.md). For the diagnostic UX
standard that applies workspace-wide, see
[error-diagnostics-ux](error-diagnostics-ux.md).

## Two Parse Modes

### Strict (`ParsedChat.parse()`)

Used by engines that require a valid AST to produce correct output:
`add_morphosyntax_batched`, `extract_nlp_words`, etc.

- **Rejects** on any error — raises `ValueError` in Python.
- Error string includes all error codes and locations:

```text
Parse error: error[E316]: Could not parse content (line 5, bytes 100..120)
```

### Lenient (`ParsedChat.parse_lenient()`)

Used by engines that can tolerate partial results:
`parse_and_serialize`, `add_forced_alignment`, `add_translation`.

- **Recovers** from errors using tree-sitter error recovery.
- Tainted tiers are marked via `ParseHealth` flags so downstream
  validation skips them.
- Parse warnings are captured in `ParsedChat.warnings` and available
  via `parse_warnings()` (JSON array).
- Only rejects when the file is completely empty after recovery.

## Structured Error Access (PyO3)

`ParsedChat` exposes three error-related methods to Python:

| Method | Returns | Purpose |
|---|---|---|
| `validate()` | `list[str]` | Human-readable alignment errors (uses `Display`); backward-compatible logging |
| `validate_structured()` | JSON string | Same alignment checks as `validate()`, but structured |
| `parse_warnings()` | JSON string | Warnings collected during lenient parsing; empty `"[]"` for strict parses or clean files |

`validate_structured()` example payload:

```json
[
  {
    "code": "E705",
    "severity": "error",
    "line": null,
    "column": null,
    "message": "Main tier has 2 alignable items, but %mor tier has 1 items\n...",
    "suggestion": "Each alignable word in main tier must have corresponding %mor item"
  }
]
```

Used by the direct Python API and debugging surfaces that need
structured alignment errors without re-parsing message text.

## Pre-Serialization Validation Gate

After Rust-owned processing stages have injected their results and
before final serialization, the production path validates the
generated `ChatFile` again to catch bugs in our own generation code
(MOR/GRA count mismatch, terminator identity errors).

```rust,ignore
// crates/batchalign/src/validate.rs
let errors = validate_chat(&chat_file);
if !errors.is_empty() {
    return Err(ChatValidationError { errors });
}
```

The exception message includes error codes and line numbers:

```text
Pre-serialization validation failed:
  - E705: Main tier has 2 alignable items, but %mor tier has 1 items
  - E716: Main tier terminator "." does not match %mor terminator "?" (line 23)
```

For full validation gates (G0–G14, validity levels L0–L2, post-serialization
checks), see [validation](validation.md).

## `CHATValidationException`

Defined in `batchalign/errors.py`:

```python
class CHATValidationException(Exception):
    def __init__(self, message: str,
                 errors: list[dict[str, object]] | None = None) -> None:
        super().__init__(message)
        self.errors: list[dict[str, object]] = errors or []
```

The `errors` list contains the structured dicts from
`validate_structured()`. Code that catches this exception can inspect
`exc.errors` for programmatic access to error codes, line numbers,
and suggestions without parsing the message string.

This is the Python-facing exception surface; the Rust processing
server and CLI do not depend on a `pipeline.py` wrapper to surface
validation failures. Backward compatible: `CHATValidationException("plain message")`
still works and sets `errors=[]`.

## Runtime Error Classification

Error category mapping is centralized in `batchalign/errors.py`
(`classify_error(exc)`) and used by server-side job accounting.
Exceptions classify into four categories:

| Category | Meaning | Examples |
|---|---|---|
| `input` | Bad CHAT content | `CHATValidationException`, parse errors |
| `media` | Missing audio/video files | `FileNotFoundError` |
| `system` | Infrastructure failure | `MemoryError` |
| `processing` | Unexpected errors during processing | Everything else |

Classification is done by `classify_error(exc)`. Parse errors are
identified by `CHATValidationException` type or by the
`"Parse error"` prefix in `ValueError` messages.

## CLI Failure Summary

Rust CLI dispatch aggregates failures per job/server and prints
structured summaries after polling. Error details shown to users
are derived from server `FileStatusEntry` fields (`error`,
`error_category`, and any structured validation metadata).

## Error Flow

```text
┌─────────────────────────────────────────────────────────────┐
│                    Rust Parser                              │
│  parse_chat_file() ──► ParseError { code, line, message }   │
│  parse_chat_file_streaming() ──► ErrorSink collects warnings│
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                  batchalign_core (PyO3)                     │
│  parse_strict_pure()  ──► format!("{}", e) ──► ValueError   │
│  parse_lenient_pure() ──► (ChatFile, warnings)              │
│  validate_structured()──► errors_to_json() ──► JSON string  │
│  parse_warnings()     ──► errors_to_json() ──► JSON string  │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│              Python API / Legacy adapters                   │
│  batchalign/errors.py may wrap structured validation JSON   │
│  into CHATValidationException(msg, errors=[...])            │
└──────────────────────┬──────────────────────────────────────┘
                       │
              ┌────────┴────────┐
              ▼                 ▼
┌──────────────────┐  ┌──────────────────────────┐
│        CLI       │  │    Processing Server     │
│  polls file/job  │  │  validates generated AST │
│  status, formats │  │  maps failures into      │
│  failure summary │  │  FileStatusEntry metadata│
└──────────────────┘  └──────────────────────────┘
```
