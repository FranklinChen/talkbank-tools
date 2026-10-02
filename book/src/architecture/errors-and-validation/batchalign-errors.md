# Errors: Batchalign Runtime

**Status:** Current
**Last updated:** 2026-10-01 20:24 EDT

How Batchalign produces, propagates, and surfaces errors specific to
the ML runtime: parse modes, ML/IPC failures, network errors,
ASR-API errors, worker-crash recovery, the Python-facing exception
hierarchy, and the CLI failure summary. For the CHAT-core error
infrastructure (codes, sinks, severities), now owned by chatter, see
[chat-core-errors](https://talkbank.github.io/chatter/architecture/errors-and-validation/chat-core-errors.html).
For the typed boundary between Python and Rust workers see
[python-rust-errors](python-rust-errors.md). For the diagnostic UX
standard that applies workspace-wide, see
[error-diagnostics-ux](https://talkbank.github.io/chatter/architecture/errors-and-validation/error-diagnostics-ux.html).

## Two Parse Modes

### Strict (`ParsedChat.parse()`)

Used by engines that require a valid AST to produce correct output:
`add_morphosyntax_batched`, `extract_nlp_words`, etc.

- **Rejects** on any error, raises `ValueError` in Python.
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

Structured CHAT-validation errors reach Python through the typed
boundary, not via a `ParsedChat` method surface (the legacy
`ParsedChat` binding was removed in the 2026-03-21 PyO3 slimdown to
worker-runtime-only, see [Python ↔ Rust Boundary](../python-rust-boundary/python-rust-boundary.md)).
Validation failures inside the Rust worker construct
`BatchalignBoundaryError::ChatValidation { entries, … }` (defined at
`crates/batchalign-pyo3/src/error.rs`); the boundary lowers that into
`CHATValidationException` on the Python side with a populated
`errors: list[ValidationErrorEntry]` field.

`ValidationErrorEntry` is a TypedDict (Python view of the Rust struct
at `crates/batchalign-pyo3/src/error.rs:85`) with these fields:

| Field | Type | Notes |
|---|---|---|
| `code` | `str` | e.g. `"E705"` |
| `severity` | `str` | `"error"` / `"warning"` |
| `line` | `Optional[int]` | 1-based; `None` when unavailable |
| `column` | `Optional[int]` | 1-based; `None` when unavailable |
| `message` | `str` | Full diagnostic message |
| `suggestion` | `Optional[str]` | Optional remediation hint |

Python callers inspect `exc.errors[0].code` and friends rather than
parsing message text. The boundary contract is enforced by
`batchalign/tests/test_pyo3_error_typing.py`.

## Pre-Serialization Validation Gate

After Rust-owned processing stages have injected their results and
before final serialization, the production path validates the
generated `ChatFile` again to catch bugs in our own generation code
(MOR/GRA count mismatch, terminator identity errors). The check
lives in `crates/talkbank-transform/src/validate.rs`:

```rust,ignore
use talkbank_transform::validate::{validate_output, validate_to_level};

// Pre-validation: input must meet the command's required ValidityLevel.
validate_to_level(&chat_file, &parse_errors, hooks.validity)?;
// ... command body runs, mutates chat_file ...
// Post-validation: catch regressions introduced by command code.
validate_output(&chat_file, hooks.command)?;
```

Call sites: `crates/batchalign/src/pipeline/text_infer.rs` and
`crates/batchalign/src/coref.rs` import both functions from
`talkbank_transform::validate`. The exception message includes error
codes and line numbers:

```text
Pre-serialization validation failed:
  - E705: Main tier has 2 alignable items, but %mor tier has 1 items
  - E716: Main tier terminator "." does not match %mor terminator "?" (line 23)
```

For full validation gates (G0-G14, validity levels L0-L2, post-serialization
checks), see [validation](validation.md).

## `CHATValidationException`

Defined in Rust at `crates/batchalign-pyo3/src/error.rs` via
`pyo3::create_exception!` and re-exported through Python:

```rust,ignore
// crates/batchalign-pyo3/src/error.rs
use pyo3::create_exception;
create_exception!(batchalign_core, BatchalignError, PyException);
create_exception!(batchalign_core, CHATValidationException, BatchalignError);
```

```python
# batchalign/errors.py
from batchalign_core import (
    BatchalignError,
    CHATValidationException,
    ...
)
```

When `BatchalignBoundaryError::ChatValidation { entries, .. }` crosses
the PyO3 boundary, the `From<BatchalignBoundaryError> for PyErr` impl
constructs the Python exception with a populated `errors:
list[ValidationErrorEntry]`. Code that
catches the exception inspects `exc.errors[i].code`,
`exc.errors[i].line`, etc. for programmatic access without parsing
the message string.

See [Python ↔ Rust errors](python-rust-errors.md) for the full
boundary contract, including the typed-exception hierarchy and the
internals-leakage scan.

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
