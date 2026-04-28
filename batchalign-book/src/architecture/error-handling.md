# Error Handling in Batchalign

**Status:** Current
**Last updated:** 2026-03-16

This document describes how errors are produced, propagated, and displayed
across the Rust parser, the Python pipeline, the processing server, and the
CLI.

---

## 1. Error Origin: The Rust Parser

All CHAT parsing happens in Rust via `batchalign_core`.  The parser produces
`ParseError` structs (defined in `talkbank-errors`) with rich structured
fields:

| Field        | Type               | Example                                       |
|--------------|--------------------|-----------------------------------------------|
| `code`       | `ErrorCode` enum   | `E502`, `E705`, `W042`                        |
| `severity`   | `Severity` enum    | `error` or `warning`                          |
| `line`       | `Option<usize>`    | `47`                                          |
| `column`     | `Option<usize>`    | `12`                                          |
| `message`    | `String`           | `"Main tier has 2 items, but %mor has 1"`     |
| `suggestion` | `Option<String>`   | `"Each main-tier word needs a %mor item"`     |
| `location`   | `SourceLocation`   | Byte span `1234..1260`                        |
| `context`    | `ErrorContext`      | Source snippet, offending text, expected items|
| `labels`     | `Vec<ErrorLabel>`  | Secondary spans for multi-span diagnostics    |

The `Display` impl formats these as:

```
error[E705]: Main tier has 2 items, but %mor has 1 (line 42, column 10, bytes 1200..1250)
```

### Error codes

Error codes follow the pattern `E` + 3 digits (errors) or `W` + 3 digits
(warnings).  Ranges:

- **E0xx, E1xx** -- Generic/structural (empty file, missing headers)
- **E2xx** -- Syntax (quotation balance, overlap markers)
- **E3xx** -- Content (undeclared speakers, invalid codes)
- **E4xx** -- Dependent tier structure
- **E5xx** -- File-level constraints (missing @End, media mismatch)
- **E7xx** -- Alignment diagnostics (main-mor count, mor-gra count, terminator mismatch)

The full catalog lives in the sibling `talkbank-tools` repo under
`spec/errors/`.

---

## 2. Two Parse Modes

### Strict (`ParsedChat.parse()`)

Used by engines that require a valid AST to produce correct output:
`add_morphosyntax_batched`, `extract_nlp_words`, etc.

- **Rejects** on any error -- raises `ValueError` in Python.
- Error string includes all error codes and locations:
  ```
  Parse error: error[E316]: Could not parse content (line 5, bytes 100..120)
  ```

### Lenient (`ParsedChat.parse_lenient()`)

Used by engines that can tolerate partial results:
`parse_and_serialize`, `add_forced_alignment`, `add_translation`.

- **Recovers** from errors using tree-sitter error recovery.
- Tainted tiers are marked via `ParseHealth` flags so downstream
  validation skips them.
- Parse warnings are captured in `ParsedChat.warnings` and available via
  the `parse_warnings()` method (JSON array).
- Only rejects when the file is completely empty after recovery.

---

## 3. Structured Error Access (PyO3 Methods)

`ParsedChat` exposes three error-related methods to Python:

### `handle.validate() -> list[str]`

Runs tier alignment checks (main↔mor, mor↔gra, main↔wor, main↔pho,
main↔sin).  Returns human-readable strings using the `Display` impl.
Backward-compatible; used for logging.

### `handle.validate_structured() -> str`

Same alignment checks, but returns a **JSON array** string.  Each element
is an object:

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

Used by the direct Python API and debugging surfaces that need structured
alignment errors without reparsing message text.

### `handle.parse_warnings() -> str`

Returns a JSON array of warnings collected during lenient parsing.  Same
schema as `validate_structured()`.  Returns `"[]"` for strict parses or
clean files.

---

## 4. Validation Gate

After Rust-owned processing stages have injected their results and before final
serialization, the production path validates the generated `ChatFile` again to
catch bugs in our own generation code (for example MOR/GRA count mismatch or
terminator identity errors).

Validation is performed by the Rust server after result injection:

```rust
// batchalign-chat-ops/src/validate.rs
let errors = validate_chat(&chat_file);
if !errors.is_empty() {
    return Err(ChatValidationError { errors });
}
```

The exception message includes error codes and line numbers:

```
Pre-serialization validation failed:
  - E705: Main tier has 2 alignable items, but %mor tier has 1 items
  - E716: Main tier terminator "." does not match %mor terminator "?" (line 23)
```

---

## 5. `CHATValidationException`

Defined in `batchalign/errors.py`:

```python
class CHATValidationException(Exception):
    def __init__(self, message: str,
                 errors: list[dict[str, object]] | None = None) -> None:
        super().__init__(message)
        self.errors: list[dict[str, object]] = errors or []
```

The `errors` list contains the same structured dicts from
`validate_structured()`. Code that catches this exception can inspect
`exc.errors` for programmatic access to error codes, line numbers, and
suggestions without parsing the message string. This remains the Python-facing
exception surface; the Rust processing server and CLI do not depend on a
`pipeline.py` wrapper to surface validation failures.

Backward compatible: `CHATValidationException("plain message")` still
works and sets `errors=[]`.

---

## 6. Runtime Error Classification

Error category mapping is centralized in `batchalign/errors.py`
(`classify_error(exc)`) and used by server-side job accounting.
Exceptions are classified into four categories:

| Category     | Meaning                              | Examples                                  |
|--------------|--------------------------------------|-------------------------------------------|
| `input`      | Bad CHAT content                     | `CHATValidationException`, parse errors   |
| `media`      | Missing audio/video files            | `FileNotFoundError`                       |
| `system`     | Infrastructure failure               | `MemoryError`                             |
| `processing` | Unexpected errors during processing  | Everything else                           |

Classification is done by `classify_error(exc)`. Parse errors are
identified by `CHATValidationException` type or by the `"Parse error"`
prefix in `ValueError` messages.

---

## 7. CLI Failure Summary

Rust CLI dispatch aggregates failures per job/server and prints structured
summaries after polling. Error details shown to users are derived from server
`FileStatusEntry` fields (`error`, `error_category`, and any structured
validation metadata).

---

## 8. Error Flow Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    Rust Parser                               │
│  parse_chat_file() ──► ParseError { code, line, message }   │
│  parse_chat_file_streaming() ──► ErrorSink collects warnings │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                  batchalign_core (PyO3)                      │
│  parse_strict_pure()  ──► format!("{}", e) ──► ValueError   │
│  parse_lenient_pure() ──► (ChatFile, warnings)              │
│  validate_structured()──► errors_to_json() ──► JSON string  │
│  parse_warnings()     ──► errors_to_json() ──► JSON string  │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│              Python API / Legacy adapters                    │
│  batchalign/errors.py may wrap structured validation JSON    │
│  into CHATValidationException(msg, errors=[...])             │
└──────────────────────┬──────────────────────────────────────┘
                       │
              ┌────────┴────────┐
              ▼                 ▼
┌──────────────────┐  ┌──────────────────────────┐
│        CLI        │  │    Processing Server      │
│  polls file/job   │  │  validates generated AST  │
│  status, formats  │  │  maps failures into       │
│  failure summary  │  │  FileStatusEntry metadata │
└──────────────────┘  └──────────────────────────┘
```

---

## 9. Adding New Error Codes

1. Add the variant to `ErrorCode` enum in
   `talkbank-tools/crates/talkbank-model/src/errors/codes/error_code.rs`
   with a `#[code("Exxx")]` attribute.

2. Create a spec file in `talkbank-tools/spec/errors/Exxx-description.md`
   following the existing template.

3. Construct `ParseError::new(ErrorCode::YourVariant, ...)` at the
   detection site in the parser or validator.

4. Run `make test-gen` to regenerate test fixtures from specs.

5. Run `cargo nextest run --manifest-path pyo3/Cargo.toml` and rebuild with maturin.

No Python changes are needed -- the structured JSON validation surface
picks up new error codes.
