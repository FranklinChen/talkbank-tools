# Validation

**Status:** Current
**Last verified:** 2026-03-11

Batchalign validates CHAT files at multiple points in the processing
pipeline. All validation logic is implemented in Rust
(`batchalign-chat-ops/src/validate.rs`).

## Validity Levels

The `ValidityLevel` enum defines three cumulative validation levels. Each
level includes all checks from lower levels.

| Level | Name | Checks |
|-------|------|--------|
| L0 | `Parseable` | No parse errors (clean tree-sitter CST) |
| L1 | `StructurallyComplete` | @Participants present, @Languages present, all speaker codes declared, every utterance has a terminator |
| L2 | `MainTierValid` | Well-formed words, valid timing bullets if present |

### Pre-validation gates

Each command requires input to meet a minimum level before processing:

| Command | Required level |
|---------|---------------|
| morphotag | `MainTierValid` |
| utseg | `StructurallyComplete` |
| translate | `StructurallyComplete` |
| coref | `StructurallyComplete` |
| align | `Parseable` (lenient — must handle messy real-world files) |

`validate_to_level()` checks the file against the required level and returns
all failures found. Invalid files are rejected early with diagnostics,
before any compute is spent on inference.

## Post-Serialization Validation

After an orchestrator injects results and serializes CHAT output, the server
runs `validate_output()`, which performs:

1. **Alignment validation** — checks that %mor/%gra/%wor tier word counts
   match the main tier. ParseHealth-aware: utterances flagged as
   unparseable during lenient parsing are excluded from alignment checks.

2. **Semantic validation** — full CHAT validation covering:
   - **E362**: Non-monotonic timestamps (utterance bullets must increase)
   - **E701/E704**: Temporal constraints (overlap rules, same-speaker timing)
   - **Header correctness**: Required headers present and well-formed
   - **Cross-utterance patterns**: Speaker code consistency

   Only blocks on `severity="error"`, not warnings.

## Severity Posture

Validation intentionally distinguishes errors from warnings:

- **Errors** block output. The server will not write CHAT with error-level
  validation failures.
- **Warnings** are reported but do not block. Legacy corpora contain
  widespread minor violations that must remain processable.

This distinction matters especially for `%gra`:
- Existing broken `%gra` in old corpora may be accepted with warnings so
  files remain processable.
- Newly generated `%gra` from batchalign3 is validated more strictly before
  writeback.

## Bug Reports and Cache Purges

When post-serialization validation fails:

1. A structured bug report is written to `~/.batchalign3/bug-reports/`.
2. Cache entries that produced the invalid output are purged
   (self-correcting cache).

This prevents broken results from being served on future runs.

## Validation in the PyO3 Interface

The `ParsedChat` PyO3 handle exposes validation methods for the Python API
path:

| Method | Returns | Description |
|--------|---------|-------------|
| `validate()` | `list[str]` | Tier alignment errors (human-readable) |
| `validate_structured()` | JSON string | Tier alignment errors (structured) |
| `validate_chat_structured()` | JSON string | Full semantic validation |

These are used by the direct Python pipeline facade and tests. The server path uses the
same underlying validation functions directly on the `ChatFile` AST.
