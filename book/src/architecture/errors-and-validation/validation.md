# Validation

**Status:** Current
**Last updated:** 2026-10-01 20:24 EDT

CHAT validation runs at multiple points in the processing pipeline.
All validation logic is in Rust: `talkbank-model::validation` owns
CHAT-core validation, and `talkbank_transform::validate`
(`crates/talkbank-transform/src/validate.rs`) owns the Batchalign-side
pre/post validation gate functions (`validate_to_level`,
`validate_output`). This page covers validity levels, pre/post
validation gates, severity posture, the verification-gate set
(G0-G14), and how validation failures interact with caches and bug
reports.

For the CHAT-core error-code infrastructure (codes, sinks, severities,
layers), now owned by chatter, see
[chat-core-errors](https://talkbank.github.io/chatter/architecture/errors-and-validation/chat-core-errors.html).
For the diagnostic UX standard, see
[error-diagnostics-ux](https://talkbank.github.io/chatter/architecture/errors-and-validation/error-diagnostics-ux.html).

## Validity Levels

The `ValidityLevel` enum defines three cumulative validation levels.
Each level includes all checks from lower levels.

| Level | Name | Checks |
|---|---|---|
| L0 | `Parseable` | No parse errors (clean tree-sitter CST) |
| L1 | `StructurallyComplete` | `@Participants` and `@Languages` present, all speaker codes declared, every utterance has a terminator |
| L2 | `MainTierValid` | Well-formed words, valid timing bullets if present |

### Pre-validation gates

Each command requires input to meet a minimum level before
processing:

| Command | Required level |
|---|---|
| `morphotag` | `MainTierValid` |
| `utseg` | `StructurallyComplete` |
| `translate` | `StructurallyComplete` |
| `coref` | `StructurallyComplete` |
| `align` | `Parseable` (lenient, must handle messy real-world files) |

`validate_to_level()` checks the file against the required level and
returns all failures found. Invalid files are rejected early with
diagnostics, before any compute is spent on inference.

## Post-Serialization Validation

After an orchestrator injects results and serializes CHAT output, the
server runs `validate_output()`:

1. **Alignment validation**: checks that `%mor`/`%gra`/`%wor` tier
   word counts match the main tier. ParseHealth-aware: utterances
   flagged as unparseable during lenient parsing are excluded.
2. **Semantic validation**: full CHAT validation:
   - **E362**: non-monotonic timestamps (utterance bullets must
     increase).
   - **E701** / **E704**: temporal constraints (overlap rules,
     same-speaker timing).
   - Header correctness, required headers present and well-formed.
   - Cross-utterance patterns, speaker code consistency.

Only blocks on `severity="error"`, not warnings.

## Severity Posture

Validation intentionally distinguishes errors from warnings:

- **Errors** block output. The server will not write CHAT with
  error-level validation failures.
- **Warnings** are reported but do not block. Legacy corpora contain
  widespread minor violations that must remain processable.

This distinction matters especially for `%gra`:

- Existing broken `%gra` in old corpora may be accepted with warnings
  so files remain processable.
- Newly generated `%gra` from batchalign3 is validated more strictly
  before writeback.

## When Validation Fails

When post-serialization validation fails, the file fails with the
validation errors as its error, and no output is written for it. No report
file is written and no cache entry is purged: the cache entries behind the
output stay, so a rerun reuses them (use the command's cache override to
recompute).

## Verification Gates (`make verify`)

The mandatory pre-merge / pre-push gate. All gates must pass before
any code or doc change ships:

| Gate | What it checks |
|---|---|
| G0 | Parser signature guardrail |
| G1 | Rust workspace compile check |
| G2 | Spec tools compile check |
| G3 | Spec runtime tools compile check |
| G4 | CHAT manual anchor links |
| G5 | Generated parser corpus equivalence suite |
| G6 | Golden fragment validity (words + tiers) |
| G7 | Bare-timestamp regression gate |
| G8 | Reference corpus semantic equivalence |
| G9 | `%wor` tier parsing and alignment |
| G10 | Golden tier roundtrip (`%mor`, `%gra`, `%pho`, `%wor`) |
| G11 | Reference corpus node coverage |
| G12 | Generated artifacts match committed sources |
| G13 | Fuzz workspace isolation |
| G14 | Imported Batchalign Rust/PyO3 gate |

The reference corpus at `corpus/reference/` is the sacred semantic
target, every file must be valid CHAT, and `make verify` runs each
gate against it.

The pre-push hook (`make install-hooks`) runs the fast subset (`fmt`,
affected compile, parser guardrail, `generated-check`, `fuzz-check`)
locally before push. CI runs the full set.

## Validation at the PyO3 Boundary

There is no public Python validation API. The `ParsedChat` handle
that previously exposed `validate()` / `validate_structured()` /
`validate_chat_structured()` was retired in the 2026-03-21 PyO3
slimdown to worker-runtime-only. Validation now runs entirely on the
Rust side; when a worker invocation detects a failure it constructs
`BatchalignBoundaryError::ChatValidation { entries, … }` which the
PyO3 boundary lowers into a `CHATValidationException` carrying a
populated `errors: list[ValidationErrorEntry]` on the Python side.

Python callers that need structured validation results invoke
`batchalign3` via subprocess and catch the exception:

```python
from batchalign_core import CHATValidationException

try:
    batchalign_core.execute_v2(request)
except CHATValidationException as exc:
    for entry in exc.errors:
        print(entry.code, entry.line, entry.message)
```

See [Errors, Batchalign Runtime](batchalign-errors.md) and
[Python ↔ Rust errors](python-rust-errors.md) for the full boundary
contract.

## Known limitations

- **Validation rules are intentionally permissive on legacy data.**
  Some checks emit warnings rather than errors so legacy corpora
  remain processable while still surfacing the issue. Examples: pre-existing
  malformed `%gra` (warned, not blocked, so files that already shipped
  with bad `%gra` round-trip cleanly); some bullet-format minor
  variants. Newly generated tiers from batchalign are validated more
  strictly before writeback.
- **`%wor` word counts are not validated against the main tier.** `%wor`
  is a timing-annotation tier with no downstream positional indexing;
  legacy files may have `xxx`, fragments, or nonwords in `%wor`
  without producing alignment errors.
- **Cross-utterance quotation validation is gated off by default**
  (`enable_quotation_validation` flag), the cross-utterance walker
  exists but is not yet wired into the standard validation gate.
- **Some error-spec / validator pairs are not yet implemented.**
  Tracked in `spec/errors/` files marked `Status: not_implemented`;
  these generate `#[ignore]` tests via `make test-gen` rather than
  failing CI. Run `grep -rl "Status.*not_implemented" spec/errors/`
  to enumerate.
