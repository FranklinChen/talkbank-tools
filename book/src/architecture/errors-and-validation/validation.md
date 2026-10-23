# Validation

**Status:** Current
**Last updated:** 2026-10-25 08:12 EDT

Chatter owns CHAT parsing and validation. BA3 owns the command-specific
admission, transformation and output policy that uses those facilities. The
[command contracts](../../batchalign/architecture/command-contracts.md) are
the authoritative table of accepted inputs and output states; this page
explains the boundaries rather than duplicating that table.

For CHAT error codes, sinks and severity, see
[Chatter's error infrastructure](https://talkbank.github.io/chatter/architecture/errors-and-validation/chat-core-errors.html).

## Validity Levels

Legacy `ValidityLevel` checks describe narrower workflow prerequisites. They
are not a certificate of complete CHAT validity and are not a substitute for
Chatter's checked document admission. In particular, a recovered AST or a
successful `parse_lenient` call does not authorize downstream semantic work.

Morphotag admits the document under its replacement-tier policy, so tiers it
will replace are distinguished from content it must preserve. Its
`ParsedFile` transition also resolves the file language and admits Stanza
capability. The CA honor policy passes CA documents through without
morphological analysis; explicit analysis is a separate policy.

## Input and Output Admission

- Input admission binds the selected command policy to the parsed document.
  A consumer receives that admitted state, not a raw AST plus a claim that
  somebody checked it.
- A transform must establish the state its output writer requires. The
  admission boundary includes the applicable semantic and tier-alignment
  checks; serialization is not itself validation.
- Generating commands can retain a `DiagnosedOutput` with its findings under
  their declared contract. A diagnosed transcript is not labeled clean.
  Optional stages may keep their admitted predecessor when their own output
  is refused, recording a shortfall.
- Localized diagnosis can hold affected utterances out of supported
  per-utterance processing. Document-wide findings have a different effect.
  Do not infer that every parser-recovery result is safe to process.

The implementation owners are `pipeline/post_validate.rs`,
`pipeline/morphosyntax/states.rs`, and the command's own pipeline in
`crates/batchalign/src/`. See
[CHAT validation failures](../../batchalign/developer/chat-validation-failures.md)
for the concrete user-visible outcomes and recovery steps.

## Severity Posture

The selected validation policy decides which diagnostics block its admission.
Warnings, rejected input, diagnosed generated output and incomplete analysis
are distinct states. There is no blanket rule that all old malformed tiers
are only warnings or that every file carrying an error is silently discarded.
Use the command's documented contract and retained diagnostic category.

## When Validation Fails

Read the file's diagnostic findings and shortfalls before deciding whether
there is an output to inspect. A worker failure, output-admission refusal and
missing model capability are different causes. A cache hit does not certify
the resulting CHAT, and a validation failure does not automatically purge
cached inference. Recompute only when the cache override and evidence justify
it; see [cache overrides](../../batchalign/architecture/cache-override-guide.md).

## Verification Gates

The current publication check is `make gate`, implemented by
`scripts/gate.sh`. The pre-push hook verifies its receipt against the tree
being pushed; it does not run an older G0-G14 parser gate matrix. The gate
checks Rust, Python source typing/lint, build-workflow contracts, generated
interfaces and the book. It does not run the full Python inference suite.
PyO3/worker changes also need appropriate Python boundary tests using a
verified-current extension. See [building](../../batchalign/developer/building.md)
and the Makefile for exact supported targets.

## Validation at the PyO3 Boundary

The retired `ParsedChat` binding is not a public Python validation API.
The extension serves typed worker operations; structured boundary failures
can use `CHATValidationException` and its `ValidationErrorEntry` dictionaries.
An `execute_v2` operation can also return a typed failure outcome. Neither a
subprocess exit nor a protocol error should be treated as though it were a
Python CHAT-validation exception.

See [runtime errors](batchalign-errors.md) and
[Python ↔ Rust errors](python-rust-errors.md) for those separate contracts.
