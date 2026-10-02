# L2 Morphotag: Current Status

**Status:** Current
**Last updated:** 2026-10-01 17:37 EDT

L2 dispatch is on by default (`--no-l2-morphotag` opts out). It routes
`@s` (code-switched) words to the Stanza model of their own language and
combines that analysis with the primary model's attachment of the span.
The design is in [L2 Morphotag](l2-morphotag.md); changes are recorded in
the repository's `CHANGELOG.md`.

## What is in place

### Ownership rule

- The secondary model owns each `@s` word's category, lemma and features,
  and the relations inside the span it analysed.
- The primary model owns the span's attachment: the host word it hangs
  from and its relation.
- Where the primary's relation contradicts the category of the secondary
  root that carries it, the relation is corrected
  (`ExternalRelation::Corrected`); the category never is.
- A phrasal-verb particle (`compound:prt` to a secondary VERB) is written
  PART.

### Modules

```text
crates/batchalign-transform/src/morphosyntax/l2/
├── alignment.rs: UdAlignment, the one CHAT-word to UD-word alignment
├── extract.rs: deferred @s positions and the primary's structure for them
├── plan.rs: contiguous spans (owning their positions) and L2Attachment
├── merge.rs: MergedL2Span, ModelAssignedPos, the external relation
├── deprel.rs: UdDeprel, relation-to-category check, relation inference
├── splice.rs: splice merged spans into the ChatFile, validate, roll back
└── pipeline_tests.rs: end-to-end tests on recorded worker analyses
```

**Dispatch** (`crates/batchalign/src/morphosyntax/batch.rs`,
`dispatch_secondary_l2`): plan spans from the deferred positions, send
each span to a secondary Stanza worker (`infer_batch`, Stanza owning
tokenization), merge each response with its span
(`merge_planned_secondary_span`), and splice the merged spans
(`splice_l2_into_chat`). The batch, single-file and incremental paths all
call it.

### Fallback and reporting

Every `@s` word keeps the `L2|xxx` the primary pass wrote unless its span
splices, and every way to keep it is reported:

| Where | Cause | Report |
|-------|-------|--------|
| extract | primary analysis does not align to the utterance's words | `L2Extraction::unaligned`, logged by `into_reported_positions` |
| dispatch | no Stanza model for the target, or dispatch failed | `L2 morphotag:` log lines with the word count |
| merge | secondary analysis does not align, map, or has no root | `L2MergeError`, logged with line and word count |
| splice | the spliced `%gra` breaks an invariant | categorised warning, span rolled back |

### Tests

- `alignment.rs`: the alignment across a contraction, multi-word-token
  representatives, terminator rows, and each refusal.
- `pipeline_tests.rs`: end-to-end runs (payload collection, extraction,
  primary injection, planning, merge, splice) on UD analyses recorded from
  BA3's worker on Stanza 1.15.0: an `@s` word after a contraction, a span
  after a contraction, phrasal verbs with a terminator row, the span's
  external relation and its correction, categories both models agree on,
  a secondary verb, and the `time out` compound noun.
- `plan.rs`, `merge.rs`, `extract.rs`, `splice.rs`: unit tests of each step.
- ML golden (`crates/batchalign/tests/ml_golden/morphotag/golden_l2.rs`):
  real models through a live session for eng-spa, deu-eng, contractions,
  phrasal verbs, cat-spa, dan-eng, fra-nld, and the flag-off case. Each
  assertion prints the `%mor` line it inspects.

### Input policy and repair tooling

- E255 rejects whole-utterance same-language all-`@s` patterns at
  validation time; transcripts should use `[- lang]`.
- E254 is warn-only when explicit `@s:LANG` names a language absent from
  `@Languages`; dispatch still uses the explicit target language.
- `chatter debug fix-s` repairs both transcript-side issues: it rewrites the
  qualifying whole-utterance `@s` pattern, appends missing explicit languages
  to `@Languages`, and leaves already-correct files untouched.

### Interaction with the English grammatical-invariant rewrite

A Rust rewrite rule (see [Stanza Limitations, Defect 1](stanza-limitations.md))
runs on the **primary** English UD analysis inside injection. L2
extraction reads the primary responses before injection takes them, so the
rewrite cannot change what extraction aligns.

## What is not done

- Host words whose primary head is a span word attach to the span's first
  chunk (chatter's splice), not to the secondary root; see Limitations in
  [L2 Morphotag](l2-morphotag.md#limitations).

## Documentation

- `l2-morphotag.md`: design, architecture, index spaces, Mermaid diagrams
- `l2-morphotag-literature.md`: literature survey
- `l2-eval-runs/`: aggregate evaluation evidence (per-pair and per-word CSVs)
