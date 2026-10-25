# L2 Morphotag: Current Status

**Status:** Current
**Last updated:** 2026-10-29

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
├── extract.rs: deferred @s positions and the primary's structure for them
├── plan.rs: contiguous spans (owning their positions) and L2Attachment
├── merge.rs: MergedL2Span, ModelAssignedPos, the external relation
├── deprel.rs: UdDeprel, relation-to-category check, relation inference
├── splice.rs: splice merged spans into the ChatFile, validate, roll back
└── pipeline_tests.rs: end-to-end tests on recorded worker analyses
```

Shared `UdAlignment` lives in `crates/batchalign-transform/src/morphosyntax/alignment.rs`.

**Dispatch** (`crates/batchalign/src/morphosyntax/batch.rs`,
`dispatch_secondary_l2`): plan spans from the deferred positions, send
each span to a secondary Stanza worker through `FixedSpanBatch` (CHAT
boundaries preserved, MWT components retained), merge each admitted response
with its owning span
(`merge_planned_secondary_span`), and splice the merged spans
(`splice_l2_into_chat`). The batch, single-file and incremental paths all
call it.

### Fallback and reporting

The [availability and reporting policy](morphosyntax.md#availability-and-reporting-policy)
keeps unavailable inline secondary analysis local to the affected word/span
when L2 dispatch is enabled.
`PendingSecondaryWord` projects `L2|xxx` until an admitted secondary span is
actually installed. Successful host morphology and other supported spans remain.

| Boundary | Cause | Report |
|----------|-------|--------|
| extract | unresolved/ambiguous language or unplaceable primary analysis | `L2Extraction::unaligned`, durable job shortfall |
| dispatch | runtime explicitly lacks target processors | unsupported-language exclusion |
| dispatch | inference failure, response mismatch or empty response | producer-specific secondary-analysis shortfall |
| merge | unalignable analysis or an invalid rooted dependency block | `L2MergeError`, original host analysis retained |
| splice | insertion refusal or invalid resulting grammar | addressed retention cause, exact host snapshot retained |

Job records retain counts and bounded examples. A saved availability `@Comment`
names unsupported languages and counts or summarizes incomplete analysis. A
primary/effective whole-utterance language capability refusal, including an
unavailable registry, is a separate file-level policy; it cannot borrow the inline marker rule.

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
acts on primary English UD analysis. L2 extraction uses the same admitted
analysis and item placement that injection mapped, so its source positions
remain bound to the written host morphology.

## What is not done

- Whole-utterance host syntax is still the primary model's analysis. Dependents
  of the attachment-source word follow the secondary block's admitted root;
  other replaced words follow their counterpart chunks. Valid graph structure
  does not establish linguistic correctness.
- Model coverage and secondary analyses still require language-specific quality
  evaluation; see [L2 Morphotag](l2-morphotag.md#limitations).

## Documentation

- `l2-morphotag.md`: design, architecture, index spaces, Mermaid diagrams
- `l2-morphotag-literature.md`: literature survey
- `l2-eval-runs/`: aggregate evaluation evidence (per-pair and per-word CSVs)
