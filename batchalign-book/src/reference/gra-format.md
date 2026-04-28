# %gra Format Conventions

This page describes the `%gra` forms that `batchalign3` currently accepts when
reading corpora and the stricter form it generates when writing new `%gra`
tiers.

## Accepted Root Conventions

When parsing existing CHAT data, `batchalign3` accepts both root styles that
occur in TalkBank corpora:

1. `head=0` for the `ROOT` relation
2. `head=self` for the `ROOT` relation

Examples:

**`head=0`**
```
%gra:	1|2|SUBJ 2|0|ROOT 3|2|OBJ 4|2|PUNCT
```

**`head=self`**
```
%gra:	1|3|DET 2|3|AMOD 3|3|ROOT 4|6|NSUBJ 5|6|ADVMOD 6|3|ACL-RELCL 7|3|PUNCT
```

Current `%gra` generation in `batchalign3 morphotag` emits `head=0`.

## Other TalkBank `%gra` Conventions

- Relation labels are uppercase, such as `NSUBJ`, `ADVMOD`, and `ACL-RELCL`.
- Relation subtypes use dashes rather than UD colons, such as `ACL-RELCL` and
  `NMOD-POSS`.
- `%gra` and `%mor` remain item-aligned: each `%mor` item has a corresponding
  `%gra` item.
- The utterance terminator gets its own `PUNCT` relation whose head points to
  the root word.

For comparison, a UD-style rendering would use lowercase labels and colon
subtypes:

```
%gra:	1|3|det 2|3|amod 3|0|root 4|6|nsubj 5|6|advmod 6|3|acl:relcl 7|3|punct
```

## Parser Validation for Existing Data

When reading existing CHAT files, `batchalign3` keeps `%gra` validation lenient
enough to ingest historical corpora that contain invalid dependency trees.

Current parser-side checks:

- `E721`: indices must be sequential (`1..N`) and remain an error
- `W722`: no `ROOT` relation
- `W723`: multiple `ROOT` relations
- `W724`: circular dependency

The root and cycle checks remain warnings at parse time so older corpora stay
processable even when `%gra` is malformed.

## Generator Validation for New `%gra`

When `batchalign3 morphotag` generates new `%gra`, validation is stricter. The
current implementation in
`crates/batchalign-chat-ops/src/nlp/mapping.rs` validates:

1. sequential indices
2. exactly one non-terminator root (`head=0` or `head=self`)
3. no dependency cycles
4. no head references outside the utterance

If validation fails, generation returns `Err(MappingError)` and the caller logs
and skips the utterance rather than writing invalid `%gra`.

## Current Write Contract

Current Rust `%gra` generation avoids the older positional-repair failure mode
by:

1. mapping IDs explicitly rather than relying on brittle array-position repair
2. rejecting invalid or unmappable structures before writeback
3. validating root/head invariants before returning the tier

Current migration rationale for the `head=0` write contract lives in the
migration docs; this page keeps only the current parser/generator behavior.
