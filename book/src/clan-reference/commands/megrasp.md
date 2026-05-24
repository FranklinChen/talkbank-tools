# MEGRASP -- Grammar Relation Parsing (deliberately not implemented)

**Status:** Reference -- stub command
**Last updated:** 2026-05-22 13:14 EDT

## Purpose

In legacy CLAN, `megrasp` adds `%gra` (grammatical relation) tiers
to CHAT files. It builds dependency-style head/relation labels on top
of an existing `%mor` tier produced by [`mor`](mor.md), using the
MEGRASP rule engine.

This project does **not** implement `megrasp`. Like its `%mor`
counterpart, the legacy hand-coded grammar rules are replaced by
neural inference. The CHAT data model uses UD-style dependency
relations on `%gra`, which Stanza emits directly alongside `%mor`.

## What to use instead

Use the [batchalign](../../batchalign/introduction.md) morphotag
pipeline. It produces `%mor` and `%gra` tiers from a single Stanza
pass, with UD-style relations (`NSUBJ`, `OBJ`, `ROOT`, etc.) instead
of CLAN's GRASP labels. The mapping layer is documented in the
[morphosyntax reference](../../batchalign/reference/morphosyntax.md).

```bash
batchalign3 morphotag corpus/  # emits %mor + %gra together
```

## Behavior

Invoking `chatter clan megrasp` prints an error directing users to
batchalign and exits with a non-zero status. No CHAT files are
modified.

## See also

- [MOR](mor.md), [POST](post.md) -- companion stubs in the same
  legacy grammar family
- [batchalign morphosyntax reference](../../batchalign/reference/morphosyntax.md)
  -- the UD-to-CHAT mapping that replaces this pipeline

## CLAN `+`-flag coverage audit

**Exempt** — see [MOR](mor.md#clan-flag-coverage-audit) for the
shared MOR-pipeline policy: chatter does not consume CLAN's
`+`-flags for any command in this family. CLAN's own usage text
documents the legacy flag surface; chatter emits a refusal that
points users to batchalign.
