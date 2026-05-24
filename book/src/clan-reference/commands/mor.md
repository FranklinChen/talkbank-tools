# MOR -- Morphological Analysis (deliberately not implemented)

**Status:** Reference -- stub command
**Last updated:** 2026-05-22 13:14 EDT

## Purpose

In legacy CLAN, `mor` adds `%mor` dependent tiers to CHAT files by
performing morphological analysis of main-tier words against
language-specific lexicon databases (trie-based, ~11,000 lines of C)
and five rule engines (A-rules, C-rules, D-rules, PREPOST rules,
allomorph rules).

This project does **not** implement `mor`. The CHAT grammar and data
model have moved to UD-style (Universal Dependencies) morphological
representation, which is incompatible with the legacy CLAN MOR format.
A faithful port is impractical and would diverge from the rest of the
toolchain on the very dimension the command is meant to serve.

## What to use instead

Use the [batchalign](../../batchalign/introduction.md) morphotag
pipeline, which produces `%mor` and `%gra` tiers via Stanza's
UD-trained neural models. It supports more languages with higher
accuracy than the legacy CLAN MOR grammars.

```bash
batchalign3 morphotag corpus/  # neural morphosyntax pipeline
```

See the [batchalign morphosyntax reference](../../batchalign/reference/morphosyntax.md)
for the full pipeline.

## Behavior

Invoking `chatter clan mor` prints an error directing users to
batchalign and exits with a non-zero status. No CHAT files are
modified.

## CLAN `+`-flag coverage audit

**Exempt** — MOR is a deliberate non-implementation (see Purpose
above). Per the workspace policy, the MOR-pipeline commands
(`mor`, `post`, `postlist`, `postmodrules`, `posttrain`,
`megrasp`) are stubs that emit a refusal and direct users to
batchalign's neural pipeline. The CLAN `+`-flag surface for each
is documented in CLAN's own usage text and is not mirrored here
because chatter does not consume those flags.

## See also

- [POST, POSTLIST, POSTMODRULES, POSTTRAIN](post.md) -- same status
- [MEGRASP](megrasp.md) -- same status, dependency-relation variant
- [POSTMORTEM](postmortem.md) -- the post-processing step that **is**
  implemented (operates on an existing `%mor` tier)
