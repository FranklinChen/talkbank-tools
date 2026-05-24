# POSTTRAIN -- Train POST Model (deliberately not implemented)

**Status:** Reference -- stub command
**Last updated:** 2026-05-22 13:14 EDT

## Purpose

In legacy CLAN, `posttrain` trains the POST disambiguation model
from a hand-annotated training corpus. It is the offline counterpart
to the runtime [`post`](post.md) disambiguator.

This project does **not** implement `posttrain`. The MOR/POST training
pipeline is replaced by Stanza's neural-model training infrastructure,
which lives upstream of batchalign and is not part of this toolchain.

## What to use instead

To train new morphological models, use Stanza's training pipeline
directly (see the [Stanza documentation](https://stanfordnlp.github.io/stanza/)),
then route the resulting models through batchalign's morphotag
pipeline.

## Behavior

Invoking `chatter clan posttrain` prints an error and exits non-zero.

## See also

- [MOR](mor.md), [POST](post.md), [POSTLIST](postlist.md),
  [POSTMODRULES](postmodrules.md) -- companion stubs

## CLAN `+`-flag coverage audit

**Exempt** — see [MOR](mor.md#clan-flag-coverage-audit) for the
shared MOR-pipeline policy: chatter does not consume CLAN's
`+`-flags for any command in this family. CLAN's own usage text
documents the legacy flag surface; chatter emits a refusal that
points users to batchalign.
