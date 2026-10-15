---
title: Resolve alignment policy check matches against actual code
author: Codex
date: 2026-10-16
branch: main
cost: { minutes: 301, tokens: 4200337 }
touches:
  - crates/batchalign/src/chat_ops/fa/mod.rs
---

## Decided
The projection/input split keeps historical text-shaped cache identities and makes healing explicit. The eight trail-check matches inspected for this change describe that separation or older request/output admission; none restores coupling.

## Tried and ruled out
A keyword match alone was insufficient to establish a semantic conflict.

## Watch out
Recommendations and successful verification in a Tried and ruled out section can be misread as prohibitions by a lexical checker. Inspect the cited source and preserve the actual policy; explain the disposition in the commit.
