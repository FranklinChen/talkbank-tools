---
title: FA requests own context separately from output words
author: Codex
date: 2026-10-16
branch: main
touches:
  - crates/batchalign/src/chat_ops/fa/split/context.rs
  - crates/batchalign/src/fa/units.rs
  - crates/batchalign/src/chat_ops/fa/postprocess.rs
  - crates/batchalign/src/fa/mod.rs
  - crates/batchalign/src/types/traces.rs
---

## Decided
Keep disjoint output ownership separate from overlapping lexical and acoustic request context. Borrow source-bounded context within the configured duration budget, without crossing an unavailable range. An anchored tail can join its following request; positive duration alone must not isolate it. Cache keys and reply cardinality cover the full request, while assembly selects only owned words. Bind admitted timings to their request before projection and preserve that authority through postprocessing. Bound derived terminal ends with provenance; refuse out-of-request measurements.

## Tried and ruled out
A maximum-duration bound is not evidence that a very short request supplies enough acoustic context. Exact lexical anchors do not prove acoustic cuts. Clamping only during onset parsing is insufficient because later explicit healing can extend the result again. Reusing old onset-derived cache projections would preserve the defect; raw evidence remains replayable through the current parser, and unchanged direct intervals remain compatible.

## Watch out
FA evidence schema 8 records both ownership and context. Consumers assuming shared nonoverlapping request cuts must refuse it until updated. Derived cache schema 2 rejects schema-1 onset-only projections. Synthetic pipeline, worker-contract and containment checks establish software behavior, not acoustic accuracy, deployment or corpus acceptance.
