---
title: Keep alignment projection policy out of model input identity
author: Codex
date: 2026-10-16
branch: main
cost: { minutes: 301, tokens: 4200337 }
touches:
  - crates/batchalign/src/chat_ops/fa/mod.rs
  - crates/batchalign/src/runner/dispatch/options.rs
  - crates/batchalign/src/worker/request_builder_v2.rs
---

## Decided
Alignment preserves measured word ends by default. Explicit --word-gaps heal enables heuristic projection. WhisperTextMode names model-input shaping separately; inference items and raw cache key builders cannot accept WordGapHealing.

## Tried and ruled out
Changing only the old gap flag would also change Whisper tokenization and cache identity. Preserve historical cache literals according to the actual text mode. CLI, stored options, dispatch, final CHAT and worker-boundary tests cover the split.

## Watch out
The legacy --pauses option still selects character-spaced Whisper input. Onset-only parsing still derives ends. Existing transcript timing reuse and boundary clamps remain separate; preserving measurements does not certify acoustic accuracy or undo previously healed input timing.
