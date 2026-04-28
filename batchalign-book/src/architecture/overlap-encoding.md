# Overlap Encoding: `&*` and `+<` Internals

**Status:** Current
**Last updated:** 2026-03-17

## AST Representation

### `&*` — `OtherSpokenEvent`

**Model (talkbank-tools):** `crates/talkbank-model/src/model/content/other_spoken.rs`

```rust
pub struct OtherSpokenEvent {
    pub speaker: SpeakerCode,     // e.g., "INV"
    pub text: smol_str::SmolStr,  // e.g., "oh_okay_yeah"
    pub span: Span,               // source location (skipped in serde)
}
```

Appears in two enum locations:
- `UtteranceContent::OtherSpokenEvent(OtherSpokenEvent)` — top-level content
- `BracketedItem::OtherSpokenEvent(OtherSpokenEvent)` — inside groups

**Parser (talkbank-tools):**
`crates/talkbank-parser/src/parser/tree_parsing/main_tier/content/`

The tree-sitter grammar accepts `&*` + speaker chars + `:` + non-whitespace chars.

**Serialization:** `&*SPK:text` — roundtrips cleanly via `WriteChat`.

### `+<` — `Linker::LazyOverlapPrecedes`

**Model (talkbank-tools):** `crates/talkbank-model/src/model/content/linker.rs`

```rust
pub enum Linker {
    LazyOverlapPrecedes,  // +<
    OtherCompletion,      // ++
    QuickUptakeOverlap,   // +^
    // ...
}
```

Stored on `TierContent.linkers: TierLinkers` (a `Vec<Linker>` newtype).
Linkers appear at the start of an utterance's content, before words.

## Content Walker Behavior

The content walker (`for_each_leaf` / `for_each_leaf_mut`) **skips**
`OtherSpokenEvent` entirely. It is listed in the no-op match arm alongside
events, pauses, overlap points, and other non-alignable content:

```rust
UtteranceContent::OtherSpokenEvent(_) => {}  // skipped
```

This means `&*` content:
- Is **not counted** in word alignment (Wor, Mor, Pho, Sin domains)
- Does **not appear** in `%wor` tier generation
- Is **not extracted** by `collect_fa_words()` for forced alignment
- Is **not included** in the UTR reference word sequence

## Two-Pass UTR Strategy

When `+<` or CA overlap markers (`⌊`) are present, the alignment pipeline uses a
two-pass UTR strategy. See [Forced Alignment — UTR](../reference/forced-alignment.md)
for the algorithm details and [Content Traversal — Overlap Iterator](content-traversal.md#for_each_overlap_point--overlap-marker-iterator)
for the `for_each_overlap_point` API.

Key points:
- Pass 1 excludes overlap utterances from the global DP alignment
- Pass 2 recovers their timing from the predecessor's audio window
- CA overlap markers (position) narrow the pass-2 search window via proportional onset estimation
- Best-of-both fallback compares FA group counts to avoid regression on non-English

**Code:** `crates/batchalign-chat-ops/src/fa/utr.rs` and
`crates/batchalign-chat-ops/src/fa/utr/two_pass.rs`

## `&*` → `+<` Conversion

A private workspace experiment tool outside this repo
(`../analysis/per-speaker-utr-experiment-2026-03-16/` from the workspace root)
includes a `convert` subcommand that transforms `&*` to separate `+<`
utterances using the typed AST:

1. Walk each utterance's content (including inside groups).
2. Extract `OtherSpokenEvent` nodes, recording speaker + text.
3. Remove them from the host utterance.
4. For each extracted event, create a new `Utterance` with `+<` linker and
   words split from the underscore-joined text.
5. Insert after the host utterance.

### Edge cases handled

- Multiple `&*` in one utterance (each becomes its own `+<` utterance)
- Multi-word `&*` with underscores (`oh_okay_yeah` → `oh okay yeah`)
- `&*` inside groups (`<... &*INV:mhm ...> [//]`)
- Reverse direction (`&*PAR:yeah` on INV's line)
- Host utterances with and without timing bullets
- Host dependent tiers preserved (they were already `&*`-invisible)

## Corpus Statistics

### `&*` (OtherSpokenEvent)

| Corpus | Files | Total markers | Single-word % |
|--------|-------|---------------|---------------|
| ca-data | 256 | 12,016 | 96% |
| aphasia-data | 644 | 10,161 | 88% |
| rhd-data | 190 | 5,160 | 83% |
| psychosis-data | 236 | 2,799 | 98% |
| tbi-data | 135 | 2,105 | 90% |
| dementia-data | 390 | 1,680 | 89% |
| slabank-data | 191 | 774 | — |
| childes-data | 146 | 411 | — |
| **Total** | | **~35,000** | **91%** |

Top words: mhm (~12,500), yeah (~5,500), okay (~3,300), mm (~1,400).

### `+<` (LazyOverlapPrecedes)

| Corpus | Files | `+<` utterances |
|--------|-------|----------------|
| childes-data | 10,596 | 194,720 |
| phon-data | 614 | 50,892 |
| biling-data | 248 | 37,727 |
| aphasia-data | 1,241 | 15,720 |
| tbi-data | 251 | 7,469 |
| ca-data | 242 | 6,606 |
| dementia-data | 1,536 | 4,745 |
| **Total** | | **~327,000** |

Of these, ~131,000 (40%) already have timing bullets.

## File Locations

| File | Purpose |
|------|---------|
| `crates/batchalign-chat-ops/src/fa/utr.rs` | UtrStrategy trait, GlobalUtr, select_strategy, run_global_utr |
| `crates/batchalign-chat-ops/src/fa/utr/two_pass.rs` | TwoPassOverlapUtr, recover_overlap_timing |
| `crates/batchalign-chat-ops/src/fa/tests.rs` | Integration tests for both strategies |
| `crates/batchalign-app/src/runner/dispatch/utr.rs` | resolve_strategy, UtrPassContext.overlap_strategy |
| `crates/batchalign-app/src/types/options.rs` | UtrOverlapStrategy enum |
| `crates/batchalign-cli/src/args/commands.rs` | `--utr-strategy` CLI flag |
