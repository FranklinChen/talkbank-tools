# Alignment Structures and Algorithms

**Status:** Current
**Last updated:** 2026-03-26 00:52 EDT

This document inventories all alignment data structures, algorithms, and
abstractions used in batchalign3 and its upstream dependency `talkbank-tools`.

## Alignment Layers

Alignment in the TalkBank toolchain operates at three distinct layers:

| Layer | Where | Purpose |
|-------|-------|---------|
| **Tier alignment** | `talkbank-model` | 1:1 mapping between main tier and dependent tiers (%mor, %pho, %wor, %sin, %gra) |
| **NLP alignment** | `batchalign-chat-ops` | Mapping NLP model output (Stanza tokens, FA timings) back to CHAT AST words |
| **Sequence alignment** | `batchalign-chat-ops` | Edit-distance alignment for WER evaluation and transcript comparison |

These layers use different data structures because they solve fundamentally
different problems. Tier alignment is structural (counting AST nodes),
NLP alignment is positional (character/token indices), and sequence alignment
is edit-distance (Hirschberg DP).

---

## Layer 1: Tier Alignment (talkbank-model)

**Location:** `talkbank-tools/crates/talkbank-model/src/alignment/`

Tier alignment validates that dependent tiers have the correct number of items
relative to the main tier. Each domain has different counting rules for the
same utterance content.

### Core Types

#### `TierDomain` (`helpers/domain.rs`)

```rust
enum TierDomain { Mor, Pho, Sin, Wor }
```

Determines which main-tier elements participate in alignment counting.
The same utterance produces different counts per domain:

| Domain | Skips retraces? | Includes pauses? | PhoGroup is... | SinGroup is... |
|--------|----------------|-------------------|----------------|----------------|
| Mor | Yes | No | Recurse into | Recurse into |
| Pho | No | Yes | 1 atomic unit | Skip (0) |
| Sin | No | No | Skip (0) | 1 atomic unit |
| Wor | No | No | Recurse into | Recurse into |

#### `AlignmentPair` (`types.rs`)

```rust
struct AlignmentPair {
    source_index: Option<usize>,  // main tier position
    target_index: Option<usize>,  // dependent tier position
}
```

The universal index-pair primitive. `Some`/`Some` = matched, one `None` =
insertion/deletion placeholder for mismatch diagnostics.

#### Per-Domain Alignment Results

| Type | Function | Source → Target |
|------|----------|-----------------|
| `MorAlignment` | `align_main_to_mor()` | Main → %mor items |
| `PhoAlignment` | `align_main_to_pho()` | Main → %pho tokens |
| `SinAlignment` | `align_main_to_sin()` | Main → %sin tokens |
| `WorAlignment` | `align_main_to_wor()` | Main → %wor tokens |
| `GraAlignment` | `align_mor_to_gra()` | %mor chunks → %gra relations |

Each contains a `Vec<AlignmentPair>` plus domain-specific error diagnostics.
All implement `TierAlignmentResult` (shared trait for generic inspection).

### Trait Abstractions (`traits.rs`)

| Trait | Purpose | Implementors |
|-------|---------|-------------|
| `IndexPair` | `source()/target()` on any pair type | `AlignmentPair`, `GraAlignmentPair` |
| `TierAlignmentResult` | `pairs()/errors()/push_*()` accumulator | All 5 alignment result types |
| `AlignableTier` | What a tier provides for generic alignment | `PhoTier`, `SinTier`, `WorTier` |
| `TierCountable` | `count_tier_positions()/collect_tier_items()` methods | `[UtteranceContent]` |

The generic `positional_align()` function uses `AlignableTier` to eliminate
duplication: `align_main_to_pho()`, `align_main_to_sin()`, and
`align_main_to_wor()` are thin wrappers around it.

### Counting Rules (`helpers/count.rs`)

`count_tier_positions()` and `count_tier_positions_until()` traverse the
utterance content tree, applying domain-specific rules:

- **Words**: filtered by `counts_for_tier(word, domain)` — Mor excludes
  fragments/untranscribed; Wor excludes nonwords, fragments, timing tokens;
  Pho/Sin include everything
- **ReplacedWords**: Mor/Wor align to replacement words; Pho/Sin align to
  the original (what was spoken)
- **Separators**: only tag markers (`,` `„` `‡`) count, and only for Mor
- **Retrace**: dedicated `Retrace` variant — skipped entirely for Mor, recursed for other domains
- **PhoGroup/SinGroup**: atomic units in their own domain, recurse in others

Both `UtteranceContent` (24 variants) and `BracketedItem` (22 variants) are
handled with exhaustive match — no catch-all arms.

### Content Walker (`helpers/walk/`)

`walk_words()` / `walk_words_mut()` centralize recursive traversal
so callers provide only leaf-handling closures:

```rust
walk_words(&utterance.content, Some(TierDomain::Mor), |leaf| {
    match leaf {
        WordItem::Word(word) => { /* handle */ }
        WordItem::ReplacedWord(replaced) => { /* handle */ }
        WordItem::Separator(sep) => { /* handle */ }
    }
});
```

Domain-aware gating is built into the walker:
- `Some(Mor)` → skip `Retrace` variants (not morphologically analyzed)
- `Some(Pho)` → skip PhoGroup (atomic), `Some(Sin)` → skip SinGroup (atomic)
- `None` → recurse all groups unconditionally

**Not suitable for**: container mutation (`strip_timing_from_content()` uses
`retain()`), and `count.rs` (Pho/Sin treat PhoGroup/SinGroup as counted
atomic units rather than skipped containers).

### Parse-Health Gating

Alignment diagnostics consult `ParseHealth` metadata before reporting
mismatches. If a dependent tier's domain is parse-tainted (malformed input),
mismatch errors are suppressed for that domain pair to avoid false positives.

---

## Layer 2: NLP Alignment (batchalign-chat-ops)

**Location:** `batchalign3/crates/batchalign-chat-ops/src/`

NLP alignment maps external model outputs (Stanza tokens, FA word timings)
back to CHAT AST positions. All algorithms here are deterministic — no DP
remapping at runtime.

### Word Extraction (`extract.rs`)

`extract_words()` uses the content walker to pull words from the AST in
domain-specific order. Returns `Vec<ExtractedWord>` with:

```rust
struct ExtractedWord {
    text: String,              // cleaned word text
    word_index: usize,         // position in utterance content
    is_separator: bool,        // tag-marker separator
    special_form: Option<...>, // retrace, filler, etc.
}
```

The separator counter sync gotcha: tag-marker separators (`,` `„` `‡`) are
included as NLP words in Mor domain because they have %mor items (`cm|cm`,
`end|end`, `beg|beg`). Any code counting words must also count these.

### Retokenize Mapping (`retokenize/mapping.rs`)

Maps original CHAT words to Stanza token indices after Stanza may have
split or merged words.

```rust
struct WordTokenMapping {
    inner: Vec<SmallVec<[usize; 4]>>,  // word_idx → [token_idx...]
}
```

**Design choices:**
- Dense `Vec` indexed by word position (O(1) lookup, no hashing)
- `SmallVec<[usize; 4]>` keeps 1-2 token mappings inline (no heap alloc)
- Two-stage algorithm: deterministic span-join first, length-aware monotonic
  fallback when text diverges

### Tokenizer Realignment (`tokenizer_realign/`)

Maps Stanza's re-tokenized output back to original CHAT words using
character-position arrays:

```rust
fn align_tokens(
    original_words: &[String],
    stanza_tokens: &[String],
    alpha2: &str,
) -> Vec<PatchedToken>
```

**Algorithm (O(n) character-level):**
1. Concatenate original words, build per-char owner array
2. Concatenate Stanza tokens, build per-char owner array
3. Walk both arrays in parallel to determine which original word each
   Stanza token belongs to
4. Apply language-specific MWT patches (French, Italian, Portuguese, Dutch)

**Output types:**
- `PatchedToken::Plain(String)` — token maps cleanly to one CHAT word
- `PatchedToken::Hint(String, bool)` — token with MWT expansion hint

### FA Response Alignment (`fa/alignment.rs`)

Maps forced-alignment timing responses back to extracted words:

- **Indexed path**: 1:1 by position, no remapping needed
- **Token-level path**: deterministic token→word stitching when FA returns
  sub-word tokens; unmatched words remain untimed (no DP)

### FA Injection (`fa/injection.rs`)

Injects word-level timings into the AST using the content walker. Walks
utterance content with `walk_words_mut()` in Wor domain, applying timing
bullets to each word in traversal order.

### FA Postprocess (`fa/postprocess.rs`)

Post-alignment cleanup:
- `enforce_monotonicity()` — strips timing from regression violations (E362)
- `strip_e704_same_speaker_overlaps()` — removes conflicting same-speaker timing
- Proportional boundary estimation for untimed utterances

---

## Layer 3: Sequence Alignment (dp_align.rs)

**Location:** `batchalign-chat-ops/src/dp_align.rs`

Hirschberg divide-and-conquer edit-distance alignment. Linear space O(mn),
with a `SMALL_CUTOFF = 2048` threshold for the full-table fast path.

### Cost Model

| Operation | Cost |
|-----------|------|
| Match | 0 |
| Substitution | 2 |
| Insertion/Deletion | 1 |

### Optimizations

**Prefix/suffix stripping.** Before entering the O(mn) DP core, `align()`
and `align_chars()` strip matching prefixes and suffixes in O(n). For the
primary use case (WER/transcript comparison where accuracy is 80-95%), this
reduces the effective problem size by 10-100x — the DP only runs on the
differing middle portion.

**Generic `Alignable` trait.** Both `String` (word-level) and `char`
(character-level) variants share a single generic implementation via the
`Alignable` trait (`matches()` + `to_key()`). Monomorphization ensures
zero overhead while eliminating ~200 lines of code duplication.

**Flat table for small problems.** `align_small()` uses a flat `Vec`
instead of `Vec<Vec<...>>`, reducing allocation count from `rows + 1` to 1.

**Scratch buffer reuse.** `row_costs()` reuses two `Vec<usize>` buffers
across DP rows via `std::mem::swap`, avoiding per-row allocation.

### Types

```rust
enum AlignResult {
    Match { key, payload_idx, reference_idx },
    ExtraPayload { key, payload_idx },
    ExtraReference { key, reference_idx },
}

enum MatchMode {
    Exact,           // byte-for-byte
    CaseInsensitive, // for ASR vs reference comparison
}
```

### Call Sites

| Caller | Purpose |
|--------|---------|
| `benchmark.rs` | WER computation (hypothesis vs reference) |
| `compare.rs` | Transcript comparison, `%xsrep` / `%xsmor` tier annotation |
| Python `batchalign_core.dp_align` | PyO3 bridge for Python callers |

### DP Policy

Runtime DP is restricted to intrinsic uses (WER, CTC, DTW) and
architecturally unavoidable cases. A policy test
(`test_dp_allowlist.py`) fails CI if new runtime DP callsites appear
outside the allowlist. See [Dynamic Programming](dynamic-programming.md)
for the full inventory and necessity assessment.

---

## Cross-Layer Design Principles

1. **No string hacking.** All alignment operates on typed AST structures
   (`Word`, `MorTier`, `AlignmentPair`), never on serialized CHAT text.

2. **Domain-aware from the start.** `TierDomain` gates traversal at the
   walker level, so downstream code never needs to re-implement retrace/group
   skipping logic.

3. **Deterministic over approximate.** Runtime alignment paths (FA injection,
   retokenize, tokenizer realign) use deterministic algorithms. DP is reserved
   for intrinsically approximate problems (WER, CTC forced alignment).

4. **Dense indexed structures.** `WordTokenMapping` uses `Vec<SmallVec>` instead
   of `HashMap` for O(1) lookup without hashing. `AlignmentPair` uses
   `Option<usize>` indices rather than storing cloned data.

5. **Exhaustive matching.** Every `match` on `UtteranceContent` (24 variants)
   or `BracketedItem` (22 variants) lists all variants explicitly — no
   catch-all `_ =>` arms that could silently drop new content types.

6. **Content walker as shared primitive.** `walk_words()` eliminates
   ~330 lines of duplicated traversal boilerplate across 7 call sites.
   Callers provide only leaf-handling closures.

---

## Key Files

| File | Lines | Purpose |
|------|-------|---------|
| `talkbank-model/src/alignment/mod.rs` | 104 | Module root, public API exports |
| `talkbank-model/src/alignment/types.rs` | 53 | `AlignmentPair` |
| `talkbank-model/src/alignment/helpers/domain.rs` | 41 | `TierDomain` enum |
| `talkbank-model/src/alignment/helpers/rules.rs` | 170 | `counts_for_tier()`, `should_skip_group()`, tag-marker predicates |
| `talkbank-model/src/alignment/helpers/count.rs` | 572 | Domain-specific counting and extraction |
| `talkbank-model/src/alignment/helpers/walk/mod.rs` | ~200 | `walk_words()` / `walk_words_mut()` |
| `talkbank-model/src/alignment/mor.rs` | — | `align_main_to_mor()` |
| `talkbank-model/src/alignment/pho.rs` | — | `align_main_to_pho()` |
| `talkbank-model/src/alignment/sin.rs` | — | `align_main_to_sin()` |
| `talkbank-model/src/alignment/wor.rs` | — | `align_main_to_wor()` |
| `talkbank-model/src/alignment/gra/` | — | `align_mor_to_gra()` (%mor chunks → %gra) |
| `chat-ops/src/extract.rs` | — | Domain-aware word extraction from AST |
| `chat-ops/src/retokenize/mapping.rs` | — | `WordTokenMapping` (word→token dense map) |
| `chat-ops/src/tokenizer_realign/mod.rs` | ~280 | Character-position realignment |
| `chat-ops/src/tokenizer_realign/mwt_overrides.rs` | ~300 | Language-specific MWT patches |
| `chat-ops/src/fa/alignment.rs` | — | FA response parsing and deterministic alignment |
| `chat-ops/src/fa/injection.rs` | — | Timing injection into AST via content walker |
| `chat-ops/src/fa/postprocess.rs` | — | Monotonicity enforcement, overlap cleanup |
| `chat-ops/src/dp_align.rs` | — | Hirschberg sequence alignment |
| `chat-ops/src/compare.rs` | — | WER computation using `dp_align` |
