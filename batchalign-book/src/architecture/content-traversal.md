# Content Traversal Patterns

**Status:** Current
**Last updated:** 2026-03-18

Every NLP task in batchalign3 needs to walk CHAT main-tier content — extracting words for inference, injecting results back, or scanning for structural markers. This chapter explains the traversal patterns and when to use each one.

## The Two-Layer Walker Design

The walker is a two-layer design:
- **`walk_content`** — generic, visits all content items (not covered here; used for custom traversals)
- **`walk_words`** — filtered to words and separators, with domain-aware gating

Groups are descended transparently, and `AnnotatedWord`/`Event`/`Action` are unwrapped automatically. Callers receive only `WordItem` values.

## The Content Walker: `walk_words`

The primary traversal primitive is `walk_words()` / `walk_words_mut()` from `talkbank-model`. It walks all three content levels (see [CHAT Content Model](chat-content-model.md)) and calls a closure for each word, replaced word, or separator:

```rust
use talkbank_model::alignment::helpers::{
    walk_words, walk_words_mut,
    WordItem, WordItemMut,
    TierDomain,
};

// Read-only traversal
walk_words(content, Some(TierDomain::Wor), &mut |leaf| {
    match leaf {
        WordItem::Word(word) => {
            // Process a word
        }
        WordItem::ReplacedWord(replaced) => {
            // Process a replaced word
        }
        WordItem::Separator(sep) => {
            // Process a separator (Mor domain only)
        }
    }
});

// Mutable traversal (e.g., injecting timing)
walk_words_mut(content, None, &mut |leaf| {
    match leaf {
        WordItemMut::Word(word) => {
            word.inline_bullet = Some(bullet);
        }
        // ...
    }
});
```

### Domain-Aware Gating

The second argument controls which groups the walker enters:

| Argument | Behavior | Use when |
|----------|----------|----------|
| `None` | Recurse into all groups unconditionally | FA (need all words), %wor generation |
| `Some(Mor)` | Skip retrace groups (`<word> [/]`) | %mor/%gra alignment |
| `Some(Pho)` | Skip `PhoGroup` (`‹...›`) | %pho alignment |
| `Some(Sin)` | Skip `SinGroup` (`〔...〕`) | %sin alignment |
| `Some(Wor)` | No skipping (same as `None`) | %wor alignment |

### What the Walker Handles

The walker handles the full traversal of all 24 `UtteranceContent` variants and all 22 `BracketedItem` variants, including:
- Descending into `Group`, `AnnotatedGroup`, `PhoGroup`, `SinGroup`, `Quotation` transparently
- Unwrapping `AnnotatedWord`, `AnnotatedEvent`, `AnnotatedAction` to access the inner item
- Handling `ReplacedWord` (choosing replacement words vs surface form)
- Skipping non-word items (pauses, events, actions, overlap points, etc.)

### What the Walker Does NOT Handle

The walker only visits words and separators. It does **not** visit:
- **Overlap markers** (`OverlapPoint`) — including intra-word markers in `WordContent`
- **CA elements** within words
- **Events, pauses, actions** — these are skipped
- **Internal bullets** — timing markers within content

If you need to visit these, you must write a custom traversal. See the validation module (`talkbank-model/validation/utterance/overlap.rs`) for a reference implementation of overlap marker collection.

## The Extract → Infer → Inject Pattern

Every NLP task in batchalign3 follows the same pattern:

```
┌─────────┐     ┌──────────┐     ┌──────────┐     ┌──────────┐
│  Parse   │────▶│ Extract  │────▶│  Infer   │────▶│  Inject  │
│  CHAT    │     │  Words   │     │  (Python) │     │  Results │
└─────────┘     └──────────┘     └──────────┘     └──────────┘
    Rust            Rust              IPC             Rust
```

1. **Parse:** `parse_lenient()` or `parse_strict()` → `ChatFile` AST
2. **Extract:** Walk content with `walk_words`, collect words/payloads
3. **Infer:** Send structured payload to Python worker via IPC, receive raw ML output
4. **Inject:** Walk content again with `walk_words_mut`, inject results back into the AST

### Per-Task Implementation

| Task | Extract module | Inject module | Domain |
|------|---------------|---------------|--------|
| Morphosyntax | `morphosyntax.rs` (`collect_payloads`) | `morphosyntax.rs` (`inject`) | `Mor` |
| Utterance segmentation | `utseg.rs` (`collect_payloads`) | `utseg.rs` (`apply`) | `Wor` |
| Translation | `translate.rs` (`collect_payloads`) | `translate.rs` (`inject_translation`) | N/A (text) |
| Coreference | `coref.rs` (`collect_payloads`) | `coref.rs` (`inject_coref`) | `Wor` |
| Forced alignment | `fa/extraction.rs` (`collect_fa_words`) | `fa/injection.rs` | `None` (all words) |

### Word Extraction Example

The FA extraction module is the simplest example:

```rust
pub fn collect_fa_words(content: &[UtteranceContent], out: &mut Vec<String>) {
    walk_words(content, None, &mut |leaf| match leaf {
        WordItem::Word(word) => {
            if counts_for_tier(word, TierDomain::Wor) {
                out.push(word.cleaned_text().to_string());
            }
        }
        WordItem::ReplacedWord(replaced) => {
            // Use replacement words if available, else surface form
            if !replaced.replacement.words.is_empty() {
                for word in &replaced.replacement.words {
                    if counts_for_tier(word, TierDomain::Wor) {
                        out.push(word.cleaned_text().to_string());
                    }
                }
            } else if counts_for_tier(&replaced.word, TierDomain::Wor) {
                out.push(replaced.word.cleaned_text().to_string());
            }
        }
        WordItem::Separator(_) => {}
    });
}
```

### Result Injection Example

FA injection walks the same content and assigns timing bullets:

```rust
walk_words_mut(content, None, &mut |leaf| {
    match leaf {
        WordItemMut::Word(word) => {
            if counts_for_tier(word, TierDomain::Wor) {
                if let Some(timing) = word_timings.get(word_idx) {
                    word.inline_bullet = Some(Bullet::new(
                        timing.start_ms, timing.end_ms,
                    ));
                }
                word_idx += 1;
            }
        }
        // ...
    }
});
```

**Critical invariant:** The extract and inject traversals must visit words in the same order. The content walker guarantees this — same domain argument, same traversal order.

## Custom Traversals

When the content walker doesn't fit (e.g., you need overlap marker positions, not just words), write a custom traversal following the validation module pattern:

1. Match all `UtteranceContent` variants explicitly (no `_ =>`)
2. Recurse into groups via `BracketedContent.content.0`
3. Check inside words via `Word.content` for `WordContent` items
4. Handle all `Annotated<T>` variants by accessing `.inner`

### `walk_overlap_points` — Overlap Marker Iterator

For overlap markers specifically, `talkbank-model` provides a second iterator:

```rust
use talkbank_model::alignment::helpers::{
    walk_overlap_points, OverlapPointVisit,
};

walk_overlap_points(content, &mut |visit: OverlapPointVisit| {
    println!(
        "Marker {:?} (index {:?}) at word position {}",
        visit.point.kind,
        visit.point.index,
        visit.word_position,
    );
});
```

This visits every `OverlapPoint` at all three content levels (top-level, inside groups, inside words) with its word-position context. Used by both the alignment pipeline (onset estimation) and the validator (pairing checks).

For pre-paired regions matched by index, use `extract_overlap_info()` which builds `OverlapRegion` structs from the raw markers.

### Region-Based Overlap Analysis: `extract_overlap_info`

For higher-level overlap analysis, `extract_overlap_info()` collects all markers
and pairs them by (kind, index) into `OverlapRegion` structs:

```rust
use talkbank_model::alignment::helpers::{
    extract_overlap_info, OverlapRegion, OverlapRegionKind,
};

let info = extract_overlap_info(&utterance.main.content.content.0);

for region in &info.regions {
    match (region.kind, region.is_well_paired()) {
        (OverlapRegionKind::Top, true) => {
            // Well-paired ⌈...⌉ region
            let begin = region.begin_at_word.unwrap();
            let end = region.end_at_word.unwrap();
            println!("Top overlap spans words {begin}..{end}");
        }
        (OverlapRegionKind::Bottom, _) => {
            // ⌊...⌋ region (may be unpaired if onset-only)
        }
        (_, false) => {
            // Unpaired marker — onset-only or orphaned closing
        }
    }
}

// Proportional onset estimation for alignment windowing
if let Some(fraction) = info.top_onset_fraction() {
    let onset_ms = info.estimate_onset_ms(start_ms, end_ms).unwrap();
}
```

**Index-aware pairing:** Markers are matched by both kind (top/bottom) and index.
`⌈2...⌉2` forms a separate region from `⌈...⌉`. Mismatched indices (⌈2 with ⌉3)
leave both unpaired.

**Unpaired markers:** Onset-only marking (⌈ without ⌉) is a legitimate CA
practice. The region will have `begin_at_word = Some(n)` but `end_at_word = None`,
and `is_well_paired()` returns false. The onset fraction is still usable.

### Cross-Utterance Overlap Groups

For whole-file analysis, `analyze_file_overlaps()` matches top regions (⌈) with
bottom regions (⌊) across utterances with 1:N support — one speaker's ⌈ can be
matched by multiple respondents' ⌊. Used by the E347 validator and the
`chatter debug overlap-audit` command.

### Validation of Overlap Markers

The validator uses `extract_overlap_info` and `analyze_file_overlaps` to check:
- **E348** (intra-utterance): Unpaired markers within a single utterance (warning)
- **E347** (cross-utterance): Orphaned tops/bottoms with 1:N matching (warning)
- **E373**: Invalid overlap index values (must be 2-9)
- **E704**: Same speaker encoding both top and bottom overlap (error)

## Adding a New Traversal

When implementing a new NLP task or analysis:

1. **Can you use `walk_words`?** If you only need words/separators, use it. Pick the right `TierDomain`.
2. **Need non-leaf items too?** Write a custom traversal. Copy the structure from `validation/utterance/overlap.rs`.
3. **Consider adding to talkbank-model:** If your traversal is generic CHAT logic (not batchalign-specific), it belongs in `talkbank-model` alongside `walk_words`. Use the closure-based internal iterator pattern.
4. **Test with intra-word content:** CA transcriptions embed overlap markers, prosody markers, and other content inside words. Test with files from `corpus/reference/ca/`.
