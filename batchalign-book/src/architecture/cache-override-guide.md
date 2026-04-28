# Cache Override Guide

**Status:** Current
**Last updated:** 2026-04-08 15:08 EDT

When fixing a bug or changing behavior, the first question is: **do deployed
users need `--override-media-cache`?** This guide provides the mental model and
decision matrix to answer that question quickly.

For what's cached and how keys work, see [Caching](caching.md). This page is
the complement: what sits *inside* vs *outside* the cache boundary, and what
that means for deploying fixes.

## Core Mental Model

Every cached command has a **cache boundary** — a line between what's stored in
the cache (raw ML output) and what's computed fresh on every run (Rust
post-processing). The rule is simple:

- **Change inside the boundary** (the cached value itself is wrong) →
  `--override-media-cache` needed
- **Change outside the boundary** (post-processing that runs after retrieval) →
  fix applies automatically, no override needed

## Per-Command Cache Boundaries

### Morphosyntax

| Stage | Inside/Outside | Code |
|-------|---------------|------|
| Word extraction from CHAT AST | Outside (pre-cache) | `extract.rs` |
| Cache key: `BLAKE3(words \| lang \| MWT lexicon)` | — | `morphosyntax/cache.rs` |
| Stanza inference → raw %mor/%gra JSON | **Inside** | Python `morphosyntax.py` |
| Retokenization (Stanza word splits/merges) | **Inside** (happens before cache store) | `retokenize/` |
| Deserialize cached JSON | Outside | `morphosyntax/cache.rs` |
| Patch MorTier terminator | Outside | `inject_from_cache()` |
| Validate MOR/GRA chunk count alignment | Outside | `inject_from_cache()` |
| Inject %mor/%gra into AST | Outside | `morphosyntax/cache.rs` |

### Utterance Segmentation (utseg)

| Stage | Inside/Outside | Code |
|-------|---------------|------|
| Word extraction | Outside (pre-cache) | `extract.rs` |
| Cache key: `BLAKE3(words \| lang)` | — | `utseg.rs` |
| Stanza constituency parse → boundary assignments | **Inside** | Python `utseg.py` |
| `apply_utseg_results()` (split utterances) | Outside | `utseg.rs` |

### Translation

| Stage | Inside/Outside | Code |
|-------|---------------|------|
| `preprocess_for_translate()` (CJK space stripping) | Outside (pre-cache) | `translate.rs` |
| Cache key: `BLAKE3(text \| src_lang \| tgt_lang)` | — | `translate.rs` |
| Google Translate / Seamless M4T → translated string | **Inside** | Python `translate.py` |
| `postprocess_translation()` (quote normalization, punctuation spacing) | Outside | `translate.rs` |
| Inject as %xtra tier | Outside | `translate.rs` |

### Forced Alignment (FA)

| Stage | Inside/Outside | Code |
|-------|---------------|------|
| Tier 1: check reusable %wor timing | Outside (bypasses cache entirely) | `fa/mod.rs` |
| Group utterances by time windows | Outside (pre-cache) | `fa/mod.rs` |
| Word extraction per group | Outside (pre-cache) | `fa/mod.rs` |
| Cache key: `BLAKE3(audio_identity \| start_ms \| end_ms \| text \| timing_flag \| engine)` | — | `fa/mod.rs` |
| Whisper/Wave2Vec inference → `Vec<Option<WordTiming>>` | **Inside** | Python `fa.py` |
| `postprocess_utterance_timings()` | Outside | `fa/postprocess.rs` |
| - Continuous mode: backward end-time propagation | Outside | `fa/postprocess.rs` |
| - WithPauses mode: use engine end times | Outside | `fa/postprocess.rs` |
| - Clamp to utterance bullet range | Outside | `fa/postprocess.rs` |
| `update_utterance_bullet()` (overwrite UTR hints; union with authoritative) | Outside | `fa/orchestrate.rs` |
| %wor tier generation | Outside | `fa/orchestrate.rs` |
| E362/E704 enforcement | Outside | validation layer |

### UTR (Utterance Timing Recovery)

| Stage | Inside/Outside | Code |
|-------|---------------|------|
| Full-file key: `BLAKE3(utr_asr \| audio_identity \| lang)` | — | `fa/utr.rs` |
| Segment key: `BLAKE3(utr_asr_segment \| audio_identity \| start_ms \| end_ms \| lang)` | — | `fa/utr.rs` |
| ASR inference → `Vec<AsrTimingToken>` | **Inside** | Python `asr.py` |
| Global Hirschberg DP alignment (words ↔ ASR tokens) | Outside | `fa/utr.rs` |
| Utterance bullet injection | Outside | `fa/utr.rs` |

### Coref

Not cached. Document-level scope requires full context.

### Transcribe

Not cached at file level. Sub-tasks (FA, UTR) cache individually. ASR
post-processing (compound merging, number expansion, Cantonese normalization,
retokenization) runs fresh every time.

## Decision Matrix

| What I changed | Override needed? | Why |
|---------------|-----------------|-----|
| Post-processing logic (injection, bullet computation, %wor generation, retokenization after cache, terminator patching) | **No** | Runs after cache retrieval — cached value is still correct |
| Cache key computation | **No** | Old entries become orphans (different key = automatic miss). New keys miss and re-infer. |
| Word extraction logic (changes which words are sent to the model) | **Yes** | Cached result was computed from different input words |
| ML model/engine code (Python worker) | **Automatic** if `engine_version` changes; **Yes** if version string unchanged | Engine version scoping handles model upgrades transparently |
| Serialization format of cached value | **Usually no** | Deserialization failure triggers re-inference with a warning (see below) |
| Parse logic (changes how CHAT is parsed before extraction) | **Depends** | If extraction produces different words → yes (different key). If same words → no. |
| Pre-cache text normalization (e.g., `preprocess_for_translate`) | **Yes** | Key is computed from normalized text; same key now maps to wrong cached result |

## Worked Example: The Bullet-Shrinking Bug

**Bug (2026-03-16, a user, ACWT corpus):** `update_utterance_bullet()` computed
the FA timing span from only the aligned words, then *replaced* the original
utterance bullet with it. Fillers, pauses, and gestures (which FA cannot align)
lost their timing coverage.

**Analysis:**

1. What's cached? `Vec<Option<WordTiming>>` — the raw per-word timings from
   Whisper/Wave2Vec.
2. Where's the bug? In `update_utterance_bullet()` — post-processing that runs
   *after* cache retrieval.
3. Are the cached timings wrong? No — the word-level timings are correct. The
   bug was in how we used them to update the utterance bullet.

**Fix:** `update_utterance_bullet()` now uses `BulletSource` provenance to decide
whether to overwrite or union:

- **`BulletSource::Authoritative`** (hand-linked, parsed from file, or FA-derived):
  **union** — never shrink. Preserves filler/gesture coverage.
- **`BulletSource::Utr`** (provisional UTR hint, set by `Bullet::utr_hint()`):
  **overwrite** — FA word span is authoritative. The UTR window was a rough
  estimate; the FA alignment is more precise.

**Verdict: No `--override-media-cache` needed.** The source-aware update logic
applies automatically to cached FA results. Both behaviors are correct for their
respective bullet types.

## Self-Correcting Cache Purges

When post-serialization validation detects an invalid result, the server
auto-deletes the cache entry that produced it and writes a bug report to
`~/.batchalign3/bug-reports/`. This means:

- **Helps when:** A cached value produces output that fails validation. Next run
  re-infers and (if the underlying model is correct) produces valid output.
- **Does NOT help when:** The cached value is *wrong but valid* — e.g., it
  passes validation but contains incorrect timings. Validation can't catch
  semantic correctness.
- **Does NOT help when:** The post-processing is buggy — the cache entry will be
  deleted, but re-inference produces the same cached value, which the same buggy
  post-processing corrupts again. Fix the post-processing first.

## Deserialization Failure Fallback

If a cached entry fails to deserialize (e.g., because the stored format changed
between versions), the cache layer logs a warning and falls back to
re-inference. The stale entry is **not** automatically deleted — it becomes a
permanent miss that re-infers every time until `--override-media-cache` forces a fresh
store. This is conservative: it avoids data loss from format migration bugs.

## Deployment Checklist

When deploying a fix to the fleet (net, worker-machine, etc.):

1. **Identify the change category** using the decision matrix above.
2. **If override is NOT needed:** Deploy the new binary. Cached results are
   reprocessed through the fixed post-processing automatically.
3. **If override IS needed:** Deploy the new binary, then re-run affected
   commands with `--override-media-cache` on the target corpora. For large corpora,
   consider running only on affected files rather than the full dataset.
4. **If engine version changed:** No action needed — version scoping
   automatically invalidates stale entries. Verify by checking cache stats in
   server logs (should show misses on first run).
5. **If unsure:** `--override-media-cache` is always safe. The cost is re-inference
   time, not correctness. When in doubt, override.
