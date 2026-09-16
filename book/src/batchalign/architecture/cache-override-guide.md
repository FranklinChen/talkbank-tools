# Cache Policy Guide

**Status:** Current
**Last updated:** 2026-09-15 19:40 EDT

When fixing a bug or changing behavior, ask two questions: **does the run need
fresh inference (`--override-media-cache`), or must it prove that reusable
evidence is sufficient (`--require-media-cache`)?** This guide provides the
mental model and decision matrix.

For what's cached and how keys work, see [Audio-Task Cache](../../architecture/runtime/audio-task-cache.md). This page is
the complement: what sits *inside* vs *outside* the cache boundary, and what
that means for deploying fixes.

## Core Mental Model

Every cached command has a **cache boundary**: a line between what's stored in
the cache (raw ML output) and what's computed fresh on every run (Rust
post-processing). The rule is simple:

- **Change inside the boundary** (the cached value itself is wrong) →
  `--override-media-cache` needed
- **Change outside the boundary** (post-processing that runs after retrieval) →
  fix applies automatically, no override needed

## Per-Command Cache Boundaries

### Morphosyntax

Not cached. Stanza inference, retokenization, validation, and injection run on
every invocation.

### Utterance Segmentation (utseg)

Not cached. Model inference and boundary application run on every invocation.

### Translation

Not cached. Provider/model inference, post-processing, and `%xtra` injection
run on every invocation.

### Forced Alignment (FA)

| Stage | Inside/Outside | Code |
|-------|---------------|------|
| Tier 1: check reusable %wor timing | Outside (bypasses cache entirely) | `fa/mod.rs` |
| Group utterances by time windows | Outside (pre-cache) | `fa/mod.rs` |
| Word extraction per group | Outside (pre-cache) | `fa/mod.rs` |
| Cache key: `BLAKE3(audio_identity \| start_ms \| end_ms \| text \| healing_flag \| engine)` | Boundary | `chat_ops/fa/mod.rs` |
| Whisper/Wave2Vec inference → `Vec<Option<WordTiming>>` | **Inside** | Python `fa.py` |
| `postprocess_utterance_timings()` | Outside | `fa/postprocess.rs` |
| - `WordGapHealing::Heal`: backward end-time propagation, bounded by plausibility caps | Outside | `fa/postprocess.rs` |
| - `WordGapHealing::PreserveMeasured` (`--pauses`): leave each word's end alone | Outside | `fa/postprocess.rs` |
| - Clamp to utterance bullet range | Outside | `fa/postprocess.rs` |
| `update_utterance_bullet()` (overwrite UTR hints; union with authoritative) | Outside | `fa/orchestrate.rs` |
| %wor tier generation | Outside | `fa/orchestrate.rs` |
| E362/E704 enforcement | Outside | validation layer |

### UTR (Utterance Timing Recovery)

| Stage | Inside/Outside | Code |
|-------|---------------|------|
| Full-file key: `BLAKE3(utr_asr_v2 \| UTR engine \| audio_identity \| lang)` | Boundary | `chat_ops/fa/utr.rs` |
| Segment key: `BLAKE3(utr_asr_segment_v2 \| UTR engine \| audio_identity \| start_ms \| end_ms \| lang)` | Boundary | `chat_ops/fa/utr.rs` |
| Namespace: `utr-asr-v1:<UTR engine>:<composition>` then one `\|<role>=<id>@<revision>` per model (`UtrAsrCacheNamespace`, the only namespace type `cache::tasks::UTR_ASR` accepts), the UTR engine's own and its pinned models, never the FA engine's. A plan that is not fully pinned is `Floating` and is neither read nor written | Boundary | `cache/mod.rs` |
| ASR inference → `Vec<AsrTimingToken>` | **Inside** | Python `asr.py` |
| Global Hirschberg DP alignment (words ↔ ASR tokens) | Outside | `runner/dispatch/utr.rs` |
| Utterance bullet injection | Outside | `runner/dispatch/utr.rs` |

### Coref

Not cached. Document-level scope requires full context.

### Transcribe

Not cached at file level. Raw Rev.AI transcript evidence and dedicated speaker
evidence are cached before local projection; speaker evidence has separate raw
and normalized layers. Other ordinary ASR output is not cached. ASR
post-processing (compound merging, number expansion, Cantonese normalization,
retokenization) runs fresh every time.

`--require-media-cache` fails before a raw Rev or speaker miss can become an
inference authorization. A derived-speaker miss may still be rebuilt from a
validated raw hit. FA requires every unresolved group to be reusable or cached;
its worker batch accepts a typed authorization that required-cache misses
cannot construct. Rev-backed UTR may rebuild normalized UTR evidence from a
raw Rev hit; the raw resolver still refuses a provider call on a miss.

An `align` run resolves FA and UTR independently. `FaParams` carries only the
`forced_alignment` policy, while `FaDispatchPlan::utr_cache_policy` carries the
`utr_asr` policy into both the initial UTR pass and retry fallback. This split
is load-bearing: a selective UTR refresh must not refresh FA, and a selective
FA refresh must not change UTR evidence reuse.

## Decision Matrix

| What I changed | Override needed? | Why |
|---------------|-----------------|-----|
| Post-processing logic (injection, bullet computation, %wor generation, retokenization after cache, terminator patching) | **No** | Runs after cache retrieval, cached value is still correct |
| Cache key computation | **No** | Old entries become orphans (different key = automatic miss). New keys miss and re-infer. |
| Word extraction logic (changes which words are sent to the model) | **Yes** | Cached result was computed from different input words |
| ML model/engine code (Python worker) | **Automatic** for FA if the reported FA engine identity changes; **Yes** if the identity string is unchanged, and for a UTR model change within one UTR engine | FA rows are namespaced by the reported FA engine (`FaCacheNamespace`); UTR ASR rows by the UTR engine, which does not yet distinguish model revisions |
| Serialization format of legacy FA/UTR derived values | **Usually no** | Normal mode may treat an unreadable derived value as work to recompute; required mode refuses the unresolved group. |
| Serialization format of raw Rev/speaker evidence | **No automatic refresh** | Corruption fails closed so local damage cannot authorize a paid call. Change the schema/revision deliberately. |
| Parse logic (changes how CHAT is parsed before extraction) | **Depends** | If extraction produces different words → yes (different key). If same words → no. |
| Pre-cache text normalization (e.g., the script rendering in `TranslationSource::render`) | **Yes** | Key is computed from normalized text; same key now maps to wrong cached result |

## Worked Example: The Bullet-Shrinking Bug

**Bug (2026-03-16, a user, ACWT corpus):** `update_utterance_bullet()` computed
the FA timing span from only the aligned words, then *replaced* the original
utterance bullet with it. Fillers, pauses, and gestures (which FA cannot align)
lost their timing coverage.

**Analysis:**

1. What's cached? `Vec<Option<WordTiming>>`: the raw per-word timings from
   Whisper/Wave2Vec.
2. Where's the bug? In `update_utterance_bullet()`: post-processing that runs
   *after* cache retrieval.
3. Are the cached timings wrong? No, the word-level timings are correct. The
   bug was in how we used them to update the utterance bullet.

**Fix:** `update_utterance_bullet()` now uses `BulletSource` provenance to decide
whether to overwrite or union:

- **`BulletSource::Authoritative`** (hand-linked, parsed from file, or FA-derived):
  **union**: never shrink. Preserves filler/gesture coverage.
- **`BulletSource::Utr`** (provisional UTR hint, set by `Bullet::utr_hint()`):
  **overwrite**: FA word span is authoritative. The UTR window was a rough
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
- **Does NOT help when:** The cached value is *wrong but valid*, e.g., it
  passes validation but contains incorrect timings. Validation can't catch
  semantic correctness.
- **Does NOT help when:** The post-processing is buggy, the cache entry will be
  deleted, but re-inference produces the same cached value, which the same buggy
  post-processing corrupts again. Fix the post-processing first.

## Deserialization Failure Policy

Raw Rev and speaker evidence fails closed on cache read, envelope, provenance,
or validation errors. It does not become a miss, because a miss is the value
that can authorize a provider call. FA and the older normalized UTR cache may
recompute unreadable derived entries in ordinary use, but
`--require-media-cache` refuses any unresolved inference group. A deliberate
`--override-media-cache` is the explicit operator decision to replace evidence.

## Deployment Checklist

When deploying a fix to the fleet (production server, worker hosts, etc.):

1. **Identify the change category** using the decision matrix above.
2. **If override is NOT needed:** Deploy the new binary. Cached results are
   reprocessed through the fixed post-processing automatically.
3. **If override IS needed:** Deploy the new binary, then re-run affected
   commands with `--override-media-cache` on the target corpora. For large corpora,
   consider running only on affected files rather than the full dataset.
4. **If engine version changed:** No action needed, version scoping
   automatically invalidates stale entries. Verify by checking cache stats in
   server logs (should show misses on first run).
5. **If unsure:** inspect the raw/derived boundary first. Use
   `--require-media-cache` to prove an experiment can replay without inference;
   use `--override-media-cache` only when fresh inference is the intended and
   budgeted experimental variable.
