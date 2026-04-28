# batchalign-chat-ops — CHAT Manipulation for Server-Side Orchestration

**Status:** Current
**Last modified:** 2026-04-27 10:28 EDT

## Overview

Shared library implementing the extract→modify→inject round-trip for all NLP tasks.
Used by both the PyO3 bridge (`batchalign-core`) and the standalone server
(`batchalign-app`). This crate owns **all** pre-processing, post-processing, and domain
logic — Python workers are pure model servers that return raw ML output. All text normalization,
DP alignment, WER computation, ASR post-processing, compound merging, number expansion,
retokenization, and result injection live here.

## Module Map

| Module | Purpose |
|--------|---------|
| `parse.rs` | `parse_lenient()`, `parse_strict()`, `is_dummy()` |
| `serialize.rs` | `to_chat_string()` |
| `extract.rs` | `extract_words()` — domain-aware word extraction (Mor/Wor/Pho/Sin) |
| `inject.rs` | `inject_morphosyntax()`, `replace_or_add_tier()` |
| `retokenize/` | Stanza-induced word splits/merges: deterministic mapping with length-aware fallback |
| `morphosyntax.rs` | Payload collection, clear/inject for %mor/%gra |
| `utseg.rs` | Utterance segmentation payloads, apply |
| `translate.rs` | Translation payloads, inject %xtra |
| `coref.rs` | Coreference payloads (document-level), inject %xcoref (sparse) |
| `fa/` | Forced alignment: grouping, extraction, injection, postprocess, DP alignment, UTR timing recovery |
| `dp_align.rs` | Hirschberg divide-and-conquer sequence alignment (linear space) |
| `text_types.rs` | Provenance newtypes: `ChatRawText`, `ChatCleanedText`, `SpeakerCode` |
| `nlp/` | UD types (`UdWord`, `UniversalPos`), UD→CHAT mapping, validation, language-specific rules |
| `asr_postprocess/` | ASR post-processing: compound merging, number expansion, Cantonese normalization, retokenization |
| `wer_conform.rs` | WER word normalization: compound splitting, name replacement, filler/contraction expansion |

## Key Commands

```bash
cargo nextest run -p batchalign-chat-ops
cargo clippy -p batchalign-chat-ops -- -D warnings
```

## Debugging and Per-Stage Inspection

When a token reaches `transcript_from_asr_utterances` in an unexpected
shape (e.g., a stray quote or digit-bearing alphanumeric), the right
move is to inspect the post-processed utterances at the gate, *not*
to read the pipeline source and guess. The patterns:

### `BA3_DUMP_UTTERANCES` env-var

Setting `BA3_DUMP_UTTERANCES=/path/to/file.json` causes
`transcript_from_asr_utterances` (in `build_chat.rs`) to write the full
post-processed `Vec<Utterance>` to that path before validation. Useful
for grep/jq-driven exploration without re-running a multi-minute ASR
job. Token timings are preserved so individual words can be located in
the audio.

```bash
BA3_DUMP_UTTERANCES=/tmp/utts.json batchalign3 transcribe in/ -o out/ --lang eng
jq '.[N].words[I]' /tmp/utts.json   # inspect specific token
jq '[.[].words[] | select(.text == "?")] | length' /tmp/utts.json  # count occurrences
```

### `BA3_DUMP_ASR_PIPELINE` env-var

Setting `BA3_DUMP_ASR_PIPELINE=/path/to/file.json` causes the
transcribe pipeline (in `batchalign-app/src/pipeline/transcribe.rs`)
to populate an `AsrPipelineSnapshot` capturing every stage of ASR
post-processing — raw elements, after compound merge, after timing
extract, after multi-word split, after number expand, after Cantonese
norm, after long-turn split, and final utterances — and write the
JSON shape (`AsrPipelineTrace`) to disk after the post-processing
stage completes.

```bash
BA3_DUMP_ASR_PIPELINE=/tmp/asr-pipeline.json batchalign3 transcribe in/ -o out/ --lang eng
jq '.raw_tokens | length' /tmp/asr-pipeline.json
jq '.after_multiword_split[] | select(.text | contains("\""))' /tmp/asr-pipeline.json
```

This was the gap that originally blinded the validation-gap
investigation: `AsrPipelineTrace` was defined in
`batchalign-app/src/types/traces.rs` but never populated. Now it is —
the snapshot lives on `TranscribePipelineContext` and is converted via
`crate::types::results::snapshot_into_pipeline_trace` when the env-var
is set. Hooking the snapshot into the persistent `trace_store` (so the
dashboard renders the same data) is a follow-up; the env-var dump is
the working diagnostic today.

### Per-stage tracing (temporary instrumentation pattern)

When a token reaches the gate in an unexpected shape, the canonical
move is to add temporary `tracing::warn!` lines inside
`prepare_words_pre_expansion` after each stage, gated on an
`std::env::var("BA3_TRACE_<NAME>").is_ok()` check, that fire only when
the token in question matches a predicate. Re-run with the flag set to
localize which stage introduced or failed to handle it. Remove the
trace lines once the root cause is fixed (don't accumulate dead
diagnostics).

### Trace infrastructure (`batchalign-app::trace_store`)

`batchalign-app/src/types/traces.rs` defines `FileTraces` with four
trace shapes:
- `FaTimelineTrace` — **populated** for `align` jobs (see
  `runner/dispatch/fa_pipeline.rs::into_timeline_trace`). Captures FA
  groups, pre/post-injection timings, fallback events, violations.
- `DpAlignmentTrace`, `AsrPipelineTrace`, `RetokenizationTrace` —
  **defined but not populated** today. Reserved for future
  instrumentation; do not assume the dashboard receives data through
  them.

For ASR pipeline / retokenize / DP debugging, the env-var dumps above
are the current path. Do not waste time reading the source to guess at
intermediate state — instrument and capture instead.

### Tracing levels and `RUST_LOG`

Library code uses `tracing` macros. To get verbose pipeline output
without code changes:

```bash
RUST_LOG=batchalign_chat_ops=debug,batchalign_app=debug \
  batchalign3 transcribe in/ -o out/ --lang eng
```

Per-stage `tracing::info!`/`warn!` already log key events
(`prepare_words_pre_expansion` doesn't yet — temporary instrumentation
is acceptable when investigating a specific failure mode, then
removed once the root cause is fixed).

## NLP Task Modules

Each task module exports: **batch item type**, **response type**,
**collect_payloads()**, and **apply/inject results**. FA additionally
exports a `cache_key()`; UTR exports its own helpers (see "UTR" below).
Text-NLP tasks (morphotag, utseg, translate, coref) deliberately do
not cache — see `batchalign3/CLAUDE.md` "Utterance Cache".

| Task | Granularity | Caching |
|------|-------------|---------|
| Morphosyntax | Per-utterance | None — always recomputed |
| Utseg | Per-utterance | None — always recomputed |
| Translate | Per-utterance | None — always recomputed |
| Coref | Per-document | None — always recomputed (full-document context) |
| FA | Per-group (time-windowed) | BLAKE3 over `(audio_identity, start, end, text, pauses, engine)`; cache call sites in `batchalign-app/src/fa/{incremental,mod}.rs` |
| UTR ASR | Per-file (full) or per-window (partial) | BLAKE3 over `(audio_identity, lang)` or `(audio_identity, lang, start_ms, end_ms)`; cache call sites in `batchalign-app/src/runner/dispatch/utr.rs` |

### UTR (Utterance Timing Recovery)

`fa/utr.rs` — Pre-pass that injects utterance-level timing from ASR tokens
into untimed CHAT utterances. Uses a single global Hirschberg DP alignment
(`dp_align::align(..., CaseInsensitive)`) of ALL document words against ALL
ASR tokens. Timed utterances participate to anchor the alignment but their
bullets are left unchanged. The global approach avoids token starvation that
per-utterance windowed alignment suffered from, but it is still a monotonic
aligner, so dense overlap / text-audio reordering remains a known limitation.

**UTR bullets are provisional hints.** UTR injects bullets with
`Bullet::utr_hint(start_ms, end_ms)` (`BulletSource::Utr`). After FA
injects word timings, `update_utterance_bullet()` **overwrites** UTR hints
with the FA word span. This is the self-healing property: valid FA word
timings → valid utterance bullet by construction, regardless of the UTR
estimate. See `talkbank-model::model::BulletSource` for the full semantics.

**FA word timing clamping policy.** `postprocess_utterance_timings()` in
`fa/postprocess.rs` clamps FA word timings to the utterance bullet range ONLY
when BOTH conditions hold:

1. The bullet is `BulletSource::Authoritative` (not a runtime UTR hint).
2. The utterance already has a `%wor` tier (i.e., this is a RE-alignment, not
   a first-time alignment).

The `%wor` discriminator is critical: after `transcribe` + `utseg`, utterance
bullets come from ASR token timestamps via UTR, are serialized to CHAT text,
and are re-parsed as `BulletSource::Authoritative` (BulletSource is not stored
in CHAT). These ASR-derived bullets may be far narrower than the actual speech
(e.g., Rev.AI stamps 220ms for a 3-second utterance when it only matches the
first word). On first-time alignment (no `%wor`), clamping to these bullets
drops valid FA word timings. The `%wor` tier is only present after a previous
FA run, so its presence reliably identifies whether the current utterance bullet
was established by FA (trustworthy) or by ASR/UTR (potentially too narrow).

UTR-hinted bullets (`BulletSource::Utr`, set at runtime) are also skipped
regardless of `%wor` presence, for the same reason.

Key types: `AsrTimingToken` (text + start_ms + end_ms), `UtrResult`
(injected/skipped/unmatched counts).

Entry point: `inject_utr_timing(&mut ChatFile, &[AsrTimingToken]) -> UtrResult`.

Detection helper: `count_utterance_timing(&ChatFile) -> (timed, untimed)` in
`fa/grouping.rs`.

Cache key helpers: `utr_asr_cache_key()` (full-file), `utr_asr_segment_cache_key()`
(partial-window). Both produce BLAKE3-keyed `CacheKey` values.

Window finder: `find_untimed_windows(&ChatFile, total_audio_ms, padding_ms) -> Vec<(u64, u64)>`
identifies audio windows covering contiguous untimed utterances for partial-window ASR.

### Compound Filler Splitting

CHAT uses underscores to join multi-word fillers into single tokens:
`&-you_know`, `&-sort_of`, `&-I_mean`. ASR engines hear these as separate
words ("you", "know") and return separate timings.

**Extraction** (`fa/extraction.rs:push_fa_word`): splits compound fillers at
underscores before sending to FA. `&-you_know` → `["you", "know"]`.

**Injection** (`fa/injection.rs:inject_timing_on_word`): consumes N timings
from the cursor (where N = underscore-split count) and merges them into one
span for the single CHAT token.

Only `WordCategory::Filler` words are split. Regular compound words
(`ice_cream`) are NOT split — ASR may tokenize them differently.

See `fa/COMPOUND_FILLER_ALIGNMENT.md` for the full design document.

## Dependencies

Path deps to `talkbank-tools` crates (talkbank-model, talkbank-direct-parser,
talkbank-parser).

Re-exports `ChatFile` and `LanguageCode` for downstream convenience.

## Design Principles

- **No string hacking** — all CHAT operations through AST manipulation
- **Provenance types** — `ChatRawText` vs `ChatCleanedText` (CHAT direction) and `AsrRawText` → `AsrNormalizedText` → `ChatWordText` (ASR direction) prevent mixing text at different pipeline stages. `ChatWordText` additionally enforces its postcondition at construction: it is constructible **only** via fallible `try_from[_lang][_with_parser]` which delegate to `TreeSitterParser::parse_word_fragment` plus the typed `Terminator::is_chat_terminator` short-circuit, plus (for the language-aware variants) `Word::validate` under a `ValidationContext` carrying the declared language. See `asr_postprocess/asr_types.rs` module docs
- **Domain-aware extraction** — `TierDomain` selects which word properties to extract
- **Alignment validation** — %mor word count must match main tier word count before injection
- **Content walker** — `walk_words()` / `walk_words_mut()` from `talkbank-model` centralizes
  UtteranceContent/BracketedItem traversal. Callers provide only leaf-handling closures.
  Used by `extract.rs`, `fa/extraction.rs`, `fa/injection.rs`, `fa/postprocess.rs`.

## ASR Post-Processing (`asr_postprocess/`)

Ported from Python `batchalign/pipelines/asr/utils.py`. Transforms raw ASR tokens
into utterances ready for CHAT assembly. Sub-modules:

| File | Purpose |
|------|---------|
| `asr_types.rs` | Provenance newtypes: `AsrRawText`, `AsrNormalizedText`, `ChatWordText`, `AsrTimestampSecs`, `SpeakerIndex` |
| `mod.rs` | Pipeline orchestrator: `prepare_words_pre_expansion()` (stages 1-3), `finalize_words_to_chunks()` (stages 4b-5b), `process_raw_asr()` (monolithic sync fallback), retokenization |
| `compounds.rs` | 3,584 compound word pairs, `merge_compounds()` with O(1) HashSet lookup |
| `num2text.rs` | Number detection and expansion: `detect_expansion()`, sync `expand_number()` covering CJK, currency, percent, dash-ranges, NUM2LANG cardinal lookup, and English ordinal/decade routing into `ordinal_year_eng`. Single per-word Rust pass; no Python IPC after the 2026-04-26 Round-2 rework. Architecture, per-language coverage matrix, and rework plan at `book/src/architecture/number-expansion.md` (single source of truth — keep updated in lock-step with code changes). |
| `ordinal_year_eng.rs` | Deterministic English ordinal / year / decade composition (`expand_ordinal_eng`, `expand_year_eng`, `expand_decade_eng`); cross-validated against Python `num2words` via `data/eng_ordinal_year_fixtures.json` at build time. |
| `registry.rs` | Per-language `NumberExpander` registry (`NUMBER_EXPANDERS` static). Exposes `expander_for(lang)`. |
| `num2chinese.rs` | Chinese/Japanese number converter (simplified + traditional, up to 10^48) |
| `cantonese.rs` | Cantonese text normalization: `zhconv` zh-HK + 31-entry Aho-Corasick replacement table |

### Pipeline Stages

The pipeline is split for Python IPC integration.  `prepare_words_pre_expansion()`
runs stages 1-3, then `detect_expansion()` collects expandable numbers for Python
`num2words` (56 languages, cardinals + ordinals + decades), and
`finalize_words_to_chunks()` runs stages 4b-5b.

```
1. Compound merging                     ─┐
2. Timed word extraction (seconds → ms)  ├─ prepare_words_pre_expansion()
3. Multi-word splitting                 ─┘
4. Number expansion:
   - Cardinals/ordinals/decades → Python num2words (56 langs)
   - CJK → Rust num2chinese
   - Currency → Rust try_expand_currency
4b. Cantonese normalization             ─┐
5. Long turn splitting (>300 words)      ├─ finalize_words_to_chunks()
5b. Pause-based splitting               ─┘
6. Retokenization (split by punctuation)
7. Disfluency replacement ("um" → "&-um")
8. N-gram retrace detection
```

### Cantonese Normalization

Migrated from Python `batchalign/inference/hk/_common.py` to Rust. Uses:
- **`zhconv`** crate (pure Rust, 100-200 MB/s) — Aho-Corasick automata compiled from OpenCC + MediaWiki rulesets for `Variant::ZhHK` conversion
- **Domain replacement table** — 31 entries (13 multi-char + 18 single-char) for Cantonese-specific character corrections, applied via a second Aho-Corasick pass with leftmost-longest matching

Exposed to Python via `batchalign_core.normalize_cantonese()` and `batchalign_core.cantonese_char_tokens()`. Python `_common.py` delegates to these Rust functions — **no OpenCC Python dependency needed**.

Data files in `data/`: `compounds.json` (3,660 pairs, 76 duplicates), `num2lang.json` (12 languages),
`names.json` (~6,700 proper names), `abbrev.json` (~400 abbreviations).

### ReplacedWord Extraction/Injection Policy

For FA (forced alignment), a `ReplacedWord` like `foo [: bar baz]` contributes
**exactly one word to FA**: the original spoken word (`foo`), not the
replacement words (`bar`, `baz`). This policy is shared across four sites that
**must stay in sync**:

| Site | File | What it does |
|------|------|-------------|
| Extraction | `fa/extraction.rs::collect_fa_words` | Sends `foo` to FA worker |
| Injection | `fa/injection.rs::inject_timings_for_utterance` | Consumes 1 cursor slot, sets `foo.inline_bullet` |
| Count | `fa/mod.rs::count_alignable_main_words` | Counts 1 for `foo` |
| Preservation | `fa/mod.rs::collect_existing_fa_word_timings` | Reads `foo.inline_bullet` |

If any site uses replacement words instead of the original, extraction and
injection go out of sync. For a 2-word replacement, injection would consume 2
cursor slots while extraction only sent 1 word — shifting every subsequent word
in the same FA group by +1, corrupting all downstream timings.

The `retokenize/` and `extract.rs` (Mor domain) modules intentionally use
replacement words — that is correct for morphosyntax, which analyses the
correction, not the error.

### `%wor` Exclusion Policy

Three token categories are excluded from both FA extraction and `%wor`
generation. This is enforced by `TierDomain::Wor` in
`talkbank-model/src/alignment/helpers/rules.rs`:

```rust
TierDomain::Wor => {
    !is_wor_timing_token(word)
        && word.untranscribed().is_none()
        && !is_wor_excluded_category(word)  // Nonword | PhonologicalFragment
}
```

| Token type | Excluded? | Reason |
|---|---|---|
| `xxx`, `yyy`, `www` | **Yes** | No phoneme sequence; CTC cannot align unknown material |
| `&+word` (fragment) | **Yes** | Matches BA2 `TokenType.ANNOT`; incomplete phoneme sequences |
| `&~word` (nonword) | **Yes** | Matches BA2 `TokenType.ANNOT`; interactional sounds, no stable phoneme content |
| `&-word` (filler) | No | Matches BA2 `TokenType.FP`; real spoken words with alignable sequences |
| Regular words | No | Always included |

`%pho` still counts all three excluded types (a vocalization event occurred).
`%mor` already excluded them (no linguistic content).

There is **no positional indexing** into `%wor` from any CLAN tool or from
batchalign3 itself. `%wor` word count is no longer validated against the main
tier count — old corpus files may have `xxx`, fragments, or nonwords in `%wor`
(pre-2026-04 behavior) without triggering false validation errors.

---
Last Updated: 2026-04-12 06:57 EDT
