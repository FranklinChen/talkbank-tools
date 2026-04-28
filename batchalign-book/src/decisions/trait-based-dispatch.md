# Trait-Based Dispatch

**Status:** Current decision
**Last updated:** 2026-03-17

## Decision

Use traits for pluggable algorithm implementations where we have or plan
multiple Rust-side implementations of the same operation and need controlled
experiments.  Do not use traits for language-specific dispatch or engine
selection where simpler mechanisms already work.

## Context

Batchalign3 has several dispatch dimensions:

1. **Algorithm strategies** — different implementations of the same
   operation (e.g., global vs. per-speaker UTR alignment).
2. **Engine selection** — choosing which ML backend to use (Whisper vs.
   Rev.AI for ASR, Wave2Vec vs. Whisper for FA, Pyannote vs. NeMo for
   diarization).
3. **Language-specific processing** — per-language rules for morphosyntax,
   number expansion, text normalization.
4. **Command-level pipelines** — the top-level structure of each command
   (transcribe, align, morphotag, etc.).

Each dimension has different characteristics that call for different
dispatch mechanisms.

## Where Traits Are the Right Choice

### Algorithm strategies with multiple Rust implementations

The motivating case is UTR alignment, where we plan three implementations:

| Strategy | Description | Status |
|----------|-------------|--------|
| Global UTR | Current single-stream monotonic DP | Implemented |
| Backbone UTR | Strip `&*` segments before DP | Proposed |
| Per-speaker UTR | Diarize, then per-speaker DP | Proposed |

These share the same inputs (CHAT file + audio + ASR tokens), produce the
same outputs (timing injected into the CHAT file), and need to be compared
head-to-head on the same corpus.  A trait makes this clean:

```rust
/// Strategy for recovering utterance timing from ASR output.
///
/// Implementations own the full UTR pass: reference extraction,
/// ASR token acquisition (possibly per-speaker), DP alignment,
/// and timing injection.  The orchestrator calls `run()` and
/// gets back a coverage result.
pub trait UtrStrategy: Send + Sync {
    /// Run the full UTR pass, modifying `chat_file` in place.
    fn run(
        &self,
        ctx: &UtrPassContext<'_>,
        chat_file: &mut ChatFile,
        audio_path: &Path,
    ) -> impl Future<Output = Result<UtrResult>> + Send;
}
```

The trait surface is deliberately wide — `run()` owns the entire UTR pass,
not just the DP step — because per-speaker UTR needs to control ASR calls
(per-speaker segments rather than one global call).  A narrower trait that
only covered reference extraction and DP injection would force per-speaker
UTR to work around the trait boundary.

**Selection** is now via a visible CLI flag for the validated overlap-aware UTR
surface:

```rust
/// UTR strategy selection.
///
/// Visible from --help. Default is auto.
#[arg(long, value_enum, default_value_t)]
pub utr_strategy: UtrOverlapStrategy,

#[derive(Clone, Copy, ValueEnum)]
pub enum UtrOverlapStrategy {
    Auto,
    Global,
    TwoPass,
}
```

The earlier broader experiment sketch (`Global` / `Backbone` / `PerSpeaker`)
did not ship as the public surface. The validated trait boundary stayed smaller:
the current strategies are `GlobalUtr` and `TwoPassOverlapUtr`.

**Why a trait, not just an enum match:**

- Each strategy is 100-500 lines with its own internal state (per-speaker
  UTR holds diarization results, speaker mappings, per-speaker caches).
  Putting all of that behind a match arm in a single function would produce
  a 1000+ line function.
- The trait enforces that all strategies have the same contract: same inputs,
  same output type, same error handling.  A match arm doesn't enforce this —
  it's easy for one branch to silently return a different result shape.
- Strategies are independently unit-testable.  Each impl gets its own test
  module without coupling to the others.
- Future strategies (non-monotonic local alignment, etc.) can be added
  without modifying existing code.

### Other candidates

If we later build multiple Rust-side implementations for other operations,
the same pattern applies.  Plausible future candidates:

- **FA grouping strategies** — different ways to partition utterances into
  FA windows (current fixed-window, adaptive window, trouble-window).
- **Monotonicity enforcement strategies** — strip timing (current), reorder
  utterances, accept non-monotonic (if CLAN ever supports it).

These are speculative.  Do not pre-build trait abstractions for them.

## Where Traits Are Not the Right Choice

### Engine selection (ASR, FA, Speaker)

The engine enums (`AsrBackendV2`, `FaBackendV2`, `SpeakerBackendV2`) select
which Python worker to talk to.  The Rust side doesn't contain alternate
implementations — it builds a request, sends it to the worker, and parses
the response.  The variation is in the request format and the Python-side
model, not in Rust logic.

Current mechanism: enum match in the dispatch layer.

```rust
match backend {
    AsrBackendV2::LocalWhisper => build_whisper_request(...),
    AsrBackendV2::Revai => build_revai_request(...),
    AsrBackendV2::HkTencent => build_hk_tencent_request(...),
    ...
}
```

This is the right level of abstraction.  A `trait AsrEngine` would add
indirection (vtable dispatch, boxed futures) for no benefit — the match arms
are 5-10 lines each and the "polymorphism" is just selecting request
parameters.

**When to reconsider:** If we ever bring an ASR or FA engine fully into Rust
(e.g., a Rust CTC decoder), that engine would be a genuine alternate
implementation and a trait would make sense.  Until then, enum match is
simpler.

### Language-specific processing

The current language dispatch uses three mechanisms, all appropriate for
their scale:

**1. Single conditional (ASR post-processing)**

```rust
// asr_postprocess/mod.rs — one branch for Cantonese
if lang == "yue" {
    words = normalize_cantonese_words(words);
}
```

One language has special handling.  A `trait LanguagePostProcessor` with
methods like `normalize_asr_words()` would require a default no-op impl for
every other language, a registry to look up the trait impl by language code,
and a dynamic dispatch call — all to replace a one-line conditional.

**2. Per-language modules (morphosyntax)**

```
nlp/lang_en.rs  — 271 lines, English irregular verbs
nlp/lang_fr.rs  — 262 lines, French-specific rules
nlp/lang_ja.rs  — 460 lines, Japanese verb form patterns
```

Called from two `if lang2(&ctx.lang) == "ja"` checks in `mor_word.rs`.
This is already the right shape — each language's rules are isolated in
their own module, the dispatch point is obvious, and adding a new language
means adding a module and a conditional.  A trait would formalize the
interface but wouldn't reduce code or improve safety.

**3. Table-driven lookup (number expansion)**

```rust
// num2text.rs — 12 languages in a static lookup table
static NUM2LANG: LazyLock<HashMap<&str, LangTable>> = ...;
```

Adding a language is adding a table entry.  This is more flexible than a
trait (data-driven, no code change) and has zero dispatch overhead.

**When to reconsider:** If we reach 6+ languages with distinct
post-processing logic (not just table entries), the conditionals would
become unwieldy and a trait registry would be cleaner.  We currently have
1 (Cantonese) for ASR post-processing and 3 (English, French, Japanese) for
morphosyntax.  We are not close to that threshold.

### Command-level pipelines

Each command (transcribe, align, morphotag, translate, utseg, coref) has a
distinct pipeline structure with different stages, different worker
interactions, and different output shapes.  They share infrastructure
(worker dispatch, caching, progress reporting) but not control flow.

A `trait Command` with a single `run()` method would be a false
abstraction — the implementations would share nothing beyond the method
signature.  The current explicit pipeline functions (`run_transcribe_pipeline`,
`run_fa_pipeline`, `infer_batched`) are clearer because each pipeline's
structure is visible in one place.

**Exception:** If we add a "pipeline combinator" that chains commands
(e.g., transcribe → morphotag → align in one pass), a shared trait for
pipeline stages would help.  This is not currently planned.

## Implementation Guidelines

### For the UTR strategy trait

1. Define the trait in `batchalign-chat-ops/src/fa/utr.rs` (or a new
   `fa/utr/` module if the file gets too large).
2. Move the current global UTR logic into `struct GlobalUtr` implementing
   the trait.
3. Add `UtrStrategyChoice` to CLI args with `hide = true`.
4. Wire strategy selection in `runner/dispatch/utr.rs` — construct the
   appropriate impl and call `strategy.run(ctx, chat_file, audio_path)`.
5. Add backbone and per-speaker implementations as separate files
   (`fa/utr/backbone.rs`, `fa/utr/per_speaker.rs`).

### For future trait candidates

Before introducing a new trait:

1. Confirm there are at least 2 concrete implementations that exist or are
   being built in the same change.  Do not create a trait for a single
   implementation with a vague plan for a second.
2. Confirm the implementations share the same input/output contract.  If
   the "alternate" implementation needs different inputs, it's not the same
   trait — it's a different operation.
3. Prefer the simplest mechanism that works: conditional → enum match →
   module-per-variant → trait.  Escalate only when the simpler mechanism
   becomes unwieldy.

### Hidden experimental flags

Use clap's `hide = true` for experimental strategy flags:

```rust
#[arg(long, hide = true, default_value = "global")]
pub utr_strategy: UtrStrategyChoice,
```

Promotion path:
1. Hidden flag, default = current behavior.  Used only in development and
   corpus experiments.
2. Visible flag, default = current behavior.  Documented in `--help` with a
   note that the alternate strategies are experimental.
3. Change the default if the new strategy proves better on real data.
4. Remove the flag if the old strategy is no longer needed.

Never skip step 3 → 4.  Keep the old strategy available until the new one
has been validated on production data by users, not just by developers.

## Summary

| Dispatch dimension | Mechanism | Why |
|--------------------|-----------|-----|
| Algorithm strategies (UTR, future FA grouping) | Trait | Multiple Rust impls, need controlled comparison, independently testable |
| Engine selection (ASR, FA, Speaker) | Enum match | Variation is in Python worker, not Rust logic; match arms are trivial |
| Language processing (morphosyntax, ASR post) | Conditional + module + table | 1-3 languages with special handling; below threshold for trait overhead |
| Command pipelines | Explicit functions | Pipelines share infrastructure, not control flow; false polymorphism |
