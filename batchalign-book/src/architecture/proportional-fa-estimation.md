# Proportional FA Estimation for Untimed Utterances

**Date:** 2026-02-11
**Status:** Implemented
**Last updated:** 2026-03-21 15:30

---

This design note describes functionality that is now implemented in current
Rust/Python code paths.

## Problem

The Rust FA orchestrator (`batchalign_core.add_forced_alignment`) groups
utterances by their existing timing bullets to determine which audio segment
to pass to the FA model. Utterances without timing bullets are silently
skipped (`forced_alignment.rs:65-71`).

Currently, the `align` pipeline works around this by auto-injecting UTR
(Utterance Timing Recovery) before FA. UTR runs a full Whisper ASR pass
(5-7 min for 30-min audio) solely to produce rough timing that FA immediately
overwrites. This is wasteful.

## Solution: Proportional Estimation

When the caller provides `total_audio_ms` (the total audio file duration),
the Rust grouping algorithm estimates boundaries for untimed utterances
proportionally by word count:

```
For each untimed utterance:
    estimated_start = (words_before / total_words) * total_audio_ms
    estimated_end   = (words_before + this_utt_words) / total_words * total_audio_ms
```

This gives the FA model a reasonable audio window to search within. The FA
model (Whisper/Wave2Vec) conditions on the transcript text and finds where
it occurs in the audio, so the window only needs to be approximately correct.

A 2-second buffer is added on each side of the estimated window to account
for estimation error, capped at `[0, total_audio_ms]`.

### Why Proportional?

- **No new dependencies** — pure arithmetic, runs in microseconds
- **Deterministic** — same input always produces same windows
- **Good enough for FA** — FA only needs an approximate window; it does
  precise alignment within the window using the actual audio signal
- **Graceful degradation** — if the estimate is off, FA may produce slightly
  less accurate timing, but won't crash or skip utterances

### Mixed files

Files that have some timed and some untimed utterances are handled naturally.
Timed utterances use their real bullets. Untimed utterances use proportional
estimates. Both are grouped normally.

## Changes

### Rust: `forced_alignment.rs`

Add `total_audio_ms: Option<u64>` parameter to `group_utterances()`.

**Two-pass approach:**
1. First pass: count total alignable words across ALL utterances (timed and
   untimed). If `total_audio_ms` is `None`, skip untimed utterances as before.
2. Second pass (existing loop): for untimed utterances when `total_audio_ms`
   is `Some`, compute proportional estimate and use it as the bullet.

The 2-second buffer is applied as:
```rust
let buffer_ms = 2000;
let start = estimated_start.saturating_sub(buffer_ms);
let end = (estimated_end + buffer_ms).min(total_audio_ms);
```

### Rust: `lib.rs`

Add `total_audio_ms: Option<u64> = None` parameter to the
`add_forced_alignment` PyO3 function. Pass it through to
`group_utterances()`.

Also update the post-processing loop to handle utterances that were untimed
on input but received timing from FA (they need `postprocess_utterance_timings`
and `add_wor_tier` too).

### Python: `inference/fa.py`

Compute audio duration from the loaded `ASRAudioFile`:
```python
duration_ms = int(round(f.tensor.shape[0] / f.rate * 1000))
```

Pass it to `batchalign_core.add_forced_alignment(..., total_audio_ms=duration_ms)`.

For audio files, use `torchaudio.info()` to get duration without loading
the full file.

### Relationship to UTR

With UTR wired in the pipeline (the default since 2026-03-12), the coverage
tiers for untimed utterances are:

1. **UTR enabled (default):** `inject_utr_timing()` sets utterance bullets
   from ASR tokens before FA grouping. All utterances get timing from ASR →
   ~100% coverage.
2. **UTR disabled (`--no-utr`):** Proportional estimation kicks in during
   `group_utterances()` when `total_audio_ms` is available. ~96% coverage.
3. **Neither:** Untimed utterances are skipped from FA grouping.

Proportional estimation is the **fallback** when UTR is disabled or when UTR's
ASR inference fails. It is also the fallback for individual utterances that UTR
could not match (the `unmatched` count in `UtrResult`).

**Decision: always pass `total_audio_ms` when available.** It's a no-op for
pre-timed files (Rust never hits the estimation path). For files where UTR
partially succeeds, it catches the remaining untimed utterances.

## Implementation status

The documented changes are now present in current code:

- `ParsedChat.add_forced_alignment(..., total_audio_ms=...)` is exposed in
  `pyo3/src/parsed_chat/fa.rs`.
- The Rust FA path threads `total_audio_ms` through
  `pyo3/src/fa_ops.rs`.
- Grouping with proportional estimation is implemented in
  `crates/batchalign-chat-ops/src/fa/grouping.rs`.
- The Rust server computes audio duration and passes it into FA orchestration in
  `crates/batchalign-app/src/runner/dispatch/fa_pipeline.rs`.
- Tests cover untimed utterances with and without `total_audio_ms` in
  Rust unit tests (`crates/batchalign-chat-ops/src/fa/grouping.rs`).

## Testing

### Rust unit tests (`forced_alignment.rs`)
- Untimed utterances grouped with proportional estimates when `total_audio_ms` is provided
- Untimed utterances still skipped when `total_audio_ms` is `None` (backwards compat)
- Mixed timed/untimed utterances grouped correctly
- Buffer clamped to `[0, total_audio_ms]`

### Python integration tests
- `add_forced_alignment` with untimed CHAT + `total_audio_ms` produces timing
- `add_forced_alignment` with untimed CHAT + no `total_audio_ms` produces no timing (backwards compat)
