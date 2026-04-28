# Per-Speaker UTR Design

**Status:** Draft
**Last updated:** 2026-03-16

Per-speaker UTR is the theoretically correct solution for alignment of
transcripts where text order diverges substantially from audio temporal
order — the class of failure documented in the 2265_T4 post-mortem (36.5%
timing loss).

**Important caveat:** The impact estimates in this document are unvalidated
architectural reasoning.  The simulation experiment described in
[Overlap-Aware Alignment Improvements](backchannel-aware-alignment.md#empirical-validation-before-building)
must be run before committing to this implementation.  In particular, we do
not know whether within-speaker word order is preserved after heavy
transcript restructuring — if it isn't, per-speaker UTR won't help either.

This page is the concrete implementation plan, contingent on validation.

## Core Idea

Today UTR flattens all speakers' words into one interleaved reference
sequence and aligns it against one interleaved ASR word sequence.  When the
interleaving order differs between transcript and ASR (inevitable for
restructured transcripts), the monotonic DP cannot represent the crossing.

Per-speaker UTR eliminates the interleaving:

1. Run diarization to get speaker time boundaries.
2. Extract audio per speaker.
3. Run ASR per speaker.
4. Align each speaker's transcript utterances against only that speaker's
   ASR stream.
5. Merge per-speaker timing back into the single CHAT file.

Each per-speaker alignment is monotonic within that speaker's stream, which
is the correct constraint — one speaker's words are in temporal order.

## Existing Infrastructure

Everything needed already exists in the codebase.  No new ML models, no new
worker protocol operations, no new external dependencies.

| Capability | Where | How we reuse it |
|-----------|-------|-----------------|
| Speaker diarization | `inference/speaker.py` → `SpeakerSegment(start_ms, end_ms, speaker)` | Run diarization as a pre-pass, same as `transcribe` does |
| Audio segment extraction | `worker/artifacts_v2.rs` → `extract_prepared_audio_segment_f32le()` | Extract per-speaker audio windows via ffmpeg |
| Per-segment ASR | `runner/dispatch/utr.rs` → `run_utr_pass()` partial-window path | Same Whisper ASR call, just on speaker-specific segments |
| Hirschberg DP | `fa/utr.rs` → `inject_utr_timing()` | Run N times (once per speaker) instead of once |
| Speaker reassignment logic | `speaker.rs` → `reassign_speakers()` | Reuse speaker-to-CHAT-code mapping |
| Cache keys | `cache_key.rs` → `CacheKey::from_content()` | Add speaker ID to key |

## Data Flow

```
                         ┌─────────────────────────┐
                         │  CHAT file + audio       │
                         └────────────┬────────────┘
                                      │
                              ┌───────▼───────┐
                              │  Diarization   │
                              │  (speaker.py)  │
                              └───────┬───────┘
                                      │
                         Vec<SpeakerSegment>
                         (speaker labels + time spans)
                                      │
                    ┌─────────────────┼─────────────────┐
                    │                 │                  │
             ┌──────▼──────┐  ┌──────▼──────┐   ┌──────▼──────┐
             │ Speaker A    │  │ Speaker B    │   │ Speaker C    │
             │ segments     │  │ segments     │   │ segments     │
             └──────┬──────┘  └──────┬──────┘   └──────┬──────┘
                    │                │                  │
             extract audio    extract audio       extract audio
             (ffmpeg)         (ffmpeg)             (ffmpeg)
                    │                │                  │
             Whisper ASR      Whisper ASR         Whisper ASR
                    │                │                  │
             ASR tokens A     ASR tokens B        ASR tokens C
                    │                │                  │
             ┌──────▼──────┐  ┌──────▼──────┐   ┌──────▼──────┐
             │ Filter CHAT  │  │ Filter CHAT  │   │ Filter CHAT │
             │ utterances   │  │ utterances   │   │ utterances  │
             │ for speaker A│  │ for speaker B│   │ for speaker │
             └──────┬──────┘  └──────┬──────┘   └──────┬──────┘
                    │                │                  │
             Hirschberg DP    Hirschberg DP       Hirschberg DP
             (monotonic)      (monotonic)          (monotonic)
                    │                │                  │
                    └─────────────────┼─────────────────┘
                                      │
                              ┌───────▼───────┐
                              │  Merge timing  │
                              │  into CHAT     │
                              └───────┬───────┘
                                      │
                              ┌───────▼───────┐
                              │  Monotonicity  │
                              │  enforcement   │
                              └───────┬───────┘
                                      │
                              ┌───────▼───────┐
                              │  FA (as usual) │
                              └───────────────┘
```

## Detailed Design

### Step 1: Diarization pre-pass

Run speaker diarization on the full audio file.  This is the same call that
`transcribe` uses (`build_speaker_request_v2` → `ExecuteRequestV2` with
`InferenceTaskV2::Speaker`).

Output: `Vec<SpeakerSegment>` with `start_ms`, `end_ms`, `speaker` fields.

**Caching:** Diarization results are cached by audio identity + backend +
speaker count, same as in `transcribe`.  Repeat runs on the same audio are
instant.

**Speaker mapping:** The diarization model produces abstract labels
(`SPEAKER_0`, `SPEAKER_1`, ...).  We need to map these to the CHAT file's
participant codes (`INV`, `PAR`, `REL1`, ...).  Two approaches:

- **Heuristic mapping** (like `reassign_speakers` does for `transcribe`):
  Assign diarization speakers to CHAT speakers by temporal overlap — the
  diarization speaker whose segments overlap most with a CHAT speaker's
  timed utterances gets that CHAT speaker code.  For `align`, the input
  file may already have some timed utterances (from a previous run or
  from CLAN) that can anchor the mapping.

- **Fallback for untimed input:** If the file has no timed utterances at
  all (raw transcript), the mapping must be inferred from word content.
  Run a quick first-pass global DP (the current UTR) to get coarse timing,
  then use those coarse timestamps to anchor the speaker mapping.  This is
  acceptable because the first-pass DP only needs to be approximately right
  — it's used for speaker assignment, not for final timing.

### Step 2: Per-speaker audio extraction

For each diarization speaker, collect their segments and extract audio.

Two strategies:

- **Concatenated extraction:** Extract each speaker's segments and
  concatenate them into one continuous audio stream per speaker.  Simpler
  for ASR (one call per speaker), but requires tracking segment boundaries
  to map ASR timestamps back to absolute file time.

- **Per-segment extraction:** Extract each segment individually, run ASR
  per segment, and offset timestamps by segment start.  More ASR calls,
  but each is short and independently cacheable.  This is what
  partial-window UTR already does.

**Recommendation:** Per-segment extraction.  It reuses the existing
partial-window infrastructure (`extract_prepared_audio_segment_f32le`),
each result is independently cacheable, and the timestamp offset logic
already exists in `run_utr_pass`.

Merging adjacent segments for the same speaker (gap < 500ms) reduces the
number of ASR calls without losing boundary information.

### Step 3: Per-speaker ASR

For each speaker's audio segments, run Whisper ASR.  This is the same
`InferenceTaskV2::Asr` call used by the current UTR, just on smaller audio
segments.

Output per speaker: `Vec<AsrTimingToken>` with absolute timestamps (offset
by segment start).

**Cache keys:**

```
per_speaker_utr_asr|{audio_identity}|{speaker_label}|{segment_start_ms}|{segment_end_ms}|{lang}
```

This extends the existing `utr_asr_segment` key pattern with a speaker
label.  Each speaker-segment result is cached independently.

### Step 4: Per-speaker DP alignment

For each CHAT speaker, filter the transcript's utterances to only those
attributed to that speaker.  Build a reference word sequence from those
utterances only.  Run `inject_utr_timing` against that speaker's ASR tokens.

This is the critical step: each per-speaker DP alignment sees only one
speaker's words in both the reference and the ASR.  No interleaving, no
crossing matches.

**`&*` segments:** Utterances attributed to speaker A may contain `&*B:words`
embedded segments.  These belong to speaker B's audio, not speaker A's.
The backbone extraction from Approach 1 strips these before building the
per-speaker reference.  The two approaches compose naturally: backbone
extraction cleans the reference, per-speaker UTR ensures each reference is
matched against the right audio stream.

**Untimed utterances without speaker attribution:** Rare in practice (CHAT
files almost always have speaker codes on main tier lines), but if it
occurs, these utterances fall through to the existing global DP path.

### Step 5: Merge

After all per-speaker alignments complete, each untimed utterance has either
received a timestamp from its speaker's alignment or remained untimed
(because the per-speaker DP couldn't match it).

Merge is straightforward: for each utterance in the CHAT file, if its
speaker's per-speaker alignment produced timing for it, apply that timing.
The existing `inject_utr_timing` already modifies utterances in place by
index — the per-speaker version just does so for a subset of indices.

### Step 6: Monotonicity enforcement

After merge, run the existing monotonicity enforcement pass.  Per-speaker
alignment produces per-speaker monotonic timing, but the merged result may
still have cross-speaker monotonicity violations (speaker A's utterance at
text position 5 has a later timestamp than speaker B's utterance at text
position 6, because they overlap in the audio).

The enforcement pass strips timing from violations, same as today.  The
difference is that far fewer utterances should violate: the per-speaker
timing is correct within each speaker's stream, so violations only arise
at speaker-transition boundaries where text order and audio order differ.

### Step 7: FA continues as usual

The FA pipeline sees the UTR-timed file and runs word-level alignment.
No changes to FA are needed.

## When to Use Per-Speaker UTR

Per-speaker UTR is more expensive than global UTR: it requires a diarization
call plus N speaker x M segment ASR calls instead of 1 (or a few) ASR calls.
It should not run on every file.

**Heuristic activation:**

```
if file has >2 speakers AND >20% of utterances contain &* markers:
    use per-speaker UTR
elif file has >10% untimed utterances after global UTR:
    retry with per-speaker UTR
else:
    use global UTR (current behavior)
```

The retry-after-global-UTR path is the safest: run the cheap global path
first, measure coverage, and only escalate to per-speaker if coverage is
poor.  This means most files (which work fine with global UTR) pay zero
extra cost.

**Explicit flag:** `--per-speaker-utr` forces per-speaker mode regardless
of heuristics.  Useful for testing and for users who know their files have
dense overlap.

## Cost Analysis

For a 25-minute 4-speaker APROCSA recording:

| Step | Global UTR (today) | Per-speaker UTR |
|------|--------------------|-----------------|
| Diarization | — | ~10-15s (one-time, cached) |
| ASR calls | 1 full-file (~30s) | ~4 speakers x ~5 segments each = ~20 calls (~2-5s each, parallelizable) |
| Total ASR wall time | ~30s | ~10-15s (parallel) to ~60s (serial) |
| DP alignment | 1 global (~<1s) | 4 per-speaker (~<1s each) |
| Cache benefit | Full-file cached | Per-speaker-segment cached |

With caching, repeat runs are instant regardless of approach.  The first-run
cost is roughly 1.5-2x for per-speaker UTR, dominated by the diarization
call.  If diarization is already cached (from a prior `transcribe` run), the
overhead is just the additional smaller ASR calls, which can run in parallel.

## Error Handling

### Diarization fails

Fall back to global UTR.  Log a warning.  This is the same fallback path as
when the diarization worker is not available.

### Speaker mapping is ambiguous

If the heuristic mapping cannot confidently assign diarization speakers to
CHAT speakers (e.g., two CHAT speakers have similar temporal overlap with the
same diarization speaker), fall back to global UTR for the ambiguous speakers
and use per-speaker UTR only for the confidently mapped ones.

### Per-speaker ASR returns empty

Some segments may be silence or noise.  Empty ASR results are normal — the
per-speaker DP simply has fewer tokens to match against and some utterances
remain untimed.  This is no worse than the global UTR case.

### Per-speaker UTR produces worse coverage than global

Possible in theory if diarization is bad (wrong speaker boundaries cause
words to appear in the wrong speaker's ASR stream).  The retry heuristic
should compare per-speaker coverage against global coverage and use whichever
is better.  Log the comparison for debugging.

## Rust Implementation

### New types

```rust
/// Per-speaker UTR context for one speaker.
struct SpeakerUtrContext {
    /// CHAT speaker code (e.g., "PAR", "INV").
    chat_speaker: String,
    /// Diarization speaker label (e.g., "SPEAKER_0").
    diarization_speaker: String,
    /// Indices of utterances in the CHAT file attributed to this speaker.
    utterance_indices: Vec<usize>,
    /// Diarization segments for this speaker.
    segments: Vec<SpeakerSegment>,
}

/// Mapping from diarization speaker labels to CHAT speaker codes.
struct SpeakerMapping {
    /// Map from diarization label → CHAT code.
    mapping: BTreeMap<String, String>,
    /// Confidence score (0.0–1.0) based on temporal overlap quality.
    confidence: f64,
}

/// Per-speaker UTR result.
struct PerSpeakerUtrResult {
    /// Per-speaker injection counts.
    per_speaker: BTreeMap<String, UtrResult>,
    /// Overall injection counts (sum of per-speaker).
    overall: UtrResult,
    /// Whether any speaker fell back to global UTR.
    had_fallback: bool,
}
```

### New cache task

```rust
// cache_key.rs
impl CacheTaskName {
    pub const PER_SPEAKER_UTR_ASR: &'static str = "per_speaker_utr_asr";
}

pub fn per_speaker_utr_cache_key(
    audio_identity: &str,
    speaker_label: &str,
    start_ms: u64,
    end_ms: u64,
    lang: &LanguageCode3,
) -> CacheKey {
    CacheKey::from_content(&format!(
        "per_speaker_utr_asr|{audio_identity}|{speaker_label}|{start_ms}|{end_ms}|{lang}"
    ))
}
```

### New functions in `utr.rs`

```rust
/// Build per-speaker UTR contexts from diarization results and CHAT speakers.
pub fn build_speaker_contexts(
    chat_file: &ChatFile,
    diarization: &[SpeakerSegment],
) -> Result<Vec<SpeakerUtrContext>>;

/// Map diarization speakers to CHAT speaker codes using temporal overlap.
pub fn map_diarization_to_chat_speakers(
    chat_file: &ChatFile,
    diarization: &[SpeakerSegment],
) -> SpeakerMapping;

/// Inject UTR timing for a single speaker's utterances.
/// Same algorithm as inject_utr_timing but filtered to one speaker.
pub fn inject_utr_timing_for_speaker(
    chat_file: &mut ChatFile,
    asr_tokens: &[AsrTimingToken],
    context: &SpeakerUtrContext,
) -> UtrResult;

/// Merge adjacent diarization segments for the same speaker when the
/// gap is below threshold_ms.
pub fn merge_adjacent_segments(
    segments: &[SpeakerSegment],
    threshold_ms: u64,
) -> Vec<SpeakerSegment>;
```

### New dispatch in `runner/dispatch/utr.rs`

```rust
/// Run per-speaker UTR: diarize → extract per-speaker audio → ASR per
/// speaker → per-speaker DP → merge.
pub async fn run_per_speaker_utr_pass(
    ctx: &UtrPassContext<'_>,
    chat_file: &mut ChatFile,
    audio_path: &Path,
) -> Result<PerSpeakerUtrResult>;
```

This function orchestrates:

1. Request diarization via existing `build_speaker_request_v2`.
2. Call `map_diarization_to_chat_speakers`.
3. For each speaker (parallelizable with `futures::join_all`):
   a. `merge_adjacent_segments` for the speaker.
   b. For each merged segment, check cache → extract audio → run ASR.
   c. Collect all ASR tokens for the speaker.
   d. Call `inject_utr_timing_for_speaker`.
4. Return `PerSpeakerUtrResult`.

### Integration via `UtrStrategy` trait

Per-speaker UTR is one implementation of the `UtrStrategy` trait described
in [Trait-Based Dispatch](../decisions/trait-based-dispatch.md).  The
orchestrator selects the strategy based on `--utr-strategy per-speaker`
(hidden flag during experiments) and calls `strategy.run(ctx, chat_file,
audio_path)`.

The escalation heuristic (run global first, retry per-speaker if coverage
is poor) is implemented as a composite strategy that wraps both:

```rust
/// Tries global UTR first, escalates to per-speaker if coverage is poor.
struct EscalatingUtr {
    global: GlobalUtr,
    per_speaker: PerSpeakerUtr,
}

impl UtrStrategy for EscalatingUtr {
    async fn run(&self, ctx, chat_file, audio_path) -> Result<UtrResult> {
        let global_result = self.global.run(ctx, chat_file, audio_path).await?;
        if should_escalate(&global_result, chat_file) {
            let ps_result = self.per_speaker.run(ctx, chat_file, audio_path).await?;
            if ps_result.coverage() > global_result.coverage() {
                return Ok(ps_result);
            }
            // Per-speaker was worse — revert to global.
            warn!("per-speaker UTR worse, reverting to global");
            revert_to_global(ctx, chat_file, audio_path).await?;
        }
        Ok(global_result)
    }
}
```

This keeps the orchestrator simple — it just calls `strategy.run()` — and
moves the escalation logic into a dedicated strategy impl.

## Suggested Implementation Order

1. **Speaker mapping** (`map_diarization_to_chat_speakers`).  Unit-testable
   with synthetic CHAT files and mock diarization segments.

2. **Per-speaker reference filtering** (`build_speaker_contexts`).
   Unit-testable: given a CHAT file, verify correct utterance index
   partitioning by speaker.

3. **Per-speaker DP** (`inject_utr_timing_for_speaker`).  Refactor existing
   `inject_utr_timing` to accept an utterance index filter.  Verify
   identical results when the filter includes all utterances.

4. **Per-speaker dispatch** (`run_per_speaker_utr_pass`).  Integration test
   with a real audio file and a CHAT file with multiple speakers.

5. **Escalation heuristic** (`should_escalate_to_per_speaker`).  Run on
   APROCSA corpus, measure coverage delta.

6. **Backbone extraction composition.**  Integrate the `&*` stripping from
   Approach 1 into the per-speaker reference building.

## Test Strategy

- **Unit tests:** Speaker mapping with synthetic diarization segments and
  CHAT files.  Per-speaker reference partitioning.  Per-speaker DP on
  hand-crafted word sequences.

- **Integration tests:** Full per-speaker UTR pass on a real audio file
  with known diarization output.  Compare timed-utterance coverage against
  global UTR.

- **Golden tests:** Run on existing golden test files.  Files without
  dense overlap should produce identical results (per-speaker UTR should
  not activate or should produce equivalent timing).

- **Corpus validation:** Run on APROCSA and other overlap-heavy corpora.
  Report per-file timed-utterance percentage for global UTR vs. per-speaker
  UTR.  Target: 2265_T4 class drops from ~35% untimed to ~5-10%.

- **Regression guard:** Coverage comparison (per-speaker vs. global) is
  logged for every file.  If per-speaker ever produces worse coverage, it
  falls back and logs a warning.  The test suite should include at least one
  file where per-speaker UTR is expected to help and one where it is
  expected to be neutral.
