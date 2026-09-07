# Forced Alignment Design

**Status:** Current
**Last updated:** 2026-09-06 23:59 EDT

## Overview

The `align` command adds word-level timestamps to CHAT files. Given a transcript
and an audio file, it determines exactly where each word appears in the recording.

```mermaid
flowchart TD
    chat["CHAT file + audio"]
    parse["Parse CHAT → AST"]
    reuse{"Complete reusable\n%wor timing?"}
    refresh["Refresh main-tier bullets\nfrom %wor + optionally\nregenerate %wor"]
    partial{"Per-utterance\npartial %wor?"}
    partial_refresh["Refresh clean utterances\nfrom %wor, track reusable set"]
    count{"Untimed\nutterances?"}
    utr["UTR pre-pass:\nrun_utr_pass() with ASR caching\n+ partial-window optimization"]
    group["Group utterances\nby time window + token limit"]
    extract["Extract words per group"]
    cache{"Cache\nlookup"}
    fa["execute_v2('fa', prepared_audio + prepared_text)\n→ Whisper/Wave2Vec"]
    dp["DP alignment\n(Hirschberg O(n+m) space)\ntokens → transcript words"]
    inject["Inject timings\n+ generate %wor tier"]
    mono["Enforce document order\nstrip backward starts; resolve same-speaker\noverlap by speaker stream, not file adjacency"]
    out["Serialize → CHAT"]
    retry{"FA failed +\nuntimed?"}
    fallback["Fallback UTR:\nrun_utr_pass() (once)"]

    chat --> parse --> reuse
    reuse -->|"yes"| refresh --> out
    reuse -->|"no"| partial
    partial -->|"some clean"| partial_refresh --> count
    partial -->|"none"| count
    count -->|"yes + utr_engine"| utr --> group
    count -->|"no (all timed)"| group
    count -->|"yes + no utr_engine"| group
    group --> extract --> cache
    cache -->|miss| fa --> dp --> inject
    cache -->|hit| inject
    inject --> mono --> out

    fa -.->|"error"| retry
    retry -->|"yes (first time)"| fallback --> group
    retry -->|"no"| backoff["Retry with backoff"]
    backoff --> extract
```

## Prerequisites

Before alignment can begin, two steps must succeed:

1. **Media resolution**: The server locates the audio file for the CHAT
   file from server-visible local paths, either alongside it (paths mode),
   through a shared `source_dir`, via local `media_mappings`, or via an
   explicit `--media-dir`. See
   [Media Conversion](media-conversion.md#media-resolution).
2. **Media conversion**: If the audio is in a container format that
   `soundfile` cannot read (MP4, M4A, WebM, WMA), it is automatically
   converted to 16 kHz mono WAV via ffmpeg and cached at
   `~/.batchalign3/media_cache/`. See [ensure_wav](media-conversion.md#ensure_wav--conversion-cache).

Both steps happen in the Rust server before any Python worker is invoked.

## Execution flow and ownership

The most important architectural fact is that **direct mode and explicit server
mode share the same FA pipeline**.

- Without `--server`, the CLI runs `align` through `DirectHost`. No HTTP server
  or daemon is spawned for that path.
- With `--server`, `align` submits a shared-filesystem `paths_mode` job. The
  execution host must be able to read the submitted source path, resolve media
  from that host's local filesystem view, and write the requested output path.

Both routes end up in `process_one_fa_file()` in
`crates/batchalign/src/runner/dispatch/fa_pipeline.rs`.

### Single-parse architecture

The CHAT file is parsed **once** into a `ChatFile` AST. UTR mutates the AST in
place (no serialization). FA receives the same AST directly via
`run_fa_from_ast()` (no re-parse). The file is serialized **once** at the end.

```mermaid
flowchart TD
    start["align request"]
    host{"direct or\n--server?"}
    direct["DirectHost\ninline execution"]
    server["ServerBackend\nqueued paths_mode job"]
    media["Resolve media on the\nexecution host"]
    wav["ensure_wav()\ncontainer → cached WAV if needed"]
    parse["Parse CHAT → ChatFile AST\n(single parse)"]
    utr["UTR: mutate ChatFile in-place\n(no serialize)"]
    fa["run_fa_from_ast(ChatFile)\nFA grouping + worker dispatch\n(no re-parse)"]
    inject["Inject timings + %wor"]
    finalize["FaFinalized\noptional repair, then typed monotonicity"]
    serial["Serialize ChatFile → CHAT text\n(single serialize)"]

    start --> host
    host --> direct --> media
    host --> server --> media
    media --> wav --> parse --> utr --> fa --> inject --> finalize --> serial
```

**Key functions:**
- `run_utr_pass(&mut ChatFile, ...)`: mutates AST, returns `UtrResult`
- `run_fa_from_ast(ChatFile, ...)`: accepts AST directly, returns `FaResult`
- `process_fa(&str, ...)`: parse-then-delegate wrapper for callers that only have text (transcribe, incremental)

## Pipeline: UTR then FA

Alignment runs as a two-step pipeline:

1. **UTR (Utterance Timing Recovery)**: Assigns utterance-level timing boundaries.
2. **FA (Forced Alignment)**: Assigns word-level timing within those boundaries.

### Step 0: Cheap reuse for already word-timed files

Before UTR or FA grouping, `align` now checks whether the parsed file already
contains a complete reusable `%wor` tier. This is the cheapest safe rerun path
for files that have already been aligned once and are being passed through
`align` again.

The important detail is that this check is **not** based only on
`main`-tier `Word.inline_bullet`. After a CHAT parse roundtrip, main-tier word
timing may be represented as `InternalBullet` tokens while `%wor` carries the
durable first-class timing bullets. The server therefore:

1. verifies that every alignable main-tier word has a clean main↔`%wor`
   positional mapping,
2. verifies that every mapped `%wor` word has a timing bullet,
3. copies those timings back onto main-tier words,
4. removes parsed `InternalBullet` tokens, and
5. refreshes utterance bullets and optionally regenerates `%wor`.

If that verifier succeeds, `align` skips FA entirely for the file.

When the whole-file check fails, a per-utterance check
(`find_reusable_utterance_indices`) identifies which utterances still have
clean `%wor`. Those are refreshed in place; the rest proceed through normal
FA grouping. During the group partition step, groups where **all** utterances
are in the reusable set have their timings collected directly from the
refreshed main tier (no cache lookup or worker call needed).

```mermaid
flowchart TD
    start["Parse CHAT"]
    wor{"Complete reusable\n%wor timing?"}
    map["Verify main↔%wor mapping\nand collect word timings"]
    rehydrate["Rehydrate main-tier word bullets\nfrom %wor"]
    refresh["Refresh utterance bullets\n+ optionally regenerate %wor"]
    partial{"Any utterances with\nclean %wor?"}
    partial_refresh["Refresh clean utterances\ntrack reusable set"]
    normal["Continue to UTR/FA pipeline\n(stale groups only need workers)"]

    start --> wor
    wor -->|"yes"| map --> rehydrate --> refresh
    wor -->|"no"| partial
    partial -->|"some"| partial_refresh --> normal
    partial -->|"none"| normal
```

### Step 1: UTR (detect-and-skip)

UTR mutates the `ChatFile` AST **in place** via `run_utr_pass(&mut ChatFile, ...)`.
No serialization occurs, the same AST flows directly to FA. The orchestration
lives in `crates/batchalign/src/runner/dispatch/utr.rs`; the core injection
algorithm lives in `crates/batchalign/src/chat_ops/fa/utr.rs`.

**Detection:** `count_utterance_timing()` counts timed vs untimed utterances.
If all utterances are timed, UTR is skipped entirely (the common case for
production CHAT files from CLAN).

**Recovery:** When untimed utterances exist and a UTR engine is configured
(`--utr`, the default), the `run_utr_pass()` helper:

1. Checks the **ASR cache** for a prior result (key includes audio identity + lang).
   On hit, skips inference entirely, repeat runs are instant.
2. Chooses **partial-window** or **full-file** mode:
   - **Partial-window** (when >50% timed and audio >60s): `find_untimed_windows()`
     identifies time regions covering only the untimed utterances (with 500ms
     padding). Each window is extracted via `extract_audio_segment()` (ffmpeg
     `-ss`/`-to` → cached WAV) and ASR runs only on those segments. Token
     timestamps are offset by the window start time. Each segment's result is
     cached independently.
   - **Full-file** (mostly-untimed or short audio): ASR runs on the full audio
     and the result is cached as a single entry.
3. Converts ASR response tokens to `AsrTimingToken` (text + start_ms + end_ms).
4. Calls `inject_utr_timing()`. That function first tries a cheap exact
   monotonic subsequence match for the whole document word list against the ASR
   word list. If that match is unique, timing assignment is linear-time and no
   DP is needed. If the match is missing or ambiguous, UTR falls back to a
   **single global Hirschberg DP alignment** of all document words (timed +
   untimed) against the projected ASR words, using **case-insensitive exact
   matching** for the default global strategy. Timed utterances participate in
   the alignment to anchor their neighbors but their bullets are left
   unchanged. For each untimed utterance, the min/max matched ASR token
   indices determine the utterance bullet's time span. The global alignment
   avoids the token-exhaustion problem that per-utterance windowed approaches
   suffer from. It is still a monotonic aligner, so dense overlap /
   text-audio reordering remains a known limitation.

   **Two-pass only:** `--utr-fuzzy <threshold>` configures the experimental
   `--utr-strategy two-pass` strategy (default 0.85; 1.0 for exact-only
   matching). It does not change the default global strategy.
5. Re-serializes the CHAT with recovered bullets.

```mermaid
flowchart TD
    words["Transcript words + ASR words"]
    unique{"Unique exact\nmonotonic subsequence?"}
    exact["Assign utterance token ranges\nwithout DP"]
    dp["Run one global Hirschberg DP\nacross the whole file"]
    bullets["Inject/recover\nutterance bullets"]

    words --> unique
    unique -->|"yes"| exact --> bullets
    unique -->|"no"| dp --> bullets
```

#### UTR alignment evidence and offline replay

Provider tokens may contain whole phrases. `UtrLexicalStream` projects each
token into nonempty words separated by Unicode whitespace before global or
local overlap matching. It owns the relationship between those words and the
original token stream. Matches record the original `token_index` and the
`word_index` within its text; skipped blank tokens do not renumber later
provider tokens. This removes the mismatch in which transcript words were
compared against whole Whisper segments.

Projection preserves the provider's interval for every word in a segment.
It does not interpolate word timestamps, remove punctuation, or infer language
segmentation where whitespace is absent. Two utterances matched within one
segment can therefore receive the same coarse interval. Forced alignment must
still establish word boundaries. Existing cached ASR responses remain reusable:
the projection happens after reading the retained provider response, and no
new inference is required to replay it.

The global alignment plan is a first-class typed value. It records the chosen
strategy, participation policy, every utterance state, every monotone
word-to-token match, its lexical relation, and the timing proposal derived
from the first and last matched token. Deliberately excluded overlap lines are
distinct from attempted but unmatched lines. A matched utterance has a
nonempty match collection by construction. Positive and nonpositive proposals
are distinct states.

`UtrResult` retains this plan even though production projection still changes
bullets only for untimed utterances. This distinction lets experiments inspect
what the same global alignment proposed for already timed neighbors without
claiming that BA3 changed those authoritative bullets.

For an offline replay over retained debug artifacts:

```bash
batchalign3 eval utr-alignment \
  --chat recording_utr_input.cha \
  --tokens recording_utr_tokens.json \
  --fuzzy-threshold 0.85 \
  --output recording_utr_alignment.json
```

The command performs no inference and does not mutate CHAT. It requires a
clean parse, fingerprints both inputs with BLAKE3, records the executable build
identity, and atomically publishes a complete report without overwriting an
existing output. Omit
`--fuzzy-threshold` for case-insensitive exact matching. Add
`--participation exclude-marked-overlap` to
replay the participation rule used by the first pass of two-pass UTR.

Offline report schema 2 adds the within-token word address and uses this
lexical projection. The input token JSON format is unchanged. Older reports
retain their original meaning and should be kept alongside a fresh replay.

The resulting proposal is evidence, not a final main-tier or `%wor` policy.
Complete lexical coverage does not by itself establish word-boundary accuracy,
and partial or unmatched states require an explicit abstention or fallback in
downstream research code.

#### Overlap Strategy Selection

When a CHAT file contains overlapping speech (`+<` linkers or `⌊` CA markers),
the standard global UTR alignment degrades because the monotonic matcher cannot
represent the temporal crossing of overlapping speakers.

**Current default:** `auto` always uses `GlobalUtr`. The experimental
two-pass strategy is available via `--utr-strategy two-pass` but is not
production-ready, it has not been validated on enough corpora and can
regress alignment on real files.

The two strategies are:

```mermaid
flowchart TD
    start(["UTR alignment requested"])
    scan{"Any utterance has\n+&lt; linker or ⌊ CA\noverlap markers?"}
    scan -->|No| global["GlobalUtr\n(crates/batchalign/src/chat_ops/fa/utr.rs)\nSingle monotonic DP\nover all words"]
    scan -->|Yes| twopass["TwoPassOverlapUtr\n(crates/batchalign/src/chat_ops/fa/utr/two_pass.rs)"]
    twopass --> density{"Overlap density\n> max_exclusion_density?"}
    density -->|"Yes (>30%)"| global_anyway["Include +&lt; utterances\nin global DP\n(too many to exclude)"]
    density -->|"No (≤30%)"| pass1["Pass 1: Global DP\nexcluding +&lt; utterances"]
    pass1 --> pass2["Pass 2: Per-overlap recovery\nnarrow ASR window around\nprevious utterance's bullet"]
    global --> result(["UtrResult"])
    global_anyway --> result
    pass2 --> result
```

**Strategy selection** (`select_strategy()` in `crates/batchalign/src/chat_ops/fa/utr.rs`):

| Strategy | When selected | Behavior |
|----------|--------------|----------|
| `GlobalUtr` | No `+<` or `⌊` markers in file | Single monotonic DP over all words (original algorithm) |
| `TwoPassOverlapUtr` | Any utterance has `+<` or `⌊` | Pass 1 excludes overlap utterances from global DP; Pass 2 recovers their timing from a narrow ASR window around the previous utterance |

The two-pass approach prevents overlap utterances from desynchronizing the
global alignment. Pass 2 uses CA markers (`⌈⌉⌊⌋`) when present to narrow the
search window further (±`tight_buffer_ms` around the CA onset position).

When overlap density exceeds `max_exclusion_density` (default 30%), excluding
overlaps would starve the global DP of context, so all utterances are included
in a single pass instead.

**CLI flags for overlap control:**

| Flag | Default | Effect |
|------|---------|--------|
| `--utr-strategy auto\|global\|two-pass` | `auto` | Override automatic strategy selection |
| `--utr-ca-markers enabled\|disabled` | `enabled` | Whether Pass 2 uses CA markers for window narrowing |
| `--utr-density-threshold <0.0-1.0>` | `0.30` | Overlap fraction above which two-pass falls back to global |
| `--utr-tight-buffer <ms>` | `500` | Buffer around previous utterance for Pass 2 recovery window |
| `--utr-fuzzy <threshold>` | `0.85` | Two-pass only: Jaro-Winkler similarity threshold |

The fuzzy threshold and overlap-density threshold are finite closed-interval
types. CLI and serialized configuration inputs outside `0.0` through `1.0`
are rejected before alignment policy exists.

The two-pass defaults were tuned on SBCSAE, Jefferson NB, TaiwanHakka, and
APROCSA corpora but have not been broadly validated. Use `--utr-strategy
two-pass` to opt in for experimentation.

#### %wor Suppression for CA Transcripts

When `@Options: CA` is set, `align` automatically suppresses `%wor` tier
generation. Conversation Analysis transcripts use prosodic notation
(`⌈⌉⌊⌋`, arrows, lengthening marks) that `%wor` cannot represent, so
generating it adds noise that CA researchers must manually remove. The
`--nowor` flag achieves the same effect for non-CA files.

#### End-Time Overlap Clamping

After alignment, `enforce_monotonicity()`'s Pass 2 resolves end overlap in
TWO sweeps, in this order (2026-09-01 review, item 15):

1. **Same-speaker, by SPEAKER STREAM, unconditionally.** Each speaker's own
   bulleted utterances, in file order, are paired and resolved
   CONSECUTIVELY WITHIN THAT SPEAKER'S OWN STREAM -- an intervening
   other-speaker utterance is skipped, never breaks the pairing. This runs
   regardless of `--end-overlap-policy`: E704 (CLAN 133, a speaker may not
   overlap themself) is defined on the speaker's own sequence, not on
   physical line adjacency, and an intervening line (ordinary A-B-A
   dialogue) must not hide a same-speaker overlap from resolution. The word
   "adjacent" does NOT describe this sweep's pairing; only file position
   within one speaker's own stream does.
2. **Then, additionally under `--end-overlap-policy clamp-all-adjacent`
   only** (the default is `preserve-cross-speaker`), every PHYSICALLY
   adjacent pair regardless of speaker. This sweep cannot undo sweep 1: every
   resolution below only SHRINKS the pair it touches, so a pair sweep 1
   already resolved satisfies sweep 2's own entry guard (no overlap left) and
   is silently skipped, not re-clamped.

Either sweep classifies from MEASURED word timings, never guessed, into one
of three cases (`coverage_only`, `boundary_from_words`, `interleaved_words`;
see [Monotonicity warnings](#monotonicity-warnings) below for the full
breakdown), and either sweep's `BoundaryFromWords` guards against creating a
fresh FILE-ORDER start violation the SAME way, regardless of which sweep
formed the pair (2026-09-01 review, item 16; see below): the next BULLETED
utterance in file order, any speaker, not "whatever this sweep would pair
`next` with" -- that narrower reading was itself the item-16 regression,
since sweep 1's own next same-speaker utterance can sit well past an
intervening different-speaker line the move would actually violate. UTR
token-range assignment is one source of overlap, but a rerun
can also expose stale prior bullets, segmentation conflict, conversational
overlap, or fresh FA evidence that crosses the next main-tier boundary. The
decision records the neutral cause `adjacent_utterance_overlap` (a legacy
name predating sweep 1; it names the CAUSE category, not the pairing rule)
plus which of the three resolutions applied. Only `interleaved_words` (a
genuine word conflict) requests review; the other two never touch a measured
word and never need one. This is current behavior, not evidence that a
clamped boundary is acoustically correct.

A `BoundaryFromWords` resolution can also move the FOLLOWING utterance's
start forward, to its own measured word hull. When doing so would push it
to or past the start of the next BULLETED utterance IN FILE ORDER, ANY
SPEAKER (2026-09-01 review, item 16 -- not that speaker's own next
utterance, corrected from an earlier, narrower guard), the resolution falls
back to `interleaved_words` for that pair instead, so the move never
happens and nothing is stripped. This is Pass 1's own rule (file-order
start monotonicity across every speaker) applied to Pass 2's own output:
Pass 1 already ran and cannot see a start Pass 2 is about to create, so
Pass 2 must not undo what Pass 1 established. `orchestrate::file_order_successor_start_ms`
is the ONE function both Pass 2 sweeps and repair's own boundary-averaging
call for this (2026-09-01 review, item 16): it walks forward from the pair's
`next` utterance, live, to the first utterance that CURRENTLY has a bullet,
regardless of speaker, so an earlier pair's strip in the SAME sweep is
already reflected. Real-data regression this closed: chatter's E362 fired
1,353 times across 178 files when a same-speaker `BoundaryFromWords` moved
a start past an intervening different-speaker line's start, because the
per-speaker-stream sweep's guard (item 15) checked only that speaker's OWN
next utterance, missing the intervening line entirely.

Source: strategy selection in `crates/batchalign/src/chat_ops/fa/utr.rs`,
two-pass config and algorithm in `crates/batchalign/src/chat_ops/fa/utr/two_pass.rs`.

**Current debugging hook:** When `$BATCHALIGN_DEBUG_DIR` is set, UTR writes the
pre-injection CHAT and the ASR timing tokens that fed `inject_utr_timing()`.
That is enough to reproduce token-starvation failures offline. It is not yet a
full stage-by-stage trace: normalized word lists, DP match pairs, unmatched
utterance reports, interpolation windows, and post-monotonicity stripping still
need richer tracing support.

**Fallback UTR:** When the initial UTR pre-pass fails (ASR error) or is skipped
(`--no-utr`) and FA subsequently fails with a retryable error, the retry handler
attempts UTR once before the next retry. This recovers files where bad
interpolated timing caused FA failure. The `utr_fallback_attempted` flag ensures
at most one extra ASR call across all retries.

**No engine fallback:** When no UTR engine is configured (`--no-utr`), untimed
utterances fall back to proportional interpolation (see
[Proportional FA Estimation](../../architecture/alignment/proportional-fa-estimation.md)).

UTR injects **utterance-level bullets only**: it does not set word-level
timing. FA handles word-level alignment after UTR provides the boundaries.

### Step 2: FA

FA takes the utterance boundaries (from UTR or from the original CHAT) and
groups them into segments. For each segment, it extracts the corresponding audio
chunk and runs the FA model (Whisper cross-attention DTW or Wave2Vec CTC
alignment) to get precise word-level timestamps.

The FA model returns timestamps **relative to the chunk start** (0-based). These
must be converted to absolute timestamps by adding the group's `audio_start_ms`
offset before injection into the AST.

#### FA grouping strategy

Groups are formed by `group_utterances()` (at
`crates/batchalign/src/chat_ops/fa/grouping.rs:42`). Each group maps
to one FA worker call.
Grouping is driven by two independent constraints, a group is flushed and a new
one started when either is exceeded:

| Constraint | Default | Rationale |
|------------|---------|-----------|
| **Time window** (`max_group_ms`) | 20 000 ms | Caps audio segment length sent to the FA worker. Larger windows give the model more context but increase latency and memory. |
| **Character-token limit** (`WHISPER_FA_MAX_LABEL_TOKENS = 448`) | 448 chars | Whisper's CTC forced-alignment backend (`ctc_loss` / `ctc_best_path`) counts each character of each word as one label token. The maximum allowed sequence length is exactly 448. Exceeding it produces a hard `ValueError: Labels' sequence length N cannot exceed the maximum allowed length of 448 tokens`. |

The char-token limit exists because the time window alone is insufficient. Dense
speech (fast talkers, long-word languages like Spanish) can accumulate hundreds
of words, and thus thousands of characters, inside a normal 20-second window.
The first time this was observed in production was on the `biling-data` corpus
(`DiazCollazos/09.cha`, Spanish), where group 0 carried 2043 chars in 10.97
seconds, and `DiazCollazos/01.cha` group 14 carried 2523 chars in 14.56 seconds.
Both crashed Whisper FA.

```mermaid
flowchart TD
    utt(["Next utterance\n(with timing bullet or estimate)"])
    extract["collect_fa_words()\nExtract words → count chars\n(utt_chars)"]
    over_time{"(current duration +\nnew duration) > max_group_ms\nor time went backwards?"}
    over_chars{"current_chars +\nutt_chars > 448?"}
    flush["Flush current group\nReset: current_chars = 0\nExtend audio window into gap"]
    add["Add utterance words to group\ncurrent_chars += utt_chars"]
    more{"More\nutterances?"}
    final["Flush final group\n(extend into trailing audio)"]

    utt --> extract --> over_time
    over_time -->|"yes"| flush
    over_time -->|"no"| over_chars
    over_chars -->|"yes and group non-empty"| flush
    over_chars -->|"no"| add
    flush --> add
    add --> more
    more -->|"yes"| utt
    more -->|"no"| final
```

**Edge case, single utterance exceeds 448 chars:** If one utterance alone
exceeds the limit, the flush guard is skipped (`!current_words.is_empty()` is
false) and the utterance is included in its own group regardless. The worker
will fail for that group, but the error is scoped to that group alone. The
alternative, silently dropping the utterance, would corrupt injection by
skewing the word-cursor alignment for every subsequent group.

**Character count vs. actual tokens:** The 448-char limit is a conservative
proxy. Whisper's CTC vocabulary has some multi-byte characters and some
characters that map to the CTC blank token (contributing a wildcard instead of
themselves). In practice, the character count is a safe upper bound: the actual
label sequence is never longer than the sum of word character lengths.

**Implementation note, why the word count is computed before the split
decision:** The split must know the new utterance's char count before deciding
whether to flush, so `collect_fa_words()` is called at the top of the loop
(before the flush check) and the words are held in `extracted` until after the
flush decision. This avoids calling `collect_fa_words()` twice.

Source: `crates/batchalign/src/chat_ops/fa/grouping.rs`,
constant `WHISPER_FA_MAX_LABEL_TOKENS` (line 15).

### Failure points, recovery, and what the user sees

`align` has two distinct error scopes: **group-level** failures (affect one audio
window; other groups continue) and **file-level** failures (abort the whole
file).  The diagrams below show every fallback path.

#### Full fallback map

The outermost loop is the file-level retry loop in
`fa_pipeline.rs:process_one_fa_file()`.  The inner loop is the per-group
dispatch in `fa/transport.rs:infer_groups_v2()`.  These two loops share one
entry arrow in the diagram: `run_fa_from_ast()` / `process_fa_incremental()`.

```mermaid
flowchart TD
    start(["process_one_fa_file()\nfa_pipeline.rs"])
    media["Resolve audio\n(5-tier search)"]
    parse["Parse CHAT → ChatFile AST\n(single parse, fa/mod.rs)"]
    utr_check{"Untimed\nutterances?"}
    utr_run["UTR pre-pass\nrun_utr_pass()\n(crates/batchalign/src/chat_ops/fa/utr.rs)"]
    utr_warn["WARN: untimed utterances,\nno UTR engine, proportional\ninterpolation only"]
    incremental{"--before PATH\nprovided?"}
    fa_full["run_fa_from_ast()\n(fa/mod.rs)"]
    fa_inc["process_fa_incremental()\n(fa/incremental.rs)"]
    fa_groups["Group utterances\n+ per-group dispatch\n(fa/transport.rs)"]
    fa_ok{"All groups\nresolved?"}
    finalize["FaFinalized\noptional bullet repair first"]
    mono["enforce_monotonicity_with_policy()\nstrip non-monotonic starts\nclamp ends per typed policy\n(chat_ops/fa/orchestrate.rs)"]
    mono_warn["WARN monotonicity:\nend_clamped / start_stripped\nfor affected utterances"]
    post_val["Post-validation\n(warn-only, never fatal)"]
    out["Serialize → CHAT\nwrite output"]
    fa_err{"Error type?"}
    utr_fallback{"Retryable +\nuntimed +\nfallback UTR\nnot yet tried?"}
    utr_fb["Fallback UTR:\nrun_utr_pass() once\nthen retry FA"]
    backoff["Retry with\nexponential backoff"]
    terminal["Terminal file error\nwith real message"]

    start --> media --> parse --> utr_check
    utr_check -->|"all timed"| incremental
    utr_check -->|"untimed + engine"| utr_run --> incremental
    utr_check -->|"untimed + no engine"| utr_warn --> incremental
    incremental -->|"yes"| fa_inc --> fa_groups
    incremental -->|"no"| fa_full --> fa_groups
    fa_groups --> fa_ok
    fa_ok -->|"yes"| finalize --> mono
    mono --> mono_warn -.->|"(warn only)"| post_val --> out
    fa_ok -->|"no"| fa_err
    fa_err -->|"retryable worker error"| utr_fallback
    utr_fallback -->|"yes"| utr_fb --> fa_groups
    utr_fallback -->|"no"| backoff --> fa_groups
    fa_err -->|"non-retryable\nor retry budget exhausted"| terminal
```

> Source files: `crates/batchalign/src/runner/dispatch/fa_pipeline.rs`,
> `crates/batchalign/src/fa/mod.rs`, `crates/batchalign/src/fa/transport.rs`,
> `crates/batchalign/src/chat_ops/fa/orchestrate.rs` (enforce_monotonicity, strip_e704_same_speaker_overlaps)

#### Per-group fallback detail

Each FA group goes through its own dispatch in `infer_groups_v2()`.
Group-level failures either resolve silently (leaving words unaligned) or
propagate upward as a file-level failure.

```mermaid
flowchart TD
    group(["FA group\n(audio window + words)"])
    wor{"Reusable corroborated\n%wor timing?"}
    raw{"Raw FA evidence\nadmitted?"}
    replay["Replay raw response through\ncurrent Rust projection"]
    derived{"Derived timing vector\nadmitted?"}
    cached_timings["Reuse derived timings\n(compatibility fallback)"]
    policy{"Cache policy permits\ninference?"}
    required_fail["RequireCache failure\n(no dispatch authority)"]
    build["build_forced_alignment_request_v2()\n(worker/request_builder_v2.rs)"]
    empty{"EmptyAudioSegment?\n(0 PCM frames after ffmpeg)"}
    skip_empty["WARN: group past end of file\nLeave words unaligned\n→ continue to next group"]
    dispatch["dispatch_execute_v2()\n→ Python worker"]
    parse{"parse_group_response()\nparse_forced_alignment_result_v2()"}
    ok["Group timings resolved"]
    err_kind{"Error kind?\n(is_fa_runtime_failure,\nwhisper_fallback_reason,\nis_whisper_model_unavailable)"}
    ctc["Wave2Vec CTC fallback\n(see diagram below)"]
    model_unavail["ModelUnavailable: \ncapability gap\nLeave words unaligned\n+ WARN in server log"]
    runtime_fail["RuntimeFailure: \ndata-driven model error\nLeave words unaligned\n+ WARN in server log"]
    other["Other error: \ninfrastructure failure\n→ propagate to file loop"]

    group --> wor
    wor -->|"yes"| ok
    wor -->|"no"| raw
    raw -->|"yes"| replay --> ok
    raw -->|"absent/refused"| derived
    derived -->|"yes"| cached_timings --> ok
    derived -->|"no"| policy
    policy -->|"UseCache/SkipCache"| build --> empty
    policy -->|"RequireCache"| required_fail
    empty -->|"yes"| skip_empty
    empty -->|"no"| dispatch --> parse
    parse -->|"OK"| ok
    parse -->|"error"| err_kind
    err_kind -->|"Wave2Vec engine\n+ CTC overflow"| ctc
    err_kind -->|"ModelUnavailable:\nno whisper FA host"| model_unavail
    err_kind -->|"RuntimeFailure:\nmodel exception"| runtime_fail
    err_kind -->|"other"| other
```

#### Wave2Vec → Whisper CTC fallback

When the FA engine is Wave2Vec and the worker returns one of three specific
PyTorch CTC errors, `infer_groups_v2` retries **that single group** with Whisper
FA.  No other error triggers this retry.

```mermaid
flowchart TD
    w2v["Wave2Vec worker response\n(parse_group_response fails)"]
    reason{"whisper_fallback_reason()\n(fa/transport.rs)"}

    w2v --> reason

    r1["Reason: targets length\nis too long for CTC\n, group has too many words\nfor CTC context window"]
    r2["Reason: targets Tensor\nshouldn't contain blank index\n, word contains char that\nmaps to CTC blank token"]
    r3["Reason: Kernel size\ncan't be greater than\nactual input size\n, audio segment &lt; ~25ms\n(Wave2Vec conv layer 7\nneeds ≥ 2 samples after\n6 conv layers at 16 kHz)"]
    none["None, no fallback:\nerror propagates to\nfile-level handler"]

    reason -->|"'targets length is too long\nfor CTC'"| r1
    reason -->|"'targets Tensor shouldn't\ncontain blank index'"| r2
    reason -->|"'Kernel size can't be greater\nthan actual input size'"| r3
    reason -->|"any other error"| none

    whisper["Retry group with\nFaEngineName::Whisper\n(new request namespace\nto avoid artifact collision)"]
    fallback_ok["Group timings resolved\n+ FaFallbackEventTrace\nrecorded in job traces"]
    fallback_err{"Whisper response OK?"}
    model_unavail_err["ModelUnavailable:\nno Whisper model loaded: \nleave words unaligned\n+ WARN in server log\n(is_whisper_model_unavailable())"]
    runtime_fail_err["RuntimeFailure:\nmodel exception on this input, \nleave words unaligned\n+ WARN in server log\n(is_fa_runtime_failure())"]
    other_whisper_err["Other error\n(infrastructure failure)\n→ file-level failure"]

    r1 --> whisper
    r2 --> whisper
    r3 --> whisper
    whisper --> fallback_err
    fallback_err -->|"yes"| fallback_ok
    fallback_err -->|"ModelUnavailable"| model_unavail_err
    fallback_err -->|"RuntimeFailure"| runtime_fail_err
    fallback_err -->|"other error"| other_whisper_err
```

The `WARN` log line for a successful fallback:

```text
WARN fa_transport: Wave2Vec FA hit recoverable target constraint; retrying group with Whisper FA
     group=24 start_ms=379515 end_ms=381395 reason="targets length is too long for CTC"
```

A successful fallback appends a `FaFallbackEventTrace` to the job's trace
payload.  Inspect it with:

```bash
curl http://127.0.0.1:8001/jobs/JOB_ID/traces | python3 -m json.tool
```

#### Whisper fallback when worker has no Whisper model

The Python worker loads **exactly one** FA model at startup, controlled by
`engine_overrides["fa"]` in the worker bootstrap.  When Wave2Vec is the primary
engine, `whisper_fa_model` in `_WorkerState` is `None`.  If Wave2Vec hits a CTC
overflow and `infer_groups_v2` dispatches the Whisper fallback, the worker
returns `ExecuteOutcomeV2::Error { code: ModelUnavailable }`.

`is_whisper_model_unavailable()` in `crates/batchalign/src/fa/transport.rs`
detects the distinctive `"ModelUnavailable: no whisper FA host loaded"` substring
and treats this as a **worker capability gap**, not a data error.  The group's
words are left unaligned (same treatment as an empty audio segment) and the
server logs:

```text
WARN fa_transport: Whisper FA unavailable (worker has no Whisper model loaded);
     leaving group words unaligned
     group=24 start_ms=379515 end_ms=381395
```

The file is written with all other groups aligned normally.

To diagnose whether a worker has Whisper loaded, check the `health` endpoint:
if `loaded_pipelines` contains only `profile:gpu:eng`, Whisper FA is not
available and CTC-overflow groups will be left unaligned.

When Wave2Vec is the primary engine and CTC overflow occurs, the affected
utterances lose word-level timing.  Use the default Wave2Vec FA engine
(`--fa-engine whisper` or omit `--fa-engine`) to avoid this entirely.

#### RuntimeFailure errors are always group-local

Any `RuntimeFailure` from the FA worker, regardless of the specific Python
exception, is treated as a group-level skip. This is an architectural
invariant, not a special case for known patterns.

**Why:** A `RuntimeFailure` means the Python worker successfully received and
parsed the request, then the model raised an exception while processing
*this group's specific words and audio*. The failure is data-driven. Other
groups have different content; they will not trigger the same exception. There
is no value in aborting the file, the remaining groups can and should be
aligned normally.

**Contrast:** Infrastructure failures (`ProcessExited` = worker crash,
`Protocol` = IPC deserialization failure) indicate that the worker or channel
is broken. Every subsequent call would also fail. Those errors remain
file-level so the retry loop and fallback UTR path can attempt recovery.

**Detection:** `is_fa_runtime_failure()` in `transport.rs` matches any
`ServerError::Validation` message containing `"RuntimeFailure:"`. This
substring is inserted by `parse_forced_alignment_result_v2()` when formatting
a `ProtocolErrorCodeV2::RuntimeFailure` response. It does not appear in
`ModelUnavailable`, `Protocol`, or IPC parse errors.

**Ordering in `infer_groups_v2()`:**

1. `whisper_fallback_reason()` is checked first, Wave2Vec CTC patterns still
   trigger the Whisper retry (which may produce timings). `is_fa_runtime_failure`
   is only reached when the fallback logic has already decided no retry is
   possible.
2. `is_whisper_model_unavailable()` is checked before `is_fa_runtime_failure`
   in the Whisper fallback path, the capability-gap path emits a more specific
   warning message.

**Log lines:**
```text
WARN fa_transport: FA group failed with model RuntimeFailure (data-driven);
     leaving words unaligned
     group=0 start_ms=0 end_ms=10970 error="..."

WARN fa_transport: Whisper FA fallback also failed with model RuntimeFailure;
     leaving group words unaligned
     group=7 start_ms=100000 end_ms=111000 error="..."
```

#### Summary: all recovery behaviors for a single group

| Condition | Scope | Outcome | Log |
|-----------|-------|---------|-----|
| FA cache hit | Group | Reuse cached timings silently |, |
| Audio segment past EOF | Group | Leave words unaligned; continue | `WARN: group past end of file` |
| Wave2Vec CTC target overflow (3 patterns) | Group | Retry with Whisper; record fallback trace | `WARN: retrying group with Whisper FA` |
| Whisper retry succeeds | Group | Group timings resolved |, |
| Whisper retry: `ModelUnavailable` (worker has no Whisper model) | Group | Leave words unaligned; continue | `WARN: Whisper FA unavailable … leaving group words unaligned` |
| Worker `RuntimeFailure` (any model exception: token overflow, shape error, OOM, etc.) | Group | Leave words unaligned; continue, `is_fa_runtime_failure()` demotes to group-level | `WARN: FA group failed with model RuntimeFailure` |
| Whisper fallback also hits `RuntimeFailure` | Group | Leave words unaligned; continue | `WARN: Whisper FA fallback also failed with model RuntimeFailure` |
| Other worker error (retryable) | File | Retry with backoff; fallback UTR if untimed | `WARN: FA error (raw)` |
| Retry budget exhausted | File | Terminal failure with real error message |, |

#### Monotonicity warnings

After all groups are resolved, monotonicity enforcement makes Pass 1
(strip backward starts) then Pass 2, itself two sweeps (2026-09-01 review,
item 15): sweep 2a resolves same-speaker end overlap by SPEAKER STREAM
(unconditionally, any `--end-overlap-policy`, skipping intervening
other-speaker lines rather than requiring physical adjacency); sweep 2b
additionally resolves every physically adjacent pair, regardless of speaker,
under `--end-overlap-policy clamp-all-adjacent` only (the default,
`preserve-cross-speaker`, runs sweep 2b too, but it then skips every
cross-speaker pair and finds every same-speaker pair already resolved by
sweep 2a). See [End-Time Overlap Clamping](#end-time-overlap-clamping) above
for the full account and the proof sweep 2b cannot undo sweep 2a. Each
stripping or resolving decision is recorded in structured evidence and emits
a `WARN` log line. BA3 does not project these records into `%xalign` or
`%xrev` (see [Decision evidence](../user-guide/review-tiers-guide.md)). The
decision strategies have different severity and review priority:

| Decision | Cause | Needs review? | Action needed? |
|----------|-------|:---:|---|
| `end_clamped_coverage_only` | Only the bullet's inherited coverage overshot the next utterance's start; no measured word conflicted | **No** | Informational; automatic |
| `end_clamped_boundary_from_words` | Both bullets' inherited boundary replaced by their measured word hulls; the words never conflicted | **No** | Informational; automatic |
| `end_clamped_interleaved_words` | The words themselves interleave, or the next utterance has none: a genuine conflict | **Yes** | Adjudicate; the bullet and every affected word were clamped together |
| `start_stripped` | Utterance start precedes previous accepted start, full timing removed | **Yes** | Review utterance; may indicate transcript/audio reordering |

The three `end_clamped_*` strategies are classified from what is MEASURED
(word timings), never from the bullet's own extent, since the bullet at this
point may already be wider than its words (`update_utterance_bullet`'s
`Preserve` policy unions a bullet with prior coverage and never shrinks it).
Each strategy corresponds one-for-one to a `MonotonicityEffect::EndClamped*`
variant in structured evidence (`fa/orchestrate.rs`), and each variant embeds
one `OverlapEdge` (the two utterances involved: line index, ordinal, speaker,
on both sides) rather than repeating those seven fields per variant:

- **`end_clamped_coverage_only`**: the previous utterance's last measured
  word already ends at or before the next utterance's start (or the
  utterance has no measured words at all). Only the bullet's inherited
  coverage overshot; the bullet end moves to the word hull (or, with no
  words, to the next start directly). Words are untouched.
- **`end_clamped_boundary_from_words`**: the previous utterance's last
  measured word ends after the next start, but the two utterances' words do
  not interleave (the next utterance's first measured word starts at or
  after that end). The boundary is itself measurable, so BOTH bullets take
  their word-hull edges instead of the arbitrary next-start clamp; the far
  side of each bullet is untouched. Words are untouched. This resolution is
  refused, falling back to `end_clamped_interleaved_words` instead, when
  moving the next utterance's start to its measured hull would reach or
  pass the start of the next BULLETED utterance in FILE ORDER, ANY SPEAKER
  (2026-09-01 review, item 16): that would undo Pass 1's own file-order
  start-monotonicity guarantee, which already ran and cannot see a start
  Pass 2 is about to create, so the move never happens and nothing is
  stripped.
- **`end_clamped_interleaved_words`**: the two utterances' measured words
  genuinely overlap in time, or the next utterance has no measured word to
  fix a boundary against (or the boundary-from-words move above was
  refused). A real conflict between segmentation and FA: the bullet is
  clamped to the next start AND every previous-utterance word past that
  bound is clamped with it (through the same `clamped_to_bullet` route
  postprocessing uses, so the clamp is a real, invariant-checked cut rather
  than a second hand-rolled `.min()`), flagged for review.

```text
WARN monotonicity: strategy="end_clamped_interleaved_words" speaker=PAR line_idx=59
     reason="end_truncated_by=2160ms clamped_to=136005
             cuts_word_timing=true resolution=interleaved_words
             words_trimmed=1 words_dropped=0 cause=adjacent_utterance_overlap"

WARN monotonicity: strategy="start_stripped" speaker=INV line_idx=23
     reason="start_before_previous=130000 previous_start=131500"
```

`words_trimmed` and `words_dropped` are two different facts, not one count: a
trimmed word kept a shorter positive extent and still has a timing; a dropped
word's start was already at or past the bound, so nothing survives the cut
and a MEASURED (or transcript-carried) timing is thrown away. The prior
single `words_clamped` count could not tell these apart, so a caller reading
the log could not tell whether a review was looking at a word that merely
got shorter or one that lost its timing outright. Both are carried the same
way through `MonotonicityEffect::EndClampedInterleavedWords` and its
`FaTimingDecisionTrace` wire form, and the dropped side is not a count at all:
it is one record per word, carrying the extent that was lost.

Those discarded extents are also written to the run's evidence artifact as a
flat `dropped_word_timings` section, so a reviewer can recover every
measurement this pass threw away without walking the tagged effect union. Each
entry names the line and utterance, the speaker, the tier, the word's position
on that tier, the measured start and end, and the bound it exceeded. See
[Decision provenance](../developer/decision-provenance.md).

A note on why the words are cut rather than kept and marked: the natural
alternative is to preserve the timing and annotate the overlap. No typed CHAT
construct expresses that here. The discarded timings live on `%wor`, which is a
flat list of words and separators that carries no annotations at all; and
CHAT's overlap annotations `[<]` / `[>]` assert simultaneous speech by two
DIFFERENT speakers, while under the default policy this resolution only fires
on a pair sharing one speaker code, where a speaker overlapping themself is
exactly what the validator's speaker-self-overlap rule rejects. An untimed
`%wor` slot honestly says the moment is unknown; the evidence artifact is where
the measurement survives.

Neither `end_clamped_coverage_only` nor `end_clamped_boundary_from_words`
records identify an alignment defect. They are routine: the bullet's
inherited coverage was reconciled against the words that were actually
measured, and no word timing changed (`end_clamped_boundary_from_words` may
move a word's CONTAINER, i.e. the bullet, without moving the word itself).
`end_clamped_interleaved_words` is different: segmentation and word timing
genuinely disagree, and the output no longer contains every admitted word
interval on the previous utterance. BA3 keeps
that distinction in the decision reason and `needs_review`; the FA evidence
sidecar retains the original and clamped ends for controlled listening rather
than attributing every case to UTR. `%wor`, when written, is ALWAYS
regenerated after monotonicity resolves, never before, on every path: fresh
alignment (`apply_fa_results_with_projection_policy`), the all-reusable fast
path (`refresh_reusable_alignment` +
`projection_without_injection_with_touched`), and per-utterance partial reuse
(`refresh_reusable_utterances`, folded into the fresh-injection write phase
via `FaApplied::also_touched`). `add_wor_tier` is `pub(crate)`, and in a
PRODUCTION build it has exactly one caller, that one write phase
(`FaApplied::then_enforce_monotonicity`); its only other callers are
`#[cfg(test)]` (2026-09-01 review, item 12), so a word clamped by
`end_clamped_interleaved_words` can never be left stale on `%wor` while the
main tier (or vice versa) shows the uncut value, on any of those paths. The
write phase also does not trust its own candidate list: it re-derives, from
each utterance's CURRENT words, whether there is anything timed to write at
all, so an utterance that lost every word to a clamp or a strip (Pass 1,
Pass 2, or `repair_bullets`'s LIS removal) gets no `%wor` tier, never an
untimed one (2026-09-01 review, item 9). The defect this classification
replaces: the pass used to move only the bullet end back to the next
utterance's start, after `%wor` had already been written, so the measured
words ended after their own bullet (in one 253-file corpus, 30,858 word slots
did) and, under the former any-speaker default, real cross-speaker overlap
was cut as well.

`start_stripped` is a genuine alignment concern, the utterance's start
timestamp precedes the previous accepted start, which means time went backwards
in the file. This happens when text and audio order diverge (overlapping speech,
post-hoc transcript restructuring). The utterance's timing is completely removed
and its structured decision record has `needs_review=true`.

These warnings are **not errors**: they indicate normal behavior on overlapping
or densely-annotated speech.  High volumes of `start_stripped` warnings on a
file that previously aligned cleanly are a signal that the transcript was
restructured after the last alignment run.

See the [Monotonicity Invariant](#monotonicity-invariant) section for full detail.

### Post-processing

After FA injects raw word timings into the AST, three post-processing steps run
**in this order** for each utterance:

```mermaid
flowchart TD
    inject_words["inject_timings_for_utterance()\nSets word.inline_bullet on each\naligned word in the utterance"]

    inject_words --> postprocess["postprocess_utterance_timings()\nChains word end times;\nclamps word timings to utterance bullet range"]

    postprocess --> update["update_utterance_bullet()\nRecomputes utterance bullet\nfrom word inline_bullets"]

    update --> source{"utterance bullet\nBulletSource?"}

    source -->|"Utr\n(UTR provisional hint)"| overwrite["Overwrite: bullet = word span\nFA is authoritative.\nUTR estimate discarded."]
    source -->|"Authoritative\n(hand-linked or FA-derived)"| union_op["Union: bullet = min(word.start, existing.start)\n             ..max(word.end, existing.end)\nNever shrink, preserves\nfiller/gesture coverage"]
    source -->|"None\n(untimed, UTR also failed)"| set_new["Set: bullet = word span"]

    overwrite --> mark["Result marked Authoritative"]
    union_op --> mark
    set_new --> mark

    mark --> wor["add_wor_tier()\nGenerates %wor dependent tier\nfrom word inline_bullets"]
```

**Why `BulletSource` matters:** utterance bullets can come from two very
different origins with opposite desired behaviors after FA:

- **Authoritative bullets** (hand-linked by the researcher, or set by a
  previous FA run) may cover content that FA cannot align, a leading filler
  (`&-uh`) that FA returns `None` for, or a trailing gesture (`&=laughs`) that
  has no alignable word. Overwriting these bullets with the FA word span would
  silently shrink them and lose the hand-annotated timing context. The **union**
  ensures the bullet covers at least as much as before.

- **UTR hint bullets** (`BulletSource::Utr`) are rough grouping estimates
  derived from ASR token streams. They are provisional, their purpose is to
  give FA a window to work in, not to define the final timing. Once FA produces
  precise per-word timings, the hint should be discarded and the bullet set
  directly from the FA word span. This is the **self-healing property**: valid
  FA word timings always produce a valid utterance bullet, independent of how
  accurate the UTR estimate was.

`BulletSource` is a non-serialized field (`#[serde(skip)]`), it never appears
in CHAT output and has no effect on the file format. The distinction lives only
in memory during the align pipeline.

**Order matters:** `postprocess_utterance_timings` runs first so it can use the
UTR bullet for bounding word end times. If `update_utterance_bullet` ran first,
it would recompute from raw onset-only timings (Whisper FA gives only start
times), producing a bullet too tight for proper end-time chaining.

#### Word end times for an onset-only engine

Whisper FA returns onset times only: each token has a start time and no end
time. An end must therefore be derived, and the only end available is the next
word's onset, so the parser chains them: each word's end = the next word's
start.

This is not a mode, and there is no alternative treatment. A word whose end
equals its own start has no duration, which is the absence of a timing rather
than a timing; a `%wor` tier of such words is the symptom, not a setting.

The **last word** of each utterance has no next word to chain to. In the
parser it takes a named fallback (`LAST_WORD_FALLBACK_MS`, 500 ms); in
post-processing its end is extended to the utterance bullet end (from UTR or
original CHAT) when one is available.

#### Word timing clamping policy

After end-time chaining, `postprocess_utterance_timings()` optionally clamps
word timings to the utterance bullet range. This clamping is intentionally
**restricted** because two very different sources can produce an utterance
bullet, with opposite implications for whether clamping is safe.

**The two bullet sources:**

- **`BulletSource::Authoritative`**: the bullet was parsed from CHAT text.
  After a previous FA run (or after a researcher links audio manually in CLAN),
  the bullet covers the full speech span for that utterance. Clamping FA word
  timings to it is safe and desirable: it prevents a freshly-aligned word from
  wandering outside the known boundary.

- **`BulletSource::Utr`**: the bullet was injected by UTR at runtime
  (never serialized to disk). UTR's estimates are rough grouping hints, not
  confirmed boundaries. A UTR bullet may be far narrower than the actual speech.
  Rev.AI, for example, can produce a 220 ms hint for a 3-second utterance when
  it only matched the first word. Clamping FA word timings to a UTR hint would
  discard every correctly-aligned word beyond the first.

**The `BulletSource` persistence problem.**  `BulletSource` is a runtime-only
field, it is not serialized to CHAT text. When a file goes through the
`transcribe` + `utseg` pipeline, UTR injects utterance bullets, those bullets
are serialized to CHAT text, and then on the next `align` run they are parsed
back as `BulletSource::Authoritative`. Even though they originated from UTR
timestamps, there is no record of that fact after re-parsing.

After `transcribe` + `utseg`, ASR-derived bullets may be as narrow as the ASR
engine's first-word match, a systematic property of Rev.AI and similar engines
that emit short tokens only for the words they are confident about. If `align`
clamped FA word timings to these narrow ASR-derived bullets, it would silently
drop timings for every word after the first, producing output that appeared
aligned but missed most of the utterance.

**The `%wor` discriminator.** The presence of a `%wor` tier distinguishes
first-time alignment (post-transcribe, no `%wor`) from re-alignment (previous
FA run, `%wor` present). `%wor` is generated only by FA, so its presence
guarantees:

1. A previous FA run already aligned this utterance correctly, and
2. The utterance bullet was set from FA word timings and is wide enough to cover
   the full speech span.

Clamping is therefore **only applied when BOTH conditions hold**:

1. `bullet.source == BulletSource::Authoritative`: not a runtime UTR hint, AND
2. `utterance.wor_tier().is_some()`: this is a re-alignment, not a first-time alignment.

When either condition fails, FA word timings are injected as-is. The
`update_utterance_bullet()` call that follows then overwrites the narrow bullet
with the actual FA word span, the self-healing property described above. This
means valid FA timings always produce a valid utterance bullet, regardless of how
accurate or narrow the prior estimate was.

**Practical consequence.** On the `transcribe` → `align` workflow (the most
common first-time alignment case), clamping is never applied. All FA word
timings are accepted. The utterance bullet is widened to cover them. On
subsequent `align` re-runs (e.g., after transcript editing), clamping is
applied using the now-authoritative FA-derived bullet, preventing edited words
from being placed outside the established audio window.

**Source:** `crates/batchalign/src/chat_ops/fa/postprocess.rs:37`,
`postprocess_utterance_timings()`.

## What is fast today vs later

Today the implemented fast paths are:

- skip FA entirely when `%wor` already provides complete reusable word timing
- **per-utterance partial `%wor` reuse on plain reruns**: when the whole-file
  check fails (e.g. a user edited a few utterances), detect which utterances
  still have clean `%wor`, refresh those, and only send stale FA groups through
  workers. Groups where all utterances are reusable are preserved without cache
  lookup or worker call. Remaining groups resolve through raw-evidence replay,
  a derived-timing compatibility fallback, or an authorized inference miss.
- skip UTR when every utterance already has timing
- use a unique exact-subsequence UTR match when the transcript and ASR word
  streams line up without ambiguity

The next possible optimization is more selective escalation inside UTR:
identify local divergence regions ("trouble windows") and run global DP only
there. That is **not** implemented yet. The current global-DP path remains the
correctness baseline for difficult files.

## Why Two Steps

FA models (both Whisper and Wave2Vec) work best on short audio segments. Feeding
a 30-minute recording directly into the model produces poor alignment because
the attention/emission matrix gets too large and diluted. Chunking is necessary.

To chunk correctly, you need to know where each utterance sits in the audio.
That is what UTR provides. For already-timed input, the existing bullets serve
the same purpose.

## Untimed Utterance Handling

Four tiers of coverage for untimed utterances, in order of preference:

```mermaid
flowchart TD
    start{Untimed\nutterances?} -->|No| done([All timed: skip UTR])
    start -->|Yes| engine{UTR engine\nconfigured?}

    engine -->|Yes| ratio{">50% timed\n+ audio >60s?"}
    ratio -->|Yes| partial["Tier 1a: Partial-window ASR\nASR only on untimed regions\n(saves time on first run)"]
    ratio -->|No| full["Tier 1b: Full-file ASR\nASR on entire audio"]

    partial --> inject["inject_utr_timing()\n~100% coverage"]
    full --> inject

    engine -->|No| interp["Tier 2: Proportional interpolation\nEstimate from neighbors\n~96% coverage"]

    inject -->|"unmatched > 0"| interp
    inject -->|"all matched"| done2([Proceed to FA])

    interp --> done2

    done2 --> fa_fail{FA failed?}
    fa_fail -->|Yes + untimed| fallback["Tier 3: Fallback UTR\nrun_utr_pass() once\nbefore next retry"]
    fallback --> retry([Retry FA])
    fa_fail -->|No| output([Output])
```

1. **UTR with partial-window (default, mostly-timed files):** When >50% of
   utterances are timed and audio exceeds 60 seconds, ASR runs only on the
   untimed windows. Each window is extracted via ffmpeg and cached independently.
   First run saves time proportional to the untimed fraction; repeat runs hit
   cache.
2. **UTR full-file (default, mostly-untimed files):** ASR runs on the full
   audio. The result is cached for instant repeat runs.
3. **Proportional interpolation:** When UTR is disabled (`--no-utr`) or when
   UTR's ASR inference fails, the grouping algorithm interpolates untimed
   utterances between neighboring timed utterances by word count, with a
   2-second buffer. Achieves ~96% coverage.
4. **Fallback UTR:** When FA fails with a retryable error and untimed
   utterances were not recovered, UTR is attempted once before the next retry.
   This can recover files where bad interpolated timing caused the FA failure.
5. **Skip:** When `total_audio_ms` is unavailable and no timed neighbors exist,
   untimed utterances are excluded from FA grouping.

This is **smarter than ba2's skip logic**: ba2 skipped UTR for the entire file
if *any* utterance had timing (potentially missing untimed ones). ba3 skips
UTR only if *all* utterances are timed, and when UTR can't match a specific
utterance, proportional interpolation provides a fallback.

## Engine Selection

| Engine | Model | Response format | Default? |
|--------|-------|-----------------|----------|
| `whisper_fa` | Whisper large-v2 cross-attention DTW | `TokenLevel` (token text + onset seconds) | Yes |
| `wav2vec_fa` | MMS_FA CTC forced alignment | `WordLevel` (word text + start/end ms) | No |

Both return chunk-relative timestamps. The Rust orchestration (`parse_fa_response`)
handles offset addition for both formats.

## Offset Handling

This is the most critical correctness invariant in the pipeline.

Every FA model receives an audio chunk extracted from position `audio_start_ms`
to `audio_end_ms` in the full recording. The model returns timestamps relative
to the chunk start (time 0 = `audio_start_ms` in the full recording).

Before injecting timestamps into the CHAT AST, the offset must be added:

- **TokenLevel** (Whisper): `absolute_ms = time_s * 1000 + audio_start_ms`
- **WordLevel** (Wave2Vec): `absolute_ms = start_ms + audio_start_ms`

Failing to add the offset produces timestamps that are internally consistent
(words are correctly spaced relative to each other) but placed at the wrong
absolute position in the recording.

## Caching

Two independent cache layers affect alignment:

### UTR ASR cache

UTR ASR results are cached in the analysis cache (`CacheTaskName::UtrAsr`).
Two cache key schemes are used:

```mermaid
flowchart LR
    subgraph full ["Full-file cache key"]
        ff_input["BLAKE3(utr_asr_v2 | ASR provider | audio_identity | lang)"]
        ff_input --> ff_entry["Single AsrResponse"]
    end

    subgraph partial ["Segment cache keys"]
        seg_input["BLAKE3(utr_asr_segment_v2 | ASR provider | audio_identity | start_ms | end_ms | lang)"]
        seg_input --> seg_entry["Per-window AsrResponse"]
    end
```

Full-file keys are used for mostly-untimed files. Segment keys are used when
partial-window mode activates (>50% timed, audio >60s). Full-file and segment
entries are separate: a full-file entry does not guarantee a segment hit.
Provider identity is mandatory in both key constructors. Old normalized entries
without provider identity are retained on disk but are not automatically replayed;
they cannot establish which ASR backend produced their tokens. Separately retained
raw Rev evidence still has its own provider-aware request identity.

The current outer cache lookup also checks the FA worker's engine-version
partition. This conservative extra partition can cause an ASR miss when only FA
changes; it is not an attestation of the loaded ASR model revision. Retain model
provenance when comparing runs.

`--override-media-cache` bypasses lookups but still stores results for future use.

### FA cache

FA keeps two per-group layers under the same semantic group key:

- an immutable worker-protocol response, preferred on reads and replayed
  through the current Rust timing projection, inside an envelope that proves
  the requested engine, selected-worker version, semantic key, and word count;
- a derived `WordTiming` vector in a second versioned envelope, retained as a
  compatibility fallback when raw evidence is absent or refused. Historical
  bare vectors are no longer admitted because they do not identify the worker
  version or distinguish direct inference from fallback output.

The key is
`BLAKE3("{audio identity}|{start}|{end}|{words}|{gap healing}|{engine}")`, so a
different engine, a different audio window, or a different gap-healing policy
cannot collide. The `--override-media-cache` flag bypasses this cache.

Both caches use the same SQLite database (the analysis cache, see [Filesystem Paths](../reference/filesystem-paths.md#analysis-cache-sqlite)).

### What invalidates alignment cache?

| Change | UTR cache | FA cache |
|--------|-----------|----------|
| Edit transcript text | Stays cached (audio unchanged) | Groups with changed words re-run |
| Re-record audio | Re-runs (audio identity changed) | Re-runs (audio identity changed) |
| Change `--fa-engine` | May miss when the worker-version partition changes | Misses (engine is part of FA key) |
| Change `--lang` | Re-runs (lang is part of UTR key) | Re-runs (lang is part of FA key) |
| Change `--utr-engine` | Misses: ASR provider is part of the key | Reuse depends on resulting windows and words |
| Second run, nothing changed | Can reuse matching retained entries | Can reuse matching retained entries |

**Audio identity** is computed from the file's canonical path, modification time, and size, not
a content hash. Symlinks and alternate path spellings resolve to the same identity.
If you move or rename the audio file, the cache will miss even if the
content is identical. Conversely, overwriting a file in place with different content
will miss only if the modification time or size changes (which the OS updates on write).

### Why fallback traces are live-run evidence today

The admitted in-memory response records the requested engine, effective
engine, and fallback reason together, so a successful Wave2Vec-to-Whisper
fallback is fully observable in that run's trace. It does not cross the cache
boundary. The primary Wave request namespace contains the selected Wave worker
version but not the effective Whisper model version; replaying it later would
therefore claim more provenance than the stored facts prove.

Only direct, version-identified FA evidence can construct the replayable
typestate used by both raw and derived cache writers. Fallback timings remain
valid for the current output, but a later run repeats inference. This is a
deliberate correctness tradeoff until the request identity can carry both the
primary and effective fallback capabilities.

When you specifically need to reproduce a **new live worker pass**, bypass both
FA cache layers for that run:

```bash
batchalign3 align \
  --override-media-cache-tasks forced_alignment \
  --debug-dir /tmp/ba-debug \
  -o output/ \
  file.cha
```

That forces a new FA inference pass and makes the fallback event observable in
the trace payload if the condition is reproduced.

## Diagnostics and trace inspection

`align` now has two useful diagnostic layers:

1. **Better terminal/server errors**: per-group FA parse failures are surfaced
   with group index and audio window instead of being swallowed and only
   appearing later as a generic missing-timings failure.
2. **Optional trace capture**: `--debug-dir PATH` enables `debug_traces`, which
   stores FA timeline traces including fallback events.

```mermaid
flowchart TD
    run["Run align with --debug-dir"]
    mode{"direct or\nserver?"}
    direct["DirectHost\nexports debug-traces.json\nin job staging dir"]
    server["GET /jobs/{id}/traces\nreturns JobTraces JSON"]
    fa_trace["fa_timeline"]
    fallback["fallback_events[]\n(group_index, from_engine,\nto_engine, reason,\naudio_start_ms, audio_end_ms)"]

    run --> mode
    mode -->|direct| direct --> fa_trace
    mode -->|server| server --> fa_trace
    fa_trace --> fallback
```

For explicit server mode:

```bash
curl http://127.0.0.1:8001/jobs/JOB_ID/traces | python3 -m json.tool
```

For direct mode, the CLI prints stable debug handles after submission; the trace
payload is exported as `debug-traces.json` in the job staging directory.

This is especially useful for confirming:

- whether a file succeeded entirely from cache,
- whether a Wave2Vec group retried with Whisper,
- which group/audio window triggered the fallback,
- whether the final output was produced after fallback or after a plain cache hit.

## Known Pitfalls

### Audio timestamps past end of file

**Symptom if the empty-segment check is absent:** One or more files fail alignment with:

```text
FA processing failed: validation error: failed to parse worker protocol V2 FA
response for group N (Xms..Yms): worker protocol V2 forced-alignment request
failed with RuntimeFailure: RuntimeError: Calculated padded input size per
channel: (0). Kernel size: (10). Kernel size can't be greater than actual
input size
```

**Root cause:** The FA grouping algorithm assigns timestamps to groups using
utterance bullets already present in the CHAT file.  When the last few
utterances carry timestamps that extend past the actual end of the audio (a
common occurrence in PWA/aphasia corpora where utterance-end times are
hand-estimated or interpolated beyond the recording), the corresponding FA
group contains no real audio samples.

`ffmpeg` exits with code 0 in this case but writes an empty PCM file.  Without
an explicit check, the 0-frame PCM is passed to Wave2Vec, which crashes with the
PyTorch convolution error above because its kernel is larger than the input
tensor.

`extract_prepared_audio_segment_f32le` checks `frame_count == 0` after ffmpeg
and returns `PreparedArtifactErrorV2::EmptyAudioSegment` instead of a 0-frame
descriptor.  The transport layer (`infer_groups_v2`) catches this before
dispatching to the worker, skips the group with all-`None` timings, and logs:

```text
WARN fa_transport: FA group has no audio (segment past end of file); leaving words unaligned
     group=N start_ms=X end_ms=Y path=<audio file>
```

The affected words in the CHAT output have no bullets but the file is otherwise
complete.  This is the correct conservative outcome: the words exist in the
transcript but their timing is unknown.

**Affected corpora:** Typically aphasia/PWA corpora where post-session
annotation extends utterance times beyond the recorded audio.  The Croatian
PWA corpus (APROCSA) triggered this in production (job `6795bfbe-467`,
`1-ZH-76-1.cha`, group 62, 908154-909196 ms).

**Implementation:** `crates/batchalign/src/worker/artifacts_v2.rs`
(`EmptyAudioSegment` variant), `request_builder_v2.rs` (build-error mapping),
`fa/transport.rs` (skip logic in `infer_groups_v2`).

### Whisper pipeline chunking

The HuggingFace Whisper pipeline processes long audio in 25-second chunks with
3-second overlap (`chunk_length_s=25, stride_length_s=3`). The pipeline is
responsible for converting chunk-relative timestamps to absolute. However, this
conversion has been unreliable in some configurations, notably when `batch_size`
is passed to the pipeline constructor. The `batch_size` parameter was removed
from the constructor to avoid this issue (inference uses `batch_size=1`
regardless).

### Untimed input vs timed input

For **timed input** (production CHAT from CLAN): UTR is skipped, FA uses the
existing bullets directly. This path is well-tested and reliable.

For **untimed input** (raw transcripts without timing): UTR must first discover
where each utterance lives in the audio. This depends on Whisper ASR producing
correct absolute timestamps, which depends on the HF pipeline's chunking
working correctly. This path is more fragile.

## Warning reference

All `WARN` lines emitted to the server log during `align`.  None of these are
fatal by themselves; a `WARN` on a group or a post-processing pass does not
prevent output from being written.  Only file-level errors prevent output.

| Log source | Message pattern | When it fires | Action needed? |
|------------|----------------|--------------|---------------|
| `fa_pipeline` | `untimed utterances present but no UTR engine configured` | `--no-utr` and file has untimed utterances | No, proportional interpolation runs instead |
| `crates/batchalign/src/chat_ops/fa/utr.rs` | `no timing bullet and no estimate, skipping from FA grouping` | Utterance has no bullet and no timed neighbors for interpolation | Inspect file; utterance will be left unaligned |
| `crates/batchalign/src/chat_ops/fa/transport.rs` | `FA group has no audio (segment past end of file)` | Group's time window exceeds audio duration | Check audio file length and CHAT bullets |
| `crates/batchalign/src/chat_ops/fa/transport.rs` | `Wave2Vec FA hit recoverable target constraint; retrying group with Whisper FA` | Wave2Vec CTC overflow on one group | Informational; Whisper retry follows automatically |
| `crates/batchalign/src/chat_ops/fa/transport.rs` | `Whisper FA unavailable … leaving group words unaligned` | Wave2Vec overflow AND worker has no Whisper model | Informational; affected utterances lose word timing but file completes |
| `crates/batchalign/src/chat_ops/fa/transport.rs` | `FA error (raw) … category=validation` | Group failed worker inference for a non-CTC reason | Check server log for root cause; file may fail |
| `crates/batchalign/src/chat_ops/fa/mod.rs` | `FA cache batch lookup failed` | SQLite cache read error (non-fatal; misses treated as cache misses) | Check disk space and cache file permissions |
| `crates/batchalign/src/chat_ops/fa/mod.rs` | `Failed to deserialize cached FA timings` | Cache entry written by a different schema version | Wipe FA cache: `rm ~/Library/Caches/batchalign3/cache.db*` |
| `fa/mod.rs` | `Failed to cache FA result` | SQLite cache write error (non-fatal; inference result still used) | Check disk space |
| `fa/mod.rs` | `Post-validation warnings` | CHAT structural issues after injection | Review structured decision evidence and output |
| `fa/orchestrate.rs` | `monotonicity: strategy="end_clamped_coverage_only"\|"end_clamped_boundary_from_words"\|"end_clamped_interleaved_words"` | Same-speaker BY STREAM, i.e. skipping intervening other-speaker lines (default), or additionally every physically adjacent pair (`clamp-all-adjacent`); end overlap resolved from measured word timings, the last variant's reason carries how many words were cut | Review only `end_clamped_interleaved_words` (`needs_review=true`); the other two are automatic and informational |
| `fa/orchestrate.rs` | `monotonicity: strategy="start_stripped"` | Utterance start precedes previous accepted start, full timing stripped | Review utterance; structured decision has `needs_review=true` |
| `fa_pipeline` | `FA failed with untimed utterances; attempting fallback UTR` | FA error on file that still has untimed utterances | Informational; fallback UTR retry follows |
| `fa_pipeline` | `Fallback UTR recovered timing` | Fallback UTR succeeded before retry | Informational; timing injected, FA retry queued |

## Comparison with BA2

Understanding what BA2 did is useful context for why the fallback machinery
exists and why it is currently incomplete.

### BA2 forced alignment

BA2 was a pure-Python pipeline.  Forced alignment used **separate, mutually
exclusive engine classes**: `WhisperFAEngine` and `Wave2VecFAEngine` in
`batchalign/pipelines/fa/`.  The dispatcher chose one engine at pipeline
creation time based on the user's `--engine` flag.  **There was no fallback.**
If the chosen engine failed on a group, the file failed.

| Aspect | BA2 (Jan 2026) | BA3 (current) |
|--------|---------------|---------------|
| Default FA engine | Whisper (cross-attention DTW) | Wave2Vec (word-level start+end) |
| Wave2Vec available | Yes, via `--engine wave2vec_fa` | Yes, via `--fa-engine wav2vec` |
| Both models loaded simultaneously | No, one at a time by design | No, one at a time, same reason |
| Wave2Vec CTC overflow → fallback | **None**: file failed | Retry that group with Whisper |
| CTC overflow handling | User had to rerun with `--engine whisper_fa` | Automatic per-group retry |
| Silent file drop on CTC overflow | No, file errored visibly | No, affected groups leave words unaligned; file completes |
| Fallback telemetry | None | `FaFallbackEventTrace` in job traces |
| FA result caching | Per-file Python shelve | Per-group BLAKE3 SQLite cache |

### What BA3 improved

The Wave2Vec CTC fallback (added March 2026) handles three real-world failure
modes that BA2 users had to manually work around: groups with very long word
sequences, groups with characters that map to the CTC blank token (typically
hyphenated or cross-linguistic tokens), and groups shorter than ~25 ms that
crash Wave2Vec's convolutional feature extractor.

BA3's per-group Whisper retry means a single difficult utterance no longer
aborts an entire file, 90+% of a file that previously failed can now complete.

### What BA3 regressed

The fallback implementation has a critical gap: it assumes Whisper is always
available as a fallback model, but the worker loads only one FA model at
startup.  When Wave2Vec is the primary engine and CTC overflow occurs, the
Whisper fallback dispatch returns `ModelUnavailable`, which currently propagates
as a file-level failure.  BA2 users who selected Wave2Vec would see an explicit
error; BA3 users see a silent file drop, a regression in observability.

`is_whisper_model_unavailable()` in `fa/transport.rs` provides the
graceful-degradation behavior BA2 had via its explicit-error path: the affected
utterances lose word-level timing, but the file completes.

### Default engine and timing resolution

The two forced-alignment engines do not report the same thing, and the
difference decides what a `%wor` tier can express:

| engine | reports | word duration |
|---|---|---|
| Wave2Vec | word start AND end | measured by the engine |
| Whisper FA | token ONSETS only | derived from the next onset |

Because Whisper FA reports only when a word starts, a word's end has to be
inferred from its neighbour. Selecting it as the alignment engine without that
inference in place yields zero-duration words: `%wor` entries whose start and
end are equal. Wave2Vec is therefore the default, and Whisper FA is an explicit
opt-in via `--fa-engine whisper`.

Aligning the same material through both engines shows the difference directly:
Wave2Vec yields word durations in the 250 to 500 ms range conversational
English occupies, while Whisper FA yields 0 ms for every word unless the
onset-to-interval step runs.

The CTC fallback described above still catches genuine Wave2Vec refusals, which
are rare: a group is capped at 448 label characters, and MMS_FA offers one
20 ms frame per target, so reaching the ceiling takes more than 50 characters
per second against the 12 to 15 that speech produces.

## Design Rationale

We evaluated simpler alternatives:

- **Single-pass full-audio FA**: Would avoid the UTR dependency, but FA models
  degrade on long audio. Not viable for recordings over ~30 seconds.
- **Lightweight boundary detection (VAD + text alignment)**: Would be cheaper
  than full Whisper ASR for UTR, but more complex to implement and less
  accurate. The current design works correctly for production data (which is
  already timed), and UTR only runs on the uncommon untimed case.
- **Single Whisper pass for both timing and alignment**: Would halve compute
  time, but Whisper ASR output has different characteristics than Whisper FA
  output (ASR may hallucinate or miss words; FA is constrained to the known
  transcript).

The current two-step design (UTR for boundaries, FA for word timing) is the
standard approach in the field. It works correctly when both steps produce valid
absolute timestamps.

## Monotonicity Invariant

CHAT requires that utterance-level timing bullets increase monotonically through
the file (E362).  CLAN players seek into the audio by bullet position; a
regression means the player would seek backwards, which is undefined behavior.

### The fundamental limitation

The alignment engine is a **monotonic matcher**: it assumes that words appearing
later in the transcript also appear later in the audio.  This assumption holds
for most CHAT files, but breaks down when the transcript's text order diverges
from the audio's temporal order.

The most common cause is **overlapping speech** annotated with `&*` markers.
CHAT convention embeds one speaker's words inside another's utterance:

```text
*PAR:  I'm hoping to play here &*INV:yeah in a month or two .
```

In the audio, INV's "yeah" occurs *between* PAR's words.  But in the text,
INV's word is interleaved into PAR's utterance.  When the monotonic matcher
tries to align this against the ASR's flat temporal word sequence, it cannot
represent the crossing -- INV's word in the ASR sits between PAR's words that
are on the same text line.  The matcher must either skip the interleaved word
or lose sync with subsequent words.

When this happens across many utterances in a dense overlapping region, the
matcher loses sync entirely and leaves whole blocks of utterances untimed.

### What `align` does about it

UTR's DP alignment preserves LCS ordering: if CHAT word at reference
position *i* matches ASR word at payload position *p*, then reference position
*j > i* can only match payload position *q > p*.  When text and audio order
diverge, some utterances get no UTR timing.  FA's proportional estimation can
then assign a correct-but-earlier timestamp to those untimed utterances, breaking
monotonicity.

**Post-FA enforcement pass**: After all utterances have been
force-aligned, `enforce_monotonicity()` (at
`crates/batchalign/src/chat_ops/fa/orchestrate.rs:213`) walks
utterances in text order, tracking the last accepted start
timestamp. Any utterance whose start precedes the previous accepted
start has its timing stripped entirely, utterance bullet, inline
word bullets, and the `%wor` tier are all removed (the structured
`start_stripped` decision records the event; see the
[Monotonicity warnings table](#monotonicity-warnings)). The
utterance is left as plain untimed text, identical to how it would
look before alignment.

This is the conservative choice: no information is corrupted, only
alignment coverage is reduced. The correctly-timed surrounding
utterances retain their full word-level alignment.

**Pre-serialization validation gate**: As an additional safety net,
the post-validation walk at the end of
`process_one_fa_file()` (in
`crates/batchalign/src/runner/dispatch/fa_pipeline.rs`) checks the
full output against the talkbank-model validators (including E362
monotonicity) before serialization. The walk is warn-only, output
is always serialized so it can be inspected, but the warnings are
the early signal that the post-FA enforcement layer above missed
something.

### Real-world impact

In testing on hand-edited APROCSA aphasia protocol files with dense overlapping
speech, alignment loss reached 36.5% of utterances (232 of 636), clustered in
large contiguous blocks of up to 25 consecutive untimed utterances.  The
severity scales with the density of `&*` markers and the degree of
text/audio order divergence.

Files most affected:

- Aphasia protocols with frequent backchannels and completions from
  investigators and relatives
- Conversation analysis transcripts with dense turn overlap annotation
- Any file where a reviewer has restructured utterance order after ASR

Files least affected:

- Single-speaker or low-overlap recordings
- Transcripts in temporal order (most CLAN-produced files)
- Files re-run through `align` without structural editing

### What users should do

See the [troubleshooting guide](../user-guide/troubleshooting.md#some-utterances-lose-timing-after-align)
for practical guidance on identifying and working around untimed utterances.

### Roadmap

Two improvements are under consideration:

1. **Backbone extraction**: stripping `&*` segments before UTR alignment and
   interpolating their timing afterward.  This is cheap to implement and helps
   for moderate-overlap files, but does not solve the worst cases where
   transcript restructuring (not just `&*` markers) is the dominant divergence.

2. **Per-speaker UTR**: running ASR per speaker channel and matching each
   speaker's utterances independently.  This is the correct solution for
   heavily restructured transcripts but requires diarization infrastructure
   and is more complex to implement.

See [Overlap-Aware Alignment Improvements](../developer/backchannel-aware-alignment.md)
for the design proposals and honest impact estimates. A complementary
unimplemented "trouble-window" approach would preserve existing
timing during re-runs on hand-edited files; today, that workflow is
covered by `align --before` (incremental alignment that reuses
unchanged utterances).

## Known limitations

- **FA cannot recover utterances under ~200 ms of audio.** The Whisper
  and Wave2Vec FA models need a minimum number of frames to produce
  reliable timing. Utterances narrower than ~200 ms (single short
  backchannels, brief interjections) are left untimed by FA. The
  proportional-FA-estimation fallback may still place an estimated
  bullet on them when `total_audio_ms` is available; otherwise they
  remain untimed.
- **Audio timestamps past end of file leave words unaligned.** When an
  FA group's utterance timestamps extend past the actual end of the
  audio file (common in PWA corpora where end-times are
  hand-estimated), the prepared-audio extractor returns
  `EmptyAudioSegment` and the group is skipped with all-`None`
  timings. The rest of the file aligns normally. Workaround: trim the
  audio or correct end-times before re-running.
- **Same-speaker overlap windows can lose timing.** If two utterances
  from the same speaker overlap (E704), `strip_e704_same_speaker_overlaps()`
  drops the earlier one's timing rather than emit a contradiction.
  The later utterance keeps its timing.
