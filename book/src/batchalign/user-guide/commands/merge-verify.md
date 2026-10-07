# merge-verify

**Status:** Current
**Last updated:** 2026-10-05 07:30 EDT

Tier the machine-flagged utterance placements of a merged draft set
against engine verdicts, rewriting promoted flags into provenance
notes and exporting the rest as a review queue. Fully offline: no
daemon, no models, no audio; the heavy signals arrive as a verdicts
JSON produced separately.

A "merge" workflow places utterances from one source (e.g. ASR with
timings) into a manually transcribed session and flags every placement
it is not sure of with a `%com` comment. `merge-verify` is the second
pass: given per-line verification verdicts from three engines (forced
alignment confirmation, pitch banding, a machine ear), it applies a
calibrated composed rule to every flagged line:

- **Auto-trust:** the flag is REWRITTEN into a machine-verified
  provenance note carrying the three signals; never silently deleted.
  A human transcriber note sharing the same `%com` tier survives
  verbatim ahead of the note.
- **Review:** the flag is left unchanged and the line is exported to
  `review-queue.json` (directly consumable as a review-campaign spot
  scope).
- **Hold:** the flag is left unchanged and the line is not queued
  (categories the calibration says need session-level treatment).
- **Demote:** a previously unflagged line whose verdicts contradict
  its placement gains a review flag; text and timing are never moved.

The rewritten drafts preserve every typed main tier, including timing and
annotation structure. This pass edits only `%com` text segments or appends a
demotion comment. Existing comment bullets, pictures, continuation markers and
text outside the matching flag segment remain intact. The marker is matched
within a typed text segment, never across a bullet, picture or continuation.
As before, text from the marker to the end of that segment is the flag payload.

Input must pass complete named CHAT admission, including retained dependent
tiers; no regeneration exemption applies. Duplicate sessions (including
case-only and canonically equivalent Unicode aliases), duplicate utterance
verdicts (even identical duplicates), out-of-range ordinals and unsafe filename
stems refuse the run. Every session
is admitted and every transformed document is construction-validated before
the output directory is created. Semantic refusal writes nothing; filesystem
failure during persistence can still leave partial output.
The selected session set remains in memory as typed documents until these
checks finish; this is not a streaming transform.

Output uses Chatter's typed CHAT writer, so whitespace and attached punctuation
may normalize. Byte-identical untouched lines are not promised. The verdicts
and exported review-queue JSON formats and tier-selection policy are unchanged.
The queue is an interchange artifact, not a persistent review database.

### Developer admission boundary

The source-bound `PreparedSession` owns a `ValidChatFile` together with unique,
in-range verdicts. Applying it consumes that proof before editing. Only
`CheckedSession`, containing newly construction-admitted output, exposes the
writer. Neither raw CHAT nor a caller-selected validity claim grants write
permission; output admission does not reparse serialized CHAT. Source failures
retain Chatter's internal-failure distinction, and output-admission failure is
a tool failure rather than invalid input.

---

## Quick start

```bash
batchalign3 merge-verify \
    --draft merged-drafts/ \
    --verdicts verdicts.json \
    --out verified-drafts/ \
    --flag-prefix merge
```

Output: one rewritten `.cha` per input session plus
`review-queue.json` in `--out`, and a one-line tier summary on stderr.

## Options

| Flag | Meaning |
|------|---------|
| `--draft <DIR>` | Directory of merged `.cha` drafts (one per session) |
| `--verdicts <FILE>` | Engine verdicts JSON (shape below) |
| `--out <DIR>` | Output directory for rewritten drafts + queue |
| `--flag-prefix <P>` | The `%com` flag marker (default `verify`; merged corpora typically use `merge`) |

## Verdicts JSON

```json
{
 "sessions": [
  {
   "session": "S-001",
   "lines": [
    {"utterance_index": 12, "category": "other",
     "fa_mean_score": 0.52, "pitch": "child", "ear": "yes"}
   ]
  }
 ]
}
```

`utterance_index` is the 0-based ordinal over main-tier utterance
lines. `category` is the flag taxonomy the calibration was performed
over; `pitch` is `child` / `adult` / `ambiguous`; `ear` is `yes` /
`no`. Entries retain input-session and utterance order. Consumers can use
`fa_mean_score` to prioritize review worst-first; it is never a promote/demote
gate (calibration finding: disfluent child speech aligns poorly, and the
aligner happily aligns the wrong voice).

## The verify engines

The verdicts are produced by the three calibration-locked engines in
`batchalign.inference`: `fa_confirm` (windowed MMS_FA alignment
scoring), `pitch_band` (librosa pyin child/adult banding), and
`machine_ear` (local audio-LLM YES/NO). Each module documents its
calibration constants; changing any of them requires recalibration
against blind listening verdicts.
