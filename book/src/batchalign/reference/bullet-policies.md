# Bullet and word-timing policies

**Status:** Current source policy
**Last updated:** {{git-dates:page}}

This is the policy index for alignment output. It separates CHAT validity,
word timing, and main-tier projection. A well-formed interval is not proof of
acoustic accuracy, and preserving a supplied interval does not certify it.
For algorithm details see [Forced Alignment](forced-alignment.md); for the
word membership and splitting contracts see [%wor Tier](wor-tier.md).

## Word bullets are not utterance bullets

A bullet is an unsigned millisecond pair between U+0015 delimiters.
Main-tier bullets describe utterance spans; `%wor` bullets attach to individual
word slots. There is no final tier-level `%wor` bullet.

BA3 admits a produced `WordTiming` only when `start < end`. Missing,
zero-duration, reversed, or wholly clipped-away timing leaves the word untimed;
the word itself remains in the transcript. Postprocessing revalidates the
interval before writing it. It does not turn an absent extent into a 1 ms
success. Incomplete required coverage normally produces diagnosed output and
CLI exit 7, not fabricated timing. Outstanding regeneration obligations or
other invalid output can still prevent writing; see [align](../user-guide/commands/align.md).

**Positive does not mean disjoint or ordered.** A word at `100_300` followed
by one at `200_400` is possible, as is a later word with an earlier start.
`WordTiming` validates one interval, not adjacency across a sequence.
`--pauses` bypasses gap healing; neither it nor the default healing pass is a
general proof of disjoint word intervals. Main-tier monotonicity and
end-overlap resolution likewise do not impose such a proof within `%wor`.
No new within-tier overlap prohibition is implied by the zero-duration rule.

Chatter's current source validation rejects zero-duration `%wor` bullets with
E362 (as well as reversed ones); older dependency pins may still admit zero
duration at that boundary. BA3's positive-output rule is independent of that
dependency rollout. Partial untimed words, positive overlapping word intervals,
and positive out-of-order word intervals remain distinct from invalid duration.

## Independent alignment policies

| Control | Default | What it controls |
| --- | --- | --- |
| `--pauses` | Off | Off: bounded gap healing. On: preserve engine-reported ends rather than heal gaps. For Whisper it also selects the historical character-spaced input mode. |
| `--existing-wor-boundaries` | `preserve` | How fresh word evidence interacts with an existing `%wor` run's main-tier boundary; `rebuild-from-evidence` instead reconstructs coverage from the admitted word hull. |
| `--main-bullets` | `derive` | Whether supplied main bullets, and optionally supplied absences, are immutable. |
| `--end-overlap-policy` | `preserve-cross-speaker` | Whether utterance end-overlap resolution also acts on physically adjacent different-speaker turns. |
| `--bullet-repair` | Off | Optional repair before final monotonicity, subject to the same kept-bullet authority. |
| `--wor` / `--nowor` | `--wor` | Whether to generate the word-timing sidecar; not permission to invent timing or ignore an existing sidecar. |

The three boundary projection options do not change raw FA cache identity.
They are not interchangeable with `--pauses`, which also affects Whisper input.
`exact` additionally rejects enabled UTR before execution: it requires
`--no-utr`. A projection flag alone neither authorizes fresh inference nor
guarantees that an all-reusable input reaches raw-evidence replay.

### Where word ends come from

Measured-end engines supply both boundaries. Whisper FA supplies onsets, so
BA3 derives each end from the next onset even without `--pauses`. Coincident
onsets cannot supply a positive interval for the earlier word. The last-word
fallback is named `LAST_WORD_FALLBACK_MS` (500 ms), bounded by available audio;
postprocessing may extend to a retained transcript boundary, never merely to
a provisional UTR hint. Such derived timing is not a measured end.

Default gap healing is bounded by plausibility checks, with special handling
for fillers, compounds, and very short words. It is not an unconditional
instruction to join every pair of words. `--pauses` does not disable boundary
clamping, kept-bullet authority, or utterance-level resolution.

### Prior boundaries and supplied main bullets

Under compatibility `preserve`, the ordinary rerun clamp uses a retained
`Authoritative` boundary together with existing `%wor`. A runtime UTR hint
does not qualify. These are implementation categories, not proof of human
annotation or correct acoustics: serialization loses `BulletSource`, and a
stale `%wor` can survive an edit. `rebuild-from-evidence` changes this projection,
not the quality of the evidence.

- `derive`: main bullets may widen, shrink, or disappear. Compatibility
  projection may retain wider inherited coverage; it is not always the exact
  minimum/maximum word hull.
- `keep`: every supplied main bullet is restored unchanged. Words straddling
  its edge are clipped; words outside it or left with no extent become untimed.
  Initially unbulleted utterances may acquire timing.
- `exact`: preserves both supplied bullets and supplied absences. Words in an
  initially unbulleted utterance remain in alignment context but lose their
  timing at projection; the utterance stays unbulleted. Requires `--no-utr`.

Kept authority is captured at admission, before provisional grouping edits.
It takes precedence over rebuilding, repair, and overlap projection. A kept
conflict is recorded rather than silently moving the supplied boundary, and
final validation may refuse output. The internal ability to represent an empty
kept extent is not a promise that complete CHAT admission accepts it.
An off-record `[+ diary]` utterance is not aligned; only `keep`/`exact` may
restore its supplied main bullet, without manufacturing word timing.

### Utterance overlap and start order

Under either end-overlap policy, same-speaker resolution follows each speaker's
own sequence, including A-B-A turns. It first uses measured word-hull edges to
remove inherited coverage overlap without moving words. If the word evidence
itself conflicts, it clips the conflicting extent and word timings together
and records review evidence. A removed word extent is retained in diagnostic
evidence, not disguised as zero duration.

`preserve-cross-speaker` leaves different-speaker end overlap alone.
`clamp-all-adjacent` additionally applies resolution to every physically
adjacent pair regardless of speaker. Neither disables start-regression
enforcement. Preserving end overlap does not promise acceptance of globally
backward main-tier starts: Chatter's complete validator still checks those
with E362, separately from per-speaker E701 and the E704 self-overlap tolerance
of 500 ms. BA3's projection is not simply that tolerance copied into an aligner.

Order is enforced through the projection types: impose kept authority, perform
optional repair, resolve monotonicity/overlap, then write `%wor` from the
resolved state. Fresh, incremental, and reusable paths share this ordering.
`--nowor` suppresses generation, not validation or the timing policies. An
existing `%wor` may still need in-place fitting to a kept boundary; it is not
an unconditional instruction to erase every retained sidecar.

## Admission, linkage, and limits of the evidence

Chatter checks timing/media consistency: timing without `@Media` is E752;
`unlinked` with timing is E552; declared linkage without timing is E544.
Actual word bullets count as timing even if no main bullet exists, whereas an
untimed `%wor` tier does not. Align's source-bound admission can replace bad
word tiers and carry a timing-regeneration obligation, but does not exempt
retained source faults. `NoAlign` is a valid-input no-op, not a validity bypass.

Reusable word timing needs the canonical word membership, matching counts,
lexical corroboration, and positive complete timing required by its consumer.
Those checks do not certify acoustic accuracy, immutable origin, word
non-overlap, or main-bullet containment. Likewise, splitting a timed utterance
uses complete child word hulls or the documented parent-only state; it never
assigns a parent's entire span to one of several children as a guess.

## Implementation owners

- `chat_ops/fa/mod.rs`: `WordTiming`, `WordEndPolicy`, the boundary policy
  enums, and the single-owner defaults.
- `chat_ops/fa/postprocess.rs`: bounded healing and positive-span settlement.
- `chat_ops/fa/main_bullets.rs`: captured kept authority and verified restoration.
- `chat_ops/fa/orchestrate.rs`: ordered finalization and `%wor` emission.
- `fa/completion.rs`: diagnosed coverage, off-record exclusions, and output admission.

Paths above are relative to `crates/batchalign/src/`. This chapter describes
source behavior, not a claim that a particular installed server has that build.
