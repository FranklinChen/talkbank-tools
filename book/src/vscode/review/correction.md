# Interactive Bullet Correction

**Status:** Current
**Last updated:** 2026-03-30 13:40 EDT

> **Status:** This feature is planned but not yet implemented. This chapter
> documents the design for when it ships.

Beyond rating utterances as good/early/late/wrong, the extension will
support **interactive correction** — clicking on the waveform to set the
correct bullet boundary.

## Why correction is more valuable than rating

- Rating "early" tells us the direction but not the magnitude
- A corrected bullet tells us "was 12095ms, should be 12295ms" — a precise
  measurement of how far off the algorithm was
- Corrected bullets become ground truth for tuning algorithm thresholds
- The `%xalign` tier records both values: `machine=12095_12335 corrected=12295_12335`

## Planned workflow

1. During Review Mode, navigate to a flagged utterance
2. **Click on the waveform** where speech actually begins — the bullet
   start time updates
3. **Shift-click** to set the end time
4. **Preview** the adjusted region to verify
5. **Accept** (Enter) — the correction is saved and `%xrev` set to `[corrected]`
6. **Revert** (Escape) — restore the original bullet

## What gets recorded

When you correct a bullet, the `%xalign` tier is updated with both the
machine's original timing and your correction:

```text
%xalign:	boundary_averaged overlap=155ms machine=12095_12335 corrected=12295_12335 delta_start=+200ms
%xrev:	[corrected]
```

The `delta_start=+200ms` field tells us exactly how much the algorithm was
off, in which direction. Aggregated across many corrections, this data
reveals systematic biases that we can fix in the algorithm.

## Stamping untimed utterances

For utterances where align stripped timing entirely (LIS removal), you can
stamp a new bullet from scratch using the same F4 mechanism as Transcription
Mode:

1. Play the audio and listen for the utterance
2. Press F4 at the start to stamp the begin time
3. Press F4 again at the end to stamp the end time
4. The `%xrev` tier is set to `[stamped]`

```text
%xalign:	lis_removal same_speaker_non_monotonic machine=none stamped=11940_27820
%xrev:	[stamped] timing was actually correct
```
