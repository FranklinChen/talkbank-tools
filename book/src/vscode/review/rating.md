# Rating Utterances

**Status:** Current
**Last updated:** 2026-03-30 13:40 EDT

## Rating values

| Rating | Key | Marker | Meaning |
|--------|-----|--------|---------|
| Good | `1` | `[ok]` | Bullet timing is correct — audio matches text |
| Early | `2` | `[early]` | Bullet starts before speech actually begins |
| Late | `3` | `[late]` | Bullet starts after speech actually begins |
| Wrong | `4` | `[wrong]` | Completely wrong — hearing different content |
| Skip | `5` | (unchanged) | Can't tell or don't want to rate — stays `[?]` |

## What "Good" means

The bullet points to the audio location where the words in the utterance
are actually spoken. A few milliseconds of imprecision is fine — "Good"
means "I can hear the right words when I play from this bullet."

## What "Early" means

The bullet plays audio from before the utterance starts. You hear silence,
the previous speaker's words, or background noise before the current
utterance's words begin.

**This is the most common issue** with boundary-averaged bullets — the
algorithm split the difference between two overlapping utterances, but split
it in the wrong place.

## What "Late" means

The bullet plays audio that starts partway through the utterance — you
miss the first word or two. The speaker is already talking when playback
begins.

## What "Wrong" means

The bullet is wildly off — you hear a completely different part of the
conversation, a different speaker, or content that doesn't match the
text at all. This typically happens when the alignment algorithm lost sync.

## Adding notes

After pressing a rating key, the `%xrev` tier updates automatically. If
you want to add a note, edit the line directly in the editor:

```text
%xrev:	[early] starts about 200ms before speech, consistent with other INV lines
```

Notes don't affect the rating — they're free text after the marker. The
harvesting script collects them for manual analysis.

### Useful note patterns

- **Timing offset**: "early by ~300ms" or "late by about half a second"
- **Speaker pattern**: "all INV backchannels seem early"
- **Overlap pattern**: "this is where two speakers talk over each other"
- **Algorithm feedback**: "LIS shouldn't have stripped this — timing was correct"
- **Correction hint**: "should start at roughly 12300ms, not 12095ms"
