# speaker-identify

**Status:** Current
**Last updated:** 2026-09-09 13:24 EDT

Score each timed utterance of a CHAT transcript against one or more voices you
enroll from the recording itself, and write the scores and verdicts beside the
transcript as evidence.

You supply one or more **enrollment spans**: stretches of the same recording
known to contain a single speaker alone. The command embeds each of them,
embeds every timed utterance, and reports how acoustically similar each
utterance is to each enrolled voice.

**It does not modify the transcript.** The output is a JSON evidence file. What
to do with a verdict, in particular whether to change a speaker code, is a
decision about your corpus's own conventions, and it stays yours.

---

## Quick start

```bash
# One investigator, enrolled from the opening span of the recording
batchalign3 speaker-identify session.cha \
  --enroll 1500-9000:INV \
  --threshold 0.62 \
  -o evidence/

# Two known voices, and only the utterances currently on *PAR0
batchalign3 speaker-identify corpus/ \
  --enroll 1500-9000:INV \
  --enroll 12000-20000:CHI \
  --tiers PAR0 \
  --threshold 0.62 \
  -o evidence/
```

For `session.cha` the artifact is `session_speaker_identity.json`.

---

## What an enrollment span is

A span of the recording, in milliseconds from its start, that you know holds
**one speaker, alone**. The same coordinates a CHAT timing bullet uses.

```text
--enroll <start_ms>-<end_ms>:<label>
```

The label names the voice (`INV`, `CHI`, `MOT`, anything without a colon, a
dash or a space). It is what appears in every score and verdict.

Choosing the span is the part that matters, and only you can do it:

- **Longer is better.** A few seconds of continuous speech is a much steadier
  acoustic identity than half a second. Spans below the model's own minimum
  (about 105 ms) are refused outright.
- **One voice only.** If anyone else speaks inside the span, or the span
  includes a long silence, the enrolled vector describes a mixture, and every
  score computed against it inherits that.
- **From this recording.** Enrolling from a different session's audio compares
  the two rooms and the two microphones as much as the two people.

Common source of a good span: an opening stretch where one person is speaking
alone before anyone else joins.

---

## Options

| Option | Default | Meaning |
| --- | --- | --- |
| `PATHS...` | | Input `.cha` transcripts and/or directories |
| `-o, --output DIR` | | Output directory for the evidence artifacts |
| `--enroll SPAN` | **required** | `<start_ms>-<end_ms>:<label>`. Repeat once per known voice |
| `--threshold F` | **required** | Similarity at or above which a voice counts as a match |
| `--tiers CODES` | all tiers | Speaker tiers to score, comma-separated or repeated |
| `--permutations N` | `1000` | Shuffled labellings drawn for each track contrast's p-value; a thousand resolve p to 0.001 |
| `--permutation-seed S` | `0` | Seed of that shuffle, so a run reproduces byte for byte. Both are recorded in the file |
| `--lang CODE` | `eng` | 3-letter ISO code, for worker-pool selection only; embedding is language-independent |

### Why `--threshold` has no default

How much acoustic agreement counts as "the same person" depends on the
recording, the microphone, how much enrollment audio you gave, and what a wrong
answer would cost you. Nothing in this tool knows any of that. A built-in
number would produce confident verdicts under a value nobody chose, and no
reader of the output could tell it had been chosen by accident. So you state
it, and the value you stated is written into the evidence beside the verdicts
it produced.

**How to pick one.** Run once on a session where you already know the answer
for a handful of utterances, read the `scores` in the output, and choose a
value that separates them. Cosine similarity runs from -1 to 1; values in the
0.5-0.75 region are where useful thresholds usually fall for this model, but
that is an observation about where to start looking, not a recommendation.

---

## Enrollment rules the command enforces

- **At least one** `--enroll`. With none there is nothing to identify against.
- **Labels are unique.** Each label names one voice.
- **Enrollments may not overlap.** Two spans claiming the same audio for two
  different single speakers cannot both be true, so the run is refused rather
  than producing two contaminated vectors that look like ordinary ones.

Spans that touch end-to-start (`0-5000` and `5000-9000`) do not overlap and are
fine.

---

## Output format

For `session.cha`, `session_speaker_identity.json`:

```json
{
  "provenance": {
    "schema_version": 2,
    "interpretation": "Scores are acoustic AGREEMENT with an enrolled span ...",
    "transcript": "session.cha",
    "media": "/corpus/media/session.mp3",
    "prepared_sample_rate_hz": 16000,
    "embedding_backend": "pyannote",
    "embedding_model_revision": "pyannote-embedding:0ae88dca...",
    "embedding_dimension": 256,
    "embedding_minimum_frames": 1680,
    "match_threshold": 0.62,
    "tiers": ["PAR0"],
    "enrollments": [
      { "label": "INV", "start_ms": 1500, "end_ms": 9000 }
    ],
    "permutation": { "seed": 0, "count": 1000 },
    "produced_by": "batchalign3 <build>"
  },
  "utterances": [
    {
      "utterance_index": 0,
      "line": 42,
      "speaker": "PAR0",
      "start_ms": 12000,
      "end_ms": 14500,
      "scores": [{ "label": "INV", "score": 0.81 }],
      "verdict": { "verdict": "matches", "label": "INV", "score": 0.81 }
    }
  ],
  "tracks": [
    { "kind": "voiced", "track": "PAR0", "lines_embedded": 212, "lines_refused": 3,
      "centroid": ["..."], "scores": [{ "label": "INV", "score": 0.71 }] }
  ],
  "track_contrasts": [
    { "kind": "one_track", "label": "INV", "track": "PAR0" }
  ]
}
```

Every utterance carries its similarity to **every** enrolled voice, not only
the winner, so you can try a different threshold without re-running the model.

### Tracks as voices

Beside the per-line verdicts, every speaker code among the scored tiers is
scored as **one voice**: the centroid of its lines' vectors, compared to each
enrolled voice. The mean of a track's line scores is not the similarity of
the track's voice, and the centroid is robust to the short lines that drag a
mean down. For each enrolled voice, `track_contrasts` then says how far the
best track stands out from the runner-up, with a permutation p-value in
place of a margin somebody chose: line-to-track membership is shuffled with
track sizes kept, and the p-value is how often a shuffle reaches the observed
margin. The seed and count are yours to set and are written into the file.
Field by field: the [evidence reference](../../reference/speaker-identity-evidence.md).

### The three verdicts

| `verdict` | Meaning |
| --- | --- |
| `matches` | Exactly one enrolled voice scored at or above the threshold, with nothing tied. Carries that `label` and `score`. |
| `no_match` | No single voice was established. `best` names every label that reached the highest score, and that score. |
| `unscored` | No similarity was computed, and `reason` says why. |

**A tie never matches.** If two enrolled voices reach the same highest score,
even above the threshold, the evidence does not say which speaker it is,
because it does not know. `best` then lists both.

### Why an utterance is `unscored`

| `reason` | Meaning |
| --- | --- |
| `too_short_for_embedding` | Shorter than the model can measure. Carries `frames` and `minimum_frames`. |
| `no_bullet` | The utterance has no timing, so there is no audio to embed. |
| `no_comparable_embedding` | It had timing, but no enrolled embedding could be compared with its embedding. |
| `audio_missing` | Its bullet names audio the recording does not contain. Carries the bullet and the recording's length. |
| `overlaps_enrollment` | It falls inside an enrolled span. Scoring it would compare that audio with a vector computed from it. |

Unscored utterances are **reported, never omitted**: a file that dropped them
would let you conclude the transcript has fewer utterances than it has.

---

## Scores are agreement, not accuracy

A score says how similar two stretches of audio are under one embedding model.
It does not say the speaker is who you think. The enrolled span is your own
claim about who is talking, and it carries your error rate: if the span holds
the wrong person, or two people, every score against it is confidently wrong in
a direction nothing in the output reveals.

Treat a high score as evidence to act on, not as a verdict to publish. The
evidence file repeats this in its own `interpretation` field, because the file
is what gets forwarded and this page is not.

---

## Gotchas

**The enrollment span is not scored against itself.** Utterances inside an
enrolled span come back `unscored` with `overlaps_enrollment`. That is
deliberate; a similarity there would measure the arithmetic.

**No Hugging Face account is needed.** Unlike `diarize`, this command loads only
the speaker-embedding model, which lives in a public, ungated repository. It
never builds the diarization pipeline, so it never reaches that pipeline's gated
calibration dependency.

**Enrollment spans must be inside the recording.** A span past the end fails the
run rather than scoring everything against a truncated vector; an *utterance*
past the end is `unscored`, because one bad bullet should not end a file.

**A transcript with no timings produces an all-`unscored` file.** That is the
correct answer, not a failure: run `align` first if you want timings.

---

## Related documentation

- [Speaker identity evidence](../../reference/speaker-identity-evidence.md), the artifact's field-by-field reference
- [diarize](diarize.md), which finds anonymous speaker turns without enrollment
- [align](align.md), which produces the timings this command reads
