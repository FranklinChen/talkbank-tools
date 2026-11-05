# Taiwanese (Min Nan) Language Support

**Status:** Current
**Last updated:** 2026-11-05 12:59 EST

Taiwanese (`nan`) is supported for transcription: `batchalign3 transcribe
--lang nan` turns a recording into a CHAT transcript in Taiwan-standard
traditional characters. Two recognition engines accept it, and they work
differently, so this page shows how to try each one and what to check.

Standalone `utseg`, `align` and `morphotag` do not support Taiwanese. See
[What is not supported](#what-is-not-supported).

## Quick start

```bash
# Qwen3: runs locally and needs no provider account. The documented
# Taiwanese engine; the first run downloads the model weights.
batchalign3 transcribe recording.wav -o out/ --lang nan --asr-engine qwen

# Tencent: cloud recognition, billed to the Tencent account configured in
# ~/.batchalign.ini (see "Tencent setup" below). Tencent's integrated
# diarization needs the number of speakers.
batchalign3 transcribe recording.wav -o out/ --lang nan --asr-engine tencent --num-speakers 2
```

To separate speakers with Qwen3, add `--diarization enabled`. That uses the
paid pyannoteAI service by default; add `--speaker-engine pyannote` to keep
all audio local. See [transcribe](../../user-guide/commands/transcribe.md).

## Trying both engines on the same recording

The engines fail in different ways, and neither replaces listening, so the
most useful first step is to run both on a short excerpt and compare:

```bash
batchalign3 transcribe excerpt.wav -o out-qwen/ --lang nan --asr-engine qwen
batchalign3 transcribe excerpt.wav -o out-tencent/ --lang nan --asr-engine tencent --num-speakers 2
```

Each output records which engine and model produced it in its provenance
comment, for example `asr=tencent ; asr_model=tencent-asr@16k_zh_en_2.0`, so
the two files stay distinguishable after they leave their folders. See
[Provenance](../../user-guide/provenance.md#transcribe).

When comparing, listen for:

| Check | Why |
| --- | --- |
| Taiwanese words written as Mandarin, or as sound-alike strings | Both engines are Chinese recognizers; how often each writes a Taiwanese word in a Taiwanese written form (for example 毋 rather than 不) varies by engine and passage. |
| Missing speech | An engine can drop backchannels, overlapping speech or whole stretches; compare where each output has utterances, not only what they say. |
| Speaker attribution | Tencent's speakers come from its own integrated diarization; Qwen3 assigns speakers only with `--diarization enabled`. Check turns by ear. |
| Word timings | Timings come from each route's own aligner; neither is a trained Taiwanese forced aligner. |

Treat every output as a draft for review. Language admission is not a
quality guarantee, and a few clips do not rank the engines in general.

## Tencent setup

Tencent recognizes Min Nan only with its Mandarin-English-dialect engines.
The account chooses its engine in `~/.batchalign.ini`, `[asr]` section, beside
its keys:

```ini
engine.tencent.model = 16k_zh_en_2.0
```

Engines that recognize Min Nan:

| Engine | Tier |
| --- | --- |
| `16k_zh_en_2.0` | Large Model 2.0 |
| `16k_zh_en_meeting` | Large Model 2.0 |
| `16k_zh_en` | Large Model 1.0 |
| `8k_zh_large` | Large Model 1.0 (telephone audio) |

The account must have bought recognition for that engine's tier; otherwise
Tencent refuses the request and BA3 reports the engine and tier the account
needs. Tencent charges the account whose keys are configured. The full engine
table and billing behavior are in
[Tencent (cloud ASR)](../language-code-resolution.md#tencent-cloud-asr) and
[Cantonese processing](../../user-guide/cantonese-processing.md).

## What both engines share

Whichever engine recognizes the speech, the `nan` route then applies the same
steps (details in
[Taiwanese transcription](../language-code-resolution.md#taiwanese-transcription-nan)):

- **Script.** Output is converted character for character to Taiwan-standard
  traditional characters with OpenCC `s2tw`, keeping each word's timing.
- **Words.** Runs of one-character Han words are grouped into words by the
  taibun dictionary tokeniser; the provenance comment records it as
  `word_seg=taibun@<version>`.
- **Language.** The CHAT header keeps `@Languages: nan`; nothing relabels
  the speech as Mandarin.

## What is not supported

- **`utseg`, `align`, `morphotag`** on Taiwanese transcripts: there is no
  Taiwanese utterance segmenter, trained aligner or morphological model in
  BA3. Transcription uses the recognizer's own punctuation and speaker
  boundaries instead.
- **Orthography beyond script conversion.** BA3 does not choose between
  competing Taiwanese written forms or add romanization; it keeps the
  characters the recognizer wrote.
