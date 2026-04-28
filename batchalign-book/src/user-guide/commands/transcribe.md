# transcribe

**Status:** Current
**Last updated:** 2026-04-17 18:34 EDT

Create a new CHAT transcript from audio files using automatic speech
recognition (ASR). Produces `.cha` files alongside or in a separate output
directory. **Never modifies the input audio.**

---

## Quick start

```bash
# Transcribe a single recording, output alongside input
batchalign3 transcribe interview.wav

# Transcribe all audio files in a directory
batchalign3 transcribe recordings/ -o transcripts/ --lang eng

# Auto-detect language (useful for bilingual/code-switched audio)
batchalign3 transcribe bilingual.wav -o out/ --lang auto

# Transcribe with dedicated speaker diarization (Whisper, multiple speakers)
batchalign3 transcribe interview.wav -o out/ --asr-engine whisper --diarization enabled

# Use the fleet server
batchalign3 --server http://net:8001 transcribe corpus/ -o out/ --lang eng
```

If you enable dedicated speaker diarization for the first time on a machine,
authenticate Hugging Face first:

```bash
hf auth login
```

You may also need to accept the diarization model's terms in the browser once
before retrying the command.

At the moment this uses **ambient Hugging Face auth**, not the Rust-owned
provider-credential path used for Rev.AI. In practice that means the machine
running `batchalign3` must already have a valid `hf auth login` cache/keychain
entry, or an `HF_TOKEN` exported in that process environment.

---

## Pipeline

```mermaid
flowchart TD
    start([transcribe invoked]) --> resolve[Resolve audio file]
    resolve --> ensure_wav[ensure_wav — convert if needed]

    ensure_wav --> diarize_check{--diarization?}
    diarize_check -->|"enabled"| transcribe_s["Command: transcribe_s\nASR + dedicated speaker relabeling\nRev or Whisper"]
    diarize_check -->|"auto/disabled\n(default)"| transcribe_m["Command: transcribe\nDefault path\nRev labels used directly when present"]

    transcribe_s --> engine_check
    transcribe_m --> engine_check

    engine_check{--asr-engine?}
    engine_check -->|whisper| whisper[Whisper local ASR]
    engine_check -->|whisper_hub| whisper_hub["HF Whisper fine-tune\n(per-language model_id)"]
    engine_check -->|rev| rev_preflight["Rev.AI preflight\nPre-submit audio in parallel\nskip_postprocessing=true for en/fr"]
    engine_check -->|whisperx| whisperx[WhisperX ASR]
    engine_check -->|whisper_oai| whisper_oai[OpenAI Whisper ASR]

    rev_preflight --> rev_poll[Poll Rev.AI for results]
    rev_poll --> asr_tokens

    whisper --> asr_tokens
    whisper_hub --> asr_tokens
    whisperx --> asr_tokens
    whisper_oai --> asr_tokens

    asr_tokens["Raw ASR tokens\nword + start_s + end_s + optional speaker + confidence"]
    asr_tokens --> convert["convert_asr_response()\nGroups tokens by speaker label"]
    convert --> dedicated_check{"--diarization enabled?"}
    dedicated_check -->|No| postprocess
    dedicated_check -->|Yes| speaker_v2["execute_v2(task=speaker)\nprepared audio → diarization segments\nPost-ASR relabeling via Pyannote or NeMo"]
    speaker_v2 --> postprocess

    subgraph postprocess ["Rust post-processing: process_raw_asr()"]
        direction TB
        p1[1. Compound merging] --> p2[2. Timed word extraction\nseconds → milliseconds]
        p2 --> p3[3. Multi-word splitting\ntimestamp interpolation]
        p3 --> p4[4. Number expansion\ndigits → word form]
        p4 --> p4check{lang=yue?}
        p4check -->|Yes| p4b["4b. Cantonese normalization\nzhconv + domain replacements"]
        p4check -->|No| p5
        p4b --> p5[5. Long-turn splitting\nchunk at >300 words]
        p5 --> p6[6. Retokenization\npunctuation-based utterance splitting]
    end

    postprocess --> build_chat["build_chat → ChatFile AST\nHeaders, participants, %wor tiers"]
    build_chat --> speaker_apply{Dedicated speaker\nsegments present?}
    speaker_apply -->|Yes| reassign["reassign_speakers()\nRewrite utterance speakers +\n@Participants + @ID headers"]
    speaker_apply -->|No| utseg_check{"with_utseg?\ndefault: true"}
    reassign --> utseg_check

    utseg_check -->|Yes| run_utseg[process_utseg\nBERT-based re-segmentation]
    utseg_check -->|No| mor_check{"with_morphosyntax?\ndefault: false"}

    run_utseg --> mor_check
    mor_check -->|Yes| run_mor[process_morphosyntax\nPOS + lemma + depparse]
    mor_check -->|No| merge_check

    run_mor --> merge_check{--merge-abbrev?}
    merge_check -->|Yes| merge[merge_abbreviations]
    merge_check -->|No| output

    merge --> output[Serialize → .cha output]
    output --> done([Output .cha file])
```

---

## Options

### Path options

| Option | Meaning |
| --- | --- |
| `PATHS...` | Audio files (`.mp3`, `.mp4`, `.wav`) or directories |
| `-o`, `--output DIR` | Output directory for new `.cha` files |
| `--file-list FILE` | Read input paths from a text file |
| `--in-place` | Write `.cha` files alongside the audio inputs |

### ASR and language options

| Option | Default | Meaning |
| --- | --- | --- |
| `--lang CODE` | `eng` | 3-letter ISO language code, or `auto` for language auto-detection |
| `--asr-engine {rev,whisper,whisper_hub,whisperx,whisper-oai}` | `rev` | ASR engine. See [`whisper-hub-asr.md`](../../reference/whisper-hub-asr.md) for `whisper_hub` (HuggingFace community fine-tunes). |
| `--asr-engine-custom NAME` | — | Override ASR engine by name (e.g. `tencent`, `funaudio`) |
| `-n`, `--num-speakers N` | `2` | Expected number of speakers |
| `--diarization {auto,enabled,disabled}` | `auto` | Dedicated Pyannote speaker diarization stage (`auto` = disabled) |
| `--wor` / `--nowor` | `--nowor` | Include or suppress the `%wor` word-timing tier |
| `--merge-abbrev` | off | Merge abbreviations in the output |

---

## Speaker labeling: Rev.AI vs Whisper

This is the most common source of confusion with `transcribe`.

**Rev.AI (default engine):** Rev.AI returns speaker labels as part of its ASR
response. These labels are **always** applied — you get multi-speaker output
without passing `--diarization enabled`. Passing `--diarization enabled`
explicitly runs an additional Pyannote post-ASR relabeling stage on top of the
Rev labels, matching BA2's audited `transcribe_s` pipeline behavior.

**Whisper-based engines** (`--asr-engine whisper`, `whisperx`, `whisper-oai`):
these engines produce no speaker labels. Without `--diarization enabled`, all
utterances are attributed to a single default speaker. Pass
`--diarization enabled` to run a dedicated Pyannote stage that assigns speaker
identities.

**`--diarization auto`** (the default) = disabled dedicated stage. Equivalent
to BA2's `--nodiarize`. The BA2 help text claiming Rev.AI ignored `--diarize`
was stale — the actual BA2 `transcribe_s` pipeline wiring ran the dedicated
stage.

---

## `--lang auto` behavior

With `--asr-engine whisper`, `--lang auto` omits the language parameter from
Whisper's generation kwargs, letting the model detect the spoken language from
the audio. The multilingual `openai/whisper-large-v3` model is always used
with `auto` — language-specific fine-tuned models are bypassed because they
are trained for a single language.

With Rev.AI, `--lang auto` submits a true auto-language request to the Rev.AI
API. Note that Rev.AI auto-detect and explicit `--lang eng` can produce
different punctuation, diarization, and turn boundaries from the provider.

---

## What gets created

A new `.cha` file per audio input (audio extension replaced: `foo.wav` →
`foo.cha`). Contains:

- `@Comment` with Batchalign version and ASR engine name
- `@Languages`, `@Participants`, `@ID` headers
- Utterance lines with timing bullets
- `%wor` tier (if `--wor` is set)

No `%mor` or `%gra` tiers are created by `transcribe`. Run `morphotag`
afterwards if morphosyntactic analysis is needed.

---

## Gotchas

**Rev.AI `skip_postprocessing`:** For English (`eng`) and French (`fra`),
Rev.AI is called with `skip_postprocessing=true`. This lets BA3's own BERT
utterance segmentation model handle sentence boundaries from raw ASR output,
rather than relying on Rev.AI's punctuation (which produces large monologue
blobs). For all other languages, Rev.AI's post-processing is applied.

**`--server` requires server-visible audio.** With `--server`, the server
resolves audio paths on its own filesystem. Paths valid on your machine must
also be reachable from the server, or you must use a shared media mount.

**Memory on developer machines.** Each Whisper model instance uses 2–15 GB.
For large corpus runs (>5 files or >1 GB audio), use `net` (256 GB M3 Ultra)
rather than a developer machine. Always pass `--workers 1` for local smoke
tests.

---

## Related documentation

- [Rev.AI Integration](../rev-ai.md) — API key setup, engine behavior
- [Cantonese Engines](../hk-cantonese-engines.md) — Tencent, Aliyun, FunASR engines
- [Utterance Segmentation](../../reference/utterance-segmentation.md) — post-ASR BERT utseg
- [Command I/O: transcribe](../../reference/command-io.md#2-transcribe) — I/O patterns and mutation behavior
- [Command Flowcharts: transcribe](../../architecture/command-flowcharts.md#transcribe) — full architecture flowchart
- [ASR Token Pipeline](../../architecture/asr-token-pipeline.md) — ASR post-processing details
