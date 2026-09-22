# Cantonese Engines

**Status:** Current
**Last updated:** 2026-09-22 14:20 EDT

Batchalign includes alternative ASR and forced alignment engines for Cantonese.
These are built-in modules shipped in the base package, selected with the
per-category engine flags below.

## Available Engines

| Engine | Task | Description |
|--------|------|-------------|
| `qwen` | ASR | Qwen3-ASR-1.7B local model (Alibaba). Open-weight Cantonese-capable ASR; external evaluations report competitive CER on per-utterance child speech. Downloads ~4.1 GB of weights on first use, plus ~1.8 GB for the forced aligner it is paired with; no cloud credentials. |
| `tencent` | ASR | Tencent Cloud speech recognition with speaker diarization. |
| `aliyun` | ASR | Alibaba Cloud NLS real-time speech recognition (Cantonese only). |
| `funaudio` | ASR | FunASR/SenseVoice local model (no cloud credentials needed). |
| `cantonese` | FA | Cantonese forced alignment with jyutping preprocessing. |

## Installation

The standard install (see the [Installation guide](installation.md))
already includes these engines.

For a source checkout, the standard build (`cargo build -p batchalign`
plus `uv run maturin develop` for the PyO3 bridge) already includes
these engines. There are no Cantonese-specific extras to install.

## Usage

Each engine category has one flag, and `--help` lists every value it
accepts: `--asr-engine` for transcription, `--fa-engine` for forced
alignment, `--utr-engine` for utterance timing recovery.

`--engine-overrides` is for per-engine PARAMETERS, such as which
checkpoint to load. It is no longer how an engine is chosen; earlier
versions of this page used it that way, which is why the Cantonese
engines were hard to find.

```bash
# Recommended: Qwen3-ASR (local, no credentials)
batchalign3 transcribe input/ -o output/ --lang yue \
  --asr-engine qwen

# Pick the 0.6B model for faster inference on tight hardware
batchalign3 transcribe input/ -o output/ --lang yue \
  --asr-engine qwen --engine-overrides '{"qwen_model": "Qwen/Qwen3-ASR-0.6B-hf"}'

# Transcribe with Tencent Cloud ASR (cloud, needs CAM credentials)
batchalign3 transcribe input/ -o output/ --lang yue \
  --asr-engine tencent

# Transcribe with FunASR (local, no credentials)
batchalign3 transcribe input/ -o output/ --lang yue \
  --asr-engine funaudio

# Benchmark against a gold CHAT companion in the input directory
batchalign3 benchmark input/ --output output/ --lang yue --num-speakers 1 \
  --asr-engine qwen

# Force align with Cantonese FA engine
batchalign3 align input/ -o output/ --lang yue \
  --fa-engine cantonese

# Align a transcript whose utterances have no timings yet, so utterance
# timing recovery has to run. Rev.AI, the default UTR engine, does NOT
# support Cantonese; pass one that does.
batchalign3 align input/ -o output/ --lang yue --utr \
  --fa-engine cantonese --utr-engine tencent
```

### Utterance timing recovery on Cantonese

`--utr` recovers timings for utterances that have none. Its default
engine is Rev.AI, which has no Cantonese support, so a Cantonese file
needing UTR fails validation unless another engine is named:

| `--utr-engine` | Cantonese | Notes |
|---|---|---|
| `rev` (default) | no | Rev.AI has no `yue` model. |
| `whisper` | yes | Local, no credentials. |
| `tencent` | yes | Cloud, needs CAM credentials. Chinese variants only. |

The error names the engines that would work, so there is nothing to
guess. If none does, `--no-utr` skips the pass and alignment proceeds
with interpolated timings.

## Credential Configuration

Cloud engines (Tencent, Aliyun) require API credentials in
`~/.batchalign.ini`:

### Tencent Cloud

```ini
[asr]
engine.tencent.id = <secret-id>
engine.tencent.key = <secret-key>
engine.tencent.region = ap-guangzhou
engine.tencent.bucket = <cos-bucket-name>
```

### Aliyun NLS

```ini
[asr]
engine.aliyun.ak_id = <access-key-id>
engine.aliyun.ak_secret = <access-key-secret>
engine.aliyun.ak_appkey = <appkey>
```

Missing or empty credentials raise `ConfigError` with a clear message
indicating which keys are needed.

### Qwen3-ASR

Qwen3-ASR has no cloud credentials, it is a local HuggingFace model
downloaded on first use (~3.4 GB for the default `Qwen/Qwen3-ASR-1.7B-hf`).
Two `--engine-overrides` knobs are recognized:

- `qwen_model`: override the HuggingFace model id. The 1.7B default
  is the recommended-quality variant; pass `Qwen/Qwen3-ASR-0.6B-hf` for
  faster inference at some accuracy cost.
- `qwen_device`: `"cpu"` (default), `"cuda"`, or `"mps"`. The
  Apple Silicon fleet defaults to CPU because empirical testing
  found MPS inference produced degraded output on the 1.7B model
  as of 2026-05-26.

## Cantonese Text Normalization

All Cantonese ASR output is automatically normalized from simplified/mixed
Chinese to Traditional Chinese. This normalization:

1. **Simplified → Traditional** via the `ferrous-opencc` Rust engine
   (embedded OpenCC `S2hk` conversion tables)
2. **Domain-specific corrections** via a 31-entry replacement table for
   Cantonese character variants (e.g., 系→係, 呀→啊, 中意→鍾意)

It runs in the Rust server during ASR post-processing for `lang=yue`, once per
speaker monologue, before anything splits the words. No configuration, no
additional Python dependencies (like OpenCC), and nothing engine-specific: the
ASR engines report their own characters and the server normalizes them, so the
same recording reads the same whether it was transcribed by FunASR, Tencent,
Aliyun, Qwen or Whisper. Until 2026-09-16 some engines normalized their own
output and others did not, which is why older Cantonese transcripts from Qwen
and Whisper may still carry simplified characters; re-running them corrects it.

Two consequences worth knowing:

- **Phrases are converted, not characters in isolation.** The whole monologue is
  converted in one pass, so a replacement that spans two words (`真` `系` to
  `真` `係`) still applies even though the engines report one word per
  character.
- **A run that could not be normalized fails the file, with both character
  counts in the message.** Handing each word back its own characters requires
  the count to be unchanged; a transcript whose words silently moved onto
  different characters would carry wrong timings and look correct. No input
  measured so far does this (191,125 strings checked, none changed length), so
  this is a guard rather than something to expect.

## An empty transcript is a failure, not a result

A Cantonese run that recognizes nothing now fails and names the stage that came
up empty, instead of completing with a transcript of headers and no utterances.
Three stages can each end with no words, and they mean different things:

- **the ASR engine returned no words at all.** Nothing downstream ran, so
  normalization is not implicated: it only executes when there are words to
  normalize. Check that the engine has a Cantonese model and that `--lang yue`
  reached it.
- **post-processing kept no utterance** from the words the engine returned.
- **CHAT assembly kept no utterance line.** The words were all terminators or
  separators. This is the shape a provider produces when it returns only CJK
  sentence marks: each `。` becomes a bare `.`, and an utterance of nothing but
  a terminator has no content to write.

Until 2026-09-16 the first of these produced a completed job and an empty file.

## What a cloud provider does not send

Tencent and Aliyun both document every field of a result as nullable, and their
SDKs leave an attribute as `None` when the service omits it. Batchalign treats
those absences as facts rather than as zeros:

- **A word the provider did not time arrives untimed.** It appears in the
  transcript with no timing bullet of its own rather than at the start of its
  segment. Previously a missing offset read as `0`, which is a real time, so
  the word claimed a position the provider never gave it.
- **A Tencent segment with no `StartMs` leaves every word in it untimed**,
  because the per-word offsets are relative to that start and locate nothing
  without it. Previously the whole segment was placed at the beginning of the
  recording.
- **A word with no text refuses the file**, naming the segment and word.
  Previously it became an empty string and was dropped silently, so a word the
  service failed to return left no trace at all.
- **A field of the wrong type refuses the file**, as does a time that is
  negative, inverted, or far larger than any recording (the shape a provider
  returning an absolute timestamp would produce).

A refusal names the provider, the position and the fault, for example
`invalid Tencent ASR output at segment 3 word 7: Word is absent`. Re-run the
file after the provider issue is resolved; batchalign does not guess a value in
order to finish.

Aliyun performs no speaker separation, and the FunASR engines do not produce
speaker labels either, so their output is one undiarized track. Use
`--diarization enabled` (see [transcribe](commands/transcribe.md)) when speaker
attribution is needed with those engines.

## Engine Details

### Tencent Cloud ASR

- Supports speaker diarization with configurable speaker count
- Uploads audio to COS (Tencent Cloud Object Storage), submits ASR job, polls
  for results
- 10-minute safety timeout on ASR polling
- Automatic COS cleanup after transcription
- Per-word timestamps with speaker attribution

### Aliyun NLS ASR

- Cantonese only (`lang=yue` required, other languages rejected at load time)
- WebSocket streaming with real-time sentence callbacks
- Automatic token refresh (23-hour TTL)
- WAV format input required (16 kHz mono)
- Shared result shaping and Cantonese fallback tokenization happen in Rust,
  not in the Python transport adapter

### FunASR/SenseVoice

- Local model, no cloud credentials, no network at inference time
- SenseVoice (~0.9 GB) and its voice-activity model download from Hugging
  Face; the Paraformer checkpoint (~1.0 GB) and its voice-activity and
  punctuation (~0.3 GB) companions download from ModelScope. A host that can
  reach Hugging Face but not ModelScope can run `funaudio` and cannot run
  `paraformer`.
- Default model is `FunAudioLLM/SenseVoiceSmall`. Pass
  `--asr-engine funaudio --engine-overrides '{"funaudio_model": "<hf-id>"}'`
  to swap to a different FunASR model (e.g. a Paraformer variant); the
  loader's downstream code branches on whether the chosen model name
  contains `paraformer`.
- Which checkpoint you name changes what the transcript records. This build
  pins the default SenseVoice checkpoint and its voice-activity model, and
  pins the Paraformer checkpoint together with the voice-activity and
  punctuation models it loads; the stamp's `asr_model=` names every one of
  them with its revision. A checkpoint this build does not pin loads anyway
  and is recorded at the revision the worker reports for it, so an override
  never leaves the transcript naming the engine alone.
- VAD (Voice Activity Detection) built in via `fsmn-vad`
- Timestamps are paired with FunASR's OWN units, never with a retokenization of
  the display surface: SenseVoice's `words`, which FunASR builds in lockstep
  with its `timestamp` array, and Paraformer's pre-punctuation `raw_text`,
  which holds one whitespace token per timestamp. The unit and timestamp counts
  must agree before anything is paired

### Qwen3-ASR

- Local model via the [`qwen-asr`](https://github.com/QwenLM/Qwen3-ASR)
  PyPI package, no cloud credentials, no network at inference time.
- Default model is `Qwen/Qwen3-ASR-1.7B-hf`. The 0.6B variant is
  noticeably faster (smaller model, lighter compute) at some
  accuracy cost.
- First run downloads ~4.1 GB (1.7B) or ~1.6 GB (0.6B) from Hugging Face,
  plus ~1.8 GB for `Qwen/Qwen3-ForcedAligner-0.6B-hf`, which the worker
  always loads beside it; subsequent runs read from the local cache.
- The `qwen-asr` package handles long-audio chunking internally;
  no per-utterance pre-segmentation is required at the call site.
- Word-level timestamps emitted when the model returns them; falls
  back to whole-utterance text when timestamps aren't available.
- Single-speaker output (no built-in diarization); BA3's downstream
  diarization stage attaches speaker tags.
- Apache-2.0 licensed.

### Cantonese FA

- Converts Chinese characters to jyutping romanization (via pycantonese)
- Strips tones from jyutping (Wave2Vec MMS expects toneless input)
- Runs Wave2Vec forced alignment on the romanized text
- Maps word-level timings back to original Chinese characters

## See Also

- [Cantonese and CJK, Architecture](../../architecture/language-and-multilingual/cantonese-and-cjk.md), engine architecture, normalization pipeline, segmenter selection
- [Adding Inference Providers](../developer/adding-engines.md), how to add new built-in engines
