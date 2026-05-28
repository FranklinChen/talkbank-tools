# Cantonese Language Support

**Status:** Current
**Last updated:** 2026-05-27 21:27 EDT

User reference for Cantonese (`yue`) processing in batchalign3 — ASR engine
options, credentials, retokenize usage, and what to expect from each
pipeline stage. For the architecture and rationale (engine dispatch,
normalization pipeline, segmenter selection, source-file map), see
[Cantonese and CJK — Architecture](../../../architecture/language-and-multilingual/cantonese-and-cjk.md).

## Quick Reference

| Pipeline stage | Cantonese-specific behavior |
|---|---|
| ASR | 5 engine options: FunASR/SenseVoice (default), Tencent Cloud, Aliyun NLS, Qwen3-ASR, Whisper |
| Text normalization | Simplified → Traditional + 31-entry domain replacement (automatic) |
| Number expansion | Traditional Chinese characters (五, 四十二, 一萬) |
| Character tokenization | Per-character splitting for timestamp alignment |
| Word segmentation | PyCantonese `segment()` via `--retokenize` |
| Utterance segmentation | PolyU BERT model (`PolyU-AngelChanLab/Cantonese-Utterance-Segmentation`) in standalone `utseg` and transcribe pre-CHAT segmentation; falls back to punctuation |
| Morphosyntax (POS) | PyCantonese override (~95% on core vocab) layered on Stanza Chinese (`zh`) |
| Morphosyntax (depparse) | Stanza Chinese (`zh`) — Mandarin-trained, but better than nothing |
| Forced alignment | Jyutping romanization (PyCantonese) → Wave2Vec MMS |

## ASR Engine Options

**The default for `yue` is FunASR/SenseVoice** — a local model that
empirically outperforms vanilla Whisper-large-v3 by a wide margin
on Cantonese child speech (42.8% CER vs 81.9% CER on TalkBank Tier
3 fixtures; see the 2026-05-26 Cantonese ASR benchmark). The
default is wired in `batchalign/worker/_model_loading/asr.py`'s
`_LANG_DEFAULTS` table and applies when no `--engine-overrides
'{"asr":...}'` is set and no Rev.AI key is configured. Alternatives
are activated via `--engine-overrides`.

| Engine | Type | Credentials | Word output | Strength |
|---|---|---|---|---|
| FunASR/SenseVoice (**default for yue**) | Local | None | Per-character | No cloud, VAD built-in, lowest measured CER on child speech |
| Tencent Cloud | Cloud | Required | Per-character | Speaker diarization, strong on clean adult speech |
| Aliyun NLS | Cloud | Required | Per-character | Real-time streaming |
| Qwen3-ASR | Local | None | Per-character | Alibaba open-weight ASR; competitive on per-utterance Cantonese child speech in external evaluations (unverified on TalkBank's longer-form fixtures) |
| Whisper | Local | None | Per-character | General-purpose multilingual; **worst measured on TalkBank Cantonese** — not recommended unless other engines are unavailable |

### Usage

```bash
# Default (FunASR/SenseVoice) — no flag needed
batchalign3 transcribe input/ -o output/ --lang yue

# Tencent Cloud ASR (requires credentials)
batchalign3 transcribe input/ -o output/ --lang yue \
  --engine-overrides '{"asr": "tencent"}'

# Aliyun NLS ASR (requires credentials)
batchalign3 transcribe input/ -o output/ --lang yue \
  --engine-overrides '{"asr": "aliyun"}'

# Qwen3-ASR (default 1.7B variant; pinned via qwen_model for 0.6B)
batchalign3 transcribe input/ -o output/ --lang yue \
  --engine-overrides '{"asr": "qwen"}'
batchalign3 transcribe input/ -o output/ --lang yue \
  --engine-overrides '{"asr": "qwen", "qwen_model": "Qwen/Qwen3-ASR-0.6B"}'

# Whisper (explicit opt-in; not recommended for Cantonese)
batchalign3 transcribe input/ -o output/ --lang yue \
  --engine-overrides '{"asr": "whisper"}'

# Cantonese forced alignment
batchalign3 align input/ -o output/ --lang yue \
  --engine-overrides '{"fa": "wav2vec_canto"}'
```

### Credentials

Cloud engines (Tencent, Aliyun) require credentials in `~/.batchalign.ini`:

```ini
[asr]
# Tencent Cloud
engine.tencent.id = YOUR_SECRET_ID
engine.tencent.key = YOUR_SECRET_KEY
engine.tencent.region = ap-guangzhou
engine.tencent.bucket = YOUR_COS_BUCKET

# Aliyun NLS
engine.aliyun.ak_id = YOUR_ACCESS_KEY_ID
engine.aliyun.ak_secret = YOUR_ACCESS_KEY_SECRET
engine.aliyun.ak_appkey = YOUR_APPKEY
```

Missing or empty credentials raise `ConfigError` with a clear message.

### Engine details

**Tencent Cloud ASR.** Speaker diarization with configurable count.
Uploads audio to COS, submits ASR job, polls for results (10-min
timeout). Returns pre-segmented words with per-word timestamps and
speaker attribution. Automatic COS cleanup after job completes.

**Aliyun NLS ASR.** Cantonese only (`lang=yue` required). WebSocket
streaming with real-time callbacks. Automatic token refresh (23-hour TTL).
WAV format required (16 kHz mono).

**FunASR/SenseVoice.** Local model — no cloud credentials, no network.
Auto model selection: Paraformer or SenseVoice based on availability.
VAD built in. Per-character timestamp alignment. Wired as the
per-language default for `yue` via `_LANG_DEFAULTS` in
`batchalign/worker/_model_loading/asr.py` so a bare
`batchalign3 transcribe --lang yue ...` invocation no longer falls
through to Whisper.

**Qwen3-ASR.** Alibaba's open-weight Cantonese-capable ASR
(`qwen-asr` Python package, model downloaded from HuggingFace on
first use). Two variants are publicly released — 1.7B (default,
heavier) and 0.6B (lighter, smaller download). Select the 0.6B
variant via `--engine-overrides '{"asr": "qwen", "qwen_model":
"Qwen/Qwen3-ASR-0.6B"}'`; per-engine extras like `qwen_model` and
`qwen_device` are accepted by the `EngineOverrides` schema and
forwarded to the worker. External evaluations report competitive
CER on per-utterance Cantonese child speech with the 1.7B variant;
TalkBank's own longer-form Cantonese fixtures show this engine
benefits from per-utterance segmentation rather than full-session
input.

The Qwen3-ASR worker always pairs the ASR model with
`Qwen/Qwen3-ForcedAligner-0.6B`, Qwen's companion forced-aligner
model. The aligner is loaded at worker bootstrap (~1.2 GB
additional download on first use, then cached) so the ASR pipeline
emits word-level timestamps the downstream FA stage can consume.
This pairing is required, not optional: the `qwen-asr` library
raises `ValueError` from `model.transcribe(...,
return_time_stamps=True)` whenever the aligner was not supplied at
`Qwen3ASRModel.from_pretrained(...)`. Dropping
`return_time_stamps=True` to "fix" that error degrades every
downstream `%wor` tier and is therefore not a permitted shortcut;
the only correct configuration is the aligner-paired one.

**Cantonese forced alignment.** Converts Chinese characters to jyutping
romanization (via PyCantonese), strips tone numbers for Wave2Vec
compatibility, runs Wave2Vec FA on romanized text, maps word-level
timings back to original characters.

## Text Normalization

All Cantonese ASR output is automatically normalized regardless of which
engine produced it. No configuration. Simplified → Traditional via
OpenCC `s2hk`, then a 31-entry domain replacement table for
Cantonese-specific corrections (`真系→真係`, `中意→鍾意`, `系→係`,
`呀→啊`, `松→鬆`, …).

Full example: `你真系好吵呀` → `你真係好嘈啊`.

The replacement table was originally written by Chuqiao Song in
batchalign2's `replace_cantonese_words()` (Python + OpenCC C++). Rebuilt
in Rust for batchalign3 — no C++ dependency, always available, correct
overlapping pattern handling.

## Word Segmentation — `--retokenize`

FunASR/SenseVoice and Whisper output per-character tokens for Cantonese:
each character becomes a separate word on the main tier. This makes word
counts, MLU, and POS tagging unreliable.

```bash
# Morphotag has no --lang flag — the per-file @Languages: header drives
# routing. For Cantonese files (yue), retokenize is the right default.
batchalign3 morphotag --retokenize corpus/ -o output/
```

This uses PyCantonese's `segment()` to group per-character tokens into
words before Stanza POS tagging. Cantonese files are detected from each
file's `@Languages: yue` header — there is no morphotag `--lang` flag.

**Before** (per-character):

```text
*CHI:	故 事 係 好 .
%mor:	n|故 n|事 v|係 adj|好 .
```

**After** (`--retokenize`):

```text
*CHI:	故事 係 好 .
%mor:	n|故事 v|係 adj|好 .
```

Without `--retokenize`, tokenization is preserved unchanged. A diagnostic
warning is emitted when Cantonese input appears per-character:

```text
warn: Cantonese input appears to be per-character tokens (42/50 single-CJK words).
      Consider --retokenize for word-level analysis.
```

### Validation across all 9 TalkBank Cantonese corpora

Word segmentation was tested against all 9 Cantonese corpora in TalkBank
(over 737,000 utterances). Multi-character preservation 84–90%,
vocabulary coverage 98–100% across MOST, LeeWongLeung, CHCC, EACMC, HKU
(CHILDES), MAIN, GlobalTales, and Aphasia HKU. Test:
`batchalign/tests/languages/cantonese/morphosyntax/test_cantonese_all_corpora.py`.

## Number Expansion

Cantonese uses **traditional** Chinese number characters: `5` → `五`,
`42` → `四十二`, `10000` → `一萬` (not `一万`). Implemented via
`num2chinese(n, ChineseScript::Traditional)` in Rust. Runs as Stage 4 of
ASR post-processing, before Stage 4b (text normalization).

See [Number Expansion](../number-expansion.md) for the full language
table.

## Utterance Segmentation

Uses the PolyU BERT model
`PolyU-AngelChanLab/Cantonese-Utterance-Segmentation`. Falls back to
punctuation-based splitting if the model is unavailable. The same model is used
for `transcribe`'s pre-CHAT segmentation when `--lang yue`. See
[Utterance Segmentation](../utterance-segmentation.md).

## Mixed-language morphotag (`@s`)

In bilingual files, Cantonese-marked words (`@s:yue` or bare `@s` resolved to
`yue`) go through the same default-on secondary-language L2 morphotag path as
other supported languages. Successful secondary dispatch produces real
`%mor`/`%gra`; unresolved or unsupported cases still fall back to `L2|xxx`.

## Known limitations

- **POS tagging on Cantonese vocabulary.** Stanza's `zh` model is
  Mandarin-trained — `佢/佢哋` (he/they) → `PROPN`, `嘢` (thing) →
  `PUNCT`, `唔` (not) → `VERB`, `係` (is) → `VERB`. PyCantonese POS
  override fixes core vocabulary as post-processing but has dictionary
  gaps on compound nouns, some SFPs, and resultative verbs. See the
  architecture page for the full rationale and the trained-but-undeployed
  Cantonese model.
- **Word segmentation depends on PyCantonese dictionary.** Words not in
  the dictionary won't be grouped.
- **All four ASR engines produce per-character output for Cantonese** —
  `--retokenize` is needed for all Cantonese morphotag.
- **FunASR CER varies with speech clarity** — increases with
  overlapping/soft/child speech.
- **Per-character warning threshold (80%) is empirical-without-basis** —
  not yet validated against real corpus data.
- **Daemon warning visibility.** The `tracing::warn!` for per-character
  input fires in the daemon process, not the CLI. Users may not see it
  until SSE events or job results surface it.
