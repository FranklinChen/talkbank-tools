# Language Code Resolution

**Status:** Current
**Last updated:** 2026-03-19

This page documents how batchalign3 maps language codes to models, Stanza
pipelines, and processing behavior.

## Design Principle: No Silent Fallbacks

batchalign3 **never silently substitutes a different language** when the
requested language is unsupported. Every engine boundary must either:

1. **Succeed** — map the code correctly and proceed, or
2. **Fail explicitly** — return a clear error naming the unsupported language,
   the engine that rejected it, and suggested alternatives

Silent fallbacks (e.g., defaulting to English when a language is unknown)
produce output that looks plausible but is completely wrong. A clear error
is always better — users can recover from "language not supported" but
cannot recover from a transcript in the wrong language.

This policy applies to all engine boundaries: Rev.AI, Whisper, Stanza,
Tencent, Aliyun, and any future engines. See the
[migration book](../migration/algorithms-and-language.md) for the history
of how this policy was established after finding and fixing two silent
fallback regressions.

## Internal Representation: ISO 639-3

batchalign3 uses **3-letter ISO 639-3 codes** everywhere internally:

- CLI: `--lang=eng`, `--lang=yue`, `--lang=heb`
- CHAT headers: `@Languages: eng`, `@Languages: yue`
- Cache keys, IPC payloads, batch items
- Worker task bootstrap: `python -m batchalign.worker --task morphosyntax --lang eng`

The 3-letter code is the source of truth. Conversion to other formats happens
only at external boundaries.

## ISO 639-3 → ISO 639-1 (Stanza)

Stanza uses 2-letter ISO 639-1 codes. The mapping lives in
`batchalign/worker/_stanza_loading.py`:

```python
{
    "eng": "en", "spa": "es", "fra": "fr", "deu": "de",
    "ita": "it", "por": "pt", "nld": "nl", "zho": "zh",
    "jpn": "ja", "kor": "ko", "ara": "ar", "heb": "he",
    "hin": "hi", "tur": "tr", "fin": "fi", "hun": "hu",
    "pol": "pl", "ces": "cs", "ron": "ro", "bul": "bg",
    "hrv": "hr", "srp": "sr", "slk": "sk", "slv": "sl",
    "ukr": "uk", "lit": "lt", "lav": "lv", "est": "et",
    "cat": "ca", "eus": "eu", "glg": "gl", "ell": "el",
    "ind": "id", "msa": "ms", "vie": "vi", "tha": "th",
    "dan": "da", "nor": "nb", "swe": "sv", "afr": "af",
    "isl": "is",
    "yue": "zh",  # Cantonese → Chinese (Stanza has no yue-specific model)
    "cmn": "zh",  # Mandarin → Chinese
}
```

### Special Mappings

| 3-letter | 2-letter | Notes |
|----------|----------|-------|
| `yue` | `zh` | Cantonese shares Stanza's Chinese model. No yue-specific Stanza model exists. |
| `cmn` | `zh` | Mandarin. Same Stanza model as yue. |
| `heb` | `he` | Hebrew. Stanza's Hebrew model supports HebBinyan/HebExistential features. |

The mapping is exhaustive for supported languages. If an unknown code reaches
the worker, `iso3_to_alpha2()` falls back to the first two characters and logs
a warning before handing that value to Stanza.

## Model Resolution (ASR)

Language-specific ASR model selection only applies to the HuggingFace Whisper
engine (`--asr-engine whisper`). The resolver maps language codes to fine-tuned
models:

| Language | Code | Model | Base/Tokenizer | Size |
|----------|------|-------|----------------|------|
| English | eng | `talkbank/CHATWhisper-en` | `openai/whisper-large-v2` | large-v2 |
| Cantonese | yue | `alvanlii/whisper-small-cantonese` | same | small |
| Hebrew | heb | `ivrit-ai/whisper-large-v3` | same | large-v3 |
| All others | * | `openai/whisper-large-v3` | same | large-v3 |

Other ASR engines ignore language for model selection:
- `--asr-engine whisper-oai`: always `whisper-turbo`
- `--asr-engine whisperx`: always `whisper-large-v2`
- Rev.AI: cloud API, handles language internally

## Model Resolution (UTR)

UTR (Utterance Timing Recovery) has its own model resolution:

| Language | Code | Model |
|----------|------|-------|
| English | eng | `talkbank/CHATWhisper-en-large-v1` |
| All others | * | `openai/whisper-large-v2` |

## Model Resolution (Utterance Segmentation)

| Language | Code | Model |
|----------|------|-------|
| English | eng | `talkbank/CHATUtterance-en` |
| Mandarin | cmn/zho | `talkbank/CHATUtterance-zh_CN` |
| Cantonese | yue | `PolyU-AngelChanLab/Cantonese-Utterance-Segmentation` |
| All others | * | None (punctuation-based fallback) |

## Pipeline Stage Dispatch

The 3-letter code drives behavior at multiple pipeline stages:

```mermaid
flowchart TD
    lang["lang code (3-letter)"]
    asr["ASR model\nresolution"]
    num["Number expansion\n(zho/cmn → simplified,\nyue/jpn → traditional,\n12 table langs,\nothers → passthrough)"]
    canto["Cantonese norm\n(yue only)"]
    rtl["RTL punct\n(ara, heb, fas, urd)"]
    retok["Retokenize\n(jpn: 。→.)"]
    stanza["Stanza\n(3→2 letter mapping)"]
    mwt["MWT dispatch\n(39 currently enabled)"]
    workaround["Lang workarounds\n(eng, fra, jpn, ita, por, nld, heb)"]
    fa["FA engine\n(yue: wav2vec_canto option)"]

    lang --> asr
    lang --> num
    lang --> canto
    lang --> rtl
    lang --> retok
    lang --> stanza --> mwt
    lang --> workaround
    lang --> fa
```

## MWT Language Dispatch

Multi-Word Token (MWT) processing is driven by an explicit allowlist in
`batchalign/worker/_stanza_loading.py`.

Current behavior:

- Languages in `MWT_LANGS` load Stanza's `mwt` processor.
- Languages not in that allowlist skip `mwt`.
- Notably, Japanese (`ja`), Korean (`ko`), and Chinese (`zh`) are excluded.
- Thai (`th`), Vietnamese (`vi`), Indonesian (`id`), and Malay (`ms`) are
  currently included because they are present in the live allowlist.

See [Non-English Workarounds](../developer/non-english-workarounds.md) §X1 for the full
dispatch table.

## Rev.AI Language Codes

Rev.AI uses a mix of ISO 639-1 and specific codes. The translation lives in
`crates/batchalign-app/src/revai/preflight.rs`:

The translation now uses ~75 explicit entries in `try_revai_language_hint()`
with a fallback to `"auto"` (Rev.AI auto-detection) + warning log for
unknown codes. See `crates/batchalign-app/src/revai/preflight.rs`.

### Regression Fix: Truncation Fallback (fixed 2026-03-19)

The original batchalign3 implementation had a `&other[..2]` truncation
fallback for unknown ISO 639-3 codes. This was a **regression from
batchalign2**, which used `pycountry.languages.get(alpha_3=lang).alpha_2`
for correct conversion of all ~200+ languages.

The truncation produced wrong codes for many languages (e.g., `pol` → `po`
instead of `pl`, `hak` → `ha` which doesn't exist). This has been replaced
with:

1. A **comprehensive explicit mapping table** (~75 entries covering all
   Rev.AI-supported languages) in `revai/preflight.rs`
2. A **`try_revai_language_hint()`** function that returns `None` for
   unsupported languages (enabling callers to report clear diagnostics)
3. A **fallback to `"auto"`** (Rev.AI's auto-detection) with a warning log
   for unknown codes, rather than silently submitting a wrong code

## Whisper Language Strings

Some engines use human-readable language names rather than ISO codes.
The mapping in the worker:

```python
special = {"yue": "Cantonese", "cmn": "chinese"}
```

For all other languages, the worker resolves the ISO-639-3 code through
`pycountry` and passes the lower-cased language name to Whisper.

### Fixed: Silent English Fallback (fixed 2026-03-19)

Previously, if `pycountry` had no entry for an ISO 639-3 code, the Whisper
language name silently defaulted to `"english"`. This was a regression from
batchalign2 (which crashed on unknown codes via uncaught `AttributeError`).

Now raises `ValueError` with a clear message naming the unrecognized code.
See the [no-silent-fallback policy](#design-principle-no-silent-fallbacks).

## Cross-Engine Language Support Summary

There is **no central language translation layer**. Each engine handles
code conversion independently, with different mapping tables, different
fallback behaviors, and different supported language sets:

```mermaid
flowchart TD
    lang["LanguageCode3\n(ISO 639-3)"] --> stanza["Stanza\n55-entry explicit table\nfallback: truncate + warn"]
    lang --> revai["Rev.AI\n~75 explicit entries\nfallback: auto-detect + warn\n(fixed 2026-03-19)"]
    lang --> whisper["Whisper\npycountry lookup\nhard error on unknown\n(fixed 2026-03-19)"]
    lang --> tencent["Tencent\nChinese-only\n(zho/yue/wuu/nan/hak)"]
    lang --> aliyun["Aliyun\nCantonese-only\n(yue)"]
    lang --> funasr["FunASR\nprovider_lang_code()"]
    lang --> num["Number Expansion\n12 table languages\nothers: passthrough"]
    lang --> utr["UTR Strategy\neng → auto/two-pass\nothers → global"]

    %% All engines now have explicit validation — no red highlights needed
```

## Pre-Validation: Language Support Diagnostics

**Requirement:** Every engine must validate language support **before
processing begins** and produce a clear, actionable diagnostic. Users
should never see a cryptic HTTP 400 from Rev.AI or get silently wrong
English transcriptions because Whisper didn't know their language.

### The Problem

When a user passes `--lang hak` (Hakka), errors surface at different
points depending on the engine — all too late:

| Engine | When error surfaces | Error message | User impact |
|---|---|---|---|
| Rev.AI | **At job submission** | Clear error with alternatives | None — user redirected to Whisper/Tencent |
| Whisper | At inference time | `ValueError: Unrecognized ISO 639-3 code` | Clear error, no silent fallback |
| Stanza | **At job submission** | Clear error listing all 55 supported codes | None — immediate feedback |
| Tencent | **At job submission** | Clear error with alternatives | None — user redirected to Whisper/Rev.AI |
| Aliyun | **At job submission** | Clear error: Cantonese only | None — user redirected |

### Required Behavior

Validation should happen at **job submission time** (`POST /jobs`), before
any audio processing or model loading. The diagnostic should name the
engine, the unsupported language, and suggest alternatives:

```
Error: Language 'hak' (Hakka) is not supported by Rev.AI ASR.
  Supported alternatives:
  - Use --asr-engine whisper for Hakka (Whisper supports all languages)
  - Use --lang auto to let Rev.AI auto-detect the language
  - Use --asr-engine-custom tencent for Chinese/Hakka via Tencent ASR
```

### Per-Engine Language Support: Can We Query at Runtime?

**No engine provides a runtime "what languages do you support?" API.**
All language support is determined by static tables — either hardcoded in
our mapping code or documented on vendor websites. This means we must
maintain our own validation tables.

#### Rev.AI (cloud ASR)

- **No API endpoint** to query supported languages
- Supported languages documented on Rev.AI's website only
- Our mapping: `try_revai_language_hint()` in `revai/preflight.rs` (~75 entries)
- **Validation approach:** `try_revai_language_hint(lang)` returns `None` for
  unsupported languages; callers log a warning and fall back to `"auto"`
- **Important runtime nuance:** Rev `"auto"` is a real second request path, not
  just a late alias for English. If Rev language ID resolves to English before
  submission, BA3 uses the explicit-English request settings. If language ID
  fails and BA3 submits true Rev auto, later downstream stages may still
  resolve the transcript to English, but the provider request was different.
- **Example validation:**
  ```rust
  if try_revai_language_hint(&lang).is_none() {
      warn!("Language {lang} not in Rev.AI supported set; using auto-detection");
  }
  ```

#### Whisper (local HuggingFace ASR)

- **No programmatic language list** in the model API
- The model's `GenerationConfig` has `is_multilingual: true` but no language
  list; supported languages are implicit in the tokenizer's language tokens
- Our mapping: `iso3_to_language_name()` in `batchalign/inference/asr.py`
  uses `pycountry` lookup with special cases for `yue`/`cmn`
- **Fixed (2026-03-19):** Unknown codes now raise `ValueError` instead of
  silently falling back to English.
- Whisper `large-v3` nominally supports 99 languages, but quality varies
  dramatically (English/European languages are much better than others)

#### Stanza (morphosyntax)

- **Stanza has a `stanza.resources` module** that lists available models, but
  batchalign does not query it — instead using a hardcoded 55-entry mapping
- Could be queried at startup: `stanza.resources.common.list_available_languages()`
- Our mapping: `iso3_to_alpha2()` in `batchalign/worker/_stanza_loading.py`
- **Validation approach:** Check if the language exists in the explicit mapping
  table before attempting to load the pipeline

#### Tencent (cloud Chinese ASR)

- **No API endpoint**; hardcoded to Chinese variants only
- Our validation: `_CHINESE_CODES = {"zho", "yue", "wuu", "nan", "hak"}`
  in `batchalign/inference/hk/_tencent_asr.py`
- Already raises `ValueError` for non-Chinese — this is good, but happens
  at worker load time rather than job submission time

#### Aliyun (cloud Cantonese ASR)

- **No API endpoint**; hardcoded to Cantonese only
- Already raises `ValueError` if `lang != "yue"` — same timing issue

#### Google Translate / SeamlessM4T

- Google supports ~130 languages; SeamlessM4T ~96 languages
- **No pre-validation in batchalign3** — language is passed through directly
- Both have online documentation of supported languages but no programmatic query

#### NeMo / Pyannote (speaker diarization)

- **Language-independent** — diarization works on any language
- No validation needed

### Cached Language Support Table

Since no engine provides a runtime query API, we maintain a **time-stamped
reference table** of known language support. This table should be updated
whenever we upgrade an engine dependency or discover new support/regressions.

**Last verified: 2026-03-19**

| ISO 639-3 | Language | Rev.AI | Whisper | Stanza | Tencent | Notes |
|---|---|---|---|---|---|---|
| `eng` | English | `en` | `english` | `en` | — | Best quality across all engines |
| `spa` | Spanish | `es` | `spanish` | `es` | — | |
| `fra` | French | `fr` | `french` | `fr` | — | |
| `deu` | German | `de` | `german` | `de` | — | |
| `ita` | Italian | `it` | `italian` | `it` | — | |
| `por` | Portuguese | `pt` | `portuguese` | `pt` | — | |
| `nld` | Dutch | `nl` | `dutch` | `nl` | — | |
| `jpn` | Japanese | `ja` | `japanese` | `ja` | — | MWT disabled |
| `kor` | Korean | `ko` | `korean` | `ko` | — | MWT disabled |
| `rus` | Russian | `ru` | `russian` | `ru` | — | |
| `ara` | Arabic | `ar` | `arabic` | `ar` | — | RTL punctuation |
| `tur` | Turkish | `tr` | `turkish` | `tr` | — | |
| `zho` | Chinese | `cmn` | `chinese` | `zh` | `16k_zh_large` | Maps to cmn/zh depending on engine |
| `cmn` | Mandarin | `cmn` | `chinese` | `zh` | `16k_zh_large` | UTSeg model available |
| `yue` | Cantonese | — | `cantonese` | `zh` | `16k_zh_large` | Fine-tuned Whisper; Aliyun/CantoneseFA; UTSeg model |
| `hak` | Hakka | — | `hakka` | — | `16k_zh_large` | Rev.AI unsupported; Stanza unsupported |
| `pol` | Polish | `pl` | `polish` | `pl` | — | |
| `ces` | Czech | `cs` | `czech` | `cs` | — | |
| `ron` | Romanian | `ro` | `romanian` | `ro` | — | |
| `hun` | Hungarian | `hu` | `hungarian` | `hu` | — | |
| `heb` | Hebrew | `he` | `hebrew` | `he` | — | Fine-tuned Whisper; RTL punctuation |
| `hin` | Hindi | `hi` | `hindi` | `hi` | — | |
| `cym` | Welsh | `cy` | `welsh` | `cy` | — | |
| `srp` | Serbian | `sr` | `serbian` | `sr` | — | |
| `afr` | Afrikaans | `af` | `afrikaans` | `af` | — | |
| `fin` | Finnish | `fi` | `finnish` | `fi` | — | |
| `dan` | Danish | `da` | `danish` | `da` | — | |
| `swe` | Swedish | `sv` | `swedish` | `sv` | — | |
| `nor` | Norwegian | `no` | `norwegian` | `nb` | — | |

*(Partial list — full table covers ~75 Rev.AI languages. `—` = not supported.)*

**Maintenance policy:** Update this table whenever:
- A new engine version is deployed (check changelogs for language additions)
- A user reports a language failure (add the language with its support status)
- We add a new engine to batchalign3

### Implementation Plan

A `validate_language_support(lang, command, engines)` function should:

1. Check each engine the command will use against its supported language set
2. Return a structured diagnostic listing which engines support the language
   and which don't, with suggested alternatives
3. Be called at job submission time (Rust server) — before any processing
4. Optionally be callable from the CLI for `batchalign3 setup --check-lang hak`

This is tracked as a future improvement. The Rev.AI mapping table
(`try_revai_language_hint`) provides the pattern to follow for other engines.

## Adding a New Language

To add language-specific behavior for a new language:

1. **Stanza mapping** — Add to the ISO 639-3 → 639-1 table in `_stanza_loading.py`
2. **Rev.AI mapping** — Add an explicit entry in `revai/preflight.rs` (do NOT
   rely on the truncation fallback)
3. **ASR model** — Optionally add a fine-tuned model entry in the resolver
4. **Number expansion** — Add a table to `data/num2lang.json` or handle in
   `num2text.rs`
5. **Utterance segmentation** — Optionally train a BERT boundary model
6. **Morphosyntax workarounds** — Add a `nlp/lang_XX.rs` file if Stanza
   produces systematic errors for this language
7. **MWT dispatch** — Determine if the language uses contractions
8. **Test with `benchmark`** — Verify the language code is accepted by all
   engines in the pipeline (ASR, morphotag, compare)
