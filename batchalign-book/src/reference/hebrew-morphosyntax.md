# Hebrew Morphosyntax

**Date:** 2026-03-09

Hebrew-specific handling in batchalign3's morphosyntax pipeline.

## Language Code

| Internal (ISO 639-3) | Stanza (ISO 639-1) | Notes |
|----------------------|---------------------|-------|
| `heb` | `he` | Standard mapping |

## ASR

Hebrew uses a fine-tuned Whisper model for the HuggingFace engine:

| Engine | Model |
|--------|-------|
| `--asr-engine whisper` | `ivrit-ai/whisper-large-v3` (fine-tuned for Hebrew) |
| `--asr-engine whisper-oai` | `openai/whisper-turbo` (generic) |
| Rev.AI | Cloud API (supports Hebrew) |

The `ivrit-ai/whisper-large-v3` model is trained on Hebrew conversational
speech and significantly outperforms generic Whisper on Hebrew audio.

## RTL Punctuation

Hebrew text may contain Arabic-script punctuation from mixed content. The ASR
post-processing pipeline normalizes RTL punctuation to ASCII:

| RTL | ASCII | Unicode |
|-----|-------|---------|
| ؟ | ? | U+061F Arabic Question Mark |
| ۔ | . | U+06D4 Arabic Full Stop |
| ، | , | U+060C Arabic Comma |
| ؛ | ; | U+061B Arabic Semicolon |

This normalization runs for all languages, not just Hebrew — it ensures CHAT
files contain only ASCII punctuation terminators regardless of source script.

## Morphosyntax Features

Hebrew has two language-specific UD features that batchalign3 maps to CHAT
%mor suffixes: **HebBinyan** and **HebExistential**.

### HebBinyan (Verb Conjugation Pattern)

Hebrew verbs belong to one of seven binyanim (conjugation patterns):
PAAL, NIFAL, PIEL, PUAL, HIFIL, HUFAL, HITPAEL.

Stanza's Hebrew model outputs the `HebBinyan` feature on verbs. batchalign3
converts it to a lowercase suffix in %mor:

```
UD features: HebBinyan=PAAL|Number=Sing|Person=3|Tense=Past|VerbForm=Fin
%mor suffix: -paal&3S&PAST
```

The binyan is lowercased in the suffix: `PAAL` → `paal`, `HIFIL` → `hifil`.

### HebExistential

The Hebrew existential (יש/אין — "there is"/"there isn't") gets a special
feature in Stanza:

```
UD features: HebExistential=True|VerbForm=Fin
%mor suffix: -true
```

The value is lowercased: `True` → `true`.

### Feature Format

The full verb suffix format (shared across all languages):

```
-VerbForm-Aspect-Mood-Tense-Polarity-Polite-HebBinyan-HebExistential-NumberPerson-irr
```

Hebrew-specific features slot into their dedicated positions. The `-irr`
suffix (English irregular verbs) is **not applied** to Hebrew — it is
gated to English only.

## Implementation

The feature extraction is language-agnostic in implementation — the code
in `features.rs` checks for `HebBinyan` and `HebExistential` in any
language's feature set, but only Stanza's Hebrew model actually produces
these features:

```rust
// batchalign-chat-ops/src/nlp/features.rs
if let Some(v) = feats.get("HebBinyan") {
    parts.push(v.to_lowercase());
}
if let Some(v) = feats.get("HebExistential") {
    parts.push(v.to_lowercase());
}
```

## MWT

Hebrew uses the MWT processor. Stanza's Hebrew model handles Hebrew
contractions (preposition + article combinations like בַּ → ב + ה).

## Number Expansion

Hebrew does not have a dedicated number expansion table in `num2lang.json`.
Digit strings in Hebrew ASR output pass through unexpanded. This is a known
gap — Hebrew numbers in CHAT output will appear as digits rather than
Hebrew word forms (אחת, שתיים, שלוש, etc.).

## No Other Language-Specific Workarounds

Unlike English, French, Japanese, Italian, Portuguese, and Dutch, Hebrew has
**no Stanza workarounds** in batchalign3. The HebBinyan and HebExistential
feature mapping is standard UD feature processing, not a bug workaround.

If systematic Stanza errors are discovered for Hebrew, a `nlp/lang_he.rs`
file should be created following the pattern of existing language files.

## Source Files

| File | What |
|------|------|
| `batchalign-chat-ops/src/nlp/features.rs` | HebBinyan/HebExistential extraction |
| `batchalign-chat-ops/src/nlp/mapping.rs` | `heb` → `he` code mapping, integration tests |
| `batchalign/worker/_stanza_loading.py` | Stanza pipeline configuration for Hebrew |

## Test Coverage

| Test | File | What |
|------|------|------|
| `test_hebrew_verb_hebbinyan` | `mapping.rs:2334` | HebBinyan=PAAL → lowercase suffix |
| `test_hebrew_verb_hebexistential` | `mapping.rs:2364` | HebExistential=True → lowercase suffix |
| `test_hebrew_3letter_code_works` | `mapping.rs:2777` | "heb" (not "he") processes HebBinyan |
