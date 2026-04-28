# Hebrew Language Support

**Status:** Current
**Last updated:** 2026-03-23 12:15 EDT

Hebrew (`heb`) has a fine-tuned Whisper model, RTL punctuation handling,
MWT support, and Hebrew-specific UD features (HebBinyan, HebExistential).

## Quick Reference

| Pipeline Stage | Hebrew-Specific Behavior |
|---------------|--------------------------|
| ASR | `ivrit-ai/whisper-large-v3` fine-tuned Hebrew Whisper |
| Text normalization | RTL punctuation → ASCII (؟→?, ۔→., ،→,) |
| Number expansion | **Not supported** — digits pass through unexpanded |
| Morphosyntax | MWT enabled (preposition+article contractions), HebBinyan/HebExistential features |
| Forced alignment | Wave2Vec MMS (standard) |

## ASR

| Engine | Model | Notes |
|--------|-------|-------|
| `--asr-engine whisper` | `ivrit-ai/whisper-large-v3` | Fine-tuned for Hebrew conversational speech |
| `--asr-engine whisper-oai` | `openai/whisper-turbo` | Generic, lower accuracy |
| Rev.AI | Cloud API | Supports Hebrew |

The `ivrit-ai/whisper-large-v3` model significantly outperforms generic Whisper
on Hebrew audio.

## RTL Punctuation

Hebrew text may contain Arabic-script punctuation. The ASR post-processing
pipeline normalizes these to ASCII (runs for all languages, not just Hebrew):

| RTL | ASCII | Unicode |
|-----|-------|---------|
| ؟ | ? | U+061F Arabic Question Mark |
| ۔ | . | U+06D4 Arabic Full Stop |
| ، | , | U+060C Arabic Comma |
| ؛ | ; | U+061B Arabic Semicolon |

## Morphosyntax

### MWT (Multi-Word Tokens)

Hebrew uses Stanza's MWT processor for preposition+article contractions
(e.g., בַּ → ב + ה). Hebrew is in `MWT_LANGS`, not `_MWT_EXCLUSION`.

### HebBinyan (Verb Conjugation Pattern)

Hebrew verbs belong to one of seven binyanim: PAAL, NIFAL, PIEL, PUAL,
HIFIL, HUFAL, HITPAEL. Stanza outputs this as the `HebBinyan` UD feature.
batchalign3 converts to lowercase %mor suffix:

```
UD: HebBinyan=PAAL|Number=Sing|Person=3|Tense=Past|VerbForm=Fin
%mor: -paal&3S&PAST
```

### HebExistential

The existential (יש/אין) gets a special `HebExistential=True` feature,
mapped to `%mor` suffix `-true`.

### No Other Workarounds

Unlike English, French, Japanese, Italian, Portuguese, and Dutch, Hebrew has
**no Stanza workarounds** — the HebBinyan/HebExistential mapping is standard
UD feature processing.

## Known Limitations

### No Hebrew number expansion
Hebrew digits pass through unexpanded. CHAT output will show `5` instead of
`חמש`. This is a known gap — a Hebrew number table for `num2lang.json` has
not been created.

### RTL text layout
The CHAT format is line-oriented and primarily designed for LTR text. Hebrew
CHAT files work correctly for parsing and analysis, but display rendering in
editors depends on the editor's BiDi support.

## Verified Behavior

| Test | What |
|------|------|
| `test_hebrew_verb_hebbinyan` | HebBinyan=PAAL → lowercase suffix |
| `test_hebrew_verb_hebexistential` | HebExistential=True → lowercase suffix |
| `test_hebrew_3letter_code_works` | "heb" maps to Stanza "he" correctly |

## Open Questions

1. **Hebrew number expansion** — should we add a Hebrew number table? What
   are the conventions for Hebrew CHAT transcripts (digits vs word forms)?
2. **Hebrew-specific ASR errors** — are there systematic Stanza POS/dep
   errors for Hebrew that need workaround rules (like English/French/Japanese)?

## Source Files

| File | Role |
|------|------|
| `batchalign-chat-ops/src/nlp/features.rs` | HebBinyan/HebExistential extraction |
| `batchalign-chat-ops/src/nlp/mapping.rs` | `heb` → `he` mapping, integration tests |
| `worker/_stanza_loading.py` | Stanza pipeline config (MWT enabled) |
