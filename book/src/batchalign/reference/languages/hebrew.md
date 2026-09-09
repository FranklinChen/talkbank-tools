# Hebrew Language Support

**Status:** Current
**Last updated:** 2026-09-07 07:04 EDT

Hebrew (`heb`) ships with RTL punctuation handling, capability-driven
MWT for preposition+article contractions, and Hebrew-specific UD
features (HebBinyan, HebExistential).

## Quick Reference

| Pipeline Stage | Hebrew-Specific Behavior |
|---------------|--------------------------|
| ASR | `openai/whisper-large-v3` via `--asr-engine whisper` (no per-language fine-tune wired today) |
| Text normalization | RTL punctuation → ASCII (؟→?, ۔→., ،→,) |
| Number expansion | **Not supported**: digits pass through unexpanded |
| Morphosyntax | MWT enabled (preposition+article contractions), HebBinyan/HebExistential features |
| Forced alignment | Wave2Vec MMS (standard) |

## ASR

| Engine | Model | Notes |
|--------|-------|-------|
| `--asr-engine whisper` | `openai/whisper-large-v3` | Default; same model for every language |
| `--asr-engine whisper_hub` | (empty entry today) | Opt-in HuggingFace fine-tune loader; requires explicit `--engine-overrides` `model_id`. No Hebrew fine-tune is seeded in `batchalign/models/resolve.py::_RESOLVER["whisper_hub"]` yet. |
| Rev.AI | Cloud API | Supports Hebrew |

If a Hebrew-specific fine-tune (such as the community
`ivrit-ai/whisper-large-v3` checkpoint) is wanted, add it reactively to
`_RESOLVER["whisper_hub"]` with a dated provenance comment and an
empirical evaluation note. See
[Whisper Hub ASR](../whisper-hub-asr.md) for the conventions.

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
(e.g., בַּ → ב + ה). MWT eligibility is **capability-driven**:
`should_request_mwt(alpha2, get_cached_capability_table())` at
`batchalign/worker/_stanza_loading.py:40` consults the cached Stanza
catalog and reports `has_mwt=True` for `he`. The earlier hardcoded
`MWT_LANGS` / `_MWT_EXCLUSION` sets were deleted,
[Stanza Limitations Defect 5](../stanza-limitations.md) has the
rewrite rationale.

### HebBinyan (Verb Conjugation Pattern)

Hebrew verbs belong to one of seven binyanim: PAAL, NIFAL, PIEL, PUAL,
HIFIL, HUFAL, HITPAEL. Stanza outputs this as the `HebBinyan` UD feature.
batchalign3 converts to lowercase %mor suffix:

```text
UD: HebBinyan=PAAL|Number=Sing|Person=3|Tense=Past|VerbForm=Fin
%mor: -paal&3S&PAST
```

### HebExistential

The existential (יש/אין) gets a special `HebExistential=True` feature,
mapped to `%mor` suffix `-true`.

### No Other Workarounds

Unlike English, French, Japanese, Italian, Portuguese, and Dutch, Hebrew has
**no Stanza workarounds**: the HebBinyan/HebExistential mapping is standard
UD feature processing.

## Known Limitations

### No Hebrew number expansion
Hebrew digits pass through unexpanded. CHAT output will show `5` instead of
`חמש`. This is a known gap, a Hebrew number table for `num2lang.json` has
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

1. **Hebrew number expansion**: should we add a Hebrew number table? What
   are the conventions for Hebrew CHAT transcripts (digits vs word forms)?
2. **Hebrew-specific ASR errors**: are there systematic Stanza POS/dep
   errors for Hebrew that need workaround rules (like English/French/Japanese)?

## Source Files

| File | Role |
|------|------|
| `crates/batchalign-transform/src/morphosyntax/features.rs:43,46` | HebBinyan/HebExistential extraction |
| `crates/batchalign-transform/src/morphosyntax/types.rs:38` | `heb` → `he` Stanza alpha-2 mapping |
| `crates/batchalign/src/chat_ops/nlp/mapping/tests/lang_de_es_he.rs` | Hebrew integration tests (`:15`, `:45`, `:268`) |
| `batchalign/worker/_stanza_loading.py` | Stanza pipeline config; MWT eligibility driven by `should_request_mwt()` |
