# Language Type System Redesign

**Status:** Reference
**Last updated:** 2026-03-20

Research report and recommendations for unifying language code handling across
talkbank-tools and batchalign3. The full design doc is tracked out-of-tree.

## Problem Summary

- **6 independent mapping tables** (ISO 639-3 to engine-specific formats) with no shared source of truth
- **3 incompatible language code types**: `LanguageCode` (talkbank-model, `Arc<str>`), `LanguageCode3` (batchalign-app, `String`), `LanguageCode` (Python, bare `str`)
- **Zero conversions** between talkbank-model and batchalign types
- **Python has zero type safety** — `LanguageCode = str` is a bare alias
- **"auto" sentinel** leaks through `from_worker_lang()` and flows freely in Python
- Each ML engine expects a different format (Whisper: English names, Rev.AI: ISO 639-1, Stanza: ISO 639-1, HK engines: mixed)

## Recommendations

1. **Shared `talkbank-lang` crate** with `Iso639_3`, `Iso639_1`, bidirectional mapping
2. **Engine-specific newtypes** (`WhisperLang`, `RevAiLang`, `StanzaLang`) with typed conversions
3. **Separate `Auto` from codes everywhere** — enforce resolution before dispatch
4. **Single mapping table** (JSON data file) with build-time codegen for all engines
5. **Python type safety** — `NewType` + Pydantic validators, remove empty-string defaults
6. **ISO 639-3 registry validation** — embed the ~7,000 code list, warn on unrecognized

See the [existing language architecture doc](language-architecture.md) for the current pipeline flow.
