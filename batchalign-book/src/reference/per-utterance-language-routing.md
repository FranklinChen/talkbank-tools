# Per-Utterance Language Routing

**Status:** Current
**Last updated:** 2026-03-19

## What it does

When a CHAT file contains utterances in multiple languages, batchalign3
routes each utterance to the correct language-specific Stanza model for
morphosyntax analysis. This means a bilingual interview where the
interviewer speaks English and the participant speaks French will get
correct `%mor` and `%gra` tiers for both languages — each processed by
the appropriate Stanza pipeline.

## How it works

Language is determined per utterance in this priority order:

1. **`[- lang]` precode** on the utterance (e.g., `[- fra]`) — highest priority
2. **`@Languages` header** — first declared language used as fallback
3. **`--lang` CLI flag** — used when no file-level language is declared

For example, in a bilingual English/French file:

```
@Languages: eng, fra
*INV: how are you today ? 0_3000
*PAR: [- fra] je suis bien merci . 3000_6000
```

The investigator's utterance is processed with the English Stanza pipeline,
and the participant's utterance (marked `[- fra]`) is processed with the
French Stanza pipeline.

## Stanza pipeline loading

Stanza pipelines are loaded **on demand**. The worker starts with the
primary language model, then loads additional language models as it
encounters utterances in new languages. This means:

- No upfront cost for monolingual files (only one model loaded)
- First utterance in a new language may take a few seconds (model download + load)
- Subsequent utterances in the same language reuse the loaded pipeline

## Improvement over batchalign2

batchalign2 parsed the `[- lang]` precode but **did not use it for
routing** — all utterances were processed with the primary language's
Stanza pipeline regardless of their language directive. Non-primary
utterances either got wrong-language morphosyntax (default) or were
skipped entirely (`skipmultilang=True`).

batchalign3 implements true per-utterance routing: each language group
is batched separately and sent to its own Stanza pipeline.

## Limitations

- **Per-word routing is not supported.** Words marked with `@s:lang`
  (language-switched words within an utterance) are not routed to a
  different Stanza model. They receive an `L2|xxx` placeholder in `%mor`.
  See [Per-Word Language Routing](per-word-language-routing.md).

- **ASR is not per-utterance.** The `--lang` flag selects a single ASR
  engine/model for the entire file. Transcription of multilingual audio
  uses one language model throughout.

## See also

- [Per-Word Language Routing](per-word-language-routing.md) — `@s:` handling
- [Multilingual Support](multilingual.md) — overview
- [Language Code Resolution](language-code-resolution.md) — how codes map to models
- [Language Data Model](language-handling.md) — internal representation
