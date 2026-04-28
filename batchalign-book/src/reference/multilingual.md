# Multilingual Support

**Status:** Current behavior reference  
**Last verified:** 2026-03-19

## Current multilingual boundary

Batchalign supports multilingual corpora primarily through:

- file-level language metadata
- per-utterance language directives
- conservative handling of per-word code-switch markers

This is the current public contract. It is more explicit than older BA2-era
single-language assumptions, but it is not the same as full per-word bilingual
analysis.

## Current morphosyntax behavior

Current morphosyntax handling is language-aware at the utterance level.

Practical consequences:

- utterances can be processed with language-aware routing rather than assuming
  the entire file is one language
- code-switched words marked at word level are handled conservatively rather
  than being forced through a possibly wrong language model
- cache and payload handling keep language as part of the processing boundary

## Current output consequences

Users working with multilingual corpora should expect:

- better behavior than a file-wide single-language assumption
- clearer distinction between utterance-level language handling and word-level
  foreign-language marking
- some foreign/code-switched words to remain conservatively represented instead
  of receiving overconfident morphology

## Current limits

Batchalign does not currently promise:

- full per-word routing into multiple language-specific NLP pipelines
- perfect code-switch analysis inside one utterance
- complete elimination of manual review for difficult bilingual material

## Related references

- [Language-Specific Processing](language-specific-processing.md) — how each pipeline stage diverges per language
- [Language Code Resolution](language-code-resolution.md) — ISO mapping, model resolution
- [Per-Utterance Language Routing](per-utterance-language-routing.md)
- [Per-Word Language Routing](per-word-language-routing.md)
- [Language Data Model](language-handling.md)
- [L2 & Language Switching](l2-handling.md)
