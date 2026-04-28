# Per-Word Language Routing

**Status:** Current behavior reference  
**Last verified:** 2026-03-05

## Current behavior

Batchalign parses per-word language markers such as `@s:lang`, but current
runtime behavior does not do full per-word language routing into separate NLP
pipelines.

Practical current behavior:

- per-word language-marked forms are recognized structurally
- the current morphosyntax path does not send full per-word language codes
  through as a routing key for Python NLP inference
- code-switched words are handled conservatively rather than being analyzed as
  if high-confidence per-word language routing were already implemented

## What this means in output

For current `%mor` handling, language-marked code-switched words are treated as
special forms rather than as fully language-routed lexical items.

This is the safe current boundary:

- preserve that a word is foreign/code-switched
- do not overclaim morphology from the wrong language model

## Relationship to per-utterance routing

Per-word routing is more limited than per-utterance routing.

- per-utterance language directives can influence utterance-level processing
- per-word language markers do not currently drive full per-word model routing

See [Per-Utterance Language Routing](per-utterance-language-routing.md).

## Current limitation

If a transcript contains multiple code-switched words from different languages
inside one utterance, the current runtime does not route each word to a
different language-specific model and then merge the result back at word
granularity.

That limitation is intentional in current public docs: it is better to state
the boundary clearly than to imply richer routing than the release actually
provides.

## Legacy note

Earlier versions of this page analyzed hypothetical implementation options for
per-word routing. Those branch-era design notes are not the public reference
format now. This page documents only the current behavior and current limit.
