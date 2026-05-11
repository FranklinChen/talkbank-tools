# L2 and Language Switching

**Status:** Current behavior reference
**Last updated:** 2026-05-06 20:33 EDT

## Current behavior

Batchalign distinguishes between:

- utterance-level language directives
- word-level language markers such as `@s` and `@s:lang`

L2 dispatch is on by default. The legacy `L2|xxx` behavior is
available via the `--no-l2-morphotag` opt-out.

## `%mor` behavior A: real morphology via secondary dispatch (default)

For `@s` or `@s:lang` words, the pipeline routes each word to a
secondary-language Stanza model and merges the response with the
primary model's structural analysis. The `%mor` tier carries real
POS/lemma/features:

```text
# Default
%mor:  ... adp|auf noun|film noun|study-Plur .
```

## `%mor` behavior B: conservative `L2|xxx` (opt-out)

With `--no-l2-morphotag`, `@s` words are blanked:

- the word is recognized as foreign/code-switched
- `%mor` output is `L2|xxx`
- no lexical/morphological analysis is preserved inside `%mor`

This is the legacy behavior researchers cite in work published
before the L2 morphotag feature landed, and remains the honest
fallback when a secondary Stanza model is known to be weak.

```text
# --no-l2-morphotag
%mor:  ... adp|auf L2|xxx L2|xxx .
```

Validated at scale: **99.96% dispatch rate** (16,838 / 16,845 `@s`
words successfully routed to a secondary-language Stanza model;
7 fell back to `L2|xxx`) across 19 language pairs in the
`l2-eval-runs/2026-04-15/per-pair.csv` aggregate eval.
Contractions expand correctly (`it's@s:eng` →
`pron|it~aux|be`), phrasal verbs are recognized (`wake up@s` →
`verb|wake part|up`). See
[L2 Morphotag: Per-Word Code-Switching Analysis](l2-morphotag.md)
for the full design and merge algorithm.

## Utterance-level versus word-level behavior

### Utterance-level

Utterance-level language directives affect utterance handling and routing
boundaries.

### Word-level

Word-level language markers identify foreign/code-switched words and, during
`morphotag`, do trigger per-word secondary-language routing when the target
language resolves cleanly and a supported Stanza path exists.

## Current limits

The remaining conservative fallbacks are:

- `--no-l2-morphotag` opts back into legacy `L2|xxx`
- unresolved / ambiguous markers such as `@s:eng+spa` or `@s:eng&spa` do not
  dispatch to one secondary model
- unsupported target languages still preserve the code-switch signal via
  `L2|xxx` rather than inventing morphology

## Unsupported non-primary languages

`morphotag` only requires the **primary** `@Languages` code to be
Stanza-supported; files whose primary is unsupported are skipped with
a typed diagnostic before the pipeline runs. When the primary IS
supported, non-primary content targeting an unsupported language is
processed cleanly with an `L2|xxx` fallback:

- `[- UNSUPPORTEDLANG]` whole-utterance precodes — the utterance is
  grouped under `UNSUPPORTEDLANG`, the worker partitions that group
  out of Stanza dispatch (`partition_groups_by_stanza_support`), and
  every word receives `L2|xxx`.
- `@s:UNSUPPORTEDLANG` per-word markers — the secondary L2 dispatch
  span is short-circuited the same way; the host primary analysis is
  preserved and the marker's slot stays `L2|xxx`.

The worker never crashes on an unsupported secondary, and other
utterances or spans in the same file targeting supported languages
continue to receive real morphology.

## Validation and normalization policy

- Whole-utterance same-language all-`@s` patterns are rejected by validation
  (E255). The accepted transcript form is utterance-level `[- lang]`, not
  `word@s word@s ...` for an entire utterance.
- Explicit `@s:LANG` still dispatches to `LANG` even when `LANG` is missing from
  `@Languages`, but validation emits warn-only E254 so the header mismatch is
  visible.
- Batchalign does not silently rewrite either case during morphotag. Use
  `chatter debug fix-s` to normalize whole-utterance `@s` runs and append
  missing explicit languages to `@Languages` without touching already-correct
  files. The fix-s predicate verifies that **every** word-bearing item
  on the main tier — words, fillers (`&~`/`&-`/`&+`), nonwords, AND
  retraced material — carries an explicit language attribution
  resolving to the same target, and clears bare `@s` shortcuts on
  fillers and nonwords as part of the rewrite (otherwise the new
  `[- LANG]` precode would flip their resolved language).

## Related references

- [L2 Morphotag: Per-Word Code-Switching Analysis](l2-morphotag.md)
  — full design, merge algorithm, phrasal-verb diagram
- [Language Routing](../../architecture/language-and-multilingual/language-routing.md) — per-utterance + per-word routing, auto-detection
- [Language Data Model](language-handling.md)
