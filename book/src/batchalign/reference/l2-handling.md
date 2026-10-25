# L2 and Language Switching

**Status:** Current behavior reference
**Last updated:** 2026-10-29

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

Historical April 2026 evaluation: **99.96% dispatch rate** (16,838 / 16,845 `@s`
words successfully routed to a secondary-language Stanza model;
7 fell back to `L2|xxx`) across 19 language pairs in the
`l2-eval-runs/2026-04-15/per-pair.csv` aggregate eval. This historical
dispatch measurement does not qualify the current build or model accuracy.
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

A language listed as secondary in `@Languages` does not itself require a
model. Inline `@s:LANG` spans use a supported secondary model when one is
available. When the admitted runtime registry explicitly lacks that model,
only those spans retain `L2|xxx`; the primary-language words and other
successful secondary spans keep their analysis.

`L2|xxx` records unavailable lexical analysis. It does not assert that a model
identified an unknown part of speech. Job diagnostics distinguish unsupported
language exclusions from failed inference, empty responses, invalid secondary
graphs and unresolved or ambiguous language markers. The saved transcript also
gets an `@Comment` beginning `[batchalign morphology availability]`, naming
unsupported languages and counts, or summarizing analysis that could not be
completed. Contributor comments are retained. Re-running replaces this one
application-owned availability comment rather than accumulating duplicates.

For example, an unsupported inline span in `hello mundo@s:spa .` can yield
`intj|hello L2|xxx .`, with a comment saying that Spanish L2 morphology was
unsupported. Its dependency attachment comes from the primary utterance's
context; it does not certify Spanish syntactic analysis.

Whole-utterance `[- LANG]` precodes require a primary analysis of that entire
utterance. If its effective language lacks required processors, capability
admission refuses the requested file analysis with a typed diagnostic before
inference. BA3 does not relabel the utterance or invent a whole tier of lexical
analyses. A file's unsupported primary language is likewise refused. These
requirements differ from an unavailable inline span embedded in an analyzable
utterance. An absent registry is unavailable capability evidence, never proof
that a language is unsupported.

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
  on the main tier, words, fillers (`&~`/`&-`/`&+`), nonwords, AND
  retraced material, carries an explicit language attribution
  resolving to the same target, and clears bare `@s` shortcuts on
  fillers and nonwords as part of the rewrite (otherwise the new
  `[- LANG]` precode would flip their resolved language).

## Related references

- [L2 Morphotag: Per-Word Code-Switching Analysis](l2-morphotag.md)
 , full design, merge algorithm, phrasal-verb diagram
- [Language Routing](../../architecture/language-and-multilingual/language-routing.md), per-utterance + per-word routing, auto-detection
- [Language Data Model](language-handling.md)
