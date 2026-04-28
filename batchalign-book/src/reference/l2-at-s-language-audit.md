# Audit: @s Language Codes vs @Languages Header

**Status:** Reference
**Last updated:** 2026-04-04 07:29 EDT

## Background

CHAT has two mechanisms for declaring languages:

- **`@Languages` header** — declares the substantial languages of the
  transcript (e.g., `@Languages: deu, eng` for a German-English bilingual)
- **`@s` word markers** — mark individual code-switched words, either
  with an explicit language code (`@s:eng`) or a bare shortcut (`@s`)
  that toggles to the secondary language declared in `@Languages`

These serve different purposes:
- `@Languages` identifies languages with substantial content
- `@s:CODE` can reference **any language**, including stray words from
  a tertiary language not listed in `@Languages`

## Audit Results

Scanned all 24 TalkBank data repos. Of 4,423 files containing `@s:CODE`
words (with explicit language codes):

| Metric | Count |
|--------|------:|
| Files with @s:CODE | 4,423 |
| Files where CODE ∉ @Languages | **2,291 (52%)** |
| Distinct unlisted language codes | **47** |

**Over half of all files with explicit @s codes use languages not
declared in the `@Languages` header.**

## Top Unlisted Languages

| Code | Files | Typical context |
|------|------:|----------------|
| `eng` | 1,326 | English words in non-English transcripts (Cantonese, Croatian, French, etc.) |
| `fra` | 302 | French words in English aphasia/clinical transcripts |
| `spa` | 208 | Spanish words in English transcripts |
| `ita` | 123 | Italian words in English clinical transcripts |
| `deu` | 82 | German words in English transcripts |
| `eus` | 56 | Basque words in Spanish CHILDES transcripts |
| `sun` | 47 | Sundanese words in Swedish-Finnish SLA transcripts |
| `jpn` | 45 | Japanese words in French ASD and bilingual transcripts |
| `hun` | 37 | Hungarian words in English CHILDES (MacWhinney family data) |
| `hin` | 28 | Hindi words in English MICASE/Manchester transcripts |
| `nan` | 25 | Taiwanese (Hokkien) words in Mandarin CHILDES transcripts |
| `lat` | 22 | Latin words in English aphasia/conversation transcripts |
| `zho` | 21 | Chinese words in English ASD/CHILDES transcripts |

The full list includes 47 languages spanning:
- Major world languages (Arabic, Portuguese, Greek, Dutch, Russian)
- Regional languages (Basque, Sundanese, Cantonese, Hakka)
- Heritage languages (Yiddish, Hawaiian, Swahili)
- Classical languages (Latin)
- Undetermined (`und` — 15 files)
- Erroneous 2-letter codes (`cy`, `es`, `sp`, `nle`, `enh` — annotation errors)

## Impact on L2 Morphotag

### Explicit @s:CODE — works correctly

When a word has `@s:CODE` (e.g., `ok@s:eng`), the `resolve_word_language`
function resolves to `LanguageResolution::Single("eng")` **regardless of
what `@Languages` declares**. The L2 dispatch sends the word to the
English Stanza model. This is correct behavior.

### Bare @s shortcut — standard behavior

Bare `@s` (no code) is the standard CHAT shortcut. The resolver toggles
from the current language to the other language declared in `@Languages`.
This works correctly whenever `@Languages` declares two languages.

578 files have bare `@s` with only one declared language — these are a
data annotation issue (the file should declare a second language if it
has code-switching). The resolver correctly produces `Unresolved` →
`L2|xxx` fallback. No code change needed.

### Stanza model availability

Not all 47 unlisted languages have Stanza models:

| Has Stanza model | Example languages |
|:-:|---|
| Yes | eng, fra, spa, ita, deu, jpn, hin, zho, nld, rus, kor, tur, ara, heb, ell, por, pol |
| No | eus (Basque), sun (Sundanese), nan (Taiwanese), hun (Hungarian)¹, yid (Yiddish), lat (Latin), haw (Hawaiian), swa (Swahili) |

¹ Hungarian does have a Stanza model but with limited accuracy.

For unsupported languages, the L2 dispatch correctly falls back to
`L2|xxx` (the Stanza support check prevents dispatching to nonexistent
models).

## Erroneous Codes

Several files use non-standard language codes:

| Code | Likely intended | Files |
|------|----------------|------:|
| `cy` | `cym` (Welsh) | 1 |
| `es` | `spa` (Spanish) | 1 |
| `sp` | `spa` (Spanish) | 1 |
| `nle` | `nld` (Dutch) | 1 |
| `enh` | `eng` (English) | 1 |
| `cye` | `cym` (Welsh) | 4 |
| `ena` | `eng` (English)? | 2 |
| `tze` | `tzh` (Tzeltal)? | 1 |

These should be validated and corrected in the data repos.

## Recommendations

1. **No code changes needed for explicit @s:CODE** — the dispatch
   already handles unlisted languages correctly by resolving the
   explicit code directly.

2. **Bare @s in single-language files** — 578 files have bare `@s` with
   only one declared language. `resolve_word_language` returns `Unresolved`
   which is tagged `#[validation_tag(error)]` — `chatter validate`
   already reports this as a validation error.

3. **Erroneous codes** — the 8 non-standard codes should be reported
   to corpus maintainers for correction.

4. **Document in CHAT manual** — clarify that `@s:CODE` works
   independently of `@Languages`, while bare `@s` requires at least
   two declared languages.

## Related

- [L2 Morphotag](l2-morphotag.md)
- [L2 Morphotag Aggregate Eval (2026-04-15)](l2-eval-runs/2026-04-15/summary.md)
