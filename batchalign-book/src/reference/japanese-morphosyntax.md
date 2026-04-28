# Japanese Morphosyntax Pipeline

**Date**: 2026-02-26
**Status**: Implemented (batchalign3)

---

## Overview

Japanese is a non-MWT language with a "combined" Stanza package. Unlike
MWT-capable languages (English, French, Italian) that split contractions into
syntactic sub-words, Japanese uses a fundamentally different tokenization
strategy where Stanza's neural models may merge or re-segment CJK characters.

Two tokenization modes (`retokenize` vs keep-tokens) behave very differently
for Japanese, and several Stanza output artifacts require language-specific
cleanup in the Rust POS mapping layer.

This document explains the full pipeline end-to-end: from Stanza configuration
through verb form overrides, POS mapping, whitespace artifact handling, and
current limitations.

---

## Stanza Configuration

Japanese uses two distinct Stanza configurations depending on the `retokenize`
flag. Both modes force the `combined` package for all four processors.

### Package Selection

All Japanese Stanza pipelines use `combined` instead of `default`
(`engine.py:127–132`):

```python
if alpha2 == "ja":
    processors["tokenize"] = "combined"
    processors["pos"] = "combined"
    processors["lemma"] = "combined"
    processors["depparse"] = "combined"
```

The `combined` package bundles tokenization, POS tagging, lemmatization, and
dependency parsing into a single model trained jointly. Using `default` for any
processor would load a mismatched model.

### MWT Exclusion

Japanese is in `_MWT_EXCLUSION` (`engine.py:45–49`):

```python
_MWT_EXCLUSION = frozenset({
    "hr", "zh", "zh-hans", "zh-hant", "ja", "ko",
    "sl", "sr", "bg", "ru", "et", "hu",
    "eu", "el", "he", "af", "ga", "da",
})
```

This means Japanese never loads the `mwt` processor. The `combined` package is
forced separately (above) because `_MWT_EXCLUSION` only controls whether MWT
expansion runs — the package override is needed regardless of mode.

### Keep-Tokens Mode (`retokenize=False`)

Default mode. Stanza's tokenizer is completely bypassed via
`tokenize_pretokenized=True`. The CHAT words are passed directly as
pre-tokenized input, giving a safe 1:1 mapping between input words and Stanza
tokens. No word merging or splitting can occur.

### Retokenize Mode (`retokenize=True`)

Stanza's neural tokenizer runs with `tokenize_no_ssplit=True` (no sentence
splitting). The `combined` tokenizer may merge adjacent CHAT words or split
single words into sub-tokens. The Rust retokenize algorithm (`retokenize.rs`)
then realigns the modified tokenization back onto the CHAT AST using
character-level DP alignment.

| Property | Keep-Tokens | Retokenize |
|----------|-------------|------------|
| Stanza tokenizer | Bypassed (`pretokenized`) | Runs (`no_ssplit`) |
| Word boundaries | Preserved | May change |
| Token mapping | 1:1 | N:M (DP alignment) |
| Whitespace artifacts | Rare | Common |
| Use case | Morphotag on existing transcripts | Full re-analysis |

---

## Stanza Whitespace Artifacts

When Stanza's `combined` tokenizer runs (retokenize mode), it may merge
adjacent CHAT words into a single token while preserving the ASCII space from
the join. For example, two CHAT words `ふ` and `す` become a single Stanza
token `"ふ す"` (with internal space).

This whitespace is a tokenization artifact, not a word boundary. It must be
stripped — not split — because the space does not represent a separate word.

### Fix 1: Token Text (`morphosyntax/inject.rs:103–115`)

Before passing tokens to the retokenize algorithm, whitespace is stripped from
token text:

```rust
if retokenize {
    // Stanza's combined tokenizer (e.g. Japanese) sometimes merges
    // adjacent CHAT words into a single token while preserving the
    // ASCII space from the join.  Strip any whitespace so the token
    // text is a valid CHAT word.
    let tokens: Vec<String> = ud_sentence
        .words
        .iter()
        .map(|w| {
            if w.text.contains(char::is_whitespace) {
                w.text.chars().filter(|c| !c.is_whitespace()).collect()
            } else {
                w.text.clone()
            }
        })
        .collect();
    retokenize::retokenize_utterance(utt, &words, &tokens, /* ... */);
}
```

### Fix 2: Lemma Text (`validation.rs:19–22`)

Stanza's lemmatizer can also produce lemmas with internal whitespace (e.g.,
`"ふ す"`). The `sanitize_mor_text()` function strips all whitespace before
%mor assembly:

```rust
pub fn sanitize_mor_text(s: &str) -> String {
    let mut result = s.replace(['|', '#', '-', '&', '$', '~'], "_");
    result.retain(|c| !c.is_whitespace());
    result
}
```

This also replaces MOR structural separators (`|`, `#`, `-`, `&`, `$`, `~`)
with underscores, preventing syntactic contamination of the %mor tier.

### Why Stripping, Not Splitting

The internal space is a Stanza artifact from merging two CHAT tokens. The
merged token is a single linguistic unit — splitting on the space would create
two %mor entries for what Stanza considers one word, breaking the word↔mor
alignment. Stripping produces a valid CHAT word that correctly maps to one %mor
item.

---

## Lemma Cleaning

The `clean_lemma()` function (`mapping.rs:441–554`) performs generic lemma
cleanup, but several rules are Japanese-relevant:

### Japanese Quote Fallback

When Stanza returns a Japanese bracket quote as the lemma, the function falls
back to the surface text (`mapping.rs:445–456`):

```rust
// Handle Japanese quotes
if target.trim() == "\u{300D}" || target.trim() == "\u{300C}" {  // 」 or 「
    target = text.to_string();
}
```

After the fallback, any remaining quote characters are stripped:

```rust
target = target.replace('\u{300D}', ""); // 」
target = target.replace('\u{300C}', ""); // 「
```

### Smart Quote Handling

If the lemma contains a left smart quote (U+201C `"`), the function falls back
to the surface text (`mapping.rs:517–519`). This catches cases where Stanza's
lemmatizer produces a quote character instead of the actual lemma.

### Empty Lemma Safeguard

After all cleaning, if the lemma is empty, the function falls back to the
surface text. If the surface text is also empty, it uses `"x"` as a
placeholder (`mapping.rs:546–551`). This prevents the E342 "bare pipe" parse
error (`pos|` with no stem).

---

## POS Mapping

The `map_ud_word_to_mor()` function (`mapping.rs:345–399`) applies
Japanese-specific overrides in steps 3–4.

### Step 3: Verb Form Overrides

If the language is Japanese, verb form overrides run before generic POS mapping
(`mapping.rs:359–365`):

```rust
if lang2(&ctx.lang) == "ja"
    && let Some(ovr) = lang_ja::japanese_verbform(&effective_pos, &cleaned_lemma, &ud.text)
{
    effective_pos = ovr.pos.to_string();
    cleaned_lemma = ovr.lemma.to_string();
    cleaned_lemma = cleaned_lemma.replace(',', "cm");
}
```

The comma→`"cm"` replacement handles the case where a verb form override
produces a lemma containing a comma (which would be illegal in a %mor stem).

### Step 4: PUNCT → cm

All Japanese `PUNCT` tokens map to the `cm` (comma marker) POS category, and
Japanese commas (both full-width `、` and ASCII `,`) also map to `cm`
(`mapping.rs:367–375`):

```rust
if lang2(&ctx.lang) == "ja" {
    if matches!(ud.upos, UdPunctable::Value(UniversalPos::Punct)) {
        effective_pos = "cm".to_string();
    }
    if ud.lemma == "、" || ud.lemma == "," {
        effective_pos = "cm".to_string();
    }
}
```

| Input | UPOS | Resulting POS |
|-------|------|---------------|
| `。` | PUNCT | `cm` |
| `、` | PUNCT | `cm` |
| `,` | PUNCT | `cm` |
| `「` | PUNCT | `cm` |

This differs from other languages where `PUNCT` maps to `punct` and only
actual commas map to `cm`.

---

## Verb Form Overrides

The `japanese_verbform()` function (`lang_ja.rs`, 460 lines, 65 override
rules) is ported from Python's `batchalign/pipelines/morphosyntax/ja/verbforms.py`.

### Structure

```rust
pub struct JaOverride {
    pub pos: &'static str,   // New POS category
    pub lemma: &'static str, // New lemma
}

pub fn japanese_verbform(upos: &str, target: &str, text: &str) -> Option<JaOverride>
```

The function takes the lowercased UPOS tag, cleaned lemma, and surface text.
It returns `Some(JaOverride)` if a match is found, `None` otherwise.

### Categories

The 65 rules cover these categories (in match order):

| Category | Count | Examples |
|----------|-------|---------|
| Conditional/subjunctive conjunctions | 3 | ちゃ→ば, なきゃ, じゃ |
| Auxiliary verbs | ~8 | られる, ちゃう, おう, たら |
| Interjections | ~9 | はい, うん, おっ, ほら, あのね |
| Pronouns | ~2 | あたし |
| Verb lemma corrections | ~11 | 撮る, 貼る, 混ぜる, 釣る, 帰る |
| Noun overrides | ~8 | バツ, ブラシ, 引き出し, マヨネーズ |
| Adjective specializations | ~3 | 速い |
| 為る context overrides | ~5 | Verb/noun/aux disambiguation |
| Participles and other | ~16 | Various form-specific overrides |

### Order Dependence

**Order is significant.** The function mirrors Python's exact `if/elif` chain
(`lang_ja.rs:34`): earlier rules take precedence. For example, a word
containing both `ちゃ` and `なきゃ` would match the `ちゃ` rule because it
appears first.

### Execution Timing

Verb form overrides run **before** POS mapping (`mapping.rs:359`). This means
they can change both the POS category and lemma that flow into feature
computation and %mor assembly.

---

## No Clitic Detection

The `is_clitic()` function (`mapping.rs:322–329`) identifies MWT sub-tokens
that are clitics (e.g., English `n't`, `'s`; French `l'`, `-ce`). Japanese has
no entries — the function returns `false` for all Japanese tokens:

```rust
fn is_clitic(text: &str, ctx: &MappingContext) -> bool {
    match lang2(&ctx.lang) {
        "en" => text == "n't" || text == "'s" || text == "'ve" || text == "'ll",
        "fr" => text.ends_with('\'') || text == "-ce" || text == "-être" || text == "-là",
        "it" => text.ends_with('\''),
        _ => false,  // Japanese falls through here
    }
}
```

This is correct: Japanese does not use MWT expansion, so there are no clitic
sub-tokens to identify.

---

## Per-Word Language Routing — Current Limitation

CHAT supports per-word language markers (`@s:jpn`) for code-switching. The
Rust extraction layer (`extract.rs`) correctly extracts these markers into a
`WordLanguageMarker` enum with variants for bare (`@s`), explicit
(`@s:jpn`), multiple (`@s:eng+jpn`), and ambiguous (`@s:eng&jpn`).

However, the language code is currently discarded at the Rust→Python boundary
during morphosyntax processing. All language-marked words become `L2|xxx` in
the `%mor` output regardless of the specified language. Per-utterance language
routing is the current supported boundary; per-word language routing is not part
of the current public runtime contract.

---

## Code Reference

| Concept | File | Lines |
|---------|------|-------|
| `_MWT_EXCLUSION` list | `engine.py` | 45–49 |
| `combined` package forcing | `engine.py` | 127–132 |
| Stanza config modes | `engine.py` | 105–169 |
| Token text whitespace strip | `lib.rs` | 1447–1462 |
| Lemma whitespace strip | `validation.rs` | 19–22 |
| `clean_lemma()` (quote handling) | `mapping.rs` | 441–554 |
| `map_ud_word_to_mor()` (JA overrides) | `mapping.rs` | 345–399 |
| Japanese PUNCT → cm | `mapping.rs` | 367–375 |
| `is_clitic()` (no JA entries) | `mapping.rs` | 322–329 |
| Verb form overrides | `lang_ja.rs` | 1–460 |
| Retokenize algorithm | `retokenize.rs` | full file |
| `@s:` marker extraction | `extract.rs` | 31–42 |
