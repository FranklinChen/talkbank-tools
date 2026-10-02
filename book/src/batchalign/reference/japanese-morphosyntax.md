# Japanese Morphosyntax Pipeline

**Status:** Current
**Last updated:** 2026-10-01 08:17 EDT

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

All Japanese Stanza pipelines use `combined` instead of `default`.
The override is wired in
`batchalign/worker/_stanza_loading.py:196-209` (the `if alpha2 ==
"ja"` branch that constructs the `stanza.Pipeline` with an explicit
`package={...combined...}`):

```python
if alpha2 == "ja":
    nlp = stanza.Pipeline(
        lang=alpha2,
        processors=processors,
        download_method=DownloadMethod.REUSE_RESOURCES,
        tokenize_no_ssplit=True,
        tokenize_pretokenized=True,
        package={
            "tokenize": "combined",
            "pos": "combined",
            "lemma": "combined",
            "depparse": "combined",
        },
    )
```

The `combined` package bundles tokenization, POS tagging, lemmatization, and
dependency parsing into a single model trained jointly. Using `default` for any
processor would load a mismatched model.

### MWT Exclusion

Japanese never loads the `mwt` processor. The historical hardcoded
`_MWT_EXCLUSION` frozenset was retired in favour of the runtime
capability-driven helper `should_request_mwt()` at
`batchalign/worker/_stanza_loading.py:40` (see Defect 5 in
[Stanza Defect Mitigation Map](../architecture/stanza-defect-mitigation-map.md)):
the helper consults the live Stanza capability table and excludes
`mwt` for any language whose model does not ship that processor.
Japanese falls into the exclusion naturally, its `combined`
package does not include an `mwt` processor, so `should_request_mwt`
returns `False` for `ja` without a per-language entry. The package
override above is needed regardless of MWT mode, so the two
concerns stay independent.

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
stripped, not split, because the space does not represent a separate word.

### Fix 1: Token Text (`crates/batchalign-transform/src/morphosyntax/injection.rs`)

Before passing tokens to the retokenize algorithm, whitespace is stripped from
token text:

```rust,ignore
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

### Fix 2: Lemma Text (`crates/batchalign-transform/src/morphosyntax/ud_types.rs:426`)

Stanza's lemmatizer can also produce lemmas with internal whitespace
(e.g., `"ふ す"`). The `sanitize_mor_text()` function strips all
whitespace before %mor assembly:

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
merged token is a single linguistic unit, splitting on the space would create
two %mor entries for what Stanza considers one word, breaking the word↔mor
alignment. Stripping produces a valid CHAT word that correctly maps to one %mor
item.

---

## Lemma Cleaning

The `clean_lemma()` function
(`crates/batchalign-transform/src/morphosyntax/mor_word.rs:81`)
performs generic lemma cleanup, but several rules are
Japanese-relevant:

### Japanese Quote Fallback

When Stanza returns a Japanese bracket quote as the lemma, the
function falls back to the surface text (in the body of `clean_lemma`
at `mor_word.rs:81`):

```text
// Handle Japanese quotes
if target.trim() == "\u{300D}" || target.trim() == "\u{300C}" {  // 」 or 「
    target = text.to_string();
}
```

After the fallback, any remaining quote characters are stripped:

```text
target = target.replace('\u{300D}', ""); // 」
target = target.replace('\u{300C}', ""); // 「
```

### Smart Quote Handling

If the lemma contains a left smart quote (U+201C `"`), the function
falls back to the surface text (within `clean_lemma` at
`mor_word.rs:81`). This catches cases where Stanza's lemmatizer
produces a quote character instead of the actual lemma.

### Empty Lemma Safeguard

After all cleaning, if the lemma is empty, the function falls back
to the surface text. If the surface text is also empty, it uses
`"x"` as a placeholder (still within `clean_lemma` at
`mor_word.rs:81`). This prevents the E342 "bare pipe" parse error
(`pos|` with no stem).

---

## POS Mapping

The `map_ud_word()` function
(`crates/batchalign-transform/src/morphosyntax/mor_word.rs:13`) applies
Japanese-specific overrides in steps 3-4.

### Step 3: Verb Form Overrides

If the language is Japanese, verb form overrides run before generic
POS mapping (inside `map_ud_word` in `mor_word.rs`). The category a word is
written with is a typed `MorPos` (a UD category, or Japanese `cm`), so the
override returns a `UniversalPos`, never a category name:

```rust,ignore
let mut written = MorPos::Upos(tagged);
if lang2(&ctx.lang) == "ja" {
    if let Some(ovr) = japanese_verbform(tagged, &cleaned_lemma, &ud.text) {
        written = MorPos::Upos(ovr.pos);
        cleaned_lemma = ovr.lemma.replace(',', "cm");
    }
    // Step 4 below.
}
```

The comma→`"cm"` replacement handles the case where a verb form override
produces a lemma containing a comma (which would be illegal in a %mor stem).

### Step 4: PUNCT → cm

All Japanese `PUNCT` tokens map to the `cm` (comma marker) POS
category, and Japanese commas (both full-width `、` and ASCII `,`)
also map to `cm` (still inside `map_ud_word`):

```rust,ignore
if matches!(ud.upos, UdPunctable::Value(UniversalPos::Punct))
    || ud.lemma == "、"
    || ud.lemma == ","
{
    written = MorPos::JapaneseComma;
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

The `japanese_verbform()` function
(`crates/batchalign-transform/src/morphosyntax/lang_ja.rs`, 65 override
rules) is ported from the BA2 Python verb-form override file.

### Structure

```rust,ignore
pub struct JaOverride {
    pub pos: UniversalPos,   // New POS category
    pub lemma: &'static str, // New lemma
}

pub fn japanese_verbform(upos: UniversalPos, target: &str, text: &str) -> Option<JaOverride>
```

The function takes Stanza's UPOS, the cleaned lemma, and the surface text.
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

**Order is significant.** The function mirrors the original
Python's exact `if/elif` chain in
`crates/batchalign-transform/src/morphosyntax/lang_ja.rs`: earlier
rules take precedence. For example, a word containing both `ちゃ`
and `なきゃ` would match the `ちゃ` rule because it appears first.

### Execution Timing

Verb form overrides run **before** POS mapping (inside
`map_ud_word` at
`crates/batchalign-transform/src/morphosyntax/mor_word.rs:13`). This
means they can change both the POS category and lemma that flow
into feature computation and %mor assembly.

---

## No Clitic Detection

The `is_clitic()` function
(`crates/batchalign-transform/src/morphosyntax/mor_word.rs:200`)
identifies MWT sub-tokens that are clitics (e.g., English `n't`,
`'s`; French `l'`, `-ce`). Japanese has no entries, the function
returns `false` for all Japanese tokens:

```rust,ignore
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

## Per-Word Language Routing: Current Limitation

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

| Concept | File | Anchor |
|---------|------|--------|
| Capability-driven MWT exclusion (retired the historical `_MWT_EXCLUSION` frozenset) | `batchalign/worker/_stanza_loading.py` | `should_request_mwt()` @40 |
| `combined` package forcing for Japanese | `batchalign/worker/_stanza_loading.py` | :196-209 (`if alpha2 == "ja"` branch) |
| Stanza config (keep-tokens vs no-MWT) modes | `batchalign/worker/_stanza_loading.py` | `load_stanza_models()` @126 |
| Token text whitespace strip | `crates/batchalign-transform/src/morphosyntax/injection.rs` | retokenize-mode token sanitizer |
| Lemma whitespace strip | `crates/batchalign-transform/src/morphosyntax/ud_types.rs` | `sanitize_mor_text()` @426 |
| `clean_lemma()` (quote handling) | `crates/batchalign-transform/src/morphosyntax/mor_word.rs` | @81 |
| `map_ud_word()` (JA overrides) | `crates/batchalign-transform/src/morphosyntax/mor_word.rs` | @13 |
| Japanese PUNCT → cm | `crates/batchalign-transform/src/morphosyntax/mor_word.rs` | inside `map_ud_word` @13 |
| `is_clitic()` (no JA entries) | `crates/batchalign-transform/src/morphosyntax/mor_word.rs` | @200 |
| Verb form overrides | `crates/batchalign-transform/src/morphosyntax/lang_ja.rs` | `japanese_verbform()` |
| Retokenize algorithm | `crates/batchalign-transform/src/retokenize.rs` (+ `retokenize/{rebuild,parse_helpers}.rs`) | full module |
| `@s:` marker extraction | `../chatter/crates/talkbank-transform/src/extract.rs` | `WordLanguageMarker` extraction |
