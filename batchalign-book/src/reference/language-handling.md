# Language Handling in CHAT: Complete Data Model

**Date**: 2026-02-15
**Status**: Comprehensive reference for language scoping in CHAT format

---

## Overview

CHAT supports multi-level language specification for multilingual corpora and code-switching analysis:

1. **File level**: `@Languages` header declares all languages used in the file
2. **Tier level**: `[- lang]` directive switches utterance language
3. **Word level**: `@s` and `@s:lang` markers for code-switched individual words

This document explains how these layers work together and how our data model represents them.

---

## 1. File-Level Languages: @Languages Header

### Purpose

Declares the **primary**, **secondary**, and optionally **tertiary** languages used throughout the file.

### Syntax

```
@Languages:   eng, spa, fra
              ^^^  ^^^  ^^^
              1°   2°   3°
```

- **Primary language** (1st position): Default for all tiers and words
- **Secondary language** (2nd position): Referenced by bare `@s` shortcut
- **Tertiary+ languages** (3rd+ positions): Must use explicit `@s:lang` markers

### Data Model

**Header enum**:
```rust
pub enum Header {
    Languages { codes: LanguageCodes },
    // ...
}

pub struct LanguageCodes(pub Vec<LanguageCode>);
```

**Extraction** (from ChatFile):
```rust
let declared_languages: Vec<LanguageCode> = chat_file
    .headers()
    .find_map(|h| {
        if let Header::Languages { codes } = h {
            Some(codes.0.clone())  // LanguageCodes tuple struct
        } else {
            None
        }
    })
    .unwrap_or_else(|| vec![primary_lang.clone()]);
```

### Example

```chat
@Languages:   eng, spa
@Participants: CHI Child, MOT Mother
@ID: eng|corpus|CHI|...
@ID: eng|corpus|MOT|...

*CHI: I want biberon@s please .
      ^^^^^^^^^^^^^^^^^^^^^^^^^^^
      Primary language (eng)
              ^^^^^^^
              Secondary language (spa), marked with @s
```

---

## 2. Tier-Level Language: [- lang] Directive

### Purpose

**Switches the language for an entire utterance**. Used when a speaker produces a full utterance in a different language.

### Syntax

```
*MOT: hola cómo estás [- spa] ?
      ^^^^^^^^^^^^^^^
      Entire utterance is Spanish
```

The directive applies to **the entire tier** (all words in the utterance).

### Data Model

**MainTierContent struct**:
```rust
pub struct MainTierContent {
    pub content: Vec<UtteranceContent>,
    pub terminator: Option<Terminator>,
    pub language_code: Option<LanguageCode>,  // ← [- lang] directive
    // ...
}
```

- `language_code: None` → Use primary language from @Languages
- `language_code: Some("spa")` → This utterance is Spanish

### Morphosyntax Processing

When Rust builds batch payloads for Stanza, it includes the tier language:

```rust
let utterance_lang = utt.main.content.language_code
    .clone()
    .unwrap_or_else(|| primary_lang.clone());

// Include in batch payload
MorphosyntaxBatchItem {
    words: vec!["hola", "cómo", "estás"],
    lang: utterance_lang,  // "spa"
    // ...
}
```

Python then routes the entire utterance to the Spanish Stanza model.

### Skipping vs. Processing

**`skipmultilang` flag**:
- `true`: Skip utterances with `[- lang]` directive (only process primary language)
- `false` (default): Process utterances in their declared language

```rust
let skip = skipmultilang
    && utt.main.content.language_code.is_some()
    && utt.main.content.language_code.as_ref() != Some(&primary_lang);

if !skip {
    // Process with Stanza model for utterance_lang
}
```

---

## 3. Word-Level Language: @s Markers

### Purpose

Marks **individual code-switched words** within an utterance.

### Syntax

| Marker | Meaning | Example |
|--------|---------|---------|
| `@s` | Shortcut: secondary language from @Languages | `biberon@s` → Spanish |
| `@s:spa` | Explicit single language | `biberon@s:spa` → Spanish |
| `@s:eng+fra` | Multiple languages (word legal in both) | `cafe@s:eng+fra` |
| `@s:eng&spa` | Ambiguous (could be either language) | `no@s:eng&spa` |

### Data Model: CHAT Syntax Layer

**Word struct**:
```rust
pub struct Word {
    // ...
    pub lang: Option<WordLanguageMarker>,
}

pub enum WordLanguageMarker {
    /// Bare @s (shortcut to secondary language)
    Shortcut,
    /// Single explicit language @s:spa
    Explicit(WordLanguage),
    /// Multiple languages @s:eng+fra (legal in all)
    Multiple(Vec<WordLanguage>),
    /// Ambiguous languages @s:eng&spa (unclear which)
    Ambiguous(Vec<WordLanguage>),
}

pub struct WordLanguage {
    pub code: Option<LanguageCode>,  // ISO 639-3 code
    pub variant: Option<String>,     // Optional variant like :spa%mex
}
```

This is the **CHAT syntax representation** - it preserves the exact marker as written in the file.

### Data Model: Semantic Resolution Layer

**LanguageResolution enum** (from `resolve_word_language()`):
```rust
pub enum LanguageResolution {
    /// Single definite language (after resolving @s shortcut)
    Single(LanguageCode),
    /// Multiple languages (code-mixing): @s:eng+fra
    Multiple(Vec<LanguageCode>),
    /// Ambiguous between languages: @s:eng&spa
    Ambiguous(Vec<LanguageCode>),
    /// No language could be resolved (error)
    Unresolved,
}
```

This is the **semantic representation** - it resolves shortcuts and produces actual language codes.

### Resolution Algorithm

**`resolve_word_language()` function**:

```rust
pub fn resolve_word_language(
    word: &Word,
    tier_language: Option<&LanguageCode>,
    declared_languages: &[LanguageCode],
) -> (LanguageResolution, Vec<ParseError>)
```

**Resolution rules**:

1. **@s shortcut** → Secondary language from @Languages
   - Example: `@Languages: eng, spa` → `@s` resolves to `spa`
   - Error if used in tertiary language tier (E244)

2. **@s:spa** → Single(spa)

3. **@s:eng+fra** → Multiple([eng, fra])

4. **@s:eng&spa** → Ambiguous([eng, spa])

5. **No marker** → Use tier language (or primary if no tier directive)

### Example Resolution

**Input CHAT**:
```
@Languages:   eng, spa, fra

*CHI: I want biberon@s and croissant@s:fra please .
```

**Resolution**:
- `"I"` → No marker → Single(eng) (tier default)
- `"want"` → No marker → Single(eng)
- `"biberon@s"` → Shortcut → Single(spa) (secondary language)
- `"and"` → No marker → Single(eng)
- `"croissant@s:fra"` → Explicit → Single(fra)
- `"please"` → No marker → Single(eng)

---

## 4. Batch Payload: Rust → Python

### Current Implementation (Semantic Resolution)

When building morphosyntax batch payloads, Rust:
1. Resolves each word's language using `resolve_word_language()`
2. Sends **semantic resolution**, not CHAT syntax

**Batch payload struct**:
```rust
#[derive(serde::Serialize, serde::Deserialize)]
struct MorphosyntaxBatchItem {
    words: Vec<String>,                          // Word texts
    terminator: String,                          // ".", "?", "!"
    special_forms: Vec<(
        Option<FormType>,                        // @c, @s, @b markers
        Option<LanguageResolution>,              // Resolved language
    )>,
    lang: LanguageCode,                          // Tier-level language
}
```

**Example JSON payload**:
```json
{
  "words": ["I", "want", "biberon", "please"],
  "special_forms": [
    [null, null],                              // "I" - no marker, primary lang
    [null, null],                              // "want"
    ["S", {"Single": "spa"}],                  // "biberon@s" → resolved to Spanish
    [null, null]                               // "please"
  ],
  "lang": "eng"                                // Utterance language
}
```

**What Python receives** (semantics, not syntax):
- `"spa"` - resolved language code, not `Shortcut` enum variant
- `{"Multiple": ["eng", "fra"]}` - multiple languages, not `+` syntax
- `{"Ambiguous": ["eng", "spa"]}` - ambiguous, not `&` syntax

### Why Semantic Resolution?

**Problem with sending syntax**:
Python would receive `Shortcut` and need to:
1. Know the @Languages header order
2. Know the current tier language
3. Implement the same resolution logic as Rust

**Solution with semantic resolution**:
Rust handles all resolution complexity once; Python receives clean language codes ready for routing.

---

## 5. Current Morphosyntax Behavior

### Per-Utterance Language Routing ✅ IMPLEMENTED

**Mechanism**: `[- lang]` directive
**Routing**: Group utterances by `lang` field, send entire batches to appropriate Stanza model

```python
# Python batch callback groups by language
by_lang = defaultdict(list)
for item in batch:
    by_lang[item["lang"]].append(item)

# Route to language-specific models
for lang_code, items in by_lang.items():
    if lang_code == "spa":
        results = stanza_es.process(items)
    elif lang_code == "fra":
        results = stanza_fr.process(items)
    # ...
```

**Status**: Fully working since 2026-02-15.

### Per-Word Language Routing ❌ NOT IMPLEMENTED (L2|xxx placeholder)

**Current behavior** (Rust post-processing):
```rust
for (mor, (form_type, resolved_lang)) in mors.iter_mut().zip(special_forms.iter()) {
    // If word has ANY language marker, replace with L2|xxx
    if resolved_lang.is_some() {
        word.part_of_speech = PartOfSpeech::new("L2");
        word.stem = MorStem::new("xxx");
        continue;  // Skip Stanza result
    }
    // ...
}
```

**Why L2|xxx?**
- English Stanza model doesn't know Spanish/French morphology
- Wrong to include English model's hallucinated morphology for foreign words
- `L2|xxx` is a safe placeholder that signals "foreign word"

**What's missing**:
- Python doesn't route `biberon@s:spa` to Spanish Stanza model
- Instead, it's processed by English model, then replaced with `L2|xxx`

Current limit:
- per-word language routing is not the current public runtime boundary
- foreign/code-switched words are handled conservatively instead

---

## 6. Special Cases and Edge Cases

### Tertiary Languages Need Explicit Markers

**Valid**:
```
@Languages:   eng, spa, fra

*CHI: I want croissant@s:fra .
              ^^^^^^^^^^^^^^^
              Explicit @s:fra marker required
```

**Invalid** (error E244):
```
@Languages:   eng, spa, fra
*CHI: [- fra] je veux croissant@s .
                          ^^^^^^^
                          @s shortcut not allowed in tertiary tier
```

**Why**: `@s` shortcut only resolves to **secondary** language (2nd position). Tertiary languages must use explicit `@s:lang`.

### Multiple/Ambiguous Languages

**Multiple** (`+`): Word is valid in ALL listed languages
```
*CHI: I want cafe@s:eng+fra .
             ^^^^
             English "café" or French "café" - both valid
```

**Ambiguous** (`&`): Unclear which language the word belongs to
```
*CHI: no@s:eng&spa quiero .
      ^^
      English "no" or Spanish "no"? Ambiguous.
```

**Validation**: Both forms require the word to be valid in ALL listed languages.

### No @Languages Header

If `@Languages` is missing:
- Inferred from `@ID` headers (first 3-letter language code)
- Falls back to primary language passed to processing function

---

## 7. Validation and Errors

### Language Resolution Errors

| Error | Trigger | Example |
|-------|---------|---------|
| **E244** | @s shortcut in tertiary language tier | `[- fra] word@s` when fra is 3rd+ language |
| **E361** | Invalid language code | `word@s:xyz` (xyz not in ISO 639-3) |
| **Unresolved** | No language context available | Word with @s but no @Languages header |

### Validation Context

```rust
let (resolved, errors) = resolve_word_language(
    &word,
    tier_language,
    &declared_languages,
);
```

Errors are collected during resolution and reported via the validation system.

---

## 8. Summary Tables

### Language Scope Levels

| Level | Syntax | Scope | Data Model Field |
|-------|--------|-------|------------------|
| **File** | `@Languages: eng, spa` | All utterances | `Header::Languages { codes }` |
| **Tier** | `[- spa]` | Single utterance | `MainTierContent.language_code` |
| **Word** | `@s:spa` | Single word | `Word.lang` |

### Word Language Markers

| Marker | Meaning | Syntax Enum | Resolved Enum | Example |
|--------|---------|-------------|---------------|---------|
| (none) | Tier default | `None` | `Single(tier_lang)` | `hello` |
| `@s` | Secondary lang | `Shortcut` | `Single(spa)` | `biberon@s` |
| `@s:spa` | Explicit | `Explicit(spa)` | `Single(spa)` | `biberon@s:spa` |
| `@s:eng+fra` | Multiple | `Multiple([eng,fra])` | `Multiple([eng,fra])` | `cafe@s:eng+fra` |
| `@s:eng&spa` | Ambiguous | `Ambiguous([eng,spa])` | `Ambiguous([eng,spa])` | `no@s:eng&spa` |

### Morphosyntax Processing Status

| Feature | Scope | Status | Implementation |
|---------|-------|--------|----------------|
| **Per-utterance routing** | `[- lang]` | ✅ Implemented | Rust batches by tier language, Python routes to Stanza |
| **Per-word routing** | `@s:lang` | ❌ Not implemented | Rust sends semantic resolution, Python ignores, replaces with L2\|xxx |

---

## 9. Code References

### Rust (talkbank-model)

- **Language syntax**: `model/content/word/language_marker.rs`
- **Resolution**: `validation/word/language/resolve.rs`
- **Header**: `model/header/header_enum/header.rs`
- **Tier content**: `model/content/main_tier.rs`

### Rust (batchalign-core)

- **Batch payload**: `lib.rs` (MorphosyntaxBatchItem struct)
- **Resolution invocation**: `lib.rs:880-897` (resolve_word_language calls)
- **L2|xxx replacement**: `lib.rs:995-1003` (post-processing)

### Python (stateless inference only)

- **Inference function**: `inference/morphosyntax.py:batch_infer_morphosyntax()` — calls Stanza `nlp()`, returns raw output
- **Per-utterance grouping**: Groups by `lang` field, routes to Stanza models

---

## 10. Future Work (Low Priority)

**Per-word language routing** (see `docs/per-word-language-switching-ideal-approach.md`):
1. Python checks `resolved_lang` in `special_forms`
2. Extracts code-switched words, sends to target language Stanza
3. Merges morphology back into primary language utterance
4. Requires careful dependency graph merging

**Complexity**: High - needs dual-pass analysis and cross-model merging.

**Current approach**: `L2|xxx` is safe, simple, and correct (doesn't mislead with wrong morphology).

---

## References

- CHAT Manual: https://talkbank.org/0info/manuals/CHAT.html
- ISO 639-3 Language Codes: https://iso639-3.sil.org/
- Related docs: `per-word-language-divergence.md`, `per-word-language-switching-ideal-approach.md`
