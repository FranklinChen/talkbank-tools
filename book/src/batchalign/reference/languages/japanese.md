# Japanese Language Support

**Status:** Current
**Last updated:** 2026-05-20 20:17 EDT

Japanese (`jpn`) uses Stanza's `combined` package and the retokenize system
for word boundary handling. This page covers the full picture: what works,
what doesn't, and what's planned.

## Quick Reference

| Pipeline Stage | Japanese-Specific Behavior |
|---------------|---------------------------|
| ASR | Whisper (default), no Japanese-specific alternatives |
| Text normalization | None, Japanese characters passed through as-is |
| Number expansion | Chinese number system (chatter's traditional-script CJK numerals) |
| Retokenize | Stanza `combined` package merges/splits CJK tokens |
| Morphosyntax | Stanza `ja` with forced `combined` package for all processors |
| MWT | Excluded, Japanese has no contractions |
| Forced alignment | Wave2Vec MMS (standard, no language-specific preprocessing) |

## Stanza Configuration

Japanese uses two distinct Stanza configurations depending on `--retokenize`:

| Property | Keep-Tokens (default) | Retokenize |
|----------|----------------------|------------|
| Stanza tokenizer | Bypassed (`pretokenized=True`) | Runs (`no_ssplit=True`) |
| Package | `combined` (all 4 processors) | `combined` (all 4 processors) |
| Word boundaries | Preserved from CHAT | Stanza may merge/split |
| MWT processor | Not loaded | Not loaded |

The `combined` package is forced for all four processors (`tokenize`, `pos`,
`lemma`, `depparse`) because it's trained jointly. Using `default` for any
processor would load a mismatched model.

### Why `combined`?

Standard Stanza packages train processors independently. Japanese `combined`
trains all four jointly, which is critical because Japanese tokenization,
POS tagging, and dependency parsing are interdependent (word boundaries affect
POS which affects dependencies).

## Retokenize Behavior

With `--retokenize`, Stanza's neural tokenizer may:
- **Merge** adjacent CHAT words into one token (e.g., ふ + す → ふす)
- **Split** one CHAT word into sub-tokens

The Rust retokenize module at
`crates/batchalign-transform/src/retokenize.rs:195::retokenize_utterance`
handles the AST rewrite using the same character-level span mapping
used for English contractions and CJK word segmentation. Sibling
helpers live under `crates/batchalign-transform/src/retokenize/`
(`parse_helpers.rs`, `rebuild.rs`).

## Known Limitations

### No Japanese-specific ASR engine
Japanese uses Whisper like most other languages. There are no
Japanese-specific ASR alternatives (unlike Cantonese which has Tencent,
Aliyun, and FunASR options).

### Stanza `combined` package quality
The `combined` package is trained on the Japanese UD treebank, which is
based on formal written Japanese. Performance may differ on:
- Spoken/colloquial Japanese
- Child speech
- Code-mixed Japanese-English

### Whitespace artifacts in Stanza output
Stanza sometimes produces whitespace artifacts in Japanese lemmas and POS
tags. These require language-specific cleanup in the Rust POS mapping layer.
See the [Japanese Morphosyntax Pipeline](../japanese-morphosyntax.md) for
detailed whitespace handling documentation.

## Verified Behavior

- Retokenize split-and-merge: covered by inline `#[cfg(test)]` tests
  in `crates/batchalign-transform/src/retokenize.rs` (e.g.,
  `deterministic_mapping_succeeds_for_split_and_merge` at `:287`).
- Stanza `combined` package loads correctly for `ja`
  (`batchalign/worker/_stanza_loading.py:196-207`).
- MWT exclusion: `should_request_mwt()` reports `has_mwt=False` for
  `ja`; the retokenize/utseg path uses a local
  `mwt_exclude = {"zh", "ja", "ko", "th", "vi", "my"}` set at
  `batchalign/worker/_stanza_loading.py:310`. The global
  `_MWT_EXCLUSION` was retired alongside `MWT_LANGS` (see
  [Stanza Limitations Defect 5](../stanza-limitations.md)).

## Open Questions

1. **Would a Japanese-specific ASR model improve accuracy?** Whisper handles
   Japanese reasonably but a dedicated model might improve CER.
2. **How does retokenize affect timing?** When words are merged/split, existing
   `%wor` timing bullets become stale. Is this communicated clearly to users?

## Detailed Reference

For the complete Stanza configuration details, POS mapping rules, verb form
overrides, and whitespace artifact handling, see:
[Japanese Morphosyntax Pipeline](../japanese-morphosyntax.md)

## Source Files

| File | Role |
|------|------|
| `batchalign/worker/_stanza_loading.py` | Japanese `combined` package selection (`:196-207`); MWT eligibility via `should_request_mwt()` |
| `batchalign/inference/morphosyntax.py` | Stanza inference (shared with all languages) |
| `crates/batchalign-transform/src/retokenize.rs` + `crates/batchalign-transform/src/retokenize/{parse_helpers,rebuild}.rs` | AST rewrite for merged/split tokens |
| `crates/batchalign-transform/src/morphosyntax/lang_ja.rs` | POS mapping with Japanese-specific rules |
