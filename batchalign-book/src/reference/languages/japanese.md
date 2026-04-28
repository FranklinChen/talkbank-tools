# Japanese Language Support

**Status:** Current
**Last updated:** 2026-03-23 12:15 EDT

Japanese (`jpn`) uses Stanza's `combined` package and the retokenize system
for word boundary handling. This page covers the full picture: what works,
what doesn't, and what's planned.

## Quick Reference

| Pipeline Stage | Japanese-Specific Behavior |
|---------------|---------------------------|
| ASR | Whisper (default), no Japanese-specific alternatives |
| Text normalization | None — Japanese characters passed through as-is |
| Number expansion | Chinese number system (`num2chinese` with simplified script) |
| Retokenize | Stanza `combined` package merges/splits CJK tokens |
| Morphosyntax | Stanza `ja` with forced `combined` package for all processors |
| MWT | Excluded — Japanese has no contractions |
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

The Rust retokenize module (`retokenize/mod.rs`) handles the AST rewrite
using the same character-level span mapping used for English contractions
and CJK word segmentation.

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

- Retokenize N:1 merge (ふ + す → ふす): tested in `retokenize/tests.rs`
- Stanza `combined` package loads correctly for `ja`
- MWT exclusion: Japanese is in `_MWT_EXCLUSION` set

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
| `worker/_stanza_loading.py` | Japanese `combined` package selection, MWT exclusion |
| `inference/morphosyntax.py` | Stanza inference (shared with all languages) |
| `crates/batchalign-chat-ops/src/retokenize/` | AST rewrite for merged/split tokens |
| `crates/batchalign-chat-ops/src/nlp/` | POS mapping with Japanese-specific rules |
