# Mandarin Language Support

**Status:** Current
**Last updated:** 2026-03-23 12:15 EDT

Mandarin (`cmn`/`zho`) shares the Stanza `zh` model and Chinese number
expansion system with Cantonese, but has distinct word segmentation behavior
and no alternative ASR engines.

## Quick Reference

| Pipeline Stage | Mandarin-Specific Behavior |
|---------------|---------------------------|
| ASR | Whisper (default), no Mandarin-specific alternatives |
| Text normalization | None (Cantonese normalization is `yue`-only) |
| Number expansion | Chinese number system (`num2chinese` with simplified script for `zho`, traditional for `cmn`) |
| Word segmentation | Stanza neural tokenizer via `--retokenize` |
| Morphosyntax | Stanza Chinese (`zh-hans`) model, MWT excluded |
| Forced alignment | Wave2Vec MMS (standard) |

## Language Codes

| ISO 639-3 | Stanza | Notes |
|-----------|--------|-------|
| `cmn` (Mandarin) | `zh` | Standard Mandarin |
| `zho` (Chinese, generic) | `zh` | Maps to same Stanza model |

Both `cmn` and `zho` map to Stanza `zh` (which is `zh-hans` internally).

## Word Segmentation

Mandarin ASR output (from Whisper or Paraformer) may contain per-character
tokens without word boundaries — the same problem that affects Cantonese.

### The `--retokenize` Solution

```bash
batchalign3 morphotag --retokenize corpus/ -o output/ --lang cmn
```

This uses Stanza's neural Chinese tokenizer (`tokenize_pretokenized=False`)
to segment text into words before POS tagging. The tokenizer model is loaded
lazily on first `--retokenize` request and cached in worker state under key
`"{lang}:retok"`.

```mermaid
flowchart TD
    input["CHAT input\n(per-character tokens)"]
    retok{"--retokenize?"}
    lazy{"Retok pipeline\nloaded?"}
    load["load_stanza_retokenize_model()\n(_stanza_loading.py)"]
    stanza_retok["Stanza zh tokenizer\n(pretokenized=False)"]
    stanza_std["Stanza zh\n(pretokenized=True)"]
    rust["Rust retokenize module\n(retokenize/mod.rs)"]
    out_word["CHAT output\n(word-level tokens)"]
    out_char["CHAT output\n(per-char tokens)"]

    input --> retok
    retok -->|yes| lazy
    retok -->|no| stanza_std --> out_char
    lazy -->|no| load --> stanza_retok
    lazy -->|yes| stanza_retok
    stanza_retok --> rust --> out_word
```

### Segmentation Quality

Verified with real Stanza `zh` model (package `gsdsimp`):

| Input | Stanza Output | Correct? |
|-------|--------------|----------|
| 我去商店买东西 | 我 去 商店 买 东 西 | Mostly — groups 商店 but splits 东西 |

Stanza handles common compounds (商店 "store") correctly but may split
ambiguous compounds where individual characters have independent meanings
(东西 "things" → 东 "east" + 西 "west").

**For word count and MLU analysis, this is substantially better than
per-character tokenization but should not be treated as ground truth.**

## Number Expansion

Mandarin uses the Chinese number expansion system:

| Code | Script | Example |
|------|--------|---------|
| `zho` | Simplified | 5 → 五, 10000 → 一万 |
| `cmn` | Traditional | 5 → 五, 10000 → 一萬 |

## Morphosyntax

Mandarin uses Stanza's `zh-hans` model (for utterance segmentation) or `zh`
(for morphosyntax). MWT is excluded — Chinese has no contractions.

Default mode: `tokenize_pretokenized=True` (Stanza annotates existing word
boundaries without re-tokenizing).

## Known Limitations

### Stanza tokenizer is imperfect
The `zh` tokenizer is trained on the Chinese Treebank (Mandarin, formal text).
Performance may degrade on:
- Spoken/colloquial Mandarin
- Child speech
- Technical or domain-specific vocabulary
- Ambiguous compounds (东西, 大小, 多少)

### No Mandarin-specific ASR engine
Unlike Cantonese (which has Tencent, Aliyun, FunASR), Mandarin uses only
Whisper. A Mandarin-specific ASR engine could improve CER.

### No Cantonese normalization
Text normalization (simplified → HK traditional + domain replacements) only
runs for `yue`. Mandarin text passes through without character normalization.

## Verified Behavior

| What | Test | Result |
|------|------|--------|
| Stanza `zh` tokenizer segments words | `test_stanza_chinese_tokenizer_segments_multichar_words` | Groups 商店 correctly |
| `pretokenized=True` preserves chars | `test_stanza_pretokenized_true_preserves_chars` | 7 chars stay as 7 tokens |
| Language code mapping | `test_language_code_mapping` | cmn → zh, zho → zh |

## Open Questions

1. **Paraformer output format** — does Paraformer actually produce per-character
   tokens for Mandarin, or does it attempt some word segmentation? (Conflicting
   reports from users.)
2. **Child Mandarin speech** — how does Stanza's tokenizer perform on child
   language data?
3. **Would jieba be better than Stanza for Mandarin?** jieba is a widely-used
   Chinese word segmenter; comparison with Stanza's neural tokenizer would be
   informative.

## Source Files

| File | Role |
|------|------|
| `worker/_stanza_loading.py` | `load_stanza_retokenize_model()` for lazy `zh` retok pipeline |
| `inference/morphosyntax.py` | Mandarin retokenize path in `batch_infer_morphosyntax()` |
| `crates/batchalign-chat-ops/src/asr_postprocess/num2chinese.rs` | Chinese number expansion |
| `crates/batchalign-chat-ops/src/retokenize/` | AST rewrite (language-agnostic) |
