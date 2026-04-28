# Why Replace Python CHAT Handling with Rust

**Status:** Completed (2026-02)

---

## The Problem

Batchalign's morphosyntax pipeline (`morphotag`) is the most-used command in the
TalkBank processing workflow, run against hundreds of thousands of CHAT files. The
existing Python implementation has several areas where the Rust rewrite improves on the original:

1. **Fragile text manipulation.** The Python code converts CHAT transcripts to flat
   strings, sends them through Stanza for NLP analysis, then uses expensive string
   alignment (O(n*m) dynamic programming) to map Stanza's output back to the original
   transcript. This "flatten → process → realign" pipeline is the source of most
   morphotag bugs — small tokenization differences between Stanza and CHAT cause
   misaligned %mor/%gra tiers.

2. **60+ lines of `.replace()` calls.** Python's `annotation_clean()` function strips
   CHAT annotations character-by-character to produce "clean" text for Stanza. This
   is destructive — it loses structural information — and every new annotation type
   requires adding more `.replace()` calls.

3. **Retokenization is a separate ~120-line code path** that operates at the character
   level, building a "backplate" of alignable/non-alignable chunks and applying ~15
   regex cleanups. It's a complex subsystem with many language-specific cases.

4. **No structural understanding of CHAT.** Python treats CHAT as flat text. It can't
   distinguish between a word inside a retrace group (`<I want> [/]`) and a regular
   word without string-level pattern matching. Every edge case requires a new regex.

---

## The Solution: Rust CHAT AST

We've built a Rust implementation that parses CHAT into a strongly-typed Abstract
Syntax Tree (AST) and operates on it structurally. The key insight: **when your data
model matches the format's structure, you don't need string hacking.**

### How It Works

```
              Rust                           Python (ML only)
         ┌────────────┐                   ┌──────────────┐
CHAT  →  │ Parse AST  │  → words JSON →  │  Stanza NLP  │
file     │ Extract    │  ← mor/gra JSON ← │  (unchanged)  │
         │ Inject     │                   └──────────────┘
         │ Serialize  │  → CHAT file
         └────────────┘
```

Rust parses the CHAT file once into a proper AST, extracts the words that need NLP
analysis by traversing the tree (no string manipulation), sends them to Python's Stanza
via a callback, receives %mor/%gra analysis, attaches it directly to the AST nodes, and
serializes back to CHAT. The callback boundary means Python's role is strictly limited
to neural network inference.

---

## Why It's Safe

### 150 automated tests across two languages

| Suite | Tests | What's Verified |
|-------|-------|----------------|
| Rust unit tests | 65 | Every module: parsing, alignment, injection, retokenization |
| Python integration tests | 27 | End-to-end pipeline with test doubles |
| Python-Rust equivalence tests | ~20 | Same inputs produce same word extractions |
| DP alignment equivalence tests | ~29 | Rust alignment matches Python alignment exactly |
| CHAT round-trip tests | ~9 | Parse → modify → serialize produces valid CHAT |

### No mocks — real test doubles

Following the project's strict no-mock policy, all tests use lightweight alternate
implementations (e.g., `FakeStanzaNLP` that returns predetermined analyses for known
words). This means tests exercise the actual code paths, not mocked interfaces.

### Boundary-based architecture makes testing natural

The callback pattern creates a clean boundary: Rust sends words, Python returns analysis.
Tests can verify each side independently:
- **Rust side:** Given this CHAT and this callback response, is the output correct?
- **Python side:** Given these words, does the callback produce the right %mor/%gra?
- **End-to-end:** Does CHAT → Rust → callback → CHAT produce valid output?

### Python fallback preserves backward compatibility

If the Rust extension isn't available (e.g., on systems where it hasn't been built),
the code automatically falls back to the existing Python implementation. No user-facing
changes are required. This means deployment is incremental — we can enable Rust on one
system at a time.

### Round-trip verification

Every test that modifies a CHAT file also verifies the output re-parses correctly.
The Rust parser and serializer are inverses — if the parser can't read its own
serializer's output, the test fails.

### Structural correctness by construction

The AST approach eliminates entire categories of bugs:

| Bug Class | Python (Text) | Rust (AST) |
|-----------|---------------|------------|
| Retrace confusion | Regex to detect `<...> [/]` | `UtteranceContent::Retrace` — dedicated variant, structurally distinct |
| Special form handling | String search for `@c`, `@s` | `Word.form_type` field — parsed once, always available |
| Word boundary errors | Character-level offset tracking | Words are tree nodes; boundaries are implicit |
| Tier count mismatch | Count words in string, count mor items in string, hope they match | Traverse same tree for extraction and injection — count is always consistent |
| Annotation stripping | `.replace()` for each character class | AST traversal selects the right nodes; annotations are never destroyed |

---

## Why It's Efficient

### Rust is dramatically faster for text processing

The Hirschberg DP alignment (used for character-level mapping) runs **orders of
magnitude faster** in Rust:

| Input Size | Python | Rust |
|-----------|--------|------|
| n=1,000 | ~500ms | <1ms |
| n=5,000 | minutes | <1s |

For morphotag, the bottleneck is Stanza's neural network inference (unchanged). But
the per-utterance Rust overhead (parsing, extraction, injection, serialization) is
negligible compared to Python's text processing. For large files with thousands of
utterances, eliminating Python's string manipulation overhead adds up.

### Less code to maintain

| Component | Python | Rust |
|-----------|--------|------|
| Word extraction | `annotation_clean()` + lexer filtering: ~100 lines of `.replace()` and regex | `extract.rs`: 263 lines of typed traversal |
| %mor/%gra construction | String concatenation in `morphoanalyze()`: scattered across ~200 lines | `mor_parser.rs` + `inject.rs`: 1,056 lines, self-contained |
| Retokenization | ~120 lines of character-level backplate + 15 regex fixups | `retokenize.rs`: 938 lines, operates on AST nodes |

The Rust code is longer in raw lines but **each line does one thing**. The Python code
is shorter but relies on implicit knowledge about string formats, character positions,
and annotation conventions that are never checked at compile time.

### The AST is shared infrastructure

The `talkbank-model` and `talkbank-parser` crates aren't just for morphotag.
They power the TalkBank LSP (Language Server Protocol) server, the CHAT validation
tooling, and will be the foundation for migrating forced alignment and utterance
segmentation to Rust. Every pipeline that currently uses `annotation_clean()` will
benefit from the same AST.

---

## What's Been Migrated

### Complete

- **Standard morphosyntax** (`retokenize=false`): CHAT → extract words → Stanza callback → inject %mor/%gra → CHAT
- **Retokenized morphosyntax** (`retokenize=true`): Same, plus AST restructuring to match Stanza's tokenization
- **Special forms** (`@c`, `@s`, `@b`): Extracted from AST, passed in callback, restored after analysis
- **Multi-language skip** (`skipmultilang`): Utterances with `[- lang]` override correctly skipped
- **Progress reporting**: Per-utterance progress callbacks work through Rust
- **Caching**: Handled at the Python callback level (transparent to Rust)

### No Python Fallback

The Rust AST path is the only code path. The former Python `morphoanalyze()`
implementation and its `_process_python()` fallback have been removed. The
`batchalign_core` Rust extension is required.

---

## Completed Follow-Ups

All follow-up steps from the original decision have been completed:

1. **Production testing against large corpus** — Completed. Rust pipeline validated
   against existing annotations.
2. **Forced alignment migration** — Completed. FA uses the same AST + callback
   pattern via `inference/fa.py`.
3. **Utterance segmentation migration** — Completed. Utseg uses the same pattern
   via `inference/utseg.py`.
4. **Python fallback removed** — The Python `morphoanalyze()` code has been removed.
   All morphosyntax processing goes through the Rust AST path.

---

## Summary

The Rust morphosyntax pipeline is safer (AST eliminates string-hacking bugs), faster
(orders of magnitude for DP alignment), better tested (150 tests across Rust and Python),
and architecturally cleaner (callback pattern isolates ML from text processing). The
Python fallback ensures zero risk during rollout, and the underlying AST infrastructure
is reusable for every TalkBank processing pipeline.
