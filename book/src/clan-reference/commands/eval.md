# EVAL -- Language Sample Evaluation

**Status:** Current
## Purpose

Comprehensive morphosyntactic analysis computing lexical diversity, grammatical category counts, error rates, and MLU. EVAL was originally designed for clinical evaluation of adult aphasic speech samples (Saffran, Berndt & Schwartz, 1989) and produces a detailed profile of morphosyntactic abilities.

Requires a `%mor` dependent tier for morpheme-level metrics. Word-level metrics are computed from the main tier regardless.

See the [CLAN manual](https://talkbank.org/0info/manuals/CLAN.html#_Toc87376473) for the original EVAL command specification.

## Usage

```bash
chatter clan eval file.cha
chatter clan eval --speaker CHI file.cha
chatter clan eval --format json file.cha
```

## Options

| Option | CLAN Flag | Description |
|--------|-----------|-------------|
| `--speaker <CODE>` | `+t*CHI` | Include speaker |
| `--exclude-speaker <CODE>` | `-t*CHI` | Exclude speaker |
| `--format <FMT>` | -- | Output format: text, json, csv, clan |

EVAL has no command-specific flags beyond the shared `CommonAnalysisArgs`
set; the per-speaker normative-comparison aspect lives in EVAL-D
(DementiaBank norms) and KIDEVAL (child norms), invoked as separate
subcommands.

## Metrics

EVAL produces a comprehensive profile per speaker:

### Lexical measures
- **Utterances**: Total utterance count
- **Total words**: All countable words
- **NDW**: Number of different words (types)
- **TTR**: Type-token ratio (types / tokens)

### MLU
- **MLU-w**: Mean length of utterance in words
- **MLU-m**: Mean length of utterance in morphemes (from %mor)

### Part-of-speech counts (from %mor)
- Nouns, verbs, auxiliaries, modals
- Prepositions, adjectives, adverbs
- Conjunctions, determiners, pronouns

### Inflectional morphology (from %mor)
- Plurals (`PL`)
- Past tense (`PAST`)
- Present participle (`PRESP`)
- Past participle (`PASTP`)

### Error and ratio measures
- **Word errors**: Count of `[*]` markers
- **Open/closed class ratio**: Content words vs function words

## Differences from CLAN

### Word and morpheme identification

Uses AST-based `is_countable_word()` and typed POS categories instead of CLAN's string-prefix matching. POS classification operates on structured `MorWord` types rather than parsing POS tag strings at analysis time.

### Error extraction

`[*]` error markers are extracted from parsed AST annotations (the `ErrorMarker` node type) rather than raw text pattern matching. This ensures accurate counting even with complex nested annotations.

### Output formats

Supports text, JSON, and CSV. CLAN produces text only. JSON output provides structured access to all metrics for programmatic use.

### Golden test parity

Verified against CLAN C binary output.

## EVAL-D variant

`chatter clan eval-d` is identical to EVAL but uses DementiaBank protocol norms instead of AphasiaBank norms for normative comparison. See [EVAL-D](eval-d.md) for details.
