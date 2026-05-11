# DSS -- Developmental Sentence Scoring

**Status:** Current
## Purpose

Assigns point values to utterances based on grammatical complexity, using a configurable rule file that defines pattern-matching rules for morphosyntactic categories. DSS is a clinical tool developed by Laura Lee and Susan Canter for evaluating children's grammatical development by scoring complete sentences on eight grammatical categories.

Scoring requires a `%mor` dependent tier on each utterance. Utterances without `%mor` are silently skipped.

See the [CLAN manual](https://talkbank.org/0info/manuals/CLAN.html#DSS_Command) for the original DSS command specification and the full rule set.

## Usage

```bash
chatter clan dss --speaker CHI file.cha
chatter clan dss --rules english.scr file.cha
chatter clan dss --max-utterances 100 file.cha
chatter clan dss --format json file.cha
```

## Options

| Option | Description |
|--------|-------------|
| `--speaker <CODE>` | Include speaker |
| `--exclude-speaker <CODE>` | Exclude speaker |
| `--rules <PATH>` | Custom DSS rules file (.scr) |
| `--max-utterances <N>` | Maximum utterances to score (default: 50) |
| `--format <FMT>` | Output format: text, json, csv, clan |

## Scoring Categories

DSS scores utterances on eight grammatical categories:

1. **Indefinite pronouns / noun modifiers** (it, this, that)
2. **Personal pronouns** (I, me, my, mine, you, your)
3. **Main verbs** (uninflected, copula, auxiliary)
4. **Secondary verbs** (non-finite: infinitives, gerunds, participles)
5. **Negation** (no, not, can't, don't)
6. **Conjunctions** (and, but, or, if, because)
7. **Interrogative reversals** (is he, can you, do they)
8. **Wh-questions** (who, what, where, when, why, how)

Each category earns 1-8 points based on developmental complexity. A **sentence point** is awarded if the utterance is a complete grammatical sentence.

## Algorithm

1. Parse each utterance's `%mor` tier for POS-tagged morphemes
2. Match morpheme patterns against category rules
3. For each category, award points for the highest-scoring matched pattern
4. Award sentence point for complete sentences (heuristic: subject + verb POS)
5. Sum across categories + sentence point = utterance score
6. DSS = mean score across scored utterances

## Output

Per-speaker DSS total with per-category breakdown and per-utterance scores.

## Differences from CLAN

### Built-in rules

The default rules are a simplified subset of the canonical DSS rule set (10 categories). For full clinical scoring, supply a complete `.scr` rules file via `--rules`. When a rules file is not provided, DSS produces approximate scores suitable for screening but not clinical reporting.

### Sentence-point assignment

Uses a heuristic (presence of subject + verb POS tags in `%mor`) rather than full syntactic analysis. This may under-award sentence points for syntactically complex but structurally unusual utterances.

### Maximum utterances

Defaults to 50 per speaker (configurable via `--max-utterances`). CLAN also defaults to 50 but the implementation differs in how utterances are selected when the sample exceeds the maximum.

### Output formats

Supports text, JSON, and CSV. CLAN produces text only.

### Golden test parity

Verified against CLAN C binary output.
