# IPSYN -- Index of Productive Syntax

**Status:** Current
## Purpose

Computes a syntactic complexity score by awarding points for distinct syntactic structures observed in a child's utterances. Each structure type (rule) can earn at most 2 points -- one per distinct utterance in which the structure appears. The total across all rules yields the IPSyn score.

See the [CLAN manual](https://talkbank.org/0info/manuals/CLAN.html#_Toc220409276) for the original IPSYN command specification.

## Usage

```bash
chatter clan ipsyn file.cha
chatter clan ipsyn --speaker CHI file.cha
chatter clan ipsyn --rules ipsyn.rules file.cha
chatter clan ipsyn --max-utterances 100 file.cha
```

## Options

| Option | Description |
|--------|-------------|
| `--speaker <CODE>` | Include speaker |
| `--exclude-speaker <CODE>` | Exclude speaker |
| `--rules <PATH>` | Custom IPSYN rules file |
| `--max-utterances <N>` | Maximum utterances to analyze (default: 100) |
| `--format <FMT>` | Output format: text, json, csv, clan |

## Rule Categories

Rules are organized into four categories:

| Category | Code | Description | Example structures |
|----------|------|-------------|-------------------|
| Noun Phrase | **N** | Noun phrase complexity | Two-word NP, article+noun, possessive |
| Verb Phrase | **V** | Verb phrase complexity | Copula, auxiliary, modal, infinitive |
| Question | **Q** | Question formation | Yes/no, wh-question, tag question |
| Sentence | **S** | Sentence structure | Conjoined, embedded, relative clause |

The full English IPSyn has ~56 rules. The built-in default set provides a representative subset.

## Algorithm

1. For each utterance, serialize the `%mor` tier to text
2. Match each rule pattern against the serialized `%mor` content
3. For each rule, record the first two distinct utterances that match (max 2 points per rule)
4. Sum all rule scores across categories
5. Report total score and per-category subtotals

### Scoring example

If rule N1 ("Two-word NP") matches in utterances 3 and 7, it earns 2 points. If it only matches in utterance 3, it earns 1 point. If it never matches, 0 points.

## Output

Per-speaker IPSyn total score with per-category subtotals (N, V, Q, S) and optional per-rule detail.

## Differences from CLAN

### Rule set

The built-in rule set is a simplified subset. For full 56-rule coverage, supply the official IPSYN rules file via `--rules`.

### Pattern matching

Uses substring-based matching on the serialized `%mor` tier text rather than structured POS/morpheme matching. This produces equivalent results for most patterns but may differ for edge cases involving complex morphological structures.

### Maximum utterances

Defaults to 100 (matching CLAN). Configurable via `--max-utterances`.

### Output formats

Supports text, JSON, and CSV. CLAN produces text only.

### Golden test parity

Verified against CLAN C binary output.
