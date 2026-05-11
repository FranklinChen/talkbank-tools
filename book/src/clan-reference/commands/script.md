# SCRIPT — Compare Utterances to a Template

**Status:** Current
## Purpose

Compares subject CHAT data against an ideal template file to compute accuracy metrics: words produced vs. expected, correct matches, omissions (in template but not produced), and additions (produced but not in template). Useful for evaluating scripted language samples such as picture descriptions or story retells.

See the [CLAN manual](https://talkbank.org/0info/manuals/CLAN.html#_Toc220409234) for the original SCRIPT command specification.

## Usage

```bash
chatter clan script file.cha --template template.cha
chatter clan script corpus/ --template template.cha --speaker CHI
```

## Algorithm

1. Parse the template CHAT file and build a word frequency map (ideal counts)
2. For each subject utterance, accumulate word frequency counts
3. At finalization, compute per-word matches (minimum of ideal and actual), omissions, and additions

## Options

| Option | CLAN flag | Description |
|--------|-----------|-------------|
| `--template <path>` | — | Path to template/script file (required) |
| `--speaker <code>` | `+t*CODE` | Restrict to specific speaker |
| `--format <fmt>` | — | Output format: text, json, csv |

## Output

Per file:

- Words produced by subject
- Words expected from template
- Correct words (matched)
- Omitted words (in template but not produced)
- Added words (produced but not in template)
- Percentage correct

Overall totals across all files.

## Differences from CLAN

- Template file is parsed into a typed AST (not raw text comparison)
- Word matching uses `NormalizedWord` for case-insensitive comparison
- Omissions and additions are computed from frequency maps rather than positional alignment, which may produce different results when word order matters
- Output supports text, JSON, and CSV formats
- **Golden test parity**: Verified against CLAN C binary output
