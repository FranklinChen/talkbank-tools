# MODREP — Model/Replica Comparison

**Status:** Current
## Purpose

Compares the model (target) pronunciation on the `%mod` tier with the actual (replica) pronunciation on the `%pho` tier, tracking word-by-word mappings between model forms and replica forms. This is used in phonological analysis to assess how closely a speaker's productions match the adult target forms.

See the [CLAN manual](https://talkbank.org/0info/manuals/CLAN.html#_Toc220409226) for the original MODREP command specification.

## Usage

```bash
chatter clan modrep file.cha
chatter clan modrep file.cha --speaker CHI
```

## Algorithm

1. For each utterance with both `%mod` and `%pho` tiers:
   - Extract word lists from both tiers (flattening groups)
   - Pair words positionally (model word N <-> replica word N)
   - Record each (model, replica) pair in a frequency map per speaker
2. Report per-speaker tables of model words with their replica variants and frequency counts

## Options

| Option | CLAN flag | Description |
|--------|-----------|-------------|
| `--speaker <code>` | `+t*CODE` | Restrict to specific speaker |
| `--format <fmt>` | — | Output format: text, json, csv |

## Output

Per-speaker listing of model words, each with its set of replica variants and their frequency counts, sorted alphabetically by model word.

## Differences from CLAN

- Model and replica extraction uses parsed `%mod` and `%pho` tier structures from the AST rather than raw text line parsing
- Word pairing operates on typed `PhoWord` content rather than string splitting
- Output supports text, JSON, and CSV formats (CLAN produces text only)
- Deterministic output ordering via sorted collections
- **Golden test parity**: Verified against CLAN C binary output
