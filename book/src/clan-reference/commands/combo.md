# COMBO -- Boolean Keyword Search

**Status:** Current
## Purpose

Searches for utterances matching boolean combinations of keywords. Supports AND (`+`) and OR (`,`) logic with case-insensitive substring matching. This is the primary search tool for finding utterances containing specific words or word combinations.

See the [CLAN manual](https://talkbank.org/0info/manuals/CLAN.html#_Toc220409095) for the original COMBO command specification.

## Usage

```bash
chatter clan combo -s "want+cookie" file.cha
chatter clan combo -s "want,milk" file.cha
chatter clan combo -s "want+cookie" --speaker CHI file.cha
```

## Options

| Option | Description |
|--------|-------------|
| `--speaker <CODE>` | Include speaker |
| `-s <EXPR>` | Search expression (repeatable; multiple `-s` flags combined with OR) |
| `--format <FMT>` | Output format: text, json, csv, clan |

## Search Syntax

- `+` between terms means AND (all terms must be present in the utterance)
- `,` between terms means OR (at least one term must be present)
- Terms are case-insensitive substring matches against countable words
- Multiple `-s` flags are combined with OR (any expression matching counts)
- AND takes precedence if both `+` and `,` appear in one expression

## CLAN Equivalence

| CLAN command | Rust equivalent |
|---|---|
| `combo +s"want^cookie" file.cha` | `chatter clan combo file.cha -s "want+cookie"` |
| `combo +s"want\|milk" file.cha` | `chatter clan combo file.cha -s "want,milk"` |
| `combo +s"want^cookie" +t*CHI file.cha` | `chatter clan combo file.cha -s "want+cookie" --speaker CHI` |

## Output

Each matching utterance with:

- Source filename
- Speaker code
- Full utterance text (CHAT format)
- Summary counts of matching vs. total utterances

## Differences from CLAN

- **Operator syntax**: CLAN uses `^` for AND and `\|` for OR; this implementation uses `+` and `,` respectively for shell-friendliness.
