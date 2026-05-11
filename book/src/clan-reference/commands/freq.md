# FREQ -- Word Frequency

**Status:** Current
## Purpose

Counts word tokens and types and computes type-token ratio (TTR). The legacy manual describes `FREQ` as one of CLAN's most powerful and easiest-to-use programs, producing word-frequency counts and lexical-diversity measures over selected files and speakers.

In `talkbank-clan`, `FREQ` counts words on the main tier by default, or morphemes from the `%mor` tier when `--mor` is set.

See the [CLAN manual](https://talkbank.org/0info/manuals/CLAN.html#_Toc220409093) for the original FREQ command specification.

## Usage

```bash
chatter clan freq file.cha
chatter clan freq --speaker CHI file.cha
chatter clan freq --format json corpus/
chatter clan freq --mor file.cha
chatter clan freq --include-word "the" file.cha
```

> **`+k` / `--case-sensitive` is currently non-functional.** The
> legacy `+k` flag rewrites to `--case-sensitive` (see
> `crates/talkbank-clan/src/clan_args.rs:104`), but no `clap` field
> consumes that token in the current `Freq`/`CommonAnalysisArgs`
> structs, so passing it produces a parse error. Word matching is
> case-insensitive today.

## Options

| Option | CLAN Flag | Description |
|--------|-----------|-------------|
| `--speaker <CODE>` | `+t*CHI` | Include speaker |
| `--exclude-speaker <CODE>` | `-t*CHI` | Exclude speaker |
| `--include-word <WORD>` | `+s"word"` | Only count matching word |
| `--exclude-word <WORD>` | `-s"word"` | Skip matching word |
| `--gem <LABEL>` | `+g"label"` | Restrict to gem segment |
| `--range <START-END>` | `+z25-125` | Utterance range |
| ~~`--case-sensitive`~~ | ~~`+k`~~ | **Currently non-functional** — see callout above |
| `--format <FMT>` | -- | Output format: text, json, csv, clan |
| `--mor` | -- | Count morphemes from `%mor` tier instead of words from main tier |

## CLAN Equivalence

| CLAN command | Rust equivalent |
|---|---|
| `freq file.cha` | `chatter clan freq file.cha` |
| `freq +t*CHI file.cha` | `chatter clan freq file.cha --speaker CHI` |
| `freq +s"the" file.cha` | `chatter clan freq file.cha --include-word "the"` (case-sensitive matching not currently supported — see callout above) |
| `freq *.cha` | `chatter clan freq corpus/` |

## Output

Per-speaker frequency tables with:

- Word frequency counts (sorted by count descending, then alphabetically)
- Total types (unique words) and tokens (total words)
- TTR (type-token ratio = types / tokens)

### Example output (text)

```text
Speaker: CHI
  the       12
  I         8
  want      6
  a         5
  go        4
  ...
Types: 45
Tokens: 127
TTR: 0.354
```

### Example output (JSON)

```json
{
  "speakers": {
    "CHI": {
      "words": { "the": 12, "I": 8, "want": 6, ... },
      "types": 45,
      "tokens": 127,
      "ttr": 0.354
    }
  }
}
```

## Word Normalization

Words are grouped using `NormalizedWord`, which lowercases and strips compound markers (`+`) for counting purposes, while preserving the original CLAN display form (with `+`) for output. This means `wanna+go` and `Wanna+Go` are counted as the same word.

## Differences from CLAN

### Word identification

The legacy manual says `FREQ` ignores `xxx`, `www`, and words beginning with `0`, `&`, `+`, `-`, or `#` by default, and also ignores header and code tiers unless selected. CLAN implements much of this with character-level string-prefix matching:

```c
if (word[0] == '0') continue;     // omitted words
if (word[0] == '&') continue;     // fillers/nonwords
if (word[0] == '+') continue;     // terminators
```

Our implementation uses AST-based `is_countable_word()`, which checks semantic type rather than string prefixes. This is more precise -- a filler (`&-um`) and a phonological fragment (`&+fr`) have distinct semantic types in our model, even though CLAN lumps them together under the `&` prefix.

### Manual features not yet mirrored directly

The legacy manual documents several advanced `FREQ` workflows, including `+s@file` lexical-group lists, `%mor`/`%gra` combined search with `+d7`, and multilingual searches. Some of those behaviors are covered in `talkbank-clan` through broader filtering infrastructure, but the command chapter should not imply one-for-one flag parity unless explicitly implemented.

### Output ordering

Output is deterministic via sorted collections (count descending, then alphabetically). CLAN's ordering can vary across runs.

### Output formats

Supports text, JSON, and CSV formats. CLAN produces text only. Use `--format clan` for character-level CLAN-compatible output.

### Multi-file behavior

Results are merged across files by default (`+u` behavior). CLAN requires explicit `+u` flag. Use `chatter clan freq dir/` for recursive directory traversal (CLAN uses shell globs).

### Golden test parity

Verified against CLAN C binary output. 100% parity.
