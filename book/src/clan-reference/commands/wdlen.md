# WDLEN -- Word Length Distribution

**Status:** Current
## Purpose

Computes six distribution tables matching CLAN's output format. WDLEN provides detailed histograms of word and utterance lengths, useful for studying vocabulary complexity and utterance structure development.

See the [CLAN manual](https://talkbank.org/0info/manuals/CLAN.html#_Toc220409247) for the original WDLEN command specification.

## Usage

```bash
chatter clan wdlen file.cha
chatter clan wdlen --speaker CHI file.cha
chatter clan wdlen --format json file.cha
```

## Options

| Option | CLAN Flag | Description |
|--------|-----------|-------------|
| `--speaker <CODE>` | `+t*CHI` | Include speaker |
| `--exclude-speaker <CODE>` | `-t*CHI` | Exclude speaker |
| `--format <FMT>` | -- | Output format: text, json, csv |

## Six Distribution Sections

| Section | What it measures | Source |
|---------|-----------------|--------|
| 1. Word lengths in characters | Character count per word | Main tier |
| 2. Utterance lengths in words | Word count per utterance | Main tier |
| 3. Turn lengths in utterances | Utterances per turn | Main tier |
| 4. Turn lengths in words | Words per turn | Main tier |
| 5. Word lengths in morphemes | Morphemes per word (stem + Brown's suffixes) | `%mor` tier |
| 6. Utterance lengths in morphemes | Morphemes per utterance (POS + stem + Brown's suffixes) | `%mor` tier |

Each section shows a histogram (value -> count), mean, and total.

## Differences from CLAN

### Brown's morpheme rules

Sections 5 and 6 use distinct counting methods:

- **Section 5**: stem + Brown's suffix count (no POS tag counted). Clitic pairs (`~`) are merged as one word.
- **Section 6**: POS tag + stem + Brown's suffix count. POS is counted only for the main word (not clitics).

Brown's suffix strings: `PL`, `PAST`, `Past`, `POSS`, `PASTP`, `Pastp`, `PRESP` (same 7 as MLU).

### Character counting

CLAN strips apostrophes before counting character length. Our implementation matches this behavior.

### Speaker ordering

CLAN outputs speakers in reverse encounter order (an artifact of its C linked-list prepend pattern). Our implementation replicates this ordering for parity.

### XML footer

CLAN appends `</Table></Worksheet></Workbook>` XML tags at the end of output. Our implementation matches this for parity.

### Output formats

Supports text, JSON, and CSV. CLAN produces text only.

### Golden test parity

100% parity with CLAN C binary output.
