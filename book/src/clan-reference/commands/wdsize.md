# WDSIZE — Word Size Distribution

**Status:** Current
Character-length histogram for word stems from the `%mor` tier.

## Usage

```bash
chatter clan wdsize file.cha
chatter clan wdsize corpus/ --speaker CHI
chatter clan wdsize file.cha --main-tier    # Use main tier words instead
chatter clan wdsize file.cha --format json
```

## What It Measures

WDSIZE counts the character length of each word stem extracted from the `%mor` tier and produces a histogram showing how many words of each length appear. By default it uses morphological stems (lemmas); with `--main-tier` it uses surface forms from the main tier.

## Output

Per speaker:
- Character-length histogram (length → count)
- Total words measured
- Mean word size in characters

## Differences from WDLEN

| Feature | WDSIZE | WDLEN |
|---------|--------|-------|
| Default source | `%mor` stems | Main tier words |
| Sections | 1 (character lengths only) | 6 (chars, words/utt, turns, morphemes) |
| Morpheme counting | No | Yes (sections 5-6) |

## Differences from CLAN

- Uses typed `MorTier` items with `MorWord.lemma` rather than raw string parsing
- Supports JSON and CSV output
- Falls back to main tier words when `%mor` is absent
