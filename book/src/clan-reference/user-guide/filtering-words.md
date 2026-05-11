# Word Filtering

**Status:** Current
Word filters restrict analysis to utterances containing (or not containing) specific words. Primarily useful with KWAL (keyword search) and COMBO (boolean search), but available on all commands.

## Include words

Only process utterances containing a specific word:

```bash
chatter clan kwal --include-word "the" file.cha
chatter clan combo --include-word "dog" --include-word "cat" file.cha
```

CLAN equivalent: `+s"the"`, `+s"dog" +s"cat"`

Multiple `--include-word` flags use OR logic by default: utterances containing *any* listed word are included.

## Exclude words

Skip utterances containing specific words:

```bash
chatter clan freq --exclude-word "um" --exclude-word "uh" file.cha
```

CLAN equivalent: `-s"um" -s"uh"`

## Case sensitivity

By default, word matching is case-insensitive (`the` matches `The`, `THE`, `the`). Use `--case-sensitive` for exact matching:

```bash
chatter clan kwal --include-word "The" --case-sensitive file.cha
```

CLAN equivalent: `+k`

## What counts as a "word"

Word matching uses the same countable-word logic as other commands:
- Regular words and proper nouns match
- Untranscribed markers (`xxx`, `yyy`, `www`) do not match
- Zero words (`0word`) do not match
- Fillers and fragments (`&-um`, `&~frag`) do not match
- Events (`&=laughs`) do not match
