# Word Filtering

**Status:** Current
**Last updated:** 2026-05-23 18:52 EDT

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

By default, word matching is case-insensitive (`the` matches `The`, `THE`, `the`).

The CLAN `+k` flag (`--case-sensitive` after rewriting) is **fully landed** across the search/frequency family (FREQ, KWAL, VOCD, COMBO, FREQPOS, DIST, MAXWD): FREQ via `WordFilter`-driven pattern matching plus case-preserving frequency-table keying (so `Want`, `want`, `WANT` become three distinct entries); KWAL via keyword and word comparison both verbatim instead of via `NormalizedWord`'s lowercasing pass; VOCD via the same pattern-matching layer plus the D-statistic token stream skipping its default `to_lowercase` so case variants count as distinct types in the lexical-diversity calculation; COMBO via `SearchExpr::parse_with_case` preserving case in the stored terms and the word stream populating via `cleaned_text()`; FREQPOS, DIST, and MAXWD via case-preserving key derivation in `process_utterance` (MAXWD's unique-length and exclude-length filters then count case variants as distinct words). Other commands (MLU/MLT/WDLEN/WDSIZE/CHAINS/CODES) inherit `+k` from `cutt.cpp::mainusage` but it's a semantic no-op since they don't word-match. Per-command status lives in each command page under `clan-reference/commands/`. See also [`flag-translation.md`](../getting-started/flag-translation.md).

## What counts as a "word"

Word matching uses the same countable-word logic as other commands:
- Regular words and proper nouns match
- Untranscribed markers (`xxx`, `yyy`, `www`) do not match
- Zero words (`0word`) do not match
- Fillers and fragments (`&-um`, `&~frag`) do not match
- Events (`&=laughs`) do not match
