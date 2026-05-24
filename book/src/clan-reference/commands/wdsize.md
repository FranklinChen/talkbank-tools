# WDSIZE — Word Size Distribution

**Status:** Current
**Last updated:** 2026-05-23 15:00 EDT

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

## CLAN `+`-flag coverage audit

WDSIZE is an **analysis** command (banner-emitting). Sources:
`OSX-CLAN/src/clan/wdsize.cpp::usage`,
`crates/talkbank-clan/src/commands/wdsize.rs`,
`crates/talkbank-cli/src/cli/args/clan_commands.rs::Wdsize`.

(Status legend: same as [FREQ](./freq.md#status-legend).)

### WDSIZE-specific `+`-flags (from `wdsize.cpp::usage`)

| CLAN flag | Meaning | Chatter | Status | Notes |
|---|---|---|---|---|
| `+a` | Use `%mor` only (no fallback to main tier) | (default) | Done | chatter's default uses `%mor` stems; the `--main-tier` flag is the opt-out. The CLAN/chatter defaults match. |
| `+bS` | Add chars in `S` to morpheme-delimiter list | — | Missing | Same gap as WDLEN/MAXWD's `+bS`. |
| `-bS` | Remove chars from delimiter list (`-b` clears all) | — | Missing | |
| `+cS` | Clause-marker delimiter `S` | — | Missing | Same as MLU/MLT's `+c`. |
| `+wCN` | Include only words `C` (`>`, `<`, `=`) than `N` characters | `--length-filter <gt:N\|lt:N\|eq:N>` | Done | Landed 2026-05-23. Per-word length gate before the histogram accumulator. The rewriter intercepts `+w[>|<|=]N` under WDSIZE before the general `+wN` context-window arm (which doesn't apply to WDSIZE anyway). `LengthFilter::FromStr` parses the `<comparator>:<N>` value-arg. Pinned by `length_filter_greater_than`, `length_filter_less_than`, `length_filter_equal`, `length_filter_includes_predicate`, `length_filter_from_str_parses_rewriter_output`, plus three rewriter tests. End-to-end smoke: `+w>3` on `I want a Cookie .` drops the length-1 tokens, leaving lengths 4 and 6. |
| `--main-tier` (chatter extension) | Use main-tier words instead of `%mor` stems | `--main-tier` | Chatter-only | No CLAN analog. |

### Audit summary

| Bucket | Count |
|---|---|
| Done | 8 (default + 6 general inherited as documented on FREQ + `+wCN`) |
| Partial | 1 |
| Rewriter only | 4 |
| Missing | 3 |

WDSIZE shares its morpheme-delimiter and clause-marker gaps with
WDLEN/MAXWD/MLU/MLT — the cluster of commands that all read
`%mor` and would benefit from a shared "morpheme-delimiter
customization" feature. Filed as a Phase 1.7 cross-cutting
follow-up.
