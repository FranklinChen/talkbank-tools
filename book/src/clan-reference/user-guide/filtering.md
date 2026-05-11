# Filtering

**Status:** Current
The framework provides a unified filtering system shared by all commands. Filters restrict which utterances, speakers, tiers, and words are processed.

Multiple filters can be combined — they are applied as an AND (all must match for an utterance to be included).

```mermaid
flowchart TD
    utt["Each utterance"]
    speaker{"Speaker\nfilter?"}
    tier{"Tier\nfilter?"}
    word{"Word\nfilter?"}
    gem{"Gem\nfilter?"}
    range{"Range\nfilter?"}
    process["process_utterance()"]
    skip["Skip"]

    utt --> speaker
    speaker -->|pass| tier
    speaker -->|fail| skip
    tier -->|pass| word
    tier -->|fail| skip
    word -->|pass| gem
    word -->|fail| skip
    gem -->|pass| range
    gem -->|fail| skip
    range -->|pass| process
    range -->|fail| skip
```

## Filter types

| Filter | Flag | CLAN equivalent | Effect |
|--------|------|-----------------|--------|
| [Speaker](filtering-speakers.md) | `--speaker` / `--exclude-speaker` | `+t*` / `-t*` | Include/exclude by speaker code |
| [Tier](filtering-tiers.md) | `--tier` / `--exclude-tier` | `+t%` / `-t%` | Include/exclude dependent tiers |
| [Word](filtering-words.md) | `--include-word` / `--exclude-word` | `+s` / `-s` | Filter by word content |
| [Gem](filtering-gems.md) | `--gem` / `--exclude-gem` | `+g` / `-g` | Restrict to gem segments |
| [Range](filtering-range.md) | `--range` | `+z` | Utterance number range |
| ID | `--id-filter` | `+t@ID=` | Filter by @ID header fields |

## Examples

```bash
# Child speaker only, within a gem
chatter clan freq --speaker CHI --gem "story" file.cha

# All speakers except investigator, utterances 10-50
chatter clan mlu --exclude-speaker INV --range 10-50 file.cha

# Only utterances containing "the"
chatter clan kwal --include-word "the" file.cha
```
