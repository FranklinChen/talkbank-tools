# Tier Filtering

**Status:** Current
Tier filters control which dependent tiers are included in command output. This primarily affects commands like KWAL and COMBO that echo matching utterances with their dependent tiers.

## Include tiers

Show only specific dependent tiers:

```bash
chatter clan kwal --tier mor --include-word "want" file.cha
```

CLAN equivalent: `+t%mor`

This outputs matching utterances with only the `%mor` tier, omitting `%gra`, `%pho`, and other tiers.

## Exclude tiers

Hide specific dependent tiers from output:

```bash
chatter clan kwal --exclude-tier gra file.cha
```

CLAN equivalent: `-t%gra`

## Common dependent tiers

| Tier | Full name | Content |
|------|-----------|---------|
| `mor` | Morphology | POS tags and lemmas: `noun\|dog-PL` |
| `gra` | Grammar | Dependency relations: `1\|3\|NSUBJ` |
| `pho` | Phonology | Phonological transcription |
| `flo` | Fluent output | Simplified main-tier text |
| `cod` | Codes | Researcher-assigned codes |
| `mod` | Model | Target pronunciation |
| `ret` | Retrace | Copy of main tier |

## Notes

- Tier names are specified without the `%` prefix
- Tier filtering does not affect the analysis itself (e.g., MLU still counts from `%mor` even if `%mor` is excluded from display)
- Main speaker tiers (`*CHI:`, `*MOT:`) are always included
