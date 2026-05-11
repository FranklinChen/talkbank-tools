# Gem Filtering

**Status:** Current
Gem filters restrict analysis to utterances within `@Bg` / `@Eg` (Begin Gem / End Gem) boundaries. Gems are used to mark segments of a transcript — for example, a story retelling, a conversation about a specific topic, or a test activity.

## Include gems

Analyze only utterances within a labeled gem segment:

```bash
chatter clan freq --gem "story" file.cha
chatter clan mlu --gem "frog story" file.cha
```

CLAN equivalent: `+g"story"`

## Exclude gems

Skip utterances within specific gem segments:

```bash
chatter clan freq --exclude-gem "warmup" file.cha
```

CLAN equivalent: `-g"warmup"`

## How gems work in CHAT files

Gem boundaries are marked with header lines:

```text
@Bg:	story
*CHI:	once upon a time .
*CHI:	there was a frog .
@Eg:	story
```

Only utterances between matching `@Bg`/`@Eg` pairs with the specified label are included. Utterances outside all gem boundaries are excluded.

## Related commands

- [GEMLIST](../commands/gemlist.md) — list all gem segment labels in a file
- `freq --gem` — frequency analysis restricted to gem segments (equivalent to legacy GEMFREQ)
