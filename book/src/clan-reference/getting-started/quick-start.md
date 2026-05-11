# Quick Start

**Status:** Current
All CLAN commands follow the pattern:

```bash
chatter clan <command> [options] <file.cha>
```

## Your first analysis

Given a CHAT transcript `sample.cha`:

```bash
# Word frequency count
chatter clan freq sample.cha

# Mean length of utterance for the child speaker
chatter clan mlu --speaker CHI sample.cha

# Mean length of turn
chatter clan mlt sample.cha

# Vocabulary diversity (D statistic)
chatter clan vocd sample.cha
```

## Working with directories

Process all `.cha` files in a directory:

```bash
chatter clan freq corpus/
```

## Output formats

Every analysis command supports text (default), JSON, and CSV output:

```bash
chatter clan freq --format json sample.cha
chatter clan mlu --format csv sample.cha
```

## Filtering

Restrict analysis to specific speakers, tiers, words, or utterance ranges:

```bash
# Only the CHI speaker
chatter clan freq --speaker CHI sample.cha

# Exclude the investigator
chatter clan freq --exclude-speaker INV sample.cha

# Only utterances 10-50
chatter clan freq --range 10-50 sample.cha

# Only within gem boundaries
chatter clan freq --gem "story" sample.cha
```

See the [Filtering](../user-guide/filtering.md) chapter for the full guide.

## Transforms

Transform commands modify CHAT files and write the result:

```bash
# Add fluent output tier
chatter clan flo sample.cha

# String replacement from a rules file
chatter clan chstring rules.cex sample.cha

# Reorder dependent tiers
chatter clan tierorder sample.cha
```

## Format conversion

Convert between CHAT and other formats:

```bash
chatter clan elan2chat recording.eaf
chatter clan praat2chat recording.TextGrid
chatter clan chat2srt sample.cha
```
