# Utterance Range Filtering

**Status:** Current
Range filters restrict analysis to a numbered subset of utterances, useful for analyzing specific portions of a transcript without splitting the file.

## Usage

```bash
chatter clan freq --range 10-50 file.cha
chatter clan mlu --range 1-100 file.cha
```

CLAN equivalent: `+z10-50`

## Behavior

- Utterance numbering starts at 1
- Both endpoints are inclusive (e.g., `--range 10-50` includes utterances 10, 11, ..., 50)
- Only main-tier utterances (`*SPK:` lines) are counted; headers, dependent tiers, and comments are not numbered
- Range filtering is applied after speaker filtering — if you filter to `--speaker CHI`, utterance 1 is the first CHI utterance, not the first utterance in the file

## Open-ended ranges

To analyze from a point to the end of the file, use a large upper bound:

```bash
chatter clan freq --range 50-9999 file.cha
```

## Combining with other filters

Range composites with all other filters:

```bash
# First 50 CHI utterances within a gem segment
chatter clan mlu --speaker CHI --range 1-50 --gem "story" file.cha
```
