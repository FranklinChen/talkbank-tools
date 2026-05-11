# Harvesting Review Results

**Status:** Current
**Last updated:** 2026-03-30 13:40 EDT

After reviewers have rated utterances in aligned CHAT files, the ratings
need to be collected for analysis. The `%xrev` and `%xalign` tiers are
part of the CHAT files, so harvesting is just parsing.

## Using the harvesting script

```bash
python3 scripts/harvest_reviews.py path/to/reviewed/files/ --output reviews.csv
```

This scans all `.cha` files in the directory, extracts `%xrev` and `%xalign`
tiers, and produces a CSV with one row per reviewed utterance:

| Column | Description |
|--------|-------------|
| `file` | Filename |
| `utterance_idx` | Utterance number in the file |
| `speaker` | Speaker code (e.g., "CHI", "PIL") |
| `strategy` | What `%xalign` reports (e.g., "boundary_averaged") |
| `rating` | Human rating: ok, early, late, wrong, or ? |
| `machine_start` | Original bullet start (ms) |
| `machine_end` | Original bullet end (ms) |
| `corrected_start` | Corrected start if `[corrected]` (ms), else empty |
| `corrected_end` | Corrected end if `[corrected]` (ms), else empty |
| `delta_ms` | Displacement from machine timing (ms), else empty |
| `note` | Free-form reviewer note |

## What to do with the results

### Aggregate quality metrics

```bash
# How many rated good vs early vs late vs wrong?
cut -d, -f5 reviews.csv | sort | uniq -c | sort -rn
```

### Per-strategy breakdown

```bash
# Are boundary-averaged bullets rated differently than gap-filled?
awk -F, '{print $4, $5}' reviews.csv | sort | uniq -c | sort -rn
```

### Displacement analysis (for corrected bullets)

```bash
# Average displacement for corrected bullets
awk -F, '$5=="corrected" && $11!="" {sum+=$11; n++} END {print sum/n "ms avg"}' reviews.csv
```

### Pattern detection

Look for systematic issues in the notes column:

```bash
grep -i 'backchannel\|overlap\|speaker' reviews.csv
```

## Comparing conditions

To compare raw align vs `--bullet-repair` vs FIXBULLETS, review the same
files under each condition and harvest separately:

```bash
python3 scripts/harvest_reviews.py raw-output/ --output raw.csv
python3 scripts/harvest_reviews.py repaired-output/ --output repaired.csv
python3 scripts/harvest_reviews.py fixbullets-output/ --output fixbullets.csv
```

Then compare the `%good` rate across conditions.
