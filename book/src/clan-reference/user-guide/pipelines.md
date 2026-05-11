# Pipelines and Scripting

**Status:** Current
## JSON for Scripting

For programmatic consumption, use `--format json` and pipe to `jq`:

```bash
# Extract total word count
chatter clan freq --format json file.cha | jq '.total_words'

# Get MLU value for CHI
chatter clan mlu --speaker CHI --format json file.cha | jq '.speakers[0].mlu'
```

## Batch Processing

Process many files and collect results:

```bash
# Frequency analysis across an entire corpus
chatter clan freq --format csv corpus/ > results.csv

# Per-file MLU in JSON
for f in corpus/*.cha; do
    echo "=== $f ==="
    chatter clan mlu --speaker CHI --format json "$f"
done
```

## Combining with Other Tools

```bash
# Find files with low MLU
chatter clan mlu --format json corpus/ | jq 'select(.speakers[].mlu < 2.0)'

# Word frequency sorted by count
chatter clan freq --format json file.cha | jq '.words | to_entries | sort_by(-.value) | .[0:10]'
```

## Transform Chains

Transforms modify files in place (writing to output files), so they can be chained:

```bash
# Clean up a file: lowercase, then compute ages
chatter clan lowcase file.cha
chatter clan dates file.cha
```
