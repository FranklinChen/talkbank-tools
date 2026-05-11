# Working with Directories

**Status:** Current
Pass a directory path to process all `.cha` files recursively:

```bash
chatter clan freq corpus/
chatter clan mlu corpus/English/
```

## Behavior

- All `.cha` files under the directory are discovered and processed
- Subdirectories are traversed recursively
- Non-`.cha` files are silently ignored
- Results are aggregated across all files (same as CLAN's default behavior)

## Per-file output

Some commands support `--per-file` mode to get separate results for each file:

```bash
chatter clan mlu --per-file corpus/
```

## Combining with filters

Directory processing composes with all filters:

```bash
# CHI utterances across an entire corpus
chatter clan freq --speaker CHI corpus/

# First 100 utterances in each file
chatter clan mlu --range 1-100 corpus/English/
```

## JSON output for scripting

Combine directory processing with JSON output for batch analysis:

```bash
chatter clan mlu --format json corpus/ > results.json
```

See [Pipelines and Scripting](pipelines.md) for more advanced workflows.
