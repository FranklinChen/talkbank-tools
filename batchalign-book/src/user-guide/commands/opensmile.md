# opensmile

**Status:** Current
**Last updated:** 2026-04-08 07:40 EDT

Extract acoustic features from audio files using openSMILE. Produces
`.opensmile.csv` output — **not CHAT**. This is the only processing command
that does not produce `.cha` output.

Uses positional `INPUT_DIR OUTPUT_DIR` arguments (not the shared
`PATHS... -o OUTPUT` form used by `align`, `morphotag`, etc.).

---

## Quick start

```bash
# Extract default eGeMAPSv02 features from all audio in a directory
batchalign3 opensmile input_dir/ output_dir/

# Use a different feature set
batchalign3 opensmile input_dir/ output_dir/ --feature-set ComParE_2016

# Use the fleet server
batchalign3 --server http://net:8001 opensmile input_dir/ output_dir/
```

---

## Pipeline

```mermaid
flowchart TD
    start([opensmile invoked]) --> resolve[Resolve audio files]
    resolve --> prep[Rust audio prep\nprepare mono PCM artifact]
    prep --> feature_check{--feature-set?}
    feature_check -->|eGeMAPSv02| egemaps[eGeMAPSv02 features\n88 acoustic descriptors]
    feature_check -->|ComParE_2016| compare[ComParE_2016 features\n6,373 acoustic descriptors]
    feature_check -->|Custom| custom[Custom feature set name]

    egemaps --> worker
    compare --> worker
    custom --> worker

    worker["execute_v2(task='opensmile') → Python worker\nExtracts acoustic features from prepared audio"]
    worker --> output[Write CSV output\nContent-type: csv]
    output --> done([Output .opensmile.csv files])
```

---

## Options

### Positional arguments

| Argument | Meaning |
| --- | --- |
| `INPUT_DIR` | Directory containing audio files |
| `OUTPUT_DIR` | Directory for output `.opensmile.csv` files |

### opensmile options

| Option | Default | Meaning |
| --- | --- | --- |
| `--feature-set SET` | `eGeMAPSv02` | Feature set: `eGeMAPSv02`, `eGeMAPSv01b`, `GeMAPSv01b`, or `ComParE_2016` |
| `--lang CODE` | `eng` | 3-letter ISO language code |

---

## Output format

Each audio file produces `FILE.opensmile.csv` with feature names as column
headers and one row per file (or one row per frame for frame-level sets).
BA3 uses a row-oriented CSV (feature names as columns), which differs from
BA2's transposed feature-per-row export.

---

## Related documentation

- [Command I/O: opensmile](../../reference/command-io.md#10-opensmile) — I/O patterns
- [Command Flowcharts: opensmile](../../architecture/command-flowcharts.md#opensmile) — full architecture flowchart
