# opensmile: Developer Reference

**Status:** Current
**Last updated:** 2026-10-25 08:12 EDT

Implementation guide for the `opensmile` command. For user-facing
documentation, see [User Guide: opensmile](../../user-guide/commands/opensmile.md).

---

## Implementation map

| Layer | Location | Responsibility |
|-------|----------|----------------|
| CLI args | `crates/batchalign/src/cli/args/commands.rs`: `OpensmileArgs` | Positional input/output dirs, feature-set, lang |
| Catalog entry | `crates/batchalign/src/recipe_runner/catalog.rs` | the `CatalogEntry` for `opensmile` |
| Stage recipe | `crates/batchalign/src/recipe_runner/recipes.rs` | `OPENSMILE_RECIPE` |
| Audio prep | Shared media prep in `crates/batchalign/src/runner/` | Converts audio to mono PCM artifact |
| Worker IPC | `batchalign/inference/opensmile.py`: `extract_features()` | Loads openSMILE, returns feature dict |
| Response admission and CSV writer | `crates/batchalign/src/runner/dispatch/media_analysis_v2.rs` | `AdmittedOpenSmileResult` → feature/value rows; typed execute failures retain their retry category |

---

## Positional I/O

`opensmile` does not use `CommonOpts` (`PATHS... -o DIR`). It uses positional
`INPUT_DIR OUTPUT_DIR` syntax. This is an intentional deviation inherited from
BA2 and documented in [Command I/O](../../reference/command-io.md#10-opensmile).

---

## Output format change from BA2

BA2 wrote a transposed CSV (feature per row, file per column). BA3 writes a
row-oriented CSV (file per row, feature per column). This is a breaking output
format change. The feature names and values are identical.

---

## Worker IPC: opensmile task (V2 protocol)

```text
execute_v2 request:
{
  "task": "opensmile",
  "audio_ref_id": <prepared-audio-ref>,
  "feature_set": "eGeMAPSv02",
  "feature_level": "functionals"
}

execute_v2 response:
{
  "features": { "F0semitoneFrom27.5Hz_sma3nz_amean": 12.3, ... }
}
```

---

## Related developer documentation

- [Command Flowcharts: opensmile](../../architecture/command-flowcharts.md#opensmile)
