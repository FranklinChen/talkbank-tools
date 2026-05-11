# MORTABLE -- Morphological Category Cross-Tabulation

**Status:** Current
## Purpose

Produces a per-speaker frequency table of morphosyntactic categories by matching POS tags from the `%mor` tier against patterns defined in a language-specific script file.

Requires a language script file (e.g., `eng.cut`) that defines patterns and their labels for categorizing morphemes from the `%mor` tier. Each rule line contains a quoted label and `+`/`-` prefixed POS patterns. Rules can be grouped as OR (first match wins) or AND (all must match).

See the [CLAN manual](https://talkbank.org/0info/manuals/CLAN.html#_Toc220409286) for the original MORTABLE command specification.

> **⚠️ Currently unusable.** As of 2026-05-02, running any
> `chatter clan mortable …` invocation (including `--help`) panics
> at startup with `Short option names must be unique for each
> argument, but '-f' is in use by both 'script' and 'format'`. The
> conflict is between `Mortable.script: PathBuf` (declared with
> `#[arg(short = 'f', long)]` at
> `crates/talkbank-cli/src/cli/args/clan_commands.rs:259-261`) and
> the universal `--format`/`-f` short on the shared
> `CommonAnalysisArgs`. Until the `script` short is removed or
> renamed, this command cannot be invoked. The flag set documented
> below describes the *intended* surface; treat the command as
> blocked until the clap conflict is resolved.

## Usage

```bash
chatter clan mortable --script eng.cut file.cha
chatter clan mortable --script eng.cut --speaker CHI file.cha
```

## Options

| Option | Description |
|--------|-------------|
| `--speaker <CODE>` | Include speaker |
| `--script <PATH>` | Language script file (.cut) -- required |
| `--format <FMT>` | Output format: text, json, csv, clan |

## Differences from CLAN

- **POS matching**: Operates on parsed `%mor` tier data rather than raw text line scanning.
- **POS matching detail**: POS tags are read directly from typed `%mor` items instead of reparsing serialized `%mor` content.
- **Script file format**: Compatible with CLAN's `.cut` files.
- **Output formats**: Supports text, JSON, and CSV formats.
- **Deterministic ordering**: `BTreeMap` ordering ensures deterministic output across runs.
- **Golden test parity**: Verified against CLAN C binary output.
