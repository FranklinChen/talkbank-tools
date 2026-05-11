# Running Commands

**Status:** Current
## Basic invocation

```bash
chatter clan <command> [options] <input>
```

The `<input>` can be a single `.cha` file or a directory (all `.cha` files are processed recursively).

## Getting help

```bash
chatter clan --help              # List all commands
chatter clan freq --help         # Help for a specific command
```

## Command categories

Commands are grouped into three categories:

- **Analysis commands** — read CHAT files and produce statistics (FREQ, MLU, etc.)
- **Transform commands** — read CHAT files and write modified CHAT files (FLO, CHSTRING, etc.)
- **Format converters** — convert between CHAT and other formats (ELAN2CHAT, CHAT2SRT, etc.)

## Exit codes

| Code | Meaning |
|------|---------|
| 0 | Success |
| 1 | Error (parse failure, missing file, invalid options) |

## Environment variables

| Variable | Effect |
|----------|--------|
| `RUST_LOG` | Control log verbosity (`debug`, `trace`, etc.) |
| `NO_COLOR` | Disable colored output |
