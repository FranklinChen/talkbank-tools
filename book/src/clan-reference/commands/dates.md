# DATES -- Age Computation from Birth and Date Headers

**Status:** Current
## Purpose

Reimplements CLAN's `dates` command, which computes the age of each participant at the time of transcription by subtracting `@Birth` dates from the file-level `@Date` header. Computed ages are inserted as `@Comment: Age of CHI is Y;M.D` headers after the `@ID`/`@Birth` block.

See the [CLAN manual](https://talkbank.org/0info/manuals/CLAN.html#_Toc220409311) for the original command documentation.

The computed age uses CHAT format: `years;months.days` (e.g., `2;6.15` for two years, six months, and fifteen days).

## Usage

```bash
chatter clan dates file.cha
```

## Options

This command has no configurable options.

## Behavior

1. Collects the `@Date` header value and all `@Birth` headers with their participant codes.
2. Computes each participant's age by subtracting the birth date from the file date.
3. Inserts `@Comment: Age of <PARTICIPANT> is <age>` headers into the file.
4. If no `@Date` header is present, the file is left unchanged.

## Differences from CLAN

- Operates on AST rather than raw text.
- Uses the framework transform pipeline (parse -> transform -> serialize -> write).
