# Output Formats

**Status:** Current
Every analysis command produces a typed result that can be rendered in multiple formats.

## Text (default)

Human-readable output, designed for terminal display:

```bash
chatter clan freq file.cha
```

## CLAN-compatible text

Legacy CLAN output format, used for parity testing. Activated internally by `render_clan()`.

## JSON

Structured output for programmatic consumption:

```bash
chatter clan freq --format json file.cha
```

JSON output preserves all fields from the typed result struct, including metadata that text format may omit.

## CSV

Tabular output for spreadsheet tools:

```bash
chatter clan mlu --format csv file.cha
```

## Notes

- Transform commands write modified CHAT files, not analysis output — format flags don't apply
- Converters produce the target format (SRT, ELAN XML, etc.)
