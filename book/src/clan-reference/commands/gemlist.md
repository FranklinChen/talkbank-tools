# GEMLIST — List Gem Segments

**Status:** Current
**Last updated:** 2026-05-22 13:45 EDT

## Purpose

Lists all gem segments (`@Bg`/`@Eg` bracketed regions) found in CHAT files, reporting the label, utterance count, and participating speakers for each gem.

See the [CLAN manual](https://talkbank.org/0info/manuals/CLAN.html#_Toc220409206) for the original GEM command specification.

## Usage

```bash
chatter clan gemlist file.cha
chatter clan gemlist file.cha --speaker CHI
```

## Options

| Option | CLAN flag | Description |
|--------|-----------|-------------|
| `--speaker <code>` | `+t*CODE` | Restrict to specific speaker |
| `--format <fmt>` | — | Output format: text, json, csv |

## CLAN `+`-flag coverage audit

GEMLIST is an **analysis** command. Sources:
`OSX-CLAN/src/clan/gemlist.cpp::usage`,
`crates/talkbank-clan/src/commands/gemlist.rs`.

### GEMLIST-specific `+`-flags (from `gemlist.cpp::usage`)

| CLAN flag | Meaning | Chatter | Status |
|---|---|---|---|
| `-wS` | Exclude tiers with postcodes `S` | — | Missing |

Audit summary: 1 Done (default gem listing), 1 Missing.
chatter's GEMLIST surfaces gem labels and per-gem statistics
without a postcode-exclusion filter. The general
`CommonAnalysisArgs` set is inherited.

## Output

Per gem label:

- Number of utterances within the gem scope
- Number of occurrences (how many `@Bg`/`@Eg` pairs with this label)
- Speakers who produced utterances within the gem
- Source files containing this gem

## Implementation Note

Gem boundaries (`@Bg`/`@Eg`) are interleaved headers in `ChatFile.lines`. Since the parser does not populate `Utterance.preceding_headers`, this command scans the full line array in `end_file()` rather than relying on per-utterance callbacks.

## Differences from CLAN

- Gem boundary detection operates on parsed `Header` variants from the AST rather than raw text line matching
- Output supports text, JSON, and CSV formats (CLAN produces text only)
- Deterministic output ordering via sorted collections
- **Golden test parity**: Verified against CLAN C binary output
