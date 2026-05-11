# Flag Translation Guide

**Status:** Current
Complete mapping between legacy CLAN `+flag`/`-flag` syntax and modern `--flag` equivalents. Both styles are accepted by `chatter clan`.

## Speaker and tier filters

| CLAN | Modern | Description |
|------|--------|-------------|
| `+t*CHI` | `--speaker CHI` | Include speaker |
| `-t*CHI` | `--exclude-speaker CHI` | Exclude speaker |
| `+t%mor` | `--tier mor` | Include dependent tier |
| `-t%gra` | `--exclude-tier gra` | Exclude dependent tier |
| `+t@ID="..."` | `--id-filter "..."` | Filter by @ID fields |

## Word filters

| CLAN | Modern | Description |
|------|--------|-------------|
| `+s<word>` | `--include-word <word>` | Include word |
| `-s<word>` | `--exclude-word <word>` | Exclude word |
| `+k` | `--case-sensitive` | Case-sensitive matching |

## Gem and range

| CLAN | Modern | Description |
|------|--------|-------------|
| `+g<label>` | `--gem <label>` | Include gem segment |
| `-g<label>` | `--exclude-gem <label>` | Exclude gem segment |
| `+z25-125` | `--range 25-125` | Utterance range |

## Display and output

| CLAN | Modern | Description |
|------|--------|-------------|
| `+dN` | `--display-mode N` | Display mode |
| `+fEXT` | `--output-ext EXT` | Output file extension |
| `+u` | *(default)* | Merge speakers (always on) |
| `+r6` | `--include-retracings` | Count retraced material |

## Context

| CLAN | Modern | Description |
|------|--------|-------------|
| `+wN` | `--context-after N` | Context lines after match |
| `-wN` | `--context-before N` | Context lines before match |
