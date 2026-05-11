# Flag Translation Guide

**Status:** Current
The original CLAN uses a `+flag`/`-flag` syntax that differs from standard CLI conventions. The Rust reimplementation accepts both styles — legacy CLAN flags are automatically rewritten to modern `--flag` equivalents before parsing.

## Complete Flag Mapping

| CLAN Flag | Modern Equivalent | Meaning |
|-----------|-------------------|---------|
| `+t*CHI` | `--speaker CHI` | Include speaker |
| `-t*CHI` | `--exclude-speaker CHI` | Exclude speaker |
| `+t%mor` | `--tier mor` | Include dependent tier |
| `-t%gra` | `--exclude-tier gra` | Exclude dependent tier |
| `+t@ID="..."` | `--id-filter "..."` | Filter by @ID fields |
| `+s<word>` | `--include-word <word>` | Include word |
| `-s<word>` | `--exclude-word <word>` | Exclude word |
| `+g<label>` | `--gem <label>` | Include gem segment |
| `-g<label>` | `--exclude-gem <label>` | Exclude gem segment |
| `+z25-125` | `--range 25-125` | Utterance range |
| `+r6` | `--include-retracings` | Count retraced material |
| `+u` | *(default behavior)* | Merge speakers (already default) |
| `+dN` | `--display-mode N` | Display mode |
| `+k` | `--case-sensitive` | Case-sensitive matching |
| `+fEXT` | `--output-ext EXT` | Output file extension |
| `+wN` | `--context-after N` | Context lines after match |
| `-wN` | `--context-before N` | Context lines before match |

## Examples

### Speaker Filtering

```bash
# Original CLAN: include CHI, exclude MOT
freq +t*CHI -t*MOT file.cha

# Modern equivalent
chatter clan freq --speaker CHI --exclude-speaker MOT file.cha
```

### Word Search

```bash
# Original CLAN: search for "want" and "need"
combo +s"want" +s"need" file.cha

# Modern equivalent
chatter clan combo --include-word want --include-word need file.cha
```

### Combined Filters

```bash
# Original CLAN: CHI speaker, utterances 10-50, case-sensitive
freq +t*CHI +z10-50 +k file.cha

# Modern equivalent
chatter clan freq --speaker CHI --range 10-50 --case-sensitive file.cha
```

## Notes

- The `+u` flag (merge speakers into a single analysis) is the default behavior and is accepted but ignored.
- Flags are position-independent — they can appear before or after file arguments.
- Unknown flags that don't match CLAN patterns pass through unchanged to clap, which will report an error with suggestions.
