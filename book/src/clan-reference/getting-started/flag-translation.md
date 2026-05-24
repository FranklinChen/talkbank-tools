# Flag Translation Guide

**Status:** Current
**Last updated:** 2026-05-23 18:52 EDT

The original CLAN uses a `+flag`/`-flag` syntax that differs from standard CLI conventions. The Rust reimplementation accepts both styles — legacy CLAN flags are automatically rewritten to modern `--flag` equivalents before parsing.

## Complete Flag Mapping

| CLAN Flag | Modern Equivalent | Meaning |
|-----------|-------------------|---------|
| `+t*CHI` | `--speaker CHI` | Include speaker |
| `-t*CHI` | `--exclude-speaker CHI` | Exclude speaker |
| `+t%mor` | `--tier mor` | Include dependent tier |
| `-t%gra` | `--exclude-tier gra` | Exclude dependent tier |
| `+t@ID="..."` | `--id-filter "..."` | Filter by `@ID` header pattern |
| `+s<word>` | `--include-word <word>` | Include word |
| `-s<word>` | `--exclude-word <word>` | Exclude word |
| `+g<label>` | `--gem <label>` | Include gem segment |
| `-g<label>` | `--exclude-gem <label>` | Exclude gem segment |
| `+z25-125` | `--range 25-125` | Utterance range |
| `+r6` | `--include-retracings` | Count retraced material |
| `+u` | *(default behavior)* | Merge speakers (already default) |
| `+dN` | `--display-mode N` *(generic)* / per-command typed flag | Display mode — **partially landed.** Each `+dN` value is rewritten command-by-command. FREQ `+d1` → `--word-list-only`, `+d2` → `--format csv`, `+d3` → `--types-tokens-only --format csv`, `+d4` → `--types-tokens-only`. COOCCUR `+d` → `--no-frequency-counts`. FREQPOS `+d` → `--position-classification second`. Other `+dN` values still fall through to the generic `--display-mode N` placeholder, which clap does not consume yet (see Phase 3 plan). |
| `+k` | `--case-sensitive` | Case-sensitive matching — **fully landed** across the search/frequency family (FREQ, KWAL, VOCD, COMBO, FREQPOS, DIST, MAXWD). FREQ: pattern matching via `WordFilter` + case-preserving frequency-table keying. KWAL: keyword and word compared verbatim instead of via `NormalizedWord` lowercasing. VOCD: pattern matching + D-statistic token stream skipping its default `to_lowercase`. COMBO: `SearchExpr::parse_with_case` preserves case in the stored terms and the word stream populates via `cleaned_text()`. FREQPOS / DIST / MAXWD: case-preserving key derivation in `process_utterance` (MAXWD's unique-length and exclude-length filters then count case variants as distinct words). Other commands inherit `+k` from `cutt.cpp::mainusage` but it's a semantic no-op since they don't word-match. |
| ~~`+fEXT`~~ | ~~`--output-ext EXT`~~ | Output file extension — **currently non-functional**: the rewriter at `crates/talkbank-clan/src/clan_args.rs:107` produces `--output-ext`, but no `clap` field consumes it. Tracked as Phase 2 of the rewriter-honor plan (blocked on a batch-output prerequisite). |
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
# Original CLAN: CHI speaker, utterances 10-50, include retraced material
freq +t*CHI +z10-50 +r6 file.cha

# Modern equivalent
chatter clan freq --speaker CHI --range 10-50 --include-retracings file.cha
```

## Notes

- The `+u` flag (merge speakers into a single analysis) is the default behavior and is accepted but ignored.
- Flags are position-independent — they can appear before or after file arguments.
- Unknown flags that don't match CLAN patterns pass through unchanged to clap, which will report an error with suggestions.
- `+dN` and `+k` are now **partially functional** — see the per-command status tables under `clan-reference/commands/` for which `N` values and which commands are wired up. `+fEXT` is still rewritten by the legacy-flag layer but rejected by `clap` (no consuming field yet). See [`migrating-from-clan.md`](../../chatter/user-guide/migrating-from-clan.md) for the same caveats and the rewriter-honor plan that tracks the remaining gaps.
