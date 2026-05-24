# COMBO -- Boolean Keyword Search

**Status:** Current
**Last updated:** 2026-05-23 14:50 EDT

## Purpose

Searches for utterances matching boolean combinations of keywords. Supports AND (`+`) and OR (`,`) logic with case-insensitive substring matching. This is the primary search tool for finding utterances containing specific words or word combinations.

See the [CLAN manual](https://talkbank.org/0info/manuals/CLAN.html#_Toc220409095) for the original COMBO command specification.

## Usage

```bash
chatter clan combo -S "want+cookie" file.cha
chatter clan combo -S "want,milk" file.cha
chatter clan combo -S "want+cookie" --speaker CHI file.cha
```

## Options (chatter-native)

| Option | CLAN flag | Description |
|--------|-----------|-------------|
| `--speaker <CODE>` | `+t*CHI` (or `+tCHI`) | Include speaker |
| `--exclude-speaker <CODE>` | `-t*CHI` (or `-tCHI`) | Exclude speaker |
| `-S`, `--search <EXPR>` | `+s"EXPR"` | Search expression (required, repeatable; multiple `--search` flags combined with OR) |
| `--gem <LABEL>` | `+g"label"` | Restrict to gem segment |
| `--range <START-END>` | `+z25-125` | Utterance range |
| `--id-filter <PATTERN>` | `+t@ID="..."` | Filter by @ID pattern |
| `--include-retracings` | `+r6` | Include retraced words in counting |
| `--format <FMT>` | -- | Output format: clan (default), text, json, csv |

## CLAN `+`-flag coverage audit

Authoritative enumeration of every CLAN `combo` flag, mapped
against chatter's coverage. Sources:

* `OSX-CLAN/src/clan/combo.cpp` — `usage()` and `getflag()`.
* `OSX-CLAN/src/clan/cutt.cpp` — `mainusage()` COMBO branches.
* `crates/talkbank-clan/src/clan_args.rs` — chatter's rewriter.
* `crates/talkbank-cli/src/cli/args/clan_commands.rs::Combo` plus
  `clan_common.rs::CommonAnalysisArgs`.

(Status legend: same as [FREQ](./freq.md#status-legend).)

### COMBO-specific `+`-flags (from `combo.cpp::getflag`)

| CLAN flag | Meaning | Chatter | Status | Notes |
|---|---|---|---|---|
| `+bN` | Search an `N` number-cluster unit | — | Missing | Multi-utterance match. |
| `+g1` | String-oriented search on the whole tier (default: word-oriented) | — | Missing | |
| `+g2` | String-oriented search on a single word | — | Missing | |
| `+g3` | Find only the first match per utterance | `--first-match-only` | Done | Landed 2026-05-22. When set, the per-expression matcher short-circuits after the first hit per utterance — `expr_hits` records only the first matching expression. Default behaviour reports every matching expression. |
| `+g4` | Exclude utterance delimiters from the search | (no-op accept) | Done | Landed 2026-05-23. chatter's COMBO calls `countable_words` on the main tier, which iterates `UtteranceContent::Word`/`AnnotatedWord`/`ReplacedWord` (and recursively into groups) but never reaches `Terminator` — terminators live at a separate AST level. So `+g4`'s effect (exclude delimiters) is the chatter default; the rewriter consumes the flag, clap never sees it. Without the arm, `+g4` would fall through to the generic gem-segment branch and mis-route to `--gem 4`. Pinned by `combo_g4_is_silently_consumed_as_noop`. |
| `+g5` | Use `^` as AND operator instead of "followed-by" | (no-op accept) | Done | Landed 2026-05-22. chatter's `combo` already uses `+` as AND (matching `+g5`'s semantic), so `+g5` is consumed by the rewriter as a no-op — no clap field is needed, the flag is silently dropped. |
| `+g6` | Include the tier's code name in the search | — | Missing | |
| `+g7` | Do not count duplicate matches | `--dedupe-matches` | Done | Landed 2026-05-22. Repeated word forms within a single utterance contribute at most one entry to each expression's `matched_words`; first-encounter order preserved. Mainly affects OR expressions (`cookie,milk`) over utterances with repeated tokens. |
| `+sS` / `-sS` | Pattern (required) — output tiers matching / not matching | `-S` / `--search` plus `--exclude-search` | Done | Both polarities mapped 2026-05-22. The rewriter routes `+sS` → `--search S` and `-sS` → `--exclude-search S` for COMBO specifically (other commands keep the per-word `+s`/`-s` semantic). Utterances matching any `--exclude-search` expression are dropped even if they match an `--search` expression. |
| `+s@F` / `-s@F` | Load search expressions from file F (one per line) | `--search-file` / `--exclude-search-file` | Done | Landed 2026-05-22. Each surviving line is parsed by `SearchExpr::parse`, so AND (`+`) and OR (`,`) operators work per line, just like inline `--search`. File format matches `cutt.cpp::rdexclf`: blank lines, `# `-comments, `;%* `-annotation lines skipped; UTF-8 BOM stripped. Repeatable. Pinned by `combo_search_at_sigil_routes_to_search_file` and `combo_exclude_search_at_sigil_routes_to_exclude_search_file` in `clan_args.rs`. |
| `+dv` | Display all parsed individual parts of the search pattern | — | Missing | Search-debug output. |
| `+d`, `+d1`..`+d5` | Display modes (CHAT, line numbers, file names, matched-only, etc.) | — | Rewriter only | Same `--display-mode N` rewriter as KWAL. |

### General `+`-flags COMBO inherits (from `cutt.cpp::mainusage`)

| CLAN flag | Meaning | Chatter | Status | Notes |
|---|---|---|---|---|
| `+t*X` / `-t*X` | Include/exclude speaker | `--speaker` / `--exclude-speaker` | Done | `+tX` accepted post-2026-05-21. |
| `+t%X` / `-t%X` | Include/exclude dependent tier | `--tier` / `--exclude-tier` (rewriter target) | Rewriter only | |
| `+t@ID="..."` | Filter by @ID pattern | `--id-filter` | Done | |
| `+gX` | Gem filter | `--gem` | Done | Distinct from `+g1`..`+g7` (no S suffix). |
| `+zN-M` | Utterance range | `--range` | Done | |
| `+rN` | Retrace / clitic / prosodic controls | `--include-retracings` (`+r6`) | Partial | |
| `+u` | Combine across files | (default) | Done | |
| `+re` | Recurse | (default) | Done | |
| `+pS` | Word delimiter | — | Missing | |
| `+k` | Case-sensitive | `--case-sensitive` | Done | Landed 2026-05-23. Reads `CommonAnalysisArgs::case_sensitive`. Two-layer fix: `SearchExpr::parse_with_case` preserves case in the stored terms (default `parse()` lowercases), and `process_utterance` populates words via `cleaned_text()` instead of `NormalizedWord::from_word`. Both sides must agree, so `ComboConfig::case_sensitive` is the single switch threaded into both. Pinned by `combo_case_sensitive_uppercase_keyword_misses_lowercase_word` and `combo_case_sensitive_matches_when_case_aligned`. |
| `+wN` / `-wN` | Context window | `--context-after N` / `--context-before N` | Done | Landed 2026-05-23. Same shape as KWAL: `ComboState` carries a `VecDeque` ring buffer for pre-context plus an `awaiting_after` Vec for post-context. `process_utterance` interleaves the bookkeeping with match-detection. Exclude (`-sS`) utterances still feed the windows (they count as non-matches), so context bookkeeping is correct around excluded utterances. Pinned by `combo_context_after_captures_post_match_lines`, `combo_context_before_captures_pre_match_lines`, `combo_default_no_context_window`. |
| `+f` / `+fEXT` | Output to file | `--output-ext` (rewriter target) | Rewriter only | Phase 1.1. |

### Audit summary

| Bucket | Count |
|---|---|
| Done | 13 |
| Partial | 2 |
| Rewriter only | 7 |
| Missing | 7 |

COMBO has the second-largest "Rewriter only" bucket after KWAL,
driven by the same `+dN` display-mode story. The COMBO-specific
`+g1`..`+g7` search-mode switches are also load-bearing: they each
change the matching semantic, and a researcher pasting
`combo +g1 +s"want milk" file.cha` into chatter today gets
chatter's word-oriented match (no analog for `+g1`).

## Search Syntax

- `+` between terms means AND (all terms must be present in the utterance)
- `,` between terms means OR (at least one term must be present)
- Terms are case-insensitive substring matches against countable words
- Multiple `--search` flags are combined with OR (any expression matching counts)
- AND takes precedence if both `+` and `,` appear in one expression

## CLAN Equivalence

| CLAN command | Rust equivalent |
|---|---|
| `combo +s"want^cookie" file.cha` | `chatter clan combo file.cha -S "want+cookie"` |
| `combo +s"want\|milk" file.cha` | `chatter clan combo file.cha -S "want,milk"` |
| `combo +s"want^cookie" +t*CHI file.cha` | `chatter clan combo file.cha -S "want+cookie" --speaker CHI` |

## Display Modes (`+dN` / `--display-mode N`) — DRAFT, awaiting PI review

> **Status: drafted from CLAN manual; not yet implemented.** Rewriter
> at `crates/talkbank-clan/src/clan_args.rs:101` translates
> `+dN` → `--display-mode N`; no `clap` field consumes it today.
> Drafted from CLAN manual §7.7.10 (`Unique Options`, COMBO) for
> PI review.

| N | CLAN behavior (verbatim from manual) |
|---|---|
| `+d` (no number) | "Normally, combo outputs the location of the tier where the match occurs. When the `+d` switch is turned on you can output only each matched sentence in a simple legal chat format." |
| `+d1` | "Outputs legal chat format along with line numbers and file names." |
| `+d2` | "Outputs files names once per file only." |
| `+d3` | "Outputs legal chat format, but with only the actual words matched by the search string, along with `@Comment` headers that are ignored by other programs." |
| `+d4` | "Use of the `+d4` switch was described in the previous section." (Manual cross-reference; resolution pending.) |
| `+d7` | "Search for words linked between two tiers." |

### Open questions for PI review

1. `+d`/`+d1`/`+d2` parallel KWAL's `+d`/`+d1`/`+d2` almost exactly.
   Worth defining a shared `--display-mode` enum across search-style
   commands (KWAL + COMBO) with the same variant names?
2. `+d3` "matched words plus `@Comment` headers" is COMBO-specific.
   Probably a separate enum variant.
3. `+d4`: manual cross-references the "previous section". This needs
   PI clarification — the immediately-previous section is general
   COMBO description, not a `+d`-table.
4. `+d7` (cross-tier linkage) overlaps with FREQ `+d7`. If both
   commands' `+d7` is "compare two tiers", the enum variant name
   should match.

## Output

Each matching utterance with:

- Source filename
- Speaker code
- Full utterance text (CHAT format)
- Summary counts of matching vs. total utterances

## Differences from CLAN

- **Operator syntax**: CLAN uses `^` for AND and `\|` for OR; this implementation uses `+` and `,` respectively for shell-friendliness.
