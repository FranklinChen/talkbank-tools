# KWAL -- Keyword And Line

**Status:** Current
**Last updated:** 2026-05-23 14:20 EDT

## Purpose

Searches for clusters containing specified keywords and displays the matching lines with context. The legacy manual gives `KWAL` a dedicated section and describes it as operating on "clusters": the main tier plus the selected dependent tiers associated with that line.

In `talkbank-clan`, keywords are currently matched against countable words on the main tier, with the matched utterance shown in context.

## Usage

```bash
chatter clan kwal -k want file.cha
chatter clan kwal -k want --speaker CHI file.cha
chatter clan kwal -k want -k cookie file.cha
```

## Options (chatter-native)

| Option | CLAN flag | Description |
|--------|-----------|-------------|
| `--speaker <CODE>` | `+t*CHI` (or `+tCHI`) | Include speaker |
| `--exclude-speaker <CODE>` | `-t*CHI` (or `-tCHI`) | Exclude speaker |
| `-k <WORD>` / `--keyword <WORD>` | `+s"WORD"` | Keyword to search for (repeatable) |
| `--gem <LABEL>` | `+g"label"` | Restrict to gem segment |
| `--range <START-END>` | `+z25-125` | Utterance range |
| `--id-filter <PATTERN>` | `+t@ID="..."` | Filter by @ID pattern |
| `--include-retracings` | `+r6` | Include retraced words in counting |
| `--format <FMT>` | -- | Output format: clan (default), text, json, csv |

## CLAN `+`-flag coverage audit

Authoritative enumeration of every CLAN `kwal` flag, mapped against
chatter's coverage. Sources:

* `OSX-CLAN/src/clan/kwal.cpp` — `usage()` and `getflag()`.
* `OSX-CLAN/src/clan/cutt.cpp` — `mainusage()` KWAL branches.
* `crates/talkbank-clan/src/clan_args.rs` — chatter's rewriter.
* `crates/talkbank-cli/src/cli/args/clan_commands.rs::Kwal` plus
  `clan_common.rs::CommonAnalysisArgs`.

(Status legend: same as [FREQ](./freq.md#status-legend).)

### KWAL-specific `+`-flags (from `kwal.cpp::getflag`)

| CLAN flag | Meaning | Chatter | Status | Notes |
|---|---|---|---|---|
| `+a` | Sort output in alphabetical order of keywords (repeats tier per keyword) | — | Missing | KWAL's keyword-alphabetised mode. |
| `+b` | Apply `+s` match only when the keyword is the *only* item on the tier | `--strict-match` | Done | Landed 2026-05-23. Gate runs before keyword matching: utterances with `words.len() != 1` are rejected outright. Pinned by `kwal_strict_match_only_solo_word_matches` (single-word `["want"]` matches, `["I", "want", "cookie"]` doesn't) and `kwal_default_matches_anywhere_on_tier` (default matches both). |
| `+nS` | Include all utterances from speaker `S` when they *follow* a match for `+s` | — | Missing | Sliding-window-by-speaker — distinct from `+wN` general context window. |
| `-nS` | Exclude all utterances from speaker `S` when they follow a match | — | Missing | |
| `+d` (no N) | Output legal CHAT format | `--legal-chat` | Done | Landed 2026-05-23. Rewriter maps bare `+d` for KWAL; `render_clan` skips the `---` separator and `*** File … Keyword: X` decoration so only the matching `*Speaker:` lines remain. The dependent tiers attached to the match (e.g. `%mor`) come through `to_chat_string()` as legal CHAT. Pinned by `kwal_legal_chat_format_drops_location_decoration` and the rewriter test `kwal_legal_chat_format`. `+dN` for N ≥ 1 still falls through to `--display-mode N`. |
| `+d1`..`+d4`, `+d30`, `+d31`, `+d40`, `+d7`, `+d90`, `+d99` | Various output-format and tier-linking variants | — | Rewriter only | All rewrite to `--display-mode N`; no consuming clap field. |

### General `+`-flags KWAL inherits (from `cutt.cpp::mainusage`)

| CLAN flag | Meaning | Chatter | Status | Notes |
|---|---|---|---|---|
| `+t*X` / `-t*X` | Include/exclude speaker | `--speaker` / `--exclude-speaker` | Done | `+tX` accepted post-2026-05-21. |
| `+t%X` / `-t%X` | Include/exclude dependent tier | `--tier` / `--exclude-tier` (rewriter target) | Rewriter only | KWAL stores `fDepTierName` / `sDepTierName` for cluster output; chatter does not expose this. |
| `+t@ID="..."` | Filter by @ID pattern | `--id-filter` | Done | |
| `+t#ROLE` | Filter by role | `--role` | Done | Fixed 2026-05-22; see [FREQ](./freq.md) for the shared implementation. |
| `+s"word"` / `-s"word"` | Keyword (KWAL's primary input) | `-k / --keyword` / `--exclude-word` | Done | KWAL requires `+s`; chatter's `-k / --keyword` is the required CLI form. |
| `+s@F` / `-s@F` | Keywords / exclude words from file | `--include-word-file` / `--exclude-word-file` | Done | Landed 2026-05-22. File format: one pattern per line; blank lines, `# `-comments, and `;%* `-annotation lines skipped. Repeatable. |
| `+gX` | Gem filter | `--gem` | Done | KWAL uses the inherited semantic (gem segment), not an overloaded one. |
| `+zN-M` | Utterance range | `--range` | Done | |
| `+rN` | Retrace / clitic / prosodic controls | `--include-retracings` (`+r6`) | Partial | |
| `+u` | Combine across files | (default) | Done | Inverse default vs CLAN. |
| `+re` | Recurse | (default) | Done | |
| `+pS` | Word delimiter | — | Missing | |
| `+k` | Case-sensitive | `--case-sensitive` | Done | Landed 2026-05-23. Reads `CommonAnalysisArgs::case_sensitive`; KWAL `process_utterance` skips the `NormalizedWord` lowercasing on both keyword and word sides when set. Other commands' `+k` is still Rewriter-only — they continue to lowercase before matching. Pinned by `kwal_case_sensitive_uppercase_keyword_misses_lowercase_word` and `kwal_case_sensitive_matches_when_case_aligned`. |
| `+wN` / `-wN` | Context window | `--context-after N` / `--context-before N` | Done | Landed 2026-05-23. `+wN` emits N utterances *following* each match; `-wN` emits N utterances *preceding*. `KwalState` carries a `VecDeque` ring buffer of recent utterance texts (for pre-context) and a `Vec<(match_idx, remaining)>` list of matches still collecting post-context. Per-utterance flow: feed any open awaiting-after entries, detect match, snapshot ring as `pre_context`, register new awaiting-after, then update ring. `render_clan` emits pre-context lines + match body + post-context lines per match block. Distinct from KWAL's `+nS` speaker-context (separate Missing item). Pinned by `kwal_context_after_captures_post_match_lines`, `kwal_context_before_captures_pre_match_lines`, and `kwal_default_no_context_window`. |
| `+f` / `+fEXT` | Output to file | `--output-ext` (rewriter target) | Rewriter only | Phase 1.1. |

### Audit summary

| Bucket | Count |
|---|---|
| Done (byte-parity or in scope) | 12 |
| Partial | 1 |
| Rewriter only | 9 |
| Missing | 6 |

KWAL has the **largest "Rewriter only" bucket** of any audited
command so far (12). Almost all the variants come from `+dN`'s
nine sub-modes (`+d`, `+d1`..`+d4`, `+d30`, `+d31`, `+d40`, `+d7`,
`+d90`, `+d99`), each of which materially changes KWAL's output
shape. Until `--display-mode N` is wired in, a researcher pasting
`kwal +d1 +s"want" file.cha` into chatter gets a parse error.

## CLAN Equivalence

| CLAN command | Rust equivalent |
|---|---|
| `kwal +s"want" file.cha` | `chatter clan kwal file.cha -k want` |
| `kwal +s"want" +t*CHI file.cha` | `chatter clan kwal file.cha -k want --speaker CHI` |

## Display Modes (`+dN` / `--display-mode N`) — partially landed

> **Status: `+d` (no number) implemented as `--legal-chat`; `+dN`
> for N ≥ 1 still drafted from CLAN manual.** The generic rewriter
> at `crates/talkbank-clan/src/clan_args.rs` translates each `+dN`
> to `--display-mode N`, but no `clap` field consumes that token
> for N ≥ 1 today. This table is drafted from CLAN manual §7.17.5
> (`Unique Options`, KWAL) verbatim for PI review. Tracked in
> `<workspace>/docs/superpowers/plans/2026-05-11-clan-rewriter-honor-three-flags.md`
> Phase 3.

`KWAL` uses `+d` to switch the output shape (plain CHAT, with filenames,
Excel form, etc.). Quoted from CLAN manual §7.17.5:

| N | CLAN behavior (verbatim from manual) |
|---|---|
| `+d` (no number) | "Normally, kwal outputs the location of the tier where the match occurs. When the `+d` switch is turned on you can \[output\] in these formats: ... outputs legal CHAT format." |
| `+d1` | "Outputs legal CHAT format plus file names and line numbers." |
| `+d2` | "Outputs file names once per file only." |
| `+d3` | "Outputs ONLY matched items." |
| `+d30` | "Outputs ONLY matched items without any defaults removed. The `+d30` and the `+d3` switches can be combined." |
| `+d99` | "Convert 'word \[x 2\]' to 'word \[/\] word' and so on." |
| `+d4` | "Outputs for Excel." |
| `+d40` | "Outputs for Excel, repeating the same tier for every keyword match." |
| `+d7` | "Compares items across dependent tiers." Example: `kwal +d7 +s@\|-cop +sROOT +t%gra +t%mor t.cha` |

### Open questions for PI review

1. ~~`+d` (no number)~~: **resolved 2026-05-23.** Implemented as
   `--legal-chat` (boolean) — chatter's default render emits the
   `*** File ... Keyword: X` location decoration; `+d` switches to
   the matching utterance lines as a legal CHAT fragment. The
   `--display-mode 0` numeric framing was rejected in favour of a
   named boolean since the rest of the `+dN` table is genuinely
   distinct shape/format options rather than a numeric severity
   axis.
2. `+d30` is "`+d3` + don't strip defaults" — combinable. Maps to
   `--display-mode matched --no-strip-defaults` or
   `--display-mode 30`?
3. `+d99` is conceptually orthogonal to the others (it's a
   transformation, not an output shape). Worth splitting into a
   separate `--expand-repetition` flag rather than overloading
   `--display-mode`.
4. `+d4`/`+d40` for Excel: same Excel question as FREQ — overlap with
   `--format csv`.
5. `+d7` cross-tier comparison: deeply specific. In scope for the
   first `--display-mode` pass, or future work alongside `freqpos` /
   `mortable`?

## Output

Each matching utterance with:

- Speaker code
- Full utterance text
- File path (for multi-file searches)
- Match count summary per keyword

## Differences from CLAN

- **Manual intent**: `KWAL` is a cluster-oriented search command, not just a main-tier keyword matcher.
- **Search**: Operates on parsed AST word content rather than raw text lines.
- **Word identification**: Uses AST-based `is_countable_word()` instead of CLAN's string-prefix matching.
- **Scope reduction**: The legacy manual describes richer tier-selection and output-shaping behavior, including cluster searches over selected dependent tiers and `%mor`/`%gra` combined searches with `+d7`. The current implementation is narrower.
- **Output formats**: Supports text, JSON, and CSV formats (CLAN produces text only).
- **Golden test parity**: Verified against CLAN C binary output.
