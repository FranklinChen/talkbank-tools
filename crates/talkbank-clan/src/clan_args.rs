//! CLAN argument pre-processor.
//!
//! Rewrites legacy CLAN `+flag`/`-flag` syntax into modern `--flag` equivalents
//! so that clap can parse them. This allows users to write either:
//!
//! ```text
//! clan analyze freq +t*CHI +s"want" +z25-125 file.cha
//! ```
//!
//! or the modern equivalent:
//!
//! ```text
//! clan analyze freq --speaker CHI --include-word want --range 25-125 file.cha
//! ```
//!
//! The rewriter is a pure function that operates on the raw argument list before
//! clap sees it. It only touches arguments that look like CLAN flags (`+` or `-`
//! prefix followed by a known flag letter); everything else passes through unchanged.

/// Rewrite CLAN-style `+flag`/`-flag` arguments into modern `--flag` equivalents.
///
/// The function scans `args` for patterns like `+t*CHI`, `+s"word"`, `+z25-125`,
/// etc., and replaces them with `--speaker CHI`, `--include-word word`,
/// `--range 25-125`, etc. Unrecognised arguments pass through unchanged.
///
/// This is intentionally applied to the full argument list (including the binary
/// name and subcommand tokens). Subcommand names like `analyze`, `freq`, etc.
/// never start with `+` or `-` followed by a CLAN flag letter, so they are
/// never matched.
///
/// The rewriter is context-aware for the `check` subcommand: `+g1`–`+g5` are
/// CHECK generic options (not gem labels), so they are rewritten to
/// `--check-target`, `--check-id`, `--check-unused` etc. For all other
/// subcommands, `+g` is gem filtering as usual.
/// The set of CLAN analysis subcommands chatter knows about for
/// the purpose of per-subcommand `+`-flag dispatch.
///
/// CLAN's `+`-flag semantics depend on which analysis command the
/// user invoked: `+cN` is `--bullets` under CHECK, `--limit` under
/// MAXWD, and `--max-utterances` under IPSYN/DSS. The rewriter
/// needs to know which subcommand is active to pick the right
/// rewrite. This enum captures the subset of subcommand identities
/// the rewriter currently branches on. Subcommands not enumerated
/// here use the inherited general semantic for every flag.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ClanSubcommandKind {
    Check,
    Maxwd,
    Ipsyn,
    Dss,
    Mortable,
    Script,
    Uniq,
    Mlu,
    Mlt,
    Sugar,
    Keymap,
    Makemod,
    Lines,
    Ort,
    Fixbullets,
    Combo,
    Freq,
    Vocd,
    Dist,
    Kwal,
    Wdsize,
    Freqpos,
    Cooccur,
    Lowcase,
    Combtier,
    Chains,
    Modrep,
    Other,
}

impl ClanSubcommandKind {
    fn detect(args: &[String]) -> Self {
        // The CLAN subcommand is always the first non-flag token in
        // args after position 0 (typically index 1, but a leading
        // global flag can push it back). Scan from left to right for
        // the first known subcommand name.
        for arg in args {
            match arg.as_str() {
                "check" => return Self::Check,
                "maxwd" => return Self::Maxwd,
                "ipsyn" => return Self::Ipsyn,
                "dss" => return Self::Dss,
                "mortable" => return Self::Mortable,
                "script" => return Self::Script,
                "uniq" => return Self::Uniq,
                "mlu" => return Self::Mlu,
                "mlt" => return Self::Mlt,
                "sugar" => return Self::Sugar,
                "keymap" => return Self::Keymap,
                "makemod" => return Self::Makemod,
                "lines" => return Self::Lines,
                "ort" => return Self::Ort,
                "fixbullets" => return Self::Fixbullets,
                "combo" => return Self::Combo,
                "freq" => return Self::Freq,
                "vocd" => return Self::Vocd,
                "dist" => return Self::Dist,
                "kwal" => return Self::Kwal,
                "wdsize" => return Self::Wdsize,
                "freqpos" => return Self::Freqpos,
                "cooccur" => return Self::Cooccur,
                "lowcase" => return Self::Lowcase,
                "combtier" => return Self::Combtier,
                "chains" => return Self::Chains,
                "modrep" => return Self::Modrep,
                _ => {}
            }
        }
        Self::Other
    }
}

pub fn rewrite_clan_args(args: &[String]) -> Vec<String> {
    let subcommand = ClanSubcommandKind::detect(args);

    let mut out = Vec::with_capacity(args.len());
    let mut i = 0;

    while i < args.len() {
        let arg = &args[i];

        // Only attempt rewriting on args starting with + or - that look like
        // CLAN flags (second char is a known flag letter, not a digit or '-').
        if let Some(rewritten) = try_rewrite_clan_flag(arg, subcommand) {
            out.extend(rewritten);
            i += 1;
            continue;
        }

        // Pass through unchanged.
        out.push(arg.clone());
        i += 1;
    }

    out
}

/// Attempt to rewrite a single CLAN-style argument.
///
/// Returns `Some(vec![...])` with the replacement tokens, or `None` if the
/// argument is not a recognised CLAN flag.
fn try_rewrite_clan_flag(arg: &str, subcommand: ClanSubcommandKind) -> Option<Vec<String>> {
    use ClanSubcommandKind::*;
    let bytes = arg.as_bytes();
    if bytes.len() < 2 {
        return None;
    }

    let polarity = bytes[0];
    if polarity != b'+' && polarity != b'-' {
        return None;
    }

    let flag_char = bytes[1];
    let rest = &arg[2..];

    match (polarity, flag_char) {
        // +t*CHI / -t*CHI — speaker include/exclude
        // MLU / MLT `-t%mor` — CLAN's documented escape hatch:
        // when `%mor` is present but the user wants word-mode
        // counts, `-t%mor` implies `--words`. Without this special-
        // case, the rewriter would emit `--exclude-tier mor` which
        // MLU's/MLT's clap doesn't accept. Scoped to `%mor` only;
        // other `-t%X` values fall through to the generic
        // exclude-tier path.
        (b'-', b't') if matches!(subcommand, Mlu | Mlt) && rest == "%mor" => {
            Some(vec!["--words".into()])
        }

        // COMBTIER `+tS` (bare-prefix form) — tier label to combine,
        // per `OSX-CLAN/src/clan/combtier.cpp` usage
        // ("+tS: Combine all tiers S into one tier."). COMBTIER
        // overloads `+tS` away from the analysis-command convention
        // (`+tCHI` = speaker filter); route the bareword form to
        // `--tier S` instead of letting `rewrite_tier_speaker`
        // produce `--speaker S`. The `+t%X` form is handled by
        // `rewrite_tier_speaker`'s `%` branch, which also emits
        // `--tier X`, so only the bare-prefix case needs intercept.
        (b'+', b't')
            if subcommand == Combtier
                && !rest.is_empty()
                && !matches!(rest.as_bytes()[0], b'*' | b'%' | b'@' | b'#') =>
        {
            rewrite_subcommand_value_flag(rest, "--tier")
        }

        (b'+', b't') | (b'-', b't') => rewrite_tier_speaker(polarity, rest),

        // MLU / MLT `-bw` — switch the counting unit from morphemes
        // (`%mor`-based, the default) to words (main-tier-based).
        // Without this arm, `-bw` falls through to clap, which parses
        // it as a `-b -w` short-flag pair and errors on the unknown
        // `-b`. Scoped to Mlu | Mlt; other commands don't share the
        // morphemes-vs-words counting axis.
        (b'-', b'b') if matches!(subcommand, Mlu | Mlt) && rest == "w" => {
            Some(vec!["--words".into()])
        }

        // +s"word" / +sword / -s"word" / -sword — word include/exclude
        // `+sF` under SCRIPT is the template-file argument
        // (`--template F`); SCRIPT's `+s` is the only CLAN command
        // where the value is interpreted as a filesystem path
        // rather than a search keyword.
        (b'+', b's') if subcommand == Script => rewrite_subcommand_value_flag(rest, "--template"),
        // COMBO's `+s@FILE` / `-s@FILE` load boolean search
        // expressions from disk (one per line). Routed to
        // dedicated `--search-file` / `--exclude-search-file`
        // because COMBO's per-line value is a `SearchExpr`, not
        // a per-word pattern — must precede the generic
        // `+s@`/`-s@` word-file arms below.
        (b'+', b's') if subcommand == Combo && rest.starts_with('@') => {
            rewrite_subcommand_value_flag(&rest[1..], "--search-file")
        }
        (b'-', b's') if subcommand == Combo && rest.starts_with('@') => {
            rewrite_subcommand_value_flag(&rest[1..], "--exclude-search-file")
        }
        // COMBO's `+sS` / `-sS` are compound boolean expressions
        // (e.g. `want+cookie`, `want,milk`), distinct from the
        // general per-word `+s` include/exclude — route to
        // `--search` / `--exclude-search`.
        (b'+', b's') if subcommand == Combo => rewrite_subcommand_value_flag(rest, "--search"),
        (b'-', b's') if subcommand == Combo => {
            rewrite_subcommand_value_flag(rest, "--exclude-search")
        }
        // +s@FILE / -s@FILE — load word-list from file (CLAN's
        // `cutt.cpp::rdexclf`). Ordered after the SCRIPT and
        // COMBO command-specific arms because those commands'
        // `+s` value isn't a per-word pattern and the `@FILE`
        // semantic differs.
        (b'+', b's') if rest.starts_with('@') => {
            rewrite_subcommand_value_flag(&rest[1..], "--include-word-file")
        }
        (b'-', b's') if rest.starts_with('@') => {
            rewrite_subcommand_value_flag(&rest[1..], "--exclude-word-file")
        }
        (b'+', b's') | (b'-', b's') => rewrite_search_word(polarity, rest),

        // +g: command-dependent.
        //   * CHECK       → generic options (`+g1`..`+g5` map to
        //                   `--check-target` / `--check-id` / etc.)
        //   * MLU / MLT   → solo-word elision (drop utterances
        //                   consisting solely of word S):
        //                   `+gS` → `--exclude-solo-word S`.
        //                   CLAN's MLU/MLT `getflag()` intercepts `+g`
        //                   before the inherited gem semantic; chatter
        //                   matches by routing here. Documented as the
        //                   "+g overload" pattern in the parity audit.
        //   * other       → gem-segment filter (`--gem S`).
        (b'+', b'g') if subcommand == Check => rewrite_check_generic(rest),
        // MLU/MLT `+g@F` loads the solo-word exclusion list from a
        // file (same idiom as `+s@F` → `--include-word-file`).
        // Must precede the per-word `+gS` arm so the `@`-prefix is
        // intercepted before being treated as a literal pattern.
        (b'+', b'g') if matches!(subcommand, Mlu | Mlt) && rest.starts_with('@') => {
            rewrite_subcommand_value_flag(&rest[1..], "--exclude-solo-word-file")
        }
        (b'+', b'g') if matches!(subcommand, Mlu | Mlt) => {
            rewrite_subcommand_value_flag(rest, "--exclude-solo-word")
        }
        // COMBO `+gN` search-mode switches (CLAN's `+g1..+g7`). Most
        // are documented gaps; the ones below are wired:
        //   * `+g3` — only the first matching expression per
        //     utterance → `--first-match-only`.
        //   * `+g4` — exclude utterance delimiters from search.
        //     chatter's COMBO operates on `countable_words`, which
        //     never returns terminators/separators — so `+g4` is
        //     the chatter default. No-op accept.
        //   * `+g5` — use `+` (or `^`) as AND operator. chatter's
        //     `+` is already AND by default, so `+g5` is a no-op
        //     accept; rewriter consumes the flag (`Some(vec![])`)
        //     so clap never sees it.
        //   * `+g7` — deduplicate repeated word matches within an
        //     utterance → `--dedupe-matches`.
        (b'+', b'g') if subcommand == Combo && rest == "3" => {
            Some(vec!["--first-match-only".into()])
        }
        (b'+', b'g') if subcommand == Combo && rest == "4" => Some(Vec::new()),
        (b'+', b'g') if subcommand == Combo && rest == "5" => Some(Vec::new()),
        (b'+', b'g') if subcommand == Combo && rest == "7" => Some(vec!["--dedupe-matches".into()]),
        // DIST's bare `+g` is a counting policy ("one occurrence
        // per turn"), distinct from the inherited `+gLABEL` gem
        // filter. Only the no-rest form routes here; `+gLABEL`
        // falls through to the gem branch.
        (b'+', b'g') if subcommand == Dist && rest.is_empty() => {
            Some(vec!["--once-per-turn".into()])
        }
        (b'+', b'g') | (b'-', b'g') => rewrite_gem(polarity, rest),

        // +aN under SUGAR sets the minimum-utterance threshold
        // (CLAN docs: "set minimal utterances number limit to N
        // utterances (default: 50 minimal limit)"). Routes to
        // `--min-utterances N`. SUGAR is the only command with
        // this `+aN` semantic; other commands either don't use
        // `+a` or use it as a different flag.
        (b'+', b'a') if subcommand == Sugar => {
            rewrite_subcommand_value_flag(rest, "--min-utterances")
        }

        // `+a` under MAKEMOD is a no-value boolean — print all
        // alternative pronunciations (default: first only). Routes
        // to `--all-alternatives`.
        (b'+', b'a') if rest.is_empty() && subcommand == Makemod => {
            Some(vec!["--all-alternatives".into()])
        }

        // `+n` under LINES is a no-value boolean — remove existing
        // line numbers (default: add them). Routes to `--remove`.
        (b'+', b'n') if rest.is_empty() && subcommand == Lines => Some(vec!["--remove".into()]),

        // `+cF` under ORT specifies the homons-table dictionary.
        // Maps `+ceng.cut` → `--dictionary eng.cut`.
        (b'+', b'c') if !rest.is_empty() && subcommand == Ort => {
            rewrite_subcommand_value_flag(rest, "--dictionary")
        }

        // +bS under KEYMAP sets a key-code to track. Routes to
        // `--keyword S` (repeatable). KEYMAP's `+b` semantic is
        // distinct from FREQ's `+bN` (MATTR frame size), MLU's `-bw`
        // (word-mode toggle), WDLEN/MAXWD's `+bS`/`-bS` (morpheme
        // delimiters) — those are documented audit gaps tracked
        // under Phase 1.7 follow-ups and remain unrewritten.
        // `+b@F` (key-codes-from-file) is also unrewritten today.
        (b'+', b'b') if subcommand == Keymap && !rest.starts_with('@') => {
            rewrite_subcommand_value_flag(rest, "--keyword")
        }

        // +z25-125 — utterance range
        (b'+', b'z') => rewrite_range(rest),

        // +r6 — include retracings
        (b'+', b'r') if rest == "6" => Some(vec!["--include-retracings".into()]),

        // +re — recurse subdirectories. chatter recurses by default
        // when given a directory argument, so the flag is a global
        // no-op (same shape as `+u` on non-CHECK commands above).
        // Drop the token rather than passing it through to clap,
        // which would land it in the path-arg list and emit a
        // confusing `Warning: "+re" is not a file or directory`.
        (b'+', b'r') if rest == "e" => Some(vec![]),

        // +u: For CHECK, +u means validate UD features; for other commands, merge speakers (no-op)
        (b'+', b'u') if rest.is_empty() && subcommand == Check => Some(vec!["--check-ud".into()]),
        (b'+', b'u') if rest.is_empty() => Some(vec![]),

        // FREQPOS `+d` (no N) switches position classification
        // from first/last/other to first/second/other. Intercepted
        // before the generic +dN display-mode arm so the bare-`d`
        // form isn't lost to the empty-rest short-circuit.
        (b'+', b'd') if subcommand == Freqpos && rest.is_empty() => {
            Some(vec!["--position-classification".into(), "second".into()])
        }

        // COOCCUR `+d` (no N) strips the leading count column from
        // the output. Same empty-rest intercept pattern.
        (b'+', b'd') if subcommand == Cooccur && rest.is_empty() => {
            Some(vec!["--no-frequency-counts".into()])
        }

        // COOCCUR `+nN` sets the cluster size (number of adjacent
        // words counted as a unit). Default 2 = bigrams; +n3 =
        // trigrams; etc. Rejected with no rest (just `+n`) because
        // CLAN requires the N value.
        (b'+', b'n') if subcommand == Cooccur && rest.parse::<u8>().is_ok() => {
            Some(vec!["--cluster-size".into(), rest.to_string()])
        }

        // KWAL `+d` (no N) switches the output from CLAN's
        // location-annotated default to a legal CHAT fragment
        // (drop the `---` separator and `*** File ... Keyword: X`
        // decoration). `+d1`/`+d2`/... and other KWAL display
        // modes still fall through to the generic `--display-mode
        // N` arm.
        (b'+', b'd') if subcommand == Kwal && rest.is_empty() => Some(vec!["--legal-chat".into()]),

        // FREQ `+d1` emits one word per line with no frequency or
        // other info — suitable as a `kwal +s@FILE` input. Other
        // `+dN` FREQ display modes (0, 2..8) still fall through to
        // the generic `--display-mode N` rewrite below.
        (b'+', b'd') if subcommand == Freq && rest == "1" => Some(vec!["--word-list-only".into()]),

        // FREQ `+d4` emits only per-speaker type/token/TTR summary
        // (no per-word entries). `+d3` (same content, spreadsheet
        // form) is a separate item that combines this with CSV
        // output.
        (b'+', b'd') if subcommand == Freq && rest == "4" => {
            Some(vec!["--types-tokens-only".into()])
        }

        // FREQ `+d3` is `+d4` content in spreadsheet (CSV) form —
        // the rewriter emits both the types-tokens-only filter and
        // the `--format csv` selector together. `+d2` (same
        // spreadsheet form but with per-word rows) is a separate
        // item.
        (b'+', b'd') if subcommand == Freq && rest == "3" => Some(vec![
            "--types-tokens-only".into(),
            "--format".into(),
            "csv".into(),
        ]),

        // FREQ `+d2`: spreadsheet/Excel output of the per-speaker
        // per-word frequency table. The existing `render_csv` path
        // already produces this; the rewriter only needs to flip
        // the format selector.
        (b'+', b'd') if subcommand == Freq && rest == "2" => {
            Some(vec!["--format".into(), "csv".into()])
        }

        // LOWCASE `+d2` — "ignore dict file, lowercase everything",
        // per `OSX-CLAN/src/clan/lowcase.cpp` case 'd' (integer 0..=2
        // toggles dict-preserving / dict-capitalizing / ignore-dict).
        // chatter's `transforms/lowcase.rs` lowercases unconditionally,
        // matching the `+d2` semantic, so the flag is a no-op.
        // Intercepted before the generic `+dN → --display-mode N`
        // catch-all; lowcase has no `--display-mode` clap field.
        // `+d`/`+d0`/`+d1` (dict-using modes) are documented Missing
        // and intentionally still fall through to fail clap.
        (b'+', b'd') if subcommand == Lowcase && rest == "2" => Some(vec![]),

        // CHAINS `+d`/`+dN` — `onlydata` output-detail level (0-1
        // per `OSX-CLAN/src/clan/chains.cpp:1089`: `+d` → 1,
        // `+d0` → 1, `+d1` → 2). chatter has no `--only-data`
        // flag; pass through so clap rejects the literal token
        // rather than the misleading `--display-mode` rewrite from
        // the catch-all below.
        (b'+', b'd') if subcommand == Chains => None,

        // MODREP `+d` — no-arg Excel/spreadsheet toggle per
        // `OSX-CLAN/src/clan/modrep.cpp:1492` (`no_arg_option(f)`
        // + `isExcel = TRUE`). chatter has no `--format csv` for
        // MODREP; pass through.
        (b'+', b'd') if subcommand == Modrep => None,

        // IPSYN `+d`/`+dN` — `onlydata` output-detail level
        // bounded by `OnlydataLimit` per `OSX-CLAN/src/clan/ipsyn.cpp:3945`.
        // chatter has no `--only-data` flag; pass through.
        (b'+', b'd') if subcommand == Ipsyn => None,

        // +dN — display mode
        (b'+', b'd') => rewrite_display_mode(rest),

        // +k — case sensitive
        (b'+', b'k') if rest.is_empty() => Some(vec!["--case-sensitive".into()]),

        // +fEXT — output extension
        (b'+', b'f') if !rest.is_empty() => Some(vec!["--output-ext".into(), rest.to_string()]),

        // WDSIZE `+w[>|<|=]N` — length-bounded histogram. Intercept
        // before the general `+wN` context-window arm: presence of
        // a leading comparator (`>`, `<`, or `=`) disambiguates
        // the length-filter form from the inherited context-window
        // form (`+w3` etc.). Match-guard binds the parsed result
        // so we parse `rest` exactly once.
        (b'+', b'w')
            if subcommand == Wdsize
                && let Some(args) = rewrite_wdsize_length_filter(rest) =>
        {
            Some(args)
        }

        // +wN / -wN — context window
        (b'+', b'w') => rewrite_context_window("+w", rest),
        (b'-', b'w') => rewrite_context_window("-w", rest),

        // `+cN` is subcommand-dependent:
        //   * CHECK       → bullet check level (`--bullets N`)
        //   * MAXWD       → number of longest items to display (`--limit N`)
        //   * IPSYN / DSS → max utterances to analyse (`--max-utterances N`)
        //   * other       → no rewrite today; FREQ's `+c0..7` (capitalised-
        //                   word and multi-word search variants) and VOCD's
        //                   `+c` (capitalised-only) are documented gaps,
        //                   tracked under Phase 1.7 follow-ups.
        (b'+', b'c') if subcommand == Maxwd => rewrite_subcommand_value_flag(rest, "--limit"),
        // MAXWD `+a` — restrict to words whose length is unique
        // within a speaker's lexicon (CLAN: "Consider ONLY unique-
        // length words"). No CLAN `+aN` variant exists.
        (b'+', b'a') if subcommand == Maxwd && rest.is_empty() => {
            Some(vec!["--unique-length-only".into()])
        }
        // MAXWD `+xN` — drop words of character length N from
        // output. Repeatable in CLAN argv (`+x5 +x7`); each rewrite
        // emits an `--exclude-length N` argv pair. The numeric
        // guard ensures non-numeric `+x<S>` (other-command futures)
        // doesn't accidentally route here.
        (b'+', b'x') if subcommand == Maxwd && rest.parse::<usize>().is_ok() => {
            rewrite_subcommand_value_flag(rest, "--exclude-length")
        }
        // KWAL `+b` — strict-match: keyword must be the *only*
        // item on the tier (single-word utterance). No CLAN `+bS`
        // variant exists for KWAL.
        (b'+', b'b') if subcommand == Kwal && rest.is_empty() => {
            Some(vec!["--strict-match".into()])
        }
        (b'+', b'c') if matches!(subcommand, Ipsyn | Dss) => {
            rewrite_subcommand_value_flag(rest, "--max-utterances")
        }
        (b'+', b'c') if subcommand == Check => rewrite_subcommand_value_flag(rest, "--bullets"),
        // FREQ / VOCD `+c` / `+c0` / `+c1` — capitalization-mode
        // filter. Both commands share the `--capitalization` enum-
        // valued clap field (`initial` or `mid`). CLAN spellings:
        //   * `+c` / `+c0` → `--capitalization initial`
        //   * `+c1`        → `--capitalization mid`
        // VOCD's manual lists only `+c`; FREQ extends to `+c1`.
        (b'+', b'c') if matches!(subcommand, Freq | Vocd) && (rest.is_empty() || rest == "0") => {
            Some(vec!["--capitalization".into(), "initial".into()])
        }
        (b'+', b'c') if matches!(subcommand, Freq | Vocd) && rest == "1" => {
            Some(vec!["--capitalization".into(), "mid".into()])
        }

        // `+lF` is subcommand-dependent:
        //   * IPSYN / DSS → rules file (`--rules F`)
        //   * MORTABLE    → language script file (`--script F`)
        (b'+', b'l') if matches!(subcommand, Ipsyn | Dss) => {
            rewrite_subcommand_value_flag(rest, "--rules")
        }
        (b'+', b'l') if subcommand == Mortable => rewrite_subcommand_value_flag(rest, "--script"),

        // `-o` under UNIQ is the sort-by-frequency switch
        // (`--sort`). UNIQ is the only CLAN command with a meaningful
        // `-o` (other commands' `-o` excludes an extra output tier,
        // which is not yet wired in chatter).
        (b'-', b'o') if rest.is_empty() && subcommand == Uniq => Some(vec!["--sort".into()]),

        // FREQ `+o` / `+o0` — descending-frequency sort. chatter's
        // FREQ result sorts by count descending unconditionally
        // (`crates/talkbank-clan/src/commands/freq.rs` finalize),
        // so the flag is a no-op. Without this arm `+o` survives
        // to clap as a path arg and triggers
        // `Warning: "+o" is not a file or directory`. Match before
        // the `+o1` arm so the `rest.is_empty()` / `rest == "0"`
        // guards take precedence over the catch-all `rest == "1"`
        // check. `+o2` (non-CHAT spreadsheet output) is a separate
        // documented gap — falls through to default.
        (b'+', b'o') if subcommand == Freq && (rest.is_empty() || rest == "0") => Some(vec![]),

        // FREQ `+o1` — sort by reverse concordance. `+o` / `+o0`
        // handled above; `+o2` (non-CHAT spreadsheet output) is a
        // separate documented gap — not handled here.
        (b'+', b'o') if subcommand == Freq && rest == "1" => {
            Some(vec!["--reverse-concordance".into()])
        }

        // COOCCUR `+o` — descending-frequency sort. chatter's
        // COOCCUR `finalize` step at
        // `crates/talkbank-clan/src/commands/cooccur.rs:292` already
        // sorts by `count` descending, then alphabetically; CLAN's
        // `cooccur.cpp` uses a BST with `larger num_occ goes left`
        // invariant so in-order traversal produces the same
        // descending order. No-op rewrite drops the token.
        (b'+', b'o') if subcommand == Cooccur && rest.is_empty() => Some(vec![]),

        // `+oN` / `-oN` under FIXBULLETS specify a signed time-offset
        // shift in milliseconds (`+o800` adds 800 ms, `-o800`
        // subtracts 800 ms). FIXBULLETS overloads `+o` here away from
        // the general "include extra output tier" semantic; the
        // numeric guard distinguishes the two — `+oS` with a non-
        // numeric `S` (extra tier code) falls through unchanged.
        (b'+', b'o') if subcommand == Fixbullets && rest.parse::<u32>().is_ok() => {
            Some(vec!["--offset".into(), rest.to_string()])
        }
        (b'-', b'o') if subcommand == Fixbullets && rest.parse::<u32>().is_ok() => {
            Some(vec!["--offset".into(), format!("-{rest}")])
        }
        // +eN — include error / +e — list errors
        (b'+', b'e') => rewrite_check_error(rest),
        // -eN — exclude error
        (b'-', b'e') if !rest.is_empty() => Some(vec!["--exclude-error".into(), rest.to_string()]),

        _ => None,
    }
}

/// Rewrite `+t*CHI` → `--speaker CHI`, `-t*MOT` → `--exclude-speaker MOT`,
/// `+t%mor` → `--tier mor`, `-t%gra` → `--exclude-tier gra`.
///
/// CLAN also accepts `+tCHI` (no `*` sigil) and treats it identically
/// to `+t*CHI`; this function does the same — when the first character
/// of the value is not one of `*`, `%`, or `@`, the value is taken as
/// an implicit speaker code.
fn rewrite_tier_speaker(polarity: u8, rest: &str) -> Option<Vec<String>> {
    if rest.is_empty() {
        return None;
    }

    match rest.as_bytes()[0] {
        b'*' => {
            let speaker = &rest[1..];
            if speaker.is_empty() {
                return None;
            }
            let flag = if polarity == b'+' {
                "--speaker"
            } else {
                "--exclude-speaker"
            };
            Some(vec![flag.into(), speaker.to_string()])
        }
        b'%' => {
            let tier = &rest[1..];
            if tier.is_empty() {
                return None;
            }
            let flag = if polarity == b'+' {
                "--tier"
            } else {
                "--exclude-tier"
            };
            Some(vec![flag.into(), tier.to_string()])
        }
        b'@' => {
            // +t@ID="eng|*|CHI|*" → --id-filter "eng|*|CHI|*"
            if rest.len() >= 4 && rest[1..].starts_with("ID=") {
                let value = strip_quotes(&rest[4..]);
                if value.is_empty() {
                    return None;
                }
                Some(vec!["--id-filter".into(), value])
            } else {
                None
            }
        }
        b'#' => {
            // `+t#Target_Child` → `--role Target_Child`.
            // The `-t#ROLE` exclude-by-role form is not currently
            // supported by CLAN (per `mainusage()` the role flag is
            // include-only), so polarity `b'-'` falls through to the
            // default branch below and is treated as a literal
            // speaker code, matching CLAN's `+tCHI`/`-tCHI` shape.
            if polarity != b'+' {
                return None;
            }
            let role = &rest[1..];
            if role.is_empty() {
                return None;
            }
            Some(vec!["--role".into(), role.to_string()])
        }
        _ => {
            // `+tCHI` / `-tMOT` — CLAN treats the value as an implicit
            // speaker code (equivalent to `+t*CHI` / `-t*MOT`). Match
            // that behaviour.
            let flag = if polarity == b'+' {
                "--speaker"
            } else {
                "--exclude-speaker"
            };
            Some(vec![flag.into(), rest.to_string()])
        }
    }
}

/// Rewrite `+s"word"` or `+sword` → `--include-word word`,
/// `-s"word"` or `-sword` → `--exclude-word word`.
fn rewrite_search_word(polarity: u8, rest: &str) -> Option<Vec<String>> {
    if rest.is_empty() {
        return None;
    }
    let word = strip_quotes(rest);
    if word.is_empty() {
        return None;
    }
    let flag = if polarity == b'+' {
        "--include-word"
    } else {
        "--exclude-word"
    };
    Some(vec![flag.into(), word])
}

/// Rewrite `+glabel` → `--gem label`, `-glabel` → `--exclude-gem label`.
fn rewrite_gem(polarity: u8, rest: &str) -> Option<Vec<String>> {
    if rest.is_empty() {
        return None;
    }
    let label = strip_quotes(rest);
    if label.is_empty() {
        return None;
    }
    let flag = if polarity == b'+' {
        "--gem"
    } else {
        "--exclude-gem"
    };
    Some(vec![flag.into(), label])
}

/// Rewrite `+z25-125` → `--range 25-125`.
fn rewrite_range(rest: &str) -> Option<Vec<String>> {
    if rest.is_empty() {
        return None;
    }
    Some(vec!["--range".into(), rest.to_string()])
}

/// Build a `[long_flag, value]` token pair for the simple `+X<value>`
/// shape shared by the per-subcommand routing branches
/// (`+cN`/`+lF`/`+sF`/`+gS`/`+aN`). Returns `None` when there is no
/// value (the caller treats that as "not this branch"); the caller is
/// responsible for the subcommand guard.
fn rewrite_subcommand_value_flag(rest: &str, long_flag: &str) -> Option<Vec<String>> {
    if rest.is_empty() {
        return None;
    }
    Some(vec![long_flag.into(), rest.to_string()])
}

/// Rewrite `+dN` → `--display-mode N`.
fn rewrite_display_mode(rest: &str) -> Option<Vec<String>> {
    if rest.is_empty() {
        return None;
    }
    // Validate that rest is a number
    if rest.chars().all(|c| c.is_ascii_digit()) {
        Some(vec!["--display-mode".into(), rest.to_string()])
    } else {
        None
    }
}

/// Rewrite `+wN` → `--context-after N`, `-wN` → `--context-before N`.
/// Parse WDSIZE's `+w[>|<|=]N` length-filter argument and emit an
/// `--length-filter <gt|lt|eq>:N` argv pair. Returns `None` when
/// the input doesn't lead with a recognized comparator — in which
/// case the caller falls through to the general `+wN` context-
/// window rewrite. CLAN's WDSIZE only documents these three
/// comparators.
fn rewrite_wdsize_length_filter(rest: &str) -> Option<Vec<String>> {
    let bytes = rest.as_bytes();
    if bytes.is_empty() {
        return None;
    }
    let (tag, n_str) = match bytes[0] {
        b'>' => ("gt", &rest[1..]),
        b'<' => ("lt", &rest[1..]),
        b'=' => ("eq", &rest[1..]),
        _ => return None,
    };
    n_str.parse::<usize>().ok()?;
    Some(vec!["--length-filter".into(), format!("{tag}:{n_str}")])
}

fn rewrite_context_window(prefix: &str, rest: &str) -> Option<Vec<String>> {
    if rest.is_empty() {
        return None;
    }
    if !rest.chars().all(|c| c.is_ascii_digit()) {
        return None;
    }
    let flag = if prefix == "+w" {
        "--context-after"
    } else {
        "--context-before"
    };
    Some(vec![flag.into(), rest.to_string()])
}

/// Strip surrounding double quotes from a string value.
fn strip_quotes(s: &str) -> String {
    let s = s.trim();
    if s.len() >= 2 && s.starts_with('"') && s.ends_with('"') {
        s[1..s.len() - 1].to_string()
    } else {
        s.to_string()
    }
}

/// Rewrite CHECK's `+g1`–`+g5` generic options.
///
/// | Flag | Meaning |
/// |------|---------|
/// | `+g1` | Check prosodic delimiters (no-op — always on) |
/// | `+g2` | Check CHI has Target_Child role |
/// | `+g3` | Word detail checks (partially implemented via parser) |
/// | `+g4` | Check for missing @ID tiers (on by default) |
/// | `+g5` | Check for unused speakers |
///
/// Falls back to gem rewriting if the rest is not a single digit 1–5.
fn rewrite_check_generic(rest: &str) -> Option<Vec<String>> {
    match rest {
        "1" => Some(vec![]), // no-op: prosodic delimiters always recognized
        "2" => Some(vec!["--check-target".into()]),
        "3" => Some(vec![]), // no-op: word checks via parser
        "4" => Some(vec!["--check-id".into(), "true".into()]),
        "5" => Some(vec!["--check-unused".into()]),
        // Not a CHECK generic option — fall back to gem
        _ => rewrite_gem(b'+', rest),
    }
}

/// Rewrite `+eN` → `--error N`, `+e` → `--list-errors`.
fn rewrite_check_error(rest: &str) -> Option<Vec<String>> {
    if rest.is_empty() {
        Some(vec!["--list-errors".into()])
    } else {
        Some(vec!["--error".into(), rest.to_string()])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args(s: &str) -> Vec<String> {
        s.split_whitespace().map(String::from).collect()
    }

    #[test]
    fn speaker_include() {
        let input = args("clan analyze freq +t*CHI file.cha");
        let result = rewrite_clan_args(&input);
        assert_eq!(result, args("clan analyze freq --speaker CHI file.cha"));
    }

    #[test]
    fn speaker_exclude() {
        let input = args("clan analyze freq -t*MOT file.cha");
        let result = rewrite_clan_args(&input);
        assert_eq!(
            result,
            args("clan analyze freq --exclude-speaker MOT file.cha")
        );
    }

    #[test]
    fn multiple_speakers() {
        let input = args("clan analyze freq +t*CHI +t*MOT file.cha");
        let result = rewrite_clan_args(&input);
        assert_eq!(
            result,
            args("clan analyze freq --speaker CHI --speaker MOT file.cha")
        );
    }

    /// CLAN silently treats `+tCHI` (no `*` sigil) the same as
    /// `+t*CHI` — the sigil is implicit when the first character is
    /// not `*`, `%`, or `@`. chatter must do the same so a user
    /// pasting `freq +tCHI file.cha` from a CLAN script reaches the
    /// `--speaker` field, not the fallthrough that drops the flag.
    /// Asymmetrically true for `-tCHI` → `--exclude-speaker CHI`.
    #[test]
    fn speaker_include_no_asterisk() {
        let input = args("clan analyze freq +tCHI file.cha");
        let result = rewrite_clan_args(&input);
        assert_eq!(result, args("clan analyze freq --speaker CHI file.cha"));
    }

    #[test]
    fn speaker_exclude_no_asterisk() {
        let input = args("clan analyze freq -tMOT file.cha");
        let result = rewrite_clan_args(&input);
        assert_eq!(
            result,
            args("clan analyze freq --exclude-speaker MOT file.cha")
        );
    }

    #[test]
    fn tier_include() {
        let input = args("clan analyze freq +t%mor file.cha");
        let result = rewrite_clan_args(&input);
        assert_eq!(result, args("clan analyze freq --tier mor file.cha"));
    }

    #[test]
    fn tier_exclude() {
        let input = args("clan analyze freq -t%gra file.cha");
        let result = rewrite_clan_args(&input);
        assert_eq!(
            result,
            args("clan analyze freq --exclude-tier gra file.cha")
        );
    }

    #[test]
    fn search_word_quoted() {
        let input: Vec<String> = vec![
            "clan".into(),
            "analyze".into(),
            "freq".into(),
            "+s\"want\"".into(),
            "file.cha".into(),
        ];
        let result = rewrite_clan_args(&input);
        assert_eq!(
            result,
            args("clan analyze freq --include-word want file.cha")
        );
    }

    #[test]
    fn search_word_unquoted() {
        let input = args("clan analyze freq +swant file.cha");
        let result = rewrite_clan_args(&input);
        assert_eq!(
            result,
            args("clan analyze freq --include-word want file.cha")
        );
    }

    #[test]
    fn exclude_word() {
        let input = args("clan analyze freq -swant file.cha");
        let result = rewrite_clan_args(&input);
        assert_eq!(
            result,
            args("clan analyze freq --exclude-word want file.cha")
        );
    }

    #[test]
    fn gem_include() {
        let input = args("clan analyze freq +gstory file.cha");
        let result = rewrite_clan_args(&input);
        assert_eq!(result, args("clan analyze freq --gem story file.cha"));
    }

    #[test]
    fn gem_exclude() {
        let input = args("clan analyze freq -gstory file.cha");
        let result = rewrite_clan_args(&input);
        assert_eq!(
            result,
            args("clan analyze freq --exclude-gem story file.cha")
        );
    }

    #[test]
    fn utterance_range() {
        let input = args("clan analyze freq +z25-125 file.cha");
        let result = rewrite_clan_args(&input);
        assert_eq!(result, args("clan analyze freq --range 25-125 file.cha"));
    }

    #[test]
    fn mlu_minus_bw_to_words() {
        // CLAN `-bw` on MLU/MLT switches the counting unit from
        // morphemes to words. The audit page lists this as a Done
        // mapping (`-bw` → `--words`), but the rewriter had no arm
        // for `-bw` — only a stale comment. clap parsed `-bw` as a
        // short-flag-with-value form and errored on the unknown
        // `-b`. This test guards the new Mlu/Mlt-scoped arm.
        let input = args("clan analyze mlu -bw file.cha");
        let result = rewrite_clan_args(&input);
        assert_eq!(result, args("clan analyze mlu --words file.cha"));
    }

    #[test]
    fn mlt_minus_bw_to_words() {
        let input = args("clan analyze mlt -bw file.cha");
        let result = rewrite_clan_args(&input);
        assert_eq!(result, args("clan analyze mlt --words file.cha"));
    }

    #[test]
    fn freq_minus_bw_unchanged() {
        // The `-bw` rewrite is scoped to MLU/MLT — other commands
        // don't share the morphemes-vs-words counting axis, so
        // `-bw` should fall through unchanged for them.
        let input = args("clan analyze freq -bw file.cha");
        let result = rewrite_clan_args(&input);
        assert_eq!(result, args("clan analyze freq -bw file.cha"));
    }

    #[test]
    fn recurse_flag_dropped() {
        // CLAN `+re` requests subdirectory recursion. chatter
        // recurses by default for directory args, so the flag is a
        // global no-op. Without this drop, `+re` survives the
        // rewriter and lands in the path-arg list, triggering a
        // confusing `Warning: "+re" is not a file or directory`.
        let input = args("clan analyze freq +re corpus/");
        let result = rewrite_clan_args(&input);
        assert_eq!(result, args("clan analyze freq corpus/"));
    }

    #[test]
    fn cooccur_sort_flag_dropped() {
        // CLAN COOCCUR `+o` enables a frequency-descending sort over
        // the cluster table. The semantic is encoded in
        // `OSX-CLAN/src/clan/cooccur.cpp`: `case 'o': isSort = TRUE;`
        // at line 337 toggles a BST whose invariant ("larger num_occ
        // goes left") makes in-order traversal emit clusters by
        // descending count.
        //
        // chatter's COOCCUR finalize step at
        // `crates/talkbank-clan/src/commands/cooccur.rs:292` already
        // sorts unconditionally by `count` descending (then
        // alphabetically as tiebreak), so `+o` is a no-op on the
        // chatter side. Drop the token rather than passing it to
        // clap, which would land it in the path-arg list and emit
        // `Warning: "+o" is not a file or directory`.
        let input = args("clan analyze cooccur +o file.cha");
        let result = rewrite_clan_args(&input);
        assert_eq!(result, args("clan analyze cooccur file.cha"));
    }

    #[test]
    fn freq_o_dropped() {
        // CLAN FREQ `+o` (bare) requests descending-frequency sort,
        // which is chatter's default in the FREQ finalize step.
        // No-op rewrite — same shape as `cooccur_sort_flag_dropped`.
        let input = args("clan analyze freq +o file.cha");
        let result = rewrite_clan_args(&input);
        assert_eq!(result, args("clan analyze freq file.cha"));
    }

    #[test]
    fn freq_o0_dropped() {
        // CLAN FREQ `+o0` is the explicit form of `+o` (same
        // descending-frequency-sort semantic). No-op rewrite.
        let input = args("clan analyze freq +o0 file.cha");
        let result = rewrite_clan_args(&input);
        assert_eq!(result, args("clan analyze freq file.cha"));
    }

    #[test]
    fn freq_o1_still_routes_to_reverse_concordance() {
        // Regression guard: the new `+o`/`+o0` no-op arms must
        // not shadow the existing `+o1 → --reverse-concordance`
        // arm. Match-arm ordering matters here.
        let input = args("clan analyze freq +o1 file.cha");
        let result = rewrite_clan_args(&input);
        assert_eq!(
            result,
            args("clan analyze freq --reverse-concordance file.cha")
        );
    }

    #[test]
    fn combtier_bare_tier_routes_to_tier_not_speaker() {
        // CLAN COMBTIER `+tS` selects the tier label to combine
        // (e.g. `+tcom` for `%com`) per `OSX-CLAN/src/clan/combtier.cpp`
        // usage: "+tS: Combine all tiers S into one tier." This
        // overrides the analysis-command convention where `+tCHI`
        // means "speaker filter", so the per-Combtier intercept
        // routes the bareword form to `--tier` instead of letting
        // `rewrite_tier_speaker`'s fallback emit `--speaker`.
        let input = args("clan analyze combtier +tcom file.cha");
        let result = rewrite_clan_args(&input);
        assert_eq!(result, args("clan analyze combtier --tier com file.cha"));
    }

    #[test]
    fn combtier_percent_tier_form_still_works() {
        // Regression guard: the existing `+t%X → --tier X` rewrite
        // (via the `%` branch in `rewrite_tier_speaker`) must
        // continue to fire for COMBTIER too, so `combtier +t%com`
        // produces the same `--tier com` as the bareword form.
        // The combtier-specific intercept added for the bareword
        // case must not shadow the `%`-prefix path.
        let input = args("clan analyze combtier +t%com file.cha");
        let result = rewrite_clan_args(&input);
        assert_eq!(result, args("clan analyze combtier --tier com file.cha"));
    }

    #[test]
    fn lowcase_d2_dropped() {
        // CLAN LOWCASE `+d2` = "ignore dict, lowercase everything"
        // per `OSX-CLAN/src/clan/lowcase.cpp` case 'd' (integer 0..=2
        // toggles dict-preserving / dict-capitalizing / ignore-dict).
        // chatter's `transforms/lowcase.rs` lowercases unconditionally,
        // matching the `+d2` semantic — no-op rewrite.
        let input = args("clan analyze lowcase +d2 file.cha");
        let result = rewrite_clan_args(&input);
        assert_eq!(result, args("clan analyze lowcase file.cha"));
    }

    /// CHAINS `+d`/`+d0`/`+d1` are `onlydata` output-detail levels
    /// per `OSX-CLAN/src/clan/chains.cpp:1089` — real CLAN behavior
    /// chatter does not implement. The per-CHAINS rewriter arm
    /// passes the token through unchanged so clap reports a clean
    /// "unexpected argument '+d1'" error instead of the misleading
    /// "--display-mode" rewrite from the catch-all.
    #[test]
    fn chains_dn_passes_through() {
        let input = args("clan chains +d1 file.cha");
        let result = rewrite_clan_args(&input);
        assert_eq!(result, input);
    }

    /// Bare `+d` on CHAINS also passes through.
    #[test]
    fn chains_d_bare_passes_through() {
        let input = args("clan chains +d file.cha");
        let result = rewrite_clan_args(&input);
        assert_eq!(result, input);
    }

    /// MODREP `+d` is a no-arg Excel toggle per
    /// `OSX-CLAN/src/clan/modrep.cpp:1492`. Per-MODREP arm passes
    /// it through; no `--format csv` for MODREP in chatter.
    #[test]
    fn modrep_d_passes_through() {
        let input = args("clan modrep +d file.cha");
        let result = rewrite_clan_args(&input);
        assert_eq!(result, input);
    }

    /// IPSYN `+d`/`+dN` are `onlydata` levels per
    /// `OSX-CLAN/src/clan/ipsyn.cpp:3945`. Per-IPSYN arm passes
    /// them through; no `--only-data` flag in chatter.
    #[test]
    fn ipsyn_dn_passes_through() {
        let input = args("clan ipsyn +d1 file.cha");
        let result = rewrite_clan_args(&input);
        assert_eq!(result, input);
    }

    #[test]
    fn include_retracings() {
        let input = args("clan analyze mlu +r6 file.cha");
        let result = rewrite_clan_args(&input);
        assert_eq!(
            result,
            args("clan analyze mlu --include-retracings file.cha")
        );
    }

    #[test]
    fn merge_noop() {
        let input = args("clan analyze freq +u file.cha");
        let result = rewrite_clan_args(&input);
        // +u is a no-op (merge is default), so it's dropped
        assert_eq!(result, args("clan analyze freq file.cha"));
    }

    /// Generic `+dN` display-mode fallback for values that have no
    /// command-specific rewriter arm. `+d6` is still Rewriter-only
    /// in the FREQ audit; this test pins that the fallback path
    /// still works. (`+d1`/`+d2`/`+d3`/`+d4` each have their own
    /// FREQ-specific arms that route to typed flags.)
    #[test]
    fn display_mode_fallback() {
        let input = args("clan analyze freq +d6 file.cha");
        let result = rewrite_clan_args(&input);
        assert_eq!(result, args("clan analyze freq --display-mode 6 file.cha"));
    }

    #[test]
    fn case_sensitive() {
        let input = args("clan analyze freq +k file.cha");
        let result = rewrite_clan_args(&input);
        assert_eq!(result, args("clan analyze freq --case-sensitive file.cha"));
    }

    /// FREQ's `+c` (and `+c0` alias) is the "count only capitalised
    /// words" filter. CLAN treats them identically; chatter routes
    /// both to `--capitalization initial`.
    #[test]
    fn freq_capitalized_only_bare() {
        let input = args("clan freq +c file.cha");
        let result = rewrite_clan_args(&input);
        assert_eq!(result, args("clan freq --capitalization initial file.cha"));
    }

    /// `+c0` is FREQ's documented alias for `+c`; same rewriter
    /// target. Pinned separately so a future regression on either
    /// spelling fails its own test.
    #[test]
    fn freq_capitalized_only_zero_suffix() {
        let input = args("clan freq +c0 file.cha");
        let result = rewrite_clan_args(&input);
        assert_eq!(result, args("clan freq --capitalization initial file.cha"));
    }

    /// FREQ's `+c1` is the mid-word-uppercase variant: only count
    /// words with an uppercase letter AFTER position 0
    /// (e.g. `McDonald`, `iPhone`).
    #[test]
    fn freq_capitalized_mid_uppercase() {
        let input = args("clan freq +c1 file.cha");
        let result = rewrite_clan_args(&input);
        assert_eq!(result, args("clan freq --capitalization mid file.cha"));
    }

    /// COOCCUR's `+d` (no N) strips the leading count column from
    /// the output. Distinct from the generic `+dN` display-mode
    /// rewrite — COOCCUR-specific arm intercepts before the
    /// empty-rest fall-through.
    #[test]
    fn cooccur_cluster_size() {
        let input = args("clan cooccur +n3 file.cha");
        let result = rewrite_clan_args(&input);
        assert_eq!(result, args("clan cooccur --cluster-size 3 file.cha"));
    }

    #[test]
    fn cooccur_no_frequency_counts() {
        let input = args("clan cooccur +d file.cha");
        let result = rewrite_clan_args(&input);
        assert_eq!(result, args("clan cooccur --no-frequency-counts file.cha"));
    }

    /// FREQPOS's `+d` (no N) switches position classification
    /// from first/last/other to first/second/other. Distinct from
    /// the generic `+dN` display-mode rewrite (FREQPOS-specific
    /// arm intercepts before the generic +dN routing).
    #[test]
    fn freqpos_second_mode_classification() {
        let input = args("clan freqpos +d file.cha");
        let result = rewrite_clan_args(&input);
        assert_eq!(
            result,
            args("clan freqpos --position-classification second file.cha")
        );
    }

    /// `+d` under non-FREQPOS subcommands continues to fall
    /// through to the generic display-mode handler (which itself
    /// returns None for empty rest). Pinned with a different
    /// subcommand to ensure scope-narrowing.
    #[test]
    fn freq_d_bare_does_not_match_position_classification() {
        let input = args("clan freq +d file.cha");
        let result = rewrite_clan_args(&input);
        // `+d` with empty rest under FREQ doesn't get rewritten —
        // it stays in the argv as-is (downstream clap will error
        // since there's no `+d` consumer).
        assert_eq!(result, args("clan freq +d file.cha"));
    }

    /// FREQ's `+o1` is the reverse-concordance sort: words are
    /// sorted by their reversed character sequence (so words with
    /// the same suffix cluster together). Routes to the boolean
    /// `--reverse-concordance` flag.
    #[test]
    fn freq_reverse_concordance_sort() {
        let input = args("clan freq +o1 file.cha");
        let result = rewrite_clan_args(&input);
        assert_eq!(result, args("clan freq --reverse-concordance file.cha"));
    }

    /// FREQ's `+d1` emits one word per line with no frequencies or
    /// other info — meant as input to `kwal +s@FILE`. Routes to
    /// `--word-list-only`. The bare `+d` and the broader `+dN`
    /// display-mode rewrites are separate items.
    #[test]
    fn freq_word_list_only() {
        let input = args("clan freq +d1 file.cha");
        let result = rewrite_clan_args(&input);
        assert_eq!(result, args("clan freq --word-list-only file.cha"));
    }

    /// FREQ's `+d4` outputs only the per-speaker type/token/TTR
    /// summary, dropping all per-word frequency entries. Routes to
    /// `--types-tokens-only`. Distinct from `+d3` (same content,
    /// but spreadsheet form via `+f`/CSV).
    #[test]
    fn freq_types_tokens_only() {
        let input = args("clan freq +d4 file.cha");
        let result = rewrite_clan_args(&input);
        assert_eq!(result, args("clan freq --types-tokens-only file.cha"));
    }

    /// FREQ's `+d3` = `+d4` content in spreadsheet (CSV) form. The
    /// rewriter must emit both flags together: the types/tokens
    /// suppression AND the CSV format selector. CLAN manual:
    /// "Essentially the same as that for `+d2`, but with only the
    /// statistics on types, tokens, and the type-token ratio."
    #[test]
    fn freq_types_tokens_only_csv() {
        let input = args("clan freq +d3 file.cha");
        let result = rewrite_clan_args(&input);
        assert_eq!(
            result,
            args("clan freq --types-tokens-only --format csv file.cha"),
        );
    }

    /// FREQ's `+d2` is the per-speaker per-word output in Excel
    /// (CSV) form — `render_csv` already produces this, so the
    /// rewriter only needs to flip the format selector. Distinct
    /// from `+d3` (same form but only summary rows) and `+d4`
    /// (same content as `+d3` but CLAN text format).
    #[test]
    fn freq_spreadsheet() {
        let input = args("clan freq +d2 file.cha");
        let result = rewrite_clan_args(&input);
        assert_eq!(result, args("clan freq --format csv file.cha"));
    }

    /// KWAL's bare `+d` switches output from CLAN's location-
    /// annotated default to a legal CHAT fragment (just the
    /// matching `*Speaker:` lines, no `---` separator, no `*** File
    /// ... Keyword: X` line). Routes to `--legal-chat`.
    #[test]
    fn kwal_legal_chat_format() {
        let input = args("clan kwal +d file.cha");
        let result = rewrite_clan_args(&input);
        assert_eq!(result, args("clan kwal --legal-chat file.cha"));
    }

    /// `+c` under non-FREQ subcommands keeps its existing meaning
    /// (MAXWD: `--limit N`; CHECK: `--bullets N`; IPSYN/DSS:
    /// `--max-utterances N`). Regression-pin for MAXWD so adding
    /// the FREQ arm doesn't accidentally swallow `+c50`.
    #[test]
    fn maxwd_plus_c_still_maps_to_limit() {
        let input = args("clan maxwd +c50 file.cha");
        let result = rewrite_clan_args(&input);
        assert_eq!(result, args("clan maxwd --limit 50 file.cha"));
    }

    /// VOCD's `+c` has the same semantic as FREQ's: count only words
    /// starting with an uppercase letter.
    #[test]
    fn vocd_capitalized_only_bare() {
        let input = args("clan vocd +c file.cha");
        let result = rewrite_clan_args(&input);
        assert_eq!(result, args("clan vocd --capitalization initial file.cha"));
    }

    /// VOCD's `+c0` is the documented alias for `+c`.
    #[test]
    fn vocd_capitalized_only_zero_suffix() {
        let input = args("clan vocd +c0 file.cha");
        let result = rewrite_clan_args(&input);
        assert_eq!(result, args("clan vocd --capitalization initial file.cha"));
    }

    /// VOCD's `+c1` (mid-uppercase) — sibling of FREQ `+c1`.
    #[test]
    fn vocd_capitalized_mid_uppercase() {
        let input = args("clan vocd +c1 file.cha");
        let result = rewrite_clan_args(&input);
        assert_eq!(result, args("clan vocd --capitalization mid file.cha"));
    }

    /// COMBO's `+g3` (first-match-per-utterance) routes to the
    /// boolean `--first-match-only` flag on the Combo subcommand.
    #[test]
    fn combo_g3_routes_to_first_match_only() {
        let input = args("clan combo -S want +g3 file.cha");
        let result = rewrite_clan_args(&input);
        assert_eq!(
            result,
            args("clan combo -S want --first-match-only file.cha")
        );
    }

    /// COMBO's `+g5` is a no-op for chatter — `+` is already the
    /// default AND operator. Rewriter consumes the flag silently;
    /// downstream clap never sees a stale `+g5`.
    #[test]
    fn combo_g5_is_silently_consumed_as_noop() {
        let input = args("clan combo -S want +g5 file.cha");
        let result = rewrite_clan_args(&input);
        assert_eq!(result, args("clan combo -S want file.cha"));
    }

    /// COMBO's `+g4` is "Exclude utterance delimiters from the
    /// search" — chatter's COMBO already operates on
    /// `countable_words`, which never returns terminators or
    /// separators. So `+g4` is the chatter default; the rewriter
    /// consumes the flag and clap never sees it. Same shape as
    /// the `+g5` no-op accept.
    #[test]
    fn combo_g4_is_silently_consumed_as_noop() {
        let input = args("clan combo -S want +g4 file.cha");
        let result = rewrite_clan_args(&input);
        assert_eq!(result, args("clan combo -S want file.cha"));
    }

    /// COMBO's `+g7` (no-duplicate-matches) routes to the boolean
    /// `--dedupe-matches` flag on the Combo subcommand.
    #[test]
    fn combo_g7_routes_to_dedupe_matches() {
        let input = args("clan combo -S want +g7 file.cha");
        let result = rewrite_clan_args(&input);
        assert_eq!(result, args("clan combo -S want --dedupe-matches file.cha"));
    }

    /// DIST's `+g` is a per-turn-deduplicate counting policy
    /// (CLAN: "count only one occurrence of each word per turn"),
    /// distinct from the inherited gem-segment filter. Routes to
    /// `--once-per-turn` on the Dist subcommand; gem-label filters
    /// still go through `+gLABEL`.
    #[test]
    fn dist_g_bare_routes_to_once_per_turn() {
        let input = args("clan dist +g file.cha");
        let result = rewrite_clan_args(&input);
        assert_eq!(result, args("clan dist --once-per-turn file.cha"));
    }

    /// `+gLABEL` (gem filter) on DIST is unchanged by the new arm.
    #[test]
    fn dist_g_with_label_still_routes_to_gem() {
        let input = args("clan dist +gStory file.cha");
        let result = rewrite_clan_args(&input);
        assert_eq!(result, args("clan dist --gem Story file.cha"));
    }

    /// COMBO's gem-segment filter `+gLABEL` is unaffected by the
    /// new `+g3` / `+g5` arms.
    #[test]
    fn combo_g_with_label_still_routes_to_gem() {
        let input = args("clan combo -S want +gStory file.cha");
        let result = rewrite_clan_args(&input);
        assert_eq!(result, args("clan combo -S want --gem Story file.cha"));
    }

    /// MAXWD's `+cN` selects the number of longest items to display
    /// (CLAN's `+c50` ↔ chatter's `--limit 50`). Without this branch,
    /// `+cN` falls through to the CHECK-style `--bullets N` rewrite,
    /// which `Maxwd`'s clap struct does not accept.
    #[test]
    fn maxwd_limit() {
        let input = args("clan maxwd +c50 file.cha");
        let result = rewrite_clan_args(&input);
        assert_eq!(result, args("clan maxwd --limit 50 file.cha"));
    }

    /// MAXWD's `+a` restricts output to words whose length is
    /// unique within a speaker's lexicon (CLAN: "Consider ONLY
    /// unique-length words"). Routes to `--unique-length-only`.
    #[test]
    fn maxwd_unique_length_only() {
        let input = args("clan maxwd +a file.cha");
        let result = rewrite_clan_args(&input);
        assert_eq!(result, args("clan maxwd --unique-length-only file.cha"));
    }

    /// MLU's `-t%mor` is CLAN's documented escape hatch when the
    /// `%mor` tier is present but should be ignored — implies
    /// `--words` semantics. Without this special-case, the rewriter
    /// routes `-t%X` to the generic `--exclude-tier X` which MLU's
    /// clap doesn't accept.
    #[test]
    fn mlu_exclude_mor_tier_maps_to_words() {
        let input = args("clan mlu -t%mor file.cha");
        let result = rewrite_clan_args(&input);
        assert_eq!(result, args("clan mlu --words file.cha"));
    }

    /// Same escape hatch applies to MLT (clause-level mean length,
    /// shares MLU's %mor-vs-main-tier choice).
    #[test]
    fn mlt_exclude_mor_tier_maps_to_words() {
        let input = args("clan mlt -t%mor file.cha");
        let result = rewrite_clan_args(&input);
        assert_eq!(result, args("clan mlt --words file.cha"));
    }

    /// `-t%X` for a non-%mor tier still routes to the generic
    /// `--exclude-tier` path even under MLU. The special-case is
    /// scoped to `-t%mor` specifically.
    #[test]
    fn mlu_exclude_non_mor_tier_falls_through() {
        let input = args("clan mlu -t%pho file.cha");
        let result = rewrite_clan_args(&input);
        assert_eq!(result, args("clan mlu --exclude-tier pho file.cha"));
    }

    /// KWAL's `+b` is the strict-match mode: an utterance matches
    /// the keyword only when the keyword is the *only* item on
    /// the tier. Routes to the boolean `--strict-match` flag.
    #[test]
    fn kwal_strict_match() {
        let input = args("clan kwal -s want +b file.cha");
        let result = rewrite_clan_args(&input);
        assert_eq!(result, args("clan kwal -s want --strict-match file.cha"));
    }

    /// WDSIZE's `+w>N` filters the histogram to words with length
    /// strictly greater than N. Distinct from the general `+wN`
    /// context-window rewrite because the first character of rest
    /// is a comparator (`>`, `<`, or `=`).
    #[test]
    fn wdsize_length_filter_greater_than() {
        let input = args("clan wdsize +w>4 file.cha");
        let result = rewrite_clan_args(&input);
        assert_eq!(result, args("clan wdsize --length-filter gt:4 file.cha"));
    }

    /// `+w<N` → strictly less than.
    #[test]
    fn wdsize_length_filter_less_than() {
        let input = args("clan wdsize +w<5 file.cha");
        let result = rewrite_clan_args(&input);
        assert_eq!(result, args("clan wdsize --length-filter lt:5 file.cha"));
    }

    /// `+w=N` → equal to.
    #[test]
    fn wdsize_length_filter_equal() {
        let input = args("clan wdsize +w=3 file.cha");
        let result = rewrite_clan_args(&input);
        assert_eq!(result, args("clan wdsize --length-filter eq:3 file.cha"));
    }

    /// MAXWD's `+xN` excludes words of length N. Repeatable
    /// (`+x5 +x6` excludes both). Routes to argv-pair
    /// `--exclude-length N`.
    #[test]
    fn maxwd_exclude_length_single() {
        let input = args("clan maxwd +x5 file.cha");
        let result = rewrite_clan_args(&input);
        assert_eq!(result, args("clan maxwd --exclude-length 5 file.cha"));
    }

    /// Repeated `+xN` flags produce repeated `--exclude-length N`
    /// pairs in argv order.
    #[test]
    fn maxwd_exclude_length_multiple() {
        let input = args("clan maxwd +x5 +x7 file.cha");
        let result = rewrite_clan_args(&input);
        assert_eq!(
            result,
            args("clan maxwd --exclude-length 5 --exclude-length 7 file.cha")
        );
    }

    /// CHECK retains the existing `+cN` ↔ `--bullets N` behaviour
    /// — proving the new MAXWD branch is gated on subcommand.
    #[test]
    fn check_bullets_unchanged_by_maxwd_branch() {
        let input = args("clan check +c3 file.cha");
        let result = rewrite_clan_args(&input);
        assert_eq!(result, args("clan check --bullets 3 file.cha"));
    }

    /// IPSYN's `+cN` selects the number of unique utterances to
    /// analyse (CLAN default 100; chatter's `--max-utterances 100`).
    /// Without per-subcommand routing this fell through to the
    /// CHECK-style `--bullets N`, which `Ipsyn`'s clap struct does
    /// not accept.
    #[test]
    fn ipsyn_max_utterances() {
        let input = args("clan ipsyn +c50 file.cha");
        let result = rewrite_clan_args(&input);
        assert_eq!(result, args("clan ipsyn --max-utterances 50 file.cha"));
    }

    /// DSS's `+cN` selects the number of unique utterances to score
    /// (CLAN default 50). Same routing as IPSYN.
    #[test]
    fn dss_max_utterances() {
        let input = args("clan dss +c30 file.cha");
        let result = rewrite_clan_args(&input);
        assert_eq!(result, args("clan dss --max-utterances 30 file.cha"));
    }

    /// IPSYN's `+lF` specifies the rules-file path
    /// (CLAN: language script). Maps to `--rules <PATH>`.
    #[test]
    fn ipsyn_rules() {
        let input = args("clan ipsyn +leng.ips file.cha");
        let result = rewrite_clan_args(&input);
        assert_eq!(result, args("clan ipsyn --rules eng.ips file.cha"));
    }

    /// DSS's `+lF` specifies the rules-file path. Same routing.
    #[test]
    fn dss_rules() {
        let input = args("clan dss +lengu.scr file.cha");
        let result = rewrite_clan_args(&input);
        assert_eq!(result, args("clan dss --rules engu.scr file.cha"));
    }

    /// MORTABLE's `+lF` specifies the language script file
    /// (CLAN: words-group definition with `.cut` extension).
    /// Maps to `--script <PATH>`.
    #[test]
    fn mortable_script() {
        let input = args("clan mortable +leng.cut file.cha");
        let result = rewrite_clan_args(&input);
        assert_eq!(result, args("clan mortable --script eng.cut file.cha"));
    }

    /// SCRIPT's `+sF` is the template-file argument (an exception
    /// to the general `+sS` ↔ `--include-word S` rule, since
    /// SCRIPT's `+s` value is a filesystem path).
    #[test]
    fn script_template() {
        let input = args("clan script +stemplate.cha file.cha");
        let result = rewrite_clan_args(&input);
        assert_eq!(result, args("clan script --template template.cha file.cha"));
    }

    /// UNIQ's `-o` flag is the sort-by-descending-frequency switch.
    /// Routes to `--sort`. UNIQ is the only command with a
    /// meaningful `-o`.
    #[test]
    fn uniq_sort() {
        let input = args("clan uniq -o file.cha");
        let result = rewrite_clan_args(&input);
        assert_eq!(result, args("clan uniq --sort file.cha"));
    }

    /// MLU's `+gS` is CLAN's command-specific solo-word elision
    /// flag (drop utterances consisting solely of word S). The
    /// general `+gS` ↔ `--gem S` semantic — gem-segment filter —
    /// would silently produce wrong output for researchers
    /// pasting `mlu +gum file.cha`; the MLU/MLT branch routes
    /// here instead.
    #[test]
    fn mlu_solo_word() {
        let input = args("clan mlu +gum file.cha");
        let result = rewrite_clan_args(&input);
        assert_eq!(result, args("clan mlu --exclude-solo-word um file.cha"));
    }

    /// MLT shares MLU's `+gS` semantic.
    #[test]
    fn mlt_solo_word() {
        let input = args("clan mlt +gum file.cha");
        let result = rewrite_clan_args(&input);
        assert_eq!(result, args("clan mlt --exclude-solo-word um file.cha"));
    }

    /// MLU `+g@F` loads the solo-word exclusion list from a file,
    /// same idiom as `+s@F` → `--include-word-file`. Must precede
    /// the per-word `+gS` arm so the `@`-prefix is intercepted
    /// before being treated as a literal solo-word pattern.
    #[test]
    fn mlu_solo_word_from_file() {
        let input = args("clan mlu +g@list.txt file.cha");
        let result = rewrite_clan_args(&input);
        assert_eq!(
            result,
            args("clan mlu --exclude-solo-word-file list.txt file.cha")
        );
    }

    /// MLT shares MLU's `+g@F` semantic.
    #[test]
    fn mlt_solo_word_from_file() {
        let input = args("clan mlt +g@list.txt file.cha");
        let result = rewrite_clan_args(&input);
        assert_eq!(
            result,
            args("clan mlt --exclude-solo-word-file list.txt file.cha")
        );
    }

    /// FREQ keeps the general `+gS` ↔ `--gem S` semantic.
    /// Proves the MLU/MLT branch is gated on subcommand.
    #[test]
    fn freq_gem_unchanged_by_mlu_branch() {
        let input = args("clan freq +gstory file.cha");
        let result = rewrite_clan_args(&input);
        assert_eq!(result, args("clan freq --gem story file.cha"));
    }

    /// SUGAR's `+aN` sets the minimum-utterance threshold
    /// (CLAN default 50). Routes to `--min-utterances N`.
    #[test]
    fn sugar_min_utterances() {
        let input = args("clan sugar +a30 file.cha");
        let result = rewrite_clan_args(&input);
        assert_eq!(result, args("clan sugar --min-utterances 30 file.cha"));
    }

    /// KEYMAP's `+bS` sets a key-code to track. Routes to
    /// `--keyword S` (repeatable).
    #[test]
    fn keymap_keyword() {
        let input = args("clan keymap +b$CW file.cha");
        let result = rewrite_clan_args(&input);
        assert_eq!(result, args("clan keymap --keyword $CW file.cha"));
    }

    /// KEYMAP's `+b@F` file-list form is documented as not-yet-
    /// rewritten — passes through unchanged. The leading `@`
    /// distinguishes it from the inline-value form.
    #[test]
    fn keymap_keyword_file_passes_through() {
        let input = args("clan keymap +b@codes.cut file.cha");
        let result = rewrite_clan_args(&input);
        // `+b@codes.cut` unrewritten — clap rejects at parse time
        // (better than silently misinterpreting as an inline keyword
        // literally named "@codes.cut").
        assert_eq!(result, args("clan keymap +b@codes.cut file.cha"));
    }

    /// MAKEMOD's `+a` is the all-alternatives boolean.
    #[test]
    fn makemod_all_alternatives() {
        let input = args("clan makemod +a file.cha");
        let result = rewrite_clan_args(&input);
        assert_eq!(result, args("clan makemod --all-alternatives file.cha"));
    }

    /// LINES's `+n` is the remove-line-numbers boolean.
    #[test]
    fn lines_remove() {
        let input = args("clan lines +n file.cha");
        let result = rewrite_clan_args(&input);
        assert_eq!(result, args("clan lines --remove file.cha"));
    }

    /// ORT's `+cF` is the homons-table dictionary path.
    #[test]
    fn ort_dictionary() {
        let input = args("clan ort +ceng.cut file.cha");
        let result = rewrite_clan_args(&input);
        assert_eq!(result, args("clan ort --dictionary eng.cut file.cha"));
    }

    /// COMBO's `+sS` and `-sS` are compound boolean expressions —
    /// not per-word patterns. Route to `--search` / `--exclude-search`.
    #[test]
    fn combo_search_routes_to_search_not_include_word() {
        let input = args("clan combo +swant+cookie file.cha");
        let result = rewrite_clan_args(&input);
        assert_eq!(result, args("clan combo --search want+cookie file.cha"));
    }

    #[test]
    fn combo_exclude_search() {
        let input = args("clan combo +swant -scookie file.cha");
        let result = rewrite_clan_args(&input);
        assert_eq!(
            result,
            args("clan combo --search want --exclude-search cookie file.cha")
        );
    }

    #[test]
    fn include_word_file_from_at_sigil() {
        let input = args("clan freq +s@nouns.cut file.cha");
        let result = rewrite_clan_args(&input);
        assert_eq!(
            result,
            args("clan freq --include-word-file nouns.cut file.cha")
        );
    }

    #[test]
    fn exclude_word_file_from_at_sigil() {
        let input = args("clan freq -s@stopwords.cut file.cha");
        let result = rewrite_clan_args(&input);
        assert_eq!(
            result,
            args("clan freq --exclude-word-file stopwords.cut file.cha")
        );
    }

    #[test]
    fn include_word_file_for_kwal() {
        let input = args("clan kwal +s@queries.cut file.cha");
        let result = rewrite_clan_args(&input);
        assert_eq!(
            result,
            args("clan kwal --include-word-file queries.cut file.cha")
        );
    }

    /// COMBO's `+s@FILE` loads search expressions from disk —
    /// one boolean expression per line, parsed downstream by
    /// `SearchExpr::parse`. Separate from the per-word
    /// `--include-word-file` because COMBO's `+s` value is a
    /// boolean expression, not a per-word pattern.
    #[test]
    fn combo_search_at_sigil_routes_to_search_file() {
        let input = args("clan combo +s@queries.cut file.cha");
        let result = rewrite_clan_args(&input);
        assert_eq!(
            result,
            args("clan combo --search-file queries.cut file.cha")
        );
    }

    /// COMBO's `-s@FILE` loads exclude search expressions from
    /// disk — same file format, opposite polarity.
    #[test]
    fn combo_exclude_search_at_sigil_routes_to_exclude_search_file() {
        let input = args("clan combo -s@stopwords.cut file.cha");
        let result = rewrite_clan_args(&input);
        assert_eq!(
            result,
            args("clan combo --exclude-search-file stopwords.cut file.cha")
        );
    }

    /// SCRIPT's `+s` carries a template-file path. `@`-prefixed
    /// values stay routed to `--template`, not to the generic
    /// word-list-from-file path.
    #[test]
    fn script_template_at_sigil_routes_to_template() {
        let input = args("clan script +s@list.cha file.cha");
        let result = rewrite_clan_args(&input);
        assert_eq!(result, args("clan script --template @list.cha file.cha"));
    }

    /// FIXBULLETS' `+oN` adds N ms to all bullet timings.
    #[test]
    fn fixbullets_offset_positive() {
        let input = args("clan fixbullets +o800 file.cha");
        let result = rewrite_clan_args(&input);
        assert_eq!(result, args("clan fixbullets --offset 800 file.cha"));
    }

    /// FIXBULLETS' `-oN` subtracts N ms.
    #[test]
    fn fixbullets_offset_negative() {
        let input = args("clan fixbullets -o800 file.cha");
        let result = rewrite_clan_args(&input);
        assert_eq!(result, args("clan fixbullets --offset -800 file.cha"));
    }

    /// `+oS` with a non-numeric value should NOT rewrite under
    /// FIXBULLETS (the numeric-only guard distinguishes the
    /// time-offset use from the general "extra tier code"
    /// semantic). The arg passes through unchanged.
    #[test]
    fn fixbullets_o_with_non_numeric_passes_through() {
        let input = args("clan fixbullets +omor file.cha");
        let result = rewrite_clan_args(&input);
        assert_eq!(result, args("clan fixbullets +omor file.cha"));
    }

    /// CLAN's `+t#ROLE` selects speakers by their `@ID:` role field.
    /// Routes to `--role ROLE`; the role string is passed verbatim
    /// (case-insensitive match happens at filter time).
    #[test]
    fn role_filter_include() {
        let input = args("clan freq +t#Target_Child file.cha");
        let result = rewrite_clan_args(&input);
        assert_eq!(result, args("clan freq --role Target_Child file.cha"));
    }

    /// CLAN does not support `-t#ROLE` exclude-by-role (per
    /// `mainusage()` the `#ROLE` form is include-only). The `-t#…`
    /// shape produces no rewrite — the arg passes through unchanged
    /// to clap, which rejects it with a parse error. This is the
    /// preferred failure mode: a loud parse error beats a silently-
    /// wrong include semantic.
    #[test]
    fn role_exclude_polarity_not_rewritten() {
        let input = args("clan freq -t#Target_Child file.cha");
        let result = rewrite_clan_args(&input);
        // Arg passes through verbatim — no rewrite.
        assert_eq!(result, args("clan freq -t#Target_Child file.cha"));
    }

    /// Outside SCRIPT, `+s` keeps its general meaning (include-word
    /// search keyword). Proves the SCRIPT branch is gated.
    #[test]
    fn freq_search_word_unchanged_by_script_branch() {
        let input = args("clan freq +scat file.cha");
        let result = rewrite_clan_args(&input);
        assert_eq!(result, args("clan freq --include-word cat file.cha"));
    }

    #[test]
    fn output_extension() {
        let input = args("clan analyze freq +fcex file.cha");
        let result = rewrite_clan_args(&input);
        assert_eq!(result, args("clan analyze freq --output-ext cex file.cha"));
    }

    #[test]
    fn context_after() {
        let input = args("clan analyze kwal +w3 file.cha");
        let result = rewrite_clan_args(&input);
        assert_eq!(result, args("clan analyze kwal --context-after 3 file.cha"));
    }

    #[test]
    fn context_before() {
        let input = args("clan analyze kwal -w2 file.cha");
        let result = rewrite_clan_args(&input);
        assert_eq!(
            result,
            args("clan analyze kwal --context-before 2 file.cha")
        );
    }

    #[test]
    fn id_filter() {
        let input: Vec<String> = vec![
            "clan".into(),
            "analyze".into(),
            "freq".into(),
            "+t@ID=\"eng|*|CHI|*\"".into(),
            "file.cha".into(),
        ];
        let result = rewrite_clan_args(&input);
        assert_eq!(
            result,
            vec![
                "clan".to_string(),
                "analyze".to_string(),
                "freq".to_string(),
                "--id-filter".to_string(),
                "eng|*|CHI|*".to_string(),
                "file.cha".to_string(),
            ]
        );
    }

    #[test]
    fn mixed_clan_and_modern_flags() {
        let input = args("clan analyze freq +t*CHI --format json file.cha");
        let result = rewrite_clan_args(&input);
        assert_eq!(
            result,
            args("clan analyze freq --speaker CHI --format json file.cha")
        );
    }

    #[test]
    fn combined_flags() {
        let input = args("clan analyze freq +t*CHI +swant +z1-50 +r6 file.cha");
        let result = rewrite_clan_args(&input);
        assert_eq!(
            result,
            args(
                "clan analyze freq --speaker CHI --include-word want --range 1-50 --include-retracings file.cha"
            )
        );
    }

    #[test]
    fn unknown_flag_passes_through() {
        let input = args("clan analyze freq +x123 file.cha");
        let result = rewrite_clan_args(&input);
        // Unknown +x flag is not rewritten
        assert_eq!(result, args("clan analyze freq +x123 file.cha"));
    }

    #[test]
    fn modern_flags_pass_through() {
        let input = args("clan analyze freq --speaker CHI --per-file file.cha");
        let result = rewrite_clan_args(&input);
        assert_eq!(
            result,
            args("clan analyze freq --speaker CHI --per-file file.cha")
        );
    }

    #[test]
    fn empty_args() {
        let result = rewrite_clan_args(&[]);
        assert!(result.is_empty());
    }

    #[test]
    fn bare_plus_minus_pass_through() {
        let input = args("+ -");
        let result = rewrite_clan_args(&input);
        assert_eq!(result, args("+ -"));
    }

    #[test]
    fn r_without_6_passes_through() {
        let input = args("clan analyze freq +r3 file.cha");
        let result = rewrite_clan_args(&input);
        // +r3 is not +r6, so it passes through
        assert_eq!(result, args("clan analyze freq +r3 file.cha"));
    }

    #[test]
    fn display_mode_non_numeric_passes_through() {
        let input = args("clan analyze freq +dabc file.cha");
        let result = rewrite_clan_args(&input);
        // +dabc is not a valid display mode
        assert_eq!(result, args("clan analyze freq +dabc file.cha"));
    }

    // CHECK-specific flag tests

    #[test]
    fn check_bullets() {
        let input = args("check +c0 file.cha");
        let result = rewrite_clan_args(&input);
        assert_eq!(result, args("check --bullets 0 file.cha"));
    }

    #[test]
    fn check_list_errors() {
        let input = args("check +e file.cha");
        let result = rewrite_clan_args(&input);
        assert_eq!(result, args("check --list-errors file.cha"));
    }

    #[test]
    fn check_include_error() {
        let input = args("check +e6 file.cha");
        let result = rewrite_clan_args(&input);
        assert_eq!(result, args("check --error 6 file.cha"));
    }

    #[test]
    fn check_exclude_error() {
        let input = args("check -e6 file.cha");
        let result = rewrite_clan_args(&input);
        assert_eq!(result, args("check --exclude-error 6 file.cha"));
    }

    #[test]
    fn check_g2_target_child() {
        let input = args("check +g2 file.cha");
        let result = rewrite_clan_args(&input);
        assert_eq!(result, args("check --check-target file.cha"));
    }

    #[test]
    fn check_g5_unused_speakers() {
        let input = args("check +g5 file.cha");
        let result = rewrite_clan_args(&input);
        assert_eq!(result, args("check --check-unused file.cha"));
    }

    #[test]
    fn check_g4_check_id() {
        let input = args("check +g4 file.cha");
        let result = rewrite_clan_args(&input);
        assert_eq!(result, args("check --check-id true file.cha"));
    }

    #[test]
    fn check_g1_noop() {
        let input = args("check +g1 file.cha");
        let result = rewrite_clan_args(&input);
        // +g1 is a no-op (prosodic delimiters always recognized)
        assert_eq!(result, args("check file.cha"));
    }

    #[test]
    fn check_u_maps_to_check_ud() {
        let input = args("check +u file.cha");
        let result = rewrite_clan_args(&input);
        assert_eq!(result, args("check --check-ud file.cha"));
    }

    #[test]
    fn non_check_u_is_noop() {
        let input = args("freq +u file.cha");
        let result = rewrite_clan_args(&input);
        // +u is a no-op (merge is default) for non-CHECK commands
        assert_eq!(result, args("freq file.cha"));
    }

    #[test]
    fn non_check_g_is_gem() {
        // For non-check commands, +g is always gem filtering
        let input = args("freq +g2 file.cha");
        let result = rewrite_clan_args(&input);
        assert_eq!(result, args("freq --gem 2 file.cha"));
    }

    #[test]
    fn check_g_with_label_falls_back_to_gem() {
        // +g with a non-digit label (even in check context) falls back to gem
        let input = args("check +gstory file.cha");
        let result = rewrite_clan_args(&input);
        assert_eq!(result, args("check --gem story file.cha"));
    }
}
