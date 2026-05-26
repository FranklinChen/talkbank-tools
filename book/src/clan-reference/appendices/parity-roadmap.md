# CLAN Parity Roadmap

**Status:** Current
**Last updated:** 2026-05-26 08:21 EDT

Planning doc for the remaining CLAN flag-parity work. Source of truth
for "how much is left" so future sessions don't have to re-derive it
from the per-command audit tables.

## Headline numbers

Counted from the Audit-summary tables in every
`book/src/clan-reference/commands/*.md` (80 command pages). See
[`status-matrix.md`](./status-matrix.md) for the per-command-binary
coverage table, which is a different cut (binary modules exist; flags
on those modules are what this page tracks).

| Bucket | Count | % | Meaning |
|---|---|---|---|
| **Done** | 220 | 33% | Implemented and verified |
| **Partial** | 51 | 8% | Common case works; edge cases incomplete |
| **Rewriter only** | 153 | 23% | CLAN `+flag` rewrites to a chatter `--flag` that `clap` rejects — no consuming field yet |
| **Missing** | 249 | 37% | No rewriter, no clap field |
| **Total flag rows** | **673** | | |

Raw counts mislead — flag rows aren't equal-weight. The breakdown by
tier below is what should drive sequencing.

## Why parity matters (strategic framing)

This is a **long-haul succession project**, not a sprint. The
original CLAN is in production today and works for its users; nobody
is blocked on chatter completing parity. The reasons to keep
chipping away:

1. **TalkBank longevity.** The legacy C/C++ CLAN codebase is decades
   old and accreted; a modern Rust reimplementation gives new
   developers a stack they can actually learn and contribute to in
   the post-current-team era.
2. **AST-based unification.** chatter's parser/model/validation are
   already AST-based for CHAT; getting all CLAN command logic on
   the same AST eliminates the "two semantic models" tax (legacy
   CLAN's string-tokenization model vs. chatter's typed model).
3. **Reproducibility from source.** The whole stack — grammar,
   parser, commands, tests, CI — built and tested from `cargo`
   against a checked-in toolchain. No "CLAN binary on the
   maintainer's laptop" provenance for output that researchers
   cite in papers.

Cadence: chip away over weeks and months, prioritize correctness
over speed, keep the audit tables and this roadmap current as the
single canonical view of remaining work. No external deadline.

## Historical context

CLAN is the work of decades (Leonid Spektor, with Brian MacWhinney,
since roughly the early 1980s). The flag long-tail reflects 40+ years
of feature accretion across commands as researchers asked for more
ways to slice the same corpora. Many flags are display-mode variants
that produce alternate output shapes for the same analysis; some are
inherited from the `cutt.cpp::mainusage` general flag set and act as
no-ops on commands that don't semantically use them. A handful are
genuinely domain-heavy linguistic features (FLUCALC syllable mode,
IPSYN rules, CHIP partner-child tagging) that need PI input on
semantics before they can be implemented.

## Tier breakdown

### Tier 1 — Blocked on upstream phases (~30 rows)

The single biggest unlock. Most of these are `+f`/`+fEXT` (file
output) which depend on the sidecar-file pattern.

| Phase | Description | Status | Unlocks |
|---|---|---|---|
| **Phase 0** | OSX-CLAN makefile fixes (wdsize/complexity/corelex) | Pending — upstream CLAN, not chatter work | Snapshot generation for those commands |
| **Phase 1.1** | Sidecar-file output pattern (`pipeout.<cmd>.{cex,xls}`) | Pending | ~20 `+f`/`+fEXT` rows across most commands |
| **Phase 1.2** | 4th banner shape for codes/complexity | Pending | Codes/complexity byte-parity |
| **Phase 1.8** | Side-effect files (codes, chains) | Pending | Codes, chains output files |

**Effort:** Phase 1.1 is ~2-3 days for the pattern + ~1 day per
consuming command. Phase 1.2 and 1.8 are smaller (~1 day each).

### Tier 2 — Inherited no-ops (~40-60 rows of the 153 "Rewriter only")

Shared flags from `cutt.cpp::mainusage` (`+k`, `+wN`/`-wN`, `+t%X`,
`+f`/`+fEXT`) appear on every command that links the general flag
table — but they're semantic no-ops for commands that don't use the
relevant subsystem:

- `+k` (case-sensitive) — meaningful only for commands that key
  words (FREQ/KWAL/VOCD/COMBO/FREQPOS/DIST/MAXWD, all **Done**).
  No-op for MLU/MLT/WDLEN/WDSIZE/CHAINS/CODES which don't word-match.
- `+wN`/`-wN` (context window) — meaningful for match-and-emit
  commands (KWAL/COMBO, **Done**). No-op for aggregate commands.
- `+t%X` (dependent-tier filter) — semantic varies per command;
  KWAL/COMBO would use it as alternate search target (a real
  feature — see Tier 4), most others just filter.
- `+f`/`+fEXT` — see Tier 1.

CLAN itself silently accepts these as no-ops. Current chatter
behavior is "clap errors," so a user pasting `mlu +k file.cha` gets
rejected. Fix is either accept-and-ignore at the clap layer (~1 day
across commands) or document as deliberate-strict. Not functionality.

### Tier 3 — Display modes requiring CLAN binary snapshots (~50-80 rows)

Per-command `+dN` variants: each `N` (often 0..8, 20, 30, 40, 90, 99)
selects a different output shape for the same analysis. Many
audit-page draft sections list these with manual quotes, marked
"pending PI review" because the exact byte format isn't fully
specified — only a CLAN binary run produces the authoritative
output.

Concrete instances (all currently Rewriter-only):

- **KWAL**: `+d1`..`+d4`, `+d7`, `+d30`/`+d31`, `+d40`, `+d90`/`+d99`
- **COMBO**: `+d1`..`+d5`, `+d7`
- **FREQ**: `+d0`, `+d5`/`+d6`/`+d7`/`+d8`, `+d20`, `+dCN`/`+d<N`/`+d>=N`/`+d=N`/`+d>N`
- **VOCD**: `+d`/`+d1`/`+d2`/`+d3`
- **DIST**: `+d`
- **DSS**: `+d`/`+d1`
- **CHAINS**: `+d`/`+d1`
- **GEM/GEMFREQ/COREELEX/FLUCALC**: `+dN` variants

Each requires:
1. Run real CLAN with the flag → capture byte-exact output snapshot
2. Add the snapshot to `tests/fixtures/clan-snapshots/`
3. Implement against the snapshot with TDD

**Effort:** ~1 hour per snapshot, ~2-4 hours per mode implementation.
Bounded by snapshot generation speed. ~150 person-hours if all done.

### Tier 4 — Genuine implementable features in heavily-used commands (~50-80 rows)

Real engineering work, command-specific:

- **`+t%X` tier-search switching** (KWAL/COMBO): currently main-tier
  search only. Switching the search target to a dependent tier
  (`%mor`, `%gra`, `%pho`) is a multi-day change per command because
  the word-collection layer needs an alternate path.
- **`+nS`/`-nS` speaker context windows** (KWAL): like `+wN` but
  bounded by speaker rather than utterance count.
- **`+bN` MATTR sliding-window** (FREQ): Moving-Average TTR.
- **`+c2`/`+c3`/`+c4`/`+c7` multi-word search variants** (FREQ):
  multi-word groups with various boundary semantics.
- **`+gnS`/`-gnS`/`+gdS`/`-gdS` LRD computation** (VOCD): Limiting
  Relative Diversity numerator/denominator filters.
- **`+o2` reverse concordance + non-CHAT** (FREQ): separate format.
- **`+pS` word-delimiter customization** (most commands): tokenizer
  config.
- **`+x` exclude-utterances-by-content** (FREQ).

**Effort:** Bulk of the real engineering left. ~3-6 weeks for the top
10 commands if pursued sequentially.

### Tier 5 — Domain-heavy specialty commands (~80-120 rows)

These commands need PI input on linguistic semantics; some need ML
models. Multi-week per command if implemented end-to-end.

| Command | Open work | Estimate |
|---|---|---|
| FLUCALC | Syllable-vs-word mode, side-effect files, pause computation, repetition counting | 1-2 weeks |
| IPSYN | Rule customization, custom-pattern files | 1-2 weeks |
| DSS | Rule customization | 1-2 weeks |
| CHIP | Adult/child speaker-role tagging, substitution coding | 1-2 weeks |
| COMPLEXITY | Complexity-specific metrics | ~1 week |
| KIDEVAL | Database modifications, output extensions | 1-2 weeks |
| COREELEX | Threshold/comparison modes | ~1 week |
| MORTABLE | Customization | ~1 week |

These could be deferred to a "v2 / post-practical-parity" milestone
if not on the critical path.

### Tier 6 — Lower-priority format converters (~30-50 rows)

Default mode works; flag variants are mostly cosmetic.

Converters: chat2elan, chat2srt, chat2vtt, chat2text, praat2chat,
chstring, lowcase, fixbullets, dataclean, dates, indent, postmortem,
trim, retrace, tierorder, delim, quotes, repeat, combtier, compound,
flo.

**Effort:** ~1-2 weeks if all done.

## Milestones

These are rolling targets, not deadlines. Cadence is "chip away" —
batch a few items per session, keep the audit tables current.

### "Practical parity" — common-use flags across the top 10 commands

Target commands: FREQ, KWAL, COMBO, VOCD, DIST, COOCCUR, MAXWD,
FREQPOS, MLU, MLT.

Scope: Tier 1 (sidecar pattern), Tier 3 (high-traffic `+dN` modes),
Tier 4 (real features), Tier 2 cleanup pass.

**Effort, if pursued sequentially:** roughly the equivalent of 3-6
focused work-weeks. At a chip-away cadence (a few items per
session, intermixed with other priorities), spread over **a few
months** of calendar time.

### "Full byte-level parity" — every flag in every command

Adds Tier 5 (domain-heavy commands) and Tier 6 (format converter
flags).

**Effort, if pursued sequentially:** equivalent of 3-6 work-months.
At chip-away cadence, spread over **a year or more** of calendar
time. Some Tier 5 items may never be worth it for chatter's
mandate; that's fine — we can decide per command as we approach
it.

## Current blockers

1. **CLAN binary access for snapshot generation.** Tier 3 work is
   gated on running real CLAN with specific flags and capturing
   byte-exact output. The discrepancy-adjudication rule says "stop
   and ask" rather than guess — so snapshot-less display modes
   shouldn't be implemented by inference.

2. **Phase 1.1 sidecar pattern not designed.** Until the pattern is
   landed, ~20 `+f`/`+fEXT` rewriter-only entries can't be unblocked.

3. **PI clarification needed on several open questions** flagged
   in per-command audit pages (KWAL `+d1` format, DIST `+d` ↔
   `--format csv` mapping, COMBO `+d3` `@Comment` headers).

## Recently completed (last batch)

Tracking what just landed so future sessions have context. Each item
is a single commit on `main` in `talkbank-tools`.

**Conversation batch May 22-23, 2026** (~22 commits, ~30 new tests):

- COOCCUR `+d` strip frequency counts
- FREQ `+d1`/`+d2`/`+d3`/`+d4` (word list / CSV / types-tokens-only)
- `+k` case-sensitive rollout across **all 7 search/frequency
  commands**: FREQ, KWAL, VOCD, COMBO, FREQPOS, DIST, MAXWD (now
  fully landed; cross-cutting migration docs updated)
- KWAL `+d` legal-CHAT-fragment output
- KWAL `+wN`/`-wN` context window (stateful `VecDeque` + awaiting-
  after machinery)
- COMBO `+wN`/`-wN` context window
- MLU/MLT `+g@F` solo-word exclusions from file
- COOCCUR `+nN` cluster size (generalized bigrams → N-grams via
  `SmallVec`)
- Two `/simplify` passes: `NormalizedWord::from_word_cased` helper
  unification, KWAL hot-path precomputed-keyword fix,
  `AnalysisRequest::kwal` config-instead-of-bools, KWAL/COMBO
  zero-context fast path (skip `to_chat_string` per utterance
  when no window is active), COOCCUR `SmallVec<[N; 2]>` for
  bigram inline storage

Tests: 966 → 1006 (+40 over the batch).

**Audit-vs-runtime sweep (2026-05-23 evening, 3 commits):**

- **MLU/MLT/WDLEN `+k` audit-page flip.** The prior `+k` rollout
  reached these non-word-keying commands via the
  `CommonAnalysisArgs.case_sensitive` flatten, but their audit pages
  still said `Rewriter only`. Verified by direct probe and flipped to
  `Done (no-op per CLAN)`. Pure documentation; no code change.
- **`+re` global no-op rewriter arm.** CLAN's `+re` requests
  subdirectory recursion; chatter already recurses by default. The
  rewriter had no arm for the token, so `+re` survived to clap's
  path-arg list and emitted a confusing
  `Warning: "+re" is not a file or directory` on every invocation.
  Added a one-line `(b'+', b'r') if rest == "e" => Some(vec![])`
  next to the existing `+u` no-op. ~10 commands' "Done" rows are now
  truthful in addition to operationally correct.
- **MLU/MLT `-bw` → `--words` rewriter arm.** Audit-vs-code drift:
  audit pages had marked `-bw` Done forever, but the rewriter only
  had a comment — no actual arm. clap parsed `-bw` as a `-b -w`
  short-flag pair and errored on the unknown `-b`. Added the missing
  `(b'-', b'b') if matches!(subcommand, Mlu | Mlt) && rest == "w"`
  arm. Three regression tests guard the rollout and the scope
  boundary.

Three new rewriter tests (`mlu_minus_bw_to_words`,
`mlt_minus_bw_to_words`, `freq_minus_bw_unchanged`,
`recurse_flag_dropped`) shipped with the second and third commits.

**Tier-2 follow-on (2026-05-26, 2 commits):**

- **`+wN` / `-wN` inherited context window on the six aggregate
  commands** (MLU, MLT, WDLEN, MAXWD, FREQPOS, FREQ). CLAN exposes
  the post-/pre-context flags on every analysis command via shared
  common-args; on aggregate commands (means, totals, histograms)
  they are runtime no-ops, but CLAN's parser still accepts them.
  Chatter's rewriter at `clan_args.rs:411-412` already converts
  `+w3` / `-w2` to `--context-after 3` / `--context-before 2`
  globally; only the clap-side consumer was missing on the six
  aggregate commands, so the rewritten flag arrived as
  `error: unexpected argument '--context-after' found`. Landed via
  a shared `InheritedContextArgs` sub-args group flattened into
  each of the six command variants. KWAL/COMBO are deliberately
  excluded — they have real consumers driving per-match context
  emission. 6 audit-table rows flipped from `Rewriter only` to
  `Done (no-op per CLAN)`.

One feat commit (`feat(clan): InheritedContextArgs accept-and-ignore
on 6 aggregate commands`) plus a docs commit (audit-page flips +
this roadmap entry). 7 new clap-acceptance tests
(`mlu_accepts_inherited_context_after` + 6 parameterized
`<cmd>_inherits_context_both_directions`); 12 end-to-end smokes
against `corpus/reference/core/basic-conversation.cha`.

## Next-up candidates

Pick one when you have a session. Not a sequence — each is
self-contained and any can come first. Update this list as items
land so future sessions don't re-derive priorities.

1. **Tier 1: Land Phase 1.1 sidecar pattern.** Single biggest
   unlock; cascades across most commands. Design a `SidecarWriter`
   trait + a per-command `pipeout.<cmd>.{cex,xls}` consumer. Multi-
   day investment but unblocks ~20 entries at once.

2. **Tier 2: Accept-and-ignore inherited no-ops at clap layer.**
   ~1 day. Flips ~40-60 "Rewriter only" rows to "Done (no-op
   per CLAN's no-op semantic)" with matching audit-doc updates.
   Low risk, mechanical.

3. **Tier 3: Generate CLAN snapshots for KWAL `+d1` and `+d2`.**
   The two most-cited "pending PI review" modes; a single CLAN
   run capturing snapshot output unblocks implementation.
   Needs CLAN binary access.

4. **Tier 4: Pick one heavily-used command's real feature.**
   Candidates: FREQ `+c2`/`+c3` (multi-word search), KWAL `+nS`
   (speaker-bounded context), VOCD `+gnS`/`+gdS` (LRD), FREQ
   `+bN` (MATTR sliding window). Each is 1-3 commits.

5. **Tier 6: One format-converter flag pass.** Lowest-stakes;
   good for a "warm up" session. chstring / chat2text / etc.

## Methodology notes

- **Counts re-derived per session** from
  `awk -F'|' '/^\| Done.*\| [0-9]+ \|$/ { ...}' book/src/clan-reference/commands/*.md`
  to keep this doc in sync. Update the headline numbers when the
  per-command audit tables change.

- **Tier categorization is judgement-based**, not encoded in the
  audit-summary tables. If a row's tier is unclear, look at the
  command's `+`-flag table notes column — entries like "same as
  KWAL" or "EVAL-style overload" hint at Tier 4/5; entries like
  "rewriter target" hint at Tier 2 (if the flag is a shared
  no-op for that command) or Tier 1 (if it's `+f`/`+fEXT`).

- **The audit pages themselves are the source of truth.** This doc
  is a navigational summary; per-flag implementation details live
  on each command's page.
