# Transcriber `$POS` Hints

**Status:** Reference — default on; opt out via `--no-pos-hints`
**Last updated:** 2026-04-22 20:50 EDT

CHAT main-tier words may carry a `$POS` suffix that encodes the
transcriber's part-of-speech annotation in CLAN-MOR conventions
(e.g. `school@s:eng$n`, `जब@s:hin$adv:temp`, `कि@s:hin$comp`). By
default the morphotag pipeline treats those hints as authoritative
POS evidence: after Stanza produces `%mor`, each hinted word's POS
category is compared against the transcriber's CLAN tag, and the
`%mor` POS is overridden on disagreement. Lemma and morphological
features from Stanza are preserved — only the POS category changes.

Pass `--no-pos-hints` on any `morphotag` invocation to suppress the
post-pass and keep Stanza's POS decisions as-is.

## Why the feature exists

The 2026-04-15 aggregate L2 eval (`l2-eval-runs/2026-04-15/`)
identified `FeaturePosMismatch` as the dominant structural error
class — 328 cases (1.9% of `@s` words) where a finite verb in the
embedded language was tagged as `NOUN`, `PROPN`, or `CCONJ` because
the primary model's deprel constrained the merge away from `VERB`.
Separately, the 2026-04-21 Hindi POC (`hindi-experiment/`) observed
the same pattern on matrix-language Hindi function words (`हाँ`
tagged `pron` rather than `intj`; `ना` tagged `pron` rather than
`part`).

In both regimes, when the transcriber has bothered to annotate a
word's POS with `$n`, `$v`, `$adv`, etc., they are encoding
linguistic knowledge Stanza lacks — either because the word is
low-resource, domain-mismatched, or embedded in a construction the
UD parser's deprel constraints don't cover. Honoring those hints is
a cheap, near-zero-risk correction when hints disagree with Stanza.

## Where it sits in the pipeline

The hint post-pass runs after the L2 secondary dispatch and splice,
before post-validation and serialization. This placement is
deliberate: the hint pass works on the final `%mor` state regardless
of whether the value came from the primary Stanza run, the L2
secondary dispatch, the phrasal-verb Priority 0 merge, or a fallback
to `L2|xxx`.

```mermaid
flowchart TD
    A["Parse CHAT\n(parse_lenient)"] --> B["Extract primary payloads\n(collect_payloads)"]
    B --> C["Stanza primary inference\n(infer_batch, per-language)"]
    C --> D["Inject %mor + %gra\n(inject_morphosyntax)"]
    D --> E{"L2 @s words present?"}
    E -->|yes| F["Dispatch secondary language Stanza\n(dispatch_secondary_l2)"]
    E -->|no| G["Skip L2 dispatch"]
    F --> H["Merge primary+secondary UD\n(resolve_merged_pos_with_context)"]
    H --> I["Splice merged Mor into ChatFile\n(splice_l2_into_chat)"]
    G --> J{"--no-pos-hints set?"}
    I --> J
    J -->|no (default)| K["apply_pos_hints(&mut ChatFile)\n(pos_hints::apply_pos_hints)"]
    J -->|yes| L["Skip hint post-pass"]
    K --> M["Validate alignment\n(validate_mor_alignment)"]
    L --> M
    M --> N["Serialize CHAT\n(to_chat_string)"]
```

Source verified: `crates/batchalign-app/src/morphosyntax/batch.rs`
(`run_morphosyntax_batch_impl`), `crates/batchalign-chat-ops/src/morphosyntax/pos_hints.rs`
(`apply_pos_hints`), and `crates/batchalign-chat-ops/src/morphosyntax/l2/splice.rs`
(`splice_l2_into_chat`).

## Per-hint decision flow

For every main-tier word in every utterance, the pass asks four
questions in order: is there a hint? does the CLAN tag map to a UD
UPOS? is there a `%mor` item to modify? does the UPOS disagree with
Stanza? The flow runs exactly once per word and is
pure — no Stanza re-invocation, no network I/O.

```mermaid
flowchart TD
    Start(["For each main-tier word\n(walk_words, TierDomain::Mor)"]) --> H{"Word has $POS?\n(word.part_of_speech)"}
    H -->|no| Skip["No record; continue"]
    H -->|yes| Lookup{"clan_to_ud_upos(tag)\n(talkbank_model)"}
    Lookup -->|None| UnmappedCLAN["record: UnmappedCLAN\nleave %mor untouched"]
    Lookup -->|"Some(upos_name)"| Enum{"upos_name_to_enum(name)\n→ UniversalPos"}
    Enum -->|None| UnmappedUPOS["record: UnmappedCLAN\n(future-safety: new UPOS in table)"]
    Enum -->|Some| MorCheck{"mor.items.get_mut(word_idx)\nexists?"}
    MorCheck -->|no| NoMor["record: NoMorItem\n(utterance skipped / count mismatch)"]
    MorCheck -->|yes| Compare{"stanza_pos == hinted_upos?\n(lowercased CHAT POS names)"}
    Compare -->|equal| Agreement["record: Agreement\n(no change — Stanza got it right)"]
    Compare -->|differ| Override["mor.main.pos = hinted\n(features, lemma preserved)\nrecord: Overridden"]
```

Source verified: `crates/batchalign-chat-ops/src/morphosyntax/pos_hints.rs`
(`apply_pos_hints`, `upos_name_to_enum`, `upos_to_chat_pos`).

## The CLAN → UD UPOS table

The mapping lives in `talkbank-model` so it is a cross-cutting
artifact useable outside this feature (parity audits, CLAN-vs-UD
reconciliation, future tools):

```mermaid
classDiagram
    class CLAN_tag {
        +&str clan_tag
        +split(":") coarse, refinement
    }
    class UD_UPOS {
        +&str upos_name
    }
    class clan_to_ud_upos {
        +fn(clan_tag: &str) Option~&'static str~
        "Special case: n:prop → PROPN"
        "Coarse table on head before colon"
        "Unknown → None"
    }
    CLAN_tag --> clan_to_ud_upos : input
    clan_to_ud_upos --> UD_UPOS : output (or None)
```

Source verified: `talkbank-tools/crates/talkbank-model/src/model/dependent_tier/mor/analysis/clan_ud_mapping.rs`.

Coverage (see the `#[test]` suite in `clan_ud_mapping.rs` for the
exhaustive list):

| CLAN tag family | UD UPOS | Notes |
|---|---|---|
| `n` | NOUN | |
| `n:prop` | PROPN | refinement crosses UPOS boundary |
| `n:gerund`, `n:deverbal`, … | NOUN | other `n:*` refinements stay NOUN |
| `v` | VERB | |
| `adj`, `adj:att`, … | ADJ | |
| `adv`, `adv:temp`, … | ADV | |
| `pro`, `pro:per`, `pro:dem`, `pro:sub`, `pro:int`, `pro:rel` | PRON | subtype isn't tracked in UPOS |
| `det`, `det:dem`, `det:poss`, `det:art` | DET | |
| `prep`, `post` | ADP | postpositions for Hindi/Tamil/etc. |
| `conj` | CCONJ | default coordinating |
| `comp` | SCONJ | complementizer (e.g. `कि`, "that") |
| `part` | PART | |
| `mod`, `aux` | AUX | |
| `qn` | DET | UD has no separate quantifier UPOS |
| `num` | NUM | |
| `co`, `int`, `intj` | INTJ | |
| `sym` | SYM | |
| `punct`, `cm`, `end`, `beg` | PUNCT | |
| anything else | None | unmapped — hint ignored |

## CLI usage

```bash
# Default behavior — hints respected automatically.
batchalign3 morphotag input.cha --output out/ --lang hin

# Opt out for a single job:
batchalign3 morphotag --no-pos-hints input.cha --output out/ --lang hin

# `--no-pos-hints` is orthogonal to --retokenize, --skipmultilang,
# --no-l2-morphotag, etc.
batchalign3 morphotag \
    --no-pos-hints \
    --no-l2-morphotag \
    --lang eng \
    input/
```

With hints on (the default), every `$POS`-carrying word in every
utterance is considered. The pass is idempotent — running twice on
the same input produces the same output, because the second run sees
every hint as an Agreement.

## What gets preserved

| Field | Preserved? |
|---|---|
| Main tier (word order, `@s` tags, `$POS` suffixes, markup) | ✓ unchanged |
| `%mor` lemma (`MorStem`) | ✓ Stanza value kept |
| `%mor` features (tense, case, number, gender, …) | ✓ Stanza value kept |
| `%mor` POS category | **overwritten on disagreement** |
| `%gra` relations | ✓ unchanged |
| Post-clitics (`~aux|be` after `pron|it`) | ✓ unchanged (outer item only gets POS override) |
| `%xmor`, `%xgra`, `%com`, `%eng` and other user tiers | ✓ unchanged |

The pass never adds, removes, or reorders words or tiers. It only
mutates the single `PosCategory` string on `%mor` items whose paired
main-tier word has a disagreeing `$POS`.

## Known limitations

1. **Only applies to utterances with a `%mor` tier.** If Stanza
   skipped an utterance due to MOR-vs-main count mismatch (MWT,
   comma-handling, etc.), no `%mor` exists, so no hint can apply.
   The hint pass records these as `NoMorItem` but takes no action.
   On the Hindi POC 36% of utterances fell into this category — a
   bigger quality issue than the hint feature addresses. See
   `docs/investigations/2026-04-21-l2-morphotag-corpus-state.md`.
2. **Unknown CLAN tags are silent.** The mapping is intentionally
   conservative: unknown tags return `None`, the record is logged
   as `UnmappedCLAN`, and Stanza's POS is kept. Widening the
   mapping is a matter of adding entries to
   `clan_ud_mapping.rs` and the corresponding unit tests.
3. **Refinements don't become UD features.** `$pro:dem` could
   plausibly propagate `PronType=Dem` to `%mor` features, but today
   only POS category is overridden. A future revision could handle
   refinements → features.
4. **Transcriber errors propagate.** If the transcriber wrote `$v`
   on a word Stanza tagged `DET` with full determiner features, the
   hint wins and produces `verb|the…Det-features`. A cross-check
   warning (feature vs POS consistency) is a candidate followup.
5. **No `%gra` deprel upgrade.** Changing a word's POS can make its
   `%gra` deprel inconsistent (e.g., `NOUN` → `VERB` on an item with
   deprel `OBJ`). Today we leave the deprel as-is; the cross-check
   is deferred.

## Rollout history

| Phase | State | Flag | Status |
|---|---|---|---|
| 1 | POC evidence | uncommitted prototype | done (2026-04-21 Hindi POC) |
| 2 | Opt-in ship | `--respect-pos-hints` default off | shipped briefly on 2026-04-22 |
| 3 | Default-on | `--no-pos-hints` opt-out | **current** (2026-04-22) |
| 4 | Flag removal | no flag | TBD — after wide corpus observation without regression reports |

The phase-2 → phase-3 flip skipped a formal ungating eval because
the hint pass is narrow (POS-only overrides, Stanza features and
lemma preserved) and idempotent. If the default-on behavior produces
regressions in practice, `--no-pos-hints` provides immediate
per-invocation relief while a fix is prepared.

## Related documentation

- [L2 Morphotag design](l2-morphotag.md) — the feature the hint pass
  augments; `$POS` hints are a merge-algorithm-adjacent signal, not
  an L2-specific one.
- [L2 Morphotag Aggregate Eval (2026-04-15)](l2-eval-runs/2026-04-15/summary.md)
  — the corpus available for follow-up evaluation; reruns now
  include the hint pass by default.
- [L2 Morphotag Status](l2-morphotag-status.md) — L2 feature overview.
- `hindi-experiment/REPORT.md` (private meta-repo) — the POC that
  motivated this feature, including per-override linguistic verdicts.
- `talkbank-tools/crates/talkbank-model/src/model/dependent_tier/mor/analysis/clan_ud_mapping.rs`
  — the mapping source of truth.
- `batchalign3/crates/batchalign-chat-ops/src/morphosyntax/pos_hints.rs`
  — the applicator source.

## Reproducing the POC evidence

The 2026-04-21 Hindi POC used a twin morphotag run (stock vs
prototype) on a 100-utterance sample of Devanagari-converted
classroom speech. Reproduce with:

```bash
# 1. Stock run (hints disabled — the old pre-default behavior)
batchalign3 morphotag --no-pos-hints sample-100-devanagari.cha \
    --output stock/ --lang hin --sequential --workers 1

# 2. Hint-respecting run (current default)
batchalign3 morphotag sample-100-devanagari.cha \
    --output proto/ --lang hin --sequential --workers 1

# 3. Compare
python3 hindi-experiment/scripts/compare_morphotag.py \
    stock/sample-100-devanagari.cha \
    proto/sample-100-devanagari.cha
```

On that sample: 5 POS overrides out of 26 hints applied; 3 of 5
unambiguously correct; 2 defensible; zero regressions.
