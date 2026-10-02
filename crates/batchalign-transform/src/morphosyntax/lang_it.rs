//! Italian-specific morphosyntax rules.
//!
//! This module exists to work around known Stanza 1.11.1 Italian
//! MWT mis-splits (Defect 6 / Defect 7) that corrupt %mor content.
//! When Stanza emits an MWT Range for an Italian input token that
//! we know it mis-analyzes, the reconciler here overrides with a
//! single hand-curated Mor carrying the correct POS, lemma, and
//! features: bypassing the `~`-joined `verb|STEM~pron|CLITIC`
//! assembly that Stanza's split would otherwise produce.
//!
//! This is explicitly a **hack**. The allowlist
//! ([`IT_MIS_SPLIT_OVERRIDES`]) is a closed table of specific
//! mis-splits observed in production Italian corpora. It does not
//! auto-detect novel cases; new ones require an explicit entry plus
//! a regression test. Periodic re-audits should retire entries
//! whose Stanza-raw output has been fixed upstream.
//!
//! See the user-facing documentation at
//! `book/src/batchalign/reference/languages/italian.md`
//! (§"Reconciler hacks").

use super::mor_word::map_ud_mor_word;
use crate::morphosyntax::UdId;
use crate::morphosyntax::alignment::{AlignedUd, WalkedUdToken};
use crate::morphosyntax::{
    ChunkHead, ChunkProvenance, CuratedFeats, MappedItem, MappingContext, MappingError,
    UdPunctable, UdWord, UdWordId, UniversalPos, assemble_mors, normalize_deprel,
    provenance_for_ud_word,
};

/// One entry in the Italian Defect-6/7 mis-split allowlist.
///
/// Each entry documents an input token text that Stanza's Italian
/// MWT processor splits into a fake verb+clitic (or fake
/// article+suffix) analysis. When the reconciler matches an entry,
/// it synthesizes a single-word `UdWord` with the overridden POS /
/// lemma / feats and runs it through the normal
/// `map_ud_word` pipeline to produce a correct `Mor`.
#[derive(Debug, Clone)]
pub(crate) struct MisSplitOverride {
    /// The input token text that Stanza mis-analyzes. Matched
    /// case-insensitively against the MWT Range parent's `text`
    /// field (i.e., the original input word before any split).
    pub joined_text: &'static str,
    /// POS to assign to the reconciled single-word Mor.
    pub pos: UniversalPos,
    /// Lemma for the reconciled Mor (the correct lexical root
    /// for the word, ignoring Stanza's nonsense component lemma).
    pub lemma: &'static str,
    /// UD feats for the reconciled Mor.
    pub feats: CuratedFeats,
}

/// Table of known Italian mis-splits (Defect 6 + Defect 7 family).
///
/// Each entry is sourced from a specific committed %mor regression
/// or a known Stanza-limitation test pin in
/// `_cases/italian.py`. When a new Stanza-mis-split case surfaces
/// in production, add one row here plus a regression test in
/// `morphosyntax/mod.rs` tests.
pub(crate) const IT_MIS_SPLIT_OVERRIDES: &[MisSplitOverride] = &[
    // Defect 6: verb: `parla → par + la` (2sg/3sg indicative, also
    // 2sg imperative: Stanza's analysis varies by context but the
    // override is the same: `parlare` is the correct lemma either
    // way). Source: childes-other-data/Frogs/Italian-Roma/06/06danbov.cha.
    MisSplitOverride {
        joined_text: "parla",
        pos: UniversalPos::Verb,
        lemma: "parlare",
        feats: CuratedFeats::new("Mood=Ind|Number=Sing|Person=3|Tense=Pres|VerbForm=Fin"),
    },
    // Defect 6: noun: `arancione → arancio + ne`. Source:
    // childes-romance-germanic-data/Romance/Italian/Burgato/23.
    MisSplitOverride {
        joined_text: "arancione",
        pos: UniversalPos::Noun,
        lemma: "arancione",
        feats: CuratedFeats::new("Gender=Masc|Number=Sing"),
    },
    // Defect 6: adjective: `piccolo → picco + lo`. Source:
    // childes-romance-germanic-data/Romance/Italian/Calambrone/Martina/020322.
    MisSplitOverride {
        joined_text: "piccolo",
        pos: UniversalPos::Adj,
        lemma: "piccolo",
        feats: CuratedFeats::new("Gender=Masc|Number=Sing"),
    },
    // Defect 6: noun: `gomitolo → gomito + lo`. Source:
    // childes-romance-germanic-data/Romance/Italian/Tonelli/Marco/011026.
    MisSplitOverride {
        joined_text: "gomitolo",
        pos: UniversalPos::Noun,
        lemma: "gomitolo",
        feats: CuratedFeats::new("Gender=Masc|Number=Sing"),
    },
    // Defect 6: noun: `divano → diva + no`. Note that `no` is not
    // even a valid Italian enclitic ending, yet Stanza tags the
    // split as verb+pron. Source:
    // childes-romance-germanic-data/Romance/Italian/Tonelli/Marco/010803.
    MisSplitOverride {
        joined_text: "divano",
        pos: UniversalPos::Noun,
        lemma: "divano",
        feats: CuratedFeats::new("Gender=Masc|Number=Sing"),
    },
    // Defect 6 non-verb entries surfaced by the 2026-04-24 CHILDES-ita
    // scan + direct probe. All four mis-splits share the shape
    // `verb|STEM + pron|CLITIC` with a nonsense STEM lemma
    // (`pallare`, `bastare`, `cappere`, `diffire`, none are real
    // Italian verbs). Corpus frequencies: 94 / 48 / 56 / 46
    // respectively across the 184-file scan.
    MisSplitOverride {
        joined_text: "pallone",
        pos: UniversalPos::Noun,
        lemma: "pallone",
        feats: CuratedFeats::new("Gender=Masc|Number=Sing"),
    },
    MisSplitOverride {
        joined_text: "bastone",
        pos: UniversalPos::Noun,
        lemma: "bastone",
        feats: CuratedFeats::new("Gender=Masc|Number=Sing"),
    },
    MisSplitOverride {
        joined_text: "cappello",
        pos: UniversalPos::Noun,
        lemma: "cappello",
        feats: CuratedFeats::new("Gender=Masc|Number=Sing"),
    },
    // `difficile` is an adjective rather than a noun, the `-le`
    // ending masquerades as the pronoun `le` to Stanza's splitter.
    MisSplitOverride {
        joined_text: "difficile",
        pos: UniversalPos::Adj,
        lemma: "difficile",
        feats: CuratedFeats::new("Number=Sing"),
    },
    // Defect 6 non-verb entries surfaced by the 2026-04-24
    // audit_italian_mor_content.py run against a pre-parsed JSON
    // corpus mirror covering the Italian data repos (childes-*,
    // aphasia, slabank, ca-data MOVIN). Detected via the
    // `STEM + CLITIC == surface` signature in committed %mor
    // with nonsense STEM lemma. Current-Stanza probes on
    // 2026-04-24 confirmed each is still mis-split.
    MisSplitOverride {
        joined_text: "seggiola",
        pos: UniversalPos::Noun,
        lemma: "seggiola",
        feats: CuratedFeats::new("Gender=Fem|Number=Sing"),
    },
    // `piccola` is the feminine of already-handled `piccolo`;
    // Stanza splits it as `picco + la` (distinct from `picco + lo`
    // for the masculine), so a separate allowlist entry is needed.
    MisSplitOverride {
        joined_text: "piccola",
        pos: UniversalPos::Adj,
        lemma: "piccolo",
        feats: CuratedFeats::new("Gender=Fem|Number=Sing"),
    },
    MisSplitOverride {
        joined_text: "trottola",
        pos: UniversalPos::Noun,
        lemma: "trottola",
        feats: CuratedFeats::new("Gender=Fem|Number=Sing"),
    },
    MisSplitOverride {
        joined_text: "bottone",
        pos: UniversalPos::Noun,
        lemma: "bottone",
        feats: CuratedFeats::new("Gender=Masc|Number=Sing"),
    },
    // Singleton audit hits (2026-04-24), each appeared exactly
    // once in the committed-corpus audit, but each is a common
    // enough Italian word that adding the entry is low-cost
    // insurance against future corpus processing. Rare/dialectal
    // singletons (`soffioni`, `coccolo`, `pettole`, `babbolo`,
    // `tecala`) were deliberately skipped, see the pause notes
    // for the long-tail deferral reasoning.
    MisSplitOverride {
        joined_text: "cielo",
        pos: UniversalPos::Noun,
        lemma: "cielo",
        feats: CuratedFeats::new("Gender=Masc|Number=Sing"),
    },
    // Italian UD tags consonant-final adjectives without gender
    // (the form is gender-invariant); only Number is marked.
    MisSplitOverride {
        joined_text: "normale",
        pos: UniversalPos::Adj,
        lemma: "normale",
        feats: CuratedFeats::new("Number=Sing"),
    },
    // Augmentative of the common noun `cavallo`; Stanza tags
    // `cavallo` itself correctly but splits the `-one`
    // augmentative. Keep lemma as the augmentative surface per
    // Italian convention (augmentatives lemmatize to themselves,
    // not to the base noun).
    MisSplitOverride {
        joined_text: "cavallone",
        pos: UniversalPos::Noun,
        lemma: "cavallone",
        feats: CuratedFeats::new("Gender=Masc|Number=Sing"),
    },
    MisSplitOverride {
        joined_text: "coccole",
        pos: UniversalPos::Noun,
        lemma: "coccole",
        feats: CuratedFeats::new("Gender=Fem|Number=Plur"),
    },
    // Defect 7: sentence-initial article `la → il + i`. Stanza
    // expands the feminine singular article into a masculine
    // singular + masculine plural article pair. Source:
    // childes-other-data/Frogs/Italian-Roma/10/10dancop.cha
    // "la storia parla di un bambino".
    MisSplitOverride {
        joined_text: "la",
        pos: UniversalPos::Det,
        lemma: "il",
        feats: CuratedFeats::new("Definite=Def|Gender=Fem|Number=Sing|PronType=Art"),
    },
];

/// Look up a potential mis-split override for an MWT Range's
/// parent token.
///
/// The key is the Range parent's `text` field, the original input
/// word Stanza received before any MWT expansion. Matching is
/// case-insensitive.
///
/// Returns `None` for any input not in the allowlist. Genuine
/// verb+clitic compounds (`portalo`) then keep Stanza's split on the
/// `~`-merged path; the ones in `IT_COMPOUND_IMPERATIVES` (`dammela`)
/// take the curated analysis instead (see `try_handle_italian_range_override`).
fn check_italian_mis_split(range_parent_text: &str) -> Option<&'static MisSplitOverride> {
    // All allowlist `joined_text` entries are pure ASCII, so
    // `eq_ignore_ascii_case` is correct and zero-allocation.
    IT_MIS_SPLIT_OVERRIDES
        .iter()
        .find(|o| o.joined_text.eq_ignore_ascii_case(range_parent_text))
}

/// Apply a mis-split override by synthesizing a single-word
/// `UdWord` with the overridden POS/lemma/feats and mapping it
/// through the normal `map_ud_word` pipeline.
///
/// The one chunk keeps the attachment of the producer-selected
/// representative, through `main`, so the GRA
/// relation built for it preserves the sentence's dependency structure.
fn apply_mis_split_override(
    over: &MisSplitOverride,
    attachment: &UdWord,
    main: ChunkProvenance,
    ctx: &MappingContext,
) -> Result<MappedItem, MappingError> {
    let synthetic = UdWord::curated_reading(
        attachment,
        over.joined_text,
        over.lemma,
        over.pos,
        over.feats,
    );
    Ok(MappedItem::word(map_ud_mor_word(&synthetic, ctx)?, main))
}

/// Handle Italian Range-token reconciler cases if the parent token is in one of
/// the known Stanza mis-split allowlists.
pub fn try_handle_italian_range_override(
    token: WalkedUdToken<'_, '_>,
    ctx: &MappingContext,
) -> Result<Option<MappedItem>, MappingError> {
    let AlignedUd::Mwt {
        range: ud,
        components,
    } = token.ud()
    else {
        return Ok(None);
    };
    let attachment = token.representative();
    if let Some(over) = check_italian_mis_split(&ud.text) {
        let UdId::Range(start, end) = ud.id else {
            return Ok(None);
        };
        let deprel = normalize_deprel(&attachment.deprel, || {
            format!("collapsed Range {:?}", ud.text)
        })?;
        let main = ChunkProvenance::collapsed_range(
            UdWordId::range(start, end),
            ChunkHead::of_word(attachment),
            deprel,
        );
        return apply_mis_split_override(over, attachment, main, ctx).map(Some);
    }

    // A known compound imperative that Stanza split: the curated analysis
    // replaces Stanza's components however it split them. Stanza 1.14 split
    // `dammela` as `da` VERB with imperfect-indicative features; 1.15, in
    // context, as `da` ADP. Neither is the imperative the word is.
    if let Some(over) = compound_imperative_for_surface(&ud.text) {
        let UdId::Range(start, end) = ud.id else {
            return Ok(None);
        };
        let deprel = normalize_deprel(&attachment.deprel, || {
            format!("collapsed Range {:?}", ud.text)
        })?;
        let main = ChunkProvenance::collapsed_range(
            UdWordId::range(start, end),
            ChunkHead::of_word(attachment),
            deprel,
        );
        return apply_compound_imperative_override(over, ud, main, ctx).map(Some);
    }

    if let Some(rewrite) = check_italian_component_rewrite(&ud.text) {
        let rewritten = apply_component_rewrite(rewrite, components);
        return assemble_mors(&rewritten, ctx).map(Some);
    }

    Ok(None)
}

// ─── Defect 8: mid-sentence compound imperative mis-classification ──
//
// Stanza mis-analyses known imperative+enclitic compounds, and the
// shape of the mistake has changed across versions:
//
// - 1.11.1: mid-sentence `dammela` was ONE word, ADJ, lemma `dammelo`
//   (no MWT Range), while the isolated word split correctly.
// - 1.14.0: the word split into `da` VERB + `me` + `la`, but with
//   imperfect-indicative features (`Mood=Ind|Tense=Imp`) and the feature
//   name misspelled `Verbform`.
// - 1.15.0 in context: the word split into `da` ADP + `me` + `la`.
//
// The curated table below is the analysis of each surface, whatever Stanza
// did with it: `try_handle_italian_single_override` applies it to an
// unsplit word tagged ADJ, NOUN or VERB, and `try_handle_italian_range_override`
// to a split one, replacing the components. Either way the result is one
// `Mor` of the imperative verb with its clitics stacked as post-clitics
// (`verb|dare-Fin-Imp-S2~pron|me-...~pron|la-...`), and the verb keeps the
// word's attachment to the rest of the clause.

/// One post-clitic in a reconciled Italian compound imperative.
///
/// Each entry in `IT_COMPOUND_IMPERATIVES` carries a static list
/// of post-clitics; the reconciler synthesizes one `MorWord` per
/// clitic via `map_ud_word` and attaches it to the main
/// verb `Mor` via `with_post_clitic`. This produces the full
/// multi-chunk `verb|LEMMA~pron|X~pron|Y` output that matches
/// Stanza's native analysis for the bare-compound case.
///
/// An empty clitic slice means "single-chunk mode", the
/// reconciler emits only the main verb Mor. Use this when the
/// historical single-chunk output is sufficient and the clitic
/// decomposition isn't worth spelling out.
#[derive(Debug, Clone)]
pub(crate) struct CliticSpec {
    /// Surface text of the clitic (`me`, `mi`, `la`, `lo`, …).
    pub text: &'static str,
    /// UD lemma for the clitic. Italian UD tends to use the
    /// clitic's canonical form as lemma (`me`, `la`, `gli`).
    pub lemma: &'static str,
    /// UD POS. Typically `Pron`; `Adv` only for `ne` in some
    /// analyses.
    pub upos: UniversalPos,
    /// UD feats for the clitic: number, person, gender, PronType, etc.
    pub feats: CuratedFeats,
    /// UD dependency relation from this clitic to the main verb:
    /// `obj` for direct-object clitics (`la`, `lo`, `le`, `li`),
    /// `iobj` for indirect-object clitics (`me`, `mi`, `ti`,
    /// `ci`, `vi`, `gli`), etc. The GRA builder uses this to
    /// emit a relation pointing from the clitic chunk back to
    /// the main verb's chunk.
    pub deprel: &'static str,
}

/// Known Italian imperative+enclitic compounds that Stanza
/// mis-classifies mid-sentence.
///
/// Each entry pairs the surface form (as Stanza sees it) with the
/// correct verb lemma, feats, and the clitic stack. Add entries
/// when corpus scans surface new mis-classifications.
///
/// The Stanza POS tags that signal a potential mis-classification,
/// and so allow the gate to fire, are inlined in the `matches!` at
/// the top of `check_italian_compound_imperative`. (This used to
/// name a constant `IT_COMPOUND_IMPERATIVE_GATE_POS`, which does
/// not exist anywhere in the workspace: a doc value that outlived
/// the thing it described.)
#[derive(Debug, Clone)]
pub(crate) struct CompoundImperativeOverride {
    /// Surface form Stanza received, matched case-insensitively
    /// against the UdWord's `text` field.
    pub surface: &'static str,
    /// The correct verb lemma (e.g. `dare` for `dammela`,
    /// `portare` for `portalo`).
    pub verb_lemma: &'static str,
    /// UD feats for the reconciled verb Mor. Encodes imperative
    /// 2sg (or 2pl) by default.
    pub verb_feats: CuratedFeats,
    /// Post-clitics stacked on the main verb, in serialization
    /// order. Empty slice means "emit main verb only"
    /// single-chunk mode preserved for allowlist entries where
    /// the clitic decomposition isn't worth spelling out.
    pub clitics: &'static [CliticSpec],
}

/// Common Italian clitic specs reused across multiple entries.
/// Kept as module-level constants so `IT_COMPOUND_IMPERATIVES`
/// entries stay compact and mutation-free.
const CLITIC_ME: CliticSpec = CliticSpec {
    text: "me",
    lemma: "me",
    upos: UniversalPos::Pron,
    feats: CuratedFeats::new("Number=Sing|Person=1|PronType=Prs"),
    deprel: "iobj",
};
const CLITIC_LA: CliticSpec = CliticSpec {
    text: "la",
    lemma: "la",
    upos: UniversalPos::Pron,
    feats: CuratedFeats::new("Gender=Fem|Number=Sing|Person=3|PronType=Prs"),
    deprel: "obj",
};
const CLITIC_LO: CliticSpec = CliticSpec {
    text: "lo",
    lemma: "lo",
    upos: UniversalPos::Pron,
    feats: CuratedFeats::new("Gender=Masc|Number=Sing|Person=3|PronType=Prs"),
    deprel: "obj",
};
const CLITIC_LI: CliticSpec = CliticSpec {
    text: "li",
    lemma: "li",
    upos: UniversalPos::Pron,
    feats: CuratedFeats::new("Gender=Masc|Number=Plur|Person=3|PronType=Prs"),
    deprel: "obj",
};
const CLITIC_LE: CliticSpec = CliticSpec {
    text: "le",
    lemma: "le",
    upos: UniversalPos::Pron,
    feats: CuratedFeats::new("Gender=Fem|Number=Plur|Person=3|PronType=Prs"),
    deprel: "obj",
};

/// Table of known Italian mid-sentence compound imperative
/// mis-classifications. Conservative seed, only two surfaces
/// confirmed via direct Stanza observation (2026-04-23 probes).
/// Extension is mechanical: one row per newly-observed surface.
pub(crate) const IT_COMPOUND_IMPERATIVES: &[CompoundImperativeOverride] = &[
    // dammela / dammelo (dare + me + la/lo): direct probe observation
    //: Stanza mid-sentence tags ADJ with lemma=`dammelo`.
    CompoundImperativeOverride {
        surface: "dammela",
        verb_lemma: "dare",
        verb_feats: CuratedFeats::new("Mood=Imp|Number=Sing|Person=2|VerbForm=Fin"),
        clitics: &[CLITIC_ME, CLITIC_LA],
    },
    CompoundImperativeOverride {
        surface: "dammelo",
        verb_lemma: "dare",
        verb_feats: CuratedFeats::new("Mood=Imp|Number=Sing|Person=2|VerbForm=Fin"),
        clitics: &[CLITIC_ME, CLITIC_LO],
    },
    // prendilo / prendila / prendili / prendile
    // (prendere + lo/la/li/le): surfaced by the 2026-04-23 corpus
    // scan (`scripts/analysis/scan_italian_compound_imperative_candidates.py`);
    // `prendilo` confirmed via direct Stanza probe as ADJ-tagged
    // mid-sentence. The other three are family additions, Stanza's
    // behavior is consistent within an imperative+clitic paradigm
    // once one surface mis-classifies, but each should receive its
    // own probe-observation in a follow-up pass.
    CompoundImperativeOverride {
        surface: "prendilo",
        verb_lemma: "prendere",
        verb_feats: CuratedFeats::new("Mood=Imp|Number=Sing|Person=2|VerbForm=Fin"),
        clitics: &[CLITIC_LO],
    },
    CompoundImperativeOverride {
        surface: "prendila",
        verb_lemma: "prendere",
        verb_feats: CuratedFeats::new("Mood=Imp|Number=Sing|Person=2|VerbForm=Fin"),
        clitics: &[CLITIC_LA],
    },
    CompoundImperativeOverride {
        surface: "prendili",
        verb_lemma: "prendere",
        verb_feats: CuratedFeats::new("Mood=Imp|Number=Sing|Person=2|VerbForm=Fin"),
        clitics: &[CLITIC_LI],
    },
    CompoundImperativeOverride {
        surface: "prendile",
        verb_lemma: "prendere",
        verb_feats: CuratedFeats::new("Mood=Imp|Number=Sing|Person=2|VerbForm=Fin"),
        clitics: &[CLITIC_LE],
    },
    // `-ire` family surfaces where Stanza fails to detect the MWT
    // boundary and mis-classifies the single-word surface.
    // Observed 2026-04-24 via direct probe:
    // - `aprila` / `aprili`, tagged NOUN (not ADJ); `aprili`
    //   additionally gets a homograph lemma `aprile` (month April).
    //   Both should be imperative `aprire` + clitic.
    // - `finila`: tagged ADJ with lemma `finile` (vowel-normalized
    //   surface echo); should be imperative `finire` + clitic.
    // Gate accepts both ADJ and NOUN since both signatures appear.
    CompoundImperativeOverride {
        surface: "aprila",
        verb_lemma: "aprire",
        verb_feats: CuratedFeats::new("Mood=Imp|Number=Sing|Person=2|VerbForm=Fin"),
        clitics: &[CLITIC_LA],
    },
    CompoundImperativeOverride {
        surface: "aprili",
        verb_lemma: "aprire",
        verb_feats: CuratedFeats::new("Mood=Imp|Number=Sing|Person=2|VerbForm=Fin"),
        clitics: &[CLITIC_LI],
    },
    CompoundImperativeOverride {
        surface: "finila",
        verb_lemma: "finire",
        verb_feats: CuratedFeats::new("Mood=Imp|Number=Sing|Person=2|VerbForm=Fin"),
        clitics: &[CLITIC_LA],
    },
    // Defect 12: `aprilo`: Stanza tags it as single-word VERB
    // with correct lemma `aprire`, but fails to do the MWT
    // expansion. Only the clitic decomposition is missing.
    // Direct probe 2026-04-24.
    CompoundImperativeOverride {
        surface: "aprilo",
        verb_lemma: "aprire",
        verb_feats: CuratedFeats::new("Mood=Imp|Number=Sing|Person=2|VerbForm=Fin"),
        clitics: &[CLITIC_LO],
    },
    // Defect 13: `leggila`: Stanza tags it as single-word VERB
    // with a **fabricated** lemma `leggilare` (not a real Italian
    // verb). Should be imperative `leggere` + `la`. Direct probe
    // 2026-04-24.
    CompoundImperativeOverride {
        surface: "leggila",
        verb_lemma: "leggere",
        verb_feats: CuratedFeats::new("Mood=Imp|Number=Sing|Person=2|VerbForm=Fin"),
        clitics: &[CLITIC_LA],
    },
];

/// Look up a potential mid-sentence compound imperative override
/// for a single UdWord.
///
/// The gate is two-pronged: Stanza must have tagged the word with
/// a POS we have seen indicate mis-classification (ADJ, NOUN, or
/// VERB with missing MWT expansion) AND the text must appear in
/// the allowlist. Returns `None` when either fails, leaving the
/// normal `map_ud_word` path in charge.
///
/// Gate rationale for accepting ADJ, NOUN, and VERB:
/// - ADJ: the original Defect 8 signature (`dammela`, `prendilo`,
///   `finila`: Stanza vowel-normalizes the lemma and tags
///   adjective).
/// - NOUN: `-ire` imperative+clitic surfaces where Stanza doesn't
///   detect the MWT boundary and tags the whole compound as a
///   single noun (`aprila → aprila/NOUN/aprila`) or a homograph
///   noun (`aprili → aprili/NOUN/aprile`).
/// - VERB: Defects 12 and 13, Stanza gets POS=VERB right but
///   either fails to MWT-expand (`aprilo → aprilo/VERB/aprire`)
///   or invents a fabricated lemma (`leggila → leggila/VERB/
///   leggilare`, where `leggilare` is not a real Italian verb).
///
/// The allowlist is a closed curated set; accepting all three
/// POS values expands the set of mis-classifications the gate
/// can catch. Legitimate tokens whose surface matches an
/// allowlist entry would only be over-matched if a real word
/// shares spelling with one of the entries, no such collisions
/// are known for current entries. Each entry also preserves the
/// original UdWord's head/deprel for the main chunk so GRA
/// reindexing stays consistent.
fn check_italian_compound_imperative(
    text: &str,
    upos: &UdPunctable<UniversalPos>,
) -> Option<&'static CompoundImperativeOverride> {
    if !matches!(
        upos,
        UdPunctable::Value(UniversalPos::Adj)
            | UdPunctable::Value(UniversalPos::Noun)
            | UdPunctable::Value(UniversalPos::Verb),
    ) {
        return None;
    }
    compound_imperative_for_surface(text)
}

/// The curated analysis of a known compound imperative, by surface alone.
fn compound_imperative_for_surface(text: &str) -> Option<&'static CompoundImperativeOverride> {
    // All allowlist `surface` entries are pure ASCII.
    IT_COMPOUND_IMPERATIVES
        .iter()
        .find(|o| o.surface.eq_ignore_ascii_case(text))
}

/// Apply a compound-imperative override by synthesizing the main
/// verb UdWord plus one UdWord per post-clitic, mapping each
/// through `map_ud_word`, and stacking the clitics onto
/// the main `Mor`.
///
/// The result is a multi-chunk `Mor` of shape
/// `verb|LEMMA-<feats>~pron|CLITIC1-<feats>~pron|CLITIC2-<feats>`,
/// matching the output Stanza produces natively for the bare-
/// compound case. When `over.clitics` is empty the output is a
/// single-chunk Mor: that mode is still available but not used
/// by any current allowlist entry.
///
/// The main chunk takes its attachment from `main`; each post-clitic
/// depends on the main chunk (`ChunkHead::OwningMorMain`).
fn apply_compound_imperative_override(
    over: &CompoundImperativeOverride,
    row: &UdWord,
    main: ChunkProvenance,
    ctx: &MappingContext,
) -> Result<MappedItem, MappingError> {
    let main_ud = UdWord::curated_reading(
        row,
        over.surface,
        over.verb_lemma,
        UniversalPos::Verb,
        over.verb_feats,
    );
    let item = MappedItem::word(map_ud_mor_word(&main_ud, ctx)?, main);
    over.clitics.iter().try_fold(item, |item, clitic| {
        let clitic_ud =
            UdWord::curated_reading(row, clitic.text, clitic.lemma, clitic.upos, clitic.feats);
        let deprel = normalize_deprel(clitic.deprel, || {
            format!("synthesized clitic {:?} in {:?}", clitic.text, over.surface)
        })?;
        Ok(item.with_post_clitic(
            map_ud_mor_word(&clitic_ud, ctx)?,
            ChunkProvenance::synthetic_post_clitic(deprel),
        ))
    })
}

/// Handle Italian single-token compound-imperative reconciler cases if the
/// token matches one of the known Stanza misclassification allowlists.
pub fn try_handle_italian_single_override(
    ud: &UdWord,
    ctx: &MappingContext,
) -> Result<Option<MappedItem>, MappingError> {
    match check_italian_compound_imperative(&ud.text, &ud.upos) {
        Some(over) => {
            apply_compound_imperative_override(over, ud, provenance_for_ud_word(ud)?, ctx).map(Some)
        }
        None => Ok(None),
    }
}

// ─── Defect 9: Range-expansion with wrong head POS ──────────────
//
// Stanza-1.11.1 expands certain imperative+clitic compound verbs as
// a structurally-correct MWT Range but tags the head component with
// a homographic non-verb POS (e.g. ADP for `da`) and a surface-echo
// lemma. Observed case: `dagliela` (2sg imperative of `dare` + 3sg
// dative `glie` + 3sg.f.acc `la`) expands as
// `da/ADP/da + glie/PRON/gli + la/PRON/la` instead of
// `da/VERB/dare + glie/PRON/gli + la/PRON/la`. The MWT expansion
// shape is right: it's the head POS/lemma that's wrong.
//
// Sibling imperatives in the same dative-stack family
// (`digliela`, `portagliela`, `prendigliela`) are analyzed
// correctly by Stanza. The defect is specific to surfaces where the
// head clitic-stripped form is homographic with a non-verb word.
//
// Fix shape: rewrite component 0's POS/lemma/feats in-place before
// `assemble_mors` runs. The remaining components pass through
// unchanged, so the 3-chunk `~`-joined Mor shape is preserved
// unlike Defect 6 which collapses the Range into a single chunk.

/// One entry in the Italian Defect-9 component-rewrite allowlist.
///
/// Matched case-insensitively against the Range parent's `text`
/// field (i.e. the joined surface form Stanza received before MWT
/// expansion). When matched, the reconciler produces a rewritten
/// slice of components with component 0's POS/lemma/feats replaced,
/// then passes the rewritten slice to `assemble_mors`.
#[derive(Debug, Clone)]
pub(crate) struct ComponentRewriteOverride {
    /// The Range parent text Stanza mis-analyzes. Matched
    /// case-insensitively. Example: `"dagliela"`.
    pub joined_text: &'static str,
    /// POS to assign to component 0 of the expansion.
    pub head_pos: UniversalPos,
    /// Lemma for component 0 (the correct lexical root).
    pub head_lemma: &'static str,
    /// UD feats string for component 0 (e.g. imperative 2sg).
    pub head_feats: CuratedFeats,
}

/// Table of known Italian Defect-9 component rewrites. Closed
/// allowlist; each entry is seeded from a specific direct-probe
/// observation against Stanza. See the module docstring for the
/// scope trade-off.
pub(crate) const IT_COMPONENT_REWRITES: &[ComponentRewriteOverride] = &[
    // Defect 9: `dagliela` (imperative `dare` + `glie` + `la`):
    // Stanza emits `da/ADP/da + glie/PRON/gli + la/PRON/la`. The
    // expansion is shape-correct (3 pieces for a 3-piece clitic
    // stack) but the head is mis-tagged ADP with surface-echo lemma.
    // Probe source:
    // `batchalign/tests/investigations/_cases/italian.py`
    // `dagliela_mid_sentence` (2026-04-24). Sibling forms
    // `digliela`/`portagliela`/`prendigliela` are Stanza-correct
    // and must stay off this allowlist.
    ComponentRewriteOverride {
        joined_text: "dagliela",
        head_pos: UniversalPos::Verb,
        head_lemma: "dare",
        head_feats: CuratedFeats::new("Mood=Imp|Number=Sing|Person=2|VerbForm=Fin"),
    },
    // Defect 10: `posala` / `posalo` (imperative `posare` +
    // `la`/`lo`): Stanza emits `posa/VERB/posa + la/PRON`, shape
    // correct, head POS correct, but head lemma is surface-echo
    // `posa` instead of the canonical infinitive `posare`. Unlike
    // Defect 9, the POS rewrite is a no-op; only the lemma is wrong.
    // Fits the same component-rewrite shape, so no new allowlist is
    // needed.
    //
    // Verb-specific to `posare`: the 2026-04-24 cross-verb probe
    // confirmed `guardare`, `toccare`, `aspettare`, `mangiare` all
    // lemmatize correctly in this position. Stanza's Italian model
    // has a specific weakness on the `posare` paradigm.
    //
    // Audit source: 1 hit in the 2026-04-24 fleet JSON audit
    // (`posala`). `posalo` added as a family member via direct
    // probe confirmation.
    ComponentRewriteOverride {
        joined_text: "posala",
        head_pos: UniversalPos::Verb,
        head_lemma: "posare",
        head_feats: CuratedFeats::new("Mood=Imp|Number=Sing|Person=2|VerbForm=Fin"),
    },
    ComponentRewriteOverride {
        joined_text: "posalo",
        head_pos: UniversalPos::Verb,
        head_lemma: "posare",
        head_feats: CuratedFeats::new("Mood=Imp|Number=Sing|Person=2|VerbForm=Fin"),
    },
];

/// Look up a potential Defect-9 component-rewrite override for an
/// MWT Range's parent token. Returns `None` when the input is not
/// in the allowlist, leaving the normal `assemble_mors` path in
/// charge, which is what keeps correctly-analyzed siblings
/// (`digliela`, `portagliela`, `prendigliela`, `dammela`, ...) on
/// the correct path.
fn check_italian_component_rewrite(
    range_parent_text: &str,
) -> Option<&'static ComponentRewriteOverride> {
    // All allowlist `joined_text` entries are pure ASCII.
    IT_COMPONENT_REWRITES
        .iter()
        .find(|o| o.joined_text.eq_ignore_ascii_case(range_parent_text))
}

/// Apply a Defect-9 component-rewrite by cloning the components
/// slice and overwriting component 0's POS/lemma/feats. The
/// returned Vec is owned by the caller and can be passed directly
/// to `assemble_mors`.
///
/// The rewrite is purely local, only component 0's `upos`,
/// `lemma`, and `feats` change. All other fields (id, text, head,
/// deprel, deps, misc) are preserved so GRA arc reindexing stays
/// consistent with the normal Range path.
fn apply_component_rewrite(over: &ComponentRewriteOverride, components: &[UdWord]) -> Vec<UdWord> {
    let mut rewritten: Vec<UdWord> = components.to_vec();
    if let Some(head) = rewritten.first_mut() {
        head.upos = UdPunctable::Value(over.head_pos);
        head.lemma = over.head_lemma.to_string();
        head.apply_curated(over.head_feats);
    }
    rewritten
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::morphosyntax::UdWordAnalysis;

    #[test]
    fn check_returns_some_for_allowlist_entries() {
        for over in IT_MIS_SPLIT_OVERRIDES {
            let found = check_italian_mis_split(over.joined_text);
            assert!(
                found.is_some(),
                "expected {} to be found in allowlist",
                over.joined_text
            );
        }
    }

    #[test]
    fn check_is_case_insensitive() {
        assert!(check_italian_mis_split("Parla").is_some());
        assert!(check_italian_mis_split("PARLA").is_some());
        assert!(check_italian_mis_split("PiCcOlO").is_some());
    }

    #[test]
    fn check_returns_none_for_genuine_compounds() {
        // Genuine verb+clitic compounds must not be in the allowlist.
        assert!(check_italian_mis_split("dammela").is_none());
        assert!(check_italian_mis_split("dammelo").is_none());
        assert!(check_italian_mis_split("portalo").is_none());
        assert!(check_italian_mis_split("guardarla").is_none());
    }

    #[test]
    fn check_returns_none_for_unrelated_words() {
        assert!(check_italian_mis_split("casa").is_none());
        assert!(check_italian_mis_split("bambino").is_none());
        assert!(check_italian_mis_split("").is_none());
    }

    // ─── Defect 9 component-rewrite tests ──

    #[test]
    fn component_rewrite_fires_for_dagliela() {
        assert!(check_italian_component_rewrite("dagliela").is_some());
        assert!(check_italian_component_rewrite("DAGLIELA").is_some());
        assert!(check_italian_component_rewrite("DaGlIeLa").is_some());
    }

    #[test]
    fn component_rewrite_skips_stanza_correct_siblings() {
        // Stanza analyses these correctly, must NOT be in the
        // Defect 9 allowlist or the reconciler would corrupt their
        // already-good head POS/lemma.
        assert!(check_italian_component_rewrite("digliela").is_none());
        assert!(check_italian_component_rewrite("portagliela").is_none());
        assert!(check_italian_component_rewrite("prendigliela").is_none());
        // Bare 2sg imperatives with clitics (Defect 8 / correct MWT
        // territory, not Defect 9).
        assert!(check_italian_component_rewrite("dammela").is_none());
        assert!(check_italian_component_rewrite("portalo").is_none());
    }

    #[test]
    fn apply_component_rewrite_only_touches_head() {
        let over = &IT_COMPONENT_REWRITES[0];
        let components = vec![
            UdWord::from(UdWordAnalysis {
                id: UdId::Single(1),
                text: "da".into(),
                lemma: "da".into(),
                upos: UdPunctable::Value(UniversalPos::Adp),
                xpos: None,
                feats: None,
                head: 0,
                deprel: "root".into(),
                deps: None,
                misc: None,
            }),
            UdWord::from(UdWordAnalysis {
                id: UdId::Single(2),
                text: "glie".into(),
                lemma: "gli".into(),
                upos: UdPunctable::Value(UniversalPos::Pron),
                xpos: None,
                feats: Some("Person=3".into()),
                head: 1,
                deprel: "iobj".into(),
                deps: None,
                misc: None,
            }),
        ];
        let rewritten = apply_component_rewrite(over, &components);
        assert_eq!(rewritten.len(), 2);
        // Head rewritten.
        assert!(matches!(
            rewritten[0].upos,
            UdPunctable::Value(UniversalPos::Verb)
        ));
        assert_eq!(rewritten[0].lemma, "dare");
        assert_eq!(
            rewritten[0].features().to_string(),
            "Mood=Imp|Number=Sing|Person=2|VerbForm=Fin"
        );
        // The table's values, and they say so.
        assert!(
            rewritten[0]
                .features()
                .iter()
                .all(|(_, value)| value.source() == crate::morphosyntax::FeatSource::Curated)
        );
        // Head text / id / head / deprel preserved so GRA reindexing
        // stays consistent.
        assert_eq!(rewritten[0].text, "da");
        assert_eq!(rewritten[0].deprel, "root");
        // Tail component untouched.
        assert_eq!(rewritten[1], components[1]);
    }
}
