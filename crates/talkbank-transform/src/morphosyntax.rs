//! Per-utterance morphotag outcome classification.
//!
//! The morphotag pipeline emits exactly one [`MorOutcome`] per utterance it
//! visits. The outcome is a typed statement of what happened, not a
//! side-effect of logging or an implicit silent skip:
//!
//! - [`MorOutcomeKind::NotApplicable`] — the utterance had zero Mor-alignable
//!   words under CHAT policy (fillers, fragments, untranscribed material
//!   only), so no `%mor` tier is produced. This is correct, expected
//!   behavior, not a failure.
//! - [`MorOutcomeKind::Aligned`] — Stanza returned exactly N tokens for N
//!   CHAT words after MWT reassembly; `%mor` / `%gra` were injected.
//! - [`MorOutcomeKind::MisalignmentBug`] — the `|stanza_tokens| = |chat_words|`
//!   invariant was violated. This is a bug in one of extraction, Stanza
//!   realignment, MWT reassembly, or the terminator filter. The diagnostic
//!   carries enough data to triage the stage without re-running.
//!
//! The invariant is deterministic by construction:
//!
//! 1. CHAT-side extraction uses
//!    [`counts_for_tier`](talkbank_model::alignment::helpers::counts_for_tier)
//!    to yield exactly N alignable Mor-domain words.
//! 2. The Python worker sets
//!    `tok_ctx.original_words = word_lists`
//!    (`batchalign/inference/morphosyntax.py:348-349`), so Stanza's
//!    tokenizer realigns to the pre-specified CHAT word boundaries.
//! 3. MWT expansions are signaled via Range token IDs and reassembled
//!    1-chunk-per-CHAT-word in `nlp::mapping::map_ud_sentence`.
//!
//! When all three cooperate, the count matches by construction.
//! [`MisalignmentBug`](MorOutcomeKind::MisalignmentBug) is therefore
//! never silently absorbed — it is typed, logged, and surfaced through
//! [`DecisionRecord`] so operators see it and developers can fix it.
//!
//! See `book/src/architecture/morphotag-invariants.md` for the full
//! architectural discussion.

use std::collections::{BTreeMap, HashSet};
use std::sync::LazyLock;

use talkbank_model::WriteChat;
use talkbank_model::alignment::helpers::{TierDomain, WordItem, walk_words};
use talkbank_model::model::{LanguageCode, Line, SpeakerCode, Utterance, WordCategory};

use crate::decisions::{DecisionRecord, DecisionStrategy, MorphosyntaxStrategy};
use crate::extract::{self, ExtractedWord};
pub use crate::inject::{MisalignmentClass, MisalignmentDiagnostic};

mod features;
mod gra_validate;
mod injection;
mod invariants;
pub mod l2;
mod lang_en;
mod lang_fr;
mod lang_it;
mod lang_ja;
mod mapping_helpers;
mod mapping_provenance;
mod mor_word;
mod sentence_mapping;
mod stanza_raw;
pub use gra_validate::validate_generated_gra;
pub use injection::{InjectionResult, RetokenizationInfo, inject_results};
pub use invariants::*;
pub use lang_en::is_irregular;
pub use lang_fr::{french_pronoun_case, is_apm_noun};
pub use lang_it::{try_handle_italian_range_override, try_handle_italian_single_override};
pub use lang_ja::{JaOverride, japanese_verbform};
pub use mapping_helpers::{assemble_mors, normalize_deprel, provenance_for_ud_word};
pub use mapping_provenance::{ChunkHead, ChunkProvenance, MorProvenance};
pub use mor_word::{clean_lemma, is_clitic, map_ud_word_to_mor};
pub use sentence_mapping::{
    TerminatorPolicy, build_gra_and_validate, is_terminator_punct, map_ud_sentence,
    map_ud_sentence_expanded, map_ud_sentence_with_overrides,
};
pub use stanza_raw::*;

/// Alias for the MWT lexicon: surface form → expansion tokens.
pub type MwtDict = BTreeMap<String, Vec<String>>;

/// Context for UD-to-CHAT morphosyntax mapping and language-specific rewrites.
pub struct MappingContext {
    /// Language code used to select language-specific override rules.
    pub lang: LanguageCode,
}

/// Normalize a language code to its 2-letter form.
///
/// The pipeline passes 3-letter ISO 639-3 codes ("eng", "fra", "jpn"), but
/// some language-specific logic is keyed by 2-letter ISO 639-1 codes ("en",
/// "fr", "ja"). Unknown codes are returned unchanged.
pub fn lang2(code: &str) -> &str {
    match code {
        "eng" => "en",
        "fra" | "fre" => "fr",
        "jpn" => "ja",
        "deu" | "ger" => "de",
        "ita" => "it",
        "spa" => "es",
        "por" => "pt",
        "zho" | "cmn" | "chi" => "zh",
        "heb" => "he",
        "ara" => "ar",
        "nld" | "dut" => "nl",
        "cat" => "ca",
        s if s.len() <= 2 => s,
        s => s,
    }
}

/// Structured error type for UD-to-CHAT mapping failures.
#[derive(Debug, thiserror::Error)]
pub enum MappingError {
    /// A word produced an empty MOR stem after lemma cleaning and sanitization.
    #[error("Empty MOR stem: word={word:?}, lemma={lemma:?}, upos={upos:?}")]
    EmptyStem {
        /// Original word form.
        word: String,
        /// Lemma after cleaning.
        lemma: String,
        /// Universal POS tag.
        upos: String,
    },

    /// The generated %gra tier has a circular dependency.
    #[error("Circular dependency in generated %gra: {details}")]
    CircularDependency {
        /// Description of the cycle.
        details: String,
    },

    /// The generated %gra tier has an invalid head reference.
    #[error("Invalid head reference in generated %gra: {details}")]
    InvalidHeadReference {
        /// Description of the invalid reference.
        details: String,
    },

    /// Generated %mor and %gra have mismatched chunk counts.
    #[error("%mor has {mor_chunks} chunks but %gra has {gra_count} relations")]
    ChunkCountMismatch {
        /// Number of %mor chunks.
        mor_chunks: usize,
        /// Number of %gra relations.
        gra_count: usize,
    },

    /// The generated %gra tier has no root or multiple roots.
    #[error("Invalid root structure in generated %gra: {details}")]
    InvalidRoot {
        /// Description of the root problem.
        details: String,
    },

    /// A UD word has a deprel value that cannot produce a valid CHAT %gra relation.
    #[error("Invalid deprel in UD parse: {details}")]
    InvalidDeprel {
        /// Description of the invalid deprel.
        details: String,
    },

    /// `assemble_mors` was called with an empty component slice.
    #[error("assemble_mors called with empty components — structural bug in caller")]
    EmptyRangeComponents,
}

/// Controls whether the morphosyntax pipeline retokenizes using Stanza's
/// neural tokenizer or preserves original CHAT word boundaries.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum TokenizationMode {
    /// Preserve original CHAT tokenization.
    Preserve,
    /// Allow Stanza retokenization to rewrite CHAT words.
    StanzaRetokenize,
}

impl From<bool> for TokenizationMode {
    fn from(retokenize: bool) -> Self {
        if retokenize {
            Self::StanzaRetokenize
        } else {
            Self::Preserve
        }
    }
}

/// Controls whether utterances marked with a non-primary language are processed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum MultilingualPolicy {
    /// Process all utterances regardless of `@s` language marking.
    ProcessAll,
    /// Skip utterances whose `@s` language marker differs from the primary file language.
    SkipNonPrimary,
}

impl MultilingualPolicy {
    /// Convert from the legacy boolean flag used at CLI and PyO3 boundaries.
    pub fn from_skip_flag(skip: bool) -> Self {
        if skip {
            Self::SkipNonPrimary
        } else {
            Self::ProcessAll
        }
    }

    /// Whether non-primary-language utterances should be skipped.
    pub fn should_skip_non_primary(self) -> bool {
        matches!(self, Self::SkipNonPrimary)
    }
}

/// Batch item for morphosyntax NLP processing.
#[derive(Clone, serde::Serialize, serde::Deserialize, schemars::JsonSchema)]
pub struct MorphosyntaxBatchItem {
    /// Word texts for NLP processing.
    pub words: Vec<String>,
    /// Utterance terminator string.
    pub terminator: String,
    /// Special form and language per word: (form_type, resolved_language).
    #[serde(serialize_with = "serialize_special_forms")]
    #[schemars(with = "Vec<(Option<String>, Option<String>)>")]
    pub special_forms: Vec<(
        Option<talkbank_model::model::FormType>,
        Option<talkbank_model::validation::LanguageResolution>,
    )>,
    /// Language code for this utterance (ISO 639-3).
    #[schemars(with = "String")]
    pub lang: talkbank_model::model::LanguageCode,
}

fn serialize_special_forms<S: serde::Serializer>(
    forms: &[(
        Option<talkbank_model::model::FormType>,
        Option<talkbank_model::validation::LanguageResolution>,
    )],
    serializer: S,
) -> Result<S::Ok, S::Error> {
    use serde::ser::SerializeSeq;

    let mut seq = serializer.serialize_seq(Some(forms.len()))?;
    for (form_type, lang_res) in forms {
        let ft_str: Option<String> = form_type.as_ref().map(|ft| {
            let mut buf = String::new();
            #[allow(clippy::expect_used)]
            ft.write_chat(&mut buf)
                .expect("writing CHAT to a String should be infallible");
            buf
        });
        let lang_str: Option<String> = lang_res
            .as_ref()
            .and_then(|lr| lr.languages().first().map(|lc| lc.to_string()));
        seq.serialize_element(&(ft_str, lang_str))?;
    }
    seq.end()
}

/// A collected batch item with its position in the `ChatFile`, for injection.
pub type BatchItemWithPosition = (usize, usize, MorphosyntaxBatchItem, Vec<ExtractedWord>);

/// Validation warning for a single utterance.
#[derive(Debug)]
pub struct AlignmentWarning {
    /// Zero-based line index in the `ChatFile`.
    pub line_idx: usize,
    /// Main tier word count (alignable words in the Mor domain).
    pub main_count: usize,
    /// `%mor` item count.
    pub mor_count: usize,
}

impl std::fmt::Display for AlignmentWarning {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "line {}: main tier has {} alignable words but %mor has {} items",
            self.line_idx, self.main_count, self.mor_count,
        )
    }
}

/// Result of walking a `ChatFile` for morphotag payload collection.
pub struct PayloadCollection {
    /// Utterances that will be sent to the NLP worker.
    pub batch_items: Vec<BatchItemWithPosition>,
    /// Utterances that had zero Mor-alignable content.
    pub not_applicable: Vec<MorOutcome>,
    /// Total number of utterance lines in the file.
    pub total_utterances: usize,
}

/// Walk utterances, build typed payloads, and classify every utterance that had
/// zero Mor-alignable content into a `MorOutcome`.
pub fn collect_payloads(
    chat_file: &talkbank_model::model::ChatFile,
    primary_lang: &talkbank_model::model::LanguageCode,
    declared_languages: &[talkbank_model::model::LanguageCode],
    multilingual_policy: MultilingualPolicy,
) -> PayloadCollection {
    let total_utts = chat_file
        .lines
        .iter()
        .filter(|l| matches!(l, Line::Utterance(_)))
        .count();

    let mut batch_items: Vec<BatchItemWithPosition> = Vec::new();
    let mut not_applicable: Vec<MorOutcome> = Vec::new();
    let mut utt_idx = 0usize;

    for (line_idx, line) in chat_file.lines.iter().enumerate() {
        let utt = match line {
            Line::Utterance(u) => u,
            _ => continue,
        };

        let utterance_lang = utt.main.content.language_code.clone().unwrap_or_else(|| {
            declared_languages
                .first()
                .cloned()
                .unwrap_or_else(|| primary_lang.clone())
        });

        let skip = multilingual_policy.should_skip_non_primary()
            && utt.main.content.language_code.is_some()
            && utt.main.content.language_code.as_ref() != Some(primary_lang);

        let has_mor = utt.dependent_tiers.iter().any(|t| match t {
            talkbank_model::model::DependentTier::Mor(m) => !m.items.is_empty(),
            _ => false,
        });

        if !skip && !has_mor {
            let mut words = Vec::new();
            extract::collect_utterance_content(
                &utt.main.content.content,
                TierDomain::Mor,
                &mut words,
            );

            if !words.is_empty() {
                let terminator_str = utt
                    .main
                    .content
                    .terminator
                    .as_ref()
                    .map(|t| t.to_chat_string())
                    .unwrap_or_else(|| ".".to_string());

                let tier_language = utt
                    .main
                    .content
                    .language_code
                    .as_ref()
                    .or(Some(primary_lang));

                let special_forms: Vec<(
                    Option<talkbank_model::model::FormType>,
                    Option<talkbank_model::validation::LanguageResolution>,
                )> = words
                    .iter()
                    .map(|w| {
                        let resolved_lang = if let Some(ref lang_marker) = w.lang {
                            use talkbank_model::model::Word;
                            use talkbank_model::validation::resolve_word_language;

                            let mut temp_word =
                                Word::new_unchecked(w.text.as_str(), w.text.as_str());
                            temp_word.lang = Some(lang_marker.clone());

                            let (resolved, lang_errors) = resolve_word_language(
                                &temp_word,
                                tier_language,
                                declared_languages,
                            );
                            for err in &lang_errors {
                                tracing::warn!(error = %err, "word language resolution issue");
                            }
                            Some(resolved)
                        } else {
                            None
                        };

                        (w.form_type.clone(), resolved_lang)
                    })
                    .collect();

                let word_texts: Vec<String> =
                    words.iter().map(|w| w.text.as_str().to_string()).collect();

                batch_items.push((
                    line_idx,
                    utt_idx,
                    MorphosyntaxBatchItem {
                        words: word_texts,
                        terminator: terminator_str,
                        special_forms,
                        lang: utterance_lang,
                    },
                    words,
                ));
            } else {
                not_applicable.push(MorOutcome {
                    line_idx,
                    speaker: SpeakerCode::new(utt.main.speaker.as_str()),
                    kind: MorOutcomeKind::NotApplicable {
                        reason: classify_not_applicable(utt),
                    },
                });
            }
        }

        utt_idx += 1;
    }

    PayloadCollection {
        batch_items,
        not_applicable,
        total_utterances: total_utts,
    }
}

/// Extract declared languages from the `@Languages` header, with fallback to
/// `primary_lang` if none were declared.
pub fn declared_languages(
    chat_file: &talkbank_model::model::ChatFile,
    primary_lang: &talkbank_model::model::LanguageCode,
) -> Vec<talkbank_model::model::LanguageCode> {
    if chat_file.languages.is_empty() {
        vec![primary_lang.clone()]
    } else {
        chat_file.languages.0.clone()
    }
}

/// Reset every existing `%mor` and `%gra` tier to an empty body in place,
/// preserving original dependent-tier order.
pub fn clear_morphosyntax(chat_file: &mut talkbank_model::model::ChatFile) {
    for line in chat_file.lines.iter_mut() {
        if let Line::Utterance(utt) = line {
            reset_mor_gra_in_place(utt);
        }
    }
}

fn reset_mor_gra_in_place(utterance: &mut talkbank_model::model::Utterance) {
    use talkbank_model::model::DependentTier;
    use talkbank_model::model::dependent_tier::{GraTier, MorTier};

    for tier in utterance.dependent_tiers.iter_mut() {
        match tier {
            DependentTier::Mor(m) => *m = MorTier::new_mor(Vec::new()),
            DependentTier::Gra(g) => *g = GraTier::new_gra(Vec::new()),
            _ => {}
        }
    }
}

/// Remove any `%mor` or `%gra` tiers that are still empty after the inject pass.
pub fn remove_empty_morphosyntax_placeholders(chat_file: &mut talkbank_model::model::ChatFile) {
    use talkbank_model::model::DependentTier;

    for line in chat_file.lines.iter_mut() {
        if let Line::Utterance(utt) = line {
            utt.dependent_tiers.retain(|tier| match tier {
                DependentTier::Mor(m) => !m.items.is_empty(),
                DependentTier::Gra(g) => !g.relations.is_empty(),
                _ => true,
            });
        }
    }
}

/// Clear `%mor`/`%gra` tiers only from utterances at specific ordinals.
pub fn clear_morphosyntax_selective(
    chat_file: &mut talkbank_model::model::ChatFile,
    utterance_ordinals: &std::collections::HashSet<usize>,
) {
    let mut utt_idx = 0usize;
    for line in chat_file.lines.iter_mut() {
        if let Line::Utterance(utt) = line {
            if utterance_ordinals.contains(&utt_idx) {
                reset_mor_gra_in_place(utt);
            }
            utt_idx += 1;
        }
    }
}

/// Validate that every utterance's `%mor` word count equals the main-tier
/// alignable word count.
pub fn validate_mor_alignment(
    chat_file: &talkbank_model::model::ChatFile,
) -> Vec<AlignmentWarning> {
    use talkbank_model::alignment::helpers::{TierDomain, count_tier_positions};
    use talkbank_model::model::DependentTier;

    let mut warnings = Vec::new();

    for (line_idx, line) in chat_file.lines.iter().enumerate() {
        let utt = match line {
            Line::Utterance(u) => u,
            _ => continue,
        };

        let mor_tier = utt.dependent_tiers.iter().find_map(|t| match t {
            DependentTier::Mor(m) => Some(m),
            _ => None,
        });

        let Some(mor) = mor_tier else {
            continue;
        };

        let main_count = count_tier_positions(&utt.main.content.content, TierDomain::Mor);
        let mor_count = mor.len();

        if main_count != mor_count {
            warnings.push(AlignmentWarning {
                line_idx,
                main_count,
                mor_count,
            });
        }
    }

    warnings
}

/// Join words with spaces and strip parentheses for morphosyntax inference.
pub fn prepare_text(words: &[String]) -> String {
    let joined = words.join(" ");
    joined.replace(['(', ')'], "").trim().to_string()
}

/// The 17 Universal POS tags as defined by Universal Dependencies v2.
#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "UPPERCASE")]
pub enum UniversalPos {
    /// Adjective.
    Adj,
    /// Adposition.
    Adp,
    /// Adverb.
    Adv,
    /// Auxiliary verb.
    Aux,
    /// Coordinating conjunction.
    Cconj,
    /// Determiner.
    Det,
    /// Pronoun.
    Pron,
    /// Common noun.
    Noun,
    /// Proper noun.
    Propn,
    /// Numeral.
    Num,
    /// Particle.
    Part,
    /// Main verb.
    Verb,
    /// Subordinating conjunction.
    Sconj,
    /// Punctuation.
    Punct,
    /// Symbol.
    Sym,
    /// Interjection.
    Intj,
    /// Other / unknown.
    X,
}

impl UniversalPos {
    /// The lowercase CHAT POS category name for this UPOS.
    pub fn to_chat_pos_name(self) -> &'static str {
        match self {
            Self::Adj => "adj",
            Self::Adp => "adp",
            Self::Adv => "adv",
            Self::Aux => "aux",
            Self::Cconj => "cconj",
            Self::Det => "det",
            Self::Intj => "intj",
            Self::Noun => "noun",
            Self::Num => "num",
            Self::Part => "part",
            Self::Pron => "pron",
            Self::Propn => "propn",
            Self::Punct => "punct",
            Self::Sconj => "sconj",
            Self::Sym | Self::X => "x",
            Self::Verb => "verb",
        }
    }

    /// Parse a POS category name into a `UniversalPos`.
    pub fn from_pos_name(name: &str) -> Option<Self> {
        let eq = |s: &str| name.eq_ignore_ascii_case(s);
        if eq("adj") {
            Some(Self::Adj)
        } else if eq("adp") {
            Some(Self::Adp)
        } else if eq("adv") {
            Some(Self::Adv)
        } else if eq("aux") {
            Some(Self::Aux)
        } else if eq("cconj") {
            Some(Self::Cconj)
        } else if eq("det") {
            Some(Self::Det)
        } else if eq("intj") {
            Some(Self::Intj)
        } else if eq("noun") {
            Some(Self::Noun)
        } else if eq("num") {
            Some(Self::Num)
        } else if eq("part") {
            Some(Self::Part)
        } else if eq("pron") {
            Some(Self::Pron)
        } else if eq("propn") {
            Some(Self::Propn)
        } else if eq("punct") {
            Some(Self::Punct)
        } else if eq("sconj") {
            Some(Self::Sconj)
        } else if eq("verb") {
            Some(Self::Verb)
        } else if eq("sym") || eq("x") {
            Some(Self::X)
        } else {
            None
        }
    }
}

/// Universal Dependencies relation label.
///
/// Known values get dedicated variants so call sites compile-check against
/// typos. Unknown relations land in `Other(String)` so round-tripping stays
/// lossless without allocating on the known-value hot path.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum DepRel {
    /// `root` — the sentence-level root.
    Root,
    /// `nsubj` — nominal subject.
    NSubj,
    /// `nsubj:pass` — nominal passive subject.
    NSubjPass,
    /// `obj` — direct object.
    Obj,
    /// `aux` — auxiliary.
    Aux,
    /// `aux:pass` — passive auxiliary.
    AuxPass,
    /// `cop` — copula.
    Cop,
    /// `case` — case-marking word, including possessive `'s`.
    Case,
    /// `nmod:poss` — possessive nominal modifier.
    NmodPoss,
    /// `det` — determiner.
    Det,
    /// `cc` — coordinating conjunction.
    Cc,
    /// `conj` — conjoined element.
    Conj,
    /// `compound` — compound modifier.
    Compound,
    /// `compound:prt` — phrasal verb particle.
    CompoundPrt,
    /// `amod` — adjectival modifier.
    Amod,
    /// `advmod` — adverbial modifier.
    AdvMod,
    /// `punct` — punctuation.
    Punct,
    /// `discourse` — discourse element.
    Discourse,
    /// `mark` — subordinating marker.
    Mark,
    /// `expl` — expletive, such as existential `there`.
    Expl,
    /// Any other UD relation, preserved as its original string.
    Other(String),
}

impl DepRel {
    /// Parse a UD relation string into a typed variant.
    pub fn parse(s: &str) -> Self {
        match s {
            "root" => Self::Root,
            "nsubj" => Self::NSubj,
            "nsubj:pass" => Self::NSubjPass,
            "obj" => Self::Obj,
            "aux" => Self::Aux,
            "aux:pass" => Self::AuxPass,
            "cop" => Self::Cop,
            "case" => Self::Case,
            "nmod:poss" => Self::NmodPoss,
            "det" => Self::Det,
            "cc" => Self::Cc,
            "conj" => Self::Conj,
            "compound" => Self::Compound,
            "compound:prt" => Self::CompoundPrt,
            "amod" => Self::Amod,
            "advmod" => Self::AdvMod,
            "punct" => Self::Punct,
            "discourse" => Self::Discourse,
            "mark" => Self::Mark,
            "expl" => Self::Expl,
            other => Self::Other(other.to_string()),
        }
    }

    /// Serialize back to the UD relation string.
    pub fn as_str(&self) -> &str {
        match self {
            Self::Root => "root",
            Self::NSubj => "nsubj",
            Self::NSubjPass => "nsubj:pass",
            Self::Obj => "obj",
            Self::Aux => "aux",
            Self::AuxPass => "aux:pass",
            Self::Cop => "cop",
            Self::Case => "case",
            Self::NmodPoss => "nmod:poss",
            Self::Det => "det",
            Self::Cc => "cc",
            Self::Conj => "conj",
            Self::Compound => "compound",
            Self::CompoundPrt => "compound:prt",
            Self::Amod => "amod",
            Self::AdvMod => "advmod",
            Self::Punct => "punct",
            Self::Discourse => "discourse",
            Self::Mark => "mark",
            Self::Expl => "expl",
            Self::Other(s) => s.as_str(),
        }
    }
}

/// UD feat string for a finite indicative present 3rd-person-singular form
/// such as a contracted copula or auxiliary.
pub const FINITE_COPULA_PRES_3SG: &str = "Mood=Ind|Number=Sing|Person=3|Tense=Pres|VerbForm=Fin";

/// UD feat string for a present participle, such as `going` or `washing`.
pub const PRESENT_PARTICIPLE: &str = "Tense=Pres|VerbForm=Part";

/// Typed view over UD's `VerbForm` feature values.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum VerbForm {
    /// Finite form.
    Fin,
    /// Participle form.
    Part,
    /// Gerund.
    Ger,
    /// Infinitive form.
    Inf,
    /// Supine form.
    Sup,
    /// Converb.
    Conv,
    /// Verbal noun.
    Vnoun,
    /// Any other UD `VerbForm` value, preserved as written.
    Other(String),
}

impl VerbForm {
    /// Parse a UD `VerbForm` value into a typed variant.
    pub fn parse(s: &str) -> Self {
        match s {
            "Fin" => Self::Fin,
            "Part" => Self::Part,
            "Ger" => Self::Ger,
            "Inf" => Self::Inf,
            "Sup" => Self::Sup,
            "Conv" => Self::Conv,
            "Vnoun" => Self::Vnoun,
            other => Self::Other(other.to_string()),
        }
    }

    /// Serialize this typed value back to its UD string.
    pub fn as_str(&self) -> &str {
        match self {
            Self::Fin => "Fin",
            Self::Part => "Part",
            Self::Ger => "Ger",
            Self::Inf => "Inf",
            Self::Sup => "Sup",
            Self::Conv => "Conv",
            Self::Vnoun => "Vnoun",
            Self::Other(s) => s.as_str(),
        }
    }
}

/// Whether a feat string contains an exact `Key=Value` pair.
pub fn has_key_value(feats: Option<&str>, key: &str, value: &str) -> bool {
    let Some(s) = feats else {
        return false;
    };
    let target = format!("{key}={value}");
    s.split('|').any(|pair| pair == target)
}

/// Whether a feat string declares a finite verb form (`VerbForm=Fin`).
pub fn has_verb_form_fin(feats: Option<&str>) -> bool {
    feats.is_some_and(|s| s.contains("VerbForm=Fin"))
}

/// UD IDs can be single integers (`1`), ranges (`1-2`) for MWTs, or decimals
/// (`1.1`) for empty nodes.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq)]
#[serde(untagged)]
pub enum UdId {
    /// Regular word index.
    Single(usize),
    /// Multi-word token range.
    Range(usize, usize),
    /// Empty-node index.
    Decimal(f64),
}

/// Wrapper for UD fields that may contain either a semantic value or raw
/// punctuation.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq)]
#[serde(untagged)]
pub enum UdPunctable<T> {
    /// A semantic value.
    Value(T),
    /// A punctuation token with no semantic category.
    Punct(String),
}

/// Typed UD token record used by the morphosyntax mapping layer.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq)]
pub struct UdWord {
    /// Word index within the sentence.
    pub id: UdId,
    /// Surface form.
    pub text: String,
    /// Lemma or stem.
    pub lemma: String,
    /// Universal part-of-speech tag.
    pub upos: UdPunctable<UniversalPos>,
    /// Language-specific part-of-speech tag, if present.
    pub xpos: Option<String>,
    /// UD feature bundle, if present.
    pub feats: Option<String>,
    /// Head token index, with `0` meaning root.
    pub head: usize,
    /// Universal dependency relation to the head.
    pub deprel: String,
    /// Enhanced dependency information, if present.
    pub deps: Option<String>,
    /// Miscellaneous annotation, if present.
    pub misc: Option<String>,
}

impl UdWord {
    /// Typed view over this word's dependency relation.
    pub fn dep_rel(&self) -> DepRel {
        DepRel::parse(&self.deprel)
    }

    /// Whether this word carries a finite-verb marker (`VerbForm=Fin`).
    pub fn has_finite_verb_form(&self) -> bool {
        has_verb_form_fin(self.feats.as_deref())
    }

    /// Build a synthetic `UdWord` for language-specific repair passes.
    pub fn synthetic(
        text: impl Into<String>,
        lemma: impl Into<String>,
        upos: UniversalPos,
        feats: Option<&str>,
        head: usize,
        deprel: impl Into<String>,
    ) -> Self {
        Self {
            id: UdId::Single(0),
            text: text.into(),
            lemma: lemma.into(),
            upos: UdPunctable::Value(upos),
            xpos: None,
            feats: feats.map(|f| f.to_string()),
            head,
            deprel: deprel.into(),
            deps: None,
            misc: None,
        }
    }
}

/// A single UD sentence: an ordered sequence of token records.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq)]
pub struct UdSentence {
    /// Ordered token records for this sentence.
    pub words: Vec<UdWord>,
}

/// Top-level UD response for one utterance.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq)]
pub struct UdResponse {
    /// One or more UD sentences produced by the NLP engine.
    pub sentences: Vec<UdSentence>,
}

/// Apply post-parse validation and cleanup to one Stanza-produced UD word.
pub fn validate_and_clean(word: &mut UdWord) {
    if word.deprel.starts_with('<') && word.deprel.ends_with('>') {
        tracing::warn!(
            deprel = %word.deprel,
            text = %word.text,
            "Stanza emitted pad deprel — replacing with 'dep'"
        );
        word.deprel = "dep".to_string();
    }

    if !matches!(word.id, UdId::Range(_, _)) && is_bogus_lemma(&word.text, &word.lemma) {
        tracing::warn!(
            lemma = %word.lemma,
            text = %word.text,
            "Stanza returned bogus lemma — falling back to surface form"
        );
        word.lemma = word.text.clone();
    }
}

/// Detect when Stanza returns a pure-punctuation lemma for a word with letters.
pub fn is_bogus_lemma(text: &str, lemma: &str) -> bool {
    if text == lemma || lemma.is_empty() {
        return false;
    }

    let text_has_letters = text.chars().any(|c| c.is_alphabetic());
    let lemma_all_punct = lemma
        .chars()
        .all(|c| !c.is_alphanumeric() && !c.is_whitespace() && !c.is_control());

    text_has_letters && lemma_all_punct
}

/// Sanitize a string for use in a `%mor` field by replacing structural
/// separators with underscores and stripping whitespace.
pub fn sanitize_mor_text(s: &str) -> String {
    let mut result = s.replace(['|', '#', '-', '&', '$', '~'], "_");
    result.retain(|c| !c.is_whitespace());
    result
}

/// Counts of hint-application outcomes across one `apply_pos_hints` run.
#[derive(Debug, Default, Clone, Copy)]
pub struct HintOutcome {
    /// Total `$POS`-annotated words encountered.
    pub hints_considered: usize,
    /// Hints where Stanza's POS already matched the transcriber's hint.
    pub hints_agreed: usize,
    /// Hints where Stanza's POS was replaced with the transcriber's hint.
    pub hints_overridden: usize,
    /// CLAN tags with no UD UPOS mapping.
    pub hints_unmapped: usize,
    /// Hints on utterances with no `%mor` tier to modify.
    pub hints_skipped_no_mor: usize,
}

/// Walk every utterance in `chat_file` and override `%mor` POS categories where
/// the transcriber's `$POS` hint disagrees with Stanza's output.
pub fn apply_pos_hints(chat_file: &mut talkbank_model::model::ChatFile) -> HintOutcome {
    use talkbank_model::model::content::word::Word;
    use talkbank_model::model::dependent_tier::mor::{MorTier, clan_to_ud_upos};
    use talkbank_model::model::{DependentTier, Utterance};

    fn collect_hints(line: &Line) -> Vec<(usize, String)> {
        let Line::Utterance(utt) = line else {
            return Vec::new();
        };
        let mut hints = Vec::new();
        let mut idx: usize = 0;
        walk_words(
            &utt.main.content.content,
            Some(TierDomain::Mor),
            &mut |leaf: WordItem| {
                let word: Option<&Word> = match leaf {
                    WordItem::Word(w) => Some(w),
                    WordItem::ReplacedWord(rw) => Some(&rw.word),
                    WordItem::Separator(_) => None,
                };
                if let Some(w) = word
                    && let Some(pos) = &w.part_of_speech
                {
                    hints.push((idx, pos.to_string()));
                }
                idx += 1;
            },
        );
        hints
    }

    fn mor_tier_mut(utt: &mut Utterance) -> Option<&mut MorTier> {
        utt.dependent_tiers.iter_mut().find_map(|t| match t {
            DependentTier::Mor(m) => Some(m),
            _ => None,
        })
    }

    enum HintResolution {
        Agreed,
        Overridden,
        Unmapped,
        NoMorItem,
    }

    fn resolve_hint(clan_tag: &str, mor: &mut MorTier, word_idx: usize) -> HintResolution {
        let Some(upos_name) = clan_to_ud_upos(clan_tag) else {
            return HintResolution::Unmapped;
        };
        let Some(hinted) = UniversalPos::from_pos_name(upos_name) else {
            return HintResolution::Unmapped;
        };
        let Some(mor_item) = mor.items.0.get_mut(word_idx) else {
            return HintResolution::NoMorItem;
        };

        let stanza = UniversalPos::from_pos_name(mor_item.main.pos.as_ref());
        if stanza == Some(hinted) {
            return HintResolution::Agreed;
        }
        mor_item.override_main_pos(hinted.to_chat_pos_name());
        HintResolution::Overridden
    }

    let mut outcome = HintOutcome::default();

    for line_idx in 0..chat_file.lines.len() {
        let hints = collect_hints(&chat_file.lines[line_idx]);
        if hints.is_empty() {
            continue;
        }

        let utt = match &mut chat_file.lines[line_idx] {
            Line::Utterance(u) => u,
            _ => continue,
        };
        let Some(mor) = mor_tier_mut(utt) else {
            outcome.hints_considered += hints.len();
            outcome.hints_skipped_no_mor += hints.len();
            continue;
        };

        for (word_idx, clan_tag) in hints {
            outcome.hints_considered += 1;
            match resolve_hint(&clan_tag, mor, word_idx) {
                HintResolution::Agreed => outcome.hints_agreed += 1,
                HintResolution::Overridden => outcome.hints_overridden += 1,
                HintResolution::Unmapped => outcome.hints_unmapped += 1,
                HintResolution::NoMorItem => outcome.hints_skipped_no_mor += 1,
            }
        }
    }

    outcome
}

/// ISO 639-3 codes that have a known Stanza pipeline.
static SUPPORTED_STANZA_CODES: LazyLock<HashSet<&'static str>> = LazyLock::new(|| {
    [
        "eng", "spa", "fra", "deu", "ita", "por", "nld", "cat", "glg", "dan", "swe", "nor", "fin",
        "est", "lav", "lit", "isl", "pol", "ces", "ron", "hun", "bul", "hrv", "slk", "slv", "ukr",
        "rus", "ell", "cym", "gle", "gla", "eus", "mlt", "ara", "heb", "fas", "hin", "urd", "tur",
        "tam", "tel", "tha", "vie", "ind", "zho", "cmn", "yue", "jpn", "kor", "kat", "hye", "afr",
        "lat",
    ]
    .into_iter()
    .collect()
});

/// Check whether a language code is supported by the Stanza worker.
pub fn is_stanza_supported(lang: &talkbank_model::model::LanguageCode) -> bool {
    SUPPORTED_STANZA_CODES.contains(lang.as_ref())
}

/// Sorted list of ISO-639-3 codes the Rust gate considers Stanza-supported.
pub fn supported_iso3_codes() -> &'static [&'static str] {
    static SORTED: LazyLock<Vec<&'static str>> = LazyLock::new(|| {
        let mut v: Vec<&'static str> = SUPPORTED_STANZA_CODES.iter().copied().collect();
        v.sort_unstable();
        v
    });
    &SORTED
}

// ---------------------------------------------------------------------------
// Top-level outcome
// ---------------------------------------------------------------------------

/// One morphotag outcome for one utterance.
///
/// Carries enough information (line index, speaker, kind) to be converted
/// into a [`DecisionRecord`] for `%xalign` tier emission without further
/// context from the caller.
#[derive(Debug, Clone)]
pub struct MorOutcome {
    /// Index into `ChatFile.lines` identifying the utterance.
    pub line_idx: usize,
    /// Speaker code for the affected utterance.
    pub speaker: SpeakerCode,
    /// What happened on this utterance.
    pub kind: MorOutcomeKind,
}

/// The three possible morphotag outcomes per utterance.
#[derive(Debug, Clone)]
pub enum MorOutcomeKind {
    /// The utterance had zero Mor-alignable words under CHAT policy.
    /// No `%mor`/`%gra` was produced, and that is correct behavior —
    /// there is no morphological content to analyze.
    NotApplicable {
        /// Which class of non-linguistic content the utterance held.
        reason: NotApplicableReason,
    },

    /// Stanza returned N tokens for N CHAT words after MWT reassembly;
    /// `%mor`/`%gra` were injected successfully.
    Aligned {
        /// The alignable-word count that was matched on both sides.
        n_words: usize,
    },

    /// The `|stanza_tokens| = |chat_words|` invariant was violated.
    /// Always a bug in extraction, realignment, MWT reassembly, or the
    /// terminator filter. Never silently absorbed.
    MisalignmentBug(MisalignmentDiagnostic),
}

// ---------------------------------------------------------------------------
// NotApplicable — structured reason
// ---------------------------------------------------------------------------

/// Why an utterance had no Mor-alignable content.
///
/// These reasons are mutually exclusive at the point of classification:
/// when an utterance yields zero Mor-alignable words, exactly one of
/// these describes what was there instead. The classifier walks the
/// utterance content once and picks the most specific variant that
/// matches every non-separator word in the utterance.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NotApplicableReason {
    /// The utterance body was empty after parsing (no words, no
    /// separators, no annotations that carry content).
    Empty,
    /// Every word in the utterance is a filler (`&-um`, `&-hmm`, …).
    /// `%mor` does not annotate paralinguistic fillers.
    FillerOnly,
    /// Every word in the utterance is a phonological fragment (`&+le`).
    FragmentOnly,
    /// Every word in the utterance is a nonword (`&~ach`, `&~uh`).
    NonwordOnly,
    /// Every word in the utterance is untranscribed (`xxx`, `yyy`, `www`).
    UntranscribedOnly,
    /// The utterance has words, but all of them are inside retrace
    /// groups (`<...> [/]`, `<...> [//]`), which Mor excludes.
    AllRetraced,
    /// The utterance contains a mix of non-linguistic categories
    /// (e.g. fillers + fragments + untranscribed) where no single
    /// narrower reason above fully describes it.
    MixedNonLinguistic,
}

impl NotApplicableReason {
    /// Short label for `%xalign` tier output: `not_applicable:<label>`.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Empty => "empty",
            Self::FillerOnly => "filler_only",
            Self::FragmentOnly => "fragment_only",
            Self::NonwordOnly => "nonword_only",
            Self::UntranscribedOnly => "untranscribed_only",
            Self::AllRetraced => "all_retraced",
            Self::MixedNonLinguistic => "mixed_nonlinguistic",
        }
    }
}

// ---------------------------------------------------------------------------
// Classification helper: inspect an utterance and decide the NotApplicable
// reason when extraction yields zero Mor-alignable words.
// ---------------------------------------------------------------------------

/// Inspect an utterance whose Mor-domain extraction yielded zero words,
/// and classify the [`NotApplicableReason`] that best describes why.
///
/// The classifier walks the content once, recording which word
/// categories it sees. If every content-bearing element is of one
/// category, that specific variant is returned; otherwise
/// [`NotApplicableReason::MixedNonLinguistic`] is used.
///
/// Callers must only invoke this for utterances where
/// `extract::collect_utterance_content(…, TierDomain::Mor, …)` returned
/// an empty vector; otherwise the classification is meaningless.
pub fn classify_not_applicable(utterance: &Utterance) -> NotApplicableReason {
    // Two walks: domain-free sees all content (including retrace);
    // Mor-domain sees everything except retrace. If Mor sees nothing
    // but domain-free saw words, retrace is the reason. Otherwise
    // classify by category from the Mor walk. `walk_words` doesn't
    // expose retrace context to the closure, so a single-walk fusion
    // would require a custom walker; the per-call cost is tiny (invoked
    // only on empty-payload utterances, ~3K across a 54-file corpus).
    let mut total = ContentCategories::default();
    walk_words(&utterance.main.content.content, None, &mut |item| {
        accumulate(&mut total, item)
    });

    let mut mor_only = ContentCategories::default();
    walk_words(
        &utterance.main.content.content,
        Some(TierDomain::Mor),
        &mut |item| accumulate(&mut mor_only, item),
    );

    if total.total_words == 0 {
        return NotApplicableReason::Empty;
    }
    if mor_only.total_words == 0 {
        // domain-free saw content but Mor saw none → all retrace
        return NotApplicableReason::AllRetraced;
    }
    mor_only.reason_when_nothing_alignable()
}

#[derive(Default, Debug)]
struct ContentCategories {
    /// Total word-like leaves seen (excluding separators).
    total_words: usize,
    filler_count: usize,
    fragment_count: usize,
    nonword_count: usize,
    untranscribed_count: usize,
    /// Linguistic words that would have been Mor-alignable — their
    /// presence means extraction would have returned non-empty, so
    /// the caller's precondition (zero alignable words) was violated.
    linguistic_count: usize,
}

impl ContentCategories {
    /// Pick the narrowest reason that explains "nothing alignable".
    fn reason_when_nothing_alignable(&self) -> NotApplicableReason {
        // If there are any linguistic words, the caller's precondition
        // is violated: extraction would have returned non-empty. Fall
        // through to MixedNonLinguistic conservatively.
        if self.linguistic_count > 0 {
            return NotApplicableReason::MixedNonLinguistic;
        }

        // Mutually-exclusive single-category cases.
        let nonzero_cats = [
            (self.filler_count > 0, NotApplicableReason::FillerOnly),
            (self.fragment_count > 0, NotApplicableReason::FragmentOnly),
            (self.nonword_count > 0, NotApplicableReason::NonwordOnly),
            (
                self.untranscribed_count > 0,
                NotApplicableReason::UntranscribedOnly,
            ),
        ];
        let active: Vec<NotApplicableReason> = nonzero_cats
            .iter()
            .filter_map(|(b, r)| b.then_some(*r))
            .collect();

        match active.as_slice() {
            [] => NotApplicableReason::Empty,
            [single] => *single,
            _ => NotApplicableReason::MixedNonLinguistic,
        }
    }
}

fn accumulate(cats: &mut ContentCategories, item: WordItem<'_>) {
    let word: &talkbank_model::model::Word = match item {
        WordItem::Word(w) => w,
        WordItem::ReplacedWord(rw) => {
            // For Mor, the replacement words are what would be aligned;
            // but for NotApplicable classification we care about what
            // the transcriber actually wrote, so use the original word.
            &rw.word
        }
        WordItem::Separator(_) => {
            // Tag-marker separators (`,`, `„`, `‡`) contribute to the
            // alignable count in the Mor domain. If they're present
            // alongside non-linguistic content, the utterance would
            // extract non-empty, so classify would not run. Ignore here.
            return;
        }
    };

    if word.cleaned_text().is_empty() {
        return;
    }
    cats.total_words += 1;

    if word.untranscribed().is_some() {
        cats.untranscribed_count += 1;
        return;
    }

    match &word.category {
        Some(WordCategory::Filler) => cats.filler_count += 1,
        Some(WordCategory::PhonologicalFragment) => cats.fragment_count += 1,
        Some(WordCategory::Nonword) => cats.nonword_count += 1,
        _ => cats.linguistic_count += 1,
    }
}

// ---------------------------------------------------------------------------
// DecisionRecord conversion
// ---------------------------------------------------------------------------

impl MorOutcome {
    /// Convert this outcome into a [`DecisionRecord`] for `%xalign`
    /// tier emission.
    ///
    /// [`MorOutcomeKind::Aligned`] outcomes return `None` because the
    /// happy path is not review-worthy and would produce a tier entry
    /// per successfully-morphotagged utterance — noise, not signal.
    /// Callers that want to surface aligned counts should aggregate
    /// separately.
    pub fn to_decision_record(&self) -> Option<DecisionRecord> {
        match &self.kind {
            MorOutcomeKind::Aligned { .. } => None,
            MorOutcomeKind::NotApplicable { reason } => Some(DecisionRecord {
                line_idx: self.line_idx,
                speaker: self.speaker.as_str().to_string(),
                strategy: DecisionStrategy::Morphosyntax(MorphosyntaxStrategy::NotApplicable),
                reason: format!("reason={}", reason.as_str()),
                // NotApplicable is correct behavior, not a failure,
                // so it does not require review.
                needs_review: false,
            }),
            MorOutcomeKind::MisalignmentBug(diag) => Some(DecisionRecord {
                line_idx: self.line_idx,
                speaker: self.speaker.as_str().to_string(),
                strategy: DecisionStrategy::Morphosyntax(MorphosyntaxStrategy::MisalignmentBug),
                reason: format!(
                    "class={} expected={} actual={} chat_words={:?} stanza_tokens={:?}",
                    diag.suspected_class.as_str(),
                    diag.expected,
                    diag.actual,
                    diag.chat_words,
                    diag.stanza_tokens_after_mapping,
                ),
                // Misalignment bugs always want human attention —
                // they indicate something the pipeline got wrong.
                needs_review: true,
            }),
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use talkbank_model::alignment::helpers::{MorAlignableWordCount, MorItemCount};
    use talkbank_model::model::ChatFile;
    use talkbank_parser::TreeSitterParser;

    fn parse_chat(text: &str) -> ChatFile {
        let parser = TreeSitterParser::new().expect("parser init");
        parser.parse_chat_file(text).unwrap()
    }

    fn one_utterance(main_tier: &str) -> String {
        format!(
            "@UTF8\n\
             @Begin\n\
             @Languages:\teng\n\
             @Participants:\tCHI Target_Child\n\
             @ID:\teng|test|CHI||female|||Target_Child|||\n\
             *CHI:\t{main_tier}\n\
             @End\n"
        )
    }

    fn first_utterance(chat: &ChatFile) -> &Utterance {
        for line in &chat.lines {
            if let talkbank_model::model::Line::Utterance(u) = line {
                return u;
            }
        }
        panic!("no utterance in chat file")
    }

    #[test]
    fn classify_filler_only() {
        let chat = parse_chat(&one_utterance("&-hmm ."));
        assert_eq!(
            classify_not_applicable(first_utterance(&chat)),
            NotApplicableReason::FillerOnly,
        );
    }

    #[test]
    fn classify_multiple_fillers() {
        let chat = parse_chat(&one_utterance("&-hmm &-hmm ."));
        assert_eq!(
            classify_not_applicable(first_utterance(&chat)),
            NotApplicableReason::FillerOnly,
        );
    }

    #[test]
    fn classify_fragment_only() {
        let chat = parse_chat(&one_utterance("&+le ."));
        assert_eq!(
            classify_not_applicable(first_utterance(&chat)),
            NotApplicableReason::FragmentOnly,
        );
    }

    #[test]
    fn classify_nonword_only() {
        let chat = parse_chat(&one_utterance("&~ach ."));
        assert_eq!(
            classify_not_applicable(first_utterance(&chat)),
            NotApplicableReason::NonwordOnly,
        );
    }

    #[test]
    fn classify_untranscribed_only() {
        let chat = parse_chat(&one_utterance("xxx ."));
        assert_eq!(
            classify_not_applicable(first_utterance(&chat)),
            NotApplicableReason::UntranscribedOnly,
        );
    }

    #[test]
    fn classify_mixed_nonlinguistic() {
        let chat = parse_chat(&one_utterance("&-hmm &+le ."));
        assert_eq!(
            classify_not_applicable(first_utterance(&chat)),
            NotApplicableReason::MixedNonLinguistic,
        );
    }

    #[test]
    fn to_decision_record_aligned_is_none() {
        let outcome = MorOutcome {
            line_idx: 5,
            speaker: SpeakerCode::new("CHI"),
            kind: MorOutcomeKind::Aligned { n_words: 3 },
        };
        assert!(outcome.to_decision_record().is_none());
    }

    #[test]
    fn to_decision_record_not_applicable_has_reason() {
        let outcome = MorOutcome {
            line_idx: 5,
            speaker: SpeakerCode::new("CHI"),
            kind: MorOutcomeKind::NotApplicable {
                reason: NotApplicableReason::FillerOnly,
            },
        };
        let d = outcome.to_decision_record().unwrap();
        assert!(matches!(
            d.strategy,
            crate::decisions::DecisionStrategy::Morphosyntax(
                crate::decisions::MorphosyntaxStrategy::NotApplicable
            )
        ));
        assert_eq!(d.reason, "reason=filler_only");
        assert!(!d.needs_review);
    }

    #[test]
    fn to_decision_record_misalignment_has_diagnostic() {
        let outcome = MorOutcome {
            line_idx: 5,
            speaker: SpeakerCode::new("CHI"),
            kind: MorOutcomeKind::MisalignmentBug(MisalignmentDiagnostic {
                chat_words: vec!["hello".into(), "world".into()],
                stanza_tokens_after_mapping: vec!["hello".into()],
                expected: MorAlignableWordCount::new(2),
                actual: MorItemCount::new(1),
                suspected_class: MisalignmentClass::TerminatorFilterBug,
            }),
        };
        let d = outcome.to_decision_record().unwrap();
        assert!(matches!(
            d.strategy,
            crate::decisions::DecisionStrategy::Morphosyntax(
                crate::decisions::MorphosyntaxStrategy::MisalignmentBug
            )
        ));
        assert!(d.needs_review);
        assert!(d.reason.contains("class=terminator_filter_bug"));
        assert!(d.reason.contains("expected=2"));
        assert!(d.reason.contains("actual=1"));
    }

    #[test]
    fn universal_pos_round_trips_to_chat_name_and_back() {
        for v in [
            UniversalPos::Adj,
            UniversalPos::Adp,
            UniversalPos::Adv,
            UniversalPos::Aux,
            UniversalPos::Cconj,
            UniversalPos::Det,
            UniversalPos::Intj,
            UniversalPos::Noun,
            UniversalPos::Num,
            UniversalPos::Part,
            UniversalPos::Pron,
            UniversalPos::Propn,
            UniversalPos::Punct,
            UniversalPos::Sconj,
            UniversalPos::Verb,
        ] {
            let name = v.to_chat_pos_name();
            assert_eq!(UniversalPos::from_pos_name(name), Some(v));
        }
        assert_eq!(UniversalPos::Sym.to_chat_pos_name(), "x");
        assert_eq!(UniversalPos::X.to_chat_pos_name(), "x");
        assert_eq!(UniversalPos::from_pos_name("x"), Some(UniversalPos::X));
        assert_eq!(UniversalPos::from_pos_name("sym"), Some(UniversalPos::X));
    }

    #[test]
    fn universal_pos_accepts_case_insensitive_names() {
        assert_eq!(
            UniversalPos::from_pos_name("NOUN"),
            Some(UniversalPos::Noun)
        );
        assert_eq!(
            UniversalPos::from_pos_name("noun"),
            Some(UniversalPos::Noun)
        );
        assert_eq!(
            UniversalPos::from_pos_name("Noun"),
            Some(UniversalPos::Noun)
        );
        assert_eq!(UniversalPos::from_pos_name("notreal"), None);
    }

    #[test]
    fn stanza_language_support_matches_expected_examples() {
        use talkbank_model::model::LanguageCode;

        for code in ["eng", "spa", "fra", "deu", "zho", "jpn", "rus", "ara"] {
            assert!(is_stanza_supported(&LanguageCode::new(code)));
        }
        for code in ["que", "jam", "nan", "taq", "und", "xmm", "jav", "wuu"] {
            assert!(!is_stanza_supported(&LanguageCode::new(code)));
        }
        assert!(is_stanza_supported(&LanguageCode::new("yue")));
        assert!(is_stanza_supported(&LanguageCode::new("cmn")));
        for code in ["ben", "kan", "mal", "msa", "tgl", "ltz"] {
            assert!(!is_stanza_supported(&LanguageCode::new(code)));
        }
    }

    #[test]
    fn dep_rel_roundtrips_known_variants() {
        for rel in [
            "root",
            "nsubj",
            "nsubj:pass",
            "obj",
            "aux",
            "aux:pass",
            "cop",
            "case",
            "nmod:poss",
            "det",
            "cc",
            "conj",
            "compound",
            "compound:prt",
            "amod",
            "advmod",
            "punct",
            "discourse",
            "mark",
            "expl",
        ] {
            assert_eq!(DepRel::parse(rel).as_str(), rel);
        }
    }

    #[test]
    fn dep_rel_preserves_unknown_values() {
        let rel = DepRel::parse("orphan");
        assert_eq!(rel, DepRel::Other("orphan".to_string()));
        assert_eq!(rel.as_str(), "orphan");
    }

    #[test]
    fn verb_form_roundtrips() {
        for value in ["Fin", "Part", "Ger", "Inf", "Sup", "Conv", "Vnoun"] {
            assert_eq!(VerbForm::parse(value).as_str(), value);
        }
    }

    #[test]
    fn has_verb_form_fin_matches_expected_cases() {
        assert!(has_verb_form_fin(Some(
            "Mood=Ind|Number=Sing|Person=3|Tense=Pres|VerbForm=Fin"
        )));
        assert!(has_verb_form_fin(Some("VerbForm=Fin")));
        assert!(!has_verb_form_fin(Some("Tense=Pres|VerbForm=Part")));
        assert!(!has_verb_form_fin(None));
    }

    #[test]
    fn has_key_value_matches_exact_pairs() {
        let feats = Some("Mood=Ind|Number=Sing|Person=3|VerbForm=Fin");
        assert!(has_key_value(feats, "VerbForm", "Fin"));
        assert!(has_key_value(feats, "Number", "Sing"));
        assert!(has_key_value(feats, "Person", "3"));
        assert!(!has_key_value(feats, "Tense", "Past"));
        assert!(!has_key_value(None, "VerbForm", "Fin"));
    }

    #[test]
    fn canonical_ud_feat_bundles_are_alphabetical() {
        for bundle in [FINITE_COPULA_PRES_3SG, PRESENT_PARTICIPLE] {
            let keys: Vec<&str> = bundle
                .split('|')
                .map(|pair| pair.split('=').next().expect("feature key"))
                .collect();
            let mut sorted = keys.clone();
            sorted.sort();
            assert_eq!(keys, sorted, "bundle {bundle:?} is not alphabetized");
        }
    }

    #[test]
    fn bogus_lemma_detection_matches_expected_cases() {
        assert!(is_bogus_lemma("hello", "."));
        assert!(is_bogus_lemma("world", ","));
        assert!(is_bogus_lemma("cat", "–"));
        assert!(!is_bogus_lemma("hello", "hello"));
        assert!(!is_bogus_lemma("hello", ""));
        assert!(!is_bogus_lemma(".", "."));
        assert!(!is_bogus_lemma(",", "--"));
        assert!(!is_bogus_lemma("running", "run"));
        assert!(!is_bogus_lemma("cats", "cat"));
    }

    #[test]
    fn validate_and_clean_fixes_pad_deprel_and_bogus_lemma() {
        let mut word = UdWord {
            id: UdId::Single(1),
            text: "hello".to_string(),
            lemma: ".".to_string(),
            upos: UdPunctable::Value(UniversalPos::Intj),
            xpos: None,
            feats: None,
            head: 0,
            deprel: "<pad>".to_string(),
            deps: None,
            misc: None,
        };

        validate_and_clean(&mut word);

        assert_eq!(word.lemma, "hello");
        assert_eq!(word.deprel, "dep");
    }

    #[test]
    fn sanitize_mor_text_replaces_structural_separators() {
        assert_eq!(sanitize_mor_text("foo|bar"), "foo_bar");
        assert_eq!(sanitize_mor_text("a#b-c&d$e~f"), "a_b_c_d_e_f");
    }

    #[test]
    fn sanitize_mor_text_strips_whitespace() {
        assert_eq!(sanitize_mor_text("ふ す"), "ふす");
        assert_eq!(sanitize_mor_text(" hello world "), "helloworld");
        assert_eq!(sanitize_mor_text("a\tb\nc"), "abc");
    }

    #[test]
    fn sanitize_mor_text_handles_combined_issues() {
        assert_eq!(sanitize_mor_text("foo | bar"), "foo_bar");
        assert_eq!(sanitize_mor_text("ふ す#test"), "ふす_test");
    }

    #[test]
    fn sanitize_mor_text_passthroughs_clean_text() {
        assert_eq!(sanitize_mor_text("hello"), "hello");
        assert_eq!(sanitize_mor_text("ふす"), "ふす");
    }

    #[test]
    fn lang2_normalizes_common_codes() {
        assert_eq!(lang2("eng"), "en");
        assert_eq!(lang2("fra"), "fr");
        assert_eq!(lang2("jpn"), "ja");
        assert_eq!(lang2("deu"), "de");
        assert_eq!(lang2("heb"), "he");
        assert_eq!(lang2("en"), "en");
    }
}
