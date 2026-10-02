//! Payload collection and `%mor`/`%gra` mutation passes.
//!
//! [`collect_payloads`] walks a `ChatFile`, builds the per-utterance
//! [`MorphosyntaxBatchItem`] list that gets sent to the Stanza worker,
//! and classifies every utterance with zero alignable content into a
//! [`MorOutcome`]. The mutation helpers ([`clear_morphosyntax`],
//! [`remove_empty_morphosyntax_placeholders`],
//! [`clear_morphosyntax_selective`], [`validate_mor_alignment`]) and
//! the small [`prepare_text`] adapter live here too because they share
//! the same iteration shape over `ChatFile.lines`.

use talkbank_model::WriteChat;
use talkbank_model::alignment::helpers::PositionalDomain;
use talkbank_model::model::{Line, SpeakerCode};

use crate::decisions::LineIdx;
use crate::extract::{self, ExtractedWord};
use talkbank_model::GoverningMarkKind;

use super::outcome::{MorOutcome, MorOutcomeKind, classify_not_applicable};
use super::types::MultilingualPolicy;

// The Stanza placeholder constant moved to
// `talkbank_model::ChatCleanedText::stanza_placeholder()` as the only
// blessed exception to provenance sealing. See its doc comment for
// details and the post-Stanza synthesis recognition logic in
// `morphosyntax/synthesis/`.

/// What one word of a batch item is to the morphosyntax pipeline.
#[derive(Debug, Clone, PartialEq)]
pub enum WordRole {
    /// The model analyses the word in the utterance's language.
    Analysed,
    /// A special form (`@c`, `@b`, ...): the model sees a placeholder, and
    /// the word's `%mor` is synthesized from its form type.
    SpecialForm(talkbank_model::model::FormType),
    /// A code-switched word (its own `@s`, or inside an `[@s:...]` span):
    /// `L2|xxx` from the primary pass, then the secondary model when its
    /// language resolved. Takes precedence over a form type the word also
    /// carries.
    CodeSwitched(talkbank_model::validation::LanguageResolution),
}

impl WordRole {
    /// The role of a word with this form type and governing language: a
    /// code-switch wins over a form type.
    pub fn of(
        form_type: Option<talkbank_model::model::FormType>,
        language: Option<talkbank_model::validation::LanguageResolution>,
    ) -> Self {
        match (language, form_type) {
            (Some(language), _) => Self::CodeSwitched(language),
            (None, Some(form_type)) => Self::SpecialForm(form_type),
            (None, None) => Self::Analysed,
        }
    }
}

/// One word of a batch item: the text the model receives, and the word's
/// role. [`BatchWord::new`] is the only constructor, and it sends a special
/// form's placeholder rather than its text, so the two cannot disagree.
#[derive(Debug, Clone)]
pub struct BatchWord {
    text: talkbank_model::ChatCleanedText,
    role: WordRole,
}

impl BatchWord {
    /// A word with its role. A special form is sent as
    /// `ChatCleanedText::stanza_placeholder()`: the model sees the
    /// placeholder, not the non-word, so the surrounding parse stays clean,
    /// and injection replaces the placeholder's analysis with the form
    /// type's `%mor`.
    pub fn new(text: talkbank_model::ChatCleanedText, role: WordRole) -> Self {
        let text = match role {
            WordRole::SpecialForm(_) => talkbank_model::ChatCleanedText::stanza_placeholder(),
            WordRole::Analysed | WordRole::CodeSwitched(_) => text,
        };
        Self { text, role }
    }

    /// A word the model analyses in the utterance's language.
    pub fn analysed(text: talkbank_model::ChatCleanedText) -> Self {
        Self::new(text, WordRole::Analysed)
    }

    /// The text the model receives.
    pub fn text(&self) -> &talkbank_model::ChatCleanedText {
        &self.text
    }

    /// The word's role.
    pub fn role(&self) -> &WordRole {
        &self.role
    }
}

/// Batch item for morphosyntax NLP processing: one utterance's words, each
/// with its role, its terminator and its language.
#[derive(Clone)]
pub struct MorphosyntaxBatchItem {
    words: Vec<BatchWord>,
    /// Utterance terminator. Serializes to its CHAT surface form (`.`, `?`,
    /// `!`, etc.) over the IPC boundary.
    pub terminator: talkbank_model::Terminator,
    /// Language code for this utterance (ISO 639-3).
    pub lang: talkbank_model::model::LanguageCode,
}

impl MorphosyntaxBatchItem {
    /// An utterance's words, terminator and language.
    pub fn new(
        words: Vec<BatchWord>,
        terminator: talkbank_model::Terminator,
        lang: talkbank_model::model::LanguageCode,
    ) -> Self {
        Self {
            words,
            terminator,
            lang,
        }
    }

    /// The words, each with its role.
    pub fn words(&self) -> &[BatchWord] {
        &self.words
    }
}

// The batch item as the worker receives it: the words' texts, and per word
// its form type and resolved language (`special_forms`), which the worker does
// not read. The JSON Schema `ipc-schema` publishes is this shape, under the
// item's name and with the description below.
/// Batch item for morphosyntax NLP processing.
#[derive(serde::Serialize, schemars::JsonSchema)]
#[schemars(rename = "MorphosyntaxBatchItem")]
struct MorphosyntaxBatchItemWire<'a> {
    /// Word texts for NLP processing. Each word is provenance-sealed
    /// `ChatCleanedText` derived from a parsed `Word` or `Separator`
    /// or, for non-`@s` special-form positions, the blessed
    /// `ChatCleanedText::stanza_placeholder()` constant.
    #[schemars(with = "Vec<String>")]
    words: Vec<&'a talkbank_model::ChatCleanedText>,
    /// Utterance terminator. Typed; serializes to its CHAT surface form
    /// (`.`, `?`, `!`, etc.) over the IPC boundary so the Stanza worker
    /// continues to receive a plain string.
    #[serde(serialize_with = "serialize_terminator_ref_as_chat_str")]
    #[schemars(with = "String")]
    terminator: &'a talkbank_model::Terminator,
    /// Special form and language per word: (form_type, resolved_language).
    #[schemars(with = "Vec<(Option<String>, Option<String>)>")]
    special_forms: Vec<(Option<String>, Option<String>)>,
    /// Language code for this utterance (ISO 639-3).
    #[schemars(with = "String")]
    lang: &'a talkbank_model::model::LanguageCode,
}

impl<'a> MorphosyntaxBatchItemWire<'a> {
    fn of(item: &'a MorphosyntaxBatchItem) -> Self {
        Self {
            words: item.words.iter().map(BatchWord::text).collect(),
            terminator: &item.terminator,
            special_forms: item
                .words
                .iter()
                .map(|word| wire_role(&word.role))
                .collect(),
            lang: &item.lang,
        }
    }
}

/// A word's role on the wire: its form type's CHAT text and its first
/// resolved language.
fn wire_role(role: &WordRole) -> (Option<String>, Option<String>) {
    match role {
        WordRole::Analysed => (None, None),
        WordRole::SpecialForm(form_type) => {
            let mut buf = String::new();
            #[allow(clippy::expect_used)]
            form_type
                .write_chat(&mut buf)
                .expect("writing CHAT to a String should be infallible");
            (Some(buf), None)
        }
        WordRole::CodeSwitched(resolution) => (
            None,
            resolution.languages().first().map(|lc| lc.to_string()),
        ),
    }
}

impl serde::Serialize for MorphosyntaxBatchItem {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        MorphosyntaxBatchItemWire::of(self).serialize(serializer)
    }
}

impl schemars::JsonSchema for MorphosyntaxBatchItem {
    fn schema_name() -> std::borrow::Cow<'static, str> {
        MorphosyntaxBatchItemWire::schema_name()
    }

    fn json_schema(generator: &mut schemars::SchemaGenerator) -> schemars::Schema {
        MorphosyntaxBatchItemWire::json_schema(generator)
    }
}

/// The terminator to hand Stanza for `utt`.
///
/// The terminator is an INPUT to the model, not decoration: Stanza's models
/// are trained on sentences that end in punctuation and change their analysis
/// without one. The Italian model reads `dammela` as an ADJ and declines to
/// MWT-expand it when the terminator is missing, and produces the correct
/// `dare` + me + la when it is present.
///
/// CA-mode utterances may legitimately lack a main-tier terminator, and Stanza
/// needs a sentence-final signal regardless, so a Period is synthesized for
/// that case only. That matches the BA2 default. It is NOT a sentinel: it is
/// the canonical Stanza-input default explicitly chosen for the ambiguous
/// case, not a stand-in for a parse failure.
///
/// One owner, deliberately. The secondary L2 dispatch path needs exactly this
/// answer, and when it had no way to ask for it, it hardcoded a Period for
/// every span instead, which told Stanza that every question was a statement.
pub(crate) fn stanza_input_terminator(
    utt: &talkbank_model::model::Utterance,
) -> talkbank_model::Terminator {
    utt.main
        .content
        .terminator
        .clone()
        .unwrap_or(talkbank_model::Terminator::Period {
            span: talkbank_model::Span::DUMMY,
        })
}

fn serialize_terminator_ref_as_chat_str<S: serde::Serializer>(
    terminator: &&talkbank_model::Terminator,
    serializer: S,
) -> Result<S::Ok, S::Error> {
    serializer.serialize_str(&terminator.to_string())
}

/// A request sends its items, whoever holds them: a collected utterance
/// (`CollectedUtterance`) or a bare item (a secondary-language span).
impl AsRef<MorphosyntaxBatchItem> for MorphosyntaxBatchItem {
    fn as_ref(&self) -> &MorphosyntaxBatchItem {
        self
    }
}

/// One utterance of a file collected for the worker: where it is, the item
/// sent for it, and the words its analysis is injected into.
///
/// Built only by [`collect_payloads`], so its position names an utterance of
/// the file it was collected from.
#[derive(Clone)]
pub struct CollectedUtterance {
    line: LineIdx,
    utt_ordinal: usize,
    item: MorphosyntaxBatchItem,
    words: Vec<ExtractedWord>,
}

impl CollectedUtterance {
    /// The utterance's line in `ChatFile.lines`.
    pub fn line(&self) -> LineIdx {
        self.line
    }

    /// The utterance's 0-based ordinal among the file's utterances.
    pub fn utt_ordinal(&self) -> usize {
        self.utt_ordinal
    }

    /// The item sent to the worker.
    pub fn item(&self) -> &MorphosyntaxBatchItem {
        &self.item
    }

    /// The item alone, for a caller that sends it and injects nothing.
    pub fn into_item(self) -> MorphosyntaxBatchItem {
        self.item
    }

    /// The utterance's words, as extracted from its main tier.
    pub fn words(&self) -> &[ExtractedWord] {
        &self.words
    }
}

impl AsRef<MorphosyntaxBatchItem> for CollectedUtterance {
    fn as_ref(&self) -> &MorphosyntaxBatchItem {
        &self.item
    }
}

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
    pub batch_items: Vec<CollectedUtterance>,
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

    let mut batch_items: Vec<CollectedUtterance> = Vec::new();
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

        let has_mor = utt.dependent_tiers.iter().any(|t| match &t.tier {
            talkbank_model::model::DependentTier::Mor(m) => !m.items().is_empty(),
            _ => false,
        });

        if !skip && !has_mor {
            let mut words = Vec::new();
            extract::collect_utterance_content(
                &utt.main.content.content,
                PositionalDomain::Mor,
                &mut words,
            );

            if !words.is_empty() {
                let terminator_typed = stanza_input_terminator(utt);

                // Resolution must use the same language as dispatch.
                // Pre-2026-05-02 this had a separate `or(Some(primary_lang))`
                // fallback that skipped the `declared_languages.first()`
                // step used by `utterance_lang` above. For a Catalan/Spanish
                // file with no per-utterance precoding and a job-level
                // `primary_lang="eng"` (fabricated by the dispatch layer
                // when `WorkerLanguage::Unspecified`), the two paths
                // disagreed: dispatch ran as `cat`, resolution ran as
                // `eng`. The mismatch produced an `Unresolved` (after
                // today's resolver rule-6d fix) for every `@s` position
                // and a fabricated `Single("eng")` before the fix
                // which is the dona@s observed bug.
                let tier_language = Some(&utterance_lang);

                let batch_words: Vec<BatchWord> = words
                    .iter()
                    .map(|w| {
                        // The GOVERNING mark, which is the word's own `@s` if it
                        // has one and otherwise any enclosing `<...> [@s:hin]`
                        // span. Reading `w.lang` alone (a word's own marker) was
                        // the bug: every unmarked word inside a Hindi span looked
                        // unlanguaged, so it fell out of L2 dispatch and was
                        // morphotagged against the tier language instead.
                        //
                        // No throwaway `Word` any more either. This used to build
                        // a `Word::new_unchecked` purely to satisfy a resolver
                        // signature that wanted a `&Word` for its span. As of
                        // chatter 0.16.0 the mark CARRIES the word's span, so
                        // there is no span to pass and no way to pair a mark
                        // with a different word's position: `resolve_language`
                        // takes only the language context.
                        let resolved_lang = match w.language_kind() {
                            GoverningMarkKind::Utterance => None,
                            // Explicit arms, not a catch-all binding: a fourth
                            // variant added in chatter must fail to compile here
                            // rather than silently routing into this branch.
                            GoverningMarkKind::Own | GoverningMarkKind::Span => {
                                let outcome = w.resolve_language(tier_language, declared_languages);
                                for err in &outcome.diagnostics {
                                    tracing::warn!(
                                        error = %err,
                                        "word language resolution issue"
                                    );
                                }
                                Some(outcome.resolution)
                            }
                        };

                        BatchWord::new(
                            w.text.clone(),
                            WordRole::of(w.form_type.clone(), resolved_lang),
                        )
                    })
                    .collect();

                batch_items.push(CollectedUtterance {
                    line: LineIdx::new(line_idx),
                    utt_ordinal: utt_idx,
                    item: MorphosyntaxBatchItem::new(batch_words, terminator_typed, utterance_lang),
                    words,
                });
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
        chat_file.languages.to_vec()
    }
}

/// Reset every existing `%mor` and `%gra` tier to an empty body in place,
/// preserving original dependent-tier order.
pub fn clear_morphosyntax(chat_file: &mut talkbank_model::model::ChatFile) {
    for line in chat_file.lines.as_mut_slice().iter_mut() {
        if let Line::Utterance(utt) = line {
            reset_mor_gra_in_place(utt);
        }
    }
}

fn reset_mor_gra_in_place(utterance: &mut talkbank_model::model::Utterance) {
    use talkbank_model::model::dependent_tier::{GraTier, MorTier};
    use talkbank_model::model::{DependentTier, Terminator};

    // Position-preserving so subsequent `replace_or_add_tier` finds
    // the same variant slot; a `retain`-removal would append the
    // re-injected tier at the end and reorder dependent tiers.
    for entry in utterance.dependent_tiers.iter_mut() {
        // Computed before assigning so the discriminant read and the write do
        // not overlap as borrows. Assigning `entry.tier` rather than the whole
        // entry also keeps the line's separator provenance, which is what
        // "position-preserving" now means for a DependentTierEntry.
        let emptied = match &entry.tier {
            DependentTier::Mor(_) => Some(DependentTier::Mor(MorTier::new_mor(
                Vec::new(),
                Terminator::Period {
                    span: talkbank_model::Span::DUMMY,
                },
            ))),
            DependentTier::Gra(_) => Some(DependentTier::Gra(GraTier::new_gra(Vec::new()))),
            _ => None,
        };
        if let Some(tier) = emptied {
            entry.tier = tier;
        }
    }
}

/// Remove any `%mor` or `%gra` tiers that are still empty after the inject pass.
pub fn remove_empty_morphosyntax_placeholders(chat_file: &mut talkbank_model::model::ChatFile) {
    use talkbank_model::model::DependentTier;

    for line in chat_file.lines.as_mut_slice().iter_mut() {
        if let Line::Utterance(utt) = line {
            utt.dependent_tiers.retain(|tier| match &tier.tier {
                DependentTier::Mor(m) => !m.items().is_empty(),
                DependentTier::Gra(g) => !g.relations().is_empty(),
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
    for line in chat_file.lines.as_mut_slice().iter_mut() {
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
    use talkbank_model::alignment::helpers::count_tier_positions;
    use talkbank_model::model::DependentTier;

    let mut warnings = Vec::new();

    for (line_idx, line) in chat_file.lines.iter().enumerate() {
        let utt = match line {
            Line::Utterance(u) => u,
            _ => continue,
        };

        let mor_tier = utt.dependent_tiers.iter().find_map(|t| match &t.tier {
            DependentTier::Mor(m) => Some(m),
            _ => None,
        });

        let Some(mor) = mor_tier else {
            continue;
        };

        let main_count = count_tier_positions(&utt.main.content.content, PositionalDomain::Mor);
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
