//! The deferred `@s` positions of an utterance, read from the analysis that
//! injection maps.
//!
//! Injection walks each utterance's UD sentence once, after the grammatical
//! invariants rewrote it, and maps that walk to `%mor`; the deferred positions
//! are read from the same walk, aligned to the utterance's words
//! ([`UdAlignment`]), in the space of the `%mor` items injection wrote
//! ([`ItemPlacement`]). So a position and the item it names cannot come from
//! two different analyses or two different item spaces.

use std::ops::Range;

use talkbank_model::alignment::MorItemIndex;
use talkbank_model::model::LanguageCode;
use talkbank_model::validation::LanguageResolution;

use super::deprel::UdDeprel;
use crate::morphosyntax::alignment::{AlignedWord, HeadTarget, UdAlignment, UdAlignmentError};
use crate::morphosyntax::{
    BatchWord, MorphosyntaxBatchItem, TokenItems, UdPunctable, UniversalPos, WordRole,
};
use crate::retokenize::WordTokenMapping;

/// Structural information from the primary model for one @s word.
///
/// This captures what the primary model can know about a word of another
/// language: where it attaches in the host utterance and with which
/// relation, and the relations of the host words that attach to it. Its
/// category is not here: the secondary model owns that. For an `@s`
/// word the primary split into a multi-word token, every field describes
/// the token's representative, the component whose head lies outside it.
///
/// Its fields are private: it is read from an aligned word
/// ([`Self::of`]), never assembled from loose values.
#[derive(Debug, Clone, PartialEq)]
pub struct PrimaryStructuralInfo {
    deprel: UdDeprel,
    head: HeadTarget<MorItemIndex>,
    dependent_deprels: Vec<UdDeprel>,
    head_upos: Option<UniversalPos>,
}

impl PrimaryStructuralInfo {
    /// The primary structure of an aligned word, its head in `head`'s
    /// item space.
    fn of(aligned: &AlignedWord<'_, '_, MorItemIndex>, head: HeadTarget<MorItemIndex>) -> Self {
        Self {
            deprel: UdDeprel::new(&aligned.representative().deprel),
            head,
            dependent_deprels: aligned
                .dependents()
                .map(|dependent| UdDeprel::new(&dependent.deprel))
                .collect(),
            head_upos: aligned.head_word().and_then(|head| upos_of(&head.upos)),
        }
    }

    /// UD dependency relation from the primary model.
    pub fn deprel(&self) -> &UdDeprel {
        &self.deprel
    }

    /// The host `%mor` item the primary model attached this word to.
    pub fn head(&self) -> HeadTarget<MorItemIndex> {
        self.head
    }

    /// Dependency relations of words that attach TO this word.
    pub fn dependent_deprels(&self) -> &[UdDeprel] {
        &self.dependent_deprels
    }

    /// UPOS of the head word (for GRA upgrade decisions).
    pub fn head_upos(&self) -> Option<UniversalPos> {
        self.head_upos
    }

    /// Whether any dependent has deprel "case" (oblique/prepositional phrase).
    pub fn has_case_dependent(&self) -> bool {
        self.dependent_deprels.iter().any(|d| d.base() == "case")
    }

    /// A primary structure for a test fixture.
    #[cfg(test)]
    pub(crate) fn for_test(
        deprel: &str,
        head: HeadTarget<MorItemIndex>,
        head_upos: Option<UniversalPos>,
    ) -> Self {
        Self {
            deprel: UdDeprel::new(deprel),
            head,
            dependent_deprels: Vec::new(),
            head_upos,
        }
    }
}

/// A deferred @s position that needs secondary dispatch.
///
/// Created by injection, from the walk it mapped, for an utterance it
/// injected; consumed by the secondary dispatch phase. Its fields are
/// private and only the deferral builds one.
#[derive(Debug, Clone, PartialEq)]
pub struct L2DeferredPosition {
    line_idx: usize,
    word_idx: MorItemIndex,
    target_lang: LanguageCode,
    primary: PrimaryStructuralInfo,
    word: talkbank_model::ChatCleanedText,
    terminator: talkbank_model::Terminator,
}

impl L2DeferredPosition {
    /// Line index in `ChatFile.lines`.
    pub fn line_idx(&self) -> usize {
        self.line_idx
    }

    /// The `%mor` item the position's word was written as.
    pub fn word_idx(&self) -> MorItemIndex {
        self.word_idx
    }

    /// Resolved target language for secondary dispatch.
    pub fn target_lang(&self) -> &LanguageCode {
        &self.target_lang
    }

    /// Structural info from the primary model.
    pub fn primary(&self) -> &PrimaryStructuralInfo {
        &self.primary
    }

    /// The provenance-sealed text to send to the secondary model: the batch
    /// word the position was aligned from.
    pub fn word(&self) -> &talkbank_model::ChatCleanedText {
        &self.word
    }

    /// Terminator of the utterance this position sits in: an INPUT to the
    /// secondary Stanza model, which changes its analysis when
    /// sentence-final punctuation is absent or wrong.
    pub fn terminator(&self) -> &talkbank_model::Terminator {
        &self.terminator
    }

    /// A position for a test fixture, ending its utterance with a period.
    #[cfg(test)]
    pub(crate) fn for_test(
        line_idx: usize,
        word_idx: MorItemIndex,
        target_lang: &str,
        word: &str,
        primary: PrimaryStructuralInfo,
    ) -> Self {
        Self {
            line_idx,
            word_idx,
            target_lang: LanguageCode::new(target_lang).expect("valid test language code"),
            primary,
            word: crate::parsed_word_text_cleaned(word),
            terminator: talkbank_model::Terminator::Period {
                span: talkbank_model::Span::DUMMY,
            },
        }
    }

    /// The same position in an utterance with another terminator.
    #[cfg(test)]
    pub(crate) fn with_terminator(mut self, terminator: talkbank_model::Terminator) -> Self {
        self.terminator = terminator;
        self
    }
}

/// An utterance whose `@s` words could not be deferred, and why.
///
/// Its `@s` words keep the `L2|xxx` placeholder injection writes.
#[derive(Debug, Clone, PartialEq)]
pub struct UnalignedL2Utterance {
    /// Line index in `ChatFile.lines`.
    pub line_idx: usize,
    /// How many `@s` words fall back.
    pub at_s_words: usize,
    /// Why the primary analysis could not be aligned to the utterance.
    pub error: L2ExtractError,
}

/// Why an utterance's `@s` words could not be deferred.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum L2ExtractError {
    /// The primary UD sentence does not align to the utterance's words.
    #[error("primary analysis does not align: {0}")]
    Alignment(#[from] UdAlignmentError),
    /// With `--retokenize`, the word's items by its text (the mapping that
    /// rebuilds the main tier) are not its items by the analysis.
    #[error(
        "word {word}: its tokens by text {by_text:?} are not its tokens by the analysis {by_analysis:?}"
    )]
    TokenizationDisagrees {
        /// The CHAT word.
        word: usize,
        /// Its `%mor` items by the text mapping.
        by_text: Vec<usize>,
        /// Its `%mor` items by the analysis.
        by_analysis: Vec<usize>,
    },
    /// With `--retokenize`, the `@s` word became more than one `%mor` item,
    /// so no one item can take the secondary analysis.
    #[error("word {word} became {items} %mor items")]
    SplitWord {
        /// The CHAT word.
        word: usize,
        /// How many items it became.
        items: usize,
    },
    /// The analysis names a word the mapping wrote no item for (a broken
    /// invariant of the walk, reported rather than guessed).
    #[error("UD word {id} has no %mor item")]
    UnmappedWord {
        /// The UD id.
        id: usize,
    },
}

/// The deferred `@s` positions of a batch, with the utterances whose `@s`
/// words could not be deferred.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct L2Extraction {
    positions: Vec<L2DeferredPosition>,
    unaligned: Vec<UnalignedL2Utterance>,
}

impl L2Extraction {
    /// The deferred positions, in transcript order (tests only:
    /// production takes them through `into_reported_positions`).
    #[cfg(test)]
    pub(crate) fn positions(&self) -> &[L2DeferredPosition] {
        &self.positions
    }

    /// The utterances whose `@s` words fall back, and why.
    pub fn unaligned(&self) -> &[UnalignedL2Utterance] {
        &self.unaligned
    }

    /// Report every utterance whose `@s` words fall back, then hand over the
    /// positions. The only way to the positions outside a test, so a
    /// failure cannot be dropped on the way.
    pub fn into_reported_positions(self) -> Vec<L2DeferredPosition> {
        for unaligned in &self.unaligned {
            tracing::warn!(
                line_idx = unaligned.line_idx,
                at_s_words = unaligned.at_s_words,
                error = %unaligned.error,
                "L2 extract: @s words fall back to L2|xxx because the primary \
                 analysis cannot be aligned to the utterance's words"
            );
        }
        self.positions
    }

    /// Defer the `@s` words of one injected utterance: read each one's
    /// primary structure from `alignment` (the walk injection mapped), and
    /// name the `%mor` items by `placement`. An utterance with no
    /// dispatchable `@s` word adds nothing; one whose words cannot be placed
    /// is reported in [`Self::unaligned`].
    pub(crate) fn defer_utterance(
        &mut self,
        line_idx: usize,
        item: &MorphosyntaxBatchItem,
        alignment: Result<&UdAlignment<'_, MorItemIndex>, &UdAlignmentError>,
        placement: &ItemPlacement<'_>,
    ) {
        let at_s_words = item
            .words()
            .iter()
            .filter(|word| dispatch_target(word).is_some())
            .count();
        if at_s_words == 0 {
            return;
        }
        let positions = alignment
            .map_err(|error| L2ExtractError::Alignment(error.clone()))
            .and_then(|alignment| placement.positions(line_idx, item, alignment));
        match positions {
            Ok(positions) => self.positions.extend(positions),
            Err(error) => self.unaligned.push(UnalignedL2Utterance {
                line_idx,
                at_s_words,
                error,
            }),
        }
    }
}

/// Where the `%mor` items injection wrote for an utterance lie: the space a
/// deferred position's word and head are named in.
pub(crate) enum ItemPlacement<'p> {
    /// One item per CHAT word, in order (the main tier is kept, or every
    /// word's `%mor` was synthesized): a word's item is its own index, and a
    /// head is the CHAT word it lies in.
    PerChatWord,
    /// The main tier was rebuilt from the model's tokens (`--retokenize`):
    /// one item per UD word, after the MWT lexicon's expansion.
    Retokenized(RetokenizedItems<'p>),
}

/// The items of a retokenized utterance.
pub(crate) struct RetokenizedItems<'p> {
    /// Which items each token of the mapped walk produced.
    pub(crate) mapped: &'p TokenItems,
    /// Per mapped item, the items it became after the MWT lexicon expanded
    /// its token.
    pub(crate) expansion: &'p [Range<usize>],
    /// The mapping by text that rebuilds the main tier.
    pub(crate) by_text: &'p WordTokenMapping,
}

impl ItemPlacement<'_> {
    /// The positions of an utterance's dispatchable `@s` words.
    fn positions(
        &self,
        line_idx: usize,
        item: &MorphosyntaxBatchItem,
        alignment: &UdAlignment<'_, MorItemIndex>,
    ) -> Result<Vec<L2DeferredPosition>, L2ExtractError> {
        // Two sequences of one length (checked by the alignment), walked
        // together rather than indexed.
        alignment
            .words()
            .zip(item.words())
            .filter_map(|(aligned, word)| dispatch_target(word).map(|lang| (aligned, word, lang)))
            .map(|(aligned, word, target_lang)| {
                Ok(L2DeferredPosition {
                    line_idx,
                    word_idx: self.word_item(&aligned)?,
                    target_lang,
                    primary: PrimaryStructuralInfo::of(
                        &aligned,
                        self.head_item(&aligned, alignment)?,
                    ),
                    word: word.text().clone(),
                    terminator: item.terminator.clone(),
                })
            })
            .collect()
    }

    /// The one `%mor` item an aligned CHAT word was written as.
    fn word_item(
        &self,
        aligned: &AlignedWord<'_, '_, MorItemIndex>,
    ) -> Result<MorItemIndex, L2ExtractError> {
        match self {
            Self::PerChatWord => Ok(aligned.index()),
            Self::Retokenized(items) => {
                let word = aligned.index().as_usize();
                let by_analysis: Vec<usize> = items
                    .mapped
                    .items_of(aligned.token_index())
                    .into_iter()
                    .flatten()
                    .flat_map(|mapped| items.expansion.get(mapped).cloned().into_iter().flatten())
                    .collect();
                let by_text = items.by_text.tokens_for_word(word);
                if by_text != by_analysis.as_slice() {
                    return Err(L2ExtractError::TokenizationDisagrees {
                        word,
                        by_text: by_text.to_vec(),
                        by_analysis,
                    });
                }
                match by_analysis.as_slice() {
                    [only] => Ok(MorItemIndex::new(*only)),
                    split => Err(L2ExtractError::SplitWord {
                        word,
                        items: split.len(),
                    }),
                }
            }
        }
    }

    /// The `%mor` item the aligned word's head was written in.
    fn head_item(
        &self,
        aligned: &AlignedWord<'_, '_, MorItemIndex>,
        alignment: &UdAlignment<'_, MorItemIndex>,
    ) -> Result<HeadTarget<MorItemIndex>, L2ExtractError> {
        match (self, aligned.head_id()) {
            (Self::PerChatWord, _) => Ok(aligned.head()),
            (Self::Retokenized(_), None) => Ok(HeadTarget::Root),
            (Self::Retokenized(items), Some(id)) => items
                .mapped
                .item_of_word(alignment.tokens(), id)
                .and_then(|mapped| items.expansion.get(mapped))
                .map(|expanded| HeadTarget::Word(MorItemIndex::new(expanded.start)))
                .ok_or(L2ExtractError::UnmappedWord { id: id.get() }),
        }
    }
}

/// The language a batch word is dispatched to: the first language its
/// code-switch resolved to; none for any other role, or an unresolved one
/// (which keeps `L2|xxx`).
fn dispatch_target(word: &BatchWord) -> Option<LanguageCode> {
    match word.role() {
        WordRole::CodeSwitched(resolution) => resolve_dispatch_lang(resolution),
        WordRole::Analysed | WordRole::SpecialForm(_) => None,
    }
}

/// Resolve the dispatch target language from a `LanguageResolution`.
///
/// Uses `LanguageResolution::languages()` which returns the first language
/// for Single, all languages for Multiple/Ambiguous (we take the first),
/// and empty for Unresolved.
fn resolve_dispatch_lang(resolution: &LanguageResolution) -> Option<LanguageCode> {
    resolution.languages().first().cloned()
}

/// A word's UPOS, or `None` for a raw punctuation tag.
fn upos_of(upos: &UdPunctable<UniversalPos>) -> Option<UniversalPos> {
    match upos {
        UdPunctable::Value(pos) => Some(*pos),
        UdPunctable::Punct(_) => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::morphosyntax::InjectionResult;
    use crate::morphosyntax::payload::collect_payloads;
    use crate::morphosyntax::tests::{one_utterance_in, parse_chat};
    use crate::morphosyntax::types::MultilingualPolicy;
    use crate::morphosyntax::{TokenizationMode, UdResponse, UdSentence, inject_results};

    /// Inject `sentence` as the primary analysis of one `eng, spa`
    /// utterance, in preserve mode.
    fn inject(main_tier: &str, sentence: UdSentence) -> InjectionResult {
        let mut chat = parse_chat(&one_utterance_in("eng, spa", main_tier));
        let eng = LanguageCode::new("eng").expect("valid language code");
        let spa = LanguageCode::new("spa").expect("valid language code");
        let payloads = collect_payloads(
            &chat,
            &eng,
            &[eng.clone(), spa],
            MultilingualPolicy::ProcessAll,
        );
        let parser = talkbank_parser::TreeSitterParser::new().expect("parser");
        inject_results(
            &parser,
            &mut chat,
            payloads.batch_items,
            vec![UdResponse {
                sentences: vec![sentence],
            }],
            &eng,
            TokenizationMode::Preserve,
            &std::collections::BTreeMap::new(),
        )
        .expect("injection")
    }

    /// A deferred position carries the word and the terminator of the
    /// utterance it came from.
    ///
    /// This is where both facts ENTER the L2 path, and it is the only place
    /// they are in hand together with the index that names them: the
    /// position is built from the batch word the alignment pairs it with.
    #[test]
    fn deferred_positions_carry_their_word_and_terminator() {
        let injection = inject(
            "I took the camino@s:spa ?",
            sentence(&[
                ("I", 2, "nsubj"),
                ("took", 0, "root"),
                ("the", 4, "det"),
                ("camino", 2, "obj"),
            ]),
        );

        let position = injection
            .l2
            .positions()
            .first()
            .expect("the @s word must defer to secondary dispatch");
        assert_eq!(
            position.word().as_str(),
            "camino",
            "the position must carry its own word, not one looked up later"
        );
        assert!(
            matches!(
                position.terminator(),
                talkbank_model::Terminator::Question { .. }
            ),
            "the position must carry the utterance's own `?`; got {:?}. A \
             period here tells the secondary Stanza model that a question is \
             a statement, which changes the parse it returns.",
            position.terminator()
        );
    }

    /// A sentence of single words, `(form, head, deprel)` each, ids from 1.
    fn sentence(words: &[(&str, usize, &str)]) -> UdSentence {
        UdSentence {
            words: words
                .iter()
                .enumerate()
                .map(|(position, (form, head, deprel))| {
                    crate::morphosyntax::UdWord::from(crate::morphosyntax::UdWordAnalysis {
                        id: crate::morphosyntax::UdId::Single(position + 1),
                        text: (*form).to_string(),
                        lemma: (*form).to_string(),
                        upos: UdPunctable::Value(UniversalPos::X),
                        xpos: None,
                        feats: None,
                        head: *head,
                        deprel: (*deprel).to_string(),
                        deps: None,
                        misc: None,
                    })
                })
                .collect(),
        }
    }

    /// An utterance whose analysis does not align to its words gets no
    /// `%mor` to splice into, so nothing is deferred: the utterance is
    /// reported as a misalignment. It used to be deferred anyway, with a
    /// structure invented for the missing UD word (`dep`, no POS, head 0),
    /// which the planner then read as "this word is the utterance root".
    #[test]
    fn a_misaligned_utterance_defers_nothing() {
        // Three UD words for four CHAT words.
        let injection = inject(
            "I took the camino@s:spa .",
            sentence(&[("I", 2, "nsubj"), ("took", 0, "root"), ("the", 2, "obj")]),
        );

        assert!(injection.l2.positions().is_empty());
        assert!(injection.l2.unaligned().is_empty());
        assert_eq!(
            injection
                .decisions
                .iter()
                .map(|decision| decision.strategy.strategy_name())
                .collect::<Vec<_>>(),
            ["misalignment_bug"]
        );
    }

    /// An utterance whose every word is code-switched is injected without
    /// the model; when the model's analysis does not align to its words,
    /// its `@s` words are reported and keep `L2|xxx`.
    #[test]
    fn an_unaligned_analysis_of_a_synthesized_utterance_is_reported() {
        // Two UD words for one CHAT word.
        let injection = inject(
            "camino@s:spa .",
            sentence(&[("cami", 0, "root"), ("no", 1, "advmod")]),
        );

        assert!(injection.l2.positions().is_empty());
        assert_eq!(
            injection.l2.unaligned(),
            &[UnalignedL2Utterance {
                line_idx: injection.l2.unaligned()[0].line_idx,
                at_s_words: 1,
                error: L2ExtractError::Alignment(UdAlignmentError::WordCountMismatch {
                    chat_words: 1,
                    ud_tokens: 2,
                }),
            }]
        );
    }

    /// The dispatch target is the first language a resolution names; an
    /// unresolved word has none and keeps `L2|xxx`.
    #[test]
    fn dispatch_target_is_the_first_resolved_language() {
        let eng = LanguageCode::new("eng").expect("valid language code");
        let spa = LanguageCode::new("spa").expect("valid language code");
        assert_eq!(
            resolve_dispatch_lang(&LanguageResolution::Multiple(vec![
                eng.clone(),
                spa.clone()
            ])),
            Some(eng)
        );
        assert_eq!(
            resolve_dispatch_lang(&LanguageResolution::Single(spa.clone())),
            Some(spa)
        );
        assert_eq!(resolve_dispatch_lang(&LanguageResolution::Unresolved), None);
    }
}
