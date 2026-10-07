//! Result injection, clearing, and alignment validation.

use std::ops::Range;

use talkbank_model::alignment::MorItemIndex;
use talkbank_model::model::dependent_tier::GrammaticalRelation;
use talkbank_model::model::dependent_tier::mor::{Mor, MorWord};
use talkbank_model::model::{GrammaticalRelationType, LanguageCode, Line};

use super::alignment::{UdAlignment, UdAlignmentError, UdTokens};
use super::l2::{ItemPlacement, L2Extraction, RetokenizedItems};
use super::responses::{AdmittedUdResponse, BoundMorphosyntaxResponses, ResponseAdmissionError};
use super::synthesis::synthesize_special_form_mor;
use super::{
    BatchWord, CollectedUtterance, ItemLayout, MappedTokens, MappingContext, MappingError,
    MatchedMorphosyntaxResponses, MisalignmentClass, MisalignmentDiagnostic, MwtDict,
    TokenizationMode, UdResponse, WordRole, apply_grammatical_invariants, map_tokens,
};
use crate::decisions::{DecisionRecord, DecisionStrategy, MorphosyntaxStrategy};

/// Context supplied by the caller to [`enrich_diagnostic`] so the
/// suspected misalignment class can be inferred more precisely.
enum RetokenizationContext {
    /// Non-retokenize path: Stanza was told to realign to CHAT
    /// boundaries. A mismatch here means realignment did not hold.
    Preserve,
    /// `StanzaRetokenize` mode was used; CHAT was rewritten to match
    /// Stanza's own tokenization, so a mismatch suggests a retokenize
    /// or MWT-expansion bug rather than a realignment issue.
    StanzaRetokenize,
}

/// Enrich an inner [`MisalignmentDiagnostic`] with caller-held context
/// that the inner validator didn't have access to: the Stanza tokens
/// that were sent, and a best-effort suspected-class classification.
fn enrich_diagnostic(
    mut diag: MisalignmentDiagnostic,
    stanza_tokens: &[String],
    context: RetokenizationContext,
) -> MisalignmentDiagnostic {
    if diag.stanza_tokens_after_mapping.is_empty() && !stanza_tokens.is_empty() {
        diag.stanza_tokens_after_mapping = stanza_tokens.to_vec();
    }

    // Infer the suspected class when the inner validator left it
    // `Unknown`. This is a heuristic, the real diagnosis requires a
    // developer looking at the logs, but it points them at the right
    // stage to investigate first.
    if matches!(diag.suspected_class, MisalignmentClass::Unknown) {
        diag.suspected_class = match context {
            RetokenizationContext::StanzaRetokenize => {
                // In retokenize mode the main tier is rewritten to match
                // Stanza's tokenization; a mismatch here usually means
                // the rebuild step dropped or duplicated tokens.
                MisalignmentClass::MwtReassemblyBug
            }
            RetokenizationContext::Preserve => {
                // Non-retokenize: Stanza was told to realign. A mismatch
                // implies either the realignment context wasn't set for
                // this call (RealignmentSkipped), a terminator-filter
                // regression (TerminatorFilterBug), or an MWT-reassembly
                // bug. Without more signal, Unknown is honest.
                MisalignmentClass::Unknown
            }
        };
    }

    diag
}

/// Completed injection of every requested utterance. Only the injector can
/// construct this value; incomplete analysis returns an [`InjectionError`].
///
/// Completion cannot be fabricated from empty diagnostics and deferred data:
/// ```compile_fail
/// use batchalign_transform::morphosyntax::InjectionResult;
/// let fabricated = InjectionResult {
///     retokenization_traces: Vec::new(),
///     l2: Default::default(),
/// };
/// ```
#[derive(Debug, Clone)]
pub struct InjectionResult {
    /// Per-utterance retokenization traces for debugging.
    pub retokenization_traces: Vec<RetokenizationInfo>,
    /// The `@s` positions of the injected utterances, read from the same
    /// walk of the same (invariant-rewritten) analysis that was mapped to
    /// `%mor`, and named in the item space injection wrote.
    l2: L2Extraction,
}

impl InjectionResult {
    /// Inspect the secondary-language evidence produced by completed injection.
    /// ```
    /// use batchalign_transform::morphosyntax::InjectionResult;
    /// fn inspect(completed: &InjectionResult) {
    ///     let _ = completed.l2();
    /// }
    /// ```
    pub fn l2(&self) -> &L2Extraction {
        &self.l2
    }

    /// Consume completed injection into its trace and secondary-language data.
    pub fn into_parts(self) -> (Vec<RetokenizationInfo>, L2Extraction) {
        (self.retokenization_traces, self.l2)
    }
}

/// Mutable progress is not a completed injection result. Only a successful
/// traversal of every requested utterance promotes its data into completion.
#[derive(Default)]
struct InjectionProgress {
    retokenization_traces: Vec<RetokenizationInfo>,
    l2: L2Extraction,
}

/// A batch cannot establish completed morphology for its destination.
#[derive(Debug, thiserror::Error)]
pub enum InjectionError {
    /// The worker did not return one response per submitted utterance.
    #[error(transparent)]
    ResponseCount(#[from] super::responses::ResponseCountMismatch),
    /// One utterance response contains multiple sentences, none safely selected.
    #[error("worker response {index}: {error}")]
    SentenceCount {
        /// Zero-based submitted payload position.
        index: usize,
        /// Retained cardinality failure from the model boundary.
        #[source]
        error: super::responses::UnexpectedSentenceCount,
    },
    /// A collected destination no longer identifies an utterance.
    #[error("Line at index {index} is no longer an utterance")]
    InvalidPosition {
        /// Collected line that no longer names an utterance.
        index: usize,
    },
    /// Model analysis or injection failed for required lexical content.
    #[error("Incomplete morphology: {}", .0.evidence_summary())]
    Incomplete(Box<DecisionRecord>),
}

impl From<ResponseAdmissionError> for InjectionError {
    fn from(error: ResponseAdmissionError) -> Self {
        match error {
            ResponseAdmissionError::ResponseCount(error) => Self::ResponseCount(error),
            ResponseAdmissionError::SentenceCount { index, error } => {
                Self::SentenceCount { index, error }
            }
        }
    }
}

type UtteranceCompletion = Result<(), Box<DecisionRecord>>;

/// Retokenization info collected during injection, for trace visualization.
#[derive(Debug, Clone)]
pub struct RetokenizationInfo {
    /// Utterance ordinal (0-based, among processed utterances).
    pub utterance_ordinal: usize,
    /// Original CHAT words.
    pub original_words: Vec<String>,
    /// Stanza tokens after retokenization.
    pub stanza_tokens: Vec<String>,
    /// Word→token index mapping: `mapping[word_idx]` = list of token indices.
    pub mapping: Vec<Vec<usize>>,
    /// Whether the fallback (length-proportional) mapping was used.
    pub used_fallback: bool,
}

// ---------------------------------------------------------------------------
// Result injection (from NLP callback)
// ---------------------------------------------------------------------------

/// Deprel label written to `%gra` for non-analyzable special-form positions
/// whose head is non-zero (the form-marker token is a dependent of some
/// other chunk). UD `dep` = "no specific role applies." The head=0 case must
/// keep `ROOT` instead: the joint invariant `(head == 0) ⟺ (deprel ==
/// "ROOT")` is enforced by the validator (E722/E723).
const DEP_RELATION_LABEL: &str = "DEP";
/// Deprel label written when the form-marker token is the syntactic root of
/// the utterance (head=0), as the joint root invariant requires.
const ROOT_RELATION_LABEL: &str = "ROOT";
/// Deprel label written for the terminator's relation in synthetic (no
/// model) `%gra`.
const PUNCT_RELATION_LABEL: &str = "PUNCT";

/// Every word's `%mor` from its role, when no word is the model's to
/// analyse: special forms from their form type, code-switched words as the
/// `L2|xxx` placeholder the L2 splice fills. `None` for an utterance with a
/// word the model analyses (or no words).
///
/// The model's analysis is bypassed for such an utterance because, for
/// all-placeholder input (`xbxxx .`), Stanza may return no sentence, or
/// split `xbxxx` into `xb` + `xxx` so the item count no longer matches the
/// word count; the morphology of these words is fully determined by their
/// roles.
fn synthesize_all_special_forms(
    item: &super::payload::MorphosyntaxBatchItem,
    words: &[crate::extract::ExtractedWord],
) -> Option<Vec<Mor>> {
    let mors: Vec<Mor> = item
        .words()
        .iter()
        .zip(words.iter())
        .map(|(batch_word, word)| match batch_word.role() {
            WordRole::CodeSwitched(_) => Some(Mor::new(MorWord::l2_placeholder())),
            WordRole::SpecialForm(form_type) => {
                Some(synthesize_special_form_mor(form_type, word.text.as_str()))
            }
            WordRole::Analysed => None,
        })
        .collect::<Option<_>>()?;
    (!mors.is_empty()).then_some(mors)
}

/// The star-shaped `%gra` of a synthesized utterance: the first chunk is the
/// root, every other chunk depends on it, and the terminator on chunk 1.
fn star_gras(mors: &[Mor]) -> Vec<GrammaticalRelation> {
    let chunk_count: usize = mors.iter().map(Mor::count_chunks).sum();
    let mut gras: Vec<GrammaticalRelation> = (1..=chunk_count)
        .map(|chunk| match chunk {
            1 => GrammaticalRelation::new(chunk, 0, ROOT_RELATION_LABEL),
            _ => GrammaticalRelation::new(chunk, 1, DEP_RELATION_LABEL),
        })
        .collect();
    gras.push(GrammaticalRelation::new(
        chunk_count + 1,
        1,
        PUNCT_RELATION_LABEL,
    ));
    gras
}

/// Relabel the items of special-form and code-switched words, in place.
///
/// `items_of` gives, for each CHAT word, the `%mor` items it was written as;
/// a word's relation is found at its items' own chunks (the chunk index of an
/// item counts the chunks of every item before it), never by the item's
/// position, which differs from its chunk after a contraction. A
/// code-switched word's items become the `L2|xxx` placeholder (its chunks
/// keep their relations for the L2 splice); a special form's item is
/// synthesized from its form type, and its relation becomes `DEP`, or stays
/// `ROOT` where the word is the root.
fn relabel_special_forms<'w>(
    mors: &mut [Mor],
    gras: &mut [GrammaticalRelation],
    words: impl Iterator<Item = (&'w BatchWord, &'w crate::extract::ExtractedWord, Vec<usize>)>,
) {
    // The chunk each item starts at, before any item is replaced.
    let first_chunks: Vec<usize> = mors
        .iter()
        .scan(0, |chunks, mor| {
            let first = *chunks;
            *chunks += mor.count_chunks();
            Some(first)
        })
        .collect();
    for (batch_word, word, items) in words {
        for item in items {
            let (Some(mor), Some(&first_chunk)) = (mors.get_mut(item), first_chunks.get(item))
            else {
                continue;
            };
            match batch_word.role() {
                WordRole::Analysed => {}
                WordRole::CodeSwitched(_) => mor.main.reset_to_l2_placeholder(),
                WordRole::SpecialForm(form_type) => {
                    *mor = synthesize_special_form_mor(form_type, word.text.as_str());
                    if let Some(gra) = gras.get_mut(first_chunk) {
                        let label = match gra.head {
                            0 => ROOT_RELATION_LABEL,
                            _ => DEP_RELATION_LABEL,
                        };
                        gra.relation = GrammaticalRelationType::new(label);
                    }
                }
            }
        }
    }
}

/// Inject UD NLP results back into utterances.
///
/// Applies special form overrides (@c -> c|, @s -> L2|xxx) and
/// optionally retokenizes the main tier based on the [`TokenizationMode`].
///
/// # Errors
///
/// Returns `Err` without mutation when payload/response counts differ, an
/// utterance response contains multiple sentences, or a
/// destination index is missing or no longer an utterance. Per-utterance
/// retokenization and linguistic diagnostics retain their existing policy.
pub fn inject_results(
    parser: &talkbank_parser::TreeSitterParser,
    chat_file: &mut talkbank_model::model::ChatFile,
    batch_items: Vec<CollectedUtterance>,
    responses: Vec<UdResponse>,
    _lang: &LanguageCode,
    tokenization_mode: TokenizationMode,
    mwt: &MwtDict,
) -> Result<InjectionResult, InjectionError> {
    MatchedMorphosyntaxResponses::new(batch_items, responses)?.inject(
        parser,
        chat_file,
        tokenization_mode,
        mwt,
    )
}

impl MatchedMorphosyntaxResponses {
    /// Inject an admitted response batch, consuming its payload/response ownership.
    ///
    /// Failure to cover any requested utterance refuses the batch. The caller
    /// must discard the partially modified destination on error, not publish it.
    pub fn inject(
        self,
        parser: &talkbank_parser::TreeSitterParser,
        chat_file: &mut talkbank_model::model::ChatFile,
        tokenization_mode: TokenizationMode,
        mwt: &MwtDict,
    ) -> Result<InjectionResult, InjectionError> {
        self.bind(chat_file)
            .map_err(|error| InjectionError::InvalidPosition { index: error.index })?
            .inject(parser, tokenization_mode, mwt)
    }
}

impl BoundMorphosyntaxResponses<'_> {
    fn inject(
        self,
        parser: &talkbank_parser::TreeSitterParser,
        tokenization_mode: TokenizationMode,
        mwt: &MwtDict,
    ) -> Result<InjectionResult, InjectionError> {
        let (chat_file, batch) = self.into_parts();
        let mut result = InjectionProgress::default();
        for (ud_resp, collected) in batch.into_pairs() {
            let line_idx = collected.line().raw();
            let utt = match &mut chat_file.lines.as_mut_slice()[line_idx] {
                Line::Utterance(u) => u,
                _ => {
                    return Err(InjectionError::InvalidPosition { index: line_idx });
                }
            };
            UtteranceInjection {
                parser,
                utt,
                line_idx,
                utt_ordinal: collected.utt_ordinal(),
                item: collected.item(),
                words: collected.words(),
                result: &mut result,
            }
            .inject(ud_resp, tokenization_mode, mwt)
            .map_err(InjectionError::Incomplete)?;
        }
        Ok(InjectionResult {
            retokenization_traces: result.retokenization_traces,
            l2: result.l2,
        })
    }
}

/// One utterance's injection: its destination, its batch item and words,
/// and where its outcomes go.
struct UtteranceInjection<'a, 'u> {
    parser: &'a talkbank_parser::TreeSitterParser,
    utt: &'u mut talkbank_model::model::Utterance,
    line_idx: usize,
    utt_ordinal: usize,
    item: &'a super::payload::MorphosyntaxBatchItem,
    words: &'a [crate::extract::ExtractedWord],
    result: &'a mut InjectionProgress,
}

impl UtteranceInjection<'_, '_> {
    /// Inject one response: rewrite the analysis by the grammatical
    /// invariants once, walk it once, map the walk, place the special forms
    /// through the alignment, inject, and defer the `@s` words from the same
    /// walk.
    fn inject(
        self,
        ud_resp: AdmittedUdResponse,
        mode: TokenizationMode,
        mwt: &MwtDict,
    ) -> UtteranceCompletion {
        let ctx = MappingContext {
            lang: self.item.lang.clone(),
        };
        // Apply grammatical-invariant rewrites to correct known Stanza
        // defects (e.g., English copula 's + progressive misanalyzed as
        // possessive-gerund). The rewrite returns the input unchanged when
        // no rule fires. See `crate::morphosyntax` for the current rule set
        // and `book/src/batchalign/reference/stanza-limitations.md` for the
        // versioned defect registry. This is the one analysis both the
        // `%mor` mapping and the `@s` deferral read.
        // The analysis as the model returned it, beside its rewrite: the
        // retokenizing path rebuilds the main tier from the model's own
        // tokens, never from words a rewrite added.
        let analysed = ud_resp.sentence().map(|raw| {
            // What the transcriber wrote that Stanza never saw: the pauses,
            // aligned to the payload's words. Read only by a chain that
            // uses it.
            let content = &self.utt.main.content.content;
            let words = self.words;
            let rescued = apply_grammatical_invariants(raw, &ctx, || {
                super::evidence::UtteranceEvidence::from_utterance(content, words)
            });
            (raw, rescued)
        });

        // Utterances whose every word is a special form or a code-switch
        // do not need the model's analysis (see `synthesize_all_special_forms`).
        if let Some(mors) = synthesize_all_special_forms(self.item, self.words) {
            let gras = star_gras(&mors);
            let item_count = mors.len();
            return match crate::inject::inject_morphosyntax(
                self.utt,
                mors,
                self.item.terminator.clone(),
                gras,
            ) {
                Err(diag) => self.misalignment(diag, &[], RetokenizationContext::Preserve),
                Ok(()) => {
                    // The `@s` words still take their primary structure from
                    // the model's analysis, when there is one.
                    if let Some((_, rescued)) = &analysed {
                        let alignment = UdAlignment::<MorItemIndex>::new(rescued, item_count);
                        self.result.l2.defer_utterance(
                            self.line_idx,
                            self.item,
                            alignment.as_ref(),
                            &ItemPlacement::PerChatWord,
                        );
                    }
                    Ok(())
                }
            };
        }

        let Some((raw, rescued)) = analysed else {
            // The model returned nothing for an utterance with words it
            // must analyse: record it and leave the utterance untouched.
            return Err(Box::new(DecisionRecord::new_and_trace(
                self.line_idx,
                self.utt.main.speaker.as_str().to_string(),
                DecisionStrategy::Morphosyntax(MorphosyntaxStrategy::NlpNoSentences),
                "stanza_returned_empty_response".into(),
                true,
            )));
        };

        // The one walk. A malformed analysis, or one that does not map, is
        // recorded and the utterance left untouched: Stanza occasionally
        // returns structurally invalid UD (e.g. multiple heads=0).
        let layout = match mode {
            // One item per CHAT word: a contraction's components join as
            // clitics (`verb|go~part|to`).
            TokenizationMode::Preserve => ItemLayout::PerToken,
            // One item per model word: the main tier is rewritten to them.
            TokenizationMode::StanzaRetokenize => ItemLayout::PerWord,
        };
        let walked = UdTokens::walk(&rescued)
            .map_err(MappingError::from)
            .and_then(|tokens| map_tokens(&tokens, &ctx, layout).map(|mapped| (tokens, mapped)));
        let (tokens, mapped) = match walked {
            Ok(walked) => walked,
            Err(e) => {
                return Err(Box::new(DecisionRecord::new_and_trace(
                    self.line_idx,
                    self.utt.main.speaker.as_str().to_string(),
                    DecisionStrategy::Morphosyntax(MorphosyntaxStrategy::MappingFailed),
                    format!("ud_to_chat_error={e}"),
                    true,
                )));
            }
        };
        match mode {
            TokenizationMode::Preserve => {
                let alignment = tokens.align::<MorItemIndex>(self.words.len());
                self.inject_preserved(mapped, alignment)
            }
            TokenizationMode::StanzaRetokenize => {
                let surface = match surface_tokens(raw, self.item, self.words) {
                    Ok(surface) => surface,
                    Err(SurfaceError::Walk(e)) => {
                        return Err(Box::new(DecisionRecord::new_and_trace(
                            self.line_idx,
                            self.utt.main.speaker.as_str().to_string(),
                            DecisionStrategy::Morphosyntax(MorphosyntaxStrategy::MappingFailed),
                            format!("ud_to_chat_error={e}"),
                            true,
                        )));
                    }
                    Err(SurfaceError::Unplaced(refused)) => {
                        let tokens = refused.stanza_tokens_after_mapping.clone();
                        return self.misalignment(
                            refused,
                            &tokens,
                            RetokenizationContext::StanzaRetokenize,
                        );
                    }
                };
                let alignment = tokens.align::<MorItemIndex>(self.words.len());
                self.inject_retokenized(surface, mapped, alignment, mwt)
            }
        }
    }

    /// Preserve mode: one item per CHAT word, so a word's item is its own
    /// index once the walk aligns to the words.
    fn inject_preserved(
        self,
        mapped: MappedTokens,
        alignment: Result<UdAlignment<'_, MorItemIndex>, UdAlignmentError>,
    ) -> UtteranceCompletion {
        let (mut mors, mut gras) = mapped.into_parts();
        // Without an alignment the item count differs from the word count,
        // and injection below reports the utterance; nothing can be placed.
        if alignment.is_ok() {
            relabel_special_forms(
                &mut mors,
                &mut gras,
                self.item
                    .words()
                    .iter()
                    .zip(self.words)
                    .enumerate()
                    .map(|(word, (batch_word, extracted))| (batch_word, extracted, vec![word])),
            );
        }
        match crate::inject::inject_morphosyntax(self.utt, mors, self.item.terminator.clone(), gras)
        {
            // Per-utterance injection failure: the 1-to-1 invariant (CHAT
            // alignable-word count == Mor count after mapping) was violated.
            // Refuse the file rather than publishing missing annotations.
            Err(diag) => self.misalignment(diag, &[], RetokenizationContext::Preserve),
            Ok(()) => {
                self.result.l2.defer_utterance(
                    self.line_idx,
                    self.item,
                    alignment.as_ref(),
                    &ItemPlacement::PerChatWord,
                );
                Ok(())
            }
        }
    }

    /// Retokenize mode: the main tier is rebuilt from the model's tokens
    /// (`surface`, read from the analysis as the model returned it), one item
    /// per model word (after the MWT lexicon's expansion), and a CHAT word's
    /// items are the tokens the text mapping gives it: the same mapping that
    /// rebuilds the main tier.
    fn inject_retokenized(
        self,
        surface: GroupedSurface,
        mapped: MappedTokens,
        alignment: Result<UdAlignment<'_, MorItemIndex>, UdAlignmentError>,
        mwt: &MwtDict,
    ) -> UtteranceCompletion {
        use crate::retokenize::MappingBasis;

        let (mors, gras, token_items) = mapped.into_indexed();
        let expanded = SurfaceItems::pair(surface, mors, gras, self.words)
            .and_then(|paired| paired.expand(mwt));
        let Expanded {
            tokens,
            mut mors,
            mut gras,
            expansion,
            by_text,
        } = match expanded {
            Ok(expanded) => expanded,
            // The utterance is left as it was, and the failure reported.
            Err(refused) => {
                let tokens = refused.stanza_tokens_after_mapping.clone();
                return self.misalignment(
                    refused,
                    &tokens,
                    RetokenizationContext::StanzaRetokenize,
                );
            }
        };

        // A special form's or code-switched word's items are found through
        // the mapping, and a mapping spread by length can hand them another
        // word's items (a `c|` analysis on the word before a special form).
        // Such an utterance is reported, not placed by guesswork.
        let places_words = self
            .item
            .words()
            .iter()
            .any(|word| !matches!(word.role(), WordRole::Analysed));
        if places_words && by_text.basis() == MappingBasis::Length {
            let refused = crate::inject::MisalignmentDiagnostic {
                chat_words: self
                    .words
                    .iter()
                    .map(|w| w.text.as_str().to_string())
                    .collect(),
                expected: talkbank_model::alignment::helpers::MorAlignableWordCount::new(
                    self.words.len(),
                ),
                actual: talkbank_model::alignment::helpers::MorItemCount::new(tokens.len()),
                stanza_tokens_after_mapping: tokens,
                suspected_class: crate::inject::MisalignmentClass::Unknown,
            };
            let tokens = refused.stanza_tokens_after_mapping.clone();
            return self.misalignment(refused, &tokens, RetokenizationContext::StanzaRetokenize);
        }
        relabel_special_forms(
            &mut mors,
            &mut gras,
            self.item.words().iter().zip(self.words).enumerate().map(
                |(word, (batch_word, extracted))| {
                    (
                        batch_word,
                        extracted,
                        by_text.tokens_for_word(word).to_vec(),
                    )
                },
            ),
        );

        // Collect retokenization trace info before modifying the AST.
        self.result.retokenization_traces.push(RetokenizationInfo {
            utterance_ordinal: self.utt_ordinal,
            original_words: self
                .words
                .iter()
                .map(|w| w.text.as_str().to_string())
                .collect(),
            stanza_tokens: tokens.clone(),
            mapping: (0..self.words.len())
                .map(|i| by_text.tokens_for_word(i).to_vec())
                .collect(),
            used_fallback: by_text.basis() == MappingBasis::Length,
        });

        match crate::retokenize::retokenize_utterance(
            self.parser,
            self.utt,
            self.words,
            &tokens,
            &by_text,
            mors,
            self.item.terminator.clone(),
            gras,
        ) {
            // File-level absorption: the typed diagnostic becomes a
            // `MisalignmentBug` outcome with its own strategy label, loud
            // rather than silently absorbed.
            Err(diag) => self.misalignment(diag, &tokens, RetokenizationContext::StanzaRetokenize),
            Ok(()) => {
                let placement = ItemPlacement::Retokenized(RetokenizedItems {
                    mapped: &token_items,
                    expansion: &expansion,
                    by_text: &by_text,
                });
                self.result.l2.defer_utterance(
                    self.line_idx,
                    self.item,
                    alignment.as_ref(),
                    &placement,
                );
                Ok(())
            }
        }
    }

    /// Record a misalignment the injector reported, with the caller's
    /// context, as a `MisalignmentBug` outcome. Retokenization failures share
    /// the outcome class but have their own stable strategy label.
    fn misalignment(
        self,
        diag: crate::inject::MisalignmentDiagnostic,
        tokens: &[String],
        context: RetokenizationContext,
    ) -> UtteranceCompletion {
        let strategy = match context {
            RetokenizationContext::Preserve => MorphosyntaxStrategy::MisalignmentBug,
            RetokenizationContext::StanzaRetokenize => MorphosyntaxStrategy::RetokenizationFailed,
        };
        let diag = enrich_diagnostic(diag, tokens, context);
        Err(Box::new(DecisionRecord::new_and_trace(
            self.line_idx,
            self.utt.main.speaker.as_str().to_string(),
            DecisionStrategy::Morphosyntax(strategy),
            format!(
                "class={} expected={} actual={} chat_words={:?} stanza_tokens={:?}",
                diag.suspected_class.as_str(),
                diag.expected,
                diag.actual,
                diag.chat_words,
                diag.stanza_tokens_after_mapping,
            ),
            true,
        )))
    }
}

/// Why a retokenized utterance has no surface tokens.
enum SurfaceError {
    /// The analysis as returned does not walk.
    Walk(MappingError),
    /// A special form cannot be written back: the model's tokens do not stand
    /// one per CHAT word.
    Unplaced(Refused),
}

/// The model's tokens of a retokenized utterance: its words as the model
/// returned them, before any grammatical-invariant rewrite. Range rows and
/// the terminator are not tokens (the terminator travels as the typed
/// `Terminator`), exactly as in the walk the items were mapped from.
///
/// A special form was sent to the model as its placeholder, so its token is
/// written back as the CHAT word: the text mapping then holds, and the
/// rebuild keeps the word as written (`gumma@c`). That needs the model's
/// tokens to stand one per CHAT word; when they do not, the utterance is
/// refused.
/// Component spellings and their source ownership travel together. Only the
/// admitted UD walk can construct this boundary payload.
struct GroupedSurface {
    tokens: Vec<String>,
    mapping: crate::retokenize::WordTokenMapping,
}

fn surface_tokens(
    raw: &super::UdSentence,
    item: &super::payload::MorphosyntaxBatchItem,
    words: &[crate::extract::ExtractedWord],
) -> Result<GroupedSurface, SurfaceError> {
    let tokens = UdTokens::walk(raw).map_err(|e| SurfaceError::Walk(MappingError::from(e)))?;
    let per_token: Vec<Vec<String>> = tokens
        .iter()
        .map(|(_, token)| {
            token
                .words()
                .iter()
                .map(|word| word.text.chars().filter(|c| !c.is_whitespace()).collect())
                .collect()
        })
        .collect();
    let has_special_form = item
        .words()
        .iter()
        .any(|word| matches!(word.role(), WordRole::SpecialForm(_)));
    if has_special_form && per_token.len() != words.len() {
        let surface: Vec<String> = per_token.into_iter().flatten().collect();
        return Err(SurfaceError::Unplaced(
            crate::inject::MisalignmentDiagnostic {
                chat_words: words.iter().map(|w| w.text.as_str().to_string()).collect(),
                expected: talkbank_model::alignment::helpers::MorAlignableWordCount::new(
                    words.len(),
                ),
                actual: talkbank_model::alignment::helpers::MorItemCount::new(surface.len()),
                stanza_tokens_after_mapping: surface,
                suspected_class: crate::inject::MisalignmentClass::Unknown,
            },
        ));
    }
    let mut surfaces = Vec::with_capacity(per_token.len());
    let mut flattened = Vec::new();
    let mut ranges = Vec::with_capacity(per_token.len());
    for ((at, token), mut components) in tokens.iter().zip(per_token) {
        let mut surface = match token.ud() {
            super::alignment::AlignedUd::Word(word) => word.text.clone(),
            super::alignment::AlignedUd::Mwt { range, .. } => range.text.clone(),
        };
        if has_special_form
            && matches!(item.words()[at.as_usize()].role(), WordRole::SpecialForm(_))
        {
            surface = words[at.as_usize()].text.as_str().to_owned();
            components = vec![surface.clone()];
        }
        surfaces.push(surface.chars().filter(|c| !c.is_whitespace()).collect());
        let start = flattened.len();
        flattened.extend(components);
        ranges.push(start..flattened.len());
    }
    let surface_mapping = crate::retokenize::build_word_token_mapping(words, &surfaces);
    let mapping = if surface_mapping.basis() == crate::retokenize::MappingBasis::Text {
        surface_mapping.expanded(&ranges).ok_or_else(|| {
            SurfaceError::Unplaced(crate::inject::MisalignmentDiagnostic {
                chat_words: words.iter().map(|w| w.text.as_str().to_owned()).collect(),
                expected: talkbank_model::alignment::helpers::MorAlignableWordCount::new(
                    words.len(),
                ),
                actual: talkbank_model::alignment::helpers::MorItemCount::new(flattened.len()),
                stanza_tokens_after_mapping: flattened.clone(),
                suspected_class: crate::inject::MisalignmentClass::Unknown,
            })
        })?
    } else {
        // A model may omit/change a range surface while its components still
        // spell the source exactly. Retain that existing text-bound path.
        crate::retokenize::build_word_token_mapping(words, &flattened)
    };
    Ok(GroupedSurface {
        tokens: flattened,
        mapping,
    })
}

/// Deprel label of a later piece of an MWT lexicon expansion: the pieces
/// share the token's one analysis, so a later piece is attached to the first
/// as part of one fixed expression.
const LEXICON_PIECE_RELATION_LABEL: &str = "FIXED";

/// A retokenized utterance's items, each paired with the model token the
/// main tier is rebuilt from, and its relations (one per chunk, in order,
/// then the terminator's). Built only by [`SurfaceItems::pair`], which
/// refuses an analysis whose item count differs from the model's token
/// count: an item a rewrite added (the English contraction table's `have` +
/// `to` for a `hafta` the model left whole) has no token of the model's to
/// be written as, and the main tier never gains a word the transcriber did
/// not write.
struct SurfaceItems {
    tokens: Vec<String>,
    mors: Vec<Mor>,
    gras: Vec<GrammaticalRelation>,
    mapping: crate::retokenize::WordTokenMapping,
}

/// A refused retokenization: the diagnostic the utterance is reported with.
type Refused = crate::inject::MisalignmentDiagnostic;

impl SurfaceItems {
    /// Pair each item with its token, or refuse.
    fn pair(
        surface: GroupedSurface,
        mors: Vec<Mor>,
        gras: Vec<GrammaticalRelation>,
        words: &[crate::extract::ExtractedWord],
    ) -> Result<Self, Refused> {
        let GroupedSurface { tokens, mapping } = surface;
        if tokens.len() == mors.len() {
            return Ok(Self {
                tokens,
                mors,
                gras,
                mapping,
            });
        }
        Err(crate::inject::MisalignmentDiagnostic {
            chat_words: words.iter().map(|w| w.text.as_str().to_string()).collect(),
            expected: talkbank_model::alignment::helpers::MorAlignableWordCount::new(tokens.len()),
            actual: talkbank_model::alignment::helpers::MorItemCount::new(mors.len()),
            stanza_tokens_after_mapping: tokens,
            suspected_class: crate::inject::MisalignmentClass::Unknown,
        })
    }

    /// Apply the MWT lexicon: a token the lexicon names becomes its pieces,
    /// one item each. The first piece keeps the token's analysis and
    /// relation; each later piece repeats the analysis and is attached to the
    /// first (`FIXED`). Every chunk index and head is renumbered, the
    /// terminator's relation included, so the relations stay one per chunk
    /// and form the tree the analysis did.
    fn expand(self, mwt: &MwtDict) -> Result<Expanded, Refused> {
        let Self {
            tokens,
            mors,
            gras,
            mapping,
        } = self;
        let pieces: Vec<Vec<String>> = tokens
            .into_iter()
            .map(|token| {
                match mwt
                    .get(&token.to_lowercase())
                    .or_else(|| mwt.get(token.as_str()))
                {
                    Some(expansion) if !expansion.is_empty() => expansion.clone(),
                    // A token the lexicon does not name (or names with no
                    // pieces) is its own one piece.
                    Some(_) | None => vec![token],
                }
            })
            .collect();
        let chunks: Vec<usize> = mors.iter().map(Mor::count_chunks).collect();
        let chunk_total: usize = chunks.iter().sum();
        // The relations must be the mapper's: one per chunk, in order, then
        // the terminator's. Anything else cannot be renumbered.
        let in_order = gras.len() == chunk_total + 1
            && gras.iter().enumerate().all(|(at, gra)| gra.index == at + 1);
        if !in_order {
            return Err(crate::inject::MisalignmentDiagnostic {
                chat_words: Vec::new(),
                stanza_tokens_after_mapping: pieces.into_iter().flatten().collect(),
                expected: talkbank_model::alignment::helpers::MorAlignableWordCount::new(
                    chunk_total + 1,
                ),
                actual: talkbank_model::alignment::helpers::MorItemCount::new(gras.len()),
                suspected_class: crate::inject::MisalignmentClass::MorGraCountMismatch,
            });
        }

        // Per item: its first chunk before (1-based) and its first piece's
        // first chunk after.
        let mut old_first = Vec::with_capacity(chunks.len());
        let mut new_first = Vec::with_capacity(chunks.len());
        let (mut old_next, mut new_next) = (1usize, 1usize);
        for (count, item_pieces) in chunks.iter().zip(&pieces) {
            old_first.push(old_next);
            new_first.push(new_next);
            old_next += count;
            new_next += count * item_pieces.len();
        }
        let (old_terminator, new_terminator) = (old_next, new_next);
        let renumber = |chunk: usize| -> usize {
            match chunk {
                0 => 0,
                c if c == old_terminator => new_terminator,
                c => {
                    // The item whose chunks hold `c`: the last one starting
                    // at or before it.
                    let item = old_first.partition_point(|&first| first <= c) - 1;
                    new_first[item] + (c - old_first[item])
                }
            }
        };

        let mut next = 0;
        let expansion: Vec<_> = pieces
            .iter()
            .map(|parts| {
                let start = next;
                next += parts.len();
                start..next
            })
            .collect();
        let by_text = mapping.expanded(&expansion).ok_or_else(|| Refused {
            chat_words: Vec::new(),
            expected: talkbank_model::alignment::helpers::MorAlignableWordCount::new(
                expansion.len(),
            ),
            actual: talkbank_model::alignment::helpers::MorItemCount::new(next),
            stanza_tokens_after_mapping: pieces.iter().flatten().cloned().collect(),
            suspected_class: crate::inject::MisalignmentClass::Unknown,
        })?;
        let mut out = Expanded {
            tokens: Vec::with_capacity(new_terminator),
            mors: Vec::with_capacity(new_terminator),
            gras: Vec::with_capacity(new_terminator),
            expansion,
            by_text,
        };
        let mut relations = gras.into_iter();
        for (item, (mor, item_pieces)) in mors.into_iter().zip(pieces).enumerate() {
            let count = chunks[item];
            for relation in relations.by_ref().take(count) {
                out.gras.push(GrammaticalRelation::new(
                    renumber(relation.index),
                    renumber(relation.head),
                    relation.relation.as_str(),
                ));
            }
            for (piece_at, piece) in item_pieces.into_iter().enumerate() {
                if piece_at > 0 {
                    for offset in 0..count {
                        out.gras.push(GrammaticalRelation::new(
                            new_first[item] + piece_at * count + offset,
                            new_first[item] + offset,
                            LEXICON_PIECE_RELATION_LABEL,
                        ));
                    }
                }
                out.tokens.push(piece);
                out.mors.push(mor.clone());
            }
        }
        // The terminator's relation, the one left.
        for relation in relations {
            out.gras.push(GrammaticalRelation::new(
                renumber(relation.index),
                renumber(relation.head),
                relation.relation.as_str(),
            ));
        }
        Ok(out)
    }
}

/// The items of a retokenized utterance after the MWT lexicon expanded its
/// tokens, with where each mapped item went.
struct Expanded {
    tokens: Vec<String>,
    mors: Vec<Mor>,
    gras: Vec<GrammaticalRelation>,
    /// Per mapped item, the items it became.
    expansion: Vec<Range<usize>>,
    by_text: crate::retokenize::WordTokenMapping,
}
