//! Owned phases. Only a completed phase exposes its next transition.

use super::resolve_per_file_lang;
use crate::chat_ops::morphosyntax_ops::{
    CollectedUtterance, MultilingualPolicy, MwtDict, PosHintEvidence, TokenizationMode,
    apply_pos_hint_evidence, collect_payloads, collect_pos_hints, declared_languages,
    remove_empty_morphosyntax_placeholders, validate_mor_alignment,
};
use crate::chat_ops::{ChatFile, LanguageCode};
use crate::morphosyntax::identity::{AdmittedMorphosyntaxResponse, AppliedAnalyses};
use crate::morphosyntax::infer_batch;
use crate::pipeline::post_validate::{DiagnosedOutput, LocalizedDiagnosis, PostValidated};
use crate::{api::LanguageCode3, error::ServerError, pipeline::PipelineServices};
use batchalign_transform::morphosyntax::MatchedMorphosyntaxResponses;
use talkbank_model::model::{ChatOptionFlag, Header, TranscriptName};
use talkbank_model::validation::ValidChatFile;
use talkbank_parser::ReplacementTiers;
use tracing::warn;

/// Immutable inputs shared across transitions; job-level language is excluded.
pub(super) struct RunOptions<'a> {
    services: PipelineServices<'a>,
    tokenization: TokenizationMode,
    multilingual: MultilingualPolicy,
    mwt: &'a MwtDict,
    l2: crate::params::L2MorphotagPolicy,
    hints: crate::params::PosHintPolicy,
    progress: Option<&'a crate::execution::morphotag::progress::BackendProgressPort>,
    cancellation: crate::infer_retry::Cancellation<'a>,
}

impl<'a> RunOptions<'a> {
    pub(super) fn new(
        services: PipelineServices<'a>,
        params: &crate::params::MorphosyntaxParams<'a>,
    ) -> Self {
        Self {
            services,
            tokenization: params.tokenization_mode,
            multilingual: params.multilingual_policy,
            mwt: params.mwt,
            l2: params.policy.l2,
            hints: params.policy.pos_hints,
            progress: params.progress,
            cancellation: params.cancellation,
        }
    }
}

/// CA pass-through never claims a language or an analysis admission.
pub(crate) enum ParsedFile {
    PassThrough(ValidChatFile),
    Analyze(Analysis<Admitted>),
}

/// A diagnosed generated document whose findings are confined to some
/// utterances, read for morphosyntax.
pub(crate) enum LocalizedParse {
    /// `@Options: CA` under a policy that honors it: morphosyntax declines
    /// the document, which is handed back unchanged.
    Declined(DiagnosedOutput),
    /// Analyzed everywhere but at the held-out utterances. Boxed: it owns
    /// the whole document.
    Analyze(Box<Analysis<Admitted, OutsideHeldOut>>),
}

struct FileLanguage {
    api: LanguageCode3,
    model: LanguageCode,
}

impl FileLanguage {
    /// Resolve the file's language and admit it for analysis.
    fn resolve(chat: &ChatFile) -> Result<Self, ServerError> {
        let api = resolve_per_file_lang(chat)?;
        crate::morphosyntax::AnalysisUnavailable::admit_primary(&api)?;
        let model = LanguageCode::new(api.as_ref()).map_err(|e| {
            ServerError::Validation(format!(
                "morphotag: invalid resolved language code {:?}: {e}",
                api.as_ref()
            ))
        })?;
        Ok(Self { api, model })
    }
}

/// The document and per-file language survive each phase by ownership, with
/// the scope that says which utterances are analyzed and how the finished
/// document is judged.
pub(crate) struct Analysis<S, D = WholeDocument> {
    chat: ChatFile,
    language: FileLanguage,
    scope: D,
    state: S,
}

/// Which utterances an analysis edits, and how its finished document is
/// judged. The two belong together: a document analyzed outside some
/// utterances cannot pass the strict gate those utterances already failed.
pub(crate) trait AnalysisScope: Send {
    /// What judging the finished document yields.
    type Judged: Send;
    /// Whether the utterance at this ordinal (among the file's utterances)
    /// is analyzed.
    fn analyzes(&self, utterance: usize) -> bool;
    /// Judge the finished document.
    fn judge(self, chat: ChatFile) -> Result<Self::Judged, ServerError>;
}

/// An admitted document: every utterance is analyzed and the result must pass
/// the strict gate, failing the stage otherwise.
pub(crate) struct WholeDocument;

impl AnalysisScope for WholeDocument {
    type Judged = PostValidated;

    fn analyzes(&self, _utterance: usize) -> bool {
        true
    }

    fn judge(self, chat: ChatFile) -> Result<PostValidated, ServerError> {
        PostValidated::gate_owned(chat, crate::api::ReleasedCommand::Morphotag)
            .map_err(|failure| failure.into_server_error())
    }
}

/// A generated document diagnosed for findings confined to the held-out
/// utterances: every other utterance is analyzed, the held-out ones are never
/// sent to a worker and keep their form, and the result is judged by
/// [`PostValidated::produced_outside`], which refuses it only if the analysis
/// added a finding of its own.
pub(crate) struct OutsideHeldOut {
    held_out: crate::pipeline::post_validate::HeldOutUtterances,
}

impl AnalysisScope for OutsideHeldOut {
    type Judged = crate::pipeline::post_validate::ProducedOutput;

    fn analyzes(&self, utterance: usize) -> bool {
        !self.held_out.contains(utterance)
    }

    fn judge(
        self,
        chat: ChatFile,
    ) -> Result<crate::pipeline::post_validate::ProducedOutput, ServerError> {
        PostValidated::produced_outside(
            chat,
            &self.held_out,
            crate::api::ReleasedCommand::Morphotag,
        )
        .map_err(|failure| failure.into_server_error())
    }
}

pub(crate) struct Admitted;
pub(super) struct Collected {
    items: Vec<CollectedUtterance>,
    hints: HintPlan,
}
pub(super) struct Inferred {
    work: InferenceWork,
    /// What the workers reported about the admitted responses, before any L2
    /// re-analysis.
    applied: AppliedAnalyses,
}
/// Applied, carrying what every analysis that reached the document reported:
/// the models, which the provenance comment names, and the relation repairs
/// made inside them, which it counts.
pub(super) struct Applied {
    applied: AppliedAnalyses,
}

/// The terminal phase: the document is finished AND judged by its scope (for
/// an admitted document, the post-validation gate).
///
/// It carries the judgement rather than re-deriving it, so the bytes that
/// were validated are exactly the bytes `serialize` hands back; there is no
/// second serialization for a caller to get wrong.
pub(super) struct PostChecked<O> {
    output: O,
}

/// Captured evidence, rather than a policy flag paired with a missing value.
enum HintPlan {
    Ignored,
    Captured(PosHintEvidence),
}
enum InferenceWork {
    NoWork,
    Responses {
        batch: MatchedMorphosyntaxResponses,
        hints: HintPlan,
    },
}

impl ParsedFile {
    pub(crate) fn parse(
        text: &str,
        policy: crate::options::CaMorphotagPolicy,
    ) -> Result<Self, ServerError> {
        // The planner's choice, kept to hold the admission to it.
        let mut planned = None;
        let admitted = crate::chat_parser()
            .admit_planned_tiers(text, TranscriptName::Anonymous, |headers| {
                let ca = headers.iter().any(|header| {
                    matches!(header,
                        Header::Options { options } if options.iter().any(|flag|
                            matches!(flag, ChatOptionFlag::Ca))
                    )
                });
                planned = match (policy, ca) {
                    (crate::options::CaMorphotagPolicy::Honor, true) => None,
                    (crate::options::CaMorphotagPolicy::Honor, false)
                    | (crate::options::CaMorphotagPolicy::Analyze, _) => {
                        Some(ReplacementTiers::Morphosyntax)
                    }
                };
                planned
            })
            .map_err(ServerError::ChatReplacementAdmission)?;
        let selected = admitted.selection();
        let contradicted = || ServerError::ReplacementPlanContradicted {
            planned,
            admitted: selected,
        };
        if selected != planned {
            return Err(contradicted());
        }
        match selected {
            None => Ok(Self::PassThrough(admitted.into_valid_file())),
            Some(ReplacementTiers::Morphosyntax) => {
                let chat = admitted.into_valid_file().into_unchecked();
                Self::analyze(chat)
            }
            // The plan above never selects word timing.
            Some(ReplacementTiers::WordTiming) => Err(contradicted()),
        }
    }

    /// Continue from admitted constructed output, never its serialization.
    ///
    /// Admitted by its type. A generating producer's diagnosed output is
    /// analyzed only through [`Self::from_localized`], outside the utterances
    /// its findings are confined to.
    pub(crate) fn from_output(
        output: crate::pipeline::post_validate::PostValidated,
        policy: crate::options::CaMorphotagPolicy,
    ) -> Result<Self, ServerError> {
        let mut chat = output.into_judged_document();
        if declines_ca(&chat, policy) {
            let chat = chat
                .validate_construction_with_policy(
                    talkbank_model::validation::ValidationPolicy::new(
                        talkbank_model::RuleSelection::new(),
                        talkbank_model::validation::AlignmentValidation::IncludeTierAlignment,
                    ),
                    &talkbank_model::NullErrorSink,
                    TranscriptName::Anonymous,
                )
                .map_err(|failure| {
                    crate::pipeline::post_validate::PostValidationFailure::construction(
                        crate::api::ReleasedCommand::Morphotag,
                        Vec::new(),
                        &failure,
                    )
                    .into_server_error()
                })?;
            return Ok(Self::PassThrough(chat));
        }
        batchalign_transform::morphosyntax::clear_morphosyntax(&mut chat);
        Self::analyze(chat)
    }

    fn analyze(chat: ChatFile) -> Result<Self, ServerError> {
        let language = FileLanguage::resolve(&chat)?;
        Ok(Self::Analyze(Analysis {
            chat,
            language,
            scope: WholeDocument,
            state: Admitted,
        }))
    }

    /// Continue from a generated document diagnosed for findings confined to
    /// some utterances: those are held out, every other one is analyzed.
    ///
    /// The diagnosed model is never serialized or reparsed. The analyzed
    /// utterances' existing morphosyntax tiers are cleared as for an
    /// admitted document; the held-out ones are left exactly as they are.
    pub(crate) fn from_localized(
        localized: LocalizedDiagnosis,
        policy: crate::options::CaMorphotagPolicy,
    ) -> Result<LocalizedParse, ServerError> {
        if declines_ca(localized.model(), policy) {
            return Ok(LocalizedParse::Declined(localized.into_diagnosed()));
        }
        let (mut chat, held_out) = localized.into_model();
        let utterances = chat
            .lines
            .iter()
            .filter(|line| matches!(line, talkbank_model::model::Line::Utterance(_)))
            .count();
        let analyzed: std::collections::HashSet<usize> = (0..utterances)
            .filter(|ordinal| !held_out.contains(*ordinal))
            .collect();
        batchalign_transform::morphosyntax::clear_morphosyntax_selective(&mut chat, &analyzed);
        let language = FileLanguage::resolve(&chat)?;
        Ok(LocalizedParse::Analyze(Box::new(Analysis {
            chat,
            language,
            scope: OutsideHeldOut { held_out },
            state: Admitted,
        })))
    }
}

/// Whether morphosyntax declines this document: it declares `@Options: CA`
/// and the policy honors that.
fn declines_ca(chat: &ChatFile, policy: crate::options::CaMorphotagPolicy) -> bool {
    let ca = chat.lines.iter().any(|line| {
        matches!(line,
            talkbank_model::model::Line::Header { header, .. }
                if matches!(header.as_ref(), Header::Options { options }
                    if options.iter().any(|flag| matches!(flag, ChatOptionFlag::Ca)))
        )
    });
    match policy {
        crate::options::CaMorphotagPolicy::Honor => ca,
        crate::options::CaMorphotagPolicy::Analyze => false,
    }
}

impl Analysis<Admitted> {
    pub(crate) fn document(&self) -> &ChatFile {
        &self.chat
    }

    /// Editing consumes admission; language remains bound to the same file.
    pub(crate) fn into_incremental(self) -> IncrementalInput {
        IncrementalInput {
            chat: self.chat,
            language: self.language.model,
            api_language: self.language.api,
        }
    }
}

pub(crate) struct IncrementalInput {
    pub(crate) chat: ChatFile,
    pub(crate) language: LanguageCode,
    pub(crate) api_language: LanguageCode3,
}

impl<D: AnalysisScope> Analysis<Admitted, D> {
    /// Collect the payload of every utterance the scope analyzes. An
    /// utterance outside it is never sent to a worker, so it gets no
    /// analysis to inject.
    pub(super) fn collect(self, options: &RunOptions<'_>) -> Analysis<Collected, D> {
        let languages = declared_languages(&self.chat, &self.language.model);
        let collected = collect_payloads(
            &self.chat,
            &self.language.model,
            &languages,
            options.multilingual,
        );
        let hints = if options.hints.should_apply() {
            HintPlan::Captured(collect_pos_hints(&self.chat))
        } else {
            HintPlan::Ignored
        };
        let items = collected
            .batch_items
            .into_iter()
            .filter(|item| self.scope.analyzes(item.utt_ordinal()))
            .collect();
        // Per-file reporting of collected.not_applicable remains a separate concern.
        Analysis {
            chat: self.chat,
            language: self.language,
            scope: self.scope,
            state: Collected { items, hints },
        }
    }
}

impl<D: AnalysisScope> Analysis<Collected, D> {
    pub(super) async fn infer(
        self,
        options: &RunOptions<'_>,
    ) -> Result<Analysis<Inferred, D>, ServerError> {
        let responses = if self.state.items.is_empty() {
            Vec::new()
        } else {
            infer_batch(
                options.services.pool,
                &self.state.items,
                &self.language.api,
                options.mwt,
                options.tokenization == TokenizationMode::StanzaRetokenize,
                options.progress,
                options.cancellation,
            )
            .await?
        };
        self.with_responses(responses)
    }

    /// Admit the worker boundary once, retaining payloads with their responses
    /// and what the workers reported about them.
    pub(super) fn with_responses(
        self,
        responses: Vec<AdmittedMorphosyntaxResponse>,
    ) -> Result<Analysis<Inferred, D>, ServerError> {
        let (responses, applied) = AppliedAnalyses::take_applied(responses);
        let batch = MatchedMorphosyntaxResponses::from_admitted(self.state.items, responses)
            .map_err(batchalign_transform::morphosyntax::InjectionError::from)?;
        let work = if batch.items().is_empty() {
            InferenceWork::NoWork
        } else {
            InferenceWork::Responses {
                batch,
                hints: self.state.hints,
            }
        };
        Ok(Analysis {
            chat: self.chat,
            language: self.language,
            scope: self.scope,
            state: Inferred { work, applied },
        })
    }
}

impl<D: AnalysisScope> Analysis<Inferred, D> {
    pub(super) async fn apply(
        mut self,
        options: &RunOptions<'_>,
    ) -> Result<Analysis<Applied, D>, ServerError> {
        let Inferred { work, mut applied } = self.state;
        match work {
            InferenceWork::NoWork => {}
            InferenceWork::Responses { batch, hints } => {
                let injection = batch
                    .inject(
                        &crate::chat_parser(),
                        &mut self.chat,
                        options.tokenization,
                        options.mwt,
                    )
                    .map_err(ServerError::MorphosyntaxInjection)?;
                // The `@s` positions come from the analysis injection mapped,
                // in the items it wrote.
                let (retokenization_traces, l2) = injection.into_parts();
                let deferred = if options.l2.should_analyze() {
                    l2.into_reported_positions()
                } else {
                    Vec::new()
                };
                if !deferred.is_empty() {
                    applied.extend(
                        crate::morphosyntax::dispatch_secondary_l2(
                            &mut self.chat,
                            deferred,
                            options.services,
                            "single-file",
                            options.cancellation,
                        )
                        .await?,
                    );
                }
                if let HintPlan::Captured(evidence) = hints {
                    let outcome =
                        apply_pos_hint_evidence(&mut self.chat, evidence, &retokenization_traces);
                    tracing::debug!(?outcome, "Applied transcriber POS hints");
                }
                for warning in validate_mor_alignment(&self.chat) {
                    warn!(warning = %warning, "Morphosyntax alignment mismatch");
                }
            }
        }
        Ok(Analysis {
            chat: self.chat,
            language: self.language,
            scope: self.scope,
            state: Applied { applied },
        })
    }
}

impl<D: AnalysisScope> Analysis<Applied, D> {
    #[cfg(test)]
    pub(super) fn into_chat(self) -> ChatFile {
        self.chat
    }

    /// Finish the document and gate it, fail-closed.
    ///
    /// The finishing edits (decision-tier strip, provenance, empty-placeholder
    /// removal) moved here from `serialize` so that the thing validated is the
    /// thing written. Gating before them would have proved a draft.
    ///
    /// Input admission checks all retained content. The output judgement also
    /// adds the `%mor`-item-count check that is specific to morphotag.
    pub(super) fn postcheck(
        mut self,
        options: &RunOptions<'_>,
    ) -> Result<PostChecked<D::Judged>, ServerError> {
        batchalign_transform::decisions::strip_decision_tiers(&mut self.chat);
        if let Some(count) = self.state.applied.repair_count() {
            // The file-level report no worker can make: a worker sees one
            // batch, not the document these relations belong to. The stamp
            // below carries the count into the file itself; this line is the
            // operator's copy, and neither is the only record any more.
            warn!(
                repairs = count.get(),
                tally = %self.state.applied.repair_tally(),
                lang = %self.language.api,
                "Stanza relations repaired before injection"
            );
        }
        let provenance = crate::provenance::morphotag_provenance(
            &self.language.api,
            &self.state.applied,
            options.tokenization == TokenizationMode::StanzaRetokenize,
        );
        crate::provenance::inject_provenance(&mut self.chat, &provenance);
        remove_empty_morphosyntax_placeholders(&mut self.chat);

        let output = self.scope.judge(self.chat)?;
        Ok(PostChecked { output })
    }
}

impl<O> PostChecked<O> {
    /// Hand back the judged output.
    ///
    /// Nothing is serialized here: `postcheck` already produced the bytes it
    /// validated, and re-serializing would reopen the gap between what was
    /// checked and what is written.
    pub(super) fn serialize(self) -> O {
        self.output
    }
}
