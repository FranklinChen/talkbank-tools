//! Owned phases. Only a completed phase exposes its next transition.

use super::{resolve_per_file_lang, unsupported_primary_language_error};
use crate::chat_ops::morphosyntax_ops::{
    BatchItemWithPosition, MultilingualPolicy, MwtDict, PosHintEvidence, TokenizationMode,
    apply_pos_hint_evidence, clear_morphosyntax, collect_payloads, collect_pos_hints,
    declared_languages, l2, remove_empty_morphosyntax_placeholders, validate_mor_alignment,
};
use crate::chat_ops::nlp::UdResponse;
use crate::chat_ops::{ChatFile, LanguageCode};
use crate::morphosyntax::{MorphotagDisposition, infer_batch};
use crate::pipeline::post_validate::PostValidated;
use crate::{api::LanguageCode3, error::ServerError, pipeline::PipelineServices};
use batchalign_transform::morphosyntax::MatchedMorphosyntaxResponses;
use batchalign_transform::parse::parse_lenient;
use batchalign_transform::validate::{ValidityLevel, validate_to_level};
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
pub(super) enum ParsedFile {
    PassThrough(ChatFile),
    Analyze(Analysis<Parsed>),
}

struct FileLanguage {
    api: LanguageCode3,
    model: LanguageCode,
}

/// The document and per-file language survive each phase by ownership.
pub(super) struct Analysis<S> {
    chat: ChatFile,
    language: FileLanguage,
    state: S,
}

pub(super) struct Parsed {
    errors: Vec<crate::chat_ops::ParseError>,
}
pub(super) struct Admitted;
pub(super) struct Prepared;
pub(super) struct Collected {
    items: Vec<BatchItemWithPosition>,
    hints: HintPlan,
}
pub(super) struct Inferred {
    work: InferenceWork,
}
pub(super) struct Applied;

/// The terminal phase: the document is finished AND its bytes have passed the
/// post-validation gate.
///
/// It carries the proof rather than re-deriving it, so the bytes that were
/// validated are exactly the bytes `serialize` hands back; there is no second
/// serialization for a caller to get wrong.
pub(super) struct PostChecked {
    output: PostValidated,
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
    pub(super) fn parse(
        text: &str,
        policy: crate::options::CaMorphotagPolicy,
    ) -> Result<Self, ServerError> {
        let (chat, errors) = parse_lenient(&crate::chat_parser(), text);
        if !errors.is_empty() {
            warn!(
                num_errors = errors.len(),
                "Parse errors in morphosyntax input (continuing with recovery)"
            );
        }
        match policy.disposition_for(&chat) {
            MorphotagDisposition::PassThroughCa => Ok(Self::PassThrough(chat)),
            MorphotagDisposition::Analyze => {
                if let Some(message) = unsupported_primary_language_error(&chat) {
                    warn!(reason = %message, "Morphotag rejected unsupported primary language");
                    return Err(ServerError::Validation(message));
                }
                let api = resolve_per_file_lang(&chat)?;
                let model = LanguageCode::new(api.as_ref()).map_err(|e| {
                    ServerError::Validation(format!(
                        "morphotag: invalid resolved language code {:?}: {e}",
                        api.as_ref()
                    ))
                })?;
                Ok(Self::Analyze(Analysis {
                    chat,
                    language: FileLanguage { api, model },
                    state: Parsed { errors },
                }))
            }
        }
    }
}

impl Analysis<Parsed> {
    pub(super) fn admit(self) -> Result<Analysis<Admitted>, ServerError> {
        if let Err(errors) =
            validate_to_level(&self.chat, &self.state.errors, ValidityLevel::MainTierValid)
        {
            let messages: Vec<String> = errors.iter().map(ToString::to_string).collect();
            return Err(ServerError::Validation(format!(
                "morphotag pre-validation failed: {}",
                messages.join("; ")
            )));
        }
        Ok(Analysis {
            chat: self.chat,
            language: self.language,
            state: Admitted,
        })
    }
}

impl Analysis<Admitted> {
    pub(super) fn clear(mut self) -> Analysis<Prepared> {
        clear_morphosyntax(&mut self.chat);
        Analysis {
            chat: self.chat,
            language: self.language,
            state: Prepared,
        }
    }
}

impl Analysis<Prepared> {
    pub(super) fn collect(self, options: &RunOptions<'_>) -> Analysis<Collected> {
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
        // Per-file reporting of collected.not_applicable remains a separate concern.
        Analysis {
            chat: self.chat,
            language: self.language,
            state: Collected {
                items: collected.batch_items,
                hints,
            },
        }
    }
}

impl Analysis<Collected> {
    pub(super) async fn infer(
        self,
        options: &RunOptions<'_>,
    ) -> Result<Analysis<Inferred>, ServerError> {
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

    /// Admit the worker boundary once, retaining payloads with their responses.
    pub(super) fn with_responses(
        self,
        responses: Vec<UdResponse>,
    ) -> Result<Analysis<Inferred>, ServerError> {
        let batch = MatchedMorphosyntaxResponses::new(self.state.items, responses)
            .map_err(|error| ServerError::Validation(error.to_string()))?;
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
            state: Inferred { work },
        })
    }
}

impl Analysis<Inferred> {
    pub(super) async fn apply(
        mut self,
        options: &RunOptions<'_>,
    ) -> Result<Analysis<Applied>, ServerError> {
        match self.state.work {
            InferenceWork::NoWork => {}
            InferenceWork::Responses { batch, hints } => {
                let deferred = if options.l2.should_analyze() {
                    l2::extract_l2_deferred_positions(batch.items(), batch.responses())
                } else {
                    Vec::new()
                };
                let injection = batch
                    .inject(
                        &crate::chat_parser(),
                        &mut self.chat,
                        options.tokenization,
                        options.mwt,
                    )
                    .map_err(|e| {
                        ServerError::Validation(format!("Result injection failed: {e}"))
                    })?;
                if !deferred.is_empty() {
                    crate::morphosyntax::dispatch_secondary_l2(
                        &mut self.chat,
                        &deferred,
                        options.services,
                        "single-file",
                    )
                    .await;
                }
                if let HintPlan::Captured(evidence) = hints {
                    let outcome = apply_pos_hint_evidence(
                        &mut self.chat,
                        evidence,
                        &injection.retokenization_traces,
                    );
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
            state: Applied,
        })
    }
}

impl Analysis<Applied> {
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
    /// `MainTierValid` is the level `admit` demanded of the input, so this is
    /// the no-degradation half of the rule; `validate_output` inside the gate
    /// adds the `%mor`-item-count check that is specific to morphotag.
    pub(super) fn postcheck(
        mut self,
        options: &RunOptions<'_>,
    ) -> Result<Analysis<PostChecked>, ServerError> {
        batchalign_transform::decisions::strip_decision_tiers(&mut self.chat);
        let provenance = crate::provenance::morphotag_provenance(
            self.language.api.as_ref(),
            options.services.engine_version.as_ref(),
            options.tokenization == TokenizationMode::StanzaRetokenize,
            false,
        );
        crate::provenance::inject_provenance(&mut self.chat, &provenance);
        remove_empty_morphosyntax_placeholders(&mut self.chat);

        let output = PostValidated::gate(
            &self.chat,
            ValidityLevel::MainTierValid,
            crate::api::ReleasedCommand::Morphotag,
        )
        .map_err(|failure| ServerError::Validation(failure.to_string()))?;
        Ok(Analysis {
            chat: self.chat,
            language: self.language,
            state: PostChecked { output },
        })
    }
}

impl Analysis<PostChecked> {
    /// Hand back the gate-proven output.
    ///
    /// Nothing is serialized here: `postcheck` already produced the bytes it
    /// validated, and re-serializing would reopen the gap between what was
    /// checked and what is written.
    pub(super) fn serialize(self) -> PostValidated {
        self.state.output
    }
}
