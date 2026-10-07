//! Morphosyntax as consuming transitions over parsed, admitted and inferred data.

mod language;
mod states;
#[cfg(test)]
mod tests;

use super::plan::{StageId, observe_stage};
use crate::pipeline::post_validate::PostValidated;
use crate::{error::ServerError, pipeline::PipelineServices};
pub(crate) use language::resolve_per_file_lang;
pub(crate) use states::ParsedFile;
use states::{Admitted, Analysis, AnalysisScope, LocalizedParse, RunOptions};

/// Run morphosyntax without exposing an uninitialized or out-of-order context.
pub(crate) async fn run_morphosyntax_pipeline(
    chat_text: &str,
    services: PipelineServices<'_>,
    params: &crate::params::MorphosyntaxParams<'_>,
) -> Result<PostValidated, ServerError> {
    let parsed = observe_stage(
        "morphotag",
        StageId::Parse,
        Box::pin(async { ParsedFile::parse(chat_text, params.policy.ca_policy) }),
    )
    .await?;
    run_admitted_morphosyntax(parsed, services, params).await
}

/// Full regeneration consumes an existing admission instead of reparsing when
/// an incremental plan discovers that no prior analysis can be reused.
pub(crate) async fn run_admitted_morphosyntax(
    parsed: ParsedFile,
    services: PipelineServices<'_>,
    params: &crate::params::MorphosyntaxParams<'_>,
) -> Result<PostValidated, ServerError> {
    let options = RunOptions::new(services, params);
    let parsed = match parsed {
        ParsedFile::PassThrough(chat) => {
            return observe_stage(
                "morphotag",
                StageId::Serialize,
                Box::pin(async {
                    // A CA file morphotag declines to analyze is handed back
                    // unmodified APART FROM the decision-tier strip, which is why
                    // this is not `pass_through`: the latter carries the input's
                    // own bytes, and these are not them. The constructor performs
                    // the strip, so the two cannot come apart here.
                    PostValidated::declined_stripping_decision_tiers(
                        chat,
                        crate::api::ReleasedCommand::Morphotag,
                    )
                    .map_err(|failure| failure.into_server_error())
                }),
            )
            .await;
        }
        ParsedFile::Analyze(parsed) => parsed,
    };
    analyze_in_scope(parsed, &options).await
}

/// What morphosyntax made of a generated transcript diagnosed for findings
/// confined to some utterances.
pub(crate) enum LocalizedMorphosyntax {
    /// The document declares `@Options: CA` and the policy honors it: not
    /// analyzed, handed back unchanged.
    Declined(crate::pipeline::post_validate::DiagnosedOutput),
    /// Analyzed everywhere but at the held-out utterances, and judged afresh
    /// (still diagnosed for them, or admitted if the analysis happened to
    /// remove the findings' cause).
    Analyzed(crate::pipeline::post_validate::ProducedOutput),
}

/// Morphosyntax on a generated transcript whose findings are confined to some
/// utterances: every other utterance is analyzed, the held-out ones are never
/// sent to a worker. An analysis that adds a finding of its own outside them
/// is refused as `ServerError::OutputAdmission`, as for an admitted document.
pub(crate) async fn run_localized_morphosyntax(
    localized: crate::pipeline::post_validate::LocalizedDiagnosis,
    services: PipelineServices<'_>,
    params: &crate::params::MorphosyntaxParams<'_>,
) -> Result<LocalizedMorphosyntax, ServerError> {
    let options = RunOptions::new(services, params);
    match ParsedFile::from_localized(localized, params.policy.ca_policy)? {
        LocalizedParse::Declined(diagnosed) => Ok(LocalizedMorphosyntax::Declined(diagnosed)),
        LocalizedParse::Analyze(parsed) => Ok(LocalizedMorphosyntax::Analyzed(
            analyze_in_scope(*parsed, &options).await?,
        )),
    }
}

/// The analysis phases, the same whatever the document's standing; the scope
/// decides which utterances are analyzed and how the result is judged.
async fn analyze_in_scope<D: AnalysisScope>(
    parsed: Analysis<Admitted, D>,
    options: &RunOptions<'_>,
) -> Result<D::Judged, ServerError> {
    let collected = observe_stage(
        "morphotag",
        StageId::CollectPayloads,
        Box::pin(async { Ok(parsed.collect(options)) }),
    )
    .await?;
    let inferred = observe_stage(
        "morphotag",
        StageId::Infer,
        Box::pin(collected.infer(options)),
    )
    .await?;
    let applied = observe_stage(
        "morphotag",
        StageId::ApplyResults,
        Box::pin(inferred.apply(options)),
    )
    .await?;
    let checked = observe_stage(
        "morphotag",
        StageId::PostValidate,
        Box::pin(async { applied.postcheck(options) }),
    )
    .await?;
    observe_stage(
        "morphotag",
        StageId::Serialize,
        Box::pin(async { Ok(checked.serialize()) }),
    )
    .await
}
