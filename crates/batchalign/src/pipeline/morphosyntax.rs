//! Morphosyntax as consuming transitions over parsed, admitted and inferred data.

mod language;
mod states;
#[cfg(test)]
mod tests;

use super::plan::{StageId, observe_stage};
use crate::{error::ServerError, pipeline::PipelineServices};
pub(crate) use language::{resolve_per_file_lang, unsupported_primary_language_error};
use states::{ParsedFile, RunOptions};

/// Run morphosyntax without exposing an uninitialized or out-of-order context.
pub(crate) async fn run_morphosyntax_pipeline(
    chat_text: &str,
    services: PipelineServices<'_>,
    params: &crate::params::MorphosyntaxParams<'_>,
) -> Result<String, ServerError> {
    let options = RunOptions::new(services, params);
    let parsed = observe_stage("morphotag", StageId::Parse, async {
        ParsedFile::parse(chat_text, params.policy.ca_policy)
    })
    .await?;
    let parsed = match parsed {
        ParsedFile::PassThrough(mut chat) => {
            return observe_stage("morphotag", StageId::Serialize, async {
                batchalign_transform::decisions::strip_decision_tiers(&mut chat);
                Ok(batchalign_transform::serialize::to_chat_string(&chat))
            })
            .await;
        }
        ParsedFile::Analyze(parsed) => parsed,
    };
    let admitted =
        observe_stage("morphotag", StageId::PreValidate, async { parsed.admit() }).await?;
    let prepared = observe_stage("morphotag", StageId::ClearExisting, async {
        Ok(admitted.clear())
    })
    .await?;
    let collected = observe_stage("morphotag", StageId::CollectPayloads, async {
        Ok(prepared.collect(&options))
    })
    .await?;
    let inferred = observe_stage("morphotag", StageId::Infer, collected.infer(&options)).await?;
    let applied =
        observe_stage("morphotag", StageId::ApplyResults, inferred.apply(&options)).await?;
    let checked = observe_stage("morphotag", StageId::PostValidate, async {
        Ok(applied.postcheck())
    })
    .await?;
    observe_stage("morphotag", StageId::Serialize, async {
        Ok(checked.serialize(&options))
    })
    .await
}
