//! Gateway correspondence is admitted once before any publication effect.

use super::PreparedTextFile;
use crate::runner::util::{FileRunTracker, FileStage, TerminalRecorded, set_file_progress};
use crate::scheduling::FailureCategory;
use crate::text_batch::{TextBatchFileResult, TextBatchFileResults};

struct BoundTextResult {
    owner: PreparedTextFile,
    stamp: crate::api::FileStampOutcome,
    outcome: Result<
        crate::morphosyntax::report::MorphosyntaxOutput,
        crate::text_batch::TextWorkflowFileError,
    >,
}

/// Raw gateway responses are admitted once against this file's owner.
/// A missing, duplicate or foreign response fails the originating file.
pub(super) async fn publish_one(
    owner: PreparedTextFile,
    results: TextBatchFileResults,
) -> TerminalRecorded {
    let [result] = match <[TextBatchFileResult; 1]>::try_from(results) {
        Ok(result) => result,
        Err(results) => {
            return owner
                .fail(&format!(
                    "Expected one text result for {}, received {}",
                    owner.filename(),
                    results.len()
                ))
                .await;
        }
    };
    if result.filename != *owner.filename() {
        return owner
            .fail(&format!(
                "Unexpected text result {} for {}",
                result.filename,
                owner.filename()
            ))
            .await;
    }
    publish(BoundTextResult {
        owner,
        stamp: result.stamp,
        outcome: result.result,
    })
    .await
}

pub(super) async fn publish_batch(
    mut owners: Vec<PreparedTextFile>,
    mut results: TextBatchFileResults,
) {
    // Gateway responses may arrive in another order. Validate the whole
    // correspondence before publication, retaining every owner on refusal.
    owners.sort_by(|left, right| left.filename().cmp(right.filename()));
    results.sort_by(|left, right| left.filename.cmp(&right.filename));
    if owners.len() != results.len()
        || owners
            .iter()
            .zip(&results)
            .any(|(owner, result)| *owner.filename() != result.filename)
    {
        // A batch reports no per-file outcome: recording is the whole effect,
        // so each file's proof is dropped here.
        for owner in owners {
            let _recorded = owner
                .fail("Text results do not match the prepared source batch")
                .await;
        }
        return;
    }
    for (owner, result) in owners.into_iter().zip(results) {
        let _recorded = publish(BoundTextResult {
            owner,
            stamp: result.stamp,
            outcome: result.result,
        })
        .await;
    }
}

/// Outcomes produced for the single source locally already have correspondence.
pub(super) async fn publish_unstamped(
    owner: PreparedTextFile,
    outcome: Result<
        crate::morphosyntax::report::MorphosyntaxOutput,
        crate::text_batch::TextWorkflowFileError,
    >,
) -> TerminalRecorded {
    publish(BoundTextResult {
        owner,
        stamp: crate::api::FileStampOutcome::Unrecorded,
        outcome,
    })
    .await
}

async fn publish(bound: BoundTextResult) -> TerminalRecorded {
    let BoundTextResult {
        owner,
        stamp,
        outcome,
    } = bound;
    let sink = owner.host().sink().clone();
    let lifecycle = FileRunTracker::new(sink.as_ref(), owner.filename().as_ref());
    match outcome {
        Ok(output_chat) => {
            // Located against the document written below; done once
            // here rather than per artifact, because the answer cannot
            // differ between a file's artifacts.
            let crate::morphosyntax::report::FinishedMorphosyntax {
                document: finished,
                shortfalls,
                exclusions,
            } = match output_chat.finish() {
                Ok(finished) => finished,
                // A record that cannot name its utterance is an internal
                // fault: the file fails rather than reporting a position
                // nobody can find, classified as every such fault is.
                Err(error) => {
                    let error = crate::error::ServerError::from(error);
                    return lifecycle
                        .fail(
                            &error.to_string(),
                            crate::runner::util::classify_server_error(&error),
                        )
                        .await;
                }
            };
            let finished = crate::pipeline::post_validate::ProducedOutput::Admitted(finished);
            set_file_progress(
                sink.as_ref(),
                owner.filename().as_ref(),
                FileStage::Writing,
                Some(1),
                Some(1),
            )
            .await;
            let outputs = crate::recipe_runner::runtime::SourceOutputs::for_text_file(&owner);
            let target = match outputs.primary_target() {
                Ok(target) => target,
                Err(error) => {
                    return lifecycle
                        .fail(&error.to_string(), FailureCategory::System)
                        .await;
                }
            };
            let artifact = outputs.primary();
            if let Err(error) =
                crate::recipe_runner::runtime::write_chat_output_artifact_with_provenance_gate(
                    target, &finished,
                )
                .await
            {
                return lifecycle
                    .fail(&error.to_string(), FailureCategory::System)
                    .await;
            }
            use crate::pipeline::post_validate::OutputReport;
            let diagnostics = match OutputReport::of(&finished, &shortfalls) {
                OutputReport::Clean => None,
                OutputReport::Diagnosed(draft) => {
                    let sidecar = outputs.diagnostics_path();
                    Some(draft.record(&sidecar).await)
                }
            };
            lifecycle
                .complete_reported(
                    crate::store::CompletedFileOutput {
                        filename: artifact.display_path().clone(),
                        content_type: artifact.content_type(),
                        stamp,
                        exclusions,
                    },
                    diagnostics,
                )
                .await
        }
        Err(error) => lifecycle.fail(&error.to_string(), error.category()).await,
    }
}
