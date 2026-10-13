use std::collections::HashMap;
use std::path::PathBuf;

use crate::planning;
use crate::recipe_runner::materialize::MaterializedArtifactRole;
use crate::runner::DispatchHostContext;
use crate::runner::util::{FileRunTracker, FileStage, set_file_progress};
use crate::scheduling::FailureCategory;
use crate::store::{PendingJobFile, RunnerJobSnapshot};
use crate::text_batch::{TextBatchFileInput, TextBatchFileResults};

pub(crate) struct LoadedTextInputs {
    pub(crate) file_texts: Vec<TextBatchFileInput>,
    pub(crate) before_texts: HashMap<String, String>,
}

pub(crate) async fn load_text_inputs(
    job: &RunnerJobSnapshot,
    host: &DispatchHostContext,
    load_before_texts: bool,
) -> LoadedTextInputs {
    let mut loaded = LoadedTextInputs {
        file_texts: Vec::with_capacity(job.pending_files.len()),
        before_texts: HashMap::new(),
    };

    for file in &job.pending_files {
        let lifecycle = FileRunTracker::new(
            host.sink().as_ref(),
            &job.identity.job_id,
            file.filename.as_ref(),
        );
        lifecycle.stage(FileStage::Reading).await;

        let read_path = resolve_input_path(job, file);
        match tokio::fs::read_to_string(&read_path).await {
            Ok(text) => {
                if load_before_texts
                    && let Some(before_path) = resolve_before_path(job, file)
                    && let Ok(before_text) = tokio::fs::read_to_string(before_path).await
                {
                    loaded
                        .before_texts
                        .insert(file.filename.to_string(), before_text);
                }
                loaded
                    .file_texts
                    .push(TextBatchFileInput::new(file.filename.clone(), text));
            }
            Err(error) => {
                lifecycle
                    .fail(
                        &format!("Failed to read input: {error}"),
                        FailureCategory::InputMissing,
                    )
                    .await;
            }
        }
    }

    loaded
}

pub(crate) async fn write_text_results(
    job: &RunnerJobSnapshot,
    host: &DispatchHostContext,
    plan: &planning::JobPlan,
    results: TextBatchFileResults,
    should_merge_abbrev: bool,
    missing_artifact_label: &str,
) {
    let sink = host.sink().clone();
    let total_results = results.len() as i64;
    for (result_idx, file_result) in results.into_iter().enumerate() {
        let lifecycle = FileRunTracker::new(
            sink.as_ref(),
            &job.identity.job_id,
            file_result.filename.as_ref(),
        );
        let file_index = job
            .pending_files
            .iter()
            .find(|file| file.filename == file_result.filename)
            .map(|file| file.file_index)
            .unwrap_or(0);
        // What the command decided about stamping this file, recorded on the
        // file's own status so the job detail can report it.
        let stamp = file_result.stamp;
        match file_result.result {
            Ok(output_chat) => {
                let (output_chat, report) = output_chat.into_parts();
                let (shortfalls, exclusions) = report.into_records();
                // The abbreviation merge is a step INSIDE the gate, so the
                // bytes written below are bytes a gate judged. It used to run
                // on `output_chat.as_str()` in the artifact loop and the
                // writer wrote ITS result, which put a transform after the
                // proof: the proof described a document that never reached
                // disk. Done once here rather than per artifact, because the
                // answer cannot differ between a file's artifacts.
                let finished = if should_merge_abbrev {
                    match output_chat.with_abbreviations_merged() {
                        Ok(finished) => finished,
                        // POLICY, stated here because the refusal hands back
                        // `refused.unmerged`, an admissible output this seam
                        // deliberately declines to write: a merge that breaks
                        // its own gate is a defect in the transform, and
                        // shipping past it would leave nobody looking at it.
                        // The failure names the merge, so the operator is not
                        // told their command's output was invalid.
                        Err(refused) => {
                            lifecycle
                                .fail(&refused.to_string(), FailureCategory::Validation)
                                .await;
                            continue;
                        }
                    }
                } else {
                    output_chat
                };
                let finished = crate::pipeline::post_validate::ProducedOutput::Admitted(finished);
                set_file_progress(
                    sink.as_ref(),
                    &job.identity.job_id,
                    file_result.filename.as_ref(),
                    FileStage::Writing,
                    Some(result_idx as i64 + 1),
                    Some(total_results),
                )
                .await;
                let Some(artifacts) =
                    planning::artifact_set_for_source(plan, &file_result.filename)
                else {
                    lifecycle
                        .fail(
                            &format!(
                                "{missing_artifact_label} job plan missing artifacts for {}",
                                file_result.filename
                            ),
                            FailureCategory::Validation,
                        )
                        .await;
                    continue;
                };
                for artifact in &artifacts.files {
                    if artifact.role != MaterializedArtifactRole::Primary {
                        continue;
                    }
                    let target = crate::recipe_runner::runtime::ChatOutputTarget::new(
                        &job.filesystem,
                        file_index,
                        &artifact.display_path,
                    );
                    if let Err(error) = crate::recipe_runner::runtime::write_chat_output_artifact_with_provenance_gate(
                        &target,
                        &finished,
                    )
                    .await
                    {
                        lifecycle
                            .fail(&error.to_string(), FailureCategory::System)
                            .await;
                        continue;
                    }
                    use crate::pipeline::post_validate::OutputReport;
                    let diagnostics = match OutputReport::of(&finished, &shortfalls) {
                        OutputReport::Clean => None,
                        OutputReport::Diagnosed(draft) => {
                            let sidecar = job
                                .filesystem
                                .staging_dir
                                .as_path()
                                .join("diagnostics")
                                .join(format!("{}.findings.json", artifact.display_path));
                            Some(draft.record(&sidecar).await)
                        }
                    };
                    lifecycle
                        .complete_reported(
                            crate::store::CompletedFileOutput {
                                filename: artifact.display_path.clone(),
                                content_type: artifact.content_type,
                                stamp: stamp.clone(),
                                exclusions: exclusions.clone(),
                            },
                            diagnostics,
                        )
                        .await;
                }
            }
            Err(error) => {
                // The error states its own category. It used to be hardcoded
                // `ProviderTerminal`, which reported a file refused on
                // validity grounds as if a provider had given up on it.
                lifecycle.fail(&error.to_string(), error.category()).await;
            }
        }
    }
}

pub(crate) fn resolve_input_path(job: &RunnerJobSnapshot, file: &PendingJobFile) -> PathBuf {
    if job.filesystem.paths_mode && file.file_index < job.filesystem.source_paths.len() {
        job.filesystem.source_paths[file.file_index]
            .assume_shared_filesystem()
            .as_path()
            .to_owned()
    } else {
        job.filesystem
            .staging_dir
            .join("input")
            .join(file.filename.as_ref())
            .as_path()
            .to_owned()
    }
}

pub(crate) fn resolve_before_path(
    job: &RunnerJobSnapshot,
    file: &PendingJobFile,
) -> Option<PathBuf> {
    job.filesystem
        .before_path(file.file_index)
        .map(|path| path.as_path().to_owned())
}

#[cfg(test)]
mod reporting_tests {
    use super::*;
    use crate::api::{
        FileStatusKind, LanguageSpec, SecondaryMorphologyFailure, SecondaryMorphologyFailureKind,
        SecondaryMorphologySpan,
    };
    use crate::morphosyntax::report::{MorphosyntaxOutput, SecondaryReport};
    use crate::pipeline::post_validate::PostValidated;
    use crate::text_batch::TextBatchFileResult;
    use std::sync::Arc;

    /// Real planned writeback and store projection, not just serialization of
    /// a report. Fixtures represent producer outcomes; no model is simulated.
    #[tokio::test]
    async fn secondary_reports_survive_text_writeback_and_file_status() {
        const CHAT: &str = "@UTF8\n@Begin\n@Languages:\teng\n@Participants:\tPAR Participant\n@ID:\teng|test|PAR|||||Participant|||\n*PAR:\thello mundo@s:spa .\n%mor:\tintj|hello L2|xxx .\n%gra:\t1|0|ROOT 2|1|OBJ 3|1|PUNCT\n@End\n";
        for (unavailable, provenance) in [(true, false), (false, false), (false, true)] {
            let temp = tempfile::tempdir().unwrap();
            let mut job = crate::store::test_support::running_job_fixture();
            job.dispatch.lang = LanguageSpec::PerFile;
            job.filesystem.staging_dir =
                batchalign_types::paths::ServerPath::new(temp.path().to_str().unwrap());
            let snapshot = job.to_runner_snapshot();
            tokio::fs::create_dir_all(temp.path().join("input"))
                .await
                .unwrap();
            tokio::fs::write(temp.path().join("input/job-file.cha"), CHAT)
                .await
                .unwrap();
            let plan = planning::build_job_plan(&snapshot).unwrap();
            let (tx, _rx) = tokio::sync::broadcast::channel(crate::ws::BROADCAST_CAPACITY);
            let store = Arc::new(crate::store::JobStore::new(
                crate::config::ServerConfig::default(),
                None,
                tx,
                Arc::new(crate::clock::SystemClock),
            ));
            store.submit(job).await.unwrap();
            let host = DispatchHostContext::from_store(store.clone());
            let address = SecondaryMorphologySpan {
                utterance: 1,
                first_word: 2,
                words: 1,
                language: "spa".into(),
            };
            let report = if unavailable {
                SecondaryReport::for_test(
                    vec![SecondaryMorphologyFailure {
                        span: address,
                        cause: SecondaryMorphologyFailureKind::Worker,
                        detail: "model unavailable".into(),
                    }],
                    Vec::new(),
                )
            } else if provenance {
                SecondaryReport::default()
            } else {
                SecondaryReport::for_test(Vec::new(), vec![address])
            };
            let proof = PostValidated::for_test(CHAT, crate::api::ReleasedCommand::Morphotag);
            let expected = proof.as_str().to_owned();
            let mut output = MorphosyntaxOutput::new(proof, report);
            if provenance {
                output = output.with_provenance(
                    crate::provenance::MorphologyProvenanceReport::Incomplete(
                        crate::provenance::IncrementalMorphotagStampError::PriorIncomplete,
                    ),
                );
            }
            write_text_results(
                &snapshot,
                &host,
                &plan,
                vec![TextBatchFileResult::ok("job-file.cha", output)],
                false,
                "morphotag",
            )
            .await;
            assert_eq!(
                tokio::fs::read_to_string(temp.path().join("output/job-file.cha"))
                    .await
                    .unwrap(),
                expected
            );
            let info = store.get(&snapshot.identity.job_id).await.unwrap();
            let entry = &info.file_statuses[0];
            assert!(entry.error.is_none());
            assert_eq!(info.completed_files, 1);
            if unavailable || provenance {
                assert_eq!(entry.status, FileStatusKind::Diagnosed);
                assert!(
                    entry
                        .diagnostics
                        .as_ref()
                        .unwrap()
                        .lines()
                        .iter()
                        .any(|line| line.contains(if provenance {
                            "morphology provenance incomplete"
                        } else {
                            "secondary morphology incomplete"
                        }))
                );
                assert!(entry.exclusions.is_empty());
            } else {
                assert_eq!(entry.status, FileStatusKind::Done);
                assert!(entry.diagnostics.is_none());
                assert_eq!(entry.exclusions.len(), 1);
            }
        }
    }
}
