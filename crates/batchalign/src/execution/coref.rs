//! Coreference retains its cross-file batch and source-owned publication.

use super::text_io::{CoreferenceText, PreparedTextJob};
use super::worker_gateway::CorefGateway;
use crate::runner::util::{FileRunTracker, FileStage};
use crate::scheduling::WorkUnitKind;

pub(crate) async fn dispatch_coref_job(
    files: PreparedTextJob<CoreferenceText>,
    gateway: &dyn CorefGateway,
) -> Result<(), crate::error::ServerError> {
    let batch = files.read_batch().await;
    if batch.inputs().is_empty() {
        return Ok(());
    }
    for file in batch.inputs() {
        FileRunTracker::new(batch.host().sink().as_ref(), file.filename.as_ref())
            .begin_first_attempt(WorkUnitKind::BatchInfer, FileStage::ResolvingCoreference)
            .await;
    }
    let results = crate::runner::util::observing_file_waits(
        batch.host().sink(),
        batch.inputs().iter().map(|file| file.filename.to_string()),
        gateway.coref_batch(batch.inputs(), batch.cancellation()),
    )
    .await;
    batch.complete(results).await;
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;
    use std::sync::Mutex;

    use async_trait::async_trait;
    use tokio_util::sync::CancellationToken;

    use super::*;
    use crate::api::{
        CorrelationId, DisplayPath, JobId, LanguageSpec, NumSpeakers, ReleasedCommand,
    };
    use crate::options::{CommandOptions, CommonOptions, CorefOptions};
    use crate::runner::DispatchHostContext;
    use crate::store::RunnerJobSnapshot;
    use crate::text_batch::{TextBatchFileInput, TextBatchFileResult, TextBatchFileResults};

    #[derive(Default)]
    struct FakeCorefGateway {
        batches: Mutex<Vec<ObservedCoreference>>,
    }

    #[derive(Debug, PartialEq)]
    struct ObservedCoreference {
        sources: Vec<DisplayPath>,
    }

    #[async_trait]
    impl CorefGateway for FakeCorefGateway {
        async fn coref_batch(
            &self,
            files: &[TextBatchFileInput],
            _cancellation: crate::infer_retry::Cancellation<'_>,
        ) -> TextBatchFileResults {
            self.batches.lock().unwrap().push(ObservedCoreference {
                sources: files.iter().map(|file| file.filename.clone()).collect(),
            });
            files
                .iter()
                .map(|file| {
                    let content = file.chat_text.replace("@End", "%xcoref:\t(1)\n@End");
                    TextBatchFileResult::ok(
                        file.filename.clone(),
                        crate::pipeline::post_validate::PostValidated::for_test(
                            content,
                            crate::api::ReleasedCommand::Coref,
                        ),
                    )
                })
                .collect()
        }
    }

    fn coref_snapshot(staging_dir: &std::path::Path) -> RunnerJobSnapshot {
        let text = "@UTF8\n@Begin\n*CHI:\the saw him .\n@End\n";
        let input_dir = staging_dir.join("input");
        std::fs::create_dir_all(&input_dir).unwrap();
        std::fs::write(input_dir.join("a.cha"), text).unwrap();
        std::fs::write(input_dir.join("b.cha"), text).unwrap();
        crate::store::test_support::with_pending(RunnerJobSnapshot {
            identity: crate::store::RunnerJobIdentity {
                job_id: JobId::from("job-coref"),
                correlation_id: CorrelationId::from("corr-coref"),
            },
            dispatch: crate::store::RunnerDispatchConfig {
                command: ReleasedCommand::Coref,
                // The shape a real coref submission has: coref takes no
                // `--lang`, so submission validation requires `PerFile`. The
                // fixture used to carry `Resolved(eng)`, which is a shape the
                // wire boundary rejects, and that is why the dispatch defect
                // this test now covers was invisible here.
                lang: LanguageSpec::PerFile,
                num_speakers: NumSpeakers(1),
                options: CommandOptions::Coref(CorefOptions {
                    common: CommonOptions::default(),
                }),
                runtime_state: BTreeMap::new(),
                debug_traces: false,
            },
            filesystem: crate::store::RunnerFilesystemConfig {
                inputs: crate::store::JobInputs::Content(
                    ["a.cha", "b.cha"]
                        .into_iter()
                        .map(|name| crate::store::ContentInput {
                            display: name.into(),
                            kind: crate::store::InputKind::Chat,
                        })
                        .collect(),
                ),
                staging_dir: batchalign_types::paths::ServerPath::new(
                    staging_dir.display().to_string(),
                ),
                media_mapping: Default::default(),
                media_subdir: Default::default(),
                source_dir: batchalign_types::paths::ClientPath::new(
                    staging_dir.display().to_string(),
                ),
            },
            cancel_token: CancellationToken::new(),
            pending_files: Vec::new(),
        })
    }

    fn host() -> DispatchHostContext {
        DispatchHostContext::for_test_effects()
    }

    /// Guards: Two declared CHAT files share one coreference request and retain its xcoref output.
    /// Kind: contract
    /// Not a type: CorefGateway restricts the available effect, while the actual ordered source population and serialized xcoref bytes require observing the dispatcher and filesystem.
    /// State: Isolated temporary CHAT inputs exercise the real reader and writer. The recording gateway captures one source-bound batch and the shared sink records lifecycle effects.
    /// Concurrency: The fake gateway mutex records this test alone. Each run owns its directory and recording sink.
    #[tokio::test]
    async fn coref_batches_and_persists_xcoref_output() {
        let temp = tempfile::tempdir().unwrap();
        let host = host();
        let gateway = FakeCorefGateway::default();
        let job = coref_snapshot(temp.path());

        dispatch_coref_job(
            PreparedTextJob::for_test(job, host, crate::api::NumWorkers(1)),
            &gateway,
        )
        .await
        .expect("coref dispatch");

        {
            let batches = gateway.batches.lock().unwrap();
            assert_eq!(
                *batches,
                [ObservedCoreference {
                    sources: vec!["a.cha".into(), "b.cha".into()],
                }],
            );
        }

        let output = std::fs::read_to_string(temp.path().join("output").join("a.cha")).unwrap();
        assert!(output.contains("%xcoref:\t(1)"));
    }
}
