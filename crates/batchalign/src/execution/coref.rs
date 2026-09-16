use crate::infer_retry::Cancellation;
use crate::planning;
use crate::runner::DispatchHostContext;
use crate::runner::util::{FileRunTracker, FileStage};
use crate::scheduling::WorkUnitKind;
use crate::store::{RunnerJobSnapshot, unix_now};

use super::text_io::{load_text_inputs, write_text_results};
use super::worker_gateway::WorkerGateway;

/// Dispatch a coref job: one cross-file batch, no job-level language.
///
/// **Coref takes no language, and this dispatch cannot ask for one.** Coref is
/// a per-file command ([`crate::dispatch_language`]): it has no `--lang` on the
/// CLI, so submission validation requires it to arrive as
/// [`LanguageSpec::PerFile`](crate::api::LanguageSpec::PerFile), and the
/// command itself is English-only, reading per-file English-ness from each
/// file's `@Languages:` header and holding the inference language at the
/// constant `eng`.
///
/// Until this was fixed, coref went through a shared "simple batched text"
/// dispatch that began by demanding `job.dispatch.lang.as_resolved()`. On a
/// per-file job that is always `None`, so EVERY coref job was refused before
/// any work was dispatched, with a message telling the operator to pass a
/// `--lang` flag coref does not have. The language it demanded was then
/// discarded by the batch itself, which hardcodes `eng`. The job could not
/// succeed and the message named the wrong thing.
///
/// The shared path had exactly one caller (this one), so it was deleted rather
/// than made conditional: a runtime check for a condition one caller can never
/// satisfy is a defect, and a second caller would have inherited it.
///
/// Cross-file batching is preserved deliberately. Coref resolves chains over a
/// whole document and has no per-file language to route on, so unlike translate
/// and utseg there is nothing to gain by splitting the batch per file.
pub(crate) async fn dispatch_coref_job(
    job: &RunnerJobSnapshot,
    host: &DispatchHostContext,
    gateway: &dyn WorkerGateway,
    should_merge_abbrev: bool,
) -> Result<(), crate::error::ServerError> {
    let plan = planning::build_job_plan(job).map_err(|error| {
        crate::error::ServerError::Validation(format!("Coref planning failed: {error}"))
    })?;
    let sink = host.sink().clone();
    let started_at = unix_now();

    for file in &job.pending_files {
        FileRunTracker::new(sink.as_ref(), &job.identity.job_id, file.filename.as_ref())
            .begin_first_attempt(
                WorkUnitKind::BatchInfer,
                started_at,
                FileStage::ResolvingCoreference,
            )
            .await;
    }

    let inputs = load_text_inputs(job, host, false).await;
    if inputs.file_texts.is_empty() {
        return Ok(());
    }

    let results = gateway
        .coref_batch(&inputs.file_texts, Cancellation::Token(&job.cancel_token))
        .await;

    write_text_results(job, host, &plan, results, should_merge_abbrev, "Coref").await;
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;
    use std::sync::{Arc, Mutex};

    use crate::chat_ops::morphosyntax_ops::MwtDict;
    use async_trait::async_trait;
    use tokio_util::sync::CancellationToken;

    use super::*;
    use crate::api::{
        CorrelationId, DisplayPath, JobId, LanguageCode3, LanguageSpec, NumSpeakers,
        ReleasedCommand,
    };
    use crate::execution::worker_gateway::MorphotagRuntimeOptions;
    use crate::options::{CommandOptions, CommonOptions, CorefOptions};
    use crate::store::PendingJobFile;
    use crate::text_batch::{TextBatchFileInput, TextBatchFileResult, TextBatchFileResults};

    #[derive(Default)]
    struct FakeCorefGateway {
        state: Mutex<FakeCorefState>,
    }

    #[derive(Default)]
    struct FakeCorefState {
        batch_calls: usize,
        batch_sizes: Vec<usize>,
    }

    #[async_trait]
    impl WorkerGateway for FakeCorefGateway {
        async fn morphotag_for_compare(
            &self,
            _chat_text: &str,
            _lang: &LanguageCode3,
            _mwt: &MwtDict,
            _cancellation: crate::infer_retry::Cancellation<'_>,
        ) -> Result<crate::pipeline::post_validate::PostValidated, crate::error::ServerError>
        {
            unreachable!()
        }

        async fn morphotag_single(
            &self,
            _chat_text: &str,
            _before_text: Option<&str>,
            _lang: &LanguageCode3,
            _options: MorphotagRuntimeOptions,
            _progress: Option<&crate::execution::morphotag::progress::BackendProgressPort>,
            _cancellation: crate::infer_retry::Cancellation<'_>,
        ) -> Result<crate::pipeline::post_validate::PostValidated, crate::error::ServerError>
        {
            unreachable!()
        }

        async fn utseg_batch(
            &self,
            _files: &[TextBatchFileInput],
            _lang: &LanguageCode3,
            _allow_stanza_fallback: bool,
            _cancellation: crate::infer_retry::Cancellation<'_>,
        ) -> TextBatchFileResults {
            unreachable!()
        }

        async fn translate_batch(
            &self,
            _files: &[TextBatchFileInput],
            _lang: &LanguageCode3,
            _cancellation: crate::infer_retry::Cancellation<'_>,
        ) -> TextBatchFileResults {
            unreachable!()
        }

        async fn coref_batch(
            &self,
            files: &[TextBatchFileInput],
            _cancellation: crate::infer_retry::Cancellation<'_>,
        ) -> TextBatchFileResults {
            let mut state = self.state.lock().unwrap();
            state.batch_calls += 1;
            state.batch_sizes.push(files.len());
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
        RunnerJobSnapshot {
            run_generation: crate::store::RunGeneration::FIRST,
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
                    merge_abbrev: false.into(),
                }),
                runtime_state: BTreeMap::new(),
                debug_traces: false,
            },
            filesystem: crate::store::RunnerFilesystemConfig {
                paths_mode: false,
                source_paths: Vec::new(),
                output_paths: Vec::new(),
                before_paths: Vec::new(),
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
            pending_files: vec![
                PendingJobFile {
                    file_index: 0,
                    filename: DisplayPath::from("a.cha"),
                    has_chat: true,
                },
                PendingJobFile {
                    file_index: 1,
                    filename: DisplayPath::from("b.cha"),
                    has_chat: true,
                },
            ],
        }
    }

    fn host() -> DispatchHostContext {
        let (tx, _rx) = tokio::sync::broadcast::channel(crate::ws::BROADCAST_CAPACITY);
        DispatchHostContext::from_store(Arc::new(crate::store::JobStore::new(
            crate::config::ServerConfig::default(),
            None,
            tx,
        )))
    }

    /// RED FIRST: a coref job submitted in its only legal shape (`PerFile`)
    /// reaches the gateway at all. Before the fix this dispatch refused the
    /// job outright, so the gateway was never called and no file was written.
    #[tokio::test]
    async fn coref_dispatches_a_per_file_job_instead_of_refusing_it() {
        let temp = tempfile::tempdir().unwrap();
        let host = host();
        let gateway = FakeCorefGateway::default();
        let job = coref_snapshot(temp.path());

        dispatch_coref_job(&job, &host, &gateway, false)
            .await
            .expect("a per-file coref job must dispatch");

        let state = gateway.state.lock().unwrap();
        assert_eq!(
            state.batch_calls, 1,
            "coref must reach the gateway; it used to be refused before dispatch"
        );
    }

    #[tokio::test]
    async fn coref_batches_all_files_in_one_gateway_call() {
        let temp = tempfile::tempdir().unwrap();
        let host = host();
        let gateway = FakeCorefGateway::default();
        let job = coref_snapshot(temp.path());

        dispatch_coref_job(&job, &host, &gateway, false)
            .await
            .expect("coref dispatch");

        let state = gateway.state.lock().unwrap();
        assert_eq!(state.batch_calls, 1);
        assert_eq!(state.batch_sizes, vec![2]);
    }

    #[tokio::test]
    async fn coref_write_path_persists_xcoref_output() {
        let temp = tempfile::tempdir().unwrap();
        let host = host();
        let gateway = FakeCorefGateway::default();
        let job = coref_snapshot(temp.path());

        dispatch_coref_job(&job, &host, &gateway, false)
            .await
            .expect("coref dispatch");

        let output = std::fs::read_to_string(temp.path().join("output").join("a.cha")).unwrap();
        assert!(output.contains("%xcoref:\t(1)"));
    }
}
