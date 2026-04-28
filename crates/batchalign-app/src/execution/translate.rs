use crate::runner::DispatchHostContext;
use crate::runner::util::FileStage;
use crate::store::RunnerJobSnapshot;

use super::simple_batched_text::dispatch_simple_batched_text_job;
use super::worker_gateway::WorkerGateway;

pub(crate) async fn dispatch_translate_job(
    job: &RunnerJobSnapshot,
    host: &DispatchHostContext,
    gateway: &dyn WorkerGateway,
    should_merge_abbrev: bool,
) -> Result<(), crate::error::ServerError> {
    dispatch_simple_batched_text_job(
        job,
        host,
        should_merge_abbrev,
        FileStage::Translating,
        "Translate",
        "Translate",
        |files, lang| async move { gateway.translate_batch(&files, &lang).await },
    )
    .await
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;
    use std::sync::{Arc, Mutex};
    use std::time::Duration;

    use async_trait::async_trait;
    use batchalign_chat_ops::morphosyntax::MwtDict;
    use tokio_util::sync::CancellationToken;

    use super::*;
    use crate::api::{
        CorrelationId, DisplayPath, JobId, LanguageCode3, LanguageSpec, NumSpeakers,
        ReleasedCommand, WorkerLanguage,
    };
    use crate::capability::WorkerCapabilitySnapshot;
    use crate::execution::worker_gateway::MorphotagRuntimeOptions;
    use crate::options::{CommandOptions, CommonOptions, TranslateOptions};
    use crate::store::PendingJobFile;
    use crate::text_batch::{TextBatchFileInput, TextBatchFileResult, TextBatchFileResults};

    #[derive(Default)]
    struct FakeTranslateGateway {
        state: Mutex<FakeTranslateState>,
    }

    #[derive(Default)]
    struct FakeTranslateState {
        batch_calls: usize,
        batch_sizes: Vec<usize>,
    }

    #[async_trait]
    impl WorkerGateway for FakeTranslateGateway {
        async fn ensure_command_capabilities(
            &self,
            _command: ReleasedCommand,
            _lang: WorkerLanguage,
            _engine_overrides: &str,
        ) -> Result<WorkerCapabilitySnapshot, String> {
            unreachable!()
        }

        async fn morphotag_for_compare(
            &self,
            _chat_text: &str,
            _lang: &LanguageCode3,
            _mwt: &MwtDict,
        ) -> Result<String, crate::error::ServerError> {
            unreachable!()
        }

        async fn morphotag_batch(
            &self,
            _files: &[TextBatchFileInput],
            _lang: &LanguageCode3,
            _options: MorphotagRuntimeOptions<'_>,
            _progress_tx: Option<
                tokio::sync::mpsc::Sender<crate::types::worker_v2::ProgressEventV2>,
            >,
            _group_timeout: Duration,
        ) -> TextBatchFileResults {
            unreachable!()
        }

        async fn morphotag_single(
            &self,
            _chat_text: &str,
            _lang: &LanguageCode3,
            _options: MorphotagRuntimeOptions<'_>,
        ) -> Result<String, crate::error::ServerError> {
            unreachable!()
        }

        async fn morphotag_incremental(
            &self,
            _before_text: &str,
            _after_text: &str,
            _lang: &LanguageCode3,
            _options: MorphotagRuntimeOptions<'_>,
        ) -> Result<String, crate::error::ServerError> {
            unreachable!()
        }

        async fn utseg_batch(
            &self,
            _files: &[TextBatchFileInput],
            _lang: &LanguageCode3,
        ) -> TextBatchFileResults {
            unreachable!()
        }

        async fn translate_batch(
            &self,
            files: &[TextBatchFileInput],
            _lang: &LanguageCode3,
        ) -> TextBatchFileResults {
            let mut state = self.state.lock().unwrap();
            state.batch_calls += 1;
            state.batch_sizes.push(files.len());
            files
                .iter()
                .map(|file| {
                    let translated = file.chat_text.replace("@End", "%xtra:\ttranslated\n@End");
                    TextBatchFileResult::ok(file.filename.clone(), translated)
                })
                .collect()
        }

        async fn coref_batch(
            &self,
            _files: &[TextBatchFileInput],
            _lang: &LanguageCode3,
        ) -> TextBatchFileResults {
            unreachable!()
        }
    }

    fn translate_snapshot(staging_dir: &std::path::Path) -> RunnerJobSnapshot {
        let text = "@UTF8\n@Begin\n*PAR:\thello world .\n@End\n";
        let input_dir = staging_dir.join("input");
        std::fs::create_dir_all(&input_dir).unwrap();
        std::fs::write(input_dir.join("a.cha"), text).unwrap();
        std::fs::write(input_dir.join("b.cha"), text).unwrap();
        RunnerJobSnapshot {
            identity: crate::store::RunnerJobIdentity {
                job_id: JobId::from("job-translate"),
                correlation_id: CorrelationId::from("corr-translate"),
            },
            dispatch: crate::store::RunnerDispatchConfig {
                command: ReleasedCommand::Translate,
                lang: LanguageSpec::Resolved(LanguageCode3::eng()),
                num_speakers: NumSpeakers(1),
                options: CommandOptions::Translate(TranslateOptions {
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

    #[tokio::test]
    async fn translate_batches_all_files_in_one_gateway_call() {
        let temp = tempfile::tempdir().unwrap();
        let host = host();
        let gateway = FakeTranslateGateway::default();
        let job = translate_snapshot(temp.path());

        dispatch_translate_job(&job, &host, &gateway, false)
            .await
            .expect("translate dispatch");

        let state = gateway.state.lock().unwrap();
        assert_eq!(state.batch_calls, 1);
        assert_eq!(state.batch_sizes, vec![2]);
    }

    #[tokio::test]
    async fn translate_write_path_persists_xtra_output() {
        let temp = tempfile::tempdir().unwrap();
        let host = host();
        let gateway = FakeTranslateGateway::default();
        let job = translate_snapshot(temp.path());

        dispatch_translate_job(&job, &host, &gateway, false)
            .await
            .expect("translate dispatch");

        let output = std::fs::read_to_string(temp.path().join("output").join("a.cha")).unwrap();
        assert!(output.contains("%xtra:\ttranslated"));
    }
}
