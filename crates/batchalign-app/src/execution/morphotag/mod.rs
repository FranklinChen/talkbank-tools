use std::collections::HashMap;

use crate::api::LanguageCode3;
use crate::planning;
use crate::runner::DispatchHostContext;
use crate::runner::util::{FileRunTracker, FileStage, set_file_progress};
use crate::scheduling::WorkUnitKind;
use crate::store::{RunnerJobSnapshot, unix_now};
use crate::text_batch::{TextBatchFileInput, TextBatchFileResult, TextBatchFileResults};

use super::worker_gateway::{MorphotagRuntimeOptions, WorkerGateway};

mod input;
mod progress;
mod window_policy;
mod writeback;

use input::load_morphotag_inputs;
use progress::BatchProgressReporter;
use window_policy::{MorphotagExecutionMode, WindowPlan, select_execution_mode};
use writeback::write_morphotag_results;

pub(crate) async fn dispatch_morphotag_job(
    job: &RunnerJobSnapshot,
    host: &DispatchHostContext,
    gateway: &dyn WorkerGateway,
    options: MorphotagRuntimeOptions<'_>,
) -> Result<(), crate::error::ServerError> {
    let plan = planning::build_job_plan(job).map_err(|error| {
        crate::error::ServerError::Validation(format!("Morphotag planning failed: {error}"))
    })?;
    let sink = host.sink().clone();
    let started_at = unix_now();

    for file in &job.pending_files {
        FileRunTracker::new(sink.as_ref(), &job.identity.job_id, file.filename.as_ref())
            .begin_first_attempt(WorkUnitKind::BatchInfer, started_at, FileStage::Analyzing)
            .await;
    }

    let inputs = load_morphotag_inputs(job, host).await;
    if inputs.file_texts.is_empty() {
        return Ok(());
    }

    let lang = resolved_lang(job);
    let execution_mode = select_execution_mode(
        &inputs.file_texts,
        &inputs.before_texts,
        job.dispatch.options.common().batch_window,
        host.config().audio_task_timeout_s.max(1800),
    );
    let results = match execution_mode {
        MorphotagExecutionMode::Incremental => {
            run_incremental_morphotag(
                gateway,
                options,
                &inputs.file_texts,
                &inputs.before_texts,
                &lang,
            )
            .await
        }
        MorphotagExecutionMode::Windowed(plan) => {
            run_windowed_morphotag(
                job,
                host,
                gateway,
                options,
                &inputs.file_texts,
                &lang,
                &plan,
            )
            .await
        }
    };

    write_morphotag_results(job, host, &plan, results, options.should_merge_abbrev).await;
    Ok(())
}

fn resolved_lang(job: &RunnerJobSnapshot) -> LanguageCode3 {
    job.dispatch
        .lang
        .as_resolved()
        .cloned()
        .unwrap_or_else(LanguageCode3::eng)
}

async fn run_incremental_morphotag(
    gateway: &dyn WorkerGateway,
    options: MorphotagRuntimeOptions<'_>,
    file_texts: &[TextBatchFileInput],
    before_texts: &HashMap<String, String>,
    lang: &LanguageCode3,
) -> TextBatchFileResults {
    let mut results = Vec::with_capacity(file_texts.len());
    for file in file_texts {
        let result = if let Some(before_text) = before_texts.get(file.filename.as_ref()) {
            gateway
                .morphotag_incremental(before_text, file.chat_text.as_ref(), lang, options)
                .await
        } else {
            gateway
                .morphotag_single(file.chat_text.as_ref(), lang, options)
                .await
        };
        match result {
            Ok(text) => results.push(TextBatchFileResult::ok(file.filename.clone(), text)),
            Err(error) => results.push(TextBatchFileResult::err(
                file.filename.clone(),
                error.to_string(),
            )),
        }
    }
    results
}

async fn run_windowed_morphotag(
    job: &RunnerJobSnapshot,
    host: &DispatchHostContext,
    gateway: &dyn WorkerGateway,
    options: MorphotagRuntimeOptions<'_>,
    file_texts: &[TextBatchFileInput],
    lang: &LanguageCode3,
    plan: &WindowPlan,
) -> TextBatchFileResults {
    let sink = host.sink().clone();
    let mut results = Vec::with_capacity(file_texts.len());
    let reporter = BatchProgressReporter::spawn(job.identity.job_id.clone(), sink.clone());

    for (window_idx, window) in plan.windows.iter().enumerate() {
        for file in &file_texts[window.start..window.end] {
            set_file_progress(
                sink.as_ref(),
                &job.identity.job_id,
                file.filename.as_ref(),
                FileStage::Parsing,
                Some(window_idx as i64 + 1),
                Some(plan.windows.len() as i64),
            )
            .await;
        }

        let window_results = gateway
            .morphotag_batch(
                &file_texts[window.start..window.end],
                lang,
                options,
                Some(reporter.sender()),
                plan.group_timeout,
            )
            .await;
        results.extend(window_results);
    }

    reporter.finish().await;
    results
}

#[cfg(test)]
mod tests {
    use std::collections::{BTreeMap, HashMap};
    use std::sync::{Arc, Mutex};
    use std::time::Duration;

    use async_trait::async_trait;
    use batchalign_chat_ops::morphosyntax::{MultilingualPolicy, MwtDict, TokenizationMode};
    use tokio_util::sync::CancellationToken;

    use super::*;
    use crate::api::{
        CorrelationId, DisplayPath, JobId, LanguageSpec, NumSpeakers, ReleasedCommand,
    };
    use crate::options::{CommandOptions, CommonOptions, MorphotagOptions};
    use crate::store::PendingJobFile;

    #[derive(Default)]
    struct FakeMorphotagGateway {
        state: Mutex<FakeMorphotagState>,
    }

    #[derive(Default)]
    struct FakeMorphotagState {
        batch_calls: usize,
        batch_windows: Vec<usize>,
        single_calls: usize,
        incremental_calls: usize,
        tokenization_modes: Vec<TokenizationMode>,
        multilingual_policies: Vec<MultilingualPolicy>,
        l2_values: Vec<bool>,
    }

    #[async_trait]
    impl WorkerGateway for FakeMorphotagGateway {
        async fn ensure_command_capabilities(
            &self,
            _command: ReleasedCommand,
            _lang: crate::api::WorkerLanguage,
            _engine_overrides: &str,
        ) -> Result<crate::capability::WorkerCapabilitySnapshot, String> {
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
            files: &[TextBatchFileInput],
            _lang: &LanguageCode3,
            options: MorphotagRuntimeOptions<'_>,
            _progress_tx: Option<
                tokio::sync::mpsc::Sender<crate::types::worker_v2::ProgressEventV2>,
            >,
            _group_timeout: Duration,
        ) -> TextBatchFileResults {
            let mut state = self.state.lock().unwrap();
            state.batch_calls += 1;
            state.batch_windows.push(files.len());
            state.tokenization_modes.push(options.tokenization_mode);
            state
                .multilingual_policies
                .push(options.multilingual_policy);
            state.l2_values.push(options.l2_morphotag);
            files
                .iter()
                .map(|file| TextBatchFileResult::ok(file.filename.clone(), file.chat_text.clone()))
                .collect()
        }

        async fn morphotag_single(
            &self,
            chat_text: &str,
            _lang: &LanguageCode3,
            options: MorphotagRuntimeOptions<'_>,
        ) -> Result<String, crate::error::ServerError> {
            let mut state = self.state.lock().unwrap();
            state.single_calls += 1;
            state.tokenization_modes.push(options.tokenization_mode);
            state
                .multilingual_policies
                .push(options.multilingual_policy);
            state.l2_values.push(options.l2_morphotag);
            Ok(chat_text.to_string())
        }

        async fn morphotag_incremental(
            &self,
            _before_text: &str,
            after_text: &str,
            _lang: &LanguageCode3,
            options: MorphotagRuntimeOptions<'_>,
        ) -> Result<String, crate::error::ServerError> {
            let mut state = self.state.lock().unwrap();
            state.incremental_calls += 1;
            state.tokenization_modes.push(options.tokenization_mode);
            state
                .multilingual_policies
                .push(options.multilingual_policy);
            state.l2_values.push(options.l2_morphotag);
            Ok(after_text.to_string())
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
            _files: &[TextBatchFileInput],
            _lang: &LanguageCode3,
        ) -> TextBatchFileResults {
            unreachable!()
        }

        async fn coref_batch(
            &self,
            _files: &[TextBatchFileInput],
            _lang: &LanguageCode3,
        ) -> TextBatchFileResults {
            unreachable!()
        }
    }

    fn morphotag_snapshot(
        staging_dir: &std::path::Path,
        utterances_per_file: usize,
        with_before: bool,
        batch_window: usize,
        retokenize: bool,
        skipmultilang: bool,
        no_l2_morphotag: bool,
    ) -> RunnerJobSnapshot {
        let text = make_chat(utterances_per_file);
        let input_dir = staging_dir.join("input");
        std::fs::create_dir_all(&input_dir).unwrap();
        std::fs::write(input_dir.join("a.cha"), &text).unwrap();
        std::fs::write(input_dir.join("b.cha"), &text).unwrap();
        let before_paths = if with_before {
            let before_dir = staging_dir.join("before");
            std::fs::create_dir_all(&before_dir).unwrap();
            std::fs::write(before_dir.join("a.cha"), &text).unwrap();
            vec![batchalign_types::paths::ClientPath::from(
                before_dir.join("a.cha").display().to_string(),
            )]
        } else {
            Vec::new()
        };
        RunnerJobSnapshot {
            identity: crate::store::RunnerJobIdentity {
                job_id: JobId::from("job-morphotag"),
                correlation_id: CorrelationId::from("corr-morphotag"),
            },
            dispatch: crate::store::RunnerDispatchConfig {
                command: ReleasedCommand::Morphotag,
                lang: LanguageSpec::Resolved(LanguageCode3::eng()),
                num_speakers: NumSpeakers(1),
                options: CommandOptions::Morphotag(MorphotagOptions {
                    common: CommonOptions {
                        batch_window,
                        ..Default::default()
                    },
                    retokenize,
                    skipmultilang,
                    no_l2_morphotag,

                    ..Default::default()
                }),
                runtime_state: BTreeMap::new(),
                debug_traces: false,
            },
            filesystem: crate::store::RunnerFilesystemConfig {
                paths_mode: false,
                source_paths: Vec::new(),
                output_paths: Vec::new(),
                before_paths,
                staging_dir: batchalign_types::paths::ServerPath::new(staging_dir),
                media_mapping: Default::default(),
                media_subdir: Default::default(),
                source_dir: batchalign_types::paths::ClientPath::new("/source"),
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

    fn make_chat(utterances: usize) -> String {
        let mut lines = vec!["@UTF8".to_string(), "@Begin".to_string()];
        for i in 0..utterances {
            lines.push(format!("*PAR:\tword{i} ."));
        }
        lines.push("@End".to_string());
        lines.join("\n")
    }

    #[tokio::test]
    async fn morphotag_batches_multiple_files_in_one_gateway_call() {
        let tempdir = tempfile::tempdir().unwrap();
        let snapshot = morphotag_snapshot(tempdir.path(), 2, false, 25, false, false, false);
        let (tx, _rx) = tokio::sync::broadcast::channel(crate::ws::BROADCAST_CAPACITY);
        let host = DispatchHostContext::from_store(Arc::new(crate::store::JobStore::new(
            crate::config::ServerConfig::default(),
            None,
            tx,
        )));
        let gateway = FakeMorphotagGateway::default();
        let mwt = MwtDict::default();

        dispatch_morphotag_job(
            &snapshot,
            &host,
            &gateway,
            MorphotagRuntimeOptions {
                tokenization_mode: TokenizationMode::Preserve,
                multilingual_policy: MultilingualPolicy::ProcessAll,
                mwt: &mwt,
                l2_morphotag: true,
                respect_pos_hints: false,
                should_merge_abbrev: false,
            },
        )
        .await
        .unwrap();

        let state = gateway.state.lock().unwrap();
        assert_eq!(state.batch_calls, 1);
        assert_eq!(state.batch_windows, vec![2]);
        assert_eq!(state.incremental_calls, 0);
        assert_eq!(state.single_calls, 0);
        assert!(tempdir.path().join("output").join("a.cha").exists());
    }

    #[tokio::test]
    async fn morphotag_before_mode_disables_batching() {
        let tempdir = tempfile::tempdir().unwrap();
        let snapshot = morphotag_snapshot(tempdir.path(), 2, true, 25, false, false, false);
        let (tx, _rx) = tokio::sync::broadcast::channel(crate::ws::BROADCAST_CAPACITY);
        let host = DispatchHostContext::from_store(Arc::new(crate::store::JobStore::new(
            crate::config::ServerConfig::default(),
            None,
            tx,
        )));
        let gateway = FakeMorphotagGateway::default();
        let mwt = MwtDict::default();

        dispatch_morphotag_job(
            &snapshot,
            &host,
            &gateway,
            MorphotagRuntimeOptions {
                tokenization_mode: TokenizationMode::Preserve,
                multilingual_policy: MultilingualPolicy::ProcessAll,
                mwt: &mwt,
                l2_morphotag: true,
                respect_pos_hints: false,
                should_merge_abbrev: false,
            },
        )
        .await
        .unwrap();

        let state = gateway.state.lock().unwrap();
        assert_eq!(state.batch_calls, 0);
        assert_eq!(state.incremental_calls, 1);
        assert_eq!(state.single_calls, 1);
    }

    #[tokio::test]
    async fn morphotag_windowing_preserves_small_batch_policy() {
        let tempdir = tempfile::tempdir().unwrap();
        let snapshot = morphotag_snapshot(tempdir.path(), 100, false, 1, false, false, false);
        let (tx, _rx) = tokio::sync::broadcast::channel(crate::ws::BROADCAST_CAPACITY);
        let host = DispatchHostContext::from_store(Arc::new(crate::store::JobStore::new(
            crate::config::ServerConfig::default(),
            None,
            tx,
        )));
        let gateway = FakeMorphotagGateway::default();
        let mwt = MwtDict::default();

        dispatch_morphotag_job(
            &snapshot,
            &host,
            &gateway,
            MorphotagRuntimeOptions {
                tokenization_mode: TokenizationMode::Preserve,
                multilingual_policy: MultilingualPolicy::ProcessAll,
                mwt: &mwt,
                l2_morphotag: true,
                respect_pos_hints: false,
                should_merge_abbrev: false,
            },
        )
        .await
        .unwrap();

        let state = gateway.state.lock().unwrap();
        assert_eq!(state.batch_calls, 2);
        assert_eq!(state.batch_windows, vec![1, 1]);
    }

    #[tokio::test]
    async fn morphotag_option_wiring_reaches_gateway() {
        let tempdir = tempfile::tempdir().unwrap();
        let snapshot = morphotag_snapshot(tempdir.path(), 2, false, 25, true, true, true);
        let (tx, _rx) = tokio::sync::broadcast::channel(crate::ws::BROADCAST_CAPACITY);
        let host = DispatchHostContext::from_store(Arc::new(crate::store::JobStore::new(
            crate::config::ServerConfig::default(),
            None,
            tx,
        )));
        let gateway = FakeMorphotagGateway::default();
        let mwt = MwtDict::default();

        dispatch_morphotag_job(
            &snapshot,
            &host,
            &gateway,
            MorphotagRuntimeOptions {
                tokenization_mode: TokenizationMode::StanzaRetokenize,
                multilingual_policy: MultilingualPolicy::SkipNonPrimary,
                mwt: &mwt,
                l2_morphotag: false,
                respect_pos_hints: false,
                should_merge_abbrev: false,
            },
        )
        .await
        .unwrap();

        let state = gateway.state.lock().unwrap();
        assert_eq!(
            state.tokenization_modes,
            vec![TokenizationMode::StanzaRetokenize]
        );
        assert_eq!(
            state.multilingual_policies,
            vec![MultilingualPolicy::SkipNonPrimary]
        );
        assert_eq!(state.l2_values, vec![false]);
    }

    #[test]
    fn execution_mode_prefers_incremental_when_any_before_text_exists() {
        let file_texts = vec![
            TextBatchFileInput::new("a.cha", make_chat(2)),
            TextBatchFileInput::new("b.cha", make_chat(2)),
        ];
        let before_texts = HashMap::from([("a.cha".to_string(), make_chat(2))]);

        let mode = select_execution_mode(&file_texts, &before_texts, 25, 1800);

        assert_eq!(mode, MorphotagExecutionMode::Incremental);
    }

    #[test]
    fn input_resolution_uses_paths_mode_and_parallel_before_index() {
        let tempdir = tempfile::tempdir().unwrap();
        let mut snapshot = morphotag_snapshot(tempdir.path(), 2, true, 25, false, false, false);
        snapshot.filesystem.paths_mode = true;
        snapshot.filesystem.source_paths = vec![
            batchalign_types::paths::ClientPath::from("/tmp/source-a.cha"),
            batchalign_types::paths::ClientPath::from("/tmp/source-b.cha"),
        ];

        let input_path = super::input::resolve_input_path(&snapshot, &snapshot.pending_files[1]);
        let before_path =
            super::input::resolve_before_path(&snapshot, &snapshot.pending_files[0]).unwrap();

        assert_eq!(input_path, std::path::PathBuf::from("/tmp/source-b.cha"));
        assert!(
            before_path.ends_with("before/a.cha"),
            "before path should resolve by file index"
        );
    }
}
