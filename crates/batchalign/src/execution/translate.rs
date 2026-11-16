//! Per-file translation retains source-language routing and immediate durability.

use super::text_io::{PreparedTextJob, TranslationText};
use super::worker_gateway::TranslateGateway;
use crate::runner::util::{FileRunTracker, FileStage};
use crate::scheduling::WorkUnitKind;

pub(crate) async fn dispatch_translate_job(
    mut files: PreparedTextJob<TranslationText>,
    gateway: &dyn TranslateGateway,
) -> Result<(), crate::error::ServerError> {
    let options = std::sync::Arc::clone(files.translation_options());
    while let Some(file) = files.next_file().await {
        FileRunTracker::new(file.host().sink().as_ref(), file.input().filename.as_ref())
            .begin_first_attempt(WorkUnitKind::BatchInfer, FileStage::Translating)
            .await;
        let parser = crate::chat_parser();
        let (chat, _) =
            batchalign_transform::parse::parse_lenient(&parser, &file.input().chat_text);
        let results = match crate::pipeline::morphosyntax::resolve_per_file_lang(&chat) {
            Ok(language) => {
                let route = crate::translate::TranslationRoute::for_file(language, &options);
                crate::runner::util::observing_file_waits(
                    file.host().sink(),
                    [file.input().filename.to_string()],
                    gateway.translate_file(file.input(), &route, file.cancellation()),
                )
                .await
            }
            Err(error) => vec![crate::text_batch::TextBatchFileResult::err(
                file.input().filename.clone(),
                crate::text_batch::TextWorkflowFileError::validation(error.to_string()),
            )],
        };
        file.complete(results).await;
    }
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
        CorrelationId, DisplayPath, JobId, LanguageCode3, LanguageSpec, NumSpeakers,
        ReleasedCommand,
    };
    use crate::options::{CommandOptions, TranslateOptions};
    use crate::runner::DispatchHostContext;
    use crate::store::RunnerJobSnapshot;
    use crate::text_batch::{TextBatchFileInput, TextBatchFileResult, TextBatchFileResults};

    #[derive(Default)]
    struct FakeTranslateGateway {
        requests: Mutex<Vec<ObservedTranslation>>,
    }

    struct ObservedTranslation {
        source: DisplayPath,
        route: crate::translate::TranslationRoute,
    }

    #[async_trait]
    impl TranslateGateway for FakeTranslateGateway {
        async fn translate_file(
            &self,
            file: &TextBatchFileInput,
            route: &crate::translate::TranslationRoute,
            _cancellation: crate::infer_retry::Cancellation<'_>,
        ) -> TextBatchFileResults {
            self.requests.lock().unwrap().push(ObservedTranslation {
                source: file.filename.clone(),
                route: route.clone(),
            });
            let translated = file.chat_text.replace("@End", "%xtra:\ttranslated\n@End");
            vec![TextBatchFileResult::ok(
                file.filename.clone(),
                crate::pipeline::post_validate::PostValidated::for_test(
                    translated,
                    crate::api::ReleasedCommand::Translate,
                ),
            )]
        }
    }

    fn translate_snapshot(
        staging_dir: &std::path::Path,
        options: TranslateOptions,
    ) -> RunnerJobSnapshot {
        // Translate dispatch resolves the source language *per file* from
        // each file's `@Languages:` header (BA2 parity, see module doc).
        // A missing header is a typed failure, not a silent fallback
        // so the test fixture must declare a language explicitly.
        // CHAT requires `@Languages` and `@Participants` to appear after
        // `@Begin` for the parser to extract them into the typed model.
        let text = "@UTF8\n@Begin\n@Languages:\teng\n@Participants:\tPAR Subject\n@ID:\teng|test|PAR|||||Subject|||\n*PAR:\thello world .\n@End\n";
        let input_dir = staging_dir.join("input");
        std::fs::create_dir_all(&input_dir).unwrap();
        std::fs::write(input_dir.join("a.cha"), text).unwrap();
        std::fs::write(input_dir.join("b.cha"), text).unwrap();
        crate::store::test_support::with_pending(RunnerJobSnapshot {
            identity: crate::store::RunnerJobIdentity {
                job_id: JobId::from("job-translate"),
                correlation_id: CorrelationId::from("corr-translate"),
            },
            dispatch: crate::store::RunnerDispatchConfig {
                command: ReleasedCommand::Translate,
                // Translate is a per-file-language command (BA2 parity, see
                // module doc): submission validation rejects any other
                // `LanguageSpec` for this command.
                lang: LanguageSpec::PerFile,
                num_speakers: NumSpeakers(1),
                options: CommandOptions::Translate(options),
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

    /// Guards: Each CHAT source dispatches separately and its translated text reaches the written xtra tier.
    /// Kind: contract
    /// Not a type: TranslateGateway accepts one source, while actual scheduling of each source and serialized xtra bytes require observing requests and written output.
    /// State: Temporary CHAT sources and output exercise actual text IO. The recording gateway associates source names with routes and the shared sink records lifecycle effects.
    /// Concurrency: Each run owns its directory and gateway mutex. No external worker or shared store is started.
    #[tokio::test]
    async fn translate_dispatches_one_gateway_call_per_file() {
        // BA2 parity (2026-05-03): translate dispatches one gateway call per
        // input file, each tagged with that file's per-file source language
        // resolved from the file's `@Languages:` header. Cross-file pooling
        // was removed because it forced a single job-level lang across files
        //: the same shape that caused the morphotag incident. See
        // `dispatch_translate_job` doc for the full rationale.
        let temp = tempfile::tempdir().unwrap();
        let host = host();
        let gateway = FakeTranslateGateway::default();
        let job = translate_snapshot(temp.path(), TranslateOptions::default());

        dispatch_translate_job(
            PreparedTextJob::for_test(job, host, crate::api::NumWorkers(1)),
            &gateway,
        )
        .await
        .expect("translate dispatch");

        let requests = gateway.requests.lock().unwrap();
        assert_eq!(
            requests
                .iter()
                .map(|request| request.source.as_ref())
                .collect::<Vec<_>>(),
            ["a.cha", "b.cha"],
        );
        let output = std::fs::read_to_string(temp.path().join("output/a.cha")).unwrap();
        assert!(output.contains("%xtra:\ttranslated"));
    }

    /// Guards: Per-file headers retain distinct source languages with the requested target and translation engine.
    /// Kind: contract
    /// Not a type: The sealed route retains its admitted language, target and engine. Reading distinct external CHAT headers and associating their routes with the scheduled sources remains a dispatcher effect.
    /// State: Two temporary CHAT files declare different languages. The recording gateway captures each source name with its actual route after the reader runs.
    /// Concurrency: The directory, gateway mutex and shared sink instance belong to this test. No external process is used.
    #[tokio::test]
    async fn translate_routes_each_source_to_the_selected_target_and_engine() {
        let temp = tempfile::tempdir().unwrap();
        let host = host();
        let gateway = FakeTranslateGateway::default();
        let mut options = TranslateOptions {
            target: LanguageCode3::fra(),
            ..TranslateOptions::default()
        };
        options.common.engine_overrides.translate =
            Some(crate::types::engines::TranslateEngineName::Nllb);
        let job = translate_snapshot(temp.path(), options);
        let path = temp.path().join("input/b.cha");
        let spanish = std::fs::read_to_string(&path)
            .unwrap()
            .replace("eng", "spa");
        std::fs::write(path, spanish).unwrap();
        dispatch_translate_job(
            PreparedTextJob::for_test(job, host, crate::api::NumWorkers(1)),
            &gateway,
        )
        .await
        .unwrap();
        let requests = gateway.requests.lock().unwrap();
        assert_eq!(
            requests
                .iter()
                .map(|request| (request.source.as_ref(), request.route.source().as_ref()))
                .collect::<Vec<_>>(),
            [("a.cha", "eng"), ("b.cha", "spa")]
        );
        assert!(
            requests
                .iter()
                .all(|request| request.route.target() == &LanguageCode3::fra()
                    && request.route.engine() == &crate::types::engines::TranslateEngineName::Nllb)
        );
    }
}
