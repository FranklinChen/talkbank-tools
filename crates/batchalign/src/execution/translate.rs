//! Per-file translation retains source-language routing and immediate durability.

use super::text_io::{PreparedTextJob, TranslationText};
use super::worker_gateway::TranslateGateway;

pub async fn dispatch_translate_job(
    mut files: PreparedTextJob<TranslationText>,
    gateway: &dyn TranslateGateway,
) -> Result<(), crate::error::ServerError> {
    while let Some(file) = files.next_file().await {
        file.submit(gateway).await;
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
    use crate::execution::text_io::TranslateSubmission;
    use crate::options::{CommandOptions, TranslateOptions};
    use crate::runner::DispatchHostContext;
    use crate::store::RunnerJobSnapshot;
    use crate::text_batch::{TextBatchFileResult, TextBatchFileResults};

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
            submission: TranslateSubmission<'_>,
        ) -> TextBatchFileResults {
            let (file, route, _) = submission.into_parts();
            let (filename, admitted) = file.into_parts();
            self.requests.lock().unwrap().push(ObservedTranslation {
                source: filename.clone(),
                route,
            });
            let translated = admitted
                .source()
                .replace("@End", "%xtra:\ttranslated\n@End");
            vec![TextBatchFileResult::ok(
                filename.clone(),
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
        // so the fixture includes both admitted and refused source bytes.
        // CHAT requires `@Languages` and `@Participants` to appear after
        // `@Begin` for the parser to extract them into the typed model.
        let text = "@UTF8\n@Begin\n@Languages:\teng\n@Participants:\tPAR Subject\n@ID:\teng|test|PAR|||||Subject|||\n*PAR:\thello world .\n@End\n";
        let input_dir = staging_dir.join("input");
        std::fs::create_dir_all(&input_dir).unwrap();
        std::fs::write(input_dir.join("a.cha"), text).unwrap();
        std::fs::write(input_dir.join("b.cha"), text).unwrap();
        std::fs::write(
            input_dir.join("missing.cha"),
            text.replace("@Languages:\teng\n", ""),
        )
        .unwrap();
        std::fs::write(
            input_dir.join("corrupt.cha"),
            text.replace("@End", "%mor:\tnoun| .\n@End"),
        )
        .unwrap();
        std::fs::write(
            input_dir.join("mismatch.cha"),
            text.replace("*PAR:", "@Media:\tsample, audio\n*PAR:")
                .replace("world .", "world . \u{15}0_1000\u{15}"),
        )
        .unwrap();
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
                    [
                        "a.cha",
                        "b.cha",
                        "missing.cha",
                        "corrupt.cha",
                        "mismatch.cha",
                    ]
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

    /// Guards: Each admitted CHAT dispatches separately and writes translated bytes. A missing route or corrupt source reports validation without dispatch or output.
    /// Kind: contract
    /// Not a type: Actual CHAT bytes can lack an eligible route, carry a malformed tier or name different media. Scheduling, located diagnostics and written xtra text observe what happened to each source.
    /// State: Temporary CHAT sources and output exercise actual text IO. The gateway records submissions and the sink records the source-bound refusal.
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
        let sink = std::sync::Arc::new(crate::runner::util::test_sink::RecordingSink::default());
        let host = DispatchHostContext::for_recorded_effects(std::sync::Arc::clone(&sink));
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
        let errors = sink.errors();
        assert_eq!(errors.len(), 3, "all source refusals are reported");
        for name in ["missing.cha", "corrupt.cha", "mismatch.cha"] {
            assert!(!temp.path().join("output").join(name).exists());
            assert!(
                errors.iter().any(|error| error.filename == name
                    && error.category == crate::scheduling::FailureCategory::Validation),
                "missing source refusal for {name}"
            );
        }
        let corrupted = errors
            .iter()
            .find(|error| error.filename == "corrupt.cha")
            .unwrap();
        assert!(
            corrupted.error.contains("line 7: E702"),
            "expected the located original finding, got {:?}",
            corrupted.error
        );
        let mismatch = errors
            .iter()
            .find(|error| error.filename == "mismatch.cha")
            .unwrap();
        assert!(mismatch.error.contains("E531"), "{}", mismatch.error);
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
            .replace("eng", "spa")
            .replace("@Languages:\tspa", "@Languages:\tspa, eng");
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
