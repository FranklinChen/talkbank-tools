//! Real shell/writeback boundary with controlled FA/provider outcomes, no ML.
use super::*;
use crate::provenance::UtrRecovery;

const CHAT: &str = "@UTF8\n@Begin\n@Languages:\teng\n@Participants:\tPAR Participant\n@ID:\teng|test|PAR|||||Participant|||\n@Media:\taudio, audio\n*PAR:\thello . \u{15}100_400\u{15}\n%wor:\thello \u{15}100_400\u{15} .\n@End\n";

#[derive(Clone, Copy, PartialEq, Eq)]
enum Route {
    InitialOnly,
    FailedFallback,
    CancelledFallback,
}

struct ReportedTask {
    route: Route,
    attempts: usize,
    recovery: UtrRecovery,
}

fn provider_failure() -> ServerError {
    ServerError::Worker(WorkerError::WorkerResponse(
        "controlled provider unavailable".into(),
    ))
}

#[async_trait]
impl AudioFileTask for ReportedTask {
    type AttemptOutput = ();

    async fn run_attempt(
        &mut self,
        _: crate::runner::util::ProgressSender,
    ) -> Result<(), ServerError> {
        self.attempts += 1;
        if self.attempts == 1 && self.route != Route::InitialOnly {
            return Err(ServerError::Worker(WorkerError::ReadyTimeout {
                timeout_s: crate::api::PositiveSeconds::literal::<1>(),
            }));
        }
        Ok(())
    }

    async fn on_retryable_worker_failure(
        &mut self,
        _: &FileRunTracker<'_>,
        _: &ServerError,
    ) -> Result<(), crate::chat_ops::fa::utr::UtrCancelled> {
        let pending = self
            .recovery
            .fallback()
            .expect("only one fallback is permitted");
        let error = if self.route == Route::CancelledFallback {
            ServerError::Cancelled
        } else {
            provider_failure()
        };
        pending.record(Err(error))
    }

    async fn finalize_success(&mut self, _: ()) -> Result<FileOutput, ServerError> {
        let mut document =
            batchalign_transform::parse::parse_lenient(&crate::chat_parser(), CHAT).0;
        let stamp = crate::provenance::align_provenance(
            &LanguageCode3::eng(),
            &crate::engine_reports::FaCacheNamespace::for_test("test-fa"),
            self.recovery.contribution(),
            false,
            false,
            crate::chat_ops::fa::MainBulletPolicy::DeriveFromWords,
        );
        crate::provenance::inject_provenance(&mut document, &stamp);
        let text = batchalign_transform::serialize::to_chat_string(&document);
        Ok(FileOutput::Chat(ChatOutput {
            document: crate::pipeline::post_validate::PostValidated::for_test(
                &text,
                ReleasedCommand::Align,
            )
            .into(),
            shortfalls: self.recovery.contribution().shortfalls(),
            exclusions: Vec::new(),
            merge_abbreviations: MergeAbbreviations::Leave,
        }))
    }
}

#[tokio::test]
async fn utr_failures_survive_real_audio_writeback_and_cancellation_stops_retry() {
    for route in [
        Route::InitialOnly,
        Route::FailedFallback,
        Route::CancelledFallback,
    ] {
        let tmp = tempfile::tempdir().unwrap();
        let mut job = fake_job(&tmp);
        job.dispatch.command = ReleasedCommand::Align;
        job.dispatch.options = CommandOptions::Align(Default::default());
        job.pending_files[0].filename = DisplayPath::from("nested/test.cha");
        job.pending_files[0].has_chat = true;
        job.filesystem.source_paths[0] = ClientPath::new("/input/test.cha");
        let file = job.pending_files[0].clone();
        let (sink, state) = RecordingSink::new();
        let lifecycle =
            FileRunTracker::new(sink.as_ref(), &job.identity.job_id, file.filename.as_ref());
        lifecycle
            .begin_first_attempt(WorkUnitKind::FileForcedAlignment, FileStage::ResolvingAudio)
            .await;
        let recovery =
            UtrRecovery::from_initial(&crate::options::UtrEngine::RevAi, Err(provider_failure()))
                .unwrap();
        let mut task = ReportedTask {
            route,
            attempts: 0,
            recovery,
        };
        let outcome = run_audio_file_task(
            &job,
            sink.clone(),
            &file,
            &lifecycle,
            AudioTaskReporting {
                work_unit_kind: WorkUnitKind::FileForcedAlignment,
                running_stage: FileStage::Aligning,
                command_label: "Alignment",
            },
            &mut task,
        )
        .await;
        assert!(matches!(outcome, FileTaskOutcome::TerminalStateRecorded));
        let state = state.lock().unwrap();
        let path = tmp.path().join("requested/test.cha");
        if route == Route::CancelledFallback {
            assert_eq!(task.attempts, 1, "cancellation never retries FA");
            assert_eq!(state.retries, 0);
            assert!(state.diagnosed.is_empty());
            assert!(!path.exists());
            continue;
        }
        assert_eq!(state.errors, 0, "unavailable UTR is not a file failure");
        assert_eq!(state.diagnosed.len(), 1);
        assert!(
            state.diagnosed[0].first_findings().is_empty(),
            "no invented CHAT finding"
        );
        assert_eq!(
            state.diagnosed[0].shortfalls.len(),
            if route == Route::InitialOnly { 1 } else { 2 }
        );
        let written = std::fs::read_to_string(path).unwrap();
        assert!(written.contains("*PAR:\thello . \u{15}100_400\u{15}"));
        assert!(written.contains("%wor:\thello \u{15}100_400\u{15} ."));
        let provenance = crate::provenance::extract_provenance(&written).unwrap();
        let fields = &provenance[0].fields;
        assert!(
            !fields.contains_key("utr"),
            "failed inference cannot claim a producing engine"
        );
        assert_eq!(
            fields["utr_unavailable"],
            if route == Route::InitialOnly {
                "initial:rev:provider_transient"
            } else {
                "initial:rev:provider_transient+fallback:rev:provider_transient"
            }
        );
    }
}
