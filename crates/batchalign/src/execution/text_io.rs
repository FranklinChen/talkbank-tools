//! Text source admission, consuming preparation and publication.

#[cfg(test)]
mod boundary_tests;
mod preparation;
pub use preparation::{
    CorefSubmission, CoreferenceText, MorphotagSource, MorphotagSubmission, PreparedTextFile,
    PreparedTextJob, SegmentationText, TranslateSubmission, TranslationText, UtsegSubmission,
};

#[cfg(test)]
struct OnceReply<T>(std::sync::Mutex<Option<T>>);

#[cfg(test)]
impl<T> OnceReply<T> {
    fn new(reply: T) -> Self {
        Self(std::sync::Mutex::new(Some(reply)))
    }

    fn take(&self) -> T {
        self.0.lock().unwrap().take().expect("one fixture reply")
    }
}

#[cfg(test)]
#[async_trait::async_trait]
impl super::worker_gateway::MorphotagGateway
    for OnceReply<crate::morphosyntax::report::MorphosyntaxOutput>
{
    async fn morphotag_single(
        &self,
        _submission: MorphotagSubmission<'_>,
    ) -> Result<crate::morphosyntax::report::MorphosyntaxOutput, crate::error::ServerError> {
        Ok(self.take())
    }
}

#[cfg(test)]
mod reporting_tests {
    use super::*;
    use crate::api::{FileStatusKind, LanguageSpec, SecondaryMorphologyFailureKind};
    use crate::morphosyntax::report::{MorphosyntaxOutput, SecondaryReport};
    use crate::pipeline::post_validate::PostValidated;
    use crate::runner::DispatchHostContext;
    use std::sync::Arc;

    enum SecondaryCase {
        Unavailable,
        Excluded,
        ProvenanceIncomplete,
    }

    /// Real planned writeback and store projection, not just serialization of
    /// a report. Fixtures represent producer outcomes; no model is simulated.
    /// Guards: Secondary unavailability, exclusions and provenance shortfalls survive writeback with the correct terminal file projection.
    /// Kind: contract
    /// Not a type: Closed producer cases exclude invalid combinations but cannot prove written CHAT bytes or real store diagnostics.
    /// State: Temporary CHAT output and an admitted registry run exercise actual writeback and store projection. SQLite and models contribute no effect.
    /// Concurrency: Each case owns its directory, registry actor and broadcast channel. Its consumed claim completes after all writes finish.
    #[tokio::test]
    async fn secondary_reports_survive_text_writeback_and_file_status() {
        const CHAT: &str = "@UTF8\n@Begin\n@Languages:\teng\n@Participants:\tPAR Participant\n@ID:\teng|test|PAR|||||Participant|||\n*PAR:\thello mundo@s:spa .\n%mor:\tintj|hello L2|xxx .\n%gra:\t1|0|ROOT 2|1|OBJ 3|1|PUNCT\n@End\n";
        for case in [
            SecondaryCase::Unavailable,
            SecondaryCase::Excluded,
            SecondaryCase::ProvenanceIncomplete,
        ] {
            let temp = tempfile::tempdir().unwrap();
            let mut job = crate::store::test_support::queued_morphotag_job_fixture(
                crate::store::ContentInput {
                    display: "job-file.cha".into(),
                    kind: crate::store::InputKind::Chat,
                },
            );
            job.dispatch.lang = LanguageSpec::PerFile;
            job.filesystem.staging_dir =
                batchalign_types::paths::ServerPath::new(temp.path().to_str().unwrap());
            let job_id = job.identity.job_id.clone();
            tokio::fs::create_dir_all(temp.path().join("input"))
                .await
                .unwrap();
            tokio::fs::write(temp.path().join("input/job-file.cha"), CHAT)
                .await
                .unwrap();
            let (tx, _rx) = tokio::sync::broadcast::channel(crate::ws::BROADCAST_CAPACITY);
            let store = Arc::new(crate::store::JobStore::for_tests(
                crate::config::ServerConfig::default(),
                None,
                tx,
                Arc::new(crate::clock::SystemClock),
            ));
            store.submit(job).await.unwrap();
            let claim = crate::store::test_support::claim_runner(&store, &job_id).await;
            let snapshot = claim.snapshot();
            let host = DispatchHostContext::for_run(&claim);
            let address = crate::morphosyntax::report::SpanAddress::for_test(0, 2, 1, "spa");
            let report = match case {
                SecondaryCase::Unavailable => SecondaryReport::for_test(
                    vec![(
                        address,
                        SecondaryMorphologyFailureKind::Worker,
                        "model unavailable",
                    )],
                    Vec::new(),
                ),
                SecondaryCase::Excluded => SecondaryReport::for_test(Vec::new(), vec![address]),
                SecondaryCase::ProvenanceIncomplete => SecondaryReport::default(),
            };
            let proof = PostValidated::for_test(CHAT, crate::api::ReleasedCommand::Morphotag);
            let expected = proof.as_str().to_owned();
            let mut output = MorphosyntaxOutput::new(
                proof,
                crate::morphosyntax::report::AnalysisReports {
                    primary: crate::morphosyntax::report::PrimaryReport::none(),
                    secondary: report,
                },
            );
            if matches!(case, SecondaryCase::ProvenanceIncomplete) {
                output = output.with_provenance(
                    crate::provenance::MorphologyProvenanceReport::Incomplete(
                        crate::provenance::IncrementalMorphotagStampError::PriorIncomplete,
                    ),
                );
            }
            PreparedTextJob::<crate::execution::MorphotagRuntimeOptions>::for_test(
                (**snapshot).clone(),
                host,
                crate::api::NumWorkers(1),
            )
            .next_file_for_test()
            .await
            .expect("read prepared source")
            .submit(&OnceReply::new(output), None)
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
            match case {
                SecondaryCase::Unavailable | SecondaryCase::ProvenanceIncomplete => {
                    assert_eq!(entry.status, FileStatusKind::Diagnosed);
                    assert!(
                        entry
                            .diagnostics
                            .as_ref()
                            .unwrap()
                            .lines()
                            .iter()
                            .any(|line| line.contains(
                                if matches!(case, SecondaryCase::ProvenanceIncomplete) {
                                    "morphology provenance incomplete"
                                } else {
                                    "secondary morphology incomplete"
                                }
                            ))
                    );
                    assert!(entry.exclusions.is_empty());
                }
                SecondaryCase::Excluded => {
                    assert_eq!(entry.status, FileStatusKind::Done);
                    assert!(entry.diagnostics.is_none());
                    assert_eq!(entry.exclusions.len(), 1);
                }
            }
            claim.complete().await.expect("complete admitted run");
        }
    }
}
